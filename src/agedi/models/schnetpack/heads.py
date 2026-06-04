from typing import Callable, Dict

import schnetpack.nn as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from agedi.models.head import Head


def build_gated_equivariant_mlp(
    s_in: int,
    v_in: int,
    n_out: int,
    n_layers: int = 2,
    activation: Callable = F.silu,
    sactivation: Callable = F.silu,
) -> nn.Sequential:
    """
    Build neural network analog to MLP with `GatedEquivariantBlock`s instead of dense layers.

    Parameters
    ----------
    n_in: int
        Number of input nodes.
    n_out: int
        Number of output nodes.
    n_layers: int
        Number of layers.
    activation: Callable
        Activation function.
    sactivation: Callable
        Activation function for the skip connection.
    n_hidden: int
        Number of hidden nodes.

    Returns
    -------
    nn.Module

    """
    # get list of number of nodes in input, hidden & output layers
    s_neuron = s_in
    v_neuron = v_in
    s_neurons = []
    v_neurons = []
    for i in range(n_layers):
        s_neurons.append(s_neuron)
        v_neurons.append(v_neuron)
        s_neuron = max(n_out, s_neuron // 2)
        v_neuron = max(n_out, v_neuron // 2)
    s_neurons.append(n_out)
    v_neurons.append(n_out)

    n_gating_hidden = s_neurons[:-1]

    # assign a GatedEquivariantBlock (with activation function) to each hidden layer
    layers = [
        snn.GatedEquivariantBlock(
            n_sin=s_neurons[i],
            n_vin=v_neurons[i],
            n_sout=s_neurons[i + 1],
            n_vout=v_neurons[i + 1],
            n_hidden=n_gating_hidden[i],
            activation=activation,
            sactivation=sactivation,
        )
        for i in range(n_layers - 1)
    ]
    # assign a GatedEquivariantBlock (without scalar activation function)
    # to the output layer
    layers.append(
        snn.GatedEquivariantBlock(
            n_sin=s_neurons[-2],
            n_vin=v_neurons[-2],
            n_sout=s_neurons[-1],
            n_vout=v_neurons[-1],
            n_hidden=n_gating_hidden[-1],
            activation=activation,
            sactivation=None,
        )
    )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net


class PositionsScore(Head):
    """Predict the positions score of the atoms in the structure.

    Optionally applies EDM preconditioning (Karras et al., NeurIPS 2022) around
    the network output:

    .. math::

        D_\\theta(\\mathbf{x}, \\sigma) =
            c_{\\text{skip}}(\\sigma)\\,\\mathbf{x} +
            c_{\\text{out}}(\\sigma)\\,F_\\theta\\!\\left(
                c_{\\text{in}}(\\sigma)\\,\\mathbf{s},\\,
                c_{\\text{in}}(\\sigma)\\,\\mathbf{v}
            \\right)

    where :math:`\\mathbf{s}` and :math:`\\mathbf{v}` are the scalar and vector
    representations, :math:`\\mathbf{x}` are the (noisy) atom positions, and:

    .. math::

        c_{\\text{skip}} = \\frac{\\sigma_{\\text{data}}^2}{\\sigma^2 + \\sigma_{\\text{data}}^2},\\quad
        c_{\\text{out}}  = \\frac{\\sigma\\,\\sigma_{\\text{data}}}{\\sqrt{\\sigma^2 + \\sigma_{\\text{data}}^2}},\\quad
        c_{\\text{in}}   = \\frac{1}{\\sqrt{\\sigma^2 + \\sigma_{\\text{data}}^2}}

    :math:`\\sigma_{\\text{data}}` should be set to the empirical standard
    deviation of (zero-COM) atom positions in the training set.

    Preconditioning requires ``pos_sigma`` (= :math:`\\sqrt{\\text{var}(t)}`)
    to be present in the translated batch, which the
    :class:`~agedi.diffusion.noisers.pos.PositionsNoiser` stores automatically
    during the forward (noising) step.

    Parameters
    ----------
    input_dim_scalar : int
        Dimension of the scalar input features.
    input_dim_vector : int
        Dimension of the vector input features.
    gated_blocks : int
        Number of gated equivariant blocks.
    precondition : bool, optional
        Whether to apply EDM preconditioning.  Defaults to ``False``.
    sigma_data : float, optional
        Empirical std of (zero-COM) atom positions in the training set, in Å.
        Only used when ``precondition=True``.  Defaults to ``1.0``.
    """

    _key = "pos"

    def __init__(
        self,
        input_dim_scalar: int = 66,
        input_dim_vector: int = 64,
        gated_blocks: int = 3,
        precondition: bool = False,
        sigma_data: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_dim_scalar = input_dim_scalar
        self.input_dim_vector = input_dim_vector
        self.gated_blocks = gated_blocks
        self.precondition = precondition
        self.sigma_data = sigma_data
        self.net = build_gated_equivariant_mlp(
            input_dim_scalar,
            input_dim_vector,
            1,
            n_layers=gated_blocks,
        )

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this positions score head."""
        return {
            **super().get_hparams(),
            "input_dim_scalar": self.input_dim_scalar,
            "input_dim_vector": self.input_dim_vector,
            "gated_blocks": self.gated_blocks,
            "precondition": self.precondition,
            "sigma_data": self.sigma_data,
        }

    def _score(self, batch: dict) -> torch.Tensor:
        """Predict the positions score of the atoms in the structure.

        Parameters
        ----------
        batch : dict
            The translated input batch.  Must contain
            ``scalar_representation`` and ``vector_representation``.  When
            ``self.precondition`` is ``True`` it must also contain
            ``pos_sigma`` (shape ``(n_atoms, 1)``) and ``pos`` (noisy atom
            positions, shape ``(n_atoms, 3)``).

        Returns
        -------
        torch.Tensor
            Predicted positions score, shape ``(n_atoms, 3)``.
        """
        scalar = batch["scalar_representation"]
        vector = batch["vector_representation"]

        if self.precondition:
            sigma = batch["pos_sigma"]           # (n_atoms, 1)
            pos   = batch["pos"]                 # (n_atoms, 3)
            sd2   = self.sigma_data ** 2
            denom = (sigma ** 2 + sd2).sqrt()    # (n_atoms, 1)

            c_in   = 1.0 / denom                 # (n_atoms, 1)
            c_skip = sd2 / (sigma ** 2 + sd2)    # (n_atoms, 1)
            c_out  = sigma * self.sigma_data / denom  # (n_atoms, 1)

            # Scale representations before the head MLP.
            scalar = scalar * c_in
            vector = vector * c_in.unsqueeze(-1)  # (n_atoms, n_feat, 3)

            _, raw_vec = self.net([scalar, vector])
            raw_vec = raw_vec.squeeze(-1)         # (n_atoms, 3)

            return c_out * raw_vec + c_skip * pos

        _, vector = self.net([scalar, vector])
        return vector.squeeze(-1)


class TypesScore(Head):
    """Predict the types score of the atoms in the structure.

    Parameters
    ----------
    input_dim_scalar: int
        The dimension of the scalar input.
    input_dim_vector: int
        The dimension of the vector input.

    Returns
    -------
    Head

    """

    _key = "x"

    def __init__(self, input_dim_scalar: int = 66, input_dim_vector: int = 64, n_classes: int = 100, **kwargs) -> None:
        """Initialize the types score head.

        Parameters
        ----------
        input_dim_scalar : int, optional
            Dimension of the scalar input features.
        input_dim_vector : int, optional
            Dimension of the vector input features (unused, kept for API
            consistency).
        n_classes : int, optional
            Number of atom-type classes (output logits).  Must match the
            ``n_classes`` of the corresponding
            :class:`~agedi.diffusion.noisers.Types` noiser.  Defaults to 100.
        **kwargs
            Additional keyword arguments forwarded to :class:`~agedi.models.head.Head`.
        """
        super().__init__(**kwargs)
        self.input_dim_scalar = input_dim_scalar
        self.input_dim_vector = input_dim_vector
        self.n_classes = n_classes
        self.net = nn.Linear(input_dim_scalar, n_classes)
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this types score head."""
        return {
            **super().get_hparams(),
            "input_dim_scalar": self.input_dim_scalar,
            "input_dim_vector": self.input_dim_vector,
            "n_classes": self.n_classes,
        }

    def _score(self, batch: dict) -> torch.Tensor:
        """Predict the types score of the atoms in the structure.

        Parameters
        ----------
        batch : dict
            The translated input batch with a ``scalar_representation`` key.

        Returns
        -------
        torch.Tensor
            The predicted types score.

        """
        scalar_representation = batch["scalar_representation"]

        pred = self.net(scalar_representation)
        return pred

