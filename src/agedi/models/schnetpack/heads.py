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

    Parameters
    ----------
    input_dim_scalar: int
        The dimension of the scalar input.
    input_dim_vector: int
        The dimension of the vector input.
    gated_blocks: int
        The number of gated blocks in the network.

    Returns
    -------
    Head

    """

    _key = "pos"

    def __init__(
        self, input_dim_scalar: int = 66, input_dim_vector: int = 64, gated_blocks: int = 3, **kwargs
    ) -> None:
        """Initialize the positions score head.

        Parameters
        ----------
        input_dim_scalar : int, optional
            Dimension of the scalar input features.
        input_dim_vector : int, optional
            Dimension of the vector input features.
        gated_blocks : int, optional
            Number of gated equivariant blocks in the network.
        **kwargs
            Additional keyword arguments forwarded to :class:`~agedi.models.head.Head`.
        """
        super().__init__(**kwargs)
        self.input_dim_scalar = input_dim_scalar
        self.input_dim_vector = input_dim_vector
        self.gated_blocks = gated_blocks
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
        }

    def _score(self, batch: dict) -> torch.Tensor:
        """Predict the positions score of the atoms in the structure.

        Parameters
        ----------
        batch : dict
            The translated input batch with ``scalar_representation`` and
            ``vector_representation`` keys.

        Returns
        -------
        torch.Tensor
            The predicted positions score.

        """
        scalar_representation = batch["scalar_representation"]
        vector_representation = batch["vector_representation"]

        scalar, vector = self.net([scalar_representation, vector_representation])

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


class CellScore(Head):
    """Predict the score for the unit cell lower-triangular entries.

    The network aggregates per-atom scalar representations to a per-graph
    representation, predicts 6 values corresponding to the 6 lower-triangular
    entries of the 3×3 canonical cell matrix, and returns the full 3×3 matrix
    with the upper-triangular entries set to zero.

    Parameters
    ----------
    input_dim_scalar : int
        The dimension of the scalar input representation.
    input_dim_vector : int
        The dimension of the vector input representation (unused; kept for
        API consistency with other heads).
    **kwargs
        Additional keyword arguments forwarded to :class:`~agedi.models.head.Head`.

    Returns
    -------
    Head

    """

    _key = "cell"

    # Indices of the lower-triangular entries in row-major order:
    #   (0,0), (1,0), (1,1), (2,0), (2,1), (2,2)
    _tril_rows = [0, 1, 1, 2, 2, 2]
    _tril_cols = [0, 0, 1, 0, 1, 2]

    def __init__(self, input_dim_scalar: int = 66, input_dim_vector: int = 64, ip=True, **kwargs):
        """Initialise the CellScore head.

        Parameters
        ----------
        input_dim_scalar : int, optional
            Dimension of the per-atom scalar features.
        input_dim_vector : int, optional
            Dimension of the per-atom vector features (unused).
        **kwargs
            Additional keyword arguments forwarded to
            :class:`~agedi.models.head.Head`.
        """
        super().__init__(**kwargs)
        self.input_dim_scalar = input_dim_scalar
        self.input_dim_vector = input_dim_vector
        self.ip = ip
        self.net = nn.Sequential(
            nn.Linear(input_dim_scalar, input_dim_scalar, bias=True),
            nn.SiLU(),
            nn.Linear(input_dim_scalar, 6, bias=False),
        )

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this cell score head."""
        return {
            **super().get_hparams(),
            "input_dim_scalar": self.input_dim_scalar,
            "input_dim_vector": self.input_dim_vector,
        }

    def _score(self, batch: dict) -> torch.Tensor:
        """Predict the cell score.

        Parameters
        ----------
        batch : dict
            The translated input batch with ``scalar_representation`` and
            ``_idx_m`` keys.

        Returns
        -------
        torch.Tensor
            Predicted score of shape ``(n_graphs, 3, 3)`` with upper-triangular
            entries set to zero.

        """
        scalar_representation = batch["scalar_representation"]
        idx_m = batch["_idx_m"]
        n_graphs = int(idx_m.max().item()) + 1

        # Aggregate atom features to graph-level: (n_graphs, input_dim_scalar)
        structure_rep = torch.zeros(n_graphs, scalar_representation.size(1),
                                    device=scalar_representation.device,
                                    dtype=scalar_representation.dtype)
        structure_rep.index_add_(0, idx_m, scalar_representation)
        counts = idx_m.bincount(minlength=n_graphs).to(dtype=scalar_representation.dtype).unsqueeze(-1)
        structure_rep = structure_rep / counts

        # Predict 6 lower-triangular entries
        values = self.net(structure_rep)  # (n_graphs, 6)

        # Assemble 3×3 lower-triangular matrix
        cell_score = torch.zeros(n_graphs, 3, 3,
                                 device=values.device, dtype=values.dtype)
        cell_score[:, self._tril_rows, self._tril_cols] = values

        # if self.ip:
        #     cell = batch["_cell"].view(-1, 3, 3)
        #     cell_score = torch.einsum("nij,njk->nik", cell_score, cell)
        

        return cell_score
