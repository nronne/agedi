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

class Forces(Head):
    """Predict the atomic force on the atoms in the structure.

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

    _key = "forces"

    def __init__(
        self, input_dim_scalar: int = 64, input_dim_vector: int = 64, gated_blocks: int = 3, **kwargs
    ) -> None:
        """Initialize the forces prediction head.

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
        """Return hyperparameters sufficient to reconstruct this head.

        Returns
        -------
        dict
            Hyperparameter dictionary.
        """
        return {
            **super().get_hparams(),
            "input_dim_scalar": self.input_dim_scalar,
            "input_dim_vector": self.input_dim_vector,
            "gated_blocks": self.gated_blocks,
        }

    def _score(self, translated_batch: dict) -> torch.Tensor:
        """Predict the force on the atoms in the structure.

        Parameters
        ----------
        translated_batch: dict
            The translated input batch.

        Returns
        -------
        torch.Tensor
            The predicted forces tensor.

        """
        scalar_representation = translated_batch["scalar_representation"]
        vector_representation = translated_batch["vector_representation"]

        scalar, vector = self.net([scalar_representation, vector_representation])

        return vector.squeeze(-1)

    def predict(self, translated_batch: dict) -> torch.Tensor:
        """Predict forces – alias for :meth:`Forces._score` kept for backwards compatibility.

        Parameters
        ----------
        translated_batch: dict
            The translated input batch.

        Returns
        -------
        torch.Tensor
            The predicted forces tensor.
        """
        return self._score(translated_batch)


class Energy(Head):
    """Predict the potential energy of the structure.

    Parameters
    ----------
    input_dim_scalar: int
        The dimension of the scalar input.

    Returns
    -------
    Head

    """

    _key = "energy"

    def __init__(
        self, input_dim_scalar: int = 64, **kwargs
    ) -> None:
        """Initialize the energy prediction head.

        Parameters
        ----------
        input_dim_scalar : int, optional
            Dimension of the scalar input features.
        **kwargs
            Additional keyword arguments forwarded to :class:`~agedi.models.head.Head`.
        """
        super().__init__(**kwargs)
        self.input_dim_scalar = input_dim_scalar
        self.net = nn.Sequential(
            nn.Linear(input_dim_scalar, input_dim_scalar // 2),
            nn.SiLU(),
            nn.Linear(input_dim_scalar // 2, 1, bias=False),
        )

    def get_hparams(self) -> Dict:
        """Return hyperparameters sufficient to reconstruct this head.

        Returns
        -------
        dict
            Hyperparameter dictionary.
        """
        return {
            **super().get_hparams(),
            "input_dim_scalar": self.input_dim_scalar,
        }

    def _score(self, translated_batch: dict) -> torch.Tensor:
        """Predict the force on the atoms in the structure.

        Parameters
        ----------
        translated_batch: dict
            The translated input batch.

        Returns
        -------
        torch.Tensor
            The predicted forces tensor.

        """
        scalar_representation = translated_batch["scalar_representation"]


        atomic_energies = self.net(scalar_representation)
        idx = translated_batch["_idx_m"]

        num_classes = idx.max().item() + 1
        energy = torch.zeros(num_classes, dtype=atomic_energies.dtype, device=atomic_energies.device)
        
        energy.scatter_add_(dim=0, index=idx, src=atomic_energies.squeeze(-1))

        return energy

    def predict(self, translated_batch: dict) -> torch.Tensor:
        """Predict forces – alias for :meth:`Forces._score` kept for backwards compatibility.

        Parameters
        ----------
        translated_batch: dict
            The translated input batch.

        Returns
        -------
        torch.Tensor
            The predicted forces tensor.
        """
        return self._score(translated_batch)
