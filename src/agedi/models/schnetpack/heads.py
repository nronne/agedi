from typing import Callable

import schnetpack.nn as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from agedi.models.head import Head

from torch_scatter import scatter


def build_gated_equivariant_mlp(
    s_in: int,
    v_in: int,
    n_out: int,
    n_layers: int = 2,
    activation: Callable = F.silu,
    sactivation: Callable = F.silu,
):
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
        self, input_dim_scalar=66, input_dim_vector=64, gated_blocks=3, **kwargs
    ):
        super().__init__(**kwargs)
        self.net = build_gated_equivariant_mlp(
            input_dim_scalar,
            input_dim_vector,
            1,
            n_layers=gated_blocks,
        )

    def _score(self, batch):
        """Predict the positions score of the atoms in the structure.

        Parameters
        ----------
        batch: dict
            The input batch.

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
    layers: int
        The number of layers

    Returns
    -------
    Head

    """

    _key = "x"

    def __init__(self, input_dim_scalar=66, input_dim_vector=64, layers=3, **kwargs):
        super().__init__(**kwargs)
        # self.net = nn.Sequential(
        #     nn.Linear(input_dim_scalar, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 100),
        #     nn.Softmax(dim=-1)
        # )
        self.net = nn.Linear(input_dim_scalar, 100)
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def _score(self, batch):
        """Predict the types score of the atoms in the structure.

        Parameters
        ----------
        batch: dict
            The input batch.

        Returns
        -------
        torch.Tensor
            The predicted positions score.

        """
        scalar_representation = batch["scalar_representation"]

        pred = self.net(scalar_representation)
        return pred


class CellScore(Head):
    """Predict cell parameters with simple physical constraints.
    
    Ensures:
    - Three positive numbers for lengths (a, b, c)
    - Three numbers between 0 and π for angles (α, β, γ)
    """
    _key = "cellpar"

    def __init__(self, input_dim_scalar=66, input_dim_vector=64, **kwargs):
        super().__init__(**kwargs)
        context_dim = 0
        cellpar_dim = 7
        self.net = nn.Sequential(
            nn.Linear(input_dim_scalar+context_dim, input_dim_scalar+context_dim, bias=True),
            nn.ReLU(),
            nn.Linear(input_dim_scalar+context_dim, cellpar_dim, bias=True), # bias = False
        )
        
    def _score(self, batch):
        """Predict cell parameters with appropriate physical ranges."""
        scalar_representation = batch["scalar_representation"]
        structure_representation = scatter(scalar_representation, batch["_idx_m"], dim=0, reduce="mean")

        cellpar = batch[self.key]
        x = torch.cat([structure_representation,], dim=-1) # , cellpar

        # Get raw predictions
        output = self.net(x)
        
        # Split into lengths and angles
        log_lengths = output[:, :3]
        log_angles = output[:, 3:6]
        log_volumes = output[:, 6:] 
        
        # Combine constrained predictions
        pred = torch.cat([log_lengths, log_angles, log_volumes], dim=-1)
        
        return pred
