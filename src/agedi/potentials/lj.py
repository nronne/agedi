import torch
import numpy as np
from .potential import Potential
from agedi.data import AtomsGraph

import torch
from torch_geometric.data import Batch
import torch.nn as nn


class LennardJones(Potential):
    """Lennard-Jones potential implementation for atomic systems.

    Calculates the Lennard-Jones potential energy and forces for a given
    atomic configuration. Works with both single AtomsGraph and batched graphs.

    V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]
    """

    def __init__(self, epsilon=1.0, sigma=1.0, C=40, **kwargs):
        """
        Parameters
        ----------
        epsilon : float
            Depth of the potential well
        sigma : float
            Distance at which the potential is zero
        """
        super().__init__(**kwargs)
        # self.register_buffer("epsilon", torch.tensor(
        #     epsilon, dtype=torch.float))
        # self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float))

        self.epsilon = epsilon
        self.sigma = sigma
        self.C = C

    def forward(self, graph):
        """Calculate LJ energy and forces

        Parameters
        ----------
        graph : AtomsGraph or Batch
            Atomic configuration as a graph or batch of graphs

        Returns
        -------
        dict
            Dictionary containing 'energy' and 'forces'
        """
        # Ensure positions require gradient for force calculation
        grad_enabled = torch.is_grad_enabled()
        pos = graph.pos.clone()

        # Calculate energy
        energy = self.energy(graph, pos)

        if grad_enabled and pos.requires_grad:
            # Already in a gradient-enabled context, use it
            forces = -torch.autograd.grad(
                energy.sum(), pos, create_graph=True, retain_graph=True
            )[0]
        else:
            # In a non-gradient context (like validation)
            # Create a temporary gradient context just for force calculation
            with torch.enable_grad():
                pos_temp = pos.detach().requires_grad_(True)
                energy_temp = self.energy(graph, pos_temp)
                forces = -torch.autograd.grad(energy_temp.sum(), pos_temp)[0]

            
        return {"energy": energy, "forces": forces}

    def energy(self, graph, pos=None):
        """Calculate the LJ potential energy

        Parameters
        ----------
        graph : AtomsGraph or Batch
            Atomic configuration
        pos : torch.Tensor, optional
            Positions to use instead of graph.pos

        Returns
        -------
        torch.Tensor
            Energy per graph in the batch
        """
        if pos is None:
            pos = graph.pos

        edge_index = graph.edge_index
        shift_vectors = graph.shift_vectors

        # Get source and target nodes for each edge
        src, dst = edge_index

        # Get positions considering periodic boundary conditions
        src_pos = pos[src]
        dst_pos = pos[dst] + shift_vectors # should probably multiply by cell here!

        # Calculate pairwise distances
        r_vec = dst_pos - src_pos
        r = torch.norm(r_vec, dim=1)

        # Avoid division by zero
        r = torch.clamp(r, min=1e-10)

        # Calculate LJ potential
        sr6 = torch.pow(self.sigma / r, 6)
        sr12 = sr6 * sr6

        pair_energy = 4.0 * self.epsilon * (sr12 - sr6)

        # For batched data, sum energies per graph
        if isinstance(graph, Batch):
            # batch_idx = graph.batch
            # src_batch = batch_idx[src]

            # # Sum contributions for each graph
            # num_graphs = batch_idx.max().item() + 1
            # energy = torch.zeros(num_graphs, device=pos.device)

            # # Accumulate each pair contribution to the correct graph
            # # Divide by 2 to avoid double counting each pair
            # for i in range(num_graphs):
            #     mask = (src_batch == i)
            #     energy[i] = pair_energy[mask].sum() / 2.0


            # NEW: Use scatter to sum contributions for each graph in one operation
            batch_idx = graph.batch
            src_batch = batch_idx[src]

            # Sum contributions for each graph
            num_graphs = batch_idx.max().item() + 1
            energy = torch.zeros(num_graphs, device=pos.device)

            # Use scatter_add_ to sum pair energies by graph in one operation
            energy.scatter_add_(0, src_batch, pair_energy / 2.0)

            
        else:
            energy = pair_energy.sum() / 2.0

        return energy + self.C

    def energy_and_forces(self, graph):
        """Calculate energy and forces

        Parameters
        ----------
        graph : AtomsGraph or Batch
            Atomic configuration

        Returns
        -------
        energy : torch.Tensor
            Potential energy
        forces : torch.Tensor
            Forces on each atom
        """
        result = self.forward(graph)
        return result["energy"], result["forces"]

    def forces(self, graph):
        """Calculate forces

        Parameters
        ----------
        graph : AtomsGraph or Batch
            Atomic configuration

        Returns
        -------
        torch.Tensor
            Forces on each atom
        """
        return self.energy_and_forces(graph)[1]


