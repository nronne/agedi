from typing import Dict, List, Optional, Union, Tuple
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.data import Batch

from agedi.data import AtomsGraph
from agedi.diffusion import Diffusion
from agedi.diffusion.noisers import Noiser
from agedi.models import ScoreModel
from agedi.potentials import Potential


class DenoisingEnergyModel(Diffusion):
    """ Implements the diffusion energy model.

    See paper: https://arxiv.org/abs/2402.06121 for details.

    Parameters
    ----------
    mc_samples: int
        The number of Monte Carlo samples to be used for the loss estimation.

    """

    def __init__(self, potential: Potential, mc_samples: int = 10, **kwargs) -> None:
        """Initializes the DiffusionEnergyModel.

        Parameters
        ----------
        potential: Potential
                The potential to be used for the loss estimation.
        mc_samples: int
            The number of Monte Carlo samples to be used for the loss estimation

        """
        super().__init__(**kwargs)
        self.potential = potential
        self.mc_samples = mc_samples

    def loss(self, batch: AtomsGraph, batch_idx: torch.Tensor) -> Dict:
        """Computes the loss for the diffusion energy model.

        Parameters
        ----------
        batch: AtomsGraph
            A batch of AtomsGraph data.
        batch_idx: torch.Tensor
            The index of the batch.

        Returns
        -------
        losses: dict
            A dictionary of losses.

        """
        noised_batch = batch.clone()

        self.sample_time(noised_batch)

        noised_batch = Batch.from_data_list([self.forward_step(g) for g in noised_batch.to_data_list()])

        # create monte carlo samples
        mc_batch = self._monte_carlo_sample(noised_batch)
        
        # calculate loss
        score_estimate = self._score_estimate(mc_batch)

        noised_batch = self.score_model(noised_batch)        
        score_pred = noised_batch["pos_score"]
        score_pred = batch.apply_mask(score_pred)

        loss = (score_estimate - score_pred).pow(2).mean()

        losses = {
            "loss": loss
        }

        return losses

    def _monte_carlo_sample(self, noised_batch: AtomsGraph) -> AtomsGraph:
        """Creates Monte Carlo samples for the noised batch.

        Parameters
        ----------
        noised_batch: AtomsGraph
            The noised batch.

        Returns
        -------
        mc_batch: AtomsGraph
            The Monte Carlo samples.

        """
        data_list = noised_batch.to_data_list()
        mc_data_list = []
        for data in data_list:
            mc_data_list += [self.forward_step(data.clone()) for _ in range(self.mc_samples)]

        mc_batch = Batch.from_data_list(mc_data_list)
        return mc_batch

    def _score_estimate(self, batch: AtomsGraph) -> torch.Tensor:
        """Calculates the log expectation for the diffusion energy model.

        NB: i'm not sure if this is the correct implementation!!
        But it's a start.

        Parameters
        ----------


        """

        E, F = self.potential(batch)
        index = torch.arange(len(batch)//self.mc_samples).repeat_interleave(self.mc_samples)
        Z = torch.scatter_add(torch.zeros(len(batch)//self.mc_samples, dtype=E.dtype), 0, index, torch.exp(-E))
        
        w = torch.exp(-E)/Z.repeat_interleave(self.mc_samples)
        log_expectation = w.repeat_interleave(
            batch.n_atoms.view(-1)).unsqueeze(1)*F

        # index needs to look like: [0,1,0,1,0,1..., 2,3,2,3,2,3...]
        index = self._create_patterned_index(self.mc_samples, len(
            batch)//self.mc_samples).unsqueeze(1).repeat(1, 3)

        S = torch.zeros(batch.x.shape[0]//self.mc_samples, 3,
                        dtype=log_expectation.dtype, device=log_expectation.device)
        S = torch.scatter_add(S, 0, index, log_expectation)

        return S

    def _create_patterned_index(self, block_length, num_blocks):
        # Total number of elements
        total_length = block_length * num_blocks

        # Create indices from 0 to total_length - 1
        indices = torch.arange(total_length)

        # Calculate block index for each position (0 to num_blocks-1)
        block_indices = indices // block_length

        # Calculate position within block (0 to block_length-1)
        position_in_block = indices % block_length

        # Calculate if position is odd or even within block (0 or 1)
        odd_even = position_in_block % 2

        # Calculate the final index
        # For each block i, indices alternate between 2i and 2i+1
        result = 2 * block_indices + odd_even

        return result
