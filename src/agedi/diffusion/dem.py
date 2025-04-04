from typing import Dict, List, Optional, Union, Tuple
from tqdm import tqdm
from copy import copy

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

    def __init__(self, potential: Potential, mc_samples: int = 10, temperature=1.0, max_force=30, **kwargs) -> None:
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
        self.temperature = temperature
        self.max_force = max_force

    # def loss(self, batch: AtomsGraph, batch_idx: torch.Tensor) -> Dict:
    #     """Computes the loss for the diffusion energy model.

    #     Parameters
    #     ----------
    #     batch: AtomsGraph
    #         A batch of AtomsGraph data.
    #     batch_idx: torch.Tensor
    #         The index of the batch.

    #     Returns
    #     -------
    #     losses: dict
    #         A dictionary of losses.

    #     """
    #     noised_batch = batch.clone()

    #     self.sample_time(noised_batch)

    #     #Noise and predict score
    #     noised_batch = self.forward_step(noised_batch)
    #     noised_batch = self.score_model(noised_batch)        
    #     score_pred = noised_batch["pos_score"]
    #     score_pred = batch.apply_mask(score_pred)


        
    #     # create monte carlo samples and estimate score
    #     noised_batch.retain_graph = True
    #     mc_batch = self._monte_carlo_sample(noised_batch)
    #     score_estimate = self._score_estimate(mc_batch)

    #     # Calculate loss
    #     loss = 100*(score_estimate - score_pred).pow(2).mean()

    #     # Debug logging
    #     se_norm = torch.norm(score_estimate)
    #     sp_norm = torch.norm(score_pred)
    #     diff_norm = torch.norm(score_estimate - score_pred)
       
    #     print(f"Step {batch_idx}: score_est_norm={se_norm.item():.6f}, "
    #           f"score_pred_norm={sp_norm.item():.6f}, diff={diff_norm.item():.6f}")


    #     losses = {
    #         "loss": loss
    #     }

    #     return losses


    # def loss_debug(self, batch: AtomsGraph, batch_idx: torch.Tensor) -> Dict:
    #     """Computes the loss with enhanced monitoring for zero-convergence issues"""
    #     noised_batch = batch.clone()
    #     self.sample_time(noised_batch)

    #     # Noise and predict score
    #     noised_batch = self.forward_step(noised_batch)
    #     noised_batch = self.score_model(noised_batch)        
    #     score_pred = noised_batch["pos_score"]
    #     score_pred = batch.apply_mask(score_pred)

    #     # Create Monte Carlo samples and estimate score
    #     noised_batch.retain_graph = True
    #     mc_batch = self._monte_carlo_sample(noised_batch)
    #     score_estimate = self._score_estimate(mc_batch)
    #     score_estimate = batch.apply_mask(score_estimate)

    #     # Calculate norms for monitoring
    #     pred_norm = torch.norm(score_pred, dim=1)
    #     est_norm = torch.norm(score_estimate, dim=1)
    #     diff_norm = torch.norm(score_estimate - score_pred, dim=1)

    #     # Monitor zero-prediction issue
    #     pred_zero_ratio = (pred_norm < 0.01).float().mean().item()

    #     # Log basic statistics
    #     self.log("train/score_pred_norm_mean", pred_norm.mean(), on_step=True)
    #     self.log("train/score_est_norm_mean", est_norm.mean(), on_step=True)
    #     self.log("train/score_diff_norm", diff_norm.mean(), on_step=True)
    #     self.log("train/zero_pred_ratio", pred_zero_ratio, on_step=True)

    #     # Log percentiles for distribution analysis
    #     percentiles = [25, 50, 75, 90, 95]
    #     pred_np = pred_norm.detach().cpu().numpy()
    #     est_np = est_norm.detach().cpu().numpy()

    #     for p in percentiles:
    #         if len(pred_np) > 0:  # Ensure we have data to compute percentiles
    #             pred_p = float(np.percentile(pred_np, p))
    #             est_p = float(np.percentile(est_np, p))
    #             self.log(f"train/score_pred_p{p}", pred_p, on_step=False, on_epoch=True)
    #             self.log(f"train/score_est_p{p}", est_p, on_step=False, on_epoch=True)

    #     # Directional alignment metric (cosine similarity)
    #     # Avoid division by zero with small epsilon
    #     eps = 1e-8
    #     pred_direction = score_pred / (pred_norm.unsqueeze(1) + eps)
    #     est_direction = score_estimate / (est_norm.unsqueeze(1) + eps)
    #     cosine_sim = (pred_direction * est_direction).sum(dim=1).mean()
    #     self.log("train/direction_cosine_sim", cosine_sim, on_step=True)

    #     # Log ratio of magnitudes (to detect scaling issues)
    #     magnitude_ratio = (pred_norm / (est_norm + eps)).clamp(0, 10).mean()
    #     self.log("train/magnitude_ratio", magnitude_ratio, on_step=True)

    #     # Calculate loss (original implementation)
    #     loss = 100 * (score_estimate - score_pred).pow(2).mean()
    #     self.log("train/loss", loss, on_step=True)

    #     losses = {
    #         "loss": loss
    #     }

    #     return losses


    def loss(self, batch, batch_idx):
        # Get predicted and estimated scores
        noised_batch = self.forward_step(batch.clone())
        noised_batch = self.score_model(noised_batch)
        score_pred = noised_batch["pos_score"]
        score_pred = batch.apply_mask(score_pred)

        # Get score estimate from Monte Carlo
        mc_batch = self._monte_carlo_sample(noised_batch)
        score_estimate = self._score_estimate(mc_batch)
        score_estimate = batch.apply_mask(score_estimate)

        # Decompose into direction and magnitude
        eps = 1e-8
        pred_norm = torch.norm(score_pred, dim=1, keepdim=True) + eps
        est_norm = torch.norm(score_estimate, dim=1, keepdim=True) + eps

        pred_direction = score_pred / pred_norm
        est_direction = score_estimate / est_norm

        # Direction loss (1 - cos similarity)
        dir_loss = (1.0 - torch.sum(pred_direction * est_direction, dim=1)).mean()

        # Magnitude loss (relative error in log space)
        mag_loss = (torch.log(pred_norm + 1.0) - torch.log(est_norm + 1.0)).pow(2).mean()

        # Zero-prediction regularizer
        pred_norm_mean = pred_norm.mean()
        zero_reg = 0.1 * torch.exp(-5.0 * pred_norm_mean)

        # Combined loss with weighting
        loss = 10.0 * dir_loss + 1.0 * mag_loss #+ zero_reg

        # Log the loss components
        self.log("train/loss", loss, on_step=True)
        self.log("train/dir_loss", dir_loss, on_step=True) 
        self.log("train/mag_loss", mag_loss, on_step=True)
        self.log("train/zero_reg", zero_reg, on_step=True)

        # Log norm statistics
        self.log("train/pred_norm_mean", pred_norm.mean(), on_step=True)
        self.log("train/est_norm_mean", est_norm.mean(), on_step=True)

        # Calculate and log zero prediction ratio
        pred_zero_ratio = (pred_norm < 0.01).float().mean().item()
        self.log("train/zero_pred_ratio", pred_zero_ratio, on_step=True)

        # Directional alignment metric (cosine similarity)
        cosine_sim = torch.sum(pred_direction * est_direction, dim=1).mean()
        self.log("train/direction_cosine_sim", cosine_sim, on_step=True)

        # Log ratio of magnitudes
        magnitude_ratio = (pred_norm / (est_norm + eps)).clamp(0, 10).mean()
        self.log("train/magnitude_ratio", magnitude_ratio, on_step=True)

        # Log percentiles for distribution analysis
        if batch_idx % 10 == 0:  # Don't compute every step to save computation
            percentiles = [25, 50, 75, 90, 95]
            pred_np = pred_norm.detach().cpu().numpy()
            est_np = est_norm.detach().cpu().numpy()

            for p in percentiles:
                if len(pred_np) > 0:  # Ensure we have data
                    pred_p = float(np.percentile(pred_np, p))
                    est_p = float(np.percentile(est_np, p))
                    self.log(f"train/score_pred_p{p}", pred_p, on_step=True)
                    self.log(f"train/score_est_p{p}", est_p, on_step=True)

        # Log histograms occasionally
        if hasattr(self, 'global_step') and self.global_step % 100 == 0:
            # Flatten for histogram computation
            pred_flat = score_pred.reshape(-1).detach()
            est_flat = score_estimate.reshape(-1).detach()

            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_histogram(
                    "train/score_pred_distribution", 
                    pred_flat.cpu(), 
                    self.global_step
                )
                self.logger.experiment.add_histogram(
                    "train/score_est_distribution", 
                    est_flat.cpu(), 
                    self.global_step
                )

        return {"loss": loss, "dir_loss": dir_loss,
                "mag_loss": mag_loss, "zero_reg": zero_reg}    
    
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
        data_list = noised_batch.to_data_list() # sometimes this fails as well!
        mc_data_list = []
        for data in data_list:
            mc_data_list +=[copy(data) for _ in range(self.mc_samples)]

        mc_batch = Batch.from_data_list(mc_data_list) # , exclude_keys=["edge_index",]

        mc_batch = self.forward_step(mc_batch, update_graph=False)
        return mc_batch

    def _score_estimate_old(self, batch: AtomsGraph) -> torch.Tensor:
        """Calculates the log expectation for the diffusion energy model.

        NB: i'm not sure if this is the correct implementation!!
        But it's a start.

        Parameters
        ----------


        """

        E, F = self.potential.energy_and_forces(batch)
        F = batch.apply_mask(F) # new and important!
        
        index = torch.arange(len(batch)//self.mc_samples, device=E.device).repeat_interleave(self.mc_samples)
        Z = torch.scatter_add(torch.zeros(len(batch)//self.mc_samples, dtype=E.dtype, device=E.device), 0, index, torch.exp(-E/self.temperature))

        w = torch.exp(-E/self.temperature)/Z.repeat_interleave(self.mc_samples)
        log_expectation = w.repeat_interleave(
            batch.n_atoms.view(-1)).unsqueeze(1)*F

        # index needs to look like: [0,1,0,1,0,1..., 2,3,2,3,2,3...]
        index = self._create_patterned_index(self.mc_samples, len(
            batch)//self.mc_samples, device=E.device).unsqueeze(1).repeat(1, 3)

        S = torch.zeros(batch.x.shape[0]//self.mc_samples, 3,
                        dtype=log_expectation.dtype, device=log_expectation.device)
        S = torch.scatter_add(S, 0, index, log_expectation)

        return S

    def _score_estimate(self, batch: AtomsGraph) -> torch.Tensor:
        """Calculates the score estimate using log-weight forces approach for numerical stability.

        Parameters
        ----------
        batch: AtomsGraph
            Batch containing MC samples for each original data point

        Returns
        -------
        torch.Tensor
            The estimated score for each original data point
        """
        # Get energies and forces
        E, F = self.potential.energy_and_forces(batch)
        F = batch.apply_mask(F)  # Apply mask to forces

        # Compute log weights
        log_weights = -E/self.temperature

        # Create index for grouping MC samples
        index = torch.arange(len(batch)//self.mc_samples, device=E.device).repeat_interleave(self.mc_samples)

        # Compute logsumexp for each group using a vectorized approach
        # We'll use a scatter operation to compute this efficiently
        max_log_weights = torch.zeros(len(batch)//self.mc_samples, dtype=log_weights.dtype, device=log_weights.device)
        max_log_weights = torch.scatter_reduce(
            max_log_weights, 0, index, log_weights, reduce="amax", include_self=False
        )

        # Stabilized exponentiation
        exp_centered = torch.exp(log_weights - max_log_weights[index])

        # Sum the centered exponentials for each group
        sumexp = torch.zeros(len(batch)//self.mc_samples, dtype=exp_centered.dtype, device=exp_centered.device)
        sumexp = torch.scatter_add(sumexp, 0, index, exp_centered)

        # Complete the logsumexp calculation
        log_Z = max_log_weights + torch.log(sumexp)

        # Normalize log weights (subtract logsumexp)
        normalized_log_weights = log_weights - log_Z[index]

        # Convert to weights by exponentiating
        weights = torch.exp(normalized_log_weights)

        # clamp forces to have max norm of 20
        F_norm = torch.norm(F, dim=1, keepdim=True)
        F = F * torch.clamp(self.max_force/(F_norm+1e-6), max=1)

        # Apply weights to forces
        weighted_forces = weights.repeat_interleave(batch.n_atoms.view(-1)).unsqueeze(1) * F

        # Create index for aggregating forces
        force_index = self._create_patterned_index(self.mc_samples, len(batch)//self.mc_samples, 
                                                 device=E.device).unsqueeze(1).repeat(1, 3)

        # Aggregate weighted forces
        S = torch.zeros(batch.x.shape[0]//self.mc_samples, 3,
                      dtype=weighted_forces.dtype, device=weighted_forces.device)
        S = torch.scatter_add(S, 0, force_index, weighted_forces)

        return S

    def _create_patterned_index(self, block_length, num_blocks, device):
        # Total number of elements
        total_length = block_length * num_blocks

        # Create indices from 0 to total_length - 1
        indices = torch.arange(total_length, device=device)

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
