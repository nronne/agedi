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

    def loss(self, batch: AtomsGraph, batch_idx: torch.Tensor) -> Dict:
        # Get predicted and estimated scores
        noised_batch = batch.clone()
        self.sample_time(noised_batch)

        noised_batch = self.forward_step(noised_batch)
        noised_batch = self.score_model(noised_batch)
        
        score_pred = noised_batch["pos_score"]
        score_pred = batch.apply_mask(score_pred)

        # Get score estimate from Monte Carlo
        mc_batch = self._monte_carlo_sample(noised_batch)
        score_estimate = self._score_estimate(mc_batch)
        score_estimate = batch.apply_mask(score_estimate)

        # Decompose into direction and magnitude
        eps = 1e-8
        pred_norm = torch.norm(score_pred[~batch.positions_mask].reshape(-1, 3), dim=1, keepdim=True) + eps
        est_norm = torch.norm(score_estimate[~batch.positions_mask].reshape(-1, 3), dim=1, keepdim=True) + eps

        pred_direction = score_pred[~batch.positions_mask].reshape(-1, 3) / pred_norm
        est_direction = score_estimate[~batch.positions_mask].reshape(-1, 3) / est_norm

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

    def _score_estimate(self, batch: AtomsGraph) -> torch.Tensor:
        """Calculates score estimate using vectorized operations for efficiency.
        Assumes all samples have the same atom count.

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
        F = batch.apply_mask(F)

        # # clamp forces to have max norm of 20
        # F_norm = torch.norm(F, dim=1, keepdim=True)
        # F = F * torch.clamp(self.max_force/(F_norm+1e-6), max=1)

        # Number of original data points
        batch_size = len(batch) // self.mc_samples

        # Get number of atoms per sample
        atoms_per_sample = batch.n_atoms[0].item()

        # Compute log weights for numerical stability
        log_weights = -E / self.temperature

        # Reshape forces to [batch_size, mc_samples, atoms_per_sample, 3]
        F_reshaped = F.reshape(batch_size, self.mc_samples, atoms_per_sample, 3)

        # Reshape log weights to [batch_size, mc_samples]
        log_weights_reshaped = log_weights.reshape(batch_size, self.mc_samples)

        # Normalize weights within each batch using logsumexp for stability
        log_weights_norm = log_weights_reshaped - torch.logsumexp(log_weights_reshaped, dim=1, keepdim=True)
        weights = torch.exp(log_weights_norm)  # Shape: [batch_size, mc_samples]

        # Apply weights: [batch_size, mc_samples, 1, 1] * [batch_size, mc_samples, atoms_per_sample, 3]
        weighted_forces = weights.view(batch_size, self.mc_samples, 1, 1) * F_reshaped

        # Sum over MC samples: [batch_size, atoms_per_sample, 3]
        score_estimate = torch.sum(weighted_forces, dim=1)

        # Flatten to match expected output format: [total_atoms, 3]
        score_estimate = score_estimate.reshape(-1, 3)

        return score_estimate

    def loss_cartesian(self, batch: AtomsGraph, batch_idx: torch.Tensor) -> Dict:
        """Computes the loss with enhanced monitoring for zero-convergence issues"""
        noised_batch = batch.clone()
        self.sample_time(noised_batch)

        # Noise and predict score
        noised_batch = self.forward_step(noised_batch)
        noised_batch = self.score_model(noised_batch)        
        score_pred = noised_batch["pos_score"]
        score_pred = batch.apply_mask(score_pred)

        # Create Monte Carlo samples and estimate score
        noised_batch.retain_graph = True
        mc_batch = self._monte_carlo_sample(noised_batch)
        score_estimate = self._score_estimate(mc_batch)
        score_estimate = batch.apply_mask(score_estimate)

        # Calculate norms for monitoring
        pred_norm = torch.norm(score_pred, dim=1)
        est_norm = torch.norm(score_estimate, dim=1)
        diff_norm = torch.norm(score_estimate - score_pred, dim=1)

        # Monitor zero-prediction issue
        pred_zero_ratio = (pred_norm < 0.01).float().mean().item()

        # Log basic statistics
        self.log("train/score_pred_norm_mean", pred_norm.mean(), on_step=True)
        self.log("train/score_est_norm_mean", est_norm.mean(), on_step=True)
        self.log("train/score_diff_norm", diff_norm.mean(), on_step=True)
        self.log("train/zero_pred_ratio", pred_zero_ratio, on_step=True)

        # Log percentiles for distribution analysis
        percentiles = [25, 50, 75, 90, 95]
        pred_np = pred_norm.detach().cpu().numpy()
        est_np = est_norm.detach().cpu().numpy()

        for p in percentiles:
            if len(pred_np) > 0:  # Ensure we have data to compute percentiles
                pred_p = float(np.percentile(pred_np, p))
                est_p = float(np.percentile(est_np, p))
                self.log(f"train/score_pred_p{p}", pred_p, on_step=False, on_epoch=True)
                self.log(f"train/score_est_p{p}", est_p, on_step=False, on_epoch=True)

        # Directional alignment metric (cosine similarity)
        # Avoid division by zero with small epsilon
        eps = 1e-8
        pred_direction = score_pred / (pred_norm.unsqueeze(1) + eps)
        est_direction = score_estimate / (est_norm.unsqueeze(1) + eps)
        cosine_sim = (pred_direction * est_direction).sum(dim=1).mean()
        self.log("train/direction_cosine_sim", cosine_sim, on_step=True)

        # Log ratio of magnitudes (to detect scaling issues)
        magnitude_ratio = (pred_norm / (est_norm + eps)).clamp(0, 10).mean()
        self.log("train/magnitude_ratio", magnitude_ratio, on_step=True)

        # Calculate loss (original implementation)
        loss = (score_estimate - score_pred).pow(2).mean()
        self.log("train/loss", loss, on_step=True)

        losses = {
            "loss": loss
        }

        return losses

