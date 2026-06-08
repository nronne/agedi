"""Force-field guidance utilities for diffusion sampling.

This module provides:
- :class:`ForcefieldGuidanceConfig` – configuration dataclass.
- :class:`LBFGSStepSizer` – per-graph L-BFGS step-size adapter.
- :class:`BatchedLBFGSStepSizer` – batched wrapper around :class:`LBFGSStepSizer`.
- :func:`force_field_guidance_step` – one guidance step (module-level).
- :func:`post_diffusion_relaxation_step` – post-diffusion relaxation (module-level).
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import Optional

import torch

from agedi.data import AtomsGraph


@dataclasses.dataclass
class ForcefieldGuidanceConfig:
    """Configuration for force-field guided sampling.

    Parameters
    ----------
    guidance : float
        Scale of the force-field guidance applied at each reverse step.
        Set to ``0.0`` (the default) to disable guidance entirely.
    zeta : float
        Exponent for the time-dependent weight factor ``(1 - t)**zeta``.
        Higher values concentrate guidance near the end of the trajectory.
    force_threshold : float
        Convergence criterion for the optional post-diffusion relaxation: the
        maximum per-atom force magnitude (eV/Å) below which relaxation stops.
    max_extra_steps : int
        Maximum number of additional relaxation steps performed after the
        main diffusion trajectory when ``guidance > 0``.
    """

    guidance: float = 0.0
    zeta: float = 3.0
    force_threshold: float = 0.05
    max_extra_steps: int = 0


class LBFGSStepSizer:
    """L-BFGS approach for determining optimal step sizes in force field guidance."""

    def __init__(
        self,
        memory_size: int = 10,
        initial_step: float = 0.1,
        device: str = "cuda",
    ) -> None:
        """Initialize the L-BFGS step sizer.

        Parameters
        ----------
        memory_size : int, optional
            Number of previous iterations to store.
        initial_step : float, optional
            Initial step size scaling factor.
        device : str, optional
            Computation device (e.g. ``"cuda"`` or ``"cpu"``).
        """
        self.memory_size = memory_size
        self.initial_step = initial_step
        self.device = device

        # Storage for position and gradient differences
        self.s_list = deque(maxlen=memory_size)  # Position differences
        self.y_list = deque(maxlen=memory_size)  # Gradient (force) differences
        self.rho_list = deque(maxlen=memory_size)  # ρᵢ = 1/(yᵢᵀsᵢ)

        self.prev_pos = None
        self.prev_forces = None
        self.H0_scaling = 1.0  # Initial Hessian approximation scaling

    def compute_step(self, pos: torch.Tensor, forces: torch.Tensor) -> torch.Tensor:
        """Compute the optimal step using L-BFGS approximation.

        Parameters
        ----------
        pos : torch.Tensor
            Current atomic positions (B×N×3 tensor).
        forces : torch.Tensor
            Current forces (B×N×3 tensor).

        Returns
        -------
        torch.Tensor
            Optimal step vector (B×N×3 tensor).
        """
        if self.prev_pos is None or self.prev_forces is None:
            self.prev_pos = pos.clone().detach()
            self.prev_forces = forces.clone().detach()

            # First iteration, use simple scaling
            avg_force_mag = torch.norm(forces, dim=1).mean()
            adaptive_scale = min(self.initial_step, 0.1 / max(avg_force_mag, 1e-6))
            initial_step = adaptive_scale * forces
            return initial_step

        # Compute position and gradient differences
        s = pos - self.prev_pos  # Position difference
        y = self.prev_forces - forces  # Force difference (negative gradient)

        # Store differences if they satisfy curvature condition
        sy = torch.sum(s * y)
        if sy > 1e-10:  # Ensure positive curvature
            self.s_list.append(s)
            self.y_list.append(y)
            self.rho_list.append(1.0 / sy)

            # Update H0 scaling using Barzilai-Borwein formula
            self.H0_scaling = sy / torch.sum(y * y)

        # Apply L-BFGS two-loop recursion algorithm
        q = forces.clone()  # Start with gradient
        alpha_list = []

        # First loop
        for i in range(len(self.s_list) - 1, -1, -1):
            rho = self.rho_list[i]
            s_i = self.s_list[i]
            y_i = self.y_list[i]
            alpha_i = rho * torch.sum(s_i * q)
            alpha_list.append(alpha_i)
            q = q - alpha_i * y_i

        # Apply initial Hessian approximation
        r = self.H0_scaling * q

        # Second loop
        for i in range(len(self.s_list)):
            rho = self.rho_list[i]
            s_i = self.s_list[i]
            y_i = self.y_list[i]
            beta = rho * torch.sum(y_i * r)
            alpha = alpha_list.pop()
            r = r + (alpha - beta) * s_i

        # Save current values for next iteration
        self.prev_pos = pos.clone().detach()
        self.prev_forces = forces.clone().detach()

        # Return step (r is the approximate H⁻¹∇f)
        return r

    def reset(self) -> None:
        """Reset the L-BFGS memory."""
        self.s_list.clear()
        self.y_list.clear()
        self.rho_list.clear()
        self.prev_pos = None
        self.prev_forces = None
        self.H0_scaling = 1.0


class BatchedLBFGSStepSizer:
    """Batched wrapper around :class:`LBFGSStepSizer` for use with batched graphs.

    Maintains one :class:`LBFGSStepSizer` per graph in a batch and dispatches
    the step computation to the appropriate instance based on batch indices.
    """

    def __init__(
        self,
        batch_size: int,
        memory_size: int = 10,
        initial_step: float = 0.1,
    ) -> None:
        """Initialize one step-sizer per graph in the batch.

        Parameters
        ----------
        batch_size : int
            Number of graphs in the batch.
        memory_size : int, optional
            L-BFGS memory length (number of past iterations to retain).
        initial_step : float, optional
            Initial step-size scaling factor.
        """
        self.step_sizers = [
            LBFGSStepSizer(memory_size, initial_step) for _ in range(batch_size)
        ]

    def compute_step(
        self,
        pos: torch.Tensor,
        forces: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute steps for batched data.

        Parameters
        ----------
        pos : torch.Tensor
            Current atomic positions.
        forces : torch.Tensor
            Current forces acting on the atoms.
        batch_idx : torch.Tensor
            Index tensor mapping each atom to its graph in the batch.

        Returns
        -------
        torch.Tensor
            Combined step tensor with the same shape as *pos*.
        """
        results = []

        # Group positions and forces by batch index
        for i in range(len(self.step_sizers)):
            mask = batch_idx == i
            if torch.any(mask):
                pos_i = pos[mask]
                forces_i = forces[mask]
                step_i = self.step_sizers[i].compute_step(pos_i, forces_i)
                results.append(step_i)

        # Recombine results in original order
        combined_step = torch.zeros_like(pos)
        for i, step_i in enumerate(results):
            mask = batch_idx == i
            combined_step[mask] = step_i

        return combined_step

    def reset(self) -> None:
        """Reset the L-BFGS memory for all step-sizers in the batch."""
        for step_sizer in self.step_sizers:
            step_sizer.reset()


def force_field_guidance_step(
    batch: AtomsGraph,
    regressor_model: "torch.nn.Module",
    lbfgs_step_sizer: BatchedLBFGSStepSizer,
    scale: float,
    zeta: float = 3.0,
    max_step_size: float = 0.1,
) -> AtomsGraph:
    """Apply one force-field guidance step with batched L-BFGS step-size adaptation.

    Parameters
    ----------
    batch : AtomsGraph
        A batch of AtomsGraph data.
    regressor_model : torch.nn.Module
        The regressor model used to compute forces.
    lbfgs_step_sizer : BatchedLBFGSStepSizer
        The L-BFGS step sizer (one per graph in the batch).
    scale : float
        Base scale of the force field guidance.
    zeta : float, optional
        Exponent for the time-dependent weight ``(1 - t)**zeta``.
    max_step_size : float, optional
        Maximum allowed step size magnitude.  Default is 0.1.

    Returns
    -------
    AtomsGraph
        Updated batch after applying the guidance step.
    """
    if regressor_model is None:
        return batch

    # Apply regressor model to get forces
    batch = regressor_model(batch)

    if "forces_prediction" not in batch:
        raise ValueError("Regressor model does not compute forces.")

    # Get current positions and forces
    positions = batch.pos
    forces = batch.forces_prediction
    batch_idx = batch.batch

    # Initialize L-BFGS step sizer if not already done
    if lbfgs_step_sizer is None:
        batch_size = batch.batch_size
        lbfgs_step_sizer = BatchedLBFGSStepSizer(batch_size=batch_size)

    # Get time-dependent scaling factor
    time_factor = (1.0 - batch.time) ** zeta

    # Use L-BFGS to compute optimal step direction and magnitude
    lbfgs_step = lbfgs_step_sizer.compute_step(positions, forces, batch_idx)

    # Apply step with base scale and time factor, clamping to max_step_size
    step = scale * time_factor * lbfgs_step
    step_magnitude = torch.norm(step, dim=1, keepdim=True)
    too_large = step_magnitude > max_step_size
    if torch.any(too_large):
        scaling_factor = torch.ones_like(step_magnitude)
        scaling_factor[too_large] = max_step_size / step_magnitude[too_large]
        step = step * scaling_factor

    # Calculate new positions
    new_pos = batch.pos + step

    # Check if we need to apply confinement
    if hasattr(batch, "confinement") and batch.confinement is not None:
        z_min = batch.confinement[:, 0].unsqueeze(1)  # [B, 1]
        z_max = batch.confinement[:, 1].unsqueeze(1)  # [B, 1]

        batch_indices = batch.batch

        z_min_per_atom = z_min[batch_indices].squeeze()  # [N]
        z_max_per_atom = z_max[batch_indices].squeeze()  # [N]

        new_pos[:, 2] = torch.clamp(
            new_pos[:, 2], min=z_min_per_atom, max=z_max_per_atom
        )

    batch.pos = new_pos
    return batch


def post_diffusion_relaxation_step(
    batch: AtomsGraph,
    regressor_model: "torch.nn.Module",
    lbfgs_step_sizer: Optional[BatchedLBFGSStepSizer],
    scale: float = 0.1,
    max_step_size: float = 0.1,
) -> AtomsGraph:
    """Perform a pure force-based relaxation step after diffusion is complete.

    Parameters
    ----------
    batch : AtomsGraph
        A batch of AtomsGraph data.
    regressor_model : torch.nn.Module
        The regressor model used to compute forces.
    lbfgs_step_sizer : BatchedLBFGSStepSizer or None
        The L-BFGS step sizer.  Initialised from ``batch`` if ``None``.
    scale : float, optional
        Step size scaling factor for relaxation.

    Returns
    -------
    AtomsGraph
        Updated batch after relaxation step.
    """
    if regressor_model is None:
        return batch

    # Get forces from regressor model
    batch = regressor_model(batch)

    if "forces_prediction" not in batch:
        raise ValueError("Regressor model does not compute forces.")

    positions = batch.pos
    forces = batch.forces_prediction
    batch_idx = batch.batch

    if lbfgs_step_sizer is None:
        lbfgs_step_sizer = BatchedLBFGSStepSizer(
            batch_size=batch.batch_size,
            memory_size=10,
            initial_step=0.1,
        )

    lbfgs_step = lbfgs_step_sizer.compute_step(positions, forces, batch_idx)

    step = scale * lbfgs_step
    step_magnitude = torch.norm(step, dim=1, keepdim=True)
    too_large = step_magnitude > max_step_size
    if torch.any(too_large):
        scaling_factor = torch.ones_like(step_magnitude)
        scaling_factor[too_large] = max_step_size / step_magnitude[too_large]
        step = step * scaling_factor

    new_pos = batch.pos + step

    if hasattr(batch, "confinement") and batch.confinement is not None:
        z_min = batch.confinement[:, 0].unsqueeze(1)  # [B, 1]
        z_max = batch.confinement[:, 1].unsqueeze(1)  # [B, 1]

        batch_indices = batch.batch

        z_min_per_atom = z_min[batch_indices].squeeze()  # [N]
        z_max_per_atom = z_max[batch_indices].squeeze()  # [N]

        new_pos[:, 2] = torch.clamp(
            new_pos[:, 2], min=z_min_per_atom, max=z_max_per_atom
        )

    batch.pos = new_pos

    batch.wrap_positions()
    batch.update_graph()

    return batch
