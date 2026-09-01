"""Force-field guidance utilities for diffusion sampling.

This module provides:
- :class:`ForcefieldGuidanceConfig` – configuration dataclass.
- :class:`LBFGSStepSizer` – per-graph L-BFGS step-size adapter.
- :class:`BatchedLBFGSStepSizer` – batched wrapper around :class:`LBFGSStepSizer`.
- :func:`force_field_guidance_step` – one guidance step (module-level).
- :func:`post_diffusion_relaxation_step` – post-diffusion relaxation (module-level).
- :func:`max_force_per_graph` – per-structure convergence measure.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import Optional, Tuple

import torch

from agedi.data import AtomsGraph


def minimum_image(
    d: torch.Tensor,
    cell: Optional[torch.Tensor],
    pbc: Optional[torch.Tensor],
) -> torch.Tensor:
    """Map displacement vectors to their shortest periodic image.

    Positions are wrapped back into the unit cell after every step (see
    :meth:`~agedi.data.AtomsGraph.wrap_positions`), so differencing two stored
    position tensors reports a full lattice vector whenever an atom crossed a
    cell face — even though the atom barely moved.  Feeding such a displacement
    into the L-BFGS history destroys the curvature estimate, so every
    displacement reconstructed from stored positions must pass through here
    first.

    Parameters
    ----------
    d : torch.Tensor
        Displacement vectors, shape ``(n_atoms, 3)``.
    cell : torch.Tensor or None
        Unit cell of the structure, shape ``(3, 3)``, row-vector convention
        (``r = f @ cell``).  ``None`` disables the correction.
    pbc : torch.Tensor or None
        Boolean periodicity flags, shape ``(3,)``.  Non-periodic directions are
        left untouched.  ``None`` disables the correction.

    Returns
    -------
    torch.Tensor
        Displacements with any lattice-vector jumps removed.
    """
    if cell is None or pbc is None or not bool(pbc.any()):
        return d

    cell = cell.view(3, 3).to(d.dtype)
    # Degenerate (zero) cells appear on non-periodic graphs; nothing to wrap.
    if not bool(torch.linalg.det(cell).abs() > 1e-12):
        return d

    # r = f @ cell  =>  f = solve(cell.T, r.T).T   (matches AtomsGraph.pos_to_frac)
    frac = torch.linalg.solve(cell.transpose(0, 1), d.transpose(0, 1)).transpose(0, 1)
    shift = torch.round(frac) * pbc.to(frac.dtype)
    return d - shift @ cell


def max_force_per_graph(
    forces: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    """Return the maximum per-atom force magnitude of every graph in a batch.

    Relaxation convergence is a per-structure property: taking the maximum over
    the whole batch keeps every structure stepping until the worst one is done.

    Parameters
    ----------
    forces : torch.Tensor
        Per-atom forces, shape ``(n_atoms, 3)``.
    batch_idx : torch.Tensor
        Graph membership index (``batch.batch``), shape ``(n_atoms,)``.
    num_graphs : int
        Number of graphs in the batch.

    Returns
    -------
    torch.Tensor
        Maximum force magnitude per graph, shape ``(num_graphs,)``.
    """
    norms = torch.norm(forces, dim=1)
    out = torch.zeros(num_graphs, dtype=norms.dtype, device=norms.device)
    out.scatter_reduce_(0, batch_idx, norms, reduce="amax")
    return out


def _cell_and_pbc(batch: AtomsGraph) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract per-graph ``(cell, pbc)`` tensors from *batch*, if available."""
    cell = getattr(batch, "cell", None)
    pbc = getattr(batch, "pbc", None)
    if cell is None or pbc is None:
        return None, None
    return cell.view(-1, 3, 3), pbc.view(-1, 3)


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
        Maximum number of L-BFGS relaxation steps performed after the main
        diffusion trajectory.  Independent of ``guidance``: set this alone to
        relax the final structures without perturbing the diffusion trajectory
        itself.  Relaxation stops early once the maximum per-atom force drops
        below ``force_threshold``, and is skipped entirely if the structures
        are already converged.  Requires a model with a regressor (forces)
        head.  ``0`` (the default) disables it.
    """

    guidance: float = 0.0
    zeta: float = 3.0
    force_threshold: float = 0.05
    max_extra_steps: int = 0


class LBFGSStepSizer:
    """L-BFGS optimiser step, mirroring :class:`ase.optimize.LBFGS`.

    The algorithm follows ASE's implementation closely so that relaxation
    behaves the way users expect from ASE:

    * The inverse-Hessian seed ``H0 = 1/alpha`` is **constant**.  ASE notes
      that this emulates BFGS and is deliberately never updated; an adaptive
      Barzilai-Borwein estimate here made the step length oscillate.
    * The step is limited by scaling the **whole** displacement by a single
      factor ``maxstep / longest_atom_step`` (ASE's ``determine_step``).
      Rescaling atoms individually would rotate the search direction away from
      the eigendirection rather than simply shortening the step.
    * History pairs are stored unconditionally, as in ASE.

    Like ASE's default ``LBFGS`` (``use_line_search=False``) there is no line
    search, so the energy is not guaranteed to decrease monotonically; the step
    cap is what keeps the trajectory stable.

    Parameters
    ----------
    memory_size : int, optional
        Number of history pairs retained.  ASE default: ``100``.
    maxstep : float, optional
        Maximum distance any single atom may move in one step, in Å.
        ASE default: ``0.2``.
    alpha : float, optional
        Initial guess for the curvature of the energy surface; the
        inverse-Hessian seed is ``1/alpha``.  ASE default: ``70.0``.  Lower
        values take larger steps and converge faster at the cost of stability.
    damping : float, optional
        The computed step is multiplied by this before being returned.
        ASE default: ``1.0``.
    """

    def __init__(
        self,
        memory_size: int = 100,
        maxstep: float = 0.2,
        alpha: float = 70.0,
        damping: float = 1.0,
    ) -> None:
        self.memory_size = memory_size
        self.maxstep = maxstep
        self.damping = damping
        # Initial approximation of the inverse Hessian; constant, as in ASE.
        self.H0 = 1.0 / alpha

        self.s_list = deque(maxlen=memory_size)  # position differences
        self.y_list = deque(maxlen=memory_size)  # gradient differences
        self.rho_list = deque(maxlen=memory_size)  # 1 / (yᵢ·sᵢ)

        self.prev_pos = None
        self.prev_forces = None

    def compute_step(
        self,
        pos: torch.Tensor,
        forces: torch.Tensor,
        maxstep: Optional[float] = None,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the L-BFGS displacement for one structure.

        Parameters
        ----------
        pos : torch.Tensor
            Current atomic positions, shape ``(n_atoms, 3)``.
        forces : torch.Tensor
            Current forces, shape ``(n_atoms, 3)``.  Note these are forces,
            i.e. the *negative* gradient.
        maxstep : float, optional
            Overrides :attr:`maxstep` for this call.
        cell : torch.Tensor, optional
            Unit cell, shape ``(3, 3)``.  Required together with *pbc* for
            periodic structures so that the displacement reconstructed from
            stored positions is taken in the minimum-image convention; without
            it, an atom that wrapped across a cell face injects a
            lattice-vector-sized ``s0`` into the history and the search
            direction stops following the forces.
        pbc : torch.Tensor, optional
            Boolean periodicity flags, shape ``(3,)``.

        Returns
        -------
        torch.Tensor
            Displacement to add to *pos*, already step-limited and damped.
        """
        # --- Update history (ASE: LBFGS.update) ---
        if self.prev_pos is not None:
            s0 = minimum_image(pos - self.prev_pos, cell, pbc)
            # We use the gradient, which is minus the force.
            y0 = self.prev_forces - forces
            ys = torch.sum(y0 * s0)
            # ASE stores every pair; guard only against division by zero,
            # which would poison the whole history with inf.
            if torch.abs(ys) > 1e-30:
                self.s_list.append(s0)
                self.y_list.append(y0)
                self.rho_list.append(1.0 / ys)

        # --- Two-loop recursion (ASE: LBFGS.step) ---
        q = -forces.clone()
        alphas = []
        for i in range(len(self.s_list) - 1, -1, -1):
            alpha_i = self.rho_list[i] * torch.sum(self.s_list[i] * q)
            alphas.append(alpha_i)
            q = q - alpha_i * self.y_list[i]

        z = self.H0 * q

        for i in range(len(self.s_list)):
            beta = self.rho_list[i] * torch.sum(self.y_list[i] * z)
            # alphas was filled in reverse, so popping walks it forward again.
            alpha_i = alphas.pop()
            z = z + self.s_list[i] * (alpha_i - beta)

        p = -z

        self.prev_pos = pos.clone().detach()
        self.prev_forces = forces.clone().detach()

        return self.determine_step(p, maxstep) * self.damping

    def determine_step(
        self, dr: torch.Tensor, maxstep: Optional[float] = None
    ) -> torch.Tensor:
        """Limit the step according to *maxstep* (ASE: ``determine_step``).

        All atoms are scaled by the same factor, derived from the longest
        single-atom displacement, so the step shortens along the eigendirection
        instead of being bent by per-atom clamping.
        """
        limit = self.maxstep if maxstep is None else maxstep
        longest_step = torch.norm(dr, dim=1).max()
        if longest_step >= limit:
            dr = dr * (limit / longest_step)
        return dr

    def reset(self) -> None:
        """Reset the L-BFGS memory."""
        self.s_list.clear()
        self.y_list.clear()
        self.rho_list.clear()
        self.prev_pos = None
        self.prev_forces = None


class BatchedLBFGSStepSizer:
    """Batched wrapper around :class:`LBFGSStepSizer` for use with batched graphs.

    Maintains one :class:`LBFGSStepSizer` per graph in a batch and dispatches
    the step computation to the appropriate instance based on batch indices.
    """

    def __init__(
        self,
        batch_size: int,
        memory_size: int = 100,
        maxstep: float = 0.2,
        alpha: float = 70.0,
        damping: float = 1.0,
    ) -> None:
        """Initialize one step-sizer per graph in the batch.

        Parameters
        ----------
        batch_size : int
            Number of graphs in the batch.
        memory_size : int, optional
            L-BFGS memory length (number of past iterations to retain).
        maxstep : float, optional
            Maximum single-atom displacement per step, in Å.
        alpha : float, optional
            Initial curvature guess; the inverse-Hessian seed is ``1/alpha``.
        damping : float, optional
            Multiplier applied to the computed step.

        See :class:`LBFGSStepSizer` for the meaning of each parameter; the
        defaults match :class:`ase.optimize.LBFGS`.
        """
        self.step_sizers = [
            LBFGSStepSizer(
                memory_size=memory_size,
                maxstep=maxstep,
                alpha=alpha,
                damping=damping,
            )
            for _ in range(batch_size)
        ]

    def compute_step(
        self,
        pos: torch.Tensor,
        forces: torch.Tensor,
        batch_idx: torch.Tensor,
        maxstep: Optional[float] = None,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
        active: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute steps for batched data.

        Each graph is optimised by its own :class:`LBFGSStepSizer`, so the
        step limit applies per structure exactly as it would when relaxing
        that structure alone in ASE.

        Parameters
        ----------
        pos : torch.Tensor
            Current atomic positions.
        forces : torch.Tensor
            Current forces acting on the atoms.
        batch_idx : torch.Tensor
            Index tensor mapping each atom to its graph in the batch.
        maxstep : float, optional
            Overrides the per-sizer step limit for this call.
        cell : torch.Tensor, optional
            Per-graph cells, shape ``(num_graphs, 3, 3)``.  Forwarded to
            :meth:`LBFGSStepSizer.compute_step` for the minimum-image
            correction.
        pbc : torch.Tensor, optional
            Per-graph periodicity flags, shape ``(num_graphs, 3)``.
        active : torch.Tensor, optional
            Boolean mask over graphs, shape ``(num_graphs,)``.  Graphs marked
            ``False`` get a zero step and their L-BFGS history is left
            untouched, so a structure that has already converged is not
            disturbed while the rest of the batch keeps relaxing.

        Returns
        -------
        torch.Tensor
            Combined step tensor with the same shape as *pos*.
        """
        combined_step = torch.zeros_like(pos)

        # Scatter straight back into the graph's own rows.  Collecting results
        # into a list and re-enumerating would misalign every graph after an
        # empty one, silently giving atoms another structure's step.
        for i, step_sizer in enumerate(self.step_sizers):
            if active is not None and not bool(active[i]):
                continue
            mask = batch_idx == i
            if torch.any(mask):
                combined_step[mask] = step_sizer.compute_step(
                    pos[mask],
                    forces[mask],
                    maxstep=maxstep,
                    cell=None if cell is None else cell[i],
                    pbc=None if pbc is None else pbc[i],
                )

        return combined_step

    def reset(self) -> None:
        """Reset the L-BFGS memory for all step-sizers in the batch."""
        for step_sizer in self.step_sizers:
            step_sizer.reset()


def _restrict_to_inpainted(batch: AtomsGraph, new_pos: torch.Tensor) -> torch.Tensor:
    """Undo a guidance step's displacement of known (non-inpainted) atoms.

    Force-field guidance and post-diffusion relaxation move every atom that
    isn't hard-frozen via ``batch.mask`` -- they have no notion of the
    "known vs. regenerated" split used by inpainting (``batch.inpaint_mask``),
    so left alone they nudge known/context atoms off their reference
    trajectory using predicted forces, up to and including the final step
    where inpainting otherwise guarantees an exact reconstruction. When
    ``batch.inpaint_mask`` is present, only the atoms it marks for
    regeneration keep the guidance step; every other atom keeps its
    pre-guidance position. A no-op for ordinary (non-inpainting) sampling,
    where the attribute is absent.
    """
    if "inpaint_mask" not in batch:
        return new_pos
    select = batch.inpaint_mask.view(-1, *([1] * (new_pos.dim() - 1)))
    return torch.where(select, new_pos, batch.pos)


def reassert_known_positions(batch: AtomsGraph, pre_wrap_pos: torch.Tensor) -> None:
    """Undo a periodic-image flip ``wrap_positions()`` may cause for known atoms.

    ``AtomsGraph.wrap_positions()`` can flip an atom sitting near a
    periodic-cell boundary to the adjacent image -- a jump by a full lattice
    vector, not the small guidance/relaxation displacement it was meant to
    represent. Callers that write ``batch.pos`` via :func:`_restrict_to_inpainted`
    and then call ``wrap_positions()`` must call this immediately afterwards
    (and before ``update_graph()``, so the returned neighbor list is built
    from the final, exact positions) to restore known atoms to *pre_wrap_pos*
    -- their value right after the restricted guidance write, before
    wrapping. A no-op for ordinary (non-inpainting) sampling, where
    ``inpaint_mask`` is absent.

    Parameters
    ----------
    batch : AtomsGraph
        The batch, already guidance-updated and wrapped.
    pre_wrap_pos : torch.Tensor
        ``batch.pos`` as returned by the guidance step, captured before
        ``wrap_positions()`` was called.
    """
    if "inpaint_mask" not in batch:
        return
    known = ~batch.inpaint_mask
    select = known.view(-1, *([1] * (batch.pos.dim() - 1)))
    batch.pos = torch.where(select, pre_wrap_pos, batch.pos)


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

    # Use L-BFGS to compute the step direction and magnitude.  The sizer caps
    # the displacement per structure, scaling the whole step uniformly so the
    # search direction is preserved; guidance strength is then applied on top.
    cell, pbc = _cell_and_pbc(batch)
    lbfgs_step = lbfgs_step_sizer.compute_step(
        positions, forces, batch_idx, maxstep=max_step_size, cell=cell, pbc=pbc
    )

    step = scale * time_factor * lbfgs_step

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

    batch.pos = _restrict_to_inpainted(batch, new_pos)
    return batch


def post_diffusion_relaxation_step(
    batch: AtomsGraph,
    regressor_model: "torch.nn.Module",
    lbfgs_step_sizer: Optional[BatchedLBFGSStepSizer],
    scale: float = 1.0,
    max_step_size: float = 0.2,
    forces: Optional[torch.Tensor] = None,
    active: Optional[torch.Tensor] = None,
) -> AtomsGraph:
    """Perform one L-BFGS relaxation step after diffusion is complete.

    Equivalent to a single ``ase.optimize.LBFGS`` step applied to every
    structure in the batch independently.

    Parameters
    ----------
    batch : AtomsGraph
        A batch of AtomsGraph data.
    regressor_model : torch.nn.Module
        The regressor model used to compute forces.
    lbfgs_step_sizer : BatchedLBFGSStepSizer or None
        The L-BFGS step sizer.  Initialised from ``batch`` if ``None`` —
        though a persistent sizer should be passed, since a fresh one carries
        no curvature history and degrades the relaxation to seed-length steps.
    scale : float, optional
        Multiplier on the computed step, equivalent to ASE's ``damping``.
        Defaults to ``1.0`` (ASE's default), i.e. take the full L-BFGS step.
    max_step_size : float, optional
        Maximum single-atom displacement per step, in Å.  Defaults to ``0.2``
        (ASE's default).  Applied by scaling the whole step uniformly.
    forces : torch.Tensor, optional
        Forces at the current positions.  When given, *regressor_model* is not
        called: the caller's convergence check already evaluated the forces at
        exactly these positions, and re-evaluating them here would double the
        cost of every relaxation step.
    active : torch.Tensor, optional
        Boolean mask over graphs, shape ``(num_graphs,)``.  Structures marked
        ``False`` are left untouched.

    Returns
    -------
    AtomsGraph
        Updated batch after relaxation step.
    """
    if regressor_model is None:
        return batch

    if forces is None:
        # Get forces from regressor model
        batch = regressor_model(batch)

        if "forces_prediction" not in batch:
            raise ValueError("Regressor model does not compute forces.")
        forces = batch.forces_prediction

    positions = batch.pos
    batch_idx = batch.batch

    if lbfgs_step_sizer is None:
        lbfgs_step_sizer = BatchedLBFGSStepSizer(batch_size=batch.batch_size)

    # The sizer already applies the maxstep limit, uniformly per structure.
    cell, pbc = _cell_and_pbc(batch)
    step = scale * lbfgs_step_sizer.compute_step(
        positions,
        forces,
        batch_idx,
        maxstep=max_step_size,
        cell=cell,
        pbc=pbc,
        active=active,
    )

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

    restricted_pos = _restrict_to_inpainted(batch, new_pos)
    batch.pos = restricted_pos

    atom_mask = None if active is None else active[batch.batch]
    batch.wrap_positions(atom_mask=atom_mask)
    reassert_known_positions(batch, restricted_pos)
    batch.update_graph()

    return batch
