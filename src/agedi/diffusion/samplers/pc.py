"""Predictor-corrector sampler — EM predictor with Langevin correctors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

import torch

from .base import Sampler

if TYPE_CHECKING:
    from agedi.data import AtomsGraph
    from agedi.diffusion.noisers import Noiser


class PredictorCorrectorSampler(Sampler):
    """Euler-Maruyama predictor with Langevin corrector steps.

    After the standard EM predictor step advances the state from ``t`` to
    ``t - dt``, this sampler applies ``corrector_steps`` Langevin corrector
    steps at the new noise level ``t - dt``.  This is the conventional
    predictor-corrector scheme from Song et al. (2021).

    Parameters
    ----------
    score_fn : callable
        Score-model forward function.
    noisers : list of Noiser
        Noisers in forward order.
    corrector_steps : int, optional
        Number of Langevin corrector passes per predictor step.  Default: 1.
    corrector_step_size : float, optional
        Step size for each Langevin corrector step.  Default: 1e-3.

    Notes
    -----
    The corrector runs at the new time ``t_{i-1} = t_i - dt`` (after the
    predictor), which is the standard convention.  The legacy
    ``corrector_steps`` integer parameter on :meth:`~agedi.diffusion.Diffusion.sample`
    uses the same convention via this class.
    """

    def __init__(
        self,
        score_fn: Callable[["AtomsGraph"], "AtomsGraph"],
        noisers: List["Noiser"],
        corrector_steps: int = 1,
        corrector_step_size: float = 1e-3,
    ) -> None:
        super().__init__(score_fn, noisers)
        self.corrector_steps = corrector_steps
        self.corrector_step_size = corrector_step_size
        self._corrector_dt: Optional[torch.Tensor] = None

    def step(
        self,
        batch: "AtomsGraph",
        dt: torch.Tensor,
        last: bool,
    ) -> "AtomsGraph":
        """Predictor-corrector step.

        1. EM predictor: evaluate score, apply ``noiser.denoise()``, wrap & rebuild.
        2. Advance ``batch.time`` to ``t - dt``.
        3. For each corrector iteration: evaluate score, apply Langevin step via
           ``noiser.langevin_step()``, wrap & rebuild.

        When :attr:`~agedi.diffusion.samplers.Sampler.save_corrector_frames` is
        set, the post-predictor state and every corrector state except the last
        are recorded in ``_pending_frames``.  The last corrector state is the
        return value, which the outer loop records itself.
        """
        self._reset_pending()

        # --- Predictor (EM) ---
        batch = self.score_fn(batch)
        for noiser in self.noisers[::-1]:
            batch = noiser.denoise(batch, dt, last=last)
        batch.wrap_positions()
        self._check_finite(batch, "PC predictor (EM) step")
        batch.update_graph()

        if self.corrector_steps == 0:
            return batch

        self._capture_frame(batch)

        # Advance time to t_{i-1} for the corrector.
        batch.time = (batch.time - dt).clamp(min=0.0)

        # Lazily create corrector_dt tensor on the correct device/dtype.
        if (
            self._corrector_dt is None
            or self._corrector_dt.device != dt.device
            or self._corrector_dt.dtype != dt.dtype
        ):
            self._corrector_dt = torch.tensor(
                self.corrector_step_size, dtype=dt.dtype, device=dt.device
            )

        # --- Corrector (Langevin at t_{i-1}) ---
        for i in range(self.corrector_steps):
            batch = self.score_fn(batch)
            for noiser in self.noisers[::-1]:
                batch = noiser.langevin_step(batch, self._corrector_dt)
            batch.wrap_positions()
            self._check_finite(
                batch,
                f"Langevin corrector step (corrector_step_size={self.corrector_step_size})",
            )
            batch.update_graph()
            if i < self.corrector_steps - 1:
                self._capture_frame(batch)

        return batch
