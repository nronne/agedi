"""Stochastic Heun sampler — second-order reverse-SDE integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

import torch

from .base import Sampler

if TYPE_CHECKING:
    from agedi.data import AtomsGraph
    from agedi.diffusion.noisers import Noiser


class HeunSampler(Sampler):
    """Second-order stochastic sampler (Karras et al. 2022).

    Uses two score evaluations per step to achieve second-order accuracy for
    the stochastic reverse SDE:

    1. First score at ``(x_t, t)``:
       ``s_1 = s_θ(x_t, t)``
    2. Deterministic EM predictor (no noise) for SDE noisers:
       ``x̃ = x_t + dt·(g(t)²·s_1 - f(x_t, t))``
    3. Second score at ``(x̃, t - dt)``:
       ``s_2 = s_θ(x̃, t - dt)``
    4. Average scores:
       ``s_avg = 0.5·(s_1 + s_2)``
    5. Stochastic EM step using averaged score:
       ``x_{t-dt} = x_t + dt·(g(t)²·s_avg - f(x_t, t)) + √dt·g(t)·z``
       (no noise when ``last=True``)

    The score averaging provides second-order accuracy in the SDE sense,
    analogous to the stochastic Heun method from Karras et al. (2022)
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    For noisers with a ``.sde`` attribute (continuous SDE-based noisers such
    as :class:`~agedi.diffusion.noisers.PositionsNoiser`), the two-step
    procedure is applied.  For noisers without ``.sde`` (e.g. discrete types
    noiser), a single EM step is applied using the first score evaluation,
    since score averaging is less principled for discrete diffusion.

    Parameters
    ----------
    score_fn : callable
        Score-model forward function.
    noisers : list of Noiser
        Noisers in forward order.

    Notes
    -----
    ``noiser.denoise()`` reads positions and scores directly from tensor
    attributes and does **not** require an up-to-date neighbour list.  This
    allows the original positions ``x_t`` to be restored (step 5) without
    rebuilding the graph before the final denoising call in step 5 — the graph
    is only rebuilt once at the very end of the step.
    """

    def __init__(
        self,
        score_fn: Callable[["AtomsGraph"], "AtomsGraph"],
        noisers: List["Noiser"],
    ) -> None:
        super().__init__(score_fn, noisers)

    def step(
        self,
        batch: "AtomsGraph",
        dt: torch.Tensor,
        last: bool,
    ) -> "AtomsGraph":
        """Second-order stochastic Heun step."""
        sde_noisers = [n for n in self.noisers if hasattr(n, "sde")]
        other_noisers = [n for n in self.noisers if not hasattr(n, "sde")]

        # Save original SDE states and time.
        original_states = {n.key: batch[n.key].clone() for n in sde_noisers}
        t_current = batch.time.clone()

        # --- Step 1: First score call at (x_t, t) ---
        batch = self.score_fn(batch)
        s1 = {
            n.key + "_score": batch[n.key + "_score"].clone()
            for n in sde_noisers
        }
        # Save scores for non-SDE noisers; they will use the first evaluation.
        s1_other = {
            n.key + "_score": batch[n.key + "_score"].clone()
            for n in other_noisers
        }

        # --- Step 2: Deterministic EM predictor for SDE noisers ---
        # last=True suppresses the stochastic noise term, giving:
        #   x_pred = x_t + dt*(g(t)²*s1 - f(x_t, t))
        for noiser in reversed(sde_noisers):
            batch = noiser.denoise(batch, dt, last=True)
        batch.wrap_positions()

        # --- Step 3: Advance time to t-dt, rebuild graph for second score call ---
        batch.time = (t_current - dt).clamp(min=0.0)
        self._check_finite(batch, "Heun SDE predictor step")
        batch.update_graph()

        # --- Step 4: Second score call at (x_pred, t-dt) ---
        batch = self.score_fn(batch)
        s2 = {
            n.key + "_score": batch[n.key + "_score"].clone()
            for n in sde_noisers
        }

        # --- Step 5: Average scores; restore original SDE states and time ---
        # The averaged score is stored in-place on the batch.
        for key in s1:
            batch[key] = 0.5 * (s1[key] + s2[key])
        # Restore scores for non-SDE noisers to the first evaluation.
        for key, val in s1_other.items():
            batch[key] = val
        # Restore original SDE states (pos setter clears the stale graph).
        for n in sde_noisers:
            if n.key == "pos":
                batch.pos = original_states["pos"]
            else:
                batch[n.key] = original_states[n.key]
        batch.time = t_current

        # --- Step 6: Stochastic EM step using averaged scores ---
        # noiser.denoise() reads batch[key] (= x_t) and batch[key+"_score"]
        # (= averaged score) as plain tensors; no graph required.
        for noiser in reversed(sde_noisers):
            batch = noiser.denoise(batch, dt, last=last)

        # --- Step 7: Single EM step for non-SDE noisers ---
        for noiser in reversed(other_noisers):
            batch = noiser.denoise(batch, dt, last=last)

        batch.wrap_positions()
        self._check_finite(batch, "Heun SDE corrector step")
        batch.update_graph()
        return batch
