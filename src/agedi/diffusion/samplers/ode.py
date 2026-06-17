"""Probability-flow ODE samplers — deterministic reverse diffusion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

import torch

from .base import Sampler

if TYPE_CHECKING:
    from agedi.data import AtomsGraph
    from agedi.diffusion.noisers import Noiser


class ProbabilityFlowODESampler(Sampler):
    """Deterministic probability-flow ODE sampler (DDIM / Anderson 1982).

    Integrates the reverse-time ODE:

    .. math::

        \\frac{d\\mathbf{x}}{dt} = f(\\mathbf{x}, t)
            - \\tfrac{1}{2}\\, g(t)^2\\, \\nabla_{\\mathbf{x}} \\log p_t(\\mathbf{x})

    which is the deterministic counterpart of the reverse-time SDE.  The
    factor of ``0.5`` on the diffusion term (vs ``1.0`` in Euler-Maruyama)
    removes the stochastic component while preserving the marginal
    distributions.

    The update for one step ``t → t - dt`` is:

    .. math::

        \\mathbf{x}_{t - \\Delta t} =
            \\mathbf{x}_t
            + \\Delta t \\bigl(
                \\tfrac{1}{2}\\, g(t)^2 \\, s_\\theta(\\mathbf{x}_t, t)
                - f(\\mathbf{x}_t, t)
              \\bigr)

    For noisers with a ``.sde`` attribute (continuous SDE-based noisers, such
    as positions), the ODE update is applied directly using the SDE's
    ``drift`` and ``diffusion`` methods.  For other noisers (e.g. discrete
    types), this sampler falls back to a deterministic EM step
    (``noiser.denoise(batch, dt, last=True)``).

    Unlike :class:`EulerMaruyamaSampler`, this sampler is fully deterministic:
    repeated calls with identical inputs produce identical outputs.

    Parameters
    ----------
    score_fn : callable
        Score-model forward function.
    noisers : list of Noiser
        Noisers in forward order.
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
        """Probability-flow ODE step.

        1. Evaluate score model.
        2. For each SDE-based noiser: apply the ODE update (half diffusion, no noise).
        3. For other noisers: apply a deterministic EM step.
        4. Wrap positions and rebuild the neighbour list.

        The ``last`` parameter is accepted for API compatibility but has no
        effect — this sampler is always deterministic.
        """
        batch = self.score_fn(batch)
        for noiser in self.noisers[::-1]:
            if hasattr(noiser, "sde"):
                batch = self._ode_step(batch, noiser, dt)
            else:
                batch = noiser.denoise(batch, dt, last=True)
        batch.wrap_positions()
        batch.update_graph()
        return batch

    @staticmethod
    def _ode_step(
        batch: "AtomsGraph",
        noiser: "Noiser",
        dt: torch.Tensor,
    ) -> "AtomsGraph":
        """Apply the probability-flow ODE update for one SDE-based noiser.

        Computes:

        .. math::

            \\mathbf{r}_{t-\\Delta t} =
                \\mathbf{r}_t
                + \\Delta t \\bigl(
                    \\tfrac{1}{2}\\, g(t)^2\\, s_\\theta - f(\\mathbf{r}_t, t)
                  \\bigr)

        Parameters
        ----------
        batch : AtomsGraph
            Current batch state.
        noiser : Noiser
            An SDE-based noiser with a ``.sde`` attribute.
        dt : torch.Tensor
            Step size.

        Returns
        -------
        AtomsGraph
            Batch with the updated attribute.
        """
        r = batch[noiser.key]
        r_score = batch[noiser.key + "_score"]

        # Zero out NaN scores (mirrors PositionsNoiser._denoise).
        r_score = torch.where(
            torch.isnan(r_score), torch.zeros_like(r_score), r_score
        )

        t = batch.time

        # Convert epsilon-prediction → score when needed.
        if getattr(noiser, "prediction_type", "score") == "epsilon":
            sigma = torch.sqrt(noiser.sde.var(t))
            r_score = -r_score / sigma

        drift = noiser.sde.drift(r, t)   # f(r, t)
        g = noiser.sde.diffusion(t)      # g(t)

        # ODE update: half the diffusion coefficient, no noise term.
        new_r = r + dt * (0.5 * g**2 * r_score - drift)

        # Replicate confinement clamping from PositionsNoiser._denoise.
        if batch.confinement is not None and noiser.key == "pos":
            confinement = batch.confinement[batch.batch]  # (n_atoms, 2)
            mobile = ~batch.mask
            clamped_z = torch.clamp(
                new_r[:, 2],
                min=confinement[:, 0],
                max=confinement[:, 1],
            )
            new_r = new_r.clone()
            new_r[:, 2] = torch.where(mobile, clamped_z, new_r[:, 2])

        setattr(batch, noiser.key, new_r)
        return batch


class HeunODESampler(Sampler):
    """Second-order deterministic ODE sampler (Heun's method).

    Applies Heun's method to the probability-flow ODE, achieving second-order
    accuracy with two score evaluations per step:

    1. First score at ``(x_t, t)``: ``s_1 = s_θ(x_t, t)``
    2. ODE predictor: ``x̃ = x_t + dt·(0.5·g(t)²·s_1 - f(x_t, t))``
    3. Second score at ``(x̃, t - dt)``: ``s_2 = s_θ(x̃, t - dt)``
    4. Averaged score: ``s_avg = 0.5·(s_1 + s_2)``
    5. ODE corrector: ``x_{t-dt} = x_t + dt·(0.5·g(t)²·s_avg - f(x_t, t))``

    Like :class:`ProbabilityFlowODESampler`, this sampler is fully
    deterministic.  It uses two score evaluations per step vs one for the
    first-order ODE sampler, but achieves better quality at lower step counts.

    For noisers without ``.sde`` (discrete types), a single deterministic EM
    step is applied using the first score evaluation.

    Parameters
    ----------
    score_fn : callable
        Score-model forward function.
    noisers : list of Noiser
        Noisers in forward order.
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
        """Second-order Heun ODE step.

        The ``last`` parameter is accepted for API compatibility but has no
        effect — this sampler is always deterministic.
        """
        sde_noisers = [n for n in self.noisers if hasattr(n, "sde")]
        other_noisers = [n for n in self.noisers if not hasattr(n, "sde")]

        # Save original positions and time.
        x_t = batch.pos.clone()
        t_current = batch.time.clone()

        # --- Step 1: First score call at (x_t, t) ---
        batch = self.score_fn(batch)
        s1 = {
            n.key + "_score": batch[n.key + "_score"].clone()
            for n in sde_noisers
        }
        # Also save scores for non-SDE noisers (used in the single EM step later).
        s1_other = {
            n.key + "_score": batch[n.key + "_score"].clone()
            for n in other_noisers
        }

        # --- Step 2: ODE predictor for SDE noisers ---
        for noiser in reversed(sde_noisers):
            batch = ProbabilityFlowODESampler._ode_step(batch, noiser, dt)
        batch.wrap_positions()

        # --- Step 3: Advance time to t-dt, rebuild graph ---
        batch.time = (t_current - dt).clamp(min=0.0)
        batch.update_graph()

        # --- Step 4: Second score call at (x_pred, t-dt) ---
        batch = self.score_fn(batch)
        s2 = {
            n.key + "_score": batch[n.key + "_score"].clone()
            for n in sde_noisers
        }

        # --- Step 5: Average scores; restore x_t and time ---
        for key in s1:
            batch[key] = 0.5 * (s1[key] + s2[key])
        # Restore scores for non-SDE noisers to the first evaluation.
        for key, val in s1_other.items():
            batch[key] = val
        # Restore original position and time (pos setter clears graph).
        batch.pos = x_t
        batch.time = t_current

        # --- Step 6: ODE corrector with averaged scores ---
        for noiser in reversed(sde_noisers):
            batch = ProbabilityFlowODESampler._ode_step(batch, noiser, dt)

        # --- Step 7: Deterministic EM step for non-SDE noisers ---
        for noiser in reversed(other_noisers):
            batch = noiser.denoise(batch, dt, last=True)

        batch.wrap_positions()
        batch.update_graph()
        return batch
