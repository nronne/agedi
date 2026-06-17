"""Force-field predictor-corrector sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar, List, Optional

import torch

from .base import Sampler

if TYPE_CHECKING:
    from agedi.data import AtomsGraph
    from agedi.diffusion.noisers import Noiser


class ForcefieldCorrectorSampler(Sampler):
    """EM predictor followed by force-field gradient-descent corrector steps.

    After each standard Euler-Maruyama reverse-SDE step (the predictor), the
    force-field model is evaluated *corrector_steps* times and positions are
    nudged along the predicted force direction.  This drives structures toward
    low-energy configurations at every diffusion step rather than only at the
    end via post-diffusion relaxation.

    Requires a regressor model with a forces head to be attached to the
    diffusion model.  When ``ff_fn`` is ``None`` (no regressor model
    available) the corrector is a no-op and the sampler is equivalent to
    :class:`EulerMaruyamaSampler`.

    Parameters
    ----------
    score_fn : callable
        Score model call: ``score_fn(batch) -> batch``.
    noisers : list[Noiser]
        Noisers in forward order; iterated in reverse for denoising.
    ff_fn : callable or None
        Force-field step: ``ff_fn(batch, scale) -> batch``.  Provided by
        :meth:`~agedi.diffusion.Diffusion._resolve_sampler`; ``None`` when
        no regressor model is attached.
    corrector_steps : int
        Number of force-field gradient-descent steps per reverse-SDE step.
        Default is ``1``.
    corrector_scale : float
        Base scale passed to *ff_fn* at each corrector step.  The
        force-field guidance function applies an additional time-dependent
        weight ``(1 - t)**zeta`` on top of this value.  Default is ``0.01``.

    String alias
    ------------
    ``"ffpc"`` — registered in :mod:`agedi.diffusion.samplers`.

    Notes
    -----
    Do not combine ``sampler="ffpc"`` with a non-zero ``ff_guidance`` scale
    in :meth:`~agedi.diffusion.Diffusion.sample`.  Both apply the force field,
    so using them together will double the guidance at each step.
    """

    uses_force_field: ClassVar[bool] = True

    def __init__(
        self,
        score_fn: Callable[["AtomsGraph"], "AtomsGraph"],
        noisers: List["Noiser"],
        ff_fn: Optional[Callable[["AtomsGraph", float], "AtomsGraph"]] = None,
        corrector_steps: int = 1,
        corrector_scale: float = 0.01,
    ) -> None:
        super().__init__(score_fn, noisers)
        self.ff_fn = ff_fn
        self.corrector_steps = corrector_steps
        self.corrector_scale = corrector_scale

    def step(
        self,
        batch: "AtomsGraph",
        dt: torch.Tensor,
        last: bool,
    ) -> "AtomsGraph":
        # Predictor: standard Euler-Maruyama reverse-SDE step
        batch = self.score_fn(batch)
        for noiser in self.noisers[::-1]:
            batch = noiser.denoise(batch, dt, last=last)
        batch.wrap_positions()
        batch.update_graph()

        # Corrector: force-field gradient descent at the predicted positions
        if self.ff_fn is not None and self.corrector_steps > 0:
            for _ in range(self.corrector_steps):
                batch = self.ff_fn(batch, self.corrector_scale)
                batch.wrap_positions()
                batch.update_graph()

        return batch
