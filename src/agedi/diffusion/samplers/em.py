"""Euler-Maruyama sampler — one score call per reverse step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

import torch

from .base import Sampler

if TYPE_CHECKING:
    from agedi.data import AtomsGraph
    from agedi.diffusion.noisers import Noiser


class EulerMaruyamaSampler(Sampler):
    """Standard Euler-Maruyama reverse-SDE sampler.

    Performs one score-model evaluation per reverse step and delegates the
    position update to each noiser's :meth:`~agedi.diffusion.noisers.Noiser.denoise`
    method.  The update formula used (EM or DDPM posterior mean) is controlled
    by the ``sampler`` attribute of each :class:`~agedi.diffusion.noisers.PositionsNoiser`.

    This is the default sampler and exactly reproduces the behaviour of
    :meth:`~agedi.diffusion.Diffusion.reverse_step` (minus guidance and timings).

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
        """Euler-Maruyama reverse step.

        1. Evaluate score model.
        2. Apply each noiser's denoising update in reverse order.
        3. Wrap positions and rebuild the neighbour list.
        """
        batch = self.score_fn(batch)
        for noiser in self.noisers[::-1]:
            batch = noiser.denoise(batch, dt, last=last)
        batch.wrap_positions()
        self._check_finite(batch, "EM denoising step")
        batch.update_graph()
        return batch
