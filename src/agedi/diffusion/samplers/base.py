"""Abstract base class for reverse-diffusion samplers."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Callable, ClassVar, Dict, List

import torch

if TYPE_CHECKING:
    from agedi.data import AtomsGraph
    from agedi.diffusion.noisers import Noiser


class Sampler(abc.ABC):
    """Abstract base class for reverse-diffusion samplers.

    A sampler encapsulates one complete reverse-diffusion outer step, from
    time ``t`` to ``t - dt``.  It may call the score model (and optionally
    the force-field model) any number of times within that outer step —
    enabling predictor-corrector schemes, second-order methods, and
    deterministic ODE integration.

    The outer loop in :meth:`~agedi.diffusion.Diffusion._sample_batch` sets
    ``batch.time`` before each call to :meth:`step` and handles trajectory
    saving, force-field guidance, and post-diffusion relaxation.  The sampler
    is responsible only for the diffusion update itself.

    Parameters
    ----------
    score_fn : callable
        A function ``score_fn(batch) -> batch`` that runs the score model and
        stores per-key scores in ``batch.{key}_score``.
    noisers : list of Noiser
        The noisers attached to the diffusion model, in forward order.  The
        sampler iterates them in reverse when denoising.

    Registry
    --------
    Use :meth:`register` to make a sampler accessible by string name so that
    it can be selected via ``Diffusion.sample(sampler="name")``.

    Built-in aliases (populated in ``samplers/__init__.py``):

    * ``"em"``       — :class:`~agedi.diffusion.samplers.EulerMaruyamaSampler`
    * ``"pc"``       — :class:`~agedi.diffusion.samplers.PredictorCorrectorSampler`
    * ``"heun"``     — :class:`~agedi.diffusion.samplers.HeunSampler`
    * ``"ddim"``     — :class:`~agedi.diffusion.samplers.ProbabilityFlowODESampler`
    * ``"heun_ode"`` — :class:`~agedi.diffusion.samplers.HeunODESampler`
    * ``"ffpc"``     — :class:`~agedi.diffusion.samplers.ForcefieldCorrectorSampler`
    """

    _registry: ClassVar[Dict[str, Callable[..., "Sampler"]]] = {}

    #: Set to ``True`` in subclasses that call a force-field model internally
    #: (e.g. :class:`~agedi.diffusion.samplers.ForcefieldCorrectorSampler`).
    #: ``_sample_batch`` uses this flag to initialise the L-BFGS step sizer.
    uses_force_field: ClassVar[bool] = False

    def __init__(
        self,
        score_fn: Callable[["AtomsGraph"], "AtomsGraph"],
        noisers: List["Noiser"],
    ) -> None:
        self.score_fn = score_fn
        self.noisers = noisers

    @classmethod
    def register(cls, name: str, factory: Callable[..., "Sampler"]) -> None:
        """Register a sampler factory under a string alias.

        The factory is called with ``score_fn`` and ``noisers`` as keyword
        arguments and may accept additional sampler-specific keyword arguments
        (e.g. ``corrector_steps``).

        Parameters
        ----------
        name : str
            Alias used to look up this sampler, e.g. ``"heun"``.
        factory : callable
            ``factory(score_fn, noisers, **kwargs) -> Sampler``.

        Examples
        --------
        Register a custom sampler so it can be selected by name::

            from agedi.diffusion.samplers import Sampler

            class MySampler(Sampler):
                ...

            Sampler.register("my_sampler",
                lambda score_fn, noisers, **kw: MySampler(score_fn, noisers))
        """
        cls._registry[name] = factory

    @abc.abstractmethod
    def step(
        self,
        batch: "AtomsGraph",
        dt: torch.Tensor,
        last: bool,
    ) -> "AtomsGraph":
        """Perform one complete reverse-diffusion step from ``t`` to ``t - dt``.

        ``batch.time`` is set to the current time ``t`` by ``_sample_batch``
        before this method is called.  The implementation must return a batch
        with a valid neighbour list — i.e. it must call
        ``batch.wrap_positions()`` and ``batch.update_graph()`` (or equivalent)
        before returning.

        Parameters
        ----------
        batch : AtomsGraph
            Current state of the batch.  ``batch.time`` is the current time.
        dt : torch.Tensor
            Positive step size: ``dt = t_i - t_{i+1}``.
        last : bool
            Whether this is the final reverse-diffusion step.  When ``True``,
            stochastic samplers suppress the noise term so the final sample is
            deterministic (matches the convention in the underlying noisers).

        Returns
        -------
        AtomsGraph
            Updated batch with a valid neighbour list.
        """
