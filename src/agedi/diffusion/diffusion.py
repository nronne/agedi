"""Diffusion: pure sampling logic without Lightning dependency.

This module provides :class:`Diffusion`, a plain Python class that
holds the score model, noisers, and an optional regressor model and exposes
the full sampling pipeline --- including predictor-corrector sampling.

It is designed to be used standalone (e.g. for inference) or as a mixin base
for :class:`~agedi.diffusion.Agedi` (the Lightning training wrapper).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser

from .guidance import (
    BatchedLBFGSStepSizer,
    ForcefieldGuidanceConfig,
    force_field_guidance_step,
    max_force_per_graph,
    post_diffusion_relaxation_step,
)
from .novelty import FeatureArchive, NoveltyGuidanceConfig, novelty_guidance_step


@dataclasses.dataclass
class SamplingTimings:
    initialization: float = 0.0
    batch_setup: float = 0.0
    initial_neighbor_list: float = 0.0
    score_model: float = 0.0
    denoise: float = 0.0
    wrap_positions: float = 0.0
    neighbor_list: float = 0.0
    force_field_guidance: float = 0.0
    novelty_guidance: float = 0.0
    guidance_wrap_positions: float = 0.0
    guidance_neighbor_list: float = 0.0
    post_diffusion_force_eval: float = 0.0
    post_diffusion_relaxation: float = 0.0
    post_diffusion_wrap_positions: float = 0.0
    post_diffusion_neighbor_list: float = 0.0
    post_diffusion_relaxation_force_eval: float = 0.0
    total_wall: float = 0.0
    reverse_step_calls: int = 0
    score_model_calls: int = 0
    force_field_calls: int = 0
    novelty_guidance_calls: int = 0
    neighbor_list_calls: int = 0
    neighbor_list_rebuilds: int = 0
    guidance_neighbor_list_calls: int = 0
    guidance_neighbor_list_rebuilds: int = 0
    post_diffusion_relaxation_steps: int = 0
    post_diffusion_neighbor_list_calls: int = 0
    post_diffusion_neighbor_list_rebuilds: int = 0

    @property
    def total_neighbor_list(self) -> float:
        return (
            self.initial_neighbor_list
            + self.neighbor_list
            + self.guidance_neighbor_list
            + self.post_diffusion_neighbor_list
        )

class Diffusion:
    """Pure-Python sampling core for diffusion models.

    Holds the score model, noisers, and an optional regressor and provides
    the full forward / reverse / sampling pipeline.  This class does **not**
    inherit from :class:`torch.nn.Module` or
    :class:`lightning.LightningModule` and therefore has no training hooks.

    When used through :class:`~agedi.diffusion.Agedi` (which inherits
    from both this class and :class:`lightning.LightningModule`), the
    Lightning infrastructure manages device placement and module registration.
    When used standalone, device information is derived from the score model's
    parameters via the :attr:`device` property.

    Parameters
    ----------
    score_model : ScoreModel
        The score model.
    noisers : List[Noiser]
        A list of noisers.
    regressor_model : torch.nn.Module, optional
        An optional regressor model used for force-field guidance during
        sampling.
    eps : float, optional
        Minimum value for the diffusion time step (used in
        :meth:`sample_time`).
    """

    def __init__(
        self,
        score_model: "ScoreModel",
        noisers: List[Noiser],
        regressor_model: Optional["torch.nn.Module"] = None,
        eps: float = 1e-5,
    ) -> None:
        self.score_model = score_model
        self.noisers = noisers
        self.regressor_model = regressor_model
        self.eps = eps
        self.lbfgs_step_sizer: Optional[BatchedLBFGSStepSizer] = None
        self.zeta: float = 3.0

        self.noiser_keys = [noiser.key for noiser in noisers]
        self.score_keys = [head.key for head in score_model.heads]

        if not set(self.noiser_keys) == set(self.score_keys):
            raise ValueError("Keys of noisers and score model heads do not match")

        # Lazily-compiled reverse step; populated on first access of the
        # compiled_reverse_step property.  Cached per-instance so that two
        # Diffusion objects with different architectures do not share a
        # compiled kernel.
        self._compiled_reverse_step = None

    @property
    def device(self) -> torch.device:
        """Infer the computation device from the score model's parameters.

        When used through :class:`~agedi.diffusion.Agedi` (which also
        inherits :class:`lightning.LightningModule`), Lightning's own
        ``device`` property takes precedence.
        """
        try:
            return next(self.score_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # Core forward / reverse steps
    # ------------------------------------------------------------------

    def sample_time(self, batch: AtomsGraph) -> None:
        """Sample a random diffusion time for each graph in *batch*.

        Draws times uniformly from ``[eps, 1]`` and assigns them to
        ``batch.time`` at atom resolution.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data; modified in-place.
        """
        batch_size = batch.batch_size
        time = torch.rand(batch_size) * (1.0 - self.eps) + self.eps
        batch.time = time.to(self.device)[batch.batch].unsqueeze(1)

    def forward_step(self, batch: AtomsGraph) -> AtomsGraph:
        """Forward diffusion step (corruption).

        Applies each noiser in order to corrupt the batch.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.

        Returns
        -------
        AtomsGraph
            The corrupted batch.
        """
        for noiser in self.noisers:
            batch = noiser.noise(batch)

        batch.update_graph()
        return batch

    def reverse_step(
        self,
        batch: AtomsGraph,
        delta_t: float,
        force_field_guidance: float,
        last: bool = False,
        timings: Optional[SamplingTimings] = None,
    ) -> AtomsGraph:
        """Reverse diffusion step (denoising).

        Evaluates the score model and applies one reverse-SDE step through
        all noisers.  Optionally applies force-field guidance afterwards.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.
        delta_t : float
            The time step.
        force_field_guidance : float
            Scale of the force-field guidance (``0.0`` disables it).
        last : bool, optional
            Whether this is the final denoising step.
        timings : SamplingTimings, optional
            If provided, timing measurements are accumulated here.

        Returns
        -------
        AtomsGraph
            The denoised batch.
        """
        if timings is not None:
            timings.reverse_step_calls += 1
            batch = self._time_sampling_call(
                batch.pos.device, timings, "score_model", self.score_model, batch
            )
        else:
            batch = self.score_model(batch)

        for noiser in self.noisers[::-1]:
            if timings is None:
                batch = noiser.denoise(batch, delta_t, last=last)
            else:
                batch = self._time_sampling_call(
                    batch.pos.device,
                    timings,
                    "denoise",
                    noiser.denoise,
                    batch,
                    delta_t,
                    last=last,
                )

        if timings is None:
            batch.wrap_positions()
            batch.update_graph()
        else:
            self._time_sampling_call(
                batch.pos.device, timings, "wrap_positions", batch.wrap_positions
            )
            rebuilt = self._time_sampling_call(
                batch.pos.device, timings, "neighbor_list", batch.update_graph
            )
            timings.neighbor_list_calls += 1
            if rebuilt:
                timings.neighbor_list_rebuilds += 1

        if self.regressor_model is not None and force_field_guidance > 0.0:
            if timings is None:
                batch = self.force_field_guidance_step(
                    batch, force_field_guidance * delta_t
                )
                batch.wrap_positions()
                batch.update_graph()
            else:
                batch = self._time_sampling_call(
                    batch.pos.device,
                    timings,
                    "force_field_guidance",
                    self.force_field_guidance_step,
                    batch,
                    force_field_guidance * delta_t,
                )
                self._time_sampling_call(
                    batch.pos.device,
                    timings,
                    "guidance_wrap_positions",
                    batch.wrap_positions,
                )
                guidance_rebuilt = self._time_sampling_call(
                    batch.pos.device,
                    timings,
                    "guidance_neighbor_list",
                    batch.update_graph,
                )
                timings.guidance_neighbor_list_calls += 1
                if guidance_rebuilt:
                    timings.guidance_neighbor_list_rebuilds += 1

        return batch

    # ------------------------------------------------------------------
    # Sampler resolution
    # ------------------------------------------------------------------

    def _resolve_sampler(
        self,
        sampler: "Optional[Union[str, Sampler]]",
        corrector_steps: int = 0,
        corrector_step_size: float = 1e-3,
        sampler_kwargs: "Optional[Dict]" = None,
    ) -> "Sampler":
        """Resolve a sampler string/instance or build one from legacy params.

        Parameters
        ----------
        sampler : str, Sampler, or None
            ``None`` → :class:`~agedi.diffusion.samplers.EulerMaruyamaSampler`
            (or :class:`~agedi.diffusion.samplers.PredictorCorrectorSampler`
            when *corrector_steps* > 0).
            A string looks up the sampler in the registry.
            A :class:`~agedi.diffusion.samplers.Sampler` instance is returned
            as-is.
        corrector_steps : int, optional
            Used when *sampler* is ``None`` and *corrector_steps* > 0 to
            build a :class:`~agedi.diffusion.samplers.PredictorCorrectorSampler`.
        corrector_step_size : float, optional
            Step size forwarded to the predictor-corrector sampler.
        sampler_kwargs : dict, optional
            Extra keyword arguments forwarded to the sampler constructor when
            *sampler* is a string alias.  Keys override the defaults supplied
            by *corrector_steps* / *corrector_step_size*.

        Returns
        -------
        Sampler
            A ready-to-use sampler instance.
        """
        from agedi.diffusion.samplers import (
            Sampler as _Sampler,
            EulerMaruyamaSampler,
            PredictorCorrectorSampler,
        )

        if sampler is None:
            if corrector_steps > 0:
                return PredictorCorrectorSampler(
                    self.score_model,
                    self.noisers,
                    corrector_steps=corrector_steps,
                    corrector_step_size=corrector_step_size,
                )
            return EulerMaruyamaSampler(self.score_model, self.noisers)

        if isinstance(sampler, str):
            if sampler not in _Sampler._registry:
                raise ValueError(
                    f"Unknown sampler {sampler!r}. "
                    f"Available: {sorted(_Sampler._registry)}"
                )
            # Don't forward the legacy corrector defaults — they apply only to
            # the sampler=None path.  Each registry factory defines its own
            # sensible defaults; sampler_kwargs lets callers override them.
            merged = dict(sampler_kwargs) if sampler_kwargs else {}
            if merged:
                import inspect as _inspect
                _auto = {"score_fn", "noisers", "regressor_fn"}
                _factory_sig = _inspect.signature(_Sampler._registry[sampler])
                _valid = {
                    name
                    for name, param in _factory_sig.parameters.items()
                    if name not in _auto
                    and param.kind != _inspect.Parameter.VAR_KEYWORD
                }
                _unknown = set(merged) - _valid
                if _unknown:
                    raise ValueError(
                        f"Unknown sampler_kwargs for sampler {sampler!r}: "
                        f"{sorted(_unknown)}. "
                        f"Valid options: {sorted(_valid)}"
                    )
            return _Sampler._registry[sampler](
                score_fn=self.score_model,
                noisers=self.noisers,
                regressor_fn=self.regressor_model,
                **merged,
            )

        if isinstance(sampler, _Sampler):
            return sampler

        raise TypeError(
            f"sampler must be a str, Sampler instance, or None; "
            f"got {type(sampler)!r}"
        )

    def corrector_step(
        self,
        batch: AtomsGraph,
        corrector_dt: float,
    ) -> AtomsGraph:
        """Langevin corrector step at constant time.

        Evaluates the score model and applies one Langevin corrector step
        through all noisers (in reverse order).

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.
        corrector_dt : float
            Step size for the Langevin corrector.

        Returns
        -------
        AtomsGraph
            The corrected batch.
        """
        batch = self.score_model(batch)
        for noiser in self.noisers[::-1]:
            batch = noiser.langevin_step(batch, corrector_dt)
        batch.wrap_positions()
        batch.update_graph()
        return batch

    # ------------------------------------------------------------------
    # Guidance helpers (thin wrappers around module-level functions)
    # ------------------------------------------------------------------

    def force_field_guidance_step(
        self,
        batch: AtomsGraph,
        scale: float,
        max_step_size: float = 0.1,
    ) -> AtomsGraph:
        """Apply one force-field guidance step.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.
        scale : float
            Base scale of the force field guidance.
        max_step_size : float, optional
            Maximum allowed step size magnitude.

        Returns
        -------
        AtomsGraph
            Updated batch.
        """
        return force_field_guidance_step(
            batch,
            self.regressor_model,
            self.lbfgs_step_sizer,
            scale=scale,
            zeta=self.zeta,
            max_step_size=max_step_size,
        )

    def post_diffusion_relaxation_step(
        self,
        batch: AtomsGraph,
        scale: float = 1.0,
        max_step_size: float = 0.2,
        forces: Optional[torch.Tensor] = None,
        active: Optional[torch.Tensor] = None,
    ) -> AtomsGraph:
        """Perform one L-BFGS relaxation step, as ``ase.optimize.LBFGS`` would.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.
        scale : float, optional
            Multiplier on the computed step (ASE's ``damping``).  Defaults to
            ``1.0``: take the full L-BFGS step.
        max_step_size : float, optional
            Maximum single-atom displacement per step, in Å.  Defaults to
            ``0.2``, matching ASE.
        forces : torch.Tensor, optional
            Forces at the current positions.  Supplying them skips the
            regressor call inside the step.
        active : torch.Tensor, optional
            Boolean mask over graphs; structures marked ``False`` are left
            untouched.

        Returns
        -------
        AtomsGraph
            Updated batch.
        """
        return post_diffusion_relaxation_step(
            batch,
            self.regressor_model,
            self.lbfgs_step_sizer,
            scale=scale,
            max_step_size=max_step_size,
            forces=forces,
            active=active,
        )

    # ------------------------------------------------------------------
    # Graph initialisation
    # ------------------------------------------------------------------

    def _initialize_graph(self, cutoff: float, fully_connected: bool = False, **kwargs) -> AtomsGraph:
        """Initialise a single graph from noiser priors.

        Parameters
        ----------
        cutoff : float
            Cutoff radius for the neighbour list.
        fully_connected : bool, optional
            When ``True`` the graph is rebuilt as a fully connected graph at
            every reverse step instead of using a finite cutoff.  Recommended
            for gas-phase molecules and clusters.  Defaults to ``False``.
        **kwargs
            Additional keyword arguments passed to the graph (e.g. ``cell``,
            ``template``, ``pbc``).

        Returns
        -------
        AtomsGraph
            The initialised graph.
        """
        graph = AtomsGraph.empty(cutoff=cutoff, fully_connected=fully_connected)
        if "template" in kwargs:
            template = kwargs.pop("template")
        else:
            template = None

        if "cell" in kwargs:
            cell = kwargs.pop("cell")
            setattr(graph, "cell", cell)

        # Pop pbc explicitly so it can be applied to new_graph after creation
        # (in both template and non-template branches).
        pbc = kwargs.pop("pbc", None)

        for k, v in kwargs.items():
            setattr(graph, k, v)

        for noiser in self.noisers[::-1]:
            noiser.initialize_graph(graph)

        if template is not None:
            new_graph = template.clone()

            setattr(
                new_graph,
                "x",
                torch.cat([template.x, graph.x]),
            )

            setattr(
                new_graph,
                "pos",
                torch.cat([template.pos, graph.pos]),
            )

            setattr(
                new_graph,
                "mask",
                torch.cat([
                    torch.ones_like(template.x, dtype=torch.bool),
                    torch.zeros_like(graph.x, dtype=torch.bool),
                ]),
            )

            setattr(new_graph, "n_atoms", template.n_atoms + graph.n_atoms)

            # Apply explicit pbc, overriding what was cloned from the template.
            if pbc is not None:
                setattr(new_graph, "pbc", pbc)
        else:
            new_graph = graph
            setattr(new_graph, "mask", torch.zeros_like(graph.x, dtype=torch.bool))
            if pbc is not None:
                setattr(new_graph, "pbc", pbc)

        return new_graph

    @staticmethod
    def _expand_mask_like(mask: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        """Broadcast a per-atom bool mask to the trailing shape of *tensor*."""
        if tensor.dim() > mask.dim():
            shape = [mask.shape[0]] + [1] * (tensor.dim() - 1)
            return mask.view(*shape)
        return mask

    def _initialize_inpaint_graph(
        self,
        source: AtomsGraph,
        inpaint_mask: torch.Tensor,
        freeze_mask: Optional[torch.Tensor],
        t_start: float,
    ) -> AtomsGraph:
        """Build one inpainting starting graph from an existing structure.

        Clones *source*, records its clean state as the ``{key}0`` reference
        for every noiser (used by :meth:`~agedi.diffusion.noisers.Noiser.forward_marginal`
        throughout sampling), then draws the initial noised state:

        * ``t_start == 1.0``: atoms selected by *inpaint_mask* are drawn from
          each noiser's prior (matching ordinary from-scratch sampling);
          atoms not selected are drawn from the forward marginal
          ``q(z_1 | z_0)`` instead of the prior, since their identity is known.
        * ``t_start < 1.0``: every atom (selected or not) is drawn from the
          forward marginal ``q(z_{t_start} | z_0)``. The distinction between
          selected and known atoms only takes effect once reverse diffusion
          starts regenerating the selected ones.

        Parameters
        ----------
        source : AtomsGraph
            The input structure (unbatched), already carrying ``cutoff``.
        inpaint_mask : torch.Tensor
            Bool tensor, ``True`` for atoms to regenerate.
        freeze_mask : torch.Tensor, optional
            Bool tensor, ``True`` for atoms to hard-freeze (never move),
            stored as ``mask``. Must be disjoint from *inpaint_mask*.
        t_start : float
            Starting diffusion time.

        Returns
        -------
        AtomsGraph
            The initialised graph, not yet batched or graph-built.

        """
        graph = source.clone()
        device = graph.pos.device

        inpaint_mask = inpaint_mask.to(device=device, dtype=torch.bool)
        if freeze_mask is None:
            freeze_mask = torch.zeros_like(inpaint_mask)
        else:
            freeze_mask = freeze_mask.to(device=device, dtype=torch.bool)

        setattr(graph, "inpaint_mask", inpaint_mask)
        setattr(graph, "mask", freeze_mask)

        t = torch.full(
            (graph.pos.shape[0], 1), float(t_start), device=device, dtype=graph.pos.dtype
        )
        graph.time = t

        for noiser in self.noisers:
            key = noiser.key
            ref = graph[key].clone()
            graph.add_batch_attr(key + "0", ref, type="node")

            marginal = noiser.forward_marginal(graph, ref)
            if t_start >= 1.0:
                prior = noiser.prior.get_callable(graph)()
                select = self._expand_mask_like(inpaint_mask, prior)
                new_val = torch.where(select, prior, marginal)
            else:
                new_val = marginal

            setattr(graph, key, new_val)

        return graph

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sync_for_timing(device: Optional[torch.device]) -> None:
        if device is None or device.type != "cuda" or not torch.cuda.is_available():
            return
        torch.cuda.synchronize(device)

    def _time_sampling_call(
        self,
        device: Optional[torch.device],
        timings: SamplingTimings,
        key: str,
        fn,
        *args,
        **kwargs,
    ):
        self._sync_for_timing(device)
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        self._sync_for_timing(device)
        setattr(timings, key, getattr(timings, key) + (time.perf_counter() - start))
        return result

    @staticmethod
    def _format_timing_line(
        label: str, value: float, count: Optional[int] = None
    ) -> str:
        if count is None or count == 0:
            return f"  {label}: {value:.3f}s"
        return f"  {label}: {value:.3f}s ({value / count:.3f}s/call over {count} calls)"

    def _print_sampling_timings(self, timings: SamplingTimings) -> None:
        print("Sampling timing breakdown:")
        print(self._format_timing_line("graph initialization", timings.initialization))
        print(self._format_timing_line("batch setup", timings.batch_setup))
        print(
            self._format_timing_line(
                "initial neighbor list", timings.initial_neighbor_list, 1
            )
        )
        print(
            self._format_timing_line(
                "score model", timings.score_model, timings.score_model_calls
            )
        )
        print(
            self._format_timing_line(
                "denoise steps", timings.denoise, timings.reverse_step_calls
            )
        )
        print(
            self._format_timing_line(
                "wrap positions", timings.wrap_positions, timings.reverse_step_calls
            )
        )
        print(
            self._format_timing_line(
                "neighbor list updates",
                timings.neighbor_list,
                timings.neighbor_list_calls,
            )
        )
        if (
            timings.force_field_calls > 0
            or timings.force_field_guidance > 0
            or timings.guidance_neighbor_list > 0
        ):
            print(
                self._format_timing_line(
                    "force-field guidance",
                    timings.force_field_guidance,
                    timings.force_field_calls,
                )
            )
            print(
                self._format_timing_line(
                    "guidance wrap positions",
                    timings.guidance_wrap_positions,
                    timings.guidance_neighbor_list_calls,
                )
            )
            print(
                self._format_timing_line(
                    "guidance neighbor list updates",
                    timings.guidance_neighbor_list,
                    timings.guidance_neighbor_list_calls,
                )
            )
        if timings.novelty_guidance_calls > 0:
            print(
                self._format_timing_line(
                    "novelty guidance",
                    timings.novelty_guidance,
                    timings.novelty_guidance_calls,
                )
            )
        if timings.post_diffusion_force_eval > 0:
            print(
                self._format_timing_line(
                    "post-diffusion force eval", timings.post_diffusion_force_eval, 1
                )
            )
        if timings.post_diffusion_relaxation > 0:
            print(
                self._format_timing_line(
                    "post-diffusion relaxation",
                    timings.post_diffusion_relaxation,
                    timings.post_diffusion_relaxation_steps,
                )
            )
        if timings.post_diffusion_neighbor_list > 0:
            print(
                self._format_timing_line(
                    "post-relaxation neighbor list updates",
                    timings.post_diffusion_neighbor_list,
                    timings.post_diffusion_neighbor_list_calls,
                )
            )
        if timings.post_diffusion_relaxation_force_eval > 0:
            print(
                self._format_timing_line(
                    "post-relaxation force eval",
                    timings.post_diffusion_relaxation_force_eval,
                    timings.post_diffusion_relaxation_steps,
                )
            )
        print(
            self._format_timing_line(
                "total neighbor list",
                timings.total_neighbor_list,
                1
                + timings.neighbor_list_calls
                + timings.guidance_neighbor_list_calls
                + timings.post_diffusion_neighbor_list_calls,
            )
        )
        print(self._format_timing_line("total wall", timings.total_wall))

    # ------------------------------------------------------------------
    # Compiled reverse step
    # ------------------------------------------------------------------

    @property
    def compiled_reverse_step(self):
        """Lazily compile :meth:`reverse_step` with ``torch.compile``.

        The compiled kernel is cached as ``self._compiled_reverse_step`` so
        that compilation happens at most once per model instance.  Using a
        per-instance cache (rather than a class-level ``@torch.compile``
        decorator) means that two :class:`Diffusion` objects with different
        architectures will each compile their own kernel and never interfere.

        .. note::
            ``timings`` must **not** be passed to the compiled function —
            ``time.perf_counter`` is not traceable by Dynamo.  Time the
            compiled call from outside in :meth:`_sample_batch` using the
            ``is_compiled`` flag.
        """
        if self._compiled_reverse_step is None:
            def _compiled_fn(batch, delta_t, force_field_guidance, last=False):
                # timings must NOT be passed here --- time.perf_counter is
                # untraceable by Dynamo.  Timing is handled externally.
                return self.reverse_step(
                    batch, delta_t, force_field_guidance, last=last, timings=None
                )
            self._compiled_reverse_step = torch.compile(_compiled_fn, mode="default")
        return self._compiled_reverse_step

    # ------------------------------------------------------------------
    # Internal sampling loop
    # ------------------------------------------------------------------
    def _sample_batch(
        self,
        batch: Batch,
        steps: int,
        eps: float,
        force_field_guidance: float,
        save_trajectory: bool,
        progress_bar: bool,
        force_threshold: float,
        max_extra_steps: int,
        corrector_steps: int = 0,
        corrector_step_size: float = 1e-3,
        timings: Optional[SamplingTimings] = None,
        reverse_step_fn=None,
        is_compiled: bool = False,
        sampler=None,
        sampler_kwargs=None,
        novelty_guidance: Optional[NoveltyGuidanceConfig] = None,
        novelty_archive: Optional[FeatureArchive] = None,
        save_corrector_frames: bool = False,
        t_start: float = 1.0,
    ) -> List[AtomsGraph]:
        """Run the reverse-diffusion loop for a pre-built batch.

        Parameters
        ----------
        batch : Batch
            A batch of :class:`~agedi.data.AtomsGraph` data at ``t=1``.
        steps : int
            Number of reverse-diffusion steps.
        eps : float
            Minimum time value (end of trajectory).
        force_field_guidance : float
            Scale of the force-field guidance (``0.0`` disables it).
        save_trajectory : bool
            Whether to collect and return all intermediate states.
        progress_bar : bool
            Whether to display a tqdm progress bar.
        force_threshold : float
            Maximum per-atom force for terminating post-diffusion relaxation.
        max_extra_steps : int
            Maximum extra relaxation steps after the main trajectory.
        corrector_steps : int, optional
            Number of Langevin corrector passes after each predictor step.
            ``0`` (default) disables the corrector (standard DDPM/EM sampling).
        corrector_step_size : float, optional
            Step size used for each Langevin corrector step.  Defaults to
            ``1e-3``.
        timings : SamplingTimings, optional
            If provided, timing measurements are accumulated here.
        reverse_step_fn : callable, optional
            The reverse step function to use.  Defaults to
            ``self.reverse_step``.  Pass a ``torch.compile``-wrapped
            version to enable compiled sampling.
        is_compiled : bool, optional
            Whether ``reverse_step_fn`` is a compiled function.
        sampler : str, Sampler, or None, optional
            Sampler instance or string alias controlling the reverse-diffusion
            algorithm.  When provided (and *is_compiled* is ``False``), the
            sampler's :meth:`~agedi.diffusion.samplers.Sampler.step` is called
            instead of *reverse_step_fn*.  ``None`` (default) falls back to an
            :class:`~agedi.diffusion.samplers.EulerMaruyamaSampler` (or a
            :class:`~agedi.diffusion.samplers.PredictorCorrectorSampler` when
            *corrector_steps* > 0).
        sampler_kwargs : dict, optional
            Extra constructor arguments forwarded to the sampler when *sampler*
            is a string alias.  Keys override the defaults supplied by
            *corrector_steps* / *corrector_step_size*.
        novelty_guidance : NoveltyGuidanceConfig, optional
            Feature-space novelty guidance configuration.  ``None`` (default)
            disables it.  Not supported on the compiled path.
        novelty_archive : FeatureArchive, optional
            Features of already-found structures to repel from.  When ``None``,
            only the in-batch repulsion term contributes.
        save_corrector_frames : bool, optional
            Also record every Langevin corrector sub-step in the saved
            trajectory.  Only meaningful together with *save_trajectory* and a
            sampler that runs correctors (``"pc"`` / ``"ffpc"``).  ``False``
            (default) records one frame per outer diffusion step.
        t_start : float, optional
            Starting diffusion time.  ``1.0`` (default) runs the full reverse
            trajectory.  Values below ``1.0`` start from a partially-noised
            state, e.g. for inpainting-style local refinement via
            :meth:`inpaint`.

        Returns
        -------
        List[AtomsGraph]
            Final structures, or (when *save_trajectory* is ``True``) a list of
            trajectories (one per graph).
        """
        if reverse_step_fn is None:
            reverse_step_fn = self.reverse_step

        novelty_enabled = (
            novelty_guidance is not None and novelty_guidance.guidance != 0.0
        )
        if novelty_enabled and is_compiled:
            raise ValueError(
                "Novelty guidance is not supported with compile=True. It "
                "differentiates the backbone with respect to the atomic "
                "positions, which the compiled reverse step does not expose. "
                "Sample with compile=False to use novelty guidance."
            )

        if steps < 2:
            return batch.to_data_list()

        # Resolve the sampler for the non-compiled path.
        _sampler = None
        if not is_compiled:
            _sampler = self._resolve_sampler(
                sampler, corrector_steps, corrector_step_size, sampler_kwargs
            )
            _sampler.save_corrector_frames = save_trajectory and save_corrector_frames
        elif sampler is not None:
            raise ValueError(
                "compile=True is only compatible with the default Euler-Maruyama / "
                "Predictor-Corrector sampler (sampler=None). "
                f"Got sampler={sampler!r}. Either set compile=False or remove the sampler argument."
            )

        # max_extra_steps is included so that post-diffusion relaxation gets a
        # persistent step sizer even with guidance disabled.  Without one,
        # post_diffusion_relaxation_step builds a throwaway sizer per call and
        # no curvature history ever accumulates.
        needs_lbfgs = (
            (force_field_guidance > 0 or max_extra_steps > 0 or (
                _sampler is not None and _sampler.uses_force_field
            ))
            and self.regressor_model is not None
        )
        if needs_lbfgs:
            self.lbfgs_step_sizer = BatchedLBFGSStepSizer(
                batch_size=batch.batch_size
            )

        ts = torch.linspace(t_start, eps, steps, device=self.device)
        dt = ts[0] - ts[1]

        # Inject call-counting wrappers so timings tracks actual score/ff invocations.
        _orig_score_fn = None
        _orig_regressor_fn = None
        if timings is not None and _sampler is not None:
            _orig_score_fn = _sampler.score_fn

            def _counted_score_fn(batch, _f=_orig_score_fn, _t=timings):
                _t.score_model_calls += 1
                return _f(batch)

            _sampler.score_fn = _counted_score_fn

            from agedi.diffusion.samplers import ForcefieldCorrectorSampler as _FFPC

            if isinstance(_sampler, _FFPC) and _sampler.regressor_fn is not None:
                _orig_regressor_fn = _sampler.regressor_fn

                def _counted_regressor_fn(batch, _f=_orig_regressor_fn, _t=timings):
                    _t.force_field_calls += 1
                    return _f(batch)

                _sampler.regressor_fn = _counted_regressor_fn

        if save_trajectory:
            path = []

        if progress_bar:
            iterator = tqdm(range(steps))
        else:
            iterator = range(steps)

        for i in iterator:
            if save_trajectory:
                path.append(batch.to_data_list())

            batch.add_batch_attr("time", ts[i].repeat(batch.x.shape[0], 1), type="node")
            last_step = i == steps - 1

            if is_compiled:
                # Compiled path: uses the pre-compiled reverse_step_fn.
                # Sampler classes are not compatible with torch.compile.
                if timings is not None:
                    batch = self._time_sampling_call(
                        batch.pos.device,
                        timings,
                        "score_model",
                        reverse_step_fn,
                        batch,
                        dt,
                        force_field_guidance,
                        last=last_step,
                    )
                    timings.reverse_step_calls += 1
                    timings.score_model_calls += 1
                else:
                    batch = reverse_step_fn(
                        batch, dt, force_field_guidance, last=last_step
                    )
            else:
                # Sampler path: the sampler owns the full algorithmic step,
                # including any corrector sub-steps.
                if timings is not None:
                    timings.reverse_step_calls += 1
                    batch = self._time_sampling_call(
                        batch.pos.device,
                        timings,
                        "score_model",
                        _sampler.step,
                        batch,
                        dt,
                        last_step,
                    )
                else:
                    batch = _sampler.step(batch, dt, last_step)

                # Force-field guidance applied after the sampler step,
                # consistent with the current reverse_step() behaviour.
                if self.regressor_model is not None and force_field_guidance > 0.0:
                    if timings is not None:
                        batch = self._time_sampling_call(
                            batch.pos.device,
                            timings,
                            "force_field_guidance",
                            self.force_field_guidance_step,
                            batch,
                            force_field_guidance * dt,
                        )
                        timings.force_field_calls += 1
                        self._time_sampling_call(
                            batch.pos.device,
                            timings,
                            "guidance_wrap_positions",
                            batch.wrap_positions,
                        )
                        guidance_rebuilt = self._time_sampling_call(
                            batch.pos.device,
                            timings,
                            "guidance_neighbor_list",
                            batch.update_graph,
                        )
                        timings.guidance_neighbor_list_calls += 1
                        if guidance_rebuilt:
                            timings.guidance_neighbor_list_rebuilds += 1
                    else:
                        batch = self.force_field_guidance_step(
                            batch, force_field_guidance * dt
                        )
                        batch.wrap_positions()
                        batch.update_graph()

                # Feature-space novelty guidance, applied after the sampler
                # step on the same footing as the force-field guidance above.
                if novelty_enabled:
                    if timings is not None:
                        batch = self._time_sampling_call(
                            batch.pos.device,
                            timings,
                            "novelty_guidance",
                            novelty_guidance_step,
                            batch,
                            self.score_model,
                            novelty_archive,
                            novelty_guidance,
                        )
                        timings.novelty_guidance_calls += 1
                    else:
                        batch = novelty_guidance_step(
                            batch,
                            self.score_model,
                            novelty_archive,
                            novelty_guidance,
                        )
                    batch.wrap_positions()
                    batch.update_graph()

                # Append sub-step frames produced inside sampler.step() —
                # corrector steps when save_corrector_frames is set, and the
                # ffpc terminal dynamics frames on the last diffusion step.
                if save_trajectory:
                    pending = getattr(_sampler, "_pending_frames", None)
                    if pending:
                        path.extend(pending)

        # Terminal dynamics end on the state the sampler returned, so it is
        # already the last pending frame.  Corrector capture deliberately stops
        # one sub-step short, so the final state still needs appending.
        _final_captured = bool(getattr(_sampler, "_pending_includes_final", False))

        # Restore original score_fn / regressor_fn if they were wrapped for counting.
        if _orig_score_fn is not None:
            _sampler.score_fn = _orig_score_fn
        if _orig_regressor_fn is not None:
            _sampler.regressor_fn = _orig_regressor_fn

        # Optional post-diffusion relaxation.  Independent of guidance: asking
        # for relaxation steps is enough to get them, so a clean diffusion
        # trajectory can still be relaxed at the end.  The guidance term is
        # retained in the condition because it also populates
        # ``forces_prediction`` on the returned structures.
        if (
            force_field_guidance > 0 or max_extra_steps > 0
        ) and self.regressor_model is not None:
            # Reset LBFGS memory: history from the noisy diffusion trajectory
            # carries stale curvature information that corrupts relaxation steps.
            if self.lbfgs_step_sizer is not None:
                self.lbfgs_step_sizer.reset()

            if timings is None:
                batch = self.regressor_model(batch)
            else:
                batch = self._time_sampling_call(
                    batch.pos.device,
                    timings,
                    "post_diffusion_force_eval",
                    self.regressor_model,
                    batch,
                )
            # Convergence is tracked per structure: a batch-wide maximum keeps
            # every structure stepping until the worst one is done, jostling
            # the ones that already converged.
            per_graph_forces = max_force_per_graph(
                batch.forces_prediction, batch.batch, batch.num_graphs
            )
            max_forces = per_graph_forces.max()

            if max_forces > force_threshold and max_extra_steps > 0:
                if progress_bar:
                    print(
                        f"Max force after diffusion: {max_forces:.4f}, "
                        "continuing relaxation..."
                    )
                    extra_iterator = tqdm(
                        range(max_extra_steps), desc="Post-diffusion relaxation"
                    )
                else:
                    extra_iterator = range(max_extra_steps)

                batch.add_batch_attr(
                    "time", torch.zeros_like(batch.time), type="node"
                )

                for i in extra_iterator:
                    # Structures already below the threshold sit out the rest
                    # of the relaxation instead of being stepped further.
                    active = per_graph_forces > force_threshold
                    # The forces were evaluated at exactly these positions by
                    # the convergence check (or the initial eval above), so
                    # they are passed in rather than recomputed.
                    forces = batch.forces_prediction

                    # Full L-BFGS step (ASE damping=1.0).  Scaling the step
                    # down here would slow convergence without improving
                    # stability — the maxstep limit is what bounds the step.
                    if timings is None:
                        batch = self.post_diffusion_relaxation_step(
                            batch, forces=forces, active=active
                        )
                    else:
                        batch = self._time_sampling_call(
                            batch.pos.device,
                            timings,
                            "post_diffusion_relaxation",
                            self.post_diffusion_relaxation_step,
                            batch,
                            forces=forces,
                            active=active,
                        )
                        timings.post_diffusion_relaxation_steps += 1

                    if timings is None:
                        batch = self.regressor_model(batch)
                    else:
                        batch = self._time_sampling_call(
                            batch.pos.device,
                            timings,
                            "post_diffusion_relaxation_force_eval",
                            self.regressor_model,
                            batch,
                        )
                    per_graph_forces = max_force_per_graph(
                        batch.forces_prediction, batch.batch, batch.num_graphs
                    )
                    max_forces = per_graph_forces.max()

                    if save_trajectory:
                        path.append(batch.to_data_list())
                        _final_captured = True

                    if max_forces <= force_threshold:
                        if progress_bar:
                            print(
                                f"Relaxation converged after {i+1} steps, "
                                f"max force: {max_forces:.4f}"
                            )
                        break

                if progress_bar and max_forces > force_threshold:
                    print(
                        f"Relaxation did not converge, "
                        f"final max force: {max_forces:.4f}"
                    )

        if save_trajectory:
            if not _final_captured:
                path.append(batch.to_data_list())
            return list(map(list, zip(*path)))

        return batch.to_data_list()

    def _sample(
        self,
        N: int,
        steps: int,
        cutoff: float,
        eps: float,
        force_field_guidance: float,
        force_threshold: float,
        max_extra_steps: int,
        progress_bar: bool,
        save_trajectory: bool,
        corrector_steps: int = 0,
        corrector_step_size: float = 1e-3,
        print_timings: bool = False,
        compile: bool = False,
        sampler=None,
        sampler_kwargs=None,
        novelty_guidance: Optional[NoveltyGuidanceConfig] = None,
        novelty_archive: Optional[FeatureArchive] = None,
        save_corrector_frames: bool = False,
        t_start: float = 1.0,
        graph_factory: Optional[Callable[[], AtomsGraph]] = None,
        **kwargs,
    ) -> List[AtomsGraph]:
        """Build *N* graphs from priors and run the sampling loop.

        Parameters
        ----------
        N : int
            Number of structures to generate.
        steps : int
            Number of reverse-diffusion steps.
        cutoff : float
            Cutoff radius for the neighbour list.
        eps : float
            Minimum time value (end of trajectory).
        force_field_guidance : float
            Scale of the force-field guidance.
        force_threshold : float
            Maximum per-atom force for post-diffusion relaxation.
        max_extra_steps : int
            Maximum extra relaxation steps.
        progress_bar : bool
            Show tqdm progress bar.
        save_trajectory : bool
            Collect all intermediate states.
        corrector_steps : int, optional
            Langevin corrector passes per predictor step.
        corrector_step_size : float, optional
            Step size for each corrector pass.
        print_timings : bool, optional
            Print a timing breakdown after sampling completes.
        compile : bool, optional
            Use ``torch.compile`` on the reverse diffusion step.
        sampler_kwargs : dict, optional
            Extra keyword arguments forwarded to the sampler constructor when
            *sampler* is a string alias.
        t_start : float, optional
            Starting diffusion time, forwarded to :meth:`_sample_batch`.
            Defaults to ``1.0`` (full reverse trajectory).
        graph_factory : callable, optional
            When given, called with no arguments once per structure instead
            of ``self._initialize_graph(cutoff, **kwargs)`` to build the
            initial (unbatched) graph.  Used by :meth:`inpaint` to build
            graphs from an existing structure rather than from noiser priors;
            *kwargs* is ignored when this is provided.
        **kwargs
            Keyword arguments forwarded to :meth:`_initialize_graph`.  Ignored
            when *graph_factory* is given.

        Returns
        -------
        List[AtomsGraph]
            Sampled structures (or trajectories when *save_trajectory* is ``True``).
        """
        timings = SamplingTimings()
        self._sync_for_timing(self.device)
        total_start = time.perf_counter()

        data = []
        init_start = time.perf_counter()
        for _ in range(N):
            if graph_factory is not None:
                data.append(graph_factory())
            else:
                data.append(self._initialize_graph(cutoff, **kwargs))
        timings.initialization += time.perf_counter() - init_start

        batch_setup_start = time.perf_counter()
        batch = Batch.from_data_list(data).to(self.device)
        self._sync_for_timing(batch.pos.device)
        timings.batch_setup += time.perf_counter() - batch_setup_start

        # When torch.compile is requested, estimate cell-list sizes and
        # max_neighbors via NVIDIA nvalchemiops so that all neighbor-list
        # buffers have fixed shapes before the first update_graph() call.
        # Fixed shapes are required to trace the reverse step only once.
        if compile:
            batch.prepare_for_compile(cutoff)

        self._time_sampling_call(
            batch.pos.device,
            timings,
            "initial_neighbor_list",
            batch.update_graph,
        )

        # Optionally compile the reverse step after the first neighbor list
        # has been built (so all buffer shapes are known and fixed).
        reverse_step_fn = self.compiled_reverse_step if compile else self.reverse_step

        out = self._sample_batch(
            batch,
            steps,
            eps,
            force_field_guidance,
            save_trajectory,
            progress_bar,
            force_threshold,
            max_extra_steps,
            corrector_steps=corrector_steps,
            corrector_step_size=corrector_step_size,
            timings=timings,
            reverse_step_fn=reverse_step_fn,
            is_compiled=compile,
            sampler=sampler,
            sampler_kwargs=sampler_kwargs,
            novelty_guidance=novelty_guidance,
            novelty_archive=novelty_archive,
            save_corrector_frames=save_corrector_frames,
            t_start=t_start,
        )
        self._sync_for_timing(batch.pos.device)
        timings.total_wall = time.perf_counter() - total_start
        if print_timings:
            self._print_sampling_timings(timings)
        return out

    # ------------------------------------------------------------------
    # Public sampling API
    # ------------------------------------------------------------------

    def sample(
        self,
        N: int,
        template=None,
        batch_size: Optional[int] = 64,
        steps: Optional[int] = 500,
        cutoff: Optional[float] = 6.0,
        eps: Optional[float] = 1e-3,
        n_atoms: Optional[int] = None,
        atomic_numbers: Optional[List[int]] = None,
        formula: Optional[str] = None,
        positions: Optional[np.ndarray] = None,
        cell: Optional[np.ndarray] = None,
        pbc: Optional[np.ndarray] = None,
        confinement: Optional[Tuple[float, float]] = None,
        compile: bool = False,
        ff_guidance: Optional[ForcefieldGuidanceConfig] = None,
        novelty_guidance: Optional[NoveltyGuidanceConfig] = None,
        novelty_archive: Optional[FeatureArchive] = None,
        property: Optional[Dict] = None,
        progress_bar: Optional[bool] = False,
        save_trajectory: Optional[bool] = False,
        save_corrector_frames: Optional[bool] = False,
        print_timings: Optional[bool] = False,
        corrector_steps: int = 0,
        corrector_step_size: float = 1e-3,
        sampler=None,
        sampler_kwargs=None,
    ) -> List[AtomsGraph]:
        """Sample structures from the diffusion model.

        The minimum required arguments depend on the configured noisers and
        whether a template is provided:

        * ``n_atoms`` -- always required unless derivable from
          ``atomic_numbers`` or ``formula``.
        * ``atomic_numbers`` -- required unless a types-noiser is configured
          (key ``"x"``), or derivable from ``formula``.
        * ``positions`` -- required when no positions-noiser is configured
          (type-only diffusion).
        * ``cell`` -- required for periodic systems when no ``template`` is given.
          Not required when ``pbc=[False, False, False]``.
        * ``pbc`` -- optional; defaults to ``[True, True, True]``.  Pass
          ``[False, False, False]`` for non-periodic systems.

        Parameters
        ----------
        N : int
            Number of structures to generate.
        template : AtomsGraph or ase.Atoms, optional
            Template structure.  ``cell`` and ``pbc`` are taken from the
            template when not explicitly provided.
        batch_size : int, optional
            Internal batch size for splitting large *N*.
        steps : int, optional
            Number of reverse-diffusion steps.
        cutoff : float, optional
            Cutoff radius for the neighbour list.
        eps : float, optional
            Minimum time value at the end of the trajectory.
        n_atoms : int, optional
            Number of atoms per structure.
        atomic_numbers : List[int], optional
            Atomic numbers of the atoms to generate.
        formula : str, optional
            Chemical formula (e.g. ``"H2O"``).
        positions : np.ndarray, optional
            Fixed atom positions (shape ``(n_atoms, 3)``).
        cell : np.ndarray, optional
            Unit-cell matrix (3x3).
        pbc : np.ndarray, optional
            Periodic boundary conditions.
        confinement : Tuple[float, float], optional
            Z-directional confinement ``(z_min, z_max)``.
        compile : bool, optional
            When ``True``, use ``torch.compile`` on the reverse diffusion
            step for improved throughput on CUDA hardware.
        ff_guidance : ForcefieldGuidanceConfig, optional
            Force-field guidance configuration.
        novelty_guidance : NoveltyGuidanceConfig, optional
            Feature-space novelty guidance configuration, which repels samples
            from already-found structures.  ``None`` (default) disables it.
            Incompatible with ``compile=True``.
        novelty_archive : FeatureArchive, optional
            Features of the already-found structures to repel from, built with
            :meth:`~agedi.diffusion.novelty.FeatureArchive.from_structures`.
            Must be rebuilt whenever the score model is retrained, and — when
            sampling on a *template* — built with ``n_template`` set to the
            template's atom count, so the references are pooled over the same
            atoms as the samples.  When ``None``, only the in-batch repulsion
            term contributes.
        property : dict, optional
            Conditioning properties (key -> scalar tensor).
        progress_bar : bool, optional
            Show a tqdm progress bar.
        save_trajectory : bool, optional
            Return full trajectories instead of final structures.  One frame
            per reverse-diffusion step, plus any ``ffpc`` terminal-dynamics
            frames and post-diffusion relaxation frames.
        save_corrector_frames : bool, optional
            Additionally record every Langevin corrector sub-step, giving a
            complete frame-by-frame trajectory.  Requires *save_trajectory* and
            a sampler that runs correctors (``"pc"`` / ``"ffpc"``, or
            ``corrector_steps > 0``).  This multiplies the trajectory length by
            roughly ``corrector_steps``, so it is off by default.
        print_timings : bool, optional
            Print a timing breakdown after sampling completes.
        corrector_steps : int, optional
            Number of Langevin corrector passes after each predictor step.
            ``0`` (default) gives standard (predictor-only) sampling.
            Ignored when *sampler* is provided explicitly.
        corrector_step_size : float, optional
            Step size for each corrector pass.  Defaults to ``1e-3``.
        sampler : str, Sampler, or None, optional
            Reverse-diffusion algorithm.  Pass a string alias or a
            :class:`~agedi.diffusion.samplers.Sampler` instance.

            Available string aliases:

            * ``"em"``       — Euler-Maruyama (default)
            * ``"pc"``       — Predictor-corrector (use with *corrector_steps*)
            * ``"heun"``     — Stochastic Heun, 2nd-order (2 score calls/step)
            * ``"ddim"``     — Deterministic probability-flow ODE (DDIM)
            * ``"heun_ode"`` — Deterministic 2nd-order ODE (Heun on PF-ODE)
            * ``"ffpc"``     — Force-field corrector (use with *sampler_kwargs*)

            When ``None`` (default), uses ``"em"`` (or ``"pc"`` when
            *corrector_steps* > 0).
        sampler_kwargs : dict, optional
            Sampler-specific constructor arguments, forwarded when *sampler*
            is a string alias.  Keys override the top-level *corrector_steps*
            and *corrector_step_size* defaults.  Examples::

                # PC with 3 corrector steps and a custom step size
                model.sample(N, sampler="pc",
                             sampler_kwargs={"corrector_steps": 3,
                                            "corrector_step_size": 5e-4})

                # FFPC with 5 force-field corrector steps
                model.sample(N, sampler="ffpc",
                             sampler_kwargs={"corrector_steps": 5,
                                            "corrector_scale": 0.005})

        Returns
        -------
        List[AtomsGraph]
            Sampled structures, or trajectories when *save_trajectory* is ``True``.
        """
        if ff_guidance is None:
            ff_guidance = ForcefieldGuidanceConfig()

        self.score_model.sample_mode()

        # Convert an ASE Atoms template to AtomsGraph if needed.
        if template is not None:
            from ase import Atoms as _AseAtoms

            if isinstance(template, _AseAtoms):
                template = AtomsGraph.from_atoms(
                    template, cutoff=cutoff, confinement=confinement
                )

        # Derive n_atoms / atomic_numbers from a molecular formula if given.
        if formula is not None:
            from ase import Atoms as _AseAtoms

            _formula_atoms = _AseAtoms(formula)
            if n_atoms is None:
                n_atoms = len(_formula_atoms)
            if atomic_numbers is None and "x" not in self.noiser_keys:
                atomic_numbers = _formula_atoms.get_atomic_numbers().tolist()

        # When a template is provided but no cell is given, borrow the
        # template's cell so noiser priors (e.g. UniformCell) can use it.
        if template is not None and cell is None:
            cell = template.cell.detach().cpu().numpy()

        kwargs: Dict = {}
        # Sampling-control parameters passed explicitly to _sample.
        sample_kwargs: Dict = {
            "progress_bar": progress_bar,
            "save_trajectory": save_trajectory,
            "save_corrector_frames": save_corrector_frames,
            "force_threshold": ff_guidance.force_threshold,
            "max_extra_steps": ff_guidance.max_extra_steps,
            "corrector_steps": corrector_steps,
            "corrector_step_size": corrector_step_size,
            "print_timings": print_timings,
            "compile": compile,
            "sampler": sampler,
            "sampler_kwargs": sampler_kwargs,
            "novelty_guidance": novelty_guidance,
            "novelty_archive": novelty_archive,
        }
        self.zeta = ff_guidance.zeta

        if n_atoms is not None:
            kwargs["n_atoms"] = torch.tensor([n_atoms]).reshape(1, 1)
        if positions is not None:
            kwargs["pos"] = torch.tensor(
                np.array(positions), dtype=torch.float
            ).reshape(-1, 3)
            if "n_atoms" not in kwargs:
                kwargs["n_atoms"] = torch.tensor(
                    [kwargs["pos"].shape[0]]
                ).reshape(1, 1)
        if atomic_numbers is not None:
            kwargs["x"] = torch.tensor(atomic_numbers, dtype=torch.long).reshape(-1)
            if "n_atoms" not in kwargs:
                kwargs["n_atoms"] = torch.tensor([len(atomic_numbers)]).reshape(1, 1)

        if cell is not None:
            kwargs["cell"] = torch.tensor(
                np.array(cell), dtype=torch.float
            ).reshape(3, 3)

        if property is not None:
            for k, v in property.items():
                kwargs[k] = torch.tensor(v, dtype=torch.float)

        fully_connected = getattr(self, "fully_connected", False)
        _pbc_all_false = pbc is not None and not any(pbc)
        _cell_not_needed = _pbc_all_false

        for key in ["pos", "x", "cell", "n_atoms"]:
            if key not in kwargs and key not in self.noiser_keys:
                if key == "pos" and "frac" in self.noiser_keys:
                    continue
                if key == "cell" and _cell_not_needed:
                    continue
                raise ValueError(
                    f"Missing default values for key {key} in kwargs."
                )

        if confinement is not None:
            kwargs["confinement"] = torch.tensor(
                confinement, dtype=torch.float
            ).reshape(1, 2)

        if template is not None:
            kwargs["template"] = template
        else:
            n_atoms = kwargs["n_atoms"].item()

        if pbc is not None:
            kwargs["pbc"] = torch.tensor(pbc, dtype=torch.bool).reshape(3)

        if fully_connected:
            kwargs["fully_connected"] = True

        if N > batch_size:
            from rich.console import Console as _Console
            _console = _Console()
            n_full = N // batch_size
            n_remainder = N % batch_size
            n_batches = n_full + (1 if n_remainder > 0 else 0)
            out = []
            for i in range(n_full):
                _console.print(f"Sampling batch {i + 1}/{n_batches}...")
                out += self._sample(
                    batch_size, steps, cutoff, eps, ff_guidance.guidance,
                    **sample_kwargs, **kwargs,
                )
            if n_remainder > 0:
                _console.print(f"Sampling batch {n_batches}/{n_batches}...")
                out += self._sample(
                    n_remainder, steps, cutoff, eps, ff_guidance.guidance,
                    **sample_kwargs, **kwargs,
                )
            return out
        else:
            return self._sample(
                N, steps, cutoff, eps, ff_guidance.guidance,
                **sample_kwargs, **kwargs,
            )

    # ------------------------------------------------------------------
    # Public inpainting API
    # ------------------------------------------------------------------

    def inpaint(
        self,
        structure: AtomsGraph,
        inpaint_mask,
        N: int = 1,
        batch_size: int = 64,
        steps: int = 500,
        eps: float = 1e-3,
        t_start: float = 1.0,
        freeze=None,
        n_resample: int = 1,
        jump_length: int = 1,
        compile: bool = False,
        ff_guidance: Optional[ForcefieldGuidanceConfig] = None,
        property: Optional[Dict] = None,
        progress_bar: bool = False,
        save_trajectory: bool = False,
        save_corrector_frames: bool = False,
        print_timings: bool = False,
        sampler=None,
        sampler_kwargs=None,
    ) -> List[AtomsGraph]:
        """Regenerate a chosen subset of atoms in an existing structure.

        Runs masked reverse diffusion ("inpainting"): atoms selected by
        *inpaint_mask* are regenerated from noise (or from a partially-noised
        state when *t_start* < 1), while every other atom is, at each reverse
        step, replaced by a fresh sample of the forward process ``q(z_t |
        z_0)`` of *structure* — so the whole batch always sits at a
        self-consistent noise level for the score model, and the non-selected
        atoms converge back onto their input positions (and, for the
        atomic-type noiser, their input species) exactly.

        Parameters
        ----------
        structure : AtomsGraph
            The input structure (unbatched), e.g. from
            :meth:`~agedi.data.AtomsGraph.from_atoms`.
        inpaint_mask : array-like of bool, shape (n_atoms,)
            ``True`` for atoms to regenerate.
        N : int, optional
            Number of independent inpainted structures to generate. Defaults
            to ``1``.
        batch_size : int, optional
            Maximum number of structures sampled in one batch; larger *N* is
            chunked. Defaults to ``64``.
        steps : int, optional
            Number of reverse-diffusion steps. Defaults to ``500``.
        eps : float, optional
            Minimum time value (end of trajectory). Defaults to ``1e-3``.
        t_start : float, optional
            Starting diffusion time. ``1.0`` (default) fully re-noises the
            selected atoms (de-novo generation of that region). Values below
            ``1.0`` start from a partially-noised state for a local
            rattle-and-relax refinement instead.
        freeze : array-like of bool, shape (n_atoms,), optional
            ``True`` for atoms to hard-freeze: they never move and are
            excluded from the forward-marginal replacement applied to the
            other known atoms. Must be disjoint from *inpaint_mask*.
        n_resample : int, optional
            Number of RePaint-style resampling passes per reverse step.
            ``1`` (default) disables resampling.
        jump_length : int, optional
            Sub-steps per resampling pass before jumping back; see
            :class:`~agedi.diffusion.samplers.InpaintingSampler`. Only
            meaningful when *n_resample* > 1. Defaults to ``1``.
        compile : bool, optional
            Not supported for inpainting (the compiled path bypasses
            samplers). Must be ``False``.
        ff_guidance : ForcefieldGuidanceConfig, optional
            Force-field guidance configuration.
        property : dict, optional
            Conditioning property values, e.g. ``{"energy": -3.5}``.
        progress_bar : bool, optional
            Show a tqdm progress bar.
        save_trajectory : bool, optional
            Return one trajectory per structure instead of final structures.
        save_corrector_frames : bool, optional
            Also record every corrector/resampling sub-step.
        print_timings : bool, optional
            Print a timing breakdown after sampling.
        sampler : str, Sampler, or None, optional
            The *inner* reverse-diffusion algorithm wrapped by the inpainting
            logic, e.g. ``"em"``, ``"pc"``, ``"heun"``. Defaults to Euler-Maruyama.
        sampler_kwargs : dict, optional
            Extra keyword arguments forwarded to the inner sampler.

        Returns
        -------
        List[AtomsGraph]
            Inpainted structures, or trajectories when *save_trajectory* is ``True``.

        """
        if compile:
            raise ValueError(
                "compile=True is not supported for inpaint(): the compiled "
                "reverse step bypasses samplers entirely, and inpainting is "
                "implemented as a sampler. Use compile=False."
            )
        if not (0.0 < t_start <= 1.0):
            raise ValueError(f"t_start must be in (0, 1], got {t_start}")

        if ff_guidance is None:
            ff_guidance = ForcefieldGuidanceConfig()

        self.score_model.sample_mode()

        device = structure.pos.device
        inpaint_mask_t = torch.as_tensor(
            np.asarray(inpaint_mask), dtype=torch.bool, device=device
        )
        if inpaint_mask_t.shape[0] != structure.pos.shape[0]:
            raise ValueError(
                f"inpaint_mask has {inpaint_mask_t.shape[0]} entries but "
                f"structure has {structure.pos.shape[0]} atoms."
            )
        if not inpaint_mask_t.any():
            raise ValueError("inpaint_mask selects no atoms; nothing to inpaint.")

        freeze_t = None
        if freeze is not None:
            freeze_t = torch.as_tensor(
                np.asarray(freeze), dtype=torch.bool, device=device
            )
            if freeze_t.shape[0] != structure.pos.shape[0]:
                raise ValueError(
                    f"freeze has {freeze_t.shape[0]} entries but "
                    f"structure has {structure.pos.shape[0]} atoms."
                )
            if (freeze_t & inpaint_mask_t).any():
                raise ValueError(
                    "freeze and inpaint_mask must be disjoint: an atom cannot "
                    "be both frozen and selected for regeneration."
                )

        if property is not None:
            for k, v in property.items():
                setattr(structure, k, torch.tensor(v, dtype=torch.float))

        # Only used by _sample() for torch.compile buffer sizing, which is
        # disallowed above; extracted defensively either way.
        _cutoff = getattr(structure, "cutoff", 6.0)
        cutoff = float(_cutoff.reshape(-1)[0].item()) if torch.is_tensor(_cutoff) else float(_cutoff)

        base_sampler = self._resolve_sampler(sampler, 0, 1e-3, sampler_kwargs)
        from agedi.diffusion.samplers import InpaintingSampler

        inpainting_sampler = InpaintingSampler(
            base_sampler, self.noisers, n_resample=n_resample, jump_length=jump_length
        )

        def graph_factory() -> AtomsGraph:
            return self._initialize_inpaint_graph(
                structure, inpaint_mask_t, freeze_t, t_start
            )

        self.zeta = ff_guidance.zeta

        sample_kwargs: Dict = {
            "progress_bar": progress_bar,
            "save_trajectory": save_trajectory,
            "save_corrector_frames": save_corrector_frames,
            "force_threshold": ff_guidance.force_threshold,
            "max_extra_steps": ff_guidance.max_extra_steps,
            "print_timings": print_timings,
            "compile": False,
            "sampler": inpainting_sampler,
            "sampler_kwargs": None,
            "t_start": t_start,
            "graph_factory": graph_factory,
        }

        if N > batch_size:
            from rich.console import Console as _Console

            _console = _Console()
            n_full = N // batch_size
            n_remainder = N % batch_size
            n_batches = n_full + (1 if n_remainder > 0 else 0)
            out = []
            for i in range(n_full):
                _console.print(f"Inpainting batch {i + 1}/{n_batches}...")
                out += self._sample(
                    batch_size, steps, cutoff, eps, ff_guidance.guidance,
                    **sample_kwargs,
                )
            if n_remainder > 0:
                _console.print(f"Inpainting batch {n_batches}/{n_batches}...")
                out += self._sample(
                    n_remainder, steps, cutoff, eps, ff_guidance.guidance,
                    **sample_kwargs,
                )
            return out
        else:
            return self._sample(
                N, steps, cutoff, eps, ff_guidance.guidance,
                **sample_kwargs,
            )
