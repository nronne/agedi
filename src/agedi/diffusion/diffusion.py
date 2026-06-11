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
from typing import Dict, List, Optional, Tuple, Union

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
    post_diffusion_relaxation_step,
)


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
    guidance_wrap_positions: float = 0.0
    guidance_neighbor_list: float = 0.0
    post_diffusion_force_eval: float = 0.0
    post_diffusion_relaxation: float = 0.0
    post_diffusion_wrap_positions: float = 0.0
    post_diffusion_neighbor_list: float = 0.0
    post_diffusion_relaxation_force_eval: float = 0.0
    total_wall: float = 0.0
    reverse_step_calls: int = 0
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
        scale: float = 0.1,
    ) -> AtomsGraph:
        """Perform a pure force-based relaxation step.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.
        scale : float, optional
            Step size scaling factor.

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
                "score model", timings.score_model, timings.reverse_step_calls
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
        if timings.force_field_guidance > 0 or timings.guidance_neighbor_list > 0:
            print(
                self._format_timing_line(
                    "force-field guidance",
                    timings.force_field_guidance,
                    timings.reverse_step_calls,
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

        Returns
        -------
        List[AtomsGraph]
            Final structures, or (when *save_trajectory* is ``True``) a list of
            trajectories (one per graph).
        """
        if reverse_step_fn is None:
            reverse_step_fn = self.reverse_step

        if steps < 2:
            return batch.to_data_list()

        if force_field_guidance > 0 and self.regressor_model is not None:
            self.lbfgs_step_sizer = BatchedLBFGSStepSizer(
                batch_size=batch.batch_size
            )

        ts = torch.linspace(1, eps, steps, device=self.device)
        dt = ts[0] - ts[1]

        # Pre-create corrector delta_t tensor once to avoid per-call allocation.
        corrector_dt: Optional[torch.Tensor] = None
        if corrector_steps > 0:
            corrector_dt = torch.tensor(
                corrector_step_size, dtype=dt.dtype, device=self.device
            )

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

            # Predictor step
            if is_compiled:
                # compiled_reverse_step cannot accept timings (time.perf_counter
                # is not traceable by Dynamo); time the whole call from outside.
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
                else:
                    batch = reverse_step_fn(
                        batch, dt, force_field_guidance, last=last_step
                    )
            else:
                batch = reverse_step_fn(
                    batch, dt, force_field_guidance, last=last_step, timings=timings
                )

            # Corrector steps at constant time t_i
            for _ in range(corrector_steps):
                if corrector_dt is None:
                    raise RuntimeError(
                        "corrector_dt is None but corrector_steps > 0; "
                        "this indicates a bug in _sample_batch initialisation."
                    )
                batch.add_batch_attr(
                    "time", ts[i].repeat(batch.x.shape[0], 1), type="node"
                )
                batch = self.corrector_step(batch, corrector_dt)

        # Optional post-diffusion relaxation
        if force_field_guidance > 0 and self.regressor_model is not None:
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
            max_forces = torch.norm(batch.forces_prediction, dim=1).max(dim=0)[0]

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
                    if timings is None:
                        batch = self.post_diffusion_relaxation_step(batch, scale=0.1)
                    else:
                        batch = self._time_sampling_call(
                            batch.pos.device,
                            timings,
                            "post_diffusion_relaxation",
                            self.post_diffusion_relaxation_step,
                            batch,
                            scale=0.1,
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
                    max_forces = torch.norm(batch.forces_prediction, dim=1).max(
                        dim=0
                    )[0]

                    if save_trajectory:
                        path.append(batch.to_data_list())

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
        **kwargs
            Keyword arguments forwarded to :meth:`_initialize_graph`.

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
        property: Optional[Dict] = None,
        progress_bar: Optional[bool] = False,
        save_trajectory: Optional[bool] = False,
        print_timings: Optional[bool] = False,
        corrector_steps: int = 0,
        corrector_step_size: float = 1e-3,
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
        property : dict, optional
            Conditioning properties (key -> scalar tensor).
        progress_bar : bool, optional
            Show a tqdm progress bar.
        save_trajectory : bool, optional
            Return full trajectories instead of final structures.
        print_timings : bool, optional
            Print a timing breakdown after sampling completes.
        corrector_steps : int, optional
            Number of Langevin corrector passes after each predictor step.
            ``0`` (default) gives standard (predictor-only) sampling.
        corrector_step_size : float, optional
            Step size for each corrector pass.  Defaults to ``1e-3``.

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
            "force_threshold": ff_guidance.force_threshold,
            "max_extra_steps": ff_guidance.max_extra_steps,
            "corrector_steps": corrector_steps,
            "corrector_step_size": corrector_step_size,
            "print_timings": print_timings,
            "compile": compile,
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
