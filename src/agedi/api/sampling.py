"""Sampling from a trained diffusion model."""

import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from rich.console import Console

from agedi.data import AtomsGraph

from ._display import _print_sampling_config


def sample(
    diffusion: "Agedi",
    *,
    n_samples: int,
    n_atoms: Optional[int] = None,
    atomic_numbers: Optional[List[int]] = None,
    formula: Optional[str] = None,
    positions: Optional[np.ndarray] = None,
    cell: Optional[np.ndarray] = None,
    pbc: Optional[np.ndarray] = None,
    template: Optional[Union[AtomsGraph, Atoms]] = None,
    confinement: Optional[Tuple[float, float]] = None,
    compile: bool = False,
    steps: int = 500,
    eps: float = 1e-3,
    cutoff: float = 6.0,
    batch_size: int = 64,
    ff_guidance: Optional["ForcefieldGuidanceConfig"] = None,
    novelty_guidance: Optional["NoveltyGuidanceConfig"] = None,
    novelty_reference: Optional[Sequence[Atoms]] = None,
    property: Optional[Dict[str, float]] = None,
    progress_bar: bool = False,
    save_trajectory: bool = False,
    save_corrector_frames: bool = False,
    print_timings: bool = False,
    as_atoms: bool = True,
    sampler=None,
    sampler_kwargs=None,
) -> Union[List[AtomsGraph], List[Atoms], List[List[AtomsGraph]], List[List[Atoms]]]:
    """Sample structures from a trained diffusion model.

    Parameters
    ----------
    diffusion:
        A trained :class:`~agedi.Agedi` model.
    n_samples:
        Number of structures to generate.
    n_atoms:
        Number of atoms per structure. Automatically determined from
        ``formula`` if provided, or from the length of ``atomic_numbers``
        when ``n_atoms`` is not explicitly given.
    atomic_numbers:
        Atomic numbers of the generated atoms.  Not required when the model
        has a types-noiser or when ``formula`` is provided.
    formula:
        Chemical formula (e.g. ``"H2O"``).  Used to derive ``n_atoms`` and
        ``atomic_numbers`` when they are not provided explicitly.
    positions:
        Fixed positions of the atoms (shape ``(n_atoms, 3)``).  Required
        when no positions-noiser is configured (type-only diffusion).
        Positions will not be modified during sampling.
    cell:
        Unit-cell matrix (3×3 array or flat length-9 array).  Not required
        when ``template`` is provided (the template's cell is used instead).
    pbc:
        Periodic boundary conditions as a length-3 boolean array (e.g.
        ``[True, True, False]``).  When ``template`` is provided its ``pbc``
        is used unless this argument is given explicitly.  Defaults to
        ``[True, True, True]`` (fully periodic) when neither ``template``
        nor ``pbc`` is supplied.
    template:
        Template structure.  May be an :class:`~agedi.AtomsGraph` or an
        ASE :class:`~ase.Atoms` object; the latter is automatically converted
        to an :class:`~agedi.AtomsGraph` (with ``confinement`` applied when
        provided).  When given, ``cell`` and ``pbc`` are taken from the
        template unless explicitly provided.
    cutoff:
        Neighbour-list cutoff radius in Ångström, used both for the sampled
        graphs and for featurising *novelty_reference*.  Should match the
        cutoff the model was trained with.  Defaults to ``6.0``.
    ff_guidance:
        Force-field guidance configuration.  When ``None`` (default) a
        :class:`~agedi.diffusion.ForcefieldGuidanceConfig` with default
        values is used (i.e. guidance is disabled).
    novelty_guidance:
        Feature-space novelty guidance configuration, which repels samples
        away from structures that have already been found.  ``None`` (default)
        disables it.  Incompatible with ``compile=True``.
    novelty_reference:
        Already-found structures to repel from.  Featurised here with the
        *current* score model — features are only comparable within one model
        generation, so pass the reference set afresh after every retraining.
        When ``None`` (and *novelty_guidance* is enabled), samples are only
        repelled from each other within the batch.
    compile:
        When ``True``, use ``torch.compile`` on the reverse diffusion step
        for faster sampling.  Before the sampling loop starts, the maximum
        number of neighbors and cell-list dimensions are estimated
        automatically via NVIDIA nvalchemiops
        (``estimate_max_neighbors`` and ``estimate_cell_list_sizes``), and
        all neighbor-list buffers are pre-allocated with fixed shapes.
        Requires NVIDIA nvalchemiops.  Defaults to ``False``.
    print_timings:
        When ``True``, print a per-stage timing breakdown at the end of
        each sampling batch (graph init, score model, denoise, neighbor
        list, etc.).  Defaults to ``False``.
    save_trajectory:
        When ``True``, return one trajectory per structure instead of a flat
        list of final structures.  Each trajectory holds one frame per
        reverse-diffusion step, plus any ``ffpc`` terminal-dynamics frames and
        post-diffusion relaxation frames.  Defaults to ``False``.
    save_corrector_frames:
        When ``True``, also record every Langevin corrector sub-step, giving a
        complete frame-by-frame trajectory.  Requires ``save_trajectory`` and a
        sampler that runs correctors (``"pc"`` / ``"ffpc"``).  Multiplies
        trajectory length by roughly the corrector count.  Defaults to
        ``False``.
    """
    from agedi.diffusion import ForcefieldGuidanceConfig

    # Convert an ASE Atoms template to AtomsGraph if needed.
    if template is not None and isinstance(template, Atoms):
        template = AtomsGraph.from_atoms(
            template, cutoff=cutoff, confinement=confinement
        )

    _ff = ff_guidance if ff_guidance is not None else ForcefieldGuidanceConfig()

    # Featurise the reference structures with the current score model.  The
    # archive is deliberately rebuilt on every call: features live in the
    # backbone's activation space and are meaningless across retrainings.
    _archive = None
    if novelty_reference is not None and len(novelty_reference) > 0:
        from agedi.diffusion.novelty import FeatureArchive

        _archive = FeatureArchive.from_structures(
            diffusion.score_model,
            novelty_reference,
            cutoff=cutoff,
            pool=novelty_guidance.pool if novelty_guidance is not None else "mean",
        )

    # Determine display name for the top-level sampler algorithm.
    if sampler is not None:
        _sampler = sampler if isinstance(sampler, str) else type(sampler).__name__
    else:
        _sampler = None  # default (EM) — not shown separately

    _print_sampling_config(
        n_samples=n_samples,
        steps=steps,
        eps=eps,
        batch_size=batch_size,
        formula=formula,
        n_atoms=n_atoms,
        template=template,
        cell=cell,
        confinement=confinement,
        property=property,
        force_field_guidance=_ff.guidance,
        novelty_guidance=(
            novelty_guidance.guidance if novelty_guidance is not None else 0.0
        ),
        novelty_references=len(_archive) if _archive is not None else 0,
        sampler=_sampler,
    )

    _start = time.monotonic()

    diffusion.eval()
    with torch.no_grad():
        sampled = diffusion.sample(
            N=n_samples,
            template=template,
            batch_size=batch_size,
            steps=steps,
            eps=eps,
            cutoff=cutoff,
            n_atoms=n_atoms,
            atomic_numbers=atomic_numbers,
            formula=formula,
            positions=positions,
            cell=cell,
            pbc=pbc,
            confinement=confinement,
            compile=compile,
            ff_guidance=_ff,
            novelty_guidance=novelty_guidance,
            novelty_archive=_archive,
            property=property,
            progress_bar=progress_bar,
            save_trajectory=save_trajectory,
            save_corrector_frames=save_corrector_frames,
            print_timings=print_timings,
            sampler=sampler,
            sampler_kwargs=sampler_kwargs,
        )

    elapsed = time.monotonic() - _start
    n_generated = len(sampled)
    Console().print(f"[green]✓[/green] Generated {n_generated} structure(s) in {elapsed:.1f}s")

    if not as_atoms:
        return sampled

    if save_trajectory:
        return [[graph.to_atoms() for graph in trajectory] for trajectory in sampled]
    return [graph.to_atoms() for graph in sampled]
