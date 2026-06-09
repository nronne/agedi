"""Sampling from a trained diffusion model."""

import time
from typing import Dict, List, Optional, Tuple, Union

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
    batch_size: int = 64,
    ff_guidance: Optional["ForcefieldGuidanceConfig"] = None,
    property: Optional[Dict[str, float]] = None,
    progress_bar: bool = False,
    save_trajectory: bool = False,
    print_timings: bool = False,
    as_atoms: bool = True,
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
    ff_guidance:
        Force-field guidance configuration.  When ``None`` (default) a
        :class:`~agedi.diffusion.ForcefieldGuidanceConfig` with default
        values is used (i.e. guidance is disabled).
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
    """
    from agedi.diffusion import ForcefieldGuidanceConfig

    # Convert an ASE Atoms template to AtomsGraph if needed.
    if template is not None and isinstance(template, Atoms):
        template = AtomsGraph.from_atoms(template, confinement=confinement)

    _ff = ff_guidance if ff_guidance is not None else ForcefieldGuidanceConfig()

    # Extract sampler from the first position noiser.
    _sampler = None
    for _n in diffusion.noisers:
        if hasattr(_n, "sampler"):
            _sampler = _n.sampler
            break

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
            n_atoms=n_atoms,
            atomic_numbers=atomic_numbers,
            formula=formula,
            positions=positions,
            cell=cell,
            pbc=pbc,
            confinement=confinement,
            compile=compile,
            ff_guidance=_ff,
            property=property,
            progress_bar=progress_bar,
            save_trajectory=save_trajectory,
            print_timings=print_timings,
        )

    elapsed = time.monotonic() - _start
    n_generated = len(sampled)
    Console().print(f"[green]✓[/green] Generated {n_generated} structure(s) in {elapsed:.1f}s")

    if not as_atoms:
        return sampled

    if save_trajectory:
        return [[graph.to_atoms() for graph in trajectory] for trajectory in sampled]
    return [graph.to_atoms() for graph in sampled]
