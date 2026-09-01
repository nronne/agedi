"""Local structure relaxation using a trained force-field regressor."""

from typing import List, Optional, Sequence, Union

import torch
from ase import Atoms
from ase.constraints import FixAtoms
from rich.console import Console

from agedi.data import AtomsGraph
from agedi.diffusion.guidance import (
    BatchedLBFGSStepSizer,
    max_force_per_graph,
    post_diffusion_relaxation_step,
)

from ._common import resolve_cutoff


def _graph_from_atoms(
    atoms: Atoms, cutoff: float, fully_connected: bool = False
) -> AtomsGraph:
    """Build a graph from *atoms*, carrying ``FixAtoms`` over to the mask.

    Masked atoms have their predicted forces zeroed by the regressor and their
    positions restored on assignment, so the mask is what makes ASE's
    ``FixAtoms`` constraint take effect during relaxation.

    ``fully_connected`` must match the model's own setting: it is stored on
    the graph and read by :meth:`~agedi.data.AtomsGraph.update_graph`, which
    is called every relaxation step. Leaving it unset on a fully-connected
    model makes ``update_graph`` rebuild a cutoff-based neighbour list
    instead, silently changing the physics and, as the per-graph edge count
    drifts, eventually breaking ``Batch.to_data_list()`` (whose slicing
    metadata is fixed at ``Batch.from_data_list()`` time).
    """
    graph = AtomsGraph.from_atoms(
        atoms, cutoff=cutoff, initialize_mask=True, fully_connected=fully_connected
    )
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            graph.mask[constraint.index] = True
    return graph


def relax(
    diffusion: "Agedi",
    structures: Union[Sequence[Atoms], Sequence[Sequence[Atoms]]],
    *,
    fmax: float = 0.05,
    steps: int = 200,
    batch_size: int = 64,
    cutoff: Optional[float] = None,
    max_step_size: float = 0.2,
    trajectory: bool = False,
    progress_bar: bool = False,
) -> Union[List[Atoms], List[List[Atoms]]]:
    """Relax structures with a trained force-field regressor.

    Runs the same batched L-BFGS optimiser that
    :func:`~agedi.api.sample` uses for its post-diffusion relaxation, but on
    structures you supply — no diffusion sampling involved.  The algorithm
    mirrors :class:`ase.optimize.LBFGS`: a constant inverse-Hessian seed, the
    two-loop recursion, and a per-structure ``maxstep`` cap.

    Each structure in a batch is optimised by its own L-BFGS instance and drops
    out as soon as its own maximum force falls below *fmax*, so a slow
    structure never perturbs one that has already converged.

    ASE :class:`~ase.constraints.FixAtoms` constraints on the input are
    honoured — constrained atoms keep their positions and are excluded from the
    convergence check.

    Parameters
    ----------
    diffusion:
        A trained :class:`~agedi.Agedi` model with a force-field regressor
        (trained with ``force_field=True``).
    structures:
        Input ASE :class:`~ase.Atoms` objects to relax, as a flat list, or a
        list of lists — the grouped shape :func:`~agedi.api.inpainting.inpaint`
        returns for a list of input structures.  Nesting is auto-detected
        from the first element; the return value is grouped the same way.
    fmax:
        Convergence criterion: the maximum per-atom force magnitude in eV/Å.
        Defaults to ``0.05``, matching ASE's usual choice.
    steps:
        Maximum number of optimiser steps per structure.  Defaults to ``200``.
    batch_size:
        Number of structures relaxed simultaneously.  Defaults to ``64``.
    cutoff:
        Neighbour-list cutoff in Å.  When ``None`` (default), it is read from
        the model's representation.
    max_step_size:
        Maximum single-atom displacement per step, in Å.  Defaults to ``0.2``
        (ASE's default).
    trajectory:
        When ``True``, return the full relaxation trajectory of every structure
        instead of only the final frame.
    progress_bar:
        Show a per-batch progress bar over optimiser steps.

    Returns
    -------
    list
        The relaxed structures, each with a
        :class:`~ase.calculators.singlepoint.SinglePointCalculator` holding the
        predicted energy and forces.  When *trajectory* is ``True``, a list of
        trajectories (one list of :class:`~ase.Atoms` per input structure),
        each starting at the input geometry.  Flat for a flat input, or
        grouped the same way as a nested input — so
        ``inpaint() -> relax() -> predict()`` chains without flattening at
        any step.

    Raises
    ------
    ValueError
        If the model does not have a force-field regressor.

    Examples
    --------
    >>> from agedi.api import load_diffusion, relax
    >>> model = load_diffusion("lightning_logs/version_0")
    >>> relaxed = relax(model, structures, fmax=0.02)
    >>> relaxed[0].get_potential_energy()
    """
    from torch_geometric.data import Batch

    if diffusion.regressor_model is None:
        raise ValueError(
            "This model does not have a force-field regressor. "
            "Re-train with force_field=True to enable relaxation."
        )

    is_nested = len(structures) > 0 and isinstance(structures[0], (list, tuple))
    if is_nested:
        group_sizes = [len(g) for g in structures]
        flat_structures = [a for g in structures for a in g]
    else:
        flat_structures = list(structures)

    cutoff = resolve_cutoff(diffusion, cutoff)
    fully_connected = getattr(diffusion, "fully_connected", False)
    device = next(diffusion.parameters()).device

    graphs = [
        _graph_from_atoms(atoms, cutoff, fully_connected=fully_connected)
        for atoms in flat_structures
    ]
    n_structures = len(graphs)

    console = Console()
    console.print(
        f"Relaxing {n_structures} structure(s) "
        f"(fmax={fmax}, max steps={steps}, batch_size={batch_size})..."
    )

    diffusion.eval()
    results: List = []
    n_converged = 0

    with torch.no_grad():
        for start in range(0, n_structures, batch_size):
            chunk = graphs[start : start + batch_size]
            batch = Batch.from_data_list(chunk).to(device)
            batch.update_graph()

            sizer = BatchedLBFGSStepSizer(
                batch_size=batch.num_graphs, maxstep=max_step_size
            )

            batch = diffusion.regressor_model(batch)
            per_graph_forces = _mobile_max_force(batch)

            frames = [batch.to_data_list()] if trajectory else None

            iterator = range(steps)
            if progress_bar:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Relaxation")

            for _ in iterator:
                active = per_graph_forces > fmax
                if not bool(active.any()):
                    break

                batch = post_diffusion_relaxation_step(
                    batch,
                    diffusion.regressor_model,
                    sizer,
                    max_step_size=max_step_size,
                    forces=batch.forces_prediction,
                    active=active,
                )
                batch = diffusion.regressor_model(batch)
                per_graph_forces = _mobile_max_force(batch)

                if trajectory:
                    frames.append(batch.to_data_list())

            n_converged += int((per_graph_forces <= fmax).sum())

            originals = flat_structures[start : start + batch_size]
            if trajectory:
                # frames is [step][structure]; transpose to [structure][step].
                for i, per_structure in enumerate(zip(*frames)):
                    results.append(
                        [
                            _finalise(graph.to_atoms(), originals[i])
                            for graph in per_structure
                        ]
                    )
            else:
                for i, graph in enumerate(batch.to_data_list()):
                    results.append(_finalise(graph.to_atoms(), originals[i]))

    console.print(
        f"[green]✓[/green] Relaxed {n_structures} structure(s); "
        f"{n_converged}/{n_structures} reached fmax <= {fmax}"
    )

    if not is_nested:
        return results

    grouped: List[List] = []
    idx = 0
    for size in group_sizes:
        grouped.append(results[idx : idx + size])
        idx += size
    return grouped


def _finalise(atoms: Atoms, original: Atoms) -> Atoms:
    """Carry the input constraints over to a relaxed structure."""
    if original.constraints:
        atoms.set_constraint(original.constraints)
    return atoms


def _mobile_max_force(batch) -> torch.Tensor:
    """Per-graph maximum force over the *mobile* atoms only.

    A fixed atom may carry a large force without that meaning anything for
    convergence — it cannot move.  The regressor normally zeroes those forces
    itself (``mask_forces=True``), but that is configurable, so the mask is
    applied here explicitly.
    """
    forces = batch.forces_prediction
    if "mask" in batch:
        forces = forces.clone()
        forces[batch.positions_mask] = 0.0
    return max_force_per_graph(forces, batch.batch, batch.num_graphs)
