"""Structure relaxation using a trained regressor and batched L-BFGS."""

from typing import List, Optional, Sequence, Union

import torch
from ase import Atoms
from rich.console import Console
from tqdm import tqdm

from agedi.data import AtomsGraph
from agedi.diffusion.guidance import BatchedLBFGSStepSizer, post_diffusion_relaxation_step

from ._selection import _fixed_mask


def relax(
    diffusion: "Agedi",
    structures: Union[Sequence[Atoms], Sequence[Sequence[Atoms]]],
    *,
    batch_size: int = 64,
    cutoff: Optional[float] = None,
    max_steps: int = 200,
    force_threshold: float = 0.05,
    scale: float = 1.0,
    max_step_size: float = 0.2,
    progress_bar: bool = False,
) -> Union[List[Atoms], List[List[Atoms]]]:
    """Relax structures with batched L-BFGS, using a trained force-field.

    The model must have been trained with ``force_field=True`` (i.e. it must
    have a ``regressor_model`` attached). Each structure is relaxed
    independently but in the same batch, taking one L-BFGS step (as
    ``ase.optimize.LBFGS`` would, with ``damping=scale`` and the same
    ``maxstep`` semantics) per iteration, re-evaluating forces from the
    regressor after every step. Relaxation for the whole batch stops once the
    largest per-atom force magnitude across *any* structure in the batch
    drops to or below *force_threshold*, or after *max_steps* iterations,
    whichever comes first — the same convergence criterion used for
    post-diffusion relaxation during sampling. The final energy and forces
    are attached to the returned :class:`~ase.Atoms` objects via a
    :class:`~ase.calculators.singlepoint.SinglePointCalculator`, exactly like
    :func:`~agedi.api.prediction.predict`.

    Atoms held by an ASE :class:`~ase.constraints.FixAtoms` constraint on the
    input structure are frozen for the relaxation (never move) — this is the
    one behavioural difference from :func:`~agedi.api.prediction.predict`,
    which is otherwise mirrored exactly (same batching, same cutoff
    resolution, same flat/grouped input and output shapes).

    Parameters
    ----------
    diffusion:
        A trained :class:`~agedi.Agedi` model with a force-field
        regressor (trained with ``--force_field``).
    structures:
        Input ASE :class:`~ase.Atoms` objects to relax, as a flat list, or a
        list of lists -- the grouped shape
        :func:`~agedi.api.inpainting.inpaint` returns for a list of input
        structures. Nesting is auto-detected from the first element; the
        return value is grouped the same way as the input.
    batch_size:
        Number of structures per relaxation batch. Defaults to ``64``.
    cutoff:
        Neighbour-list cutoff in Å. When ``None`` (default), the cutoff is
        read from the model's representation automatically.
    max_steps:
        Maximum number of L-BFGS steps. Defaults to ``200``.
    force_threshold:
        Convergence threshold on the maximum per-atom force magnitude
        (eV/Å), checked across the whole batch. Defaults to ``0.05``.
    scale:
        Multiplier on the computed L-BFGS step, equivalent to ASE's
        ``damping``. Defaults to ``1.0`` (take the full step).
    max_step_size:
        Maximum single-atom displacement per step, in Å. Defaults to ``0.2``,
        matching ASE's default.
    progress_bar:
        Show a tqdm progress bar and print convergence status per batch.

    Returns
    -------
    List[Atoms] or List[List[Atoms]]
        The relaxed structures with a
        :class:`~ase.calculators.singlepoint.SinglePointCalculator` attached
        containing the final energy and/or forces. Flat for a flat input, or
        grouped the same way as a nested input.

    Raises
    ------
    ValueError
        If the model does not have a force-field regressor.
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

    if cutoff is None:
        try:
            cf = diffusion.score_model.representation.cutoff_fn
            if hasattr(cf, "cutoff") and cf.cutoff.numel() > 0:
                cutoff = float(cf.cutoff[0])
            else:
                cutoff = 6.0
        except AttributeError:
            cutoff = 6.0

    device = next(diffusion.parameters()).device

    graphs = []
    for atoms in flat_structures:
        graph = AtomsGraph.from_atoms(atoms, cutoff=cutoff)
        fixed = _fixed_mask(atoms)
        if fixed.any():
            graph.mask = torch.as_tensor(fixed, dtype=torch.bool)
        graphs.append(graph)

    n_structures = len(graphs)
    console = Console()
    console.print(
        f"Relaxing {n_structures} structure(s) (batch_size={batch_size}, "
        f"max_steps={max_steps}, force_threshold={force_threshold})..."
    )

    diffusion.eval()
    results: List[Atoms] = []
    with torch.no_grad():
        for i in range(0, n_structures, batch_size):
            batch_graphs = graphs[i : i + batch_size]
            batch = Batch.from_data_list(batch_graphs).to(device)

            batch = diffusion.regressor_model(batch)
            max_forces = torch.norm(batch.forces_prediction, dim=1).max(dim=0)[0]

            if max_forces > force_threshold and max_steps > 0:
                # A persistent step sizer across the whole relaxation: a
                # fresh one per step carries no curvature history and
                # degrades to seed-length steps.
                step_sizer = BatchedLBFGSStepSizer(batch_size=batch.batch_size)
                iterator = (
                    tqdm(range(max_steps), desc="Relaxing")
                    if progress_bar
                    else range(max_steps)
                )
                for step in iterator:
                    batch = post_diffusion_relaxation_step(
                        batch, diffusion.regressor_model, step_sizer,
                        scale=scale, max_step_size=max_step_size,
                    )
                    batch = diffusion.regressor_model(batch)
                    max_forces = torch.norm(batch.forces_prediction, dim=1).max(dim=0)[0]
                    if max_forces <= force_threshold:
                        if progress_bar:
                            console.print(
                                f"Converged after {step + 1} steps, "
                                f"max force: {max_forces:.4f}"
                            )
                        break
                else:
                    if progress_bar:
                        console.print(
                            f"Did not converge, final max force: {max_forces:.4f}"
                        )

            for graph in batch.to_data_list():
                results.append(graph.to_atoms())

    console.print(f"[green]✓[/green] Relaxation complete for {len(results)} structure(s)")

    if not is_nested:
        return results

    grouped: List[List[Atoms]] = []
    idx = 0
    for size in group_sizes:
        grouped.append(results[idx : idx + size])
        idx += size
    return grouped
