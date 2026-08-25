"""Force-field prediction using a trained regressor."""

from typing import List, Optional, Sequence, Union

import torch
from ase import Atoms
from rich.console import Console

from agedi.data import AtomsGraph


def predict(
    diffusion: "Agedi",
    structures: Union[Sequence[Atoms], Sequence[Sequence[Atoms]]],
    *,
    batch_size: int = 64,
    cutoff: Optional[float] = None,
) -> Union[List[Atoms], List[List[Atoms]]]:
    """Predict energies and forces for input structures using a trained force-field.

    The model must have been trained with ``force_field=True`` (i.e. it must
    have a ``regressor_model`` attached).  The predicted energy and forces are
    attached to the returned :class:`~ase.Atoms` objects via an
    :class:`~ase.calculators.singlepoint.SinglePointCalculator`.

    Parameters
    ----------
    diffusion:
        A trained :class:`~agedi.Agedi` model with a force-field
        regressor (trained with ``--force_field``).
    structures:
        Input ASE :class:`~ase.Atoms` objects to run predictions on, as a
        flat list, or a list of lists -- the grouped shape
        :func:`~agedi.api.inpainting.inpaint` returns for a list of input
        structures. Nesting is auto-detected from the first element; the
        return value is grouped the same way as the input.
    batch_size:
        Number of structures per inference batch.  Defaults to ``64``.
    cutoff:
        Neighbour-list cutoff in Å.  When ``None`` (default), the cutoff is
        read from the model's representation automatically.

    Returns
    -------
    List[Atoms] or List[List[Atoms]]
        The input structures with a
        :class:`~ase.calculators.singlepoint.SinglePointCalculator` attached
        containing the predicted energy and/or forces. Flat for a flat input,
        or grouped the same way as a nested input.

    Raises
    ------
    ValueError
        If the model does not have a force-field regressor.
    """
    from torch_geometric.data import Batch

    if diffusion.regressor_model is None:
        raise ValueError(
            "This model does not have a force-field regressor. "
            "Re-train with force_field=True to enable predictions."
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

    graphs = [AtomsGraph.from_atoms(atoms, cutoff=cutoff) for atoms in flat_structures]

    n_structures = len(graphs)
    console = Console()
    console.print(f"Running predictions on {n_structures} structure(s) (batch_size={batch_size})...")

    diffusion.eval()
    results: List[Atoms] = []
    with torch.no_grad():
        for i in range(0, n_structures, batch_size):
            batch_graphs = graphs[i : i + batch_size]
            batch = Batch.from_data_list(batch_graphs).to(device)
            batch = diffusion.regressor_model(batch)
            for graph in batch.to_data_list():
                results.append(graph.to_atoms())

    console.print(f"[green]✓[/green] Predictions complete for {len(results)} structure(s)")

    if not is_nested:
        return results

    grouped: List[List[Atoms]] = []
    idx = 0
    for size in group_sizes:
        grouped.append(results[idx : idx + size])
        idx += size
    return grouped
