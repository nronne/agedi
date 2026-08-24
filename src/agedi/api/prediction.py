"""Force-field prediction using a trained regressor."""

from typing import List, Optional, Sequence

import torch
from ase import Atoms
from rich.console import Console

from agedi.data import AtomsGraph

from ._common import resolve_cutoff


def predict(
    diffusion: "Agedi",
    structures: Sequence[Atoms],
    *,
    batch_size: int = 64,
    cutoff: Optional[float] = None,
) -> List[Atoms]:
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
        Input ASE :class:`~ase.Atoms` objects to run predictions on.
    batch_size:
        Number of structures per inference batch.  Defaults to ``64``.
    cutoff:
        Neighbour-list cutoff in Å.  When ``None`` (default), the cutoff is
        read from the model's representation automatically.

    Returns
    -------
    List[Atoms]
        The input structures with a
        :class:`~ase.calculators.singlepoint.SinglePointCalculator` attached
        containing the predicted energy and/or forces.

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

    cutoff = resolve_cutoff(diffusion, cutoff)

    device = next(diffusion.parameters()).device

    graphs = [AtomsGraph.from_atoms(atoms, cutoff=cutoff) for atoms in structures]

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
    return results
