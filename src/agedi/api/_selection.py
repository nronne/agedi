"""Atom-selection helpers for :func:`~agedi.api.inpainting.inpaint`."""

from typing import Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms


def _fixed_mask(atoms: Atoms) -> np.ndarray:
    """Return a bool mask, ``True`` for atoms held by a ``FixAtoms`` constraint."""
    fixed = np.zeros(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed[constraint.index] = True
    return fixed


def _grow_contiguous_selection(
    atoms: Atoms, candidates: np.ndarray, n_select: int, rng: np.random.Generator
) -> np.ndarray:
    """Greedily grow a spatially-connected cluster of *n_select* candidates.

    Starts from one random candidate, then repeatedly attaches whichever
    remaining candidate is geometrically closest to *any* atom already in the
    growing cluster (nearest-neighbor agglomeration) -- the same idea as
    building a minimum spanning tree one edge at a time. This keeps the
    selection a single contiguous blob around the seed atom rather than
    scattered points, without requiring a bonding cutoff. Distances use the
    minimum-image convention when *atoms* has any periodic direction, so a
    cluster can wrap across a periodic boundary correctly.
    """
    dist_matrix = atoms.get_all_distances(mic=bool(atoms.pbc.any()))

    start = int(rng.choice(candidates))
    selected = [start]
    remaining = set(candidates.tolist())
    remaining.discard(start)

    while len(selected) < n_select and remaining:
        remaining_arr = np.fromiter(remaining, dtype=int)
        min_dists = dist_matrix[np.ix_(remaining_arr, selected)].min(axis=1)
        nearest = int(remaining_arr[np.argmin(min_dists)])
        selected.append(nearest)
        remaining.discard(nearest)

    return np.asarray(selected, dtype=int)


def select_atoms(
    atoms: Atoms,
    *,
    indices: Optional[Sequence[int]] = None,
    symbols: Optional[Sequence[str]] = None,
    z_range: Optional[Tuple[float, float]] = None,
    sphere: Optional[Tuple[Sequence[float], float]] = None,
    from_atoms: bool = False,
    fraction: float = 0.25,
    contiguous: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Resolve which atoms to inpaint from a combination of selection criteria.

    Every criterion that is given contributes atoms via a set **union**; e.g.
    passing both *symbols* and *z_range* selects atoms matching either. When
    **no** criterion is given at all, the default is a random *fraction* of
    the atoms not held by an ASE :class:`~ase.constraints.FixAtoms`
    constraint.

    Parameters
    ----------
    atoms : ase.Atoms
        The input structure.
    indices : sequence of int, optional
        Explicit atom indices to select.
    symbols : sequence of str, optional
        Select every atom whose chemical symbol is in this list.
    z_range : (float, float), optional
        ``(z_min, z_max)``: select atoms whose Cartesian z-coordinate falls
        in this range (inclusive).
    sphere : (center, radius), optional
        ``center`` a length-3 array-like, ``radius`` a float: select atoms
        within Euclidean distance *radius* of *center*.
    from_atoms : bool, optional
        Read the selection off *atoms* itself: ``atoms.arrays["inpaint_mask"]``
        when present (e.g. round-tripping a previous inpainting result via
        :meth:`~agedi.data.AtomsGraph.to_atoms`), otherwise the complement of
        any ``FixAtoms`` constraint (every atom *not* held fixed).
    fraction : float, optional
        Fraction of the non-fixed atoms to select at random when no other
        criterion is given. Defaults to ``0.25``.
    contiguous : bool, optional
        When ``True``, the *fraction* fallback selects a spatially-connected
        cluster of neighboring atoms instead of a scattered random subset:
        one random seed atom, then repeatedly the geometrically closest
        remaining candidate to the growing cluster, until *fraction* is
        reached. Uses the minimum-image convention when *atoms* has any
        periodic direction. Has no effect when any other selection criterion
        is given (``fraction`` itself only applies as a fallback). Defaults
        to ``False``.
    seed : int, optional
        Seed for the random fraction fallback, for reproducible selections.

    Returns
    -------
    numpy.ndarray of bool, shape (len(atoms),)
        ``True`` for atoms to regenerate.

    Raises
    ------
    ValueError
        If the resolved selection is empty or covers every atom.

    """
    n_atoms = len(atoms)
    criteria_given = any(
        c is not None and c is not False
        for c in (indices, symbols, z_range, sphere)
    ) or from_atoms

    mask = np.zeros(n_atoms, dtype=bool)

    if indices is not None:
        idx = np.asarray(list(indices), dtype=int)
        mask[idx] = True

    if symbols is not None:
        symbol_arr = np.asarray(atoms.get_chemical_symbols())
        mask |= np.isin(symbol_arr, list(symbols))

    if z_range is not None:
        z_min, z_max = z_range
        z = atoms.get_positions()[:, 2]
        mask |= (z >= z_min) & (z <= z_max)

    if sphere is not None:
        center, radius = sphere
        center = np.asarray(center, dtype=float).reshape(3)
        dist = np.linalg.norm(atoms.get_positions() - center, axis=1)
        mask |= dist <= radius

    if from_atoms:
        if "inpaint_mask" in atoms.arrays:
            mask |= np.asarray(atoms.arrays["inpaint_mask"], dtype=bool)
        else:
            mask |= ~_fixed_mask(atoms)

    if not criteria_given:
        fixed = _fixed_mask(atoms)
        candidates = np.flatnonzero(~fixed)
        if candidates.size == 0:
            raise ValueError(
                "select_atoms: no criterion was given and every atom is held "
                "by a FixAtoms constraint, so there are no atoms left to pick "
                "a random fraction from."
            )
        rng = np.random.default_rng(seed)
        n_select = max(1, int(round(fraction * candidates.size)))
        if contiguous:
            chosen = _grow_contiguous_selection(atoms, candidates, n_select, rng)
        else:
            chosen = rng.choice(candidates, size=n_select, replace=False)
        mask[chosen] = True

    if not mask.any():
        raise ValueError(
            "select_atoms: the resolved selection is empty; nothing to inpaint."
        )
    if mask.all():
        raise ValueError(
            "select_atoms: the resolved selection covers every atom; leave at "
            "least one atom out of the selection to inpaint against."
        )

    return mask
