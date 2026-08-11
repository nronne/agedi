"""Per-species reference energies for force-field training.

Total DFT energies are dominated by a large, composition-dependent offset that
carries no structural information.  Subtracting a per-species reference energy
:math:`E^0_Z` turns the regression target into a *formation-like* energy of much
smaller magnitude, which makes the energy head far easier to fit.

The reference energies are obtained by a linear least-squares fit of

.. math::

    E_\\text{total}(s) \\approx \\sum_Z n_Z(s) \\, E^0_Z

over the training structures, where :math:`n_Z(s)` is the number of atoms of
species :math:`Z` in structure :math:`s`.  Users can also supply their own
values (e.g. isolated-atom energies) instead of fitting them.

Inside the model the references are applied as a fixed atomic offset that is
*added back* to the predicted energy (see
:class:`~agedi.models.schnetpack.regressor_heads.Energy`), so the network only
has to learn the residual while predictions stay on the original absolute
energy scale.
"""

from typing import Dict, List, Mapping, Optional, Sequence, Union
import warnings

import numpy as np
import torch
from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS
from ase.data import chemical_symbols as ASE_CHEMICAL_SYMBOLS


#: Size of the reference-energy lookup table (index == atomic number).  Matches
#: the default vocabulary size used by the type embeddings.
MAX_ATOMIC_NUMBER = 100

#: Accepted user specifications: ``{"Cu": -3.5}``, ``{29: -3.5}`` or ``None``.
ReferenceEnergySpec = Union[Mapping[Union[int, str], float], None]


__all__ = [
    "MAX_ATOMIC_NUMBER",
    "normalize_reference_energies",
    "fit_reference_energies",
    "reference_energies_to_tensor",
    "tensor_to_reference_energies",
    "format_reference_energies",
]


def normalize_reference_energies(spec: ReferenceEnergySpec) -> Dict[int, float]:
    """Normalise a user-supplied reference-energy mapping to ``{Z: energy}``.

    Parameters
    ----------
    spec : Mapping or None
        Mapping from chemical symbol (``"Cu"``) or atomic number (``29``) to
        the reference energy of that species, in the same energy unit as the
        training data.  ``None`` (or an empty mapping) yields an empty dict.

    Returns
    -------
    dict
        Mapping from atomic number to reference energy.

    Raises
    ------
    ValueError
        If a key is neither a known chemical symbol nor a valid atomic number,
        or if a value cannot be converted to ``float``.
    """
    if spec is None:
        return {}
    if not isinstance(spec, Mapping):
        raise ValueError(
            "reference_energies must be a mapping from species (symbol or atomic "
            f"number) to energy, got {type(spec).__name__}."
        )

    normalized: Dict[int, float] = {}
    for key, value in spec.items():
        if isinstance(key, str):
            symbol = key.strip().capitalize()
            if symbol not in ASE_ATOMIC_NUMBERS:
                raise ValueError(f"Unknown chemical symbol in reference_energies: '{key}'")
            number = int(ASE_ATOMIC_NUMBERS[symbol])
        elif isinstance(key, (int, np.integer)):
            number = int(key)
        else:
            raise ValueError(
                "reference_energies keys must be chemical symbols or atomic "
                f"numbers, got {type(key).__name__}."
            )

        if not 0 < number < MAX_ATOMIC_NUMBER:
            raise ValueError(
                f"Atomic number {number} in reference_energies is out of range "
                f"(must be in [1, {MAX_ATOMIC_NUMBER - 1}])."
            )

        try:
            normalized[number] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Reference energy for species {number} is not a number: {value!r}"
            ) from exc

    return normalized


def _structure_energy(atoms) -> Optional[float]:
    """Return the total energy stored on *atoms*, or ``None`` when absent.

    Only already-computed calculator results are read so that no (potentially
    expensive) calculation is triggered — mirroring
    :meth:`agedi.data.Dataset._has_energy_forces`.
    """
    if getattr(atoms, "calc", None) is None:
        return None
    results = getattr(atoms.calc, "results", {})
    if "energy" not in results:
        return None
    return float(results["energy"])


def fit_reference_energies(
    data: Sequence["Atoms"],
    *,
    warn: bool = True,
) -> Dict[int, float]:
    """Fit per-species reference energies by linear least squares.

    Solves :math:`\\min_{E^0} \\| C E^0 - E \\|_2` where ``C`` is the matrix of
    per-species atom counts and ``E`` the vector of total energies of all
    structures in *data* that carry an energy.

    Parameters
    ----------
    data : Sequence[Atoms]
        ASE structures.  Structures without a computed energy are skipped.
    warn : bool, optional
        Emit :class:`UserWarning` when the fit is under-determined or when no
        structure carries an energy.  Defaults to ``True``.

    Returns
    -------
    dict
        Mapping from atomic number to fitted reference energy.  Empty when no
        structure in *data* has an energy.

    Notes
    -----
    When the compositions in *data* do not span the species space (e.g. a
    single fixed composition), the system is rank-deficient and
    :func:`numpy.linalg.lstsq` returns the minimum-norm solution.  The
    resulting individual values are then not physically meaningful, but their
    composition-weighted sum — which is all the model uses — still reproduces
    the mean total energy.
    """
    energies: List[float] = []
    counts: List[Dict[int, int]] = []
    species: set = set()

    for atoms in data:
        energy = _structure_energy(atoms)
        if energy is None:
            continue
        numbers, n = np.unique(np.asarray(atoms.get_atomic_numbers()), return_counts=True)
        composition = {int(z): int(c) for z, c in zip(numbers, n)}
        species.update(composition)
        counts.append(composition)
        energies.append(energy)

    if not energies:
        if warn:
            warnings.warn(
                "No structure in the training data carries a total energy; "
                "per-species reference energies could not be fitted and no "
                "energy offset will be applied.",
                UserWarning,
                stacklevel=2,
            )
        return {}

    ordered_species = sorted(species)
    index = {z: i for i, z in enumerate(ordered_species)}

    matrix = np.zeros((len(energies), len(ordered_species)), dtype=np.float64)
    for row, composition in enumerate(counts):
        for z, c in composition.items():
            matrix[row, index[z]] = c
    target = np.asarray(energies, dtype=np.float64)

    solution, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)

    if warn and rank < len(ordered_species):
        warnings.warn(
            f"Per-species reference-energy fit is rank-deficient (rank {rank} < "
            f"{len(ordered_species)} species): the compositions in the training "
            "data do not determine the individual species energies uniquely. "
            "Using the minimum-norm solution, which still removes the mean "
            "composition-dependent offset.",
            UserWarning,
            stacklevel=2,
        )

    return {z: float(solution[i]) for z, i in index.items()}


def reference_energies_to_tensor(
    reference_energies: ReferenceEnergySpec,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the ``(MAX_ATOMIC_NUMBER,)`` lookup table indexed by atomic number.

    Parameters
    ----------
    reference_energies : Mapping or None
        Reference energies keyed by symbol or atomic number.
    dtype : torch.dtype, optional
        Dtype of the returned tensor.  Defaults to ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(MAX_ATOMIC_NUMBER,)`` with zeros for every species
        that has no reference energy.
    """
    table = torch.zeros(MAX_ATOMIC_NUMBER, dtype=dtype)
    for number, energy in normalize_reference_energies(reference_energies).items():
        table[number] = energy
    return table


def tensor_to_reference_energies(table: torch.Tensor) -> Dict[int, float]:
    """Convert a reference-energy lookup table back to a ``{Z: energy}`` dict.

    Species with a zero entry are omitted (a zero offset is a no-op).

    Parameters
    ----------
    table : torch.Tensor
        Tensor of shape ``(MAX_ATOMIC_NUMBER,)``.

    Returns
    -------
    dict
        Mapping from atomic number to reference energy.
    """
    values = table.detach().cpu()
    return {
        int(z): float(values[z])
        for z in torch.nonzero(values, as_tuple=False).flatten().tolist()
    }


def format_reference_energies(reference_energies: Mapping[int, float]) -> str:
    """Render ``{Z: energy}`` as a compact ``"Cu: -3.500, O: -4.200"`` string.

    Parameters
    ----------
    reference_energies : Mapping[int, float]
        Reference energies keyed by atomic number.

    Returns
    -------
    str
        Human-readable one-line summary (``"none"`` when empty).
    """
    if not reference_energies:
        return "none"
    return ", ".join(
        f"{ASE_CHEMICAL_SYMBOLS[z]}: {reference_energies[z]:.3f}"
        for z in sorted(reference_energies)
    )
