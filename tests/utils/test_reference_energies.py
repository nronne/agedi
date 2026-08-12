import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from agedi.utils.reference_energies import (
    MAX_ATOMIC_NUMBER,
    fit_reference_energies,
    format_reference_energies,
    normalize_reference_energies,
    reference_energies_to_tensor,
    tensor_to_reference_energies,
)


REFERENCE = {29: -3.72, 8: -4.95}


def _structure(n_cu: int, n_o: int, energy=None, noise: float = 0.0):
    atoms = Atoms("Cu" * n_cu + "O" * n_o, positions=np.zeros((n_cu + n_o, 3)))
    if energy is None:
        energy = n_cu * REFERENCE[29] + n_o * REFERENCE[8] + noise
    atoms.calc = SinglePointCalculator(atoms, energy=float(energy))
    return atoms


def test_fit_recovers_exact_reference_energies():
    data = [_structure(4, 1), _structure(2, 3), _structure(5, 2), _structure(1, 1)]

    fitted = fit_reference_energies(data)

    assert set(fitted) == {29, 8}
    assert fitted[29] == pytest.approx(REFERENCE[29])
    assert fitted[8] == pytest.approx(REFERENCE[8])


def test_fit_is_robust_to_noise():
    rng = np.random.default_rng(0)
    data = [
        _structure(int(rng.integers(1, 6)), int(rng.integers(1, 6)), noise=rng.normal(0, 0.02))
        for _ in range(40)
    ]

    fitted = fit_reference_energies(data)

    assert fitted[29] == pytest.approx(REFERENCE[29], abs=0.02)
    assert fitted[8] == pytest.approx(REFERENCE[8], abs=0.02)


def test_fit_skips_structures_without_energy():
    data = [_structure(2, 1), Atoms("Cu2O", positions=np.zeros((3, 3))), _structure(1, 2)]

    fitted = fit_reference_energies(data)

    assert fitted[29] == pytest.approx(REFERENCE[29])
    assert fitted[8] == pytest.approx(REFERENCE[8])


def test_fit_without_any_energy_warns_and_returns_empty():
    data = [Atoms("Cu2O", positions=np.zeros((3, 3)))]

    with pytest.warns(UserWarning, match="does not|no structure|energy"):
        fitted = fit_reference_energies(data)

    assert fitted == {}


def test_fit_warns_when_rank_deficient():
    # A single composition cannot determine the two species energies uniquely.
    data = [_structure(2, 1), _structure(2, 1, energy=-12.0)]

    with pytest.warns(UserWarning, match="rank-deficient"):
        fitted = fit_reference_energies(data)

    # The minimum-norm solution still reproduces the mean total energy.
    mean_energy = np.mean([a.get_potential_energy() for a in data])
    assert 2 * fitted[29] + fitted[8] == pytest.approx(mean_energy)


def test_normalize_accepts_symbols_and_numbers():
    assert normalize_reference_energies({"Cu": -3.72, 8: -4.95}) == {29: -3.72, 8: -4.95}
    assert normalize_reference_energies(None) == {}


@pytest.mark.parametrize(
    "spec",
    [
        {"Xx": -1.0},           # unknown symbol
        {0: -1.0},              # atomic number out of range
        {MAX_ATOMIC_NUMBER: 1.0},
        {(1, 2): -1.0},         # unusable key type
        {"Cu": "not-a-number"},
        [("Cu", -1.0)],         # not a mapping
    ],
)
def test_normalize_rejects_invalid_specs(spec):
    with pytest.raises(ValueError):
        normalize_reference_energies(spec)


def test_tensor_round_trip():
    table = reference_energies_to_tensor({"Cu": -3.72, "O": -4.95})

    assert table.shape == (MAX_ATOMIC_NUMBER,)
    assert table[29] == pytest.approx(-3.72)
    assert torch.count_nonzero(table) == 2

    recovered = tensor_to_reference_energies(table)
    assert recovered.keys() == {29, 8}
    assert recovered[8] == pytest.approx(-4.95)


def test_format_reference_energies():
    assert format_reference_energies({}) == "none"
    assert format_reference_energies({29: -3.72, 8: -4.95}) == "O: -4.950, Cu: -3.720"
