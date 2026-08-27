"""Tests for the standalone ``relax`` API."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.constraints import FixAtoms

from agedi.api import relax

emt = pytest.importorskip("ase.calculators.emt")

L = 12.0


class EMTRegressor(torch.nn.Module):
    """Stand-in for a trained forces head: exact EMT forces.

    Mirrors :class:`~agedi.models.RegressorModel` closely enough for the API —
    predictions are registered as node attributes and masked atoms get zero
    force.
    """

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        pos = batch.pos.detach().cpu().numpy().astype(float)
        forces = np.zeros_like(pos)
        offset = 0
        for i in range(batch.num_graphs):
            sel = (batch.batch == i).cpu().numpy()
            n = int(sel.sum())
            atoms = Atoms(
                numbers=batch.x[torch.tensor(sel)].cpu().numpy(),
                positions=pos[sel],
                cell=np.eye(3) * L,
                pbc=True,
            )
            atoms.calc = emt.EMT()
            forces[sel] = atoms.get_forces()
            offset += n
        f = torch.tensor(forces, dtype=batch.pos.dtype)
        if "mask" in batch:
            f[batch.positions_mask] = 0.0
        batch.add_batch_attr("forces_prediction", f, type="node")
        return batch


class FakeModel(torch.nn.Module):
    """Minimal stand-in for :class:`~agedi.Agedi` as ``relax`` uses it."""

    def __init__(self, regressor=None):
        super().__init__()
        self.regressor_model = regressor
        self.score_model = None


def _structure(displacement=0.25, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.array(
        [[0.0, 0.0, 0.0], [2.6, 0.0, 0.0], [0.0, 2.6, 0.0], [2.6, 2.6, 0.0]]
    ) + rng.normal(scale=displacement, size=(4, 3))
    pos[:, 0] += 11.98 - pos[0, 0]  # straddle the periodic boundary
    return Atoms("Cu4", positions=pos % L, cell=np.eye(3) * L, pbc=True)


def _fmax(atoms):
    a = atoms.copy()
    a.calc = emt.EMT()
    return float(np.abs(a.get_forces()).max())


def _energy(atoms):
    a = atoms.copy()
    a.calc = emt.EMT()
    return float(a.get_potential_energy())


@pytest.fixture
def model():
    return FakeModel(EMTRegressor())


class TestRelaxBasics:
    def test_returns_one_structure_per_input(self, model):
        structures = [_structure(seed=i) for i in range(3)]
        out = relax(model, structures, fmax=0.05, steps=5, cutoff=6.0)
        assert len(out) == 3
        assert all(isinstance(a, Atoms) for a in out)

    def test_attaches_predicted_forces(self, model):
        out = relax(model, [_structure()], fmax=0.05, steps=5, cutoff=6.0)
        assert out[0].calc is not None
        assert out[0].get_forces().shape == (4, 3)

    def test_preserves_composition_and_cell(self, model):
        original = _structure()
        out = relax(model, [original], fmax=0.05, steps=5, cutoff=6.0)[0]
        assert out.get_chemical_symbols() == original.get_chemical_symbols()
        assert np.allclose(out.get_cell(), original.get_cell())

    def test_does_not_mutate_the_input(self, model):
        original = _structure()
        before = original.get_positions().copy()
        relax(model, [original], fmax=0.05, steps=10, cutoff=6.0)
        assert np.allclose(original.get_positions(), before)

    def test_requires_a_regressor(self):
        with pytest.raises(ValueError, match="force-field regressor"):
            relax(FakeModel(None), [_structure()])


class TestRelaxConverges:
    def test_reduces_forces(self, model):
        structure = _structure()
        out = relax(model, [structure], fmax=0.05, steps=60, cutoff=6.0)[0]
        assert _fmax(out) < _fmax(structure)

    def test_reduces_energy(self, model):
        structure = _structure()
        out = relax(model, [structure], fmax=0.05, steps=60, cutoff=6.0)[0]
        assert _energy(out) < _energy(structure)

    def test_stops_early_once_converged(self, model):
        """A structure already inside fmax costs one force evaluation, not `steps`."""
        relaxed = relax(model, [_structure()], fmax=0.05, steps=60, cutoff=6.0)[0]
        assert _fmax(relaxed) < 0.5  # the loosened threshold used below
        model.regressor_model.calls = 0
        relax(model, [relaxed], fmax=0.5, steps=60, cutoff=6.0)
        assert model.regressor_model.calls == 1

    def test_one_force_evaluation_per_step(self, model):
        steps = 5
        model.regressor_model.calls = 0
        relax(model, [_structure(displacement=0.4)], fmax=0.0, steps=steps, cutoff=6.0)
        # 1 initial evaluation + 1 per step; a second call per step would mean
        # the forces are being recomputed at unchanged positions.
        assert model.regressor_model.calls == steps + 1


class TestRelaxConstraints:
    def test_fixed_atoms_do_not_move(self, model):
        structure = _structure()
        structure.set_constraint(FixAtoms(indices=[0, 1]))
        before = structure.get_positions()
        out = relax(model, [structure], fmax=0.01, steps=30, cutoff=6.0)[0]
        assert np.allclose(out.get_positions()[:2], before[:2], atol=1e-5)
        assert not np.allclose(out.get_positions()[2:], before[2:], atol=1e-5)

    def test_constraints_are_carried_to_the_output(self, model):
        structure = _structure()
        structure.set_constraint(FixAtoms(indices=[0]))
        out = relax(model, [structure], fmax=0.05, steps=5, cutoff=6.0)[0]
        assert any(isinstance(c, FixAtoms) for c in out.constraints)


class TestRelaxBatching:
    def test_batched_matches_one_at_a_time(self, model):
        structures = [_structure(seed=i) for i in range(3)]
        batched = relax(model, structures, fmax=0.05, steps=30, cutoff=6.0)
        singly = [
            relax(model, [s], fmax=0.05, steps=30, cutoff=6.0)[0] for s in structures
        ]
        for b, s in zip(batched, singly):
            assert np.allclose(b.get_positions(), s.get_positions(), atol=1e-4)

    def test_converged_structure_is_left_alone(self, model):
        """One structure converging must not perturb it while others relax."""
        converged = relax(
            model, [_structure(seed=1)], fmax=0.01, steps=80, cutoff=6.0
        )[0]
        displaced = _structure(displacement=0.4, seed=2)
        out = relax(
            model, [converged, displaced], fmax=0.05, steps=40, cutoff=6.0
        )
        assert np.allclose(
            out[0].get_positions(), converged.get_positions(), atol=1e-5
        )
        assert not np.allclose(
            out[1].get_positions(), displaced.get_positions(), atol=1e-5
        )

    def test_batch_size_does_not_change_the_result(self, model):
        structures = [_structure(seed=i) for i in range(4)]
        big = relax(model, structures, fmax=0.05, steps=20, batch_size=4, cutoff=6.0)
        small = relax(model, structures, fmax=0.05, steps=20, batch_size=2, cutoff=6.0)
        for a, b in zip(big, small):
            assert np.allclose(a.get_positions(), b.get_positions(), atol=1e-4)


def _cramped_cluster(n, seed, spread=1.5):
    """A cluster of *n* atoms crammed into a small sphere.

    Deliberately starts with almost every pair inside a small cutoff, then
    lets EMT forces push the atoms apart towards their equilibrium spacing —
    so the cutoff-based neighbour count drops sharply over the relaxation,
    which is what exposes the ``fully_connected`` regression below.
    """
    rng = np.random.default_rng(seed)
    center = np.array([L / 2] * 3)
    pos = center + rng.uniform(-spread / 2, spread / 2, size=(n, 3))
    return Atoms("Cu" + str(n), positions=pos, cell=np.eye(3) * L, pbc=True)


class TestRelaxFullyConnected:
    def test_fully_connected_model_does_not_crash(self, model):
        """Regression test: on a model trained with ``fully_connected=True``,
        ``relax()`` must build its graphs with ``fully_connected=True`` too.

        Otherwise ``AtomsGraph.update_graph()`` (called every step) silently
        takes the cutoff-rebuild branch instead of the static fully-connected
        one. As atoms move, per-graph edge counts drift away from what
        ``Batch.from_data_list()`` recorded in ``_slice_dict``/``_inc_dict``,
        and ``Batch.to_data_list()`` eventually slices ``edge_index`` with
        stale offsets, raising e.g. "start (188) + length (240) exceeds
        dimension size (342)". Confirmed to reproduce that crash with these
        exact parameters when ``fully_connected`` is not threaded through
        ``relax()``.
        """
        model.fully_connected = True
        structures = [
            _cramped_cluster(n, seed=i) for i, n in enumerate([8, 12, 16, 20])
        ]

        out = relax(
            model,
            structures,
            fmax=0.0001,
            steps=150,
            cutoff=3.0,
            batch_size=4,
        )

        assert len(out) == 4
        assert all(isinstance(a, Atoms) for a in out)


class TestRelaxTrajectory:
    def test_returns_a_trajectory_per_structure(self, model):
        structures = [_structure(seed=i) for i in range(2)]
        trajs = relax(
            model, structures, fmax=0.05, steps=5, cutoff=6.0, trajectory=True
        )
        assert len(trajs) == 2
        assert all(len(t) >= 2 for t in trajs)

    def test_trajectory_starts_at_the_input_geometry(self, model):
        structure = _structure()
        traj = relax(
            model, [structure], fmax=0.05, steps=5, cutoff=6.0, trajectory=True
        )[0]
        assert np.allclose(
            traj[0].get_positions(), structure.get_positions(), atol=1e-4
        )

    def test_trajectory_ends_where_the_plain_call_ends(self, model):
        structure = _structure()
        traj = relax(
            model, [structure], fmax=0.05, steps=20, cutoff=6.0, trajectory=True
        )[0]
        final = relax(model, [structure], fmax=0.05, steps=20, cutoff=6.0)[0]
        assert np.allclose(traj[-1].get_positions(), final.get_positions(), atol=1e-5)
