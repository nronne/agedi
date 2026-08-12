"""Regression tests for the L-BFGS relaxation used by ``max_extra_steps``.

The regressor is replaced by an exact analytic force field, so any failure to
reach the minimum is a defect in the optimiser rather than model error.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from torch_geometric.data import Batch

from agedi.data import AtomsGraph
from agedi.diffusion.guidance import (
    BatchedLBFGSStepSizer,
    LBFGSStepSizer,
    max_force_per_graph,
    minimum_image,
    post_diffusion_relaxation_step,
)

L = 10.0
K = 4.0  # eV/Å², spring constant of the toy potential


def _mic(d: np.ndarray) -> np.ndarray:
    """Minimum-image displacement in the cubic cell of side ``L``."""
    return d - L * np.round(d / L)


class HarmonicField:
    """Exact PES: every atom is bound to its own site by a spring.

    ``U = 0.5 K Σ |mic(r_i - r0_i)|²`` — periodic, conservative, and with a
    single known minimum at ``r0``, so the optimiser has nowhere to hide.
    """

    def __init__(self, sites: np.ndarray):
        self.sites = sites
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1
        pos = batch.pos.detach().cpu().numpy().astype(float)
        sites = np.tile(self.sites, (batch.num_graphs, 1))[: len(pos)]
        d = _mic(pos - sites)
        batch["forces_prediction"] = torch.tensor(-K * d, dtype=batch.pos.dtype)
        return batch

    def energy(self, batch) -> float:
        pos = batch.pos.detach().cpu().numpy().astype(float)
        sites = np.tile(self.sites, (batch.num_graphs, 1))[: len(pos)]
        return float(0.5 * K * (_mic(pos - sites) ** 2).sum())


def _sites(on_boundary: bool) -> np.ndarray:
    """Four equilibrium sites, either straddling a cell face or mid-cell."""
    base = np.array([[0.0, 2.0, 2.0], [0.0, 5.0, 2.0], [0.0, 2.0, 5.0], [0.0, 5.0, 5.0]])
    base[:, 0] = 0.02 if on_boundary else 5.0
    return base


def _build(on_boundary: bool, n_structures: int = 1, displacement: float = 0.35):
    """Batch of Cu4 structures displaced from the minimum of :class:`HarmonicField`."""
    sites = _sites(on_boundary)
    rng = np.random.default_rng(0)
    graphs = []
    for _ in range(n_structures):
        pos = (sites + rng.normal(scale=displacement, size=sites.shape)) % L
        atoms = Atoms("Cu4", positions=pos, cell=np.eye(3) * L, pbc=True)
        graphs.append(AtomsGraph.from_atoms(atoms, cutoff=6.0))
    batch = Batch.from_data_list(graphs)
    batch.update_graph()
    return batch, HarmonicField(sites)


def _relax(batch, field, steps=60, sizer=None):
    """Drive ``post_diffusion_relaxation_step`` the way ``_sample_batch`` does."""
    if sizer is None:
        sizer = BatchedLBFGSStepSizer(batch_size=batch.num_graphs)
    batch = field(batch)
    for _ in range(steps):
        batch = post_diffusion_relaxation_step(
            batch, field, sizer, forces=batch.forces_prediction
        )
        batch = field(batch)
    return batch


class TestMinimumImage:
    """The displacement correction that keeps the L-BFGS history sane."""

    def test_removes_lattice_jump(self):
        cell = torch.eye(3) * L
        pbc = torch.tensor([True, True, True])
        d = torch.tensor([[9.8, 0.0, 0.0]])
        assert torch.allclose(
            minimum_image(d, cell, pbc), torch.tensor([[-0.2, 0.0, 0.0]]), atol=1e-5
        )

    def test_leaves_small_displacements_untouched(self):
        cell = torch.eye(3) * L
        pbc = torch.tensor([True, True, True])
        d = torch.tensor([[0.1, -0.2, 0.05]])
        assert torch.allclose(minimum_image(d, cell, pbc), d, atol=1e-6)

    def test_non_periodic_directions_are_not_wrapped(self):
        cell = torch.eye(3) * L
        pbc = torch.tensor([True, False, False])
        d = torch.tensor([[9.8, 9.8, 0.0]])
        out = minimum_image(d, cell, pbc)
        assert out[0, 0] == pytest.approx(-0.2, abs=1e-5)
        assert out[0, 1] == pytest.approx(9.8, abs=1e-5)

    def test_no_cell_is_a_no_op(self):
        d = torch.tensor([[9.8, 0.0, 0.0]])
        assert torch.allclose(minimum_image(d, None, None), d)


class TestLBFGSHistoryAcrossPeriodicBoundary:
    """The mechanism this module exists for.

    ``wrap_positions`` maps every position back into the cell after each step,
    so an atom crossing a face reappears a full lattice vector away.  The
    displacement the step sizer reconstructs by differencing stored positions
    must not contain that jump — one such history pair is enough to send the
    search direction somewhere unrelated to the forces.
    """

    def test_a_wrap_actually_happens_in_this_fixture(self):
        """Guard the guard: the test below is only meaningful if atoms wrap."""
        batch, field = _build(on_boundary=True)
        before = batch.pos.clone()
        _relax(batch, field, steps=12)
        # Some atom must have crossed the face for the regression to be exercised.
        assert (batch.pos - before).abs().max() > L / 2

    def test_history_contains_no_lattice_jump(self):
        batch, field = _build(on_boundary=True)
        sizer = BatchedLBFGSStepSizer(batch_size=batch.num_graphs)
        _relax(batch, field, steps=12, sizer=sizer)

        stored = sizer.step_sizers[0].s_list
        assert stored, "no history was accumulated"
        longest = max(float(s.abs().max()) for s in stored)
        # Every step is capped at maxstep=0.2 A; anything near L is a wrap that
        # leaked into the history.
        assert longest < L / 2


class TestRelaxationOnEMT:
    """End-to-end behaviour on a real PES, with atoms on the cell face."""

    @staticmethod
    def _setup():
        emt = pytest.importorskip("ase.calculators.emt")

        rng = np.random.default_rng(0)
        pos = np.array(
            [[0.0, 0.0, 0.0], [2.6, 0.0, 0.0], [0.0, 2.6, 0.0], [2.6, 2.6, 0.0],
             [1.3, 1.3, 2.3], [3.9, 1.3, 2.3], [1.3, 3.9, 2.3], [3.9, 3.9, 2.3]]
        ) + rng.normal(scale=0.15, size=(8, 3))
        pos[:, 0] += 9.98 - pos[0, 0]  # first atom sits on the cell face
        pos %= L

        def make_atoms(p):
            a = Atoms("Cu8", positions=p, cell=np.eye(3) * L, pbc=True)
            a.calc = emt.EMT()
            return a

        def regressor(batch):
            a = make_atoms(batch.pos.detach().cpu().numpy().astype(float))
            batch["forces_prediction"] = torch.tensor(
                a.get_forces(), dtype=batch.pos.dtype
            )
            return batch

        return pos, make_atoms, regressor

    @classmethod
    def _relax_emt(cls, steps=40):
        pos, make_atoms, regressor = cls._setup()
        batch = Batch.from_data_list(
            [AtomsGraph.from_atoms(make_atoms(pos), cutoff=6.0)]
        )
        batch.update_graph()
        sizer = BatchedLBFGSStepSizer(batch_size=1)
        batch = regressor(batch)
        for _ in range(steps):
            batch = post_diffusion_relaxation_step(
                batch, regressor, sizer, forces=batch.forces_prediction
            )
            batch = regressor(batch)
        final = make_atoms(batch.pos.detach().cpu().numpy().astype(float))
        return make_atoms(pos), final

    def test_energy_decreases(self):
        """Pre-fix this rose by ~4.8 eV instead of falling."""
        start, final = self._relax_emt()
        assert final.get_potential_energy() < start.get_potential_energy()

    def test_forces_decrease(self):
        start, final = self._relax_emt()
        f_start = np.abs(start.get_forces()).max()
        f_final = np.abs(final.get_forces()).max()
        assert f_final < 0.5 * f_start

    def test_matches_ase_lbfgs(self):
        """End-to-end parity with ``ase.optimize.LBFGS``."""
        from ase.optimize import LBFGS as ASELBFGS

        start, final = self._relax_emt()
        reference = start
        ASELBFGS(reference, logfile=None).run(fmax=1e-3, steps=40)

        # Tolerance is set by graph positions being float32, not by the
        # optimiser: before the minimum-image fix the gap here was ~2 eV.
        assert final.get_potential_energy() == pytest.approx(
            reference.get_potential_energy(), abs=5e-3
        )


class TestForceEvaluationCount:
    """Forces are evaluated once per step, not twice."""

    def test_supplied_forces_skip_the_regressor_call(self):
        batch, field = _build(on_boundary=False)
        batch = field(batch)
        calls_before = field.calls
        post_diffusion_relaxation_step(
            batch, field, BatchedLBFGSStepSizer(batch_size=1),
            forces=batch.forces_prediction,
        )
        assert field.calls == calls_before

    def test_omitted_forces_still_evaluate(self):
        batch, field = _build(on_boundary=False)
        calls_before = field.calls
        post_diffusion_relaxation_step(
            batch, field, BatchedLBFGSStepSizer(batch_size=1)
        )
        assert field.calls == calls_before + 1

    def test_relaxation_loop_uses_one_evaluation_per_step(self):
        steps = 10
        batch, field = _build(on_boundary=False)
        _relax(batch, field, steps=steps)
        # 1 initial evaluation + 1 per step.
        assert field.calls == steps + 1


class TestPerStructureConvergence:
    """A converged structure sits out the rest of the batch's relaxation."""

    def test_inactive_structures_do_not_move(self):
        batch, field = _build(on_boundary=False, n_structures=2)
        batch = field(batch)
        before = batch.pos.clone()
        active = torch.tensor([True, False])
        batch = post_diffusion_relaxation_step(
            batch,
            field,
            BatchedLBFGSStepSizer(batch_size=2),
            forces=batch.forces_prediction,
            active=active,
        )
        moved_0 = (batch.pos[batch.batch == 0] - before[batch.batch == 0]).abs().max()
        moved_1 = (batch.pos[batch.batch == 1] - before[batch.batch == 1]).abs().max()
        assert moved_0 > 0
        assert moved_1 == 0

    def test_inactive_structure_history_is_untouched(self):
        batch, field = _build(on_boundary=False, n_structures=2)
        batch = field(batch)
        sizer = BatchedLBFGSStepSizer(batch_size=2)
        post_diffusion_relaxation_step(
            batch, field, sizer,
            forces=batch.forces_prediction,
            active=torch.tensor([True, False]),
        )
        assert sizer.step_sizers[0].prev_pos is not None
        assert sizer.step_sizers[1].prev_pos is None


class TestMaxForcePerGraph:
    """Per-structure convergence measure."""

    def test_reports_each_graph_separately(self):
        forces = torch.tensor(
            [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]]
        )
        batch_idx = torch.tensor([0, 0, 1])
        out = max_force_per_graph(forces, batch_idx, 2)
        assert out[0] == pytest.approx(3.0)
        assert out[1] == pytest.approx(0.5)


class TestLBFGSStepSizerDirection:
    """The very first step, with no history, must follow the forces."""

    def test_first_step_is_along_the_forces(self):
        sizer = LBFGSStepSizer()
        pos = torch.zeros(2, 3)
        forces = torch.tensor([[1.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
        step = sizer.compute_step(pos, forces)
        assert torch.all(step * forces >= 0)
        assert (step * forces).sum() > 0
