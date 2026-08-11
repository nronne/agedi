"""Extended tests for Agedi: LBFGSStepSizer, forward_step, reverse_step,
configure_optimizers, regressor-related paths, and regressor_training property."""
from types import SimpleNamespace

import torch
import pytest

from agedi.diffusion.agedi import Agedi, LBFGSStepSizer, BatchedLBFGSStepSizer
from agedi.data import AtomsGraph


# ── LBFGSStepSizer ───────────────────────────────────────────────────────────

class TestLBFGSStepSizer:
    def _make(self):
        return LBFGSStepSizer(memory_size=5)

    def test_first_step_uses_scaling(self):
        sizer = self._make()
        pos = torch.zeros((4, 3))
        forces = torch.ones((4, 3))
        step = sizer.compute_step(pos, forces)
        assert step.shape == forces.shape

    def test_second_step_uses_lbfgs(self):
        sizer = self._make()
        pos = torch.zeros((4, 3))
        forces = torch.ones((4, 3))
        sizer.compute_step(pos, forces)
        pos2 = pos + 0.01 * forces
        forces2 = forces * 0.9
        step = sizer.compute_step(pos2, forces2)
        assert step.shape == forces2.shape

    def test_reset_clears_memory(self):
        sizer = self._make()
        pos = torch.zeros((4, 3))
        forces = torch.ones((4, 3))
        sizer.compute_step(pos, forces)
        sizer.reset()
        assert sizer.prev_pos is None
        assert len(sizer.s_list) == 0


class TestBatchedLBFGSStepSizer:
    def test_compute_step_correct_shape(self):
        sizer = BatchedLBFGSStepSizer(batch_size=2, memory_size=3)
        pos = torch.zeros((6, 3))
        forces = torch.ones((6, 3))
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1])
        step = sizer.compute_step(pos, forces, batch_idx)
        assert step.shape == pos.shape

    def test_reset_delegates(self):
        sizer = BatchedLBFGSStepSizer(batch_size=2, memory_size=3)
        pos = torch.zeros((4, 3))
        forces = torch.ones((4, 3))
        batch_idx = torch.tensor([0, 0, 1, 1])
        sizer.compute_step(pos, forces, batch_idx)
        sizer.reset()
        for inner in sizer.step_sizers:
            assert inner.prev_pos is None


# ── forward_step / reverse_step ──────────────────────────────────────────────

def test_forward_step_adds_noise(diffusion, batch):
    diffusion.sample_time(batch)
    pos_before = batch.pos.clone()
    out = diffusion.forward_step(batch)
    assert not torch.allclose(out.pos, pos_before)


def test_reverse_step_changes_positions(diffusion, batch):
    diffusion.score_model.sample_mode()
    diffusion.sample_time(batch)
    diffusion.forward_step(batch)
    pos_noised = batch.pos.clone()
    diffusion.reverse_step(batch, delta_t=torch.tensor(0.001), force_field_guidance=0.0)
    assert not torch.allclose(batch.pos, pos_noised)


# ── configure_optimizers ─────────────────────────────────────────────────────

def test_configure_optimizers_returns_optimizer_and_scheduler(diffusion):
    cfg = diffusion.configure_optimizers()
    assert "optimizer" in cfg
    assert "lr_scheduler" in cfg


def test_configure_optimizers_falls_back_when_validation_empty(diffusion):
    diffusion._trainer = SimpleNamespace(datamodule=SimpleNamespace(val_idx=[]))
    cfg = diffusion.configure_optimizers()
    assert cfg["monitor"] == "train_loss_epoch"


# ── setup ────────────────────────────────────────────────────────────────────

def test_setup_puts_score_model_in_training_mode(diffusion):
    diffusion.setup()
    assert not diffusion.score_model.sample


# ── regressor_training property ──────────────────────────────────────────────

def test_regressor_training_returns_false_without_regressor(diffusion):
    assert diffusion.regressor_training is False


def test_regressor_training_setter_ignored_without_regressor(diffusion):
    diffusion.regressor_training = True
    assert diffusion.regressor_training is False


# ── regressor_loss raises without regressor ──────────────────────────────────

def test_regressor_loss_raises_without_regressor(diffusion, batch):
    with pytest.raises(ValueError, match="Regressor model is not defined"):
        diffusion.regressor_loss(batch, None)


# ── sample with > batch_size ─────────────────────────────────────────────────

def test_sample_split_batches(diffusion):
    out = diffusion.sample(
        5,
        batch_size=2,
        steps=3,
        atomic_numbers=[6, 8],
        cell=torch.diag(torch.tensor([8.0, 8.0, 8.0])),
        property={"property": 1.0},
    )
    assert len(out) == 5
    assert all(isinstance(g, AtomsGraph) for g in out)


def test_sample_prints_timing_breakdown_when_print_timings_enabled(diffusion, capsys):
    diffusion.sample(
        2,
        steps=3,
        atomic_numbers=[6, 8],
        cell=torch.diag(torch.tensor([8.0, 8.0, 8.0])),
        property={"property": 1.0},
        print_timings=True,
    )
    captured = capsys.readouterr()
    assert "Sampling timing breakdown:" in captured.out
    assert "neighbor list updates" in captured.out
    assert "total neighbor list" in captured.out


def test_sample_no_timing_breakdown_without_flag(diffusion, capsys):
    diffusion.sample(
        2,
        steps=3,
        atomic_numbers=[6, 8],
        cell=torch.diag(torch.tensor([8.0, 8.0, 8.0])),
        property={"property": 1.0},
        print_timings=False,
    )
    captured = capsys.readouterr()
    assert "Sampling timing breakdown:" not in captured.out


# ── Combined-loss training (regressor present) ───────────────────────────────

@pytest.fixture
def diffusion_with_regressor(diffusion):
    """Agedi fixture that includes a Forces regressor."""
    from agedi.models.regressor import RegressorModel
    from agedi.models.schnetpack.regressor_heads import Forces

    feature_size = 64
    forces_head = Forces(input_dim_scalar=feature_size, input_dim_vector=feature_size)
    regressor = RegressorModel(
        translator=diffusion.score_model.translator,
        representation=diffusion.score_model.representation,
        heads=[forces_head],
    )
    diffusion.regressor_model = regressor
    return diffusion


def test_configure_optimizers_with_regressor_deduplicates_params(diffusion_with_regressor):
    """Optimizer should cover all unique parameters exactly once."""
    cfg = diffusion_with_regressor.configure_optimizers()
    assert "optimizer" in cfg
    opt_params = {id(p) for p in cfg["optimizer"].param_groups[0]["params"]}

    score_params = {id(p) for p in diffusion_with_regressor.score_model.parameters()}
    reg_params = {id(p) for p in diffusion_with_regressor.regressor_model.parameters()}
    all_unique = score_params | reg_params

    assert opt_params == all_unique


def test_loss_combines_diffusion_and_regressor_when_forces_present(diffusion_with_regressor, batch):
    """loss() total must equal diffusion_loss + weight * regressor_loss (exact identity)."""
    import numpy as np
    diffusion_with_regressor.score_model.training_mode()
    batch.forces = torch.randn_like(batch.pos)

    # Save RNG state so loss() and diffusion_loss() draw the same random noise/time.
    torch_rng_state = torch.random.get_rng_state()
    numpy_rng_state = np.random.get_state()

    losses = diffusion_with_regressor.loss(batch, 0)

    torch.random.set_rng_state(torch_rng_state)
    np.random.set_state(numpy_rng_state)
    diff_only = diffusion_with_regressor.diffusion_loss(batch, 0)

    assert "regressor_loss" in losses, "regressor_loss key should be present"
    assert "loss" in losses

    expected_total = (
        diff_only["loss"]
        + diffusion_with_regressor.regressor_loss_weight * losses["regressor_loss"]
    )
    assert torch.isclose(losses["loss"], expected_total), (
        "Combined loss must equal diffusion_loss + weight * regressor_loss"
    )


def test_loss_without_forces_is_diffusion_only(diffusion_with_regressor, batch):
    """loss() should fall back to diffusion loss when forces are absent."""
    diffusion_with_regressor.score_model.training_mode()
    if hasattr(batch, "forces"):
        del batch.forces

    losses = diffusion_with_regressor.loss(batch, 0)
    assert "regressor_loss" not in losses, "regressor_loss key should not be present without forces"


def test_loss_respects_regressor_loss_weight(diffusion_with_regressor, batch):
    """regressor_loss_weight should scale the contribution of the regressor loss."""
    import numpy as np
    diffusion_with_regressor.score_model.training_mode()
    batch.forces = torch.randn_like(batch.pos)

    diffusion_with_regressor.regressor_loss_weight = 0.0

    # Save RNG state so loss() and diffusion_loss() use the same sampled randomness.
    torch_rng_state = torch.random.get_rng_state()
    numpy_rng_state = np.random.get_state()

    losses_zero = diffusion_with_regressor.loss(batch, 0)

    torch.random.set_rng_state(torch_rng_state)
    np.random.set_state(numpy_rng_state)
    diffusion_only = diffusion_with_regressor.diffusion_loss(batch, 0)

    # The regressor contribution should still be returned, but weighted out of total loss.
    assert "regressor_loss" in losses_zero
    assert torch.isclose(losses_zero["loss"], diffusion_only["loss"]), (
        "With weight=0, combined loss should equal the diffusion-only loss"
    )


class TestMatchesASELBFGS:
    """The step sizer must reproduce ``ase.optimize.LBFGS`` step for step."""

    @staticmethod
    def _cluster():
        from ase.calculators.emt import EMT
        from ase.cluster import Icosahedron

        atoms = Icosahedron("Cu", 2)
        atoms.rattle(0.2, seed=1)
        atoms.calc = EMT()
        return atoms

    def test_trajectory_matches_ase(self):
        """20 steps on an EMT cluster must agree with ASE to float64 precision."""
        import numpy as np
        from ase.optimize import LBFGS

        n_steps = 20

        ase_atoms = self._cluster()
        opt = LBFGS(ase_atoms, logfile=None)
        ase_positions = []
        for _ in range(n_steps):
            opt.step()
            ase_positions.append(ase_atoms.get_positions().copy())

        our_atoms = self._cluster()
        sizer = LBFGSStepSizer()
        for i, ase_pos in enumerate(ase_positions):
            step = sizer.compute_step(
                torch.tensor(our_atoms.get_positions(), dtype=torch.float64),
                torch.tensor(our_atoms.get_forces(), dtype=torch.float64),
            ).numpy()
            our_atoms.set_positions(our_atoms.get_positions() + step)
            assert np.allclose(our_atoms.get_positions(), ase_pos, atol=1e-10), (
                f"diverged from ASE at step {i}"
            )

    def test_defaults_match_ase(self):
        """Defaults must track ASE's, or trajectories silently drift apart."""
        from ase.optimize import LBFGS

        sizer = LBFGSStepSizer()
        assert sizer.maxstep == LBFGS.defaults["maxstep"] == 0.2
        assert sizer.memory_size == 100
        assert sizer.damping == 1.0
        assert sizer.H0 == 1.0 / 70.0

    def test_h0_is_constant(self):
        """H0 is a fixed seed in ASE; an adaptive one makes the step oscillate."""
        sizer = LBFGSStepSizer()
        pos = torch.zeros((4, 3), dtype=torch.float64)
        forces = torch.ones((4, 3), dtype=torch.float64)
        h0_before = sizer.H0
        for k in range(5):
            sizer.compute_step(pos + 0.01 * k * forces, forces * (1.0 - 0.1 * k))
        assert sizer.H0 == h0_before

    def test_step_limit_preserves_direction(self):
        """maxstep scales the whole step, so the direction is unchanged."""
        sizer = LBFGSStepSizer(maxstep=0.05)
        dr = torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]],
            dtype=torch.float64,
        )
        limited = sizer.determine_step(dr.clone())

        assert torch.isclose(
            torch.norm(limited, dim=1).max(), torch.tensor(0.05, dtype=torch.float64)
        )
        # Every atom scaled by the same factor => directions preserved.
        ratios = torch.norm(limited, dim=1) / torch.norm(dr, dim=1)
        assert torch.allclose(ratios, ratios[0])

    def test_short_step_is_not_scaled_up(self):
        """A step below maxstep is left alone rather than stretched to the limit."""
        sizer = LBFGSStepSizer(maxstep=1.0)
        dr = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float64)
        assert torch.allclose(sizer.determine_step(dr.clone()), dr)


class TestBatchedStepIsolation:
    """Each structure in a batch must be optimised independently."""

    def test_per_graph_steps_do_not_leak(self):
        """A graph's step must land on its own atoms, not a neighbour's rows."""
        sizer = BatchedLBFGSStepSizer(batch_size=3)
        pos = torch.zeros((6, 3), dtype=torch.float64)
        forces = torch.zeros((6, 3), dtype=torch.float64)
        # Graph 1 has no atoms; graphs 0 and 2 have distinct force directions.
        batch_idx = torch.tensor([0, 0, 2, 2, 2, 2])
        forces[batch_idx == 0] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        forces[batch_idx == 2] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

        step = sizer.compute_step(pos, forces, batch_idx)

        # Graph 0 moves only in x, graph 2 only in z.  Re-indexing results by
        # position rather than graph id would put graph 2's step on graph 0.
        assert torch.all(step[batch_idx == 0][:, 1:] == 0)
        assert torch.all(step[batch_idx == 0][:, 0] > 0)
        assert torch.all(step[batch_idx == 2][:, :2] == 0)
        assert torch.all(step[batch_idx == 2][:, 2] > 0)

    def test_maxstep_applied_per_structure(self):
        """The step cap is per structure, as it would be relaxing each alone."""
        sizer = BatchedLBFGSStepSizer(batch_size=2, maxstep=0.1)
        pos = torch.zeros((4, 3), dtype=torch.float64)
        forces = torch.zeros((4, 3), dtype=torch.float64)
        batch_idx = torch.tensor([0, 0, 1, 1])
        forces[batch_idx == 0] = torch.tensor([1000.0, 0.0, 0.0], dtype=torch.float64)
        forces[batch_idx == 1] = torch.tensor([1e-6, 0.0, 0.0], dtype=torch.float64)

        step = sizer.compute_step(pos, forces, batch_idx)

        # The huge-force structure is capped; the tiny-force one is untouched
        # rather than being dragged along by its batch-mate.
        assert torch.isclose(
            torch.norm(step[batch_idx == 0], dim=1).max(),
            torch.tensor(0.1, dtype=torch.float64),
        )
        assert torch.norm(step[batch_idx == 1], dim=1).max() < 1e-7
