"""Tests for separable Sampler classes."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from agedi.data import AtomsGraph
from agedi.diffusion.noisers import CellPositions
from agedi.diffusion.samplers import (
    EulerMaruyamaSampler,
    ForcefieldCorrectorSampler,
    HeunODESampler,
    HeunSampler,
    PredictorCorrectorSampler,
    ProbabilityFlowODESampler,
    Sampler,
)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_has_all_builtin_aliases():
    """All built-in string aliases must be present in the registry."""
    for alias in ("em", "pc", "heun", "ddim", "heun_ode", "ffpc"):
        assert alias in Sampler._registry, f"alias {alias!r} missing from registry"


def test_register_and_lookup_custom_sampler():
    """Custom sampler can be registered and looked up by name."""

    class _CustomSampler(EulerMaruyamaSampler):
        pass

    Sampler.register(
        "_test_custom",
        lambda score_fn, noisers, **kw: _CustomSampler(score_fn, noisers),
    )
    assert "_test_custom" in Sampler._registry
    # Clean up so other tests are not affected.
    del Sampler._registry["_test_custom"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_samplers(diffusion):
    """Return a dict of sampler-name → Sampler instance for the given diffusion."""
    score_fn = diffusion.score_model
    noisers = diffusion.noisers
    return {
        "em": EulerMaruyamaSampler(score_fn, noisers),
        "pc": PredictorCorrectorSampler(score_fn, noisers, corrector_steps=1),
        "heun": HeunSampler(score_fn, noisers),
        "ddim": ProbabilityFlowODESampler(score_fn, noisers),
        "heun_ode": HeunODESampler(score_fn, noisers),
    }


def _single_step(diffusion, sampler_instance, batch, last=False):
    """Run a single sampler step with a freshly set time."""
    diffusion.score_model.sample_mode()
    diffusion.sample_time(batch)
    diffusion.forward_step(batch)
    ts = torch.linspace(1.0, 1e-3, 5, device=batch.pos.device)
    dt = ts[0] - ts[1]
    batch.add_batch_attr("time", ts[0].repeat(batch.x.shape[0], 1), type="node")
    batch.update_graph()
    return sampler_instance.step(batch, dt, last=last)


# ---------------------------------------------------------------------------
# EulerMaruyamaSampler
# ---------------------------------------------------------------------------


class TestEulerMaruyamaSampler:
    def test_em_step_changes_positions(self, diffusion, batch):
        """EM sampler step should change atom positions."""
        pos_before = batch.pos.clone()
        sampler = EulerMaruyamaSampler(diffusion.score_model, diffusion.noisers)
        _single_step(diffusion, sampler, batch)
        assert not torch.allclose(batch.pos, pos_before)

    def test_em_sampler_returns_valid_graph(self, diffusion, batch):
        """EM step should return a batch with a valid neighbour list."""
        sampler = EulerMaruyamaSampler(diffusion.score_model, diffusion.noisers)
        out = _single_step(diffusion, sampler, batch)
        assert out.pos is not None
        assert out.edge_index is not None

    def test_em_last_true_is_deterministic(self, diffusion, batch):
        """With last=True, EM step should produce the same result on two calls."""
        diffusion.score_model.sample_mode()
        diffusion.sample_time(batch)
        diffusion.forward_step(batch)
        ts = torch.linspace(1.0, 1e-3, 5, device=batch.pos.device)
        dt = ts[0] - ts[1]
        batch.add_batch_attr("time", ts[0].repeat(batch.x.shape[0], 1), type="node")
        batch.update_graph()

        sampler = EulerMaruyamaSampler(diffusion.score_model, diffusion.noisers)

        pos_saved = batch.pos.clone()
        scores_saved = {k + "_score": batch.get(k + "_score") for n in diffusion.noisers
                        for k in [n.key] if batch.get(k + "_score") is not None}

        torch.manual_seed(0)
        out_a = sampler.step(batch, dt, last=True)
        pos_a = out_a.pos.clone()

        # Restore state
        batch.pos = pos_saved
        for k, v in scores_saved.items():
            if v is not None:
                batch[k] = v
        batch.update_graph()

        torch.manual_seed(0)
        out_b = sampler.step(batch, dt, last=True)
        pos_b = out_b.pos.clone()

        assert torch.allclose(pos_a, pos_b)


# ---------------------------------------------------------------------------
# PredictorCorrectorSampler
# ---------------------------------------------------------------------------


class TestPredictorCorrectorSampler:
    def test_pc_changes_positions(self, diffusion, batch):
        """PC sampler step should change positions."""
        pos_before = batch.pos.clone()
        sampler = PredictorCorrectorSampler(
            diffusion.score_model, diffusion.noisers, corrector_steps=1
        )
        _single_step(diffusion, sampler, batch)
        assert not torch.allclose(batch.pos, pos_before)

    def test_pc_corrector_steps_0_matches_em_output_count(self, diffusion):
        """PC with corrector_steps=0 should return the right number of structures."""
        sampler = PredictorCorrectorSampler(
            diffusion.score_model, diffusion.noisers, corrector_steps=0
        )
        out = diffusion.sample(
            2,
            steps=3,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            sampler=sampler,
        )
        assert len(out) == 2

    def test_pc_corrector_changes_vs_em(self, diffusion):
        """PC with corrector_steps>0 should give different result from plain EM."""
        torch.manual_seed(7)
        out_em = diffusion.sample(
            1,
            steps=4,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            sampler="em",
        )
        torch.manual_seed(7)
        out_pc = diffusion.sample(
            1,
            steps=4,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            sampler=PredictorCorrectorSampler(
                diffusion.score_model, diffusion.noisers, corrector_steps=2
            ),
        )
        assert not torch.allclose(out_em[0].pos, out_pc[0].pos)


# ---------------------------------------------------------------------------
# HeunSampler
# ---------------------------------------------------------------------------


class TestHeunSampler:
    def test_heun_changes_positions(self, diffusion, batch):
        """Heun sampler step should change positions."""
        pos_before = batch.pos.clone()
        sampler = HeunSampler(diffusion.score_model, diffusion.noisers)
        _single_step(diffusion, sampler, batch)
        assert not torch.allclose(batch.pos, pos_before)

    def test_heun_returns_valid_graph(self, diffusion, batch):
        """Heun step should return a batch with valid edges."""
        sampler = HeunSampler(diffusion.score_model, diffusion.noisers)
        out = _single_step(diffusion, sampler, batch)
        assert out.edge_index is not None

    def test_heun_differs_from_em(self, diffusion):
        """Heun sampler should give different positions than EM (different # calls)."""
        torch.manual_seed(3)
        out_em = diffusion.sample(
            1,
            steps=4,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            sampler="em",
        )
        torch.manual_seed(3)
        out_heun = diffusion.sample(
            1,
            steps=4,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            sampler="heun",
        )
        assert not torch.allclose(out_em[0].pos, out_heun[0].pos)


# ---------------------------------------------------------------------------
# ProbabilityFlowODESampler (DDIM)
# ---------------------------------------------------------------------------


class TestProbabilityFlowODESampler:
    def test_ddim_is_deterministic(self, diffusion, batch):
        """DDIM step should be fully deterministic — same output on two calls."""
        diffusion.score_model.sample_mode()
        diffusion.sample_time(batch)
        diffusion.forward_step(batch)
        ts = torch.linspace(1.0, 1e-3, 5, device=batch.pos.device)
        dt = ts[0] - ts[1]
        batch.add_batch_attr("time", ts[0].repeat(batch.x.shape[0], 1), type="node")
        batch.update_graph()

        sampler = ProbabilityFlowODESampler(diffusion.score_model, diffusion.noisers)

        pos_saved = batch.pos.clone()

        out_a = sampler.step(batch, dt, last=False)
        pos_a = out_a.pos.clone()

        batch.pos = pos_saved
        batch.update_graph()

        out_b = sampler.step(batch, dt, last=False)
        pos_b = out_b.pos.clone()

        assert torch.allclose(pos_a, pos_b)

    def test_ddim_differs_from_em(self, diffusion):
        """DDIM (ODE) should produce different results than EM (SDE)."""
        torch.manual_seed(5)
        out_em = diffusion.sample(
            1,
            steps=4,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            sampler="em",
        )
        out_ddim = diffusion.sample(
            1,
            steps=4,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            sampler="ddim",
        )
        assert not torch.allclose(out_em[0].pos, out_ddim[0].pos)

    def test_ddim_changes_positions(self, diffusion, batch):
        """DDIM step should change positions."""
        pos_before = batch.pos.clone()
        sampler = ProbabilityFlowODESampler(diffusion.score_model, diffusion.noisers)
        _single_step(diffusion, sampler, batch)
        assert not torch.allclose(batch.pos, pos_before)


# ---------------------------------------------------------------------------
# HeunODESampler
# ---------------------------------------------------------------------------


class TestHeunODESampler:
    def test_heun_ode_is_deterministic(self, diffusion, batch):
        """HeunODE step should be fully deterministic."""
        diffusion.score_model.sample_mode()
        diffusion.sample_time(batch)
        diffusion.forward_step(batch)
        ts = torch.linspace(1.0, 1e-3, 5, device=batch.pos.device)
        dt = ts[0] - ts[1]
        batch.add_batch_attr("time", ts[0].repeat(batch.x.shape[0], 1), type="node")
        batch.update_graph()

        sampler = HeunODESampler(diffusion.score_model, diffusion.noisers)
        pos_saved = batch.pos.clone()

        out_a = sampler.step(batch, dt, last=False)
        pos_a = out_a.pos.clone()

        batch.pos = pos_saved
        batch.update_graph()

        out_b = sampler.step(batch, dt, last=False)
        pos_b = out_b.pos.clone()

        assert torch.allclose(pos_a, pos_b)

    def test_heun_ode_differs_from_ddim(self, diffusion):
        """HeunODE (2nd-order) should differ from 1st-order DDIM."""
        out_ddim = diffusion.sample(
            1,
            steps=4,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            sampler="ddim",
        )
        out_heun_ode = diffusion.sample(
            1,
            steps=4,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            sampler="heun_ode",
        )
        assert not torch.allclose(out_ddim[0].pos, out_heun_ode[0].pos)

    def test_heun_ode_changes_positions(self, diffusion, batch):
        """HeunODE step should change positions."""
        pos_before = batch.pos.clone()
        sampler = HeunODESampler(diffusion.score_model, diffusion.noisers)
        _single_step(diffusion, sampler, batch)
        assert not torch.allclose(batch.pos, pos_before)


# ---------------------------------------------------------------------------
# Integration: Diffusion.sample() with sampler parameter
# ---------------------------------------------------------------------------


class TestSamplerIntegrationViaDiffusionSample:
    @pytest.mark.parametrize("alias", ["em", "pc", "heun", "ddim", "heun_ode"])
    def test_sample_with_string_alias(self, diffusion, alias):
        """sample() should accept all built-in string aliases."""
        out = diffusion.sample(
            2,
            steps=3,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            sampler=alias,
        )
        assert len(out) == 2
        assert all(isinstance(g, AtomsGraph) for g in out)

    def test_sample_with_sampler_instance(self, diffusion):
        """sample() should accept a pre-built Sampler instance."""
        sampler = HeunSampler(diffusion.score_model, diffusion.noisers)
        out = diffusion.sample(
            1,
            steps=3,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            sampler=sampler,
        )
        assert len(out) == 1
        assert isinstance(out[0], AtomsGraph)

    def test_sample_sampler_none_is_backward_compat(self, diffusion):
        """sampler=None (default) should work as before."""
        out = diffusion.sample(
            2,
            steps=3,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
        )
        assert len(out) == 2

    def test_sample_legacy_corrector_steps_still_works(self, diffusion):
        """corrector_steps>0 without explicit sampler should still change positions."""
        out_no_corr = diffusion.sample(
            1,
            steps=4,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            corrector_steps=0,
        )
        out_with_corr = diffusion.sample(
            1,
            steps=4,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            corrector_steps=2,
            corrector_step_size=1e-3,
        )
        assert not torch.allclose(out_no_corr[0].pos, out_with_corr[0].pos)

    def test_sample_invalid_string_raises_value_error(self, diffusion):
        """An unregistered sampler string should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown sampler"):
            diffusion.sample(
                1,
                steps=3,
                atomic_numbers=[6, 8],
                cell=np.diag([10.0, 10.0, 10.0]),
                property={"property": 1.0},
                sampler="does_not_exist",
            )

    def test_sample_wrong_type_raises_type_error(self, diffusion):
        """Passing an invalid type for sampler should raise TypeError."""
        with pytest.raises(TypeError, match="sampler must be"):
            diffusion.sample(
                1,
                steps=3,
                atomic_numbers=[6, 8],
                cell=np.diag([10.0, 10.0, 10.0]),
                property={"property": 1.0},
                sampler=42,
            )


# ---------------------------------------------------------------------------
# Score call counting
# ---------------------------------------------------------------------------


class TestScoreCallCount:
    """Verify that each sampler makes the expected number of score evaluations."""

    def _count_score_calls(self, diffusion, sampler_name_or_obj, steps=4):
        call_count = [0]
        original_score_fn = diffusion.score_model

        def counting_score_fn(batch):
            call_count[0] += 1
            return original_score_fn(batch)

        if isinstance(sampler_name_or_obj, str):
            from agedi.diffusion.samplers import Sampler as _S
            sampler = _S._registry[sampler_name_or_obj](
                score_fn=counting_score_fn,
                noisers=diffusion.noisers,
            )
        else:
            # Re-wire an existing instance with counting wrapper
            sampler_name_or_obj.score_fn = counting_score_fn
            sampler = sampler_name_or_obj

        diffusion.score_model.sample_mode()

        ts = torch.linspace(1.0, 1e-3, steps, device=diffusion.device)
        dt = ts[0] - ts[1]

        import numpy as _np
        from torch_geometric.data import Batch
        graphs = [diffusion._initialize_graph(6.0, n_atoms=torch.tensor([[3]]),
                                               x=torch.tensor([6, 8, 8]),
                                               cell=torch.tensor(
                                                   _np.diag([10.0, 10.0, 10.0]),
                                                   dtype=torch.float),
                                               property=torch.tensor(1.0))]
        batch = Batch.from_data_list(graphs).to(diffusion.device)
        batch.update_graph()

        for i in range(steps):
            batch.add_batch_attr("time", ts[i].repeat(batch.x.shape[0], 1), type="node")
            last = i == steps - 1
            batch = sampler.step(batch, dt, last)

        return call_count[0]

    def test_em_makes_one_call_per_step(self, diffusion):
        steps = 4
        count = self._count_score_calls(diffusion, "em", steps=steps)
        assert count == steps

    def test_heun_makes_two_calls_per_step(self, diffusion):
        steps = 4
        count = self._count_score_calls(diffusion, "heun", steps=steps)
        assert count == 2 * steps

    def test_heun_ode_makes_two_calls_per_step(self, diffusion):
        steps = 3
        count = self._count_score_calls(diffusion, "heun_ode", steps=steps)
        assert count == 2 * steps

    def test_ddim_makes_one_call_per_step(self, diffusion):
        steps = 4
        count = self._count_score_calls(diffusion, "ddim", steps=steps)
        assert count == steps

    def test_pc_makes_extra_calls_for_correctors(self, diffusion):
        steps = 3
        corrector_steps = 2
        pc = PredictorCorrectorSampler(
            diffusion.score_model,
            diffusion.noisers,
            corrector_steps=corrector_steps,
        )
        count = self._count_score_calls(diffusion, pc, steps=steps)
        # Each outer step: 1 predictor + corrector_steps correctors
        assert count == steps * (1 + corrector_steps)


# ---------------------------------------------------------------------------
# ForcefieldCorrectorSampler (ffpc)
# ---------------------------------------------------------------------------


def _make_mock_regressor(batch_ref):
    """Return a regressor_fn that sets forces_prediction to zeros."""
    def _fn(batch):
        batch["forces_prediction"] = torch.zeros_like(batch.pos)
        return batch
    return _fn


def _make_ffpc_batch(diffusion):
    """Build a minimal single-graph batch suitable for ffpc tests."""
    import numpy as _np
    from torch_geometric.data import Batch as _Batch

    diffusion.score_model.sample_mode()
    graphs = [
        diffusion._initialize_graph(
            6.0,
            n_atoms=torch.tensor([[3]]),
            x=torch.tensor([6, 8, 8]),
            cell=torch.tensor(_np.diag([10.0, 10.0, 10.0]), dtype=torch.float),
            property=torch.tensor(1.0),
        )
    ]
    batch = _Batch.from_data_list(graphs).to(diffusion.device)
    batch.update_graph()
    ts = torch.linspace(1.0, 1e-3, 5, device=diffusion.device)
    dt = ts[0] - ts[1]
    batch.add_batch_attr("time", ts[0].repeat(batch.x.shape[0], 1), type="node")
    return batch, dt


class TestForcefieldCorrectorSampler:
    """Tests for ForcefieldCorrectorSampler (ffpc)."""

    def test_ffpc_step_changes_positions(self, diffusion):
        """A single ffpc step should move atom positions."""
        batch, dt = _make_ffpc_batch(diffusion)
        pos_before = batch.pos.clone()
        sampler = ForcefieldCorrectorSampler(
            diffusion.score_model,
            diffusion.noisers,
            regressor_fn=_make_mock_regressor(batch),
            corrector_steps=1,
        )
        batch = sampler.step(batch, dt, last=False)
        assert not torch.allclose(batch.pos, pos_before)

    def test_ffpc_no_regressor_still_runs(self, diffusion):
        """ffpc without a regressor_fn falls back to EM-only (no corrector)."""
        batch, dt = _make_ffpc_batch(diffusion)
        sampler = ForcefieldCorrectorSampler(
            diffusion.score_model,
            diffusion.noisers,
            regressor_fn=None,
            corrector_steps=1,
        )
        out = sampler.step(batch, dt, last=False)
        assert out.pos.isfinite().all()

    def test_ffpc_pending_frames_empty_when_not_last(self, diffusion):
        """_pending_frames must be empty for non-terminal steps."""
        batch, dt = _make_ffpc_batch(diffusion)
        sampler = ForcefieldCorrectorSampler(
            diffusion.score_model,
            diffusion.noisers,
            regressor_fn=_make_mock_regressor(batch),
            terminal_steps=10,
        )
        sampler.step(batch, dt, last=False)
        assert sampler._pending_frames == []

    def test_ffpc_terminal_overdamped_frame_count(self, diffusion):
        """Terminal overdamped steps: _pending_frames has 1 bridge + N terminal."""
        terminal_steps = 5
        batch, dt = _make_ffpc_batch(diffusion)
        sampler = ForcefieldCorrectorSampler(
            diffusion.score_model,
            diffusion.noisers,
            regressor_fn=_make_mock_regressor(batch),
            corrector_steps=0,
            terminal_steps=terminal_steps,
            terminal_dynamics="overdamped",
        )
        sampler.step(batch, dt, last=True)
        # 1 bridge frame + terminal_steps frames
        assert len(sampler._pending_frames) == 1 + terminal_steps

    def test_ffpc_terminal_langevin_md_frame_count(self, diffusion):
        """Terminal langevin_md steps: _pending_frames has 1 bridge + N terminal."""
        terminal_steps = 4
        batch, dt = _make_ffpc_batch(diffusion)
        sampler = ForcefieldCorrectorSampler(
            diffusion.score_model,
            diffusion.noisers,
            regressor_fn=_make_mock_regressor(batch),
            corrector_steps=0,
            terminal_steps=terminal_steps,
            terminal_dynamics="langevin_md",
            temperature=0.026,
        )
        sampler.step(batch, dt, last=True)
        assert len(sampler._pending_frames) == 1 + terminal_steps

    def test_ffpc_corrector_and_terminal_no_duplicate_bridge(self, diffusion):
        """With corrector_steps>0 and terminal_steps>0, exactly one bridge frame
        appears — not two even though correctors also ran before terminal steps."""
        terminal_steps = 3
        corrector_steps = 2
        batch, dt = _make_ffpc_batch(diffusion)
        sampler = ForcefieldCorrectorSampler(
            diffusion.score_model,
            diffusion.noisers,
            regressor_fn=_make_mock_regressor(batch),
            corrector_steps=corrector_steps,
            terminal_steps=terminal_steps,
            terminal_dynamics="overdamped",
        )
        sampler.step(batch, dt, last=True)
        # 1 bridge + terminal_steps (no extra bridge from corrector path)
        assert len(sampler._pending_frames) == 1 + terminal_steps

    def test_ffpc_save_trajectory_frame_count(self, diffusion):
        """save_trajectory with terminal steps produces N + 1 + T frames."""
        steps = 4
        terminal_steps = 3

        # Attach a mock regressor so terminal steps actually run.
        def _mock_regressor(batch):
            batch["forces_prediction"] = torch.zeros_like(batch.pos)
            return batch

        diffusion.regressor_model = _mock_regressor

        try:
            out = diffusion.sample(
                1,
                steps=steps,
                atomic_numbers=[6, 8, 8],
                cell=np.diag([10.0, 10.0, 10.0]),
                property={"property": 1.0},
                sampler="ffpc",
                sampler_kwargs={
                    "corrector_steps": 0,
                    "terminal_steps": terminal_steps,
                    "terminal_dynamics": "overdamped",
                },
                save_trajectory=True,
            )
        finally:
            diffusion.regressor_model = None

        trajectory = out[0]
        # steps pre-step frames + 1 bridge + terminal_steps
        assert len(trajectory) == steps + 1 + terminal_steps


class TestSamplerKwargsValidation:
    """Validate that sampler_kwargs typos are caught early."""

    def test_unknown_kwarg_raises_value_error(self, diffusion):
        """A typo in sampler_kwargs should raise ValueError naming the bad key."""
        with pytest.raises(ValueError, match="ffpc_terminal_steps"):
            diffusion.sample(
                1,
                steps=3,
                atomic_numbers=[6, 8],
                cell=np.diag([10.0, 10.0, 10.0]),
                property={"property": 1.0},
                sampler="ffpc",
                sampler_kwargs={"ffpc_terminal_steps": 10},
            )

    def test_valid_kwargs_do_not_raise(self, diffusion):
        """Valid sampler_kwargs should be accepted without error."""
        diffusion.sample(
            1,
            steps=3,
            atomic_numbers=[6, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            sampler="ffpc",
            sampler_kwargs={"corrector_steps": 0, "terminal_steps": 0},
        )


class TestSaveCorrectorFrames:
    """Trajectory capture of Langevin corrector sub-steps."""

    def test_correctors_not_captured_by_default(self, diffusion):
        """Without save_corrector_frames, a pc step records no sub-step frames."""
        batch, dt = _make_ffpc_batch(diffusion)
        sampler = PredictorCorrectorSampler(
            diffusion.score_model, diffusion.noisers, corrector_steps=3
        )
        sampler.step(batch, dt, last=False)
        assert sampler._pending_frames == []

    def test_pc_captures_one_frame_per_corrector(self, diffusion):
        """A pc step records the predictor state plus all but the last corrector.

        The last corrector state is the return value, which the outer sampling
        loop records itself — capturing it here would duplicate a frame.
        """
        corrector_steps = 3
        batch, dt = _make_ffpc_batch(diffusion)
        sampler = PredictorCorrectorSampler(
            diffusion.score_model, diffusion.noisers, corrector_steps=corrector_steps
        )
        sampler.save_corrector_frames = True
        sampler.step(batch, dt, last=False)
        # 1 predictor frame + (corrector_steps - 1) corrector frames
        assert len(sampler._pending_frames) == corrector_steps
        assert sampler._pending_includes_final is False

    def test_pc_zero_correctors_captures_nothing(self, diffusion):
        """corrector_steps=0 degenerates to EM, so there is no sub-step to record."""
        batch, dt = _make_ffpc_batch(diffusion)
        sampler = PredictorCorrectorSampler(
            diffusion.score_model, diffusion.noisers, corrector_steps=0
        )
        sampler.save_corrector_frames = True
        sampler.step(batch, dt, last=False)
        assert sampler._pending_frames == []

    def test_ffpc_captures_correctors_and_terminal(self, diffusion):
        """On the last ffpc step, corrector frames precede bridge + terminal frames."""
        corrector_steps = 2
        terminal_steps = 3
        batch, dt = _make_ffpc_batch(diffusion)
        sampler = ForcefieldCorrectorSampler(
            diffusion.score_model,
            diffusion.noisers,
            regressor_fn=_make_mock_regressor(batch),
            corrector_steps=corrector_steps,
            terminal_steps=terminal_steps,
            terminal_dynamics="overdamped",
        )
        sampler.save_corrector_frames = True
        sampler.step(batch, dt, last=True)
        # corrector_steps sub-step frames + 1 bridge + terminal_steps
        assert len(sampler._pending_frames) == corrector_steps + 1 + terminal_steps
        assert sampler._pending_includes_final is True

    def test_trajectory_frame_count_with_correctors(self, diffusion):
        """Every corrector sub-step appears exactly once in the saved trajectory."""
        steps = 3
        corrector_steps = 2

        out = diffusion.sample(
            1,
            steps=steps,
            atomic_numbers=[6, 8, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            corrector_steps=corrector_steps,
            save_trajectory=True,
            save_corrector_frames=True,
        )

        # Per outer step: 1 pre-step frame + corrector_steps sub-step frames,
        # then the final structure.
        assert len(out[0]) == steps * (1 + corrector_steps) + 1

    def test_trajectory_frame_count_without_correctors_unchanged(self, diffusion):
        """Default capture still yields exactly one frame per step plus the final."""
        steps = 3

        out = diffusion.sample(
            1,
            steps=steps,
            atomic_numbers=[6, 8, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            corrector_steps=2,
            save_trajectory=True,
        )

        assert len(out[0]) == steps + 1

    def test_no_duplicate_frames(self, diffusion):
        """Every captured frame is a distinct state, with none recorded twice.

        If capture also recorded the state ``step()`` returns, that state would
        be snapshotted twice — once as the last sub-step frame of step *i*, once
        as the pre-step frame of step *i+1* — leaving a duplicate at every step
        boundary.

        ``eps`` is raised well above its default here on purpose.  At the
        default ``1e-3`` the final ``last=True`` update is smaller than float32
        resolution at these coordinates and can round to no change at all, so
        the last two frames come out bit-identical for reasons unrelated to
        capture.  A larger ``eps`` keeps every step's update resolvable.
        """
        out = diffusion.sample(
            1,
            steps=3,
            eps=0.1,
            atomic_numbers=[6, 8, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            corrector_steps=2,
            save_trajectory=True,
            save_corrector_frames=True,
        )

        trajectory = out[0]
        for i in range(len(trajectory) - 1):
            assert not torch.equal(
                trajectory[i].pos, trajectory[i + 1].pos
            ), f"frames {i} and {i + 1} are identical"

    def test_ffpc_trajectory_frame_count_with_correctors_and_terminal(self, diffusion):
        """ffpc end-to-end: corrector, bridge and terminal frames all appear once."""
        steps = 3
        corrector_steps = 2
        terminal_steps = 3

        def _mock_regressor(batch):
            batch["forces_prediction"] = torch.zeros_like(batch.pos)
            return batch

        diffusion.regressor_model = _mock_regressor
        try:
            out = diffusion.sample(
                1,
                steps=steps,
                atomic_numbers=[6, 8, 8],
                cell=np.diag([10.0, 10.0, 10.0]),
                property={"property": 1.0},
                sampler="ffpc",
                sampler_kwargs={
                    "corrector_steps": corrector_steps,
                    "terminal_steps": terminal_steps,
                    "terminal_dynamics": "overdamped",
                },
                save_trajectory=True,
                save_corrector_frames=True,
            )
        finally:
            diffusion.regressor_model = None

        # steps × (1 pre-step + corrector_steps) + 1 bridge + terminal_steps.
        # The final structure is the last terminal frame, so it is not re-added.
        assert len(out[0]) == steps * (1 + corrector_steps) + 1 + terminal_steps

    def test_flag_ignored_without_save_trajectory(self, diffusion):
        """save_corrector_frames alone must not turn on capture or change output."""
        out = diffusion.sample(
            1,
            steps=3,
            atomic_numbers=[6, 8, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            corrector_steps=2,
            save_corrector_frames=True,
        )
        assert len(out) == 1
        assert out[0].pos.isfinite().all()


class TestMissingRegressorWarning:
    """ffpc must not silently drop its force-field behaviour."""

    def test_warns_when_no_regressor(self, diffusion):
        """Constructing ffpc without a forces model warns about the fallback."""
        with pytest.warns(UserWarning, match="no force-field model available"):
            ForcefieldCorrectorSampler(
                diffusion.score_model, diffusion.noisers, regressor_fn=None
            )

    def test_warning_names_the_skipped_terminal_steps(self, diffusion):
        """The warning states how many terminal steps are being dropped."""
        with pytest.warns(UserWarning, match="all 200 terminal langevin_md steps"):
            ForcefieldCorrectorSampler(
                diffusion.score_model,
                diffusion.noisers,
                regressor_fn=None,
                terminal_steps=200,
                terminal_dynamics="langevin_md",
                temperature=0.5,
            )

    def test_no_warning_when_regressor_present(self, diffusion):
        """A model with a forces head must not trigger the fallback warning."""
        batch, _ = _make_ffpc_batch(diffusion)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ForcefieldCorrectorSampler(
                diffusion.score_model,
                diffusion.noisers,
                regressor_fn=_make_mock_regressor(batch),
                terminal_steps=10,
                temperature=0.5,
            )

    def test_temperature_warning_suppressed_without_regressor(self, diffusion):
        """Only the fallback warning fires; temperature is moot with no terminal phase."""
        with pytest.warns(UserWarning) as record:
            ForcefieldCorrectorSampler(
                diffusion.score_model,
                diffusion.noisers,
                regressor_fn=None,
                terminal_steps=10,
            )
        messages = [str(w.message) for w in record]
        assert len(messages) == 1, messages
        assert "no force-field model available" in messages[0]

    def test_sample_warns_for_model_without_forces_head(self, diffusion):
        """End-to-end: requesting terminal steps on a score-only model warns."""
        assert diffusion.regressor_model is None
        with pytest.warns(UserWarning, match="no force-field model available"):
            diffusion.sample(
                1,
                steps=3,
                atomic_numbers=[6, 8, 8],
                cell=np.diag([10.0, 10.0, 10.0]),
                property={"property": 1.0},
                sampler="ffpc",
                sampler_kwargs={"terminal_steps": 5, "temperature": 0.5},
            )


class TestPostDiffusionRelaxation:
    """Relaxation is driven by max_extra_steps, not by the guidance scale."""

    @staticmethod
    def _unconverged_regressor(batch):
        """Constant forces well above any threshold, so relaxation never converges."""
        batch["forces_prediction"] = torch.ones_like(batch.pos)
        return batch

    def _sample(self, diffusion, guidance, max_extra_steps, steps=3, **kw):
        from agedi.diffusion import ForcefieldGuidanceConfig

        diffusion.regressor_model = self._unconverged_regressor
        try:
            return diffusion.sample(
                1,
                steps=steps,
                atomic_numbers=[6, 8, 8],
                cell=np.diag([10.0, 10.0, 10.0]),
                property={"property": 1.0},
                save_trajectory=True,
                ff_guidance=ForcefieldGuidanceConfig(
                    guidance=guidance,
                    force_threshold=0.05,
                    max_extra_steps=max_extra_steps,
                ),
                **kw,
            )[0]
        finally:
            diffusion.regressor_model = None

    def test_relaxation_runs_without_guidance(self, diffusion):
        """max_extra_steps alone triggers relaxation; it used to be a silent no-op."""
        steps, extra = 3, 5
        traj = self._sample(diffusion, guidance=0.0, max_extra_steps=extra)
        # steps pre-step frames + extra relaxation frames; the last relaxation
        # frame is the final structure, so nothing is appended after it.
        assert len(traj) == steps + extra

    def test_no_relaxation_when_max_extra_steps_zero(self, diffusion):
        """Without relaxation steps the trajectory is just the diffusion path."""
        steps = 3
        traj = self._sample(diffusion, guidance=0.0, max_extra_steps=0)
        assert len(traj) == steps + 1

    def test_guidance_still_triggers_relaxation(self, diffusion):
        """The original guidance-driven path is unchanged."""
        steps, extra = 3, 5
        traj = self._sample(diffusion, guidance=1.0, max_extra_steps=extra)
        assert len(traj) == steps + extra

    def test_relaxation_without_guidance_gets_persistent_step_sizer(self, diffusion):
        """A persistent L-BFGS sizer is allocated so curvature history accumulates.

        Without it, post_diffusion_relaxation_step builds a throwaway sizer on
        every call and the relaxation degenerates to fixed-size steps.
        """
        self._sample(diffusion, guidance=0.0, max_extra_steps=5)
        assert diffusion.lbfgs_step_sizer is not None

    def test_relaxation_runs_with_ffpc(self, diffusion):
        """Relaxation composes with a sampler that has its own force-field use."""
        steps, extra = 3, 5
        traj = self._sample(
            diffusion,
            guidance=0.0,
            max_extra_steps=extra,
            steps=steps,
            sampler="ffpc",
            sampler_kwargs={
                "corrector_steps": 1,
                "terminal_steps": 0,
                "temperature": 0.5,
            },
        )
        assert len(traj) == steps + extra

    def test_relaxation_skipped_when_already_converged(self, diffusion):
        """Structures below force_threshold need no relaxation steps."""
        from agedi.diffusion import ForcefieldGuidanceConfig

        def _converged(batch):
            batch["forces_prediction"] = torch.zeros_like(batch.pos)
            return batch

        steps = 3
        diffusion.regressor_model = _converged
        try:
            traj = diffusion.sample(
                1,
                steps=steps,
                atomic_numbers=[6, 8, 8],
                cell=np.diag([10.0, 10.0, 10.0]),
                property={"property": 1.0},
                save_trajectory=True,
                ff_guidance=ForcefieldGuidanceConfig(
                    guidance=0.0, force_threshold=0.05, max_extra_steps=5
                ),
            )[0]
        finally:
            diffusion.regressor_model = None

        assert len(traj) == steps + 1

    def test_no_relaxation_without_regressor(self, diffusion):
        """max_extra_steps cannot relax a model that has no forces head."""
        from agedi.diffusion import ForcefieldGuidanceConfig

        assert diffusion.regressor_model is None
        steps = 3
        traj = diffusion.sample(
            1,
            steps=steps,
            atomic_numbers=[6, 8, 8],
            cell=np.diag([10.0, 10.0, 10.0]),
            property={"property": 1.0},
            save_trajectory=True,
            ff_guidance=ForcefieldGuidanceConfig(
                guidance=0.0, force_threshold=0.05, max_extra_steps=5
            ),
        )[0]
        assert len(traj) == steps + 1


class TestTerminalConfinement:
    """ffpc terminal dynamics must respect the z-confinement slab."""

    CONF = (2.0, 8.0)

    @staticmethod
    def _push_up(batch):
        """Strong constant +z force, enough to drive atoms through the wall."""
        f = torch.zeros_like(batch.pos)
        f[:, 2] = 50.0
        batch["forces_prediction"] = f
        return batch

    def _sample(self, diffusion, **sampler_kwargs):
        diffusion.regressor_model = self._push_up
        try:
            out = diffusion.sample(
                2,
                steps=4,
                atomic_numbers=[6, 8, 8],
                cell=np.diag([10.0, 10.0, 10.0]),
                property={"property": 1.0},
                confinement=self.CONF,
                sampler="ffpc",
                sampler_kwargs=sampler_kwargs,
                save_trajectory=True,
            )
        finally:
            diffusion.regressor_model = None
        return out

    def _assert_confined(self, trajectories):
        """Assert every frame stays in the slab, except the initial prior draw.

        Frame 0 is the raw sample from the positions prior.  Confinement is
        imposed from the first denoise step onward, so an unconfined prior
        (as in this fixture) can start outside the slab.
        """
        lo, hi = self.CONF
        for traj in trajectories:
            for i, frame in enumerate(traj[1:], start=1):
                z = frame.pos[:, 2]
                assert z.min() >= lo - 1e-4, f"frame {i}: z={z.min():.4f} below {lo}"
                assert z.max() <= hi + 1e-4, f"frame {i}: z={z.max():.4f} above {hi}"

    def test_terminal_overdamped_respects_confinement(self, diffusion):
        """Overdamped terminal steps must not push atoms out of the slab."""
        self._assert_confined(
            self._sample(
                diffusion,
                corrector_steps=1,
                terminal_steps=30,
                terminal_dynamics="overdamped",
                terminal_step_size=0.05,
                temperature=0.1,
            )
        )

    def test_terminal_langevin_md_respects_confinement(self, diffusion):
        """BAOAB terminal steps must not push atoms out of the slab."""
        self._assert_confined(
            self._sample(
                diffusion,
                corrector_steps=1,
                terminal_steps=30,
                terminal_dynamics="langevin_md",
                terminal_step_size=1.0,
                temperature=0.1,
            )
        )

    def test_unconfined_sampling_is_unaffected(self, diffusion):
        """Without confinement the terminal phase is free to move atoms anywhere."""
        diffusion.regressor_model = self._push_up
        try:
            out = diffusion.sample(
                1,
                steps=4,
                atomic_numbers=[6, 8, 8],
                cell=np.diag([10.0, 10.0, 10.0]),
                property={"property": 1.0},
                sampler="ffpc",
                sampler_kwargs={
                    "corrector_steps": 0,
                    "terminal_steps": 30,
                    "terminal_dynamics": "langevin_md",
                    "terminal_step_size": 1.0,
                    "temperature": 0.1,
                },
            )
        finally:
            diffusion.regressor_model = None
        assert out[0].pos.isfinite().all()

    def test_velocity_is_reflected_at_the_wall(self, diffusion):
        """An atom driven into a wall must bounce, not stick to it.

        A bare clamp leaves the velocity pointing into the wall, so the atom
        stays pinned at the boundary for the rest of the run.
        """
        trajectories = self._sample(
            diffusion,
            corrector_steps=0,
            terminal_steps=40,
            terminal_dynamics="langevin_md",
            terminal_step_size=1.0,
            temperature=0.1,
        )

        # Terminal frames are the tail of the trajectory.
        z_max_per_frame = [frame.pos[:, 2].max().item() for frame in trajectories[0]]
        terminal = z_max_per_frame[-40:]

        # With reflection the topmost atom leaves the wall again; pinned-at-the
        # -ceiling would make every terminal frame sit exactly at z_max.
        at_wall = sum(1 for z in terminal if abs(z - self.CONF[1]) < 1e-4)
        assert at_wall < len(terminal), "every terminal frame is pinned at the wall"


# ---------------------------------------------------------------------------
# Masked-atom time consistency (regression test)
# ---------------------------------------------------------------------------


def _make_time_recording_score_fn(record):
    """Return a score_fn stub that logs every ``batch.time`` it is called with."""

    def score_fn(batch):
        record.append(batch.time.clone())
        batch["pos_score"] = torch.zeros_like(batch.pos)
        return batch

    return score_fn


def _make_zero_regressor():
    def regressor_fn(batch):
        batch["forces_prediction"] = torch.zeros_like(batch.pos)
        return batch

    return regressor_fn


def _prep_masked_batch(batch, t_val, dtype=torch.float):
    """Set a uniform starting time and build the graph.

    Mirrors ``Diffusion._sample_batch``'s own main loop, which writes time via
    ``add_batch_attr`` directly rather than the ``.time`` setter, so every atom
    (masked or not) starts at the same global time.
    """
    t = torch.tensor(t_val, dtype=dtype)
    batch.add_batch_attr("time", t.repeat(batch.x.shape[0], 1), type="node")
    batch.update_graph()
    return batch


@pytest.mark.parametrize("batch", ["surface"], indirect=True)
class TestMaskedAtomTimeConsistency:
    """A masked atom must see the same intermediate diffusion time as every
    unmasked atom at every score call within a single sampler step.

    Regression test for a bug where samplers advanced ``batch.time`` mid-step
    via the ``.time`` property setter, which zeroes masked atoms' time instead
    of giving them the real global time (unlike the main sampling loop, which
    writes time directly via ``add_batch_attr``).
    """

    dt = torch.tensor(0.1)
    t_val = 0.8

    def _assert_uniform_and_correct(self, times, expected_values):
        assert len(times) == len(expected_values)
        for t, expected in zip(times, expected_values):
            assert t.min() == t.max(), (
                f"batch.time is not uniform across atoms: {t.unique()}"
            )
            assert torch.allclose(t, torch.full_like(t, expected)), (
                f"expected time {expected}, got {t.unique()}"
            )

    def test_pc_corrector_time_is_uniform(self, batch):
        batch = _prep_masked_batch(batch, self.t_val)
        record = []
        sampler = PredictorCorrectorSampler(
            _make_time_recording_score_fn(record),
            [CellPositions()],
            corrector_steps=1,
        )
        sampler.step(batch, self.dt, last=False)
        self._assert_uniform_and_correct(
            record, [self.t_val, self.t_val - self.dt.item()]
        )

    def test_heun_second_score_call_time_is_uniform(self, batch):
        batch = _prep_masked_batch(batch, self.t_val)
        record = []
        sampler = HeunSampler(
            _make_time_recording_score_fn(record),
            [CellPositions()],
        )
        sampler.step(batch, self.dt, last=False)
        self._assert_uniform_and_correct(
            record, [self.t_val, self.t_val - self.dt.item()]
        )
        # After the step, batch.time must be restored to t_current, not 0.
        assert batch.time.min() == batch.time.max() == self.t_val

    def test_ffpc_corrector_time_is_uniform(self, batch):
        batch = _prep_masked_batch(batch, self.t_val)
        record = []
        sampler = ForcefieldCorrectorSampler(
            _make_time_recording_score_fn(record),
            [CellPositions()],
            regressor_fn=_make_zero_regressor(),
            corrector_steps=1,
            temperature=0.5,
        )
        sampler.step(batch, self.dt, last=False)
        self._assert_uniform_and_correct(
            record, [self.t_val, self.t_val - self.dt.item()]
        )

    def test_heun_ode_second_score_call_time_is_uniform(self, batch):
        batch = _prep_masked_batch(batch, self.t_val)
        record = []
        sampler = HeunODESampler(
            _make_time_recording_score_fn(record),
            [CellPositions()],
        )
        sampler.step(batch, self.dt, last=False)
        self._assert_uniform_and_correct(
            record, [self.t_val, self.t_val - self.dt.item()]
        )
        # After the step, batch.time must be restored to t_current, not 0.
        assert batch.time.min() == batch.time.max() == self.t_val
