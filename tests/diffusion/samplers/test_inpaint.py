"""Tests for InpaintingSampler and the underlying noiser forward-marginal hooks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agedi.diffusion.samplers import (
    EulerMaruyamaSampler,
    HeunODESampler,
    HeunSampler,
    InpaintingSampler,
    PredictorCorrectorSampler,
    ProbabilityFlowODESampler,
)


def _attach_inpaint_state(diffusion, batch, frac_selected=0.5, seed=0):
    """Attach inpaint_mask / {key}0 reference tensors to a fixture batch.

    Mirrors what Diffusion._initialize_inpaint_graph does per-graph, but
    operates directly on an already-built batch for sampler-level testing.
    """
    torch.manual_seed(seed)
    n_atoms = batch.pos.shape[0]
    inpaint_mask = torch.rand(n_atoms) < frac_selected
    # Never select an atom that is already hard-frozen (mask=True in the
    # 'surface' fixture) -- same invariant the API layer enforces.
    inpaint_mask = inpaint_mask & ~batch.mask
    if not inpaint_mask.any():
        inpaint_mask[0] = True
    batch.inpaint_mask = inpaint_mask

    for noiser in diffusion.noisers:
        key = noiser.key
        batch.add_batch_attr(key + "0", batch[key].clone(), type="node")

    return batch


def _make_inpainting_sampler(diffusion, base_cls=EulerMaruyamaSampler, **kw):
    base = base_cls(diffusion.score_model, diffusion.noisers, **kw) if kw else base_cls(
        diffusion.score_model, diffusion.noisers
    )
    return InpaintingSampler(base, diffusion.noisers)


def _run_reverse_loop(diffusion, sampler, batch, steps=5, eps=1e-2, t_start=1.0):
    diffusion.score_model.sample_mode()
    ts = torch.linspace(t_start, eps, steps, device=batch.pos.device)
    dt = ts[0] - ts[1]
    for i in range(steps):
        batch.add_batch_attr("time", ts[i].repeat(batch.x.shape[0], 1), type="node")
        last = i == steps - 1
        batch = sampler.step(batch, dt, last)
    return batch


class TestInpaintingSamplerBasics:
    def test_known_atoms_reconstructed_exactly(self, diffusion, batch):
        """After a full reverse trajectory, known (non-selected) atoms must
        match their pos0 reference to within float tolerance."""
        pos0_ref = batch.pos.clone()
        batch = _attach_inpaint_state(diffusion, batch, frac_selected=0.4)
        known = ~batch.inpaint_mask

        sampler = _make_inpainting_sampler(diffusion)
        out = _run_reverse_loop(diffusion, sampler, batch, steps=5, eps=1e-2)

        assert torch.allclose(out.pos[known], pos0_ref[known], atol=1e-3)

    def test_selected_atoms_change(self, diffusion, batch):
        """Selected atoms should differ from their reference after a full run."""
        pos0_ref = batch.pos.clone()
        batch = _attach_inpaint_state(diffusion, batch, frac_selected=0.4, seed=1)
        selected = batch.inpaint_mask
        assert selected.any(), "fixture must select at least one atom"

        sampler = _make_inpainting_sampler(diffusion)
        out = _run_reverse_loop(diffusion, sampler, batch, steps=5, eps=1e-2)

        # At least one selected atom should have moved measurably from where
        # it started -- t_start=1.0 fully re-noises the selection.
        disp = (out.pos[selected] - pos0_ref[selected]).norm(dim=-1)
        assert disp.max().item() > 1e-3

    def test_frozen_atoms_bit_exact_through_trajectory(self, diffusion, batch):
        """mask=True (hard-frozen) atoms must never move, even mid-trajectory."""
        if not batch.mask.any():
            pytest.skip("fixture has no hard-frozen atoms")
        frozen_ref = batch.pos[batch.mask].clone()
        batch = _attach_inpaint_state(diffusion, batch, frac_selected=0.4, seed=2)

        sampler = _make_inpainting_sampler(diffusion)
        out = _run_reverse_loop(diffusion, sampler, batch, steps=4, eps=1e-2)

        assert torch.allclose(out.pos[out.mask], frozen_ref, atol=1e-7)

    def test_returns_finite_positions(self, diffusion, batch):
        batch = _attach_inpaint_state(diffusion, batch, frac_selected=0.4, seed=3)
        sampler = _make_inpainting_sampler(diffusion)
        out = _run_reverse_loop(diffusion, sampler, batch, steps=4, eps=1e-2)
        assert out.pos.isfinite().all()

    def test_t_start_less_than_one_reconstructs_known_atoms(self, diffusion, batch):
        pos0_ref = batch.pos.clone()
        batch = _attach_inpaint_state(diffusion, batch, frac_selected=0.4, seed=4)
        known = ~batch.inpaint_mask

        sampler = _make_inpainting_sampler(diffusion)
        out = _run_reverse_loop(diffusion, sampler, batch, steps=4, eps=1e-2, t_start=0.3)

        assert torch.allclose(out.pos[known], pos0_ref[known], atol=1e-3)


class TestInpaintingSamplerComposesWithBaseSamplers:
    @pytest.mark.parametrize(
        "base_cls",
        [EulerMaruyamaSampler, PredictorCorrectorSampler, HeunSampler,
         ProbabilityFlowODESampler, HeunODESampler],
    )
    def test_runs_with_each_base_sampler(self, diffusion, batch, base_cls):
        pos0_ref = batch.pos.clone()
        batch = _attach_inpaint_state(diffusion, batch, frac_selected=0.4, seed=5)
        known = ~batch.inpaint_mask

        sampler = _make_inpainting_sampler(diffusion, base_cls=base_cls)
        out = _run_reverse_loop(diffusion, sampler, batch, steps=3, eps=1e-2)

        assert out.pos.isfinite().all()
        assert torch.allclose(out.pos[known], pos0_ref[known], atol=1e-3)


class TestInpaintingSamplerResampling:
    def test_n_resample_runs_and_reconstructs(self, diffusion, batch):
        pos0_ref = batch.pos.clone()
        batch = _attach_inpaint_state(diffusion, batch, frac_selected=0.4, seed=6)
        known = ~batch.inpaint_mask

        base = EulerMaruyamaSampler(diffusion.score_model, diffusion.noisers)
        sampler = InpaintingSampler(base, diffusion.noisers, n_resample=2, jump_length=2)
        out = _run_reverse_loop(diffusion, sampler, batch, steps=3, eps=1e-2)

        assert out.pos.isfinite().all()
        assert torch.allclose(out.pos[known], pos0_ref[known], atol=1e-3)

    def test_n_resample_calls_score_more_than_once_per_step(self, diffusion, batch):
        """n_resample=2, jump_length=2 should call the score model 4x per
        outer step (2 resample passes x 2 micro-steps), vs 1x for n_resample=1."""
        batch = _attach_inpaint_state(diffusion, batch, frac_selected=0.4, seed=7)
        diffusion.score_model.sample_mode()

        calls = {"n": 0}
        real_score_fn = diffusion.score_model

        def counting_score_fn(b):
            calls["n"] += 1
            return real_score_fn(b)

        base = EulerMaruyamaSampler(counting_score_fn, diffusion.noisers)
        sampler = InpaintingSampler(base, diffusion.noisers, n_resample=2, jump_length=2)

        ts = torch.linspace(1.0, 1e-2, 3, device=batch.pos.device)
        dt = ts[0] - ts[1]
        batch.add_batch_attr("time", ts[0].repeat(batch.x.shape[0], 1), type="node")
        sampler.step(batch, dt, last=False)

        assert calls["n"] == 4

    def test_invalid_n_resample_raises(self, diffusion):
        base = EulerMaruyamaSampler(diffusion.score_model, diffusion.noisers)
        with pytest.raises(ValueError):
            InpaintingSampler(base, diffusion.noisers, n_resample=0)

    def test_invalid_jump_length_raises(self, diffusion):
        base = EulerMaruyamaSampler(diffusion.score_model, diffusion.noisers)
        with pytest.raises(ValueError):
            InpaintingSampler(base, diffusion.noisers, jump_length=0)


def _make_sparse_batch(diffusion):
    """A minimal, sparse single-graph batch whose edge count is unlikely to
    change across a few reverse steps -- mirrors test_samplers.py's
    _make_ffpc_batch. Deliberately independent of the shared 'batch' fixture:
    to_data_list() on a *dense* multi-graph periodic batch after
    update_graph() has rebuilt edge_index to a different size hits a
    pre-existing bug in Batch._slice_dict bookkeeping (stale absolute
    offsets, not recomputed from the new tensor shape), unrelated to
    inpainting -- see the task filed for it. A sparse 3-atom system keeps its
    neighbor count constant in practice, sidestepping that bug entirely.
    """
    from torch_geometric.data import Batch as _Batch

    diffusion.score_model.sample_mode()
    graphs = [
        diffusion._initialize_graph(
            6.0,
            n_atoms=torch.tensor([[3]]),
            x=torch.tensor([6, 8, 8]),
            cell=torch.tensor(np.diag([10.0, 10.0, 10.0]), dtype=torch.float),
            property=torch.rand(1),
        )
    ]
    return _Batch.from_data_list(graphs).to(diffusion.device)


class TestInpaintingSamplerTrajectory:
    def test_save_corrector_frames_forwarded_to_base(self, diffusion):
        """save_corrector_frames set on the wrapper should propagate to the
        base sampler and its captured frames should surface on the wrapper.
        """
        single = _make_sparse_batch(diffusion)
        single = _attach_inpaint_state(diffusion, single, frac_selected=0.4, seed=8)
        diffusion.score_model.sample_mode()

        base = PredictorCorrectorSampler(diffusion.score_model, diffusion.noisers, corrector_steps=2)
        sampler = InpaintingSampler(base, diffusion.noisers)
        sampler.save_corrector_frames = True
        assert base.save_corrector_frames is True

        ts = torch.linspace(1.0, 1e-2, 3, device=single.pos.device)
        dt = ts[0] - ts[1]
        single.add_batch_attr("time", ts[0].repeat(single.x.shape[0], 1), type="node")
        sampler.step(single, dt, last=False)

        # PC captures one frame after the predictor plus corrector_steps-1
        # sub-steps; those should have been drained into the wrapper.
        assert len(sampler._pending_frames) > 0
