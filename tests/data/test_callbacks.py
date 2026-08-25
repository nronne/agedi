import types

import pytest
import torch
from torch import nn

from agedi.data.callbacks import EMACallback, TrainingPhase


def test_prepare_epoch_increments_counter_without_phase_change():
    callback = TrainingPhase(n_phases=2, epochs_per_phase=[2, 2])
    trainer = types.SimpleNamespace(
        current_epoch=0,
        datamodule=types.SimpleNamespace(set_phase=lambda phase: None),
    )

    callback._prepare_epoch(trainer, model=None)

    assert callback.current_phase == 0
    assert callback.epoch_counter == 1


def test_prepare_epoch_advances_phase_and_resets_counter():
    called = []
    callback = TrainingPhase(n_phases=3, epochs_per_phase=[1, 1, 1])
    callback.epoch_counter = 1
    trainer = types.SimpleNamespace(
        current_epoch=1,
        datamodule=types.SimpleNamespace(set_phase=lambda phase: called.append(phase)),
    )

    callback._prepare_epoch(trainer, model=None)

    assert callback.current_phase == 1
    assert callback.epoch_counter == 0
    assert called == [1]


def test_prepare_epoch_noop_in_last_phase():
    called = []
    callback = TrainingPhase(n_phases=2, epochs_per_phase=[1, 1])
    callback.current_phase = 1
    callback.epoch_counter = 10
    trainer = types.SimpleNamespace(
        current_epoch=10,
        datamodule=types.SimpleNamespace(set_phase=lambda phase: called.append(phase)),
    )

    callback._prepare_epoch(trainer, model=None)

    assert callback.current_phase == 1
    assert callback.epoch_counter == 10
    assert called == []


def test_on_validation_end_delegates_to_prepare_epoch():
    callback = TrainingPhase(n_phases=2, epochs_per_phase=[1, 1])
    trainer = types.SimpleNamespace(
        current_epoch=0,
        datamodule=types.SimpleNamespace(set_phase=lambda phase: None),
    )

    callback.on_validation_end(trainer, model=None)

    assert callback.epoch_counter == 1



# ---------------------------------------------------------------------------
# EMACallback
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """A minimal stand-in LightningModule: just needs named_parameters()."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=True)

    def named_parameters(self, *args, **kwargs):
        return nn.Module.named_parameters(self, *args, **kwargs)


def _set_all_params(model, value: float):
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(value)


def test_rejects_decay_outside_open_interval():
    with pytest.raises(ValueError, match="decay"):
        EMACallback(decay=0.0)
    with pytest.raises(ValueError, match="decay"):
        EMACallback(decay=1.0)


def test_on_train_start_seeds_the_shadow_from_live_weights():
    model = _TinyModel()
    _set_all_params(model, 3.0)
    cb = EMACallback(decay=0.9)

    cb.on_train_start(trainer=None, pl_module=model)

    for name, p in model.named_parameters():
        assert torch.allclose(cb._shadow[name], p)


def test_on_train_start_does_not_reseed_on_a_later_call():
    """The shadow must persist across repeated fit() calls on one Trainer,
    which is how GO-Diff reuses a single Trainer across its outer iterations."""
    model = _TinyModel()
    _set_all_params(model, 1.0)
    cb = EMACallback(decay=0.9)
    cb.on_train_start(trainer=None, pl_module=model)
    seeded_shadow = {k: v.clone() for k, v in cb._shadow.items()}

    # Weights move (simulating a completed training stage) before fit() is
    # called again on the same Trainer/callback.
    _set_all_params(model, 5.0)
    cb.on_train_start(trainer=None, pl_module=model)

    for name in seeded_shadow:
        assert torch.allclose(cb._shadow[name], seeded_shadow[name])


def test_batch_end_moves_shadow_toward_live_weights_by_one_minus_decay():
    model = _TinyModel()
    _set_all_params(model, 0.0)
    cb = EMACallback(decay=0.9)
    cb.on_train_start(trainer=None, pl_module=model)  # shadow = 0

    _set_all_params(model, 10.0)
    cb.on_train_batch_end(
        trainer=None, pl_module=model, outputs=None, batch=None, batch_idx=0
    )

    for _, p in model.named_parameters():
        pass  # just confirm no crash; check exact value below
    for name, shadow in cb._shadow.items():
        # shadow = 0.9*0 + 0.1*10 = 1.0
        assert torch.allclose(shadow, torch.full_like(shadow, 1.0))


def test_shadow_converges_toward_a_held_steady_weight():
    model = _TinyModel()
    _set_all_params(model, 0.0)
    cb = EMACallback(decay=0.5)
    cb.on_train_start(trainer=None, pl_module=model)

    _set_all_params(model, 8.0)
    for _ in range(20):
        cb.on_train_batch_end(
            trainer=None, pl_module=model, outputs=None, batch=None, batch_idx=0
        )

    for name, shadow in cb._shadow.items():
        assert torch.allclose(shadow, torch.full_like(shadow, 8.0), atol=1e-3)


def test_on_train_end_copies_shadow_into_the_live_model():
    model = _TinyModel()
    _set_all_params(model, 0.0)
    cb = EMACallback(decay=0.5)
    cb.on_train_start(trainer=None, pl_module=model)

    _set_all_params(model, 8.0)
    for _ in range(5):
        cb.on_train_batch_end(
            trainer=None, pl_module=model, outputs=None, batch=None, batch_idx=0
        )
    # Live weights are still the raw (un-averaged) 8.0 at this point.
    for p in model.parameters():
        assert torch.allclose(p, torch.full_like(p, 8.0))

    cb.on_train_end(trainer=None, pl_module=model)

    # After on_train_end, the live weights equal the (smoothed) shadow,
    # which has not fully caught up to 8.0 yet.
    for name, p in model.named_parameters():
        assert torch.allclose(p, cb._shadow[name])
        assert not torch.allclose(p, torch.full_like(p, 8.0))


def test_new_parameter_appearing_mid_training_is_picked_up_gracefully():
    """A parameter absent when the shadow was seeded (e.g. a lazily-built
    head) should not crash on_train_batch_end; it starts being tracked from
    whatever the live value is when first seen."""
    model = _TinyModel()
    cb = EMACallback(decay=0.9)
    cb.on_train_start(trainer=None, pl_module=model)

    # Simulate a newly-appeared parameter.
    model.extra = nn.Parameter(torch.tensor([2.0]))
    cb.on_train_batch_end(
        trainer=None, pl_module=model, outputs=None, batch=None, batch_idx=0
    )

    assert cb._shadow["extra"].item() == pytest.approx(2.0)


def test_frozen_parameters_are_not_tracked():
    model = _TinyModel()
    model.linear.weight.requires_grad_(False)
    cb = EMACallback(decay=0.9)
    cb.on_train_start(trainer=None, pl_module=model)

    assert not any(name.startswith("linear.weight") for name in cb._shadow)
    assert any(name.startswith("linear.bias") for name in cb._shadow)


def test_state_dict_round_trip():
    model = _TinyModel()
    _set_all_params(model, 4.0)
    cb = EMACallback(decay=0.7)
    cb.on_train_start(trainer=None, pl_module=model)

    state = cb.state_dict()

    restored = EMACallback(decay=0.1)
    restored.load_state_dict(state)

    assert restored.decay == 0.7
    for name, tensor in cb._shadow.items():
        assert torch.allclose(restored._shadow[name], tensor)
