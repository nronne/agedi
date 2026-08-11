import torch
import pytest

from agedi.data import Representation
from agedi.models.head import Head
from agedi.models.regressor import RegressorModel
from agedi.models.translator import Translator


class DummyTranslator(Translator):
    def _translate(self, batch):
        return {"batch": batch, "representation": batch.representation}

    def _get_representation(self, batch, out):
        return out

    def _translate_representation(self, rep, translated_batch):
        translated_batch["representation"] = rep
        return translated_batch


class DummyRepresentation(torch.nn.Module):
    def forward(self, translated_batch):
        n = translated_batch["batch"].num_nodes
        return Representation(
            scalar=torch.ones((n, 2, 1)),
            vector=torch.ones((n, 2, 3)),
        )


class OffsetHead(Head):
    def __init__(self, key, offset):
        super().__init__()
        self._key = key
        self.offset = offset

    def _score(self, translated_batch):
        return translated_batch["batch"][self.key] + self.offset


def test_regressor_init_rejects_unknown_head_key():
    with pytest.raises(ValueError):
        RegressorModel(
            translator=DummyTranslator(),
            representation=DummyRepresentation(),
            heads=[OffsetHead("unknown", 1.0)],
        )


def test_regressor_forward_adds_prediction(batch):
    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[OffsetHead("forces", 1.0)],
    )
    batch.forces = torch.randn_like(batch.pos)

    out = model.forward(batch)

    assert "forces_prediction" in out.keys()
    assert out.forces_prediction.shape == batch.forces.shape


def test_regressor_loss_without_weighting(batch):
    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[OffsetHead("forces", 1.0)],
        mask_forces=False,
        force_loss="mse",
    )
    batch.forces = torch.randn_like(batch.pos)

    loss = model.loss(batch)["loss"]

    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)


def test_regressor_force_loss_defaults_to_huber(batch):
    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[OffsetHead("forces", 1.0)],
        mask_forces=False,
    )
    batch.forces = torch.randn_like(batch.pos)

    loss = model.loss(batch)["loss"]

    # Constant error of 1.0 is far above the default delta of 0.01, so the
    # Huber loss is in its linear regime: delta * (|e| - delta / 2).
    expected = 0.01 * (1.0 - 0.005)
    assert model.force_loss == "huber"
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-7)


def test_regressor_huber_delta_is_configurable(batch):
    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[OffsetHead("forces", 1.0)],
        mask_forces=False,
        huber_delta=10.0,
    )
    batch.forces = torch.randn_like(batch.pos)

    loss = model.loss(batch)["loss"]

    # Error of 1.0 is below delta, so the loss is quadratic: 0.5 * e^2.
    assert torch.isclose(loss, torch.tensor(0.5), atol=1e-6)


def test_regressor_mae_force_loss(batch):
    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[OffsetHead("forces", 1.0)],
        mask_forces=False,
        force_loss="mae",
    )
    batch.forces = torch.randn_like(batch.pos)

    loss = model.loss(batch)["loss"]

    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [{"force_loss": "l2"}, {"energy_loss": "hinge"}, {"huber_delta": 0.0}],
)
def test_regressor_rejects_invalid_loss_config(kwargs):
    with pytest.raises(ValueError):
        RegressorModel(
            translator=DummyTranslator(),
            representation=DummyRepresentation(),
            heads=[OffsetHead("forces", 1.0)],
            **kwargs,
        )


def test_regressor_get_config_round_trip():
    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[OffsetHead("forces", 1.0)],
        force_loss="mse",
        huber_delta=0.5,
        head_weights={"forces": 3.0},
    )

    config = model.get_config()
    rebuilt = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[OffsetHead("forces", 1.0)],
        **config,
    )

    assert config["force_loss"] == "mse"
    assert config["huber_delta"] == 0.5
    assert rebuilt.get_config() == config


def test_regressor_loss_with_weighting(batch):
    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[OffsetHead("forces", 1.0)],
        head_weights={"forces": 2.0},
        use_weighting=True,
        mask_forces=False,
        force_loss="mse",
    )
    batch.forces = torch.randn_like(batch.pos)
    batch.weight = torch.arange(1, batch.num_graphs + 1, dtype=torch.float)
    weights = batch.weight[batch.batch]

    loss = model.loss(batch)["loss"]

    expected = 2.0 * weights.mean()
    assert torch.isclose(loss, expected, atol=1e-6)

def test_regressor_mask_forces(batch):
        model = RegressorModel(
                translator=DummyTranslator(),
                representation=DummyRepresentation(),
                heads=[OffsetHead("forces", 1.0)],
                mask_forces=True,
        )
        batch.forces = torch.randn_like(batch.pos)
        
        out = model.forward(batch)
        
        if hasattr(batch, 'mask'):
            assert torch.all(out.forces_prediction[batch.positions_mask] == 0.0)

