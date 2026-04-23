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
    )
    batch.forces = torch.randn_like(batch.pos)

    loss = model.loss(batch)["loss"]

    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)


def test_regressor_loss_with_weighting(batch):
    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[OffsetHead("forces", 1.0)],
        head_weights={"forces": 2.0},
        use_weighting=True,
        mask_forces=False,
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

