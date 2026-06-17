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


def test_regressor_loss_energy_weighting(batch):
    """Energy weighting must use per-graph weights [B], not per-atom [N]."""
    from agedi.data import AtomsGraph
    from torch_geometric.data import Batch as PyGBatch

    # Build a minimal two-graph batch with known energies.
    graphs = []
    for e in [2.0, 4.0]:
        g = AtomsGraph.from_atoms(__import__("ase").build.molecule("H2O"))
        g.energy = torch.tensor(e)
        graphs.append(g)
    b = PyGBatch.from_data_list(graphs)
    b.n_atoms = torch.tensor([[3], [3]])

    class EnergyOffsetHead(Head):
        def __init__(self, offset):
            super().__init__()
            self._key = "energy"
            self.offset = offset

        def _score(self, translated_batch):
            n = translated_batch["batch"].num_graphs
            return torch.zeros(n) + self.offset

    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[EnergyOffsetHead(1.0)],
        use_weighting=True,
        mask_forces=False,
    )
    # Weights [w0, w1] = [1, 2]
    b.weight = torch.tensor([1.0, 2.0])

    loss = model.loss(b)["loss"]

    # f = energy / n_atoms = [2/3, 4/3]; f_pred = [1/3, 1/3]
    # per-graph MSE: [(2/3-1/3)^2, (4/3-1/3)^2] = [1/9, 1]
    # weighted mean: ([1/9]*1 + [1]*2) / 2
    f = torch.tensor([2.0 / 3, 4.0 / 3])
    f_pred = torch.tensor([1.0 / 3, 1.0 / 3])
    weights = torch.tensor([1.0, 2.0])
    expected = ((f - f_pred) ** 2 * weights).mean()
    assert torch.isclose(loss, expected, atol=1e-5)


def test_regressor_huber_loss(batch):
    """Huber loss should be used for forces when force_loss_type='huber'."""
    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[OffsetHead("forces", 2.0)],
        force_loss_type="huber",
        huber_delta=1.0,
        mask_forces=False,
    )
    batch.forces = torch.randn_like(batch.pos)

    loss = model.loss(batch)["loss"]

    # With offset=2.0 and delta=1.0 every element is in the linear region:
    # huber(2.0, delta=1.0) = delta*(|error| - 0.5*delta) = 1*(2 - 0.5) = 1.5
    expected = torch.tensor(1.5)
    assert torch.isclose(loss, expected, atol=1e-5)


def test_regressor_masked_atoms_excluded_from_loss(batch):
    """Masked atoms should not contribute to the force loss."""
    if not hasattr(batch, "mask") or not batch.mask.any():
        pytest.skip("batch has no masked atoms")

    # Use offset=0 so prediction == target for non-masked atoms → loss = 0.
    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[OffsetHead("forces", 0.0)],
        mask_forces=True,
    )
    batch.forces = torch.randn_like(batch.pos)

    loss = model.loss(batch)["loss"]

    # Non-masked atoms: pred == target → zero loss.
    # If masked atoms were included, their target forces (non-zero) would
    # inflate the loss above zero.
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_regressor_invalid_loss_type():
    with pytest.raises(ValueError, match="force_loss_type"):
        RegressorModel(
            translator=DummyTranslator(),
            representation=DummyRepresentation(),
            heads=[OffsetHead("forces", 1.0)],
            force_loss_type="invalid",
        )

