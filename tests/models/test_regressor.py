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



# ---------------------------------------------------------------------------
# Conservative forces (F = -dE/dR)
# ---------------------------------------------------------------------------


class QuadraticEnergyHead(Head):
    """Toy energy head with a known analytic gradient: E = sum_i |R_i|^2."""

    _key = "energy"

    def _score(self, translated_batch):
        graph = translated_batch["batch"]
        atomic = (graph.pos**2).sum(-1)
        energy = torch.zeros(
            graph.num_graphs, dtype=atomic.dtype, device=atomic.device
        )
        energy.scatter_add_(0, graph.batch, atomic)
        return energy


def test_conservative_forces_rejects_forces_head():
    with pytest.raises(ValueError):
        RegressorModel(
            translator=DummyTranslator(),
            representation=DummyRepresentation(),
            heads=[QuadraticEnergyHead(), OffsetHead("forces", 1.0)],
            conservative_forces=True,
        )


def test_conservative_forces_requires_energy_head():
    with pytest.raises(ValueError):
        RegressorModel(
            translator=DummyTranslator(),
            representation=DummyRepresentation(),
            heads=[OffsetHead("forces", 1.0)],
            conservative_forces=True,
        )


def test_conservative_forces_analytic_gradient(batch):
    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[QuadraticEnergyHead()],
        conservative_forces=True,
        mask_forces=False,
    )
    model.eval()

    original_pos = batch._store["pos"]

    out = model.forward(batch)

    assert "forces_prediction" in out.keys()
    assert torch.allclose(out.forces_prediction, -2.0 * original_pos, atol=1e-5)

    # The batch must be restored exactly: no leftover grad tracking, and the
    # cached neighbour list must survive (the swap must bypass the pos
    # setter's clear_graph()).
    assert out._store["pos"] is original_pos
    assert not out.pos.requires_grad
    assert "edge_index" in out._store
    assert "shift_vectors" in out._store


def test_conservative_forces_mask_forces(batch):
    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[QuadraticEnergyHead()],
        conservative_forces=True,
        mask_forces=True,
    )
    model.eval()

    out = model.forward(batch)

    if hasattr(batch, "mask"):
        assert torch.all(out.forces_prediction[out.positions_mask] == 0.0)


def test_conservative_forces_trains_energy_head(batch):
    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[QuadraticEnergyHead()],
        conservative_forces=True,
        mask_forces=False,
    )
    model.train()
    batch.forces = torch.zeros_like(batch.pos)
    batch.energy = torch.zeros(batch.num_graphs)

    loss = model.loss(batch)["loss"]
    loss.backward()

    # No learnable parameters on the toy head/representation, but the
    # gradient must flow through the double backward without erroring, and
    # the "forces" loss term must actually be present.
    assert "forces_loss" in model.loss(batch)


def test_conservative_forces_get_config_round_trip():
    model = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[QuadraticEnergyHead()],
        conservative_forces=True,
    )
    config = model.get_config()
    assert config["conservative_forces"] is True

    rebuilt = RegressorModel(
        translator=DummyTranslator(),
        representation=DummyRepresentation(),
        heads=[QuadraticEnergyHead()],
        **config,
    )
    assert rebuilt.get_config() == config


def test_conservative_forces_matches_finite_difference(package, cutoff):
    """End-to-end check against the real SchNetPack/PaiNN backend."""
    from ase.build import molecule
    from torch_geometric.data import Batch as PyGBatch

    from agedi.data import AtomsGraph
    from agedi.models.schnetpack.regressor_heads import Energy

    torch.manual_seed(0)
    translator, representation, _ = package

    model = RegressorModel(
        translator=translator,
        representation=representation,
        heads=[Energy(input_dim_scalar=64)],
        conservative_forces=True,
        mask_forces=False,
    ).double()
    model.eval()

    a = molecule("H2O")
    a.set_cell([10, 10, 10])
    a.set_pbc(True)
    a.center()
    graph = AtomsGraph.from_atoms(a, cutoff=cutoff)
    graph._store["pos"] = graph._store["pos"].double()
    graph._store["cell"] = graph._store["cell"].double()
    graph._store["shift_vectors"] = graph._store["shift_vectors"].double()
    graph_batch = PyGBatch.from_data_list([graph])

    with torch.no_grad():
        analytic = model(graph_batch).forces_prediction.clone()

    def energy_at(pos: torch.Tensor) -> float:
        original = graph_batch._store["pos"]
        graph_batch._store["pos"] = pos
        with torch.no_grad():
            energy = model._forward_heads(graph_batch).energy_prediction.item()
        graph_batch._store["pos"] = original
        return energy

    delta = 1e-4
    base_pos = graph_batch.pos
    numeric = torch.zeros_like(base_pos)
    for i in range(base_pos.shape[0]):
        for d in range(3):
            pos_p = base_pos.clone()
            pos_p[i, d] += delta
            pos_m = base_pos.clone()
            pos_m[i, d] -= delta
            numeric[i, d] = -(energy_at(pos_p) - energy_at(pos_m)) / (2 * delta)

    assert torch.allclose(analytic, numeric, atol=1e-5)
