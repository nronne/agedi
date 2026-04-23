"""Tests for Constant and Uniform (including UniformCellConfined) distributions."""
import torch
import pytest
from ase.build import molecule

from agedi.data import AtomsGraph
from agedi.diffusion.distributions.constant import Constant
from agedi.diffusion.distributions.uniform import Uniform, UniformCell, UniformCellConfined


def _build_single_graph():
    atoms = molecule("H2O")
    atoms.set_cell([8.0, 8.0, 8.0])
    atoms.set_pbc(True)
    atoms.center()
    return AtomsGraph.from_atoms(atoms)


# ── Constant ─────────────────────────────────────────────────────────────────

class TestConstant:
    def test_sample_with_explicit_shape(self):
        d = Constant(value=5, dtype=torch.int64)
        out = d._sample(shape=(4,))
        assert out.shape == (4,)
        assert (out == 5).all()

    def test_sample_with_batch_setup(self, batch):
        d = Constant(value=0, key="x")
        callable_fn = d.get_callable(batch)
        out = callable_fn()
        assert out.shape[0] == batch.num_nodes
        assert (out == 0).all()

    def test_default_value_is_zero(self):
        d = Constant()
        out = d._sample(shape=(10,))
        assert (out == 0).all()


# ── Uniform ──────────────────────────────────────────────────────────────────

class TestUniform:
    def test_sample_explicit_shape(self):
        d = Uniform(low=2.0, high=5.0)
        out = d._sample(shape=(8, 3))
        assert out.shape == (8, 3)
        assert (out >= 2.0).all()
        assert (out <= 5.0).all()

    def test_sample_uses_stored_shape(self, batch):
        d = Uniform(key="x")
        c = d.get_callable(batch)
        out = c()
        assert out.shape == batch.x.shape


# ── UniformCell non-batch path ────────────────────────────────────────────────

class TestUniformCellNonBatch:
    def test_sample_returns_correct_shape(self):
        graph = _build_single_graph()
        d = UniformCell()
        # non-batch: graph.batch is None
        graph_batch = graph.clone()
        object.__setattr__(graph_batch, "batch", None)
        d._setup(graph_batch)
        out = d._sample()
        n_atoms = graph.n_atoms.item()
        assert out.shape == (n_atoms, 3)


# ── UniformCellConfined ───────────────────────────────────────────────────────

class TestUniformCellConfined:
    def test_setup_adjusts_cell_and_corner(self):
        graph = _build_single_graph()
        graph.confinement = torch.tensor([[1.0, 7.0]])
        # Non-batch path
        object.__setattr__(graph, "batch", None)
        d = UniformCellConfined()
        d._setup(graph)
        assert d.corner[0, 2].item() == pytest.approx(1.0)
        assert d.cell[2, 2].item() == pytest.approx(6.0)

    def test_setup_raises_when_confinement_is_none(self, batch):
        batch.confinement = None
        d = UniformCellConfined()
        with pytest.raises(ValueError, match="confinement"):
            d._setup(batch)

    def test_setup_raises_when_confinement_wrong_shape(self, batch):
        # Provide only 1 row regardless of how many graphs are in the batch
        batch.confinement = torch.tensor([[1.0, 7.0]])
        d = UniformCellConfined()
        with pytest.raises(ValueError, match="confinement"):
            d._setup(batch)

    def test_batched_samples_within_bounds(self, batch):
        """Batched UniformCellConfined should produce z-coordinates in per-graph bounds."""
        n_graphs = batch.num_graphs
        # Give each graph distinct z bounds
        z_lo = torch.arange(n_graphs, dtype=torch.float) * 2.0 + 1.0
        z_hi = z_lo + 3.0
        batch.confinement = torch.stack([z_lo, z_hi], dim=1)

        d = UniformCellConfined()
        sampler = d.get_callable(batch)
        samples = sampler()  # (n_atoms, 3)

        # Check every atom's z is within its graph's bounds
        for g in range(n_graphs):
            mask = batch.batch == g
            z_vals = samples[mask, 2]
            assert (z_vals >= z_lo[g].item() - 1e-6).all(), (
                f"graph {g}: z below lower bound"
            )
            assert (z_vals <= z_hi[g].item() + 1e-6).all(), (
                f"graph {g}: z above upper bound"
            )

    def test_batched_matches_single_graph(self):
        """Batched path should produce the same distribution as the single-graph path."""
        graph = _build_single_graph()
        # Build a two-graph batch from identical graphs
        graph2 = graph.clone()
        from torch_geometric.data import Batch

        z_lo, z_hi = 1.0, 7.0
        for g in (graph, graph2):
            g.confinement = torch.tensor([[z_lo, z_hi]])

        batched = Batch.from_data_list([graph, graph2])

        d_batch = UniformCellConfined()
        sampler = d_batch.get_callable(batched)
        samples = sampler()

        # All z values for both graphs must be within [z_lo, z_hi]
        assert (samples[:, 2] >= z_lo - 1e-6).all()
        assert (samples[:, 2] <= z_hi + 1e-6).all()

        # Single-graph path should also be within bounds
        object.__setattr__(graph, "batch", None)
        d_single = UniformCellConfined()
        d_single._setup(graph)
        s_single = d_single._sample()
        assert (s_single[:, 2] >= z_lo - 1e-6).all()
        assert (s_single[:, 2] <= z_hi + 1e-6).all()
