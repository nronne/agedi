"""Tests for the Cell noiser."""
import pytest
import torch
from torch_geometric.data import Batch

from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Cell
from agedi.diffusion.distributions import StandardNormal


@pytest.fixture
def cell_batch(batch):
    """Add per-node time attribute and a graph-level cell score."""
    # Time is per-node; Cell noiser reads time at the first atom of each graph.
    batch.time = torch.full((batch.num_nodes, 1), 0.5)
    n_graphs = batch.num_graphs
    batch.cell_score = torch.randn(n_graphs * 3, 3)
    return batch


def test_cell_noiser_init():
    noiser = Cell()
    assert noiser is not None
    assert noiser.key == "cell"


def test_initialize_graph_calls_prior():
    """Cell.initialize_graph must delegate to self.prior, not bypass it."""
    called = []

    class TrackingPrior(StandardNormal):
        def sample(self, batch, **kwargs):
            called.append(True)
            return super().sample(batch, **kwargs)

    noiser = Cell(prior=TrackingPrior())
    graph = AtomsGraph.empty(cutoff=6.0)
    noiser.initialize_graph(graph)

    assert len(called) == 1, "Prior.sample was not called during initialize_graph"


def test_initialize_graph_lower_triangular():
    """Cell initialized from prior must be lower triangular."""
    noiser = Cell()
    graph = AtomsGraph.empty(cutoff=6.0)
    noiser.initialize_graph(graph)

    cell = graph.cell.view(3, 3)
    upper_mask = ~torch.ones(3, 3, dtype=torch.bool).tril()
    assert (cell[upper_mask] == 0).all(), "Initialized cell is not lower-triangular"


def test_noise_preserves_frac_coords(cell_batch):
    """Noising the cell must keep fractional coordinates unchanged."""
    frac_before = cell_batch.frac.clone()
    noiser = Cell()
    noiser.noise(cell_batch)
    frac_after = cell_batch.frac
    assert torch.allclose(frac_before, frac_after, atol=1e-5), (
        "Fractional coordinates changed after Cell.noise; "
        f"max diff = {(frac_after - frac_before).abs().max().item():.2e}"
    )


def test_noise_cell_lower_triangular(cell_batch):
    """The noised cell must remain lower-triangular."""
    noiser = Cell()
    noiser.noise(cell_batch)
    n_graphs = cell_batch.num_graphs
    cell = cell_batch.cell.view(n_graphs, 3, 3)
    upper_mask = ~torch.ones(3, 3, dtype=torch.bool).tril()
    assert (cell[:, upper_mask] == 0).all(), "Noised cell is not lower-triangular"


def test_noise_adds_cell_noise_attr(cell_batch):
    """Cell.noise must store cell_noise on the batch."""
    noiser = Cell()
    out = noiser.noise(cell_batch)
    assert "cell_noise" in out.keys()


@pytest.mark.parametrize("last", [True, False])
def test_denoise_preserves_frac_coords(cell_batch, last):
    """Denoising the cell must keep fractional coordinates unchanged."""
    frac_before = cell_batch.frac.clone()
    noiser = Cell()
    noiser.denoise(cell_batch, delta_t=1e-3, last=last)
    frac_after = cell_batch.frac
    assert torch.allclose(frac_before, frac_after, atol=1e-5), (
        f"Fractional coordinates changed after Cell.denoise(last={last}); "
        f"max diff = {(frac_after - frac_before).abs().max().item():.2e}"
    )


@pytest.mark.parametrize("last", [True, False])
def test_denoise_cell_lower_triangular(cell_batch, last):
    """The denoised cell must remain lower-triangular."""
    noiser = Cell()
    noiser.denoise(cell_batch, delta_t=1e-3, last=last)
    n_graphs = cell_batch.num_graphs
    cell = cell_batch.cell.view(n_graphs, 3, 3)
    upper_mask = ~torch.ones(3, 3, dtype=torch.bool).tril()
    assert (cell[:, upper_mask] == 0).all(), "Denoised cell is not lower-triangular"


def test_loss_positive(cell_batch):
    """Cell loss should be positive after noising."""
    noiser = Cell()
    noiser.noise(cell_batch)
    cell_batch.cell_score = torch.randn_like(cell_batch.cell_noise)
    loss = noiser.loss(cell_batch)
    assert loss > 0
