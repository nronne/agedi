import pytest
import torch

from agedi.diffusion.distributions import StandardNormal, Normal, TruncatedNormal, Uniform, UniformCell

def test_standard_normal() -> None:
    d = StandardNormal()
    assert d._sample((10,3)).shape == (10, 3)


def test_standard_normal_per_molecule_std(batch: "Batch") -> None:
    """Prior std must scale per-molecule (cube-root of each graph's atom count).

    When the batch contains graphs of different sizes the standard deviation
    of samples belonging to a small graph must be strictly less than those
    belonging to a large graph.
    """
    from agedi.data import AtomsGraph
    import torch_geometric.data as tgd

    # Build two single-graph AtomsGraphs with very different atom counts so
    # the stds are clearly distinguishable.
    small_n, large_n = 2, 64

    def _make_graph(n: int) -> AtomsGraph:
        g = AtomsGraph.empty(cutoff=6.0)
        g.n_atoms = torch.tensor([n])
        g.pos = torch.zeros(n, 3)
        return g

    small_g = _make_graph(small_n)
    large_g = _make_graph(large_n)
    batched = tgd.Batch.from_data_list([small_g, large_g])

    d = StandardNormal()
    d.key = "pos"
    d._setup(batched)

    assert d._per_atom_std.shape == (small_n + large_n,)

    small_std = d._per_atom_std[:small_n].mean().item()
    large_std = d._per_atom_std[small_n:].mean().item()

    expected_small = 0.8 * small_n ** (1 / 3)
    expected_large = 0.8 * large_n ** (1 / 3)

    assert abs(small_std - expected_small) < 1e-5, f"{small_std} != {expected_small}"
    assert abs(large_std - expected_large) < 1e-5, f"{large_std} != {expected_large}"
    assert large_std > small_std, "larger molecule should have larger prior std"

    # Verify that _sample() produces a tensor with the right shape
    samples = d._sample()
    assert samples.shape == (small_n + large_n, 3)

def test_normal() -> None:
    d = Normal()
    assert d._sample(torch.rand((10, 3)), 1).shape == (10, 3)

def test_truncated_normal(batch: "Batch") -> None:
    min_val, max_val = batch.pos.min(), batch.pos.max()
    batch.confinement = torch.tensor([min_val, max_val]).repeat(batch.num_graphs, 1)
    d = TruncatedNormal()
    d._setup(batch)

    mu = batch.pos
    sigma = torch.ones((batch.num_nodes, 3))
    print(batch.pos[:,2])
    print(d._sample(mu, sigma)[:,2])
    
    assert (d._sample(mu, sigma)[:,2] < max_val).all()
    assert (d._sample(mu, sigma)[:,2] > min_val).all()

    
def test_get_callable(batch: "Batch") -> None:
    d = Normal()
    c = d.get_callable(batch)
    assert c(batch.pos, torch.ones((batch.num_nodes, 3))).shape == (batch.num_nodes, 3)


def test_uniform() -> None:
    d = Uniform()
    assert d._sample(shape=(10, 3)).shape == (10, 3)

def test_cell_uniform(batch: "Batch") -> None:
    d = UniformCell()
    c = d.get_callable(batch)
    assert c().shape == (batch.num_nodes, 3)


# ---------------------------------------------------------------------------
# TruncatedNormal: out-of-bounds mu clamping
# ---------------------------------------------------------------------------

def test_truncated_normal_out_of_bounds_mu_does_not_raise(batch: "Batch") -> None:
    """Sampling must succeed even when mu_z is outside [z_lo, z_hi]."""
    z_lo, z_hi = 1.0, 5.0
    batch.confinement = torch.tensor([[z_lo, z_hi]]).expand(batch.num_graphs, -1).clone()
    d = TruncatedNormal()
    d._setup(batch)

    # Push mu way outside bounds
    mu = batch.pos.clone()
    mu[:, 2] = 100.0
    sigma = torch.ones_like(mu)

    # Should not raise ValueError
    samples = d._sample(mu, sigma)
    assert (samples[~batch.mask, 2] >= z_lo - 1e-4).all()
    assert (samples[~batch.mask, 2] <= z_hi + 1e-4).all()


def test_truncated_normal_samples_within_bounds_near_boundary(batch: "Batch") -> None:
    """Samples must be within bounds when mu is very close to (but inside) bounds."""
    z_lo, z_hi = 2.0, 4.0
    batch.confinement = torch.tensor([[z_lo, z_hi]]).expand(batch.num_graphs, -1).clone()
    d = TruncatedNormal()
    d._setup(batch)

    mu = batch.pos.clone()
    # Place mu right at the boundary
    mu[:, 2] = z_lo + 1e-5
    sigma = torch.ones_like(mu)

    samples = d._sample(mu, sigma)
    assert (samples[~batch.mask, 2] >= z_lo - 1e-4).all()
    assert (samples[~batch.mask, 2] <= z_hi + 1e-4).all()
