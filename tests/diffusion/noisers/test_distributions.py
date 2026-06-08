import pytest
import torch

from agedi.diffusion.distributions import StandardNormal, Normal, TruncatedNormal, Uniform, UniformCell

def test_standard_normal() -> None:
    d = StandardNormal()
    assert d._sample((10,3)).shape == (10, 3)

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


# ---------------------------------------------------------------------------
# ZeroComNormal and ZeroComStandardNormal
# ---------------------------------------------------------------------------

def test_zero_com_normal_zero_com_per_graph(batch: "Batch") -> None:
    """ZeroComNormal noise increment must have zero COM for every graph."""
    from agedi.diffusion.distributions import ZeroComNormal
    d = ZeroComNormal()
    d._setup(batch)
    mu = batch.pos
    sigma = torch.ones_like(mu)
    samples = d._sample(mu, sigma)
    noise = samples - mu
    n_graphs = batch.num_graphs
    for g in range(n_graphs):
        mask_g = batch.batch == g
        com = noise[mask_g].mean(0)
        assert torch.allclose(com, torch.zeros(3), atol=1e-5), (
            f"ZeroComNormal: COM of graph {g} is {com.tolist()}, expected zero"
        )


def test_zero_com_standard_normal_zero_com_per_graph(batch: "Batch") -> None:
    """ZeroComStandardNormal samples must have zero COM for every graph."""
    from agedi.diffusion.distributions import ZeroComStandardNormal
    d = ZeroComStandardNormal(scale=1.0)
    d._setup(batch)
    shape = torch.Size([batch.num_nodes, 3])
    samples = d._sample(shape=shape)
    n_graphs = batch.num_graphs
    for g in range(n_graphs):
        mask_g = batch.batch == g
        com = samples[mask_g].mean(0)
        assert torch.allclose(com, torch.zeros(3), atol=1e-5), (
            f"ZeroComStandardNormal: COM of graph {g} is {com.tolist()}, expected zero"
        )


def test_zero_com_standard_normal_respects_scale(batch: "Batch") -> None:
    """ZeroComStandardNormal samples should have std ≈ scale (up to COM projection)."""
    from agedi.diffusion.distributions import ZeroComStandardNormal
    scale = 3.0
    d = ZeroComStandardNormal(scale=scale)
    d._setup(batch)
    shape = torch.Size([batch.num_nodes, 3])
    # Draw many samples and check std is in a reasonable range.
    torch.manual_seed(0)
    samples = torch.stack([d._sample(shape=shape) for _ in range(500)])
    std = samples.std().item()
    # After COM projection the effective std is slightly less than scale
    assert std > 0.5 * scale and std < 1.5 * scale, (
        f"ZeroComStandardNormal scale={scale}: sample std={std:.3f} out of expected range"
    )


def test_standard_normal_scale_parameter() -> None:
    """StandardNormal should respect the scale parameter."""
    from agedi.diffusion.distributions import StandardNormal
    scale = 5.0
    d = StandardNormal(scale=scale)
    torch.manual_seed(0)
    samples = torch.stack([d._sample((100, 3)) for _ in range(50)])
    std = samples.std().item()
    assert abs(std - scale) < 1.0, f"StandardNormal(scale={scale}): std={std:.3f}"


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
