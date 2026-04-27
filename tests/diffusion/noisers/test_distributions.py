import pytest
import torch

from agedi.diffusion.distributions import (
    StandardNormal, Normal, TruncatedNormal, Uniform, UniformCell,
    PriorDistribution, NoiseDistribution, UniformCellConfined, Constant,
)
from agedi.diffusion.distributions.categorical import Categorical
from agedi.diffusion.distributions.normal import WrappedNormal


# ---------------------------------------------------------------------------
# PriorDistribution / NoiseDistribution type hierarchy
# ---------------------------------------------------------------------------

def test_prior_subclasses():
    """Verify that the expected classes are instances of PriorDistribution."""
    for cls in (StandardNormal, UniformCell, UniformCellConfined, Constant):
        assert issubclass(cls, PriorDistribution), f"{cls.__name__} should be a PriorDistribution"


def test_noise_sampler_subclasses():
    """Verify that the expected classes are instances of NoiseDistribution."""
    for cls in (Normal, TruncatedNormal, WrappedNormal, Categorical):
        assert issubclass(cls, NoiseDistribution), f"{cls.__name__} should be a NoiseDistribution"


def test_prior_sample_interface(batch):
    """PriorDistribution.sample(batch) should return a tensor of the right shape."""
    d = UniformCell()
    result = d.sample(batch)
    assert result.shape == batch.pos.shape


def test_standard_normal(batch) -> None:
    d = StandardNormal(key="pos")
    result = d.sample(batch)
    assert result.shape == batch.pos.shape


def test_normal(batch) -> None:
    d = Normal()
    mu = batch.pos
    sigma = torch.ones_like(batch.pos)
    assert d.sample(batch, mu=mu, sigma=sigma).shape == (batch.num_nodes, 3)


def test_truncated_normal(batch: "Batch") -> None:
    min_val, max_val = batch.pos.min(), batch.pos.max()
    batch.confinement = torch.tensor([min_val, max_val]).repeat(batch.num_graphs, 1)

    d = TruncatedNormal()
    mu = batch.pos
    sigma = torch.ones((batch.num_nodes, 3))
    print(batch.pos[:,2])
    print(d.sample(batch, mu=mu, sigma=sigma)[:,2])
    
    assert (d.sample(batch, mu=mu, sigma=sigma)[:,2] < max_val).all()
    assert (d.sample(batch, mu=mu, sigma=sigma)[:,2] > min_val).all()


def test_uniform(batch) -> None:
    d = Uniform(key="pos")
    result = d.sample(batch)
    assert result.shape == batch.pos.shape


def test_cell_uniform(batch: "Batch") -> None:
    d = UniformCell()
    result = d.sample(batch)
    assert result.shape == (batch.num_nodes, 3)


# ---------------------------------------------------------------------------
# TruncatedNormal: out-of-bounds mu clamping
# ---------------------------------------------------------------------------

def test_truncated_normal_out_of_bounds_mu_does_not_raise(batch: "Batch") -> None:
    """Sampling must succeed even when mu_z is outside [z_lo, z_hi]."""
    z_lo, z_hi = 1.0, 5.0
    batch.confinement = torch.tensor([[z_lo, z_hi]]).expand(batch.num_graphs, -1).clone()
    d = TruncatedNormal()

    # Push mu way outside bounds
    mu = batch.pos.clone()
    mu[:, 2] = 100.0
    sigma = torch.ones_like(mu)

    # Should not raise ValueError
    samples = d.sample(batch, mu=mu, sigma=sigma)
    assert (samples[~batch.mask, 2] >= z_lo - 1e-4).all()
    assert (samples[~batch.mask, 2] <= z_hi + 1e-4).all()


def test_truncated_normal_samples_within_bounds_near_boundary(batch: "Batch") -> None:
    """Samples must be within bounds when mu is very close to (but inside) bounds."""
    z_lo, z_hi = 2.0, 4.0
    batch.confinement = torch.tensor([[z_lo, z_hi]]).expand(batch.num_graphs, -1).clone()
    d = TruncatedNormal()

    mu = batch.pos.clone()
    # Place mu right at the boundary
    mu[:, 2] = z_lo + 1e-5
    sigma = torch.ones_like(mu)

    samples = d.sample(batch, mu=mu, sigma=sigma)
    assert (samples[~batch.mask, 2] >= z_lo - 1e-4).all()
    assert (samples[~batch.mask, 2] <= z_hi + 1e-4).all()
