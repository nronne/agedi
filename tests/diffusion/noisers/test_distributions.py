import pytest
import torch

from agedi.diffusion.noisers.distributions import StandardNormal, Normal, TruncatedNormal, Uniform, UniformCell

def test_standard_normal() -> None:
    d = StandardNormal()
    assert d._sample(torch.rand((10, 3)), 1).shape == (10, 3)

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
    assert d._sample(torch.rand((10, 3)), 1).shape == (10, 3)

def test_cell_uniform(batch: "Batch") -> None:
    d = UniformCell()
    c = d.get_callable(batch)
    assert c(batch.pos, torch.ones((batch.num_nodes, 3))).shape == (batch.num_nodes, 3)
