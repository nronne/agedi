"""Tests for the Categorical distribution."""
import torch
from agedi.diffusion.distributions.categorical import Categorical


def test_categorical_sample_valid_class(batch):
    d = Categorical()
    probs = torch.zeros(batch.num_nodes, 10)
    probs[:, 3] = 1.0
    out = d.sample(batch, probs=probs)
    assert out.shape == (batch.num_nodes,)
    assert (out == 3).all()


def test_categorical_sample_returns_valid_classes(batch):
    d = Categorical()
    probs = torch.softmax(torch.randn((batch.num_nodes, 100)), dim=-1)
    out = d.sample(batch, probs=probs)
    assert out.shape == (batch.num_nodes,)
    assert ((out >= 0) & (out < 100)).all()
