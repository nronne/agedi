"""Tests for SDENoiser concrete subclass covering _noise, _denoise, _loss."""
import torch
import pytest

from agedi.diffusion.noisers.sde import SDENoiser
from agedi.diffusion.sdes import VE
from agedi.diffusion.distributions import Normal
from agedi.diffusion.distributions.constant import Constant


class ConcreteSDE(SDENoiser):
    """Minimal concrete implementation of the abstract SDENoiser."""

    _key = "pos"

    def postprocess_score(self, score):
        return score

    def postprocess_noise(self, noise):
        return noise


@pytest.fixture
def sde_noiser():
    return ConcreteSDE(
        sde=VE(),
        distribution=Normal(),
        prior=Normal(),
    )


def test_sde_noiser_init(sde_noiser):
    assert sde_noiser is not None
    assert sde_noiser.key == "pos"


def test_sde_noiser_noise_adds_noise_key(sde_noiser, batch):
    batch.time = torch.rand((batch.num_nodes, 1))
    out = sde_noiser.noise(batch)
    assert "pos_noise" in out.keys()
    assert out.pos_noise.shape == batch.pos.shape


def test_sde_noiser_denoise_last(sde_noiser, batch):
    batch.time = torch.rand((batch.num_nodes, 1))
    batch.pos_score = torch.randn_like(batch.pos)
    pos_before = batch.pos.clone()
    sde_noiser.denoise(batch, delta_t=torch.tensor(0.001), last=True)
    assert not torch.allclose(batch.pos, pos_before)


def test_sde_noiser_denoise_step(sde_noiser, batch):
    batch.time = torch.rand((batch.num_nodes, 1))
    batch.pos_score = torch.randn_like(batch.pos)
    pos_before = batch.pos.clone()
    sde_noiser.denoise(batch, delta_t=torch.tensor(0.001), last=False)
    assert not torch.allclose(batch.pos, pos_before)


def test_sde_noiser_loss_positive(sde_noiser, batch):
    batch.time = torch.rand((batch.num_nodes, 1))
    out = sde_noiser.noise(batch)
    out.pos_score = torch.randn_like(out.pos)
    loss = sde_noiser.loss(out)
    assert loss > 0
