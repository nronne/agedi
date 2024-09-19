import pytest
import torch

from agedi.diffusion.noisers import PositionsNoiser

def test_init_noiser() -> None:
    noiser = PositionsNoiser()
    assert noiser is not None

def test_noise_no_time(batch: "Batch") -> None:
    noiser = PositionsNoiser(key="positions")
    with pytest.raises(TypeError):
        noiser.noise(batch)
        
def test_noise(batch: "Batch") -> None:
    batch.time = torch.rand((batch.num_graphs, 1))[batch.batch]
    noiser = PositionsNoiser(key="positions")
    noised = noiser.noise(batch)
    assert "positions_noise" in noised.keys()

def test_denoise_no_score(batch: "Batch") -> None:
    noiser = PositionsNoiser(key="positions")
    with pytest.raises(TypeError):
        noiser.denoise(batch)

def test_denoise(batch: "Batch") -> None:
    pos = batch.pos.clone()
    batch.time = torch.rand((batch.num_graphs, 1))[batch.batch]
    batch.positions_score = torch.randn_like(batch.pos)
    noiser = PositionsNoiser(key="positions")
    noiser.denoise(batch, torch.tensor(0.001))
    assert not torch.allclose(pos, batch.pos)

def test_loss(batch: "Batch") -> None:
    batch.time = torch.rand((batch.num_graphs, 1))[batch.batch]    
    pos = batch.pos.clone()
    
    noiser = PositionsNoiser(key="positions")

    noised = noiser.noise(batch)
    noised.positions_score = torch.randn_like(batch.pos)
    l = noiser.loss(noised)
    assert l > 0
        
