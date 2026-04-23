import pytest
import torch

from agedi.diffusion.noisers import PositionsNoiser, Positions, CellPositions, ConfinedCellPositions

@pytest.fixture(params=['true', 'false'])
def last(request: str) -> bool:
    if request.param == 'true':
        return True
    else:
        return False
        
def test_init_noiser() -> None:
    noiser = PositionsNoiser()
    assert noiser is not None

def test_noise_no_time(batch: "Batch") -> None:
    noiser = PositionsNoiser()
    with pytest.raises(TypeError):
        noiser.noise(batch)
        
def test_noise(batch: "Batch") -> None:
    batch.time = torch.rand((batch.num_graphs, 1))[batch.batch]
    noiser = PositionsNoiser()
    noised = noiser.noise(batch)
    assert "pos_noise" in noised.keys()

def test_denoise_no_score(batch: "Batch", last: bool) -> None:
    noiser = PositionsNoiser()
    with pytest.raises(KeyError):
        noiser.denoise(batch, torch.tensor(0.001), last=last)

def test_denoise(batch: "Batch", last: bool) -> None:
    pos = batch.pos.clone()
    batch.time = torch.rand((batch.num_graphs, 1))[batch.batch]
    batch.pos_score = torch.randn_like(batch.pos)
    noiser = PositionsNoiser()
    noiser.denoise(batch, torch.tensor(0.001), last=last)
    assert not torch.allclose(pos, batch.pos)

def test_loss(batch: "Batch") -> None:
    batch.time = torch.rand((batch.num_graphs, 1))[batch.batch]    
    pos = batch.pos.clone()
    
    noiser = PositionsNoiser()

    noised = noiser.noise(batch)
    noised.pos_score = torch.randn_like(batch.pos)
    l = noiser.loss(noised)
    assert l > 0


# ---------------------------------------------------------------------------
# Tests for the named positions-noiser subclasses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [Positions, CellPositions])
def test_named_noiser_init(cls):
    noiser = cls()
    assert noiser is not None


def test_positions_uses_standard_normal_prior():
    from agedi.diffusion.distributions import StandardNormal
    noiser = Positions()
    assert isinstance(noiser.prior, StandardNormal)


def test_cell_positions_uses_uniform_cell_prior():
    from agedi.diffusion.distributions import UniformCell
    noiser = CellPositions()
    assert isinstance(noiser.prior, UniformCell)


def test_confined_cell_positions_uses_uniform_cell_confined_prior():
    from agedi.diffusion.distributions import UniformCellConfined, TruncatedNormal
    noiser = ConfinedCellPositions()
    assert isinstance(noiser.prior, UniformCellConfined)
    assert isinstance(noiser.distribution, TruncatedNormal)


def test_named_noiser_get_hparams():
    for cls in (Positions, CellPositions, ConfinedCellPositions):
        noiser = cls()
        hp = noiser.get_hparams()
        assert "_target_" in hp
        assert cls.__qualname__ in hp["_target_"]
        assert "sde" in hp
        assert "loss_scaling" in hp
        # distribution and prior are fixed – not needed for reconstruction
        assert "distribution" not in hp
        assert "prior" not in hp


def test_cell_positions_noise(batch: "Batch") -> None:
    batch.time = torch.rand((batch.num_graphs, 1))[batch.batch]
    noiser = CellPositions()
    noised = noiser.noise(batch)
    assert "pos_noise" in noised.keys()


def test_cell_positions_loss(batch: "Batch") -> None:
    batch.time = torch.rand((batch.num_graphs, 1))[batch.batch]
    noiser = CellPositions()
    noised = noiser.noise(batch)
    noised.pos_score = torch.randn_like(batch.pos)
    assert noiser.loss(noised) > 0


# ---------------------------------------------------------------------------
# Confinement clamping in _denoise
# ---------------------------------------------------------------------------

def test_denoise_confinement_clamps_mobile_z(batch: "Batch") -> None:
    """After a denoise step, mobile-atom z coords must stay within [z_min, z_max]."""
    z_lo, z_hi = 1.0, 5.0
    n_graphs = batch.num_graphs
    batch.confinement = torch.tensor([[z_lo, z_hi]]).expand(n_graphs, -1).clone()

    # Push all z values way outside the confinement region
    batch.pos[:, 2] = 100.0
    batch.time = torch.rand((n_graphs, 1))[batch.batch]
    batch.pos_score = torch.zeros_like(batch.pos)

    noiser = PositionsNoiser()
    noiser.denoise(batch, torch.tensor(0.001), last=True)

    mobile = ~batch.mask
    z_mobile = batch.pos[mobile, 2]
    assert (z_mobile >= z_lo - 1e-6).all(), "mobile atom z dropped below z_min"
    assert (z_mobile <= z_hi + 1e-6).all(), "mobile atom z rose above z_max"


def test_denoise_confinement_does_not_clamp_masked_atoms(batch: "Batch") -> None:
    """Masked (fixed) atoms must not be clamped by the confinement logic."""
    z_lo, z_hi = 1.0, 5.0
    n_graphs = batch.num_graphs
    batch.confinement = torch.tensor([[z_lo, z_hi]]).expand(n_graphs, -1).clone()

    # Push all z values outside bounds
    batch.pos[:, 2] = 100.0
    original_masked_z = batch.pos[batch.mask, 2].clone()

    batch.time = torch.rand((n_graphs, 1))[batch.batch]
    batch.pos_score = torch.zeros_like(batch.pos)

    noiser = PositionsNoiser()
    noiser.denoise(batch, torch.tensor(0.001), last=True)

    # Masked atoms should be unchanged (drift=0, score=0, so pos stays at 100)
    assert torch.allclose(batch.pos[batch.mask, 2], original_masked_z, atol=1e-3), (
        "masked atoms should not be affected by confinement clamping"
    )

