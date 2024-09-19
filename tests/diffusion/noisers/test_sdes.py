import torch
import pytest

from agedi.diffusion.noisers import VP
from agedi.diffusion.noisers.sdes import SDE
from agedi.diffusion.noisers.distributions import Normal

def test_init():
    vp = VP()
    assert isinstance(vp, SDE)

def test_VP_beta() -> None:
    vp = VP()
    assert vp.beta(torch.tensor([0.0])) == vp.beta_min
    assert vp.beta(torch.tensor([1.0])) == vp.beta_max

def test_VP_alpha() -> None:
    vp = VP()
    assert vp.alpha(torch.tensor([0.0])) == 0.0
    assert vp.alpha(torch.tensor([1.0])) == vp.beta_min + 0.5 * (vp.beta_max - vp.beta_min)
    
def test_VP_drift() -> None:
    vp = VP()
    x = torch.randn((10, 3))
    assert vp.drift(x, torch.rand((10,1))).shape == x.shape

def test_VP_diffusion() -> None:
    vp = VP()
    assert vp.diffusion(torch.rand((10,1))).shape == (10, 1)
    
def test_VP_mean() -> None:
    vp = VP()
    t = torch.rand((10,1))
    assert vp.mean(t).shape == t.shape

def test_VP_var() -> None:
    vp = VP()
    t = torch.rand((10,1))
    assert vp.var(t).shape == t.shape
    
def test_SDE_transition_kernel(batch: "Batch") -> None:
    vp = VP()
    x = torch.randn((10, 3))
    t = torch.rand((10,1))
    w = Normal().get_callable(batch)
    assert vp.transition_kernel(x, t, w).shape == x.shape

def test_noise() -> None:
    vp = VP()
    
    t = torch.rand((10,1))
    x = torch.randn((10, 3))
    w = torch.randn((10, 3))
    xt = vp.mean(t) * x + torch.sqrt(vp.var(t)) * w
    
    assert torch.allclose(vp.noise(x, xt, t), w, atol=1e-5)
    

