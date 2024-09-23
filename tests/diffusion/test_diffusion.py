import pytest

from agedi.diffusion import Diffusion
from agedi.models import ScoreModel


def test_init(cutoff, package, conditionings, noisers):

    translator, representation, heads = package

    score_model = ScoreModel(
        translator=translator,
        representation=representation,
        conditionings=conditionings,
        heads=heads,
    )
    
    diffusion = Diffusion(score_model, noisers)
    
    assert diffusion is not None

def test_sample_time(diffusion, batch):
    diffusion.sample_time(batch)
    assert batch.time is not None
    
def test_forward(diffusion, batch):
    diffusion.sample_time(batch)
    out = diffusion(batch)
    assert out is not None

def test_loss(diffusion, batch):
    loss = diffusion.loss(batch, None)
    assert loss['loss'] > 0

def test_backward(diffusion, batch):
    loss = diffusion.loss(batch, None)
    loss['loss'].backward()
    assert True

def test_training_step(diffusion, batch):
    loss = diffusion.training_step(batch, None)
    assert loss > 0

def test_validation_step(diffusion, batch):
    loss = diffusion.validation_step(batch, None)
    assert loss > 0

    
    
