import pytest
import torch

from agedi.models.schnetpack.translator import SchNetPackTranslator
from agedi.data import Representation

def test_init():
    translator = SchNetPackTranslator()
    assert translator is not None

    
def test_call(batch: "Batch"):
    translator = SchNetPackTranslator()
    result = translator(batch)
    assert isinstance(result, dict)

def test_call_error():
    translator = SchNetPackTranslator()
    with pytest.raises(ValueError):
        translator({})


def test_call_with_input_module(batch: "Batch"):
    translator = SchNetPackTranslator(input_modules=[
        lambda x: x,
        lambda x: x,
    ])
    result = translator(batch)
    assert isinstance(result, dict)

def test_call_with_representation(batch: "Batch"):
    N = batch.pos.shape[0]
    d = 64
    translator = SchNetPackTranslator()
    rep = Representation(scalar=torch.rand((N, d, 1)),
                         vector=torch.rand((N, d, 3)))
    batch.representation = rep
    result = translator(batch)
    
    assert result["scalar_representation"].shape == (N, d)
    assert result["vector_representation"].shape == (N, 3, d)
    
def test_add_representation(batch: "Batch"):
    N = batch.pos.shape[0]
    d = 64
    translator = SchNetPackTranslator()

    results = {}
    results["scalar_representation"] = torch.rand((N, d))
    results["vector_representation"] = torch.rand((N, 3, d))

    translator.add_representation(batch, results)
    assert isinstance(batch.representation, Representation)
    rep = batch.representation
    assert rep.scalar.shape == (N, d, 1)
    assert rep.vector.shape == (N, d, 3)
    
    
def test_add_scores(batch: "Batch"):
    translator = SchNetPackTranslator()

    scores = {
        "positions": torch.rand(batch.pos.shape),
        "types": torch.randint(1, 92, batch.x.shape),
    }

    translator.add_scores(batch, scores)

    assert batch.positions_score.shape == scores["positions"].shape
    assert batch.types_score.shape == scores["types"].shape

def test_add_properties(batch: "Batch"):
    batch.energy = torch.rand((batch.num_graphs, 1))
    batch.forces = torch.rand(batch.pos.shape)
    
    translator = SchNetPackTranslator()

    out = translator(batch)
    
    assert out.get("energy") is not None
    assert out.get("forces") is not None

