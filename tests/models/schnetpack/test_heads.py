import torch
import pytest

from agedi.models.schnetpack.heads import PositionsScore
from agedi.models.schnetpack.translator import SchNetPackTranslator
from agedi.data import Representation
    
def test_positions_score_init():
    head = PositionsScore()
    assert head is not None

def test_positions_score_forward(batch: "Batch"):
    d = 64
    head = PositionsScore(input_dim_scalar=d+2, input_dim_vector=d)
    translator = SchNetPackTranslator()

    N = batch.pos.shape[0]    
    rep = Representation(scalar=torch.rand((N, d+2, 1)),
                         vector=torch.rand((N, d, 3)))
    
    batch.representation = rep

    translated_batch = translator(batch)

    out = head(translated_batch)

    assert out.shape == (N, 3)
    

