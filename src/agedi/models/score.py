import torch

from typing import List

from torch_geometric.data import Batch
from agedi.models.conditionings import Conditioning, TimeConditioning
from agedi.models.translator import Translator
from agedi.data import Representation
from agedi.models.head import Head

class ScoreModel(torch.nn.Module):
    """
    Class that defines a the score model, which is a combination of a translator,
    a representation, a list of conditionings and a list of heads.

    Args:
        translator: Translator
            The translator that will be used to translate the input batch.
        representation: Representation
            The representation that will be used to represent the translated batch.
        conditionings: List[Conditioning]
            The list of conditionings that will be applied to the representation.
        heads: List[Head]
            The list of heads that will be used to compute scores.
    
    """
    def __init__(
        self,
        translator: Translator,
        representation: Representation,
        conditionings: List[Conditioning]=[
            TimeConditioning(),
        ],
        heads: List[Head]=[],
        **kwargs
    ):
        """
        Constructor for the ScoreModel class.
        
        """
        super().__init__(**kwargs)
        self.translator = translator
        self.representation = representation
        self.conditionings = conditionings
        self.heads = heads

    def forward(self, batch: Batch) -> Batch:
        """
        Forward pass of the model.

        Args: 
            batch: Batch
                The input batch that will be used to compute the scores.

        Returns:
            Batch
                The output batch containing the scores.
    
        """
        translated_batch = self.translator(batch)
        rep = self.representation(translated_batch)
        
        batch = self.translator.add_representation(batch, rep)
        for conditioning in self.conditionings:
            batch = conditioning(batch)

        translated_batch = self.translator(batch)
        scores = {}
        for head in self.heads:
            scores[head.key] = head(translated_batch)

        batch = self.translator.add_scores(batch, scores)

        return batch
