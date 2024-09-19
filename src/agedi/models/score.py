import torch

from agedi.models.conditionings import TimeConditioning


class ScoreModel(torch.nn.Module):
    def __init__(
        self,
        translator,
        representation,
        conditionings=[
            TimeConditioning(),
        ],
        input_modules=[],
        heads=[],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.translator = translator
        self.representation = representation
        self.conditionings = conditionings
        self.input_modules = input_modules
        self.heads = heads

    def forward(self, batch):
        translated_batch = self.translator(batch)
        for module in self.input_modules:
            translated_batch = module(translated_batch)
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
