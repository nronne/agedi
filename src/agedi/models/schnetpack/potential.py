import torch
from typing import Dict
import schnetpack as spk

class Potential(spk.model.NeuralNetworkPotential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # initialize derivatives for response properties
        inputs = self.initialize_derivatives(inputs)

        if inputs.get("scalar_representation") is None:
            for m in self.input_modules:
                inputs = m(inputs)

            inputs = self.representation(inputs)

            
        for m in self.output_modules:
            inputs = m(inputs)

        # apply postprocessing (if enabled)
        inputs = self.postprocess(inputs)
        results = self.extract_outputs(inputs)

        return results    
