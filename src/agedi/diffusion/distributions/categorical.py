import torch
from agedi.diffusion.distributions import Distribution, NoiseDistribution
from agedi.data import AtomsGraph


class Categorical(NoiseDistribution):
    """Categorical Distribution

    Implements hard sampling using the Gumbel-Max trick.

    """

    def sample(self, batch: AtomsGraph, probs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample from the categorical distribution.

        Uses the Gumbel-Max trick for hard sampling.  ``probs`` defines the
        likelihood of the masked (absorbing, index 0) value.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data (unused, present for interface consistency).
        probs : torch.Tensor
            The probabilities of each category.

        Returns
        -------
        torch.Tensor
            Sampled tensor.
        """
        gumbel_norm = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
        return (probs / gumbel_norm).argmax(dim=-1)
