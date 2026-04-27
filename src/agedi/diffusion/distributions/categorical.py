import torch
from agedi.diffusion.distributions import Distribution, NoiseDistribution

class Categorical(NoiseDistribution):
    """Categorical Distribution

    Implements hard sampling using the Gumbel-Max trick.

    """
    def _sample(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Sample from the categorical distribution where
        probabilites define the likelihood of mu value
        to be set to the masked, 0, value

        Parameters
        ----------
        probs : torch.Tensor
            The probabilities of each category
        
        Returns
        -------
        torch.Tensor
            Sampled tensor

        """
        gumbel_norm = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
        return (probs / gumbel_norm).argmax(dim=-1)

        
