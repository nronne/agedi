import torch
from typing import Dict, Type
from agedi.diffusion.distributions import Distribution, PriorDistribution
from agedi.data import AtomsGraph


class Constant(PriorDistribution):
    """Constant Integer Distribution"""

    def __init__(
        self,
        value: float = 0,
        key: str = "x",
        dtype: Type = torch.int64,
        **kwargs,
    ) -> None:
        """Initialize the distribution

        Parameters
        ----------
        value : float
            The value of the constant
        key : str
            The key to access the data in the batch

        """
        super().__init__(key=key, **kwargs)
        self.value = value
        self.dtype = dtype

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this distribution."""
        return {**super().get_hparams(), "value": self.value}

    def sample(self, batch: AtomsGraph, **kwargs) -> torch.Tensor:
        """Sample a constant tensor.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data.  The shape is derived from
            ``batch.n_atoms``.

        Returns
        -------
        torch.Tensor
            Constant tensor of shape ``(n_atoms,)``.
        """
        shape = (batch.n_atoms.sum().item(),)
        return torch.ones(shape, dtype=self.dtype) * self.value
