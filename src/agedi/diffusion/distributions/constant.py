import torch
from typing import Dict, Type, Optional
from agedi.diffusion.distributions import Distribution
from agedi.data import AtomsGraph


class Constant(Distribution):
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

    def _setup(self, batch: AtomsGraph) -> None:
        """Prepare the distribution for sampling from *batch*.

        Sets ``self.shape`` based on the total number of atoms in the batch.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data.
        """
        if self.key is not None:
            self.shape = (batch.n_atoms.sum().item(),)

    def _sample(self, shape: Optional[torch.Size] = None) -> torch.Tensor:
        """
        Sample from the integer distribution

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the distribution
        sigma : torch.Tensor
            Standard deviation of the distribution

        Returns
        -------
        torch.Tensor
            Sampled tensor

        """
        shape = shape if shape is not None else self.shape
        return torch.ones(shape, dtype=self.dtype) * self.value
