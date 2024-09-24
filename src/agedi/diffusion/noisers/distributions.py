from abc import ABC, abstractmethod
from typing import Callable

import torch

from agedi.data import AtomsGraph
from agedi.utils import TruncatedNormal as TN



class Distribution(ABC):
    """Base Class for noise distributions

    Parameters
    ----------
    key : str
        Key to identify the property from the batch

    Returns
    -------
    Distribution
    
    """

    def __init__(self, **kwargs):
        """Initialize the distribution

        """
        self.key = None

    @abstractmethod
    def _sample(self, mu: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample distribution
        
        Sample from the distribution and return tensor of shape self.key

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
        pass

    def _setup(self, batch: AtomsGraph) -> None:
        """Prepare distribution
        
        Prepare the distribution for sampling of the batch

        Parameters
        ----------
        batch : AtomsGraph
            Batch of data

        Returns
        -------
        None
        
        """
        pass

    def get_callable(self, batch: AtomsGraph) -> Callable:
        """Get callable function
        
        Return a callable function that samples from the distribution

        Parameters
        ----------
        batch : AtomsGraph
            Batch of data

        Returns
        -------
        Callable
            Callable function that samples from the distribution

        """
        self._setup(batch)

        def callable(mu, sigma, **kwargs):
            return self._sample(mu, sigma, **kwargs)

        return callable

class StandardNormal(Distribution):
    """Standard Normal Distribution

    """

    def _sample(self, mu, sigma, **kwargs) -> torch.Tensor:
        """Sample from the standard normal distribution

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
        shape = mu.shape
        return torch.normal(0.0, 1.0, size=shape)

class Normal(Distribution):
    """Normal Distribution
    
    """

    def _sample(self, mu, sigma, **kwargs) -> torch.Tensor:
        """Sample from the normal distribution

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
        return torch.normal(mu, sigma)

class TruncatedNormal(Distribution):
    """Truncated Normal Distribution

    Parameters
    ----------
    index : int
        The index of the property to truncate
    
    """

    def __init__(self, index: int = 2, **kwargs) -> None:
        """Initialize the distribution

        """
        super().__init__(**kwargs)
        self.index = index

    def _setup(self, batch: AtomsGraph) -> None:
        """Setup the distribution
        
        Prepare the distribution for sampling of the batch

        Parameters
        ----------
        batch : AtomsGraph
            Batch of data

        Returns
        -------
        None

        """

        self.confinement = batch.confinement[batch.batch]

    def _sample(self, mu, sigma, **kwargs) -> torch.Tensor:
        """Sample from the truncated normal distribution

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
        x = []
        for i in range(mu.shape[1]):
            if i == self.index:
                x.append(
                    TN(
                        mu[:, i],
                        sigma[:, i],
                        self.confinement[:, 0],
                        self.confinement[:, 1],
                    ).sample()
                )
            else:
                x.append(torch.normal(mu[:, i], sigma[:, i]))
        return torch.stack(x, dim=1)

class WrappedNormal(Distribution):
    pass

class Uniform(Distribution):
    """Uniform Distribution

    Parameters
    ----------
    low : float
        The lower bound of the distribution
    high : float
        The upper bound of the distribution
    
    """

    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        """Initialize the distribution

        """
        self.low = low
        self.high = high

    def _sample(self, mu, sigma) -> torch.Tensor:
        """
        Sample from the uniform distribution

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
        shape = mu.shape
        return torch.rand(shape) * (self.high - self.low) + self.low

class UniformCell(Uniform):
    """
    Uniform Prior Distribution for cell parameters
    """

    def _setup(self, batch: AtomsGraph) -> None:
        """
        Prepare the distribution for sampling of the batch

        Parameters
        ----------
        batch : AtomsGraph
            Batch of data

        Returns
        -------
        None

        """
        self.cell = batch.cell.view(-1, 3, 3)[batch.batch]
        self.corner = torch.zeros(self.cell.shape[0], 3)
        

        
    def _sample(self, mu, sigma) -> torch.Tensor:
        """Sample from the uniform distribution

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
        f = super()._sample(mu, sigma)  # (n_atoms, 3)

        r = torch.matmul(self.cell, f.unsqueeze(2)).squeeze(2) + self.corner  # (n_atoms, 3)
        return r



