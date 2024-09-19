from abc import ABC, abstractmethod
from typing import Callable

import torch

from agedi.utils import TruncatedNormal as TN

# from agedi.data import AtomsGraph


"""
Noise distributions
"""


class Distribution(ABC):
    """
    Base Class for noise distributions

    attrs:
        key: str
            The key to access property from batch
    """

    def __init__(self, **kwargs):
        """
        Initialize the distribution

        """
        self.key = None

    @abstractmethod
    def sample(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Sample from the distribution and return tensor of shape self.key

        args:
            mu: torch.Tensor
                The mean of the distribution

            sigma: torch.Tensor
                The standard deviation of the distribution
        """
        pass

    def _setup(self, batch) -> None:
        """
        Prepare the distribution for sampling of the batch

        args:
            batch: AtomsGraph

        returns:
            None
        """
        pass

    def get_callable(self, batch) -> Callable:
        """
        Return a callable function that samples from the distribution

        args:
            batch: AtomsGraph

        returns:
            Callable

        """
        self._setup(batch)

        def callable(mu, sigma):
            return self.sample(mu, sigma)

        return callable


class StandardNormal(Distribution):
    """
    Standard Normal Distribution
    """

    def sample(self, mu, sigma) -> torch.Tensor:
        """
        Sample from the standard normal distribution

        args:
            mu: torch.Tensor
                The mean of the distribution

            sigma: torch.Tensor
                The standard deviation of the distribution

        returns:
            torch.Tensor
        """
        shape = mu.shape
        return torch.normal(0.0, 1.0, size=shape)


class Normal(Distribution):
    """
    Normal Distribution
    """

    def sample(self, mu, sigma) -> torch.Tensor:
        """
        Sample from the normal distribution

        args:
            mu: torch.Tensor
                The mean of the distribution

            sigma: torch.Tensor
                The standard deviation of the distribution

        returns:
            torch.Tensor
        """
        return torch.normal(mu, sigma)


class TruncatedNormal(Distribution):
    """
    Truncated Normal Distribution
    """

    def __init__(self, index: int = 2, **kwargs) -> None:
        """
        Initialize the distribution

        args:
            index: int
                The index to truncatethe batch
        """
        super().__init__(**kwargs)
        self.index = index

    def _setup(self, batch) -> None:
        """
        Prepare the distribution for sampling of the batch

        args:
            batch: AtomsGraph

        returns:
            None

        """

        self.confinement = batch.confinement[batch.batch]

    def sample(self, mu, sigma) -> torch.Tensor:
        """
        Sample from the truncated normal distribution

        args:
            mu: torch.Tensor
                The mean of the distribution

            sigma: torch.Tensor
                The standard deviation of the distribution

        returns:
            torch.Tensor
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


"""
Prior distributions
"""


class Prior(ABC):
    """
    Base Class for prior distributions

    attrs:
        key: str
            The key to access property from batch
    """

    def __init__(self, **kwargs):
        self.key = None

    @abstractmethod
    def sample(self, batch: "AtomsGraph") -> torch.Tensor:
        """
        Sample from the distribution and return tensor of shape self.key

        args:
            batch: AtomsGraph

        returns:
            torch.Tensor
        """
        pass


class Uniform(Prior):
    """
    Uniform Prior Distribution

    attrs:
        low: float
            The lower bound of the distribution
        high: float
            The upper bound of the distribution
    """

    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        """
        Initialize the distribution

        args:
            low: float
                The lower bound of the distribution
            high: float
                The upper bound of the distribution

        """

        self.low = low
        self.high = high

    def sample(self, batch: "AtomsGraph") -> torch.Tensor:
        """
        Sample from the uniform distribution

        args:
            batch: AtomsGraph

        returns:
            torch.Tensor

        """
        shape = batch[self.key].shape
        return torch.rand(shape) * (self.high - self.low) + self.low


class UniformCell(Uniform):
    """
    Uniform Prior Distribution for cell parameters
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the distribution
        """
        super().__init__()

    def sample(self, batch: "AtomsGraph") -> torch.Tensor:
        """
        Sample from the uniform distribution

        args:
            batch: AtomsGraph

        returns:
            torch.Tensor
        """
        f = super().sample(batch)  # (n_atoms, 3)

        cell = (
            batch.cell if batch.confinement_cell is None else batch.confinement_cell
        )  # (n_graphs * 3, 3)
        cells = cell.view(-1, 3, 3)  # (n_graphs, 3, 3)
        corner = (
            torch.zeros((cell.shape[0], 3))
            if batch.confinement_corner is None
            else batch.confinement_corner
        )
        if batch.batch is not None:
            cells = cells[batch.batch]  # (n_atoms, 3, 3)
            corner = corner[batch.batch]  # (n_atoms, 3)
        else:
            cells = cells.unsqueeze(0)
            corner = corner.unsqueeze(0)

        r = torch.matmul(cells, f) + corner  # (n_atoms, 3)
        return r
