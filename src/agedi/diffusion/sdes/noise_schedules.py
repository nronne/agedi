from abc import ABC, abstractmethod
from typing import Dict
import math
import torch


class NoiseSchedule(ABC):
    """Abstract base class for diffusion noise schedules.

    A noise schedule defines a function ``f(t)`` that controls the noise level
    during the forward diffusion process, where ``t ∈ [0, 1]``.
    """

    def __init__(self, min: float, max: float) -> None:
        """Initialize the noise schedule.

        Parameters
        ----------
        min : float
            Noise level at ``t = 0``.
        max : float
            Noise level at ``t = 1``.
        """
        self.min = min
        self.max = max

    def get_hparams(self) -> Dict:
        """Return hyperparameters sufficient to reconstruct this noise schedule.

        Returns a dictionary with a ``_target_`` key plus ``min`` and ``max``.
        Subclasses can call ``super().get_hparams()`` and add their own params.

        Returns
        -------
        dict
            Hyperparameter dictionary.
        """
        return {
            "_target_": f"{type(self).__module__}.{type(self).__qualname__}",
            "min": self.min,
            "max": self.max,
        }
    
    @abstractmethod
    def f(self, t: torch.Tensor) -> torch.Tensor:
        """Returns the noise schedule value at time t."""
        pass

    @abstractmethod
    def fprime(self, t: torch.Tensor) -> torch.Tensor:
        """Returns the derivative of the noise schedule at time t."""
        pass

    @abstractmethod
    def fint(self, t: torch.Tensor) -> torch.Tensor:
        """Return the integral of the noise schedule at time t"""
        pass

    def df2dt(self, t: torch.Tensor) -> torch.Tensor:
        """Return the time derivative of f(t)² at time *t*.

        Computed as ``2 * f(t) * f'(t)``.
        """
        return 2 * self.f(t) * self.fprime(t)
    

class Linear(NoiseSchedule):
    """Linear noise schedule: ``f(t) = min + (max - min) * t``."""

    def f(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the noise schedule at time *t*."""
        return self.min + (self.max - self.min) * t

    def fprime(self, t: torch.Tensor) -> torch.Tensor:
        """Return the derivative of the noise schedule at time *t*."""
        return self.max - self.min

    def fint(self, t: torch.Tensor) -> torch.Tensor:
        """Return the integral of the noise schedule from 0 to *t*."""
        return self.min * t + 0.5 * (self.max - self.min) * t **2

class Exponential(NoiseSchedule):
    """Exponential noise schedule: ``f(t) = min * (max/min)^t``."""

    def f(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the noise schedule at time *t*."""
        return self.min * (self.max / self.min) ** t

    def fprime(self, t: torch.Tensor) -> torch.Tensor:
        """Return the derivative of the noise schedule at time *t*."""
        return self.min * (self.max / self.min) ** t * math.log(self.max / self.min)

    def fint(self, t: torch.Tensor) -> torch.Tensor:
        """Return the integral of the noise schedule from 0 to *t*."""
        return self.min * ((self.max / self.min) ** t - 1) / math.log(self.max / self.min)


class Cosine(NoiseSchedule):
    """Cosine noise schedule: ``f(t) = min + (max - min) * (1 - cos(πt)) / 2``."""

    def f(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the noise schedule at time *t*."""
        return self.min + (self.max - self.min) * (1 - torch.cos(t * math.pi)) / 2
    
    def fprime(self, t: torch.Tensor) -> torch.Tensor:
        """Return the derivative of the noise schedule at time *t*."""
        return (self.max - self.min) * math.pi * torch.sin(t * math.pi) / 2
    
    def fint(self, t: torch.Tensor) -> torch.Tensor:
        """Return the integral of the noise schedule from 0 to *t*."""
        return (self.max - self.min) * (t / 2 - torch.sin(2 * t * math.pi) / (4 * math.pi))

        


    


