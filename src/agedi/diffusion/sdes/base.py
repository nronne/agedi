from abc import ABC, abstractmethod
from typing import Callable, Dict
from .noise_schedules import NoiseSchedule, Linear
import torch


class SDE(ABC):
    """SDE base class"""
    def __init__(self, noise_schedule: NoiseSchedule=Linear):
        """Initializes the SDE."""
        super().__init__()
        self.noise_schedule_cls = noise_schedule

    def get_hparams(self) -> Dict:
        """Return hyperparameters sufficient to reconstruct this SDE.

        Returns a dictionary with a ``_target_`` key (the fully-qualified class
        name).  Subclasses should call ``super().get_hparams()`` and merge in
        their own constructor parameters.

        Returns
        -------
        dict
            Hyperparameter dictionary.
        """
        return {"_target_": f"{type(self).__module__}.{type(self).__qualname__}"}

    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift term of the SDE.

        Must be implemented by subclass.

        Defines the drift term of the SDE:
        .. math::
            f(x, t) = ...

        Parameters
        ----------
        x: torch.Tensor
            The positions of the atoms.
        t: torch.Tensor
            The time at which to calculate the drift term.

        Returns
        -------
        drift: torch.Tensor
            The drift term of the SDE.

        """
        pass

    @abstractmethod
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion term of the SDE.

        Must be implemented by subclass.

        Defines the diffusion term of the SDE:
        .. math::
            g(t) = ...

        Parameters
        ----------
        t: torch.Tensor
            The time at which to calculate the diffusion term.

        Returns
        -------
        diffusion: torch.Tensor
            The diffusion term of the SDE.

        """
        pass

    @abstractmethod
    def mean(self, t: torch.Tensor) -> torch.Tensor:
        """Mean of the SDE.

        Must be implemented by subclass.

        Calculates the mean of transition kernel at time t:
        .. math::
            \mu_t = ...

        Parameters
        ----------
        t: torch.Tensor
            The time at which to calculate the mean.

        Returns
        -------
        mean: torch.Tensor
            The mean of the diffusion process.

        """
        pass

    @abstractmethod
    def var(self, t: torch.Tensor) -> torch.Tensor:
        """Variance of the SDE.

        Must be implemented by subclass.

        Calculates the variance of transition kernel at time t:
        .. math::
            \sigma_t^2 = ...

        Parameters
        ----------
        t: torch.Tensor
            The time at which to calculate the variance.

        Returns
        -------
        var: torch.Tensor
            The variance of the diffusion process.

        """
        pass

    def transition_kernel(
        self, x: torch.Tensor, t: torch.Tensor, w: Callable
    ) -> torch.Tensor:
        """Transition kernel of the SDE.

        Calculates the transition kernel of the diffusion process:
        .. math::

        p(\mathbf{x}_t | \mathbf{x}_0) = \mu_t \mathbf{x} + \sigma_t \mathbf{w},
                with :math:`\mathbf{w} \sim N(0,1)`.

        Parameters
        ----------
        x: torch.Tensor
            The positions of the atoms.
        w: torch.Tensor
            The noise term.
        t: torch.Tensor
            The time at which to calculate the transition kernel.

        Returns
        -------
        transition_kernel: torch.Tensor
            The transition kernel of the diffusion process.

        """
        mean = self.mean(t) * x
        sigma = torch.sqrt(self.var(t))
        x_t = w(mean, sigma)  # mean*x + sigma*w
        return x_t

    def noise(
        self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Noise term of the SDE.

        Calculates the noise term of the SDE:
        .. math::
        \mathbf{w} = \frac{\mathbf{x}_t - \mu_t \mathbf{x}_0}{\sigma_t}

        Parameters
        ----------
        x0: torch.Tensor
            x at time 0.
        xt: torch.Tensor
            x at time t.
        t: torch.Tensor
            The time at which to calculate the noise term.

        Returns
        -------
        noise: torch.Tensor
            The noise term of the diffusion process.

        """
        return (xt - self.mean(t) * x0) / torch.sqrt(self.var(t))
