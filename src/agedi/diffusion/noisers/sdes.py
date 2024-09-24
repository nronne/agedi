from abc import ABC, abstractmethod
from typing import Callable
import torch

class SDE(ABC):
    """SDE base class

    """
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

    def transition_kernel(self, x: torch.Tensor, t: torch.Tensor, w: Callable) -> torch.Tensor:
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
        std = torch.sqrt(self.var(t))
        x_t = w(mean, std) # mean*x + var*w
        return x_t
    
    def noise(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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

    
class VP(SDE):
    """Implements variance-preserving (VP) SDE.

    Parameters
    ----------
    beta_min: float
        The minimum value of the beta parameter.
    beta_max: float
        The maximum value of the beta parameter.

    Returns
    -------
    VP

    """
    def __init__(self, beta_min:float=1e-2, beta_max:float=3):
        """Initializes the VP SDE.

        """
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Implement VP drift term.
        
        Defines the drift term of the SDE: f(x, t).

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
        return -0.5 * self.beta(t) * x        

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Implement VP diffusion term.
        
        Defines the diffusion term of the SDE: g(t).

        Parameters
        ----------
        t: torch.Tensor
            The time at which to calculate the diffusion term.

        Returns
        -------
        diffusion: torch.Tensor
            The diffusion term of the SDE.
        
        """
        return torch.sqrt(self.beta(t))

    def mean(self, t: torch.Tensor) -> torch.Tensor:
        """Implement VP mean term.
        
        Calculates the mean of transition kernel at time t: mu(t).

        Parameters
        ----------
        t: torch.Tensor
            The time at which to calculate the mean.

        Returns
        -------
        mean: torch.Tensor
            The mean of the diffusion process.
        
        """
        return torch.exp(-0.5*self.alpha(t))

    def var(self, t: torch.Tensor) -> torch.Tensor:
        """Implement VP variance term.
        
        Calculates the variance of transition kernel at time t: sigma^2(t).

        Parameters
        ----------
        t: torch.Tensor
            The time at which to calculate the variance.

        Returns
        -------
        var: torch.Tensor
            The variance of the diffusion process.
        
        """
        return 1 - torch.exp(-self.alpha(t))

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """VP Beta function
        
        Calculates the value of beta at time t.

        Parameters
        ----------
        t: torch.Tensor
            The time at which to calculate beta.

        Returns
        -------
        beta: torch.Tensor
            The value of beta at time t.

        """

        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """VP Alpha function
        
        Calculates the value of alpha at time t with
        .. math::
        \alpha(t) = int_{0}^{t} beta(s) ds.

        Parameters
        ----------
        t: torch.Tensor
            The time at which to calculate alpha.

        Returns
        -------
        alpha: torch.Tensor
            The value of alpha at time t.

        """
        return t * self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)


