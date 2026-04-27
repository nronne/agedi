import torch
from typing import Dict, Optional
from agedi.diffusion.distributions import Distribution, PriorDistribution, NoiseDistribution
from agedi.data import AtomsGraph
from agedi.utils import TruncatedNormal as TN

_CONFINEMENT_CLAMP_EPS = 1e-4


class StandardNormal(PriorDistribution):
    """Standard Normal Distribution"""

    def _setup(self, batch: AtomsGraph) -> None:
        """Prepare the distribution for sampling from *batch*.

        Sets ``self.shape`` to the shape of the target attribute in the batch.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data.
        """
        if self.key is not None:
            self.shape = batch[self.key].shape

    def _sample(self, shape: Optional[torch.Size] = None, **kwargs) -> torch.Tensor:
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
        if shape is None:
            shape = self.shape
        std = 0.8 * shape[0]**(1/3)
        return torch.normal(0.0, std, size=shape)


class Normal(NoiseDistribution):
    """Normal Distribution"""

    def _sample(self, mu: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
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


class TruncatedNormal(NoiseDistribution):
    """Truncated Normal Distribution

    Parameters
    ----------
    index : int
        The index of the property to truncate

    """

    def __init__(self, index: int = 2, **kwargs) -> None:
        """Initialize the distribution"""
        super().__init__(**kwargs)
        self.index = index

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this distribution."""
        return {**super().get_hparams(), "index": self.index}

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
        self.mask = batch.mask

    def _sample(self, mu: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
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
                if mu[:, i].isnan().any():
                    raise ValueError(
                        "NaN mean (probably position) values.\n"
                        + "See troubleshooting in the documentation:\n"
                        + "https://agedi.readthedocs.io/en/latest/troubleshooting.html"
                    )

                z_lo = self.confinement[:, 0][~self.mask]
                z_hi = self.confinement[:, 1][~self.mask]
                mu_z = mu[:, i][~self.mask].clamp(
                    min=z_lo + _CONFINEMENT_CLAMP_EPS,
                    max=z_hi - _CONFINEMENT_CLAMP_EPS,
                )
                sampled = TN(
                    mu_z,
                    sigma[:, 0][~self.mask],
                    z_lo,
                    z_hi,
                ).sample()

                xi = torch.zeros_like(mu[:, i])
                xi[~self.mask] = sampled
                x.append(xi)
            else:
                x.append(torch.normal(mu[:, i], sigma[:, 0]))
        return torch.stack(x, dim=1)


class WrappedNormal(NoiseDistribution):
    """Wrapped Normal Distribution"""

    def __init__(self, N: int = 10, T: float = 1.0, **kwargs) -> None:
        """Initialize the distribution"""
        super().__init__(**kwargs)
        self.N = N
        self.T = T

    def _sample(self, mu: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample from the wrapped normal distribution

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
        return mu + sigma * torch.randn_like(mu)

    def p(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate the probability density function of the wrapped normal distribution

        Parameters
        ----------
        x : torch.Tensor
            Sampled tensor
        mu : torch.Tensor
            Mean of the distribution
        sigma : torch.Tensor
            Standard deviation of the distribution

        Returns
        -------
        torch.Tensor
            Probability density function

        """
        p_ = 0
        for i in range(-self.N, self.N + 1):
            p_ += torch.exp(-((x + self.T * i) ** 2) / 2 / sigma**2)
        return p_
        
        
    def d_log_p(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate the gradient of the log probability density function of the wrapped normal distribution

        Parameters
        ----------
        x : torch.Tensor
            Sampled tensor
        mu : torch.Tensor
            Mean of the distribution
        sigma : torch.Tensor
            Standard deviation of the distribution

        Returns
        -------
        torch.Tensor
            Gradient of the log probability density function

        """
        p_ = 0
        for i in range(-self.N, self.N + 1):
            p_ -= (x + self.T * i) / sigma**2 * torch.exp(-((x + self.T * i) ** 2) / 2 / sigma**2)
        return p_ / self.p(x, sigma)


    def sigma_norm(self, sigma: torch.Tensor, sn: int=10000) -> torch.Tensor:
        """Calculate the normalization constant of the wrapped normal distribution

        Parameters
        ----------
        sigma : torch.Tensor
            Standard deviation of the distribution

        sn : int
            Number of samples to use for the calculation

        Returns
        -------
        torch.Tensor
            Normalization constant

        """
        sigmas = sigma.repeat(1, sn)
        x_sample = sigma * torch.randn_like(sigmas)
        x_sample = x_sample % self.T
        normal_ = self.d_log_p(x_sample, sigmas, T=self.T)
        return (normal_**2).mean(dim=1)[..., None]
        
