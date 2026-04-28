import torch
from typing import Dict, Optional
from agedi.diffusion.distributions import Distribution, PriorDistribution, NoiseDistribution
from agedi.data import AtomsGraph
from agedi.utils import TruncatedNormal as TN

_CONFINEMENT_CLAMP_EPS = 1e-4


class StandardNormal(PriorDistribution):
    """Standard Normal Distribution"""

    def sample(self, batch: AtomsGraph, **kwargs) -> torch.Tensor:
        """Sample from the standard normal distribution.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data.  The shape is derived from
            ``batch[self.key]``.

        Returns
        -------
        torch.Tensor
            Sampled tensor.
        """
        shape = batch[self.key].shape
        std = 0.8 * shape[0] ** (1 / 3)
        return torch.normal(0.0, std, size=shape)


class Normal(NoiseDistribution):
    """Normal Distribution"""

    def sample(self, batch: AtomsGraph, sigma: torch.Tensor, mu: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Sample noise from the normal distribution.

        Returns the noise contribution ``w = σ · ε`` where ``ε ~ N(0, I)``.
        The caller is responsible for constructing ``x_t = mean + w``.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data (unused, present for interface consistency).
        sigma : torch.Tensor
            Standard deviation of the distribution.
        mu : torch.Tensor, optional
            Not used for the mean; passed only as a shape reference when
            *sigma* broadcasts (e.g. shape ``(N, 1)`` while the target has
            shape ``(N, D)``).

        Returns
        -------
        torch.Tensor
            Noise tensor ``σ · ε``.
        """
        ref = mu if mu is not None else sigma
        return sigma * torch.randn_like(ref)


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

    def sample(self, batch: AtomsGraph, mu: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample noise from the truncated normal distribution.

        Returns the noise contribution ``w = x_sampled − mu`` where
        ``x_sampled`` is drawn from a truncated normal within the confinement
        bounds.  The caller is responsible for constructing ``x_t = mu + w``.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data.  ``batch.confinement`` and ``batch.mask``
            are read directly.
        mu : torch.Tensor
            Mean of the distribution; used for truncation-bound clamping.
        sigma : torch.Tensor
            Standard deviation of the distribution.

        Returns
        -------
        torch.Tensor
            Noise tensor ``x_sampled − mu``.
        """
        batch_idx = batch.batch if batch.batch is not None else torch.zeros(mu.shape[0], dtype=torch.long, device=mu.device)
        confinement = batch.confinement[batch_idx]
        mask = batch.mask if batch.mask is not None else torch.zeros(mu.shape[0], dtype=torch.bool, device=mu.device)
        x = []
        for i in range(mu.shape[1]):
            if i == self.index:
                if mu[:, i].isnan().any():
                    raise ValueError(
                        "NaN mean (probably position) values.\n"
                        + "See troubleshooting in the documentation:\n"
                        + "https://agedi.readthedocs.io/en/latest/troubleshooting.html"
                    )

                z_lo = confinement[:, 0][~mask]
                z_hi = confinement[:, 1][~mask]
                mu_z = mu[:, i][~mask].clamp(
                    min=z_lo + _CONFINEMENT_CLAMP_EPS,
                    max=z_hi - _CONFINEMENT_CLAMP_EPS,
                )
                sampled = TN(
                    mu_z,
                    sigma[:, 0][~mask],
                    z_lo,
                    z_hi,
                ).sample()

                xi = torch.zeros_like(mu[:, i])
                xi[~mask] = sampled - mu[:, i][~mask]
                x.append(xi)
            else:
                noise_i = sigma[:, 0] * torch.randn_like(mu[:, i])
                x.append(noise_i)
        return torch.stack(x, dim=1)


class WrappedNormal(NoiseDistribution):
    """Wrapped Normal Distribution"""

    def __init__(self, N: int = 10, T: float = 1.0, **kwargs) -> None:
        """Initialize the distribution"""
        super().__init__(**kwargs)
        self.N = N
        self.T = T

    def sample(self, batch: AtomsGraph, sigma: torch.Tensor, mu: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Sample noise from the wrapped normal distribution.

        Returns the noise contribution ``w = σ · ε`` where ``ε ~ N(0, I)``.
        The caller is responsible for constructing ``x_t = mean + w`` and
        applying any periodic wrapping.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data (unused, present for interface consistency).
        sigma : torch.Tensor
            Standard deviation of the distribution.
        mu : torch.Tensor, optional
            Not used for the mean; passed only as a shape reference when
            *sigma* broadcasts.

        Returns
        -------
        torch.Tensor
            Noise tensor ``σ · ε``.
        """
        ref = mu if mu is not None else sigma
        return sigma * torch.randn_like(ref)

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
        
