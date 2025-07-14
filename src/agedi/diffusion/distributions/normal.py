import torch
from agedi.diffusion.distributions import Distribution
from agedi.data import AtomsGraph
from agedi.utils import TruncatedNormal as TN


class StandardNormal(Distribution):
    """Standard Normal Distribution"""

    def _setup(self, batch: AtomsGraph) -> None:
        if self.key is not None:
            self.shape = getattr(batch, self.key).shape

    def _sample(self, shape=None, **kwargs) -> torch.Tensor:
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


class Normal(Distribution):
    """Normal Distribution"""

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
        """Initialize the distribution"""
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
        self.mask = batch.mask

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
                if mu[:, i].isnan().any():
                    raise ValueError(
                        "NaN mean (probably position) values.\n"
                        + "See troubleshooting in the documentation:\n"
                        + "https://agedi.readthedocs.io/en/latest/troubleshooting.html"
                    )

                sampled = TN(
                    mu[:, i][~self.mask],
                    sigma[:, 0][~self.mask],
                    self.confinement[:, 0][~self.mask],
                    self.confinement[:, 1][~self.mask],
                ).sample()

                xi = torch.zeros_like(mu[:, i])
                xi[~self.mask] = sampled
                x.append(xi)
            else:
                x.append(torch.normal(mu[:, i], sigma[:, 0]))
        return torch.stack(x, dim=1)


