import torch
from typing import Dict, Optional
from agedi.diffusion.distributions import Distribution
from agedi.data import AtomsGraph
from agedi.utils import TruncatedNormal as TN


def _zero_com(x: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    """Subtract the per-graph center of mass from *x*.

    Projects *x* onto the zero-COM subspace: each graph's mean column is
    subtracted so that the COM of the returned tensor is exactly zero for
    every graph.  This is the translational-invariance projection from the
    EDM paper (Hoogeboom et al., NeurIPS 2022, arXiv:2203.17003).

    Parameters
    ----------
    x : torch.Tensor, shape (n_atoms, d)
    batch_idx : torch.Tensor, shape (n_atoms,)
        Graph membership index (``batch.batch``).

    Returns
    -------
    torch.Tensor, shape (n_atoms, d)
    """
    n_graphs = int(batch_idx.max().item()) + 1
    count = batch_idx.bincount(minlength=n_graphs).float().view(-1, 1)
    com = torch.zeros(n_graphs, x.shape[1], dtype=x.dtype, device=x.device)
    com.scatter_add_(0, batch_idx.unsqueeze(1).expand_as(x), x)
    com = com / count
    return x - com[batch_idx]

_CONFINEMENT_CLAMP_EPS = 1e-4


class StandardNormal(Distribution):
    """Standard Normal Distribution

    Parameters
    ----------
    scale : float, optional
        Standard deviation used when sampling.  Defaults to ``1.0``.
        Set this to the SDE's ``sigma_max`` (or ``sqrt(var(t=1))``) so that
        the prior matches the forward-process marginal at T=1, replacing the
        old ``0.8 * N**(1/3)`` heuristic which was arbitrary and broke for
        non-compact or heterogeneously-sized systems.
    """

    def __init__(self, scale: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scale = scale

    def get_hparams(self) -> dict:
        """Return hyperparameters for this distribution."""
        return {**super().get_hparams(), "scale": self.scale}

    def _setup(self, batch: AtomsGraph) -> None:
        """Prepare the distribution for sampling from *batch*.

        Sets ``self.shape`` to ``(n_atoms, *trailing)`` where ``n_atoms`` is
        read from ``batch.n_atoms`` and the trailing dimensions come from the
        existing attribute.  Using ``n_atoms`` rather than the attribute's
        leading dimension avoids a shape-mismatch when called during graph
        initialisation (via :meth:`~agedi.diffusion.noisers.Noiser.initialize_graph`),
        where the attribute tensor may still be empty even though ``n_atoms``
        has already been set.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data.
        """
        if self.key is not None:
            attr = batch[self.key]
            n_atoms = int(batch.n_atoms.sum().item())
            self.shape = torch.Size([n_atoms] + list(attr.shape[1:]))

    def _sample(self, shape: Optional[torch.Size] = None, **kwargs) -> torch.Tensor:
        """Sample from the standard normal distribution.

        Parameters
        ----------
        shape : torch.Size, optional
            Output shape.  Defaults to ``self.shape`` set during ``_setup``.

        Returns
        -------
        torch.Tensor
            Sampled tensor with std equal to ``self.scale``.
        """
        if shape is None:
            shape = self.shape
        return torch.normal(0.0, self.scale, size=shape)


class Normal(Distribution):
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


class ZeroComNormal(Normal):
    """Normal distribution whose noise increment has zero center of mass per graph.

    Drop-in replacement for :class:`Normal` for use with the
    :class:`~agedi.diffusion.noisers.Positions` noiser on gas-phase molecules
    and clusters.  After sampling ``x = N(mu, sigma)``, the COM of the noise
    increment ``(x - mu)`` is subtracted per graph so that the diffusion
    process operates in the translationally-invariant subspace.

    Reference: Hoogeboom et al., "Equivariant Diffusion for Molecule Generation
    in 3D", NeurIPS 2022. arXiv:2203.17003
    """

    def _setup(self, batch: AtomsGraph) -> None:
        self.batch_idx = batch.batch

    def _sample(self, mu: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        raw = super()._sample(mu, sigma, **kwargs)
        noise = _zero_com(raw - mu, self.batch_idx)
        return mu + noise


class ZeroComStandardNormal(StandardNormal):
    """Standard Normal distribution with zero center of mass per graph.

    Drop-in replacement for :class:`StandardNormal` intended as the *prior*
    distribution for the :class:`~agedi.diffusion.noisers.Positions` noiser.
    Sampled positions are centered at the origin for every graph.

    Parameters
    ----------
    scale : float, optional
        Standard deviation of the prior.  Should be set to
        ``sqrt(sde.var(t=1))`` (i.e. ``sigma_max`` for a VE-SDE) so that the
        prior matches the forward-process marginal at T=1.  Defaults to
        ``1.0``; :class:`~agedi.diffusion.noisers.Positions` sets this
        automatically from the SDE.

    Reference: Hoogeboom et al., "Equivariant Diffusion for Molecule Generation
    in 3D", NeurIPS 2022. arXiv:2203.17003
    """

    def _setup(self, batch: AtomsGraph) -> None:
        super()._setup(batch)
        if batch.batch is not None:
            self.batch_idx = batch.batch
        else:
            # Single un-batched graph: all atoms belong to graph 0.
            n_atoms = int(batch.n_atoms.sum().item())
            self.batch_idx = torch.zeros(n_atoms, dtype=torch.long)

    def _sample(self, shape: Optional[torch.Size] = None, **kwargs) -> torch.Tensor:
        x = super()._sample(shape=shape, **kwargs)
        return _zero_com(x, self.batch_idx)
