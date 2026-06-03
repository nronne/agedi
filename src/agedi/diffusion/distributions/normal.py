import torch
from typing import Dict, Optional
from agedi.diffusion.distributions import Distribution
from agedi.data import AtomsGraph
from agedi.utils import TruncatedNormal as TN

_CONFINEMENT_CLAMP_EPS = 1e-4


class StandardNormal(Distribution):
    """Standard Normal Distribution

    When used as a prior for a :class:`~agedi.diffusion.noisers.Positions`
    noiser the standard deviation is scaled by the cube root of the number of
    atoms **in each individual graph**, so that larger molecules are spread
    over a proportionally larger region.  This avoids atoms being initialised
    too close together when sampling clusters or gas-phase molecules.
    """

    def _setup(self, batch: AtomsGraph) -> None:
        """Prepare the distribution for sampling from *batch*.

        Sets ``self.shape`` to ``(n_atoms, *trailing)`` where ``n_atoms`` is
        read from ``batch.n_atoms`` and the trailing dimensions come from the
        existing attribute.  Using ``n_atoms`` rather than the attribute's
        leading dimension avoids a shape-mismatch when called during graph
        initialisation (via :meth:`~agedi.diffusion.noisers.Noiser.initialize_graph`),
        where the attribute tensor may still be empty even though ``n_atoms``
        has already been set.

        Also computes a per-atom standard deviation vector where each atom's
        std is ``0.8 * N_i**(1/3)`` with ``N_i`` the number of atoms in that
        atom's graph.  This ensures that the prior spread scales with the size
        of each individual molecule rather than the total batch size.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data.
        """
        if self.key is not None:
            attr = batch[self.key]
            n_atoms = int(batch.n_atoms.sum().item())
            self.shape = torch.Size([n_atoms] + list(attr.shape[1:]))
        # Build a per-atom n_atoms vector without relying on batch.batch (which
        # may be None for single un-batched graphs).  repeat_interleave
        # expands each graph's atom count into that many repeated entries.
        n_atoms_per_graph = batch.n_atoms.view(-1).long()
        per_atom_n = torch.repeat_interleave(
            n_atoms_per_graph.float(), n_atoms_per_graph
        )  # shape: (total_atoms,)
        self._per_atom_std = 0.8 * per_atom_n ** (1 / 3)

    def _sample(self, shape: Optional[torch.Size] = None, **kwargs) -> torch.Tensor:
        """Sample from the standard normal distribution.

        When called after :meth:`_setup` (i.e. via
        :meth:`~agedi.diffusion.distributions.Distribution.get_callable`), each
        atom is sampled with a standard deviation proportional to the cube root
        of the number of atoms in its graph.  When called directly with an
        explicit *shape* (e.g. in unit tests), a single global std equal to
        ``0.8 * shape[0]**(1/3)`` is used as a fallback.

        Parameters
        ----------
        shape : torch.Size, optional
            Output shape.  Defaults to the shape set by :meth:`_setup`.

        Returns
        -------
        torch.Tensor
            Sampled tensor.
        """
        if shape is None:
            shape = self.shape
            if hasattr(self, '_per_atom_std'):
                # Expand (n_atoms,) → (n_atoms, *trailing) for broadcast
                std = self._per_atom_std.view(
                    shape[0], *([1] * (len(shape) - 1))
                ).expand(shape)
                return torch.normal(
                    torch.zeros(shape, device=self._per_atom_std.device), std
                )
        std = 0.8 * shape[0] ** (1 / 3)
        return torch.normal(0.0, std, size=shape)


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


