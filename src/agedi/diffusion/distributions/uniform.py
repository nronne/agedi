import torch
from typing import Dict, Optional
from agedi.diffusion.distributions import Distribution, PriorDistribution
from agedi.data import AtomsGraph


class Uniform(PriorDistribution):
    """Uniform Distribution

    Parameters
    ----------
    low : float
        The lower bound of the distribution
    high : float
        The upper bound of the distribution

    """

    def __init__(
        self, low: float = 0.0, high: float = 1.0, key: str = "x", **kwargs
    ) -> None:
        """Initialize the distribution"""
        super().__init__(key=key, **kwargs)
        self.low = low
        self.high = high

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this distribution."""
        return {**super().get_hparams(), "low": self.low, "high": self.high}

    def sample(self, batch: AtomsGraph, **kwargs) -> torch.Tensor:
        """Sample from the uniform distribution.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data.  The shape is derived from
            ``getattr(batch, self.key)``.

        Returns
        -------
        torch.Tensor
            Sampled tensor.
        """
        shape = getattr(batch, self.key).shape
        return torch.rand(shape) * (self.high - self.low) + self.low


class UniformCell(Uniform):
    """
    Uniform Prior Distribution for cell parameters
    """

    def sample(self, batch: AtomsGraph, **kwargs) -> torch.Tensor:
        """Sample uniformly within the unit cell.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data.

        Returns
        -------
        torch.Tensor
            Sampled Cartesian positions, shape ``(n_atoms, 3)``.
        """
        cell = batch.cell.clone()
        n_atoms = batch.n_atoms.sum().item()
        if batch.batch is not None:
            cell = cell.view(-1, 3, 3)[batch.batch]
            shape = (n_atoms, 3, 1)
            corner = torch.zeros(cell.shape[0], 3)
        else:
            shape = (n_atoms, 3)
            corner = torch.zeros(1, 3)

        f = torch.rand(shape) * (self.high - self.low) + self.low  # fractional coords

        if cell.dim() == 3:
            # Batched path: cell is (n_atoms, 3, 3), f is (n_atoms, 3, 1).
            r = (
                torch.matmul(cell, f).view((shape[0], shape[1]))
                + corner
            )  # (n_atoms, 3)
        else:
            # Single-graph path: cell is (3, 3), f is (n_atoms, 3).
            r = f @ cell + corner

        return r


class UniformCellConfined(UniformCell):
    """
    Uniform Prior Distribution for cell parameters with Z-directional confinement
    """

    def sample(self, batch: AtomsGraph, **kwargs) -> torch.Tensor:
        """Sample uniformly within the unit cell, confined to a Z range.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data.  ``batch.confinement`` must be set as a
            tensor of shape ``[num_graphs, 2]`` with ``z_min`` and ``z_max``
            per graph.

        Returns
        -------
        torch.Tensor
            Sampled Cartesian positions, shape ``(n_atoms, 3)``.
        """
        if batch.confinement is None:
            raise ValueError(
                "UniformCellConfined requires 'batch.confinement' to be set "
                "(a tensor of shape [num_graphs, 2] with z_min and z_max per graph)."
            )
        confinement = batch.confinement
        cell = batch.cell.clone()
        n_atoms = batch.n_atoms.sum().item()
        if batch.batch is not None:
            if confinement.shape[0] != batch.num_graphs:
                raise ValueError(
                    f"batch.confinement has {confinement.shape[0]} rows but "
                    f"the batch contains {batch.num_graphs} graphs. "
                    "Provide one [z_min, z_max] row per graph."
                )
            cell = cell.view(-1, 3, 3)[batch.batch]
            shape = (n_atoms, 3, 1)
            corner = torch.zeros(cell.shape[0], 3)

            conf_per_atom = confinement[batch.batch]  # (n_atoms, 2)
            z_dist = conf_per_atom[:, 1] - conf_per_atom[:, 0]
            z_min = conf_per_atom[:, 0]
            cell[:, 2, :2] = 0.0
            cell[:, 2, 2] = z_dist
            corner[:, 2] = z_min
        else:
            shape = (n_atoms, 3)
            corner = torch.zeros(1, 3)
            z_dist = confinement[:, 1] - confinement[:, 0]
            z_min = confinement[:, 0]
            cell[2, :2] = torch.tensor([0.0, 0.0])
            cell[2, 2] = z_dist
            corner[0, 2] = z_min

        f = torch.rand(shape) * (self.high - self.low) + self.low

        if cell.dim() == 3:
            r = (
                torch.matmul(cell, f).view((shape[0], shape[1]))
                + corner
            )
        else:
            r = f @ cell + corner

        return r
