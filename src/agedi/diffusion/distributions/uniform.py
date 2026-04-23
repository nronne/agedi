import torch
from typing import Dict, Optional
from agedi.diffusion.distributions import Distribution
from agedi.data import AtomsGraph


class Uniform(Distribution):
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

    def _setup(self, batch: AtomsGraph) -> None:
        """Prepare the distribution for sampling from *batch*.

        Sets ``self.shape`` to the shape of the target attribute in the batch.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data.
        """
        if self.key is not None:
            self.shape = getattr(batch, self.key).shape

    def _sample(self, shape: Optional[torch.Size] = None, **kwargs) -> torch.Tensor:
        """
        Sample from the uniform distribution

        Parameters
        ----------
        shape : torch.Size
            The shape of the sample

        Returns
        -------
        torch.Tensor
            Sampled tensor

        """
        shape = shape if shape is not None else self.shape
        return torch.rand(shape) * (self.high - self.low) + self.low


class UniformCell(Uniform):
    """
    Uniform Prior Distribution for cell parameters
    """

    def _setup(self, batch: AtomsGraph) -> None:
        """
        Prepare the distribution for sampling of the batch

        Parameters
        ----------
        batch : AtomsGraph
            Batch of data

        Returns
        -------
        None

        """
        super()._setup(batch)
        self.cell = batch.cell.clone()
        n_atoms = batch.n_atoms.sum().item()
        if batch.batch is not None:
            self.cell = self.cell.view(-1, 3, 3)[batch.batch]
            self.shape = (n_atoms, 3, 1)
            self.corner = torch.zeros(self.cell.shape[0], 3)

        else:
            self.shape = (n_atoms, 3)
            self.corner = torch.zeros(1, 3)

    def _sample(self, **kwargs) -> torch.Tensor:
        """Sample from the uniform distribution

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
        f = super()._sample()  # (n_atoms, 3)
        if self.cell.dim() == 3:
            # Batched path: self.cell is (n_atoms, 3, 3), f is (n_atoms, 3, 1).
            # cell[i] @ f[i] gives the Cartesian position for atom i.
            r = (
                torch.matmul(self.cell, f).view((self.shape[0], self.shape[1]))
                + self.corner
            )  # (n_atoms, 3)
        else:
            # Single-graph path: self.cell is (3, 3), f is (n_atoms, 3).
            # f @ cell maps fractional coordinates to Cartesian (row-vector convention).
            r = f @ self.cell + self.corner

        return r


class UniformCellConfined(UniformCell):
    """
    Uniform Prior Distribution for cell parameters with Z-directional confinement
    """

    def _setup(self, batch: AtomsGraph) -> None:
        """
        Prepare the distribution for sampling of the batch

        Parameters
        ----------
        batch : AtomsGraph
            Batch of data

        Returns
        -------
        None

        """
        super()._setup(batch)
        if batch.confinement is None:
            raise ValueError(
                "UniformCellConfined requires 'batch.confinement' to be set "
                "(a tensor of shape [num_graphs, 2] with z_min and z_max per graph)."
            )
        self.confinement = batch.confinement
        if batch.batch is not None:
            if self.confinement.shape[0] != batch.num_graphs:
                raise ValueError(
                    f"batch.confinement has {self.confinement.shape[0]} rows but "
                    f"the batch contains {batch.num_graphs} graphs. "
                    "Provide one [z_min, z_max] row per graph."
                )
            conf_per_atom = self.confinement[batch.batch]  # (n_atoms, 2)
            z_dist = conf_per_atom[:, 1] - conf_per_atom[:, 0]  # (n_atoms,)
            z_min = conf_per_atom[:, 0]  # (n_atoms,)
            # self.cell is (n_atoms, 3, 3) from parent _setup
            self.cell[:, 2, :2] = 0.0
            self.cell[:, 2, 2] = z_dist
            self.corner[:, 2] = z_min
        else:
            z_dist = self.confinement[:, 1] - self.confinement[:, 0]
            z_min = self.confinement[:, 0]
            self.cell[2, :2] = torch.tensor([0.0, 0.0])
            self.cell[2, 2] = z_dist
            self.corner[0, 2] = z_min
