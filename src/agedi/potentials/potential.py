from typing import Tuple
import torch
from abc import ABC, abstractmethod
from agedi.data import AtomsGraph


class Potential(ABC):
    """Potential Base class

    Implements a potential that can be used for the loss estimation.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments to be passed to the Potential class.

    Returns
    -------
    Potential

    """

    def __init__(self, **kwargs) -> None:
        """Initializes the Potential."""
        super().__init__(**kwargs)

    @abstractmethod
    def energy(self, batch: AtomsGraph) -> torch.Tensor:
        """Computes the energy of the atomistic structure.

        Must be implemented by the subclass.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be evaluated.

        Returns
        -------
        energy: torch.Tensor
            The energy of the atomistic structure.

        """
        pass

    @abstractmethod
    def forces(self, batch: AtomsGraph) -> torch.Tensor:
        """Computes the forces of the atomistic structure.

        Must be implemented by the subclass.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be evaluated.

        Returns
        -------
        forces: torch.Tensor
            The forces of the atomistic structure.

        """
        pass

    def __call__(self, batch: AtomsGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the energy and forces of the atomistic structure.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be evaluated.

        Returns
        -------
        energy: torch.Tensor
            The energy of the atomistic structure.
        forces: torch.Tensor
            The forces of the atomistic structure.

        """
        return self.energy(batch), self.forces(batch)
