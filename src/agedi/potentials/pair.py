import torch
import numpy as np
from .potential import Potential
from agedi.data import AtomsGraph


class Pair(Potential):
    """Lennard Jones Potential

    Implements the Lennard Jones potential.

    Parameters
    ----------
    **kwargs
            Additional keyword arguments to be passed to the Potential class.

    Returns
    -------
    EMT

    """

    def __init__(self, **kwargs) -> None:
        """Initializes the EMT potential."""
        super().__init__(**kwargs)

    def energy(self, batch: AtomsGraph) -> torch.Tensor:
        """Computes the energy of the atomistic structure.

        Parameters
        ----------
        batch: AtomsGraph
        The atomistic structure (or batch hereof) to be evaluated.

        Returns
        -------
        energy: torch.Tensor
        The energy of the atomistic structure.

        """
        graph_list = batch.to_data_list()
        Es = []
        for g in graph_list:
            r = g.pos
            Es.append(self._energy(r))

        return torch.tensor(Es)

    def _energy(self, r: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(r, r)
        E = torch.sum((dist-1.0)**2)
        
        return E


    def forces(self, batch: AtomsGraph) -> torch.Tensor:
        """Computes the forces of _energy using autograd

        Parameters
        ----------
        batch: AtomsGraph
        The atomistic structure (or batch hereof) to be evaluated.

        Returns
        -------
        forces: torch.Tensor
        The forces of the atomistic structure.

        """

        graph_list = batch.to_data_list()
        Fs = []
        with torch.enable_grad():
            for g in graph_list:
                r = g.pos.clone().detach().requires_grad_(True)
                E = self._energy(r)

                F = -torch.autograd.grad(E, r)[0]
                Fs.append(F)

        return torch.cat(Fs)
