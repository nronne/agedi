import torch
import numpy as np
from .potential import Potential
from agedi.data import AtomsGraph
from ase.calculators.emt import EMT as aseEMT


class EMT(Potential):
    """EMT Potential

    Implements the EMT potential.

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
        atoms_list = [g.to_atoms() for g in graph_list]

        Es = []
        for atoms in atoms_list:
            atoms.calc = aseEMT()
            Es.append(atoms.get_potential_energy())

        return torch.tensor(Es)

    def forces(self, batch: AtomsGraph) -> torch.Tensor:
        """Computes the forces of the atomistic structure.

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
        atoms_list = [g.to_atoms() for g in graph_list]

        Fs = []
        for atoms in atoms_list:
            atoms.calc = aseEMT()
            Fs.append(atoms.get_forces())

        Fs = np.vstack(Fs)
            
        return torch.tensor(Fs)
