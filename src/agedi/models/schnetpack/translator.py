from typing import Dict

import torch
from schnetpack.properties import (R, Z, cell, energy, forces, idx_i, idx_j,
                                   idx_m, n_atoms, offsets, pbc)

from agedi.data import Representation
from agedi.models.translator import Translator


class SchNetPackTranslator(Translator):
    """Translator for SchNetPack models.

    This class is used to translate the input data to the format required by the SchNetPack models.

    """

    def _translate(self, batch: "AtomsGraph") -> Dict[str, torch.Tensor]:
        """Translate the input batch to the format required by the model.
        
        The schnetpack model uses a dictionary format for the input data.

        The keywords in the dictionary given in schnetpack.properties and describes:
        - n_atoms: number of atoms in the system
        - Z: atomic numbers
        - R: atomic positions
        - cell: cell vectors
        - pbc: periodic boundary conditions
        - idx_i: edge indices
        - idx_j: edge indices
        - offsets: shift vectors
        - idx_m: batch indices describing which atoms belong to which structure

        Additionally energy and forces targets can be added to the dictionary.

        Parameters
        ----------
        batch: AtomsGraph
            The input batch of data.

        Returns
        -------
        Dict
            The translated batch of data.

        """

        out = {
            n_atoms: batch.n_atoms[:, 0],
            Z: batch.x,
            R: batch.pos,
            cell: batch.cell.reshape(-1, 3, 3),
            pbc: batch.pbc,
            offsets: batch.shift_vectors,
            idx_m: batch.batch,
            idx_i: batch.edge_index[0],
            idx_j: batch.edge_index[1],
        }

        if hasattr(batch, "energy"):
            out[energy] = batch.energy.view(-1)
        if hasattr(batch, "forces"):
            out[forces] = batch.forces

        return out

    def _translate_representation(
        self, representation: Representation, translated_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Translate the representation to the format required by the model.

        SchnetPack uses scalar_representation and vector_representation for the two types of representations.

        Parameters
        ----------
        representation: Representation
            The input representation.
        translated_batch: Dict
            The translated batch of data.

        Returns
        -------
        Dict
            The translated batch with representation keys.
        
        """
        translated_batch["scalar_representation"] = representation.scalar.squeeze(2)
        translated_batch["vector_representation"] = representation.vector.permute(
            0, 2, 1
        )
        return translated_batch

    def _get_representation(self, batch, translated_batch):
        """Get the representation from the output of the model.

        Parameters
        ----------
        batch: AtomsGraph
            The input batch of data.
        translated_batch: Dict
            The output of the model.

        Returns
        -------
        Representation
            The representation output of the model.
        
        """
        
        s, v = translated_batch["scalar_representation"], translated_batch["vector_representation"]
        s = s.unsqueeze(2)
        v = torch.permute(v, (0, 2, 1))
        return Representation(scalar=s, vector=v)
