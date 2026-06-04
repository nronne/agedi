from typing import Any, Dict

import torch
from schnetpack.properties import R, Z, cell, energy, forces, idx_i, idx_j, idx_m, n_atoms, offsets, pbc

from agedi.data import Representation
from agedi.models.translator import Translator


class SchNetPackTranslator(Translator):
    """Translator for SchNetPack models.

    This class is used to translate the input data to the format required by the SchNetPack models.

    """

    def get_representation_hparams(self, representation: Any) -> Dict:
        """Extract hyperparameters from a SchNetPack representation object.

        Supports :class:`schnetpack.representation.PaiNN`.  Extracts
        ``n_atom_basis``, ``n_interactions``, and nested configs for the
        ``radial_basis`` and ``cutoff_fn`` so that the representation can be
        fully reconstructed with :func:`~agedi.functional._instantiate_from_config`.

        Parameters
        ----------
        representation : any
            An instantiated SchNetPack representation object.

        Returns
        -------
        dict
            Hyperparameter dictionary with a ``_target_`` key and
            representation-specific parameters.

        Raises
        ------
        NotImplementedError
            If the representation type is not recognised.
        """
        import schnetpack as spk

        if isinstance(representation, spk.representation.PaiNN):
            hparams: Dict[str, Any] = {
                "_target_": f"{type(representation).__module__}.{type(representation).__qualname__}",
                "n_atom_basis": representation.n_atom_basis,
                "n_interactions": representation.n_interactions,
            }
            # Encode radial_basis as a nested instantiation config
            if hasattr(representation, "radial_basis"):
                rb = representation.radial_basis
                rb_hparams: Dict[str, Any] = {
                    "_target_": f"{type(rb).__module__}.{type(rb).__qualname__}",
                    "n_rbf": int(rb.n_rbf),
                }
                # Cutoff is stored implicitly in the last offset value
                if hasattr(rb, "offsets") and rb.offsets.numel() > 0:
                    rb_hparams["cutoff"] = float(rb.offsets[-1])
                hparams["radial_basis"] = rb_hparams
            # Encode cutoff_fn as a nested instantiation config
            if hasattr(representation, "cutoff_fn"):
                cf = representation.cutoff_fn
                cf_hparams: Dict[str, Any] = {
                    "_target_": f"{type(cf).__module__}.{type(cf).__qualname__}",
                }
                if hasattr(cf, "cutoff") and cf.cutoff.numel() > 0:
                    cf_hparams["cutoff"] = float(cf.cutoff[0])
                hparams["cutoff_fn"] = cf_hparams
            return hparams

        raise NotImplementedError(
            f"get_representation_hparams is not implemented for "
            f"representation type '{type(representation).__name__}'"
        )

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
            'cellpar': batch.cellpar,
        }

        if hasattr(batch, "energy"):
            out[energy] = batch.energy.view(-1)
        if hasattr(batch, "forces"):
            out[forces] = batch.forces

        # EDM preconditioning: expose σ = √var(t) and noisy positions to heads.
        if hasattr(batch, "pos_sigma"):
            out["pos_sigma"] = batch.pos_sigma
        out["pos"] = batch.pos  # noisy positions for skip connection

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

    def _get_representation(self, batch: "AtomsGraph", translated_batch: Dict[str, torch.Tensor]) -> "Representation":
        """Get the representation from the output of the model.

        Parameters
        ----------
        batch : AtomsGraph
            The input batch of data.
        translated_batch : Dict[str, torch.Tensor]
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
