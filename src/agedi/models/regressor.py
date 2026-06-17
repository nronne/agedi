import torch
from torch.nn import functional as F
from lightning import LightningModule

from typing import Dict, List

from torch_geometric.data import Batch
from agedi.models.translator import Translator
from agedi.data import Representation
from agedi.models.head import Head


class RegressorModel(LightningModule):
    """Class that defines a regressor model.

    It is a combination of a translator, a representation
    and a list of heads.

    Parameters
    ----------
    translator: Translator
        The translator that will be used to translate the input batch.
    representation: Representation
        The representation that will be used to represent the translated batch.
    heads: List[Head]
        The list of heads that will be used to compute scores.
    head_weights : dict, optional
        Per-head loss scaling factors.
    use_weighting : bool, optional
        When ``True``, per-structure weights stored in ``batch.weight`` are
        applied to each head's loss.
    mask_forces : bool, optional
        When ``True``, predicted forces on fixed atoms (marked in
        ``batch.mask``) are zeroed and those atoms are excluded from the
        force loss.
    force_loss_type : str, optional
        Loss function used for the forces head.  ``"mse"`` (default) uses
        mean-squared error.  ``"huber"`` uses Huber loss (smooth L1), which
        is more robust to the large forces that occur in non-equilibrium
        structures.  Energy loss always uses MSE.
    huber_delta : float, optional
        Threshold parameter for Huber loss (in eV/Å).  Ignored when
        ``force_loss_type="mse"``.  Default: ``1.0``.
    """

    def __init__(
        self,
        translator: Translator,
        representation: Representation,
        heads: List[Head] = [],
        head_weights = {},
        use_weighting: bool = False,
        mask_forces: bool = True,
        force_loss_type: str = "mse",
        huber_delta: float = 1.0,
        **kwargs
    ):
        """Constructor for the ScoreModel class."""
        super().__init__(**kwargs)
        self.translator = translator
        self.representation = representation
        self.head_weights = head_weights
        self.use_weighting = use_weighting
        self.mask_forces = mask_forces
        if force_loss_type not in ("mse", "huber"):
            raise ValueError(
                f"force_loss_type must be 'mse' or 'huber', got {force_loss_type!r}"
            )
        self.force_loss_type = force_loss_type
        self.huber_delta = huber_delta
        
        self.head_keys = [head.key for head in heads]
        for key in self.head_keys:
            if key not in ["energy", "forces"]:
                raise ValueError(f"Head key {key} not recognized.")
        
        self.heads = torch.nn.ModuleList(heads)

    def get_hparams(self) -> Dict:
        """Return hyperparameters sufficient to reconstruct this regressor model.

        Returns
        -------
        dict
            Hyperparameter dictionary with a ``_target_`` key and nested
            ``translator``, ``representation``, and ``heads`` entries.
        """
        return {
            "_target_": f"{type(self).__module__}.{type(self).__qualname__}",
            "translator": self.translator.get_hparams(),
            "representation": self.translator.get_representation_hparams(self.representation),
            "heads": [h.get_hparams() for h in self.heads],
        }

    def forward(self, batch: Batch) -> Batch:
        """Forward pass of the model.

        Parameters
        ----------
        batch: Batch
            The input batch that will be used to compute the scores.

        Returns
        -------
        Batch
            The output batch containing the scores.

        """
        translated_batch = self.translator.translate_input(batch)
        
        rep = self.representation(translated_batch)
        batch = self.translator.add_representation(batch, rep)
        translated_batch = self.translator.translate_with_representation(batch)

        for head in self.heads:
            predictions = {}
            predictions[head.key] = head(translated_batch)

            if head.key == "forces":
                if hasattr(batch, 'mask') and self.mask_forces:
                    predictions[head.key][batch.positions_mask] = 0.0

            if head.key == "energy":
                type = "graph"
            elif head.key == "forces":
                type = "node"
            else:
                type = None
            batch = self.translator.add_prediction(batch, predictions, type=type)


        return batch

    def loss(self, batch: Batch) -> Dict:
        """Compute the loss of the model.

        Parameters
        ----------
        batch: Batch
            The input batch that will be used to compute the loss.

        Returns
        -------
        dict
            A dictionary containing the loss and the individual head losses.

        """
        batch = self(batch)

        loss = {"loss": 0.0}
        for key in self.head_keys:
            f = batch[key]
            f_pred = batch[f"{key}_prediction"]

            if key == "energy":
                n_atoms = batch.n_atoms.squeeze(-1)
                f = f / n_atoms
                f_pred = f_pred / n_atoms

            # For forces with fixed atoms: exclude them from the loss entirely.
            # Their predictions are already zeroed in forward(), so they carry
            # no gradient, but including them inflates the reported loss by a
            # constant (target² for each masked atom).
            if key == "forces" and hasattr(batch, "mask") and self.mask_forces:
                movable = ~batch.mask  # [N] bool — True for non-fixed atoms
                f = f[movable]
                f_pred = f_pred[movable]

            if self.use_weighting and "weight" in batch:
                if key == "energy":
                    # Per-graph weights [B] matching the energy tensors [B].
                    weights = batch.weight
                else:
                    # Per-atom weights expanded from per-graph; trim to movable
                    # atoms when the mask is active.
                    atom_w = batch.weight[batch.batch]  # [N]
                    if key == "forces" and hasattr(batch, "mask") and self.mask_forces:
                        atom_w = atom_w[movable]        # [M]
                    weights = atom_w.unsqueeze(-1)      # [N/M, 1]

                if key == "forces" and self.force_loss_type == "huber":
                    raw = F.huber_loss(f_pred, f, reduction="none", delta=self.huber_delta)
                else:
                    raw = F.mse_loss(f, f_pred, reduction="none")
                head_loss = self.head_weights.get(key, 1.0) * (raw * weights).mean()
            else:
                if key == "forces" and self.force_loss_type == "huber":
                    head_loss = self.head_weights.get(key, 1.0) * F.huber_loss(
                        f_pred, f, delta=self.huber_delta
                    )
                else:
                    head_loss = self.head_weights.get(key, 1.0) * F.mse_loss(f, f_pred)

            loss["loss"] += head_loss
            loss[key + "_loss"] = head_loss

        return loss

    



