import torch
from torch.nn import functional as F
from lightning import LightningModule

from typing import Dict, List

from torch_geometric.data import Batch
from agedi.models.translator import Translator
from agedi.data import Representation
from agedi.models.head import Head


#: Point-wise loss functions selectable via ``force_loss`` / ``energy_loss``.
LOSS_FUNCTIONS = ("huber", "mse", "mae")


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
    head_weights: Dict[str, float]
        Per-head multiplicative weights applied to the individual losses.
    use_weighting: bool
        Whether to weight the loss by the per-structure ``weight`` attribute.
    mask_forces: bool
        Whether to zero the predicted forces on masked (fixed) atoms.
    force_loss: str
        Point-wise loss used for the forces head: ``"huber"`` (default),
        ``"mse"``, or ``"mae"``.  The Huber loss behaves like MSE for small
        errors and like MAE beyond ``huber_delta``, which makes force training
        robust to the outliers typically present in DFT force labels.
    huber_delta: float
        Transition point of the Huber loss, in the force unit of the training
        data (eV/Å for ASE data).  Defaults to ``0.01``.
    energy_loss: str
        Point-wise loss used for the energy head.  Defaults to ``"mse"``.

    """

    def __init__(
        self,
        translator: Translator,
        representation: Representation,
        heads: List[Head] = [],
        head_weights = {},
        use_weighting: bool = False,
        mask_forces: bool = True,
        force_loss: str = "huber",
        huber_delta: float = 0.01,
        energy_loss: str = "mse",
        **kwargs
    ):
        """Constructor for the ScoreModel class."""
        super().__init__(**kwargs)
        self.translator = translator
        self.representation = representation
        self.head_weights = head_weights
        self.use_weighting = use_weighting
        self.mask_forces = mask_forces

        for name, kind in (("force_loss", force_loss), ("energy_loss", energy_loss)):
            if kind not in LOSS_FUNCTIONS:
                raise ValueError(
                    f"{name}='{kind}' is not recognized. "
                    f"Valid options: {', '.join(LOSS_FUNCTIONS)}."
                )
        if huber_delta <= 0:
            raise ValueError(f"huber_delta must be positive, got {huber_delta}.")

        self.force_loss = force_loss
        self.huber_delta = float(huber_delta)
        self.energy_loss = energy_loss

        self.head_keys = [head.key for head in heads]
        for key in self.head_keys:
            if key not in ["energy", "forces"]:
                raise ValueError(f"Head key {key} not recognized.")

        self.heads = torch.nn.ModuleList(heads)

    def get_config(self) -> Dict:
        """Return the non-module constructor arguments of this regressor.

        These are the settings that are *not* recoverable from the heads,
        translator, or representation, and therefore have to be carried
        separately when the regressor is rebuilt on top of a shared backbone
        (see :meth:`agedi.diffusion.Agedi.get_hparams`).

        Returns
        -------
        dict
            Keyword arguments accepted by :class:`RegressorModel`.
        """
        return {
            "head_weights": dict(self.head_weights),
            "use_weighting": bool(self.use_weighting),
            "mask_forces": bool(self.mask_forces),
            "force_loss": self.force_loss,
            "huber_delta": float(self.huber_delta),
            "energy_loss": self.energy_loss,
        }

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
            **self.get_config(),
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

    def _pointwise_loss(
        self, key: str, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the un-reduced point-wise loss for a given head.

        Parameters
        ----------
        key: str
            Head key (``"energy"`` or ``"forces"``); selects between
            ``energy_loss`` and ``force_loss``.
        prediction: torch.Tensor
            The predicted values.
        target: torch.Tensor
            The target values.

        Returns
        -------
        torch.Tensor
            Element-wise loss with the same shape as *prediction*.

        """
        kind = self.force_loss if key == "forces" else self.energy_loss

        if kind == "huber":
            return F.huber_loss(
                prediction, target, reduction="none", delta=self.huber_delta
            )
        if kind == "mae":
            return F.l1_loss(prediction, target, reduction="none")
        return F.mse_loss(prediction, target, reduction="none")

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

            pointwise = self._pointwise_loss(key, f_pred, f)

            if self.use_weighting and 'weight' in batch:
                # Energies are per-structure, forces per-atom: expand the
                # per-structure weights to the resolution of the head.
                if key == "energy":
                    weights = batch.weight.view(-1)
                else:
                    weights = batch.weight[batch.batch].unsqueeze(-1)
                head_loss = self.head_weights.get(key, 1.0) * (pointwise * weights).mean()
            else:
                head_loss = self.head_weights.get(key, 1.0) * pointwise.mean()

            loss["loss"] += head_loss
            loss[key + "_loss"] = head_loss

        return loss

    



