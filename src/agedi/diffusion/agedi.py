"""Agedi Lightning module.

This module contains :class:`Agedi`, a :class:`~lightning.LightningModule`
that wraps :class:`~agedi.diffusion.Diffusion` and adds
PyTorch-Lightning training/validation hooks, loss computation, and
checkpoint serialisation via :meth:`~Agedi.get_hparams`.

Force-field guidance utilities are provided by
:mod:`agedi.diffusion.guidance`, re-exported here for backwards compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from lightning import LightningModule
import torch

from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser
from agedi.models import ScoreModel
from agedi.utils.loss_balance import LossBalanceSpec, normalize_loss_balance

# Re-export from new locations for backwards compatibility
from .guidance import (  # noqa: F401
    ForcefieldGuidanceConfig,
    LBFGSStepSizer,
    BatchedLBFGSStepSizer,
)
from .diffusion import Diffusion


class Agedi(LightningModule, Diffusion):
    """Full diffusion model: training + sampling.

    Combines the :class:`~agedi.diffusion.Diffusion` sampling
    pipeline with :class:`~lightning.LightningModule` training hooks.

    Parameters
    ----------
    score_model : ScoreModel
        The score model.
    noisers : List[Noiser]
        A list of noisers.
    regressor_model : torch.nn.Module, optional
        An optional regressor model used for force-field guidance during
        sampling.  When present, its loss is added to the diffusion loss
        during training.
    regressor_heads : List, optional
        When provided, a :class:`~agedi.models.regressor.RegressorModel` is
        built internally using these heads while **sharing** the translator
        and representation from ``score_model``.  Use this parameter (instead
        of ``regressor_model``) when the backbone should be shared.
    regressor_kwargs : dict, optional
        Extra keyword arguments forwarded to
        :class:`~agedi.models.regressor.RegressorModel` when it is built from
        ``regressor_heads`` (e.g. ``force_loss``, ``huber_delta``).
    regressor_loss_weight : float, optional
        Absolute weight applied to the regressor loss:
        ``loss = diffusion_loss + regressor_loss_weight * regressor_loss``.
        Ignored when ``loss_balance`` is set.  Defaults to ``1.0``.
    loss_balance : float, str, sequence, or None, optional
        Relative split between the diffusion and regressor losses, e.g.
        ``"50:50"``, ``"80:20"``, ``(0.8, 0.2)``, or a single number giving the
        regressor fraction.  Each term is divided by a running estimate of its
        own magnitude before the fractions are applied, so the split means the
        same thing regardless of the raw loss scales of the system being
        studied.  ``None`` (default) uses the absolute
        ``regressor_loss_weight`` instead.  See
        :mod:`agedi.utils.loss_balance`.
    loss_balance_momentum : float, optional
        Momentum of the exponential moving averages tracking the two loss
        magnitudes.  Higher values average over more steps.  Defaults to
        ``0.99``.
    optim_config : dict, optional
        Keyword arguments forwarded to :class:`torch.optim.AdamW`.
    scheduler_config : dict, optional
        Keyword arguments forwarded to
        :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`.
    eps : float, optional
        Minimum diffusion time value.
    """

    def __init__(
        self,
        score_model: ScoreModel,
        noisers: List[Noiser],
        regressor_model: Optional[torch.nn.Module] = None,
        regressor_heads: Optional[List] = None,
        regressor_kwargs: Optional[Dict] = None,
        regressor_loss_weight: float = 1.0,
        loss_balance: "LossBalanceSpec" = None,
        loss_balance_momentum: float = 0.99,
        optim_config: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
        eps: float = 1e-5,
        fully_connected: bool = False,
    ) -> None:
        """Initializes the model."""
        if optim_config is None:
            optim_config = {"lr": 1e-4}
        if scheduler_config is None:
            scheduler_config = {"factor": 0.5, "patience": 10}
        # Initialise the nn.Module infrastructure first so that attribute
        # assignment (self.score_model = ...) correctly registers submodules.
        LightningModule.__init__(self)

        # Build or adopt the regressor, recording whether the backbone is shared.
        if regressor_heads is not None:
            from agedi.models.regressor import RegressorModel

            regressor_model = RegressorModel(
                translator=score_model.translator,
                representation=score_model.representation,
                heads=list(regressor_heads),
                **(regressor_kwargs or {}),
            )
            self._regressor_shares_backbone = True
        elif regressor_model is not None:
            self._regressor_shares_backbone = (
                regressor_model.translator is score_model.translator
                and regressor_model.representation is score_model.representation
            )
        else:
            self._regressor_shares_backbone = False

        # Initialise the sampler (sets score_model, noisers, regressor_model,
        # noiser_keys, score_keys, eps, lbfgs_step_sizer, zeta).
        Diffusion.__init__(self, score_model, noisers, regressor_model, eps)

        # Lightning-specific training attributes
        self.regressor_loss_weight = regressor_loss_weight
        self.loss_balance = normalize_loss_balance(loss_balance)
        self.loss_balance_momentum = float(loss_balance_momentum)
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self._regressor_training = False
        self.fully_connected = fully_connected

        # Running magnitudes of the two loss terms, used to make `loss_balance`
        # scale-free.  Non-persistent so that checkpoints written before this
        # existed still load; the averages re-initialise from the first
        # training batch after a resume.
        self.register_buffer("_loss_scales", torch.ones(2), persistent=False)
        self.register_buffer(
            "_loss_scales_initialized", torch.zeros((), dtype=torch.bool), persistent=False
        )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Write ``hparams.yaml`` to the trainer log directory at training start."""
        if self.trainer is None:
            return
        logger = getattr(self.trainer, "logger", None)
        if logger is None:
            return
        log_dir_str = getattr(logger, "log_dir", None)
        if not log_dir_str:
            return
        log_dir = Path(log_dir_str)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "hparams.yaml", "w") as fh:
            yaml.safe_dump({"diffusion": self.get_hparams()}, fh, default_flow_style=False)

    def get_hparams(self) -> Dict:
        """Return hyperparameters sufficient to reconstruct this diffusion model.

        Returns
        -------
        dict
            Hyperparameter dictionary with ``_target_``, ``score_model``,
            ``noisers``, ``optim_config``, ``scheduler_config``, ``eps``,
            and optionally ``regressor_heads`` or ``regressor_model``.
        """
        hparams: Dict = {
            "_target_": f"{type(self).__module__}.{type(self).__qualname__}",
            "score_model": self.score_model.get_hparams(),
            "noisers": [n.get_hparams() for n in self.noisers],
            "optim_config": dict(self.optim_config),
            "scheduler_config": dict(self.scheduler_config),
            "eps": self.eps,
            "regressor_loss_weight": float(self.regressor_loss_weight),
            "loss_balance": list(self.loss_balance) if self.loss_balance is not None else None,
            "loss_balance_momentum": float(self.loss_balance_momentum),
            "fully_connected": self.fully_connected,
        }
        if self.regressor_model is not None:
            if self._regressor_shares_backbone:
                hparams["regressor_heads"] = [
                    h.get_hparams() for h in self.regressor_model.heads
                ]
                if hasattr(self.regressor_model, "get_config"):
                    hparams["regressor_kwargs"] = self.regressor_model.get_config()
            else:
                hparams["regressor_model"] = self.regressor_model.get_hparams()
        return hparams

    def setup(self, stage: str = None) -> None:
        """Set up the model (put score model in training mode)."""
        self.score_model.training_mode()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, batch: AtomsGraph) -> AtomsGraph:
        """Forward pass through the score model.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.

        Returns
        -------
        AtomsGraph
            The output of the score model forward pass.
        """
        return self.score_model(batch)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    #: Floor applied to the running loss magnitudes, so that a term collapsing
    #: to zero cannot blow up the balanced loss.
    _LOSS_SCALE_EPS = 1e-12

    @torch.no_grad()
    def _update_loss_scales(
        self,
        diffusion_loss: torch.Tensor,
        regressor_loss: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Update and return the running magnitudes of the two loss terms.

        The averages are only advanced while training, so validation reuses the
        scales learned from the training batches and its reported loss stays
        comparable across epochs.

        Parameters
        ----------
        diffusion_loss : torch.Tensor
            The current diffusion loss.
        regressor_loss : torch.Tensor, optional
            The current regressor loss, or ``None`` when the batch carries no
            force labels (its scale is then left untouched).

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(2,)`` with the ``(diffusion, regressor)``
            magnitudes, floored at :attr:`_LOSS_SCALE_EPS`.
        """
        scales = self._loss_scales
        if self.training:
            observed = scales.clone()
            observed[0] = diffusion_loss.detach().abs()
            if regressor_loss is not None:
                observed[1] = regressor_loss.detach().abs()

            if bool(self._loss_scales_initialized):
                momentum = self.loss_balance_momentum
                scales.mul_(momentum).add_(observed, alpha=1.0 - momentum)
            else:
                scales.copy_(observed)
                self._loss_scales_initialized.fill_(True)

        return scales.clamp_min(self._LOSS_SCALE_EPS)

    def _combine_losses(
        self,
        diffusion_loss: torch.Tensor,
        regressor_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Combine the diffusion and regressor losses into the training objective.

        With ``loss_balance`` unset this is the plain weighted sum
        ``diffusion_loss + regressor_loss_weight * regressor_loss``.  With
        ``loss_balance`` set, each term is first divided by a running estimate
        of its own magnitude, so the requested fractions describe the actual
        contributions irrespective of the raw loss scales.

        Parameters
        ----------
        diffusion_loss : torch.Tensor
            The diffusion (denoising score-matching) loss.
        regressor_loss : torch.Tensor, optional
            The force-field loss, or ``None`` when unavailable for this batch.

        Returns
        -------
        Tuple[torch.Tensor, Dict]
            The total loss and a dict of extra metrics to log (the achieved
            contribution fractions, when balancing is active).
        """
        balance = self.loss_balance if self.regressor_model is not None else None

        if balance is None:
            if regressor_loss is None:
                return diffusion_loss, {}
            return diffusion_loss + self.regressor_loss_weight * regressor_loss, {}

        diffusion_weight, regressor_weight = balance
        scales = self._update_loss_scales(diffusion_loss, regressor_loss)

        diffusion_term = diffusion_weight * diffusion_loss / scales[0]
        total = diffusion_term
        if regressor_loss is None:
            return total, {}

        regressor_term = regressor_weight * regressor_loss / scales[1]
        total = total + regressor_term

        # Report what the split actually came out as this step, so the
        # requested balance can be verified during training.
        denominator = total.detach().abs().clamp_min(self._LOSS_SCALE_EPS)
        metrics = {
            "diffusion_fraction": diffusion_term.detach() / denominator,
            "regressor_fraction": regressor_term.detach() / denominator,
        }
        return total, metrics

    def loss(self, batch: AtomsGraph, batch_idx: torch.Tensor) -> Dict:
        """Compute the combined diffusion + regressor loss.

        Always computes the diffusion (denoising) loss on a noised copy of
        the batch.  When a regressor model is present and the batch contains
        force labels, the regressor loss is combined in according to
        ``loss_balance`` (relative split) or ``regressor_loss_weight``
        (absolute weight); see :meth:`_combine_losses`.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.
        batch_idx : torch.Tensor
            The index of the batch.

        Returns
        -------
        dict
            A dictionary of losses.
        """
        losses = self.diffusion_loss(batch, batch_idx)

        if self.regressor_model is not None and hasattr(batch, "forces"):
            reg_losses = self.regressor_loss(batch, batch_idx)
            total, metrics = self._combine_losses(losses["loss"], reg_losses.pop("loss"))
            losses["loss"] = total
            losses |= reg_losses
            losses |= metrics

        return losses

    def diffusion_loss(self, batch: AtomsGraph, batch_idx: torch.Tensor) -> Dict:
        """Compute the diffusion (denoising score-matching) loss.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.
        batch_idx : torch.Tensor
            The index of the batch.

        Returns
        -------
        dict
            A dictionary of losses.
        """
        noised_batch = batch.clone()
        if self.fully_connected:
            noised_batch["fully_connected"] = torch.tensor([1], device=noised_batch.pos.device)

        self.sample_time(noised_batch)
        noised_batch = self.forward_step(noised_batch)
        noised_batch = self.score_model(noised_batch)

        losses = {f"{noiser.key}_loss": 0 for noiser in self.noisers}
        losses["loss"] = 0.0
        for noiser in self.noisers:
            l = noiser.loss_scaling * noiser.loss(noised_batch)
            losses["loss"] += l
            losses[f"{noiser.key}_loss"] = l

        return losses

    def regressor_loss(self, batch: AtomsGraph, batch_idx: torch.Tensor) -> Dict:
        """Compute the regressor loss on the un-noised batch.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.
        batch_idx : torch.Tensor
            The index of the batch.

        Returns
        -------
        dict
            A dictionary of losses.

        Raises
        ------
        ValueError
            If no regressor model is attached.
        """
        if self.regressor_model is None:
            raise ValueError("Regressor model is not defined.")

        loss = self.regressor_model.loss(batch)
        loss["regressor_loss"] = loss["loss"]

        return loss

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: torch.Tensor) -> torch.Tensor:
        """Perform a training step.

        Computes the combined diffusion + regressor loss (see :meth:`loss`).

        When the :class:`~agedi.data.Dataset` was set up with a dedicated
        regressor dataset (via :meth:`~agedi.data.Dataset.add_regressor_data`),
        ``batch`` is a dict with two keys:

        * ``"main"`` – a regular training batch used for both the diffusion
          and regressor loss.
        * ``"regressor"`` – a regressor-only batch whose structures are *only*
          forwarded through the regressor loss (not the diffusion loss).

        When no regressor dataset is present ``batch`` is a plain
        :class:`~agedi.data.AtomsGraph` batch and the behaviour is identical
        to the pre-existing implementation.

        Parameters
        ----------
        batch : AtomsGraph or dict
            A batch of AtomsGraph data, or a dict with ``"main"`` and
            ``"regressor"`` keys when a dedicated regressor dataset is used.
        batch_idx : torch.Tensor
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The combined loss.
        """
        if isinstance(batch, dict):
            main_batch = batch["main"]
            regressor_batch = batch["regressor"]

            # Diffusion loss only on the main (equilibrium) batch.
            losses = self.diffusion_loss(main_batch, batch_idx)

            # Regressor loss on both batches whenever forces are available.
            if self.regressor_model is not None:
                reg_loss_total = torch.tensor(0.0, device=self.device)
                n_reg_batches = 0

                for b in (main_batch, regressor_batch):
                    if hasattr(b, "forces"):
                        reg_losses = self.regressor_loss(b, batch_idx)
                        reg_loss_total = reg_loss_total + reg_losses["loss"]
                        n_reg_batches += 1

                if n_reg_batches > 0:
                    reg_loss_avg = reg_loss_total / n_reg_batches
                    total, metrics = self._combine_losses(losses["loss"], reg_loss_avg)
                    losses["loss"] = total
                    losses["regressor_loss"] = reg_loss_avg
                    losses |= metrics

            total_batch_size = main_batch.num_graphs + regressor_batch.num_graphs
        else:
            losses = self.loss(batch, batch_idx)
            total_batch_size = batch.num_graphs

        for k, v in losses.items():
            name = "train_loss" if k == "loss" else f"train/{k}"
            self.log(name, v, on_step=True, on_epoch=True, batch_size=total_batch_size)
        return losses["loss"]

    def validation_step(
        self, batch: AtomsGraph, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """Perform a validation step.

        Parameters
        ----------
        batch : AtomsGraph
            A batch of AtomsGraph data.
        batch_idx : torch.Tensor
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The combined loss.
        """
        losses = self.loss(batch, batch_idx)
        for k, v in losses.items():
            name = "val_loss" if k == "loss" else f"val/{k}"
            self.log(name, v, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
        return losses["loss"]

    def configure_optimizers(self) -> Dict:
        """Configure optimizers and learning-rate schedulers.

        When a regressor model is present a single optimizer is built over
        the deduplicated union of ``score_model`` and ``regressor_model``
        parameters (shared parameters appear only once).

        Returns
        -------
        dict
            A dictionary with ``"optimizer"``, ``"lr_scheduler"``, and
            ``"monitor"`` keys.
        """
        if self.regressor_model is not None:
            seen: set = set()
            params = []
            for p in (
                list(self.score_model.parameters())
                + list(self.regressor_model.parameters())
            ):
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
            optimizer = torch.optim.AdamW(params, **self.optim_config)
        else:
            optimizer = torch.optim.AdamW(
                self.score_model.parameters(), **self.optim_config
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.scheduler_config
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self._scheduler_monitor(),
        }

    def _scheduler_monitor(self) -> str:
        """Return the metric used by ReduceLROnPlateau."""
        trainer = getattr(self, "_trainer", None)
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
        val_idx = getattr(datamodule, "val_idx", None) if datamodule is not None else None
        if val_idx is not None and len(val_idx) == 0:
            return "train_loss_epoch"
        return "val_loss"

    # ------------------------------------------------------------------
    # Regressor training toggle
    # ------------------------------------------------------------------

    @property
    def regressor_training(self) -> bool:
        """Whether the regressor model is in training mode."""
        if self.regressor_model is None:
            return False
        return self._regressor_training

    @regressor_training.setter
    def regressor_training(self, value: bool) -> None:
        """Set the regressor training flag.

        Parameters
        ----------
        value : bool
            New value.
        """
        if self.regressor_model is None:
            self._regressor_training = False
            return

        self._regressor_training = value
