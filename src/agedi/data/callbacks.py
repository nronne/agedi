import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch_geometric.transforms import BaseTransform


class GradNormLogger(Callback):
    """Logs the total gradient norm of the score-model parameters before each optimizer step.

    Parameters
    ----------
    log_every_n_steps:
        Log the gradient norm every this many optimizer steps (default: ``50``).
        Set to ``1`` to log every step.
    """

    def __init__(self, log_every_n_steps: int = 50):
        if log_every_n_steps < 1:
            raise ValueError(f"log_every_n_steps must be >= 1, got {log_every_n_steps}")
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        norms = [
            p.grad.detach().norm()
            for p in pl_module.score_model.parameters()
            if p.grad is not None
        ]
        if norms:
            total_norm = torch.stack(norms).norm()
            pl_module.log("grad_norm", total_norm, on_step=True, on_epoch=False)


class EpochProgressPrinter(Callback):
    """Prints epoch-level training progress to stdout at a configurable interval.

    Parameters
    ----------
    print_epoch_interval:
        Print a summary line every this many epochs (default: 10).
    """

    def __init__(self, print_epoch_interval: int = 10):
        if print_epoch_interval < 1:
            raise ValueError(
                f"print_epoch_interval must be >= 1, got {print_epoch_interval}"
            )
        self.print_epoch_interval = print_epoch_interval
        self._fit_start_time: float = 0.0

    def on_fit_start(self, trainer, pl_module):
        self._fit_start_time = time.monotonic()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch + 1
        if epoch % self.print_epoch_interval != 0:
            return

        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss_epoch", metrics.get("train_loss"))
        val_loss = metrics.get("val_loss")

        # LearningRateMonitor logs LR as 'lr-<OptimizerClass>' or 'lr-<name>'
        lr = None
        for k, v in metrics.items():
            if k.lower().startswith("lr"):
                lr = float(v)
                break

        parts = [f"Epoch {epoch:>6d}"]
        if train_loss is not None:
            parts.append(f"train_loss: {float(train_loss):.4f}")
        if val_loss is not None:
            parts.append(f"val_loss: {float(val_loss):.4f}")
        if lr is not None:
            parts.append(f"lr: {lr:.2e}")

        print(" | ".join(parts))

    def on_fit_end(self, trainer, pl_module):
        elapsed = time.monotonic() - self._fit_start_time
        hours, rem = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(rem, 60)

        best_val_loss: Optional[float] = None
        best_ckpt_path: Optional[str] = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.monitor == "val_loss":
                if cb.best_model_score is not None:
                    best_val_loss = float(cb.best_model_score)
                best_ckpt_path = cb.best_model_path or None
                break

        print(f"\nTraining complete  |  elapsed: {hours:02d}h {minutes:02d}m {seconds:02d}s")
        if best_val_loss is not None:
            print(f"Best val_loss      : {best_val_loss:.6f}")
        if best_ckpt_path:
            print(f"Best checkpoint    : {best_ckpt_path}")


def _flatten_hparams(d: dict, prefix: str = "", sep: str = "/") -> dict:
    """Recursively flatten a nested dict into a flat dict with dotted keys.

    Only scalar values (int, float, str, bool) are kept; lists and nested
    dicts are flattened recursively.  This is required for TensorBoard's
    ``log_hyperparams`` which only accepts scalar values.

    Parameters
    ----------
    d : dict
        The nested hyperparameter dict to flatten.
    prefix : str
        Key prefix to prepend (used in recursion).
    sep : str
        Separator between key segments (default ``"/"``).

    Returns
    -------
    dict
        Flat dict with scalar values only.
    """
    result: dict = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            result.update(_flatten_hparams(v, prefix=key, sep=sep))
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    result.update(_flatten_hparams(item, prefix=f"{key}/{i}", sep=sep))
                elif isinstance(item, (int, float, str, bool)):
                    result[f"{key}/{i}"] = item
        elif isinstance(v, (int, float, str, bool)):
            result[key] = v
    return result


class HParamsMetricLogger(Callback):
    """Manages hyperparameter logging and populates the TensorBoard hp_metric panel.

    When a full ``hparams`` dict is provided (including training metadata such
    as ``distribution``, ``prior``, ``sde``, ``conditioning``, ``batch_size``,
    etc.) it is written to ``hparams.yaml`` at training start, complementing the
    baseline written by
    :meth:`~agedi.diffusion.diffusion.Diffusion.on_fit_start`.  When no dict
    is provided the callback falls back to calling ``pl_module.get_hparams()``,
    which returns only the model-architecture config.

    For non-TensorBoard loggers (e.g. WandB) the resolved hparams dict is
    forwarded to ``log_hyperparams`` at training start.

    Parameters
    ----------
    hparams:
        Full hyperparameter dictionary to log (architecture + training metadata).
        When ``None`` the callback resolves hparams from ``pl_module.get_hparams()``.
    """

    def __init__(self, hparams: Optional[Dict] = None):
        self._hparams = hparams

    def _resolve_hparams(self, pl_module) -> Dict:
        if self._hparams is not None:
            return self._hparams
        if hasattr(pl_module, "get_hparams"):
            return {"diffusion": pl_module.get_hparams()}
        return {}

    # Keys excluded from TensorBoard hparam logging (kept in hparams.yaml).
    _TB_EXCLUDE_KEYS = frozenset({"cell"})

    def _flatten_for_tb(self, resolved: dict) -> dict:
        """Flatten hparams for TensorBoard, excluding keys that are not useful there."""
        filtered = {k: v for k, v in resolved.items() if k not in self._TB_EXCLUDE_KEYS}
        return _flatten_hparams(filtered)

    def on_train_start(self, trainer, pl_module):
        from lightning.pytorch.loggers import TensorBoardLogger

        resolved = self._resolve_hparams(pl_module)
        # Write (or overwrite) hparams.yaml with the full resolved dict so
        # that metadata fields are persisted and available to `agedi inspect`.
        if isinstance(trainer.logger, TensorBoardLogger):
            log_dir = Path(trainer.logger.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / "hparams.yaml", "w") as fh:
                yaml.safe_dump(resolved, fh, default_flow_style=False)
            # Log hparams to TensorBoard at training start (with a nan placeholder
            # metric) so runs appear in the HPARAMS tab immediately and can be
            # compared across concurrent or interrupted runs.
            flat = self._flatten_for_tb(resolved)
            if flat:
                trainer.logger.log_hyperparams(flat, {"hp_metric": float("nan")})
        elif trainer.logger is not None:
            # For non-TensorBoard loggers (e.g. WandB), forward the resolved hparams.
            trainer.logger.log_hyperparams(resolved)

    def on_validation_epoch_end(self, trainer, pl_module):
        from lightning.pytorch.loggers import TensorBoardLogger

        if trainer.sanity_checking or not isinstance(trainer.logger, TensorBoardLogger):
            return

        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return

        hparams = self._resolve_hparams(pl_module)
        flat = self._flatten_for_tb(hparams)
        if flat:
            trainer.logger.log_hyperparams(flat, {"hp_metric": float(val_loss)})

    def on_fit_end(self, trainer, pl_module):
        from lightning.pytorch.loggers import TensorBoardLogger

        if not isinstance(trainer.logger, TensorBoardLogger):
            return

        best_val_loss: Optional[float] = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.monitor == "val_loss":
                if cb.best_model_score is not None:
                    best_val_loss = float(cb.best_model_score)
                break

        hparams = self._resolve_hparams(pl_module)
        flat = self._flatten_for_tb(hparams)
        if not flat:
            return
        # Call log_hyperparams at the end of training so TensorBoard shows the
        # final best val_loss as the hp_metric value.
        metrics = {"hp_metric": best_val_loss} if best_val_loss is not None else {"hp_metric": float("nan")}
        trainer.logger.log_hyperparams(flat, metrics)


class TrainingPhase(Callback):
    """Lightning callback that advances the dataset through training phases.

    Each phase can use a different set of data augmentation transforms (e.g.
    supercell repeats).  The callback monitors epoch count and calls
    :meth:`~agedi.data.Dataset.set_phase` on the datamodule when it is time
    to move to the next phase.
    """

    def __init__(
        self,
        n_phases: int,
        epochs_per_phase: List[int],
        **kwargs
    ) -> None:
        """Initialize the training phase callback.

        Parameters
        ----------
        n_phases : int
            Total number of training phases.
        epochs_per_phase : list[int]
            Number of epochs to spend in each phase (length ``n_phases - 1``).
        **kwargs
            Additional keyword arguments forwarded to :class:`~lightning.pytorch.callbacks.Callback`.
        """
        super().__init__(**kwargs)
        self.n_phases = n_phases

        self.epochs_per_phase = epochs_per_phase
        self.epoch_counter = 0

        self.current_phase = 0


    def _prepare_epoch(self, trainer: Trainer, model: LightningModule) -> None:
        """Advance to the next training phase if enough epochs have elapsed.

        Called at the end of each validation epoch.  When the epoch counter
        reaches the threshold for the current phase, the datamodule is
        instructed to switch to the next phase via
        :meth:`~agedi.data.Dataset.set_phase`.

        Parameters
        ----------
        trainer : Trainer
            The active Lightning trainer.
        model : LightningModule
            The model being trained (unused, required by Lightning callback API).
        """

        if self.current_phase == self.n_phases - 1:
            return

        epoch = trainer.current_epoch
        if self.epoch_counter >= self.epochs_per_phase[self.current_phase]:
            self.current_phase += 1
            trainer.datamodule.set_phase(self.current_phase)
            
            self.epoch_counter = 0
        else:
            self.epoch_counter += 1
            

    def on_validation_end(self, trainer: Trainer, model: LightningModule) -> None:
        """Hook called by Lightning at the end of each validation epoch.

        Delegates to :meth:`_prepare_epoch` to check whether the current
        training phase should advance.

        Parameters
        ----------
        trainer : Trainer
            The active Lightning trainer.
        model : LightningModule
            The model being trained.
        """
        self._prepare_epoch(trainer, model)
