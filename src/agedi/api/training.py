"""Training orchestration."""

import logging
import math
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from ase import Atoms
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from agedi.data import Dataset
from agedi.data.callbacks import (
    EpochProgressPrinter,
    GradNormLogger,
    HParamsMetricLogger,
    TrainingPhase,
)

from ._display import _extract_data_info, _print_log_path, _print_training_config
from .dataset import create_dataset
from .diffusion import create_diffusion, load_diffusion


# ---------------------------------------------------------------------------
# Config key sets for train_from_config
# ---------------------------------------------------------------------------

#: Parameters forwarded to :func:`train_from_atoms` from the config dict.
_TRAIN_FROM_ATOMS_KEYS = frozenset(
    [
        "model",
        "cutoff",
        "feature_size",
        "n_blocks",
        "n_rbf",
        "radial_basis",
        "noisers",
        "sde",
        "conditioning",
        "conditioning_type",
        "mask",
        "confinement",
        "force_field",
        "batch_size",
        "train_split",
        "val_split",
        "repeat",
        "canonical_cell",
        "lr",
        "lr_factor",
        "lr_patience",
        "weight_decay",
        "eps",
        "guidance_weight",
        "n_classes",
        "checkpoint",
    ]
)

#: Parameters forwarded to :func:`create_trainer` (as ``**trainer_kwargs``).
_TRAINER_KEYS = frozenset(
    [
        "epochs",
        "max_time",
        "logger",
        "log_dir",
        "project",
        "name",
        "log_interval",
        "gradient_clip_val",
        "progress_bar",
        "repeat_epoch",
    ]
)


def create_trainer(
    *,
    epochs: int = -1,
    max_time: Optional[Union[int, Dict, timedelta]] = 24,
    accelerator: str = "auto",
    devices: int = 1,
    logger: str = "tensorboard",
    log_dir: str = "logs",
    project: str = "agedi",
    name: str = "agedi",
    log_interval: int = 10,
    gradient_clip_val: float = 10.0,
    progress_bar: bool = False,
    print_epoch_interval: int = 10,
    log_grad_norm: bool = True,
    repeat: Optional[int] = None,
    repeat_epoch: Optional[int] = None,
    hparams: Optional[Dict] = None,
    extra_callbacks: Optional[List[Callback]] = None,
) -> Trainer:
    """Create a Lightning trainer configured for AGeDi.

    Parameters
    ----------
    epochs:
        Maximum number of training epochs (``-1`` = unlimited).
    max_time:
        Wall-clock time limit for training.  Accepts:

        * ``int``   – number of *hours* (e.g. ``24`` ≡ 24 hours).
        * ``dict``  – Lightning-style mapping, e.g.
          ``{"days": 0, "hours": 12, "minutes": 30, "seconds": 0}``.
        * :class:`datetime.timedelta` – a Python timedelta object.
        * ``None``  – no time limit.
    accelerator:
        Hardware accelerator to use (e.g. ``"auto"``, ``"gpu"``, ``"cpu"``).
        Default: ``"auto"``.
    devices:
        Number of devices to train on.  Default: ``1``.
    logger:
        Logging backend: ``"tensorboard"`` (default) or ``"wandb"``.
    log_dir:
        Root directory for logs and checkpoints.  Default: ``"logs"``.
    project:
        WandB project name (only used when ``logger="wandb"``).
    name:
        Experiment display name used by TensorBoard and WandB as the
        run sub-directory / run name.  Default: ``"agedi"``.
    log_interval:
        How often (in steps) to log metrics.  Default: ``10``.
    gradient_clip_val:
        Maximum gradient norm for gradient clipping.  Default: ``10.0``.
    progress_bar:
        Whether to show a Lightning progress bar.  Default: ``False``.
    print_epoch_interval:
        Print a one-line training summary to stdout every this many epochs.
        Set to ``0`` to disable.  Default: ``10``.
    log_grad_norm:
        Whether to log the total gradient norm during training.
        Disable for large models where the per-step overhead is undesirable.
        Default: ``True``.
    repeat:
        Number of repetition levels for cell-repeat data augmentation.
        Must be set together with *repeat_epoch*.  When ``None`` (default),
        no repetition augmentation is applied.
    repeat_epoch:
        How many epochs between repetition-level increases.  Required when
        *repeat* is set.
    hparams:
        Hyperparameters dict logged to ``hparams.yaml`` via
        :class:`~agedi.data.callbacks.HParamsMetricLogger`.  When ``None``
        (default), no extra hyperparameter logging is performed.
    extra_callbacks:
        Extra Lightning callbacks to append to the default callback list.
        When ``None`` (default) only the built-in callbacks are used.

    Returns
    -------
    lightning.Trainer
        A configured :class:`~lightning.Trainer` ready to call
        ``trainer.fit(diffusion, dataset)``.
    """
    if max_time is None:
        _max_time = None
    elif isinstance(max_time, int):
        _max_time = {"hours": max_time}
    elif isinstance(max_time, (dict, timedelta)):
        _max_time = max_time
    else:
        raise TypeError(
            f"max_time must be None, int (hours), dict, or timedelta; "
            f"got {type(max_time).__name__}"
        )
    if logger == "tensorboard":
        run_logger = TensorBoardLogger(save_dir=log_dir, name=name)
    elif logger == "wandb":
        run_logger = WandbLogger(
            save_dir=log_dir,
            project=project,
            name=name,
        )
    else:
        raise ValueError(
            f'Unknown logger "{logger}". Valid options are: tensorboard, wandb'
        )

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            filename="best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        ModelCheckpoint(
            filename="last_model",
            monitor=None,
            save_top_k=1,
            every_n_epochs=1,
        ),
    ]

    if log_grad_norm:
        callbacks.append(GradNormLogger(log_every_n_steps=log_interval))

    if print_epoch_interval > 0:
        callbacks.append(EpochProgressPrinter(print_epoch_interval))

    if repeat is not None:
        if repeat_epoch is None:
            raise ValueError("repeat_epoch must be set when repeat is not None")
        callbacks.append(TrainingPhase(repeat, [repeat_epoch for _ in range(repeat - 1)]))

    if hparams is not None:
        callbacks.append(HParamsMetricLogger(hparams))

    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    return Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=epochs,
        max_time=_max_time,
        logger=run_logger,
        callbacks=callbacks,
        gradient_clip_val=gradient_clip_val,
        enable_progress_bar=progress_bar,
        enable_model_summary=False,
        log_every_n_steps=log_interval,
        reload_dataloaders_every_n_epochs=1 if repeat is not None else 0,
        inference_mode=False,
    )


def train(
    diffusion: "Agedi",
    dataset: Dataset,
    trainer: Optional[Trainer] = None,
    ckpt_path: Optional[Union[str, Path]] = None,
    **trainer_kwargs,
) -> Trainer:
    """Train a diffusion model and return the trainer used.

    Parameters
    ----------
    diffusion:
        The diffusion model to train.
    dataset:
        The dataset to train on.
    trainer:
        A pre-configured Lightning :class:`~lightning.Trainer`.  When
        ``None`` a new trainer is created from *trainer_kwargs*.
    ckpt_path:
        Path to a Lightning checkpoint (``.ckpt``) to resume training from.
        When provided the full training state (model weights, optimiser,
        LR-scheduler, and epoch counter) is restored before fitting.
        Equivalent to passing ``ckpt_path`` to ``trainer.fit()``.
    **trainer_kwargs:
        Additional keyword arguments forwarded to :func:`create_trainer`
        when *trainer* is ``None``.
    """
    # Suppress Lightning's verbose INFO output; our Rich panels provide that context.
    _lightning_logger = logging.getLogger("lightning.pytorch")
    _prev_level = _lightning_logger.level
    _lightning_logger.setLevel(logging.WARNING)
    try:
        current_trainer = trainer or create_trainer(**trainer_kwargs)
        _print_log_path(current_trainer)
        if ckpt_path is not None:
            current_trainer.fit(diffusion, dataset, ckpt_path=str(ckpt_path))
        else:
            current_trainer.fit(diffusion, dataset)
    finally:
        _lightning_logger.setLevel(_prev_level)
    return current_trainer


def train_from_atoms(
    data: Sequence[Atoms],
    *,
    model: str = "PaiNN",
    cutoff: Optional[float] = None,
    feature_size: int = 64,
    n_blocks: int = 4,
    n_rbf: int = 30,
    radial_basis: str = "gaussian",
    noisers: Sequence[str] = ("CellPositions",),
    sde: Union[str, "SDE"] = "ve",
    conditioning: str = "none",
    conditioning_type: str = "scalar",
    mask: str = "none",
    confinement: Optional[Tuple[float, float]] = None,
    force_field: bool = False,
    batch_size: int = 64,
    train_split: Union[float, int] = 0.9,
    val_split: Union[float, int] = 0.1,
    repeat: Optional[int] = None,
    canonical_cell: bool = False,
    lr: float = 1e-4,
    lr_factor: float = 0.95,
    lr_patience: int = 100,
    weight_decay: float = 0.0,
    eps: float = 1e-5,
    guidance_weight: float = -1.0,
    data_path: Optional[str] = None,
    regressor_data: Optional[Sequence[Atoms]] = None,
    checkpoint: Optional[Union[str, Path]] = None,
    trainer: Optional[Trainer] = None,
    n_classes: Optional[int] = None,
    precondition: bool = False,
    sigma_data: float = 1.0,
    prediction_type: str = "score",
    sampler: str = "em",
    fully_connected: bool = False,
    **trainer_kwargs,
) -> Tuple["Agedi", Dataset, Trainer]:
    """Build (or restore), train, and return an AGeDi model from ASE Atoms data.

    When a ``"Types"`` noiser is included and no *checkpoint* is given, the
    unique element types present in *data* are automatically detected and a
    compact type map is built so that the vocabulary size equals the number of
    distinct element types (plus the absorbing state at index 0).  The
    ``n_classes`` parameter can be used to restrict the vocabulary to the
    *n_classes* most frequently occurring element types (sorted by atomic
    number).

    Parameters
    ----------
    data:
        ASE :class:`~ase.Atoms` objects to train on.
    model:
        GNN backbone architecture name.  Looked up in the model registry;
        use :func:`register_model` to add custom backends.  Default:
        ``"PaiNN"`` (SchNetPack PaiNN).
    cutoff:
        Neighbour-list cutoff radius in Å.  Default: ``6.0``.
    feature_size:
        Embedding / feature dimension.  Default: ``64``.
    n_blocks:
        Number of interaction blocks in the GNN backbone.  Default: ``4``.
    n_rbf:
        Number of radial basis functions.  Default: ``30``.
    noisers:
        Sequence of noiser identifiers.  Recognised string identifiers:
        ``"Positions"``, ``"CellPositions"``, ``"ConfinedCellPositions"``,
        ``"Types"`` (snake_case aliases also accepted).
        Default: ``("CellPositions",)``.
    sde:
        SDE for position noisers.  Short aliases: ``"ve"`` (default),
        ``"vp"``.  Pass an instantiated
        :class:`~agedi.diffusion.sdes.SDE` for full control.
    conditioning:
        Per-structure property to condition on (read from
        ``atoms.info[conditioning]`` or ``atoms.get_<conditioning>()``),
        or ``"none"`` for time-only conditioning (default).
    conditioning_type:
        Type of the conditioning module: ``"scalar"`` (default) or
        ``"integer"``.
    mask:
        Atom-masking strategy: ``"MaskFixed"`` (freeze atoms tagged with
        ASE :class:`~ase.constraints.FixAtoms`) or ``"none"`` (default).
    confinement:
        Z-direction confinement bounds ``(z_min, z_max)`` in Å.  Required
        when using the ``"ConfinedCellPositions"`` noiser.
    force_field:
        When ``True``, attach a regressor head (sharing the backbone) that
        predicts per-atom forces and total energy.  Enables force-field
        guided sampling via :class:`~agedi.diffusion.ForcefieldGuidanceConfig`.
        The training data must contain DFT (or other) forces and energy.
        Default: ``False``.
    batch_size:
        Mini-batch size used during training.  Default: ``64``.
    train_split:
        Fraction or absolute count of structures for the training split.
        Default: ``0.9``.
    val_split:
        Fraction or absolute count of structures for the validation split.
        Default: ``0.1``.
    repeat:
        When given, augment the dataset by repeating each structure up to
        ``repeat`` times along the first two cell vectors.  Requires
        ``repeat_epoch`` (passed via ``**trainer_kwargs``) to specify how
        often the repetition level increases.
    canonical_cell:
        Store unit cells in canonical lower-triangular form.  Default:
        ``False``.
    lr:
        Learning rate.  Default: ``1e-4``.
    lr_factor:
        LR-scheduler reduction factor.  Default: ``0.95``.
    lr_patience:
        LR-scheduler patience (epochs).  Default: ``100``.
    weight_decay:
        Optimiser weight decay.  Default: ``0.0``.
    eps:
        Minimum diffusion time value.  Default: ``1e-5``.
    guidance_weight:
        Classifier-free guidance weight.  Default: ``-1.0`` (disabled).
    data_path:
        String path to the training data file; stored in ``hparams.yaml``
        for reference only.  When ``None``, no path metadata is saved.
    regressor_data:
        Optional additional ASE Atoms objects used *exclusively* for
        training the force-field regressor head.  Structures here are never
        passed through the diffusion loss.  Each structure must have an ASE
        calculator with energy and forces attached.
    checkpoint:
        Path to a previously saved run directory (containing ``hparams.yaml``)
        or directly to a ``.ckpt`` checkpoint file.  When provided the model
        architecture and weights are loaded from the checkpoint instead of
        being built from the architecture parameters (*model*, *cutoff*,
        *feature_size*, etc.).  The full training state (optimiser,
        LR-scheduler, epoch counter) is also restored so that training
        continues seamlessly.  Supply *data* to train on new data, or use
        the original data path to resume on the same dataset.
    trainer:
        A pre-configured Lightning :class:`~lightning.Trainer`.  When
        ``None`` (default) a new trainer is built from ``**trainer_kwargs``.
    n_classes:
        Number of element-type classes to use for the
        :class:`~agedi.diffusion.noisers.Types` noiser (not counting the
        absorbing state at index 0).  When ``None`` (default), all distinct
        element types present in *data* are used.  Must not exceed the number
        of distinct types in the training data.  Ignored when *checkpoint* is
        provided (the vocabulary is loaded from the checkpoint).
    **trainer_kwargs:
        Additional keyword arguments forwarded to :func:`create_trainer`
        when *trainer* is ``None``.  Common keys: ``epochs``, ``max_time``,
        ``logger``, ``log_dir``, ``gradient_clip_val``, ``repeat_epoch``.

    Returns
    -------
    Tuple[Agedi, Dataset, Trainer]
        The trained diffusion model, the dataset, and the Lightning trainer.
    """
    from agedi.api._registry import _build_type_map_from_data

    ckpt_file: Optional[str] = None
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        diffusion = load_diffusion(checkpoint_path)
        # Determine the actual .ckpt file for Lightning's ckpt_path so the
        # full training state (optimiser, LR-scheduler, epoch counter) is
        # restored alongside the model weights.
        if checkpoint_path.is_file() and checkpoint_path.suffix == ".ckpt":
            ckpt_file = str(checkpoint_path)
        else:
            ckpt_candidate = checkpoint_path / "checkpoints" / "last_model.ckpt"
            if not ckpt_candidate.exists():
                raise FileNotFoundError(
                    f"No checkpoint file found at '{ckpt_candidate}'. "
                    "Ensure the directory contains 'checkpoints/last_model.ckpt', "
                    "or provide a direct path to a '.ckpt' file."
                )
            ckpt_file = str(ckpt_candidate)
    else:
        # ------------------------------------------------------------------ #
        # Auto-detect type_map when a Types noiser is requested               #
        # ------------------------------------------------------------------ #
        _noiser_names = [
            n if isinstance(n, str) else type(n).__name__ for n in noisers
        ]
        has_types_noiser = any(n in ("Types", "types") for n in _noiser_names)

        type_map: Optional[List[int]] = None
        if has_types_noiser:
            detected_map = _build_type_map_from_data(data)  # [0, z1, z2, ...]
            n_detected = len(detected_map) - 1  # exclude absorbing state

            if n_classes is not None:
                if n_classes > n_detected:
                    raise ValueError(
                        f"n_classes={n_classes} exceeds the number of distinct element "
                        f"types in the training data ({n_detected}).  "
                        f"Types present: {detected_map[1:]}"
                    )
                # Keep only the first n_classes types (sorted by atomic number).
                type_map = [0] + detected_map[1 : n_classes + 1]
            else:
                type_map = detected_map

        diffusion = create_diffusion(
            model=model,
            cutoff=cutoff,
            feature_size=feature_size,
            n_blocks=n_blocks,
            n_rbf=n_rbf,
            radial_basis=radial_basis,
            noisers=noisers,
            sde=sde,
            conditioning=conditioning,
            conditioning_type=conditioning_type,
            confinement=confinement,
            force_field=force_field,
            lr=lr,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            weight_decay=weight_decay,
            eps=eps,
            guidance_weight=guidance_weight,
            type_map=type_map,
            precondition=precondition,
            sigma_data=sigma_data,
            prediction_type=prediction_type,
            sampler=sampler,
        )
        ckpt_file = None

    dataset = create_dataset(
        data,
        cutoff=cutoff,
        batch_size=batch_size,
        train_split=train_split,
        val_split=val_split,
        mask=mask,
        confinement=confinement,
        conditioning=conditioning,
        conditioning_type=conditioning_type,
        repeat=repeat,
        canonical_cell=canonical_cell,
        regressor_data=regressor_data,
        fully_connected=fully_connected,
    )

    n_parameters = sum(
        p.numel() for p in diffusion.score_model.parameters() if p.requires_grad
    )

    hparams = {
        "diffusion": diffusion.get_hparams(),
        # Metadata — not needed for model reconstruction, but useful for display.
        "noisers": list(noisers),
        "sde": sde if isinstance(sde, str) else type(sde).__name__,
        "conditioning": conditioning,
        "conditioning_type": conditioning_type,
        "confinement": list(confinement) if confinement is not None else None,
        "batch_size": batch_size,
        "train_split": train_split,
        "val_split": val_split,
        "n_train": len(dataset.train_idx),
        "n_val": len(dataset.val_idx),
        "mask": mask,
        "canonical_cell": canonical_cell,
        "gradient_clip_val": (
            trainer.gradient_clip_val
            if trainer is not None and hasattr(trainer, "gradient_clip_val")
            else trainer_kwargs.get("gradient_clip_val", 10.0)
        ),
        "n_parameters": n_parameters,
        "repeat": repeat,
        "repeat_epoch": trainer_kwargs.get("repeat_epoch"),
        "data_path": data_path,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
    } | _extract_data_info(list(data))

    _print_training_config(hparams)

    if trainer is None:
        # Clamp log_interval to the actual number of training batches to avoid
        # Lightning's "fewer batches than log_every_n_steps" warning.
        n_train_batches = max(1, math.ceil(len(dataset.train_idx) / batch_size))
        raw_interval = trainer_kwargs.get("log_interval", 10)
        trainer_kwargs["log_interval"] = min(raw_interval, n_train_batches)

        trainer_kwargs.setdefault("repeat", repeat)
        # Pass the full hparams dict so HParamsMetricLogger writes all metadata
        # to hparams.yaml, not just the diffusion architecture config.
        trainer_kwargs.setdefault("hparams", hparams)
    elif hasattr(trainer, "logger") and getattr(trainer, "logger") is not None:
        logger = getattr(trainer, "logger")
        if hasattr(logger, "log_hyperparams"):
            logger.log_hyperparams({"diffusion": diffusion.get_hparams()})

    fit_trainer = train(
        diffusion=diffusion,
        dataset=dataset,
        trainer=trainer,
        ckpt_path=ckpt_file,
        **trainer_kwargs,
    )

    return diffusion, dataset, fit_trainer


def train_from_config(
    config: Union[str, Path, Dict],
) -> Tuple["Agedi", "Dataset", Trainer]:
    """Train an AGeDi model from a YAML configuration file or dictionary.

    This is the *Hydra-style* entry point.  The configuration can be provided
    as:

    * a path to a YAML file (``str`` or :class:`~pathlib.Path`),
    * a plain Python ``dict``,
    * a Hydra / OmegaConf ``DictConfig``.

    The function loads the training data from ``config["data_path"]`` (an
    ASE-readable file) and delegates to :func:`train_from_atoms` with the
    remaining configuration values.

    The minimal required key is ``data_path``.  All other keys are optional
    and fall back to the same defaults as :func:`train_from_atoms`.

    A ready-to-edit template is shipped with the package at
    ``agedi/conf/train.yaml``.

    Parameters
    ----------
    config:
        Configuration source – a YAML file path, a ``dict``, or an OmegaConf
        ``DictConfig``.

    Returns
    -------
    Tuple[Agedi, Dataset, Trainer]
        The trained diffusion model, the dataset used, and the Lightning
        trainer.

    Examples
    --------
    Minimal Python usage::

        from agedi import train_from_config
        diffusion, dataset, trainer = train_from_config("conf/train.yaml")

    Programmatic override::

        from agedi import train_from_config
        cfg = {"data_path": "train.traj", "epochs": 50, "feature_size": 128}
        diffusion, _, _ = train_from_config(cfg)
    """
    import yaml

    # ------------------------------------------------------------------ #
    # 1. Resolve config to a plain dict                                    #
    # ------------------------------------------------------------------ #
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        with open(config_path) as fh:
            cfg: Dict = yaml.safe_load(fh) or {}
    else:
        # Accept OmegaConf DictConfig transparently when hydra is present.
        try:
            from omegaconf import OmegaConf  # type: ignore[import]

            if hasattr(config, "_metadata"):
                cfg = OmegaConf.to_container(config, resolve=True)  # type: ignore[assignment]
            else:
                cfg = dict(config)
        except ImportError:
            cfg = dict(config)

    # ------------------------------------------------------------------ #
    # 2. Validate mandatory key and load data                              #
    # ------------------------------------------------------------------ #
    data_path = cfg.get("data_path")
    if data_path is None:
        raise ValueError(
            "'data_path' is required in the config but was not found. "
            "Set it to the path of your ASE-readable training data file."
        )
    from ase.io import read as ase_read

    data = ase_read(str(data_path), ":")

    # Load optional regressor-only dataset.
    regressor_data = None
    regressor_data_path = cfg.get("regressor_data_path")
    if regressor_data_path is not None:
        regressor_data = ase_read(str(regressor_data_path), ":")

    # ------------------------------------------------------------------ #
    # 3. Split config keys between train_from_atoms and create_trainer    #
    # ------------------------------------------------------------------ #
    train_kwargs: Dict = {k: cfg[k] for k in _TRAIN_FROM_ATOMS_KEYS if k in cfg}
    trainer_kwargs: Dict = {k: cfg[k] for k in _TRAINER_KEYS if k in cfg}

    # Warn about unrecognised keys (but don't error, for forward compat).
    known = _TRAIN_FROM_ATOMS_KEYS | _TRAINER_KEYS | {"data_path", "regressor_data_path"}
    unknown = set(cfg) - known
    if unknown:
        warnings.warn(
            f"train_from_config: unrecognised config keys ignored: {sorted(unknown)}",
            stacklevel=2,
        )

    return train_from_atoms(
        data,
        data_path=str(Path(data_path).resolve()),
        regressor_data=regressor_data,
        **train_kwargs,
        **trainer_kwargs,
    )
