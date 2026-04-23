from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import logging
import math
import os
import time
import warnings

import numpy as np
import torch
import yaml
from ase import Atoms
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from agedi import Diffusion
from agedi.data import AtomsGraph, Dataset
from agedi.data.callbacks import (
    EpochProgressPrinter,
    GradNormLogger,
    HParamsMetricLogger,
    TrainingPhase,
)
from agedi.data.transforms import Repeat
from agedi.diffusion import ForcefieldGuidanceConfig
from agedi.models import ScoreModel


# ---------------------------------------------------------------------------
# Private helpers (model/data construction utilities)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model backend registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, "Callable"] = {}  # type: ignore[type-arg]


def register_model(name: str, factory: "Callable") -> None:  # type: ignore[type-arg]
    """Register a custom score model backbone factory under *name*.

    The factory is called with the keyword arguments ``cutoff``,
    ``heads``, ``feature_size``, ``n_blocks``, ``head_dim``, and ``n_rbf``
    and must return a 3-tuple ``(translator, representation, List[Head])``.

    Registered models can be selected by passing ``model=name`` to
    :func:`create_diffusion`.

    Parameters
    ----------
    name : str
        Alias used to select this backend (e.g. ``"PaiNN"``).
    factory : Callable
        Factory function with signature::

            factory(cutoff, heads, feature_size, n_blocks, head_dim, n_rbf)
                -> Tuple[Translator, nn.Module, List[Head]]

    Examples
    --------
    ::

        from agedi.functional import register_model

        def my_factory(cutoff, heads, feature_size, n_blocks, head_dim, n_rbf):
            ...
            return translator, representation, head_list

        register_model("MyModel", my_factory)
    """
    _MODEL_REGISTRY[name] = factory



def _resolve_sde(alias: Union[str, "SDE"]) -> "SDE":
    """Resolve an SDE alias string to an :class:`~agedi.diffusion.sdes.SDE` instance.

    Parameters
    ----------
    alias : str or SDE
        A short alias string or an already-instantiated SDE object.
        Recognised aliases are ``"ve"`` and ``"vp"``.

    Returns
    -------
    SDE
        The resolved SDE instance.
    """
    from agedi.diffusion.sdes import SDE, VE, VP

    if isinstance(alias, SDE):
        return alias

    _SDE_ALIASES = {
        "ve": VE,
        "vp": VP,
    }
    if alias not in _SDE_ALIASES:
        raise ValueError(
            f"Unknown SDE alias '{alias}'. "
            f"Valid aliases are: {sorted(_SDE_ALIASES)}"
        )
    return _SDE_ALIASES[alias]()


def _build_noisers(
    noisers: Sequence[Union[str, "Noiser"]],
    sde: Union[str, "SDE"] = "ve",
) -> List["Noiser"]:
    """Build a list of Noiser objects from a sequence of noiser names or objects.

    Parameters
    ----------
    noisers : Sequence[Union[str, Noiser]]
        A sequence of noiser identifiers or already-instantiated
        :class:`~agedi.diffusion.noisers.Noiser` objects.  String
        identifiers are resolved via the noiser registry (see
        :meth:`~agedi.diffusion.noisers.Noiser.register`).  Built-in
        identifiers (CamelCase preferred; snake_case aliases also accepted):

        * ``"Positions"`` – :class:`~agedi.diffusion.noisers.Positions`
          (StandardNormal prior + Normal distribution, for clusters).
        * ``"CellPositions"`` – :class:`~agedi.diffusion.noisers.CellPositions`
          (UniformCell prior + Normal distribution, for periodic systems).
        * ``"ConfinedCellPositions"`` –
          :class:`~agedi.diffusion.noisers.ConfinedCellPositions`
          (UniformCellConfined prior + TruncatedNormal distribution, for
          Z-confined surfaces/porous materials).
        * ``"Types"`` – :class:`~agedi.diffusion.noisers.Types`.

    sde : str or SDE, optional
        Stochastic differential equation to use for position noisers.  Either a
        short alias (``"ve"``, ``"vp"``) or an already-instantiated
        :class:`~agedi.diffusion.sdes.SDE` object.  Defaults to ``"ve"``.

    Returns
    -------
    List[Noiser]
        Instantiated noisers in the same order as *noisers*.
    """
    from agedi.diffusion.noisers import Noiser

    resolved_sde = _resolve_sde(sde)
    noiser_list = []
    for noiser in noisers:
        if isinstance(noiser, Noiser):
            noiser_list.append(noiser)
            continue
        if noiser not in Noiser._registry:
            raise ValueError(
                f"Unknown noiser '{noiser}'. "
                f"Available built-in noisers: {sorted(Noiser._registry)}. "
                "Use Noiser.register() to add a custom noiser."
            )
        noiser_list.append(Noiser._registry[noiser](sde=resolved_sde))

    return noiser_list


def _build_conditioning(condition: str, type: Optional[str] = None) -> List["Conditioning"]:
    """Build a list of conditioning modules.

    Always includes a :class:`~agedi.models.conditionings.TimeConditioning`.
    When *condition* is not ``"none"``, an additional property-conditioning
    module is appended.

    Parameters
    ----------
    condition : str
        Name of the property to condition on, or ``"none"`` for
        time-only conditioning.
    type : str, optional
        Type of the conditioning module: ``"scalar"`` or ``"integer"``.
        Required when *condition* is not ``"none"``.

    Returns
    -------
    List[Conditioning]
        The list of conditioning modules.
    """
    from agedi.models.conditionings import TimeConditioning

    conditioning = [TimeConditioning()]

    if condition != "none":
        from agedi.models.conditionings import ScalarConditioning, IntegerConditioning

        if type == "scalar":
            conditioning.append(ScalarConditioning(property=condition))
        elif type == "integer":
            conditioning.append(IntegerConditioning(property=condition))
        else:
            raise ValueError(f"Unknown conditioning type '{type}'")

    return conditioning


def _build_score_components(model: str, cutoff: float, heads: Sequence[str], feature_size: int, n_blocks: int, head_dim: int, n_rbf: int = 30) -> Tuple["Translator", "torch.nn.Module", List["Head"]]:
    """Instantiate the translator, representation, and score heads for a model.

    Parameters
    ----------
    model : str
        Name of the GNN backbone.  The name is looked up in the model
        registry populated via :func:`register_model`.  Use ``"PaiNN"``
        for the built-in SchNetPack PaiNN backend.
    cutoff : float
        Cutoff radius (Å) for the neighbour list.
    heads : Sequence[str]
        Names of score heads to build (``"positions"``, ``"types"``).
    feature_size : int
        Embedding/feature dimension for the backbone.
    n_blocks : int
        Number of interaction blocks in the backbone.
    head_dim : int
        Input dimension for each score head (typically
        ``feature_size + conditioning output dims``).
    n_rbf : int, optional
        Number of radial basis functions.  Default is 30.

    Returns
    -------
    Tuple[Translator, nn.Module, List[Head]]
        A 3-tuple of the translator, the representation backbone, and the list
        of score-head modules.
    """
    if model not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model}'. "
            f"Available built-in models: {sorted(_MODEL_REGISTRY)}. "
            "Use register_model() to add a custom backend."
        )
    return _MODEL_REGISTRY[model](
        cutoff=cutoff,
        heads=heads,
        feature_size=feature_size,
        n_blocks=n_blocks,
        head_dim=head_dim,
        n_rbf=n_rbf,
    )


def _painn_factory(cutoff: float, heads: Sequence[str], feature_size: int, n_blocks: int, head_dim: int, n_rbf: int) -> Tuple["Translator", "torch.nn.Module", List["Head"]]:
    """Factory for the SchNetPack PaiNN score model backend."""
    import schnetpack as spk
    from agedi.models.schnetpack import (
        PositionsScore,
        TypesScore,
        SchNetPackTranslator,
    )

    translator = SchNetPackTranslator(
        input_modules=[spk.atomistic.PairwiseDistances()]
    )
    representation = spk.representation.PaiNN(
        n_atom_basis=feature_size,
        n_interactions=n_blocks,
        radial_basis=spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff),
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )

    h = []
    for head in heads:
        match head:
            case (
                "Positions"
                | "CellPositions"
                | "ConfinedCellPositions"
                | "positions"
                | "cell_positions"
                | "confined_cell_positions"
            ):
                h.append(PositionsScore(input_dim_scalar=head_dim))
            case "Types" | "types":
                h.append(TypesScore(input_dim_scalar=head_dim))
            case _ if hasattr(head, "_key") and head._key == "positions":
                h.append(PositionsScore(input_dim_scalar=head_dim))
            case _ if hasattr(head, "_key") and head._key == "x":
                n_classes = getattr(head, "n_classes", 100)
                h.append(TypesScore(input_dim_scalar=head_dim, n_classes=n_classes))
            case _:
                raise ValueError(f"Unknown head '{head}'")

    return translator, representation, h


register_model("PaiNN", _painn_factory)


def _build_regressor(
    translator: "Translator",
    representation: "Representation",
    feature_size: int,
) -> "RegressorModel":
    """Build a :class:`~agedi.models.regressor.RegressorModel` with an Energy and a Forces head.

    The force field regressor **shares** the ``translator`` and ``representation``
    from the score model so that the atomic embeddings are learned jointly with
    the diffusion score.  Only the :class:`~agedi.models.schnetpack.regressor_heads.Forces`
    and :class:`~agedi.models.schnetpack.regressor_heads.Energy` heads are added on top of the shared representation.

    The resulting model is attached to the :class:`~agedi.Diffusion` object as
    ``regressor_model`` so that force-field guidance can be used during
    sampling (see :class:`~agedi.diffusion.ForcefieldGuidanceConfig`).

    Parameters
    ----------
    translator : Translator
        The translator from the score model (shared, not copied).
    representation : Representation
        The representation from the score model (shared, not copied).
    feature_size : int
        Embedding/feature dimension of the shared representation.

    Returns
    -------
    RegressorModel
        An initialised force-regression model (not yet trained).
    """
    from agedi.models.regressor import RegressorModel
    from agedi.models.schnetpack.regressor_heads import Energy, Forces

    energy_head = Energy(input_dim_scalar=feature_size)
    forces_head = Forces(input_dim_scalar=feature_size, input_dim_vector=feature_size)
    return RegressorModel(
        translator=translator,
        representation=representation,
        heads=[energy_head, forces_head],
    )


def _extract_data_info(data: Sequence[Atoms]) -> Dict:
    """Extract summary information from a list of ASE Atoms objects.

    Parameters
    ----------
    data : Sequence[Atoms]
        List of ASE :class:`~ase.Atoms` objects to inspect.

    Returns
    -------
    dict
        A dictionary with the following keys:

        * ``"cell"`` – flattened 9-element list of the (shared) unit-cell
          matrix, or ``None`` if cells differ between structures.
        * ``"symbols"`` – list of unique chemical symbols present in the data.
        * ``"n_training_data"`` – total number of structures.
    """
    elements = set()
    out = {"cell": None}
    check_cell = True
    for d in data:
        elements.update(d.get_chemical_symbols())
        if check_cell:
            if d.cell is not None:
                out["cell"] = np.array(d.cell)
            else:
                if not np.all(out["cell"] == d.cell):
                    check_cell = False
    out["cell"] = out["cell"].flatten().tolist() if out["cell"] is not None else None
    out |= {
        "symbols": list(elements),
        "n_training_data": len(data),
    }
    return out


def _fmt_target(target: str) -> str:
    """Return the short class name from a dotted ``_target_`` string."""
    return target.rsplit(".", 1)[-1] if "." in target else target


def _check_head_dimensions(diffusion_cfg: dict) -> list:
    """Validate that head input dimensions match feature_size + conditioning output dims.

    Parameters
    ----------
    diffusion_cfg : dict
        The ``diffusion`` sub-dict from ``hparams.yaml``.

    Returns
    -------
    list of str
        Warning messages for any dimension mismatches, empty if all match.
    """
    warnings = []
    score_model = diffusion_cfg.get("score_model", {})
    representation = score_model.get("representation", {})
    feature_size = representation.get("n_atom_basis")
    if feature_size is None:
        return warnings

    conditionings = score_model.get("conditionings", [])
    cond_output_dims = [
        c.get("output_dim")
        for c in conditionings
        if isinstance(c, dict)
    ]
    if any(d is None for d in cond_output_dims):
        # One or more conditionings are missing output_dim (e.g. older
        # hparams.yaml written before output_dim was added to get_hparams).
        # Skip the check to avoid false-positive dimension mismatch warnings.
        return warnings
    cond_dim = sum(cond_output_dims)  # type: ignore[arg-type]
    expected_dim = feature_size + cond_dim

    heads = score_model.get("heads", [])
    for head in heads:
        if not isinstance(head, dict):
            continue
        head_name = _fmt_target(head.get("_target_", "head"))
        actual_dim = head.get("input_dim_scalar")
        if actual_dim is not None and actual_dim != expected_dim:
            warnings.append(
                f"[yellow]⚠ {head_name}: input_dim_scalar={actual_dim} "
                f"but feature_size({feature_size}) + conditioning_dims({cond_dim}) "
                f"= {expected_dim}[/yellow]"
            )
    return warnings


def _render_config_tree(data, tree: Tree) -> None:  # type: ignore[type-arg]
    """Recursively populate a Rich :class:`~rich.tree.Tree` with config key/values.

    * Dicts: each key becomes a leaf (scalar) or sub-branch (dict/list).
      The ``_target_`` key is skipped — callers show it as the branch label.
    * Lists: each item becomes a sub-branch labelled by its ``_target_`` class
      name (or a numeric index for plain scalars).
    * Scalars and ``None``: shown as ``key  value``; ``None`` is omitted.
    """
    if isinstance(data, dict):
        for key, val in data.items():
            if key == "_target_":
                continue
            if isinstance(val, dict):
                cls = _fmt_target(val.get("_target_", ""))
                label = f"[bold cyan]{key}[/bold cyan]" + (
                    f"  [dim green]{cls}[/dim green]" if cls else ""
                )
                branch = tree.add(label)
                _render_config_tree(val, branch)
            elif isinstance(val, list):
                branch = tree.add(f"[bold cyan]{key}[/bold cyan]")
                _render_config_tree(val, branch)
            elif val is None:
                pass  # omit null values
            else:
                tree.add(f"[cyan]{key}[/cyan]  [white]{val}[/white]")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                target = item.get("_target_", "")
                label = (
                    f"[magenta]{_fmt_target(target)}[/magenta]"
                    if target
                    else f"[white][{i}][/white]"
                )
                branch = tree.add(label)
                _render_config_tree(
                    {k: v for k, v in item.items() if k != "_target_"}, branch
                )
            elif item is not None:
                tree.add(f"[white]{item}[/white]")
    elif data is not None:
        tree.add(f"[white]{data}[/white]")


def _extract_diffusion_display_info(diffusion_cfg: dict) -> dict:
    """Extract human-readable display values from a nested Hydra diffusion config.

    Parameters
    ----------
    diffusion_cfg : dict
        The ``diffusion`` sub-dict from ``hparams.yaml``, as returned by
        :meth:`~agedi.diffusion.diffusion.Diffusion.get_hparams`.

    Returns
    -------
    dict
        Flat dict with display-friendly keys: ``model``, ``feature_size``,
        ``n_blocks``, ``cutoff``, ``noisers``, ``conditionings``, ``lr``,
        ``lr_factor``, ``lr_patience``, ``weight_decay``.
    """
    info: dict = {}
    score_cfg = diffusion_cfg.get("score_model", {})
    rep_cfg = score_cfg.get("representation", {})

    # Representation info
    rep_target = rep_cfg.get("_target_", "")
    info["model"] = rep_target.rsplit(".", 1)[-1] if rep_target else ""
    info["feature_size"] = rep_cfg.get("n_atom_basis", "")
    info["n_blocks"] = rep_cfg.get("n_interactions", "")
    cutoff_fn = rep_cfg.get("cutoff_fn", {})
    info["cutoff"] = cutoff_fn.get("cutoff", rep_cfg.get("cutoff", ""))

    # Noiser names (last component of _target_)
    noiser_cfgs = diffusion_cfg.get("noisers", [])
    info["noisers"] = [
        cfg.get("_target_", "").rsplit(".", 1)[-1] for cfg in noiser_cfgs
    ]

    # Conditioning names (skip TimeConditioning for brevity)
    cond_cfgs = score_cfg.get("conditionings", [])
    non_time = [
        cfg.get("_target_", "").rsplit(".", 1)[-1]
        for cfg in cond_cfgs
        if "Time" not in cfg.get("_target_", "")
    ]
    info["conditionings"] = non_time

    # Optimizer
    optim = diffusion_cfg.get("optim_config", {})
    info["lr"] = optim.get("lr", "")
    info["weight_decay"] = optim.get("weight_decay", 0.0)
    sched = diffusion_cfg.get("scheduler_config", {})
    info["lr_factor"] = sched.get("factor", "")
    info["lr_patience"] = sched.get("patience", "")

    return info


def _get_device_info() -> str:
    """Return a human-readable string describing the available compute device."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        n = torch.cuda.device_count()
        return f"CUDA – {name}" + (f" ×{n}" if n > 1 else "")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    return "CPU"


def _print_training_config(hparams: dict) -> None:
    """Print a Rich-formatted training configuration panel.

    Parameters
    ----------
    hparams : dict
        The hparams dict as saved to ``hparams.yaml``.  Must contain a
        ``"diffusion"`` key with the nested Hydra config.  Metadata keys
        (``n_train``, ``n_val``, ``batch_size``, etc.) are read directly.
    """
    diffusion_cfg = hparams.get("diffusion", {})

    console = Console()

    # ── Metadata table (dataset / trainer / hardware) ─────────────────────
    meta = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    meta.add_column("Key", style="bold cyan", min_width=22, no_wrap=True)
    meta.add_column("Value", style="white")

    meta.add_row("[bold]Dataset[/bold]", "")
    if hparams.get("data_path"):
        meta.add_row("  data", str(hparams["data_path"]))
    meta.add_row("  n_train", str(hparams.get("n_train", "")))
    meta.add_row("  n_val", str(hparams.get("n_val", "")))
    meta.add_row("  batch_size", str(hparams.get("batch_size", "")))
    mask = hparams.get("mask", "none")
    if mask and mask != "none":
        meta.add_row("  mask", str(mask))
    repeat = hparams.get("repeat")
    if repeat is not None:
        meta.add_row("  repeat", str(repeat))
        meta.add_row("  repeat_epoch", str(hparams.get("repeat_epoch", "")))

    meta.add_row("", "")
    meta.add_row("[bold]Trainer[/bold]", "")
    meta.add_row("  gradient_clip_val", str(hparams.get("gradient_clip_val", "")))
    n_parameters = hparams.get("n_parameters")
    if n_parameters is not None:
        meta.add_row("  parameters", f"{n_parameters:,}")

    meta.add_row("", "")
    meta.add_row("[bold]Hardware[/bold]", "")
    meta.add_row("  device", _get_device_info())

    console.print(
        Panel(meta, title="[bold]AGeDi Training — Run Configuration[/bold]", border_style="blue")
    )

    # ── Full model-architecture tree ───────────────────────────────────────
    arch_tree = Tree("[bold]Diffusion[/bold]")
    _render_config_tree(diffusion_cfg, arch_tree)
    console.print(
        Panel(arch_tree, title="[bold]AGeDi Training — Model Architecture[/bold]", border_style="cyan")
    )

    # ── Dimension validation ───────────────────────────────────────────────
    dim_warnings = _check_head_dimensions(diffusion_cfg)
    for w in dim_warnings:
        console.print(w)


def _print_log_path(trainer: Trainer) -> None:
    """Print the resolved log directory for this training run."""
    try:
        log_dir = trainer.logger.log_dir  # type: ignore[union-attr]
    except AttributeError:
        return
    if log_dir:
        Console().print(f"[bold cyan]Log dir:[/bold cyan] {log_dir}")


def _print_loaded_model_info(params: dict, checkpoint_path: Path, device) -> None:
    """Print a Rich-formatted summary of a loaded diffusion model."""
    diffusion_cfg = params.get("diffusion", {})

    console = Console()

    # ── Checkpoint / device summary ────────────────────────────────────────
    ckpt_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    ckpt_table.add_column("Key", style="bold cyan", min_width=20, no_wrap=True)
    ckpt_table.add_column("Value", style="white")
    ckpt_table.add_row("  path", str(checkpoint_path))
    ckpt_table.add_row("  device", str(device))

    console.print(
        Panel(ckpt_table, title="[bold]AGeDi Model Loaded[/bold]", border_style="green")
    )

    # ── Full model-architecture tree ───────────────────────────────────────
    arch_tree = Tree("[bold]Diffusion[/bold]")
    _render_config_tree(diffusion_cfg, arch_tree)
    console.print(
        Panel(arch_tree, title="[bold]AGeDi Model Architecture[/bold]", border_style="cyan")
    )

    # ── Dimension validation ───────────────────────────────────────────────
    dim_warnings = _check_head_dimensions(diffusion_cfg)
    for w in dim_warnings:
        console.print(w)


def _print_sampling_config(
    n_samples: int,
    steps: int,
    eps: float,
    batch_size: int,
    formula=None,
    n_atoms=None,
    template=None,
    cell=None,
    confinement=None,
    property=None,
    force_field_guidance: float = 0.0,
) -> None:
    """Print a Rich-formatted sampling configuration panel."""
    console = Console()
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Key", style="bold cyan", min_width=14, no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("  n_samples", str(n_samples))
    table.add_row("  steps", str(steps))
    table.add_row("  eps", str(eps))
    table.add_row("  batch_size", str(batch_size))
    if formula is not None:
        table.add_row("  formula", str(formula))
    elif n_atoms is not None:
        table.add_row("  n_atoms", str(n_atoms))
    if template is not None:
        table.add_row("  template", "provided")
    if cell is not None:
        table.add_row("  cell", "provided")
    if confinement is not None:
        table.add_row("  confinement", f"{confinement[0]} – {confinement[1]} Å")
    if property is not None:
        for k, v in property.items():
            table.add_row(f"  {k}", str(v))
    if force_field_guidance > 0.0:
        table.add_row("  ff_guidance", str(force_field_guidance))

    console.print(
        Panel(table, title="[bold]AGeDi Sampling Configuration[/bold]", border_style="blue")
    )


def create_diffusion(
    model: str = "PaiNN",
    cutoff: float = 6.0,
    feature_size: int = 64,
    n_blocks: int = 4,
    n_rbf: int = 30,
    noisers: Sequence[Union[str, "Noiser"]] = ("CellPositions",),
    sde: Union[str, "SDE"] = "ve",
    conditioning: str = "none",
    conditioning_type: str = "scalar",
    confinement: Optional[Tuple[float, float]] = None,
    force_field: bool = False,
    lr: float = 1e-4,
    lr_factor: float = 0.95,
    lr_patience: int = 100,
    weight_decay: float = 0.0,
    eps: float = 1e-5,
    guidance_weight: float = -1.0,
    device: Optional[Union[str, torch.device]] = None,
) -> Diffusion:
    """Create a diffusion model for script-based training and sampling.

    Parameters
    ----------
    model : str, optional
        GNN backbone architecture.  The name is looked up in the model
        registry; use :func:`register_model` to add custom backends.
        The built-in default is ``"PaiNN"`` (SchNetPack PaiNN).
    cutoff : float, optional
        Neighbour-list cutoff radius in Å.  Defaults to ``6.0``.
    feature_size : int, optional
        Embedding / feature dimension.  Defaults to ``64``.
    n_blocks : int, optional
        Number of interaction blocks.  Defaults to ``4``.
    n_rbf : int, optional
        Number of radial basis functions.  Defaults to ``30``.
    noisers : Sequence[str or Noiser], optional
        Noiser identifiers or instances to include.  Defaults to
        ``("CellPositions",)``.  Recognised string identifiers (CamelCase
        preferred; snake_case aliases also accepted for backwards compatibility):

        * ``"Positions"`` / ``"positions"`` – :class:`~agedi.diffusion.noisers.Positions`
          (StandardNormal prior + Normal, for gas-phase clusters).
        * ``"CellPositions"`` / ``"cell_positions"`` – :class:`~agedi.diffusion.noisers.CellPositions`
          (UniformCell prior + Normal, for periodic bulk/surface systems).
        * ``"ConfinedCellPositions"`` / ``"confined_cell_positions"`` –
          :class:`~agedi.diffusion.noisers.ConfinedCellPositions`
          (UniformCellConfined prior + TruncatedNormal, for Z-confined systems).
        * ``"Types"`` / ``"types"`` – :class:`~agedi.diffusion.noisers.Types`.

    sde : str or SDE, optional
        SDE for position noisers.  Short aliases: ``"ve"`` (default),
        ``"vp"``.  Pass an instantiated
        :class:`~agedi.diffusion.sdes.SDE` for full control.
    conditioning : str, optional
        Property to condition on, or ``"none"`` for time-only
        conditioning.  Defaults to ``"none"``.
    conditioning_type : str, optional
        Type of the conditioning module: ``"scalar"`` or ``"integer"``.
        Defaults to ``"scalar"``.
    confinement : Tuple[float, float], optional
        Z-direction confinement bounds ``(z_min, z_max)`` in Å.
    force_field : bool, optional
        When ``True``, attach a ``diffusion.regressor_model``.  The heads **shares** the
        same representation and translator as the score model so that atomic
        embeddings are learned jointly.  It is trained whenever the training
        batch contains per-atom forces and total energies (i.e. the ASE training structures have
        DFT (or other) energy and forces).  The trained forces head enables force-field guided
        sampling via :class:`~agedi.diffusion.ForcefieldGuidanceConfig`.
        Defaults to ``False``.
    lr : float, optional
        Learning rate.  Defaults to ``1e-4``.
    lr_factor : float, optional
        LR-scheduler reduction factor.  Defaults to ``0.95``.
    lr_patience : int, optional
        LR-scheduler patience (epochs).  Defaults to ``100``.
    weight_decay : float, optional
        Optimizer weight-decay.  Defaults to ``0.0``.
    eps : float, optional
        Minimum diffusion time.  Defaults to ``1e-5``.
    guidance_weight : float, optional
        Classifier-free guidance weight.  Defaults to ``-1.0`` (disabled).
    device : str or torch.device, optional
        Target compute device.  When ``None`` CUDA is used if available,
        otherwise CPU.

    Returns
    -------
    Diffusion
        A freshly initialised :class:`~agedi.Diffusion` model.
    """

    torch_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    conditioning_modules = _build_conditioning(conditioning, type=conditioning_type)
    head_dim = feature_size + sum(module.output_dim for module in conditioning_modules)

    translator, representation, heads = _build_score_components(
        model,
        cutoff,
        noisers,
        feature_size,
        n_blocks,
        head_dim=head_dim,
        n_rbf=n_rbf,
    )

    noiser_modules = _build_noisers(noisers, sde=sde)

    score_model = ScoreModel(
        translator=translator,
        representation=representation,
        conditionings=conditioning_modules,
        heads=list(heads),
        w=guidance_weight,
    )

    regressor_model = None
    if force_field:
        regressor_model = _build_regressor(
            translator=translator,
            representation=representation,
            feature_size=feature_size,
        )

    return Diffusion(
        score_model=score_model,
        noisers=noiser_modules,
        regressor_model=regressor_model,
        optim_config={"lr": lr, "weight_decay": weight_decay},
        scheduler_config={"factor": lr_factor, "patience": lr_patience},
        eps=eps,
    ).to(torch_device)


def create_dataset(
    data: Sequence[Atoms],
    cutoff: float = 6.0,
    batch_size: int = 64,
    train_split: Union[float, int] = 0.9,
    val_split: Union[float, int] = 0.1,
    mask: str = "none",
    confinement: Optional[Tuple[float, float]] = None,
    conditioning: str = "none",
    conditioning_type: str = "scalar",
    repeat: Optional[int] = None,
    canonical_cell: bool = False,
) -> Dataset:
    """Create and setup an AGeDi Dataset from ASE Atoms objects."""
    phase_transforms = None
    if repeat is not None:
        if repeat < 2:
            raise ValueError(f"repeat must be at least 2, got {repeat}")

        property_kinds = {"mask": "node"}
        if confinement is not None:
            property_kinds["confinement"] = "none"
        if conditioning != "none":
            property_kinds[conditioning] = (
                "node" if conditioning_type == "node" else "none"
            )
        phase_transforms = [[]]
        for i in range(2, repeat + 1):
            phase_transforms.append([Repeat((i, i, 1), property=property_kinds)])

    dataset = Dataset(
        cutoff=cutoff,
        batch_size=batch_size,
        n_train=train_split,
        n_val=val_split,
        phase_transforms=phase_transforms,
        num_workers=min(4, os.cpu_count() or 1),
    )

    properties = None
    if conditioning != "none":
        properties = []
        for sample in data:
            value = None
            try:
                value = getattr(sample, f"get_{conditioning}")()
            except AttributeError:
                pass

            try:
                value = sample.info[conditioning]
            except KeyError:
                pass

            if value is None:
                value = 0
                warnings.warn(
                    f"Conditioning '{conditioning}' not found for one sample; using 0.",
                    stacklevel=2,
                )

            properties.append({conditioning: value})

    dataset.add_atoms_data(
        list(data),
        mask_method=mask,
        confinement=confinement,
        properties=properties,
        canonical_cell=canonical_cell,
    )
    dataset.setup()
    return dataset


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
    devices:
        Number of devices to train on.
    print_epoch_interval:
        Print a one-line training summary to stdout every this many epochs.
        Set to ``0`` to disable (default: 10).
    log_grad_norm:
        Whether to log the total gradient norm during training (default: ``True``).
        Disable for large models where the per-step overhead is undesirable.
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
        run_logger = TensorBoardLogger(save_dir=log_dir, name="")
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
    diffusion: Diffusion,
    dataset: Dataset,
    trainer: Optional[Trainer] = None,
    **trainer_kwargs,
) -> Trainer:
    """Train a diffusion model and return the trainer used."""
    # Suppress Lightning's verbose INFO output; our Rich panels provide that context.
    _lightning_logger = logging.getLogger("lightning.pytorch")
    _prev_level = _lightning_logger.level
    _lightning_logger.setLevel(logging.WARNING)
    try:
        current_trainer = trainer or create_trainer(**trainer_kwargs)
        _print_log_path(current_trainer)
        current_trainer.fit(diffusion, dataset)
    finally:
        _lightning_logger.setLevel(_prev_level)
    return current_trainer


def sample(
    diffusion: Diffusion,
    *,
    n_samples: int,
    n_atoms: Optional[int] = None,
    atomic_numbers: Optional[List[int]] = None,
    formula: Optional[str] = None,
    positions: Optional[np.ndarray] = None,
    cell: Optional[np.ndarray] = None,
    template: Optional[AtomsGraph] = None,
    confinement: Optional[Tuple[float, float]] = None,
    steps: int = 500,
    eps: float = 1e-3,
    batch_size: int = 64,
    ff_guidance: Optional[ForcefieldGuidanceConfig] = None,
    property: Optional[Dict[str, float]] = None,
    progress_bar: bool = False,
    save_trajectory: bool = False,
    save_path: Optional[bool] = None,
    as_atoms: bool = True,
) -> Union[List[AtomsGraph], List[Atoms], List[List[AtomsGraph]], List[List[Atoms]]]:
    """Sample structures from a trained diffusion model.

    Parameters
    ----------
    diffusion:
        A trained :class:`~agedi.Diffusion` model.
    n_samples:
        Number of structures to generate.
    n_atoms:
        Number of atoms per structure. Automatically determined from
        ``formula`` if provided, or from the length of ``atomic_numbers``
        when ``n_atoms`` is not explicitly given.
    atomic_numbers:
        Atomic numbers of the generated atoms.  Not required when the model
        has a types-noiser or when ``formula`` is provided.
    formula:
        Chemical formula (e.g. ``"H2O"``).  Used to derive ``n_atoms`` and
        ``atomic_numbers`` when they are not provided explicitly.
    positions:
        Fixed positions of the atoms (shape ``(n_atoms, 3)``).  Required
        when no positions-noiser is configured (type-only diffusion).
        Positions will not be modified during sampling.
    cell:
        Unit-cell matrix (3×3 array or flat length-9 array).  Not required
        when ``template`` is provided (the template's cell is used instead).
    template:
        Template :class:`~agedi.AtomsGraph`.  When given, ``cell`` and
        ``pbc`` are taken from the template unless explicitly provided.
    ff_guidance:
        Force-field guidance configuration.  When ``None`` (default) a
        :class:`~agedi.diffusion.ForcefieldGuidanceConfig` with default
        values is used (i.e. guidance is disabled).
    """
    if save_path is not None:
        warnings.warn(
            "'save_path' is deprecated; use 'save_trajectory' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        save_trajectory = save_path

    _ff = ff_guidance if ff_guidance is not None else ForcefieldGuidanceConfig()

    _print_sampling_config(
        n_samples=n_samples,
        steps=steps,
        eps=eps,
        batch_size=batch_size,
        formula=formula,
        n_atoms=n_atoms,
        template=template,
        cell=cell,
        confinement=confinement,
        property=property,
        force_field_guidance=_ff.guidance,
    )

    _start = time.monotonic()

    diffusion.eval()
    with torch.no_grad():
        sampled = diffusion.sample(
            N=n_samples,
            template=template,
            batch_size=batch_size,
            steps=steps,
            eps=eps,
            n_atoms=n_atoms,
            atomic_numbers=atomic_numbers,
            formula=formula,
            positions=positions,
            cell=cell,
            confinement=confinement,
            ff_guidance=_ff,
            property=property,
            progress_bar=progress_bar,
            save_path=save_trajectory,
        )

    elapsed = time.monotonic() - _start
    n_generated = len(sampled)
    Console().print(f"[green]✓[/green] Generated {n_generated} structure(s) in {elapsed:.1f}s")

    if not as_atoms:
        return sampled

    if save_trajectory:
        return [[graph.to_atoms() for graph in trajectory] for trajectory in sampled]
    return [graph.to_atoms() for graph in sampled]


def load_diffusion(
    path: Union[str, Path],
    checkpoint: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Diffusion:
    """Load a trained diffusion model from an AGeDi log directory.

    The model architecture is fully reconstructed from the Hydra-compatible
    ``diffusion`` config stored in ``hparams.yaml``, so no additional
    parameters are needed.

    Parameters
    ----------
    path:
        Path to the AGeDi log / model directory (or directly to the
        ``hparams.yaml`` file).
    checkpoint:
        Path to a specific checkpoint file.  When ``None`` the latest
        checkpoint (``checkpoints/last_model.ckpt``) is loaded automatically.
    device:
        Device to load the model onto.  When ``None`` CUDA is used if
        available, otherwise CPU.
    """
    from hydra.utils import instantiate as hydra_instantiate

    root_path = Path(path)
    if root_path.is_file():
        root_path = root_path.parent.parent

    params_path = root_path / "hparams.yaml"
    if not params_path.exists():
        raise FileNotFoundError(f"Could not find hparams file: {params_path}")

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    if "diffusion" not in params:
        raise ValueError(
            f"hparams.yaml at '{params_path}' does not contain a 'diffusion' key. "
            "Only the current Hydra-based format is supported."
        )

    current_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    diffusion = hydra_instantiate(params["diffusion"], _convert_="all")
    diffusion = diffusion.to(current_device)

    checkpoint_path = (
        Path(checkpoint)
        if checkpoint is not None
        else root_path / "checkpoints" / "last_model.ckpt"
    )
    checkpoint_data = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location=current_device,
    )
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    diffusion.load_state_dict(state_dict)
    diffusion.eval()
    _print_loaded_model_info(params, checkpoint_path, current_device)
    return diffusion


def train_from_atoms(
    data: Sequence[Atoms],
    *,
    model: str = "PaiNN",
    cutoff: float = 6.0,
    feature_size: int = 64,
    n_blocks: int = 4,
    n_rbf: int = 30,
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
    trainer: Optional[Trainer] = None,
    **trainer_kwargs,
) -> Tuple[Diffusion, Dataset, Trainer]:
    """Build, train, and return an AGeDi model from ASE Atoms data."""
    diffusion = create_diffusion(
        model=model,
        cutoff=cutoff,
        feature_size=feature_size,
        n_blocks=n_blocks,
        n_rbf=n_rbf,
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
    )
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
        **trainer_kwargs,
    )

    return diffusion, dataset, fit_trainer


# ---------------------------------------------------------------------------
# Hydra / config-file based training entry point
# ---------------------------------------------------------------------------

#: Parameters forwarded to :func:`train_from_atoms` from the config dict.
_TRAIN_FROM_ATOMS_KEYS = frozenset(
    [
        "model",
        "cutoff",
        "feature_size",
        "n_blocks",
        "n_rbf",
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


def train_from_config(
    config: Union[str, Path, Dict],
) -> Tuple[Diffusion, "Dataset", Trainer]:
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
    Tuple[Diffusion, Dataset, Trainer]
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

    # ------------------------------------------------------------------ #
    # 3. Split config keys between train_from_atoms and create_trainer    #
    # ------------------------------------------------------------------ #
    train_kwargs: Dict = {k: cfg[k] for k in _TRAIN_FROM_ATOMS_KEYS if k in cfg}
    trainer_kwargs: Dict = {k: cfg[k] for k in _TRAINER_KEYS if k in cfg}

    # Warn about unrecognised keys (but don't error, for forward compat).
    known = _TRAIN_FROM_ATOMS_KEYS | _TRAINER_KEYS | {"data_path"}
    unknown = set(cfg) - known
    if unknown:
        import warnings as _warnings

        _warnings.warn(
            f"train_from_config: unrecognised config keys ignored: {sorted(unknown)}",
            stacklevel=2,
        )

    return train_from_atoms(
        data,
        data_path=str(Path(data_path).resolve()),
        **train_kwargs,
        **trainer_kwargs,
    )
