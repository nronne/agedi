"""Rich UI / console display utilities."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree


def _fmt_target(target: str) -> str:
    """Return the short class name from a dotted ``_target_`` string."""
    return target.rsplit(".", 1)[-1] if "." in target else target


def _extract_data_info(data: Sequence["Atoms"]) -> Dict:
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
    out: Dict = {"cell": None}
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
        :meth:`~agedi.diffusion.agedi.Agedi.get_hparams`.

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
    arch_tree = Tree("[bold]Agedi[/bold]")
    _render_config_tree(diffusion_cfg, arch_tree)
    console.print(
        Panel(arch_tree, title="[bold]AGeDi Training — Model Architecture[/bold]", border_style="cyan")
    )

    # ── Dimension validation ───────────────────────────────────────────────
    dim_warnings = _check_head_dimensions(diffusion_cfg)
    for w in dim_warnings:
        console.print(w)


def _print_log_path(trainer: "Trainer") -> None:
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
    arch_tree = Tree("[bold]Agedi[/bold]")
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
    sampler: Optional[str] = None,
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
        table.add_row("  confinement", f"{confinement[0]:.2f} – {confinement[1]:.2f} Å")
    if property is not None:
        for k, v in property.items():
            table.add_row(f"  {k}", str(v))
    if force_field_guidance > 0.0:
        table.add_row("  ff_guidance", str(force_field_guidance))
    if sampler is not None:
        table.add_row("  sampler", str(sampler))

    console.print(
        Panel(table, title="[bold]AGeDi Sampling Configuration[/bold]", border_style="blue")
    )
