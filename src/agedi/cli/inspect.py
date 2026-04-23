import yaml
import rich_click as click
from pathlib import Path
from typing import Optional
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def _extract_checkpoint_metrics(ckpt: dict) -> dict:
    """Extract training metrics from a Lightning checkpoint state dict.

    Returns a dict with optional keys: ``epoch``, ``global_step``,
    ``val_loss``, ``train_loss``, ``best_val_loss``, ``best_checkpoint``.
    """
    result: dict = {}

    epoch = ckpt.get("epoch")
    if epoch is not None:
        result["epoch"] = epoch

    global_step = ckpt.get("global_step")
    if global_step is not None:
        result["global_step"] = global_step

    # ModelCheckpoint callback state
    callbacks_state = ckpt.get("callbacks", {})
    for cb_key, cb_state in callbacks_state.items():
        if "ModelCheckpoint" not in str(cb_key):
            continue
        if not isinstance(cb_state, dict):
            continue

        monitor = cb_state.get("monitor")
        best_score = cb_state.get("best_model_score")
        current_score = cb_state.get("current_score")
        best_path = cb_state.get("best_model_path") or None

        if monitor == "val_loss":
            if best_score is not None:
                try:
                    result["best_val_loss"] = float(best_score)
                except (TypeError, ValueError):
                    pass
            if current_score is not None:
                try:
                    result["val_loss"] = float(current_score)
                except (TypeError, ValueError):
                    pass
            if best_path:
                result["best_checkpoint"] = best_path

    # ``callback_metrics`` / ``logged_metrics`` (available in recent Lightning)
    for attr in ("callback_metrics", "logged_metrics"):
        metrics = ckpt.get(attr, {})
        if not isinstance(metrics, dict):
            continue
        for key in ("train_loss", "train_loss_epoch"):
            if key in metrics:
                try:
                    result.setdefault("train_loss", float(metrics[key]))
                except (TypeError, ValueError):
                    pass
        for key in ("val_loss",):
            if key in metrics:
                try:
                    result.setdefault("val_loss", float(metrics[key]))
                except (TypeError, ValueError):
                    pass

    return result


@click.command()
@click.argument("path", type=click.Path(exists=True))
def inspect(path: str) -> None:
    """Inspect a trained AGeDi model directory.

    Reads and prints the hyperparameters stored in ``hparams.yaml`` inside the
    given model directory, and reports available training metrics (epochs,
    loss, etc.) from the latest checkpoint.
    """
    from agedi.functional import _print_training_config

    console = Console()
    root_path = Path(path)
    # If the user points at a checkpoint file, go up two levels to the run dir.
    if root_path.is_file():
        root_path = root_path.parent.parent

    params_path = root_path / "hparams.yaml"
    if not params_path.exists():
        console.print(f"[red]Error:[/red] No hparams.yaml found in {root_path}")
        raise SystemExit(1)

    with open(params_path, "r") as fh:
        params = yaml.safe_load(fh) or {}

    # --- Hyperparameter panel ---
    _print_training_config(params)

    # --- Training metrics from checkpoints ---
    ckpt_dir = root_path / "checkpoints"
    last_ckpt: Optional[Path] = root_path / "checkpoints" / "last_model.ckpt"
    best_ckpt: Path = root_path / "checkpoints" / "best_model.ckpt"

    if not last_ckpt.exists():
        ckpts = sorted(ckpt_dir.glob("*.ckpt")) if ckpt_dir.is_dir() else []
        last_ckpt = ckpts[-1] if ckpts else None

    if last_ckpt is None or not last_ckpt.exists():
        return

    try:
        import torch

        last_ckpt_data = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        last_metrics = _extract_checkpoint_metrics(last_ckpt_data)

        best_metrics: dict = {}
        if best_ckpt.exists() and best_ckpt.resolve() != last_ckpt.resolve():
            best_ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            best_metrics = _extract_checkpoint_metrics(best_ckpt_data)

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        table.add_column("Key", style="bold cyan", min_width=26, no_wrap=True)
        table.add_column("Value", style="white")

        # --- Last checkpoint ---
        table.add_row("[bold]Last checkpoint[/bold]", "")
        table.add_row("  path", str(last_ckpt))
        if "epoch" in last_metrics:
            table.add_row("  epoch", str(last_metrics["epoch"]))
        if "global_step" in last_metrics:
            table.add_row("  global_step", str(last_metrics["global_step"]))
        if "train_loss" in last_metrics:
            table.add_row("  train_loss", f"{last_metrics['train_loss']:.6f}")
        if "val_loss" in last_metrics:
            table.add_row("  val_loss", f"{last_metrics['val_loss']:.6f}")

        # --- Best checkpoint ---
        table.add_row("", "")
        table.add_row("[bold]Best checkpoint[/bold]", "")
        best_ckpt_path_str = last_metrics.get("best_checkpoint")
        if best_ckpt_path_str:
            table.add_row("  path", best_ckpt_path_str)
        elif best_ckpt.exists():
            table.add_row("  path", str(best_ckpt))

        if "epoch" in best_metrics:
            table.add_row("  epoch", str(best_metrics["epoch"]))
        if "train_loss" in best_metrics:
            table.add_row("  train_loss", f"{best_metrics['train_loss']:.6f}")
        # best_val_loss from last ckpt callback state is the authoritative value
        best_val_loss = last_metrics.get("best_val_loss")
        if best_val_loss is not None:
            table.add_row("  val_loss", f"{best_val_loss:.6f}")
        elif "val_loss" in best_metrics:
            table.add_row("  val_loss", f"{best_metrics['val_loss']:.6f}")

        console.print(
            Panel(
                table,
                title="[bold]Training Metrics[/bold]",
                border_style="magenta",
            )
        )
    except Exception as exc:
        console.print(f"[yellow]Could not read checkpoint metrics:[/yellow] {exc}")
