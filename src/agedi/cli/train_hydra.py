import rich_click as click
from rich.console import Console


@click.command("train-hydra", hidden=True, deprecated=True)
@click.argument("config", type=click.Path(exists=True))
@click.argument("overrides", nargs=-1, metavar="[KEY=VALUE ...]")
def train_hydra(config: str, overrides: tuple) -> None:
    """[Deprecated] Train from a YAML config file.

    This command is deprecated. Use ``agedi train <config.yaml>`` instead:

    \b
        agedi train my_train.yaml
        agedi train my_train.yaml feature_size=128 epochs=200
    """
    from agedi.cli.train import _parse_override_value
    import yaml
    from agedi.functional import train_from_config

    console = Console()
    console.print(
        "[yellow]⚠ 'agedi train-hydra' is deprecated. "
        "Use 'agedi train <config.yaml>' instead.[/yellow]"
    )

    with open(config) as fh:
        cfg: dict = yaml.safe_load(fh) or {}

    for override in overrides:
        if "=" not in override:
            raise click.UsageError(
                f"Override '{override}' is not in KEY=VALUE format."
            )
        key, _, raw_value = override.partition("=")
        cfg[key] = _parse_override_value(raw_value)

    train_from_config(cfg)

    log_dir = cfg.get("log_dir", "logs")
    console.print("\n[green]✓ Training complete.[/green]")
    console.print("To sample from the model run:")
    console.print(f"  [bold]agedi sample {log_dir} -f ...[/bold]")
