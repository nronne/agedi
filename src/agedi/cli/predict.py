import rich_click as click
from rich.console import Console
from pathlib import Path


click.rich_click.OPTION_GROUPS.update(
    {
        "agedi predict": [
            {"name": "Model Options", "options": ["path"]},
            {
                "name": "Input / Output Options",
                "options": [
                    "--output",
                    "--name",
                    "--batch_size",
                ],
            },
        ]
    }
)


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("input_path", metavar="INPUT", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    show_default=True,
    default=".",
    help="Directory to save the output trajectory to.",
)
@click.option(
    "--name",
    type=str,
    show_default=True,
    default="predicted",
    help="Base name for the output trajectory file (without extension).",
)
@click.option(
    "--batch_size",
    "-b",
    type=int,
    show_default=True,
    default=64,
    help="Number of structures per inference batch.",
)
def predict(path: str, input_path: str, **kwargs) -> None:
    """Predict energies and forces for structures in INPUT.

    Loads the trained AGeDi model from PATH and runs the force-field
    regressor on each structure in INPUT.  The structures with predicted
    energies and forces are saved to the output directory.

    The model must have been trained with the ``--force_field`` flag.
    """
    from ase.io import read, write
    from agedi.functional import load_diffusion, predict as functional_predict

    console = Console()
    console.print(f"Loading model from: [cyan]{path}[/cyan]")

    diffusion = load_diffusion(path)

    if diffusion.regressor_model is None:
        console.print(
            "[red]Error:[/red] This model does not have a force-field regressor. "
            "Re-train with [bold]--force_field[/bold] to enable predictions."
        )
        raise SystemExit(1)

    console.print(f"Reading structures from: [cyan]{input_path}[/cyan]")
    structures = read(input_path, index=":")
    if not isinstance(structures, list):
        structures = [structures]

    predicted = functional_predict(
        diffusion, structures, batch_size=kwargs["batch_size"]
    )

    output_dir = Path(kwargs["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{kwargs['name']}.traj"
    write(str(out_path), predicted)

    console.print(
        f"[green]✓[/green] Saved {len(predicted)} structure(s) to: [cyan]{out_path}[/cyan]"
    )
    console.print("To inspect predicted properties, load the trajectory in ASE:")
    console.print(f"  [bold]ase gui {out_path}[/bold]")
