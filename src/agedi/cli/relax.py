import rich_click as click
from rich.console import Console
from pathlib import Path


click.rich_click.OPTION_GROUPS.update(
    {
        "agedi relax": [
            {"name": "Model Options", "options": ["path"]},
            {
                "name": "Relaxation Options",
                "options": [
                    "--fmax",
                    "--steps",
                    "--max_step_size",
                    "--batch_size",
                ],
            },
            {
                "name": "Input / Output Options",
                "options": [
                    "--output",
                    "--name",
                    "--save_trajectory",
                    "--progress_bar",
                ],
            },
        ]
    }
)


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("input_path", metavar="INPUT", type=click.Path(exists=True))
@click.option(
    "--fmax",
    type=float,
    show_default=True,
    default=0.05,
    help="Convergence criterion: maximum per-atom force in eV/Å.",
)
@click.option(
    "--steps",
    type=int,
    show_default=True,
    default=200,
    help="Maximum number of L-BFGS steps per structure.",
)
@click.option(
    "--max_step_size",
    type=float,
    show_default=True,
    default=0.2,
    help="Maximum single-atom displacement per step, in Å (ASE's maxstep).",
)
@click.option(
    "--batch_size",
    "-b",
    type=int,
    show_default=True,
    default=64,
    help="Number of structures relaxed simultaneously.",
)
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
    default="relaxed",
    help="Base name for the output trajectory file (without extension).",
)
@click.option(
    "--save_trajectory",
    is_flag=True,
    help="Save every optimiser step instead of only the final structure.",
)
@click.option("--progress_bar", is_flag=True, help="Show progress bar")
def relax(path: str, input_path: str, **kwargs) -> None:
    """Relax structures in INPUT with a trained force field.

    Loads the trained AGeDi model from PATH and runs a batched L-BFGS
    relaxation — equivalent to ``ase.optimize.LBFGS`` — driven by the model's
    force-field regressor.  No diffusion sampling is involved.

    ASE ``FixAtoms`` constraints on the input structures are respected.

    The model must have been trained with the ``--force_field`` flag.
    """
    from ase.io import read, write
    from agedi.functional import load_diffusion, relax as functional_relax

    console = Console()
    console.print(f"Loading model from: [cyan]{path}[/cyan]")

    diffusion = load_diffusion(path)

    if diffusion.regressor_model is None:
        console.print(
            "[red]Error:[/red] This model does not have a force-field regressor. "
            "Re-train with [bold]--force_field[/bold] to enable relaxation."
        )
        raise SystemExit(1)

    console.print(f"Reading structures from: [cyan]{input_path}[/cyan]")
    structures = read(input_path, index=":")
    if not isinstance(structures, list):
        structures = [structures]

    relaxed = functional_relax(
        diffusion,
        structures,
        fmax=kwargs["fmax"],
        steps=kwargs["steps"],
        batch_size=kwargs["batch_size"],
        max_step_size=kwargs["max_step_size"],
        trajectory=kwargs["save_trajectory"],
        progress_bar=kwargs["progress_bar"],
    )

    output_dir = Path(kwargs["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{kwargs['name']}.traj"

    if kwargs["save_trajectory"]:
        # Flatten per-structure trajectories into one file, in order.
        frames = [atoms for traj in relaxed for atoms in traj]
        write(str(out_path), frames)
        console.print(
            f"[green]✓[/green] Saved {len(frames)} frame(s) from "
            f"{len(relaxed)} structure(s) to: [cyan]{out_path}[/cyan]"
        )
    else:
        write(str(out_path), relaxed)
        console.print(
            f"[green]✓[/green] Saved {len(relaxed)} structure(s) to: [cyan]{out_path}[/cyan]"
        )

    console.print("To inspect the relaxed structures, load the trajectory in ASE:")
    console.print(f"  [bold]ase gui {out_path}[/bold]")
