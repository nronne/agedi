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
                    "--max_steps",
                    "--force_threshold",
                    "--scale",
                    "--max_step_size",
                    "--progress_bar",
                ],
            },
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
    "--max_steps",
    type=int,
    show_default=True,
    default=200,
    help="Maximum number of L-BFGS steps.",
)
@click.option(
    "--force_threshold",
    type=float,
    show_default=True,
    default=0.05,
    help="Convergence threshold on the maximum per-atom force (eV/Å), checked across the whole batch.",
)
@click.option(
    "--scale",
    type=float,
    show_default=True,
    default=1.0,
    help="Multiplier on the computed L-BFGS step (ASE's damping).",
)
@click.option(
    "--max_step_size",
    type=float,
    show_default=True,
    default=0.2,
    help="Maximum single-atom displacement per step, in Å.",
)
@click.option(
    "--progress_bar",
    is_flag=True,
    help="Show a progress bar and print convergence status per batch.",
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
    "--batch_size",
    "-b",
    type=int,
    show_default=True,
    default=64,
    help="Number of structures per relaxation batch.",
)
def relax(path: str, input_path: str, **kwargs) -> None:
    """Relax structures in INPUT with batched L-BFGS.

    Loads the trained AGeDi model from PATH and relaxes each structure in
    INPUT with the force-field regressor, taking L-BFGS steps until the
    maximum per-atom force drops below --force_threshold or --max_steps is
    reached. Atoms held by a FixAtoms constraint on the input structure stay
    frozen. The relaxed structures, with predicted energies and forces
    attached, are saved to the output directory.

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
        batch_size=kwargs["batch_size"],
        max_steps=kwargs["max_steps"],
        force_threshold=kwargs["force_threshold"],
        scale=kwargs["scale"],
        max_step_size=kwargs["max_step_size"],
        progress_bar=kwargs["progress_bar"],
    )

    output_dir = Path(kwargs["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{kwargs['name']}.traj"
    write(str(out_path), relaxed)

    console.print(
        f"[green]✓[/green] Saved {len(relaxed)} structure(s) to: [cyan]{out_path}[/cyan]"
    )
    console.print("To inspect relaxed structures, load the trajectory in ASE:")
    console.print(f"  [bold]ase gui {out_path}[/bold]")
