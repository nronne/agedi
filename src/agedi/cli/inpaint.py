from pathlib import Path

import rich_click as click
from rich.console import Console
from ase.io import read, write

from agedi.functional import load_diffusion, inpaint as functional_inpaint


click.rich_click.OPTION_GROUPS.update(
    {
        "agedi inpaint": [
            {"name": "Model / Structure Options", "options": ["path", "structure_path"]},
            {
                "name": "Selection Options",
                "options": [
                    "--indices",
                    "--symbols",
                    "--z_range",
                    "--sphere_center",
                    "--sphere_radius",
                    "--from_atoms",
                    "--fraction",
                    "--contiguous",
                    "--freeze",
                ],
            },
            {
                "name": "Sampling Hyperparameters",
                "options": [
                    "--n_samples",
                    "--output",
                    "--name",
                    "--steps",
                    "--seed",
                    "--eps",
                    "--t_start",
                    "--batch_size",
                    "--sampler",
                    "--n_resample",
                    "--jump_length",
                    "--progress_bar",
                    "--print_timings",
                ],
            },
            {
                "name": "FFPC Sampler Options",
                "options": [
                    "--ffpc_corrector_steps",
                    "--ffpc_corrector_step_size",
                    "--ffpc_zeta",
                    "--ffpc_temperature",
                    "--ffpc_terminal_steps",
                    "--ffpc_terminal_step_size",
                    "--ffpc_terminal_dynamics",
                    "--ffpc_terminal_friction",
                ],
            },
            {
                "name": "Force-field Guidance",
                "options": [
                    "--ff_guidance",
                    "--ff_zeta",
                ],
            },
        ]
    }
)


def _parse_int_list(value):
    if value is None:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_str_list(value):
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("structure_path", type=click.Path(exists=True))
@click.option(
    "--indices", type=str, default=None,
    help="Comma-separated atom indices to regenerate, e.g. '0,1,5'.",
)
@click.option(
    "--symbols", type=str, default=None,
    help="Comma-separated chemical symbols to regenerate, e.g. 'O,H'.",
)
@click.option(
    "--z_range", nargs=2, type=float, default=None,
    help="Regenerate atoms with z-coordinate in [min, max].",
)
@click.option(
    "--sphere_center", nargs=3, type=float, default=None,
    help="Center (x, y, z) of a spherical selection region. Use with --sphere_radius.",
)
@click.option(
    "--sphere_radius", type=float, default=None,
    help="Radius of the spherical selection region. Use with --sphere_center.",
)
@click.option(
    "--from_atoms", is_flag=True,
    help="Read the selection from the structure file: atoms.arrays['inpaint_mask'] "
         "if present, else every atom not held by a FixAtoms constraint.",
)
@click.option(
    "--fraction", type=float, default=0.25, show_default=True,
    help="Fraction of non-fixed atoms selected at random when no other "
         "selection option is given.",
)
@click.option(
    "--contiguous", is_flag=True,
    help="With --fraction, select a spatially-connected cluster of "
         "neighboring atoms (grown from a random seed atom) instead of a "
         "scattered random subset.",
)
@click.option(
    "--freeze", type=str, default=None,
    help="Comma-separated atom indices to hard-freeze (never move). "
         "Must not overlap the inpainted selection.",
)
@click.option("--n_samples", "-n", type=int, show_default=True, default=4)
@click.option("--seed", "-s", type=int, show_default=True, default=42)
@click.option("--steps", type=int, show_default=True, default=500)
@click.option("--eps", type=float, show_default=True, default=0.005)
@click.option(
    "--t_start", type=float, show_default=True, default=1.0,
    help="Starting diffusion time in (0, 1]. 1.0 fully re-noises the "
         "selected atoms; lower values give a local rattle-and-relax refinement.",
)
@click.option("--batch_size", "-b", show_default=True, type=int, default=64)
@click.option("--output", "-o", type=click.Path(), show_default=True, default=".")
@click.option("--name", type=str, show_default=True, default="inpainted")
@click.option(
    "--sampler",
    type=click.Choice(["em", "pc", "heun", "ddim", "heun_ode", "ffpc"]),
    default=None,
    show_default=True,
    help="Inner reverse-diffusion sampler wrapped by the inpainting logic.",
)
@click.option(
    "--n_resample", type=int, show_default=True, default=1,
    help="RePaint-style resampling passes per reverse step. 1 disables resampling.",
)
@click.option(
    "--jump_length", type=int, show_default=True, default=1,
    help="Sub-steps per resampling pass before jumping back. Only used when --n_resample > 1.",
)
@click.option("--ffpc_corrector_steps", type=int, default=1, show_default=True)
@click.option("--ffpc_corrector_step_size", type=float, default=1e-3, show_default=True)
@click.option("--ffpc_zeta", type=float, default=1.0, show_default=True)
@click.option("--ffpc_temperature", type=float, default=1.0, show_default=True)
@click.option("--ffpc_terminal_steps", type=int, default=0, show_default=True)
@click.option("--ffpc_terminal_step_size", type=float, default=None, show_default=False)
@click.option(
    "--ffpc_terminal_dynamics",
    type=click.Choice(["overdamped", "langevin_md"]),
    default="overdamped",
    show_default=True,
)
@click.option("--ffpc_terminal_friction", type=float, default=None, show_default=False)
@click.option("--progress_bar", is_flag=True, help="Show progress bar")
@click.option("--print_timings", is_flag=True, help="Print per-stage timing breakdown after sampling")
@click.option("--save_trajectory", is_flag=True, help="Save entire diffusion trajectory")
@click.option(
    "--save_corrector_frames", is_flag=True,
    help="Also save every corrector/resampling sub-step. Requires --save_trajectory.",
)
@click.option(
    "--ff_guidance", type=float, default=0.0, show_default=True,
    help="Force-field guidance scale. Set > 0 to enable (requires a Forces head).",
)
@click.option("--ff_zeta", type=float, default=3.0, show_default=True)
def inpaint(path: str, structure_path: str, **kwargs) -> None:
    """Inpaint a structure: regenerate a chosen subset of its atoms.

    Loads the model from PATH and the input structure(s) from STRUCTURE_PATH,
    regenerates the selected atoms with masked reverse diffusion, and writes
    the result(s) to the output directory. See the Selection Options for how
    to choose which atoms are regenerated; with none given, a random fraction
    of the non-fixed atoms is selected. When STRUCTURE_PATH contains more
    than one frame, every frame is batched together and inpainted with the
    same selection spec, re-resolved independently per structure -- one
    output file per input structure (per input structure and sample when
    --save_trajectory is also given).
    """
    from agedi.diffusion import ForcefieldGuidanceConfig
    import torch

    console = Console()
    console.print(f"Loading model from: [cyan]{path}[/cyan]")
    diffusion = load_diffusion(path)

    torch.manual_seed(kwargs["seed"])

    frames = read(structure_path, index=":")
    atoms = frames[0] if len(frames) == 1 else frames
    is_multi = isinstance(atoms, list)

    ff_guidance = None
    if kwargs["ff_guidance"] > 0.0:
        ff_guidance = ForcefieldGuidanceConfig(
            guidance=kwargs["ff_guidance"],
            zeta=kwargs["ff_zeta"],
        )

    _sampler = kwargs["sampler"]
    _sampler_kwargs = None
    if _sampler == "ffpc":
        _sampler_kwargs = dict(
            corrector_steps=kwargs["ffpc_corrector_steps"],
            corrector_step_size=kwargs["ffpc_corrector_step_size"],
            mixing_zeta=kwargs["ffpc_zeta"],
            temperature=kwargs["ffpc_temperature"],
            terminal_steps=kwargs["ffpc_terminal_steps"],
            terminal_step_size=kwargs["ffpc_terminal_step_size"],
            terminal_dynamics=kwargs["ffpc_terminal_dynamics"],
            terminal_friction=kwargs["ffpc_terminal_friction"],
        )

    sphere = None
    if kwargs["sphere_center"] is not None and kwargs["sphere_radius"] is not None:
        sphere = (kwargs["sphere_center"], kwargs["sphere_radius"])

    structures = functional_inpaint(
        diffusion,
        atoms,
        indices=_parse_int_list(kwargs["indices"]),
        symbols=_parse_str_list(kwargs["symbols"]),
        z_range=kwargs["z_range"],
        sphere=sphere,
        from_atoms=kwargs["from_atoms"],
        fraction=kwargs["fraction"],
        contiguous=kwargs["contiguous"],
        seed=kwargs["seed"],
        freeze=_parse_int_list(kwargs["freeze"]),
        n_samples=kwargs["n_samples"],
        t_start=kwargs["t_start"],
        steps=kwargs["steps"],
        eps=kwargs["eps"],
        batch_size=kwargs["batch_size"],
        n_resample=kwargs["n_resample"],
        jump_length=kwargs["jump_length"],
        sampler=_sampler,
        sampler_kwargs=_sampler_kwargs,
        ff_guidance=ff_guidance,
        progress_bar=kwargs["progress_bar"],
        save_trajectory=kwargs["save_trajectory"],
        save_corrector_frames=kwargs["save_corrector_frames"],
        print_timings=kwargs["print_timings"],
        as_atoms=True,
    )

    output_dir = Path(kwargs["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    name = kwargs["name"]

    if is_multi:
        n_files = 0
        if kwargs["save_trajectory"]:
            for j, per_structure in enumerate(structures):
                for i, trajectory in enumerate(per_structure):
                    write(output_dir / f"{name}_struct{j}_sample{i}.traj", trajectory)
                    n_files += 1
        else:
            for j, per_structure in enumerate(structures):
                write(output_dir / f"{name}_struct{j}.traj", per_structure)
                n_files += 1
        out_desc = f"{n_files} file(s) in {output_dir}/"
    elif kwargs["save_trajectory"]:
        for i, trajectory in enumerate(structures):
            write(output_dir / f"{name}_{i}.traj", trajectory)
        out_desc = f"{len(structures)} trajectory file(s) in {output_dir}/"
    else:
        out_path = output_dir / f"{name}.traj"
        write(out_path, structures)
        out_desc = str(out_path)

    console.print(f"Saved to: [cyan]{out_desc}[/cyan]")
