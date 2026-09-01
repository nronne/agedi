import rich_click as click
from rich.console import Console
from pathlib import Path

click.rich_click.OPTION_GROUPS.update(
    {
        "agedi train": [
            {
                "name": "Score Model Options",
                "options": ["--model", "--cutoff", "--feature_size", "--n_blocks"],
            },
            {
                "name": "Diffusion Model Options",
                "options": ["--noisers", "--sde", "--conditioning", "--conditioning_type", "--force_field", "--n_classes"],
            },
            {
                "name": "Force Field Options",
                "options": [
                    "--reference_energies",
                    "--force_loss",
                    "--huber_delta",
                    "--regressor_loss_weight",
                    "--loss_balance",
                    "--conservative_forces",
                ],
            },
            {
                "name": "Training Options",
                "options": [
                    "--epochs",
                    "--max_time",
                    "--max_time_minutes",
                    "--lr",
                    "--batch_size",
                    "--lr_patience",
                    "--lr_factor",
                    "--progress_bar",
                    "--gradient_clip_val",
                    "--checkpoint",
                ],
            },
            {
                "name": "Data Options",
                "options": [
                    "--mask",
                    "--confinement",
                    "--repeat",
                    "--repeat_epoch",
                    "--canonical_cell",
                ],
            },
            {
                "name": "Logging Options",
                "options": [
                    "--logger",
                    "--log_dir",
                    "--project",
                    "--name",
                    "--log_interval",
                ],
            },
        ]
    }
)


_VALID_NOISERS = {
    "Positions",
    "CellPositions",
    "ConfinedCellPositions",
    "Types",
    "positions",
    "cell_positions",
    "confined_cell_positions",
    "types",
}
_DEFAULT_NOISER = "CellPositions"


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("overrides", nargs=-1, metavar="[KEY=VALUE ...]")
@click.option(
    "--sde",
    type=click.Choice(["ve", "vp"]),
    default="ve",
    show_default=True,
    help="SDE to use for position noisers",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["PaiNN"]),
    default="PaiNN",
    show_default=True,
    help="Representation to use for the model",
)
@click.option(
    "--cutoff",
    "-r",
    type=float,
    default=6.0,
    show_default=True,
    help="Cutoff for the representation in Å",
)
@click.option(
    "--feature_size",
    "-f",
    type=int,
    default=64,
    show_default=True,
    help="Feature size for the representation",
)
@click.option(
    "--n_blocks",
    type=int,
    default=4,
    show_default=True,
    help="Number of blocks for the representation",
)
@click.option(
    "--noisers",
    "-n",
    type=str,
    default=(_DEFAULT_NOISER,),
    multiple=True,
    show_default=True,
    help=(
        "Noiser(s) to use for diffusion. "
        "Valid values: Positions, CellPositions, ConfinedCellPositions, Types "
        "(snake_case aliases also accepted). "
        "Use a comma-separated list to specify multiple noisers in a single flag "
        "(e.g. '--noisers ConfinedCellPositions,Types'), "
        "or repeat the flag (e.g. '--noisers ConfinedCellPositions --noisers Types')."
    ),
)
@click.option(
    "--conditioning",
    "-c",
    type=str,
    default="none",
    help="Property to condition on",
    hidden=True,
)
@click.option(
    "--conditioning_type",
    type=click.Choice(["scalar", "integer", "node"]),
    default="scalar",
    show_default=True,
    help="Type of conditioning to use",
)
@click.option(
    "--force_field",
    is_flag=True,
    default=False,
    help=(
        "Train a force field jointly with the diffusion score. Make sure the training data contains energy and force labels. "
        "Enables force-field guided sampling (--ff_guidance)."
    ),
)
@click.option(
    "--reference_energies",
    type=str,
    default="auto",
    show_default=True,
    help=(
        "Per-species reference energies subtracted from the energy target when "
        "--force_field is used. 'auto' fits them from the training data by linear "
        "least squares, 'none' disables the subtraction, or give explicit values as "
        "a comma-separated list such as 'Cu:-3.72,O:-4.95'."
    ),
)
@click.option(
    "--force_loss",
    type=click.Choice(["huber", "mse", "mae"]),
    default="huber",
    show_default=True,
    help="Loss function used for the forces head of the force field.",
)
@click.option(
    "--huber_delta",
    type=float,
    default=0.01,
    show_default=True,
    help="Transition point (eV/Å) of the Huber force loss: quadratic below, linear above.",
)
@click.option(
    "--regressor_loss_weight",
    type=float,
    default=1.0,
    show_default=True,
    help=(
        "Absolute weight of the force-field loss: "
        "loss = diffusion_loss + weight * regressor_loss. Raise it to prioritise "
        "energy/force accuracy, lower it to keep the force field from dominating "
        "the score model. Ignored when --loss_balance is given."
    ),
)
@click.option(
    "--loss_balance",
    type=str,
    default=None,
    help=(
        "Relative split between the diffusion and force-field losses, e.g. "
        "'50:50' or '80:20' (diffusion:regressor). Each term is normalised by a "
        "running estimate of its own magnitude first, so the same split transfers "
        "between systems — unlike --regressor_loss_weight. Not set by default."
    ),
)
@click.option(
    "--conservative_forces",
    is_flag=True,
    default=False,
    help=(
        "Derive forces as -dE/dR by autograd through the energy head instead of "
        "using a dedicated forces head. Guarantees energy/force consistency at the "
        "cost of a backward pass on every regressor call (also slows down "
        "force-field guided sampling). Only used when --force_field is set."
    ),
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=-1,
    show_default=True,
    help="Number of epochs to train for",
)
@click.option(
    "--max_time",
    "-T",
    type=int,
    default=0,
    show_default=True,
    help="Maximum training time in hours (use -t/--max_time_minutes for minutes)",
)
@click.option(
    "--max_time_minutes",
    "-t",
    type=int,
    default=5,
    show_default=True,
    help="Additional minutes to add to the maximum training time (combined with --max_time)",
)
@click.option("--lr", type=float, default=1e-4, show_default=True, help="Learning rate")
@click.option(
    "--batch_size", "-b", type=int, default=64, show_default=True, help="Batch size"
)
@click.option(
    "--lr_patience",
    type=int,
    default=100,
    show_default=True,
    help="Number of epochs to wait before reducing the learning rate",
)
@click.option(
    "--lr_factor",
    type=float,
    default=0.95,
    show_default=True,
    help="Factor to reduce the learning rate by",
)
@click.option(
    "--gradient_clip_val",
    type=float,
    default=10.0,
    show_default=True,
    help="Gradient clipping value",
)
@click.option(
    "--mask",
    type=click.Choice(["MaskFixed", "none"]),
    default="none",
    help="Masking to use for the data",
)
@click.option(
    "--confinement",
    nargs=2,
    type=float,
    default=None,
    help="Z-confinement to use for the data. Give min and max value",
)
@click.option(
    "--repeat",
    type=int,
    default=None,
    help="Maximal number of times to repeat the data for training",
)
@click.option(
    "--repeat_epoch",
    type=int,
    default=None,
    help="How many epochs between repeats",
)
@click.option(
    "--canonical_cell",
    "canonical_cell",
    default=False,
    is_flag=True,
    show_default="store as given in the input data",
    help="Store cell in canonical lower-triangular form",
)
@click.option(
    "--logger",
    type=click.Choice(["tensorboard", "wandb"]),
    default="tensorboard",
    help="Logger to use",
)
@click.option(
    "--log_dir",
    type=click.Path(),
    default="logs",
    show_default=True,
    help="Directory to save logs to",
)
@click.option(
    "--project",
    type=str,
    default="agedi",
    show_default=True,
    help="Project name for wandb",
)
@click.option(
    "--name",
    type=str,
    default="agedi",
    show_default=True,
    help="Display name for wandb",
)
@click.option(
    "--log_interval", type=int, default=10, show_default=True, help="Interval to log at"
)
@click.option(
    "--n_classes",
    type=int,
    default=None,
    help=(
        "Number of element-type classes for the Types noiser (not counting the "
        "absorbing state). When not set, all distinct element types in the training "
        "data are used automatically. Only relevant when '--noisers Types' is specified."
    ),
)
@click.option("--progress_bar", is_flag=True, help="Show progress bar")
@click.option(
    "--checkpoint",
    type=click.Path(),
    default=None,
    help=(
        "Path to a run directory or checkpoint file to continue training from. "
        "The model weights and full training state (optimiser, LR-scheduler, "
        "epoch counter) are restored before training resumes. "
        "Supply new input data to fine-tune on a different dataset."
    ),
)
def train(**params) -> None:
    """Train an AGeDi diffusion model from the command line.

    INPUT can be a trajectory file or a YAML configuration file.

    \b
    Trajectory mode — build the model from CLI options:

        agedi train training_data.traj --noisers ConfinedCellPositions,Types

    \b
    Config mode — load all settings from a YAML file:

        agedi train my_train.yaml

    In config mode, optional KEY=VALUE pairs can be appended to override
    individual config entries without editing the file:

    \b
        agedi train my_train.yaml feature_size=128 epochs=200

    A ready-to-edit YAML template is available at:

    \b
        python -c "import agedi, pathlib; print(pathlib.Path(agedi.__file__).parent / 'conf' / 'train.yaml')"
    """
    import yaml
    from agedi.functional import train_from_config

    console = Console()
    input_path = Path(params["input"])

    if input_path.suffix.lower() in (".yaml", ".yml"):
        # ── Config-file mode ──────────────────────────────────────────────
        with open(input_path) as fh:
            cfg: dict = yaml.safe_load(fh) or {}

        for override in params["overrides"]:
            if "=" not in override:
                raise click.UsageError(
                    f"Override '{override}' is not in KEY=VALUE format."
                )
            key, _, raw_value = override.partition("=")
            cfg[key] = _parse_override_value(raw_value)

        # CLI --checkpoint overrides any checkpoint key already in the config.
        if params["checkpoint"] is not None:
            cfg["checkpoint"] = params["checkpoint"]

        train_from_config(cfg)
        log_dir = cfg.get("log_dir", "logs")
    else:
        # ── Trajectory mode ───────────────────────────────────────────────
        from ase.io import read
        from agedi.functional import train_from_atoms

        # Parse comma-separated values and flatten into a single list
        noisers: list[str] = []
        for entry in params["noisers"]:
            for part in entry.split(","):
                part = part.strip()
                if not part:
                    continue
                if part not in _VALID_NOISERS:
                    raise click.BadParameter(
                        f"'{part}' is not a valid noiser. "
                        f"Valid options: {', '.join(sorted(_VALID_NOISERS))}",
                        param_hint="'--noisers'",
                    )
                noisers.append(part)
        if not noisers:
            noisers = [_DEFAULT_NOISER]

        data_path = str(input_path.resolve())
        data = read(data_path, ":")

        max_time_minutes = params["max_time_minutes"]
        if max_time_minutes > 0:
            max_time = {"hours": params["max_time"], "minutes": max_time_minutes}
        else:
            max_time = params["max_time"]

        train_from_atoms(
            data,
            model=params["model"],
            cutoff=params["cutoff"],
            feature_size=params["feature_size"],
            n_blocks=params["n_blocks"],
            noisers=noisers,
            sde=params["sde"],
            conditioning=params["conditioning"],
            conditioning_type=params["conditioning_type"],
            force_field=params["force_field"],
            reference_energies=_parse_reference_energies(params["reference_energies"]),
            force_loss=params["force_loss"],
            huber_delta=params["huber_delta"],
            regressor_loss_weight=params["regressor_loss_weight"],
            loss_balance=params["loss_balance"],
            conservative_forces=params["conservative_forces"],
            mask=params["mask"],
            confinement=params["confinement"],
            batch_size=params["batch_size"],
            repeat=params["repeat"],
            canonical_cell=params["canonical_cell"],
            lr=params["lr"],
            lr_factor=params["lr_factor"],
            lr_patience=params["lr_patience"],
            data_path=data_path,
            epochs=params["epochs"],
            max_time=max_time,
            logger=params["logger"],
            log_dir=params["log_dir"],
            project=params["project"],
            name=params["name"],
            log_interval=params["log_interval"],
            gradient_clip_val=params["gradient_clip_val"],
            progress_bar=params["progress_bar"],
            repeat_epoch=params["repeat_epoch"],
            n_classes=params["n_classes"],
            checkpoint=params["checkpoint"],
        )
        log_dir = params["log_dir"]

    console.print(f"\n[green]✓ Training complete.[/green]")
    console.print(f"To sample from the model run:")
    console.print(f"  [bold]agedi sample {log_dir} -f ...[/bold]")


def _parse_reference_energies(raw: str):
    """Parse the ``--reference_energies`` option value.

    Accepts ``"auto"`` (fit from the training data), ``"none"``/``"null"``
    (disabled), or a comma-separated list of ``SPECIES:ENERGY`` pairs such as
    ``"Cu:-3.72,O:-4.95"``.

    Parameters
    ----------
    raw : str
        The raw option string.

    Returns
    -------
    str, dict, or None
        A value accepted by :func:`~agedi.functional.train_from_atoms`.

    Raises
    ------
    click.BadParameter
        If the string is neither a keyword nor a valid list of pairs.
    """
    text = (raw or "").strip()
    if text.lower() == "auto":
        return "auto"
    if text.lower() in ("none", "null", "off", ""):
        return None

    reference: dict = {}
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        species, sep, value = pair.partition(":")
        if not sep:
            raise click.BadParameter(
                f"'{pair}' is not a SPECIES:ENERGY pair (e.g. 'Cu:-3.72').",
                param_hint="'--reference_energies'",
            )
        try:
            reference[species.strip()] = float(value)
        except ValueError:
            raise click.BadParameter(
                f"'{value.strip()}' is not a number in '{pair}'.",
                param_hint="'--reference_energies'",
            ) from None

    if not reference:
        raise click.BadParameter(
            "No SPECIES:ENERGY pairs found.", param_hint="'--reference_energies'"
        )
    return reference


def _parse_override_value(raw: str):
    """Coerce a CLI override string to int, float, bool, or str."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() in ("null", "none", "~"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw
