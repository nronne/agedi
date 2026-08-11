"""Diffusion model creation and loading."""

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import torch
import yaml

from agedi.models import ScoreModel

from ._display import _print_loaded_model_info
from ._registry import _build_conditioning, _build_noisers, _build_regressor, _build_score_components


_FC_DEFAULT_CUTOFF = 50.0   # backbone cutoff auto-used when fully_connected=True


def create_diffusion(
    model: str = "PaiNN",
    cutoff: Optional[float] = None,
    feature_size: int = 64,
    n_blocks: int = 4,
    n_rbf: int = 30,
    noisers: Sequence[Union[str, "Noiser"]] = ("CellPositions",),
    sde: Union[str, "SDE", None] = None,
    conditioning: str = "none",
    conditioning_type: str = "scalar",
    confinement: Optional[Tuple[float, float]] = None,
    force_field: bool = False,
    reference_energies: Optional[Mapping[Union[int, str], float]] = None,
    force_loss: str = "huber",
    huber_delta: float = 0.01,
    lr: float = 1e-4,
    lr_factor: float = 0.95,
    lr_patience: int = 100,
    weight_decay: float = 0.0,
    eps: float = 1e-5,
    guidance_weight: float = -1.0,
    device: Optional[Union[str, torch.device]] = None,
    type_map: Optional[List[int]] = None,
    prediction_type: str = "score",
    sampler: str = "em",
    loss_weighting: str = "uniform",
    fully_connected: bool = False,
) -> "Agedi":
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
    reference_energies : Mapping, optional
        Per-species reference energies keyed by chemical symbol (``"Cu"``) or
        atomic number (``29``), in the energy unit of the training data.  The
        Energy head subtracts them from the regression target (equivalently:
        adds them back to its output), so the network only has to learn the
        residual while predictions stay on the absolute energy scale.
        ``None`` (default) disables the offset.  :func:`train_from_atoms`
        fits these from the training data automatically when *force_field* is
        enabled.  Only used when ``force_field=True``.
    force_loss : str, optional
        Point-wise loss for the forces head: ``"huber"`` (default), ``"mse"``,
        or ``"mae"``.  Only used when ``force_field=True``.
    huber_delta : float, optional
        Transition point of the Huber force loss in eV/Å — below it the loss is
        quadratic, above it linear.  Defaults to ``0.01``.
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
    type_map : List[int], optional
        Compact type map for the :class:`~agedi.diffusion.noisers.Types`
        noiser.  ``type_map[0]`` must be ``0`` (absorbing state) and
        ``type_map[i]`` is the atomic number for compact index ``i``.
        When provided, the ``Types`` noiser and the ``TypesScore`` head use
        a reduced vocabulary of size ``len(type_map)`` instead of the
        default 100.  Auto-populated by :func:`train_from_atoms` when a
        ``"Types"`` noiser is requested.

    Returns
    -------
    Agedi
        A freshly initialised :class:`~agedi.Agedi` model.
    """
    from agedi import Agedi

    torch_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    conditioning_modules = _build_conditioning(conditioning, type=conditioning_type)
    head_dim = feature_size + sum(module.output_dim for module in conditioning_modules)

    # Resolve the cutoff sentinel.
    # - Default (None) with fully_connected=True  → 50 Å so the backbone's RBFs
    #   and CosineCutoff cover the full range of distances seen during sampling.
    # - Default (None) otherwise                  → 6 Å (standard neighbour list).
    # - Explicit value                            → always respected as-is.
    if cutoff is None:
        backbone_cutoff = _FC_DEFAULT_CUTOFF if fully_connected else 6.0
    else:
        backbone_cutoff = cutoff

    # Build noiser objects first so that the TypesScore head can inherit the
    # correct n_classes from the Types noiser (via the object-based fallback
    # in _painn_factory).
    noiser_modules = _build_noisers(noisers, sde=sde, type_map=type_map, prediction_type=prediction_type, sampler=sampler, loss_weighting=loss_weighting)

    translator, representation, heads = _build_score_components(
        model,
        backbone_cutoff,
        noiser_modules,
        feature_size,
        n_blocks,
        head_dim=head_dim,
        n_rbf=n_rbf,
    )

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
            reference_energies=reference_energies,
            force_loss=force_loss,
            huber_delta=huber_delta,
        )

    return Agedi(
        score_model=score_model,
        noisers=noiser_modules,
        regressor_model=regressor_model,
        optim_config={"lr": lr, "weight_decay": weight_decay},
        scheduler_config={"factor": lr_factor, "patience": lr_patience},
        eps=eps,
        fully_connected=fully_connected,
    ).to(torch_device)


def load_diffusion(
    path: Union[str, Path],
    checkpoint: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> "Agedi":
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
    _print_loaded_model_info({"diffusion": diffusion.get_hparams()}, checkpoint_path, current_device)
    return diffusion
