"""Model-backend registry and builder helpers."""

from typing import Dict, List, Optional, Sequence, Tuple, Union


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


def _build_type_map_from_data(data: Sequence["Atoms"]) -> List[int]:
    """Build a compact type map from the element types present in training data.

    The map is ``[0, z1, z2, ...]`` where ``z1 < z2 < ...`` are the sorted
    unique atomic numbers found in *data*.  Index 0 is reserved for the
    absorbing state.

    Parameters
    ----------
    data : Sequence[Atoms]
        List of ASE :class:`~ase.Atoms` objects to inspect.

    Returns
    -------
    List[int]
        A list where ``type_map[i]`` is the atomic number corresponding to
        compact index ``i`` (and ``type_map[0] == 0`` for the absorbing state).
    """
    unique_z: set = set()
    for atoms in data:
        unique_z.update(int(z) for z in atoms.get_atomic_numbers())
    return [0] + sorted(unique_z)


def _build_noisers(
    noisers: Sequence[Union[str, "Noiser"]],
    sde: Union[str, "SDE"] = "ve",
    type_map: Optional[List[int]] = None,
    prediction_type: str = "score",
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
    type_map : List[int], optional
        Compact type map (see :func:`_build_type_map_from_data`) to pass to
        the :class:`~agedi.diffusion.noisers.Types` noiser when building it
        from a string identifier.  Ignored for all other noiser types and for
        already-instantiated noiser objects.

    Returns
    -------
    List[Noiser]
        Instantiated noisers in the same order as *noisers*.
    """
    from agedi.diffusion.noisers import Noiser, Types

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
        if noiser in ("Types", "types") and type_map is not None:
            noiser_list.append(Types(type_map=type_map))
        else:
            noiser_list.append(Noiser._registry[noiser](sde=resolved_sde, prediction_type=prediction_type))

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


def _build_score_components(
    model: str,
    cutoff: float,
    heads: Sequence[str],
    feature_size: int,
    n_blocks: int,
    head_dim: int,
    n_rbf: int = 30,
    precondition: bool = False,
    sigma_data: float = 1.0,
) -> Tuple["Translator", "torch.nn.Module", List["Head"]]:
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
    precondition : bool, optional
        Whether to apply EDM preconditioning in the positions score head.
        Defaults to ``False``.
    sigma_data : float, optional
        Empirical std of (zero-COM) atom positions in the training set (Å).
        Only used when *precondition* is ``True``.  Defaults to ``1.0``.

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
        precondition=precondition,
        sigma_data=sigma_data,
    )


def _painn_factory(
    cutoff: float,
    heads: Sequence[str],
    feature_size: int,
    n_blocks: int,
    head_dim: int,
    n_rbf: int,
    precondition: bool = False,
    sigma_data: float = 1.0,
) -> Tuple["Translator", "torch.nn.Module", List["Head"]]:
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
                h.append(
                    PositionsScore(
                        input_dim_scalar=head_dim,
                        input_dim_vector=feature_size,
                        precondition=precondition,
                        sigma_data=sigma_data,
                    )
                )
            case "Types" | "types":
                h.append(TypesScore(input_dim_scalar=head_dim))
            case _ if hasattr(head, "_key") and head._key in ("pos", "positions"):
                h.append(
                    PositionsScore(
                        input_dim_scalar=head_dim,
                        input_dim_vector=feature_size,
                        precondition=precondition,
                        sigma_data=sigma_data,
                    )
                )
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

    The resulting model is attached to the :class:`~agedi.Agedi` object as
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
