"""Feature-space novelty guidance for diffusion sampling.

Where :mod:`agedi.diffusion.guidance` pulls samples *towards* low-energy
geometries, this module pushes them *away* from structures that have already
been seen.  It is intended for global-optimisation loops, where the score model
is repeatedly retrained on what has been found so far and therefore tends to
re-propose the same minima.

Each structure is summarised by a feature vector — the pooled scalar
representation of the score model's backbone — and a repulsive potential is
applied in that feature space during the reverse trajectory:

.. math::

    \\Phi_g = \\sum_{j} \\exp\\left(
        -\\frac{\\lVert \\hat f_g - \\hat f_j \\rVert^2}{2\\sigma^2}\\right),
    \\qquad
    \\Delta x = -\\eta(t)\\, \\nabla_x \\sum_g \\Phi_g

where :math:`j` runs over a persistent *archive* of already-found structures
and over the other members of the current batch.

The potential is a plain sum of Gaussians — the bias metadynamics deposits —
and *not* its logarithm.  This matters: for a single reference,
:math:`\\log` of a Gaussian is an inverted parabola whose gradient grows
without bound with distance, so a structure that is already far from
everything known would be pushed hardest.  The un-logged sum has the opposite
and correct behaviour, its gradient decaying with the Gaussian tail, so
repulsion is felt only near already-visited structures and distant references
drop out of the sum automatically.

This is the `Particle Guidance <https://arxiv.org/abs/2310.13102>`_ framework
of Corso et al. (ICLR 2024) extended with a fixed archive term — equivalently,
a metadynamics history bias applied in feature space during denoising.

This module provides:

- :class:`NoveltyGuidanceConfig` – configuration dataclass.
- :func:`structure_features` – pooled backbone features for a batch.
- :class:`FeatureArchive` – reference features of already-found structures.
- :func:`novelty_guidance_step` – one guidance step (module-level).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from agedi.data import AtomsGraph

if TYPE_CHECKING:
    from ase import Atoms

    from agedi.models.score import ScoreModel


@dataclasses.dataclass
class NoveltyGuidanceConfig:
    """Configuration for feature-space novelty guidance.

    Parameters
    ----------
    guidance : float
        Scale of the repulsion applied at each reverse step.  Dimensionless,
        like :attr:`~agedi.diffusion.guidance.ForcefieldGuidanceConfig.guidance`
        — the raw gradient magnitude depends on the backbone activations, so
        this needs calibrating per model against ``max_step_size``.
        Set to ``0.0`` (the default) to disable novelty guidance entirely.
    sigma : float
        Bandwidth of the Gaussian kernel in unit-normalised feature space,
        where distances lie in ``[0, 2]``.

        The repulsive force on a structure at feature distance ``d`` from a
        reference goes as ``d * exp(-d**2 / (2 * sigma**2))``: it is **zero at
        ``d = 0``**, peaks at ``d = sigma``, and decays beyond.  So ``sigma``
        sets the radius of "too similar" — pick it near the feature distance
        that separates structures you consider duplicates, and anything more
        than a few ``sigma`` away is left alone.  Calibrate against the
        observed distribution of pairwise feature distances in the archive
        rather than guessing.
    zeta : float
        Exponent for the time-dependent weight factor ``t**zeta``.  Note this
        is the *opposite* end of the trajectory from
        :class:`~agedi.diffusion.guidance.ForcefieldGuidanceConfig`, which uses
        ``(1 - t)**zeta``: which basin a sample falls into is decided at high
        noise, so novelty guidance must be front-loaded.  Late repulsion only
        distorts an already-committed geometry.
    max_step_size : float
        Hard cap on the per-atom displacement magnitude (Å).  Because the raw
        gradient is used unnormalised, this is what bounds a single step.
    include_batch : bool
        Whether to also repel the members of the current batch from each other.
        Costs nothing extra — the features are already computed — and prevents
        a whole batch from collapsing into the same new basin.
    pool : str
        How to pool per-atom features into a structure feature: ``"mean"``
        (default, size-invariant) or ``"sum"``.
    """

    guidance: float = 0.0
    sigma: float = 0.5
    zeta: float = 1.0
    max_step_size: float = 0.1
    include_batch: bool = True
    pool: str = "mean"


def _mobile_node_weights(batch: AtomsGraph) -> Optional[torch.Tensor]:
    """Return per-node weights that exclude fixed (template) atoms.

    Parameters
    ----------
    batch : AtomsGraph
        The batch to inspect.

    Returns
    -------
    torch.Tensor or None
        A ``(n_nodes,)`` float tensor that is ``0.0`` for fixed atoms and
        ``1.0`` for mobile ones, or ``None`` when the batch carries no mask
        (in which case every atom is mobile).
    """
    if "mask" not in batch._store:
        return None
    return (~batch.mask).to(batch.pos.dtype)


def structure_features(
    batch: AtomsGraph,
    score_model: "ScoreModel",
    positions: Optional[torch.Tensor] = None,
    pool: str = "mean",
    normalize: bool = True,
) -> torch.Tensor:
    """Compute one feature vector per structure in *batch*.

    Runs the score model's backbone and pools the resulting per-atom scalar
    representation over the mobile atoms of each graph.  Fixed template atoms
    are excluded from the pooling: they are identical across every sample and
    would only dilute the signal.

    This deliberately re-runs the backbone rather than reading
    ``batch.representation``, because
    :meth:`~agedi.models.score.ScoreModel.forward_sample` mutates that
    representation in place by concatenating conditioning columns onto it.

    Parameters
    ----------
    batch : AtomsGraph
        A batch of structures.  Its neighbour list is used as-is.
    score_model : ScoreModel
        The score model whose translator and backbone define the feature.
    positions : torch.Tensor, optional
        Positions to substitute for ``batch.pos``.  Pass a tensor with
        ``requires_grad=True`` to obtain features that are differentiable with
        respect to the atomic positions.  When ``None``, ``batch.pos`` is used.
    pool : str, optional
        ``"mean"`` (default) or ``"sum"``.
    normalize : bool, optional
        Whether to scale each feature vector to unit length.  Defaults to
        ``True``, which makes the kernel bandwidth independent of the overall
        magnitude of the backbone activations.

    Returns
    -------
    torch.Tensor
        Feature tensor of shape ``(n_graphs, n_features)``.

    Raises
    ------
    ValueError
        If *pool* is not ``"mean"`` or ``"sum"``.
    """
    if pool not in ("mean", "sum"):
        raise ValueError(f"pool must be 'mean' or 'sum', got {pool!r}")

    translated = score_model.translator.translate_input(batch, positions=positions)
    out = score_model.representation(translated)
    rep = score_model.translator.extract_representation(batch, out)

    # (n_nodes, n_features, 1) -> (n_nodes, n_features)
    scalar = rep.scalar.squeeze(-1)

    idx = batch.batch
    n_graphs = int(batch.batch_size)

    weights = _mobile_node_weights(batch)
    if weights is not None:
        scalar = scalar * weights.unsqueeze(-1)

    features = torch.zeros(
        n_graphs, scalar.shape[1], dtype=scalar.dtype, device=scalar.device
    ).index_add_(0, idx, scalar)

    if pool == "mean":
        ones = (
            weights if weights is not None else torch.ones_like(idx, dtype=scalar.dtype)
        )
        counts = torch.zeros(
            n_graphs, dtype=scalar.dtype, device=scalar.device
        ).index_add_(0, idx, ones)
        features = features / counts.clamp(min=1.0).unsqueeze(-1)

    if normalize:
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-12)

    return features


def _squared_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances between two sets of vectors.

    Computed in expanded form rather than via :func:`torch.cdist` so that the
    gradient stays finite when two vectors coincide — which happens routinely
    here, since duplicate structures are exactly what the repulsion is meant to
    separate.  ``cdist`` differentiates the square root and produces NaN at
    zero distance.

    Parameters
    ----------
    a : torch.Tensor
        Tensor of shape ``(n, d)``.
    b : torch.Tensor
        Tensor of shape ``(m, d)``.

    Returns
    -------
    torch.Tensor
        Squared distances of shape ``(n, m)``, clamped to be non-negative.
    """
    a2 = (a * a).sum(dim=-1, keepdim=True)  # (n, 1)
    b2 = (b * b).sum(dim=-1).unsqueeze(0)  # (1, m)
    return (a2 + b2 - 2.0 * a @ b.transpose(0, 1)).clamp(min=0.0)


class FeatureArchive:
    """Reference features of structures that have already been found.

    .. warning::

       Features are produced by the score model's backbone and are therefore
       only comparable *within one model generation*.  Whenever the score model
       is retrained — every iteration of a global-optimisation loop — the
       archive must be rebuilt with :meth:`from_structures`.  Reusing features
       computed by a previous model silently compares vectors from two
       different spaces.

    .. warning::

       Features are also only comparable when pooled over the *same* set of
       atoms.  Sampled structures built on a template carry a mask that excludes
       the template atoms from the pooling, so reference structures must be
       featurised with the matching ``n_template`` — see
       :meth:`from_structures`.

    Parameters
    ----------
    features : torch.Tensor, optional
        Pre-computed feature tensor of shape ``(n_references, n_features)``.
        When ``None``, the archive is empty.
    """

    def __init__(self, features: Optional[torch.Tensor] = None) -> None:
        """Initialise the archive from an optional feature tensor."""
        self.features = features

    def __len__(self) -> int:
        """Return the number of reference structures in the archive."""
        return 0 if self.features is None else int(self.features.shape[0])

    @classmethod
    def from_structures(
        cls,
        score_model: "ScoreModel",
        structures: Sequence["Atoms"],
        cutoff: float = 6.0,
        batch_size: int = 64,
        pool: str = "mean",
        n_template: int = 0,
        device: Optional[torch.device] = None,
        fully_connected: bool = False,
    ) -> "FeatureArchive":
        """Build an archive by featurising a set of ASE structures.

        Parameters
        ----------
        score_model : ScoreModel
            The score model defining the feature space.
        structures : Sequence[Atoms]
            The already-found structures to repel from.
        cutoff : float, optional
            Neighbour-list cutoff used when converting to graphs.  Should match
            the cutoff the score model was trained with.
        batch_size : int, optional
            Number of structures featurised per forward pass.
        pool : str, optional
            Pooling mode, forwarded to :func:`structure_features`.
        n_template : int, optional
            Number of leading atoms that are template atoms and must be excluded
            from the pooling, matching how sampled structures are built (the
            template comes first, and its atoms are masked).  **Getting this
            wrong silently breaks the archive**: pooling over the template as
            well averages in atoms that are identical across every structure,
            which both offsets the reference features away from the sampled
            ones and collapses the references towards each other.  ``0`` (the
            default) is correct only for template-free sampling.
        device : torch.device, optional
            Device to run on.  Defaults to the score model's device.
        fully_connected : bool, optional
            Build each reference graph with the fully-connected backbone
            topology instead of a *cutoff*-based neighbour list.  Must match
            how the sampled structures being compared against were built
            (:func:`agedi.sample` sets this from ``diffusion.fully_connected``
            automatically).  Getting it wrong is silent: both sides still
            produce a feature vector, they are just off the manifold the
            backbone was trained on.  Defaults to ``False``.

        Returns
        -------
        FeatureArchive
            An archive holding one feature vector per input structure.

        Raises
        ------
        ValueError
            If *n_template* is negative, or if some structure has no atoms left
            once the leading *n_template* are excluded.
        """
        from torch_geometric.data import Batch

        if n_template < 0:
            raise ValueError(f"n_template must be non-negative, got {n_template}")

        if len(structures) == 0:
            return cls(None)

        if n_template > 0:
            too_small = [len(a) for a in structures if len(a) <= n_template]
            if too_small:
                raise ValueError(
                    f"n_template={n_template} leaves no mobile atoms for "
                    f"{len(too_small)} reference structure(s) with "
                    f"{sorted(set(too_small))} atoms. Reference structures must "
                    "contain the template followed by at least one mobile atom."
                )

        if device is None:
            device = next(score_model.parameters()).device

        chunks = []
        with torch.no_grad():
            for start in range(0, len(structures), batch_size):
                graphs = [
                    AtomsGraph.from_atoms(
                        atoms, cutoff=cutoff, fully_connected=fully_connected
                    )
                    for atoms in structures[start : start + batch_size]
                ]
                for graph in graphs:
                    if n_template > 0:
                        # Build the mask outright rather than mutating whatever
                        # from_atoms happened to initialise, so the exclusion
                        # holds regardless of how the graph was constructed.
                        mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
                        mask[:n_template] = True
                        graph.mask = mask
                    graph.update_graph()
                batch = Batch.from_data_list(graphs).to(device)
                chunks.append(structure_features(batch, score_model, pool=pool))

        return cls(torch.cat(chunks, dim=0))

    def to(self, device: torch.device) -> "FeatureArchive":
        """Move the stored features to *device*, in place.

        Parameters
        ----------
        device : torch.device
            Target device.

        Returns
        -------
        FeatureArchive
            This archive, for chaining.
        """
        if self.features is not None:
            self.features = self.features.to(device)
        return self


def novelty_guidance_step(
    batch: AtomsGraph,
    score_model: "ScoreModel",
    archive: Optional[FeatureArchive],
    config: NoveltyGuidanceConfig,
) -> AtomsGraph:
    """Apply one feature-space repulsion step.

    The raw gradient is used directly, scaled by ``config.guidance`` and the
    time factor and then capped at ``config.max_step_size``.  It is
    deliberately *not* renormalised per structure: the magnitude of
    :math:`\\nabla\\Phi` decays with the Gaussian tail as a structure moves away
    from the reference set, which is precisely the desired behaviour — a sample
    that is already novel should be left alone.  Rescaling every structure to a
    common step size would throw that signal away and would amplify numerical
    noise for samples whose repulsion has legitimately vanished.

    Because the repulsion is bounded and short-ranged, a ``guidance`` value
    large enough to matter for near-duplicates will saturate
    ``max_step_size`` for them; that is expected, and structures further than a
    few ``sigma`` from anything known will still barely move.

    One consequence worth knowing: for two *exactly* identical structures the
    gradient is exactly zero — coincident features sit at an unstable
    equilibrium of the potential.  This is harmless in practice because the
    stochastic reverse trajectory never produces exact duplicates, but it does
    mean the repulsion cannot separate a perfectly degenerate pair on its own.

    Fixed template atoms need no special handling: assigning to ``batch.pos``
    re-imposes the position mask, so masked atoms cannot move.

    Parameters
    ----------
    batch : AtomsGraph
        A batch of structures at the current time step.
    score_model : ScoreModel
        The score model whose backbone defines the feature space.
    archive : FeatureArchive or None
        Features of already-found structures to repel from.  May be ``None``
        or empty, in which case only the in-batch term contributes.
    config : NoveltyGuidanceConfig
        Guidance configuration.

    Returns
    -------
    AtomsGraph
        The batch with updated positions.  The caller is responsible for
        wrapping positions and rebuilding the neighbour list.
    """
    if config.guidance == 0.0:
        return batch

    n_graphs = int(batch.batch_size)
    reference = archive.features if archive is not None else None
    has_archive = reference is not None and reference.shape[0] > 0
    has_batch_term = config.include_batch and n_graphs > 1

    # With a single structure and no archive there is nothing to repel from.
    if not has_archive and not has_batch_term:
        return batch

    with torch.enable_grad():
        pos = batch.pos.detach().clone().requires_grad_(True)
        features = structure_features(
            batch, score_model, positions=pos, pool=config.pool
        )

        inv = -1.0 / (2.0 * config.sigma**2)
        kernels = []

        if has_batch_term:
            d2 = _squared_distances(features, features)
            # Drop the self-pair.  Its gradient is identically zero, but
            # leaving it in would add a constant 1.0 per structure for no
            # reason.
            self_pairs = torch.eye(n_graphs, dtype=torch.bool, device=d2.device)
            kernels.append((d2 * inv).exp().masked_fill(self_pairs, 0.0))

        if has_archive:
            reference = reference.to(features.device)
            kernels.append((_squared_distances(features, reference) * inv).exp())

        potential = torch.cat(kernels, dim=1).sum()
        (grad,) = torch.autograd.grad(potential, pos)

    with torch.no_grad():
        # Descend the potential: away from the nearest reference structure.
        time_factor = batch.time**config.zeta
        step = -config.guidance * time_factor * grad

        step_magnitude = step.norm(dim=1, keepdim=True)
        too_large = step_magnitude > config.max_step_size
        if torch.any(too_large):
            scaling_factor = torch.ones_like(step_magnitude)
            scaling_factor[too_large] = config.max_step_size / step_magnitude[too_large]
            step = step * scaling_factor

        new_pos = batch.pos + step

        if getattr(batch, "confinement", None) is not None:
            z_min = batch.confinement[:, 0].unsqueeze(1)  # [B, 1]
            z_max = batch.confinement[:, 1].unsqueeze(1)  # [B, 1]

            batch_indices = batch.batch

            z_min_per_atom = z_min[batch_indices].squeeze()  # [N]
            z_max_per_atom = z_max[batch_indices].squeeze()  # [N]

            new_pos[:, 2] = torch.clamp(
                new_pos[:, 2], min=z_min_per_atom, max=z_max_per_atom
            )

        batch.pos = new_pos

    return batch
