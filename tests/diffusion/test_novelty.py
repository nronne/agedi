"""Tests for feature-space novelty guidance."""

import numpy as np
import pytest
import torch
from ase.build import fcc111, molecule
from torch_geometric.data import Batch

from agedi import sample
from agedi.data import AtomsGraph
from agedi.diffusion.novelty import (
    FeatureArchive,
    NoveltyGuidanceConfig,
    novelty_guidance_step,
    structure_features,
)

CUTOFF = 6.0


@pytest.fixture(scope="module")
def score_model():
    """A small untrained PaiNN score model, shared across the module."""
    import schnetpack as spk

    from agedi.models import ScoreModel
    from agedi.models.conditionings import TimeConditioning
    from agedi.models.schnetpack import PositionsScore, SchNetPackTranslator

    torch.manual_seed(0)
    model = ScoreModel(
        translator=SchNetPackTranslator(
            input_modules=[spk.atomistic.PairwiseDistances()]
        ),
        representation=spk.representation.PaiNN(
            n_atom_basis=64,
            n_interactions=3,
            radial_basis=spk.nn.GaussianRBF(n_rbf=30, cutoff=CUTOFF),
            cutoff_fn=spk.nn.CosineCutoff(CUTOFF),
        ),
        conditionings=[TimeConditioning()],
        heads=[PositionsScore(input_dim_scalar=64 + 2, input_dim_vector=64)],
    )
    model.eval()
    return model


@pytest.fixture
def water():
    a = molecule("H2O")
    a.set_cell([12.0, 12.0, 12.0])
    a.set_pbc(True)
    a.center()
    return a


@pytest.fixture
def ammonia():
    a = molecule("NH3")
    a.set_cell([12.0, 12.0, 12.0])
    a.set_pbc(True)
    a.center()
    return a


def make_batch(atoms_list, time=0.8, mask_first=0):
    """Build a batch with a valid neighbour list and a time attribute."""
    graphs = []
    for atoms in atoms_list:
        graph = AtomsGraph.from_atoms(atoms, cutoff=CUTOFF)
        if mask_first:
            graph.mask[:mask_first] = True
        graph.update_graph()
        graphs.append(graph)
    batch = Batch.from_data_list(graphs)
    batch.add_batch_attr("time", torch.full((batch.x.shape[0], 1), time), type="node")
    return batch


def max_move(batch_before, batch_after):
    return (batch_after.pos - batch_before.pos).norm(dim=1).max().item()


# ---------------------------------------------------------------------------
# structure_features
# ---------------------------------------------------------------------------


def test_features_shape_and_normalisation(score_model, water, ammonia):
    with torch.no_grad():
        f = structure_features(make_batch([water, ammonia]), score_model)

    assert f.shape == (2, 64)
    assert torch.allclose(f.norm(dim=-1), torch.ones(2), atol=1e-5)


@pytest.mark.parametrize("transform", ["translation", "permutation", "rotation"])
def test_features_are_invariant(score_model, water, transform):
    """Features must not change under symmetries of the structure."""
    moved = water.copy()
    if transform == "translation":
        moved.positions += [0.37, -0.21, 0.55]
    elif transform == "permutation":
        moved = water[[2, 0, 1]]
    else:
        moved.rotate(37, "z", center="COU")

    with torch.no_grad():
        base = structure_features(make_batch([water]), score_model)
        other = structure_features(make_batch([moved]), score_model)

    assert (base - other).norm().item() < 1e-4


def test_features_distinguish_structures(score_model, water, ammonia):
    """Genuinely different structures must have different features."""
    with torch.no_grad():
        a = structure_features(make_batch([water]), score_model)
        b = structure_features(make_batch([ammonia]), score_model)

    assert (a - b).norm().item() > 1e-2


def test_features_ignore_fixed_atoms(score_model):
    """Pooling excludes masked template atoms."""
    surf = fcc111("Au", (2, 2, 3), vacuum=10.0)
    surf.set_pbc(True)
    batch = make_batch([surf], mask_first=8)

    with torch.no_grad():
        f = structure_features(batch, score_model)

    assert f.shape == (1, 64)
    assert torch.isfinite(f).all()


def test_features_reject_bad_pool(score_model, water):
    with pytest.raises(ValueError, match="pool must be"):
        structure_features(make_batch([water]), score_model, pool="median")


def test_features_do_not_clobber_batch_representation(score_model, water):
    """structure_features must not overwrite a representation on the batch.

    The score model's sampling path stores a representation with conditioning
    columns concatenated onto it; featurising must leave that untouched.
    """
    from agedi.data import Representation

    batch = make_batch([water])
    sentinel = Representation(
        scalar=torch.full((batch.pos.shape[0], 5, 1), 1.234),
        vector=torch.zeros(batch.pos.shape[0], 5, 3),
    )
    batch.representation = sentinel

    with torch.no_grad():
        structure_features(batch, score_model)

    assert torch.allclose(batch.representation.scalar, sentinel.scalar)


# ---------------------------------------------------------------------------
# novelty_guidance_step
# ---------------------------------------------------------------------------


def test_guidance_zero_is_a_no_op(score_model, water, ammonia):
    batch = make_batch([water, ammonia])
    before = batch.pos.clone()
    out = novelty_guidance_step(
        batch, score_model, None, NoveltyGuidanceConfig(guidance=0.0)
    )
    assert torch.equal(out.pos, before)


def test_single_structure_without_archive_is_a_no_op(score_model, water):
    """Nothing to repel from: no archive and only one structure in the batch."""
    batch = make_batch([water])
    before = batch.pos.clone()
    out = novelty_guidance_step(
        batch, score_model, None, NoveltyGuidanceConfig(guidance=10.0)
    )
    assert torch.equal(out.pos, before)


def test_identical_structures_are_an_unstable_equilibrium(score_model, water):
    """Coincident features give a vanishing gradient — a documented limitation."""
    batch = make_batch([water, water.copy()])
    before = make_batch([water, water.copy()])
    out = novelty_guidance_step(
        batch, score_model, None, NoveltyGuidanceConfig(guidance=10.0)
    )
    assert max_move(before, out) < 1e-5


def test_in_batch_repulsion_separates_near_duplicates(score_model, water):
    """Two near-identical structures are pushed apart in feature space."""
    torch.manual_seed(0)
    near = water.copy()
    near.positions += np.random.default_rng(0).normal(scale=0.05, size=(len(water), 3))

    batch = make_batch([water, near])
    with torch.no_grad():
        before = structure_features(batch, score_model)
    d_before = (before[0] - before[1]).norm().item()

    out = novelty_guidance_step(
        batch, score_model, None, NoveltyGuidanceConfig(guidance=50.0, sigma=0.5)
    )
    out.wrap_positions()
    out.update_graph()
    with torch.no_grad():
        after = structure_features(out, score_model)
    d_after = (after[0] - after[1]).norm().item()

    assert d_after > d_before


def test_archive_repulsion_increases_distance(score_model, water):
    """A sample is pushed away from an archive entry it resembles."""
    near = water.copy()
    near.positions += np.random.default_rng(1).normal(scale=0.05, size=(len(water), 3))

    with torch.no_grad():
        archive = FeatureArchive(
            structure_features(make_batch([water]), score_model).clone()
        )
        batch = make_batch([near])
        d_before = (
            (structure_features(batch, score_model) - archive.features).norm().item()
        )

    out = novelty_guidance_step(
        batch, score_model, archive, NoveltyGuidanceConfig(guidance=50.0, sigma=0.5)
    )
    out.wrap_positions()
    out.update_graph()
    with torch.no_grad():
        d_after = (
            (structure_features(out, score_model) - archive.features).norm().item()
        )

    assert d_after > d_before


def test_distant_structures_are_pushed_less(score_model, water, ammonia):
    """The Gaussian bias is short-ranged: many sigma away means (almost) no force.

    A Gaussian exerts zero force at its centre and peaks at ``d = sigma``, so
    "already novel" means "many sigma away".  Sigma is chosen here so the
    near-duplicate sits well inside the well and ammonia sits several sigma out.
    """
    near = water.copy()
    near.positions += np.random.default_rng(2).normal(scale=0.05, size=(len(water), 3))

    with torch.no_grad():
        archive = FeatureArchive(
            structure_features(make_batch([water]), score_model).clone()
        )

    # Small guidance so neither case saturates max_step_size.
    config = NoveltyGuidanceConfig(guidance=1e-3, sigma=0.15)

    def move(atoms):
        before = make_batch([atoms])
        after = novelty_guidance_step(make_batch([atoms]), score_model, archive, config)
        return max_move(before, after)

    assert move(ammonia) < move(near)


def test_fixed_atoms_do_not_move(score_model):
    """Template atoms stay exactly put while mobile atoms are displaced."""
    surf = fcc111("Au", (2, 2, 3), vacuum=10.0)
    surf.set_pbc(True)
    perturbed = surf.copy()
    perturbed.positions[8:] += np.random.default_rng(3).normal(
        scale=0.05, size=(len(surf) - 8, 3)
    )

    batch = make_batch([surf, perturbed], mask_first=8)
    before = batch.pos.clone()
    out = novelty_guidance_step(
        batch, score_model, None, NoveltyGuidanceConfig(guidance=50.0, sigma=0.5)
    )

    delta = (out.pos - before).norm(dim=1)
    assert delta[out.mask].max().item() == 0.0
    assert delta[~out.mask].max().item() > 0.0


def test_step_is_capped_by_max_step_size(score_model, water):
    near = water.copy()
    near.positions += np.random.default_rng(4).normal(scale=0.05, size=(len(water), 3))

    batch = make_batch([water, near])
    before = batch.pos.clone()
    out = novelty_guidance_step(
        batch,
        score_model,
        None,
        NoveltyGuidanceConfig(guidance=1e6, sigma=0.5, max_step_size=0.05),
    )
    assert (out.pos - before).norm(dim=1).max().item() <= 0.05 + 1e-6


def test_time_factor_scales_the_step(score_model, water):
    """Novelty guidance is front-loaded: larger t gives a larger step."""
    near = water.copy()
    near.positions += np.random.default_rng(5).normal(scale=0.05, size=(len(water), 3))
    config = NoveltyGuidanceConfig(guidance=1e-2, sigma=0.5, zeta=1.0)

    def move(time):
        before = make_batch([water, near], time=time)
        after = novelty_guidance_step(
            make_batch([water, near], time=time), score_model, None, config
        )
        return max_move(before, after)

    assert move(0.9) > move(0.1)


def test_runs_under_enclosing_no_grad(score_model, water):
    """Sampling wraps everything in no_grad; guidance must still differentiate."""
    near = water.copy()
    near.positions += np.random.default_rng(6).normal(scale=0.05, size=(len(water), 3))

    with torch.no_grad():
        batch = make_batch([water, near])
        before = batch.pos.clone()
        out = novelty_guidance_step(
            batch, score_model, None, NoveltyGuidanceConfig(guidance=50.0, sigma=0.5)
        )

    assert (out.pos - before).norm().item() > 0.0


def test_guidance_leaves_positions_finite_and_graph_valid(score_model, water):
    near = water.copy()
    near.positions += np.random.default_rng(7).normal(scale=0.05, size=(len(water), 3))

    batch = make_batch([water, near])
    out = novelty_guidance_step(
        batch, score_model, None, NoveltyGuidanceConfig(guidance=50.0, sigma=0.5)
    )
    out.wrap_positions()
    out.update_graph()

    assert out.pos.isfinite().all()
    assert out.edge_index.shape[0] == 2


def test_guidance_does_not_accumulate_parameter_gradients(score_model, water):
    """autograd.grad on the positions must not leave grads on model weights."""
    near = water.copy()
    near.positions += np.random.default_rng(8).normal(scale=0.05, size=(len(water), 3))

    for param in score_model.parameters():
        param.grad = None

    novelty_guidance_step(
        make_batch([water, near]),
        score_model,
        None,
        NoveltyGuidanceConfig(guidance=50.0, sigma=0.5),
    )

    assert all(p.grad is None for p in score_model.parameters())


# ---------------------------------------------------------------------------
# FeatureArchive
# ---------------------------------------------------------------------------


def test_empty_archive(score_model):
    archive = FeatureArchive.from_structures(score_model, [])
    assert len(archive) == 0
    assert archive.features is None


def test_archive_from_structures(score_model, water, ammonia):
    archive = FeatureArchive.from_structures(
        score_model, [water, ammonia, water], cutoff=CUTOFF, batch_size=2
    )
    assert len(archive) == 3
    assert archive.features.shape == (3, 64)
    # Batching must not change the result: entries 0 and 2 are the same structure
    # but land in different chunks with batch_size=2.
    assert torch.allclose(archive.features[0], archive.features[2], atol=1e-5)


def test_archive_to_device(score_model, water):
    archive = FeatureArchive.from_structures(score_model, [water])
    assert archive.to(torch.device("cpu")) is archive
    assert archive.features.device.type == "cpu"


def test_archive_fully_connected_uses_fc_topology(score_model, water):
    """``fully_connected=True`` must reach ``AtomsGraph.from_atoms`` and
    change the graph actually fed to the backbone.

    A water molecule is small enough that a 6 A cutoff neighbour list is
    already fully connected internally, so use a cutoff that only spans O-H
    (about 1 A) and not H-H (about 1.6 A): the cutoff-graph and the
    fully-connected graph then have a genuinely different edge count, and the
    resulting pooled features must differ. This is the regression test for
    the bug where reference structures were always featurised on a
    cutoff-based graph regardless of whether the score model was trained
    ``fully_connected=True`` (e.g. gas-phase clusters trained with
    ``create_diffusion(fully_connected=True, cutoff=12.0)``), silently
    running the backbone on out-of-distribution topology.
    """
    narrow_cutoff = 1.2

    cutoff_graph = AtomsGraph.from_atoms(water, cutoff=narrow_cutoff)
    cutoff_graph.update_graph()
    n_edges_cutoff = cutoff_graph.edge_index.shape[1]

    fc_graph = AtomsGraph.from_atoms(
        water, cutoff=narrow_cutoff, fully_connected=True
    )
    fc_graph.update_graph()
    n_edges_fc = fc_graph.edge_index.shape[1]

    assert n_edges_fc > n_edges_cutoff, (
        "test setup: the narrow cutoff must NOT already be fully connected, "
        "otherwise this test cannot distinguish the two code paths"
    )

    archive_cutoff = FeatureArchive.from_structures(
        score_model, [water], cutoff=narrow_cutoff, fully_connected=False
    )
    archive_fc = FeatureArchive.from_structures(
        score_model, [water], cutoff=narrow_cutoff, fully_connected=True
    )

    assert not torch.allclose(
        archive_cutoff.features, archive_fc.features, atol=1e-5
    ), "fully_connected=True did not change the featurised graph"


# ---------------------------------------------------------------------------
# Archive / sample pooling consistency
# ---------------------------------------------------------------------------


def _adsorbate_structures(n_configs, seed=0):
    """A fixed Au(111) template with *n_configs* different Pt2 placements."""
    from ase import Atoms

    surf = fcc111("Au", (3, 3, 2), vacuum=8.0)
    surf.set_pbc(True)
    rng = np.random.default_rng(seed)
    z = surf.positions[:, 2].max()

    structures = []
    for _ in range(n_configs):
        offsets = rng.uniform([0.0, 0.0, 2.0], [6.0, 6.0, 4.0], size=(2, 3))
        full = surf + Atoms("Pt2", positions=offsets + [0.0, 0.0, z])
        full.set_cell(surf.get_cell())
        full.set_pbc(True)
        structures.append(full)
    return surf, structures


def test_archive_pooling_matches_sampled_pooling_with_template(score_model):
    """A reference structure must featurise the same way a sample would.

    Sampled structures carry the template first with its atoms masked out of
    the pooling.  ``from_structures`` must exclude the same atoms, otherwise the
    two sides of the archive comparison are different quantities and the
    repulsion no longer measures similarity to what was already found.
    """
    surf, (structure,) = _adsorbate_structures(1)
    n_template = len(surf)

    # The sampling path: template atoms masked.
    sampled = make_batch([structure], mask_first=n_template)
    with torch.no_grad():
        f_sampled = structure_features(sampled, score_model)

    archive = FeatureArchive.from_structures(
        score_model, [structure], cutoff=CUTOFF, n_template=n_template
    )

    assert torch.allclose(f_sampled, archive.features, atol=1e-5)


def test_archive_with_template_separates_distinct_configurations(score_model):
    """Excluding the template keeps the references distinguishable.

    Pooling over the template as well averages in atoms that are identical
    across every structure, which collapses the references towards each other
    and destroys the novelty signal.
    """
    surf, structures = _adsorbate_structures(6, seed=1)
    n_template = len(surf)

    masked = FeatureArchive.from_structures(
        score_model, structures, cutoff=CUTOFF, n_template=n_template
    )
    unmasked = FeatureArchive.from_structures(score_model, structures, cutoff=CUTOFF)

    def median_pairwise(features):
        d = torch.cdist(features, features)
        iu = torch.triu_indices(len(structures), len(structures), 1)
        return d[iu[0], iu[1]].median().item()

    assert median_pairwise(masked.features) > 5.0 * median_pairwise(unmasked.features)


def test_archive_rejects_template_covering_whole_structure(score_model, water):
    with pytest.raises(ValueError, match="no mobile atoms"):
        FeatureArchive.from_structures(score_model, [water], n_template=len(water))


def test_archive_rejects_negative_n_template(score_model, water):
    with pytest.raises(ValueError, match="non-negative"):
        FeatureArchive.from_structures(score_model, [water], n_template=-1)


def test_sample_excludes_the_template_when_building_the_archive(
    score_model, monkeypatch
):
    """``sample()`` must pass the template size through to the archive.

    Without it the archive is pooled over the template as well, and a reference
    lands far away in feature space from a sample of the very same structure,
    so the archive term stops measuring novelty at all.
    """
    from agedi.diffusion import Agedi
    from agedi.diffusion.noisers import CellPositions
    from agedi.diffusion.novelty import FeatureArchive as RealArchive

    surf, structures = _adsorbate_structures(2, seed=2)
    captured = {}

    class SpyArchive(RealArchive):
        @classmethod
        def from_structures(cls, *args, **kwargs):
            captured.update(kwargs)
            return RealArchive.from_structures(*args, **kwargs)

    monkeypatch.setattr("agedi.diffusion.novelty.FeatureArchive", SpyArchive)

    diffusion = Agedi(score_model, [CellPositions()])
    sample(
        diffusion,
        n_samples=2,
        formula="Pt2",
        template=surf,
        confinement=(surf.positions[:, 2].max(), surf.positions[:, 2].max() + 4.0),
        steps=3,
        cutoff=CUTOFF,
        novelty_guidance=NoveltyGuidanceConfig(guidance=1.0, sigma=0.1),
        novelty_reference=structures,
    )

    assert captured["n_template"] == len(surf)
