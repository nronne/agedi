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
    resolve_novelty_config,
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
        batch, score_model, None, NoveltyGuidanceConfig(guidance=10.0, sigma=0.5)
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


def _move_at_time(score_model, structures, config, time):
    """Largest per-atom displacement produced by one guidance step at *time*."""
    before = make_batch(structures, time=time)
    after = novelty_guidance_step(
        make_batch(structures, time=time), score_model, None, config
    )
    return max_move(before, after)


def test_power_schedule_is_front_loaded(score_model, water):
    """With schedule="power" the guidance still decays monotonically with t."""
    near = water.copy()
    near.positions += np.random.default_rng(5).normal(scale=0.05, size=(len(water), 3))
    config = NoveltyGuidanceConfig(
        guidance=1e-2, sigma=0.5, schedule="power", zeta=1.0
    )

    def move(t):
        return _move_at_time(score_model, [water, near], config, t)

    assert move(0.9) > move(0.1)


def test_gaussian_schedule_is_a_window(score_model, water):
    """The default schedule peaks mid-trajectory and dies at both ends.

    Both ends matter: at t -> 1 the samples are a noise gas whose features are
    far from anything real, and at t -> 0 the basin is already committed.
    """
    near = water.copy()
    near.positions += np.random.default_rng(5).normal(scale=0.05, size=(len(water), 3))
    config = NoveltyGuidanceConfig(
        guidance=1e-2, sigma=0.5, schedule="gaussian", t_center=0.5, t_width=0.15
    )

    def move(t):
        return _move_at_time(score_model, [water, near], config, t)

    assert move(0.5) > move(0.9)
    assert move(0.5) > move(0.05)


def test_schedule_is_validated():
    with pytest.raises(ValueError, match="schedule must be"):
        NoveltyGuidanceConfig(schedule="triangular")
    with pytest.raises(ValueError, match="t_width must be positive"):
        NoveltyGuidanceConfig(t_width=0.0)
    with pytest.raises(ValueError, match="sigma must be positive"):
        NoveltyGuidanceConfig(sigma=0.0)
    with pytest.raises(ValueError, match="sigma_quantile must lie"):
        NoveltyGuidanceConfig(sigma_quantile=1.0)


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


def test_step_scales_linearly_with_dt(score_model, water):
    """``dt`` scaling is what makes ``guidance`` step-count independent."""
    near = water.copy()
    near.positions += np.random.default_rng(11).normal(scale=0.05, size=(len(water), 3))
    # Guidance large enough that the displacement is well clear of float32
    # position resolution, but with the cap lifted so the relation stays linear.
    config = NoveltyGuidanceConfig(guidance=10.0, sigma=0.5, max_step_size=1e9)

    def step(dt):
        before = make_batch([water, near])
        after = novelty_guidance_step(
            make_batch([water, near]), score_model, None, config, dt=dt
        )
        return after.pos - before.pos

    half, full = step(0.2), step(0.4)
    # atol covers one float32 ulp of the ~10 A positions the step is added to.
    assert half.norm().item() > 1e-3
    assert torch.allclose(full, 2.0 * half, rtol=1e-3, atol=2e-6)


def test_cap_rescales_a_structure_without_rotating_it(score_model, water):
    """The cap must bound the step, not shear the structure.

    A per-atom clip would shorten only the atoms that exceed the cap, changing
    the direction of the displacement field; one factor per structure keeps it
    parallel to the gradient.
    """
    near = water.copy()
    near.positions += np.random.default_rng(12).normal(scale=0.05, size=(len(water), 3))

    def step(max_step_size):
        before = make_batch([water, near])
        after = novelty_guidance_step(
            make_batch([water, near]),
            score_model,
            None,
            NoveltyGuidanceConfig(
                guidance=1e6, sigma=0.5, max_step_size=max_step_size
            ),
        )
        return after.pos - before.pos

    capped = step(0.05)
    uncapped_direction = step(1e9)

    # Magnitude is bounded ...
    assert capped.norm(dim=1).max().item() <= 0.05 + 1e-6
    # ... and every atom still points the same way it did before capping.
    cos = torch.nn.functional.cosine_similarity(capped, uncapped_direction, dim=1)
    assert cos.min().item() > 1.0 - 1e-5


def test_cap_ignores_fixed_atoms(score_model):
    """A template atom's gradient must not eat into the mobile atoms' budget.

    Fixed atoms carry a gradient — they shape their mobile neighbours'
    representations through message passing — but their displacement is
    discarded, so counting them in the per-structure maximum would shrink the
    step of the atoms that actually move.
    """
    surf = fcc111("Au", (2, 2, 3), vacuum=10.0)
    surf.set_pbc(True)
    other = surf.copy()
    other.positions[-1] += [0.3, 0.2, 0.1]

    def step(mask_first):
        before = make_batch([surf, other], mask_first=mask_first)
        after = novelty_guidance_step(
            make_batch([surf, other], mask_first=mask_first),
            score_model,
            None,
            NoveltyGuidanceConfig(guidance=1e4, sigma=0.5, max_step_size=0.05),
        )
        return before, after

    before, after = step(mask_first=8)
    delta = (after.pos - before.pos).norm(dim=1)

    assert delta[after.mask].max().item() == 0.0
    # The mobile atoms use the full budget rather than a fraction of it.
    assert delta[~after.mask].max().item() == pytest.approx(0.05, rel=1e-3)


def test_cap_is_applied_per_structure(score_model, water, ammonia):
    """Capping one structure must not shrink another one in the same batch."""
    near = water.copy()
    near.positions += np.random.default_rng(13).normal(scale=0.02, size=(len(water), 3))

    # water/near are near-duplicates and saturate; ammonia is far from both.
    before = make_batch([water, near, ammonia])
    after = novelty_guidance_step(
        make_batch([water, near, ammonia]),
        score_model,
        None,
        NoveltyGuidanceConfig(guidance=1e4, sigma=0.5, max_step_size=0.05),
    )
    delta = (after.pos - before.pos).norm(dim=1)
    per_structure = [delta[before.batch == g].max().item() for g in range(3)]

    assert all(d <= 0.05 + 1e-6 for d in per_structure)
    assert per_structure[2] < per_structure[0]


def test_density_normalisation_bounds_growth_with_archive_size(score_model, water):
    """The same guidance must not get more aggressive as the archive fills up.

    A plain sum of Gaussians grows with the number of nearby references, so a
    campaign that keeps adding structures keeps raising the effective guidance.
    """
    near = water.copy()
    near.positions += np.random.default_rng(14).normal(scale=0.05, size=(len(water), 3))

    def archive_of(n):
        with torch.no_grad():
            f = structure_features(make_batch([water]), score_model).clone()
        return FeatureArchive(f.repeat(n, 1))

    def move(n, normalize_density):
        before = make_batch([near])
        after = novelty_guidance_step(
            make_batch([near]),
            score_model,
            archive_of(n),
            NoveltyGuidanceConfig(
                guidance=1.0,
                sigma=0.5,
                max_step_size=1e9,
                normalize_density=normalize_density,
            ),
        )
        return max_move(before, after)

    # Unnormalised: ten copies of the same reference push ten times as hard.
    assert move(10, False) == pytest.approx(10.0 * move(1, False), rel=1e-3)
    # Normalised: the extra copies change nothing once past one effective
    # neighbour, because they scale numerator and denominator alike.
    assert move(10, True) == pytest.approx(move(1, True), rel=1e-3)


def test_density_normalisation_leaves_distant_structures_alone(
    score_model, water, ammonia
):
    """Below one effective neighbour the denominator clamps and does nothing."""
    with torch.no_grad():
        archive = FeatureArchive(
            structure_features(make_batch([water]), score_model).clone()
        )

    def move(normalize_density):
        before = make_batch([ammonia])
        after = novelty_guidance_step(
            make_batch([ammonia]),
            score_model,
            archive,
            NoveltyGuidanceConfig(
                guidance=1.0,
                sigma=0.15,
                max_step_size=1e9,
                normalize_density=normalize_density,
            ),
        )
        return max_move(before, after)

    assert move(True) == pytest.approx(move(False), rel=1e-6)


def test_in_batch_pairs_are_counted_once(score_model, water):
    """An in-batch neighbour must repel exactly as hard as an archive entry.

    The double sum over the batch visits every pair twice, so the in-batch
    kernel is halved; without that the two terms cannot be balanced.
    """
    near = water.copy()
    near.positions += np.random.default_rng(15).normal(scale=0.05, size=(len(water), 3))
    config = dict(guidance=10.0, sigma=0.5, max_step_size=1e9, normalize_density=False)

    # near repelled by water as an archive entry ...
    with torch.no_grad():
        archive = FeatureArchive(
            structure_features(make_batch([water]), score_model).clone()
        )
    before = make_batch([near])
    after = novelty_guidance_step(
        make_batch([near]),
        score_model,
        archive,
        NoveltyGuidanceConfig(include_batch=False, **config),
    )
    archive_step = after.pos - before.pos

    # ... versus water as an in-batch neighbour.
    before_batch = make_batch([near, water])
    after_batch = novelty_guidance_step(
        make_batch([near, water]),
        score_model,
        None,
        NoveltyGuidanceConfig(include_batch=True, **config),
    )
    in_batch_step = (after_batch.pos - before_batch.pos)[before_batch.batch == 0]

    assert archive_step.norm().item() > 1e-3
    assert torch.allclose(in_batch_step, archive_step, rtol=1e-3, atol=2e-6)


def test_batch_weight_scales_the_in_batch_term(score_model, water):
    near = water.copy()
    near.positions += np.random.default_rng(16).normal(scale=0.05, size=(len(water), 3))

    def step(batch_weight):
        before = make_batch([water, near])
        after = novelty_guidance_step(
            make_batch([water, near]),
            score_model,
            None,
            NoveltyGuidanceConfig(
                guidance=1.0,
                sigma=0.5,
                max_step_size=1e9,
                batch_weight=batch_weight,
                normalize_density=False,
            ),
        )
        return after.pos - before.pos

    single = step(1.0)
    assert single.norm().item() > 1e-3
    assert torch.allclose(step(2.0), 2.0 * single, rtol=1e-3, atol=2e-6)


def test_unresolved_sigma_is_rejected(score_model, water, ammonia):
    """A step must never run on an uncalibrated bandwidth."""
    with pytest.raises(ValueError, match="config.sigma is None"):
        novelty_guidance_step(
            make_batch([water, ammonia]),
            score_model,
            None,
            NoveltyGuidanceConfig(guidance=1.0),
        )


def test_non_finite_gradient_is_reported(score_model, water, monkeypatch):
    """A diverged backbone must abort loudly, not write NaN into the batch."""
    import agedi.diffusion.novelty as novelty

    real = novelty.structure_features

    def poisoned(batch, score_model, positions=None, **kwargs):
        out = real(batch, score_model, positions=positions, **kwargs)
        if positions is not None:
            out = out + positions.sum() * float("nan")
        return out

    monkeypatch.setattr(novelty, "structure_features", poisoned)

    near = water.copy()
    near.positions += np.random.default_rng(17).normal(scale=0.05, size=(len(water), 3))
    with pytest.raises(RuntimeError, match="Non-finite gradient"):
        novelty.novelty_guidance_step(
            make_batch([water, near]),
            score_model,
            None,
            NoveltyGuidanceConfig(guidance=1.0, sigma=0.5),
        )


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


def test_archive_distance_quantiles(score_model, water, ammonia):
    """The quantiles describe the archive's own pairwise distance spread."""
    archive = FeatureArchive.from_structures(
        score_model, [water, ammonia, water], cutoff=CUTOFF
    )
    q = archive.distance_quantiles((0.0, 0.5, 1.0))

    assert q.shape == (3,)
    assert q[0] <= q[1] <= q[2]
    # water appears twice, so the closest pair is coincident up to float32
    # noise, while water/ammonia are genuinely apart.
    assert q[0].item() < 1e-2
    assert q[2].item() > 100.0 * q[0].item()


def test_archive_distance_quantiles_need_a_pair(score_model, water):
    assert FeatureArchive(None).distance_quantiles() is None
    assert FeatureArchive.from_structures(
        score_model, [water]
    ).distance_quantiles() is None


def test_archive_distance_quantiles_subsample(score_model, water, ammonia):
    """Large archives are subsampled before the O(n^2) distance matrix."""
    archive = FeatureArchive.from_structures(
        score_model, [water, ammonia] * 8, cutoff=CUTOFF
    )
    torch.manual_seed(0)
    q = archive.distance_quantiles((0.5,), max_references=4)
    assert q.shape == (1,)
    assert torch.isfinite(q).all()


def test_resolve_sigma_reads_it_off_the_archive(score_model, water, ammonia):
    """sigma=None is calibrated to a quantile of the archive's distances."""
    archive = FeatureArchive.from_structures(
        score_model, [water, ammonia], cutoff=CUTOFF
    )
    resolved = resolve_novelty_config(
        NoveltyGuidanceConfig(guidance=1.0, sigma_quantile=0.5), archive
    )

    expected = archive.distance_quantiles((0.5,))[0].item()
    assert resolved.sigma == pytest.approx(expected, rel=1e-6)
    # The original config is left alone.
    assert resolved is not None


def test_resolve_sigma_is_a_no_op_when_set(score_model, water, ammonia):
    archive = FeatureArchive.from_structures(
        score_model, [water, ammonia], cutoff=CUTOFF
    )
    config = NoveltyGuidanceConfig(guidance=1.0, sigma=0.42)
    assert resolve_novelty_config(config, archive) is config


def test_resolve_sigma_needs_an_archive():
    with pytest.raises(ValueError, match="fewer than two"):
        resolve_novelty_config(NoveltyGuidanceConfig(guidance=1.0), None)


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


def test_sample_calibrates_sigma_from_the_archive(score_model, monkeypatch):
    """``sample()`` resolves ``sigma=None`` before the loop starts.

    The loop must never see an unresolved bandwidth, and the resolution has to
    happen once — not once per reverse step, where it would cost an O(n^2)
    reduction over the archive every time.
    """
    from agedi.diffusion import Agedi
    from agedi.diffusion.noisers import CellPositions
    import agedi.diffusion.novelty as novelty

    surf, structures = _adsorbate_structures(4, seed=3)

    seen = []
    real_step = novelty.novelty_guidance_step

    def spy(batch, score_model_, archive, config, dt=1.0):
        seen.append((config.sigma, float(dt)))
        return real_step(batch, score_model_, archive, config, dt)

    monkeypatch.setattr("agedi.diffusion.diffusion.novelty_guidance_step", spy)

    steps = 4
    diffusion = Agedi(score_model, [CellPositions()])
    sample(
        diffusion,
        n_samples=2,
        formula="Pt2",
        template=surf,
        confinement=(surf.positions[:, 2].max(), surf.positions[:, 2].max() + 4.0),
        steps=steps,
        eps=1e-3,
        cutoff=CUTOFF,
        novelty_guidance=NoveltyGuidanceConfig(guidance=1.0, sigma_quantile=0.5),
        novelty_reference=structures,
    )

    assert len(seen) == steps
    sigmas = {sigma for sigma, _ in seen}
    assert len(sigmas) == 1
    sigma = sigmas.pop()
    assert sigma is not None and sigma > 0.0

    # And dt reaches the step, so the accumulated bias is step-count
    # independent rather than proportional to `steps`.
    expected_dt = (1.0 - 1e-3) / (steps - 1)
    assert all(dt == pytest.approx(expected_dt, rel=1e-5) for _, dt in seen)
