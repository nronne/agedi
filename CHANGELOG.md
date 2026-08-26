# Changelog

All notable changes to AGeDi will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-08-24

### Added
- **Feature-space novelty guidance** — repels samples away from structures that
  have already been found, for global-optimisation loops where the model is
  retrained on its own discoveries and tends to re-propose the same minima.
  Enabled via ``novelty_guidance=NoveltyGuidanceConfig(...)`` and
  ``novelty_reference=[...]`` on ``sample()``.
  - Each structure is summarised by its pooled (mobile-atom) backbone scalar
    representation, and a sum-of-Gaussians potential is descended in that
    feature space during the reverse trajectory — a metadynamics history bias
    applied during denoising, equivalently
    [Particle Guidance](https://arxiv.org/abs/2310.13102) (Corso et al.,
    ICLR 2024) extended with a persistent archive term.
  - Samples are repelled both from the archive and from each other within the
    batch (``include_batch``, on by default), which prevents a whole batch from
    collapsing into a single new basin.
  - ``sigma`` sets the "too similar" radius: the force is zero at zero feature
    distance, peaks at ``d = sigma``, and decays beyond, so structures that are
    already novel are left alone.
  - ``zeta`` weights the guidance by ``t**zeta`` — deliberately the opposite end
    of the trajectory from ``ForcefieldGuidanceConfig``'s ``(1 - t)**zeta``,
    since which basin a sample falls into is decided at high noise.
  - When sampling on a template, the reference structures are featurised with
    the template atoms excluded from the pooling, matching how the samples
    themselves are pooled (``FeatureArchive.from_structures(n_template=...)``,
    set automatically by ``sample()``).  Pooling over the template as well
    would average in atoms that are identical across every structure, which
    offsets the references from the samples and collapses them towards each
    other.
  - New public API in ``agedi.diffusion``: ``NoveltyGuidanceConfig``,
    ``FeatureArchive``, ``structure_features``, ``novelty_guidance_step``.
- **Automatic calibration of the novelty guidance scale.**  Set
  ``guidance=None`` and give ``target_displacement`` instead (default
  ``0.2`` Å): how far novelty guidance should move a structure at typical
  repulsion over the whole trajectory.  New ``NoveltyCalibrator`` measures the
  gradient magnitude once, at the peak of the time window, and solves for the
  scale that spends exactly that budget over the remaining schedule.
  - ``guidance`` multiplies a raw backbone gradient, so its useful magnitude is
    a property of the model's activations and changes with every retraining —
    which in a global-optimisation loop is every iteration.  A displacement in
    Ångström is a question that transfers; a hand-tuned scale is not.
  - Measured at the window peak rather than the first step: at ``t -> 1`` the
    samples are a noise gas whose gradient is both tiny and uninformative, and
    dividing by it would produce a scale that saturates ``max_step_size`` for
    the rest of the run.  Guidance is therefore zero on the rising edge, which
    is the region where the kernel is dead anyway.
  - The scale is then held fixed, so the per-structure spread survives:
    structures repelled harder than typical still move further, and ones that
    are already novel still barely move.
  - One calibrator is built per ``sample()`` call and shared across batches, so
    every sample in a run is driven at the same strength.  The result is
    printed after sampling and left on ``diffusion.novelty_calibrator``
    (``.guidance``, ``.gradient_scale``, ``.calibrated_at``) so it can be
    pinned explicitly for a reproducible rerun.
- Novelty guidance aborts with a clear ``RuntimeError`` when the feature
  gradient is non-finite (typically two atoms driven onto each other, making
  the interatomic unit vectors 0/0), and the resulting positions are checked
  with the samplers' ``_check_finite`` before the neighbour-list kernel sees
  them.
- ``Translator.translate_input()`` accepts an optional ``positions`` override,
  applied before the input modules so that position-derived quantities are
  recomputed from it.  This allows a backbone pass that is differentiable with
  respect to the atomic positions while reusing the batch's existing neighbour
  list.  Backends supply the position key via the new ``_set_positions()`` hook.
- ``Translator.extract_representation()`` returns a representation from a
  backbone output without storing it on the batch (unlike
  ``add_representation()``).
- ``sample()`` gained a ``cutoff`` parameter (default ``6.0``), now forwarded to
  the sampling call and used when featurising ``novelty_reference``.
- **`loss_balance` — relative weighting of the diffusion and force-field
  losses.**  `regressor_loss_weight` is an absolute multiplier whose useful
  value depends on the raw magnitude of the two losses, so it has to be
  re-tuned per system.  `loss_balance` instead takes the split you want
  (`"50:50"`, `"80:20"`, `(0.8, 0.2)`, or a bare number giving the regressor
  fraction) and divides each term by a running estimate of its own magnitude
  before applying the fractions:
  `loss = w_d · L_diffusion/s_d + w_r · L_regressor/s_r`.  Each term then
  contributes its requested share of the total regardless of scale, so the same
  setting transfers between systems.  Available on `create_diffusion()`,
  `train_from_atoms()`, the config, and `agedi train --loss_balance 80:20`.
  Unset by default, which keeps the existing absolute weighting exactly.
  - `s_d` / `s_r` are detached EMAs (`loss_balance_momentum`, default `0.99`)
    updated only on training batches, so validation loss stays comparable
    across epochs.
  - The achieved split is logged as `train/diffusion_fraction` and
    `train/regressor_fraction`.
  - Note that balancing makes the total loss O(1) regardless of the raw scales,
    which changes the effective learning rate and how `gradient_clip_val` bites
    relative to an unbalanced run.
- **`agedi.utils.loss_balance`** — `normalize_loss_balance()` accepting the
  string/number/pair/mapping forms above, plus `format_loss_balance()`.
- **`relax()` API and `agedi relax` command** — run the batched L-BFGS
  relaxation on structures you supply, independently of diffusion sampling.
  Each structure in a batch is optimised by its own L-BFGS instance and drops
  out as soon as its own maximum force falls below `fmax`, and ASE `FixAtoms`
  constraints on the input are honoured.  `trajectory=True` /
  `--save_trajectory` returns every optimiser step.
- **Inpainting-style diffusion** — `agedi.inpaint()` / `agedi inpaint` regenerate
  a chosen subset of atoms in an existing structure instead of generating a new
  one from scratch. Selected atoms are noised and denoised like a from-scratch
  atom; every other ("known") atom is, at each reverse-diffusion step, replaced
  by a fresh sample of the forward process `q(z_t | z_0)` of the input
  structure, so the whole batch stays at a self-consistent noise level and
  known atoms converge back onto their input positions (and species, for the
  types noiser) exactly as `t -> eps`.
  - Atom selection via `agedi.api.select_atoms()`: `indices`, `symbols`,
    `z_range`, `sphere`, or `from_atoms`, combined by union; with none given, a
    random `fraction` (default 0.25) of the non-fixed atoms is selected.
    `contiguous=True` changes the `fraction` fallback to grow a single
    spatially-connected cluster (a random seed atom, then repeatedly the
    closest remaining candidate) instead of a scattered random subset.
  - `t_start` below `1.0` starts from a partially-noised state for local
    rattle-and-relax refinement instead of full regeneration.
  - Optional RePaint-style resampling (`n_resample`, `jump_length`, off by
    default) to better harmonize the regenerated region with its surroundings.
  - `freeze` hard-freezes a subset of atoms in addition to the regenerated
    selection.
  - Implemented as `InpaintingSampler`, a wrapper around any existing sampler
    (`em`, `pc`, `heun`, `ddim`, `heun_ode`, `ffpc`), so it composes with all
    of them; not compatible with `compile=True`.
  - `atoms` accepts a list of structures (need not share atom count,
    composition, or cell) to batch several inpainting runs together for GPU
    throughput; every selection argument stays a single spec, re-resolved
    independently per structure, and `n_samples` becomes samples per
    structure. Results are grouped one list per input structure. The CLI
    picks this up automatically when the input file has more than one frame.
- `agedi.predict()` accepts the grouped `List[List[Atoms]]` shape
  `inpaint()` returns for a list of input structures (in addition to a flat
  list, unchanged), and returns predictions grouped the same way — so a
  multi-structure `inpaint(...)` result can be passed straight into
  `predict(...)` without flattening it first.
- `agedi.relax()` accepts the same grouped `List[List[Atoms]]` shape as
  `predict()` — nesting is auto-detected and the result is grouped the same
  way — so `inpaint() -> relax() -> predict()` chains without flattening at
  any step.
- `Noiser.forward_marginal()` / `Noiser.renoise()` hooks (implemented for the
  SDE-based position noisers and the discrete `Types` noiser) powering
  inpainting.
- `AtomsGraph.to_atoms()` writes the inpainting selection back as
  `atoms.arrays["inpaint_mask"]` when present, so a result can be re-fed as
  `from_atoms=True` input.
- **Conservative forces for force-field training** — `conservative_forces=True`
  on `create_diffusion()` / `train_from_atoms()` / the training config (or
  `agedi train --conservative_forces`) derives forces as `F = -dE/dR` by
  autograd through the energy head instead of using a dedicated forces head.
  This guarantees energy/force consistency and lets force labels also train
  the energy surface, at the cost of a backward pass on every regressor call
  (this also slows down force-field guided sampling and post-diffusion
  relaxation). Implemented in `agedi.models.regressor.RegressorModel`;
  disabled by default, so existing checkpoints and behaviour are unchanged.

### Fixed
- **Post-diffusion relaxation no longer breaks on periodic structures.**
  Positions are wrapped back into the cell after every step, so an atom
  crossing a cell face reappeared a full lattice vector away; the L-BFGS step
  sizer reconstructed its displacement by differencing stored positions and so
  recorded that jump as a history pair.  A single such pair sent the search
  direction somewhere unrelated to the forces, and because the history holds
  100 pairs it corrupted the rest of the run — the energy rose instead of
  falling.  Displacements are now taken in the minimum-image convention.
  Relaxing an 8-atom periodic Cu cell against exact EMT forces went from
  10.53 → 15.35 eV (diverging) to 10.53 → 7.60 eV, matching
  `ase.optimize.LBFGS` to within float32 precision.  Affects
  `sample(max_extra_steps=...)` and force-field guidance; non-periodic systems
  were never affected.
- **`post_diffusion_relaxation_step` no longer perturbs converged structures.**
  Wrapping positions into the cell round-trips through fractional coordinates,
  which is not bit-exact even for a no-op wrap, so every already-converged
  structure in a batch picked up ~1e-7 Å of numerical drift each relaxation
  step it should have been skipping (`active=False`).  `AtomsGraph.wrap_positions()`
  now takes an optional per-atom mask and leaves unmasked atoms' positions
  untouched instead of round-tripping them; caught by
  `TestPerStructureConvergence::test_inactive_structures_do_not_move`, which
  was intermittently failing in CI.

### Changed
- **Novelty guidance stability pass.**  Six changes to how the repulsion is
  scaled and scheduled; all of them alter behaviour, so an existing
  ``guidance`` value needs recalibrating (see the note below).
  - The step is now scaled by ``dt``, so ``guidance`` is a property of the
    trajectory rather than of its discretisation.  Previously the accumulated
    bias grew linearly with ``steps``, and a value tuned at 200 steps was 2.5x
    too strong at 500.
  - ``schedule="gaussian"`` is the new default time weight,
    ``exp(-(t - t_center)**2 / (2 * t_width**2))`` with ``t_center=0.5`` and
    ``t_width=0.2``.  The old front-loaded ``t**zeta`` is still available as
    ``schedule="power"``.  A bell is the right shape because the guidance is
    only meaningful in a window: at ``t -> 1`` the samples are a noise gas
    whose features sit far from every archive entry, so the kernel is dead and
    the in-batch term merely amplifies noise, while at ``t -> 0`` the basin is
    already committed and repulsion only distorts a finished geometry.
  - ``sigma`` now defaults to ``None``, meaning *calibrate it against the
    archive*: it is set to the ``sigma_quantile`` (default ``0.05``) quantile
    of the archive's own pairwise feature distances.  A fixed bandwidth is a
    guess in a feature space that is rebuilt on every retraining.  The resolved
    value and the archive's 1/5/50% distance quantiles are printed in the
    sampling-configuration panel.  New: ``FeatureArchive.distance_quantiles()``
    and ``agedi.diffusion.resolve_novelty_config()``.
  - ``max_step_size`` is applied as one rescaling per structure instead of a
    per-atom clip.  The pooled-feature gradient is typically concentrated on a
    handful of atoms, so per-atom clipping shortened only those and sheared the
    structure; a single factor bounds the magnitude while keeping the step
    parallel to the gradient.  Fixed template atoms are excluded from that
    maximum — their displacement is discarded anyway, but they carry a
    gradient and would otherwise shrink the step of the atoms that do move.
  - ``normalize_density`` (new, on by default) divides each structure's
    repulsion by its own kernel sum, clamped below at ``1.0``.  Without it the
    gradient grows with the density of the archive, so the same ``guidance``
    becomes steadily more aggressive as a global-optimisation campaign fills
    the archive up.  The denominator is detached, and the clamp means a sample
    far from everything is untouched.
  - In-batch pairs are now counted once, like archive pairs.  The double sum
    over the batch visits every pair twice, so ``include_batch=True`` silently
    made in-batch repulsion twice as strong as the archive term and the two
    could not be balanced.  New ``batch_weight`` sets their relative weight
    explicitly.
- Post-diffusion relaxation now evaluates the force field **once** per step
  instead of twice — the convergence check's forces are reused by the next
  step, halving the cost of relaxation.
- Relaxation convergence is tracked **per structure** rather than across the
  whole batch: a structure that reaches `force_threshold` stops being stepped
  instead of continuing until the worst structure in the batch converges.

### Notes
- Novelty guidance costs roughly one extra score-model forward *and* backward
  per reverse step (~2x measured), and is incompatible with ``compile=True``
  (a clear ``ValueError`` is raised).
- Recalibrating ``guidance``: prefer not to.  Set ``guidance=None`` and pick a
  ``target_displacement``, and the scale is derived per run.  If you do keep an
  explicit number, the ``dt`` factor alone means the old value must be
  multiplied by roughly ``steps``, and the density normalisation and halved
  in-batch term reduce it further for dense archives.
- Features live in the backbone's activation space and are only comparable
  within one model generation.  ``FeatureArchive`` must be rebuilt after every
  retraining; passing ``novelty_reference`` to ``sample()`` does this
  automatically.

## [1.4.0] - 2026-08-11

### Added
- **Per-species reference energies for force-field training** — when
  `force_field=True`, the energy target is offset by per-species reference
  energies `E⁰_Z` so the network only learns the (much smaller) residual.
  By default (`reference_energies="auto"`) they are fitted from the training
  data by linear least squares on `E_total ≈ Σ_Z n_Z·E⁰_Z`; explicit values can
  be supplied as a mapping keyed by chemical symbol or atomic number, and
  `None` disables the offset.  The offset is applied inside
  `agedi.models.schnetpack.regressor_heads.Energy`, so predicted energies stay
  on the absolute scale of the training data and forces are unaffected.
  Available as `reference_energies` on `create_diffusion()` /
  `train_from_atoms()` / the training config, and as
  `agedi train --reference_energies` (`auto` | `none` | `Cu:-3.72,O:-4.95`).
- **`agedi.utils.reference_energies`** — `fit_reference_energies()`,
  `normalize_reference_energies()`, and helpers for the reference-energy table.
- **`regressor_loss_weight` is now reachable from the public API** — the weight
  balancing the force-field loss against the diffusion loss
  (`loss = diffusion_loss + regressor_loss_weight · regressor_loss`) existed on
  `Agedi` but could only be set by constructing the model by hand.  It is now a
  parameter of `create_diffusion()` and `train_from_atoms()`, a
  `regressor_loss_weight` config key, and `agedi train --regressor_loss_weight`.
  Default `1.0` (unchanged behaviour).
- Force-field settings (reference energies, force loss, regressor loss weight)
  are shown in the training run-configuration panel and stored in
  `hparams.yaml`.

### Changed
- **The force-field forces head is now trained with a Huber loss by default**
  (`force_loss="huber"`, `huber_delta=0.01` eV/Å) instead of MSE: the loss is
  quadratic below `huber_delta` and linear above, so a few large force labels
  cannot dominate the gradient.  Select `"mse"` or `"mae"` via `force_loss` on
  `create_diffusion()` / `train_from_atoms()` / the config, or
  `agedi train --force_loss`.
- `RegressorModel` gained `force_loss`, `huber_delta`, and `energy_loss`
  parameters plus `get_config()`; the loss configuration is now carried through
  `Agedi.get_hparams()` (as `regressor_kwargs`) so it survives a save/load
  cycle for shared-backbone regressors.

### Fixed
- `RegressorModel.loss` with `use_weighting=True` applied per-atom weights to
  the per-structure energy loss; energies now use per-structure weights.

## [1.3.2] - 2026-08-06

### Added
- **`save_corrector_frames`** on `sample()` / `functional.sample()` /
  `agedi sample --save_corrector_frames` — records every Langevin corrector
  sub-step in the saved trajectory, giving a complete frame-by-frame record of
  sampling.  Requires `save_trajectory` and a sampler that runs correctors
  (`"pc"`, `"ffpc"`, or `corrector_steps > 0`).  Off by default because it
  multiplies trajectory length by roughly the corrector count.

  With capture on, one outer diffusion step contributes `1 + corrector_steps`
  frames, so a run has `steps · (1 + corrector_steps) + 1` frames — plus
  `1 + terminal_steps` when ffpc terminal dynamics are active.
- `Sampler._capture_frame()` / `Sampler._reset_pending()` — frame-capture
  helpers on the sampler base class, so custom samplers can contribute
  sub-step frames to the trajectory.

### Changed
- **`LBFGSStepSizer` now mirrors `ase.optimize.LBFGS`.**  Verified to match
  ASE's trajectory to float64 precision (`7e-15` over 20 steps on an EMT
  cluster).  Three deviations were corrected:
  - The step limit now scales the **whole** displacement by
    `maxstep / longest_atom_step`, as ASE's `determine_step` does.  It
    previously rescaled each atom independently, which rotated the search
    direction instead of shortening the step — and with the old 0.1 Å cap it
    was engaging on nearly every relaxation step.
  - The inverse-Hessian seed `H0 = 1/alpha` is now constant, as in ASE.  It was
    being updated each step by a Barzilai-Borwein estimate, which made the step
    length oscillate.
  - Defaults now match ASE: `maxstep=0.2` (was 0.1), `memory=100` (was 10),
    `alpha=70.0`, `damping=1.0`.  Post-diffusion relaxation takes the full
    L-BFGS step; it previously scaled every step by 0.1, roughly tenfold
    slowing convergence.

  `LBFGSStepSizer(memory_size=..., initial_step=...)` becomes
  `LBFGSStepSizer(memory_size=..., maxstep=..., alpha=..., damping=...)`.

### Fixed
- `ffpc` terminal dynamics ignored the z-confinement slab, in both
  `overdamped` and `langevin_md` modes.  They write `batch.pos` directly and so
  never inherited the clamp that `PositionsNoiser._denoise` applies, letting
  atoms drift out of the slab during the terminal phase.  (The diffusion steps,
  correctors, force-field guidance and post-diffusion relaxation were all
  unaffected.)  Atoms reaching a wall now have their z-velocity reflected
  rather than only clamped, so they bounce instead of staying pinned to the
  boundary for the rest of the run.
- `BatchedLBFGSStepSizer.compute_step` assigned steps to the wrong structures
  when any graph in the batch had no atoms: results were collected into a list
  and re-indexed by list position rather than graph id, shifting every
  subsequent graph's step onto its neighbour.
- `save_trajectory` no longer appends a duplicate final frame when
  post-diffusion relaxation ran; the relaxation loop already captured that
  state.
- Post-diffusion relaxation is no longer gated on `guidance > 0`.
  `ForcefieldGuidanceConfig(guidance=0.0, max_extra_steps=200)` previously did
  nothing at all; `max_extra_steps` now enables relaxation on its own, so the
  final structures can be relaxed without guidance perturbing the diffusion
  trajectory.  A persistent L-BFGS step sizer is allocated in that case too —
  without one, each relaxation step built a throwaway sizer and no curvature
  history accumulated.
- `ForcefieldCorrectorSampler` now emits a `UserWarning` when the diffusion
  model has no regressor (forces) head.  Previously `sampler="ffpc"` on a
  score-only model silently dropped both the terminal dynamics and the
  force-field blending in the corrector, degrading to plain
  predictor-corrector sampling; the only symptom was a saved trajectory
  missing all `terminal_steps` frames.  The warning names the number of
  terminal steps being skipped.  The separate "temperature not set" warning is
  suppressed in that case, since temperature only affects the terminal phase
  that will not run.

## [1.3.1] - 2026-07-02

### Fixed
- ``ForcefieldCorrectorSampler``: passing ``temperature=None`` (or omitting it)
  now emits a :class:`UserWarning` and falls back to ``1.0`` instead of raising
  a cryptic ``TypeError`` inside the terminal dynamics methods.

## [1.3.0] - 2026-07-02

### Added
- **Pluggable sampler architecture** — reverse-diffusion algorithm is now
  selectable via `sampler` parameter on `sample()` / `functional.sample()` /
  `agedi sample --sampler`.  Built-in aliases:
  - `"em"` — Euler–Maruyama (default, unchanged behaviour)
  - `"pc"` — predictor-corrector: EM predictor + N Langevin corrector steps at
    t_{i-1} (conventional PC convention)
  - `"heun"` — 2nd-order stochastic sampler (Karras et al., 2022); two score
    evaluations per step
  - `"ddim"` — deterministic probability-flow ODE (Anderson, 1982 / Ho et al.
    DDIM); no noise, one score evaluation per step
  - `"heun_ode"` — 2nd-order deterministic ODE (Heun's method on the PF-ODE)
  - `"ffpc"` — force-field augmented predictor-corrector (see below)
- **`ForcefieldCorrectorSampler` (`"ffpc"`)** — predictor-corrector sampler
  that blends the neural score with force-field gradients in the corrector:
  `s̃ = (1-f(t))·s_θ + f(t)·F`, where `f(t) = (1-t)^ζ`.  Optionally runs
  additional terminal dynamics after the last diffusion step:
  - `terminal_dynamics="overdamped"` — overdamped Langevin with reduced step
    `ε = terminal_step_size` (T-independent stability, correct Boltzmann
    distribution `∝ exp(-U/T)` via `sqrt(2εT)` noise)
  - `terminal_dynamics="langevin_md"` — standard Langevin MD (BAOAB integrator)
    with real ASE atomic masses and Maxwell-Boltzmann velocity initialisation
  - `terminal_step_size` and `terminal_friction` default to `None` and
    auto-select sensible values (`ε=1e-3` / `dt=1.0 fs`, `γ·dt=0.1`)
- **`Sampler.register()`** class method for registering custom sampler
  algorithms by string alias.
- **`sampler_kwargs` validation** — unknown keys in `sampler_kwargs` raise
  `ValueError` immediately, naming the offending key (catches typos such as
  `"ffpc_terminal_steps"` that were previously silently ignored).
- **Terminal step frames in `save_trajectory`** — when `ffpc` terminal steps
  are active, saved trajectories include a bridge frame (the denoised structure
  before terminal dynamics) followed by all terminal step frames.  Total frame
  count: `steps + 1 + terminal_steps`.

### Changed
- Temperature (`temperature`) is **not** applied inside the ffpc corrector
  blended score.  Dividing forces by a small T caused numerical blowup; the
  force-field contribution is controlled via `mixing_zeta` and
  `corrector_step_size` instead.
- `PositionsNoiser` hparam key renamed from `"sampler"` to `"denoising_step"`
  to avoid confusion with the new top-level `sampler` parameter.  Old
  checkpoints with `"sampler"` in `hparams.yaml` are loaded transparently via
  a backward-compatibility alias.

### Fixed
- `save_trajectory` no longer appends a duplicate trailing frame when terminal
  dynamics (ffpc) already captured the final state via `_pending_frames`.
- `sampler=em` no longer appeared spuriously under "positions noisers" in the
  architecture summary when a different top-level sampler was selected.

## [1.2.0] - 2026-06-12

### Added
- Non-periodic (gas-phase) training and sampling via `fully_connected=True` in
  `create_diffusion` / `train_from_atoms`.  Builds a fully connected graph at
  every reverse step so that atom pairs are never missed as molecules spread
  during diffusion.
- `Positions` noiser for non-periodic systems: uses zero-center-of-mass
  (`ZeroComNormal` / `ZeroComStandardNormal`) distributions by default,
  projecting noise onto the translationally-invariant subspace (Hoogeboom et
  al., NeurIPS 2022).
- `prediction_type` parameter (`"score"` / `"epsilon"`) on `PositionsNoiser`,
  `create_diffusion`, and `train_from_atoms`.  Epsilon prediction is the
  recommended choice with VP-SDE (uniform gradient magnitude across noise
  levels).
- `sampler` parameter (`"em"` / `"ddpm"`) for the reverse-diffusion update
  rule.  DDPM posterior-mean step (Ho et al., NeurIPS 2020) is available with
  `prediction_type="epsilon"` and is more stable than EM for large `beta_max`.
- `StandardNormal.scale` — replaces the hard-coded `0.8·N^(1/3)` prior
  heuristic with an SDE-derived scale set automatically to `sqrt(var(T))`.

### Fixed
- Cosine noise-schedule `fint` had a factor-of-2 error in the argument of
  `sin`; corrected to `sin(π·t)`.
- VP-SDE reverse drift sign was wrong in the Euler–Maruyama denoising step.
- VP-SDE default parameters updated to `beta_min=0.1`, `beta_max=20.0`
  (standard DDPM values).
- `cell_to_cellpar` no longer produces NaN for zero-cell (non-periodic) graphs.
- `wrap_positions` is now skipped when `pbc=[False, False, False]`.
- NVIDIA neighbor-list backend is bypassed when `pbc` is all-False, avoiding
  incorrect results for non-periodic systems.

## [1.1.0] - 2026-06-03

### Added
- Add `pbc` parameter to sampling API

### Fixed
- `pbc` propagation in `_initialize_graph`

## [1.0.2] - 2026-06-02

### Fixed
- `Positions` noiser crashed during sampling with
  `RuntimeError: batch_idx length (N) does not match num_atoms (0)`.
  `StandardNormal._setup()` now reads the leading dimension from
  `batch.n_atoms.sum()` instead of `batch[key].shape[0]`, consistent with
  `Constant._setup()` and `UniformCell._setup()`.  The old code read the
  shape from the (empty) `pos` tensor that `AtomsGraph.empty()` initialises
  before the prior has a chance to populate it.

## [1.0.0] - 2026-05-22

### Added
- Predictor-corrector (Langevin corrector) sampling via `corrector_steps` /
  `corrector_step_size` parameters.
- Force-field guided sampling (`ForcefieldGuidanceConfig`, `--ff_guidance` /
  `--ff_zeta` CLI flags, `force_field=True` training option).
- Post-diffusion relaxation loop (auto-triggered when `ff_guidance` is enabled
  and forces exceed `force_threshold`).
- `torch.compile` support for the reverse diffusion step (`compile=True` /
  `--compile`), compiled per-instance to avoid cross-model interference.
- `agedi predict` CLI command and `functional.predict()` API for energy/force
  inference with trained regressor heads.
- `agedi inspect` CLI command for inspecting saved model checkpoints.
- `agedi train-hydra` CLI command and `train_from_config()` API supporting a
  full YAML config file (`conf/train.yaml`).
- Resume training from checkpoint (`--checkpoint` / `checkpoint:` config key).
- Separate `regressor_data_path` support in config-file training for
  non-equilibrium regressor data.
- `Noiser.register()` class method for registering custom noisers.
- `register_model()` function for registering custom GNN backbone factories.
- `save_trajectory` parameter on `sample()` / `functional.sample()` (replaces
  the old `save_path` parameter which has been removed).
- `print_timings` parameter for per-stage sampling timing breakdown.
- Skin-based neighbor-list caching in `update_graph()` to skip full rebuilds
  when atoms have moved less than the skin distance.
- `canonical_cell` option in `from_atoms()` / CLI / config to store cells in
  canonical lower-triangular form.
- `n_classes` option to explicitly set the number of atom-type classes for the
  `Types` noiser.
- `repeat` / `repeat_epoch` cell-repeat data augmentation.
- `batch_naive_neighbor_list` pure-PyTorch fallback neighbor-list (used in the
  compiled path when `nvalchemiops` is not available).

### Changed
- **Breaking**: `save_path` parameter removed from `Diffusion.sample()`,
  `Diffusion._sample()`, and `functional.sample()`.  Use `save_trajectory`
  instead.
- `nvalchemiops` is now an **optional** dependency (`pip install agedi[cuda]`).
  The non-compiled sampling path works without it.
- `Agedi.__init__` `optim_config` and `scheduler_config` default to `None`
  (computed internally) rather than shared mutable dicts.
- `AtomsGraph.from_atoms` cell-canonicalization now emits `warnings.warn`
  instead of `print`.
- Compiled reverse step is now a lazy per-instance property instead of a
  class-level `@torch.compile` decorator.
- CI matrix extended to macOS runners; `.[test,full]` installed in one step.

### Fixed
- Python version badge corrected to `3.12+`.
- `AtomsGraph` class docstring typo corrected.
- `from_atoms` `canonical_cell` parameter docstring corrected (default is
  `False`, not `True`).
