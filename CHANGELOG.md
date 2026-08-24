# Changelog

All notable changes to AGeDi will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
  - `t_start` below `1.0` starts from a partially-noised state for local
    rattle-and-relax refinement instead of full regeneration.
  - Optional RePaint-style resampling (`n_resample`, `jump_length`, off by
    default) to better harmonize the regenerated region with its surroundings.
  - `freeze` hard-freezes a subset of atoms in addition to the regenerated
    selection.
  - Implemented as `InpaintingSampler`, a wrapper around any existing sampler
    (`em`, `pc`, `heun`, `ddim`, `heun_ode`, `ffpc`), so it composes with all
    of them; not compatible with `compile=True`.
- `Noiser.forward_marginal()` / `Noiser.renoise()` hooks (implemented for the
  SDE-based position noisers and the discrete `Types` noiser) powering the
  above.
- `AtomsGraph.to_atoms()` writes the inpainting selection back as
  `atoms.arrays["inpaint_mask"]` when present, so a result can be re-fed as
  `from_atoms=True` input.

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
- Force-field settings (reference energies, force loss) are shown in the
  training run-configuration panel and stored in `hparams.yaml`.

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
