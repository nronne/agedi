# Changelog

All notable changes to AGeDi will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Fixed
- `save_trajectory` no longer appends a duplicate final frame when
  post-diffusion relaxation ran; the relaxation loop already captured that
  state.

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
