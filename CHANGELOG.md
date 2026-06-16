# Changelog

All notable changes to AGeDi will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Force-field guidance L-BFGS step sizer now resets after each predictor
  step during the main diffusion loop.  Previously the sizer accumulated
  history across all diffusion steps; because `s_k = pos_{k+1} - pos_k` is
  dominated by the denoiser's stochastic position update, the stored
  curvature reflected noiser noise rather than the energy landscape.  Near
  the end of diffusion (where guidance is strongest) this caused misdirected
  0.1 Å steps that blew up molecular structures.  The L-BFGS now only
  accumulates valid curvature history during post-diffusion relaxation, where
  positions change solely due to guidance steps.
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
