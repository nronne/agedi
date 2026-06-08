# Changelog

All notable changes to AGeDi will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-06-08

### Added

- **Fully-connected graph support** (`fully_connected=True` in `create_diffusion`,
  `train_from_atoms`, `create_dataset`, `AtomsGraph.from_atoms`, `AtomsGraph.empty`).
  Builds an all-pairs graph (no self-loops, zero shift vectors) suitable for
  gas-phase molecules and clusters where a finite cutoff would miss long-range
  pairs as atoms spread during sampling.  Edges are built once, cached under
  internal store keys that survive `pos` assignment, and restored on every
  `update_graph` call — no neighborlist timing overhead for FC graphs.

- **VP-SDE epsilon-prediction** (`prediction_type="epsilon"` on position noisers).
  The network predicts the normalised noise `ε` directly; loss gradient is
  uniform across all noise levels (no `var(t)` attenuation), which is essential
  for VP-SDE.  During sampling the score is recovered as `s = −ε / √var(t)`.

- **DDPM posterior-mean sampler** (`sampler="ddpm"`, requires `prediction_type="epsilon"`).
  Replaces the Euler–Maruyama update with the Ho et al. (NeurIPS 2020) posterior-mean
  step `x_{t−Δt} = (x_t − β·Δt/√var · ε) / √(1−β·Δt) + σ_t·z`, whose denominator
  cancels per-step amplification that destabilises EM at large `beta_max`.

- **Min-SNR-γ loss weighting** (`loss_weighting="min_snr"` on position noisers).
  Caps the per-sample loss weight at `min(SNR, 5)` following Hang et al.
  (ICCV 2023, arXiv:2303.09556), reducing the gradient variance at low noise.

- **EDM preconditioning** (`precondition=True`, `sigma_data` in `create_diffusion` /
  `train_from_atoms`).  Wraps the `PositionsScore` head output with σ-dependent
  skip and scale factors (Karras et al., NeurIPS 2022) so the network always
  operates on unit-scale inputs and outputs.

- **Zero-COM distributions** (`ZeroComNormal`, `ZeroComStandardNormal`).
  Drop-in replacements for `Normal` / `StandardNormal` that project the noise
  increment onto the translationally-invariant subspace per graph (Hoogeboom et
  al., NeurIPS 2022).  `Positions` auto-uses `ZeroComStandardNormal` as its
  prior, with scale set to `sqrt(var(T=1))` so the prior matches the
  forward-process marginal — replacing the old `0.8·N^{1/3}` heuristic.

- **Selectable radial basis for PaiNN** (`radial_basis="gaussian"` (default) or
  `"bessel"` in `create_diffusion` / `train_from_atoms`).  Bessel RBFs offer
  better short-range resolution for small cutoffs; Gaussian remains the safe
  default for large-cutoff / FC-graph use.

- **`cutoff` auto-resolution**: passing `cutoff=None` (new default) chooses
  50 Å when `fully_connected=True` and 6 Å otherwise, ensuring the backbone
  RBFs cover the full distance range without requiring an explicit override.

- **`fully_connected` flag persisted on `Agedi` model** and propagated
  automatically to `sample()` — models saved and reloaded with
  `fully_connected=True` sample with the correct topology without requiring
  the user to pass the flag again.

- **Fix VP reverse-SDE drift sign** in `_denoise`.  The drift term was added
  instead of subtracted in the EM update, producing incorrect denoising
  trajectories for VP-SDE.

- **Fix Cosine noise-schedule `fint` formula** and reset VP-SDE default
  parameters to literature values (`beta_min=0.1`, `beta_max=20.0`).

- **Fix `pos_sigma` KeyError** during sampling.  `PositionsNoiser.initialize_graph`
  now seeds `pos_sigma` with `√var(t=1)` so EDM-preconditioned heads have
  the correct σ from the very first score evaluation.

- **Selectable radial basis preserved through model serialisation**:
  `SchNetPackTranslator.get_representation_hparams` now reads the cutoff from
  `cutoff_fn` (always present) instead of `rb.offsets` (absent on BesselRBF),
  so `load_diffusion` round-trips correctly for both RBF types.

- **Fix post-diffusion L-BFGS relaxation divergence** (PR #73):
  - Reset L-BFGS memory before relaxation to discard stale curvature history
    accumulated during the noisy diffusion trajectory.
  - Add `max_step_size` clamp (0.1 Å) to `post_diffusion_relaxation_step`,
    matching the clamp already present in `force_field_guidance_step`.

### Changed

- `cutoff` parameter in `create_diffusion`, `train_from_atoms`, and
  `create_dataset` is now `Optional[float]` defaulting to `None` (was `6.0`).
  Existing calls with an explicit cutoff are unaffected.

- `StandardNormal` prior for `Positions` noiser now accepts a `scale` parameter
  and uses an SDE-consistent scale (`sqrt(var(T=1))`) instead of the old
  `0.8·N^{1/3}` heuristic.  Existing models are unaffected; the prior is
  sampled only during the forward pass.

- VP-SDE default parameters updated to literature values:
  `beta_min=0.1` (was `0.01`), `beta_max=20.0` (was `3`).

- `CellPositions` and `ConfinedCellPositions` now accept explicit `distribution`
  and `prior` keyword arguments, enabling custom noise or prior distributions
  without subclassing.

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
