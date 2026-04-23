Pitfalls and troubleshooting
============================

Most common failure modes are due to missing required sampling inputs or
mismatched configuration between training and sampling.

Sampling input pitfalls
-----------------------

This error is raised when the z-coordinate mean passed to the
``TruncatedNormal`` distribution contains ``NaN`` values during a
reverse-diffusion (sampling) step.

**What causes it?**

When confinement is active (e.g. ``--confinement z_min z_max``),
mobile-atom z-coordinates are kept inside ``[z_min, z_max]`` by
sampling from a truncated normal distribution at each step.  If the
Euler-Maruyama update pushes a coordinate outside the confinement
bounds, the mean supplied to the next truncated-normal sample becomes
invalid, eventually producing ``NaN`` values.

Common triggers:

* **Step-size too large** — using a small number of reverse steps
  (``--steps``) means each step is large and more likely to overshoot
  the confinement boundary.  Try increasing ``--steps``.
* **Confinement mismatch** — if the ``--confinement`` bounds used
  during *sampling* differ from those used during *training*, the score
  model is evaluated outside its training distribution and can produce
  scores that drive atoms out of bounds.  Ensure training and sampling
  use the same confinement bounds.

**How is it handled in the current code?**

As of the current version, positions are clamped back to
``[z_min, z_max]`` after every reverse step, and the mean passed to the
truncated-normal distribution is also clamped before constructing the
distribution.  These two safeguards together mean this error should
only be triggered if a ``NaN`` arises from a different source (e.g. the
score model itself returning ``NaN`` gradients).  If you still see this
error, check the model output for numerical instabilities.
- **Missing ``cell`` without template**

  If you sample without ``template``, provide cell information directly or use
  a run whose ``hparams.yaml`` includes ``cell`` metadata.

- **Missing atom specification**

  You must provide enough information to infer generated atoms:
  ``formula``, or ``n_atoms`` + ``atomic_numbers`` as required by your noisers.

- **Type-only / position-only configurations**

  If no position noiser is active, fixed ``positions`` are required at sampling.
  If no type noiser is active, types must come from ``formula`` or ``atomic_numbers``.

Training pitfalls
-----------------

- **No SchNetPack installed for PaiNN workflows**

  Install with ``agedi[full]``.

- **Mask behavior misunderstood**

  ``MaskFixed`` only freezes atoms marked by ASE ``FixAtoms`` constraints.

- **Confinement mismatch**

  Keep training and sampling confinement bounds consistent for slab/surface tasks.

- **Repeat settings**

  If ``repeat`` is set, ``repeat_epoch`` must also be set.

Data pitfalls
-------------

- Ensure periodic cells/pbc are set consistently in ASE data.
- Extremely small datasets can yield unstable train/val splits and noisy metrics.
- Large cutoffs and batch sizes increase memory usage substantially.

Operational checks
------------------

- Inspect run config with ``agedi inspect <log_dir>``.
- Confirm ``hparams.yaml`` and checkpoints exist before loading.
- Start with smaller ``steps`` / ``batch_size`` to diagnose OOM issues.
