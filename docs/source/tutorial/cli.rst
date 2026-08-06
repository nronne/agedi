Command-line interface
======================

AGeDi installs a CLI entrypoint named ``agedi``.

Discover commands
-----------------

.. code-block:: console

   agedi --help

Main commands
-------------

- ``agedi train``: train a diffusion model from a trajectory file or YAML config
- ``agedi sample``: sample structures from a saved training run
- ``agedi predict``: predict energies and forces for input structures (requires ``--force_field`` training)
- ``agedi inspect``: print ``hparams.yaml`` from a run directory

To get information about options for each use

.. code-block:: console

   agedi train --help

for ``train`` and likewise for ``sample``, ``predict``, and ``inspect``.

Training
--------

Choose one of the three position noisers to match your system type:

.. list-table:: Position noisers
   :header-rows: 1
   :widths: 30 25 25 20

   * - Noiser
     - Prior
     - Distribution
     - Use case
   * - ``Positions``
     - StandardNormal
     - Normal
     - Gas-phase (molecules, clusters)
   * - ``CellPositions``
     - UniformCell
     - Normal
     - Periodic bulk / surface (default)
   * - ``ConfinedCellPositions``
     - UniformCellConfined
     - TruncatedNormal
     - Surface overlayer/adsorbate

Minimal training example for a **surface system with Z-confinement**:

.. code-block:: console

   agedi train --noisers ConfinedCellPositions --mask MaskFixed --confinement 2 10 training_data.traj

Minimal training example for a **periodic bulk or surface** system:

.. code-block:: console

   agedi train --noisers CellPositions training_data.traj

Minimal training example for a **periodic bulk system with atomic types diffusion**:

.. code-block:: console

   agedi train --noisers CellPositions,Types training_data.traj

Minimal training example for a **gas-phase cluster**:

.. code-block:: console

   agedi train --noisers Positions training_data.traj

Important options:

- ``--max_time_minutes/-t`` or ``--epochs/-e``: stopping criteria (use ``-T`` to specify time in hours instead of minutes)
- ``--noisers``: ``CellPositions`` (default), ``ConfinedCellPositions``, ``Positions``, ``Types``.
  Accepts a comma-separated list to specify multiple noisers in one flag (e.g. ``--noisers ConfinedCellPositions,Types``),
  or repeat the flag (e.g. ``--noisers ConfinedCellPositions --noisers Types``).
- ``--sde``: ``ve`` (default), ``vp``
- ``--mask MaskFixed``: freezes atoms tagged with ASE ``FixAtoms``
- ``--confinement zmin zmax``: z-direction confinement bounds (required for ``ConfinedCellPositions``)
- ``--n_classes N``: restrict the ``Types`` noiser vocabulary to the first *N* element types
  (sorted by atomic number); defaults to all distinct types found in the training data
- ``--canonical_cell``: store unit cells in canonical lower-triangular form
- ``--force_field``: train a force-field head jointly with the diffusion score (see below)

Continue training from a checkpoint
-------------------------------------

To resume an interrupted run or continue fine-tuning on new data, pass
``--checkpoint`` with either a run directory or a specific ``.ckpt`` file:

.. code-block:: console

   # Resume the last checkpoint of a previous run (same data)
   agedi train training_data.traj --checkpoint logs/agedi/version_0

   # Resume from a specific checkpoint file
   agedi train training_data.traj --checkpoint logs/agedi/version_0/checkpoints/best_model.ckpt

   # Fine-tune on new data starting from a previous checkpoint
   agedi train new_data.traj --checkpoint logs/agedi/version_0

In all cases the model architecture and weights are loaded from the checkpoint,
and the full training state (optimiser, LR-scheduler, epoch counter) is
restored.  Combine with ``--epochs`` or ``--max_time`` to control how long
the continued run should train.

When using a config file, set the ``checkpoint`` key:

.. code-block:: yaml

   data_path: training_data.traj
   checkpoint: logs/agedi/version_0   # or a .ckpt file path

From Python:

.. code-block:: python

   from agedi import train_from_atoms

   diffusion, dataset, trainer = train_from_atoms(
       data,
       checkpoint="logs/agedi/version_0",
       epochs=100,
   )


Sampling
--------

.. code-block:: console

   agedi sample logs/agedi/version_0 -f Pd2O2 --template_path template.traj --steps 500 --confinement 2 10

This samples using the ``last_model.ckpt`` checkpoint found in
``logs/agedi/version_0``. If you want to use a different checkpoint, you can
specify the exact path to it.

Important options:

- ``-f/--formula`` or ``-a/--n_atoms``
- ``--template_path`` for template-guided generation
- ``--steps``, ``--eps`` for reverse diffusion resolution
- ``--save_trajectory``: save the full reverse-diffusion trajectory for each sample
  (one file per sample rather than only the final structures)
- ``--save_corrector_frames``: with ``--save_trajectory``, also save every Langevin
  corrector sub-step, giving a complete frame-by-frame record.  Only has an effect
  with a sampler that runs correctors (``--sampler pc`` or ``ffpc``); multiplies
  trajectory length by roughly the corrector count
- ``--print_timings``: print a per-stage timing breakdown after each sampling batch
  (useful for profiling GPU bottlenecks)
- ``--compile``: compile the reverse-diffusion step with ``torch.compile`` for faster
  GPU sampling; neighbor-list buffer sizes are estimated automatically (requires
  NVIDIA nvalchemiops)

Choosing a sampler
~~~~~~~~~~~~~~~~~~

Use ``--sampler`` to select the reverse-diffusion algorithm:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``--sampler``
     - Description
   * - *(default)*
     - Euler–Maruyama (EM): one score evaluation per step
   * - ``em``
     - Euler–Maruyama (explicit alias)
   * - ``pc``
     - Predictor-corrector: EM predictor + Langevin corrector steps at t_{i-1}
   * - ``heun``
     - 2nd-order stochastic (Karras et al. 2022): two score evaluations per step,
       often gives better quality at fewer steps
   * - ``ddim``
     - Deterministic probability-flow ODE: no noise, fully reproducible; one score
       evaluation per step
   * - ``heun_ode``
     - 2nd-order deterministic ODE (Heun's method): two score evaluations per step
   * - ``ffpc``
     - Force-field augmented predictor-corrector (requires ``--force_field``
       training); see below

Example:

.. code-block:: console

   agedi sample logs/agedi/version_0 -f Pd2O2 --sampler heun --steps 200

Force-field augmented sampling (``ffpc``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the model has been trained with ``--force_field``, the ``ffpc`` sampler
blends the neural score with the predicted forces during the corrector phase,
and can optionally run additional Langevin dynamics after the last diffusion
step:

.. code-block:: console

   # EM predictor + force-field augmented Langevin corrector
   agedi sample logs/agedi/version_0 -f Pd2O2 --sampler ffpc

   # With overdamped terminal steps for extra relaxation
   agedi sample logs/agedi/version_0 -f Pd2O2 --sampler ffpc \
       --ffpc_terminal_steps 200 --ffpc_temperature 0.026

   # With standard Langevin MD terminal steps (BAOAB, real atomic masses)
   agedi sample logs/agedi/version_0 -f Pd2O2 --sampler ffpc \
       --ffpc_terminal_steps 500 --ffpc_terminal_dynamics langevin_md \
       --ffpc_temperature 0.026

Key ``ffpc`` options:

- ``--ffpc_corrector_steps`` (default ``1``): Langevin corrector iterations per
  diffusion step.  Set to ``0`` to disable the corrector (pure EM + terminal steps only).
- ``--ffpc_corrector_step_size`` (default ``1e-3``): step size for the corrector.
- ``--ffpc_zeta`` (default ``1.0``): exponent for the force-field mixing schedule
  ``f(t) = (1-t)^ζ``; higher values concentrate force-field influence near the end.
- ``--ffpc_temperature`` (default ``1.0``): temperature *T* for the terminal phase.
  For overdamped this scales the noise amplitude; for ``langevin_md`` this is
  k\ :sub:`B`\ T in the same units as model forces (e.g. eV).
- ``--ffpc_terminal_steps`` (default ``0``, disabled): number of post-diffusion
  terminal dynamics steps.
- ``--ffpc_terminal_dynamics`` (``overdamped`` / ``langevin_md``): dynamics mode for
  the terminal phase.  ``overdamped`` uses gradient-based Langevin without momenta;
  ``langevin_md`` uses the BAOAB integrator with real ASE atomic masses and
  Maxwell-Boltzmann velocity initialisation.
- ``--ffpc_terminal_step_size``: step size for the terminal phase.  Auto-selected
  when not specified (``1e-3`` for overdamped, ``1.0`` fs for langevin_md).
- ``--ffpc_terminal_friction``: friction coefficient γ for ``langevin_md``
  (units: 1/terminal_step_size).  Auto-selected when not specified (γ·dt = 0.1).

Force-field guided training and sampling
-----------------------------------------

To also train a forces prediction head alongside the diffusion model, add the
``--force_field`` flag during training:

.. code-block:: console

   agedi train --noisers ConfinedCellPositions --mask MaskFixed --confinement 2 10 --force_field training_data.traj

The training data must contain DFT (or other source) per-atom forces
and total energy (e.g. loaded from a
VASP/GPAW calculation via ASE).  The force field is trained jointly with the
diffusion score.

**Regressor-only dataset**

You can optionally supply a second dataset that is used *exclusively* to train
the force-field head — its structures are never passed through the diffusion
loss.  This is useful when you have non-equilibrium structures (e.g. from MD
or NEB calculations) that would be unsuitable as diffusion training targets
but contain valuable force/energy information for the regressor:

.. code-block:: console

   agedi train --noisers ConfinedCellPositions --mask MaskFixed --confinement 2 10 --force_field training_data.traj

and in ``train.yaml``:

.. code-block:: yaml

   data_path: training_data.traj
   force_field: true
   regressor_data_path: nonequilibrium_data.traj

Or from Python:

.. code-block:: python

   from agedi import train_from_atoms

   diffusion, dataset, trainer = train_from_atoms(
       equilibrium_structures,
       force_field=True,
       regressor_data=nonequilibrium_structures,
   )

Once training is complete, force-field guidance can be used during sampling
via the ``--ff_guidance`` option:

.. code-block:: console

   agedi sample logs/agedi/version_0 -f Pd2O2 --ff_guidance 5.0

- ``--ff_guidance``: guidance scale (``0`` = disabled, ``> 0`` enables guidance).
  Higher values increase the influence of the predicted forces on the generated structures.
- ``--ff_zeta``: time-weight exponent (default ``3.0``).
  Higher values concentrate guidance near the end of the reverse trajectory.

In Python this is equivalent to:

.. code-block:: python

   from agedi.functional import load_diffusion, sample
   from agedi.diffusion import ForcefieldGuidanceConfig

   diffusion = load_diffusion("logs/agedi/version_0")
   structures = sample(
       diffusion,
       n_samples=10,
       formula="Pd2O2",
       ff_guidance=ForcefieldGuidanceConfig(
           guidance=5.0,
           zeta=3.0,
           force_threshold=0.05,   # max per-atom force (eV/Å) for post-diffusion relaxation
           max_extra_steps=0,      # number of extra relaxation steps after the trajectory
       ),
   )

``ForcefieldGuidanceConfig`` fields:

- ``guidance`` (float): guidance scale; ``0.0`` disables guidance entirely.
- ``zeta`` (float): time-weight exponent ``(1-t)**zeta``; default ``3.0``.
- ``force_threshold`` (float): convergence criterion (max per-atom force in eV/Å)
  for the optional post-diffusion relaxation; default ``0.05``.
- ``max_extra_steps`` (int): maximum L-BFGS relaxation steps performed after the main
  diffusion trajectory; default ``0`` (disabled).

Relaxing without guidance
~~~~~~~~~~~~~~~~~~~~~~~~~~

``max_extra_steps`` is independent of ``guidance``.  Set it alone to relax the
final structures while leaving the diffusion trajectory itself untouched:

.. code-block:: python

   structures = sample(
       diffusion,
       n_samples=10,
       formula="Pd2O2",
       ff_guidance=ForcefieldGuidanceConfig(
           guidance=0.0,           # no guidance during diffusion
           max_extra_steps=200,    # L-BFGS relaxation afterwards
           force_threshold=0.05,   # stop early once forces drop below this
       ),
   )

Each step is one :class:`ase.optimize.LBFGS` step, applied to every structure
in the batch independently and with ASE's defaults (``maxstep=0.2 Å``,
``memory=100``, ``alpha=70``).  Relaxation stops as soon as the maximum
per-atom force falls below ``force_threshold``, and is skipped entirely when
the structures are already converged.  It requires a model with a regressor
(forces) head.  With ``save_trajectory=True`` the relaxation steps appear as
extra frames at the end of each trajectory.

Pick ``force_threshold`` to suit the model, not the DFT convention.  The
default ``0.05`` eV/Å assumes the regressor can resolve near-zero forces, which
requires relaxed structures in its training set.  A model trained only on
high-force configurations cannot predict forces below the range it has seen, so
relaxation will plateau above the threshold and always consume every step.
Compare against the force distribution of the training data before choosing a
value.  Note also that the ``Forces`` head predicts forces directly rather than
as :math:`-\partial E/\partial R`, so the predicted field is not conservative
and has no exact zero-force fixed point; over many steps the relaxation can
drift away from the lowest-energy structure it visited.

Note that guidance and the ``ffpc`` sampler both apply the same force field by
different routes — ``ffpc`` blends forces into the corrector score, guidance
takes a separate L-BFGS-sized step after each sampler step, each on its own
``zeta`` schedule.  Combining them compounds the force-field pull in a way
neither parameter's default accounts for; prefer one mechanism at a time.

Predicting energies and forces
-------------------------------

When the model has been trained with ``--force_field``, you can run energy
and force predictions on existing structures with ``agedi predict``:

.. code-block:: console

   agedi predict logs/agedi/version_0 structures.traj

This reads all structures from ``structures.traj``, runs the force-field
regressor, and writes the results (with predicted energies and forces
attached as an ASE ``SinglePointCalculator``) to ``predicted.traj`` in
the current directory.

Important options:

- ``-o/--output``: directory to save the output file (default: ``.``)
- ``--name``: base name for the output file (default: ``predicted``)
- ``-b/--batch_size``: number of structures per inference batch (default: ``64``)

In Python this is equivalent to:

.. code-block:: python

   from ase.io import read, write
   from agedi import load_diffusion, predict

   diffusion = load_diffusion("logs/agedi/version_0")
   structures = read("structures.traj", index=":")
   predicted = predict(diffusion, structures)
   write("predicted.traj", predicted)

Inspect run metadata
--------------------

.. code-block:: console

   agedi inspect logs/agedi/version_0

This prints the saved hyperparameters from the run directory (for example, the parsed contents of ``hparams.yaml``).

Logging options
--------------------

AGeDi saves TensorBoard logs by default. WandB can be saved instead
using the ``--logger wandb`` option when training.

To follow training use

.. code-block:: console

   tensorboard --logdir .

This hosts TensorBoard at localhost. Remember to forward a specific
port to your local machine if using HPC. You can use the ``--port
xxxx`` option for TensorBoard to host at this specific port.

