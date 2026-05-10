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
     - PriorDistribution
     - NoiseDistribution
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

Continue training from a checkpoint
-------------------------------------

To resume an interrupted run or continue fine-tuning on new data, pass
``--checkpoint`` with either a run directory or a specific ``.ckpt`` file:

.. code-block:: console

   # Resume the last checkpoint of a previous run (same data)
   agedi train training_data.traj --checkpoint logs/version_0

   # Resume from a specific checkpoint file
   agedi train training_data.traj --checkpoint logs/version_0/checkpoints/best_model.ckpt

   # Fine-tune on new data starting from a previous checkpoint
   agedi train new_data.traj --checkpoint logs/version_0

In all cases the model architecture and weights are loaded from the checkpoint,
and the full training state (optimiser, LR-scheduler, epoch counter) is
restored.  Combine with ``--epochs`` or ``--max_time`` to control how long
the continued run should train.

When using a config file, set the ``checkpoint`` key:

.. code-block:: yaml

   data_path: training_data.traj
   checkpoint: logs/version_0   # or a .ckpt file path

From Python:

.. code-block:: python

   from agedi import train_from_atoms

   diffusion, dataset, trainer = train_from_atoms(
       data,
       checkpoint="logs/version_0",
       epochs=100,
   )


Sampling
--------

.. code-block:: console

   agedi sample logs/version_0 -f Pd2O2 --template_path template.traj --steps 500 --confinement 2 10

This samples using the ``last_model.ckpt`` checkpoint found in
``logs/version_0``. If you want to use a different checkpoint, you can
specify the exact path to it.

Important options:

- ``-f/--formula`` or ``-a/--n_atoms``
- ``--template_path`` for template-guided generation
- ``--steps``, ``--eps`` for reverse diffusion resolution

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

   agedi sample logs/version_0 -f Pd2O2 --ff_guidance 5.0

- ``--ff_guidance``: guidance scale (``0`` = disabled, ``> 0`` enables guidance).
  Higher values increase the influence of the predicted forces on the generated structures.
- ``--ff_zeta``: time-weight exponent (default ``3.0``).
  Higher values concentrate guidance near the end of the reverse trajectory.

In Python this is equivalent to:

.. code-block:: python

   from agedi.functional import load_diffusion, sample
   from agedi.diffusion import ForcefieldGuidanceConfig

   diffusion = load_diffusion("logs/version_0")
   structures = sample(
       diffusion,
       n_samples=10,
       formula="Pd2O2",
       ff_guidance=ForcefieldGuidanceConfig(guidance=5.0, zeta=3.0),
   )

Predicting energies and forces
-------------------------------

When the model has been trained with ``--force_field``, you can run energy
and force predictions on existing structures with ``agedi predict``:

.. code-block:: console

   agedi predict logs/version_0 structures.traj

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

   diffusion = load_diffusion("logs/version_0")
   structures = read("structures.traj", index=":")
   predicted = predict(diffusion, structures)
   write("predicted.traj", predicted)

Inspect run metadata
--------------------

.. code-block:: console

   agedi inspect logs/version_0

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

