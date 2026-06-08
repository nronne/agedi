YAML-based training
===================

The ``agedi train`` command accepts either a trajectory file **or** a YAML
configuration file as its first argument, so you can choose whichever workflow
suits your project.

Quick start
-----------

1. Copy the bundled template::

      cp $(python -c "import agedi; import pathlib; print(pathlib.Path(agedi.__file__).parent / 'conf' / 'train.yaml')") my_train.yaml

2. Edit ``my_train.yaml`` (at minimum set ``data_path``).

3. Run training::

      agedi train my_train.yaml

4. Override individual keys without editing the file::

      agedi train my_train.yaml feature_size=128 epochs=200 noisers=CellPositions

Configuration file reference
-----------------------------

A fully annotated template is reproduced below. Every key has a sensible
default so you only need to set the values that differ from those defaults.

.. code-block:: yaml

   # ---------------------------------------------------------------------------
   # Data
   # ---------------------------------------------------------------------------
   data_path: /path/to/train.traj   # Required – ASE-readable file

   # Optional separate dataset used exclusively for regressor (force-field) training.
   # Structures here are only forwarded through the regressor loss, never the diffusion
   # loss.  Useful for non-equilibrium structures (e.g. from MD or NEB).
   regressor_data_path: null        # Optional – ASE-readable file

   # ---------------------------------------------------------------------------
   # Score-model architecture
   # ---------------------------------------------------------------------------
   model: PaiNN          # Currently only PaiNN is supported
   cutoff: null          # Cutoff in Å. null → 6 Å (or 50 Å when fully_connected: true)
   feature_size: 64      # Embedding / feature dimension
   n_blocks: 4           # Number of interaction blocks
   n_rbf: 30             # Number of radial basis functions

   # Radial basis type: gaussian (default, bounded in [0,1]) or bessel (better short-range
   # resolution for cutoff ≤ 8 Å; avoid with fully_connected or large cutoffs).
   radial_basis: gaussian

   # EDM preconditioning (Karras et al., NeurIPS 2022).
   # precondition: true enables σ-dependent skip/scale factors on the score head.
   # sigma_data: empirical std of (zero-COM) training positions in Å.
   precondition: false
   sigma_data: 1.0

   # ---------------------------------------------------------------------------
   # Diffusion / noiser configuration
   # ---------------------------------------------------------------------------
   noisers:
     - CellPositions    # One or more of:
                        #   Positions               : ZeroComStandardNormal prior + ZeroComNormal (gas-phase clusters)
                        #   CellPositions           : UniformCell prior + Normal (periodic bulk/surface)
                        #   ConfinedCellPositions   : UniformCellConfined prior + TruncatedNormal (Z-confined)
                        #   Types                   : discrete atom-type diffusion

   # SDE for position noisers.
   #   ve : Variance-Exploding SDE (default)
   #   vp : Variance-Preserving SDE
   sde: ve

   # Use a fully connected graph (all atom pairs, no neighbour list).
   # Recommended for gas-phase molecules and clusters.
   # Sets cutoff to 50 Å automatically when cutoff is null.
   fully_connected: false

   # Prediction parameterisation.
   #   score   : predict score ∝ ∇log p_t(x)  (default, recommended for ve)
   #   epsilon : predict normalised noise ε     (recommended for vp)
   prediction_type: score

   # Denoising formula during sampling.
   #   em   : Euler-Maruyama (default)
   #   ddpm : DDPM posterior-mean step (requires prediction_type: epsilon)
   sampler: em

   # Loss weighting.
   #   uniform : weight all noise levels equally (default)
   #   min_snr : cap weight at min(SNR, 5) — reduces gradient variance at low noise
   loss_weighting: uniform

   # Property conditioning (optional).  Set to "none" to disable.
   conditioning: none
   conditioning_type: scalar   # scalar | integer

   # Z-confinement range [z_min, z_max] in Å – null to disable.
   # Required when using the 'ConfinedCellPositions' noiser.
   confinement: null

   # Train a force-field alongside the diffusion score.
   # Set to true when training data contains per-atom DFT forces and you want to
   # use force-field guidance during sampling.
   force_field: false

   # Number of element-type classes for the Types noiser (excluding the absorbing
   # state at index 0).  When null, all distinct element types in the training data
   # are used automatically.  Only relevant when 'Types' is in noisers.
   n_classes: null

   # ---------------------------------------------------------------------------
   # Dataset splits and augmentation
   # ---------------------------------------------------------------------------
   batch_size: 64
   train_split: 0.9      # Fraction (float) or absolute count (int) for training
   val_split: 0.1        # Fraction (float) or absolute count (int) for validation
   mask: none            # Masking strategy: none | MaskFixed
   canonical_cell: false # Store cells in canonical lower-triangular form
   repeat: null          # Number of repetition levels (null = disabled)
   repeat_epoch: null    # Epochs between repetition-level increases

   # ---------------------------------------------------------------------------
   # Optimiser
   # ---------------------------------------------------------------------------
   lr: 0.0001
   lr_factor: 0.95
   lr_patience: 100
   weight_decay: 0.0
   eps: 0.00001
   guidance_weight: -1.0

   # ---------------------------------------------------------------------------
   # Trainer / logging
   # ---------------------------------------------------------------------------
   epochs: -1            # -1 = unlimited (stop by max_time or manually)
   max_time: 24          # Wall-clock limit in hours (null = no limit)
   gradient_clip_val: 10.0

   logger: tensorboard   # tensorboard | wandb
   log_dir: logs
   project: agedi        # WandB project name
   name: agedi           # WandB run display name
   log_interval: 10
   progress_bar: false

Noiser selection
----------------

The ``noisers`` list controls what is diffused. Choose based on your system:

.. list-table::
   :header-rows: 1
   :widths: 35 25 25 25

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

You can combine position and type noisers, e.g.:

.. code-block:: yaml

   noisers:
     - CellPositions
     - Types

Example: surface system with Z-confinement
------------------------------------------

.. code-block:: yaml

   data_path: PdO_training_data.traj

   noisers:
     - ConfinedCellPositions

   confinement: [2.0, 10.0]
   mask: MaskFixed

   max_time: 3      # hours
   feature_size: 64
   n_blocks: 4

Train with:

.. code-block:: console

   agedi train surface.yaml

Override the time limit on the fly:

.. code-block:: console

   agedi train surface.yaml max_time=6

Using ``train_from_config`` from Python
----------------------------------------

The same YAML file can be used directly from Python:

.. code-block:: python

   from agedi import train_from_config

   diffusion, dataset, trainer = train_from_config("my_train.yaml")

Programmatic overrides are also supported by passing a dict:

.. code-block:: python

   from agedi import train_from_config

   cfg = {
       "data_path": "train.traj",
       "noisers": ["CellPositions"],
       "feature_size": 128,
       "max_time": 6,
   }
   diffusion, dataset, trainer = train_from_config(cfg)
