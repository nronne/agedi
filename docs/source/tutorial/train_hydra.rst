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

   # ---------------------------------------------------------------------------
   # Score-model architecture
   # ---------------------------------------------------------------------------
   model: PaiNN          # Currently only PaiNN is supported
   cutoff: 6.0           # Neighbour-list cutoff in Å
   feature_size: 64      # Embedding / feature dimension
   n_blocks: 4           # Number of interaction blocks
   n_rbf: 30             # Number of radial basis functions

   # ---------------------------------------------------------------------------
   # Diffusion / noiser configuration
   # ---------------------------------------------------------------------------
   noisers:
     - CellPositions    # One or more of:
                        #   Positions               : StandardNormal prior + Normal (gas-phase clusters)
                        #   CellPositions           : UniformCell prior + Normal (periodic bulk/surface)
                        #   ConfinedCellPositions   : UniformCellConfined prior + TruncatedNormal (Z-confined)
                        #   Types                   : discrete atom-type diffusion

   # SDE for position noisers.
   #   ve : Variance-Exploding SDE (default)
   #   vp : Variance-Preserving SDE
   sde: ve

   # Property conditioning (optional).  Set to "none" to disable.
   conditioning: none
   conditioning_type: scalar   # scalar | integer

   # Z-confinement range [z_min, z_max] in Å – null to disable.
   # Required when using the 'ConfinedCellPositions' noiser.
   confinement: null

   # Train a force-field alongside the diffusion score.
   # Set to true when training data contains per-atom DFT forces and you want to
   # use force-field guidance during sampling.
   force-field: false

   # ---------------------------------------------------------------------------
   # Dataset splits and augmentation
   # ---------------------------------------------------------------------------
   batch_size: 64
   train_split: 0.9      # Fraction (float) or absolute count (int) for training
   val_split: 0.1        # Fraction (float) or absolute count (int) for validation
   mask: none            # Masking strategy: none | MaskFixed
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
