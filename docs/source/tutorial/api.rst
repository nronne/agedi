Python API workflow
===================

This page shows the script-based workflow using functions from
:mod:`agedi.functional`, re-exported at the top-level :mod:`agedi` package.
Using the functional API allows for more customisation than relying on
the CLI.

Position noisers
----------------

Choose the noiser that matches your system type:

.. list-table:: Position noisers
   :header-rows: 1
   :widths: 35 25 25 25

   * - Noiser string / class
     - Prior
     - Distribution
     - Use case
   * - ``"Positions"`` / :class:`~agedi.diffusion.noisers.Positions`
     - StandardNormal
     - Normal
     - Gas-phase (molecules, clusters)
   * - ``"CellPositions"`` / :class:`~agedi.diffusion.noisers.CellPositions`
     - UniformCell
     - Normal
     - Periodic bulk / surface (default)
   * - ``"ConfinedCellPositions"`` / :class:`~agedi.diffusion.noisers.ConfinedCellPositions`
     - UniformCellConfined
     - TruncatedNormal
     - Surface overlayer/adsorbate


Training
---------

Here we show the same example as with the CLI, using
:func:`~agedi.functional.train_from_atoms`.

.. code-block:: python

   from ase.io import read
   from agedi import train_from_atoms

   data = read("training_data.traj", ":")

   diffusion, dataset, trainer = train_from_atoms(
       data,
       noisers=("ConfinedCellPositions",),
       mask="MaskFixed",
       confinement=(2.0, 10.0),
       max_time=2,  # hours
       log_dir="logs",
   )

More detailed workflow
-----------------------

Here we show a more detailed example setting up the diffusion model,
the dataset and the trainer individually.

.. code-block:: python

   from ase.io import read
   from agedi import create_diffusion, create_dataset, create_trainer, train

   data = read("training_data.traj", ":")

   diffusion = create_diffusion(
       noisers=("ConfinedCellPositions",),
   )

   dataset = create_dataset(
       data,
       mask="MaskFixed",
       confinement=(2.0, 10.0)
   )

   trainer = create_trainer(
       max_time=2,  # hours
       log_dir="logs"
   )

   train(diffusion, dataset, trainer=trainer)

Sampling with template
-----------------------

To sample from a trained model:

.. code-block:: python

   from ase.io import read, write
   from agedi import load_diffusion, sample, AtomsGraph

   diffusion = load_diffusion("logs/version_0")

   template = AtomsGraph.from_atoms(read("template.traj"), confinement=(2.0, 10.0))

   structures = sample(
       diffusion,
       n_samples=12,
       formula="X2Y3",
       template=template,
       confinement=(2.0, 10.0),
       steps=500,
   )

   write("sampled.traj", structures)

Similar to the CLI, this samples using the ``last_model.ckpt`` checkpoint found in
``logs/version_0``. If you want to use a different checkpoint, you can
specify the exact path to it when calling :func:`~agedi.functional.load_diffusion`.


Core public functions
----------------------

- :func:`~agedi.functional.create_diffusion`
- :func:`~agedi.functional.create_dataset`
- :func:`~agedi.functional.create_trainer`
- :func:`~agedi.functional.train`
- :func:`~agedi.functional.train_from_atoms`
- :func:`~agedi.functional.load_diffusion`
- :func:`~agedi.functional.sample`
