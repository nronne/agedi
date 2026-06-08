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
     - ZeroComStandardNormal
     - ZeroComNormal
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

Gas-phase molecules and clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``fully_connected=True`` so all atom pairs are connected regardless of
how far apart they drift during sampling.  For VP-SDE, prefer
``prediction_type="epsilon"`` and ``sampler="ddpm"``:

.. code-block:: python

   from ase.io import read
   from agedi import train_from_atoms

   data = read("molecules.traj", ":")

   diffusion, dataset, trainer = train_from_atoms(
       data,
       noisers=("Positions",),
       sde="vp",
       prediction_type="epsilon",  # uniform gradient across noise levels
       sampler="ddpm",             # stable posterior-mean denoising
       loss_weighting="min_snr",   # optional: cap weight at min(SNR, 5)
       fully_connected=True,       # all-pairs graph, no cutoff
       max_time=4,                 # hours
       log_dir="logs",
   )

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

Force-field training with a regressor dataset
----------------------------------------------

To train a force-field head alongside the diffusion model, pass
``force_field=True``.  You can additionally supply a separate
``regressor_data`` sequence of :class:`~ase.Atoms` objects that will be used
*only* to train the force-field head (not the diffusion score).  This is
useful for non-equilibrium structures that carry informative forces but would
be unsuitable as diffusion training targets:

.. code-block:: python

   from ase.io import read
   from agedi import train_from_atoms

   equilibrium = read("training_data.traj", ":")
   nonequilibrium = read("nonequilibrium.traj", ":")

   diffusion, dataset, trainer = train_from_atoms(
       equilibrium,
       force_field=True,
       regressor_data=nonequilibrium,
       noisers=("ConfinedCellPositions",),
       mask="MaskFixed",
       confinement=(2.0, 10.0),
   )

Using :func:`~agedi.functional.create_dataset` directly:

.. code-block:: python

   from ase.io import read
   from agedi import create_dataset

   dataset = create_dataset(
       read("training_data.traj", ":"),
       mask="MaskFixed",
       confinement=(2.0, 10.0),
       regressor_data=read("nonequilibrium.traj", ":"),
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

   diffusion = load_diffusion("logs/agedi/version_0")

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
``logs/agedi/version_0``. If you want to use a different checkpoint, you can
specify the exact path to it when calling :func:`~agedi.functional.load_diffusion`.


Force-field training and prediction
-------------------------------------

To train a forces prediction head alongside the diffusion model, pass
``force_field=True`` to :func:`~agedi.functional.train_from_atoms`.  The
training data must include per-atom forces and total energy (e.g. from a
DFT calculation loaded via ASE):

.. code-block:: python

   from ase.io import read
   from agedi import train_from_atoms

   data = read("training_data.traj", ":")  # must contain forces and energy

   diffusion, dataset, trainer = train_from_atoms(
       data,
       noisers=("ConfinedCellPositions",),
       mask="MaskFixed",
       confinement=(2.0, 10.0),
       force_field=True,
       max_time=2,
   )

Once trained, use :func:`~agedi.functional.predict` to run energy and force
predictions on existing structures.  The results are returned as ASE
:class:`~ase.Atoms` objects with a
:class:`~ase.calculators.singlepoint.SinglePointCalculator` attached:

.. code-block:: python

   from ase.io import read, write
   from agedi import load_diffusion, predict

   diffusion = load_diffusion("logs/agedi/version_0")

   structures = read("structures.traj", index=":")
   predicted = predict(diffusion, structures)

   # Access predictions on the first structure
   print(predicted[0].get_potential_energy())  # eV
   print(predicted[0].get_forces())            # eV/Å

   write("predicted.traj", predicted)


Core public functions
----------------------

- :func:`~agedi.functional.create_diffusion`
- :func:`~agedi.functional.create_dataset`
- :func:`~agedi.functional.create_trainer`
- :func:`~agedi.functional.train`
- :func:`~agedi.functional.train_from_atoms`
- :func:`~agedi.functional.train_from_config`
- :func:`~agedi.functional.load_diffusion`
- :func:`~agedi.functional.predict`
- :func:`~agedi.functional.sample`
- :func:`~agedi.functional.register_model`

EDM preconditioning
--------------------

Enable σ-dependent skip/scale preconditioning (Karras et al., NeurIPS 2022) by
passing ``precondition=True``.  This keeps network inputs and outputs at unit scale
regardless of the noise level, which can stabilise training for large noise ranges:

.. code-block:: python

   from ase.io import read
   import numpy as np
   from agedi import train_from_atoms

   data = read("training_data.traj", ":")

   # Estimate sigma_data from the training set
   sigma_data = float(np.std(
       [a.get_positions() - a.get_center_of_mass() for a in data]
   ))

   diffusion, dataset, trainer = train_from_atoms(
       data,
       noisers=("CellPositions",),
       precondition=True,
       sigma_data=sigma_data,
       max_time=4,
   )

Custom model backends
----------------------

AGeDi ships with the ``"PaiNN"`` SchNetPack backend.  You can register
your own GNN backbone via :func:`~agedi.functional.register_model`:

.. code-block:: python

   from agedi import register_model

   def my_factory(cutoff, heads, feature_size, n_blocks, head_dim, n_rbf):
       # Build and return (translator, representation, head_list)
       ...

   register_model("MyModel", my_factory)

   # Then use it in create_diffusion / train_from_atoms:
   diffusion = create_diffusion(model="MyModel", ...)

Additional sampling options
-----------------------------

The :func:`~agedi.functional.sample` function supports several advanced
options beyond the basic ``n_samples`` / ``formula`` / ``steps`` arguments:

- ``compile=True`` — compile the reverse-diffusion step with
  ``torch.compile`` for faster GPU sampling.  Requires NVIDIA
  nvalchemiops.  Neighbor-list buffer sizes are estimated automatically
  before the sampling loop.

- ``save_trajectory=True`` — return a list of per-sample diffusion
  trajectories (one list of :class:`~agedi.AtomsGraph` / ASE
  :class:`~ase.Atoms` per sample) instead of only the final structures.

- ``print_timings=True`` — print a per-stage timing breakdown after each
  sampling batch (graph init, score model, denoise step, neighbor list,
  etc.).  Useful for profiling.

- ``property`` — pass a dict of property values to condition sampling on
  (requires the model to have been trained with ``conditioning``).

.. code-block:: python

   from agedi import load_diffusion, sample

   diffusion = load_diffusion("logs/agedi/version_0")

   # Compiled, 500 steps, save full trajectories
   trajectories = sample(
       diffusion,
       n_samples=4,
       formula="Pd4O4",
       steps=500,
       compile=True,
       save_trajectory=True,
       print_timings=True,
   )
   # trajectories[i] is the full reverse-diffusion path for sample i

Property conditioning
----------------------

Models can optionally be conditioned on a scalar or integer per-structure
property (e.g. formation energy, band gap, or total magnetisation).  Enable
conditioning at training time and then supply the target value at sampling
time.

Training with conditioning:

.. code-block:: python

   from agedi import train_from_atoms

   diffusion, dataset, trainer = train_from_atoms(
       data,
       noisers=("CellPositions",),
       conditioning="energy",        # key in atoms.info or atoms.get_energy()
       conditioning_type="scalar",   # "scalar" (default) or "integer"
   )

Sampling with a conditioning value:

.. code-block:: python

   from agedi import load_diffusion, sample

   diffusion = load_diffusion("logs/agedi/version_0")

   structures = sample(
       diffusion,
       n_samples=10,
       formula="Pd4O4",
       property={"energy": -3.5},   # target value for the conditioned property
   )

The ``conditioning`` key must match the ``atoms.info`` key (or an
``atoms.get_<key>()`` method) used in the training data.

