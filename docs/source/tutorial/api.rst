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


Choosing a sampler
-------------------

Pass ``sampler`` to :func:`~agedi.functional.sample` to select the
reverse-diffusion algorithm:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``sampler``
     - Description
   * - ``None`` (default)
     - Euler–Maruyama (EM): one score evaluation per step
   * - ``"em"``
     - Euler–Maruyama (explicit alias)
   * - ``"pc"``
     - Predictor-corrector: EM predictor + Langevin corrector steps at t_{i-1}
   * - ``"heun"``
     - 2nd-order stochastic (Karras et al. 2022): two score evaluations per step
   * - ``"ddim"``
     - Deterministic probability-flow ODE: no noise, fully reproducible
   * - ``"heun_ode"``
     - 2nd-order deterministic ODE (Heun's method on the PF-ODE)
   * - ``"ffpc"``
     - Force-field augmented predictor-corrector (requires a force-field head)

.. code-block:: python

   structures = sample(diffusion, n_samples=10, formula="Pd2O2",
                       template=template, sampler="heun", steps=200)

Additional keyword arguments are passed via ``sampler_kwargs``:

.. code-block:: python

   structures = sample(
       diffusion, n_samples=10, formula="Pd2O2", template=template,
       sampler="pc",
       sampler_kwargs=dict(corrector_steps=3, corrector_step_size=1e-3),
   )

You can also pass a :class:`~agedi.diffusion.samplers.Sampler` instance
directly instead of a string alias:

.. code-block:: python

   from agedi.diffusion.samplers import HeunSampler

   sampler = HeunSampler(diffusion.score_model, diffusion.noisers)
   structures = sample(diffusion, n_samples=10, formula="Pd2O2",
                       template=template, sampler=sampler)

Force-field augmented sampling (``ffpc``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ffpc`` sampler blends the neural score with the force-field gradient
during the corrector phase:

.. math::

   \tilde{s}(x, t) = (1 - f(t))\,s_\theta(x) + f(t)\,F(x)

where :math:`f(t) = (1-t)^\zeta`.  It optionally runs additional Langevin
dynamics after the last diffusion step via ``terminal_steps``.

Terminal dynamics respect ``confinement``: atoms are kept inside the z-slab and
bounce off its walls, and frozen (masked) template atoms never move.

``ffpc`` requires a model trained with a forces head.  Without one there is no
force field to blend or to drive the terminal dynamics, so the sampler warns
and degrades to plain predictor-corrector sampling: ``mixing_zeta`` is ignored
and no terminal steps run, leaving saved trajectories short by exactly
``1 + terminal_steps`` frames.  If your trajectories are missing their terminal
frames, check that the model actually has a regressor head.

.. code-block:: python

   from agedi import load_diffusion, sample

   diffusion = load_diffusion("logs/agedi/version_0")

   # EM predictor + force-field augmented Langevin corrector
   structures = sample(
       diffusion, n_samples=10, formula="Pd2O2", template=template,
       sampler="ffpc",
       sampler_kwargs=dict(corrector_steps=1, mixing_zeta=1.0),
   )

   # Add overdamped Langevin terminal steps for extra relaxation at 300 K
   structures = sample(
       diffusion, n_samples=10, formula="Pd2O2", template=template,
       sampler="ffpc",
       sampler_kwargs=dict(
           corrector_steps=0,
           terminal_steps=200,
           terminal_dynamics="overdamped",
           temperature=0.026,          # eV ≈ 300 K
           # terminal_step_size auto-selected (1e-3, T-independent)
       ),
   )

   # BAOAB Langevin MD terminal steps with real atomic masses
   structures = sample(
       diffusion, n_samples=10, formula="Pd2O2", template=template,
       sampler="ffpc",
       sampler_kwargs=dict(
           corrector_steps=0,
           terminal_steps=500,
           terminal_dynamics="langevin_md",
           temperature=0.026,          # eV ≈ 300 K
           # terminal_step_size auto-selected (1.0 fs for eV/Å models)
           # terminal_friction  auto-selected (γ·dt = 0.1)
       ),
       save_trajectory=True,           # includes bridge + terminal frames
   )

With ``save_trajectory=True`` the returned list contains one trajectory per
sample.  Each trajectory has ``steps + 1 + terminal_steps`` frames: the
pre-step diffusion frames, a bridge frame (the denoised structure before
terminal dynamics), and then the terminal step frames.

Full list of ``ffpc`` kwargs:

- ``corrector_steps`` (default ``1``): corrector iterations per diffusion step
- ``corrector_step_size`` (default ``1e-3``): Langevin corrector step size
- ``mixing_zeta`` (default ``1.0``): mixing schedule exponent
- ``temperature`` (default ``1.0``): temperature for terminal dynamics
- ``terminal_steps`` (default ``0``): post-diffusion terminal steps; ``0`` disables
- ``terminal_dynamics`` (default ``"overdamped"``): ``"overdamped"`` or ``"langevin_md"``
- ``terminal_step_size`` (default ``None``): auto-selected per mode
- ``terminal_friction`` (default ``None``): auto-selected (``langevin_md`` only)


.. _saving-every-step:

Saving every sampling step
---------------------------

``save_trajectory=True`` records one frame per outer reverse-diffusion step,
plus any ffpc terminal-dynamics frames and post-diffusion relaxation frames.
Langevin corrector sub-steps happen *inside* one outer step and are not
recorded by default — with ``corrector_steps=5`` you still get one frame per
diffusion step.

Add ``save_corrector_frames=True`` to record those too:

.. code-block:: python

   trajectories = sample(
       diffusion,
       n_samples=4,
       formula="Pd4O4",
       steps=200,
       sampler="ffpc",
       sampler_kwargs=dict(
           corrector_steps=5,
           terminal_steps=200,
           terminal_dynamics="langevin_md",
           temperature=0.026,
       ),
       save_trajectory=True,
       save_corrector_frames=True,
   )

Each outer step then contributes ``1 + corrector_steps`` frames — the state
entering the step, the post-predictor state, and each corrector state — so a
run has ``steps * (1 + corrector_steps) + 1`` frames, plus
``1 + terminal_steps`` when terminal dynamics are active.  For the example
above that is ``200 * 6 + 1 + 200 = 1401`` frames per sample, versus ``401``
without corrector capture.

Because capture multiplies trajectory length by roughly the corrector count,
it is off by default.  Turn it on when you need to inspect the Langevin
relaxation itself — for instance when diagnosing a ``corrector_step_size``
that is too large — rather than for routine production runs.


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

**Per-species reference energies**

Total DFT energies are dominated by a large composition-dependent offset that
carries no structural information.  By default (``reference_energies="auto"``)
AGeDi fits per-species reference energies :math:`E^0_Z` from the training data
by linear least squares on

.. math::

   E_\text{total} \approx \sum_Z n_Z E^0_Z

and subtracts them from the energy target, so the network only has to learn the
much smaller residual.  The offset is applied *inside* the energy head, so
predicted energies stay on the absolute scale of the training data.

Supply your own values (e.g. isolated-atom energies) with a mapping keyed by
chemical symbol or atomic number, or pass ``None`` to disable the subtraction:

.. code-block:: python

   diffusion, dataset, trainer = train_from_atoms(
       data,
       force_field=True,
       reference_energies={"Pd": -3.72, "O": -4.95},   # or "auto" / None
   )

**Force loss**

The forces head is trained with a Huber loss by default: quadratic below
``huber_delta`` (in eV/Å) and linear above it, so a handful of large force
labels cannot dominate the gradient.  Both settings are configurable:

.. code-block:: python

   diffusion, dataset, trainer = train_from_atoms(
       data,
       force_field=True,
       force_loss="huber",   # "huber" (default) | "mse" | "mae"
       huber_delta=0.01,     # eV/Å
   )

**Conservative forces**

By default the forces are predicted by a dedicated head, independently of the
energy — so they are not guaranteed to be the gradient of the predicted
energy. Passing ``conservative_forces=True`` instead derives the forces as
:math:`F = -\partial E/\partial R` by differentiating the energy prediction,
which guarantees energy/force consistency and lets the force labels also
train the energy surface. It drops the forces head entirely, so it cannot be
combined with an explicit ``force_loss``/``huber_delta`` setup for a separate
head — those options simply apply to the derived forces instead:

.. code-block:: python

   diffusion, dataset, trainer = train_from_atoms(
       data,
       force_field=True,
       conservative_forces=True,
   )

Conservative forces require a backward pass through the energy head on every
regressor call, which roughly doubles the cost of force-field guided sampling
(``ff_guidance``) and post-diffusion relaxation. Prediction and training cost
increase similarly but are usually a minor fraction of total runtime.

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

