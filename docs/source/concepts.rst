Concepts and model behavior
===========================

Graph representation
--------------------

AGeDi uses ``AtomsGraph`` as the main data object:

- Nodes: atomic numbers (``x``) and positions (``pos``)
- Edges: neighbor graph from periodic cutoff
- Graph-level data: cell, pbc, optional confinement
- Optional mask marks fixed atoms during diffusion updates

Diffusion components
--------------------

``Agedi`` combines:

- A score model (predicts scores for configured targets)
- One or more noisers (e.g., positions, types)
- Optimizer/scheduler configuration for Lightning training

Supported score/noiser pairing is enforced by key matching.

Position noisers
----------------

Three position noisers are available, each with a fixed prior and noise
distribution baked in.  Choose based on the physics of your system:

.. list-table::
   :header-rows: 1
   :widths: 35 25 25 25

   * - Class / identifier
     - Prior
     - Distribution
     - Use case
   * - :class:`~agedi.diffusion.noisers.Positions` / ``"Positions"``
     - :class:`~agedi.diffusion.distributions.ZeroComStandardNormal`
     - :class:`~agedi.diffusion.distributions.ZeroComNormal`
     - Gas-phase (molecules, clusters)
   * - :class:`~agedi.diffusion.noisers.CellPositions` / ``"CellPositions"``
     - :class:`~agedi.diffusion.distributions.UniformCell`
     - :class:`~agedi.diffusion.distributions.Normal`
     - Periodic bulk / surface (default)
   * - :class:`~agedi.diffusion.noisers.ConfinedCellPositions` / ``"ConfinedCellPositions"``
     - :class:`~agedi.diffusion.distributions.UniformCellConfined`
     - :class:`~agedi.diffusion.distributions.TruncatedNormal`
     - Surface overlayer/adsorbate

The **prior** is the distribution used to initialise atomic positions at the
start of the reverse (sampling) process.  The **distribution** is the noise
kernel applied during the forward (training) process.  The SDE can still be
chosen freely on all three classes (default: Variance-Exploding, ``"ve"``).

The ``Positions`` noiser uses zero-COM distributions
(:class:`~agedi.diffusion.distributions.ZeroComNormal`,
:class:`~agedi.diffusion.distributions.ZeroComStandardNormal`) that project
the noise increment onto the translationally-invariant subspace, preventing
center-of-mass drift and improving training stability for gas-phase molecules.
The prior scale is set automatically from the SDE's marginal at :math:`T=1`
(i.e. ``sqrt(var(T=1))``), replacing the old ad-hoc ``0.8·N^{1/3}`` scaling.

Discrete atom types can be diffused by adding a
:class:`~agedi.diffusion.noisers.Types` to the noiser list.

Fully-connected graphs for gas-phase systems
--------------------------------------------

By default AGeDi builds a neighbour-list with a fixed cutoff radius.  For
gas-phase molecules and clusters this can miss long-range pairs when atoms
spread during sampling.  Pass ``fully_connected=True`` to use an all-pairs
graph instead:

.. code-block:: python

   diffusion = create_diffusion(
       noisers=("Positions",),
       fully_connected=True,   # connects every atom pair, no cutoff
   )

With ``fully_connected=True`` and ``cutoff=None`` (the default), the backbone
cutoff is automatically set to 50 Å so the radial basis functions and cosine
envelope cover the full range of inter-atomic distances seen during sampling.
FC edges are computed once and cached internally; ``update_graph`` restores them
from the cache in O(1) — there is no neighbour-list overhead.

The ``fully_connected`` flag is stored on the ``Agedi`` object and propagated
automatically to :meth:`~agedi.diffusion.Diffusion.sample`, so models saved with
``fully_connected=True`` do not require the flag to be re-specified at sampling time.

Prediction type and sampler
----------------------------

Position noisers support two parameterisations (``prediction_type``) and two
denoising formulas (``sampler``), all interchangeable with any SDE:

**prediction_type**

* ``"score"`` (default) — the network predicts a quantity proportional to the
  score :math:`\nabla \log p_t(\mathbf{x})`.
  Loss: :math:`\|\boldsymbol{\varepsilon} + r_\text{score} \cdot \mathrm{var}(t)\|^2`.
  Recommended for VE-SDE.

* ``"epsilon"`` — the network predicts the normalised noise
  :math:`\boldsymbol{\varepsilon} = (\mathbf{x}_t - \mu(t)\mathbf{x}_0)/\sqrt{\mathrm{var}(t)}`.
  Loss: :math:`\|r_\text{score} - \boldsymbol{\varepsilon}\|^2`.
  Gradient magnitude is uniform across all noise levels (no :math:`\mathrm{var}(t)` weighting),
  which is essential for VP-SDE.  **Recommended with** ``sde="vp"``.

**sampler**

* ``"em"`` (default) — Euler–Maruyama update.

* ``"ddpm"`` — DDPM posterior-mean step (Ho et al., NeurIPS 2020).
  Requires ``prediction_type="epsilon"``.  More stable than EM at large
  ``beta_max`` because the denominator :math:`\sqrt{1 - \beta\Delta t}` cancels
  per-step amplification.

**loss_weighting**

* ``"uniform"`` (default) — all noise levels weighted equally.

* ``"min_snr"`` — caps per-sample weights at :math:`\min(\mathrm{SNR}, 5)`,
  following Hang et al. (ICCV 2023).  Reduces gradient variance at low-noise levels.

EDM preconditioning
-------------------

Enable EDM preconditioning (Karras et al., NeurIPS 2022) on the positions score
head with ``precondition=True``:

.. code-block:: python

   diffusion = create_diffusion(
       noisers=("CellPositions",),
       precondition=True,
       sigma_data=1.8,   # empirical std of (zero-COM) training positions in Å
   )

The head wraps its output with :math:`\sigma`-dependent skip and scale factors
so the network always operates on unit-scale inputs and outputs regardless of the
noise level.  ``sigma_data`` should be estimated from the training set:

.. code-block:: python

   import numpy as np
   from ase.io import read
   data = read("train.traj", ":")
   sigma_data = np.std([a.get_positions() - a.get_center_of_mass() for a in data])

Radial basis functions
----------------------

The PaiNN backbone supports two radial basis types (``radial_basis``):

* ``"gaussian"`` (default) — GaussianRBF.  Values are bounded in :math:`[0, 1]`
  for all distances; safe for any cutoff including the 50 Å FC default.

* ``"bessel"`` — BesselRBF.  Offers better short-range resolution because the
  basis functions are uniformly spaced in frequency rather than distance.
  **Only recommended for small cutoffs (≤ 8 Å).**  At large cutoffs or with
  fully-connected graphs, BesselRBF values at very short inter-atomic distances
  (which occur when atoms overlap during noisy training) can exceed 8, causing
  activation overflow and NaN loss.

Sampling semantics
------------------

During sampling, required defaults depend on enabled noisers:

- ``n_atoms`` can come from explicit input, ``atomic_numbers``, or ``formula``
- ``atomic_numbers`` are needed if type noising is not enabled and formula is not provided
- ``positions`` are needed if position noising is not enabled
- ``cell`` is needed unless a template provides it

If a template is provided, generated atoms are appended to template atoms and
template atoms are masked as fixed.

Training outputs
----------------

By default, training writes to ``logs/version_x``:

- ``hparams.yaml``: run hyperparameters and data metadata
- ``checkpoints/``: model checkpoints

``load_diffusion`` reconstructs the model from these artifacts.

Property conditioning
---------------------

The score model can be conditioned on a per-structure scalar or integer
property so that sampling can be steered towards a target value (e.g.
formation energy or band gap).  Use the ``conditioning`` parameter (CLI:
``--conditioning``) to specify the property name and
``conditioning_type`` (CLI: ``--conditioning_type``) to choose between
``"scalar"`` (continuous, default) and ``"integer"`` (discrete) encoding.

The property value is looked up from ``atoms.info[conditioning]`` or
``atoms.get_<conditioning>()`` for each training structure.  At sampling
time pass the target value in the ``property`` dict:

.. code-block:: python

   structures = sample(diffusion, n_samples=10, formula="Pd4O4",
                       property={"energy": -3.5})

Data augmentation (cell repeat)
---------------------------------

For periodic systems it can be beneficial to augment the training data by
tiling each structure along the first two cell vectors.  Enable this with
``repeat`` (CLI: ``--repeat``) and set the epoch interval at which the
repetition level increases with ``repeat_epoch`` (CLI: ``--repeat_epoch``).

For example, ``repeat=3, repeat_epoch=50`` starts training on the original
cells, increases to 2×2×1 at epoch 50, then to 3×3×1 at epoch 100.

