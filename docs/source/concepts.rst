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
     - :class:`~agedi.diffusion.distributions.StandardNormal`
     - :class:`~agedi.diffusion.distributions.Normal`
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

Discrete atom types can be diffused by adding a
:class:`~agedi.diffusion.noisers.Types` to the noiser list.

Sampling semantics
------------------

During sampling, required defaults depend on enabled noisers:

- ``n_atoms`` can come from explicit input, ``atomic_numbers``, or ``formula``
- ``atomic_numbers`` are needed if type noising is not enabled and formula is not provided
- ``positions`` are needed if position noising is not enabled
- ``cell`` is needed unless a template provides it

If a template is provided, generated atoms are appended to template atoms and
template atoms are masked as fixed.

Inpainting
----------

:func:`~agedi.api.inpaint` regenerates a chosen subset of atoms in an
*existing* structure, rather than generating a new structure from scratch.
It uses a second, distinct notion of "which atoms move" from the ``mask``
used for templates and ``FixAtoms`` handling above:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Field
     - Meaning
     - Behaviour
   * - ``mask`` (``freeze=``)
     - Hard-frozen
     - Never moves, at any point in the trajectory. Same mechanism as
       template / ``MaskFixed`` handling.
   * - ``inpaint_mask`` (the selection)
     - Regenerate
     - Fully re-noised (at ``t_start=1.0``) and denoised by the model like a
       from-scratch atom.

Every other atom — selected by neither — is a **known** atom: at each
reverse-diffusion step it is replaced by a fresh sample of the forward
process :math:`q(z_t \mid z_0)` of the input structure, so the whole
structure always sits at a self-consistent noise level for the score model.
Known atoms therefore visibly move over the course of the trajectory and
converge back onto their input positions (and, when a types noiser is
active, their input species) exactly as :math:`t \to \varepsilon`.
``freeze=`` atoms are a stricter subset that skip this replacement entirely
and stay bit-exact throughout.

Which atoms are selected for regeneration is controlled by
:func:`~agedi.api.select_atoms`: explicit ``indices``, ``symbols``,
``z_range``, ``sphere``, or ``from_atoms`` (reading
``atoms.arrays["inpaint_mask"]`` or the complement of any ``FixAtoms``
constraint). With none given, a random ``fraction`` of the non-fixed atoms
is selected. ``t_start`` below ``1.0`` starts from a partially-noised state
for a local rattle-and-relax refinement instead of full regeneration; the
optional ``n_resample`` / ``jump_length`` parameters enable RePaint-style
resampling (`Lugmayr et al. 2022 <https://arxiv.org/abs/2201.09865>`_) to
better harmonize the regenerated region with its surroundings, at the cost
of extra score-model evaluations.

.. code-block:: python

   from agedi import inpaint, load_diffusion
   from ase.io import read

   diffusion = load_diffusion("logs/agedi/version_0")
   atoms = read("structure.traj")

   structures = inpaint(diffusion, atoms, symbols=["O"], n_samples=4, steps=500)

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

