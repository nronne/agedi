Overview
========

What AGeDi is
-------------

AGeDi (Atomistic Generative Diffusion) is a framework for periodic atomistic
structure generation using diffusion models.

Core capabilities
-----------------

- Build graph data from ASE ``Atoms`` objects
- Train diffusion models on atomic coordinates and/or atom types
- Generate new structures from formulas, templates, or explicit defaults
- Use either a command line interface or a Python API

High-level package layout
-------------------------

- ``agedi.data``

  - ``AtomsGraph``: graph structure used by model/noisers
  - ``Dataset``: Lightning DataModule for splitting and batching training data

- ``agedi.models``

  - ``ScoreModel``: combines representation + conditioning + score heads
  - ``agedi.models.schnetpack``: PaiNN-based translator and heads

- ``agedi.diffusion``

  - ``Diffusion``: LightningModule orchestrating loss, training, and sampling
  - ``noisers``: forward/reverse diffusion components by variable type
  - ``distributions``: ``PriorDistribution`` (initial-state samplers) and ``NoiseDistribution`` (step samplers)
  - ``sdes``: stochastic differential equations (VE, VP) and noise schedules
    (Linear, Exponential, Cosine, DiscreteExponential)

- ``agedi.functional``

  - script-friendly entry points (``create_*``, ``train``, ``train_from_atoms``, ``sample``)

- ``agedi.cli``

  - ``agedi train`` / ``agedi sample`` / ``agedi inspect``

Typical workflow
----------------

1. Load ASE structures.
2. Build/train model (CLI or Python API).
3. Inspect saved hyperparameters/checkpoints.
4. Sample new structures.
5. Export as ASE trajectory for downstream evaluation.
