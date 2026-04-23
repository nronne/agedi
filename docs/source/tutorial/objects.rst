Object-based workflow
=====================

AGeDi exposes all its building blocks as plain Python classes so you can
construct, configure, and compose them directly — without going through the
functional helper functions.  This gives you full control over every detail of
the model architecture, the diffusion process, and the training loop.

.. note::

   This page assumes familiarity with
   `PyTorch Geometric <https://pytorch-geometric.readthedocs.io>`_ and
   `Lightning <https://lightning.ai/docs/pytorch/stable/>`_.

Position noisers at a glance
-----------------------------

Choose the position noiser that fits your system:

.. list-table::
   :header-rows: 1
   :widths: 35 25 25 25

   * - Class
     - Prior
     - Distribution
     - Use case
   * - :class:`~agedi.diffusion.noisers.Positions`
     - StandardNormal
     - Normal
     - Gas-phase (molecules, clusters)
   * - :class:`~agedi.diffusion.noisers.CellPositions`
     - UniformCell
     - Normal
     - Periodic bulk / surface
   * - :class:`~agedi.diffusion.noisers.ConfinedCellPositions`
     - UniformCellConfined
     - TruncatedNormal
     - Surface overlayer/adsorbate

Building the score model
-------------------------

The score model is assembled from three components:

1. A **translator** that maps an ``AtomsGraph`` to the format expected by the
   representation backend.
2. A **representation** (here PaiNN from SchNetPack) that produces per-atom
   features.
3. One or more **score heads** that project those features to scores for each
   noised variable.

.. code-block:: python

   import schnetpack as spk

   from agedi.models import ScoreModel
   from agedi.models.schnetpack import SchNetPackTranslator, PositionsScore
   from agedi.models.conditionings import TimeConditioning

   cutoff = 6.0
   feature_size = 64

   translator = SchNetPackTranslator(
       input_modules=[spk.atomistic.PairwiseDistances()]
   )

   representation = spk.representation.PaiNN(
       n_atom_basis=feature_size,
       n_interactions=4,
       radial_basis=spk.nn.GaussianRBF(n_rbf=30, cutoff=cutoff),
       cutoff_fn=spk.nn.CosineCutoff(cutoff),
   )

   conditionings = [TimeConditioning()]

   head_dim = feature_size + sum(c.output_dim for c in conditionings)
   heads = [PositionsScore(input_dim_scalar=head_dim)]

   score_model = ScoreModel(
       translator=translator,
       representation=representation,
       conditionings=conditionings,
       heads=heads,
   )

Building noisers
----------------

Pick one or more noisers.  All position noisers accept an optional ``sde``
argument to swap the stochastic differential equation:

.. code-block:: python

   from agedi.diffusion.noisers import CellPositions
   from agedi.diffusion.sdes import VE

   # Default VE SDE
   noiser = CellPositions()

   # Custom SDE parameters
   noiser = CellPositions(sde=VE(sigma_min=0.01, sigma_max=5.0))

For a Z-confined surface system:

.. code-block:: python

   from agedi.diffusion.noisers import ConfinedCellPositions

   noiser = ConfinedCellPositions()

For a gas-phase cluster:

.. code-block:: python

   from agedi.diffusion.noisers import Positions

   noiser = Positions()

Combining position and type diffusion:

.. code-block:: python

   from agedi.diffusion.noisers import CellPositions, Types

   noisers = [CellPositions(), Types()]

Building the Diffusion module
------------------------------

:class:`~agedi.diffusion.Diffusion` is a Lightning module that wires the score
model and noisers together:

.. code-block:: python

   from agedi.diffusion import Diffusion

   diffusion = Diffusion(
       score_model=score_model,
       noisers=[noiser],
       optim_config={"lr": 1e-4, "weight_decay": 0.0},
       scheduler_config={"factor": 0.95, "patience": 100},
       eps=1e-5,
   )

Building the dataset
---------------------

:class:`~agedi.data.Dataset` is a Lightning DataModule.  Pass it your
``AtomsGraph`` data together with any masking or confinement options:

.. code-block:: python

   from ase.io import read
   from agedi.data import Dataset, AtomsGraph

   raw = read("training_data.traj", ":")
   graphs = [AtomsGraph.from_atoms(a) for a in raw]

   dataset = Dataset(
       cutoff=6.0,
       batch_size=64,
       n_train=0.9,
       n_val=0.1,
   )
   dataset.add_atoms_data(
       list(raw),
       mask_method="MaskFixed",        # or "none"
       confinement=(2.0, 10.0),        # omit if not confined
   )
   dataset.setup()

Training with Lightning
-----------------------

Use a standard Lightning ``Trainer`` to drive the fit loop:

.. code-block:: python

   from lightning import Trainer
   from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
   from lightning.pytorch.loggers import TensorBoardLogger

   logger = TensorBoardLogger(save_dir="logs", name="")

   callbacks = [
       LearningRateMonitor(logging_interval="epoch"),
       ModelCheckpoint(
           filename="best_model",
           monitor="val_loss",
           mode="min",
           save_top_k=1,
       ),
       ModelCheckpoint(
           filename="last_model",
           monitor=None,
           save_top_k=1,
           every_n_epochs=1,
       ),
   ]

   trainer = Trainer(
       max_time={"hours": 3},
       accelerator="auto",
       devices=1,
       logger=logger,
       callbacks=callbacks,
       gradient_clip_val=10.0,
       enable_progress_bar=False,
       log_every_n_steps=10,
   )

   trainer.fit(diffusion, dataset)

.. note::

   :func:`~agedi.functional.train_from_atoms` prints a Rich-formatted model-architecture
   panel and run-configuration table automatically — the same output shown by ``agedi train``.
   In the object-based workflow you can inspect the full architecture via
   ``diffusion.get_hparams()``.
--------

After training, load the checkpoint and call
:meth:`~agedi.diffusion.Diffusion.sample` directly on the model:

.. code-block:: python

   import torch
   from ase.io import read, write
   from agedi.data import AtomsGraph
   from agedi.diffusion import Diffusion

   # Reconstruct model from saved hparams (recommended)
   from agedi import load_diffusion
   diffusion = load_diffusion("logs/version_0")
   # load_diffusion prints a Rich model-architecture panel automatically.

   # --- or load manually ---
   # diffusion = Diffusion(score_model, noisers)
   # ckpt = torch.load("logs/version_0/checkpoints/last_model.ckpt", weights_only=True)
   # diffusion.load_state_dict(ckpt["state_dict"])

   diffusion.eval()

   template = AtomsGraph.from_atoms(
       read("template.traj"), confinement=(2.0, 10.0)
   )

   with torch.no_grad():
       graphs = diffusion.sample(
           N=12,
           template=template,
           formula="Pd2O2",
           confinement=(2.0, 10.0),
           steps=500,
           eps=1e-3,
           batch_size=64,
       )

   structures = [g.to_atoms() for g in graphs]
   write("sampled.traj", structures)

Full minimal script
-------------------

Putting it all together for a Z-confined surface overlayer system:

.. code-block:: python

   import schnetpack as spk
   from ase.io import read, write
   from lightning import Trainer
   from lightning.pytorch.callbacks import ModelCheckpoint
   from lightning.pytorch.loggers import TensorBoardLogger

   from agedi.data import Dataset
   from agedi.diffusion import Diffusion
   from agedi.diffusion.noisers import ConfinedCellPositions
   from agedi.models import ScoreModel
   from agedi.models.conditionings import TimeConditioning
   from agedi.models.schnetpack import SchNetPackTranslator, PositionsScore

   # --- Hyperparameters ---
   cutoff = 6.0
   feature_size = 64

   # --- Score model ---
   translator = SchNetPackTranslator(
       input_modules=[spk.atomistic.PairwiseDistances()]
   )
   representation = spk.representation.PaiNN(
       n_atom_basis=feature_size,
       n_interactions=4,
       radial_basis=spk.nn.GaussianRBF(n_rbf=30, cutoff=cutoff),
       cutoff_fn=spk.nn.CosineCutoff(cutoff),
   )
   conditionings = [TimeConditioning()]
   head_dim = feature_size + sum(c.output_dim for c in conditionings)
   score_model = ScoreModel(
       translator=translator,
       representation=representation,
       conditionings=conditionings,
       heads=[PositionsScore(input_dim_scalar=head_dim)],
   )

   # --- Diffusion ---
   diffusion = Diffusion(
       score_model=score_model,
       noisers=[ConfinedCellPositions()],
       optim_config={"lr": 1e-4},
       scheduler_config={"factor": 0.95, "patience": 100},
   )

   # --- Dataset ---
   raw = read("training_data.traj", ":")
   dataset = Dataset(cutoff=cutoff, batch_size=64, n_train=0.9, n_val=0.1)
   dataset.add_atoms_data(
       list(raw),
       mask_method="MaskFixed",
       confinement=(2.0, 10.0),
   )
   dataset.setup()

   # --- Trainer ---
   trainer = Trainer(
       max_time={"hours": 3},
       accelerator="auto",
       logger=TensorBoardLogger(save_dir="logs", name=""),
       callbacks=[
           ModelCheckpoint(filename="last_model", save_top_k=1, every_n_epochs=1),
       ],
       gradient_clip_val=10.0,
       log_every_n_steps=10,
   )

   trainer.fit(diffusion, dataset)
