<p align="center">
  <img height="250" src="https://raw.githubusercontent.com/nronne/agedi/refs/heads/main/docs/agedi.svg?sanitize=true" />
</p>

# AGeDi

**AGeDi** (Atomistic Generative Diffusion) is a Python package for training and sampling diffusion models for atomistic structures.
It is built around **PyTorch**, **PyTorch Geometric**, **PyTorch Lightning**, and **ASE**.

[![Build Status](https://github.com/nronne/agedi/actions/workflows/python-package.yml/badge.svg)](https://github.com/nronne/agedi/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/agedi/badge/?version=latest)](https://agedi.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


**[Documentation](https://agedi.readthedocs.io)**


**AGeDi** pronounced "A Jedi" is a library for **A**tomistic **Ge**nerative
**Di**ffusion built on PyG, Lightning and ASE and offers customizable
diffusion models for periodic atomistic material generation.

- Full docs: https://agedi.readthedocs.io
- CLI entrypoint: `agedi`
- Primary model backend today: **PaiNN** (via SchNetPack)

> [!CAUTION]
> This project is under active development.

## What AGeDi does

AGeDi provides:

- Data conversion from ASE `Atoms` to graph data (`AtomsGraph`)
- Training pipeline for diffusion models over positions and atom types
- Sampling pipeline from trained checkpoints (with optional templates)
- CLI and Python functional API for reproducible workflows

## Installation

Minimal install:

```bash
pip install "agedi @ git+https://github.com/nronne/agedi.git"
```

This installs the core package only. For the current release, training and
sampling require PaiNN via SchNetPack:

```bash
pip install "agedi[full] @ git+https://github.com/nronne/agedi.git"
```

For contributors:

```bash
pip install -e ".[test,full]"
```

## Quickstart (CLI)

```bash
# Train (example: 3 hours, surface/slab system)
agedi train -t 3 --noisers ConfinedCellPositions --mask MaskFixed --confinement 2 10 PdO_training_data.traj

# Inspect saved hyperparameters
agedi inspect logs/version_0

# Sample structures
agedi sample logs/version_0 -f Pd2O2 --template_path template.traj --confinement 2 10
```

## Quickstart (Python API)

```python
from ase.io import read
from agedi import train_from_atoms, sample, AtomsGraph

data = read("PdO_training_data.traj", ":")
diffusion, dataset, trainer = train_from_atoms(
    data,
    noisers=("Positions",),
    style="surface",
    mask="MaskFixed",
    confinement=(2.0, 10.0),
    max_time=3,
)

template = AtomsGraph.from_atoms(read("template.traj"), confinement=(2.0, 10.0))
structures = sample(diffusion, n_samples=8, formula="Pd2O2", template=template)
```

## Documentation map

The documentation has dedicated pages for:

- System overview and code architecture
- Installation and environment setup
- CLI and Python workflows
- End-to-end PdO tutorial
- Pitfalls and troubleshooting
- Publication references and citation text
- API reference (auto-generated)

## References

- N. Rønne, A. Aspuru-Guzik, B. Hammer, *Phys. Rev. B* **110**, 235427 (2024): https://doi.org/10.1103/PhysRevB.110.235427
- AGeDi preprint: https://arxiv.org/abs/2507.18314

## Citation

If you use AGeDi in research, please cite the paper above and the AGeDi preprint.
