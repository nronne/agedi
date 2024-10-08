<p align="center">
  <img height="250" src="https://raw.githubusercontent.com/nronne/agedi/refs/heads/main/docs/agedi.svg?sanitize=true" />
</p>

______________________________________________________________________


[![Build
Status](https://cdn.prod.website-files.com/5e0f1144930a8bc8aace526c/65dd9eb5aaca434fac4f1c7c_Build-Passing-brightgreen.svg)]()
[![Documentation Status](https://readthedocs.org/projects/agedi/badge/?version=latest)](https://agedi.readthedocs.io/en/latest/?badge=latest)


**[Documentation](https://agedi.readthedocs.io)**


**AGeDI** pronounced "A Jedi" is a library for **A**tomistic **Ge**nerative
**Di**ffusion build on PyG, Lightning and ASE and offers customizable
diffusion models for periodic atomistic material generation. 

> [!CAUTION]
> This project is under active development.

## Interfaced Equivariant Models
At the moment only PaiNN is possible to use as a score model
architecture.

We expect to implement interfaces to GemNet-dQ, NequIP and possibly Mace. 

## Implemented Noisers and Scorers
Below is an overview of the different available noisers and for which
models there is an score-model implementation.

|                                       | Cartesian Coordinates | Fractional Coordinates | Atomic Types         | Cell                 |
| ------------------------------------- | --------------------- | ---------------------- | -------------------- | -------------------- |
| PaiNN                                 | :white_check_mark:    | :white_large_square:   | :white_large_square: | :white_large_square: |
| GemNet-dQ                             | :white_large_square:  | :white_large_square:   | :white_large_square: | :white_large_square: |
| NequIP                                | :white_large_square:  | :white_large_square:   | :white_large_square: | :white_large_square: |


