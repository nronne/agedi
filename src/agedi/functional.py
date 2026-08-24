"""Backward-compatibility shim.

All public symbols are now implemented in :mod:`agedi.api`.
This module re-exports them so that existing code using
``from agedi.functional import X`` continues to work unchanged.
"""

from functools import wraps
from threading import RLock

from agedi.api import (  # noqa: F401
    create_dataset,
    create_diffusion,
    create_trainer,
    inpaint,
    load_diffusion,
    predict,
    register_model,
    sample,
    select_atoms,
    train,
)
from agedi.api import training as _training
from agedi.api._registry import _build_type_map_from_data

_PATCH_LOCK = RLock()


@wraps(_training.train_from_atoms)
def train_from_atoms(*args, **kwargs):
    with _PATCH_LOCK:
        original_load_diffusion = _training.load_diffusion
        _training.load_diffusion = load_diffusion
        try:
            return _training.train_from_atoms(*args, **kwargs)
        finally:
            _training.load_diffusion = original_load_diffusion


@wraps(_training.train_from_config)
def train_from_config(*args, **kwargs):
    with _PATCH_LOCK:
        original_train_from_atoms = _training.train_from_atoms
        _training.train_from_atoms = train_from_atoms
        try:
            return _training.train_from_config(*args, **kwargs)
        finally:
            _training.train_from_atoms = original_train_from_atoms


__all__ = [
    "create_dataset",
    "create_diffusion",
    "create_trainer",
    "_build_type_map_from_data",
    "inpaint",
    "load_diffusion",
    "predict",
    "register_model",
    "sample",
    "select_atoms",
    "train",
    "train_from_atoms",
    "train_from_config",
]
