"""Dataset creation."""

import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ase import Atoms

from agedi.data import Dataset
from agedi.data.transforms import Repeat


def create_dataset(
    data: Sequence[Atoms],
    cutoff: float = 6.0,
    batch_size: int = 64,
    train_split: Union[float, int] = 0.9,
    val_split: Union[float, int] = 0.1,
    mask: str = "none",
    confinement: Optional[Tuple[float, float]] = None,
    conditioning: str = "none",
    conditioning_type: str = "scalar",
    repeat: Optional[int] = None,
    canonical_cell: bool = False,
    regressor_data: Optional[Sequence[Atoms]] = None,
    properties: Optional[List[Dict]] = None,
    fully_connected: bool = False,
) -> Dataset:
    """Create and setup an AGeDi Dataset from ASE Atoms objects.

    Parameters
    ----------
    data : Sequence[Atoms]
        ASE Atoms objects to add to the dataset.
    cutoff : float, optional
        Neighbour-list cutoff radius in Ångström.
    batch_size : int, optional
        Mini-batch size used during training/validation.
    train_split : Union[float, int], optional
        Fraction or absolute number of samples for the training split.
    val_split : Union[float, int], optional
        Fraction or absolute number of samples for the validation split.
    mask : str, optional
        Atom-mask method (e.g. ``"MaskFixed"`` or ``"none"``).
    confinement : Tuple[float, float], optional
        Z-axis confinement bounds ``(z_min, z_max)``.
    conditioning : str, optional
        Name of the per-structure property to use as a conditioning signal.
        The value is read from ``atoms.info[conditioning]`` or the
        corresponding ``atoms.get_<conditioning>()`` method.  Ignored when
        set to ``"none"`` (default).
    conditioning_type : str, optional
        ``"scalar"`` (default) or ``"node"``; controls how the conditioning
        property is broadcast onto the graph.
    repeat : int, optional
        When given, augment the dataset by repeating each structure up to
        ``repeat`` times along the first two cell vectors.
    canonical_cell : bool, optional
        Store cells in canonical lower-triangular form.
    regressor_data : Sequence[Atoms], optional
        Additional ASE Atoms objects used to train a regressor head.
    properties : List[Dict], optional
        Per-structure property dictionaries; **must** contain exactly one
        entry per element in *data*.  Each dictionary is merged into the
        corresponding graph object via ``setattr``, matching the layout
        accepted by :meth:`~agedi.data.Dataset.add_atoms_data`.  Keys
        already produced by the *conditioning* logic are overwritten by
        values in *properties* when both are present.

    Returns
    -------
    Dataset
        A fully set-up :class:`~agedi.data.Dataset` ready for training.
    """
    phase_transforms = None
    if repeat is not None:
        if repeat < 2:
            raise ValueError(f"repeat must be at least 2, got {repeat}")

        property_kinds = {"mask": "node"}
        if confinement is not None:
            property_kinds["confinement"] = "none"
        if conditioning != "none":
            property_kinds[conditioning] = (
                "node" if conditioning_type == "node" else "none"
            )
        phase_transforms = [[]]
        for i in range(2, repeat + 1):
            phase_transforms.append([Repeat((i, i, 1), property=property_kinds)])

    dataset = Dataset(
        cutoff=cutoff,
        batch_size=batch_size,
        n_train=train_split,
        n_val=val_split,
        phase_transforms=phase_transforms,
        num_workers=min(4, os.cpu_count() or 1),
        fully_connected=fully_connected,
    )

    conditioning_properties = None
    if conditioning != "none":
        conditioning_properties = []
        for sample in data:
            value = None
            try:
                value = getattr(sample, f"get_{conditioning}")()
            except AttributeError:
                pass

            try:
                value = sample.info[conditioning]
            except KeyError:
                pass

            if value is None:
                value = 0
                warnings.warn(
                    f"Conditioning '{conditioning}' not found for one sample; using 0.",
                    stacklevel=2,
                )

            conditioning_properties.append({conditioning: value})

    if conditioning_properties is not None and properties is not None:
        if len(properties) != len(data):
            raise ValueError(
                f"properties must have the same length as data "
                f"({len(properties)} != {len(data)})"
            )
        merged_properties = [
            {**cond, **user}
            for cond, user in zip(conditioning_properties, properties)
        ]
    elif conditioning_properties is not None:
        merged_properties = conditioning_properties
    else:
        if properties is not None and len(properties) != len(data):
            raise ValueError(
                f"properties must have the same length as data "
                f"({len(properties)} != {len(data)})"
            )
        merged_properties = properties

    dataset.add_atoms_data(
        list(data),
        mask_method=mask,
        confinement=confinement,
        properties=merged_properties,
        canonical_cell=canonical_cell,
    )

    if regressor_data is not None:
        dataset.add_regressor_data(list(regressor_data), canonical_cell=canonical_cell)

    dataset.setup()
    return dataset
