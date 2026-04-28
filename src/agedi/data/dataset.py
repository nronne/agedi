import os
from typing import Dict, List, Optional, Tuple, Union

from lightning import LightningDataModule
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

from .atoms_graph import AtomsGraph


class Dataset(LightningDataModule):
    """Defines a custom dataset for AtomsGraph data

    Parameters
    ----------
    batch_size : int
        The batch size for the DataLoader
    n_train : Union[float, int]
        The number of training samples. If float, it is interpreted as a fraction of the dataset size
    n_val : Union[float, int]
        The number of validation samples. If float, it is interpreted as a fraction of the dataset size
    n_test : Union[float, int]
        The number of test samples. If float, it is interpreted as a fraction of the dataset size
    shuffle : bool
        Whether to shuffle the dataset
    properties : List[str]
        The properties to include in the dataset. Can be "energy", "forces", or both
    cutoff : float
        The cutoff radius for the neighbor list
    phase_transforms : Optional[List[List[BaseTransform]]]
        The data augmentation transforms to apply to each training phase

    Returns
    -------
    Dataset

    """

    def __init__(
        self,
        batch_size: int = 32,
        n_train: Union[float, int] = 0.9,
        n_val: Union[float, int] = 0.1,
        n_test: Union[float, int] = 0.0,
        shuffle: bool = True,
        properties: List[str] = ["energy", "forces"],
        cutoff: float = 6.0,
        phase_transforms: Optional[List[List[BaseTransform]]] = None,
        num_workers: int = 0,
        **kwargs,
    ) -> None:
        """Initializes the Dataset object"""
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test

        self.properties = properties
        self.cutoff = cutoff

        self.dataset = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

        self.phase_transforms = phase_transforms
        self.num_workers = num_workers

        self.regressor_dataset = None
        self.regressor_train_loader = None

    def add_atoms_data(self, data: List[Atoms], mask_method: Optional[str] = None, confinement: Optional[Tuple[float, float]] = None, properties: Optional[List[Dict]] = None, canonical_cell: bool = False) -> None:
        """Add ASE data to the dataset

        Converts a list of ASE Atoms objects to AtomsGraph objects and adds them to the dataset

        Parameters
        ----------
        data : List[Atoms]
            A list of ASE Atoms objects
        mask_method : str, optional
            Method for computing the atom mask (e.g. ``"MaskFixed"``).
        confinement : Tuple[float, float], optional
            Z-axis confinement bounds ``(z_min, z_max)`` applied to every structure.
        properties : List[Dict], optional
            Per-structure property dictionaries; each entry is mapped to the
            corresponding graph via :func:`setattr`.
        canonical_cell : bool, optional
            When ``True`` (the default), cells are stored in canonical
            lower-triangular form.  Set to ``False`` to store cells exactly
            as provided by ASE.

        Returns
        -------
        None

        """
        dataset = []
        for i, d in enumerate(data):
            ag = AtomsGraph.from_atoms(d, cutoff=self.cutoff, canonical_cell=canonical_cell)
            
            if properties is not None:
                props = properties[i]
                for key, value in props.items():
                    setattr(ag, key, torch.tensor(value, dtype=torch.float32))

            #Add energy and forces if they are present
            has_E, has_F = self._has_energy_forces(d)
            if has_E:
                E = d.get_potential_energy()
                setattr(ag, "energy", torch.tensor(E, dtype=torch.float32))
            if has_F:
                F = d.get_forces(apply_constraint=False)
                setattr(ag, "forces", torch.tensor(F, dtype=torch.float32))
                    
            
            if mask_method is not None:
                match mask_method:
                    case "MaskFixed":
                        mask = ag.mask
                        for constraint in d.constraints:
                            if isinstance(constraint, FixAtoms):
                                mask[constraint.index] = True
                        ag.mask = mask
                    case "none":
                        pass
                    case _:
                        raise ValueError("Invalid mask type")

            if confinement is not None:
                ag.confinement = torch.tensor(confinement, dtype=torch.float32).reshape(1, 2)

            dataset.append(ag)

        if confinement is not None:
            self._check_confinement(dataset, confinement)

        if self.dataset is None:
            self.dataset = dataset
        else:
            self.dataset.extend(dataset)

    def add_graph_data(self, data: List[AtomsGraph]) -> None:
        """Add AtomsGraph data to the dataset

        Adds a list of AtomsGraph objects to the dataset

        Parameters
        ----------
        data : List[AtomsGraph]
            A list of AtomsGraph objects

        Returns
        -------
        None

        """
        if self.dataset is None:
            self.dataset = data
        else:
            self.dataset.extend(data)

    def add_regressor_data(self, data: List[Atoms], canonical_cell: bool = False) -> None:
        """Add atoms data that will be used exclusively for regressor training.

        Structures in this dataset are only used to train the regressor model
        (e.g. force-field heads) and are never passed through the diffusion
        loss.  This allows the regressor to learn from non-equilibrium
        structures that would be unsuitable as diffusion training targets.

        Energy and forces are read from the ASE calculator attached to each
        :class:`~ase.Atoms` object when available.

        Parameters
        ----------
        data : List[Atoms]
            A list of ASE :class:`~ase.Atoms` objects, each with an attached
            calculator that provides energy and forces.
        canonical_cell : bool, optional
            When ``True``, cells are stored in canonical lower-triangular form.
            Defaults to ``False``.

        Returns
        -------
        None

        """
        dataset = []
        for d in data:
            ag = AtomsGraph.from_atoms(d, cutoff=self.cutoff, canonical_cell=canonical_cell)

            has_E, has_F = self._has_energy_forces(d)
            if has_E:
                ag.energy = torch.tensor(d.get_potential_energy(), dtype=torch.float32)
            if has_F:
                ag.forces = torch.tensor(d.get_forces(apply_constraint=False), dtype=torch.float32)

            dataset.append(ag)

        if self.regressor_dataset is None:
            self.regressor_dataset = dataset
        else:
            self.regressor_dataset.extend(dataset)

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up train/validation/test splits and initialise data loaders.

        Performs a random split of the dataset (if not already split) and
        calls :meth:`set_phase` to create the initial data loaders.

        Parameters
        ----------
        stage : str, optional
            Lightning stage identifier (``"fit"``, ``"test"``, etc.).
            Not used internally; present for API compatibility.
        """
        if self.train_idx is None:
            train_subset, val_subset, test_subset = torch.utils.data.random_split(
                torch.arange(len(self.dataset), dtype=int),
                [self.n_train, self.n_val, self.n_test],
            )
            self.train_idx = train_subset.indices
            self.val_idx = val_subset.indices
            self.test_idx = test_subset.indices

        self.set_phase(0)

    def train_dataloader(self) -> DataLoader:
        """Get the training DataLoader

        Returns a DataLoader for the training dataset.  When a separate
        regressor dataset has been added via :meth:`add_regressor_data`, a
        :class:`~lightning.pytorch.utilities.CombinedLoader` is returned so
        that each training step receives both a regular batch (key
        ``"main"``) and a regressor-only batch (key ``"regressor"``).

        Returns
        -------
        DataLoader or CombinedLoader

        """
        if self.regressor_train_loader is not None:
            from lightning.pytorch.utilities import CombinedLoader
            return CombinedLoader(
                {"main": self.train_loader, "regressor": self.regressor_train_loader},
                mode="max_size_cycle",
            )
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Get the validation DataLoader

        Returns a DataLoader for the validation dataset

        Returns
        -------
        DataLoader

        """
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        """Get the test DataLoader

        Returns a DataLoader for the test dataset

        Returns
        -------
        DataLoader
        """
        return self.test_loader

    def set_phase(self, phase: int) -> None:
        """Switch the dataset to the given training phase.

        Applies the phase-specific transforms to the dataset splits and
        re-creates the data loaders with the augmented data.

        Parameters
        ----------
        phase : int
            Zero-based phase index.  Phase 0 uses the original data;
            subsequent phases append transformed copies according to
            ``phase_transforms[phase]``.
        """
        self.phase = phase

        if self.phase_transforms is not None:
            new_datasets = []
            for idx in [self.train_idx, self.val_idx, self.test_idx]:
                for i in idx.copy():
                    graph = self.dataset[i]
                    for transform in self.phase_transforms[phase]:
                        graph = transform(graph)
                        self.dataset.append(graph)
                        idx.append(len(self.dataset) - 1)

                        
        self.train_loader = DataLoader(
            [self.dataset[i] for i in self.train_idx],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

        self.val_loader = DataLoader(
            [self.dataset[i] for i in self.val_idx],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

        self.test_loader = DataLoader(
            [self.dataset[i] for i in self.test_idx],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

        if self.regressor_dataset is not None:
            self.regressor_train_loader = DataLoader(
                self.regressor_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
            )
        else:
            self.regressor_train_loader = None

    def _check_confinement(self, dataset: List["AtomsGraph"], confinement: Tuple[float, float]) -> None:
        """Check that all unmasked atoms in *dataset* lie within *confinement*.

        Parameters
        ----------
        dataset : List[AtomsGraph]
            The list of graphs to validate.
        confinement : Tuple[float, float]
            The ``(z_min, z_max)`` confinement bounds.

        Raises
        ------
        ValueError
            If any unmasked atom has a Z position outside the confinement.
            The error message includes a suggested confinement that covers all
            unmasked atoms.
        """
        z_min, z_max = float(confinement[0]), float(confinement[1])

        all_z: List[torch.Tensor] = []
        for ag in dataset:
            pos = ag.pos  # shape [N, 3]
            if "mask" in ag:
                unmasked = ~ag.mask
                z_positions = pos[unmasked, 2]
            else:
                z_positions = pos[:, 2]
            if z_positions.numel() > 0:
                all_z.append(z_positions)

        if not all_z:
            return

        all_z_cat = torch.cat(all_z)
        actual_min = all_z_cat.min()
        actual_max = all_z_cat.max()

        if actual_min < z_min or actual_max > z_max:
            raise ValueError(
                f"Unmasked atoms have Z positions outside the confinement "
                f"[{z_min:.3f}, {z_max:.3f}]. "
                f"Actual Z range of unmasked atoms: [{actual_min.item():.3f}, {actual_max.item():.3f}]. "
                f"Consider using confinement=({actual_min.floor().item():.1f}, {actual_max.ceil().item():.1f}) instead."
            )

    def _has_energy_forces(self, atoms):
        """
        Check if the given ASE Atoms object has energy and forces information available.
        This method checks if a calculator is attached to the Atoms object and if it contains the 'energy' and 'forces' properties in its results.
        It avoids a calculation if there is a calculator, but it has not yet been used.

        Parameters
        ----------
        atoms : Atoms
            The ASE Atoms object to check for energy and forces information.
        
        Returns
        -------
        Tuple[bool, bool]
            A tuple indicating whether energy and forces information is available, respectively.
        """
        # 1. Check if a calculator is even attached
        if atoms.calc is None:
            return False, False

        # 2. Check if the specific properties exist in the results dict
        # Using .get() prevents KeyErrors if 'results' isn't initialized
        results = getattr(atoms.calc, 'results', {})

        has_energy = 'energy' in results
        has_forces = 'forces' in results

        return has_energy, has_forces
                

