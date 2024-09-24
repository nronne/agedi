from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from ase import Atoms
from torch_geometric.loader import DataLoader

from .atoms_graph import AtomsGraph


class Dataset(pl.LightningDataModule):
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

    Returns
    -------
    Dataset
    
    """
    def __init__(
        self,
        batch_size: int = 32,
        n_train: Union[float, int] = 0.8,
        n_val: Union[float, int] = 0.1,
        n_test: Union[float, int] = 0.1,
        shuffle: bool = True,
        properties: List[str] = ["energy", "forces"],
        cutoff: float = 6.0,
        **kwargs
    ) -> None:
        """Initializes the Dataset object
        
        """
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

    def add_atoms_data(self, data: List[Atoms]) -> None:
        """Add ASE data to the dataset
        
        Converts a list of ASE Atoms objects to AtomsGraph objects and adds them to the dataset

        Parameters
        ----------
        data : List[Atoms]
            A list of ASE Atoms objects

        Returns
        -------
        None

        """
        dataset = []
        for d in data:
            ag = AtomsGraph.from_atoms(d, cutoff=self.cutoff)
            if "energy" in self.properties and d.calc is not None:
                ag.energy = torch.tensor(d.get_potential_energy()).reshape(1, 1)
            if "forces" in self.properties and d.calc is not None:
                ag.forces = torch.tensor(d.get_forces()).reshape(-1, 3)
            dataset.append(ag)

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

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_idx is None:
            train_subset, val_subset, test_subset = torch.utils.data.random_split(
                torch.arange(len(self.dataset), dtype=int), [self.n_train, self.n_val, self.n_test]
            )
            self.train_idx = train_subset.indices
            self.val_idx = val_subset.indices
            self.test_idx = test_subset.indices

    def train_dataloader(self) -> DataLoader:
        """Get the training DataLoader
        
        Returns a DataLoader for the training dataset

        Returns
        -------
        DataLoader
        
        """
        return DataLoader(
            [self.dataset[i] for i in self.train_idx], batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation DataLoader
        
        Returns a DataLoader for the validation dataset

        Returns
        -------
        DataLoader
        
        """
        return DataLoader([self.dataset[i] for i in self.val_idx], batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Get the test DataLoader
        
        Returns a DataLoader for the test dataset

        Returns
        -------
        DataLoader
        """
        return DataLoader([self.dataset[i] for i in self.test_idx], batch_size=self.batch_size)
