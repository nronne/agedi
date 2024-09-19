from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from ase import Atoms
from torch_geometric.loader import DataLoader

from .atoms_graph import AtomsGraph


class Dataset(pl.LightningDataModule):
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
        dataset = []
        for d in data:
            ag = AtomsGraph.from_atoms(d, cutoff=self.cutoff)
            if "energy" in self.properties and d.calc is not None:
                ag.energy = torch.tensor(d.get_potential_energy()).reshape(1, 1)
            if "forces" in self.properties and d.calc is not None:
                ag.forces = torch.tensor(d.get_forces()).reshape(-1, 3)
            dataset.append(ag)

        self.dataset = dataset

    def add_graph_data(self, data: List[AtomsGraph]) -> None:
        self.dataset = data

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_idx is None:
            train_subset, val_subset, test_subset = torch.utils.data.random_split(
                torch.arange(len(self.dataset), dtype=int), [self.n_train, self.n_val, self.n_test]
            )
            self.train_idx = train_subset.indices
            self.val_idx = val_subset.indices
            self.test_idx = test_subset.indices

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            [self.dataset[i] for i in self.train_idx], batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader([self.dataset[i] for i in self.val_idx], batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader([self.dataset[i] for i in self.test_idx], batch_size=self.batch_size)
