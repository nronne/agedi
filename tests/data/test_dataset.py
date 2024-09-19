import numpy as np
import pytest
import torch

from agedi.data import Dataset, AtomsGraph
from ase.calculators.singlepoint import SinglePointCalculator as sp
from torch_geometric.loader import DataLoader
    
def test_init() -> None:
    dataset = Dataset()
    assert dataset is not None

def test_add_atoms_data(atoms: "Atoms") -> None:
    dataset = Dataset()
    dataset.add_atoms_data([atoms])
    assert isinstance(dataset.dataset[0], AtomsGraph)

def test_add_atoms_data_with_ef(atoms: "Atoms") -> None:
    atoms.calc = sp(atoms, energy=0.0, forces=np.zeros((len(atoms), 3)))
    dataset = Dataset()
    dataset.add_atoms_data([atoms])
    assert isinstance(dataset.dataset[0], AtomsGraph)

def test_add_graph_data(graph: AtomsGraph) -> None:
    dataset = Dataset()
    dataset.add_graph_data([graph])
    assert isinstance(dataset.dataset[0], AtomsGraph)
    
def test_setup(atoms: "Atoms") -> None:
    dataset = Dataset()
    dataset.add_atoms_data([atoms])
    dataset.setup()
    
    assert dataset.train_idx is not None

def test_train_dataloader(atoms: "Atoms") -> None:
    dataset = Dataset()
    dataset.add_atoms_data([atoms])
    dataset.setup()
    dataloader = dataset.train_dataloader()
    
    assert isinstance(dataloader, DataLoader)

def test_val_dataloader(atoms: "Atoms") -> None:
    dataset = Dataset()
    dataset.add_atoms_data([atoms])
    dataset.setup()
    dataloader = dataset.val_dataloader()
    
    assert isinstance(dataloader, DataLoader)

def test_test_dataloader(atoms: "Atoms") -> None:
    dataset = Dataset()
    dataset.add_atoms_data([atoms])
    dataset.setup()
    dataloader = dataset.test_dataloader()
    
    assert isinstance(dataloader, DataLoader)
