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


def test_add_regressor_data(atoms: "Atoms") -> None:
    """add_regressor_data should populate regressor_dataset."""
    atoms.calc = sp(atoms, energy=-1.0, forces=np.zeros((len(atoms), 3)))
    dataset = Dataset()
    dataset.add_atoms_data([atoms])
    dataset.add_regressor_data([atoms])
    assert dataset.regressor_dataset is not None
    assert len(dataset.regressor_dataset) == 1
    assert isinstance(dataset.regressor_dataset[0], AtomsGraph)


def test_add_regressor_data_stores_energy_forces(atoms: "Atoms") -> None:
    """Graphs in the regressor dataset should carry energy and forces."""
    atoms.calc = sp(atoms, energy=-2.5, forces=np.ones((len(atoms), 3)))
    dataset = Dataset()
    dataset.add_atoms_data([atoms])
    dataset.add_regressor_data([atoms])
    graph = dataset.regressor_dataset[0]
    assert hasattr(graph, "energy")
    assert hasattr(graph, "forces")


def test_add_regressor_data_no_calc(atoms: "Atoms") -> None:
    """add_regressor_data should work even when no calculator is attached."""
    dataset = Dataset()
    dataset.add_atoms_data([atoms])
    dataset.add_regressor_data([atoms])
    assert len(dataset.regressor_dataset) == 1


def test_add_regressor_data_accumulates(atoms: "Atoms") -> None:
    """Calling add_regressor_data twice should accumulate entries."""
    dataset = Dataset()
    dataset.add_atoms_data([atoms])
    dataset.add_regressor_data([atoms])
    dataset.add_regressor_data([atoms])
    assert len(dataset.regressor_dataset) == 2


def test_train_dataloader_returns_combined_loader_with_regressor(atoms: "Atoms") -> None:
    """When regressor data is present train_dataloader should return a CombinedLoader."""
    from lightning.pytorch.utilities import CombinedLoader

    atoms.calc = sp(atoms, energy=-1.0, forces=np.zeros((len(atoms), 3)))
    dataset = Dataset()
    dataset.add_atoms_data([atoms])
    dataset.add_regressor_data([atoms])
    dataset.setup()
    loader = dataset.train_dataloader()
    assert isinstance(loader, CombinedLoader)


def test_train_dataloader_plain_without_regressor(atoms: "Atoms") -> None:
    """Without regressor data the train_dataloader should return a plain DataLoader."""
    dataset = Dataset()
    dataset.add_atoms_data([atoms])
    dataset.setup()
    loader = dataset.train_dataloader()
    assert isinstance(loader, DataLoader)


def test_combined_loader_has_main_and_regressor_keys(atoms: "Atoms") -> None:
    """The CombinedLoader should expose 'main' and 'regressor' iterables."""
    from lightning.pytorch.utilities import CombinedLoader

    atoms.calc = sp(atoms, energy=-1.0, forces=np.zeros((len(atoms), 3)))
    dataset = Dataset(batch_size=1)
    dataset.add_atoms_data([atoms])
    dataset.add_regressor_data([atoms])
    dataset.setup()
    loader = dataset.train_dataloader()
    assert isinstance(loader, CombinedLoader)
    # Iterate one step and verify the dict structure.
    # CombinedLoader yields (batch_dict, batch_idx, dataloader_idx) tuples.
    item = next(iter(loader))
    batch = item[0] if isinstance(item, tuple) else item
    assert "main" in batch
    assert "regressor" in batch

