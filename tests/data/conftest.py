import pytest

from agedi.data import AtomsGraph
from ase.build import molecule, bulk, fcc111

@pytest.fixture(params=['molecule', 'surface', 'bulk'])
def atoms(request) -> 'Atoms':
    """
    Create a batch three different types of structures: a molecule, a surface with masking and bulk.
    """
    if request.param == 'molecule':
        a = molecule('H2O')
        a.set_cell([10, 10, 10])
        a.set_pbc(True)
        a.center()
    elif request.param == 'surface':
        a = fcc111('Au', (3, 3, 3), vacuum=10)
        a.set_pbc(True)
    elif request.param == 'bulk':
        a = bulk('Cu', 'fcc', a=3.6, cubic=True)
        a.set_pbc(True)
        
    return a
        
    



@pytest.fixture()
def graph(atoms: "Atoms") -> 'AtomsGraph':
    """
    Create a batch three different types of structures: a molecule, a surface with masking and bulk.
    """
    return AtomsGraph.from_atoms(atoms)
        
