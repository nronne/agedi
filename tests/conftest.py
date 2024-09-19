import pytest

import torch
from agedi.data import AtomsGraph
from torch_geometric.data import Batch

from ase.build import molecule, bulk, fcc111, bcc100

@pytest.fixture(params=['molecule', 'surface', 'bulk'])
def batch(request: str) -> Batch:
    """
    Create a batch three different types of structures: a molecule, a surface with masking and bulk.
    """
    graphs = []
    if request.param == 'molecule':
        for s in ['H2', 'H2O', 'NH3', 'CH4']:
            a = molecule('H2O')
            a.set_cell([10, 10, 10])
            a.set_pbc(True)
            a.center()
            graphs.append(AtomsGraph.from_atoms(a))
        

    elif request.param == 'surface':
        a = fcc111('Au', (3, 3, 3), vacuum=10)
        a.set_pbc(True)
        g = AtomsGraph.from_atoms(a)
        g.mask[:9] = True
        graphs.append(g)

        a = bcc100('Cu', (4, 4, 2), vacuum=12, a=3.6)
        a.set_pbc(True)
        graphs.append(AtomsGraph.from_atoms(a))
        

    elif request.param == 'bulk':
        a = bulk('Cu', 'fcc', a=3.6, cubic=True)
        a.set_pbc(True)
        graphs.append(AtomsGraph.from_atoms(a))

        a = bulk('Al', 'bcc', a=3.6)
        a.set_pbc(True)
        graphs.append(AtomsGraph.from_atoms(a))
        

    batch = Batch.from_data_list(graphs)
    
    return batch
    
    
