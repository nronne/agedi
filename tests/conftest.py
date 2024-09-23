import pytest

import torch
from agedi.data import AtomsGraph
from torch_geometric.data import Batch

from agedi.diffusion import Diffusion
from agedi.models import ScoreModel    

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
    

@pytest.fixture
def cutoff():
    return 6.0

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def feature_size():
    return 64

@pytest.fixture(params=["schnetpack"])
def package(request, cutoff, feature_size):
    if request.param == "schnetpack":
        import schnetpack as spk
        from agedi.models.schnetpack import SchNetPackTranslator, PositionsScore
        
        feature_size = 64
        input_modules = [
            spk.atomistic.PairwiseDistances(),
        ]

        translator = SchNetPackTranslator(input_modules=input_modules)

        representation = spk.representation.PaiNN(
            n_atom_basis=feature_size,
            n_interactions=4,
            radial_basis=spk.nn.GaussianRBF(n_rbf=30, cutoff=cutoff),
            cutoff_fn=spk.nn.CosineCutoff(cutoff),
        )
        
        heads = [
            PositionsScore(),   # .to(device)
        ]

        return translator, representation, heads


@pytest.fixture(params=["time"])
def conditionings(request):
    if request.param == "time":
        from agedi.models.conditionings import TimeConditioning
        return [TimeConditioning(),]

@pytest.fixture(params=["positions"])
def noisers(request):
    if request.param == "positions":
        from agedi.diffusion.noisers import PositionsNoiser
        return [PositionsNoiser(),]


@pytest.fixture
def diffusion(package, conditionings, noisers):
    translator, representation, heads = package

    score_model = ScoreModel(
        translator=translator,
        representation=representation,
        conditionings=conditionings,
        heads=heads,
    )
    
    diffusion = Diffusion(score_model, noisers)
    
    return diffusion

