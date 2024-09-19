import numpy as np
import pytest
import torch

from agedi.data import AtomsGraph, Representation


def test_from_atoms(atoms: "Atoms") -> None:
    graph = AtomsGraph.from_atoms(atoms)
    assert isinstance(graph, AtomsGraph)

def test_to_atoms(atoms: "Atoms") -> None:
    graph = AtomsGraph.from_atoms(atoms)
    a = graph.to_atoms()

    assert np.allclose(a.positions, atoms.positions)
    assert np.allclose(a.cell, atoms.cell)
    assert np.allclose(a.pbc, atoms.pbc)
    assert np.equal(a.numbers, atoms.numbers).all()

def test_make_graph(atoms: "Atoms") -> None:
    edge_index, shift_vectors = AtomsGraph.make_graph(
        torch.tensor(atoms.positions),
        torch.tensor(np.array(atoms.cell)),
        6.0,
        torch.tensor(atoms.pbc),
    )
    print(edge_index.shape, shift_vectors.shape)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == shift_vectors.shape[0]
    assert shift_vectors.shape[1] == 3

def test_clear_graph(graph: AtomsGraph) -> None:
    graph.clear_graph()

    assert "edge_index" not in graph.keys()
    assert "shift_vectors" not in graph.keys()
    
def test_update_graph(atoms: "Atoms") -> None:
    graph = AtomsGraph.from_atoms(atoms)
    graph.clear_graph()
    graph.update_graph()
    
    assert len(graph) == len(atoms)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == graph.shift_vectors.shape[0]
    assert graph.shift_vectors.shape[1] == 3

def test_copy(graph: AtomsGraph) -> None:
    graph_copy = graph.copy()
    graph_copy.pos[0,0] += 1.0
    assert not np.allclose(graph.pos, graph_copy.pos)

def test_len(atoms: "Atoms") -> None:
    graph = AtomsGraph.from_atoms(atoms)
    assert len(graph) == len(atoms)

@pytest.mark.parametrize("type", ["node", "graph"])
def test_add_batch_attr(type: str, batch: "Batch") -> None:
    if type == "node":
        attr = torch.randn(len(batch), 3)
        t = "x"
    elif type == "graph":
        attr = torch.randn((batch.num_graphs,))
        t = "n_atoms"

    batch.add_batch_attr("test", attr, type=type)

    assert (batch["test"] == attr).all()
    assert (batch._slice_dict["test"] == batch._slice_dict[t]).all()

def test_add_batch_attr_fail(batch: "Batch") -> None:
    attr = torch.randn(1)
    with pytest.raises(ValueError):
        batch.add_batch_attr("test", attr, type="other")

def test_positions_mask(graph: AtomsGraph) -> None:
    mask = graph.positions_mask
    assert mask.shape == (len(graph),3)

def test_pos_setter_clear(graph: AtomsGraph) -> None:
    graph.pos = torch.randn_like(graph.pos)
    
    assert "edge_index" not in graph.keys()
    assert "shift_vectors" not in graph.keys()
    
def test_pos_setter_shape(graph: AtomsGraph) -> None:
    new_pos = torch.randn_like(graph.pos).unsqueeze(0)
    with pytest.raises(IndexError):
        graph.pos = new_pos

def test_frac(graph: AtomsGraph) -> None:
    f = graph.frac
    f = graph.frac # test caching
    a = graph.to_atoms()
    assert np.allclose(f.detach().numpy(), a.get_scaled_positions())

def test_frac_setter(atoms: "Atoms") -> None:
    atoms.positions += 1e-4
    atoms.wrap()
    graph = AtomsGraph.from_atoms(atoms)
    f = torch.tensor(atoms.get_scaled_positions(wrap=True), dtype=torch.float32)
    positions = torch.tensor(atoms.positions, dtype=torch.float32)
    graph.frac = f

    assert torch.allclose(graph.pos, positions)

def test_frac_setter_clear(graph: AtomsGraph) -> None:
    graph.frac = torch.rand_like(graph.frac)

    assert "edge_index" not in graph.keys()
    assert "shift_vectors" not in graph.keys()

def test_pos_frac_batch(batch: "Batch") -> None:
    batch.frac = torch.rand_like(batch.frac)
    
    assert batch.pos.shape[0] == batch.batch.shape[0]

@pytest.mark.parametrize("mask", [True, False])
def test_x_setter_mask(graph: AtomsGraph, mask: bool) -> None:
    if mask:
        graph.mask = torch.rand(graph.mask.shape) > 0.5

    x_old = graph.x.clone()
    x = torch.randint(1, 92, graph.x.shape)
    graph.x = x.clone()

    if mask:
        assert torch.equal(graph.x[graph.mask], x_old[graph.mask])
        assert torch.equal(graph.x[~graph.mask], x[~graph.mask])
    else:
        assert torch.equal(graph.x, x)
                
def test_time_none(graph: AtomsGraph) -> None:
    assert graph.time == None

def test_time_setter(graph: AtomsGraph) -> None:
    t = torch.rand((graph.num_nodes,1), dtype=torch.float32)
    graph.time = t.clone()
    assert torch.equal(graph.time, t)

def test_time_mask(graph: AtomsGraph) -> None:
    t = torch.rand((graph.num_nodes,1), dtype=torch.float32)
    graph.mask = torch.rand(graph.mask.shape) > 0.5
    graph.time = t.clone()
    assert (graph.time[graph.mask] == 0.0).all()
    
def test_wrap(atoms: "Atoms") -> None:
    atoms.positions += 3
    graph = AtomsGraph.from_atoms(atoms)
    atoms.wrap()
    pos = torch.tensor(atoms.positions, dtype=torch.float32)
    
    graph.wrap_positions()
    assert np.allclose(graph.pos, pos)
    
def test_apply_mask(graph: AtomsGraph) -> None:
    mask = torch.rand(graph.mask.shape) > 0.5
    graph.mask = mask.clone()

    x = torch.randn((graph.num_nodes, ))
    masked_x = graph.apply_mask(x, val=-1)
    assert (masked_x[mask] == -1).all()

def test_apply_pos_mask(graph: AtomsGraph) -> None:
    mask = torch.rand(graph.mask.shape) > 0.5
    graph.mask = mask.clone()

    x = torch.randn((graph.num_nodes, 3))
    masked_x = graph.apply_mask(x, val=-1)
    assert (masked_x[mask, :] == -1).all()

def test_apply_mask_error(graph: AtomsGraph) -> None:
    mask = torch.rand(graph.mask.shape) > 0.5
    graph.mask = mask.clone()

    x = torch.randn((graph.num_nodes, 1))
    with pytest.raises(ValueError):
        graph.apply_mask(x, val=-1)

def test_from_prior() -> None:
    graph = AtomsGraph.from_prior(
        n_atoms=3,
        atomic_numbers=[1, 6, 8],
        positions=torch.rand((3, 3)),
        cell=torch.eye(3),
    )
    assert isinstance(graph, AtomsGraph)
        
def test_representation_to_tensor() -> None:
    N, d = 12, 64
    scalar = torch.randn((N, d, 1))
    vector = torch.randn((N, d, 3))
    tensor = torch.randn((N, d, 5))

    rep = Representation(scalar=scalar, vector=vector, tensor=tensor)

    t, _, _ = rep.to_tensor(n_graphs=1)
    assert t.shape == (N, d*9)

def test_representation_from_tensor() -> None:
    N, d = 12, 64
    scalar = torch.randn((N, d, 1))
    vector = torch.randn((N, d, 3))

    rep = Representation(scalar=scalar.clone(), vector=vector.clone())

    tu = rep.to_tensor(n_graphs=1)
    
    rep2 = Representation.from_tensor(*tu)

    assert torch.allclose(scalar, rep2.scalar)
    assert torch.allclose(vector, rep2.vector)

def test_representation_setters() -> None:
    N, d = 12, 64
    scalar = torch.randn((N, d, 1))
    vector = torch.randn((N, d, 3))

    rep = Representation(scalar=scalar.clone(), vector=vector.clone())

    scalar2 = torch.randn((N, d, 1))
    vector2 = torch.randn((N, d, 3))

    rep.scalar = scalar2.clone()
    rep.vector = vector2.clone()

    assert torch.allclose(rep.scalar, scalar2)
    assert torch.allclose(rep.vector, vector2)

def test_get_representation(graph: AtomsGraph) -> None:
    N, d = graph.num_nodes, 64
    scalar = torch.randn((N, d, 1))
    vector = torch.randn((N, d, 3))

    rep = Representation(scalar=scalar.clone(), vector=vector.clone())

    graph.representation = rep

    rep2 = graph.representation

    assert torch.allclose(scalar, rep2.scalar)
    assert torch.allclose(vector, rep2.vector)

    
