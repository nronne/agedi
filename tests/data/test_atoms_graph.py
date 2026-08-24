import numpy as np
import pytest
import torch
from ase import Atoms
from torch_geometric.data import Batch

import agedi.data.atoms_graph as atoms_graph_module
from agedi.data import AtomsGraph, Representation


def test_from_atoms(atoms: "Atoms") -> None:
    graph = AtomsGraph.from_atoms(atoms)
    assert isinstance(graph, AtomsGraph)

def test_to_atoms(atoms: "Atoms") -> None:
    graph = AtomsGraph.from_atoms(atoms)
    a = graph.to_atoms()

    # With canonical_cell=False (default) the cell is stored as-is so
    # fractional coordinates are trivially preserved.
    orig_frac = atoms.get_scaled_positions(wrap=False)
    new_frac = a.get_scaled_positions(wrap=False)
    assert np.allclose(
        (orig_frac + 0.5) % 1, (new_frac + 0.5) % 1, atol=1e-5
    )
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
    close1 = np.isclose(f.detach().numpy(), a.get_scaled_positions())
    close2 = np.isclose((f.detach().numpy()+0.5)%1.0, (a.get_scaled_positions()+0.5)%1.0)
    allclose = np.logical_or(close1, close2)
    assert allclose.all()

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
    assert graph.time is None

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

def test_empty() -> None:
    graph = AtomsGraph.empty()
    assert isinstance(graph, AtomsGraph)


def test_cell_is_canonical(atoms) -> None:
    """Cell stored in AtomsGraph must be canonical (cellpar round-trip) when canonical_cell=True."""
    graph = AtomsGraph.from_atoms(atoms, canonical_cell=True)
    cell = graph.cell
    cell_params = AtomsGraph.cell_to_vectors(cell)
    canonical = AtomsGraph.vector_to_cell(cell_params).view(3, 3)
    assert torch.allclose(cell, canonical, atol=1e-5)


def test_cell_setter_preserves_frac(graph: AtomsGraph) -> None:
    """Setting a new cell must not change fractional coordinates."""
    frac_before = graph.frac.clone()
    # Rotate the cell slightly by applying a small perturbation to the cellpar
    cell_params = AtomsGraph.cell_to_vectors(graph.cell).squeeze(0)
    cell_params[3] += 0.05  # shift alpha a little
    new_cell = AtomsGraph.vector_to_cell(cell_params).view(3, 3)
    graph.cell = new_cell
    frac_after = graph.frac
    assert torch.allclose(frac_before, frac_after, atol=1e-5)


def test_cell_setter_clears_graph(graph: AtomsGraph) -> None:
    """Setting the cell must invalidate the edge index."""
    cell_params = AtomsGraph.cell_to_vectors(graph.cell).squeeze(0)
    cell_params[0] += 0.1
    graph.cell = AtomsGraph.vector_to_cell(cell_params).view(3, 3)
    assert "edge_index" not in graph.keys()
    assert "shift_vectors" not in graph.keys()



def test_representation_to_tensor() -> None:
    N, d = 12, 64
    scalar = torch.randn((N, d, 1))
    vector = torch.randn((N, d, 3))

    rep = Representation(scalar=scalar, vector=vector)

    t, _, _ = rep.to_tensor(n_graphs=1)
    assert t.shape == (N, d * 4)

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


# ---------------------------------------------------------------------------
# Non-periodic neighbor list (_make_graph_no_pbc)
# ---------------------------------------------------------------------------

def test_make_graph_no_pbc_returns_correct_edges() -> None:
    """_make_graph_no_pbc must find all pairs within cutoff, no self-loops."""
    cutoff = 3.0
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=torch.float32)
    ei, sv = AtomsGraph._make_graph_no_pbc(pos, cutoff)

    assert ei.shape[0] == 2
    assert sv.shape == (ei.shape[1], 3)
    assert (sv == 0).all(), "Shift vectors must be zero for non-periodic graphs"

    # Atoms 0 and 1 are within cutoff; atom 2 is not.
    pairs = set(map(tuple, ei.T.tolist()))
    assert (0, 1) in pairs
    assert (1, 0) in pairs
    assert (0, 2) not in pairs
    assert (2, 0) not in pairs
    # No self-loops
    assert all(s != d for s, d in pairs)


def test_make_graph_no_pbc_all_pairs_within_cutoff() -> None:
    """Edge distances must never exceed the cutoff."""
    cutoff = 4.0
    pos = torch.randn(10, 3, dtype=torch.float32) * 2
    ei, _ = AtomsGraph._make_graph_no_pbc(pos, cutoff)
    src, dst = ei
    dists = (pos[dst] - pos[src]).norm(dim=-1)
    assert (dists <= cutoff + 1e-5).all()


def test_make_graph_no_pbc_matches_matscipy() -> None:
    """_make_graph_no_pbc and _make_graph_matscipy must agree on a small molecule."""
    from ase.build import molecule
    cutoff = 5.0
    atoms = molecule("H2O")
    pos = torch.tensor(atoms.positions, dtype=torch.float32)
    # Provide a cell large enough that pbc=False matscipy gives the same result.
    cell = torch.eye(3) * 50.0
    pbc = torch.tensor([False, False, False])

    ei_nopbc, _ = AtomsGraph._make_graph_no_pbc(pos, cutoff)
    ei_matscipy, _ = AtomsGraph._make_graph_matscipy(pos, cell, cutoff, pbc)

    pairs_nopbc = set(map(tuple, ei_nopbc.T.tolist()))
    pairs_matscipy = set(map(tuple, ei_matscipy.T.tolist()))
    assert pairs_nopbc == pairs_matscipy


def test_make_graph_no_pbc_batched() -> None:
    """Batched _make_graph_no_pbc must not create cross-graph edges."""
    cutoff = 3.0
    pos = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # graph 0 — close pair
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # graph 1 — same positions, different graph
    ], dtype=torch.float32)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

    ei, sv = AtomsGraph._make_graph_no_pbc(pos, cutoff, batch_idx=batch_idx)

    assert (sv == 0).all()
    pairs = set(map(tuple, ei.T.tolist()))
    # Intra-graph edges only
    assert (0, 1) in pairs
    assert (1, 0) in pairs
    assert (2, 3) in pairs
    assert (3, 2) in pairs
    # No cross-graph edges
    assert (0, 2) not in pairs
    assert (1, 3) not in pairs


def test_make_graph_no_pbc_via_make_graph() -> None:
    """make_graph must route to _make_graph_no_pbc when pbc is all False."""
    cutoff = 5.0
    pos = torch.randn(8, 3, dtype=torch.float32)
    cell = torch.zeros(3, 3)
    pbc = torch.tensor([False, False, False])

    ei_route, sv_route = AtomsGraph.make_graph(pos, cell, cutoff, pbc)
    ei_direct, sv_direct = AtomsGraph._make_graph_no_pbc(pos, cutoff)

    assert ei_route.shape == ei_direct.shape
    assert (sv_route == 0).all()


def test_from_atoms_no_pbc_no_cell() -> None:
    """from_atoms must work for a gas-phase molecule with pbc=False and no cell."""
    from ase.build import molecule
    atoms = molecule("H2O")  # pbc=False, cell=zeros by default
    graph = AtomsGraph.from_atoms(atoms, cutoff=5.0)
    assert graph.edge_index.shape[0] == 2
    assert graph.shift_vectors.shape[1] == 3
    assert (graph.shift_vectors == 0).all()

