import functools
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from matscipy.neighbours import neighbour_list
from torch_geometric.data import Batch, Data

def batched(update_keys: Optional[Sequence[str]] = None, return_batch: bool = False) -> Callable:
    """Batched decorator
    
    Decorator for functions that return Data objects, but can with this operator be
    called with batched inputs. The function will be called for each element in the
    batch, and the results will be concatenated into a single Data object.

    If called with a Data-object as input, the function will be called with as if it
    not decorated.
    

    Parameters
    ----------
    update_keys: Optional[Sequence[str]]
        The keys in the Batch object that should be updated. If None, no keys will be updated.
    return_batch: bool
        If True, the function will return a Batch object instead of None.

    Returns
    -------
    Callable
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Union[Data, Batch]:
            if isinstance(self, Batch):
                data_list = self.to_data_list()
                for d in data_list:
                    func(d, *args, **kwargs)

                new_batch = Batch.from_data_list(data_list)
                if update_keys is not None:
                    for key in update_keys:
                        setattr(self, key, new_batch[key])
                if return_batch:
                    return new_batch
            elif isinstance(self, Data):
                return func(self, *args, **kwargs)
            else:
                raise TypeError("Object must be of type Data or Batch.")

        return wrapper

    return decorator

class Representation:
    """Representation class
    
    Class defining a general representation. The representation is a dictionary of tensors, where each tensor
    is a representation of a certain type of information. The tensors are stored in a dictionary, where the keys
    are degree of the representation l (with dim = 2l+1), and the values are the tensors themselves.

    The representation can be initialized with either a scalar or a vector representation, or both. The scalar
    representation is a tensor of shape (n_nodes, n_features, 1), and the vector representation is a tensor of shape
    (n_nodes, n_features, 3). The representation can be accessed with the properties scalar and vector, respectively.

    Parameters
    ----------
    scalar: Optional[torch.Tensor]
        The scalar representation of the atoms. Default is None.
    vector: Optional[torch.Tensor]
        The vector representation of the atoms. Default is None.
    kwargs: Dict[str, torch.Tensor]
        Additional representations of the atoms. The keys are the degrees of the representations, and the values are

    Returns
    -------
    Representation
    
    """
    def __init__(self, **kwargs):
        """Initialize the representation with the given tensors.
        
        """

        scalar = kwargs.pop("scalar", None)
        if scalar is not None:
            kwargs["l0"] = scalar

        vector = kwargs.pop("vector", None)
        if vector is not None:
            kwargs["l1"] = vector

        self._tensors = {}
        for key, value in kwargs.items():
            self._tensors[key] = value

    @property
    def scalar(self) -> torch.Tensor:
        """Return the scalar representation tensor.

        Returns
        -------
        torch.Tensor
        """
        return self._tensors["l0"]

    @scalar.setter
    def scalar(self, value: torch.Tensor) -> None:
        """Set the scalar representation tensor.

        Parameters
        ----------
        value: torch.Tensor
            The new scalar representation tensor.

        Returns
        -------
        None
        
        """
        self._tensors["l0"] = value
    
    @property
    def vector(self) -> torch.Tensor:
        """Return the vector representation tensor.

        Returns
        -------
        torch.Tensor
        
        """
        return self._tensors["l1"]

    @vector.setter
    def vector(self, value: torch.Tensor) -> None:
        """Set the vector representation tensor.

        Parameters
        ----------
        value: torch.Tensor
            The new vector representation tensor.

        Returns
        -------
        None
        """
        self._tensors["l1"] = value
    
    def to_tensor(self, n_graphs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to tensor
        
        Convert the representation to a tensor. The representations are concatenated along the
        1st dimension, and the resulting tensor is returned along with slices and names of the
        representations.

        Parameters
        ----------
        n_graphs: int
            The number of graphs in the batch.

        Returns
        -------
        tensor: torch.Tensor
            The tensor representation of the batch with shape (n_nodes, n_features).
        slices: torch.Tensor
            The slices of the tensor representation with shape (n_graphs, n_slices).
        ls: torch.Tensor
            The degrees of the tensor representation with shape (n_graphs, 1).
        
        """
        nodes = self.scalar.shape[0]

        tensor = []
        slices = [0]
        ls = []
        for name, value in self._tensors.items():
            ls.append((value.shape[2] - 1)/2)
            tensor.append(value.reshape(nodes, -1))
            slices.append(tensor[-1].shape[1])


        slices = torch.cumsum(torch.tensor(slices, dtype=int), dim=0).repeat(n_graphs, 1)
        tensor = torch.cat(tensor, dim=1)
        ls = torch.tensor(ls, dtype=int).repeat(n_graphs, 1)

        return tensor, slices, ls

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, slices: torch.Tensor, ls: torch.Tensor) -> "Representation":
        """Get representation from tensor
        
        Create a representation from a tensor. The tensor is split into the different representations
        according to the slices and the degrees of the representations are given by ls.

        Parameters
        ----------
        tensor: torch.Tensor
            The tensor representation of the batch with shape (n_nodes, n_features).
        slices: torch.Tensor
            The slices of the tensor representation with shape (n_graphs, n_slices).
        ls: torch.Tensor
            The degrees of the tensor representation with shape (n_graphs, 1).

        Returns
        -------
        representation: Representation
            The representation object.

        """
        n_nodes = tensor.shape[0]
        slices = slices[0]
        ls = ls[0]
        names = [f"l{l}" for l in ls]
        d = {}
        for i, (l, name) in enumerate(zip(ls, names)):
            d[name] = tensor[:, slices[i].item() : slices[i+1].item()].reshape(n_nodes, -1, 2*l.item()+1)
            
        return cls(**d)

class AtomsGraph(Data):
    """Atomistic Graph Class
    
    Class defining a graph with atoms as nodes and edges formed between all atoms
    within a finite curoff.formed betw

    Parameters
    ----------
    pos: torch.Tensor
        The positions of the atoms with shape (n_atoms, 3).
    x: torch.Tensor
        The node features i.e atomic types of the graph with shape (n_nodes, 1).
    edge_index: torch.Tensor
        The edge index tensor of the graph with shape (2, n_edges).
    edge_attr: torch.Tensor
        The edge attributes of the graph with shape (n_edges, n_edge_features).
    y: Optional[torch.Tensor]
        The target tensor of the graph with shape (n_targets,).
    representation: Optional[Representation]
        The representation of the atoms in the graph.
    kwargs: Dict[str, torch.Tensor]

    """
    @classmethod
    def from_atoms(cls, atoms: Atoms, cutoff: int=6.0, dtype: torch.dtype=torch.float) -> "AtomsGraph":
        """Create a graph from an ASE Atoms object.

        Parameters
        ----------
        atoms: Atoms
            The ASE Atoms object.
        cutoff: float
            The cutoff radius for the edges.
        dtype: torch.dtype
            The data type of the tensors.

        Returns
        -------
        graph: AtomsGraph
            The graph object.
        
        """
        # Nodes: The initial node features are just the atomic numbers.
        node_feat = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long).reshape(
            -1
        )
        mask = torch.zeros_like(node_feat, dtype=torch.bool)

        positions = torch.tensor(atoms.get_positions(), dtype=dtype)
        cell = torch.tensor(np.array(atoms.get_cell()), dtype=dtype)
        pbc = torch.tensor(atoms.get_pbc())

        edge_index, shift_vectors = cls.make_graph(positions, cell, cutoff, pbc)

        n_atoms = torch.tensor([len(atoms)]).reshape(1, 1)

        return cls(
            x=node_feat,
            edge_index=edge_index,
            pos=positions,
            n_atoms=n_atoms,
            cell=cell,
            pbc=pbc,
            shift_vectors=shift_vectors,
            cutoff=cutoff,
            mask=mask,
        )

    @classmethod
    def empty(
        cls,
        cutoff: int=6.0
    ) -> "AtomsGraph":
        """Create an empty graph.

        Parameters
        ----------
        cutoff: float
            The cutoff radius for the edges.

        Returns
        -------
        graph: AtomsGraph
            The graph object.

        """
        return cls(
            x=torch.empty(0, dtype=torch.long),
            pos=torch.empty(0, 3),
            n_atoms=torch.tensor([0]),
            cell=torch.empty(3, 3),
            pbc=torch.tensor([True, True, True], dtype=torch.bool),
            cutoff=cutoff,
            # mask=torch.empty(0, dtype=torch.bool),
        )

    def add_batch_attr(self, key: str, value: torch.Tensor, type: str="node") -> None:
        """Add a batch attribute to the graph.

        Parameters
        ----------
        key: str
            The key of the attribute.
        value: torch.Tensor
            The value of the attribute.
        type: str
            The type of the attribute. Can be either "node" or "edge".

        Returns
        -------
        None
        
        """
        self._store[key] = value

        if hasattr(self, "_slice_dict"):
            if type == "node":
                k = "x"
            elif type == "graph":
                k = "n_atoms"
            else:
                raise ValueError("Invalid type")

            self._slice_dict[key] = self._slice_dict[k]
            self._inc_dict[key] = self._inc_dict[k]

    def to_atoms(self) -> Atoms:
        """Convert the graph to an ASE Atoms object.

        Only works on unbatched graphs.

        Returns
        -------
        atoms: ase.Atoms
            The atoms object.
        
        """
        numbers = self.x.detach().numpy()
        positions = self.pos.detach().numpy()
        return Atoms(
            numbers=numbers,
            positions=positions,
            cell=self.cell.detach().numpy(),
            pbc=self.pbc,
        )

    @staticmethod
    def make_graph(
        positions: torch.Tensor,
        cell: torch.Tensor,
        cutoff: int,
        pbc: torch.Tensor,
        dtype: torch.dtype=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """Create the graph-edges from the positions and cell.

        Parameters
        ----------
        positions: torch.Tensor
            The positions of the atoms.
        cell: torch.Tensor
            The cell of the system.
        cutoff: float
            The cutoff radius for the edges.
        pbc: torch.Tensor
            The periodic boundary conditions.
        dtype: torch.dtype
            The data type of the output.

        Returns
        -------
        edge_index: torch.Tensor
            The edge index tensor.
        shift_vectors: torch.Tensor
            The shift vectors tensor.
        
        """
        if dtype is None:
            dtype = positions.dtype

        with torch.no_grad():
            i, j, S = neighbour_list(
                "ijS", positions=positions, cell=cell, cutoff=cutoff, pbc=pbc
            )

            ij = np.array([i, j])
            edge_index = torch.tensor(ij, dtype=torch.long)

            # Shift vectors:
            shifts = torch.tensor(
                S, dtype=dtype
            )  # These are integer shifts in the unit cell.
            shift_vectors = torch.einsum(
                "ij,jk->ik", shifts, cell
            )  # Convert the shifts to vectors in Å.

        return edge_index, shift_vectors

    @batched(update_keys=["edge_index", "shift_vectors"])
    def update_graph(self) -> None:
        """Update the graph with new edges
        
        This should be called after changing any of the positions or cell.

        Returns
        -------
        None
        
        """

        cutoff = self.cutoff.item() if isinstance(self.cutoff, torch.Tensor) else self.cutoff
        
        device = self.pos.device
        edge_index, shift_vectors = self.make_graph(
            self.pos.detach().cpu(),
            self.cell.detach().cpu(),
            cutoff,
            self.pbc.detach().cpu(),
        )
        self.edge_index = edge_index.to(device)
        self.shift_vectors = shift_vectors.to(device)

        # # Why is this here?
        # if self.pbc.any():
        #     atoms = self.get_atoms()
        #     atoms.wrap()
        #     positions = atoms.get_positions()
        #     self.pos.dat

    def clear_graph(self) -> None:
        """Clear the graph removing all edges

        Returns
        -------
        None
        """
        del self.edge_index
        del self.shift_vectors

    def copy(self) -> "AtomsGraph":
        """ Copy the graph.
        
        Create a copy of the graph with each tensor cloned.

        Returns
        -------
        copy: Graph
            The copied graph.
        
        """    
        return self.__copy__()

    @batched(return_batch=True)
    def __copy__(self) -> "AtomsGraph":
        """Copy the graph.

        Returns
        -------
        copy: Graph
            The copied graph.

        """
        return AtomsGraph(
            x=self.x.clone(),
            edge_index=self.edge_index.clone(),
            pos=self.pos.clone(),
            n_atoms=self.n_atoms.clone(),
            cell=self.cell.clone(),
            pbc=self.pbc.clone(),
            shift_vectors=self.shift_vectors.clone(),
            cutoff=self.cutoff,
            mask=self.mask.clone(),
        )

    def __len__(self) -> int:
        """Return the number of atoms in the graph.

        Returns
        -------
        n_atoms: int
            The number of atoms in the graph.

        """
        return self.pos.shape[0]

    @Data.pos.setter
    def pos(self, pos: torch.Tensor) -> None:
        """Set the positions of the atoms.

        Parameters
        ----------
        pos: torch.Tensor
            The new positions of the atoms.

        Returns
        -------
        None
        
        """
        if "pos" in self._store:
            self.clear_graph()
        if "mask" in self._store:
            pos[self.positions_mask] = self.pos[self.positions_mask]
        Data.pos.fset(self, pos)

        if "cell" in self._store:
            f = self.pos_to_frac(self.pos)
            self.add_batch_attr("frac", f, type="node")

    @property
    def frac(self) -> torch.Tensor:
        """Return the fractional coordinates of the positions

        Returns
        -------
        frac: torch.Tensor
            The fractional coordinates of the atoms.
        
        """
        if "frac" in self._store:
            return self["frac"]
        else:
            f = self.pos_to_frac(self.pos)
            self.add_batch_attr("frac", f, type="node")
            return f
        
    @frac.setter
    def frac(self, frac: torch.Tensor) -> None:
        """Set fractional coordinates.

        All positions are also updated.

        Parameters
        ----------
        frac: torch.Tensor
            The fractional coordinates of the atoms.

        Returns
        -------
        None
        
        """
        frac %= 1
        if "frac" in self._store:
            self.clear_graph()
        if "mask" in self._store:
            frac[self.positions_mask] = self.frac[self.positions_mask]

        self.add_batch_attr("frac", frac, type="node")
        
        r = self.frac_to_pos(frac)
        Data.pos.fset(self, r)

    def frac_to_pos(self, f: torch.Tensor) -> torch.Tensor:
        """Fraction -> Cartesian coordinates.
        
        Convert fractional coordinates to cartesian coordinates.

        Parameters
        ----------
        f: torch.Tensor
            The fractional coordinates.

        Returns
        -------
        r: torch.Tensor
            The cartesian coordinates.
        
        """
        cells = self.cell
        if isinstance(self, Batch):
            cells = cells.view(-1, 3, 3)[self.batch]
        else:
            cells = cells.repeat(f.shape[0], 1, 1)

        r = (f.unsqueeze(1) @ cells).reshape(-1, 3)
        return r

    def pos_to_frac(self, r: torch.Tensor) -> torch.Tensor:
        """Cartesian -> Fractional coordinates.
        
        Convert cartesian coordinates to fractional coordinates.

        Parameters
        ----------
        r: torch.Tensor
            The cartesian coordinates.

        Returns
        -------
        f: torch.Tensor
            The fractional coordinates.
        
        """
        cells = self.cell
        if isinstance(self, Batch):
            cells = cells.view(-1, 3, 3)[self.batch]
        else:
            cells = cells.repeat(r.shape[0], 1, 1)

        f = torch.linalg.solve(torch.transpose(cells, 1, 2), r)
        return f % 1
        
    @Data.x.setter
    def x(self, x: torch.Tensor) -> None:
        """Set the atomic types of the graph
        
        Parameters
        ----------
        x: torch.Tensor
            The atomic types of the graph

        Returns
        -------
        None
        
        """
        if "mask" in self._store:
            x[self.mask] = self.x[self.mask]
        Data.x.fset(self, x)

    @property
    def positions_mask(self) -> torch.Tensor:
        """Return the mask of the positions that are fixed.
        
        True for fixed atom-positions and else false.

        Returns
        -------
        mask: torch.Tensor
            The mask of the positions that are fixed.
        
        """
        pos_mask = torch.zeros_like(self.pos, dtype=torch.bool)
        pos_mask[self.mask, :] = True
        return pos_mask

    @property
    def time(self) -> torch.Tensor:
        """Return the time of the graph.

        Returns
        -------
        time: torch.Tensor
            The time of the graph.
        """
        return self["time"] if "time" in self._store else None

    @time.setter
    def time(self, t: torch.Tensor) -> None:
        """Set the time of the graph.

        Parameters
        ----------
        t: torch.Tensor
            The time of the graph.

        Returns
        -------
        None
        
        """
        if "mask" in self._store:
            t = self.apply_mask(t.squeeze()).unsqueeze(1)
        # self._store.t = t
        self.add_batch_attr("time", t, type="node")

    @property
    def representation(self) -> Representation:
        """Return the representation of the graph.

        Returns
        -------
        representation: Representation
            The representation of the graph.
        
        """
        return (
            Representation.from_tensor(self.repr, self.repr_slices, self.repr_ls)
            if "repr" in self._store
            else None
        )

    @representation.setter
    def representation(self, representation: Representation) -> None:
        """Set the representation of the graph.

        Parameters
        ----------
        representation: Representation
            The representation of the graph.

        Returns
        -------
        None
        
        """
        n_graphs = self.num_graphs if "num_graphs" in self._store else 1
        tensor, slices, ls = representation.to_tensor(n_graphs)
        
        self.add_batch_attr("repr", tensor, type="node")
        self.add_batch_attr("repr_slices", slices.repeat(self.n_atoms.shape[0], 1), type="graph")
        self.add_batch_attr("repr_ls", ls.repeat(self.n_atoms.shape[0], 1), type="graph")
        
    @batched(update_keys=["pos"])
    def wrap_positions(self) -> None:
        """Wrap the positions of the atoms to the unit cell.
        
        Returns
        -------
        None
        
        """
        atoms = self.to_atoms()
        atoms.wrap()
        positions = atoms.get_positions()
        self.pos.data = torch.tensor(positions, dtype=self.pos.dtype)

    def apply_mask(self, x: torch.Tensor, val: float=0.0) -> torch.Tensor:
        """Apply the mask to the tensor x.

        Parameters
        ----------
        x: torch.Tensor
            The tensor to apply the mask to.
        val: float
            The value to set the masked values to.

        Returns
        -------
        x: torch.Tensor
            The tensor with the mask applied.
        
        """
        
        if x.shape == self.mask.shape:
            x[self.mask] = val
        elif x.shape == self.positions_mask.shape:
            x[self.positions_mask] = val
        else:
            raise ValueError("Invalid shape for mask.")
        return x

