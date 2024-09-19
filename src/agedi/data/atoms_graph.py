import functools
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from ase import Atoms
from matscipy.neighbours import neighbour_list
from torch_geometric.data import Batch, Data

from agedi.diffusion.noisers import Noiser


def batched(update_keys: Optional[Sequence[str]] = None, return_batch: bool = False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
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
    def __init__(self, **kwargs):

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
    def scalar(self):
        return self._tensors["l0"]

    @scalar.setter
    def scalar(self, value):
        self._tensors["l0"] = value
    
    @property
    def vector(self):
        return self._tensors["l1"]

    @vector.setter
    def vector(self, value):
        self._tensors["l1"] = value
    
    def to_tensor(self, n_graphs):
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
    def from_tensor(cls, tensor, slices, ls):
        n_nodes = tensor.shape[0]
        slices = slices[0]
        ls = ls[0]
        names = [f"l{l}" for l in ls]
        d = {}
        for i, (l, name) in enumerate(zip(ls, names)):
            d[name] = tensor[:, slices[i].item() : slices[i+1].item()].reshape(n_nodes, -1, 2*l.item()+1)
            
        return cls(**d)


class AtomsGraph(Data):
    @classmethod
    def from_atoms(cls, atoms: Atoms, cutoff=6.0, dtype=torch.float):
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
    def from_prior(
        cls,
        n_atoms,
        atomic_numbers,
        positions,
        cell=None,
        pbc=None,
        template=None,
        cutoff=6.0,
    ):
        if isinstance(n_atoms, Noiser):
            n_atoms = n_atoms.prior()
        else:
            n_atoms = torch.tensor(n_atoms).reshape(1, 1)

        if isinstance(atomic_numbers, Noiser):
            atomic_numbers = atomic_numbers.prior(n_atoms)
        else:
            atomic_numbers = torch.tensor(atomic_numbers).reshape(-1, 1)

        if isinstance(cell, Noiser):
            cell = cell.prior()
        elif cell is not None:
            cell = torch.tensor(cell).reshape(3, 3)
        else:
            assert template is not None

        if template is not None:
            if isinstance(template, AtomsGraph):
                n_template = template.n_atoms.item()
                atomic_numbers = torch.cat([template.x, atomic_numbers])
                template_positions = template.pos
                cell = template.cell
                pbc = template.pbc
            elif isinstance(template, Atoms):
                n_template = len(template)
                atomic_numbers = torch.cat(
                    [torch.tensor(template.get_atomic_numbers()), atomic_numbers]
                )
                template_positions = torch.tensor(template.get_positions())
                cell = torch.tensor(template.get_cell())
                pbc = torch.tensor(template.get_pbc())
            else:
                raise ValueError("Invalid template type.")

            n_atoms += n_template
        else:
            template_positions = torch.empty(0, 3)
            n_template = 0

        if isinstance(positions, Noiser):
            positions = positions.prior(n_atoms, cell)
        else:
            positions = torch.tensor(positions).reshape(-1, 3)

        positions = torch.cat([template_positions, positions])

        if pbc is None:
            pbc = torch.tensor([True, True, True], dtype=torch.bool).reshape(3)

        mask = torch.zeros(n_atoms, dtype=torch.bool)
        mask[:n_template] = True

        edge_index, shift_vectors = cls.make_graph(
            positions, cell, cutoff=cutoff, pbc=pbc
        )

        return cls(
            x=atomic_numbers,
            edge_index=edge_index,
            pos=positions,
            n_atoms=n_atoms,
            cell=cell,
            pbc=pbc,
            shift_vectors=shift_vectors,
            cutoff=cutoff,
            mask=mask,
        )

    def add_batch_attr(self, key, value, type="node"):
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

    def to_atoms(self, threshold=None):
        numbers = self.x.detach().numpy()
        positions = self.pos.detach().numpy()
        return Atoms(
            numbers=numbers,
            positions=positions,
            cell=self.cell.detach().numpy(),
            pbc=self.pbc,
        )

    @staticmethod
    def make_graph(positions, cell, cutoff, pbc, dtype=None):
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
    def update_graph(self):
        device = self.pos.device
        edge_index, shift_vectors = self.make_graph(
            self.pos.detach().cpu(),
            self.cell.detach().cpu(),
            self.cutoff,
            self.pbc.detach().cpu(),
        )
        self.edge_index = edge_index.to(device)
        self.shift_vectors = shift_vectors.to(device)

        # # Why is this here?
        # if self.pbc.any():
        #     atoms = self.get_atoms()
        #     atoms.wrap()
        #     positions = atoms.get_positions()
        #     self.pos.data = torch.tensor(positions, dtype=self.pos.dtype)

    def clear_graph(self):
        del self.edge_index
        del self.shift_vectors

    def copy(self):
        return self.__copy__()

    @batched(return_batch=True)
    def __copy__(self):
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

    def __len__(self):
        return self.pos.shape[0]

    # properties

    @Data.pos.setter
    def pos(self, pos):
        if "pos" in self._store:
            self.clear_graph()
        if "mask" in self._store:
            pos[self.positions_mask] = self.pos[self.positions_mask]
        Data.pos.fset(self, pos)

        if "cell" in self._store:
            f = self.pos_to_frac(self.pos)
            self.add_batch_attr("frac", f, type="node")

    @property
    def frac(self):
        if "frac" in self._store:
            return self["frac"]
        else:
            f = self.pos_to_frac(self.pos)
            self.add_batch_attr("frac", f, type="node")
            return f
        
    @frac.setter
    def frac(self, frac):
        frac %= 1
        if "frac" in self._store:
            self.clear_graph()
        if "mask" in self._store:
            frac[self.positions_mask] = self.frac[self.positions_mask]

        self.add_batch_attr("frac", frac, type="node")
        
        r = self.frac_to_pos(frac)
        Data.pos.fset(self, r)

    def frac_to_pos(self, f):
        cells = self.cell
        if isinstance(self, Batch):
            cells = cells.view(-1, 3, 3)[self.batch]
        else:
            cells = cells.repeat(f.shape[0], 1, 1)

        r = (f.unsqueeze(1) @ cells).reshape(-1, 3)
        return r

    def pos_to_frac(self, r):
        cells = self.cell
        if isinstance(self, Batch):
            cells = cells.view(-1, 3, 3)[self.batch]
        else:
            cells = cells.repeat(r.shape[0], 1, 1)

        f = torch.linalg.solve(torch.transpose(cells, 1, 2), r)
        return f % 1
        
    @Data.x.setter
    def x(self, x):
        if "mask" in self._store:
            x[self.mask] = self.x[self.mask]
        Data.x.fset(self, x)

    @property
    def positions_mask(self):
        pos_mask = torch.zeros_like(self.pos, dtype=torch.bool)
        pos_mask[self.mask, :] = True
        return pos_mask

    @property
    def time(self):
        return self["time"] if "time" in self._store else None

    @time.setter
    def time(self, t):
        if "mask" in self._store:
            t = self.apply_mask(t.squeeze()).unsqueeze(1)
        # self._store.t = t
        self.add_batch_attr("time", t, type="node")

    @property
    def representation(self):
        return (
            Representation.from_tensor(self.repr, self.repr_slices, self.repr_ls)
            if "repr" in self._store
            else None
        )

    @representation.setter
    def representation(self, representation):
        n_graphs = self.num_graphs if "num_graphs" in self._store else 1
        tensor, slices, ls = representation.to_tensor(n_graphs)
        self.add_batch_attr("repr", tensor, type="node")
        self.add_batch_attr("repr_slices", slices, type="graph")
        self.add_batch_attr("repr_ls", ls, type="graph")
        
        # self._store.repr_splits = slice


    # Utility functions:
    @batched(update_keys=["pos"])
    def wrap_positions(self):
        atoms = self.to_atoms()
        atoms.wrap()
        positions = atoms.get_positions()
        self.pos.data = torch.tensor(positions, dtype=self.pos.dtype)

    def apply_mask(self, x, val=0.0):
        if x.shape == self.mask.shape:
            x[self.mask] = val
        elif x.shape == self.positions_mask.shape:
            x[self.positions_mask] = val
        else:
            raise ValueError("Invalid shape for mask.")
        return x

