import functools
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from matscipy.neighbours import neighbour_list
from torch_geometric.data import Batch, Data


def batched(
    update_keys: Optional[Sequence[str]] = None, return_batch: bool = False
) -> Callable:
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
        """Wrap *func* so it can be called on both :class:`Data` and :class:`Batch` objects."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Union[Data, Batch]:
            """Dispatch the wrapped function to each element in a batch or call it directly."""
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
        """Initialize the representation with the given tensors."""

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
            ls.append((value.shape[2] - 1) / 2)
            tensor.append(value.reshape(nodes, -1))
            slices.append(tensor[-1].shape[1])

        slices = torch.cumsum(torch.tensor(slices, dtype=int), dim=0).repeat(
            n_graphs, 1
        )
        tensor = torch.cat(tensor, dim=1)
        ls = torch.tensor(ls, dtype=int).repeat(n_graphs, 1)

        return tensor, slices, ls

    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor, slices: torch.Tensor, ls: torch.Tensor
    ) -> "Representation":
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
            d[name] = tensor[:, slices[i].item() : slices[i + 1].item()].reshape(
                n_nodes, -1, 2 * l.item() + 1
            )

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
    confinement: Optional[torch.Tensor]
        z-directional confinement of the atoms with shape (1,2).
    kwargs: Dict[str, torch.Tensor]

    """

    @classmethod
    def from_atoms(
        cls,
        atoms: Atoms,
        cutoff: int = 6.0,
        dtype: torch.dtype = torch.float,
        initialize_mask: Optional[bool] = None,
        confinement: Optional[Tuple[float, float]] = None,
        canonical_cell: bool = False,
        use_pbc_graph: bool = True,
    ) -> "AtomsGraph":
        """Create a graph from an ASE Atoms object.

        Parameters
        ----------
        atoms: Atoms
            The ASE Atoms object.
        cutoff: float
            The cutoff radius for the edges.
        dtype: torch.dtype
            The data type of the tensors.
        initialize_mask: Optional[bool]
            Whether to initialize the mask tensor.  When ``None`` (the
            default), the mask is initialised only when ``confinement`` is
            not provided (i.e. ``initialize_mask`` defaults to ``False``
            for template / confinement graphs).
        confinement: Optional[Tuple[float, float]]
            Optional z-directional confinement bounds ``(z_min, z_max)`` to
            attach to the graph.  When provided, a ``confinement`` tensor of
            shape ``(1, 2)`` is stored on the graph.  When ``None`` (the
            default), no confinement attribute is added.
        canonical_cell: bool
            When ``True`` (the default), the cell is stored in canonical
            lower-triangular form.  If the input cell is not already
            canonical, Cartesian positions are recomputed to preserve
            fractional coordinates and a warning is printed.  Set to
            ``False`` to store the cell exactly as provided by ASE (no
            rotation or recomputation is performed).
        use_pbc_graph: bool
            When ``True`` (the default) and all periodic boundary conditions
            are enabled, :meth:`make_graph_pbc` is used to build the graph
            instead of the matscipy-based :meth:`make_graph`.
            :meth:`make_graph_pbc` is more robust for small or highly
            non-orthorhombic unit cells.  Set to ``False`` to always use
            :meth:`make_graph`.

        Returns
        -------
        graph: AtomsGraph
            The graph object.

        """
        if initialize_mask is None:
            initialize_mask = confinement is None

        # Nodes: The initial node features are just the atomic numbers.
        kwargs = {
            "cutoff": cutoff,
            "use_pbc_graph": use_pbc_graph,
        }

        kwargs["x"] = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.long
        ).reshape(-1)
        if initialize_mask:
            kwargs["mask"] = torch.zeros_like(kwargs["x"], dtype=torch.bool)

        # Canonicalize the cell (cellpar -> cell round-trip) and update
        # Cartesian positions so that fractional coordinates are preserved.
        # Use float64 for the round-trip to avoid precision loss from log/exp.
        cell_np = np.array(atoms.get_cell())
        pos_np = np.array(atoms.get_positions())
        cell_f64 = torch.tensor(cell_np, dtype=torch.float64)

        if not canonical_cell:
            # Caller opted out: store the cell exactly as provided by ASE.
            final_cell_f64 = cell_f64
            final_pos = torch.tensor(pos_np, dtype=dtype)
        elif cls._is_lower_triangular(cell_f64):
            # Cell is already in canonical lower-triangular form; keep it as-is
            # to avoid introducing floating-point rounding artefacts.
            final_cell_f64 = cell_f64
            final_pos = torch.tensor(pos_np, dtype=dtype)
        else:
            print(
                "AtomsGraph.from_atoms: cell is not in canonical lower-triangular "
                "form; canonicalizing. Cartesian positions will be recomputed to "
                "preserve fractional coordinates."
            )
            final_cell_f64 = cls.vector_to_cell(cls.cell_to_vectors(cell_f64)).view(3, 3)
            pos_f64 = torch.tensor(pos_np, dtype=torch.float64)
            frac_f64 = torch.linalg.solve(cell_f64.T, pos_f64.T).T
            final_pos = (frac_f64 @ final_cell_f64).to(dtype)

        kwargs["pos"] = final_pos
        kwargs["cell"] = final_cell_f64.to(dtype)
        kwargs["pbc"] = torch.tensor(atoms.get_pbc())

        if use_pbc_graph and kwargs["pbc"].all():
            edge_index, shift_vectors = cls.make_graph_pbc(
                kwargs["pos"], kwargs["cell"], cutoff
            )
        else:
            edge_index, shift_vectors = cls.make_graph(
                kwargs["pos"], kwargs["cell"], cutoff, kwargs["pbc"]
            )
        kwargs["edge_index"] = edge_index
        kwargs["shift_vectors"] = shift_vectors

        kwargs["n_atoms"] = torch.tensor([len(atoms)]).reshape(1, 1)

        if confinement is not None:
            kwargs["confinement"] = torch.tensor(
                list(confinement), dtype=dtype
            ).reshape(1, 2)

        return cls(**kwargs)

    @classmethod
    def empty(cls, cutoff: int = 6.0) -> "AtomsGraph":
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

    def add_batch_attr(self, key: str, value: torch.Tensor, type: str = "node") -> None:
        """Add a batch attribute to the graph.

        Parameters
        ----------
        key: str
            The key of the attribute.
        value: torch.Tensor
            The value of the attribute.
        type: str
            The type of the attribute. Can be either "node" or "graph"

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
        numbers = self.x.detach().cpu().numpy()
        positions = self.pos.detach().cpu().numpy()
        atoms = Atoms(
            numbers=numbers,
            positions=positions,
            cell=self.cell.detach().cpu().numpy(),
            pbc=self.pbc.detach().cpu().numpy(),
        )

        if "energy_prediction" in self._store:
            energy = self.energy_prediction.item()
            atoms.calc = SinglePointCalculator(atoms, energy=energy)
            atoms.calc.name = "AGeDi"

        if "forces_prediction" in self._store:
            forces = self.forces_prediction.detach().cpu().numpy()
            if hasattr(atoms, "calc") and atoms.calc is not None:
                atoms.calc.results["forces"] = forces
            else:
                atoms.calc = SinglePointCalculator(atoms, forces=forces)
                atoms.calc.name = "AGeDi"
            
        return atoms

    @staticmethod
    def make_graph(
        positions: torch.Tensor,
        cell: torch.Tensor,
        cutoff: int,
        pbc: torch.Tensor,
        dtype: torch.dtype = None,
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

    @staticmethod
    def make_graph_pbc(
        positions: torch.Tensor,
        cell: torch.Tensor,
        cutoff: float,
        max_num_neighbors: int = 20,
        dtype: torch.dtype = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create graph edges for a periodic structure, robust to small/odd cells.

        Unlike :meth:`make_graph` (which delegates to matscipy's
        ``neighbour_list``), this method explicitly computes the number of
        periodic-image repetitions required in each direction from the cell
        geometry and the cutoff radius.  This makes it work correctly for
        small or highly non-orthorhombic unit cells where neighbour-list
        libraries may behave unexpectedly or run indefinitely.

        A hard cap on the number of neighbours per atom (``max_num_neighbors``)
        is applied, retaining only the closest neighbours.  This bounds the
        output size and ensures the method terminates for any input cell.

        Parameters
        ----------
        positions : torch.Tensor
            Cartesian atom positions, shape ``(n_atoms, 3)``.
        cell : torch.Tensor
            Cell matrix of shape ``(3, 3)`` where **row** ``i`` is lattice
            vector :math:`\\mathbf{a}_i`.
        cutoff : float
            Edge cutoff radius in Ångström.
        max_num_neighbors : int
            Maximum number of neighbours kept per atom (closest first).
            Defaults to 20.
        dtype : torch.dtype, optional
            Output tensor dtype.  Defaults to ``positions.dtype``.

        Returns
        -------
        edge_index : torch.Tensor
            Shape ``(2, n_edges)``.  ``edge_index[0]`` contains the indices of
            center atoms and ``edge_index[1]`` contains neighbour indices,
            matching the convention of :meth:`make_graph`.
        shift_vectors : torch.Tensor
            Cartesian shift vectors of shape ``(n_edges, 3)``.  Adding
            ``shift_vectors[k]`` to ``positions[edge_index[1, k]]`` gives the
            Cartesian position of the neighbour in the periodic image.
        """
        if dtype is None:
            dtype = positions.dtype

        max_num_neighbors = int(max_num_neighbors)
        n_atoms = positions.shape[0]
        device = positions.device
        cell = cell.to(dtype)
        positions = positions.to(dtype)

        # Compute interplanar distances to determine the number of unit-cell
        # repetitions needed in each direction to fully cover the cutoff sphere.
        # The minimum distance between planes whose normal is a_j x a_k equals
        # the cell volume divided by the area of that face.
        cross_a2a3 = torch.linalg.cross(cell[1], cell[2])
        cross_a3a1 = torch.linalg.cross(cell[2], cell[0])
        cross_a1a2 = torch.linalg.cross(cell[0], cell[1])
        cell_vol = torch.abs(torch.dot(cell[0], cross_a2a3))

        min_dist_a1 = cell_vol / torch.norm(cross_a2a3)
        min_dist_a2 = cell_vol / torch.norm(cross_a3a1)
        min_dist_a3 = cell_vol / torch.norm(cross_a1a2)

        rep_a1 = int(torch.ceil(cutoff / min_dist_a1).item())
        rep_a2 = int(torch.ceil(cutoff / min_dist_a2).item())
        rep_a3 = int(torch.ceil(cutoff / min_dist_a3).item())

        # Generate all integer cell-offset triples within the required range.
        # Use torch.long for exact integer arithmetic; convert to dtype only
        # when computing Cartesian shifts via matrix multiplication.
        r1 = torch.arange(-rep_a1, rep_a1 + 1, device=device, dtype=torch.long)
        r2 = torch.arange(-rep_a2, rep_a2 + 1, device=device, dtype=torch.long)
        r3 = torch.arange(-rep_a3, rep_a3 + 1, device=device, dtype=torch.long)
        grid = torch.meshgrid(r1, r2, r3, indexing="ij")
        int_offsets = torch.stack(
            [g.reshape(-1) for g in grid], dim=-1
        )  # (n_offsets, 3) integer offsets
        cell_shifts = int_offsets.to(dtype) @ cell  # Cartesian shift vectors (n_offsets, 3)
        n_offsets = int_offsets.shape[0]

        # All (center, neighbour) atom-index pairs, including self-pairs which
        # are removed afterwards via the distance mask.
        idx_i = torch.arange(n_atoms, device=device).repeat_interleave(n_atoms)
        idx_j = torch.arange(n_atoms, device=device).repeat(n_atoms)
        n_pairs = n_atoms * n_atoms

        # Expand each pair over every cell offset.
        idx_i_exp = idx_i.repeat_interleave(n_offsets)  # (n_pairs * n_offsets,)
        idx_j_exp = idx_j.repeat_interleave(n_offsets)
        shifts_exp = cell_shifts.repeat(n_pairs, 1)  # (n_pairs * n_offsets, 3)

        # Displacement from center atom i to image of atom j.
        disp = positions[idx_j_exp] + shifts_exp - positions[idx_i_exp]
        dist_sq = (disp * disp).sum(dim=-1)

        # Keep only pairs within the cutoff; remove strict self-interactions
        # (same atom, zero shift → dist² ≈ 0).
        mask = (dist_sq <= cutoff * cutoff) & (dist_sq > 1e-8)
        idx_i_exp = idx_i_exp[mask]
        idx_j_exp = idx_j_exp[mask]
        dist_sq = dist_sq[mask]
        shifts_exp = shifts_exp[mask]

        # Sort by distance, then stable-sort by center-atom index so that
        # within each group edges are ordered nearest-first.
        perm = torch.argsort(dist_sq)
        idx_i_exp = idx_i_exp[perm]
        idx_j_exp = idx_j_exp[perm]
        dist_sq = dist_sq[perm]
        shifts_exp = shifts_exp[perm]

        perm2 = torch.argsort(idx_i_exp, stable=True)
        idx_i_exp = idx_i_exp[perm2]
        idx_j_exp = idx_j_exp[perm2]
        shifts_exp = shifts_exp[perm2]

        # Compute within-group rank and apply the max_num_neighbors cap.
        counts = torch.bincount(idx_i_exp, minlength=n_atoms)
        group_start = torch.zeros(n_atoms + 1, dtype=torch.long, device=device)
        group_start[1:] = counts.cumsum(0)
        within_rank = (
            torch.arange(idx_i_exp.shape[0], device=device)
            - group_start[idx_i_exp]
        )
        keep = within_rank < max_num_neighbors

        edge_index = torch.stack([idx_i_exp[keep], idx_j_exp[keep]], dim=0)
        shift_vectors = shifts_exp[keep].to(dtype)

        return edge_index, shift_vectors

    @batched(update_keys=["edge_index", "shift_vectors"])
    def update_graph(self) -> None:
        """Update the graph with new edges

        This should be called after changing any of the positions or cell.

        Returns
        -------
        None

        """

        cutoff = (
            self.cutoff.item() if isinstance(self.cutoff, torch.Tensor) else self.cutoff
        )

        device = self.pos.device
        use_pbc = getattr(self, "use_pbc_graph", True)
        pbc = self.pbc.detach().cpu()
        if use_pbc and pbc.all():
            edge_index, shift_vectors = self.make_graph_pbc(
                self.pos.detach().cpu(),
                self.cell.detach().cpu(),
                cutoff,
            )
        else:
            edge_index, shift_vectors = self.make_graph(
                self.pos.detach().cpu(),
                self.cell.detach().cpu(),
                cutoff,
                pbc,
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

    def __len__(self) -> int:
        """Return the number of atoms in the graph.

        Returns
        -------
        n_atoms: int
            The number of atoms in the graph.

        """
        return self.pos.shape[0]

    @property
    def cell(self) -> torch.Tensor:
        """Return the canonical cell matrix of the graph.

        Returns
        -------
        cell: torch.Tensor
            The cell matrix of shape ``(3, 3)``.
        """
        return self._store["cell"]

    @cell.setter
    def cell(self, cell: torch.Tensor) -> None:
        """Set the cell matrix, preserving fractional coordinates.

        Cartesian positions are recomputed when both ``pos`` and an old
        ``cell`` are already present so that fractional coordinates remain
        unchanged.  No canonicalization is performed; use
        :meth:`from_atoms` with ``canonical_cell=True`` if you need the
        cell stored in canonical lower-triangular form.

        Parameters
        ----------
        cell: torch.Tensor
            The new cell matrix.

        Returns
        -------
        None
        """
        # Only invalidate the edge cache when the cell is being *changed*
        # (i.e. a cell already existed) on an individual graph.  Skip during
        # Batch construction (where edge_index is already correct) and skip
        # during initial construction (where no prior cell exists yet).
        cell_was_set = "cell" in self._store
        # Preserve fractional coordinates when both pos and old cell exist.
        if "pos" in self._store and cell_was_set:
            frac = self.pos_to_frac(self.pos)
            self._store["cell"] = cell
            pos = self.frac_to_pos(frac)
            Data.pos.fset(self, pos)
            if "frac" in self._store:
                del self._store["frac"]
        else:
            self._store["cell"] = cell

        if cell_was_set and not isinstance(self, Batch) and (
            "edge_index" in self._store or "shift_vectors" in self._store
        ):
            self.clear_graph()

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
        if "frac" in self._store:
            del self["frac"]
        if "mask" in self._store:
            pos[self.positions_mask] = self.pos[self.positions_mask]
        Data.pos.fset(self, pos)

        # if "cell" in self._store:
        #     f = self.pos_to_frac(self.pos)
        #     self.add_batch_attr("frac", f, type="node")

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
        frac %= 1.0
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
        return f

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
        self.add_batch_attr(
            "repr_slices", slices.repeat(self.n_atoms.shape[0], 1), type="graph"
        )
        self.add_batch_attr(
            "repr_ls", ls.repeat(self.n_atoms.shape[0], 1), type="graph"
        )

    def wrap_positions(self) -> None:
        """Wrap the positions of the atoms to the unit cell.

        Returns
        -------
        None

        """
        pbc = torch.repeat_interleave(self.pbc.view(-1, 3), self.n_atoms.view(-1), dim=0)
        f = self.frac
        f[pbc] = f[pbc] % 1
        self.pos = self.frac_to_pos(f)


    def apply_mask(self, x: torch.Tensor, val: float = 0.0) -> torch.Tensor:
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

    @property
    def confinement(self) -> torch.Tensor:
        """Return the confinement of the graph.

        Returns
        -------
        confinement: torch.Tensor
            The confinement of the graph.

        """
        return self["confinement"] if "confinement" in self._store else None

    @confinement.setter
    def confinement(self, confinement: torch.Tensor) -> None:
        """Set the confinement bounds for the graph.

        Parameters
        ----------
        confinement : torch.Tensor
            Tensor of shape ``(1, 2)`` containing the lower and upper
            Z-confinement bounds.
        """
        self.add_batch_attr("confinement", confinement, type="graph")


    @property
    def cellpar(self) -> torch.Tensor:
        """Return the cell parameters of the graph."""
        return self.cell_to_vectors(self.cell)

    @cellpar.setter
    def cellpar(self, cellpar: torch.Tensor) -> None:
        """Set the cell parameters of the graph.

        Parameters
        ----------
        cellpar: torch.Tensor
            The cell parameters of the graph.

        Returns
        -------
        None

        """
        cell = self.vector_to_cell(cellpar)
        if cell.ndim == 3 and cell.shape[0] == 1:
            cell = cell.reshape(3, 3)
        self.cell = cell
        
    @staticmethod
    def _is_lower_triangular(cell: torch.Tensor) -> bool:
        """Return True if *cell* is in canonical lower-triangular form.

        A cell matrix is considered canonical when the three strictly
        upper-triangular entries (cell[0,1], cell[0,2], cell[1,2]) are
        all zero (within a tight floating-point tolerance of 1e-10).

        Parameters
        ----------
        cell : torch.Tensor
            The cell matrix.

        Returns
        -------
        bool
            True if the cell is already lower-triangular.
        """
        c = cell.reshape(3, 3)
        return bool(
            c[0, 1].abs() < 1e-10
            and c[0, 2].abs() < 1e-10
            and c[1, 2].abs() < 1e-10
        )

    @staticmethod
    def cell_to_vectors(cell: torch.Tensor) -> torch.Tensor:
        """Convert cell matrix to cell parameters.

        Parameters
        ----------
        cell : torch.Tensor
            The cell matrix of shape ``(N, 3)`` or ``(N, 3, 3)``.

        Returns
        -------
        torch.Tensor
            The cell parameters of shape ``(N, 6)``.

        """
        cell = cell.view(-1, 3, 3)
        
        a = torch.norm(cell[..., 0, :], dim=-1)
        b = torch.norm(cell[..., 1, :], dim=-1)
        c = torch.norm(cell[..., 2, :], dim=-1)

        alpha = torch.acos(
            torch.sum(cell[..., 1, :] * cell[..., 2, :], dim=-1) / (b * c)
        )
        beta = torch.acos(
            torch.sum(cell[..., 0, :] * cell[..., 2, :], dim=-1) / (a * c)
        )
        gamma = torch.acos(
            torch.sum(cell[..., 0, :] * cell[..., 1, :], dim=-1) / (a * b)
        )

        # alpha = alpha * 180 / torch.pi
        # beta = beta * 180 / torch.pi
        # gamma = gamma * 180 / torch.pi

        a,b,c = torch.log(a), torch.log(b), torch.log(c)
        alpha, beta, gamma = alpha - torch.pi / 2, beta - torch.pi / 2, gamma - torch.pi / 2

        return torch.stack([a, b, c, alpha, beta, gamma], dim=-1)

    @staticmethod
    def vector_to_cell(cellpar: torch.Tensor) -> torch.Tensor:
        """Convert cell parameters to cell matrix.

        Parameters
        ----------
        cellpar : torch.Tensor
            The cell parameters of shape ``(N, 6)``.

        Returns
        -------
        torch.Tensor
            The cell matrix of shape ``(N, 3, 3)`` where each row is a lattice vector.

        """
        a, b, c, alpha, beta, gamma = cellpar.unbind(-1)

        a, b, c = torch.exp(a), torch.exp(b), torch.exp(c)
        alpha, beta, gamma = alpha + torch.pi / 2, beta + torch.pi / 2, gamma + torch.pi / 2
        

        cos_alpha = torch.cos(alpha)
        cos_beta = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)

        cell = torch.zeros(cellpar.shape[:-1] + (3, 3), device=cellpar.device, dtype=cellpar.dtype)
        cell[..., 0, 0] = a
        cell[..., 1, 0] = b * cos_gamma
        cell[..., 2, 0] = c * cos_beta

        cell[..., 1, 1] = b * sin_gamma
        cell[..., 2, 1] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma

        cell[..., 2, 2] = c * torch.sqrt(torch.clamp(1 - cos_beta ** 2 - cell[..., 2, 1] ** 2 / c ** 2, min=0))

        return cell

