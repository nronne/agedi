import dataclasses
import functools
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from matscipy.neighbours import neighbour_list as matscipy_neighbour_list
from torch_geometric.data import Batch, Data

try:
    from nvalchemiops.torch.neighbors import neighbor_list as nvidia_neighbor_list
    NVIDIA_NEIGHBOR_IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError, TypeError) as exc:
    nvidia_neighbor_list = None
    # Stored so callers can inspect why NVIDIA ops are unavailable when debugging.
    NVIDIA_NEIGHBOR_IMPORT_ERROR = exc

try:
    from nvalchemiops.torch.neighbors.batch_cell_list import (
        batch_build_cell_list,
        batch_query_cell_list,
        estimate_batch_cell_list_sizes,
    )
    from nvalchemiops.torch.neighbors.neighbor_utils import (
        allocate_cell_list,
        estimate_max_neighbors,
    )
    NVIDIA_CELL_LIST_IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError, TypeError) as exc:
    batch_build_cell_list = None
    batch_query_cell_list = None
    estimate_batch_cell_list_sizes = None
    allocate_cell_list = None
    estimate_max_neighbors = None
    NVIDIA_CELL_LIST_IMPORT_ERROR = exc


NEIGHBOR_CACHE_KEYS = (
    "edge_index",
    "shift_vectors",
)


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


@dataclasses.dataclass
class Representation:
    """Representation class

    A simple container holding the scalar (l=0) and vector (l=1) equivariant
    representations produced by the backbone network.  Both fields are optional
    so that the class can also be used for partial representations.

    Registered as a ``torch.utils._pytree`` node so that ``torch.compile``
    can traverse instances transparently without introducing graph breaks.

    Parameters
    ----------
    scalar: Optional[torch.Tensor]
        Per-node scalar features of shape ``(n_nodes, n_features, 1)``.
        Default is ``None``.
    vector: Optional[torch.Tensor]
        Per-node vector features of shape ``(n_nodes, n_features, 3)``.
        Default is ``None``.
    """

    scalar: Optional[torch.Tensor] = None
    vector: Optional[torch.Tensor] = None

    def to_tensor(self, n_graphs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Serialise scalar and vector tensors into a single flat representation.

        Concatenates ``scalar`` and ``vector`` (when present) along the feature
        dimension.  Returns the concatenated tensor together with per-graph
        slice boundaries and degree values so that
        :meth:`from_tensor` can reconstruct the original fields.

        Parameters
        ----------
        n_graphs: int
            The number of graphs in the batch.  The slice and degree tensors
            are repeated once per graph so they can be stored as graph-level
            attributes.

        Returns
        -------
        tensor: torch.Tensor
            Concatenated representation of shape ``(n_nodes, total_features)``.
        slices: torch.Tensor
            Cumulative slice boundaries of shape ``(n_graphs, n_parts + 1)``.
        ls: torch.Tensor
            Degree values of shape ``(n_graphs, n_parts)``.
        """
        tensors_ordered = []
        if self.scalar is not None:
            tensors_ordered.append((0, self.scalar))
        if self.vector is not None:
            tensors_ordered.append((1, self.vector))

        nodes = tensors_ordered[0][1].shape[0]
        flat = []
        slices = [0]
        ls = []
        for degree, value in tensors_ordered:
            ls.append(degree)
            flat.append(value.reshape(nodes, -1))
            slices.append(flat[-1].shape[1])

        slices = torch.cumsum(torch.tensor(slices, dtype=int), dim=0).repeat(
            n_graphs, 1
        )
        tensor = torch.cat(flat, dim=1)
        ls = torch.tensor(ls, dtype=int).repeat(n_graphs, 1)

        return tensor, slices, ls

    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor, slices: torch.Tensor, ls: torch.Tensor
    ) -> "Representation":
        """Reconstruct a :class:`Representation` from a flat serialised form.

        Parameters
        ----------
        tensor: torch.Tensor
            Flat representation of shape ``(n_nodes, total_features)``.
        slices: torch.Tensor
            Cumulative slice boundaries of shape ``(n_graphs, n_parts + 1)``.
        ls: torch.Tensor
            Degree values of shape ``(n_graphs, n_parts)``.

        Returns
        -------
        Representation
        """
        n_nodes = tensor.shape[0]
        slices = slices[0]
        degrees = ls[0]

        scalar = None
        vector = None
        for i, degree in enumerate(degrees):
            t = tensor[:, slices[i].item() : slices[i + 1].item()].reshape(
                n_nodes, -1, 2 * degree.item() + 1
            )
            if degree.item() == 0:
                scalar = t
            elif degree.item() == 1:
                vector = t

        return cls(scalar=scalar, vector=vector)


# Register Representation as a transparent pytree node so that torch.compile
# can traverse instances without introducing graph breaks.
torch.utils._pytree.register_pytree_node(
    Representation,
    lambda rep: ([rep.scalar, rep.vector], None),
    lambda fields, _ctx: Representation(scalar=fields[0], vector=fields[1]),
)


class AtomsGraph(Data):
    """Atomistic Graph Class

    Class defining a graph with atoms as nodes and edges formed between all
    atoms within a finite cutoff radius.

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
        cutoff: float = 6.0,
        dtype: torch.dtype = torch.float,
        initialize_mask: Optional[bool] = None,
        confinement: Optional[Tuple[float, float]] = None,
        canonical_cell: bool = False,
        fully_connected: bool = False,
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
            When ``True``, the cell is stored in canonical lower-triangular
            form.  If the input cell is not already canonical, Cartesian
            positions are recomputed to preserve fractional coordinates and a
            warning is raised.  Set to ``False`` (the default) to store the
            cell exactly as provided by ASE (no rotation or recomputation is
            performed).

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
            warnings.warn(
                "AtomsGraph.from_atoms: cell is not in canonical lower-triangular "
                "form; canonicalizing. Cartesian positions will be recomputed to "
                "preserve fractional coordinates.",
                UserWarning,
                stacklevel=2,
            )
            final_cell_f64 = cls.vector_to_cell(cls.cell_to_vectors(cell_f64)).view(3, 3)
            pos_f64 = torch.tensor(pos_np, dtype=torch.float64)
            frac_f64 = torch.linalg.solve(cell_f64.T, pos_f64.T).T
            final_pos = (frac_f64 @ final_cell_f64).to(dtype)

        kwargs["pos"] = final_pos
        kwargs["cell"] = final_cell_f64.to(dtype)
        kwargs["pbc"] = torch.tensor(atoms.get_pbc())

        if fully_connected:
            edge_index, shift_vectors = cls.make_fully_connected_graph(
                kwargs["pos"], dtype=dtype
            )
            kwargs["fully_connected"] = torch.tensor([1])
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
    def empty(cls, cutoff: float = 6.0, fully_connected: bool = False) -> "AtomsGraph":
        """Create an empty graph.

        Parameters
        ----------
        cutoff: float
            The cutoff radius for the edges.
        fully_connected : bool, optional
            When ``True`` the graph will be rebuilt as a fully connected graph
            (all atom pairs, no self-loops, zero shift vectors) every time
            :meth:`update_graph` is called.  Suitable for gas-phase molecules
            and clusters where a finite cutoff misses pairs when atoms spread
            during the reverse diffusion process.  Defaults to ``False``.

        Returns
        -------
        graph: AtomsGraph
            The graph object.

        """
        kwargs = dict(
            x=torch.empty(0, dtype=torch.long),
            pos=torch.empty(0, 3),
            n_atoms=torch.tensor([0]),
            cell=torch.empty(3, 3),
            pbc=torch.tensor([True, True, True], dtype=torch.bool),
            cutoff=cutoff,
        )
        if fully_connected:
            kwargs["fully_connected"] = torch.tensor([1])
        return cls(**kwargs)

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

    def _get_scalar_attr(self, key: str) -> Optional[float]:
        value = self._store.get(key)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            flat = value.reshape(-1)
            if flat.numel() == 0:
                return None
            if isinstance(self, Batch):
                unique = flat.unique()
                if unique.numel() > 1:
                    raise ValueError(
                        f"All graphs in the batch must have the same {key}, "
                        f"but found: {unique.tolist()}."
                    )
                return float(unique[0].item())
            return float(flat[0].item())
        return float(value)

    def prepare_for_compile(self, cutoff: float) -> None:
        """Pre-allocate neighbor-list buffers for ``torch.compile`` compatibility.

        Estimates the maximum number of neighbors per atom using
        :func:`~nvalchemiops.torch.neighbors.neighbor_utils.estimate_max_neighbors`
        and the cell-list dimensions using
        :func:`~nvalchemiops.torch.neighbors.cell_list.estimate_cell_list_sizes`,
        then allocates the cell list and all output buffers with fixed shapes.
        Fixed shapes are required for ``torch.compile`` to trace the reverse
        diffusion step once without retracing on subsequent iterations.

        Must be called on a :class:`~torch_geometric.data.Batch` **before**
        the first :meth:`update_graph` call.

        Requires the ``nvalchemiops`` package.

        Parameters
        ----------
        cutoff : float
            Neighbor-list cutoff radius (Å).

        Raises
        ------
        RuntimeError
            When ``nvalchemiops`` is not installed.
        TypeError
            When called on an unbatched :class:`AtomsGraph` instead of a
            :class:`~torch_geometric.data.Batch`.
        """
        if batch_build_cell_list is None or estimate_batch_cell_list_sizes is None or allocate_cell_list is None or estimate_max_neighbors is None:
            raise RuntimeError(
                "NVIDIA nvalchemiops is required for torch.compile support. "
                f"Import error was: {NVIDIA_NEIGHBOR_IMPORT_ERROR or NVIDIA_CELL_LIST_IMPORT_ERROR}"
            )
        if not isinstance(self, Batch):
            raise TypeError(
                "prepare_for_compile must be called on a batched graph (Batch), "
                "not on an individual AtomsGraph."
            )

        num_atoms = self.pos.shape[0]
        device = self.pos.device
        batch_idx = self.batch.to(torch.int32)
        cell = self.cell.view(-1, 3, 3).contiguous()
        pbc = self.pbc.view(-1, 3).contiguous()

        # Estimate cell-list geometry parameters so all buffer shapes are fixed.
        max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell, pbc, cutoff
        )

        # Allocate the cell list cache (pre-allocated tensors passed to
        # build_cell_list / query_cell_list on every step).
        cell_list_cache = allocate_cell_list(
            total_atoms=num_atoms,
            max_total_cells=max_total_cells,
            neighbor_search_radius=neighbor_search_radius,
            device=device,
        )

        # Estimate maximum neighbors per atom to pre-allocate output buffers.
        max_n = estimate_max_neighbors(cutoff)

        neighbor_matrix = torch.full(
            (num_atoms, max_n), -1, dtype=torch.int32, device=device
        )
        neighbor_shifts = torch.zeros(
            (num_atoms, max_n, 3), dtype=torch.int32, device=device
        )
        num_neighbors_arr = torch.zeros(num_atoms, dtype=torch.int32, device=device)

        self._store["cell_list_cache"] = cell_list_cache
        self._store["neighbor_matrix"] = neighbor_matrix
        self._store["neighbor_shifts"] = neighbor_shifts
        self._store["num_neighbors_arr"] = num_neighbors_arr

        # Also expose the same objects as direct instance attributes so that
        # the compiled path in update_graph() can access them without any
        # Python dict lookup or .item() call (both of which break
        # torch.compile's FX graph).
        self._cutoff_scalar: float = cutoff
        self._cell_list_cache_buffers = cell_list_cache
        self._neighbor_matrix_buf: torch.Tensor = neighbor_matrix
        self._neighbor_shifts_buf: torch.Tensor = neighbor_shifts
        self._num_neighbors_arr_buf: torch.Tensor = num_neighbors_arr
        self._compile_ready: bool = True


    @staticmethod
    def _cell_list_to_graph(
        neighbor_matrix: torch.Tensor,
        neighbor_shifts: torch.Tensor,
        cell: torch.Tensor,
        dtype: torch.dtype,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert cell-list query output to ``(edge_index, shift_vectors)``."""
        mask = neighbor_matrix != -1
        src = torch.where(mask)[0].to(torch.long)
        tgt = neighbor_matrix[mask].to(torch.long)
        edge_index = torch.stack([src, tgt], dim=0)

        unit_shifts = neighbor_shifts[mask].to(cell.dtype)
        if batch_idx is None:
            shift_vectors = torch.einsum("ij,jk->ik", unit_shifts, cell.view(3, 3))
        else:
            edge_cells = torch.index_select(
                cell.view(-1, 3, 3), 0, batch_idx.to(torch.long)[src]
            )
            shift_vectors = torch.einsum("ni,nij->nj", unit_shifts, edge_cells)
        return edge_index, shift_vectors.to(dtype)

    # @batched(update_keys=["edge_index", "shift_vectors"])
    def update_graph(self) -> bool:
        """Update the graph with new edges

        This should be called after changing any of the positions or cell.

        Returns
        -------
        rebuilt: bool
            ``True`` when the neighbor list was fully recomputed.
        """

        if getattr(self, '_compile_ready', False):
            # Compiled path: read cutoff and neighbor-list buffers from direct
            # instance attributes set by prepare_for_compile().  This avoids
            # the _get_scalar_attr() .item() call and all Python dict
            # membership tests, both of which break torch.compile's FX graph.
            batch_idx = self.batch.to(torch.int32)
            cell = self.cell.view(-1, 3, 3).contiguous()
            pbc = self.pbc.view(-1, 3).contiguous()

            batch_build_cell_list(
                self.pos,
                self._cutoff_scalar,
                cell,
                pbc,
                batch_idx,
                *self._cell_list_cache_buffers,
            )

            self._neighbor_matrix_buf.fill_(-1)
            self._neighbor_shifts_buf.fill_(0)
            self._num_neighbors_arr_buf.fill_(0)

            batch_query_cell_list(
                self.pos,
                cell,
                pbc,
                self._cutoff_scalar,
                batch_idx,
                *self._cell_list_cache_buffers,
                self._neighbor_matrix_buf,
                self._neighbor_shifts_buf,
                self._num_neighbors_arr_buf,
            )

            self.edge_index, self.shift_vectors = self._cell_list_to_graph(
                neighbor_matrix=self._neighbor_matrix_buf,
                neighbor_shifts=self._neighbor_shifts_buf,
                cell=cell,
                dtype=self.pos.dtype,
                batch_idx=batch_idx,
            )
            return True

        # Fully-connected path: rebuild edges as all-pairs for non-periodic systems.
        fc = self._get_scalar_attr("fully_connected")
        if fc is not None and bool(fc):
            batch_idx = self.batch.to(torch.int32) if isinstance(self, Batch) else None
            self.edge_index, self.shift_vectors = self.make_fully_connected_graph(
                self.pos, dtype=self.pos.dtype, batch_idx=batch_idx
            )
            return True

        cutoff = self._get_scalar_attr("cutoff")
        if cutoff is None:
            raise ValueError(
                "cutoff must be set on the graph before calling update_graph()."
            )

        if isinstance(self, Batch):
            batch_idx = self.batch.to(torch.int32)
            cell = self.cell.view(-1, 3, 3).contiguous()
            pbc = self.pbc.view(-1, 3).contiguous()

            if "cell_list_cache" in self._store:
                # Compiled path: use build_cell_list + query_cell_list with
                # pre-allocated, fixed-shape buffers (required for torch.compile).
                neighbor_matrix = self._store["neighbor_matrix"]
                neighbor_shifts = self._store["neighbor_shifts"]
                num_neighbors_arr = self._store["num_neighbors_arr"]
                cell_list_cache = self._store["cell_list_cache"]

                batch_build_cell_list(
                    self.pos, cutoff, cell, pbc, batch_idx, *cell_list_cache
                )

                neighbor_matrix.fill_(-1)
                neighbor_shifts.fill_(0)
                num_neighbors_arr.fill_(0)

                batch_query_cell_list(
                    self.pos,
                    cell,
                    pbc,
                    cutoff,
                    batch_idx,
                    *cell_list_cache,
                    neighbor_matrix,
                    neighbor_shifts,
                    num_neighbors_arr,
                )

                self.edge_index, self.shift_vectors = self._cell_list_to_graph(
                    neighbor_matrix=neighbor_matrix,
                    neighbor_shifts=neighbor_shifts,
                    cell=cell,
                    dtype=self.pos.dtype,
                    batch_idx=batch_idx,
                )
            else:
                self.edge_index, self.shift_vectors = self.make_graph(
                    self.pos, cell, cutoff, pbc, batch_idx=batch_idx
                )
        else:
            self.edge_index, self.shift_vectors = self.make_graph(
                self.pos,
                self.cell,
                cutoff,
                self.pbc,
            )
        return True

    @staticmethod
    def _make_graph_matscipy(
        positions: torch.Tensor,
        cell: torch.Tensor,
        cutoff: float,
        pbc: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_dtype = dtype or positions.dtype
        if batch_idx is None:
            i, j, shifts = matscipy_neighbour_list(
                "ijS",
                positions=positions.detach().cpu().numpy(),
                cell=cell.detach().cpu().numpy(),
                cutoff=cutoff,
                pbc=pbc.detach().cpu().numpy(),
            )
            edge_index = torch.tensor(
                np.stack([i, j]), dtype=torch.long, device=positions.device
            )
            unit_shifts = torch.tensor(
                shifts, dtype=cell.dtype, device=positions.device
            )
            shift_vectors = torch.einsum("ij,jk->ik", unit_shifts, cell.view(3, 3))
            return edge_index, shift_vectors.to(output_dtype)

        edge_index_parts = []
        shift_vector_parts = []
        batch_cells = cell.view(-1, 3, 3)
        batch_pbc = pbc.view(-1, 3)
        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
        for graph_idx in range(num_graphs):
            atom_idx = torch.where(batch_idx == graph_idx)[0]
            if atom_idx.numel() == 0:
                continue
            local_edge_index, local_shift_vectors = AtomsGraph._make_graph_matscipy(
                positions[atom_idx],
                batch_cells[graph_idx],
                cutoff,
                batch_pbc[graph_idx],
                dtype=output_dtype,
            )
            edge_index_parts.append(atom_idx[local_edge_index])
            shift_vector_parts.append(local_shift_vectors)

        if not edge_index_parts:
            return (
                torch.empty((2, 0), dtype=torch.long, device=positions.device),
                torch.empty((0, 3), dtype=output_dtype, device=positions.device),
            )

        return torch.cat(edge_index_parts, dim=1), torch.cat(shift_vector_parts, dim=0)

    @staticmethod
    def make_fully_connected_graph(
        positions: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a fully connected graph: every atom is connected to every other.

        No self-loops are included.  All shift vectors are zero (non-periodic).
        This is the correct topology for gas-phase molecules and clusters where
        the finite cutoff of a standard neighbour list would miss long-range
        pairs when atoms spread apart during the reverse diffusion process.

        Parameters
        ----------
        positions : torch.Tensor, shape (n_atoms, 3)
        dtype : torch.dtype, optional
            Data type of the shift-vector output.  Defaults to ``positions.dtype``.
        batch_idx : torch.Tensor of shape (n_atoms,), optional
            Graph-membership index for batched graphs.  When ``None`` all atoms
            are treated as belonging to a single graph.

        Returns
        -------
        edge_index : torch.Tensor, shape (2, n_edges)
        shift_vectors : torch.Tensor, shape (n_edges, 3)
            All zeros (no periodic images).
        """
        device = positions.device
        output_dtype = dtype or positions.dtype
        n_atoms = positions.shape[0]

        if batch_idx is None:
            idx = torch.arange(n_atoms, device=device)
            src, dst = torch.meshgrid(idx, idx, indexing="ij")
            mask = src != dst
            edge_index = torch.stack([src[mask], dst[mask]], dim=0)
        else:
            n_graphs = int(batch_idx.max().item()) + 1 if n_atoms > 0 else 0
            parts = []
            for g in range(n_graphs):
                atom_idx = torch.where(batch_idx == g)[0]
                n = atom_idx.shape[0]
                if n <= 1:
                    continue
                local = torch.arange(n, device=device)
                ls, ld = torch.meshgrid(local, local, indexing="ij")
                m = ls != ld
                parts.append(torch.stack([atom_idx[ls[m]], atom_idx[ld[m]]]))
            edge_index = (
                torch.cat(parts, dim=1)
                if parts
                else torch.empty((2, 0), dtype=torch.long, device=device)
            )

        shift_vectors = torch.zeros(
            edge_index.shape[1], 3, dtype=output_dtype, device=device
        )
        return edge_index, shift_vectors

    @staticmethod
    def make_graph(
        positions: torch.Tensor,
        cell: torch.Tensor,
        cutoff: float,
        pbc: torch.Tensor,
        dtype: torch.dtype = None,
        batch_idx: Optional[torch.Tensor] = None,
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

        with torch.no_grad():
            if nvidia_neighbor_list is None:
                return AtomsGraph._make_graph_matscipy(
                    positions,
                    cell,
                    cutoff,
                    pbc,
                    dtype=dtype,
                    batch_idx=batch_idx,
                )

            kwargs = {
                "positions": positions,
                "cutoff": cutoff,
                "cell": cell.view(-1, 3, 3) if batch_idx is not None else cell,
                "pbc": pbc,
                "return_neighbor_list": True,
            }
            if batch_idx is not None:
                kwargs["batch_idx"] = batch_idx

            edge_index, _, shifts = nvidia_neighbor_list(**kwargs)
            edge_index = edge_index.to(torch.long)
            if batch_idx is None:
                shift_vectors = torch.einsum(
                    "ij,jk->ik", shifts.to(cell.dtype), cell.view(3, 3)
                )
            else:
                batch_cells = cell.view(-1, 3, 3)
                edge_cells = torch.index_select(batch_cells, 0, batch_idx[edge_index[0]])
                shift_vectors = torch.einsum(
                    "ni,nij->nj", shifts.to(edge_cells.dtype), edge_cells
                )

        return edge_index, shift_vectors

    def clear_graph(self) -> None:
        """Clear the graph removing all edges

        Returns
        -------
        None
        """
        for key in NEIGHBOR_CACHE_KEYS:
            if key in self._store:
                del self._store[key]

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
    def representation(self) -> Optional[Representation]:
        """Return the representation of the graph.

        Returns
        -------
        representation: Optional[Representation]
            The representation of the graph, or ``None`` if not set.
        """
        if "repr_scalar" in self._store:
            vector = self._store.get("repr_vector", None)
            return Representation(scalar=self.repr_scalar, vector=vector)
        # Legacy format stored by earlier versions of this code.
        if "repr" in self._store:
            return Representation.from_tensor(self.repr, self.repr_slices, self.repr_ls)
        return None

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
        self.add_batch_attr("repr_scalar", representation.scalar, type="node")
        if representation.vector is not None:
            self.add_batch_attr("repr_vector", representation.vector, type="node")

    def wrap_positions(self) -> None:
        """Wrap the positions of the atoms to the unit cell.

        Returns
        -------
        None

        """
        pbc = torch.repeat_interleave(self.pbc.view(-1, 3), self.n_atoms.view(-1), dim=0)
        f = self.pos_to_frac(self.pos)
        f = torch.where(pbc, f % 1, f)
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
