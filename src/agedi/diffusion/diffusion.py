from typing import Dict, List, Optional, Union, Tuple
from tqdm import tqdm

import numpy as np
from lightning import LightningModule
import torch
from torch_geometric.data import Batch
from torch_geometric.data.collate import collate

from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser
from agedi.models import ScoreModel

from collections import deque

class LBFGSStepSizer:
    """
    L-BFGS approach for determining optimal step sizes in force field guidance.
    """
    def __init__(self, memory_size: int = 10, initial_step: float = 0.1, device: str = 'cuda') -> None:
        """
        Initialize the L-BFGS step sizer.
        
        Parameters
        ----------
        memory_size : int, optional
            Number of previous iterations to store.
        initial_step : float, optional
            Initial step size scaling factor.
        device : str, optional
            Computation device (e.g. ``"cuda"`` or ``"cpu"``).
        """
        self.memory_size = memory_size
        self.initial_step = initial_step
        self.device = device
        
        # Storage for position and gradient differences
        self.s_list = deque(maxlen=memory_size)  # Position differences
        self.y_list = deque(maxlen=memory_size)  # Gradient (force) differences
        self.rho_list = deque(maxlen=memory_size)  # ρᵢ = 1/(yᵢᵀsᵢ)
        
        self.prev_pos = None
        self.prev_forces = None
        self.H0_scaling = 1.0  # Initial Hessian approximation scaling
    
    def compute_step(self, pos: torch.Tensor, forces: torch.Tensor) -> torch.Tensor:
        """
        Compute the optimal step using L-BFGS approximation.
        
        Parameters
        ----------
        pos : torch.Tensor
            Current atomic positions (B×N×3 tensor).
        forces : torch.Tensor
            Current forces (B×N×3 tensor).
            
        Returns
        -------
        torch.Tensor
            Optimal step vector (B×N×3 tensor).
        """
        if self.prev_pos is None or self.prev_forces is None:
            self.prev_pos = pos.clone().detach()
            self.prev_forces = forces.clone().detach()
            
            # First iteration, use simple scaling
            avg_force_mag = torch.norm(forces, dim=1).mean()
            adaptive_scale = min(self.initial_step, 0.1 / max(avg_force_mag, 1e-6))
            initial_step = adaptive_scale * forces
            return initial_step
        
        # Compute position and gradient differences
        s = pos - self.prev_pos  # Position difference
        y = self.prev_forces - forces  # Force difference (negative gradient)
        
        # Store differences if they satisfy curvature condition
        sy = torch.sum(s * y)
        if sy > 1e-10:  # Ensure positive curvature
            self.s_list.append(s)
            self.y_list.append(y)
            self.rho_list.append(1.0 / sy)
            
            # Update H0 scaling using Barzilai-Borwein formula
            self.H0_scaling = sy / torch.sum(y * y)
        
        # Apply L-BFGS two-loop recursion algorithm
        q = forces.clone()  # Start with gradient
        alpha_list = []
        
        # First loop
        for i in range(len(self.s_list)-1, -1, -1):
            rho = self.rho_list[i]
            s_i = self.s_list[i]
            y_i = self.y_list[i]
            alpha_i = rho * torch.sum(s_i * q)
            alpha_list.append(alpha_i)
            q = q - alpha_i * y_i
        
        # Apply initial Hessian approximation
        r = self.H0_scaling * q
        
        # Second loop
        for i in range(len(self.s_list)):
            rho = self.rho_list[i]
            s_i = self.s_list[i]
            y_i = self.y_list[i]
            beta = rho * torch.sum(y_i * r)
            alpha = alpha_list.pop()
            r = r + (alpha - beta) * s_i
        
        # Save current values for next iteration
        self.prev_pos = pos.clone().detach()
        self.prev_forces = forces.clone().detach()
        
        # Return step (r is the approximate H⁻¹∇f)
        return r
    
    def reset(self) -> None:
        """Reset the L-BFGS memory."""
        self.s_list.clear()
        self.y_list.clear()
        self.rho_list.clear()
        self.prev_pos = None
        self.prev_forces = None
        self.H0_scaling = 1.0

class BatchedLBFGSStepSizer:
    """Batched wrapper around :class:`LBFGSStepSizer` for use with batched graphs.

    Maintains one :class:`LBFGSStepSizer` per graph in a batch and dispatches
    the step computation to the appropriate instance based on batch indices.
    """

    def __init__(self, batch_size: int, memory_size: int = 10, initial_step: float = 0.1) -> None:
        """Initialize one step-sizer per graph in the batch.

        Parameters
        ----------
        batch_size : int
            Number of graphs in the batch.
        memory_size : int, optional
            L-BFGS memory length (number of past iterations to retain).
        initial_step : float, optional
            Initial step-size scaling factor.
        """
        self.step_sizers = [LBFGSStepSizer(memory_size, initial_step) for _ in range(batch_size)]
    
    def compute_step(self, pos: torch.Tensor, forces: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """Compute steps for batched data.

        Parameters
        ----------
        pos : torch.Tensor
            Current atomic positions.
        forces : torch.Tensor
            Current forces acting on the atoms.
        batch_idx : torch.Tensor
            Index tensor mapping each atom to its graph in the batch.

        Returns
        -------
        torch.Tensor
            Combined step tensor with the same shape as *pos*.
        """
        results = []
        
        # Group positions and forces by batch index
        for i in range(len(self.step_sizers)):
            mask = batch_idx == i
            if torch.any(mask):
                pos_i = pos[mask]
                forces_i = forces[mask]
                step_i = self.step_sizers[i].compute_step(pos_i, forces_i)
                results.append(step_i)
        
        # Recombine results in original order
        combined_step = torch.zeros_like(pos)
        for i, step_i in enumerate(results):
            mask = batch_idx == i
            combined_step[mask] = step_i
            
        return combined_step
    
    def reset(self):
        """Reset the L-BFGS memory for all step-sizers in the batch."""
        for step_sizer in self.step_sizers:
            step_sizer.reset()

class Diffusion(LightningModule):
    """Class defining the full diffusion model.

    This class brings together the score model and the noisers and allow
    training and sampling

    Parameters
    ----------
    score_model: ScoreModel
        The score model.
    noisers: List[Noiser]
        A list of noisers.
    regressor_model: Optional[torch.nn.Module], optional
        An optional regressor model used for force-field guidance during sampling.
    optim_config: Dict
        The optimizer configuration.
    scheduler_config: Dict
        The scheduler configuration.
    eps: float
        Minimum value for the time step.

    Returns
    -------
    Diffusion
    """

    def __init__(
        self,
        score_model: ScoreModel,
        noisers: List[Noiser],
        regressor_model: Optional[torch.nn.Module] = None,
        optim_config: Dict = {"lr": 1e-4},
        scheduler_config: Dict = {"factor": 0.5, "patience": 10},
        eps: float = 1e-5,
    ) -> None:
        """Initializes the model."""
        super().__init__()
        self.score_model = score_model
        self.regressor_model = regressor_model
        self.lbfgs_step_sizer = None
        self.noisers = noisers

        self.noiser_keys = [noiser.key for noiser in noisers]
        self.score_keys = [head.key for head in score_model.heads]

        # if not set(self.noiser_keys) == set(self.score_keys):
        #     raise ValueError("Keys of noisers and score model heads do not match")

        for key in self.noiser_keys:
            if key not in ["x", "pos", "frac", "cellpar", "n_atoms", "forces"]:
                raise ValueError(f"Key {key} is not supported")

        for key in self.score_keys:
            if key not in ["x", "pos", "frac", "cellpar", "n_atoms", "forces"]:
                raise ValueError(f"Key {key} is not supported")

        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.eps = eps

        self._regressor_training = False

    def forward(self, batch: AtomsGraph) -> AtomsGraph:
        """Forward pass.

        Parameters
        ----------
        batch: AtomsGraph
            A batch of AtomsGraph data.

        Returns
        -------
        output: AtomsGraph
            The output of the forward pass.

        """
        return self.score_model(batch)

    def loss(self, batch: AtomsGraph, batch_idx: torch.Tensor) -> Dict:
        """Computes the loss.

        Parameters
        ----------
        batch: AtomsGraph
            A batch of AtomsGraph data.
        batch_idx: torch.Tensor
            The index of the batch.

        Returns
        -------
        losses: dict
            A dictionary of losses.

        """
        if self.regressor_training:
            return self.regressor_loss(batch, batch_idx)
        else:
            return self.diffusion_loss(batch, batch_idx)
        
    def diffusion_loss(self, batch: AtomsGraph, batch_idx: torch.Tensor) -> Dict:
        """Computes the loss.

        Parameters
        ----------
        batch: AtomsGraph
            A batch of AtomsGraph data.
        batch_idx: torch.Tensor
            The index of the batch.

        Returns
        -------
        losses: dict
            A dictionary of losses.

        """
        noised_batch = batch.clone()
        
        self.sample_time(noised_batch)
        noised_batch = self.forward_step(noised_batch)
        noised_batch = self.score_model(noised_batch)

        losses = {f"{noiser.key}_loss": 0 for noiser in self.noisers}
        losses["loss"] = 0.0
        for noiser in self.noisers:
            l = noiser.loss_scaling * noiser.loss(noised_batch)
            losses["loss"] += l
            losses[f"{noiser.key}_loss"] = l

        return losses

    def regressor_loss(self, batch: AtomsGraph, batch_idx: torch.Tensor) -> Dict:
        """Computes the loss.

        Parameters
        ----------
        batch: AtomsGraph
            A batch of AtomsGraph data.
        batch_idx: torch.Tensor
            The index of the batch.

        Returns
        -------
        losses: dict
            A dictionary of losses.

        """
        if self.regressor_model is None:
            raise ValueError("Regressor model is not defined.")

        loss = self.regressor_model.loss(batch)
        
        losses = {
            "regressor_loss": loss,
            "loss": loss
        }
        return losses

    def setup(self, stage: str = None) -> None:
        """Sets up the model.

        Parameters
        ----------
        stage: str
            The stage of training.

        Returns
        -------
        None

        """

        # self.offsets = torch.tensor(OFFSET_LIST).float().to(self.device)
        self.score_model.training_mode()
    
    def training_step(self, batch: AtomsGraph, batch_idx: torch.Tensor) -> torch.Tensor:
        """Performs a training step.

        Parameters
        ----------
        batch: AtomsGraph
            A batch of AtomsGraph data.
        batch_idx: torch.Tensor
            The index of the batch.

        Returns
        -------
        loss: torch.Tensor
            The loss of the training step.

        """
        losses = self.loss(batch, batch_idx)
        for k, v in losses.items():
            name = "train_loss" if k == "loss" else f"train/{k}"
            self.log(name, v, on_step=True, on_epoch=True, batch_size=batch.num_graphs)
        return losses["loss"]

    def validation_step(
        self, batch: AtomsGraph, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """Performs a validation step.

        Parameters
        ----------
        batch: AtomsGraph
            A batch of AtomsGraph data.
        batch_idx: torch.Tensor
            The index of the batch.

        Returns
        -------
        loss: torch.Tensor
            The loss of the validation step.

        """
        # if self.potential_model is not None:
        #     torch.set_grad_enabled(True)

        losses = self.loss(batch, batch_idx)
        for k, v in losses.items():
            name = "val_loss" if k == "loss" else f"val/{k}"
            self.log(name, v, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
        return losses["loss"]

    def configure_optimizers(self) -> Dict:
        """Configures the optimizers.

        Configures the optimizer and learning rate scheduler.

        Returns
        -------
        optimizers: Dict
            A dictionary of optimizers and learning rate schedulers.

        """
        if self.regressor_training:
            optimizer = torch.optim.AdamW(self.regressor_model.parameters(), **self.optim_config)
        else:
            optimizer = torch.optim.AdamW(self.score_model.parameters(), **self.optim_config)
            
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.scheduler_config
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def sample_time(self, batch: AtomsGraph) -> None:
        """Samples the time.

        Parameters
        ----------
        batch: AtomsGraph
            A batch of AtomsGraph data.

        Returns
        -------
        None

        """
        batch_size = batch.batch_size
        time = torch.rand(batch_size) * (1.0 - self.eps) + self.eps
        batch.time = time.to(self.device)[batch.batch].unsqueeze(1)

    def _initialize_graph(self, cutoff: float, **kwargs) -> AtomsGraph:
        """Initializes a graph.

        Initializes a graph with the provided keyword arguments and
        from the noisers prior distributions.

        Parameters
        ----------
        cutoff : float
            Cutoff radius for the neighbour list.
        **kwargs
            Additional keyword arguments passed to the graph (e.g. ``cell``,
            ``template``).

        Returns
        -------
        AtomsGraph
            The initialized graph.

        """
        graph = AtomsGraph.empty(cutoff=cutoff)
        if "template" in kwargs:
            template = kwargs.pop("template")
        else:
            template = None

        if 'cell' in kwargs:
            cell = kwargs.pop('cell')
            setattr(graph, 'cell', cell)
            
        for k, v in kwargs.items():
            setattr(graph, k, v)

        for noiser in self.noisers[::-1]:
            noiser.initialize_graph(graph)

        if template is not None:
            new_graph = template.clone()

            setattr(new_graph, "x", torch.cat([
                template.x,
                graph.x
            ]))

            setattr(new_graph, "pos", torch.cat([
                template.pos,
                graph.pos
            ]))
            
            setattr(new_graph, "mask", torch.cat([
                torch.ones_like(template.x, dtype=torch.bool),
                torch.zeros_like(graph.x, dtype=torch.bool)
            ]))

            setattr(new_graph, "n_atoms", template.n_atoms + graph.n_atoms)
        else:
            new_graph = graph
            setattr(new_graph, "mask", torch.zeros_like(graph.x, dtype=torch.bool))

        return new_graph

    def sample(
        self,
        N: int,
        template: Optional[AtomsGraph] = None,
        batch_size: Optional[int] = 64,
        steps: Optional[int] = 500,
        cutoff: Optional[float] = 6.0,
        eps: Optional[float] = 1e-3,
        n_atoms: Optional[int] = None,
        atomic_numbers: Optional[List[int]] = None,
        formula: Optional[str] = None,
        positions: Optional[np.ndarray] = None,
        cell: Optional[np.ndarray] = None,
        pbc: Optional[np.ndarray] = None,
        confinement: Optional[Tuple[float, float]] = None,
        force_field_guidance: float = 0.0,
        zeta: float = 3.0,
        force_threshold: float = 0.05,
        max_extra_steps: int = 100,
        property: Optional[Dict] = None,
        progress_bar: Optional[bool] = False,
        save_path: Optional[bool] = False,
    ) -> List[AtomsGraph]:
        """Samples from the model.

        External method to sample from the model.
        Sets up the kwargs for the internal _sample method with
        atomic_numbers, n_atoms, positions and cell.

        The minimum required arguments depend on the configured noisers and
        whether a template is provided:

        * ``n_atoms`` – always required unless derivable from ``atomic_numbers``
          or ``formula``.
        * ``atomic_numbers`` – required unless a types-noiser is configured
          (key ``"x"``), or derivable from ``formula``.
        * ``positions`` – required when no positions-noiser is configured (e.g.
          type-only diffusion).  Positions are kept fixed during sampling.
          Not required when a positions-noiser is present (they are sampled
          from the prior at initialisation).
        * ``cell`` – required when no ``template`` is given; inferred from the
          template when one is provided.
        * ``pbc`` – optional; defaults to ``[True, True, True]`` when not given.

        Parameters
        ----------
        N: int
            The number of samples to generate.
        template: Optional[AtomsGraph]
            Template structure. The ``cell`` and ``pbc`` are taken from the
            template when not explicitly provided.
        batch_size: Optional[int]
            The batch size.
        steps: Optional[int]
            The number of steps to take.
        cutoff: Optional[float]
            The cutoff distance.
        eps: Optional[float]
            Minimum time value during for sampling.
        n_atoms: Optional[int]
            The number of atoms to generate. Derived from ``formula`` or
            ``atomic_numbers`` when not given.
        atomic_numbers: Optional[List[int]]
            Atomic numbers of the atoms to generate. Not required when a
            types-noiser is configured or when ``formula`` is provided.
        formula: Optional[str]
            Chemical formula (e.g. ``"H2O"``). Used to derive ``n_atoms``
            and ``atomic_numbers`` when they are not provided explicitly.
        positions: Optional[np.ndarray]
            Fixed positions of the atoms (shape ``(n_atoms, 3)``).  Required
            when no positions-noiser is configured (type-only diffusion).
            Positions will not be modified during sampling.
        cell: Optional[np.ndarray]
            Unit-cell matrix (3×3). Not required when ``template`` is given.
        pbc: Optional[np.ndarray]
            Periodic boundary conditions. Not required when ``template`` is
            given.
        confinement: Optional[Tuple[float, float]]
            Z-directional confinement if noiser distribution supports it.
        property: Dict[str, float]
            The property to condition on.
        progress_bar: Optional[bool]
            Whether to show a progress bar.
        """

        self.score_model.sample_mode()

        # Derive n_atoms / atomic_numbers from a molecular formula if given.
        if formula is not None:
            from ase import Atoms as _AseAtoms
            _formula_atoms = _AseAtoms(formula)
            if n_atoms is None:
                n_atoms = len(_formula_atoms)
            if atomic_numbers is None and "x" not in self.noiser_keys:
                atomic_numbers = _formula_atoms.get_atomic_numbers().tolist()

        # When a template is provided but no cell is given, borrow the
        # template's cell so noiser priors (e.g. UniformCell) can use it.
        if template is not None and cell is None:
            cell = template.cell.detach().cpu().numpy()

        kwargs = {
            "progress_bar": progress_bar,
            "save_path": save_path,
            "force_threshold": force_threshold,
            "max_extra_steps": max_extra_steps,
        }
        self.zeta = zeta

        if n_atoms is not None:
            kwargs["n_atoms"] = torch.tensor([n_atoms]).reshape(1, 1)
        if positions is not None:
            kwargs["pos"] = torch.tensor(np.array(positions), dtype=torch.float).reshape(
                -1, 3
            )
            if "n_atoms" not in kwargs:
                kwargs["n_atoms"] = torch.tensor([kwargs["pos"].shape[0]]).reshape(1, 1)
        if atomic_numbers is not None:
            kwargs["x"] = torch.tensor(atomic_numbers, dtype=torch.long).reshape(-1)
            if "n_atoms" not in kwargs:
                kwargs["n_atoms"] = torch.tensor([len(atomic_numbers)]).reshape(1, 1)

        if cell is not None:
            kwargs["cell"] = torch.tensor(np.array(cell), dtype=torch.float).reshape(
                3, 3
            )

        if property is not None:
            for k, v in property.items():
                kwargs[k] = torch.tensor(v, dtype=torch.float)

        for key in ["pos", "x", "cell", "n_atoms"]:
            if key not in kwargs and key not in self.noiser_keys:
                if key == "pos" and "frac" in self.noiser_keys:
                    continue
                raise ValueError(f"Missing default values for key {key} in kwargs.")

        if confinement is not None:
            kwargs["confinement"] = torch.tensor(confinement, dtype=torch.float).reshape(1, 2)

        if template is not None:
            kwargs["template"] = template
        else:
            n_atoms = kwargs["n_atoms"].item()

        if pbc is not None:
            kwargs["pbc"] = torch.tensor(pbc, dtype=torch.bool).reshape(3)

        if N > batch_size:
            n_full = N // batch_size
            n_remainder = N % batch_size
            n_batches = n_full + (1 if n_remainder > 0 else 0)
            out = []
            for i in range(n_full):
                print(f"Sampling batch {i + 1}/{n_batches}...")
                out += self._sample(batch_size, steps, cutoff, eps, force_field_guidance, **kwargs)
            if n_remainder > 0:
                print(f"Sampling batch {n_batches}/{n_batches}...")
                out += self._sample(n_remainder, steps, cutoff, eps, force_field_guidance, **kwargs)
            return out
        else:
            return self._sample(N, steps, cutoff, eps, force_field_guidance, **kwargs)

    def _sample(
            self, N: int, steps: int, cutoff: float, eps: float, force_field_guidance: float, progress_bar: bool, save_path: bool, **kwargs
    ) -> List[AtomsGraph]:
        """Samples from the model.

        Internal method that performs the sampling.

        Parameters
        ----------
        N: int
            The number of samples to generate.
        steps: int
            The number of steps to take.
        cutoff: float
            The cutoff distance.
        eps: float
            Minimum time value during for sampling.
        force_field_guidance: float
                The scale of the force field guidance.
        kwargs: dict
            The keyword arguments.

        Returns
        -------
        samples: List[AtomsGraph]
            The samples.

        """
        data = []
        for _ in range(N):
            data.append(self._initialize_graph(cutoff, **kwargs))

        batch = Batch.from_data_list(data).to(self.device)
        batch.update_graph()

        return self._sample_batch(batch, steps, eps, force_field_guidance, save_path, progress_bar)


    def _sample_batch(self, batch: Batch, steps: int, eps: float, force_field_guidance: float, save_path: bool, progress_bar: bool, force_threshold: float = 0.05, max_extra_steps: int = 100) -> List[AtomsGraph]:
        """Samples a batch of data.
        Internal method that performs the sampling for a batch of data.
        Parameters
        ----------
        batch: Batch
                A batch of AtomsGraph data.
        steps: int
                The number of steps to take.
        eps: float
                Minimum time value during for sampling.
        force_field_guidance: float
                The scale of the force field guidance.
        save_path: bool
                Whether to save the path of the sampling.
        progress_bar: bool
                Whether to show a progress bar.
        force_threshold: float
                Maximum allowed force for terminating relaxation.
        max_extra_steps: int
                Maximum number of extra relaxation steps to perform.
        
        Returns
        -------
        samples: List[AtomsGraph]
                The samples.
        """
        if steps < 2:
            return batch.to_data_list()

        if force_field_guidance > 0 and self.regressor_model is not None:
            self.lbfgs_step_sizer = BatchedLBFGSStepSizer(batch_size=batch.batch_size)
        
            
        ts = torch.linspace(1, eps, steps, device=self.device)
        dt = ts[0] - ts[1]
        
        if save_path:
            path = []
            
        if progress_bar:
            iterator = tqdm(range(steps))
        else:
            iterator = range(steps)
            
        for i in iterator:
            if save_path:
                path.append(batch.to_data_list())
                
            batch.add_batch_attr("time", ts[i].repeat(batch.x.shape[0], 1), type="node")
            if i < steps - 1:
                batch = self.reverse_step(batch, dt, force_field_guidance)
            else:
                batch = self.reverse_step(batch, dt, force_field_guidance, last=True)


        # Now check if further relaxation is needed
        if force_field_guidance > 0 and self.regressor_model is not None:
            # Apply regressor to get forces
            batch = self.regressor_model(batch)
            max_forces = torch.norm(batch.forces_prediction, dim=1).max(dim=0)[0]

            # Check if forces exceed threshold
            if max_forces > force_threshold:
                if progress_bar:
                    print(f"Max force after diffusion: {max_forces:.4f}, continuing relaxation...")
                    extra_iterator = tqdm(range(max_extra_steps), desc="Post-diffusion relaxation")
                else:
                    extra_iterator = range(max_extra_steps)

                # Set time to zero for post-diffusion relaxation
                batch.add_batch_attr("time", torch.zeros_like(batch.time), type="node")

                # Continue relaxation until forces are below threshold or max steps reached
                for i in extra_iterator:
                    # Apply relaxation step
                    batch = self.post_diffusion_relaxation_step(batch, scale=0.1)

                    # Check if forces are now below threshold
                    batch = self.regressor_model(batch)
                    max_forces = torch.norm(batch.forces_prediction, dim=1).max(dim=0)[0]

                    if save_path:
                        path.append(batch.to_data_list())

                    if max_forces <= force_threshold:
                        if progress_bar:
                            print(f"Relaxation converged after {i+1} steps, max force: {max_forces:.4f}")
                        break

                if progress_bar and max_forces > force_threshold:
                    print(f"Relaxation did not converge, final max force: {max_forces:.4f}")

                
        if save_path:
            path.append(batch.to_data_list())
            return list(map(list, zip(*path)))

        return batch.to_data_list()

    def forward_step(self, batch: AtomsGraph) -> AtomsGraph:
        """Forward diffusion step

        Performs a forward step in the diffusion model.
        This corresponds to the forward pass of the noisers and
        thus corrupts the data.

        Parameters
        ----------
        batch: AtomsGraph
            A batch of AtomsGraph data.

        Returns
        -------
        batch: AtomsGraph
            The output of the forward step.
        """
        for noiser in self.noisers:
            batch = noiser.noise(batch)
        
        batch.update_graph()
        return batch

    def reverse_step(self, batch: AtomsGraph, delta_t: float, force_field_guidance: float, last: bool=False) -> AtomsGraph:
        """Reverse diffusion step

        Performs a reverse step in the diffusion model.
        This corresponds to the calculating the score and performing a reverse
        sampling step in the noisers.

        Parameters
        ----------
        batch: AtomsGraph
            A batch of AtomsGraph data.
        delta_t: float
            The time step.

        Returns
        -------
        batch: AtomsGraph
            The output of the reverse step.

        """
        batch = self.score_model(batch)
        for noiser in self.noisers[::-1]:
            batch = noiser.denoise(batch, delta_t, last=last)

        batch.wrap_positions()
        batch.update_graph()

            
        if self.regressor_model is not None and force_field_guidance > 0.0:
            batch = self.force_field_guidance_step(batch, force_field_guidance*delta_t)
            batch.wrap_positions()
            batch.update_graph()

        return batch

    # def force_field_guidance_step(self, batch: AtomsGraph, scale: float) -> AtomsGraph:
    #     """Applies force field guidance to the batch.

    #     Parameters
    #     ----------
    #     batch: AtomsGraph
    #         A batch of AtomsGraph data.
    #     scale: float
    #         The scale of the force field guidance.

    #     Returns
    #     -------
    #     batch: AtomsGraph
    #         The output of the force field guidance step.

    #     """
    #     if self.regressor_model is None:
    #         return batch

    #     batch = self.regressor_model(batch)
        
    #     if "forces_prediction" not in batch:
    #         raise ValueError("Regressor model does not compute forces.")

    #     batch.pos = batch.pos + scale * (1-batch.time) * batch.forces_prediction
    #     return batch

    def force_field_guidance_step(self, batch: AtomsGraph, scale: float, max_step_size: float = 0.1) -> AtomsGraph:
        """Applies force field guidance with batched L-BFGS step size adaptation.

        Parameters
        ----------
        batch: AtomsGraph
            A batch of AtomsGraph data.
        scale: float
            The base scale of the force field guidance.
        max_step_size : float, optional
            Maximum allowed step size magnitude.  Default is 0.1.

        Returns
        -------
        batch: AtomsGraph
            The output of the force field guidance step.
        """
        if self.regressor_model is None:
            return batch

        # Apply regressor model to get forces
        batch = self.regressor_model(batch)

        if "forces_prediction" not in batch:
            raise ValueError("Regressor model does not compute forces.")

        # Get current positions and forces
        positions = batch.pos
        forces = batch.forces_prediction
        batch_idx = batch.batch

        # Initialize L-BFGS step sizer if not already done
        if self.lbfgs_step_sizer is None:
            batch_size = batch.batch_size
            self.lbfgs_step_sizer = BatchedLBFGSStepSizer(batch_size=batch_size)

        # Get time-dependent scaling factor
        time_factor = (1.0 - batch.time)**self.zeta

        # Use L-BFGS to compute optimal step direction and magnitude
        lbfgs_step = self.lbfgs_step_sizer.compute_step(positions, forces, batch_idx)

        # Apply step with base scale and time factor, clamping to max_step_size
        step = scale * time_factor * lbfgs_step
        step_magnitude = torch.norm(step, dim=1, keepdim=True)
        too_large = step_magnitude > max_step_size
        if torch.any(too_large):
            scaling_factor = torch.ones_like(step_magnitude)
            scaling_factor[too_large] = max_step_size / step_magnitude[too_large]
            step = step * scaling_factor

        # Calculate new positions
        new_pos = batch.pos + step

        # Check if we need to apply confinement
        if hasattr(batch, 'confinement') and batch.confinement is not None:
            # Assuming confinement is [min_z, max_z] for z-direction
            z_min = batch.confinement[:, 0].unsqueeze(1)  # [B, 1]
            z_max = batch.confinement[:, 1].unsqueeze(1)  # [B, 1]

            # Get batch indices for each atom
            batch_indices = batch.batch

            # Convert batch-level confinement to atom-level
            z_min_per_atom = z_min[batch_indices].squeeze()  # [N]
            z_max_per_atom = z_max[batch_indices].squeeze()  # [N]

            # Clamp z-positions to stay within confinement
            new_pos[:, 2] = torch.clamp(new_pos[:, 2], min=z_min_per_atom, max=z_max_per_atom)

        batch.pos = new_pos
        return batch

    def post_diffusion_relaxation_step(self, batch: AtomsGraph, scale: float = 0.1) -> AtomsGraph:
        """Performs a pure force-based relaxation step after diffusion is complete.

        Parameters
        ----------
        batch: AtomsGraph
            A batch of AtomsGraph data.
        scale: float
            Step size scaling factor for relaxation.

        Returns
        -------
        batch: AtomsGraph
            Updated batch after relaxation step.
        """
        if self.regressor_model is None:
            return batch

        # Get forces from regressor model
        batch = self.regressor_model(batch)

        if "forces_prediction" not in batch:
            raise ValueError("Regressor model does not compute forces.")

        # Get current positions and forces
        positions = batch.pos
        forces = batch.forces_prediction
        batch_idx = batch.batch

        # Use the L-BFGS step sizer to determine optimal step
        # If it doesn't exist yet, initialize it
        if self.lbfgs_step_sizer is None:
            self.lbfgs_step_sizer = BatchedLBFGSStepSizer(
                batch_size=batch.batch_size,
                memory_size=10,  # Default value
                initial_step=0.1  # Default value
            )

        # Compute step using L-BFGS
        lbfgs_step = self.lbfgs_step_sizer.compute_step(positions, forces, batch_idx)

        # Calculate new positions
        new_pos = batch.pos + scale * lbfgs_step

        # Check if we need to apply confinement
        if hasattr(batch, 'confinement') and batch.confinement is not None:
            # Assuming confinement is [min_z, max_z] for z-direction
            z_min = batch.confinement[:, 0].unsqueeze(1)  # [B, 1]
            z_max = batch.confinement[:, 1].unsqueeze(1)  # [B, 1]

            # Get batch indices for each atom
            batch_indices = batch.batch

            # Convert batch-level confinement to atom-level
            z_min_per_atom = z_min[batch_indices].squeeze()  # [N]
            z_max_per_atom = z_max[batch_indices].squeeze()  # [N]

            # Clamp z-positions to stay within confinement
            new_pos[:, 2] = torch.clamp(new_pos[:, 2], min=z_min_per_atom, max=z_max_per_atom)

        batch.pos = new_pos

        # Wrap positions and update graph
        batch.wrap_positions()
        batch.update_graph()

        return batch

    @property
    def regressor_training(self) -> bool:
        """Whether the regressor model is in training mode."""
        if self.regressor_model is None:
            return False
        return self._regressor_training

    @regressor_training.setter
    def regressor_training(self, value: bool) -> None:
        """Sets the regressor model in training mode.

        Parameters
        ----------
        value: bool
            Whether to set the regressor model in training mode.

        Returns
        -------
        None

        """
        if self.regressor_model is None:
            self._regressor_training = False
            return

        self._regressor_training = value
    
