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


class Diffusion(LightningModule):
    """Class defining the full diffusion model.

    This class brings together the score model and the noisers and allow
    training and sampling

    Parameters
    ----------
    score_model: torch.nn.Module
        The score model.
    noisers: List[Noiser]
        A list of noisers.
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
        noisers: list[Noiser],
        optim_config: Dict = {"lr": 1e-4},
        scheduler_config: Dict = {"factor": 0.5, "patience": 10},
        eps: float = 1e-5,
    ) -> None:
        """Initializes the model."""
        super().__init__()
        self.score_model = score_model
        self.noisers = noisers

        self.noiser_keys = [noiser.key for noiser in noisers]
        self.score_keys = [head.key for head in score_model.heads]

        if not set(self.noiser_keys) == set(self.score_keys):
            raise ValueError("Keys of noisers and score model heads do not match")

        for key in self.noiser_keys:
            if key not in ["x", "pos", "cell", "n_atoms"]:
                raise ValueError(f"Key {key} is not supported")

        for key in self.score_keys:
            if key not in ["x", "pos", "cell", "n_atoms"]:
                raise ValueError(f"Key {key} is not supported")

        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.eps = eps

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
            self.log("train_" + k, v)
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
            self.log("val_" + k, v)
        return losses["loss"]

    def configure_optimizers(self) -> Dict:
        """Configures the optimizers.

        Configures the optimizer and learning rate scheduler.

        Returns
        -------
        optimizers: Dict
            A dictionary of optimizers and learning rate schedulers.

        """
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

    def _initialize_graph(self, cutoff, **kwargs) -> AtomsGraph:
        """Initializes a graph.

        Initializes a graph with the provided keyword arguments and
        from the noisers prior distributions.

        Parameters
        ----------
        kwargs: dict
            The keyword arguments.

        Returns
        -------
        graph: AtomsGraph
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
            setattr(
                graph,
                noiser.key,
                noiser.prior.get_callable(graph)(),
            )
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
            setattr(new_graph, "n_atoms", torch.tensor(new_graph.x.shape[0]).reshape(1, 1))
        else:
            new_graph = graph

        return new_graph

    def sample(
        self,
        N: int,
        template: Optional[AtomsGraph] = None,
        batch_size: Optional[int] = 64,
        steps: Optional[int] = 500,
        cutoff: Optional[float] = 6.0,
        eps: Optional[float] = 1e-4,
        n_atoms: Optional[int] = None,
        positions: Optional[np.ndarray] = None,
        atomic_numbers: Optional[List[int]] = None,
        cell: Optional[np.ndarray] = None,
        pbc: Optional[np.ndarray] = None,
        confinement: Optional[Tuple[float, float]] = None,
        property: Optional[Dict] = None,
        progress_bar: Optional[bool] = False,
        save_path: Optional[bool] = False,
    ) -> List[AtomsGraph]:
        """Samples from the model.

        External method to sample from the model.
        Sets up the kwargs for the internal _sample method with positions,
        atomic_numbers, n_atoms and cell.

        Parameters
        ----------
        N: int
            The number of samples to generate.
        template: Optional[AtomsGraph]
            The template to use for sampling.
        batch_size: Optional[int]
            The batch size.
        steps: Optional[int]
            The number of steps to take.
        cutoff: Optional[float]
            The cutoff distance.
        eps: Optional[float]
            Minimum time value during for sampling.
        n_atoms: Optional[int]
            The number of atoms.
        positions: Optional[np.ndarray]
            The positions of the atoms.
        atomic_numbers: Optional[List[int]]
            The atomic numbers of the atoms.
        cell: Optional[np.ndarray]
            The cell of the atoms.
        confinement: Optional[Tuple[float, float]]
            Z-directional confinement if noiser distribution supports it.
        property: Dict[str: float]
            The property to condition on.
        progress_bar: Optional[bool]
            Whether to show a progress bar.
        """

        self.score_model.sample_mode()
        
        # check that kwargs include
        # except if their in self.noiser_keys
        kwargs = {
            "progress_bar": progress_bar,
            "save_path": save_path,
        }

        if n_atoms is not None:
            kwargs["n_atoms"] = torch.tensor([n_atoms]).reshape(1, 1)
        if positions is not None:
            kwargs["pos"] = torch.tensor(positions, dtype=torch.float).reshape(
                n_atoms, 3
            )
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
            out = []
            for _ in range(N // batch_size):
                out += self._sample(batch_size, steps, cutoff, eps, **kwargs)
            out += self._sample(N % batch_size, steps, cutoff, eps, **kwargs)
            return out
        else:
            return self._sample(N, steps, cutoff, eps, **kwargs)

    def _sample(
            self, N: int, steps: int, cutoff: float, eps: float, progress_bar: bool, save_path: bool, **kwargs
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

        if steps < 2:
            return batch.to_data_list()
            
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
                batch = self.reverse_step(batch, dt)
            else:
                batch = self.reverse_step(batch, dt, last=True)

        if save_path:
            path.append(batch.to_data_list())
            return list(map(list, zip(*path)))

        return batch.to_data_list()

    def forward_step(self, batch: AtomsGraph, update_graph=True) -> AtomsGraph:
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

        if update_graph:
            batch.update_graph()
            
        return batch

    def reverse_step(self, batch: AtomsGraph, delta_t: float, last: bool=False) -> AtomsGraph:
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
        return batch

    
