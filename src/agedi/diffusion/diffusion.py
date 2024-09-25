from typing import Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch

from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser
from agedi.models import ScoreModel


class Diffusion(pl.LightningModule):
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

        loss = 0
        for noiser in self.noisers:
            loss += noiser.loss(noised_batch)

        return {"loss": loss}

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
        pass

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
        optimizer = torch.optim.Adam(self.score_model.parameters(), **self.optim_config)
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
        time = torch.rand(batch_size).to(self.device)[batch.batch].unsqueeze(1)
        batch.time = time

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
        for k, v in kwargs.items():
            setattr(graph, k, v)

        for noiser in self.noisers[::-1]:
            mu = graph[noiser.key]
            setattr(
                graph,
                noiser.key,
                noiser.prior.get_callable(graph)(mu, None),
            )
        return graph

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
        """
        # check that kwargs include
        # except if their in self.noiser_keys
        kwargs = {}

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

        for key in ["pos", "x", "cell", "n_atoms"]:
            if key not in kwargs and key not in self.noiser_keys:
                raise ValueError(f"Missing default values for key {key} in kwargs.")

        if template is not None:
            raise NotImplementedError(
                "Sampling with a template is not yet implemented."
            )
        else:
            n_atoms = kwargs["n_atoms"].item()
            # kwargs["mask"] = torch.zeros((n_atoms,), dtype=torch.bool)
            # print(kwargs["mask"])

        if N > batch_size:
            out = []
            for _ in range(N // batch_size):
                out += self._sample(batch_size, steps, cutoff, eps, **kwargs)
            out += self._sample(N % batch_size, steps, cutoff, eps, **kwargs)
            return out
        else:
            return self._sample(N, steps, cutoff, eps, **kwargs)

    def _sample(
        self, N: int, steps: int, cutoff: float, eps: float, **kwargs
    ) -> List[AtomsGraph]:
        """Samples from the model.

        Internal method that performs the sampling.that performs the samp

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

        batch = Batch.from_data_list(data)
        batch.update_graph()

        ts = torch.linspace(1, eps, steps)
        dt = ts[0] - ts[1]
        for t in ts:
            batch.add_batch_attr("time", t.repeat(batch.x.shape[0], 1), type="node")
            batch = self.reverse_step(batch, dt)

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

    def reverse_step(self, batch: AtomsGraph, delta_t: float) -> AtomsGraph:
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
            batch = noiser.denoise(batch, delta_t)

        batch.update_graph()
        return batch
