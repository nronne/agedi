from typing import Dict, Union, List

import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch

from agedi.models import ScoreModel
from agedi.diffusion.noisers import Noiser
from agedi.data import AtomsGraph


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
        optim_config: Dict={"lr": 1e-4},
        scheduler_config: Dict={"factor": 0.5, "patience": 10},
    ) -> None:
        """Initializes the model.
        
        """
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

    def validation_step(self, batch: AtomsGraph, batch_idx: torch.Tensor) -> torch.Tensor:
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

    def initialize_graph(self, cutoff, **kwargs) -> AtomsGraph:
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
            setattr(graph, noiser.key, noiser.prior.get_callable(graph)())

        return graph
        
    def sample(
        self,
        N: int,
        template: AtomsGraph=None,
        batch_size: int=64,
        steps: int=500,
        cutoff: float=6.0,
        eps: float =1e-4,
        **kwargs
    ) -> List[AtomsGraph]:
        """Samples from the model.

        Parameters
        ----------
        N: int
            The number of samples to generate.
        template: Union[AtomsGraph, Atoms]
            The template to use for sampling.
        batch_size: int
            The batch size.
        steps: int
            The number of steps to take.
        kwargs: dict
            Additional keyword arguments. Which may include:
            - `n_atoms`: torch.Tensor
            - `pos`: torch.Tensor
            - `cell`: torch.Tensor
            - `x`: torch.Tensor
        
        """
        # check that kwargs include 
        # except if their in self.noiser_keys
        for key in ["pos", "x", "cell", "n_atoms"]:
            if key not in kwargs or key not in self.noiser_keys:
                raise ValueError(f"Missing default values for key {key} in kwargs.")
            
        if N > batch_size:
            out = []
            for _ in range(N // batch_size):
                out += self.sample(batch_size, template, steps, **kwargs)
            out += self.sample(N % batch_size, template, steps, **kwargs)
            return out

        if template is not None:
            raise NotImplementedError("Sampling from a template is not yet implemented.")

        data = []
        for _ in range(N):
            data.append(self.initialize_graph(cutoff, **kwargs))

        batch = Batch.from_data_list(data)
        batch.update_graph()

        ts = torch.linspace(1, eps, steps)
        dt = ts[1] - ts[0]
        for t in ts:
            batch.time = t.repeat(batch.x.shape[0], 1).to(self.device)
            batch = self.forward_step(batch, dt)

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
