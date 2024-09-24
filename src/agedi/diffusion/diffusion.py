from typing import Dict

import pytorch_lightning as pl
import torch

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

    def sample(self, N, template, **kwargs):
        pass

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

    def reverse_step(self, batch: AtomsGraph) -> AtomsGraph:
        """Reverse diffusion step
        
        Performs a reverse step in the diffusion model.
        This corresponds to the calculating the score and performing a reverse
        sampling step in the noisers.

        Parameters
        ----------
        batch: AtomsGraph
            A batch of AtomsGraph data.

        Returns
        -------
        batch: AtomsGraph
            The output of the reverse step.
        
        """
        batch = self.score_model(batch)
        for noiser in self.noisers[::-1]:
            batch = noiser.denoise(batch)

        batch.update_graph()
        return batch
