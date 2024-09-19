from typing import Dict

import pytorch_lightning as pl
import torch


class Diffusion(pl.LightningModule):
    def __init__(
        self,
        score_model,
        noisers,
        optim_config={"lr": 1e-4},
        scheduler_config={"factor": 0.5, "patience": 10},
    ):
        super().__init__()
        self.score_model = score_model
        self.noisers = noisers

        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        

    def forward(self, batch):
        """
        Forward pass.

        Args
        ______
        batch: dict
            A batch of data.

        Returns
        _______
        output: dict
            The output of the forward pass.

        """
        return self.score_model(batch)

    def loss(self, batch, batch_idx):
        """
        Computes the loss.

        Args
        ______
        batch: dict
            A batch of data.
        batch_idx: torch.Tensor
            The index of the batch.

        Returns
        _______
        losses: dict
            A dictionary of losses.

        """
        noised_batch = batch.clone()

        self.sample_time(noised_batch)
        for noiser in self.noisers:
            noised_batch = noiser.noise(noised_batch)

        noised_batch.update_graph()
        noised_batch = self.forward(noised_batch)

        loss = 0
        for noiser in self.noisers:
            loss += noiser.loss(noised_batch)

        return {"loss": loss}

    def setup(self, stage: str = None) -> None:
        """
        Sets up the model.

        Args
        ______
        stage: str
            The stage of training.

        """

        # self.offsets = torch.tensor(OFFSET_LIST).float().to(self.device)
        pass

    def training_step(self, batch, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Performs a training step.

        Args
        ______
        batch: dict
            A batch of data.
        batch_idx: torch.Tensor
            The index of the batch.

        Returns
        _______
        loss: torch.Tensor
            The loss of the training step.

        """
        losses = self.loss(batch, batch_idx)
        for k, v in losses.items():
            self.log("train_" + k, v)
        return losses["loss"]

    def validation_step(self, batch, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Performs a validation step.

        Args
        ______
        batch: dict
            A batch of data.
        batch_idx: torch.Tensor
            The index of the batch.

        Returns
        _______
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
        """
        Configures the optimizer and learning rate scheduler.

        Returns
        _______
        optimizers: dict
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

    def sample_time(self, batch):
        """
        Samples the time.

        Args
        ______
        batch: dict
            A batch of data.

        Returns
        _______
        time: torch.Tensor
            The sampled time.

        """
        batch_size = batch.batch_size
        time = torch.rand(batch_size).to(self.device)[batch.batch].unsqueeze(1)
        batch.time = time

    def sample(self, N, template, **kwargs):
        pass

    def forward_step(self, batch):
        pass

    def reverse_step(self, batch):
        pass
