from typing import List
from lightning.pytorch.callbacks import Callback
from torch_geometric.transforms import BaseTransform


class TrainingPhase(Callback):
    def __init__(
        self,
        n_phases: int,
        epochs_per_phase: List[int],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_phases = n_phases

        self.epochs_per_phase = epochs_per_phase
        self.epoch_counter = 0

        self.current_phase = 0

    def _prepare_epoch(self, trainer, model):
        # phase = ...
        # trainer.datamodule.set_phase(phase)

        if self.current_phase == self.n_phases - 1:
            return

        epoch = trainer.current_epoch
        if self.epoch_counter >= self.epochs_per_phase[self.current_phase]:
            self.current_phase += 1
            trainer.datamodule.set_phase(self.current_phase)

            self.epoch_counter = 0
        else:
            self.epoch_counter += 1

    def on_validation_end(self, trainer, model):
        self._prepare_epoch(trainer, model)
