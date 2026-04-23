from abc import ABC, abstractmethod
from typing import Dict

import torch
from lightning import LightningModule


class Conditioning(ABC, LightningModule):
    """Conditioning Base Class

    Parameters
    ----------
    property: str
        The property of the batch to condition on
    input_dim: int
        The dimension of the input conditioning
    output_dim: int
        The dimension of the output conditioning
    concatenation_type: str
        The type of concatenation to use. Default is "scalar"
    probability: float
        The probability of conditioning. Default is 0.5. Only used in training mode

    Returns
    -------
    Conditioning
    
    """

    def __init__(
        self,
        property: str,
        input_dim: int,
        output_dim: int,
        concatenation_type: str = "scalar",
        probability: float = 0.8,
        **kwargs,
    ) -> None:
        """Constructor for the Conditioning class
        """
        super().__init__(**kwargs)
        self.property = property
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.concatenation_type = concatenation_type
        self.probability = probability

    def get_hparams(self) -> Dict:
        """Return hyperparameters sufficient to reconstruct this conditioning module.

        Returns a dictionary with a ``_target_`` key (the fully-qualified class
        name) plus ``property``, ``probability``, ``input_dim``, and
        ``output_dim`` from the base class.
        Subclasses should call ``super().get_hparams()`` and merge in their own
        constructor parameters.

        Returns
        -------
        dict
            Hyperparameter dictionary.
        """
        return {
            "_target_": f"{type(self).__module__}.{type(self).__qualname__}",
            "property": self.property,
            "probability": self.probability,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }

    @abstractmethod
    def get_conditioning(self, x: torch.Tensor) -> torch.Tensor:
        """Abstract method to get the conditioning from the input

        Must be implemented by the subclass

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The conditioning tensor
        
        """
        pass

    @abstractmethod
    def get_empty_conditioning(self, n: int) -> torch.Tensor:
        """Abstract method to get an empty conditioning tensor

        Must be implemented by the subclass

        Parameters
        ----------
        n: int
            The number of nodes in the batch

        Returns
        torch.Tensor
            The empty conditioning tensor
        
        """
        
    def forward(self, batch: "AtomsGraph", empty: bool=False) -> "AtomsGraph":
        """Forward method to get the conditioning from the input

        Parameters
        ----------
        batch: AtomsGraph
            The input batch
        empty: bool
            If True, return an empty conditioning tensor

        Returns
        -------
        AtomsGraph
            The batch with the conditioning added to the representation
        
        """
        if self.sample:
            if empty:
                c = self.get_empty_conditioning(batch[self.property].shape[0])
            else:
                c = self.get_conditioning(batch[self.property])

        else:
            n = batch.batch_size
            cond_size = batch[self.property].shape[0]
            cond_idx = torch.rand(n, device=batch.batch.device) < self.probability
            if cond_size != n:
                cond_idx = cond_idx[batch.batch]

            c = self.get_empty_conditioning(cond_size)
            c[cond_idx] = self.get_conditioning(batch[self.property])[cond_idx]

        self.concatenate(batch, c)
        return batch

    def concatenate(self, batch: "AtomsGraph", c: torch.Tensor) -> None:
        """Concatenate the conditioning to the batch

        Parameters
        ----------
        batch: AtomsGraph
            The input batch
        c: torch.Tensor
            The conditioning tensor

        Returns
        -------
        None
        
        """
        if self.concatenation_type == "scalar":
            rep = batch.representation
            scalar = rep.scalar

            if scalar.shape[0] != c.shape[0]:
                # expand from structure level -> node level
                c = c[batch.batch, ..., None]

            if len(scalar.shape) != len(c.shape):
                c = c[..., None]
                
            new_scalar = torch.cat((scalar, c), dim=1)
            rep.scalar = new_scalar
            batch.representation = rep

        else:
            raise ValueError(
                f"Concatenation type {self.concatenation_type} not supported"
            )

    def sample_mode(self) -> None:
        """Set the model to sample mode

        Returns
        -------
        None
        
        """
        self.sample = True

    def training_mode(self) -> None:
        """Set the model to train mode

        Returns
        -------
        None
        
        """
        self.sample = False
        
