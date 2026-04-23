import torch
from typing import Dict

from .base import Conditioning


class TimeConditioning(Conditioning):
    """Condition the model on the time t.

    Parameters
    ----------
    t : torch.Tensor
        Time tensor of shape (Nodes, 1).

    """
    def __init__(self, input_dim: int = 1, output_dim: int = 2, **kwargs) -> None:
        """Initialize the TimeConditioning class.

        Parameters
        ----------
        input_dim : int, optional
            Dimension of the time input. Defaults to 1.
        output_dim : int, optional
            Dimension of the sinusoidal output (sin + cos). Defaults to 2.
        **kwargs
            Keyword arguments forwarded to :class:`~agedi.models.conditionings.base.Conditioning`.
        """
        kwargs.pop("property", None)
        super().__init__(
            property="time",
            input_dim=input_dim,
            output_dim=output_dim,
            concatenation_type="scalar",
            **kwargs
        )
        
        self.omega = torch.pi

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this time conditioning module."""
        hparams = super().get_hparams()
        hparams.pop("property", None)
        return hparams

    def get_conditioning(self, t: torch.Tensor) -> torch.Tensor:
        """Get the conditioning tensor for the time t.

        ::math::
            \begin{align*}
            \mathbf{c} = \begin{bmatrix} \sin(\omega t) \\ \cos(\omega t) \end{bmatrix}
            \end{align*}

        Parameters
        ----------
        t : torch.Tensor
            Time tensor of shape (Nodes, 1).

        Returns
        -------
        torch.Tensor
            Conditioning tensor of shape (Nodes, 2).

        """
        c = torch.cat(
            (torch.sin(self.omega * t), torch.cos(self.omega * t)), dim=-1
        ).unsqueeze(-1)

        return c

    def get_empty_conditioning(self, n: int) -> torch.Tensor:
        """Get an empty conditioning tensor.

        Returns
        -------
        torch.Tensor
            Empty conditioning tensor of shape (1, 2).

        """
        return torch.zeros(n, self.output_dim, device=self.device)


    def forward(self, batch: "AtomsGraph", empty: bool=False) -> "AtomsGraph":
        """Forward method to get the conditioning from the input

        This ignores training and empty flags.

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
        c = self.get_conditioning(batch[self.property])

        self.concatenate(batch, c)

        return batch
    
