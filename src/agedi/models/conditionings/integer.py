import torch
from typing import Dict
from .base import Conditioning

class IntegerConditioning(Conditioning):
    """Conditioning module for integer-valued properties.

    Embeds an integer property (e.g. number of atoms) into a fixed-size
    representation using :class:`torch.nn.Embedding`.
    """

    def __init__(self, max_int: int = 200, input_dim: int = 1, output_dim: int = 64, *args, **kwargs) -> None:
        """Initialize the integer conditioning module.

        Parameters
        ----------
        max_int : int, optional
            Maximum integer value supported by the embedding table.
        input_dim : int, optional
            Dimension of the integer input. Defaults to 1.
        output_dim : int, optional
            Dimension of the embedding output. Defaults to 64.
        *args
            Positional arguments forwarded to :class:`~agedi.models.conditionings.base.Conditioning`.
        **kwargs
            Keyword arguments forwarded to :class:`~agedi.models.conditionings.base.Conditioning`.
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, *args, **kwargs)
        self.max_int = max_int
        self.embedder = torch.nn.Embedding(max_int, self.output_dim)

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this integer conditioning module."""
        return {**super().get_hparams(), "max_int": self.max_int}


    def get_conditioning(self, x: torch.Tensor) -> torch.Tensor:
        """Get the conditioning tensor for x

        Parameters
        ----------
        x : torch.Tensor
            Time tensor of shape (Nodes, 1).

        Returns
        -------
        torch.Tensor
            Conditioning tensor of shape (Nodes, 2).

        """
        x = x.long()  # Ensure x is of type long for embedding
        c = self.embedder(x)

        return c

    def get_empty_conditioning(self, n: int) -> torch.Tensor:
        """Get an empty conditioning tensor.

        Returns
        -------
        torch.Tensor
            Empty conditioning tensor of shape (n, 2).

        """
        return torch.zeros(n, self.output_dim, device=self.device)


