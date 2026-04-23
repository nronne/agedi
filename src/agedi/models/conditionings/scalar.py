import torch
from .base import Conditioning

class ScalarConditioning(Conditioning):
    """Conditioning module for continuous scalar properties.

    Projects a scalar property through a learned linear layer and encodes it
    with sinusoidal features (``cos`` and ``sin``), producing a 2-dimensional
    conditioning vector.
    """

    def __init__(self, *args, input_dim: int = 1, output_dim: int = 2, **kwargs) -> None:
        """Initialize the scalar conditioning module.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :class:`~agedi.models.conditionings.base.Conditioning`.
        input_dim : int, optional
            Keyword-only dimension of the scalar input. Defaults to 1.
        output_dim : int, optional
            Keyword-only dimension of the output conditioning (cos + sin). Defaults to 2.
        **kwargs
            Keyword arguments forwarded to :class:`~agedi.models.conditionings.base.Conditioning`.
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, *args, **kwargs)

        self.embedder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.input_dim),
        )

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
        x = x.view(-1, 1)
        c = self.embedder(x)
        c = torch.cat([torch.cos(c), torch.sin(c)], dim=-1)

        return c

    def get_empty_conditioning(self, n: int) -> torch.Tensor:
        """Get an empty conditioning tensor.

        Returns
        -------
        torch.Tensor
            Empty conditioning tensor of shape (n, 2).

        """
        return torch.zeros(n, self.output_dim, device=self.device)


