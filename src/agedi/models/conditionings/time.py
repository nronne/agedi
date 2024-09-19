import torch

from .base import Conditioning


class TimeConditioning(Conditioning):
    """
    Condition the model on the time t.

    Args:
        t (torch.Tensor): Time tensor of shape (Nodes, 1).

    """
    def __init__(self, **kwargs):
        super().__init__(
            property="time",
            input_dim=1,
            output_dim=2,
            concatenation_type="scalar",
            **kwargs
        )
        
        self.omega = torch.pi

    def get_conditioning(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get the conditioning tensor for the time t.

        ::math::
            \begin{align*}
            \mathbf{c} = \begin{bmatrix} \sin(\omega t) \\ \cos(\omega t) \end{bmatrix}
            \end{align*}

        Args:
            t (torch.Tensor): Time tensor of shape (Nodes, 1).

        Returns:
            torch.Tensor: Conditioning tensor of shape (Nodes, 2).

        """
        c = torch.cat(
            (torch.sin(self.omega * t), torch.cos(self.omega * t)), dim=-1
        ).unsqueeze(-1)

        return c
