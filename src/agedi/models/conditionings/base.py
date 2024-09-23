from abc import ABC, abstractmethod

import torch


class Conditioning(ABC, torch.nn.Module):
    """
    Conditioning Base Class

    Args:

    property: str
        The property of the batch to condition on
    input_dim: int
        The dimension of the input conditioning
    output_dim: int
        The dimension of the output conditioning
    concatenation_type: str
        The type of concatenation to use. Default is "scalar"


    """

    def __init__(
        self,
        property: str,
        input_dim: int,
        output_dim: int,
        concatenation_type: str = "scalar",
        **kwargs,
    ) -> None:
        """
        Constructor for the Conditioning class

        """
        super().__init__(**kwargs)
        self.property = property
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.concatenation_type = concatenation_type

    @abstractmethod
    def get_conditioning(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to get the conditioning from the input

        Args:
        x: torch.Tensor
            The input tensor

        Returns:
        torch.Tensor
            The conditioning tensor
        """
        pass

    def forward(self, batch: "AtomsGraph") -> "AtomsGraph":
        """
        Forward method to get the conditioning from the input

        Args:
        batch: AtomsGraph
            The input batch

        Returns:
        AtomsGraph
            The batch with the conditioning added to the representation
        
        """
        c = self.get_conditioning(batch[self.property])

        self.concatenate(batch, c)

        return batch

    def concatenate(self, batch: "AtomsGraph", c: torch.Tensor) -> None:
        """
        Concatenate the conditioning to the batch

        Args:
        batch: AtomsGraph
            The input batch
        c: torch.Tensor
            The conditioning tensor

        Returns:
        None
        """
        if self.concatenation_type == "scalar":
            rep = batch.representation
            scalar = rep.scalar

            if scalar.shape[0] != c.shape[0]:
                raise ValueError(
                    "Scalar and conditioning have different number of nodes"
                )

            new_scalar = torch.cat((scalar, c), dim=1)
            rep.scalar = new_scalar
            batch.representation = rep

        else:
            raise ValueError(
                f"Concatenation type {self.concatenation_type} not supported"
            )
