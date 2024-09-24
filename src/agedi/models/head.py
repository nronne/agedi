import torch
from abc import ABC, abstractmethod
from typing import Any

class Head(ABC,torch.nn.Module):
    """Abstract base class for any score model heads.

    The head is responsible for taking the translated batch with precalculated
    representation and returning a score tensor.

    The score tensor should have the same shape as the original tensor for the
    key of the head.

    Returns
    -------
    Head
    
    """
    _key: str
    
    def __init__(self, **kwargs) -> None:
        """Initializes the head with the key.
        """
        super(Head, self).__init__(**kwargs)
        

    @property
    def key(self) -> str:
        """The key of the attribute to be noised and denoised.

        """
        return self._key
        
    def forward(self, translated_batch: Any) -> torch.Tensor:
        """Forward pass of the head using a translated batch
        
        The output shape must match the either the positions (pos), types (x) or
        cell (cell) of the original batch.

        Parameters
        ----------
        translated_batch: Any
            The translated batch to be used in the forward pass

        Returns
        -------
        torch.Tensor
            The output of the forward pass. The shape of the tensor depends on the key of the head.
        
        """
        out = self._score(translated_batch)
        
        return out
        
    @abstractmethod
    def _score(self, translated_batch) -> torch.Tensor:
        """Abstract method for the forward pass of the head.

        Must be implemented by the subclass.
        
        Parameters
        ----------
        translated_batch: Any
            The translated batch to be used in the forward pass

        Returns
        -------
        torch.Tensor
            The output of the forward pass. The shape of the tensor depends on the key of the head.
        
        """
        pass
