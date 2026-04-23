import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

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
    
    def __init__(self, score_clip: Optional[float] = None, **kwargs) -> None:
        """Initializes the head with the key.

        Parameters
        ----------
        score_clip : float, optional
            If provided, the score output is clamped to ``[-score_clip, score_clip]``.
        **kwargs
            Additional keyword arguments forwarded to :class:`torch.nn.Module`.
        """
        super(Head, self).__init__(**kwargs)
        self._score_clip = score_clip
        

    @property
    def key(self) -> str:
        """The key of the attribute to be noised and denoised.

        """
        return self._key

    def get_hparams(self) -> Dict:
        """Return hyperparameters sufficient to reconstruct this head.

        Returns a dictionary with a ``_target_`` key (the fully-qualified class
        name) plus ``score_clip`` from the base class.  Subclasses should call
        ``super().get_hparams()`` and merge in their own constructor parameters.

        Returns
        -------
        dict
            Hyperparameter dictionary.
        """
        return {
            "_target_": f"{type(self).__module__}.{type(self).__qualname__}",
            "score_clip": self._score_clip,
        }

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
        if self._score_clip is not None:
            out = torch.clamp(out, -self._score_clip, self._score_clip)
        
        return out
        
    @abstractmethod
    def _score(self, translated_batch: Any) -> torch.Tensor:
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
