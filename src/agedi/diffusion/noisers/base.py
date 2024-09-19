from abc import ABC, abstractmethod
from typing import Type, Dict

from .sdes import SDE
from .distributions import Distribution, Prior
# from agedi.data import "AtomsGraph"

import torch


class Noiser(ABC, torch.nn.Module):
    """
    Impments a noiser that can noise and denoise a atomistic structure attribute.

    Args:

    sde_class: Type[SDE]
        The class of the SDE to be used for the noising and denoising.
    sde_kwargs: dict
        The keyword arguments to be passed to the SDE class.
    distribution: Distribution
        The distribution to be used for the noising.
    prior: Prior
        The prior to be used for the denoising.
    key: str
        The key of the attribute to be noised and denoised.

    """
    def __init__(
        self,
        sde_class: Type[SDE],
        sde_kwargs: Dict,
        distribution: Distribution,
        prior: Prior,
        key: str,
        **kwargs
    ):
        """
        Initializes the Noiser.
        
        """
        super().__init__(**kwargs)
        self.sde = sde_class(**sde_kwargs)
        
        self.distribution = distribution
        self.distribution.key = key

        self.prior = prior
        self.prior.key = key
        
        self.key = key

        
    @abstractmethod
    def noise(self, batch: "AtomsGraph") -> "AtomsGraph":
        """
        Noises the attribute of the atomistic structure.

        Args:

        batch: "AtomsGraph"
            The atomistic structure (or batch hereof) to be noised.

        Returns:

        "AtomsGraph"
            The noised atomistic structure (or bach hereof).
        
        """
        pass

    @abstractmethod
    def denoise(self, batch: "AtomsGraph") -> "AtomsGraph":
        """
        Denoises the attribute of the atomistic structure.

        Args:

        batch: "AtomsGraph"
            The atomistic structure (or batch hereof) to be denoised.

        Returns:

        "AtomsGraph"
            The denoised atomistic structure (or bach hereof).
        
        """
        pass

    @abstractmethod
    def loss(self, batch: "AtomsGraph") -> float:
        """
        Computes the loss of the noised and denoised atomistic structure.

        Args:

        batch: "AtomsGraph"
            The atomistic structure (or batch hereof) to be noised and denoised.

        Returns:

        float
            The loss of the noised and denoised atomistic structure.

        """
        pass
