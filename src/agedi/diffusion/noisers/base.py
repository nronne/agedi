from abc import ABC, abstractmethod
from typing import Type, Dict

from .sdes import SDE
from .distributions import Distribution
from agedi.data import AtomsGraph

import torch


class Noiser(ABC, torch.nn.Module):
    """Noiser Base class

    Impments a noiser that can noise and denoise a atomistic structure attribute.

    Parameters
    ----------
    sde_class: Type[SDE]
        The class of the SDE to be used for the noising and denoising.
    sde_kwargs: dict
        The keyword arguments to be passed to the SDE class.
    distribution: Distribution
        The distribution to be used for the noising.
    prior: Distribution
        The prior to be used for the denoising.
    key: str
        The key of the attribute to be noised and denoised.

    Returns
    -------
    Noiser

    """
    _key: str
    
    def __init__(
        self,
        sde_class: Type[SDE],
        sde_kwargs: Dict,
        distribution: Distribution,
        prior: Distribution,
        **kwargs
    ):
        """Initializes the Noiser.
        
        """
        super().__init__(**kwargs)
        self.sde = sde_class(**sde_kwargs)
        
        self.distribution = distribution
        self.distribution.key = self.key

        self.prior = prior
        self.prior.key = self.key
        

    @property
    def key(self) -> str:
        """The key of the attribute to be noised and denoised.

        """
        return self._key

    
    @abstractmethod
    def _noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Noises the attribute of the atomistic structure.

        Must be implemented by the subclass.
        
        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or bach hereof).
        
        """
        pass

    @abstractmethod
    def _denoise(self, batch: AtomsGraph, delta_t: float) -> AtomsGraph:
        """Denoises the attribute of the atomistic structure.

        Must be implemented by the subclass.
        
        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be denoised.

        delta_t: float
            The time step to be used for the denoising.

        Returns
        -------
        AtomsGraph
            The denoised atomistic structure (or bach hereof).
        
        """
        pass

    @abstractmethod
    def _loss(self, batch: AtomsGraph) -> float:
        """Computes the training loss.

        Must be implemented by the subclass.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised and denoised.

        Returns
        -------
        float
            The loss of the noised and denoised atomistic structure.

        """
        pass

    def noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Noises the attribute of the atomistic structure.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or bach hereof).
        
        """
        return self._noise(batch)

    def denoise(self, batch: AtomsGraph, delta_t: float) -> AtomsGraph:
        """Denoises the attribute of the atomistic structure.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be denoised.
        delta_t: float
            The time step to be used for the denoising.

        Returns
        -------
        AtomsGraph
            The denoised atomistic structure (or bach hereof).
        
        """
        return self._denoise(batch, delta_t)

    def loss(self, batch: AtomsGraph) -> float:
        """Compute the training loss.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised and denoised.

        Returns
        -------
        float
            The loss of the noised and denoised atomistic structure.

        """
        return self._loss(batch)
    
