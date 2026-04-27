from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

import torch

from agedi.data import AtomsGraph


class Distribution(ABC):
    """Base Class for noise distributions

    Parameters
    ----------
    key : str
        Key to identify the property from the batch

    Returns
    -------
    Distribution

    """

    def __init__(self, key:Optional[str] = None, **kwargs):
        """Initialize the distribution"""
        self.key = key

    def get_hparams(self) -> Dict:
        """Return hyperparameters sufficient to reconstruct this distribution.

        Returns a dictionary with a ``_target_`` key (the fully-qualified class
        name) plus any constructor arguments stored on the base class.
        Subclasses should call ``super().get_hparams()`` and merge in their
        own parameters.

        Returns
        -------
        dict
            Hyperparameter dictionary.
        """
        return {"_target_": f"{type(self).__module__}.{type(self).__qualname__}"}

    @abstractmethod
    def _sample(self, **kwargs) -> torch.Tensor:
        """Sample distribution

        Sample from the distribution and return tensor of shape self.key

        Parameters
        ----------
        kwargs : dict
            The parameters of the distribution

        Returns
        -------
        torch.Tensor
            Sampled tensor
        """
        pass

    def _setup(self, batch: AtomsGraph) -> None:
        """Prepare distribution

        Prepare the distribution for sampling of the batch

        Parameters
        ----------
        batch : AtomsGraph
            Batch of data

        Returns
        -------
        None

        """
        pass

    def get_callable(self, batch: AtomsGraph) -> Callable:
        """Get callable function

        Return a callable function that samples from the distribution

        Parameters
        ----------
        batch : AtomsGraph
            Batch of data

        Returns
        -------
        Callable
            Callable function that samples from the distribution

        """
        self._setup(batch)

        def _sampler(*args, **kwargs):
            """Call the distribution's ``_sample`` method with the provided arguments."""
            return self._sample(*args, **kwargs)

        return _sampler


class PriorDistribution(Distribution):
    """Abstract base class for prior distributions.

    A ``PriorDistribution`` is used to initialise the noised state at the start of the
    reverse (generative) trajectory.  It knows only about the :class:`~agedi.data.AtomsGraph`
    batch and returns a complete tensor that is assigned to a graph attribute.

    Concrete subclasses include :class:`~agedi.diffusion.distributions.UniformCell`,
    :class:`~agedi.diffusion.distributions.UniformCellConfined`,
    :class:`~agedi.diffusion.distributions.StandardNormal`, and
    :class:`~agedi.diffusion.distributions.Constant`.
    """

    def sample(self, batch: AtomsGraph) -> torch.Tensor:
        """Sample the initial state from the prior.

        Parameters
        ----------
        batch : AtomsGraph
            The atomistic graph (or batch thereof) used to determine the shape
            and any geometry-dependent parameters (e.g. unit-cell vectors).

        Returns
        -------
        torch.Tensor
            Initial state tensor, ready to be assigned to a graph attribute.
        """
        self._setup(batch)
        return self._sample()


class NoiseDistribution(Distribution):
    """Abstract base class for noise (step) distributions.

    A ``NoiseDistribution`` is used during the forward and reverse diffusion steps.
    Given a current state ``mu`` and a scale ``sigma``, it returns a perturbed
    sample.  Unlike :class:`PriorDistribution`, it does not need to know about graph
    geometry, only the current diffusion step parameters.

    Concrete subclasses include :class:`~agedi.diffusion.distributions.Normal`,
    :class:`~agedi.diffusion.distributions.TruncatedNormal`,
    :class:`~agedi.diffusion.distributions.WrappedNormal`, and
    :class:`~agedi.diffusion.distributions.Categorical`.
    """

