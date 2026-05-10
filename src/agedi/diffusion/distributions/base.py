from abc import ABC, abstractmethod
from typing import Dict, Optional

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

    def __init__(self, key: Optional[str] = None, **kwargs):
        """Initialize the distribution"""
        self.key = key
        self._last_noise: Optional[torch.Tensor] = None

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
    def sample(self, batch: AtomsGraph, **kwargs) -> torch.Tensor:
        """Sample from the distribution.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data.
        **kwargs
            Distribution-specific parameters (e.g. ``mu``, ``sigma``,
            ``probs``).

        Returns
        -------
        torch.Tensor
            Sampled tensor.
        """
        pass


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

    @abstractmethod
    def sample(self, batch: AtomsGraph, **kwargs) -> torch.Tensor:
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
        pass


class NoiseDistribution(Distribution):
    """Abstract base class for noise (step) distributions.

    A ``NoiseDistribution`` is used during the forward and reverse diffusion steps.
    Given a mean ``mu`` and scale ``sigma``, :meth:`sample` returns the full noised
    sample ``x_t = mu + sigma * w`` where ``w`` is a unit-scale noise draw.
    The raw noise ``w`` is cached after each :meth:`sample` call and can be
    retrieved via :meth:`last_noise`.

    Concrete subclasses include :class:`~agedi.diffusion.distributions.Normal`,
    :class:`~agedi.diffusion.distributions.TruncatedNormal`,
    :class:`~agedi.diffusion.distributions.WrappedNormal`, and
    :class:`~agedi.diffusion.distributions.Categorical`.
    """

    @abstractmethod
    def sample(self, batch: AtomsGraph, **kwargs) -> torch.Tensor:
        """Sample the noised value ``x_t = mu + sigma * w``.

        Implementations must cache the unit-scale noise ``w`` so that
        :meth:`last_noise` returns the ``w`` corresponding to the most recent
        :meth:`sample` call.

        Parameters
        ----------
        batch : AtomsGraph
            Batch of atomistic data (may be used by subclasses that need
            geometry-dependent parameters such as confinement bounds).
        **kwargs
            Distribution-specific parameters (e.g. ``mu``, ``sigma``,
            ``probs``).

        Returns
        -------
        torch.Tensor
            Full noised sample ``x_t = mu + sigma * w``.
        """
        pass

    def last_noise(self) -> Optional[torch.Tensor]:
        """Return the unit-scale noise ``w`` from the most recent :meth:`sample` call.

        The returned tensor satisfies ``x_t = mu + sigma * w`` where ``x_t``
        was the value returned by the preceding :meth:`sample` call.  Returns
        ``None`` if :meth:`sample` has not been called yet, or if the
        distribution does not produce a separable noise term (e.g.
        :class:`~agedi.diffusion.distributions.Categorical`).

        Returns
        -------
        torch.Tensor or None
            Cached unit-scale noise ``w``, or ``None``.
        """
        return self._last_noise

