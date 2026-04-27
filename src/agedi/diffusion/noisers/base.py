from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Optional, Union

from agedi.diffusion.distributions import Distribution, Prior, NoiseSampler
from agedi.data import AtomsGraph

import torch


class Noiser(ABC, torch.nn.Module):
    """Noiser Base class

    Implements a noiser that can noise and denoise a atomistic structure attribute.

    Parameters
    ----------
    distribution: NoiseSampler
        The noise sampler used during the forward and reverse diffusion steps.
        Accepts any :class:`~agedi.diffusion.distributions.NoiseSampler` (or,
        for backward compatibility, a plain :class:`~agedi.diffusion.distributions.Distribution`).
    prior: Prior
        The prior distribution used to initialise the noised state at the
        start of the reverse trajectory.
        Accepts any :class:`~agedi.diffusion.distributions.Prior` (or, for
        backward compatibility, a plain :class:`~agedi.diffusion.distributions.Distribution`).
    loss_scaling: float
        Scaling factor applied to this noiser's loss contribution.
    key: str, optional
        Override the class-level ``_key`` for the attribute to be noised and
        denoised.  Useful for reusing a noiser class on a different attribute
        without subclassing purely to change ``_key``.

    Returns
    -------
    Noiser

    """

    _key: str
    _registry: ClassVar[Dict[str, "Callable[..., Noiser]"]] = {}  # type: ignore[name-defined]

    def __init__(
        self,
        distribution: Union[NoiseSampler, Distribution],
        prior: Union[Prior, Distribution],
        loss_scaling: float = 1.0,
        key: Optional[str] = None,
        **kwargs
    ):
        """Initializes the Noiser."""
        super().__init__(**kwargs)

        if key is not None:
            self._key = key

        self.distribution = distribution
        self.prior = prior
        self.prior.key = self.key

        self.loss_scaling = loss_scaling

    @classmethod
    def register(cls, name: str, factory: "Callable[..., Noiser]") -> None:  # type: ignore[name-defined]
        """Register a noiser factory callable under *name*.

        The factory is called with ``sde`` as a keyword argument containing the
        resolved :class:`~agedi.diffusion.sdes.SDE` instance.  Noisers that do
        not use an SDE can safely ignore it via ``**kwargs``.

        Parameters
        ----------
        name : str
            Alias string used to reference the noiser in
            :func:`~agedi.functional.create_diffusion`.
        factory : Callable
            A callable that accepts ``sde`` as a keyword argument and returns a
            :class:`Noiser` instance.

        Examples
        --------
        Register a custom noiser so it can be referenced by its alias::

            from agedi.diffusion.noisers import Noiser

            class MyNoiser(Noiser):
                ...

            Noiser.register("my_noiser", lambda sde: MyNoiser(sde=sde))
        """
        cls._registry[name] = factory

    def get_hparams(self) -> Dict:
        """Return hyperparameters sufficient to reconstruct this noiser.

        Returns a dictionary with a ``_target_`` key (the fully-qualified class
        name) plus ``distribution``, ``prior``, and ``loss_scaling`` entries
        taken from the base class.  Subclasses should call
        ``super().get_hparams()`` and merge in their own constructor parameters.

        Returns
        -------
        dict
            Hyperparameter dictionary.
        """
        return {
            "_target_": f"{type(self).__module__}.{type(self).__qualname__}",
            "distribution": self.distribution.get_hparams(),
            "prior": self.prior.get_hparams(),
            "loss_scaling": self.loss_scaling,
        }

    @property
    def key(self) -> str:
        """The key of the attribute to be noised and denoised."""
        return self._key

    @abstractmethod
    def noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Noise the attribute of the atomistic structure.

        Must be implemented by the subclass.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch thereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or batch thereof).

        """
        pass

    @abstractmethod
    def denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoise the attribute of the atomistic structure.

        Must be implemented by the subclass.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch thereof) to be denoised.

        delta_t: float
            The time step to be used for the denoising.

        last: bool
            Whether this is the final denoising step.

        Returns
        -------
        AtomsGraph
            The denoised atomistic structure (or batch thereof).

        """
        pass

    @abstractmethod
    def loss(self, batch: AtomsGraph) -> float:
        """Compute the training loss.

        Must be implemented by the subclass.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch thereof) to be noised and denoised.

        Returns
        -------
        float
            The loss of the noised and denoised atomistic structure.

        """
        pass

    def initialize_graph(self, batch: AtomsGraph) -> None:
        """Initialise the graph attribute from the prior distribution.

        Can be overwritten by subclasses for specific initializations.

        Uses :meth:`~agedi.diffusion.distributions.Prior.sample` when the
        prior is a :class:`~agedi.diffusion.distributions.Prior`, falling back
        to the legacy ``get_callable`` interface for backward compatibility.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch thereof) to be initialised.

        """
        if isinstance(self.prior, Prior):
            value = self.prior.sample(batch)
        else:
            value = self.prior.get_callable(batch)()
        setattr(batch, self.key, value)

