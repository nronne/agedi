from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Optional, Union

from agedi.diffusion.distributions import Distribution
from agedi.data import AtomsGraph

import torch


class Noiser(ABC, torch.nn.Module):
    """Noiser Base class

    Impments a noiser that can noise and denoise a atomistic structure attribute.

    Parameters
    ----------
    distribution: Distribution
        The distribution to be used for the noising.
    prior: Distribution
        The prior to be used for the denoising.
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
        distribution: Distribution,
        prior: Distribution,
        loss_scaling: float = 1.0,
        key: Optional[str] = None,
        **kwargs
    ):
        """Initializes the Noiser."""
        super().__init__(**kwargs)

        if key is not None:
            self._key = key

        self.distribution = distribution
        self.distribution.key = self.key

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
    def _denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
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

    def denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoises the attribute of the atomistic structure.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be denoised.
        delta_t: float
            The time step to be used for the denoising.
        last: bool
            If the denoising is the last step of the denoising.

        Returns
        -------
        AtomsGraph
            The denoised atomistic structure (or bach hereof).

        """
        return self._denoise(batch, delta_t, last)

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

    def langevin_step(
        self,
        batch: AtomsGraph,
        step_size: Union[float, torch.Tensor] = 0.01,
    ) -> AtomsGraph:
        """Perform a Langevin corrector step at the current (constant) time.

        Applies a small score-corrected Langevin update without advancing the
        diffusion time.  The score must already be stored in
        ``batch[key + "_score"]`` (i.e. the score model must have been called
        before invoking this method).

        The default implementation delegates to :meth:`_denoise` with
        ``last=False`` and a fixed *step_size*.  Subclasses may override this
        for a more specialised corrector.

        Parameters
        ----------
        batch : AtomsGraph
            The atomistic structure (or batch hereof) to be corrected.
        step_size : float or torch.Tensor, optional
            Size of the Langevin corrector step.  Passing a pre-created
            :class:`torch.Tensor` avoids repeated tensor allocation when this
            method is called in a tight loop.  Defaults to ``0.01``.

        Returns
        -------
        AtomsGraph
            The corrected atomistic structure.
        """
        if not isinstance(step_size, torch.Tensor):
            step_size = torch.tensor(
                step_size, dtype=batch.time.dtype, device=batch.time.device
            )
        return self._denoise(batch, step_size, last=False)

    def initialize_graph(self, batch: AtomsGraph) -> None:
        """Initializes the graph with the prior distribution.

        Can be overwritten by subclasses for specific initializations.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised and denoised.

        """
        setattr(
            batch,
            self.key,
            self.prior.get_callable(batch)(),
        )

    def forward_marginal(self, batch: AtomsGraph, ref: torch.Tensor) -> torch.Tensor:
        """Sample the forward marginal ``q(z_t | z_0=ref)`` at ``batch.time``.

        Used by inpainting to re-noise the *known* (context) atoms to the
        current diffusion time, so that a partially denoised structure with a
        mix of generated and known atoms is always self-consistent: every
        atom sits at the noise level the score model expects for ``t``.

        Must be implemented by subclasses that support inpainting.  The base
        implementation raises :class:`NotImplementedError` so that using an
        unsupported noiser with :meth:`~agedi.diffusion.Diffusion.inpaint`
        fails loudly rather than silently leaving an attribute un-noised.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof); ``batch.time`` gives
            the target time ``t``.
        ref: torch.Tensor
            The clean reference value (``z_0``) to condition on, e.g. the
            input structure's positions or atomic numbers.

        Returns
        -------
        torch.Tensor
            A sample of ``z_t`` for every atom (not masked by ``inpaint_mask``
            here; the caller selects which atoms to keep).

        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement forward_marginal(), "
            "required for inpainting."
        )

    def renoise(
        self,
        batch: AtomsGraph,
        current: torch.Tensor,
        t_from: torch.Tensor,
        t_to: torch.Tensor,
    ) -> torch.Tensor:
        """Forward step from an already-noised state, for RePaint resampling.

        Advances ``current`` (the state at time ``t_from``) forward to
        ``t_to`` (``t_to >= t_from``) by sampling the SDE's forward
        transition kernel. Used for the RePaint-style "jump back" step that
        re-noises a partially denoised structure before repeating a reverse
        step, letting the known and generated atoms harmonize.

        Must be implemented by subclasses that support inpainting resampling.
        The base implementation raises :class:`NotImplementedError`.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof).
        current: torch.Tensor
            The current (partially denoised) value at time ``t_from``.
        t_from: torch.Tensor
            The current time.
        t_to: torch.Tensor
            The target time to re-noise to (``t_to >= t_from``).

        Returns
        -------
        torch.Tensor
            A sample of the state at time ``t_to``.

        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement renoise(), "
            "required for inpainting resampling (n_resample > 1)."
        )
