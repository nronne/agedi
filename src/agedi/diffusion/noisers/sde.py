import torch

from abc import ABC, abstractmethod
from typing import Dict, Optional
from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser

from agedi.diffusion.sdes import SDE
from agedi.diffusion.distributions import Distribution, NoiseDistribution, PriorDistribution


class SDENoiser(Noiser, ABC):
    """Base class for SDE-backed noisers.

    Centralises the SDE wiring (``sde_class``/``sde_kwargs``/``sde`` pattern)
    and provides generic :meth:`noise`, :meth:`denoise`, and :meth:`loss`
    implementations that are suitable for many continuous-score noisers.
    Subclasses that need custom forward/reverse logic can override any of
    those three methods.

    Optional hooks :meth:`postprocess_score` and :meth:`postprocess_noise`
    can be overridden to apply per-noiser post-processing inside the generic
    ``loss`` implementation (default: identity).

    Parameters
    ----------
    sde_class : SDE
        The class of the SDE to be used for the noising.
    sde_kwargs : Dict
        The keyword arguments to be passed to the SDE class.
    distribution : NoiseDistribution
        The noise sampler to be used for the noise.
    prior : PriorDistribution
        The prior distribution to be used for the noise.
    sde : SDE, optional
        An already-instantiated SDE object.  When provided, *sde_class* and
        *sde_kwargs* are ignored.
    key : str
        The key to be used for the noising.
    **kwargs
        Additional keyword arguments to be passed to the Noiser class.

    Returns
    -------
    Noiser
        The noiser for the atoms positions in Cartesian coordinates.

    """

    _key = None

    def __init__(
        self,
        sde_class: SDE,
        sde_kwargs: Optional[Dict],
        distribution: NoiseDistribution,
        prior: PriorDistribution,
        sde: Optional[SDE] = None,
        **kwargs
    ) -> None:
        """Initialize the SDE noiser.

        Parameters
        ----------
        sde_class : SDE
            Class of the SDE to use for noising.  Ignored when *sde* is provided.
        sde_kwargs : dict, optional
            Keyword arguments forwarded to *sde_class*.  Ignored when *sde* is provided.
        distribution : NoiseDistribution
            Noise sampler used during noising and denoising.
        prior : PriorDistribution
            Prior distribution used to sample starting values.
        sde : SDE, optional
            Pre-instantiated SDE object.  When provided, *sde_class* and
            *sde_kwargs* are ignored.
        **kwargs
            Additional keyword arguments forwarded to :class:`~agedi.diffusion.noisers.Noiser`.
        """
        super().__init__(distribution, prior, **kwargs)
        if sde is not None:
            self.sde = sde
        else:
            if sde_kwargs is None:
                sde_kwargs = {}
            self.sde = sde_class(**sde_kwargs)

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this SDE noiser."""
        return {**super().get_hparams(), "sde": self.sde.get_hparams()}

    def postprocess_score(self, score: torch.Tensor) -> torch.Tensor:
        """Post-process the predicted score before computing the generic loss.

        The default implementation is the identity.  Override in subclasses
        that need e.g. masking or re-weighting.

        Parameters
        ----------
        score : torch.Tensor
            Raw predicted score tensor.

        Returns
        -------
        torch.Tensor
            Post-processed score tensor.
        """
        return score

    def postprocess_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Post-process the noise tensor before computing the generic loss.

        The default implementation is the identity.  Override in subclasses
        that need e.g. periodic corrections.

        Parameters
        ----------
        noise : torch.Tensor
            Raw noise tensor.

        Returns
        -------
        torch.Tensor
            Post-processed noise tensor.
        """
        return noise

    def noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Add noise to the atomistic structure.

        Added noise is stored in the ``self.key + "_noise"`` attribute.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch thereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or batch thereof).

        """
        z = batch[self.key]
        t = batch.time

        mean = self.sde.mean(t) * z
        sigma = torch.sqrt(self.sde.var(t))
        batch[self.key] = self.distribution.sample(batch, mu=mean, sigma=sigma)
        batch[self.key + "_noise"] = batch.apply_mask(self.sde.noise(z, batch[self.key], t))

        return batch

    def denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoise the atomistic structure using the Euler-Maruyama scheme.

        The update rule is:

        .. math::

            R_{i+1} = R_i + \\Delta t (f(R_i, t) + g(t)^2 s(R_i, t))
                      + \\sqrt{\\Delta t} g(t) w

        The score is expected to be stored in ``self.key + "_score"``.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch thereof) to be denoised.
        delta_t: float
            The time step for the denoising.
        last: bool
            Whether this is the final denoising step.

        Returns
        -------
        AtomsGraph
            The denoised atomistic structure (or batch thereof).

        """
        z = batch[self.key]
        z_score = batch[self.key + "_score"]
        t = batch.time

        drift = self.sde.drift(z, t)
        diffusion = self.sde.diffusion(t)

        if last:
            batch[self.key] = batch[self.key] + delta_t * (diffusion**2 * z_score + drift)
        else:
            mean = batch[self.key] + delta_t * (diffusion**2 * z_score + drift)
            sigma = torch.sqrt(delta_t) * diffusion
            batch[self.key] = self.distribution.sample(batch, mu=mean, sigma=sigma)

        return batch

    def loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Compute the noiser loss.

        The score-matching loss is:

        .. math::

            L = \\sum_i \\|\\sigma_t w_i + \\sigma_t^2 s(R_i)\\|^2

        The score is expected in ``self.key + "_score"`` and the noise in
        ``self.key + "_noise"``.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch thereof) to be noised and denoised.

        Returns
        -------
        torch.Tensor
            The loss of the noised and denoised atomistic structure.

        """
        t = batch.time
        z_score = batch[self.key + "_score"]
        z_noise = batch[self.key + "_noise"]

        var = self.sde.var(t)

        z_score = self.postprocess_score(z_score)
        z_noise = self.postprocess_noise(z_noise)

        lt = 1.0

        loss = torch.mean(
            lt * torch.sum((z_noise + z_score * var) ** 2, dim=-1, keepdim=True)
        )
        return loss

