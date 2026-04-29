import torch
import torch.nn.functional as F

from typing import Optional
from agedi.data import AtomsGraph
from agedi.diffusion.noisers.sde import SDENoiser
from agedi.diffusion.sdes import SDE, VE
from agedi.diffusion.distributions import NoiseDistribution, PriorDistribution, StandardNormal, UniformCell
from agedi.diffusion.distributions.normal import WrappedNormal
from agedi.diffusion.sdes.noise_schedules import Exponential
from agedi.utils import OFFSET_LIST


class Fractional(SDENoiser):
    """Implements noising of atoms positions in fractional coordinates.

    Parameters
    ----------
    sde : SDE, optional
        An already-instantiated SDE object that defines the diffusion style.
        Defaults to :class:`~agedi.diffusion.sdes.VE` with an
        :class:`~agedi.diffusion.sdes.noise_schedules.Exponential` schedule.
    distribution : NoiseDistribution
        The noise sampler to be used for the noise.
    prior : PriorDistribution
        The prior distribution to be used for the noise.
    key : str
        The key to be used for the noising.
    **kwargs
        Additional keyword arguments to be passed to the Noiser class.

    Returns
    -------
    Noiser
        The noiser for the atoms positions in Cartesian coordinates.

    """

    _key = "frac"

    def __init__(
        self,
        sde: Optional[SDE] = None,
        distribution: NoiseDistribution = WrappedNormal(),
        prior: PriorDistribution = UniformCell(),
        **kwargs
    ) -> None:
        """Initialize the fractional positions noiser.

        Parameters
        ----------
        sde : SDE, optional
            Instantiated SDE object.  Defaults to
            :class:`~agedi.diffusion.sdes.VE` with an
            :class:`~agedi.diffusion.sdes.noise_schedules.Exponential` schedule.
        distribution : NoiseDistribution, optional
            Noise sampler used during noising and denoising.
            Defaults to :class:`~agedi.diffusion.distributions.normal.WrappedNormal`.
        prior : PriorDistribution, optional
            Prior distribution used to sample starting positions.
            Defaults to :class:`~agedi.diffusion.distributions.UniformCell`.
        **kwargs
            Additional keyword arguments forwarded to :class:`~agedi.diffusion.noisers.sde.SDENoiser`.
        """
        if sde is None:
            sde = VE(noise_schedule=Exponential)
        super().__init__(sde=sde, distribution=distribution, prior=prior, **kwargs)

    def noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Add noise to the fractional atom coordinates.

        Added noise is stored in ``frac_noise`` and the training target in
        ``frac_target``.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch thereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or batch thereof).

        """
        t = batch.time
        frac = batch.frac
        
        sigmas = torch.sqrt(self.sde.var(t)) #self.sde.sigma(t)
        
        sigmas_norm = self.distribution.sigma_norm(sigmas)
        frac = self.distribution.sample(batch, mu=frac, sigma=sigmas)
        noise = self.distribution.last_noise()
        
        loss_target = self.distribution.d_log_p(sigmas*noise, sigmas)/torch.sqrt(sigmas_norm)  # [B_n, 1]

        batch[self.key + "_target"] = loss_target
        batch[self.key + "_noise"] = noise#/sigmas


        batch.frac = frac# + noise
        return batch

    def denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoise the fractional atom coordinates using Euler-Maruyama.

        The update rule is:

        .. math::

            R_{i+1} = R_i + \\Delta t (f(R_i, t) + g(t)^2 s(R_i, t))
                      + \\sqrt{\\Delta t} g(t) w

        The score is expected to be stored in ``pos_score``.

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
        t = batch.time        
        frac = batch.frac
        pred = batch.pos_score
        
        sigmas = torch.sqrt(self.sde.var(t)) #self.sde.noise_schedule.f(t)
        
        sigmas_norm = self.distribution.sigma_norm(sigmas)
        frac_score = pred * torch.sqrt(sigmas_norm)


        drift = self.sde.drift(frac, t)
        diffusion = self.sde.diffusion(t)

        if last:
            new_frac = frac + delta_t * (diffusion**2 * frac_score + drift) 
        else:
            move = self.distribution.sample(
                batch,
                mu=delta_t * (diffusion**2 * frac_score + drift),
                sigma=torch.sqrt(delta_t) * diffusion)
            
            new_frac = frac + move
            
        batch.frac = new_frac

        return batch

    def loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Compute the fractional noiser loss.

        Expects the training target in ``frac_target`` and the predicted score in
        ``pos_score``.

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
        # sigmas = torch.sqrt(self.sde.var(t)) #self.sde.sigma(t)
        # sigmas_norm = self.distribution.sigma_norm(sigmas)
        
        score = batch.pos_score
        target = batch[self.key + "_target"]

        loss = F.mse_loss(score, target)

        return loss

