import torch
import torch.nn.functional as F

from typing import Dict, Optional
from agedi.data import AtomsGraph
from agedi.diffusion.noisers.sde import SDENoiser
from agedi.diffusion.sdes import SDE, VE
from agedi.diffusion.distributions import NoiseSampler, Prior, StandardNormal, UniformCell
from agedi.diffusion.distributions.normal import WrappedNormal
from agedi.diffusion.sdes.noise_schedules import Exponential
from agedi.utils import OFFSET_LIST


class Fractional(SDENoiser):
    """Implements noising of atoms positions in fractional coordinates.

    Parameters
    ----------
    sde_class : SDE
        The class of the SDE to be used for the noising.
    sde_kwargs : Dict
        The keyword arguments to be passed to the SDE class.
    distribution : NoiseSampler
        The noise sampler to be used for the noise.
    prior : Prior
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
        sde_class: SDE = VE,
        sde_kwargs: Optional[Dict] = {"noise_schedule": Exponential},
        distribution: NoiseSampler = WrappedNormal(),
        prior: Prior = UniformCell(),
        sde: Optional[SDE] = None,
        **kwargs
    ) -> None:
        """Initialize the fractional positions noiser.

        Parameters
        ----------
        sde_class : SDE, optional
            Class of the SDE to use.  Defaults to :class:`~agedi.diffusion.sdes.VE`.
        sde_kwargs : dict, optional
            Keyword arguments forwarded to *sde_class*.
        distribution : NoiseSampler, optional
            Noise sampler used during noising and denoising.
            Defaults to :class:`~agedi.diffusion.distributions.normal.WrappedNormal`.
        prior : Prior, optional
            Prior distribution used to sample starting positions.
            Defaults to :class:`~agedi.diffusion.distributions.UniformCell`.
        sde : SDE, optional
            Pre-instantiated SDE object.  When provided, *sde_class* and
            *sde_kwargs* are ignored.
        **kwargs
            Additional keyword arguments forwarded to :class:`~agedi.diffusion.noisers.sde.SDENoiser`.
        """
        super().__init__(sde_class, sde_kwargs, distribution, prior, sde, **kwargs)

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
        sigmas = self.sde.sigma(t)
        sigmas_norm = self.distribution.sigma_norm(sigmas)
        

        # mean is 1
        frac_coords = batch.frac
        noise_coords = torch.randn_like(frac_coords)

        # NEW IMPLEMENTATION
        target_coords = self.distribution.d_log_p(sigmas * noise_coords, sigmas)/torch.sqrt(sigmas_norm)  # [B_n, 1]

        batch[self.key + "_target"] = target_coords
        batch[self.key + "_noise"] = sigmas * noise_coords
        
        x_t_coords = (frac_coords + sigmas*noise_coords) % 1.0

        batch.frac = x_t_coords
        
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
        r = batch.frac
        sigmas = self.sde.noise_schedule.f(t)
        sigmas_norm = self.distribution.sigma_norm(sigmas)
        pred = batch["pos_score"]

        r_score = pred * torch.sqrt(sigmas_norm)
        drift = self.sde.drift(r, t)
        diffusion = self.sde.diffusion(t)

        w = torch.randn_like(r)
        if last:
            new_pos = r + delta_t * (diffusion**2 * r_score + drift) 
        else:
            new_pos = r + delta_t * (diffusion**2 * r_score + drift) + torch.sqrt(delta_t) * diffusion * w
            
        new_pos = new_pos % 1.0
        batch.frac = new_pos

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
        sigmas = self.sde.sigma(t)
        sigmas_norm = self.distribution.sigma_norm(sigmas)
        r_score = batch["pos_score"]
        r_target = batch[self.key + "_target"]
        loss_coords = F.mse_loss(r_score, r_target)

        loss = loss_coords

        return loss

