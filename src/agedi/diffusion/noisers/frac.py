import torch
import torch.nn.functional as F

from typing import Dict, Optional
from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser
from agedi.diffusion.sdes import SDE, VE
from agedi.diffusion.distributions import Distribution, StandardNormal, UniformCell
from agedi.diffusion.distributions.normal import WrappedNormal
from agedi.diffusion.sdes.noise_schedules import Exponential
from agedi.utils import OFFSET_LIST


class Fractional(Noiser):
    """Implements noising of atoms positions in fractional coordinates.

    Parameters
    ----------
    sde_class : SDE
        The class of the SDE to be used for the noising.
    sde_kwargs : Dict
        The keyword arguments to be passed to the SDE class.
    distribution : Distribution
        The distribution to be used for the noise.
    prior : Distribution
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
        distribution: Distribution = WrappedNormal(),
        prior: Distribution = UniformCell(),
        sde: Optional[SDE] = None,
        **kwargs
    ) -> None:
        """Initialize the positions noiser.

        Parameters
        ----------
        sde_class : SDE, optional
            Class of the SDE to use.  Defaults to :class:`~agedi.diffusion.sdes.VE`.
        sde_kwargs : dict, optional
            Keyword arguments forwarded to *sde_class*.
        distribution : Distribution, optional
            Noise distribution used during noising and denoising.
            Defaults to :class:`~agedi.diffusion.distributions.Normal`.
        prior : Distribution, optional
            Prior distribution used to sample starting positions.
            Defaults to :class:`~agedi.diffusion.distributions.UniformCell`.
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
        

    def _noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Initializes the noise for the positions noiser.

        Added noise is stored in the self.key+"_noise", which by default is
        "positions_noise".

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or bach hereof).

        """
        t = batch.time
        sigmas = self.sde.noise_schedule.f(t)
        sigmas_norm = self.distribution.sigma_norm(sigmas)
        

        # mean is 1
        frac_coords = batch.frac
        noise_coords = torch.randn_like(frac_coords)

        # NEW IMPLEMENTATION
        target_coords = self.distribution.d_log_p(sigmas * noise_coords, sigmas) / torch.sqrt(sigmas_norm)  # [B_n, 1]

        batch[self.key + "_target"] = target_coords
        batch[self.key + "_noise"] = sigmas * noise_coords
        
        x_t_coords = (frac_coords + sigmas*noise_coords) % 1.0

        batch.frac = x_t_coords
        
        return batch

    def _denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoises the positions of the atomistic structure.

        The denoising follows the Euler-Maruyama scheme.
        ::math::
        R_i+1 = R_i +
                \Delta t (f(R_i, t) + g(t)**2 * s(R_i, t)) +
                \sqrt{\Delta t} g(t) * w

        The used score is expected to be stored in the self.key+"_score",
        which by default is "pos_score".

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be denoised.
        delta_t: float
            The time step for the denoising.
        last: bool
            If the denoising is the last step of the denoising.

        Returns
        -------
        AtomsGraph
            The denoised atomistic structure (or bach hereof).

        """
        t = batch.time        
        r = batch.frac
        # sigmas = self.sde.var(t)
        sigmas = self.sde.noise_schedule.f(t)
        sigmas_norm = self.distribution.sigma_norm(sigmas)
        pred = batch["pos_score"]


        # NEW IMPLEMENTATION
        std = torch.sqrt(sigmas)
        if last:
            w = torch.zeros_like(r)
        else:
            w = torch.randn_like(r)
        pred = pred * torch.sqrt(sigmas_norm)

        new_pos = r + delta_t * pred + torch.sqrt(delta_t) * std * w

        # # OLD
        # r_score = pred * torch.sqrt(sigmas_norm)
        # drift = self.sde.drift(r, t)
        # diffusion = self.sde.diffusion(t)

        # w = torch.randn_like(r)
        # if last:
        #     new_pos = r + delta_t * (diffusion**2 * r_score + drift) 
        # else:
        #     new_pos = r + delta_t * (diffusion**2 * r_score + drift) + torch.sqrt(delta_t) * diffusion * w
            

        new_pos = new_pos % 1.0
        batch.frac = new_pos

        return batch

    def _loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Compute the noiser loss.

        Computes the loss of the diffusion model for the positions noiser

        Expects the total added positions noise to be stored in the self.key+"_noise",
        which by default is "pos_noise" and the predicted score to be stored in the
        self.key+"_score", which by default is "pos_score".

        The loss is computed as
        ::math::
        L = \sum_i ||\sigma_t w_i + \sigma_t^2 s(R_i)||^2

        With the noise taking into account periodic boundary conditions.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised and denoised.

        Returns
        -------
        float
            The loss of the noised and denoised atomistic structure.

        """
        t = batch.time
        var = self.sde.var(t)
        r_score = batch["pos_score"]
        r_target = batch[self.key + "_target"]
        r_noise = batch[self.key + "_noise"]

        loss_coords = F.mse_loss(r_score, r_target)

        # loss_coords = torch.mean((r_score - r_target) ** 2)
        
        # loss_coords = torch.mean(
        #     torch.sum((r_noise + r_score * var) ** 2, dim=-1, keepdim=True)
        # )


        loss = loss_coords

        return loss

