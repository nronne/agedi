import torch
import torch.nn.functional as F

from typing import Dict
from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser
from agedi.diffusion.sdes import SDE, VE
from agedi.diffusion.distributions import Distribution, StandardNormal, UniformCell
from agedi.utils import OFFSET_LIST


class FractionalNoiser(Noiser):
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
        sde_kwargs: Dict = {},
        distribution: Distribution = WrappedNormal(),
        prior: Distribution = UniformCell(),
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
        r0 = batch[self.key]
        t = batch.time

        w = self.distribution.get_callable(batch)

        rt = self.sde.transition_kernel(r0, t, w)
        noise = self.sde.noise(r0, rt, t)
        rt = rt % 1.0

        setattr(batch, self.key, rt)
        batch[self.key + "_noise"] = batch.apply_mask(noise)
        
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
        r = batch[self.key]
        r_score = batch[self.key + "_score"]
        
        r_score[torch.isnan(r_score)] = 0.0
        t = batch.time

        drift = self.sde.drift(r, t)
        diffusion = self.sde.diffusion(t)

        w = self.distribution.get_callable(batch)
        
        if last:
            new_pos = r + delta_t * (diffusion**2 * r_score + drift) 
        else:
            new_pos = w(
                r + delta_t * (diffusion**2 * r_score + drift),  # mean
                torch.sqrt(delta_t) * diffusion,  # variance
            )
        new_pos = new_pos % 1.0
        
        setattr(batch, self.key, new_pos)

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
        r_score = batch[self.key + "_score"]
        r_noise = batch[self.key + "_noise"]

        var = self.sde.var(t)
        sigma = torch.sqrt(var)
        sigma_norm = self.distribution.sigma_norm(sigma)

        r_score = batch.apply_mask(r_score)

        lt = 1.0  # /var.sqrt()


        r_target = self.distribution.d_log_p(sigma*r_noise, sigma) / sigma_norm
        
        loss = F.mse_loss(r_score, r_target)
        
        return loss

