import torch
import numpy as np

from typing import Dict
from agedi.data import AtomsGraph
from agedi.diffusion.noisers.sde import SDENoiser
from agedi.diffusion.sdes import SDE, VP, VE
from agedi.diffusion.distributions import NoiseDistribution, PriorDistribution, Normal, StandardNormal


class CellNoiser(SDENoiser):
    """Implements noising of the cell.

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
    key : str
        The key to be used for the noising.
    **kwargs
        Additional keyword arguments to be passed to the Noiser class.

    Returns
    -------
    Noiser
        The noiser for the atoms positions in Cartesian coordinates.

    """

    _key = "cellpar"

    def __init__(
        self,
        sde_class: SDE = VE,
        sde_kwargs: Dict = {"sigma_max": 0.1},
        distribution: NoiseDistribution = Normal(),
        prior: PriorDistribution = Normal(),
        **kwargs
    ) -> None:
        super().__init__(sde_class, sde_kwargs, distribution, prior, **kwargs)

    def noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Add noise to the cell parameters.

        Added noise is stored in ``cellpar_noise``.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch thereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or batch thereof).

        """

        cellpar = getattr(batch, self.key)
        f = batch.frac.clone()
        t = batch.time[batch.ptr[:-1]].reshape(-1, 1)

        mean = self.sde.mean(t) * cellpar
        sigma = torch.sqrt(self.sde.var(t))
        noised_cellpar = self.distribution.sample(batch, mu=mean, sigma=sigma)


        a, b, c, alpha, beta, gamma, V = noised_cellpar.unbind(-1)
        
        noised_cellpar = torch.stack([a,b,c,alpha,beta,gamma,V], dim=-1)
        
        setattr(batch, self.key, noised_cellpar)
        batch.frac = f
        batch.add_batch_attr(self.key + "_noise", self.sde.noise(cellpar, noised_cellpar, t), type="graph")

        return batch

    def denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoise the cell parameters using the Euler-Maruyama scheme.

        ::math::
        R_i+1 = R_i +
                \Delta t (f(R_i, t) + g(t)**2 * s(R_i, t)) +
                \sqrt{\Delta t} g(t) * w

        The score is expected to be stored in ``cellpar_score``.

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
        c = getattr(batch, self.key)
        f = batch.frac.clone()
        c_score = batch[self.key + "_score"]
        if c_score.isnan().any():
            breakpoint()
            
        t = batch.time[batch.ptr[:-1]].reshape(-1, 1)

        drift = self.sde.drift(c, t)
        diffusion = self.sde.diffusion(t)

        if last:
            cellpar = c + delta_t * (diffusion**2 * c_score + drift)
        else:
            mean = c + delta_t * (diffusion**2 * c_score + drift)
            sigma = torch.sqrt(delta_t) * diffusion
            cellpar = self.distribution.sample(batch, mu=mean, sigma=sigma)


        a, b, c, alpha, beta, gamma, V = cellpar.unbind(-1)
        
        cellpar = torch.stack([a,b,c,alpha,beta,gamma,V], dim=-1)

        setattr(batch, self.key, cellpar)
        batch.frac = f

        return batch

    def loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Compute the cell noiser loss.

        Expects the noise in ``cellpar_noise`` and the predicted score in
        ``cellpar_score``.

        The loss is computed as:
        ::math::
        L = \sum_i ||\sigma_t w_i + \sigma_t^2 s(C_i)||^2

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch thereof) to be noised and denoised.

        Returns
        -------
        torch.Tensor
            The loss of the noised and denoised atomistic structure.

        """
        t = batch.time[batch.ptr[:-1]].reshape(-1, 1)
        c_score = batch[self.key + "_score"]
        c_noise = batch[self.key + "_noise"]

        var = self.sde.var(t)

        lt = 1.0

        loss = torch.mean(
            lt * torch.sum((c_noise + c_score * var) ** 2, dim=-1, keepdim=True)
        )

        return loss

