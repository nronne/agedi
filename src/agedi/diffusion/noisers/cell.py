import torch
import numpy as np

from typing import Dict
from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser
from agedi.diffusion.sdes import SDE, VP, VE
from agedi.diffusion.distributions import Distribution, Normal, StandardNormal


class CellNoiser(Noiser):
    """Implements noising of the cell.

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

    _key = "cellpar"

    def __init__(
        self,
        sde_class: SDE = VE,
        sde_kwargs: Dict = {"sigma_max": 0.1},
        distribution: Distribution = Normal(),
        prior: Distribution = Normal(),
        **kwargs
    ) -> None:
        super().__init__(distribution, prior, **kwargs)
        self.sde = sde_class(**sde_kwargs)

    def _noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Initializes the noise for the cell noiser.

        Added noise is stored in the self.key+"_noise", which by default is
        "cell_noise".

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or bach hereof).

        """

        cellpar = getattr(batch, self.key)
        f = batch.frac.clone()
        t = batch.time[batch.ptr[:-1]].reshape(-1, 1)

        w = self.distribution.get_callable(batch)
        noised_cellpar = self.sde.transition_kernel(cellpar, t, w)


        a, b, c, alpha, beta, gamma, V = noised_cellpar.unbind(-1)
        # a, b, c = torch.clamp(a, min=0), torch.clamp(b, min=0), torch.clamp(c, min=0)
        # alpha, beta, gamma = torch.clamp(alpha, min=-1.0, max=1.0), torch.clamp(beta, min=-1.0, max=1.0), torch.clamp(gamma, min=-1.0, max=1.0)
        
        noised_cellpar = torch.stack([a,b,c,alpha,beta,gamma,V], dim=-1)
        
        setattr(batch, self.key, noised_cellpar)
        batch.frac = f
        # batch[self.key + "_noise"] = self.sde.noise(cellpar, noised_cellpar, t)
        batch.add_batch_attr(self.key + "_noise", self.sde.noise(cellpar, noised_cellpar, t), type="graph")

        # batch.wrap_positions()

        return batch

    def _denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoises the cell of the atomistic structure.

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
        c = getattr(batch, self.key)
        f = batch.frac.clone()
        c_score = batch[self.key + "_score"]
        if c_score.isnan().any():
            breakpoint()
            
        t = batch.time[batch.ptr[:-1]].reshape(-1, 1)

        drift = self.sde.drift(c, t)
        diffusion = self.sde.diffusion(t)

        w = self.distribution.get_callable(batch)
        if last:
            cellpar = c + delta_t * (diffusion**2 * c_score + drift)
        else:
            cellpar = w(
                c + delta_t * (diffusion**2 * c_score + drift),  # mean
                torch.sqrt(delta_t) * diffusion,  # variance
            )


        a, b, c, alpha, beta, gamma, V = cellpar.unbind(-1)
        # a, b, c = torch.clamp(a, min=0.3, max=1.3), torch.clamp(b, min=0.3, max=1.3), torch.clamp(c, min=0.3, max=1.3)
        # alpha, beta, gamma = torch.clamp(alpha, min=-1.0, max=1.0), torch.clamp(beta, min=-1.0, max=1.0), torch.clamp(gamma, min=-1.0, max=1.0)
        
        cellpar = torch.stack([a,b,c,alpha,beta,gamma,V], dim=-1)

        # print('lenghts:', torch.exp(a), torch.exp(b), torch.exp(c))
        # print('angles:', (alpha + np.pi/2)*180/np.pi, (beta + np.pi/2)*180/np.pi, (gamma + np.pi/2)*180/np.pi)
        setattr(batch, self.key, cellpar)
        batch.frac = f
        
        # print(batch.cell)


        return batch

    def _loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Compute the noiser loss.

        Computes the loss of the diffusion model for the cell noiser

        Expects the total added cell noise to be stored in the self.key+"_noise",
        which by default is "cell_noise" and the predicted score to be stored in the
        self.key+"_score", which by default is "cell_score".

        The loss is computed as
        ::math::
        L = \sum_i ||\sigma_t w_i + \sigma_t^2 s(C_i)||^2

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised and denoised.

        Returns
        -------
        float
            The loss of the noised and denoised atomistic structure.

        """
        shape = getattr(batch, self.key).shape

        t = batch.time[batch.ptr[:-1]].reshape(-1, 1)
        c_score = batch[self.key + "_score"]
        c_noise = batch[self.key + "_noise"]

        var = self.sde.var(t)

        lt = 1.0  # /var.sqrt()

        loss = torch.mean(
            lt * torch.sum((c_noise + c_score * var) ** 2, dim=-1, keepdim=True)
        )

        return loss
    
