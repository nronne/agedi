import torch

from typing import Dict
from agedi.diffusion.noisers import SDE, VP
from agedi.diffusion.noisers.base import Noiser
from agedi.diffusion.noisers.distributions import Distribution, Normal, UniformCell
from agedi.utils import OFFSET_LIST


class PositionsNoiser(Noiser):
    """Implements noising of atoms positions in Cartesian coordinates.

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
    _key = "pos"
    
    def __init__(
        self,
        sde_class: SDE=VP,
        sde_kwargs: Dict={},
        distribution: Distribution=Normal(),
        prior: Distribution=UniformCell(),
        **kwargs
    ) -> None:
        print('key', self.key)
        super().__init__(sde_class, sde_kwargs, distribution, prior, **kwargs)


    def _noise(self, batch: "AtomsGraph") -> "AtomsGraph":
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
        r = batch.pos
        t = batch.time

        w = self.distribution.get_callable(batch)
        batch.pos = self.sde.transition_kernel(r, t, w)
        batch[self.key + "_noise"] = batch.apply_mask(self.sde.noise(r, batch.pos, t))
        return batch

    def _denoise(self, batch: "AtomsGraph", delta_t: float) -> "AtomsGraph":
        """Denoises the positions of the atomistic structure.

        The denoising follows the Euler-Maruyama scheme.
        ::math::
        R_i+1 = R_i + \Delta t (f(R_i, t) + g(t)**2 * s(R_i, t)) + \sqrt{\Delta t} g(t) * w
        
        The used score is expected to be stored in the self.key+"_score", which by default is
        "positions_score".

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be denoised.
        delta_t: float
            The time step for the denoising.

        Returns
        -------
        AtomsGraph
            The denoised atomistic structure (or bach hereof).
        
        """
        r = batch.pos
        r_score = batch[self.key + "_score"]
        t = batch.time

        drift = self.sde.drift(r, t)
        diffusion = self.sde.diffusion(t)

        w = self.distribution.get_callable(batch)
        batch.pos = w(
            delta_t * (drift + diffusion**2 * r_score),
            torch.sqrt(delta_t) * diffusion
        )

        return batch

    def _loss(self, batch: "AtomsGraph") -> torch.Tensor:
        """Compute the noiser loss.
        
        Computes the loss of the diffusion model for the positions noiser

        Expects the total added positions noise to be stored in the self.key+"_noise", which by default is
        "positions_noise" and the predicted score to be stored in the self.key+"_score", which by default is
        "positions_score".

        The loss is computed as
        ::math::
        L = \sum_i ||w_i + \sigma_t s(R_i)||^2

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

        r_score = batch.apply_mask(r_score)
        r_noise = self.periodic_distance(batch.pos, r_noise, batch.cell, batch.batch)

        loss = torch.mean(torch.sum((r_noise + r_score * var) ** 2, dim=-1))
        return loss

    def periodic_distance(
        self, X: torch.tensor, N: torch.tensor, cells: torch.tensor, idxs: torch.tensor
    ) -> torch.tensor:
        """Periodic distance computation.
        
        Takes X and N (noise) and computes the minimum distance between X and Y=X+N
        taking into account periodic boundary conditions.

        Parameters
        ----------
        X: torch.Tensor
            The positions (N, 3)
        N: torch.Tensor
            The noise (N, 3)
        cell: torch.Tensor
            The cell (3*K, 3)
        idxs: torch.Tensor
            The indices of atoms in graphs (N,)

        Returns
        -------
        dist: torch.Tensor
            The distance between X and Y=X+N
        
        """
        cells = cells.view(-1, 3, 3)
        cell_offsets = torch.matmul(torch.tensor(OFFSET_LIST, dtype=cells.dtype, device=cells.device), cells)  # m x 27 x 3
        cell_offsets = cell_offsets[idxs, :, :]  # 1 x 27 x 3

        Y = X + N
        Y = Y.unsqueeze(1)

        Y = Y + cell_offsets
        distances = torch.norm(X.unsqueeze(1) - Y, dim=2)

        argmin_distances = torch.argmin(distances, dim=1)
        Y = Y[torch.arange(Y.shape[0]), argmin_distances]
        min_N = Y - X

        return min_N
