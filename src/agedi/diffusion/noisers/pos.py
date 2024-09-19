import torch

from agedi.diffusion.noisers import VP
from agedi.diffusion.noisers.base import Noiser
from agedi.diffusion.noisers.distributions import Normal, UniformCell
from agedi.utils import OFFSET_LIST


class PositionsNoiser(Noiser):
    """
    Implements noising of atoms positions
    
    """
    def __init__(
        self,
        sde_class=VP,
        sde_kwargs={},
        distribution=Normal(),
        prior=UniformCell(),
        key="positions",
        **kwargs
    ):
        super().__init__(sde_class, sde_kwargs, distribution, prior, key, **kwargs)

    def noise(self, batch):
        r = batch.pos
        t = batch.time

        w = self.distribution.get_callable(batch)
        batch.pos = self.sde.transition_kernel(r, t, w)
        batch[self.key + "_noise"] = batch.apply_mask(self.sde.noise(r, batch.pos, t))
        return batch

    def denoise(self, batch, step_size):
        r = batch.pos
        r_score = batch[self.key + "_score"]
        t = batch.time

        drift = self.sde.drift(r, t)
        diffusion = self.sde.diffusion(t)

        w = self.distribution.get_callable(batch)
        batch.pos = w(
            step_size * (drift + diffusion**2 * r_score),
            torch.sqrt(step_size) * diffusion
        ) # r = r + step_size * (drift + diffusion**2 * r_score) + torch.sqrt(step_size) * diffusion * noise

    def loss(self, batch):
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
        """
        Takes X and N (noise) and computes the minimum distance between X and Y=X+N
        taking into account periodic boundary conditions.

        Args
        ______
        X: torch.Tensor
            The positions (N, 3)
        N: torch.Tensor
            The noise (N, 3)
        cell: torch.Tensor
            The cell (3*K, 3)
        idxs: torch.Tensor
            The indices of atoms in graphs (N,)

        Returns
        _______
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
