import math
import torch

from typing import Optional
from agedi.data import AtomsGraph
from agedi.diffusion.noisers.sde import SDENoiser
from agedi.diffusion.sdes import SDE, VP
from agedi.diffusion.distributions import NoiseDistribution, PriorDistribution, Normal, StandardNormal


class Cell(SDENoiser):
    """Implements noising of the unit cell using VP diffusion.

    The cell is represented as a 3×3 lower-triangular matrix (canonical form).
    The score model predicts 6 values corresponding to the 6 entries of the
    lower triangular part; the upper triangular entries are kept at zero
    throughout the forward and reverse diffusion.

    Parameters
    ----------
    sde : SDE, optional
        An already-instantiated SDE object that defines the diffusion style.
        Defaults to :class:`~agedi.diffusion.sdes.VP`.
    distribution : NoiseDistribution
        The noise sampler to be used during forward and reverse diffusion.
        Defaults to :class:`~agedi.diffusion.distributions.Normal`.
    prior : PriorDistribution
        The prior distribution used to initialise the cell at the start of
        the reverse trajectory.  Defaults to
        :class:`~agedi.diffusion.distributions.StandardNormal`.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`~agedi.diffusion.noisers.sde.SDENoiser`.

    Returns
    -------
    Noiser
        The noiser for the unit cell.

    """

    _key = "cell"

    def __init__(
        self,
        sde: Optional[SDE] = None,
        distribution: NoiseDistribution = Normal(),
        prior: PriorDistribution = StandardNormal(),
        **kwargs
    ) -> None:
        if sde is None:
            sde = VP(beta_min=1e-2, beta_max=0.5)
        super().__init__(sde=sde, distribution=distribution, prior=prior, **kwargs)

    @staticmethod
    def _tril_mask(device: torch.device) -> torch.Tensor:
        """Return a lower-triangular boolean mask of shape ``(3, 3)``."""
        return torch.ones(3, 3, dtype=torch.bool, device=device).tril()

    def initialize_graph(self, batch: AtomsGraph) -> None:
        """Initialise the cell from a standard-normal lower-triangular prior.

        Parameters
        ----------
        batch : AtomsGraph
            The atomistic structure (or batch thereof) to be initialised.

        """
        n_graphs = batch.cell.view(-1, 3, 3).shape[0]
        device = batch.cell.device
        dtype = batch.cell.dtype
        cell = torch.randn(n_graphs, 3, 3, device=device, dtype=dtype)
        cell = cell * self._tril_mask(device)
        batch._store["cell"] = cell.reshape(-1, 3)

    def noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Add noise to the cell.

        The noised cell is stored in ``batch.cell`` and the normalised noise
        (unit-scale, lower-triangular) is stored in ``cell_noise``.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch thereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or batch thereof).

        """
        cell = batch.cell.view(-1, 3, 3)  # (n_graphs, 3, 3)
        f = batch.frac.clone()
        t = batch.time[batch.ptr[:-1]].reshape(-1, 1, 1)

        mean = self.sde.mean(t) * cell
        sigma = torch.sqrt(self.sde.var(t))
        noised_cell = self.distribution.sample(batch, mu=mean, sigma=sigma)

        # Enforce lower-triangular structure
        tril_mask = self._tril_mask(cell.device)
        noised_cell = noised_cell * tril_mask

        batch._store["cell"] = noised_cell.reshape(-1, 3)
        batch.frac = f

        noise = self.distribution.last_noise()
        if noise is None:
            raise RuntimeError(
                f"{type(self.distribution).__name__}.last_noise() returned None after sample(). "
                "Distributions used with Cell must cache unit-scale noise in last_noise()."
            )
        normalized_noise = noise * tril_mask
        batch.add_batch_attr(self.key + "_noise", normalized_noise, type="graph")

        return batch

    def denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoise the cell using the Euler-Maruyama scheme.

        .. math::

            C_{i+1} = C_i + \\Delta t (f(C_i, t) + g(t)^2 s(C_i, t))
                      + \\sqrt{\\Delta t}\\, g(t)\\, w

        The score is expected to be stored in ``cell_score``.

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
        cell = batch.cell.view(-1, 3, 3)  # (n_graphs, 3, 3)
        f = batch.frac.clone()
        c_score = batch[self.key + "_score"].view(-1, 3, 3)  # (n_graphs, 3, 3)
        t = batch.time[batch.ptr[:-1]].reshape(-1, 1, 1)

        drift = self.sde.drift(cell, t)
        diffusion = self.sde.diffusion(t)

        tril_mask = self._tril_mask(cell.device)

        if last:
            new_cell = cell + delta_t * (diffusion**2 * c_score + drift)
        else:
            mean = cell + delta_t * (diffusion**2 * c_score + drift)
            sigma = math.sqrt(delta_t) * diffusion
            new_cell = self.distribution.sample(batch, mu=mean, sigma=sigma)

        new_cell = new_cell * tril_mask

        batch._store["cell"] = new_cell.reshape(-1, 3)
        batch.frac = f

        return batch

    def loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Compute the cell noiser loss.

        Expects the normalised noise in ``cell_noise`` and the predicted score
        in ``cell_score``.

        The loss is:

        .. math::

            L = \\frac{1}{N} \\sum_i \\|\\sigma_t^2 s(C_i) + w_i\\|^2

        where the sum is over the lower-triangular entries only.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch thereof).

        Returns
        -------
        torch.Tensor
            The scalar loss.

        """
        t = batch.time[batch.ptr[:-1]].reshape(-1, 1, 1)
        c_score = batch[self.key + "_score"].view(-1, 3, 3)
        c_noise = batch[self.key + "_noise"].view(-1, 3, 3)

        var = self.sde.var(t)

        loss = torch.mean(
            torch.sum((c_noise + c_score * var) ** 2, dim=(-2, -1), keepdim=True)
        )

        return loss


#: Backward-compatible alias for :class:`Cell`.
CellNoiser = Cell
