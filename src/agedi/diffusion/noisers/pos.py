import warnings
import torch

from typing import Dict, Optional
from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser
from agedi.diffusion.noisers.sde import SDENoiser
from agedi.diffusion.sdes import SDE, VE
from agedi.diffusion.distributions import (
    Distribution,
    NoiseSampler,
    Prior,
    Normal,
    TruncatedNormal,
    StandardNormal,
    UniformCell,
    UniformCellConfined,
)
from agedi.utils import OFFSET_LIST


class PositionsNoiser(SDENoiser):
    """Implements noising of atoms positions in Cartesian coordinates.

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
    sde : SDE, optional
        An already-instantiated SDE object.  When provided, *sde_class* and
        *sde_kwargs* are ignored.  Useful for reconstructing a noiser from
        saved hyperparameters.
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
        sde_class: SDE = VE,
        sde_kwargs: Optional[Dict] = None,
        distribution: NoiseSampler = Normal(),
        prior: Prior = UniformCell(),
        sde: Optional[SDE] = None,
        **kwargs
    ) -> None:
        """Initialize the positions noiser.

        Parameters
        ----------
        sde_class : SDE, optional
            Class of the SDE to use.  Defaults to :class:`~agedi.diffusion.sdes.VE`.
            Ignored when *sde* is provided.
        sde_kwargs : dict, optional
            Keyword arguments forwarded to *sde_class*.
            Ignored when *sde* is provided.
        distribution : NoiseSampler, optional
            Noise sampler used during noising and denoising.
            Defaults to :class:`~agedi.diffusion.distributions.Normal`.
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
        """Add noise to the atom positions.

        Added noise is stored in ``pos_noise``.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or batch hereof).

        """
        r = batch[self.key]
        t = batch.time

        w = self.distribution.get_callable(batch)
        setattr(batch, self.key, self.sde.transition_kernel(r, t, w))
        batch[self.key + "_noise"] = batch.apply_mask(self.sde.noise(r, batch.pos, t))

        return batch

    def denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoise the atom positions using the Euler-Maruyama scheme.

        The update rule is:

        .. math::

            R_{i+1} = R_i + \\Delta t (f(R_i, t) + g(t)^2 s(R_i, t))
                      + \\sqrt{\\Delta t} g(t) w

        The score is expected to be stored in ``pos_score``.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be denoised.
        delta_t: float
            The time step for the denoising.
        last: bool
            Whether this is the final denoising step.

        Returns
        -------
        AtomsGraph
            The denoised atomistic structure (or batch hereof).

        """
        r = batch[self.key]
        r_score = batch[self.key + "_score"]
        nan_mask = torch.isnan(r_score)

        if nan_mask.any():
            if batch.confinement is not None:
                warnings.warn(
                    "NaN score values detected for confined atoms. "
                    "This may indicate atoms drifted outside the confinement region. "
                    "Zeroing affected scores and continuing.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            r_score[nan_mask] = 0.0

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
        if batch.confinement is not None:
            confinement = batch.confinement[batch.batch]  # (n_atoms, 2)
            mobile = ~batch.mask
            new_pos = new_pos.clone()
            new_pos[mobile, 2] = new_pos[mobile, 2].clamp(
                min=confinement[mobile, 0],
                max=confinement[mobile, 1],
            )

        setattr(batch, self.key, new_pos)

        return batch

    def loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Compute the positions noiser loss.

        Expects the noise in ``pos_noise`` and the predicted score in
        ``pos_score``.

        The loss is:

        .. math::

            L = \\sum_i \\|\\sigma_t w_i + \\sigma_t^2 s(R_i)\\|^2

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised and denoised.

        Returns
        -------
        torch.Tensor
            The loss of the noised and denoised atomistic structure.

        """
        t = batch.time
        r_score = batch[self.key + "_score"]
        r_noise = batch[self.key + "_noise"]

        var = self.sde.var(t)

        r_score = batch.apply_mask(r_score)

        lt = 1.0

        loss = torch.mean(
            lt * torch.sum((r_noise + r_score * var) ** 2, dim=-1, keepdim=True)
        )
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
        cell_offsets = torch.matmul(
            torch.tensor(OFFSET_LIST, dtype=cells.dtype, device=cells.device), cells
        )  # m x 27 x 3
        cell_offsets = cell_offsets[idxs, :, :]  # 1 x 27 x 3

        Y = X + N
        Y = Y.unsqueeze(1)

        Y = Y + cell_offsets
        distances = torch.norm(X.unsqueeze(1) - Y, dim=2)

        argmin_distances = torch.argmin(distances, dim=1)
        Y = Y[torch.arange(Y.shape[0]), argmin_distances]
        min_N = Y - X

        return min_N


class Positions(PositionsNoiser):
    """Positions noiser with :class:`~agedi.diffusion.distributions.StandardNormal` prior
    and :class:`~agedi.diffusion.distributions.Normal` noise sampler.

    This is the base positions noiser suited for gas-phase clusters or systems
    where positions are not constrained to a periodic unit cell.  The SDE can
    still be chosen freely via the *sde* parameter.  Subclasses can override the
    ``distribution`` and ``prior`` while still delegating to this class through
    ``super()``.

    Parameters
    ----------
    sde_class : SDE, optional
        Class of the SDE to use.  Defaults to :class:`~agedi.diffusion.sdes.VE`.
        Ignored when *sde* is provided.
    sde_kwargs : dict, optional
        Keyword arguments forwarded to *sde_class*.
        Ignored when *sde* is provided.
    sde : SDE, optional
        Pre-instantiated SDE object.  When provided *sde_class* and
        *sde_kwargs* are ignored.
    distribution : NoiseSampler, optional
        Noise sampler.  Subclasses may supply a different default.
    prior : Prior, optional
        Prior distribution.  Subclasses may supply a different default.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`~agedi.diffusion.noisers.PositionsNoiser`.
    """

    def __init__(
        self,
        sde_class: SDE = VE,
        sde_kwargs: Optional[Dict] = None,
        sde: Optional[SDE] = None,
        distribution: NoiseSampler = Normal(),
        prior: Prior = StandardNormal(),
        **kwargs,
    ) -> None:
        super().__init__(
            sde_class=sde_class,
            sde_kwargs=sde_kwargs,
            distribution=distribution,
            prior=prior,
            sde=sde,
            **kwargs,
        )

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this positions noiser.

        Only includes :attr:`sde` and :attr:`loss_scaling`; the distribution
        and prior are fixed by the class and not needed for reconstruction.
        """
        return {
            "_target_": f"{type(self).__module__}.{type(self).__qualname__}",
            "sde": self.sde.get_hparams(),
            "loss_scaling": self.loss_scaling,
        }


class CellPositions(Positions):
    """Positions noiser with :class:`~agedi.diffusion.distributions.UniformCell` prior
    and :class:`~agedi.diffusion.distributions.Normal` noise sampler.

    Suited for periodic bulk or surface systems where atoms should be
    initialised uniformly within the unit cell.  Inherits from
    :class:`Positions`; the SDE can still be chosen freely.

    Parameters
    ----------
    sde_class : SDE, optional
        Class of the SDE to use.  Defaults to :class:`~agedi.diffusion.sdes.VE`.
        Ignored when *sde* is provided.
    sde_kwargs : dict, optional
        Keyword arguments forwarded to *sde_class*.
        Ignored when *sde* is provided.
    sde : SDE, optional
        Pre-instantiated SDE object.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`~agedi.diffusion.noisers.PositionsNoiser`.
    """

    def __init__(
        self,
        sde_class: SDE = VE,
        sde_kwargs: Optional[Dict] = None,
        sde: Optional[SDE] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            sde_class=sde_class,
            sde_kwargs=sde_kwargs,
            distribution=Normal(),
            prior=UniformCell(),
            sde=sde,
            **kwargs,
        )


class ConfinedCellPositions(Positions):
    """Positions noiser with :class:`~agedi.diffusion.distributions.UniformCellConfined`
    prior and :class:`~agedi.diffusion.distributions.TruncatedNormal` noise sampler.

    Suited for surface adsorption or porous-material systems where atoms are
    confined to a Z-range within the unit cell.  Inherits from
    :class:`Positions`; the SDE can still be chosen freely.

    Parameters
    ----------
    sde_class : SDE, optional
        Class of the SDE to use.  Defaults to :class:`~agedi.diffusion.sdes.VE`.
        Ignored when *sde* is provided.
    sde_kwargs : dict, optional
        Keyword arguments forwarded to *sde_class*.
        Ignored when *sde* is provided.
    sde : SDE, optional
        Pre-instantiated SDE object.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`~agedi.diffusion.noisers.PositionsNoiser`.
    """

    def __init__(
        self,
        sde_class: SDE = VE,
        sde_kwargs: Optional[Dict] = None,
        sde: Optional[SDE] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            sde_class=sde_class,
            sde_kwargs=sde_kwargs,
            distribution=TruncatedNormal(),
            prior=UniformCellConfined(),
            sde=sde,
            **kwargs,
        )

