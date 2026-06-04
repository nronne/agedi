import math
import torch

from typing import Dict, Optional
from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser
from agedi.diffusion.sdes import SDE, VE
from agedi.diffusion.distributions import (
    Distribution,
    Normal,
    TruncatedNormal,
    StandardNormal,
    ZeroComNormal,
    ZeroComStandardNormal,
    UniformCell,
    UniformCellConfined,
)
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
        distribution: Distribution = Normal(),
        prior: Distribution = UniformCell(),
        sde: Optional[SDE] = None,
        loss_weighting: str = "uniform",
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
        distribution : Distribution, optional
            Noise distribution used during noising and denoising.
            Defaults to :class:`~agedi.diffusion.distributions.Normal`.
        prior : Distribution, optional
            Prior distribution used to sample starting positions.
            Defaults to :class:`~agedi.diffusion.distributions.UniformCell`.
        sde : SDE, optional
            Pre-instantiated SDE object.  When provided, *sde_class* and
            *sde_kwargs* are ignored.
        loss_weighting : str, optional
            Loss weighting strategy.  ``"uniform"`` weights all noise levels
            equally.  ``"min_snr"`` caps the per-sample weight at
            ``min(SNR, 5)`` following Hang et al. (ICCV 2023, arXiv:2303.09556),
            which balances contributions from high- and low-noise timesteps and
            typically accelerates convergence.  Defaults to ``"uniform"``.
        **kwargs
            Additional keyword arguments forwarded to :class:`~agedi.diffusion.noisers.Noiser`.
        """
        super().__init__(distribution, prior, **kwargs)
        if loss_weighting not in ("uniform", "min_snr"):
            raise ValueError(
                f"loss_weighting must be 'uniform' or 'min_snr', got {loss_weighting!r}"
            )
        self.loss_weighting = loss_weighting
        if sde is not None:
            self.sde = sde
        else:
            if sde_kwargs is None:
                sde_kwargs = {}
            self.sde = sde_class(**sde_kwargs)

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this positions noiser."""
        return {
            **super().get_hparams(),
            "sde": self.sde.get_hparams(),
            "loss_weighting": self.loss_weighting,
        }

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
        r = batch[self.key]
        t = batch.time

        w = self.distribution.get_callable(batch)
        setattr(batch, self.key, self.sde.transition_kernel(r, t, w))
        batch[self.key + "_noise"] = batch.apply_mask(self.sde.noise(r, batch.pos, t))

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

        # Zero out NaN scores unconditionally with a tensor op to avoid the
        # data-dependent ``if nan_mask.any():`` guard that causes torch.compile
        # to recompile every time the NaN pattern changes.
        r_score = torch.where(torch.isnan(r_score), torch.zeros_like(r_score), r_score)

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
            # Clamp on the full-size tensor then apply only to mobile atoms via
            # torch.where.  Avoids boolean-indexed clamp (variable-size tensor)
            # that Dynamo cannot trace.
            clamped_z = torch.clamp(
                new_pos[:, 2],
                min=confinement[:, 0],
                max=confinement[:, 1],
            )
            new_pos[:, 2] = torch.where(mobile, clamped_z, new_pos[:, 2])

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

        r_score = batch.apply_mask(r_score)
        # r_noise = self.periodic_distance(batch.pos, r_noise, batch.cell, batch.batch)

        if self.loss_weighting == "min_snr":
            # Min-SNR-γ weighting (γ=5): caps per-sample weight at min(SNR, 5).
            # Balances loss contributions across noise levels and typically
            # accelerates convergence.  Hang et al., ICCV 2023, arXiv:2303.09556.
            snr = 1.0 / var - 1.0
            lt = torch.minimum(snr, torch.tensor(5.0, device=snr.device))
        else:
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
    """Positions noiser with :class:`~agedi.diffusion.distributions.ZeroComStandardNormal`
    prior and :class:`~agedi.diffusion.distributions.ZeroComNormal` noise distribution.

    This is the base positions noiser suited for gas-phase clusters or systems
    where positions are not constrained to a periodic unit cell.  The SDE can
    still be chosen freely via the *sde* parameter.  Subclasses can override the
    ``distribution`` and ``prior`` while still delegating to this class through
    ``super()``.

    When *prior* is not supplied, the prior scale is set automatically to
    ``sqrt(sde.var(t=1))`` — equal to ``sigma_max`` for a VE-SDE — so that
    the prior matches the forward-process marginal at T=1.

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
    distribution : Distribution, optional
        Noise distribution.  Subclasses may supply a different default.
    prior : Distribution, optional
        Prior distribution.  When ``None`` (default), a
        :class:`~agedi.diffusion.distributions.ZeroComStandardNormal` with
        ``scale = sqrt(sde.var(t=1))`` is created automatically.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`~agedi.diffusion.noisers.PositionsNoiser`.
    """

    def __init__(
        self,
        sde_class: SDE = VE,
        sde_kwargs: Optional[Dict] = None,
        sde: Optional[SDE] = None,
        distribution: Distribution = ZeroComNormal(),
        prior: Optional[Distribution] = None,
        **kwargs,
    ) -> None:
        # Build the SDE first so we can read sigma_max from it.
        if sde is not None:
            _sde = sde
        else:
            _sde = sde_class(**(sde_kwargs or {}))

        if prior is None:
            scale = math.sqrt(float(_sde.var(torch.tensor(1.0)).item()))
            prior = ZeroComStandardNormal(scale=scale)

        super().__init__(
            sde_class=sde_class,
            sde_kwargs=sde_kwargs,
            distribution=distribution,
            prior=prior,
            sde=_sde,
            **kwargs,
        )

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this positions noiser.

        Only includes :attr:`sde`, :attr:`loss_scaling`, and
        :attr:`loss_weighting`; the distribution and prior are fixed by the
        class and not needed for reconstruction.
        """
        return {
            "_target_": f"{type(self).__module__}.{type(self).__qualname__}",
            "sde": self.sde.get_hparams(),
            "loss_scaling": self.loss_scaling,
            "loss_weighting": self.loss_weighting,
        }


class CellPositions(Positions):
    """Positions noiser with :class:`~agedi.diffusion.distributions.UniformCell` prior
    and :class:`~agedi.diffusion.distributions.Normal` noise distribution.

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
    prior and :class:`~agedi.diffusion.distributions.TruncatedNormal` noise distribution.

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
