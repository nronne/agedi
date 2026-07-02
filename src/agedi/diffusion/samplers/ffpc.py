"""Force-field augmented predictor-corrector sampler."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, ClassVar, List, Literal, Optional

import torch

from .base import Sampler

if TYPE_CHECKING:
    from agedi.data import AtomsGraph
    from agedi.diffusion.noisers import Noiser


class ForcefieldCorrectorSampler(Sampler):
    """EM predictor with force-field augmented Langevin corrector.

    Follows the standard predictor-corrector scheme from Song et al. (2021)
    but uses a force-field augmented score in the corrector steps:

    **Predictor** — standard EM step using the neural score only:

    .. math::

        x_{t-\\Delta t} = x_t
            + \\Delta t\\,(g(t)^2\\, s_\\theta(x_t) - f(x_t,\\,t))
            + \\sqrt{\\Delta t}\\, g(t)\\, z

    **Corrector** — ``corrector_steps`` Langevin steps at the new noise level
    ``t - \\Delta t``, using a score that blends the neural model with the
    force-field gradient:

    .. math::

        \\tilde{s}(x, t) = (1 - f(t))\\, s_\\theta(x) + f(t)\\, F(x)

    where :math:`f(t) = (1 - t)^\\zeta` increases as :math:`t \\to 0`, so the
    force field has no influence at the start of sampling and full influence at
    the end.  Temperature is **not** applied here; the relative weight of the
    force field is controlled entirely by ``mixing_zeta`` and
    ``corrector_step_size``.

    **Terminal phase** (optional) — applies ``terminal_steps`` additional
    refinement steps after the last diffusion step using only the force field.
    Two dynamics modes are supported via ``terminal_dynamics``:

    *overdamped* (default) — overdamped Langevin (no momenta):

    .. math::

        x_{n+1} = x_n + \\varepsilon\\, F(x_n) + \\sqrt{2\\varepsilon T}\\, z

    where :math:`\\varepsilon = \\texttt{terminal\\_step\\_size}` is a
    **reduced step** :math:`\\varepsilon = \\Delta t / T`.  This parameterization
    gives T-independent stability (condition: :math:`\\varepsilon < 2/k_{\\max}`)
    while preserving the correct Boltzmann stationary distribution
    :math:`\\propto \\exp(-U / T)` through the :math:`\\sqrt{T}` factor in the
    noise amplitude.

    *langevin_md* — standard Langevin MD (BAOAB integrator) with atomic masses
    from ASE.  Momenta are initialised from the Maxwell-Boltzmann distribution
    at temperature *T*:

    .. math::

        p_i \\sim \\mathcal{N}\\!\\left(0,\\, m_i T\\right)

    followed by BAOAB integration:

    .. math::

        p &\\leftarrow p + \\tfrac{\\Delta t}{2}\\, F \\\\
        x &\\leftarrow x + \\tfrac{\\Delta t}{2}\\, p / m \\\\
        p &\\leftarrow e^{-\\gamma \\Delta t}\\, p
          + \\sqrt{T\\, m\\,(1 - e^{-2\\gamma \\Delta t})}\\, z \\\\
        x &\\leftarrow x + \\tfrac{\\Delta t}{2}\\, p / m \\\\
        F &\\leftarrow F(x) \\\\
        p &\\leftarrow p + \\tfrac{\\Delta t}{2}\\, F

    When no regressor is attached (``regressor_fn=None``), the corrector falls
    back to a pure Langevin step with the neural score, and terminal steps are
    skipped.

    Parameters
    ----------
    score_fn : callable
        Neural score model: ``score_fn(batch) -> batch`` with ``pos_score`` set.
    noisers : list[Noiser]
        Noisers in forward order; iterated in reverse for denoising.
    regressor_fn : callable or None
        Force-field model: ``regressor_fn(batch) -> batch`` with
        ``forces_prediction`` set.  Passed by
        :meth:`~agedi.diffusion.Diffusion._resolve_sampler`.
    corrector_steps : int
        Number of Langevin corrector steps per predictor step.  Default: ``1``.
    corrector_step_size : float
        Step size for each Langevin corrector step.  Subject to the Langevin
        stability bound ``corrector_step_size < 2·var(t_{i-1})``.
        Default: ``1e-3``.
    mixing_zeta : float
        Exponent for the mixing schedule ``f(t) = (1 - t)**mixing_zeta``.
        ``1.0`` (default) gives linear mixing; higher values concentrate
        force-field influence near the end of the trajectory.
    temperature : float
        Temperature *T* used **only** in the terminal phase (not in the
        corrector steps, where dividing forces by T causes blowup at low T).

        * *overdamped*: appears as :math:`\\sqrt{T}` in the noise amplitude
          (see the update rule above) to preserve the correct Boltzmann
          distribution.  The step-size stability bound is T-independent.
        * *langevin_md*: the thermal energy :math:`k_B T` in the same units
          as the model forces (typically eV).  Sets the noise amplitude in
          the Ornstein-Uhlenbeck step and the Maxwell-Boltzmann velocity
          initialisation.

        Default: ``None`` — falls back to ``1.0`` with a :class:`UserWarning`
        reminding you to set an explicit value.
    terminal_steps : int
        Number of refinement steps applied after the last diffusion step.
        ``0`` (default) disables them.
    terminal_step_size : float or None
        Step size for each terminal step.

        * *overdamped*: the **reduced step** :math:`\\varepsilon = \\Delta t / T`.
          Stability requires :math:`\\varepsilon < 2 / k_{\\max}` where
          :math:`k_{\\max}` is the largest force-constant eigenvalue — a bound
          that is independent of *T*.  ``None`` (default) auto-selects
          ``1e-3``, which is conservative for most force fields.
        * *langevin_md*: physical time step in units consistent with the
          model forces and ASE masses.  When forces are in eV/Å and masses
          in amu the unit is femtoseconds.  ``None`` (default) auto-selects
          ``1.0`` (1 fs), a safe starting point for typical ML potentials.
    terminal_dynamics : ``"overdamped"`` or ``"langevin_md"``
        Dynamics used for the terminal phase.  ``"overdamped"`` (default) uses
        overdamped Langevin (no momenta).  ``"langevin_md"`` uses standard
        Langevin MD (BAOAB) with real atomic masses from ASE and momenta
        initialised from the Maxwell-Boltzmann distribution.
    terminal_friction : float or None
        Friction coefficient :math:`\\gamma` for the Langevin thermostat in
        *langevin_md* dynamics (ignored for *overdamped*).  Units are the
        inverse of ``terminal_step_size``; when ``terminal_step_size`` is in
        fs a physically reasonable range is ``0.001``–``0.1`` fs⁻¹
        (1–100 ps⁻¹).  ``None`` (default) auto-selects
        :math:`\\gamma = 0.1 / \\Delta t`, giving a dimensionless reduced
        friction :math:`\\gamma \\Delta t = 0.1` (moderately damped).

    String alias
    ------------
    ``"ffpc"`` — registered in :mod:`agedi.diffusion.samplers`.

    Notes
    -----
    This sampler calls :attr:`regressor_fn` directly, *not*
    ``force_field_guidance_step``.  The LBFGS step sizer is therefore not
    required; ``uses_force_field`` is ``False``.
    """

    # Does not call force_field_guidance_step, so LBFGS is not needed.
    uses_force_field: ClassVar[bool] = False

    def __init__(
        self,
        score_fn: Callable[["AtomsGraph"], "AtomsGraph"],
        noisers: List["Noiser"],
        regressor_fn: Optional[Callable[["AtomsGraph"], "AtomsGraph"]] = None,
        corrector_steps: int = 1,
        corrector_step_size: float = 1e-3,
        mixing_zeta: float = 1.0,
        temperature: Optional[float] = None,
        terminal_steps: int = 0,
        terminal_step_size: Optional[float] = None,
        terminal_dynamics: Literal["overdamped", "langevin_md"] = "overdamped",
        terminal_friction: Optional[float] = None,
    ) -> None:
        if temperature is None:
            import warnings
            warnings.warn(
                "ForcefieldCorrectorSampler: temperature not set, defaulting to 1.0. "
                "For physical terminal dynamics set temperature to k_B·T in the same "
                "units as your model forces (e.g. 0.026 eV for 300 K).",
                UserWarning,
                stacklevel=2,
            )
            temperature = 1.0
        super().__init__(score_fn, noisers)
        self.regressor_fn = regressor_fn
        self.corrector_steps = corrector_steps
        self.corrector_step_size = corrector_step_size
        self.mixing_zeta = mixing_zeta
        self.temperature = temperature
        self.terminal_steps = terminal_steps
        self.terminal_step_size = terminal_step_size
        self.terminal_dynamics = terminal_dynamics
        self.terminal_friction = terminal_friction
        # Cache the positions noiser for overdamped terminal steps.
        self._pos_noiser = next((n for n in noisers if n.key == "pos"), None)
        self._corrector_dt: Optional[torch.Tensor] = None
        # Terminal frames collected during the last step; consumed by _sample_batch
        # to extend the saved trajectory.  Cleared at the start of every step().
        self._pending_frames: List = []

    def step(
        self,
        batch: "AtomsGraph",
        dt: torch.Tensor,
        last: bool,
    ) -> "AtomsGraph":
        """Predictor-corrector step with force-field augmented corrector.

        1. EM predictor with neural score.
        2. Advance ``batch.time`` to ``t - dt``.
        3. For each corrector iteration: evaluate neural score, blend with
           force-field gradient, apply Langevin step.
        4. If ``last`` and ``terminal_steps > 0``: terminal phase with the
           chosen dynamics.  Intermediate frames are stored in
           :attr:`_pending_frames` for trajectory capture.
        """
        self._pending_frames.clear()

        # --- Step 1: EM predictor (neural score only) ---
        batch = self.score_fn(batch)
        for noiser in self.noisers[::-1]:
            batch = noiser.denoise(batch, dt, last=last)
        batch.wrap_positions()
        self._check_finite(batch, "FFPC predictor (EM) step")
        batch.update_graph()

        if self.corrector_steps == 0:
            if last and self.terminal_steps > 0 and self.regressor_fn is not None:
                self._run_terminal(batch)
            return batch

        # Advance time to t_{i-1} for the corrector.
        batch.time = (batch.time - dt).clamp(min=0.0)

        # Lazily create corrector_dt tensor on the correct device/dtype.
        if (
            self._corrector_dt is None
            or self._corrector_dt.device != dt.device
            or self._corrector_dt.dtype != dt.dtype
        ):
            self._corrector_dt = torch.tensor(
                self.corrector_step_size, dtype=dt.dtype, device=dt.device
            )

        # --- Step 2: Corrector (Langevin with augmented score) ---
        for _ in range(self.corrector_steps):
            batch = self.score_fn(batch)
            if self.regressor_fn is not None:
                batch = self.regressor_fn(batch)
                # f(t) = (1-t)^zeta, shape (n_atoms, 1) → broadcasts over (n_atoms, 3)
                f_t = (1.0 - batch.time).clamp(min=0.0).pow(self.mixing_zeta)
                batch["pos_score"] = (
                    (1.0 - f_t) * batch["pos_score"]
                    + f_t * batch["forces_prediction"]
                )
            for noiser in self.noisers[::-1]:
                batch = noiser.langevin_step(batch, self._corrector_dt)
            batch.wrap_positions()
            self._check_finite(batch, "FFPC augmented Langevin corrector step")
            batch.update_graph()

        # --- Step 3: Terminal phase (force-field only, last step) ---
        if last and self.terminal_steps > 0 and self.regressor_fn is not None:
            self._run_terminal(batch)

        return batch

    def _run_terminal(self, batch: "AtomsGraph") -> None:
        """Dispatch to the selected terminal dynamics.

        Prepends a bridge frame (the denoised state before terminal dynamics
        start) unless corrector frames already captured it as their last
        entry — which is the case when ``corrector_steps > 0``.
        """
        if not self._pending_frames:
            self._pending_frames.append(batch.to_data_list())
        if self.terminal_dynamics == "langevin_md":
            self._terminal_langevin_md(batch)
        else:
            self._terminal_overdamped(batch)

    def _terminal_overdamped(self, batch: "AtomsGraph") -> None:
        """Overdamped Langevin terminal steps (no momenta).

        ``terminal_step_size`` is the reduced step ε = dt/T.
        Update: x += ε·F + sqrt(2εT)·z, giving stationary ∝ exp(-U/T).
        Stability condition: ε < 2/k_max, independent of T.
        """
        eps = self.terminal_step_size if self.terminal_step_size is not None else 1e-3
        noise_std = torch.tensor(
            math.sqrt(2.0 * eps * self.temperature),
            dtype=batch.pos.dtype,
            device=batch.pos.device,
        )
        for _ in range(self.terminal_steps):
            batch = self.regressor_fn(batch)
            mean = batch.pos + eps * batch.forces_prediction
            if self._pos_noiser is not None:
                w = self._pos_noiser.distribution.get_callable(batch)
                batch.pos = w(mean, noise_std)
            else:
                batch.pos = mean + noise_std * torch.randn_like(batch.pos)
            batch.wrap_positions()
            self._check_finite(batch, "FFPC terminal overdamped Langevin step")
            batch.update_graph()
            self._pending_frames.append(batch.to_data_list())

    def _terminal_langevin_md(self, batch: "AtomsGraph") -> None:
        """Standard Langevin MD terminal steps (BAOAB, real atomic masses).

        Masses are looked up from ASE using ``batch.x`` (atomic numbers).
        Temperature *T* is the thermal energy :math:`k_B T` in model force
        units.  Time step units must be consistent with forces and masses
        (e.g. fs when forces are in eV/Å and masses in amu).
        """
        from ase.data import atomic_masses as _ase_masses

        # Atomic masses (n_atoms, 1) in amu, on the same device/dtype as pos.
        masses = torch.tensor(
            [_ase_masses[int(z)] for z in batch.x.tolist()],
            dtype=batch.pos.dtype,
            device=batch.pos.device,
        ).unsqueeze(1)  # (n_atoms, 1) broadcasts over (n_atoms, 3)

        # Initialise velocities from the Maxwell-Boltzmann distribution:
        # v_i ~ N(0, kT / m_i)
        sigma_v = torch.sqrt(self.temperature / masses)  # (n_atoms, 1)
        vel = sigma_v * torch.randn_like(batch.pos)

        # Remove centre-of-mass drift so the simulation cell doesn't translate.
        vel -= (masses * vel).sum(0, keepdim=True) / masses.sum()

        # BAOAB Langevin thermostat constants.
        # terminal_step_size=None → 1.0 (e.g. 1 fs for eV/Å + amu force fields)
        # terminal_friction=None → γ·dt = 0.1 (moderately damped)
        dt = self.terminal_step_size if self.terminal_step_size is not None else 1.0
        gamma = (
            self.terminal_friction
            if self.terminal_friction is not None
            else (0.1 / dt)
        )
        alpha = math.exp(-gamma * dt)
        # Per-atom noise std for the O step: sqrt(kT / m * (1 - alpha^2))
        sigma_ou = torch.sqrt(
            self.temperature / masses * (1.0 - alpha ** 2)
        )  # (n_atoms, 1)

        # Initial forces.
        batch = self.regressor_fn(batch)
        forces = batch.forces_prediction.clone()

        for _ in range(self.terminal_steps):
            # B: half-step velocity kick from forces.
            vel = vel + 0.5 * dt * forces / masses

            # A: half-step position drift.
            batch.pos = batch.pos + 0.5 * dt * vel

            # O: Ornstein-Uhlenbeck thermostat.
            vel = alpha * vel + sigma_ou * torch.randn_like(vel)

            # A: half-step position drift.
            batch.pos = batch.pos + 0.5 * dt * vel
            batch.wrap_positions()
            self._check_finite(batch, "FFPC terminal Langevin MD step (BAOAB)")
            batch.update_graph()
            self._pending_frames.append(batch.to_data_list())

            # Recompute forces at the new position.
            batch = self.regressor_fn(batch)
            forces = batch.forces_prediction.clone()

            # B: half-step velocity kick from new forces.
            vel = vel + 0.5 * dt * forces / masses
