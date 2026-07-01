"""Pluggable reverse-diffusion samplers for AGeDi.

Each sampler encapsulates one complete reverse-diffusion outer step, from
time ``t`` to ``t - dt``.  Samplers may call the score model (and optionally
the force-field model) any number of times within that outer step.

Available samplers
------------------
:class:`EulerMaruyamaSampler`
    Standard Euler-Maruyama reverse-SDE step (the default).  One score call
    per step.  String alias: ``"em"``.

:class:`PredictorCorrectorSampler`
    EM predictor followed by ``N`` Langevin corrector steps at the new noise
    level.  String alias: ``"pc"``.

:class:`HeunSampler`
    Second-order stochastic sampler (Karras et al. 2022).  Two score calls
    per step; the averaged score is used in the final stochastic update.
    String alias: ``"heun"``.

:class:`ProbabilityFlowODESampler`
    Deterministic probability-flow ODE sampler (DDIM / Anderson 1982).  One
    score call per step; applies half the diffusion coefficient with no noise.
    String alias: ``"ddim"``.

:class:`HeunODESampler`
    Second-order deterministic ODE sampler (Heun's method on the PF-ODE).
    Two score calls per step; fully deterministic.  String alias: ``"heun_ode"``.

Usage
-----
Pass a sampler by string alias or instance to
:meth:`~agedi.diffusion.Diffusion.sample`::

    structures = model.sample(10, steps=200, sampler="heun")
    structures = model.sample(10, steps=500,
        sampler=PredictorCorrectorSampler(model.score_model, model.noisers,
                                          corrector_steps=3))

Register a custom sampler::

    from agedi.diffusion.samplers import Sampler

    class MySampler(Sampler):
        def step(self, batch, dt, last):
            ...

    Sampler.register("my_sampler",
        lambda score_fn, noisers, **kw: MySampler(score_fn, noisers))
"""

from .base import Sampler
from .em import EulerMaruyamaSampler
from .ffpc import ForcefieldCorrectorSampler
from .heun import HeunSampler
from .ode import HeunODESampler, ProbabilityFlowODESampler
from .pc import PredictorCorrectorSampler

# ---------------------------------------------------------------------------
# Built-in registry entries
# ---------------------------------------------------------------------------

Sampler.register(
    "em",
    lambda score_fn, noisers, **kw: EulerMaruyamaSampler(score_fn, noisers),
)

Sampler.register(
    "pc",
    lambda score_fn, noisers,
    corrector_steps=1, corrector_step_size=1e-3, **kw:
    PredictorCorrectorSampler(
        score_fn,
        noisers,
        corrector_steps=corrector_steps,
        corrector_step_size=corrector_step_size,
    ),
)

Sampler.register(
    "heun",
    lambda score_fn, noisers, **kw: HeunSampler(score_fn, noisers),
)

Sampler.register(
    "ddim",
    lambda score_fn, noisers, **kw: ProbabilityFlowODESampler(score_fn, noisers),
)

Sampler.register(
    "heun_ode",
    lambda score_fn, noisers, **kw: HeunODESampler(score_fn, noisers),
)

Sampler.register(
    "ffpc",
    lambda score_fn, noisers,
    regressor_fn=None, corrector_steps=1, corrector_step_size=1e-3,
    mixing_zeta=1.0, temperature=1.0,
    terminal_steps=0, terminal_step_size=1e-3,
    terminal_dynamics="overdamped", terminal_friction=1.0, **kw:
    ForcefieldCorrectorSampler(
        score_fn,
        noisers,
        regressor_fn=regressor_fn,
        corrector_steps=corrector_steps,
        corrector_step_size=corrector_step_size,
        mixing_zeta=mixing_zeta,
        temperature=temperature,
        terminal_steps=terminal_steps,
        terminal_step_size=terminal_step_size,
        terminal_dynamics=terminal_dynamics,
        terminal_friction=terminal_friction,
    ),
)

__all__ = [
    "Sampler",
    "EulerMaruyamaSampler",
    "PredictorCorrectorSampler",
    "HeunSampler",
    "ProbabilityFlowODESampler",
    "HeunODESampler",
    "ForcefieldCorrectorSampler",
]
