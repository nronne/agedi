"""Relative weighting of the diffusion and force-field losses.

``regressor_loss_weight`` is an *absolute* multiplier: the useful value depends
on how large the two losses happen to be, which varies with the system, the
units of the labels, and the noise schedule.  A weight tuned on one dataset
rarely transfers to another.

``loss_balance`` instead expresses the split in *relative* terms — 50/50, 80/20
— and the model divides each term by a running estimate of its own magnitude
before applying the fractions:

.. math::

    \\mathcal{L} = w_d \\frac{\\mathcal{L}_\\text{diffusion}}{s_d}
                 + w_r \\frac{\\mathcal{L}_\\text{regressor}}{s_r},
    \\qquad w_d + w_r = 1

where :math:`s_d, s_r` are detached exponential moving averages of the two
losses.  Since :math:`\\mathcal{L}_\\bullet / s_\\bullet \\approx 1` for both
terms, each contributes its requested fraction of the total regardless of the
raw scales — so the same ``loss_balance`` transfers between systems.

This balances *loss magnitude*, not gradient norm; the two coincide only when
the terms have comparable curvature.  See
:meth:`agedi.diffusion.Agedi._combine_losses` for the implementation.
"""

from typing import Mapping, Optional, Sequence, Tuple, Union


#: Accepted user specifications for a diffusion/regressor split.
LossBalanceSpec = Union[float, int, str, Sequence[float], Mapping[str, float], None]


__all__ = ["LossBalanceSpec", "normalize_loss_balance", "format_loss_balance"]


_SEPARATORS = (":", "/", "-", ",")


def _from_string(text: str) -> Tuple[float, float]:
    """Parse ``"80:20"``, ``"80/20"``, ``"0.8-0.2"`` or a bare ``"0.2"``."""
    cleaned = text.strip().replace("%", "")
    for separator in _SEPARATORS:
        if separator in cleaned:
            parts = [p for p in cleaned.split(separator) if p.strip()]
            if len(parts) != 2:
                raise ValueError(
                    f"loss_balance='{text}' must contain exactly two values, "
                    "e.g. '80:20'."
                )
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                raise ValueError(
                    f"loss_balance='{text}' contains a non-numeric value."
                ) from None
    try:
        regressor_fraction = float(cleaned)
    except ValueError:
        raise ValueError(
            f"loss_balance='{text}' is not recognized; use 'D:R' (e.g. '80:20') "
            "or a single number giving the regressor fraction."
        ) from None
    return 1.0 - regressor_fraction, regressor_fraction


def normalize_loss_balance(spec: LossBalanceSpec) -> Optional[Tuple[float, float]]:
    """Normalise a loss-balance specification to ``(w_diffusion, w_regressor)``.

    Parameters
    ----------
    spec : float, str, sequence, mapping, or None
        Accepted forms:

        * ``None`` – disabled; the absolute ``regressor_loss_weight`` is used.
        * ``0.2`` – a single number is the **regressor** fraction (here 80/20).
        * ``"80:20"``, ``"80/20"``, ``"0.8-0.2"``, ``"50%:50%"`` – a pair.
        * ``(0.8, 0.2)`` or ``[8, 2]`` – ``(diffusion, regressor)``.
        * ``{"diffusion": 0.8, "regressor": 0.2}``.

        Pairs do not have to sum to one; they are normalised by their sum, so
        ``80:20``, ``8:2`` and ``0.8:0.2`` are equivalent.

    Returns
    -------
    Tuple[float, float] or None
        The normalised ``(w_diffusion, w_regressor)`` fractions summing to
        ``1.0``, or ``None`` when balancing is disabled.

    Raises
    ------
    ValueError
        If the specification is malformed, contains negative values, or sums
        to zero.
    """
    if spec is None:
        return None

    if isinstance(spec, str):
        weights = _from_string(spec)
    elif isinstance(spec, Mapping):
        unknown = set(spec) - {"diffusion", "regressor"}
        if unknown:
            raise ValueError(
                f"loss_balance mapping has unknown keys: {sorted(unknown)}; "
                "expected 'diffusion' and 'regressor'."
            )
        weights = (float(spec.get("diffusion", 0.0)), float(spec.get("regressor", 0.0)))
    elif isinstance(spec, (int, float)) and not isinstance(spec, bool):
        regressor_fraction = float(spec)
        weights = (1.0 - regressor_fraction, regressor_fraction)
    elif isinstance(spec, Sequence):
        if len(spec) != 2:
            raise ValueError(
                f"loss_balance must have exactly two entries "
                f"(diffusion, regressor), got {len(spec)}."
            )
        try:
            weights = (float(spec[0]), float(spec[1]))
        except (TypeError, ValueError):
            raise ValueError(f"loss_balance={spec!r} contains a non-numeric value.") from None
    else:
        raise ValueError(
            "loss_balance must be a number, a 'D:R' string, a two-element "
            f"sequence, or a mapping; got {type(spec).__name__}."
        )

    if any(w < 0 for w in weights):
        raise ValueError(f"loss_balance weights must be non-negative, got {weights}.")

    total = weights[0] + weights[1]
    if total <= 0:
        raise ValueError("loss_balance weights must not sum to zero.")

    return (weights[0] / total, weights[1] / total)


def format_loss_balance(balance: Optional[Tuple[float, float]]) -> str:
    """Render a normalised balance as ``"80% / 20% (diffusion / regressor)"``.

    Parameters
    ----------
    balance : Tuple[float, float] or None
        Normalised fractions, as returned by :func:`normalize_loss_balance`.

    Returns
    -------
    str
        Human-readable summary (``"disabled"`` when *balance* is ``None``).
    """
    if balance is None:
        return "disabled"
    return (
        f"{balance[0] * 100:.0f}% / {balance[1] * 100:.0f}% "
        "(diffusion / regressor)"
    )
