"""Helpers shared by the inference-side API functions."""

from typing import Optional


def resolve_cutoff(diffusion: "Agedi", cutoff: Optional[float] = None) -> float:
    """Return the neighbour-list cutoff to build graphs with.

    Parameters
    ----------
    diffusion:
        A trained :class:`~agedi.Agedi` model.
    cutoff:
        Explicit cutoff in Å.  When ``None`` (default), it is read from the
        model's representation, falling back to ``6.0`` when the
        representation does not expose one.

    Returns
    -------
    float
        The cutoff radius in Å.
    """
    if cutoff is not None:
        return float(cutoff)

    try:
        cf = diffusion.score_model.representation.cutoff_fn
        if hasattr(cf, "cutoff") and cf.cutoff.numel() > 0:
            return float(cf.cutoff[0])
    except AttributeError:
        pass
    return 6.0
