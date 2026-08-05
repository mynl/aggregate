"""Shared semantics of the two-panel exhibit: a density and its tail.

Five charts draw this shape (reins, sev, agg, port, pnl), and the parts
that are *meaning* rather than styling are the same in all five: which
slice of the grid is worth looking at, how deep the survival panel reads,
and where a curve stops being tail and starts being float dust. They live
here so the five emitters cannot drift apart, which is the whole reason
the semantics were lifted out of the app in the first place.

Nothing here touches matplotlib, and nothing here decides how any of it
looks.
"""

import numpy as np

from ..constants import LOG_FLOOR

__all__ = ['SURVIVAL_FLOOR', 'WINDOW_PAD', 'gapped', 'pad_window',
           'survival_window']

#: The deepest survival worth a panel, one over the longest return period
#: anyone reads off a picture. Past it the curve is a line of float dust.
SURVIVAL_FLOOR = 1e-9

#: Fraction of the window added either side by the emitter, so a curve is
#: not drawn hard against the end of its own data.
WINDOW_PAD = 0.02


def pad_window(lo, hi):
    """``(lo, hi)`` widened by :data:`WINDOW_PAD` either side, or None.

    Parameters
    ----------
    lo, hi : float
        The window before padding.

    Returns
    -------
    tuple of float or None
        None when the window is empty or degenerate, which callers pass
        through as "no suggestion" rather than inventing one.
    """
    if not hi > lo:
        return None
    pad = WINDOW_PAD * (hi - lo)
    return (float(lo - pad), float(hi + pad))


def gapped(values):
    """Survival values as a series payload, float dust becoming gaps.

    Parameters
    ----------
    values : array_like
        Survival probabilities, in grid order.

    Returns
    -------
    tuple
        Floats, with ``None`` wherever the value is at or under
        :data:`~aggregate.constants.LOG_FLOOR`.

    Notes
    -----
    A gap, not a zero, and not the floor either. A log axis cannot place
    these values at all, and drawing them at the floor would put a fringe
    of arithmetic noise exactly where a reader expects the tail to be.
    """
    return tuple(float(v) if v > LOG_FLOOR else None for v in values)


def survival_window(series):
    """``(decade under the deepest survival drawn, 1.0)``.

    Parameters
    ----------
    series : iterable of iterable
        The survival payloads that will share the panel, gaps included.

    Returns
    -------
    tuple of float

    Notes
    -----
    Fixing the axis at ``(SURVIVAL_FLOOR, 1)`` would be simpler and wastes
    the panel: a book whose deepest survival is 1e-3 would draw six empty
    decades. Rounding down to a whole decade keeps the gridlines on round
    numbers, which is how a log axis is read.
    """
    seen = [v for s in series for v in s if v is not None and v > LOG_FLOOR]
    lo = max(SURVIVAL_FLOOR, min(seen)) if seen else SURVIVAL_FLOOR
    return (float(10.0 ** np.floor(np.log10(lo))), 1.0)
