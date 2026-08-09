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

__all__ = ['SURVIVAL_FLOOR', 'WINDOW_PAD', 'gapped', 'loss_window',
           'pad_window', 'quantile_curve', 'survival_window']

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


def loss_window(q, first):
    """``(lo, hi)`` for an outcome axis: the slice worth looking at.

    Parameters
    ----------
    q : callable
        The distribution's quantile function, ``q(p) -> outcome``.
    first : float
        The first point of the grid, which is what says whether the axis is
        signed.

    Returns
    -------
    tuple of float or None
        Padded by :data:`WINDOW_PAD`, or None for a degenerate window.

    Notes
    -----
    Mirrors ``Aggregate._limits``, as the app does: a heavy tail otherwise
    squashes every visible mass into a sliver at the origin. An unsigned
    grid is read from zero, because starting a loss axis at ``q(0.001)``
    hides the mass at and near zero that a discrete book routinely has; a
    signed grid has no such anchor and takes ``q(0.001)``.
    """
    lo = float(q(0.001)) if first < 0 else min(0.0, float(first))
    return pad_window(lo, float(q(0.999)))


def quantile_curve(outcome, mass):
    """``(p, outcome)`` for a Lee diagram: the quantile function sideways.

    Parameters
    ----------
    outcome : array_like
        Grid points, ascending.
    mass : array_like
        The probability at each grid point.

    Returns
    -------
    tuple of ndarray
        Non-exceedance probability and outcome, trimmed to the support.

    Notes
    -----
    The trim is meaning, not tidying, and it is the same statement at both
    ends: a bucket outside the support carries no probability, so drawing
    it asserts an outcome the book cannot have. Past the top, ``F`` has
    already reached its maximum and the line would run flat out to the
    largest number the grid happens to hold; below the bottom, ``F`` is
    still zero and the line would drop at ``p = 0`` to the grid's left
    edge, which on a signed grid is an arbitrary distance below the worst
    thing that can happen. So the curve starts at the smallest outcome
    carrying probability and ends at the largest.
    """
    outcome = np.asarray(outcome, dtype=float)
    p = np.cumsum(np.asarray(mass, dtype=float))
    if not p.size or not p.max() > 0:
        return p, outcome
    lo = int(np.argmax(p > 0))
    hi = int(np.argmax(p >= p.max())) + 1
    return p[lo:hi], outcome[lo:hi]


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
