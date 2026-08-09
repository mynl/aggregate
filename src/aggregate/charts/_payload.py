"""How a series carries its coordinates: the cheap form, never a lossy one.

Two facts about this library's data make its documents much smaller than
the point count suggests, and neither of them is compression: both are
statements about the data that happen to save bytes.

A computed grid is a **lattice**. An aggregate lives on ``k * bs``, so
listing 65,536 evenly spaced numbers spells out what three numbers already
say, and it does it once per series that shares the grid. Four series over
one loss grid cost four copies of it, which on a ``log2 = 16`` book is 2 MB
of a 7.4 MB document.

An empty stretch of that grid is **one fact, not thousands**. A lattice
severity leaves most buckets with no probability at all, and a run of them
carries no information beyond where it starts and where it stops.

Both are exact. The lattice form is used only where the values really are
arithmetic, checked rather than assumed, and the run collapse keeps the
zero either side of each run so the drawn shape is identical under every
rung of the renderer's ladder: stems land on the same atoms, steps still
fall to the floor and rise from it, and a line still passes through zero.

Nothing here rounds, thins or samples. A document is the data.
"""

import numpy as np

__all__ = ['collapse_empty_runs', 'lattice_payload']


def lattice_payload(values, step, prefix='x'):
    """``dict`` of the cheapest exact payload for a coordinate.

    Parameters
    ----------
    values : array_like
        The coordinates, in grid order.
    step : float or None
        The spacing they are expected to have (a ``bs``). ``None`` says the
        caller does not claim one, and the values go out in full.
    prefix : str
        'x' or 'y', which coordinate this is.

    Returns
    -------
    dict
        ``{'x': (...)}`` or ``{'x_lattice': (start, step, count)}``, ready
        to splat into a :class:`~aggregate.charts.ir.ChartSeries`.

    Notes
    -----
    The lattice form is taken only when ``start + step * arange(count)``
    reproduces the values **exactly**, so a grid that has been trimmed,
    collapsed or is genuinely irregular (a severity's quantile grid, a
    cumulative probability) falls back to the explicit form on its own. The
    check costs one array comparison and removes the need for any caller to
    reason about whether its grid is still regular.
    """
    v = np.asarray(values, dtype=float)
    if step and v.size:
        start, count = float(v[0]), int(v.size)
        if np.array_equal(start + float(step) * np.arange(count), v):
            return {f'{prefix}_lattice': (start, float(step), count)}
    return {prefix: tuple(float(u) for u in v)}


def collapse_empty_runs(x, mass):
    """Drop the interior of every run of zero mass, keeping its endpoints.

    Parameters
    ----------
    x, mass : ndarray
        A grid and the probability at each point.

    Returns
    -------
    tuple of ndarray
        ``(x, mass)``, unchanged unless collapsing pays for itself.

    Notes
    -----
    The endpoints stay because they are what pins the drawing to the floor:
    delete a run outright and a stepped or straight line bridges the hole
    and draws mass where the law has none. Keeping one zero either side
    makes the collapse exact under every rung of the ladder rather than
    only under stems.

    Collapsing is not always worth doing, and the arithmetic says when. An
    untouched lattice ships as ``count`` masses plus three numbers for the
    grid; a collapsed one ships ``k`` masses **and** ``k`` explicit
    coordinates, because it is no longer arithmetic. So it pays exactly
    when ``2k < count``, which on a smooth book (a handful of empty buckets
    in 65,536) it never does, and on a lattice book (98% empty) it always
    does. No threshold was chosen: that is the break-even.
    """
    x = np.asarray(x, dtype=float)
    mass = np.asarray(mass, dtype=float)
    positive = mass > 0
    if positive.all():
        return x, mass
    keep = (positive | np.r_[False, positive[:-1]]
            | np.r_[positive[1:], False])
    if 2 * int(keep.sum()) >= keep.size:
        return x, mass
    return x[keep], mass[keep]
