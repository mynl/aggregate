"""Layer 1 content worker: the Lee / quantile panel.

A Lee diagram plots loss (outcome) on the vertical axis against the
non-exceedance probability ``p = F(loss)`` on the horizontal axis -- the
quantile function drawn "sideways". The worker consumes a
:class:`~aggregate._grid_distribution.GridDistribution` (GD): the curve is the
GD's own ``(cumsum(p), x)`` and the orientation (which tail is "bad") is read
off ``gd.is_loss_value`` -- no side-channel flag threaded through the
compositors. Shared by the ``Aggregate`` and ``Severity`` compositors, which
differ only in draw style (stepped for discrete grids, smooth for continuous)
passed through ``**kwargs``.

The worker also owns the **return-period** transform (``quantile_x='return'``):
instead of plotting against ``p`` on a linear axis it plots against the return
period ``T`` on a log axis, which spreads the rare (bad) tail so it can be read
off directly. The GD supplies the map via :meth:`GridDistribution.return_period`,
which branches on the GD's own role:

- **loss** (``gd.is_loss_value`` True): ``T = 1 / (1 - p)`` -- large ``p`` (the
  big-loss tail) maps to large ``T``.
- **payoff** (``gd.is_loss_value`` False, e.g. a ``PnL`` net): ``T = 1 / p`` --
  small ``p`` (the low-payoff / P&L-loss tail) maps to large ``T``.

Layer discipline: the worker renders into a **provided** ``Axes`` and never
creates a figure. It owns the *content* (the ``(p, loss)`` curve derived from
the GD, trimmed at the saturating top of the quantile function) and -- in return
mode -- the return-period *x* axis (scale, decade ticks, "Return period" label,
the ``MAX_RETURN_PERIOD`` cap, and the y rescale the cap forces). The compositor
sets the linear-mode panel limits, label, and title *before* calling the worker;
in linear mode the worker leaves those untouched, so the two modes are selected
without the compositor ever branching on ``quantile_x`` -- it simply forwards
the keyword through ``**kwargs``.
"""

import numpy as np

from ._style import ticker

__all__ = ['plot_quantile', 'MAX_RETURN_PERIOD']

#: Largest return period drawn in ``quantile_x='return'`` mode. The quantile
#: function saturates as ``p -> 1`` (loss) / ``p -> 0`` (payoff), where ``T``
#: diverges; capping at a billion-year event keeps both axes finite and the
#: outcome (y) axis bounded by the deepest plotted point.
MAX_RETURN_PERIOD = 1e9


def plot_quantile(ax, gd, quantile_x='linear', max_return_period=MAX_RETURN_PERIOD,
                  **kwargs):
    """Draw one Lee/quantile curve for a ``GridDistribution`` into a provided ``Axes``.

    The curve is the GD's quantile function drawn sideways: outcome ``gd.x`` on
    the vertical axis against the non-exceedance probability ``p = cumsum(gd.p)``
    on the horizontal axis. The orientation that the ``'return'`` transform needs
    is read off the GD itself (``gd.is_loss_value`` via :meth:`gd.return_period`),
    so the caller passes a self-describing object rather than a side-channel flag.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (created and pre-configured by the compositor).
    gd : GridDistribution
        The distribution to draw. The curve is ``(cumsum(gd.p), gd.x)``; the
        return-period branch reads ``gd.is_loss_value``.
    quantile_x : {'linear', 'return'}, default 'linear'
        ``'linear'`` plots outcome against ``p`` on the axis as the compositor
        configured it (the classic Lee diagram). ``'return'`` plots against the
        return period ``T`` on a log axis, capped at ``max_return_period`` -- see
        the module docstring for the loss/payoff branches.
    max_return_period : float, default ``MAX_RETURN_PERIOD`` (1e9)
        Return-period cap. Points beyond it (the saturating ``p -> 1`` / ``p ->
        0`` endpoint, where ``T`` diverges) are dropped so both axes stay
        finite. Ignored when ``quantile_x='linear'``.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.plot` (``label``, ``lw``,
        ``drawstyle``, ...).

    Returns
    -------
    list of matplotlib.lines.Line2D
        The drawn line(s), as returned by ``ax.plot``.

    Notes
    -----
    The curve is trimmed at the **saturating top** of the quantile function:
    once ``p`` reaches its maximum the remaining grid points are zero-mass
    buckets that would shoot the Lee line up to the largest grid value, so the
    worker keeps up to (and including) the first point at the max -- the same
    trim the compositors used to apply via ``df.loc[:F.idxmax()]``.

    In return mode the cap is applied as ``T <= max_return_period`` (i.e.
    ``p <= 1 - 1/max_return_period`` for a loss, ``p >= 1/max_return_period``
    for a payoff). The comparison drops the diverging endpoint and any ``NaN``
    gaps (so pre-trimmed survival curves pass through cleanly), and the x axis
    is log-scaled with decade ticks. The cap bounds the outcome (y) axis too:
    after plotting, the worker re-enables autoscaling so y fits the deepest
    *plotted* point rather than the (now off-screen) saturating tail.
    """
    if quantile_x not in ('linear', 'return'):
        raise ValueError(
            f"quantile_x must be 'linear' or 'return', got {quantile_x!r}")

    loss = np.asarray(gd.x, dtype=float)
    p = np.cumsum(np.asarray(gd.p, dtype=float))    # non-exceedance F at each grid point
    # Trim the saturating top of the quantile function: keep up to (and
    # including) the first point at the max, dropping the flat zero-mass tail
    # that would otherwise run the Lee line up to the largest grid value.
    if len(p):
        cut = int(np.argmax(p >= p.max())) + 1
        p, loss = p[:cut], loss[:cut]

    if quantile_x == 'linear':
        # The compositor already configured this axis; just draw.
        return ax.plot(p, loss, **kwargs)

    # The loss/payoff branch lives on the GD (gd.return_period -> shared
    # return_period_map); the worker reads the same map rather than re-deriving
    # 1/(1-p) vs 1/p.
    t = gd.return_period(p)
    # Cap directly on T: T <= max drops the diverging endpoint, and any NaN gap
    # (nan <= max is False) too, so a pre-trimmed survival curve passes cleanly.
    keep = t <= max_return_period
    lines = ax.plot(t[keep], loss[keep], **kwargs)

    # Own the x axis: log scale, decade ticks, label. Then re-enable
    # autoscaling so the cap also bounds y to the deepest plotted outcome
    # (the compositor's linear y-limit was set for the p axis, not this one).
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.set_xlabel('Return period')
    ax.relim()
    ax.autoscale(enable=True, axis='both')
    return lines
