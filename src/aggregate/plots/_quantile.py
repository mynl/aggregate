"""Layer 1 content worker: the Lee / quantile panel.

A Lee diagram plots loss (outcome) on the vertical axis against the
non-exceedance probability ``p = F(loss)`` on the horizontal axis -- the
quantile function drawn "sideways". Shared by the ``Aggregate`` and
``Severity`` compositors, both of which draw the same ``(p, loss)`` curve and
differ only in draw style (stepped for discrete grids, smooth for continuous).

The worker also owns the **return-period** transform (``quantile_x='return'``):
instead of plotting against ``p`` on a linear axis it plots against the return
period ``T`` on a log axis, which spreads the rare (bad) tail so it can be read
off directly. The bad tail sits at large ``p`` for a loss value and at small
``p`` for a payoff, so the transform branches on the value-type role:

- **loss** (``is_loss_value`` True): ``T = 1 / (1 - p)`` -- large ``p`` (the
  big-loss tail) maps to large ``T``.
- **payoff** (``is_loss_value`` False, e.g. a ``PnL`` net): ``T = 1 / p`` --
  small ``p`` (the low-payoff / P&L-loss tail) maps to large ``T``.

Layer discipline: the worker renders into a **provided** ``Axes`` and never
creates a figure. It owns only the *content* and -- in return mode -- the
return-period *x* axis (scale, decade ticks, "Return period" label, the
``MAX_RETURN_PERIOD`` cap, and the y rescale the cap forces). The compositor
sets the linear-mode panel limits, label, and title *before* calling the worker;
in linear mode the worker leaves those untouched, so the two modes are selected
without the compositor ever branching on ``quantile_x`` -- it simply forwards
the keyword through ``**kwargs``.
"""

import numpy as np

from .._grid_distribution import return_period_map
from ._style import ticker

__all__ = ['plot_quantile', 'MAX_RETURN_PERIOD']

#: Largest return period drawn in ``quantile_x='return'`` mode. The quantile
#: function saturates as ``p -> 1`` (loss) / ``p -> 0`` (payoff), where ``T``
#: diverges; capping at a billion-year event keeps both axes finite and the
#: outcome (y) axis bounded by the deepest plotted point.
MAX_RETURN_PERIOD = 1e9


def plot_quantile(ax, p, loss, quantile_x='linear', is_loss_value=True,
                  max_return_period=MAX_RETURN_PERIOD, **kwargs):
    """Draw one Lee/quantile curve ``(p, loss)`` into a provided ``Axes``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (created and pre-configured by the compositor).
    p : array_like
        Non-exceedance probabilities ``F(loss)`` on the horizontal axis.
    loss : array_like
        Outcomes on the vertical axis.
    quantile_x : {'linear', 'return'}, default 'linear'
        ``'linear'`` plots outcome against ``p`` on the axis as the compositor
        configured it (the classic Lee diagram). ``'return'`` plots against the
        return period ``T`` on a log axis, capped at ``max_return_period`` -- see
        the module docstring for the loss/payoff branches.
    is_loss_value : bool, default True
        Value-type role used by the ``'return'`` transform: ``True`` (loss)
        maps the large-``p`` tail to large ``T`` via ``T = 1 / (1 - p)``;
        ``False`` (payoff) maps the small-``p`` tail via ``T = 1 / p``. Ignored
        when ``quantile_x='linear'``.
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

    if quantile_x == 'linear':
        # The compositor already configured this axis; just draw.
        return ax.plot(p, loss, **kwargs)

    p = np.asarray(p, dtype=float)
    loss = np.asarray(loss, dtype=float)
    # The loss/payoff branch lives on the GridDistribution (return_period_map);
    # the worker reads the same map rather than re-deriving 1/(1-p) vs 1/p.
    t = return_period_map(p, is_loss_value)
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
