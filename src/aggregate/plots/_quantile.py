"""Layer 1 content worker: the Lee / quantile panel.

A Lee diagram plots loss (outcome) on the vertical axis against the
non-exceedance probability ``p = F(loss)`` on the horizontal axis -- the
quantile function drawn "sideways". Shared by the ``Aggregate`` and
``Severity`` compositors, both of which draw the same ``(p, loss)`` curve and
differ only in draw style (stepped for discrete grids, smooth for continuous)
and the axis limits/labels their compositor sets afterwards.

Layer discipline: this renders into a **provided** ``Axes`` and never creates a
figure -- the compositor owns the canvas and sets the panel's limits, title and
legend.
"""

__all__ = ['plot_quantile']


def plot_quantile(ax, p, loss, **kwargs):
    """Draw one Lee/quantile curve ``(p, loss)`` into a provided ``Axes``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (created and configured by the compositor).
    p : array_like
        Non-exceedance probabilities ``F(loss)`` on the horizontal axis.
    loss : array_like
        Outcomes on the vertical axis.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.plot` (``label``, ``lw``,
        ``drawstyle``, ...).

    Returns
    -------
    list of matplotlib.lines.Line2D
        The drawn line(s), as returned by ``ax.plot``.
    """
    return ax.plot(p, loss, **kwargs)
