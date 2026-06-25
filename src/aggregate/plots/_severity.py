"""Layer 2 compositor for :class:`aggregate.distributions.Severity`.

Holds the body of ``Severity.plot`` -- a four-panel ``'AB\\nCD'`` mosaic of
density (A), log density (B), distribution (C) and the Lee/quantile diagram
(D). The class keeps only a one-line delegating stub.

Severity has no ``_df`` surface (it is a function of ``x``, not a discretised
grid), so the panels evaluate ``_pdf`` / ``_cdf`` / ``_isf`` on a sampling grid
-- the rare in-§2 case where the plot is bound to the object's own callables.
The Lee panel reuses the shared :func:`aggregate.plots._quantile.plot_quantile`
worker.
"""

import numpy as np

from .._grid_distribution import GridDistribution
from ._style import make_mosaic, FIG_W, FIG_H
from ._quantile import plot_quantile


def plot_severity(sev, n=100, axd=None, figsize=(2 * FIG_W, 2 * FIG_H), layout='AB\nCD',
                  **kwargs):
    """Quick four-panel plot of a :class:`Severity`.

    Density (A), log density (B), distribution (C) and Lee/quantile (D), each
    evaluated on a linear grid out to a deep survival quantile.

    Parameters
    ----------
    sev : Severity
        The object to plot.
    n : int, default 100
        Number of points to plot per panel.
    axd : dict of str to Axes, optional
        Mosaic with keys ``'A'``, ``'B'``, ``'C'``, ``'D'``. A new figure is
        created if omitted.
    figsize : tuple of float, default ``(2*FIG_W, 2*FIG_H)``
        Figure size used when ``axd`` is None.
    layout : str, default ``'AB\\nCD'``
        Mosaic layout passed to the canvas creator.
    **kwargs
        Lee-panel (D) options forwarded to
        :func:`aggregate.plots._quantile.plot_quantile` -- notably
        ``quantile_x={'linear', 'return'}`` and ``max_return_period``. A
        severity is a loss value, so the return-period branch is ``1/(1-p)``.
    """
    from ..distributions import SeverityDHistogram

    xs = np.linspace(0, sev._isf(1e-4), n)
    xs2 = np.linspace(0, sev._isf(1e-12), n)

    if axd is None:
        _, axd = make_mosaic(layout, figsize=figsize)

    # ``fixed`` is a degenerate dhistogram (single point mass); both
    # benefit from the step-post draw style. Continuous histograms and
    # the scipy zoo use ordinary line plots.
    ds = 'steps-post' if isinstance(sev, SeverityDHistogram) else 'default'

    ax = axd['A']
    ys = sev._pdf(xs)
    ax.plot(xs, ys, drawstyle=ds)
    ax.set(title='Probability density', xlabel='Loss')
    yl = ax.get_ylim()

    ax = axd['B']
    ys2 = sev._pdf(xs2)
    ax.plot(xs2, ys2, drawstyle=ds)
    ax.set(title='Log density', xlabel='Loss', yscale='log', ylim=[1e-14, 2 * yl[1]])

    ax = axd['C']
    ys = sev._cdf(xs)
    ax.plot(xs, ys, drawstyle=ds)
    ax.set(title='Probability distribution', xlabel='Loss', ylim=[-0.025, 1.025])

    # Configure the linear panel first; the worker leaves it for
    # ``quantile_x='linear'`` and overrides it for ``'return'``.
    ax = axd['D']
    ax.set(title='Quantile (Lee) plot', xlabel='Non-exceeding probability $p$',
           xlim=[-0.025, 1.025])
    # A severity has no discrete grid of its own, so wrap the plot-grid cdf as a
    # throwaway GD whose ``cumsum(p)`` reconstructs ``F = cdf(xs)``. A standalone
    # Severity carries no loss/payoff orientation (only ``self.signed``, the
    # support-sign axis), so default to loss (True); a severity drawn inside an
    # Aggregate instead inherits the aggregate's role via
    # ``_sev_grid_distribution()``.
    gd = GridDistribution(xs, np.diff(ys, prepend=0.0), name=sev.name,
                          is_loss_value=True)
    plot_quantile(ax, gd, drawstyle=ds, **kwargs)
