"""Layer 2 compositor for :class:`aggregate.bivariate.BivariateAggregate`.

Holds the body of ``BivariateAggregate.plot`` -- a two-panel contour exhibit
of the per-claim severity (left) and the joint aggregate (right) -- plus the
single filled-contour content worker it calls. The class keeps a one-line
delegating stub.

Single-consumer content: the 2-D contour panel has no cross-class reuse, so its
worker (:func:`_contourf`) lives here alongside the compositor (the §1
single-consumer case).
"""

import numpy as np

from ._style import make_grid


def _contourf(ax, xgrid, ygrid, Z, title, xlabel, ylabel, levels, log,
              **kwargs):
    """Single filled-contour panel of a 2D density on ``(xgrid, ygrid)``."""
    xx, yy = np.meshgrid(xgrid, ygrid)
    z = Z.T   # density indexed [axis0, axis1]; contourf wants Z[row=y, col=x]
    if log:
        pos = z[z > 0]
        floor = pos.min() if pos.size else 1e-300
        z = np.log10(np.maximum(z, floor))
    ax.contourf(xx, yy, z, levels=levels, **kwargs)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return ax


def plot_bivariate(biv, axs=None, levels=14, log=False, **kwargs):
    """Two-panel contour plot: per-claim severity (left), aggregate (right).

    Parameters
    ----------
    biv : BivariateAggregate
        The joint object to plot; must have a computed density.
    axs : array of matplotlib Axes, optional
        Two target axes (e.g. ``ax0, ax1 = axs.flat``). A new ``1 x 2`` figure
        is created if omitted; the figure is stored on ``biv.figure``.
    levels : int, default 14
        Number of contour levels.
    log : bool, default False
        Contour ``log10`` of each density (useful for heavy tails).
    **kwargs
        Passed through to ``Axes.contourf``.

    Returns
    -------
    None
        The drawn axes are reachable via ``biv.figure.axes`` (or the ``axs`` you
        passed in). Returning ``None`` -- matching ``Aggregate.plot`` -- avoids
        the Jupyter inline backend rendering the figure twice.

    Notes
    -----
    The left panel is the joint **per-claim severity** ``S`` (the copula
    coupling -- or the comonotone ``(ceded, net)`` scatter in ``netceded`` mode
    -- on the severity grids); the right panel is the joint **aggregate**
    density (on the output grids, P&L-relabelled for any ``pnl`` axis).
    """
    biv._require_density()
    if axs is None:
        biv.figure, axs = make_grid(1, 2, squeeze=True)
    else:
        biv.figure = np.asarray(axs).flat[0].figure
    ax0, ax1 = np.asarray(axs).flat[:2]
    n0, n1 = biv.unit_names
    _contourf(ax0, biv._sev_xs[0], biv._sev_xs[1], biv._S,
              'severity', n0, n1, levels, log, **kwargs)
    _contourf(ax1, biv.axis_xs[0], biv.axis_xs[1], biv.density,
              'aggregate', n0, n1, levels, log, **kwargs)


def plot_bivariate_distribution(bd, ax=None, levels=14, log=False, **kwargs):
    """Filled contour plot of a :class:`BivariateDistribution` joint density.

    Parameters
    ----------
    bd : BivariateDistribution
        The joint (ceded, net) distribution to plot.
    ax : matplotlib Axes, optional
        Target axes; a new figure is created if omitted (using the project
        ``FIG_W`` / ``FIG_H`` and constrained layout).
    levels : int, default 14
        Number of contour levels.
    log : bool, default False
        Contour ``log10`` of the density (clipped at the smallest positive
        value) -- useful for the heavy-tailed dependency structure.
    **kwargs
        Passed through to ``Axes.contourf``.

    Returns
    -------
    matplotlib Axes
        The axes drawn on.
    """
    if ax is None:
        _, ax = make_grid(1, 1, squeeze=True)
    names = bd.meta.get('axis_names', ('ceded', 'net'))
    return _contourf(ax, bd.ceded, bd.net, bd.density,
                     f'Joint density\n{bd.meta.get("name", "")}',
                     f'Aggregate {names[0]}', f'Aggregate {names[1]}',
                     levels, log, **kwargs)
