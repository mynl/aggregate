"""Layer 2 compositor for :class:`aggregate.portfolio.Portfolio`.

Holds the bodies of ``Portfolio.plot`` (density / log-density two-panel
exhibit), ``Portfolio.scatter`` (exeqa scatter matrix) and the plotting half of
``Portfolio.sample_compare``. The class keeps one-line delegating stubs.

The limit logic stays on the class as ``Portfolio._limits`` (a data method).
"""

import pandas as pd

from ._style import make_mosaic, FIG_W, FIG_H


def plot_portfolio(port, axd=None, figsize=(2 * FIG_W, FIG_H)):
    """Density and log-density two-panel plot of a :class:`Portfolio`.

    Draws the total and each unit on its native grid: linear density (A) and
    log density (B).

    Parameters
    ----------
    port : Portfolio
        The (updated) portfolio to plot.
    axd : dict of str to Axes, optional
        Mosaic with keys ``'A'`` and ``'B'``. A new figure is created if
        omitted and stored on ``port.figure``.
    figsize : tuple of float, default ``(2*FIG_W, FIG_H)``
        Figure size used when ``axd`` is None.
    """
    if axd is None:
        port.figure, axd = make_mosaic('AB', figsize=figsize)

    ax = axd['A']
    xl = port._limits()
    yl = port._limits(stat='density', zero_mass='exclude')
    # total first = Book standard, then each unit on its native grid
    # (numerics-1); on a legacy zero-origin book the grids coincide.
    bit = pd.concat(
        [port.density_df.p_total] +
        [port.unit_density(unit) for unit in port.unit_names], axis=1)
    bit.plot(ax=ax, xlim=xl, ylim=yl)
    ax.set(xlabel='Loss', ylabel='Density')
    ax.legend()

    ax = axd['B']
    xl = port._limits(kind='log')
    yl = port._limits(stat='logy')
    bit.plot(ax=ax, logy=True, xlim=xl, ylim=yl)
    ax.set(xlabel='Loss', ylabel='Log density')
    ax.legend().set(visible=False)


def plot_scatter(port, marker='.', s=5, alpha=1, figsize=(10, 10), diagonal='kde', **kwargs):
    """Scatter matrix of the per-unit ``exeqa`` marginals against one another.

    Designed for use with samples; uses :func:`pandas.plotting.scatter_matrix`
    on the ``exeqa_*`` columns.

    Parameters
    ----------
    port : Portfolio
        The portfolio to plot.
    marker, s, alpha, figsize, diagonal, **kwargs
        Forwarded to :func:`pandas.plotting.scatter_matrix`.

    Returns
    -------
    ndarray of matplotlib.axes.Axes
    """
    from pandas.plotting import scatter_matrix

    bit = port.density_df.query('p_total > 0').filter(regex='exeqa_[a-zA-Z]')
    ax = scatter_matrix(bit, marker='.', s=5, alpha=1,
                        figsize=(10, 10), diagonal='kde', **kwargs)
    return ax


def plot_sample_compare(port, ax=None):
    """Compare the sample-based portfolio total to the independent marginal sum.

    Plots the two survival curves on ``ax`` (if provided) and returns the
    side-by-side ``('total', 'empirical')`` moment frame for the independent
    and sample views.

    Parameters
    ----------
    port : Portfolio
        Must have an ``independent_density_df`` (i.e. a sample has been built).
    ax : matplotlib.axes.Axes, optional
        If given, overlay the independent and sample survival functions.

    Returns
    -------
    pandas.DataFrame
        Concatenated independent / sample moment columns.
    """
    if port.independent_density_df is None:
        raise ValueError('No independent_density_df, cannot compare')

    if ax is not None:
        ax.plot(port.independent_density_df.index, port.independent_density_df['S'], lw=1, label='independent')
        ax.plot(port.density_df.index, port.density_df['S'], lw=1, label='sample')
        ax.legend()

    return pd.concat(
        (port.independent_stats_df[['total', 'empirical']],
         port.stats_df[['total', 'empirical']]),
        keys=['independent', 'sample'], axis=1,
    )
