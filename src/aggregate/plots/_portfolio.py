"""What is left of the layer 2 compositor for :class:`Portfolio`.

``plot_portfolio`` is gone: ``Portfolio.plot`` draws the document
``charts.chart_port`` emits, whose second panel is the kappa reading rather
than a log density drawn as a picture of its own.

Holds the bodies of ``Portfolio.scatter`` (exeqa scatter matrix) and the
plotting half of ``Portfolio.sample_compare``, both listed as bespoke: a
scatter matrix is a family of panels the schema has no way to say, and the
sample comparison is a diagnostic rather than an exhibit. The class keeps
one-line delegating stubs, and the limit logic stays on it as
``Portfolio._limits`` (a data method).
"""

import pandas as pd

from ._style import make_mosaic, FIG_W, FIG_H


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
