"""Layer 2 compositor for :class:`aggregate.distributions.Aggregate`.

Holds the bodies of ``Aggregate.plot`` (the canonical density / distribution /
Lee three-panel exhibit) and ``Aggregate.reins_occ_plot`` (occurrence
reinsurance). The class keeps only one-line delegating stubs.

These read the public ``density_df`` / ``sev_density_df`` / ``reins_density_df``
surfaces and the object's grid metadata; the limit logic stays on the class as
``Aggregate._limits`` (it is a data method, shared with reporting).
"""

import logging

import numpy as np

from ..utilities import nice_multiple
from ._style import plt, ticker, make_mosaic, make_grid, FIG_W, FIG_H
from ._quantile import plot_quantile

logger = logging.getLogger(__name__)


def plot_aggregate(agg, axd=None, xmax=0, **kwargs):
    """Density, distribution, and Lee (quantile) plot for an :class:`Aggregate`.

    Lays out an ``'ABC'`` mosaic -- probability mass/density (A), distribution
    or log density (B), and the Lee/quantile diagram (C) -- drawing the
    aggregate and its severity on each. Discrete (``bs == 1``, small mean) and
    continuous objects take separate branches.

    Parameters
    ----------
    agg : Aggregate
        The object to plot; must be updated.
    axd : dict of str to Axes, optional
        Mosaic with keys ``'A'``, ``'B'``, ``'C'``. A new figure is created if
        omitted and stored on ``agg.figure``.
    xmax : float, default 0
        Hint for the x-axis upper limit, e.g. to put gross and net on a common
        scale. Only used on linear scales.
    **kwargs
        Passed to the canvas creator (e.g. ``figsize``).
    """
    if axd is None:
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (3 * FIG_W, FIG_H)
        agg.figure, axd = make_mosaic('ABC', **kwargs)
    else:
        agg.figure = axd['A'].figure

    if agg.bs == 1 and abs(agg.est_m) < 1025:
        # treat as discrete
        if xmax > 0:
            mx = xmax
        else:
            mx = agg.q(1) * 1.05
        # Window-aware left edge: an ordinary 0-based aggregate anchors at 0
        # (unchanged); a windowed grid anchors below the mass instead of
        # clipping at ~0 or padding empty space. Signed P&L keeps its exact
        # ``min(q(0), 0)`` floor; a thin-tailed window (origin > 0) anchors at
        # the true support minimum.
        if agg.xs is not None and agg.xs[0] < 0:
            mn = min(float(agg.q(0)), 0.0)
        elif agg.xs is not None and agg.xs[0] > 0:
            mn = float(agg.q(0))
        else:
            mn = 0.0
        span = nice_multiple(mx - mn)
        left = mn - (mx - mn) / 25

        # Aggregate from density_df; severity from its own grid
        # (sev_density_df), which may differ from the aggregate window.
        df = agg.density_df[['p_total', 'F', 'loss']].copy()
        # anchor a zero-mass row just left of the support so the steps/stems
        # start from the baseline (at mn - 0.5, not a fixed -0.5). ``loss``
        # must equal the row's own index, not 0: the Lee plot (panel C) plots
        # ``loss`` against ``F``, so a stray ``loss=0`` here would draw a
        # spurious vertical segment from 0 down to the first (signed) point.
        df.loc[mn - 0.5, :] = (0, 0, mn - 0.5)
        df = df.sort_index()
        sdf = agg.sev_density_df
        if mx <= 60:
            # stem plot for small means
            axd['A'].stem(df.index, df.p_total, basefmt='none', linefmt='C0-', markerfmt='C0.', label='Aggregate')
            axd['A'].stem(sdf.loss, sdf.p_sev, basefmt='none', linefmt='C1-', markerfmt='C1,', label='Severity')
        else:
            df.p_total.plot(ax=axd['A'], drawstyle='steps-mid', lw=2, label='Aggregate')
            sdf.p_sev.plot(ax=axd['A'], drawstyle='steps-mid', lw=1, label='Severity')

        axd['A'].set(xlim=[left, mx + 1], title='Probability mass functions')
        axd['A'].legend()
        if span > 0:
            axd['A'].xaxis.set_major_locator(ticker.MultipleLocator(span))
        # for discrete plot F next
        df.F.plot(ax=axd['B'], drawstyle='steps-post', lw=2, label='Aggregate')
        sdf.F_sev.plot(ax=axd['B'], drawstyle='steps-post', lw=1, label='Severity')
        axd['B'].set(xlim=[left, mx + 1], title='Distribution functions')
        axd['B'].legend().set(visible=False)
        if span > 0:
            axd['B'].xaxis.set_major_locator(ticker.MultipleLocator(span))

        # for Lee diagrams
        ax = axd['C']
        # trim so that the Lee plot doesn't spuriously tend up to infinity
        # little care: may not exaclty equal 1
        idx = (df.F == df.F.max()).idxmax()
        dft = df.loc[:idx]
        plot_quantile(ax, dft.F, dft.loss, drawstyle='steps-pre', lw=3, label='Aggregate')
        # same trim for severity (on its own grid)
        sidx = (sdf.F_sev >= 1).idxmax()
        sdft = sdf.loc[:sidx]
        plot_quantile(ax, sdft.F_sev, sdft.loss, drawstyle='steps-pre', lw=1, label='Severity')
        ax.set(xlim=[-0.025, 1.025], ylim=[left, mx + 1], title='Quantile (Lee) plot')
        ax.legend().set(visible=False)
    else:
        # continuous
        df = agg.density_df
        sdf = agg.sev_density_df       # severity on its own grid
        if xmax > 0:
            xlim = [-xmax / 50, xmax * 1.025]
        else:
            xlim = agg._limits(stat='range', kind='linear')
        xlim2 = agg._limits(stat='range', kind='log')
        ylim = agg._limits(stat='density')

        ax = axd['A']
        # divide by bucket size...approximating the density
        (df.p_total / agg.bs).plot(ax=ax, lw=2, label='Aggregate')
        (sdf.p_sev / agg.bs).plot(ax=ax, lw=1, label='Severity')
        ax.set(xlim=xlim, ylim=ylim, title='Probability density')
        ax.legend()

        (df.p_total / agg.bs).plot(ax=axd['B'], lw=2, label='Aggregate')
        (sdf.p_sev / agg.bs).plot(ax=axd['B'], lw=1, label='Severity')
        ylim = axd['B'].get_ylim()
        ylim = [1e-15, ylim[1] * 2]
        axd['B'].set(xlim=xlim2, ylim=ylim, title='Log density', yscale='log')
        axd['B'].legend().set(visible=False)

        ax = axd['C']
        # to do: same trimming for p-->1 needed?
        plot_quantile(ax, df.F, df.loss, lw=2, label='Aggregate')
        plot_quantile(ax, sdf.F_sev, sdf.loss, lw=1, label='Severity')
        ax.set(xlim=[-0.02, 1.02], ylim=xlim, title='Quantile (Lee) plot', xlabel='Non-exceeding probability p')
        ax.legend().set(visible=False)


def plot_reins_occ(agg, axs=None):
    """Occurrence-reinsurance plot: occurrence log density and aggregate Lee.

    Two panels -- occurrence log density (gross / ceded / net, left) and the
    aggregate quantile plot (right) -- read from ``reins_density_df``.

    Parameters
    ----------
    agg : Aggregate
        Must carry occurrence reinsurance (``occ_reins`` not ``None``).
    axs : array of Axes, optional
        Two target axes; a new ``1 x 2`` figure is created if omitted and
        stored on ``agg.figure``.
    """
    if agg.occ_reins is None:
        logger.warning('reins_occ_plot called with no occurrence reinsurance.')
        return
    if axs is None:
        fig, axs = make_grid(1, 2, figsize=(2 * FIG_W, FIG_H))
        agg.figure = fig
    ax0, ax1 = axs.flat

    rd = agg.reins_density_df
    rd[['p_sev_gross', 'p_sev_ceded', 'p_sev_net']].rename(
        columns={'p_sev_gross': 'gross', 'p_sev_ceded': 'ceded',
                 'p_sev_net': 'net'}).plot(ax=ax0, logy=True)
    xl = ax0.get_xlim()
    l = agg.spec['exp_limit']
    if type(l) != float:
        l = np.max(l)
    if l < np.inf:
        xl = [-l / 50, l * 1.025]
    ax0.set(xlim=xl, xlabel='Loss', ylabel='Occurrence log density', title='Occurrence')

    y = rd.loss.values
    for c, col in [('gross', 'p_agg_gross'), ('ceded', 'p_agg_ceded_occ'),
                   ('net', 'p_agg_net_occ')]:
        # Plot-cosmetic de-fuzz: deliberate carve-out from the shared
        # ``remove_fuzz`` -- a looser 1e-15 threshold plus the 0 -> nan step
        # below so empty buckets drop out of the survival line.
        s = rd[col].to_numpy().copy()
        s[np.abs(s) < 1e-15] = 0
        s_values = s[::-1].cumsum()[::-1]
        s_values = np.where(np.abs(s_values) < 1e-15, 0, s_values)
        s_values = np.where(s_values == 0, np.nan, s_values)
        ax1.plot(1 - s_values, y, label=c)
    ax1.set(xlabel='Probability of non-exceedance', ylabel='Loss', title='Aggregate')
    ax1.legend()
