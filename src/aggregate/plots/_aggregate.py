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
        Lee-panel (C) options forwarded to
        :func:`aggregate.plots._quantile.plot_quantile` -- notably
        ``quantile_x={'linear', 'return'}`` (``'return'`` plots panel C against
        log return period) and ``max_return_period``. ``figsize`` is consumed
        here for the canvas.
    """
    figsize = kwargs.pop('figsize', (3 * FIG_W, FIG_H))
    if axd is None:
        agg.figure, axd = make_mosaic('ABC', figsize=figsize)
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

        # for Lee diagrams. Configure the linear panel first; the worker leaves
        # it as-is for ``quantile_x='linear'`` and overrides it for ``'return'``.
        ax = axd['C']
        ax.set(xlim=[-0.025, 1.025], ylim=[left, mx + 1], title='Quantile (Lee) plot')
        # trim so that the Lee plot doesn't spuriously tend up to infinity
        # little care: may not exaclty equal 1
        idx = (df.F == df.F.max()).idxmax()
        dft = df.loc[:idx]
        # Panel orientation comes from the aggregate's self-describing GD; the
        # severity curve follows the aggregate role (it shares this panel).
        lee_is_loss = agg._grid_distribution().is_loss_value
        plot_quantile(ax, dft.F, dft.loss, is_loss_value=lee_is_loss,
                      drawstyle='steps-pre', lw=3, label='Aggregate', **kwargs)
        # same trim for severity (on its own grid)
        sidx = (sdf.F_sev >= 1).idxmax()
        sdft = sdf.loc[:sidx]
        plot_quantile(ax, sdft.F_sev, sdft.loss, is_loss_value=lee_is_loss,
                      drawstyle='steps-pre', lw=1, label='Severity', **kwargs)
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

        # Configure the linear panel first; the worker leaves it for
        # ``quantile_x='linear'`` and overrides it for ``'return'``.
        ax = axd['C']
        ax.set(xlim=[-0.02, 1.02], ylim=xlim, title='Quantile (Lee) plot',
               xlabel='Non-exceeding probability p')
        # to do: same trimming for p-->1 needed?
        # Panel orientation from the aggregate GD; severity follows (shared panel).
        lee_is_loss = agg._grid_distribution().is_loss_value
        plot_quantile(ax, df.F, df.loss, is_loss_value=lee_is_loss,
                      lw=2, label='Aggregate', **kwargs)
        plot_quantile(ax, sdf.F_sev, sdf.loss, is_loss_value=lee_is_loss,
                      lw=1, label='Severity', **kwargs)
        ax.legend().set(visible=False)


def plot_pnl(pnl, axd=None, **kwargs):
    """Net P&L (Margin) density and distribution for a :class:`PnL`.

    Two panels -- the Margin density (A) and distribution (B) -- read from
    ``pnl.pnl_df``. There is **no severity panel**: a P&L is an affine of its
    aggregate, not a compound of a severity (that ``d/dx`` panel belongs to an
    :class:`Aggregate`; plot the bare risky leg via ``pnl.agg.plot()``). The
    break-even line at 0 is marked. For a Gross/Ceded/Net position
    (``make_pnl(gross=, ceded=)``) the three legs' margins are overlaid, ``Net``
    the heavy line.

    Parameters
    ----------
    pnl : PnL
        The (updated) position to plot.
    axd : dict of str to Axes, optional
        Mosaic with keys ``'A'`` (density) and ``'B'`` (distribution). A new
        figure is created if omitted and stored on ``pnl.figure``.
    **kwargs
        Passed to the canvas creator (e.g. ``figsize``).
    """
    if axd is None:
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (2 * FIG_W, FIG_H)
        pnl.figure, axs = make_grid(1, 2, **kwargs)
        axd = {'A': axs[0], 'B': axs[1]}
    else:
        pnl.figure = axd['A'].figure

    bs = pnl.agg.bs
    discrete = bs == 1 and abs(pnl.mean) < 1025
    ax = axd['A']

    # The legs to draw: Net only (single leg), or Gross/Ceded/Net overlaid.
    if pnl._gcn is not None:
        xs = pnl.agg.xs
        pg, pc = pnl._gcn['gross'], pnl._gcn['ceded']
        legs = [
            ('Gross', pnl._frame_from(pg - xs, pnl.agg.agg_density_gross), 1),
            ('Ceded', pnl._frame_from(xs - pc, pnl.agg.agg_density_ceded), 1),
            ('Net', pnl.pnl_df, 2.5),
        ]
    else:
        legs = [('Margin', pnl.pnl_df, 2)]

    for label, f, lw in legs:
        if discrete:
            f.p_total.plot(ax=ax, drawstyle='steps-mid', lw=lw, label=label)
            f.F.plot(ax=axd['B'], drawstyle='steps-post', lw=lw, label=label)
        else:
            (f.p_total / bs).plot(ax=ax, lw=lw, label=label)
            f.F.plot(ax=axd['B'], lw=lw, label=label)
    ax.set(title='Probability mass function' if discrete else 'Probability density',
           xlabel='P&L')
    # break-even reference (a P&L can be a loss)
    for a in (ax, axd['B']):
        a.axvline(0.0, lw=0.75, color='C7', ls='--')
    ax.legend()
    axd['B'].set(title='Distribution function', xlabel='P&L')
    return pnl.figure


def plot_reins_occ(agg, axs=None, **kwargs):
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
    **kwargs
        Lee-panel (right) options forwarded to
        :func:`aggregate.plots._quantile.plot_quantile` -- notably
        ``quantile_x={'linear', 'return'}`` and ``max_return_period``.
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

    # Configure the linear panel first; the worker leaves it for
    # ``quantile_x='linear'`` and overrides it for ``'return'``.
    ax1.set(xlabel='Probability of non-exceedance', ylabel='Loss', title='Aggregate')
    y = rd.loss.values
    lee_is_loss = agg._grid_distribution().is_loss_value
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
        plot_quantile(ax1, 1 - s_values, y, is_loss_value=lee_is_loss,
                      label=c, **kwargs)
    ax1.legend()
