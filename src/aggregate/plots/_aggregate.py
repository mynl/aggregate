"""What is left of the layer 2 compositor for :class:`Aggregate`.

``plot_aggregate`` is gone: ``Aggregate.plot`` draws the document
``charts.chart_agg`` emits, through ``plots._chartdoc``. Its three panels
became two, its two branches became one, and the discrete-or-continuous
question it asked (``bs == 1`` and a small mean) is now the renderer's
ladder reading the declared support and the room on screen.

What remains here is ``plot_pnl`` (until ``[Chart-PnL]``) and
``plot_reins_occ``. Both read the public ``density_df`` /
``reins_density_df`` surfaces and the object's grid metadata; the limit
logic stays on the class as ``Aggregate._limits`` (a data method, shared
with reporting).
"""

import logging

import numpy as np

from .._grid_distribution import GridDistribution
from ._style import make_grid
from ._quantile import plot_quantile

logger = logging.getLogger(__name__)


def plot_pnl(pnl, axd=None, **kwargs):
    """Net result density and distribution for a :class:`~aggregate.PnL`.

    Two panels -- the result (net) density (A) and distribution / CDF (B) --
    read from the P&L's exact result :class:`GridDistribution`
    (:attr:`PnL.result`). There is **no severity panel** (a P&L is an accounting
    object, not a compound of a severity), and no component overlay: the cession
    waterfall is the ledger's own rows. The break-even line at 0 is marked.

    Parameters
    ----------
    pnl : PnL
        The position to plot.
    axd : dict of str to Axes, optional
        Mosaic with keys ``'A'`` (density) and ``'B'`` (distribution). A new
        figure is created if omitted and stored on ``pnl.figure``.
    **kwargs
        Passed to the canvas creator (e.g. ``figsize``).
    """
    import numpy as np
    import pandas as pd
    if axd is None:
        # make_grid defaults to the house (2*FIG_W, FIG_H) for a 1x2 grid
        pnl.figure, axs = make_grid(1, 2, **kwargs)
        axd = {'A': axs[0], 'B': axs[1]}
    else:
        pnl.figure = axd['A'].figure

    gd = pnl.result                              # the exact net result GD
    label = pnl.result_name
    ser = gd.to_series(name=label)
    cdf = pd.Series(np.cumsum(gd.p), index=gd.x, name=label)
    ser.plot(ax=axd['A'], drawstyle='steps-mid', lw=2)
    cdf.plot(ax=axd['B'], drawstyle='steps-post', lw=2)
    axd['A'].set(title='Probability mass function', xlabel='P&L')
    for a in (axd['A'], axd['B']):
        a.axvline(0.0, lw=0.75, color='C7', ls='--')   # break-even reference
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
        fig, axs = make_grid(1, 2)
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
    # Each gross / ceded / net aggregate PMF becomes a cheap GD carrying the
    # aggregate's orientation; the worker derives the (p, loss) curve and the
    # return-period map off it. (Replaces the old hand-rolled survival + 1e-15
    # de-fuzz; the worker's saturating-top trim handles the flat tail instead.)
    loss = rd.loss.to_numpy()
    for c, col in [('gross', 'p_agg_gross'), ('ceded', 'p_agg_ceded_occ'),
                   ('net', 'p_agg_net_occ')]:
        gd = GridDistribution(loss, rd[col].to_numpy(), bs=agg.bs, name=c,
                              is_loss_value=agg._is_loss_value)
        plot_quantile(ax1, gd, label=c, **kwargs)
    ax1.legend()
