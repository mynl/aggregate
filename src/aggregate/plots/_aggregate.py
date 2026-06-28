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

from .._grid_distribution import GridDistribution
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
        # Both curves are self-describing GDs: the worker derives (p, loss) and
        # trims the saturating top, and reads orientation off the GD. The
        # severity GD carries the aggregate's role (shared panel, one tail).
        # The aggregate curve wraps the *anchored* ``df`` (the zero-mass baseline
        # row at ``mn - 0.5`` shared with panels A/B) so the steps-pre Lee line
        # rises from the baseline at its own signed index, not a stray ``loss=0``.
        agg_gd = GridDistribution(
            df.loss.to_numpy(), df.p_total.to_numpy(), bs=agg.bs, name=agg.name,
            is_loss_value=agg._is_loss_value)
        plot_quantile(ax, agg_gd,
                      drawstyle='steps-pre', lw=3, label='Aggregate', **kwargs)
        plot_quantile(ax, agg._sev_grid_distribution(),
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
        # Self-describing GDs: the worker derives (p, loss), trims the saturating
        # top, and reads orientation off the GD. Severity follows the aggregate
        # role (shared panel).
        plot_quantile(ax, agg._grid_distribution(),
                      lw=2, label='Aggregate', **kwargs)
        plot_quantile(ax, agg._sev_grid_distribution(),
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

    # The legs to draw: Net only (single leg), or the retained waterfall
    # ``Gross -> [Net occ] -> Net`` overlaid. The gross / net-of-occurrence
    # aggregate loss distributions live in ``reins_density_df`` -- occurrence
    # reinsurance does **not** populate the ``agg_density_*`` attributes (those
    # are aggregate-reins only), which is why an occurrence-only GCN P&L
    # previously dropped its gross / net-occ legs. Each leg is positioned by the
    # same per-perspective premium / expense the GCN table uses (``_gcn_magnitudes``)
    # so the curves and the exhibit means agree; the final ``Net`` leg is the
    # canonical net P&L (``pnl_df``, matching ``summary_df`` / ``q`` / ``cdf``).
    rd = pnl.agg.reins_density_df
    if pnl._gcn is not None and rd is not None:
        prem_mag, exp_mag, has_occ, has_agg = pnl._gcn_magnitudes()
        xs = rd['loss'].to_numpy()
        sign = -1.0 if pnl.agg._is_loss_value else 1.0

        def _stage(persp, col):
            margin = prem_mag[persp] + sign * xs - exp_mag[persp]
            return pnl._frame_from(margin, rd[col].to_numpy())

        legs = [('Gross', _stage('gross', 'p_agg_gross'), 1)]
        if has_occ and has_agg:
            legs.append(('Net occ', _stage('net_occ', 'p_agg_net_occ'), 1))
        legs.append(('Net', pnl.pnl_df, 2.5))
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


def plot_reinstatement(an, axd=None, **kwargs):
    """One mosaic telling the reinstatement story (pre-plan section 18).

    Four panels of a :class:`aggregate.reinstatement.ReinstatementAnalysis`:

    - **A** the joint ``(L, R)`` log density with the reinstatement breakpoints,
      the reinstated-capacity line ``m*y`` and the recovery cap ``(m+1)*y``;
    - **B** the deterministic maps ``A(R)`` (recovery), ``h(R)`` (reinstatement
      premium), ``D + h(R)`` (ceded premium) and ``D + h - A`` (the ceded
      underwriting drag), all as functions of the unlimited recovery ``R``;
    - **C** gross vs net underwriting-result return-period (Lee) curves; and
    - **D** the cession impact ``q_p(net) - q_p(gross)`` across ``p``.

    Parameters
    ----------
    an : ReinstatementAnalysis
        The (built) analysis to plot.
    axd : dict of str to Axes, optional
        Mosaic keys ``'A'``/``'B'``/``'C'``/``'D'``; a new figure is created if
        omitted and stored on ``an.figure``.
    **kwargs
        Passed to the canvas creator (e.g. ``figsize``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if axd is None:
        kwargs.setdefault('figsize', (2 * FIG_W, 2 * FIG_H))
        an.figure, axd = make_mosaic('AB\nCD', **kwargs)
    else:
        an.figure = axd['A'].figure

    t = an.terms
    src = an.source
    L = np.asarray(src.axis0, dtype=float)        # gross loss
    R = np.asarray(src.axis1, dtype=float)        # unlimited ceded recovery
    dens = src.density
    xname, yname = src.axis_names
    Y = t.total_recovery_capacity                 # (m+1) y
    m_y = t.reinstatement_capacity                # m y

    # --- Panel A: joint (L, R) log density, L clipped to the occupied range ---
    axA = axd['A']
    m_L = dens.sum(axis=1)
    cum = np.cumsum(m_L) / max(m_L.sum(), 1e-300)
    hi = int(np.searchsorted(cum, 0.995)) + 1
    hi = min(max(hi, 2), len(L))
    stride = max(1, hi // 300)                     # keep the contour grid light
    Li = L[:hi:stride]
    Zi = dens[:hi:stride, :]
    with np.errstate(divide='ignore'):
        Z = np.log10(np.where(Zi.T > 0, Zi.T, np.nan))
    if np.isfinite(Z).any():
        axA.contourf(Li, R, Z, levels=14)
    for b in t._breakpoints[1:]:                   # premium tranche breakpoints
        axA.axhline(b, lw=0.5, color='C7', ls=':')
    axA.axhline(m_y, lw=0.9, color='C3', ls='--', label=f'm·y={m_y:g}')
    axA.axhline(Y, lw=0.9, color='C1', ls='--', label=f'(m+1)·y={Y:g}')
    axA.set(xlabel=f'{xname} loss L', ylabel=f'{yname} recovery R',
            title='Joint (L, R) log density')
    axA.legend(loc='upper right', fontsize='small')

    # --- Panel B: the deterministic recovery / premium maps vs R ---
    axB = axd['B']
    rr = np.linspace(0.0, Y * 1.15, 400)
    A = t.recovery(rr)
    h = t.reinstatement_premium(rr)
    D = t.deposit
    axB.plot(rr, A, label='A(R) recovery')
    axB.plot(rr, h, label='h(R) reinst. premium')
    axB.plot(rr, D + h, label='D + h(R) ceded premium')
    axB.plot(rr, D + h - A, label='D + h − A (ceded UW −)')
    axB.axvline(m_y, lw=0.5, color='C3', ls='--')
    axB.axvline(Y, lw=0.5, color='C1', ls='--')
    axB.set(xlabel='Unlimited recovery R', ylabel='Currency',
            title='Reinstatement economics')
    axB.legend(fontsize='small')

    # --- Panel C: gross vs net underwriting result, return-period (Lee) ---
    # Net of the deterministic expense / commission (a constant shift), so the
    # curves match the gcn_df / summary UW.
    axC = axd['C']
    net_leg = an._final_net_uw            # net of everything (agg cover aware)
    net_persp = 'net_agg' if an.agg_recovery is not None else 'net_occ'
    gross_uw = an._uw_with_expense('gross_uw', 'gross', False)
    net_uw = an._uw_with_expense(net_leg, net_persp, False)
    for gd, lbl in [(gross_uw, 'gross'), (net_uw, 'net')]:
        plot_quantile(axC, gd, label=lbl)
    axC.set(title='Underwriting result: gross vs net')
    axC.legend(fontsize='small')

    # --- Panel D: cession impact q_p(net) - q_p(gross) ---
    axD = axd['D']
    p = np.linspace(0.001, 0.999, 200)
    gw = np.asarray(gross_uw.q(p), dtype=float)
    nw = np.asarray(net_uw.q(p), dtype=float)
    axD.plot(p, nw - gw)
    axD.axhline(0.0, lw=0.6, color='C7', ls='--')
    axD.set(xlabel='Non-exceedance probability p',
            ylabel='net − gross UW result', title='Cession impact')
    return an.figure
