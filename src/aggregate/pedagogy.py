"""
Pedagogical helpers — figures and worked examples for the docs, blog, and papers.

Nothing in this module is part of the core API; nothing is re-exported from the
top-level :mod:`aggregate` namespace. Items here are imported by hand from
``aggregate.pedagogy`` when reproducing a specific figure or example.

The intent is to give paper/blog figure code a single home rather than scatter
it across modules. Future additions: figure-generation helpers currently in
``aggregate.ft`` (``poisson_example``, ``fft_wrapping_illustration``,
``recentering_convolution[_example]``) and ``aggregate.tweedie``
(``tweedie_illustration``).
"""
from itertools import count
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from pandas.io.formats.format import EngFormatter
import scipy.fft as sfft
import scipy.stats as ss

from .bounds import Bounds
from .constants import FIG_W, FIG_H
from .spectral import Distortion
from .underwriter import Underwriter, build

logger = logging.getLogger(__name__)


def bodoff_exhibit(port, reg_p):
    """Bodoff capital-allocation exhibit at the ``reg_p``-VaR asset level.

    Parameters
    ----------
    port : Portfolio
        The portfolio whose units we're allocating across.
    reg_p : float
        Regulatory probability — the asset level is ``port.q(reg_p, 'lower')``.

    Returns
    -------
    pandas.DataFrame
        Rows: ``EX``, ``sa VaR``, ``sa TVaR``, ``pct EX``, ``coVaR``, ``alt
        coVaR``, ``naive coTVaR``, ``coTVaR``, ``plc``. Columns: per-unit plus
        ``total``.

    Notes
    -----
    Implements the four-step Bodoff allocation comparison (stand-alone VaR/TVaR,
    pct-of-expected, conditional-VaR, conditional-TVaR with calibrated threshold)
    plus the policyholder layer-charge (``plc``) row.
    """
    basic = pd.DataFrame(
        index=pd.Index(['EX', 'sa VaR', 'sa TVaR', 'pct EX',
                        'coVaR', 'alt coVaR', 'naive coTVaR', 'coTVaR'],
                       name='method'),
        columns=port.unit_names_ex, dtype=float)

    basic.loc['EX'] = [
        float(port.stats_df.loc[('agg', 'mean'), name])
        for name in port.unit_names
    ] + [float(port.stats_df.loc[('agg', 'mean'), 'total'])]
    basic.loc['sa VaR'] = port.var_dict(reg_p, 'lower').values()
    basic.loc['sa TVaR'] = port.var_dict(reg_p, 'tvar').values()
    a = port.q(reg_p, 'lower')

    basic.loc['pct EX'] = basic.loc['EX'] / basic.loc['EX', 'total'] * a
    basic.loc['coVaR'] = (port.density_df.loc[a, [f'exeqa_{i}' for i in port.unit_names_ex]]
                         / port.density_df.at[a, 'exeqa_total']).values * a
    basic.loc['alt coVaR'] = (port.density_df.loc[
        a - port.bs,
        [f'exi_xgta_{i}' for i in port.unit_names] + ['exi_xgta_sum']] * a).values
    basic.loc['naive coTVaR'] = (port.density_df.loc[
        a - port.bs, [f'exgta_{i}' for i in port.unit_names_ex]]
        / port.density_df.at[a - port.bs, 'exgta_total']).values * a

    pt = port.tvar_threshold(reg_p, 'lower')
    av = port.q(pt)
    basic.loc['coTVaR'] = port.density_df.loc[
        av, [f'exgta_{l}' for l in port.unit_names_ex]].values

    bit = port.density_df[[f'exi_xgta_{i}' for i in port.unit_names]].shift(1).cumsum() * port.bs
    bit['total'] = bit.sum(1)
    basic.loc['plc'] = bit.loc[a].values

    return basic


def plot_max_min(bounds, ax):
    """
    Plot the min/max envelope of a ``Bounds`` cloud as a shaded band.

    Pulled out of ``Bounds.cloud_view``/``plot_envelope`` so that case-study
    code can compose a custom figure layout from the cloud.
    """
    ax.fill_between(bounds.cloud_df.index, bounds.cloud_df.min(1),
                    bounds.cloud_df.max(1), facecolor='C7', alpha=.15)
    bounds.cloud_df.min(1).plot(ax=ax, label='_nolegend_', lw=0.5, ls='-', c='k')
    bounds.cloud_df.max(1).plot(ax=ax, label='_nolegend_', lw=0.5, ls='-', c='k')


def plot_lee(port, ax, c, lw=1):
    """
    Lee diagram (quantile function) for ``port``, drawn step-style.

    Plots ``port.q(p)`` against ``p`` on ``ax``.
    """
    p_ = np.linspace(0, 1, 1001)
    qs = [port.q(p) for p in p_]
    ax.step(p_, qs, lw=lw, c=c, label=port.name)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, max(qs) + .05],
           title=f'{port.name} Lee diagram')


def plot_similar_risks_graphs(axd, bounds, port, pnew, roe, prem, p_reg=1):
    """
    Stand-alone version of the similar-risks figure used in the bounded-beta
    case studies. ONLY WORKS FOR BOUNDED PORTFOLIOS.

    Inputs:

    - ``axd``        mosaic dict from ``fig.subplot_mosaic``
    - ``bounds``     a calibrated ``Bounds`` object on the base portfolio
    - ``port``       the base portfolio
    - ``pnew``       the new portfolio whose pricing range is to be visualised
    - ``roe``, ``prem``   parameters used to construct the ccoc/TVaR distortions
    - ``p_reg``      regulatory probability (1 == unbounded)

    Provenance: ``make_port`` in ``Examples_2022_post_publish``.
    fka similar_risks_graphs_sa
    """
    if axd is None:
        fig = plt.figure(constrained_layout=True, figsize=(12, 6))
        axd = fig.subplot_mosaic(
            """
            AAAABBFF
            AAAACCFF
            AAAADDEE
            AAAADDEE
        """
        )

    df = bounds.weight_df.copy()
    df['test'] = df['t_upper'] * df.weight + df.t_lower * (1 - df.weight)

    # HERE IS ISSUE - should really use tvar with bounds and incorporate the bound
    if p_reg < 1:
        logger.warning('figuring tvars with bounds')
        btemp = Bounds(pnew, premium=prem)
        b = pnew.q(p_reg)
        tvar1 = {p: btemp._tvar_x_func(p) for p in bounds.p_knots}
    else:
        tvar1 = {p: float(pnew.tvar(p)) for p in bounds.p_knots}
    df['t1_lower'] = [tvar1[p] for p in df.index.get_level_values(0)]
    df['t1_upper'] = [tvar1[p] for p in df.index.get_level_values(1)]
    df['t1'] = df.t1_upper * df.weight + df.t1_lower * (1 - df.weight)

    roe_d = Distortion('ccoc', r=roe)
    tvar_d = Distortion('tvar', p=bounds.p_star)
    idx = df.index.get_locs(df.idxmax()['t1'])[0]
    pl, pu, tl, tu, w = df.reset_index().iloc[idx, :-4]
    max_d = Distortion('wtdtvar', ps=[pl, pu], wts=[1 - w, w])

    tmax = float(df.iloc[idx]['t1'])
    n_ = len(df.query('t1 == @tmax'))
    logger.warning(f'Ties for max: {n_}')
    n_ = len(df.query(f't1 >= {tmax} - 1e-4'))
    logger.warning(f'Near ties for max: {n_}')

    idn = df.index.get_locs(df.idxmin()['t1'])[0]
    pln, pun, tl, tu, wn = df.reset_index().iloc[idn, :-4]
    min_d = Distortion('wtdtvar', ps=[pln, pun], wts=[1 - wn, wn])

    ax = axd['A']
    plot_max_min(bounds, ax)
    n = len(ax.lines)
    roe_d.plot(ax=ax, both=False)
    tvar_d.plot(ax=ax, both=False)
    max_d.plot(ax=ax, both=False)
    min_d.plot(ax=ax, both=False)

    ax.lines[n + 0].set(label='roe', color='C0', ls='--')
    ax.lines[n + 2].set(color='C1', label='tvar', ls='-.')
    ax.lines[n + 4].set(color='C4', label='max', lw=1)
    ax.lines[n + 6].set(color='C5', label='min', lw=1)
    bounds.cloud_df.mean(1).plot(ax=ax, c='C3', ls='-.', lw=1.5, label='Avg extreme')
    ax.legend(loc='upper left')

    ax.set(title=f'Max ({pl}, {pu}), min ({pln}, {pun})')

    ax = axd['B']
    bounds.plot_weights(ax)

    bit = df['t1'].unstack(1)
    ax = axd['C']
    img = ax.contourf(bit.columns, bit.index, bit, cmap='viridis_r', levels=20)
    ax.set(xlabel='p1', ylabel='p0', title='Pricing on New Risk', aspect='equal')
    ax.get_figure().colorbar(img, ax=ax, shrink=.5, aspect=16, label='rho(X_new)')
    ax.plot(pu, pl, '.', c='w')
    ax.plot(pun, pln, 's', ms=3, c='white')

    ax = axd['D']
    plot_lee(port, ax, 'C0', lw=1)
    plot_lee(pnew, ax, 'C1')
    ax.set(ylim=[0, port.q(0.999)])
    ax.legend()

    ax = axd['E']
    try:
        port.density_df.p_total.plot(ax=ax, logy=True, lw=1, label=port.name)
    except AttributeError:
        logger.error('Attribute error...continuing')
    try:
        pnew.density_df.p_total.plot(ax=ax, logy=True, lw=1, label=pnew.name)
    except AttributeError:
        logger.error('Attribute error...continuing')
    ax.legend()
    ax.set(title='Total, log densities')

    ax = axd['F']
    plot_max_min(bounds, ax)
    for c, dd in zip(['C0', 'C1', 'C2'], ['ph', 'wang', 'dual']):
        port.distortions[dd].plot(ax=ax, both=False, lw=1)
        ax.lines[n].set(c=c, label=dd)
        n += 2
    ax.legend(loc='lower right')

    return df


def similar_risks_example():
    """
    Bounded-beta example showing how to use ``similar_risks_graphs_sa``.

    Builds a base UNIF portfolio, calibrates distortions, then a candidate BETA
    portfolio whose pricing range we visualise against the base.
    """
    uw = Underwriter()
    p_base = uw.build('''
    port UNIF
        agg ONE 1 claim sev 1 * beta 1 1 fixed
    ''')
    p_base.update(11, 1 / 1024, remove_fuzz=True)
    prem = p_base.tvar(0.2, 'interp')
    a = 1
    d = (prem - p_base.ex) / (a - p_base.ex)
    v = 1 - d
    roe = d / v
    p_base.calibrate_distortions(roe, a=a)
    bounds = Bounds(p_base, premium=prem, a=a)

    f, axs = plt.subplots(1, 3, figsize=(18.0, 6.0), layout='constrained')
    bounds.plot_envelope(axs=axs.flatten(), n_resamples=0, alpha=1,
                         title=f'Premium={prem:,.1f}, a={a:,.0f}, p*={bounds.p_star:.3f}',
                         distortions=[{k: p_base.distortions[k] for k in ['ccoc', 'tvar']},
                                      {k: p_base.distortions[k] for k in ['ph', 'wang', 'dual']}])
    for ax in axs.flatten()[1:]:
        ax.legend(ncol=1, loc='lower right')
    for ax in axs.flatten():
        ax.set(title=None)

    program = '''
    port BETA
        agg TWO 1 claim sev 1 * beta [200 300 400 500 600 7] [600 500 400 300 200 1] wts=6 fixed
    '''
    p_new = uw.build(program)
    p_new.update(11, 1 / 1024, remove_fuzz=True)
    p_new.plot(figsize=(6, 4))

    axd = plt.figure(constrained_layout=True, figsize=(16, 8)).subplot_mosaic(
        """
        AAAABBFF
        AAAACCFF
        AAAADDEE
        AAAADDEE
    """
    )
    df = plot_similar_risks_graphs(axd, bounds, p_base, p_new, roe, prem)
    return df


# ---------------------------------------------------------------------------
# Documentation figures
# ---------------------------------------------------------------------------
# The functions in this section are referenced from the technical-guide RSTs
# in ``docs/5_technical_guides``. Original names retained so existing doc
# imports keep working. They were consolidated here from
# ``extensions/figures.py`` and ``extensions/pir_figures.py`` at 1.0.0a12.

def prob_format(axis):
    """Axis formatter that prints probabilities tidily (0, 0.25, 0.5, ..., 1)."""
    axis.set_major_formatter(ticker.FuncFormatter(
        lambda x, y: '0' if x == 0
        else ('1' if x >= 0.999
        else (f'{x:.2f}' if np.allclose(x, 0.25) or np.allclose(x, 0.75)
        else f'{x:.1f}'))))


def adjusting_layer_losses():
    """
    Figure to illustrate the process of adjusting layer losses.

    Used by ``docs/5_technical_guides/5_x_adjusting_losses.rst``.
    """
    f, ax = plt.subplots(1, 1, figsize=(FIG_H, FIG_W), constrained_layout=True)
    fz = ss.lognorm(.4)
    xs = np.linspace(0, fz.isf(1e-3), 1001, endpoint=False)
    F = fz.cdf(xs)
    ax.plot(F, xs)
    a1 = 1.5
    a0 = 1
    ax.axhline(a1, c='k', lw=.25)
    ax.axhline(a0, c='k', lw=.25)
    p0 = fz.cdf(a0)
    p1 = fz.cdf(a1)
    ax.plot([p0, p0], [0, a0], c='k', lw=.25)
    ax.plot([p1, p1], [0, a1], c='k', lw=.25)
    ax.set_xticks([0, p0, p1, 1])
    ax.set_xticklabels(['0', '$F(a_{i-1})$', '$F(a_{i})$', ''])
    ax.set_yticks([0, a0, a1, 3.5])
    ax.set_yticklabels(['0', '$a_{i-1}$', '$a_{i}$', ''])
    ax.text((p0 + p1) / 2, a0 / 2, '$f_i$', ha='center', va='center')
    ax.text((1 + p1) / 2, 1.25, '$e_i$', ha='center', va='center')
    xx = [.8]
    yy = [1.15]
    ll = ['$m_i$']
    ho = -0.2
    vo = 0.55
    ax.set(xlabel='Nonexceedance probability\n$F(x)$', ylabel='Outcome $x$',
           ylim=[0, 3.5], xlim=[-0.05, 1.05],
           title='Lee diagram\n$x$ vs $F(x)$')
    for x, y, l in zip(xx, yy, ll):
        ax.annotate(text=l, xy=(x, y), xytext=(x + ho, y + vo),
                    arrowprops={'arrowstyle': '->', 'linewidth': .5})
    return f


def savings_charge():
    """
    Figure to illustrate the insurance savings and charge.

    Used by ``docs/5_technical_guides/5_x_ir_pricing.rst``.
    """
    f, ax = plt.subplots(1, 1, figsize=(FIG_H, FIG_W), constrained_layout=True)
    fz = ss.lognorm(.4)
    xs = np.linspace(0, fz.isf(1e-3), 1001, endpoint=False)
    F = fz.cdf(xs)
    ax.plot(F, xs, lw=3)
    a1 = 1.25
    ax.axhline(a1, c='C7', lw=1.5)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_yticks([0, a1, 3.5])
    ax.set_yticklabels(['0', '$r$', ''])
    ax.set(xlabel='Nonexceedance probability', ylabel='Scaled outcome',
           ylim=[0, 3.5], xlim=[0, 1], title='Insurance savings and expense')
    xx = [.9, .25]
    yy = [1.05 * a1, .8 * a1]
    ll = ['Insurance\ncharge, $\\phi(r)$',
          'Insurance\nsavings, $\\psi(r)$']
    hos = [-0.1, 0.25]
    vos = [1.2, .65]
    ax.text(.5, a1 * .4, '$E[X\\wedge l]$', ha='center', va='center')
    for x, y, l, ho, vo in zip(xx, yy, ll, hos, vos):
        ax.annotate(text=l, xy=(x, y), xytext=(x + ho, y + vo),
                    ha='right', va='bottom',
                    arrowprops={'arrowstyle': '->', 'linewidth': .5})
    return f


def mixing_convergence(freq_cv, sev_cv, bs=1 / 64):
    """
    Illustrate convergence of mixed distributions to the mixing distribution.

    Used by ``docs/5_technical_guides/5_x_severity_irrelevant.rst``.
    """
    a = build('agg M 1 claims sev gamma 1 cv ' + str(sev_cv) +
              ' mixed gamma ' + str(freq_cv), log2=16, bs=bs)
    dfnb = a.density_df[['p']].rename(columns={'p': 1})
    assert np.abs(a.est_m / a.actual_m - 1) < 1e-3
    for freq in [2, 5, 10, 20, 50, 100, 200]:
        a = build(f'agg M {freq} claims sev gamma 1 cv {sev_cv} '
                  f'mixed gamma {freq_cv}', log2=16, bs=bs)
        dfnb[freq] = a.density_df[['p']]
        assert np.abs(a.est_m / a.actual_m - 1) < 1e-3

    a = build(f'agg M 1 claims sev gamma 1 cv {sev_cv} poisson', log2=16, bs=bs)
    dfp = a.density_df[['p']].rename(columns={'p': 1})
    assert np.abs(a.est_m / a.actual_m - 1) < 1e-3
    for freq in [2, 5, 10, 20, 50, 100, 200]:
        a = build(f'agg M {freq} claims sev gamma 1 cv {sev_cv} poisson',
                  log2=16, bs=bs)
        dfp[freq] = a.density_df[['p']]
        assert np.abs(a.est_m / a.actual_m - 1) < 1e-3

    fig, axs = plt.subplots(2, 2, figsize=(2 * FIG_W, 2 * FIG_H),
                            constrained_layout=True)
    axi = iter(axs.flat)
    for lbl, df in zip(['Poisson distribution', 'Mixed frequency distribution'],
                       [dfp, dfnb]):
        ax0 = next(axi)
        ax1 = next(axi)
        for c in df:
            ax0.plot(df.index / c, c * df[c], lw=1, label=str(c))
            ax1.plot(df.index / c, df[c].cumsum(), lw=1, label=str(c))
        if ax0 is axs.flat[0]:
            ax0.legend()
        ax0.set(ylim=[0, 1e-1], xlim=[-0.25, 5])
        ax1.set(ylim=[-0.05, 1.05], xlim=[-0.25, 5])
        ax0.set(ylabel='Density', xlabel='Normalized loss', title=lbl)
        ax1.set(ylabel='Distribution', xlabel='Normalized loss', title=lbl)

    alpha = freq_cv ** -2
    fz = ss.gamma(alpha, scale=1 / alpha)
    ps = np.linspace(0, 5, 501)
    ax = axs.flat[-1]
    ax.plot(ps, fz.cdf(ps), lw=2, alpha=.5, c='k', label='Mixing')
    ax.legend(loc='lower right')


def power_variance_family():
    """
    Graph illustrating the power-variance exponential family.

    Used by ``docs/5_technical_guides/5_x_tweedie.rst``.

    Reference: Jørgensen, Bent (1997), *The Theory of Dispersion Models*, CRC.
    """
    alpha = np.linspace(-2, 2, 101)
    p = (alpha - 2) / (alpha - 1)
    alphabar = -(alpha + 1)
    f, ax = plt.subplots(figsize=(FIG_W * 2, FIG_W * 2))
    ax.plot(alphabar, p, lw=3)
    ax.set(ylim=[-5, 10])
    ax.axhline(1, c='k', lw=1)
    ax.axhline(0, c='k', lw=1)
    ax.axvline(-2, c='k', lw=.5)
    ax.axvline(-3 / 2, c='k', lw=0.5, ls='--')
    ax.axhline(3, c='k', lw=0.5, ls='--')
    ax.axhline(2, c='r', lw=1)
    ax.axvline(-1, c='r', lw=1)
    ax.set(xlabel='$\\bar\\alpha=-(\\alpha+1)$, base jump density is $x^{\\bar\\alpha}$',
           ylabel='Variance power function $p$, $V(\\mu)=\\phi\\mu^p$')
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(-4, 10)))

    def ql(x, y, t, dot=True, rhs=None):
        ax.text(x + .05, y + 0.2, t)
        if dot:
            ax.plot(x, y, 'rd', ms=5)
        else:
            if rhs is None:
                rhs = x + 1
            ax.plot([x, rhs], [y, y], lw=3)

    ql(-3, 0, 'Normal')
    ql(-3, 6, 'Stable', False)
    ql(-2, 9, 'Cauchy')
    ql(-2, -2, 'Pos Stable', False)
    ql(-1.5, 3, 'Stab 3/2\nIG')
    ql(-1, 2, 'Gamma')
    ql(1, 1, 'Poisson')
    ql(-1, -2, 'Tweedie', False, 1)
    ax.set(title='Power Variance Exponential Family Distributions')


def plot_quantile_illustration():
    """Illustrate the definition of quantiles when ``F`` has a flat spot and a jump.

    Originally ``fig_4_1`` (PIR Figure 4.1). Used by
    ``docs/5_technical_guides/5_x_quantiles.rst``.
    """
    fz = ss.lognorm(.5)
    xs = np.linspace(0, 5, 501)[1:]
    xsx = np.linspace(0, 5, 501)[1:]
    xsx[89:149] = xsx[89]
    F = fz.cdf(xsx)

    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H),
                           constrained_layout=True, squeeze=True)
    lt = F < .6
    gt = F > .6
    for f in [lt, gt]:
        if f is lt:
            ax.plot(xs[f], F[f], lw=2, label='Distribution, $F$')
        else:
            ax.plot(xs[f], F[f], lw=2, label=None)

    ax.plot([0, 5], [0.6, 0.6], ls='--', c='k', lw=1, label='$p=0.6$')
    p = fz.cdf(xs[89])
    ax.plot([0, 5], [p, p], ls='--', lw=1, c='C2', label=f'$p={p:.3f}$')
    ax.set(xlabel='$x$', ylabel='$F(x)$')
    ax.axvline(1.50, lw=0.5)
    xx = 0.75
    pp = fz.cdf(xx)
    ax.plot([0, xx], [pp, pp], ls='-', lw=.5, c='k', label=f'$p={pp:.3f}$')
    ax.plot([xx, xx], [0, pp], ls='-', lw=.5, c='k', label=None)
    ax.legend(loc='lower right')

    p1 = fz.cdf(xs[149])
    x = 1.5
    s = .1
    ax.plot(x, p, 'ok', ms=5, fillstyle='none')
    ax.plot(x, p1, 'ok', ms=5)
    ax.text(x + s, p + s / 4, f'$Pr(X<1.5)={p:.3f}$')
    ax.text(x + s, p1 - s / 4, f'$Pr(X ≤ 1.5)={fz.cdf(1.5):.3f}$')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    return fig


def _discrete_example():
    """Shared discrete sample (PIR Example 4.9): ten equiprobable atoms with ties.

    Returns ``(ps, cps, xs, df)`` used by
    :func:`plot_discrete_distribution_quantile`,
    :func:`plot_continuous_distribution_quantile`, :func:`plot_tvar_quantile`.
    """
    ps = np.ones(10) / 10
    cps = np.hstack((0, np.cumsum(ps)))
    xs = np.array([0, 0, 1, 1, 1, 2, 3, 4, 8, 12, 25])
    df = pd.DataFrame({'x': xs[1:], 'p': ps})
    df = pd.DataFrame(df.groupby('x').p.sum())
    df['F'] = df.p.cumsum()
    df = df.reset_index(drop=False)
    return ps, cps, xs, df


def plot_discrete_distribution_quantile():
    """Discrete (step) distribution function and lower-quantile VaR.

    Originally ``fig_4_5`` (PIR Figure 4.5). Used by
    ``docs/5_technical_guides/5_x_nm_discrete_rep.rst``.
    """
    ps, cps, xs, df = _discrete_example()
    fig, axs = plt.subplots(1, 2, figsize=(2 * FIG_W, FIG_W + .2))
    ax0, ax1 = axs.flat
    ax = ax0
    ax.plot(xs, cps, drawstyle='steps-post')
    ax.plot(xs[1:], cps[1:], 'o')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(.1))
    prob_format(ax.yaxis)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set(xlim=[-.5, 25.5], ylim=[-.025, 1.025],
           title='Distribution function\nright continuous',
           aspect=(26 / 1.05) / (4.5 / 3.25) / 1.15,
           ylabel='$F(x)$', xlabel='Outcome, $x$')

    ax = ax1
    ax.plot(cps, xs, drawstyle='steps-pre')
    ax.plot(cps[1:], xs[1:], 'o')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(.1))
    prob_format(ax.xaxis)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set(ylim=[-.5, 25.5], xlim=[-.025, 1.025],
           title='Lower quantile VaR function\nleft continuous',
           aspect=(4.5 / 3.25) / (26 / 1.05),
           xlabel='$F(x)$', ylabel='Outcome, $x$')


def plot_continuous_distribution_quantile():
    """Continuous (piecewise-linear, filled-in) distribution function and lower-quantile VaR.

    Originally ``fig_4_6`` (PIR Figure 4.6). Used by
    ``docs/5_technical_guides/5_x_nm_discrete_rep.rst``.
    """
    ps, cps, xs, df = _discrete_example()
    fig, axs = plt.subplots(1, 2, figsize=(2 * FIG_W, FIG_W + .2))
    ax0, ax1 = axs.flat
    ax = ax0
    ax.plot(df.x, df.F, c='C0')
    ax.plot([0, 0], [0, df.F.iloc[0]], c='C0')
    ax.plot(df.x, df.F, 'o', c='C0')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(.1))
    prob_format(ax.yaxis)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set(xlim=[-.5, 25.5], ylim=[-.025, 1.025],
           title='Distribution function\n',
           aspect=(26 / 1.05) / (3.5 / 2.45),
           ylabel='$F(x)$', xlabel='Outcome, $x$')

    ax = ax1
    ax.plot(df.F, df.x, c='C0')
    ax.plot([0, df.F.iloc[0]], [0, 0], c='C0')
    ax.plot(df.F, df.x, 'o', c='C0')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(.1))
    prob_format(ax.xaxis)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set(ylim=[-.5, 25.5], xlim=[-.025, 1.025],
           title='Lower quantile VaR function\n',
           aspect=(3.5 / 2.45) / (26 / 1.05),
           xlabel='$F(x)$', ylabel='Outcome, $x$')


def plot_tvar_quantile():
    """TVaR overlaid on quantile VaR, discrete and continuous samples.

    Originally ``fig_4_8`` (PIR Figure 4.8). Used by
    ``docs/5_technical_guides/5_x_quantiles.rst``.
    """
    ps, cps, xs, df = _discrete_example()
    ad = build(f'agg Empirical 1 claim sev dhistogram xps {df.x.values} {df.p.values} fixed', bs=1)
    xv = np.hstack((1e-10, df.x.values))
    adc = build(f'agg Empirical 1 claim sev chistogram xps {xv} {df.p.values} fixed', bs=1 / 128)
    qps = np.linspace(0, 1, 1000, endpoint=True)
    tvar = ad.tvar(qps)
    ctvar = adc.tvar(qps)

    fig, axs = plt.subplots(1, 2, figsize=(2 * FIG_H, FIG_W + .3), sharey=True)
    ax0, ax1 = axs.flat

    ax = ax0
    ad.density_df.loss = np.minimum(ad.density_df.loss, 25)
    ad.density_df.plot(y='loss', x='F', drawstyle='steps-pre',
                       ylim=[-1, 25.2], xlim=[-0.02, 1.02], ax=ax, ls='--',
                       label='Quantile')
    ax.plot(cps[:2], [0, 0], ls='--', label='_none_')
    ax.plot(cps[1:], xs[1:], 'o', ms=5, c='C0', label='_none_')
    ax.plot(qps, tvar, c='C0', lw=1, label='TVaR')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(.1))
    prob_format(ax.xaxis)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set(ylim=[-.5, 25.2], xlim=[-.025, 1.025],
           title='TVaR and lower quantile VaR,\ndiscrete sample',
           aspect=(4.5 / 3.25) / (26 / 1.05),
           xlabel='$F(x)$', ylabel='Outcome, $x$')
    ax.legend()

    ax = ax1
    adc.density_df.plot(y='loss', x='F', drawstyle='steps-pre',
                        ylim=[-1, 25.2], xlim=[-0.02, 1.02], ax=ax, ls='--')
    ax.plot(df.F, df.x, 'o', ms=5)
    ax.plot(qps, ctvar, c='C0')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(.1))
    prob_format(ax.xaxis)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set(ylim=[-.5, 25.2], xlim=[-.025, 1.025],
           title='TVaR and lower quantile VaR,\ncontinuous sample',
           aspect=(4.5 / 3.25) / (26 / 1.05),
           xlabel='$F(x)$', ylabel='Outcome, $x$')
    ax.legend().set(visible=False)


class ClassicalPremium:
    """Classical (pre-spectral) premium principles, calibrated to a target premium.

    Methods covered: expected value, VaR, variance, standard deviation,
    semi-variance (Artzner p. 210), exponential (zero-utility, convex),
    Esscher, Dutch, Fischer.

    Used by :func:`plot_ruin_surplus_paths` for the surplus-path illustration
    in ``docs/5_technical_guides/5_x_pk.rst``. Originally in ``hack.py`` then the
    PIR ``CaseStudy`` machinery.
    """

    __classical_methods__ = ['Expected Value', 'VaR', 'Variance', 'Standard Deviation',
                             'Semi-Variance', 'Exponential', 'Esscher', 'Dutch', 'Fischer']

    def __init__(self, ports, calibration_premium):
        self.ports = ports
        self.calibration_premium = calibration_premium

    def distribution(self, port_name, unit_name):
        """Pull the per-unit marginal and basic moment stats out of ``ports[port_name]``.

        The total reads ``density_df.p_total``; units read their native
        pmf via :meth:`~aggregate.portfolio.Portfolio.unit_density`.
        """
        port = self.ports[port_name]
        if unit_name == 'total':
            ser = port.density_df.p_total
            ob = port
        else:
            ser = port.unit_density(unit_name)
            ob = port[unit_name]
        df = pd.DataFrame({'loss': np.asarray(ser.index, dtype=float),
                           'p': ser.to_numpy()}, index=ser.index)
        df['F'] = df.p.cumsum()
        # empirical moments straight from the pmf (the removed
        # ``audit_df`` carried exactly these as EmpEX1/EmpEX2)
        mn = float((df.loss * df.p).sum())
        ex2 = float((df.loss ** 2 * df.p).sum())
        var = ex2 - mn ** 2
        sd = var ** 0.5
        stats = pd.Series({'EmpEX1': mn, 'EmpEX2': ex2}, name=unit_name)
        return df, ob, stats, mn, var, sd

    def calibrate(self, port_name, unit_name, calibration_premium,
                  df=None, ob=None, stats=None, mn=None, var=None, sd=None):
        """Calibrate each classical method to reproduce ``calibration_premium``."""
        from scipy.optimize import newton
        self.calibration_premium = calibration_premium
        if df is None:
            df, ob, stats, mn, var, sd = self.distribution(port_name, unit_name)

        ans = {}
        ans['Expected Value'] = calibration_premium / mn - 1
        ans['VaR'] = float(ob.cdf(calibration_premium))
        ans['Variance'] = (calibration_premium - mn) / var
        ans['Standard Deviation'] = (calibration_premium - mn) / sd

        for method in self.__classical_methods__:
            if method not in ans:
                if method in ['Exponential', 'Esscher']:
                    x0 = 1e-10
                elif method == 'TVaR':
                    x0 = ans['VaR'] - 0.025
                else:
                    x0 = 0.5
                try:
                    a = newton(lambda x: self.price(
                        x, port_name, unit_name, method, df, ob, stats, mn, var, sd
                    ) - calibration_premium, x0=x0)
                    ans[method] = float(a)
                except RuntimeError as e:
                    print(method, e)
        return ans

    def prices(self, port_name, unit_name, method_dict):
        """Apply each calibrated method to ``port_name``/``unit_name``."""
        df, ob, stats, mn, var, sd = self.distribution(port_name, unit_name)
        ans = {}
        for method, param in method_dict.items():
            ans[method] = self.price(param, port_name, unit_name, method,
                                     df, ob, stats, mn, var, sd)
        return ans

    def price(self, param, port_name, unit_name, method,
              df=None, ob=None, stats=None, mn=None, var=None, sd=None):
        """Price ``port_name``/``unit_name`` under one classical method."""
        if df is None:
            df, ob, stats, mn, var, sd = self.distribution(port_name, unit_name)
        if method == 'Expected Value':
            return mn * (1 + param)
        if method == 'VaR':
            return ob.q(param)
        if method == 'TVaR':
            return ob.tvar(param)
        if method == 'Variance':
            return mn + param * var
        if method == 'Standard Deviation':
            return mn + param * sd
        if method == 'Semi-Variance':
            sd_plus = np.sum(np.maximum(0, df.loss - mn) ** 2 * df.p)
            return mn + param * sd_plus
        if method == 'Exponential':
            eax = np.sum(np.exp(param * df.loss) * df.p)
            return (1 / param) * np.log(eax)
        if method == 'Esscher':
            eax = np.sum(np.exp(param * df.loss) * df.p)
            exax = np.sum(df.loss * np.exp(param * df.loss) * df.p)
            return exax / eax
        if method == 'Dutch':
            excess = np.sum(np.maximum(df.loss - mn, 0) * df.p)
            return mn + param * excess
        if method == 'Fischer':
            excess = np.sum(np.maximum(df.loss - mn, 0) ** self.p * df.p) ** (1 / self.p)
            return mn + param * excess

    def illustrate(self, port_name, unit_name, ax, margin,
                   *, p=0, K=0, n_big=10000, n_sample=25, show_bounds=True,
                   padding=2):
        """Simulate ``n_sample`` surplus paths over ``n_big`` policies and plot to ``ax``.

        Capital ``K`` is supplied directly, or implicitly via ``p`` (an eventual
        ruin probability) using the Cramér-Lundberg interpolation.
        """
        if unit_name == 'total':
            raise ValueError('Cannot use total in ClassicalPremium.illustrate.')
        ag = self.ports[port_name][unit_name]
        if p and K == 0:
            self.ruin, self.u, self.mean, self._dfi = ag.pollaczeck_khinchine(
                margin, kind='interpolate', padding=padding)
            K = self.u(p)
        elif K == 0:
            raise ValueError('Must input one of K and p')

        ea = np.sum(ag.density_df.loss * ag.density_df.p)
        ans = [(-1, ea)]
        ns = np.arange(n_big)
        prems = K + (1 + margin) * ea * ns
        means = K + margin * ea * ns
        alpha = min(1, 50 / n_sample)
        bombs = 0
        for i in range(n_sample):
            samp = ag.density_df.loss.sample(
                n=n_big, weights=ag.density_df.p, replace=True)
            ans.append((i, np.mean(samp), np.max(samp)))
            ser = prems - samp.cumsum()
            idzero = np.argmax(ser <= 0)
            if idzero > 0:
                ser.iloc[idzero:] = ser.iloc[idzero]
            rc = np.random.rand()
            ax.plot(ns, ser, ls='-', lw=0.5, color=(rc, rc, rc), alpha=alpha)
            if idzero:
                ax.plot(ns[idzero], ser.iloc[idzero], 'x', ms=5)
                bombs += 1
        ax.plot(ns, means, 'k-', linewidth=2)
        ef = EngFormatter(3, True)
        ax.set(xlabel='Number of risks',
               ylabel=f'Cumulative capital, $u_0={ef(K)}$')
        lln = np.nan_to_num(np.log(abs(np.log(ns))))
        lln[lln < 0] = 0
        if show_bounds:
            var_a = (ag.actual_m * ag.actual_cv) ** 2
            lb = means - np.sqrt(2.0 * var_a * ns * lln)
            ub = means + np.sqrt(2.0 * var_a * ns * lln)
            ax.plot(ns, lb, '--', lw=1)
            ax.plot(ns, ub, '--', lw=1)
        title = f'{bombs}/{n_sample} = {bombs / n_sample:.3f} ruins'
        ax.set(title=title)
        return pd.DataFrame(ans, columns=['n', 'ea', 'min'])


def plot_ruin_surplus_paths(port):
    """Cramér-Lundberg ruin probability and sampled surplus paths.

    Originally ``fig_9_1`` (PIR Figure 9.1). Used by
    ``docs/5_technical_guides/5_x_pk.rst``. Depends on
    :class:`ClassicalPremium` in this module.
    """
    port_name = 'gross'
    unit_names = ['Limit1', 'Limit10']
    margin = 0.1
    ruins = {}
    find_us = {}
    dfis = {}
    for unit_name in unit_names:
        ag = port[unit_name]
        ruins[unit_name], find_us[unit_name], mean, dfi = ag.pollaczeck_khinchine(
            margin, kind='interpolate')
        dfis[unit_name] = pd.Series(dfi, index=ruins[unit_name].index)
    xmaxs = {'Limit1': 10e6, 'Limit10': 50e6}
    limit_dict = {f'Limit{n}': n * 1e6 for n in [1, 10]}
    n_big_dict = {'Limit1': 10000, 'Limit10': 50000}
    cp = ClassicalPremium({'gross': port}, 110)
    fig, axs = plt.subplots(2, 2, figsize=(2 * 3.5, 2 * 2.45),
                            constrained_layout=True)
    axi = iter(axs.flat)
    for unit_name in unit_names:
        ax0 = next(axi)
        ax1 = next(axi)
        ax_ = ax0.twinx()
        xmax = xmaxs[unit_name]
        ruins[unit_name].index.name = 'Starting capital'
        ruins[unit_name].plot(ax=ax0)
        ax0.axhline(1 / (1 + margin), lw=1)
        ruins[unit_name].plot(ax=ax_, ls='--', lw=1)
        ax_.set(ylim=[0.5e-6, 2], ylabel='log probability', yscale='log')
        ax_.yaxis.set_minor_locator(ticker.LogLocator(subs='all', numticks=20))
        ax0.set(xlim=[-xmax / 50, xmax], ylim=[-0.05, 1.05],
                ylabel='Probability of eventual default',
                title=f'Limit {limit_dict[unit_name] / 1e6:.0f}M, margin {margin}')
        ax_.set(xlim=[-xmax / 50, xmax])
        p_default = 0.05
        cp.illustrate(port_name, unit_name, ax1, margin,
                      p=p_default, n_big=n_big_dict[unit_name], n_sample=100)
        ax1.set(xlabel='Volume or time')


def natural_scale(port):
    """PIR Table 9.15: minimum-market-size constraint by limit, margin, ruin prob.

    Used by ``docs/5_technical_guides/5_x_pk.rst``.
    """
    margins = np.hstack((np.linspace(.025, .1, 4), np.linspace(.15, .25, 3)))
    roe = .1
    p_defaults = [.01, .05, 0.1, .25]
    df = pd.DataFrame(columns=['limit', 'p_default', 'margin', 'roe', 'exi',
                                'cvxi', 'lambda', 'u', 'mean_g', 'max_index'],
                      dtype=float)
    limit_dict = {f'Limit{n}': n * 1e6 for n in [1, 5, 10]}
    counter = count(0, 1)
    for unit_name in port.unit_names[:3]:
        ag = port[unit_name]
        ag_ex = ag.actual_m
        for margin in margins:
            try:
                ruin, find_u, mean, dfi = ag.pollaczeck_khinchine(
                    margin, kind='index', padding=2)
                dfi = pd.Series(dfi, index=ruin.index)
                ex = np.sum(dfi * dfi.index)
                ex2 = np.sum(dfi * dfi.index ** 2)
                cv = np.sqrt(ex2 - ex * ex) / ex
                mean_g = ex / margin
                for p_default, i in zip(p_defaults, counter):
                    u = find_u(p_default)
                    n_lambda = roe * u / (margin * ag_ex)
                    df.loc[i] = [limit_dict[unit_name], p_default, margin, roe,
                                 ex, cv, n_lambda, u, mean_g, ruin.index[-1]]
            except IndexError as e:
                print(e)
    df['u/r'] = df.u / df.margin
    bit = df.set_index(['limit', 'margin', 'p_default'])['lambda'].unstack(1)
    bit.index.names = ['Limit', 'p']
    bit.columns.name = 'Margin'
    return bit


# ---------------------------------------------------------------------------
# Renamed figures
# ---------------------------------------------------------------------------
# Curated figures used across paper/blog work, renamed for clarity. Original
# names noted in the docstring where they came from PIR.


def _get_ax_size(fig, ax):
    """Axes size in pixels — helper for :func:`_curly_brace`.

    Reference:
    https://uk.mathworks.com/matlabcentral/fileexchange/38716-curly-brace-annotation
    """
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_width, ax_height = bbox.width, bbox.height
    ax_width *= fig.dpi
    ax_height *= fig.dpi
    return ax_width, ax_height


def _curly_brace(ax, p1, p2, k_r=0.1, bool_auto=True, str_text='',
                 int_line_num=2, fontdict=None, **kwargs):
    """Plot a curly bracket on ``ax``. Helper for :func:`plot_distortion_and_ins_stats`.

    ``k_r`` controls bracket curvature; ``bool_auto=True`` rescales coordinates
    for non-equal aspect axes. Original credit: Dr. GAO Siyu (MATLAB Central).
    """
    if fontdict is None:
        fontdict = {}
    fig = ax.get_figure()
    pt1 = [None, None]
    pt2 = [None, None]
    ax_width, ax_height = _get_ax_size(fig, ax)
    ax_xlim = list(ax.get_xlim())
    ax_ylim = list(ax.get_ylim())

    if 'log' in ax.get_xaxis().get_scale():
        for src, dst in ((p1[0], pt1), (p2[0], pt2)):
            if src > 0.0:
                dst[0] = np.log(src)
            elif src < 0.0:
                dst[0] = -np.log(abs(src))
            else:
                dst[0] = 0.0
        for i in range(0, len(ax_xlim)):
            if ax_xlim[i] > 0.0:
                ax_xlim[i] = np.log(ax_xlim[i])
            elif ax_xlim[i] < 0.0:
                ax_xlim[i] = -np.log(abs(ax_xlim[i]))
            else:
                ax_xlim[i] = 0.0
    else:
        pt1[0] = p1[0]
        pt2[0] = p2[0]
    if 'log' in ax.get_yaxis().get_scale():
        for src, dst in ((p1[1], pt1), (p2[1], pt2)):
            if src > 0.0:
                dst[1] = np.log(src)
            elif src < 0.0:
                dst[1] = -np.log(abs(src))
            else:
                dst[1] = 0.0
        for i in range(0, len(ax_ylim)):
            if ax_ylim[i] > 0.0:
                ax_ylim[i] = np.log(ax_ylim[i])
            elif ax_ylim[i] < 0.0:
                ax_ylim[i] = -np.log(abs(ax_ylim[i]))
            else:
                ax_ylim[i] = 0.0
    else:
        pt1[1] = p1[1]
        pt2[1] = p2[1]

    xscale = ax_width / abs(ax_xlim[1] - ax_xlim[0])
    yscale = ax_height / abs(ax_ylim[1] - ax_ylim[0])
    if not bool_auto:
        xscale = 1.0
        yscale = 1.0

    pt1[0] = (pt1[0] - ax_xlim[0]) * xscale
    pt1[1] = (pt1[1] - ax_ylim[0]) * yscale
    pt2[0] = (pt2[0] - ax_xlim[0]) * xscale
    pt2[1] = (pt2[1] - ax_ylim[0]) * yscale

    theta = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    r = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]) * k_r

    x11 = pt1[0] + r * np.cos(theta)
    y11 = pt1[1] + r * np.sin(theta)
    x22 = (pt2[0] + pt1[0]) / 2.0 - 2.0 * r * np.sin(theta) - r * np.cos(theta)
    y22 = (pt2[1] + pt1[1]) / 2.0 + 2.0 * r * np.cos(theta) - r * np.sin(theta)
    x33 = (pt2[0] + pt1[0]) / 2.0 - 2.0 * r * np.sin(theta) + r * np.cos(theta)
    y33 = (pt2[1] + pt1[1]) / 2.0 + 2.0 * r * np.cos(theta) + r * np.sin(theta)
    x44 = pt2[0] - r * np.cos(theta)
    y44 = pt2[1] - r * np.sin(theta)

    q = np.linspace(theta, theta + np.pi / 2.0, 50)
    t = q[::-1]
    arc1x = r * np.cos(t + np.pi / 2.0) + x11
    arc1y = r * np.sin(t + np.pi / 2.0) + y11
    arc2x = r * np.cos(q - np.pi / 2.0) + x22
    arc2y = r * np.sin(q - np.pi / 2.0) + y22
    arc3x = r * np.cos(q + np.pi) + x33
    arc3y = r * np.sin(q + np.pi) + y33
    arc4x = r * np.cos(t) + x44
    arc4y = r * np.sin(t) + y44

    arc1x = arc1x / xscale + ax_xlim[0]
    arc2x = arc2x / xscale + ax_xlim[0]
    arc3x = arc3x / xscale + ax_xlim[0]
    arc4x = arc4x / xscale + ax_xlim[0]
    arc1y = arc1y / yscale + ax_ylim[0]
    arc2y = arc2y / yscale + ax_ylim[0]
    arc3y = arc3y / yscale + ax_ylim[0]
    arc4y = arc4y / yscale + ax_ylim[0]

    ax.plot(arc1x, arc1y, **kwargs)
    ax.plot(arc2x, arc2y, **kwargs)
    ax.plot(arc3x, arc3y, **kwargs)
    ax.plot(arc4x, arc4y, **kwargs)
    ax.plot([arc1x[-1], arc2x[1]], [arc1y[-1], arc2y[1]], **kwargs)
    ax.plot([arc3x[-1], arc4x[1]], [arc3y[-1], arc4y[1]], **kwargs)

    if str_text:
        int_line_num = int(int_line_num)
        str_temp = '\n' * int_line_num
        ang = np.degrees(theta) % 360.0
        if (ang >= 0.0) and (ang <= 90.0):
            rotation = ang
            str_text = str_text + str_temp
        elif (ang > 90.0) and (ang < 270.0):
            rotation = ang + 180.0
            str_text = str_temp + str_text
        elif (ang >= 270.0) and (ang <= 360.0):
            rotation = ang
            str_text = str_text + str_temp
        else:
            rotation = ang
        ax.text(arc2x[-1], arc2y[-1], str_text, ha='center', va='center',
                rotation=rotation, fontdict=fontdict)


def plot_distortion_and_ins_stats(dist=None, s=0.3):
    """Two-panel distortion figure: ``(s, g(s))`` with split loss/premium/margin/capital.

    Provenance: PIR Figure 10.3. Originally ``fig_10_3``.
    """
    fig, axs = plt.subplots(1, 2, figsize=(2 * FIG_W, FIG_W), constrained_layout=True)
    if dist is None:
        dist = Distortion('ph', a=0.4)

    g = dist.g
    N = 1000
    ps = np.linspace(0, 1, N, endpoint=False)
    gs = g(ps)
    sm = 0.085
    g_s = g(s)
    lbl = str(dist).replace('\n', ' ')

    def setbg(t):
        t.set_bbox(dict(facecolor=plt.rcParams['axes.facecolor'], alpha=0.85,
                        edgecolor='none', boxstyle='square,pad=.1'))

    for a in axs.flat:
        a.plot(ps, gs, c='C1', lw=1.5, label='Premium, $g(s)$')
        a.plot(ps, ps, linewidth=1.5, c='C0', alpha=1, label='Loss cost, $s$')
        a.axis([-0.025, 1.025, -0.025, 1.025])
        a.set(aspect='equal', xlabel='$s$', ylabel='$g(s)$',
              title=f'Insurance Statistics\n{lbl}')
    axs[0].legend(loc='upper left')

    a = axs[1]
    a.plot([s, s], [0, s], c='C0', ls='--', alpha=1, linewidth=2.5)
    a.plot([s, s], [s, g_s], c='C1', ls='--', alpha=1, linewidth=2.5)
    a.plot([s, s], [g_s, 1], c='C2', ls='--', alpha=1, linewidth=2.5)
    a.text(s + sm, s / 2, 'Loss $=s$', va='center')
    t = a.text(s + sm, (g_s + s) / 2, 'Margin\n$=g(s)-s$', va='center')
    setbg(t)

    if s > 0.3:
        a.text(s - sm, (1 + g_s) / 2, 'Capital =\n$1-g(s)$', ha='right', va='center')
    else:
        t = a.text(s + sm, (1 + g_s) / 2, 'Capital\n$=1-g(s)$',
                   ha='left', va='center')
        setbg(t)

    delta = 0.02
    p3 = (s + delta, 0)
    p2 = (s + delta, s)
    p1 = (s + delta, dist.g(s))
    p0 = (s + delta, 1)
    p2m = (s + 1.5 * delta, s)
    p1m = (s + 1.5 * delta, dist.g(s))

    _curly_brace(a, p0, p1, str_text=None, int_line_num=2, k_r=0.055, c='k', lw=0.5)
    _curly_brace(a, p1m, p2m, str_text=None, int_line_num=2, k_r=0.075, c='k', lw=0.5)
    _curly_brace(a, p2, p3, str_text=None, int_line_num=2, k_r=0.075, c='k', lw=0.5)
    g_s = dist.g(s)
    _curly_brace(a, (.625, g_s), (.625, 0), str_text=None,
                 int_line_num=2, k_r=0.0375, c='k', lw=0.5)
    a.text(.625 + sm, g_s / 2, 'Premium\n$=g(s)$', va='center', ha='left')
    a.plot([0, .626], [g_s, g_s], lw=.5, c='k', ls='-')

    for ax in axs.flat:
        ax.set(title=None, xlabel='$s$, probability of loss to layer $1_{U<s}$',
               ylabel='Price of layer $1_{U<s}$', aspect='equal')


def plot_spectral_three_panel(port=None, dist=None, s=0.3, x=None,
                         return_period_max=100):
    """Three-panel distortion picture: layer view, filled view, traditional bar.

    Provenance: PIR Figure 10.5. Originally ``fig_10_5``.

    Maps from ``s`` space into loss space; the second panel adds the filled
    [loss][margin][capital] horizontal bar. ``return_period_max`` sets the
    y-axis extent (in return-period years).
    """
    fig, axs = plt.subplots(1, 3, figsize=(3 * FIG_H, FIG_W), constrained_layout=True)
    ax0, ax1, ax2 = axs.flat

    if port is None:
        port = build('port Test agg A 10 claims sev lognorm 50 cv 1 mixed gamma .5',
                     bs=1 / 16)
    if dist is None:
        dist = build('distortion myph ph 0.4')

    if x is None:
        x = port.q(1 - s)
    else:
        s = port.sf(x)

    g = dist.g
    K = port.q(1 - 1 / return_period_max)
    xs = port.density_df.loss
    S = port.density_df.S
    gS = g(S)
    gS[0] = 1.0

    Fx = port.cdf(x)
    gFx = 1 - g(1 - Fx)

    idx = int(port.cdf(K) * len(S))
    lev = np.trapz(S.iloc[:idx], x=xs.iloc[:idx]) + xs[0]
    levg = np.trapz(np.array(gS)[:idx], x=xs.iloc[:idx]) + xs[0]

    dist_name = str(dist).replace('\n', ' ')

    ax0.plot(1 - S, xs, lw=1.5, c='C0', label='Loss, $S(x)$', drawstyle='steps-pre')
    ax0.plot(1 - gS, xs, lw=1.5, c='C1', drawstyle='steps-pre',
             label=f'Premium $g(S(x))$\nDistortion {dist_name}')
    ax0.plot([Fx, Fx], [0, x], linewidth=0.25, c='C7')
    ax0.plot([gFx, gFx], [0, x], linewidth=0.25, c='C1')
    ax0.plot([0, gFx], [x, x], linewidth=2.5, ls='--', c='C2', alpha=1)
    ax0.plot([Fx, 1], [x, x], linewidth=2.5, ls='--', c='C0', alpha=1)
    ax0.plot([gFx, Fx], [x, x], linewidth=2.5, ls='--', c='C1', alpha=1)
    ax0.set(ylabel='Asset layer', xlabel='Probability of\nnon-exceedance',
            ylim=(0, K), xlim=(0, 1))
    ax0.xaxis.set_ticks([0, gFx, Fx, 1])
    ax0.xaxis.set_ticklabels(['0', '$\\tilde p$', '$p$', '1'])
    ax0.yaxis.set_ticks([0, x, K])
    ax0.yaxis.set_ticklabels(['', '$x$', '$a$'])
    ax0.annotate('Layer\ncapital', ((gFx) / 2 + 0.04, x),
                 ((gFx) / 2 - 0.1 + 0.04, x + 0.3 * lev),
                 va='baseline', ha='center', arrowprops={'arrowstyle': '->'})
    ax0.annotate('Layer\nmargin', ((Fx + gFx) / 2, x),
                 ((Fx + gFx) / 2 - 0.1, x + 0.3 * lev),
                 va='baseline', ha='center', arrowprops={'arrowstyle': '->'})
    ax0.annotate('Layer\nloss', ((Fx + 1) / 2, x), ((Fx + 1) / 2, x - 0.5 * lev),
                 va='baseline', ha='center', arrowprops={'arrowstyle': '->'})

    ax1.plot(1 - S, xs, lw=1.5, c='C0', label='Loss, $S(x)$', drawstyle='steps-pre')
    ax1.plot(1 - gS, xs, lw=1.5, c='C1', drawstyle='steps-pre',
             label=f'Premium $g(S(x))$\ndistortion {dist_name}')

    loss_line = [(port.cdf(i), i) for i in np.linspace(K, .01, 200)]
    prem_line = [(1 - g(1 - port.cdf(i)), i) for i in np.linspace(K, .01, 200)]
    patches = [Polygon([(0, 0), (0, K), (1 - g(1 - port.cdf(K)), K)] + prem_line, closed=True)]
    patches.append(Polygon([(1, 0), (1, K), (port.cdf(K), K)] + loss_line, closed=True))
    patches.append(Polygon([(1, 0)] + loss_line[::-1] + prem_line, closed=True))
    p = PatchCollection(patches, alpha=.25,
                        facecolors=['lightsteelblue', 'C0', 'C1'])
    ax1.add_collection(p)

    ax1.text(0.5, lev / 2, 'Loss', ha='center')
    ax1.text(0.5, (lev + levg) / 2, 'Margin', ha='center')
    ax1.text(0.5, (K + levg) / 2, 'Capital', ha='center')
    ax1.set(ylabel=None, xlabel='Probability of\nnon-exceedance',
            ylim=(0, K), xlim=(0, 1))
    ax1.xaxis.set_ticks([0, 1])
    ax1.xaxis.set_ticklabels(['0', '1'])
    ax1.yaxis.set_ticks(np.linspace(0, K, 1))
    ax1.yaxis.set_ticklabels('')
    ax1.legend().set_visible(False)

    ax2.bar(0, height=lev, width=1, align='edge', alpha=.25)
    ax2.bar(0, height=levg - lev, bottom=lev, width=1, align='edge', alpha=.25)
    ax2.axhline(lev, c='C0', lw=1.5)
    ax2.axhline(levg, c='C1', lw=1.5)
    ax2.text(0.5, lev / 2, 'Loss', ha='center', va='center')
    ax2.text(0.5, (lev + levg) / 2, 'Margin', ha='center', va='center')
    ax2.text(0.5, (K + levg) / 2, 'Capital', ha='center')
    ax2.set(xlabel=None)
    ax2.yaxis.set_visible(False)
    ax2.xaxis.set_ticks([0, 1])
    ax2.xaxis.set_ticklabels(['0', '1'])
    ax2.set(ylim=[0, K], xlim=[-.0, 1.], xlabel='Traditional\nlayer diagram')


def plot_bivariate(port, fig, ax, min_loss, max_loss, jump,
                   log=True, cmap='Greys', min_density=1e-15, levels=30,
                   lines=None, linecolor='w', colorbar=False, normalize=False,
                   **kwargs):
    """Contour plot of the bivariate density of two-unit ``port``.

    Provenance: originally ``biv_contour_plot`` in ``portfolio_pir``.

    Assumes ``port`` has exactly two units. Samples ``density_df`` at
    ``np.arange(min_loss, max_loss, jump)`` for the outer-product evaluation —
    pick ``jump`` carefully so the grid stays manageable (``100 * bs`` is a
    reasonable default).
    """
    npts = np.arange(min_loss, max_loss, jump)
    ps = [f'p_{i}' for i in port.unit_names]
    # native unit pmfs scattered onto the sample points (numerics-1);
    # zero where a unit has no bucket at the sampled loss.
    bit = pd.concat([port.unit_density(unit) for unit in port.unit_names],
                    axis=1).reindex(npts).fillna(0.0)
    n = len(bit)
    Z = bit[ps[1]].to_numpy().reshape(n, 1) @ bit[ps[0]].to_numpy().reshape(1, n)
    if normalize:
        Z = Z / np.sum(Z)
    X, Y = np.meshgrid(bit.index, bit.index)

    if log:
        z = np.log10(Z)
        mask = np.zeros_like(z)
        mask[z == -np.inf] = True
        mz = np.ma.array(z, mask=mask)
        cp = ax.contourf(X, Y, mz, levels=levels, cmap=cmap, **kwargs)
        if colorbar:
            cb = fig.colorbar(cp, fraction=.1, shrink=0.5, aspect=10)
            cb.set_label('Log10(Density)')
    else:
        mask = np.zeros_like(Z)
        mask[Z < min_density] = True
        mz = np.ma.array(Z, mask=mask)
        cp = ax.contourf(X, Y, mz, levels=levels, cmap=cmap, **kwargs)
        if colorbar:
            cb = fig.colorbar(cp)
            cb.set_label('Density')

    if lines is None:
        lines = np.arange(max_loss / 4, 2 * max_loss, max_loss / 4)
    try:
        for x in lines:
            ax.plot([0, x], [x, 0], lw=.75, c=linecolor, label=f'Sum = {x:,.0f}')
    except Exception:
        pass

    title = (
        f'Bivariate Log Density Contour Plot\n{port.name.replace("_", " ").title()}'
        if log
        else f'Bivariate Density Contour Plot\n{port.name.replace("_", " ")}'
    )
    ax.set(xlabel=f'Unit {port.unit_names[0]}',
           ylabel=f'Unit {port.unit_names[1]}',
           xlim=[-max_loss / 50, max_loss],
           ylim=[-max_loss / 50, max_loss],
           title=title, aspect=1)
    port.X = X
    port.Y = Y
    port.Z = Z


def plot_twelve(port, fig, axs, distortion_name, p=0.999, p2=0.9999,
                xmax=0, ymax2=0, biv_log=True, legend_font=0,
                contour_scale=10, sort_order=None, kind='two', cmap='viridis'):
    """Twelve-panel ASTIN/PIR diagnostic plot of a distorted portfolio.

    Provenance: originally ``twelve_plot`` in ``portfolio_pir``.

    Must run a distortion first, e.g.
    ``port.apply_distortion(port.distortions['ph'])``. Layer curves come
    from :meth:`Portfolio.allocation_diagnostics` (the explicit diagnostic
    frame); unit densities from the numerics-1 native accessors.

    Panels (by row × column index in ``axs``):

    - (1,1) density, (1,2) log density, (1,3) bivariate density
    - (2,1) κ, (2,2) α, (2,3) β
    - (3,1)/(4,1) per-unit S, gS, αS, βgS
    - (3,2) margin density M_i, (4,2) cumulative margin M̄_i
    - (3,3) stand-alone M, (4,3) natural M

    ``sort_order`` reorders the unit indices for plotting; defaults to ``[1, 2, 0]``.
    """
    # local renamer
    def _short_renamer(port, prefix='', postfix=''):
        """Map ``f'{prefix}_{unit}_{postfix}'`` columns to title-cased unit names.

        Private helper for :func:`plot_twelve` (and any other figure that
        pretty-prints per-unit columns out of ``density_df``).
        """
        if prefix:
            prefix = prefix + '_'
        if postfix:
            postfix = '_' + postfix
        knobble = lambda x: 'Total' if x == 'total' else x  # noqa: E731
        return {f'{prefix}{i}{postfix}': knobble(i).title() for i in port.unit_names_ex}

    a11, a12, a13, a21, a22, a23, a31, a32, a33, a41, a42, a43 = axs.flat
    col_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lss = ['solid', 'dashed', 'dotted', 'dashdot']

    if sort_order is None:
        sort_order = [1, 2, 0]
    if xmax == 0:
        xmax = port.q(p)
    ymax = xmax

    # total + native unit pmfs (numerics-1); on a legacy zero-origin book
    # the grids coincide and this matches the old p_{unit} columns.
    temp = (
        pd.concat([port.density_df.p_total] +
                  [port.unit_density(unit) for unit in port.unit_names],
                  axis=1)
        .rename(columns=_short_renamer(port, 'p'))
        .sort_index(axis=1).loc[:xmax]
    )
    temp = temp.iloc[:, sort_order]
    temp.index.name = 'Loss'
    kwargs = {'drawstyle': 'steps-mid'}
    l1 = temp.plot(ax=a11, lw=1, **kwargs)
    l2 = temp.plot(ax=a12, lw=1, logy=True, **kwargs)
    l1.lines[-1].set(linewidth=1.5)
    l2.lines[-1].set(linewidth=1.5)
    a11.set(title='Density')
    a12.set(title='Log density')
    a11.legend()
    a12.legend()

    if kind == 'two':
        xmax = port.snap(xmax)
        min_loss, max_loss, jump = 0, xmax, port.snap(xmax / 255)
        min_density = 1e-15
        levels = 30
        color_bar = False
        ps = [f'p_{i}' for i in port.unit_names]
        title = 'Bivariate density'
        # native unit pmfs on the union of their grids (numerics-1)
        biv = pd.concat([port.unit_density(unit)
                         for unit in port.unit_names], axis=1).fillna(0.0)
        nz = (biv[ps] > 0).any(axis=1)
        if int(nz.sum()) < 512:
            logger.info('Contour plot has few points...going discrete...')
            bit = biv.loc[nz]
            n = len(bit)
            Z = bit[ps[1]].to_numpy().reshape(n, 1) @ bit[ps[0]].to_numpy().reshape(1, n)
            X, Y = np.meshgrid(bit.index, bit.index)
            norm = mpl.colors.Normalize(vmin=-10, vmax=np.log10(np.max(Z.flat)))
            cm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            mapper = cm.to_rgba
            a13.scatter(x=X.flat, y=Y.flat, s=1000 * Z.flatten(),
                        c=mapper(np.log10(Z.flat)))
            a13.set(xlim=[min_loss - (max_loss - min_loss) / 10, max_loss],
                    ylim=[min_loss - (max_loss - min_loss) / 10, max_loss])
        else:
            npts = np.arange(min_loss, max_loss, jump)
            bit = biv.reindex(npts).fillna(0.0)
            n = len(bit)
            Z = bit[ps[1]].to_numpy().reshape(n, 1) @ bit[ps[0]].to_numpy().reshape(1, n)
            Z = Z / np.sum(Z)
            X, Y = np.meshgrid(bit.index, bit.index)
            if biv_log:
                z = np.log10(Z)
                mask = np.zeros_like(z)
                mask[z == -np.inf] = True
                mz = np.ma.array(z, mask=mask)
                cp = a13.contourf(X, Y, mz, levels=np.linspace(-17, 0, levels), cmap=cmap)
                if color_bar:
                    cb = fig.colorbar(cp)
                    cb.set_label('Log10(Density)')
            else:
                mask = np.zeros_like(Z)
                mask[Z < min_density] = True
                mz = np.ma.array(Z, mask=mask)
                cp = a13.contourf(X, Y, mz, levels=levels, cmap=cmap)
                if color_bar:
                    cb = fig.colorbar(cp)
                    cb.set_label('Density')
            a13.set(xlim=[min_loss, max_loss], ylim=[min_loss, max_loss])

        lines = np.arange(contour_scale / 4, 2 * contour_scale + 1, contour_scale / 4)
        logger.debug(f'Contour lines based on {contour_scale} gives {lines}')
        for x in lines:
            a13.plot([0, x], [x, 0], ls='solid', lw=.35, c='k',
                     alpha=0.5, label=f'Sum = {x:,.0f}')

        a13.set(xlabel=f'Unit {port.unit_names[0]}',
                ylabel=f'Unit {port.unit_names[1]}',
                title=title, aspect=1)
    else:
        l1 = (1 - temp.cumsum()).plot(ax=a13, lw=1)
        l1.lines[-1].set(linewidth=1.5)
        a13.set(title='Survival Function')
        a13.legend()

    bit = port.density_df.loc[:xmax].filter(regex=f'^exeqa_({port.unit_name_pipe})$')
    bit = bit.iloc[:, sort_order]
    bit.rename(columns=_short_renamer(port, 'exeqa')).replace(0, np.nan). \
        sort_index(axis=1).iloc[:, sort_order].plot(ax=a21, lw=1)
    a21.set(title=r'$\kappa_i(x)=E[X_i\mid X=x]$')
    a21.set(xlim=[0, xmax], ylim=[0, xmax], aspect='equal')
    a21.legend(loc='upper left')

    # layer-curve diagnostic frame (kappa/alpha/beta + layer margin and
    # cumulative margin); the core pricing frame no longer carries these.
    aug_df = port.allocation_diagnostics(distortion_name, surface='lifted')
    aug_df.filter(regex=f'exi_xgta_({port.unit_name_pipe})'). \
        rename(columns=_short_renamer(port, 'exi_xgta')). \
        sort_index(axis=1).plot(ylim=[-0.05, 1.05], ax=a22, lw=1)
    for ln, ls in zip(a22.lines, lss[1:]):
        ln.set_linestyle(ls)
    a22.legend()
    a22.set(xlim=[0, xmax], title=r'$\alpha_i(x)=E[X_i/X\mid X>x]$')

    bit = aug_df.query(f'loss < {xmax}').filter(regex=f'exi_xgtag?_({port.unit_name_pipe})')
    bit.rename(columns=_short_renamer(port, 'exi_xgtag')). \
        sort_index(axis=1).plot(ylim=[-0.05, 1.05], ax=a23)
    for i, l in enumerate(a23.lines[len(port.unit_names):]):
        if l.get_label()[0:3] == 'exi':
            a23.lines[i].set(linewidth=2, ls=lss[1 + i])
            l.set(color=f'C{i}', linestyle=lss[1 + i], linewidth=1,
                  alpha=.5, label=None)
    a23.legend(loc='upper left')
    a23.set(xlim=[0, xmax], title=r'$\beta_i(x)=E_{Q}[X_i/X \mid X> x]$')

    aug_df.filter(regex='^layer_margin_'). \
        rename(columns=_short_renamer(port, 'layer_margin')). \
        sort_index(axis=1).iloc[:, sort_order].plot(ax=a32, lw=1)
    a32.set(xlim=[0, xmax], title='Margin density $M_i(x)$')

    aug_df.filter(regex='^cum_margin_'). \
        rename(columns=_short_renamer(port, 'cum_margin')). \
        sort_index(axis=1).iloc[:, sort_order].plot(ax=a42, lw=1)
    a42.set(xlim=[0, xmax], title=r'Margin $\bar M_i(x)$')

    adf = aug_df.loc[:xmax]
    if kind == 'two':
        zipper = zip(range(2), sorted(port.unit_names), [a31, a41])
    else:
        zipper = zip(range(3), sorted(port.unit_names), [a31, a41, a33])
    for i, unit, a in zipper:
        a.plot(adf.loss, adf.S, c=col_list[2], ls=lss[1], lw=1, alpha=0.5, label='$S$')
        a.plot(adf.loss, adf.gS, c=col_list[2], ls=lss[0], lw=1, alpha=0.5, label='$g(S)$')
        a.plot(adf.loss, adf.S * adf[f'exi_xgta_{unit}'], c=col_list[i],
               ls=lss[1], lw=1, label=fr'$\alpha S$ {unit}')
        a.plot(adf.loss, adf.gS * adf[f'exi_xgtag_{unit}'], c=col_list[i],
               ls=lss[0], lw=1, label=fr'$\beta g(S)$ {unit}')
        a.set(xlim=[0, ymax])
        a.set(title=f'Unit = {unit}')
        a.legend()
        a.set(xlim=[0, ymax])
        a.legend(loc='upper right')

    alpha = 0.05
    if kind == 'two':
        ymax = ymax2 if ymax2 > 0 else port.q(p2)
        p2 = port.cdf(ymax)
        for cn, ln in enumerate(sort_order):
            unit = sorted(port.unit_names_ex)[ln]
            c = col_list[cn]
            s = lss[cn]
            # total from the portfolio frame; units native (numerics-1)
            ser = (port.density_df.p_total if unit == 'total'
                   else port.unit_density(unit))
            f1 = ser.cumsum()
            idx = (f1 < p2) & (f1 > 1.0 - p2)
            f1 = f1[idx]
            gf = 1 - port.distortion.g(1 - f1)
            x = f1.index
            a33.plot(gf, x, c=c, ls=s, lw=1, label=None)
            a33.plot(f1, x, ls=s, c=c, lw=1, label=None)
            a33.fill_betweenx(x, gf, f1, color=c, alpha=alpha, label=unit.title())
        a33.set(ylim=[0, ymax], title='Stand-alone $M$')
        a33.legend(loc='upper left')

    lw, up = port.q(1 - p2), ymax
    bit = aug_df.query(f' {lw} <= loss <= {up} ')
    F = bit['F']
    gF = bit['gF']
    for cn, ln in enumerate(sort_order):
        unit = sorted(port.unit_names_ex)[ln]
        c = col_list[cn]
        s = lss[cn]
        ser = bit[f'exeqa_{unit}']
        if kind == 'three':
            a43.plot(1 / (1 - F), ser, lw=1, ls=lss[1], c=c)
            a43.plot(1 / (1 - gF), ser, lw=1, c=c)
            a43.set(xlim=[1, 1e4], xscale='log')
            a43.fill_betweenx(ser, 1 / (1 - gF), 1 / (1 - F),
                              color=c, alpha=alpha, lw=0.5, label=unit.title())
        else:
            a43.plot(F, ser, lw=1, ls=s, c=c)
            a43.plot(gF, ser, lw=1, ls=s, c=c)
            a43.fill_betweenx(ser, gF, F,
                              color=c, alpha=alpha, lw=0.5, label=unit.title())
    a43.set(ylim=[0, ymax], title='Natural $M$')
    a43.legend(loc='upper left')

    if legend_font:
        for ax in axs.flat:
            try:
                if ax is not a13:
                    ax.legend(prop={'size': 7})
            except Exception:
                pass


# ======================================================================
# Exponential tilting (Grübel–Hermesmeier aliasing-control illustration)
# ======================================================================
# Tilting is a *teaching device* about FFT aliasing control; it is NOT part of
# the production convolution path, which controls aliasing purely through
# padding (the ``tilt``/``tilt_vector`` arguments were dropped from the core
# ``ft``/``ift``/``update_work`` in the 1.0 refactor). These helpers keep the
# tilted transform entirely local so the numerical-methods docs can reproduce
# the Grübel & Hermesmeier (1999) example without touching the core.


def tilt_vector(theta, n):
    r"""Exponential tilt vector :math:`e^{-\theta k}` for ``k = 0 \dots n-1``.

    Parameters
    ----------
    theta : float
        Tilt amount :math:`\theta`. Embrechts–Frei recommend
        :math:`\theta\, n \le 20` to keep the untilting amplification bounded.
    n : int
        Grid length (``2**log2``).

    Returns
    -------
    numpy.ndarray
        The length-``n`` vector ``exp(-theta * arange(n))``.
    """
    return np.exp(-theta * np.arange(n))


def tilted_aggregate_density(agg, *, log2, bs, padding=0, tilt=None,
                             normalize=False):
    r"""Aggregate density via FFT with optional exponential tilting (pedagogy).

    Illustrates Grübel–Hermesmeier exponential tilting for aliasing control. The
    severity density :math:`p` is multiplied by the tilt vector
    :math:`e^{-\theta k}`, transformed, run through the frequency PGF, inverted,
    and un-tilted by :math:`e^{+\theta k}`. Tilting damps the wrapped (aliased)
    mass before the transform so it does not contaminate the low buckets, at the
    cost of amplifying round-off in the tail on the un-tilt — hence the
    :math:`\theta\, n \le 20` rule of thumb.

    **Not part of the production update path** — a self-contained teaching helper
    that mirrors the pre-refactor ``ft``/``ift`` tilt logic locally. The core
    convolution is tilt-free and controls aliasing through padding.

    Parameters
    ----------
    agg : Aggregate or str
        A built or buildable aggregate; only its discretized severity and
        frequency PGF are used. The object is updated in place onto the
        ``(log2, bs)`` grid.
    log2 : int
        Grid is ``N = 2**log2`` buckets.
    bs : float
        Bucket size.
    padding : int, default 0
        FFT zero-padding factor (transform length ``N << padding``).
    tilt : float or None, default None
        Tilt amount :math:`\theta`; ``None`` (or ``0``) means no tilt. The tilt
        vector is :math:`e^{-\theta k}`; :math:`\theta\, N \le 20` recommended.
    normalize : bool, default False
        Severity normalization passed to ``update``. ``False`` matches the GH
        doc example, which compares un-normalized tail probabilities.

    Returns
    -------
    pandas.Series
        Aggregate density indexed by loss (``agg.xs``), name ``p_total`` — read
        ``series / bs`` for the density at a chosen loss, as the doc does.

    Notes
    -----
    With ``tilt=None`` this reproduces the ordinary (untilted) convolution on the
    same grid byte-for-byte: the transform is ``irfft(freq_pgf(n, rfft(p, M)))``
    with ``M = N << padding``, exactly the non-negative path of
    ``Aggregate._fft_aggregate``. Only the non-negative severity case is covered
    (the GH example is Levy on ``[0, ∞)``); signed severities are out of scope
    for this illustration.
    """
    if isinstance(agg, str):
        agg = build(agg, update=False)
    # Discretize the severity and set up the frequency PGF on the target grid.
    # (update also runs the untilted convolution; we ignore its agg_density and
    # re-transform below with the tilt applied.)
    agg.update(log2=log2, bs=bs, padding=padding, normalize=normalize)
    N = 1 << log2
    p = np.asarray(agg.sev_density, dtype=float)
    tv = tilt_vector(tilt, N) if tilt else None

    # Classic tilted transform, kept local: z*tilt -> rfft -> freq_pgf -> irfft
    # -> /tilt. Padding is handled by the rfft length (N << padding).
    z = sfft.rfft(p * tv if tv is not None else p, N << padding)
    # base_mean, not n: under zm / zt these differ (n is the shifted mean)
    ftagg = agg.frequency.freq_pgf(agg.base_mean, z)
    a = np.real(sfft.irfft(ftagg, N << padding))
    if padding:
        a = a[:N]
    if tv is not None:
        a = a / tv
    return pd.Series(a, index=agg.xs, name='p_total')


def _gh_levy_exact(x, claims):
    r"""Exact aggregate density at integer ``x`` for the Poisson/Levy example.

    The Levy distribution is :math:`\alpha = 1/2` stable, so a sum of ``i`` iid
    Levy variables is distributed as :math:`i^2 X`. Conditioning on the Poisson
    claim count gives the exact aggregate probability of the bucket
    :math:`(x-\tfrac12,\, x+\tfrac12)` as
    :math:`\sum_i P(N=i)\,[F((x+\tfrac12)/i^2) - F((x-\tfrac12)/i^2)]`.

    Parameters
    ----------
    x : float
        Loss bucket centre.
    claims : float
        Poisson mean claim count :math:`\lambda`.

    Returns
    -------
    float
        The exact bucket probability (truncating the Poisson sum at 100 terms).
    """
    lam = claims
    n = 100
    p = np.zeros(n)
    contrib = np.zeros(n)
    p[0] = np.exp(-lam)
    fz = ss.levy()
    for i in range(1, n):
        p[i] = p[i - 1] * lam / i
        contrib[i] = fz.cdf((x + 0.5) / i ** 2) - fz.cdf((x - 0.5) / i ** 2)
    return float(np.sum(p * contrib))


def gh_tilting_exhibit(claims=20, bs=1, accurate_log2=16, tilt_log2=10,
                       tilts=(None, 1 / 1024, 5 / 1024, 25 / 1024),
                       index=(1, 10, 100, 1000)):
    r"""Reproduce Grübel & Hermesmeier (1999) Table 1 (Poisson/Levy tilting).

    Assembles the full comparison in one call: an ``accurate`` column (a
    high-resolution padded convolution), an ``exact`` column (closed-form via
    Levy stability, :func:`_gh_levy_exact`), and a sweep of tilted convolutions
    at a coarse grid showing aliasing reduction as the tilt increases.

    Parameters
    ----------
    claims : float, default 20
        Poisson mean claim count.
    bs : float, default 1
        Bucket size.
    accurate_log2 : int, default 16
        ``log2`` for the high-resolution accurate column (padded).
    tilt_log2 : int, default 10
        ``log2`` for the coarse tilted columns (no padding) — the grid where
        aliasing bites and tilting helps.
    tilts : iterable of (float or None), default ``(None, 1/1024, 5/1024, 25/1024)``
        Tilt amounts for the sweep; ``None`` is the untilted baseline.
    index : iterable of int, default ``(1, 10, 100, 1000)``
        Loss buckets to tabulate.

    Returns
    -------
    pandas.DataFrame
        Rows indexed by loss; columns ``accurate``, ``exact``, and one
        ``tilt {theta:.4f}`` per swept tilt. Each cell is a density
        (probability / ``bs``).

    Notes
    -----
    With ``bs=1`` the table matches the published paper table: the untilted
    coarse column is badly aliased at small ``x`` and converges to the accurate /
    exact column as the tilt rises toward :math:`\theta\, N = 25`.
    """
    idx = list(index)
    a = build(f'agg L {claims} claim sev levy poisson', update=False)
    a.update(log2=accurate_log2, bs=bs, padding=2, normalize=False)
    df = a.density_df.loc[idx, ['p_total']] / a.bs
    df.columns = ['accurate']
    df['exact'] = [_gh_levy_exact(x, claims) for x in idx]
    for tilt in tilts:
        series = tilted_aggregate_density(a, log2=tilt_log2, bs=bs, padding=0,
                                          tilt=tilt)
        key = 0.0 if tilt is None else tilt
        df[f'tilt {key:.4f}'] = series.loc[idx].to_numpy() / bs
    return df
