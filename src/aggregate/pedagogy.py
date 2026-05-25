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
from itertools import cycle
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from .bounds import Bounds
from .spectral import Distortion
from .underwriter import Underwriter

logger = logging.getLogger(__name__)


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


def similar_risks_graphs_sa(axd, bounds, port, pnew, roe, prem, p_reg=1):
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

    roe_d = Distortion('roe', roe)
    tvar_d = Distortion('tvar', bounds.p_star)
    idx = df.index.get_locs(df.idxmax()['t1'])[0]
    pl, pu, tl, tu, w = df.reset_index().iloc[idx, :-4]
    max_d = Distortion('wtdtvar', [pl, pu], df=[1 - w, w])

    tmax = float(df.iloc[idx]['t1'])
    n_ = len(df.query('t1 == @tmax'))
    logger.warning(f'Ties for max: {n_}')
    n_ = len(df.query(f't1 >= {tmax} - 1e-4'))
    logger.warning(f'Near ties for max: {n_}')

    idn = df.index.get_locs(df.idxmin()['t1'])[0]
    pln, pun, tl, tu, wn = df.reset_index().iloc[idn, :-4]
    min_d = Distortion('wtdtvar', [pln, pun], df=[1 - wn, wn])

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
    bounds.plot_envelope(axs=axs.flatten(), n_resamples=0, alpha=1, pricing=True,
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
    df = similar_risks_graphs_sa(axd, bounds, p_base, p_new, roe, prem)
    return df
