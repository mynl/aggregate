"""Layer 2 compositor for :class:`aggregate.ft.FourierTools`.

Holds the matplotlib-backed plot bodies of ``FourierTools`` -- ``plot`` (the
six-panel density / transform exhibit), ``plot_wraps`` (aliasing illustration),
``plot_simpson`` (Simpson approximation) and the ``_plot_fourier1d`` content
worker. The class keeps one-line delegating stubs.

The ``plot_fourier3d`` / ``plot_fourier3da`` methods stay on the class: they use
mayavi / plotly, not matplotlib, so they fall outside this subsystem's boundary.

Single-consumer content: the FourierTools exhibits have no cross-class reuse, so
their content workers and compositors collapse into this one module.
"""

import logging

import numpy as np
import pandas as pd

from ._style import plt, ticker, _fit_to_page

logger = logging.getLogger(__name__)


def plot_fourier1d(ftools, ax, min_abs=1e-20):
    """Simple plot of the Fourier transform on one provided ``Axes``."""
    fhat = ftools._fourier.copy()
    c = np.abs(fhat)
    num_large = np.sum(c > min_abs)
    fhat = fhat[:num_large]
    c = c[:num_large]

    ax.plot(np.real(fhat), np.imag(fhat), '-o', ms=2, lw=.5, label='ft')
    fhat = fhat[c > 0] / c[c > 0]
    ax.plot(np.real(fhat), np.imag(fhat), '-o', ms=1, lw=.5, label='ft / |ft|')
    lim = [-1.05, 1.05]
    ax.set(xlim=lim, ylim=lim, aspect='equal', xlabel='real(ft)',
           ylabel='imag(ft)', title='Fourier transform')
    ax.legend(loc='upper left')
    ax.axhline(0, c='k', alpha=.5, lw=.5)
    ax.axvline(0, c='k', alpha=.5, lw=.5)


def plot_fourier(ftools, suptitle='', xlim=None, verbose=True):
    """Compare density, log density, and the amplitude/argument of the transform.

    Parameters
    ----------
    ftools : FourierTools
        Must have been inverted (``invert()``); ``compute_exact()`` optional.
    suptitle : str, default ''
        Super title for the figure.
    xlim : tuple, optional
        x-axis limits; inferred from the grid if omitted.
    verbose : bool, default True
        Six-panel "full monty" if True, else the two density panels only.
    """
    assert ftools._df is not None, 'Must recompute first. Run invert() and compute_exact().'
    has_exact = ftools._df_exact is not None
    if not has_exact:
        logger.warning('No exact! Maybe run compute_exact().')

    # plot four graphs per invert()
    if verbose:
        ftools.last_fig, axs = plt.subplots(2, 3, figsize=_fit_to_page((3 * 2.5, 2 * 2)), constrained_layout=True)
        ax0, ax1, ax2, ax3, ax4, ax5 = axs.flat
    else:
        ftools.last_fig, axs = plt.subplots(1, 2, figsize=_fit_to_page((2 * 2.5, 1 * 2.5)), constrained_layout=True)
        ax0, ax1 = axs.flat

    x = np.array(ftools._df.index)
    p = ftools._df.p.values
    b = x[1] - x[0]
    if has_exact:
        xe = np.array(ftools._df_exact.index)
        pe = ftools._df_exact.p.values
        be = xe[1] - xe[0]
    else:
        # avoid an error below when no exact
        pe = ftools.df.p.values
    if xlim is None:
        if x[0] == 0:
            lower = -x[-1] / 25
        else:
            lower = x[0]
        xlim = [lower, x[-1]]

    for ax in [ax0, ax1]:
        ax.plot(x, p / b, label='Fourier', lw=1)
        if has_exact:
            ax.plot(xe, pe, ls='--', c='C3', label='exact', lw=1)
        ax.legend()
    ax0.set(xlim=xlim, title='Density', xlabel='Outcome, x')
    # mn = min(np.log10(exact).min(), np.log10(x).min())
    mn0 = np.log10(p / b).min() * 1.25
    mn = 10 ** np.floor(mn0)
    mx = max(np.log10(pe).max(), np.log10(x).max())
    mx = 10 ** np.ceil(mx)
    if np.isnan(mn):
        mn = 1e-17
    if np.isnan(mx):
        mx = 1
    ax1.set(yscale='log', ylim=[mn, mx], xlim=xlim, title='Log density', xlabel='Outcome, x')
    if not verbose:
        if suptitle != '':
            ftools.last_fig.suptitle(suptitle)
        return

    # else: verbose mode: full monty with six plots
    if ftools.discrete and len(ftools._df) <= 64:
        drawstyle = 'steps-post'
    else:
        drawstyle = 'default'
    for ax in [ax2, ax5]:
        ax.plot(x, np.cumsum(p), label='cdf Fourier', lw=1,
                c='C0', drawstyle=drawstyle)
        if has_exact:
            ax.plot(xe, np.cumsum(pe * be), label='cdf exact', ls='--', lw=.5,
                c='C3', drawstyle=drawstyle)
        ax.plot(x, np.cumsum(p[::-1])[::-1], label='sf Fourier', lw=1,
                c='C2', drawstyle=drawstyle)
        if has_exact:
            ax.plot(xe, np.cumsum(pe[::-1] * be)[::-1], label='sf exact', ls='--', lw=.5,
                c='C4', drawstyle=drawstyle)

    ax2.set(title='sf and cdf', xlabel='Outcome, x', xlim=xlim)
    ax2.legend()
    ax5.set(yscale='log', title='log sf and cdf', ylim=[mn, 10],
            xlim=xlim, xlabel='Outcome, x')

    # ft on ax3, probably should inline this
    plot_fourier1d(ftools, ax3)

    # amplitude and phase both on ax4
    ax4r = ax4.twinx()
    ax4.plot(ftools._ts, np.abs(ftools._fourier), '-', lw=1.5, c='C4', label='Amplidude')
    # for amplitude, only look at nonzero fts, find index of non-zero (inz) items
    inz = np.abs(ftools._fourier) > 0
    tnz = ftools._ts[inz]
    fnz = ftools._fourier[inz]
    anz = np.angle(fnz)
    uw = np.unwrap(anz)
    if len(ftools._df) <= 256:
        kw = {'ls': '-', 'marker': '.', 'ms': 3, 'lw': 1}
    else:
        kw = {'lw': 1}
    ax4r.plot(tnz, uw, c='C2', **kw, label='phase, unwrapped')
    kw['lw'] = 0.5
    kw['marker'] = None
    kw['ls'] = ':'
    ax4r.plot(tnz, anz, c='C2', **kw, label='phase, wrapped')
    ax4r.legend(loc='upper right')
    ax4.legend(loc='center right')
    ax4.set(ylabel='Amplitude |ft|', yscale='log', xlabel='frequency', xlim=[-.05, 0.5 / ftools.bs + 0.05])
    if ftools.bs == 1:
        ax4.set(xticks=[0, .25, .5])
    ax4r.set(title='Amplitude and phase',
             ylabel='Phase / 2π', xlabel='frequency / 2π')
    if suptitle != '':
        ftools.last_fig.suptitle(suptitle)


def plot_fourier_wraps(ftools, wraps=None, calc='survival', add_tail=False):
    """Illustrate wrapping/aliasing. Slow unless ``fz.pdf`` is cheap.

    Parameters
    ----------
    ftools : FourierTools
        Must have been inverted and have ``compute_exact()`` output.
    wraps : list, optional
        Wrap values, e.g. ``[-1, 1]`` for one above and one below ``[0, P)``.
    calc : {'survival', 'density'}, default 'survival'
        How to estimate the density outside the base range.
    add_tail : bool, default False
        Plot the shifted exact densities in panels 2 and 4.

    Returns
    -------
    pandas.DataFrame
        Per-wrap probability table.
    """
    assert ftools._df is not None, 'Must recompute first. Run invert().'
    assert ftools._df_exact is not None, "Must run compute_exact() first."
    # extract values
    x = np.array(ftools._df.index)
    b = x[1] - x[0]
    p = ftools._df['p'].values / b    # here and below divide by b to convert to a density
    xe = np.array(ftools._df_exact.index)
    be = xe[1] - xe[0]
    pe = ftools._df_exact.p.values
    x_range = ftools.x_max - ftools.x_min
    rt = pe.copy()

    # duplicated...
    mn0 = np.log10(x).min() * 1.25
    mn = 10 ** np.floor(mn0)
    mx = max(np.log10(p).max(), np.log10(x).max())
    mx = 10 ** np.ceil(mx)
    if np.isnan(mn):
        mn = 1e-17
    if np.isnan(mx):
        mx = 1

    # report answer
    ans = []
    ftools.last_fig, axs = plt.subplots(2, 2, figsize=_fit_to_page((2 * 2.5, 2 * 2.)), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axs.flat
    for ax in axs.flat:
        ax.plot(x, p, label='Fourier', lw=2)
    lw = .5
    if calc == 'density':
        if hasattr(ftools.fz, 'pdf'):
            pdf = ftools.fz.pdf
        elif hasattr(ftools.fz, 'pmf'):
            pdf = ftools.fz.pmf
        else:
            raise ValueError('fz must have pdf or pmf method for density method')
    for i, b_wrap in enumerate(wraps):
        if type(b_wrap) == int:
            bl = f'{b_wrap:d}'
        else:
            bl = f'{b_wrap:.3f}'
        xs2 = b_wrap * x_range + xe
        if calc == 'survival':
            # for computing probs using survival method
            xs2d = np.hstack((xs2 - be / 2, xs2[-1] + be / 2))
            adj = -np.diff(ftools.fz.sf(xs2d)) / be
            rt += adj
        elif calc == 'density':
            adj = pdf(xs2)
            rt += adj
        else:
            raise ValueError('calc must be "survival" or "density"')
        c = f'C{i+1}'
        ax0.plot(xe, rt, label=bl, lw=lw, c=c)
        ax1.plot(xe, adj, label=bl, lw=lw, c=c)
        if add_tail:
            ax1.plot(xs2, adj, label=bl, lw=lw, c=c, ls=':')
            ax3.plot(xs2, adj, label=bl, lw=lw, c=c, ls=':')
        ax2.plot(xe, rt, label=bl, lw=lw, c=c)
        ax3.plot(xe, adj, label=bl, lw=lw, c=c)
        ans.append([b_wrap, xs2[0], xs2[-1], ftools.fz.cdf(xs2[-1]), ftools.fz.cdf(xs2[0]),
                    ftools.fz.cdf(xs2[-1]) - ftools.fz.cdf(xs2[0])])
    for ax in axs.flat:
        ax.plot(xe, pe, label='exact', ls='--', lw=1, c='C3')
    ax0.set(yscale='linear',
            title='Cumulative aliasing',
            xlabel='Outcome, x',
            ylabel='Density')
    ax2.set(yscale='log',
            title='Cumulative - log scale',
            xlabel='Outcome, x',
            ylabel='Log density')
    ax1.set(yscale='linear',
            title='Incremental aliasing',
            xlabel='Outcome, x',
            ylabel='Density')
    ax3.set(yscale='log',
            title='Incremental - log scale',
            xlabel='Outcome, x',
            ylabel='Log density')
    ax2.legend(ncol=2)
    if add_tail:
        ax1.legend(ncol=2)
    if add_tail:
        if 0 not in wraps:
            wraps.append(0)
        if 1 not in wraps:
            wraps.append(1)
        wraps.append(max(wraps) + 1)
        wraps = sorted(wraps)
        for b_wrap in wraps:
            ax1.axvline(x[0] + b_wrap * x_range, lw=.25, c='k', ls=':')
            ax3.axvline(x[0] + b_wrap * x_range, lw=.25, c='k', ls=':')
        for ax in [ax1, ax3]:
          ax.set(xticks=ftools.x_min + np.arange(0, max(wraps), 2) * ftools.x_max)
          ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    df = pd.DataFrame(ans, columns=['Wrap', 'x0', 'x1', 'Pr(X≤x1)', 'Pr(X≤x0)', 'Pr(X in Wrap)'])
    df = df.set_index('Wrap')
    return df


def plot_fourier_simpson(ftools, ylim=1e-16):
    """Plot Simpson's approximation."""
    fig, ax = plt.subplots(1, 1, figsize=_fit_to_page((5, 3.25)), constrained_layout=True)
    if ftools._df_exact is not None:
        ftools.df_exact.p.plot(ax=ax, lw=1, c='C1')
    xs = np.array(ftools.df.index)
    if 'p' in ftools.df:
        ax.plot(xs, ftools.df.p / ftools.bs, label='basic', c='C0', lw=1)
    ax.plot(xs, ftools.df.simpson / ftools.bs, label='simpson', c='C3', ls=':')
    ax.set(yscale='log', ylim=ylim)
    ax.legend()


# ======================================================================
# The same pictures over an AggregateCHF ([Characteristic-Function-Pedagogy]).
# The engine holds bucket masses on ``xs`` with bucket ``bs``; the exact law
# comes from the family's reference where there is one.
# ======================================================================

def _chf_reference_density(obj, max_points=257):
    """``(x, density)`` of the family's reference law at a few hundred grid points, or ``None``."""
    from ..pedagogy import chf_exact_comparison
    if obj.family.reference is None:
        return None
    df = chf_exact_comparison(obj, max_points=max_points)
    return df.index.to_numpy(), df['exact'].to_numpy()


def plot_chf_inversion(obj, suptitle='', xlim=None, verbose=True):
    """Density, log density, cdf and sf, and the sampled transform of an ``AggregateCHF``.

    Parameters
    ----------
    obj : AggregateCHF
        Updated.
    suptitle : str
    xlim : tuple, optional
        Inferred from the grid if omitted.
    verbose : bool
        Six panels if True, else the two density panels only.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x, p, h = obj.xs, obj.agg_density, obj.bs
    exact = _chf_reference_density(obj)
    if verbose:
        fig, axs = plt.subplots(2, 3, figsize=_fit_to_page((3 * 2.5, 2 * 2)),
                                constrained_layout=True)
        ax0, ax1, ax2, ax3, ax4, ax5 = axs.flat
    else:
        fig, axs = plt.subplots(1, 2, figsize=_fit_to_page((2 * 2.5, 2.5)),
                                constrained_layout=True)
        ax0, ax1 = axs.flat
    if xlim is None:
        xlim = (x[0] if x[0] < 0 else -x[-1] / 25, x[-1])
    for ax in (ax0, ax1):
        ax.plot(x, p / h, label='grid', lw=1)
        if exact is not None:
            ax.plot(*exact, ls='--', c='C3', label='exact', lw=1)
        ax.legend()
    ax0.set(xlim=xlim, title='Density', xlabel='Outcome, x', ylabel='Density')
    pos = p > 0
    ax1.set(xlim=xlim, yscale='log', title='Log density', xlabel='Outcome, x',
            ylim=(max(float(p[pos].min()) / h, 1e-17), float(p.max()) / h * 2))
    if not verbose:
        if suptitle:
            fig.suptitle(suptitle)
        return fig
    F = np.cumsum(p)
    for ax in (ax2, ax5):
        ax.plot(x, F, label='cdf', lw=1, c='C0')
        ax.plot(x, 1 - F, label='sf', lw=1, c='C2')
        ax.legend()
    ax2.set(xlim=xlim, title='Distribution', xlabel='Outcome, x')
    ax5.set(xlim=xlim, yscale='log', title='Distribution, log scale', xlabel='Outcome, x')
    n = len(x)
    t = 2 * np.pi * np.arange(n // 2 + 1) / (n * h)
    phi = obj.family.chf(t)
    kw = {'ls': '-', 'marker': '.', 'ms': 3, 'lw': 1} if n <= 256 else {'lw': 1}
    ax3.plot(t / (2 * np.pi), np.abs(phi), **kw)
    ax3.set(yscale='log', title='Amplitude of the transform', xlabel='frequency / 2π',
            ylabel='|φ(t)|', ylim=(1e-20, 2))
    ax4.plot(t / (2 * np.pi), np.angle(phi) / (2 * np.pi), **kw)
    ax4.set(title='Phase of the transform', xlabel='frequency / 2π', ylabel='Phase / 2π')
    if suptitle:
        fig.suptitle(suptitle)
    return fig


def plot_chf_wraps(obj, wraps=(-1, 1), add_tail=False, max_points=257):
    """Illustrate aliasing: the exact law shifted by whole windows and summed onto the grid.

    Parameters
    ----------
    obj : AggregateCHF
        Updated, with a scipy reference law.
    wraps : iterable of int
        Window shifts, ``[-1, 1]`` for one window below and one above.
    add_tail : bool
        Also draw each shifted density where it sits.
    max_points : int
        Reference evaluations per wrap.

    Returns
    -------
    tuple
        ``(figure, table)``; the table has one row per wrap with its
        interval and the exact probability in it.
    """
    ref = obj.family.reference
    if ref is None or hasattr(ref, 'agg_density'):
        raise ValueError(f'{obj.name}: the wrapping figure needs a scipy reference law')
    x, p, h = obj.xs, obj.agg_density, obj.bs
    W = len(x) * h
    xe, pe = _chf_reference_density(obj, max_points)
    be = xe[1] - xe[0]
    fig, axs = plt.subplots(2, 2, figsize=_fit_to_page((2 * 2.5, 2 * 2)),
                            constrained_layout=True)
    ax0, ax1, ax2, ax3 = axs.flat
    for ax in axs.flat:
        ax.plot(x, p / h, label='grid', lw=2)
    running = pe.copy()
    rows = []
    for i, w in enumerate(list(wraps)):
        xs2 = w * W + xe
        edges = np.append(xs2 - be / 2, xs2[-1] + be / 2)
        adj = -np.diff(ref.sf(edges)) / be
        running = running + adj
        c = f'C{i + 1}'
        ax0.plot(xe, running, label=f'{w:d}', lw=1, c=c)
        ax1.plot(xe, adj, label=f'{w:d}', lw=1, c=c)
        ax2.plot(xe, running, lw=1, c=c)
        ax3.plot(xe, adj, lw=1, c=c)
        if add_tail:
            ax1.plot(xs2, adj, lw=1, c=c, ls=':')
            ax3.plot(xs2, adj, lw=1, c=c, ls=':')
        rows.append([w, xs2[0], xs2[-1], float(ref.cdf(xs2[-1]) - ref.cdf(xs2[0]))])
    for ax in axs.flat:
        ax.plot(xe, pe, label='exact', ls='--', lw=1, c='C3')
    ax0.set(title='Cumulative aliasing', xlabel='Outcome, x', ylabel='Density')
    ax1.set(title='Incremental aliasing', xlabel='Outcome, x', ylabel='Density')
    ax2.set(title='Cumulative, log scale', xlabel='Outcome, x', yscale='log')
    ax3.set(title='Incremental, log scale', xlabel='Outcome, x', yscale='log')
    ax0.legend(ncol=2)
    table = pd.DataFrame(rows, columns=['wrap', 'from', 'to', 'probability']).set_index('wrap')
    return fig, table


def plot_chf_simpson(obj, ylim=1e-16):
    """The grid's density against Simpson's rule on the Fourier integral, on a log axis."""
    from ..pedagogy import chf_simpson_inversion
    df = chf_simpson_inversion(obj)
    fig, ax = plt.subplots(figsize=_fit_to_page((3, 2.25)), constrained_layout=True)
    ax.plot(df.index, df['grid'], label='grid mass / bs', c='C0', lw=1)
    ax.plot(df.index, df['simpson'], label='Simpson', c='C1', lw=1, ls='--')
    exact = _chf_reference_density(obj)
    if exact is not None:
        ax.plot(*exact, label='exact', c='C3', lw=1, ls=':')
    ax.set(yscale='log', ylim=(ylim, None), xlabel='Outcome, x', ylabel='Density',
           title=f'{obj.label}: two inversions')
    ax.legend()
    return fig
