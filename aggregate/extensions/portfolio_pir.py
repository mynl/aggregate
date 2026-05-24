"""PIR-book exhibit machinery extracted from Portfolio.

These functions were originally methods on ``aggregate.portfolio.Portfolio``
that generated the exhibits and figures for *Pricing Insurance Risk* (Mildenhall
& Major, 2022). They are book-specific presentation code rather than core
portfolio machinery, and they were moved out of ``Portfolio`` in the 1.0
refactor so the core class stays small.

Use them as free functions taking a ``Portfolio`` as the first argument::

    from aggregate.extensions.portfolio_pir import premium_capital
    premium_capital(port, a=10_000)

They mutate ``port`` by attaching ``EX_*`` attributes where the originals did
(e.g. ``port.EX_premium_capital``).

Contents
--------

- Renamers: ``renamer`` (the big density-frame → display-name dictionary),
  ``short_renamer``, ``premium_capital_renamer`` (module-level dict).
- Premium-capital and accounting exhibits: ``premium_capital``,
  ``multi_premium_capital``, ``accounting_economic_balance_sheet``,
  ``make_all``, ``show_enhanced_exhibits``, ``set_a_p``.
- Plots: ``profit_segment_plot``, ``natural_profit_segment_plot``,
  ``density_sample``, ``biv_contour_plot``, ``twelve_plot``.
- Layer effectiveness: ``gamma``.
- Stand-alone pricing: ``stand_alone_pricing_work``, ``stand_alone_pricing``,
  ``calibrate_blends`` (with helpers ``check01``, ``make_array``,
  ``convex_points``).
- Bulk constructors: ``from_DataFrame``, ``from_Excel``, ``from_dict_of_aggs``.
"""
from __future__ import annotations

import logging
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pandas.io.formats.format import EngFormatter
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from IPython.display import HTML, display

from ..constants import WL
from ..spectral import Distortion
from ..results import GammaResult
from ..utilities import subsets


logger = logging.getLogger(__name__)


__all__ = [
    'premium_capital_renamer',
    'renamer',
    'short_renamer',
    'set_a_p',
    'premium_capital',
    'multi_premium_capital',
    'accounting_economic_balance_sheet',
    'make_all',
    'show_enhanced_exhibits',
    'profit_segment_plot',
    'natural_profit_segment_plot',
    'density_sample',
    'biv_contour_plot',
    'twelve_plot',
    'gamma',
    'stand_alone_pricing_work',
    'stand_alone_pricing',
    'calibrate_blends',
    'from_DataFrame',
    'from_Excel',
    'from_dict_of_aggs',
    'check01',
    'make_array',
    'convex_points',
]


# ---------------------------------------------------------------------------
# Renamers
# ---------------------------------------------------------------------------

premium_capital_renamer = {
    'Assets':  '0. Assets',
    'T.A':     '1. Allocated assets',
    'T.P':     '2. Market value liability',
    'T.L':     '3. Expected incurred loss',
    'T.M':     '4. Margin',
    'T.LR':    '5. Loss ratio',
    'T.Q':     '6. Allocated equity',
    'T.ROE':   '7. Cost of allocated equity',
    'T.PQ':    '8. Premium to surplus ratio',
    'EPD':     '9. Expected pol holder deficit',
}


def renamer(port):
    """Build a dict of friendly column names for ``port.density_df``.

    Caches the result on ``port._pir_renamer`` so repeated calls are cheap.
    The mapping covers density / lev / exa / exag / exi family columns,
    augmented_df T-/M-/V- columns, gamma columns, and assorted symbolic
    forms used by the PIR exhibits.
    """
    if getattr(port, '_pir_renamer', None) is None:
        out = {}
        meta_namer = dict(
            p_=('', ' density'),
            lev_=('LEV[', 'a]'),
            exag_=('EQ[', '(a)]'),
            exa_=('E[', '(a)]'),
            exlea_=('E[', ' | X<=a]'),
            exgta_=('E[', ' | X>a]'),
            exeqa_=('E[', ' | X=a]'),
            e1xi_1gta_=('E[1/', ' 1(X>a)]'),
            exi_x_=('E[', '/X]'),
            exi_xgta_sum=('Sum Xi/X gt', ''),
            exi_xeqa_sum=('Sum Xi/X eq', ''),
            exi_xgta_=('α=E[', '/X | X>a]'),
            exi_xeqa_=('E[', '/X | X=a]'),
            exi_xlea_=('E[', '/X | X<=a]'),
            epd_0_=('EPD(', ') stand alone'),
            epd_1_=('EPD(', ') within X'),
            epd_2_=('EPD(', ') second pri'),
            e2pri_=('E[X', '(a) second pri]'),
            ημ_=('All but ', ' density'),
        )

        for l in port.density_df.columns:
            if re.search('^ημ_', l):
                out[l] = re.sub(r'^ημ_([0-9A-Za-z\-_.,]+)', r'not \1 density', l)
            else:
                l0 = l.replace('ημ_', 'not ')
                for k, v in meta_namer.items():
                    d1 = l0.find(k)
                    if d1 >= 0:
                        d1 += len(k)
                        b, a = v
                        out[l] = f'{b}{l0[d1:]}{a}'.replace('total', 'X')
                        break

        for l in port.line_names_ex:
            out[f'exag_{l}'] = f'EQ[{l}(a)]'
            out[f'exi_xgtag_{l}'] = f'β=EQ[{l}/X | X>a]'
            out[f'exi_xleag_{l}'] = f'EQ[{l}/X | X<=a]'
            out[f'e1xi_1gta_{l}'] = f'E[{l}/X 1(X >a)]'
        out['exag_sumparts'] = 'Sum of EQ[Xi(a)]'

        for pre, m1 in zip(['M', 'T'], ['Marginal', 'Total']):
            for post, m2 in zip(['L', 'P', 'LR', 'Q', 'ROE', 'PQ', 'M'],
                                ['Loss', 'Premium', 'Loss Ratio', 'Equity',
                                 'ROE', 'Leverage (P:S)', 'Margin']):
                out[f'{pre}.{post}'] = f'{m1} {m2}'
        for line in port.line_names_ex:
            for pre, m1 in zip(['M', 'T'], ['Marginal', 'Total']):
                for post, m2 in zip(['L', 'P', 'LR', 'Q', 'ROE', 'PQ', 'M'],
                                    ['Loss', 'Premium', 'Loss Ratio',
                                     'Equity', 'ROE', 'Leverage (P:S)',
                                     'Margin']):
                    out[f'{pre}.{post}_{line}'] = f'{m1} {m2} {line}'
        out['A'] = 'Assets'

        for l in port.line_names:
            out[f'gamma_{l}_sa'] = f'γ {l} stand-alone'
            out[f'gamma_{port.name}_{l}'] = f'γ {l} part of {port.name}'
            out[f'p_{l}'] = f'{l} stand-alone density'
            out[f'S_{l}'] = f'{l} stand-alone survival'
        out['p_total'] = f'{port.name} total density'
        out['S_total'] = f'{port.name} total survival'
        out[f'gamma_{port.name}_total'] = f'γ {port.name} total'

        port._pir_renamer = out

    out = port._pir_renamer
    out['mv'] = '$\\mathit{MVL}(a)$??'
    for orig in port.line_names_ex:
        l = port.line_renamer.get(orig, orig).replace('$', '')
        if orig == 'total':
            out['S'] = f'$S_{{{l}}}(a)$'
        else:
            out[f'S_{orig}'] = f'$S_{{{l}}}(a)$'
        out[f'lev_{orig}'] = f'$E[{l}\\wedge a]$'
        out[f'levg_{orig}'] = f'$\\rho({l}\\wedge a)$'
        out[f'exa_{orig}'] = f'$E[{l}(a)]$'
        out[f'exag_{orig}'] = f'$\\rho({l}\\subseteq X^c\\wedge a)$'
        out[f'ro_da_{orig}'] = '$\\Delta a_{ro}$'
        out[f'ro_dmv_{orig}'] = '$\\Delta \\mathit{MVL}_{ro}(a)$'
        out[f'ro_eva_{orig}'] = '$\\mathit{EGL}_{ro}(a)$'
        out[f'gc_da_{orig}'] = '$\\Delta a_{gc}$'
        out[f'gc_dmv_{orig}'] = '$\\Delta \\mathit{MVL}_{gc}(a)$'
        out[f'gc_eva_{orig}'] = '$\\mathit{EGL}_{gc}(a)$'
    return out


def short_renamer(port, prefix='', postfix=''):
    """Map ``f'{prefix}_{line}_{postfix}'`` columns to title-cased line names."""
    if prefix:
        prefix = prefix + '_'
    if postfix:
        postfix = '_' + postfix
    knobble = lambda x: 'Total' if x == 'total' else x  # noqa: E731
    return {f'{prefix}{i}{postfix}': knobble(i).title() for i in port.line_names_ex}


# ---------------------------------------------------------------------------
# Premium capital + accounting exhibits
# ---------------------------------------------------------------------------

def set_a_p(port, a, p):
    """Reconcile (assets, probability) inputs; supply defaults if neither given.

    Returns ``(a, p)`` snapped to the density grid. If neither is provided
    defaults to ``p = 0.995``.
    """
    if a == 0 and p == 0:
        p = 0.995
        a = port.q(p)
        p = port.cdf(a)
    elif a:
        p = port.cdf(a)
        a = port.q(p)
    elif p:
        a = port.q(p)
        p = port.cdf(a)
    return a, float(p)


def premium_capital(port, a=0, p=0):
    """Populate ``port.EX_premium_capital`` from ``augmented_df`` at level ``a``.

    Pricing story (run-off) report drawn from the augmented_df T.{L,P,M,Q,LR,ROE}
    columns. Mutates ``port`` by attaching ``_raw_premium_capital`` (raw) and
    ``EX_premium_capital`` (display-renamed).
    """
    a, p = set_a_p(port, a, p)
    dm = port.augmented_df.filter(
        regex=f'T.[MPQLROE]+.({port.line_name_pipe})').loc[[a]].T
    dm.index = dm.index.str.split('_', expand=True)
    port._raw_premium_capital = dm.unstack(1)
    port._raw_premium_capital = port._raw_premium_capital.droplevel(axis=1, level=0)
    port._raw_premium_capital.loc['T.A', :] = (
        port._raw_premium_capital.loc['T.Q', :]
        + port._raw_premium_capital.loc['T.P', :]
    )
    port._raw_premium_capital.index.name = 'Item'
    port.EX_premium_capital = (
        port._raw_premium_capital
        .rename(index=premium_capital_renamer, columns=port.line_renamer)
        .sort_index()
    )


def multi_premium_capital(port, As, keys=None):
    """Concatenate ``premium_capital`` exhibits at several asset levels."""
    if keys is None:
        keys = [f'a={i:.1f}' for i in As]
    ans = []
    for a in As:
        premium_capital(port, a)
        ans.append(port.EX_premium_capital.copy())
    port.EX_multi_premium_capital = pd.concat(
        ans, axis=1, keys=keys, names=['Assets', 'Line'])


def accounting_economic_balance_sheet(port, a=0, p=0):
    """Populate ``port.EX_accounting_economic_balance_sheet`` at level ``a``.

    Assumes line 0 = reserves and line 1 = prospective. Calls
    :func:`premium_capital` first to ensure ``_raw_premium_capital`` is fresh.
    """
    premium_capital(port, a, p)

    aebs = pd.DataFrame(
        0.0,
        index=port.line_names_ex + ['Assets', 'Equity'],
        columns=['Statutory', 'Objective', 'Market', 'Difference'],
    )
    slc = slice(0, len(port.line_names_ex))
    # Per-unit + portfolio-total theoretical agg means. Per-unit columns
    # of stats_df hold each Aggregate's mixed view; the portfolio total
    # is the ``total`` column.
    aebs.iloc[slc, 0] = [
        float(port.stats_df.loc[('agg', 'mean'), name])
        for name in port.line_names
    ] + [float(port.stats_df.loc[('agg', 'mean'), 'total'])]
    aebs.iloc[slc, 1] = port._raw_premium_capital.loc['T.L']
    aebs.iloc[slc, 2] = port._raw_premium_capital.loc['T.P']
    aebs.loc['Assets', :] = port._raw_premium_capital.loc['T.A', 'total']
    aebs.loc['Equity', :] = aebs.loc['Assets'] - aebs.loc['total']
    aebs['Difference'] = aebs.Market - aebs.Objective
    aebs = aebs.iloc[[-2] + list(range(len(port.line_names) + 1)) + [-1]]
    aebs.index.name = 'Item'
    port.EX_accounting_economic_balance_sheet = aebs.rename(index=port.line_renamer)


def make_all(port, p=0, a=0, As=None):
    """Build the standard PIR exhibit set with sensible defaults.

    If a distortion has been applied to ``port`` (``port.distortion is not
    None``), builds ``premium_capital``, optionally ``multi_premium_capital``
    over ``As``, and ``accounting_economic_balance_sheet``.
    """
    a, p = set_a_p(port, a, p)
    if port.distortion is not None:
        premium_capital(port, a=a, p=p)
        if As is not None:
            multi_premium_capital(port, As)
        accounting_economic_balance_sheet(port, a=a, p=p)


def show_enhanced_exhibits(port, fmt='{:.5g}'):
    """Display all ``EX_*`` exhibits on ``port`` as styled DataFrames (HTML)."""
    display(HTML(
        f'<h2>Exhibits for {port.name.replace("_", " ").title()} Portfolio</h2>'))
    for x in dir(port):
        if x[0:3] == 'EX_':
            ob = getattr(port, x)
            if isinstance(ob, pd.DataFrame):
                display(HTML(f'<h3>{x[3:].replace("_", " ").title()}</h3>'))
                display(ob.style.format(
                    fmt, subset=ob.select_dtypes(np.number).columns))
                display(HTML('<hr>'))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def profit_segment_plot(port, ax, p, line_names, dist_name,
                        colors=None, translations=None):
    """Lee diagram for each requested line on a stand-alone basis.

    Risk-adjusted premium uses the named distortion in ``port.distortions``.
    Optionally supply per-line ``colors`` (integers → ``C{n}``) and
    ``translations`` (additive y-axis shifts, useful for layering a CAT line
    on top of NC).

    Example::

        port.gross.profit_segment_plot(ax, 0.99999,
            ['total', 'CAT', 'NC'], 'wang', [2,0,1])
    """
    dist = port.distortions[dist_name]
    if colors is None:
        colors = range(len(line_names))
    if translations is None:
        translations = [0] * len(line_names)
    for line, cn, translation in zip(line_names, colors, translations):
        c = f'C{cn}'
        f1 = port.density_df[f'p_{line}'].cumsum()
        idx = (f1 < p) * (f1 > 1.0 - p)
        f1 = f1[idx]
        gf = 1 - dist.g(1 - f1)
        x = port.density_df.loss[idx] + translation
        ax.plot(gf, x, '-', c=c,
                label=f'Risk Adj {line}' if translation == 0 else None)
        ax.plot(f1, x, '--', c=c,
                label=line if translation == 0 else None)
        if translation == 0:
            ax.fill_betweenx(x, gf, f1, color=c, alpha=0.5)
        else:
            ax.fill_betweenx(x, gf, f1, color=c, edgecolor='black', alpha=0.5)
    ax.set(ylim=[0, port.q(p)])
    ax.legend(loc='upper left')


def natural_profit_segment_plot(port, ax, p, line_names, colors, translations):
    """Natural-allocation profit segment plot.

    Plots between the (1-p) and p quantiles using ``augmented_df``. Requires
    a distortion has already been applied to ``port``.
    """
    lw, up = port.q(1 - p), port.q(p)
    bit = port.augmented_df.query(f' {lw} <= loss <= {up} ')
    F = bit['F']
    gF = bit['gF']
    for line, cn, translation in zip(line_names, colors, translations):
        c = f'C{cn}'
        ser = bit[f'exeqa_{line}']
        ax.plot(F, ser, ls='dashed', c=c)
        ax.plot(gF, ser, c=c)
        if translation == 0:
            ax.fill_betweenx(ser + translation, gF, F, color=c, alpha=0.5, label=line)
        else:
            ax.fill_betweenx(ser, gF, F, color=c, alpha=0.5, label=line)
    ax.set(ylim=[0, up], title=port.distortion)
    ax.legend(loc='upper left')


def density_sample(port, n=20, reg='loss|p_|exeqa_'):
    """Return ``n`` equally-likely-spaced rows from ``port.density_df``."""
    ps = np.linspace(0.001, 0.999, n)
    xs = [port.q(i) for i in ps]
    return port.density_df.filter(regex=reg).loc[xs, :].rename(columns=renamer(port))


def biv_contour_plot(port, fig, ax, min_loss, max_loss, jump,
                     log=True, cmap='Greys', min_density=1e-15, levels=30,
                     lines=None, linecolor='w', colorbar=False, normalize=False,
                     **kwargs):
    """Contour plot of the bivariate density of (line A, line B).

    Assumes ``port`` has exactly two lines. Samples ``density_df`` at
    ``np.arange(min_loss, max_loss, jump)`` for the outer-product evaluation —
    pick ``jump`` carefully so the grid stays manageable (``100 * bs`` is a
    reasonable default).
    """
    npts = np.arange(min_loss, max_loss, jump)
    ps = [f'p_{i}' for i in port.line_names]
    bit = port.density_df.loc[npts, ps]
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
    ax.set(xlabel=f'Line {port.line_names[0]}',
           ylabel=f'Line {port.line_names[1]}',
           xlim=[-max_loss / 50, max_loss],
           ylim=[-max_loss / 50, max_loss],
           title=title,
           aspect=1)
    port.X = X
    port.Y = Y
    port.Z = Z


def twelve_plot(port, fig, axs, p=0.999, p2=0.9999, xmax=0, ymax2=0,
                biv_log=True, legend_font=0, contour_scale=10,
                sort_order=None, kind='two', cmap='viridis'):
    """Twelve-panel diagnostic plot used in the ASTIN paper and PIR book.

    Must run a distortion first, e.g.
    ``port.apply_distortion(port.distortions['ph'], efficient=False)``.

    Panels (by row × column index in ``axs``):

    - (1,1) density, (1,2) log density, (1,3) bivariate density
    - (2,1) κ, (2,2) α, (2,3) β
    - (3,1)/(4,1) per-line S, gS, αS, βgS
    - (3,2) margin density M_i, (4,2) cumulative margin M̄_i
    - (3,3) stand-alone M, (4,3) natural M

    ``sort_order`` reorders the line indices for plotting; defaults to
    ``[1, 2, 0]``.
    """
    a11, a12, a13, a21, a22, a23, a31, a32, a33, a41, a42, a43 = axs.flat
    col_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lss = ['solid', 'dashed', 'dotted', 'dashdot']

    if sort_order is None:
        sort_order = [1, 2, 0]

    if xmax == 0:
        xmax = port.q(p)
    ymax = xmax

    # density and log density
    temp = (
        port.density_df.filter(regex='p_')
        .rename(columns=short_renamer(port, 'p'))
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

    # biv density / survival
    if kind == 'two':
        xmax = port.snap(xmax)
        min_loss, max_loss, jump = 0, xmax, port.snap(xmax / 255)
        min_density = 1e-15
        levels = 30
        color_bar = False

        ps = [f'p_{i}' for i in port.line_names]
        title = 'Bivariate density'
        query = ' or '.join([f'`p_{i}` > 0' for i in port.line_names])
        if port.density_df.query(query).shape[0] < 512:
            logger.info('Contour plot has few points...going discrete...')
            bit = port.density_df.query(query)
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
            bit = port.density_df.loc[npts, ps]
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

        a13.set(xlabel=f'Line {port.line_names[0]}',
                ylabel=f'Line {port.line_names[1]}',
                title=title,
                aspect=1)
    else:
        l1 = (1 - temp.cumsum()).plot(ax=a13, lw=1)
        l1.lines[-1].set(linewidth=1.5)
        a13.set(title='Survival Function')
        a13.legend()

    # kappa
    bit = port.density_df.loc[:xmax].filter(regex=f'^exeqa_({port.line_name_pipe})$')
    bit = bit.iloc[:, sort_order]
    bit.rename(columns=short_renamer(port, 'exeqa')).replace(0, np.nan). \
        sort_index(axis=1).iloc[:, sort_order].plot(ax=a21, lw=1)
    a21.set(title=r'$\kappa_i(x)=E[X_i\mid X=x]$')
    a21.set(xlim=[0, xmax], ylim=[0, xmax], aspect='equal')
    a21.legend(loc='upper left')

    # alpha and beta
    aug_df = port.augmented_df
    aug_df.filter(regex=f'exi_xgta_({port.line_name_pipe})'). \
        rename(columns=short_renamer(port, 'exi_xgta')). \
        sort_index(axis=1).plot(ylim=[-0.05, 1.05], ax=a22, lw=1)
    for ln, ls in zip(a22.lines, lss[1:]):
        ln.set_linestyle(ls)
    a22.legend()
    a22.set(xlim=[0, xmax], title=r'$\alpha_i(x)=E[X_i/X\mid X>x]$')

    bit = aug_df.query(f'loss < {xmax}').filter(regex=f'exi_xgtag?_({port.line_name_pipe})')
    bit.rename(columns=short_renamer(port, 'exi_xgtag')). \
        sort_index(axis=1).plot(ylim=[-0.05, 1.05], ax=a23)
    for i, l in enumerate(a23.lines[len(port.line_names):]):
        if l.get_label()[0:3] == 'exi':
            a23.lines[i].set(linewidth=2, ls=lss[1 + i])
            l.set(color=f'C{i}', linestyle=lss[1 + i], linewidth=1,
                  alpha=.5, label=None)
    a23.legend(loc='upper left')
    a23.set(xlim=[0, xmax], title=r'$\beta_i(x)=E_{Q}[X_i/X \mid X> x]$')

    aug_df.filter(regex='M.M').rename(columns=short_renamer(port, 'M.M')). \
        sort_index(axis=1).iloc[:, sort_order].plot(ax=a32, lw=1)
    a32.set(xlim=[0, xmax], title='Margin density $M_i(x)$')

    aug_df.filter(regex='T.M').rename(columns=short_renamer(port, 'T.M')). \
        sort_index(axis=1).iloc[:, sort_order].plot(ax=a42, lw=1)
    a42.set(xlim=[0, xmax], title=r'Margin $\bar M_i(x)$')

    adf = port.augmented_df.loc[:xmax]
    if kind == 'two':
        zipper = zip(range(2), sorted(port.line_names), [a31, a41])
    else:
        zipper = zip(range(3), sorted(port.line_names), [a31, a41, a33])
    for i, line, a in zipper:
        a.plot(adf.loss, adf.S, c=col_list[2], ls=lss[1], lw=1, alpha=0.5, label='$S$')
        a.plot(adf.loss, adf.gS, c=col_list[2], ls=lss[0], lw=1, alpha=0.5, label='$g(S)$')
        a.plot(adf.loss, adf.S * adf[f'exi_xgta_{line}'], c=col_list[i],
               ls=lss[1], lw=1, label=fr'$\alpha S$ {line}')
        a.plot(adf.loss, adf.gS * adf[f'exi_xgtag_{line}'], c=col_list[i],
               ls=lss[0], lw=1, label=fr'$\beta g(S)$ {line}')
        a.set(xlim=[0, ymax])
        a.set(title=f'Line = {line}')
        a.legend()
        a.set(xlim=[0, ymax])
        a.legend(loc='upper right')

    alpha = 0.05
    if kind == 'two':
        ymax = ymax2 if ymax2 > 0 else port.q(p2)
        p2 = port.cdf(ymax)
        for cn, ln in enumerate(sort_order):
            line = sorted(port.line_names_ex)[ln]
            c = col_list[cn]
            s = lss[cn]
            f1 = port.density_df[f'p_{line}'].cumsum()
            idx = (f1 < p2) * (f1 > 1.0 - p2)
            f1 = f1[idx]
            gf = 1 - port.distortion.g(1 - f1)
            x = port.density_df.loss[idx]
            a33.plot(gf, x, c=c, ls=s, lw=1, label=None)
            a33.plot(f1, x, ls=s, c=c, lw=1, label=None)
            a33.fill_betweenx(x, gf, f1, color=c, alpha=alpha, label=line.title())
        a33.set(ylim=[0, ymax], title='Stand-alone $M$')
        a33.legend(loc='upper left')

    lw, up = port.q(1 - p2), ymax
    bit = port.augmented_df.query(f' {lw} <= loss <= {up} ')
    F = bit['F']
    gF = bit['gF']
    for cn, ln in enumerate(sort_order):
        line = sorted(port.line_names_ex)[ln]
        c = col_list[cn]
        s = lss[cn]
        ser = bit[f'exeqa_{line}']
        if kind == 'three':
            a43.plot(1 / (1 - F), ser, lw=1, ls=lss[1], c=c)
            a43.plot(1 / (1 - gF), ser, lw=1, c=c)
            a43.set(xlim=[1, 1e4], xscale='log')
            a43.fill_betweenx(ser, 1 / (1 - gF), 1 / (1 - F),
                              color=c, alpha=alpha, lw=0.5, label=line.title())
        else:
            a43.plot(F, ser, lw=1, ls=s, c=c)
            a43.plot(gF, ser, lw=1, ls=s, c=c)
            a43.fill_betweenx(ser, gF, F,
                              color=c, alpha=alpha, lw=0.5, label=line.title())
    a43.set(ylim=[0, ymax], title='Natural $M$')
    a43.legend(loc='upper left')

    if legend_font:
        for ax in axs.flat:
            try:
                if ax is not a13:
                    ax.legend(prop={'size': 7})
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Layer effectiveness γ
# ---------------------------------------------------------------------------

def gamma(port, a=None, p=None, kind='lower', compute_stand_alone=False,
          axs=None, plot_mode='return'):
    r"""Conditional layer effectiveness γ_a(x) given assets ``a``.

    See *Pricing Insurance Risk* "Main Result for Conditional Layer
    Effectiveness; Piano Diagram". In total
    γ_a(x) = E[(a ∧ X) / X | X > x] is the average reimbursement rate above
    x given capital a. By line
    γ_{a,i}(x) = E[ E[X_i | X] / X · (X ∧ a) / X · 1_{X>x} ]
                 / E[ E[X_i | X] / X · 1_{X>x} ].

    Returns a :class:`GammaResult` carrying the augmented density frame. If
    ``axs`` is provided, also draws diagnostic plots.
    """
    if a is None:
        assert p is not None
        a = port.q(p, kind)
    else:
        p = port.cdf(a)
        ql = port.q(p, 'lower')
        qu = port.q(p, 'upper')
        logger.log(
            WL,
            f'Input a={a} to gamma; computed p={p:.8g}, '
            f'lower and upper quantiles are {ql:.8g} and {qu:.8g}',
        )

    temp = port.density_df.filter(
        regex='^(p_|e1xi_1gta_|exi_xgta_|exi_xeqa_|exeqa_)[a-zA-Z]|^S$|^loss$'
    ).copy()

    var_dict = port.var_dict(p, kind, 'total')
    extreme_var_dict = port.var_dict(1 - (1 - p) / 100, kind, 'total')  # noqa: F841

    min_xa = np.minimum(temp.loss, a) / temp.loss
    temp['min_xa_x'] = min_xa

    ln = 'total'
    gam_name = f'gamma_{port.name}_{ln}'
    temp[f'exi_x1gta_{ln}'] = (
        (temp['loss'] * temp.p_total / temp.loss)
        .shift(-1)[::-1].cumsum() * port.bs
    )
    s_ = temp.p_total.shift(-1)[::-1].cumsum() * port.bs
    t1, t2 = (
        np.allclose(s_[:-1:-1], temp[f'exi_x1gta_{ln}'].iloc[-1]),
        np.allclose(temp[f'exi_x1gta_{ln}'].iloc[:-1], temp.S.iloc[:-1] * port.bs),
    )
    logger.debug(f'TEMP: the following should be close: {t1}, {t2} (expect True/True)')

    temp[gam_name] = (
        (min_xa * 1.0 * temp.p_total).shift(-1)[::-1].cumsum()
        / temp[f'exi_x1gta_{ln}'] * port.bs
    )

    for ln in port.line_names:
        if axs is not None or compute_stand_alone:
            a_l = var_dict[ln]
            a_l_ = a_l - port.bs
            xinv = temp[f'e1xi_1gta_{ln}']
            gam_name = f'gamma_{ln}_sa'
            s = 1 - temp[f'p_{ln}'].cumsum()
            temp[f'S_{ln}'] = s
            temp[gam_name] = 0
            temp.loc[0:a_l_, gam_name] = (s[0:a_l_] - s[a_l] + a_l * xinv[a_l]) / s[0:a_l_]
            temp.loc[a_l:, gam_name] = a_l * xinv[a_l:] / s[a_l:]
            temp[gam_name] = temp[gam_name].shift(1)

        gam_name = f'gamma_{port.name}_{ln}'
        temp[f'exi_x1gta_{ln}'] = (
            (temp[f'exeqa_{ln}'] * temp.p_total / temp.loss)
            .shift(-1)[::-1].cumsum() * port.bs
        )
        temp[gam_name] = (
            (min_xa * temp[f'exi_xeqa_{ln}'] * temp.p_total)
            .shift(-1)[::-1].cumsum()
            / temp[f'exi_x1gta_{ln}'] * port.bs
        )

    if axs is not None:
        axi = iter(axs.flat)
        nr, nc = axs.shape
        v = port.var_dict(.996, 'lower', 'total')
        ef = EngFormatter(3, True)

        if plot_mode == 'return':
            def transformer(x):
                return np.where(x == 0, np.nan, 1.0 / (1 - x))
            yscale = 'log'
        else:
            def transformer(x):
                return x
            yscale = 'linear'

        s1 = 1 / temp.S
        for i, ln in enumerate(port.line_names_ex):
            r, c = i // nc, i % nc
            ax = next(axi)
            if ln != 'total':
                ls1 = 1 / temp[f'S_{ln}']
                ax.plot(ls1, transformer(temp[f'gamma_{ln}_sa']),
                        c='C1', label=f'SA {ln}')
                ax.plot(s1, transformer(temp[f'gamma_{port.name}_total']),
                        lw=1, c='C7', label='total')
                color = 'C1'
            else:
                ls1 = s1
                color = 'C0'
            ax.plot(s1, transformer(temp[f'gamma_{port.name}_{ln}']),
                    c='C0', label=f'Pooled {ln}')
            temp['WORST'] = np.maximum(0, 1 - (1 - p) * ls1)
            temp['BEST'] = 1
            temp.loc[v[ln]:, 'BEST'] = v[ln] / temp.loc[v[ln]:, 'loss']
            ax.fill_between(ls1, transformer(temp.WORST), transformer(temp.BEST),
                            lw=.5, ls='--', alpha=.1, color=color, label='Possible range')
            ax.plot(ls1, transformer(temp.BEST), lw=.5, ls='--', alpha=1, c=color)
            ax.plot(ls1, transformer(temp.WORST), lw=.5, ls=':', alpha=1, c=color)
            ax.set(xlim=[1, 1e9], xscale='log', yscale=yscale,
                   xlabel='Loss return period' if r == nr - 1 else None,
                   ylabel='Coverage Effectiveness (RP)' if c == 0 else None,
                   title=f'{ln}, a={ef(v[ln]).strip()}')
            ax.axvline(1 / (1 - .996), ls='--', c='C7', lw=.5, label='Capital p')
            ll = ticker.LogLocator(10, numticks=10)
            ax.xaxis.set_major_locator(ll)
            lm = ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
            ax.xaxis.set_minor_locator(lm)
            ax.grid(lw=.25)
            ax.legend(loc='upper right')
        try:
            while 1:
                ax.figure.delaxes(next(axi))
        except StopIteration:
            pass
    temp.drop(columns=['BEST', 'WORST'])
    return GammaResult(gamma_df=temp.sort_index(axis=1), base=port.name,
                       assets=a, p=p, kind=kind)


# ---------------------------------------------------------------------------
# Stand-alone pricing
# ---------------------------------------------------------------------------

def stand_alone_pricing_work(port, dist, p, kind, roe, S_calc='cumsum'):
    """Per-line stand-alone pricing for one ``dist`` (or traditional method).

    ``dist`` can be a :class:`Distortion` (spectral) or the literal strings
    ``'traditional'`` / ``'traditional - no default'`` (in which case
    ``roe`` is used to build a constant-CoC pricing scheme).

    Returns a DataFrame with rows L / LR / M / P / PQ / Q / ROE (plus an
    aggregated ``sop`` sum-of-parts column).
    """
    assert S_calc in ('S', 'cumsum')

    var_dict = port.var_dict(p, kind=kind, total='total', snap=True)
    exhibit = pd.DataFrame(0.0,
                           index=['L', 'LR', 'M', 'P', 'PQ', 'Q', 'ROE'],
                           columns=['sop'])

    def tidy_and_write(exhibit, ax, exa, prem):
        roe_ = (prem - exa) / (ax - prem)
        exhibit.loc[['L', 'LR', 'M', 'P', 'PQ', 'Q', 'ROE'], l] = (
            exa, exa / prem, prem - exa, prem,
            prem / (ax - prem), ax - prem, roe_,
        )

    if dist == 'traditional - no default':
        method = dist
        d = roe / (1 + roe)
        v = 1 - d
        for l in port.line_names_ex:
            ax = var_dict[l]
            # Per-line empirical mean from the FFT marginal. For ``total``
            # the canonical value lives in ``stats_df['empirical']``; for
            # per-unit ``l`` compute inline from ``density_df``.
            if l == 'total':
                exa = float(port.stats_df.loc[('agg', 'mean'), 'empirical'])
            else:
                exa = float(np.sum(port.density_df['loss'] *
                                   port.density_df[f'p_{l}']))
            prem = v * exa + d * ax
            tidy_and_write(exhibit, ax, exa, prem)
    elif dist == 'traditional':
        method = dist
        d = roe / (1 + roe)
        v = 1 - d
        for l in port.line_names_ex:
            ax = var_dict[l]
            exa = port.density_df.loc[ax, f'lev_{l}']
            prem = v * exa + d * ax
            tidy_and_write(exhibit, ax, exa, prem)
    else:
        method = f'sa {str(dist)}'
        for l, ag in zip(port.line_names_ex, port.agg_list + [None]):
            if ag is None:
                if S_calc == 'S':
                    S = port.density_df.S
                else:
                    S = (1 - port.density_df['p_total'].cumsum())
                gS = pd.Series(dist.g(S), index=S.index)
                exag = gS.shift(1, fill_value=0).cumsum() * port.bs
            else:
                ag.apply_distortion(dist)
                exag = ag.density_df.exag
            ax = var_dict[l]
            exa = port.density_df.loc[ax, f'lev_{l}']
            prem = exag.loc[ax]
            tidy_and_write(exhibit, ax, exa, prem)

    exhibit.loc['a'] = exhibit.loc['P'] + exhibit.loc['Q']
    exhibit['sop'] = exhibit.filter(regex='[A-Z]').sum(axis=1)
    exhibit.loc['LR', 'sop'] = exhibit.loc['L', 'sop'] / exhibit.loc['P', 'sop']
    exhibit.loc['ROE', 'sop'] = (exhibit.loc['M', 'sop']
                                 / (exhibit.loc['a', 'sop'] - exhibit.loc['P', 'sop']))
    exhibit.loc['PQ', 'sop'] = exhibit.loc['P', 'sop'] / exhibit.loc['Q', 'sop']

    exhibit['method'] = method
    exhibit = exhibit.reset_index()
    exhibit = exhibit.set_index(['method', 'index'])
    exhibit.index.names = ['method', 'stat']
    exhibit.columns.name = 'line'
    exhibit = exhibit.sort_index(axis=1)
    return exhibit


def stand_alone_pricing(port, dist, p=0, kind='var', S_calc='cumsum'):
    """Run distortion pricing then traditional + default for comparison."""
    assert isinstance(dist, (Distortion, list))
    if type(dist) != list:
        dist = [dist]
    ex1s = []
    for d in dist:
        ex1s.append(stand_alone_pricing_work(port, d, p=p, kind=kind, roe=0, S_calc=S_calc))
        if len(ex1s) == 1:
            roe = ex1s[0].at[(f'sa {str(d)}', 'ROE'), 'total']
    ex2 = stand_alone_pricing_work(port, 'traditional - no default',
                                   p=p, kind=kind, roe=roe, S_calc=S_calc)
    ex3 = stand_alone_pricing_work(port, 'traditional',
                                   p=p, kind=kind, roe=roe, S_calc=S_calc)
    return pd.concat(ex1s + [ex2, ex3])


# ---------------------------------------------------------------------------
# Convex-hull helpers + calibrate_blends
# ---------------------------------------------------------------------------

def check01(s):
    """Add the endpoints 0 and 1 to ``s`` if not already present."""
    if 0 not in s:
        s = np.hstack((0, s))
    if 1 not in s:
        s = np.hstack((s, 1))
    return s


def make_array(s, gs):
    """Stack two sequences, padded to include 0 and 1, into an Nx2 array."""
    s = np.array(s)
    gs = np.array(gs)
    s = check01(s)
    gs = check01(gs)
    return np.array((s, gs)).T


def convex_points(s, gs):
    """Return the points on the convex envelope of ``(s, gs)`` plus 0/1."""
    points = make_array(s, gs)
    hull = ConvexHull(points)
    hv = hull.vertices[::-1]
    hv = np.roll(hv, -np.argmin(hv))
    return points[hv, :].T


def calibrate_blends(port, a, premium, s_values, gs_values=None,
                     spread_values=None, debug=False):
    """Calibrate blend distortions to a given premium at asset level ``a``.

    Either pass ``gs_values`` directly or pass ``spread_values`` (market
    yield spreads, converted internally via ``g(s) = 1 - 1/(1+spread)``).

    Builds candidate blends using the four combinations of including or
    excluding the endpoints of the (s, g(s)) curve, then takes pairs that
    bracket ``premium`` and returns weighted-average blends.

    Returns a dict of ``Distortion`` objects (or ``(dict, df, pricer, dists,
    wts)`` if ``debug`` is True).
    """
    if gs_values is None:
        gs_values = 1 - 1 / (1 + np.array(spread_values))

    s_values, gs_values = convex_points(s_values, gs_values)
    if len(s_values) < 4:
        raise ValueError('Input s,gs points do not generate enough separate points.')

    df = port.density_df
    bs = port.bs
    S = (1 - df.p_total[0:a - bs].cumsum())

    def pricer(g):
        nonlocal S, bs
        return np.sum(g(S)) * bs

    def make_g(s, gs):
        i = interp1d(s, gs, bounds_error=False, fill_value='extrapolate')

        def f(x):
            return np.minimum(1, i(x))
        return f

    ans = []
    dists = {}
    for left in ['e', '0']:
        for right in ['e', '1']:
            s = s_values.copy()
            gs = gs_values.copy()
            if left == 'e':
                s = s[1:]
                gs = gs[1:]
            if right == 'e':
                s = s[:-1]
                gs = gs[:-1]
            dists[(left, right)] = make_g(s, gs)
            new_p = pricer(dists[(left, right)])
            ans.append((left, right, new_p, new_p > premium))

    df = pd.DataFrame(ans, columns=['left', 'right', 'premium', 'gt'])
    wts = {}
    wdists = {}
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        pi = df.iat[i, 2]
        pj = df.iat[j, 2]
        if min(pi, pj) <= premium <= max(pi, pj):
            il = df.iat[i, 0]
            ir = df.iat[i, 1]
            jl = df.iat[j, 0]
            jr = df.iat[j, 1]
            w = (premium - pj) / (pi - pj)
            wts[(il, ir, jl, jr)] = w
            temp = Distortion('ph', .599099)
            temp.name = 'blend'
            temp.g = lambda x, w=w, il=il, ir=ir, jl=jl, jr=jr: (
                w * dists[(il, ir)](x) + (1 - w) * dists[(jl, jr)](x))
            temp.g_inv = None
            wdists[(il, ir, jl, jr)] = temp
    if len(wdists) == 0:
        logger.warning('Failed to fit blend')
        wdists[0] = Distortion('ph', .599099)
        wdists[0].name = 'blend'
    if debug is True:
        return wdists, df, pricer, dists, wts
    return wdists


# ---------------------------------------------------------------------------
# Bulk constructors
# ---------------------------------------------------------------------------

def from_DataFrame(name, df):
    """Build a Portfolio from a DataFrame with one row per aggregate spec.

    Each unique ``name`` value becomes one aggregate; the remaining columns
    are passed as kwargs to ``Aggregate``.
    """
    from ..portfolio import Portfolio
    spec_list = [g.dropna(axis=1).to_dict(orient='records') for n, g in df.groupby('name')]
    spec_list = [i[0] for i in spec_list]
    return Portfolio(name, spec_list)


def from_Excel(name, ffn, sheet_name, **kwargs):
    """Build a Portfolio from an Excel sheet via :func:`from_DataFrame`."""
    df = pd.read_excel(ffn, sheet_name=sheet_name, **kwargs)
    df = df.dropna(axis=1, how='all')
    return from_DataFrame(name, df)


def from_dict_of_aggs(prefix, agg_dict, sub_ports=None, uw=None,
                      bs=0, log2=0, padding=2, **kwargs):
    """Build one or more Portfolios from a dict of DecL aggregate snippets.

    For each ``sub_port`` (a tuple of keys in ``agg_dict``) constructs the
    DecL ``port`` program by concatenating the named snippets, and either
    parses it via ``uw.build_many`` (if ``uw`` is given) or returns the
    program text. Sub-portfolios are named ``f'{prefix}_{concat keys}'``.
    """
    agg_names = list(agg_dict.keys())
    ports: dict = {}
    if sub_ports == 'all':
        sub_ports = subsets(agg_names)

    for sub_port in sub_ports:
        name = ''.join(sub_port)
        if prefix != '':
            name = f'{prefix}_{name}'
        pgm = f'port {name}\n'
        for l in agg_names:
            if l in sub_port:
                pgm += f'\t{agg_dict[l]}\n'
        if uw:
            ports[name] = uw.build_many(pgm, update=False)
        else:
            ports[name] = pgm
        if uw and bs * log2 > 0:
            ports[name].update(log2=log2, bs=bs, padding=padding, **kwargs)
    return ports
