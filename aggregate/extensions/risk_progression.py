# risk progression functions: decomposition of the natural allocation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from aggregate import make_mosaic_figure


def ff(n):
    s = f'{{x:.{n}f}}'
    pd.set_option('float_format', lambda x: s.format(x=x))


def make_distribution(df, prob_column='p_total'):
    """
    Make df, with columns [something, prob_column], into a distribution: groupby
    something, sum probs. Return the distribution and the sum of probs.
    """
    c = list(df.columns)
    c.remove(prob_column)
    out = df.set_index(c[0]).groupby(c[0]).sum()[prob_column]
    sp = out.sum()
    # normalize?
    out = out / sp
    return out, sp


def make_projection_distributions(self):
    """
    self = Portfolio
    make distributions from exeqa projections.

    """
    projections = {}
    sum_probs = {}
    for unit in self.unit_names:
        proj, sum_probs[unit] = \
            make_distribution(self.density_df[[f'exeqa_{unit}', 'p_total']])
        proj = proj.to_frame()
        proj['F'] = proj.p_total.cumsum()
        proj['S'] = 1 - proj.F
        projections[unit] = proj

    return projections, sum_probs


def plot_comparison(self, projections, axs, smooth):
    """
    Plot to compare unit, projection of unit, and total in self.
    self is a Portfolio

    :param self: Portfolio
    :param projections: dict of projections
    :param axs: list of axes
    :param smooth: smoothing factor for densities
    """
    lw = 2
    for unit, proj, axd, ax, axr, lc in \
        zip(self.unit_names, projections.values(), axs.flat[::3],
            axs.flat[1::3], axs.flat[2::3], ['C1', 'C2']):

        # figure the mean
        mn = np.sum(proj.p_total * proj.index)
        # print(
        #     f'{unit}, agg est mean {self[unit].est_m}, sum prod mean {mn}')

        # normalized densities: exeqa
        p = proj.p_total.copy()
        # rebucket to smooth out
        quant = self.bs * smooth
        p.index = quant * np.round(p.index / quant)
        p = p.groupby(level=0).sum()
        p.iloc[:-1] = p.iloc[:-1] / np.diff(p.index) * mn
        p.index = p.index / mn
        p.plot(ax=axd, c=lc, ls='--', label='Projection of ' + unit)

        # original density
        p = self[unit].density_df.p_total.copy()
        p.index = p.index / self[unit].est_m
        p = p * self[unit].est_m / self.bs
        p.plot(ax=axd, c=lc, label=unit)

        # total density
        p = self.density_df.p_total.copy()
        p.index = p.index / self.est_m
        p = p * self.est_m / self.bs
        p.plot(ax=axd, c='C0', label='total')
        # remember: normalized so 0 to 5 is universal
        axd.set(xlim=[0, 5],
                xlabel='loss', ylabel='density')
        yl = axd.get_ylim()
        if yl[1] > 100:
            axd.set(yscale='log', ylabel='log density')
        axd.legend(loc='upper right').set(title='NORMALIZED losses')

        # plot normalized distributions on linear and return period scale
        ax.plot(self.density_df[f'p_{unit}'].cumsum(),
                self.density_df.loss / self[unit].est_m, c=lc, lw=lw/2, label=unit)
        ax.plot(proj.F, np.array(proj.index) / mn,
                c=lc, ls='--', lw=lw, label='Projection')
        ax.plot(self.density_df['F'], self.density_df.loss /
                self.est_m, c='C0', lw=lw, label='total')
        ax.set(ylim=[0, 5], xlabel='probability', ylabel='normalized loss')
        ax.axhline(1, lw=.5, ls='--', c='C7')
        ax.legend(loc='upper left')

        axr.plot(1 / (1 - self.density_df[f'p_{unit}'].cumsum()),
                 self.density_df.loss / self[unit].est_m, c=lc, lw=lw/2, label=unit)
        proj = proj.query('F > 1e-11 and S > 1e-11')
        axr.plot(1 / proj.S, np.array(proj.index) / mn,
                 c=lc, ls='--', lw=lw, label='Projection')
        axr.plot(1 / self.density_df['S'], self.density_df.loss /
                 self.est_m, c='C0', lw=lw, label='total')
        axr.set(xlim=[1, 1e4], ylim=1e-1, xscale='log', yscale='log',
                xlabel='log return period', ylabel='log normalized loss')
        axr.axhline(1, lw=.5, ls='--', c='C7')
        axr.legend(loc='lower right')


def make_up_down(ser):
    """
    Split ser, regarded as a function on its index, into its increasing and decreasing
    parts. Also return the reconstruction.

    [was difference_increasing]

    Example::

        df = pd.DataFrame({'x': np.linspace(0, 20, 2001)})
        df['y'] = 2 * np.sin(df.x) + 1.8 * np.cos(df.x * 3 + 2)
        df = df.set_index('x')
        u,d,c = make_up_down(df.y)
        ax = u.plot()
        d.plot(ax=ax)
        c.plot(ax=ax)
        df.y.plot(ax=ax, lw=.5, c='w')
        ax.legend()

    :param ser: pandas Series
    """
    dy = ser.diff()
    dy.iloc[0] = ser.iloc[0]
    u = np.maximum(0, dy).cumsum()
    d = -np.minimum(0, dy).cumsum()
    c = u - d
    u.name = 'up'
    d.name = 'down'
    c.name = 'recreated'
    return u, d, c


def up_down_distributions(self):
    """
    Write the projections as differences of increasing functions and
    figure the distributions of those two functions.
    self is a Portfolio
    """
    up_functions = {}
    down_functions = {}
    up_distributions = {}
    down_distributions = {}
    for unit in self.unit_names:
        u, d, c = make_up_down(self.density_df[f'exeqa_{unit}'])
        up_functions[unit] = u
        down_functions[unit] = d

        u = u.to_frame()
        u['p_total'] = self.density_df.p_total
        du, _ = make_distribution(u)
        up_distributions[unit] = du

        d = d.to_frame()
        d['p_total'] = self.density_df.p_total
        dd, _ = make_distribution(d)
        down_distributions[unit] = dd

    UDD = namedtuple('UDD', ['up_functions', 'down_functions',
                             'up_distributions', 'down_distributions'])
    ans = UDD(up_functions, down_functions,
              up_distributions, down_distributions)
    return ans


def plot_up_down(self, udd, axs):
    """
    Plot the decomposition of the projections into up and down parts
    self - portfolio
    udd = UDD named tuple (above)
    """

    for unit, ax in zip(self.unit_names, axs.flat):
        ax = self.density_df[f'exeqa_{unit}'].plot(ax=ax, lw=4, c='C7')
        udd.up_functions[unit].plot(ax=ax)
        udd.down_functions[unit].plot(ax=ax)
        (udd.up_functions[unit] - udd.down_functions[unit]
         ).plot(ax=ax, lw=1.5, ls=':', c='C2', label='recreated')
        ax.legend()
        ax.set(xlabel='loss', ylabel='up or down function')

    # plot ud distributions
    ax = axs.flat[-1]
    for (k, v), c in zip(udd.up_distributions.items(), ['C0', 'C1']):
        v.cumsum().plot(c=c, ax=ax, label=f'Up {k}')
    for (k, v), c in zip(udd.down_distributions.items(), ['C0', 'C1']):
        v.cumsum().plot(c=c, ls=':', ax=ax, label=f'Down {k}')
    ax.legend(loc='lower right')
    ax.set(xlabel='loss', ylabel='cumulative probability')


def price_work(dn, series, names_ex):
    """
    price using dn across a number of series (already distributions) with given
    assets (that tie back to probs... caller OR pass in Port? )

    NO ASSET VERSION
    uses dn.price not price2

    """

    bit0 = pd.concat([
        pd.DataFrame(dn.price(ser, np.inf, kind='both'),
                     index=['bid', 'el', 'ask'], columns=[1]).T
        for ser in series],
        keys=names_ex, names=['unit', 'view'])
    bit = bit0.stack().unstack('view')
    # add sum of parts
    bit.loc[('sum', 'bid'), :] = 0.0
    bit.loc[('sum', 'el'), :] = 0.0
    bit.loc[('sum', 'ask'), :] = 0.0
    # TODO Kludge
    bit.loc['sum'] = bit.loc[names_ex[0]].values + bit.loc[names_ex[1]].values

    return bit


def price_compare(self, dn, projection_dists, ud_dists):
    """
    Build out pricing comparison waterfall

    :param self: Portfolio
    :param dn: Distortion
    """

    # linear natural allocation pricing
    # KLUDGE
    lna_pricea = self.price(self.q(1), dn, view='ask')
    # display(lna_pricea.df)
    lna_priceb = self.price(self.q(1), dn, view='bid')
    na_price = lna_pricea.df[['L', 'P']]
    na_price.columns = ['el', 'ask']
    # if this is nan it gets dropped from stack?!
    na_price['bid'] = lna_priceb.df['P']
    na_price = na_price.stack(dropna=False).to_frame()
    # beware : order matters here
    na_price.loc[('sum', 'el'), :] = 0.0
    na_price.loc[('sum', 'ask'), :] = 0.0
    na_price.loc[('sum', 'bid'), :] = 0.0
    na_price.loc['sum'] = na_price.loc[self.unit_names[0]
                                       ].values + na_price.loc[self.unit_names[1]].values
    # call pricing function FOUR times; original dist; proj; u and d of proj

    # original dists = stand alone pricing
    series = [self.density_df[f'p_{unit}'] for unit in self.unit_names_ex]
    sa = price_work(dn, series, self.unit_names_ex)

    # projected sa
    series = [v['p_total']
              for v in projection_dists.values()] + [self.density_df.p_total]
    proj_sa = price_work(dn, series, self.unit_names_ex)

    # up and down sa
    series = [v for v in ud_dists.up_distributions.values()] + \
        [self.density_df.p_total]
    udu = price_work(dn, series, self.unit_names_ex)
    udu.loc['total'] = np.nan
    # up and down sa
    series = [v for v in ud_dists.down_distributions.values()] + \
        [self.density_df.p_total]
    udd = price_work(dn, series, self.unit_names_ex)
    udd.loc['total'] = np.nan

    compare = pd.concat([
        na_price, sa, proj_sa, udu, udd
    ],
        axis=1,
        keys=['lna', 'sa', 'proj_sa', 'up', 'down']).droplevel(1, 1)

    # put in ask, el, bid order
    compare = compare.iloc[np.hstack(
        [np.array([0, 2, 1]) + 3*i for i in range(4)])]
    return compare


def full_monty(self, dn, truncate=True, smooth=16):
    """
    One stop shop for a Portfolio self
    Unlimited assets
    Prints all on one giant figure
    """

    # figure for all plots
    fig, axs = plt.subplots(4, 3, figsize=(
        3 * 3.5, 4 * 2.45), constrained_layout=True)

    # in the known bounded case we can truncate
    regex = ''.join([i[0] for i in self.line_names_ex])
    if truncate:
        self.density_df = self.density_df.loc[:self.density_df.F.idxmax()]
        self._linear_quantile_function = None

    # density and exa plots
    axd = {'A': axs[0, 0], 'B': axs[0, 1], 'C': axs[0, 2]}
    self.plot(axd=axd)
    self.density_df.filter(regex=f'exeqa_[{regex}]').plot(ax=axd['C'])
    axd['C'].set(xlabel='loss', ylabel='Conditional expectation')

    # projection distributions
    projection_dists, sum_probs = make_projection_distributions(self)
    if not np.allclose(list(sum_probs.values()), 1):
        print(sum_probs)

    # impact of projections on distributions
    axs1 = axs[1:3, :]
    plot_comparison(self, projection_dists, axs1, smooth)

    # up and down decomp
    ud_dists = up_down_distributions(self)

    # plot UD
    axs1 = axs[3, :]
    plot_up_down(self, ud_dists, axs1)

    compare = price_compare(self, dn, projection_dists, ud_dists)
    compare['umd'] = compare['up'] - compare['down']

    RiskProgression = namedtuple('RiskProgression', ['compare_df', 'projection_dists', 'ud_dists'])
    ans = RiskProgression(compare, projection_dists, ud_dists)
    return ans
