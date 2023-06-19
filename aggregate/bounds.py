from itertools import cycle
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import numpy as np
from numpy.linalg import pinv
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

# from aggregate.utilities import FigureManager
from . import Portfolio, Aggregate, Distortion, Underwriter, FigureManager
from . constants import *

logger = logging.getLogger(__name__)


class Bounds(object):
    """
    Implement IME 2022 pricing bounds methodology.

    Typical usage: First, create a Portfolio or Aggregate object a. Then ::

        bd = cd.Bounds(a)
        bd.tvar_cloud('line', premium=, a=, n_tps=, s=, kind=)
        p_star = bd.p_star('line', premium)
        bd.cloud_view(axes, ...)

    :param distribution_spec: A Portfolio or Portfolio.density_df dataframe or pd.Series (must have loss as index)
            If DataFrame or Series values interpreted as desnsity, sum to 1. F, S, exgta all computed using Portfolio
            methdology
            If DataFrame line --> p_{line}
    """
    # from common_scripts.cs

    def __init__(self, distribution_spec):
        assert isinstance(distribution_spec, (pd.Series,
                          pd.DataFrame, Portfolio, Aggregate))
        self.distribution_spec = distribution_spec
        # although passed as input to certain functions (tvar with bounds) b is actually fixed
        self.b = 0
        self.Fb = 0
        # change in 0.14.0 with new tvar methodology, got rid of confusingly named tvar_function.
        # TVaR for X
        self.tvar_unlimited_function = None
        # Series of tvar values
        self.tvars = None
        self.tps = None
        self.weight_df = None
        self.idx = None
        self.hinges = None
        self.cloud_df = None
        # uniform mode
        self._t_mode = 'u'
        # data frame with tvar weights and principal extreme distortion weights by method
        self.pedw_df = None
        self._tvar_df = None
        # hack for beta distribution, you want to force 1 to be in tvar ps, but Fp = 1
        # TODO figure out why p_star grinds to a halt if you input b < inf
        self.add_one = True
        logger.warning('Deprecatation warning. The kind argument is now ignored. Functionality '
                       'is equivalent to kind="tail", which was the most accurate method.')

    def __repr__(self):
        """
        Gets called automatically but so we can tweak.
        :return:
        """
        return 'Bounds Object at ' + super(Bounds, self).__repr__()

    @property
    def tvar_df(self):
        if self._tvar_df is None:
            self._tvar_df = pd.DataFrame(
                {'p': self.tps, 'tvar': self.tvars}).set_index('p')
        return self._tvar_df

    @property
    def t_mode(self):
        return self._t_mode

    @t_mode.setter
    def t_mode(self, val):
        assert val in ['u', 'gl']
        self._t_mode = val

    def make_tvar_function(self, line, b=np.inf):
        """
        Change in 0.14.0 with new tvar methodology, this function reflects the b limit, it is the
        TVaR of min(X, b)

        Make unlimited TVaR function for line, ``self.tvar_unlimited_function``, and set self.Fb.

        - Portfolio or Aggregate: get from object
        - DataFrame: make from p_{line} column
        - Series: make from Series

        In the last two cases, uses aggregate.utilties.make_var_tvar_function.

        Includes determining sup and putting in value for zero.
        If sup is largest value in index, sup set to inf.

        You generally want to apply with a limit, call ``self.tvar_with_bounds``.

        :param line: only used for portfolio objects, to specify line (or 'total')
        :param b:  bound on the losses, e.g., to model limited liability insurer
        :return:
        """

        self.b = b
        p_total = None

        # Note Fb calc varies by type
        if isinstance(self.distribution_spec, Portfolio):
            assert line in self.distribution_spec.line_names_ex
            if line == 'total':
                self.tvar_unlimited_function = self.distribution_spec.tvar
                self.Fb = self.distribution_spec.cdf(b)
            else:
                ag = getattr(self.distribution_spec, line)
                self.tvar_unlimited_function = ag.tvar
                self.Fb = ag.cdf(b)
            if np.isinf(b):
                self.Fb = 1.0

        elif isinstance(self.distribution_spec, Aggregate):
            self.tvar_unlimited_function = self.distribution_spec.tvar
            self.Fb = self.distribution_spec.cdf(b)
            if np.isinf(b):
                self.Fb = 1.0

        else:
            # next two instances fall through
            if isinstance(self.distribution_spec, pd.DataFrame):
                assert f'p_{line}' in self.distribution_spec.columns
                # given a port.density_df
                p_total = self.distribution_spec[f'p_{line}']

            elif isinstance(self.distribution_spec, pd.Series):
                logger.info('tvar_array using Series')
                p_total = self.distribution_spec

            # if here, then p_total is a series, index = loss, values = p_total pmf
            from .utilities import make_var_tvar
            # ensure p_total suitable for make_var_tvar
            assert p_total.index.is_unique, 'Index must be unique'
            assert p_total.index.is_monotonic_increasing, 'Index must be monotone increasing'
            # subset to p_total > 0
            p_total = p_total[p_total > 0]
            # now can call make_var_tvar and extract the tvar function
            self.tvar_unlimited_function = make_var_tvar(p_total).tvar

            # set F(b)
            if np.isinf(b):
                self.Fb = 0
            else:
                F = p_total.cumsum()
                self.Fb = F[b]

    def make_ps(self, n, mode):
        """
        If add_one then you want n = 2**m + 1 to ensure nicely spaced points.

        Mode: making s points (always uniform) or tvar p points (use t_mode).
        self.t_mode == 'u': make uniform s points against which to evaluate g from 0 to 1
        self.t_mode == 'gl': make Gauss-Legndre p points at which TVaRs are evaluated from 0 inclusive to 1 exclusive with more around 1

        :param n:
        :return:
        """
        assert mode in ('s', 't')

        if mode == 't' and (self.Fb < 1 or self.add_one):
            # we will add 1 at the end
            n -= 1

        # Gauus Legendre points
        lg = np.polynomial.legendre.leggauss

        if self.t_mode == 'gl':
            if mode == 's':
                x, wts = lg(n - 2)
                ps = np.hstack((0, (x + 1) / 2, 1))
            elif mode == 't':
                x, wts = lg(n * 2 + 1)
                ps = x[n:]

        elif self.t_mode == 'u':
            if mode == 's':
                ps = np.linspace(1 / n, 1, n)
            elif mode == 't':
                # exclude 1 (sup distortion) at the end; 0=mean
                ps = np.linspace(0, 1, n, endpoint=False)

        # always ensure that 1  is in ps for t mode when b < inf if Fb < 1
        if mode == 't' and (self.Fb < 1 or self.add_one):
            ps = np.hstack((ps, 1))

        return ps

    def tvar_array(self, line, n_tps=257, b=np.inf, kind='interp'):
        """
        Compute tvars at n equally spaced points, tps.


        :param line:
        :param n_tps:  number of tvar p points, default 257 (assuming add-one mode)
        :param b: cap on losses applied before computing TVaRs (e.g., adjust losses for finite assets b).
               Use np.inf for unlimited losses.
        :param kind: now ignored.
        :return:
        """
        self.make_tvar_function(line, b)
        logger.info(f'F(b) = {self.Fb:.5f}')
        self.tps = self.make_ps(n_tps, 't')
        self.tvars = self.tvar_with_bound(self.tps, b)

    def p_star(self, line, premium, b=np.inf, kind='interp'):
        """
        Compute p* so TVaR @ p* of min(X, b) = premium

        In this case the cap b has an impact (think of integrating q(p) over p to 1, q is impacted by b)

        premium <= b is required (no rip off condition)

        If b < inf then must solve TVaR(p) - (1 - F(b)) / (1 - p)[TVaR(F(b)) - b] = premium
        Let k = (1 - F(b)) [TVaR(F(b)) - b], so solving

        f(p) = TVaR(p) - k / (1 - p) - premium == 0

        using NR

        :param line:
        :param premium: target premium
        :param b:  bound
        :param kind: now ignored
        :return:
        """
        if premium > b:
            raise ValueError(
                f'p_star must have premium ({premium}) <= largest loss bound ({b})')

        self.make_tvar_function(line, b)

        if premium < self.tvar_unlimited_function(0):
            raise ValueError(
                f'p_star must have premium ({premium}) >= mean ({self.tvar_unlimited_function(0)})')

        def f(p):
            return self.tvar_with_bound(p, b, 'tail') - premium

        fp = 100
        p = 0.5
        iters = 0
        delta = 1e-10
        while abs(fp) > 1e-6 and iters < 20:
            fp = f(p)
            fpp = (f(p + delta) - f(p)) / delta
            pnew = p - fp / fpp
            if 0 <= pnew <= 1:
                p = pnew
            elif pnew < 0:
                p = p / 2
            else:
                #  pnew > 1:
                p = (1 + p) / 2

        if iters == 20:
            logger.warning(
                f'Questionable convergence solving for p_star, last error {fp}.')
        p_star = p
        return p_star

    def tvar_with_bound(self, p, b=np.inf, kind='interp'):
        """
        Compute tvar taking bound into account.
        Assumes tvar_unfunction setup.

        Warning: b must equal the b used when calibrated. The issue is computing F
        which varies with the type of underlying portfolio. This is fragile.
        Added storing b and checking equal. For backwards comp. need to keep b argument

        :param p:
        :param b:
        :param kind: now ignored
        :return:
        """
        assert self.tvar_unlimited_function is not None, 'tvar_unlimited_function is None, must call make_tvar_function first'
        assert b == self.b, f'b ({b}) must equal b used when calibrated, self.b ({self.b})'

        tvar = self.tvar_unlimited_function(p)
        if b == np.inf:
            # no adjustment needed for unlimited losses
            return tvar
        tvar = np.where(p < self.Fb,
                        tvar - (1 - self.Fb) * (self.tvar_unlimited_function(self.Fb) - b) / (1 - p),
                        b)
        return tvar

    def compute_weight(self, premium, p0, p1, b=np.inf, kind='interp'):
        """
        compute the weight for a single TVaR p0 < p1 value pair

        :param line:
        :param premium:
        :param tp:
        :param b:
        :return:
        """

        assert p0 < p1
        assert self.tvar_unlimited_function is not None

        lhs = self.tvar_with_bound(p0, b, kind)
        rhs = self.tvar_with_bound(p1, b, kind)

        assert lhs != rhs
        weight = (premium - lhs) / (rhs - lhs)
        return weight

    def compute_weights(self, line, premium, n_tps, b=np.inf, kind='interp'):
        """
        Compute the weights of the extreme distortions

        Applied to min(line, b)  (allows to work for net)

        Note: independent of the asset level

        :param line: within port, or total
        :param premium: target premium for the line
        :param n_tps: number of tvar p points (tps)number of tvar p points (tps)number of tvar p points
            (tps)number of tvar p points (tps).
        :param b: loss bound: compute weights for min(line, b); generally used for net losses only.
        :return:
        """

        self.tvar_array(line, n_tps, b, kind)
        # you add zero, so there will be one additional point
        # n_tps += 1
        p_star = self.p_star(line, premium, b, kind)
        logger.info(f'compute weights: p_star {p_star} with premium {premium}, b={b}, and kind={kind}')
        # if p_star in self.tps:
        #     logger.critical(f'a Found p_star = {p_star} in tps!!')
        # else:
        #     logger.info('p_star not in tps')

        lhs = self.tps[self.tps <= p_star]
        rhs = self.tps[self.tps > p_star]

        tlhs = self.tvars[self.tps <= p_star]
        trhs = self.tvars[self.tps > p_star]

        lhs, rhs = np.meshgrid(lhs, rhs)
        tlhs, trhs = np.meshgrid(tlhs, trhs)

        df = pd.DataFrame({'p_lower': lhs.flat, 'p_upper': rhs.flat,
                           't_lower': tlhs.flat, 't_upper': trhs.flat,
                           })
        # will fail when p_star in self.ps; let's deal with then when it happens
        df['weight'] = (premium - df.t_lower) / (df.t_upper - df.t_lower)

        df = df.set_index(['p_lower', 'p_upper'], verify_integrity=True)
        df = df.sort_index()

        if p_star in self.tps:
            # raise ValueError('Found pstar in ps')
            logger.critical(f'Found p_star = {p_star} in tps; setting weight to 1.')
            df.at[(p_star, p_star), 'weight'] = 1.0

        logger.info(f'p_star={p_star:.4f}, len(p<=p*) = {len(df.index.levels[0])}, '
                    f'len(p>p*) = {len(df.index.levels[1])}; '
                    f' pstar in ps: {p_star in self.tps}')

        self.weight_df = df

        # index for tp values
        r = np.arange(n_tps)
        r_rhs, r_lhs = np.meshgrid(r[self.tps > p_star], r[self.tps <= p_star])
        self.idx = np.vstack((r_lhs.flat, r_rhs.flat)).reshape((2, r_rhs.size))

    def tvar_hinges(self, s):
        """
        make the tvar hinge functions by evaluating each tvar_p(s) = min(1, s/(1-p) for p in tps, at EP points s

        all arguments in [0,1] x [0,1]

        :param s:
        :return:
        """

        self.hinges = coo_matrix(np.minimum(1.0, s.reshape(
            1, len(s)) / (1.0 - self.tps.reshape(len(self.tps), 1))))

    def tvar_cloud(self, line, premium, a, n_tps, s, kind='interp'):
        """
        weight down tvar functions to the extremal convex measures

        asset level a acts like an agg stop on what is being priced, i.e. we are working with min(X, a)

        :param line:
        :param premium:
        :param a:
        :param n_tps:
        :param s:
        :param b:  bound, applies to min(line, b)
        :return:
        """

        self.compute_weights(line, premium, n_tps, a, kind)

        if type(s) == int:
            # points at which g is evaluated - all OK to include 0 and 1
            # s = np.linspace(0, 1, s+1, endpoint=True)
            s = self.make_ps(s, 's')

        self.tvar_hinges(s)

        ml = coo_matrix((1 - self.weight_df.weight, (np.arange(len(self.weight_df)), self.idx[0])),
                        shape=(len(self.weight_df), len(self.tps)))
        mr = coo_matrix((self.weight_df.weight, (np.arange(len(self.weight_df)), self.idx[1])),
                        shape=(len(self.weight_df), len(self.tps)))
        m = ml + mr

        logger.info(
            f'm shape = {m.shape}, hinges shape = {self.hinges.shape}, types {type(m)}, {type(self.hinges)}')

        self.cloud_df = pd.DataFrame(
            (m @ self.hinges).T.toarray(), index=s, columns=self.weight_df.index)
        self.cloud_df.index.name = 's'

    def cloud_view(self, *, axs=None, n_resamples=0, scale='linear', alpha=0.05, pricing=True, distortions='ordered',
                   title='', lim=(-0.025, 1.025), check=False, add_average=True):
        """
        Visualize the distortion cloud with n_resamples. Execute after computing weights.

        :param axs:
        :param n_resamples: if random sample
        :param scale: linear or return
        :param alpha: opacity
        :param pricing: restrict to p_max = 0, ensuring g(s)<1 when s<1
        :param distortions: 'ordered' shows the usual calibrated distortions, else list of dicts name:distortion.
        :param title: optional title (applied to all plots)
        :param lim: axis limits
        :param check:   construct and plot Distortions to check working ; reduces n_resamples to 5
        :return:
        """
        assert scale in ['linear', 'return']
        if axs is None:
            # include squeeze to make consistent with %%sf cell magic
            fig, axs = plt.subplots(1, 3, figsize=(3 * FIG_W, FIG_W), constrained_layout=True, squeeze=False)
            axs = axs[0]
        else:
            if axs.ndim == 2:
                axs = axs[0]
            fig = axs[0].get_figure()

        assert not distortions or (len(axs.flat) > 1)

        if distortions == 'ordered':
            assert isinstance(self.distribution_spec, Portfolio), \
                'distortion=ordered only available for Portfolio'
            distortions = [
                {k: self.distribution_spec.dists[k] for k in ['ccoc', 'tvar']},
                {k: self.distribution_spec.dists[k] for k in ['ph', 'wang', 'dual']}]

        bit = None
        if check:
            n_resamples = min(n_resamples, 5)
        norm = mpl.colors.Normalize(0, 1)
        cm = mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r')
        mapper = cm.get_cmap()
        s = np.linspace(0, 1, 1001)

        def plot_max_min(ax):
            ax.fill_between(self.cloud_df.index, self.cloud_df.min(
                1), self.cloud_df.max(1), facecolor='C7', alpha=.15)
            self.cloud_df.min(1).plot(
                ax=ax, label='_nolegend_', lw=1, ls='-', c='k')
            self.cloud_df.max(1).plot(
                ax=ax, label="_nolegend_", lw=1, ls='-', c='k')

        logger.info('starting cloudview...')
        if scale == 'linear':
            ax = axs[0]
            if n_resamples > 0:
                if pricing:
                    bit = self.weight_df.xs(0, drop_level=False).sample(
                        n=n_resamples, replace=True).reset_index()
                else:
                    bit = self.weight_df.sample(
                        n=n_resamples, replace=True).reset_index()
                logger.info('cloudview...done 1')
                # display(bit)
                for i in bit.index:
                    pl, pu, tl, tu, w = bit.loc[i]
                    self.cloud_df[(pl, pu)].plot(
                        ax=ax, lw=1, c=mapper(w), alpha=alpha, label=None)
                    if check:
                        # put in actual for each sample
                        d = Distortion('wtdtvar', w, df=[pl, pu])
                        gs = d.g(s)
                        ax.plot(s, gs, c=mapper(w), lw=2, ls='--',
                                alpha=.5, label=f'ma ({pl:.3f}, {pu:.3f}) ')
                ax.get_figure().colorbar(cm, ax=ax, shrink=.5, aspect=16,
                                         label='Weight to Higher Threshold')
            else:
                logger.info('cloudview: no resamples, skipping 1')
            logger.info('cloudview: start max/min')
            plot_max_min(ax)
            logger.info('cloudview: done with max/min')
            for ln in ax.lines:
                ln.set(label=None)
            if check:
                ax.legend(loc='lower right', fontsize='large')
            ax.plot([0, 1], [0, 1], c='k', lw=.25, ls='-')
            ax.set(xlim=lim, ylim=lim, aspect='equal')

            if type(distortions) == dict:
                distortions = [distortions]
            if distortions == 'space':
                ax = axs[1]
                plot_max_min(ax)
                ax.plot([0, 1], [0, 1], c='k', lw=.25,
                        ls='-', label='_nolegend_')
                ax.legend(loc='lower right', ncol=3, fontsize='large')
                ax.set(xlim=lim, ylim=lim, aspect='equal')
            elif type(distortions) == list:
                logger.info('cloudview: start 4 adding distortions')
                name_mapper = {
                    'roe': 'CCoC', 'tvar': 'TVaR(p*)', 'ph': 'PH', 'wang': 'Wang', 'dual': 'Dual'}
                lss = list(mpl.lines.lineStyles.keys())
                for ax, dist_dict in zip(axs[1:], distortions):
                    lssi = iter(cycle(lss))
                    for k, d in dist_dict.items():
                        gs = d.g(s)
                        k = name_mapper.get(k, k)
                        ax.plot(s, gs, lw=1, ls=next(lssi), label=k)
                    plot_max_min(ax)
                    ax.plot([0, 1], [0, 1], c='k', lw=.25,
                            ls='-', label='_nolegend_')
                    ax.legend(loc='lower right', ncol=3, fontsize='large')
                    ax.set(xlim=lim, ylim=lim, aspect='equal')
                if add_average:
                    self.cloud_df.mean(1).plot(ax=ax, c=f'C{len(distortions[-1])}',
                                               ls='-.', lw=.5, label='Avg extreme')
            else:
                # do nothing
                pass

        elif scale == 'return':
            ax = axs[0]
            bit = self.cloud_df.sample(n=n_resamples, axis=1)
            bit.index = 1 / bit.index
            bit = 1 / bit
            bit.plot(ax=ax, lw=.5, c='C7', alpha=alpha)
            ax.plot([0, 1000], [0, 1000], c='C0', lw=1)
            ax.legend().set(visible=False)
            ax.set(xscale='log', yscale='log')
            ax.set(xlim=[2000, 1], ylim=[2000, 1])

        if title != '':
            for ax in axs:
                if bit is not None:
                    title1 = f'{title}, n={len(bit)} samples'
                else:
                    title1 = title
                ax.set(title=title1)
        for ax in axs[1:]:
            ax.legend(ncol=1, loc='lower right')
        for ax in axs:
            ax.set(title=None)
        return fig, axs

    def weight_image(self, ax, levels=20, colorbar=True):
        bit = self.weight_df.weight.unstack()
        img = ax.contourf(bit.columns, bit.index, bit,
                          cmap='viridis_r', levels=levels)
        ax.set(xlabel='p1', ylabel='p0', title='Weight for p1', aspect='equal')
        if colorbar:
            ax.get_figure().colorbar(img, ax=ax, shrink=.5, aspect=16, label='Weight to p_upper')

    def distortion(self, pl, pu):
        """
        Return the BiTVaR with probabilities pl and pu
        """
        assert self.weight_df is not None, 'Must create weight_df before running this function'

        tl, tu, w = self.weight_df.loc[(pl, pu)]
        return Distortion('bitvar', w, df=[pl, pu])

    def quick_price(self, distortion, a):
        """
        price total to assets a using distortion

        requires distribution_spec has a density_df dataframe with a p_total or p_total

        TODO: add ability to price other lines
        :param distortion:
        :param a:
        :return:
        """

        if isinstance(self.distribution_spec, (Portfolio, Aggregate)):
            df = self.distribution_spec.density_df
            bs = self.distribution_spec.bs
        elif isinstance(self.distribution_spec, pd.DataFrame):
            df = self.distribution_spec
            bs = df.index[1]
        else:
            raise NotImplemented('Must input Aggregate, Portfolio, or DataFrame, '
                                 f'not type {type(self.distribution_spec)}')

        temp = distortion.g(
            df.p_total.shift(-1, fill_value=0)[::-1].cumsum())[::-1]

        if isinstance(temp, np.ndarray):
            # not all g functions return Series (you can't guarantee it is called on something with an index)
            temp = pd.Series(temp, index=df.index)

        temp = temp.shift(1, fill_value=0).cumsum() * bs

        if np.isinf(a):
            r = len(temp) - 1
        else:
            r = temp.index.get_loc(a)

        return temp.iloc[r] * bs

    def principal_extreme_distortion_analysis(self, gs, pricing=False):
        """
        Find the principal extreme distortion analysis to solve for gs = g(s), s=self.cloud_df.index

        Assumes that tvar_cloud has been called and that cloud_df exists
        len(gs) = len(cloud_df)

        E.g., call

            b = Bounds(port)
            b.t_mode = 'u'
            # set premium and asset level a
            b.tvar_cloud('total', premium, a)
            # make gs
            b.principal_extreme_distortion_analysis(gs)

        :param gs: either g(s) evaluated on s = cloud_df.index or the name of a calibrated distortion in
            distribution_spec.dists (created by a call to calibrate_distortions)
        :param pricing: if try, try just using pricing distortions
        :return:
        """

        assert self.cloud_df is not None

        if type(gs) == str:
            s = np.array(self.cloud_df.index)
            gs = self.distribution_spec.dists[gs].g(s)

        assert len(gs) == len(self.cloud_df)

        if pricing:
            _ = self.cloud_df.xs(0, axis=1, level=0, drop_level=False)
            X = _.to_numpy()
            idx = _.columns
        else:
            _ = self.cloud_df
            X = _.to_numpy()
            idx = _.columns
        n = X.shape[1]

        print(X.shape, self.cloud_df.shape)

        # Moore Penrose solution
        mp = pinv(X) @ gs
        logger.info('Moore-Penrose solved...')

        # optimization solutions
        A = np.hstack((X, np.eye(X.shape[0])))
        b_eq = gs
        c = np.hstack((np.zeros(X.shape[1]), np.ones_like(b_eq)))

        lprs = linprog(c, A_eq=A, b_eq=b_eq, method='revised simplex')
        logger.info(
            f'Revised simpled solved...\nSum of added variables={np.sum(lprs.x[n:])} (should be zero for exact)')
        self.lprs = lprs

        lpip = linprog(c, A_eq=A, b_eq=b_eq, method='interior-point')
        logger.info(
            f'Interior point solved...\nSum of added variables={np.sum(lpip.x[n:])}')
        self.lpip = lpip

        print(lprs.x, lpip.x)

        # consolidate answers
        self.pedw_df = pd.DataFrame(
            {'w_mp': mp, 'w_rs': lprs.x[:n], 'w_ip': lpip.x[:n]}, index=idx)
        self.pedw_df['w_upper'] = self.weight_df.weight

        # diagnostics
        for c in self.pedw_df.columns[:-1]:
            answer = self.pedw_df[c].values
            ganswer = answer[answer > 1e-16]
            logger.info(
                f'Method {c}\tMinimum parameter {np.min(answer)}\tNumber non-zero {len(ganswer)}')

        return gs

    def ped_distortion(self, n, solver='rs'):
        """
        make the approximating distortion from the first n Principal Extreme Distortions (PED)s using rs or ip solutions

        :param n:
        :return:
        """
        assert solver in ['rs', 'ip']

        # the weight column for solver
        c = f'w_{solver}'
        # pull off the tvar and PED weights
        df = self.pedw_df.sort_values(c, ascending=False)
        bit = df.loc[df.index[:n], [c, 'w_upper']]
        # re-weight partial (method / lp-solve) weights to 1
        bit[c] /= bit[c].sum()
        # multiply lp-solve weights with the weigh_df extreme distortion p_lower/p_upper weights
        bit['c_lower'] = (1 - bit.w_upper) * bit[c]
        bit['c_upper'] = bit.w_upper * bit[c]
        # gather into data frame of p and total weight (labeled c)
        bit2 = bit.reset_index().drop([c, 'w_upper'], 1)
        bit2.columns = bit2.columns.str.split('_', expand=True)
        bit2 = bit2.stack(1).groupby('p')['c'].sum()
        # bit2 has index = probability points and values = weights for the wtd tvar distortion
        d = Distortion.wtd_tvar(bit2.index, bit2.values, f'PED({solver}, {n})')
        return d


def similar_risks_graphs_sa(axd, bounds, port, pnew, roe, prem, p_reg=1):
    """
    stand-alone
    ONLY WORKS FOR BOUNDED PORTFOLIOS (use for beta mixture examples)
    Updated version in CaseStudy
    axd from mosaic
    bounds = Bounds class from port (calibrated to some base)it
    pnew = new portfolio
    input new beta(a,b) portfolio, using existing bounds object

    sample: see similar_risks_sample()

    Provenance : from make_port in Examples_2022_post_publish
    """

    if axd is None:
        fig = plt.figure(constrained_layout=True, figsize=(12, 6))
        axd = fig.subplot_mosaic(
            '''
            AAAABBFF
            AAAACCFF
            AAAADDEE
            AAAADDEE
        ''')

    df = bounds.weight_df.copy()
    df['test'] = df['t_upper'] * df.weight + df.t_lower * (1 - df.weight)

    # HERE IS ISSUE - should really use tvar with bounds and incorporate the bound
    if p_reg < 1:
        logger.warning('figuring tvars with bounds')
        btemp = Bounds(pnew)
        b = pnew.q(p_reg)
        btemp.make_tvar_function('total', b=b)
        tvar1 = {p: btemp.tvar_with_bound(p, b=b) for p in bounds.tps}
    else:
        tvar1 = {p: float(pnew.tvar(p)) for p in bounds.tps}
    df['t1_lower'] = [tvar1[p] for p in df.index.get_level_values(0)]
    df['t1_upper'] = [tvar1[p] for p in df.index.get_level_values(1)]
    df['t1'] = df.t1_upper * df.weight + df.t1_lower * (1 - df.weight)

    roe_d = Distortion('roe', roe)
    tvar_d = Distortion('tvar', bounds.p_star('total', prem))
    idx = df.index.get_locs(df.idxmax()['t1'])[0]
    pl, pu, tl, tu, w = df.reset_index().iloc[idx, :-4]
    max_d = Distortion('wtdtvar', w, df=[pl, pu])

    tmax = float(df.iloc[idx]['t1'])
    n_ = len(df.query('t1 == @tmax'))
    logger.warning(f'Ties for max: {n_}')
    n_ = len(df.query(f't1 >= {tmax} - 1e-4'))
    logger.warning(f'Near ties for max: {n_}')

    idn = df.index.get_locs(df.idxmin()['t1'])[0]
    pln, pun, tl, tu, wn = df.reset_index().iloc[idn, :-4]
    min_d = Distortion('wtdtvar', wn, df=[pln, pun])

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
    # the average
    bounds.cloud_df.mean(1).plot(ax=ax, c='C3',
                               ls='-.', lw=1.5, label='Avg extreme')
    ax.legend(loc='upper left')

    ax.set(title=f'Max ({pl}, {pu}), min ({pln}, {pun})')

    ax = axd['B']
    bounds.weight_image(ax)

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
        port.dists[dd].plot(ax=ax, both=False, lw=1)
        ax.lines[n].set(c=c, label=dd)
        n += 2
    ax.legend(loc='lower right')

    return df


def similar_risks_example():
    """
    Interesting beta risks and how to use similar_risks_graphs_sa.


    :return:
    """
    # stand alone hlep from the code; split at program = to run different options
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
    prem, roe
    p_base.calibrate_distortions(As=[1], ROEs=[roe], strict='ordered')
    bounds = Bounds(p_base)
    bounds.tvar_cloud('total', prem, a, 128 * 2, 64 * 2, 'interp')
    p_star = bounds.p_star('total', prem, kind='interp')

    smfig = FigureManager(cycle='c', color_mode='color', font_size=10, legend_font='small',
                          default_figsize=(FIG_W, FIG_H))

    f, axs = smfig(1, 3, (18.0, 6.0), )
    ax0, ax1, ax2 = axs.flat
    axi = iter(axs.flat)
    # all with base portfolio

    bounds.cloud_view(axs.flatten(), 0, alpha=1, pricing=True,
                      title=f'Premium={prem:,.1f}, a={a:,.0f}, p*={p_star:.3f}',
                      distortions=[{k: p_base.dists[k] for k in ['ccoc', 'tvar']},
                                   {k: p_base.dists[k] for k in ['ph', 'wang', 'dual']}])
    for ax in axs.flatten()[1:]:
            ax.legend(ncol=1, loc='lower right')
    for ax in axs.flatten():
        ax.set(title=None)

    program = '''
    port BETA
        agg TWO 1 claim sev 1 * beta [200 300 400 500 600 7] [600 500 400 300 200 1] wts=6 fixed
        # never worked
        # agg TWO 1 claim sev 1 * beta [1 2000 4000 6000 50] [100 6000 4000 2000 1] wts[0.1875 0.1875 0.1875 0.1875 .25] fixed
        # interior solution:
        # agg TWO 1 claim sev 1 * beta [300 400 500 600 35] [500 400 300 200 5] wts[.125 .25 .125 .25 .25] fixed
        #
        # agg TWO 1 claim sev 1 * beta [50 30 1] [1 40 10] wts=3 fixed
        # agg TWO 1 claim sev 1 * beta [50 30 1] [1 40 10] wts[.375 .375 .25] fixed

    '''
    p_new = uw.build(program)
    p_new.update(11, 1 / 1024, remove_fuzz=True)

    p_new.plot(figsize=(6, 4))

    axd = plt.figure(constrained_layout=True, figsize=(16, 8)).subplot_mosaic(
        '''
        AAAABBFF
        AAAACCFF
        AAAADDEE
        AAAADDEE
    '''
    )
    df = similar_risks_graphs_sa(axd, bounds, p_base, p_new, roe, prem)
    return df


def plot_max_min(self, ax):
    """
    Extracted from bounds, self=Bounds object
    """
    ax.fill_between(self.cloud_df.index, self.cloud_df.min(
        1), self.cloud_df.max(1), facecolor='C7', alpha=.15)
    self.cloud_df.min(1).plot(ax=ax, label='_nolegend_', lw=0.5, ls='-', c='k')
    self.cloud_df.max(1).plot(ax=ax, label="_nolegend_", lw=0.5, ls='-', c='k')


def plot_lee(port, ax, c, lw=1):
    """
    Lee diagram by hand
    """
    p_ = np.linspace(0, 1, 1001)
    qs = [port.q(p) for p in p_]
    ax.step(p_, qs, lw=lw, c=c, label=port.name)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, max(qs) + .05],
           title=f'{port.name} Lee diagram')
