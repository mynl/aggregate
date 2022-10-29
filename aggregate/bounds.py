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


logger = logging.getLogger(__name__)


class Bounds(object):
    """
    Implement IME 2022 pricing bounds methodology.

    Typical usage: First, create a Portfolio or Aggregate object a. Then ::

        bd = cd.Bounds(a)
        bd.tvar_cloud('line', premium=, a=, n_tps=, s=, kind=)
        p_star = bd.p_star('line', premium)
        bd.cloud_view(axes, ...)

    :param: distribution_spec = Portfolio or Portfolio.density_df dataframe or pd.Series (must have loss as index)
            If DataFrame or Series values interpreted as desnsity, sum to 1. F, S, exgta all computed using Portfolio
            methdology
            If DataFrame line --> p_{line}
    """
    # from common_scripts.cs

    def __init__(self, distribution_spec):
        assert isinstance(distribution_spec, (pd.Series, pd.DataFrame, Portfolio, Aggregate))
        self.distribution_spec = distribution_spec
        # although passed as input to certain functions (tvar with bounds) b is actually fixed
        self.b = 0
        self.Fb = 0
        self.tvar_function = None
        self.tvars = None
        self.tps = None
        self.weight_df = None
        self.idx = None
        self.hinges = None
        # in cases where we hold the tvar function here
        self._tail_var = None
        self._inverse_tail_var = None
        self.cloud_df = None
        # uniform mode
        self._t_mode = 'u'
        # data frame with tvar weights and principal extreme distortion weights by method
        self.pedw_df = None
        self._tvar_df = None
        # hack for beta distribution, you want to force 1 to be in tvar ps, but Fp = 1
        # TODO figure out why p_star grinds to a halt if you input b < inf
        self.add_one = False

    def __repr__(self):
        """
        Gets called automatically but so we can tweak.

        :return:
        """
        return 'My Bounds Object at ' + super(Bounds, self).__repr__()

    def __str__(self):
        return 'Hello' + super(Bounds, self).__repr__()

    @property
    def tvar_df(self):
        if self._tvar_df is None:
            self._tvar_df = pd.DataFrame({'p': self.tps, 'tvar': self.tvars}).set_index('p')
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
        make the tvar function from a Series p_total indexed by loss
        Includes determining sup and putting in value for zero
        If sup is largest value in index, sup set to inf

        also sets self.Fb

        Applies to min(Line, b)

        :param line:
        :param b:  bound on the losses, e.g., to model limited liability insurer
        :return:
        """
        self.b = b
        if isinstance(self.distribution_spec, Portfolio):
            assert line in self.distribution_spec.line_names_ex
            if line == 'total':
                self.tvar_function = self.distribution_spec.tvar
                self.Fb = self.distribution_spec.cdf(b)
            else:
                ag = getattr(self.distribution_spec, line)
                self.tvar_function = ag.tvar
                self.Fb = ag.cdf(b)
            if np.isinf(b): self.Fb = 1.0
            return

        elif isinstance(self.distribution_spec, Aggregate):
            self.tvar_function = self.distribution_spec.tvar
            self.Fb = self.distribution_spec.cdf(b)
            if np.isinf(b): self.Fb = 1.0
            return

        elif isinstance(self.distribution_spec, pd.DataFrame):
            assert f'p_{line}' in self.distribution_spec.columns
            # given a port.density_df
            p_total = self.distribution_spec[f'p_{line}']

        elif isinstance(self.distribution_spec, pd.Series):
            logger.info('tvar_array using Series')
            p_total = self.distribution_spec

        # need to create tvar function on the fly, using same method as Portfolio and Aggregate:
        bs = p_total.index[1]
        F = p_total.cumsum()
        if np.isinf(b):
            self.Fb = 0
        else:
            self.Fb = F[b]

        S = p_total.shift(-1, fill_value=min(p_total.iloc[-1], max(0, 1. - (p_total.sum()))))[::-1].cumsum()[::-1]
        lev = S.shift(1, fill_value=0).cumsum() * bs
        ex1 = lev.iloc[-1]
        ex = np.sum(p_total * p_total.index)
        logger.info(f'Computed mean loss for {line} = {ex:,.15f} (diff {ex - ex1:,.15f}) max F = {max(F)}')
        exgta = (ex - lev) / S + S.index
        sup = (p_total[::-1] > 0).idxmax()
        if sup == p_total.index[-1]:
            sup = np.inf
        exgta[S == 0] = sup
        logger.info(f'sup={sup}, max = {(p_total[::-1] > 0).idxmax()} "inf" = {p_total.index[-1]}')

        def _tvar(p, kind='interp'):
            """
            UNLIMITED tvar function!
            :param p:
            :param kind:
            :return:
            """
            if kind == 'interp':
                # original implementation interpolated
                if self._tail_var is None:
                    # make tvar function
                    self._tail_var = interp1d(F, exgta, kind='linear', bounds_error=False,
                                              fill_value=(0, sup))
                return self._tail_var(p)
            elif kind == 'inverse':
                if self._inverse_tail_var is None:
                    # make tvar function
                    self._inverse_tail_var = interp1d(exgta, F, kind='linear', bounds_error=False,
                                                      fill_value='extrapolate')
                return self._inverse_tail_var(p)

        self.tvar_function = _tvar

    def make_ps(self, n, mode):
        """
        Mode are you making s points (always uniform) or tvar p points (use t_mode)?
        self.t_mode == 'u': make uniform s points against which to evaluate g from 0 to 1 inclusive with more around 0
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
        if mode == 't' and self.Fb < 1 or self.add_one:
            ps = np.hstack((ps, 1))
        return ps

    def tvar_array(self, line, n_tps=256, b=np.inf, kind='interp'):
        """
        Compute tvars at n equally spaced points, tps.


        :param line:
        :param n_tps:  number of tvar p points, default 256
        :param b: cap on losses applied before computing TVaRs (e.g., adjust losses for finite assets b).
               Use np.inf for unlimited losses.
        :param kind: if interp  uses the standard function, easy, for continuous distributions; if 'tail' uses
               explicit integration of tail values, for discrete distributions
        :return:
        """
        assert kind in ('interp', 'tail')
        self.make_tvar_function(line, b)

        logger.info(f'F(b) = {self.Fb:.5f}')
        # tvar p values should linclude 0 (the mean) but EXCLUDE 1
        # self.tps = np.linspace(0.5 / n_tps, 1 - 0.5 / n_tps, n_tps)
        self.tps = self.make_ps(n_tps, 't')

        if kind == 'interp':
            self.tvars = self.tvar_function(self.tps)
            if not np.isinf(b):
                # subtract S(a)(TVaR(F(a)) - a)
                # do all at once here - do not call self.tvar_with_bounds function
                self.tvars = np.where(self.tps <= self.Fb,
                                      self.tvars - (1 - self.Fb) * (self.tvar_function(self.Fb) - b) / (1 - self.tps),
                                      b)
        elif kind == 'tail':
            self.tvars = np.array([self.tvar_with_bound(i, b, kind) for i in self.tps])

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
        :return:
        """
        assert kind in ('interp', 'tail')
        if premium > b:
            raise ValueError(f'p_star must have premium ({premium}) <= largest loss bound ({b})')

        if kind == 'interp':

            self.make_tvar_function(line, b)

            if np.isinf(b):
                p_star = self.tvar_function(premium, 'inverse')
            else:
                # nr, remember F(a) is self.Fa set by make_tvar_function
                k = (1 - self.Fb) * (self.tvar_function(self.Fb) - b)

                def f(p):
                    return self.tvar_function(p) - k / (1 - p) - premium

                # should really compute f' numerically, but...
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
                    logger.warning(f'Questionable convergence solving for p_star, last error {fp}.')
                p_star = p

        elif kind == 'tail':
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
                logger.warning(f'Questionable convergence solving for p_star, last error {fp}.')
            p_star = p

        return p_star

    def tvar_with_bound(self, p, b=np.inf, kind='interp'):
        """
        Compute tvar taking bound into account.
        Assumes tvar_function setup.

        Warning: b must equal the b used when calibrated. The issue is computing F
        varies with the type of underlying portfolio. This is fragile.
        Added storing b and checking equal. For backwards comp. need to keep b argument

        :param p:
        :param b:
        :return:
        """
        assert self.tvar_function is not None
        assert b == self.b

        if kind == 'interp':
            tvar = self.tvar_function(p)
            if not np.isinf(b):
                if p < self.Fb:
                    tvar = tvar - (1 - self.Fb) * (self.tvar_function(self.Fb) - b) / (1 - p)
                else:
                    tvar = b
        elif kind == 'tail':
            # use the tail method for discrete distributions
            tvar = self.distribution_spec.tvar(p, 'tail')
            if not np.isinf(b):
                if p < self.Fb:
                    tvar = tvar - (1 - self.Fb) * (self.distribution_spec.tvar(self.Fb, 'tail') - b) / (1 - p)
                else:
                    tvar = b
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
        assert self.tvar_function is not None

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
        if p_star in self.tps:
            logger.critical('p_star in tps')
            # raise ValueError()

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
            logger.critical(f'Found p_star = {p_star} in ps!!')
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

        self.hinges = coo_matrix(np.minimum(1.0, s.reshape(1, len(s)) / (1.0 - self.tps.reshape(len(self.tps), 1))))

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

        logger.info(f'm shape = {m.shape}, hinges shape = {self.hinges.shape}, types {type(m)}, {type(self.hinges)}')

        self.cloud_df = pd.DataFrame((m @ self.hinges).T.toarray(), index=s, columns=self.weight_df.index)
        self.cloud_df.index.name = 's'

    def cloud_view(self, axs, n_resamples, scale='linear', alpha=0.05, pricing=True, distortions=None,
                   title='', lim=(-0.025, 1.025), check=False):
        """
        visualize the cloud with n_resamples

        after you have recomputed...

        if there are distortions plot on second axis

        :param axs:
        :param n_resamples: if random sample
        :param scale: linear or return
        :param alpha: opacity
        :param pricing: restrict to p_max = 0, ensuring g(s)<1 when s<1
        :param distortions:
        :param title: optional title (applied to all plots)
        :param lim: axis limits
        :param check:   construct and plot Distortions to check working ; reduces n_resamples to 5
        :return:
        """
        assert scale in ['linear', 'return']
        assert not distortions or (len(axs.flat) > 1)
        bit = None
        if check: n_resamples = min(n_resamples, 5)
        norm = mpl.colors.Normalize(0, 1)
        cm = mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r')
        mapper = cm.get_cmap()
        s = np.linspace(0, 1, 1001)

        def plot_max_min(ax):
            ax.fill_between(self.cloud_df.index, self.cloud_df.min(1), self.cloud_df.max(1), facecolor='C7', alpha=.15)
            self.cloud_df.min(1).plot(ax=ax, label='_nolegend_', lw=1, ls='-', c='k')
            self.cloud_df.max(1).plot(ax=ax, label="_nolegend_", lw=1, ls='-', c='k')

        logger.info('starting cloudview...')
        if scale == 'linear':
            ax = axs[0]
            if n_resamples > 0:
                if pricing:
                    bit = self.weight_df.xs(0, drop_level=False).sample(n=n_resamples, replace=True).reset_index()
                else:
                    bit = self.weight_df.sample(n=n_resamples, replace=True).reset_index()
                logger.info('cloudview...done 1')
                # display(bit)
                for i in bit.index:
                    pl, pu, tl, tu, w = bit.loc[i]
                    self.cloud_df[(pl, pu)].plot(ax=ax, lw=1, c=mapper(w), alpha=alpha, label=None)
                    if check:
                        # put in actual for each sample
                        d = Distortion('wtdtvar', w, df=[pl, pu])
                        gs = d.g(s)
                        ax.plot(s, gs, c=mapper(w), lw=2, ls='--', alpha=.5, label=f'ma ({pl:.3f}, {pu:.3f}) ')
                ax.get_figure().colorbar(cm, ax=ax, shrink=.5, aspect=16, label='Weight to Higher Threshold')
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
                ax.plot([0, 1], [0, 1], c='k', lw=.25, ls='-', label='_nolegend_')
                ax.legend(loc='lower right', ncol=3, fontsize='large')
                ax.set(xlim=lim, ylim=lim, aspect='equal')
            elif type(distortions) == list:
                logger.info('cloudview: start 4 adding distortions')
                name_mapper = {'roe': 'CCoC', 'tvar': 'TVaR(p*)', 'ph': 'PH', 'wang': 'Wang', 'dual': 'Dual'}
                lss = list(mpl.lines.lineStyles.keys())
                for ax, dist_dict in zip(axs[1:], distortions):
                    ii = 1
                    for k, d in dist_dict.items():
                        gs = d.g(s)
                        k = name_mapper.get(k, k)
                        ax.plot(s, gs, lw=1, ls=lss[ii], label=k)
                        ii += 1
                    plot_max_min(ax)
                    ax.plot([0, 1], [0, 1], c='k', lw=.25, ls='-', label='_nolegend_')
                    ax.legend(loc='lower right', ncol=3, fontsize='large')
                    ax.set(xlim=lim, ylim=lim, aspect='equal')
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

    def weight_image(self, ax, levels=20, colorbar=True):
        bit = self.weight_df.weight.unstack()
        img = ax.contourf(bit.columns, bit.index, bit, cmap='viridis_r', levels=levels)
        ax.set(xlabel='p1', ylabel='p0', title='Weight for p1', aspect='equal')
        if colorbar:
            ax.get_figure().colorbar(img, ax=ax, shrink=.5, aspect=16, label='Weight to p_upper')

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

        temp = distortion.g(df.p_total.shift(-1, fill_value=0)[::-1].cumsum())[::-1]

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
        logger.info(f'Interior point solved...\nSum of added variables={np.sum(lpip.x[n:])}')
        self.lpip = lpip

        print(lprs.x, lpip.x)

        # consolidate answers
        self.pedw_df = pd.DataFrame({'w_mp': mp, 'w_rs': lprs.x[:n], 'w_ip': lpip.x[:n]}, index=idx)
        self.pedw_df['w_upper'] = self.weight_df.weight

        # diagnostics
        for c in self.pedw_df.columns[:-1]:
            answer = self.pedw_df[c].values
            ganswer = answer[answer > 1e-16]
            logger.info(f'Method {c}\tMinimum parameter {np.min(answer)}\tNumber non-zero {len(ganswer)}')

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


def similar_risks_graphs_sa(axd, bounds, port, pnew, roe, prem):
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

    df = bounds.weight_df.copy()
    df['test'] = df['t_upper'] * df.weight + df.t_lower * (1 - df.weight)

    # HERE IS ISSUE - should really use tvar with bounds and incorporate the bound
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
    print('Ties for max: ', len(df.query('t1 == @tmax')))
    print('Near ties for max: ', len(df.query('t1 >= @tmax - 1e-4')))

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

    ax.lines[n + 0].set(label='roe')
    ax.lines[n + 2].set(color='green', label='tvar')
    ax.lines[n + 4].set(color='red', label='max')
    ax.lines[n + 6].set(color='purple', label='min')
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
    plot_lee(port, ax, 'k', lw=1)
    plot_lee(pnew, ax, 'r')

    ax = axd['E']
    pnew.density_df.p_total.plot(ax=ax)
    ax.set(xlim=[-0.05, 1.05], title='Density')

    ax = axd['F']
    plot_max_min(bounds, ax)
    for c, dd in zip(['r', 'g', 'b'], ['ph', 'wang', 'dual']):
        port.dists[dd].plot(ax=ax, both=False, lw=1)
        ax.lines[n].set(c=c, label=dd)
        n += 2
    ax.legend(loc='lower right')

    return df


def similar_risks_example():
    """
    Interesting beta risks and how to use similar_risks_sa.


    :return:
    """
    # stand alone hlep from the code; split at program = to run different options
    uw = Underwriter()
    p_base = uw.write('''
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
                          default_figsize=(5, 3.5))

    f, axs = smfig(1, 3, (18.0, 6.0), )
    ax0, ax1, ax2 = axs.flat
    axi = iter(axs.flat)
    # all with base portfolio

    bounds.cloud_view(axs.flatten(), 0, alpha=1, pricing=True,
                      title=f'Premium={prem:,.1f}, a={a:,.0f}, p*={p_star:.3f}',
                      distortions=[{k: p_base.dists[k] for k in ['roe', 'tvar']},
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
    p_new = uw.write(program)
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
    ax.fill_between(self.cloud_df.index, self.cloud_df.min(1), self.cloud_df.max(1), facecolor='C7', alpha=.15)
    self.cloud_df.min(1).plot(ax=ax, label='_nolegend_', lw=0.5, ls='-', c='w')
    self.cloud_df.max(1).plot(ax=ax, label="_nolegend_", lw=0.5, ls='-', c='w')


def plot_lee(port, ax, c, lw=2):
    """
    Lee diagram by hand
    """
    p_ = np.linspace(0, 1, 1001)
    qs = [port.q(p) for p in p_]
    ax.step(p_, qs, lw=lw, c=c)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, max(qs) + .05], title=f'Lee Diagram {port.name}', aspect='equal')

