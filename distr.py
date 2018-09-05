import scipy.stats as ss
import numpy as np
from aggregate.utils import cv_to_shape, mean_to_scale
from scipy.integrate import quad
import pandas as pd
import collections
import matplotlib.pyplot as plt
import logging
from .utils import sln_fit, sgamma_fit, ft, ift, \
    axiter_factory, estimate_agg_percentile, suptitle_and_tight, MomentAggregator
from .spectral import Distortion
from scipy import interpolate

# import matplotlib.cm as cm
# from scipy import interpolate
# from copy import deepcopy
# from ruamel import yaml
# from . utils import *

LOGFILE = 'c:/S/TELOS/python/aggregate/aggregate.log'
logging.basicConfig(filename=LOGFILE,
                    filemode='w',
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                    level=logging.DEBUG)
logging.info('aggregate.__init__ | New trash Session started')


class Aggregate(object):
    """
    Aggregate help placeholder

    """

    def __init__(self, name, el=0, premium=0, lr=0, en=0, attachment=0, limit=np.inf, sev_name='', sev_a=0, sev_b=0,
                 sev_mean=0, sev_cv=0, sev_loc=0, sev_scale=0, sev_xs=None, sev_ps=None, mix_wt=1,
                 freq_name='', freq_a=0, freq_b=0):
        """

        el -> en
        prem x lr -> el
        x . en -> el
        always have en x and el; may have prem and lr
        if prem then lr computed; if lr then premium computed

        el is determined using np.where(el==0, prem*lr, el)
        if el==0 then el = freq * sev
        assert np.all( el>0 or en>0 )

        call with el (or prem x lr) (or n) expressing a mixture, with the same severity
        call with el expressing lines of business with an array of severities
        call with single el and array of sevs expressing a mixture; [] broken down by weights

        n is the CONDITIONAL claim count
        X is the GROUND UP severity, so X | X > attachment is used and generates n claims

        For fixed or histogram have to separate the parameter so they are not broad cast; otherwise
        you end up with multiple lines when you intend only one

        TODO: later do both, for now assume one or other is a scalar
        call with both that does the cross product... (note the sev array may not be equal sizes)

        :param name:
        :param el:   expected loss or vector or matrix
        :param premium:
        :param lr:  loss ratio
        :param en:  expected claim count per segment (self.n = total claim count)
        :param attachment: occ attachment
        :param limit: occ limit
        :param sev_name: Severity class object or similar or vector or matrix
        :param sev_a:
        :param sev_b:
        :param sev_mean:
        :param sev_cv:
        :param sev_loc:
        :param sev_scale:
        :param sev_xs:  xs and ps must be provided if sev_name is (c|d)histogram or fixed
        :param sev_ps:
        :param mix_wt: weight for mixed distribution
        :param freq_name: name of frequency distribution
        :param freq_a: freq dist shape1 OR CV = sq root contagion
        :param freq_b: freq dist shape2

        """

        assert np.sum(mix_wt) == 1

        self.spec = dict(name=name, el=el, premium=premium, lr=lr, en=en,
                         attachment=attachment, limit=limit,
                         sev_name=sev_name, sev_a=sev_a, sev_b=sev_b,
                         sev_mean=sev_mean, sev_cv=sev_cv, sev_loc=sev_loc, sev_scale=sev_scale,
                         sev_xs=sev_xs, sev_ps=sev_ps, mix_wt=mix_wt,
                         freq_name=freq_name, freq_a=freq_a, freq_b=freq_b)

        # class variables (mostly)
        self.name = name
        self.freq_name = freq_name
        self.freq_a = freq_a
        self.freq_b = freq_b
        self.sev_density = None
        self.ftagg_density = None
        self.agg_density = None
        self.xs = None
        self.fzapprox = None
        self.n = 0  # total frequency
        self.aggm, self.aggcv, self.aggskew = 0, 0, 0

        # get other variables defined in init
        self.sev_density = None
        self.fzapprox = None
        self.agg_density = None
        self.ftagg_density = None
        self.xs = None
        self.bs = 0
        self.dh_agg_density = None
        self.dh_sev_density = None
        self.beta_name = ''  # name of the beta function used to create dh distortion
        self.sevs = None
        self.stats = pd.DataFrame(columns=['limit', 'attachment', 'sevcv_param', 'el', 'prem', 'lr'] +
                                          MomentAggregator.column_names(agg_only=False) +
                                          ['contagion', 'mix_cv'])
        self.stats_total = self.stats.copy()
        ma = MomentAggregator(freq_name, freq_a, freq_b)

        # broadcast arrays: first line forces them all to be arrays
        # TODO this approach forces EITHER a mixture OR multi exposures
        # TODO need to expand to product for general case, however, that should be easy

        if not isinstance(el, collections.Iterable):
            el = np.array([el])

        # pyCharm formatting
        self.en = None
        self.attachment = None
        self.limit = None
        el, premium, lr, self.en, self.attachment, self.limit, sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, \
        sev_scale, mix_wt = \
            np.broadcast_arrays(el, premium, lr, en, attachment, limit, sev_name, sev_a, sev_b, sev_mean, sev_cv,
                                sev_loc, sev_scale, mix_wt)
        # just one overall line/class of business?
        self.scalar_business = \
            np.all(list(map(lambda x: len(x) == 1, [el, premium, lr, self.en, self.attachment, self.limit])))

        # holder for the severity distributions
        self.sevs = np.empty(el.shape, dtype=type(Severity))
        el = np.where(el > 0, el, premium * lr)
        # compute the grand total for approximations
        # overall freq CV with common mixing TODO this is dubious
        c = freq_a
        root_c = np.sqrt(freq_a)
        # counter
        r = 0
        for _el, _pr, _lr, _en, _at, _y, sn, sa, sb, sm, scv, sloc, ssc, smix in \
                zip(el, premium, lr, self.en, self.attachment, self.limit, sev_name, sev_a, sev_b, sev_mean,
                    sev_cv, sev_loc, sev_scale, mix_wt):

            self.sevs[r] = Severity(sn, _at, _y, sm, scv, sa, sb, sloc, ssc, sev_xs, sev_ps, True)
            sev1, sev2, sev3 = self.sevs[r].moms()

            if _el > 0:
                _en = _el / sev1
            elif _en > 0:
                _el = _en * sev1
            if _pr > 0:
                _lr = _el / _pr
            elif _lr > 0:
                _pr = _el / _lr

            # scale for the mix
            _pr *= smix
            _el *= smix
            _lr *= smix
            _en *= smix

            # accumulate moments
            ma.add_fs(_en, sev1, sev2, sev3)

            # store
            label = f'{r}: {_y} / {_at} n={_en}'
            self.stats.loc[label, :] = [_y, _at, scv, _el, _pr, _lr] + ma.get_fsa_stats(total=False) + [c, freq_a]
            r += 1

        # average limit and attachment
        avg_limit = np.sum(self.stats.limit * self.stats.freq_1) / ma.tot_freq_1
        avg_attach = np.sum(self.stats.attachment * self.stats.freq_1) / ma.tot_freq_1
        # assert np.allclose(ma.freq_1, self.stats.en)

        # store answer for total
        self.stats_total.loc[self.name, :] = \
            [avg_limit, avg_attach, 0, self.stats.el.sum(), self.stats.prem.sum(),
             self.stats.el.sum() / self.stats.prem.sum()] + ma.get_fsa_stats(total=True, remix=True) + [c, root_c]
        self.stats_total.loc[f'{self.name} independent freq', :] = \
            [avg_limit, avg_attach, 0, self.stats.el.sum(), self.stats.prem.sum(),
             self.stats.el.sum() / self.stats.prem.sum()] + ma.get_fsa_stats(total=True, remix=False) + [c, root_c]
        self.stats['wt'] = self.stats.freq_1 / ma.tot_freq_1
        self.stats_total['wt'] = self.stats.wt.sum()  # better equal 1.0!
        self.n = ma.tot_freq_1
        self.statistics_df = pd.concat((self.stats, self.stats_total))

        # finally, need a report series for Portfolio to consolidate
        self.report = ma.stats_series(self.name, np.max(self.limit), 0.999, total=True)
        # TODO fill in missing p99                                   ^ pctile

    def __str__(self):
        """
        Goal: readability

        :return:
        """
        # pull out agg stats
        ags = self.stats_total.loc[self.name, :]
        s = f"Aggregate: {self.name}\n\tEN={ags['freq_1']}, CV(N)={ags['freq_cv']:5.3f}\n\t" \
            f"{len(self.sevs)} severities, EX={ags['sev_1']:,.1f}, CV(X)={ags['sev_cv']:5.3f}\n\t" \
            f"EA={ags['agg_1']:,.1f}, CV={ags['agg_cv']:5.3f}"
        return s

    def __repr__(self):
        """
        Goal unmbiguous
        :return: MUST return a string
        """
        return str(self.spec)

    def discretize(self, method, approx_calc='survival'):
        """


        :param method:  continuous or discrete or raw (for...)
        :param approx_calc:  survival, distribution or both
        :return:
        """

        if method == 'continuous':
            adj_xs = np.hstack((self.xs, np.inf))
        elif method == 'discrete':
            adj_xs = np.hstack((self.xs - self.bs / 2, np.inf))
        elif method == 'raw':
            adj_xs = self.xs
        else:
            raise ValueError(
                f'Invalid parameter {method} passed to double_diff; options are raw, discrete or histogram')

        # bed = bucketed empirical distribution
        beds = []
        for fz in self.sevs:
            if approx_calc == 'both':
                beds.append(np.maximum(np.diff(fz.cdf(adj_xs)), -np.diff(fz.sf(adj_xs))))
            elif approx_calc == 'survival':
                beds.append(-np.diff(fz.sf(adj_xs)))
            elif approx_calc == 'distribution':
                beds.append(np.diff(fz.cdf(adj_xs)))
            else:
                raise ValueError(f'Invalid options {approx_calc} to double_diff; options are density, survival or both')

        return beds

    def density(self, xs, padding=1, tilt_vector=None, approximation='exact', sev_calc='discrete',
                discretization_calc='survival', force_severity=False, verbose=False):
        """
        Compute the density
        :param xs:
        :param padding:
        :param tilt_vector:
        :param approximation:
        :param sev_calc:   discrete = suitable for fft, continuous = for rv_histogram cts version
        :param discretization_calc: use survival, distribution or both (=max(cdf, sf)) which is most accurate calc
        :param force_severity: make severities even if using approximation, for plotting
        :param verbose: make partial plots and return details of all moments
        :return:
        """

        if verbose:
            axm = axiter_factory(None, len(self.sevs) + 2, aspect=1.66)
            df = pd.DataFrame(
                columns=['n', 'limit', 'attachment', 'en', 'emp ex1', 'emp cv', 'sum p_i', 'wt', 'nans', 'max', 'wtmax',
                         'min'])
            df = df.set_index('n')
            dm = iter(range(300000))
            audit = None
        else:
            axm = None
            df = None
            dm = 0
            audit = None

        aa = al = 0
        self.xs = xs
        self.bs = xs[1]

        # make the severity vector
        # case 1 (all for now) it is the claim count weighted average of the severities
        if approximation == 'exact' or force_severity:
            wts = self.stats.freq_1 / self.stats.freq_1.sum()
            self.sev_density = np.zeros_like(xs)
            beds = self.discretize(sev_calc, discretization_calc)
            for temp, w, a, l, n in zip(beds, wts, self.attachment, self.limit, self.en):
                self.sev_density += temp * w
                if verbose:
                    _m = np.sum(self.xs * temp)
                    _cv = np.sqrt(np.sum((self.xs ** 2) * temp) - (_m ** 2)) / _m
                    df.loc[next(dm), :] = [l, a, n, _m, _cv,
                                           temp.sum(),
                                           w, np.sum(np.where(np.isinf(temp), 1, 0)),
                                           temp.max(), w * temp.max(), temp.min()]
                    next(axm).plot(xs, temp, label='compt', lw=0.5, drawstyle='steps-post')
                    axm.ax.plot(xs, self.sev_density, label='run tot', lw=0.5, drawstyle='steps-post')
                    if np.all(self.limit < np.inf):
                        axm.ax.set(xlim=(0, np.max(self.limit) * 1.1), title=f'{l:,.0f} xs {a:,.0f}\twt={w:.2f}')
                    else:
                        axm.ax.set(title=f'{l:,.0f} xs {a:,.0f}\twt={w:.2f}')
                    axm.ax.legend()
            if verbose:
                next(axm).plot(xs, self.sev_density, lw=0.5, drawstyle='steps-post')
                axm.ax.set_title('occ')
                aa = float(np.sum(df.attachment * df.wt))
                al = float(np.sum(df.limit * df.wt))
                if np.all(self.limit < np.inf):
                    axm.ax.set_xlim(0, np.max(self.limit))
                else:
                    axm.ax.set_xlim(0, xs[-1])
                _m = np.sum(self.xs * np.nan_to_num(self.sev_density))
                _cv = np.sqrt(np.sum(self.xs ** 2 * np.nan_to_num(self.sev_density)) - _m ** 2) / _m
                df.loc["Occ", :] = [al, aa, self.n, _m, _cv,
                                    self.sev_density.sum(),
                                    np.sum(wts), np.sum(np.where(np.isinf(self.sev_density), 1, 0)),
                                    self.sev_density.max(), np.nan, self.sev_density.min()]
        if force_severity:
            return
        if approximation == 'exact':
            if self.freq_name == 'poisson':
                # TODO ignoring contagion! Where are other freq dists!!!
                # convolve for compound Poisson
                # TODO put back!
                if self.n > 100:
                    logging.warning(f' | warning, {self.n} very high claim count ')
                # assert self.n < 100
                self.ftagg_density = np.exp(self.n * (ft(self.sev_density, padding, tilt_vector) - 1))
                self.agg_density = np.real(ift(self.ftagg_density, padding, tilt_vector))
            elif self.freq_name == 'fixed':
                # fixed count distribution...still need to do convolution
                self.ftagg_density = ft(self.sev_density, padding, tilt_vector) ** self.n
                if self.n == 1:
                    self.agg_density = self.sev_density
                else:
                    self.agg_density = np.real(ift(self.ftagg_density, padding, tilt_vector))
            elif self.freq_name == 'bernoulli':
                # binomial M_N(t) = p M_X(t) + (1-p) at zero point
                assert ((self.n > 0) and (self.n < 1))
                self.ftagg_density = self.n * ft(self.sev_density, padding, tilt_vector)
                self.ftagg_density += (1 - self.n) * np.ones_like(self.ftagg_density)
                self.agg_density = np.real(ift(self.ftagg_density, padding, tilt_vector))
            else:
                raise ValueError(f'Inadmissible value for fixed {self.freq_name}'
                                 ' Allowable values are -1 (or bernoulli) 1 (or fixed), missing or 0 (Poisson)')
        else:
            if approximation == 'slognorm':
                shift, mu, sigma = sln_fit(self.aggm, self.aggcv, self.aggskew)
                self.fzapprox = ss.lognorm(sigma, scale=np.exp(mu), loc=shift)

            elif approximation == 'sgamma':
                shift, alpha, theta = sgamma_fit(self.aggm, self.aggcv, self.aggskew)
                self.fzapprox = ss.gamma(alpha, scale=theta, loc=shift)
            else:
                raise ValueError(f'Invalid approximation {approximation} option passed to CAgg density. '
                                 'Allowable options are: exact | slogorm | sgamma')

            ps = self.fzapprox.pdf(xs)
            self.agg_density = ps / np.sum(ps)
            self.ftagg_density = ft(self.agg_density, padding, tilt_vector)

        if verbose:
            ax = next(axm)
            ax.plot(xs, self.agg_density, 'b')
            ax.set(xlim=(0, self.aggm * (1 + 5 * self.aggcv)), title='aggregate')
            suptitle_and_tight(f'{self.name} severity audit')
            _m = np.sum(self.xs * np.nan_to_num(self.agg_density))
            _cv = np.sqrt(np.sum(self.xs ** 2 * np.nan_to_num(self.agg_density)) - _m ** 2) / _m
            df.loc['Agg', :] = [al, aa, self.n, _m, _cv,
                                self.agg_density.sum(),
                                np.nan, np.sum(np.where(np.isinf(self.agg_density), 1, 0)),
                                self.agg_density.max(), np.nan, self.agg_density.min()]
            audit = pd.concat((df[['limit', 'attachment', 'emp ex1', 'emp cv']],
                               pd.concat((self.stats, self.stats_total))[[
                                   'en', 'sev_1', 'sev_cv']]), axis=1)
            audit.iloc[-1, -1] = self.stats_total.loc['Agg independent', 'agg_cv']
            audit.iloc[-1, -2] = self.stats_total.loc['Agg independent', 'agg_1']
            audit['abs sev err'] = audit.sev_1 - audit['emp ex1']
            audit['rel sev err'] = audit['abs sev err'] / audit['emp ex1'] - 1

        return df, audit

    def emp_stats(self):
        """
        report on empirical stats

        :return:
        """

        ex = np.sum(self.xs * self.agg_density)
        ex2 = np.sum(self.xs ** 2 * self.agg_density)
        # ex3 = np.sum(self.xs**3 * self.agg_density)
        v = ex2 - ex * ex
        sd = np.sqrt(v)
        cv = sd / ex
        s1 = pd.Series([ex, sd, cv], index=['mean', 'sd', 'cv'])
        if self.dh_sev_density is not None:
            ex = np.sum(self.xs * self.dh_agg_density)
            ex2 = np.sum(self.xs ** 2 * self.dh_agg_density)
            # ex3 = np.sum(self.xs**3 * self.dh_agg_density)
            v = ex2 - ex * ex
            sd = np.sqrt(v)
            cv = sd / ex
            s2 = pd.Series([ex, sd, cv], index=['mean', 'sd', 'cv'])
            df = pd.DataFrame({'numeric': s1, self.beta_name: s2})
        else:
            df = pd.DataFrame(s1, columns=['numeric'])
        df.loc['mean', 'theory'] = self.stats_total.loc['Agg', 'agg1']
        df.loc['sd', 'theory'] = np.sqrt(self.stats_total.loc['Agg', 'agg2'] -
                                         self.stats_total.loc['Agg', 'agg1'] ** 2)
        df.loc['cv', 'theory'] = self.stats_total.loc['Agg', 'aggcv']  # report[('agg', 'cv')]
        df['err'] = df['numeric'] / df['theory'] - 1
        return df

    def delbaen_haezendonck_density(self, xs, padding, tilt_vector, beta, beta_name=""):
        """
        Compare the base and Delbaen Haezendonck transformed aggregates

        * beta(x) = alpha + gamma(x)
        * alpha = log(freq' / freq): log of the increase in claim count
        * gamma = log(RND of adjusted severity) = log(tilde f / f)

        Adjustment guarantees a positive loading iff beta is an increasing function
        iff gamma is increasing iff tilde f / f is increasing.
        cf. eqn 3.7 and 3.8

        Note conditions that E(exp(beta(X)) and E(X exp(beta(X)) must both be finite (3.4, 3.5)
        form of beta function described in 2.23 via, 2.16-17 and 2.18

        From examples on last page of paper:

            beta(x) = a ==> adjust frequency by factor of e^a
            beta(x) = log(1 + b(x - E(X)))  ==> variance principle EN(EX + bVar(X))
            beta(x) = ax- logE_P(exp(a x))  ==> Esscher principle

        :param xs:
        :param padding:
        :param tilt_vector:
        :param beta: function R+ to R with appropriate properties or name of prob distortion function
        :param beta_name:
        :return:
        """
        if self.agg_density is None:
            # update
            self.density(xs, padding, tilt_vector, 'exact')
        if isinstance(beta, Distortion):
            # passed in a distortion function
            beta_name = beta.name
            self.dh_sev_density = np.diff(beta.g(np.cumsum(np.hstack((0, self.sev_density)))))
            # expect ex_beta = 1 but allow to pass multiples....
        else:
            self.dh_sev_density = self.sev_density * np.exp(beta.g(xs))
        ex_beta = np.sum(self.dh_sev_density)
        self.dh_sev_density = self.dh_sev_density / ex_beta
        adj_n = ex_beta * self.n
        if self.freq_name == 'poisson':
            # convolve for compound Poisson
            ftagg_density = np.exp(adj_n * (ft(self.dh_sev_density, padding, tilt_vector) - 1))
            self.dh_agg_density = np.real(ift(ftagg_density, padding, tilt_vector))
        else:
            raise ValueError('Must use compound Poisson for DH density')
        self.beta_name = beta_name

    def plot(self, kind='long', axiter=None, aspect=1, figsize=None):
        """
        make a quick plot of fz: computed density and aggregate

        :param kind:
        :param aspect:
        :param axiter:
        :param figsize:
       :return:
        """

        if self.agg_density is None:
            print('Cannot plot before update')
            return

        if kind == 'long':

            set_tight = (axiter is None)
            axiter = axiter_factory(axiter, 10, aspect=aspect, figsize=figsize)

            max_lim = min(self.xs[-1], np.max(self.limit)) * 1.05

            next(axiter).plot(self.xs, self.sev_density, drawstyle='steps-post')
            axiter.ax.set(title='Severity', xlim=(0, max_lim))

            next(axiter).plot(self.xs, self.sev_density)
            axiter.ax.set(title='Log Severity', yscale='log', xlim=(0, max_lim))

            next(axiter).plot(self.xs, self.sev_density.cumsum(), drawstyle='steps-post')
            axiter.ax.set(title='Severity Distribution', xlim=(0, max_lim))

            next(axiter).plot(self.xs, self.agg_density, label='aggregate')
            axiter.ax.plot(self.xs, self.sev_density, lw=0.5, drawstyle='steps-post', label='severity')
            axiter.ax.set(title='Aggregate')
            axiter.ax.legend()

            next(axiter).plot(self.xs, self.agg_density, label='aggregate')
            axiter.ax.set(title='Aggregate')

            next(axiter).plot(self.xs, self.agg_density, label='aggregate')
            axiter.ax.set(yscale='log', title='Aggregate, log scale')

            F = self.agg_density.cumsum()
            next(axiter).plot(self.xs, 1 - F)
            axiter.ax.set(title='Survival Function')

            next(axiter).plot(self.xs, 1 - F)
            axiter.ax.set(title='Survival Function, log scale', yscale='log')

            next(axiter).plot(1 - F, self.xs, label='aggregate')
            axiter.ax.plot(1 - self.sev_density.cumsum(), self.xs, label='severity')
            axiter.ax.set(title='Lee Diagram')
            axiter.ax.legend()

            # figure for extended plotting of return period:
            maxp = F[-1]
            if maxp > 0.9999:
                _n = 10
            else:
                _n = 5
            if maxp >= 1:
                maxp = 1 - 1e-10
            k = (maxp / 0.99) ** (1 / _n)
            extraps = 0.99 * k ** np.arange(_n)
            q = interpolate.interp1d(F, self.xs, kind='linear', fill_value=0, bounds_error=False)
            ps = np.hstack((np.linspace(0, 1, 100, endpoint=False), extraps))
            qs = q(ps)
            next(axiter).plot(1 / (1 - ps), qs)
            axiter.ax.set(title='Return Period', xscale='log')

            if set_tight:
                suptitle_and_tight(f'{self.name} Distributions')
        else:  # kind == 'quick':
            if self.dh_agg_density is not None:
                n = 4
            else:
                n = 3

            set_tight = (axiter is None)
            axiter = axiter_factory(axiter, n, figsize, aspect=aspect)

            F = np.cumsum(self.agg_density)
            mx = np.argmax(F > 1 - 1e-5)
            if mx == 0:
                mx = len(F) + 1
            dh_F = None
            if self.dh_agg_density is not None:
                dh_F = np.cumsum(self.dh_agg_density)
                mx = max(mx, np.argmax(dh_F > 1 - 1e-5))
                dh_F = dh_F[:mx]
            F = F[:mx]

            if self.sev_density is None:
                self.density(self.xs, 1, None, sev_calc='rescale', force_severity=True)
            xs = self.xs[:mx]
            d = self.agg_density[:mx]
            sevF = np.cumsum(self.sev_density)
            sevF = sevF[:mx]
            f = self.sev_density[:mx]

            ax = next(axiter)
            ax.plot(xs, d, label='agg')
            ax.plot(xs, f, label='sev')
            if self.dh_agg_density is not None:
                ax.plot(xs, self.dh_agg_density[:mx], label='dh {:} agg'.format(self.beta_name))
                ax.plot(xs, self.dh_sev_density[:mx], label='dh {:} sev'.format(self.beta_name))
            ax.set_ylim(0, min(2 * np.max(d), np.max(f[1:])))
            ax.legend()
            ax.set_title('Density')
            ax = next(axiter)
            ax.plot(xs, d, label='agg')
            ax.plot(xs, f, label='sev')
            if self.dh_agg_density is not None:
                ax.plot(xs, self.dh_agg_density[:mx], label='dh {:} agg'.format(self.beta_name))
                ax.plot(xs, self.dh_sev_density[:mx], label='dh {:} sev'.format(self.beta_name))
            ax.set_yscale('log')
            ax.legend()
            ax.set_title('Log Density')

            ax = next(axiter)
            ax.plot(F, xs, label='Agg')
            ax.plot(sevF, xs, label='Sev')
            if self.dh_agg_density is not None:
                dh_F = np.cumsum(self.dh_agg_density[:mx])
                ax.plot(dh_F, xs, label='dh {:} agg'.format(self.beta_name))
            ax.legend()
            ax.set_title('Lee Diagram')

            if self.dh_agg_density is not None:
                # if dh computed graph comparision
                ax = next(axiter)
                ax.plot(1 - F, 1 - dh_F, label='g(S) vs S')
                ax.plot(1 - F, 1 - F, 'k', linewidth=.5, label=None)
            if set_tight:
                plt.tight_layout()

    def recommend_bucket(self, N=10):
        """
        recommend a bucket size given 2**N buckets

        :param N:
        :return:
        """
        moment_est = estimate_agg_percentile(self.aggm, self.aggcv, self.aggskew) / N
        limit_est = self.limit.max() / N
        if limit_est == np.inf:
            limit_est = 0
        logging.info(f'Agg.recommend_bucket | {self.name} moment: {moment_est}, limit {limit_est}')
        return max(moment_est, limit_est)


class Severity(ss.rv_continuous):
    """

    A continuous random variable, subclasses ``scipy.stats.rv_continuous``.

    adds layer and attachment to scipy stats continuous random variable class
    overrides

    * cdf
    * pdf
    * isf
    * ppf
    * moments

    Should consider over-riding: sf, **stats** ?munp



    """

    def __init__(self, name, attachment=0, limit=np.inf, mean=0, cv=0, a=0, b=0, loc=0, scale=0, hxs=None,
                 hps=None, conditional=True):
        """

        :param name: scipy stats continuous distribution | (c|d)histogram  cts or discerte | fixed
        :param attachment:
        :param limit:
        :param mean:
        :param cv:
        :param a:
        :param b:
        :param loc:
        :param scale:
        :param hxs: for fixed or histogram classes
        :param hps:
        :param conditional: conditional or unconditional; for severities use conditional
        """

        ss.rv_continuous.__init__(self, name=f'{name}[{limit} xs {attachment:,.0f}]')
        self.limit = limit
        self.attachment = attachment
        self.detachment = limit + attachment
        self.fz = None
        self.pattach = 0
        self.pdetach = 0
        self.conditional = conditional
        self.name = name
        self.sev1 = self.sev2 = self.sev3 = None

        # there are two types: if hxs and hps provided then fixed/histogram, else scpiy dist
        # allows you to define fixed with just xs=1 (no p)
        if hxs is not None:
            if name == 'fixed':
                # fixed is a special case of dhistogram with just one point
                name = 'dhistogram'
                hps = np.array(1)
            assert name[1:] == 'histogram'
            # TODO: make histogram work with limit and attachment; currently they are ignored
            xs, ps = np.broadcast_arrays(np.array(hxs), np.array(hps))
            if not np.isclose(np.sum(ps), 1.0):
                logging.error(f'Severity.init | {name} histogram/fixed severity with probs do not sum to 1, '
                              f'{np.sum(ps)}')
            # need to limit distribution
            limit = min(np.min(limit), xs.max())
            if name == 'chistogram':
                # continuous histogram: uniform between xs's
                xss = np.sort(np.hstack((xs, xs[-1] + xs[1])))
                # midpoints
                xsm = (xss[:-1] + xss[1:]) / 2
                self.sev1 = np.sum(xsm * ps)
                self.sev2 = np.sum(xsm ** 2 * ps)
                self.sev3 = np.sum(xsm ** 3 * ps)
                self.fz = ss.rv_histogram((ps, xss))
            elif name == 'dhistogram':
                # discrete histogram: point masses at xs's
                self.sev1 = np.sum(xs * ps)
                self.sev2 = np.sum(xs ** 2 * ps)
                self.sev3 = np.sum(xs ** 3 * ps)
                xss = np.sort(np.hstack((xs - 1e-5, xs)))  # was + but F(x) = Pr(X<=x) so seems shd be to left
                pss = np.vstack((ps, np.zeros_like(ps))).reshape((-1,), order='F')[:-1]
                self.fz = ss.rv_histogram((pss, xss))
            else:
                raise ValueError('Histogram must be chistogram (continuous) or dhistogram (discrete)'
                                 f', you passed {name}')

        elif name in ['norm', 'expon']:
            # distributions with no shape parameters
            #     Normal (and possibly others) does not have a shape parameter
            if loc == 0 and mean > 0:
                loc = mean
            if scale == 0 and cv > 0:
                scale = cv * loc
            gen = getattr(ss, name)
            self.fz = gen(loc=loc, scale=scale)

        elif name in ['beta']:
            # distributions with two shape parameters
            # require specific inputs
            gen = getattr(ss, name)
            self.fz = gen(a, b, loc=loc, scale=scale)
        else:
            # distributions with one shape parameter
            if a == 0:  # TODO figuring 0 is invalid shape...
                a, _ = cv_to_shape(name, cv)
            if scale == 0 and mean > 0:
                scale, self.fz = mean_to_scale(name, a, mean)
            else:
                gen = getattr(ss, name)
                self.fz = gen(a, scale=scale, loc=loc)

        if self.detachment == np.inf:
            self.pdetach = 0
        else:
            self.pdetach = self.fz.sf(self.detachment)

        if self.attachment == 0:
            self.pattach = 1
        else:
            self.pattach = self.fz.sf(self.attachment)

        if mean > 0 or cv > 0:
            # if you input a mean or cv check we are close to target
            st = self.fz.stats('mv')
            m = st[0]
            acv = st[1] ** .5 / m  # achieved cv
            if mean > 0:
                assert (np.isclose(mean, m))
            if cv > 0:
                assert (np.isclose(cv, acv))
            # print('ACHIEVED', mean, cv, m, acv, self.fz.stats(), self._stats())
            logging.info(f'Severity.__init__ | parameters {a}, {scale}: target/actual {mean} vs {m};  {cv} vs {acv}')

        lim_name = f'{limit:,.0f}' if limit != np.inf else "Unlimited"
        try:
            self.long_name = f'{name}({self.fz.args[0]:.2f})[{lim_name} xs {attachment:,.0f}]'
        except:
            # 'rv_histogram' object has no attribute 'args'
            self.long_name = f'{name}[{lim_name} xs {attachment:,.0f}]'

        assert self.fz is not None

    def __enter__(self):
        """ Support with Severity as f: """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def _pdf(self, x, *args):
        if self.conditional:
            return np.where(x > self.limit, 0,
                            np.where(x == self.limit, np.inf if self.pdetach > 0 else 0,
                                     self.fz.pdf(x + self.attachment) / self.pattach))
        else:
            if self.pattach < 1:
                return np.where(x < 0, 0,
                                np.where(x == 0, np.inf,
                                         np.where(x == self.detachment, np.inf,
                                                  np.where(x > self.detachment, 0,
                                                           self.fz.pdf(x + self.attachment, *args)))))
            else:
                return np.where(x < 0, 0,
                                np.where(x == self.detachment, np.inf,
                                         np.where(x > self.detachment, 0,
                                                  self.fz.pdf(x + self.attachment, *args))))

    def _cdf(self, x, *args):
        if self.conditional:
            return np.where(x > self.limit, 1,
                            np.where(x < 0, 0,
                                     (self.fz.cdf(x + self.attachment) - (1 - self.pattach)) / self.pattach))
        else:
            return np.where(x < 0, 0,
                            np.where(x == 0, 1 - self.pattach,
                                     np.where(x > self.limit, 1,
                                              self.fz.cdf(x + self.attachment, *args))))

    def _sf(self, x, *args):
        if self.conditional:
            return np.where(x > self.limit, 0,
                            np.where(x < 0, 1,
                                     self.fz.sf(x + self.attachment, *args) / self.pattach))
        else:
            return np.where(x < 0, 1,
                            np.where(x == 0, self.pattach,
                                     np.where(x > self.limit, 0,
                                              self.fz.sf(x + self.attachment, *args))))

    def _isf(self, q, *args):
        if self.conditional:
            return np.where(q < self.pdetach / self.pattach, self.limit,
                            self.fz.isf(q * self.pattach) - self.attachment)
        else:
            return np.where(q >= self.pattach, 0,
                            np.where(q < self.pdetach, self.limit,
                                     self.fz.isf(q, *args) - self.attachment))

    def _ppf(self, q, *args):
        if self.conditional:
            return np.where(q > 1 - self.pdetach / self.pattach, self.limit,
                            self.fz.ppf(1 - self.pattach * (1 - q)) - self.attachment)
        else:
            return np.where(q <= 1 - self.pattach, 0,
                            np.where(q > 1 - self.pdetach, self.limit,
                                     self.fz.ppf(q, *args) - self.attachment))

    def _stats(self, *args, **kwds):
        ex1, ex2, ex3 = self.moms()
        var = ex2 - ex1 ** 2
        skew = (ex3 - 3 * ex1 * ex2 + 2 * ex1 ** 3) / var ** 1.5
        return np.array([ex1, var, skew, np.nan])

    def _munp(self, n, *args):
        print('wow, called munp')
        pass

    def moms(self):
        """
        moms method
        remember have to use integral of survival function otherwise quad can fail to "see" the distribution
        for unlimited integral of a low sd variable

        :return:
        """

        if self.sev1 is not None:
            # precomputed (fixed and (c|d)histogram classes
            # TODO ignores layer and attach...
            return self.sev1, self.sev2, self.sev3

        ex1 = quad(lambda x: self.fz.sf(x), self.attachment, self.detachment)
        ex2 = quad(lambda x: 2 * (x - self.attachment) * self.fz.sf(x), self.attachment, self.detachment)
        ex3 = quad(lambda x: 3 * (x - self.attachment) ** 2 * self.fz.sf(x), self.attachment, self.detachment,
                   limit=100, full_output=False)

        # quad returns abs error
        eps = 1e-5
        if not ((ex1[1] / ex1[0] < eps or ex1[1] < 1e-4) and
                (ex2[1] / ex2[0] < eps or ex2[1] < 1e-4) and
                (ex3[1] / ex3[0] < eps or ex3[1] < 1e-6)):
            logging.info(f'Severity.moms | **DOUBTFUL** convergence of integrals, abs errs '
                         f'\t{ex1[1]}\t{ex2[1]}\t{ex3[1]} \trel errors \t{ex1[1]/ex1[0]}\t{ex2[1]/ex2[0]}\t'
                         f'{ex3[1]/ex3[0]}')
            # raise ValueError(f' Severity.moms | doubtful convergence of integrals, abs errs '
            #                  f'{ex1[1]}, {ex2[1]}, {ex3[1]} rel errors {ex1[1]/ex1[0]}, {ex2[1]/ex2[0]}, '
            #                  f'{ex3[1]/ex3[0]}')
        # logging.info(f'Severity.moms | GOOD convergence of integrals, abs errs '
        #                      f'\t{ex1[1]}\t{ex2[1]}\t{ex3[1]} \trel errors \
        #  t{ex1[1]/ex1[0]}\t{ex2[1]/ex2[0]}\t{ex3[1]/ex3[0]}')

        ex1 = ex1[0]
        ex2 = ex2[0]
        ex3 = ex3[0]

        if self.conditional:
            ex1 /= self.pattach
            ex2 /= self.pattach
            ex3 /= self.pattach

        return ex1, ex2, ex3

    def plot(self, N=100):
        """
        quick plot

        :param N:
        :return:
        """

        ps = np.linspace(0, 1, N, endpoint=False)
        xs = np.linspace(0, self._isf(1e-4), N)

        it = axiter_factory(None, 4)

        ax = next(it)
        ys = self._pdf(xs)
        ax.plot(xs, ys)
        ax.grid(which='major', axis='both', linestyle='-', linewidth='0.1', color='blue', alpha=0.5)
        ax.set_title('PDF')

        ys = self._cdf(xs)
        ax = next(it)
        ax.plot(xs, ys, drawstyle='steps-post', lw=1)
        ax.grid(which='major', axis='both', linestyle='-', linewidth='0.1', color='blue', alpha=0.5)
        ax.set(title='CDF', ylim=(0, 1))

        ys = self._isf(ps)
        ax = next(it)
        ax.plot(ps, ys, drawstyle='steps-post', lw=1)
        ax.grid(which='major', axis='both', linestyle='-', linewidth='0.1', color='blue', alpha=0.5)
        ax.set(title='ISF', xlim=(0, 1))

        ax = next(it)
        ax.plot(1 - ps, ys, drawstyle='steps-post', lw=1)
        ax.grid(which='major', axis='both', linestyle='-', linewidth='0.1', color='blue', alpha=0.5)
        ax.set(title='Lee diagram', xlim=(0, 1))

        suptitle_and_tight(self.long_name)
