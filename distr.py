import scipy.stats as ss
import numpy as np
from aggregate.utils import cv_to_shape, mean_to_scale
from scipy.integrate import quad
import pandas as pd
import collections
import matplotlib.pyplot as plt
import logging
from .utils import cumulate_moments, moments_to_mcvsk, sln_fit, sgamma_fit, ft, ift, \
    axiter_factory, estimate_agg_percentile, suptitle_and_tight, stats_series
from . spectral import Distortion


# from scipy import interpolate
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
                 sev_mean=0, sev_cv=0, sev_loc=0, sev_scale=0, mix_wt=1, freq_name='', freq_a=0, freq_b=0):
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

        TODO: later do both, for now assume one or other is a scalar
        call with both that does the cross product... (note the sev array may not be equal sizes)

        :param name:
        :param el:   expected loss or vector or matrix
        :param premium:
        :param lr:  loss ratio
        :param en:  expected claim count
        :param attachment: occ attachment
        :param limit: occ limit
        :param sev_name: Severity class object or similar or vector or matrix
        :param sev_a:
        :param sev_b:
        :param sev_mean:
        :param sev_cv:
        :param sev_loc:
        :param sev_scale:
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
                         mix_wt=mix_wt, freq_name=freq_name, freq_a=freq_a, freq_b=freq_b)

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
        self.stats = pd.DataFrame(columns=['limit', 'attachment', 'sevcv_param', 'el', 'prem', 'lr',
                                           'en', 'freqcv', 'freqskew', 'sev1', 'sev2', 'sev3', 'sevcv', 'sevskew',
                                           'agg1', 'agg2', 'agg3', 'aggcv', 'aggskew', 'contagion', 'mix_cv'])
        self.stats_total = self.stats.copy()

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
        r = 0
        # accumulators for total moments
        agg_tot1 = agg_tot2 = agg_tot3 = freq_tot1 = freq_tot2 = freq_tot3 = 0
        sev_tot1 = sev_tot2 = sev_tot3 = c = 0
        for _el, _pr, _lr, _en, _at, _y, sn, sa, sb, sm, scv, sloc, ssc, smix in \
                zip(el, premium, lr, self.en, self.attachment, self.limit, sev_name, sev_a, sev_b, sev_mean,
                    sev_cv, sev_loc, sev_scale, mix_wt):

            self.sevs[r] = Severity(sn, _at, _y, sm, scv, sa, sb, sloc, ssc)
            sev1, sev2, sev3 = self.sevs[r].moms()

            m, scv, ssk = moments_to_mcvsk(sev1, sev2, sev3)
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

            # freq moments (old style for now) c = contagion
            c = freq_a * freq_a
            freq1 = _en
            if freq_name == 'fixed':
                # fixed distribution N=self.n certainly
                freq2 = freq1 ** 2
                freq3 = freq1 * freq2
            elif freq_name == 'bernoulli':
                # code for bernoulli self.n, E(N^k) = E(N) = self.n
                freq2 = _en
                freq3 = _en
            elif c == 0:
                # Poisson
                freq2 = freq1 * (1 + freq1)
                freq3 = freq1 * (1 + freq1 * (3 + freq1))
            else:
                # for gamma alpha, k with density x^alpha e^-kx, EX^n = Gamma(alpha + n) / Gamma(n) k^-n
                # EX = a/k = 1, so a=k
                # EX2 = (a+1)a/k^2 = a^2/k^2 + a/k^2 = (EX)^2 + a/k^2 = 1 + 1/k, hence var = a/k^2
                # if EX=1 and var = c then var = a/k/k = 1/k = c, so k = 1/c
                # then a = 1/c
                # Finally EX3 = (a+2)(a+1)a/k^3 = (c+1)(c+2)
                # Iman Conover paper page 14
                freq2 = freq1 * (1 + freq1 * (1 + c))  # note 1+c = E(G^2)
                freq3 = freq1 * (1 + freq1 * (3 * (1 + c) + freq1 * (1 + c) * (1 + 2 * c)))

            # raw moments of aggregate, not central moments
            agg1 = freq1 * sev1
            agg2 = freq1 * sev2 + (freq2 - freq1) * sev1 ** 2
            agg3 = freq1 * sev3 + freq3 * sev1 ** 3 + 3 * (freq2 - freq1) * sev1 * sev2 + (
                    - 3 * freq2 + 2 * freq1) * sev1 ** 3
            am, acv, ask = moments_to_mcvsk(agg1, agg2, agg3)
            fm, fcv, fsk = moments_to_mcvsk(freq1, freq2, freq3)
            # TODO make first scv the theoretic input value not the computed value
            self.stats.loc[r, :] = [_y, _at, scv, _el, _pr, _lr, _en, fcv, fsk, sev1, sev2, sev3, scv, ssk, agg1, agg2,
                                    agg3,
                                    acv, ask, c, freq_a]
            agg_tot1, agg_tot2, agg_tot3 = cumulate_moments(agg_tot1, agg_tot2, agg_tot3, agg1, agg2, agg3)
            freq_tot1, freq_tot2, freq_tot3 = cumulate_moments(freq_tot1, freq_tot2, freq_tot3, freq1, freq2, freq3)
            sev_tot1 += freq1 * sev1
            sev_tot2 += freq1 * sev2
            sev_tot3 += freq1 * sev3
            r += 1

        # compute the grand total for approximations
        sev_tot1 /= freq_tot1
        sev_tot2 /= freq_tot1
        sev_tot3 /= freq_tot1
        # freq_tot2 and freq_tot3 need to reflect the same mixing; compute impact of mixing:
        _, freq_cvi, freq_ski = moments_to_mcvsk(freq_tot1, freq_tot2, freq_tot3)
        freq_tot2 = freq_tot1 * (1 + freq_tot1 * (1 + c))  # note 1+c = E(G^2)
        freq_tot3 = freq_tot1 * (1 + freq_tot1 * (3 * (1 + c) + freq_tot1 * (1 + c) * (1 + 2 * c)))
        _, freq_cv, freq_sk = moments_to_mcvsk(freq_tot1, freq_tot2, freq_tot3)
        _, _sev_cv, sev_sk = moments_to_mcvsk(sev_tot1, sev_tot2, sev_tot3)
        aggmi, aggcvi, aggskewi = moments_to_mcvsk(agg_tot1, agg_tot2, agg_tot3)

        agg1 = freq_tot1 * sev_tot1
        agg2 = freq_tot1 * sev_tot2 + (freq_tot2 - freq_tot1) * sev_tot1 ** 2
        agg3 = freq_tot1 * sev_tot3 + freq_tot3 * sev_tot1 ** 3 + 3 * (
                freq_tot2 - freq_tot1) * sev_tot1 * sev_tot2 + (
                       - 3 * freq_tot2 + 2 * freq_tot1) * sev_tot1 ** 3
        self.aggm, self.aggcv, self.aggskew = moments_to_mcvsk(agg1, agg2, agg3)

        # overall CV
        c = (freq_tot2 - freq_tot1 ** 2 - freq_tot1) / (freq_tot1 ** 2)
        root_c = np.sqrt(c)
        # average limit and attachment
        avg_limit = np.sum(self.stats.limit * self.stats.en) / freq_tot1
        # assert np.allclose(freq_tot1, self.stats.en)
        avg_attach = np.sum(self.stats.attachment * self.stats.en) / freq_tot1
        self.stats_total.loc['Agg', :] = [avg_limit, avg_attach, 0, self.stats.el.sum(), self.stats.prem.sum(),
                                          self.stats.el.sum() / self.stats.prem.sum(),
                                          freq_tot1, freq_cv, freq_sk, sev_tot1, sev_tot2, sev_tot3, _sev_cv, sev_sk,
                                          agg1, agg2, agg3, self.aggcv, self.aggskew, c, root_c]
        self.stats_total.loc['Agg independent', :] = [avg_limit, avg_attach, 0, self.stats.el.sum(),
                                                      self.stats.prem.sum(),
                                                      self.stats.el.sum() / self.stats.prem.sum(),
                                                      freq_tot1, freq_cvi, freq_ski, sev_tot1, sev_tot2, sev_tot3,
                                                      _sev_cv, sev_sk,
                                                      agg_tot1, agg_tot2, agg_tot3, aggcvi, aggskewi, c, root_c]
        self.stats['wt'] = self.stats['en'] / self.stats['en'].sum()
        self.stats_total['wt'] = np.nan
        self.n = freq_tot1
        # finally, need a report series for Portfolio to consolidate
        self.report = stats_series([self.aggm, self.aggcv, self.aggskew,
                                    freq_tot1, freq_cv, freq_sk,
                                    sev_tot1, _sev_cv, sev_sk,
                                    agg1, agg2, agg3,
                                    freq_tot1, freq_tot2, freq_tot3,
                                    sev_tot1, sev_tot2, sev_tot3, self.limit, np.nan], self.name)
        # TODO fill in missing p99                                            ^ pctile

    def __str__(self):
        """
        Goal: readability

        :return:
        """
        # pull out agg stats
        ags = self.stats_total.loc["Agg", :]
        s = f"Aggregate: {self.name}\n\tEN={self.n}, CV(N)={ags['freqcv']:5.3f}\n\t" \
            f"{len(self.sevs)} severities, EX={ags['sev1']:,.1f}, CV(X)={ags['sevcv']:5.3f}\n\t" \
            f"EA={ags['agg1']:,.1f}, CV={ags['aggcv']:5.3f}"
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
            wts = self.stats['en'] / self.stats.en.sum()
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
                next(axm).plot(xs, self.sev_density)
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
                                   'en', 'sev1', 'sevcv']]), axis=1)
            audit.iloc[-1, -1] = self.stats_total.loc['Agg independent', 'aggcv']
            audit.iloc[-1, -2] = self.stats_total.loc['Agg independent', 'agg1']
            audit['sev err'] = audit.sev1 - audit['emp ex1']

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

    def quick_visual(self, axiter=None, figsize=(9, 3)):
        """
        Plot severity and agg, density, distribution and Lee diagram
        :param axiter:
        :param figsize:
        :return:
        """

        if self.dh_agg_density is not None:
            n = 4
        else:
            n = 3

        set_tight = False
        if axiter is None:
            axiter = axiter_factory(axiter, n, figsize)
            set_tight = True

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

    def plot(self, N=100, p=1e-4, axiter=None):
        """
        make a quick plot of fz
        :param axiter:
        :param N:
        :param p:
        :return:
        """
        print('plot is NYI')

        # # for now just severity
        # if axiter is None:
        #     axiter = make_axes(2, (6, 3))
        # # TODO use interpolation if complex severity, otherwise pass through to self.sevs[0]
        # if len(self.sevs) ==  1:
        #     fzisf = self.sevs[0].isf
        #     fzpdf = self.sevs[0].pdf
        # else:
        #     fzisf = interpolate.interp1d( np.cumsum(self.sev_density), self.xs, kind='linear', assume_sorted=True )
        #     fzpdf = interpolate.interp1d( np.cumsum(self.sev_density), self.xs, kind='linear', assume_sorted=True )
        # x0 = fzisf(1 - p)
        # if x0 < 0.1:
        #     x0 = 0
        # x1 = fzisf(p)
        # xs = np.linspace(x0, x1, N)
        # ps = np.linspace(1 / N, 1, N, endpoint=False)
        # den = self.fz.pdf(xs)
        # qs = self.fz.ppf(ps)
        # # plt.figure()
        # next(axiter).plot(xs, den)
        # next(axiter).plot(ps, qs)
        # plt.tight_layout()
        # for now just severity

        # original
        # if axiter is None:
        #     axiter = AxisManager(2, (6, 3))
        #
        # x0 = self.fz.isf(1 - p)
        # if x0 < 0.1:
        #     x0 = 0
        # x1 = self.fz.isf(p)
        # xs = np.linspace(x0, x1, N)
        # ps = np.linspace(1 / N, 1, N, endpoint=False)
        # den = self.fz.pdf(xs)
        # qs = self.fz.ppf(ps)
        # # plt.figure()
        # next(axiter).plot(xs, den)
        # next(axiter).plot(ps, qs)
        # plt.tight_layout()

    def recommend_bucket(self, N=10):
        """
        recommend a bucket size given 2**N buckets

        :param N:
        :return:
        """
        moment_est = estimate_agg_percentile(self.aggm, self.aggcv, self.aggskew) / N
        limit_est = np.max(self.limit[self.limit < np.inf]) / N
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

    def __init__(self, name, attachment=0, limit=np.inf, mean=0, cv=0, a=0, b=0, loc=0, scale=0, conditional=True):
        """

        :param name:
        :param attachment:
        :param limit:
        :param mean:
        :param cv:
        :param a:
        :param b:
        :param loc:
        :param scale:
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

        gen = getattr(ss, name)
        if name in ['norm']:
            # distributions with no shape parameters
            #     Normal (and possibly others) does not have a shape parameter
            if loc == 0 and mean > 0:
                loc = mean
            if scale == 0 and cv > 0:
                scale = cv * loc
            self.fz = gen(loc=mean, scale=scale)

        elif name in ['beta']:
            # distributions with two shape parameters
            # require specific inputs
            self.fz = gen(a, b, loc=loc, scale=scale)

        else:
            # distributions with one shape parameter
            if a == 0:  # TODO figuring 0 is invalid shape...
                a, _ = cv_to_shape(name, cv)
            if scale == 0 and mean > 0:
                scale, self.fz = mean_to_scale(name, a, mean)
            else:
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

        lim_name = f'{limit:,.0f}' if limit != np.inf else "Unlimited"
        self.long_name = f'{name}({self.fz.args[0]:.2f})[{lim_name} xs {attachment:,.0f}]'

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
                            (self.fz.cdf(x + self.attachment) - (1-self.pattach)) / self.pattach)
        else:
            return np.where(x < 0, 0,
                            np.where(x == 0, 1 - self.pattach,
                                     np.where(x > self.detachment - self.attachment, 1,
                                              self.fz.cdf(x + self.attachment, *args))))

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
                            self.fz.ppf(1 - self.pattach(1 - q)) - self.attachment)
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
        ex1 = quad(lambda x: self.fz.sf(x), self.attachment, self.detachment)
        ex2 = quad(lambda x: 2 * (x - self.attachment) * self.fz.sf(x), self.attachment, self.detachment)
        ex3 = quad(lambda x: 3 * (x - self.attachment) ** 2 * self.fz.sf(x), self.attachment, self.detachment,
                   limit=100,
                   full_output=False)

        # quad returns abs error
        eps = 1e-5
        if not ((ex1[1] / ex1[0] < eps or ex1[1] < 1e-4) and
                (ex2[1] / ex2[0] < eps or ex2[1] < 1e-4) and
                (ex3[1] / ex3[0] < eps or ex3[1] < 1e-6)):
            logging.info(f'Severity.moms | **DOUBTFUL** convergence of integrals, abs errs '
                         f'\t{ex1[1]}\t{ex2[1]}\t{ex3[1]} \trel errors \t{ex1[1]/ex1[0]}\t{ex2[1]/ex2[0]}\t{ex3[1]/ex3[0]}')
            raise ValueError(f' Severity.moms | doubtful convergence of integrals, abs errs '
                             f'{ex1[1]}, {ex2[1]}, {ex3[1]} rel errors {ex1[1]/ex1[0]}, {ex2[1]/ex2[0]}, {ex3[1]/ex3[0]}')
        # logging.info(f'Severity.moms | GOOD convergence of integrals, abs errs '
        #                      f'\t{ex1[1]}\t{ex2[1]}\t{ex3[1]} \trel errors \t{ex1[1]/ex1[0]}\t{ex2[1]/ex2[0]}\t{ex3[1]/ex3[0]}')

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

        it = axiter_factory(None, 4, figsize=(12, 2))

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
