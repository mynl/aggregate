import collections
import json
import inspect
import itertools
import logging
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
import pandas as pd
from scipy.integrate import quad
import scipy.stats as ss
from scipy import interpolate
from scipy.optimize import newton
from IPython.core.display import display
from scipy.special import kv, gammaln, hyp1f1
from scipy.optimize import broyden2, newton_krylov
from scipy.optimize.nonlin import NoConvergence
from scipy.interpolate import interp1d

from .utilities import sln_fit, sgamma_fit, ft, ift, \
    axiter_factory, estimate_agg_percentile, suptitle_and_tight, \
    MomentAggregator, xsden_to_meancv, round_bucket, make_ceder_netter, MomentWrangler, \
    make_mosaic_figure, nice_multiple, xsden_to_meancvskew, friendly, \
    mu_sigma_from_mean_cv

from .spectral import Distortion

logger = logging.getLogger(__name__)


class Frequency(object):
    """
    Manages Frequency distributions: creates moment function and MGF.

    * freq_moms(n): returns EN, EN^2 and EN^3 when EN=n
    * mgf(n, z): returns the moment generating function applied to z when EN=n

    Frequency distributions are either non-mixture types or mixture types.

    **Non-Mixture** Frequency Types

    * ``fixed``: no parameters
    * ``bernoulli``: exp_en interpreted as a probability, must be < 1
    * ``binomial``: Binomial(n, p) where p = freq_a, and n = exp_en
    * ``poisson``: Poisson(freq_a)
    * ``poisson``: geometric(freq_a)
    * ``pascal``: pascal-poisson distribution, a poisson stopped sum of negative binomial; exp_en gives the overall
      claim count. freq_a is the CV of the negative binomial distribution and freq_b is the
      number of claimants per claim (or claims per occurrence). Hence the Poisson component
      has mean exp_en / freq_b and the number of claims per occurrence has mean freq_b and
      cv freq_a

    **Mixture** Frequency Types

    These distributions are G-mixed Poisson, so N | G ~ Poisson(n G). They are labelled by
    the name of the mixing distribution or the common name for the resulting frequency
    distribution. See Panjer and Willmot or JKK.

    In all cases freq_a is the CV of the mixing distribution which corresponds to the
    asympototic CV of the frequency distribution and of any aggregate when the severity has a variance.

    * ``gamma``: negative binomial, freq_a = cv of gamma distribution
    * ``delaporte``: shifted gamma, freq_a = cv of mixing disitribution, freq_b = proportion of
      certain claims = shift. freq_b must be between 0 and 1.
    * ``ig``: inverse gaussian, freq_a = cv of mixing distribution
    * ``sig``: shifted inverse gaussian, freq_a = cv of mixing disitribution, freq_b = proportion of
      certain claims = shift. freq_b must be between 0 and 1.
    * ``sichel``: generalized inverse gaussian mixing distribution, freq_a = cv of mixing distribution and
      freq_b = lambda value. The beta and mu parameters solved to match moments. Note lambda =
      -0.5 corresponds to inverse gaussian and 0.5 to reciprocal inverse gauusian. Other special
      cases are available.
    * ``sichel.gamma``: generalized inverse gaussian mixture where the parameters match the moments of a
      delaporte distribution with given freq_a and freq_b
    * ``sichel.ig``: generalized inverse gaussian mixture where the parameters match the moments of a
      shifted inverse gaussian distribution with given freq_a and freq_b. This parameterization
      has poor numerical stability and may fail.
    * ``beta``: beta mixing with freq_a = Cv where beta is supported on the interval [0, freq_b]. This
      method should be used carefully. It has poor numerical stability and can produce bizzare
      aggregates when the alpha or beta parameters are < 1 (so there is a mode at 0 or freq_b).

    :param freq_name:
    :param freq_a:
    :param freq_b:

    """

    __slots__ = ['freq_moms', 'mgf', 'freq_name', 'freq_a', 'freq_b', 'freq_zm', 'freq_p0']

    def __init__(self, freq_name, freq_a, freq_b, freq_zm, freq_p0):
        """
        Creates the mgf and moment function:

        freq_zm True if zero modified, default False
        freq_p0 modified value of p0

        # check zero mod is acceptable? --> parser
        if freq_zm is True:
            assertg freq_name in ['poisson', 'binomial', 'geometric',
                    'logarithmic']

        logarithmic??
        Enter NB not as mixed to allow easy creation of zm?

        * moment function(n) returns EN, EN^2, EN^3 when EN=n.
        * mgf(n, z) is the mgf evaluated at log(z) when EN=n

        """
        self.freq_name = freq_name
        self.freq_a = freq_a
        self.freq_b = freq_b
        self.freq_zm = freq_zm
        self.freq_p0 = freq_p0

        if freq_zm is True:
            # add implemented methdods here....
            if freq_name not in ('poisson', 'gamma'):
                raise NotImplementedError(f'Zero modification not implemented for {freq_name}')

        logger.debug(
            f'Frequency.__init__ | creating new Frequency {self.freq_name} at {super(Frequency, self).__repr__()}')

        if self.freq_name == 'fixed':
            def _freq_moms(n):
                # fixed distribution N=n certainly
                freq_2 = n ** 2
                freq_3 = n * freq_2
                return n, freq_2, freq_3

            def mgf(n, z):
                return z ** n

        elif self.freq_name == 'bernoulli':
            def _freq_moms(n):
                # code for bernoulli n, E(N^k) = E(N) = n
                # n in this case only means probability of claim (=expected claim count)
                freq_2 = n
                freq_3 = n
                return n, freq_2, freq_3

            def mgf(n, z):
                # E(e^tlog(z)) = p z + (1-p), z = ft(severity)
                return z * n + np.ones_like(z) * (1 - n)

        elif self.freq_name == 'binomial':
            def _freq_moms(n):
                # binomial(N, p) with mean n, N=n/p
                # http://mathworld.wolfram.com/BinomialDistribution.html
                p = self.freq_a
                N = n / p  # correct mean
                freq_1 = N * p
                freq_2 = N * p * (1 - p + N * p)
                freq_3 = N * p * (1 + p * (N - 1) * (3 + p * (N - 2)))
                return freq_1, freq_2, freq_3

            def mgf(n, z):
                N = n / self.freq_a
                return (z * self.freq_a + np.ones_like(z) * (1 - self.freq_a)) ** N

        elif self.freq_name == 'poisson' and self.freq_a == 0:
            def _freq_moms(n):
                # Poisson
                freq_2 = n * (1 + n)
                freq_3 = n * (1 + n * (3 + n))
                return n, freq_2, freq_3

            def mgf(n, z):
                return np.exp(n * (z - 1))

        elif self.freq_name == 'geometric' and self.freq_a == 0:
            # as for poisson, single parameter
            # https://mathworld.wolfram.com/GeometricDistribution.html and Wikipedia
            # e.g. tester: agg =uw('agg GEOM 3 claims sev dhistogram xps [1] [1] geometric')
            def _freq_moms(n):
                p = 1 / (n + 1)
                freq_2 = (2 - p) * (1 - p) / p ** 2
                freq_3 = (1 - p) * (6 + (p - 6) * p) / p ** 3
                return n, freq_2, freq_3

            def mgf(n, z):
                p = 1 / (n + 1)
                return p / (1 - (1 - p) * z)

        elif self.freq_name == 'pascal':
            # generalized Poisson-Pascal distribution, Panjer Willmot green book. p. 324
            # solve for local c to hit overall c=ν^2 value input
            ν = self.freq_a  # desired overall cv
            κ = self.freq_b  # claims per occurrence

            def _freq_moms(n):
                c = (n * ν ** 2 - 1 - κ) / κ
                # a = 1 / c
                # θ = κ * c
                λ = n / κ  # poisson parameter for number of claims
                g = κ * λ * (
                        2 * c ** 2 * κ ** 2 + 3 * c * κ ** 2 * λ + 3 * c * κ ** 2 + 3 * c * κ + κ ** 2 * λ ** 2 +
                        3 * κ ** 2 * λ + κ ** 2 + 3 * κ * λ + 3 * κ + 1)
                return n, n * (κ * (1 + c + λ) + 1), g

            def mgf(n, z):
                c = (n * ν ** 2 - 1 - κ) / κ
                a = 1 / c
                θ = κ * c
                λ = n / κ  # poisson parameter for number of claims
                return np.exp(λ * ((1 - θ * (z - 1)) ** -a - 1))

        elif self.freq_name == 'empirical':
            # stated en here...need to reach up to agg to set that?!
            # parameters are entered as nps, to a is n values and b is probability masses

            def _freq_moms(n):
                # independent of n, it will be -1
                en = np.sum(self.freq_a * self.freq_b)
                en2 = np.sum(self.freq_a ** 2 * self.freq_b)
                en3 = np.sum(self.freq_a ** 3 * self.freq_b)
                return en, en2, en3

            def mgf(n, z):
                # again, independent of n, not going overboard in method here...
                return self.freq_b @ np.power(z, self.freq_a.reshape((self.freq_a.shape[0], 1)))

        # the remaining options are all mixed poisson ==================================================
        # the factorial moments of the mixed poisson are the noncentral moments of the mixing distribution
        # so for each case we compute the noncentral moments of mix and then convert factorial to non-central
        # the mixing distributions have mean 1 so they can be scaled as appropriate
        # they all use the same f
        elif self.freq_name == 'gamma':
            # gamma parameters a (shape) and  theta (scale)
            # a = 1/c, theta = c
            c = self.freq_a * self.freq_a
            a = 1 / c
            θ = c
            g = 1 + 3 * c + 2 * c * c

            def _freq_moms(n):
                freq_2 = n * (1 + (1 + c) * n)
                freq_3 = n * (1 + n * (3 * (1 + c) + n * g))
                return n, freq_2, freq_3

            def mgf(n, z):
                return (1 - θ * n * (z - 1)) ** -a

        elif self.freq_name == 'delaporte':
            # shifted gamma, freq_a is CV mixing and freq_b  = proportion of certain claims (f for fixed claims)
            ν = self.freq_a
            c = ν * ν
            f = self.freq_b
            # parameters of mixing distribution (excluding the n)
            a = (1 - f) ** 2 / c
            θ = (1 - f) / a
            g = 2 * ν ** 4 / (1 - f) + 3 * c + 1

            def _freq_moms(n):
                freq_2 = n * (1 + (1 + c) * n)
                freq_3 = n * (1 + n * (3 * (1 + c) + n * g))
                return n, freq_2, freq_3

            def mgf(n, z):
                return np.exp(f * n * (z - 1)) * (1 - θ * n * (z - 1)) ** -a

        elif self.freq_name == 'ig':
            # inverse Gaussian distribution
            ν = self.freq_a
            c = ν ** 2
            μ = c
            λ = 1 / μ
            # skewness and E(G^3)
            γ = 3 * np.sqrt(μ)
            g = γ * ν ** 3 + 3 * c + 1

            def _freq_moms(n):
                freq_2 = n * (1 + (1 + c) * n)
                freq_3 = n * (1 + n * (3 * (1 + c) + n * g))
                return n, freq_2, freq_3

            def mgf(n, z):
                return np.exp(1 / μ * (1 - np.sqrt(1 - 2 * μ ** 2 * λ * n * (z - 1))))

        elif self.freq_name == 'sig':
            # shifted pig with a proportion of certain claims
            ν = self.freq_a
            f = self.freq_b
            c = ν * ν  # contagion
            μ = c / (1 - f) ** 2
            λ = (1 - f) / μ
            γ = 3 * np.sqrt(μ)
            g = γ * ν ** 3 + 3 * c + 1

            def _freq_moms(n):
                freq_2 = n * (1 + (1 + c) * n)
                freq_3 = n * (1 + n * (3 * (1 + c) + n * g))
                return n, freq_2, freq_3

            def mgf(n, z):
                return np.exp(f * n * (z - 1)) * np.exp(1 / μ * (1 - np.sqrt(1 - 2 * μ ** 2 * λ * n * (z - 1))))

        elif self.freq_name == 'beta':
            # beta-Poisson mixture [0, b] with mean 1 and cv ν
            # warning: numerically unstable
            ν = self.freq_a  # cv of beta
            c = ν * ν
            r = self.freq_b  # rhs of beta which must be > 1 for mean to equal 1
            assert r > 1

            # mean = a / (a + b) = n / r, var = a x b / [(a + b)^2( a + b + 1)] = c x mean

            def _freq_moms(n):
                b = (r - n * (1 + c)) * (r - n) / (c * n * r)
                a = n / (r - n) * b
                g = r ** 3 * np.exp(gammaln(a + b) + gammaln(a + 3) - gammaln(a + b + 3) - gammaln(a))
                freq_2 = n * (1 + (1 + c) * n)
                freq_3 = n * (1 + n * (3 * (1 + c) + n * g))
                return n, freq_2, freq_3

            def mgf(n, z):
                b = (r - n * (1 + c)) * (r - n) / (c * n * r)
                a = (r - n * (1 + c)) / (c * r)
                return hyp1f1(a, a + b, r * (z - 1))

        elif self.freq_name[0:6] == 'sichel':
            # flavors: sichel.gamma = match to delaporte moments, .ig = match to spig moments (not very numerically
            # stable)
            # sichel: treat freq_b as lambda
            _type = self.freq_name.split('.')
            add_sichel = True
            ν = self.freq_a
            c = ν * ν
            if len(_type) > 1:
                # .gamma or .ig forms
                f = self.freq_b
                λ = -0.5
                μ = 1
                β = ν ** 2
                if _type[1] == 'gamma':
                    # sichel_case 2: match delaporte moments
                    # G = f + G'; E(G') = 1 - f, SD(G) = SD(G') = ν, skew(G') = skew(G)
                    # a = ((1 - f) / ν) ** 2
                    # FWIW θ = ν / (1 - f)  # (1 - f) / a
                    target = np.array([1, ν, 2 * ν / (1 - f)])  # / np.sqrt(a)])
                elif _type[1] == 'ig':
                    # match shifted IG moments
                    # μ = (ν / (1 - f)) ** 2
                    target = np.array([1, ν, 3.0 * ν / (1 - f)])  # np.sqrt(μ)])
                else:
                    raise ValueError(f'Inadmissible frequency type {self.freq_name}...')

                def f(arrIn):
                    """
                    calibration function to match target mean, cv and skewness (keeps the scale about the same)
                    :param arrIn:
                    :return:
                    """
                    μ, β, λ = arrIn
                    # mu and beta are positive...
                    μ = np.exp(μ)
                    β = np.exp(β)
                    ex1, ex2, ex3 = np.array([μ ** r * kv(λ + r, μ / β) / kv(λ, μ / β) for r in (1, 2, 3)])
                    sd = np.sqrt(ex2 - ex1 * ex1)
                    skew = (ex3 - 3 * ex2 * ex1 + 2 * ex1 ** 3) / (sd ** 3)
                    return np.array([ex1, sd, skew]) - target

                try:
                    params = broyden2(f, (np.log(μ), np.log(β), λ), verbose=False, iter=10000,
                                      f_rtol=1e-11)  # , f_rtol=1e-9)  , line_search='wolfe'
                    if np.linalg.norm(params) > 20:
                        λ = -0.5
                        μ = 1
                        β = ν ** 2
                        params1 = newton_krylov(f, (np.log(μ), np.log(β), λ), verbose=False, iter=10000, f_rtol=1e-11)
                        logger.warning(
                            f'Frequency.__init__ | {self.freq_name} type Broyden gave large result {params},'
                            f'Newton Krylov {params1}')
                        if np.linalg.norm(params) > np.linalg.norm(params1):
                            params = params1
                            logger.warning('Frequency.__init__ | using Newton K')
                except NoConvergence as e:
                    print('ERROR: broyden did not converge')
                    print(e)
                    add_sichel = False
                    raise e
            else:
                # pure sichel, match cv and use
                λ = self.freq_b
                target = np.array([1, ν])
                μ = 1
                β = ν ** 2

                def f(arrIn):
                    """
                    calibration function to match target mean = 1 and cv
                    :param arrIn:
                    :return:
                    """
                    μ, β = arrIn
                    # mu and beta are positive...
                    μ = np.exp(μ)
                    β = np.exp(β)
                    ex1, ex2 = np.array([μ ** r * kv(λ + r, μ / β) / kv(λ, μ / β) for r in (1, 2)])
                    sd = np.sqrt(ex2 - ex1 * ex1)
                    return np.array([ex1, sd]) - target

                try:
                    params = broyden2(f, (np.log(μ), np.log(β)), verbose=False, iter=10000,
                                      f_rtol=1e-11)  # , f_rtol=1e-9)  , line_search='wolfe'

                except NoConvergence as e:
                    print('ERROR: broyden did not converge')
                    print(e)
                    add_sichel = False
                    raise e

            # if parameters found...
            logger.debug(f'{self.freq_name} type, params from Broyden {params}')
            if add_sichel:
                if len(_type) == 1:
                    μ, β = params
                else:
                    μ, β, λ = params
                μ, β = np.exp(μ), np.exp(β)
                g = μ ** 2 * kv(λ + 2, μ / β) / kv(λ, μ / β)

                def _freq_moms(n):
                    freq_2 = n * (1 + (1 + c) * n)
                    freq_3 = n * (1 + n * (3 * (1 + c) + n * g))
                    return n, freq_2, freq_3

                def mgf(n, z):
                    kernel = n * (z - 1)
                    inner = np.sqrt(1 - 2 * β * kernel)
                    return inner ** (-λ) * kv(λ, μ * inner / β) / kv(λ, μ / β)

        else:
            raise ValueError(f'Inadmissible frequency type {self.freq_name}...')

        self.freq_moms = _freq_moms
        self.mgf = mgf

    def __str__(self):
        """
        wrap default with name
        :return:
        """
        return f'Frequency object of type {self.freq_name}\n{super(Frequency, self).__repr__()}'


class Aggregate(Frequency):

    # TODO must be able to automate this with inspect
    aggregate_keys = ['name', 'exp_el', 'exp_premium', 'exp_lr', 'exp_en', 'exp_attachment', 'exp_limit', 'sev_name',
                      'sev_a', 'sev_b', 'sev_mean', 'sev_cv', 'sev_loc', 'sev_scale', 'sev_xs', 'sev_ps',
                      'sev_wt', 'occ_kind', 'occ_reins', 'freq_name', 'freq_a', 'freq_b', 'freq_zm', 'freq_p0',
                      'agg_kind', 'agg_reins', 'note']

    @property
    def spec(self):
        """
        Get the dictionary specification, but treat as a read only
        property

        :return:
        """
        return self._spec

    @property
    def spec_ex(self):
        """
        All relevant info.

        :return:
        """
        return {'type': type(self), 'spec': self.spec, 'bs': self.bs, 'log2': self.log2,
                'sevs': len(self.sevs)}

    @property
    def density_df(self):
        """
        Create and return the density_df data frame. A read only property, though if you write d = a.density_df you
        can obviously edit d. Some duplication of columns (p and p_total) to ensure consistency with Portfolio.

        :return: DataFrame similar to Portfolio.density_df.

        """
        if self._density_df is None:
            # really should have one of these anyway...
            if self.agg_density is None:
                raise ValueError('Update Aggregate before asking for density_df')

            # really convenient to have p=p_total to be consistent with Portfolio objects
            self._density_df = pd.DataFrame(dict(loss=self.xs, p=self.agg_density, p_total=self.agg_density,
                                                 p_sev=self.sev_density))
            # remove the fuzz, same method as Portfolio.remove_fuzz
            eps = np.finfo(np.float).eps
            # may not have a severity, remember...
            self._density_df.loc[:, self._density_df.select_dtypes(include=['float64']).columns] = \
                self._density_df.select_dtypes(include=['float64']).applymap(lambda x: 0 if abs(x) < eps else x)

            # reindex
            self._density_df = self._density_df.set_index('loss', drop=False)
            self._density_df['log_p'] = np.log(self._density_df.p)
            # when no sev this causes a problem
            if self._density_df.p_sev.dtype == np.dtype('O'):
                self._density_df['log_p_sev'] = np.nan
            else:
                self._density_df['log_p_sev'] = np.log(self._density_df.p_sev)

            # generally acceptable for F, by construction
            self._density_df['F'] = self._density_df.p.cumsum()
            self._density_df['F_sev'] = self._density_df.p_sev.cumsum()

            # S is more difficult. Can use 1-F or reverse cumsum pf p. Former is accurate on the
            # left, latter more accurate in the right tail. For lev and similar calcs, care about
            # the left (int of S). Upshot: need to pick the fill value carefully. Here is what
            # Portfolio does
            # fill_value = min(self._density_df.p_total.iloc[-1], max(0, 1. - (self._density_df.F.iloc[-1])))
            fill_value = max(0, 1. - (self._density_df.F.iloc[-1]))
            self._density_df['S'] = self._density_df.p.shift(-1, fill_value=fill_value)[::-1].cumsum()
            fill_value = max(0, 1. - (self._density_df.F_sev.iloc[-1]))
            self._density_df['S_sev'] = self._density_df.p_sev.shift(-1, fill_value=fill_value)[::-1].cumsum()

            # add LEV, TVaR to each threshold point...
            self._density_df['lev'] = self._density_df.S.shift(1, fill_value=0).cumsum() * self.bs
            self._density_df['exa'] = self._density_df['lev']
            self._density_df['exlea'] = \
                (self._density_df.lev - self._density_df.loss * self._density_df.S) / self._density_df.F
            # fix very small values, see port add_exa
            n_ = self._density_df.shape[0]
            if n_ < 1100:
                mult = 1
            elif n_ < 15000:
                mult = 10
            else:
                mult = 100
            loss_max = self._density_df[['loss', 'exlea']].query(' exlea > loss ').loss.max()
            if np.isnan(loss_max):
                loss_max = 0
            else:
                loss_max += mult * self.bs
            self._density_df.loc[0:loss_max, 'exlea'] = 0
            # expected value and epd
            self._density_df['e'] = np.sum(self._density_df.p * self._density_df.loss)
            self._density_df.loc[:, 'epd'] = \
                np.maximum(0, (self._density_df.loc[:, 'e'] - self._density_df.loc[:, 'lev'])) / \
                self._density_df.loc[:, 'e']
            self._density_df['exgta'] = self._density_df.loss + (
                    self._density_df.e - self._density_df.exa) / self._density_df.S
            self._density_df['exeqa'] = self._density_df.loss  # E(X | X=a) = a(!) included for symmetry was exa

        return self._density_df

    @property
    def reins_audit_df(self):
        """
        Create and return the _reins_audit_df data frame.
        Read only property.

        :return:
        """
        if self._reins_audit_df is None:
            # really should have one of these anyway...
            if self.agg_density is None:
                raise ValueError('Update Aggregate before asking for density_df')

            ans = []
            keys = []
            if self.occ_reins is not None:
                ans.append(self._reins_audit_df_work(kind='occ'))
                keys.append('occ')
            if self.agg_reins is not None:
                ans.append(self._reins_audit_df_work(kind='agg'))
                keys.append('agg')

            if len(ans):
                self._reins_audit_df = pd.concat(ans, keys=keys, names=['kind', 'share', 'limit', 'attach'])

        return self._reins_audit_df

    def _reins_audit_df_work(self, kind='occ'):
        """
        Apply each re layer separately and aggregate loss and other stats.

        """
        ans = []
        assert self.sev_density is not None

        # reins = self.occ_reins if kind == 'occ' else self.agg_reins

        # TODO what about agg?
        if kind == 'occ':
            if self.sev_gross_density is None:
                self.sev_gross_density = self.sev_density
            reins = self.occ_reins
            for (s, y, a) in reins:
                c, n, df = self._apply_reins_work([(s, y, a)], self.sev_gross_density, False)
                ans.append(df)
            ans.append(self.occ_reins_df)
        elif kind == 'agg':
            if self.agg_gross_density is None:
                self.agg_gross_density = self.agg_density
            reins = self.agg_reins
            for (s, y, a) in reins:
                c, n, df = self._apply_reins_work([(s, y, a)], self.agg_gross_density, False)
                ans.append(df)
            ans.append(self.agg_reins_df)

        df = pd.concat(ans, keys=reins + [('all', np.inf, 'gup')], names=['share', 'limit', 'attach', 'loss'])
        # subset and reindex
        df = df.filter(regex='^(F|p)')
        df.columns = df.columns.str.split('_', expand=True)
        df = df.sort_index(axis=1)

        # summarize
        def f(bit):
            # summary function to compute stats
            xs = bit.index.levels[3]
            xs2 = xs * xs
            xs3 = xs2 * xs

            def g(p):
                ex = np.sum(xs * p)
                ex2 = np.sum(xs2 * p)
                ex3 = np.sum(xs3 * p)
                mw = MomentWrangler()
                mw.noncentral = (ex, ex2, ex3)
                return mw.stats

            return bit['p'].apply(g)

        return df.groupby(level=(0, 1, 2)).apply(f).unstack(-1).sort_index(level='attach')

    def rescale(self, scale, kind='homog'):
        """
        Return a rescaled Aggregate object - used to compute derivatives.

        All need to be safe multiplies because of array specification there is an array that is not a numpy array

        TODO have parser return numpy arrays not lists!

        :param scale:  amount of scale
        :param kind:  homog of inhomog

        :return:
        """
        spec = self._spec.copy()

        def safe_scale(sc, x):
            """
            if x is a list wrap it

            :param x:
            :param sc:
            :return: sc x
            """

            if type(x) == list:
                return sc * np.array(x)
            else:
                return sc * x

        nm = spec['name']
        spec['name'] = f'{nm}:{kind}:{scale}'
        if kind == 'homog':
            # do NOT scale en... that is inhomog
            # do scale EL etc. to keep the count the same
            spec['exp_el'] = safe_scale(scale, spec['exp_el'])
            spec['exp_premium'] = safe_scale(scale, spec['exp_premium'])
            spec['exp_attachment'] = safe_scale(scale, spec['exp_attachment'])
            spec['exp_limit'] = safe_scale(scale, spec['exp_limit'])
            spec['sev_loc'] = safe_scale(scale, spec['sev_loc'])
            # note: scaling the scale takes care of the mean, so do not double count
            # default is 0. Can't ask if array is...but if array have to deal with it
            if (type(spec['sev_scale']) not in (int, float)) or spec['sev_scale']:
                spec['sev_scale'] = safe_scale(scale, spec['sev_scale'])
            else:
                spec['sev_mean'] = safe_scale(scale, spec['sev_mean'])
            if spec['sev_xs']:
                spec['sev_xs'] = safe_scale(scale, spec['sev_xs'])
        elif kind == 'inhomog':
            # just scale up the volume, including en
            spec['exp_el'] = safe_scale(scale, spec['exp_el'])
            spec['exp_premium'] = safe_scale(scale, spec['exp_premium'])
            spec['exp_en'] = safe_scale(scale, spec['exp_en'])
        else:
            raise ValueError(f'Inadmissible option {kind} passed to rescale, kind should be homog or inhomog.')
        return Aggregate(**spec)

    def __init__(self, name, exp_el=0, exp_premium=0, exp_lr=0, exp_en=0, exp_attachment=0, exp_limit=np.inf,
                 sev_name='', sev_a=np.nan, sev_b=0, sev_mean=0, sev_cv=0, sev_loc=0, sev_scale=0,
                 sev_xs=None, sev_ps=None, sev_wt=1, sev_conditional=True,
                 occ_reins=None, occ_kind='',
                 freq_name='', freq_a=0, freq_b=0, freq_zm=False, freq_p0=np.nan,
                 agg_reins=None, agg_kind='',
                 note=''):
        """
        Aggregate distribution class manages creation and calculation of aggregate distributions.
        It allows for very flexible creation of Aggregate distributions. Severity
        can express a limit profile, a mixed severity or both. Mixed frequency types share
        a mixing distribution across all broadcast terms to ensure an appropriate inter-
        class correlation.

        :param name:            name of the aggregate
        :param exp_el:          expected loss or vector
        :param exp_premium:     premium volume or vector  (requires loss ratio)
        :param exp_lr:          loss ratio or vector  (requires premium)
        :param exp_en:          expected claim count per segment (self.n = total claim count)
        :param exp_attachment:  occurrence attachment
        :param exp_limit:       occurrence limit
        :param sev_name:        severity name or sev.BUILTIN_SEV or meta.var agg or port or similar or vector or matrix
        :param sev_a:           scipy stats shape parameter
        :param sev_b:           scipy stats shape parameter
        :param sev_mean:        average (unlimited) severity
        :param sev_cv:          unlimited severity coefficient of variation
        :param sev_loc:         scipy stats location parameter
        :param sev_scale:       scipy stats scale parameter
        :param sev_xs:          xs and ps must be provided if sev_name is (c|d)histogram, xs are the bucket break points
        :param sev_ps:          ps are the probability densities within each bucket; if buckets equal size no adjustments needed
        :param sev_wt:          weight for mixed distribution
        :param sev_conditional: if True, severity is conditional, else unconditional.
        :param occ_reins:       layers: share po layer xs attach or XXXX
        :param occ_kind:        ceded to or net of
        :param freq_name:       name of frequency distribution
        :param freq_a:          cv of freq dist mixing distribution
        :param freq_b:          claims per occurrence (delaporte or sig), scale of beta or lambda (Sichel)
        :param freq_zm:         True/False zero modified flag
        :param freq_p0:         if freq_zm, provides the modified value of p0; default is nan
        :param agg_reins:       layers
        :param agg_kind:        ceded to or net of
        :param note:            note, enclosed in {}
        """

        # have to be ready for inputs to be in a list, e.g. comes that way from Pandas via Excel
        def get_value(v):
            if isinstance(v, list):
                return v[0]
            else:
                return v

        # class variables
        self.name = get_value(name)
        # for persistence, save the raw called spec... (except lookups have been replaced...)
        # TODO want to use the trick with setting properties so that if they are altered spec gets altered...
        # self._spec = dict(name=name, exp_el=exp_el, exp_premium=exp_premium, exp_lr=exp_lr, exp_en=exp_en,
        #                   exp_attachment=exp_attachment, exp_limit=exp_limit,
        #                   sev_name=sev_name, sev_a=sev_a, sev_b=sev_b, sev_mean=sev_mean, sev_cv=sev_cv,
        #                   sev_loc=sev_loc, sev_scale=sev_scale, sev_xs=sev_xs, sev_ps=sev_ps, sev_wt=sev_wt,
        #                   sev_conditional=sev_conditional,
        #                   occ_reins=occ_reins, occ_kind=occ_kind,
        #                   freq_name=freq_name, freq_a=freq_a, freq_b=freq_b,
        #                   agg_reins=agg_reins, agg_kind=agg_kind, note=note)
        # using inspect, more robust...must call before you create other variables
        frame = inspect.currentframe()
        self._spec = inspect.getargvalues(frame).locals
        for n in ['frame', 'get_value', 'self']:
            if n in self._spec: self._spec.pop(n)

        logger.debug(
            f'Aggregate.__init__ | creating new Aggregate {self.name}')
        Frequency.__init__(self, get_value(freq_name), get_value(freq_a), get_value(freq_b),
                           get_value(freq_zm), get_value(freq_p0))
        self.figure = None
        self.xs = None
        self.bs = 0
        self.log2 = 0
        self.ex = 0
        self.note = note
        self.program = ''  # can be set externally
        self.en = None     # this is for a sublayer e.g. for limit profile
        self.n = 0         # this is total frequency
        self.attachment = None
        self.limit = None
        self.agg_density = None
        self.sev_density = None
        self.dh_agg_density = None
        self.dh_sev_density = None
        self.ftagg_density = None
        self.fzapprox = None
        self._tail_var = None
        self._tail_var2 = None
        self._inverse_tail_var = None
        # self.agg_m, self.agg_cv, self.agg_skew = 0, 0, 0
        self._linear_quantile_function = None
        self._cdf = None
        self._pdf = None
        self.beta_name = ''  # name of the beta function used to create dh distortion
        self.sevs = None
        self.audit_df = None
        self.verbose_audit_df = None
        self._careful_q = None
        self._density_df = None
        self._reins_audit_df = None
        self.q_temp = None
        self.occ_reins = occ_reins
        self.occ_kind = occ_kind
        self.occ_netter = None
        self.occ_ceder = None
        self.occ_reins_df = None
        self.agg_reins = agg_reins
        self.agg_kind = agg_kind
        self.agg_netter = None
        self.agg_ceder = None
        self.agg_reins_df = None
        self.sev_ceded_density = None
        self.sev_net_density = None
        self.sev_gross_density = None
        self.agg_ceded_density = None
        self.agg_net_density = None
        self.agg_gross_density = None
        self.sev_calc = ""
        self.discretization_calc = ""
        self.normalize = ""
        self.statistics_df = pd.DataFrame(columns=['name', 'limit', 'attachment', 'sevcv_param', 'el', 'prem', 'lr'] +
                                                  MomentAggregator.column_names() +
                                                  ['mix_cv'])
        self.statistics_total_df = self.statistics_df.copy()
        ma = MomentAggregator(self.freq_moms)

        # broadcast arrays: force answers all to be arrays (?why only these items?!)
        if not isinstance(exp_el, collections.Iterable):
            exp_el = np.array([exp_el])
        if not isinstance(sev_wt, collections.Iterable):
            sev_wt = np.array([sev_wt])

        # broadcast together and create container for the severity distributions
        if np.sum(sev_wt) == len(sev_wt):
            # do not perform the exp / sev product, in this case
            # broadcast all exposure and sev terms together
            exp_el, exp_premium, exp_lr, en, attachment, limit, \
            sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, sev_wt = \
                np.broadcast_arrays(exp_el, exp_premium, exp_lr, exp_en, exp_attachment, exp_limit,
                                    sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, sev_wt)
            exp_el = np.where(exp_el > 0, exp_el, exp_premium * exp_lr)
            all_arrays = list(zip(exp_el, exp_premium, exp_lr, en, attachment, limit,
                                  sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, sev_wt))
            self.en = en
            self.attachment = attachment
            self.limit = limit
            n_components = len(all_arrays)
            logger.debug(f'Aggregate.__init__ | Broadcast/align: exposures + severity = {len(exp_el)} exp = '
                         f'{len(sev_a)} sevs = {n_components} componets')
            self.sevs = np.empty(n_components, dtype=type(Severity))

        else:
            # perform exp / sev product
            # broadcast exposure terms (el, epremium, en, lr, attachment, limit) and sev terms (sev_) separately
            # then we take an "outer product" of the two parts...
            exp_el, exp_premium, exp_lr, en, attachment, limit = \
                np.broadcast_arrays(exp_el, exp_premium, exp_lr, exp_en, exp_attachment, exp_limit)
            sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, sev_wt = \
                np.broadcast_arrays(sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, sev_wt)
            exp_el = np.where(exp_el > 0, exp_el, exp_premium * exp_lr)
            exp_arrays = [exp_el, exp_premium, exp_lr, en, attachment, limit]
            sev_arrays = [sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, sev_wt]
            all_arrays = [[k for j in i for k in j] for i in itertools.product(zip(*exp_arrays), zip(*sev_arrays))]
            self.en = np.array([i[3] * i[-1] for i in all_arrays])
            self.attachment = np.array([i[4] for i in all_arrays])
            self.limit = np.array([i[5] for i in all_arrays])
            n_components = len(all_arrays)
            logger.debug(
                f'Aggregate.__init__ | Broadcast/product: exposures x severity = {len(exp_arrays)} x {len(sev_arrays)} '
                f'=  {n_components}')
            self.sevs = np.empty(n_components, dtype=type(Severity))

        # overall freq CV with common mixing
        mix_cv = self.freq_a
        # counter to label components
        r = 0
        # perform looping creation of severity distribution
        for _el, _pr, _lr, _en, _at, _y, _sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _swt in all_arrays:

            # WARNING: note sev_xs and sev_ps are NOT broadcast
            self.sevs[r] = Severity(_sn, _at, _y, _sm, _scv, _sa, _sb, _sloc, _ssc, sev_xs, sev_ps, _swt, sev_conditional)
            sev1, sev2, sev3 = self.sevs[r].moms()

            # input claim count trumps input loss
            if _en > 0:
                _el = _en * sev1
            elif _el > 0:
                _en = _el / sev1
            # if premium compute loss ratio, if loss ratio compute premium
            if _pr > 0:
                _lr = _el / _pr
            elif _lr > 0:
                _pr = _el / _lr

            # for empirical freq claim count entered as -1
            if _en < 0:
                _en = np.sum(self.freq_a * self.freq_b)
                _el = _en * sev1

            # scale for the mix - OK because we have split the exposure and severity components
            _pr *= _swt
            _el *= _swt
            _lr *= _swt
            _en *= _swt

            # accumulate moments
            ma.add_f1s(_en, sev1, sev2, sev3)

            # store
            self.statistics_df.loc[r, :] = \
                [self.name, _y, _at, _scv, _el, _pr, _lr] + ma.get_fsa_stats(total=False) + [mix_cv]
            r += 1

        # average exp_limit and exp_attachment
        avg_limit = np.sum(self.statistics_df.limit * self.statistics_df.freq_1) / ma.tot_freq_1
        avg_attach = np.sum(self.statistics_df.attachment * self.statistics_df.freq_1) / ma.tot_freq_1
        # assert np.allclose(ma.freq_1, self.statistics_df.exp_en)

        # store answer for total
        tot_prem = self.statistics_df.prem.sum()
        tot_loss = self.statistics_df.el.sum()
        if tot_prem > 0:
            lr = tot_loss / tot_prem
        else:
            lr = np.nan
        self.statistics_total_df.loc[f'mixed', :] = \
            [self.name, avg_limit, avg_attach, 0, tot_loss, tot_prem, lr] + ma.get_fsa_stats(total=True, remix=True) \
            + [mix_cv]
        self.statistics_total_df.loc[f'independent', :] = \
            [self.name, avg_limit, avg_attach, 0, tot_loss, tot_prem, lr] + ma.get_fsa_stats(total=True, remix=False) \
            + [mix_cv]
        self.statistics_df['wt'] = self.statistics_df.freq_1 / ma.tot_freq_1
        self.statistics_total_df['wt'] = self.statistics_df.wt.sum()  # better equal 1.0!
        self.n = ma.tot_freq_1
        self.agg_m = self.statistics_total_df.loc['mixed', 'agg_m']
        self.agg_cv = self.statistics_total_df.loc['mixed', 'agg_cv']
        self.agg_skew = self.statistics_total_df.loc['mixed', 'agg_skew']
        # variance and sd come up in exam questions
        self.agg_sd = self.agg_m * self.agg_cv
        self.agg_var = self.agg_sd * self.agg_sd
        # finally, need a report_ser series for Portfolio to consolidate
        self.report_ser = ma.stats_series(self.name, np.max(self.limit), 0.999, remix=True)
        self._middle_q = None
        self._q = None

    def __repr__(self):
        """
        wrap default with name
        :return:
        """
        return f'{super(Aggregate, self).__repr__()} name: {self.name}'

    def __str__(self):
        """
        Goal: readability

        :return:
        """
        # pull out agg statistics_df
        ags = self.statistics_total_df.loc['mixed', :]
        s = f"Aggregate: {self.name}\n\tEN={ags['freq_1']}, CV(N)={ags['freq_cv']:5.3f}\n\t" \
            f"{len(self.sevs)} severit{'ies' if len(self.sevs) > 1 else 'y'}, EX={ags['sev_1']:,.1f}, " \
            f"CV(X)={ags['sev_cv']:5.3f}\n\t" \
            f"EA={ags['agg_1']:,.1f}, CV={ags['agg_cv']:5.3f}"
        return s

    def _repr_html_(self):
        s = [f'<h3>Aggregate object: {self.name}</h3>']
        s.append(f'Claim count {self.n:0,.2f}, {self.freq_name} distribution.<br>')
        n = len(self.statistics_df)
        if n == 1:
            sv = self.sevs[0]
            if sv.limit == np.inf and sv.attachment == 0:
                _la = 'unlimited'
            else:
                _la = f'{sv.limit} xs {sv.attachment}'
            s.append(f'Severity{sv.long_name} distribution, {_la}.<br>')
        else:
            s.append(f'Severity with {n} components.<br>')
        if self.bs > 0:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{1/self.bs:,.0f}'
            s.append(f'Updated with bucket size {bss} and log2 = {self.log2}.')
        df = self.describe
        return '\n'.join(s) + df.to_html()

    def discretize(self, sev_calc, discretization_calc, normalize):
        """
        Discretize the severity distributions and weight.

        `sev_calc='continuous'` is used when you think of the resulting distribution as continuous across the buckets
        (which we generally don't). The buckets are not shifted and so :math:`Pr(X=b_i) = Pr( b_{i-1} < X \le b_i)`.
        Note that :math:`b_{i-1}=-bs/2` is prepended.

        We use the discretized distribution as though it is fully discrete and only takes values at the bucket
        points. Hence, we should use `sev_calc='discrete'`. The buckets are shifted left by half a bucket,
        so :math:`Pr(X=b_i) = Pr( b_i - b/2 < X \le b_i + b/2)`.

        The other wrinkle is the righthand end of the range. If we extend to np.inf then we ensure we have
        probabilities that sum to 1. But that method introduces a probability mass in the last bucket that
        is often not desirable (we expect to see a smooth continuous distribution, and we get a mass). The
        other alternative is to use endpoint = 1 bucket beyond the last, which avoids this problem but can leave
        the probabilities short. We opt here for the latter and normalize (rescale).

        `discretization_calc` controls whether individual probabilities are computed using backward-differences of
        the survival function or forward differences of the distribution function, or both. The former is most
        accurate in the right-tail and the latter for the left-tail of the distribution. We are usually concerned
        with the right-tail, so prefer `survival`. Using `both` takes the greater of the two esimates giving the best
        of both worlds (underflow makes distribution zero in the right-tail and survival zero in the left tail,
        so the maximum gives the best estimate) at the expense of computing time.

        Sensible defaults: sev_calc=discrete, discretization_calc=survival, normalize=True.

        :param sev_calc:  continuous or discrete or raw (for...);
               and method becomes discrete otherwise
        :param discretization_calc:  survival, distribution or both; in addition
               the method then becomes survival
        :param normalize: if True, normalize the severity so sum probs = 1. This is generally what you want; but
               when dealing with thick tailed distributions it can be helpful to turn it off.
        :return:
        """

        if sev_calc == 'continuous':
            adj_xs = np.hstack((self.xs, self.xs[-1] + self.bs))
        elif sev_calc == 'discrete':
            # adj_xs = np.hstack((self.xs - self.bs / 2, np.inf))
            # mass at the end undesirable. can be put in with reinsurance layer in spec
            # note the first bucket is negative
            adj_xs = np.hstack((self.xs - self.bs / 2, self.xs[-1] + self.bs / 2))
        elif sev_calc == 'raw':
            adj_xs = self.xs
        else:
            raise ValueError(
                f'Invalid parameter {sev_calc} passed to discretize; options are discrete, continuous, or raw.')

        # bed = bucketed empirical distribution
        beds = []
        for fz in self.sevs:
            if discretization_calc == 'both':
                # see comments: we rescale each severity...
                appx = np.maximum(np.diff(fz.cdf(adj_xs)), -np.diff(fz.sf(adj_xs)))
            elif discretization_calc == 'survival':
                appx = -np.diff(fz.sf(adj_xs))
                # beds.append(appx / np.sum(appx))
            elif discretization_calc == 'distribution':
                appx = np.diff(fz.cdf(adj_xs))
                # beds.append(appx / np.sum(appx))
            else:
                raise ValueError(
                    f'Invalid options {discretization_calc} to double_diff; options are density, survival or both')
            if normalize:
                beds.append(appx / np.sum(appx))
            else:
                beds.append(appx)
        return beds

    def snap(self, x):
        """
        Snap value x to the index of density_df, i.e., as a multiple of self.bs.

        :param x:
        :return:
        """
        ix = self.density_df.index.get_loc(x, 'nearest')
        return self.density_df.iat[ix, 0]

    def update(self, log2=13, bs=0, debug=False, **kwargs):
        """
        Convenience function, delegates to update_work. Avoids having to pass xs. Also
        aliased as easy_update for backward compatibility.

        :param log2:
        :param bs:
        :param debug:
        :param kwargs:  passed through to update
        :return:
        """
        # guess bucket and update
        if bs == 0:
            bs = round_bucket(self.recommend_bucket(log2))
        xs = np.arange(0, 1 << log2, dtype=float) * bs
        if 'approximation' not in kwargs:
            if self.n > 100:
                kwargs['approximation'] = 'slognorm'
            else:
                kwargs['approximation'] = 'exact'
        return self.update_work(xs, debug=debug, **kwargs)

    # for backwards compatibility
    easy_update = update

    def update_work(self, xs, padding=1, tilt_vector=None, approximation='exact', sev_calc='discrete',
               discretization_calc='survival', normalize=True, force_severity=False, debug=False):
        """
        Compute a discrete approximation to the aggregate density.

        See discretize for sev_calc, discretization_calc and normalize.


        Quick simple test with log2=13 update took 5.69 ms and _eff took 2.11 ms. So quicker
        but not an issue unless you are doing many buckets or aggs.

        :param xs: range of x values used to discretize
        :param padding: for FFT calculation
        :param tilt_vector: tilt_vector = np.exp(self.tilt_amount * np.arange(N)), N=2**log2, and
               tilt_amount * N < 20 recommended
        :param approximation: 'exact' = perform frequency / severity convolution using FFTs.
               'slognorm' or 'sgamma' use a shifted lognormal or shifted gamma approximation.
        :param sev_calc: `discrete` = suitable for fft, `continuous` = for rv_histogram cts version. Only
               use discrete unless you know what you are doing!
        :param discretization_calc: use survival, distribution or both (=max(cdf, sf)) which is most accurate calc
        :param normalize: normalize severity to 1.0
        :param force_severity: make severities even if using approximation, for plotting
        :param debug: run reinsurance in debug model if True.
        :return:
        """
        self._density_df = None  # invalidate
        self._linear_quantile_function = None
        self.sev_calc = sev_calc
        self.discretization_calc = discretization_calc
        self.normalize = normalize
        self.xs = xs
        self.bs = xs[1]
        # WHOA! WTF
        self.log2 = int(np.log(len(xs)) / np.log(2))

        # make the severity vector: a claim count weighted average of the severities
        if approximation == 'exact' or force_severity:
            wts = self.statistics_df.freq_1 / self.statistics_df.freq_1.sum()
            self.sev_density = np.zeros_like(xs)
            beds = self.discretize(sev_calc, discretization_calc, normalize)
            for temp, w, a, l, n in zip(beds, wts, self.attachment, self.limit, self.en):
                self.sev_density += temp * w

        if force_severity == 'yes':
            # only asking for severity (used by plot)
            return

        # deal with per occ reinsurance
        # TODO issues with force_severity = False.... get rid of that option entirely?
        if self.occ_reins is not None:
            logger.info('Applying occurrence reinsurance.')
            if self.sev_gross_density is not None:
                # make the function an involution...
                self.sev_density = self.sev_gross_density
            self.apply_occ_reins(debug)

        if approximation == 'exact':
            if self.n > 100:
                logger.warning(f'Claim count {self.n} is high; consider an approximation ')

            if self.n == 0:
                # for dynamics it is helpful to have a zero risk return zero appropriately
                # z = ft(self.sev_density, padding, tilt_vector)
                self.agg_density = np.zeros_like(self.xs)
                self.agg_density[0] = 1
                # extreme idleness...but need to make sure it is the right shape and type
                self.ftagg_density = ft(self.agg_density, padding, tilt_vector)
            else:
                # usual calculation...this is where the magic happens!
                # have already dealt with per occ reinsurance
                # don't loose accuracy and time by going through this step if freq is fixed 1
                # these are needed when agg is part of a portfolio
                z = ft(self.sev_density, padding, tilt_vector)
                self.ftagg_density = self.mgf(self.n, z)
                if np.sum(self.en) == 1 and self.freq_name == 'fixed':
                    logger.info('FIXED 1: skipping FFT calculation')
                    # copy to be safe
                    self.agg_density = self.sev_density.copy()
                else:
                    # logger.info('Performing fft convolution')
                    self.agg_density = np.real(ift(self.ftagg_density, padding, tilt_vector))

                # NOW have to apply agg reinsurance to this line
                self.apply_agg_reins(debug)

        else:
            # regardless of request if skew == 0 have to use normal
            # must check there is no per occ reinsurance... it won't work
            assert self.occ_reins is None

            if self.agg_skew == 0:
                self.fzapprox = ss.norm(scale=self.agg_m * self.agg_cv, loc=self.agg_m)
            elif approximation == 'slognorm':
                shift, mu, sigma = sln_fit(self.agg_m, self.agg_cv, self.agg_skew)
                self.fzapprox = ss.lognorm(sigma, scale=np.exp(mu), loc=shift)
            elif approximation == 'sgamma':
                shift, alpha, theta = sgamma_fit(self.agg_m, self.agg_cv, self.agg_skew)
                self.fzapprox = ss.gamma(alpha, scale=theta, loc=shift)
            else:
                raise ValueError(f'Invalid approximation {approximation} option passed to CAgg density. '
                                 'Allowable options are: exact | slogorm | sgamma')

            ps = self.fzapprox.pdf(xs)
            self.agg_density = ps / np.sum(ps)
            self.ftagg_density = ft(self.agg_density, padding, tilt_vector)
            # can still apply aggregate in this mode
            self.apply_agg_reins(debug)

        # make a suitable audit_df
        # originally...irritating no freq cv or sev cv
        # cols = ['name', 'limit', 'attachment', 'el', 'freq_1', 'sev_1', 'agg_m', 'agg_cv', 'agg_skew']
        cols = ['name', 'limit', 'attachment', 'el', 'freq_1', 'freq_cv', 'freq_skew',
                'sev_1', 'sev_cv', 'sev_skew', 'agg_m', 'agg_cv', 'agg_skew']
        self.audit_df = pd.concat((self.statistics_df[cols],
                                   self.statistics_total_df.loc[['mixed'], cols]),
                                  axis=0)
        # add empirical stats
        if self.sev_density is not None:
            _m, _cv, _sk = xsden_to_meancvskew(self.xs, self.sev_density)
        else:
            _m = np.nan
            _cv = np.nan
            _sk = np.nan
        self.audit_df.loc['mixed', 'emp_sev_1'] = _m
        self.audit_df.loc['mixed', 'emp_sev_cv'] = _cv
        self.audit_df.loc['mixed', 'emp_sev_skew'] = _sk
        _m, _cv, _sk = xsden_to_meancvskew(self.xs, self.agg_density)
        self.audit_df.loc['mixed', 'emp_agg_1'] = _m
        self.ex = _m
        self.audit_df.loc['mixed', 'emp_agg_cv'] = _cv
        self.audit_df.loc['mixed', 'emp_agg_skew'] = _sk

        # invalidate stored functions
        self.nearest_quantile_function = None
        self._cdf = None

    def update_efficiently(self, xs, padding=1, approximation='exact', sev_calc='discrete',
                           discretization_calc='survival', normalize=True):
        """
        Compute the density with absolute minimum overhead. Called by port.update_efficiently
        Started with code for update and removed frills
        No tilting!
        :param xs:  range of x values used to discretize
        :param padding: for FFT calculation
        :param approximation: exact = perform frequency / severity convolution using FFTs. slognorm or
                sgamma apply shifted lognormal or shifted gamma approximations.
        :param sev_calc:   discrete = suitable for fft, continuous = for rv_histogram cts version
        :param discretization_calc: use survival, distribution or both (=max(cdf, sf)) which is most accurate calc
        :return:
        """

        r = 0
        self.xs = xs
        self.bs = xs[1]
        self.log2 = int(np.log(len(xs)) / np.log(2))
        tilt_vector = None

        # make the severity vector: a claim count weighted average of the severities
        if approximation == 'exact':
            wts = self.statistics_df.freq_1 / self.statistics_df.freq_1.sum()
            self.sev_density = np.zeros_like(xs)
            beds = self.discretize(sev_calc, discretization_calc, normalize)
            for temp, w, a, l, n in zip(beds, wts, self.attachment, self.limit, self.en):
                self.sev_density += temp * w

        if approximation == 'exact':
            if self.n == 0:
                # for dynamics it is helpful to have a zero risk return zero appropriately
                # z = ft(self.sev_density, padding, tilt_vector)
                self.agg_density = np.zeros_like(self.xs)
                self.agg_density[0] = 1
                # extreme idleness...but need to make sure it is the right shape and type
                self.ftagg_density = ft(self.agg_density, padding, tilt_vector)
            else:
                # usual calculation...this is where the magic happens!
                z = ft(self.sev_density, padding, tilt_vector)
                self.ftagg_density = self.mgf(self.n, z)
                self.agg_density = np.real(ift(self.ftagg_density, padding, tilt_vector))
        else:
            # regardless of request if skew == 0 have to use normal
            if self.agg_skew == 0:
                self.fzapprox = ss.norm(scale=self.agg_m * self.agg_cv, loc=self.agg_m)
            elif approximation == 'slognorm':
                shift, mu, sigma = sln_fit(self.agg_m, self.agg_cv, self.agg_skew)
                self.fzapprox = ss.lognorm(sigma, scale=np.exp(mu), loc=shift)
            elif approximation == 'sgamma':
                shift, alpha, theta = sgamma_fit(self.agg_m, self.agg_cv, self.agg_skew)
                self.fzapprox = ss.gamma(alpha, scale=theta, loc=shift)
            else:
                raise ValueError(f'Invalid approximation {approximation} option passed to CAgg density. '
                                 'Allowable options are: exact | slogorm | sgamma')

            ps = self.fzapprox.pdf(xs)
            self.agg_density = ps / np.sum(ps)
            self.ftagg_density = ft(self.agg_density, padding, tilt_vector)

        # invalidate stored functions
        self.nearest_quantile_function = None
        self._cdf = None
        self.verbose_audit_df = None

    def _apply_reins_work(self, reins_list, base_density, debug=False):
        """
        Actually do the work. Called by apply_reins and reins_audit_df.
        Only needs self to get limits, which it must guess without q (not computed
        at this stage). Does not need to know if occ or agg reins,
        only that the correct base_density is supplied.

        :param reins_list:
        :param kind: occ or agg, for debug plotting
        :param debug:
        :return: ceder, netter,
        """
        ans = make_ceder_netter(reins_list, debug)
        if debug:
            # debug xs and ys are the knot points of the interpolation function; good for plotting
            ceder, netter, xs, ys = ans
        else:
            ceder, netter = ans
        # assemble df for answers
        reins_df = pd.DataFrame({'loss': self.xs, 'p_subject': base_density,
                                 'F_subject': base_density.cumsum()}).set_index('loss', drop=False)
        reins_df['loss_net'] = netter(reins_df.loss)
        reins_df['loss_ceded'] = ceder(reins_df.loss)
        # summarized n and c
        sn = reins_df.groupby('loss_net').p_subject.sum()
        sc = reins_df.groupby('loss_ceded').p_subject.sum()
        # It can be that sn or sc has one row. For example, if the reinsurance cedes everything
        # then net is 0. That case must be handled separately.
        # -100: this value should never appear. use big value to make it obvious
        if len(sn) == 1:
            # net is a fixed value, need a step function
            loss = sn.index[0]
            value = sn.iloc[0]
            logger.warning(f'Only one net value at {loss} with prob = {value}')
            reins_df['F_net'] = 0.0
            reins_df.loc[loss:, 'F_net'] = value
        else:
            netter_interp = interp1d(sn.index, sn.cumsum(), fill_value=(-100, 1), bounds_error=False)
            reins_df['F_net'] = netter_interp(reins_df.loss)
        if len(sc) == 1:
            loss = sc.index[0]
            value = sc.iloc[0]
            logger.warning(f'Only one net value at {loss} with prob = {value}')
            reins_df['F_ceded'] = 0.0
            reins_df.loc[loss:, 'F_ceded'] = value
        else:
            ceder_interp = interp1d(sc.index, sc.cumsum(), fill_value=(-100, 1), bounds_error=False)
            reins_df['F_ceded'] = ceder_interp(reins_df.loss)
        reins_df['p_net'] = np.diff(reins_df.F_net, prepend=0)
        reins_df['p_ceded'] = np.diff(reins_df.F_ceded, prepend=0)

        if debug is False:
            return ceder, netter, reins_df

        logger.debug('making re graphs.')
        # quick debug; need to know kind=occ|agg here
        f = plt.figure(constrained_layout=True, figsize=(12, 9))
        axd = f.subplot_mosaic('AB\nCD')
        xlim = self.limits()
        # scale??
        x = np.linspace(0, xlim[1], 201)
        y = ceder(x)
        n = x - y
        nxs = netter(x)

        ax = axd['A']
        ax.plot(x, y, 'o')
        ax.plot(x, y)
        ax.plot(x, x, lw=.5, c='C7')
        ax.set(aspect='equal', xlim=xlim, ylim=xlim,
               xlabel='Subject', ylabel='Ceded',
               title=f'Subject and ceded\nMax ceded loss {y[-1]:,.1f}')

        ax = axd['B']
        ax.plot(x, nxs, 'o')
        ax.plot(x, n)
        ax.plot(x, x, lw=.5, c='C7')
        ax.set(aspect='equal', ylim=xlim,
               xlabel='Subject', ylabel='Net',
               title=f'Subject and net\nMax net loss {n[-1]:,.1f}')

        ax = axd['C']
        sn.cumsum().plot(ax=ax, lw=4, alpha=0.3, label='net')
        sc.cumsum().plot(ax=ax, lw=4, alpha=0.3, label='ceded')
        reins_df.filter(regex='F').plot(xlim=xlim, ax=ax)
        ax.set(title=f'Subject, net and ceded\ndistributions')
        ax.legend()

        ax = axd['D']
        reins_df.filter(regex='p_').plot(xlim=xlim, drawstyle='steps-post', ax=ax)
        ax.set(title=f'Subject, net and ceded\ndensities')
        ax.legend()

        return ceder, netter, reins_df

    def apply_occ_reins(self, debug=False):
        """
        Apply the entire occ reins structure and save output
        For by layer detail create reins_audit_df
        Makes sev_gross_density, sev_net_density and sev_ceded_density, and updates sev_density to the requested view.

        Treatment in stats?

        :return:
        """
        # generic function makes netter and ceder functions
        if self.occ_reins is None:
            return

        occ_ceder, occ_netter, occ_reins_df = self._apply_reins_work(self.occ_reins, self.sev_density, debug)
        # store stuff
        self.occ_reins_df = occ_reins_df
        self.sev_gross_density = self.sev_density
        self.sev_net_density = occ_reins_df['p_net']
        self.sev_ceded_density = occ_reins_df['p_ceded']
        if self.occ_kind == 'ceded to':
            self.sev_density = self.sev_ceded_density
        elif self.occ_kind == 'net of':
            self.sev_density = self.sev_net_density
        else:
            raise ValueError(f'Unexpected kind of occ reinsurace, {self.occ_kind}')

    def apply_agg_reins(self, debug=False, padding=1, tilt_vector=None):
        """
        Apply the entire agg reins structure and save output
        For by layer detail create reins_audit_df
        Makes sev_gross_density, sev_net_density and sev_ceded_density, and updates sev_density to the requested view.

        Treatment in stats?

        :return:
        """
        # generic function makes netter and ceder functions
        if self.agg_reins is None:
            return
        logger.info(f'Applying aggregate reinsurance for {self.name}')
        # aggregate moments (lose f x sev view) are computed after this step, so no adjustment needed there
        # agg: no way to make total = f x sev
        # initial empirical moments
        _m, _cv = xsden_to_meancv(self.xs, self.agg_density)

        agg_ceder, agg_netter, agg_reins_df = self._apply_reins_work(self.agg_reins, self.agg_density, debug)

        # store stuff
        self.agg_reins_df = agg_reins_df
        self.agg_gross_density = self.agg_density
        self.agg_net_density = agg_reins_df['p_net']
        self.agg_ceded_density = agg_reins_df['p_ceded']
        if self.agg_kind == 'ceded to':
            self.agg_density = self.agg_ceded_density
        elif self.agg_kind == 'net of':
            self.agg_density = self.agg_net_density
        else:
            raise ValueError(f'Unexpected kind of agg reinsurance, {self.agg_kind}')

        # update ft of agg
        self.ftagg_density = ft(self.agg_density, padding, tilt_vector)

        # see impact on moments
        _m2, _cv2 = xsden_to_meancv(self.xs, self.agg_density)
        # self.audit_df.loc['mixed', 'emp_agg_1'] = _m
        # old_m = self.ex
        self.ex = _m2
        # self.audit_df.loc['mixed', 'emp_agg_cv'] = _cv
        # invalidate quantile function

        logger.info(f'Applying agg reins to {self.name}\tOld mean and cv= {_m:,.3f}\t{_m:,.3f}\n'
                    f'New mean and cv = {_m2:,.3f}\t{_cv2:,.3f}')

    def reinsurance_description(self, kind='both'):
        """
        Text description of the reinsurance.

        :param kind: both, occ, or agg
        """
        ans = []
        if self.occ_reins is not None and kind in ['occ', 'both']:
            ans.append(self.occ_kind)
            ra = []
            for (s, y, a) in self.occ_reins:
                if np.isinf(y):
                    ra.append(f'{s:,.2%} share of unlimited xs {a:,.2f}')
                else:
                    if s == y:
                        ra.append(f'{y:,.2f} xs {a:,.2f}')
                    else:
                        ra.append(f'{s:,.2f} part of {y:,.2f} xs {a:,.2f}')
            ans.append(' and '.join(ra))
            ans.append('per occurrence')
        if self.agg_reins is not None and kind in ['agg', 'both']:
            if len(ans):
                ans.append('then')
            ans.append(self.agg_kind)
            ra = []
            for (s, y, a) in self.agg_reins:
                if np.isinf(y):
                    ra.append(f'{s:,.2%} share of unlimited xs {a:,.2f}')
                else:
                    if s == y:
                        ra.append(f'{y:,.2f} xs {a:,.2f}')
                    else:
                        ra.append(f'{s:,.2f} part of {y:,.2f} xs {a:,.2f}')
            ans.append(' and '.join(ra))
            ans.append('in the aggregate.')
        if len(ans):
            reins = 'Reinsurance: ' + ' '.join(ans)
        else:
            reins = 'Reinsurance: None'
        return reins

    def reinsurance_kinds(self):
        """
        Text desciption of kinds of reinsurance applied: None, Occurrence, Aggergate, both.

        :return:
        """
        n = 1 if  self.occ_reins is not None else 0
        n += 2 if  self.agg_reins is not None else 0
        if n == 0:
            return "None"
        elif n == 1:
            return 'Occurrence only'
        elif n == 2:
            return 'Aggregate only'
        else:
            return 'Occurrence and aggregate'

    def apply_distortion(self, dist):
        """
        apply distortion to the aggregate density and append as exag column to density_df
        TODO: implement original and revised calculation method
        :param dist:
        :return:
        """
        if self.agg_density is None:
            logger.warning(f'You must update before applying a distortion ')
            return

        S = self.density_df.S
        # some dist return np others don't this converts to numpy...
        gS = np.array(dist.g(S))

        self.density_df['gS'] = gS
        self.density_df['exag'] = np.hstack((0, gS[:-1])).cumsum() * self.bs

    def cramer_lundberg(self, rho, cap=0, excess=0, stop_loss=0, kind='index', padding=0):
        """
        Return the CL function relating surplus to eventual probability of ruin.

        Assumes frequency is Poisson

        rho = prem / loss - 1 is the margin-to-loss ratio

        cap = cap severity at cap - replace severity with X | X <= cap
        excess = replace severit with X | X > cap (i.e. no shifting)
        stop_loss = apply stop loss reinsurance to cap, so  X > stop_loss replaced with Pr(X > stop_loss) mass

        Embrechts, Kluppelberg, Mikosch 1.2 Page 28 Formula 1.11

        Pollaczeck-Khinchine Capital

        returns ruin vector as pd.Series and function to lookup (no interpolation if
        kind==index; else interp) capitals

        :param rho:
        :param cap:
        :param excess:
        :param stop_loss:
        :param kind:
        :param padding:
        :return:
        """

        if self.sev_density is None:
            raise ValueError("Must recalc before computing Cramer Lundberg distribution.")

        bit = self.density_df.p_sev.copy()
        if cap:
            idx = np.searchsorted(bit.index, cap, 'right')
            bit.iloc[idx:] = 0
            bit = bit / bit.sum()
        elif excess:
            # excess may not be in the index...
            idx = np.searchsorted(bit.index, excess, 'right')
            bit.iloc[:idx] = 0
            bit = bit / bit.sum()
        elif stop_loss:
            idx = np.searchsorted(bit.index, stop_loss, 'left')
            xsprob = bit.iloc[idx + 1:].sum()
            bit.iloc[idx] += xsprob
            bit.iloc[idx + 1:] = 0
        mean = np.sum(bit * bit.index)

        # integrated F function
        fi = bit.shift(-1, fill_value=0)[::-1].cumsum()[::-1].cumsum() * self.bs / mean
        # difference = probability density
        dfi = np.diff(fi, prepend=0)
        # use loc FFT, with wrapping
        fz = ft(dfi, padding, None)
        mfz = 1 / (1 - fz / (1 + rho))
        f = ift(mfz, padding, None)
        f = np.real(f) * rho / (1 + rho)
        f = np.cumsum(f)
        ruin = pd.Series(1 - f, index=bit.index)

        if kind == 'index':
            def find_u(p):
                idx = len(ruin) - ruin[::-1].searchsorted(p, 'left')
                return ruin.index[idx]
        else:
            def find_u(p):
                below = len(ruin) - ruin[::-1].searchsorted(p, 'left')
                above = below - 1
                q_below = ruin.index[below]
                q_above = ruin.index[above]
                p_below = ruin.iloc[below]
                p_above = ruin.iloc[above]
                q = q_below + (p - p_below) / (p_above - p_below) * (q_above - q_below)
                return q

        return ruin, find_u, mean, dfi  # , ruin2

    def delbaen_haezendonck_density(self, xs, padding, tilt_vector, beta, beta_name=""):
        """
        Compare the base and Delbaen Haezendonck transformed aggregates.

        * beta(x) = alpha + gamma(x)
        * alpha = log(freq' / freq): log of the increase in claim count
        * gamma = log(Radon Nikodym derv of adjusted severity) = log(tilde f / f)

        Adjustment guarantees a positive loading iff beta is an increasing function
        iff gamma is increasing iff tilde f / f is increasing.
        cf. eqn 3.7 and 3.8.

        Note conditions that E(exp(beta(X)) and E(X exp(beta(X)) must both be finite (3.4, 3.5)
        form of beta function described in 2.23 via, 2.16-17 and 2.18

        From examples on last page of paper: ::

            beta(x) = a ==> adjust frequency by factor of e^a
            beta(x) = log(1 + b(x - E(X)))  ==> variance principle EN(EX + bVar(X))
            beta(x) = ax- logE_P(exp(a x))  ==> Esscher principle

        To make a 'multiple' of an existing distortion you can use a simple wrapper class like this:

        ::

            class dist_wrap(agg.Distortion):
                '''
                wrap a distortion to include higher or lower freq
                in DH α is actually exp(α)
                this will pass isinstance(g2, agg.Distortion)
                '''
                def __init__(self, α, dist):
                    def loc_g(s):
                        return α * dist.g(s)
                    self.g = loc_g
                    self.name = dist.name

        :param xs: is part of agg so can use that
        :param padding: = 1 (default)
        :param tilt_vector: None (default)
        :param beta: function R+ to R with appropriate properties or name of prob distortion function
        :param beta_name:
        :return:
        """
        if self.agg_density is None:
            # update
            self.update_work(xs, padding, tilt_vector, 'exact')
        if isinstance(beta, Distortion):
            # passed in a distortion function
            beta_name = beta.name
            self.dh_sev_density = -np.diff(beta.g(1 - np.cumsum(np.hstack((0, self.sev_density)))))
            # ex_beta from Radon N derv, e^beta = dh / objective, so E[e^beta] = int dh/obj x obj = sum(dh)
            # which we expect to equal 1...hummm not adjusting the freq?!
            ex_beta = np.sum(self.dh_sev_density)
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

    def plot(self, axd=None, xmax=0, **kwargs):
        """
        New style basic plot with severity and aggregate, linear and log plots and Lee plot.

        :param xmax: Enter a "hint" for the xmax scale. E.g., if plotting gross and net you want all on
               the same scale. Only used on linear scales?
        :param axd:
        :param **kwargs: passed to make_mosaic_figure
        :return:
        """
        if axd is None:
            self.figure, axd = make_mosaic_figure('ABC', **kwargs)
        else:
            self.figure = axd['A'].figure

        if self.bs == 1 and self.ex < 1025:
            # treat as discrete
            if xmax > 0:
                mx = xmax
            else:
                mx = self.q(1) * 1.05
            span = nice_multiple(mx)

            df = self.density_df[['p_total', 'p_sev', 'F', 'loss']].copy()
            df['sevF'] = df.p_sev.cumsum()
            df.loc[-0.5, :] = (0, 0, 0, 0, 0)
            df = df.sort_index()
            if mx <= 60:
                # stem plot for small means
                axd['A'].stem(df.index, df.p_total, basefmt='none', linefmt='C0-', markerfmt='C0.', label='Aggregate')
                axd['A'].stem(df.index, df.p_sev,   basefmt='none', linefmt='C1-', markerfmt='C1,', label='Severity')
            else:
                df.p_total.plot(ax=axd['A'], drawstyle='steps-mid', lw=2, label='Aggregate')
                df.p_sev.plot(ax=axd['A'], drawstyle='steps-mid', lw=1, label='Severity')

            axd['A'].set(xlim=[-mx / 25, mx + 1], title='Probability mass functions')
            axd['A'].legend()
            if span > 0:
                axd['A'].xaxis.set_major_locator(ticker.MultipleLocator(span))
            # for discrete plot F next
            df.F.plot(ax=axd['B'], drawstyle='steps-post', lw=2, label='Aggregate')
            df.p_sev.cumsum().plot(ax=axd['B'], drawstyle='steps-post', lw=1, label='Severity')
            axd['B'].set(xlim=[-mx / 25, mx + 1], title='Distribution functions')
            axd['B'].legend().set(visible=False)
            if span > 0:
                axd['B'].xaxis.set_major_locator(ticker.MultipleLocator(span))

            # for Lee diagrams
            ax = axd['C']
            # trim so that the Lee plot doesn't spuriously tend up to infinity
            # little care: may not exaclty equal 1
            idx = (df.F == df.F.max()).idxmax()
            dft = df.loc[:idx]
            ax.plot(dft.F, dft.loss, drawstyle='steps-pre', lw=3, label='Aggregate')
            # same trim for severity
            df['sevF'] = df.p_sev.cumsum()
            idx = (df.sevF == 1).idxmax()
            df = df.loc[:idx]
            ax.plot(df.p_sev.cumsum(), df.loss, drawstyle='steps-pre', lw=1, label='Severity')
            ax.set(xlim=[-0.025, 1.025], ylim=[-mx / 25, mx + 1], title='Quantile (Lee) plot')
            ax.legend().set(visible=False)
        else:
            # continuous
            df = self.density_df
            if xmax > 0:
                xlim = [-xmax / 50, xmax * 1.025]
            else:
                xlim = self.limits(stat='range', kind='linear')
            xlim2 = self.limits(stat='range', kind='log')
            ylim = self.limits(stat='density')

            ax = axd['A']
            # divide by bucket size...approximating the density
            (df.p_total / self.bs).plot(ax=ax, lw=2, label='Aggregate')
            (df.p_sev / self.bs).plot(ax=ax, lw=1, label='Severity')
            ax.set(xlim=xlim, ylim=ylim, title='Probability density')
            ax.legend()

            (df.p_total / self.bs).plot(ax=axd['B'], lw=2, label='Aggregate')
            (df.p_sev / self.bs).plot(ax=axd['B'], lw=1, label='Severity')
            ylim = axd['B'].get_ylim()
            ylim = [1e-15, ylim[1]*2]
            axd['B'].set(xlim=xlim2, ylim=ylim, title='Log density', yscale='log')
            axd['B'].legend().set(visible=False)

            ax = axd['C']
            # to do: same trimming for p-->1 needed?
            ax.plot(df.F, df.loss, lw=2, label='Aggregate')
            ax.plot(df.p_sev.cumsum(), df.loss, lw=1, label='Severity')
            ax.set(xlim=[-0.02, 1.02], ylim=xlim, title='Quantile (Lee) plot', xlabel='Non-exceeding probability p')
            ax.legend().set(visible=False)

    def plot_old(self, kind='quick', axiter=None, aspect=1, figsize=(10, 3)):
        """
        Plot computed density and aggregate. To be removed!

        **kind** option:

        * quick (default): Density for sev and agg on nominal and log scale; Lee diagram sev and agg
        * long: severity, log sev density, sev dist, agg with sev, agg on own, agg on log, S, Lee, return period

        :param kind: quick or long
        :param axiter: optional axiter object
        :param aspect: optional aspect ratio of individual plots
        :param figsize: optional overall figure size
        :return:
        """

        if self.agg_density is None:
            raise ValueError('Cannot plot before update')
            return
        if self.sev_density is None:
            self.update_work(self.xs, 1, None, sev_calc='discrete', force_severity='yes')

        set_tight = (axiter is None)

        if kind == 'long':
            axiter = axiter_factory(axiter, 10, aspect=aspect, figsize=figsize)

            max_lim = min(self.xs[-1], np.max(self.limit)) * 1.05
            if max_lim < 1: max_lim = 1

            next(axiter).plot(self.xs, self.sev_density)  # , drawstyle='steps-post')
            axiter.ax.set(title='Severity', xlim=(0, max_lim))

            next(axiter).plot(self.xs, self.sev_density)
            axiter.ax.set(title='Log Severity')
            if np.sum(self.sev_density == 1) >= 1:
                # sev density is degenerate, 1,0,0,... log scales won't work
                axiter.ax.set(title='Severity Degenerate')
                axiter.ax.set(xlim=(0, max_lim * 2))
            else:
                axiter.ax.set(title='Log Severity')
                axiter.ax.set(title='Log Severity', yscale='log')
                axiter.ax.set(xlim=(0, max_lim))

            next(axiter).plot(self.xs, self.sev_density.cumsum(), drawstyle='steps-post')
            axiter.ax.set(title='Severity Distribution')
            axiter.ax.set(xlim=(0, max_lim))

            next(axiter).plot(self.xs, self.agg_density, label='aggregate_project')
            axiter.ax.plot(self.xs, self.sev_density, lw=0.5, drawstyle='steps-post', label='severity')
            axiter.ax.set(title='Aggregate')
            axiter.ax.legend()

            next(axiter).plot(self.xs, self.agg_density, label='aggregate_project')
            axiter.ax.set(title='Aggregate')

            next(axiter).plot(self.xs, self.agg_density, label='aggregate_project')
            axiter.ax.set(yscale='log', title='Aggregate, log scale')

            F = self.agg_density.cumsum()
            next(axiter).plot(self.xs, 1 - F)
            axiter.ax.set(title='Survival Function')

            next(axiter).plot(self.xs, 1 - F)
            axiter.ax.set(title='Survival Function, log scale', yscale='log')

            next(axiter).plot(1 - F, self.xs, label='aggregate_project')
            axiter.ax.plot(1 - self.sev_density.cumsum(), self.xs, label='severity')
            axiter.ax.set(title='Lee Diagram')
            axiter.ax.legend()

            # figure for extended plotting of return period:
            max_p = F[-1]
            if max_p > 0.9999:
                _n = 10
            else:
                _n = 5
            if max_p >= 1:
                max_p = 1 - 1e-10
            k = (max_p / 0.99) ** (1 / _n)
            extraps = 0.99 * k ** np.arange(_n)
            q = interpolate.interp1d(F, self.xs, kind='linear', fill_value='extrapolate', bounds_error=False)
            ps = np.hstack((np.linspace(0, 1, 100, endpoint=False), extraps))
            qs = q(ps)
            next(axiter).plot(1 / (1 - ps), qs)
            axiter.ax.set(title='Return Period', xscale='log')

        elif kind == 'quick':
            if self.dh_agg_density is not None:
                n = 4
            else:
                n = 3

            axiter = axiter_factory(axiter, n, figsize, aspect=aspect)

            F = np.cumsum(self.agg_density)
            mx = np.argmax(F > 1 - 1e-5)
            if mx == 0:
                mx = len(F) + 1
            else:
                mx += 1  # a little extra room
            dh_F = None
            if self.dh_agg_density is not None:
                dh_F = np.cumsum(self.dh_agg_density)
                mx = max(mx, np.argmax(dh_F > 1 - 1e-5))
                dh_F = dh_F[:mx]
            F = F[:mx]

            xs = self.xs[:mx]
            d = self.agg_density[:mx]
            sevF = np.cumsum(self.sev_density)
            sevF = sevF[:mx]
            f = self.sev_density[:mx]

            ax = next(axiter)
            # ? correct format?
            ax.plot(xs, d, label='agg', drawstyle='steps-post')
            ax.plot(xs, f, label='sev', drawstyle='steps-post')
            if np.sum(f > 1e-6) < 20:
                # if there are few points...highlight the points
                ax.plot(xs, f, 'o', label=None, )
            if self.dh_agg_density is not None:
                ax.plot(xs, self.dh_agg_density[:mx], label='dh {:} agg'.format(self.beta_name))
                ax.plot(xs, self.dh_sev_density[:mx], label='dh {:} sev'.format(self.beta_name))
            max_y = min(2 * np.max(d), np.max(f[1:])) * 1.05  # want some extra space...
            if max_y > 0:
                ax.set_ylim(0, max_y)
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
                axiter.tidy()
                suptitle_and_tight(f'Aggregate {self.name}')

        else:
            raise ValueError(f'Unknown option to plot_old, kind={kind}')

    def limits(self, stat='range', kind='linear', zero_mass='include'):
        """
        Suggest sensible plotting limits for kind=range, density, etc., same as Portfolio.

        Should optionally return a locator for plots?

        Called by ploting routines. Single point of failure!

        Must work without q function when not computed (apply_reins_work for
        occ reins; then use report_ser instead).

        :param stat:  range or density (for y axis)
        :param kind:  linear or log (this is the y-axis, not log of range...that is rarely plotted)
        :param zero_mass:  include exclude, for densities
        :return:
        """

        # fudge l/r factors
        def f(x):
            fl, fr = 0.02, 1.02
            return [-fl * x, fr * x]

        # lower bound for log plots
        eps = 1e-16

        # if not computed
        # GOTCHA: if you call q and it fails because not agg_density then q is set to {}
        # which is not None
        if self.agg_density is None:
            return f(self.report_ser[('agg', 'P99.9e')])

        if stat == 'range':
            if kind == 'linear':
                return f(self.q(0.999))
            else:
                # wider range for log density plots
                return f(self.q(0.99999))

        elif stat == 'density':
            # for density need to divide by bs
            mx = self.agg_density.max() / self.bs
            mxx0 = self.agg_density[1:].max() / self.bs
            if kind == 'linear':
                if zero_mass == 'include':
                    return f(mx)
                else:
                    return f(mxx0)
            else:
                return [eps, mx * 1.5]
        else:
            # if you fall through to here, wrong args
            raise ValueError(f'Inadmissible stat/kind passsed, expected range/density and log/linear.')

    @property
    def report_df(self):
        """
        Created on the fly report to audit creation of object.
        There were some bad choices of columns in audit_df...but it [maybe] embedded in other code....
        Eg the use of _1 vs _m for mean is inconsistent.

        :return:
        """

        if self.audit_df is not None:
            # want both mixed and unmixed
            cols = ['name', 'limit', 'attachment', 'el', 'freq_m', 'freq_cv', 'freq_skew',
                    'sev_m', 'sev_cv', 'sev_skew', 'agg_m', 'agg_cv', 'agg_skew']
            # massaged version of original audit_df, including indep and mixed total views
            df = pd.concat((self.statistics_df[cols], self.statistics_total_df[cols]), axis=0).T
            df['empirical'] = np.nan
            # add empirical stats
            df.loc['sev_m', 'empirical'] = self.audit_df.loc['mixed', 'emp_sev_1']
            df.loc['sev_cv', 'empirical'] = self.audit_df.loc['mixed', 'emp_sev_cv']
            df.loc['sev_skew', 'empirical'] = self.audit_df.loc['mixed', 'emp_sev_skew']
            df.loc['agg_m', 'empirical']  = self.audit_df.loc['mixed', 'emp_agg_1']
            df.loc['agg_cv', 'empirical'] = self.audit_df.loc['mixed', 'emp_agg_cv']
            df.loc['agg_skew', 'empirical'] = self.audit_df.loc['mixed', 'emp_agg_skew']
            df = df
            df['error'] = df['empirical'] / df['mixed'] - 1
            df = df.fillna('')
            # better column order  units; indep sum; mixed sum; empirical; error
            c = list(df.columns)
            c = c[:-4] + [c[-3], c[-4], c[-2], c[-1]]
            df = df[c]
            if df.shape[1] == 4:
                # only one sev, don't show extra sev column
                df = df.iloc[:, 1:]
            df.index.name = 'statistic'
            df.columns.name = 'view'
            return df

    @property
    def statistics(self):
        """
        Pandas series of theoretic frequency, severity, and aggregate 1st, 2nd, and 3rd moments.
        Mean, cv, and skewness.

        :return:
        """
        if len(self.statistics_df) > 1:
            # there are mixture components
            df = pd.concat((self.statistics_df, self.statistics_total_df), axis=0)
        else:
            df = self.statistics_df.copy()
        # edit to make equivalent to Portfolio statistics
        df = df.T
        if df.shape[1] == 1:
            df.columns = [df.iloc[0,0]]
            df = df.iloc[1:]
        df.index = df.index.str.split("_", expand=True, )
        df = df.rename(index={'1': 'ex1', '2': 'ex2', '3': 'ex3', 'm': 'mean', np.nan: ''})
        df.index.names =['component', 'measure']
        df.columns.name = 'name'
        return df

    @property
    def describe(self):
        """
        Theoretic and empirical stats. Used in _repr_html_.

        """
        st = self.statistics_total_df.loc['mixed', :]
        sev_m = st.sev_m
        sev_cv = st.sev_cv
        sev_skew = st.sev_skew
        n_m = st.freq_m
        n_cv = st.freq_cv
        a_m = st.agg_m
        a_cv = st.agg_cv
        df = pd.DataFrame({'E[X]': [sev_m, n_m, a_m], 'CV(X)': [sev_cv, n_cv, a_cv],
                           'Skew(X)': [sev_skew, self.statistics_total_df.loc['mixed', 'freq_skew'], st.agg_skew]},
                          index=['Sev', 'Freq', 'Agg'])
        df.index.name = 'X'
        if self.audit_df is not None:
            esev_m = self.audit_df.loc['mixed', 'emp_sev_1']
            esev_cv = self.audit_df.loc['mixed', 'emp_sev_cv']
            ea_m = self.audit_df.loc['mixed', 'emp_agg_1']
            ea_cv = self.audit_df.loc['mixed', 'emp_agg_cv']
            df.loc['Sev', 'Est E[X]'] = esev_m
            df.loc['Agg', 'Est E[X]'] = ea_m
            df.loc[:, 'Err E[X]'] = df['Est E[X]'] / df['E[X]'] - 1
            df.loc['Sev', 'Est CV(X)'] = esev_cv
            df.loc['Agg', 'Est CV(X)'] = ea_cv
            df.loc[:, 'Err CV(X)'] = df['Est CV(X)'] / df['CV(X)'] - 1
            df = df[['E[X]', 'Est E[X]', 'Err E[X]', 'CV(X)', 'Est CV(X)', 'Err CV(X)', 'Skew(X)']]
        df = df.fillna('')
        df = df.loc[['Freq', 'Sev', 'Agg']]
        return df

    def recommend_bucket(self, log2=10, verbose=False):
        """
        Recommend a bucket size given 2**N buckets. Not rounded.

        :param log2: log2 of number of buckets. log2=10 is default.
        :return:
        """
        N = 1 << log2
        if not verbose:
            moment_est = estimate_agg_percentile(self.agg_m, self.agg_cv, self.agg_skew) / N
            limit_est = self.limit.max() / N
            if limit_est == np.inf:
                limit_est = 0
            logger.debug(f'Agg.recommend_bucket | {self.name} moment: {moment_est}, limit {limit_est}')
            return max(moment_est, limit_est)
        else:
            for n in sorted({log2, 16, 13, 10}):
                rb = self.recommend_bucket(n)
                if n == log2:
                    rbr = rb
                print(f'Recommended bucket size with {2 ** n} buckets: {rb:,.0f}')
            if self.bs != 0:
                print(f'Bucket size set with {N} buckets at {self.bs:,.0f}')
            return rbr

    # def q_old(self, p):
    #     """
    #     Return lowest quantile, appropriate for discrete bucketing.
    #     quantile guaranteed to be in the index
    #     nearest does not work because you always want to pick rounding up
    #
    #     Definition 2.1 (Quantiles)
    #
    #     :math:`x(α) = qα(X) = inf\{x ∈ R : P[X ≤ x] ≥ α\}` is the lower α-quantile of X
    #
    #     :math:`x(α) = qα(X) = inf\{x ∈ R : P[X ≤ x] > α\}` is the upper α-quantile of X.
    #
    #     We use the x-notation if the dependence on X is evident, otherwise the q-notion.
    #     Acerbi and Tasche (2002)
    #
    #     :param p:
    #     :return:
    #     """
    #     if self._q is None:
    #         self._q = interpolate.interp1d(self.density_df.F, self.density_df.loss, kind='linear')
    #     l = float(self._q(p))
    #     # find next nearest index value if not an exact match (this is slightly faster and more robust
    #     # than l/bs related math)
    #     l1 = self.density_df.index.get_loc(l, 'bfill')
    #     l1 = self.density_df.index[l1]
    #     return l1

    def q(self, p, kind='lower'):
        """
        Compute quantile, returning element in the index. Exact same code from Portfolio.q.

        :param p:
        :param kind: lower, middle reproduces middle_q, upper
        :return:
        """
        if self._linear_quantile_function is None:
            # revised Dec 2019
            try:
                self._linear_quantile_function = {}
                self.q_temp = self.density_df[['loss', 'F']].groupby('F').agg({'loss': np.min})
                self.q_temp.loc[1, 'loss'] = self.q_temp.loss.iloc[-1]
                self.q_temp.loc[0, 'loss'] = 0
                self.q_temp = self.q_temp.sort_index()
                # that q_temp left cts, want right continuous:
                self.q_temp['loss_s'] = self.q_temp.loss.shift(-1)
                self.q_temp.iloc[-1, 1] = self.q_temp.iloc[-1, 0]
                self._linear_quantile_function['upper'] = \
                    interpolate.interp1d(self.q_temp.index, self.q_temp.loss_s, kind='previous', bounds_error=False,
                                         fill_value='extrapolate')
                # Jan 2020 see note in Portfolio: changed previous to next
                self._linear_quantile_function['lower'] = \
                    interpolate.interp1d(self.q_temp.index, self.q_temp.loss, kind='next', bounds_error=False,
                                         fill_value='extrapolate')
                # changed to loss_s
                self._linear_quantile_function['middle'] = \
                    interpolate.interp1d(self.q_temp.index, self.q_temp.loss_s, kind='linear', bounds_error=False,
                                         fill_value='extrapolate')
            except Exception as e:
                # if fails reset in case this code is within a try .... except block
                self._linear_quantile_function = None
                raise e
        l = float(self._linear_quantile_function[kind](p))
        # because we are not interpolating the returned value must (should) be in the index...
        if not (kind == 'middle' or l in self.density_df.index):
            logger.error(f'Unexpected weirdness in {self.name} quantile...computed {p}th {kind} percentile as {l} '
                         'which is not in the index but is expected to be. Make sure bs has nice binary expansion!')
        return l

    # def careful_q(self, p):
    #     """
    #     Careful calculation of q handling jumps (based of SRM_Examples Noise class originally).
    #     Note this is automatically vectorized and returns and array whereas q isn't.
    #     It doesn't necessarily return an element of the index.
    #
    #     Just for reference here is code to illustrate the problem. This code is used in Vig_0_Audit.ipynb. ::
    #
    #         uw = agg.Underwriter()
    #
    #         def plot_eg_agg(b, e, w, n=32, axs=None, x_range=1):
    #             '''
    #             makes a tricky distribution function with a poss isolated jump
    #             creates an agg object and checks the quantile function is correct
    #
    #             mass at w
    #
    #             '''
    #
    #             if axs is None:
    #                 f, axs0 = plt.subplots(2,3, figsize=(9,6))
    #                 axs = iter(axs0.flatten())
    #
    #             tm = np.linspace(0, 1, 33)
    #             tf = lambda x : f'{32*x:.0f}'
    #
    #             def pretty(axis, ticks, formatter):
    #                 maj = ticks[::4]
    #                 mnr = [i for i in ticks if i not in maj]
    #                 labels = [formatter(i) for i in maj]
    #                 axis.set_ticks(maj)
    #                 axis.set_ticks(mnr, True)
    #                 axis.set_ticklabels(labels)
    #                 axis.grid(True, 'major', lw=0.707, c='lightblue')
    #                 axis.grid(True, 'minor', lw=0.35, c='lightblue')
    #
    #             # make the distribution
    #             xs = np.linspace(0, x_range, n+1)
    #             Fx = np.zeros_like(xs)
    #             Fx[b:13] = 1
    #             Fx[20:e] = 1
    #             Fx[w] = 32 - np.sum(Fx)
    #             Fx = Fx / Fx.sum()
    #             Fx = np.cumsum(Fx)
    #
    #             # make an agg version: find the jumps and create a dhistogram
    #             temp = pd.DataFrame(dict(x=xs, F=Fx))
    #             temp['f'] = np.diff(temp.F, prepend=0)
    #             temp = temp.query('f > 0')
    #             pgm = f'agg Tricky 1 claim sev dhistogram xps {temp.x.values} {temp.f.values} fixed'
    #             a = uw(pgm)
    #             a.easy_update(10, 0.001)
    #             # plot
    #             a.plot(axiter=axs)
    #             pretty(axs0[0,0].xaxis, tm, tf)
    #             pretty(axs0[0,2].xaxis, tm, tf)
    #             pretty(axs0[0,2].yaxis, tm, tf)
    #
    #             # lower left plot: distribution function
    #             ax = next(axs)
    #             ax.step(xs, Fx, where='post', marker='.')
    #             ax.plot(a.xs, a.agg_density.cumsum(), linewidth=3, alpha=0.5, label='from agg')
    #             ax.set(title=f'b={b}, e={e}, w={w}', ylim=-0.05, aspect='equal')
    #             if x_range  == 1:
    #                 ax.set(aspect='equal')
    #             ax.legend(frameon=False, loc='upper left')
    #             pretty(ax.xaxis, tm, tf)
    #             pretty(ax.yaxis, tm, tf)
    #
    #             # lower middle plot
    #             ps = np.linspace(0, 1, 301)
    #             agg_careful = a.careful_q(ps)
    #             ax = next(axs)
    #             ax.step(Fx, xs, where='pre', marker='.', label='input')
    #             ax.plot(Fx, xs, ':', label='input joined')
    #             ax.plot(ps, agg_careful, linewidth=1, label='agg careful')
    #             ax.set(title='Inverse', ylim=-0.05)
    #             if x_range  == 1:
    #                 ax.set(aspect='equal')
    #             pretty(ax.xaxis, tm, tf)
    #             pretty(ax.yaxis, tm, tf)
    #             ax.legend()
    #
    #             # lower right plot
    #             ax = next(axs)
    #             dmq = np.zeros_like(ps)
    #             for i, p in enumerate(ps):
    #                 try:
    #                     dmq[i] = a.q(p)
    #                 except:
    #                     dmq[i] = 0
    #             ax.plot(ps, agg_careful, label='careful (agg obj)', linewidth=1, alpha=1)
    #             ax.plot(ps, dmq, label='agg version')
    #             ax.legend(frameon=False, loc='upper left')
    #             pretty(ax.xaxis, tm, tf)
    #             pretty(ax.yaxis, tm, tf)
    #             ax.set(title='Check with agg version')
    #
    #             plt.tight_layout()
    #
    #             return a
    #
    #         aw = plot_eg_agg(6, 29, 16)
    #
    #     :param p: single or vector of values of ps, 0<1
    #     :return:  quantiles
    #     """
    #     if self._careful_q is None:
    #         self._careful_q = CarefulInverse.dist_inv1d(self.xs, self.agg_density)
    #
    #     return self._careful_q(p)

    def tvar(self, p, kind='interp'):
        """
        Compute the tail value at risk at threshold p

        Definition 2.6 (Tail mean and Expected Shortfall)
        Assume E[X−] < ∞. Then
        x¯(α) = TM_α(X) = α^{−1}E[X 1{X≤x(α)}] + x(α) (α − P[X ≤ x(α)])
        is α-tail mean at level α the of X.
        Acerbi and Tasche (2002)

        We are interested in the right hand exceedence [?? note > vs ≥]
        α^{−1}E[X 1{X > x(α)}] + x(α) (P[X ≤ x(α)] − α)

        McNeil etc. p66-70 - this follows from def of ES as an integral
        of the quantile function

        q is exact quantile (most of the time)
        q1 is the smallest index element (bucket multiple) greater than or equal to q

        tvar integral is int_p^1 q(s)ds = int_q^infty xf(x)dx = q + int_q^infty S(x)dx
        we use the last approach. np.trapz approxes the integral. And the missing piece
        between q and q1 approx as a trapezoid too.

        :param p:
        :param kind:
        :return:
        """
        # match Portfolio method
        assert self.density_df is not None

        if kind == 'tail':
            # original
            # _var = self.q(p)
            # ex = self.density_df.loc[_var + self.bs:, ['p_total', 'loss']].product(axis=1).sum()
            # pip = (self.density_df.loc[_var, 'F'] - p) * _var
            # t_var_old = 1 / (1 - p) * (ex + pip)
            # revised
            if self._tail_var2 is None:
                self._tail_var2 = self.density_df[['p_total', 'loss']].product(axis=1).iloc[::-1].cumsum().iloc[::-1]
            _var = self.q(p)
            ex = self._tail_var2.loc[_var + self.bs]
            pip = (self.density_df.loc[_var, 'F'] - p) * _var
            t_var = 1 / (1 - p) * (ex + pip)
            return t_var
        elif kind == 'interp':
            # original implementation interpolated
            if self._tail_var is None:
                # make tvar function
                sup = (self.density_df.p_total[::-1] > 0).idxmax()
                if sup == self.density_df.index[-1]:
                    sup = np.inf
                    _x = self.density_df.F
                    _y = self.density_df.exgta
                else:
                    _x = self.density_df.F.values[:self.density_df.index.get_loc(sup)]
                    _y = self.density_df.exgta.values[:self.density_df.index.get_loc(sup)]
                p0 = self.density_df.at[0.0, 'F']
                if p0 > 0:
                    ps = np.linspace(0, p0, 200, endpoint=False)
                    tempx = np.hstack((ps, _x))
                    tempy = np.hstack((self.ex / (1 - ps), _y))
                    self._tail_var = interpolate.interp1d(tempx, tempy,
                                                          kind='linear', bounds_error=False,
                                                          fill_value=(self.ex, sup))
                else:
                    self._tail_var = interpolate.interp1d(_x, _y, kind='linear', bounds_error=False,
                                                          fill_value=(self.ex, sup))
            if type(p) in [float, np.float]:
                return float(self._tail_var(p))
            else:
                return self._tail_var(p)
        elif kind == 'inverse':
            if self._inverse_tail_var is None:
                # make tvar function
                self._inverse_tail_var = interpolate.interp1d(self.density_df.exgta, self.density_df.F,
                                                              kind='linear', bounds_error=False,
                                                              fill_value='extrapolate')
            if type(p) in [int, np.int, float, np.float]:
                return float(self._inverse_tail_var(p))
            else:
                return self._inverse_tail_var(p)
        else:
            raise ValueError(f'Inadmissible kind passed to tvar; options are interp (default) or tail')

        # original version
        # function not vectorized
        # q = float(self.q(p, 'middle'))
        # l1 = self.density_df.index.get_loc(q, 'bfill')
        # q1 = self.density_df.index[l1]
        #
        # i1 = np.trapz(self.density_df.loc[q1:, 'S'], dx=self.bs)
        # i2 = (q1 - q) * (2 - p - self.density_df.at[q1, 'F']) / 2  # trapz adj for first part
        # return q + (i1 + i2) / (1 - p)

    def sev_cdf(self, x, verbose=False):
        """
        Direct access to the underlying severity, exact computation.

        """
        ans = []
        F = 0
        for s in self.sevs:
            w = s.sev_wt
            c = s.cdf(x)
            F += w * c
            ans.append([s.sev_name, c, w])

        if verbose is True:
            return F, pd.DataFrame(ans, columns=['name', 'cdf', 'wt'])
        else:
            return F

    def cdf(self, x, kind='previous'):
        """
        Return cumulative probability distribution at x using kind interpolation.

        2022-10 change: kind introduced; default was linear

        :param x: loss size
        :return:
        """
        if self._cdf is None:
            self._cdf = interpolate.interp1d(self.xs, self.agg_density.cumsum(), kind=kind,
                                             bounds_error=False, fill_value='extrapolate')
        # 0+ converts to float
        return 0. + self._cdf(x)

    def sf(self, x):
        """
        Return survival function using linear interpolation.

        :param x: loss size
        :return:
        """
        return 1 - self.cdf(x)

    def pdf(self, x):
        """
        Probability density function, assuming a continuous approximation of the bucketed density.

        :param x:
        :return:
        """
        if self._pdf is None:
            self._pdf = interpolate.interp1d(self.xs, self.agg_density, kind='linear',
                                             bounds_error=False, fill_value='extrapolate')
        return self._pdf(x) / self.bs

    def pmf(self, x):
        """
        Probability mass function, treating aggregate as discrete
        x must be in the index (?)

        """
        if self.density_df is None:
            raise ValueError("Must update before computing probabilities!")

        try:
            return self.density_df.loc[x, 'p_total']
        except KeyError:
            return 0.0
            # raise KeyError(f'Value {x} must be in index for probability mass function.')

    def json(self):
        """
        Write spec to json string.

        :return:
        """
        return json.dumps(self._spec)

    def approximate(self, approx_type='slognorm', output='scipy'):
        """
        Create an approximation to self using method of moments matching.

        Compare to Portfolio.approximate which returns a single sev fixed freq agg, this
        returns a scipy dist by default.

        Use case: exam questions with the normal approacimation!

        :param approx_type: norm, lognorn, slognorm (shifted lognormal), gamma, sgamma. If 'all'
        then returns a dictionary of each approx.
        :param output: scipy - returns a frozen scipy.stats object; agg returns an Aggregate program;
        any other value returns the program and created aggregate object with fixed frequency equal to 1.
        :return: as above.
        """

        if approx_type == 'all':
            return {kind: self.approximate(kind)
                    for kind in ['norm', 'gamma', 'lognorm', 'sgamma', 'slognorm']}

        if self.audit_df is None:
            # not updated
            m = self.statistics_total_df.loc['mixed', 'agg_m']
            cv = self.statistics_total_df.loc['mixed', 'agg_cv']
            skew = self.statistics_total_df.loc['mixed', 'agg_skew']
        else:
            # use statistics_df matched to computed aggregate_project
            m, cv, skew = self.report_df.loc[['agg_m', 'agg_cv', 'agg_skew'], 'empirical']

        name = f'{approx_type[0:4]}.{self.name[0:5]}'
        agg_str = f'agg {name} 1 claim sev '

        if approx_type == 'norm':
            sd = m*cv
            if output=='scipy':
                return ss.norm(loc=m, scale=sd)
            sev = {'sev_name': 'norm', 'sev_scale': sd, 'sev_loc': m}
            agg_str += f'{sd} @ norm 1 # {m} '

        elif approx_type == 'lognorm':
            mu, sigma = mu_sigma_from_mean_cv(m, cv)
            sev = {'sev_name': 'lognorm', 'sev_shape': sigma, 'sev_scale': np.exp(mu)}
            if output=='scipy':
                return ss.lognorm(sigma, scale=np.exp(mu-sigma**2/2))
            agg_str += f'{np.exp(mu)} * lognorm {sigma} '

        elif approx_type == 'gamma':
            shape = cv ** -2
            scale = m / shape
            if output=='scipy':
                return ss.gamma(shape, scale=scale)
            sev = {'sev_name': 'gamma', 'sev_a': shape, 'sev_scale': scale}
            agg_str += f'{scale} * gamma {shape} '

        elif approx_type == 'slognorm':
            shift, mu, sigma = sln_fit(m, cv, skew)
            if output=='scipy':
                return ss.lognorm(sigma, scale=np.exp(mu-sigma**2/2), loc=shift)
            sev = {'sev_name': 'lognorm', 'sev_shape': sigma, 'sev_scale': np.exp(mu), 'sev_loc': shift}
            agg_str += f'{np.exp(mu)} * lognorm {sigma} + {shift} '

        elif approx_type == 'sgamma':
            shift, alpha, theta = sgamma_fit(m, cv, skew)
            if output=='scipy':
                return ss.gamma(alpha, loc=shift, scale=theta)
            sev = {'sev_name': 'gamma', 'sev_a': alpha, 'sev_scale': theta, 'sev_loc': shift}
            agg_str += f'{theta} * gamma {alpha} + {shift} '

        else:
            raise ValueError(f'Inadmissible approx_type {approx_type} passed to fit')

        if output == 'agg':
            agg_str += ' fixed'
            return agg_str
        else:
            return Aggregate(**{'name': name, 'note': f'frozen version of {self.name}',
                                'exp_en': 1, **sev, 'freq_name': 'fixed'})

    fit = approximate

    def entropy_fit(self, n_moments, tol=1e-10, verbose=False):
        """
        Find the max entropy fit to the aggregate based on n_moments fit.
        The constant is added (sum of probabilities constraint), for two
        moments there are n_const = 3 constrains.

        Based on discussions with, and R code from, Jon Evans

        Run ::

            ans = obj.entropy_fit(2)
            ans['ans_df'].plot()

        to compare the fits.

        :param n_moments: number of moments to match
        :param tol:
        :param verbose:
        :return:
        """
        # sum of probs constraint
        n_constraints = n_moments + 1

        # don't want to mess up the object...
        xs = self.xs.copy()
        p = self.agg_density.copy()
        # more aggressively de-fuzz
        p = np.where(abs(p) < 1e-16, 0, p)
        p = p / np.sum(p)
        p1 = p.copy()

        mtargets = np.zeros(n_constraints)
        for i in range(n_constraints):
            mtargets[i] = np.sum(p)
            p *= xs

        parm1 = np.zeros(n_constraints)
        x = np.array([xs ** i for i in range(n_constraints)])

        probs = np.exp(-x.T @ parm1)
        machieved = x @ probs
        der1 = -(x * probs) @ x.T

        er = 1
        iters = 0
        while er > tol:
            iters += 1
            try:
                parm1 = parm1 - inv(der1) @ (machieved - mtargets)
            except np.linalg.LinAlgError:
                print('Singluar matrix')
                print(der1)
                return None
            probs = np.exp(-x.T @ parm1)
            machieved = x @ probs
            der1 = -(x * probs) @ x.T
            er = (machieved - mtargets).dot(machieved - mtargets)
            if verbose:
                print(f'Error: {er}\nParameter {parm1}')
        ans = pd.DataFrame(dict(xs=xs, agg=p1, fit=probs))
        ans = ans.set_index('xs')
        return dict(params=parm1, machieved=machieved, mtargets=mtargets, ans_df=ans)

    def var_dict(self, p, kind='lower', snap=False):
        """
        Make a dictionary of value at risks for the line, mirrors Portfolio.var_dict.
        Here is just marshals calls to the appropriate var or tvar function.

        No epd. Allows the price function to run consistently with Portfolio version.

        Example Use: ::

            for p, arg in zip([.996, .996, .996, .985, .01], ['var', 'lower', 'upper', 'tvar', 'epd']):
                print(port.var_dict(p, arg,  snap=True))

        :param p:
        :param kind: var (defaults to lower), upper, lower, tvar
        :param snap: snap tvars to index
        :return:
        """
        if kind == 'var': kind = 'lower'
        if kind == 'tvar':
            d = {self.name: self.tvar(p)}
        else:
            d = {self.name: self.q(p, kind)}
        if snap and kind == 'tvar':
            d = {self.name: self.snap(d[self.name])}
        return d

    def price(self, p, g, kind='var'):
        """
        Price using regulatory and pricing g functions, mirroring Portfolio.price.
        Unlike Portfolio, cannot calibrate. Applying specified Distortions only.
        If calibration is needed, embed Aggregate in a one-line Portfolio object.

        Compute E_price (X wedge E_reg(X) ) where E_price uses the pricing distortion and E_reg uses
        the regulatory distortion.

        Regulatory capital distortion is applied on unlimited basis: ``reg_g`` can be:

        * if input < 1 it is a number interpreted as a p value and used to determine VaR capital
        * if input > 1 it is a directly input  capital number
        * d dictionary: Distortion; spec { name = dist name | var, shape=p value a distortion used directly

        ``pricing_g`` is  { name = ph|wang and shape=}, if shape (lr or roe not allowed; require calibration).

        if ly, must include ro in spec

        :param p: a distortion function spec or just a number; if >1 assets, if <1 a prob converted to quantile
        :param kind: var lower upper tvar
        :param g:  pricing distortion function
        :return:
        """

        # figure regulatory assets; applied to unlimited losses
        vd = self.var_dict(p, kind, snap=True)
        a_reg = vd[self.name]

        # figure pricing distortion
        if isinstance(g, Distortion):
            # just use it
            pass
        else:
            # Distortion spec as dict
            g = Distortion(**g)

        self.apply_distortion(g)
        aug_row = self.density_df.loc[a_reg]

        # holder for the answer
        df = pd.DataFrame(columns=['line', 'L', 'P', 'M', 'Q'], dtype=float)
        df.columns.name = 'statistic'
        df = df.set_index('line', drop=True)

        el = aug_row['exa']
        P = aug_row['exag']
        M = P - el
        Q = a_reg - P

        df.loc[self.name, :] = [el, P, M, Q]
        df['a'] = a_reg
        df['LR'] = df.L / df.P
        df['PQ'] = df.P / df.Q
        df['ROE'] = df.M / df.Q
        # ap = namedtuple('AggregatePricing', ['df', 'distortion'])
        # return ap(df, g)  # kinda dumb...
        return df


class Severity(ss.rv_continuous):

    def __init__(self, sev_name, exp_attachment=0, exp_limit=np.inf, sev_mean=0, sev_cv=0, sev_a=np.nan, sev_b=0,
                 sev_loc=0, sev_scale=0, sev_xs=None, sev_ps=None, sev_wt=1, sev_conditional=True, name='', note=''):
        """
        A continuous random variable, subclasses ``scipy.statistics_df.rv_continuous``,
        adding layer and attachment functionality. It overrides

        * **cdf**
        * **pdf**
        * **isf**
        * **ppf**
        * **sf**
        * **stats**

        TODO numerical integration with infinite support and a low standard deviation.

        :param sev_name: scipy statistics_df continuous distribution | (c|d)histogram  cts or discerte | fixed
        :param exp_attachment:
        :param exp_limit:
        :param sev_mean:
        :param sev_cv:
        :param sev_a:
        :param sev_b:
        :param sev_loc:
        :param sev_scale:
        :param sev_xs: for fixed or histogram classes
        :param sev_ps:
        :param sev_wt: this is not used directly; but it is convenient to pass it in and ignore it because sevs are
               implicitly created with sev_wt=1.
        :param sev_conditional: conditional or unconditional; for severities use conditional
        """

        from .portfolio import Portfolio

        super().__init__(self, name=f'{sev_name}[{exp_limit} xs {exp_attachment:,.0f}]')
        # ss.rv_continuous.__init__(self, name=f'{sev_name}[{exp_limit} xs {exp_attachment:,.0f}]')

        self.program = ''    # may be set externally
        self.limit = exp_limit
        self.attachment = exp_attachment
        self.detachment = exp_limit + exp_attachment
        self.fz = None
        self.pattach = 0
        self.pdetach = 0
        self.conditional = sev_conditional
        self.sev_name = sev_name
        self.name = name
        self.long_name = f'{sev_name}[{exp_limit} xs {exp_attachment:,.0f}]'
        self.note = note
        self.sev1 = self.sev2 = self.sev3 = None
        self.sev_wt = sev_wt
        logger.debug(
            f'Severity.__init__  | creating new Severity {self.sev_name} at {super().__repr__()}')
        # there are two types: if sev_xs and sev_ps provided then fixed/histogram, else scpiy dist
        # allows you to define fixed with just xs=1 (no log)
        if sev_xs is not None:
            if sev_name == 'fixed':
                # fixed is a special case of dhistogram with just one point
                sev_name = 'dhistogram'
                sev_ps = np.array(1)
            assert sev_name[1:] == 'histogram'
            # TODO: make histogram work with exp_limit and exp_attachment; currently they are ignored
            try:
                xs, ps = np.broadcast_arrays(np.array(sev_xs), np.array(sev_ps))
            except ValueError:
                # for empirical
                logger.warning(f'Severity.init | {sev_name} sev_xs and sev_ps cannot be broadcast')
                xs = np.array(sev_xs)
                ps = np.array(sev_ps)
            if not np.isclose(np.sum(ps), 1.0):
                logger.error(f'Severity.init | {sev_name} histogram/fixed severity with probs do not sum to 1, '
                             f'{np.sum(ps)}')
            # need to exp_limit distribution
            exp_limit = min(np.min(exp_limit), xs.max())
            if sev_name == 'chistogram':
                # continuous histogram: uniform between xs's
                # if the inputs are not evenly spaced this messes up because it interprets p as the
                #  height of the density over the range...hence have to rescale
                #  it DOES NOT matter that the p's add up to 1...that is handled automatically
                # changed 1 to -2 so the last bucket is bigger WHY SORTED???
                if len(xs) == len(ps):
                    xss = np.sort(np.hstack((xs, xs[-1] + xs[-2])))
                else:
                    # allows to pass in with the right hand end specified
                    xss = xs
                aps = ps / np.diff(xss)
                # this is now slightly bigger
                exp_limit = min(np.min(exp_limit), xss.max())
                # midpoints
                xsm = (xss[:-1] + xss[1:]) / 2
                self.sev1 = np.sum(xsm * ps)
                self.sev2 = np.sum(xsm ** 2 * ps)
                self.sev3 = np.sum(xsm ** 3 * ps)
                self.fz = ss.rv_histogram((aps, xss))
            elif sev_name == 'dhistogram':
                # discrete histogram: point masses at xs's
                self.sev1 = np.sum(xs * ps)
                self.sev2 = np.sum(xs ** 2 * ps)
                self.sev3 = np.sum(xs ** 3 * ps)
                # binary consistent
                xss = np.sort(np.hstack((xs - 2 ** -14, xs)))  # was + but F(x) = Pr(X<=x) so seems shd be to left
                pss = np.vstack((ps, np.zeros_like(ps))).reshape((-1,), order='F')[:-1]
                self.fz = ss.rv_histogram((pss, xss))
            else:
                raise ValueError('Histogram must be chistogram (continuous) or dhistogram (discrete)'
                                 f', you passed {sev_name}')

        elif isinstance(sev_name, Severity):
            self.fz = sev_name

        elif not isinstance(sev_name, (str, np.str_)):
            # must be a meta object - replaced in Underwriter.write
            log2 = sev_a
            bs = sev_b  # if zero it is happy to take whatever....
            if isinstance(sev_name, Aggregate):
                if log2 and (log2 != sev_name.log2 or (bs != sev_name.bs and bs != 0)):
                    # recompute
                    sev_name.easy_update(log2, bs)
                xs = sev_name.xs
                ps = sev_name.agg_density
            elif isinstance(sev_name, Portfolio):
                if log2 and (log2 != sev_name.log2 or (bs != sev_name.bs and bs != 0)):
                    # recompute
                    sev_name.update(log2, bs, add_exa=False)
                xs = sev_name.density_df.loss.values
                ps = sev_name.density_df.p_total.values
            else:
                raise ValueError(f'Object {sev_name} passed as a proto-severity type but'
                                 f' only Aggregate, Portfolio and Severity objects allowed')
            # will make as a combo discrete/continuous histogram
            # nail the bucket at zero and use a continuous approx +/- bs/2 around each other bucket
            # leaves an ugly gap between 0 and bs/2...which is ignored
            b1size = 1e-7  # size of the first "bucket"
            xss = np.hstack((-bs * b1size, 0, xs[1:] - bs / 2, xs[-1] + bs / 2))
            pss = np.hstack((ps[0] / b1size, 0, ps[1:]))
            self.fz = ss.rv_histogram((pss, xss))
            self.sev1 = np.sum(xs * ps)
            self.sev2 = np.sum(xs ** 2 * ps)
            self.sev3 = np.sum(xs ** 3 * ps)

        elif sev_name in ['norm', 'expon', 'uniform']:
            # distributions with no shape parameters
            #     Normal (and possibly others) does not have a shape parameter
            if sev_loc == 0 and sev_mean > 0:
                sev_loc = sev_mean
            if sev_scale == 0 and sev_cv > 0:
                sev_scale = sev_cv * sev_loc
            gen = getattr(ss, sev_name)
            self.fz = gen(loc=sev_loc, scale=sev_scale)

        elif sev_name in ['beta']:
            # distributions with two shape parameters
            # require specific inputs
            # for Kent examples input sev_scale=maxl, sev_mean=el and sev_cv as input
            #     beta sev_a and sev_b params given expected loss, max loss exposure and sev_cv
            #     Kent E.'s specification. Just used to create the CAgg classes for his examples (in agg.examples)
            #     https://en.wikipedia.org/wiki/Beta_distribution#Two_unknown_parameters
            if sev_name == 'beta' and sev_mean > 0 and sev_cv > 0:
                m = sev_mean / sev_scale
                v = m * m * sev_cv * sev_cv
                sev_a = m * (m * (1 - m) / v - 1)
                sev_b = (1 - m) * (m * (1 - m) / v - 1)
                # logger.error(f'{sev_mean}, {sev_cv}, {sev_scale}, {m}, {v}, {sev_a}, {sev_b}')
                self.fz = ss.beta(sev_a, sev_b, loc=0, scale=sev_scale)
            else:
                gen = getattr(ss, sev_name)
                self.fz = gen(sev_a, sev_b, loc=sev_loc, scale=sev_scale)
        else:
            # distributions with one shape parameter, which either comes from sev_a or sev_cv
            if np.isnan(sev_a) and sev_cv > 0:
                sev_a, _ = self.cv_to_shape(sev_cv)
                logger.info(f'sev_a not set, determined as {sev_a} shape from sev_cv {sev_cv}')
            elif np.isnan(sev_a):
                raise ValueError('sev_a not set and sev_cv=0 is invalid, no way to determine shape.')
            # have sev_a, now assemble distribution
            if sev_mean > 0:
                logger.info(f'creating with sev_mean={sev_mean} and sev_loc={sev_loc}')
                sev_scale, self.fz = self.mean_to_scale(sev_a, sev_mean, sev_loc)
            elif sev_scale > 0 and sev_mean==0:
                logger.info(f'creating with sev_scale={sev_scale} and sev_loc={sev_loc}')
                gen = getattr(ss, sev_name)
                self.fz = gen(sev_a, scale=sev_scale, loc=sev_loc)
            else:
                raise ValueError('sev_scale and sev_mean both equal zero.')
        if self.detachment == np.inf:
            self.pdetach = 0
        else:
            self.pdetach = self.fz.sf(self.detachment)

        if self.attachment == 0:
            self.pattach = 1
        else:
            self.pattach = self.fz.sf(self.attachment)

        if sev_mean > 0 or sev_cv > 0:
            # if you input a sev_mean or sev_cv check we are close to target
            st = self.fz.stats('mv')
            m = st[0]
            acv = st[1] ** .5 / m  # achieved sev_cv
            # sev_loc added so you can write lognorm 5 cv .3 + 10 a shifted lognorm mean 5
            if sev_mean > 0 and not np.isclose(sev_mean + sev_loc, m):
                print(f'WARNING target mean {sev_mean} and achieved mean {m} not close')
                # assert (np.isclose(sev_mean, m))
            if sev_cv > 0 and not np.isclose(sev_cv * sev_mean / (sev_mean + sev_loc), acv):
                print(f'WARNING target cv {sev_cv} and achieved cv {acv} not close')
                # assert (np.isclose(sev_cv, acv))
            # print('ACHIEVED', sev_mean, sev_cv, m, acv, self.fz.statistics_df(), self._stats())
            logger.debug(
                f'Severity.__init__ | parameters {sev_a}, {sev_scale}: target/actual {sev_mean} vs {m};  '
                f'{sev_cv} vs {acv}')

        if exp_limit < np.inf or exp_attachment > 0:
            layer_text = f'[{exp_limit:,.0f}' if exp_limit != np.inf else "Unlimited"
            layer_text += f' xs {exp_attachment:,.0f}]'
        else:
            layer_text = ''
        try:
            self.long_name = f'{name}: {sev_name}({self.fz.arg_dict[0]:.2f}){layer_text}'
        except:
            # 'rv_histogram' object has no attribute 'arg_dict'
            self.long_name = f'{name}: {sev_name}{layer_text}'

        assert self.fz is not None

    def __repr__(self):
        """
        wrap default with name
        :return:
        """
        return f'{super(Severity, self).__repr__()} of type {self.sev_name}'

    def cv_to_shape(self, cv, hint=1):
        """
        Create a frozen object of type dist_name with given cv. The
        lognormal, gamma, inverse gamma and inverse gaussian distributions
        are solved analytically.
        Other distributions solved numerically and may be unstable.

        :param cv:
        :param hint:
        :return:
        """
        # some special cases we can handle:
        if self.sev_name == 'lognorm':
            shape = np.sqrt(np.log(cv * cv + 1))
            fz = ss.lognorm(shape)
            return shape, fz

        if self.sev_name == 'gamma':
            shape = 1 / (cv * cv)
            fz = ss.gamma(shape)
            return shape, fz

        if self.sev_name == 'invgamma':
            shape = 1 / cv ** 2 + 2
            fz = ss.invgamma(shape)
            return shape, fz

        if self.sev_name == 'invgauss':
            shape = cv ** 2
            fz = ss.invgauss(shape)
            return shape, fz

        # pareto with loc=-1 alpha = 2 cv^2  / (cv^2 - 1)

        gen = getattr(ss, self.sev_name)

        def f(shape):
            fz0 = gen(shape)
            temp = fz0.stats('mv')
            return cv - temp[1] ** .5 / temp[0]

        try:
            ans = newton(f, hint)
        except RuntimeError:
            logger.error(f'cv_to_shape | error for {self.sev_name}, {cv}')
            ans = np.inf
            return ans, None
        fz = gen(ans)
        return ans, fz

    def mean_to_scale(self, shape, mean, loc=0):
        """
        Adjust the scale to achieved desired mean.
        Return a frozen instance.

        :param shape:
        :param mean:
        :param loc: location parameter (note: location is added to the mean...)
        :return:
        """
        gen = getattr(ss, self.sev_name)
        fz = gen(shape)
        m = fz.stats('m')
        scale = mean / m
        fz = gen(shape, scale=scale, loc=loc)
        return scale, fz

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

    def moms(self):
        """
        Revised moments for Severity class. Trying to compute moments of

            X(a,d) = min(d, (X-a)+)

        ==> E[X(a,d)^n] = int_a^d (x-a)^n f(x) dx + (d-a)^n S(d).

        Let x = q(p), F(x) = p, f(x)dx = dp.

        E[X(a,d)^n] = int_{F(a)}^{F(d)} (q(p)-a)^n dp + (d-a)^n S(d)

        The base is to compute int_{F(a)}^{F(d)} q(p)^n dp. These are exi below. They are then adjusted to create
        the moments needed.

        Old moments tried to compute int S(x)dx, but that is over a large, non-compact domain and
        did not work so well. With 0.9.3 old_moms was removed. Old_moms code did this: ::

            ex1 = safe_integrate(lambda x: self.fz.sf(x), 1)
            ex2 = safe_integrate(lambda x: 2 * (x - self.attachment) * self.fz.sf(x), 2)
            ex3 = safe_integrate(lambda x: 3 * (x - self.attachment) ** 2 * self.fz.sf(x), 3)

        **Test examples** ::

            def test(mu, sigma, a, y):
                global moms
                import types
                # analytic with no layer attachment
                fz = ss.lognorm(sigma, scale=np.exp(mu))
                tv = np.array([np.exp(k*mu + k * k * sigma**2/2) for k in range(1,4)])

                # old method
                s = agg.Severity('lognorm', sev_a=sigma, sev_scale=np.exp(mu),
                                 exp_attachment=a, exp_limit=y)
                est = np.array(s.old_moms())

                # swap out moment routine
                setattr(s, moms.__name__, types.MethodType(moms, s))
                ans = np.array(s.moms())

                # summarize and report
                sg = f'Example: mu={mu}  sigma={sigma}  a={a}  y={y}'
                print(f'{sg}\\n{"="*len(sg)}')
                print(pd.DataFrame({'new_ans' : ans, 'old_ans': est,
                                    'err': ans/est-1, 'no_la_analytic' : tv}))


            test(8.7, .5, 0, np.inf)
            test(8.7, 2.5, 0, np.inf)
            test(8.7, 2.5, 10e6, 200e6)

        **Example:**  ``mu=8.7``,  ``sigma=0.5``, ``a=0``,   ``y=inf`` ::

                    new_ans       old_ans           err  no_la_analytic
            0  6.802191e+03  6.802191e+03  3.918843e-11    6.802191e+03
            1  5.941160e+07  5.941160e+07  3.161149e-09    5.941160e+07
            2  6.662961e+11  6.662961e+11  2.377354e-08    6.662961e+11

        **Example:** mu=8.7  sigma=2.5  a=0  y=inf (here the old method failed) ::

                    new_ans       old_ans           err  no_la_analytic
            0  1.366256e+05  1.366257e+05 -6.942541e-08    1.366257e+05
            1  9.663487e+12  1.124575e+11  8.493016e+01    9.669522e+12
            2  2.720128e+23  7.597127e+19  3.579469e+03    3.545017e+23

        **Example:** mu=8.7  sigma=2.5  a=10000000.0  y=200000000.0 ::

                    new_ans       old_ans           err  no_la_analytic
            0  1.692484e+07  1.692484e+07  2.620126e-14    1.366257e+05
            1  1.180294e+15  1.180294e+15  5.242473e-13    9.669522e+12
            2  1.538310e+23  1.538310e+23  9.814372e-14    3.545017e+23


        The numerical issues are very sensitive. Goign for a compromise between
        speed and accuracy. Only an issue for very thick tailed distributions
        with no upper limit - not a realistic situation.
        Here is a tester program for two common cases:

        ::
            logger_level(30) # see what is going on
            for sh, dist in zip([1,2,3,4, 3.5,2.5,1.5,.5], ['lognorm']*3 + ['pareto']*4):
                s = Severity(dist, sev_a=sh, sev_scale=1, exp_attachment=0)
                print(dist,sh, s.moms())
                if dist == 'lognorm':
                    print('actual', [(n, np.exp(n*n*sh*sh/2)) for n in range(1,4)])

        :return: vector of moments. ``np.nan`` signals unreliable but finite value.
            ``np.inf`` is correct, the moment does not exist.

        """

        # def safe_integrate(f, lower, upper, level, big):
        #     """
        #     Integrate the survival function, pay attention to error messages. Split integral if needed.
        #     One shot pass.
        #
        #     """
        #
        #     argkw = dict(limit=100, epsrel=1e-8, full_output=1)
        #     if upper < big:
        #         split = 'no'
        #         ex = quad(f, lower, upper, **argkw)
        #         ans = ex[0]
        #         err = ex[1]
        #         steps = ex[2]['last']
        #         if len(ex) == 4:
        #             msg = ex[3][:33]
        #         else:
        #             msg = ''
        #
        #     else:
        #         split = 'yes'
        #         ex = quad(f, lower, big, **argkw)
        #         ans1 = ex[0]
        #         err1 = ex[1]
        #         steps1 = ex[2]['last']
        #         if len(ex) == 4:
        #             msg1 = ex[3][:33]
        #         else:
        #             msg1 = ''
        #
        #         ex2 = quad(f, big, upper, **argkw)
        #         ans2 = ex2[0]
        #         err2 = ex2[1]
        #         steps2 = ex2[2]['last']
        #         if len(ex2) == 4:
        #             msg2 = ex2[3][:33]
        #         else:
        #             msg2 = ''
        #
        #         ans = ans1 + ans2
        #         err = err1 + err2
        #         steps = steps1 + steps2
        #         msg = msg1 + ' ' + msg2
        #         logger.warning(f'integrals: lhs={ans:,.0f}, rhs={ans2:,.0f}')
        #     logger.warning(f'E[X^{level}]: split={split}, ansr={ans}, error={err}, steps={steps}; message {msg}')
        #     return ans

        def safe_integrate(f, lower, upper, level):
            """
            Integrate the survival function, pay attention to error messages.

            """

            # argkw = dict(limit=100, epsabs=1e-6, epsrel=1e-6, full_output=1)
            argkw = dict(limit=100, epsrel=1e-6 if level==1 else 1e-4, full_output=1)
            ex = quad(f, lower, upper, **argkw)
            if len(ex) == 4 or ex[0] == np.inf:  # 'The integral is probably divergent, or slowly convergent.':
                msg = ex[-1].replace("\n", " ") if ex[-1] == str else "no message"
                logger.info(f'E[X^{level}]: ansr={ex[0]}, error={ex[1]}, steps={ex[2]["last"]}; message {msg} -> splitting integral')
                # blow off other steps....
                # use nan to mean "unreliable
                # return np.nan   #  ex[0]
                # this is too slow...and we don't really use it...
                ϵ = 0.0001
                if lower == 0 and upper > ϵ:
                    logger.info(
                        f'Severity.moms | splitting {self.sev_name} EX^{level} integral for convergence reasons')
                    exa = quad(f, 1e-16, ϵ, **argkw)
                    exb = quad(f, ϵ, upper, **argkw)
                    logger.info(f'Severity.moms | [1e-16, {ϵ}] split EX^{level}: ansr={exa[0]}, error={exa[1]}, steps={exa[2]["last"]}')
                    logger.info(f'Severity.moms | [{ϵ}, {upper}] split EX^{level}: ansr={exb[0]}, error={exb[1]}, steps={exb[2]["last"]}')
                    ex = (exa[0] + exb[0], exa[1] + exb[1])
                    # if exa[2]['last'] < argkw['limit'] and exb[2]['last'] < argkw['limit']:
                    #     ex = (exa[0] + exb[0], exa[1] + exb[1])
                    # else:
                    #     # reached limit, unreliable
                    #     ex = (np.nan, np.nan)
            logger.info(f'E[X^{level}]={ex[0]}, error={ex[1]}, est rel error={ex[1]/ex[0]}')
            return ex[:2]

        # we integrate isf not q, so upper and lower are swapped
        if self.attachment == 0:
            upper = 1
        else:
            upper = self.fz.sf(self.attachment)
        if self.detachment == np.inf:
            lower = 0
        else:
            lower = self.fz.sf(self.detachment)

        # compute moments: histograms are tricky to integrate and we know the answer already...so
        if self.attachment == 0 and self.detachment == np.inf and self.sev_name.endswith('histogram'):
            ex1 = self.sev1
            ex2 = self.sev2
            ex3 = self.sev3
        else:
            if self.detachment == np.inf:
                # figure which moments actually exist
                moments_finite = list(map(lambda x: not (np.isinf(x) or np.isnan(x)), self.fz.stats('mvs')))
            else:
                moments_finite = [True, True, True]
            logger.info(str(moments_finite))
            continue_calc = True
            max_rel_error = 1e-3
            if moments_finite[0]:
                ex1 = safe_integrate(self.fz.isf, lower, upper, 1)
                if ex1[1] / ex1[0] < max_rel_error:
                    ex1 = ex1[0]
                else:
                    ex1 = np.nan
                    continue_calc = False
            else:
                logger.info('First moment does not exist.')
                ex1 = np.inf

            if continue_calc and moments_finite[1]:
                ex2 = safe_integrate(lambda x: self.fz.isf(x) ** 2, lower, upper, 2)
                # we know the mean; use that scale to determine if the absolute error is reasonable?
                if ex2[1] / ex2[0] < max_rel_error: # and ex2[1] < 0.01 * ex1**2:
                    ex2 = ex2[0]
                else:
                    ex2 = np.nan
                    continue_calc = False
            elif not continue_calc:
                ex2 = np.nan
            else:
                logger.info('Second moment does not exist.')
                ex2 = np.inf

            if continue_calc and moments_finite[2]:
                ex3 = safe_integrate(lambda x: self.fz.isf(x) ** 3, lower, upper, 3)
                if ex3[1] / ex3[0] < max_rel_error:
                    ex3 = ex3[0]
                else:
                    ex3 = np.nan
            elif not continue_calc:
                ex3 = np.nan
            else:
                logger.info('Third moment does not exist.')
                ex3 = np.inf

        # adjust
        dma = self.detachment - self.attachment
        uml = upper - lower
        a = self.attachment
        if a > 0:
            ex1a = ex1 - a * uml
            ex2a = ex2 - 2 * a * ex1 + a ** 2 * uml
            ex3a = ex3 - 3 * a * ex2 + 3 * a ** 2 * ex1 - a ** 3 * uml
        else:
            ex1a = ex1
            ex2a = ex2
            ex3a = ex3

        if self.detachment < np.inf:
            ex1a += dma * lower
            ex2a += dma ** 2 * lower
            ex3a += dma ** 3 * lower

        if self.conditional:
            ex1a /= self.pattach
            ex2a /= self.pattach
            ex3a /= self.pattach

        return ex1a, ex2a, ex3a

    # def update(self, log2=0, bs=0, **kwargs):
    #     """
    #     This is a convenience function so that update can be called on any kind of object.
    #     It has no effect.
    #
    #     :param log2:
    #     :param bs:
    #     :param kwargs:
    #     :return:
    #     """
    #     pass

    def plot(self, N=100, figsize=(12, 3)):
        """
        Quick plot, updated for 0.9.3 with mosaic and no grid lines. (F(x), x) plot
        replaced with log density plot.

        TODO better coordination of figsize! Better axis formats and ranges.

        :param N:
        :param figsize:
        :return:
        """

        xs = np.linspace(0, self._isf(1e-4), N)
        xs2 = np.linspace(0, self._isf(1e-12), N)

        f = plt.figure(constrained_layout=True, figsize=figsize)
        axd = f.subplot_mosaic('ABCD')

        ds = 'steps-post' if self.sev_name == 'dhistogram' else 'default'

        ax = axd['A']
        ys = self._pdf(xs)
        ax.plot(xs, ys, drawstyle=ds, lw=1)
        ax.set(title='Probability density', xlabel='Loss')
        yl = ax.get_ylim()

        ax = axd['B']
        ys2 = self._pdf(xs2)
        ax.plot(xs2, ys2, drawstyle=ds, lw=1)
        ax.set(title='Log density', xlabel='Loss', yscale='log', ylim=[1e-14, 2 * yl[1]])

        ax = axd['C']
        ys = self._cdf(xs)
        ax.plot(xs, ys, drawstyle=ds, lw=1)
        ax.set(title='Probability distribution', xlabel='Loss', ylim=[-0.025, 1.025])

        ax = axd['D']
        ax.plot(ys, xs, drawstyle=ds, lw=1)
        ax.set(title='Quantile (Lee) plot', xlabel='Non-exceeding probability p (or ω)', xlim=[-0.025, 1.025])


# class CarefulInverse(object):
#     """
#     From SRM_Examples Noise: careful inverse functions.
#
#     """
#
#     @staticmethod
#     def make1d(xs, ys, agg_fun=None, kind='linear', **kwargs):
#         """
#         Wrapper to make a reasonable 1d interpolation function with reasonable extrapolation
#         Does NOT handle inverse functions, for those use dist_inv1d
#         :param xs:
#         :param ys:
#         :param agg_fun:
#         :param kind:
#         :param kwargs:
#         :return:
#         """
#         temp = pd.DataFrame(dict(x=xs, y=ys))
#         if agg_fun:
#             temp = temp.groupby('x').agg(agg_fun)
#             fill_value = ((temp.y.iloc[0]), (temp.y.iloc[-1]))
#             f = interpolate.interp1d(temp.index, temp.y, kind=kind, bounds_error=False, fill_value=fill_value, **kwargs)
#         else:
#             fill_value = ((temp.y.iloc[0]), (temp.y.iloc[-1]))
#             f = interpolate.interp1d(temp.x, temp.y, kind=kind, bounds_error=False, fill_value=fill_value, **kwargs)
#         return f
#
#     @staticmethod
#     def dist_inv1d(xs, fx, kind='linear', max_Fx=1.):
#         """
#         Careful inverse of distribution function with jumps. Assumes xs is evenly spaced.
#         Assumes that if there are two or more xs values between changes in dist it is a jump,
#         otherwise is is a continuous part. Puts in -eps values to make steps around jumps.
#
#         :param xs:
#         :param fx:  density
#         :param kind:
#         :param max_Fx: what is the max allowable value of F(x)?
#         :return:
#         """
#
#         # make dataframe to allow summarization
#         df = pd.DataFrame(dict(x=xs, fx=fx))
#         # lots of problems with noise...strip it off
#         df['fx'] = np.where(np.abs(df.fx) < 1e-16, 0, df.fx)
#         # compute cumulative probabilities
#         df['Fx'] = df.fx.cumsum()
#         gs = df.groupby('Fx').agg({'x': [np.min, np.max, len]})
#         gs.columns = ['mn', 'mx', 'n']
#         # figure if a jump or not
#         gs['jump'] = 0
#         gs.loc[gs.n > 1, 'jump'] = 1
#         gs = gs.reset_index(drop=False)
#         # figure the right hand end of the jump
#         gs['nextFx'] = gs.Fx.shift(-1, fill_value=1)
#
#         # space for answer
#         ans = np.zeros((2 * len(gs), 2))
#         rn = 0
#         eps = 1e-10
#         max_Fx -= eps / 100
#         # write out known (x, y) points for lin interp
#         for n, r in gs.iterrows():
#             ans[rn, 0] = r.Fx
#             ans[rn, 1] = r.mn if r.Fx >= max_Fx else r.mx
#             rn += 1
#             if r.Fx >= max_Fx:
#                 break
#             if r.jump:
#                 if r.nextFx >= max_Fx:
#                     break
#                 ans[rn, 0] = r.nextFx - eps
#                 ans[rn, 1] = r.mx
#                 rn += 1
#         # trim up ans
#         ans = ans[:rn, :]
#
#         # make interpolation function and return
#         fv = ((ans[0, 1]), (ans[-1, 1]))
#         ff = interpolate.interp1d(ans[:, 0], ans[:, 1], bounds_error=False, fill_value=fv, kind=kind)
#         # df = input in data frame; gs = grouped df, ans = carefully selected points for inverse
#         return ff
