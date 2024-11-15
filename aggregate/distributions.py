from collections import namedtuple
from collections.abc import Iterable
from functools import lru_cache, wraps
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
from scipy.special import kv, gammaln, hyp1f1
from scipy.optimize import broyden2, newton_krylov, brentq
from scipy.optimize import NoConvergence  # noqa
from scipy.interpolate import interp1d
from textwrap import fill

from .constants import *
from .utilities import (sln_fit, sgamma_fit, ln_fit, gamma_fit, ft, ift,
                        estimate_agg_percentile, MomentAggregator,
                        xsden_to_meancv, round_bucket, make_ceder_netter, MomentWrangler,
                        make_mosaic_figure, nice_multiple, xsden_to_meancvskew,
                        pprint_ex, approximate_work, moms_analytic, picks_work,
                        integral_by_doubling, logarithmic_theta, make_var_tvar,
                        more, explain_validation)
import aggregate.random_agg as ar
from .spectral import Distortion

logger = logging.getLogger(__name__)


def max_log2(x):
    """
    Return the largest power of two d so that (x + 2**-d) - x == 2**-d, with d <= 30.
    Used in dhistogram severity types to determine the size of the step.
    """
    d = min(30, -np.log2(np.finfo(float).eps) - np.ceil(np.log2(x)) - 1)
    if (x + 2 ** -d) - x != 2 ** -d:
        raise ValueError('max_log2 failed')
    return d


def validate_discrete_distribution(xs, ps):
    """
    Make sure that outcomes are distinct, non-negative
    and sorted in asending order, and that probabiliites are
    summed across distinct outcomes. Used in
    dsev and dfreq to validate user input.
    """
    if len(xs) != len(set(xs)) or len(xs[xs<0]) > 0:
        logger.info('Duplicates in empirical distribution and/or negative values, summarizing.')
        temp_df = pd.DataFrame({'x': xs, 'p': ps})
        temp_df.loc[temp_df.x < 0, 'x'] = 0.
        temp_df = temp_df.groupby('x').sum('p')
        xs = np.array(temp_df.index)
        ps = temp_df.p.values
    return xs, ps


class Frequency(object):
    """
    Manages Frequency distributions: creates moment function and MGF.

    - freq_moms(n): returns EN, EN^2 and EN^3 when EN=n
    - freq_pgf(n, z): returns the moment generating function applied to z when EN=n

    Frequency distributions are either non-mixture types or mixture types.

    **Non-Mixture** Frequency Types

    - ``fixed``: no parameters
    - ``bernoulli``: exp_en interpreted as a probability, must be < 1
    - ``binomial``: Binomial(n/p, p) where p = freq_a, and n = exp_en
    - ``poisson``: Poisson(n)
    - ``geometric``: geometric(1/(n + 1)), supported on 0, 1, 2, ...
    - ``logarithmci``: logarithmic(theta), supported on 1, 2, ...; theta solved numerically
    - ``negymana``: Po(n/freq_a) stopped sum of Po(freq_a) freq_a = "eggs per cluster"
    - ``negbin``: freq_a is the variance multiplier, ratio of variance to mean
    - ``pascal``:
    - ``pascal``: (generalized) pascal-poisson distribution, a poisson stopped sum of negative binomial;
      exp_en gives the overall claim count. freq_a is the CV of the frequency distribution
      and freq_b is the number of claimants per claim (or claims per occurrence). Hence, the Poisson
      component has mean exp_en / freq_b and the number of claims per occurrence has mean freq_b. This
      parameterization may not be ideal(!).

    **Mixture** Frequency Types

    These distributions are G-mixed Poisson, so N | G ~ Poisson(n G). They are labelled by
    the name of the mixing distribution or the common name for the resulting frequency
    distribution. See Panjer and Willmot or JKK.

    In all cases freq_a is the CV of the mixing distribution which corresponds to the
    asympototic CV of the frequency distribution and of any aggregate when the severity has a variance.

    - ``gamma``: negative binomial, freq_a = cv of gamma distribution
    - ``delaporte``: shifted gamma, freq_a = cv of mixing disitribution, freq_b = proportion of
      certain claims = shift. freq_b must be between 0 and 1.
    - ``ig``: inverse gaussian, freq_a = cv of mixing distribution
    - ``sig``: shifted inverse gaussian, freq_a = cv of mixing disitribution, freq_b = proportion of
      certain claims = shift. freq_b must be between 0 and 1.
    - ``sichel``: generalized inverse gaussian mixing distribution, freq_a = cv of mixing distribution and
      freq_b = lambda value. The beta and mu parameters solved to match moments. Note lambda =
      -0.5 corresponds to inverse gaussian and 0.5 to reciprocal inverse gauusian. Other special
      cases are available.
    - ``sichel.gamma``: generalized inverse gaussian mixture where the parameters match the moments of a
      delaporte distribution with given freq_a and freq_b
    - ``sichel.ig``: generalized inverse gaussian mixture where the parameters match the moments of a
      shifted inverse gaussian distribution with given freq_a and freq_b. This parameterization
      has poor numerical stability and may fail.
    - ``beta``: beta mixing with freq_a = Cv where beta is supported on the interval [0, freq_b]. This
      method should be used carefully. It has poor numerical stability and can produce bizzare
      aggregates when the alpha or beta parameters are < 1 (so there is a mode at 0 or freq_b).

    Code proof for Neyman A::

        from aggregate import build, qd
        mean = 10
        eggs_per_cluster = 4
        neya = build(f'agg Neya {mean} claims dsev[1] neymana {eggs_per_cluster}')
        qd(neya)

        po = build(f'agg Po4 {eggs_per_cluster} claims dsev[1] poisson')
        po_pmf = po.density_df.query('p_total > 1e-13').p_total

        byhand = build(f'agg ByHand {mean / eggs_per_cluster} claims dsev {list(po_pmf.index)} {po_pmf.values} poisson')
        qd(byhand)

        df = pd.concat((neya.density_df.p_total, byhand.density_df.p_total), axis=1)
        df.columns = ['neya', 'byhand']
        df['err'] = df.neya - df.byhand
        assert df.err.abs().max() < 1e-5
        df.head(40)

    Code proof for Pascal::

        from aggregate import build, qd
        mean = 10
        claims_per_occ =1.24
        overall_cv = 1.255
        pascal = build(f'agg PascalEg {mean} claims dsev[1] pascal {overall_cv} {claims_per_occ}', log2=16)
        qd(pascal)

        c = (mean * overall_cv**2 - 1 - claims_per_occ) / claims_per_occ
        th = claims_per_occ * c
        a = 1 / c
        # from form of nb pgf identify r = a and beta = theta, mean is rb, var is rb(1+b)
        nb = build(f'agg NB {claims_per_occ} claims dsev[1] negbin {th + 1}', log2=16)
        nb_pmf = nb.density_df.query('p_total > 1e-13').p_total
        qd(nb)

        byhand = build(f'agg ByHand {mean / claims_per_occ} claims dsev {list(nb_pmf.index)} {nb_pmf.values} poisson', log2=16)
        qd(byhand)

        df = pd.concat((pascal.density_df.p_total, byhand.density_df.p_total), axis=1)
        df.columns = ['pascal', 'byhand']
        df['err'] = df.pascal - df.byhand
        assert df.err.abs().max() < 1e-5
        df.head(40)

    :param freq_name: name of the frequency distribution, poisson, geometric, etc.
    :param freq_a:
    :param freq_b:
    """

    __slots__ = ['freq_moms', 'freq_pgf', 'freq_name', 'freq_a', 'freq_b', 'freq_zm', 'freq_p0',
                 'panjer_ab', 'unmodified_mean', 'prn_eq_0']

    def __init__(self, freq_name, freq_a, freq_b, freq_zm, freq_p0):
        """
        Creates the freq_pgf and moment function:

        * moment function(n) returns EN, EN^2, EN^3 when EN=n.
        * freq_pgf(n, z) is the freq_pgf evaluated at log(z) when EN=n

        :param freq_name: name of the frequency distribution, poisson, geometric, etc.
        :param freq_a:
        :param freq_b:
        :param freq_zm: freq_zm True if zero modified, default False
        :param freq_p0: modified p0, probability of zero claims
        """
        self.freq_name = freq_name
        self.freq_a = freq_a
        self.freq_b = freq_b
        self.freq_zm = freq_zm
        self.freq_p0 = freq_p0
        self.panjer_ab = None
        self.unmodified_mean = None
        self.prn_eq_0 = None

        if freq_zm is True:
            # add implemented methods here....
            if freq_name not in ('poisson', 'negbin', 'geometric', 'logarithmic', 'binomial'):
                raise NotImplementedError(f'Zero modification not implemented for {freq_name}')

        def solve_n_base(n, p0):
            """
            Find n_base so that ZM distribution with unmodified mean n_base
            adjusted to p0 has mean n. Must solve::

                   (1 - p0) / (1 - p0_base) n_base = n
                ==>(1 - p0) n_base / (1 - prn_eq_0(n_base)) = n

            If p0 < exp(-n) there is less weight on 0, increasing the mean of the ZM
            distribution, so n_base < n, otherwise it is greater.

            This is generic code, independent of distribution. Expects a function
            prn_eq_0(x) giving the probability a non-modified distribution of mean
            x takes the value 0.

            Code is used to create ZM distribution wrappers.

            """
            nonlocal prn_eq_0
            f = lambda x: (1 - p0) * x / (1 - prn_eq_0(x)) - n

            if p0 < prn_eq_0(n):
                # ZM has higher mean, search to left of n
                n_base = brentq(f, a=0, b=n)
            else:
                # search to right
                n_base = brentq(f, a=n, b=n /(1 - p0))
            self.unmodified_mean = n_base
            return n_base

        # build decorators for zm
        def zero_modify_moms(moms_func):
            """
            Decorator to zero modify the frequency function moms_func(n).

            Assmes there is a nonlocal function prn_eq_0(x), giving the unmodified prob(N=0)
            when the mean is x, and the global solve_n_base function. The value p0 is also
            non-local.
            """
            nonlocal prn_eq_0, freq_p0, freq_zm, solve_n_base
            if freq_zm is False:
                return moms_func
            # else work to do
            @wraps(moms_func)
            def wrapped_moms_func(n):
                n_base = solve_n_base(n, freq_p0)
                ans = np.array(moms_func(n_base))
                return (1 - freq_p0) / (1 - prn_eq_0(n_base)) * ans
            return wrapped_moms_func

        def zero_modify_pgf(pgf_func):
            """
            Decorator to zero modify freq_pgf(n, z), as above.

            The wrapped freq_pgf is a weighted average of the trivial and original pgfs.
            """
            nonlocal prn_eq_0, freq_p0, freq_zm, solve_n_base
            if freq_zm is False:
                return pgf_func
            # else work to do
            @wraps(pgf_func)
            def wrapped_pgf_func(n, z):
                n_base = solve_n_base(n, freq_p0)
                # weight on non-trivial component
                wt = (1 - freq_p0) / (1 - prn_eq_0(n_base))
                return (1 - wt) + wt * pgf_func(n_base, z)
            return wrapped_pgf_func

        logger.debug(
            f'Frequency.__init__ | creating new Frequency {self.freq_name} at {super(Frequency, self).__repr__()}')

        if self.freq_name == 'fixed':
            def _freq_moms(n):
                # fixed distribution N=n certainly
                freq_2 = n ** 2
                freq_3 = n * freq_2
                return n, freq_2, freq_3

            def pgf(n, z):
                return z ** n

        elif self.freq_name == 'bernoulli':
            def _freq_moms(n):
                # code for bernoulli n, E(N^k) = E(N) = n
                # n in this case only means probability of claim (=expected claim count)
                freq_2 = n
                freq_3 = n
                return n, freq_2, freq_3

            def pgf(n, z):
                # E(e^tlog(z)) = p z + (1-p), z = ft(severity)
                return z * n + np.ones_like(z) * (1 - n)

        elif self.freq_name == 'binomial':
            p = self.freq_a

            def prn_eq_0(n):
                N = n / p  # correct mean
                return (1 - p) ** N

            self.prn_eq_0 = prn_eq_0

            @zero_modify_moms
            def _freq_moms(n):
                # binomial(N, p) with mean n, N=n/p
                # http://mathworld.wolfram.com/BinomialDistribution.html
                # p = self.freq_a
                nonlocal p
                N = n / p  # correct mean
                freq_1 = N * p
                freq_2 = N * p * (1 - p + N * p)
                freq_3 = N * p * (1 + p * (N - 1) * (3 + p * (N - 2)))
                self.panjer_ab = (-p / (1 - p), (N + 1) * p / (1 - p))
                return freq_1, freq_2, freq_3

            @zero_modify_pgf
            def pgf(n, z):
                nonlocal p
                N = n / p # self.freq_a
                return (z * p + np.ones_like(z) * (1 - p)) ** N

        elif self.freq_name == 'negbin':
            # freq_a parameter inputs the variance multiplier, variance as a multiple of mean
            # go with the b, r parameterization, mean rb and variance rb(1 + b). Thus
            # b = v - 1 and r = n / b
            # Univariate Discrete Distributions, 3rd Edition (Norman L. Johnson, Adrienne W. Kemp etc.)
            # p. 208 uses P (our beta) and k (our r) for the parameters (see pgf)
            beta = self.freq_a - 1

            def prn_eq_0(n):
                nonlocal beta
                r = n / beta
                return (1 + beta) ** -r

            self.prn_eq_0 = prn_eq_0

            @zero_modify_moms
            def _freq_moms(n):
                nonlocal beta
                r = n / beta
                freq_2 = n * (1 + beta * (1 + r))
                # chat gpt (!)
                # freq_3 = r * (r + 1) * (r + 2) * beta**3 + 3 * r * (r + 1) * beta ** 2 + r * beta
                freq_3 = r * beta * (1 + beta * (1 + r) * (3 + beta * (2 + r)))
                self.panjer_ab = (beta / (1 + beta), (r - 1) * beta / (1 + beta))
                return n, freq_2, freq_3

            @zero_modify_pgf
            def pgf(n, z):
                nonlocal beta
                r = n / beta
                return (1 - beta * (z - 1)) ** -r

        elif self.freq_name == 'poisson' and self.freq_a == 0:
            def prn_eq_0(n):
                return np.exp(-n)

            self.prn_eq_0 = prn_eq_0

            @zero_modify_moms
            def _freq_moms(n):
                # Poisson
                freq_2 = n * (1 + n)
                freq_3 = n * (1 + n * (3 + n))
                self.panjer_ab = (0., n)
                return n, freq_2, freq_3

            @zero_modify_pgf
            def pgf(n, z):
                return np.exp(n * (z - 1))

        elif self.freq_name == 'geometric' and self.freq_a == 0:
            # as for poisson, single parameter
            # https://mathworld.wolfram.com/GeometricDistribution.html
            # https://en.wikipedia.org/wiki/Geometric_distribution
            # e.g. tester: agg =uw('agg GEOM 3 claims sev dhistogram xps [1] [1] geometric')
            # two flavors, with mean 1/p and 1/p - 1, we are using the latter, supported
            # on 0, 1, 2,
            def prn_eq_0(n):
                p = 1 / (n + 1)
                return p

            self.prn_eq_0 = prn_eq_0

            @zero_modify_moms
            def _freq_moms(n):
                p = 1 / (n + 1)
                freq_2 = (2 - p) * (1 - p) / p ** 2
                freq_3 = (1 - p) * (6 + (p - 6) * p) / p ** 3
                self.panjer_ab = (n / (1 + n), 0.)
                return n, freq_2, freq_3

            @zero_modify_pgf
            def pgf(n, z):
                p = 1 / (n + 1)
                return p / (1 - (1 - p) * z)

        elif self.freq_name == 'pascal':
            # generalized Poisson-Pascal distribution, Panjer Willmot green book. p. 324
            # Univariate Discrete Distributions, 3rd Edition (Norman L. Johnson, Adrienne W. Kemp etc.)
            # p. 367, Poisson mixture of negative binomial distributions
            # Taking k = 1 gives a Poisson mixture of geometric distri-
            # butions, known as the Polya–Aeppli distribution
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

            def pgf(n, z):
                c = (n * ν ** 2 - 1 - κ) / κ
                a = 1 / c
                θ = κ * c
                λ = n / κ  # poisson parameter for number of claims
                return np.exp(λ * ((1 - θ * (z - 1)) ** -a - 1))

        elif self.freq_name == 'logarithmic':
            # Aka logser in scipy stats, log series
            # supported on 1, 2, 3, ...
            # https://mathworld.wolfram.com/LogarithmicDistribution.html
            # https://en.wikipedia.org/wiki/Logarithmic_distribution
            def prn_eq_0(n):
                return 0.

            self.prn_eq_0 = prn_eq_0

            @zero_modify_moms
            def _freq_moms(n):
                theta = logarithmic_theta(n)
                a_logser = -1 / np.log(1 - theta)
                freq_2 = a_logser * theta / (1 - theta) ** 2
                freq_3 = a_logser * theta * (1 + theta) / (1 - theta) ** 3
                self.panjer_ab = (theta, -theta)
                return n, freq_2, freq_3

            @zero_modify_pgf
            def pgf(n, z):
                theta = logarithmic_theta(n)
                return np.log(1 - theta * z) / np.log(1 - theta)

        elif self.freq_name in ['neyman', 'neymana', 'neymanA']:
            # Neyman A, Panjer Willmot green book. p. 324?
            m2 = self.freq_a  # number of outcomes per cluster, mean n = m1 m2

            def _freq_moms(n):
                # central
                # freq_2 = n * (1 + m2)
                # freq_3 = n * (1 + m2 * (3 + m2))
                # non central
                freq_2 = n * ((1 + m2) + n)
                freq_3 = n * ((1 + m2 * (3 + m2)) + 3 * freq_2 - 2 * n ** 2)
                return n, freq_2, freq_3

            def pgf(n, z):
                m1 = n / m2
                return np.exp(m1 * (np.exp(m2 * (z - 1)) - 1))

        elif self.freq_name == 'empirical':
            # stated en here...need to reach up to agg to set that?!
            # parameters are entered as nps, to a is n values and b is probability masses
            self.freq_a, self.freq_b  = validate_discrete_distribution(self.freq_a, self.freq_b)
            def _freq_moms(n):
                # independent of n, it will be -1
                en = np.sum(self.freq_a * self.freq_b)
                en2 = np.sum(self.freq_a ** 2 * self.freq_b)
                en3 = np.sum(self.freq_a ** 3 * self.freq_b)
                return en, en2, en3

            def pgf(n, z):
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

            def pgf(n, z):
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

            def pgf(n, z):
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

            def pgf(n, z):
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

            def pgf(n, z):
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

            def pgf(n, z):
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

                def pgf(n, z):
                    kernel = n * (z - 1)
                    inner = np.sqrt(1 - 2 * β * kernel)
                    return inner ** (-λ) * kv(λ, μ * inner / β) / kv(λ, μ / β)

        else:
            raise ValueError(f'Inadmissible frequency type {self.freq_name}...')

        self.freq_moms = _freq_moms
        self.freq_pgf = pgf

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
            self._density_df = pd.DataFrame(dict(loss=self.xs, p_total=self.agg_density))
            self._density_df = self._density_df.set_index('loss', drop=False)
            self._density_df['p'] = self._density_df.p_total
            # remove the fuzz, same method as Portfolio.remove_fuzz
            eps = np.finfo(float).eps
            # may not have a severity, remember...
            self._density_df.loc[:, self._density_df.select_dtypes(include=['float64']).columns] = \
                self._density_df.select_dtypes(include=['float64']).applymap(lambda x: 0 if abs(x) < eps else x)

            # we spend a lot of time computing sev exactly, so don't want to flush that away
            # with remove fuzz...hence add here
            self._density_df['p_sev'] = self.sev_density

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

            # Update 2021-01-28: S is best computed forwards
            self._density_df['S'] = 1 - self._density_df.p_total.cumsum()
            self._density_df['S_sev'] = 1 - self._density_df.p_sev.cumsum()

            # add LEV, TVaR to each threshold point...
            self._density_df['lev'] = self._density_df.S.shift(1, fill_value=0).cumsum() * self.bs
            self._density_df['exa'] = self._density_df['lev']
            self._density_df['exlea'] = \
                (self._density_df.lev - self._density_df.loss * self._density_df.S) / self._density_df.F

            # expected value and epd
            self._density_df['e'] = self.est_m  # np.sum(self._density_df.p * self._density_df.loss)
            self._density_df.loc[:, 'epd'] = \
                np.maximum(0, (self._density_df.loc[:, 'e'] - self._density_df.loc[:, 'lev'])) / \
                self._density_df.loc[:, 'e']
            self._density_df['exgta'] = self._density_df.loss + (
                    self._density_df.e - self._density_df.exa) / self._density_df.S
            self._density_df['exeqa'] = self._density_df.loss  # E(X | X=a) = a(!) included for symmetry was exa

        return self._density_df

    @property
    def reinsurance_df(self):
        """
        Version of density_df tailored to reinsurance. Several cases

        * occ program only: agg_density_.. is recomputed manually for all three outcomes
        * agg program only: sev_density_... not set for gcn
        * both programs: agg is gcn for the agg program applied to the requested occ output



        ``_apply_reins_work``
        """
        if self.occ_reins is None and self.agg_reins is None:
            logger.log(WL, 'Asking for reinsurance_df, but no reinsurance specified. Returning None.')
            return None

        if self._reinsurance_df is None:
            self._reinsurance_df = \
                pd.DataFrame({'loss': self.xs,
                              'p_sev_gross': self.sev_density_gross if
                              self.sev_density_gross is not None else self.sev_density,
                              'p_sev_ceded': self.sev_density_ceded,
                              'p_sev_net': self.sev_density_net
                              },
                             index=pd.Index(self.xs, name='loss'))
            if self.occ_reins is not None:
                # add agg with gcn occ
                # TODO sort out
                logger.info('Computing aggregates with gcn severities; assumes approx=exact')
                for gcn, sv in zip(['p_agg_gross_occ', 'p_agg_ceded_occ', 'p_agg_net_occ'],
                                   [self.sev_density_gross, self.sev_density_ceded, self.sev_density_net]):
                    z = ft(sv, self.padding, None)
                    ftz = self.freq_pgf(self.n, z)
                    if np.sum(self.en) == 1 and self.freq_name == 'fixed':
                        ad = sv.copy()
                    else:
                        ad = np.real(ift(ftz, self.padding, None))
                    self._reinsurance_df[gcn] = ad
            if self.agg_density_gross is None:
                # no agg program
                self._reinsurance_df['p_agg_gross'] = self.agg_density
                self._reinsurance_df['p_agg_ceded'] = None
                self._reinsurance_df['p_agg_net'] = None
            else:
                # agg program
                self._reinsurance_df['p_agg_gross'] = self.agg_density_gross
                self._reinsurance_df['p_agg_ceded'] = self.agg_density_ceded
                self._reinsurance_df['p_agg_net'] = self.agg_density_net

        return self._reinsurance_df

    @property
    def reinsurance_occ_layer_df(self):
        """
        How losses are layered by the occurrence reinsurance. Expected loss,
        CV layer loss, and expected counts to layers. 
        """
        if self.occ_reins is None:
            return None
        bit0 = self.reinsurance_audit_df.loc['occ'].xs('ex', axis=1, level=1)
        bit = self.reinsurance_audit_df.loc['occ'].xs('cv', axis=1, level=1)
        bit1 = pd.DataFrame(index=bit.index)
        bit1['ceded'] = [self.n if i == 'gup' else self.n * self.sev.sf(i)
                         for i in bit1.index.get_level_values('attach')]
        bit2 = pd.DataFrame(index=bit.index)
        # i = (share, layer, attach)
        bit3 = bit0['ceded']
        bit3 = bit3.iloc[:] / bit0['subject'].iloc[-1]
        # TODO: not sure total cession is correct
        bit2['ceded'] = [v.ceded if i[-1] == 'gup' else v.ceded / self.sev.sf(i[-1] / i[0])
                         for i, v in bit0[['ceded']].iterrows()]
        ans = pd.concat((
            bit0 * self.n,
            bit, bit1, bit2, bit3),
            axis=1, keys=['ex', 'cv', 'en', 'severity', 'pct'],
            names=['stat', 'view'])
        return ans

    @property
    def reinsurance_report_df(self):
        """
        Create and return a dataframe with the reinsurance report.
        TODO: sort out the overlap with reinsurance_audit_df (occ and agg)
        What this function adds is the ceded/net of occ aggregates before
        application of the agg reinsurance. The pure occ and agg parts are in
        reinsurance_audit_df.
        """
        if self.reinsurance_df is None:
            return None
        elif self._reinsurance_report_df is None:
            bit = self.reinsurance_df
            self._reinsurance_report_df = pd.DataFrame({c: xsden_to_meancvskew(bit.loss, bit[c])
                                                        for c in bit.columns[1:]},
                                                       index=['mean', 'cv', 'skew'])
            self._reinsurance_report_df.loc['sd'] = self._reinsurance_report_df.loc['cv'] * \
                                                    self._reinsurance_report_df.loc['mean']
            self._reinsurance_report_df = self._reinsurance_report_df.iloc[[0, 1, 3, 2], :]
        return self._reinsurance_report_df

    def reinsurance_occ_plot(self, axs=None):
        """
        Plots for occurrence reinsurance: occurrence log density and aggregate quantile plot.
        """
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(2 * FIG_W, FIG_H), constrained_layout=True)
            self.figure = fig
        ax0, ax1 = axs.flat

        self.occ_reins_df.filter(regex='p_[scn]').rename(columns=lambda x: x[2:]).plot(ax=ax0, logy=True)
        # TODO: better limit
        xl = ax0.get_xlim()
        l = self.spec['exp_limit']
        if type(l) != float:
            l = np.max(l)
        if l < np.inf:
            xl = [-l / 50, l * 1.025]
        ax0.set(xlim=xl, xlabel='Loss', ylabel='Occurrence log density', title='Occurrence')

        y = self.reinsurance_df.loss.values
        for c in ['gross', 'ceded', 'net']:
            s = self.reinsurance_df[f'p_agg_{c}_occ']
            s[np.abs(s) < 1e-15] = 0
            s = s[::-1].cumsum()[::-1].values
            s[s == 0] = np.nan
            ax1.plot(1 - s, y, label=c)
        ax1.set(xlabel='Probability of non-exceedance', ylabel='Loss', title='Aggregate')
        ax1.legend()

    @property
    def reinsurance_audit_df(self):
        """
        Create and return the _reins_audit_df data frame.
        Read only property.

        :return:
        """
        if self._reinsurance_audit_df is None:
            # really should have one of these anyway...
            if self.agg_density is None:
                logger.warning('Update Aggregate before asking for density_df')
                return None

            ans = []
            keys = []
            if self.occ_reins is not None:
                ans.append(self._reins_audit_df_work(kind='occ'))
                keys.append('occ')
            if self.agg_reins is not None:
                ans.append(self._reins_audit_df_work(kind='agg'))
                keys.append('agg')

            if len(ans):
                self._reinsurance_audit_df = pd.concat(ans, keys=keys, names=['kind', 'share', 'limit', 'attach'])

        return self._reinsurance_audit_df

    def _reins_audit_df_work(self, kind='occ'):
        """
        Apply each re layer separately and aggregate loss and other stats.

        """
        ans = []
        assert self.sev_density is not None

        # reins = self.occ_reins if kind == 'occ' else self.agg_reins

        if kind == 'occ':
            if self.sev_density_gross is None:
                self.sev_density_gross = self.sev_density
            reins = self.occ_reins
            for (s, y, a) in reins:
                c, n, df = self._apply_reins_work([(s, y, a)], self.sev_density_gross, False)
                ans.append(df)
            ans.append(self.occ_reins_df)
        elif kind == 'agg':
            if self.agg_density_gross is None:
                self.agg_density_gross = self.agg_density
            reins = self.agg_reins
            for (s, y, a) in reins:
                c, n, df = self._apply_reins_work([(s, y, a)], self.agg_density_gross, False)
                ans.append(df)
            ans.append(self.agg_reins_df)

        # gup here even though it messes up things later becasuse of sort order
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

    def __init__(self, name, exp_el=0, exp_premium=0, exp_lr=0, exp_en=0, exp_attachment=None, exp_limit=np.inf,
                 sev_name='', sev_a=np.nan, sev_b=0, sev_mean=0, sev_cv=0, sev_loc=0, sev_scale=0,
                 sev_xs=None, sev_ps=None, sev_wt=1, sev_lb=0, sev_ub=np.inf, sev_conditional=True,
                 sev_pick_attachments=None, sev_pick_losses=None,
                 occ_reins=None, occ_kind='',
                 freq_name='', freq_a=0, freq_b=0, freq_zm=False, freq_p0=np.nan,
                 agg_reins=None, agg_kind='',
                 note=''):
        """
        The :class:`Aggregate` distribution class manages creation and calculation of aggregate distributions.
        It allows for very flexible creation of Aggregate distributions. Severity
        can express a limit profile, a mixed severity or both. Mixed frequency types share
        a mixing distribution across all broadcast terms to ensure an appropriate inter-
        class correlation.

        :param name:            name of the aggregate
        :param exp_el:          expected loss or vector
        :param exp_premium:     premium volume or vector  (requires loss ratio)
        :param exp_lr:          loss ratio or vector  (requires premium)
        :param exp_en:          expected claim count per segment (self.n = total claim count)
        :param exp_attachment:  occurrence attachment; None indicates no limit clause, which is treated different
                                from an attachment of zero.
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
        :param sev_lb:          lower bound for severity (length of sev_lb must equal length of sev_ub and weights)
        :param sev_ub:          upper bound for severity
        :param sev_conditional: if True, severity is conditional, else unconditional.
        :param sev_pick_attachments:  if not None, a list of attachment points to define picks
        :param sev_pick_losses:  if not None, a list of losses by layer
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
        # default validation error
        self.validation_eps = VALIDATION_EPS
        self.est_m = 0
        self.est_cv = 0
        self.est_sd = 0
        self.est_var = 0
        self.est_skew = 0
        self.est_sev_m = 0
        self.est_sev_cv = 0
        self.est_sev_sd = 0
        self.est_sev_var = 0
        self.est_sev_skew = 0
        self._valid = None

        self.note = note
        self.program = ''  # can be set externally
        self.en = None  # this is for a sublayer e.g. for limit profile
        self.n = 0  # this is total frequency
        self.attachment = None
        self.limit = None
        self.agg_density = None
        self.sev_density = None
        self.ftagg_density = None
        self.fzapprox = None
        self._var_tvar_function = None
        self._sev_var_tvar_function = None
        self._cdf = None
        self._pdf = None
        self._sev = None
        self.sevs = None
        self.audit_df = None
        self._density_df = None
        self._reinsurance_audit_df = None
        self._reinsurance_report_df = None
        self._reinsurance_df = None
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
        self.sev_density_ceded = None
        self.sev_density_net = None
        self.sev_density_gross = None
        self.agg_density_ceded = None
        self.agg_density_net = None
        self.agg_density_gross = None
        self.sev_pick_attachments = sev_pick_attachments
        self.sev_pick_losses = sev_pick_losses
        self.sev_calc = ""
        self.discretization_calc = ""
        self.normalize = ""
        self.approximation = ''
        self.padding = 0
        self.statistics_df = pd.DataFrame(columns=['name', 'limit', 'attachment', 'sevcv_param', 'el', 'prem', 'lr'] +
                                                  MomentAggregator.column_names() +
                                                  ['mix_cv'])
        self.statistics_total_df = self.statistics_df.copy()
        ma = MomentAggregator(self.freq_moms)

        # overall freq CV with common mixing
        mix_cv = self.freq_a

        # broadcast arrays: force answers all to be arrays (?why only these items?!)
        if not isinstance(exp_el, Iterable):
            exp_el = np.array([exp_el])
        if not isinstance(sev_wt, Iterable):
            sev_wt = np.array([sev_wt])
        if not isinstance(sev_lb, Iterable):
            sev_lb = np.array([sev_lb])
        if not isinstance(sev_ub, Iterable):
            sev_ub = np.array([sev_ub])

        # counter to label components
        r = 0
        # broadcast together and create container for the severity distributions
        if np.sum(sev_wt) == len(sev_wt):
            # do not perform the exp / sev product, in this case
            # broadcast all exposure and sev terms together
            exp_el, exp_premium, exp_lr, en, attachment, limit, \
                sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, \
                sev_wt, sev_lb, sev_ub = \
                np.broadcast_arrays(exp_el, exp_premium, exp_lr, exp_en, exp_attachment, exp_limit,
                                    sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale,
                                    sev_wt, sev_lb, sev_ub)
            exp_el = np.where(exp_el > 0, exp_el, exp_premium * exp_lr)
            all_arrays = zip(exp_el, exp_premium, exp_lr, en, attachment, limit,
                             sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale,
                             sev_wt, sev_lb, sev_ub)
            self.en = en
            self.attachment = attachment
            self.limit = limit
            # these all have the same length because have been broadcast
            n_components = len(exp_el)
            logger.debug(f'Aggregate.__init__ | Broadcast/align: exposures + severity = {len(exp_el)} exp = '
                         f'{len(sev_a)} sevs = {n_components} componets')
            self.sevs = np.empty(n_components, dtype=type(Severity))

            # perform looping creation of severity distribution
            # in this case wts are all 1, so no need to broadcast
            for _el, _pr, _lr, _en, _at, _y, _sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _swt, _slb, _sub in all_arrays:
                assert _swt==1, 'Expect weights all equal to 1'

                # WARNING: note sev_xs and sev_ps are NOT broadcast
                self.sevs[r] = Severity(_sn, _at, _y, _sm, _scv, _sa, _sb, _sloc, _ssc, sev_xs, sev_ps,
                                        _swt, _slb, _sub, sev_conditional)
                sev1, sev2, sev3 = self.sevs[r].moms()

                # input claim count trumps input loss
                if _en > 0:
                    _el = _en * sev1
                elif _el > 0:
                    _en = _el / sev1
                # neither of these options can be triggered, by a dfreq dsev, for example.

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
                # _lr *= _swt  ?? seems wrong
                _en *= _swt

                # accumulate moments
                ma.add_f1s(_en, sev1, sev2, sev3)

                # store
                self.statistics_df.loc[r, :] = \
                    [self.name, _y, _at, _scv, _el, _pr, _lr] + ma.get_fsa_stats(total=False) + [mix_cv]
                r += 1

        else:
            # perform exp / sev product; but there is only one severity distribution
            # it could be a mixture - in which case we need to convert to en input (not loss)
            # and potentially re-weight for excess covers.
            # broadcast exposure terms (el, epremium, en, lr, attachment, limit) and sev terms (sev_) separately
            # then we take an "outer product" of the two parts...
            exp_el, exp_premium, exp_lr, en, attachment, limit = \
                np.broadcast_arrays(exp_el, exp_premium, exp_lr, exp_en, exp_attachment, exp_limit)
            sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, sev_wt, sev_lb, sev_ub = \
                np.broadcast_arrays(sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale,
                                    sev_wt, sev_lb, sev_ub)
            exp_el = np.where(exp_el > 0, exp_el, exp_premium * exp_lr)
            exp_arrays = [exp_el, exp_premium, exp_lr, en, attachment, limit]
            sev_arrays = [sev_name, sev_a, sev_b, sev_mean, sev_cv, sev_loc, sev_scale, sev_lb, sev_ub]
            n_components = len(exp_el) * len(sev_name)
            self.en = np.empty(n_components, dtype=float)
            self.attachment = np.empty(n_components, dtype=float)
            self.limit = np.empty(n_components, dtype=float)
            # all broadcast arrays have the same length, hence:
            logger.debug(
                f'Aggregate.__init__ | Broadcast/product: exposures x severity = {len(exp_el)} x {len(sev_name)} '
                f'=  {n_components}')
            self.sevs = np.empty(n_components, dtype=type(Severity))

            # WARNING: note sev_xs and sev_ps are NOT broadcast
            # In this case, there is only one ground up severity, but it is a mixture. We need to
            # create it ground up to determine new weights, hence we get layer severity,
            # and from that we can deduce layer claim counts and so forth.
            # remember, the weights are irrelvant to Severity EXCEPT for the sev property.
            gup_sevs = []
            for _sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _slb, _sub, _swt in zip(*sev_arrays, sev_wt):
                gup_sevs.append(Severity(_sn, 0, np.inf, _sm, _scv, _sa, _sb, _sloc, _ssc, sev_xs, sev_ps,
                                         _swt, _slb, _sub, sev_conditional))

            # perform looping creation of severity distribution
            for _el, _pr, _lr, _en, _at, _y, in zip(*exp_arrays):
                # adjust weights for excess coverage
                sev_wt0 = sev_wt.copy()
                # attachment can be None, and that needs to percolate through to Severity
                if _at is not None and _at > 0:
                    w1 = sev_wt0 * np.array([s.sf(_at) for s in gup_sevs])
                    sev_wt0 = w1 / w1.sum()

                # store actual sevs in a group (all are also appended to self.sevs) so we can compute the expected value
                # weight still irrelevant; but pull in layer and attaches which must vary for it to be meaningful
                actual_sevs = []
                for _sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _slb, _sub, _swt in zip(*sev_arrays, sev_wt):
                    actual_sevs.append(Severity(_sn, _at, _y, _sm, _scv, _sa, _sb, _sloc, _ssc, sev_xs, sev_ps,
                                                _swt, _slb, _sub, sev_conditional))

                # now we need to figure the severity across the mixture for this particular layer and  attach
                moms = []
                for s in actual_sevs:
                    # just return the first moment
                    moms.append(s.moms())

                # component mean (corresponding to the outside loop) can now be computed
                component_mean = (np.nan_to_num(np.array([m[0] for m in moms])) * sev_wt0).sum()

                # figure claim count if not entered, for the group (at this point we have not weighted down)
                # this forces subsequent calcuations to use (correct) en weighting even if premium or loss are
                # entered
                logger.info(f'{_y} xs {_at}, component_mean = {component_mean}, {[m[0] for m in moms]}')
                if _en == 0:
                    _en = _el / component_mean

                # for cases where a mixture component has no losses in the layer
                # usually because of underflow.
                zero = None

                # break up the total claim count into parts and add sevs to self.sevs
                # need the first variables for sev statistics
                for _sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _slb, _sub, s, _swt, (sev1, sev2, sev3) in \
                        zip(*sev_arrays, actual_sevs, sev_wt0, moms):

                    # store the severity
                    if np.isnan(sev1):
                        if zero is None:
                            zero = Severity('dhistogram', 0, np.inf, 0, 0, 0, 0, 0, 0, [0], [1], 0, np.inf, 0, False)
                        # replace this component with the zero distribution
                        # ignore the (small) weights that are being ignored
                        self.sevs[r] = zero
                        _sn = 'dhistogram'
                        logger.info(f'{_y} xs {_at} on {_ssc} x ({_at}, {_sm}, {_scv}, {_sa}, {_sb}) + {_sloc} '
                                    f' | {_slb} < X le {_sub} '
                                    f'component has sev=({sev1}, {sev2}, {sev3}), '
                                    f' weight = {_swt}; replacing with zero.')
                        sev1 = sev2 = sev3 = 0.0
                    else:
                        self.sevs[r] = s

                    # input claim count, figure total loss for the component
                    if _en > 0:
                        _el = _en * sev1
                    elif _en < 0:
                        # for empirical freq claim count entered as -1
                        _en = np.sum(self.freq_a * self.freq_b)
                        _el = _en * sev1
                    else:
                        logger.info(f'{_y} xs {_at} on {_ssc} x ({_at}, {_sm}, {_scv}, {_sa}, {_sb}) + {_sloc} '
                                    f' | {_slb} < X le {_sub} has '
                                    f'_en = {_en}. Adjusting el to 0.')
                        _el = 0.

                    # if premium compute loss ratio, if loss ratio compute premium
                    # TODO where are these used? are they correct?
                    if _pr > 0:
                        _lr = _el / _pr
                    elif _lr > 0:
                        _pr = _el / _lr

                    # scale for the mix - OK because we have split the exposure and severity components
                    _pr0 = _pr * _swt
                    _el0 = _el * _swt
                    _en0 = _en * _swt

                    # accumulate moments
                    ma.add_f1s(_en0, sev1, sev2, sev3)

                    # store
                    self.statistics_df.loc[r, :] = \
                        [self.name, _y, _at, _scv, _el0, _pr0, _lr] + ma.get_fsa_stats(total=False) + [mix_cv]

                    self.en[r] = _en0
                    self.attachment[r] = _at
                    self.limit[r] = _y

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
        # severity exact moments
        self.sev_m, self.sev_cv, self.sev_skew = self.statistics_total_df.loc['mixed',
        ['sev_m', 'sev_cv', 'sev_skew']]
        self.sev_sd = self.sev_m * self.sev_cv
        self.sev_var = self.sev_sd * self.sev_sd

        # finally, need a report_ser series for Portfolio to consolidate
        self.report_ser = ma.stats_series(self.name, np.max(self.limit), 0.999, remix=True)

    def __repr__(self):
        """
        String version of _repr_html_
        :return:
        """
        # [ORIGINALLY] wrap default with name
        return f'{self.name}, {super(Aggregate, self).__repr__()}'

    def __str__(self):
        """
        Goal: readability

        :return:
        """
        # pull out agg statistics_df
        s = [self.info]
        with pd.option_context('display.width', 200,
                               'display.max_columns', 15,
                               'display.float_format', lambda x: f'{x:,.5g}'):
            # get it on one row
            s.append(str(self.describe))
        # s.append(super().__repr__())
        return '\n'.join(s)

    def more(self, regex):
        """
        More information about methods and properties matching regex

        """
        more(self, regex)

    @property
    def info(self):
        s = [f'aggregate object name    {self.name}',
             f'claim count              {self.n:0,.2f}',
             f'frequency distribution   {self.freq_name}']
        n = len(self.statistics_df)
        if n == 1:
            sv = self.sevs[0]
            if sv.limit == np.inf and sv.attachment == 0:
                _la = 'unlimited'
            else:
                _la = f'{sv.limit:,.0f} xs {sv.attachment:,.0f}'
            s.append(f'severity distribution    {sv.long_name}, {_la}.')
        else:
            s.append(f'severity distribution    {n} components')
        if self.bs > 0:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{int(1 / self.bs)}'
            s.append(f'bs                       {bss}')
            s.append(f'log2                     {self.log2}')
            s.append(f'padding                  {self.padding}')
            s.append(f'sev_calc                 {self.sev_calc}')
            s.append(f'normalize                {self.normalize}')
            s.append(f'approximation            {self.approximation}')
            s.append(f'validation_eps           {self.validation_eps}')
            s.append(f'reinsurance              {self.reinsurance_kinds().lower()}')
            s.append(f'occurrence reinsurance   {self.reinsurance_description("occ").lower()}')
            s.append(f'aggregate reinsurance    {self.reinsurance_description("agg").lower()}')
            s.append(f'validation               {self.explain_validation()  }')
            s.append('')
        return '\n'.join(s)

    def explain_validation(self):
        """
        Explain validation result. Validation computed if needed.
        """
        return explain_validation(self.valid)

    def html_info_blob(self):
        """
        Text top of _repr_html_

        """
        s = [f'<h3>Aggregate object: {self.name}</h3>']
        s.append(f'<p>{self.freq_name} frequency distribution.')
        n = len(self.statistics_df)
        if n == 1:
            sv = self.sevs[0]
            if sv.limit == np.inf and sv.attachment == 0:
                _la = 'unlimited'
            else:
                _la = f'{sv.limit} xs {sv.attachment}'
            s.append(f'Severity {sv.long_name} distribution, {_la}.')
        else:
            s.append(f'Severity with {n} components.')
        if self.bs > 0:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{1 / self.bs:,.0f}'
            s.append(f'Updated with bucket size {bss} and log2 = {self.log2}.</p>')
        if self.agg_density is not None:
            r = self.valid
            if r & Validation.REINSURANCE:
                s.append(f'<p>Validation: reinsurance, n/a.</p>')
            elif r == Validation.NOT_UNREASONABLE:
                s.append('<p>Validation: not unreasonable.</p>')
            else:
                s.append('<p>Validation: <div style="color: #f00; font-weight:bold;">fails</div><pre>\n'
                         f'{self.explain_validation()}</pre></p>')

        return '\n'.join(s)

    def _repr_html_(self):
        """
        For IPython.display

        """
        return self.html_info_blob() + self.describe.to_html()

    def discretize(self, sev_calc, discretization_calc, normalize):
        """
        Discretize the severity distributions and weight.

        ``sev_calc`` describes how the severity is discretize, see `Discretizing the Severity Distribution`_. The
        options are discrete=round, forward, backward or moment.

        ``sev_calc='continuous'`` (same as forward, kept for backwards compatibility) is used when
        you think of the resulting distribution as continuous across the buckets
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

        ``discretization_calc`` controls whether individual probabilities are computed using backward-differences of
        the survival function or forward differences of the distribution function, or both. The former is most
        accurate in the right-tail and the latter for the left-tail of the distribution. We are usually concerned
        with the right-tail, so prefer `survival`. Using `both` takes the greater of the two esimates giving the best
        of both worlds (underflow makes distribution zero in the right-tail and survival zero in the left tail,
        so the maximum gives the best estimate) at the expense of computing time.

        Sensible defaults: sev_calc=discrete, discretization_calc=survival, normalize=True.

        :param sev_calc:  discrete=round, forward, backward, or continuous
               and method becomes discrete otherwise
        :param discretization_calc:  survival, distribution or both; in addition
               the method then becomes survival
        :param normalize: if True, normalize the severity so sum probs = 1. This is generally what you want; but
               when dealing with thick tailed distributions it can be helpful to turn it off.
        :return:
        """

        if sev_calc == 'discrete' or sev_calc == 'round':
            # adj_xs = np.hstack((self.xs - self.bs / 2, np.inf))
            # mass at the end undesirable. can be put in with reinsurance layer in spec
            # note the first bucket is negative
            adj_xs = np.hstack((self.xs - self.bs / 2, self.xs[-1] + self.bs / 2))
        elif sev_calc == 'forward' or sev_calc == 'continuous':
            adj_xs = np.hstack((self.xs, self.xs[-1] + self.bs))
        elif sev_calc == 'backward':
            adj_xs = np.hstack((-self.bs, self.xs))  # , np.inf))
        elif sev_calc == 'moment':
            raise NotImplementedError(
                'Moment matching discretization not implemented. Embrechts says it is not worth it.')
            #
            # adj_xs = np.hstack((self.xs, np.inf))
        else:
            raise ValueError(
                f'Invalid parameter {sev_calc} passed to discretize; options are discrete, continuous, or raw.')

        # in all cases, the first bucket must include the mass at zero
        # Also allow severity to have real support and so want to capture all the way from -inf, hence:
        adj_xs[0] = -np.inf

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
        ix = self.density_df.index.get_indexer([x], 'nearest')[0]
        return self.density_df.iloc[ix, 0]

    def update(self, log2=16, bs=0, recommend_p=RECOMMEND_P, debug=False, **kwargs):
        """
        Convenience function, delegates to update_work. Avoids having to pass xs. Also
        aliased as easy_update for backward compatibility.

        :param log2:
        :param bs:
        :param recommend_p: p value passed to recommend_bucket. If > 1 converted to 1 - 10**-p in rec bucket.
        :param debug:
        :param kwargs:  passed through to update
        :return:
        """
        # guess bucket and update
        if bs == 0:
            bs = round_bucket(self.recommend_bucket(log2, p=recommend_p))
        xs = np.arange(0, 1 << log2, dtype=float) * bs
        # if 'approximation' not in kwargs:
        #     if self.n > 10000:
        #         kwargs['approximation'] = 'slognorm'
        #     else:
        #         kwargs['approximation'] = 'exact'
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
        :param tilt_vector: tilt_vector = np.exp(tilt_amount * np.arange(N)), N=2**log2, and
               tilt_amount * N < 20 recommended
        :param approximation: 'exact' = perform frequency / severity convolution using FFTs.
               'slognorm' or 'sgamma' use a shifted lognormal or shifted gamma approximation.
        :param sev_calc:  discrete=round, forward, backward, or continuous
               and method becomes discrete otherwise
        :param discretization_calc:  survival, distribution or both; in addition
               the method then becomes survival
        :param normalize: if True, normalize the severity so sum probs = 1. This is generally what you want; but
               when dealing with thick tailed distributions it can be helpful to turn it off.
        :param force_severity: make severities even if using approximation, for plotting
        :param debug: run reinsurance in debug model if True.
        :return:
        """
        self._density_df = None  # invalidate
        self._var_tvar_function = None
        self._sev_var_tvar_function = None
        self._valid = None
        self.sev_calc = sev_calc
        self.discretization_calc = discretization_calc
        self.normalize = normalize
        self.approximation = approximation
        self.padding = padding
        self.xs = xs
        self.bs = xs[1]
        # WHOA! WTF
        self.log2 = int(np.log2(len(xs)))

        if type(tilt_vector) == float:
            tilt_vector = np.exp(-tilt_vector * np.arange(2 ** self.log2))

        # make the severity vector: a claim count weighted average of the severities
        if approximation == 'exact' or force_severity:
            wts = self.statistics_df.freq_1 / self.statistics_df.freq_1.sum()
            if self.en.sum() == 0:
                self.en = self.statistics_df.freq_1.values
            self.sev_density = np.zeros_like(xs)
            beds = self.discretize(sev_calc, discretization_calc, normalize)
            for temp, w, a, l, n in zip(beds, wts, self.attachment, self.limit, self.en):
                self.sev_density += temp * w

            # adjust for picks if necessary
            if self.sev_pick_attachments is not None:
                logger.log(WL, 'Adjusting for picks.')
                self.sev_density = self.picks(self.sev_pick_attachments, self.sev_pick_losses)

        if force_severity == 'yes':
            # only asking for severity (used by plot)
            return

        # deal with per occ reinsurance
        # TODO issues with force_severity = False.... get rid of that option entirely?
        # reinsurance converts sev_density to a Series from np.array
        if self.occ_reins is not None:
            if self.sev_density_gross is not None:
                # make the function an involution...
                self.sev_density = self.sev_density_gross
            self.apply_occ_reins(debug)

        if approximation == 'exact':
            if self.n > 100:
                logger.info(f'Claim count {self.n} is high; consider an approximation ')

            if self.n == 0:
                # for dynamics, it is helpful to have a zero risk return zero appropriately
                # z = ft(self.sev_density, padding, tilt_vector)
                self.agg_density = np.zeros_like(self.xs)
                self.agg_density[0] = 1
                # extreme idleness...but need to make sure it is the right shape and type
                self.ftagg_density = ft(self.agg_density, padding, tilt_vector)
            else:
                # usual calculation...this is where the magic happens!
                # have already dealt with per occ reinsurance
                # don't lose accuracy and time by going through this step if freq is fixed 1
                # these are needed when agg is part of a portfolio
                z = ft(self.sev_density, padding, tilt_vector)
                self.ftagg_density = self.freq_pgf(self.n, z)
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
            if self.occ_reins is not None:
                raise ValueError('Per occ reinsurance not supported with approximation')
            if not np.isfinite(self.agg_cv):
                raise ValueError('Cannot fit a distribution with infinite second moment.')
            if self.agg_skew < 0:
                logger.log(WL, 'Negative skewness, ignoring and fitting unshifted distribution.')

            if self.agg_skew == 0:
                self.fzapprox = ss.norm(scale=self.agg_m * self.agg_cv, loc=self.agg_m)
            elif approximation == 'slognorm':
                if np.isfinite(self.agg_skew) and self.agg_skew > 0:
                    shift, mu, sigma = sln_fit(self.agg_m, self.agg_cv, self.agg_skew)
                    self.fzapprox = ss.lognorm(sigma, scale=np.exp(mu), loc=shift)
                else:
                    mu, sigma = ln_fit(self.agg_m, self.agg_cv)
                    self.fzapprox = ss.lognorm(sigma, scale=np.exp(mu))
            elif approximation == 'sgamma':
                if np.isfinite(self.agg_skew) and self.agg_skew > 0:
                    shift, alpha, theta = sgamma_fit(self.agg_m, self.agg_cv, self.agg_skew)
                    self.fzapprox = ss.gamma(alpha, scale=theta, loc=shift)
                else:
                    alpha, theta = gamma_fit(self.agg_m, self.agg_cv)
                    self.fzapprox = ss.gamma(alpha, scale=theta)
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
        # add empirical sev stats
        if self.sev_density is not None:
            _m, _cv, _sk = xsden_to_meancvskew(self.xs, self.sev_density)
        else:
            _m = np.nan
            _cv = np.nan
            _sk = np.nan
        self.audit_df.loc['mixed', 'emp_sev_1'] = _m
        self.audit_df.loc['mixed', 'emp_sev_cv'] = _cv
        self.audit_df.loc['mixed', 'emp_sev_skew'] = _sk
        self.est_sev_m, self.est_sev_cv, self.est_sev_skew = _m, _cv, _sk
        self.est_sev_sd = self.est_sev_m * self.est_sev_cv
        self.est_sev_var = self.est_sev_sd * self.est_sev_sd
        # add empirical agg stats
        _m, _cv, _sk = xsden_to_meancvskew(self.xs, self.agg_density)
        self.audit_df.loc['mixed', 'emp_agg_1'] = _m
        self.audit_df.loc['mixed', 'emp_agg_cv'] = _cv
        self.audit_df.loc['mixed', 'emp_agg_skew'] = _sk
        self.est_m = _m
        self.est_cv = _cv
        self.est_sd = _m * _cv
        self.est_var = self.est_sd ** 2
        self.est_skew = _sk

        # invalidate stored functions
        self._cdf = None

    @property
    def valid(self):
        """
        Check if the model appears valid. An answer of True means the model is "not unreasonable".
        It does not guarantee the model is valid. On the other hand,
        False means it is definitely suspect. (The interpretation is similar to the null hypothesis
        in a statistical test).
        Called and reported automatically by qd for Aggregate objects.

        Checks the relative errors (from ``self.describe``) for:

        * severity mean < eps
        * severity cv < 10 * eps
        * severity skew < 100 * eps (skewness is more difficult to estimate)
        * aggregate mean < eps and < 2 * severity mean relative error (larger values
          indicate possibility of aliasing and that ``bs`` is too small).
        * aggregate cv < 10 * eps
        * aggregate skew < 100 * esp

        The default uses eps = 1e-4 relative error. This can be changed by setting the validation_eps
        variable.

        Test only applied for CV and skewness when they are > 0.

        Run with logger level 20 (info) for more information on failures.

        A Type 1 error (rejecting a valid model) is more likely than Type 2 (failing to reject an invalide one).

        :return: True (interpreted as not unreasonable) if all tests are passed, else False.

        """
        if self._valid is not None:
            return self._valid

        # logger.warning(f'{self.name} CALLING AGG VALID')
        rv = Validation.NOT_UNREASONABLE
        if self.reinsurance_kinds() != "None":
            # cannot validate when there is reinsurance
            # could possibly add manually
            return Validation.REINSURANCE
        df = self.describe.abs()
        try:
            df['Err Skew(X)'] = df['Est Skew(X)'] / df['Skew(X)'] - 1
        except ZeroDivisionError:
            df['Err Skew(X)'] = np.nan
        except TypeError:
            df['Err Skew(X)'] = np.nan
        except KeyError:
            # not updated
            self._valid = Validation.NOT_UPDATED
            return Validation.NOT_UPDATED
        eps = self.validation_eps
        if df.loc['Sev', 'Err E[X]'] > eps:
            logger.info('FAIL: Sev mean error > eps')
            rv |= Validation.SEV_MEAN

        if df.loc['Agg', 'Err E[X]'] > eps:
            logger.info('FAIL: Agg mean error > eps')
            rv |= Validation.AGG_MEAN

        # first line stops failing validation when the agg rel error is very very small
        # default eps is 1e-4, so this is 1e-12. there were examples in the documentation
        # which failed with errors around 1e-14.
        if (abs(df.loc['Agg', 'Err E[X]']) > eps ** 3 and
                abs(df.loc['Sev', 'Err E[X]']) > 0 and
                abs(df.loc['Agg', 'Err E[X]']) > 10 * abs(df.loc['Sev', 'Err E[X]'])):
            logger.info('FAIL: Agg mean error > 10 * sev error')
            rv |= Validation.ALIASING

        try:
            if np.inf > df.loc['Sev', 'CV(X)'] > 0 and df.loc['Sev', 'Err CV(X)'] > 10 * eps:
                logger.info('FAIL: Sev CV error > eps')
                rv |= Validation.SEV_CV

            if np.inf > df.loc['Agg', 'CV(X)'] > 0 and df.loc['Agg', 'Err CV(X)'] > 10 * eps:
                logger.info('FAIL: Agg CV error > eps')
                rv |= Validation.AGG_CV

            if np.inf > np.abs(df.loc['Sev', 'Skew(X)']) > 0 and np.abs(df.loc['Sev', 'Err Skew(X)']) > 100 * eps:
                logger.info('FAIL: Sev skew error > eps')
                rv |= Validation.SEV_SKEW

            if np.inf > np.abs(df.loc['Agg', 'Skew(X)']) > 0 and np.abs(df.loc['Agg', 'Err Skew(X)']) > 100 * eps:
                logger.info('FAIL: Agg skew error > eps')
                rv |= Validation.AGG_SKEW

        except (TypeError, ZeroDivisionError):
            logger.info('Caution: not all validation tests applied')
            pass
        if rv != Validation.NOT_UNREASONABLE:
            logger.info(f'Aggregate {self.name} does not fail any validation: not unreasonable')
        self._valid = rv
        return rv

    def picks(self, attachments, layer_loss_picks, debug=False):
        """
        Adjust the computed severity to hit picks targets in layers defined by a.
        Delegates work to ``utilities.picks_work``. See that function for details.

        """
        # always want to work off gross severity
        if self.sev_density_gross is not None:
            logger.info('Using GROSS severity in picks')
            sd = self.sev_density_gross
        else:
            sd = self.sev_density
        return picks_work(attachments, layer_loss_picks, self.xs, sd, n=self.n,
                          sf=self.sev.sf, debug=debug)

    def freq_pmf(self, log2):
        """
        Return the frequency probability mass function (pmf) computed using 2**log2 buckets.
        Uses self.en to compute the expected frequency. The :class:`Frequency` does not
        know the expected claim count, so this is a method of :class:`Aggregate`.

        """
        n = 1 << log2
        z = np.zeros(n)
        z[1] = 1
        fz = ft(z, 0, None)
        fz = self.freq_pgf(self.en, fz)
        dist = ift(fz, 0, None)
        # remove fuzz
        dist[dist < np.finfo(float).eps] = 0
        if not np.allclose(self.n,  self.en):
            logger.warning(f'Frequency.pmf | n {self.n} != en {self.en}; using en')
        return dist

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
            logger.info(f'Only one net value at {loss} with prob = {value}')
            reins_df['F_net'] = 0.0
            reins_df.loc[loss:, 'F_net'] = value
        else:
            netter_interp = interp1d(sn.index, sn.cumsum(), fill_value=(-100, 1), bounds_error=False)
            reins_df['F_net'] = netter_interp(reins_df.loss)
        if len(sc) == 1:
            loss = sc.index[0]
            value = sc.iloc[0]
            logger.info(f'Only one net value at {loss} with prob = {value}')
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
        Makes sev_density_gross, sev_density_net and sev_density_ceded, and updates sev_density to the requested view.

        Not reflected in statistics df.

        :param debug: More verbose.
        :return:
        """
        # generic function makes netter and ceder functions
        if self.occ_reins is None:
            return
        logger.info('running apply_occ_reins')
        occ_ceder, occ_netter, occ_reins_df = self._apply_reins_work(self.occ_reins, self.sev_density, debug)
        # store stuff
        self.occ_ceder = occ_ceder
        self.occ_netter = occ_netter
        self.occ_reins_df = occ_reins_df
        self.sev_density_gross = self.sev_density
        self.sev_density_net = occ_reins_df['p_net']
        self.sev_density_ceded = occ_reins_df['p_ceded']
        if self.occ_kind == 'ceded to':
            self.sev_density = self.sev_density_ceded
        elif self.occ_kind == 'net of':
            self.sev_density = self.sev_density_net
        else:
            raise ValueError(f'Unexpected kind of occ reinsurace, {self.occ_kind}')

        # see impact on severity moments
        _m2, _cv2, _sk2 = xsden_to_meancvskew(self.xs, self.sev_density)
        self.est_sev_m = _m2
        self.est_sev_cv = _cv2
        self.est_sev_sd = _m2 * _cv2
        self.est_sev_var = self.est_sev_sd ** 2
        self.est_sev_skew = _sk2

    def apply_agg_reins(self, debug=False, padding=1, tilt_vector=None):
        """
        Apply the entire agg reins structure and save output.
        For by layer detail create reins_audit_df.
        Makes sev_density_gross, sev_density_net and sev_density_ceded, and updates sev_density to the requested view.

        Not reflected in statistics df.

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
        logger.info('running apply_agg_reins')
        # store stuff
        self.agg_ceder = agg_ceder
        self.agg_netter = agg_netter
        self.agg_reins_df = agg_reins_df
        self.agg_density_gross = self.agg_density
        self.agg_density_net = agg_reins_df['p_net']
        self.agg_density_ceded = agg_reins_df['p_ceded']
        if self.agg_kind == 'ceded to':
            self.agg_density = self.agg_density_ceded
        elif self.agg_kind == 'net of':
            self.agg_density = self.agg_density_net
        else:
            raise ValueError(f'Unexpected kind of agg reinsurance, {self.agg_kind}')

        # update ft of agg
        self.ftagg_density = ft(self.agg_density, padding, tilt_vector)

        # see impact on moments
        _m2, _cv2, _sk2 = xsden_to_meancvskew(self.xs, self.agg_density)
        # self.audit_df.loc['mixed', 'emp_agg_1'] = _m
        # old_m = self.ex
        self.est_m = _m2
        self.est_cv = _cv2
        self.est_sd = _m2 * _cv2
        self.est_var = self.est_sd ** 2
        self.est_skew = _sk2
        # self.audit_df.loc['mixed', 'emp_agg_cv'] = _cv
        # invalidate quantile function

        logger.info(f'Applying agg reins to {self.name}\tOld mean and cv= {_m:,.3f}\t{_m:,.3f}\n'
                    f'New mean and cv = {_m2:,.3f}\t{_cv2:,.3f}')

    def reinsurance_description(self, kind='both', width=0):
        """
        Text description of the reinsurance.

        :param kind: both, occ, or agg
        :param width: width of text for textwrap.fill; omitted if width==0
        """
        ans = []
        if self.occ_reins is not None and kind in ['occ', 'both']:
            ans.append(self.occ_kind)
            ra = []
            for (s, y, a) in self.occ_reins:
                if np.isinf(y):
                    ra.append(f'{s:,.0%} share of unlimited xs {a:,.0f}')
                else:
                    if s == y:
                        ra.append(f'{y:,.0f} xs {a:,.0f}')
                    else:
                        ra.append(f'{s:,.0%} share of {y:,.0f} xs {a:,.0f}')
            ans.append(' and '.join(ra))
            ans.append('per occurrence')
        if self.agg_reins is not None and kind in ['agg', 'both']:
            if len(ans):
                ans.append('then')
            ans.append(self.agg_kind)
            ra = []
            for (s, y, a) in self.agg_reins:
                if np.isinf(y):
                    ra.append(f'{s:,.0%} share of unlimited xs {a:,.0f}')
                else:
                    if s == y:
                        ra.append(f'{y:,.0f} xs {a:,.0f}')
                    else:
                        ra.append(f'{s:,.0%} share of {y:,.0f} xs {a:,.0f}')
            ans.append(' and '.join(ra))
            ans.append('in the aggregate.')
        if len(ans):
            # capitalize
            s = ans[0]
            s = s[0].upper() + s[1:]
            ans[0] = s
            reins = ' '.join(ans)
        else:
            reins = 'No reinsurance'
        if width:
            reins = fill(reins, width)
        return reins

    def reinsurance_kinds(self):
        """
        Text desciption of kinds of reinsurance applied: None, Occurrence, Aggergate, both.

        :return:
        """
        n = 1 if self.occ_reins is not None else 0
        n += 2 if self.agg_reins is not None else 0
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
        Apply distortion to the aggregate density and append as exag column to density_df.
        # TODO check consistent with other implementations.
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

    def pollaczeck_khinchine(self, rho, cap=0, excess=0, stop_loss=0, kind='index', padding=1):
        """
        Return the Pollaczeck-Khinchine Capital function relating surplus to eventual probability of ruin.
        Assumes frequency is Poisson.

        See Embrechts, Kluppelberg, Mikosch 1.2, page 28 Formula 1.11

        TODO: Should return a named tuple.

        :param rho: rho = prem / loss - 1 is the margin-to-loss ratio
        :param cap: cap = cap severity at cap, which replaces severity with X | X <= cap
        :param excess:  excess = replace severity with X | X > cap (i.e. no shifting)
        :param stop_loss: stop_loss = apply stop loss reinsurance to cap, so  X > stop_loss replaced
          with Pr(X > stop_loss) mass
        :param kind:
        :param padding: for update (the frequency tends to be high, so more padding may be needed)
        :return: ruin vector as pd.Series and function to lookup (no interpolation if
          kind==index; else interp) capitals
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

    # for backwards compatibility
    cramer_lundberg = pollaczeck_khinchine

    def plot(self, axd=None, xmax=0, **kwargs):
        """
        Basic plot with severity and aggregate, linear and log plots and Lee plot.

        :param xmax: Enter a "hint" for the xmax scale. E.g., if plotting gross and net you want all on
               the same scale. Only used on linear scales?
        :param axd:
        :param kwargs: passed to make_mosaic_figure
        :return:
        """
        if axd is None:
            self.figure, axd = make_mosaic_figure('ABC', **kwargs)
        else:
            self.figure = axd['A'].figure

        if self.bs == 1 and self.est_m < 1025:
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
                axd['A'].stem(df.index, df.p_sev, basefmt='none', linefmt='C1-', markerfmt='C1,', label='Severity')
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
            ylim = [1e-15, ylim[1] * 2]
            axd['B'].set(xlim=xlim2, ylim=ylim, title='Log density', yscale='log')
            axd['B'].legend().set(visible=False)

            ax = axd['C']
            # to do: same trimming for p-->1 needed?
            ax.plot(df.F, df.loss, lw=2, label='Aggregate')
            ax.plot(df.p_sev.cumsum(), df.loss, lw=1, label='Severity')
            ax.set(xlim=[-0.02, 1.02], ylim=xlim, title='Quantile (Lee) plot', xlabel='Non-exceeding probability p')
            ax.legend().set(visible=False)

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
            df.loc['agg_m', 'empirical'] = self.audit_df.loc['mixed', 'emp_agg_1']
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
            df.columns = [df.iloc[0, 0]]
            df = df.iloc[1:]
        df.index = df.index.str.split("_", expand=True, )
        df = df.rename(index={'1': 'ex1', '2': 'ex2', '3': 'ex3', 'm': 'mean', np.nan: ''})
        df.index.names = ['component', 'measure']
        df.columns.name = 'name'
        return df

    @property
    def pprogram(self):
        """
        pretty print the program
        """
        return pprint_ex(self.program, 20)

    @property
    def pprogram_html(self):
        """
        pretty print the program to html
        """
        return pprint_ex(self.program, 0, html=True)

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
            df['Est Skew(X)'] = np.nan
            df.loc['Sev', 'Est Skew(X)'] = self.audit_df.loc['mixed', 'emp_sev_skew']
            df.loc['Agg', 'Est Skew(X)'] = self.audit_df.loc['mixed', 'emp_agg_skew']
            df = df[['E[X]', 'Est E[X]', 'Err E[X]', 'CV(X)', 'Est CV(X)', 'Err CV(X)', 'Skew(X)', 'Est Skew(X)']]
        df = df.loc[['Freq', 'Sev', 'Agg']]
        return df

    def recommend_bucket(self, log2=10, p=RECOMMEND_P, verbose=False):
        """
        Recommend a bucket size given 2**N buckets. Not rounded.

        For thick tailed distributions need higher p, try p=1-1e-8.

        If no second moment, throws a ValueError. You just can't guess
        in that situation.

        :param log2: log2 of number of buckets. log2=10 is default.
        :param p: percentile to use to determine needed range. Default is RECOMMEND_P. if > 1 converted to 1-10**-n.
        :param verbose: print out recommended bucket sizes for 2**n for n in {log2, 16, 13, 10}
        :return:
        """
        N = 1 << log2
        if not verbose:
            limit_est = self.limit.max() / N
            if limit_est == np.inf:
                limit_est = 0
                p = max(p, 1 - 10 ** -8)
            moment_est = estimate_agg_percentile(self.agg_m, self.agg_cv, self.agg_skew, p=p) / N
            logger.debug(f'Agg.recommend_bucket | {self.name} moment: {moment_est}, limit {limit_est}')
            return max(moment_est, limit_est)
        else:
            for n in sorted({log2, 16, 13, 10}):
                rb = self.recommend_bucket(n)
                if n == log2:
                    rbr = rb
                print(f'Recommended bucket size with {2 ** n} buckets: {rb:,.3f}')
            if self.bs != 0:
                print(f'Bucket size set with {N} buckets at {self.bs:,.3f}')
            return rbr # noqa

    def aggregate_error_analysis(self, log2, bs2_from=None, **kwargs):
        """
        Analysis of aggregate error across a range of bucket sizes. If ``bs2_from
        is None`` use recommend_bucket plus/mins 3. Note: if distribution does
        not have a second moment, you must enter bs2_from.

        :param log2:
        :param bs2_from: lower bound on bs to use, in log2 terms; estimate using
          ``recommend_bucket`` if not input.
        :param kwargs: passed to ``update``

        """
        # copy of self, updating alters the internal state of an object
        cself = Aggregate(**self.spec)

        if bs2_from is None:
            if cself.agg_cv == np.inf:
                raise ValueError('Distribution must have variance to guess bucket size. '
                                 'Input bs2_from')
            bs = self.recommend_bucket(log2)
            bs = round_bucket(bs)
            bs2 = int(np.log2(bs))
            bss = 2. ** np.arange(bs2 - 3, bs2 + 4)
        else:
            bss = 2. ** np.arange(bs2_from, bs2_from + 7)

        # analytic aggregate mean
        m = cself.agg_m
        # aggregate analysis
        agg_ans = []
        for bs in bss:
            cself.update(bs=bs, log2=log2, approximation='exact', **kwargs)
            agg_ans.append([bs, m, cself.est_m,
                            cself.est_m - m, cself.est_m / m - 1])

        agg_df = pd.DataFrame(agg_ans,
                              columns=['bs', 'agg_m', 'est_m',
                                       'abs_m', 'rel_m', ])
        m = cself.sev_m
        agg_df['rel_h'] = agg_df.bs / 2 / m
        agg_df['rel_total'] = agg_df.rel_h * np.sign(agg_df.rel_m) + agg_df.rel_m
        agg_df = agg_df.set_index('bs')
        agg_df.columns = agg_df.columns.str.split('_', expand=True)
        agg_df.columns.names = ['view', 'stat']
        return agg_df

    def severity_error_analysis(self, sev_calc='round', discretization_calc='survival',
                                normalize=True):
        """
        Analysis of severity component errors, uses the current bs in self.
        Gives detailed, component by component, error analysis of severities.
        Includes discretization error (bs large relative to mean) and
        truncation error (tail integral large).

        Total S shows the aggregate not severity. Generally about self.n * (1 - sum_p)
        (per Feller).

        """
        truncation_point = self.bs * (1 << self.log2)
        wts = self.en / self.n
        beds = self.discretize(sev_calc=sev_calc,
                               discretization_calc=discretization_calc,
                               normalize=normalize)
        sev_ans = []
        total_row = len(self.sevs)
        for i, (s, wt, en, bed) in enumerate(zip(self.sevs, wts, self.en, beds)):
            if len(self.sevs) == 1:
                i = self.name
            # exact
            m = self.statistics.loc[('sev', 'ex1'), i]
            # estimated
            em, _ = xsden_to_meancv(self.xs, bed)
            sev_ans.append([s.long_name,
                            s.limit, s.attachment,
                            truncation_point,
                            s.sf(truncation_point), bed.sum(),
                            wt, en,
                            m, 0,
                            m, em
                            ])
        # the total
        m = self.sev_m
        # attachment is None if the limit clause is missing
        min_attach = np.where(self.attachment==None, 0., self.attachment).min()
        sev_ans.append(['total',
                        self.limit.max(), min_attach,
                        truncation_point,
                        self.sf(truncation_point), self.density_df.p_sev.sum(),
                        1, self.n,
                        m, 0.,
                        m, self.est_sev_m
                        ])

        sev_df = pd.DataFrame(sev_ans,
                              columns=['name',
                                       'limit', 'attachment',
                                       'trunc',
                                       'S', 'sum_p',
                                       'wt', 'en',
                                       'agg_mean', 'agg_wt',
                                       'mean', 'est_mean'
                                       ],
                              index=range(total_row + 1))
        sev_df['agg_mean'] *= sev_df['en']
        sev_df['agg_wt'] = sev_df['agg_mean'] / \
                           sev_df.loc[0:total_row - 1, 'agg_mean'].sum()
        sev_df['abs'] = sev_df['est_mean'] - sev_df['mean']
        sev_df['rel'] = sev_df['abs'] / sev_df['mean']
        sev_df['trunc_error'] = \
            [integral_by_doubling(s.sf, truncation_point) for s in self.sevs] + \
            [integral_by_doubling(self.sev.sf, truncation_point)]
        sev_df['rel_trunc_error'] = sev_df.trunc_error / sev_df['mean']
        sev_df['h_error'] = self.bs / 2
        sev_df['rel_h_error'] = self.bs / 2 / sev_df['mean']

        # compute discretization_err_2 (was a separate function in development)
        xs = np.hstack((self.xs - self.bs / 2, self.xs[-1] + self.bs / 2))
        ans = []
        for s in self.sevs:
            # density at xs
            f = s.pdf(xs)
            # derv of f = -S''
            df = np.gradient(f, self.bs)
            # integral to quadratic adjustment term approx to S
            ans.append(np.sum(df) * self.bs ** 3 / 24)
        ans = pd.Series(ans)

        sev_df['h2_adj'] = np.hstack((ans, 0.))
        sev_df.loc[total_row, 'h2_adj'] = \
            sev_df.loc[0:total_row - 1, ['wt', 'h2_adj']].prod(1).sum()
        sev_df['rel_h2_adj'] = sev_df['h2_adj'] / sev_df['mean']

        return sev_df

    def q(self, p, kind='lower'):
        """
        Return quantile function of density_df.p_total.

        Definition 2.1 (Quantiles)
        x(α) = qα(X) = inf{x ∈ R : P[X ≤ x] ≥ α} is the lower α-quantile of X
        x(α) = qα(X) = inf{x ∈ R : P[X ≤ x] > α} is the upper α-quantile of X.

        ``kind=='middle'`` has been removed.

        :param p:
        :param kind: 'lower' or 'upper'.
        :return:
        """

        if kind == 'middle' and getattr(self, 'middle_warning', 0) == 0:
            # logger.warning(f'kind=middle is deprecated, replacing with kind=lower')
            self.middle_warning = 1

        if kind == 'middle':
            kind = 'lower'

        assert kind in ['lower', 'upper'], 'kind must be lower or upper'

        if self._var_tvar_function is None:
            # revised June 2023
            ser = self.density_df.query('p_total > 0').p_total
            self._var_tvar_function = self._make_var_tvar(ser)

        return self._var_tvar_function[kind](p)

    # for consistency with scipy
    ppf = q

    def _make_var_tvar(self, ser):
        dict_ans = {}
        qf = make_var_tvar(ser)
        dict_ans['upper'] = qf.q_upper
        dict_ans['lower'] = qf.q_lower
        dict_ans['tvar'] = qf.tvar
        return dict_ans

    def q_sev(self, p):
        """
        Compute quantile of severity distribution, returning element in the index.
        Very similar code to q, but only lower quantiles.

        :param p:
        :return:
        """

        if self._sev_var_tvar_function is None:
            # revised June 2023
            ser = self.density_df.query('p_sev > 0').p_sev
            self._sev_var_tvar_function = self._make_var_tvar(ser)

        return self._sev_var_tvar_function['lower'](p)

    def tvar_sev(self, p):
        """
        TVaR of severity - now available for free!

        added June 2023
        """
        if self._var_tvar_function is None:
            ser = self.density_df.query('p_total > 0').p_total
            self._sev_var_tvar_function = self._make_var_tvar(ser)

        return self._var_tvar_function['tvar'](p)

    def tvar(self, p, kind=''):
        """
        Updated June 2023, 0.13.0

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
        if kind != '' and getattr(self, 'c', None) is None:
            logger.warning('kind is no longer used in TVaR, new method equivalent to kind=tail but much faster. '
                           'Argument kind will be removed in the future.')
            self.c = 1

        if kind == 'inverse':
            logger.warning('kind=inverse called...??!!')

        assert self.density_df is not None, 'Must recompute prior to computing tail value at risk.'

        if self._var_tvar_function is None:
            # revised June 2023
            ser = self.density_df.query('p_total > 0').p_total
            self._var_tvar_function = self._make_var_tvar(ser)

        return self._var_tvar_function['tvar'](p)

    def sample(self, n, replace=True):
        """
        Draw a sample of n items from the aggregate distribution. Wrapper around
        pd.DataFrame.sample.


        """

        if self.density_df is None:
            raise ValueError('Must update before sampling.')
        return self.density_df[['loss']].sample(n=n, weights=self.density_df.p_total,
                                                replace=replace, random_state=ar.RANDOM,
                                                ignore_index=True)

    @property
    def sev(self):
        """
        Make exact sf, cdf and pdfs and store in namedtuple for use as sev.cdf etc.
        """
        if self._sev is None:
            SevFunctions = namedtuple('SevFunctions', ['cdf', 'sf', 'pdf'])
            if len(self.sevs) == 1:
                self._sev = SevFunctions(cdf=self.sevs[0].cdf, sf=self.sevs[0].sf,
                                         pdf=self.sevs[0].pdf)
            else:
                # multiple severites, needs more work
                wts = np.array([i.sev_wt for i in self.sevs])
                # for non-broadcast weights the sum is n = number of components; rescale
                if wts.sum() == len(self.sevs):
                    wts = self.statistics_df.freq_1.values
                wts = wts / wts.sum()

                # tried a couple of different approaches here and this is as fast as any
                def _sev_cdf(x):
                    return np.sum([wts[i] * self.sevs[i].cdf(x) for i in range(len(self.sevs))], axis=0)

                def _sev_sf(x):
                    return np.sum([wts[i] * self.sevs[i].sf(x) for i in range(len(self.sevs))], axis=0)

                def _sev_pdf(x):
                    return np.sum([wts[i] * self.sevs[i].pdf(x) for i in range(len(self.sevs))], axis=0)

                self._sev = SevFunctions(cdf=_sev_cdf, sf=_sev_sf, pdf=_sev_pdf)
        return self._sev

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
        :param output: scipy - frozen scipy.stats continuous rv object;
          sev_decl - DecL program for severity (to substituate into an agg ; no name)
          sev_kwargs - dictionary of parameters to create Severity
          agg_decl - Decl program agg T 1 claim sev_decl fixed
          any other string - created Aggregate object
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
        note = f'frozen version of {self.name}'
        return approximate_work(m, cv, skew, name, agg_str, note, approx_type, output)

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



def make_conditional_cdf(lb, ub, plb, pub):
    """
    Decorator to create a conditional CDF from a CDF.
    """
    pr = pub - plb
    def actual_decorator(fzcdf):
        def wrapper(x):
            result = (fzcdf(np.maximum(lb, np.minimum(x, ub))) - plb) / pr
            return result
        return wrapper

    return actual_decorator


def make_conditional_sf(lb, ub, plb, pub):
    """
    Decorator to create a conditional SF from a SF.
    """
    pr = pub - plb
    sub = 1 - pub
    def actual_decorator(fzsf):
        def wrapper(x):
            result = (fzsf(np.maximum(lb, np.minimum(x, ub))) - sub) / pr
            return result
        return wrapper

    return actual_decorator


def make_conditional_pdf(lb, ub, plb, pub):
    """
    Decorator to make conditional PDF from PDF.
    """
    pr = pub - plb
    def actual_decorator(fzpdf):

        def wrapper(x):
            result = np.where(x < lb, 0,
                            np.where(x > ub, 0,
                                     fzpdf(x) / pr))
            return result
        return wrapper

    return actual_decorator


def make_conditional_isf(lb, ub, plb, pub):
    """
    Decorator to make conditional ISF from ISF.
    """
    slb = 1 - plb
    sub = 1 - pub

    def actual_decorator(fzisf):

        def wrapper(s):
            result = np.where(s == 1, lb,
                            np.where(s == 0, ub,
                                     fzisf(s * slb + (1 - s) * sub)))
            return result
        return wrapper

    return actual_decorator


def make_conditional_ppf(lb, ub, plb, pub):
    """
    Decorator to make conditional PPF from PPF.
    """
    def actual_decorator(fzppf):

        def wrapper(p):
            result = np.where(p == 0, lb,
                            np.where(p == 1, ub,
                                     fzppf((1 - p) * plb + p * pub)))
            return result
        return wrapper

    return actual_decorator


class Severity(ss.rv_continuous):

    def __init__(self, sev_name, exp_attachment=None, exp_limit=np.inf, sev_mean=0, sev_cv=0, sev_a=np.nan, sev_b=0,
                 sev_loc=0, sev_scale=0, sev_xs=None, sev_ps=None, sev_wt=1, sev_lb=0, sev_ub=np.inf,
                 sev_conditional=True, name='', note=''):
        """
        A continuous random variable, subclasses ``scipy.statistics_df.rv_continuous``,
        adding layer and attachment functionality. It overrides

        - ``cdf``
        - ``pdf``
        - ``isf``
        - ``ppf``
        - ``sf``
        - ``stats``

        See `scipy.stats continuous rvs <https://docs.scipy.org/doc/scipy/tutorial/stats/continuous.html>`_ for
        more details about available distributions. The following distributions with two shape parameters
        are supported:

        - Burr (``burr``)
        - Generalized Pareto (``genpareto``)
        - Generalized gamma (``gengamma``)

        It is easy to add others in the code below. With two shape parameters the mean cv input format
        is not available.

        See code in Problems and Solutions to extract distributions from scipy stats by introsepection.

        :param sev_name: scipy statistics_df continuous distribution | (c|d)histogram  cts or discerte | fixed
        :param exp_attachment: None if layer is missing, distinct from 0; if a = 0 then losses are
               conditional on  X>a, if a = None then losses are conditional on X>=0
        :param exp_limit:
        :param sev_mean:
        :param sev_cv:
        :param sev_a: first shape parameter
        :param sev_b: second shape parameter (e.g., beta)
        :param sev_loc: scipy.stats location parameter
        :param sev_scale: scipy.stats scale parameter
        :param sev_xs: for fixed or histogram classes
        :param sev_ps:
        :param sev_wt: this is not used directly; but it is convenient to pass it in and ignore it because sevs are
          implicitly created with sev_wt=1.
        :param sev_conditional: conditional or unconditional; for severities use conditional
        :param name: name of the severity object
        :param note: optional note.
        """

        from .portfolio import Portfolio

        super().__init__(self, name=sev_name)

        self.program = ''  # may be set externally
        self.limit = exp_limit
        self.attachment = 0 if exp_attachment is None else exp_attachment
        # need to keep this information, otherwise imposssible for this object
        # to determine its treatment of zero
        self.exp_attachment = exp_attachment
        self.detachment = exp_limit + self.attachment
        self.fz = None
        self.pattach = 0
        self.moment_pattach = 0
        self.pdetach = 0
        self.conditional = sev_conditional
        self.sev_name = sev_name
        # only used when created as sev ONE blah...
        self.name = name
        # distribution name
        self.long_name = sev_name
        self.note = note
        self.sev1 = self.sev2 = self.sev3 = None
        self.sev_wt = sev_wt
        self.sev_loc = sev_loc
        self.sev_lb = sev_lb
        self.sev_ub = sev_ub
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
                # for empirical, with cts histogram xs and ps do not have the same size
                if self.sev_name != 'chistogram':
                    logger.warning(f'Severity.init | {sev_name} sev_xs and sev_ps cannot be broadcast.')
                xs = np.array(sev_xs)
                ps = np.array(sev_ps)
            if not np.isclose(np.sum(ps), 1.0):
                logger.error(f'Severity.init | {sev_name} histogram/fixed severity with probs do not sum to 1, '
                             f'{np.sum(ps)}')
            # need to exp_limit distribution
            exp_limit = min(np.min(exp_limit), xs.max())
            if sev_name == 'chistogram':
                # TODO: check this code still works!
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
                # need to be careful in case input has duplicates - these need removing
                # it is also possible the user has included values < 0
                # these must be replaced with zero. The next
                # code hanndles both cases
                xs, ps = validate_discrete_distribution(xs, ps)
                # discrete histogram: point masses at xs's; moments must be computed after dealing
                # with negative entries
                self.sev1 = np.sum(xs * ps)
                self.sev2 = np.sum(xs ** 2 * ps)
                self.sev3 = np.sum(xs ** 3 * ps)
                # Jan 2023: want the increment to be smaller than any reasonable
                # bucket size so buckets are not "split". The function max_log2
                # computes that d.
                d = max_log2(np.max(xs))
                logger.info(f'Severity.init | {sev_name} d={d}')
                xss = np.sort(np.hstack((xs - 2 ** -d, xs)))
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

        elif sev_name in ['anglit', 'arcsine', 'cauchy', 'cosine', 'expon', 'gilbrat', 'gumbel_l',
                          'gumbel_r', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant',
                          'kstwobign', 'laplace', 'levy', 'levy_l', 'logistic', 'maxwell', 'moyal',
                          'norm', 'rayleigh', 'semicircular', 'uniform', 'wald']:
            # distributions with no shape parameters
            #     Normal (and possibly others) does not have a shape parameter
            if sev_loc == 0 and sev_mean > 0:
                sev_loc = sev_mean
            if sev_scale == 0 and sev_cv > 0:
                sev_scale = sev_cv * sev_loc
            gen = getattr(ss, sev_name)
            self.fz = gen(loc=sev_loc, scale=sev_scale)

        elif sev_name in ['beta', 'betaprime', 'burr', 'burr12', 'crystalball', 'exponweib', 'f', 'gengamma',
                          'geninvgauss', 'johnsonsb', 'johnsonsu', 'kappa4', 'levy_stable', 'loguniform',
                          'mielke', 'nct', 'ncx2', 'norminvgauss', 'powerlognorm', 'reciprocal',
                          'studentized_range', 'trapezoid', 'trapz', 'truncnorm']:
            # distributions with two shape parameters require specific inputs
            # https://en.wikipedia.org/wiki/Beta_distribution#Two_unknown_parameters
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
            assert sev_name in ['alpha', 'argus', 'bradford', 'chi', 'chi2', 'dgamma', 'dweibull', 'erlang',
                                'exponnorm', 'exponpow', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm',
                                'gamma', 'genextreme', 'genhalflogistic', 'genlogistic', 'gennorm', 'genpareto',
                                'gompertz', 'halfgennorm', 'invgamma', 'invgauss', 'invweibull', 'kappa3', 'ksone',
                                'kstwo', 'laplace_asymmetric', 'loggamma', 'loglaplace', 'lognorm', 'lomax',
                                'nakagami', 'pareto', 'pearson3', 'powerlaw', 'powernorm', 'rdist', 'recipinvgauss',
                                'rice', 'skewcauchy', 'skewnorm', 't', 'triang', 'truncexpon', 'tukeylambda',
                                'vonmises', 'vonmises_line', 'weibull_max', 'weibull_min', 'wrapcauchy']
            if np.isnan(sev_a) and sev_cv > 0:
                sev_a, _ = self.cv_to_shape(sev_cv)
                logger.info(f'sev_a not set, determined as {sev_a} shape from sev_cv {sev_cv}')
            elif np.isnan(sev_a):
                raise ValueError('sev_a not set and sev_cv=0 is invalid, no way to determine shape.')
            # have sev_a, now assemble distribution
            if sev_mean > 0:
                logger.info(f'creating with sev_mean={sev_mean} and sev_loc={sev_loc}')
                sev_scale, self.fz = self.mean_to_scale(sev_a, sev_mean, sev_loc)
            elif sev_scale > 0 and sev_mean == 0:
                logger.info(f'creating with sev_scale={sev_scale} and sev_loc={sev_loc}')
                gen = getattr(ss, sev_name)
                self.fz = gen(sev_a, scale=sev_scale, loc=sev_loc)
            else:
                raise ValueError('sev_scale and sev_mean both equal zero.')

        # apply decorators to make conditional on between lb and ub
        if not(sev_lb == 0 and sev_ub == np.inf):
            plb = self.fz.cdf(sev_lb)
            pub = self.fz.cdf(sev_ub)
            self.fz.cdf = make_conditional_cdf(sev_lb, sev_ub, plb, pub)(self.fz.cdf)
            self.fz.sf  = make_conditional_sf (sev_lb, sev_ub, plb, pub)(self.fz.sf)   # noqa
            self.fz.isf = make_conditional_isf(sev_lb, sev_ub, plb, pub)(self.fz.isf)
            self.fz.ppf = make_conditional_ppf(sev_lb, sev_ub, plb, pub)(self.fz.ppf)
            self.fz.pdf = make_conditional_pdf(sev_lb, sev_ub, plb, pub)(self.fz.pdf)

        if self.detachment == np.inf:
            self.pdetach = 0
        else:
            self.pdetach = self.fz.sf(self.detachment)

        if exp_attachment is None:
            # this behaviour makes layers consistent: if a = 0 then losses are
            # conditional on  X>a, if a = None then losses are conditional on X>=0
            # self.attachment needs to be a number because it is used in all
            # the distribution functions
            if sev_name.endswith('histogram'):
                self.moment_pattach = self.pattach = 1
            else:
                # continuous distribution: need to condition on X >= 0
                # which is the same as X > 0; however, here we need
                # the correct p for the integral of isf to determine
                # the moments. Thus, we need two values:
                self.moment_pattach = self.fz.sf(self.attachment)
                self.pattach = 1
        else:
            # None has already been mapped to 0
            self.moment_pattach = self.pattach = self.fz.sf(self.attachment)

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
            return np.where(x >= self.limit, 0,
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
            return np.where(x >= self.limit, 1,
                            np.where(x < 0, 0,
                                     (self.fz.cdf(x + self.attachment) - (1 - self.pattach)) / self.pattach))
        else:
            return np.where(x < 0, 0,
                            np.where(x == 0, 1 - self.pattach,
                                     np.where(x > self.limit, 1,
                                              self.fz.cdf(x + self.attachment, *args))))

    def _sf(self, x, *args):
        if self.conditional:
            return np.where(x >= self.limit, 0,
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

    # expensive call, convenient to cache
    @lru_cache
    def moms(self):
        """
        Revised moments for Severity class. Trying to compute moments of

            X(a,d) = min(d, (X-a)+)

            ==> E[X(a,d)^n] = int_a^d (x-a)^n f(x) dx + (d-a)^n S(d).

        Let x = q(p), F(x) = p, f(x)dx = dp. for 1,2,...n(<=3).

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
            argkw = dict(limit=100, epsrel=1e-6 if level == 1 else 1e-4, full_output=1)
            ex = quad(f, lower, upper, **argkw)
            if len(ex) == 4 or ex[0] == np.inf:  # 'The integral is probably divergent, or slowly convergent.':
                msg = ex[-1].replace("\n", " ") if ex[-1] == str else "no message"
                logger.info(
                    f'E[X^{level}]: ansr={ex[0]}, error={ex[1]}, steps={ex[2]["last"]}; message {msg} -> splitting integral')
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
                    logger.info(
                        f'Severity.moms | [1e-16, {ϵ}] split EX^{level}: ansr={exa[0]}, error={exa[1]}, steps={exa[2]["last"]}')
                    logger.info(
                        f'Severity.moms | [{ϵ}, {upper}] split EX^{level}: ansr={exb[0]}, error={exb[1]}, steps={exb[2]["last"]}')
                    ex = (exa[0] + exb[0], exa[1] + exb[1])
                    # if exa[2]['last'] < argkw['limit'] and exb[2]['last'] < argkw['limit']:
                    #     ex = (exa[0] + exb[0], exa[1] + exb[1])
                    # else:
                    #     # reached limit, unreliable
                    #     ex = (np.nan, np.nan)
            logger.info(f'E[X^{level}]={ex[0]}, error={ex[1]}, est rel error={ex[1] / ex[0] if ex[0] != 0 else np.inf}')
            return ex[:2]

        ex1a = None
        # we integrate isf not q, so upper and lower are swapped
        if self.attachment == 0:
            # continuous distributions with negative support may have mass at zero
            upper = min(1, self.moment_pattach)
        else:
            upper = self.fz.sf(self.attachment)
        if self.detachment == np.inf:
            lower = 0
        else:
            lower = self.fz.sf(self.detachment)

        if self.detachment == np.inf and self.sev_name not in ['chistogram', 'dhistogram']:
            # figure which moments actually exist
            moments_finite = list(map(lambda x: not (np.isinf(x) or np.isnan(x)), self.fz.stats('mvs')))
        else:
            moments_finite = [True, True, True]
        # logger.info(str(moments_finite))

        # compute moments: histograms are tricky to integrate and we know the answer already...so
        if self.attachment == 0 and self.detachment == np.inf and self.sev_name.endswith('histogram'):
            ex1 = self.sev1
            ex2 = self.sev2
            ex3 = self.sev3

        elif self.sev_name in ['lognorm', 'pareto', 'gamma', 'expon'] and self.sev_loc == 0 \
                and self.sev_lb == 0 and self.sev_ub == np.inf:
            # have exact and note this computes the answer directly
            # no need for the subsequent adjustment
            logger.info('Analytic moments')
            ma = moms_analytic(self.fz, self.limit, self.attachment, 3)
            ex1a, ex2a, ex3a = ma[1:]
            if not moments_finite[0]:
                ex1a = np.inf
            if not moments_finite[1]:
                ex2a = np.inf
            if not moments_finite[2]:
                ex3a = np.inf
        else:
            logger.info('Numerical moments')
            continue_calc = True
            max_rel_error = 1e-3
            if moments_finite[0]:
                ex1 = safe_integrate(self.fz.isf, lower, upper, 1)
                # TODO: there is a potential issue here if the mean is very small
                # a small abs error can appear as a large relative error
                # however, that circumstance should not occur
                # (saw with agg T 1 claim sev norm fixed; before we fixed the range
                # of integration for moments to be [0.5, 1] instead of [0,1])
                if ex1[0] != 0 and ex1[1] / ex1[0] < max_rel_error:
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
                if ex2[1] / ex2[0] < max_rel_error:  # and ex2[1] < 0.01 * ex1**2:
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

        # adjust if not determined by exact formula
        if ex1a is None:
            dma = self.detachment - self.attachment
            uml = upper - lower
            a = self.attachment
            if a > 0:
                ex1a = ex1 - a * uml
                ex2a = ex2 - 2 * a * ex1 + a ** 2 * uml
                ex3a = ex3 - 3 * a * ex2 + 3 * a ** 2 * ex1 - a ** 3 * uml
            else:
                # a = 0: check if continous with mass at zero, and no limit clause (exp_attachment is None)
                # in which case scale moments down by pattach
                if self.exp_attachment is None and not self.sev_name.endswith('histogram'):
                    ex1a = self.pattach * ex1
                    ex2a = self.pattach * ex2
                    ex3a = self.pattach * ex3
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

    def plot(self, n=100, axd=None, figsize=(2 * FIG_W, 2 * FIG_H), layout='AB\nCD'):
        """
        Quick plot, updated for 0.9.3 with mosaic and no grid lines. (F(x), x) plot
        replaced with log density plot.

        :param n: number of points to plot.
        :param axd: axis dictionary, if None, create new figure. Must have keys 'A', 'B', 'C', 'D'.
        :param figsize: (width, height) in inches.
        :param layout: the subplot_mosaic layout of the figure. Default is 'AB\nCD'.
        :return:
        """
        # TODO better coordination of figsize! Better axis formats and ranges.

        xs = np.linspace(0, self._isf(1e-4), n)
        xs2 = np.linspace(0, self._isf(1e-12), n)

        if axd is None:
            f = plt.figure(constrained_layout=True, figsize=figsize)
            axd = f.subplot_mosaic(layout)

        ds = 'steps-post' if self.sev_name == 'dhistogram' else 'default'

        ax = axd['A']
        ys = self._pdf(xs)
        ax.plot(xs, ys, drawstyle=ds)
        ax.set(title='Probability density', xlabel='Loss')
        yl = ax.get_ylim()

        ax = axd['B']
        ys2 = self._pdf(xs2)
        ax.plot(xs2, ys2, drawstyle=ds)
        ax.set(title='Log density', xlabel='Loss', yscale='log', ylim=[1e-14, 2 * yl[1]])

        ax = axd['C']
        ys = self._cdf(xs)
        ax.plot(xs, ys, drawstyle=ds)
        ax.set(title='Probability distribution', xlabel='Loss', ylim=[-0.025, 1.025])

        ax = axd['D']
        ax.plot(ys, xs, drawstyle=ds)
        ax.set(title='Quantile (Lee) plot', xlabel='Non-exceeding probability $p$', xlim=[-0.025, 1.025])
