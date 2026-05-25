"""Moment accumulation across mixture components and conversion between central,
non-central, and factorial moment representations."""

import itertools
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MomentAggregator:
    """
    Purely accumulates moments. Used by ``Aggregate`` and ``Portfolio`` to
    feed their ``stats_df``. Not frequency aware.

    Internal variables agg, sev, freq, tot = running total, 1, 2, 3 = noncentral moments, E(X^k)

    :param freq_moms: function of one variable returning first three noncentral moments of the underlying
            frequency distribution

    """
    __slots__ = ['freq_1', 'freq_2', 'freq_3', 'sev_1', 'sev_2', 'sev_3', 'agg_1', 'agg_2', 'agg_3',
                 'tot_freq_1', 'tot_freq_2', 'tot_freq_3',
                 'tot_sev_1', 'tot_sev_2', 'tot_sev_3',
                 'tot_agg_1', 'tot_agg_2', 'tot_agg_3', 'freq_moms'
                 ]

    def __init__(self, freq_moms=None):
        # accumulators
        self.agg_1 = self.agg_2 = self.agg_3 = 0
        self.tot_agg_1 = self.tot_agg_2 = self.tot_agg_3 = 0
        self.freq_1 = self.freq_2 = self.freq_3 = 0
        self.tot_freq_1 = self.tot_freq_2 = self.tot_freq_3 = 0
        self.sev_1 = self.sev_2 = self.sev_3 = 0
        self.tot_sev_1 = self.tot_sev_2 = self.tot_sev_3 = 0
        # function to comptue frequency moments, hence can call add_f1s(...)
        self.freq_moms = freq_moms

    def add_fs(self, f1, f2, f3, s1, s2, s3):
        """
        accumulate new moments defined by f and s

        used by Portfolio

        compute agg for the latest values

        :param f1:
        :param f2:
        :param f3:
        :param s1:
        :param s2:
        :param s3:
        :return:
        """
        self.freq_1 = f1
        self.freq_2 = f2
        self.freq_3 = f3

        # load current sev moments
        self.sev_1 = s1
        self.sev_2 = s2
        self.sev_3 = s3

        # accumulate frequency
        self.tot_freq_1, self.tot_freq_2, self.tot_freq_3 = \
            self.cumulate_moments(self.tot_freq_1, self.tot_freq_2, self.tot_freq_3, f1, f2, f3)

        # severity
        self.tot_sev_1 = self.tot_sev_1 + f1 * s1
        self.tot_sev_2 = self.tot_sev_2 + f1 * s2
        self.tot_sev_3 = self.tot_sev_3 + f1 * s3

        # aggregate
        self.agg_1, self.agg_2, self.agg_3 = self.agg_from_fs(f1, f2, f3, s1, s2, s3)

        # finally accumulate the aggregate
        self.tot_agg_1, self.tot_agg_2, self.tot_agg_3 = \
            self.cumulate_moments(self.tot_agg_1, self.tot_agg_2, self.tot_agg_3, self.agg_1, self.agg_2, self.agg_3)

    def add_fs2(self, f1, vf, s1, vs):
        """
        accumulate based on first two moments entered as mean and variance - this
        is how questions are generally written.

        """
        f2 = vf + f1 * f1
        s2 = vs + s1 * s1
        self.add_fs(f1, f2, 0., s1, s2, 0.)

    def add_f1s(self, f1, s1, s2, s3):
        """
        accumulate new moments defined by f1 and s - fills in f2, f3 based on
        stored frequency distribution

        used by Aggregate

        compute agg for the latest values


        :param f1:
        :param s1:
        :param s2:
        :param s3:
        :return:
        """

        # fill in the frequency moments and store away
        f1, f2, f3 = self.freq_moms(f1)
        self.add_fs(f1, f2, f3, s1, s2, s3)

    def get_fsa_stats(self, total, remix=False):
        """
        Get the current f x s = agg flat moment list.
        total = true use total else, current
        remix = true for total only, re-compute freq moments based on total freq 1

        :param total: binary
        :param remix: combine all sevs and recompute the freq moments from total freq
        :return:
        """

        if total:
            if remix:
                # recompute the frequency moments; all local variables
                f1 = self.tot_freq_1
                f1, f2, f3 = self.freq_moms(f1)
                s1, s2, s3 = self.tot_sev_1 / f1, self.tot_sev_2 / f1, self.tot_sev_3 / f1
                a1, a2, a3 = self.agg_from_fs(f1, f2, f3, s1, s2, s3)
                return [f1, f2, f3, *self.static_moments_to_mcvsk(f1, f2, f3),
                        s1, s2, s3, *self.static_moments_to_mcvsk(s1, s2, s3),
                        a1, a2, a3, *self.static_moments_to_mcvsk(a1, a2, a3)]
            else:
                # running total, not adjusting freq cv
                return [self.tot_freq_1, self.tot_freq_2, self.tot_freq_3, *self.moments_to_mcvsk('freq', True),
                        self.tot_sev_1 / self.tot_freq_1, self.tot_sev_2 / self.tot_freq_1,
                        self.tot_sev_3 / self.tot_freq_1, *self.moments_to_mcvsk('sev', True),
                        self.tot_agg_1, self.tot_agg_2, self.tot_agg_3, *self.moments_to_mcvsk('agg', True)]
        else:
            # not total
            return [self.freq_1, self.freq_2, self.freq_3, *self.moments_to_mcvsk('freq', False),
                    self.sev_1, self.sev_2, self.sev_3, *self.moments_to_mcvsk('sev', False),
                    self.agg_1, self.agg_2, self.agg_3, *self.moments_to_mcvsk('agg', False)]

    @staticmethod
    def factorial_to_noncentral(f1, f2, f3):
        # eg. Panjer Willmot p 29, 2.3.13
        nc2 = f2 + f1
        nc3 = f3 + 3 * f2 + f1
        return nc2, nc3

    @staticmethod
    def agg_from_fs(f1, f2, f3, s1, s2, s3):
        """
        aggregate moments from freq and sev components


        :param f1:
        :param f2:
        :param f3:
        :param s1:
        :param s2:
        :param s3:
        :return:
        """
        return f1 * s1, \
               f1 * s2 + (f2 - f1) * s1 ** 2, \
               f1 * s3 + f3 * s1 ** 3 + 3 * (f2 - f1) * s1 * s2 + (- 3 * f2 + 2 * f1) * s1 ** 3

    @staticmethod
    def agg_from_fs2(f1, vf, s1, vs):
        """
        aggregate moments from freq and sev ex and var x


        :param f1:
        :param vf:
        :param s1:
        :param vs:
        :return:
        """
        f2 = vf + f1 * f1
        s2 = vs + s1 * s1
        a1, a2, a3 = MomentAggregator.agg_from_fs(f1, f2, 0., s1, s2, 0.)
        mw = MomentWrangler()
        mw.noncentral = a1, a2, a3
        # drop skewness
        return mw.stats[:-1]

    def moments_to_mcvsk(self, mom_type, total=True):
        """
        convert noncentral moments into mean, cv and skewness
        mom_type = agg | freq | sev
        delegates work

        :param mom_type:
        :param total:
        :return:
        """

        if mom_type == 'agg':
            if total:
                return MomentAggregator.static_moments_to_mcvsk(self.tot_agg_1, self.tot_agg_2, self.tot_agg_3)
            else:
                return MomentAggregator.static_moments_to_mcvsk(self.agg_1, self.agg_2, self.agg_3)
        elif mom_type == 'freq':
            if total:
                return MomentAggregator.static_moments_to_mcvsk(self.tot_freq_1, self.tot_freq_2, self.tot_freq_3)
            else:
                return MomentAggregator.static_moments_to_mcvsk(self.freq_1, self.freq_2, self.freq_3)
        elif mom_type == 'sev':
            if total:
                return MomentAggregator.static_moments_to_mcvsk(self.tot_sev_1 / self.tot_freq_1,
                                                                self.tot_sev_2 / self.tot_freq_1,
                                                                self.tot_sev_3 / self.tot_freq_1)
            else:
                return MomentAggregator.static_moments_to_mcvsk(self.sev_1, self.sev_2, self.sev_3)

    def moments(self, mom_type, total=True):
        """
        vector of the moments; convenience function

        :param mom_type:
        :param total:
        :return:
        """
        if mom_type == 'agg':
            if total:
                return self.tot_agg_1, self.tot_agg_2, self.tot_agg_3
            else:
                return self.agg_1, self.agg_2, self.agg_3
        elif mom_type == 'freq':
            if total:
                return self.tot_freq_1, self.tot_freq_2, self.tot_freq_3
            else:
                return self.freq_1, self.freq_2, self.freq_3
        elif mom_type == 'sev':
            if total:
                return self.tot_sev_1 / self.tot_freq_1, self.tot_sev_2 / self.tot_freq_1, \
                       self.tot_sev_3 / self.tot_freq_1
            else:
                return self.sev_1, self.sev_2, self.sev_3

    @staticmethod
    def cumulate_moments(m1, m2, m3, n1, n2, n3):
        """
        Moments of sum of indepdendent variables

        :param m1: 1st moment, E(X)
        :param m2: 2nd moment, E(X^2)
        :param m3: 3rd moment, E(X^3)
        :param n1:
        :param n2:
        :param n3:
        :return:
        """
        # figure out moments of the sum
        t1 = m1 + n1
        t2 = m2 + 2 * m1 * n1 + n2
        t3 = m3 + 3 * m2 * n1 + 3 * m1 * n2 + n3
        return t1, t2, t3

    @staticmethod
    def static_moments_to_mcvsk(ex1, ex2, ex3):
        """
        returns mean, cv and skewness from non-central moments

        :param ex1:
        :param ex2:
        :param ex3:
        :return:
        """
        m = ex1
        var = ex2 - ex1 ** 2
        # rounding errors...
        if np.allclose(var, 0):
            var = 0
        if var < 0:
            logger.error(f'MomentAggregator.static_moments_to_mcvsk | weird var < 0 = {var}; ex={ex1}, ex2={ex2}')
        sd = np.sqrt(var)
        if m == 0:
            cv = np.nan
            logger.info('MomentAggregator.static_moments_to_mcvsk | encountered zero mean, called with '
                         f'{ex1}, {ex2}, {ex3}')
        else:
            cv = sd / m
        if sd == 0:
            skew = np.nan
        else:
            skew = (ex3 - 3 * ex1 * ex2 + 2 * ex1 ** 3) / sd ** 3
        return m, cv, skew

    @staticmethod
    def column_names():
        """
        Flat moment names for f x s = a (the order matches ``get_fsa_stats``;
        bridged to the canonical ``(component, measure)`` MultiIndex by
        :func:`aggregate.distributions._flat_col_to_stats_index` during
        ``stats_df`` builds).

        :return:
        """

        return [i + j for i, j in itertools.product(['freq', 'sev', 'agg'], [f'_{i}' for i in range(1, 4)] +
                                                    ['_m', '_cv', '_skew'])]


class MomentWrangler:
    """
    Conversion between central, noncentral and factorial moments

    Input any one and ask for any translation.

    Stores moments as noncentral internally

    """

    __slots__ = ['_central', '_noncentral', '_factorial']

    def __init__(self):
        self._central = None
        self._noncentral = None
        self._factorial = None

    @property
    def central(self):
        return self._central

    @central.setter
    def central(self, value):
        self._central = value
        ex1, ex2, ex3 = value
        # p 43, 1.241
        self._noncentral = (ex1, ex2 + ex1 ** 2, ex3 + 3 * ex2 * ex1 + ex1 ** 3)
        self._make_factorial()

    @property
    def noncentral(self):
        return self._noncentral

    @noncentral.setter
    def noncentral(self, value):
        self._noncentral = value
        self._make_factorial()
        self._make_central()

    @property
    def factorial(self):
        return self._factorial

    @factorial.setter
    def factorial(self, value):
        self._factorial = value
        ex1, ex2, ex3 = value
        # p 44, 1.248
        self._noncentral = (ex1, ex2 + ex1, ex3 + 3 * ex2 + ex1)
        self._make_central()

    @property
    def stats(self):
        m, v, c3 = self._central
        sd = np.sqrt(v)
        if m == 0:
            cv = np.nan
        else:
            cv = sd / m
        if sd == 0:
            skew = np.nan
        else:
            skew = c3 / sd ** 3
        #         return pd.Series((m, v, sd, cv, skew), index=('EX', 'Var(X)', 'SD(X)', 'CV(X)', 'Skew(X)'))
        # shorter names are better
        return pd.Series((m, v, sd, cv, skew), index=('ex', 'var', 'sd', 'cv', 'skew'))

    @property
    def mcvsk(self):
        m, v, c3 = self._central
        sd = np.sqrt(v)
        if m == 0:
            cv = np.nan
        else:
            cv = sd / m
        if sd == 0:
            skew = np.nan
        else:
            skew = c3 / sd ** 3
        return m, cv, skew

    def _make_central(self):
        ex1, ex2, ex3 = self._noncentral
        # p 42, 1.240
        self._central = (ex1, ex2 - ex1 ** 2, ex3 - 3 * ex2 * ex1 + 2 * ex1 ** 3)

    def _make_factorial(self):
        """ add factorial from central """
        ex1, ex2, ex3 = self._noncentral
        # p. 43, 1.245 (below)
        self._factorial = (ex1, ex2 - ex1, ex3 - 3 * ex2 + 2 * ex1)


def xsden_to_meancv(xs, den):
    """
    Compute mean and cv from xs and density.

    Consider adding: np.nan_to_num(den)

    Note: cannot rely on pd.Series[-1] to work... it depends on the index.
    xs could be an index
    :param xs:
    :param den:
    :return:
    """
    pg = 1 - den.sum()
    xd = xs * den
    if isinstance(xs, np.ndarray):
        xsm = xs[-1]
    elif isinstance(xs, pd.Series):
        xsm = xs.iloc[-1]
    else:
        xsm = np.array(xs)[-1]
    ex1 = np.sum(xd) + pg * xsm
    # logger.log(WL, f'tail mass mean adjustment {pg * xsm}')
    ex2 = np.sum(xd * xs) + pg * xsm ** 2
    sd = np.sqrt(ex2 - ex1 ** 2)
    if ex1 != 0:
        cv = sd / ex1
    else:
        cv = np.nan
    return ex1, cv


def xsden_to_meancvskew(xs, den):
    """
    Compute mean, cv and skewness from xs and density

    Consider adding: np.nan_to_num(den)

    :param xs:
    :param den:
    :return:
    """
    pg = 1 - den.sum()
    xd = xs * den
    if isinstance(xs, np.ndarray):
        xsm = xs[-1]
        bs = xs[1] - xs[0]
    elif isinstance(xs, pd.Series):
        xsm = xs.iloc[-1]
        bs = xs.iloc[1] - xs.iloc[0]
    else:
        _ = np.array(xs)
        xsm = _[-1]
        bs = _[1] - _[0]
    xsm = xsm + bs
    ex1 = np.sum(xd) + pg * xsm
    # logger.log(WL, f'tail mass mean adjustment {pg * xsm}')
    xd *= xs
    ex2 = np.sum(xd) + pg * xsm ** 2
    ex3 = np.sum(xd * xs) + pg * xsm ** 3
    mw = MomentWrangler()
    mw.noncentral = ex1, ex2, ex3
    return mw.mcvsk
