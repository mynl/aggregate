"""Moment accumulation across mixture components and conversion between central,
non-central, and factorial moment representations."""

import itertools
import logging

import numpy as np
import pandas as pd

from .constants import VALIDATION_NOISE

logger = logging.getLogger(__name__)

__all__ = [
    'MomentAggregator', 'MomentWrangler',
    'xsden_to_mwrangler', 'ser_to_mwrangler',
    'xsden_to_meancv', 'xsden_to_meancvskew', 'xsden_to_noncentral',
]


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


def xsden_to_mwrangler(xs, den):
    """
    Build a :class:`MomentWrangler` from a discretized density.

    The single entry point for turning ``(xs, den)`` into moments: it
    populates a :class:`MomentWrangler` with the non-central moments, from
    which the caller reads ``.noncentral`` (raw moments), ``.mcvsk`` (mean,
    CV, skew), ``.central``, or ``.factorial`` as needed -- computing the
    sums only once. The convenience wrappers :func:`xsden_to_meancv`,
    :func:`xsden_to_meancvskew`, and :func:`xsden_to_noncentral` all route
    through here, so every caller treats the tail mass identically. Prefer
    this directly when more than one view of the moments is needed.

    Parameters
    ----------
    xs : numpy.ndarray, pandas.Series, or iterable
        Bucket left-endpoints of the discretized support, evenly spaced with
        bucket width ``bs = xs[1] - xs[0]``.
    den : array-like
        Probability mass on each bucket of ``xs``. For a proper distribution
        ``den.sum() == 1``.

    Returns
    -------
    MomentWrangler
        Populated with the non-central moments ``(E[X], E[X^2], E[X^3])``;
        the caller selects ``.mcvsk`` or ``.noncentral``.

    Notes
    -----
    Defective distributions (``den.sum() < 1``) are tolerated: the missing
    mass ``pg = 1 - den.sum()`` is placed at the *implied maximum loss*
    ``xsm + bs`` (the right edge of the final bucket), contributing
    ``pg * (xsm + bs)**k`` to the k-th non-central moment. For a proper
    distribution ``pg = 0`` and these terms vanish, so the result is
    unaffected. A genuine deficit (``pg > VALIDATION_NOISE``) is reported at
    INFO level; deficits at the floating-point noise floor are ignored.
    """
    # allow for defective distributions
    pg = 1 - den.sum()
    xd = xs * den
    # figure the max observed x for defective distribution adjustment
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
    # the implied max loss is the end of the last bucket, ie xsm + bs
    xsm = xsm + bs
    if pg > VALIDATION_NOISE:
        logger.info(
            'Defective distribution: probabilities sum to %.15g (deficit %.3g); '
            'missing mass placed at implied max loss %.6g.',
            1 - pg, pg, xsm)
    # pg * xsm adds defective > max adjustment
    ex1 = np.sum(xd) + pg * xsm
    xd *= xs
    ex2 = np.sum(xd) + pg * xsm ** 2
    ex3 = np.sum(xd * xs) + pg * xsm ** 3
    mw = MomentWrangler()
    mw.noncentral = ex1, ex2, ex3
    return mw


def xsden_to_meancv(xs, den):
    """
    Compute the mean and CV from a discretized density.

    Parameters
    ----------
    xs : array-like
        Bucket left-endpoints of the discretized support.
    den : array-like
        Probability mass on each bucket of ``xs``.

    Returns
    -------
    tuple
        ``(mean, cv)``. ``cv`` is ``nan`` when the mean is zero.

    Notes
    -----
    Delegates to :func:`xsden_to_mwrangler`; see its Notes for the defective
    distribution (``den.sum() < 1``) tail-mass convention.
    """
    mw = xsden_to_mwrangler(xs, den)
    m, cv, _ = mw.mcvsk
    return m, cv


def xsden_to_meancvskew(xs, den):
    """
    Compute the mean, CV, and skewness from a discretized density.

    Parameters
    ----------
    xs : array-like
        Bucket left-endpoints of the discretized support.
    den : array-like
        Probability mass on each bucket of ``xs``.

    Returns
    -------
    tuple
        ``(mean, cv, skew)``. ``cv`` is ``nan`` when the mean is zero and
        ``skew`` is ``nan`` when the standard deviation is zero.

    Notes
    -----
    Delegates to :func:`xsden_to_mwrangler`; see its Notes for the defective
    distribution (``den.sum() < 1``) tail-mass convention.
    """
    mw = xsden_to_mwrangler(xs, den)
    return mw.mcvsk


def xsden_to_noncentral(xs, den):
    """
    Compute the first three non-central moments from a discretized density.

    Parameters
    ----------
    xs : array-like
        Bucket left-endpoints of the discretized support.
    den : array-like
        Probability mass on each bucket of ``xs``.

    Returns
    -------
    tuple
        The non-central (raw) moments ``(E[X], E[X^2], E[X^3])``.

    Notes
    -----
    Delegates to :func:`xsden_to_mwrangler`; see its Notes for the defective
    distribution (``den.sum() < 1``) tail-mass convention.
    """
    mw = xsden_to_mwrangler(xs, den)
    return mw.noncentral


def ser_to_mwrangler(ser):
    """
    Build a :class:`MomentWrangler` from a Series indexed by its support.

    Convenience wrapper around :func:`xsden_to_mwrangler` for the common case
    where the x values are the Series index and the probability mass is the
    Series values (e.g. ``density_df.p_total``), so callers need not unpack
    ``ser.index`` / ``ser.values`` by hand.

    Parameters
    ----------
    ser : pandas.Series
        Probability mass indexed by the (evenly spaced) support points.

    Returns
    -------
    MomentWrangler
        See :func:`xsden_to_mwrangler` for what it carries and the defective
        distribution (``ser.sum() < 1``) tail-mass convention.
    """
    return xsden_to_mwrangler(ser.index.to_numpy(), ser.to_numpy())


def _noise_aware_rel_error(est, ref):
    """
    Relative error that degrades to absolute error when the reference is ~0.

    The usual relative error ``est / ref - 1`` is undefined (or explodes)
    when ``ref`` is genuinely zero. aggregate produces values that are
    exactly zero in theory but appear as floating-point dust (e.g. the
    skewness of a symmetric distribution); a naive relative error against
    that dust is meaningless. Where ``|ref|`` is at or below
    :data:`~aggregate.constants.VALIDATION_NOISE` the absolute error
    ``est - ref`` is returned instead.

    Parameters
    ----------
    est : float, numpy.ndarray, or pandas.Series
        Estimated (empirical) value(s).
    ref : float, numpy.ndarray, or pandas.Series
        Reference (theoretical) value(s).

    Returns
    -------
    Same shape/type as the inputs
        Relative error where ``|ref| > VALIDATION_NOISE``, absolute error
        otherwise. ``nan`` inputs propagate as ``nan``.
    """
    def _one(e, r):
        # per-element so heterogeneous object columns (e.g. the meta string
        # rows of stats_df) coerce to nan instead of raising.
        try:
            rf = float(r)
            ef = float(e)
        except (TypeError, ValueError):
            return np.nan
        if abs(rf) > VALIDATION_NOISE:
            return ef / rf - 1
        return ef - rf

    if isinstance(est, pd.Series) or isinstance(ref, pd.Series):
        idx = est.index if isinstance(est, pd.Series) else ref.index
        n = len(idx)
        est_v = est.to_numpy() if isinstance(est, pd.Series) else np.broadcast_to(est, (n,))
        ref_v = ref.to_numpy() if isinstance(ref, pd.Series) else np.broadcast_to(ref, (n,))
        return pd.Series([_one(e, r) for e, r in zip(est_v, ref_v)], index=idx)

    est_a = np.asarray(est, dtype=float)
    ref_a = np.asarray(ref, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = est_a / ref_a - 1
    out = np.where(np.abs(ref_a) > VALIDATION_NOISE, rel, est_a - ref_a)
    return out[()] if out.ndim == 0 else out


def _snap_noise(x):
    """
    Snap values indistinguishable from zero to exactly zero (display only).

    Used to keep floating-point dust (e.g. ``1.7e-14`` for the skewness of a
    symmetric distribution) out of rendered tables. Values with magnitude at
    or below :data:`~aggregate.constants.VALIDATION_NOISE` become ``0.0``;
    ``nan`` is preserved.

    Parameters
    ----------
    x : float, numpy.ndarray, or pandas.Series
        Value(s) to snap.

    Returns
    -------
    Same shape/type as the input
        ``x`` with near-zero entries replaced by ``0.0``.
    """
    arr = np.asarray(x, dtype=float)
    snapped = np.where(np.abs(arr) <= VALIDATION_NOISE, 0.0, arr)
    if isinstance(x, pd.Series):
        return pd.Series(snapped, index=x.index)
    return snapped[()] if snapped.ndim == 0 else snapped
