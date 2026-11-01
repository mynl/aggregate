"""Validation: the shared explanation formatter and the validation thresholds.

Extracted from ``distributions.py`` / ``_aggregate.py`` (Phase 1b, shared concerns). A leaf/near-leaf: it never imports ``_aggregate``/``_portfolio`` (takes plain data / a distribution object), which is what lets Portfolio reuse it in P4.
"""

import functools
import logging

import numpy as np
import pandas as pd

from . import constants
from .constants import Validation, ValidationWarning, validation_warning_suppressed, warn_once
from .config import get_settings
from .moments import MomentAggregator, _noise_aware_rel_error, skewness_is_testable

logger = logging.getLogger(__name__)


# VALIDATION_NOISE: absolute dust floor below which a quantity is treated as
# exact zero / numerical noise.
VALIDATION_NOISE = get_settings().validation.noise


# DEFICIT_MATERIALITY: economic-materiality floor on the pmf deficit
# ``1 - sum(p)``. Above it a realized law is missing enough mass to change an
# answer: the DEFECTIVE flag sets, the construction-time
# DefectiveDistributionWarning fires, and the exact-discrete Choquet helper
# refuses to price without an explicit parking policy. Deliberately three
# orders of magnitude looser than VALIDATION_NOISE, which measures arithmetic
# dust and has nothing to say about economics.
DEFICIT_MATERIALITY = get_settings().validation.deficit_materiality


# ALIASING_EPS: relative-error threshold on the convolution residual, which is
# what the ALIASING flag measures. See ``convolution_residual``.
ALIASING_EPS = get_settings().validation.aliasing_eps


def convolution_residual(en, sev_mean, agg_mean):
    """Relative error the convolution step alone introduced.

    Parameters
    ----------
    en : float
        Realized frequency mean, ``E[N]``, off the analytic frequency moments.
    sev_mean : float
        Empirical (discretized) severity mean, ``E[X]``.
    agg_mean : float
        Empirical (discretized) aggregate mean, ``E[A]``.

    Returns
    -------
    float
        ``|E[A] - E[N] * E[X]| / |E[N] * E[X]|``, or ``nan`` when the predictor
        is zero (a mean zero aggregate) or any input is not finite.

    Notes
    -----
    ``E[N] * E[X]`` is what the aggregate mean must be if the convolution is
    exact, and the severity term is the **discretized** severity's own mean, so
    severity discretization error cancels exactly rather than sitting in a
    denominator. What is left is the error the FFT step itself introduced,
    which is what the ``ALIASING`` flag claims to be about.

    This replaced a bare ratio of the two mean errors at ``1.0.0a311``. That
    ratio was floored on the numerator only, at the arithmetic dust floor
    ``VALIDATION_NOISE``, so any severity that discretizes essentially exactly
    (a discrete law, an integer lattice, a reinsurance layer landing on bucket
    edges, a ``sev agg.NAME`` reference) put near zero in the denominator and
    the ratio exploded on nothing. Measured over the 257 program corpus it had
    six firings and six false positives. See
    ``dev/done/plan-validation-punchup.md``.

    Absolute values throughout, so a signed aggregate works.
    """
    if not (np.isfinite(en) and np.isfinite(sev_mean) and np.isfinite(agg_mean)):
        return np.nan
    predicted = en * sev_mean
    if predicted == 0.0:
        return np.nan
    return abs(agg_mean - predicted) / abs(predicted)


def pmf_deficit(ob):
    """Probability mass the realized grid failed to hold, ``1 - sum(p)``.

    Works for an ``Aggregate`` (``agg_density``) or a ``Portfolio``
    (``density_df['p_total']``) by duck test, which keeps this module a leaf.
    Returns ``nan`` when the object has not been updated and there is no
    realized law to measure.

    Notes
    -----
    Mass that runs off the top of the grid is dropped rather than wrapped, so
    the realized law sums to less than one and forwards ``S = 1 - cumsum``
    and backwards ``S`` differ by exactly this amount. Above
    :data:`DEFICIT_MATERIALITY` that difference is large enough to move a
    price, which is what :attr:`~aggregate.constants.Validation.DEFECTIVE`
    and :class:`~aggregate.constants.DefectiveDistributionWarning` report.
    """
    if hasattr(ob, 'agg_list'):                  # Portfolio
        df = getattr(ob, 'density_df', None)
        if df is None or 'p_total' not in df:
            return np.nan
        return 1.0 - float(np.sum(df['p_total']))
    dens = getattr(ob, 'agg_density', None)
    if dens is None:
        return np.nan
    return 1.0 - float(np.sum(dens))


def _failure_parts(rv, deficit, subject):
    """The failures in ``rv``, as the phrases of the ``fails`` list.

    Parameters
    ----------
    rv : Validation
        The flag set.
    deficit : float or None
        The measured pmf deficit, quoted for ``DEFECTIVE``.
    subject : str
        ``'law'`` selects the aliasing wording of a law computed from its
        characteristic function.

    Returns
    -------
    list of str
    """
    # Collect failures from the SEV/AGG/ALIASING flags (suppressing higher
    # moments once a lower-order moment already failed).
    parts = []
    # DEFECTIVE leads: mass missing from the realized law makes every moment
    # comparison below it uninformative, the same argument that puts mean
    # before CV.
    if rv & Validation.DEFECTIVE:
        parts.append('pmf deficit' if deficit is None
                     else f'pmf deficit {deficit:.3e}')
    if rv & Validation.SEV_MEAN:
        parts.append('sev mean')
    if rv & Validation.AGG_MEAN:
        parts.append('agg mean')
    if rv & Validation.ALIASING:
        parts.append('mass outside the window wrapped in; raise log2' if subject == 'law'
                     else 'agg mean lost in the convolution, possible FFT wrap; '
                          'raise log2')
    if rv & Validation.TRUNCATION:
        parts.append('characteristic function truncated at pi / bs, lower bs')
    if not (rv & Validation.SEV_MEAN) and (rv & Validation.SEV_CV):
        parts.append('sev cv')
    if not (rv & Validation.AGG_MEAN) and (rv & Validation.AGG_CV):
        parts.append('agg cv')
    if not (rv & Validation.SEV_CV) and (rv & Validation.SEV_SKEW):
        parts.append('sev skew')
    if not (rv & Validation.AGG_CV) and (rv & Validation.AGG_SKEW):
        parts.append('agg skew')
    return parts


def explain_validation(rv, deficit=None, subject='severity', view=None, undefined=None):
    """
    Explain the validation result rv.
    Don't over report: if you fail CV don't need to be told you fail Skew too.

    Under reinsurance the subject (gross) object is checked against its analytic
    moments, and the reinsured view the object reports against the exact image of
    the subject on its grid. The message gives each after the ``reinsurance``
    marker, so a reader can tell which of the two failed.

    Parameters
    ----------
    rv : Validation
        The flag set to explain.
    deficit : float, optional
        The measured pmf deficit, quoted when ``rv`` carries
        :attr:`~aggregate.constants.Validation.DEFECTIVE`. Optional because
        the flag alone is enough to name the failure; the magnitude is what
        tells the reader whether it matters.
    subject : str, default 'severity'
        What an ``INFEASIBLE`` grid cannot reproduce, in the prefix ``grid
        infeasible for this <subject>``: ``'law'`` for a law computed from its
        characteristic function (``[CHF-Validation-Verdict]``).
    view : tuple, optional
        ``(word, subject_flags, view_flags)``, the split of the moment flags
        that the flag set cannot hold, since a subject and a view can fail the
        same moment. ``word`` is ``net``, ``ceded`` or ``output``. Without it the
        reinsured message cannot attribute the flags: the marker alone gives the
        subject's status when ``VIEW`` is clear, and ``subject or view`` when set.
    undefined : sequence of str, optional
        The analytic moments behind ``ANALYTIC_UNDEFINED``, ``('cv', 'skew')``,
        for the clause ``analytic cv, skew undefined``.
    """
    if rv == Validation.NOT_UNREASONABLE:
        return "not unreasonable"
    if rv & Validation.NOT_UPDATED:
        return "n/a, not updated"
    explanation = ', '.join(_failure_parts(rv, deficit, subject))
    if rv & Validation.REINSURANCE and view is not None:
        word, subject_flags, view_flags = view
        of_subject = ', '.join(_failure_parts(subject_flags, deficit, subject))
        of_view = ', '.join(_failure_parts(view_flags, deficit, subject))
        out = (f'reinsurance; subject {f"fails {of_subject}" if of_subject else "not unreasonable"}; '
               f'{word} {f"fails {of_view}" if of_view else "not unreasonable"}')
    elif rv & Validation.REINSURANCE:
        who = 'subject or view' if rv & Validation.VIEW else 'subject'
        out = (f'reinsurance; {who} fails {explanation}' if explanation
               else 'reinsurance; subject not unreasonable')
    else:
        out = f'fails {explanation}' if explanation else 'not unreasonable'
    # INFEASIBLE leads, and sits OUTSIDE the "fails" list on purpose. It is a
    # reading about the grid, not a failure: an infeasible object whose
    # moments hold still reads "not unreasonable" after the clause, and one
    # whose moments fail is told why they did. Same idiom as ``reinsurance``.
    # ANALYTIC_UNDEFINED sits outside it for the same reason: it is a fault of the
    # analytic side, and what the grid did is reported after it.
    if rv & Validation.ANALYTIC_UNDEFINED:
        out = f"analytic {', '.join(undefined) if undefined else 'moments'} undefined; {out}"
    if rv & Validation.INFEASIBLE:
        return f'grid infeasible for this {subject}; {out}'
    return out


def _skew_testable(theory, comp):
    """:func:`~aggregate.moments.skewness_is_testable` on the ``comp`` row of an analytic ``stats_df`` column."""
    return skewness_is_testable(*(float(theory[(comp, k)]) for k in ('ex1', 'ex2', 'ex3')))


def _undefined_analytic_moments(theory, realized, comp):
    """The measures of ``comp`` whose analytic value is ``nan`` through a defect of its computation.

    Parameters
    ----------
    theory, realized : pandas.Series
        The analytic and the grid's columns of ``stats_df``.
    comp : str
        ``'sev'`` or ``'agg'``.

    Returns
    -------
    list of str
        A subset of ``['cv', 'skew']``.

    Notes
    -----
    A ``nan`` is usually a fact about the law, and the moment test rightly skips
    it. A CV is undefined at a zero mean, and a skewness at a zero variance (a fixed
    count of a fixed severity), and both follow ``inf - inf`` from an infinite raw
    moment (a Pareto with no variance). What is left is a defect: a negative variance
    from raw moments that are mutually inconsistent, or a ``nan`` raw moment. There
    the library failed to compute a moment the law has, and the test would otherwise
    skip it as it skips one the law lacks, with only a ``logger.error`` to say so.
    """
    e1, e2, e3 = (float(theory[(comp, k)]) for k in ('ex1', 'ex2', 'ex3'))
    if np.isinf([e1, e2, e3]).any():
        return []
    var = e2 - e1 * e1
    degenerate = np.isfinite(var) and abs(var) <= VALIDATION_NOISE * max(abs(e2), e1 * e1)
    out = []
    if (np.isnan(float(theory[(comp, 'cv')])) and np.isfinite(float(realized[(comp, 'cv')]))
            and e1 != 0):
        out.append('cv')
    if (np.isnan(float(theory[(comp, 'skew')])) and np.isfinite(float(realized[(comp, 'skew')]))
            and not degenerate):
        out.append('skew')
    return out


def _view_moment_flags(comp, exact, realized, eps, skew_testable=True):
    """The moment flags of one component of a reinsured view, realized against exact.

    Parameters
    ----------
    comp : str
        ``'sev'`` or ``'agg'``.
    exact : tuple of float
        The exact ``(mean, cv, skew)``.
    realized : tuple of float
        The realized ``(mean, cv, skew)``.
    eps : float
        The object's ``validation_eps``.
    skew_testable : bool, default True
        :func:`~aggregate.moments.skewness_is_testable` on the exact raw moments.

    Returns
    -------
    Validation

    Notes
    -----
    Mean at ``eps``, CV at ``10 eps``, skewness at ``100 eps``, with the floors of
    the subject's test but one. The subject skips a theoretically zero CV, because
    the grid's estimate of the spread of a deterministic severity is noise. A
    cession can also be deterministic, with its one amount off the grid: 10 xs 10
    on a claim of 100 at ``bs`` 100 cedes 10 every time and is rebucketed onto 0
    and 100, a CV of 3 against 0. That spread is the whole of the error, so a zero
    exact CV is tested absolutely, at ``10 eps``.
    """
    mean_flag, cv_flag, skew_flag = {
        'sev': (Validation.SEV_MEAN, Validation.SEV_CV, Validation.SEV_SKEW),
        'agg': (Validation.AGG_MEAN, Validation.AGG_CV, Validation.AGG_SKEW)}[comp]
    rv = Validation.NOT_UNREASONABLE
    m, cv, skew = exact
    est_m, est_cv, est_skew = realized
    if np.isfinite(m) and np.isfinite(est_m) and abs(_noise_aware_rel_error(est_m, m)) > eps:
        rv |= mean_flag
    if np.isfinite(cv) and np.isfinite(est_cv):
        if abs(cv) <= VALIDATION_NOISE:
            if abs(est_cv) > 10 * eps:
                rv |= cv_flag
        elif not np.isclose(est_cv, cv, rtol=10 * eps, atol=VALIDATION_NOISE):
            rv |= cv_flag
    if (np.isfinite(skew) and abs(skew) > VALIDATION_NOISE and np.isfinite(est_skew) and skew_testable
            and not np.isclose(est_skew, skew, rtol=100 * eps, atol=VALIDATION_NOISE)):
        rv |= skew_flag
    return rv


def _book_view_flags(port):
    """The moment flags of a reinsured book's total against the sum of its units' exact views.

    Parameters
    ----------
    port : Portfolio

    Returns
    -------
    Validation

    Notes
    -----
    The total is the independent sum of the units' realized views, so its first
    three cumulants are the sums of theirs. A reinsured unit contributes the exact
    moments of its final view, any other unit its analytic moments. Summing them
    leaves the convolution of the units as the only thing under test, and needs no
    FFT, where ``Portfolio.reins_density_df`` would run six per unit on every
    update. Only the aggregate row is tested: the total's severity row is computed
    from the units' severity moments, which their own verdicts have read.

    The CV and skewness come straight from the cumulants. Rebuilding raw moments
    from them to reuse ``static_moments_to_mcvsk`` would put the mean back in only
    to cancel it, which on a symmetric total leaves a skewness of rounding above
    the noise floor.
    """
    from ._reinsurance import reins_realized_bases
    k1 = k2 = k3 = scale = 0.0
    for a in port.agg_list:
        if a.reins_kinds != 'None':
            e1, e2, e3 = reins_realized_bases(a)[-1][1]
        else:
            mixed = a.stats_df['mixed']
            e1, e2, e3 = (float(mixed[('agg', k)]) for k in ('ex1', 'ex2', 'ex3'))
        k1 += e1
        k2 += e2 - e1 * e1
        k3 += e3 - 3 * e1 * e2 + 2 * e1 ** 3
        scale += abs(e2)
    if not np.isfinite([k1, k2, k3]).all():
        return Validation.NOT_UNREASONABLE
    sd = 0.0 if abs(k2) <= VALIDATION_NOISE * scale else np.sqrt(max(k2, 0.0))
    exact = (k1, sd / k1 if k1 else np.nan, k3 / sd ** 3 if sd else np.nan)
    emp = port.stats_df['empirical']
    return _view_moment_flags(
        'agg', exact, tuple(float(emp[('agg', m)]) for m in ('mean', 'cv', 'skew')),
        float(port.validation_eps),
        skewness_is_testable(k1, k2 + k1 * k1, k3 + 3 * k1 * k2 + k1 ** 3))


def _book_aliasing(port):
    """``ALIASING`` when the book's total mean is not the sum of its units' realized means.

    The same direct measurement as ``valid_aggregate``, with the portfolio's
    predictor. The portfolio step is the convolution of the units, so what the
    total's mean must be is the sum of the units' realized aggregate means, whether
    they report a gross or a reinsured view.

    Parameters
    ----------
    port : Portfolio

    Returns
    -------
    Validation
    """
    predicted = sum(float(a.stats_df['empirical'][('agg', 'mean')])
                    for a in port.agg_list)
    residual = convolution_residual(
        1.0, predicted, float(port.stats_df['empirical'][('agg', 'mean')]))
    if (np.isfinite(residual) and residual > ALIASING_EPS
            and abs(port._deficit) <= VALIDATION_NOISE):
        logger.info('FAIL: Portfolio convolution residual %.3e > %.3e with '
                    'no deficit', residual, ALIASING_EPS)
        return Validation.ALIASING
    return Validation.NOT_UNREASONABLE


# ====================================================================
# Validation bodies (moved from Aggregate/Portfolio, Phase C). Similar but
# not identical -- kept side by side, not merged (README). The classes'
# ``valid`` / ``validation_explanation`` properties delegate here.
# ====================================================================

def valid_aggregate(agg):
    """
    Check if the model appears valid. An answer of True means the model is "not unreasonable".
    It does not guarantee the model is valid. On the other hand,
    False means it is definitely suspect. (The interpretation is similar to the null hypothesis
    in a statistical test).
    Called and reported automatically by qd for Aggregate objects.

    Checks the relative errors (from the canonical ``stats_df``) for:

    * severity mean < eps
    * severity cv < 10 * eps
    * severity skew < 100 * eps (skewness is more difficult to estimate)
    * aggregate mean < eps.
    * the convolution residual < ``ALIASING_EPS`` (:func:`convolution_residual`,
      the ``ALIASING`` test), whenever the pmf deficit is dust.
    * aggregate cv < 10 * eps
    * aggregate skew < 100 * esp

    The default uses eps = 1e-4 relative error. This can be changed by
    setting the ``validation_eps`` variable.

    All reads come from ``stats_df`` -- the single source of truth -- not
    ``validation_df`` (display).

    The CV and skew tests are applied only when the theoretical value is
    finite and its magnitude exceeds ``VALIDATION_NOISE`` -- a
    theoretically-zero skew (symmetric severity) or CV (deterministic
    severity) is skipped, because the FFT's empirical estimate of a zero
    higher moment is grid-dependent noise with no meaningful relative
    error. When the test applies, ``np.isclose`` with relative tolerance
    ``10*eps`` (CV) / ``100*eps`` (skew, harder to estimate) measures
    agreement.

    The ALIASING test measures the convolution step directly rather than
    comparing two mean errors, and fires only when the pmf deficit is
    arithmetic dust. Wrap conserves mass while moving the mean; truncation
    drops mass, which DEFECTIVE and AGG_MEAN already own. See
    :func:`convolution_residual` and ``dev/done/plan-validation-punchup.md``.

    Under reinsurance the tests above read the subject (gross) moments, and the
    reported net or ceded view is compared with the exact image of the subject on
    its grid, :func:`_view_moment_flags`. A failure there sets the same moment flags
    together with ``VIEW``, and ``agg._view_reading`` keeps the two apart for the
    verdict.

    Run with logger level 20 (info) for more information on failures.

    A Type 1 error (rejecting a valid model) is more likely than Type 2 (failing to reject an invalide one).

    :return: True (interpreted as not unreasonable) if all tests are passed, else False.

    """
    if agg._valid is not None:
        return agg._valid

    rv = Validation.NOT_UNREASONABLE
    # Not yet updated → no empirical moments to validate against.
    if pd.isna(agg.stats_df['empirical'].get(('agg', 'mean'), np.nan)):
        agg._valid = Validation.NOT_UPDATED
        return Validation.NOT_UPDATED
    # Mean / aliasing reads come straight off ``stats_df['error']`` --
    # the canonical noise-aware relative error of ``gross_empirical``
    # vs ``mixed``. Under no reinsurance ``gross_empirical ==
    # empirical`` and this is the classical theoretical-vs-empirical
    # check; under reinsurance it is the subject-validation hook from
    # §1.3 of the plan, and the reported view is checked further down.
    err = agg.stats_df['error'].abs()
    eps = agg.validation_eps
    sev_err_mean = float(err.get(('sev', 'mean'), 0.0))
    agg_err_mean = float(err.get(('agg', 'mean'), 0.0))
    if sev_err_mean > eps:
        logger.info('FAIL: Sev mean error > eps')
        rv |= Validation.SEV_MEAN

    if agg_err_mean > eps:
        logger.info('FAIL: Agg mean error > eps')
        rv |= Validation.AGG_MEAN

    # Aliasing: FFT wrap-around, measured directly. ``convolution_residual``
    # compares the realized aggregate mean against ``E[N]`` times the
    # DISCRETIZED severity mean, so severity discretization cancels and the
    # residual is the convolution step's own error.
    #
    # Gated on a dust-level deficit, which is what separates the two ways a
    # grid loses the mean. Wrap CONSERVES mass (it relocates it), so a genuine
    # wrap carries a deficit of exactly zero while the mean moves; truncation
    # DROPS mass, so it always carries a measurable deficit, and DEFECTIVE and
    # AGG_MEAN own that case. Measured on the corpus the gate has to be the
    # dust floor rather than DEFICIT_MATERIALITY: five programs lose the mean
    # to truncation with deficits between 1.5e-5 and 9.6e-5, all under
    # materiality, and calling those wrap would be the old false positive
    # wearing new clothes (author ruling 2026-08-21).
    residual = convolution_residual(
        float(agg.stats_df['mixed'][('freq', 'mean')]),
        float(agg.stats_df['gross_empirical'][('sev', 'mean')]),
        float(agg.stats_df['gross_empirical'][('agg', 'mean')]))
    if not np.isfinite(getattr(agg, '_deficit', np.nan)):
        agg._deficit = pmf_deficit(agg)
    if (np.isfinite(residual) and residual > ALIASING_EPS
            and abs(agg._deficit) <= VALIDATION_NOISE):
        logger.info('FAIL: convolution residual %.3e > %.3e with no deficit',
                    residual, ALIASING_EPS)
        rv |= Validation.ALIASING

    # CV and skew: compare subject empirical vs theoretical directly
    # from the canonical stats_df. The test is applied only when the
    # *theoretical* value is meaningfully non-zero (``abs(theo) >
    # VALIDATION_NOISE``): a theoretically-zero skew (symmetric
    # severity) or CV (deterministic severity) cannot be validated
    # against the FFT's empirical estimate, whose noise floor is grid-
    # dependent and unbounded (it can be far larger than the analytic
    # dust). ``isfinite`` skips an undefined moment (e.g. infinite CV
    # with no second moment). When the test applies, ``np.isclose``
    # with rtol 10*eps / 100*eps (skewness is harder to estimate, hence
    # looser) measures relative agreement.
    mixed = agg.stats_df['mixed']
    emp = agg.stats_df['gross_empirical']
    for comp, flag in (('sev', Validation.SEV_CV), ('agg', Validation.AGG_CV)):
        theo = float(mixed[(comp, 'cv')])
        est = float(emp[(comp, 'cv')])
        if (np.isfinite(theo) and abs(theo) > VALIDATION_NOISE and np.isfinite(est)
                and not np.isclose(est, theo, rtol=10 * eps, atol=VALIDATION_NOISE)):
            logger.info('FAIL: %s CV error > eps', comp)
            rv |= flag
    for comp, flag in (('sev', Validation.SEV_SKEW), ('agg', Validation.AGG_SKEW)):
        theo = float(mixed[(comp, 'skew')])
        est = float(emp[(comp, 'skew')])
        if (np.isfinite(theo) and abs(theo) > VALIDATION_NOISE and np.isfinite(est)
                and _skew_testable(mixed, comp)
                and not np.isclose(est, theo, rtol=100 * eps, atol=VALIDATION_NOISE)):
            logger.info('FAIL: %s skew error > eps', comp)
            rv |= flag

    # A ``nan`` the gates above skip because the computation failed, rather than
    # because the law lacks the moment.
    undefined = {m for comp in ('sev', 'agg') for m in _undefined_analytic_moments(mixed, emp, comp)}
    agg._undefined_moments = sorted(undefined)
    if undefined:
        logger.info('READING: analytic %s undefined', agg._undefined_moments)
        rv |= Validation.ANALYTIC_UNDEFINED

    # Defective: the realized law does not sum to 1, so mass ran off the top
    # of the grid. Recorded at every size (by ``update_work``, or by the
    # aliasing gate above where this object reached here by some other route),
    # flagged only above DEFICIT_MATERIALITY, where the missing mass is enough
    # to separate the two pricing routes.
    if np.isfinite(agg._deficit) and agg._deficit > DEFICIT_MATERIALITY:
        logger.info('FAIL: pmf deficit %.3e > materiality', agg._deficit)
        rv |= Validation.DEFECTIVE

    # Infeasible: the severity cannot be reproduced on this grid at any bucket
    # size. A reading about the grid rather than a verdict on the outcome, so
    # it sets whether or not the moments happen to hold and is transparent to
    # ``Validation.passes``. It explains the moment flags underneath it rather
    # than adding to them, which is why it leads the explanation the way
    # DEFECTIVE does. Deferred import: ``_bucket_window`` imports this module
    # at module scope, and this is the only edge back.
    from ._bucket_window import grid_is_infeasible
    if grid_is_infeasible(getattr(agg, '_bs_feasibility', None)):
        logger.info('READING: severity needs log2 %.2f, grid has %d',
                    agg._bs_feasibility['log2_required'],
                    agg._bs_feasibility['log2'])
        rv |= Validation.INFEASIBLE
    if getattr(agg, '_lattice_feasibility', None) is not None:
        logger.info('READING: lattice needs log2 %d, %d given',
                    agg._lattice_feasibility['log2_required'],
                    agg._lattice_feasibility['log2'])
        rv |= Validation.INFEASIBLE

    # Reinsurance: the checks above ran against the SUBJECT moments, and the
    # marker says the public surface (``agg_density`` etc.) is the after-reins
    # view. That view is checked here against the exact image of the subject on
    # its grid, which isolates what rebucketing the cession did. Its failures set
    # the same moment flags as the subject's, so the split between the two is kept
    # beside the flag set for the verdict to report.
    agg._view_reading = None
    if agg.reins_kinds != 'None':
        from ._reinsurance import reins_realized_bases
        view_flags = Validation.NOT_UNREASONABLE
        for comp, raw, realized in reins_realized_bases(agg):
            view_flags |= _view_moment_flags(
                comp, MomentAggregator.static_moments_to_mcvsk(*raw), realized, eps,
                skewness_is_testable(*raw))
        agg._view_reading = (agg._reins_after_label().lower(), rv, view_flags)
        rv |= Validation.REINSURANCE | view_flags
        if view_flags:
            logger.info('FAIL: reinsured view %s', view_flags)
            rv |= Validation.VIEW

    if rv == Validation.NOT_UNREASONABLE:
        logger.info('Aggregate %s does not fail any validation: not unreasonable', agg.name)
    agg._valid = rv
    return rv


def valid_portfolio(port):
    """
    Check if the model appears valid. See documentation for Aggregate.valid.

    An answer of True does not guarantee the model is valid, but
    False means it is definitely suspect. (Similar to the null hypothesis in a statistical test).
    Called and reported automatically by qd for Aggregate objects.

    Checks the relative errors (from ``port.stats_df['error']``) for:

    * severity mean < eps
    * severity cv < 10 * eps
    * severity skew < 100 * eps (skewness is more difficult to estimate)
    * aggregate mean < eps and < 2 * severity mean relative error (larger values
      indicate possibility of aliasing and that ``bs`` is too small).
    * aggregate cv < 10 * eps
    * aggregate skew < 100 * esp

    eps = 1e-3 by default; change in ``validation_eps`` attribute.

    The CV and skew tests are applied only when the theoretical value is
    finite and its magnitude exceeds ``VALIDATION_NOISE`` -- a
    theoretically-zero skew (symmetric total) or CV is skipped, because
    the FFT's empirical estimate of a zero higher moment is grid-
    dependent noise with no meaningful relative error. When the test
    applies, ``np.isclose`` with relative tolerance ``10*eps`` (CV) /
    ``100*eps`` (skew) measures agreement.

    :return: True if all tests are passed, else False.

    """
    if port._valid is not None:
        return port._valid

    rv = Validation.NOT_UNREASONABLE
    if port.density_df is None:
        port._valid = Validation.NOT_UPDATED
        return Validation.NOT_UPDATED

    # Defective: the portfolio's OWN total. ``update`` records it; recompute
    # only if this object reached here by some other route. Held in a separate
    # flag rather than OR-ed into ``rv`` here, because ``rv`` drives the early
    # return below, which is about the UNITS failing; a defective total is
    # not a reason to skip the total's own moment checks. (A defective unit
    # folds its flag in through ``rv |= r``, exactly as its moment failures
    # do.)
    if not np.isfinite(getattr(port, '_deficit', np.nan)):
        port._deficit = pmf_deficit(port)
    defective = (Validation.DEFECTIVE
                 if np.isfinite(port._deficit)
                 and port._deficit > DEFICIT_MATERIALITY
                 else Validation.NOT_UNREASONABLE)
    if defective:
        logger.info('FAIL: Portfolio pmf deficit %.3e > materiality',
                    port._deficit)
    # The units' readings fold in through ``rv |= r`` below; this is the book's
    # own, on the shared grid.
    if getattr(port, '_lattice_feasibility', None) is not None:
        defective |= Validation.INFEASIBLE

    # The units' subject and view failures stay apart, as on a unit, so that the
    # book's verdict can report them apart.
    subject_flags = view_flags = Validation.NOT_UNREASONABLE
    undefined = set()
    for a in port.agg_list:
        r = a.valid
        if not r.passes:
            logger.info(f'Aggregate {a.name} fails validation')
        rv |= r
        undefined.update(getattr(a, '_undefined_moments', ()))
        reading = getattr(a, '_view_reading', None)
        if reading is None:
            subject_flags |= r
        else:
            subject_flags |= reading[1]
            view_flags |= reading[2]
    reinsured = bool(rv & Validation.REINSURANCE)
    port._undefined_moments = sorted(undefined)

    def book_reading():
        return (port._reins_after_label().lower(), subject_flags | defective,
                view_flags) if reinsured else None

    if rv & ~Validation.REINSURANCE:
        logger.info('Exiting: Portfolio validation steps skipped due to failed Aggregate validation')
        port._view_reading = book_reading()
        port._valid = rv | defective
        return port._valid
    else:
        logger.info('No Aggregate object fails validation')

    # A reinsured unit puts its view into the total, so the total has no analytic
    # moments of its own. It is checked against the sum of the units' exact views.
    if reinsured:
        view_flags = _book_aliasing(port) | _book_view_flags(port)
        port._view_reading = book_reading()
        rv |= defective | view_flags
        if view_flags:
            logger.info('FAIL: Portfolio reinsured view %s', view_flags)
            rv |= Validation.VIEW
        port._valid = rv
        return rv
    port._view_reading = None

    # apply validation to the Portfolio total. SSoT: relative errors
    # come straight off ``stats_df['error']`` (noise-aware diff of
    # ``empirical`` vs ``total``) -- no detour through ``summary_df``.
    err = port.stats_df['error'].abs()
    eps = port.validation_eps
    sev_err_mean = float(err.get(('sev', 'mean'), 0.0))
    agg_err_mean = float(err.get(('agg', 'mean'), 0.0))
    if sev_err_mean > eps:
        logger.info('FAIL: Portfolio Sev mean error > eps')
        rv |= Validation.SEV_MEAN

    if agg_err_mean > eps:
        logger.info('FAIL: Portfolio Agg mean error > eps')
        rv |= Validation.AGG_MEAN

    rv |= _book_aliasing(port)

    # CV and skew: tested only when the theoretical value is meaningfully
    # non-zero (abs(theo) > VALIDATION_NOISE); a theoretically-zero skew
    # (symmetric total) or CV cannot be validated against the FFT's
    # empirical estimate, whose noise floor is grid-dependent. See
    # Aggregate.valid. Read theoretical (``total``) and empirical from the
    # canonical stats_df; isfinite skips undefined moments.
    total = port.stats_df['total']
    emp = port.stats_df['empirical']
    for comp, flag in (('sev', Validation.SEV_CV), ('agg', Validation.AGG_CV)):
        theo = float(total[(comp, 'cv')])
        est = float(emp[(comp, 'cv')])
        if (np.isfinite(theo) and abs(theo) > VALIDATION_NOISE and np.isfinite(est)
                and not np.isclose(est, theo, rtol=10 * eps, atol=VALIDATION_NOISE)):
            logger.info('FAIL: Portfolio %s CV error > eps', comp)
            rv |= flag
    for comp, flag in (('sev', Validation.SEV_SKEW), ('agg', Validation.AGG_SKEW)):
        theo = float(total[(comp, 'skew')])
        est = float(emp[(comp, 'skew')])
        if (np.isfinite(theo) and abs(theo) > VALIDATION_NOISE and np.isfinite(est)
                and _skew_testable(total, comp)
                and not np.isclose(est, theo, rtol=100 * eps, atol=VALIDATION_NOISE)):
            logger.info('FAIL: Portfolio %s skew error > eps', comp)
            rv |= flag

    undefined = {m for comp in ('sev', 'agg') for m in _undefined_analytic_moments(total, emp, comp)}
    port._undefined_moments = sorted(undefined)
    if undefined:
        rv |= Validation.ANALYTIC_UNDEFINED

    rv |= defective
    if rv == Validation.NOT_UNREASONABLE:
        logger.info('Portfolio does not fail any validation: not unreasonable')
    port._valid = rv
    return rv


def valid_characteristic_function(obj):
    r"""The :class:`~aggregate.constants.Validation` flags of a law computed from its characteristic function.

    The ``AggregateCHF`` counterpart of :func:`valid_aggregate`, kept beside it
    rather than merged (the module's convention). Cached on ``obj._valid``,
    which ``update`` clears.

    Parameters
    ----------
    obj : AggregateCHF

    Returns
    -------
    Validation

    Notes
    -----
    The moment flags are ``valid_aggregate``'s, read off the same store
    (``stats_df``): ``AGG_MEAN`` when the noise-aware relative error of the
    mean exceeds ``eps``, ``AGG_CV`` and ``AGG_SKEW`` at ``10 eps`` and
    ``100 eps``, tested only where the theoretical value is finite and above
    :data:`VALIDATION_NOISE`, so a window statistic of a moment the law lacks
    never fails. On a tilted grid of a stable law with a mean, ``error`` holds
    the error of the completed mean, the window mean plus the closed-form tail
    beyond the window (``AggregateCHF._completed_mean``), so the tail the window
    leaves out by construction does not read as a failure. An untilted grid has
    no such completion, and its mean fails by the mass that wrapped. The grid
    flags carry the inversion's checks:

    * ``DEFECTIVE``: the deficit :math:`1 - \sum p` above
      :data:`DEFICIT_MATERIALITY`, the mass a tilted window leaves out above;
    * ``ALIASING``: an untilted grid whose outside mass exceeds
      ``1 - bucket_sizing_p``, or whose heavy tail's edge error exceeds
      ``EDGE_ERROR_FLAG``, mass that wrapped in;
    * ``TRUNCATION``: ``truncation_error`` above ``TRUNCATION_FLAG``;
    * ``INFEASIBLE``: no grid up to ``RECOMMENDED_LOG2_MAX`` meets coverage,
      truncation and resolution together, checked only when one of the
      three fails here.

    The most negative bucket and the buckets per scale stay advisory rows of
    ``info``: ``Aggregate``'s verdict has no negative mass or resolution test,
    and a coarse grid shows in the moment errors.
    """
    if getattr(obj, '_valid', None) is not None:
        return obj._valid
    if obj.agg_density is None:
        return Validation.NOT_UPDATED
    from .characteristic_function import EDGE_ERROR_FLAG, TRUNCATION_FLAG
    rv = Validation.NOT_UNREASONABLE
    eps = obj.validation_eps
    st = obj.stats_df
    err = st['error'].abs()
    if float(err.get(('agg', 'mean'), 0.0)) > eps:
        rv |= Validation.AGG_MEAN
    mixed, emp = st['mixed'], st['gross_empirical']
    for measure, flag, mult in (('cv', Validation.AGG_CV, 10), ('skew', Validation.AGG_SKEW, 100)):
        theo, est = float(mixed[('agg', measure)]), float(emp[('agg', measure)])
        if (np.isfinite(theo) and abs(theo) > VALIDATION_NOISE and np.isfinite(est)
                and (measure != 'skew' or _skew_testable(mixed, 'agg'))
                and not np.isclose(est, theo, rtol=mult * eps, atol=VALIDATION_NOISE)):
            rv |= flag
    obj._deficit = pmf_deficit(obj)
    if np.isfinite(obj._deficit) and obj._deficit > DEFICIT_MATERIALITY:
        rv |= Validation.DEFECTIVE
    budget = 1.0 - obj.bucket_sizing_p
    untilted = max(obj.tilt) == 0
    coverage_fails = obj._x_min_fixed is None and obj.outside_mass > budget
    if untilted and (coverage_fails or (obj.family.power_tails
                                        and np.isfinite(obj.edge_error)
                                        and obj.edge_error > EDGE_ERROR_FLAG)):
        rv |= Validation.ALIASING
    truncated = obj.truncation_error > TRUNCATION_FLAG
    if truncated:
        rv |= Validation.TRUNCATION
    if (coverage_fails or truncated or obj._coarse()) and not obj._recommended_grid_met():
        rv |= Validation.INFEASIBLE
    obj._valid = rv
    return rv


#: The six scored terms of :func:`validation_score`, as
#: ``((component, measure), tolerance_multiple)``. The multiples are exactly the
#: ones :func:`valid_aggregate` / :func:`valid_portfolio` apply: mean at ``eps``,
#: cv at ``10 eps``, skew at ``100 eps``, skewness being the harder moment to
#: estimate. Dividing each relative error by its own tolerance puts every term on
#: one scale, in units of "fraction of the tolerance used up", so a combined
#: score of 1 is exactly the pass boundary and the score does not move when
#: ``validation_eps`` is changed.
SCORE_TERMS = (
    (('sev', 'mean'), 1.0),
    (('sev', 'cv'), 10.0),
    (('sev', 'skew'), 100.0),
    (('agg', 'mean'), 1.0),
    (('agg', 'cv'), 10.0),
    (('agg', 'skew'), 100.0),
)


def _theory_column(obj):
    """The theoretical-moment column of ``obj.stats_df``.

    ``'mixed'`` for an ``Aggregate`` (the mixed frequency/severity analytic
    moments), ``'total'`` for a ``Portfolio``. Detected by presence rather than
    by class, so this module stays a leaf.
    """
    sdf = obj.stats_df
    return sdf['mixed'] if 'mixed' in sdf.columns else sdf['total']


def validation_score_terms(obj):
    """The six normalized validation errors, keyed ``'u_sev_mean'``-style.

    The per-term detail behind :func:`validation_score`, and **independent of
    the combining power**, so a caller comparing several powers computes these
    once.

    Parameters
    ----------
    obj : Aggregate or Portfolio
        An updated object. Reads ``stats_df`` and ``validation_eps`` only.

    Returns
    -------
    dict
        ``{'u_sev_mean': float, ...}``; ``nan`` for a term that does not apply,
        ``inf`` for one whose empirical value is not finite.

    Notes
    -----
    Each term is read from ``stats_df['error']``, the canonical noise-aware
    relative error written at the end of every update, which degrades to an
    absolute error when the reference is at or below the noise floor, and is
    divided by its own tolerance from :data:`SCORE_TERMS`::

        u_i = |error_i| / tol_i

    A term is live only when its *theoretical* value is finite and above the
    noise floor, the same test :func:`valid_aggregate` applies: a theoretically
    zero skewness (symmetric severity) or CV (deterministic severity) cannot be
    validated against the FFT's grid-dependent estimate of it. The theoretical
    moments do not depend on the grid, so the live set is stable as ``bs`` and
    ``log2`` vary, which is what makes scores comparable across a
    :meth:`Aggregate.sharpen` probe. A non-finite *empirical* value is a genuine
    failure and scores infinite rather than dropping out.

    Under reinsurance ``error`` compares the SUBJECT (gross) moments, so the score
    measures the subject alone. The after reinsurance view has no analytic moments
    in ``stats_df``; :func:`valid_aggregate` checks it against the exact image of
    the subject.
    """
    err = obj.stats_df['error'].abs()
    theo = _theory_column(obj)
    eps = float(obj.validation_eps)
    terms = {}
    for key, mult in SCORE_TERMS:
        name = f'u_{key[0]}_{key[1]}'
        t = float(theo.get(key, np.nan))
        if not (np.isfinite(t) and abs(t) > VALIDATION_NOISE
                and (key[1] != 'skew' or _skew_testable(theo, key[0]))):
            terms[name] = np.nan
            continue
        e = float(err.get(key, np.nan))
        terms[name] = np.inf if not np.isfinite(e) else e / (mult * eps)
    return terms


def combine_score_terms(terms, power=2):
    """Combine normalized terms into one score by a power mean.

    ``(mean_i u_i ** power) ** (1 / power)``, or ``max_i u_i`` for infinite
    power. Averaging rather than summing keeps objects with different numbers of
    live terms comparable; the ``1/power`` root puts every ``power`` on the same
    scale. ``nan`` when no term applies, ``inf`` when any term is infinite.
    """
    live = [u for u in terms.values() if not (u is None or np.isnan(u))]
    if not live:
        return np.nan
    arr = np.array(live, dtype=float)
    if np.isinf(arr).any():
        return np.inf
    if np.isinf(power):
        return float(arr.max())
    return float(np.mean(arr ** power) ** (1.0 / power))


def validation_score(obj, power=2):
    """How well the realized grid reproduces the analytic moments. Small is good.

    **The score is in units of the validation tolerance**, so ``score <= 1``
    means the object passes validation at its own ``validation_eps`` and
    ``score = 1`` sits exactly on the pass boundary. That makes it the
    continuous refinement of the pass/fail :attr:`Aggregate.valid` verdict: a
    number to watch and to compare across grids, where the flag only says
    whether a line was crossed.

    Parameters
    ----------
    obj : Aggregate or Portfolio
        An updated object.
    power : float, default 2
        Exponent of the combining power mean. ``1`` averages the terms, ``2`` is
        the Euclidean default, ``numpy.inf`` reports the worst single term.

    Returns
    -------
    float
        The combined score over severity and aggregate mean, CV and skewness.
        See :func:`validation_score_terms` for the per-term detail.
    """
    return combine_score_terms(validation_score_terms(obj), power)


def _infeasible_subject(obj):
    """What an infeasible grid cannot reproduce, for the verdict's prefix.

    ``'severity'`` for a thick law, and for a lattice book the lattice with the
    ``log2`` and bucket that would hold it, which is the number the reader needs.
    """
    f = getattr(obj, '_lattice_feasibility', None)
    if f is None:
        return 'severity'
    from ._bucket_window import _fmt_bs
    return f'lattice (needs log2 {f["log2_required"]} at bs {_fmt_bs(f["h"])})'


def validation_description(obj):
    """Short one-line validation verdict (str).

    Shared by ``Aggregate`` and ``Portfolio`` (both delegate here). This is the
    terse phrase the ``info`` row and the one-line text intro carry:
    ``'not unreasonable'``, ``'fails agg cv'``, ``'n/a, not updated'``.
    Validation is computed if needed via ``obj.valid``.

    Notes
    -----
    a172 [FCC-Contract-Gaps]. This function is what ``validation_explanation``
    used to be. The narrative surface pairs a short ``*_description`` with a long
    ``*_explanation``, and validation carried only the name ``explanation`` on a
    string that was actually the short form. The terse text is unchanged: it
    moved to the name that describes it, and ``validation_explanation`` became
    the long form it always claimed to be.
    """
    # ``obj.valid`` first: it is what records ``_deficit``.
    rv = obj.valid
    return explain_validation(rv, getattr(obj, '_deficit', None),
                              subject=_infeasible_subject(obj),
                              view=getattr(obj, '_view_reading', None),
                              undefined=getattr(obj, '_undefined_moments', None))


def warn_if_invalid(obj):
    """Report a build whose verdict fails, with one :class:`~aggregate.constants.ValidationWarning`.

    Parameters
    ----------
    obj : Aggregate, Portfolio or AggregateCHF
        An updated object.

    Notes
    -----
    Computing the verdict reads ``stats_df`` and a few grid sums, with no FFT, and
    takes about a millisecond against 30 for the update at ``log2`` 16. It is not left
    cached: ``validation_eps`` is a plain attribute a reader may set after a build and
    before first reading ``valid``, which has always applied, so the cache stays empty
    as a lazy verdict would leave it.
    """
    if constants._VALIDATION_WARNING_SUPPRESSED:
        return
    rv = obj.valid
    if not rv.passes:
        warn_once(f'{obj.name}: {obj.validation_description}. See validation_explanation.',
                  ValidationWarning, key=('validation', obj.name, str(rv & ~Validation.INFEASIBLE)),
                  stacklevel=3)
    obj._valid = None
    for a in getattr(obj, 'agg_list', ()):
        a._valid = None


def verdict_reported_after(fn):
    """Decorate a function that moves an object's grid, so that only the final grid is reported.

    ``sharpen`` updates the object many times on the way to one grid, and the verdict
    of a cell it rejects is not the reader's concern.

    Parameters
    ----------
    fn : callable
        Takes the object first.

    Returns
    -------
    callable
    """
    @functools.wraps(fn)
    def wrapper(obj, *args, **kwargs):
        with validation_warning_suppressed():
            out = fn(obj, *args, **kwargs)
        warn_if_invalid(obj)
        return out
    return wrapper


def validation_opening(short, eps, components='severity and for the aggregate'):
    """The first two sentences of every validation explanation, as a list.

    Shared by :func:`validation_explanation` (``Aggregate`` and ``Portfolio``)
    and ``AggregateCHF.validation_explanation``, which names its one
    component.

    Parameters
    ----------
    short : str
        The one-line verdict, :func:`explain_validation`'s.
    eps : float or None
        The object's ``validation_eps``.
    components : str
        What the moments are compared for, completing ``compared for ...``.

    Returns
    -------
    list of str
    """
    tol = f'{eps:.3g}' if isinstance(eps, (int, float)) else 'the configured eps'
    return [f'Validation of the realized grid against the analytic moments: '
            f'{short}.',
            f'Each of mean, CV and skewness is compared for {components}, and a '
            f'relative error above {tol} fails; only the lowest-order failure is '
            f'reported, since a mean that is wrong makes the higher moments '
            f'uninformative.']


def validation_explanation(obj):
    """Long-narrative explanation of the validation result (str).

    Shared by ``Aggregate`` and ``Portfolio`` (both delegate here): the
    consistent narrative surface mirroring ``tail_explanation`` /
    ``bs_explanation``, and the verbose form of :func:`validation_description`.

    Spells out what was checked and against what, so a reader who has never met
    the word "unreasonable" in this library can act on the answer: the analytic
    first three moments of severity and aggregate against the ones realized on
    the FFT grid, at the object's own ``validation_eps``, plus the aliasing test
    and the reinsurance caveat where they apply.
    """
    rv = obj.valid
    short = explain_validation(rv, getattr(obj, '_deficit', None),
                               subject=_infeasible_subject(obj),
                               view=getattr(obj, '_view_reading', None),
                               undefined=getattr(obj, '_undefined_moments', None))
    if rv & Validation.NOT_UPDATED:
        return ('Not validated: the object has not been updated, so there is no '
                'realized grid to compare the analytic moments against. Call '
                'update() (or build with update=True).')
    out = validation_opening(short, getattr(obj, 'validation_eps', None))
    if getattr(obj, '_lattice_feasibility', None) is not None:
        from ._bucket_window import lattice_explain
        out.append(lattice_explain(obj._lattice_feasibility))
    elif rv & Validation.INFEASIBLE:
        f = getattr(obj, '_bs_feasibility', None) or {}
        lost = f.get('lost_at_zero', np.nan)
        lost_txt = (f' {100 * lost:.4g}% of its mean is supplied below the '
                    f'first half bucket, where the round scheme places it at '
                    f'exactly 0.' if np.isfinite(lost) else '')
        out.append(
            f'The severity cannot be reproduced on this grid at any bucket '
            f'size, which is what every moment failure above is measuring. '
            f'The mean of a thick law is furnished far above its median, so '
            f'the resolution its body needs and the reach its tail needs pull '
            f'apart faster than a fixed bucket count can span: this one wants '
            f'bs at or under {f.get("bs_max", float("nan")):.6g} and a grid '
            f'reaching {f.get("reach", float("nan")):.6g}, which together need '
            f'log2 {f.get("log2_required", float("nan")):.0f} against the '
            f'{f.get("log2", "")} used.{lost_txt} A finer bs makes it worse, '
            f'because it shortens the reach. The fix is an occurrence limit on '
            f'the severity, which truncates the size biased tail and brings the '
            f'two demands back together, or accepting the reported moment '
            f'errors as the price of an unlimited model.')
    if rv & Validation.DEFECTIVE:
        d = getattr(obj, '_deficit', None)
        amount = f'{d:.3e}' if isinstance(d, float) and np.isfinite(d) \
            else 'some'
        out.append(f'The realized law does not sum to 1: {amount} of the '
                   f'probability ran off the top of the grid and was dropped '
                   f'rather than wrapped. Forwards S = 1 - cumsum carries the '
                   f'missing mass as a tail blob and backwards S drops it, so '
                   f'the two pricing routes disagree by exactly this amount '
                   f'and neither is wrong. Raise log2, or widen the grid.')
    if rv & Validation.ALIASING:
        out.append('The aggregate mean is not E[N] times the discretized '
                   'severity mean, and no mass is missing, which is the '
                   'signature of FFT wrap-around: mass is coming off the top '
                   'of the grid and landing back at the bottom rather than '
                   'being dropped. Try a larger bs, or a larger log2 at the '
                   'same bs.')
    if rv & Validation.REINSURANCE:
        out.append('The reported view is net or ceded. The subject (gross) object '
                   'underneath is checked against its analytic moments, and the '
                   'cession against the exact image of the subject on its grid.')
    if rv & Validation.VIEW:
        out.append('The reported view departs from that image, which is the error '
                   'of rebucketing the cession onto the grid: the reported '
                   'distribution is not the distribution of the cession, whatever '
                   'its mean. A bucket that the net and ceded amounts land on '
                   'removes the error.')
    if rv & Validation.ANALYTIC_UNDEFINED:
        out.append('An analytic moment could not be computed: the raw moments are '
                   'mutually inconsistent, with a negative variance or a nan, usually '
                   'from numerical integration. That moment was not compared with the '
                   'grid, so nothing vouches for it, whatever the verdict on the rest.')
    if rv == Validation.NOT_UNREASONABLE:
        out.append('Nothing failed, so the object is not unreasonable. That is '
                   'a statement about the numerics reproducing the model, not '
                   'about the model being right for the risk.')
    return ' '.join(out)
