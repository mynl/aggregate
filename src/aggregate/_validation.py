"""Validation: the shared explanation formatter and the validation thresholds.

Extracted from ``distributions.py`` / ``_aggregate.py`` (Phase 1b, shared concerns). A leaf/near-leaf: it never imports ``_aggregate``/``_portfolio`` (takes plain data / a distribution object), which is what lets Portfolio reuse it in P4.
"""

import logging

import numpy as np
import pandas as pd

from .constants import Validation
from .config import get_settings

logger = logging.getLogger(__name__)


# VALIDATION_NOISE: absolute dust floor below which a quantity is treated as
# exact zero / numerical noise.
VALIDATION_NOISE = get_settings().validation.noise


# ALIASING_RATIO: the ALIASING flag fires when the agg-mean relative error
# exceeds this multiple of the sev-mean relative error (FFT wrap-around).
ALIASING_RATIO = get_settings().validation.aliasing_ratio


def explain_validation(rv):
    """
    Explain the validation result rv.
    Don't over report: if you fail CV don't need to be told you fail Skew too.

    Under reinsurance the realised view has no independent theoretical and
    cannot be validated, but the SUBJECT (gross) view was validated under
    the hood (§1.3 of the aggregate refactor plan). The message reports
    that subject status alongside the ``reinsurance`` marker, so the user
    can tell whether the underlying gross object is sound.
    """
    if rv == Validation.NOT_UNREASONABLE:
        return "not unreasonable"
    if rv & Validation.NOT_UPDATED:
        return "n/a, not updated"
    # Collect failures from the SEV/AGG/ALIASING flags (suppressing higher
    # moments once a lower-order moment already failed).
    parts = []
    if rv & Validation.SEV_MEAN:
        parts.append('sev mean')
    if rv & Validation.AGG_MEAN:
        parts.append('agg mean')
    if rv & Validation.ALIASING:
        parts.append('agg mean error >> sev, possible aliasing; try larger bs')
    if not (rv & Validation.SEV_MEAN) and (rv & Validation.SEV_CV):
        parts.append('sev cv')
    if not (rv & Validation.AGG_MEAN) and (rv & Validation.AGG_CV):
        parts.append('agg cv')
    if not (rv & Validation.SEV_CV) and (rv & Validation.SEV_SKEW):
        parts.append('sev skew')
    if not (rv & Validation.AGG_CV) and (rv & Validation.AGG_SKEW):
        parts.append('agg skew')
    explanation = ', '.join(parts)
    if rv & Validation.REINSURANCE:
        if explanation:
            return f'reinsurance; subject fails {explanation}'
        return 'reinsurance; subject not unreasonable'
    return f'fails {explanation}'


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
    * aggregate mean < eps and < ``ALIASING_RATIO`` * severity mean
      relative error (larger values indicate possible aliasing — i.e.
      that ``bs`` is too small).
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

    The ALIASING test silences itself when the agg-mean relative error
    is itself below ``VALIDATION_NOISE`` (genuine numerical dust, not
    aliasing) -- this replaces the old ``eps ** 3`` floor that was fitted
    to the default ``eps`` value.

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
    # §1.3 of the plan, the only apples-to-apples check available.
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

    # Aliasing fingerprint: the agg-mean error sits well above the sev-
    # mean error (the FFT amplifies sev-discretisation error during
    # convolution when ``bs`` is too small). Silenced under the
    # ``VALIDATION_NOISE`` floor where the agg error is genuine dust.
    if (agg_err_mean > VALIDATION_NOISE
            and sev_err_mean > 0
            and agg_err_mean > ALIASING_RATIO * sev_err_mean):
        logger.info('FAIL: Agg mean error > %d * sev error', ALIASING_RATIO)
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
                and not np.isclose(est, theo, rtol=100 * eps, atol=VALIDATION_NOISE)):
            logger.info('FAIL: %s skew error > eps', comp)
            rv |= flag

    # Reinsurance: the realised (after-reins) object has no independent
    # theoretical, so its sev/agg moments cannot be validated. The
    # checks above ran against the SUBJECT moments and remain
    # meaningful; mark the result with REINSURANCE so callers know the
    # public surface (``agg_density`` etc.) is the after-reins view.
    if agg.reins_kinds != 'None':
        rv |= Validation.REINSURANCE

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

    for a in port.agg_list:
        r = a.valid
        if r & Validation.REINSURANCE:
            logger.info(f'Aggregate {a.name} has reinsurance, validation n/a')
        elif not r:
            logger.info(f'Aggregate {a.name} fails validation')
        rv |= r

    if rv != Validation.NOT_UNREASONABLE:
        logger.info('Exiting: Portfolio validation steps skipped due to failed or n/a Aggregate validation')
        port._valid = rv
        return rv
    else:
        logger.info('No Aggregate object fails validation')

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

    # Aliasing fingerprint: agg error >> sev error. Silenced under
    # ``VALIDATION_NOISE`` where both are dust.
    if (agg_err_mean > VALIDATION_NOISE
            and sev_err_mean > 0
            and agg_err_mean > ALIASING_RATIO * sev_err_mean):
        logger.info('FAIL: Portfolio Agg mean error > %d * sev error', ALIASING_RATIO)
        rv |= Validation.ALIASING

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
                and not np.isclose(est, theo, rtol=100 * eps, atol=VALIDATION_NOISE)):
            logger.info('FAIL: Portfolio %s skew error > eps', comp)
            rv |= flag

    if rv == Validation.NOT_UNREASONABLE:
        logger.info('Portfolio does not fail any validation: not unreasonable')
    port._valid = rv
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

    Under reinsurance ``error`` compares the SUBJECT (gross) moments: the after
    reinsurance object has no independent theoretical, so the gross comparison is
    the only apples-to-apples check available.
    """
    err = obj.stats_df['error'].abs()
    theo = _theory_column(obj)
    eps = float(obj.validation_eps)
    terms = {}
    for key, mult in SCORE_TERMS:
        name = f'u_{key[0]}_{key[1]}'
        t = float(theo.get(key, np.nan))
        if not (np.isfinite(t) and abs(t) > VALIDATION_NOISE):
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
    return explain_validation(obj.valid)


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
    short = explain_validation(rv)
    if rv & Validation.NOT_UPDATED:
        return ('Not validated: the object has not been updated, so there is no '
                'realized grid to compare the analytic moments against. Call '
                'update() (or build with update=True).')
    eps = getattr(obj, 'validation_eps', None)
    tol = f'{eps:.3g}' if isinstance(eps, (int, float)) else 'the configured eps'
    out = [f'Validation of the realized grid against the analytic moments: '
           f'{short}.',
           f'Each of mean, CV and skewness is compared for severity and for the '
           f'aggregate, and a relative error above {tol} fails; only the '
           f'lowest-order failure is reported, since a mean that is wrong makes '
           f'the higher moments uninformative.']
    if rv & Validation.ALIASING:
        out.append('The aggregate mean error is far larger than the severity '
                   'error, which is the signature of FFT wrap-around: mass is '
                   'coming off the top of the grid and landing back at the '
                   'bottom. Try a larger bs, or a larger log2 at the same bs.')
    if rv & Validation.REINSURANCE:
        out.append('The reported view is net or ceded, and a cession has no '
                   'independent analytic moments, so it cannot be validated '
                   'directly. The verdict above is for the subject (gross) '
                   'object underneath, which is what the cession is computed '
                   'from.')
    if rv == Validation.NOT_UNREASONABLE:
        out.append('Nothing failed, so the object is not unreasonable. That is '
                   'a statement about the numerics reproducing the model, not '
                   'about the model being right for the risk.')
    return ' '.join(out)
