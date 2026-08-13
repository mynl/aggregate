"""Severity distributions: the Severity base class, kind subclasses, and support.

Extracted from ``distributions.py`` (Phase 1, kind split). Imported through the
``distributions`` facade so every existing import path keeps working.
"""

from functools import cached_property, lru_cache
import logging
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import quad, IntegrationWarning
import scipy.stats as ss
from scipy.optimize import newton
from scipy.special import loggamma, binom
from scipy.optimize import NoConvergence  # noqa
from ._help import HelpMixin
from .constants import (FIG_H, FIG_W, INFO_NA, ReflectedSeverityClampWarning,
                        info_row, warn_once)
from .moments import VALIDATION_NOISE
from ._grid_distribution import GridDistribution
from ._labeled import LabeledMixin
from ._program import ProgramMixin
from . import tail as _tail
from .tail import TailClass

from ._fits import (beta_fit, gamma_fit, invgamma_fit,
                    invgauss_fit, lognorm_fit)

logger = logging.getLogger(__name__)

__all__ = [
    'Severity',
    'SeverityScipy',
    'SeverityDHistogram',
    'SeverityCHistogram',
    'SeverityFixed',
    'SeverityMeta',
    'SeverityCopy',
]


# ---------------------------------------------------------------------------
# Single-module helpers — used only inside distributions.py.
# ---------------------------------------------------------------------------


def _partial_e_numeric(fz, a, n):
    """
    Simple numerical integration version of partial_e for auditing purposes.

    Notes
    -----
    Integrates from the distribution's own support lower bound rather than
    from 0. Quadrature over a dead region where the density is identically
    zero (a Type-I Pareto on ``[lambda, inf)`` asked to start at 0) is what
    makes ``quad`` report "the integral is probably divergent": it samples the
    flat part, finds nothing, and concludes badly. ``IntegrationWarning`` is
    suppressed because the caller already tests the returned absolute error
    estimate, which is the honest convergence check and the only one that
    knows the tolerance that matters here.
    """
    lo = float(np.max([0.0, fz.support()[0]]))
    ans = []
    for k in range(n+1):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', IntegrationWarning)
            temp = quad(lambda x: x ** k * fz.pdf(x), lo, a)
        if temp[1] > 1e-4:
            logger.debug('Potential convergence issues with numerical integral')
        ans.append(temp[0])
    return ans


def _partial_e_pareto_type_1(alpha, lam, a, n):
    r"""Partial expected values of a Type-I (single-parameter) Pareto.

    The scipy parameterisation is ``ss.pareto(alpha, scale=lam, loc=0)``:
    support :math:`[\lambda, \infty)`, :math:`S(x) = (\lambda/x)^\alpha` and
    :math:`f(x) = \alpha \lambda^\alpha x^{-\alpha-1}`.

    Parameters
    ----------
    alpha : float
        Shape (tail index).
    lam : float
        Scale, and the lower end of the support.
    a : float
        Upper limit of integration; may be ``np.inf``.
    n : int
        Highest power required; returns ``k = 0 .. n``.

    Returns
    -------
    list of float
        :math:`\int_\lambda^a x^k f(x)\,dx` for ``k = 0, ..., n``.

    Notes
    -----
    .. math::

        \int_\lambda^a x^k f(x)\,dx
          = \alpha\lambda^\alpha\,
            \frac{a^{k-\alpha} - \lambda^{k-\alpha}}{k - \alpha},
        \qquad k \neq \alpha

    and :math:`\alpha\lambda^\alpha\log(a/\lambda)` at the removable case
    :math:`k = \alpha`, which is the limit of the same expression.

    Checks: ``k = 0`` gives :math:`1 - (\lambda/a)^\alpha`, the cdf. With
    ``a = inf`` and :math:`k < \alpha` it gives
    :math:`\alpha\lambda^k/(\alpha-k)`, so ``k = 1`` is the Type-I mean
    :math:`\alpha\lambda/(\alpha-1)`. With :math:`k \geq \alpha` it gives
    ``inf``, which is correct: the moment does not exist, and the moment
    machinery downstream is guarded to report that as ``nan`` rather than
    complain (`[RuntimeWarning-Census]`, 1.0.0a220).

    This replaces a quadrature fallback that ran for *every* Type-I Pareto,
    logged a warning each time and integrated a heavy tail to infinity.
    """
    if a <= lam:
        # The limit sits at or below the support, so no mass is captured.
        return [0.0] * (n + 1)
    ans = []
    for k in range(n + 1):
        d = k - alpha
        if a == np.inf:
            # Converges only while the moment exists.
            ans.append(alpha * lam ** k / (-d) if d < 0 else np.inf)
        elif d == 0:
            ans.append(alpha * lam ** alpha * np.log(a / lam))
        else:
            ans.append(alpha * lam ** alpha * (a ** d - lam ** d) / d)
    return ans


def _partial_e(sev_name, fz, a, n):
    """
    Compute the partial expected value of fz. Computing moments is a bottleneck, so you
    want analytic computation for the most commonly used types.

    Exponential (for mixed exponentials) implemented separate from gamma even though it
    is a special case.

    .. math:

        \\int_0^a x^k fz.pdf(x)dx

    for k=0,...,n as a np.array

    To do: beta? weibull? Burr? invgamma, etc.

    :param sev_name: scipy.stats name for distribution
    :param fz: frozen scipy.stats instance
    :param a: double, limit for integral
    :param n: int, power
    :return: partial expected value
    """

    if sev_name not in ['lognorm', 'gamma', 'pareto', 'expon']:
        raise NotImplementedError(f'{sev_name} NYI for analytic moments')

    if a == 0:
        return [0] * (n+1) # for k in range(n+1)]

    if sev_name == 'lognorm':
        m = fz.stats('m')
        sigma = fz.args[0]
        mu = np.log(m) - sigma**2 / 2
        ans = [np.exp(k * mu + (k * sigma)**2 / 2) *
               (ss.norm.cdf((np.log(a) - mu - k * sigma**2)/sigma) if a < np.inf else 1.0)
               for k in range(n+1)]
        return ans

    elif sev_name == 'expon':
        # really needed for MEDs
        # expon is gamma with shape = 1
        scale = fz.stats('m')
        shape = 1.
        lgs = loggamma(shape)
        ans = [scale ** k * np.exp(loggamma(shape + k) - lgs) *
               (ss.gamma(shape + k, scale=scale).cdf(a) if a < np.inf else 1.0)
               for k in range(n + 1)]
        return ans

    elif sev_name == 'gamma':
        shape = fz.args[0]
        scale = fz.stats('m') / shape
        # magic ingredient is the norming constant
        # c = lambda sh: 1 / (scale ** sh * gamma(sh))
        # therefore c(shape)/c(shape+k) = scale**k * gamma(shape + k) / gamma(shape)
        # = scale ** k * exp(loggamma(shape + k) - loggamma(shape)) to avoid errors
        ans = [scale ** k * np.exp(loggamma(shape + k) - loggamma(shape)) *
               (ss.gamma(shape + k, scale=scale).cdf(a) if a < np.inf else 1.0)
               for k in range(n + 1)]
        return ans

    elif sev_name == 'pareto':
        # integrate xf(x) even though nx^n-1 S(x) may be more obvious
        # former fits into the overall scheme
        # a Pareto defined by agg is like so: ss.pareto(2.5, scale=1000, loc=-1000)
        α = fz.args[0]
        λ = fz.kwds['scale']
        loc = fz.kwds.get('loc', 0.0)
        # regular Pareto is scale=lambda, loc=-lambda, so this has no effect
        # single parameter Pareto is scale=lambda, loc=0
        # these formulae for regular pareto, hence
        if loc == 0.0:
            return _partial_e_pareto_type_1(α, λ, a, n)
        if λ + loc != 0:
            logger.debug('Pareto not shifted to x>0 range...using numeric moments.')
            return _partial_e_numeric(fz, a, n)
        ans = []
        # will return inf if the Pareto does not have the relevant moments
        # TODO: formula for shape=1,2,3
        for k in range(n + 1):
            b = [α * (-1) ** (k - i) * binom(k, i) * λ ** (k + α - i) *
                 ((λ + a) ** (i - α) - λ ** (i - α)) / (i - α)
                 for i in range(k + 1)]
            ans.append(sum(b))
        return ans


def _moms_analytic(fz, limit, attachment, n, analytic=True):
    """
    Return moments of :math:`E[(X-attachment)^+ \\wedge limit]^m`
    for m = 1,2,...,n.

    To check:
    ::

        # fz = ss.lognorm(1.24)
        fz = ss.gamma(6.234, scale=100)
        # fz = ss.pareto(3.4234, scale=100, loc=-100)

        a1 = _moms_analytic(fz, 50, 1234, 3)
        a2 = _moms_analytic(fz, 50, 1234, 3, False)
        a1, a2, a1-a2, (a1-a2) / a1


    :param fz: frozen scipy.stats instance
    :param limit: double, limit (layer width)
    :param attachment: double, limit
    :param n: int, power
    :param analytic: if True use analytic formula, else numerical integrals
    """
    # easy
    if limit == 0:
        return np.array([0.] * n)

    # don't know how robust this will be...
    sev_name = str(fz.__dict__['dist']).split('.')[-1].split('_')[0]

    # compute and store the partial_e
    detachment = attachment + limit
    if analytic is True:
        pe_attach = _partial_e(sev_name, fz, attachment, n)
        pe_detach = _partial_e(sev_name, fz, detachment, n)
    else:
        pe_attach = _partial_e_numeric(fz, attachment, n)
        pe_detach = _partial_e_numeric(fz, detachment, n)

    # The binomial expansion is unusable the moment any partial expectation is
    # infinite: with attachment 0 the low-order coefficients are 0, so a term
    # reads 0 * inf, and with attachment > 0 the alternating signs give
    # inf - inf. Both produce nan for a layer moment that is simply infinite.
    #
    # Short-circuit instead. A layer capped at a finite limit is bounded by
    # limit**m, so every partial expectation to a finite detachment is finite
    # and the sum is safe. Only an UNLIMITED layer can diverge, and the m-th
    # moment of an unlimited excess layer exists exactly when E[X**m] does,
    # which is what pe_detach[m] reports. Lower orders are then finite too
    # (Lyapunov), so the sum below is only ever evaluated on finite terms.
    ans1 = np.array([
        np.inf if not np.isfinite(pe_detach[m]) else
        sum([(-1) ** (m - k) * binom(m, k) * attachment ** (m - k) * (pe_detach[k] - pe_attach[k])
             for k in range(m + 1)])
        for m in range(n + 1)])

    if np.isinf(limit):
        ans2 = np.zeros_like(ans1)
    else:
        ans2 = np.array([limit ** m * fz.sf(detachment) for m in range(n+1)])

    ans = ans1 + ans2

    return ans


def validate_discrete_distribution(xs, ps, allow_negative=False):
    """
    Make sure that outcomes are distinct and sorted in ascending order, and
    that probabilities are summed across distinct outcomes. Used in dsev and
    dfreq to validate user input.

    Parameters
    ----------
    xs, ps : array-like
        Support points and probabilities.
    allow_negative : bool
        If ``False`` (default, used by ``dfreq`` -- claim counts cannot be
        negative) any outcome ``< 0`` is clamped to 0 and merged. If ``True``
        (used by signed ``dsev`` -- a profit is a negative loss) negative
        outcomes are preserved; the result is still made distinct and sorted
        ascending so the histogram bin edges / weights align.

    Returns
    -------
    (xs, ps) : tuple of np.ndarray
        Distinct, ascending support and the corresponding (duplicate-summed)
        probabilities.
    """
    xs = np.asarray(xs, dtype=float)
    ps = np.asarray(ps, dtype=float)
    has_neg = bool(np.any(xs < 0))
    has_dup = len(xs) != len(set(xs.tolist()))
    if has_dup or (has_neg and not allow_negative):
        logger.info('Duplicates in empirical distribution and/or negative values, summarizing.')
        temp_df = pd.DataFrame({'x': xs, 'p': ps})
        if not allow_negative:
            temp_df.loc[temp_df.x < 0, 'x'] = 0.
        temp_df = temp_df.groupby('x')[['p']].sum()
        xs = np.array(temp_df.index)
        ps = temp_df.p.values
    elif allow_negative:
        # signed dsev with distinct outcomes: still sort ascending so the
        # histogram construction pairs each weight with the right bin.
        order = np.argsort(xs)
        xs = xs[order]
        ps = ps[order]
    return xs, ps


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


# ---------------------------------------------------------------------------
# Layer/attachment decorator factories. Parallel to ``make_conditional_*``
# above but applied AFTER it — splice (lb/ub) modifies the severity
# distribution; layer/attachment is the policy sitting on top.
#
# Each factory resolves the conditional/unconditional branch (and, for pdf,
# the pattach<1 sub-branch) at wrap time so hot-path calls do not re-test
# these flags on every invocation. Closure captures the layer parameters
# at wrap time, so mutating the Severity instance's attachment/limit/pattach
# /pdetach/conditional AFTER construction produces stale results — see the
# Warnings note on ``Severity.__init__``.
# ---------------------------------------------------------------------------


def make_layer_attachment_cdf(attachment, limit, pattach, conditional):
    """Decorator that wraps a CDF to apply the layered-loss transform.

    Notes
    -----
    Maps layered-loss x in [0, limit] to the underlying value x + attachment
    before calling ``fzcdf``. Conditional: rescales by ``pattach`` after
    removing the truncated below-attachment mass. Unconditional: cdf(0) =
    1 - pattach (mass at zero from P(X <= attachment)); cdf(>= limit) = 1.
    """
    def actual_decorator(fzcdf):
        if conditional:
            def wrapper(x):
                return np.where(x >= limit, 1,
                                np.where(x < 0, 0,
                                         (fzcdf(x + attachment) - (1 - pattach)) / pattach))
        else:
            def wrapper(x):
                return np.where(x < 0, 0,
                                np.where(x == 0, 1 - pattach,
                                         np.where(x > limit, 1,
                                                  fzcdf(x + attachment))))
        return wrapper

    return actual_decorator


def make_layer_attachment_sf(attachment, limit, pattach, conditional):
    """Decorator that wraps a survival function for the layered-loss transform.

    Notes
    -----
    Conditional: rescales by ``pattach``. Unconditional: sf(0) = pattach
    (probability that any layer claim occurs at all); sf(>= limit) = 0.
    """
    def actual_decorator(fzsf):
        if conditional:
            def wrapper(x):
                return np.where(x >= limit, 0,
                                np.where(x < 0, 1,
                                         fzsf(x + attachment) / pattach))
        else:
            def wrapper(x):
                return np.where(x < 0, 1,
                                np.where(x == 0, pattach,
                                         np.where(x > limit, 0,
                                                  fzsf(x + attachment))))
        return wrapper

    return actual_decorator


def make_layer_attachment_pdf(attachment, limit, detachment, pattach, pdetach,
                              conditional):
    """Decorator that wraps a PDF for the layered-loss transform.

    Notes
    -----
    Three resolved bodies depending on construction:

    - Conditional: rescaled by ``pattach`` with a point mass at ``limit``
      when ``pdetach > 0`` (the layer ceiling absorbs detachment probability).
    - Unconditional with ``pattach < 1``: extra ``inf`` at x = 0 marks the
      lump from ``P(X <= attachment)``.
    - Unconditional with ``pattach == 1``: no mass at zero; otherwise as above.

    The point masses at ``limit`` / ``detachment`` reflect the absorbed
    probability beyond the policy ceiling.
    """
    def actual_decorator(fzpdf):
        if conditional:
            limit_mass = np.inf if pdetach > 0 else 0

            def wrapper(x):
                return np.where(x >= limit, 0,
                                np.where(x == limit, limit_mass,
                                         fzpdf(x + attachment) / pattach))
        elif pattach < 1:
            def wrapper(x):
                return np.where(x < 0, 0,
                                np.where(x == 0, np.inf,
                                         np.where(x == detachment, np.inf,
                                                  np.where(x > detachment, 0,
                                                           fzpdf(x + attachment)))))
        else:
            def wrapper(x):
                return np.where(x < 0, 0,
                                np.where(x == detachment, np.inf,
                                         np.where(x > detachment, 0,
                                                  fzpdf(x + attachment))))
        return wrapper

    return actual_decorator


def make_layer_attachment_isf(attachment, limit, pattach, pdetach, conditional):
    """Decorator that wraps an inverse survival function for the layered-loss transform.

    Notes
    -----
    Conditional: q is rescaled by ``pattach`` for the underlying call; the
    layer ceiling kicks in when q < pdetach / pattach. Unconditional: 0
    when q >= pattach (no layer claim); ``limit`` when q < pdetach.
    """
    def actual_decorator(fzisf):
        if conditional:
            threshold = pdetach / pattach

            def wrapper(q):
                return np.where(q < threshold, limit,
                                fzisf(q * pattach) - attachment)
        else:
            def wrapper(q):
                return np.where(q >= pattach, 0,
                                np.where(q < pdetach, limit,
                                         fzisf(q) - attachment))
        return wrapper

    return actual_decorator


def make_layer_attachment_ppf(attachment, limit, pattach, pdetach, conditional):
    """Decorator that wraps a percent-point function for the layered-loss transform.

    Notes
    -----
    Conditional: rescales the residual tail probability into the underlying
    ppf input. Unconditional: 0 below the below-layer mass and ``limit``
    above the at-detachment mass.
    """
    def actual_decorator(fzppf):
        if conditional:
            threshold = 1 - pdetach / pattach

            def wrapper(q):
                return np.where(q > threshold, limit,
                                fzppf(1 - pattach * (1 - q)) - attachment)
        else:
            def wrapper(q):
                return np.where(q <= 1 - pattach, 0,
                                np.where(q > 1 - pdetach, limit,
                                         fzppf(q) - attachment))
        return wrapper

    return actual_decorator


# ---------------------------------------------------------------------------
# Severity module-level scaffolding (Stage 1d).
#
# These helpers are used by the ``Severity`` registry/subclass machinery added
# below. They are introduced as a self-contained block of additions; the
# existing ``Severity.__init__`` / ``moms()`` body continues to function until
# the subclass forms are wired up in later refactor steps.
# ---------------------------------------------------------------------------


def _scalar_bound(v):
    """Coerce a splice bound (``sev_lb`` / ``sev_ub``) to a Python scalar.

    Notes
    -----
    The DecL parser returns single-element lists for the single-segment
    splice form (``splice [a b]`` -> ``sev_lb=[a]``, ``sev_ub=[b]``).
    Severity treats these as scalars throughout; allowing them to stay as
    1-element arrays would make ``np.where(...)`` in the wrapped scipy
    methods return ``(1,)``-shaped outputs that ``scipy.integrate.quad``
    cannot consume.
    """
    arr = np.asarray(v)
    if arr.size == 1:
        return arr.item()
    raise ValueError(
        f'Splice bound must be a scalar or length-1 sequence; got {v!r} '
        f'(size {arr.size}). Multi-segment splice is not implemented.'
    )


def _classify_sev(sev_name, sev_xs):
    """Classify a ``Severity`` constructor call into a single registry key.

    Parameters
    ----------
    sev_name : str | Severity | Aggregate | Portfolio
        Same first argument as ``Severity.__init__``.
    sev_xs : array-like | None
        The ``sev_xs`` keyword argument; presence flips classification into
        the histogram branch.

    Returns
    -------
    str
        One of ``'fixed'``, ``'dhistogram'``, ``'chistogram'``, ``'copy'``,
        ``'meta'``, ``'scipy'``. Unrecognized string ``sev_name`` values are
        classified as ``'scipy'`` so the catchall ``SeverityScipy`` can raise
        a clearer error when scipy itself can't resolve the name.
    """
    # Local imports to avoid distributions <-> portfolio / _aggregate cycles.
    from .portfolio import Portfolio
    from ._aggregate import Aggregate

    if sev_xs is not None:
        if sev_name == 'fixed':
            return 'fixed'
        if sev_name == 'dhistogram':
            return 'dhistogram'
        if sev_name == 'chistogram':
            return 'chistogram'
        # Fall through: caller passed sev_xs with a non-histogram name; let
        # the scipy path raise.
        return 'scipy'
    if isinstance(sev_name, Severity):
        return 'copy'
    if isinstance(sev_name, (Aggregate, Portfolio)):
        return 'meta'
    return 'scipy'


def _cv_to_shape(sev_name, cv, hint=1):
    """Shape parameter from coefficient of variation for a scipy distribution.

    Analytic for ``lognorm``, ``gamma``, ``invgamma``, ``invgauss``; otherwise
    falls back to a Newton solve against the frozen RV's CV.

    Parameters
    ----------
    sev_name : str
        scipy.stats distribution name.
    cv : float
        Target coefficient of variation.
    hint : float
        Initial guess for the numerical fallback.

    Returns
    -------
    (shape, fz) : tuple
        The shape parameter and a frozen scipy.stats RV with that shape.
        Returns ``(np.inf, None)`` if the numerical solver fails.
    """
    if sev_name == 'lognorm':
        _, sigma = lognorm_fit(1.0, cv)
        return sigma, ss.lognorm(sigma)
    if sev_name == 'gamma':
        alpha, _ = gamma_fit(1.0, cv)
        return alpha, ss.gamma(alpha)
    if sev_name == 'invgamma':
        a = invgamma_fit(cv)
        return a, ss.invgamma(a)
    if sev_name == 'invgauss':
        mu = invgauss_fit(cv)
        return mu, ss.invgauss(mu)

    gen = getattr(ss, sev_name)

    def _residual(shape):
        fz0 = gen(shape)
        mean, var = fz0.stats('mv')
        return cv - var ** 0.5 / mean

    try:
        shape = newton(_residual, hint)
    except RuntimeError:
        logger.error('_cv_to_shape | newton solve failed for %s, cv=%s', sev_name, cv)
        return np.inf, None
    return shape, gen(shape)


def _mean_to_scale(sev_name, shape, mean, loc=0):
    """Rescale a scipy distribution so its first moment matches ``mean``.

    Parameters
    ----------
    sev_name : str
        scipy.stats distribution name.
    shape : float
        The (already-determined) shape parameter.
    mean : float
        Target mean for the resulting frozen RV.
    loc : float
        Optional location parameter.

    Returns
    -------
    (scale, fz) : tuple
        The chosen scale and the frozen RV with ``(shape, scale=scale, loc=loc)``.

    Notes
    -----
    Uses the identity ``E[scale * X] = scale * E[X]`` after computing the
    unit-scale mean from a temporary unit-scale frozen RV.
    """
    gen = getattr(ss, sev_name)
    unit = gen(shape)
    scale = mean / unit.stats('m')
    return scale, gen(shape, scale=scale, loc=loc)


def _safe_integrate(f, lower, upper, level, sev_name=''):
    """Integrate ``f`` over ``[lower, upper]`` with scipy.integrate.quad.

    Parameters
    ----------
    f : callable
        Integrand.
    lower, upper : float
        Integration bounds.
    level : int
        Moment order; used to pick the relative-error tolerance (1e-6 for
        n=1, 1e-4 for n>=2) and for diagnostic logging.
    sev_name : str
        Severity name, included in log messages.

    Returns
    -------
    (value, abs_error) : tuple of float

    Notes
    -----
    Algorithm copied verbatim from the ``safe_integrate`` closure in the
    pre-refactor ``Severity.moms``: when ``quad`` flags divergence (or
    returns ``inf``), the integral is retried split at ``epsilon=1e-4`` to
    handle integrands that misbehave near zero.
    """
    argkw = dict(limit=100, epsrel=1e-6 if level == 1 else 1e-4, full_output=1)
    ex = quad(f, lower, upper, **argkw)
    if len(ex) == 4 or ex[0] == np.inf:
        msg = ex[-1].replace("\n", " ") if ex[-1] == str else "no message"
        logger.info(
            f'E[X^{level}]: ansr={ex[0]}, error={ex[1]}, steps={ex[2]["last"]}; '
            f'message {msg} -> splitting integral')
        ϵ = 0.0001
        if lower == 0 and upper > ϵ:
            logger.info(
                f'_safe_integrate | splitting {sev_name} EX^{level} integral '
                f'for convergence reasons')
            exa = quad(f, 1e-16, ϵ, **argkw)
            exb = quad(f, ϵ, upper, **argkw)
            logger.info(
                f'_safe_integrate | [1e-16, {ϵ}] split EX^{level}: '
                f'ansr={exa[0]}, error={exa[1]}, steps={exa[2]["last"]}')
            logger.info(
                f'_safe_integrate | [{ϵ}, {upper}] split EX^{level}: '
                f'ansr={exb[0]}, error={exb[1]}, steps={exb[2]["last"]}')
            ex = (exa[0] + exb[0], exa[1] + exb[1])
    logger.info(
        f'E[X^{level}]={ex[0]}, error={ex[1]}, '
        f'est rel error={ex[1] / ex[0] if ex[0] != 0 else np.inf}')
    return ex[:2]


def _numerical_moms(severity):
    """Numerical-integration fallback for ``Severity.moms``.

    Parameters
    ----------
    severity : Severity
        The instance whose layered moments are wanted; the function reads
        ``fz``, ``sev_name``, ``attachment``, ``detachment``, ``pattach``,
        ``moment_pattach``, ``exp_attachment``, and ``conditional`` from it.

    Returns
    -------
    (m1, m2, m3) : tuple of float
        First three moments of ``X(a, d) = min(d, (X - a)+)`` with the
        conditional adjustment applied if ``severity.conditional`` is True.
        ``np.nan`` entries signal that the numerical integration produced
        an unreliable result; ``np.inf`` entries are correct (moment does
        not exist).

    Notes
    -----
    Algorithm and tolerances preserved verbatim from the pre-refactor
    ``Severity.moms`` numerical branch. Integration is performed in
    quantile (isf) space rather than over the unbounded x-axis so the
    interval is compact and the heavy tail is naturally truncated by
    the survival probability at ``detachment``.
    """
    # Integration bounds in isf-space — note upper/lower are swapped
    # relative to x-space (isf is monotone decreasing).
    if severity.attachment == 0:
        upper = min(1, severity.moment_pattach)
    else:
        upper = severity.fz.sf(severity.attachment)
    if severity.detachment == np.inf:
        lower = 0
    else:
        lower = severity.fz.sf(severity.detachment)

    if severity.detachment == np.inf and not severity._is_histogram:
        moments_finite = list(map(
            lambda x: not (np.isinf(x) or np.isnan(x)),
            severity.fz.stats('mvs')))
    else:
        moments_finite = [True, True, True]

    logger.info('Numerical moments')
    continue_calc = True
    max_rel_error = 1e-3

    if upper <= lower:
        # Zero-width integration window: arises when the (possibly spliced)
        # severity support sits entirely above the policy attachment AND
        # the detachment also lies at or below the support — every claim
        # is exactly the full limit. Skip the integration; the
        # binomial-expansion adjustment block below produces the correct
        # full-limit moments via the ``dma * lower`` term. Without this
        # short-circuit the ``ex1[0] != 0`` rel-error check would reject
        # the genuine zero integral as failure and propagate NaN through
        # the result.
        ex1 = ex2 = ex3 = 0.0
    else:
        if moments_finite[0]:
            ex1 = _safe_integrate(severity.fz.isf, lower, upper, 1, severity.sev_name)
            if ex1[0] != 0 and ex1[1] / ex1[0] < max_rel_error:
                ex1 = ex1[0]
            else:
                ex1 = np.nan
                continue_calc = False
        else:
            logger.info('First moment does not exist.')
            ex1 = np.inf

        if continue_calc and moments_finite[1]:
            ex2 = _safe_integrate(lambda x: severity.fz.isf(x) ** 2,
                                  lower, upper, 2, severity.sev_name)
            if ex2[1] / ex2[0] < max_rel_error:
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
            ex3 = _safe_integrate(lambda x: severity.fz.isf(x) ** 3,
                                  lower, upper, 3, severity.sev_name)
            if ex3[1] / ex3[0] < max_rel_error:
                ex3 = ex3[0]
            else:
                ex3 = np.nan
        elif not continue_calc:
            ex3 = np.nan
        else:
            logger.info('Third moment does not exist.')
            ex3 = np.inf

    # Attachment/detachment adjustments: convert raw integrals into the
    # layered moments E[X(a, d)^k].
    dma = severity.detachment - severity.attachment
    uml = upper - lower
    a = severity.attachment
    if a > 0:
        ex1a = ex1 - a * uml
        ex2a = ex2 - 2 * a * ex1 + a ** 2 * uml
        ex3a = ex3 - 3 * a * ex2 + 3 * a ** 2 * ex1 - a ** 3 * uml
    else:
        # a == 0: handle the rare continuous-with-mass-at-zero case where
        # ``exp_attachment is None`` signals "no layer clause" and the raw
        # integrals must be scaled down by ``pattach``.
        if severity.exp_attachment is None and not severity._is_histogram:
            ex1a = severity.pattach * ex1
            ex2a = severity.pattach * ex2
            ex3a = severity.pattach * ex3
        else:
            ex1a = ex1
            ex2a = ex2
            ex3a = ex3

    if severity.detachment < np.inf:
        ex1a += dma * lower
        ex2a += dma ** 2 * lower
        ex3a += dma ** 3 * lower

    if severity.conditional:
        ex1a /= severity.pattach
        ex2a /= severity.pattach
        ex3a /= severity.pattach

    return ex1a, ex2a, ex3a


class Severity(HelpMixin, LabeledMixin, ProgramMixin, ss.rv_continuous):
    # Registry of concrete kind subclasses, populated by ``__init_subclass__``.
    # Keys are the string returned by ``_classify_sev``.
    _registry: dict = {}

    # Subclasses override this to register themselves; the empty default on
    # the base means "do not register".
    sev_kind: str = ''

    # Histogram-shaped kinds set this to ``True`` so the post-build helpers
    # and ``moms()`` can branch without inspecting ``sev_name`` as a string.
    _is_histogram: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only register classes that declared ``sev_kind`` on themselves —
        # an inherited value from a parent subclass should not double-register.
        kind = cls.__dict__.get('sev_kind', '')
        if kind:
            Severity._registry[kind] = cls

    def __new__(cls, sev_name=None, exp_attachment=None, exp_limit=np.inf,
                sev_mean=0, sev_cv=0, sev_a=np.nan, sev_b=0,
                sev_loc=0, sev_scale=0, sev_xs=None, *args, **kwargs):
        # Direct instantiation of a concrete subclass (e.g. ``SeverityScipy(...)``)
        # bypasses the dispatch — just construct the requested class. The
        # ``__init__`` body still runs.
        if cls is not Severity:
            return super().__new__(cls)
        # Top-level ``Severity(...)`` call: classify and dispatch. If no
        # subclass is registered for the classified kind, fall through to
        # the base class — keeps the path live during the migration while
        # subclasses are added incrementally.
        # Signature mirrors ``__init__`` up through ``sev_xs`` so positional
        # callers (e.g. ``Aggregate.__init__``) reach the histogram branch.
        kind = _classify_sev(sev_name, sev_xs)
        target = cls._registry.get(kind, cls)
        return super().__new__(target)

    def __init__(self, sev_name, exp_attachment=None, exp_limit=np.inf, sev_mean=0, sev_cv=0, sev_a=np.nan, sev_b=0,
                 sev_loc=0, sev_scale=0, sev_xs=None, sev_ps=None, sev_wt=1, sev_lb=0, sev_ub=np.inf,
                 sev_conditional=True, sev_signed=False, sev_reflect=False, name='', note='', hints='',
                 tags=(), doc='',
                 label=None, label_map=None):
        """Continuous random variable adding layer/attachment to ``ss.rv_continuous``.

        Construction is delegated to a registered subclass — ``__new__``
        classifies the inputs via :func:`_classify_sev` and dispatches to
        :class:`SeverityScipy`, :class:`SeverityDHistogram`,
        :class:`SeverityCHistogram`, :class:`SeverityFixed`,
        :class:`SeverityMeta`, or :class:`SeverityCopy`. The chosen
        subclass's ``_build`` populates ``self.fz`` from the stored spec
        inputs; the post-build helpers below then apply truncation
        decorators, compute attachment/detachment probabilities, and
        validate the achieved mean/CV against the targets.

        Parameters
        ----------
        sev_name : str | Severity | Aggregate | Portfolio
            scipy.stats distribution name (e.g. ``lognorm``), special form
            ``dhistogram`` / ``chistogram`` / ``fixed``, or an existing
            ``Severity``/``Aggregate``/``Portfolio`` instance for the
            copy / meta paths.
        exp_attachment : float | None
            Layer attachment point. ``None`` means "no layer clause" and
            conditions on ``X >= 0``; ``0`` means "conditional on ``X > 0``"
            (this distinction matters for distributions with mass at zero).
        exp_limit : float
            Layer width (the "y" in ``y xs a``).
        sev_mean, sev_cv : float
            Target mean / coefficient of variation for distributions
            parameterised by moments (lognorm, gamma, beta, …).
        sev_a, sev_b : float
            scipy shape parameters where applicable.
        sev_loc, sev_scale : float
            scipy location / scale.
        sev_xs, sev_ps : array-like
            Support points and probabilities for the histogram kinds.
        sev_wt : float
            Mixture weight (not used internally; passed through for callers).
        sev_lb, sev_ub : float
            Optional support bounds; if not the trivial ``[0, inf]`` the
            scipy methods are wrapped with conditional-truncation decorators.
        sev_conditional : bool
            Whether layered moments / functions divide out ``P(X > attachment)``.
        name : str
            Identifier (e.g. set by ``sev SOMENAME …`` in DecL).
        note : str
            Free-text annotation (from a ``note{...}`` clause).
        hints : str
            Raw ``hints{...}`` build-settings string; retained as annotation
            (a standalone ``Severity`` has no build settings to apply).

        Warnings
        --------
        Layer/attachment parameters (``attachment``, ``limit``, ``detachment``,
        ``pattach``, ``pdetach``, ``conditional``) and splice parameters
        (``sev_lb``, ``sev_ub``) are captured in closure by
        :meth:`_apply_lb_ub` and :meth:`_apply_layer_attachment` at the end of
        ``__init__``. Mutating these attributes on an existing instance
        produces stale results because the wrapped methods on ``self.fz``
        retain the original values. If the policy changes, build a new
        ``Severity`` rather than reassigning fields on an existing one.

        Raises
        ------
        ValueError
            If ``sev_lb`` / ``sev_ub`` describe a splice window with zero
            probability mass under the underlying distribution
            (``fz.cdf(sev_ub) - fz.cdf(sev_lb) <= 1e-15``); conditioning on
            a measure-zero set is mathematically undefined.
        """
        super().__init__(self, name=sev_name if isinstance(sev_name, str) else '')

        # ---- spec inputs / placeholder state -----------------------------
        self.program = ''  # may be set externally
        self.limit = exp_limit
        self.attachment = 0 if exp_attachment is None else exp_attachment
        # Distinguish "no layer clause" (None) from "explicit 0 attachment".
        # Treatment of mass at zero depends on this.
        self.exp_attachment = exp_attachment
        self.detachment = exp_limit + self.attachment
        self.fz = None
        self.pattach = 0
        self.moment_pattach = 0
        self.pdetach = 0
        self.conditional = sev_conditional
        # Signed (never-clamp) severity: a profit is a negative loss. When True
        # the layer wrappers are the identity (no x<0 -> 0 clamp) and moments are
        # the raw distribution's. Set explicitly here (``ssev`` keyword) or by a
        # discrete ``_build`` that finds negative atoms. Orthogonal to value_type.
        self.signed = bool(sev_signed)
        # Reflected (negatively-scaled) severity: ``-1 * X``, optionally shifted
        # to ``shift - X`` by a trailing ``+/- shift``. scipy cannot carry a
        # negative scale, so ``_build`` constructs the positive base at loc 0 and
        # ``_apply_reflect`` maps it to ``shift - X`` (a signed severity). The
        # shift is captured here, before ``_build`` zeroes the build loc.
        self.sev_reflect = bool(sev_reflect)
        self._reflect_shift = float(_scalar_bound(sev_loc)) if self.sev_reflect else 0.0
        self.sev_name = sev_name
        self.name = name
        # Object-level display label + interior label_map. Presentation only;
        # ``name`` stays the identity handle. See dev/plan-labels.md.
        self._init_labels(label=label, label_map=label_map)
        self.long_name = sev_name
        self.note = note
        self.hints = hints
        #: Tag slugs from the DecL ``tags{...}`` trailer ('()' when none).
        self.tags = tuple(tags)
        #: Long-form markdown recipe from ``doc{{{...}}}`` ('' when none).
        self.doc = doc
        self.sev1 = self.sev2 = self.sev3 = None
        self.sev_wt = sev_wt
        self.sev_loc = sev_loc
        # The DecL parser returns ``sev_lb`` / ``sev_ub`` as 1-element
        # sequences for single-segment splices (and may pass multi-element
        # sequences for the never-implemented multi-segment form). Coerce
        # to a Python scalar so downstream ``np.where`` calls in the
        # ``make_conditional_*`` decorators return 0-d outputs that
        # QUADPACK can consume in ``_safe_integrate``.
        self.sev_lb = _scalar_bound(sev_lb)
        self.sev_ub = _scalar_bound(sev_ub)
        self.sev_mean = sev_mean
        self.sev_cv = sev_cv
        self.sev_a = sev_a
        self.sev_b = sev_b
        self.sev_scale = sev_scale
        self.sev_xs = sev_xs
        self.sev_ps = sev_ps
        logger.debug(
            f'Severity.__init__ | creating new Severity {self.sev_name} at {super().__repr__()}')

        # ---- subclass-specific construction ------------------------------
        # ``_build`` may also set ``self.signed`` (a discrete severity that
        # finds negative atoms is signed regardless of the constructor flag).
        self._build()

        # ---- shared post-build steps -------------------------------------
        # Order is load-bearing: splice (lb/ub) modifies the underlying
        # distribution FIRST; then the optional reflection maps it to
        # ``shift - X``; then either the signed (identity) layering or the
        # ordinary clamped layer wraps on top. Attachment probabilities are
        # computed against the fully transformed fz, so the layer sees the
        # reflected law when there is one.
        #
        # Signedness and reflection are independent. ``ssev`` sets ``signed``
        # through the constructor, a discrete ``_build`` sets it when it finds a
        # negative atom, and a reflection under plain ``sev`` sets neither: it
        # clamps at zero like ``10 * norm + 5`` does. See
        # dev/done/plan-reflected-loss-severity.md.
        self._apply_lb_ub()
        if not self.signed:
            # Targets from ``mean cv`` describe the pre-reflection base X, not
            # ``shift - X``, so validate before reflecting.
            self._validate_moments()
        if self.sev_reflect:
            self._apply_reflect()
            if not self.signed:
                self._warn_reflected_clamp()
        if self.signed:
            # Signed (never-clamp) severity: keep the (possibly spliced,
            # possibly reflected) distribution. Skip the x<0 -> 0 / attachment
            # clamp and the layered-loss transform; the layered methods are the
            # raw fz methods (identity). Moments are the raw distribution's;
            # discrete kinds already populated sev1/sev2/sev3 in ``_build``.
            self._apply_signed()
        else:
            self._compute_attachment_probs()
            self._apply_layer_attachment()

        assert self.fz is not None

    def _build(self):
        """Subclass hook: populate ``self.fz`` from stored spec inputs.

        Notes
        -----
        Concrete ``Severity<Kind>`` subclasses override this. Reaching the
        base implementation means ``__new__`` found no registered subclass
        for the classified kind — usually a sign that ``Severity(...)`` was
        called with an unsupported ``sev_name``.
        """
        raise NotImplementedError(
            f'Severity._build not implemented for type {type(self).__name__!r} '
            f'(sev_name={self.sev_name!r}). Registered kinds: {sorted(Severity._registry)}'
        )

    def _support_phrase(self):
        """Short declared-support phrase, folded into the ``tail_*`` narrative.

        For a histogram / discrete severity, render the **actual support** --
        ``atoms {a, b, …}`` (shortened to first/last few when there are many,
        e.g. ``dsev [1:1001]``) or ``support [lo, hi]`` -- instead of the
        layer form ``limit xs attachment``, which is meaningless for a discrete
        or signed severity (e.g. ``dsev [-2 5]`` is *not* a ``5 xs 0`` layer).
        For a continuous severity keep the familiar ``unlimited`` /
        ``limit xs attachment`` rendering.

        Private since a149: this was the public ``support_description``, whose
        content now reads out through :attr:`tail_description` and
        :attr:`tail_explanation` (the shared narrative pair every class
        carries) rather than through a Severity-only name.

        Returns
        -------
        str
        """
        atoms = getattr(self, 'support_atoms', None)
        if atoms is not None and len(atoms):
            def _fmt(v):
                return f'{v:g}'
            if len(atoms) <= 8:
                body = ' '.join(_fmt(v) for v in atoms)
                return f'atoms [{body}]'
            head = ' '.join(_fmt(v) for v in atoms[:3])
            tail = ' '.join(_fmt(v) for v in atoms[-2:])
            return (f'{len(atoms)} atoms [{head} ... {tail}] on '
                    f'[{_fmt(atoms[0])}, {_fmt(atoms[-1])}]')
        if self.signed:
            lo, hi = self.fz.support()
            return f'signed, support [{lo:g}, {hi:g}]'
        if self.limit == np.inf and self.attachment == 0:
            return 'unlimited'
        return f'{self.limit:,.0f} xs {self.attachment:,.0f}'

    @property
    def info(self):
        """Fixed-layout multi-line summary string (terse).

        Every row is always present, in the same order, for every
        ``Severity``; a value that does not apply renders as ``n/a``. Shares
        the label/value convention (:func:`aggregate.constants.info_row`) with
        ``Aggregate`` / ``Portfolio`` / ``Frequency``. The row catalogue is
        documented in ``dev/info-strings.rst``.
        """
        rows = [
            ('severity object name', self.name or INFO_NA),
            ('severity distribution', self.long_name),
            ('declared mean', f'{self.sev_mean:,.6g}' if self.sev_mean else INFO_NA),
            ('declared cv', f'{self.sev_cv:,.6g}' if self.sev_cv else INFO_NA),
            ('layer', self._support_phrase()),
            ('conditional', self.conditional),
            ('signed', self.signed),
            ('mean', f'{self.actual_m:,.6g}'),
            ('cv', f'{self.actual_cv:,.6g}'),
            ('sd', f'{self.actual_sd:,.6g}'),
            ('skew', f'{self.actual_skew:,.6g}'),
        ]
        s = [info_row(label, value) for label, value in rows]
        s.append(info_row('severity tail', self.tail_description))
        s.append(info_row('bounded', self.bounded))
        return '\n'.join(s)

    @cached_property
    def _actual_moments(self):
        """``(mean, var, sd, cv, skew)`` of the layered severity, from :meth:`moms`.

        Cached: :meth:`moms` may fall through to numerical integration, and the
        five ``actual_*`` properties must not pay for it five times.
        """
        ex1, ex2, ex3 = (float(v) for v in self.moms())
        with np.errstate(invalid='ignore', over='ignore'):
            var = ex2 - ex1 * ex1 if np.isfinite(ex2) else np.inf
            if np.isfinite(var):
                var = max(var, 0.0)
            sd = np.sqrt(var)
            cv = sd / ex1 if ex1 else np.nan
            if np.isfinite(ex3) and np.isfinite(sd) and sd > 0:
                skew = (ex3 - 3 * ex1 * ex2 + 2 * ex1 ** 3) / sd ** 3
            else:
                skew = np.inf if not np.isfinite(ex3) else np.nan
        return float(ex1), float(var), float(sd), float(cv), float(skew)

    # The severity moment surface is ``actual_*`` -- analytic (or exactly
    # summed, for a discrete law), never read off a grid, which is why there is
    # no ``est_*`` counterpart here: a standalone Severity is never
    # discretized. (The *aggregate's* discretized severity moments live on
    # ``Aggregate.est_sev_*``.) scipy's ``mean()`` / ``var()`` / ``std()`` /
    # ``stats()`` remain available as the inherited rv_continuous surface.
    @property
    def actual_m(self):
        """Severity mean ``E[X(a, d)]`` (analytic, post-layer, post-splice)."""
        return self._actual_moments[0]

    @property
    def actual_var(self):
        """Severity variance (analytic)."""
        return self._actual_moments[1]

    @property
    def actual_sd(self):
        """Severity standard deviation (analytic)."""
        return self._actual_moments[2]

    @property
    def actual_cv(self):
        """Severity coefficient of variation (analytic).

        The *achieved* CV. Compare :attr:`sev_cv`, the CV **requested** in the
        declaration -- they differ whenever a layer, splice or shift is applied
        (and their agreement on an unlimited severity is exactly what
        ``_validate_moments`` checks at construction).
        """
        return self._actual_moments[3]

    @property
    def actual_skew(self):
        """Severity skewness (analytic)."""
        return self._actual_moments[4]

    @property
    def tail_class(self):
        """This severity's :class:`~aggregate.tail.TailClass` rung.

        Deterministic family lookup (param-aware): structural-bounded test
        first (so ``bounded`` is spec-only), then scipy family. See
        :func:`aggregate.tail.classify_severity`.
        """
        return _tail.classify_severity(self)[0]

    @property
    def bounded(self) -> bool:
        """Whether the (post-layer, post-splice) severity has bounded support.

        Derived view: ``True`` iff :attr:`tail_class` is
        :attr:`~aggregate.tail.TailClass.BOUNDED` — i.e. ``fixed`` /
        ``dhistogram`` / ``chistogram`` (finite by construction), a ``scipy``
        family in :data:`~aggregate.tail._BOUNDED_SCIPY_SEVS`, a finite layer
        ``exp_limit`` or splice ``sev_ub``, or a ``meta`` / ``copy`` wrapping a
        bounded object.
        """
        return self.tail_class == TailClass.BOUNDED

    @property
    def tail_description(self) -> str:
        """One line: family, declared support, claim-space support, tail class.

        E.g. ``lognorm, [0, inf), subexponential right tail`` for an unlimited
        severity, or ``lognorm, [0, 500], bounded; 500 xs 250`` once a layer is
        declared and ``dhistogram, [1, 6], bounded; atoms [1 2 3 4 5 6]`` for a
        ``dsev``. Derived from the same
        :func:`~aggregate.tail.severity_tail_row` as the aggregate's
        :attr:`~aggregate.distributions.Aggregate.tail_behavior_df` ``comp``
        rows, with the declared layer / atom phrase appended when it says
        something the interval does not (a149: this absorbed the old
        ``support_description``). The verbose form is :attr:`tail_explanation`.
        """
        base = _tail.describe_row(_tail.severity_tail_row(self, 'severity'))
        extra = self._support_phrase()
        return base if extra in ('', 'unlimited') else f'{base}; {extra}'

    # ``program`` / ``format_program`` / ``pprogram`` / ``pprogram_html`` come
    # from ``ProgramMixin``. The mixin's ``if not self.program: return ''``
    # guard is what a Severity always needed: an inline ``sev`` clause has no
    # program of its own (the enclosing Aggregate owns the text).

    @property
    def tail_explanation(self) -> str:
        """Verbose prose over this severity's support and tail behavior.

        The Severity twin of
        :attr:`~aggregate.distributions.Aggregate.tail_explanation`: the family
        and its claim-space support and per-side classes, then the declared
        layer / atom support, then what the tail class means for the moments
        (the power-law tail index and the first moment that fails to exist, or
        the guarantee that every moment is finite on a bounded support).
        """
        row = _tail.severity_tail_row(self, 'severity')
        out = [_tail.explain_rows([row])]
        extra = self._support_phrase()
        if extra:
            out.append(f'Declared support: {extra}.')
        rung, _, alpha = _tail.classify_severity(self)
        if self.bounded:
            out.append('Support is bounded, so every moment is finite.')
        elif rung == TailClass.POWER_LAW and alpha is not None:
            k = int(np.floor(alpha))
            out.append(f'Power-law right tail with index alpha ~ {alpha:.3g}: '
                       f'moments of order < {alpha:.3g} are finite, so E[X^{k + 1}] '
                       f'and above are infinite.')
        elif _tail.is_thick(rung):
            out.append('The right tail is thick (subexponential): the aggregate '
                       'inherits it by the single big jump.')
        else:
            out.append('The right tail is thin (exponential or lighter): every '
                       'moment is finite.')
        return ' '.join(out)

    def _apply_lb_ub(self):
        """Wrap ``self.fz`` methods with truncation decorators for ``[lb, ub]``.

        Notes
        -----
        No-op when the bounds are the trivial ``[0, inf]``. Otherwise every
        scipy method (``cdf``, ``sf``, ``isf``, ``ppf``, ``pdf``) is replaced
        on the frozen RV instance with a conditional version that rescales
        probabilities to the ``[lb, ub]`` window.

        Raises
        ------
        ValueError
            If the splice window ``[sev_lb, sev_ub]`` has zero probability
            mass under the underlying distribution
            (``fz.cdf(sev_ub) - fz.cdf(sev_lb) <= 1e-15``). Conditioning on
            a measure-zero set is mathematically undefined.
        """
        if self.sev_lb == 0 and self.sev_ub == np.inf:
            return
        plb = self.fz.cdf(self.sev_lb)
        pub = self.fz.cdf(self.sev_ub)
        if pub - plb <= 1e-15:
            raise ValueError(
                f'Severity {self.sev_name!r} splice [{self.sev_lb}, {self.sev_ub}] '
                f'has zero probability mass (CDF at lb = {plb}, CDF at ub = {pub}). '
                f'Conditioning on a measure-zero set is undefined.'
            )
        self.fz.cdf = make_conditional_cdf(self.sev_lb, self.sev_ub, plb, pub)(self.fz.cdf)
        self.fz.sf  = make_conditional_sf (self.sev_lb, self.sev_ub, plb, pub)(self.fz.sf)   # noqa
        self.fz.isf = make_conditional_isf(self.sev_lb, self.sev_ub, plb, pub)(self.fz.isf)
        self.fz.ppf = make_conditional_ppf(self.sev_lb, self.sev_ub, plb, pub)(self.fz.ppf)
        self.fz.pdf = make_conditional_pdf(self.sev_lb, self.sev_ub, plb, pub)(self.fz.pdf)
        # Keep support() honest: the spliced distribution lives on [lb, ub].
        # Without this, splicing an *unbounded* base family (e.g. lognorm) leaves
        # fz.support() reporting the underlying (0, inf), so _bounded_severity_window
        # reads an infinite upper edge and round_bucket(inf) blows up. Value-bound
        # the bounds as default args; mirrors the reflect-shift support patch in
        # _apply_reflect (the method-swap pattern above). Splice runs before
        # _apply_reflect in _build, so the reflected support composes correctly.
        self.fz.support = lambda _lo=self.sev_lb, _hi=self.sev_ub: (_lo, _hi)

    def _apply_layer_attachment(self):
        """Build layered-loss wrappers from the (splice-only) ``self.fz`` methods.

        Notes
        -----
        Stores the wrappers as ``self._layered_<method>`` rather than
        mutating ``self.fz.<method>``. The :class:`Severity` scipy
        overrides ``_pdf`` / ``_cdf`` / ``_sf`` / ``_ppf`` / ``_isf``
        route through these; ``self.fz.<method>`` stays splice-only so
        the numerical-moments integration (:func:`_numerical_moms` and
        :func:`_moms_analytic`) sees the underlying
        spliced distribution, not the doubly-transformed claim-space view.

        This is the right division because there is no such thing as a
        "raw" severity *with a layer and attachment*: the splice (``sev_lb``
        / ``sev_ub``) is part of the severity distribution itself, while
        the layer/attachment (``exp_attachment`` / ``exp_limit``) is the
        policy applied on top of that distribution. ``self.fz`` represents
        the distribution; the layered wrappers represent the policy.

        Always runs, even for the trivial ``attachment=0, limit=inf`` case,
        because the layered-loss transform clamps ``x < 0 -> 0`` regardless
        of the layer parameters. Distributions whose support extends below
        zero (e.g. ``norm``) rely on this clamp.

        Closure captures ``attachment``, ``limit``, ``detachment``,
        ``pattach``, ``pdetach``, ``conditional`` at construction. Mutating
        any of those after ``__init__`` will produce stale results from the
        layered wrappers (though ``self.fz`` would still be self-consistent).
        """
        a, l, d = self.attachment, self.limit, self.detachment
        pa, pd = self.pattach, self.pdetach
        cond = self.conditional
        self._layered_cdf = make_layer_attachment_cdf(a, l, pa, cond)(self.fz.cdf)
        self._layered_sf  = make_layer_attachment_sf (a, l, pa, cond)(self.fz.sf)   # noqa
        self._layered_pdf = make_layer_attachment_pdf(a, l, d, pa, pd, cond)(self.fz.pdf)
        self._layered_isf = make_layer_attachment_isf(a, l, pa, pd, cond)(self.fz.isf)
        self._layered_ppf = make_layer_attachment_ppf(a, l, pa, pd, cond)(self.fz.ppf)

    def _apply_reflect(self):
        """Post-build for a reflected severity ``Y = shift - X``.

        ``_build`` constructs the positive base ``X = self.fz`` at loc 0 (scipy
        cannot carry a negative scale). This maps it to ``Y = shift - X`` --
        a *reflected*, shifted distribution with signed support -- by replacing
        the frozen-RV methods (the established ``_apply_lb_ub`` pattern) and
        setting the raw moments from those of ``X``:

        ``F_Y(y) = S_X(shift - y)``, ``S_Y(y) = F_X(shift - y)``,
        ``f_Y(y) = f_X(shift - y)``, ``q_Y(p) = shift - q_X(1-p)``,
        and support ``[shift - hi_X, shift - lo_X]``. Moments:
        ``E[Y^k] = E[(shift - X)^k]`` by binomial expansion.

        Used for ``-1 * dist (+/- shift)`` (e.g. ``100 - lognorm``). Reflection
        does **not** decide signedness: ``ssev 100 - lognorm`` keeps the negative
        support through ``_apply_signed``, while ``sev 100 - lognorm`` clamps it
        at zero through the ordinary layered-loss transform, exactly as
        ``sev 10 * norm + 5`` does. See dev/done/plan-reflected-loss-severity.md.

        The raw moments are set **only** on the signed path, where ``Y`` itself
        is the answer. Under plain ``sev`` the answer is the *clamped* law
        ``max(Y, 0)``, so populating ``sev1`` here would divert :meth:`moms`
        into its precomputed-moment fast path and return the unclamped value
        (``sev -lognorm 10 cv 0.5`` would report a mean of -10 for a severity
        that is identically 0). Leaving them ``None`` routes the clamped case
        through :func:`_numerical_moms`, which integrates the reflected ``isf``
        over the positive part and is correct by construction.
        """
        Z = self.fz
        d = float(self._reflect_shift)
        if self.signed:
            # raw moments first (before the method swap below)
            z1, z2, z3 = self._raw_moments(Z)
            self.sev1 = d - z1
            self.sev2 = d * d - 2 * d * z1 + z2
            self.sev3 = d ** 3 - 3 * d * d * z1 + 3 * d * z2 - z3
        # capture the originals, then install reflected versions on the frozen RV
        zcdf, zsf, zppf, zisf, zpdf = Z.cdf, Z.sf, Z.ppf, Z.isf, Z.pdf
        zlo, zhi = Z.support()
        self.fz.cdf = lambda y, _f=zsf, _d=d: _f(_d - np.asarray(y, dtype=float))
        self.fz.sf = lambda y, _f=zcdf, _d=d: _f(_d - np.asarray(y, dtype=float))
        self.fz.pdf = lambda y, _f=zpdf, _d=d: _f(_d - np.asarray(y, dtype=float))
        self.fz.ppf = lambda q, _f=zisf, _d=d: _d - _f(q)
        self.fz.isf = lambda q, _f=zppf, _d=d: _d - _f(q)
        self.fz.support = lambda _d=d, _lo=zlo, _hi=zhi: (_d - _hi, _d - _lo)
        # ``fz.stats`` is deliberately NOT swapped. Its one internal caller is
        # the finiteness check in ``_numerical_moms``, and finiteness is
        # invariant under reflection: E[(d - X)^k] is finite exactly when
        # E[X^j] is finite for every j <= k. A reflected ``stats`` would still
        # ignore any splice (``_apply_lb_ub`` does not swap it either), so
        # patching it would look fixed without being fixed.

    def _raw_moments(self, fz):
        """First three raw non-central moments of the (possibly spliced) ``fz``.

        Parameters
        ----------
        fz : frozen scipy RV
            The distribution whose moments are wanted, after
            :meth:`_apply_lb_ub` has installed any splice.

        Returns
        -------
        (m1, m2, m3) : tuple of float

        Notes
        -----
        Two paths. With no splice, ``fz.moment(k)`` is exact and closed form for
        the scipy families, so use it and change nothing.

        With a splice, ``fz.moment`` is **wrong**: :meth:`_apply_lb_ub` swaps
        ``cdf``, ``sf``, ``pdf``, ``ppf``, ``isf``, and ``support`` on the frozen
        RV but not ``moment``, which therefore still reports the unconditional
        law. That defect reached the answer, not just a diagnostic:
        ``ssev 10 - lognorm 1.5 splice [0 10]`` reported a severity mean of
        6.9198 (``10`` less the *unspliced* 3.0802) where the truth is 8.3115,
        which failed validation on a correct build and sized the automatic
        bucket window from the wrong moments.

        The spliced path integrates in quantile space,
        :math:`E[X^k] = \\int_0^1 q(p)^k dp` with :math:`q` the (patched)
        ``isf``, reusing the same :func:`_safe_integrate` machinery as
        :func:`_numerical_moms`. Quantile space rather than the x-axis because
        the interval is compact and the splice has already truncated the tail.
        A spliced law has no closed form to give up, so nothing is lost.
        """
        if self.sev_lb == 0 and self.sev_ub == np.inf:
            return (float(fz.moment(1)), float(fz.moment(2)),
                    float(fz.moment(3)))
        return tuple(
            _safe_integrate(
                (lambda p, _n=n: np.asarray(fz.isf(p), dtype=float) ** _n),
                0.0, 1.0, n, self.sev_name)[0]
            for n in (1, 2, 3))

    def _warn_reflected_clamp(self):
        """Warn when a non-signed reflected severity clamps mass at zero.

        Notes
        -----
        Called from ``__init__`` right after :meth:`_apply_reflect`, so
        ``self.fz`` is already the reflected law and its support is exact.
        Silent unless the reflected support genuinely reaches below zero: the
        common bounded case (``sev 10 - lognorm 1.5 splice [0 10]``, support
        ``[0, 10]``) clamps nothing and says nothing.

        The clamped mass is ``P(Y < 0) = F_Y(0)``, one cdf call on the reflected
        law. It goes in the message because it is the number that tells the user
        whether they care: a 0.01% tail and a wholesale truncation are different
        situations wearing the same warning.

        The lower-edge test uses ``VALIDATION_NOISE`` rather than a bare ``< 0``
        so a support edge that lands at ``-1e-16`` through floating-point
        arithmetic does not warn about nothing.
        """
        lo, hi = self.fz.support()
        if lo >= -VALIDATION_NOISE:
            return
        p0 = float(np.asarray(self.fz.cdf(0.0)).reshape(-1)[0])
        degenerate = (' The severity is a point mass at 0.'
                      if p0 >= 1 - VALIDATION_NOISE else '')
        warn_once(
            f"Severity '{self.sev_name}': reflected support "
            f"[{lo:g}, {hi:g}] reaches below 0, so 'sev' clamps "
            f"{100 * p0:.4g}% of the mass to 0.{degenerate} "
            f"Did you mean 'ssev'?",
            ReflectedSeverityClampWarning,
            key=('reflected-clamp', str(self.sev_name),
                 self._reflect_shift, self.sev_lb, self.sev_ub),
            stacklevel=3)

    def _apply_signed(self):
        """Post-build for a signed (never-clamp) severity: identity layering.

        A signed severity (``ssev`` keyword, or a discrete severity with
        negative atoms) is the raw distribution -- a profit is a negative loss.
        There is no attachment/limit clamp and no ``x<0 -> 0`` transform, so the
        layered methods are simply the (possibly spliced) ``self.fz`` methods,
        the attachment probabilities are trivial, and the moments are the raw
        distribution's. Discrete kinds already set ``sev1``/``sev2``/``sev3``
        from their support in ``_build``; for the continuous (scipy) case we
        read the non-central moments straight off ``self.fz``.

        Notes
        -----
        Occurrence reinsurance / layering on a signed severity is out of scope
        (see ``dev/plan-negative-x-agg.md`` §6); a signed severity is the raw
        unlayered distribution.
        """
        self.pattach = 1.0
        self.moment_pattach = 1.0
        self.pdetach = 0.0
        if self.sev1 is None:
            # continuous (scipy) signed severity: raw non-central moments.
            # ``_raw_moments`` because ``fz.moment`` ignores an active splice
            # (``_apply_lb_ub`` does not swap it); see its Notes.
            self.sev1, self.sev2, self.sev3 = self._raw_moments(self.fz)
        # identity layering: the severity IS the (spliced) raw distribution
        self._layered_cdf = self.fz.cdf
        self._layered_sf = self.fz.sf
        self._layered_pdf = self.fz.pdf
        self._layered_ppf = self.fz.ppf
        self._layered_isf = self.fz.isf

    def _compute_attachment_probs(self):
        """Compute ``pdetach``, ``pattach``, and ``moment_pattach``.

        Notes
        -----
        ``pdetach = P(X > detachment)`` always.

        When ``exp_attachment is None`` (no layer clause), the severity is
        conditioned on ``X >= 0``. Histogram-shaped kinds have no mass
        below zero so ``pattach`` and ``moment_pattach`` are both 1; for
        continuous distributions ``pattach`` is 1 (downstream uses) but
        ``moment_pattach = fz.sf(0)`` because the moment integrals in
        :func:`_numerical_moms` work in isf-space and need the actual
        survival probability at zero.

        Otherwise ``pattach = moment_pattach = fz.sf(attachment)``.
        """
        if self.detachment == np.inf:
            self.pdetach = 0
        else:
            self.pdetach = self.fz.sf(self.detachment)

        if self.exp_attachment is None:
            if self._is_histogram:
                self.moment_pattach = self.pattach = 1
            else:
                self.moment_pattach = self.fz.sf(self.attachment)
                self.pattach = 1
        else:
            self.moment_pattach = self.pattach = self.fz.sf(self.attachment)

    def _validate_moments(self):
        """Warn if achieved mean / cv differ materially from targets.

        Notes
        -----
        Only fires when the user supplied a positive ``sev_mean`` or
        ``sev_cv`` (otherwise we have no target to validate against).
        ``sev_loc`` is added to the target mean so that DecL forms like
        ``lognorm 5 cv .3 + 10`` (a shifted lognormal with the loc applied
        afterwards) compare correctly.

        A **reflected** severity is the exception: there ``sev_loc`` is the
        shift in ``shift - X``, not an additive location on the base, and the
        base was built at loc 0. Adding it would compare the user's target
        against ``target + shift`` and fail every time (``100 - lognorm 80
        cv .2`` would check 180 against an achieved 80). This runs *before*
        :meth:`_apply_reflect`, so ``self.fz`` is still the base ``X`` that the
        ``mean cv`` target actually describes; compare against it directly.
        """
        if not (self.sev_mean > 0 or self.sev_cv > 0):
            return
        loc = 0.0 if self.sev_reflect else self.sev_loc
        mean, var = self.fz.stats('mv')
        acv = var ** .5 / mean
        if self.sev_mean > 0 and not np.isclose(self.sev_mean + loc, mean):
            print(f'WARNING target mean {self.sev_mean} and achieved mean {mean} not close')
        if self.sev_cv > 0 and not np.isclose(
                self.sev_cv * self.sev_mean / (self.sev_mean + loc), acv):
            print(f'WARNING target cv {self.sev_cv} and achieved cv {acv} not close')
        logger.debug(
            f'Severity.__init__ | parameters {self.sev_a}, {self.sev_scale}: '
            f'target/actual {self.sev_mean} vs {mean};  {self.sev_cv} vs {acv}')

    # ``label`` comes from ``LabeledMixin`` (the shared label surface);
    # ``name`` stays the identity handle. See dev/plan-labels.md.

    def __repr__(self):
        """
        wrap default with name
        :return:
        """
        return f'{super(Severity, self).__repr__()} of type {self.sev_name}'

    def __enter__(self):
        """ Support with Severity as f: """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    # ------------------------------------------------------------------
    # scipy ``rv_continuous`` private overrides — pass-throughs to the
    # layered wrappers built by ``_apply_layer_attachment``.
    #
    # Splice (``sev_lb`` / ``sev_ub``) lives on ``self.fz.<method>``
    # (mutated by ``_apply_lb_ub``); layer/attachment lives on
    # ``self._layered_<method>``. The two transforms are kept on different
    # objects so internal moment integration (``_numerical_moms``,
    # :func:`_moms_analytic`) can see the splice-only distribution via
    # ``self.fz``, while user-facing scipy calls (``rvs``, ``interval``,
    # ``stats``, etc.) route through these overrides to the fully-layered
    # claim-space view.
    # ------------------------------------------------------------------

    def _pdf(self, x, *args):
        return self._layered_pdf(x)

    def _cdf(self, x, *args):
        return self._layered_cdf(x)

    def _sf(self, x, *args):
        return self._layered_sf(x)

    def _isf(self, q, *args):
        return self._layered_isf(q)

    def _ppf(self, q, *args):
        return self._layered_ppf(q)

    def _stats(self, *args, **kwds):
        """Mean, variance, skew of the layered severity (from ``moms``)."""
        ex1, ex2, ex3 = self.moms()
        var = ex2 - ex1 ** 2
        skew = (ex3 - 3 * ex1 * ex2 + 2 * ex1 ** 3) / var ** 1.5
        return np.array([ex1, var, skew, np.nan])

    @lru_cache
    def moms(self):
        """First three moments of the layered severity ``X(a, d) = min(d, (X-a)+)``.

        Notes
        -----
        Three paths in order:

        1. **Histogram fast-path.** If ``sev1`` was precomputed during
           ``_build`` (histograms and meta-severities populate ``sev1`` /
           ``sev2`` / ``sev3`` directly from the support) and the layer is
           the trivial ``[0, inf]``, return those values immediately.
        2. **Analytic shortcut.** For ``lognorm``, ``pareto``, ``gamma``,
           and ``expon`` (with no shift/truncation), delegate to
           :func:`_moms_analytic`, which computes layered
           moments in closed form via partial expected values. Defensive
           ``np.inf`` overrides are applied when the underlying moment does
           not formally exist (e.g. pareto shape <= 1).
        3. **Numerical fallback** :func:`_numerical_moms` integrates the
           survival quantile function over the layer interval and applies
           the attachment/detachment/conditional adjustments.

        Returns
        -------
        (E[X(a,d)], E[X(a,d)^2], E[X(a,d)^3]) : tuple of float
            ``np.nan`` signals an unreliable numerical result; ``np.inf``
            means the moment is genuinely undefined.

        Mathematical background
        -----------------------
        With :math:`X(a, d) = \\min(d, (X-a)_+)` and using the
        quantile-space change of variables :math:`x = q(p)`, :math:`f(x)dx = dp`,

        .. math::

            E[X(a, d)^n] = \\int_{F(a)}^{F(d)} (q(p) - a)^n\\, dp
                          + (d - a)^n S(d).

        The numerical path integrates :math:`\\int q(p)^n dp` and then
        applies the binomial expansion to recover :math:`E[(X-a)^n]`.
        """
        # 0. Signed (never-clamp) severity: the raw distribution's moments,
        # set in _apply_signed / _build (no layering on a signed severity).
        if self.signed:
            return self.sev1, self.sev2, self.sev3

        # 0b. Discrete (atomic) severity: every moment -- unlimited, limited,
        # or layered -- is an exact finite sum over the atoms, so never route a
        # discrete law through the numerical isf-integration path (path 3),
        # whose quadrature on a true step isf is both inexact and fragile. This
        # also fixes the unlimited case, which previously fell to path 3 (NOT
        # the path-1 fast-path, since _build truncates the implicit limit to
        # max(xs) so detachment is finite, never inf) and returned the
        # eps-trick's trailing-9s artifact. ``layer_moments`` reproduces the
        # exact value path 3 approximates; for the no-layer case limit=max(xs)
        # and pattach=1, so it returns the plain Σ xⁿ p. Signed discrete is
        # handled above (no layering on a signed law).
        if isinstance(self.fz, _DiscreteRV):
            denom = self.pattach if self.conditional else 1.0
            return self.fz.layer_moments(self.attachment, self.limit, denom)

        # 1. Histogram fast-path: precomputed moments cover the no-layer case.
        if (self.sev1 is not None
                and self.attachment == 0
                and self.detachment == np.inf):
            return self.sev1, self.sev2, self.sev3

        # 2. Closed-form via partial expected values for the supported kinds.
        # ``not self.sev_reflect`` is required, not defensive: ``sev -lognorm``
        # leaves sev_loc at 0 with the default splice, so it satisfies every
        # other clause here and would be handed the UNREFLECTED closed form.
        # A reflected law has no partial-expected-value shortcut, so it belongs
        # on the numerical path below.
        if (isinstance(self.sev_name, str)
                and self.sev_name in ('lognorm', 'pareto', 'gamma', 'expon')
                and not self.sev_reflect
                and self.sev_loc == 0
                and self.sev_lb == 0
                and self.sev_ub == np.inf):
            logger.info('Analytic moments')
            ma = _moms_analytic(self.fz, self.limit, self.attachment, 3)
            ex1a, ex2a, ex3a = ma[1:]
            # Defensive: when there is no upper limit, override with inf
            # for moments that the underlying RV does not have (e.g. pareto
            # shape <= 1 has no mean). ``moms_analytic`` usually already
            # returns inf for these cases via ``partial_e``; this guard
            # preserves the legacy belt-and-suspenders behaviour.
            if self.detachment == np.inf:
                mf = list(map(
                    lambda v: not (np.isinf(v) or np.isnan(v)),
                    self.fz.stats('mvs')))
                if not mf[0]: ex1a = np.inf
                if not mf[1]: ex2a = np.inf
                if not mf[2]: ex3a = np.inf
            if self.conditional:
                ex1a /= self.pattach
                ex2a /= self.pattach
                ex3a /= self.pattach
            return ex1a, ex2a, ex3a

        # 3. Numerical fallback for everything else (scipy zoo minus the
        #    four analytic specials, plus any histogram with a layer).
        return _numerical_moms(self)

    def plot(self, n=None, log=False, full_range=False, reflect=False,
             return_period=False, invert=False):
        """Plot the severity: its density, and its quantile (Lee) diagram.

        Parameters
        ----------
        n : int, optional
            Grid points. The grid is quantile spaced, so this buys
            resolution in probability rather than in loss.
        log : bool
            Read both axes of the density panel on log, which is the log
            density panel this plot used to draw as a third picture.
        full_range : bool
            Show the whole computed grid, out to the 1-in-100,000 loss,
            rather than the crop at the 0.1% exceedance.
        reflect : bool
            Read the Lee panel against the exceeding probability rather
            than the non-exceeding one, so the curve drawn is the survival
            function. With ``invert`` it is ``S(x)`` the usual way round,
            and that axis offers a log reading where the non-exceeding one
            does not.
        return_period : bool
            Read the Lee panel against return period rather than
            non-exceedance probability.
        invert : bool
            Exchange the Lee panel's axes, which draws the distribution
            function: a quantile function and a cdf are inverses, so it is
            the same pairs read the other way round.

        Returns
        -------
        matplotlib.figure.Figure
            Also stashed on ``self.figure``.

        Notes
        -----
        Draws the document ``charts.chart_severity`` emits. Four panels
        became two, and neither collapse loses anything. The density and
        the log density were one quantity read two ways, so log is a
        declared reading of the one panel. The distribution and the Lee
        diagram are **inverses**, the same curve with its axes exchanged,
        so only one of them carries information the other does not; the Lee
        orientation is kept because it is the one that pairs with a
        return-period reading.

        A severity is the one genuinely continuous law in this library, so
        its series says ``support='continuous'`` and the renderer draws a
        line rather than steps. A discrete severity has no density at all,
        so it is drawn as probability mass and is atomic.
        """
        from .charts import build_chart_doc
        from .plots import plot_chartdoc
        options = {} if n is None else {'n': n}
        self.figure = plot_chartdoc(
            build_chart_doc(self, 'severity', **options),
            log=log, full_range=full_range, reflect=reflect,
            return_period=return_period, invert=invert)
        return self.figure


# ---------------------------------------------------------------------------
# Severity concrete subclasses (Stage 1d).
# ---------------------------------------------------------------------------


class SeverityScipy(Severity):
    """Severity backed by a named ``scipy.stats`` continuous distribution.

    Notes
    -----
    Shape-parameter count is determined by introspecting
    ``getattr(ss, sev_name).shapes`` rather than from a hardcoded table, so
    new scipy distributions are picked up automatically. ``cv_to_shape``
    has analytic shortcuts for ``lognorm``, ``gamma``, ``invgamma``, and
    ``invgauss`` via ``aggregate.utilities``; ``beta`` mean/cv -> shape
    parameters go through ``utilities.beta_fit``.
    """
    sev_kind = 'scipy'

    def _build(self):
        sev_name = self.sev_name
        gen = getattr(ss, sev_name)
        shapes_spec = gen.shapes
        n_shapes = 0 if shapes_spec is None else shapes_spec.count(',') + 1

        sev_mean = self.sev_mean
        sev_cv = self.sev_cv
        sev_a = self.sev_a
        sev_b = self.sev_b
        sev_loc = self.sev_loc
        sev_scale = self.sev_scale

        # Reflected severity: build the POSITIVE base at loc 0; the shift and
        # negation are applied by ``_apply_reflect`` (Y = shift - X). The shift
        # was captured as ``self._reflect_shift`` before this point.
        if self.sev_reflect:
            sev_loc = 0.0

        if n_shapes == 0:
            if sev_loc == 0 and sev_mean > 0:
                sev_loc = sev_mean
            if sev_scale == 0 and sev_cv > 0:
                sev_scale = sev_cv * sev_loc
            self.fz = gen(loc=sev_loc, scale=sev_scale)
            # Reflect any derived values back onto self so callers can
            # introspect the final parameters.
            self.sev_loc = sev_loc
            self.sev_scale = sev_scale

        elif n_shapes == 2:
            # ``beta`` is the only common 2-shape distribution that accepts
            # mean/cv inputs analytically (via ``beta_fit``); other 2-shape
            # distributions require explicit ``sev_a``/``sev_b``.
            if sev_name == 'beta' and sev_mean > 0 and sev_cv > 0:
                m = sev_mean / sev_scale
                sev_a, sev_b = beta_fit(m, sev_cv)
                self.fz = ss.beta(sev_a, sev_b, loc=0, scale=sev_scale)
            else:
                self.fz = gen(sev_a, sev_b, loc=sev_loc, scale=sev_scale)
            self.sev_a = sev_a
            self.sev_b = sev_b

        elif n_shapes == 1:
            if np.isnan(sev_a) and sev_cv > 0:
                sev_a, _ = _cv_to_shape(sev_name, sev_cv)
                logger.info(
                    f'sev_a not set, determined as {sev_a} shape from sev_cv {sev_cv}')
            elif np.isnan(sev_a):
                raise ValueError(
                    'sev_a not set and sev_cv=0 is invalid, no way to determine shape.')

            if sev_mean > 0:
                logger.info('creating with sev_mean=%s and sev_loc=%s', sev_mean, sev_loc)
                sev_scale, self.fz = _mean_to_scale(sev_name, sev_a, sev_mean, sev_loc)
            elif sev_scale > 0 and sev_mean == 0:
                logger.info('creating with sev_scale=%s and sev_loc=%s', sev_scale, sev_loc)
                self.fz = gen(sev_a, scale=sev_scale, loc=sev_loc)
            else:
                raise ValueError('sev_scale and sev_mean both equal zero.')
            self.sev_a = sev_a
            self.sev_scale = sev_scale

        else:
            raise ValueError(
                f'scipy distribution {sev_name!r} has unexpected shape spec '
                f'{shapes_spec!r}; expected 0, 1, or 2 shape parameters.')


def _broadcast_histogram_xs_ps(sev_name, sev_xs, sev_ps):
    """Broadcast histogram xs / ps inputs into matched arrays.

    Notes
    -----
    Continuous histograms (``chistogram``) allow ``xs`` and ``ps`` to have
    different lengths (``xs`` includes the right-hand bucket end); the
    broadcast attempt is suppressed for that case so the raw arrays pass
    through unchanged. For ``dhistogram`` and ``fixed`` the broadcast
    succeeds and produces aligned arrays.
    """
    try:
        xs, ps = np.broadcast_arrays(np.array(sev_xs), np.array(sev_ps))
    except ValueError:
        if sev_name != 'chistogram':
            logger.warning(
                f'Severity._build | {sev_name} sev_xs and sev_ps cannot be broadcast.')
        xs = np.array(sev_xs)
        ps = np.array(sev_ps)
    if not np.isclose(np.sum(ps), 1.0):
        logger.error(
            f'Severity._build | {sev_name} histogram severity with probs do not '
            f'sum to 1, {np.sum(ps)}')
    return xs, ps


class _DiscreteRV:
    """Frozen discrete distribution over arbitrary float support.

    A small, honest stand-in for the subset of the ``scipy.stats`` frozen
    random-variable interface that :class:`Severity` consumes, with genuine
    step-function semantics. It replaces the old ``rv_histogram`` "epsilon
    jump" hack (a *continuous* object forced to mimic a step function by
    pouring each atom's mass into a ``2**-d``-wide sliver to its left), which
    made correctness hinge on a float-resolution tightrope and produced
    ``ppf``/``isf`` artifacts like ``49.999999999`` instead of ``50``.

    Parameters
    ----------
    xs : array_like
        Support points (atoms); arbitrary finite floats, need not be sorted,
        may be negative (a profit is a negative loss).
    ps : array_like
        Probabilities at each atom; should sum to 1.

    Notes
    -----
    ``cdf`` is the exact right-continuous step function ``P(X <= x)``; ``sf``
    is its complement; ``pdf`` is ``0`` everywhere (a discrete law has no
    density). ``ppf(q)`` returns the smallest atom with ``cdf >= q``. Moments
    are exact finite sums ``Σ xⁿ pₙ`` -- see :meth:`layer_moments` for the
    layered/limited case, which :meth:`Severity.moms` uses so a discrete
    severity never routes its moments through numerical integration.

    Scalar and array inputs are both supported (the splice / layer / signed
    decorators in :class:`Severity` pass 0-d and 1-d arrays): a scalar in
    yields a numpy scalar out, an array in yields an array of the same shape.
    """

    def __init__(self, xs, ps):
        xs = np.asarray(xs, dtype=float).ravel()
        ps = np.asarray(ps, dtype=float).ravel()
        order = np.argsort(xs)
        self.xk = xs[order]
        self.pk = ps[order]
        self.cum = np.cumsum(self.pk)        # P(X <= xk_i)
        self.n = self.xk.size
        # Share the spacing-agnostic cumulative-step kernel: the cdf / sf / mean
        # / lower-quantile core is GridDistribution's, this class adds only the
        # scipy-naming + severity-specific bits (pdf == 0, moments, layered
        # moments, rvs). bs is None -- a genuine discrete law has no density.
        self._gd = GridDistribution(self.xk, self.pk)

    @staticmethod
    def _unwrap(out):
        """Return a numpy scalar for 0-d results, else the array unchanged."""
        out = np.asarray(out)
        return out[()] if out.ndim == 0 else out

    def cdf(self, x):
        """Right-continuous CDF ``P(X <= x)`` (delegates to the shared core)."""
        return self._gd.cdf(x)

    def sf(self, x):
        """Survival function ``P(X > x) = 1 - cdf(x)`` (shared core)."""
        return self._gd.sf(x)

    def pdf(self, x):
        """Density of a discrete law: identically ``0`` (mass lives in atoms)."""
        out = np.zeros_like(np.asarray(x, dtype=float))
        return self._unwrap(out)

    def ppf(self, q):
        """Quantile: the smallest atom ``xk`` with ``cdf(xk) >= q`` (shared lower-q)."""
        return self._unwrap(self._gd.q(q, 'lower'))

    def isf(self, q):
        """Inverse survival: ``ppf(1 - q)``."""
        return self.ppf(1.0 - np.asarray(q, dtype=float))

    def support(self):
        """``(min atom, max atom)``."""
        return self.xk[0], self.xk[-1]

    def moment(self, n):
        """Exact raw moment ``E[Xⁿ] = Σ xⁿ p``."""
        return float(np.sum(self.xk ** n * self.pk))

    def stats(self, moments='mv'):
        """Exact ``(mean, var[, skew[, kurtosis]])`` selected by ``moments``."""
        m = float(np.sum(self.xk * self.pk))
        v = float(np.sum((self.xk - m) ** 2 * self.pk))
        out = []
        for ch in moments:
            if ch == 'm':
                out.append(m)
            elif ch == 'v':
                out.append(v)
            elif ch == 's':
                out.append(float(np.sum((self.xk - m) ** 3 * self.pk) / v ** 1.5)
                           if v > 0 else 0.0)
            elif ch == 'k':
                out.append(float(np.sum((self.xk - m) ** 4 * self.pk) / v ** 2 - 3.0)
                           if v > 0 else 0.0)
        return tuple(out)

    def mean(self):
        """Exact mean ``E[X] = Σ x p`` (shared core)."""
        return self._gd.mean()

    def var(self):
        m = self.mean()
        return float(np.sum((self.xk - m) ** 2 * self.pk))

    def rvs(self, size=None, random_state=None):
        """Draw exact atoms (no ``atom - U(0, 2**-d)`` jitter, unlike the hack)."""
        if random_state is None or isinstance(random_state, (int, np.integer)):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        return rng.choice(self.xk, size=size, p=self.pk)

    def layer_moments(self, attachment, limit, denom):
        """Exact first three moments of the layered loss ``min(limit, (X-a)₊)``.

        Parameters
        ----------
        attachment : float
            Layer attachment ``a``.
        limit : float
            Layer width (``np.inf`` for an unlimited layer / no cap).
        denom : float
            Divisor applied to every moment -- ``P(X > a)`` for a conditional
            severity, ``1.0`` otherwise.

        Returns
        -------
        (E[Y], E[Y²], E[Y³]) : tuple of float
            where ``Y = min(limit, (X - attachment)₊)``.

        Notes
        -----
        This is the exact closed form of the integral that
        :func:`_numerical_moms` approximates by quadrature: for a discrete
        law ``∫_{F(a)}^{F(d)} (q(p)-a)ⁿ dp + (d-a)ⁿ S(d) = Σᵢ yᵢⁿ pᵢ``. A zero
        ``denom`` (layer entirely above the support ⇒ ``P(X>a)=0``, every
        ``yᵢ=0``) returns zeros rather than ``0/0``.
        """
        y = np.clip(self.xk - attachment, 0.0, limit)
        m1 = float(np.sum(y * self.pk))
        m2 = float(np.sum(y * y * self.pk))
        m3 = float(np.sum(y * y * y * self.pk))
        if denom == 0:
            return 0.0, 0.0, 0.0
        return m1 / denom, m2 / denom, m3 / denom


class SeverityDHistogram(Severity):
    """Severity with point-mass support at user-supplied loss values.

    Notes
    -----
    Pre-computes ``sev1`` / ``sev2`` / ``sev3`` directly from ``(xs, ps)``.
    ``self.fz`` is a :class:`_DiscreteRV` -- an honest frozen discrete RV with
    exact right-continuous step ``cdf``/``sf`` and exact summed moments -- in
    place of the historical ``rv_histogram`` epsilon-jump hack (which abused a
    continuous object to mimic a step function and produced ``ppf``/``isf``
    artifacts and a float-resolution dependency via ``max_log2``).
    """
    sev_kind = 'dhistogram'
    _is_histogram = True

    def _build(self):
        xs, ps = _broadcast_histogram_xs_ps(self.sev_name, self.sev_xs, self.sev_ps)
        # Truncate the implicit limit at the max support point if the
        # user did not provide a tighter one.
        self.limit = min(self.limit, xs.max())
        self.detachment = self.limit + self.attachment
        # Validate then compute raw moments from the cleaned (xs, ps).
        # ``allow_negative=True`` preserves negative atoms (a profit is a
        # negative loss); the aggregate auto-detects signed support from this.
        xs, ps = validate_discrete_distribution(xs, ps, allow_negative=True)
        # A discrete severity with a negative atom is signed (a profit is a
        # negative loss) -- the only sensible reading -- regardless of the
        # constructor flag. This is what lets ``dsev [-2 5]`` auto-sign.
        if np.any(xs < 0):
            self.signed = True
        # keep the (validated, sorted) atoms for exact-support sizing and
        # for the info display (rendered by ``_support_phrase``).
        self.support_atoms = np.asarray(xs, dtype=float)
        self.sev1 = np.sum(xs * ps)
        self.sev2 = np.sum(xs ** 2 * ps)
        self.sev3 = np.sum(xs ** 3 * ps)
        # Honest discrete RV: exact step cdf/sf at the FFT bucket edges (which
        # sit at half-bucket offsets and never coincide with an atom, so the
        # discretised density is bit-identical to the old eps-jump trick) plus
        # exact ppf/isf/support and exact moments (see _DiscreteRV).
        self.fz = _DiscreteRV(xs, ps)


class SeverityCHistogram(Severity):
    """Severity with a continuous (piecewise-uniform) histogram density.

    Notes
    -----
    The user supplies bucket boundary ``xs`` and per-bucket probabilities
    ``ps``. If ``xs`` and ``ps`` have the same length, the right-hand end
    of the last bucket is synthesized as ``xs[-1] + xs[-2]``. Bucket heights
    are ``ps / diff(xs)`` so the resulting density is properly normalised
    regardless of whether ``ps`` sums to 1.
    """
    sev_kind = 'chistogram'
    _is_histogram = True

    def _build(self):
        xs, ps = _broadcast_histogram_xs_ps(self.sev_name, self.sev_xs, self.sev_ps)
        self.limit = min(self.limit, xs.max())
        if len(xs) == len(ps):
            xss = np.sort(np.hstack((xs, xs[-1] + xs[-2])))
        else:
            xss = xs
        aps = ps / np.diff(xss)
        # The synthesised right-end may push the support beyond the
        # user-supplied limit; widen accordingly.
        self.limit = min(self.limit, xss.max())
        self.detachment = self.limit + self.attachment
        xsm = (xss[:-1] + xss[1:]) / 2
        self.sev1 = np.sum(xsm * ps)
        self.sev2 = np.sum(xsm ** 2 * ps)
        self.sev3 = np.sum(xsm ** 3 * ps)
        # density=True is explicit, not a change: ``aps`` is already a density
        # (probability per unit width) and scipy assumes exactly that when
        # ``density`` is left None. Stating it silences scipy's "Bin widths
        # are not constant" RuntimeWarning, which fires for every unequally
        # spaced ``sev_xs`` -- which is the whole point of chistogram.
        self.fz = ss.rv_histogram((aps, xss), density=True)


class SeverityFixed(SeverityDHistogram):
    """Severity concentrated at a single loss value.

    Notes
    -----
    A thin specialization of :class:`SeverityDHistogram`: when the user
    writes ``fixed`` they typically pass only ``sev_xs``; this subclass
    fills in ``sev_ps = np.array(1)`` and delegates. Kept as a distinct
    class so the registry / DecL parser surface reflects the user-visible
    kind ``fixed``.
    """
    sev_kind = 'fixed'

    def _build(self):
        if self.sev_ps is None:
            self.sev_ps = np.array(1)
        super()._build()


class SeverityMeta(Severity):
    """Severity built from the output distribution of an Aggregate or Portfolio.

    Notes
    -----
    Reuses an existing aggregate-level distribution as a severity. The
    ``sev_a`` and ``sev_b`` spec slots are repurposed here as ``log2`` and
    ``bs`` — if the source has not yet been computed at those resolutions
    (or has been computed at different ones) it is updated in place. This
    side effect is preserved verbatim from the pre-refactor behaviour;
    flagged for review in Stage 2.

    The result is a hybrid discrete/continuous histogram: a tiny bucket
    pins the probability mass at zero, while the rest of the support is
    treated as a continuous histogram with bucket size ``bs``.
    """
    sev_kind = 'meta'
    # Built as ``ss.rv_histogram`` over the source's density, so behaves
    # like the other histogram kinds for moments / fast-paths.
    _is_histogram = True

    def _build(self):
        # Local imports to avoid distributions <-> portfolio / _aggregate cycles.
        from .portfolio import Portfolio
        from ._aggregate import Aggregate

        source = self.sev_name
        log2 = self.sev_a
        bs = self.sev_b

        if isinstance(source, Aggregate):
            if log2 and (log2 != source.log2 or (bs != source.bs and bs != 0)):
                source.easy_update(log2, bs)
            xs = source.xs
            ps = source.agg_density
        elif isinstance(source, Portfolio):
            if log2 and (log2 != source.log2 or (bs != source.bs and bs != 0)):
                source.update(log2, bs, add_exa=False)
            xs = source.density_df.loss.values
            ps = source.density_df.p_total.values
        else:
            raise ValueError(
                f'Object {source} passed as a proto-severity type but only '
                f'Aggregate, Portfolio and Severity objects allowed')

        # Construct a hybrid discrete/continuous histogram. A tiny bucket
        # holds the mass at zero; the rest is continuous-uniform between
        # bucket midpoints offset by bs/2.
        b1size = 1e-7
        xss = np.hstack((-bs * b1size, 0, xs[1:] - bs / 2, xs[-1] + bs / 2))
        pss = np.hstack((ps[0] / b1size, 0, ps[1:]))
        # density=True is explicit, not a change: it is what scipy assumes
        # when ``density`` is left None. The bins here are bs*1e-7, then
        # bs/2, then bs, so they are NEVER constant and every meta severity
        # tripped scipy's "Bin widths are not constant" RuntimeWarning.
        self.fz = ss.rv_histogram((pss, xss), density=True)
        self.sev1 = np.sum(xs * ps)
        self.sev2 = np.sum(xs ** 2 * ps)
        self.sev3 = np.sum(xs ** 3 * ps)


class SeverityCopy(Severity):
    """Severity that adopts another ``Severity`` instance as its underlying RV.

    Notes
    -----
    The short-circuit ``Severity(other_severity, ...)`` path. The pre-refactor
    code stored the other ``Severity`` directly as ``self.fz``; that is
    preserved verbatim here. Layer/attachment and conditioning behaviour
    still come from the new instance's own settings.
    """
    sev_kind = 'copy'

    def _build(self):
        self.fz = self.sev_name
