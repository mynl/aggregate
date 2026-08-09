"""Method-of-moments severity fits (leaf: numpy/scipy only).

Extracted from ``distributions.py`` (Phase 1, kind split). Imported through the
``distributions`` facade so every existing import path keeps working.
"""

import logging
import warnings
import numpy as np
import scipy.stats as ss
from scipy.optimize import NoConvergence  # noqa
from .moments import (xsden_to_meancv)

logger = logging.getLogger(__name__)

__all__ = [
    'lognorm_fit',
    'sln_fit',
    'sgamma_fit',
    'gamma_fit',
    'beta_fit',
    'invgamma_fit',
    'invgauss_fit',
    'lognorm_lev',
    'lognorm_approx',
    'approximate_from_mcvsk',
]


# ---------------------------------------------------------------------------
# Method-of-moments fitting cluster.
#
# Public, symmetric ``*_fit`` family that recovers distribution parameters
# from ``(m, cv[, skew])``. Used to seed approximations (e.g. ``approximate``
# methods, severity initialisation) and to keep moment-matching exhibits in
# one place. ``approximate_from_mcvsk`` dispatches over the family.
# ---------------------------------------------------------------------------


def lognorm_fit(m, cv):
    """
    Lognormal ``(mu, sigma)`` parameters from mean ``m`` and cv ``cv``.

    Notes
    -----
    For ``ss.lognorm(sigma, scale=exp(mu))`` matching mean ``m`` and CV ``cv``,
    :math:`\\sigma^2 = \\log(1 + \\mathrm{cv}^2)` and
    :math:`\\mu = \\log(m) - \\sigma^2 / 2`.
    """
    cv = np.array(cv)
    m = np.array(m)
    sigma = np.sqrt(np.log(cv*cv + 1))
    mu = np.log(m) - sigma**2 / 2
    return mu, sigma


def sln_fit(m, cv, skew):
    """
    method of moments shifted lognormal fit matching given mean, cv and skewness

    :param m:
    :param cv:
    :param skew:
    :return:
    """
    if skew == 0:
        return -np.inf, np.inf, 0
    else:
        eta = (((np.sqrt(skew ** 2 + 4)) / 2) + (skew / 2)) ** (1 / 3) - (
                1 / (((np.sqrt(skew ** 2 + 4)) / 2) + (skew / 2)) ** (1 / 3))
        sigma = np.sqrt(np.log(1 + eta ** 2))
        shift = m - cv * m / eta
        if shift > m:
            logger.warning(f'sln_fit | shift > m, {shift} > {m}, too extreme skew {skew}')
            shift = m - 1e-6
        mu = np.log(m - shift) - sigma ** 2 / 2
        return shift, mu, sigma


def sgamma_fit(m, cv, skew):
    """
    method of moments shifted gamma fit matching given mean, cv and skewness

    :param m:
    :param cv:
    :param skew:
    :return:
    """
    if skew == 0:
        return np.nan, np.inf, 0
    else:
        alpha = 4 / (skew * skew)
        theta = cv * m * skew / 2
        shift = m - alpha * theta
        return shift, alpha, theta


def gamma_fit(m, cv):
    """
    gamma parameters from mean and cv.
    """
    alpha = cv**-2
    beta = m / alpha
    return alpha, beta


def beta_fit(m, cv):
    """
    alpha and beta parameters from mean and cv.

    """
    v = m * m * cv * cv
    sev_a = m * (m * (1 - m) / v - 1)
    sev_b = (1 - m) * (m * (1 - m) / v - 1)
    return sev_a, sev_b


def invgamma_fit(cv):
    """
    Inverse gamma shape parameter from cv.

    Notes
    -----
    For ``ss.invgamma(a)`` the squared coefficient of variation satisfies
    :math:`\\mathrm{cv}^2 = 1 / (a - 2)`, giving :math:`a = 1/\\mathrm{cv}^2 + 2`.
    Valid for :math:`a > 2`, i.e. when the variance exists.
    """
    return 1 / cv ** 2 + 2


def invgauss_fit(cv):
    """
    Inverse Gaussian shape parameter from cv.

    Notes
    -----
    For ``ss.invgauss(mu)`` the cv equals :math:`\\sqrt{\\mu}`, so
    :math:`\\mu = \\mathrm{cv}^2`.
    """
    return cv ** 2


def lognorm_lev(mu, sigma, n, limit):
    """
    return E(min(X, limit)^n) for lognormal using exact calculation
    currently only for n=1, 2

    :param mu:
    :param sigma:
    :param n:
    :param limit:
    :return:
    """
    if limit == -1:
        return np.exp(n * mu + n * n * sigma * sigma / 2)
    else:
        phi = ss.norm.cdf
        ll = np.log(limit)
        sigma2 = sigma * sigma
        phi_l = phi((ll - mu) / sigma)
        phi_l2 = phi((ll - mu - n * sigma2) / sigma)
        unlimited = np.exp(n * mu + n * n * sigma2 / 2)
        return unlimited * phi_l2 + limit ** n * (1 - phi_l)


def lognorm_approx(ser):
    """
    Lognormal approximation to series, index = loss values, values = density.
    """
    m, cv = xsden_to_meancv(ser.index, ser.values)
    mu, sigma = lognorm_fit(m, cv)
    fz = ss.lognorm(sigma, scale=np.exp(mu))
    return fz


def approximate_from_mcvsk(m, cv, skew, name, agg_str, note, approx_type, output,
                           warn_degenerate=False):
    """Dispatch ``(m, cv, skew)`` to a method-of-moments fit, in the requested form.

    Backs ``Aggregate.approximate`` and ``Portfolio.approximate``. A thin
    **output adapter** over the single fit core :func:`_approximate_sev_kwargs`,
    which owns the family fits *and* the guards (symmetric -> normal, left-skew
    -> reflected); this function only formats the core's ``sev_*`` kwargs into
    the requested surface, so the two ``approximate`` surfaces and the
    ``approximate`` DecL keyword share one implementation.

    Parameters
    ----------
    m, cv, skew : float
        Analytic (or empirical) aggregate mean, coefficient of variation, and
        skewness.
    name, agg_str, note : str
        Naming / note scaffolding for the DecL and Aggregate output forms.
    approx_type : {'norm', 'lognorm', 'gamma', 'sgamma', 'slognorm'}
        The fitted family. ``'all'`` is handled by the calling method, not here.
    output : str
        ``'scipy'`` -> frozen ``scipy.stats`` rv; ``'sev_kwargs'`` -> the
        ``Severity`` kwargs dict; ``'sev_decl'`` -> a DecL severity fragment;
        ``'agg_decl'`` -> a full ``agg ... fixed`` DecL program; any other string
        -> a fixed-frequency :class:`Aggregate`.
    warn_degenerate : bool
        When ``True`` (the interactive ``.approximate()`` method) emit a
        ``UserWarning`` if an explicitly-requested *shifted* family
        (``slognorm`` / ``sgamma``) degenerates to a normal because the
        distribution is symmetric.

    Returns
    -------
    object
        Per ``output``.

    Notes
    -----
    A left-skewed (reflected) shifted fit has no native frozen ``scipy`` or
    one-line DecL representation, so ``output='scipy'`` / ``'sev_decl'`` /
    ``'agg_decl'`` raise ``ValueError`` for it -- use ``output='sev_kwargs'`` or
    the default Aggregate object (both carry ``sev_reflect``), or
    ``approx_type='norm'``.
    """
    sev = _approximate_sev_kwargs(m, cv, skew, approx_type,
                                  warn_degenerate=warn_degenerate)
    reflected = bool(sev.get('sev_reflect', False))

    if output == 'scipy':
        if reflected:
            raise ValueError(
                f"approx_type={approx_type!r} for this left-skewed distribution "
                f"(skew={skew:.3g}) is a reflected fit with no native frozen scipy "
                f"representation; use output='sev_kwargs' or the default Aggregate "
                f"object (both reflect), or approx_type='norm'.")
        return _sev_kwargs_to_scipy(sev)
    if output == 'sev_kwargs':
        return sev
    if output in ('sev_decl', 'agg_decl'):
        decl = _sev_kwargs_to_decl(sev, reflected, approx_type, skew)
        return decl if output == 'sev_decl' else f'{agg_str}{decl} fixed'
    # any other string -> a fixed-frequency Aggregate carrying the fitted sev.
    # Local import: _fits is a leaf; Aggregate lives downstream in _aggregate.
    from ._aggregate import Aggregate
    return Aggregate(**{'name': name, 'note': note, 'exp_en': 1, **sev,
                        'freq_name': 'fixed'})


def _sev_kwargs_to_scipy(sev):
    """Frozen ``scipy.stats`` rv from method-of-moments ``sev_*`` kwargs.

    Only the non-reflected families -- the caller rejects a reflected fit, which
    has no native frozen scipy object.
    """
    nm = sev['sev_name']
    loc = float(sev.get('sev_loc', 0.0))
    scale = float(sev['sev_scale'])
    if nm == 'norm':
        return ss.norm(loc=loc, scale=scale)
    if nm == 'lognorm':
        return ss.lognorm(float(sev['sev_a']), loc=loc, scale=scale)
    if nm == 'gamma':
        return ss.gamma(float(sev['sev_a']), loc=loc, scale=scale)
    raise ValueError(f'cannot build a scipy rv for sev_name={nm!r}')


def _sev_kwargs_to_decl(sev, reflected, approx_type, skew):
    """DecL severity fragment from method-of-moments ``sev_*`` kwargs.

    A reflected (left-skew) fit has no clean one-line DecL form, so it raises --
    use ``output='sev_kwargs'`` or the default Aggregate object instead.
    """
    if reflected:
        raise ValueError(
            f"approx_type={approx_type!r} for this left-skewed distribution "
            f"(skew={skew:.3g}) is a reflected fit with no one-line DecL form; "
            f"use output='sev_kwargs' or the default Aggregate object, or "
            f"approx_type='norm'.")
    nm = sev['sev_name']
    scale = sev['sev_scale']
    loc = sev.get('sev_loc')
    if nm == 'norm':
        return f'{scale} @ norm 1 # {loc} '
    frag = f'{scale} * {nm} {sev["sev_a"]} '
    if loc not in (None, 0, 0.0):
        frag += f'+ {loc} '
    return frag


def _approximate_sev_kwargs(m, cv, skew, approx_type, warn_degenerate=False):
    """Severity kwargs for a method-of-moments approximation -- the single fit core.

    The **one** place the family fits and their guards live. Given the
    aggregate's first three moments, return the ``sev_*`` keyword dict for a
    single continuous severity whose moments match, for any of the five
    families: the unshifted ``norm`` / ``lognorm`` / ``gamma``
    (skew-independent) and the shifted ``sgamma`` / ``slognorm``. Both the
    ``approximate`` DecL keyword / constructor and the ``Aggregate`` /
    ``Portfolio`` ``approximate()`` methods (via :func:`approximate_from_mcvsk`)
    consume it, so there is no second implementation to drift.

    The shifted fits are defined only for **positive** skew, so:

    - **(near-)symmetric** (``|skew|`` below tolerance): both shifted fits'
      common limit is a **normal**, returned instead. When ``warn_degenerate``
      and a shifted family was explicitly requested, a ``UserWarning`` is
      emitted (the interactive ``.approximate()`` method sets this; the
      declarative DecL/constructor path leaves it ``False`` and degrades
      silently).
    - **left (negative) skew**: fit the *reflected* aggregate ``-A`` (which is
      right-skewed) and map back through the ``sev_reflect`` machinery
      (``Y = sev_loc - X`` with the base built at loc 0); the returned dict
      carries ``sev_reflect=True``, plus ``sev_signed=True`` on the same
      low-quantile test the other branches use. The flag is explicit because
      reflection stopped implying signedness at 1.0.0a230: a reflected severity
      under plain ``sev`` clamps at zero, and a fit that clamps would not
      reproduce the moments it was built to match.

    Parameters
    ----------
    m, cv, skew : float
        Analytic (or empirical) aggregate mean, coefficient of variation, and
        skewness. ``m`` may be negative; the helper is sign-agnostic via
        ``sd = m * cv``.
    approx_type : {'norm', 'lognorm', 'gamma', 'sgamma', 'slognorm'}
        The fitted family.
    warn_degenerate : bool
        Emit a ``UserWarning`` when a shifted family degenerates to a normal
        (symmetric input). Default ``False``.

    Returns
    -------
    dict
        ``sev_*`` keyword arguments for :class:`Severity` / the
        :class:`Aggregate` constructor: ``sev_name``, ``sev_a`` (where
        applicable), ``sev_scale``, ``sev_loc`` (shifted families only); plus
        ``sev_reflect=True`` for the left-skew case and ``sev_signed=True``
        whenever the fit carries mass at or below zero (so the loss-layering
        ``x<0 -> 0`` clamp is bypassed and the fit is represented exactly).

    Notes
    -----
    The standard deviation is ``sd = m * cv`` (exact, since ``cv = sd / m``),
    correct for either sign of ``m``. The reflected fit reuses :func:`sln_fit`
    / :func:`sgamma_fit` unchanged -- the reflection is handled here, not inside
    those fitters, so the freeze-checked windowing code that also calls them is
    unaffected.
    """
    valid = ('norm', 'lognorm', 'gamma', 'sgamma', 'slognorm')
    if approx_type not in valid:
        raise ValueError(
            f"approximate kind {approx_type!r} must be one of {', '.join(valid)}")
    m = float(m)
    cv = float(cv)
    skew = float(skew)
    sd = m * cv
    skew_tol = 1e-3
    # Mark the fitted severity signed (no ``x<0 -> 0`` clamp) only when it
    # actually places mass below zero. The deciding test is the low quantile,
    # NOT the shift/loc: a shifted gamma can have a very negative ``loc`` yet all
    # its mass on the positive axis (the ``loc`` is just the parameterisation),
    # in which case the ordinary 0-based grid is correct and cheaper.
    signed_tail = 1e-8

    def _mark_signed(sev, fz):
        if float(fz.ppf(signed_tail)) < 0:
            sev['sev_signed'] = True
        return sev

    def _mark_signed_reflected(sev, fz_base):
        """The same test for a reflected fit ``Y = sev_loc - X``.

        ``q_Y(p) = sev_loc - q_X(1 - p)``, so the low quantile of ``Y`` is the
        shift less the *upper* quantile of the base. Reflection does not decide
        signedness on its own (a reflected severity under plain ``sev`` clamps
        at zero like any other), so a reflected fit has to declare it the same
        way every other branch here does. In practice it nearly always is
        signed: the base families are unbounded above, so ``Y`` reaches well
        below zero unless the shift is enormous.
        """
        if float(sev['sev_loc'] - fz_base.isf(signed_tail)) < 0:
            sev['sev_signed'] = True
        return sev

    # ---- unshifted, skew-independent families -------------------------------
    if approx_type == 'norm':
        return _mark_signed({'sev_name': 'norm', 'sev_scale': sd, 'sev_loc': m},
                            ss.norm(loc=m, scale=sd))
    if approx_type == 'lognorm':
        mu, sigma = lognorm_fit(m, cv)
        return {'sev_name': 'lognorm', 'sev_a': sigma,
                'sev_scale': float(np.exp(mu))}
    if approx_type == 'gamma':
        shape, scale = gamma_fit(m, cv)
        return {'sev_name': 'gamma', 'sev_a': shape, 'sev_scale': scale}

    # ---- shifted families: need positive skew -------------------------------
    if abs(skew) <= skew_tol:
        # Symmetric limit of both shifted fits is a normal.
        if warn_degenerate:
            warnings.warn(
                f"approx_type={approx_type!r}: the distribution is symmetric "
                f"(skew={skew:.3g}); the shifted fit degenerates to its normal "
                f"limit, returned instead. Pass approx_type='norm' to select "
                f"the normal explicitly.", UserWarning, stacklevel=3)
        return _mark_signed({'sev_name': 'norm', 'sev_scale': sd, 'sev_loc': m},
                            ss.norm(loc=m, scale=sd))
    if skew > 0:
        if approx_type == 'sgamma':
            shift, alpha, theta = sgamma_fit(m, cv, skew)
            sev = {'sev_name': 'gamma', 'sev_a': alpha, 'sev_scale': theta,
                   'sev_loc': shift}
            fz = ss.gamma(alpha, loc=shift, scale=theta)
        else:
            shift, mu, sigma = sln_fit(m, cv, skew)
            sev = {'sev_name': 'lognorm', 'sev_a': sigma,
                   'sev_scale': float(np.exp(mu)), 'sev_loc': shift}
            fz = ss.lognorm(sigma, loc=shift, scale=float(np.exp(mu)))
        return _mark_signed(sev, fz)
    # Left (negative) skew: fit the reflected aggregate -A (right-skewed), then
    # map back via Y = sev_loc - X with sev_loc = -shift_R and X built at loc 0.
    m_r = -m
    cv_r = sd / m_r          # = -cv; the product cv_r * m_r recovers sd
    skew_r = -skew
    if approx_type == 'sgamma':
        shift_r, alpha, theta = sgamma_fit(m_r, cv_r, skew_r)
        return _mark_signed_reflected(
            {'sev_name': 'gamma', 'sev_a': alpha, 'sev_scale': theta,
             'sev_loc': -shift_r, 'sev_reflect': True},
            ss.gamma(alpha, scale=theta))
    shift_r, mu, sigma = sln_fit(m_r, cv_r, skew_r)
    return _mark_signed_reflected(
        {'sev_name': 'lognorm', 'sev_a': sigma, 'sev_scale': float(np.exp(mu)),
         'sev_loc': -shift_r, 'sev_reflect': True},
        ss.lognorm(sigma, scale=float(np.exp(mu))))
