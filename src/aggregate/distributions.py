from collections import namedtuple
from collections.abc import Iterable
from functools import lru_cache, wraps
import json
import inspect
import logging
import math
import warnings
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
import pandas as pd
import scipy.fft as sfft
from scipy.integrate import quad
import scipy.stats as ss
from scipy import interpolate
from scipy.optimize import newton
from scipy.special import kv, gammaln, hyp1f1, loggamma, binom
from scipy.optimize import broyden2, newton_krylov, brentq
from scipy.optimize import NoConvergence  # noqa
from textwrap import fill

from .constants import (ALIASING_RATIO, DefectiveDistributionWarning,
                        FIG_H, FIG_W,
                        REINS_LABEL_GROSS, REINS_LABEL_NET,
                        REINS_LABEL_CEDED, REINS_LABEL_OUTPUT,
                        Validation)
from .config import get_settings
from .moments import (MomentAggregator, MomentWrangler,
                      xsden_to_mwrangler,
                      xsden_to_meancv, xsden_to_meancvskew,
                      _noise_aware_rel_error, _snap_noise)

__all__ = [
    'Frequency', 'Severity', 'Aggregate',
    'lognorm_fit', 'sln_fit', 'sgamma_fit', 'gamma_fit', 'beta_fit',
    'invgamma_fit', 'invgauss_fit',
    'lognorm_lev', 'lognorm_approx',
    'approximate_from_mcvsk',
]
from .utilities import (ft, ift,
                        round_bucket,
                        nice_multiple,
                        decl_pprint, make_var_tvar,
                        agg_help, explain_validation, remove_fuzz)
import aggregate.random_agg as ar
from .spectral import Distortion
from .pentagon import complete_pentagon, Pentagon
from . import tail as _tail
# Re-export the bounded tables for back-compat; tail.py is the source of truth.
# (Unused here, but tests/external code import them via this module -- see
# tests/test_tail.py; keep the noqa so ruff doesn't strip the re-export.)
from .tail import _BOUNDED_FREQS, _BOUNDED_SCIPY_SEVS, TailClass  # noqa: F401

logger = logging.getLogger(__name__)

# Config-backed module constants, resolved once per session from
# aggregate.config (see that module for the cascade and reload semantics).
#
# WINDOW_NINES: probability coverage for the automatic output WINDOW (number of
# nines): the window spans roughly the 10**-WINDOW_NINES .. 1-10**-WINDOW_NINES
# quantiles. Deliberately far tighter than BUCKET_SIZING_P (the bucket-sizing
# percentile) so a P&L / signed aggregate's window captures essentially all the
# mass.
WINDOW_NINES = get_settings().discretization.window_nines
# BUCKET_SIZING_P: percentile of the fitted distribution fed to
# recommend_bucket to size bs (formerly BUCKET_SIZING_P). >1 is read as nines.
BUCKET_SIZING_P = get_settings().discretization.bucket_sizing_p
# VALIDATION_NOISE: absolute dust floor below which a quantity is treated as
# exact zero / numerical noise.
VALIDATION_NOISE = get_settings().validation.noise


def max_log2(x):
    """
    Return the largest power of two d so that (x + 2**-d) - x == 2**-d, with d <= 30.
    Used in dhistogram severity types to determine the size of the step.
    """
    d = min(30, -np.log2(np.finfo(float).eps) - np.ceil(np.log2(x)) - 1)
    if (x + 2 ** -d) - x != 2 ** -d:
        raise ValueError('max_log2 failed')
    return d


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


def approximate_from_mcvsk(m, cv, skew, name, agg_str, note, approx_type, output):
    """
    Dispatch from ``(m, cv, skew)`` to a method-of-moments approximation and
    return it in the requested form. Backs ``Aggregate.approximate`` and
    ``Portfolio.approximate``; see their documentation.

    :param output: scipy - frozen scipy.stats continuous rv object; agg_decl
      sev_decl - DecL program for severity (to substituate into an agg ; no name)
      sev_kwargs - dictionary of parameters to create Severity
      agg_decl - Decl program agg T 1 claim sev_decl fixed
      any other string - created Aggregate object
    """
    if approx_type == 'norm':
        sd = m*cv
        if output == 'scipy':
            return ss.norm(loc=m, scale=sd)
        sev = {'sev_name': 'norm', 'sev_scale': sd, 'sev_loc': m}
        decl = f'{sd} @ norm 1 # {m} '

    elif approx_type == 'lognorm':
        mu, sigma = lognorm_fit(m, cv)
        sev = {'sev_name': 'lognorm', 'sev_a': sigma, 'sev_scale': np.exp(mu)}
        if output == 'scipy':
            return ss.lognorm(sigma, scale=np.exp(mu))
        decl = f'{np.exp(mu)} * lognorm {sigma} '

    elif approx_type == 'gamma':
        shape, scale = gamma_fit(m, cv)
        if output == 'scipy':
            return ss.gamma(shape, scale=scale)
        sev = {'sev_name': 'gamma', 'sev_a': shape, 'sev_scale': scale}
        decl = f'{scale} * gamma {shape} '

    elif approx_type == 'slognorm':
        shift, mu, sigma = sln_fit(m, cv, skew)
        if output == 'scipy':
            return ss.lognorm(sigma, scale=np.exp(mu), loc=shift)
        sev = {'sev_name': 'lognorm', 'sev_a': sigma, 'sev_scale': np.exp(mu), 'sev_loc': shift}
        decl = f'{np.exp(mu)} * lognorm {sigma} + {shift} '

    elif approx_type == 'sgamma':
        shift, alpha, theta = sgamma_fit(m, cv, skew)
        if output == 'scipy':
            return ss.gamma(alpha, loc=shift, scale=theta)
        sev = {'sev_name': 'gamma', 'sev_a': alpha, 'sev_scale': theta, 'sev_loc': shift}
        decl = f'{theta} * gamma {alpha} + {shift} '

    else:
        raise ValueError(f'Inadmissible approx_type {approx_type} passed to fit')

    if output == 'agg_decl':
        agg_str += decl
        agg_str += ' fixed'
        return agg_str
    elif output == 'sev_kwargs':
        return sev
    elif output == 'sev_decl':
        return decl
    else:
        return Aggregate(**{'name': name, 'note': note,
                            'exp_en': 1, **sev, 'freq_name': 'fixed'})


def _estimate_agg_percentile(m, cv, skew, p=0.999):
    """
    Come up with an estimate of the tail of the distribution based on the three parameter fits, ln and gamma

    Updated Nov 2022 with a way to estimate p based on lognormal results. How far in the
    tail you need to go to get an accurate estimate of the mean. See 2_x_approximation_error
    in the help.

    Retain p param for backwards compatibility.

    :param m:
    :param cv:
    :param skew:
    :param p: if > 1 converted to 1 - 10**-n
    :return:
    """

    # p_estimator = interp1d([0.53294, 0.86894, 1.9418, 7.3211, 22.738, 90.012, 457.14, 2981],
    #                        [3,       4,       5,      7,      8,      9,      11,     12],
    #                        assume_sorted=True, bounds_error=False, fill_value=(3, 13))
    # p = 1 - 10**-p_estimator(cv)

    if np.isinf(cv):
        raise ValueError('Infinite variance passed to estimate_agg_percentile')

    # make vectorizable
    p = np.array(p)
    p = np.where(p > 1, 1 - 10 ** -p, p)

    pn = pl = pg = 0
    if skew <= 0:
        # neither sln nor sgamma works, use a normal
        # for negative skewness the right tail will be thin anyway so normal not outrageous
        fzn = ss.norm(scale=m * cv, loc=m)
        pn = fzn.isf(1 - p)
    elif not np.isinf(skew):
        shift, mu, sigma = sln_fit(m, cv, skew)
        fzl = ss.lognorm(sigma, scale=np.exp(mu), loc=shift)
        shift, alpha, theta = sgamma_fit(m, cv, skew)
        fzg = ss.gamma(alpha, scale=theta, loc=shift)
        pl = fzl.isf(1 - p)
        pg = fzg.isf(1 - p)
    else:
        mu, sigma = lognorm_fit(m, cv)
        fzl = ss.lognorm(sigma, scale=np.exp(mu))
        alpha, theta = gamma_fit(m, cv)
        fzg = ss.gamma(alpha, scale=theta)
        pl = fzl.isf(1 - p)
        pg = fzg.isf(1 - p)
    # throw in a mean + 3 sd approx too...
    return np.maximum(np.maximum(pn, pl), np.maximum(pg, m * (1 + ss.norm.isf(1 - p) * cv)))


def estimate_agg_window(m, sd, skew, p=BUCKET_SIZING_P):
    """Two-sided output window ``[x_lo, x_hi]`` and width ``W`` for an aggregate.

    The signed counterpart of :func:`_estimate_agg_percentile`: where that
    returns a single upper bound on a non-negative aggregate, this returns
    *both* edges from method-of-moments fits, so a profit/loss aggregate (mean
    possibly < 0, mass possibly far from 0) can be placed on a tight window
    where its mass actually lives. Used by ``Aggregate.update(x_min=None)``.

    Parameters
    ----------
    m, sd, skew : float
        Analytic aggregate mean, **standard deviation** (not cv), and skewness.
        ``m`` may be negative or zero (a net-profit / mean-zero P&L); ``sd`` is
        taken directly so the mean-zero case -- where ``cv`` is undefined --
        works.
    p : float
        Coverage. ``p > 1`` is read as ``1 - 10**-p`` (e.g. ``p=6`` ->
        ``1 - 1e-6``); the per-edge tail probability is ``1 - p``.

    Returns
    -------
    (x_lo, x_hi, W) : tuple of float
        Lower edge, upper edge, and width ``W = x_hi - x_lo``. The window is
        **not** forced to be centred on the mean -- for a skewed aggregate the
        two quantiles place it off-centre.

    Notes
    -----
    Three regimes (mirrors ``_estimate_agg_percentile`` but two-sided):

    - **Symmetric** (``|skew|`` below a tolerance, incl. any genuinely
      symmetric or mean-zero P&L): neither shifted-lognormal nor shifted-gamma
      is defined, so use a **normal** approximation for both edges
      ``m -/+ z*sd`` with ``z = norm.isf(1-p)``. This is the default fallback,
      not an afterthought.
    - **Right-skewed** (``skew > 0``): fit shifted lognormal and shifted gamma
      to ``(m, sd, skew)`` and take the **wider** window
      (``min`` of the low quantiles, ``max`` of the high quantiles).
    - **Left-skewed** (``skew < 0``): **reflect** -- fit to ``-A`` (mean
      ``-m``, ``skew -skew > 0``), then map the window back as
      ``[-hi_r, -lo_r]``.

    Falls back to the normal window if the shifted fits fail (e.g. mean ~ 0
    with non-trivial skew). The returned window is widened, if necessary, to
    contain ``m``.
    """
    sd = abs(float(sd))
    if not np.isfinite(sd):
        raise ValueError('Infinite/undefined sd passed to estimate_agg_window')
    p = float(np.where(p > 1, 1 - 10.0 ** -p, p))
    tail = 1.0 - p
    z = ss.norm.isf(tail)
    if sd == 0:
        return float(m), float(m), 0.0

    def _normal_window():
        return float(m - z * sd), float(m + z * sd), float(2 * z * sd)

    skew_tol = 1e-3
    if abs(skew) <= skew_tol or not np.isfinite(skew):
        return _normal_window()

    # Reflect for negative skew so the fits always see a right tail.
    s = 1.0 if skew > 0 else -1.0
    mm = s * m
    sk = s * skew
    cvv = sd / mm if mm != 0 else np.inf
    los, his = [], []
    if np.isfinite(cvv):
        try:
            shift, mu, sigma = sln_fit(mm, cvv, sk)
            fz = ss.lognorm(sigma, scale=np.exp(mu), loc=shift)
            los.append(float(fz.ppf(tail)))
            his.append(float(fz.isf(tail)))
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            shift, alpha, theta = sgamma_fit(mm, cvv, sk)
            fz = ss.gamma(alpha, scale=theta, loc=shift)
            los.append(float(fz.ppf(tail)))
            his.append(float(fz.isf(tail)))
        except Exception:  # pragma: no cover - defensive
            pass
    if not his:
        return _normal_window()
    lo_r, hi_r = min(los), max(his)
    if s == 1.0:
        x_lo, x_hi = lo_r, hi_r
    else:  # undo reflection: window of A is the mirror of the window of -A
        x_lo, x_hi = -hi_r, -lo_r
    # ensure the window contains the mean
    x_lo = min(x_lo, m)
    x_hi = max(x_hi, m)
    return float(x_lo), float(x_hi), float(x_hi - x_lo)


# ---------------------------------------------------------------------------
# Single-module helpers — used only inside distributions.py.
# ---------------------------------------------------------------------------


def _partial_e_numeric(fz, a, n):
    """
    Simple numerical integration version of partial_e for auditing purposes.

    """
    ans = []
    for k in range(n+1):
        temp = quad(lambda x: x ** k * fz.pdf(x), 0, a)
        if temp[1] > 1e-4:
            logger.warning('Potential convergence issues with numerical integral')
        ans.append(temp[0])
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
        if λ + loc != 0:
            logger.warning('Pareto not shifted to x>0 range...using numeric moments.')
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

    ans1 = np.array([sum([(-1) ** (m - k) * binom(m, k) * attachment ** (m - k) * (pe_detach[k] - pe_attach[k])
                          for k in range(m + 1)])
                     for m in range(n + 1)])

    if np.isinf(limit):
        ans2 = np.zeros_like(ans1)
    else:
        ans2 = np.array([limit ** m * fz.sf(detachment) for m in range(n+1)])

    ans = ans1 + ans2

    return ans


def _picks_work(attachments, layer_loss_picks, xs, sev_density, n=1, sf=None, debug=False):
    """
    Adjust the layer unconditional expected losses to target. You need int xf(x)dx, but
    that is fraught when f is a mixed distribution. So we only use the int S version.
    ``fz`` was initially a frozen continuous distribution; but adjusted to sf function
    and dropped need for pdf function.

    See notes for how the parts are defined. Notice that::

        np.allclose(p.layers.v - p.layers.f, p.layers.l - p.layers.e)

    is true.

    :param attachments: array of layer attachment points, in ascending order (bottom to top). a[0]>0
    :param layer_loss_picks: Target means. If ``len(layer_loss_picks)==len(attachments)`` then the bottom layer, 0 to a[0],
      is added. Can be input as unconditional layer severity (i.e., :math:`\\mathbb{E}[(X-a)^+\\wedge y]`) or as the
      layer loss pick (i.e., :math:`\\mathbb{E}[(X-a)^+\\wedge y]'times n` where *n* is the number of ground-up (to the
      insurer) claims. Multiplying and dividing by :math:`S(a)` shows this equals conditional severity in the layer
      times the number of claims in the layer.) Actuaries usually estimate the loss pick to the layer in pricing. When
      called from :class:`Aggregate` the number of ground up claims is known.
    :param en: ground-up expected claims. Target is divided by ``en``.
    :param xs: x values for discretization
    :param sev_density: Series of existing severity density from Aggregate.
    :param sf: cdf function for the severity distribution.
    :param debug: if True, return debug information (layers, density with adjusted probs, audit
      of layer expected values.
    """

    # want xs, attachments, and sev_density to be numpy arrays
    xs = np.array(xs)
    attachments = np.array(attachments)
    # target is the unconditional layer expected loss, E[(X-a)^+ ^ y]
    target = np.array(layer_loss_picks) / n
    # print(n, layer_loss_picks, target)
    sev_density = np.array(sev_density)
    # figure bucket size
    bs = xs[1] - xs[0]

    # dataframe of adjusted probabilties, starts here
    density = pd.DataFrame({'x': xs, 'p': sev_density}).set_index('x', drop=False)
    fill_value = max(0, 1. - density.p.sum())
    density['S'] = density.p.shift(-1, fill_value=fill_value)[::-1].cumsum()

    # numerical integrals - these match
    layers = pd.DataFrame(columns=['a', 'lev', 'int_fdx', 'aS', 'S'], index=range(1, 1+len(attachments)),
                          dtype=float)
    for i, x in enumerate(attachments):
        ix = density.loc[0:x-bs, 'S'].sum() * bs
        ix2 = density.loc[0:x-bs, ['x', 'p']].prod(axis=1).sum()
        layers.loc[i+1, :] = [x, ix, ix2, x * density.loc[x, 'S'] if x < np.inf else 0.0, density.loc[x, 'S']]

    # prob of loss in layer
    layers['p'] = layers.S.shift(1, fill_value=1) - layers.S
    # unconditional expected loss in layer
    layers['l'] = layers.lev - layers.lev.shift(1, fill_value=0)
    layers.index.name = 'layer'
    # bottom of layer
    layers['a_bottom'] = layers.a.shift(1, fill_value=0)
    # width of layer
    layers['y'] = layers.a - layers.a_bottom
    # e = rectangle to right in int S computation
    layers['e'] = layers.S * layers.y
    # f = rectangle below attachment in int xf computation
    layers['f'] = layers.p * layers.a_bottom
    # these are two versions of m (unconditional)
    # m-bit: int S - e == int xf - f
    layers['m'] = layers.l - layers.e
    # int f dx in layer
    layers['v'] = layers.f + layers.m
    # and conditional vertical loss in layer
    layers['v_c'] = layers.v / layers.p
    layers = layers[['a_bottom', 'a', 'y', 'lev', 'S', 'p', 'l', 'v', 'v_c', 'm', 'e', 'f']]

    # add weights w and offsets=omega, computed from the top layer down
    layers['t'] = target
    layers['w'] = 0.0
    layers['ω'] = 0.0

    # this computation leaves the tail unchanged and uses the same "adjust the curve" method
    # in all layers
    ω = layers.loc[len(layers), 'S']
    for i in layers.index[::-1]:
        layers.loc[i, 'w'] = (layers.loc[i, 't'] - ω * layers.loc[i, 'y']) / layers.loc[i, 'm']
        layers.loc[i, 'ω'] = ω
        ω += layers.loc[i, 'p'] * layers.loc[i, 'w']

    # adjusted S: bins -> layer number; add in offsets
    density['bin'] = pd.cut(density.x, np.hstack((0, layers.a.values)), include_lowest=True, right=True)
    # layer description returned by cut to layer number in layers
    mapper = {i:j+1 for j, i in enumerate(density.bin.unique())}
    density['layer'] = density.bin.map(mapper.get)

    density['ω'] = density.layer.map(layers.ω).astype(float)
    # S(a_n-1)
    density['Sa'] = density.layer.map(layers.S).astype(float)
    density['w'] = density.layer.map(layers.w).astype(float)

    density['S_adj'] = np.minimum(1, density.ω + (density.S - density.Sa) * density.w)
    # no change in the tail
    density.loc[attachments[-1]:, 'S_adj'] = density.loc[attachments[-1]:, 'S']
    # adj probs as difference of S
    density['p_adj'] = density['S_adj'].shift(1, fill_value=1) - density['S_adj']
    achieved = density.groupby(density.layer.shift(-1)).apply(lambda g: g['S_adj'].sum() * bs)
    # display(achieved)
    if abs(achieved.iloc[0] - target[0]) > 1e-3:
        # issues with hitting 1
        logger.warning(f'achieved[0] = {achieved.iloc[0]} != target[0] = {target[0]}')
        # take top right corner off
        if target[0] > attachments[0]:
            raise ValueError(f'target[0] = {target[0]} > first attachment[0] = {attachments[0]} which is impossible.')
        s0 = 2 * (attachments[0] - target[0]) / (1 - layers.loc[1, 'ω'])
        # snap to index
        s0 = bs * np.round(s0 / bs, 0)
        # convert to probability
        s = attachments[0] - s0
        density.loc[0:s, 'S_adj'] = 1.0
        temp = np.array(density.loc[s+bs:attachments[0]].index)
        wts = (temp - s) / s0
        density.loc[s+bs:attachments[0], 'S_adj'] = 1 - wts + layers.loc[1, 'ω'] * wts
        # update
        density['p_adj'] = density['S_adj'].shift(1, fill_value=1) - density['S_adj']
        achieved = density.groupby(density.layer.shift(-1)).apply(lambda g: g['S_adj'].sum() * bs)
        logger.warning(f'Revised layer 1 achieved = {achieved.iloc[0]}')

    density['diff S'] = density['S'] - density['Sa']

    if debug is False:
        return density['p_adj'].values

    # data frame of layer statistics from input density
    exact = None
    if sf is not None:
        logger.warning('sf passed in; computing exact layer statistics')
        exact = pd.DataFrame(columns=['a', 'lev', 'aS', 'S'],
                             index=range(1, 1+len(attachments)), dtype=float)
        for i, x in enumerate(attachments):
            ix = quad(sf, 0, x)
            # check error is small
            assert ix[1] < 1e-6
            sf_ = sf(x)
            exact.loc[i+1, :] = [x, ix[0], x * sf_ if x < np.inf else 0.0, sf_]

    if exact is None:
        l = layers.l
        ln = 'layers'
    else:
        l = exact.lev - exact.lev.shift(1, fill_value=0)
        ln = 'exact'

    t = pd.concat((l,
                   density.groupby(density.layer.shift(-1)).apply(lambda g: g['S'].sum() * bs),
                   achieved,
                   ), keys=[ln, 'computed', 'adj'], axis=1)
    t.loc['sum'] = t.sum()
    Picks = namedtuple('picks', ['layers', 'exact', 'density', 'audit'])
    return Picks(layers=layers, exact=exact, density=density, audit=t)


def _integral_by_doubling(func, x0, err=1e-8):
    r"""
    Compute :math:`\int_{x_0}^\infty f` as the sum

    .. math::

        \int_{x_0}^\infty f = \sum_{n \ge 0} \int_{2^nx_0}^{2^{n+1}x_0} f

    Caller should check the integral actually converges.

    :param func: function to be integrated.
    :param x0: starting x value
    :param err: desired accuracy: stop when incremental integral is <= err.
    """
    ans = 0.
    counter = 0
    # from to
    f, t = x0, 2 * x0
    last_int = 10
    while last_int > err:
        s = quad(func, f, t)
        if s[1] > err:
            raise ValueError(
                f'Questionable integral numeric convergence, err {s[1]:.4g}\n'
                f'f={f}, t={t}, x0={x0}, counter={counter}')
        last_int = s[0]
        ans += s[0]
        f, t = t, 2 * t
        counter += 1
        if counter > 96:
            raise ValueError(f'counter = {counter} and error = {err}')
    return -ans


@lru_cache(maxsize=128)
def _logarithmic_theta(mean):
    """
    Solve for theta parameter given mean, see JKK p. 288
    """
    f = lambda x: x / (-np.log(1 - x) * (1 - x)) - mean
    theta = brentq(f, 1e-10, 1-1e-10)
    if not np.allclose(mean, theta / (-np.log(1 - theta) * (1 - theta))):
        print('num method failed')
    else:
        return theta


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


def _normalize_freq_name(freq_name):
    """
    Map user-supplied frequency names to registry keys.

    Handles the synonym groups ``'neyman' | 'neymana' | 'neymanA'`` and the
    ``'sichel.gamma'`` / ``'sichel.ig'`` dotted forms. Unknown names pass
    through unchanged.
    """
    if freq_name in ('neyman', 'neymanA'):
        return 'neymana'
    return freq_name


def _validate_reins_layers(reins_list, tol=1e-9):
    """
    Validate that a reinsurance program is entered bottom-up and non-overlapping.

    ``make_ceder_netter`` builds the ceder by walking layers in list order and
    tracking a running height; out-of-order or overlapping layers silently
    produce wrong cessions. Enforce the contract with a hard error here, at the
    single choke point used by every reinsurance path.

    Parameters
    ----------
    reins_list : list of (share, limit, attach)
        Layer tuples; ``limit`` may be ``np.inf`` for an unlimited top layer.
    tol : float
        Tolerance for the non-overlap comparison (absorbs float noise).

    Raises
    ------
    ValueError
        If attachments are not non-decreasing, or two layers overlap.

    Notes
    -----
    Rules over ``[(share, limit, attach), ...]``:

    - attachments must be **non-decreasing**;
    - layers must **not overlap**: ``attach_{i+1} >= attach_i + limit_i - tol``.

    Gaps between layers are allowed -- represent a gap with a zero-share layer
    ``0 po L xs A``. Zero-share entries still must respect ordering. Single-layer
    programs are trivially valid.
    """
    prev_attach = None
    prev_top = None
    for (share, limit, attach) in reins_list:
        if prev_attach is not None:
            if attach < prev_attach - tol:
                raise ValueError(
                    'Reinsurance layers must be entered bottom-up (lowest '
                    f'attachment first); got attachment {attach} after '
                    f'{prev_attach}. Enter layers in ascending order and use '
                    '"0 po L xs A" to represent a gap.')
            if attach < prev_top - tol:
                raise ValueError(
                    f'Reinsurance layers overlap: layer attaching at {attach} '
                    f'starts below the top ({prev_top}) of the preceding layer. '
                    'Layers must be non-overlapping; use "0 po L xs A" to '
                    'represent a gap.')
        prev_attach = attach
        prev_top = attach + (0 if np.isinf(limit) else limit)


def make_ceder_netter(reins_list, debug=False):
    """
    Build the netter and ceder functions. It is applied to occ_reins and agg_reins,
    so should be stand-alone.

    The reinsurance functions are piecewise linear functions from 0 to inf with
    kinks as needed to express the ceded loss as a function of subject (gross) loss.

    The entries in ``reins_list`` are tuples (share of, limit, attach) where share of is the
    percentage share, between 0 and 1.

    For example, if ``reins_list = [(1, 10, 0), (0.5, 30, 20)]`` the program is 10 x 10 and
    15 part of 30 x 20 (share=0.5). This requires nodes at 0, 10, 20, 50, and inf.

    It is easiest to make the ceder function. Ceded loss at subject loss at x equals
    the sum of the limits below x plus the cession to the layer in which x lies. The
    variable ``base`` keeps track of the layer, ``h`` of the sum (height) of lower layers.
    ``xs`` tracks the knot points, ``ys`` the values.

    ::

         Break (xs)   Ceded (ys)
              0            0
             10            0
             20           10
             50           25
            inf           25


    For example:
    ::

        %%sf 1 2

        c, n, x, y = make_ceder_netter([(1, 10, 10), (0.5, 30, 20), (.25, np.inf, 50)], debug=True)

        xs = np.linspace(0,250, 251)
        ys = c(xs)

        ax0.plot(xs, ys)
        ax0.plot(xs, xs, ':C7')
        ax0.set(title='ceded')

        ax1.plot(xs, xs-ys)
        ax1.plot(xs, xs, 'C7:')
        ax1.set(title='net')

    :param reins_list: a list of (share of, limit, attach), e.g., (0.5, 3, 2) means 50% share of 3x2
        or, equivalently, 1.5 part of 3 x 2. It is better to store share rather than part
        because it still works if limit == inf.
    :param debug: if True, return layer function xs and ys in addition to the interpolation functions.
    :return: netter and ceder functions; optionally debug information.
    """
    # hard error on out-of-order / overlapping layers (single choke point)
    _validate_reins_layers(reins_list)
    # poor mans inf
    INF = 1e99
    h = 0
    base = 0
    xs = [0]
    ys = [0]
    for (share, y, a) in reins_list:
        # part of = share of times limit
        if np.isinf(y):
            y = INF
        p = share * y
        if a > base:
            # moved to new layer, write out left-hand knot point
            xs.append(a)
            ys.append(h)
        # increment height
        h += p
        # write out right-hand knot points
        xs.append(a + y)
        ys.append(h)
        # update left-hand end
        base += (a + y)
    # if not at infinity, stay flat from base to end
    if base < INF:
        xs.append(np.inf)
        ys.append(h)
    ceder = interpolate.interp1d(xs, ys)
    netter = lambda x: x - ceder(x)
    if debug:
        return ceder, netter, xs, ys
    else:
        return ceder, netter


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

    # Registry of subclasses keyed by ``freq_name`` (registry key). Populated
    # by ``__init_subclass__`` as each ``Frequency<Kind>`` subclass is imported.
    _registry: dict = {}

    # Subclass contract — overridden on each ``Frequency<Kind>``:
    #   freq_name: registry key (e.g. 'poisson'). Empty on the base class.
    #   supports_zm: True iff the subclass defines ``prn_eq_0`` and supports
    #     zero modification.
    #   prn_eq_0: class-level default ``None``; ZM subclasses override with
    #     a method that returns P(N = 0 | mean = n).
    freq_name = ''
    supports_zm = False
    prn_eq_0 = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only register concrete kinds — subclasses that don't set a
        # class-level ``freq_name`` (e.g. ``Aggregate`` during the transition
        # period) are not part of the registry.
        key = cls.__dict__.get('freq_name', '')
        if key:
            Frequency._registry[key] = cls

    def __new__(cls, freq_name=None, *args, **kwargs):
        """
        Factory dispatch: ``Frequency('poisson', ...)`` → ``FrequencyPoisson``.

        When called on a subclass directly or with ``freq_name=None``, no
        dispatch occurs. Direct calls on ``Frequency`` route to the
        registered subclass for the (normalized) ``freq_name``; if no
        subclass is registered, fall back to the base class so the legacy
        if/elif body in ``__init__`` can handle it during the migration.
        """
        if cls is not Frequency or freq_name is None:
            return object.__new__(cls)
        lookup = _normalize_freq_name(freq_name)
        subclass = cls._registry.get(lookup)
        if subclass is None:
            return object.__new__(cls)
        return object.__new__(subclass)

    def _solve_n_base(self, n):
        """
        Solve for ``n_base`` such that the ZM-adjusted mean equals ``n``.

        The ZM construction reweights an unmodified distribution of mean
        ``n_base`` so that its probability at 0 becomes ``freq_p0``. The
        new mean is ``(1 - freq_p0) n_base / (1 - prn_eq_0(n_base))``;
        invert that to recover ``n_base``.
        """
        p0 = self.freq_p0
        f = lambda x: (1 - p0) * x / (1 - self.prn_eq_0(x)) - n
        if p0 < self.prn_eq_0(n):
            # ZM has higher mean than the unmodified distribution: search left of n.
            n_base = brentq(f, a=0, b=n)
        else:
            # search right
            n_base = brentq(f, a=n, b=n / (1 - p0))
        self.unmodified_mean = n_base
        return n_base

    def _install_zm_wrappers(self):
        """
        Replace ``self.freq_moms`` and ``self.freq_pgf`` with ZM-adjusted
        versions. The wrapped form is a weighted average of the trivial
        (point-mass at 0) component and the original distribution.
        """
        orig_moms = self.freq_moms
        orig_pgf = self.freq_pgf
        freq_p0 = self.freq_p0

        @wraps(orig_moms)
        def wrapped_moms(n):
            n_base = self._solve_n_base(n)
            ans = np.array(orig_moms(n_base))
            return (1 - freq_p0) / (1 - self.prn_eq_0(n_base)) * ans

        @wraps(orig_pgf)
        def wrapped_pgf(n, z):
            n_base = self._solve_n_base(n)
            wt = (1 - freq_p0) / (1 - self.prn_eq_0(n_base))
            return (1 - wt) + wt * orig_pgf(n_base, z)

        self.freq_moms = wrapped_moms
        self.freq_pgf = wrapped_pgf

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

        # ``__new__`` dispatches ``Frequency(name, ...)`` to the matching
        # registered ``Frequency<Kind>``. Direct construction on the base
        # ``Frequency`` is only reachable if ``freq_name`` is unregistered.
        if type(self) is Frequency:
            raise ValueError(
                f'Inadmissible frequency type {freq_name!r}; '
                f'available: {sorted(Frequency._registry)}')

        self._build()
        if freq_zm:
            if not self.supports_zm:
                raise NotImplementedError(
                    f'Zero modification not implemented for {freq_name}')
            self._install_zm_wrappers()

    def __str__(self):
        """
        wrap default with name
        :return:
        """
        return f'Frequency object of type {self.freq_name}\n{super(Frequency, self).__repr__()}'


# ---------------------------------------------------------------------------
# Concrete Frequency<Kind> subclasses. Each declares its registry key as a
# class-level ``freq_name`` and implements ``_build``, ``freq_moms``,
# ``freq_pgf``, and (for kinds that support ZM) ``prn_eq_0``.
# ---------------------------------------------------------------------------


class FrequencyPoisson(Frequency):
    """
    Poisson(n) frequency. Single-parameter: mean ``n``, variance ``n``.

    ``freq_a`` is unused. PGF :math:`G_N(z) = e^{n(z - 1)}`. Supports
    zero modification via the shared ZM machinery.
    """

    freq_name = 'poisson'
    supports_zm = True

    def _build(self):
        # No precomputation; freq_a is unused for pure Poisson.
        return None

    def prn_eq_0(self, n):
        return np.exp(-n)

    def freq_moms(self, n):
        freq_2 = n * (1 + n)
        freq_3 = n * (1 + n * (3 + n))
        self.panjer_ab = (0., n)
        return n, freq_2, freq_3

    def freq_pgf(self, n, z):
        return np.exp(n * (z - 1))


class FrequencyFixed(Frequency):
    """
    Degenerate frequency: ``N = n`` with probability 1. No parameters.
    """

    freq_name = 'fixed'

    def _build(self):
        return None

    def freq_moms(self, n):
        freq_2 = n ** 2
        freq_3 = n * freq_2
        return n, freq_2, freq_3

    def freq_pgf(self, n, z):
        return z ** n


class FrequencyBernoulli(Frequency):
    """
    Bernoulli frequency. ``n`` is interpreted as the probability of a single
    claim (so ``n < 1``); :math:`E(N^k) = n` for all :math:`k`.
    """

    freq_name = 'bernoulli'

    def _build(self):
        return None

    def freq_moms(self, n):
        return n, n, n

    def freq_pgf(self, n, z):
        return z * n + np.ones_like(z) * (1 - n)


class FrequencyBinomial(Frequency):
    """
    Binomial(N, p) frequency with overall mean ``n`` and per-trial success
    probability ``p = freq_a``. The trial count is :math:`N = n / p`.
    Supports zero modification.
    """

    freq_name = 'binomial'
    supports_zm = True

    def _build(self):
        # ``freq_a`` carries the per-trial probability ``p``; trial count
        # is derived at evaluation time from the requested mean.
        return None

    def prn_eq_0(self, n):
        p = self.freq_a
        N = n / p
        return (1 - p) ** N

    def freq_moms(self, n):
        p = self.freq_a
        N = n / p
        freq_1 = N * p
        freq_2 = N * p * (1 - p + N * p)
        freq_3 = N * p * (1 + p * (N - 1) * (3 + p * (N - 2)))
        self.panjer_ab = (-p / (1 - p), (N + 1) * p / (1 - p))
        return freq_1, freq_2, freq_3

    def freq_pgf(self, n, z):
        p = self.freq_a
        N = n / p
        return (z * p + np.ones_like(z) * (1 - p)) ** N


class FrequencyNegbin(Frequency):
    """
    Negative binomial with ``freq_a`` interpreted as the variance multiplier
    (variance / mean). Parameterized via ``r, β`` with ``β = freq_a - 1``
    and ``r = n / β``; mean ``rβ``, variance ``rβ(1+β)``. Supports zero
    modification.
    """

    freq_name = 'negbin'
    supports_zm = True

    def _build(self):
        self._beta = self.freq_a - 1

    def prn_eq_0(self, n):
        beta = self._beta
        r = n / beta
        return (1 + beta) ** -r

    def freq_moms(self, n):
        beta = self._beta
        r = n / beta
        freq_2 = n * (1 + beta * (1 + r))
        freq_3 = r * beta * (1 + beta * (1 + r) * (3 + beta * (2 + r)))
        self.panjer_ab = (beta / (1 + beta), (r - 1) * beta / (1 + beta))
        return n, freq_2, freq_3

    def freq_pgf(self, n, z):
        beta = self._beta
        r = n / beta
        return (1 - beta * (z - 1)) ** -r


class FrequencyGeometric(Frequency):
    """
    Geometric distribution supported on 0, 1, 2, ... with mean ``n``, hence
    success probability ``p = 1 / (n + 1)``. Supports zero modification.
    """

    freq_name = 'geometric'
    supports_zm = True

    def _build(self):
        return None

    def prn_eq_0(self, n):
        return 1 / (n + 1)

    def freq_moms(self, n):
        p = 1 / (n + 1)
        freq_2 = (2 - p) * (1 - p) / p ** 2
        freq_3 = (1 - p) * (6 + (p - 6) * p) / p ** 3
        self.panjer_ab = (n / (1 + n), 0.)
        return n, freq_2, freq_3

    def freq_pgf(self, n, z):
        p = 1 / (n + 1)
        return p / (1 - (1 - p) * z)


class FrequencyLogarithmic(Frequency):
    """
    Logarithmic series (``logser``) supported on 1, 2, 3, ... with mean
    ``n``; the parameter ``θ`` is solved numerically by
    :func:`_logarithmic_theta`. Supports zero modification (with
    ``prn_eq_0 = 0`` for the unmodified form, so ZM only adds mass at zero).
    """

    freq_name = 'logarithmic'
    supports_zm = True

    def _build(self):
        return None

    def prn_eq_0(self, n):
        return 0.

    def freq_moms(self, n):
        theta = _logarithmic_theta(n)
        a_logser = -1 / np.log(1 - theta)
        freq_2 = a_logser * theta / (1 - theta) ** 2
        freq_3 = a_logser * theta * (1 + theta) / (1 - theta) ** 3
        self.panjer_ab = (theta, -theta)
        return n, freq_2, freq_3

    def freq_pgf(self, n, z):
        theta = _logarithmic_theta(n)
        return np.log(1 - theta * z) / np.log(1 - theta)


class FrequencyNeymanA(Frequency):
    """
    Neyman A: Poisson stopped sum of Poisson. ``freq_a`` is the mean number
    of outcomes per cluster (``m2``); the overall mean is ``n = m1 * m2``.
    Aliases ``'neyman'`` and ``'neymanA'`` route here.
    """

    freq_name = 'neymana'

    def _build(self):
        self._m2 = self.freq_a

    def freq_moms(self, n):
        m2 = self._m2
        freq_2 = n * ((1 + m2) + n)
        freq_3 = n * ((1 + m2 * (3 + m2)) + 3 * freq_2 - 2 * n ** 2)
        return n, freq_2, freq_3

    def freq_pgf(self, n, z):
        m2 = self._m2
        m1 = n / m2
        return np.exp(m1 * (np.exp(m2 * (z - 1)) - 1))


class FrequencyPascal(Frequency):
    """
    Generalized Poisson-Pascal: Poisson stopped sum of negative binomials.
    ``freq_a`` is the overall CV ``ν``; ``freq_b`` is the mean claimants per
    claim ``κ``. The Poisson component has mean ``n / κ``.
    """

    freq_name = 'pascal'

    def _build(self):
        self._nu = self.freq_a
        self._kappa = self.freq_b

    def freq_moms(self, n):
        nu = self._nu
        kappa = self._kappa
        c = (n * nu ** 2 - 1 - kappa) / kappa
        lam = n / kappa
        g = kappa * lam * (
            2 * c ** 2 * kappa ** 2 + 3 * c * kappa ** 2 * lam + 3 * c * kappa ** 2 + 3 * c * kappa
            + kappa ** 2 * lam ** 2 + 3 * kappa ** 2 * lam + kappa ** 2 + 3 * kappa * lam + 3 * kappa + 1)
        return n, n * (kappa * (1 + c + lam) + 1), g

    def freq_pgf(self, n, z):
        nu = self._nu
        kappa = self._kappa
        c = (n * nu ** 2 - 1 - kappa) / kappa
        a = 1 / c
        theta = kappa * c
        lam = n / kappa
        return np.exp(lam * ((1 - theta * (z - 1)) ** -a - 1))


class FrequencyEmpirical(Frequency):
    """
    Empirical (user-supplied) discrete frequency. ``freq_a`` is the array of
    outcomes, ``freq_b`` the array of probability masses; both are
    validated and possibly summarized via ``validate_discrete_distribution``.
    Moments are independent of the requested mean ``n`` (which is ignored).
    """

    freq_name = 'empirical'

    def _build(self):
        self.freq_a, self.freq_b = validate_discrete_distribution(
            self.freq_a, self.freq_b)

    def freq_moms(self, n):
        en = np.sum(self.freq_a * self.freq_b)
        en2 = np.sum(self.freq_a ** 2 * self.freq_b)
        en3 = np.sum(self.freq_a ** 3 * self.freq_b)
        return en, en2, en3

    def freq_pgf(self, n, z):
        return self.freq_b @ np.power(z, self.freq_a.reshape((self.freq_a.shape[0], 1)))


class _FrequencyMixedPoisson(Frequency):
    """
    Shared scaffolding for G-mixed Poisson kinds: :math:`N \\mid G \\sim
    \\text{Poisson}(nG)` where ``G`` is a non-negative mixing distribution
    with mean 1 and CV ``ν = freq_a``. Subclasses compute ``g = E[G^3]``
    in ``_build`` (stored as ``self._g``) and supply their own
    ``freq_pgf``.

    All G-mixed Poissons share the same factorial-moment formulas because
    ``EN^k`` reduces to the non-central moments of ``G`` scaled by ``n``.
    """

    def freq_moms(self, n):
        c = self._c
        g = self._g
        freq_2 = n * (1 + (1 + c) * n)
        freq_3 = n * (1 + n * (3 * (1 + c) + n * g))
        return n, freq_2, freq_3


class FrequencyGammaMixed(_FrequencyMixedPoisson):
    """
    Gamma-mixed Poisson (= negative binomial). ``freq_a`` is the CV ``ν`` of
    the gamma mixing distribution; the resulting frequency has variance
    multiplier ``1 + n ν²``.
    """

    freq_name = 'gamma'

    def _build(self):
        nu = self.freq_a
        c = nu * nu
        self._c = c
        self._a = 1 / c
        self._theta = c
        self._g = 1 + 3 * c + 2 * c * c

    def freq_pgf(self, n, z):
        return (1 - self._theta * n * (z - 1)) ** -self._a


class FrequencyDelaporteMixed(_FrequencyMixedPoisson):
    """
    Delaporte-mixed Poisson: shifted gamma with a proportion of certain
    claims ``f = freq_b`` (must lie in ``[0, 1]``). ``freq_a`` is the CV of
    the mixing distribution.
    """

    freq_name = 'delaporte'

    def _build(self):
        nu = self.freq_a
        c = nu * nu
        f = self.freq_b
        a = (1 - f) ** 2 / c
        self._c = c
        self._f = f
        self._a = a
        self._theta = (1 - f) / a
        self._g = 2 * nu ** 4 / (1 - f) + 3 * c + 1

    def freq_pgf(self, n, z):
        f = self._f
        theta = self._theta
        a = self._a
        return np.exp(f * n * (z - 1)) * (1 - theta * n * (z - 1)) ** -a


class FrequencyIGMixed(_FrequencyMixedPoisson):
    """
    Inverse-gaussian-mixed Poisson. ``freq_a`` is the CV of the mixing
    distribution.
    """

    freq_name = 'ig'

    def _build(self):
        nu = self.freq_a
        c = nu ** 2
        mu = c
        lam = 1 / mu
        gamma_skew = 3 * np.sqrt(mu)
        self._c = c
        self._mu = mu
        self._lam = lam
        self._g = gamma_skew * nu ** 3 + 3 * c + 1

    def freq_pgf(self, n, z):
        mu = self._mu
        lam = self._lam
        return np.exp(1 / mu * (1 - np.sqrt(1 - 2 * mu ** 2 * lam * n * (z - 1))))


class FrequencySIGMixed(_FrequencyMixedPoisson):
    """
    Shifted inverse-gaussian-mixed Poisson. ``freq_a`` is the CV of the
    mixing distribution; ``freq_b`` is the proportion of certain claims.
    """

    freq_name = 'sig'

    def _build(self):
        nu = self.freq_a
        f = self.freq_b
        c = nu * nu
        mu = c / (1 - f) ** 2
        lam = (1 - f) / mu
        gamma_skew = 3 * np.sqrt(mu)
        self._c = c
        self._f = f
        self._mu = mu
        self._lam = lam
        self._g = gamma_skew * nu ** 3 + 3 * c + 1

    def freq_pgf(self, n, z):
        f = self._f
        mu = self._mu
        lam = self._lam
        return (np.exp(f * n * (z - 1))
                * np.exp(1 / mu * (1 - np.sqrt(1 - 2 * mu ** 2 * lam * n * (z - 1)))))


class FrequencyBetaMixed(_FrequencyMixedPoisson):
    """
    Beta-mixed Poisson over support :math:`[0, r]` (``r = freq_b > 1``) with
    mixing CV ``ν = freq_a``. Numerically unstable when the implied
    alpha/beta parameters approach 1.
    """

    freq_name = 'beta'

    def _build(self):
        nu = self.freq_a
        r = self.freq_b
        assert r > 1, f'beta-mixed Poisson requires r > 1, got {r}'
        self._c = nu * nu
        self._r = r
        # ``g`` depends on ``n`` so is computed lazily in ``freq_moms``.

    def freq_moms(self, n):
        c = self._c
        r = self._r
        b = (r - n * (1 + c)) * (r - n) / (c * n * r)
        a = n / (r - n) * b
        g = r ** 3 * np.exp(
            gammaln(a + b) + gammaln(a + 3) - gammaln(a + b + 3) - gammaln(a))
        freq_2 = n * (1 + (1 + c) * n)
        freq_3 = n * (1 + n * (3 * (1 + c) + n * g))
        return n, freq_2, freq_3

    def freq_pgf(self, n, z):
        c = self._c
        r = self._r
        b = (r - n * (1 + c)) * (r - n) / (c * n * r)
        a = (r - n * (1 + c)) / (c * r)
        return hyp1f1(a, a + b, r * (z - 1))


class _FrequencySichelBase(_FrequencyMixedPoisson):
    """
    Shared body for the Sichel family (generalized inverse-gaussian mixing).
    Subclasses provide a ``_calibrate`` method returning ``(mu, beta, lam)``
    via Broyden / Newton-Krylov; the base class wires the resulting
    parameters into ``freq_pgf``.
    """

    def _build(self):
        nu = self.freq_a
        self._nu = nu
        self._c = nu * nu
        mu, beta, lam = self._calibrate()
        self._mu = mu
        self._beta = beta
        self._lam = lam
        self._g = mu ** 2 * kv(lam + 2, mu / beta) / kv(lam, mu / beta)

    def freq_pgf(self, n, z):
        mu = self._mu
        beta = self._beta
        lam = self._lam
        kernel = n * (z - 1)
        inner = np.sqrt(1 - 2 * beta * kernel)
        return inner ** (-lam) * kv(lam, mu * inner / beta) / kv(lam, mu / beta)


class FrequencySichel(_FrequencySichelBase):
    """
    Pure Sichel: generalized inverse-gaussian mixing with shape parameter
    ``λ = freq_b``. The other GIG parameters ``μ, β`` are calibrated by
    Broyden to match mean = 1 and CV = ``freq_a``. Special cases include
    ``λ = -0.5`` (inverse gaussian) and ``λ = 0.5`` (reciprocal IG).
    """

    freq_name = 'sichel'

    def _calibrate(self):
        nu = self.freq_a
        lam = self.freq_b
        target = np.array([1, nu])
        mu = 1
        beta = nu ** 2

        def f(arr_in):
            mu_, beta_ = arr_in
            mu_ = np.exp(mu_)
            beta_ = np.exp(beta_)
            ex1, ex2 = np.array(
                [mu_ ** r * kv(lam + r, mu_ / beta_) / kv(lam, mu_ / beta_)
                 for r in (1, 2)])
            sd = np.sqrt(ex2 - ex1 * ex1)
            return np.array([ex1, sd]) - target

        try:
            params = broyden2(f, (np.log(mu), np.log(beta)),
                              verbose=False, iter=10000, f_rtol=1e-11)
        except NoConvergence as e:
            logger.error('Sichel calibration: Broyden did not converge: %s', e)
            raise

        logger.debug('sichel params from Broyden %s', params)
        mu_, beta_ = params
        return np.exp(mu_), np.exp(beta_), lam


class _FrequencySichelMatched(_FrequencySichelBase):
    """
    Shared body for Sichel kinds calibrated by matching the first three
    moments of another distribution (delaporte or shifted IG). Subclasses
    supply ``_match_target(nu, f)`` returning the target moment vector.
    """

    def _calibrate(self):
        nu = self.freq_a
        f = self.freq_b
        lam = -0.5
        mu = 1
        beta = nu ** 2
        target = self._match_target(nu, f)

        def fn(arr_in):
            mu_, beta_, lam_ = arr_in
            mu_ = np.exp(mu_)
            beta_ = np.exp(beta_)
            ex1, ex2, ex3 = np.array(
                [mu_ ** r * kv(lam_ + r, mu_ / beta_) / kv(lam_, mu_ / beta_)
                 for r in (1, 2, 3)])
            sd = np.sqrt(ex2 - ex1 * ex1)
            skew = (ex3 - 3 * ex2 * ex1 + 2 * ex1 ** 3) / (sd ** 3)
            return np.array([ex1, sd, skew]) - target

        try:
            params = broyden2(fn, (np.log(mu), np.log(beta), lam),
                              verbose=False, iter=10000, f_rtol=1e-11)
            if np.linalg.norm(params) > 20:
                # Fall back to Newton-Krylov on suspiciously large solutions.
                params1 = newton_krylov(
                    fn, (np.log(1.0), np.log(nu ** 2), -0.5),
                    verbose=False, iter=10000, f_rtol=1e-11)
                logger.warning(
                    f'{self.freq_name}: Broyden gave large result {params}; '
                    f'Newton-Krylov {params1}')
                if np.linalg.norm(params) > np.linalg.norm(params1):
                    params = params1
                    logger.warning('%s: using Newton-Krylov', self.freq_name)
        except NoConvergence as e:
            logger.error('%s calibration: Broyden did not converge: %s', self.freq_name, e)
            raise

        logger.debug('%s params from Broyden %s', self.freq_name, params)
        mu_, beta_, lam_ = params
        return np.exp(mu_), np.exp(beta_), lam_


class FrequencySichelGamma(_FrequencySichelMatched):
    """
    Sichel calibrated to delaporte moments: ``G = f + G'`` with
    ``E(G') = 1 - f``, matching SD and skewness of the corresponding
    delaporte distribution.
    """

    freq_name = 'sichel.gamma'

    def _match_target(self, nu, f):
        return np.array([1, nu, 2 * nu / (1 - f)])


class FrequencySichelIG(_FrequencySichelMatched):
    """
    Sichel calibrated to shifted-inverse-gaussian moments. Numerically
    fragile parameterization; may fail to converge for some inputs.
    """

    freq_name = 'sichel.ig'

    def _match_target(self, nu, f):
        return np.array([1, nu, 3.0 * nu / (1 - f)])


# ---------------------------------------------------------------------------
# Stats DataFrame helpers
# ---------------------------------------------------------------------------
# ``Aggregate.stats_df`` is the canonical (component, measure) × view
# DataFrame holding theoretical + empirical moments. ``MomentAggregator``
# emits its per-component / totals statistics as flat names like ``freq_1``,
# ``agg_m``; ``_flat_col_to_stats_index`` maps each to the
# ``(component, measure)`` tuple used by the ``stats_df`` row MultiIndex.

_STATS_META_NAMES = frozenset({
    'name', 'limit', 'attachment', 'el', 'prem', 'lr', 'sevcv_param',
    'mix_cv', 'wt',
})

_STATS_MEASURE_MAP = {'1': 'ex1', '2': 'ex2', '3': 'ex3', 'm': 'mean'}


def _flat_col_to_stats_index(col):
    """Map a flat ``MomentAggregator`` moment name to ``(component, measure)``.

    Examples: ``'freq_1' → ('freq', 'ex1')``, ``'agg_m' → ('agg', 'mean')``,
    ``'limit' → ('meta', 'limit')``.

    Used to bridge the flat moment names emitted by
    :meth:`MomentAggregator.get_fsa_stats` / ``column_names()`` to the
    canonical ``(component, measure)`` MultiIndex used by ``stats_df``.
    """
    if col in _STATS_META_NAMES:
        return ('meta', col)
    comp, _, measure = col.partition('_')
    if comp in ('freq', 'sev', 'agg'):
        return (comp, _STATS_MEASURE_MAP.get(measure, measure))
    raise ValueError(f'Cannot map column {col!r} for stats_df build.')


# Canonical row MultiIndex for ``stats_df`` — written directly in __init__
# (component columns) and the post-loop totals block (``mixed`` /
# ``independent``). All ``meta`` rows up top, then freq/sev/agg moment blocks.
# The frame is all-float: ``self.name`` is already an attribute, no need for
# a ``('meta','name')`` string row that would force ``dtype=object``.
_STATS_ROW_INDEX = pd.MultiIndex.from_tuples(
    [
        ('meta', 'limit'), ('meta', 'attachment'),
        ('meta', 'el'), ('meta', 'prem'), ('meta', 'lr'),
        ('meta', 'sevcv_param'), ('meta', 'mix_cv'), ('meta', 'wt'),
        ('freq', 'ex1'), ('freq', 'ex2'), ('freq', 'ex3'),
        ('freq', 'mean'), ('freq', 'cv'), ('freq', 'skew'),
        ('sev', 'ex1'), ('sev', 'ex2'), ('sev', 'ex3'),
        ('sev', 'mean'), ('sev', 'cv'), ('sev', 'skew'),
        ('agg', 'ex1'), ('agg', 'ex2'), ('agg', 'ex3'),
        ('agg', 'mean'), ('agg', 'cv'), ('agg', 'skew'),
    ],
    names=['component', 'measure'],
)


# ``_BOUNDED_FREQS`` / ``_BOUNDED_SCIPY_SEVS`` are imported from ``tail.py``
# (the single source of truth) at the top of this module and re-exported here.


class Aggregate:
    """Compound (aggregate) probability distribution.

    Implements the FFT-based algorithm of Mildenhall (2024): discretize
    severity, FFT, apply frequency PGF, inverse FFT. See
    ``_freq_sev_convolution`` for the five-line core; ``update_work`` for the
    orchestration (severity prep → occurrence reinsurance → convolution →
    aggregate reinsurance → audit). Validation by theoretical-vs-empirical
    moment comparison (paper §4.7) lives in the ``empirical`` and ``error``
    columns of ``stats_df`` written at the end of ``update_work``.

    Construction is via the ``__init__`` arguments below, or — more usually —
    via :func:`build` parsing DecL.

    **Public surface that Portfolio and Bounds depend on.** Three stats
    surfaces — ``info`` for text, ``describe`` for the daily moment audit,
    and ``stats_df`` for everything else — plus the compute and risk-measure
    surface:

    Stats / display
        - ``info``: one-screen textual summary (frequency, severity, layer,
          grid, validation flag). Not stats.
        - ``describe``: 3-row Freq / Sev / Agg moment table with theoretical
          and (post-``update``) empirical estimates plus relative errors.
          The daily-driver display.
        - ``stats_df``: ``MultiIndex (component, measure)`` × per-component
          / ``mixed`` / ``independent`` / ``empirical`` / ``error``.
          Single source of truth for Aggregate moments — see the property
          for the row / column reference.

    Data attributes
        - ``agg_density``: empirical PMF on the bucket grid (set by
          ``update_work``; consumed by Portfolio).
        - ``ftagg_density``: FT of the aggregate density (consumed by
          Portfolio's copula combine).
        - ``density_df``: per-bucket density / CDF / risk-measure frame.
        - ``n``: total frequency.
        - ``name``, ``program``, ``note``: spec metadata.
        - ``bs``, ``log2``, ``xs``: discretization grid.

    Methods
        - ``update``, ``update_work``: trigger / drive the compute.
        - ``q``, ``q_sev``, ``tvar``, ``tvar_sev``, ``cdf``, ``sf``, ``pdf``,
          ``pmf``, ``var_dict``: risk-measure surface.
        - ``sample``: draw from the discretised aggregate.
        - ``price``: distortion-based pricing.
        - ``approximate``, ``entropy_fit``: parametric fits to the FFT output.
        - ``apply_distortion``, ``pollaczeck_khinchine``: distortion / ruin.
        - ``plot``: single plotting entry point.
        - ``snap``, ``picks``, ``unwrap``, ``recommend_bucket``,
          ``aggregate_error_analysis``, ``severity_error_analysis``: utilities.

    Methods / attributes with a leading underscore are internal —
    ``_init_stats_df``, ``_record_component``, ``_freq_sev_convolution``,
    ``_apply_reins_work``, ``_limits``, ``_html_info_blob``,
    ``_make_var_tvar``, … . The legacy ``audit_df`` / ``report_df`` /
    ``report_ser`` / ``statistics`` / ``statistics_df`` /
    ``statistics_total_df`` surface has been removed; consult ``stats_df``
    instead.
    """

    # ================================================================
    # Public read-only properties: spec, density frame, reinsurance frames
    # ================================================================

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

    def _tail_info(self):
        """Build the :class:`~aggregate.tail.TailInfo` for this aggregate.

        Spec-only (frequency name + severity families); touches no computed
        density, so it is valid *before* ``update()``. A ``_certified_bounded``
        override short-circuits to a BOUNDED result so the certify contract and
        the lifted-natural-allocation guard are preserved.

        Returns
        -------
        aggregate.tail.TailInfo
        """
        if getattr(self, '_certified_bounded', False):
            return _tail.TailInfo(
                freq=TailClass.BOUNDED, sev=TailClass.BOUNDED, agg=TailClass.BOUNDED,
                freq_lc=None, sev_lc=None, agg_lc=None, alpha=None,
                driver='certified', flags={'certified': True})
        return _tail.aggregate_tail_info(self.frequency, self.sevs)

    @property
    def tail_class(self):
        """The ``(freq, sev, agg)`` :class:`~aggregate.tail.TailClass` triple.

        The authoritative tail-thickness computation: frequency and severity
        rungs by deterministic family lookup, the aggregate rung by the
        ``max`` combine rule. ``.bounded`` and the tail text are derived from
        this. Returns a :class:`~aggregate.tail.TailClasses` namedtuple with
        ``.freq`` / ``.sev`` / ``.agg`` fields.
        """
        return self._tail_info().classes

    @property
    def bounded(self) -> bool:
        """Whether the aggregate has bounded support.

        Derived view: ``True`` iff the aggregate tail class is
        :attr:`~aggregate.tail.TailClass.BOUNDED`, i.e. the frequency *and*
        every severity component is bounded. Frequencies in
        :data:`~aggregate.tail._BOUNDED_FREQS` (``fixed``, ``bernoulli``,
        ``binomial``, ``empirical``) are bounded; mixed Poisson / negbin / etc.
        are not. A severity is bounded when it is a histogram, a fixed atom, a
        bounded scipy family, or carries a finite layer ``exp_limit`` or splice
        ``sev_ub``. Conservative: ``False`` whenever boundedness cannot be
        proved from the spec. Set ``self.bounded = True`` to certify (e.g. a
        fat-tailed scipy severity with a large layer cap the heuristic misses).
        """
        return self._tail_info().agg == TailClass.BOUNDED

    @bounded.setter
    def bounded(self, value: bool) -> None:
        if value is not True and value is not False:
            raise ValueError('bounded must be True (certify) or False (reset)')
        self._certified_bounded = bool(value)

    @property
    def tail_description(self) -> str:
        """Three aligned lines describing the frequency, severity, and aggregate tails.

        E.g.::

            frequency tail           log-concave, super-exponential (poisson)
            severity tail            subexponential (lognorm)
            aggregate tail           subexponential

        Derived from :attr:`tail_class`; the lines are also appended to
        :meth:`info`.
        """
        info = self._tail_info()
        return '\n'.join(_tail.describe_lines(
            info, self.frequency.freq_name, self._sev_label()))

    @property
    def tail_explanation(self) -> str:
        """One-sentence explanation of how the aggregate tail class arises.

        E.g. "Log-concave super-exponential poisson frequency and
        subexponential lognorm severity give a subexponential aggregate (single
        big jump: P(S>x) approx E[N]*P(X>x))." Derived from :attr:`tail_class`.
        """
        info = self._tail_info()
        return _tail.explain(info, self.frequency.freq_name, self._sev_label())

    def _sev_label(self) -> str:
        """Short severity family label for tail text (the family, or ``'N components'``)."""
        if not self.sevs:
            return ''
        if len(self.sevs) == 1:
            name = getattr(self.sevs[0], 'sev_name', '')
            return name if isinstance(name, str) else 'severity'
        return f'{len(self.sevs)} components'

    @property
    def reins_bucket(self) -> str:
        """Rebucketing scheme for reinsurance net/ceded distributions.

        ``'linear'`` (default) splits each off-grid net/ceded value's mass
        across its two bracketing grid buckets, preserving the first moment
        exactly; ``'nearest'`` rounds to the closest bucket (≤ ``bs/2``
        positional bias). Mirrors :meth:`Portfolio.allocation_method`.

        Reinsurance is baked in during :meth:`update`, so a change after
        ``build`` requires a re-``update()`` to take effect. The setter
        clears the cached reinsurance frames so they rebuild on next access.
        """
        return self._reins_bucket

    @reins_bucket.setter
    def reins_bucket(self, value: str) -> None:
        if value not in ('linear', 'nearest'):
            raise ValueError(
                f"reins_bucket must be 'linear' or 'nearest', not {value!r}")
        if value != getattr(self, '_reins_bucket', None):
            self._reins_bucket = value
            self._reins_density_df = None
            self._reins_stats_df = None
            self._reins_view_stats_cache = None
            self._reins_describe = None

    @property
    def dsev_bucket(self) -> str:
        """Scheme for placing discrete-severity atoms onto the model grid.

        Applies to ``dsev`` / ``dhistogram`` / ``fixed`` (point-mass)
        severities; continuous severities are already exact via the
        cdf-difference and ignore this. Sibling of :attr:`reins_bucket`.

        ``'linear'`` (default) splits each off-grid atom's mass across its two
        bracketing grid buckets ``k``, ``k+1`` with weights ``1-f``, ``f``
        (``f = x/bs - k``), so the discretized first moment equals
        ``Σ xₖ pₖ`` **exactly**; ``'nearest'`` snaps each atom to its closest
        bucket (the historical behaviour, with up to ``bs/2`` positional bias).
        On-grid atoms (``f == 0``, e.g. integer atoms with ``bs == 1``) give
        identical results under both schemes.

        Discretization happens in :meth:`update`, so a change after ``build``
        requires a re-``update()`` to take effect. The setter clears the cached
        density frames so they rebuild on next access.

        Notes
        -----
        Phase 1 covers **unlayered** discrete severities (the empirical-sample
        use case). A *layered* discrete severity (``a xs b dsev ...``)
        discretizes via the standard cdf-difference and so behaves as
        ``'nearest'`` regardless of this setting.
        """
        return self._dsev_bucket

    @dsev_bucket.setter
    def dsev_bucket(self, value: str) -> None:
        if value not in ('linear', 'nearest'):
            raise ValueError(
                f"dsev_bucket must be 'linear' or 'nearest', not {value!r}")
        if value != getattr(self, '_dsev_bucket', None):
            self._dsev_bucket = value
            self._density_df = None
            self._sev_density_df = None

    def _severity_in_window(self):
        """Whether the severity support overlaps the output window ``[x_min, x_max]``.

        Returns ``True`` (the common case) when at least one severity bucket
        with positive mass falls inside the aggregate output grid. ``False``
        flags the divergent case (e.g. a high-claim-count aggregate windowed
        far from 0 while the per-claim severity sits near 0), where
        ``density_df.p_sev`` is all zeros and the severity must be read from
        ``sev_density_df`` instead.
        """
        if self.sev_density is None or self.xs_sev is None:
            return True
        nz = self.sev_density > 0
        if not np.any(nz):
            return True
        s_lo = float(self.xs_sev[nz][0])
        s_hi = float(self.xs_sev[nz][-1])
        return not (s_hi < self.x_min or s_lo > self.x_max)

    def _sev_density_on_output_grid(self):
        """Severity density mapped from ``xs_sev`` onto the output grid ``xs``.

        The severity is discretised on ``xs_sev`` (physical 0 at index ``i0``);
        the output grid is ``xs = x_min + bs*arange``. This places each severity
        bucket at its physical location on the output grid, zero where the
        severity falls outside the output window. With no offset (``i0 == 0``
        and ``x_min == 0``) it returns ``self.sev_density`` unchanged.

        Returns
        -------
        np.ndarray
            Severity density aligned to ``self.xs`` (length ``len(xs)``).

        Notes
        -----
        Output index ``k`` (physical ``x_min + k*bs``) corresponds to severity
        index ``k + i0 + j0`` where ``j0 = round(x_min/bs)`` -- because the
        severity bucket ``s`` sits at physical ``(s - i0)*bs`` and we need
        ``(s - i0)*bs == x_min + k*bs``. When the severity is far from the
        output window (e.g. a tight P&L window around a large negative mean) the
        overlap is empty and the result is all zeros -- the severity simply is
        not in the displayed window.
        """
        sev = self.sev_density
        if sev is None:
            return sev
        N = len(self.xs)
        j0 = int(round(self.x_min / self.bs)) if self.bs else 0
        shift = self.i0 + j0
        if shift == 0:
            return sev
        out = np.zeros(N)
        k = np.arange(N)
        src = k + shift
        valid = (src >= 0) & (src < N)
        out[valid] = np.asarray(sev)[src[valid]]
        return out

    @property
    def density(self):
        """The "live" part of :attr:`density_df` — rows with positive total mass.

        Returns ``density_df.query('p_total > 0')``: the actual support of the
        aggregate, dropping the leading and trailing zero-probability buckets of
        the FFT grid. This is usually what you want to *see*. It is recomputed on
        each access (a plain property, not cached) because the underlying frame
        can be rebuilt by ``update``.
        """
        return self.density_df.query('p_total > 0')

    @property
    def density_df(self):
        """Per-bucket density / distribution / risk-measure frame.

        Built lazily on first access after ``update``. Stored in
        ``self._density_df``; treat as read-only.

        Columns (in construction order):

        ================  =====================================  =========================
        Column            Set from                               Read by
        ================  =====================================  =========================
        ``loss``          ``self.xs`` (also the index)           ``plot``, ``q``, bounds
        ``p_total``       ``self.agg_density``                   Portfolio (when Aggregate
                                                                  is in a port), bounds,
                                                                  ``plot``, user code
        ``p``             alias of ``p_total``                   Portfolio API compat
        ``log_p``         ``np.log(p)``                          ``plot`` log scale
        ``F``             ``p.cumsum()``                         ``q``, ``var``, ``tvar``
        ``S``             ``1 - p_total.cumsum()``               ``q``, ``tvar``, ``plot``
        ``lev``           ``S.shift(1).cumsum() * bs``           ``epd``, pricing
        ``exa``           alias of ``lev``                       Portfolio API compat
        ``exlea``         ``(lev - loss * S) / F``               pricing
        ``e``             ``self.est_m`` (constant column)       ``epd``
        ``epd``           ``max(0, e - lev) / e``                pricing, allocation
        ``exgta``         ``loss + (e - exa) / S``               pricing
        ``exeqa``         ``loss`` (since ``E[X|X=a] = a``)      Portfolio API compat
        ================  =====================================  =========================

        Duplicated columns (``p == p_total``, ``exa == lev``, ``exeqa == loss``) are
        intentional: Portfolio's ``filter(regex='p_<name>')`` / ``exeqa_*`` /
        ``exa_*`` patterns require these names exist on the unit's frame so the
        unit can be inlined into a portfolio.

        :return: DataFrame indexed by ``loss``, columns as tabulated above.
        """
        if self._density_df is None:
            # really should have one of these anyway...
            if self.agg_density is None:
                raise ValueError('Update Aggregate before asking for density_df')

            # really convenient to have p=p_total to be consistent with Portfolio objects
            self._density_df = pd.DataFrame(dict(loss=np.asarray(self.xs, dtype=float),
                                                 p_total=self.agg_density))
            self._density_df = self._density_df.set_index('loss', drop=False)
            self._density_df['p'] = self._density_df.p_total
            # remove the fuzz, same method as Portfolio.remove_fuzz
            self._density_df = remove_fuzz(self._density_df)

            # Severity columns (p_sev/F_sev/S_sev/log_p_sev) now live on their
            # own native grid in ``sev_density_df`` -- on a windowed/signed grid
            # the severity (near 0) and the aggregate (windowed) no longer share
            # a grid, so forcing the severity onto the aggregate index is at best
            # partial. See ``sev_density_df`` and ``info``'s outside-window note.

            # reindex
            self._density_df = self._density_df.set_index('loss', drop=False)
            # guard log of 0 / fp-fuzz negatives (cosmetic display column only;
            # values unchanged: log(0) = -inf, log(<0) = nan as before).
            with np.errstate(divide='ignore', invalid='ignore'):
                self._density_df['log_p'] = np.log(self._density_df.p)

            # generally acceptable for F, by construction
            self._density_df['F'] = self._density_df.p.cumsum()

            # Update 2021-01-28: S is best computed forwards
            self._density_df['S'] = 1 - self._density_df.p_total.cumsum()

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
    def sev_density_df(self):
        """Per-bucket severity density / distribution on the **severity** grid.

        The severity lives on its own grid ``xs_sev`` (physical 0 at index
        ``i0``), which equals the aggregate grid only on the default 0-based
        case. On a windowed / signed aggregate the severity (near 0) and the
        aggregate (windowed, possibly far from 0) genuinely occupy different
        grids, so the severity reporting columns live here -- indexed by the
        severity's own loss -- rather than in ``density_df``. This frame is
        always correct regardless of the aggregate output window.

        Columns
        -------
        loss : severity grid ``xs_sev`` (also the index).
        p_sev : ``self.sev_density``.
        log_p_sev : ``log(p_sev)`` (``-inf`` at zero buckets).
        F_sev : ``p_sev.cumsum()``.
        S_sev : ``1 - p_sev.cumsum()``.

        Returns
        -------
        DataFrame indexed by severity ``loss``.
        """
        if self._sev_density_df is None:
            if self.sev_density is None:
                raise ValueError('Update Aggregate before asking for sev_density_df')
            df = pd.DataFrame(dict(loss=np.asarray(self.xs_sev, dtype=float),
                                   p_sev=self.sev_density))
            df = df.set_index('loss', drop=False)
            with np.errstate(divide='ignore', invalid='ignore'):
                if df.p_sev.dtype == np.dtype('O'):
                    df['log_p_sev'] = np.nan
                else:
                    df['log_p_sev'] = np.log(df.p_sev)
            df['F_sev'] = df.p_sev.cumsum()
            df['S_sev'] = 1 - df.p_sev.cumsum()
            self._sev_density_df = df
        return self._sev_density_df

    # ================================================================
    # Reinsurance reporting (rationalized; see dev/reins-reporting.md)
    # ================================================================

    @property
    def reins_density_df(self):
        """Per-bucket gross / ceded / net densities under reinsurance.

        One row per model-grid bucket (``loss = k * bs``); the empirical,
        FFT-ready rebucketed densities. Columns are **always present**
        regardless of which stages are configured; a missing stage
        contributes the no-cession values (ceded mass at 0, net = subject).

        Columns
        -------
        loss : grid (also the index).
        p_sev_gross, p_sev_ceded, p_sev_net : occurrence-level severity
            views. With no occurrence cover ``p_sev_ceded`` is a point mass
            at 0 and ``p_sev_net == p_sev_gross``.
        p_agg_gross, p_agg_ceded_occ, p_agg_net_occ : aggregate of each
            occurrence severity view. ``p_agg_gross`` is the *true* gross
            aggregate (``_fft_aggregate`` of the gross severity).
        p_agg_subject, p_agg_ceded, p_agg_net : aggregate-cover views.
            ``p_agg_subject`` is the aggregate input to the aggregate cover
            (= aggregate of the requested occurrence output); it equals
            ``p_agg_gross`` only when there is no occurrence cover.

        Notes
        -----
        All aggregate columns route through ``_fft_aggregate`` so the
        zero-risk / fixed-1 shortcuts apply consistently. Renamed from the
        legacy ``reinsurance_df``: ``p_agg_gross_occ -> p_agg_gross`` and the
        old ``p_agg_gross`` (the agg-cover input) ``-> p_agg_subject``.
        Returns ``None`` when no reinsurance is configured.
        """
        if self.occ_reins is None and self.agg_reins is None:
            logger.warning('Asking for reins_density_df, but no reinsurance specified. Returning None.')
            return None

        if self._reins_density_df is None:
            xs = self.xs
            has_occ = self.occ_reins is not None
            has_agg = self.agg_reins is not None
            # point mass at 0 == the "no cession" density (ceded 0 w.p. 1)
            zero = np.zeros_like(xs, dtype=float)
            zero[0] = 1.0

            # --- severity (occurrence-level) views -----------------------
            sev_gross = np.asarray(
                self.sev_density_gross if self.sev_density_gross is not None
                else self.sev_density, dtype=float)
            sev_ceded = np.asarray(self.sev_density_ceded, dtype=float) if has_occ else zero
            sev_net = np.asarray(self.sev_density_net, dtype=float) if has_occ else sev_gross
            df = pd.DataFrame({
                'loss': xs,
                'p_sev_gross': sev_gross,
                'p_sev_ceded': sev_ceded,
                'p_sev_net': sev_net,
            }, index=pd.Index(xs, name='loss'))

            # --- aggregate of each occurrence severity view --------------
            # p_agg_gross is the TRUE gross aggregate (FFT of gross sev).
            agg_gross, _ = self._fft_aggregate(sev_gross, self.padding)
            df['p_agg_gross'] = agg_gross
            if has_occ:
                logger.info('Computing aggregates with gcn severities')
                agg_ceded_occ, _ = self._fft_aggregate(sev_ceded, self.padding)
                agg_net_occ, _ = self._fft_aggregate(sev_net, self.padding)
                df['p_agg_ceded_occ'] = agg_ceded_occ
                df['p_agg_net_occ'] = agg_net_occ
            else:
                df['p_agg_ceded_occ'] = zero
                df['p_agg_net_occ'] = agg_gross

            # --- aggregate-cover views -----------------------------------
            # subject = the aggregate input to the agg cover = aggregate of
            # the requested occ output (== p_agg_gross when no occ stage).
            if has_agg:
                df['p_agg_subject'] = np.asarray(self.agg_density_gross, dtype=float)
                df['p_agg_ceded'] = np.asarray(self.agg_density_ceded, dtype=float)
                df['p_agg_net'] = np.asarray(self.agg_density_net, dtype=float)
            else:
                df['p_agg_subject'] = np.asarray(self.agg_density, dtype=float)
                df['p_agg_ceded'] = zero
                df['p_agg_net'] = np.asarray(self.agg_density, dtype=float)

            self._reins_density_df = df

        return self._reins_density_df

    def reins_occ_plot(self, axs=None):
        """
        Plots for occurrence reinsurance: occurrence log density and aggregate
        quantile plot. Reads the gross/ceded/net views from ``reins_density_df``.
        """
        if self.occ_reins is None:
            logger.warning('reins_occ_plot called with no occurrence reinsurance.')
            return
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(2 * FIG_W, FIG_H), constrained_layout=True)
            self.figure = fig
        ax0, ax1 = axs.flat

        rd = self.reins_density_df
        rd[['p_sev_gross', 'p_sev_ceded', 'p_sev_net']].rename(
            columns={'p_sev_gross': 'gross', 'p_sev_ceded': 'ceded',
                     'p_sev_net': 'net'}).plot(ax=ax0, logy=True)
        xl = ax0.get_xlim()
        l = self.spec['exp_limit']
        if type(l) != float:
            l = np.max(l)
        if l < np.inf:
            xl = [-l / 50, l * 1.025]
        ax0.set(xlim=xl, xlabel='Loss', ylabel='Occurrence log density', title='Occurrence')

        y = rd.loss.values
        for c, col in [('gross', 'p_agg_gross'), ('ceded', 'p_agg_ceded_occ'),
                       ('net', 'p_agg_net_occ')]:
            # Plot-cosmetic de-fuzz: deliberate carve-out from the shared
            # ``remove_fuzz`` -- a looser 1e-15 threshold plus the 0 -> nan step
            # below so empty buckets drop out of the survival line.
            s = rd[col].to_numpy().copy()
            s[np.abs(s) < 1e-15] = 0
            s_values = s[::-1].cumsum()[::-1]
            s_values = np.where(np.abs(s_values) < 1e-15, 0, s_values)
            s_values = np.where(s_values == 0, np.nan, s_values)
            ax1.plot(1 - s_values, y, label=c)
        ax1.set(xlabel='Probability of non-exceedance', ylabel='Loss', title='Aggregate')
        ax1.legend()

    def occ_bivariate(self, bs_ceded=None, bs_net=None,
                      log2_ceded=None, log2_net=None):
        """Joint law of aggregate occurrence ceded ``C`` and net ``N`` via 2D FFT.

        Computes the *joint* distribution of the aggregate occurrence ceded
        ``C = sum c(X_i)`` and net ``N = sum n(X_i)`` losses, where each gross
        claim ``X_i`` is split deterministically by the occurrence cession map.
        The two are **not** deterministic functions of one another -- the random
        claim count decouples them -- so the joint law carries genuine
        information (their correlation, co-moments, reinsurer-vs-cedent
        dependency) beyond the two univariate margins already in
        :attr:`reins_density_df`.

        Parameters
        ----------
        bs_ceded, bs_net : float, optional
            Ceded- / net-axis bucket sizes. Default: auto-sized from the
            univariate occurrence aggregate margins (``p_agg_ceded_occ`` /
            ``p_agg_net_occ``) via :func:`aggregate.multivariate.size_axis`.
        log2_ceded, log2_net : int, optional
            Ceded- / net-axis log2 grid lengths (grid has ``1 << log2`` points).
            Default: auto-sized (target 10, grown to cover the margin, capped).

        Returns
        -------
        MultivariateAggregate
            A first-class joint object in ``netceded`` mode, with the joint
            ``density``, the two axis grids (``axis_xs``), ``marginals`` /
            ``moments`` (``E[C^i N^j]``) / ``corr`` / ``describe`` / ``stats_df``
            / ``info`` and a two-panel ``plot`` (comonotone per-claim severity
            and joint aggregate). Equivalent to the DecL ``netceded <agg>`` form.

        Raises
        ------
        ValueError
            If the object carries no occurrence reinsurance, or has not been
            updated (no severity densities present).

        Notes
        -----
        **Occurrence only.** Any *aggregate* reinsurance on the object is
        ignored: this is the joint law of the per-occurrence ceded / net
        aggregates. (The aggregate-cover bivariate is degenerate -- at the
        aggregate level ceded and net are deterministic functions of the
        aggregate, supported on a curve -- so it is out of scope.)

        **Method.** Per claim, ``(c(X), n(X))`` lies on the line ``c + n = X``.
        Placing the gross severity mass at ``(c(x_k), n(x_k))`` (rebucketed onto
        the 2D grid by the active :attr:`reins_bucket` scheme) builds the
        bivariate severity ``S``. The joint aggregate density is
        ``iFFT2(freq_pgf(n, FFT2(S)))`` -- exactly the univariate
        :meth:`_fft_aggregate` with the 1D transforms replaced by 2D transforms,
        valid because ``freq_pgf(n, z)`` is elementwise in ``z``. The zero-risk
        (``n == 0``) and fixed-count-one shortcuts mirror ``_fft_aggregate``.
        Marginalising the result over one axis recovers the corresponding
        univariate occurrence ceded / net aggregate (exact validation targets;
        see the ``Cross-check`` notes in ``dev/done/reins-bivariate.md``).

        This wraps the object as a :class:`aggregate.multivariate.MultivariateAggregate`
        in ``netceded`` mode (the engine is :func:`aggregate.multivariate.build_netceded_joint`).
        """
        from .multivariate import MultivariateAggregate

        mv = MultivariateAggregate(
            self.name, mode='netceded', nc_agg=self,
            nc_kwargs=dict(bs_ceded=bs_ceded, bs_net=bs_net,
                           log2_ceded=log2_ceded, log2_net=log2_net))
        # build eagerly so preconditions (occ reins present, object updated)
        # raise here, and the returned object is ready to query.
        mv.update()
        return mv

    # ----- reinsurance stats: exact (EX) vs rebucketed (Est) -------------

    @staticmethod
    def _reins_moments6_from_raw(e1, e2, e3):
        """``(ex1, ex2, ex3, mean, cv, skew)`` from raw moments."""
        mw = MomentWrangler()
        mw.noncentral = (e1, e2, e3)
        return (e1, e2, e3, *mw.mcvsk)

    def _reins_exact_image_raw(self, image_fn, p_subject):
        """Raw moments ``E[g(X)^j]``, ``j=1..3``, of an exact loss image
        ``g = image_fn`` weighted by ``p_subject``: the pre-bucket truth
        ``sum g(xs)^j * p_subject`` with **no** rebucketing scatter.

        Notes
        -----
        Unlike :func:`xsden_to_mwrangler` (the ``Est`` basis), no
        defective-mass tail term is added: that convention places lost mass
        at the implied max loss ``xs[-1] + bs``, which is right for a gross
        aggregate but wrong for a ceded image (where the lost mass maps to the
        capped cession). The EX basis is therefore the literal exact moment of
        the on-grid subject; the EX-vs-Est difference reflects the rebucketing
        scatter (plus, for a gross aggregate carried through an FFT, the grid
        deficit -- negligible on an adequate grid).
        """
        g = np.asarray(image_fn(self.xs), dtype=float)
        p = np.asarray(p_subject, dtype=float)
        return (float(np.sum(g * p)),
                float(np.sum(g * g * p)),
                float(np.sum(g * g * g * p)))

    def _reins_agg6_from_sev_raw(self, s1, s2, s3):
        """Compound exact severity raw moments into aggregate
        ``(ex1, ex2, ex3, mean, cv, skew)`` via the frequency."""
        f1, f2, f3 = self.frequency.freq_moms(self.n)
        a1, a2, a3 = MomentAggregator.agg_from_fs(f1, f2, f3, s1, s2, s3)
        return (a1, a2, a3, *MomentAggregator.static_moments_to_mcvsk(a1, a2, a3))

    def _reins_density6(self, p):
        """``(ex1, ex2, ex3, mean, cv, skew)`` of a density on the grid."""
        mw = xsden_to_mwrangler(self.xs, np.asarray(p, dtype=float))
        return (*mw.noncentral, *mw.mcvsk)

    @property
    def _reins_view_stats(self):
        """Per-stage reinsurance moments on two bases: exact and rebucketed.

        Internal frame feeding :meth:`reins_describe` (which surfaces the
        ``EX`` exact vs ``Est`` rebucketed comparison as its ``Change``
        column). The public per-layer summary is :meth:`reins_stats_df`.
        Shaped like ``stats_df`` but indexed by stage / view / basis instead
        of component columns.

        Rows
        ----
        ``MultiIndex (component, measure)`` with ``component in
        {freq, sev, agg}`` and ``measure in {ex1, ex2, ex3, mean, cv, skew}``.

        Columns
        -------
        ``MultiIndex (stage, view, basis)``:

        * ``stage`` ``occ`` (when ``occ_reins``) and/or ``agg`` (when
          ``agg_reins``);
        * ``view`` ``gross|ceded|net`` for ``occ``; ``subject|ceded|net`` for
          ``agg``;
        * ``basis`` ``EX`` (exact, pre-bucket) or ``Est`` (rebucketed).

        Notes
        -----
        **EX** ("theoretic"). For the occurrence stage the exact severity
        moments are ``E[g(X)^j] = sum g(xs)^j * p_sev_gross`` with ``g`` the
        identity / ``occ_ceder`` / ``occ_netter``; the aggregate row compounds
        those via the frequency. For the aggregate stage the exact aggregate
        moments are ``E[g(S)^j] = sum g(xs)^j * p_agg_subject`` with ``g`` the
        identity / ``agg_ceder`` / ``agg_netter`` (no compounding -- the cover
        acts on the aggregate directly). Frequency: the ``EX`` reference is the
        gross full moments for every view (the count is unchanged by occurrence
        reinsurance). On the ``Est`` (model-output) basis the gross view is left
        ``NaN`` (mirroring ``describe``, which never re-estimates the input
        frequency); occ ceded / net carry the *unconditional* mean ``E[N]`` only
        (so ``freq * sev == agg`` per view), cv / skew ``NaN``. The aggregate
        stage has no sev rows and a degenerate freq row (all ``NaN``).
        Columns are ordered occ before agg, views gross/subject, ceded, net,
        and ``EX`` before ``Est``.

        **Est** ("empirical") reads the rebucketed densities from
        :meth:`reins_density_df` through ``xsden_to_mwrangler``. The
        difference EX vs Est isolates the per-stage ``reins_bucket``
        rebucketing error (``linear`` preserves the mean exactly; ``nearest``
        biases it by at most ``bs/2``).

        Lazily built; invalidated by the ``reins_bucket`` setter and on
        ``update``. Returns ``None`` when no reinsurance is configured.
        """
        if self.occ_reins is None and self.agg_reins is None:
            return None
        if self._reins_view_stats_cache is not None:
            return self._reins_view_stats_cache

        measures = ['ex1', 'ex2', 'ex3', 'mean', 'cv', 'skew']
        components = ['freq', 'sev', 'agg']
        row_index = pd.MultiIndex.from_product(
            [components, measures], names=['component', 'measure'])
        nan6 = (np.nan,) * 6

        rd = self.reins_density_df
        data = {}  # (stage, view, basis) -> Series over row_index

        def put(stage, view, basis, freq6, sev6, agg6):
            s = pd.Series(np.nan, index=row_index)
            for comp, six in zip(components, (freq6, sev6, agg6)):
                for m, v in zip(measures, six):
                    s[(comp, m)] = v
            data[(stage, view, basis)] = s

        # ---- occurrence stage -------------------------------------------
        if self.occ_reins is not None:
            p_gross = rd['p_sev_gross'].to_numpy()
            n = self.n
            # Frequency is unchanged by occurrence reinsurance, so the gross
            # full moments are the theoretic reference (``EX``) for every view.
            # The model-output (``Est``) frequency is reported *unconditionally*
            # -- no division by ``P(attach)`` -- so ``freq * sev == agg`` within
            # each view; only the mean is meaningful (the per-view count is the
            # gross count), so cv / skew stay ``NaN``. The gross ``Est``
            # frequency is left ``NaN`` to mirror ``describe`` exactly (the
            # validation view never re-estimates the input frequency).
            f1, f2, f3 = self.frequency.freq_moms(n)
            freq_gross = (f1, f2, f3,
                          *MomentAggregator.static_moments_to_mcvsk(f1, f2, f3))
            freq_est_uncond = (np.nan, np.nan, np.nan, f1, np.nan, np.nan)

            # EX: exact image moments of the gross severity, compounded; the
            # frequency reference is the gross full moments for every view.
            identity = lambda x: x
            for view, image_fn in [('gross', identity),
                                   ('ceded', self.occ_ceder),
                                   ('net', self.occ_netter)]:
                s1, s2, s3 = self._reins_exact_image_raw(image_fn, p_gross)
                put('occ', view, 'EX',
                    freq_gross,
                    self._reins_moments6_from_raw(s1, s2, s3),
                    self._reins_agg6_from_sev_raw(s1, s2, s3))

            # Est: moments of the rebucketed densities; unconditional freq
            # (gross frequency left NaN to mirror describe).
            for view, scol, acol, freq6 in [
                    ('gross', 'p_sev_gross', 'p_agg_gross', nan6),
                    ('ceded', 'p_sev_ceded', 'p_agg_ceded_occ', freq_est_uncond),
                    ('net', 'p_sev_net', 'p_agg_net_occ', freq_est_uncond)]:
                put('occ', view, 'Est',
                    freq6,
                    self._reins_density6(rd[scol].to_numpy()),
                    self._reins_density6(rd[acol].to_numpy()))

        # ---- aggregate stage --------------------------------------------
        if self.agg_reins is not None:
            p_subject = rd['p_agg_subject'].to_numpy()
            identity = lambda x: x
            for view, image_fn in [('subject', identity),
                                   ('ceded', self.agg_ceder),
                                   ('net', self.agg_netter)]:
                a1, a2, a3 = self._reins_exact_image_raw(image_fn, p_subject)
                put('agg', view, 'EX', nan6, nan6,
                    self._reins_moments6_from_raw(a1, a2, a3))
            for view, acol in [('subject', 'p_agg_subject'),
                              ('ceded', 'p_agg_ceded'),
                              ('net', 'p_agg_net')]:
                put('agg', view, 'Est', nan6, nan6,
                    self._reins_density6(rd[acol].to_numpy()))

        # Canonical column order: occ before agg; within a stage the views
        # gross/subject, ceded, net (never alphabetical); EX before Est.
        ordered = []
        if self.occ_reins is not None:
            for view in ('gross', 'ceded', 'net'):
                for basis in ('EX', 'Est'):
                    ordered.append(('occ', view, basis))
        if self.agg_reins is not None:
            for view in ('subject', 'ceded', 'net'):
                for basis in ('EX', 'Est'):
                    ordered.append(('agg', view, basis))
        out = pd.DataFrame(data)[ordered]
        out.columns = pd.MultiIndex.from_tuples(
            ordered, names=['stage', 'view', 'basis'])
        self._reins_view_stats_cache = out
        return self._reins_view_stats_cache

    @property
    def reins_stats_df(self):
        """Per-layer reinsurance layering summary (empirical, model-grid).

        A layering analysis with one column per reinsurance layer plus the
        gross book and the ceded / net totals.

        Columns
        -------
        ``MultiIndex (view, layer)`` -- ``view`` is ``occ`` / ``agg``:

        * occurrence: ``Gross`` (the gross book -- always present), then, when
          ``occ_reins``, ``layer.1`` ... ``layer.k`` (one per layer) and
          ``Ceded`` / ``Net`` totals.
        * aggregate (when ``agg_reins``): ``layer.1`` ... ``layer.m``,
          ``Ceded`` / ``Net``. There is **no aggregate ``Subject`` column** --
          the subject (the occurrence output the aggregate cover applies to)
          is the column flagged ``('meta', 'output') == 1`` in the occurrence
          block, or ``Gross`` when there is no occurrence program.

        Rows
        ----
        ``MultiIndex (component, measure)``:

        * ``meta``: ``share`` (proportion covered), ``limit``, ``attach``,
          ``pr_attach`` and ``pr_detach`` -- the **ground-up exposure
          probabilities** that the underlying loss attaches / fully exhausts
          the view, ``P(X > exp_attach + view_attach)`` and ``P(X > exp_attach
          + view_attach + view_limit)``. These come from the **underlying**
          severity (``self.sevs[i].fz``), not the modeled ``sev_density``,
          which is conditional (claims to the policy layer) and would report 0
          at the policy cap. ``pr_detach`` is ``NaN`` for unlimited layers /
          net totals. ``pr_loss`` (``P(aggregate > 0)`` from the column's
          aggregate density), ``lol`` (loss on line = expected layer aggregate
          loss / placed limit), and ``output`` (``0/1`` flag marking each
          stage's output view -- two 1s for an occ+agg program; ``Gross``
          carries the 1 when there is no occurrence program). ``Gross`` carries
          the claim-count-weighted policy ``limit`` / ``attach`` (``share`` 1);
          the occurrence ``Ceded`` total carries the share-placed sum of layer
          limits and the minimum attachment. (The layer ``freq`` ``n'`` uses
          the *conditional* ``P(subject > attach | policy loss)`` -- the model
          count is claims to the policy -- a separate basis from the absolute
          ``pr_attach``.)
        * ``('freq'|'sev'|'agg', ex1|ex2|ex3|mean|cv|skew)`` -- moments
          (``ex1`` duplicates ``mean`` for easy ``filter(regex=...)`` access).

        Notes
        -----
        **Occurrence layers are conditional** on a loss reaching the layer:
        the frequency is the expected penetrating count ``n' = E[N] * P(X >
        attach)`` and the severity is the unconditional layer severity divided
        by ``P(X > attach)`` (conditional given attach), which leaves the layer
        aggregate mean ``n' * sev`` equal to the unconditional ``E[N] *
        E[ceded]``. The ``agg`` row is the column's actual aggregate
        distribution (FFT of the unconditional layer ceded severity), so its
        higher moments and ``pr_loss`` are exact.

        **The occurrence ``Ceded`` / ``Net`` columns are unconditional**
        totals: the same claim count as ``Gross`` and unconditional
        severities, so ``Ceded`` sev + ``Net`` sev == ``Gross`` sev. Aggregate
        layer means sum to the ``Ceded`` aggregate mean.

        **The aggregate block leaves ``freq`` and ``sev`` all ``NaN``** -- a
        cover on the aggregate has no per-claim frequency / severity that
        combine in the usual way.

        Lazily built; invalidated by the ``reins_bucket`` setter and on
        ``update``. Returns ``None`` when no reinsurance is configured.
        """
        if self.occ_reins is None and self.agg_reins is None:
            return None
        if self._reins_stats_df is not None:
            return self._reins_stats_df

        measures = ['ex1', 'ex2', 'ex3', 'mean', 'cv', 'skew']
        components = ['freq', 'sev', 'agg']
        meta_rows = ['share', 'limit', 'attach', 'pr_attach', 'pr_detach',
                     'pr_loss', 'lol', 'output']
        row_index = pd.MultiIndex.from_tuples(
            [('meta', m) for m in meta_rows]
            + [(c, m) for c in components for m in measures],
            names=['component', 'measure'])
        nan6 = (np.nan,) * 6
        rd = self.reins_density_df
        xs = self.xs
        n = self.n
        data = {}

        def col(share=np.nan, limit=np.nan, attach=np.nan, pr_attach=np.nan,
                pr_detach=np.nan, pr_loss=np.nan, lol=np.nan, output=0.0,
                freq6=nan6, sev6=nan6, agg6=nan6):
            s = pd.Series(np.nan, index=row_index)
            for mk, mv in zip(meta_rows, (share, limit, attach, pr_attach,
                                          pr_detach, pr_loss, lol, output)):
                s[('meta', mk)] = mv
            for comp, six in zip(components, (freq6, sev6, agg6)):
                for m, v in zip(measures, six):
                    s[(comp, m)] = v
            return s

        def moments6(raw):
            """(ex1, ex2, ex3, mean, cv, skew) from raw moments ``raw``."""
            return (*raw, *MomentAggregator.static_moments_to_mcvsk(*raw))

        def _raw3(density):
            d = np.asarray(density, dtype=float)
            return (float(np.sum(xs * d)),
                    float(np.sum(xs * xs * d)),
                    float(np.sum(xs * xs * xs * d)))

        def _pr_pos(density):
            """P(loss > 0) = 1 - mass in the zero bucket."""
            return float(1.0 - np.asarray(density, dtype=float)[0])

        def _sf(a):
            try:
                return float(self.sev.sf(a))
            except Exception:  # pragma: no cover - exotic severities
                return np.nan

        def _ge(density, t):
            """P(loss >= t) from a density, ``NaN`` if ``t`` is not finite.
            Used for the *aggregate*-level attach/detach probabilities (the
            modeled aggregate density is exact)."""
            if not np.isfinite(t):
                return np.nan
            return float(np.sum(np.asarray(density, dtype=float)[xs >= t]))

        # Claim-count weights and the ground-up severity survival, used for the
        # occurrence attach/detach probabilities. The modeled ``sev_density``
        # is *conditional* (claims to the policy layer, ``n`` the conditional
        # count), so the unconditional exposure probabilities -- P(a ground-up
        # claim's subject loss exceeds a threshold) -- come from the underlying
        # frozen severity ``fz`` with each component's policy attachment as the
        # offset. The limited ``self.sev.sf`` would report 0 at the policy cap
        # by definition; the ground-up ``fz`` gives the true detachment prob.
        en = np.asarray(self.en, dtype=float)
        ws = (en / en.sum() if en.sum() > 0
              else np.full(len(self.sevs), 1.0 / max(len(self.sevs), 1)))

        def _gsf(t, inclusive=False):
            """Weighted P(subject loss > t) (``>= t`` if ``inclusive``).

            ``subject_i = layer(X_i; attach_i, limit_i)``, so subject > t iff
            the ground-up ``X_i > attach_i + t`` (and t below the cap)."""
            try:
                tot = 0.0
                for i, sev in enumerate(self.sevs):
                    lim = float(sev.limit)
                    if t < lim or (inclusive and t <= lim):
                        tot += ws[i] * float(sev.fz.sf(float(sev.attachment) + t))
                return float(tot)
            except Exception:  # pragma: no cover - exotic severities
                return np.nan

        def _gross_detach():
            """Weighted P(policy detaches) = P(ground-up >= attach + limit)
            over finite-limit components; ``NaN`` if every component unlimited."""
            fin = [i for i, sev in enumerate(self.sevs)
                   if np.isfinite(float(sev.limit))]
            if not fin:
                return np.nan
            try:
                return float(sum(
                    ws[i] * float(self.sevs[i].fz.sf(
                        float(self.sevs[i].attachment) + float(self.sevs[i].limit)))
                    for i in fin))
            except Exception:  # pragma: no cover - exotic severities
                return np.nan

        def _lol(mean, placed_limit):
            """Loss on line: expected layer aggregate loss / placed limit."""
            if placed_limit is None or np.isnan(placed_limit):
                return np.nan
            if np.isinf(placed_limit):
                return 0.0
            return mean / placed_limit if placed_limit > 0 else np.nan

        def _layer_ceded(ceder, subject):
            """Rebucketed ceded density: subject mass through one layer."""
            return self._rebucket_to_grid(
                np.asarray(ceder(xs), dtype=float),
                np.asarray(subject, dtype=float))

        # Full gross frequency (count unchanged by reinsurance).
        f1, f2, f3 = self.frequency.freq_moms(n)
        freq_full = moments6((f1, f2, f3))

        # Claim-count-weighted gross policy limit / attachment (mixtures);
        # ``en`` / ``ws`` were set with the ground-up survival helpers above.
        lim = np.asarray(self.limit, dtype=float)
        att = np.asarray(self.attachment, dtype=float)
        if en.sum() > 0:
            gross_limit = float(np.average(lim, weights=en))
            gross_attach = float(np.average(att, weights=en))
        else:  # zero-risk fallback
            gross_limit = float(np.mean(lim))
            gross_attach = float(np.mean(att))

        # Which view carries each stage's output=1 flag.
        occ_out = (REINS_LABEL_NET if self.occ_kind == 'net of'
                   else REINS_LABEL_CEDED) if self.occ_reins is not None else None
        agg_out = (REINS_LABEL_NET if self.agg_kind == 'net of'
                   else REINS_LABEL_CEDED) if self.agg_reins is not None else None
        # Gross is the aggregate subject (output=1) only when there is an
        # aggregate program but no occurrence program.
        gross_output = 1.0 if (self.agg_reins is not None
                               and self.occ_reins is None) else 0.0

        # ----- gross book (always present) -----
        p_sev_gross = rd['p_sev_gross'].to_numpy()
        gross_agg6 = self._reins_density6(rd['p_agg_gross'].to_numpy())
        data[('occ', REINS_LABEL_GROSS)] = col(
            share=1.0, limit=gross_limit, attach=gross_attach,
            # exposure probabilities from the underlying ground-up severity
            pr_attach=_gsf(0.0),            # P(a ground-up claim hits the policy)
            pr_detach=_gross_detach(),      # P(it exhausts the policy limit)
            pr_loss=_pr_pos(rd['p_agg_gross'].to_numpy()),
            lol=_lol(gross_agg6[3], gross_limit),
            output=gross_output,
            freq6=freq_full,
            sev6=self._reins_density6(p_sev_gross),
            agg6=gross_agg6)

        # ----- occurrence layering (conditional layers; unconditional totals) -----
        if self.occ_reins is not None:
            p_gross = rd['p_sev_gross'].to_numpy()
            for k, (s, y, a) in enumerate(self.occ_reins, 1):
                ceder_k, _ = make_ceder_netter([(s, y, a)])
                ceded_k = _layer_ceded(ceder_k, p_gross)
                # Conditioning for the layer freq / sev is relative to the
                # *policy* claims (the modeled count ``n``): P(subject > a |
                # policy loss) = self.sev.sf(a). The displayed pr_attach /
                # pr_detach are the absolute ground-up exposure probabilities.
                pr = _sf(a)
                u1, u2, u3 = _raw3(ceded_k)              # raw (conditional on policy)
                if pr and not np.isnan(pr):              # condition on the layer
                    cond = (u1 / pr, u2 / pr, u3 / pr)
                else:
                    cond = (np.nan, np.nan, np.nan)
                lf = self.frequency.freq_moms(n * pr)    # conditional count n'
                agg_k, _ = self._fft_aggregate(ceded_k, self.padding)
                agg6 = self._reins_density6(agg_k)
                data[('occ', f'layer.{k}')] = col(
                    share=s, limit=y, attach=a,
                    pr_attach=_gsf(a),
                    pr_detach=_gsf(a + y, inclusive=True) if np.isfinite(y) else np.nan,
                    pr_loss=_pr_pos(agg_k),
                    lol=_lol(agg6[3], s * y),
                    freq6=moments6(lf), sev6=moments6(cond), agg6=agg6)
            placed = float(sum(s * y for (s, y, _a) in self.occ_reins))
            min_attach = float(min(a for (_s, _y, a) in self.occ_reins))
            ceded_agg6 = self._reins_density6(rd['p_agg_ceded_occ'].to_numpy())
            data[('occ', REINS_LABEL_CEDED)] = col(
                limit=placed, attach=min_attach,
                pr_attach=_gsf(min_attach),   # P(any ceding) = P(hit lowest layer)
                pr_loss=_pr_pos(rd['p_agg_ceded_occ'].to_numpy()),
                lol=_lol(ceded_agg6[3], placed),
                output=1.0 if occ_out == REINS_LABEL_CEDED else 0.0,
                freq6=freq_full,
                sev6=self._reins_density6(rd['p_sev_ceded'].to_numpy()),
                agg6=ceded_agg6)
            data[('occ', REINS_LABEL_NET)] = col(
                pr_attach=_gsf(0.0),          # P(any subject loss retained)
                pr_loss=_pr_pos(rd['p_agg_net_occ'].to_numpy()),
                output=1.0 if occ_out == REINS_LABEL_NET else 0.0,
                freq6=freq_full,
                sev6=self._reins_density6(rd['p_sev_net'].to_numpy()),
                agg6=self._reins_density6(rd['p_agg_net_occ'].to_numpy()))

        # ----- aggregate layering (freq/sev left NaN -- they don't combine) -----
        if self.agg_reins is not None:
            p_subject = rd['p_agg_subject'].to_numpy()
            pr_subject = _pr_pos(p_subject)              # P(subject > 0)

            def _pr_agg(a):
                return float(np.sum(p_subject[xs > a]))

            for k, (s, y, a) in enumerate(self.agg_reins, 1):
                ceder_k, _ = make_ceder_netter([(s, y, a)])
                ceded_k = _layer_ceded(ceder_k, p_subject)
                agg6 = self._reins_density6(ceded_k)
                data[('agg', f'layer.{k}')] = col(
                    share=s, limit=y, attach=a, pr_attach=_pr_agg(a),
                    pr_detach=_ge(p_subject, a + y), pr_loss=_pr_pos(ceded_k),
                    lol=_lol(agg6[3], s * y), agg6=agg6)
            placed = float(sum(s * y for (s, y, _a) in self.agg_reins))
            min_attach = float(min(a for (_s, _y, a) in self.agg_reins))
            ceded_agg6 = self._reins_density6(rd['p_agg_ceded'].to_numpy())
            data[('agg', REINS_LABEL_CEDED)] = col(
                limit=placed, attach=min_attach, pr_attach=pr_subject,
                pr_loss=_pr_pos(rd['p_agg_ceded'].to_numpy()),
                lol=_lol(ceded_agg6[3], placed),
                output=1.0 if agg_out == REINS_LABEL_CEDED else 0.0,
                agg6=ceded_agg6)
            data[('agg', REINS_LABEL_NET)] = col(
                pr_attach=pr_subject,
                pr_loss=_pr_pos(rd['p_agg_net'].to_numpy()),
                output=1.0 if agg_out == REINS_LABEL_NET else 0.0,
                agg6=self._reins_density6(rd['p_agg_net'].to_numpy()))

        ordered = list(data.keys())  # gross, occ block, agg block (insertion)
        out = pd.DataFrame(data)[ordered]
        out.columns = pd.MultiIndex.from_tuples(ordered, names=['view', 'layer'])
        self._reins_stats_df = out
        return self._reins_stats_df

    # Column layout shared with ``describe`` (see :meth:`_describe`): exact
    # (``EX``) value, rebucketed (``Est``) value, and ``Change`` for the mean
    # and CV; skew omits the change column (it is the hardest moment to
    # estimate). ``EX``/``Est`` here are the reins bases, not theory/empirical.
    _REINS_DESCRIBE_COLS = ['EX', 'Est EX', 'Change EX',
                            'CV', 'Est CV', 'Change CV',
                            'Sk', 'Est Sk']

    @property
    def reins_describe(self):
        """Per-stage reinsurance summary -- the daily driver.

        Mirrors the **economic view** of :meth:`describe`: compare the theoretic
        reference (the leading view -- ``Gross`` for occurrence, ``Subject`` for
        aggregate) against the model output of each view. One block per
        applicable stage; each block is a ``view x component`` table sharing the
        **same eight columns as** :meth:`describe`:

        * ``EX`` / ``Est EX`` / ``Change EX`` -- the **theoretic reference** mean
          (constant down each component), the per-view model-output mean, and
          ``(Est - reference) / reference``;
        * ``CV`` / ``Est CV`` / ``Change CV`` -- the same three for the CV;
        * ``Sk`` / ``Est Sk`` -- reference and model-output skew (no change
          column).

        ``Change`` carries two readings off one arithmetic: on the leading
        (Gross / Subject) row the reference and the model output are the same
        view, so it is the **numerical validation / rebucketing error** (~0 under
        ``linear``); on the ceded / net rows it is the **% impact of the
        cession** on that moment. This is the per-view, per-component analogue of
        the single ``Change`` column in :meth:`describe`.

        Layout (per the gross/subject convention; ``view`` and ``component``
        labels are lower-case to match the other frames):

        * **Occurrence block** -- leads with ``gross`` (the reference): rows
          ``(gross|ceded|net) x (freq|sev|agg)``.
        * **Aggregate block** -- leads with ``subject`` (the reference): rows
          ``(subject|ceded|net) x (agg)`` (sev not applicable; freq
          degenerate -> ``NaN``).

        Frequency is reported on the ``Est`` (model-output) basis
        *unconditionally* (mean ``E[N]`` only, so ``freq * sev == agg`` within a
        view; cv / skew ``NaN``) -- consistent with :meth:`reins_stats_df`. The
        leading ``gross`` row's ``Est`` frequency is left ``NaN`` to mirror
        :meth:`describe` exactly. (Only the per-layer ``layer.k`` columns of
        :meth:`reins_stats_df` are *conditional*; ``reins_describe`` is always
        unconditional.)

        Index is ``MultiIndex (stage, view, component)``. Derived from
        :meth:`reins_stats_df` / the internal view-stats frame. Returns ``None``
        when no reinsurance is configured.
        """
        if self.occ_reins is None and self.agg_reins is None:
            return None
        if self._reins_describe is not None:
            return self._reins_describe
        blocks = []
        if self.occ_reins is not None:
            blocks.append(self._reins_describe_block(
                'occ', ['gross', 'ceded', 'net'], ['freq', 'sev', 'agg']))
        if self.agg_reins is not None:
            blocks.append(self._reins_describe_block(
                'agg', ['subject', 'ceded', 'net'], ['agg']))
        self._reins_describe = pd.concat(blocks)
        return self._reins_describe

    def _reins_describe_block(self, stage, views, comps):
        """One :meth:`reins_describe` block: theoretic reference vs model output
        by view x component, mirroring the eight-column :meth:`describe` layout.

        The ``EX`` / ``CV`` / ``Sk`` columns hold the **theoretic reference** --
        the leading view's exact (pre-bucket) moments: ``Gross`` for the
        occurrence block, ``Subject`` for the aggregate block. They are therefore
        constant down each component (the same reference is compared against
        every view). ``Est *`` is the per-view model output (the rebucketed,
        model-grid moment). ``Change`` is ``(Est - reference) / reference``: on
        the leading (Gross/Subject) row it degenerates to the rebucketing /
        validation error (~0 under ``linear``); on the ceded / net rows it reads
        as the % impact of the cession on that moment.
        """
        rs = self._reins_view_stats
        ref_view = views[0]  # Gross (occ) / Subject (agg) -- theoretic reference
        rows = []
        idx = []
        for view in views:
            for comp in comps:
                def _ref_est(measure):
                    ref = float(rs.loc[(comp, measure), (stage, ref_view, 'EX')])
                    est = float(rs.loc[(comp, measure), (stage, view, 'Est')])
                    return ref, est
                ref_m, est_m = _ref_est('mean')
                ref_cv, est_cv = _ref_est('cv')
                ref_sk, est_sk = _ref_est('skew')
                rows.append([
                    ref_m, est_m, float(_noise_aware_rel_error(est_m, ref_m)),
                    ref_cv, est_cv, float(_noise_aware_rel_error(est_cv, ref_cv)),
                    ref_sk, est_sk,
                ])
                idx.append((stage, view, comp))
        mi = pd.MultiIndex.from_tuples(idx, names=['stage', 'view', 'component'])
        df = pd.DataFrame(rows, index=mi, columns=self._REINS_DESCRIBE_COLS)
        # Snap fp dust to 0 in the value columns (skew of a symmetric view);
        # the Change columns keep their dust as the rebucketing eyeball.
        for c in df.columns:
            if not c.startswith('Change'):
                df[c] = _snap_noise(df[c])
        return df

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

    # ================================================================
    # Construction: __init__ and its component-recording helper
    # ================================================================

    def __init__(self, name, exp_el=0.0, exp_premium=0.0, exp_lr=0.0, exp_en=0.0, exp_attachment=None, exp_limit=np.inf,
                 sev_name='', sev_a=np.nan, sev_b=0.0, sev_mean=0.0, sev_cv=0.0, sev_loc=0.0, sev_scale=0.0,
                 sev_xs=None, sev_ps=None, sev_wt=1.0, sev_lb=0.0, sev_ub=np.inf, sev_conditional=True,
                 sev_signed=False, sev_reflect=False,
                 sev_pick_attachments=None, sev_pick_losses=None,
                 occ_reins=None, occ_kind='',
                 freq_name='', freq_a=0.0, freq_b=0.0, freq_zm=False, freq_p0=np.nan,
                 agg_reins=None, agg_kind='',
                 reins_bucket=None, dsev_bucket=None,
                 agg_premium=None, agg_reflect=False, agg_shift=0.0, value_type='loss',
                 note='', hints=''):
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
        :param sev_signed:      if True the severity is signed (never clamps its
                                negative support; a profit is a negative loss).
                                Set by the ``ssev`` DecL keyword. Orthogonal to
                                ``value_type``. ``dsev`` with a negative atom
                                self-signs regardless of this flag.
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
        :param agg_premium:     for a ``pnl`` (premium-minus-loss) aggregate, the
                                stated premium (scalar or per-component vector);
                                ``None`` for an ordinary loss aggregate. Retained
                                for the P&L ``info`` readout.
        :param agg_reflect:     if True, reflect the finished aggregate (``A -> -A``);
                                set by ``pnl`` (a profit is a negative loss).
        :param agg_shift:       deterministic shift applied to the finished
                                aggregate **once for the book** (the total premium
                                for ``pnl``); 0 for an ordinary aggregate. Together
                                with ``agg_reflect`` this is the ``pnl`` affine
                                ``PnL = agg_shift - A``.
        :param value_type:      ``'loss'`` (default) or ``'payoff'``; ``pnl`` sets
                                ``'payoff'`` (more is better). Inert for the
                                distribution, consumed at the pricing layer.
        :param note:            free-text note, from a ``note{...}`` clause
        :param hints:           raw ``hints{...}`` build-settings string
            (``key=value;`` form). Pure annotation here; the underwriter
            parses it on build (caller-supplied ``build()`` kwargs win).
        """

        # have to be ready for inputs to be in a list, e.g. comes that way from Pandas via Excel
        def get_value(v):
            if isinstance(v, list):
                return v[0]
            else:
                return v

        # class variables
        self.name = get_value(name)
        # for persistence, save the raw called spec via inspect; must call before
        # creating any other local variables.
        frame = inspect.currentframe()
        self._spec = dict(inspect.getargvalues(frame).locals)
        for n in ['frame', 'get_value', 'self']:
            if n in self._spec: self._spec.pop(n)

        logger.debug(
            f'Aggregate.__init__ | creating new Aggregate {self.name}')
        # Composition: an Aggregate *has* a frequency model, not *is* one.
        # ``Frequency(...)`` dispatches via ``__new__`` to the correct
        # ``Frequency<Kind>`` subclass.
        self.frequency = Frequency(
            get_value(freq_name), get_value(freq_a), get_value(freq_b),
            get_value(freq_zm), get_value(freq_p0))
        # Spec pass through from constructor arguments
        self.note = note
        # Raw `hints{...}` settings string; consumed by the underwriter build
        # path (caller-wins merge), retained here for round-tripping / repr.
        self.hints = hints
        self.program = ''  # can be set externally
        self.occ_reins = occ_reins
        self.occ_kind = occ_kind
        self.agg_reins = agg_reins
        self.agg_kind = agg_kind
        self.sev_pick_attachments = sev_pick_attachments
        self.sev_pick_losses = sev_pick_losses

        # Grid + runtime config (set by update / update_work)
        self.figure = None
        self.xs = None
        self.bs = 0
        self.log2 = 0
        self.padding = 0
        # Signed-support / output-window state (set by update_work). The
        # defaults reproduce the non-negative, zero-based grid exactly:
        #   x_min == 0   -> output window starts at 0 (no output roll)
        #   i0 == 0      -> severity has no negative buckets (no input roll)
        #   xs_sev == xs -> severity and output share the grid
        # See dev/plan-negative-x-agg.md (F1 negative-x, F2 output window).
        self.x_min = 0.0
        self.x_max = None
        self.i0 = 0           # index of physical 0 in the severity array
        self.xs_sev = None    # severity discretisation grid (may differ from xs)
        self._bs_window_df = None   # inspectable bucket/window estimator summary
        # F1 opt-in: when True the severity keeps its negative support (the
        # layering clamp ``x<0 -> 0`` is bypassed). Default False preserves the
        # established non-negative behaviour. Opt-in wiring is pending a design
        # decision (DecL keyword vs flag vs dsev auto-detect).
        self._signed_sev = False
        # Aggregate-level affine wrapper (the ``pnl`` premium-minus-loss form):
        # a deterministic relabel of the finished loss aggregate -- reflect
        # (a profit is a negative loss) and shift by the total premium. Applied
        # at the end of ``update_work`` (see ``_apply_agg_affine``). Defaults
        # (reflect False, shift 0) leave every ordinary aggregate byte-for-byte
        # unchanged. ``agg_premium`` is retained for the P&L ``info`` readout.
        # See dev/done/plan-pnl-premium.md.
        self._agg_premium = agg_premium
        self._agg_reflect = bool(agg_reflect)
        self._agg_shift = float(agg_shift)
        # Sign convention: how the variable is read. Inert for the
        # distribution itself; consumed at the pricing/distortion layer
        # (actuarial loss orientation). ``pnl`` sets ``payoff``. See plan §5.5.
        self._value_type = value_type if value_type in ('loss', 'payoff') else 'loss'
        self.validation_eps = get_settings().validation.eps
        self.sev_calc = ""
        self.discretization_calc = ""
        self.normalize = ""

        # Exposure / mixture outputs (filled by broadcasting below)
        self.en = None   # per-component frequency (e.g. for a limit profile)
        self.n = 0       # total frequency
        self.attachment = None
        self.limit = None
        self.sevs = None

        # Computed densities (set by update_work)
        self.sev_density = None
        self.agg_density = None
        self.ftagg_density = None
        self.fzapprox = None
        self._density_df = None
        self._sev_density_df = None

        # Empirical moment estimates (set by update_work; consumed by q / tvar)
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

        # Cached lazy functions (built on demand)
        self._valid = None
        self._var_tvar_function = None
        self._sev_var_tvar_function = None
        self._cdf = None
        self._pdf = None
        self._sev = None

        # Reinsurance state (set by apply_occ_reins / apply_agg_reins).
        # The exact (EX) reporting path reads the ceder/netter step
        # functions retained here; the per-stage reins frames
        # (``reins_density_df``, ``reins_stats_df``, ``reins_describe``)
        # are rebuilt lazily and cached in the underscore members below.
        self.occ_netter = None
        self.occ_ceder = None
        self.agg_netter = None
        self.agg_ceder = None
        self.sev_density_ceded = None
        self.sev_density_net = None
        self.sev_density_gross = None
        self.agg_density_ceded = None
        self.agg_density_net = None
        self.agg_density_gross = None
        self._reins_density_df = None
        self._reins_stats_df = None
        self._reins_view_stats_cache = None
        self._reins_describe = None
        # rebucketing scheme for reins net/ceded distributions; set the backing
        # field directly (the setter clears the caches just initialised above)
        self._reins_bucket = reins_bucket if reins_bucket is not None else get_settings().discretization.reins_bucket
        if self._reins_bucket not in ('linear', 'nearest'):
            raise ValueError(
                f"reins_bucket must be 'linear' or 'nearest', not {self._reins_bucket!r}")
        # rebucketing scheme for discrete-severity atoms (dsev/dhistogram/fixed);
        # set the backing field directly (no caches to clear at construction)
        self._dsev_bucket = dsev_bucket if dsev_bucket is not None else get_settings().discretization.dsev_bucket
        if self._dsev_bucket not in ('linear', 'nearest'):
            raise ValueError(
                f"dsev_bucket must be 'linear' or 'nearest', not {self._dsev_bucket!r}")

        # ``stats_df`` is pre-created inside each broadcasting arm below once
        # ``n_components`` is known; see ``_init_stats_df``.
        ma = MomentAggregator(self.frequency.freq_moms)

        # overall freq CV with common mixing
        mix_cv = self.frequency.freq_a

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
            logger.debug('Aggregate.__init__ | Broadcast/align: exposures + severity = %d exp = '
                         '%d sevs = %d componets', len(exp_el), len(sev_a), n_components)
            self.sevs = np.empty(n_components, dtype=type(Severity))
            # limit-profile arm: weights all 1, single severity per exposure
            # row → mixture component ``m`` is trivially 0; ``e`` indexes the
            # broadcast exposure rows.
            self._init_stats_df([f'e{e_idx}.m0' for e_idx in range(n_components)])

            # perform looping creation of severity distribution
            # in this case wts are all 1, so no need to broadcast
            for _el, _pr, _lr, _en, _at, _y, _sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _swt, _slb, _sub in all_arrays:
                assert _swt==1, 'Expect weights all equal to 1'

                # WARNING: note sev_xs and sev_ps are NOT broadcast
                self.sevs[r] = Severity(_sn, _at, _y, _sm, _scv, _sa, _sb, _sloc, _ssc, sev_xs, sev_ps,
                                        _swt, _slb, _sub, sev_conditional, sev_signed=sev_signed, sev_reflect=sev_reflect)
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
                    _en = np.sum(self.frequency.freq_a * self.frequency.freq_b)
                    _el = _en * sev1

                # scale for the mix - OK because we have split the exposure and severity components
                _pr *= _swt
                _el *= _swt
                # _lr *= _swt  ?? seems wrong
                _en *= _swt

                self._record_component(self._comp_cols[r], ma, _at, _y, _scv,
                                       _en, _el, _pr, _lr,
                                       mix_cv, sev1, sev2, sev3)
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
            # mixture-product arm: outer exposure × inner severity-mixture
            # gives a 2-D component grid; labels carry both indices.
            _n_exp = len(exp_el)
            _n_mix = len(sev_name)
            self._init_stats_df([
                f'e{e_idx}.m{m_idx}'
                for e_idx in range(_n_exp)
                for m_idx in range(_n_mix)
            ])

            # Ground-up mixture components are needed only to reweight the
            # mixture under an excess-of attachment (the ``sf(_at)`` call
            # below). Skip the constructions entirely when no exposure row
            # carries a positive attachment — saves one ``Severity`` per
            # mixture component on the common no-excess path.
            need_gup = any(
                _at is not None and _at > 0 for _at in attachment
            )
            gup_sevs = []
            if need_gup:
                for _sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _slb, _sub, _swt in zip(*sev_arrays, sev_wt):
                    gup_sevs.append(Severity(_sn, 0, np.inf, _sm, _scv, _sa, _sb, _sloc, _ssc, sev_xs, sev_ps,
                                             _swt, _slb, _sub, sev_conditional, sev_signed=sev_signed, sev_reflect=sev_reflect))

            # perform looping creation of severity distribution
            for e_idx, (_el, _pr, _lr, _en, _at, _y) in enumerate(zip(*exp_arrays)):
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
                                                _swt, _slb, _sub, sev_conditional, sev_signed=sev_signed, sev_reflect=sev_reflect))

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
                logger.info('%s xs %s, component_mean = %s, %s',
                            _y, _at, component_mean, [m[0] for m in moms])
                if _en == 0:
                    _en = _el / component_mean

                # for cases where a mixture component has no losses in the layer
                # usually because of underflow.
                zero = None

                # break up the total claim count into parts and add sevs to self.sevs
                # need the first variables for sev statistics
                for m_idx, (_sn, _sa, _sb, _sm, _scv, _sloc, _ssc, _slb, _sub, s, _swt, (sev1, sev2, sev3)) in \
                        enumerate(zip(*sev_arrays, actual_sevs, sev_wt0, moms)):

                    # store the severity
                    if np.isnan(sev1):
                        if zero is None:
                            zero = Severity('dhistogram', 0, np.inf, 0, 0, 0, 0, 0, 0, [0], [1], 0, np.inf, 0, False)
                        # replace this component with the zero distribution
                        # ignore the (small) weights that are being ignored
                        self.sevs[r] = zero
                        _sn = 'dhistogram'
                        logger.info('%s xs %s on %s x (%s, %s, %s, %s, %s) + %s '
                                    ' | %s < X le %s '
                                    'component has sev=(%s, %s, %s), '
                                    ' weight = %s; replacing with zero.',
                                    _y, _at, _ssc, _at, _sm, _scv, _sa, _sb, _sloc,
                                    _slb, _sub, sev1, sev2, sev3, _swt)
                        sev1 = sev2 = sev3 = 0.0
                    else:
                        self.sevs[r] = s

                    # input claim count, figure total loss for the component
                    if _en > 0:
                        _el = _en * sev1
                    elif _en < 0:
                        # for empirical freq claim count entered as -1
                        _en = np.sum(self.frequency.freq_a * self.frequency.freq_b)
                        _el = _en * sev1
                    else:
                        logger.info('%s xs %s on %s x (%s, %s, %s, %s, %s) + %s '
                                    ' | %s < X le %s has '
                                    '_en = %s. Adjusting el to 0.',
                                    _y, _at, _ssc, _at, _sm, _scv, _sa, _sb, _sloc,
                                    _slb, _sub, _en)
                        _el = 0.

                    # if premium compute loss ratio, if loss ratio compute premium
                    if _pr > 0:
                        _lr = _el / _pr
                    elif _lr > 0:
                        _pr = _el / _lr

                    # scale for the mix - OK because we have split the exposure and severity components
                    _pr0 = _pr * _swt
                    _el0 = _el * _swt
                    _en0 = _en * _swt

                    self._record_component(f'e{e_idx}.m{m_idx}', ma, _at, _y, _scv,
                                           _en0, _el0, _pr0, _lr,
                                           mix_cv, sev1, sev2, sev3)

                    self.en[r] = _en0
                    self.attachment[r] = _at
                    self.limit[r] = _y

                    r += 1

        # average exp_limit and exp_attachment — weighted by per-component
        # frequency mean. Sourced from the stats_df columns populated by the
        # broadcast loop above. ``stats_df`` is now all-float, so the casts
        # that this block used to need are gone.
        _comp_cols = self._comp_cols
        _comp_limit = self.stats_df.loc[('meta', 'limit'), _comp_cols]
        _comp_attach = self.stats_df.loc[('meta', 'attachment'), _comp_cols]
        _comp_freq = self.stats_df.loc[('freq', 'ex1'), _comp_cols]
        avg_limit = float(np.sum(_comp_limit * _comp_freq) / ma.tot_freq_1)
        avg_attach = float(np.sum(_comp_attach * _comp_freq) / ma.tot_freq_1)

        # store answer for total
        tot_prem = float(self.stats_df.loc[('meta', 'prem'), _comp_cols].sum())
        tot_loss = float(self.stats_df.loc[('meta', 'el'), _comp_cols].sum())
        if tot_prem > 0:
            lr = tot_loss / tot_prem
        else:
            lr = np.nan

        # Write the post-loop totals directly into ``stats_df``: per-component
        # weights, then ``mixed`` and ``independent`` columns (theoretical
        # moments + meta).
        freq_ex1 = self.stats_df.loc[('freq', 'ex1'), _comp_cols]
        self.stats_df.loc[('meta', 'wt'), _comp_cols] = (freq_ex1 / ma.tot_freq_1).values
        # mixed and independent totals
        _flat_names = MomentAggregator.column_names()
        for _col, _remix in (('mixed', True), ('independent', False)):
            for _flat, _val in zip(_flat_names, ma.get_fsa_stats(total=True, remix=_remix)):
                self.stats_df.loc[_flat_col_to_stats_index(_flat), _col] = _val
            self.stats_df.loc[('meta', 'limit'), _col] = avg_limit
            self.stats_df.loc[('meta', 'attachment'), _col] = avg_attach
            self.stats_df.loc[('meta', 'sevcv_param'), _col] = 0
            self.stats_df.loc[('meta', 'el'), _col] = tot_loss
            self.stats_df.loc[('meta', 'prem'), _col] = tot_prem
            self.stats_df.loc[('meta', 'lr'), _col] = lr
            self.stats_df.loc[('meta', 'mix_cv'), _col] = (
                float(mix_cv) if np.isscalar(mix_cv) else np.nan
            )
            self.stats_df.loc[('meta', 'wt'), _col] = float(
                self.stats_df.loc[('meta', 'wt'), _comp_cols].sum()
            )

        self.n = ma.tot_freq_1
        # Pull the headline moments off the canonical stats_df mixed column.
        _mixed = self.stats_df['mixed']
        self.agg_m = float(_mixed[('agg', 'mean')])
        self.agg_cv = float(_mixed[('agg', 'cv')])
        self.agg_skew = float(_mixed[('agg', 'skew')])
        # variance and sd come up in exam questions. Derive them directly from
        # the second moment (var = ex2 - mean^2), NOT as mean*cv: at mean 0 the
        # CV is legitimately nan, which would poison sd = mean*cv -> nan for a
        # signed (P&L) aggregate whose sd is perfectly well defined. The clamp
        # absorbs fp dust when a symmetric ex2 - mean^2 lands slightly negative.
        self.agg_var = max(float(_mixed[('agg', 'ex2')]) - self.agg_m ** 2, 0.0)
        self.agg_sd = math.sqrt(self.agg_var)
        # severity exact moments
        self.sev_m = float(_mixed[('sev', 'mean')])
        self.sev_cv = float(_mixed[('sev', 'cv')])
        self.sev_skew = float(_mixed[('sev', 'skew')])
        self.sev_var = max(float(_mixed[('sev', 'ex2')]) - self.sev_m ** 2, 0.0)
        self.sev_sd = math.sqrt(self.sev_var)

    def _init_stats_df(self, comp_cols):
        """Pre-create the empty ``stats_df`` (NaN-filled).

        Called from each broadcasting arm of ``__init__`` once the per-
        component column labels are known. Columns are:

        * the broadcast components, named ``e{e}.m{m}`` where ``e`` is the
          exposure component (one per ``(claims|premium, limit xs attach)``
          row) and ``m`` is the severity-mixture component (one per weighted
          severity); the limit-profile arm always uses ``m=0``.
        * ``mixed`` / ``independent``: theoretical (subject / gross) totals.
        * ``empirical``: post-FFT empirical moments (the final, possibly
          after-reinsurance object).
        * ``after_occ``: empirical moments after the occurrence-reinsurance
          stage (populated in meta.4; scaffold here, NaN-filled).
        * ``occ_impact`` / ``agg_impact``: ``after_occ / mixed`` and
          ``empirical / after_occ`` ratios (scaffold).
        * ``gross_empirical``: subject empirical (the reinsurance validation
          hook from §1.3 of the plan; scaffold).
        * ``error``: noise-aware relative error of ``empirical`` vs
          ``mixed``.

        ``_record_component`` writes each component column inside the
        broadcast loop; the post-loop block writes ``mixed`` /
        ``independent``; ``update_work`` writes ``empirical`` and ``error``
        after the FFT; ``after_occ`` / ``occ_impact`` / ``agg_impact`` /
        ``gross_empirical`` are populated in meta.4 (reins reporting).

        ``stats_df`` is now an all-``float64`` frame: ``self.name`` lives on
        the attribute, so no string row is needed.
        """
        self._comp_cols = list(comp_cols)
        cols = self._comp_cols + [
            'mixed', 'independent', 'after_occ', 'empirical',
            'occ_impact', 'agg_impact', 'gross_empirical', 'error',
        ]
        self.stats_df = pd.DataFrame(
            np.nan, index=_STATS_ROW_INDEX, columns=cols, dtype=float,
        )

    def _record_component(self, col, ma, attach, layer, scv, en, el, prem, lr, mix_cv,
                          sev1, sev2, sev3):
        """Accumulate this component into ``ma`` and write its ``stats_df`` column.

        Called once per component from each of the two broadcasting arms of
        ``__init__``: the limit-profile arm (all weights == 1) and the
        mixture-product arm. Centralises which ``stats_df`` per-component
        column gets written so the two arms cannot drift apart.

        Parameters
        ----------
        col : str
            Per-component column label in ``stats_df``, of the form
            ``e{e}.m{m}`` (exposure × severity-mixture). Limit-profile arm
            uses ``m=0``.
        ma : MomentAggregator
            Accumulator collecting freq, sev, agg moments across all components.
        attach, layer : float
            Per-component attachment and layer height.
        scv : float
            Severity CV parameter for this component.
        en, el, prem, lr : float
            Per-component frequency, expected loss, premium, loss ratio, already
            scaled by the mixture weight by the caller.
        mix_cv : float
            Overall mixing-distribution CV (constant across rows).
        sev1, sev2, sev3 : float
            First three raw severity moments for this component.
        """
        ma.add_f1s(en, sev1, sev2, sev3)
        moments = ma.get_fsa_stats(total=False)
        # Write this component's data directly into the canonical ``stats_df``
        # column. Maps MA's flat moment names to the ``(component, measure)``
        # MultiIndex via ``_flat_col_to_stats_index``. ``('meta', 'wt')`` is
        # filled in by the post-loop block (it depends on total frequency).
        # ``mix_cv`` is only meaningful for true mixed-Poisson frequencies; for
        # ``dfreq`` (where ``freq_a`` is the discrete pmf array) it is NaN.
        self.stats_df.loc[('meta', 'limit'), col] = layer
        self.stats_df.loc[('meta', 'attachment'), col] = attach
        self.stats_df.loc[('meta', 'el'), col] = el
        self.stats_df.loc[('meta', 'prem'), col] = prem
        self.stats_df.loc[('meta', 'lr'), col] = lr
        self.stats_df.loc[('meta', 'sevcv_param'), col] = scv
        self.stats_df.loc[('meta', 'mix_cv'), col] = (
            float(mix_cv) if np.isscalar(mix_cv) else np.nan
        )
        for flat, val in zip(MomentAggregator.column_names(), moments):
            self.stats_df.loc[_flat_col_to_stats_index(flat), col] = val

    # ================================================================
    # Repr / info / help — string and HTML representations
    # ================================================================

    def __repr__(self):
        """
        String version of _repr_html_
        :return:
        """
        return f'{self.name}, {super(Aggregate, self).__repr__()}'

    def __str__(self):
        """
        Goal: readability

        :return:
        """
        s = [self.info]
        with pd.option_context('display.width', 200,
                               'display.max_columns', 15,
                               'display.float_format', lambda x: f'{x:,.5g}'):
            # get it on one row
            s.append(str(self.describe))
        # s.append(super().__repr__())
        return '\n'.join(s)

    def help(self, regex):
        """
        Lookup help on methods and properties matching ``regex``.

        Thin wrapper over :func:`aggregate.utilities.agg_help` — the free
        function is prefixed to avoid shadowing Python's builtin ``help`` at
        module / package scope.
        """
        agg_help(self, regex)

    @property
    def info(self):
        s = [f'aggregate object name    {self.name}',
             f'claim count              {self.n:0,.2f}',
             f'frequency distribution   {self.frequency.freq_name}']
        n = len(self.sevs)
        if n == 1:
            sv = self.sevs[0]
            s.append(f'severity distribution    {sv.long_name}, {sv.support_description}.')
        else:
            s.append(f'severity distribution    {n} components')
        if self.bs > 0:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{int(1 / self.bs)}'
            s.append(f'bs                       {bss}')
            s.append(f'log2                     {self.log2}')
            s.append(f'padding                  {self.padding}')
            s.append(f'sev_calc                 {self.sev_calc}')
            # dsev_bucket governs discrete-atom placement; only meaningful (and
            # only shown) when a point-mass severity component is present.
            if any(sv.sev_kind in ('dhistogram', 'fixed') for sv in self.sevs):
                s.append(f'dsev_bucket              {self.dsev_bucket}')
            s.append(f'normalize                {self.normalize}')
            s.append(f'value_type               {self.value_type}')
            # Signed-support / output-window line: shown only when the grid is
            # offset from the default 0-based, non-negative layout.
            if self._signed_sev or self.x_min != 0:
                s.append(f'window                   [{self.x_min:,.6g}, '
                         f'{self.x_max:,.6g}]')
                s.append(f'signed severity          {self._signed_sev}')
                # Severity-vs-window status only when the *severity* is signed;
                # for a ``pnl`` (affine) aggregate the loss severity is
                # non-negative and the comparison against the P&L window is
                # uninformative, so it is skipped in favour of the readout below.
                if self._signed_sev:
                    if self._severity_in_window():
                        s.append('severity window          inside output window')
                    else:
                        s.append('severity window          OUTSIDE output window; '
                                 'see sev_density_df')
            # P&L readout: a ``pnl`` (premium-minus-loss) aggregate. ``E[margin]``
            # and ``P(loss)`` are the headline numbers an underwriter wants;
            # ``P(loss) = P(PnL < 0)`` is read straight off the signed density.
            if self._agg_affine_active() and self.agg_density is not None:
                premium = float(self._agg_shift)
                e_margin = float(self.est_m)         # PnL mean = premium - E[loss]
                e_loss = premium - e_margin
                lr = e_loss / premium if premium else np.nan
                p_loss = float(self.agg_density[self.xs < 0].sum())
                s.append(f'premium                  {premium:,.6g}')
                s.append(f'E[loss]                  {e_loss:,.6g}')
                s.append(f'E[margin]                {e_margin:,.6g}')
                s.append(f'loss ratio               {lr:.4g}')
                s.append(f'P(loss)                  {p_loss:.4g}')
            s.append(f'validation_eps           {self.validation_eps}')
            s.append(f'reinsurance              {self.reins_kinds().lower()}')
            s.append(f'occurrence reinsurance   {self.reins_description("occ").lower()}')
            s.append(f'aggregate reinsurance    {self.reins_description("agg").lower()}')
            s.append(f'validation               {self.explain_validation()  }')
            # Tail-thickness classification (frequency / severity / aggregate).
            s.extend(_tail.describe_lines(
                self._tail_info(), self.frequency.freq_name, self._sev_label()))
            s.append('')
        return '\n'.join(s)

    def explain_validation(self):
        """
        Explain validation result. Validation computed if needed.
        """
        return explain_validation(self.valid)

    def _html_info_blob(self):
        """
        Text top of _repr_html_

        """
        s = [f'<h3>Aggregate object: {self.name}</h3>']
        s.append(f'<p>{self.frequency.freq_name} frequency distribution.')
        n = len(self.sevs)
        if n == 1:
            sv = self.sevs[0]
            s.append(f'Severity {sv.long_name} distribution, {sv.support_description}.')
        else:
            s.append(f'Severity with {n} components.')
        if self.bs > 0:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{1 / self.bs:,.0f}'
            s.append(f'Updated with bucket size {bss} and log2 = {self.log2}.</p>')
        if self.agg_density is not None:
            r = self.valid
            if r == Validation.NOT_UNREASONABLE:
                s.append('<p>Validation: not unreasonable.</p>')
            elif r == Validation.REINSURANCE:
                # Reins present, subject validated cleanly.
                s.append('<p>Validation: reinsurance; subject not unreasonable.</p>')
            else:
                s.append('<p>Validation: <div style="color: #f00; font-weight:bold;">fails</div><pre>\n'
                         f'{self.explain_validation()}</pre></p>')

        return '\n'.join(s)

    def _repr_html_(self):
        """
        For IPython.display

        """
        return self._html_info_blob() + self.describe.to_html()

    # ================================================================
    # Discretization, snap, update, FFT convolution
    # The 5-line FFT core (Mildenhall 2024, §2.2) lives in
    # ``_freq_sev_convolution`` below.
    # ================================================================

    def _agg_affine_active(self):
        """Whether the aggregate-level affine (``pnl``) wrapper is active.

        ``True`` when the finished aggregate is reflected and/or shifted -- the
        ``pnl`` premium-minus-loss form (``PnL = agg_shift - A``). The defaults
        (reflect ``False``, shift ``0``) are inert, so an ordinary aggregate
        returns ``False`` and every affine code path is bypassed.

        Returns
        -------
        bool
        """
        return bool(getattr(self, '_agg_reflect', False)) or \
            float(getattr(self, '_agg_shift', 0.0)) != 0.0

    def _signed_severity(self):
        """Whether the aggregate has signed (negative-support) severity.

        Returns ``True`` iff any component severity is signed -- declared with
        the ``ssev`` keyword (continuous) or a ``dsev`` with a negative atom,
        both recorded on ``Severity.signed`` at construction.

        This is the *convolution-grid* gate: it drives the negative-x severity
        layout (``i0``), the two-sided window in :meth:`_bs_window`, and
        ``_signed_sev``. It is **not** the public display/combine gate -- a
        ``pnl`` aggregate has an ordinary non-negative loss severity (so this is
        ``False``) but is signed at the aggregate level via the affine wrapper;
        see :meth:`_signed`.

        Returns
        -------
        bool

        Notes
        -----
        Signedness is a *parse-time property of the severity*, not a runtime
        flag -- which is what lets the analytic moments (and hence the auto
        window) be correct before any FFT, and is why there is **no**
        ``signed=`` override on ``update``: a built ``Severity`` is already
        clamped or not, and flipping it would require a deep rebuild. To change
        signedness, change the declaration (``sev`` / ``dsev`` / ``ssev``).
        See ``dev/plan-negative-x-agg.md``.
        """
        return any(getattr(s, 'signed', False)
                   for s in (self.sevs if self.sevs is not None else []))

    def _signed(self):
        """Whether the aggregate is signed (straddles 0) for display / combine.

        ``True`` when the severity itself reaches below 0
        (:meth:`_signed_severity`) **or** the aggregate-level affine wrapper is
        active (:meth:`_agg_affine_active` -- the ``pnl`` premium-minus-loss
        form, whose loss severity is non-negative but whose result straddles 0).
        This is the gate read by plotting, two-sided quantiles, the
        signed-aware ``describe`` (SD instead of CV), and the
        :class:`Portfolio` combine.

        Returns
        -------
        bool
        """
        return self._signed_severity() or self._agg_affine_active()

    def _severity_negative_buckets(self, bs):
        """Number of negative buckets the severity reaches (index of physical 0).

        Returns ``i0`` such that the severity discretisation grid
        ``xs_sev = (arange(N) - i0) * bs`` places physical 0 at index ``i0``
        and covers the severity's left tail. ``0`` for any severity supported
        on ``[0, inf)`` -- which keeps the non-negative path byte-for-byte
        unchanged. For a severity that reaches below 0 (e.g. ``norm``, a
        shifted distribution, or a ``dsev`` with negative atoms) it is the
        number of buckets from 0 down to the severity's effective lower
        endpoint.

        Parameters
        ----------
        bs : float
            Bucket size.

        Returns
        -------
        int
            ``max(0, ceil(-lo / bs))`` where ``lo`` is the smallest effective
            lower endpoint over the severity mixture components. A small
            tolerance keeps floating-point dust at exactly 0 from spuriously
            triggering signed mode.

        Notes
        -----
        ``lo`` is taken as the ``1e-12`` lower quantile of each *signed*
        component (a signed Severity has identity layering, so its ``ppf`` is
        the raw lower quantile), mirroring how the upper grid edge is sized from
        a high quantile. Any mass below the leftmost bucket is dropped and
        absorbed by the (optional) renormalisation in ``discretize`` -- at the
        ``1e-12`` level this is negligible.
        """
        # Only signed severities reach below 0; an unsigned component keeps the
        # clamp-at-0 layering (no negative buckets). i0 == 0 unless the
        # aggregate is signed, so the non-negative path is untouched.
        if not getattr(self, '_signed_sev', False):
            return 0
        if self.sevs is None or len(self.sevs) == 0:
            return 0
        los = []
        for sev in self.sevs:
            if not getattr(sev, 'signed', False):
                continue
            try:
                lo = float(sev.ppf(1e-12))   # identity layering -> raw quantile
            except Exception:  # pragma: no cover - defensive
                lo = 0.0
            if not np.isfinite(lo):
                lo = 0.0
            los.append(lo)
        lo = min(los) if los else 0.0
        if lo >= 0:
            return 0
        i0 = int(np.ceil(-lo / bs - 1e-9))
        # The severity's negative reach must fit inside the grid.
        N = len(self.xs)
        if i0 >= N:
            logger.warning(
                '%s: severity negative reach (%d buckets) exceeds grid size '
                '%d; clipping. Increase log2 or bs for a signed aggregate.',
                self.name, i0, N)
            i0 = N - 1
        return i0

    def discretize(self, sev_calc, discretization_calc, normalize):
        """
        Discretize the severity distributions and weight.

        ``sev_calc`` describes how the severity is discretize, see `Discretizing the Severity Distribution`_. The
        options are discrete=round, forward, backward or moment.

        ``sev_calc='continuous'`` (same as forward, kept for backwards compatibility) is used when
        you think of the resulting distribution as continuous across the buckets
        (which we generally don't). The buckets are not shifted and so :math:`Pr(X=b_i) = Pr( b_{i-1} < X \\le b_i)`.
        Note that :math:`b_{i-1}=-bs/2` is prepended.

        We use the discretized distribution as though it is fully discrete and only takes values at the bucket
        points. Hence, we should use `sev_calc='discrete'`. The buckets are shifted left by half a bucket,
        so :math:`Pr(X=b_i) = Pr( b_i - b/2 < X \\le b_i + b/2)`.

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

        # Severity is discretised on ``xs_sev`` (physical 0 at index i0), which
        # equals ``self.xs`` on the default 0-based grid. ``i0 > 0`` means the
        # severity reaches below 0 (signed mode).
        xs_sev = self.xs_sev if self.xs_sev is not None else self.xs
        signed = self.i0 > 0

        if sev_calc == 'discrete' or sev_calc == 'round':
            # adj_xs = np.hstack((xs_sev - self.bs / 2, np.inf))
            # mass at the end undesirable. can be put in with reinsurance layer in spec
            # note the first bucket is negative
            adj_xs = np.hstack((xs_sev - self.bs / 2, xs_sev[-1] + self.bs / 2))
        elif sev_calc == 'forward' or sev_calc == 'continuous':
            adj_xs = np.hstack((xs_sev, xs_sev[-1] + self.bs))
        elif sev_calc == 'backward':
            adj_xs = np.hstack((xs_sev[0] - self.bs, xs_sev))  # , np.inf))
        elif sev_calc == 'moment':
            raise NotImplementedError(
                'Moment matching discretization not implemented. Embrechts says it is not worth it.')
            #
            # adj_xs = np.hstack((xs_sev, np.inf))
        else:
            raise ValueError(
                f'Invalid parameter {sev_calc} passed to discretize; options are discrete, continuous, or raw.')

        if not signed:
            # Non-negative severity: the first bucket must include all mass at
            # and below 0. Capture the whole left tail from -inf, exactly as
            # before (byte-for-byte unchanged on the default path).
            adj_xs[0] = -np.inf
        # Signed mode: the leftmost bucket is treated identically to every
        # other bucket (a finite ``xs_sev[0] - bs/2`` edge, set above) -- no
        # -inf catch. Any residual mass below it (<= 1e-12 by construction of
        # i0) is dropped and absorbed by the optional renormalisation below.

        # bed = bucketed empirical distribution. A signed Severity carries
        # identity layering, so its own cdf/sf are already un-clamped; an
        # unsigned component keeps the clamp-at-0 layering. So per-component
        # ``sev.cdf`` / ``sev.sf`` are correct on the signed grid with no special
        # casing (an unsigned component simply contributes 0 to negative
        # buckets). Occurrence reinsurance on a signed severity is out of scope
        # (plan §6).
        beds = []
        for sev in self.sevs:
            if (self.dsev_bucket == 'linear'
                    and sev.sev_kind in ('dhistogram', 'fixed')
                    and sev.exp_attachment is None):
                # Unlayered discrete severity: place the atoms on the grid with
                # the mean-preserving linear scatter, so the discretized first
                # moment equals Σ xₖ pₖ exactly. The default cdf-difference path
                # (below) snaps each atom to its nearest bucket (== 'nearest'),
                # biasing the mean by up to bs/2 per atom when atoms are off-grid
                # (empirical samples, non-integer bs). On-grid atoms give f == 0
                # so this reduces to nearest -- the dice / integer-bs case is
                # unchanged. ``sev.fz`` is the _DiscreteRV holding the validated,
                # sorted atoms (xk) and masses (pk). ``exp_attachment is None``
                # is the truly-unlayered test (a discrete sev always has a finite
                # ``detachment`` = max atom, so a ``== np.inf`` test never fires).
                # Layered discrete severities fall through to the cdf-diff path
                # (Phase 1; see dsev_bucket). The bed is indexed on ``xs_sev``,
                # so the scatter origin is the severity grid origin ``xs_sev[0]``
                # (== -i0·bs), NOT the output-window origin ``x_min`` -- the two
                # differ when the output window is forced wider than the atom
                # support (e.g. an explicit ``x_min`` below the smallest atom).
                appx = self._rebucket_to_grid(sev.fz.xk, sev.fz.pk,
                                              scheme='linear', origin=xs_sev[0])
            elif discretization_calc == 'both':
                # see comments: we rescale each severity...
                appx = np.maximum(np.diff(sev.cdf(adj_xs)), -np.diff(sev.sf(adj_xs)))
            elif discretization_calc == 'survival':
                appx = -np.diff(sev.sf(adj_xs))
                # beds.append(appx / np.sum(appx))
            elif discretization_calc == 'distribution':
                appx = np.diff(sev.cdf(adj_xs))
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

    @property
    def value_type(self):
        """Sign convention for the variable: ``'loss'`` or ``'payoff'``.

        Records how the aggregate should be read -- actuarial **loss**
        ("more is worse", the default) vs. **payoff** / asset ("more is
        better"). It is **inert for the distribution itself**: density,
        moments, quantiles, deficit and plotting do not depend on it.
        It is consumed only when applying distortions / pricing (specified
        in the Portfolio plan and downstream pricing work), where a
        ``'payoff'`` object is negated / the dual distortion applied.

        See ``dev/plan-negative-x-agg.md`` §5.5.
        """
        return self._value_type

    @value_type.setter
    def value_type(self, v):
        if v not in ('loss', 'payoff'):
            raise ValueError(
                f"value_type must be 'loss' or 'payoff', not {v!r}")
        self._value_type = v

    def update(self, log2=16, bs=0, bucket_sizing_p=BUCKET_SIZING_P, debug=False,
               x_min='auto', x_max=None, **kwargs):
        """
        Convenience function, delegates to update_work. Avoids having to pass xs. Also
        aliased as easy_update for backward compatibility.

        :param log2:
        :param bs:
        :param bucket_sizing_p: p value passed to recommend_bucket. If > 1 converted to 1 - 10**-p in rec bucket.
        :param debug:
        :param x_min: lower edge of the output window. ``'auto'`` (default)
          resolves to ``0`` for an ordinary non-negative aggregate (today's
          behaviour) and to ``None`` (automatic two-sided window) for a signed
          aggregate -- so a P&L declared with ``ssev`` / negative ``dsev`` just
          works from ``build``. ``None`` forces the automatic window; a number
          forces that origin (snapped to a multiple of ``bs``; may be negative).
        :param x_max: upper edge of the output window; informational, the grid
          length is fixed by ``log2``. Currently unused when ``x_min`` is given
          explicitly (the window is ``[x_min, x_min + (2**log2)*bs)``).
        :param kwargs:  passed through to update
        :return:

        Signedness is carried by the severity declaration (``ssev`` /negative
        ``dsev``), not by an argument here -- see ``_signed``.
        """
        # Unified bucket + window estimator: runs the candidate sizing methods
        # (moment / exact_discrete / bounded_small), records them in the
        # inspectable ``self._bs_window_df``, and selects (see ``_bs_window``).
        # ``x_min='auto'`` lets the selected method choose the origin (0 for a
        # non-negative aggregate, a negative origin only when signed); a number
        # forces it; ``log2`` is a cap. The legacy non-negative ``moment`` path
        # reproduces ``recommend_bucket`` exactly, so ordinary aggregates are
        # unchanged.
        x_min_arg = None if (isinstance(x_min, str) and x_min == 'auto') else x_min
        bs, log2, x_min = self._bs_window(log2, bs, x_min_arg, bucket_sizing_p)
        N = 1 << log2
        # ``x_min`` is the loss-convolution origin chosen by ``_bs_window``: 0 for
        # an ordinary aggregate or a non-negative-loss ``pnl``; a negative origin
        # when the severity is signed -- including a signed-loss ``pnl``, whose
        # loss convolves on its genuine signed grid before ``_apply_agg_affine``
        # relabels it onto the P&L window. The affine path no longer hard-codes a
        # 0-based grid (which silently wrapped a signed loss's negative atoms to
        # the top of the FFT buffer and dropped half the mass).
        xs = x_min + np.arange(0, N, dtype=float) * bs
        return self.update_work(xs, debug=debug, x_min=x_min, x_max=x_max,
                                **kwargs)

    def update_work(self, xs, padding=1, sev_calc='discrete',
                    discretization_calc='survival', normalize=True, force_severity=False,
                    reins_bucket=None, dsev_bucket=None, debug=False, x_min=0, x_max=None):
        """
        Compute a discrete approximation to the aggregate density via FFT.

        See discretize for sev_calc, discretization_calc and normalize.

        Empirical-moment note: the aggregate raw moments -- and hence the
        empirical CV/skew shown in ``stats_df`` and ``describe`` -- are taken
        from a de-fuzzed *copy* of the FFT density (values below machine
        epsilon zeroed). Without this, sub-eps floating-point fuzz in far-tail
        buckets is amplified by ``x**3`` in the third moment and corrupts the
        empirical skew on wide grids: a symmetric distribution's skew can
        drift from ~1e-15 to ~1e-4 as log2 grows, purely from buckets the
        distribution never reaches. The fuzz is safe to drop because the FFT
        is exact up to rounding and the exact aggregate has no negative density
        even under aliasing, so any stray value is small. ``self.agg_density``
        is deliberately left as the raw FFT output (consistent with
        ``ftagg_density``); only the moment computation sees the cleaned copy.
        See the inline comment at the moment computation for full detail.

        Quick simple test with log2=13 update took 5.69 ms and _eff took 2.11 ms. So quicker
        but not an issue unless you are doing many buckets or aggs.

        :param xs: range of x values used to discretize
        :param padding: for FFT calculation
        :param sev_calc:  discrete=round, forward, backward, or continuous
               and method becomes discrete otherwise
        :param discretization_calc:  survival, distribution or both; in addition
               the method then becomes survival
        :param normalize: if True, normalize the severity so sum probs = 1. This is generally what you want; but
               when dealing with thick tailed distributions it can be helpful to turn it off.
        :param force_severity: make severities for plotting even when only the aggregate is requested
        :param reins_bucket: optional override of the net/ceded rebucketing scheme
               ('linear' or 'nearest'); defaults to the current ``self.reins_bucket``.
        :param dsev_bucket: optional override of the discrete-severity atom
               placement scheme ('linear' or 'nearest'); defaults to the current
               ``self.dsev_bucket``. See :attr:`dsev_bucket`.
        :param debug: run reinsurance in debug model if True.
        :param x_min: ``None`` requests an automatic two-sided output window
          (NYI in this stage -- treated as the grid origin ``xs[0]``);
          otherwise informational (the grid origin is read from ``xs[0]``).
        :param x_max: informational upper window edge.
        :return:
        """
        self._density_df = None  # invalidate
        self._sev_density_df = None
        self._reins_density_df = None
        self._reins_stats_df = None
        self._reins_view_stats_cache = None
        self._reins_describe = None
        self._var_tvar_function = None
        self._sev_var_tvar_function = None
        self._valid = None
        self.sev_calc = sev_calc
        self.discretization_calc = discretization_calc
        self.normalize = normalize
        self.padding = padding
        if reins_bucket is not None:
            # validating setter; takes effect for the reins applied below
            self.reins_bucket = reins_bucket
        if dsev_bucket is not None:
            # validating setter; takes effect for the discretization below
            self.dsev_bucket = dsev_bucket
        self.xs = xs
        # bs is the grid step; xs[1]-xs[0] (not xs[1]) so a signed/offset grid
        # whose origin xs[0] != 0 still reports the correct bucket size.
        self.bs = xs[1] - xs[0]
        self.log2 = int(np.log2(len(xs)))
        # Output-window origin: the physical value at output index 0. For the
        # default 0-based grid this is 0 and all the offset machinery below is
        # inert (i0 == 0, j0 == 0 -> no rolls; xs_sev == xs).
        self.x_min = float(xs[0])
        self.x_max = float(xs[-1])

        # F1 -- negative-support severity. Resolve the signed opt-in (auto for
        # discrete severities with negative atoms; explicit flag otherwise),
        # then determine the severity's negative reach as a whole number of
        # buckets ``i0`` (the index of physical 0 in the severity array). Zero
        # for any severity supported on [0, inf), so the non-negative path is
        # untouched. The severity is then discretised on its own grid
        # ``xs_sev`` (physical 0 at index i0), which may differ from the output
        # grid ``xs`` (e.g. a tight far-from-0 output window).
        self._signed_sev = self._signed_severity()
        self.i0 = self._severity_negative_buckets(self.bs)
        self.xs_sev = (np.arange(len(xs), dtype=float) - self.i0) * self.bs

        # claim-count weighted severity vector (always computed; FFT is the only path)
        freq_ex1 = self.stats_df.loc[('freq', 'ex1'), self._comp_cols].values
        wts = freq_ex1 / freq_ex1.sum()
        if self.en.sum() == 0:
            self.en = freq_ex1
        self.sev_density = np.zeros_like(xs)
        beds = self.discretize(sev_calc, discretization_calc, normalize)
        for temp, w, a, l, n in zip(beds, wts, self.attachment, self.limit, self.en):
            self.sev_density += temp * w

        # adjust for picks if necessary
        if self.sev_pick_attachments is not None:
            logger.warning('Adjusting for picks.')
            self.sev_density = self.picks(self.sev_pick_attachments, self.sev_pick_losses)

        if force_severity == 'yes':
            # only asking for severity (used by plot)
            return

        # deal with per occ reinsurance
        if self.occ_reins is not None:
            if self.sev_density_gross is not None:
                # re-applying reins on an already-updated object: restore
                # gross sev so apply_occ_reins is idempotent
                self.sev_density = self.sev_density_gross
            self.apply_occ_reins(debug)

        self._freq_sev_convolution(padding)
        if self.n > 0:
            # zero-risk case has no aggregate to reinsure
            self.apply_agg_reins(debug)

        # Empirical severity moments from the discretised distribution.
        # Compute the raw moments once and derive (mean, cv, skew) from the
        # same MomentWrangler, so the ex123 rows and the mcvsk values are
        # mutually consistent.
        if self.sev_density is not None:
            # severity lives on xs_sev (== xs on the default grid)
            _mw = xsden_to_mwrangler(self.xs_sev, self.sev_density)
            sev_ex1, sev_ex2, sev_ex3 = _mw.noncentral
            self.est_sev_m, self.est_sev_cv, self.est_sev_skew = _mw.mcvsk
            # var/sd straight off the wrangler (var = central[1]), never via
            # mean*cv -- see the theoretical site above for the mean-0 rationale.
            self.est_sev_var = max(float(_mw.central[1]), 0.0)
            self.est_sev_sd = math.sqrt(self.est_sev_var)
        else:
            sev_ex1 = sev_ex2 = sev_ex3 = np.nan
            self.est_sev_m = np.nan
            self.est_sev_cv = np.nan
            self.est_sev_skew = np.nan
            self.est_sev_var = np.nan
            self.est_sev_sd = np.nan

        # Empirical aggregate moments from the FFT output.
        #
        # WHY a de-fuzzed *copy*: the raw inverse-FFT density carries
        # sub-machine-epsilon "fuzz" (tiny +/- values) in essentially every
        # bucket. In the plain mass sum this cancels (mass is conserved), but
        # the raw moments weight each bucket by ``x**k``, so on a wide grid the
        # far-tail fuzz at large ``x`` is amplified by ``x**3`` and corrupts
        # the empirical skew -- e.g. a symmetric die's skew drifts from ~1e-15
        # to ~1e-4 as log2 grows, purely from fuzz at buckets the distribution
        # never reaches. The fuzz is genuine fp noise: the FFT is exact up to
        # rounding and the exact aggregate has no negative density even under
        # aliasing (aliasing only wraps *positive* mass), so every stray value
        # is small and zeroing ``|x| < eps`` is safe and lossless. We do this
        # on a throwaway copy and deliberately leave ``self.agg_density`` as
        # the raw output (kept consistent with ``ftagg_density``); the curated
        # view ``density_df.p_total`` applies the identical ``remove_fuzz``
        # separately. (We cannot source the moments from ``density_df.p_total``
        # here: building ``density_df`` needs ``est_m``, computed just below.)
        agg_clean = remove_fuzz(self.agg_density)
        _mw = xsden_to_mwrangler(self.xs, agg_clean)
        agg_ex1, agg_ex2, agg_ex3 = _mw.noncentral
        self.est_m, self.est_cv, self.est_skew = _mw.mcvsk
        # var/sd straight off the wrangler (var = central[1]), never via
        # mean*cv -- correct for a mean-0 signed aggregate (see above).
        self.est_var = max(float(_mw.central[1]), 0.0)
        self.est_sd = math.sqrt(self.est_var)

        # Write empirical and error columns into the canonical stats_df.
        # This is the validation showpiece of Mildenhall 2024, §4.7:
        # theoretical (``mixed`` column) vs. empirical (FFT output).
        self.stats_df.loc[('sev', 'ex1'),  'empirical'] = sev_ex1
        self.stats_df.loc[('sev', 'ex2'),  'empirical'] = sev_ex2
        self.stats_df.loc[('sev', 'ex3'),  'empirical'] = sev_ex3
        self.stats_df.loc[('sev', 'mean'), 'empirical'] = self.est_sev_m
        self.stats_df.loc[('sev', 'cv'),   'empirical'] = self.est_sev_cv
        self.stats_df.loc[('sev', 'skew'), 'empirical'] = self.est_sev_skew
        self.stats_df.loc[('agg', 'ex1'),  'empirical'] = agg_ex1
        self.stats_df.loc[('agg', 'ex2'),  'empirical'] = agg_ex2
        self.stats_df.loc[('agg', 'ex3'),  'empirical'] = agg_ex3
        self.stats_df.loc[('agg', 'mean'), 'empirical'] = self.est_m
        self.stats_df.loc[('agg', 'cv'),   'empirical'] = self.est_cv
        self.stats_df.loc[('agg', 'skew'), 'empirical'] = self.est_skew

        # Defective-distribution check. The aggregate FFT loses mass off the
        # right end of the grid when log2 is too small. A genuine deficit
        # (above VALIDATION_NOISE, well clear of fp dust) makes forwards and
        # backwards S diverge by exactly the deficit in Distortion.price,
        # so surface it loudly at construction time rather than have one
        # answer silently differ from another downstream.
        deficit = 1.0 - float(np.sum(self.agg_density))
        if deficit > VALIDATION_NOISE:
            warnings.warn(
                f'{self.name}: aggregate PMF deficit {deficit:.3e} '
                f'(Σp = 1 − {deficit:.3e} < 1); forwards and backwards '
                f'S diverge by the deficit (forwards > backwards).',
                DefectiveDistributionWarning, stacklevel=2)

        # Staged reinsurance reporting -- §1.2 of the aggregate refactor plan.
        #
        # ``empirical`` above is the final (after-occ + after-agg) realised
        # view. To express the Subject -> after-occ -> after-agg progression
        # we also need the subject (gross) empirical moments and -- when
        # reinsurance is present -- the intermediate after-occ moments.
        # Validation continues to use the subject vs theoretical comparison,
        # which is the only apples-to-apples check available under reins.
        has_occ = self.occ_reins is not None
        has_agg = self.agg_reins is not None

        # Subject severity density: when occ-reins applied, the pre-reins
        # severity is preserved on ``sev_density_gross``; otherwise the
        # current ``sev_density`` IS gross.
        subject_sev = self.sev_density_gross if has_occ else self.sev_density
        # After-occ severity is whatever the occ stage passed along
        # (= ``sev_density`` post ``apply_occ_reins``); identical to subject
        # severity when there is no occ stage.
        after_occ_sev = self.sev_density

        # Subject aggregate: with occ-reins we need one extra FFT of the
        # gross severity (the "validate the subject" hook in §1.3); with no
        # occ-reins the pre-agg-reins density already encodes gross
        # (``agg_density_gross`` when has_agg, else the final
        # ``agg_density``).
        if has_occ:
            subject_agg, _ = self._fft_aggregate(subject_sev, padding)
        elif has_agg:
            subject_agg = self.agg_density_gross
        else:
            subject_agg = self.agg_density

        # After-occ aggregate (pre-agg-reins): when an agg stage exists,
        # ``apply_agg_reins`` stored the pre-stage density in
        # ``agg_density_gross``; with only occ-reins the final
        # ``agg_density`` IS the after-occ density; with no reins there is
        # no separate stage to report.
        if has_agg:
            after_occ_agg = self.agg_density_gross
        elif has_occ:
            after_occ_agg = self.agg_density
        else:
            after_occ_agg = None

        # De-fuzzed moment helper: same |x| < eps zeroing the main
        # empirical block uses (see WHY comment above), wrapped so we can
        # reuse it on subject / after-occ densities.
        def _moments(arr, grid):
            # ``grid`` is xs_sev for severity densities, xs for aggregates
            # (they coincide on the default 0-based grid).
            if arr is None:
                return (np.nan,) * 6
            mw = xsden_to_mwrangler(grid, remove_fuzz(arr))
            return (*mw.noncentral, *mw.mcvsk)

        sub_sev_mom = _moments(subject_sev, self.xs_sev)
        sub_agg_mom = _moments(subject_agg, self.xs)
        self._write_stage_moments('gross_empirical', sub_sev_mom, sub_agg_mom,
                                  copy_freq_from='empirical')

        if has_occ or has_agg:
            aft_sev_mom = _moments(after_occ_sev, self.xs_sev)
            aft_agg_mom = _moments(after_occ_agg, self.xs)
            self._write_stage_moments('after_occ', aft_sev_mom, aft_agg_mom,
                                      copy_freq_from='empirical')

        # Per-stage impact ratios (after / before). 1.0 means no impact;
        # written only when the corresponding stage exists, so consumers can
        # detect "stage absent" by NaN.
        if has_occ:
            self.stats_df['occ_impact'] = (
                self.stats_df['after_occ'] / self.stats_df['mixed'])
        if has_agg:
            self.stats_df['agg_impact'] = (
                self.stats_df['empirical'] / self.stats_df['after_occ'])

        # ``error`` is the SUBJECT validation: gross_empirical vs mixed.
        # With no reinsurance gross_empirical == empirical and this is
        # exactly the legacy theoretical-vs-empirical column. With reins it
        # is the only apples-to-apples check (the after-reins object has no
        # independent theoretical to validate against).
        self.stats_df['error'] = _noise_aware_rel_error(
            self.stats_df['gross_empirical'], self.stats_df['mixed'])

        # Aggregate affine (``pnl``): relabel the finished loss aggregate to the
        # P&L distribution ``agg_shift - A``. Inert (no-op) for every ordinary
        # aggregate; applied after the loss validation above so ``valid`` still
        # checks the loss FFT. See ``_apply_agg_affine``.
        if self._agg_affine_active():
            self._apply_agg_affine()

        # invalidate stored functions
        self._cdf = None

    def _pnl_window(self, bs, N):
        """Lower edge of the tight, mass-centred P&L output window.

        The ``pnl`` affine reflects + shifts the loss aggregate; the resulting
        P&L distribution lives around ``agg_shift - E[A]`` with the loss
        spread. ``estimate_agg_window`` on the **affine moments** gives a tight
        two-sided window ``[lo, hi]`` (mean possibly < 0); this returns ``lo``
        snapped down to ``bs`` and fitted to the ``N``-bucket grid. Centring the
        window on the mass (rather than reversing the full loss grid, whose
        empty far tail would push the origin far below the mass) is what lets a
        :class:`Portfolio` of ``pnl`` units position the shared grid correctly.

        Parameters
        ----------
        bs : float
            Bucket size.
        N : int
            Number of grid buckets (``2**log2``).

        Returns
        -------
        float
            The snapped P&L window origin ``x_min``.
        """
        reflect = self._agg_reflect
        shift = self._agg_shift
        m = (shift - self.agg_m) if reflect else (shift + self.agg_m)
        sd = self.agg_sd
        skew = -self.agg_skew if reflect else self.agg_skew
        p = 1.0 - 10.0 ** -WINDOW_NINES
        if np.isfinite(sd) and sd > 0:
            try:
                lo, hi, _W = estimate_agg_window(m, sd, skew, p)
            except (ValueError, FloatingPointError):  # pragma: no cover - defensive
                lo, hi = m - 8.0 * sd, m + 8.0 * sd
        else:
            # No finite spread (point mass or undefined variance): fall back to
            # the full reversed-grid window (origin a whole grid below shift).
            s0 = int(round(shift / bs)) if bs else 0
            return float((s0 - N + 1) * bs)
        if hi - lo <= N * bs:
            pnl_lo = float(np.floor(lo / bs) * bs) if bs else float(lo)
        else:
            # Support wider than the grid: centre on the mean, accept a small
            # two-sided deficit (the heavy-tail residual, documented).
            pnl_lo = float(np.floor((m - 0.5 * N * bs) / bs) * bs) if bs else float(m - 0.5 * N * bs)
        return pnl_lo

    def _apply_agg_affine(self):
        """Relabel the finished loss aggregate to ``agg_shift - A`` (the ``pnl`` form).

        ``pnl`` premium-minus-loss aggregates are an aggregate-level affine
        transform of an ordinary loss aggregate: reflect (a profit is a negative
        loss) and a single deterministic shift by the total premium. This is a
        pure grid relabel of the finished density -- **no new convolution** --
        so the loss FFT, its validation, and every ordinary aggregate are
        untouched.

        Notes
        -----
        The loss density ``a`` lives on the non-negative grid ``self.xs``
        (origin ``loss_x0``, index ``j_loss = round(loss_x0/bs)``). With the
        shift snapped to the bucket (``s0 = round(agg_shift/bs)``) the P&L value
        of loss bucket ``k`` is ``(s0 - j_loss - k)*bs`` (reflect) and is placed
        onto the tight P&L grid ``xs = pnl_lo + arange(N)*bs`` (see
        :meth:`_pnl_window`). The map ``t = (s0 - j_loss - j_pnl) - k`` is a
        one-to-one reverse-and-roll; buckets falling outside the window are
        dropped (a negligible far-tail deficit).

        - **ftagg_density** is rebuilt as the rfft of the length-``M`` P&L buffer
          (physical 0 at index 0, negatives wrapped to the top) by scattering
          the P&L density at indices ``(j_pnl + arange(N)) mod M``. This is the
          exact convention the :class:`Portfolio` signed combine multiplies, so
          a book of ``pnl`` units convolves correctly with no combine-side
          change.
        - **moments (closed form):** ``mean -> shift - E[A]``, ``sd`` unchanged,
          ``skew -> -skew``. Applied to the empirical scalars (``est_*``); the
          theoretical ``stats_df`` columns are deliberately left in loss terms
          (so ``valid`` / ``_bs_window`` stay loss-correct) and the P&L view is
          formed at display time in :meth:`_describe_signed`.

        Idempotent across re-updates: ``_freq_sev_convolution`` regenerates the
        loss density / ft fresh each ``update_work`` before this runs.
        """
        N = len(self.xs)
        bs = self.bs
        padding = self.padding
        reflect = self._agg_reflect
        shift = self._agg_shift
        loss_d = self.agg_density
        loss_x0 = float(self.xs[0])
        j_loss = int(round(loss_x0 / bs)) if bs else 0
        s0 = int(round(shift / bs)) if bs else 0

        pnl_lo = self._pnl_window(bs, N)
        j_pnl = int(round(pnl_lo / bs)) if bs else 0

        # ---- density + output grid (tight signed P&L window) ------------
        # one-to-one reverse-and-roll; out-of-window buckets dropped.
        k = np.arange(N)
        if reflect:
            t = (s0 - j_loss - j_pnl) - k
        else:
            t = (s0 + j_loss - j_pnl) + k
        valid = (t >= 0) & (t < N)
        pnl_d = np.zeros(N)
        pnl_d[t[valid]] = loss_d[k[valid]]
        # The reverse-and-roll drops buckets that fall outside the tight P&L
        # window (``_pnl_window``). For an ordinary or correctly-sized signed
        # ``pnl`` this sheds only far-tail dust; a *material* deficit means the
        # window cannot represent the spread (e.g. log2 too small for a wide
        # signed loss) -- surface it the same way the loss FFT deficit is flagged
        # at construction, rather than letting one answer silently differ.
        dropped = float(loss_d.sum() - pnl_d.sum())
        if dropped > VALIDATION_NOISE:
            warnings.warn(
                f'{self.name}: P&L affine dropped mass {dropped:.3e} outside the '
                f'output window [{pnl_lo}, {pnl_lo + (N - 1) * bs}]; widen log2 or '
                f'set bs/x_min to cover the signed P&L spread.',
                DefectiveDistributionWarning, stacklevel=2)
        self.agg_density = pnl_d
        self.xs = pnl_lo + np.arange(N, dtype=float) * bs
        self.x_min = float(self.xs[0])
        self.x_max = float(self.xs[-1])

        # ---- ftagg_density: rfft of the P&L length-M buffer -------------
        # (physical 0 at index 0, negatives wrapped to the top -- the combine
        # convention). Scatter the P&L density at indices (j_pnl + k) mod M.
        M = N << padding
        h = np.zeros(M)
        h[(j_pnl + np.arange(N)) % M] = pnl_d
        self.ftagg_density = sfft.rfft(h)

        # ---- empirical moment scalars (closed-form affine) --------------
        # mean -> shift - E[A]; sd unchanged; skew sign flips under reflection.
        if reflect:
            self.est_m = self._agg_shift - self.est_m
            self.est_skew = -self.est_skew
        else:
            self.est_m = self._agg_shift + self.est_m
        # est_sd / est_var unchanged; cv = sd/mean is unstable near mean 0 and
        # deliberately not relied upon (describe shows SD when signed).
        self.est_cv = self.est_sd / self.est_m if self.est_m else np.inf

    def _fft_aggregate(self, sev_density, padding):
        """Run one FFT convolution: severity density -> aggregate density.

        Single source of truth for the FFT-PGF-iFFT core (Mildenhall 2024,
        §2.2). Used by ``_freq_sev_convolution`` (the main per-update path),
        by ``update_work`` for the subject (gross) aggregate when occ-reins
        is present, and by ``reins_density_df`` to compute gross/ceded/net
        aggregates from the corresponding severities. The zero-risk and
        fixed-1 shortcuts live here so every caller sees them consistently.

        Parameters
        ----------
        sev_density : np.ndarray
            Discretised severity on ``self.xs_sev`` (gross, net, or ceded);
            ``sev_density[j]`` is the mass at physical ``(j - i0) * bs``.
        padding : int
            FFT padding factor passed to ``ft`` / ``ift``.

        Returns
        -------
        agg_density : np.ndarray
            Aggregate density on the output grid ``self.xs``.
        ftagg_density : np.ndarray
            FT of the aggregate (padded length). Callers that don't need
            this (e.g. ``reins_density_df``) discard it.

        Notes
        -----
        Two paths, selected by whether any offset is active:

        - **Default (``i0 == 0`` and output window origin ``x_min == 0``).**
          The original ``ft`` / ``freq_pgf`` / ``ift`` path, unchanged and
          byte-for-byte identical to prior releases.
        - **Signed / windowed (F1 + F2).** Negatives live at the top of the
          padded length-``M = N << padding`` FFT buffer (period ``M*bs``); the
          severity is laid in with physical 0 at index 0 (``i0`` negative
          buckets wrapped to the top). After the FFT the result is *relabelled*
          onto the output window by a single ``np.roll`` of ``-round(x_min/bs)``
          and the first ``N`` buckets kept. Relabelling a finished, exact array
          carries no ``N·s`` shift term, so this is correct for random as well
          as fixed frequency (the key F2 clarification, plan §2). Exact when the
          aggregate support width ``W < M*bs``; a window narrower than the
          support shows up as a two-sided deficit.
        """
        N = len(self.xs)
        i0 = self.i0
        j0 = int(round(self.x_min / self.bs)) if self.bs else 0
        if i0 == 0 and j0 == 0:
            # ---- default non-negative, zero-based path (unchanged) ----
            if self.n == 0:
                out = np.zeros_like(self.xs)
                out[0] = 1.0
                return out, ft(out, padding)
            z = ft(sev_density, padding)
            ftagg = self.frequency.freq_pgf(self.n, z)
            if np.sum(self.en) == 1 and self.frequency.freq_name == 'fixed':
                return sev_density.copy(), ftagg
            return np.real(ift(ftagg, padding)), ftagg

        # ---- signed / windowed path (F1 negative-x + F2 output window) ----
        M = N << padding
        if self.n == 0:
            # Zero-risk: point mass at physical 0, placed at FFT index 0 so the
            # output-window roll below sends it to the correct output bucket.
            a = np.zeros(M)
            a[0] = 1.0
            ftagg = sfft.rfft(a)
        else:
            # Lay the severity into the length-M buffer: physical 0..(N-1-i0)*bs
            # at indices 0..N-1-i0; the i0 negative buckets wrap to the very top
            # of M (indices M-i0..M-1). Equivalent to np.roll(sev, -i0) but into
            # the padded length so the period is M*bs, not N*bs.
            g = np.zeros(M)
            g[:N - i0] = sev_density[i0:]
            if i0:
                g[M - i0:] = sev_density[:i0]
            z = sfft.rfft(g)
            ftagg = self.frequency.freq_pgf(self.n, z)
            a = sfft.irfft(ftagg, M)
        # F2: relabel onto the output window. Roll so x_min lands at output
        # index 0 (j0 may be negative when x_min < 0), then keep the first N.
        agg = np.roll(a, -j0)[:N]
        return agg, ftagg

    def _write_stage_moments(self, col, sev_mom, agg_mom, copy_freq_from=None):
        """Write a moment tuple into a single ``stats_df`` column.

        Helper to keep the staged-empirical writes in ``update_work`` tidy.
        Each moment tuple is the six values ``(ex1, ex2, ex3, mean, cv,
        skew)`` returned by ``xsden_to_mwrangler``.

        Parameters
        ----------
        col : str
            Destination column label (``empirical``, ``after_occ``,
            ``gross_empirical``, ...).
        sev_mom, agg_mom : tuple of float
            Six-tuple raw + central moments for the sev and agg rows.
        copy_freq_from : str or None
            If set, mirror the freq mean/cv/skew rows from another column
            (freq is unchanged by either reinsurance stage).
        """
        _measures = ('ex1', 'ex2', 'ex3', 'mean', 'cv', 'skew')
        for measure, value in zip(_measures, sev_mom):
            self.stats_df.loc[('sev', measure), col] = value
        for measure, value in zip(_measures, agg_mom):
            self.stats_df.loc[('agg', measure), col] = value
        if copy_freq_from is not None:
            src = self.stats_df[copy_freq_from]
            for measure in ('mean', 'cv', 'skew'):
                self.stats_df.loc[('freq', measure), col] = src[('freq', measure)]

    def _freq_sev_convolution(self, padding):
        """Compute the aggregate density by FFT convolution (Mildenhall 2024, §2.2).

        Thin wrapper that routes the (post-occ-reins) ``sev_density`` through
        ``_fft_aggregate`` and writes ``self.agg_density`` /
        ``self.ftagg_density``. The FFT core, zero-risk and fixed-1 shortcuts
        all live in ``_fft_aggregate``.

        Parameters
        ----------
        padding : int
            Padding factor passed to ``ft`` / ``ift`` to mitigate FFT aliasing
            (see Mildenhall 2024, §2.3.2).

        Notes
        -----
        Per-occurrence reinsurance is applied to ``sev_density`` *before* this
        method is called; aggregate reinsurance is applied to ``agg_density``
        *after*. The FFT here is unaware of either.
        """
        self.agg_density, self.ftagg_density = self._fft_aggregate(
            self.sev_density, padding)

    # ================================================================
    # Validation (paper §4.7), unwrap, picks, freq_pmf
    # ================================================================

    @property
    def valid(self):
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
        ``describe`` (display).

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
        if self._valid is not None:
            return self._valid

        rv = Validation.NOT_UNREASONABLE
        # Not yet updated → no empirical moments to validate against.
        if pd.isna(self.stats_df['empirical'].get(('agg', 'mean'), np.nan)):
            self._valid = Validation.NOT_UPDATED
            return Validation.NOT_UPDATED
        # Mean / aliasing reads come straight off ``stats_df['error']`` --
        # the canonical noise-aware relative error of ``gross_empirical``
        # vs ``mixed``. Under no reinsurance ``gross_empirical ==
        # empirical`` and this is the classical theoretical-vs-empirical
        # check; under reinsurance it is the subject-validation hook from
        # §1.3 of the plan, the only apples-to-apples check available.
        err = self.stats_df['error'].abs()
        eps = self.validation_eps
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
        mixed = self.stats_df['mixed']
        emp = self.stats_df['gross_empirical']
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
        if self.reins_kinds() != 'None':
            rv |= Validation.REINSURANCE

        if rv == Validation.NOT_UNREASONABLE:
            logger.info('Aggregate %s does not fail any validation: not unreasonable', self.name)
        self._valid = rv
        return rv

    def unwrap(self, p=1e-7, audit=True):
        """
        Unwrap self created with log2 that is too small to contain the answer.

        :param p: Percentile threshold. The estimated p and 1-p quantiles are
            used to determine the effective support [L, R]. R-L must fit in the
            space available, i.e., R-L <= N * self.bs.
        :param audit: If audit, return comparison of empirical moments of shifted
            answer with a.agg_m etc. analytic moments.
        :return: Unwrap named tuple containing fields y the density as a Series,
            mode of shifting/unwrapping, prob_captured the probability in the
            effective support (which should be close to 1), L, R the boundary of the
            effective support.
        """
        # figure bounds from method of moments estimates
        m, cv, skew = self.agg_m, self.agg_cv, self.agg_skew
        sc = self.bs
        L, R = _estimate_agg_percentile(m, cv, skew, p=(p, 1 - p))
        # snap to grid in both cases (can't use self.snap because outside index!)
        L = int(np.round(L / sc, 0))
        R = int(np.round(R / sc, 0))

        # number of buckets
        N = 1 << self.log2

        # is the request reasonable?
        # enough space condition: R - L <= N
        assert R - L <= N, f'{R=} - {L=} = {R-L=} > {N=}, not enough space'

        # how many "blocks" to the right are we?
        l = L // N
        r = R // N

        # extract aliased density
        y = self.density_df.p_total.values

        # there are now two cases: dist fits within one block or wraps over two
        # if it falls over more than two that is an error
        if l == r:
            # no unwrapping, range lies in one block
            # just shift index to right by correct number of chunks
            # locate correct left hand edge, index created below
            L = (L // N) * N
            # method reporting
            mode = 'Shift only'  # \n{L=}'
        elif l == r - 1:
            # must wrap answer into one block and shift
            # figure location of extreme points as remainders
            rem_r = R % N       # right hand end in fft-wrapped coords
            rem_l = L % N
            # by math this will always be true (see blog post)
            assert rem_l >= rem_r
            # unwrap amount
            roll_forward = N - (rem_l + rem_r) // 2
            y = np.roll(y, roll_forward)
            # shifted index, factoring in unwrap
            L = (L // N + 1) * N - roll_forward
            # method reporting
            mode = 'Shift and wrap'  # \n{roll_forward=}, {L=}'
        else:
            # see blog post
            print(f'Should not occur: {l=}, {r=}')

        # align with index and create answer
        i = np.arange(L, L + N, dtype=float) * sc
        ans = pd.Series(y, index=i)
        # apply scale to L and R now to match ans
        L *= sc
        R *= sc
        # document proportion of probability in selected range
        prob_captured = ans[L:R].sum()
        # package results
        Unwrap = namedtuple('Unwrap', 'y, mode, prob_captured, L, R, audit_df')
        if audit:
            em, ecv, eskew = xsden_to_meancvskew(ans.index, ans)
            audit_df = pd.DataFrame(
                {'m': [m, em],
                 'cv': [cv, ecv],
                 'skew': [skew, eskew]},
                index=['actual', 'rewrapped'])

        else:
            audit_df = None
        ans = Unwrap(ans, mode, prob_captured, L, R, audit_df)
        return ans

    def picks(self, attachments, layer_loss_picks, debug=False):
        """
        Adjust the computed severity to hit picks targets in layers defined by a.
        Delegates work to :func:`_picks_work`. See that function for details.

        """
        # always want to work off gross severity
        if self.sev_density_gross is not None:
            logger.info('Using GROSS severity in picks')
            sd = self.sev_density_gross
        else:
            sd = self.sev_density
        return _picks_work(attachments, layer_loss_picks, self.xs, sd, n=self.n,
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
        fz = ft(z, 0)
        fz = self.frequency.freq_pgf(self.en, fz)
        dist = ift(fz, 0)
        # remove fuzz -- intentionally ONE-SIDED (zeroes negatives too); a
        # frequency pmf has no legitimate negatives, so this is NOT the shared
        # two-sided ``remove_fuzz`` utility.
        dist[dist < np.finfo(float).eps] = 0
        if not np.allclose(self.n,  self.en):
            logger.warning('Frequency.pmf | n %s != en %s; using en', self.n, self.en)
        return dist

    # ================================================================
    # Reinsurance application: occ pre-FFT, agg post-FFT
    # ================================================================

    def _rebucket_to_grid(self, values, mass, scheme=None, origin=None):
        """Scatter off-grid ``mass`` at target ``values`` onto the model grid.

        The model grid is ``self.xs == bs * arange`` with ``xs[0] == 0``, so
        the (fractional) grid index of a value ``v`` is ``v / bs``. Used to
        place reinsurance net/ceded values back on the grid after the cession
        map moves them off it, and to place discrete-severity atoms on the grid
        during discretization (see :meth:`discretize`).

        Parameters
        ----------
        values : ndarray
            Target loss values (net/ceded points, or severity atoms).
        mass : ndarray
            Probability mass to redistribute, aligned with ``values``.
        scheme : {'linear', 'nearest'}, optional
            Placement scheme. Defaults to :attr:`reins_bucket` (so the
            reinsurance call sites are unchanged); the discrete-severity call
            site passes :attr:`dsev_bucket`.
        origin : float, optional
            Physical value at output index 0 -- the grid origin used to map a
            value to its (fractional) bucket index ``(v - origin) / bs``.
            Defaults to :attr:`x_min` (the output-window origin), correct for
            the reinsurance call sites. The discrete-severity call site passes
            the *severity* grid origin ``xs_sev[0] == -i0·bs``, which differs
            from ``x_min`` when the output window is forced wider than the
            severity support (e.g. an explicit ``x_min`` below the smallest
            atom).

        Returns
        -------
        ndarray
            Probability vector on ``self.xs`` (length ``self.n``).

        Notes
        -----
        Two schemes:

        - ``'nearest'`` rounds each value to its closest bucket. Full mass
          lands in one bucket, with up to ``bs/2`` positional bias.
        - ``'linear'`` splits each value's mass between its two bracketing
          buckets ``k`` and ``k+1`` with weights ``1-f`` and ``f`` where
          ``f = v/bs - k``. Because ``(1-f)·k·bs + f·(k+1)·bs == v``, the
          first moment is preserved **exactly**; both schemes preserve total
          mass (``Σp == Σmass``).

        Values at or beyond the top of the grid (``xs[-1]``) pile into the
        last bucket -- the same overflow mode as an aggregate deficit, and
        surfaced the same way. Negative targets (should not occur for valid
        cessions) clip into bucket 0.
        """
        bs = self.bs
        n = len(self.xs)
        if origin is None:
            origin = self.x_min
        # Grid index of a value v is (v - origin) / bs; origin == 0 on the
        # default grid recovers the original v / bs.
        scaled = (np.asarray(values, dtype=float) - origin) / bs
        out = np.zeros(n)
        if scheme is None:
            scheme = self.reins_bucket
        if scheme == 'nearest':
            idx = np.clip(np.round(scaled).astype(int), 0, n - 1)
            np.add.at(out, idx, mass)
        else:  # 'linear' -- mass split preserves E[X] exactly
            k = np.clip(np.floor(scaled).astype(int), 0, n - 1)
            f = np.clip(scaled - k, 0.0, 1.0)
            kp1 = np.clip(k + 1, 0, n - 1)
            np.add.at(out, k, mass * (1 - f))
            np.add.at(out, kp1, mass * f)
        return out

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
        reins_df = pd.DataFrame(
            {'loss': self.xs, 'p_subject': base_density}).set_index('loss', drop=False)
        reins_df['loss_net'] = netter(reins_df.loss)
        reins_df['loss_ceded'] = ceder(reins_df.loss)
        # Rebucket the off-grid net/ceded values back onto the uniform model
        # grid (self.xs == bs * arange, xs[0] == 0) via the selected
        # ``reins_bucket`` scheme. See _rebucket_to_grid for the mass identity.
        p_subject = np.asarray(base_density, dtype=float)
        p_net = self._rebucket_to_grid(reins_df['loss_net'].to_numpy(), p_subject)
        p_ceded = self._rebucket_to_grid(reins_df['loss_ceded'].to_numpy(), p_subject)
        reins_df['p_net'] = p_net
        reins_df['p_ceded'] = p_ceded
        # F_* columns were vestigial from the old interp1d-of-CDF algorithm;
        # the scatter computes p_net / p_ceded directly. The debug CDF panel
        # below cumsums inline.
        reins_df = reins_df[['loss', 'p_subject', 'loss_net',
                             'loss_ceded', 'p_net', 'p_ceded']]

        if debug is False:
            return ceder, netter, reins_df

        logger.debug('making re graphs.')
        # quick debug; need to know kind=occ|agg here
        f = plt.figure(constrained_layout=True, figsize=(12, 9))
        axd = f.subplot_mosaic('AB\nCD')
        xlim = self._limits()
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
        cdf = reins_df[['p_subject', 'p_net', 'p_ceded']].cumsum()
        cdf.columns = ['F_subject', 'F_net', 'F_ceded']
        cdf.plot(xlim=xlim, ax=ax)
        ax.set(title='Subject, net and ceded\ndistributions')
        ax.legend()

        ax = axd['D']
        reins_df.filter(regex='p_').plot(xlim=xlim, drawstyle='steps-post', ax=ax)
        ax.set(title='Subject, net and ceded\ndensities')
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
        # Retain the ceder/netter step functions for the exact (EX) reporting
        # path; the rebucketed densities go straight onto the gcn members and
        # into ``reins_density_df`` (no persistent per-stage frame).
        self.occ_ceder = occ_ceder
        self.occ_netter = occ_netter
        self.sev_density_gross = self.sev_density
        self.sev_density_net = occ_reins_df['p_net'].to_numpy()
        self.sev_density_ceded = occ_reins_df['p_ceded'].to_numpy()
        if self.occ_kind == 'ceded to':
            self.sev_density = self.sev_density_ceded
        elif self.occ_kind == 'net of':
            self.sev_density = self.sev_density_net
        else:
            raise ValueError(f'Unexpected kind of occ reinsurance, {self.occ_kind}')
        # ``est_sev_*`` is written by ``update_work`` from the post-reins
        # severity (the same density set above) using ``xsden_to_mwrangler``
        # -- no intermediate write needed here.

    def apply_agg_reins(self, debug=False, padding=1):
        """
        Apply the entire agg reins structure and save output.
        For by layer detail create reins_audit_df.
        Makes agg_density_gross, agg_density_net and agg_density_ceded, and
        updates agg_density to the requested view.

        Not reflected in statistics df: the post-reins empirical moments
        (``est_*`` and the ``stats_df['empirical']`` column) are written by
        ``update_work`` from the same density updated here.

        :return:
        """
        # generic function makes netter and ceder functions
        if self.agg_reins is None:
            return
        logger.info('Applying aggregate reinsurance for %s', self.name)

        agg_ceder, agg_netter, agg_reins_df = self._apply_reins_work(self.agg_reins, self.agg_density, debug)
        # Retain the ceder/netter for the exact (EX) reporting path; the
        # rebucketed densities go onto the gcn members / ``reins_density_df``.
        self.agg_ceder = agg_ceder
        self.agg_netter = agg_netter
        self.agg_density_gross = self.agg_density
        self.agg_density_net = agg_reins_df['p_net'].to_numpy()
        self.agg_density_ceded = agg_reins_df['p_ceded'].to_numpy()
        if self.agg_kind == 'ceded to':
            self.agg_density = self.agg_density_ceded
        elif self.agg_kind == 'net of':
            self.agg_density = self.agg_density_net
        else:
            raise ValueError(f'Unexpected kind of agg reinsurance, {self.agg_kind}')

        # update ft of agg
        self.ftagg_density = ft(self.agg_density, padding)

    def reins_description(self, kind='both', width=0):
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

    def reins_kinds(self):
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

    # ================================================================
    # Distortion, ruin theory, plotting
    # ================================================================

    def apply_distortion(self, dist):
        """
        Apply distortion to the aggregate density and append as exag column to density_df.

        :param dist:
        :return:
        """
        if self.agg_density is None:
            logger.warning('You must update before applying a distortion ')
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

        bit = self.sev_density_df.p_sev.copy()
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
        fz = ft(dfi, padding)
        mfz = 1 / (1 - fz / (1 + rho))
        f = ift(mfz, padding)
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
        :param kwargs: passed to ``plt.subplot_mosaic``
        :return:
        """
        if axd is None:
            if 'figsize' not in kwargs:
                kwargs['figsize'] = (3 * FIG_W, FIG_H)
            self.figure, axd = plt.subplot_mosaic('ABC', layout='constrained', **kwargs)
        else:
            self.figure = axd['A'].figure

        if self.bs == 1 and abs(self.est_m) < 1025:
            # treat as discrete
            if xmax > 0:
                mx = xmax
            else:
                mx = self.q(1) * 1.05
            # Signed aggregate: the support reaches below 0 (e.g. a P&L); use
            # the true minimum so the left of the axis isn't clipped at ~0.
            mn = min(self.q(0), 0.0) if (self.xs is not None and self.xs[0] < 0) else 0.0
            span = nice_multiple(mx - mn)
            left = mn - (mx - mn) / 25

            # Aggregate from density_df; severity from its own grid
            # (sev_density_df), which may differ from the aggregate window.
            df = self.density_df[['p_total', 'F', 'loss']].copy()
            # anchor a zero-mass row just left of the support so the steps/stems
            # start from the baseline (at mn - 0.5, not a fixed -0.5). ``loss``
            # must equal the row's own index, not 0: the Lee plot (panel C) plots
            # ``loss`` against ``F``, so a stray ``loss=0`` here would draw a
            # spurious vertical segment from 0 down to the first (signed) point.
            df.loc[mn - 0.5, :] = (0, 0, mn - 0.5)
            df = df.sort_index()
            sdf = self.sev_density_df
            if mx <= 60:
                # stem plot for small means
                axd['A'].stem(df.index, df.p_total, basefmt='none', linefmt='C0-', markerfmt='C0.', label='Aggregate')
                axd['A'].stem(sdf.loss, sdf.p_sev, basefmt='none', linefmt='C1-', markerfmt='C1,', label='Severity')
            else:
                df.p_total.plot(ax=axd['A'], drawstyle='steps-mid', lw=2, label='Aggregate')
                sdf.p_sev.plot(ax=axd['A'], drawstyle='steps-mid', lw=1, label='Severity')

            axd['A'].set(xlim=[left, mx + 1], title='Probability mass functions')
            axd['A'].legend()
            if span > 0:
                axd['A'].xaxis.set_major_locator(ticker.MultipleLocator(span))
            # for discrete plot F next
            df.F.plot(ax=axd['B'], drawstyle='steps-post', lw=2, label='Aggregate')
            sdf.F_sev.plot(ax=axd['B'], drawstyle='steps-post', lw=1, label='Severity')
            axd['B'].set(xlim=[left, mx + 1], title='Distribution functions')
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
            # same trim for severity (on its own grid)
            sidx = (sdf.F_sev >= 1).idxmax()
            sdft = sdf.loc[:sidx]
            ax.plot(sdft.F_sev, sdft.loss, drawstyle='steps-pre', lw=1, label='Severity')
            ax.set(xlim=[-0.025, 1.025], ylim=[left, mx + 1], title='Quantile (Lee) plot')
            ax.legend().set(visible=False)
        else:
            # continuous
            df = self.density_df
            sdf = self.sev_density_df       # severity on its own grid
            if xmax > 0:
                xlim = [-xmax / 50, xmax * 1.025]
            else:
                xlim = self._limits(stat='range', kind='linear')
            xlim2 = self._limits(stat='range', kind='log')
            ylim = self._limits(stat='density')

            ax = axd['A']
            # divide by bucket size...approximating the density
            (df.p_total / self.bs).plot(ax=ax, lw=2, label='Aggregate')
            (sdf.p_sev / self.bs).plot(ax=ax, lw=1, label='Severity')
            ax.set(xlim=xlim, ylim=ylim, title='Probability density')
            ax.legend()

            (df.p_total / self.bs).plot(ax=axd['B'], lw=2, label='Aggregate')
            (sdf.p_sev / self.bs).plot(ax=axd['B'], lw=1, label='Severity')
            ylim = axd['B'].get_ylim()
            ylim = [1e-15, ylim[1] * 2]
            axd['B'].set(xlim=xlim2, ylim=ylim, title='Log density', yscale='log')
            axd['B'].legend().set(visible=False)

            ax = axd['C']
            # to do: same trimming for p-->1 needed?
            ax.plot(df.F, df.loss, lw=2, label='Aggregate')
            ax.plot(sdf.F_sev, sdf.loss, lw=1, label='Severity')
            ax.set(xlim=[-0.02, 1.02], ylim=xlim, title='Quantile (Lee) plot', xlabel='Non-exceeding probability p')
            ax.legend().set(visible=False)

    def _limits(self, stat='range', kind='linear', zero_mass='include'):
        """
        Suggest sensible plotting limits for kind=range, density, etc., same as Portfolio.

        Should optionally return a locator for plots?

        Called by ploting routines. Single point of failure!

        Must work without ``q`` function when not yet computed.

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
            # No FFT output yet; estimate the 0.999 quantile from the theoretical
            # mixed-total agg moments.
            try:
                p999 = _estimate_agg_percentile(self.agg_m, self.agg_cv, self.agg_skew, 0.999)
            except ValueError:
                p999 = np.inf
            return f(p999)

        if stat == 'range':
            p = 0.999 if kind == 'linear' else 0.99999
            hi = self.q(p)
            # Signed aggregate (grid origin < 0): the mass can sit anywhere on
            # the real line, possibly entirely negative, so use a *two-sided*
            # quantile range. ``f(hi)`` would be reversed/clipped when hi < 0.
            if self.xs is not None and self.xs[0] < 0:
                lo = self.q(1 - p)
                w = hi - lo
                pad = 0.02 * w if w > 0 else max(abs(hi), 1.0)
                return [lo - pad, hi + pad]
            return f(hi)

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
            raise ValueError('Inadmissible stat/kind passsed, expected range/density and log/linear.')

    # ================================================================
    # Display reports, diagnostics, queries, risk measures, pricing
    # ================================================================

    @property
    def pprogram(self):
        """Cleaned DecL program text (notes removed, whitespace collapsed).

        For the raw input as supplied to ``build`` use ``self.program``.
        """
        return decl_pprint(self.program, split=0, show=False)

    @property
    def pprogram_html(self):
        """Syntax-highlighted DecL program for IPython / Jupyter display."""
        return decl_pprint(self.program, split=0, html=True, show=False)

    @property
    def describe(self):
        """Moment table for Freq / Sev / Agg, dense ``EX``/``CV``/``Sk`` headings.

        The daily-driver display used by ``qd(agg)`` and ``_repr_html_``.
        Three-row Freq / Sev / Agg frame.

        Two display modes, same 8-column shape and same column arithmetic:

        * **No reinsurance** -- validation view. Columns are theoretical
          ``EX | Est EX | Err EX | CV | Est CV | Err CV | Sk | Est Sk``.
          ``Err`` is the noise-aware relative error of empirical vs
          theoretical.
        * **With reinsurance** -- economic view. Columns become ``Gross EX
          | <label> EX | Change EX | Gross CV | <label> CV | Change CV |
          Gross Sk | <label> Sk``, where ``Gross`` is the theoretical
          before any cover and ``<label>`` is the model output -- ``Net``
          (all covers net of), ``Ceded`` (all ceded to), or ``Output``
          (mixed, occ and agg passing different kinds). ``Change = (output
          - gross) / gross`` -- arithmetically the same column as ``Err``
          (so the eyeball degenerates cleanly to the validation view when
          reins is absent), but now read as the % change driven by the
          cession. Labels are the ``REINS_LABEL_*`` constants.

        Sources from the canonical ``self.stats_df``: ``mixed`` for
        Gross, ``empirical`` for the realised (model-output) view.
        """
        return self._describe()

    def _describe(self, force_reins_label=None, force_sd=False):
        """Build the ``describe`` frame, optionally forced into reins view.

        Parameters
        ----------
        force_reins_label : str or None
            When ``None`` (the default, used by the ``describe`` property)
            the column format is chosen from this unit's own reinsurance:
            the economic Gross/Net/Ceded/Output view if a treaty is
            present, else the plain theory/empirical validation view.

            When a non-``None`` label is supplied, the economic view is
            forced and that label is used for the after-reins column,
            regardless of this unit's own cession. ``Portfolio.describe``
            passes a portfolio-wide label here so that every unit block —
            including units with no reinsurance — shares one column
            layout and aligns with the ``total`` block.
        force_sd : bool, default False
            Force the **SD** spread trio (instead of CV) even when this unit
            is not itself signed. A signed unit always uses SD; this flag lets
            ``Portfolio.describe`` push the whole table into SD when *any* unit
            is signed, so the unit blocks and the ``total`` block share one
            column layout (CV and SD cannot be mixed in one frame).

        Returns
        -------
        pandas.DataFrame
            Three-row Freq / Sev / Agg frame; see :meth:`describe`.
        """
        if self._signed() or force_sd:
            return self._describe_signed(force_reins_label)
        st = self.stats_df['mixed']
        rlabel = force_reins_label if force_reins_label is not None \
            else self._reins_after_label()
        df = pd.DataFrame(
            {
                'EX': [st[('freq', 'mean')], st[('sev', 'mean')], st[('agg', 'mean')]],
                'CV': [st[('freq', 'cv')],   st[('sev', 'cv')],   st[('agg', 'cv')]],
                'Sk': [st[('freq', 'skew')], st[('sev', 'skew')], st[('agg', 'skew')]],
            },
            index=['Freq', 'Sev', 'Agg'],
        )
        df.index.name = 'X'
        emp = self.stats_df['empirical']
        post_update = pd.notna(emp.get(('agg', 'mean'), np.nan))
        if post_update:
            # Realised (after-reins, or = subject if no reins) middle column.
            mid_label = rlabel or 'Est'
            df.loc['Sev', f'{mid_label} EX'] = emp[('sev', 'mean')]
            df.loc['Agg', f'{mid_label} EX'] = emp[('agg', 'mean')]
            change_label = 'Change' if rlabel else 'Err'
            df.loc[:, f'{change_label} EX'] = _noise_aware_rel_error(
                df[f'{mid_label} EX'], df['EX'])
            df.loc['Sev', f'{mid_label} CV'] = emp[('sev', 'cv')]
            df.loc['Agg', f'{mid_label} CV'] = emp[('agg', 'cv')]
            df.loc[:, f'{change_label} CV'] = _noise_aware_rel_error(
                df[f'{mid_label} CV'], df['CV'])
            df[f'{mid_label} Sk'] = np.nan
            df.loc['Sev', f'{mid_label} Sk'] = emp[('sev', 'skew')]
            df.loc['Agg', f'{mid_label} Sk'] = emp[('agg', 'skew')]
            ordered = [
                'EX', f'{mid_label} EX', f'{change_label} EX',
                'CV', f'{mid_label} CV', f'{change_label} CV',
                'Sk', f'{mid_label} Sk',
            ]
            df = df[ordered]
        # First-column label: under reinsurance the theoretical is the
        # ``Gross`` view (top of step 1, before any cover); without reins keep
        # the legacy ``EX``/``CV``/``Sk`` headings (no rename necessary).
        if rlabel:
            df = df.rename(columns={
                'EX': f'{REINS_LABEL_GROSS} EX',
                'CV': f'{REINS_LABEL_GROSS} CV',
                'Sk': f'{REINS_LABEL_GROSS} Sk'})
        # snap floating-point dust to 0 in moment-value columns for
        # display (e.g. the skew of a symmetric severity); NaN preserved.
        # Change/Err columns retain their numeric dust (they are the
        # validation eyeball).
        for c in df.columns:
            if ' EX' in c or ' CV' in c or ' Sk' in c or c in ('EX', 'CV', 'Sk'):
                if not (c.startswith('Err ') or c.startswith('Change ')):
                    df[c] = _snap_noise(df[c])
        return df

    def _describe_signed(self, force_reins_label=None):
        """``describe`` for a signed aggregate -- **SD** trio instead of CV.

        Same 8-column shape and column arithmetic as :meth:`_describe`, but the
        ``CV`` trio is replaced by an ``SD`` trio. The coefficient of variation
        ``CV = sd / mean`` is unstable and meaningless when the mean can be ~0
        (a P&L straddling break-even), so for any signed object -- a ``ssev`` /
        negative-``dsev`` aggregate **or** a ``pnl`` -- the spread is reported as
        the standard deviation, which is finite and informative regardless of
        the mean. (This also cleans up the 1.0.0a22 signed-portfolio describe.)

        For a ``pnl`` aggregate the **Agg** row is additionally the affine
        (premium-minus-loss) view: ``EX -> agg_shift - E[A]``, ``SD`` unchanged,
        ``Sk -> -Sk``. The Freq / Sev rows describe the underlying loss process
        and are unchanged. The theoretical (Gross) column is sourced from the
        loss ``stats_df`` and transformed here at display time, so the stored
        validation columns stay in loss terms (see :meth:`_apply_agg_affine`).

        Parameters
        ----------
        force_reins_label : str or None
            As in :meth:`_describe`.

        Returns
        -------
        pandas.DataFrame
        """
        st = self.stats_df['mixed']
        emp = self.stats_df['empirical']
        rlabel = force_reins_label if force_reins_label is not None \
            else self._reins_after_label()
        affine = self._agg_affine_active()
        reflect = self._agg_reflect
        shift = self._agg_shift

        def _agg_mean(m):
            if not affine:
                return m
            return (shift - m) if reflect else (shift + m)

        def _agg_skew(sk):
            return -sk if (affine and reflect) else sk

        # Theoretical (loss stats_df; Agg row affine-transformed for display).
        freq_sd = st[('freq', 'mean')] * st[('freq', 'cv')]
        sev_sd = self.sev_sd
        agg_sd = self.agg_sd  # spread is invariant under reflect + shift
        df = pd.DataFrame(
            {
                'EX': [st[('freq', 'mean')], st[('sev', 'mean')],
                       _agg_mean(st[('agg', 'mean')])],
                'SD': [freq_sd, sev_sd, agg_sd],
                'Sk': [st[('freq', 'skew')], st[('sev', 'skew')],
                       _agg_skew(st[('agg', 'skew')])],
            },
            index=['Freq', 'Sev', 'Agg'],
        )
        df.index.name = 'X'
        post_update = pd.notna(emp.get(('agg', 'mean'), np.nan))
        if post_update:
            mid_label = rlabel or 'Est'
            # Empirical: Freq/Sev from the stats_df; Agg from the affine-aware
            # scalars (est_m / est_sd / est_skew are already the P&L view).
            emp_freq_sd = emp[('freq', 'mean')] * emp[('freq', 'cv')]
            df.loc['Sev', f'{mid_label} EX'] = self.est_sev_m
            df.loc['Agg', f'{mid_label} EX'] = self.est_m
            change_label = 'Change' if rlabel else 'Err'
            df.loc[:, f'{change_label} EX'] = _noise_aware_rel_error(
                df[f'{mid_label} EX'], df['EX'])
            df.loc['Sev', f'{mid_label} SD'] = self.est_sev_sd
            df.loc['Agg', f'{mid_label} SD'] = self.est_sd
            df.loc['Freq', f'{mid_label} SD'] = emp_freq_sd
            df.loc[:, f'{change_label} SD'] = _noise_aware_rel_error(
                df[f'{mid_label} SD'], df['SD'])
            df[f'{mid_label} Sk'] = np.nan
            df.loc['Sev', f'{mid_label} Sk'] = self.est_sev_skew
            df.loc['Agg', f'{mid_label} Sk'] = self.est_skew
            ordered = [
                'EX', f'{mid_label} EX', f'{change_label} EX',
                'SD', f'{mid_label} SD', f'{change_label} SD',
                'Sk', f'{mid_label} Sk',
            ]
            df = df[ordered]
        if rlabel:
            df = df.rename(columns={
                'EX': f'{REINS_LABEL_GROSS} EX',
                'SD': f'{REINS_LABEL_GROSS} SD',
                'Sk': f'{REINS_LABEL_GROSS} Sk'})
        for c in df.columns:
            if ' EX' in c or ' SD' in c or ' Sk' in c or c in ('EX', 'SD', 'Sk'):
                if not (c.startswith('Err ') or c.startswith('Change ')):
                    df[c] = _snap_noise(df[c])
        return df

    def _reins_after_label(self):
        """Heading for the model-output column in ``describe``.

        ``Net`` when every cession passes the net; ``Ceded`` when every
        cession passes the ceded; ``Output`` when occ and agg pass
        different kinds (e.g. ``net of occ then ceded to agg`` -- a mixed
        output). Returns ``None`` when no reinsurance is configured (legacy
        validation-view headings apply).
        """
        kinds = []
        if self.occ_reins is not None:
            kinds.append(self.occ_kind)
        if self.agg_reins is not None:
            kinds.append(self.agg_kind)
        if not kinds:
            return None
        uniq = set(kinds)
        if uniq == {'net of'}:
            return REINS_LABEL_NET
        if uniq == {'ceded to'}:
            return REINS_LABEL_CEDED
        return REINS_LABEL_OUTPUT

    def _severity_lattice(self):
        """Integer-lattice step of the severity, or ``None`` if not on a lattice.

        Returns the gcd of the (integer) severity atoms -- the natural bucket
        size, since an aggregate of lattice-valued severities is itself on that
        lattice **regardless of the frequency** (e.g. Poisson x ``dsev [1:10]``
        is integer-valued, so ``bs=1``). ``None`` for any continuous component.
        This is what lets the window estimator pick a coarse, exact ``bs`` and
        shrink ``log2`` to fit, instead of defaulting to a fine ``bs`` over the
        full ``2**log2`` buckets.

        Returns
        -------
        float or None
            ``gcd`` of the integer atoms (1 for ``dsev [1:n]``; 5 for atoms
            ``[0 5 10]``), or ``None`` if any severity component is continuous /
            non-integer.
        """
        atoms = []
        for s in (self.sevs if self.sevs is not None else []):
            a = getattr(s, 'support_atoms', None)
            if a is None:
                return None
            atoms.append(np.asarray(a, dtype=float))
        if not atoms:
            return None
        allx = np.concatenate(atoms)
        if not np.allclose(allx, np.round(allx), atol=1e-9):
            return None
        ints = np.abs(np.round(allx).astype(np.int64))
        ints = ints[ints != 0]
        if len(ints) == 0:
            return 1.0
        g = int(np.gcd.reduce(ints))
        return float(g) if g > 0 else 1.0

    def _exact_discrete_window(self):
        """Exact aggregate support for a fully-discrete ``dfreq``/``fixed`` x ``dsev``.

        Returns ``(A_min, A_max, bs_lattice)`` when the frequency is discrete-
        finite (``dfreq`` -> ``empirical``, or ``fixed``) **and** every severity
        component is a discrete histogram on an integer lattice; otherwise
        ``None``. The aggregate then takes values exactly on the integer lattice
        and its support is finite and exactly computable.

        Notes
        -----
        With claim counts ``N`` (atoms, min ``N_min`` max ``N_max``, possibly
        including 0) and severity atoms ``s_min … s_max``:

        - ``A_max = max(N_max·s_max, N_min·s_max)`` (the sum is maximised by the
          largest atom repeated; over ``N`` the extreme is at ``N_max`` if
          ``s_max>0`` else ``N_min``); include ``0`` if ``0`` is a count atom.
        - ``A_min = min(N_max·s_min, N_min·s_min)`` symmetrically; include ``0``
          if ``0`` is a count atom.

        ``bs_lattice`` is 1 for integer atoms (the common case).
        """
        freq = self.frequency
        if freq.freq_name == 'empirical':
            n_atoms = np.asarray(freq.freq_a, dtype=float)
            has_zero = bool(np.any(n_atoms == 0))
        elif freq.freq_name == 'fixed':
            n_atoms = np.array([float(self.n)])
            has_zero = (self.n == 0)
        else:
            return None
        if self.sevs is None or len(self.sevs) == 0:
            return None
        atoms = []
        for s in self.sevs:
            a = getattr(s, 'support_atoms', None)
            if a is None:
                return None    # a non-discrete component -> not exact
            atoms.append(np.asarray(a, dtype=float))
        s_all = np.concatenate(atoms)

        def _allint(x):
            return bool(np.allclose(x, np.round(x), atol=1e-9))

        if not (_allint(n_atoms) and _allint(s_all)):
            return None
        s_min, s_max = float(s_all.min()), float(s_all.max())
        n_min, n_max = float(n_atoms.min()), float(n_atoms.max())
        hi = max(n_max * s_max, n_min * s_max)
        lo = min(n_max * s_min, n_min * s_min)
        if has_zero:
            hi = max(hi, 0.0)
            lo = min(lo, 0.0)
        return float(lo), float(hi), 1.0

    def _bounded_severity_window(self, p):
        """Window for a bounded severity with a (possibly small) claim count.

        Returns ``(A_lo, A_hi)`` using the severity's bounded support and a high
        frequency quantile ``N_hi`` (from the analytic frequency moments), or
        ``None`` if the severity is not bounded. For a small claim count this
        bound is tight; for a large count the law of large numbers concentrates
        the aggregate far inside ``[0, N_hi·s_max]`` and the moment window is
        tighter -- the caller (``_bs_window``) only selects this method when it
        is at least as tight as the moment window.
        """
        if self.sevs is None or len(self.sevs) == 0:
            return None
        if not all(getattr(s, 'bounded', False) for s in self.sevs):
            return None
        s_his, s_los = [], []
        for s in self.sevs:
            hi = s.limit if np.isfinite(s.limit) else float(s.fz.support()[1])
            lo = float(s.fz.support()[0]) if getattr(s, 'signed', False) else 0.0
            s_his.append(hi)
            s_los.append(lo)
        s_max, s_min = max(s_his), min(s_los)
        f1, f2, f3 = self.frequency.freq_moms(self.n)
        fsd = float(np.sqrt(max(f2 - f1 * f1, 0.0)))
        zN = ss.norm.isf(1 - p)
        n_hi = f1 + zN * fsd
        return float(min(0.0, n_hi * s_min)), float(n_hi * s_max)

    def _bs_window(self, log2, bs_in, x_min_in, bucket_sizing_p):
        """Decide ``(bs, log2, x_min)`` for ``update`` and build ``_bs_window_df``.

        Runs up to three sizing methods and records each in the expert-
        inspectable ``self._bs_window_df``, then selects per the documented
        priority. ``log2`` is a **cap** (the input value, default 16): the exact
        discrete method may use fewer buckets but never more; the other methods
        fill the cap. The 0-origin convention is preserved -- ``x_min = 0``
        whenever the support is non-negative; a negative origin is used only for
        a genuinely signed aggregate.

        Parameters
        ----------
        log2 : int
            Bucket-count cap, ``2**log2`` buckets.
        bs_in : float
            ``0`` to estimate the bucket; ``>0`` to force it (honoured, D4).
        x_min_in : float or None
            ``None`` lets the selected method choose the origin; a number forces
            it (snapped to ``bs``, D4).
        bucket_sizing_p : float
            Tail probability for the moment / bounded windows.

        Returns
        -------
        (bs, log2, x_min) : tuple
            Final grid parameters; ``self._bs_window_df`` is also populated.

        Methods (rows of ``_bs_window_df``)
        -----------------------------------
        - ``moment`` -- always; the legacy 3-moment sizing for non-negative
          aggregates (reproduces ``recommend_bucket`` exactly), two-sided
          ``estimate_agg_window`` for signed ones.
        - ``exact_discrete`` -- ``dfreq``/``fixed`` x ``dsev`` on an integer
          lattice: exact finite support, ``bs=1``, minimal ``log2`` (<= cap).
        - ``bounded_small`` -- bounded severity: ``[0, N_hi·s_max]`` from a high
          frequency quantile; selected only when tighter than ``moment``.

        Selection: ``exact_discrete`` > ``bounded_small`` (if tighter) >
        ``moment``.
        """
        N0 = 1 << log2
        m = self.agg_m
        try:
            ex2 = float(self.stats_df['mixed'][('agg', 'ex2')])
            sd = float(np.sqrt(max(ex2 - m * m, 0.0)))
        except Exception:  # pragma: no cover - defensive
            sd = self.agg_sd
        skew = self.agg_skew
        # Convolution-grid signedness: a ``pnl`` aggregate has a non-negative
        # loss severity, so the loss FFT is sized here on the ordinary
        # non-negative grid; the affine relabel onto the signed P&L window is a
        # post-step (below + in ``_apply_agg_affine``). Hence ``_signed_severity``
        # not ``_signed`` -- the latter also reports ``True`` for the affine.
        signed = self._signed_severity()
        # Window coverage: WINDOW_NINES nines (default 12) -- far tighter than
        # the legacy bucket p, so the window captures essentially all the mass.
        p = 1.0 - 10.0 ** -WINDOW_NINES
        lattice = self._severity_lattice()

        def _size(x_lo, x_hi, lattice_bs):
            """Origin ``x0`` and grid ``(bs, log2)`` for a window ``[x_lo, x_hi]``.

            Returns ``(x0, bs, log2)`` only -- the *window* edges stay the
            method's own ``[x_lo, x_hi]``; the realized grid extent
            (``x0 + 2**log2 * bs``, which power-of-2 padding makes wider than the
            window) is reported separately in the ``used`` row.

            bs: forced ``bs_in`` if given; else the integer lattice step (so an
            integer-valued aggregate uses ``bs=1`` not a fine fraction); else a
            resolution ``round_bucket(W / 2**cap)``. ``log2`` is shrunk to just
            cover the window (<= cap) when bs is free, keeping the realized grid
            sensible; if the user pinned bs they control the grid so the cap
            ``log2`` is honoured; if the window needs more than the cap, bs is
            coarsened to fit.
            """
            W = max(x_hi - x_lo, 0.0)
            if bs_in > 0:
                bs = float(bs_in)
            elif lattice_bs is not None:
                bs = float(lattice_bs)
            else:
                bs = round_bucket(W / N0) if W > 0 else 1.0
            x0 = float(np.floor(x_lo / bs) * bs) if signed else 0.0
            span = x_hi - x0
            need = int(np.ceil(np.log2(max(span / bs + 1.0, 1.0)))) if span > 0 else 0
            if bs_in > 0:
                l2 = log2
            elif need <= log2:
                # at least 1 (>= 2 buckets) so a degenerate / point-mass window
                # never collapses to a single bucket.
                l2 = min(log2, max(need, 1))
            else:
                bs = round_bucket(span / N0)
                x0 = float(np.floor(x_lo / bs) * bs) if signed else 0.0
                l2 = log2
            return x0, float(bs), int(l2)

        def _row(x_lo, x_hi, lattice_bs, coverage, note):
            # ``x_max`` is the method's own computed window top (e.g. the exact
            # support max), NOT the padded grid extent -- the ``used`` row shows
            # the realized grid.
            x0, bs_, l2_ = _size(x_lo, x_hi, lattice_bs)
            return dict(applies=True, x_min=float(x0), x_max=float(x_hi),
                        W=float(x_hi - x0), bs=bs_, log2=l2_,
                        coverage=coverage, note=note)

        rows = {}

        # ---- moment (always) --------------------------------------------
        if bs_in > 0 and not (signed and np.isfinite(sd)):
            # bs is pinned -> the grid is the user's; the moment "window" is
            # just the realized extent. Avoids estimating a window (and the
            # infinite-variance raise) when we don't need one.
            x_lo, x_hi = 0.0, float(N0 * bs_in)
        elif signed and np.isfinite(sd):
            x_lo, x_hi, _W = estimate_agg_window(m, sd, skew, p)
        else:
            x_lo = 0.0
            try:
                x_hi = float(_estimate_agg_percentile(m, self.agg_cv, skew, p))
            except ValueError:
                # no finite variance (e.g. Pareto) and bs free: last-resort
                # legacy extent (recommend_bucket bumps p / uses the limit).
                x_hi = float(N0 * round_bucket(self.recommend_bucket(log2, p=bucket_sizing_p)))
            if not np.isfinite(x_hi):
                # deterministic (sd ~ 0) or undefined skew (NaN): a few sd above
                # the mean (collapses to the mean for a point mass).
                x_hi = m + 8.0 * sd
        rows['moment'] = _row(x_lo, x_hi, lattice, f'1-1e-{WINDOW_NINES}',
                              '3-moment MoM window')

        # ---- exact_discrete ---------------------------------------------
        ed = self._exact_discrete_window()
        if ed is not None:
            a_lo, a_hi, bs_lat = ed
            r = _row(a_lo, a_hi, bs_lat, 'exact', 'dfreq/fixed x dsev integer lattice')
            if not np.isclose(r['bs'], bs_lat):
                r['coverage'] = 'support exact, bs coarsened'
            rows['exact_discrete'] = r

        # ---- bounded_small ----------------------------------------------
        bw = self._bounded_severity_window(p)
        if bw is not None:
            a_lo, a_hi = bw
            rows['bounded_small'] = _row(a_lo, a_hi, lattice, f'freq 1-1e-{WINDOW_NINES}',
                                         'bounded severity x freq quantile')

        # ---- selection (D1/D2) ------------------------------------------
        # bounded_small is selected a bit permissively -- it is a hard support
        # bound, so accept it even when modestly wider (1.5x) than the moment
        # window. With high coverage it is typically the tighter of the two.
        if 'exact_discrete' in rows:
            selected = 'exact_discrete'
        elif ('bounded_small' in rows
              and rows['bounded_small']['W'] <= 1.5 * rows['moment']['W']):
            selected = 'bounded_small'
        else:
            selected = 'moment'

        # ---- realized grid (the ``used`` row) ---------------------------
        sel_bs = float(rows[selected]['bs'])
        sel_l2 = int(rows[selected]['log2'])
        sel_x0 = float(rows[selected]['x_min'])
        if x_min_in is not None:           # explicit origin override (D4)
            sel_x0 = float(round(x_min_in / sel_bs) * sel_bs)
        grid_x_max = sel_x0 + (1 << sel_l2) * sel_bs

        # ---- aggregate affine (pnl): tight, mass-centred P&L window ------
        # The loss FFT runs on the non-negative grid sized above; the finished
        # aggregate is then reflected and shifted (``_apply_agg_affine``). The
        # *display* window is a tight two-sided window around the P&L mean
        # (``estimate_agg_window`` on the affine moments), not the full reversed
        # loss grid -- so the mass sits in the window (and the Portfolio combine,
        # which positions the shared grid from the per-unit origins, places a
        # book of ``pnl`` units correctly).
        affine = self._agg_affine_active()
        if affine:
            N = 1 << sel_l2
            x0_pnl = self._pnl_window(sel_bs, N)
            x_max_pnl = float(x0_pnl + N * sel_bs)
            # Hand ``update`` the *loss-convolution* origin, not the P&L display
            # origin: 0 for an ordinary (non-negative) loss -- byte-for-byte the
            # legacy 0-based grid -- and the signed loss origin ``sel_x0`` when the
            # loss severity is itself signed (a ``pnl`` over a negative-atom
            # ``dsev`` / ``ssev``). The P&L display window ``x0_pnl`` is recomputed
            # independently in ``_apply_agg_affine`` and reported in the ``used``
            # row below.
            ret_x0 = sel_x0 if self._signed_severity() else 0.0
        else:
            ret_x0 = sel_x0

        df = pd.DataFrame(rows).T
        # the winning method is flagged; the ``used`` row is the realized grid.
        df['selected'] = df.index == selected
        # The ``used`` row converts the selected method's window into the actual
        # power-of-2 grid: x_max = x_min + 2**log2 * bs (so a 701-point support
        # padded to 1024 reads x_max = grid top, W = the full grid width). For a
        # ``pnl`` aggregate the ``used`` row reports the reflected+shifted P&L
        # window (the loss method rows keep their non-negative loss windows).
        if affine:
            df.loc['used'] = dict(
                applies=True, x_min=x0_pnl, x_max=x_max_pnl,
                W=float(x_max_pnl - x0_pnl), bs=sel_bs, log2=sel_l2,
                coverage=rows[selected]['coverage'],
                note=f'realized P&L grid ({selected}, reflect+shift)',
                selected=False)
        else:
            df.loc['used'] = dict(
                applies=True, x_min=sel_x0, x_max=float(grid_x_max),
                W=float(grid_x_max - sel_x0), bs=sel_bs, log2=sel_l2,
                coverage=rows[selected]['coverage'],
                note=f'realized grid ({selected})', selected=False)
        self._bs_window_df = df

        return sel_bs, sel_l2, ret_x0

    def recommend_bucket(self, log2=10, p=BUCKET_SIZING_P, verbose=False):
        """
        Recommend a bucket size given 2**N buckets. Not rounded.

        For thick tailed distributions need higher p, try p=1-1e-8.

        If no second moment, throws a ValueError. You just can't guess
        in that situation.

        :param log2: log2 of number of buckets. log2=10 is default.
        :param p: percentile to use to determine needed range. Default is BUCKET_SIZING_P. if > 1 converted to 1-10**-n.
        :param verbose: print out recommended bucket sizes for 2**n for n in {log2, 16, 13, 10}
        :return:
        """
        N = 1 << log2
        if not verbose:
            limit_est = self.limit.max() / N
            if limit_est == np.inf:
                limit_est = 0
                p = max(p, 1 - 10 ** -8)
            moment_est = _estimate_agg_percentile(self.agg_m, self.agg_cv, self.agg_skew, p=p) / N
            logger.debug('Agg.recommend_bucket | %s moment: %s, limit %s',
                         self.name, moment_est, limit_est)
            recommended = max(moment_est, limit_est)
        else:
            for n in sorted({log2, 16, 13, 10}):
                rb = self.recommend_bucket(n)
                if n == log2:
                    rbr = rb
                print(f'Recommended bucket size with {2 ** n} buckets: {rb:,.3f}')
            if self.bs != 0:
                print(f'Bucket size set with {N} buckets at {self.bs:,.3f}')
            recommended = rbr # noqa
        # can fail when distribution is constant
        return 1 if np.isnan(recommended) else recommended

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
            cself.update(bs=bs, log2=log2, **kwargs)
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
            # exact theoretical sev mean from the canonical stats_df: the
            # per-component column label is ``e{e}.m{m}`` (exposure × sev-
            # mixture), enumerated in order by both broadcasting arms.
            label = self._comp_cols[i]
            m = self.stats_df.loc[('sev', 'ex1'), label]
            if len(self.sevs) == 1:
                i = self.name
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
                        self.sf(truncation_point), self.sev_density_df.p_sev.sum(),
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
            [_integral_by_doubling(s.sf, truncation_point) for s in self.sevs] + \
            [_integral_by_doubling(self.sev.sf, truncation_point)]
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
            # revised June 2023; severity now on its own grid (sev_density_df)
            ser = self.sev_density_df.query('p_sev > 0').p_sev
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
                    wts = self.stats_df.loc[('freq', 'ex1'), self._comp_cols].values
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

        # Prefer empirical moments (post-update) over theoretical (pre-update).
        emp = self.stats_df['empirical']
        if pd.notna(emp.get(('agg', 'mean'), np.nan)):
            m, cv, skew = (emp[('agg', 'mean')], emp[('agg', 'cv')], emp[('agg', 'skew')])
        else:
            mixed = self.stats_df['mixed']
            m, cv, skew = (mixed[('agg', 'mean')], mixed[('agg', 'cv')], mixed[('agg', 'skew')])

        name = f'{approx_type[0:4]}.{self.name[0:5]}'
        agg_str = f'agg {name} 1 claim sev '
        note = f'frozen version of {self.name}'
        return approximate_from_mcvsk(m, cv, skew, name, agg_str, note, approx_type, output)

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
        # de-fuzz before the moment fit (threshold standardized on machine eps)
        p = remove_fuzz(p)
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

        el = aug_row['exa']
        P = aug_row['exag']
        M = P - el
        Q = a_reg - P

        # one-row canonical pentagon (pentagon.py owns the identities + order)
        df = pd.DataFrame(
            [[el, M, P, Q]], columns=['L', 'M', 'P', 'Q'],
            index=pd.Index([self.name], name='line'),
        )
        return complete_pentagon(df)

    def price_pentagon(self, *, p=None, a=None, P=None, M=None, Q=None,
                       LR=None, PQ=None, ROE=None):
        """Complete the pricing octet at a capital level given one target.

        Fix the capital level with exactly one of ``p`` (a VaR probability,
        ``a = self.q(p)``) or ``a`` (an asset level, snapped to the grid), then
        supply exactly one pricing target -- premium ``P``, cost of capital
        ``ROE`` (a.k.a. CoC), or a loss ratio via ``LR`` (equivalently ``M``,
        ``Q`` or ``PQ``). Returns the canonical one-row (``'total'``) pentagon
        ``DataFrame`` (columns :data:`~aggregate.pentagon.PENTAGON_STATS`).
        The target keywords match the canonical stat names.

        Pure accounting completion against the object's expected loss at the
        chosen capital level -- **no distortion is involved** (contrast
        :meth:`price`, which prices with a :class:`Distortion`). The triple
        ``{L, a, target}`` is solved by :meth:`Pentagon.solve`.

        Parameters
        ----------
        p : float, optional
            VaR probability fixing the capital level; mutually exclusive with ``a``.
        a : float, optional
            Asset level fixing the capital; mutually exclusive with ``p``.
        P, M, Q, LR, PQ, ROE : float, optional
            Exactly one pricing target -- premium, margin, capital, loss ratio,
            premium-to-capital, or cost of capital (``M/Q``).

        Returns
        -------
        pandas.DataFrame
            One ``'total'`` row, eight canonical pentagon columns.

        Raises
        ------
        ValueError
            If not exactly one of ``p``/``a`` is given, or not exactly one
            pricing target is supplied.
        """
        if (p is None) == (a is None):
            raise ValueError('price_pentagon: pass exactly one of p= or a=')
        targets = {'P': P, 'M': M, 'Q': Q, 'LR': LR, 'PQ': PQ, 'ROE': ROE}
        n_targets = sum(v is not None for v in targets.values())
        if n_targets != 1:
            raise ValueError(
                'price_pentagon: pass exactly one pricing target '
                f'(one of P, M, Q, LR, PQ, ROE); got {n_targets}.')
        pent = Pentagon(obj=self)
        pent.solve_obj(p=p, a=a, P=P, M=M, Q=Q, lr=LR, pq=PQ, roe=ROE)
        return pent.as_frame(line='total')



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
    # Local import to avoid the distributions <-> portfolio cycle.
    from .portfolio import Portfolio

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


class Severity(ss.rv_continuous):
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
                 sev_conditional=True, sev_signed=False, sev_reflect=False, name='', note='', hints=''):
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
        self.long_name = sev_name
        self.note = note
        self.hints = hints
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
        if self.sev_reflect:
            # Reflected severity ``shift - X``: build the positive base (loc 0),
            # then reflect into signed support. ``_apply_reflect`` sets the raw
            # moments and the reflected fz methods; ``_apply_signed`` then wires
            # the identity (no-clamp) layering from them.
            self._apply_lb_ub()
            self._apply_reflect()
            self._apply_signed()
        elif self.signed:
            # Signed (never-clamp) severity: keep the raw distribution. Honour
            # an explicit splice, but skip the x<0 -> 0 / attachment clamp and
            # the layered-loss transform -- the layered methods are the raw fz
            # methods (identity). Moments are the raw distribution's; discrete
            # kinds already populated sev1/sev2/sev3 in ``_build``.
            self._apply_lb_ub()
            self._apply_signed()
        else:
            # Order is load-bearing: splice (lb/ub) modifies the underlying
            # distribution FIRST, then attachment probabilities are computed
            # against the already-spliced fz, then validation, then the policy
            # layer wraps on top of everything.
            self._apply_lb_ub()
            self._compute_attachment_probs()
            self._validate_moments()
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

    @property
    def support_description(self):
        """Short human description of the severity for ``info`` display.

        For a histogram / discrete severity, render the **actual support** --
        ``atoms {a, b, …}`` (shortened to first/last few when there are many,
        e.g. ``dsev [1:1001]``) or ``support [lo, hi]`` -- instead of the
        layer form ``limit xs attachment``, which is meaningless for a discrete
        or signed severity (e.g. ``dsev [-2 5]`` is *not* a ``5 xs 0`` layer).
        For a continuous severity keep the familiar ``unlimited`` /
        ``limit xs attachment`` rendering.

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

        Used for ``-1 * dist (+/- shift)`` (e.g. ``100 - lognorm``). Marks the
        severity signed so the no-clamp path in ``_apply_signed`` is used.
        """
        Z = self.fz
        d = float(self._reflect_shift)
        # raw moments first (before the method swap below)
        z1 = float(Z.moment(1)); z2 = float(Z.moment(2)); z3 = float(Z.moment(3))
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
        self.signed = True

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
            # continuous (scipy) signed severity: raw non-central moments
            self.sev1 = float(self.fz.moment(1))
            self.sev2 = float(self.fz.moment(2))
            self.sev3 = float(self.fz.moment(3))
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
        """
        if not (self.sev_mean > 0 or self.sev_cv > 0):
            return
        mean, var = self.fz.stats('mv')
        acv = var ** .5 / mean
        if self.sev_mean > 0 and not np.isclose(self.sev_mean + self.sev_loc, mean):
            print(f'WARNING target mean {self.sev_mean} and achieved mean {mean} not close')
        if self.sev_cv > 0 and not np.isclose(
                self.sev_cv * self.sev_mean / (self.sev_mean + self.sev_loc), acv):
            print(f'WARNING target cv {self.sev_cv} and achieved cv {acv} not close')
        logger.debug(
            f'Severity.__init__ | parameters {self.sev_a}, {self.sev_scale}: '
            f'target/actual {self.sev_mean} vs {mean};  {self.sev_cv} vs {acv}')

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
        if (isinstance(self.sev_name, str)
                and self.sev_name in ('lognorm', 'pareto', 'gamma', 'expon')
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

        # ``fixed`` is a degenerate dhistogram (single point mass); both
        # benefit from the step-post draw style. Continuous histograms and
        # the scipy zoo use ordinary line plots.
        ds = 'steps-post' if isinstance(self, SeverityDHistogram) else 'default'

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

    @staticmethod
    def _unwrap(out):
        """Return a numpy scalar for 0-d results, else the array unchanged."""
        return out[()] if out.ndim == 0 else out

    def cdf(self, x):
        """Right-continuous CDF ``P(X <= x)``."""
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.xk, x, side='right')   # count of atoms <= x
        out = np.where(idx == 0, 0.0, self.cum[np.clip(idx - 1, 0, self.n - 1)])
        return self._unwrap(out)

    def sf(self, x):
        """Survival function ``P(X > x) = 1 - cdf(x)``."""
        return self._unwrap(1.0 - np.asarray(self.cdf(x), dtype=float))

    def pdf(self, x):
        """Density of a discrete law: identically ``0`` (mass lives in atoms)."""
        out = np.zeros_like(np.asarray(x, dtype=float))
        return self._unwrap(out)

    def ppf(self, q):
        """Quantile: the smallest atom ``xk`` with ``cdf(xk) >= q``."""
        q = np.asarray(q, dtype=float)
        idx = np.clip(np.searchsorted(self.cum, q, side='left'), 0, self.n - 1)
        return self._unwrap(self.xk[idx])

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
        return float(np.sum(self.xk * self.pk))

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
        # for the info display (rendered by support_description).
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
        self.fz = ss.rv_histogram((aps, xss))


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
        # Local import to avoid the distributions <-> portfolio cycle.
        from .portfolio import Portfolio

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
        self.fz = ss.rv_histogram((pss, xss))
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
