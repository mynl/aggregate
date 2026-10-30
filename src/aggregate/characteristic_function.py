r"""Distributions computed by inverting a characteristic function.

``AggregateCHF`` computes a distribution on a grid by sampling its
characteristic function on the FFT's frequency grid and inverting with one
``irfft``, instead of discretizing a severity and applying a frequency
generating function. The families are infinitely divisible laws with
closed-form characteristic functions: the alpha-stable laws, the whole
Tweedie family, and the normal, gamma, inverse Gaussian, normal inverse
Gaussian, variance gamma and tempered stable laws
(``dev/done/plan-characteristic-function-engine.md``,
``[Characteristic-Function-Engine]``). The DecL statement is ``chfagg``.

Reached by submodule, ``from aggregate.characteristic_function import
AggregateCHF``, as ``Tweedie``, ``Stable`` and ``AggregateNTT`` are.

**Bucket masses, not density samples.** On the grid :math:`x_k = x_0 + kh`
with :math:`U` uniform on :math:`[-h/2, h/2]` and independent of :math:`X`,
:math:`\Pr(X \in [x_k - h/2, x_k + h/2)) = h f_{X+U}(x_k)` and
:math:`\varphi_{X+U}(t) = \varphi_X(t)\,\mathrm{sinc}(th/2)`. Inverting
:math:`\varphi_X \cdot \mathrm{sinc}` therefore returns the bucket
probabilities of the round discretization the compound engine uses, so a
``chfagg`` and an ``agg`` on one grid agree bucket for bucket up to frequency
truncation (frequencies beyond :math:`\pi/h` are dropped) and aliasing (mass
outside the window wraps onto it).

**Window edges from the tails.** Each side of every family in scope is light
(a finite cumulant generating function :math:`K`, giving the Chernoff edge
:math:`x_R = \inf_s (K(s) - \log\varepsilon)/s`) or a power tail (stable,
:math:`\Pr(X > x) \sim A x^{-\alpha}`, giving
:math:`x_R = \delta_1 + (A/\varepsilon)^{1/\alpha}`). Rule A sizes the window
from these edges at the mass ``1 - bucket_sizing_p`` split equally between the
tails and rounds the bucket with :func:`aggregate.utilities.round_bucket`;
rule C uses a given ``bs`` and ``log2``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.optimize import minimize_scalar
from scipy.special import gamma as gamma_function
from scipy.special import zeta as hurwitz_zeta

from ._aggregate import (_STATS_ROW_INDEX, SUMMARY_PERCENTILES, Aggregate,
                         _summary_pct_label, return_period_frame)
from ._bucket_window import BUCKET_SIZING_P
from ._grid_distribution import GridDistribution
from ._help import HelpMixin
from ._labeled import LabeledMixin
from ._program import ProgramMixin
from .config import get_settings
from .constants import INFO_NA, info_row
from .moments import _noise_aware_rel_error, _snap_noise
from .stable import Stable
from .utilities import _round_bucket_down, round_bucket

__all__ = ['AggregateCHF', 'Family', 'FamilyNormal', 'FamilyGamma',
           'FamilyInverseGaussian', 'FamilyStable', 'FamilyUser', 'FamilyAffine',
           'FAMILY_PARAMETERS', 'FAMILY_CV_PARAMETERS', 'make_family',
           'arithmetic_spelling', 'invert_characteristic_function', 'chernoff_edge']

logger = logging.getLogger(__name__)

#: The shape parameters of each family's DecL body, in order: the positional
#: numbers a ``chfagg`` takes, scipy's where scipy has the law. Scale and
#: location are DecL arithmetic, ``c * FAMILY shapes + b``, held in the spec
#: as ``chf_scale`` and ``chf_loc`` ([CHF-Arithmetic-Form]). ``tweedie``
#: keeps its reproductive parameters; the variance gamma and tempered stable
#: laws keep their published ones, less the location. The classes also take
#: the full parameter set as keywords from Python.
FAMILY_PARAMETERS = {
    'stable': ('alpha', 'beta'),
    'tweedie': ('p', 'mean', 'dispersion'),
    'norm': (),
    'gamma': ('shape',),
    'invgauss': ('mu',),
    'norminvgauss': ('a', 'b'),
    'variancegamma': ('sigma', 'nu', 'theta'),
    'temperedstable': ('C', 'G', 'M', 'Y'),
}

#: How many shapes each family takes; every shape is required.
FAMILY_REQUIRED = {k: len(v) for k, v in FAMILY_PARAMETERS.items()}

#: The positional parameters the ``chfagg`` body took before 1.35.0, scale
#: and location among them, kept to compute the arithmetic spelling a
#: refused body points at (:func:`arithmetic_spelling`).
LEGACY_PARAMETERS = {
    'stable': ('alpha', 'beta', 'scale', 'loc'),
    'norm': ('mean', 'sd'),
    'gamma': ('shape', 'scale'),
    'invgauss': ('mean', 'shape'),
    'norminvgauss': ('alpha', 'beta', 'delta', 'mu'),
    'variancegamma': ('sigma', 'nu', 'theta', 'mu'),
    'temperedstable': ('C', 'G', 'M', 'Y', 'mu'),
}

#: The ``mean cv`` form of the families that have one.
FAMILY_CV_PARAMETERS = {
    'tweedie': ('mean', 'cv'),
    'norm': ('mean', 'cv'),
    'gamma': ('mean', 'cv'),
    'invgauss': ('mean', 'cv'),
}

#: Buckets per body scale below which the resolution row is flagged.
MIN_BUCKETS_PER_SCALE = 10

#: Frequency truncation :math:`|\varphi(\pi/h)|` above which the row is flagged.
TRUNCATION_FLAG = 1e-6

#: Largest ``log2`` :meth:`AggregateCHF.recommended_grid` will suggest.
RECOMMENDED_LOG2_MAX = 24

#: A finite support bound replaces a Chernoff edge within this share of the
#: window of it (:func:`window_edges`).
SUPPORT_EDGE_SHARE = 0.05

#: The mass allowed below the window of a tilted law: the tilt amplifies
#: whatever wraps from below by ``exp(tau)``, and the light side's
#: super-exponential tail makes this edge cost a few scales.
TILT_LEFT_EPS = 1e-40

#: Buckets per scale a tilted grid keeps in the body: the bucket cap is
#: ``min(scale / TILT_BODY_BUCKETS, h_trunc)``.
TILT_BODY_BUCKETS = 10

#: ``log(2 ** 52)``: a tilted bucket needs ``|phi(pi / h)|`` at roundoff.
_ROUNDOFF_LOG = 52 * math.log(2)

#: Modeled worst relative bucket error above which the edge error row is flagged.
EDGE_ERROR_FLAG = 1e-3

#: The moment store's columns, ``Aggregate.stats_df``'s
#: ([CHF-Aggregate-Frames]).
STATS_COLUMNS = ['e0.m0', 'mixed', 'independent', 'after_occ', 'empirical',
                 'occ_impact', 'agg_impact', 'gross_empirical', 'error']

#: The six measures of a moment store block.
_MEASURES = ('ex1', 'ex2', 'ex3', 'mean', 'cv', 'skew')

#: The moment terms of a ``sharpen_df`` row, ``Aggregate.sharpen_df``'s, and
#: the four terms of the inversion appended after its columns ([CHF-Sharpen]).
_SHARPEN_MOMENT_TERMS = ('u_sev_mean', 'u_sev_cv', 'u_sev_skew', 'u_agg_mean',
                         'u_agg_cv', 'u_agg_skew')
_SHARPEN_INVERSION_TERMS = ('u_coverage', 'u_edge', 'u_truncation', 'u_resolution')


# ======================================================================
# the kernel
# ======================================================================

def _csinc(z):
    r"""``sin(z) / z`` for complex ``z``, with the value 1 at ``z = 0``.

    ``numpy.sinc`` is the normalized form :math:`\sin(\pi x)/(\pi x)` and is
    real only; the bucket factor needs the plain form at a complex frequency.
    """
    z = np.asarray(z, dtype=complex)
    out = np.ones_like(z)
    nz = z != 0
    out[nz] = np.sin(z[nz]) / z[nz]
    return out


def invert_characteristic_function(chf, x0, bs, n, theta=0.0, atom=None,
                                   bucket=True):
    r"""Bucket masses of a law on ``x0 + bs * arange(n)`` from its characteristic function.

    Parameters
    ----------
    chf : callable
        The characteristic function :math:`\varphi(s) = \mathsf{P}e^{isX}`,
        evaluated on a complex array. With ``theta = 0`` it is called on the
        real line only; a nonzero tilt calls it on the line
        :math:`\mathrm{Im}\,s = \theta`, where it must be the analytic
        continuation.
    x0 : float
        The left edge of the grid.
    bs : float
        The bucket ``h``.
    n : int
        The number of buckets, a power of two.
    theta : float, default 0
        The tilt :math:`\theta \ge 0`: inversion along the contour
        :math:`\mathrm{Im}\,s = \theta`.
    atom : tuple of (float, float), optional
        ``(x, w)``: a point mass ``w`` at the grid point ``x``, removed from
        :math:`\varphi` before inversion and added back to its bucket, so
        frequency truncation does not ring around it.
    bucket : bool, default True
        Apply the sinc bucket factor, so the result is bucket masses. ``False``
        returns ``h`` times density samples, which the two-sided split uses
        for the half that is convolved with the other.

    Returns
    -------
    numpy.ndarray
        The ``n`` masses, real.

    Notes
    -----
    On the real line the bucket mass is
    :math:`p_k = (1/2\pi)\int \varphi(t)\,h\,\mathrm{sinc}(th/2)\,e^{-itx_k}\,dt`.
    The integrand is analytic in the strip :math:`0 \le \mathrm{Im}\,s \le \theta`
    whenever the left tail is light, so the line moves to
    :math:`\mathrm{Im}\,s = \theta`, where :math:`e^{-isx_k} = e^{-itx_k}e^{\theta x_k}`:

    .. math::

        e^{-\theta x_k}p_k = \frac{1}{2\pi}\int \varphi(t+i\theta)\,h\,
        \mathrm{sinc}\big((t+i\theta)h/2\big)\,e^{-itx_k}\,dt .

    The discretized integral wraps the damped sequence :math:`e^{-\theta x}p`,
    not the heavy tail, and multiplying by :math:`e^{\theta x_k}` recovers the
    masses. ``irfft`` sums with :math:`e^{+2\pi i jk/n}`, so the coefficient at
    frequency :math:`t_j` is :math:`\varphi(-t_j + i\theta)`, which is the
    conjugate of :math:`\varphi(t_j + i\theta)` for a real tilted sequence.
    :math:`\theta = 0` is the untilted kernel, so there is one kernel.
    """
    n = int(n)
    t = 2 * np.pi * np.arange(n // 2 + 1) / (n * bs)
    s = -t + 1j * theta
    c = np.asarray(chf(s), dtype=complex)
    if atom is not None:
        xa, wa = atom
        c = c - wa * np.exp(1j * s * xa)
    if bucket:
        c = c * _csinc(s * bs / 2)
    c = c * np.exp(1j * t * x0)
    x = x0 + bs * np.arange(n)
    p = np.fft.irfft(c, n)
    if theta != 0:
        p = p * np.exp(theta * x)
    if atom is not None:
        xa, wa = atom
        k = int(round((xa - x0) / bs))
        if not (0 <= k < n and abs(x0 + k * bs - xa) <= 1e-9 * max(bs, 1.0)):
            raise ValueError(
                f'atom at {xa} is not a grid point of x0={x0}, bs={bs}, n={n}')
        p[k] += wa
    return p


# ======================================================================
# window edges
# ======================================================================

def chernoff_edge(cgf, s_max, log_eps, scale):
    r"""The Chernoff edge :math:`\inf_{0<s<s_{\max}} (K(s) - \log\varepsilon)/s`.

    Parameters
    ----------
    cgf : callable
        :math:`K(s) = \log\mathsf{P}e^{sX}` on an array of ``s > 0``,
        ``nan`` outside its domain.
    s_max : float
        The right end of the domain, possibly ``inf``.
    log_eps : float
        :math:`\log\varepsilon`, negative.
    scale : float
        A body scale of the law, setting the search range for ``s``.

    Returns
    -------
    float
        The edge :math:`x` with :math:`\Pr(X > x) \le \varepsilon` by the
        Chernoff bound, which is conservative. The left edge of a law is the
        negative of this value computed for :math:`-X`, that is with
        ``cgf(s) = K(-s)``.

    Notes
    -----
    :math:`g(s) = (K(s) - \log\varepsilon)/s` is scanned on a logarithmic
    ladder of about two hundred points between :math:`10^{-9}` and
    :math:`10^{9}` body scales and the best point is refined by a bounded
    Brent step between its neighbors. The scan costs two hundred evaluations
    of a closed form and needs no derivative or convexity.
    """
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    s = np.logspace(-9, 9, 217) / scale
    s = s[s < s_max]
    if len(s) == 0:
        return np.inf
    with np.errstate(all='ignore'):
        g = (np.asarray(cgf(s), dtype=float) - log_eps) / s
    ok = np.isfinite(g)
    if not ok.any():
        return np.inf
    g = np.where(ok, g, np.inf)
    i = int(np.argmin(g))
    lo = s[max(i - 1, 0)]
    hi = s[min(i + 1, len(s) - 1)]
    if hi > lo:
        def objective(u):
            k = float(cgf(np.array([u]))[0])
            return (k - log_eps) / u if np.isfinite(k) else np.inf
        with np.errstate(all='ignore'):
            res = minimize_scalar(objective, bounds=(lo, hi), method='bounded')
        if res.success and np.isfinite(res.fun) and res.fun < g[i]:
            return float(res.fun)
    return float(g[i])


def chernoff_tail(cgf, s_max, x, scale):
    r"""The Chernoff bound :math:`\exp\inf_{0<s<s_{\max}}(K(s) - sx)` on :math:`\Pr(X > x)`.

    The companion of :func:`chernoff_edge`, with the same scan and refinement;
    the left tail is this with ``cgf(s) = K(-s)`` at ``-x``. The bound is 1
    where no ``s`` improves on it.
    """
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    s = np.logspace(-9, 9, 217) / scale
    s = s[s < s_max]
    if len(s) == 0:
        return 1.0
    with np.errstate(all='ignore'):
        g = np.asarray(cgf(s), dtype=float) - s * x
    ok = np.isfinite(g)
    if not ok.any():
        return 1.0
    g = np.where(ok, g, np.inf)
    i = int(np.argmin(g))
    lo = s[max(i - 1, 0)]
    hi = s[min(i + 1, len(s) - 1)]
    best = float(g[i])
    if hi > lo:
        def objective(u):
            k = float(cgf(np.array([u]))[0])
            return k - u * x if np.isfinite(k) else np.inf
        with np.errstate(all='ignore'):
            res = minimize_scalar(objective, bounds=(lo, hi), method='bounded')
        if res.success and np.isfinite(res.fun) and res.fun < best:
            best = float(res.fun)
    return float(min(1.0, math.exp(min(best, 0.0))))


# ======================================================================
# families
# ======================================================================

def _decl_number(x):
    """A DecL number: an integral value without a decimal point, else twelve significant digits."""
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f'{x:.12g}'


def render_affine(c, b, body):
    """``c * body + b`` as DecL, a unit scale and a zero shift omitted, a negative shift written ``- |b|``."""
    out = body if c == 1 else f'{_decl_number(c)} * {body}'
    if b > 0:
        out += f' + {_decl_number(b)}'
    elif b < 0:
        out += f' - {_decl_number(-b)}'
    return out


class Family:
    r"""The closed forms of one law, as the engine needs them.

    A family supplies the characteristic function, the cumulant generating
    function and its domain, the power tails where there are any, the
    support, an atom, a body scale, the moments, a scipy reference law where
    one exists, and the DecL body as written. Subclasses are the laws; the
    engine reads the protocol and never the law.

    Parameters
    ----------
    parameters : dict
        The parameters as written in DecL, keyed by name.
    form : str or None
        ``'cv'`` for the ``mean cv`` form of a two-form family, else ``None``.

    Attributes
    ----------
    name : str
        The DecL family word.
    cgf_domain : tuple of float
        ``(s_minus, s_plus)`` with :math:`K` finite strictly inside; a power
        side has 0 as its end.
    power_tails : tuple or None
        ``(A, B, alpha)``, the right and left tail constants and the index,
        or ``None`` for a law with no power tail.
    tilt_side : str or None
        The side of the heavy tail, ``'right'``, ``'left'``, ``'split'``
        (both) or ``None`` (neither).
    location : float
        The shift the power edges are measured from, :math:`\delta_1`.
    support : tuple of float
        ``(lower, upper)`` of the law.
    atom : tuple or None
        ``(x, w)`` for a point mass, else ``None``.
    """

    name = ''
    cgf_domain = (-np.inf, np.inf)
    power_tails = None
    tilt_side = None
    location = 0.0
    support = (-np.inf, np.inf)
    atom = None

    def __init__(self, parameters, form=None):
        self.parameters = dict(parameters)
        self.form = form

    def chf(self, s):
        """The characteristic function on a complex array ``s``."""
        raise NotImplementedError

    def cgf(self, s):
        """The cumulant generating function on a real array ``s``, ``nan`` outside its domain."""
        raise NotImplementedError

    @property
    def scale(self):
        """The body scale: the standard deviation when finite, else the law's scale."""
        mean, var, skew = self.moments
        return float(np.sqrt(var)) if np.isfinite(var) and var > 0 else np.nan

    @property
    def moments(self):
        """``(mean, variance, skewness)``, ``inf`` or ``nan`` where they do not exist."""
        return (np.nan, np.nan, np.nan)

    @property
    def reference(self):
        """A scipy frozen law, or ``None``."""
        return None

    def _affine_parts(self):
        """``(c, b, body)``: the law is ``c * body + b``, ``body`` the family word and its shapes, or its ``mean cv`` form."""
        raise NotImplementedError

    def location_under(self, c, b):
        r"""The :attr:`location` of :math:`cX + b`: :math:`c\delta_1 + b`, overridden where the location is not equivariant."""
        return c * self.location + b

    def to_decl(self):
        """The DecL body, ``c * FAMILY shapes + b``, a unit scale and a zero shift omitted."""
        return render_affine(*self._affine_parts())

    def __repr__(self):
        return f'{type(self).__name__}({self.to_decl()!r})'

    # ---- helpers shared by the two-form families -----------------------
    def _cv_decl(self):
        """``name mean cv c``, the ``mean cv`` form as written."""
        a, b = FAMILY_CV_PARAMETERS[self.name]
        return (f'{self.name} {_decl_number(self.parameters[a])} cv '
                f'{_decl_number(self.parameters[b])}')


class FamilyNormal(Family):
    r"""The normal law, ``norm`` (standard, no shapes) or ``norm mean cv c``.

    :math:`\varphi(s) = \exp(i\mu s - \sigma^2 s^2/2)`, entire;
    :math:`K(s) = \mu s + \sigma^2 s^2/2` on the whole line. DecL writes
    :math:`N(\mu, \sigma^2)` as ``sigma * norm + mu``; from Python the keywords
    ``mean`` and ``sd`` default to 0 and 1.
    """

    name = 'norm'

    def __init__(self, parameters, form=None):
        super().__init__(parameters, form)
        p = self.parameters
        self.mean = float(p.get('mean', 0.0))
        self.sd = float(self.mean * p['cv']) if form == 'cv' else float(p.get('sd', 1.0))
        if self.sd <= 0:
            raise ValueError(f'norm: the standard deviation must be positive, got {self.sd}')

    def chf(self, s):
        s = np.asarray(s, dtype=complex)
        return np.exp(1j * self.mean * s - 0.5 * self.sd ** 2 * s ** 2)

    def cgf(self, s):
        s = np.asarray(s, dtype=float)
        return self.mean * s + 0.5 * self.sd ** 2 * s ** 2

    @property
    def moments(self):
        return (self.mean, self.sd ** 2, 0.0)

    @property
    def reference(self):
        return ss.norm(loc=self.mean, scale=self.sd)

    def _affine_parts(self):
        if self.form == 'cv':
            return 1.0, 0.0, self._cv_decl()
        return self.sd, self.mean, 'norm'


class FamilyGamma(Family):
    r"""The gamma law, ``gamma shape`` (scale 1) or ``gamma mean cv c``; ``scale`` from Python.

    :math:`\varphi(s) = (1 - i\theta s)^{-k}` on the principal branch, which
    is analytic on the upper half plane since
    :math:`\mathrm{Re}(1 - i\theta s) = 1 + \theta\,\mathrm{Im}\,s > 0` there;
    :math:`K(s) = -k\log(1 - \theta s)` for :math:`s < 1/\theta`. A shape
    below 1 has a density pole at 0, so :math:`|\varphi|` decays only as
    :math:`t^{-k}` and the truncation row flags it.
    """

    name = 'gamma'
    support = (0.0, np.inf)

    def __init__(self, parameters, form=None):
        super().__init__(parameters, form)
        p = self.parameters
        if form == 'cv':
            mean, cv = float(p['mean']), float(p['cv'])
            self.shape = 1 / cv ** 2
            self.theta = mean * cv ** 2
        else:
            self.shape, self.theta = float(p['shape']), float(p.get('scale', 1.0))
        if self.shape <= 0 or self.theta <= 0:
            raise ValueError('gamma: shape and scale must be positive, got '
                             f'{self.shape} and {self.theta}')
        self.cgf_domain = (-np.inf, 1 / self.theta)

    def chf(self, s):
        s = np.asarray(s, dtype=complex)
        return (1 - 1j * self.theta * s) ** (-self.shape)

    def cgf(self, s):
        s = np.asarray(s, dtype=float)
        out = np.full(s.shape, np.nan)
        ok = s < 1 / self.theta
        out[ok] = -self.shape * np.log1p(-self.theta * s[ok])
        return out

    @property
    def moments(self):
        k, th = self.shape, self.theta
        return (k * th, k * th ** 2, 2 / np.sqrt(k))

    @property
    def reference(self):
        return ss.gamma(self.shape, scale=self.theta)

    def _affine_parts(self):
        if self.form == 'cv':
            return 1.0, 0.0, self._cv_decl()
        return self.theta, 0.0, f'gamma {_decl_number(self.shape)}'


class FamilyInverseGaussian(Family):
    r"""The inverse Gaussian law, ``invgauss mu`` (scipy's shape) or ``invgauss mean cv c``.

    With mean :math:`\mu` and shape :math:`\lambda`,
    :math:`K(s) = (\lambda/\mu)\big(1 - \sqrt{1 - 2\mu^2 s/\lambda}\big)` for
    :math:`s < \lambda/(2\mu^2)` and :math:`\varphi(s) = \exp K(is)` on the
    principal square root, whose argument has positive real part on the
    upper half plane. Variance :math:`\mu^3/\lambda`, skewness
    :math:`3\sqrt{\mu/\lambda}`. The scipy law is
    ``invgauss(mu / lam, scale=lam)``, so DecL's ``invgauss mu`` is the law
    with mean ``mu`` and shape 1, and ``lam * invgauss mu`` has mean
    ``lam * mu`` and shape ``lam``. From Python the keywords ``mean`` and
    ``shape`` give the law directly.
    """

    name = 'invgauss'
    support = (0.0, np.inf)

    def __init__(self, parameters, form=None):
        super().__init__(parameters, form)
        p = self.parameters
        if form == 'cv':
            self.mean = float(p['mean'])
            self.lam = self.mean / float(p['cv']) ** 2
        elif 'mu' in p:
            self.mean, self.lam = float(p['mu']), 1.0
        else:
            self.mean, self.lam = float(p['mean']), float(p['shape'])
        if self.mean <= 0 or self.lam <= 0:
            raise ValueError('invgauss: mean and shape must be positive, got '
                             f'{self.mean} and {self.lam}')
        self.cgf_domain = (-np.inf, self.lam / (2 * self.mean ** 2))

    def chf(self, s):
        s = np.asarray(s, dtype=complex)
        mu, lam = self.mean, self.lam
        return np.exp(lam / mu * (1 - np.sqrt(1 - 2j * mu ** 2 * s / lam)))

    def cgf(self, s):
        s = np.asarray(s, dtype=float)
        mu, lam = self.mean, self.lam
        out = np.full(s.shape, np.nan)
        ok = s < lam / (2 * mu ** 2)
        out[ok] = lam / mu * (1 - np.sqrt(1 - 2 * mu ** 2 * s[ok] / lam))
        return out

    @property
    def moments(self):
        mu, lam = self.mean, self.lam
        return (mu, mu ** 3 / lam, 3 * np.sqrt(mu / lam))

    @property
    def reference(self):
        return ss.invgauss(self.mean / self.lam, scale=self.lam)

    def _affine_parts(self):
        if self.form == 'cv':
            return 1.0, 0.0, self._cv_decl()
        return self.lam, 0.0, f'invgauss {_decl_number(self.mean / self.lam)}'


class FamilyStable(Family):
    r"""The alpha-stable law, ``stable alpha beta [s0|s1]``; ``scale`` and ``loc`` from Python.

    DecL writes :math:`\gamma Z + \delta` for the standard law
    :math:`Z \sim S(\alpha, \beta, 1, 0)` as ``gamma * stable alpha beta + delta``,
    which is :math:`S(\alpha, \beta, \gamma, \delta; 0)` in S0 and, in S1,
    :math:`S(\alpha, \beta, \gamma, \delta; 1)` for :math:`\alpha \ne 1` and
    :math:`S(1, \beta, \gamma, \delta - (2/\pi)\beta\gamma\log\gamma; 1)` at
    :math:`\alpha = 1`, as scipy's ``levy_stable(loc, scale)`` defines it.

    Wraps :class:`aggregate.stable.Stable`, which holds the S1 parameters and
    the closed forms. The characteristic function is the S1 form on the real
    line; the analytic continuation a tilt needs is
    ``[Characteristic-Function-Tilting]``. The light side of a totally skewed
    law has the Laplace transform
    :math:`\Lambda(s) = \log\mathsf{P}e^{-sX} = -\gamma^\alpha\sec(\pi\alpha/2)s^\alpha - \delta_1 s`
    for :math:`\alpha \ne 1` and :math:`(2\gamma/\pi)s\log s - \delta_1 s` at
    :math:`\alpha = 1`, :math:`s \ge 0` [Nolan 2020, Prop. 3.9], which is
    :math:`K(-s)` for :math:`\beta = 1` and, by reflection, :math:`K(s)` for
    :math:`\beta = -1`. The tails are :math:`A x^{-\alpha}` and
    :math:`B x^{-\alpha}` from :attr:`Stable.tail_constant` and
    :attr:`Stable.left_tail_constant`; at :math:`\alpha = 2` the law is normal
    with variance :math:`2\gamma^2` and has no power tail.
    """

    name = 'stable'

    def __init__(self, parameters, form=None):
        super().__init__(parameters, form)
        p = self.parameters
        self.stable = Stable(p['alpha'], p['beta'], p.get('scale', 1.0),
                             p.get('loc', 0.0), p.get('parameterization', 'S1'))
        st = self.stable
        a, b = st.alpha, st.beta
        self.location = st.loc
        if a == 2:
            self.power_tails = None
            self.tilt_side = None
            self.cgf_domain = (-np.inf, np.inf)
        else:
            self.power_tails = (st.tail_constant, st.left_tail_constant, a)
            if b == 1:
                self.tilt_side = 'right'
                self.cgf_domain = (-np.inf, 0.0)
            elif b == -1:
                self.tilt_side = 'left'
                self.cgf_domain = (0.0, np.inf)
            else:
                self.tilt_side = 'split'
                self.cgf_domain = (0.0, 0.0)
        if a < 1 and b == 1:
            self.support = (st.loc, np.inf)
        elif a < 1 and b == -1:
            self.support = (-np.inf, st.loc)
        else:
            self.support = (-np.inf, np.inf)

    @classmethod
    def from_stable(cls, stable):
        """Wrap an existing :class:`~aggregate.stable.Stable`."""
        return cls({'alpha': stable.alpha, 'beta': stable.beta,
                    'scale': stable.scale, 'loc': stable.loc})

    def chf(self, s):
        r"""The characteristic function, continued onto the closed upper half plane for ``beta = 1``.

        For :math:`\beta = 1`, S1,
        :math:`\log\varphi(s) = -\gamma^\alpha\sec(\pi\alpha/2)(-is)^\alpha + i\delta_1 s`
        for :math:`\alpha \ne 1` and
        :math:`(2\gamma/\pi)(-is)\log(-is) + i\delta_1 s` at :math:`\alpha = 1`,
        principal branches, which reduce to the S1 form on the real line and
        to :math:`\Lambda(\theta)` at :math:`s = i\theta`. Off the real line
        with :math:`\beta < 1` there is no continuation: :math:`\beta = -1` is
        computed by reflection and :math:`|\beta| < 1` by the split, so a
        complex argument there is a programming error.
        """
        s = np.asarray(s, dtype=complex)
        st = self.stable
        if not np.any(s.imag):
            return st.chf(s.real)
        if st.beta != 1 or st.alpha == 2:
            raise ValueError(
                'the stable characteristic function is continued off the real '
                'line only for beta = 1; beta = -1 is computed by reflection '
                'and |beta| < 1 by the split')
        a, g, d = st.alpha, st.scale, st.loc
        w = -1j * s
        if a == 1:
            # w log w is 0 at w = 0
            body = np.zeros_like(w)
            nz = w != 0
            body[nz] = (2 * g / np.pi) * w[nz] * np.log(w[nz])
        else:
            body = -g ** a / np.cos(np.pi * a / 2) * w ** a
        return np.exp(body + 1j * d * s)

    @property
    def density_bound(self):
        r"""``sup f = Gamma(1 + 1/alpha) / (pi gamma)``, from ``(1/2pi) int |phi|``."""
        st = self.stable
        return gamma_function(1 + 1 / st.alpha) / (np.pi * st.scale)

    def tail_mass_above(self, x):
        r"""``Pr(X > x)`` for ``beta = 1``: the exact series for ``alpha < 1``, the leading term otherwise.

        Parameters
        ----------
        x : float
            A point above the location, well into the tail.

        Returns
        -------
        float

        Notes
        -----
        For :math:`\alpha \ne 1` the series
        :math:`\Pr(X > x) = -(1/\pi)\sum_{k\ge1}(-1)^k\Gamma(k\alpha)\sin(k\pi\alpha)\,s^k x^{-k\alpha}/k!`,
        :math:`s = \gamma^\alpha\sec(\pi\alpha/2)`, :math:`x` measured from
        :math:`\delta_1`, whose first term is :math:`A x^{-\alpha}`. Below
        :math:`\alpha = 1` it converges [Nolan 2020, Thm. 3.10] and is summed
        to roundoff; above, :math:`s < 0` and the series is asymptotic, summed
        to its smallest term, which takes the leading term's relative error of
        :math:`1.6\times10^{-3}` at :math:`\alpha = 1.1`, :math:`x = 2000`
        to :math:`2\times10^{-5}` (measured against a log2 20 tilted grid);
        at :math:`\alpha = 1.5` the second term vanishes. The leading term
        alone is off by :math:`3\times10^{-5}` at :math:`\alpha = 0.7` on a
        log2 16 window.

        At :math:`\alpha = 1` the expansion carries a logarithm. With
        :math:`v = (x - \delta_1 - (2\gamma/\pi)\log\gamma)/\gamma` (S1 puts
        the standard law :math:`(2\gamma/\pi)\log\gamma` above the location)
        and :math:`a = 2/\pi`,
        :math:`\Pr(X > x) = a/v + a^2(\log v + \gamma_E - 1)/v^2 + O(\log^2 v/v^3)`:
        expanding the Laplace transform :math:`\exp(a\,v\log v)` to second
        order and matching the singular terms of
        :math:`-s\int e^{-sx}\Pr(X>x)\,dx` (a tail :math:`x^{-2}\log x`
        transforms to :math:`-\tfrac12 s\log^2 s - (\gamma_E - 1)s\log s`)
        fixes both coefficients; the constant was checked against scipy's
        ``levy_stable.sf``, which gives :math:`-0.35` to :math:`-0.38` for
        :math:`x` between 30 and 240 scales, approaching
        :math:`\gamma_E - 1 = -0.4228` as the third term fades.
        """
        st = self.stable
        a, g = st.alpha, st.scale
        u = float(x) - st.loc
        if u <= 0:
            return 1.0
        if a == 2:
            return float(ss.norm(scale=np.sqrt(2) * g).sf(u))
        if a == 1:
            # the leading term and the logarithmic second term in units of the
            # scale, see Notes; S1 at alpha = 1 puts the standard law at
            # (2 gamma / pi) log gamma above the location
            v = (u - (2 * g / np.pi) * math.log(g)) / g
            if v <= 0:
                return 1.0
            return (2 / np.pi) / v + (2 / np.pi) ** 2 * (math.log(v) + np.euler_gamma - 1) / v ** 2
        # term_k = -(1/pi) (-1)^k sin(k pi alpha) Gamma(k alpha) / k! * s^k u^(-k alpha)
        # with s = gamma^alpha sec(pi alpha / 2), signed: positive below
        # alpha = 1, negative above. Logs keep Gamma(k alpha) / k! in range.
        s_signed = g ** a / math.cos(math.pi * a / 2)
        log_s = math.log(abs(s_signed))
        sign_s = 1.0 if s_signed > 0 else -1.0
        total = 0.0
        last = np.inf
        for k in range(1, 400):
            log_mag = math.lgamma(k * a) - math.lgamma(k + 1) + k * (log_s - a * math.log(u))
            if log_mag > 700:
                raise ValueError(
                    f'tail_mass_above: x = {x:g} is too close to the location '
                    f'{st.loc:g} for the series (scale {g:g})')
            mag = math.exp(log_mag)
            if a > 1 and mag > last:
                # asymptotic above alpha = 1: stop at the smallest term
                break
            last = mag
            term = -((-1) ** k) * math.sin(k * math.pi * a) * mag * sign_s ** k / math.pi
            total += term
            # the magnitude, not the term: sin(k pi alpha) vanishes at even k
            # when alpha = 1/2, which is not the end of the series
            if mag < 1e-18 * max(abs(total), 1e-300):
                break
        return float(total)

    def tail_mean_above(self, x):
        r"""``P[(X - delta_1) 1{X > x}]`` for ``beta = 1`` and ``alpha > 1``, the tail mean beyond ``x`` from the location.

        Parameters
        ----------
        x : float
            A point above the location, well into the tail.

        Returns
        -------
        float

        Notes
        -----
        With :math:`u = x - \delta_1` and the series of
        :meth:`tail_mass_above`, :math:`\Pr(X > \delta_1 + v) = \sum_k t_k v^{-k\alpha}`,
        :math:`\mathsf{P}[(X - \delta_1)1_{X > x}] = uS(x) + \int_u^\infty S = \sum_k t_k u^{1-k\alpha}\,k\alpha/(k\alpha - 1)`,
        summed to its smallest term as the tail mass is. The first term is
        :math:`A\alpha u^{1-\alpha}/(\alpha - 1)`; at :math:`\alpha = 1.1` the
        rest moves the completed mean of ``stable 1.1 1`` at log2 16 from
        3.6e-4 to under 1e-5 of the scale.
        """
        st = self.stable
        a, g = st.alpha, st.scale
        u = float(x) - st.loc
        if not (a > 1 and a < 2 and st.beta == 1 and u > 0):
            raise ValueError('tail_mean_above: needs beta = 1, 1 < alpha < 2 and x above the location')
        s_signed = g ** a / math.cos(math.pi * a / 2)
        log_s = math.log(abs(s_signed))
        sign_s = 1.0 if s_signed > 0 else -1.0
        total = 0.0
        last = np.inf
        for k in range(1, 400):
            log_mag = math.lgamma(k * a) - math.lgamma(k + 1) + k * (log_s - a * math.log(u))
            mag = math.exp(log_mag)
            if mag > last:
                break
            last = mag
            term = -((-1) ** k) * math.sin(k * math.pi * a) * mag * sign_s ** k / math.pi
            total += term * u * k * a / (k * a - 1)
            if mag < 1e-18 * max(abs(total), 1e-300):
                break
        return float(total)

    def halves(self):
        r"""The two totally skewed laws whose sum is this one: ``(plus, minus)``.

        With :math:`w_\pm = (1 \pm \beta)/2` and :math:`\gamma_\pm = \gamma w_\pm^{1/\alpha}`,
        :math:`X = X_+ + X_-` with :math:`X_+ \sim S(\alpha, 1, \gamma_+, \delta_1; 1)`
        and :math:`X_- \sim S(\alpha, -1, \gamma_-, 0; 1)`, independent, for
        every :math:`\alpha` including 1: :math:`\log\varphi` is linear in
        :math:`\beta` with coefficient proportional to :math:`\gamma^\alpha`,
        and the S1 form at :math:`\alpha = 1` has no :math:`\gamma\log\gamma`
        term. The split is exact on the real line to roundoff.

        Returns
        -------
        tuple of FamilyStable
        """
        st = self.stable
        if abs(st.beta) == 1 or st.alpha == 2:
            raise ValueError('halves: the law is already one-sided')
        a, g = st.alpha, st.scale
        g_plus = g * ((1 + st.beta) / 2) ** (1 / a)
        g_minus = g * ((1 - st.beta) / 2) ** (1 / a)
        plus = FamilyStable(dict(alpha=a, beta=1, scale=g_plus, loc=st.loc))
        minus = FamilyStable(dict(alpha=a, beta=-1, scale=g_minus, loc=0.0))
        return plus, minus

    def reflected(self):
        """The family of ``-X``: ``beta`` negated, location negated, as written."""
        p = dict(self.parameters)
        st = self.stable
        p.update(alpha=st.alpha, beta=-st.beta, scale=st.scale, loc=-st.loc)
        p.pop('parameterization', None)
        return FamilyStable(p)

    def _laplace(self, s):
        r"""``log P exp(-sY)`` for ``Y ~ S(alpha, 1, gamma, loc; 1)`` on ``s >= 0``, as a function of ``(s, loc)``."""
        st = self.stable
        a, g = st.alpha, st.scale
        s = np.asarray(s, dtype=float)
        if a == 1:
            # s log s is 0 at s = 0; the logarithm is taken only where s > 0
            body = np.zeros_like(s)
            pos = s > 0
            body[pos] = (2 * g / np.pi) * s[pos] * np.log(s[pos])
        else:
            body = -g ** a / np.cos(np.pi * a / 2) * s ** a
        return body

    def cgf(self, s):
        s = np.asarray(s, dtype=float)
        st = self.stable
        out = np.full(s.shape, np.nan)
        if st.alpha == 2:
            return st.loc * s + st.scale ** 2 * s ** 2
        if st.beta == 1:
            ok = s <= 0
            out[ok] = self._laplace(-s[ok]) + st.loc * s[ok]
        elif st.beta == -1:
            ok = s >= 0
            out[ok] = self._laplace(s[ok]) + st.loc * s[ok]
        else:
            out[s == 0] = 0.0
        return out

    @property
    def scale(self):
        st = self.stable
        return float(np.sqrt(2) * st.scale) if st.alpha == 2 else float(st.scale)

    @property
    def moments(self):
        st = self.stable
        if st.alpha == 2:
            return (st.loc, 2 * st.scale ** 2, 0.0)
        return (st.mean, np.inf, np.nan)

    @property
    def reference(self):
        return self.stable.fz

    def location_under(self, c, b):
        r"""The S1 location :math:`\delta_1` of :math:`cX + b`.

        S0 is equivariant for every :math:`\alpha`, so the S0 location of
        :math:`cX + b` is :math:`c\delta_0 + b`, converted back to S1 at the
        scale :math:`c\gamma`. For :math:`\alpha \ne 1` that is
        :math:`c\delta_1 + b`; at :math:`\alpha = 1` it is
        :math:`c\delta_1 + b - (2/\pi)\beta c\gamma\log c`.
        """
        st = self.stable
        return Stable.s0_to_s1(st.alpha, st.beta, c * st.scale, c * st.loc0 + b)

    def _affine_parts(self):
        r"""``(gamma, shift, 'stable alpha beta [s0]')``: :math:`X = \gamma Z + \text{shift}`.

        The shift is the S0 location :math:`\delta_0`, which equals the S1
        location away from :math:`\alpha = 1`: the standard laws of the two
        forms coincide at :math:`\alpha = 1` and differ by
        :math:`\beta\tan(\pi\alpha/2)` elsewhere, where S1 is equivariant.
        """
        st = self.stable
        s0 = str(self.parameters.get('parameterization', 'S1')).upper() == 'S0'
        body = f'stable {_decl_number(st.alpha)} {_decl_number(self.parameters["beta"])}'
        if s0:
            return st.scale, st.loc0, body + ' s0'
        return st.scale, (st.loc0 if st.alpha == 1 else st.loc), body


class FamilyUser(Family):
    """A user-supplied characteristic function with no closed forms.

    What :meth:`AggregateCHF.from_chf` wraps. It has no cumulant generating
    function and no tail constants, so the window cannot be sized from the
    law and the grid is given explicitly; it is never tilted.
    """

    name = 'chf'
    cgf_domain = (0.0, 0.0)

    def __init__(self, chf, label='chf'):
        super().__init__({})
        self._chf = chf
        self.label = label

    def chf(self, s):
        s = np.asarray(s, dtype=complex)
        # a user's function is written for the real line; hand it real
        # arguments there, so a scipy-style chf never sees a complex dtype
        arg = s.real if not np.any(s.imag) else s
        return np.asarray(self._chf(arg), dtype=complex)

    def cgf(self, s):
        s = np.asarray(s, dtype=float)
        out = np.full(s.shape, np.nan)
        out[s == 0] = 0.0
        return out

    def to_decl(self):
        return ''

    def __repr__(self):
        return f'FamilyUser({self.label!r})'


class FamilyTweedie(Family):
    r"""The Tweedie law for every admissible power, ``tweedie p mean dispersion`` or ``tweedie p mean cv c``.

    Wraps :class:`aggregate.tweedie.Tweedie` in its reproductive form
    :math:`(p, \mu, \sigma^2)`, which supplies the cumulant function
    :math:`K(s) = \lambda[\kappa(\theta + s/\lambda) - \kappa(\theta)]`,
    :math:`\lambda = 1/\sigma^2`, with
    :math:`\kappa(\theta) = \frac{\alpha-1}{\alpha}\big(\frac{\theta}{\alpha-1}\big)^\alpha`,
    :math:`\alpha = (p-2)/(p-1)`, and :math:`\varphi(t) = \exp K(it)`. The
    mean is :math:`\mu`, the variance :math:`\sigma^2\mu^p` and the skewness
    :math:`p\cdot\mathrm{cv}`, since the third cumulant is
    :math:`\sigma^4 p\,\mu^{2p-1}`.

    Every member with a finite mean is a natural exponential family with
    :math:`\theta` interior to its domain, so :math:`K` is finite near 0 and
    the law is light on both sides: the members with :math:`p > 2` are the
    positive extreme stable laws already tilted by their own :math:`\theta`,
    and those with :math:`p < 0` the extreme stable laws on the whole line.
    For :math:`1 < p < 2` the law is compound Poisson with an atom
    :math:`e^{-\lambda_f}` at 0, split out of :math:`\varphi` before
    inversion and added back to the bucket at 0 so that frequency truncation
    does not ring there. :math:`p = 1` is a lattice law with the dispersion
    as its step, refused with a pointer at ``agg ... poisson dsev [phi]``, as
    is :math:`0 < p < 1`, where no law exists.
    """

    name = 'tweedie'

    def __init__(self, parameters, form=None):
        from .tweedie import Tweedie
        super().__init__(parameters, form)
        p = self.parameters
        power, mean = float(p['p']), float(p['mean'])
        if power == 1:
            raise ValueError(
                'chfagg: tweedie 1 is a lattice law with the dispersion as its '
                'step; declare it as agg NAME <mean/phi> claims poisson dsev [phi]')
        if 0 < power < 1:
            raise ValueError(f'chfagg: no Tweedie law has 0 < p < 1, got p = {power}')
        if np.isinf(power):
            raise ValueError(
                'chfagg: tweedie inf is the stable law with index 1 and no tilt '
                '(its theta is 0 for every mean), which has a power tail; declare '
                'it as chfagg NAME <scale> * stable 1 1 + <shift>')
        if mean <= 0:
            raise ValueError(f'chfagg: the Tweedie mean must be positive, got {mean}')
        if form == 'cv':
            cv = float(p['cv'])
            if cv <= 0:
                raise ValueError(f'chfagg: the Tweedie cv must be positive, got {cv}')
            dispersion = cv ** 2 * mean ** (2 - power)
        else:
            dispersion = float(p['dispersion'])
            if dispersion <= 0:
                raise ValueError(f'chfagg: the Tweedie dispersion must be positive, got {dispersion}')
        self.tweedie = Tweedie(power, mean, dispersion)
        self.power = power
        self.mean = mean
        self.dispersion = dispersion
        lam = 1 / dispersion
        theta = self.tweedie.theta
        # the domain of K: theta + s / lambda stays on kappa's side of 0
        if power == 0:
            self.cgf_domain = (-np.inf, np.inf)
        elif power < 0:
            self.cgf_domain = (-lam * theta, np.inf)
        elif np.isinf(power):
            self.cgf_domain = (-np.inf, -lam * theta)
        else:
            self.cgf_domain = (-np.inf, -lam * theta)
        self.support = (-np.inf, np.inf) if power <= 0 else (0.0, np.inf)
        if 1 < power < 2:
            frequency = self.tweedie.po_gamma_shape_rate[0]
            self.atom = (0.0, float(np.exp(-frequency)))

    def chf(self, s):
        s = np.asarray(s, dtype=complex)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.asarray(self.tweedie.chf(s), dtype=complex)

    def cgf(self, s):
        s = np.asarray(s, dtype=float)
        lo, hi = self.cgf_domain
        out = np.full(s.shape, np.nan)
        ok = (s > lo) & (s < hi)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[ok] = np.real(self.tweedie.K(s[ok]))
        return out

    @property
    def moments(self):
        var = self.dispersion * self.mean ** self.power
        cv = np.sqrt(var) / self.mean
        return (self.mean, var, self.power * cv)

    @property
    def reference(self):
        """scipy for the normal, gamma and inverse Gaussian members; the compound engine's ``Aggregate`` for ``1 < p < 2``; else ``None``."""
        if self.power in (0, 2, 3) or 1 < self.power < 2:
            return self.tweedie.fz
        return None

    def _affine_parts(self):
        p = self.parameters
        if self.form == 'cv':
            body = f"tweedie {_decl_number(p['p'])} {_decl_number(p['mean'])} cv {_decl_number(p['cv'])}"
        else:
            body = (f"tweedie {_decl_number(p['p'])} {_decl_number(p['mean'])} "
                    f"{_decl_number(p['dispersion'])}")
        return 1.0, 0.0, body


class FamilyNormalInverseGaussian(Family):
    r"""The normal inverse Gaussian law, ``norminvgauss a b``, scipy's shapes.

    DecL's ``delta * norminvgauss a b + mu`` is scipy's
    ``norminvgauss(a, b, loc=mu, scale=delta)``, the law below with
    :math:`\alpha = a/\delta` and :math:`\beta = b/\delta`; from Python the
    keywords ``alpha``, ``beta``, ``delta`` and ``mu`` give it directly. With
    :math:`|\beta| < \alpha`, :math:`\delta > 0` and
    :math:`\gamma = \sqrt{\alpha^2 - \beta^2}`,
    :math:`K(s) = \mu s + \delta\big(\gamma - \sqrt{\alpha^2 - (\beta+s)^2}\big)`
    for :math:`-\alpha-\beta < s < \alpha-\beta` and
    :math:`\varphi(s) = \exp K(is)` on the principal square root
    [Barndorff-Nielsen 1997]. Mean :math:`\mu + \delta\beta/\gamma`, variance
    :math:`\delta\alpha^2/\gamma^3`, skewness :math:`3\beta/(\alpha\sqrt{\delta\gamma})`.
    The scipy law is ``norminvgauss(a=alpha*delta, b=beta*delta, loc=mu, scale=delta)``.
    """

    name = 'norminvgauss'

    def __init__(self, parameters, form=None):
        super().__init__(parameters, form)
        p = self.parameters
        if 'a' in p:
            self.alpha, self.beta, self.delta, self.mu = float(p['a']), float(p['b']), 1.0, 0.0
        else:
            self.alpha, self.beta = float(p['alpha']), float(p['beta'])
            self.delta, self.mu = float(p['delta']), float(p['mu'])
        if not abs(self.beta) < self.alpha:
            raise ValueError(f'norminvgauss: |beta| < alpha is needed, got alpha {self.alpha}, beta {self.beta}')
        if self.delta <= 0:
            raise ValueError(f'norminvgauss: delta must be positive, got {self.delta}')
        self.gamma = math.sqrt(self.alpha ** 2 - self.beta ** 2)
        self.cgf_domain = (-self.alpha - self.beta, self.alpha - self.beta)

    def chf(self, s):
        s = np.asarray(s, dtype=complex)
        a, b, d, m = self.alpha, self.beta, self.delta, self.mu
        return np.exp(1j * m * s + d * (self.gamma - np.sqrt(a ** 2 - (b + 1j * s) ** 2)))

    def cgf(self, s):
        s = np.asarray(s, dtype=float)
        lo, hi = self.cgf_domain
        out = np.full(s.shape, np.nan)
        ok = (s > lo) & (s < hi)
        out[ok] = self.mu * s[ok] + self.delta * (
            self.gamma - np.sqrt(self.alpha ** 2 - (self.beta + s[ok]) ** 2))
        return out

    @property
    def moments(self):
        a, b, d, g = self.alpha, self.beta, self.delta, self.gamma
        return (self.mu + d * b / g, d * a ** 2 / g ** 3, 3 * b / (a * math.sqrt(d * g)))

    @property
    def reference(self):
        return ss.norminvgauss(a=self.alpha * self.delta, b=self.beta * self.delta,
                               loc=self.mu, scale=self.delta)

    def _affine_parts(self):
        d = self.delta
        return d, self.mu, (f'norminvgauss {_decl_number(self.alpha * d)} '
                            f'{_decl_number(self.beta * d)}')


class FamilyVarianceGamma(Family):
    r"""The variance gamma law, ``variancegamma sigma nu theta``; the location ``mu`` is ``+ mu``.

    Madan, Carr and Chang's :math:`(\sigma, \nu, \theta)` with a location
    :math:`\mu` (DecL's shift, the keyword ``mu`` from Python, default 0):
    :math:`K(s) = \mu s - \nu^{-1}\log(1 - \theta\nu s - \sigma^2\nu s^2/2)`
    between the roots of the quadratic, and
    :math:`\varphi(s) = e^{i\mu s}(1 - i\theta\nu s + \sigma^2\nu s^2/2)^{-1/\nu}`.
    Mean :math:`\mu + \theta`, variance :math:`\sigma^2 + \nu\theta^2`, third
    cumulant :math:`2\theta^3\nu^2 + 3\sigma^2\theta\nu`. At :math:`\theta = 0`,
    :math:`\nu = 1` it is the Laplace law with scale :math:`\sigma/\sqrt2`,
    which is its scipy reference. :math:`|\varphi|` decays only as
    :math:`t^{-2/\nu}`, so a large :math:`\nu` is flagged by the truncation row.
    """

    name = 'variancegamma'

    def __init__(self, parameters, form=None):
        super().__init__(parameters, form)
        p = self.parameters
        self.sigma, self.nu = float(p['sigma']), float(p['nu'])
        self.theta, self.mu = float(p['theta']), float(p.get('mu', 0.0))
        if self.sigma <= 0 or self.nu <= 0:
            raise ValueError(f'variancegamma: sigma and nu must be positive, got {self.sigma}, {self.nu}')
        disc = math.sqrt(self.theta ** 2 * self.nu ** 2 + 2 * self.sigma ** 2 * self.nu)
        denom = self.sigma ** 2 * self.nu
        self.cgf_domain = ((-self.theta * self.nu - disc) / denom, (-self.theta * self.nu + disc) / denom)

    def chf(self, s):
        s = np.asarray(s, dtype=complex)
        return (np.exp(1j * self.mu * s)
                * (1 - 1j * self.theta * self.nu * s + 0.5 * self.sigma ** 2 * self.nu * s ** 2) ** (-1 / self.nu))

    def cgf(self, s):
        s = np.asarray(s, dtype=float)
        lo, hi = self.cgf_domain
        out = np.full(s.shape, np.nan)
        ok = (s > lo) & (s < hi)
        q = 1 - self.theta * self.nu * s[ok] - 0.5 * self.sigma ** 2 * self.nu * s[ok] ** 2
        out[ok] = self.mu * s[ok] - np.log(q) / self.nu
        return out

    @property
    def moments(self):
        var = self.sigma ** 2 + self.nu * self.theta ** 2
        k3 = 2 * self.theta ** 3 * self.nu ** 2 + 3 * self.sigma ** 2 * self.theta * self.nu
        return (self.mu + self.theta, var, k3 / var ** 1.5)

    @property
    def reference(self):
        if self.theta == 0 and self.nu == 1:
            return ss.laplace(loc=self.mu, scale=self.sigma / math.sqrt(2))
        return None

    def _affine_parts(self):
        return 1.0, self.mu, ' '.join(['variancegamma', *(_decl_number(x) for x in
                                       (self.sigma, self.nu, self.theta))])


class FamilyTemperedStable(Family):
    r"""The CGMY tempered stable law, ``temperedstable C G M Y``; the location ``mu`` is ``+ mu``.

    Carr, Geman, Madan and Yor's law with :math:`Y < 2`, :math:`Y \notin \{0, 1\}`:
    :math:`K(s) = \mu s + C\Gamma(-Y)\big[(M-s)^Y - M^Y + (G+s)^Y - G^Y\big]`
    for :math:`-G < s < M`, :math:`\varphi(s) = \exp K(is)` on the principal
    powers, and cumulants :math:`\kappa_n = C\Gamma(n-Y)\big(M^{Y-n} + (-1)^n G^{Y-n}\big)`,
    which give the mean :math:`\mu + \kappa_1`, variance and skewness.
    """

    name = 'temperedstable'

    def __init__(self, parameters, form=None):
        super().__init__(parameters, form)
        p = self.parameters
        self.C, self.G, self.M, self.Y = (float(p['C']), float(p['G']), float(p['M']), float(p['Y']))
        self.mu = float(p.get('mu', 0.0))
        if self.C <= 0 or self.G <= 0 or self.M <= 0:
            raise ValueError(f'temperedstable: C, G and M must be positive, got {self.C}, {self.G}, {self.M}')
        if not self.Y < 2 or self.Y in (0.0, 1.0):
            raise ValueError(f'temperedstable: Y must be below 2 and not 0 or 1, got {self.Y}')
        self.cgf_domain = (-self.G, self.M)
        self.gamma_minus_y = gamma_function(-self.Y)

    def _k(self, s):
        C, G, M, Y = self.C, self.G, self.M, self.Y
        return self.mu * s + C * self.gamma_minus_y * ((M - s) ** Y - M ** Y + (G + s) ** Y - G ** Y)

    def chf(self, s):
        s = np.asarray(s, dtype=complex)
        return np.exp(self._k(1j * s))

    def cgf(self, s):
        s = np.asarray(s, dtype=float)
        out = np.full(s.shape, np.nan)
        ok = (s > -self.G) & (s < self.M)
        out[ok] = self._k(s[ok])
        return out

    def cumulant(self, n):
        """The ``n``-th cumulant, :math:`C\\Gamma(n-Y)(M^{Y-n} + (-1)^n G^{Y-n})`."""
        C, G, M, Y = self.C, self.G, self.M, self.Y
        return C * gamma_function(n - Y) * (M ** (Y - n) + (-1) ** n * G ** (Y - n))

    @property
    def moments(self):
        k1, k2, k3 = (self.cumulant(n) for n in (1, 2, 3))
        return (self.mu + k1, k2, k3 / k2 ** 1.5)

    def _affine_parts(self):
        return 1.0, self.mu, ' '.join(['temperedstable', *(_decl_number(x) for x in
                                       (self.C, self.G, self.M, self.Y))])


class FamilyAffine(Family):
    r"""The law of :math:`Y = cX + b`, :math:`c > 0`, for a family's law :math:`X`: DecL's ``c * FAMILY shapes + b``.

    Implements the family protocol from the base's closed forms, so the
    engine (window, tilt, split, inversion) runs on it unchanged.

    Parameters
    ----------
    base : Family
        The law of :math:`X`.
    scale : float
        :math:`c > 0`.
    shift : float
        :math:`b`.

    Notes
    -----
    :math:`\varphi_Y(s) = e^{ibs}\varphi_X(cs)`, also on the tilted contour,
    and :math:`K_Y(s) = K_X(cs) + bs` on :math:`(s_-/c, s_+/c)`. A power tail
    :math:`\Pr(X > x) \sim A x^{-\alpha}` becomes
    :math:`\Pr(Y > y) \sim Ac^\alpha y^{-\alpha}`, so the tail constants scale
    by :math:`c^\alpha`; the support, the atom and the mean map through
    :math:`x \mapsto cx + b`, the variance scales by :math:`c^2` and the
    skewness is unchanged. The location the power edges are measured from is
    the base's :meth:`Family.location_under`, which for a stable law at
    :math:`\alpha = 1` in S1 carries the :math:`\beta\gamma\log c` term. The
    scipy reference is the base's frozen law with ``loc`` and ``scale``
    mapped, which is right for every law scipy holds as a location and scale
    family, :math:`\alpha = 1` included.
    """

    def __init__(self, base, scale=1.0, shift=0.0):
        c, b = float(scale), float(shift)
        if not (np.isfinite(c) and c > 0):
            raise ValueError(
                f'chfagg: the scale c of c * {base.name} must be positive and finite, '
                f'got {c:g}; a reflection -X is not supported (for a stable law '
                'negate beta)')
        if not np.isfinite(b):
            raise ValueError(f'chfagg: the shift must be finite, got {b:g}')
        super().__init__(base.parameters, base.form)
        self.base = base
        self.c = c
        self.b = b
        self.name = base.name
        s_minus, s_plus = base.cgf_domain
        self.cgf_domain = (s_minus / c, s_plus / c)
        if base.power_tails:
            A, B, alpha = base.power_tails
            self.power_tails = (A * c ** alpha, B * c ** alpha, alpha)
        self.tilt_side = base.tilt_side
        self.location = base.location_under(c, b)
        lo, hi = base.support
        self.support = (c * lo + b, c * hi + b)
        self.atom = None if base.atom is None else (c * base.atom[0] + b, base.atom[1])

    def chf(self, s):
        s = np.asarray(s, dtype=complex)
        return np.exp(1j * self.b * s) * self.base.chf(self.c * s)

    def cgf(self, s):
        s = np.asarray(s, dtype=float)
        return self.base.cgf(self.c * s) + self.b * s

    @property
    def scale(self):
        return self.c * self.base.scale

    @property
    def moments(self):
        mean, var, skew = self.base.moments
        return (self.c * mean + self.b, self.c ** 2 * var, skew)

    @property
    def reference(self):
        """The scipy law of ``c X + b``, or ``None`` where the base has none or it is not a scipy frozen law.

        A stable law is rebuilt through :class:`~aggregate.stable.Stable` at
        the mapped scale and S1 location, since scipy's S1 ``loc`` is Nolan's
        :math:`\\delta_1`, which is not equivariant at :math:`\\alpha = 1`
        (measured: ``2 * stable 1 0.5 + 3`` is scipy's S1 law with ``loc``
        2.5587, not 3). Every other law maps scipy's ``loc`` and ``scale``.
        """
        st = getattr(self.base, 'stable', None)
        if st is not None:
            return Stable(st.alpha, st.beta, self.c * st.scale, self.location).fz
        ref = self.base.reference
        if ref is None or not hasattr(ref, 'dist') or not hasattr(ref, 'kwds'):
            return None
        kw = dict(ref.kwds)
        kw['loc'] = self.c * kw.get('loc', 0.0) + self.b
        kw['scale'] = self.c * kw.get('scale', 1.0)
        return ref.dist(*ref.args, **kw)

    @property
    def density_bound(self):
        """The base's bound over ``c``."""
        return self.base.density_bound / self.c

    def tail_mass_above(self, x):
        """``Pr(Y > x) = Pr(X > (x - b) / c)``."""
        return self.base.tail_mass_above((float(x) - self.b) / self.c)

    def tail_mean_above(self, x):
        """``P[(Y - delta_1) 1{Y > x}] = c P[(X - delta_1) 1{X > (x - b) / c}]``, for ``alpha > 1`` where the location is equivariant."""
        return self.c * self.base.tail_mean_above((float(x) - self.b) / self.c)

    def location_under(self, c, b):
        """The location of ``c Y + b``, composed through the base."""
        return self.base.location_under(c * self.c, c * self.b + b)

    def halves(self):
        """The base's halves under the same scale, the shift on the plus half."""
        plus, minus = self.base.halves()
        return FamilyAffine(plus, self.c, self.b), FamilyAffine(minus, self.c, 0.0)

    def reflected(self):
        """The family of ``-Y = c (-X) - b``."""
        return FamilyAffine(self.base.reflected(), self.c, -self.b)

    def _affine_parts(self):
        c0, b0, body = self.base._affine_parts()
        return self.c * c0, self.c * b0 + self.b, body


def arithmetic_spelling(family, numbers, parameterization=None):
    """The ``chfagg`` body, in arithmetic form, of a body written with scale and location numbers before 1.35.0.

    Parameters
    ----------
    family : str
        The family word.
    numbers : sequence of float
        The positional numbers as written, in the order of
        :data:`LEGACY_PARAMETERS`.
    parameterization : str, optional
        ``'s0'`` or ``'s1'`` for a stable law.

    Returns
    -------
    str or None
        The body that declares the same law, ``c * FAMILY shapes + b``, or
        ``None`` when the numbers do not make a law under the old reading
        either.
    """
    names = LEGACY_PARAMETERS.get(family)
    required = {'stable': 2, 'temperedstable': 4}.get(family, len(names or ()))
    if names is None or not required <= len(numbers) <= len(names):
        return None
    params = dict(zip(names, (float(x) for x in numbers)))
    if parameterization is not None:
        params['parameterization'] = str(parameterization).upper()
    try:
        return make_family(family, None, **params).to_decl()
    except (KeyError, ValueError, TypeError):
        return None


#: Family word to class.
FAMILIES = {
    'stable': FamilyStable,
    'tweedie': FamilyTweedie,
    'norm': FamilyNormal,
    'gamma': FamilyGamma,
    'invgauss': FamilyInverseGaussian,
    'norminvgauss': FamilyNormalInverseGaussian,
    'variancegamma': FamilyVarianceGamma,
    'temperedstable': FamilyTemperedStable,
}


def make_family(family, form=None, **parameters):
    """Build the family object for a DecL family word and its parameters.

    Parameters
    ----------
    family : str
        The family word, one of :data:`FAMILIES`.
    form : str or None
        ``'cv'`` for the ``mean cv`` form.
    **parameters
        The parameters as written, keyed by name.

    Returns
    -------
    Family
    """
    try:
        cls = FAMILIES[family]
    except KeyError:
        raise ValueError(f'chfagg: unknown family {family!r}; one of '
                         f'{", ".join(FAMILIES)}') from None
    if form == 'cv' and family not in FAMILY_CV_PARAMETERS:
        raise ValueError(f'chfagg: {family} has no mean cv form')
    return cls(parameters, form)


def truncation_at(family, h):
    r"""The frequency truncation of a bucket ``h``: :math:`|\varphi(\pi/h)|` of the part the inversion transforms.

    For a law with an atom :math:`(x_a, w)` that is
    :math:`|\varphi(\pi/h) - w e^{i\pi x_a/h}|`, since the inversion
    subtracts the atom before transforming and adds it back to its bucket
    (:func:`invert_characteristic_function`); the whole law's value tends to
    :math:`w`, which no bucket reduces. ``tweedie 1.5 100 2``: 4.54e-5 for the
    whole law, 1.13e-7 for the continuous part.

    Parameters
    ----------
    family : Family
    h : float

    Returns
    -------
    float
    """
    s = np.array([np.pi / h])
    c = complex(family.chf(s)[0])
    if family.atom is not None:
        xa, w = family.atom
        c -= w * np.exp(1j * np.pi * xa / h)
    return float(abs(c))


def window_edges(family, eps_side):
    r"""The window edges ``(x_L, x_R)`` with mass ``eps_side`` beyond each.

    A power side gives :math:`\delta_1 \pm (A/\varepsilon)^{1/\alpha}`; a light
    side gives the Chernoff edge, cut at the support bound.

    Parameters
    ----------
    family : Family
    eps_side : float
        The mass allowed beyond each edge.

    Returns
    -------
    tuple of float
    """
    lo, hi = family.support
    log_eps = math.log(eps_side)
    scale = family.scale
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    A, B, alpha = family.power_tails if family.power_tails else (0.0, 0.0, None)
    s_minus, s_plus = family.cgf_domain
    if A > 0:
        x_r = family.location + (A / eps_side) ** (1 / alpha)
    elif s_plus > 0:
        x_r = chernoff_edge(family.cgf, s_plus, log_eps, scale)
    else:
        x_r = np.inf
    if B > 0:
        x_l = family.location - (B / eps_side) ** (1 / alpha)
    elif s_minus < 0:
        x_l = -chernoff_edge(lambda u: family.cgf(-u), -s_minus, log_eps, scale)
    else:
        x_l = -np.inf
    x_r = min(x_r, hi)
    x_l = max(x_l, lo)
    # A finite support bound replaces a Chernoff edge that sits within one
    # percent of the window of it: the few buckets saved are not worth the
    # mass left outside, which wraps onto the far end of the window.
    if np.isfinite(lo) and x_l > lo and x_l - lo < SUPPORT_EDGE_SHARE * (x_r - lo):
        x_l = lo
    if np.isfinite(hi) and x_r < hi and hi - x_r < SUPPORT_EDGE_SHARE * (hi - x_l):
        x_r = hi
    return float(x_l), float(x_r)


# ======================================================================
# the class
# ======================================================================

class AggregateCHF(HelpMixin, LabeledMixin, ProgramMixin):
    r"""A distribution computed by inverting its characteristic function.

    The object behind the ``chfagg`` declaration. It holds a family (the
    closed forms of one law), sizes a grid from the law's tails, samples the
    characteristic function on the FFT frequency grid, inverts with one
    ``irfft`` to bucket masses, and serves the probability accessors from a
    :class:`~aggregate._grid_distribution.GridDistribution` exactly as
    :class:`~aggregate.Aggregate` does.

    Parameters
    ----------
    name : str
        The object's name, the DecL handle.
    family : str or Family
        The family word (:data:`FAMILIES`), or a family object.
    form : str or None
        ``'cv'`` for the ``mean cv`` form of a two-form family.
    note, tags, hints, label, label_map
        The DecL trailer, as on every first-class class.
    chf_scale, chf_loc : float, default 1 and 0
        DecL's arithmetic, ``chf_scale * FAMILY shapes + chf_loc``: the law
        declared is that of :math:`cX + b`, held as a :class:`FamilyAffine`,
        with :math:`c > 0`. These are the spec keys, as ``sev_scale`` and
        ``sev_loc`` hold the same arithmetic on an ``agg`` severity.
    **parameters
        The family's parameters, keyed by name: the shapes
        (:data:`FAMILY_PARAMETERS`), the ``mean cv`` form
        (:data:`FAMILY_CV_PARAMETERS`), or from Python the full parameter set
        each family class lists, scale and location included.

    Notes
    -----
    **Not a subclass of ``Aggregate``.** The name parallels ``AggregateNTT``
    and the ``chfagg`` keyword, and the shared surface (``q``, ``tvar``,
    ``cdf``, ``density_df``, ``summary_df``, ``tail_df``, ``bs``, ``log2``,
    ``est_m`` and the rest) is by duck typing, so that the pricing, chart and
    exhibit code written against ``Aggregate`` applies unchanged;
    ``isinstance(obj, Aggregate)`` is ``False`` by design. No ``Aggregate``
    code path needs a special case for it, and no consumer that reads
    ``frequency``, ``sev`` or the reinsurance attributes is ever handed one.
    It is a first-class peer like ``PnL`` and ``BivariateAggregate``: it
    meets ``FCC_REQUIRED`` and mixes in ``HelpMixin``, ``LabeledMixin`` and
    ``ProgramMixin``.

    **Why "agg" is honest.** Every family in scope is infinitely divisible: it
    is the time-one law of a Lévy process and so a limit of compound Poisson
    aggregates. The keyword describes what the laws are, not how they are
    computed.

    **Conventions.** Loss orientation. The masses are the round
    discretization's, so ``cdf(x_k)`` is the law's :math:`F(x_k + h/2)`,
    exactly as on ``Aggregate``; a caller wanting the distribution function at
    an arbitrary point to second order interpolates the cumulative masses at
    the bucket edges :math:`x_k + h/2`. The grid is never renormalized.

    **Window and bucket.** Rule A, the default (``bs=0``): ``bucket_sizing_p``
    sets the mass :math:`\varepsilon = 1 - p` allowed beyond the window, split
    equally between the tails; the edges come from the closed forms
    (:func:`window_edges`); the bucket is
    :func:`~aggregate.utilities.round_bucket` of the width over :math:`2^{\log_2}`
    and the left edge is the bucket multiple at or below the left edge, so 0
    is a grid point. Rule C (``bs`` and ``log2`` given, or
    ``hints{log2=...; bs=...}``) uses them as given with the left edge from
    the same rule.

    **Errors.** Frequency truncation drops :math:`|\varphi(\pi/h)|`, reported
    as :attr:`truncation_error`; a gamma shape below 1 rings near zero.
    Aliasing wraps the mass outside the window onto it. For a stable law with
    :math:`\alpha < 2` the wrapped mass lands where it does the most relative
    damage, near the right edge, so an untilted heavy-tailed grid has a
    relative error of order one there whatever its total.

    **Tilting.** A totally skewed stable law (:math:`|\beta| = 1`) is inverted
    along the contour :math:`\mathrm{Im}\,s = \theta`, where the integrand
    decays like the tilted law :math:`e^{-\theta x}f(x)`, exponentially on
    the heavy side; the discretized integral wraps that damped sequence and
    :math:`e^{\theta x}` recovers the masses
    (:func:`invert_characteristic_function`). Written as :math:`\tau = \theta W`
    e-folds across the window, the two errors at the right edge are the
    damped wrap :math:`\rho e^{-\tau}` and the amplified roundoff
    :math:`\varepsilon_{\mathrm{eff}}\kappa e^{\tau}`, least at
    :math:`\tau^* = \tfrac12\log(\rho/\varepsilon_{\mathrm{eff}}\kappa)` with
    value :math:`E^* = 2\sqrt{\rho\varepsilon_{\mathrm{eff}}\kappa}`, all in
    closed form (:meth:`_edge_terms`); tilting halves the number of digits
    the window's dynamic range costs. Because truncation is amplified too,
    the bucket is capped (:meth:`_bucket_cap`) and the window ends short,
    with the mass beyond reported in :attr:`outside_above` rather than
    wrapped; the left edge is the support bound or the
    :data:`TILT_LEFT_EPS` Chernoff edge. :math:`\beta = -1` is the reflection
    of :math:`\beta = 1`. The ``tilt`` keyword of :meth:`update` chooses.

    Examples
    --------
    >>> from aggregate.characteristic_function import AggregateCHF
    >>> a = AggregateCHF('G', 'gamma', shape=4, scale=25)
    >>> a.update(log2=12)
    >>> round(a.est_m, 6), round(a.actual_m, 6)
    (100.0, 100.0)
    """

    def __init__(self, name, family, form=None, note='', tags=(), hints='',
                 label=None, label_map=None, chf_scale=1.0, chf_loc=0.0,
                 **parameters):
        self.name = name
        if isinstance(family, Family):
            fam = family
            spec = {'family': fam.to_decl() or fam.name}
        else:
            fam = make_family(family, form, **parameters)
            spec = {'family': family, 'form': form, **parameters}
        self.chf_scale = float(chf_scale)
        self.chf_loc = float(chf_loc)
        if (self.chf_scale, self.chf_loc) != (1.0, 0.0):
            fam = FamilyAffine(fam, self.chf_scale, self.chf_loc)
            spec.update(chf_scale=self.chf_scale, chf_loc=self.chf_loc)
        self.family = fam
        self._init_labels(label=label, label_map=label_map)
        self.note = note
        self.tags = tuple(tags) if tags else ()
        self.hints = hints
        #: the relative error tolerance of the moment validation, as on ``Aggregate``
        self.validation_eps = get_settings().validation.eps
        #: the constructor's arguments, the DecL spec when built from DecL
        self._spec = {'name': name, **spec, 'note': note, 'tags': list(self.tags),
                      'hints': hints, 'label': label}
        #: an explicit grid left edge, set by :meth:`from_chf`; ``None`` means
        #: the left edge comes from the window rule
        self._x_min_fixed = None
        #: the last :meth:`sharpen` probe, as on ``Aggregate``
        self._sharpen_df = None
        self._sharpen_state = None
        self._reset_grid()
        #: the moment store, ``Aggregate.stats_df``'s rows and columns
        self.stats_df = self._theoretical_stats_df()

    def _reset_grid(self):
        self.log2 = 0
        self.bs = 0.0
        self.x_min = None
        self.bucket_sizing_p = BUCKET_SIZING_P
        self.xs = None
        self.agg_density = None
        self.truncation_error = np.nan
        #: the ``tilt`` keyword as given to :meth:`update`
        self.tilt_setting = True
        #: the realized tilts ``(left, right)`` in e-folds across the window
        self.tilt = (0.0, 0.0)
        #: ``tau / W``, the tilts per unit of loss
        self.tilt_theta = (0.0, 0.0)
        #: the law's mass below and above the window, left out by a tilted
        #: grid; the mass below sits in the first bucket
        self.outside_below = 0.0
        self.outside_above = 0.0
        #: sigma of the normal moved between the halves of a two-sided law
        self.transfer_sigma = 0.0
        #: the modeled worst relative bucket error at the realized tilt
        self.edge_error = np.nan
        self._gd = None
        self._density_df = None
        self.figure = None
        self._valid = None
        self._deficit = np.nan

    # ---- constructors -----------------------------------------------------
    @classmethod
    def from_family(cls, name, family, **trailer):
        """Build from a family object, or from a :class:`~aggregate.stable.Stable`.

        Parameters
        ----------
        name : str
        family : Family or Stable
        **trailer
            ``note``, ``tags``, ``hints``, ``label``.

        Returns
        -------
        AggregateCHF
            Not updated.
        """
        if isinstance(family, Stable):
            family = FamilyStable.from_stable(family)
        if not isinstance(family, Family):
            raise TypeError(f'from_family: expected a Family or Stable, got '
                            f'{type(family).__name__}')
        return cls(name, family, **trailer)

    @classmethod
    def from_chf(cls, name, chf, *, x_min, bs, log2, **trailer):
        """Build from a user-supplied characteristic function on an explicit grid.

        The successor of ``FourierTools(chf, 'continuous')``: any
        characteristic function, inverted untilted on the grid given, which
        the object cannot size itself because it knows nothing of the law's
        tails.

        Parameters
        ----------
        name : str
        chf : callable
            :math:`\\varphi(t)` on a real array; called on a complex array
            with zero imaginary part.
        x_min, bs, log2
            The grid: left edge, bucket, log2 of the number of buckets.
        **trailer
            ``note``, ``tags``, ``hints``, ``label``.

        Returns
        -------
        AggregateCHF
            Not updated; ``update()`` with no arguments uses the grid given.
        """
        obj = cls(name, FamilyUser(chf, name), **trailer)
        obj._x_min_fixed = float(x_min)
        obj._fixed_grid = (float(bs), int(log2))
        return obj

    # ---- the grid ---------------------------------------------------------
    @staticmethod
    def _tilt_mode(tilt):
        """``'auto'``, ``'off'`` or a float ``tau >= 0`` from the ``tilt`` keyword.

        ``bool`` is tested before ``float``, so ``tilt=1`` is one e-fold and
        ``tilt=True`` is automatic.
        """
        if isinstance(tilt, (bool, np.bool_)):
            return 'auto' if tilt else 'off'
        try:
            tau = float(tilt)
        except (TypeError, ValueError):
            raise ValueError(f'tilt must be True, False or a number >= 0, got {tilt!r}') from None
        if not np.isfinite(tau) or tau < 0:
            raise ValueError(f'tilt must be True, False or a number >= 0, got {tilt!r}')
        return tau

    def _tilt_family(self):
        """The one-sided family the tilted computation runs on: this one for a right heavy tail, its reflection for a left one, ``None`` otherwise."""
        fam = self.family
        if fam.tilt_side == 'right':
            return fam
        if fam.tilt_side == 'left':
            return fam.reflected()
        return None

    def _bucket_cap(self, family=None, two_sided=False):
        r"""The largest bucket a tilted grid may use, a rung of the ladder.

        :math:`h_{\max} = \min(\gamma/10,\ \pi\gamma/(52\log 2)^{1/\alpha})`
        rounded down (:func:`~aggregate.utilities._round_bucket_down`): the
        truncation :math:`|\varphi(\pi/h)|` must sit at roundoff because the
        tilt amplifies it by :math:`e^\tau`, and the body keeps ten buckets
        per scale because a tilted grid no longer has to reach far into the
        tail. A two-sided law needs both halves at roundoff, and since
        :math:`|\varphi| = |\varphi_+||\varphi_-|` that is the whole law at
        :math:`2^{-104}`: the cap is :math:`\pi\gamma/(104\log 2)^{1/\alpha}`,
        smaller by :math:`2^{1/\alpha}`. ``inf`` for a law with no power tail.
        """
        fam = family or self.family
        if not fam.power_tails:
            return np.inf
        _, _, alpha = fam.power_tails
        gamma = fam.scale
        h_trunc = np.pi * gamma / ((2 if two_sided else 1) * _ROUNDOFF_LOG) ** (1 / alpha)
        return _round_bucket_down(min(gamma / TILT_BODY_BUCKETS, h_trunc))

    @staticmethod
    def _edge_terms(fam, x0, h, n):
        r"""``(rho, eps_eff, kappa)`` of the one-sided tilt model on the window ``[x0, x0 + n h]``.

        The wrap damped by the tilt is :math:`\rho e^{-\tau}` with
        :math:`\rho = q^{\alpha+1}\zeta(\alpha+1, 1+q)`, :math:`q = (x_R-\delta_1)/W`
        (the Hurwitz zeta sums the power tail at :math:`x_R + mW`); the
        amplified absolute error is :math:`\varepsilon_{\mathrm{eff}}\kappa e^{\tau}`
        with :math:`\varepsilon_{\mathrm{eff}} = 2^{-52} + |\varphi(\pi/h)|`
        and :math:`\kappa = \bar f (x_R-\delta_1)^{\alpha+1}/(A\alpha)` the
        dynamic range of the masses across the window. Both are relative to
        the bucket mass at the right edge, where they are worst.
        """
        A, _, alpha = fam.power_tails
        W = n * h
        x_r = x0 + W
        q = max((x_r - fam.location) / W, 1e-12)
        rho = q ** (alpha + 1) * float(hurwitz_zeta(alpha + 1, 1 + q))
        eps_eff = 2.0 ** -52 + float(abs(fam.chf(np.array([np.pi / h]))[0]))
        kappa = fam.density_bound * (x_r - fam.location) ** (alpha + 1) / (A * alpha)
        return rho, eps_eff, kappa

    @staticmethod
    def _tau_star(rho, eps_eff, kappa):
        r""":math:`\tau^* = \max(0, \tfrac12\log(\rho/(\varepsilon_{\mathrm{eff}}\kappa)))`, the tilt that balances wrap against amplification."""
        return max(0.0, 0.5 * math.log(rho / (eps_eff * kappa)))

    @staticmethod
    def _edge_error_at(tau, rho, eps_eff, kappa):
        r"""The modeled worst relative bucket error :math:`\rho e^{-\tau} + \varepsilon_{\mathrm{eff}}\kappa e^{\tau}`."""
        return rho * math.exp(-tau) + eps_eff * kappa * math.exp(tau)

    def _one_sided_window(self, fam, log2, bucket_sizing_p, bs, mode):
        """Rule A or C for a right-heavy law to be tilted: ``(h, x0, tau)``."""
        n = 2 ** log2
        eps = 1.0 - bucket_sizing_p
        A, _, alpha = fam.power_tails
        lo, _ = fam.support
        if np.isfinite(lo):
            x_l = lo
        else:
            s_minus, _ = fam.cgf_domain
            x_l = -chernoff_edge(lambda u: fam.cgf(-u), -s_minus,
                                 math.log(TILT_LEFT_EPS), fam.scale)
        x_r = fam.location + (A / eps) ** (1 / alpha)
        if bs:
            h = float(bs)
        else:
            h = min(round_bucket((x_r - x_l) / n), self._bucket_cap(fam))
        x0 = h * math.floor(x_l / h + 1e-12)
        if mode == 'auto':
            tau = self._tau_star(*self._edge_terms(fam, x0, h, n))
        else:
            tau = float(mode)
        return h, x0, tau

    def _light_outside(self, h, x0, n):
        """``(below, above)`` for an untilted law: the Chernoff bounds on the mass outside ``[x0, x0 + n h)``, which wraps in."""
        fam = self.family
        if not np.isfinite(fam.scale) and fam.cgf_domain == (0.0, 0.0):
            return 0.0, 0.0
        lo, hi = fam.support
        s_minus, s_plus = fam.cgf_domain
        scale = fam.scale if np.isfinite(fam.scale) and fam.scale > 0 else 1.0
        x_top = x0 + n * h
        above = 0.0 if x_top >= hi else (
            chernoff_tail(fam.cgf, s_plus, x_top, scale) if s_plus > 0 else 1.0)
        below = 0.0 if x0 <= lo else (
            chernoff_tail(lambda u: fam.cgf(-u), -s_minus, -x0, scale) if s_minus < 0 else 1.0)
        return float(below), float(above)

    def _plain_window(self, log2, bucket_sizing_p, bs):
        r"""Rule A or C with no tilt: ``(h, x0)``.

        Rule A sizes the window from the edges at half the budget and splits
        the slack of ``round_bucket``'s rounding between the sides, cut at a
        finite support bound, so that neither tail's wrap lands on the far
        end of the other side; 0 stays a grid point. That bucket is then
        checked against frequency truncation: where :math:`|\varphi(\pi/h)|`
        is above the mass the window leaves outside (the Chernoff bounds,
        which wrap in), the bucket walks down the ladder, the window
        shortening around the body, to the rung that minimizes
        :math:`\max(|\varphi(\pi/h)|, \text{outside})`. A law whose
        characteristic function decays slowly for its tail's reach, such as a
        Tweedie with :math:`p > 2` and a small tilt, cannot have both at one
        ``log2``, and the rule picks the balance and flags both rows; a law
        rule A already resolves keeps rule A's bucket.
        """
        n = 2 ** log2
        if self._x_min_fixed is not None:
            return float(bs or self._fixed_grid[0]), self._x_min_fixed
        eps = 1.0 - bucket_sizing_p
        fam = self.family
        x_l, x_r = window_edges(fam, eps / 2)
        if not (np.isfinite(x_l) and np.isfinite(x_r)) or x_r <= x_l:
            raise ValueError(
                f'{self.name}: cannot size a window for {fam!r}; '
                'give bs and log2')
        lo, hi = fam.support
        # the grid points are the multiples of h from an anchor: 0, or an
        # atom, which must be a grid point (a shifted Tweedie's sits at b)
        anchor = fam.atom[0] if fam.atom is not None else 0.0

        def place(h):
            """The left edge for a bucket ``h``: the slack split, or the window centered on the body when it is short."""
            width = n * h
            if width >= x_r - x_l:
                slack = width - (x_r - x_l)
                x_left = max(lo, x_l - slack / 2)
            else:
                mean = fam.moments[0]
                center = mean if np.isfinite(mean) else 0.5 * (x_l + x_r)
                x_left = min(max(lo, center - width / 2), x_r - width)
                x_left = max(lo, x_left)
            return anchor + h * math.floor((x_left - anchor) / h + 1e-12)

        if bs:
            h = float(bs)
            return h, place(h)
        h = round_bucket((x_r - x_l) / n)
        trunc = truncation_at(fam, h)
        if trunc <= TRUNCATION_FLAG:
            return h, place(h)
        # walk the bucket down the ladder while that lowers the worse of the
        # two errors; at most twenty rungs, a factor of about a million
        best_h, best_x0 = h, place(h)
        best = max(trunc, sum(self._light_outside(h, best_x0, n)))
        for _ in range(20):
            h = _round_bucket_down(h * (1 - 1e-9))
            x0 = place(h)
            trunc = truncation_at(fam, h)
            score = max(trunc, sum(self._light_outside(h, x0, n)))
            if score < best:
                best, best_h, best_x0 = score, h, x0
            elif trunc < score:
                # the outside mass now dominates and only grows
                break
        return best_h, best_x0

    def _light_edge(self, fam, h, sigma2=0.0):
        """The left edge of a right-heavy half: its support bound or the 1e-40 Chernoff edge, a multiple of ``h``.

        ``sigma2`` is the variance of a normal law added to the half by the
        resolution transfer, which widens its light tail: the edge is then
        taken from ``K(s) + sigma2 s ** 2 / 2``, and a finite support bound no
        longer applies. Mass below this edge wraps onto the top of the window
        and is amplified by the tilt, which is why the edge is so far out.
        """
        lo, _ = fam.support
        if np.isfinite(lo) and sigma2 == 0:
            x_l = lo
        else:
            s_minus, _ = fam.cgf_domain
            x_l = -chernoff_edge(lambda u: fam.cgf(-u) + 0.5 * sigma2 * u ** 2, -s_minus,
                                 math.log(TILT_LEFT_EPS), max(fam.scale, math.sqrt(sigma2)))
        return h * math.floor(x_l / h + 1e-12)

    def _two_sided_setup(self, log2, bucket_sizing_p, bs, mode):
        r"""Everything the split needs short of inverting: the bucket, the halves, their grids, the transfer and the tilts.

        Returns a dict with ``h``, ``plus`` and ``minus_reflected`` (the
        families inverted, both right-heavy), ``y0_plus`` and ``z0_minus``
        (their left edges), ``sigma`` and ``narrow`` (the resolution
        transfer), ``chf_plus`` and ``chf_minus`` (the transferred
        characteristic functions), and ``tau_plus`` and ``tau_minus``.

        Notes
        -----
        **Resolution transfer.** Near :math:`|\beta| = 1` the light half is
        narrow, under a bucket at :math:`\alpha = 1.5`, :math:`\beta = 0.99`,
        and its truncation :math:`|\varphi_-(\pi/h)|` is of order one, which
        the tilt amplifies. A normal law is moved from the wide half to the
        narrow one, :math:`\varphi = (\varphi_+ e^{\sigma^2 s^2/2})(\varphi_- e^{-\sigma^2 s^2/2})`,
        with :math:`\sigma^2\pi^2/(2h^2) = \max(0, 52\log 2 - (\pi\gamma_-/h)^\alpha)`,
        the least :math:`\sigma` that brings the narrow half's truncation to
        roundoff; the product is unchanged, and the wide half becomes a
        deconvolved, signed intermediate, which is harmless. The two-sided
        bucket cap keeps the wide half at roundoff too.
        """
        n = 2 ** log2
        fam = self.family
        A, B, alpha = fam.power_tails
        eps = 1.0 - bucket_sizing_p
        x_r = fam.location + (A / (eps / 2)) ** (1 / alpha)
        x_l = fam.location - (B / (eps / 2)) ** (1 / alpha)
        if bs:
            h = float(bs)
        else:
            h = min(round_bucket((x_r - x_l) / n), self._bucket_cap(two_sided=True))
        plus, minus = fam.halves()
        minus_r = minus.reflected()
        # the truncation exponents (pi gamma / h) ** alpha of the two halves
        e_plus = (np.pi * plus.scale / h) ** alpha
        e_minus = (np.pi * minus_r.scale / h) ** alpha
        narrow = 'minus' if e_minus <= e_plus else 'plus'
        e_narrow = min(e_plus, e_minus)
        sigma2 = max(0.0, _ROUNDOFF_LOG - e_narrow) * 2 * h ** 2 / np.pi ** 2
        sigma = math.sqrt(sigma2)

        def transferred(base, sign):
            def chf(s):
                s = np.asarray(s, dtype=complex)
                return base.chf(s) * np.exp(sign * 0.5 * sigma2 * s ** 2)
            return chf
        # the narrow half gains the normal (exp(-sigma^2 s^2 / 2)), the wide
        # half loses it; the reflected minus half sees s -> -s, under which
        # s ** 2 is unchanged
        chf_plus = transferred(plus, -1.0 if narrow == 'plus' else +1.0)
        chf_minus = transferred(minus_r, -1.0 if narrow == 'minus' else +1.0)
        y0_plus = self._light_edge(plus, h, sigma2 if narrow == 'plus' else 0.0)
        z0_minus = self._light_edge(minus_r, h, sigma2 if narrow == 'minus' else 0.0)
        bound = {'plus': plus.density_bound, 'minus': minus_r.density_bound}
        if sigma > 0:
            bound[narrow] = min(bound[narrow], 1 / (sigma * math.sqrt(2 * np.pi)))
        taus = {}
        for key, half, chf, y0 in (('plus', plus, chf_plus, y0_plus),
                                   ('minus', minus_r, chf_minus, z0_minus)):
            if mode == 'auto':
                A_h, _, _ = half.power_tails
                # each half is inverted on 2n buckets (see _two_sided_masses)
                W = 2 * n * h
                x_end = y0 + W
                q = max((x_end - half.location) / W, 1e-12)
                rho = q ** (alpha + 1) * float(hurwitz_zeta(alpha + 1, 1 + q))
                eps_eff = 2.0 ** -52 + float(abs(chf(np.array([np.pi / h]))[0]))
                kappa = bound[key] * (x_end - half.location) ** (alpha + 1) / (A_h * alpha)
                taus[key] = self._tau_star(rho, eps_eff, kappa)
            else:
                taus[key] = float(mode)
        return dict(h=h, n=n, plus=plus, minus_reflected=minus_r, y0_plus=y0_plus,
                    z0_minus=z0_minus, sigma=sigma, narrow=narrow,
                    chf_plus=chf_plus, chf_minus=chf_minus,
                    tau_plus=taus['plus'], tau_minus=taus['minus'])

    def _two_sided_masses(self, setup):
        r"""Invert both halves, convolve, crop: ``(p, x0, below, above, edge_error)``.

        Notes
        -----
        **The bucket factor is applied once.** Convolving two sets of bucket
        masses gives the masses of :math:`X + U_1 + U_2`, a second rounding
        with relative error of order :math:`(h/\gamma)^2/24`; so the plus half
        is inverted with the sinc factor and the minus half without, and the
        product of the transforms is :math:`\varphi\,\mathrm{sinc}` on the
        band.

        **Each half on twice the grid, then crop.** Mass of one half beyond
        its own window re-enters the kept window through the other half's
        heavy tail, so a half on :math:`n/2` buckets loses accuracy near both
        edges, and a half on :math:`n` still does at the edge a skewed law
        pushes the crop to, where the last kept entries pair the plus half's
        masses beyond its window with the lower half of the minus body
        (measured: 40% to 55% at :math:`\beta = 0.99`). Each half is
        therefore inverted on :math:`2n` buckets, so its window reaches a
        full :math:`W` beyond the kept one on its side, the linear
        convolution has :math:`4n`, and the :math:`n` kept are placed, within
        the range where every entry is complete, so that the mass left out
        below equals the mass left out above. The left-out masses are then
        the convolution's mass outside the kept window plus each half's
        closed-form tail beyond its own window. The cost is four inversions
        of :math:`n` and a convolution of :math:`4n`, about 40 ms at log2 16.

        **The reported edge error** is :math:`\max(E_L, E_R, \text{outside})`
        with :math:`E_L` and :math:`E_R` the one-sided formula applied to the
        whole law, with that side's tail constant and the distance from
        :math:`\delta_1` to that edge: the halves' own :math:`E^*` overstate
        the assembled law's error by up to 300 times near :math:`|\beta| = 1`.
        """
        h, n = setup['h'], setup['n']
        m = 2 * n                                     # each half's buckets
        W_half = m * h
        W = n * h
        plus, minus_r = setup['plus'], setup['minus_reflected']
        p_plus = invert_characteristic_function(
            setup['chf_plus'], setup['y0_plus'], h, m, setup['tau_plus'] / W_half, bucket=True)
        p_minus = invert_characteristic_function(
            setup['chf_minus'], setup['z0_minus'], h, m, setup['tau_minus'] / W_half, bucket=False)[::-1]
        m0 = -(setup['z0_minus'] + (m - 1) * h)       # the minus half's grid starts here
        full = np.fft.irfft(np.fft.rfft(p_plus, 2 * m) * np.fft.rfft(p_minus, 2 * m), 2 * m)
        start = setup['y0_plus'] + m0
        # the balanced crop: n consecutive buckets with equal mass left out
        # below and above, counting each half's closed-form tail beyond its
        # own window on its side
        tail_plus = plus.tail_mass_above(setup['y0_plus'] + (m - 1) * h + h / 2)
        tail_minus = minus_r.tail_mass_above(setup['z0_minus'] + (m - 1) * h + h / 2)
        cum = np.concatenate([[0.0], np.cumsum(full)])
        total = cum[-1]
        # entry j of the linear convolution is complete only where both
        # halves' windows cover the pairs it needs: not below the plus half's
        # body (its light edge sits b_plus buckets under its location) and not
        # above 2m - 2 - b_minus, so the crop is placed inside that range
        b_plus = int(math.ceil((plus.location - setup['y0_plus']) / h - 1e-9))
        b_minus = int(math.ceil((minus_r.location - setup['z0_minus']) / h - 1e-9))
        k_lo = max(0, min(b_plus, m - 1))
        k_hi = max(k_lo, 2 * m - 1 - n - max(b_minus, 0))
        j = np.arange(k_lo, k_hi + 1)
        below = cum[j] + tail_minus
        above = total - cum[j + n] + tail_plus
        k = int(j[np.argmin(np.abs(below - above))])
        p = full[k:k + n].copy()
        x0 = start + k * h
        out_below = float(cum[k] + tail_minus)
        out_above = float(total - cum[k + n] + tail_plus)
        p[0] += out_below
        # the two-sided edge error: the one-sided formula on the whole law,
        # each side with its own tail constant and distance
        fam = self.family
        A, B, alpha = fam.power_tails
        eps_eff = 2.0 ** -52 + float(abs(fam.chf(np.array([np.pi / h]))[0]))
        errors = []
        for const, dist, tau in ((A, x0 + W - fam.location, setup['tau_plus']),
                                 (B, fam.location - x0, setup['tau_minus'])):
            if const <= 0 or dist <= 0:
                continue
            q = dist / W
            rho = q ** (alpha + 1) * float(hurwitz_zeta(alpha + 1, 1 + q))
            kappa = fam.density_bound * dist ** (alpha + 1) / (const * alpha)
            errors.append(self._edge_error_at(tau, rho, eps_eff, kappa))
        edge_error = max([*errors, out_below + out_above])
        return p, float(x0), out_below, out_above, float(edge_error)

    def grid_window(self, log2=16, bucket_sizing_p=BUCKET_SIZING_P, tilt=True, bs=0):
        """The grid the rules would choose, without computing.

        Parameters
        ----------
        log2 : int
        bucket_sizing_p : float
            The mass the window must cover (rule A).
        tilt : bool or float, default True
            As on :meth:`update`.
        bs : float, default 0
            A given bucket (rule C); 0 means rule A.

        Returns
        -------
        tuple
            ``(bs, log2, x_min, tilt)`` with ``tilt`` the pair
            ``(left, right)`` of e-folds across the window the rules choose:
            ``(0, tau)`` for a right heavy tail, ``(tau, 0)`` for a left one,
            ``(tau_minus, tau_plus)`` for a two-sided law, ``(0, 0)``
            untilted. ``x_min`` is ``None`` for a tilted two-sided law, whose
            window is placed by the crop during the computation.
        """
        log2 = int(log2)
        n = 2 ** log2
        if not 0 < 1.0 - bucket_sizing_p < 1:
            raise ValueError(f'bucket_sizing_p must be in (0, 1), got {bucket_sizing_p}')
        mode = self._tilt_mode(tilt)
        if mode not in ('auto', 'off') and not self.family.power_tails:
            raise ValueError(
                f'{self.name}: tilt={tilt!r} on a {self.family.name} law, which '
                'has no power tail; such a law barely wraps, so there is '
                'nothing for a tilt to damp. Use tilt=True or tilt=False.')
        if mode != 'off' and self._x_min_fixed is None and self.family.tilt_side == 'split':
            setup = self._two_sided_setup(log2, bucket_sizing_p, bs, mode)
            return float(setup['h']), log2, None, (setup['tau_minus'], setup['tau_plus'])
        fam = self._tilt_family() if (mode != 'off' and self._x_min_fixed is None) else None
        if fam is None:
            h, x0 = self._plain_window(log2, bucket_sizing_p, bs)
            return float(h), log2, float(x0), (0.0, 0.0)
        h, y0, tau = self._one_sided_window(fam, log2, bucket_sizing_p, bs, mode)
        if self.family.tilt_side == 'left':
            # the reflection: x = -y, so the grid runs from -(y0 + (n - 1) h)
            return float(h), log2, float(-(y0 + (n - 1) * h)), (tau, 0.0)
        return float(h), log2, float(y0), (0.0, tau)

    def _outside_at(self, log2, h, x0):
        """``(below, above)``: the closed-form mass the law leaves outside the tilted window ``[x0, x0 + n h)``."""
        n = 2 ** log2
        if self.family.tilt_side == 'split':
            # an estimate for coverage advice: a window n h long placed so
            # that the two leading-term tails balance, A d+ ** -alpha = B d- ** -alpha
            A, B, alpha = self.family.power_tails
            W = n * h
            d_plus = W / (1 + (B / A) ** (1 / alpha))
            d_minus = W - d_plus
            return B * d_minus ** -alpha, A * d_plus ** -alpha
        fam = self._tilt_family()
        if fam is None:
            if self._x_min_fixed is not None:
                return 0.0, 0.0
            return self._light_outside(h, x0, n)
        if self.family.tilt_side == 'left':
            y_top = -(x0 - h / 2)
            return fam.tail_mass_above(y_top), 0.0
        return 0.0, fam.tail_mass_above(x0 + (n - 1) * h + h / 2)

    def recommended_grid(self, epsilon=None):
        """The smallest ``(log2, bs)`` whose window reaches mass ``epsilon`` beyond it at the capped bucket and full resolution.

        Parameters
        ----------
        epsilon : float, optional
            The mass allowed beyond the window; default ``1 - bucket_sizing_p``.

        Returns
        -------
        tuple
            ``(log2, bs)``; ``log2`` is at most :data:`RECOMMENDED_LOG2_MAX`,
            and the pair returned at that size may not meet the target, which
            :attr:`validation_df` says plainly.

        Notes
        -----
        For a tilted law the bucket is capped (:meth:`_bucket_cap`), so the
        window is ``2 ** log2`` capped buckets long and coverage grows only
        with ``log2``; for every other law rule A always covers, and what
        grows ``log2`` is resolution, at least :data:`MIN_BUCKETS_PER_SCALE`
        buckets per body scale (:attr:`Family.scale`).
        """
        log2, h, _ = self._recommend(epsilon)
        return log2, h

    def _recommend(self, epsilon=None):
        """``(log2, bs, met)``: :meth:`recommended_grid` and whether its grid meets the target."""
        if self._x_min_fixed is not None:
            return self._fixed_grid[1], self._fixed_grid[0], True
        eps = epsilon if epsilon is not None else 1.0 - self.bucket_sizing_p
        p = 1.0 - eps
        scale = self.family.scale
        for log2 in range(8, RECOMMENDED_LOG2_MAX + 1):
            h, _, x0, _ = self.grid_window(log2, p)
            resolved = not np.isfinite(scale) or scale / h >= MIN_BUCKETS_PER_SCALE
            truncated = truncation_at(self.family, h) > TRUNCATION_FLAG
            # a two-sided law has no x_min before the crop; _outside_at estimates it
            covered = sum(self._outside_at(log2, h, x0)) <= eps
            if resolved and covered and not truncated:
                return log2, h, True
        return RECOMMENDED_LOG2_MAX, h, False

    def _recommended_grid_met(self):
        """Whether some grid up to ``RECOMMENDED_LOG2_MAX`` meets coverage, truncation and resolution; ``False`` sets ``INFEASIBLE``."""
        return self._recommend()[2]

    def _coarse(self):
        """Whether the grid has fewer than ``MIN_BUCKETS_PER_SCALE`` buckets per body scale."""
        scale = self.family.scale
        return bool(np.isfinite(scale) and scale / self.bs < MIN_BUCKETS_PER_SCALE)

    def update(self, log2=None, bs=0, bucket_sizing_p=BUCKET_SIZING_P, tilt=True,
               **kwargs):
        r"""Compute the bucket masses on the grid the rules choose.

        Parameters
        ----------
        log2 : int, optional
            The grid has ``2 ** log2`` buckets; default 16, or the grid given
            to :meth:`from_chf`.
        bs : float, default 0
            The bucket; 0 means rule A sizes it from ``log2`` and the tails,
            or the bucket given to :meth:`from_chf`.
        bucket_sizing_p : float
            The mass the window covers under rule A.
        tilt : bool or float, default True
            ``True``: automatic. A stable law with ``beta = 1`` is inverted
            along the shifted contour at the tilt :math:`\tau^*` that balances
            the damped wrap against the amplified roundoff; ``beta = -1`` is
            the reflection, so the tilt is always toward the heavy tail; the
            bucket is capped (:meth:`_bucket_cap`); every other law is
            inverted on the real line, since a light tail barely wraps.
            ``False``: never tilt and never cap; the law wraps, which is the
            teaching comparison. A number :math:`\tau \ge 0`: tilt by exactly
            :math:`\tau` e-folds across the window, cap applied. ``tilt=8``
            damps the wrapped tail by :math:`e^{-8} \approx 3\times10^{-4}`
            and amplifies roundoff at the edge by :math:`e^{8} \approx 3\,000`.
            A number on a law with no power tail is a ``ValueError``. ``bool``
            is tested before ``float``, so ``tilt=1`` is one e-fold.
        **kwargs
            Other ``hints{}`` keys reach here and are ignored with a logged
            warning, as ``Aggregate.update`` ignores keys it does not use.

        Notes
        -----
        A tilted grid leaves the mass beyond its window out and reports it:
        :attr:`outside_above` is the closed-form tail mass above the window
        and :attr:`outside_below` the mass below it, which is **added to the
        first bucket**, so the cumulative masses, and with them ``cdf``, ``q``
        and ``sf = 1 - cdf``, are right throughout the window for every tilt.
        The first bucket then means "the left edge and everything below it".
        ``tvar``, ``lev`` and the grid moments are window statistics. An
        untilted grid wraps the outside mass in and both are 0.

        A stable law with :math:`|\beta| < 1` has two heavy tails and no
        contour serves it: it is split into its two totally skewed halves
        (:meth:`FamilyStable.halves`), each tilted toward its own light side
        on all ``2 ** log2`` buckets, combined by a linear convolution so that
        nothing wraps, and cropped to the ``2 ** log2`` buckets that leave
        equal mass below and above (:meth:`_two_sided_masses`); the window's
        left edge is then an output. A number tilts both halves alike.
        """
        if kwargs:
            logger.warning('%s.update ignores %s', type(self).__name__,
                           ', '.join(sorted(kwargs)))
        if self._x_min_fixed is not None:
            bs = bs or self._fixed_grid[0]
            log2 = log2 or self._fixed_grid[1]
        log2 = 16 if log2 is None else int(log2)
        n = 2 ** log2
        mode = self._tilt_mode(tilt)
        sigma = 0.0
        h, log2, x0, tau_pair = self.grid_window(log2, bucket_sizing_p, tilt, bs)
        fam = self._tilt_family() if mode != 'off' else None
        below = above = 0.0
        if mode != 'off' and self._x_min_fixed is None and self.family.tilt_side == 'split':
            setup = self._two_sided_setup(log2, bucket_sizing_p, bs, mode)
            p, x0, below, above, edge_error = self._two_sided_masses(setup)
            sigma = setup['sigma']
        elif fam is None:
            p = invert_characteristic_function(self.family.chf, x0, h, n, 0.0,
                                               self.family.atom)
            # the mass outside an untilted window wraps in; it is reported,
            # and stays out of the first bucket
            below, above = self._outside_at(log2, h, x0)
            edge_fam = self._tilt_family()
            if edge_fam is not None and self.family.tilt_side != 'split':
                # the wrap of an untilted heavy tail, for the record
                y0 = x0 if self.family.tilt_side == 'right' else -(x0 + (n - 1) * h)
                edge_error = self._edge_error_at(0.0, *self._edge_terms(edge_fam, y0, h, n))
            else:
                edge_error = np.nan
        else:
            tau = max(tau_pair)
            W = n * h
            y0 = x0 if self.family.tilt_side == 'right' else -(x0 + (n - 1) * h)
            p = invert_characteristic_function(fam.chf, y0, h, n, tau / W, fam.atom)
            edge_error = self._edge_error_at(tau, *self._edge_terms(fam, y0, h, n))
            if self.family.tilt_side == 'left':
                p = p[::-1].copy()
            below, above = self._outside_at(log2, h, x0)
            p[0] += below
        self.log2 = log2
        self.bs = h
        self.x_min = x0
        self.bucket_sizing_p = bucket_sizing_p
        self.tilt_setting = tilt
        self.tilt = tau_pair
        self.tilt_theta = tuple(t / (n * h) for t in tau_pair)
        self.outside_below = below
        self.outside_above = above
        self.transfer_sigma = sigma
        self.edge_error = edge_error
        self.xs = x0 + h * np.arange(n)
        self.agg_density = p
        self.truncation_error = truncation_at(self.family, h)
        self._gd = None
        self._density_df = None
        self._valid = None
        self._deficit = np.nan
        self._write_empirical_moments()

    @property
    def outside_mass(self):
        """``outside_below + outside_above``: the law's mass the window leaves out."""
        return self.outside_below + self.outside_above

    @property
    def x_max(self):
        """The last grid point, or ``None`` before :meth:`update`."""
        return None if self.xs is None else float(self.xs[-1])

    # ---- the grid distribution and its accessors -------------------------
    def _require_update(self):
        if self.agg_density is None:
            raise ValueError(f'{self.name}: update() first')

    def _output_xs_ps(self):
        """``(xs, ps)`` of the law this object outputs, for a ``sev chfagg.NAME`` reference.

        The one hook :func:`aggregate._severity._dhistogram_from_object` reads;
        see :meth:`aggregate.Aggregate._output_xs_ps`. On a tilted grid the
        first bucket holds the mass below the window and the masses above it
        are left out, so the reference carries a deficit of
        :attr:`outside_above`, which the deficit warning reports with the
        advice to raise the inner's ``log2``. ``(None, None)`` before
        ``update``.
        """
        return self.xs, self.agg_density

    def with_hints(self, **extra):
        """This law's program with its realized grid pinned into ``hints{}``.

        The certification helper for the ``sev chfagg.NAME`` reference, the
        same contract as :meth:`aggregate.Aggregate.with_hints`: the
        referenced declaration must carry explicit ``log2`` and ``bs``, so
        that the severity it stands for is pinned by the recipe base. ``tilt``
        is a hint too, and travels through ``extra``.
        """
        from ._program import with_hints
        return with_hints(self, **extra)

    def _grid_distribution(self):
        """The held :class:`~aggregate._grid_distribution.GridDistribution`, built on first use."""
        self._require_update()
        if self._gd is None:
            self._gd = GridDistribution(self.xs, self.agg_density, self.bs,
                                        self.name, True)
        return self._gd

    @property
    def _is_loss_value(self):
        """Loss orientation, as on ``Aggregate``."""
        return True

    @property
    def bounded(self):
        """``False``: every family in scope has an unbounded support."""
        return False

    @property
    def support(self):
        """``(lower, upper)`` of the law."""
        return self.family.support

    def q(self, p, kind='lower'):
        """Quantile at level ``p``, ``kind`` in ``{'lower', 'upper'}``."""
        return self._grid_distribution().q(p, kind)

    def tvar(self, p):
        """Tail value at risk at level ``p``, from the window's masses."""
        return self._grid_distribution().tvar(p)

    def cdf(self, x, kind='previous'):
        """``Pr(X <= x)`` on the grid, the step function through the cumulative masses.

        ``kind`` is accepted for signature parity with ``Aggregate.cdf`` and
        has one behavior: the right-continuous step.
        """
        return self._grid_distribution().cdf(x)

    def sf(self, x):
        """``Pr(X > x) = 1 - cdf(x)``."""
        return self._grid_distribution().sf(x)

    def pdf(self, x):
        """The density reading ``mass / bs``, linearly interpolated."""
        return self._grid_distribution().pdf(x)

    def pmf(self, x):
        """The mass at a grid point ``x``, 0 elsewhere."""
        return self._grid_distribution().pmf(x)

    def snap(self, x):
        """Snap ``x`` to the nearest grid point."""
        return self._grid_distribution().snap(x)

    def prob_loss_assets(self, *, p=None, L=None, a=None):
        """Given one of ``p``, ``L``, ``a``, all three, snapped to the grid."""
        return self._grid_distribution().prob_loss_assets(p=p, L=L, a=a)

    def var_dict(self, p, kind='lower', snap=False):
        """``{name: q(p)}``, the one-entry form of ``Portfolio.var_dict``."""
        x = self.q(p, kind)
        if snap:
            x = self.snap(x)
        return {self.name: x}

    # ---- pricing ----------------------------------------------------------
    def price_pentagon(self, *, p=None, a=None, P=None, M=None, Q=None,
                       LR=None, PQ=None, ROE=None):
        """Complete the pricing pentagon at a capital level given one target.

        The ``Aggregate`` method over this grid: fix the capital level with
        exactly one of ``p`` (a VaR probability) or ``a`` (an asset level,
        snapped to the grid), supply exactly one pricing target (``P``,
        ``ROE``, or a loss ratio via ``LR``, ``M``, ``Q`` or ``PQ``), and
        read the canonical one-row pentagon. See
        :meth:`aggregate.Aggregate.price_pentagon`.

        Notes
        -----
        Every price is at a finite asset level ``a``, so it is finite even
        for a stable law with ``alpha <= 1``. ``a`` must sit inside the
        window: on an untilted heavy-tailed grid the survival there should be
        large against :attr:`outside_mass`, the wrapped tail, and a tilted
        grid's ``S(a)`` is right to the edge. ``p = 1`` is refused, since
        :attr:`bounded` is ``False`` for every family.
        """
        from . import _pricing
        return _pricing.price_pentagon(self, p=p, a=a, P=P, M=M, Q=Q, LR=LR,
                                       PQ=PQ, ROE=ROE)

    def calibrate_distortions(self, coc=None, *, lr=None, p=None, a=None,
                              kind='lower', names=None):
        """Calibrate the standard distortion set to a pricing target on this grid.

        The ``Aggregate`` method over this grid; see
        :meth:`aggregate.Aggregate.calibrate_distortions` for the arguments
        and the result. ``names`` defaults to
        ``_pricing.DEFAULT_CALIBRATION_DISTORTIONS``. The cautions under
        :meth:`price_pentagon` apply.
        """
        from . import _pricing
        if names is None:
            names = _pricing.DEFAULT_CALIBRATION_DISTORTIONS
        return _pricing.calibrate_distortions(self, coc, lr=lr, p=p, a=a,
                                              kind=kind, names=names)

    # ---- moments ----------------------------------------------------------
    def _theoretical_moments(self):
        r"""``(ex1, ex2, ex3, mean, cv, skew)`` of the law, the six measures of a moment store block.

        Notes
        -----
        From the family's mean :math:`m`, variance :math:`v` and skewness
        :math:`k`: :math:`\mathsf{P}X^2 = v + m^2` and
        :math:`\mathsf{P}X^3 = k v^{3/2} + 3mv + m^3`. A law with no variance
        has :math:`\mathsf{P}X^2 = \infty` and :math:`\mathsf{P}X^3` of the
        sign of its heavy tail, undefined with two, the conventions the store
        uses for a power-law severity; its CV is infinite at a finite nonzero
        mean. The CV is blank (``nan``) at a mean indistinguishable from 0,
        as :meth:`aggregate.Aggregate._cv_or_nan` has it.
        """
        mean, var, skew = (float(v) for v in self.family.moments)
        if np.isfinite(var):
            sd = math.sqrt(max(var, 0.0))
            ex2 = var + mean ** 2
            ex3 = skew * sd ** 3 + 3 * mean * var + mean ** 3
            cv = Aggregate._cv_or_nan(mean, sd)
        else:
            ex2 = np.inf if var == np.inf else np.nan
            ex3 = {'right': np.inf, 'left': -np.inf}.get(self.family.tilt_side, np.nan)
            cv = np.inf if (np.isfinite(mean) and mean != 0) else np.nan
        return mean, ex2, ex3, mean, cv, skew

    def _theoretical_stats_df(self):
        """The moment store before :meth:`update`: the law's moments in the theoretical columns.

        Rows ``_STATS_ROW_INDEX`` and columns ``Aggregate.stats_df``'s. The
        ``agg`` block holds the law in ``e0.m0``, ``mixed`` and
        ``independent``; ``meta`` holds what a one-component ``Aggregate``
        with no premium and no limit has, with ``lr`` and ``sevcv_param``
        ``nan``; the ``freq`` and ``sev`` blocks and the reinsurance columns
        stay ``nan``, as on an ``Aggregate`` with no reinsurance.
        """
        df = pd.DataFrame(np.nan, index=_STATS_ROW_INDEX, columns=STATS_COLUMNS)
        th = self._theoretical_moments()
        meta = {'limit': np.inf, 'attachment': 0.0, 'el': th[3], 'prem': 0.0,
                'mix_cv': 0.0, 'wt': 1.0}
        for col in ('e0.m0', 'mixed', 'independent'):
            for measure, value in meta.items():
                df.loc[('meta', measure), col] = value
            for measure, value in zip(_MEASURES, th):
                df.loc[('agg', measure), col] = value
        return df

    def _write_empirical_moments(self):
        """Write the grid's moments into ``empirical`` and ``gross_empirical`` and refresh ``error``.

        ``error`` is the noise-aware relative error of ``gross_empirical``
        against ``mixed``, as on ``Aggregate``, and ``nan`` where the
        theoretical value is not finite: the grid value is then a statistic of
        the window.
        """
        df = self.stats_df
        x, p = self.xs, self.agg_density
        m, sd, skew = self._grid_moments()
        with np.errstate(over='ignore', invalid='ignore'):
            raw = (m, float(np.sum(x ** 2 * p)), float(np.sum(x ** 3 * p)))
        emp = (*raw, m, Aggregate._cv_or_nan(m, sd), skew)
        for col in ('empirical', 'gross_empirical'):
            for measure, value in zip(_MEASURES, emp):
                df.loc[('agg', measure), col] = value
        th, gr = df['mixed'], df['gross_empirical']
        ok = np.isfinite(th) & np.isfinite(gr)
        err = pd.Series(np.nan, index=df.index)
        err[ok] = _noise_aware_rel_error(gr[ok], th[ok])
        df['error'] = err

    @property
    def actual_m(self):
        """The theoretical mean, ``inf`` or ``nan`` where it does not exist."""
        return float(self.stats_df.loc[('agg', 'mean'), 'mixed'])

    @property
    def actual_cv(self):
        """The theoretical coefficient of variation, ``inf`` or ``nan`` where it does not exist."""
        return float(self.stats_df.loc[('agg', 'cv'), 'mixed'])

    @property
    def actual_skew(self):
        """The theoretical skewness, ``nan`` where it does not exist."""
        return float(self.stats_df.loc[('agg', 'skew'), 'mixed'])

    @property
    def actual_var(self):
        """The theoretical variance, ``inf`` where it does not exist."""
        return float(self.family.moments[1])

    @property
    def actual_sd(self):
        """The theoretical standard deviation, ``inf`` where it does not exist."""
        v = self.actual_var
        return math.sqrt(max(v, 0.0)) if np.isfinite(v) else v

    def _grid_moments(self):
        """``(mean, sd, skew)`` of the window's masses, no renormalization."""
        self._require_update()
        x, p = self.xs, self.agg_density
        m = float(np.sum(x * p))
        c = x - m
        var = float(np.sum(c ** 2 * p))
        sd = math.sqrt(max(var, 0.0))
        with np.errstate(divide='ignore', invalid='ignore'):
            skew = float(np.sum(c ** 3 * p) / sd ** 3) if sd > 0 else np.nan
        return m, sd, skew

    @property
    def est_m(self):
        """The grid mean, a window statistic when the law has no mean."""
        return self._grid_moments()[0]

    @property
    def est_sd(self):
        """The grid standard deviation."""
        return self._grid_moments()[1]

    @property
    def est_var(self):
        """The grid variance."""
        return self._grid_moments()[1] ** 2

    @property
    def est_cv(self):
        """The grid coefficient of variation, ``nan`` at a mean indistinguishable from 0 (``Aggregate._cv_or_nan``)."""
        m, sd, _ = self._grid_moments()
        return Aggregate._cv_or_nan(m, sd)

    @property
    def est_skew(self):
        """The grid skewness."""
        return self._grid_moments()[2]

    @property
    def total_mass(self):
        """The sum of the masses on the grid."""
        self._require_update()
        return float(np.sum(self.agg_density))

    # ---- frames -----------------------------------------------------------
    @property
    def density_df(self):
        """Per-bucket frame: ``loss``, ``p_total``, ``p``, ``F``, ``S``, ``lev``, ``exa``, indexed by ``loss``.

        The columns ``Aggregate.density_df`` carries that pricing reads;
        ``lev`` is the direct sum :math:`\\sum_{x \\le a} x\\,p + a\\,S(a)`,
        which carries the window origin.
        """
        if self._density_df is None:
            self._require_update()
            x = np.asarray(self.xs, dtype=float)
            p = self.agg_density
            df = pd.DataFrame({'loss': x, 'p_total': p}).set_index('loss', drop=False)
            df['p'] = df['p_total']
            F = np.cumsum(p)
            df['F'] = F
            df['S'] = 1.0 - F
            df['lev'] = np.cumsum(x * p) + x * df['S'].to_numpy()
            df['exa'] = df['lev']
            self._density_df = df
        return self._density_df

    @property
    def summary_df(self):
        """Moments and key percentiles in ``Aggregate.summary_df``'s shape: rows ``Freq``, ``Sev``, ``Agg``.

        Columns ``Mean | SD | CV | Skew | P01 | Median | P99``. The ``Freq``
        and ``Sev`` rows are ``nan``, a law computed from its characteristic
        function having neither. The ``Agg`` row holds the grid's moments and
        percentiles after :meth:`update` and the theoretical moments before,
        with ``CV`` through :meth:`aggregate.Aggregate._cv_or_nan`, blank at a
        mean indistinguishable from 0.
        """
        updated = self.agg_density is not None
        if updated:
            m, sd, skew = self._grid_moments()
        else:
            m, sd, skew = self.actual_m, self.actual_sd, self.actual_skew
        nan3 = [np.nan] * 3
        df = pd.DataFrame({'Mean': [np.nan, np.nan, m], 'SD': [np.nan, np.nan, sd],
                           'CV': [*nan3[:2], Aggregate._cv_or_nan(m, sd)],
                           'Skew': [np.nan, np.nan, skew]},
                          index=['Freq', 'Sev', 'Agg'])
        df.index.name = 'X'
        pcols = [_summary_pct_label(p) for p in SUMMARY_PERCENTILES]
        for pc in pcols:
            df[pc] = np.nan
        if updated:
            for p, pc in zip(SUMMARY_PERCENTILES, pcols):
                df.loc['Agg', pc] = self.q(p)
        for c in ('Mean', 'SD', 'Skew', *pcols):
            df[c] = _snap_noise(df[c])
        df.attrs['mean'] = m
        return df

    @property
    def tail_df(self):
        """Return-period table on the default ladder, as ``Aggregate.tail_df``."""
        return self.tail_periods_df()

    def tail_periods_df(self, periods=None):
        """Return-period table; ``None`` before :meth:`update`."""
        if self.agg_density is None:
            return None
        return return_period_frame(self.q, self.tvar, self.est_m, periods)

    def _validation_rows(self):
        """The validation checks as ``(name, value, target, flagged, note)`` tuples."""
        self._require_update()
        eps = get_settings().validation.eps
        rows = []
        total = self.total_mass
        tilted = max(self.tilt) > 0
        target = 1.0 - self.outside_above if tilted else 1.0
        rows.append(('total mass', total, target, abs(total - target) > 1e-6,
                     'sum of the masses on the grid against 1 less the mass above the '
                     'window' if tilted else
                     'sum of the masses on the grid; an untilted grid wraps everything in'))
        neg = float(np.min(self.agg_density))
        rows.append(('most negative bucket', neg, 0.0, neg < -1e-12,
                     'a negative mass is frequency truncation ringing'))
        budget = 1.0 - self.bucket_sizing_p
        om = self.outside_mass
        if self._x_min_fixed is None:
            if om > budget:
                log2, bs = self.recommended_grid()
                x0 = self.grid_window(log2, self.bucket_sizing_p)[2]
                covered = sum(self._outside_at(log2, bs, x0)) <= budget
                advice = (f'use hints{{log2={log2}; bs={bs:g}}}' if covered else
                          f'coverage needs log2 above {RECOMMENDED_LOG2_MAX}; '
                          f'log2={log2} with bs={bs:g} is the most this engine offers')
            elif tilted:
                advice = 'the mass the window leaves out, below and above'
            else:
                advice = 'the mass outside the window, which wraps in (a Chernoff bound on a light tail)'
            rows.append(('outside mass', om, budget, om > budget, advice))
        if self.family.power_tails:
            rows.append(('edge error', self.edge_error, EDGE_ERROR_FLAG,
                         self.edge_error > EDGE_ERROR_FLAG,
                         'modeled worst relative bucket error: damped wrap plus amplified roundoff'
                         if tilted else 'the relative wrap at the right edge of an untilted grid'))
            rows.append(('tilt', max(self.tilt), np.nan, False,
                         f'e-folds across the window, (left, right) = '
                         f'({self.tilt[0]:.3g}, {self.tilt[1]:.3g}); theta = '
                         f'({self.tilt_theta[0]:.3g}, {self.tilt_theta[1]:.3g})'))
            if self.transfer_sigma:
                rows.append(('transfer sigma', self.transfer_sigma, np.nan, False,
                             'sigma of the normal moved between the halves'))
        rows.append(('truncation error', self.truncation_error, TRUNCATION_FLAG,
                     self.truncation_error > TRUNCATION_FLAG,
                     '|phi(pi / bs)|, the characteristic function at the highest frequency kept'))
        scale = self.family.scale
        per = scale / self.bs if np.isfinite(scale) else np.nan
        rows.append(('buckets per scale', per, MIN_BUCKETS_PER_SCALE,
                     np.isfinite(per) and per < MIN_BUCKETS_PER_SCALE,
                     'body scale over bs; a coarse grid loses the body'))
        st = self.stats_df
        for measure in ('mean', 'cv', 'skew'):
            t, g, e = (float(st.loc[('agg', measure), c]) for c in ('mixed', 'empirical', 'error'))
            if np.isfinite(t):
                rows.append((f'{measure} rel error', e, eps, abs(e) > eps,
                             f'grid {measure} against the theoretical'))
            else:
                rows.append((f'{measure}', g, np.nan, False,
                             'not finite: window statistic'))
        return rows

    def _checks_frame(self):
        """The checks on the inversion and the moments, one row each: ``value``, ``target``, ``status``, ``note``.

        Private: the frame the checks of :meth:`_validation_rows` make, which
        ``info`` and :attr:`validation_explanation` write out in words.
        """
        rows = self._validation_rows()
        return pd.DataFrame(
            {'value': [r[1] for r in rows], 'target': [r[2] for r in rows],
             'status': ['flag' if r[3] else 'ok' for r in rows],
             'note': [r[4] for r in rows]},
            index=pd.Index([r[0] for r in rows], name='check'))

    def _checks_text(self):
        """``8 ok`` or ``7 ok, 1 flag: outside mass, use hints{...}``: the inversion checks counted, each flag named."""
        rows = [r for r in self._validation_rows() if not r[0].startswith(('mean', 'cv', 'skew'))]
        n_ok = sum(1 for r in rows if not r[3])
        flags = [r for r in rows if r[3]]
        out = f'{n_ok} ok'
        if flags:
            named = [f'{r[0]}, {r[4]}' if r[4].startswith(('use hints', 'coverage needs'))
                     else r[0] for r in flags]
            out += f", {len(flags)} flag{'s' if len(flags) > 1 else ''}: {'; '.join(named)}"
        return out

    @property
    def validation_df(self):
        """Moment validation in ``Aggregate.validation_df``'s shape: rows ``Freq``, ``Sev``, ``Agg``.

        Before :meth:`update` the columns are ``EX | CV | Sk``, the law's;
        after it ``EX | Est EX | Err EX | CV | Est CV | Err CV | Sk | Est Sk``,
        ``Err`` the noise-aware relative error of the grid's value against the
        law's, ``nan`` where the law's is not finite (a window statistic). The
        ``SD`` trio replaces ``CV`` for a signed law, one whose support reaches
        below 0, as ``Aggregate._signed`` decides it. The ``Freq`` and ``Sev``
        rows are ``nan``. Read off :attr:`stats_df`; the checks on the
        inversion itself (coverage, edge error, truncation, resolution) are
        rows of :attr:`info`.
        """
        st, emp = self.stats_df['mixed'], self.stats_df['empirical']
        spread = 'SD' if self.family.support[0] < 0 else 'CV'
        nan2 = [np.nan, np.nan]
        th_spread = self.actual_sd if spread == 'SD' else st[('agg', 'cv')]
        df = pd.DataFrame({'EX': [*nan2, st[('agg', 'mean')]],
                           spread: [*nan2, th_spread],
                           'Sk': [*nan2, st[('agg', 'skew')]]},
                          index=['Freq', 'Sev', 'Agg'], dtype=float)
        df.index.name = 'X'
        if self.agg_density is not None:
            est = {'EX': emp[('agg', 'mean')], 'CV': emp[('agg', 'cv')],
                   'SD': self.est_sd, 'Sk': emp[('agg', 'skew')]}
            for c in ('EX', spread, 'Sk'):
                df[f'Est {c}'] = [*nan2, est[c]]
            for c in ('EX', spread):
                th, e = df[c], df[f'Est {c}']
                ok = np.isfinite(th) & np.isfinite(e)
                err = pd.Series(np.nan, index=df.index)
                err[ok] = _noise_aware_rel_error(e[ok], th[ok])
                df[f'Err {c}'] = err
            df = df[['EX', 'Est EX', 'Err EX', spread, f'Est {spread}',
                     f'Err {spread}', 'Sk', 'Est Sk']]
        for c in df.columns:
            if not c.startswith('Err '):
                df[c] = _snap_noise(df[c])
        return df

    @property
    def valid(self):
        """The :class:`~aggregate.constants.Validation` flags of the grid; see :func:`aggregate._validation.valid_characteristic_function`."""
        from . import _validation
        return _validation.valid_characteristic_function(self)

    @property
    def validation_score(self):
        """Continuous validation score in units of the tolerance, ``Aggregate.validation_score``'s (float).

        The aggregate mean, CV and skewness errors over their tolerances,
        combined as a power mean with ``power=2``; the severity terms are
        ``nan`` and drop out. See
        :func:`~aggregate._validation.validation_score`.
        """
        from . import _validation
        return _validation.validation_score(self)

    @property
    def validation_description(self):
        """One-line verdict in ``Aggregate``'s words: ``not unreasonable``, ``fails agg mean``, ``n/a, not updated``."""
        from ._validation import explain_validation
        rv = self.valid
        return explain_validation(rv, self._deficit, subject='law')

    def _completed_mean(self):
        r"""``(tail_mean, completed)``: the window's mean completed by the closed-form tails it leaves out, or ``None``.

        Only for a tilted grid of a law with a power tail and a mean,
        :math:`\alpha > 1`.

        Notes
        -----
        The window sum leaves the mass above it out and puts the mass below
        it at the first grid point :math:`x_0`. With
        :math:`\Pr(X > x) \sim A(x - \delta_1)^{-\alpha}` above,
        :math:`\mathsf{P}[(X - \delta_1)1_{X > u}] = A\alpha d^{1-\alpha}/(\alpha - 1)`,
        :math:`d = u - \delta_1`, at the top bucket edge :math:`u`; below, the
        same with :math:`B` and :math:`d = \delta_1 - l` at the bottom edge
        :math:`l`, entering with the opposite sign, less the :math:`x_0` the
        grid already counted for that mass. The completed mean is the window
        mean plus both corrections; ``[CHF-Tail-Completion]`` in
        ``dev/TODO.md`` carries the same tails into ``tvar`` and ``lev``.
        """
        fam = self.family
        if self.agg_density is None or not fam.power_tails or max(self.tilt) == 0:
            return None
        A, B, alpha = fam.power_tails
        if not alpha > 1:
            return None
        d1, h = fam.location, self.bs
        upper = lower = 0.0
        if A > 0 and self.outside_above > 0:
            top = self.x_max + h / 2
            if fam.tilt_side == 'right':
                # one-sided: the series, as outside_above is
                beyond = fam.tail_mean_above(top)
            else:
                beyond = A * alpha * (top - d1) ** (1 - alpha) / (alpha - 1)
            upper = d1 * self.outside_above + beyond
        if B > 0 and self.outside_below > 0:
            bottom = self.x_min - h / 2
            if fam.tilt_side == 'left':
                beyond = fam.reflected().tail_mean_above(-bottom)
            else:
                beyond = B * alpha * (d1 - bottom) ** (1 - alpha) / (alpha - 1)
            lower = d1 * self.outside_below - beyond - self.x_min * self.outside_below
        return upper + lower, self.est_m + upper + lower

    @property
    def validation_explanation(self):
        """The verbose form of :attr:`validation_description`: ``Aggregate``'s opening, then the inversion's checks in words."""
        from . import _validation
        from .constants import Validation
        rv = self.valid
        if rv & Validation.NOT_UPDATED:
            return ('Not validated: the object has not been updated, so there is no '
                    'realized grid to compare the analytic moments against. Call '
                    'update() (or build with update=True).')
        out = _validation.validation_opening(self.validation_description,
                                             self.validation_eps, 'the aggregate')
        out.append("A law computed from its characteristic function has no frequency "
                   "or severity, so the aggregate is its one component and the "
                   "theoretical moments are the law's own, from its cumulants.")
        rows = {r[0]: r for r in self._validation_rows()}
        budget = 1.0 - self.bucket_sizing_p
        if rv & Validation.INFEASIBLE:
            log2, bs = self.recommended_grid()
            out.append(f'No grid up to log2 {RECOMMENDED_LOG2_MAX} covers this law at '
                       f'the budget while keeping the truncation and the resolution: '
                       f'log2={log2} with bs={bs:g} is the most this engine offers, '
                       f'so the failures here are the price of the law, not of the '
                       f'grid chosen.')
        if rv & Validation.DEFECTIVE:
            out.append(f'The realized masses sum to {self.total_mass:.6g}: the window '
                       f'leaves {self._deficit:.3e} of the probability out above it, '
                       f'the closed-form tail mass, rather than wrapping it in. The '
                       f'cumulative masses are right throughout the window. Raise '
                       f'log2 to widen it.')
        if rv & Validation.ALIASING:
            out.append(f'The grid is untilted, so the mass outside the window wraps '
                       f'onto it: {self.outside_mass:.3g} outside against a budget of '
                       f'{budget:g}, edge error {self.edge_error:.3g}. Raise log2, or '
                       f'let a heavy-tailed law tilt (the default, hints{{tilt=True}}).')
        if rv & Validation.TRUNCATION:
            out.append(f'The characteristic function at the highest frequency kept, '
                       f'pi / bs, is {self.truncation_error:.3g} against '
                       f'{TRUNCATION_FLAG:g}, so the masses ring at the scale of the '
                       f'bucket. Lower bs.')
        window = (f'The window from {self.x_min:,.6g} to {self.x_max:,.6g} holds '
                  f'{self.total_mass:.6g} of the mass; {self.outside_mass:.3g} lies '
                  f'outside it, {self.outside_below:.3g} below and '
                  f'{self.outside_above:.3g} above, against a budget of {budget:g}')
        if 'outside mass' in rows and rows['outside mass'][3]:
            window += f': {rows["outside mass"][4]}'
        out.append(window + '.')
        if self.family.power_tails:
            if max(self.tilt) > 0:
                out.append(f'The heavy tail is tilted by ({self.tilt[0]:.3g}, '
                           f'{self.tilt[1]:.3g}) e-folds across the window, theta '
                           f'({self.tilt_theta[0]:.3g}, {self.tilt_theta[1]:.3g}) per '
                           f'unit of loss, with a modeled edge error of '
                           f'{self.edge_error:.3g} against {EDGE_ERROR_FLAG:g}.')
            else:
                out.append(f'The heavy tail is not tilted; the relative wrap at the '
                           f'edge is {self.edge_error:.3g} against {EDGE_ERROR_FLAG:g}.')
        if self.transfer_sigma:
            out.append(f'The two halves of the split trade a normal law with sigma '
                       f"{self.transfer_sigma:.3g}, which keeps the narrow half's "
                       f'truncation at roundoff and leaves the product unchanged.')
        if not rv & Validation.TRUNCATION:
            out.append(f'The truncation, |phi(pi / bs)| of the part transformed, is '
                       f'{self.truncation_error:.3g} against {TRUNCATION_FLAG:g}.')
        scale = self.family.scale
        if np.isfinite(scale):
            out.append(f'The body scale {scale:,.4g} spans {scale / self.bs:,.3g} '
                       f'buckets, against at least {MIN_BUCKETS_PER_SCALE}; the most '
                       f'negative bucket is {float(np.min(self.agg_density)):.3g}.')
        done = self._completed_mean()
        if done is not None:
            tail, completed = done
            out.append(f'The law has a power tail, so the window mean '
                       f'{self.est_m:,.6g} omits the tail beyond it, '
                       f'{tail:,.6g} in closed form; the completed mean '
                       f'{completed:,.6g} against the theoretical {self.actual_m:,.6g} '
                       f'is the check the window cannot make.')
        if rv == Validation.NOT_UNREASONABLE:
            out.append('Nothing failed, so the object is not unreasonable. That is '
                       'a statement about the numerics reproducing the model, not '
                       'about the model being right for the risk.')
        return ' '.join(out)

    # ---- sharpen ----------------------------------------------------------
    def _sharpen_terms(self):
        """The score terms of the current grid: ``validation_score_terms``'s six, then the four of the inversion.

        ``u_agg_mean`` is dropped (``nan``) for a law with a power tail, whose
        mean shortfall is the window's tail; the severity terms are ``nan``.
        """
        from . import _validation
        terms = _validation.validation_score_terms(self)
        fam = self.family
        if fam.power_tails:
            terms['u_agg_mean'] = np.nan
        budget = 1.0 - self.bucket_sizing_p
        scale = fam.scale
        terms['u_coverage'] = self.outside_mass / budget
        terms['u_edge'] = (self.edge_error / EDGE_ERROR_FLAG
                           if fam.power_tails and np.isfinite(self.edge_error) else np.nan)
        terms['u_truncation'] = self.truncation_error / TRUNCATION_FLAG
        terms['u_resolution'] = (MIN_BUCKETS_PER_SCALE * self.bs / scale
                                 if np.isfinite(scale) and scale > 0 else np.nan)
        return terms

    def _sharpen_row(self, bs0, d_log2, power, seconds, note='', caught=()):
        """One ``sharpen_df`` row for the grid the object is on."""
        from . import _validation
        terms = self._sharpen_terms()
        row = {'d_bs': math.log2(self.bs / bs0), 'd_log2': d_log2, 'bs': float(self.bs),
               'log2': int(self.log2), 'extent': float(self.bs) * (1 << int(self.log2)),
               'x_min': float(self.x_min),
               'score': _validation.combine_score_terms(terms, power)}
        row.update({k: terms[k] for k in _SHARPEN_MOMENT_TERMS})
        row['aliasing'] = np.nan
        row['deficit'] = _validation.pmf_deficit(self)
        # the soundness gate's analog: the outside mass over the coverage budget
        row['defective'] = bool(terms['u_coverage'] > 1)
        row['validation'] = self.validation_description
        row['warnings'] = len(caught)
        row['warns'] = ','.join(sorted({type(w.message).__name__ for w in caught}))
        row['seconds'] = seconds
        row['selected'] = False
        row['note'] = note
        row.update({k: terms[k] for k in _SHARPEN_INVERSION_TERMS})
        return row

    def sharpen(self, bs=None, log2=None, *, log2_cap=20, bs_limit=16, power=2,
                good_enough=0.5, execute=True):
        r"""Audit the grid and move toward :meth:`recommended_grid`, ``Aggregate.sharpen``'s analog.

        Parameters
        ----------
        bs, log2 : float, int, optional
            The center of the probe; default the current grid. An object not
            yet updated is updated first.
        log2_cap : int, default 20
            ``log2`` is never grown beyond this.
        bs_limit : float, default 16
            Accepted for ``Aggregate.sharpen``'s signature and inert: there is
            no bucket walk here (Notes).
        power : float, default 2
            Exponent of the power mean that combines the score terms.
        good_enough : float, default 0.5
            The target score. A sound center at or under it is confirmed and
            nothing runs; ``good_enough=0`` forces the probe.
        execute : bool, default True
            ``True`` leaves the object on the chosen cell; ``False`` restores
            the center.

        Returns
        -------
        AggregateCHF
            ``self``, updated in place. :attr:`program`, :attr:`note` and
            :attr:`hints` record the outcome as on ``Aggregate``: a moved grid
            as ``hints{log2=...; bs=...}``, a confirmed one as
            ``note{sharpen: grid confirmed, no change}``.

        Notes
        -----
        **Why not Aggregate's probe.** ``Aggregate.sharpen`` walks the bucket
        while :attr:`validation_score` improves. On a law with a power tail
        the only live moment term is the mean, whose shortfall is the tail
        beyond the window and shrinks as the window widens whatever the body
        costs: measured on ``2 * stable 1.5 1 + 10`` at log2 16, every doubling
        of ``bs`` from 1/16 to 1 improves it, and the walk lands on two buckets
        per scale with an edge error of :math:`10^4`.

        **The score.** In tolerance units, the power mean of the live terms:
        ``u_agg_mean``, ``u_agg_cv`` and ``u_agg_skew`` from
        :func:`~aggregate._validation.validation_score_terms` (live by
        ``Aggregate``'s rule, except ``u_agg_mean`` for a power tail, whose
        shortfall no grid under the cap removes); ``u_coverage``, the outside
        mass over ``1 - bucket_sizing_p``; ``u_edge``, the edge error over
        :data:`EDGE_ERROR_FLAG` for a power tail; ``u_truncation``, the
        truncation over :data:`TRUNCATION_FLAG`; ``u_resolution``,
        :data:`MIN_BUCKETS_PER_SCALE` buckets per scale over the realized
        count.

        **The cells.** The center and the :meth:`recommended_grid` cell, its
        ``log2`` capped at the center's plus one and at ``log2_cap``, so a call
        grows the grid by at most one ``log2`` and a re-run continues; a
        recommended ``log2`` below the center's is taken as it is. Each cell
        is a real :meth:`update`. The pick is
        ``aggregate._bucket_window._pick_cell``'s: sound first, then thrifty,
        then the score, where a cell is sound when its outside mass is within
        the budget (``defective`` is ``u_coverage > 1``): a tilted grid
        always leaves some mass out, so ``Aggregate``'s test, a deficit above
        1e-12, would call every one unsound.
        """
        from . import _program
        from ._bucket_window import _pick_cell, _sharpen_frame
        import time
        import warnings as _warnings
        if self.agg_density is None:
            self.update(log2=log2 or 16, bs=bs or 0)
        p_sizing, tilt = self.bucket_sizing_p, self.tilt_setting
        if (bs and float(bs) != self.bs) or (log2 and int(log2) != self.log2):
            self.update(log2=int(log2 or self.log2), bs=float(bs or self.bs),
                        bucket_sizing_p=p_sizing, tilt=tilt)
        bs0, log20 = float(self.bs), int(self.log2)
        rows = [self._sharpen_row(bs0, 0, power, 0.0, note='center (current grid)')]
        center_score = rows[0]['score']
        state = {'power': power, 'good_enough': good_enough, 'execute': execute,
                 'center_score': center_score, 'bs0': bs0, 'log20': log20,
                 'bs_exact': False, 'lattice': None, 'grew': False,
                 'center_defective': rows[0]['defective'], 'passed_over': 0}
        if (np.isfinite(center_score) and center_score <= good_enough
                and not rows[0]['defective']):
            rows[0]['selected'] = True
            rows[0]['note'] = ('center (current grid); at or under target and sound, '
                               'probe not run')
            self._sharpen_df = _sharpen_frame(rows)
            self._sharpen_state = {**state, 'ran': False, 'moved': False}
            _program.pin_sharpen(self)
            return self
        log2_r, bs_r = self.recommended_grid()
        log2_c = min(log2_r, log20 + 1, int(log2_cap))
        bs_c = bs_r if log2_c == log2_r else self.grid_window(log2_c, p_sizing, tilt)[0]
        note = 'recommended grid' if log2_c == log2_r else (
            f'toward the recommended grid, log2 {log2_r}, capped at {log2_c}')
        if (bs_c, log2_c) != (bs0, log20):
            t0 = time.perf_counter()
            try:
                with _warnings.catch_warnings(record=True) as caught:
                    _warnings.simplefilter('always')
                    self.update(log2=log2_c, bs=bs_c, bucket_sizing_p=p_sizing, tilt=tilt)
                rows.append(self._sharpen_row(bs0, log2_c - log20, power,
                                              time.perf_counter() - t0, note, caught))
            except Exception as e:      # noqa: BLE001 a cell may legitimately fail
                logger.info('sharpen: cell (bs %s, log2 %s) failed: %s', bs_c, log2_c, e)
        df = pd.DataFrame(rows)
        pick = _pick_cell(df)
        pick = 0 if pick is None else pick
        df.loc[pick, 'selected'] = True
        moved = bool(pick != 0)
        if moved and execute:
            reason = 'the recommended cell scores better and was taken'
        elif moved:
            reason = 'the recommended cell scores better; not executed'
        else:
            reason = 'the center scores at least as well, so the grid was kept'
        target = df.loc[pick] if (moved and execute) else df.loc[0]
        if (float(self.bs), int(self.log2)) != (float(target['bs']), int(target['log2'])):
            self.update(log2=int(target['log2']), bs=float(target['bs']),
                        bucket_sizing_p=p_sizing, tilt=tilt)
        state['grew'] = bool(moved and df.loc[pick, 'log2'] > log20)
        state['passed_over'] = int(((df['score'] < df.loc[pick, 'score'])
                                    & df['defective'].astype(bool)).sum())
        self._sharpen_df = _sharpen_frame(df)
        self._sharpen_state = {**state, 'ran': True, 'moved': moved and execute,
                               'reason': reason, 'recommended': (log2_r, bs_r)}
        _program.pin_sharpen(self)
        return self

    @property
    def sharpen_df(self):
        """The last :meth:`sharpen` probe, ``Aggregate.sharpen_df``'s columns then ``u_coverage``, ``u_edge``, ``u_truncation`` and ``u_resolution``.

        Indexed by ``(d_bs, d_log2)``; ``d_bs`` is the base 2 logarithm of the
        bucket ratio to the center, not always an integer here, since
        :meth:`recommended_grid` moves along the ``round_bucket`` ladder. The
        severity terms and ``aliasing`` are ``nan``. ``None`` before
        :meth:`sharpen` runs.
        """
        df = getattr(self, '_sharpen_df', None)
        return None if df is None else df.copy()

    @property
    def sharpen_description(self):
        """One-line summary of the last :meth:`sharpen` probe."""
        from ._bucket_window import _fmt_bs, _move_phrase, _sharpen_fmt
        df = getattr(self, '_sharpen_df', None)
        st = getattr(self, '_sharpen_state', None)
        if df is None or st is None:
            return 'Sharpen: not run.'
        if not st['ran']:
            return (f'Sharpen: not run, the grid already scores '
                    f'{_sharpen_fmt(st["center_score"])} against a target of '
                    f'{st["good_enough"]:g} and its outside mass is within the budget.')
        win = df[df['selected']].iloc[0]
        head = (f'Sharpen: {len(df)} cells in {float(df["seconds"].sum()):.2f}s, best '
                f'score {_sharpen_fmt(win["score"])} vs '
                f'{_sharpen_fmt(st["center_score"])} at the center, target '
                f'{st["good_enough"]:g}.')
        if win['bs'] == st['bs0'] and int(win['log2']) == st['log20']:
            return f'{head} Kept bs {_fmt_bs(st["bs0"])}, log2 {st["log20"]}.'
        move = _move_phrase(st, win)
        if st['execute']:
            return f'{head} Moved: {move}.'
        return f'{head} Recommends {move}; not executed, grid restored.'

    @property
    def sharpen_explanation(self):
        """Verbose prose explaining the last :meth:`sharpen` probe, in a law's words."""
        from ._bucket_window import _fmt_bs, _sharpen_fmt
        df = getattr(self, '_sharpen_df', None)
        st = getattr(self, '_sharpen_state', None)
        if df is None or st is None:
            return 'Sharpen has not been run on this object.'
        center = df.iloc[0]
        live = [k for k in (*_SHARPEN_MOMENT_TERMS, *_SHARPEN_INVERSION_TERMS)
                if np.isfinite(center[k])]
        out = [self.sharpen_description,
               f'The score is in units of the tolerances: the power mean (power '
               f'{st["power"]:g}) of the live terms, here {", ".join(live)}. The moment '
               f'terms are the validation errors of the law over their tolerances; '
               f'u_coverage is the mass outside the window over the budget '
               f'{1 - self.bucket_sizing_p:g}, u_edge the modeled edge error over '
               f'{EDGE_ERROR_FLAG:g}, u_truncation the truncation over '
               f'{TRUNCATION_FLAG:g}, and u_resolution {MIN_BUCKETS_PER_SCALE} buckets '
               f'per body scale over the buckets the grid has.']
        if self.family.power_tails:
            out.append('The law has a power tail, so its mean term is not scored: '
                       'the window mean falls short by the tail beyond the window, '
                       'which no grid under the cap removes and the validation '
                       'explanation quantifies, and a bucket walk on it would run '
                       'to the coarsest bucket however much the body lost.')
        if not st['ran']:
            out.append(f'The center already meets the target of '
                       f'{st["good_enough"]:g} with its outside mass within the '
                       f'budget, so no other grid was tried.')
            return ' '.join(out)
        log2_r, bs_r = st['recommended']
        out.append(f'The probe compares the center, bs {_fmt_bs(st["bs0"])} and log2 '
                   f'{st["log20"]}, with the recommended grid, log2 {log2_r} and bs '
                   f'{_fmt_bs(bs_r)}, its log2 capped at one above the center so a '
                   f'call grows the grid by at most one; a re-run continues. A cell '
                   f'is sound when its outside mass is within the budget, and a sound '
                   f'cell is preferred, then the smaller grid, then the score: '
                   f'{st["reason"]}.')
        return ' '.join(out)

    @property
    def sharpen_program(self):
        """The program that rebuilds this law on the sharpened grid; ``''`` before :meth:`sharpen` (``Aggregate.sharpen_program``)."""
        from . import _program
        return _program.sharpen_program(self)

    # ---- display ----------------------------------------------------------
    def _spec_hash(self):
        """Display-only 8-hex id of the constructor's spec, as ``Aggregate._spec_hash``."""
        blob = json.dumps(self._spec, sort_keys=True, default=str)
        return hashlib.md5(blob.encode('utf-8')).hexdigest()[:8].upper()

    def _tail_info_rows(self):
        """The three tail lines of :attr:`info`, ``(label, value)``."""
        return [('frequency tail', INFO_NA), ('severity tail', INFO_NA),
                ('aggregate tail', INFO_NA)]

    @property
    def info(self):
        """Fixed-layout multi-line summary string, ``Aggregate.info``'s rows then the inversion's.

        ``Aggregate``'s rows in its order, ``n/a`` where a law with no count,
        severity, loss table, premium or cession has nothing to say; then the
        characteristic function block (family, parameters, support, scale,
        the mass on the grid and outside it, the tilt, the edge and
        truncation errors, the transfer sigma, the most negative bucket, the
        buckets per scale, and ``checks``, the inversion checks counted with
        each flag named); then the tail lines, ``bounded`` and ``id``. Every
        row is present before :meth:`update`, reading ``n/a`` where there is
        no grid.
        """
        fam = self.family
        updated = self.agg_density is not None
        lo, hi = fam.support
        na = INFO_NA

        def g3(x):
            return f'{x:.3g}' if updated and np.isfinite(x) else na

        bss = ((f'{self.bs:.6g}' if self.bs >= 1 else f'1/{1 / self.bs:,.0f}')
               if updated else na)
        rows = [
            ('aggregate object name', self.name),
            ('value_type', 'loss'),
            ('claim count', na),
            ('frequency distribution', na),
            ('severity distribution', na),
            ('loss table', na),
            ('approximate', 'exact'),
            ('bs', bss),
            ('log2', self.log2 if updated else na),
            ('padding', na),
            ('sev_calc', na),
            ('dsev_bucket', na),
            ('normalize', False if updated else na),
            ('x_min', f'{self.x_min:,.6g}' if updated else na),
            ('x_max', f'{self.x_max:,.6g}' if updated else na),
            ('premium', na),
            ('expected loss', f'{self.est_m:,.6g}' if updated else na),
            ('loss ratio', na),
            ('P(X=0)', f'{self.pmf(0.0):.6g}' if updated else na),
            ('validation_eps', get_settings().validation.eps),
            ('reinsurance', 'none'),
            ('occurrence reinsurance', 'no reinsurance'),
            ('aggregate reinsurance', 'no reinsurance'),
            ('validation', self.validation_description),
            ('family', fam.name),
            ('parameters', fam.to_decl() or na),
            ('support', f'{lo:,.6g} to {hi:,.6g}'),
            ('scale', f'{fam.scale:,.6g}' if np.isfinite(fam.scale) else na),
            ('total mass', f'{self.total_mass:.6g}' if updated else na),
            ('outside mass', f'{self.outside_mass:.3g}, below {self.outside_below:.3g}, '
             f'above {self.outside_above:.3g}' if updated else na),
            ('tilt', f'({self.tilt[0]:.3g}, {self.tilt[1]:.3g})' if updated else na),
            ('theta', f'({self.tilt_theta[0]:.3g}, {self.tilt_theta[1]:.3g})'
             if updated else na),
            ('edge error', g3(self.edge_error)),
            ('truncation error', g3(self.truncation_error)),
            ('transfer sigma', g3(self.transfer_sigma) if self.transfer_sigma else na),
            ('most negative bucket', g3(float(np.min(self.agg_density))) if updated else na),
            ('buckets per scale', g3(fam.scale / self.bs) if updated else na),
            ('checks', self._checks_text() if updated else na),
            *self._tail_info_rows(),
            ('bounded', self.bounded),
            ('id', self._spec_hash()),
        ]
        return '\n'.join(info_row(label, value) for label, value in rows)

    def _text_info_blob(self):
        """One-line plain-text intro, the twin of :meth:`_html_info_blob`."""
        s = [f'AggregateCHF object: {self.label}.',
             f'Family {self.family.to_decl() or self.family.name}.']
        if self.agg_density is not None:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{1 / self.bs:,.0f}'
            s.append(f'Updated with bucket size {bss} and log2 = {self.log2}.')
        s.append(f'Validation: {self.validation_description}.')
        return ' '.join(s)

    def _html_info_blob(self):
        """Short HTML intro: identity, family, grid and the validation result."""
        parts = [f'Family {self.family.to_decl() or self.family.name}.']
        if self.agg_density is not None:
            bss = f'{self.bs:.6g}' if self.bs >= 1 else f'1/{1 / self.bs:,.0f}'
            parts.append(f'Updated with bucket size {bss} and log2 = {self.log2}.')
        parts.append(f'Validation: {self.validation_description}.')
        return (f'<h3>AggregateCHF object: {self.label}</h3>\n'
                f'<p>{" ".join(parts)}</p>')

    def __repr__(self):
        grid = (f', log2 {self.log2}, bs {self.bs:g}' if self.agg_density is not None
                else ', not updated')
        return f'AggregateCHF({self.label!r}: {self.family.to_decl() or self.family!r}{grid})'

    def _repr_html_(self):
        fmt = lambda x: f'{x:,.5g}'
        return '\n'.join([self._html_info_blob(),
                          '<h4>Summary</h4>',
                          self.summary_df.to_html(float_format=fmt, na_rep='')])

    # ---- plot -------------------------------------------------------------
    def plot(self, figsize=(12, 3.6)):
        """Three panels: the density, its logarithm, and the survival function with the tail drawn in.

        The density is mass over ``bs``. The survival panel is on a log
        vertical axis; a power tail draws :math:`A x^{-\\alpha}` and a light
        tail the Chernoff bound :math:`\\exp\\inf_s(K(s) - sx)`.

        Parameters
        ----------
        figsize : tuple

        Returns
        -------
        matplotlib.figure.Figure
            Also stored on :attr:`figure`.
        """
        import matplotlib.pyplot as plt
        self._require_update()
        x, p, h = self.xs, self.agg_density, self.bs
        fig, axs = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        ax0, ax1, ax2 = axs
        ax0.plot(x, p / h, lw=1)
        ax0.set(xlabel='x', ylabel='density', title=self.label)
        pos = p > 0
        ax1.plot(x[pos], np.log10(p[pos] / h), lw=1)
        ax1.set(xlabel='x', ylabel='log10 density', title='log density')
        S = 1.0 - np.cumsum(p)
        okS = S > 0
        ax2.plot(x[okS], S[okS], lw=1, label='grid')
        fam = self.family
        xr = x[okS]
        if fam.power_tails and fam.power_tails[0] > 0:
            A, _, alpha = fam.power_tails
            xt = xr[xr > fam.location + fam.scale]
            if len(xt):
                ax2.plot(xt, A * (xt - fam.location) ** (-alpha), ls='--', lw=1,
                         label='power tail')
        elif fam.cgf_domain[1] > 0:
            xt = np.linspace(max(xr[0], float(self.est_m)), xr[-1], 40)
            s_max = fam.cgf_domain[1]
            scale = fam.scale if np.isfinite(fam.scale) else 1.0
            s = np.logspace(-9, 9, 217) / scale
            s = s[s < s_max]
            with np.errstate(all='ignore'):
                K = np.asarray(fam.cgf(s), dtype=float)
                bound = [float(np.nanmin(np.where(np.isfinite(K), K - si * s, np.inf)))
                         for si in xt]
            ax2.plot(xt, np.exp(bound), ls='--', lw=1, label='Chernoff bound')
        ax2.set(xlabel='x', ylabel='S(x)', yscale='log', title='survival')
        ax2.legend(loc='upper right')
        self.figure = fig
        return fig
