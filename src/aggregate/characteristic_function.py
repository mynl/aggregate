r"""Distributions computed by inverting a characteristic function.

``AggregateCHF`` computes a distribution on a grid by sampling its
characteristic function on the FFT's frequency grid and inverting with one
``irfft``, instead of discretizing a severity and applying a frequency
generating function. The families are infinitely divisible laws with
closed-form characteristic functions: the alpha-stable laws, the normal, gamma
and inverse Gaussian laws in this step, with the whole Tweedie family, the
normal inverse Gaussian, variance gamma and tempered stable laws to follow
(``dev/done/plan-characteristic-function-engine.md``,
``[Characteristic-Function-Engine]``).

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

import logging
import math

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.optimize import minimize_scalar
from scipy.special import gamma as gamma_function

from ._aggregate import (SUMMARY_PERCENTILES, _summary_pct_label,
                         return_period_frame)
from ._bucket_window import BUCKET_SIZING_P
from ._grid_distribution import GridDistribution
from ._help import HelpMixin
from ._labeled import LabeledMixin
from ._program import ProgramMixin
from .config import get_settings
from .constants import INFO_NA, info_row
from .moments import _noise_aware_rel_error, _snap_noise
from .stable import Stable
from .utilities import round_bucket

__all__ = ['AggregateCHF', 'Family', 'FamilyNormal', 'FamilyGamma',
           'FamilyInverseGaussian', 'FamilyStable', 'FamilyUser',
           'FAMILY_PARAMETERS', 'FAMILY_CV_PARAMETERS', 'make_family',
           'invert_characteristic_function', 'chernoff_edge']

logger = logging.getLogger(__name__)

#: Positional parameter names of each family's DecL body, in order. The
#: parser maps ``ID numberl`` onto these; the class takes them as keywords.
FAMILY_PARAMETERS = {
    'stable': ('alpha', 'beta', 'scale', 'loc'),
    'norm': ('mean', 'sd'),
    'gamma': ('shape', 'scale'),
    'invgauss': ('mean', 'shape'),
}

#: The ``mean cv`` form of the families that have one.
FAMILY_CV_PARAMETERS = {
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


# ======================================================================
# families
# ======================================================================

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

    def to_decl(self):
        """The DecL body as written, ``family`` first."""
        raise NotImplementedError

    def __repr__(self):
        return f'{type(self).__name__}({self.to_decl()!r})'

    # ---- helpers shared by the two-form families -----------------------
    def _positional_decl(self):
        """``name p1 p2 ...`` in the family's positional order, or the cv form."""
        if self.form == 'cv':
            a, b = FAMILY_CV_PARAMETERS[self.name]
            return f'{self.name} {self.parameters[a]:g} cv {self.parameters[b]:g}'
        parts = [f'{self.parameters[k]:g}' for k in FAMILY_PARAMETERS[self.name]
                 if k in self.parameters]
        return ' '.join([self.name, *parts])


class FamilyNormal(Family):
    r"""The normal law, ``norm mean sd`` or ``norm mean cv c``.

    :math:`\varphi(s) = \exp(i\mu s - \sigma^2 s^2/2)`, entire;
    :math:`K(s) = \mu s + \sigma^2 s^2/2` on the whole line.
    """

    name = 'norm'

    def __init__(self, parameters, form=None):
        super().__init__(parameters, form)
        p = self.parameters
        self.mean = float(p['mean'])
        self.sd = float(self.mean * p['cv']) if form == 'cv' else float(p['sd'])
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

    def to_decl(self):
        return self._positional_decl()


class FamilyGamma(Family):
    r"""The gamma law, ``gamma shape scale`` or ``gamma mean cv c``.

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
            self.shape, self.theta = float(p['shape']), float(p['scale'])
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

    def to_decl(self):
        return self._positional_decl()


class FamilyInverseGaussian(Family):
    r"""The inverse Gaussian law, ``invgauss mean shape`` or ``invgauss mean cv c``.

    With mean :math:`\mu` and shape :math:`\lambda`,
    :math:`K(s) = (\lambda/\mu)\big(1 - \sqrt{1 - 2\mu^2 s/\lambda}\big)` for
    :math:`s < \lambda/(2\mu^2)` and :math:`\varphi(s) = \exp K(is)` on the
    principal square root, whose argument has positive real part on the
    upper half plane. Variance :math:`\mu^3/\lambda`, skewness
    :math:`3\sqrt{\mu/\lambda}`. The scipy law is
    ``invgauss(mu / lam, scale=lam)``.
    """

    name = 'invgauss'
    support = (0.0, np.inf)

    def __init__(self, parameters, form=None):
        super().__init__(parameters, form)
        p = self.parameters
        self.mean = float(p['mean'])
        if form == 'cv':
            self.lam = self.mean / float(p['cv']) ** 2
        else:
            self.lam = float(p['shape'])
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

    def to_decl(self):
        return self._positional_decl()


class FamilyStable(Family):
    r"""The alpha-stable law, ``stable alpha beta [scale [loc]] [s0|s1]``.

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
        s = np.asarray(s, dtype=complex)
        if np.any(s.imag != 0):
            raise NotImplementedError(
                'the stable characteristic function off the real line is '
                '[Characteristic-Function-Tilting]')
        return self.stable.chf(s.real)

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

    def to_decl(self):
        p = self.parameters
        parts = ['stable', f'{p["alpha"]:g}', f'{p["beta"]:g}']
        if 'scale' in p:
            parts.append(f'{p["scale"]:g}')
            if 'loc' in p:
                parts.append(f'{p["loc"]:g}')
        if str(p.get('parameterization', 'S1')).upper() == 'S0':
            parts.append('s0')
        return ' '.join(parts)


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


#: Family word to class.
FAMILIES = {
    'stable': FamilyStable,
    'norm': FamilyNormal,
    'gamma': FamilyGamma,
    'invgauss': FamilyInverseGaussian,
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
    family : str
        The family word: ``'stable'``, ``'norm'``, ``'gamma'`` or
        ``'invgauss'`` (:data:`FAMILIES`).
    form : str or None
        ``'cv'`` for the ``mean cv`` form of a two-form family.
    note, tags, hints, label, label_map
        The DecL trailer, as on every first-class class.
    **parameters
        The family's parameters as written, keyed by name
        (:data:`FAMILY_PARAMETERS`, :data:`FAMILY_CV_PARAMETERS`).

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
    relative error of order one there whatever its total; the shifted
    contour that cures it is ``[Characteristic-Function-Tilting]``.

    Examples
    --------
    >>> from aggregate.characteristic_function import AggregateCHF
    >>> a = AggregateCHF('G', 'gamma', shape=4, scale=25)
    >>> a.update(log2=12)
    >>> round(a.est_m, 6), round(a.agg_m, 6)
    (100.0, 100.0)
    """

    def __init__(self, name, family, form=None, note='', tags=(), hints='',
                 label=None, label_map=None, **parameters):
        self.name = name
        if isinstance(family, Family):
            self.family = family
        else:
            self.family = make_family(family, form, **parameters)
        self._init_labels(label=label, label_map=label_map)
        self.note = note
        self.tags = tuple(tags) if tags else ()
        self.hints = hints
        #: an explicit grid left edge, set by :meth:`from_chf`; ``None`` means
        #: the left edge comes from the window rule
        self._x_min_fixed = None
        self._reset_grid()

    def _reset_grid(self):
        self.log2 = 0
        self.bs = 0.0
        self.x_min = None
        self.bucket_sizing_p = BUCKET_SIZING_P
        self.xs = None
        self.agg_density = None
        self.truncation_error = np.nan
        self._gd = None
        self._density_df = None
        self.figure = None

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
    def grid_window(self, log2=16, bucket_sizing_p=BUCKET_SIZING_P, bs=0):
        """The grid the window rule would choose, without computing.

        Parameters
        ----------
        log2 : int
        bucket_sizing_p : float
            The mass the window must cover (rule A).
        bs : float, default 0
            A given bucket (rule C); 0 means rule A.

        Returns
        -------
        tuple
            ``(bs, log2, x_min)``.
        """
        n = 2 ** int(log2)
        if self._x_min_fixed is not None:
            h = bs or self._fixed_grid[0]
            x0 = self._x_min_fixed
            return float(h), int(log2), float(x0)
        eps = 1.0 - bucket_sizing_p
        if not 0 < eps < 1:
            raise ValueError(f'bucket_sizing_p must be in (0, 1), got {bucket_sizing_p}')
        x_l, x_r = window_edges(self.family, eps / 2)
        if not (np.isfinite(x_l) and np.isfinite(x_r)) or x_r <= x_l:
            raise ValueError(
                f'{self.name}: cannot size a window for {self.family!r}; '
                'give bs and log2')
        h = float(bs) if bs else round_bucket((x_r - x_l) / n)
        # round_bucket rounds up by as much as 2x, so the window is wider than
        # the edges need; the slack is split between the sides, cut at a
        # finite support bound, so that neither tail's wrap lands on the far
        # end of the other side. 0 stays a grid point.
        slack = max(n * h - (x_r - x_l), 0.0)
        x_left = max(self.family.support[0], x_l - slack / 2)
        x0 = h * math.floor(x_left / h + 1e-12)
        return float(h), int(log2), float(x0)

    def recommended_grid(self, epsilon=None):
        """The smallest ``(log2, bs)`` whose window covers mass ``epsilon`` beyond it at full resolution.

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
        Rule A always covers, so what grows ``log2`` is resolution: the
        bucket must give at least :data:`MIN_BUCKETS_PER_SCALE` buckets per
        body scale (:attr:`Family.scale`). The bucket cap of a tilted law is
        ``[Characteristic-Function-Tilting]``.
        """
        if self._x_min_fixed is not None:
            return self._fixed_grid[1], self._fixed_grid[0]
        p = 1.0 - (epsilon if epsilon is not None else 1.0 - self.bucket_sizing_p)
        scale = self.family.scale
        for log2 in range(8, RECOMMENDED_LOG2_MAX + 1):
            h, _, _ = self.grid_window(log2, p)
            if not np.isfinite(scale) or scale / h >= MIN_BUCKETS_PER_SCALE:
                return log2, h
        return RECOMMENDED_LOG2_MAX, h

    def update(self, log2=None, bs=0, bucket_sizing_p=BUCKET_SIZING_P, **kwargs):
        """Compute the bucket masses on the grid the window rule chooses.

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
        **kwargs
            Other ``hints{}`` keys reach here and are ignored with a logged
            warning, as ``Aggregate.update`` ignores keys it does not use.
        """
        if kwargs:
            logger.warning('%s.update ignores %s', type(self).__name__,
                           ', '.join(sorted(kwargs)))
        if self._x_min_fixed is not None:
            bs = bs or self._fixed_grid[0]
            log2 = log2 or self._fixed_grid[1]
        log2 = 16 if log2 is None else int(log2)
        h, log2, x0 = self.grid_window(log2, bucket_sizing_p, bs)
        n = 2 ** log2
        p = invert_characteristic_function(self.family.chf, x0, h, n, 0.0,
                                           self.family.atom)
        self.log2 = log2
        self.bs = h
        self.x_min = x0
        self.bucket_sizing_p = bucket_sizing_p
        self.xs = x0 + h * np.arange(n)
        self.agg_density = p
        self.truncation_error = float(abs(self.family.chf(np.array([np.pi / h]))[0]))
        self._gd = None
        self._density_df = None

    @property
    def x_max(self):
        """The last grid point, or ``None`` before :meth:`update`."""
        return None if self.xs is None else float(self.xs[-1])

    # ---- the grid distribution and its accessors -------------------------
    def _require_update(self):
        if self.agg_density is None:
            raise ValueError(f'{self.name}: update() first')

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

    # ---- moments ----------------------------------------------------------
    @property
    def agg_m(self):
        """The theoretical mean, ``inf`` or ``nan`` where it does not exist."""
        return float(self.family.moments[0])

    @property
    def agg_cv(self):
        """The theoretical coefficient of variation, ``inf`` or ``nan`` where it does not exist."""
        mean, var, _ = self.family.moments
        if not np.isfinite(var):
            return float(var) if np.isfinite(mean) else np.nan
        if mean == 0:
            return np.nan
        return float(np.sqrt(var) / mean)

    @property
    def agg_skew(self):
        """The theoretical skewness, ``nan`` where it does not exist."""
        return float(self.family.moments[2])

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
    def est_cv(self):
        """The grid coefficient of variation, ``nan`` at mean zero."""
        m, sd, _ = self._grid_moments()
        return sd / m if m != 0 else np.nan

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
        """Moments and key percentiles, one ``Agg`` row with ``Aggregate.summary_df``'s columns."""
        updated = self.agg_density is not None
        if updated:
            m, sd, skew = self._grid_moments()
        else:
            m, var, skew = self.family.moments
            sd = np.sqrt(var) if np.isfinite(var) else np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = sd / m if (m != 0 and np.isfinite(m)) else np.nan
        df = pd.DataFrame({'Mean': [m], 'SD': [sd], 'CV': [cv], 'Skew': [skew]},
                          index=['Agg'])
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

    @property
    def stats_df(self):
        """Theoretical against grid moments, one block.

        Index ``mean``, ``sd``, ``cv``, ``skew``; columns ``theoretical``,
        ``grid``, ``rel error``. A theoretical moment that does not exist is
        ``inf`` or ``nan`` and its error is ``nan``: the grid value is then a
        window statistic.
        """
        mean, var, skew = self.family.moments
        sd = np.sqrt(var) if np.isfinite(var) else var
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = sd / mean if (np.isfinite(mean) and mean != 0) else np.nan
        th = [mean, sd, cv, skew]
        if self.agg_density is not None:
            m, s, k = self._grid_moments()
            with np.errstate(divide='ignore', invalid='ignore'):
                gr = [m, s, s / m if m != 0 else np.nan, k]
        else:
            gr = [np.nan] * 4
        err = [float(_noise_aware_rel_error(g, t))
               if np.isfinite(t) and np.isfinite(g) else np.nan
               for t, g in zip(th, gr)]
        df = pd.DataFrame({'theoretical': th, 'grid': gr, 'rel error': err},
                          index=pd.Index(['mean', 'sd', 'cv', 'skew'], name='measure'))
        return df

    def _validation_rows(self):
        """The validation checks as ``(name, value, target, flagged, note)`` tuples."""
        self._require_update()
        eps = get_settings().validation.eps
        rows = []
        total = self.total_mass
        rows.append(('total mass', total, 1.0, abs(total - 1.0) > 1e-6,
                     'sum of the masses on the grid against the mass the window should hold'))
        neg = float(np.min(self.agg_density))
        rows.append(('most negative bucket', neg, 0.0, neg < -1e-12,
                     'a negative mass is frequency truncation ringing'))
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
            t, g, e = st.loc[measure]
            if np.isfinite(t):
                rows.append((f'{measure} rel error', e, eps, abs(e) > eps,
                             f'grid {measure} against the theoretical'))
            else:
                rows.append((f'{measure}', g, np.nan, False,
                             'not finite: window statistic'))
        return rows

    @property
    def validation_df(self):
        """The validation checks, one row each.

        Columns ``value``, ``target``, ``status`` (``ok`` or ``flag``) and
        ``note``; rows total mass, most negative bucket, truncation error,
        buckets per scale, and each moment's relative error, or the grid
        value marked ``not finite: window statistic`` where the law has no
        such moment.
        """
        rows = self._validation_rows()
        df = pd.DataFrame(
            {'value': [r[1] for r in rows], 'target': [r[2] for r in rows],
             'status': ['flag' if r[3] else 'ok' for r in rows],
             'note': [r[4] for r in rows]},
            index=pd.Index([r[0] for r in rows], name='check'))
        return df

    @property
    def validation_description(self):
        """One-line verdict: ``not unreasonable`` or the flagged checks."""
        if self.agg_density is None:
            return 'not updated'
        flagged = [r[0] for r in self._validation_rows() if r[3]]
        return 'not unreasonable' if not flagged else 'flags ' + ', '.join(flagged)

    @property
    def validation_explanation(self):
        """The verbose form of :attr:`validation_description`, one sentence per check."""
        if self.agg_density is None:
            return 'The object has not been updated, so there is no grid to validate.'
        parts = []
        for name, value, target, flagged, note in self._validation_rows():
            verdict = 'flagged' if flagged else 'ok'
            tgt = '' if not np.isfinite(target) else f' against {target:g}'
            parts.append(f'{name} {value:.4g}{tgt}, {verdict}: {note}.')
        return ' '.join(parts)

    # ---- display ----------------------------------------------------------
    @property
    def info(self):
        """Fixed-layout multi-line summary string."""
        fam = self.family
        updated = self.agg_density is not None
        lo, hi = fam.support
        mean, var, _ = fam.moments
        rows = [
            ('chfagg object name', self.name),
            ('family', fam.name),
            ('parameters', fam.to_decl() or INFO_NA),
            ('support', f'{lo:,.6g} to {hi:,.6g}'),
            ('E[X]', f'{mean:,.6g}' if np.isfinite(mean) else str(mean)),
            ('scale', f'{fam.scale:,.6g}' if np.isfinite(fam.scale) else INFO_NA),
            ('bs', (f'{self.bs:,.6g}' if self.bs >= 1 else f'1/{1 / self.bs:,.0f}')
             if updated else INFO_NA),
            ('log2', self.log2 if updated else INFO_NA),
            ('x_min', f'{self.x_min:,.6g}' if updated else INFO_NA),
            ('x_max', f'{self.x_max:,.6g}' if updated else INFO_NA),
            ('total mass', f'{self.total_mass:.6g}' if updated else INFO_NA),
            ('expected loss', f'{self.est_m:,.6g}' if updated else INFO_NA),
            ('truncation error', f'{self.truncation_error:.3g}' if updated else INFO_NA),
            ('validation', self.validation_description),
            ('bounded', self.bounded),
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
