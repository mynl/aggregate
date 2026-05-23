"""
Distortion functions and spectral risk measures.

The module defines a registry-based ``Distortion`` class. Each named
distortion (``ph``, ``wang``, ``tvar``, ...) is a subclass that registers
itself on import via ``__init_subclass__`` and supplies its own ``g``,
``g_inv``, and (optionally) ``g_prime``. The factory call
``Distortion('ph', 0.9)`` dispatches on the name string and returns an
instance of the appropriate subclass; existing call sites are unchanged.
"""
from collections import namedtuple
from collections.abc import Iterable
from io import StringIO
import logging

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from scipy.stats import norm

logger = logging.getLogger(__name__)
phi = norm.cdf
phi_inv = norm.ppf

try:
    import numba
    njit = numba.njit
except ImportError:
    logger.info("Numba not found. Falling back to pure Python.")
    def njit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

from .constants import *
from .random_agg import RANDOM
from .utilities import short_hash


# ---------------------------------------------------------------------------
# Numba-compiled helpers for TVaR and BiTVaR (used by Distortion.quick_gS /
# .quick_ra). These compute g(1 - probs.cumsum()) or the risk-adjusted
# expectation in a single pass with no intermediate arrays.
# ---------------------------------------------------------------------------

@njit(parallel=False)
def tvar_gS(probs, p):
    """
    Compute ``gS`` for a TVaR distortion in one pass; equivalent to
    ``tvar.g(1 - probs.cumsum())``.

    Roughly 4x the speed of the array form. Cannot parallelize because of
    the shared running ``S`` variable.

    :param probs: numpy array of probabilities.
    :param p: float, distortion parameter in [0, 1].
    """
    S = 1.
    ans = np.zeros_like(probs)

    if p == 0:
        # mean
        for i in range(len(probs)):
            S -= probs[i]
            if S <= 0:
                ans[i] = 0
            elif S >= 1:
                ans[i] = 1
            else:
                ans[i] = S

    elif p == 1:
        for i in range(len(probs)):
            S -= probs[i]
            if S > 0:
                ans[i] = 1
            else:
                ans[i] = 0

    else:
        s = 1 - p
        m = 1 / s
        for i in range(len(probs)):
            S -= probs[i]
            if S < s:
                ans[i] = m * S
            else:
                ans[i] = 1
    return ans


@njit(parallel=False)
def bitvar_gS(probs, p0, p1, w):
    """
    Compute ``gS`` for a BiTVaR distortion in one pass; equivalent to
    ``bitvar.g(1 - probs.cumsum())``. Requires ``p0 < p1`` and
    ``w in (0, 1)``.

    See OneNote 2024/November Numba TVaR and BiTVaR.

    :param probs: numpy array of probabilities.
    :param p0: float, lower TVaR threshold.
    :param p1: float, upper TVaR threshold.
    :param w: float, weight on ``p1``.
    """
    if p0 == p1:
        return tvar_gS(probs, p0)
    if w == 0:
        return tvar_gS(probs, p0)
    if w == 1:
        return tvar_gS(probs, p1)

    s0 = 1 - p0
    s1 = 1 - p1
    S = 1.
    ans = np.zeros_like(probs)
    # height at kink
    pt = w + (1 - w) * s1 / s0

    if p0 == 0 and p1 == 1:
        slope = (1 - w)
        for i in range(len(probs)):
            S -= probs[i]
            # <=0 handles numerical cumulation error which can leave
            # S as a small negative
            if S <= 0:
                ans[i] = 0
            else:
                ans[i] = w + slope * S

    elif p0 > 0 and p1 == 1:
        slope = (1 - w) / s0
        for i in range(len(probs)):
            S -= probs[i]
            if S <= 0:
                ans[i] = 0
            elif S < s0:
                ans[i] = w + slope * S
            else:
                ans[i] = 1.

    elif p0 == 0 and p1 < 1:
        slope0 = pt / s1
        slope1 = (1 - pt) / p1
        for i in range(len(probs)):
            S -= probs[i]
            if S <= 0:
                ans[i] = 0
            elif S < s1:
                ans[i] = slope0 * S
            else:
                ans[i] = pt + slope1 * (S - s1)

    else:
        # p0 > 0 and p1 < 1: three kinks
        slope0 = pt / s1
        slope1 = (1 - pt) / (s0 - s1)
        for i in range(len(probs)):
            S -= probs[i]
            if S <= 0:
                ans[i] = 0
            elif S < s1:
                ans[i] = slope0 * S
            elif S < s0:
                ans[i] = pt + slope1 * (S - s1)
            else:
                ans[i] = 1.

    return ans


@njit(parallel=False)
def tvar_ra(probs, x, p):
    """
    Risk-adjusted expectation under a TVaR distortion, in one pass.

    Equivalent to ``(g.g(1 - probs.cumsum())[:-1] * np.diff(x)).sum() + x[0]``.
    Shortcuts ``p == 0`` (mean) and ``p == 1`` (max).

    :param probs: numpy array of probabilities.
    :param x: numpy array of loss outcomes, in ascending order.
    :param p: float, distortion parameter.
    """
    S = 1.
    lastx = x[0]
    ans = lastx

    if p == 0:
        ans = np.sum(probs * x)
    elif p == 1:
        ans = x[-1]
    else:
        s = 1 - p
        m = 1 / s
        for i in range(len(probs) - 1):
            S -= probs[i]
            dx = x[i + 1] - lastx
            lastx = x[i + 1]
            if S < s:
                ans += m * S * dx
            else:
                ans += dx
    return ans


@njit(parallel=False)
def bitvar_ra(probs, x, p0, p1, w):
    """
    Risk-adjusted expectation under a BiTVaR distortion, in one pass.

    Three of the four parameter regimes reduce to mean + max combinations
    or a single inner TVaR call; the remaining ``p0 > 0, p1 < 1`` case is
    computed directly via the three-kink g.

    :param probs: numpy array of probabilities.
    :param x: numpy array of loss outcomes, in ascending order.
    :param p0: float, lower TVaR threshold.
    :param p1: float, upper TVaR threshold.
    :param w: float, weight on ``p1``.
    """
    if p0 == p1:
        return tvar_ra(probs, x, p0)
    if w == 0:
        return tvar_ra(probs, x, p0)
    if w == 1:
        return tvar_ra(probs, x, p1)

    s0 = 1 - p0
    s1 = 1 - p1
    S = 1.
    lastx = x[0]
    ans = lastx
    pt = w + (1 - w) * s1 / s0

    if p0 == 0 and p1 == 1:
        m = np.sum(probs * x)
        ans = w * x[-1] + (1 - w) * m
    elif p0 > 0 and p1 == 1:
        m0 = tvar_ra(probs, x, p0)
        ans = w * x[-1] + (1 - w) * m0
    elif p0 == 0 and p1 < 1:
        m1 = tvar_ra(probs, x, p1)
        ans = w * m1 + (1 - w) * np.sum(probs * x)
    else:
        # p0 > 0 and p1 < 1: three kinks
        slope0 = pt / s1
        slope1 = (1 - pt) / (s0 - s1)
        for i in range(len(probs) - 1):
            S -= probs[i]
            dx = x[i + 1] - lastx
            lastx = x[i + 1]
            if S <= 0:
                pass
            elif S < s1:
                ans += (slope0 * S) * dx
            elif S < s0:
                ans += (pt + slope1 * (S - s1)) * dx
            else:
                ans += dx

    return ans


# ---------------------------------------------------------------------------
# Base Distortion class: registry, factory dispatch, and shared methods.
# ---------------------------------------------------------------------------

class Distortion:
    """
    Base class for distortion functions.

    Each concrete kind is a subclass declared below (``PHDistortion``,
    ``TVaRDistortion``, etc.). Subclasses register themselves by setting
    a class attribute ``kind`` (the lookup key) and are auto-collected by
    ``__init_subclass__``.

    The factory call ``Distortion('ph', 0.9)`` is the primary public API:
    it dispatches on the name string and returns an instance of the
    matching subclass. Direct subclass construction is supported but not
    part of the public API.

    Subclass contract:
        * ``kind: str`` — registry key, e.g. ``'ph'``.
        * ``med_name: str`` — short display label.
        * ``long_name: str`` — full display label.
        * ``documented: bool = True`` — included in ``available_distortions``.
        * ``pricing_ok: bool = True`` — included in pricing-only lists.
        * ``has_mass_default: bool = False`` — listed in ``_has_mass_``.
        * ``def _build(self): ...`` — set ``self.has_mass``, ``self.mass``,
          ``self.standard_shape`` (if applicable), ``self.display_name``
          (if blank), and override ``g``/``g_inv``/``g_prime`` if the
          default class methods aren't applicable.

    Note: to create a fake Distortion use a small synthetic class::

        g = type('Distortion', (),
                 {'g': your_g, 'g_inv': your_g_inv,
                  'g_dual': lambda self, x: 1 - your_g(1 - x)})()
    """

    # registry of subclasses keyed by ``kind``; populated by
    # ``__init_subclass__``. Order of insertion = declaration order = order
    # used by ``available_distortions``.
    _registry: dict[str, type] = {}

    # class-level attributes that subclasses override. The base values
    # here let unit tests/synthetic instances proceed without crashing.
    kind: str = ''
    med_name: str = ''
    long_name: str = ''
    documented: bool = False
    pricing_ok: bool = False
    # included in available_distortions(pricing=True, strict=True): single
    # shape parameter, no mass at zero, calibratable from a single number.
    strict_pricing: bool = False
    has_mass_default: bool = False

    # legacy attribute aliases preserved for back-compat with callers
    # that read class-level lists directly.
    @classmethod
    def _kinds_ordered(cls):
        return tuple(k for k, v in cls._registry.items() if v.documented)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.kind:
            Distortion._registry[cls.kind] = cls

    def __new__(cls, name=None, *args, **kwargs):
        """Factory dispatch: ``Distortion('ph', 0.9)`` → ``PHDistortion``.

        When called on a subclass directly, or with ``name=None`` (as
        happens during pickle reconstruction), no dispatch occurs.
        """
        if cls is not Distortion or name is None:
            return object.__new__(cls)
        # legacy alias: 'roe' was renamed to 'ccoc' in 0.9.4
        lookup = 'ccoc' if name == 'roe' else name
        subclass = cls._registry.get(lookup)
        if subclass is None:
            raise ValueError(
                f"Unknown distortion kind {name!r}; "
                f"available: {sorted(cls._registry)}")
        return object.__new__(subclass)

    def __init__(self, name, shape, r0=0.0, df=None, col_x='', col_y='',
                 display_name=''):
        """
        Create a distortion.

        :param name: name of an available distortion. Call
            ``Distortion.available_distortions()`` for a list.
        :param shape: float or sequence; meaning is kind-specific.
        :param r0: risk-free or rental rate of interest (used by kinds
            with a mass at zero).
        :param df: kind-specific second parameter — a number, list,
            or DataFrame.
        :param col_x: column of ``df`` used for x values (``convex``).
        :param col_y: column of ``df`` used for y values (``convex``).
        :param display_name: override label; ``str(d)`` returns this if
            set, else the kind name.
        """
        if name == 'roe':
            name = 'ccoc'
        self._name = name
        self.shape = shape
        self.r0 = r0
        self.df = df
        self.col_x = col_x
        self.col_y = col_y
        self.display_name = display_name

        # common defaults; subclass _build can override
        self.has_mass = False
        self.mass = 0.0
        self.standard_shape = np.nan
        self.error = 0.0
        self.premium_target = 0.0
        self.assets = 0.0

        self._build()

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def _build(self):
        """
        Subclass hook: configure kind-specific attributes (mass, derived
        shapes, display name) and override ``g``, ``g_inv``, ``g_prime``
        if the class-level methods aren't applicable.

        The base implementation is a no-op so that subclasses without a
        custom build (none, currently) still work.
        """
        return None

    # ------------------------------------------------------------------
    # g, g_inv, g_prime: subclasses override
    # ------------------------------------------------------------------

    def g(self, x):
        raise NotImplementedError(
            f"{type(self).__name__} must override g()")

    def g_inv(self, x):
        raise NotImplementedError(
            f"{type(self).__name__} must override g_inv()")

    def g_prime(self, x):
        """Default: central-difference numerical derivative of ``g``."""
        return (self.g(x + 1e-6) - self.g(x - 1e-6)) / 2e-6

    def g_dual(self, x):
        """The dual distortion ``1 - g(1 - x)``."""
        return 1 - self.g(1 - x)

    # ------------------------------------------------------------------
    # Class-level lists derived from the registry
    # ------------------------------------------------------------------

    @classmethod
    def available_distortions(cls, pricing=True, strict=True):
        """
        List of available distortions.

        :param pricing: only return kinds suitable for pricing
            (excludes ``convex`` and ``beta``).
        :param strict: only include kinds without a mass at zero
            (pricing only).
        """
        registry = cls._registry
        if pricing and strict:
            return tuple(k for k, v in registry.items()
                         if v.documented and v.strict_pricing
                         and not v.has_mass_default)
        elif pricing:
            return tuple(k for k, v in registry.items()
                         if v.documented and v.pricing_ok)
        else:
            return tuple(k for k, v in registry.items() if v.documented)

    # Legacy aliases (read by some callers; preserved for compat).
    @classmethod
    def _med_names(cls):
        return tuple(v.med_name for k, v in cls._registry.items()
                     if v.documented)

    @classmethod
    def _long_names(cls):
        return tuple(v.long_name for k, v in cls._registry.items()
                     if v.documented)

    @property
    def _kind_immutable(self):
        # back-compat shim for one external dispatch site
        return self._name

    # Provide the legacy class-level tuples as descriptors so existing
    # code like ``Distortion._available_distortions_`` keeps working.
    class _RegistryProperty:
        def __init__(self, fn):
            self.fn = fn

        def __get__(self, instance, owner):
            return self.fn(owner)

    _available_distortions_ = _RegistryProperty(
        lambda cls: tuple(k for k, v in cls._registry.items() if v.documented))
    _med_names_ = _RegistryProperty(
        lambda cls: tuple(v.med_name for k, v in cls._registry.items() if v.documented))
    _long_names_ = _RegistryProperty(
        lambda cls: tuple(v.long_name for k, v in cls._registry.items() if v.documented))
    _has_mass_ = _RegistryProperty(
        lambda cls: tuple(k for k, v in cls._registry.items()
                          if v.documented and v.has_mass_default))
    _distortion_names_ = _RegistryProperty(
        lambda cls: dict(zip(
            (k for k, v in cls._registry.items() if v.documented),
            (v.long_name for k, v in cls._registry.items() if v.documented),
        )))
    renamer = _distortion_names_

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def tvar_terms(p_in):
        """
        Evaluate the TVaR knot function ``min(s / (1-p), 1)`` for a vector
        of p values. ``s = 1 - p`` in reverse order. These are the knot
        evaluations used to assemble a weighted TVaR.
        """
        n = len(p_in)
        p = p_in.reshape((n, 1))
        s = (1 - p_in[::-1]).reshape((1, n))
        return np.where(s == 0,
                        np.zeros_like(p),
                        np.where(p == 1,
                                 np.ones_like(p),
                                 np.minimum(s / (1 - p), 1)))

    def tvar_info_df(self):
        """
        Return a DataFrame describing the affine pieces of a weighted-TVaR
        distortion: knots, slopes, intercepts, elasticities. Defined for
        ``wtdtvar`` kinds only; returns ``None`` for other kinds.
        """
        return None

    def plot_affine(self, ax=None, n_pts=101,
                    cmap_name='viridis', alpha=1.,
                    marker='o', marker_size=4):
        """Render the upper affine envelope of a ``wtdtvar`` distortion.
        ``None`` for other kinds."""
        return None

    def quick_gS(self, den):
        """Numba-backed ``g(1 - den.cumsum())``. Defined for ``tvar`` and
        ``bitvar`` only."""
        raise NotImplementedError(
            f"quick_gS only implemented for TVaR and BiTVaR, not {self._name}")

    def quick_ra(self, den, x=None):
        """Numba-backed risk-adjusted expectation. Defined for ``tvar``
        and ``bitvar`` only."""
        raise NotImplementedError(
            f"quick_ra only implemented for TVaR and BiTVaR, not {self._name}")

    def min_index(self, x):
        """For a ``minimum`` distortion: which member achieves the min at
        each ``x``. Returns ``None`` for other kinds."""
        return None

    def id(self):
        """Unique ID as a short string, based on constructor arguments."""
        bit = {k: v for k, v in self.__dict__.items()
               if k in ('_name', 'r0', 'df', 'shape', 'col_x', 'col_y')}
        return short_hash(str(bit))

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def name(self):
        return self.display_name if self.display_name != '' else self._name

    @name.setter
    def name(self, value):
        self._name = value

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, xs=None, n=101, both=True, ax=None, plot_points=True,
             scale='linear', c=None, c_dual=None, size='small', **kwargs):
        """
        Plot the distortion.

        :param xs: x values; defaults to a grid of length ``n``.
        :param n: number of points if ``xs`` is None.
        :param both: also plot ``g_dual``.
        :param ax: existing Axes; if None, a new figure is created.
        :param plot_points: for ``convex``, scatter the calibration points.
        :param scale: ``'linear'`` or ``'return'`` (log-log return scale).
        :param size: ``'small'``, ``'large'``, or a numeric figure side.
        :param kwargs: forwarded to ``ax.plot``.
        """
        assert scale in ['linear', 'return']

        if scale == 'return':
            xs = 10 ** np.linspace(-10, 0, n)
        elif xs is None:
            xs = np.hstack((0, np.linspace(1e-10, 1, n)))

        y1 = self.g(xs)
        y2 = self.g_dual(xs) if both else None

        if ax is None:
            if size == 'small':
                sz = FIG_H
            elif isinstance(size, (float, int)):
                sz = size
            else:
                sz = FIG_W
            fig, ax = plt.subplots(1, 1, figsize=(sz, sz), layout="constrained")

        if c is None:
            c = 'C0'
        if c_dual is None:
            c_dual = 'C1'
        if scale == 'linear':
            ax.plot(xs, y1, c=c, label=self.name, **kwargs)
            if both:
                ax.plot(xs, y2, c=c_dual, label='$g\\check$', **kwargs)
            ax.plot(xs, xs, color='k', lw=0.5, alpha=0.5)
        elif scale == 'return':
            ax.plot(xs, y1, c=c, label=self.name, **kwargs)
            if both:
                ax.plot(xs, y2, c=c_dual, label=f'Dual {self.name}', **kwargs)
            ax.set(xscale='log', yscale='log',
                   xlim=[1 / 5_000, 1], ylim=[1 / 5_000, 1])
            ax.plot(xs, xs, color='k', lw=0.5, alpha=0.5)

        self._plot_decorations(ax, xs, c, scale, plot_points)

        ax.set(title=self.name, aspect='equal')
        if scale == 'linear':
            ax.set(xticks=np.linspace(0, 1, 6),
                   yticks=np.linspace(0, 1, 6))
        if both:
            ax.legend(loc='upper left', fontsize='x-small')
        return ax

    def _plot_decorations(self, ax, xs, c, scale, plot_points):
        """Hook for kind-specific plot adornments. Overridden by
        ``ConvexDistortion`` to scatter calibration points."""
        return None

    # ------------------------------------------------------------------
    # Static factory shortcuts
    # ------------------------------------------------------------------

    @staticmethod
    def tvar(p):
        """Construct a TVaR distortion at level p."""
        return Distortion('tvar', p, display_name=f'TVaR({p:.3g})')

    @staticmethod
    def max():
        """TVaR at p=1 (the max)."""
        return Distortion('tvar', 1.0, display_name='max')

    @staticmethod
    def mean():
        """TVaR at p=0 (the mean)."""
        return Distortion('tvar', 0.0, display_name='mean')

    @staticmethod
    def wang(shape):
        """Construct a Wang distortion."""
        return Distortion('wang', shape, display_name=f'Wang({shape:.3g})')

    @staticmethod
    def ph(shape):
        """Construct a proportional-hazard distortion."""
        return Distortion('ph', shape, display_name=f'PH({shape:.3g})')

    @staticmethod
    def dual(shape):
        """Construct a dual-moment distortion."""
        return Distortion('dual', shape, display_name=f'dual({shape:.3g})')

    @staticmethod
    def bitvar(p0, p1, w=0.5):
        """
        Construct a BiTVaR with :math:`p_0 < p_1` and weight ``w`` on ``p1``.
        Degenerate combinations collapse to a TVaR.
        """
        if p0 == p1 or w == 0:
            return Distortion.tvar(p0)
        if w == 1:
            return Distortion.tvar(p1)
        return Distortion('bitvar', w, df=[p0, p1],
                          display_name=f'bitvar({p0:.3g}, {p1:.3g}; {w:.3g})')

    @staticmethod
    def ccoc(d):
        """
        Construct a CCoC distortion from discount factor ``d``. The
        default constructor takes return ``r`` instead; ``d = r / (1 + r)``.
        """
        r = d / (1. - d)
        return Distortion('ccoc', r, display_name=f'ccoc({r:.3g})')

    @staticmethod
    def minimum(distortion_list):
        """Construct a Distortion that is the pointwise minimum of others."""
        return Distortion('minimum', distortion_list,
                          display_name=f'minimum({len(distortion_list)})')

    @staticmethod
    def mixture(distortion_list, weights=None):
        """Construct a weighted mixture of distortions."""
        return Distortion('mixture', distortion_list, df=weights,
                          display_name=f'mixture({len(distortion_list)})')

    @staticmethod
    def beta(a, b):
        """Construct a beta distortion."""
        return Distortion('beta', [a, b],
                          display_name=f'beta({a:.3f}, {b:.3f})')

    @staticmethod
    def power(alpha, x0, x1):
        """Construct a power distortion."""
        return Distortion('power', alpha, df=[x0, x1],
                          display_name=f'power({alpha:.3f}, {x0:.3f}, {x1:.3f})')

    @staticmethod
    def distortions_from_params(params, index, r0=0.025, df=5.5,
                                pricing=True, strict=True):
        """
        Construct a dict of distortions from a calibration parameter table.

        ``params`` is a DataFrame indexed by (something, kind) with a
        ``param`` column; one entry per available distortion kind. Called
        by ``Portfolio.calibrate_distortions``.
        """
        temp = params.loc[index, :]
        dists = {}
        for dn in Distortion.available_distortions(pricing=pricing, strict=strict):
            param = float(temp.loc[dn, 'param'])
            dists[dn] = Distortion(name=dn, shape=param, r0=r0, df=df)
        return dists

    @staticmethod
    def convex_example(source='bond'):
        """
        Example convex distortion using yield-curve or cat-bond data.

        :param source: ``'bond'`` (BIS yield curve) or ``'cat'`` (cat bond
            ROL vs EL).
        """
        if source == 'bond':
            yield_curve = '''
            AAAA    0.000000  0.000000
            AAA     0.000018  0.006386
            AA      0.000144  0.007122
            A       0.000278  0.010291
            BBB     0.002012  0.017089
            BB      0.012674  0.036455
            B       0.040052  0.069181
            Z       1.000000  1.000000'''
            df = pd.read_fwf(StringIO(yield_curve))
            df.columns = ['Rating', 'EL', 'Yield']
            return Distortion('convex', 'Yield Curve', df=df,
                              col_x='EL', col_y='Yield')

        elif source.lower() == 'cat':
            cat_bond = '''EL,ROL
            0.116196,0.32613
            0.088113,0.2452
            0.074811,0.22769
            0.056385,0.17131
            0.046923,0.15326
            0.032961,0.12222
            0.02807,0.11037
            0.024205,0.1022
            0.011564,0.07284
            0.005813,0.06004
            0,0
            1,1'''
            df = pd.read_csv(StringIO(cat_bond))
            return Distortion('convex', 'Cat Bond', df=df,
                              col_x='EL', col_y='ROL')

        else:
            raise ValueError(
                f'Inadmissible value {source} passed to convex_example, '
                f'expected bond or cat')

    @staticmethod
    def bagged_distortion(data, proportion, samples, display_name=""):
        """
        Bootstrap-aggregated convex distortion: resample ``data``, take
        the convex envelope of each, average. ``data`` has two columns
        EL and Spread.
        """
        df = pd.DataFrame(index=np.linspace(0, 1, 10001), dtype=float)
        for i in range(samples):
            rebit = data.sample(frac=proportion, replace=False, random_state=RANDOM)
            rebit.loc[-1] = [0, 0]
            rebit.loc[max(rebit.index) + 1] = [1, 1]
            d = Distortion('convex', 0, df=rebit, col_x='EL', col_y='Spread')
            df[i] = d.g(df.index)

        df['avg'] = df.mean(axis=1)
        df2 = df['avg'].copy()
        df2.index.name = 's'
        df2 = df2.reset_index(drop=False)
        return Distortion('convex', 0, df=df2,
                          col_x='s', col_y='avg', display_name=display_name)

    @staticmethod
    def average_distortion(data, display_name, n=201,
                           el_col='EL', spread_col='Spread'):
        """
        Average distortion from (s, g(s)) pairs. Each (EL, Spread) row
        defines a wtdTVaR with knots at ``p=EL`` and ``p=1``.
        """
        els = data[el_col]
        spreads = data[spread_col]
        max_el = els.max()
        s = np.hstack((np.linspace(0, max_el, n), 1))
        ans = np.zeros((len(s), len(data)))
        for i, el, spread in zip(range(len(data)), els, spreads):
            p = 1 - el
            w = (spread - el) / (1 - el)
            d = Distortion('wtdtvar', w, df=[0, p])
            ans[:, i] = d.g(s)
        df = pd.DataFrame({'s': s, 'gs': np.mean(ans, 1)})
        return Distortion('convex', None, df=df,
                          col_x='s', col_y='gs', display_name=display_name)

    @staticmethod
    def s_gs_distortion(s, gs, display_name=''):
        """
        Convex envelope distortion built from {s, g(s)} sample points.

        Currently passes ``shape=0`` (no mass at zero) even if the
        provided (s, gs) imply one.
        """
        s = np.array(s)
        gs = np.array(gs)
        return Distortion('convex', 0,
                          df=pd.DataFrame({'s': s.flat, 'gs': gs.flat}),
                          col_x='s', col_y='gs', display_name=display_name)

    @staticmethod
    def random_distortion_ex(n=1, random_state=None):
        """
        Random distortion over several kinds (ph, wang, dual, wtdtvar,
        tvar, ccoc). Returns one Distortion if ``n == 1``, else a dict.
        """
        rng = np.random.default_rng(random_state)

        def _one(rng):
            method = rng.choice(['ph', 'wang', 'dual', 'wtdtvar', 'tvar', 'ccoc'])
            match method:
                case 'ph':
                    return Distortion.ph(rng.uniform(0.1 if rng.uniform() < 0.1 else 0.4, 1.0))
                case 'wang':
                    return Distortion.wang(rng.uniform(0.5, 1.5))
                case 'dual':
                    return Distortion.dual(rng.uniform(1.1, 2.5))
                case 'tvar':
                    return Distortion.tvar(rng.uniform(0.01, 0.99))
                case 'ccoc':
                    return Distortion.ccoc(rng.uniform(0.02, 0.35))
                case 'wtdtvar':
                    knots = rng.choice([2, 2, 3, 3, 3, 4, 4, 5, 8, 12])
                    mean = 0 if (t := rng.uniform() < 0.5) else t / 2
                    mass = 0 if (t := rng.uniform() < 0.75) else t / 4
                    return Distortion.random_distortion(
                        knots, mass, mean, random_state=rng)

        if n == 1:
            return _one(rng)
        return {i: _one(rng) for i in range(n)}

    @staticmethod
    def random_distortion(n_knots, mass=0, mean=0, wt_rng=None,
                          name="", random_state=None):
        """
        Random wtdTVaR distortion. ``mass`` (resp. ``mean``) adds a
        knot at p=1 (p=0). ``wt_rng`` can generate spiky weights, e.g.
        ``wt_rng=ss.pareto(1.5).rvs``.
        """
        random_state = random_state or np.random.default_rng()
        if mass: n_knots -= 1
        if mean: n_knots -= 1
        ps = random_state.random(n_knots)
        ps.sort()
        assert n_knots >= 0
        if n_knots == 0:
            assert mass + mean == 1, \
                'BiTVaR weighting mean and max, weights must sum to 1.'
        if wt_rng is None:
            wts = random_state.random(n_knots)
        else:
            wts = wt_rng(size=n_knots, random_state=random_state)
        wts = wts / wts.sum(dtype=np.float64) * (1 - mass - mean)
        mn = ''
        ma = ''
        if mass:
            ps = np.append(ps, 1)
            wts = np.append(wts, mass)
            ma = f', mx={mass:.3f}'
        if mean:
            ps = np.insert(ps, 0, 0)
            wts = np.insert(wts, 0, mean)
            mn = f', mn={mean:.3f}'
        return Distortion('wtdtvar', ps, df=wts,
                          display_name=name or f'Rnd {n_knots} knots{mn}{ma}')

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(self, ser, a=np.inf, kind='ask', method='dx',
              S_calculation='backwards', as_frame=False):
        """
        Compute distorted (ask/bid/both) prices of the distribution
        described by ``ser`` with asset limit ``a``.

        Replaces the older ``price`` (tuple return) and ``price2``
        (DataFrame return); strictly preferred in new code.

        ``S_calculation='backwards'`` computes
        ``S = p_total[::-1].cumsum()[::-1].shift(-1, fill_value=0)``,
        which is more reliable than the forward form for thin-tailed
        risks. The forward form remains available.

        ``method='dx'`` computes ``∫ gS dx`` (fewer diffs);
        ``method='ds'`` computes ``∫ x d(gS)``. Neither requires a
        unique index.

        Asset level ``a`` must be present in ``ser.index`` (no
        interpolation).

        :param ser: probability series indexed by outcome.
        :param a: asset level (truncation point).
        :param kind: ``'ask'``, ``'bid'``, or ``'both'``.
        :param method: ``'dx'`` or ``'ds'``.
        :param S_calculation: ``'forwards'`` or ``'backwards'``.
        :param as_frame: return a DataFrame instead of a namedtuple.
        """
        assert kind in ['bid', 'ask', 'both'], \
            "kind must be 'bid', 'ask', or 'both'"
        assert method in ['dx', 'ds'], "method must be 'dx' or 'ds'"
        assert S_calculation in ['forwards', 'backwards'], \
            "S_calculation must be 'forwards' or 'backwards'"
        if not isinstance(ser, pd.Series):
            raise ValueError(f'ser must be a pandas Series, not {type(ser)}')

        if not ser.index.is_monotonic_increasing:
            ser = ser.sort_index(ascending=True)

        if S_calculation == 'forwards':
            if a < np.inf:
                assert a in ser.index, f'a={a} must be in the index of ser'
                ser = ser.loc[:a]
            S = np.maximum(0, 1 - ser.cumsum())
            if a == np.inf:
                S.iloc[-1] = 0
        else:
            if not np.allclose(ser.sum(), 1):
                print(f'WARNING: ser.sum() = {ser.sum()} is not 1.')
            S = ser[::-1].cumsum().shift(1, fill_value=0)[::-1]
            S = np.minimum(1, S)
            if a < np.inf:
                assert a in ser.index, f'a={a} must be in the index of ser'
                S = S.loc[:a]

        gS = dual_gS = None
        el = bid = ask = np.nan

        if kind == 'ask':
            gS = np.array(self.g(S))
        elif kind == 'both':
            gS = np.array(self.g(S))
            dual_gS = np.array(self.g_dual(S))
        elif kind == 'bid':
            dual_gS = np.array(self.g_dual(S))

        if method == 'dx':
            dx = np.diff(S.index)
            x0 = S.index[0]
            el = (S.iloc[:-1].values * dx).sum() + x0
            if gS is not None:
                ask = (gS[:-1] * dx).sum() + x0
            if dual_gS is not None:
                bid = (dual_gS[:-1] * dx).sum() + x0
        else:
            # ds: the last adjustment to ∫ x dgS is to add a P(X>a)
            # provided a is finite; if a is infinite we let it equal 0
            # so the algebra collapses correctly.
            if a == np.inf:
                a = 0
            dS = np.diff(S, prepend=1.)
            x = np.array(S.index)
            el = -((x * dS).sum()) + a * S.iloc[-1]
            if gS is not None:
                dgS = np.diff(gS, prepend=1.)
                ask = -((x * dgS).sum()) + a * gS[-1]
            if dual_gS is not None:
                ddual_gS = np.diff(dual_gS, prepend=1.)
                bid = -((x * ddual_gS).sum()) + a * dual_gS[-1]

        if as_frame:
            return pd.DataFrame([[bid, el, ask, self.name, a]],
                                index=[0],
                                columns=pd.Index(
                                    ['bid', 'el', 'ask', 'distortion', 'assets']))
        Price = namedtuple('Price', 'bid,el,ask')
        return Price(bid, el, ask)

    def make_q(self, ser, a=np.inf):
        """
        Vector of risk-adjusted probabilities for use in pricing.

        Uses backwards S calculation, ask pricing, and ``method='ds'``;
        see :meth:`price` for details.

        Used as::

            q = d.make_q(x, a)
            ask = -((x * q['q']).sum()) + a * q['gS'].iloc[-1]
        """
        if not isinstance(ser, pd.Series):
            raise ValueError(f'ser must be a pandas Series, not {type(ser)}')
        if not ser.index.is_monotonic_increasing:
            ser = ser.sort_index(ascending=True)

        if ser.sum() < np.nextafter(1.0, 0.0):
            raise ValueError(
                'Sum of input probabilities must be 1. Try '
                'remove_fuzz=True if using a Portfolio')
        S = ser[::-1].cumsum().shift(1, fill_value=0)[::-1]
        S = np.minimum(1, S)
        if a < np.inf:
            assert a in ser.index, f'a={a} must be in the index of ser'
            S = S.loc[:a]

        gS = np.array(self.g(S))
        dS = -np.diff(S, prepend=1.)
        dgS = -np.diff(gS, prepend=1.)
        return pd.DataFrame({'p': dS, 'q': dgS, 'S': S, 'gS': gS},
                            index=S.index)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    # Starting shape used by ``Portfolio.calibrate_distortion`` to
    # construct the uncalibrated distortion before ``calibrate`` is
    # called. Each pricing subclass overrides; ``None`` means the kind
    # is not calibratable through the Portfolio dispatch.
    _calibration_init_shape = None

    def calibrate(self, S, bs, premium_target, *, ess_sup=0.0,
                  assets=0.0, el=None, **kwargs):
        """
        Calibrate the shape parameter so the distorted integral matches
        ``premium_target`` on the supplied survival vector ``S``.

        The pricing-distortion subclasses (``ph``, ``wang``, ``dual``,
        ``tvar``, ``ccoc``, ``ly``, ``clin``, ``lep``, ``cll``) implement
        this; the base raises ``NotImplementedError``. After convergence
        the method mutates ``self.shape`` (and audit fields ``error``,
        ``premium_target``, ``assets``) and re-runs ``_build`` so any
        cached state derived from ``shape`` is refreshed.

        Parameters
        ----------
        S : ndarray
            Survival function evaluated on the bs-grid up to the asset
            limit; must be strictly positive and weakly decreasing.
        bs : float
            Bucket size of the discretisation; the integration step.
        premium_target : float
            Premium that the distorted integral ``∫ g(S) dx`` must hit.
        ess_sup : float, optional
            Essential supremum (used by mass-at-zero kinds: ``ly``,
            ``clin``, ``lep``).
        assets : float, optional
            Asset level; recorded on the returned distortion for audit
            and used by the closed-form ``ccoc`` calibration.
        el : float, optional
            Expected loss at ``assets``; required by ``ccoc``.

        Returns
        -------
        Distortion
            ``self``, mutated in place.
        """
        raise NotImplementedError(
            f'calibrate not implemented for {type(self).__name__}')

    def _newton_iterate(self, f, shape, *, max_iter=50, tol=1e-5):
        """
        Run a Newton iteration on ``f(x) → (residual, derivative)``.

        Returns ``(converged_shape, residual)``. Callers warn on
        non-convergence via :meth:`_finalize_calibration`.

        Notes
        -----
        Faithful port of the inline Newton loop that lived in
        ``Portfolio.calibrate_distortion``: initial residual is computed
        before the loop, so a starting shape already within ``tol`` skips
        iteration entirely. Step is ``shape -= fx / fxp``; no damping,
        no line search.
        """
        fx, fxp = f(shape)
        i = 0
        while abs(fx) > tol and i < max_iter:
            shape = shape - fx / fxp
            fx, fxp = f(shape)
            i += 1
        return shape, fx

    def _finalize_calibration(self, shape, fx, premium_target, assets,
                              *, tol=1e-5):
        """Write the calibrated shape and audit fields, log on
        non-convergence, and re-run ``_build`` so cached state matches.
        """
        self.shape = shape
        self.error = fx
        self.premium_target = premium_target
        self.assets = assets
        if abs(fx) > tol:
            logger.warning(
                f'{type(self).__name__} calibration: questionable '
                f'convergence, target {premium_target} error {fx}')
        self._build()


# ===========================================================================
# Concrete distortion kinds. Order of declaration is the order that
# Distortion.available_distortions() returns.
# ===========================================================================


class CCoCDistortion(Distortion):
    """
    Constant cost-of-capital distortion. Shape is the target return r;
    the equivalent linear form has slope :math:`v = 1/(1+r)` and intercept
    :math:`d = r/(1+r)`. The legacy alias ``'roe'`` maps to this kind via
    :meth:`Distortion.__new__`.
    """
    kind = 'ccoc'
    med_name = 'Const CoC'
    long_name = 'Constant CoC'
    documented = True
    pricing_ok = True
    strict_pricing = True
    _calibration_init_shape = 0.25

    def _build(self):
        r = self.shape
        v = 1 / (1 + r)
        d = 1 - v
        self._v = v
        self._d = d
        self.has_mass = (d > 0)
        self.mass = d
        self.standard_shape = r / (1 + r)

    def g(self, x):
        d = self._d
        v = self._v
        return np.where(x == 0, 0, np.minimum(1, d + v * x))

    def g_inv(self, x):
        d = self._d
        v = self._v
        return np.where(x <= d, 0, (x - d) / v)

    def g_prime(self, x):
        return self._v

    def calibrate(self, S, bs, premium_target, *, ess_sup=0.0,
                  assets=0.0, el=None, **kwargs):
        """
        Closed-form calibration: ``r = (premium - el) / (assets - premium)``.

        No iteration; ``r0`` is forced to zero on the returned distortion.
        """
        assert el is not None and premium_target, \
            'CCoC calibration requires el and a non-zero premium_target'
        r = (premium_target - el) / (assets - premium_target)
        self.r0 = 0.0
        self._finalize_calibration(r, 0.0, premium_target, assets)
        return self


class PHDistortion(Distortion):
    """Proportional-hazard distortion: :math:`g(x) = x^\\rho`."""
    kind = 'ph'
    med_name = 'Prop Hzrd'
    long_name = 'Proportional Hazard'
    documented = True
    pricing_ok = True
    strict_pricing = True
    _calibration_init_shape = 0.95

    def _build(self):
        self.standard_shape = (1 - self.shape) / (1 + self.shape)

    def g(self, x):
        return x ** self.shape

    def g_inv(self, x):
        return x ** (1.0 / self.shape)

    def g_prime(self, x):
        rho = self.shape
        return np.where(x > 0, rho * x ** (rho - 1.0), np.inf)

    def calibrate(self, S, bs, premium_target, *, ess_sup=0.0,
                  assets=0.0, el=None, **kwargs):
        """Newton on ``∫ S^ρ dx = premium_target``."""
        lS = np.log(S)

        def f(rho):
            trho = S ** rho
            ex = np.sum(trho) * bs
            ex_prime = np.sum(trho * lS) * bs
            return ex - premium_target, ex_prime

        shape, fx = self._newton_iterate(f, self._calibration_init_shape)
        self._finalize_calibration(shape, fx, premium_target, assets)
        return self


class WangDistortion(Distortion):
    """Wang distortion: :math:`g(x) = \\Phi(\\Phi^{-1}(x) + \\lambda)`."""
    kind = 'wang'
    med_name = 'Wang'
    long_name = 'Wang-normal'
    documented = True
    pricing_ok = True
    strict_pricing = True
    _calibration_init_shape = 0.95

    def _build(self):
        n = ss.norm()
        self._norm = n
        self.standard_shape = 2 * n.cdf(self.shape / 2 ** 0.5) - 1

    def g(self, x):
        n = self._norm
        return n.cdf(n.ppf(x) + self.shape)

    def g_inv(self, x):
        n = self._norm
        return n.cdf(n.ppf(x) - self.shape)

    def g_prime(self, x):
        n = self._norm
        return n.pdf(n.ppf(x) + self.shape) / n.pdf(n.ppf(x))

    def calibrate(self, S, bs, premium_target, *, ess_sup=0.0,
                  assets=0.0, el=None, **kwargs):
        """Newton on ``∫ Φ(Φ⁻¹(S) + λ) dx = premium_target``."""
        n = ss.norm()

        def f(lam):
            temp = n.ppf(S) + lam
            tlam = n.cdf(temp)
            ex = np.sum(tlam) * bs
            ex_prime = np.sum(n.pdf(temp)) * bs
            return ex - premium_target, ex_prime

        shape, fx = self._newton_iterate(f, self._calibration_init_shape)
        self._finalize_calibration(shape, fx, premium_target, assets)
        return self


class DualDistortion(Distortion):
    """Dual-moment distortion: :math:`g(x) = 1 - (1-x)^p`."""
    kind = 'dual'
    med_name = 'Dual Mom'
    long_name = 'Dual Moment'
    documented = True
    pricing_ok = True
    strict_pricing = True
    _calibration_init_shape = 2.0

    def _build(self):
        self.standard_shape = (self.shape - 1) / (self.shape + 1)

    def g(self, x):
        return 1 - (1 - x) ** self.shape

    def g_inv(self, y):
        return 1 - (1 - y) ** (1 / self.shape)

    def g_prime(self, x):
        p = self.shape
        return p * (1 - x) ** (p - 1)

    def calibrate(self, S, bs, premium_target, *, ess_sup=0.0,
                  assets=0.0, el=None, **kwargs):
        """Newton on ``∫ (1 - (1-S)^ρ) dx = premium_target``.

        Notes
        -----
        ``log(1 - S)`` is masked to 0 wherever ``S == 1`` to avoid the
        log-singularity contaminating the derivative; on that region
        ``(1 - S)^ρ = 0`` so the derivative contribution is zero anyway.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            lS = -np.log(1 - S)
        lS[S == 1] = 0

        def f(rho):
            temp = (1 - S) ** rho
            trho = 1 - temp
            ex = np.sum(trho) * bs
            ex_prime = np.sum(temp * lS) * bs
            return ex - premium_target, ex_prime

        shape, fx = self._newton_iterate(f, self._calibration_init_shape)
        self._finalize_calibration(shape, fx, premium_target, assets)
        return self


class TVaRDistortion(Distortion):
    """
    Tail-VaR distortion at threshold ``p``: :math:`g(x) = \\min(\\alpha x, 1)`
    where :math:`\\alpha = 1/(1-p)`. ``p=0`` is the mean; ``p=1`` is the max
    (a point mass at 1).
    """
    kind = 'tvar'
    med_name = 'Tail VaR'
    long_name = 'Tail VaR'
    documented = True
    pricing_ok = True
    strict_pricing = True
    _calibration_init_shape = 0.9

    def _build(self):
        p = self.shape
        self.standard_shape = p
        if p == 1:
            self.has_mass = True
            self.mass = 1
            self._alpha = np.nan
        else:
            self.has_mass = False
            self._alpha = 1 / (1 - p)

    def g(self, x):
        if self.shape == 1:
            # <=0 handles rounding issues gracefully
            return np.where(x <= 0, 0, 1)
        return np.minimum(self._alpha * x, 1)

    def g_inv(self, x):
        return np.where(x < 1, x * (1 - self.shape), 1)

    def g_prime(self, x):
        p = self.shape
        alpha = self._alpha
        if p == 1:
            return np.where(x < 1 - p, alpha, 0)
        return np.where(x <= 1 - p, alpha, 0)

    def quick_gS(self, den):
        p = self.shape
        if isinstance(den, pd.Series):
            return tvar_gS(den.values, p)
        return tvar_gS(den, p)

    def quick_ra(self, den, x=None):
        p = self.shape
        if isinstance(den, pd.Series):
            return tvar_ra(den.values, np.array(den.index), p)
        assert x is not None
        return tvar_ra(den, x, p)

    def calibrate(self, S, bs, premium_target, *, ess_sup=0.0,
                  assets=0.0, el=None, **kwargs):
        """
        Newton on ``∫ min(S/(1-ρ), 1) dx = premium_target``.

        TVaR uses ``max_iter=200`` instead of the default 50 — the loop
        often takes longer to find a clean ρ near 1 when the premium
        target sits in the deep tail.
        """
        def f(rho):
            temp = np.where(S <= 1 - rho, S / (1 - rho), 1)
            temp2 = np.where(S <= 1 - rho, S / (1 - rho) ** 2, 1)
            ex = np.sum(temp) * bs
            ex_prime = np.sum(temp2) * bs
            return ex - premium_target, ex_prime

        shape, fx = self._newton_iterate(
            f, self._calibration_init_shape, max_iter=200)
        self._finalize_calibration(shape, fx, premium_target, assets)
        return self


class BiTVaRDistortion(Distortion):
    """
    Convex combination of two TVaR distortions at ``p0 < p1`` with weight
    ``w`` on ``p1``. Stored as ``shape=w``, ``df=[p0, p1]``.
    """
    kind = 'bitvar'
    med_name = 'BiTVaR'
    long_name = 'BiTVaR'
    documented = True
    pricing_ok = True

    def _build(self):
        if (not isinstance(self.df, (list, tuple))
                or len(self.df) != 2):
            raise ValueError(
                'Inadmissible parameters to Distortion for bitvar. '
                'Pass shape=wt for p1 and df=[p0, p1]')
        p0, p1 = self.df
        if not (p0 < p1):
            raise ValueError(f'bitvar requires p0 < p1, got {p0=}, {p1=}')
        w = self.shape
        self._p0 = p0
        self._p1 = p1
        self._w = w
        self.has_mass = (p1 == 1)
        self.mass = w if p1 == 1 else 0
        pt = (1 - p1) / (1 - p0) * (1 - w) + w
        self._pt = pt

        if p1 == 1:
            # has a point mass at 1; g uses the closed-form method below,
            # g_inv is built from knot points
            self._alpha = 1 / (1 - p0)
            # the 1e-50 wedge avoids a singularity at the corner; without
            # it interp1d collapses the segment to zero. Wasted two days.
            s = np.array([0.,   1e-50, 1 - p0, 1.])
            gs = np.array([0.,     pt, 1.,     1.])
            self.g_inv = interp1d(gs, s, kind='linear',
                                  bounds_error=False, fill_value=(0, 1))
        else:
            s = np.array([0.,  1 - p1, 1 - p0, 1.])
            gs = np.array([0.,     pt, 1.,     1.])
            self.g = interp1d(s, gs, kind='linear',
                              bounds_error=False, fill_value=(0, 1))
            self.g_inv = interp1d(gs, s, kind='linear',
                                  bounds_error=False, fill_value=(0, 1))

    def g(self, x):
        # only reached when p1 == 1
        w = self._w
        alpha = self._alpha
        return w * np.where(x <= 0, 0, 1) + (1 - w) * np.minimum(alpha * x, 1)

    def g_prime(self, x):
        p0, p1, w = self._p0, self._p1, self._w
        if p1 < 1:
            return np.where(x > 1 - p0, 0,
                            np.where(x < 1 - p1,
                                     w / (1 - p1) + (1 - w) / (1 - p0),
                                     (1 - w) / (1 - p0)))
        return np.where(x > 1 - p0, 0, (1 - w) / (1 - p0))

    def quick_gS(self, den):
        p0, p1 = self.df
        w = self.shape
        if isinstance(den, pd.Series):
            return bitvar_gS(den.values, p0, p1, w)
        return bitvar_gS(den, p0, p1, w)

    def quick_ra(self, den, x=None):
        p0, p1 = self.df
        w = self.shape
        if isinstance(den, pd.Series):
            return bitvar_ra(den.values, np.array(den.index), p0, p1, w)
        return bitvar_ra(den, x, p0, p1, w)


class WtdTVaRDistortion(Distortion):
    """
    Weighted TVaR: ``shape`` is a sorted array of p values, ``df`` is the
    matching weights. A mass at p=1 (max term) is supported. ``g`` is
    piecewise linear; ``g_prime`` is exact.
    """
    kind = 'wtdtvar'
    med_name = 'WtdTVaR'
    long_name = 'Weighted TVaR'
    documented = True
    pricing_ok = True

    def _build(self):
        ps = np.array(self.shape)
        wts = np.array(self.df)
        if 1 in ps and wts[-1] > 0:
            self.has_mass = True
            self.mass = wts[-1]
        else:
            self.has_mass = False
            self.mass = 0

        if self.display_name == '':
            self.display_name = f'wtdTVaR on {len(ps):d} points'

        assert np.all(ps[:-1] < ps[1:]), 'ps must be sorted ascending'

        # ensure 0 and 1 are present so g(0)=0, g(1)=1
        if 0 not in ps:
            ps = np.insert(ps, 0, 0)
            wts = np.insert(wts, 0, 0)
        if 1 not in ps:
            ps = np.append(ps, 1)
            wts = np.append(wts, 0)
        else:
            # mass at 1 needs an explicit prev knot for interp1d
            ps[-1] = np.nextafter(1, -1)
            ps = np.append(ps, 1)
            wts = np.append(wts, 0)

        self._ps_padded = ps
        self._wts_padded = wts
        # slopes per TVaR component (1/(1-p)); inf at p=1 is OK because
        # the active_mask only includes those components for s > p.
        self._component_slopes = np.divide(
            wts, 1.0 - ps,
            where=(ps < 1.0),
            out=np.full_like(ps, np.inf))

        s = 1 - ps[::-1]
        gs = wts @ Distortion.tvar_terms(ps)
        self.g = interp1d(s, gs, kind='linear',
                          bounds_error=False, fill_value=(0, 1))
        self.g_inv = interp1d(gs, s, kind='linear',
                              bounds_error=False, fill_value=(0, 1))

    def g_prime(self, s):
        ps = self._ps_padded
        component_slopes = self._component_slopes
        s = 1 - np.atleast_1d(s)
        s_col = s[..., np.newaxis]
        active_mask = s_col > ps
        grad = np.sum(np.where(active_mask, component_slopes, 0.0), axis=-1)
        is_cusp = np.any(np.isclose(s_col, ps, rtol=1e-14, atol=1e-14), axis=-1)
        grad[is_cusp] = np.nan
        return grad

    def tvar_info_df(self):
        p = np.array(self.shape)
        wts = np.array(self.df)
        if 0 not in p:
            p = np.hstack((0, p))
            wts = np.hstack((0, wts))
        if 1 not in p:
            p = np.hstack((p, 1))
            wts = np.hstack((wts, 0))
        knots = 1 - p
        gs = self.g(knots)
        df = pd.DataFrame({'s': knots, 'gs': gs, 'p': p, 'wts': wts})
        df = df.sort_values('s').reset_index(drop=True)
        df['slope'] = (df.gs.shift(-1) - df.gs) / (df.s.shift(-1) - df.s)
        df['intercept'] = df.gs - df.slope * df.s
        if p[-1] == 1 and wts[-1] > 0:
            df.loc[0, 'slope'] = (df.gs[1] - df.wts[0]) / df.s[1]
            df.loc[0, 'intercept'] = df.wts[0]

        def eta(s):
            s = np.asarray(s)
            return np.where(s == 0, 1,
                            np.where(s == 1, 0,
                                     s * self.g_prime(s) / self.g(s)))

        df['gprime-'] = self.g_prime((df.s + df.s.shift(1)) / 2)
        df['gprime+'] = self.g_prime((df.s + df.s.shift(-1)) / 2)
        df['eta-'] = eta((df.s + df.s.shift(1)) / 2)
        df['eta+'] = eta((df.s + df.s.shift(-1)) / 2)
        return df

    def plot_affine(self, ax=None, n_pts=101,
                    cmap_name='viridis', alpha=1.,
                    marker='o', marker_size=4):
        ax = self.plot(both=False)
        ps = np.linspace(0, 1, n_pts)
        df = self.tvar_info_df()
        n_lines = len(df)
        cmap = colormaps.get_cmap(cmap_name)
        colors = [cmap(i / max(1, n_lines - 1)) for i in range(n_lines)]
        for c, (n, r) in zip(colors, df.iterrows()):
            if np.isnan(r.slope):
                continue
            line = r.intercept + r.slope * ps
            line = np.where((line >= 0) & (line <= 1), line, np.nan)
            ax.plot(ps, line, lw=0.5, color=c, alpha=alpha)
        if len(df) < 20:
            ax.scatter(df.s, df.gs, color=colors,
                       marker=marker, s=marker_size, zorder=3)
        return ax


class ConvexDistortion(Distortion):
    """
    Convex envelope / piecewise-linear interpolated distortion. ``df`` is
    a DataFrame with columns ``col_x`` and ``col_y``; if more than two
    rows, the convex hull determines the knots.
    """
    kind = 'convex'
    med_name = 'Convex Env'
    long_name = 'Convex Envelope'
    documented = True
    pricing_ok = False

    def _build(self):
        # legacy: shape > 0 indicates mass at zero
        if isinstance(self.shape, (int, float)) and self.shape and self.shape > 0:
            self.has_mass = True
            self.mass = self.shape
        else:
            self.has_mass = False
            self.mass = 0.0

        if self.display_name == '':
            self.display_name = f'Convex on {len(self.df):d} points'

        if not (0 in self.df[self.col_x].values
                and 1 in self.df[self.col_x].values):
            self.df = self.df[[self.col_x, self.col_y]].copy().reset_index(drop=True)
            self.df.loc[len(self.df)] = (0, 0)
            self.df.loc[len(self.df)] = (1, 1)
            self.df = self.df.sort_values(self.col_x)

        if len(self.df) > 2:
            hull = ConvexHull(self.df[[self.col_x, self.col_y]])
            knots = list(set(hull.simplices.flatten()))
            self.g = interp1d(
                self.df.iloc[knots, self.df.columns.get_loc(self.col_x)],
                self.df.iloc[knots, self.df.columns.get_loc(self.col_y)],
                kind='linear', bounds_error=False, fill_value=(0, 1))
            self.g_inv = interp1d(
                self.df.iloc[knots, self.df.columns.get_loc(self.col_y)],
                self.df.iloc[knots, self.df.columns.get_loc(self.col_x)],
                kind='linear', bounds_error=False, fill_value=(0, 1))
        else:
            self.df = self.df.sort_values(self.col_x)
            self.g = interp1d(self.df[self.col_x], self.df[self.col_y],
                              kind='linear', bounds_error=False,
                              fill_value=(0, 1))
            self.g_inv = interp1d(self.df[self.col_y], self.df[self.col_x],
                                  kind='linear', bounds_error=False,
                                  fill_value=(0, 1))

    def _plot_decorations(self, ax, xs, c, scale, plot_points):
        if not plot_points:
            return
        if len(self.df) > 50:
            alpha = 0.35
        elif len(self.df) > 20:
            alpha = 0.6
        else:
            alpha = 1
        if c is None:
            c = 'C4'
        if scale == 'linear':
            ax.scatter(x=self.df[self.col_x], y=self.df[self.col_y],
                       marker='.', s=15, color=c, alpha=alpha)
        elif scale == 'return':
            ax.scatter(x=1 / self.df[self.col_x],
                       y=1 / self.df[self.col_y],
                       marker='.', s=15, color=c, alpha=alpha)


class MinimumDistortion(Distortion):
    """Pointwise minimum of several distortions. ``shape`` is the list."""
    kind = 'minimum'
    med_name = 'Minimum'
    long_name = 'Minimum'
    documented = False
    pricing_ok = False

    def _build(self):
        dists = self.shape
        self.has_mass = bool(np.all([d.has_mass for d in dists]))
        self.mass = float(np.min([d.mass for d in dists])) if self.has_mass else 0
        if self.display_name == '':
            self.display_name = f'Minimum of {len(dists):d} distortions'

    def min_index(self, x):
        g_values = np.array([gi.g(x) for gi in self.shape])
        return np.argmin(g_values, axis=0)

    def g(self, x):
        g_values = np.array([gi.g(x) for gi in self.shape])
        return np.min(g_values, axis=0)

    def g_prime(self, x):
        # the slope at x is the slope of whichever member achieves the min.
        # nudge x=1 slightly inside so argmin is unambiguous.
        x = np.where(x == 1, 1 - 1e-15, x)
        g_values = np.array([gi.g(x) for gi in self.shape])
        min_idx = np.argmin(g_values, axis=0)
        if np.isscalar(min_idx):
            return self.shape[min_idx].g_prime(x)
        if np.isscalar(x):
            return np.array([self.shape[i].g_prime(x) for i in min_idx])
        return np.array([self.shape[i].g_prime(xi)
                         for i, xi in zip(min_idx, x)])

    def g_inv(self, y):
        # inverse of a pointwise min is the pointwise max of inverses.
        inv_values = np.array([gi.g_inv(y) for gi in self.shape])
        return np.max(inv_values, axis=0)


class MixtureDistortion(Distortion):
    """Weighted mixture of distortions. ``shape`` is the list, ``df`` the
    weights (defaults to uniform)."""
    kind = 'mixture'
    med_name = 'Mixture'
    long_name = 'Mixture'
    documented = False
    pricing_ok = False

    def _build(self):
        dists = list(self.shape)
        self.has_mass = bool(np.any([d.has_mass for d in dists]))
        if self.has_mass:
            self.mass = float(np.sum([d.mass * w
                                      for d, w in zip(dists, self.df)]))
        else:
            self.mass = 0
        if self.display_name == '':
            self.display_name = f'Mixture of {len(dists):d} distortions'
        if self.df is None:
            self.df = np.array([1 / len(dists)] * len(dists))
        self._weights = np.array(self.df.copy() if hasattr(self.df, 'copy')
                                 else self.df)

    def _combine(self, values):
        w = self._weights
        if values.ndim > 2:
            flat = values.reshape(len(self.shape), -1)
            return (w @ flat).reshape(values.shape[1], values.shape[2])
        return w @ values

    def g(self, x):
        return self._combine(np.array([gi.g(x) for gi in self.shape]))

    def g_prime(self, x):
        return self._combine(np.array([gi.g_prime(x) for gi in self.shape]))

    def g_inv(self, y):
        raise NotImplementedError('Inverse of mixture not implemented')
class BetaDistortion(Distortion):
    """
    Beta distortion: :math:`g(x) = F_{a,b}(x)` for a Beta(a, b) CDF.

    Constraints: ``0 < a <= 1`` and ``b >= 1``. ``b=1`` is PH with
    :math:`\\rho = 1/a`; ``a=1`` is dual with :math:`\\rho = b`.
    Reference: Wirch and Hardy, "A synthesis of risk measures for
    capital adequacy" (IME 1999).
    """
    kind = 'beta'
    med_name = 'Beta'
    long_name = 'Beta'
    documented = True
    pricing_ok = False

    def _build(self):
        a, b = self.shape
        assert 0 < a <= 1, f'a parameter must be in (0, 1], not {a}'
        assert b >= 1, f'b parameter must be >= 1, not {b}'
        self._fz = ss.beta(a, b)

    def g(self, x):
        return self._fz.cdf(x)

    def g_inv(self, x):
        return self._fz.ppf(x)

    def g_prime(self, x):
        return self._fz.pdf(x)



class PowerDistortion(Distortion):
    """
    Power distortion built from part of a power-function distribution.
    Compare with the Bernegger approach. Allows controlled slopes at 0
    and 1. ``shape = alpha``, ``df = [x0, x1]`` with ``x0 < x1``.
    """
    kind = 'power'
    med_name = 'Power'
    long_name = 'Power'
    documented = False
    pricing_ok = False

    def _build(self):
        x0, x1 = self.df
        assert x0 < x1, 'x0 must be less than x1'
        alpha = float(self.shape)
        self._alpha = alpha
        self._x0 = x0
        self._x1 = x1
        if alpha != 1:
            self._bl = np.power(x1, -alpha + 1)
            self._br = np.power(x0, -alpha + 1)
        else:
            self._bl = np.log(x1)
            self._br = np.log(x0)

    def g(self, s):
        alpha = self._alpha
        x0, x1 = self._x0, self._x1
        bl, br = self._bl, self._br
        if alpha != 1:
            tl = np.power(x0 + s * (x1 - x0), -alpha + 1)
            return (tl - br) / (bl - br)
        t = np.log(x0 + s * (x1 - x0))
        return (t - br) / (bl - br)

    def g_prime(self, s):
        alpha = self._alpha
        x0, x1 = self._x0, self._x1
        bl, br = self._bl, self._br
        if alpha != 1:
            tl = np.power(x0 + s * (x1 - x0), -alpha)
            return (1 - alpha) * (x1 - x0) * tl / (bl - br)
        return (x1 - x0) / (bl - br) / (x0 + s * (x1 - x0))

    def g_inv(self, s):
        alpha = self._alpha
        x0, x1 = self._x0, self._x1
        bl, br = self._bl, self._br
        if alpha != 1:
            t1 = np.power(s * (bl - br) + br, 1 / (1 - alpha))
            return (t1 - x0) / (x1 - x0)
        return (np.exp(s * (bl - br) + br) - x0) / (x1 - x0)


class CLLDistortion(Distortion):
    """Capped log-linear distortion: :math:`g(x) = \\min(1, e^{r_0} x^b)`."""
    kind = 'cll'
    med_name = 'Capd Loglin'
    long_name = 'Capped Loglinear'
    documented = True
    pricing_ok = True
    strict_pricing = True
    _calibration_init_shape = 0.95

    def _build(self):
        self._ea = np.exp(self.r0)

    def g(self, x):
        b = self.shape
        ea = self._ea
        return np.where(x == 0, 0, np.minimum(1, ea * x ** b))

    def g_inv(self, x):
        b = self.shape
        ea = self._ea
        return np.where(x < 1, np.minimum(1, (x / ea) ** (1 / b)), 1)

    def calibrate(self, S, bs, premium_target, *, ess_sup=0.0,
                  assets=0.0, el=None, **kwargs):
        """Newton on ``∫ min(1, e^{r0} S^b) dx = premium_target``.

        ``log S`` is forced to 0 in the first cell because ``S[0]`` is
        often 1 (numerically), and the resulting log(1)=0 step pollutes
        the derivative with a NaN if ``S[0]`` happens to be exactly 1.
        """
        lS = np.log(S)
        lS[0] = 0
        ea = np.exp(self.r0)

        def f(b):
            uncapped = ea * S ** b
            ex = np.sum(np.minimum(1, uncapped)) * bs
            ex_prime = np.sum(np.where(uncapped < 1, uncapped * lS, 0)) * bs
            return ex - premium_target, ex_prime

        shape, fx = self._newton_iterate(f, self._calibration_init_shape)
        self._finalize_calibration(shape, fx, premium_target, assets)
        return self


class CLinDistortion(Distortion):
    """Capped linear distortion: needs shape >= 1 - r0."""
    kind = 'clin'
    med_name = 'Capped Linear'
    long_name = 'Capped Linear'
    documented = True
    pricing_ok = True
    strict_pricing = True
    has_mass_default = True
    _calibration_init_shape = 1.0

    def _build(self):
        self.has_mass = (self.r0 > 0)
        self.mass = self.r0

    def g(self, x):
        sl = self.shape
        return np.where(x == 0, 0, np.minimum(1, self.r0 + sl * x))

    def g_inv(self, x):
        sl = self.shape
        return np.where(x <= self.r0, 0, (x - self.r0) / sl)

    def calibrate(self, S, bs, premium_target, *, ess_sup=0.0,
                  assets=0.0, el=None, **kwargs):
        """Newton on ``∫ min(1, r0 + r·S) dx + ess_sup·r0 = premium_target``.

        The ``ess_sup * r0`` term accounts for the point mass at zero;
        ``r0`` is the minimum rate-on-line.
        """
        mass = ess_sup * self.r0

        def f(r):
            r0_rS = self.r0 + r * S
            ex = np.sum(np.minimum(1, r0_rS)) * bs + mass
            ex_prime = np.sum(np.where(r0_rS < 1, S, 0)) * bs
            return ex - premium_target, ex_prime

        shape, fx = self._newton_iterate(f, self._calibration_init_shape)
        self._finalize_calibration(shape, fx, premium_target, assets)
        return self


class LEPDistortion(Distortion):
    """Leverage-equivalent pricing distortion."""
    kind = 'lep'
    med_name = 'Lev Equiv'
    long_name = 'Leverage Equivalent Pricing'
    documented = True
    pricing_ok = True
    strict_pricing = True
    has_mass_default = True
    _calibration_init_shape = 0.25

    def _build(self):
        r = self.shape
        delta = r / (1 + r)
        d = self.r0 / (1 + self.r0)
        self._delta = delta
        self._d = d
        self._spread = delta - d
        self.has_mass = (d > 0)
        self.mass = d

    def g(self, x):
        d = self._d
        spread = self._spread
        return np.where(x == 0, 0,
                        np.minimum(1, d + (1 - d) * x
                                   + spread * np.sqrt(x * (1 - x))))

    def g_inv(self, y):
        d = self._d
        spread = self._spread
        spread2 = spread ** 2
        a = (1 - d) ** 2 + spread2
        mb = (2 * (y - d) * (1 - d) + spread2)  # -b
        c = (y - d) ** 2
        rad = np.sqrt(mb * mb - 4 * a * c)
        u = (mb - rad) / (2 * a)
        return np.where(y < d, 0, np.maximum(0, u))

    def calibrate(self, S, bs, premium_target, *, ess_sup=0.0,
                  assets=0.0, el=None, **kwargs):
        """Newton on the layer-equivalent-pricing distortion.

        Parameterised by ``d = r0/(1+r0)`` (mass) and ``δ* = r/(1+r)``;
        ``√(S(1-S))`` is precomputed since it does not depend on ``r``.
        """
        d = self.r0 / (1 + self.r0)
        rSF = np.sqrt(S * (1 - S))
        mass = ess_sup * d

        def f(r):
            spread = r / (1 + r) - d
            temp = d + (1 - d) * S + spread * rSF
            ex = np.sum(np.minimum(1, temp)) * bs + mass
            ex_prime = (1 + r) ** -2 * \
                np.sum(np.where(temp < 1, rSF, 0)) * bs
            return ex - premium_target, ex_prime

        shape, fx = self._newton_iterate(f, self._calibration_init_shape)
        self._finalize_calibration(shape, fx, premium_target, assets)
        return self


class LYDistortion(Distortion):
    """Linear yield distortion: ``r_0`` = occupancy, shape = consumption."""
    kind = 'ly'
    med_name = 'Lin Yield'
    long_name = 'Linear Yield'
    documented = True
    pricing_ok = True
    strict_pricing = True
    has_mass_default = True
    _calibration_init_shape = 1.25

    def _build(self):
        self.has_mass = (self.r0 > 0)
        self.mass = self.r0 / (1 + self.r0)

    def g(self, x):
        rk = self.shape
        return np.where(x == 0, 0,
                        (self.r0 + x * (1 + rk))
                        / (1 + self.r0 + rk * x))

    def g_inv(self, x):
        rk = self.shape
        return np.maximum(0,
                          (x * (1 + self.r0) - self.r0) / (1 + rk * (1 - x)))

    def calibrate(self, S, bs, premium_target, *, ess_sup=0.0,
                  assets=0.0, el=None, **kwargs):
        """Newton on the linear-yield distortion.

        Minimum rate-on-line is ``r0/(1+r0)``; the mass at zero is
        ``ess_sup·r0/(1+r0)``.
        """
        mass = ess_sup * self.r0 / (1 + self.r0)

        def f(rk):
            num = self.r0 + S * (1 + rk)
            den = 1 + self.r0 + rk * S
            tlam = num / den
            ex = np.sum(tlam) * bs + mass
            ex_prime = np.sum(S * (den ** -1 - num / (den ** 2))) * bs
            return ex - premium_target, ex_prime

        shape, fx = self._newton_iterate(f, self._calibration_init_shape)
        self._finalize_calibration(shape, fx, premium_target, assets)
        return self



# ===========================================================================
# Module-level helpers
# ===========================================================================


def approx_ccoc(roe, eps=1e-14, display_name=None):
    """
    Continuous approximation to the CCoC distortion at given ROE. Useful
    when a smooth distortion is needed in place of a CCoC with its mass
    at zero.
    """
    return Distortion('bitvar', roe / (1 + roe), df=[0, 1 - eps],
                      display_name=(f'aCCoC {roe:.2%}'
                                    if display_name is None
                                    else display_name))


def tvar_weights(d):
    """
    TVaR weight function for distortion ``d``: returns a callable
    ``wf(p)`` giving the weight density at TVaR threshold ``p``.

    For ccoc, ph, tvar this is given in closed form. For other kinds
    the derivative ``g'`` is differentiated numerically.
    """
    shape = d.shape
    nm = d.name

    if nm.lower().find('ccoc') >= 0:
        v = shape

        def wf(p):
            return np.where(p == 0, 1 - v,
                            np.where(p == 1, v, np.nan))
    elif nm == 'ph':
        def wf(p):
            return np.where(
                p == 1, shape,
                -shape * (shape - 1) * (1 - p) ** (shape - 1))
    elif nm == 'tvar':
        def wf(p):
            dp = p[1] - p[0]
            return np.where(np.abs(p - shape) < dp, 1, 0)
    else:
        def wf(p):
            gprime = d.g_prime(1 - p)
            return (1 - p) * np.gradient(gprime, p)

    return wf


def p_to_parameters(p):
    """
    Standard distortion parameters equivalent to TVaR at level ``p``.

    Returned dict has entries for ``ccoc``, ``ph``, ``wang``, ``dual``,
    and ``tvar``.
    """
    ans = {}
    ans['ccoc'] = p
    ans['ph'] = (1 - p) / (1 + p)
    ans['wang'] = 2 ** 0.5 * phi_inv((1 + p) / 2)
    ans['dual'] = (1 + p) / (1 - p)
    ans['tvar'] = p
    return ans


def consistent_distortions(p):
    """Five representative distortions, calibrated to TVaR-p."""
    params = p_to_parameters(p)
    ans = {}
    for k, sh in params.items():
        # Distortion.ccoc takes discount, but Distortion('ccoc', shape)
        # takes return
        if k == 'ccoc':
            sh = sh / (1 - sh)
        ans[k] = Distortion(k, sh)
    return ans
