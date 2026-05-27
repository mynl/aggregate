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
from functools import cached_property
from io import StringIO
import logging

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.interpolate import interp1d
from scipy.optimize import brentq
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

import hashlib

from .constants import FIG_H, FIG_W
from .random_agg import RANDOM


def _short_hash(s):
    """
    machine independent hash of a string s
    """
    ho = hashlib.md5()
    ho.update(s.encode('utf-8'))
    hv = ho.hexdigest()[:8].upper()
    return hv


# Canonical display order for distortion names. Used as a pandas
# ``CategoricalDtype`` to keep ``distortion_df`` / pricing exhibits sorted
# in this order without ad-hoc reordering.
DISTORTION_ORDER = [
    'ccoc', 'ph', 'wang', 'dual', 'tvar', 'wtdtvar',
    'lep', 'ly', 'clin', 'tt', 'cll', 'bitvar', 'blend',
]
DISTORTION_DTYPE = pd.CategoricalDtype(categories=DISTORTION_ORDER, ordered=True)


# Default grid resolution for ``density_df`` / trapezoidal moment integrals.
# Each Distortion instance can override with ``self._density_n_points``.
_DISTORTION_DENSITY_N = 101


# Public surface for ``from aggregate.spectral import *``. Distortion subclasses
# (PHDistortion, WangDistortion, ...) are deliberately NOT listed -- the
# factory ``Distortion('ph', a=0.7)`` is the primary API; direct subclass
# construction (``PHDistortion(a=0.7)``) is still supported via explicit
# import: ``from aggregate.spectral import PHDistortion``.
__all__ = [
    'Distortion',
    'approx_ccoc',
    'tvar_weights',
    'p_to_parameters', 'consistent_distortions',
    'convex_distortion', 'bagged_distortion', 'convex_example',
]


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
    # natural name of the scalar shape parameter, used by the base
    # ``__init__`` to accept ``Kind(name, a=0.7)`` in addition to the
    # positional form. ``None`` for multi-parameter kinds, which override
    # ``__init__`` entirely.
    param_name: str | None = None

    # legacy attribute aliases preserved for back-compat with callers
    # that read class-level lists directly.
    @classmethod
    def _kinds_ordered(cls):
        return tuple(k for k, v in cls._registry.items() if v.documented)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.kind:
            Distortion._registry[cls.kind] = cls
        # Auto-wrap subclass ``_build`` so the quartet cache is invalidated
        # after every reconfiguration (construction, property setter,
        # calibration finalisation all route through ``_build``).
        if '_build' in cls.__dict__:
            user_build = cls.__dict__['_build']

            def _wrapped_build(self, _orig=user_build):
                _orig(self)
                self._invalidate_cache()

            _wrapped_build.__name__ = '_build'
            _wrapped_build.__qualname__ = f'{cls.__qualname__}._build'
            _wrapped_build.__doc__ = user_build.__doc__
            cls._build = _wrapped_build

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

    def __init__(self, name=None, shape=None, *, display_name='', **natural):
        """
        Scalar-shape constructor used by ``ph``, ``wang``, ``dual``, ``tvar``.

        Multi-parameter kinds (bitvar, wtdtvar, beta, power, minimum,
        mixture) and mass-at-zero kinds (ccoc, cll, clin, ly, lep) override
        ``__init__`` entirely; see their classes for the natural-kwarg
        signatures.

        Parameters
        ----------
        name : str, optional
            Distortion kind, e.g. ``'ph'``. ``'roe'`` is accepted as a
            legacy alias for ``'ccoc'`` (handled by ``__new__``). When
            constructing a subclass directly (``PHDistortion(a=0.7)``)
            ``name`` defaults to the subclass's ``kind`` attribute.
        shape : float, optional
            Positional shape parameter. May also be passed by its natural
            name (e.g. ``a=0.7`` for ph); passing both raises ``TypeError``.
        display_name : str, optional
            Override label; ``str(d)`` returns this if set, else the kind
            name.
        **natural : float
            Accept the kind's natural parameter name (``a``, ``lam``, ``b``,
            ``p``) as a keyword. Unknown kwargs raise ``TypeError``.
        """
        if name is None:
            name = type(self).kind
        if name == 'roe':
            name = 'ccoc'
        pn = type(self).param_name
        if pn is not None and pn in natural:
            if shape is not None:
                raise TypeError(
                    f'Pass {pn}= or positional shape, not both')
            shape = natural.pop(pn)
        if natural:
            raise TypeError(
                f'{type(self).__name__}: unexpected keyword arguments '
                f'{list(natural)}')
        self._name = name
        self.shape = shape
        self.display_name = display_name
        self._common_init()
        self._build()

    def _common_init(self):
        """Initialise the audit/state fields shared by every subclass.

        Called from every ``__init__`` before ``_build``. Subclass
        ``_build`` may then overwrite ``has_mass`` / ``mass`` /
        ``standard_shape``.
        """
        self.has_mass = False
        self.mass = 0.0
        self.standard_shape = np.nan
        self.error = 0.0
        self.premium_target = 0.0
        self.assets = 0.0

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

    def _id_fields(self):
        """Tuple of identifying attribute values used by ``id()``.

        The default returns ``(_name, shape, display_name)`` — appropriate
        for scalar-shape kinds. Subclasses with extra structural state
        (``r0``, multi-parameter kinds, list-of-distortions kinds) override
        this method.
        """
        return (self._name, self.shape, self.display_name)

    def id(self):
        """Unique ID as a short string, based on the structural fields
        returned by :meth:`_id_fields`."""
        return _short_hash(str(self._id_fields()))

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
    # Quartet: info / describe / stats_df / density_df
    # ------------------------------------------------------------------
    #
    # Each is a lazy ``cached_property``. The cached value lands in
    # ``self.__dict__[name]``; ``_invalidate_cache()`` pops the keys. The base
    # ``_build()`` (subclass hook) calls ``_invalidate_cache()`` at the end,
    # so calibrate-then-read returns fresh tables (property setters and
    # ``_finalize_calibration`` both route through ``_build``).
    #
    # Conceptual setup: ``g : [0,1] -> [0,1]`` with ``g(0)=0``, ``g(1)=1``,
    # monotone non-decreasing is structurally a CDF, so ``g`` and ``g_inv``
    # induce two natural distributions ``D_g`` and ``D_g_inv``. Moments are
    # computed via tail-integral form ``E[X] = 1 - int_0^1 g(x) dx`` which
    # is robust at kinks (no ``g'`` evaluation on the grid).

    _density_n_points = _DISTORTION_DENSITY_N

    @cached_property
    def info(self):
        """Multi-line string summarising the distortion (lazy)."""
        return self._compute_info()

    @cached_property
    def describe(self):
        """Compact pair-column DataFrame (D_g, D_g_inv) plus checks (lazy)."""
        return self._compute_describe()

    @cached_property
    def stats_df(self):
        """Single-column DataFrame of D_g statistics with closed-form column
        and error column where analytics exist (lazy)."""
        return self._compute_stats_df()

    @cached_property
    def density_df(self):
        """Grid DataFrame: x, g, g_inv, g_dual, g_dual_inv, g_prime,
        g_dual_prime, kusuoka. Grid is ``_density_n_points`` uniform points
        on [0,1] with subclass-supplied knots spliced in (lazy)."""
        return self._compute_density_df()

    def _invalidate_cache(self):
        """Drop cached quartet values; called by ``_build``."""
        for k in ('info', 'describe', 'stats_df', 'density_df',
                  '_grid_moments'):
            self.__dict__.pop(k, None)

    # --- subclass hooks (defaults return empty / no overrides) -----------

    def _density_knots(self):
        """Subclass hook: extra x-values to splice into the uniform grid.

        Override for kinds with kinks in ``g`` (TVaR at ``1-p``, BiTVaR at
        ``{1-p0, 1-p1}``, WtdTVaR at each ``1-p``, etc.) so ``g_prime`` near
        the kink and the kusuoka column land on grid points.
        """
        return []

    def _describe_closed_form(self):
        """Subclass hook: dict mapping row name -> analytic moment value.

        Keys may include ``'mean'``, ``'var'``, ``'std'``, ``'cv'``,
        ``'skew'``. Missing keys show as ``NaN`` in the closed-form column.
        Multi-knot kinds (bitvar, wtdtvar, minimum, mixture) return ``{}``.
        """
        return {}

    def _kusuoka_summary(self):
        """Subclass hook: ``(mu_atom_0, mu_atom_1, has_interior_atoms)``.

        ``mu_atom_0`` is the atom of the Kusuoka spectral measure :math:`\\mu`
        at ``p=0`` (the mean-component weight). ``mu_atom_1`` is the atom at
        ``p=1`` (the max/ess-sup component weight); this equals the existing
        ``self.mass`` (i.e. the jump of ``g`` at ``x=0``). The bool indicates
        whether ``\\mu`` has any atoms in ``(0, 1)``. Default: continuous
        ``\\mu`` apart from the existing ``self.mass`` atom at ``p=1``.
        """
        return (0.0, float(getattr(self, 'mass', 0.0) or 0.0), False)

    def _kusuoka_atoms(self):
        """Subclass hook: list of ``(p_atom, mass)`` pairs for atoms of
        the Kusuoka measure :math:`\\mu` on ``[0, 1]``. Default derives
        boundary atoms from :meth:`_kusuoka_summary`; subclasses with
        interior atoms (TVaR, BiTVaR, WtdTVaR, CCoC, combos) override.
        """
        mu0, mu1, _ = self._kusuoka_summary()
        atoms = []
        if mu0 > 0.0:
            atoms.append((0.0, float(mu0)))
        if mu1 > 0.0:
            atoms.append((1.0, float(mu1)))
        return atoms

    def _kusuoka_density(self, p):
        """Subclass hook: continuous density of :math:`\\mu` at ``p``.

        Default: numerical via the identity
        :math:`\\mu_\\text{density}(p) = -(1-p)\\,g''(1-p)`. The continuous
        density covers what's left after the atoms in
        :meth:`_kusuoka_atoms`. Returns ``np.zeros_like(p)`` if numerical
        differentiation produces NaN/inf values (e.g. at endpoints).
        """
        p = np.asarray(p, dtype=float)
        s = 1.0 - p
        eps = 1e-6
        # central difference for g''; clamp to [eps, 1-eps] to keep finite
        s_clip = np.clip(s, eps, 1.0 - eps)
        gpp = (self._broadcast(self.g_prime(s_clip + eps), s_clip)
               - self._broadcast(self.g_prime(s_clip - eps), s_clip)) / (2 * eps)
        out = -(1.0 - p) * gpp
        return np.where(np.isfinite(out) & (out > 0), out, 0.0)

    # --- compute methods ------------------------------------------------

    def _build_grid(self):
        """Sorted unique grid: uniform ``_density_n_points`` on [0,1] plus
        ``_density_knots()`` and the endpoints 0 and 1. Each interior knot
        is wrapped by ``(knot - eps, knot, knot + eps)`` so a piecewise-
        linear kink is visible in ``g_prime`` on either side. If ``g`` has
        an atom at ``x=0`` (``self.mass > 0``, equivalently the Kusuoka
        atom at ``p=1`` > 0), splice in a tiny right-neighbour so
        trapezoidal integration sees the jump immediately."""
        n = self._density_n_points
        base = np.linspace(0.0, 1.0, n)
        eps = 1e-12
        extra = []
        for k in self._density_knots():
            k = float(k)
            if 0.0 < k < 1.0:
                extra.extend([max(0.0, k - eps), k, min(1.0, k + eps)])
            elif 0.0 <= k <= 1.0:
                extra.append(k)
        # Atom at x=0: ensure a knot at eps so trapz captures the jump.
        if float(getattr(self, 'mass', 0.0) or 0.0) > 0.0:
            extra.append(eps)
        # Symmetric boundary at x=1: g_inv may jump at y=1 (whenever g
        # reaches 1 before s=1, e.g. TVaR/BiTVaR/WtdTVaR with any p>0).
        # Splice 1-eps so trapezoidal integration of g_inv doesn't
        # over-count a linear ramp on the last bin.
        extra.append(1.0 - eps)
        merged = np.unique(np.concatenate([base, np.asarray(extra)]))
        return merged

    @staticmethod
    def _broadcast(values, x):
        """Cast a (possibly scalar) g/g_prime return to a 1-d array of
        the same length as ``x``. Some kinds return a constant from
        ``g_prime`` (CCoC) which would otherwise fail to reshape."""
        arr = np.asarray(values, dtype=float)
        if arr.shape == () or arr.size == 1:
            return np.full(x.shape, float(arr))
        return arr.reshape(x.shape)

    def _compute_density_df(self):
        """Build ``density_df`` on the spliced grid."""
        x = self._build_grid()
        # g and g_inv may be interp1d objects (bitvar/wtdtvar) returning
        # 0-d arrays; cast to 1-d.
        g = self._broadcast(self.g(x), x)
        try:
            g_inv = self._broadcast(self.g_inv(x), x)
        except NotImplementedError:
            # MixtureDistortion: numerical inverse on the grid via brentq.
            g_inv = self._numerical_g_inv(x, g)
        g_dual = 1.0 - self._broadcast(self.g(1.0 - x), x)
        try:
            g_dual_inv = 1.0 - self._broadcast(self.g_inv(1.0 - x), x)
        except NotImplementedError:
            g_dual_inv = 1.0 - self._numerical_g_inv(1.0 - x, 1.0 - g_dual)
        g_prime = self._broadcast(self.g_prime(x), x)
        # g_dual_prime(x) = g_prime(1 - x). Name order matters: this is the
        # derivative of the dual, NOT (g_prime) made dual.
        g_dual_prime = self._broadcast(self.g_prime(1.0 - x), x)
        kusuoka = self._kusuoka_masses(x)
        return pd.DataFrame({
            'g': g,
            'g_inv': g_inv,
            'g_dual': g_dual,
            'g_dual_inv': g_dual_inv,
            'g_prime': g_prime,
            'g_dual_prime': g_dual_prime,
            'kusuoka': kusuoka,
        }, index=pd.Index(x, name='x'))

    def _numerical_g_inv(self, qs, gs):
        """Fallback inverse via brentq on ``g``; used when ``self.g_inv``
        raises ``NotImplementedError`` (e.g. ``MixtureDistortion``). Solves
        ``g(u) = q`` for each ``q`` in ``qs``."""
        def g_scalar(u):
            v = self.g(u)
            v = np.asarray(v).ravel()
            return float(v[0]) if v.size else 0.0
        out = np.empty_like(qs, dtype=float)
        for i, q in enumerate(qs):
            if q <= 0.0:
                out[i] = 0.0
            elif q >= 1.0:
                out[i] = 1.0
            else:
                try:
                    out[i] = brentq(lambda u, q=q: g_scalar(u) - q,
                                    0.0, 1.0, xtol=1e-10)
                except (ValueError, RuntimeError):
                    out[i] = np.nan
        return out

    def _kusuoka_masses(self, x):
        """Bucket masses of the Kusuoka spectral measure :math:`\\mu` on
        ``[0, 1]``, expressed on the ``x = 1 - p`` grid.

        For each grid point ``x_i``, the entry is :math:`\\mu([p_{i-},
        p_{i+}])` where ``p_{i\\pm} = 1 - (x_i \\pm b/2)`` are the bucket
        edges in ``p``-space (with ``b`` the local bucket width). Atoms in
        :math:`\\mu` (TVaR's Dirac at ``p``, CCoC's Diracs at 0 and 1,
        WtdTVaR's discrete weights, PH's atom at ``p=0``, etc.) land in
        the bucket containing the corresponding ``x = 1 - p_atom``.

        The continuous part is integrated via the midpoint rule using
        :meth:`_kusuoka_density`. Atoms come from :meth:`_kusuoka_atoms`.

        The continuous portion of the column is renormalised so that the
        total (atoms + continuous) sums to ``1``. Atom values are
        preserved exactly; the continuous-part bucket masses are scaled
        uniformly. This means kinds with singular density (Wang, Beta,
        PH near ``p=1``) still produce a column that sums to 1 even
        though the bucket-by-bucket relative weights may be off by a few
        percent near the singularity.
        """
        x = np.asarray(x, dtype=float)
        # bucket edges in x-space
        x_edges_lo = np.empty_like(x)
        x_edges_hi = np.empty_like(x)
        x_edges_lo[0] = 0.0
        x_edges_hi[-1] = 1.0
        mids = (x[:-1] + x[1:]) / 2.0
        x_edges_hi[:-1] = mids
        x_edges_lo[1:] = mids
        bucket_widths = x_edges_hi - x_edges_lo
        # continuous part: density at x_i * bucket width (midpoint rule)
        p = 1.0 - x
        density = self._kusuoka_density(p)
        cont = density * bucket_widths
        # Sanitise the continuous part: drop NaN / negatives.
        cont = np.where(np.isfinite(cont) & (cont >= 0), cont, 0.0)
        # Atom contributions: collected separately so the renormalisation
        # below leaves them exact.
        atom_mass = np.zeros_like(x)
        atom_total = 0.0
        for p_atom, w in self._kusuoka_atoms():
            x_atom = 1.0 - float(p_atom)
            idx = int(np.argmin(np.abs(x - x_atom)))
            atom_mass[idx] += float(w)
            atom_total += float(w)
        # Renormalise continuous bucket masses so total (atoms+cont) = 1.
        cont_total = cont.sum()
        target_cont = max(0.0, 1.0 - atom_total)
        if cont_total > 0 and target_cont > 0:
            cont = cont * (target_cont / cont_total)
        return atom_mass + cont

    def _ensure_grid_moments(self):
        """Compute and cache the trapezoidal integrals used by both
        ``describe`` and ``stats_df``. One pass over ``density_df``."""
        if '_grid_moments' in self.__dict__:
            return self.__dict__['_grid_moments']
        df = self.density_df
        x = df.index.to_numpy(dtype=float)
        g = df['g'].to_numpy()
        g_inv = df['g_inv'].to_numpy()
        # E[X] = 1 - int g  (X is a CDF on [0,1])
        int_g = float(np.trapezoid(g, x))
        int_g_inv = float(np.trapezoid(g_inv, x))
        mean_g = 1.0 - int_g
        mean_g_inv = 1.0 - int_g_inv
        # E[X^2] = int 2x (1 - g) dx for X with CDF g on [0,1]
        ex2_g = float(np.trapezoid(2.0 * x * (1.0 - g), x))
        ex2_g_inv = float(np.trapezoid(2.0 * x * (1.0 - g_inv), x))
        # E[X^3] = int 3 x^2 (1 - g) dx
        ex3_g = float(np.trapezoid(3.0 * x * x * (1.0 - g), x))
        ex3_g_inv = float(np.trapezoid(3.0 * x * x * (1.0 - g_inv), x))
        var_g = max(ex2_g - mean_g ** 2, 0.0)
        var_g_inv = max(ex2_g_inv - mean_g_inv ** 2, 0.0)
        std_g = var_g ** 0.5
        std_g_inv = var_g_inv ** 0.5
        cv_g = std_g / mean_g if mean_g > 0 else np.nan
        cv_g_inv = std_g_inv / mean_g_inv if mean_g_inv > 0 else np.nan
        # third central moment, then skewness
        m3_g = ex3_g - 3.0 * mean_g * ex2_g + 2.0 * mean_g ** 3
        m3_g_inv = ex3_g_inv - 3.0 * mean_g_inv * ex2_g_inv + 2.0 * mean_g_inv ** 3
        skew_g = m3_g / std_g ** 3 if std_g > 0 else np.nan
        skew_g_inv = m3_g_inv / std_g_inv ** 3 if std_g_inv > 0 else np.nan
        moments = {
            'int_g': int_g, 'int_g_inv': int_g_inv,
            'mean_g': mean_g, 'mean_g_inv': mean_g_inv,
            'var_g': var_g, 'var_g_inv': var_g_inv,
            'std_g': std_g, 'std_g_inv': std_g_inv,
            'cv_g': cv_g, 'cv_g_inv': cv_g_inv,
            'skew_g': skew_g, 'skew_g_inv': skew_g_inv,
            'p_equiv': 2.0 * int_g - 1.0,
            'loading': int_g - 0.5,
        }
        self.__dict__['_grid_moments'] = moments
        return moments

    def _compute_stats_df(self):
        """Single-column ``D_g`` statistics + closed-form / error columns.

        Notes
        -----
        ``gini`` and ``p_equiv`` are numerically identical (both equal
        ``2 * int_0^1 g(x) dx - 1``) but actuaries read them differently:
        ``gini`` is the *positive concavity* of ``g`` (how much load above
        the actuarial price the distortion adds to a U[0,1] risk);
        ``p_equiv`` is the *TVaR level* whose pricing of U[0,1] matches
        ``g`` (inversion of :func:`p_to_parameters`). Both rows are kept.
        """
        m = self._ensure_grid_moments()
        cf = self._describe_closed_form()
        rows = [
            ('mean',    m['mean_g'],   cf.get('mean',   np.nan)),
            ('var',     m['var_g'],    cf.get('var',    np.nan)),
            ('std',     m['std_g'],    cf.get('std',    np.nan)),
            ('cv',      m['cv_g'],     cf.get('cv',     np.nan)),
            ('skew',    m['skew_g'],   cf.get('skew',   np.nan)),
            ('gini',    m['p_equiv'],  cf.get('p_equiv', np.nan)),
            ('p_equiv', m['p_equiv'],  cf.get('p_equiv', np.nan)),
            ('loading', m['loading'],  cf.get('loading', np.nan)),
        ]
        df = pd.DataFrame(rows, columns=['stat', 'D_g', 'closed_form'])
        df['error'] = df['D_g'] - df['closed_form']
        df = df.set_index('stat')
        # Atoms section: one row per Dirac atom of the Kusuoka measure mu.
        # Index labels are ``mu_<p:.3f>``; closed_form holds the p value
        # itself (so the column carries useful information even though
        # "closed-form moment" doesn't quite apply); error is NaN.
        atoms = self._kusuoka_atoms()
        if atoms:
            # Sort by p ascending so mu_0.000 (mean) comes before mu_1.000 (max).
            atoms_sorted = sorted(atoms, key=lambda pw: pw[0])
            atom_idx = [f'mu_{p:.3f}' for p, _ in atoms_sorted]
            atom_df = pd.DataFrame({
                'D_g': [float(w) for _, w in atoms_sorted],
                'closed_form': [float(p) for p, _ in atoms_sorted],
                'error': [np.nan] * len(atoms_sorted),
            }, index=pd.Index(atom_idx, name='stat'))
            df = pd.concat([df, atom_df])
        # TODO: entropy. Deferred in v1 -- mass-at-zero kinds diverge;
        # piecewise-linear kinds require per-kind exact summation.
        return df

    def _compute_describe(self):
        """Pair-column ``(D_g, D_g_inv)`` table plus checks block.

        The checks block at the bottom reports three approximate identities
        a user can read at a glance:

        * ``E[D_g] + E[D_g_inv] = 1`` -- the two means partition the unit
          square; large error here means the trapezoidal grid mis-sized the
          atoms of ``D_g`` / ``D_g_inv``.
        * ``g(g_inv(0.5)) = 0.5`` -- round-trip sanity check: ``g`` and
          ``g_inv`` are mutual inverses (within the grid resolution). For
          ``MixtureDistortion`` where ``g_inv`` is not in closed form, the
          numerical brentq fallback drives this check.
        * ``g(0) = 0, g(1) = 1`` -- the endpoint contract every distortion
          must satisfy.
        """
        m = self._ensure_grid_moments()
        cf = self._describe_closed_form()
        rows = [
            ('mean', m['mean_g'], m['mean_g_inv'],
             cf.get('mean', np.nan)),
            ('std',  m['std_g'],  m['std_g_inv'],
             cf.get('std',  np.nan)),
            ('cv',   m['cv_g'],   m['cv_g_inv'],
             cf.get('cv',   np.nan)),
            ('skew', m['skew_g'], m['skew_g_inv'],
             cf.get('skew', np.nan)),
            ('gini', m['p_equiv'], -m['p_equiv'],
             cf.get('p_equiv', np.nan)),
        ]
        moments = pd.DataFrame(
            rows, columns=['stat', 'D_g', 'D_g_inv', 'closed_form'])
        moments['error'] = moments['D_g'] - moments['closed_form']
        moments = moments.set_index('stat')
        # Kusuoka summary block: mean-component / max-component atoms of mu
        # and the interior-atoms flag. Same shape for every kind.
        mu0, mu1, interior = self._kusuoka_summary()
        kusuoka_summary = pd.DataFrame({
            'D_g': [float(mu0), float(mu1), bool(interior)],
            'D_g_inv': [np.nan, np.nan, np.nan],
            'closed_form': [np.nan, np.nan, np.nan],
            'error': [np.nan, np.nan, np.nan],
        }, index=pd.Index(
            ['mean_mass', 'max_mass', 'interior_atoms'], name='stat'))
        # Checks block. Each row reports the realised value; the user reads
        # them as approximate-identity tests against the target.
        try:
            g_inv_half = float(self.g_inv(0.5))
            g_round = float(self.g(g_inv_half))
        except NotImplementedError:
            g_round = np.nan
        sum_means = m['mean_g'] + m['mean_g_inv']
        checks = pd.DataFrame({
            'D_g': [sum_means, g_round, float(np.asarray(self.g(0.0)).ravel()[0])],
            'D_g_inv': [np.nan, np.nan, float(np.asarray(self.g(1.0)).ravel()[0])],
            'closed_form': [1.0, 0.5, 0.0],
            'error': [sum_means - 1.0, g_round - 0.5,
                      float(np.asarray(self.g(0.0)).ravel()[0])],
        }, index=pd.Index(
            ['E[D_g]+E[D_g_inv]', 'g(g_inv(0.5))', 'g(0), g(1)'], name='stat'))
        return pd.concat([moments, kusuoka_summary, checks])

    def _compute_info(self):
        """Multi-line summary string mirroring Aggregate/Portfolio.info."""
        lines = [f'Distortion: {self.name}']
        kind = self._name
        long = getattr(type(self), 'long_name', kind)
        lines.append(f'  kind:           {kind}  ({long})')
        if self.display_name:
            lines.append(f'  display:        {self.display_name}')
        pn = getattr(type(self), 'param_name', None)
        if pn is not None:
            try:
                val = getattr(self, pn)
                lines.append(f'  {pn:<15s} {float(val):.4g}')
            except (AttributeError, TypeError):
                pass
        # Multi-param / composite kinds: surface their structural state.
        for attr in ('r', 'd', 'p0', 'p1', 'w1', 'r0', 'slope',
                     'a', 'b', 'x0', 'x1', 'alpha'):
            if pn == attr:
                continue
            if hasattr(self, attr):
                v = getattr(self, attr)
                if isinstance(v, (int, float, np.floating)) and not callable(v):
                    lines.append(f'  {attr:<15s} {float(v):.4g}')
        if hasattr(self, '_ps') and hasattr(self, '_wts'):
            lines.append(f'  ps              [{len(self._ps)} knots] '
                         f'min={self._ps.min():.3g}, '
                         f'max={self._ps.max():.3g}')
            lines.append(f'  wts             sum={self._wts.sum():.4g}')
            lines.append('  (see stats_df / tvar_info_df for per-knot detail)')
        if hasattr(self, '_distortions'):
            n = len(self._distortions)
            members = ', '.join(d.name for d in self._distortions)
            lines.append(f'  members         [{n}] {members}')
            if hasattr(self, '_wts') and self._wts is not None:
                wts_str = ', '.join(f'{w:.3g}' for w in self._wts)
                lines.append(f'  wts             [{wts_str}]')
        mu0, mu1, interior = self._kusuoka_summary()
        lines.append(f'  mu({{0}})         {mu0:.4g}')
        lines.append(f'  mu({{1}})         {mu1:.4g}')
        lines.append(f'  interior atoms  {interior}')
        lines.append(f'  strict-pricing  {getattr(type(self), "strict_pricing", False)}')
        lines.append(f'  id              {self.id()}')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, xs=None, n=101, both=True, ax=None, plot_points=True,
             scale='linear', c=None, c_dual=None, size='small', **kwargs):
        """
        Plot the distortion.

        Parameters
        ----------
        xs : array_like, optional
            x values; defaults to ``density_df.index`` (linear) or a
            log-spaced grid (return scale).
        n : int
            Grid size for ``scale='return'`` (ignored on linear scale, which
            uses the cached ``density_df`` grid).
        both : bool
            Also plot ``g_dual``.
        ax : matplotlib.axes.Axes, optional
            Existing Axes; if ``None`` a new figure is created.
        plot_points : bool
            Legacy flag (was used by the removed ``ConvexDistortion``).
        scale : {'linear', 'return'}
            Linear plot on ``[0, 1]^2`` or log-log return-period scale.
        size : str or float
            ``'small'`` / ``'large'`` figure preset or a numeric side length.
        **kwargs
            Forwarded to ``ax.plot``.

        Notes
        -----
        On linear scale the curve is read straight from ``density_df`` so the
        knot splicing (TVaR kink, BiTVaR/WtdTVaR knots, mass-at-0 epsilon)
        is reflected directly in the plot.
        """
        assert scale in ['linear', 'return']

        if scale == 'return':
            xs = 10 ** np.linspace(-10, 0, n)
            y1 = self.g(xs)
            y2 = self.g_dual(xs) if both else None
        else:
            if xs is None:
                df = self.density_df
                xs = df.index.to_numpy()
                y1 = df['g'].to_numpy()
                y2 = df['g_dual'].to_numpy() if both else None
            else:
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

        ax.set(title=self.name, aspect='equal')
        if scale == 'linear':
            ax.set(xticks=np.linspace(0, 1, 6),
                   yticks=np.linspace(0, 1, 6))
        if both:
            ax.legend(loc='upper left', fontsize='x-small')
        return ax

    # ------------------------------------------------------------------
    # Static factory shortcuts
    # ------------------------------------------------------------------

    @staticmethod
    def tvar(p):
        """Construct a TVaR distortion at level ``p``."""
        return Distortion('tvar', p=p, display_name=f'TVaR({p:.3g})')

    @staticmethod
    def max():
        """TVaR at ``p=1`` (the max)."""
        return Distortion('tvar', p=1.0, display_name='max')

    @staticmethod
    def mean():
        """TVaR at ``p=0`` (the mean)."""
        return Distortion('tvar', p=0.0, display_name='mean')

    @staticmethod
    def wang(lam):
        """Construct a Wang distortion with parameter ``lam``."""
        return Distortion('wang', lam=lam, display_name=f'Wang({lam:.3g})')

    @staticmethod
    def ph(a):
        """Construct a proportional-hazard distortion with parameter ``a``."""
        return Distortion('ph', a=a, display_name=f'PH({a:.3g})')

    @staticmethod
    def dual(b):
        """Construct a dual-moment distortion with parameter ``b``."""
        return Distortion('dual', b=b, display_name=f'dual({b:.3g})')

    @staticmethod
    def bitvar(p0, p1, w=0.5):
        """
        Construct a BiTVaR with :math:`p_0 < p_1` and weight ``w`` on ``p_1``.
        Degenerate combinations collapse to a TVaR.
        """
        if p0 == p1 or w == 0:
            return Distortion.tvar(p0)
        if w == 1:
            return Distortion.tvar(p1)
        return Distortion('bitvar', p0=p0, p1=p1, w1=w,
                          display_name=f'bitvar({p0:.3g}, {p1:.3g}; {w:.3g})')

    @staticmethod
    def ccoc(d):
        """
        Construct a CCoC distortion from discount factor ``d``. The
        default constructor takes return ``r`` instead; ``d = r / (1 + r)``.
        """
        r = d / (1. - d)
        return Distortion('ccoc', r=r, display_name=f'ccoc({r:.3g})')

    @staticmethod
    def minimum(distortion_list):
        """Construct a Distortion that is the pointwise minimum of others."""
        return Distortion('minimum', distortions=distortion_list,
                          display_name=f'minimum({len(distortion_list)})')

    @staticmethod
    def mixture(distortion_list, weights=None):
        """Construct a weighted mixture of distortions."""
        return Distortion('mixture', distortions=distortion_list,
                          wts=weights,
                          display_name=f'mixture({len(distortion_list)})')

    @staticmethod
    def beta(a, b):
        """Construct a beta distortion with parameters ``a`` and ``b``."""
        return Distortion('beta', a=a, b=b,
                          display_name=f'beta({a:.3f}, {b:.3f})')

    @staticmethod
    def power(alpha, x0, x1):
        """Construct a power distortion with ``x0 < x1`` and exponent ``alpha``."""
        return Distortion('power', x0=x0, x1=x1, alpha=alpha,
                          display_name=f'power({alpha:.3f}, {x0:.3f}, {x1:.3f})')

    @staticmethod
    def distortions_from_params(params, index, r0=0.025, df=5.5,
                                pricing=True, strict=True):
        """
        Construct a dict of distortions from a calibration parameter table.

        Parameters
        ----------
        params : pandas.DataFrame
            Indexed by ``(group, kind)`` with a ``param`` column;
            one entry per available distortion kind. Called by
            ``Portfolio.calibrate_distortions``.
        index : hashable
            Group index used to slice ``params``.
        r0 : float, optional
            ``r0`` parameter for mass-at-zero kinds (cll, clin, lep, ly).
        df : float, optional
            Reserved for future kinds with a second-parameter slot.
        pricing, strict : bool
            Forwarded to :meth:`available_distortions`.

        Returns
        -------
        dict[str, Distortion]
        """
        temp = params.loc[index, :]
        dists = {}
        for dn in Distortion.available_distortions(pricing=pricing, strict=strict):
            param = float(temp.loc[dn, 'param'])
            if dn in ('ccoc',):
                dists[dn] = Distortion(dn, r=param)
            elif dn in ('cll', 'clin', 'lep', 'ly'):
                kw = {'r0': r0}
                # the natural shape kwarg name varies by kind
                pn = Distortion._registry[dn].param_name
                kw[pn] = param
                dists[dn] = Distortion(dn, **kw)
            else:
                pn = Distortion._registry[dn].param_name
                dists[dn] = Distortion(dn, **{pn: param})
        return dists

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
        return Distortion('wtdtvar', ps=ps, wts=wts,
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
        ``S = ser[::-1].cumsum().shift(1, fill_value=0)[::-1]``
        and ``S_calculation='forwards'`` computes
        ``S = 1 - ser.cumsum()``.

        If ``ser.sum() < 1``, the backwards survival calculation drops
        the missing probability mass entirely. The forwards calculation
        instead carries the missing mass in the tail, equivalent for dx
        pricing to placing it at the largest observed ``x``.
        If ``ser` is normalized if these methods should agree. If ``ser``
        is not normalized look in the mirror and ask yourself what
        you are doing. Backwards is more reliable than the forward form
        for thin-tailed risks because the tail details are lost by the
        ``1 - cumsum`` calculation.

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
    Constant cost-of-capital distortion. Parameterised by either the
    target return ``r`` or the discount intercept ``d = r/(1+r)``; the
    linear form is :math:`g(x) = \\min(1, d + v\\,x)` with
    :math:`v = 1/(1+r) = 1-d`. The legacy alias ``'roe'`` maps to this
    kind via :meth:`Distortion.__new__`.
    """
    kind = 'ccoc'
    med_name = 'Const CoC'
    long_name = 'Constant CoC'
    documented = True
    pricing_ok = True
    strict_pricing = True
    _calibration_init_shape = 0.25

    def __init__(self, name='ccoc', *, d=None, r=None, display_name=''):
        """
        Construct a CCoC distortion. Pass exactly one of ``d`` or ``r``.

        Parameters
        ----------
        name : str
            Distortion kind (``'ccoc'``). Present only because the factory
            ``Distortion('ccoc', ...)`` always passes the name.
        d : float, optional
            Discount intercept ``d = r/(1+r)``.
        r : float, optional
            Target return ``r``. Newton calibration iterates on this.
        display_name : str, optional
            Override label.

        Raises
        ------
        TypeError
            If neither or both of ``d`` and ``r`` are provided. (CCoC
            takes natural parameters as keyword-only to keep the meaning
            of a positional float explicit at the call site.)
        """
        if (d is None) == (r is None):
            raise TypeError(
                "CCoCDistortion requires exactly one of d= (discount) "
                "or r= (return); got "
                f"d={d!r}, r={r!r}")
        if r is None:
            r = d / (1.0 - d)
        else:
            d = r / (1.0 + r)
        self._name = 'ccoc' if name == 'roe' else name
        self.r = r
        self.d = d
        self.v = 1.0 - d
        self.shape = r
        self.display_name = display_name
        self._common_init()
        self._build()

    def _build(self):
        r = self.shape
        # keep r/d/v in sync after calibration writes self.shape
        self.r = r
        self.d = r / (1.0 + r)
        self.v = 1.0 - self.d
        self.has_mass = (self.d > 0)
        self.mass = self.d
        self.standard_shape = self.d

    def _id_fields(self):
        return (self._name, self.r, self.d, self.display_name)

    def g(self, x):
        d, v = self.d, self.v
        return np.where(x == 0, 0, np.minimum(1, d + v * x))

    def g_inv(self, x):
        d, v = self.d, self.v
        return np.where(x <= d, 0, (x - d) / v)

    def g_prime(self, x):
        return self.v

    def calibrate(self, S, bs, premium_target, *, ess_sup=0.0,
                  assets=0.0, el=None, **kwargs):
        """
        Closed-form calibration: ``r = (premium - el) / (assets - premium)``.

        Notes
        -----
        No iteration. Mutates ``self.shape`` (and ``self.r`` /
        ``self.d`` / ``self.v`` via ``_build``) in place.
        """
        assert el is not None and premium_target, \
            'CCoC calibration requires el and a non-zero premium_target'
        r = (premium_target - el) / (assets - premium_target)
        self._finalize_calibration(r, 0.0, premium_target, assets)
        return self

    def _describe_closed_form(self):
        """Closed-form moments of D_g for ``g(x) = d + (1-d)x`` on ``(0,1]``
        with atom of size ``d`` at ``x=0``.

        Tail-integral form ``int_0^1 (1-g) dx`` handles the atom cleanly:
        ``E[X^k] = (1-d) / (k+1)``.
        """
        d = float(self.d)
        c = 1.0 - d
        m1 = c / 2.0
        m2 = c / 3.0
        m3 = c / 4.0
        var = m2 - m1 * m1
        std = var ** 0.5
        cv = std / m1 if m1 > 0 else np.nan
        m3_central = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        skew = m3_central / std ** 3 if std > 0 else np.nan
        return {'mean': m1, 'var': var, 'std': std, 'cv': cv,
                'skew': skew, 'p_equiv': d, 'loading': d / 2.0}

    def _kusuoka_summary(self):
        # CCoC: rho(X) = (1-d)*E(X) + d*ess_sup(X). mu = (1-d) delta_0 + d delta_1.
        d = float(self.d)
        return (1.0 - d, d, False)

    def _kusuoka_atoms(self):
        d = float(self.d)
        return [(0.0, 1.0 - d), (1.0, d)]

    def _kusuoka_density(self, p):
        return np.zeros_like(np.asarray(p, dtype=float))


class PHDistortion(Distortion):
    """Proportional-hazard distortion: :math:`g(x) = x^a`.

    The natural parameter is ``a`` (was historically also called ``rho``).
    ``Distortion('ph', a=0.7)`` and ``Distortion('ph', 0.7)`` are equivalent.
    """
    kind = 'ph'
    med_name = 'Prop Hzrd'
    long_name = 'Proportional Hazard'
    documented = True
    pricing_ok = True
    strict_pricing = True
    param_name = 'a'
    _calibration_init_shape = 0.95

    @property
    def a(self):
        """Natural alias for ``self.shape``."""
        return self.shape

    @a.setter
    def a(self, value):
        self.shape = value
        self._build()

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

    def _describe_closed_form(self):
        """Closed-form moments of D_g for g(x)=x^a.

        ``E[X^k] = a/(a+k)`` (tail-integral with ``1-g = 1-x^a``).
        """
        a = float(self.shape)
        m1 = a / (a + 1.0)
        m2 = a / (a + 2.0)
        m3 = a / (a + 3.0)
        var = m2 - m1 * m1
        std = var ** 0.5
        cv = std / m1 if m1 > 0 else np.nan
        m3_central = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        skew = m3_central / std ** 3 if std > 0 else np.nan
        p_eq = (1.0 - a) / (1.0 + a)
        return {'mean': m1, 'var': var, 'std': std, 'cv': cv,
                'skew': skew, 'p_equiv': p_eq,
                'loading': 1.0 / (a + 1.0) - 0.5}

    def _kusuoka_summary(self):
        # mu = atom of size a at p=0 plus continuous density a(1-a)(1-p)^(a-1)
        # on [0,1). int density = 1-a, total mass 1.
        a = float(self.shape)
        return (a, 0.0, False)

    def _kusuoka_atoms(self):
        return [(0.0, float(self.shape))]

    def _kusuoka_density(self, p):
        a = float(self.shape)
        p = np.asarray(p, dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            out = a * (1.0 - a) * np.power(1.0 - p, a - 1.0)
        return np.where(np.isfinite(out) & (out > 0), out, 0.0)


class WangDistortion(Distortion):
    """Wang distortion: :math:`g(x) = \\Phi(\\Phi^{-1}(x) + \\lambda)`.

    The natural parameter is ``lam`` (ASCII; matches scipy convention).
    """
    kind = 'wang'
    med_name = 'Wang'
    long_name = 'Wang-normal'
    documented = True
    pricing_ok = True
    strict_pricing = True
    param_name = 'lam'
    _calibration_init_shape = 0.95

    @property
    def lam(self):
        """Natural alias for ``self.shape``."""
        return self.shape

    @lam.setter
    def lam(self, value):
        self.shape = value
        self._build()

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

    def _describe_closed_form(self):
        """Mean and ``p_equiv`` of D_g for Wang(``λ``).

        ``int g(x) dx = P(W - Z <= λ) = Φ(λ/√2)`` for ``W, Z ~ N(0,1)``
        independent. Higher moments require the bivariate normal CDF and
        are left as ``NaN`` in the table (numeric column is exact within
        grid resolution).
        """
        lam = float(self.shape)
        int_g = float(norm.cdf(lam / 2 ** 0.5))
        return {'mean': 1.0 - int_g, 'p_equiv': 2.0 * int_g - 1.0,
                'loading': int_g - 0.5}

    def _kusuoka_atoms(self):
        # Wang's Kusuoka measure is continuous; no atoms. The density has
        # integrable singularities at p=0 and p=1, so we let the base
        # numerical density + renormalization handle it.
        return []


class DualDistortion(Distortion):
    """Dual-moment distortion: :math:`g(x) = 1 - (1-x)^b`.

    The natural parameter is ``b`` (gives ``a`` / ``b`` symmetry with PH).
    """
    kind = 'dual'
    med_name = 'Dual Mom'
    long_name = 'Dual Moment'
    documented = True
    pricing_ok = True
    strict_pricing = True
    param_name = 'b'
    _calibration_init_shape = 2.0

    @property
    def b(self):
        """Natural alias for ``self.shape``."""
        return self.shape

    @b.setter
    def b(self, value):
        self.shape = value
        self._build()

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

    def _describe_closed_form(self):
        """Closed-form moments of D_g for g(x) = 1 - (1-x)^b.

        ``E[X^k] = k * B(k, b+1) = k! b! / (b+k)!`` (tail-integral form).
        """
        b = float(self.shape)
        m1 = 1.0 / (b + 1.0)
        m2 = 2.0 / ((b + 1.0) * (b + 2.0))
        m3 = 6.0 / ((b + 1.0) * (b + 2.0) * (b + 3.0))
        var = m2 - m1 * m1
        std = var ** 0.5
        cv = std / m1 if m1 > 0 else np.nan
        m3_central = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        skew = m3_central / std ** 3 if std > 0 else np.nan
        int_g = b / (b + 1.0)
        return {'mean': m1, 'var': var, 'std': std, 'cv': cv,
                'skew': skew, 'p_equiv': 2.0 * int_g - 1.0,
                'loading': int_g - 0.5}

    def _kusuoka_atoms(self):
        # Dual has no atoms in mu; the density b(b-1)(1-p)p^(b-2)
        # integrates to 1.
        return []

    def _kusuoka_density(self, p):
        b = float(self.shape)
        p = np.asarray(p, dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            out = b * (b - 1.0) * (1.0 - p) * np.power(p, b - 2.0)
        return np.where(np.isfinite(out) & (out > 0), out, 0.0)


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
    param_name = 'p'
    _calibration_init_shape = 0.9

    @property
    def p(self):
        """Natural alias for ``self.shape``."""
        return self.shape

    @p.setter
    def p(self, value):
        self.shape = value
        self._build()

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

    def _describe_closed_form(self):
        """Closed-form moments of D_g for the piecewise-linear TVaR.

        For ``p in (0,1)``: ``g`` is linear on ``[0, 1-p]`` with slope
        ``1/(1-p)`` and constant 1 on ``[1-p, 1]``. Tail integrals give
        ``E[X^k] = (1-p)^k / (k+1)``. ``p=1`` is a degenerate point mass at
        0 (all moments 0); ``p=0`` is the identity (mean 0.5, var 1/12).
        """
        p = float(self.shape)
        if p == 1:
            return {'mean': 0.0, 'var': 0.0, 'std': 0.0,
                    'cv': np.nan, 'skew': np.nan,
                    'p_equiv': 1.0, 'loading': 0.5}
        q = 1.0 - p
        m1 = q / 2.0
        m2 = q * q / 3.0
        m3 = q ** 3 / 4.0
        var = m2 - m1 * m1
        std = var ** 0.5
        cv = std / m1 if m1 > 0 else np.nan
        m3_central = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        skew = m3_central / std ** 3 if std > 0 else np.nan
        return {'mean': m1, 'var': var, 'std': std, 'cv': cv,
                'skew': skew, 'p_equiv': p, 'loading': p / 2.0}

    def _density_knots(self):
        p = float(self.shape)
        # Kink at s = 1 - p (where g transitions from linear to constant).
        return [1.0 - p] if 0.0 < p < 1.0 else []

    def _kusuoka_summary(self):
        p = float(self.shape)
        mu0 = 1.0 if p == 0 else 0.0
        mu1 = 1.0 if p == 1 else 0.0
        interior = 0.0 < p < 1.0
        return (mu0, mu1, interior)

    def _kusuoka_atoms(self):
        return [(float(self.shape), 1.0)]

    def _kusuoka_density(self, p):
        return np.zeros_like(np.asarray(p, dtype=float))


class BiTVaRDistortion(Distortion):
    """
    Convex combination of two TVaR distortions at ``p0 < p1`` with weight
    ``w1`` on ``p1``.
    """
    kind = 'bitvar'
    med_name = 'BiTVaR'
    long_name = 'BiTVaR'
    documented = True
    pricing_ok = True

    def __init__(self, name='bitvar', *, p0, p1, w1, display_name=''):
        """
        Construct a BiTVaR distortion.

        Parameters
        ----------
        name : str
            Distortion kind name (``'bitvar'``).
        p0, p1 : float
            TVaR thresholds; require ``p0 < p1``.
        w1 : float
            Weight on the upper TVaR at ``p1``; ``1 - w1`` is the weight on
            ``p0``. ``self.shape`` is set to ``w1`` so calibration code
            (which iterates on shape) targets the upper weight.
        display_name : str, optional
            Override label.
        """
        if not (p0 < p1):
            raise ValueError(f'bitvar requires p0 < p1, got {p0=}, {p1=}')
        self._name = 'bitvar' if name == 'roe' else name
        self._p0 = p0
        self._p1 = p1
        self._w1 = w1
        self.shape = w1
        self.display_name = display_name
        self._common_init()
        self._build()

    @property
    def p0(self):
        return self._p0

    @property
    def p1(self):
        return self._p1

    @property
    def w1(self):
        return self._w1

    def _build(self):
        # keep ``_w1`` in sync if a caller has mutated ``shape`` directly
        self._w1 = self.shape
        p0, p1, w = self._p0, self._p1, self._w1
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
            s = np.array([0., 1e-50, 1 - p0, 1.])
            gs = np.array([0., pt, 1., 1.])
            self.g_inv = interp1d(gs, s, kind='linear',
                                  bounds_error=False, fill_value=(0, 1))
        else:
            s = np.array([0., 1 - p1, 1 - p0, 1.])
            gs = np.array([0., pt, 1., 1.])
            self.g = interp1d(s, gs, kind='linear',
                              bounds_error=False, fill_value=(0, 1))
            self.g_inv = interp1d(gs, s, kind='linear',
                                  bounds_error=False, fill_value=(0, 1))

    def _id_fields(self):
        return (self._name, self._p0, self._p1, self._w1, self.display_name)

    def g(self, x):
        # only reached when p1 == 1
        w = self._w1
        alpha = self._alpha
        return w * np.where(x <= 0, 0, 1) + (1 - w) * np.minimum(alpha * x, 1)

    def g_prime(self, x):
        p0, p1, w = self._p0, self._p1, self._w1
        if p1 < 1:
            return np.where(x > 1 - p0, 0,
                            np.where(x < 1 - p1,
                                     w / (1 - p1) + (1 - w) / (1 - p0),
                                     (1 - w) / (1 - p0)))
        return np.where(x > 1 - p0, 0, (1 - w) / (1 - p0))

    def quick_gS(self, den):
        if isinstance(den, pd.Series):
            return bitvar_gS(den.values, self._p0, self._p1, self._w1)
        return bitvar_gS(den, self._p0, self._p1, self._w1)

    def quick_ra(self, den, x=None):
        if isinstance(den, pd.Series):
            return bitvar_ra(den.values, np.array(den.index),
                             self._p0, self._p1, self._w1)
        return bitvar_ra(den, x, self._p0, self._p1, self._w1)

    def _describe_closed_form(self):
        """Closed-form moments of D_g for the BiTVaR linear combo.

        Since ``D_g`` is the mixture ``(1-w1) D_{tvar(p0)} + w1 D_{tvar(p1)}``
        on the CDF side, raw moments combine linearly. ``p1==1`` gives a
        point-mass component at 0 with all moments 0.
        """
        p0, p1, w = float(self._p0), float(self._p1), float(self._w1)
        def tvar_moments(p):
            if p == 1:
                return (0.0, 0.0, 0.0)
            q = 1.0 - p
            return (q / 2.0, q * q / 3.0, q ** 3 / 4.0)
        m1a, m2a, m3a = tvar_moments(p0)
        m1b, m2b, m3b = tvar_moments(p1)
        m1 = (1.0 - w) * m1a + w * m1b
        m2 = (1.0 - w) * m2a + w * m2b
        m3 = (1.0 - w) * m3a + w * m3b
        var = m2 - m1 * m1
        std = var ** 0.5
        cv = std / m1 if m1 > 0 else np.nan
        m3_central = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        skew = m3_central / std ** 3 if std > 0 else np.nan
        p_eq = (1.0 - w) * p0 + w * p1
        return {'mean': m1, 'var': var, 'std': std, 'cv': cv,
                'skew': skew, 'p_equiv': p_eq, 'loading': p_eq / 2.0}

    def _density_knots(self):
        p0, p1 = float(self._p0), float(self._p1)
        return [k for k in (1.0 - p0, 1.0 - p1) if 0.0 < k < 1.0]

    def _kusuoka_summary(self):
        p0, p1, w = float(self._p0), float(self._p1), float(self._w1)
        mu0 = (1.0 - w) if p0 == 0 else 0.0
        mu1 = w if p1 == 1 else 0.0
        interior = (0.0 < p0 < 1.0) or (0.0 < p1 < 1.0)
        return (mu0, mu1, interior)

    def _kusuoka_atoms(self):
        return [(float(self._p0), 1.0 - float(self._w1)),
                (float(self._p1), float(self._w1))]

    def _kusuoka_density(self, p):
        return np.zeros_like(np.asarray(p, dtype=float))


class WtdTVaRDistortion(Distortion):
    """
    Weighted TVaR: distortion as a weighted average of TVaRs at sorted
    thresholds ``ps`` with weights ``wts``. A mass at ``p=1`` (max term)
    is supported. ``g`` is piecewise linear; ``g_prime`` is exact.
    """
    kind = 'wtdtvar'
    med_name = 'WtdTVaR'
    long_name = 'Weighted TVaR'
    documented = True
    pricing_ok = True

    def __init__(self, name='wtdtvar', *, ps, wts, display_name=''):
        """
        Construct a weighted-TVaR distortion.

        Parameters
        ----------
        name : str
            Distortion kind name (``'wtdtvar'``).
        ps : array_like
            Sorted ascending TVaR thresholds.
        wts : array_like
            Weights on the corresponding TVaR components. Must be the
            same length as ``ps``. If ``np.isclose(sum(wts), 1)`` the
            weights are normalised silently to clean up FP noise;
            otherwise a ``ValueError`` is raised.
        display_name : str, optional
            Override label.
        """
        ps_arr = np.asarray(ps, dtype=float)
        wts_arr = np.asarray(wts, dtype=float)
        if len(ps_arr) != len(wts_arr):
            raise ValueError(
                f'wtdtvar: ps and wts must have the same length, got '
                f'{len(ps_arr)} and {len(wts_arr)}')
        s = wts_arr.sum()
        if not np.isclose(s, 1.0):
            raise ValueError(
                f'wtdtvar: wts must sum to 1, got sum={s:.6g}')
        wts_arr = wts_arr / s  # silent normalise of FP noise
        self._name = 'wtdtvar' if name == 'roe' else name
        self._ps = ps_arr
        self._wts = wts_arr
        # shape is set to ps for compatibility with code that reads
        # ``self.shape`` to recover the threshold vector.
        self.shape = ps_arr
        self.display_name = display_name
        self._common_init()
        self._build()

    @property
    def ps(self):
        return self._ps

    @property
    def wts(self):
        return self._wts

    def _build(self):
        ps = np.array(self._ps)
        wts = np.array(self._wts)
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

    def _id_fields(self):
        return (self._name, tuple(self._ps), tuple(self._wts),
                self.display_name)

    def tvar_info_df(self):
        p = np.array(self._ps)
        wts = np.array(self._wts)
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

    def _describe_closed_form(self):
        """Closed-form moments of D_g for a weighted-TVaR mixture.

        Raw moments of D_g combine linearly across TVaR components:
        ``E[X^k] = sum_i w_i * q_i^k / (k+1)`` where ``q_i = 1 - p_i``
        (zero contribution from any ``p_i == 1`` component).
        """
        ps = np.asarray(self._ps, dtype=float)
        wts = np.asarray(self._wts, dtype=float)
        q = np.where(ps < 1.0, 1.0 - ps, 0.0)
        m1 = float(np.sum(wts * q / 2.0))
        m2 = float(np.sum(wts * q * q / 3.0))
        m3 = float(np.sum(wts * q ** 3 / 4.0))
        var = m2 - m1 * m1
        std = var ** 0.5
        cv = std / m1 if m1 > 0 else np.nan
        m3_central = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        skew = m3_central / std ** 3 if std > 0 else np.nan
        p_eq = float(np.sum(wts * ps))
        return {'mean': m1, 'var': var, 'std': std, 'cv': cv,
                'skew': skew, 'p_equiv': p_eq, 'loading': p_eq / 2.0}

    def _density_knots(self):
        ps = np.asarray(self._ps, dtype=float)
        return [float(1.0 - p) for p in ps if 0.0 < p < 1.0]

    def _kusuoka_summary(self):
        ps = np.asarray(self._ps, dtype=float)
        wts = np.asarray(self._wts, dtype=float)
        mu0 = float(wts[ps == 0].sum())
        mu1 = float(wts[ps == 1].sum())
        interior = bool(np.any((ps > 0) & (ps < 1)))
        return (mu0, mu1, interior)

    def _kusuoka_atoms(self):
        return [(float(p), float(w))
                for p, w in zip(self._ps, self._wts) if w > 0]

    def _kusuoka_density(self, p):
        return np.zeros_like(np.asarray(p, dtype=float))

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


class MinimumDistortion(Distortion):
    """Pointwise minimum of several distortions.

    Parameters
    ----------
    distortions : list[Distortion]
        Constituent distortions.
    """
    kind = 'minimum'
    med_name = 'Minimum'
    long_name = 'Minimum'
    documented = False
    pricing_ok = False

    def __init__(self, name='minimum', distortions=None, *, display_name=''):
        if distortions is None:
            raise TypeError(
                'MinimumDistortion requires a non-empty `distortions` list')
        self._name = 'minimum' if name == 'roe' else name
        self._distortions = list(distortions)
        self.shape = self._distortions  # back-compat: legacy code reads .shape
        self.display_name = display_name
        self._common_init()
        self._build()

    @property
    def distortions(self):
        return self._distortions

    def _build(self):
        dists = self._distortions
        self.has_mass = bool(np.all([d.has_mass for d in dists]))
        self.mass = float(np.min([d.mass for d in dists])) if self.has_mass else 0
        if self.display_name == '':
            self.display_name = f'Minimum of {len(dists):d} distortions'

    def _id_fields(self):
        return (self._name,
                tuple(d.id() for d in self._distortions),
                self.display_name)

    def min_index(self, x):
        g_values = np.array([gi.g(x) for gi in self._distortions])
        return np.argmin(g_values, axis=0)

    def g(self, x):
        g_values = np.array([gi.g(x) for gi in self._distortions])
        return np.min(g_values, axis=0)

    def g_prime(self, x):
        # the slope at x is the slope of whichever member achieves the min.
        # nudge x=1 slightly inside so argmin is unambiguous.
        x = np.where(x == 1, 1 - 1e-15, x)
        g_values = np.array([gi.g(x) for gi in self._distortions])
        min_idx = np.argmin(g_values, axis=0)
        if np.isscalar(min_idx):
            return self._distortions[min_idx].g_prime(x)
        if np.isscalar(x):
            return np.array([self._distortions[i].g_prime(x) for i in min_idx])
        return np.array([self._distortions[i].g_prime(xi)
                         for i, xi in zip(min_idx, x)])

    def g_inv(self, y):
        # inverse of a pointwise min is the pointwise max of inverses.
        inv_values = np.array([gi.g_inv(y) for gi in self._distortions])
        return np.max(inv_values, axis=0)

    def _density_knots(self):
        knots = []
        for d in self._distortions:
            knots.extend(d._density_knots())
        return knots

    def _active_at(self, s):
        """Return the index of the active (= achieves min) member at ``s``."""
        s = float(s)
        # Avoid the s==1 tie that ``g_prime`` handles by nudging inward.
        s_eval = s if s < 1.0 else 1.0 - 1e-15
        vals = np.array([float(np.asarray(d.g(s_eval)).ravel()[0])
                         for d in self._distortions])
        return int(np.argmin(vals))

    def _slope_at(self, member, s, side='right'):
        """One-sided slope of ``member.g`` at ``s``. Used to compute the
        atom mass on a kink. ``side`` only matters when ``s`` is at a
        member knot; we offset by ``1e-8`` in the requested direction."""
        eps = 1e-8
        s_eval = s + eps if side == 'right' else s - eps
        s_eval = float(np.clip(s_eval, eps, 1.0 - eps))
        v = member.g_prime(s_eval)
        v = float(np.asarray(v).ravel()[0])
        return v

    def _kusuoka_atoms(self):
        """Atoms of mu_min, found via slope discontinuities in g_min.

        Two sources:

        * **active-transition atoms** — at the s* where the min switches
          from member i to member j. brentq refines the location;
          ``mass = s* * (g_i'(s*) - g_j'(s*))``.
        * **inherited member atoms** — at member knots / member atom
          locations where the *same* member is active just to either side
          of the knot. Mass equals the member's own atom mass (the slope
          jump in g_min equals the slope jump in the active member's g).

        Boundary atoms (p=0 = "mean", p=1 = "max") are inherited from
        whichever member is active near ``s=1`` / ``s=0`` respectively.
        """
        # Build a sorted candidate list of interior s* values.
        knots = set()
        for d in self._distortions:
            for k in d._density_knots():
                k = float(k)
                if 0.0 < k < 1.0:
                    knots.add(k)
            for p_atom, _ in d._kusuoka_atoms():
                s_atom = 1.0 - float(p_atom)
                if 0.0 < s_atom < 1.0:
                    knots.add(s_atom)
        # Active-transition s* values: walk a moderately fine grid, locate
        # consecutive pairs where ``_active_at`` differs, brentq-refine.
        grid = np.unique(np.concatenate([
            np.linspace(1e-6, 1 - 1e-6, 1001),
            np.asarray(sorted(knots)),
        ]))
        active_idx = np.array([self._active_at(s) for s in grid])
        transitions = np.where(np.diff(active_idx) != 0)[0]
        atoms = []
        for ti in transitions:
            s_lo, s_hi = grid[ti], grid[ti + 1]
            i, j = int(active_idx[ti]), int(active_idx[ti + 1])
            d_i = self._distortions[i]
            d_j = self._distortions[j]
            # Brentq for where d_i.g and d_j.g cross.
            def diff(s, d_i=d_i, d_j=d_j):
                return (float(np.asarray(d_i.g(s)).ravel()[0])
                        - float(np.asarray(d_j.g(s)).ravel()[0]))
            try:
                s_star = brentq(diff, s_lo, s_hi, xtol=1e-12)
            except (ValueError, RuntimeError):
                s_star = 0.5 * (s_lo + s_hi)
            slope_left = self._slope_at(d_i, s_star, side='left')
            slope_right = self._slope_at(d_j, s_star, side='right')
            mass = s_star * (slope_left - slope_right)
            if mass > 1e-10:
                atoms.append((1.0 - s_star, mass))
        # Inherited interior atoms: for each member knot, if the SAME
        # member is active on both sides, the slope jump in g_min equals
        # the slope jump in that member's g and contributes its own atom.
        # Already-handled active transitions are skipped via near-equality.
        transition_s = {grid[ti] for ti in transitions} | {
            (grid[ti] + grid[ti + 1]) / 2 for ti in transitions}
        for s in knots:
            # Skip if this knot coincides with an active transition.
            if any(abs(s - t) < 1e-3 for t in transition_s):
                continue
            i_left = self._active_at(s - 1e-6) if s > 1e-6 else self._active_at(1e-6)
            i_right = self._active_at(s + 1e-6) if s < 1 - 1e-6 else self._active_at(1 - 1e-6)
            if i_left != i_right:
                continue  # active transition; would be double-counting
            d_active = self._distortions[i_left]
            slope_left = self._slope_at(d_active, s, side='left')
            slope_right = self._slope_at(d_active, s, side='right')
            mass = s * (slope_left - slope_right)
            if mass > 1e-10:
                atoms.append((1.0 - s, mass))
        # Boundary atoms: which member is active near s=0 and s=1.
        i_near_0 = self._active_at(1e-6)
        i_near_1 = self._active_at(1.0 - 1e-6)
        for p_atom, w in self._distortions[i_near_0]._kusuoka_atoms():
            if float(p_atom) == 1.0 and float(w) > 0:
                atoms.append((1.0, float(w)))
        for p_atom, w in self._distortions[i_near_1]._kusuoka_atoms():
            if float(p_atom) == 0.0 and float(w) > 0:
                atoms.append((0.0, float(w)))
        # Merge atoms at the same p so duplicate locations collapse.
        merged = {}
        for p, w in atoms:
            key = round(float(p), 9)
            merged[key] = merged.get(key, 0.0) + float(w)
        return [(p, w) for p, w in merged.items() if w > 0]

    def _kusuoka_summary(self):
        # Boundary masses are inherited from whichever member is active
        # at the corresponding boundary. interior is True if any active
        # transition lies in (0, 1) or if an inherited interior atom is
        # present.
        atoms = self._kusuoka_atoms()
        mean_mass = sum(w for p, w in atoms if p == 0.0)
        max_mass = sum(w for p, w in atoms if p == 1.0)
        interior = any(0.0 < p < 1.0 for p, _ in atoms)
        return (float(mean_mass), float(max_mass), bool(interior))


class MixtureDistortion(Distortion):
    """Weighted mixture of distortions.

    Parameters
    ----------
    distortions : list[Distortion]
        Constituent distortions.
    wts : array_like, optional
        Mixing weights; defaults to uniform.
    """
    kind = 'mixture'
    med_name = 'Mixture'
    long_name = 'Mixture'
    documented = False
    pricing_ok = False

    def __init__(self, name='mixture', distortions=None, *, wts=None,
                 display_name=''):
        if distortions is None:
            raise TypeError(
                'MixtureDistortion requires a non-empty `distortions` list')
        self._name = 'mixture' if name == 'roe' else name
        self._distortions = list(distortions)
        if wts is None:
            wts = np.array([1 / len(self._distortions)] * len(self._distortions))
        self._wts = np.asarray(wts, dtype=float)
        self.shape = self._distortions  # back-compat: legacy code reads .shape
        self.display_name = display_name
        self._common_init()
        self._build()

    @property
    def distortions(self):
        return self._distortions

    @property
    def wts(self):
        return self._wts

    def _build(self):
        dists = self._distortions
        self.has_mass = bool(np.any([d.has_mass for d in dists]))
        if self.has_mass:
            self.mass = float(np.sum([d.mass * w
                                      for d, w in zip(dists, self._wts)]))
        else:
            self.mass = 0
        if self.display_name == '':
            self.display_name = f'Mixture of {len(dists):d} distortions'
        self._weights = np.asarray(self._wts, dtype=float)

    def _id_fields(self):
        return (self._name,
                tuple(d.id() for d in self._distortions),
                tuple(self._wts),
                self.display_name)

    def _combine(self, values):
        w = self._weights
        if values.ndim > 2:
            flat = values.reshape(len(self._distortions), -1)
            return (w @ flat).reshape(values.shape[1], values.shape[2])
        return w @ values

    def _stack_components(self, fn, x):
        """Stack ``[fn(d, x) for d in members]`` after broadcasting any
        scalar return (CCoC's ``g_prime`` returns ``self.v``) to the
        shape of ``x``. Returns the stacked array and a flag indicating
        whether the input ``x`` was scalar (so the caller can collapse
        the output back to scalar)."""
        x_arr = np.atleast_1d(np.asarray(x))
        rows = []
        for gi in self._distortions:
            v = np.asarray(fn(gi, x), dtype=float)
            if v.shape == () or v.size == 1:
                v = np.full(x_arr.shape, float(v))
            rows.append(v.reshape(x_arr.shape))
        return np.stack(rows), np.asarray(x).ndim == 0

    def g(self, x):
        stacked, was_scalar = self._stack_components(lambda d, x: d.g(x), x)
        out = self._combine(stacked)
        return float(out[0]) if was_scalar else out

    def g_prime(self, x):
        stacked, was_scalar = self._stack_components(
            lambda d, x: d.g_prime(x), x)
        out = self._combine(stacked)
        return float(out[0]) if was_scalar else out

    def g_inv(self, y):
        raise NotImplementedError('Inverse of mixture not implemented')

    def _density_knots(self):
        knots = []
        for d in self._distortions:
            knots.extend(d._density_knots())
        return knots

    def _kusuoka_summary(self):
        wts = self._wts
        mu0 = 0.0
        mu1 = 0.0
        interior = False
        for d, w in zip(self._distortions, wts):
            m0, m1, ia = d._kusuoka_summary()
            mu0 += float(w) * m0
            mu1 += float(w) * m1
            interior = interior or ia
        return (mu0, mu1, interior)

    def _kusuoka_atoms(self):
        # mu_mix = sum_i w_i * mu_i, so member atoms aggregate with weights.
        # Merge atoms at the same p so duplicate locations collapse to one
        # row (e.g. two CCoCs in the mix both contribute at p=0 and p=1).
        merged = {}
        for d, w in zip(self._distortions, self._wts):
            for p_atom, mass in d._kusuoka_atoms():
                p_key = round(float(p_atom), 12)
                merged[p_key] = merged.get(p_key, 0.0) + float(w) * float(mass)
        return [(p, m) for p, m in merged.items() if m > 0]

    def _kusuoka_density(self, p):
        out = np.zeros_like(np.asarray(p, dtype=float))
        for d, w in zip(self._distortions, self._wts):
            out = out + float(w) * d._kusuoka_density(p)
        return out


class BetaDistortion(Distortion):
    """
    Beta distortion: :math:`g(x) = F_{a,b}(x)` for a Beta(a, b) CDF.

    Constraints: ``0 < a <= 1`` and ``b >= 1``. ``b=1`` is PH with
    :math:`\\rho = 1/a`; ``a=1`` is dual with :math:`\\rho = b`. Not
    calibratable through ``Portfolio.calibrate_distortion``.

    References
    ----------
    Wirch and Hardy, "A synthesis of risk measures for capital
    adequacy" (IME 1999).
    """
    kind = 'beta'
    med_name = 'Beta'
    long_name = 'Beta'
    documented = True
    pricing_ok = False

    def __init__(self, name='beta', *, a, b, display_name=''):
        """
        Construct a beta distortion.

        Parameters
        ----------
        name : str
            Distortion kind name (``'beta'``).
        a : float
            First shape parameter; must satisfy ``0 < a <= 1``.
        b : float
            Second shape parameter; must satisfy ``b >= 1``.
        display_name : str, optional
            Override label.
        """
        assert 0 < a <= 1, f'a parameter must be in (0, 1], not {a}'
        assert b >= 1, f'b parameter must be >= 1, not {b}'
        self._name = 'beta' if name == 'roe' else name
        self._a = a
        self._b = b
        self.shape = [a, b]
        self.display_name = display_name
        self._common_init()
        self._build()

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def _build(self):
        self._fz = ss.beta(self._a, self._b)

    def _id_fields(self):
        return (self._name, self._a, self._b, self.display_name)

    def g(self, x):
        return self._fz.cdf(x)

    def g_inv(self, x):
        return self._fz.ppf(x)

    def g_prime(self, x):
        return self._fz.pdf(x)

    def _describe_closed_form(self):
        """``D_g`` for a Beta distortion IS the underlying Beta(a, b);
        moments come straight from scipy."""
        z = self._fz
        m1, var, skew, _ = z.stats(moments='mvsk')
        m1 = float(m1)
        var = float(var)
        std = var ** 0.5
        cv = std / m1 if m1 > 0 else np.nan
        # int g(x) dx = 1 - mean of Beta(a, b) = b / (a + b)
        int_g = float(self._b) / (float(self._a) + float(self._b))
        return {'mean': m1, 'var': var, 'std': std, 'cv': cv,
                'skew': float(skew),
                'p_equiv': 2.0 * int_g - 1.0,
                'loading': int_g - 0.5}

    def _kusuoka_atoms(self):
        # Beta's Kusuoka measure is continuous; analytic density has
        # boundary singularities depending on (a, b). Let base numerical
        # density + renormalization handle it.
        return []


class PowerDistortion(Distortion):
    """
    Power distortion built from part of a power-function distribution.
    Compare with the Bernegger approach. Allows controlled slopes at 0
    and 1. NOT calibratable through ``Portfolio.calibrate_distortion``.
    """
    kind = 'power'
    med_name = 'Power'
    long_name = 'Power'
    documented = False
    pricing_ok = False

    def __init__(self, name='power', *, x0, x1, alpha, display_name=''):
        """
        Construct a power distortion.

        Parameters
        ----------
        name : str
            Distortion kind name (``'power'``).
        x0, x1 : float
            Slope-defining endpoints; require ``x0 < x1``.
        alpha : float
            Exponent. ``self.shape`` is set to ``alpha`` for legacy
            code that reads ``shape``; calibration is NOT supported.
        display_name : str, optional
            Override label.
        """
        assert x0 < x1, 'x0 must be less than x1'
        self._name = 'power' if name == 'roe' else name
        self._x0 = x0
        self._x1 = x1
        self.shape = float(alpha)
        self.display_name = display_name
        self._common_init()
        self._build()

    @property
    def x0(self):
        return self._x0

    @property
    def x1(self):
        return self._x1

    @property
    def alpha(self):
        return self.shape

    def _build(self):
        alpha = float(self.shape)
        self._alpha = alpha
        x0, x1 = self._x0, self._x1
        if alpha != 1:
            self._bl = np.power(x1, -alpha + 1)
            self._br = np.power(x0, -alpha + 1)
        else:
            self._bl = np.log(x1)
            self._br = np.log(x0)

    def _id_fields(self):
        return (self._name, self._x0, self._x1, self.shape, self.display_name)

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

    def _density_knots(self):
        x0, x1 = float(self._x0), float(self._x1)
        return [k for k in (x0, x1) if 0.0 < k < 1.0]


class CLLDistortion(Distortion):
    """Capped log-linear distortion: :math:`g(x) = \\min(1, e^{r_0} x^b)`.

    The shape (Newton-iterated) parameter is ``b``; ``r0`` is the
    mass-at-zero intercept.
    """
    kind = 'cll'
    med_name = 'Capd Loglin'
    long_name = 'Capped Loglinear'
    documented = True
    pricing_ok = True
    strict_pricing = True
    _calibration_init_shape = 0.95

    def __init__(self, name='cll', *, r0=0.0, b, display_name=''):
        """
        Construct a capped log-linear distortion.

        Parameters
        ----------
        name : str
            Distortion kind name (``'cll'``).
        r0 : float, optional
            Mass-at-zero intercept (default 0).
        b : float
            Power-law exponent; Newton calibration iterates on this.
        display_name : str, optional
            Override label.
        """
        self._name = 'cll' if name == 'roe' else name
        self.r0 = r0
        self.shape = b
        self.display_name = display_name
        self._common_init()
        self._build()

    @property
    def b(self):
        """Natural alias for ``self.shape``."""
        return self.shape

    @b.setter
    def b(self, value):
        self.shape = value
        self._build()

    def _id_fields(self):
        return (self._name, self.shape, self.r0, self.display_name)

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

    def _describe_closed_form(self):
        """Closed-form mean of D_g for ``g(x) = min(1, e^{r0} x^b)``.

        With ``x_cap = e^{-r0/b}``: if ``x_cap >= 1`` (i.e. ``r0 <= 0``) the
        cap is inactive and ``g(x) = e^{r0} x^b``; ``int g = e^{r0}/(b+1)``,
        so ``E[X] = 1 - e^{r0}/(b+1)``. Otherwise the cap is active and
        ``int g = 1 - x_cap b/(b+1)`` so ``E[X] = x_cap b/(b+1)``. Higher
        moments are tractable but tedious; leave as ``NaN`` in v1.
        """
        b = float(self.shape)
        r0 = float(self.r0)
        ea = np.exp(r0)
        if r0 <= 0:
            int_g = ea / (b + 1.0)
        else:
            x_cap = float(np.exp(-r0 / b))
            int_g = 1.0 - x_cap * b / (b + 1.0)
        m1 = 1.0 - int_g
        return {'mean': m1, 'p_equiv': 2.0 * int_g - 1.0,
                'loading': int_g - 0.5}

    def _density_knots(self):
        b = float(self.shape)
        r0 = float(self.r0)
        if r0 <= 0:
            return []
        x_cap = float(np.exp(-r0 / b))
        return [x_cap] if 0.0 < x_cap < 1.0 else []


class CLinDistortion(Distortion):
    """Capped linear distortion: :math:`g(x) = \\min(1, r_0 + s\\,x)`.

    The shape (slope) parameter is ``slope``; ``r0`` is the mass-at-zero
    intercept. Requires ``slope >= 1 - r0``.
    """
    kind = 'clin'
    med_name = 'Capped Linear'
    long_name = 'Capped Linear'
    documented = True
    pricing_ok = True
    strict_pricing = True
    has_mass_default = True
    _calibration_init_shape = 1.0

    def __init__(self, name='clin', *, r0=0.0, slope, display_name=''):
        """
        Construct a capped linear distortion.

        Parameters
        ----------
        name : str
            Distortion kind name (``'clin'``).
        r0 : float, optional
            Mass-at-zero intercept (default 0).
        slope : float
            Linear slope; Newton calibration iterates on this.
        display_name : str, optional
            Override label.
        """
        self._name = 'clin' if name == 'roe' else name
        self.r0 = r0
        self.shape = slope
        self.display_name = display_name
        self._common_init()
        self._build()

    @property
    def slope(self):
        """Natural alias for ``self.shape``."""
        return self.shape

    @slope.setter
    def slope(self, value):
        self.shape = value
        self._build()

    def _id_fields(self):
        return (self._name, self.shape, self.r0, self.display_name)

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

    def _describe_closed_form(self):
        """Closed-form mean of D_g for ``g(x) = min(1, r0 + s x)``.

        With ``x_cap = (1 - r0)/s``: if ``x_cap >= 1`` (no effective cap)
        ``int g = r0 + s/2`` so ``E[X] = 1 - r0 - s/2``. Otherwise
        ``int g = 1 - (1-r0)^2/(2s)`` so ``E[X] = (1-r0)^2/(2s)``.
        """
        sl = float(self.shape)
        r0 = float(self.r0)
        if sl <= 0:
            return {}
        x_cap = (1.0 - r0) / sl
        if x_cap >= 1.0:
            int_g = r0 + sl / 2.0
        else:
            int_g = 1.0 - (1.0 - r0) ** 2 / (2.0 * sl)
        m1 = 1.0 - int_g
        return {'mean': m1, 'p_equiv': 2.0 * int_g - 1.0,
                'loading': int_g - 0.5}

    def _density_knots(self):
        sl = float(self.shape)
        r0 = float(self.r0)
        if sl <= 0:
            return []
        x_cap = (1.0 - r0) / sl
        return [x_cap] if 0.0 < x_cap < 1.0 else []


class LEPDistortion(Distortion):
    """Leverage-equivalent pricing distortion.

    Parameterised by the target return ``r`` (shape) and rental rate
    ``r0`` (mass intercept).
    """
    kind = 'lep'
    med_name = 'Lev Equiv'
    long_name = 'Leverage Equivalent Pricing'
    documented = True
    pricing_ok = True
    strict_pricing = True
    has_mass_default = True
    _calibration_init_shape = 0.25

    def __init__(self, name='lep', *, r0=0.0, r, display_name=''):
        """
        Construct a leverage-equivalent pricing distortion.

        Parameters
        ----------
        name : str
            Distortion kind name (``'lep'``).
        r0 : float, optional
            Rental rate (mass-at-zero intercept; default 0).
        r : float
            Target return; Newton calibration iterates on this.
        display_name : str, optional
            Override label.
        """
        self._name = 'lep' if name == 'roe' else name
        self.r0 = r0
        self.shape = r
        self.display_name = display_name
        self._common_init()
        self._build()

    @property
    def r(self):
        """Natural alias for ``self.shape``."""
        return self.shape

    @r.setter
    def r(self, value):
        self.shape = value
        self._build()

    def _id_fields(self):
        return (self._name, self.shape, self.r0, self.display_name)

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

    def _describe_closed_form(self):
        """Closed-form mean and variance of D_g for the LEP distortion.

        ``int sqrt(x(1-x)) dx = pi/8`` (Beta(3/2, 3/2) normalisation) and
        ``int x sqrt(x(1-x)) dx = pi/16``, so for ``g(x) = d + (1-d)x +
        spread sqrt(x(1-x))``:
        ``int g = d + (1-d)/2 + spread*pi/8``,
        ``E[X^2] = (1-d)/3 - spread*pi/8``.
        Higher moments require ``int x^2 sqrt(x(1-x)) dx = 5*pi/128``
        and are left as ``NaN`` in v1.
        """
        d = float(self._d)
        spread = float(self._spread)
        int_g = d + (1.0 - d) / 2.0 + spread * np.pi / 8.0
        m1 = 1.0 - int_g
        m2 = (1.0 - d) / 3.0 - spread * np.pi / 8.0
        var = m2 - m1 * m1
        if var < 0:
            var = np.nan
        std = var ** 0.5 if not np.isnan(var) else np.nan
        cv = std / m1 if (m1 > 0 and not np.isnan(std)) else np.nan
        return {'mean': m1, 'var': var, 'std': std, 'cv': cv,
                'p_equiv': 2.0 * int_g - 1.0,
                'loading': int_g - 0.5}


class LYDistortion(Distortion):
    """Linear yield distortion.

    The shape parameter ``r`` is the consumption rate; ``r0`` is the
    occupancy (the mass-at-zero intercept). The name and the
    occupancy / consumption framing are due to Don Mango (the rental
    analogy for capital pricing).

    Notes
    -----
    The minimum rate-on-line is ``r0 / (1 + r0)``; the mass at zero is
    ``ess_sup * r0 / (1 + r0)``.
    """
    kind = 'ly'
    med_name = 'Lin Yield'
    long_name = 'Linear Yield'
    documented = True
    pricing_ok = True
    strict_pricing = True
    has_mass_default = True
    _calibration_init_shape = 1.25

    def __init__(self, name='ly', *, r0=0.0, r, display_name=''):
        """
        Construct a linear-yield distortion.

        Parameters
        ----------
        name : str
            Distortion kind name (``'ly'``).
        r0 : float, optional
            Occupancy rate (mass-at-zero intercept; default 0).
        r : float
            Consumption rate; Newton calibration iterates on this.
        display_name : str, optional
            Override label.
        """
        self._name = 'ly' if name == 'roe' else name
        self.r0 = r0
        self.shape = r
        self.display_name = display_name
        self._common_init()
        self._build()

    @property
    def r(self):
        """Natural alias for ``self.shape`` (the consumption rate)."""
        return self.shape

    @r.setter
    def r(self, value):
        self.shape = value
        self._build()

    def _id_fields(self):
        return (self._name, self.shape, self.r0, self.display_name)

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
    Continuous approximation to the CCoC distortion at given ROE.

    Useful when a smooth distortion is needed in place of a CCoC with
    its mass at zero. Built as a BiTVaR with ``p0=0``, ``p1=1-eps``,
    ``w1 = roe / (1 + roe)``.
    """
    return Distortion('bitvar', p0=0, p1=1 - eps, w1=roe / (1 + roe),
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
        if k == 'ccoc':
            # p_to_parameters returns the discount d for ccoc; convert to
            # return r so the natural-kwarg constructor takes r directly.
            ans[k] = Distortion('ccoc', r=sh / (1 - sh))
        else:
            pn = Distortion._registry[k].param_name
            ans[k] = Distortion(k, **{pn: sh})
    return ans


# ===========================================================================
# Distortion construction helpers
# ===========================================================================
# These functions BUILD a Distortion from sample data; they are not
# distortion kinds themselves. Kept at module level so the ``Distortion``
# namespace stays focused on actual kinds; each returns a
# ``WtdTVaRDistortion`` instance with knots taken from the upper convex
# hull of the supplied (s, g(s)) data.


def convex_distortion(s, gs, *, display_name=''):
    """
    Build a distortion as the upper convex envelope of ``(s, g(s))`` points.

    Returns a :class:`WtdTVaRDistortion` whose piecewise-linear ``g``
    interpolates the convex hull of the supplied points. The caller is
    expected to handle any DataFrame / column-name extraction pre-call;
    this function takes two raw arrays.

    Parameters
    ----------
    s : array_like
        x-coordinates (probabilities). 0 and 1 are added if absent.
    gs : array_like
        Matching ``g(s)`` values, same length as ``s``. The points
        ``(0, 0)`` and ``(1, 1)`` are added if missing.
    display_name : str, optional
        Override label.

    Returns
    -------
    WtdTVaRDistortion
        Equivalent piecewise-linear weighted-TVaR distortion.

    Notes
    -----
    For ordered hull knots ``(s_0=0, gs_0=0), …, (s_K=1, gs_K=1)`` with
    slopes ``m_i = (gs_{i+1} - gs_i) / (s_{i+1} - s_i)`` (non-increasing
    by concavity), the corresponding TVaR weights solve a small linear
    system via :meth:`Distortion.tvar_terms`.
    """
    s = np.asarray(s, dtype=float).ravel()
    gs = np.asarray(gs, dtype=float).ravel()
    if len(s) != len(gs):
        raise ValueError(
            f'convex_distortion: s and gs must have the same length, '
            f'got {len(s)} and {len(gs)}')

    # ensure 0 and 1 are present so the hull touches both corners.
    if 0.0 not in s:
        s = np.concatenate([[0.0], s])
        gs = np.concatenate([[0.0], gs])
    if 1.0 not in s:
        s = np.concatenate([s, [1.0]])
        gs = np.concatenate([gs, [1.0]])

    pts = np.column_stack([s, gs])
    if len(pts) >= 3:
        hull = ConvexHull(pts)
        knot_idx = sorted(set(hull.simplices.flatten()))
    else:
        knot_idx = list(range(len(pts)))

    s_h = s[knot_idx]
    gs_h = gs[knot_idx]
    order = np.argsort(s_h)
    s_h = s_h[order]
    gs_h = gs_h[order]
    # remove duplicate s values (the hull may double-count corners)
    keep = np.concatenate([[True], np.diff(s_h) > 0])
    s_h = s_h[keep]
    gs_h = gs_h[keep]

    # solve gs_h = wts @ tvar_terms(ps) on the (ascending) hull knots.
    # ps are derived from the hull's x-coordinates: ps = 1 - s_h[::-1].
    ps = (1 - s_h[::-1]).astype(float)
    M = Distortion.tvar_terms(ps)
    # tvar_terms returns an (n, n) matrix mapping wts -> gs evaluated at
    # the reversed s values; solve in least-squares for stability.
    wts, *_ = np.linalg.lstsq(M.T, gs_h, rcond=None)
    wts = np.clip(wts, 0.0, None)
    if wts.sum() <= 0:
        # degenerate input: fall back to uniform weighting
        wts = np.ones_like(ps) / len(ps)
    wts = wts / wts.sum()
    return WtdTVaRDistortion('wtdtvar', ps=ps, wts=wts,
                             display_name=display_name)


def bagged_distortion(data, proportion, samples, *,
                      el_col='EL', spread_col='Spread', display_name=''):
    """
    Bootstrap-aggregated convex distortion from tabular ``(EL, Spread)`` data.

    Resamples ``data`` ``samples`` times at the given ``proportion``,
    builds the upper convex envelope of each sample with
    :func:`convex_distortion`, averages ``g(s)`` across samples on a
    uniform grid, and returns the averaged distortion.

    Parameters
    ----------
    data : pandas.DataFrame
        Must contain columns named by ``el_col`` and ``spread_col``.
    proportion : float
        Fraction of rows to sample without replacement on each draw.
    samples : int
        Number of bootstrap iterations.
    el_col, spread_col : str, optional
        Column names. Default ``'EL'`` and ``'Spread'``.
    display_name : str, optional
        Override label.

    Returns
    -------
    WtdTVaRDistortion
    """
    s_grid = np.linspace(0, 1, 10001)
    accum = np.zeros_like(s_grid)
    for _ in range(samples):
        rebit = data.sample(frac=proportion, replace=False,
                            random_state=RANDOM)
        s_pts = np.concatenate([rebit[el_col].values, [0.0, 1.0]])
        gs_pts = np.concatenate([rebit[spread_col].values, [0.0, 1.0]])
        d = convex_distortion(s_pts, gs_pts)
        accum += np.asarray(d.g(s_grid), dtype=float)
    accum /= samples
    return convex_distortion(s_grid, accum, display_name=display_name)


def convex_example(source='bond'):
    """
    Example convex distortion using bundled yield-curve or cat-bond data.

    Parameters
    ----------
    source : {'bond', 'cat'}
        ``'bond'`` uses a BIS-style corporate yield curve (rating → EL,
        yield); ``'cat'`` uses ROL vs EL pairs from cat bonds.

    Returns
    -------
    WtdTVaRDistortion
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
        return convex_distortion(df['EL'].values, df['Yield'].values,
                                 display_name='Yield Curve')

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
        return convex_distortion(df['EL'].values, df['ROL'].values,
                                 display_name='Cat Bond')

    else:
        raise ValueError(
            f'Inadmissible value {source} passed to convex_example, '
            f'expected bond or cat')
