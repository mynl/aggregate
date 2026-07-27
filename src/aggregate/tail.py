"""Ordered tail-thickness classification for frequencies, severities, and aggregates.

This module is the **single source of truth** for tail shape. It exposes an
ordered 5-rung scale (:class:`TailClass`) plus a separate ``log_concave`` flag,
deterministic family-lookup classifiers for frequency and severity, the
``combine`` rule that produces the aggregate class, the layered thick/thin
**tail report** (:class:`TailRow`, :func:`build_tail_rows`, :func:`tail_frame`),
and text builders for the ``.info`` / ``.tail_description`` /
``.tail_explanation`` surfaces.

The bounded-support tables (:data:`_BOUNDED_FREQS`, :data:`_BOUNDED_SCIPY_SEVS`)
live here, and ``Aggregate.bounded`` / ``Severity.bounded`` / ``Portfolio.bounded``
are derived from the classifier (``bounded ⇔ tail class is BOUNDED``).

Design notes
------------
* **Leaf module.** Classifiers take duck-typed objects and read attributes
  (``freq_name``, ``sev_name``, ``sev_a``, ``sev_b``, ``sev_kind``, ``limit``,
  ``attachment``, ``sev_ub``, ``signed``, ``support_atoms``, ``fz``) — they do
  **not** import ``distributions`` or ``portfolio``, so those modules can import
  ``tail`` with no cycle.
* **Family lookup + structure — no numeric estimator.** Classification is exact
  and deterministic from the spec: a family table plus the structural bound
  (finite ``limit`` / splice cap / attachment). Families not in the tables
  resolve to :attr:`TailClass.UNKNOWN`, which the sizer treats *conservatively
  as thick* when the support is also unbounded — never guessed. There is **no**
  numeric density-tail estimator (mean-excess slope, log-log-S slope): estimating
  a tail numerically needs a discretised grid, but the grid is the very thing the
  estimate is meant to choose (chicken-and-egg), so numerics cannot drive the
  sizer. New families are handled by extending the tables (cheap, exact).
* **Two facts, not one.** A capped heavy family carries both its *base-family
  thickness* (e.g. a Lévy base is heavy) and its *structural bound* (a finite
  ``limit`` / splice). The **effective** class the sizer uses combines them: a
  thick base capped at ``L`` is *effective-bounded* (cover ``[0, L]``), while the
  base heaviness still informs resolution within ``[0, L]``.
* **Spec-only.** Every determination here — class, bound, support, thick/thin,
  concentration — uses structural spec information and pre-computed moments,
  never a post-``update`` density, so the whole report is valid *before*
  ``update()``.

Notes
-----
The aggregate combine rule ``agg = max(freq, sev)`` is exact under the standard
families supported here: when severity is subexponential or heavier the single-
big-jump principle gives ``P(S>x) ~ E[N]·P(X>x)`` so the aggregate inherits the
severity class; when severity is light, the compound decay is set by the heavier
of the severity and frequency decay rates. This holds while the frequency PGF is
analytic at 1 (all standard frequencies); genuinely heavy mixing (PIG / Sichel /
Neyman-A) is a watch item, classified conservatively from the family table.

Tail Classification
-------------------

Let :math:`\\overline F(x)=1-F(x)` denote the survival function and let
:math:`H(x)=-\\log\\overline F(x)` denote the cumulative hazard.  We use the
following five-class description of right-tail thickness.  A distribution is
``bounded`` when it has a finite right endpoint :math:`x_F`; in this case
:math:`H(x)\\to\\infty` as :math:`x\\uparrow x_F`.  For an unbounded distribution,
it is ``superexponential`` when :math:`H(x)/x\\to\\infty`, ``exponential`` when
:math:`H(x)/x\\to\\lambda` for some :math:`0<\\lambda<\\infty`, ``subexponential``
when

.. math::

   \\frac{H(x)}{x}\\longrightarrow 0
   \\qquad\\text{and}\\qquad
   \\frac{H(x)}{\\log x}\\longrightarrow\\infty,

and ``power`` when :math:`H(x)/\\log x\\to\\alpha` for some
:math:`0<\\alpha<\\infty`.  Thus the ``subexponential`` category comprises regular
tails lying strictly between exponential and power decay, including the
lognormal and Weibull distributions with shape parameter strictly below one.

Strictly speaking, *subexponential distribution* has the standard convolutional
meaning

.. math::

   \\overline{F*F}(x)\\sim 2\\overline F(x),

which expresses the principle that a large sum is produced asymptotically by one
large summand.  Slower-than-exponential decay alone does not imply this property
for arbitrary irregular tails.  We nevertheless use ``subexponential`` as a
tail-thickness label because, for the regular analytic distribution families
classified here, the intermediate tails satisfy the usual regularity conditions
and are convolution-subexponential.  The classification is therefore intended
for standard named families, not for arbitrary survival functions with
oscillating, discontinuous, or otherwise pathological asymptotics.  See Foss,
Korshunov, and Zachary, *An Introduction to Heavy-Tailed and Subexponential
Distributions*, 2nd ed., Springer, 2013, especially the discussion of
long-tailed and subexponential distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from .config import get_settings

__all__ = [
    'TailClass', 'TailClasses', 'TailInfo', 'TailRow',
    'classify_frequency', 'classify_severity', 'combine',
    'aggregate_tail_info', 'tail_class_label',
    'is_thick', 'thickness_label', 'severity_support', 'concentration',
    'occ_net_severity_row', 'build_tail_rows', 'tail_frame',
    'describe_row', 'describe_rows', 'explain_rows',
    'CONCENTRATION_CV',
    '_BOUNDED_FREQS', '_BOUNDED_SCIPY_SEVS',
]


# Conservative concentration cutoff: a book is "concentrated" (its mass band
# clears 0, so the windowed left-lift is eligible) only when its coefficient of
# variation is comfortably small -- the band then sits >= 1/CONCENTRATION_CV
# standard deviations above 0. Tighter than the legacy 1/z ~ 0.14 gate: lifting
# x_min when the band does not really clear 0 clips left-tail mass, so we err
# toward *not* windowing when marginal. See plan-univariate-bucket [tail-report].
CONCENTRATION_CV = get_settings().discretization.concentration_cv


# ----------------------------------------------------------------------------
# Bounded-support tables (moved here from distributions.py; single source).
# ----------------------------------------------------------------------------

# Frequencies with bounded support.
_BOUNDED_FREQS = frozenset({'fixed', 'bernoulli', 'binomial', 'empirical',
                            'renewal'})

# scipy.stats families with bounded support. Conservative: only the standard
# finite-support members.
_BOUNDED_SCIPY_SEVS = frozenset({
    'beta', 'uniform', 'arcsine', 'rdist', 'triang', 'trapezoid',
    'semicircular', 'truncnorm', 'truncexpon', 'truncpareto',
    'truncweibull_min', 'cosine', 'anglit', 'wrapcauchy', 'kstwo',
})


# ----------------------------------------------------------------------------
# The ordered scale.
# ----------------------------------------------------------------------------

class TailClass(IntEnum):
    """Ordered tail-thickness rungs, lightest to heaviest.

    The five real rungs are ``BOUNDED < SUPER_EXPONENTIAL < EXPONENTIAL <
    SUBEXPONENTIAL < POWER_LAW``, ordered by tail-decay rate so that
    :func:`combine` (and Portfolio worst-of) can use the heavier = larger
    convention. ``UNKNOWN`` is a sentinel that is **not** on the order: it
    represents "not determined" (an unrecognised family with unbounded support)
    and *poisons* :func:`combine` rather than masquerading as a thickness — see
    :func:`combine`. The sizer treats it conservatively as thick.

    Notes
    -----
    Log-concavity is a structural density property, not a strict point on the
    thickness order (Poisson and Normal are log-concave & super-exponential;
    Gamma with shape ``k ≥ 1`` is log-concave but exponential-tailed; Lognormal
    is not log-concave and subexponential), so it is reported as a separate
    boolean flag, never as a rung.
    """

    UNKNOWN = -1
    BOUNDED = 0
    SUPER_EXPONENTIAL = 1
    EXPONENTIAL = 2
    SUBEXPONENTIAL = 3
    POWER_LAW = 4

    def __str__(self) -> str:
        return _LABELS[self]


_LABELS = {
    TailClass.UNKNOWN: 'undetermined',
    TailClass.BOUNDED: 'bounded',
    TailClass.SUPER_EXPONENTIAL: 'super-exponential',
    TailClass.EXPONENTIAL: 'exponential',
    TailClass.SUBEXPONENTIAL: 'subexponential',
    TailClass.POWER_LAW: 'power-law',
}


def tail_class_label(tc: TailClass) -> str:
    """Human-readable label for a :class:`TailClass` (e.g. ``'power-law'``)."""
    return _LABELS[TailClass(tc)]


_LABEL_TO_CLASS = {label: tc for tc, label in _LABELS.items()}


def tail_class_from_label(label: str) -> TailClass:
    """Inverse of :func:`tail_class_label`: map a label back to its :class:`TailClass`.

    Used to recover the per-side rung from a rendered ``tail_behavior_df`` column (e.g. when
    a portfolio recomputes a per-side worst-of from its unit rows).
    """
    return _LABEL_TO_CLASS[label]


class TailClasses(NamedTuple):
    """The ``(freq, sev, agg)`` triple returned by ``.tail_class``."""
    freq: TailClass
    sev: TailClass
    agg: TailClass


@dataclass(frozen=True)
class TailInfo:
    """Full tail-shape summary for an aggregate (or a portfolio's worst unit).

    Built once by :func:`aggregate_tail_info`; every public tail property is a
    view or formatter over this struct, so there is no recompute and no drift.

    Attributes
    ----------
    freq, sev, agg : TailClass
        Frequency, (thickest-component) severity, and combined aggregate rungs.
    freq_lc, sev_lc, agg_lc : bool or None
        Log-concavity flags. ``agg_lc`` is always ``None`` (a random sum does not
        inherit log-concavity analytically, and there is no numeric estimator to
        set it). ``None`` means "not determined".
    alpha : float or None
        Power-law tail index of the thickest severity component, when it is a
        power-law family; otherwise ``None``. ``alpha < 2`` ⇒ infinite variance,
        ``alpha ≤ 1`` ⇒ infinite mean.
    driver : str
        ``'severity'`` or ``'frequency'`` — which component sets the aggregate
        rung (``'severity'`` on a tie, since the single-big-jump regime is the
        more informative story); ``'undetermined'`` when ``agg`` is UNKNOWN.
    flags : dict
        Extra structured flags, e.g. ``infinite_variance`` / ``infinite_mean``.
    """

    freq: TailClass
    sev: TailClass
    agg: TailClass
    freq_lc: Optional[bool] = None
    sev_lc: Optional[bool] = None
    agg_lc: Optional[bool] = None
    alpha: Optional[float] = None
    driver: str = 'undetermined'
    flags: dict = field(default_factory=dict)

    @property
    def classes(self) -> TailClasses:
        """The ``(freq, sev, agg)`` :class:`TailClasses` view."""
        return TailClasses(self.freq, self.sev, self.agg)


# ----------------------------------------------------------------------------
# Frequency classification.
# ----------------------------------------------------------------------------

# freq_name -> (TailClass, log_concave). Mixed-Poisson families default to
# EXPONENTIAL (geometric-type tail) with log_concave left None. Genuinely heavy
# mixing is a watch item; extend this table if a family warrants a heavier rung.
FREQ_TAIL: dict[str, tuple[TailClass, Optional[bool]]] = {
    'fixed':        (TailClass.BOUNDED, True),
    'bernoulli':    (TailClass.BOUNDED, True),
    'binomial':     (TailClass.BOUNDED, True),
    'empirical':    (TailClass.BOUNDED, None),
    'renewal':      (TailClass.BOUNDED, None),   # realized count = empirical
    'poisson':      (TailClass.SUPER_EXPONENTIAL, True),
    'geometric':    (TailClass.EXPONENTIAL, True),
    'negbin':       (TailClass.EXPONENTIAL, True),
    'pascal':       (TailClass.EXPONENTIAL, None),
    'logarithmic':  (TailClass.EXPONENTIAL, None),
    'delaporte':    (TailClass.EXPONENTIAL, None),
    'neymana':      (TailClass.EXPONENTIAL, None),
    'gamma':        (TailClass.EXPONENTIAL, None),   # gamma-mixed Poisson
    'ig':           (TailClass.EXPONENTIAL, None),
    'sig':          (TailClass.EXPONENTIAL, None),
    'beta':         (TailClass.EXPONENTIAL, None),
    'sichel':       (TailClass.EXPONENTIAL, None),
    'sichel.gamma': (TailClass.EXPONENTIAL, None),
    'sichel.ig':    (TailClass.EXPONENTIAL, None),
}


def classify_frequency(frequency) -> tuple[TailClass, Optional[bool]]:
    """Classify a frequency distribution's tail.

    Parameters
    ----------
    frequency : object
        Anything carrying a ``freq_name`` string attribute (an
        ``aggregate.Frequency``).

    Returns
    -------
    (TailClass, bool or None)
        The rung and the log-concavity flag. Unrecognised families return
        ``(TailClass.UNKNOWN, None)``.
    """
    name = getattr(frequency, 'freq_name', None)
    if isinstance(name, str) and name in FREQ_TAIL:
        return FREQ_TAIL[name]
    return TailClass.UNKNOWN, None


# ----------------------------------------------------------------------------
# Severity classification.
# ----------------------------------------------------------------------------

# scipy families with a fixed (param-independent) RIGHT-tail class and a
# log-concavity flag. Two-sided families list their right class here; an
# asymmetric left is recorded in ``_SEV_LEFT_CLASS``. Populated from the
# reconciled SciPy tail survey (dev / 2026-06-17 integrated.md).
SCIPY_SEV_TAIL: dict[str, tuple[TailClass, Optional[bool]]] = {
    # --- super-exponential (Gaussian-type or faster) ---
    'norm':         (TailClass.SUPER_EXPONENTIAL, True),
    'powernorm':    (TailClass.SUPER_EXPONENTIAL, True),
    'skewnorm':     (TailClass.SUPER_EXPONENTIAL, None),
    'halfnorm':     (TailClass.SUPER_EXPONENTIAL, True),
    'foldnorm':     (TailClass.SUPER_EXPONENTIAL, None),
    'chi':          (TailClass.SUPER_EXPONENTIAL, None),
    'maxwell':      (TailClass.SUPER_EXPONENTIAL, True),
    'rayleigh':     (TailClass.SUPER_EXPONENTIAL, True),
    'nakagami':     (TailClass.SUPER_EXPONENTIAL, None),
    'rice':         (TailClass.SUPER_EXPONENTIAL, None),
    'kstwobign':    (TailClass.SUPER_EXPONENTIAL, None),
    'gompertz':     (TailClass.SUPER_EXPONENTIAL, None),
    'exponpow':     (TailClass.SUPER_EXPONENTIAL, None),   # exp(-exp(x**b)); double-exp
    'gumbel_l':     (TailClass.SUPER_EXPONENTIAL, None),   # right super; left exp
    'loggamma':     (TailClass.SUPER_EXPONENTIAL, None),   # right super; left exp
    # --- exponential ---
    'expon':        (TailClass.EXPONENTIAL, True),
    'laplace':      (TailClass.EXPONENTIAL, True),
    'laplace_asymmetric': (TailClass.EXPONENTIAL, None),
    'logistic':     (TailClass.EXPONENTIAL, True),
    'genlogistic':  (TailClass.EXPONENTIAL, None),
    'hypsecant':    (TailClass.EXPONENTIAL, True),
    'dgamma':       (TailClass.EXPONENTIAL, None),
    'genhyperbolic':(TailClass.EXPONENTIAL, None),
    'norminvgauss': (TailClass.EXPONENTIAL, False),
    'chi2':         (TailClass.EXPONENTIAL, None),
    'erlang':       (TailClass.EXPONENTIAL, None),
    'fatiguelife':  (TailClass.EXPONENTIAL, False),
    'genexpon':     (TailClass.EXPONENTIAL, None),
    'geninvgauss':  (TailClass.EXPONENTIAL, None),
    'halflogistic': (TailClass.EXPONENTIAL, None),
    'invgauss':     (TailClass.EXPONENTIAL, False),   # semi-heavy; watch item
    'ncx2':         (TailClass.EXPONENTIAL, None),
    'recipinvgauss':(TailClass.EXPONENTIAL, None),
    'wald':         (TailClass.EXPONENTIAL, False),
    'gumbel_r':     (TailClass.EXPONENTIAL, None),    # right exp; left super
    'moyal':        (TailClass.EXPONENTIAL, None),    # right exp; left super
    'exponnorm':    (TailClass.EXPONENTIAL, None),    # right exp; left super
    'crystalball':  (TailClass.SUPER_EXPONENTIAL, None),  # right super; left power-law
    # --- subexponential ---
    'lognorm':      (TailClass.SUBEXPONENTIAL, False),
    'gibrat':       (TailClass.SUBEXPONENTIAL, False),
    'johnsonsu':    (TailClass.SUBEXPONENTIAL, False),
    'powerlognorm': (TailClass.SUBEXPONENTIAL, False),
}

# Power-law families: scipy name -> callable(sev_a, sev_b) -> tail index alpha.
# alpha is the exponent in P(X > x) ~ x**(-alpha). The shape slot differs by
# family (a classic bug source) so each is mapped explicitly. From the
# reconciled SciPy survey (integrated.md). ``levy_l`` / ``crystalball`` carry a
# power-law tail only on a side recorded in ``_SEV_LEFT_CLASS``.
_POWER_LAW_ALPHA = {
    'pareto':     lambda a, b: a,            # Pareto I, shape b == sev_a
    'lomax':      lambda a, b: a,            # Pareto II, shape c == sev_a
    'fisk':       lambda a, b: a,            # log-logistic, shape c == sev_a
    'loglogistic':lambda a, b: a,
    'loglaplace': lambda a, b: a,            # right ~ x**-(c+1) -> index c
    't':          lambda a, b: a,            # Student-t, df == sev_a
    'nct':        lambda a, b: a,            # noncentral t, df == sev_a
    'cauchy':     lambda a, b: 1.0,          # alpha == 1
    'halfcauchy': lambda a, b: 1.0,
    'foldcauchy': lambda a, b: 1.0,
    'skewcauchy': lambda a, b: 1.0,
    'invweibull': lambda a, b: a,            # Frechet, shape c == sev_a
    'frechet':    lambda a, b: a,
    'invgamma':   lambda a, b: a,            # tail ~ x**-(a+1) -> index a
    'kappa3':     lambda a, b: a,            # tail ~ x**-(a+1) -> index a
    'burr':       lambda a, b: a * b,        # Burr III, c*d == sev_a*sev_b
    'burr12':     lambda a, b: a * b,        # Burr XII, c*d == sev_a*sev_b
    'mielke':     lambda a, b: b,            # Dagum, s == sev_b
    'betaprime':  lambda a, b: b,            # ~ x**-(b+1) -> index b
    'f':          lambda a, b: b / 2.0,      # ~ x**-(d2/2) -> index d2/2
    'jf_skew_t':  lambda a, b: 2.0 * b,      # right ~ x**-(2b+1) -> index 2b
    'levy':       lambda a, b: 0.5,          # survival ~ x**-1/2
    'levy_l':     lambda a, b: 0.5,          # left power-law (right bounded)
    'landau':     lambda a, b: 1.0,          # right ~ x**-2 -> survival x**-1; left super
    'alpha':      lambda a, b: 1.0,          # right pdf ~ x**-2 -> survival x**-1
    'rel_breitwigner': lambda a, b: 3.0,     # pdf ~ x**-4 -> survival x**-3
}

# Two-sided families whose LEFT tail class differs from the right. The right
# class is in ``SCIPY_SEV_TAIL`` / ``_POWER_LAW_ALPHA``; this overrides the
# (otherwise symmetric) left. Only consulted when the support is open on the
# left -- a finite left support end is ``BOUNDED`` regardless.
_SEV_LEFT_CLASS: dict[str, TailClass] = {
    'gumbel_r':    TailClass.SUPER_EXPONENTIAL,
    'gumbel_l':    TailClass.EXPONENTIAL,
    'loggamma':    TailClass.EXPONENTIAL,
    'moyal':       TailClass.SUPER_EXPONENTIAL,
    'exponnorm':   TailClass.SUPER_EXPONENTIAL,
    'landau':      TailClass.SUPER_EXPONENTIAL,
    'crystalball': TailClass.POWER_LAW,
}


def _severity_bounded(severity) -> bool:
    """Structural (spec-only) bounded-support test for a single severity.

    This is the BOUNDED determination, lifted out of the legacy
    ``Severity.bounded``. It uses only spec information — never a computed
    density — so it is valid before ``update()``.

    Parameters
    ----------
    severity : object
        A single ``aggregate.Severity`` (duck-typed).

    Returns
    -------
    bool
        ``True`` for fixed / histogram atoms, finite-support scipy families,
        a finite layer ``limit`` or splice ``sev_ub``, or a wrapped meta/copy
        object that is itself bounded.
    """
    kind = getattr(severity, 'sev_kind', '')
    if kind in ('fixed', 'dhistogram', 'chistogram'):
        return True
    if kind in ('meta', 'copy'):
        inner = getattr(severity, 'sev_name', None)
        return bool(getattr(inner, 'bounded', False))
    if np.isfinite(getattr(severity, 'limit', np.inf)):
        return True
    if np.isfinite(getattr(severity, 'sev_ub', np.inf)):
        return True
    name = getattr(severity, 'sev_name', None)
    if isinstance(name, str) and name in _BOUNDED_SCIPY_SEVS:
        return True
    # Fallback: a finite scipy support on both ends is bounded -- catches every
    # finite-support family (argus, bradford, gausshyper, johnsonsb, irwinhall,
    # powerlaw, loguniform, genhalflogistic, tukeylambda with lambda>0, ...)
    # without enumerating them. ``fz.support()`` is spec-only (built in __init__).
    try:
        lo, hi = severity.fz.support()
        if np.isfinite(lo) and np.isfinite(hi):
            return True
    except Exception:
        pass
    return False


def _weibull_shape(c) -> tuple[TailClass, Optional[bool], Optional[float]]:
    """Weibull / stretched-exponential shape rule on ``exp(-x**c)``.

    ``c > 1`` super-exponential (log-concave), ``c == 1`` exponential,
    ``0 < c < 1`` subexponential; a non-finite shape falls back to exponential.
    """
    if not np.isfinite(c):
        return TailClass.EXPONENTIAL, None, None
    if c > 1.0:
        return TailClass.SUPER_EXPONENTIAL, True, None
    if c == 1.0:
        return TailClass.EXPONENTIAL, True, None
    return TailClass.SUBEXPONENTIAL, False, None


def _family_right_class(name, a, b) -> tuple[TailClass, Optional[bool], Optional[float]]:
    """RIGHT-tail class of a scipy family, ignoring any structural cap.

    The single source consulted by :func:`classify_severity` (after the
    structural-bounded test) and by the per-side / base-rung helpers. Returns
    ``(rung, log_concave, alpha)``; ``alpha`` is set only for a POWER_LAW rung.
    Param-aware families are resolved from ``sev_a`` / ``sev_b`` (the scipy shape
    slots, in order); unrecognised families return ``UNKNOWN``.
    """
    if not isinstance(name, str):
        return TailClass.UNKNOWN, None, None

    # --- parameter-aware families (shape changes the rung) ---
    if name == 'gamma':
        return TailClass.EXPONENTIAL, bool(np.isfinite(a) and a >= 1.0), None
    if name in ('weibull_min', 'dweibull', 'gennorm', 'halfgennorm'):
        # weibull_min/dweibull shape c == sev_a; gennorm/halfgennorm beta == sev_a.
        return _weibull_shape(a)
    if name in ('gengamma', 'exponweib'):
        # the Weibull exponent is the SECOND shape (sev_b); gengamma c<0 is heavy.
        if name == 'gengamma' and np.isfinite(b) and b < 0:
            alpha = (-b * a) if (np.isfinite(a) and np.isfinite(b)) else None
            return TailClass.POWER_LAW, False, alpha
        return _weibull_shape(b)
    if name == 'genpareto':
        # scipy shape c == xi. xi > 0 heavy (alpha = 1/xi); xi == 0 exponential.
        if np.isfinite(a) and a > 0:
            return TailClass.POWER_LAW, False, 1.0 / a
        return TailClass.EXPONENTIAL, None, None
    if name == 'tukeylambda':
        # lambda > 0 bounded; lambda == 0 logistic/exponential; lambda < 0 power law.
        if not np.isfinite(a):
            return TailClass.UNKNOWN, None, None
        if a > 0:
            return TailClass.BOUNDED, None, None
        if a == 0:
            return TailClass.EXPONENTIAL, True, None
        return TailClass.POWER_LAW, False, -1.0 / a
    if name == 'levy_stable':
        # stable index alpha == sev_a; alpha == 2 is Gaussian, alpha < 2 heavy.
        if np.isfinite(a) and a < 2.0:
            return TailClass.POWER_LAW, False, a
        return TailClass.SUPER_EXPONENTIAL, True, None

    # --- fixed-class lookups ---
    if name in _POWER_LAW_ALPHA:
        return TailClass.POWER_LAW, False, float(_POWER_LAW_ALPHA[name](a, b))
    if name in SCIPY_SEV_TAIL:
        cls, lc = SCIPY_SEV_TAIL[name]
        return cls, lc, None
    return TailClass.UNKNOWN, None, None


def _family_sides(name, a, b) -> tuple[TailClass, TailClass, Optional[bool], Optional[float]]:
    """Per-side decay classes ``(left, right, log_concave, alpha)`` of a family.

    The right class is :func:`_family_right_class`; the left is the same
    (symmetric) unless the family is in :data:`_SEV_LEFT_CLASS` (an asymmetric
    two-sided family such as ``gumbel_r`` or ``loggamma``). The left class is
    only material when the support is open on the left; a finite left end is
    ``BOUNDED`` regardless (handled by the support, not here).
    """
    right, lc, alpha = _family_right_class(name, a, b)
    left = _SEV_LEFT_CLASS.get(name, right) if isinstance(name, str) else right
    return left, right, lc, alpha


def classify_severity(severity) -> tuple[TailClass, Optional[bool], Optional[float]]:
    """Classify a single severity component's overall tail.

    Parameters
    ----------
    severity : object
        A single ``aggregate.Severity`` (duck-typed). Reads ``sev_kind``,
        ``sev_name``, ``sev_a``, ``sev_b``, ``limit``, ``sev_ub``, ``fz``.

    Returns
    -------
    (TailClass, bool or None, float or None)
        The **overall** rung (the heavier of the two sides), the log-concavity
        flag, and the power-law index ``alpha`` (``None`` unless POWER_LAW).

    Notes
    -----
    Order: **structural-bounded → family per-side → UNKNOWN**, returning at the
    first hit. The structural-bounded test is first (and now also accepts any
    finite scipy support) so ``bounded`` is decided from the spec alone, which
    the lifted-natural-allocation guard relies on. The overall rung is the
    heavier side (``UNKNOWN`` ranked conservatively), so an asymmetric two-sided
    family (e.g. ``gumbel_r``: super-exp left, exponential right) reports its
    heaviest tail. See :func:`_family_right_class` for the family table and
    ``dev`` ``integrated.md`` for the reconciled SciPy survey behind it.
    """
    if _severity_bounded(severity):
        return TailClass.BOUNDED, None, None
    name = getattr(severity, 'sev_name', None)
    if not isinstance(name, str):
        return TailClass.UNKNOWN, None, None
    a = getattr(severity, 'sev_a', np.nan)
    b = getattr(severity, 'sev_b', np.nan)
    left, right, lc, alpha = _family_sides(name, a, b)
    return _heaviest((left, right)), lc, alpha


def _combine_severities(sevs):
    """Combine a list of severity components: thickest rung, all-log-concave.

    Parameters
    ----------
    sevs : iterable
        Severity components (``Aggregate.sevs``).

    Returns
    -------
    (TailClass, bool or None, float or None)
        The max (thickest) rung over components (UNKNOWN poisons, matching
        :func:`combine`), ``log_concave = all(component flags)`` (``None`` if
        any component flag is ``None``), and the ``alpha`` of the thickest
        power-law component (``None`` otherwise).
    """
    rung = TailClass.BOUNDED
    lc: Optional[bool] = True
    alpha: Optional[float] = None
    any_unknown = False
    for s in sevs:
        c, c_lc, c_alpha = classify_severity(s)
        if c == TailClass.UNKNOWN:
            any_unknown = True
            continue
        if c > rung:
            rung = c
            alpha = c_alpha          # alpha of the (current) thickest component
        if c_lc is None:
            lc = None
        elif lc is not None:
            lc = lc and c_lc
    if any_unknown:
        return TailClass.UNKNOWN, None, None
    return rung, lc, alpha


def combine(freq_cls: TailClass, sev_cls: TailClass) -> TailClass:
    """Combine frequency and severity rungs into the aggregate rung.

    Parameters
    ----------
    freq_cls, sev_cls : TailClass
        Frequency and severity tail classes.

    Returns
    -------
    TailClass
        ``UNKNOWN`` if either input is ``UNKNOWN`` (the sentinel poisons the
        result rather than being treated as a real thickness); otherwise the
        heavier (``max``) of the two — exact under the single-big-jump principle
        for heavy severity and under decay-rate dominance for light severity.
    """
    if freq_cls == TailClass.UNKNOWN or sev_cls == TailClass.UNKNOWN:
        return TailClass.UNKNOWN
    return max(freq_cls, sev_cls)


# ----------------------------------------------------------------------------
# Aggregate-level builder.
# ----------------------------------------------------------------------------

def aggregate_tail_info(frequency, sevs) -> TailInfo:
    """Build the full :class:`TailInfo` for an aggregate from its spec.

    Parameters
    ----------
    frequency : object
        The aggregate's ``Frequency`` (read for ``freq_name``).
    sevs : iterable
        The severity components (``Aggregate.sevs``).

    Returns
    -------
    TailInfo
        Frequency / severity / aggregate rungs, log-concavity flags
        (``agg_lc`` is always ``None`` — not analytically determined), the
        power-law ``alpha`` if any, the driver, and structured flags.

    Notes
    -----
    Spec-only: touches no computed density, so it is valid before ``update()``.
    """
    freq_cls, freq_lc = classify_frequency(frequency)
    sev_cls, sev_lc, alpha = _combine_severities(sevs) if sevs is not None and len(sevs) \
        else (TailClass.UNKNOWN, None, None)
    agg_cls = combine(freq_cls, sev_cls)

    flags: dict = {}
    if agg_cls == TailClass.POWER_LAW and alpha is not None and np.isfinite(alpha):
        flags['alpha'] = alpha
        flags['infinite_variance'] = alpha < 2.0
        flags['infinite_mean'] = alpha <= 1.0

    if agg_cls == TailClass.UNKNOWN:
        driver = 'undetermined'
    elif sev_cls >= freq_cls:
        driver = 'severity'
    else:
        driver = 'frequency'

    return TailInfo(
        freq=freq_cls, sev=sev_cls, agg=agg_cls,
        freq_lc=freq_lc, sev_lc=sev_lc, agg_lc=None,
        alpha=(alpha if agg_cls == TailClass.POWER_LAW else None),
        driver=driver, flags=flags,
    )


# ----------------------------------------------------------------------------
# Text builders.
# ----------------------------------------------------------------------------

# Width of the label column in ``.info`` output, matching the surrounding lines.
_LABEL_W = 25

# ANSI emphasis for a *thick* (subexponential-or-heavier) tail class when a
# narrative is requested in colour -- bold red, the eye-catch for the heavy side.
_ANSI_THICK = '\x1b[1;31m'
_ANSI_RESET = '\x1b[0m'


def _fmt_bound(x: float) -> str:
    """Format a support bound: ``inf`` / ``-inf``, integer, or 4-sig-fig."""
    if x == np.inf:
        return 'inf'
    if x == -np.inf:
        return '-inf'
    if float(x).is_integer():
        return f'{int(x):,}'
    return f'{x:,.4g}'


def _support_text(lo: float, hi: float) -> str:
    """Support interval text, e.g. ``[0, inf)`` / ``[3, 18]`` / ``(-inf, 1,000]``."""
    lb = '(' if lo == -np.inf else '['
    rb = ')' if hi == np.inf else ']'
    return f'{lb}{_fmt_bound(lo)}, {_fmt_bound(hi)}{rb}'


def _class_label(rung: TailClass, color: bool = False) -> str:
    """Tail-class label, bold-red emphasised when ``color`` and the rung is thick."""
    label = tail_class_label(rung)
    return f'{_ANSI_THICK}{label}{_ANSI_RESET}' if (color and is_thick(rung)) else label


def _sides_text(left: TailClass, right: TailClass, color: bool = False) -> str:
    """Compact per-side tail-class phrase.

    ``bounded`` both ends -> ``'bounded'``; one bounded end names only the open
    side (``'subexponential right tail'``); two open ends name both (or
    ``'<class> both tails'`` when equal).
    """
    B = TailClass.BOUNDED
    if left == B and right == B:
        return 'bounded'
    if left == B:
        return f'{_class_label(right, color)} right tail'
    if right == B:
        return f'{_class_label(left, color)} left tail'
    if left == right:
        return f'{_class_label(left, color)} both tails'
    return f'left {_class_label(left, color)}, right {_class_label(right, color)}'


def describe_row(row, color: bool = False) -> str:
    """One-line phrase for a single :class:`TailRow`: family, support, sides."""
    fam = f'{row.family}, ' if row.family else ''
    return f'{fam}{_support_text(row.min, row.max)}, {_sides_text(row.left_tail, row.right_tail, color)}'


def describe_rows(rows, *, color: bool = False) -> list[str]:
    """Aligned ``.info``-style lines: frequency / severity / aggregate tail.

    The short, three-line summary over the layered :class:`TailRow` report
    (per-component detail is in :func:`explain_rows`). The severity line is the
    combined effective severity when present, else the sole component.

    Parameters
    ----------
    rows : sequence of TailRow
        From :func:`build_tail_rows`.
    color : bool
        Emphasise thick tail classes with ANSI bold-red (for a TTY).

    Returns
    -------
    list of str
        Up to three lines, each ``<label padded to 25><phrase>``.
    """
    by = {r.component: r for r in rows}
    comps = [r for r in rows if r.component.startswith('comp')]
    sev = by.get('severity') or (comps[0] if comps else None)
    out = []
    if 'frequency' in by:
        f = by['frequency']
        fam = f'{f.family}, ' if f.family else ''
        out.append(('frequency tail', f'{fam}count {_support_text(f.min, f.max)}, '
                    f'{_sides_text(f.left_tail, f.right_tail, color)}'))
    if sev is not None:
        out.append(('severity tail', describe_row(sev, color)))
    if 'aggregate' in by:
        a = by['aggregate']
        phrase = f'{_support_text(a.min, a.max)}, {_sides_text(a.left_tail, a.right_tail, color)}'
        if a.cv is not None:
            tag = 'concentrated' if a.concentrated else 'not concentrated'
            phrase += f'; {tag} (cv={a.cv:.3g})'
        if a.note:
            phrase += f' [{a.note}]'
        out.append(('aggregate tail', phrase))
    return [f'{label:<{_LABEL_W}}{phrase}' for label, phrase in out]


def explain_rows(rows, info: Optional[TailInfo] = None, *, color: bool = False) -> str:
    """Verbose prose over the layered :class:`TailRow` report.

    Walks the book bottom-up -- frequency, the severity components and their
    blend, then the aggregate -- naming the single-big-jump mechanism (or the
    frequency driver) for a thick right tail, the power-law moment failure, and
    the concentration.

    Parameters
    ----------
    rows : sequence of TailRow
        From :func:`build_tail_rows`.
    info : TailInfo, optional
        The combine result; used only for the aggregate ``driver``.
    color : bool
        Emphasise thick tail classes with ANSI bold-red.

    Returns
    -------
    str
    """
    by = {r.component: r for r in rows}
    comps = [r for r in rows if r.component.startswith('comp')]
    sev = by.get('severity') or (comps[0] if comps else None)
    out = []

    if 'frequency' in by:
        f = by['frequency']
        fam = (f.family or 'count').capitalize()
        out.append(f'{fam} frequency on counts {_support_text(f.min, f.max)} '
                   f'({_sides_text(f.left_tail, f.right_tail, color)}).')

    if len(comps) > 1 and sev is not None:
        parts = [f'{c.family} {_support_text(c.min, c.max)} '
                 f'({_class_label(c.right_tail, color)})' for c in comps]
        out.append(f'Severity blends {len(comps)} components: ' + '; '.join(parts)
                   + f'. The combined effective severity is {_support_text(sev.min, sev.max)}, '
                   f'{_sides_text(sev.left_tail, sev.right_tail, color)}.')
    elif sev is not None:
        fam = (sev.family or 'severity').capitalize()
        out.append(f'{fam} severity {_support_text(sev.min, sev.max)}, '
                   f'{_sides_text(sev.left_tail, sev.right_tail, color)}.')

    if 'aggregate' in by:
        a = by['aggregate']
        s = (f'The aggregate is {_support_text(a.min, a.max)}, '
             f'{_sides_text(a.left_tail, a.right_tail, color)}.')
        if a.right_tail == TailClass.UNKNOWN:
            s += (' Its right tail is undetermined (an unrecognised family with '
                  'unbounded support; sized conservatively as thick).')
        elif is_thick(a.right_tail):
            driver = getattr(info, 'driver', None)
            if driver == 'frequency':
                s += ' Its right tail is set by the frequency (the severity is lighter).'
            else:
                s += (' Its right tail follows the severity by the single big jump '
                      'P(S>x) ~ E[N]*P(X>x).')
        if is_thick(a.left_tail) and a.left_tail != a.right_tail:
            s += (' Its left tail is heavy (a signed / pnl book reaching far '
                  'below 0); the grid must cover that reach.')
        if a.note:
            s += f' {a.note[0].upper()}{a.note[1:]}.'
        if a.cv is not None:
            tag = 'concentrated' if a.concentrated else 'not concentrated'
            s += f' It is {tag}: cv ~ {a.cv:.3g}.'
        out.append(s)

    return ' '.join(out)


# ----------------------------------------------------------------------------
# Layered thick/thin tail report (the [tail-report] structure).
# ----------------------------------------------------------------------------

def is_thick(rung: TailClass) -> bool:
    """Whether a rung counts as a *thick* tail for bucket selection.

    Thick ⇔ subexponential-or-heavier (:attr:`~TailClass.SUBEXPONENTIAL` or
    :attr:`~TailClass.POWER_LAW`): the regime where the method-of-moments upper
    edge under-reaches and the single-big-jump floor applies. Thin ⇔
    exponential-or-lighter (BOUNDED, SUPER_EXPONENTIAL, EXPONENTIAL).
    :attr:`~TailClass.UNKNOWN` is treated as **thick** -- conservative, since an
    unrecognised family with unbounded support should be sized wide rather than
    clipped. This is the derived helper the sizer applies to a ``left_tail`` /
    ``right_tail`` column (which carry full rungs, not thick/thin).
    """
    if rung == TailClass.UNKNOWN:
        return True
    return rung >= TailClass.SUBEXPONENTIAL


def thickness_label(rung: TailClass) -> str:
    """``'thick'`` / ``'thin'`` label for a rung (see :func:`is_thick`)."""
    return 'thick' if is_thick(rung) else 'thin'


def concentration(m: float, sd: float) -> tuple[Optional[bool], Optional[float]]:
    """Conservative concentration flag and the coefficient of variation.

    Parameters
    ----------
    m, sd : float
        Aggregate mean and standard deviation (loss-space -- the axis the FFT
        and the windowed left-lift operate on).

    Returns
    -------
    (bool or None, float or None)
        ``concentrated`` -- ``True`` iff the band clears 0 by a comfortable
        margin, ``m / sd > 1 / CONCENTRATION_CV`` (equivalently ``cv <
        CONCENTRATION_CV`` for a positive-mean book), so the windowed left-lift
        is eligible -- and ``cv = sd / m``, the coefficient of variation
        (``inf`` when ``m == 0``). ``(None, None)`` when ``sd`` is undefined;
        a deterministic ``sd == 0`` point mass at ``m`` is maximally concentrated
        (``cv = 0``) and clears 0 iff ``m > 0``.

    Notes
    -----
    ``cv`` is directly interpretable, unlike the old ``Phi(mean / sd)``
    diagnostic it replaced (which saturated at ~1 for any real book). It is a
    reported diagnostic, not the gate -- the gate is the conservative
    ``m / sd > 1 / CONCENTRATION_CV`` margin, computed here so the sign of the
    mean is handled correctly (a net-negative-mean book is never concentrated).
    """
    if m is None or sd is None or not np.isfinite(sd):
        return None, None
    if sd <= 0:
        cv = 0.0 if m != 0 else float('inf')
        return bool(m > 0), cv
    z = float(m / sd)
    cv = float('inf') if m == 0 else float(sd / m)
    return bool(z > 1.0 / CONCENTRATION_CV), cv


def _side_class(end_finite: bool, decay: TailClass) -> TailClass:
    """Per-side tail class: ``BOUNDED`` at a finite support end, else the decay.

    A finite support end is a hard boundary (no tail), so that side is
    :attr:`~TailClass.BOUNDED`; an infinite end carries the family's decay rung.
    """
    return TailClass.BOUNDED if end_finite else decay


def _heaviest(rungs) -> TailClass:
    """The heaviest rung in ``rungs``, ranking ``UNKNOWN`` as most conservative."""
    def key(r):
        return int(TailClass.POWER_LAW) + 1 if r == TailClass.UNKNOWN else int(r)
    rungs = list(rungs)
    return max(rungs, key=key) if rungs else TailClass.BOUNDED


def _power_note(alpha: Optional[float]) -> str:
    """Note text for a power-law tail: index and the moment that fails."""
    if alpha is None or not np.isfinite(alpha):
        return 'power-law tail'
    s = f'power-law, alpha={alpha:.3g}'
    if alpha <= 1.0:
        return s + ', infinite mean'
    if alpha < 2.0:
        return s + ', infinite variance'
    return s


def _base_rung(severity) -> TailClass:
    """Un-layered *base-family* thickness, ignoring any finite limit/splice cap.

    The structural-bounded test in :func:`classify_severity` masks a thick base
    once a finite ``limit`` / splice caps it (correctly -- the effective loss is
    bounded). This helper recovers the base thickness (the heavier side) so the
    report can say "thick base, capped at L": it consults the family table
    *without* the structural-bounded short-circuit. Returns
    :attr:`~TailClass.UNKNOWN` for an unrecognised family.
    """
    name = getattr(severity, 'sev_name', None)
    if not isinstance(name, str):
        return TailClass.UNKNOWN
    a = getattr(severity, 'sev_a', np.nan)
    b = getattr(severity, 'sev_b', np.nan)
    left, right, _, _ = _family_sides(name, a, b)
    return _heaviest((left, right))


def severity_support(severity) -> tuple[float, float]:
    """Spec-only claim-space support ``(min, max)`` of one severity component.

    The support of the *layered loss* that feeds the FFT -- after splice,
    reflect, and the ``limit xs attachment`` policy -- read from structural spec
    only (atoms, ``fz.support()``, ``limit``, ``attachment``), so it is valid
    before ``update``.

    Parameters
    ----------
    severity : object
        A single ``aggregate.Severity`` (duck-typed).

    Returns
    -------
    (float, float)
        ``(min, max)``. A discrete / empirical severity returns its atom range
        (negative atoms preserved -> signed). A signed continuous severity
        returns its straddling ``fz.support()`` directly. Otherwise the layered
        loss has ``min = 0`` and ``max = limit`` (finite) or the underlying
        support upper edge less the attachment (possibly ``inf``).
    """
    atoms = getattr(severity, 'support_atoms', None)
    if atoms is not None and len(atoms):
        return float(atoms[0]), float(atoms[-1])

    signed = bool(getattr(severity, 'signed', False))
    try:
        lo, hi = severity.fz.support()
        lo, hi = float(lo), float(hi)
    except Exception:
        lo, hi = 0.0, np.inf

    if signed:
        return lo, hi

    limit = getattr(severity, 'limit', np.inf)
    limit = float(limit) if limit is not None else np.inf
    attach = getattr(severity, 'attachment', 0.0)
    attach = float(attach) if attach is not None else 0.0
    if np.isfinite(limit):
        return 0.0, limit
    s_hi = (hi - attach) if np.isfinite(hi) else np.inf
    return 0.0, max(s_hi, 0.0)


@dataclass(frozen=True)
class TailRow:
    """One row of the layered tail report (one component / layer).

    A row reports a layer's **support** and its **tail class on each side**. The
    public ``bounded`` view is derived in :func:`tail_frame` from ``min`` / ``max``
    (both finite); the thick/thin the sizer needs is the derived :func:`is_thick`
    of ``right_tail`` (or ``left_tail`` for a signed book).

    Attributes
    ----------
    component : str
        Row label: ``'frequency'``, ``'comp0'`` ... (per mix component),
        ``'severity'`` (the combined effective severity), or ``'aggregate'``.
    family : str
        Family / driver label, e.g. ``'poisson'``, ``'lognorm'``,
        ``'2 components'``, ``'severity-driven'``.
    min, max : float
        Structural support -- the smallest / largest **attainable** value
        (``-inf`` / ``inf`` for an unbounded end), not a reach estimate.
    left_tail, right_tail : TailClass
        The tail class of the lower / upper tail: ``BOUNDED`` at a finite support
        end (a hard boundary, no tail), else the family decay rung.
    concentrated : bool or None
        Aggregate row only: the conservative "band clears 0" flag.
    cv : float or None
        Aggregate row only: the coefficient of variation ``sd / mean``
        (``inf`` when ``mean == 0``).
    note : str
        Short structural note (e.g. ``'subexponential base, capped at 1,000'``,
        ``'power-law, alpha=1.5, infinite variance'``).
    """

    component: str
    family: str
    min: float
    max: float
    left_tail: TailClass
    right_tail: TailClass
    concentrated: Optional[bool] = None
    cv: Optional[float] = None
    note: str = ''


def _severity_capped_note(severity, base: TailClass, right_tail: TailClass) -> str:
    """Note for a *recognised heavy* base made effective-bounded by a cap.

    Fires only when a known subexponential-or-heavier family (``base >=
    SUBEXPONENTIAL``, so ``UNKNOWN`` and intrinsically-bounded histograms are
    excluded) has had its right tail capped to ``BOUNDED`` by a finite ``limit``
    / splice -- the "thick base, capped at L" case.
    """
    if right_tail != TailClass.BOUNDED or base < TailClass.SUBEXPONENTIAL:
        return ''
    limit = getattr(severity, 'limit', np.inf)
    ub = getattr(severity, 'sev_ub', np.inf)
    cap = min(float(limit) if limit is not None else np.inf,
              float(ub) if ub is not None else np.inf)
    if np.isfinite(cap):
        return f'{tail_class_label(base)} base, capped at {cap:,.0f}'
    return ''


def _sev_family_label(severity) -> str:
    """Best family label for a severity row (the scipy name, or kind)."""
    name = getattr(severity, 'sev_name', None)
    if isinstance(name, str):
        return name
    kind = getattr(severity, 'sev_kind', '')
    return str(kind) if kind else ''


def severity_tail_row(severity, component: str) -> TailRow:
    """Build the :class:`TailRow` for one severity mix component.

    Parameters
    ----------
    severity : object
        A single ``aggregate.Severity`` (duck-typed).
    component : str
        The row label (e.g. ``'comp0'``).

    Returns
    -------
    TailRow
        With per-side tail classes: a finite support end is ``BOUNDED``, an
        infinite end carries the family decay rung on that side (asymmetric
        families such as ``gumbel_r`` differ left vs right). The note flags an
        uncapped power-law (``alpha`` + the failing moment) or a capped heavy
        base.
    """
    name = getattr(severity, 'sev_name', None)
    a = getattr(severity, 'sev_a', np.nan)
    b = getattr(severity, 'sev_b', np.nan)
    left_base, right_base, _, alpha = _family_sides(name, a, b)
    base = _heaviest((left_base, right_base))
    lo, hi = severity_support(severity)
    left = _side_class(np.isfinite(lo), left_base)
    right = _side_class(np.isfinite(hi), right_base)
    if right == TailClass.POWER_LAW:
        note = _power_note(alpha)
    else:
        note = _severity_capped_note(severity, base, right)
    return TailRow(
        component=component, family=_sev_family_label(severity),
        min=lo, max=hi, left_tail=left, right_tail=right, note=note,
    )


def combined_severity_row(sevs) -> TailRow:
    """Build the combined *effective severity* row (the exposure-weighted blend).

    The support is the component union; each side's tail class is the heaviest
    component's on that side (``UNKNOWN`` ranks most conservative). The note
    carries the thickest power-law ``alpha`` when the blend's right tail is
    power-law.

    Parameters
    ----------
    sevs : sequence
        The severity components (``Aggregate.sevs``); must be non-empty.

    Returns
    -------
    TailRow
        Labelled ``'severity'``.
    """
    rows = [severity_tail_row(s, f'comp{i}') for i, s in enumerate(sevs)]
    _, _, alpha = _combine_severities(sevs)
    lo = min(r.min for r in rows)
    hi = max(r.max for r in rows)
    left = _heaviest(r.left_tail for r in rows)
    right = _heaviest(r.right_tail for r in rows)
    family = f'{len(rows)} components' if len(rows) > 1 else rows[0].family
    note = _power_note(alpha) if right == TailClass.POWER_LAW \
        else 'combined effective severity'
    return TailRow(
        component='severity', family=family,
        min=lo, max=hi, left_tail=left, right_tail=right, note=note,
    )


def occ_net_severity_row(comb: TailRow, occ_reins) -> TailRow:
    """Overlay row: the per-occurrence severity tail NET of occurrence reinsurance.

    Informational only -- the bucket sizer works on the **gross** severity
    (occurrence reinsurance is reported, never a sizing input; see
    ``[reins-gross]``), so this row annotates how the cession reshapes the
    retained per-occurrence tail *without* changing the grid. It is added after
    the combined gross-severity row, before the aggregate.

    Parameters
    ----------
    comb : TailRow
        The combined gross effective-severity row.
    occ_reins : sequence of (share, limit, attach)
        The occurrence-reinsurance layers (``Aggregate.occ_reins``).

    Returns
    -------
    TailRow
        Labelled ``'severity (net occ)'``.

    Notes
    -----
    The net is bounded above only when a top layer cedes **100%** of everything
    above its attachment to infinity (``share >= 1`` and ``limit == inf``): the
    net is then capped at that attachment. Finite layers, or partial
    (``share < 1``) unlimited cessions, leave the gross tail class in place (the
    retained tail is the same family, merely scaled).
    """
    net_cap = np.inf
    for (s, y, a) in occ_reins:
        if not np.isfinite(y) and float(s) >= 1.0:
            net_cap = min(net_cap, float(a))
    if np.isfinite(net_cap) and net_cap < comb.max:
        hi, right = net_cap, TailClass.BOUNDED
        note = f'net of occ reins: capped at {net_cap:.6g}'
    else:
        hi, right = comb.max, comb.right_tail
        note = 'net of occ reins: tail retained'
    return TailRow(
        component='severity (net occ)', family=comb.family,
        min=comb.min, max=float(hi), left_tail=comb.left_tail, right_tail=right,
        note=note,
    )


def frequency_tail_row(frequency, *, n_min: float = 0.0,
                       n_max: float = np.inf,
                       zero_truncated: bool = False) -> TailRow:
    """Build the frequency :class:`TailRow` (claim-count layer).

    Parameters
    ----------
    frequency : object
        The ``aggregate.Frequency`` (read for ``freq_name``).
    n_min, n_max : float
        Smallest / largest attainable claim count (supplied by the caller, which
        knows the exposure ``n``).
    zero_truncated : bool
        Whether the count is genuinely zero-truncated (``zm`` with ``p0 == 0``).

    Returns
    -------
    TailRow
        Labelled ``'frequency'``; the left tail is ``BOUNDED`` (counts have a
        finite floor), the right tail is ``BOUNDED`` for a finite-count family
        else the family decay rung.
    """
    rung, _ = classify_frequency(frequency)
    family = getattr(frequency, 'freq_name', '') or ''
    note = 'claim count'
    if zero_truncated:
        note += ', zero-truncated'
    return TailRow(
        component='frequency', family=family,
        min=float(n_min), max=float(n_max),
        left_tail=_side_class(np.isfinite(n_min), rung),
        right_tail=_side_class(np.isfinite(n_max), rung),
        note=note,
    )


def aggregate_tail_row(info: TailInfo, *, agg_min: float, agg_max: float,
                       left_decay: TailClass, right_decay: TailClass,
                       actual_m: float = np.nan, actual_sd: float = np.nan) -> TailRow:
    """Build the aggregate :class:`TailRow` from the structural support + decays.

    Parameters
    ----------
    info : TailInfo
        The combine result (:func:`aggregate_tail_info`) -- supplies the driver
        label and the power-law ``alpha``.
    agg_min, agg_max : float
        Structural aggregate support (``-inf`` / ``inf`` at an unbounded end),
        already mapped through any ``pnl`` affine by the caller.
    left_decay, right_decay : TailClass
        The aggregate's per-side decay rungs (used only at an infinite end),
        already affine-mapped (swapped under a reflecting ``pnl``).
    actual_m, actual_sd : float
        Loss-space aggregate mean / sd (drive :func:`concentration`).

    Returns
    -------
    TailRow
        Labelled ``'aggregate'``.
    """
    left = _side_class(np.isfinite(agg_min), left_decay)
    right = _side_class(np.isfinite(agg_max), right_decay)
    conc, conc_cv = concentration(actual_m, actual_sd)
    family = f'{info.driver}-driven' if info.driver != 'undetermined' else ''
    note = _power_note(info.alpha) if right == TailClass.POWER_LAW else ''
    return TailRow(
        component='aggregate', family=family,
        min=float(agg_min), max=float(agg_max),
        left_tail=left, right_tail=right,
        concentrated=conc, cv=conc_cv, note=note,
    )


def _mul_extreme(n: float, s: float) -> float:
    """``n * s`` with the ``0 * inf`` indeterminate resolved to ``0``."""
    if n == 0.0 or s == 0.0:
        return 0.0
    return float(n) * float(s)


def _agg_support(n_lo: float, n_hi: float,
                 s_lo: float, s_hi: float) -> tuple[float, float]:
    """Structural support of ``S = sum_{N} X`` from count and severity extents.

    ``N in [n_lo, n_hi]`` claims, each severity in ``[s_lo, s_hi]``. The maximum
    sum takes the count that maximises the (signed) per-claim extreme, likewise
    the minimum -- so an unbounded count or severity pushes the active side to
    ``+-inf``. Exact for a bounded book (e.g. fixed 3 x dice [1, 6] -> [3, 18]).
    """
    hi = _mul_extreme(n_hi, s_hi) if s_hi > 0 else _mul_extreme(n_lo, s_hi)
    lo = _mul_extreme(n_hi, s_lo) if s_lo < 0 else _mul_extreme(n_lo, s_lo)
    return lo, hi


def build_tail_rows(frequency, sevs, *, freq_min: float = 0.0,
                    freq_max: float = np.inf, freq_zero_truncated: bool = False,
                    actual_m: float = np.nan, actual_sd: float = np.nan,
                    occ_reins=None) -> list[TailRow]:
    """Assemble the layered tail report as an ordered list of :class:`TailRow`.

    Rows, bottom-up: frequency; one per severity mix component (``comp0`` ...);
    the combined effective severity (only when there is more than one component);
    an optional occurrence-reinsurance overlay (``severity (net occ)``, only when
    ``occ_reins`` is given); the aggregate. Spec-only -- valid before ``update``.

    The aggregate's **structural support** is built from the count and combined
    severity extents (:func:`_agg_support`); its per-side decay rungs combine the
    frequency rung with the combined severity's per-side rungs (single big jump).

    Parameters
    ----------
    frequency : object
        The aggregate's ``Frequency``.
    sevs : sequence or None
        The severity components (``Aggregate.sevs``).
    freq_min, freq_max : float
        Claim-count support (the caller knows the exposure).
    freq_zero_truncated : bool
        Genuine zero-truncation flag for the frequency note.
    actual_m, actual_sd : float
        Loss-space aggregate mean / sd (drive concentration).
    occ_reins : sequence of (share, limit, attach), optional
        The occurrence-reinsurance layers (``Aggregate.occ_reins``). When given,
        an informational ``severity (net occ)`` overlay row is appended after the
        combined gross severity (the sizer works on gross -- this row only
        annotates the retained tail; see :func:`occ_net_severity_row`).

    Returns
    -------
    list of TailRow
    """
    info = aggregate_tail_info(frequency, sevs)
    rows = [frequency_tail_row(frequency, n_min=freq_min, n_max=freq_max,
                               zero_truncated=freq_zero_truncated)]
    sev_list = list(sevs) if sevs is not None else []
    rows.extend(severity_tail_row(s, f'comp{i}') for i, s in enumerate(sev_list))

    if sev_list:
        comb = combined_severity_row(sev_list)
        if len(sev_list) > 1:
            rows.append(comb)
        if occ_reins is not None and len(occ_reins):
            rows.append(occ_net_severity_row(comb, occ_reins))
        s_lo, s_hi = comb.min, comb.max
        sev_left, sev_right = comb.left_tail, comb.right_tail
    else:
        s_lo, s_hi = 0.0, np.inf
        sev_left, sev_right = TailClass.BOUNDED, TailClass.UNKNOWN

    freq_rung, _ = classify_frequency(frequency)
    loss_lo, loss_hi = _agg_support(freq_min, freq_max, s_lo, s_hi)
    left_decay = combine(freq_rung, sev_left)
    right_decay = combine(freq_rung, sev_right)

    agg_min, agg_max = loss_lo, loss_hi

    rows.append(aggregate_tail_row(
        info, agg_min=agg_min, agg_max=agg_max,
        left_decay=left_decay, right_decay=right_decay,
        actual_m=actual_m, actual_sd=actual_sd))
    return rows


def tail_frame(rows) -> 'pd.DataFrame':
    """Render a list of :class:`TailRow` as the public ``tail_behavior_df`` DataFrame.

    Parameters
    ----------
    rows : sequence of TailRow
        From :func:`build_tail_rows`.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``component`` (frequency / comp* / severity / aggregate), with
        columns ``family, min, max, left_tail, right_tail, bounded, concentrated,
        cv, note``. ``bounded`` is derived as ``min`` and ``max`` both finite.
    """
    data = [{
        'component': r.component, 'family': r.family,
        'min': r.min, 'max': r.max,
        'left_tail': tail_class_label(r.left_tail),
        'right_tail': tail_class_label(r.right_tail),
        'bounded': bool(np.isfinite(r.min) and np.isfinite(r.max)),
        'concentrated': r.concentrated, 'cv': r.cv,
        'note': r.note,
    } for r in rows]
    cols = ['component', 'family', 'min', 'max', 'left_tail', 'right_tail',
            'bounded', 'concentrated', 'cv', 'note']
    return pd.DataFrame(data, columns=cols).set_index('component')
