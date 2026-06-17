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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

__all__ = [
    'TailClass', 'TailClasses', 'TailInfo', 'TailRow',
    'classify_frequency', 'classify_severity', 'combine',
    'aggregate_tail_info', 'tail_class_label',
    'is_thick', 'thickness_label', 'severity_support',
    'build_tail_rows', 'tail_frame',
    'describe_lines', 'explain',
    'CONCENTRATION_CV',
    '_BOUNDED_FREQS', '_BOUNDED_SCIPY_SEVS',
]


# Conservative concentration cutoff: a book is "concentrated" (its mass band
# clears 0, so the windowed left-lift is eligible) only when its coefficient of
# variation is comfortably small -- the band then sits >= 1/CONCENTRATION_CV
# standard deviations above 0. Tighter than the legacy 1/z ~ 0.14 gate: lifting
# x_min when the band does not really clear 0 clips left-tail mass, so we err
# toward *not* windowing when marginal. See plan-univariate-bucket [tail-report].
CONCENTRATION_CV = 0.1


# ----------------------------------------------------------------------------
# Bounded-support tables (moved here from distributions.py; single source).
# ----------------------------------------------------------------------------

# Frequencies with bounded support.
_BOUNDED_FREQS = frozenset({'fixed', 'bernoulli', 'binomial', 'empirical'})

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

# scipy families with a fixed (param-independent) class and log-concavity flag.
SCIPY_SEV_TAIL: dict[str, tuple[TailClass, Optional[bool]]] = {
    'norm':       (TailClass.SUPER_EXPONENTIAL, True),
    'expon':      (TailClass.EXPONENTIAL, True),
    'laplace':    (TailClass.EXPONENTIAL, True),
    'logistic':   (TailClass.EXPONENTIAL, True),
    'lognorm':    (TailClass.SUBEXPONENTIAL, False),
    'invgauss':   (TailClass.EXPONENTIAL, False),   # semi-heavy; watch item
}

# Power-law families: scipy name -> callable(sev_a, sev_b) -> tail index alpha.
# alpha is the exponent in P(X > x) ~ x**(-alpha). The shape slot differs by
# family, which is a classic bug source, so each is mapped explicitly.
_POWER_LAW_ALPHA = {
    'pareto':     lambda a, b: a,            # scipy shape b == sev_a
    'lomax':      lambda a, b: a,            # Pareto type II, shape c == sev_a
    'fisk':       lambda a, b: a,            # log-logistic, shape c == sev_a
    'loglogistic':lambda a, b: a,
    't':          lambda a, b: a,            # Student-t, df == sev_a
    'cauchy':     lambda a, b: 1.0,          # alpha == 1
    'invweibull': lambda a, b: a,            # Frechet, shape c == sev_a
    'frechet':    lambda a, b: a,
    'invgamma':   lambda a, b: a,            # tail ~ x**(-(a+1)) -> index a
    'burr':       lambda a, b: a * b,        # Burr XII, c*d == sev_a*sev_b
    'burr12':     lambda a, b: a * b,
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
    return False


def classify_severity(severity) -> tuple[TailClass, Optional[bool], Optional[float]]:
    """Classify a single severity component's tail.

    Parameters
    ----------
    severity : object
        A single ``aggregate.Severity`` (duck-typed). Reads ``sev_kind``,
        ``sev_name``, ``sev_a``, ``sev_b``, ``limit``, ``sev_ub``.

    Returns
    -------
    (TailClass, bool or None, float or None)
        The rung, the log-concavity flag, and the power-law index ``alpha``
        (``None`` unless the rung is POWER_LAW).

    Notes
    -----
    Checks run in order: **structural-bounded → family lookup → param-aware
    family → UNKNOWN**, returning at the first hit. The structural-bounded test
    is first so ``bounded`` is decided from the spec alone (no density), which
    the lifted-natural-allocation guard relies on. Param-aware families:
    ``gamma`` is log-concave iff shape ``≥ 1``; ``weibull_min`` with shape
    ``c > 1`` is super-exponential (log-concave), ``c == 1`` is the exponential,
    ``c < 1`` is subexponential; ``genpareto`` with shape ``ξ > 0`` is power-law
    with ``alpha = 1/ξ`` (``ξ == 0`` reduces to the exponential).
    """
    if _severity_bounded(severity):
        return TailClass.BOUNDED, None, None

    name = getattr(severity, 'sev_name', None)
    if not isinstance(name, str):
        # meta / copy that was not bounded, or an unrecognised wrapper.
        return TailClass.UNKNOWN, None, None

    a = getattr(severity, 'sev_a', np.nan)
    b = getattr(severity, 'sev_b', np.nan)

    # Param-aware families first.
    if name == 'gamma':
        return TailClass.EXPONENTIAL, bool(np.isfinite(a) and a >= 1.0), None
    if name == 'weibull_min':
        if not np.isfinite(a):
            return TailClass.EXPONENTIAL, None, None
        if a > 1.0:
            return TailClass.SUPER_EXPONENTIAL, True, None
        if a == 1.0:
            return TailClass.EXPONENTIAL, True, None      # c == 1 is exponential
        return TailClass.SUBEXPONENTIAL, False, None
    if name == 'genpareto':
        # scipy shape c == ξ. ξ > 0 heavy (alpha = 1/ξ); ξ == 0 exponential.
        if np.isfinite(a) and a > 0:
            return TailClass.POWER_LAW, False, 1.0 / a
        return TailClass.EXPONENTIAL, None, None

    if name in _POWER_LAW_ALPHA:
        alpha = float(_POWER_LAW_ALPHA[name](a, b))
        return TailClass.POWER_LAW, False, alpha

    if name in SCIPY_SEV_TAIL:
        cls, lc = SCIPY_SEV_TAIL[name]
        return cls, lc, None

    return TailClass.UNKNOWN, None, None


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


def _rung_phrase(rung: TailClass, lc: Optional[bool], family: Optional[str] = None,
                 alpha: Optional[float] = None) -> str:
    """Format one rung as ``[log-concave, ]<rung>[ (family)][, alpha=...]``."""
    parts = []
    if lc:
        parts.append('log-concave')
    label = tail_class_label(rung)
    if rung == TailClass.POWER_LAW and alpha is not None and np.isfinite(alpha):
        label = f'{label} (alpha={alpha:.3g})'
    parts.append(label)
    text = ', '.join(parts)
    if family:
        text = f'{text} ({family})'
    return text


def describe_lines(info: TailInfo, freq_label: str = '', sev_label: str = '') -> list[str]:
    """Three aligned ``.info``-style lines: frequency / severity / aggregate tail.

    Parameters
    ----------
    info : TailInfo
        The struct from :func:`aggregate_tail_info`.
    freq_label, sev_label : str
        Family names to show in parentheses (e.g. ``'poisson'``, ``'lognorm'``).

    Returns
    -------
    list of str
        Three lines, each ``<label padded to 25><phrase>``.
    """
    rows = [
        ('frequency tail', _rung_phrase(info.freq, info.freq_lc, freq_label)),
        ('severity tail', _rung_phrase(info.sev, info.sev_lc, sev_label, info.alpha)),
        ('aggregate tail', _rung_phrase(info.agg, info.agg_lc, None, info.alpha)),
    ]
    return [f'{label:<{_LABEL_W}}{phrase}' for label, phrase in rows]


def explain(info: TailInfo, freq_label: str = '', sev_label: str = '') -> str:
    """One-sentence explanation of how the aggregate tail class arises.

    Parameters
    ----------
    info : TailInfo
        The struct from :func:`aggregate_tail_info`.
    freq_label, sev_label : str
        Frequency and severity family names.

    Returns
    -------
    str
        A human-readable sentence; notes the single-big-jump mechanism for
        heavy severity, the infinite-variance / infinite-mean flags for
        power-law tails, and the analytic-PGF caveat is implicit in the
        ``undetermined`` wording for unrecognised families.
    """
    fl = f' {freq_label}' if freq_label else ''
    sl = f' {sev_label}' if sev_label else ''

    if info.agg == TailClass.UNKNOWN:
        return ('Aggregate tail undetermined (one or more components is an '
                'unrecognised family with unbounded support; it is sized '
                'conservatively as thick).')

    freq_phrase = ('log-concave ' if info.freq_lc else '') + tail_class_label(info.freq)
    sev_phrase = ('log-concave ' if info.sev_lc else '') + tail_class_label(info.sev)
    agg_label = tail_class_label(info.agg)

    sentence = (f'{freq_phrase.capitalize()}{fl} frequency and {sev_phrase}{sl} '
                f'severity give a {agg_label} aggregate')

    if info.agg in (TailClass.SUBEXPONENTIAL, TailClass.POWER_LAW):
        sentence += ' (single big jump: P(S>x) approx E[N]*P(X>x))'
    elif info.driver == 'frequency':
        sentence += ' (frequency-driven; the severity is lighter)'
    else:
        sentence += ' (tail set by the heavier of frequency and severity decay)'

    if info.agg == TailClass.POWER_LAW and info.alpha is not None and np.isfinite(info.alpha):
        sentence += f'. Tail index alpha = {info.alpha:.3g}'
        if info.alpha <= 1.0:
            sentence += ' (infinite mean)'
        elif info.alpha < 2.0:
            sentence += ' (infinite variance)'

    return sentence + '.'


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
    clipped.
    """
    if rung == TailClass.UNKNOWN:
        return True
    return rung >= TailClass.SUBEXPONENTIAL


def thickness_label(rung: TailClass) -> str:
    """``'thick'`` / ``'thin'`` label for a rung (see :func:`is_thick`)."""
    return 'thick' if is_thick(rung) else 'thin'


def concentration(agg_cv) -> tuple[Optional[bool], Optional[float]]:
    """Conservative concentration flag and margin from the aggregate ``cv``.

    Parameters
    ----------
    agg_cv : float
        The aggregate coefficient of variation ``sd / mean``.

    Returns
    -------
    (bool or None, float or None)
        ``concentrated`` -- ``True`` iff ``cv < CONCENTRATION_CV`` (the mass band
        clears 0 by a comfortable margin, so the windowed left-lift is eligible)
        -- and ``concentration_p`` -- the sd-count ``1 / cv`` by which the band
        clears 0 (reporting only, not a separate gate). ``(None, None)`` when the
        ``cv`` is undefined (e.g. a mean-0 signed P&L); ``(True, inf)`` for a
        deterministic ``cv == 0``.
    """
    if agg_cv is None or not np.isfinite(agg_cv):
        return None, None
    if agg_cv <= 0:
        return True, np.inf
    return bool(agg_cv < CONCENTRATION_CV), float(1.0 / agg_cv)


def _base_rung(severity) -> TailClass:
    """Un-layered *base-family* thickness, ignoring any finite limit/splice cap.

    The structural-bounded test in :func:`classify_severity` masks a thick base
    once a finite ``limit`` / splice caps it (correctly -- the effective loss is
    bounded). This helper recovers the base thickness so the report can say
    "thick base, capped at L": it repeats the family lookup *without* the
    structural-bounded short-circuit. Returns :attr:`~TailClass.UNKNOWN` for an
    unrecognised family.
    """
    name = getattr(severity, 'sev_name', None)
    if not isinstance(name, str):
        return TailClass.UNKNOWN
    a = getattr(severity, 'sev_a', np.nan)
    b = getattr(severity, 'sev_b', np.nan)
    if name == 'gamma':
        return TailClass.EXPONENTIAL
    if name == 'weibull_min':
        if not np.isfinite(a):
            return TailClass.EXPONENTIAL
        if a > 1.0:
            return TailClass.SUPER_EXPONENTIAL
        if a == 1.0:
            return TailClass.EXPONENTIAL
        return TailClass.SUBEXPONENTIAL
    if name == 'genpareto':
        if np.isfinite(a) and a > 0:
            return TailClass.POWER_LAW
        return TailClass.EXPONENTIAL
    if name in _POWER_LAW_ALPHA:
        return TailClass.POWER_LAW
    if name in SCIPY_SEV_TAIL:
        return SCIPY_SEV_TAIL[name][0]
    if name in _BOUNDED_SCIPY_SEVS:
        return TailClass.BOUNDED
    return TailClass.UNKNOWN


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

    Attributes
    ----------
    component : str
        Row label: ``'frequency'``, ``'comp0'`` ... (per mix component),
        ``'severity'`` (the combined effective severity), or ``'aggregate'``.
    family : str
        Family / driver label, e.g. ``'poisson'``, ``'lognorm'``,
        ``'2 components'``, ``'severity-driven'``.
    min, max : float
        Smallest / largest attainable value (``max`` may be ``inf``).
    bounded : bool
        ``min`` and ``max`` both finite (``rung is BOUNDED``).
    left, right : str
        ``'thick'`` / ``'thin'`` lower- and upper-tail labels (see
        :func:`is_thick`). ``left`` is ``'thin'`` for non-negative support.
    rung : TailClass
        The effective tail rung.
    alpha : float or None
        Power-law tail index, when applicable.
    log_concave : bool or None
        Log-concavity flag (``None`` when not determined).
    concentrated : bool or None
        Aggregate row only: the conservative "band clears 0" flag.
    concentration_p : float or None
        Aggregate row only: the sd-margin the band clears 0 (``1 / cv``).
    note : str
        Short structural note (e.g. ``'subexponential base, capped at 1,000'``).
    """

    component: str
    family: str
    min: float
    max: float
    bounded: bool
    left: str
    right: str
    rung: TailClass
    alpha: Optional[float] = None
    log_concave: Optional[bool] = None
    concentrated: Optional[bool] = None
    concentration_p: Optional[float] = None
    note: str = ''


def _severity_note(severity, eff_rung: TailClass) -> str:
    """Structural note for a severity row: flag a *recognised heavy* base capped.

    Fires only when a known subexponential-or-heavier family (``base >=
    SUBEXPONENTIAL``, so ``UNKNOWN`` and intrinsically-bounded histograms are
    excluded) has been made effective-bounded by a finite ``limit`` / splice cap
    -- the "thick base, capped at L" case.
    """
    base = _base_rung(severity)
    limit = getattr(severity, 'limit', np.inf)
    ub = getattr(severity, 'sev_ub', np.inf)
    cap = min(float(limit) if limit is not None else np.inf,
              float(ub) if ub is not None else np.inf)
    if (eff_rung == TailClass.BOUNDED and base >= TailClass.SUBEXPONENTIAL
            and np.isfinite(cap)):
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
    """
    rung, lc, alpha = classify_severity(severity)
    lo, hi = severity_support(severity)
    signed = bool(getattr(severity, 'signed', False))
    left = thickness_label(rung) if signed else 'thin'
    return TailRow(
        component=component, family=_sev_family_label(severity),
        min=lo, max=hi, bounded=(rung == TailClass.BOUNDED),
        left=left, right=thickness_label(rung),
        rung=rung, alpha=alpha, log_concave=lc,
        note=_severity_note(severity, rung),
    )


def combined_severity_row(sevs) -> TailRow:
    """Build the combined *effective severity* row (the exposure-weighted blend).

    The blend mirrors :func:`_combine_severities`: thickest rung (UNKNOWN
    poisons), ``log_concave = all`` components, the thickest power-law ``alpha``.
    The support is the component union and the left tail is thick iff any
    component's is.

    Parameters
    ----------
    sevs : sequence
        The severity components (``Aggregate.sevs``); must be non-empty.

    Returns
    -------
    TailRow
        Labelled ``'severity'``.
    """
    rung, lc, alpha = _combine_severities(sevs)
    rows = [severity_tail_row(s, f'comp{i}') for i, s in enumerate(sevs)]
    lo = min(r.min for r in rows)
    hi = max(r.max for r in rows)
    left = 'thick' if any(r.left == 'thick' for r in rows) else 'thin'
    family = f'{len(rows)} components' if len(rows) > 1 else rows[0].family
    return TailRow(
        component='severity', family=family,
        min=lo, max=hi, bounded=(rung == TailClass.BOUNDED),
        left=left, right=thickness_label(rung),
        rung=rung, alpha=alpha, log_concave=lc,
        note='combined effective severity',
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
        Whether the count is genuinely zero-truncated (``zm`` with ``p0 == 0``),
        as opposed to merely having a positive minimum atom.

    Returns
    -------
    TailRow
        Labelled ``'frequency'``; the left tail is always thin (counts >= 0).
    """
    rung, lc = classify_frequency(frequency)
    family = getattr(frequency, 'freq_name', '') or ''
    note = 'claim count'
    if zero_truncated:
        note += ', zero-truncated'
    return TailRow(
        component='frequency', family=family,
        min=float(n_min), max=float(n_max), bounded=(rung == TailClass.BOUNDED),
        left='thin', right=thickness_label(rung),
        rung=rung, alpha=None, log_concave=lc, note=note,
    )


def aggregate_tail_row(info: TailInfo, *, agg_min: float = np.nan,
                       agg_max: float = np.nan, agg_cv: float = np.nan,
                       left: str = 'thin') -> TailRow:
    """Build the aggregate :class:`TailRow` from the combine result and reach.

    Parameters
    ----------
    info : TailInfo
        The combine result (:func:`aggregate_tail_info`).
    agg_min, agg_max : float
        Estimated lower / upper reach of the aggregate (the caller's
        method-of-moments percentiles).
    agg_cv : float
        Aggregate coefficient of variation (drives :func:`concentration`).
    left : str
        The resolved lower-tail label. ``'thin'`` for a non-negative aggregate;
        the caller mirrors the appropriate severity tail for a signed one (the
        reflected severity's left for an ``ssev`` / ``dsev`` book, the loss's
        right tail for an affine ``pnl`` book) -- see the ``[q-left-combine]``
        rule.

    Returns
    -------
    TailRow
        Labelled ``'aggregate'``.
    """
    rung = info.agg
    conc, conc_p = concentration(agg_cv)
    family = f'{info.driver}-driven' if info.driver != 'undetermined' else ''
    return TailRow(
        component='aggregate', family=family,
        min=float(agg_min), max=float(agg_max), bounded=(rung == TailClass.BOUNDED),
        left=left, right=thickness_label(rung),
        rung=rung, alpha=info.alpha, log_concave=info.agg_lc,
        concentrated=conc, concentration_p=conc_p,
        note='frequency (x) combined severity',
    )


def build_tail_rows(frequency, sevs, *, freq_min: float = 0.0,
                    freq_max: float = np.inf, freq_zero_truncated: bool = False,
                    agg_min: float = np.nan, agg_max: float = np.nan,
                    agg_cv: float = np.nan, agg_left: str = 'thin') -> list[TailRow]:
    """Assemble the layered tail report as an ordered list of :class:`TailRow`.

    Rows, bottom-up: frequency; one per severity mix component (``comp0`` ...);
    the combined effective severity (only when there is more than one component);
    the aggregate. Spec-only -- valid before ``update``.

    Parameters
    ----------
    frequency : object
        The aggregate's ``Frequency``.
    sevs : sequence or None
        The severity components (``Aggregate.sevs``).
    freq_min, freq_max : float
        Claim-count support (the caller knows the exposure).
    agg_min, agg_max : float
        Aggregate reach estimates.
    agg_cv : float
        Aggregate coefficient of variation.
    agg_left : str
        The resolved aggregate lower-tail label (the caller resolves the signed
        mirror; see :func:`aggregate_tail_row`).

    Returns
    -------
    list of TailRow
    """
    info = aggregate_tail_info(frequency, sevs)
    rows = [frequency_tail_row(frequency, n_min=freq_min, n_max=freq_max,
                               zero_truncated=freq_zero_truncated)]
    sev_list = list(sevs) if sevs is not None else []
    rows.extend(severity_tail_row(s, f'comp{i}') for i, s in enumerate(sev_list))
    if sev_list and len(sev_list) > 1:
        rows.append(combined_severity_row(sev_list))
    rows.append(aggregate_tail_row(
        info, agg_min=agg_min, agg_max=agg_max,
        agg_cv=agg_cv, left=agg_left))
    return rows


def tail_frame(rows) -> 'pd.DataFrame':
    """Render a list of :class:`TailRow` as the public ``tail_df`` DataFrame.

    Parameters
    ----------
    rows : sequence of TailRow
        From :func:`build_tail_rows`.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``component`` (frequency / comp* / severity / aggregate), with
        columns ``family, min, max, left, right, bounded, tail_class, alpha,
        log_concave, concentrated, concentration_p, note``.
    """
    data = [{
        'component': r.component, 'family': r.family,
        'min': r.min, 'max': r.max,
        'left': r.left, 'right': r.right, 'bounded': r.bounded,
        'tail_class': tail_class_label(r.rung),
        'alpha': r.alpha, 'log_concave': r.log_concave,
        'concentrated': r.concentrated, 'concentration_p': r.concentration_p,
        'note': r.note,
    } for r in rows]
    cols = ['component', 'family', 'min', 'max', 'left', 'right', 'bounded',
            'tail_class', 'alpha', 'log_concave', 'concentrated',
            'concentration_p', 'note']
    return pd.DataFrame(data, columns=cols).set_index('component')
