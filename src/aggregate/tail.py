"""Ordered tail-thickness classification for frequencies, severities, and aggregates.

This module is the **single source of truth** for tail shape. It exposes an
ordered 5-rung scale (:class:`TailClass`) plus a separate ``log_concave`` flag,
deterministic family-lookup classifiers for frequency and severity, the
``combine`` rule that produces the aggregate class, and text builders for the
``.info`` / ``.tail_description`` / ``.tail_explanation`` surfaces.

The bounded-support tables (:data:`_BOUNDED_FREQS`, :data:`_BOUNDED_SCIPY_SEVS`)
live here, and ``Aggregate.bounded`` / ``Severity.bounded`` / ``Portfolio.bounded``
are derived from the classifier (``bounded ⇔ tail class is BOUNDED``).

Design notes
------------
* **Leaf module.** Classifiers take duck-typed objects and read attributes
  (``freq_name``, ``sev_name``, ``sev_a``, ``sev_b``, ``sev_kind``, ``limit``,
  ``sev_ub``, ``_certified_bounded``) — they do **not** import ``distributions``
  or ``portfolio``, so those modules can import ``tail`` with no cycle.
* **Phase 1 (this module) is exact and deterministic.** Families not in the
  tables resolve to :attr:`TailClass.UNKNOWN`; the numeric density-tail estimator
  (mean-excess slope, log-log-S slope, discrete log-concavity) is deferred to a
  future Phase 2 and is *not* implemented here.
* **bounded is spec-only.** The BOUNDED determination uses only structural spec
  information (finite family / atom / finite layer or splice cap / certify
  override), never a computed density — so ``.bounded`` is correct *before*
  ``update()``.

Notes
-----
The aggregate combine rule ``agg = max(freq, sev)`` is exact under the standard
families supported here: when severity is subexponential or heavier the single-
big-jump principle gives ``P(S>x) ~ E[N]·P(X>x)`` so the aggregate inherits the
severity class; when severity is light, the compound decay is set by the heavier
of the severity and frequency decay rates. This holds while the frequency PGF is
analytic at 1 (all standard frequencies); genuinely heavy mixing (PIG / Sichel /
Neyman-A) is a watch item that the deferred numeric estimator can refine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple, Optional

import numpy as np

__all__ = [
    'TailClass', 'TailClasses', 'TailInfo',
    'classify_frequency', 'classify_severity', 'combine',
    'aggregate_tail_info', 'tail_class_label',
    'describe_lines', 'explain',
    '_BOUNDED_FREQS', '_BOUNDED_SCIPY_SEVS',
]


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
    represents "not yet determined" (e.g. a histogram or unrecognised family in
    Phase 1) and *poisons* :func:`combine` rather than masquerading as a
    thickness — see :func:`combine`.

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
        Log-concavity flags. ``agg_lc`` is ``None`` in Phase 1 (a random sum
        does not inherit log-concavity analytically; the numeric estimator that
        would set it is deferred to Phase 2). ``None`` means "not determined".
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
# EXPONENTIAL (geometric-type tail) with log_concave left None — provisional,
# a Phase-2 numeric refinement can promote genuinely heavy mixing.
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
    'invgauss':   (TailClass.EXPONENTIAL, False),   # semi-heavy; Phase-2 watch
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
        (``agg_lc`` is ``None`` in Phase 1), the power-law ``alpha`` if any,
        the driver, and structured flags.

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
                'unrecognised or numeric-only family; a numeric density '
                'estimate is pending).')

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
