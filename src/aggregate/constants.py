"""Dependency-free leaf holding the few non-settings constants and types that
low-level ``aggregate`` modules need.

This module is deliberately tiny and import-free so it can sit *below*
``distributions`` / ``portfolio`` in the import graph without risking a cycle:
``utilities`` imports :class:`Validation` and ``spectral`` uses
:class:`DefectiveDistributionWarning`, and both are imported by
``distributions`` / ``portfolio``. A ``Flag`` enum and a ``Warning`` subclass
are types, not user settings, so they do not belong in :mod:`aggregate.config`.

The reinsurance labels are *structural* MultiIndex column / axis keys
(referenced by literal in ``bivariate`` and asserted across the reins
tests), so they are constants here, not user-tunable settings.

The first-class-citizen (FCC) contract lives here for the same reason: it is a
tuple of *names*, so stating it costs no imports, and both
``dev/regen_features.py`` and ``tests/test_fcc_surface.py`` read the one copy.

All tunable defaults (grid sizing, databases, discretization schemes,
validation tolerances — including the former numerics noise floors) and the
path names now live in :mod:`aggregate.config`. The plotting figure constants
below stay here permanently (used in default-argument expressions; user
restyling goes through matplotlib's native ``mplstyle`` / ``rcParams``).
"""

import warnings
from contextlib import contextmanager
from enum import Flag, auto


__all__ = ['FIG_W', 'FIG_H', 'FONT_SIZE', 'LEGEND_FONT',
           'PLOT_FACE_COLOR', 'FIGURE_BG_COLOR', 'LOG_FLOOR',
           'DISTORTION_DUAL_LABEL', 'DISTORTION_DUAL_TEX',
           'Validation', 'DefectiveDistributionWarning',
           'DefectiveDistributionError', 'InfiniteVarianceError',
           'IgnoredDecLClauseWarning', 'InfeasibleGridWarning',
           'ZeroModifiedExposureWarning',
           'ZeroPremiumCessionWarning',
           'CoarseJointGridWarning', 'DegenerateEvaluationWarning',
           'ReflectedSeverityClampWarning',
           'warn_once', 'reset_warn_once', 'warn_once_isolated',
           'REINS_LABEL_GROSS', 'REINS_LABEL_SUBJECT', 'REINS_LABEL_NET',
           'REINS_LABEL_CEDED', 'REINS_LABEL_OUTPUT',
           'INFO_LABEL_WIDTH', 'INFO_NA', 'info_row',
           'FIRST_CLASS_CLASSES', 'NEAR_FIRST_CLASS', 'FCC_REQUIRED',
           'FCC_CONTRACT_EXCEPTIONS', 'FCC_UNPAIRED_NARRATIVES']

# --- plotting figure defaults (permanently here, not config) ---------------
# These are used as module-level constants in default argument expressions
# (e.g. ``figsize=(2 * FIG_W, FIG_H)``); a config [plotting] section was
# considered and rejected (see dev/done/plans-considered-and-rejected.md) --
# users restyle via matplotlib's native mplstyle / rcParams.
FIG_W = 3.5
FIG_H = 2.45
FONT_SIZE = 9
LEGEND_FONT = 'x-small'
# see https://matplotlib.org/stable/gallery/color/named_colors.html
PLOT_FACE_COLOR = 'lightsteelblue'
FIGURE_BG_COLOR = 'aliceblue'
# The one float-dust floor for anything drawn on a log scale: below this a
# value is arithmetic noise, not tail, and on a log axis it draws as a
# fringe that reads as signal. Chart emitters use it as the gap floor, the
# ChartDoc renderer as the log color-scale floor, and the plots/
# compositors converge onto it chart by chart as their conversions land
# (dev/chart-inventory.md, judgment call J5, settled 2026-08-05; matches
# the app's LOG_FLOOR in theme.js). Lives here rather than in charts/ so
# the compositors, which are not chart-IR code, share the one definition.
LOG_FLOOR = 1e-15
# The dual distortion's one name, shared by the chart emitter and the
# compositor so g-check is called the same thing everywhere it is drawn.
# The plain form is what the chart IR carries (ECharts has no TeX, and
# U+01E7 is in DejaVu Sans); the TeX form is what a renderer that can
# typeset uses, and drawing the same typeset string on both sides is what
# lets the conversion and its compositor pin to one baseline image.
DISTORTION_DUAL_LABEL = 'ǧ(s)'
DISTORTION_DUAL_TEX = r'$\check g(s)$'

# Column / view labels for reinsurance reporting (``validation_df``,
# ``reins_summary_df``, ``reins_stats_df``). Centralised so the wording is
# changed in one place. These are structural keys, not user settings.
#   GROSS   -- top of step 1, before any cover (the first validation_df column).
#   SUBJECT -- what is subject to the aggregate cover (= the occurrence output).
#   NET     -- model output when every cover passes the net.
#   CEDED   -- model output when every cover passes the ceded.
#   OUTPUT  -- model output when occ and agg pass different kinds (mixed).
REINS_LABEL_GROSS = 'Gross'
REINS_LABEL_SUBJECT = 'Subject'
REINS_LABEL_NET = 'Net'
REINS_LABEL_CEDED = 'Ceded'
REINS_LABEL_OUTPUT = 'Output'

# --- shared ``info`` string convention --------------------------------------
# Every ``info`` row across Aggregate / Portfolio / Distortion is a label
# left-padded to one shared column width, no colon, value follows. The label
# width matches ``tail.describe_rows``. ``INFO_NA`` is the fixed placeholder
# for a value that is not (yet) available -- rows are never conditionally
# dropped, so two objects of one class always emit the same lines in the same
# order. The contract is documented in ``dev/info-strings.rst``.
INFO_LABEL_WIDTH = 25
INFO_NA = 'n/a'


def info_row(label, value):
    """Format one ``info`` line: ``label`` padded to ``INFO_LABEL_WIDTH``, then ``value``.

    Parameters
    ----------
    label : str
        Row label (no trailing colon).
    value : object
        Row value; rendered with ``str``. Pass ``INFO_NA`` for unavailable.

    Returns
    -------
    str
    """
    return f'{label:<{INFO_LABEL_WIDTH}}{value}'


# --- the first-class-citizen (FCC) contract ---------------------------------
# What "first class" MEANS, as names rather than as a phrase. Two criteria put a
# class on the list, and both must hold:
#
#   1. it can be created in DecL (which is why ``Bounds``, ``Frequency`` and
#      ``GridDistribution`` are out: they are reached from an object, never
#      declared), and
#   2. it flows through to the ``aggregate_api`` (aLL) SPA, which calls exactly
#      the members below on whatever it is handed.
#
# ``Severity`` is DecL-creatable and near-first-class, but it is a look-through
# onto a frozen scipy rv rather than a compute result, so it is exempt from the
# DataFrame quartet; it is listed separately rather than silently omitted.
#
# Everything OUTSIDE ``FCC_REQUIRED`` is optional and callers reach it
# defensively with ``getattr``. The narrative ``*_description`` (short) /
# ``*_explanation`` (long) strings are the main such family: not required, but
# where one half is present the other must be too (the pairs rule, checked by
# ``dev/regen_features.py`` and ``tests/test_fcc_surface.py``).
#
# Names only, no class imports: this module is the import-graph leaf, and both
# the dev audit script and the test suite read the contract from here.
FIRST_CLASS_CLASSES = ('Aggregate', 'Portfolio', 'BivariateAggregate',
                       'PnL', 'Distortion')

#: DecL-creatable, exempt from the DataFrame quartet. See the note above.
NEAR_FIRST_CLASS = ('Severity',)

#: Every member a first-class class must carry. Grouped: the discovery front
#: door and the fixed-layout text card; the three DecL trailer values; the
#: declaration round-trip; the DataFrame quartet; the plot.
#:
#: A fourth trailer value, ``doc``, was required here until 1.0.0a301, when
#: the ``doc{{{...}}}`` clause that filled it was retired
#: (``dev/done/plan-decommission-docs.md``). Nothing can set it any more, so
#: requiring it would have meant four permanently empty public attributes.
FCC_REQUIRED = ('info', 'help',
                'note', 'hints', 'tags',
                'program', 'pprogram',
                'summary_df', 'validation_df', 'stats_df', 'density_df',
                'plot')

#: Contract members a class does not carry YET, by class. Each entry would be a
#: known hole with a plan behind it, not a permanent carve-out: the audit and the
#: test subtract these so the contract can be stated before it is satisfied.
#:
#: **EMPTY since 1.0.0a172** ([FCC-Contract-Gaps]), which closed all three:
#: ``Portfolio.hints``, ``BivariateAggregate.validation_df`` and
#: ``Distortion.validation_df``. It must stay empty at 1.0.0b1. Adding an entry
#: is how a deliberate, temporary hole is declared; it is not a way to silence
#: the check.
FCC_CONTRACT_EXCEPTIONS = {}

#: Narrative stems allowed to carry only ONE half of the description /
#: explanation pair.
#:
#: **EMPTY since 1.0.0a172**, which wrote the two missing halves:
#: ``validation_description`` (the short verdict, which is what the old
#: ``validation_explanation`` actually was) and ``reins_explanation`` (the terms
#: plus what the cession does to expected loss).
FCC_UNPAIRED_NARRATIVES = ()


class Validation(Flag):
    """Flag set of validation failures surfaced by ``Aggregate.validation_explanation``.

    ``NOT_UNREASONABLE`` is the empty (passing) state; the remaining members
    are individual failure modes that combine via bitwise OR. ``SEV_*`` and
    ``AGG_*`` flag moment-matching errors (analytic vs empirical mean, CV,
    skew) above the validation ``eps`` tolerance. ``ALIASING`` flags FFT
    wrap-around; ``DEFECTIVE`` flags a realized law that does not sum to 1;
    ``REINSURANCE`` flags reinsurance-induced moment drift; ``NOT_UPDATED``
    signals the object hasn't been ``update``-d yet. ``INFEASIBLE`` says the
    severity cannot be reproduced on this grid at **any** bucket size, which
    is a reading about the grid rather than a failure, and so is transparent
    to :attr:`passes`; it explains the moment flags underneath it rather than
    adding to them.
    """

    NOT_UNREASONABLE = 0
    SEV_MEAN = auto()
    SEV_CV = auto()
    SEV_SKEW = auto()
    AGG_MEAN = auto()
    AGG_CV = auto()
    AGG_SKEW = auto()
    ALIASING = auto()
    REINSURANCE = auto()
    NOT_UPDATED = auto()
    DEFECTIVE = auto()
    INFEASIBLE = auto()

    @property
    def passes(self):
        """Whether this result clears validation (clean *or* cleanly reinsured).

        ``True`` for a clean ``NOT_UNREASONABLE`` object and for one whose only
        concern is ``REINSURANCE`` (the subject validated under the hood; the
        cession makes the realised moment audit n/a). Display surfaces (``qd``,
        the HTML reprs) read it to flag only a genuine failure.

        ``INFEASIBLE`` is **transparent** here: it is a reading about the grid
        rather than a verdict on the outcome, so it neither fails an object
        whose moments hold nor rescues one whose moments do not. It is masked
        out of the first term and deliberately does **not** join the
        ``REINSURANCE`` arm, which would let the flag hide the very problem it
        names. See ``dev/done/plan-validation-punchup.md``.
        """
        return ((self & ~Validation.INFEASIBLE) == Validation.NOT_UNREASONABLE
                or bool(self & Validation.REINSURANCE))


class DefectiveDistributionWarning(UserWarning):
    """Emitted when a realized PMF carries an economically material deficit.

    The FFT loses mass off the right end of the grid when ``log2`` is too
    small for the support. Forwards ``S = 1 - cumsum`` plateaus at the deficit
    (carrying it as a tail blob) while backwards ``S`` reaches zero (dropping
    it silently), so the two pricing answers differ by exactly the deficit.

    Fires above ``config`` ``validation.deficit_materiality`` (1e-4), the same
    floor :func:`aggregate.spectral.choquet_weights` uses to separate an
    economic problem from FFT truncation. Below it the deficit is real but
    immaterial, and reporting it says nothing the user can act on; the
    ``validation.noise`` floor (1e-12) is for arithmetic dust and is far too
    tight to interrupt anyone over.

    Reported through :func:`warn_once`, on two independent keys:

    * ``'defective-construction'`` -- the grid is losing mass. Once per
      session, because a sweep that builds the same shape 64 times has one
      fault, not 64. Every object carries its own verdict silently in
      ``valid`` / ``validation_explanation`` (:attr:`Validation.DEFECTIVE`),
      which is where to look for the rest.
    * ``'defective-pricing'`` -- a materially defective law was actually
      priced through a distortion. This is the moment the deficit stops being
      a grid property and starts changing an answer, so it is worth its own
      line even when construction already warned.

    Silence either with
    ``silence_warnings(DefectiveDistributionWarning)``; re-arm both with
    :func:`reset_warn_once`.

    Subclasses ``UserWarning`` so Python's default warning filter shows it
    (not the logger, which is silent by default).
    """


class InfeasibleGridWarning(UserWarning):
    """Emitted when a severity cannot be reproduced on the grid at any bucket size.

    The grid trades resolution against reach: at a fixed ``log2`` budget it
    spans ``2**log2 * bs``, so a finer bucket buys detail at the bottom by
    giving up extent at the top. A thick enough severity furnishes its mean so
    far above its median that the two demands cannot both be met, and no
    bucket size splits the difference. That is a statement about the model
    rather than about the numerics, and the fix is an occurrence limit on the
    severity, not a finer grid.

    The message names the three numbers that make the situation legible, the
    bucket the body needs, the share of the **mean** the first half bucket
    takes (which under the ``round`` scheme is placed at exactly 0), and the
    reach the mean is not complete without, then the one action that resolves
    it. Bucket selection is one of the largest hurdles an FFT method puts in
    front of a user, so the message is deliberately long: this is the moment
    to be educational.

    Reported through :func:`warn_once`, keyed on the severity and the ``log2``
    it was measured against, so a sweep that rebuilds the same shape reports
    one fault rather than one per build, and a genuinely different grid still
    speaks up. Every object carries the same verdict silently in ``valid``
    (:attr:`Validation.INFEASIBLE`) and in ``bs_description`` /
    ``bs_explanation``, which is where to look for the rest.

    Silence with ``silence_warnings(InfeasibleGridWarning)``; re-arm with
    :func:`reset_warn_once`. See ``dev/done/plan-validation-punchup.md``.

    Subclasses ``UserWarning`` so Python's default warning filter shows it
    (not the logger, which is silent by default).
    """


class IgnoredDecLClauseWarning(UserWarning):
    """Emitted when a DecL declaration ignores clauses it cannot use.

    A pure ``agg`` accepts every reinsurance decoration -- ceded-premium
    clauses (``deposit`` / ``rol`` / ``rate``), ceding commissions
    (``cede``), ``reinstatements``, and the variable-rating features -- but
    it has no premium context to activate them, so it builds the loss
    structure only and says so with one warning naming the ignored clauses.
    The recipe base retains the full decorated spec, so folding the agg
    into a ``pnl`` / ``xpnl`` by reference (``pnl X <premium> less agg.NAME``)
    activates the economics (the spec-injection route). See
    ``dev/plan-pnl-consolidated-xpnl-walk.md``
    ([Reins-Economics-On-Agg-Ignore-Warn]).

    Subclasses ``UserWarning`` so Python's default warning filter shows it
    (not the logger, which is silent by default).
    """


class ZeroModifiedExposureWarning(UserWarning):
    """Emitted when ``zm`` / ``zt`` shifts the mean off a monetary exposure target.

    Zero truncation and zero modification are the (a, b, 1) construction of
    Klugman, Panjer and Willmot (2012) §6.6: the base distribution is held
    fixed and reweighted, so the mean *moves* -- that is what the modification
    is for. The ``!`` marker asks DecL for exactly that textbook reading (and
    R's ``actuar``'s), taking the exposure clause as the **un-modified base
    mean**.

    A *monetary* clause (``1000 loss``, ``1000 premium at 0.65 lr``,
    ``100 exposure at 0.05 rate``) is different: it states a money target that
    the shifted mean will miss. This warning names the requested amount, what
    was delivered, and the base -> realized count move. Drop the ``!`` to pin
    the target instead, which solves for the base mean that hits it.

    Since ``1.0.0a325`` pinning is the default, so this fires only on the
    ``!`` path: a reader who did not ask for the base parameterization never
    sees it ([ZT-ZM-Recalibrate-Default]).

    Subclasses ``UserWarning`` so Python's default warning filter shows it
    (not the logger, which is silent by default).
    """


class ZeroPremiumCessionWarning(UserWarning):
    """Emitted when a ``pnl`` / ``xpnl`` cession has no ceded-premium clause.

    A cession side (occurrence / aggregate) with reinsurance layers but no
    ``deposit`` / ``rol`` / ``rate`` clause books at **zero ceded premium**:
    the recovery is real, the premium is zero, and the program still routes
    through the full guaranteed-cost machinery (consolidated ``pnl`` /
    ``xpnl`` walk) rather than degrading to the plain face
    ([XPnL-Zero-Premium-Cessions],
    ``dev/done/plan-pnl-faces-punchlist.md``). The warning names the
    side(s); silence it by pricing the cover. Feature-decorated sides are
    exempt (the feature owns the premium slot), as is a reinstated
    occurrence layer (which *requires* a base premium clause).

    Subclasses ``UserWarning`` so Python's default warning filter shows it
    (not the logger, which is silent by default).
    """


class CoarseJointGridWarning(UserWarning):
    """Emitted when a reinstatement joint's grid under-resolves a treaty kink.

    The 2-D ``(L, R)`` joint runs on a common bucket size far coarser than
    the 1-D engine grid (a cell budget, not an accuracy target). A kinked
    treaty map -- a layer, a collar -- evaluated across too few buckets
    picks up a Jensen-type O(bs) bias that the internal audits cannot see
    (they check pushforward-vs-exact on the *same* grid). Warn when a kink
    region spans fewer than
    :data:`~aggregate._pnl_builders.JOINT_KINK_MIN_BUCKETS` buckets
    ([Reinst-Joint-Grid-Adequacy],
    ``dev/done/plan-pnl-faces-punchlist.md``); the remedy is the
    joint grid knobs (``bs`` / ``log2_x`` / ``log2_y``, forwarded to
    :meth:`~aggregate.Aggregate.occ_bivariate`).

    Subclasses ``UserWarning`` so Python's default warning filter shows it
    (not the logger, which is silent by default).
    """


class DegenerateEvaluationWarning(UserWarning):
    """Emitted when a position has no breakeven acceptability level.

    ``evaluate`` solves for the distortion at which the risk-adjusted margin
    reaches 0. That level exists only when the margin is favorable on average
    and unfavorable somewhere, so the distorted mean can cross 0. Two positions
    have nothing to solve:

    * ``E[M] <= 0``: unacceptable under the identity distortion already, so
      unacceptable at every stress. Cherny and Madan set the index to 0.
    * ``M >= 0`` a.s.: an arbitrage, acceptable at every stress, so the index
      is unbounded.

    Both report ``NaN`` rather than 0 or infinity. The limiting parameter
    differs by family and carries no information about the position, so a
    number there would invite comparisons that mean nothing; the panel's
    ``status`` column names which case fired.

    Subclasses ``UserWarning`` so Python's default warning filter shows it
    (not the logger, which is silent by default).
    """


class ReflectedSeverityClampWarning(UserWarning):
    """Emitted when a reflected ``sev`` severity clamps mass at zero.

    A reflected severity (``shift - X``, or the bare ``-X``) declared with plain
    ``sev`` is an ordinary loss: the layered-loss transform clamps ``x < 0`` to
    ``0``, exactly as it does for ``sev 10 * norm + 5`` or
    ``sev lognorm 5 cv 1 - 10``. That is usually what the user wants when the
    reflection is bounded (``sev 10 - lognorm 1.5 splice [0 10]`` lives on
    ``[0, 10]`` and never clamps), and usually a mistake when it is not, because
    the negative half of a premium-minus-loss position is the interesting half.

    The warning fires only when both conditions hold: the severity is reflected,
    and its reflected support actually reaches below zero. It names the mass
    being clamped, so a 0.01% tail and a wholesale truncation read differently,
    and points at ``ssev``, which keeps the negative support instead. In the
    limit the whole law clamps (``sev -lognorm 1.5`` has support
    ``(-inf, 0]``) and the severity is a point mass at zero; that follows from
    the rule rather than being an error. See
    ``dev/done/plan-reflected-loss-severity.md`` ([Reflected-Clamp-Warning]).

    Reported through :func:`warn_once`, keyed on the severity name, shift, and
    splice window, so ten like units warn once but two genuinely different
    declarations both warn. Silence it with
    ``silence_warnings(ReflectedSeverityClampWarning)``; re-arm with
    :func:`reset_warn_once`.

    Subclasses ``UserWarning`` so Python's default warning filter shows it
    (not the logger, which is silent by default).
    """


class DefectiveDistributionError(ValueError):
    """Raised when a Choquet computation receives a materially defective law.

    A probability vector whose total falls short of 1 by more than the
    validation noise floor carries *unrepresented mass at unknown loss
    values* -- an economic problem, not numerical dust. The exact-discrete
    Choquet helper (:func:`aggregate.spectral.choquet_weights`) refuses to
    price such a law unless the caller explicitly opts into a parking
    policy (``allow_deficit=True``: forwards parks the deficit at the top
    atom, backwards at the bottom atom). See
    ``dev/../math/docs/choquet-calc-method.md``.
    """


class InfiniteVarianceError(ValueError):
    """Raised when ``bs`` must be estimated for an infinite-variance aggregate.

    Sizing the FFT grid (``bs``) relies on a method-of-moments tail estimate,
    which needs a finite variance. A power-law / heavy-tailed severity with no
    finite second moment (e.g. ``pareto`` shape ``alpha <= 2``) gives no basis
    to place the grid, so an aggregate that uses one **must** be built with an
    explicit ``bs`` -- there is no sensible default to guess.

    Subclasses :class:`ValueError` so existing broad ``except ValueError``
    handlers continue to treat it as a build failure.
    """


# ===========================================================================
# Once-per-session warnings
# ===========================================================================
#
# Python's default filter shows a warning once per (text, category, lineno),
# so a message that embeds a computed number defeats it: every distinct value
# is a distinct message and prints again. Worse, Jupyter clears
# ``__warningregistry__`` between cells, so even an identical message repeats
# once per cell. A sweep that builds the same object 64 times therefore
# printed 64 copies of two messages into the rendered documentation.
#
# The library's advisory warnings are about a *condition*, not an occurrence:
# knowing once that a grid is losing mass is what changes what the user does.
# ``warn_once`` keys on the condition rather than the text, so the first
# occurrence carries its numbers and the rest stay silent.

#: Keys already warned in this session. Never read directly; see ``warn_once``.
_WARNED_ONCE = set()

#: Depth of nested ``warn_once_isolated`` scopes. Non-zero means "probe":
#: bypass the registry entirely.
_WARN_ONCE_ISOLATION = 0

#: Appended to the one emission, so the reader knows the silence downstream is
#: policy rather than absence. One sentence on the message line, no block.
_WARN_ONCE_SUFFIX = (' Further occurrences this session are suppressed; each '
                     'object records its own verdict in validation.')


def warn_once(message, category, *, key=None, stacklevel=3, suffix=True):
    """Emit ``message`` the first time this session, then stay silent.

    Parameters
    ----------
    message : str
        The warning text. Free to carry computed numbers: the dedup key is
        ``key``, not the text.
    category : type[Warning]
        Warning class, as for :func:`warnings.warn`.
    key : hashable, optional
        Dedup key. Defaults to ``category``, giving once-per-session per
        warning class. Pass an explicit key to split one class into several
        independently-once channels, e.g. ``DefectiveDistributionWarning``
        fires once at construction and once again at pricing time, which are
        different facts about different moments in the user's workflow.
    stacklevel : int, default 3
        Passed through to :func:`warnings.warn`. The default assumes one
        wrapper frame between the reporting code and the user's call.
    suffix : bool, default True
        Append :data:`_WARN_ONCE_SUFFIX` to the emitted message.

    Returns
    -------
    bool
        Whether the warning was emitted.

    Notes
    -----
    Inside :func:`warn_once_isolated` the registry is bypassed and every call
    warns, which is what a probe that *counts* warnings needs.

    The state is process-wide and deliberately not per-object: the same grid
    fault reported by 64 sibling objects is one fact, not 64. Call
    :func:`reset_warn_once` to re-arm.
    """
    if _WARN_ONCE_ISOLATION:
        warnings.warn(message, category, stacklevel=stacklevel)
        return True
    k = category if key is None else key
    if k in _WARNED_ONCE:
        return False
    _WARNED_ONCE.add(k)
    warnings.warn(message + (_WARN_ONCE_SUFFIX if suffix else ''),
                  category, stacklevel=stacklevel)
    return True


def reset_warn_once():
    """Re-arm every once-per-session warning.

    Clears the :func:`warn_once` registry, so the next occurrence of each
    condition reports again. Intended for a long-running session that has
    moved on to a different book, and for test isolation (the aggregate test
    suite resets it before every test, otherwise the first test to trigger a
    condition would silence ``pytest.warns`` in all the others).
    """
    _WARNED_ONCE.clear()


@contextmanager
def warn_once_isolated():
    """Run a probe without touching the once-per-session budget.

    Inside the block :func:`warn_once` warns on **every** call, and on exit
    the registry is restored to what it was on entry. Both halves matter, and
    the library needs each:

    * A speculative pre-pass whose warnings are filtered away must not *spend*
      the session's one emission before the real computation runs.
    * A grid probe that evaluates many candidate cells and counts the warnings
      each one raises needs every cell to warn, not just the first.

    This is library machinery for code that deliberately provokes warnings.
    An end user who simply wants quiet wants
    :func:`aggregate.utilities.silence_warnings`.
    """
    global _WARN_ONCE_ISOLATION
    saved = set(_WARNED_ONCE)
    _WARN_ONCE_ISOLATION += 1
    try:
        yield
    finally:
        _WARN_ONCE_ISOLATION -= 1
        _WARNED_ONCE.clear()
        _WARNED_ONCE.update(saved)
