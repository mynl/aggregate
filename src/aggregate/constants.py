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

All tunable defaults (grid sizing, databases, discretization schemes,
validation tolerances — including the former numerics noise floors) and the
path names now live in :mod:`aggregate.config`. The plotting figure constants
below stay here permanently (used in default-argument expressions; user
restyling goes through matplotlib's native ``mplstyle`` / ``rcParams``).
"""

from enum import Flag, auto


__all__ = ['FIG_W', 'FIG_H', 'FONT_SIZE', 'LEGEND_FONT',
           'PLOT_FACE_COLOR', 'FIGURE_BG_COLOR',
           'Validation', 'DefectiveDistributionWarning',
           'DefectiveDistributionError', 'InfiniteVarianceError',
           'IgnoredDecLClauseWarning',
           'REINS_LABEL_GROSS', 'REINS_LABEL_SUBJECT', 'REINS_LABEL_NET',
           'REINS_LABEL_CEDED', 'REINS_LABEL_OUTPUT',
           'INFO_LABEL_WIDTH', 'INFO_NA', 'info_row']

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


class Validation(Flag):
    """Flag set of validation failures surfaced by ``Aggregate.validation_explanation``.

    ``NOT_UNREASONABLE`` is the empty (passing) state; the remaining members
    are individual failure modes that combine via bitwise OR. ``SEV_*`` and
    ``AGG_*`` flag moment-matching errors (analytic vs empirical mean, CV,
    skew) above the validation ``eps`` tolerance. ``ALIASING`` flags FFT
    wrap-around; ``REINSURANCE`` flags reinsurance-induced moment drift;
    ``NOT_UPDATED`` signals the object hasn't been ``update``-d yet.
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


class DefectiveDistributionWarning(UserWarning):
    """Emitted when an aggregate empirical PMF carries a genuine deficit.

    The aggregate FFT loses mass off the right end of the grid when ``log2``
    is too small for the support. A deficit `1 - Σp_agg` above the validation
    noise floor (``config`` ``validation.noise``) is real, not numerical dust:
    forwards `S = 1 - cumsum` plateaus at the deficit (carries it as a tail
    blob) while backwards `S` reaches zero (drops the deficit silently). The
    two pricing answers therefore differ by exactly the deficit. Surface the
    deficit at construction time so the divergence in `Distortion.price` is
    never silent.

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
    The knowledge base retains the full decorated spec, so folding the agg
    into a ``pnl`` / ``xpnl`` by reference (``pnl X <premium> less agg.NAME``)
    activates the economics (the knowledge-injection route). See
    ``dev/plan-pnl-consolidated-xpnl-walk.md``
    ([Reins-Economics-On-Agg-Ignore-Warn]).

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
