"""Dependency-free leaf holding the few non-settings constants and types that
low-level ``aggregate`` modules need.

This module is deliberately tiny and import-free so it can sit *below*
``distributions`` / ``portfolio`` in the import graph without risking a cycle:
``utilities`` imports :class:`Validation` and ``spectral`` uses
:class:`DefectiveDistributionWarning`, and both are imported by
``distributions`` / ``portfolio``. A ``Flag`` enum and a ``Warning`` subclass
are types, not user settings, so they do not belong in :mod:`aggregate.config`.

The reinsurance labels are *structural* MultiIndex column / axis keys
(referenced by literal in ``multivariate`` and asserted across the reins
tests), so they are constants here, not user-tunable settings.

All tunable defaults (grid sizing, databases, discretization schemes,
validation tolerances) and the path names now live in :mod:`aggregate.config`.
The plotting figure constants and the numerics-pending noise floors below are
slated to move to :mod:`aggregate.config` in a later phase.
"""

from enum import Flag, auto


__all__ = ['FIG_W', 'FIG_H', 'FONT_SIZE', 'LEGEND_FONT',
           'PLOT_FACE_COLOR', 'FIGURE_BG_COLOR',
           'ALIASING_RATIO', 'EXEQA_NOISE_FLOOR', 'FT_NOISE_FLOOR',
           'Validation', 'DefectiveDistributionWarning',
           'REINS_LABEL_GROSS', 'REINS_LABEL_SUBJECT', 'REINS_LABEL_NET',
           'REINS_LABEL_CEDED', 'REINS_LABEL_OUTPUT']

# --- plotting figure defaults (move to config [plotting] in Phase 2) -------
# These are used as module-level constants in default argument expressions
# (e.g. ``figsize=(2 * FIG_W, FIG_H)``), so they stay literals until the
# plotting/style work repoints those call sites.
FIG_W = 3.5
FIG_H = 2.45
FONT_SIZE = 9
LEGEND_FONT = 'x-small'
# see https://matplotlib.org/stable/gallery/color/named_colors.html
PLOT_FACE_COLOR = 'lightsteelblue'
FIGURE_BG_COLOR = 'aliceblue'

# --- numerics-pending validation floors (values await the numerics review) --
# Aliasing test ratio. The ALIASING flag fires when the relative error on the
# aggregate mean exceeds ALIASING_RATIO times the relative error on the
# severity mean: aliasing inflates the agg-mean error far above the sev-mean
# error, while a clean discretisation keeps them comparable. 10 has carried
# through the suite as the practical threshold.
ALIASING_RATIO = 10
# Floor on the per-bucket ``exeqa_err`` (``Σ exeqa_i − loss``) below which a
# bucket's conditional decomposition is treated as numerically resolved, used
# in ``Portfolio._build_augmented`` to truncate the augmented frame where
# exeqa-derived quantities become unreliable.
EXEQA_NOISE_FLOOR = 1e-4
# Floor on ``|ft_line_density|`` below which the "build up the product"
# branch is preferred over division in the per-line FT decomposition (avoids
# divide-by-near-zero).
FT_NOISE_FLOOR = 1e-10

# Column / view labels for reinsurance reporting (``describe``,
# ``reins_describe``, ``reins_stats_df``). Centralised so the wording is
# changed in one place. These are structural keys, not user settings.
#   GROSS   -- top of step 1, before any cover (the first describe column).
#   SUBJECT -- what is subject to the aggregate cover (= the occurrence output).
#   NET     -- model output when every cover passes the net.
#   CEDED   -- model output when every cover passes the ceded.
#   OUTPUT  -- model output when occ and agg pass different kinds (mixed).
REINS_LABEL_GROSS = 'Gross'
REINS_LABEL_SUBJECT = 'Subject'
REINS_LABEL_NET = 'Net'
REINS_LABEL_CEDED = 'Ceded'
REINS_LABEL_OUTPUT = 'Output'


class Validation(Flag):
    """Flag set of validation failures surfaced by ``Aggregate.explain_validation``.

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
