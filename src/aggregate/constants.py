"""Module-level constants for ``aggregate``: figure-size defaults shared by
plotting consumers, logging level ``WL``, validation tolerance and
recommendation probability, the ``Validation`` flag enum used by
:func:`Aggregate.explain_validation`, and user-data / package-data paths."""

from enum import Flag, auto


__all__ = ['FIG_W', 'FIG_H', 'WL', 'FONT_SIZE', 'LEGEND_FONT',
           'PLOT_FACE_COLOR', 'FIGURE_BG_COLOR', 'VALIDATION_EPS',
           'VALIDATION_NOISE', 'ALIASING_RATIO', 'EXEQA_NOISE_FLOOR',
           'FT_NOISE_FLOOR', 'RECOMMEND_P', 'Validation',
           'DefectiveDistributionWarning',
           'USER_DIR_NAME', 'PACKAGE_DATA_DIR', 'TEST_SUITE_FILENAME']

FIG_W = 3.5
FIG_H = 2.45

# level used for logging that replaces warning in cases where a warning is not appropriate
# but was used for debugging purposes. (WL = Warning Level)
WL = 25


FONT_SIZE = 9
LEGEND_FONT = 'x-small'
# see https://matplotlib.org/stable/gallery/color/named_colors.html
PLOT_FACE_COLOR = 'lightsteelblue'
FIGURE_BG_COLOR = 'aliceblue'
VALIDATION_EPS = 1e-4
# Absolute floor below which a quantity is treated as exact zero / pure
# numerical noise. aggregate's FFT arithmetic is essentially exact, so
# genuine dust lives around 1e-14--1e-15 (e.g. the skewness of a symmetric
# distribution); 1e-12 clears it with headroom while sitting ~3 orders above
# numpy pairwise-summation roundoff and 8 orders below VALIDATION_EPS. Used
# for defective-distribution detection and for the near-zero (absolute vs
# relative) error fallback in validation and the describe/stats_df displays.
VALIDATION_NOISE = 1e-12
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
RECOMMEND_P = 0.99999

# User-local data directory (under Path.home())
USER_DIR_NAME = '.aggregate'
# Subdirectory inside the installed `aggregate` package holding bundled .agg files
PACKAGE_DATA_DIR = 'agg'
# The canonical bundled test suite filename (lives in PACKAGE_DATA_DIR)
TEST_SUITE_FILENAME = 'test_suite.agg'


class Validation(Flag):
    """Flag set of validation failures surfaced by ``Aggregate.explain_validation``.

    ``NOT_UNREASONABLE`` is the empty (passing) state; the remaining members
    are individual failure modes that combine via bitwise OR. ``SEV_*`` and
    ``AGG_*`` flag moment-matching errors (analytic vs empirical mean, CV,
    skew) above ``VALIDATION_EPS``. ``ALIASING`` flags FFT wrap-around;
    ``REINSURANCE`` flags reinsurance-induced moment drift; ``NOT_UPDATED``
    signals the object hasn't been ``update``-d yet.
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
    is too small for the support. A deficit `1 - Σp_agg > VALIDATION_NOISE`
    is real, not numerical dust: forwards `S = 1 - cumsum` plateaus at the
    deficit (carries it as a tail blob) while backwards `S` reaches zero
    (drops the deficit silently). The two pricing answers therefore differ
    by exactly the deficit. Surface the deficit at construction time so the
    divergence in `Distortion.price` is never silent.

    Subclasses ``UserWarning`` so Python's default warning filter shows it
    (not the logger, which is silent by default).
    """

