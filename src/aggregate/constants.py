"""Module-level constants for ``aggregate``: figure-size defaults shared by
plotting consumers, logging level ``WL``, validation tolerance and
recommendation probability, the ``Validation`` flag enum used by
:func:`Aggregate.explain_validation`, and user-data / package-data paths."""

from enum import Flag, auto


__all__ = ['FIG_W', 'FIG_H', 'WL', 'FONT_SIZE', 'LEGEND_FONT',
           'PLOT_FACE_COLOR', 'FIGURE_BG_COLOR', 'VALIDATION_EPS',
           'VALIDATION_NOISE', 'RECOMMEND_P', 'Validation',
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

