# constants
from enum import Flag, auto


__all__ = ['FIG_W', 'FIG_H', 'WL', 'FONT_SIZE', 'LEGEND_FONT',
           'PLOT_FACE_COLOR', 'FIGURE_BG_COLOR', 'VALIDATION_EPS',
           'RECOMMEND_P', 'Validation']

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
RECOMMEND_P = 0.99999

class Validation(Flag):
    NOT_UNREASONABLE = auto()
    SEV_MEAN = auto()
    SEV_CV = auto()
    SEV_SKEW = auto()
    AGG_MEAN = auto()
    AGG_CV = auto()
    AGG_SKEW = auto()
    ALIASING = auto()
    REINSURANCE = auto()
    NOT_UPDATED = auto()

