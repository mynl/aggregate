# coding: utf-8 -*-

from . parser import UnderwritingLexer, UnderwritingParser, grammar
from . utilities import get_fmts, pprint, pprint_ex, ft, \
    ift, sln_fit, sgamma_fit, estimate_agg_percentile, \
    axiter_factory, AxisManager, lognorm_lev, html_title, \
    sensible_jump, suptitle_and_tight, \
    MomentAggregator, MomentWrangler, xsden_to_meancv, xsden_to_meancvskew, \
    frequency_examples, Answer, log_test, subsets, \
    round_bucket, \
    make_ceder_netter, make_mosaic_figure, nice_multiple, \
    style_df, logger_level, friendly, \
    FigureManager, tweedie_convert, tweedie_density,\
    iman_conover, rearrangement_algorithm_max_VaR, \
    mu_sigma_from_mean_cv, \
    make_corr_matrix, random_corr_matrix, \
    LoggerManager, knobble_fonts, approximate_work, \
    partial_e, partial_e_numeric, moms_analytic, qd, \
    sEngFormatter, mv, picks_work, GCN, lognorm_approx, \
    integral_by_doubling
from . spectral import Distortion
from . distributions import Frequency, Severity, Aggregate
from . portfolio import Portfolio, make_awkward
from . underwriter import Underwriter, build, debug_build
from . bounds import Bounds, plot_max_min, plot_lee
from . constants import *

import sys

# knobble warnings
# https://docs.python.org/3/library/warnings.html#temporarily-suppressing-warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


__docformat__ = 'restructuredtext'
__project__ = 'aggregate'
__author__ = "Stephen J. Mildenhall"
__copyright__ = "2018-2022, Convex Risk LLC"
__license__ = "BSD 3-Clause New License"
__email__ = "steve@convexrisk.com"
__status__ = "alpha"
# only need to change here, feeds conf.py (docs) and setup.py (build)
__version__ = "0.11.6"

# set up
from pathlib import Path
base_dir = Path.home() / 'aggregate'
base_dir.mkdir(exist_ok=True)

for p in ['cases', 'parser', 'temp', 'generated']:
    (base_dir / p).mkdir(exist_ok=True)

# print('All directories exist')

del p, base_dir


# imports

# as a default turn off all logging
logger_level(30)
knobble_fonts()

# module level doc-string
__doc__ = """
:mod:`aggregate` solves insurance, risk management, and actuarial problems using realistic models that reflect underlying frequency and severity. It makes working with an aggregate (compound) probability distribution as easy as the lognormal, delivering the speed and accuracy of parametric distributions to situations that usually require simulation. :mod:`aggregate` includes an expressive language called DecL to describe aggregate distributions and is implemented in Python under an open source BSD-license.
"""
