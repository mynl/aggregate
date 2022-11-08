# coding: utf-8 -*-

__docformat__ = 'restructuredtext'
__project__ = 'aggregate'
__author__ = "Stephen J. Mildenhall"
__copyright__ = "Copyright 2018-2022, Convex Risk LLC"
__license__ = "BSD 3-Clause New License"
__email__ = "steve@convexrisk.com"
__status__ = "alpha"
# only need to change here, feeds conf.py (docs) and setup.py (build)
__version__ = "0.9.5.4"

# set up
from pathlib import Path
base_dir = Path.home() / 'aggregate'
base_dir.mkdir(exist_ok=True)

for p in ['cases', 'parser', 'temp', 'generated']:
    (base_dir / p).mkdir(exist_ok=True)

# print('All directories exist')

del p, base_dir


# imports
from .underwriter import Underwriter, build, debug_build
from .portfolio import Portfolio, make_awkward
from .distributions import Frequency, Severity, Aggregate
from .spectral import Distortion
from .utilities import get_fmts, pprint, pprint_ex, ft, \
    ift, sln_fit, sgamma_fit, estimate_agg_percentile, \
    axiter_factory, AxisManager, lognorm_lev, html_title, \
    sensible_jump, suptitle_and_tight, \
    MomentAggregator, MomentWrangler, xsden_to_meancv, \
    frequency_examples, Answer, log_test, subsets, \
    round_bucket, \
    make_ceder_netter, make_mosaic_figure, nice_multiple, \
    style_df, logger_level, friendly, \
    FigureManager, tweedie_convert, power_variance_family, \
    iman_conover, rearrangement_algorithm_max_VaR, \
    mu_sigma_from_mean_cv, \
    make_corr_matrix, random_corr_matrix, \
    LoggerManager, knobble_fonts, approximate_work, \
    partial_e, partial_e_numeric, moms_analytic
from .parser import UnderwritingLexer, UnderwritingParser, grammar
from .bounds import Bounds, plot_max_min, plot_lee

# as a default turn off all logging
logger_level(30)
knobble_fonts()

# module level doc-string
__doc__ = """
**aggregate** is a Python package providing an expressive language and fast, accurate computations
to make working with aggregate (compound) probability distributions
easy and intuitive. It allows students and practitioners to work with realistic 
real-world distributions that reflect the underlying frequency and severity 
generating processes. It has applications in insurance, risk management, actuarial 
science, and related areas.
"""


