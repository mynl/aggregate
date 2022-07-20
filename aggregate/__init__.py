# coding: utf-8 -*-

__docformat__ = 'restructuredtext'
__author__ = "Stephen J. Mildenhall"
__copyright__ = "Copyright 2018-2022, Convex Risk LLC"
__license__ = "BSD 3-Clause New License"
__version__ = "0.9.4.1"
__email__ = "steve@convexrisk.com"
__status__ = "alpha"

# set up
from pathlib import Path
base_dir = Path.home() / 'aggregate'
base_dir.mkdir(exist_ok=True)

for p in ['cases', 'parser', 'tests', 'temp']:
    (base_dir / p).mkdir(exist_ok=True)

# print('All directories exist')

del p, base_dir


# imports
from .underwriter import Underwriter, build
from .port import Portfolio, make_awkward
from .distr import Frequency, Severity, Aggregate, CarefulInverse
from .spectral import Distortion
from .utils import get_fmts, tidy_agg_program, ft, \
    ift, sln_fit, sgamma_fit, estimate_agg_percentile, \
    axiter_factory, AxisManager, lognorm_lev, html_title, \
    sensible_jump, suptitle_and_tight, \
    MomentAggregator, MomentWrangler, xsden_to_meancv, \
    frequency_examples, Answer, log_test, subsets, \
    round_bucket, \
    make_ceder_netter, make_mosaic_figure, nice_multiple, \
    style_df, logger_level, friendly, \
    FigureManager, tweedie_convert, power_variance_family
from .parser import UnderwritingLexer, UnderwritingParser, grammar
from .bounds import Bounds, plot_max_min, plot_lee

# module level doc-string
__doc__ = """
**aggregate** is a Python package providing fast, accurate, and expressive data
structures that make working with aggregate (also called compound) probability distributions
easy and intuitive. It allows students and practitioners to work with realistic 
real-world distributions that reflect the underlying frequency and severity 
generating processes. It has applications in insurance, risk management, actuarial 
science, and related areas.

"""


