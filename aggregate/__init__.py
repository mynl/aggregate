# coding: utf-8 -*-

from .parser import UnderwritingLexer, UnderwritingParser, grammar
from .utilities import (get_fmts, pprint, pprint_ex, ft,
                        ift, sln_fit, sgamma_fit, estimate_agg_percentile,
                        axiter_factory, AxisManager, lognorm_lev, html_title,
                        sensible_jump, suptitle_and_tight,
                        MomentAggregator, MomentWrangler, xsden_to_meancv, xsden_to_meancvskew,
                        frequency_examples, Answer, log_test, subsets,
                        round_bucket,
                        make_ceder_netter, make_mosaic_figure, nice_multiple,
                        style_df, logger_level, friendly,
                        FigureManager, tweedie_convert, tweedie_density,
                        iman_conover, rearrangement_algorithm_max_VaR,
                        mu_sigma_from_mean_cv,
                        make_corr_matrix, random_corr_matrix,
                        LoggerManager, knobble_fonts, approximate_work,
                        partial_e, partial_e_numeric, moms_analytic, qd,
                        sEngFormatter, mv, picks_work, GCN, lognorm_approx,
                        integral_by_doubling, logarithmic_theta, block_iman_conover,
                        make_var_tvar, test_var_tvar, kaplan_meier, kaplan_meier_np,
                        more, parse_note, parse_note_ex, introspect, explain_validation,
                        beta_fit)
from .spectral import Distortion, approx_ccoc, tvar_weights
from .distributions import Frequency, Severity, Aggregate
from .portfolio import Portfolio, make_awkward
from .underwriter import Underwriter, build
from .bounds import Bounds, plot_max_min, plot_lee
from .constants import *
from .random_agg import *
from .decl_pygments import *
from .sly import *
from .agg_magics import *

import sys


# knobble warnings
# https://docs.python.org/3/library/warnings.html#temporarily-suppressing-warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


__docformat__ = 'restructuredtext'
__project__ = 'aggregate'
__author__ = "Stephen J. Mildenhall"
__copyright__ = "2018-2024, Convex Risk LLC"
__license__ = "BSD 3-Clause New License"
__email__ = "steve@convexrisk.com"
__status__ = "beta"
# only need to change here, feeds conf.py (docs) and pyproject.toml (build)
__version__ = "0.26.0"


# as a default turn off all logging
logger_level(30)
knobble_fonts(True)

# module level doc-string
__doc__ = """
:mod:`aggregate` solves insurance, risk management, and actuarial problems using realistic models that
reflect underlying frequency and severity. It makes working with an aggregate (compound) probability
distribution as easy as the lognormal, delivering the speed and accuracy of parametric distributions
to situations that usually require simulation. :mod:`aggregate` includes an expressive language called
DecL to describe aggregate distributions and is implemented in Python under an open source BSD-license.
"""
