# coding: utf-8 -*-

from .parser import UnderwritingLexer, UnderwritingParser, grammar
from .moments import (MomentAggregator, MomentWrangler,
                      xsden_to_meancv, xsden_to_meancvskew)
from .iman_conover import (iman_conover, block_iman_conover,
                           rearrangement_algorithm_max_VaR,
                           make_corr_matrix, random_corr_matrix)
from .utilities import (decl_pprint,
                        ft, ift,
                        subsets,
                        round_bucket,
                        make_ceder_netter, nice_multiple,
                        tweedie_convert, tweedie_density,
                        qd,
                        make_var_tvar, kaplan_meier, kaplan_meier_np,
                        agg_help, explain_validation)
from .spectral import Distortion, approx_ccoc, tvar_weights, consistent_distortions, p_to_parameters
from .distributions import (Frequency, Severity, Aggregate,
                            lognorm_fit, sln_fit, sgamma_fit, gamma_fit, beta_fit,
                            invgamma_fit, invgauss_fit,
                            lognorm_lev, lognorm_approx,
                            approximate_from_mcvsk)
from .portfolio import Portfolio, make_awkward, make_comonotonic_allocations
from .underwriter import Underwriter, build, build_many, CannotBuild
from .bounds import Bounds, plot_max_min, plot_lee
from .constants import *
from .random_agg import *
from .decl_pygments import *

import sys


# knobble warnings
# https://docs.python.org/3/library/warnings.html#temporarily-suppressing-warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


__docformat__ = 'restructuredtext'
__project__ = 'aggregate'
__author__ = "Stephen J. Mildenhall"
__copyright__ = "2018-2026, Stephen J Mildenhall"
__license__ = "BSD 3-Clause New License"
__email__ = "stephen.j.mildenhall@gmail.com"
__status__ = "beta"
from importlib.metadata import version as _pkg_version
__version__ = _pkg_version("aggregate")


# module level doc-string
__doc__ = """
:mod:`aggregate` solves insurance, risk management, and actuarial problems using realistic models that
reflect underlying frequency and severity. It makes working with an aggregate (compound) probability
distribution as easy as the lognormal, delivering the speed and accuracy of parametric distributions
to situations that usually require simulation. :mod:`aggregate` includes an expressive language called
DecL to describe aggregate distributions and is implemented in Python under an open source BSD-license.
"""
