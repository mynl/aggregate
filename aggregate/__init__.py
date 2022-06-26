# coding: utf-8 -*-

__docformat__ = 'restructuredtext'
__author__ = "Stephen J. Mildenhall"
__copyright__ = "Copyright 2018-2022, Convex Risk LLC"
__license__ = "BSD 3-Clause New License"
__version__ = "0.9.3.1"
__email__ = "steve@convexrisk.com"
__status__ = "alpha"

# imports
from .underwriter import Underwriter, build
from .port import Portfolio
from .distr import Frequency, Severity, Aggregate, CarefulInverse
from .spectral import Distortion
from .utils import get_fmts, tidy_agg_program, ft, \
    ift, sln_fit, sgamma_fit, estimate_agg_percentile, \
    axiter_factory, AxisManager, lognorm_lev, html_title, \
    sensible_jump, suptitle_and_tight, \
    MomentAggregator, MomentWrangler, xsden_to_meancv, \
    frequency_examples, Answer, log_test, subsets, \
    start_timer, report_time, stop_timer, round_bucket, \
    make_ceder_netter
from .parser import UnderwritingLexer, UnderwritingParser


# module level doc-string
__doc__ = """
aggregate - a powerful aggregate loss modeling library for Python
==================================================================

**aggregate** is a Python package providing fast, accurate, and expressive data
structures that make working with aggregate probability distributions
easy and intuitive. Aggregate distributions are also known as compound distributions.
Its primary aim is to be an educational tool, allowing
experimentation with complex, **real-world** distributions. It has applications in
insurance, risk management, actuarial science, and related areas.

"""
