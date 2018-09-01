# coding: utf-8 -*-

__docformat__ = 'restructuredtext'
__author__ = "Stephen J. Mildenhall"
__copyright__ = "Copyright 2018, Stephen J. Mildenhall"
__license__ = "BSD 3-Clause New License"
__version__ = "0.7.0"
__email__ = "mildenhs@stjohns.edu"
__status__ = "beta"

# imports
from . distr import *
from . helpers import *
from . param import *
from . port import *
from . spectral import *
from . utils import *


# module level doc-string
__doc__ = """
aggregate - a powerful aggregate loss modeling library for Python
==================================================================

**aggregate** is a Python package providing fast, flexible, and expressive data
structures designed to make working with "relational" or "labeled" data both
easy and intuitive. It aims to be the fundamental high-level building block for
doing practical, **real world** data analysis in Python. Additionally, it has
the broader goal of becoming **the most powerful and flexible open source data
analysis / manipulation tool available in any language**. It is already well on
its way toward this goal.

Main Features
-------------
Here are just a few of the things that pandas does well:

  - Easy handling of missing data in floating point as well as non-floating
    point data.
  - Size mutability: columns can be inserted and deleted from DataFrame and
    higher dimensional objects
  - Automatic and explicit data alignment: objects can be explicitly aligned
    to a set of labels, or the user can simply ignore the labels and let
    `Series`, `DataFrame`, etc. automatically align the data for you in
    computations. 
    
    
    Aggregate loss tools

    https://opensource.org/licenses/GPL-3.0

    Open: issue with mass and bodoff1 portfolio
        YAML serialization
        Does rebuilding from repr work with nested aggs?

    Aug 30:
        * Examples params split into portfolios, lines and severity
        * Examples is scriptable to return line or severity
        * Portfolio tracks all update parameters to determine recalc
        * Portfolio.update trim_df option
        * User portfolios
        * estimate bucket looks at limit and moments
        * ability to save to YAML, optionally to user set
        * Hack function to make line aggs for industry from IEE extract...very ROUGH

    Aug 29:
            removed great depenedence
            experimented with YAML persistence...tricky!
            * DONE Hash of status for last run and timestamp (np.timeformat!) [not all inputs...]
            * DONE Histogram mode: cts or discrete: other adj needed for cts? (Half bucket off?)
                - If you lower the bucket size then continuous will increase the mean by half the new (smaller) bucket size
            * DONE trim density_df to get rid of unwanted columns
            * DONE apply_distortion works on density_df and applied dist is kept for reference
            * DONE fixed severity type: how to make easy
                - 'severity': {'name': 'histogram', 'xs' : (1), 'ps': (1)}
            * DONE Rationalize graphics
                - See make_axis and make_pandas_axis...just needs to be propogated
            * DONE created example as a class, reads examples from YAML
            * DONE Include apply_distortion into density_df (easy change, easy to change back)
            removed adjust_axis_format - put in K, M, B into axis text
                - figure best format


    Aug 28: added ability to recurse and use a portfolio as a severity
            deepcopy
            drawstyle='steps-post'
            distribution_factory, deleted old verison, combined in _ex
            added logging, read_log function as dataframe
            overrode + *
            removed junk from bottom:
                old list of examples
                qd = quick display
                qdso with sort order on split _ index
                qtab quick tabulate
                cumintegralnew
                cumintegraltest
                pno - pre-iterator axes
                defuzz - now in update
                KEPT fit_distortion - ?see CAS talks, calibrate to a given distribution (? fit one transf to another?)
    Aug 27: repr and string
            uat function into CPortfolio
            insurability_triangle
            estimate function
    
"""
