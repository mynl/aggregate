# coding: utf-8 -*-

__docformat__ = 'restructuredtext'
__author__ = "Stephen J. Mildenhall"
__copyright__ = "Copyright 2018-2019, Convex Risk LLC"
__license__ = "BSD 3-Clause New License"
__version__ = "0.7.6"
__email__ = "steve@convexrisk.com"
__status__ = "alpha"

# imports
from .param import *
from .underwriter import *
from .port import *
from .distr import *
from .spectral import *
from .utils import *
from .parser import *


# module level doc-string
__doc__ = """
aggregate - a powerful aggregate loss modeling library for Python
==================================================================

**aggregate** is a Python package providing fast, accurate, and expressive data
structures designed to make working with probability distributions
easy and intuitive. Its primary aim is to be an educational tool, allowing
experimenation with complex, **real world** distributions. It has applications in
insurance, risk management, actuarial science and related areas.


"""
