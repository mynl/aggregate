.. aggregate documentation master file, created by
   sphinx-quickstart on Sat Sep  1 14:08:11 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html


aggregate: Working with Compound Probability Distributions
==========================================================

.. automodule:: aggregate

The documentation is divided into five parts. If you are new to the tool, start by glancing through the tutorials and then explore the guides, referring back as needed.

1. **Tutorials** explain *what* each object does. They provide the background to start getting stuff done.

   * Common elements of each class
   * Major classes: aggregate, portfolio,  distortions, underwriter

2. **How-to guides** provide step-by-step instructions to solve specific problems. They assume the material covered in the tutorials.

3. The **discussions** explain the *why* of design.

   * About aggregate distributions
   * Basic statistics of aggregate distributions
   * Examples
   * Computing quantiles
   * auditing quantiles, Awkward discreet examples
   * Purpose of the underwriter module
   * Aggregates versus portfolios
   * Pricing methodologies inspector missed messages

4. The **Reference** describes everything that each object can do. Once you've got the basics, look here for the details.

   * Agg language grammar and railroad diagram

5. **Technical resources** give actuarial, probability, and other non-programming backup. Background on agg loss dists, SRMs, and useful references.

**Prerequisites** You should know how to program in Python and the basics of modeling with aggregate distributions. Beginning familiarity with the topics covered in SOA exam STAM or CAS exam MAS I is helpful. Some topics (limits, reinsurance) are described in insurance terms.


Where to get it
---------------

* The source code is currently hosted on GitHub at: https://github.com/mynl/aggregate
* Install from PyPI ``pip install aggregate``, see https://pypi.org/project/aggregate/

Dependencies
------------

The usual suspects: ``numpy``, ``pandas``, ``matplotlib``, ``scipy``.

Python 3.5 or higher...much use is made of f-strings.

Plus ``sly`` - a fantastic lex/yacc for Python, https://github.com/dabeaz/sly.

License
-------

[BSD 3](LICENSE)


Table of Contents
-----------------

.. toctree::
   :maxdepth: 3

   1_Tutorials
   2_HowTo
      03_Basic_Examples.ipynb
   3_Discussion
   4_Reference
   41_underwriter
   42_parser
   43_distr
   44_port
   45_spectral
   46_utils
   47_extensions
   5_Technical

