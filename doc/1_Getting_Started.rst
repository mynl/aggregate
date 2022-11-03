===================
1. Getting Started
===================

Installation
------------

Install from PyPI ::

   pip install aggregate

See https://pypi.org/project/aggregate/.


Source Code
-----------

The source code is hosted on GitHub, https://github.com/mynl/aggregate.


Documentation
-------------

The rest of the documentation is divided into five parts. If you are new to the tool, start by glancing through the tutorials and then explore the how-to guides, referring back as needed.

.. toctree::
   :maxdepth: 1

   2_User_Guide
   3_Reference
   4_agg_Language_Reference
   5_Technical_Resources
   6_Development

1. **Tutorials** explain *what* each object does. They provide the background to start getting stuff done.

   * Common elements of each class
   * Major classes: aggregate, portfolio,  distortions, underwriter

2. **How-to guides** provide step-by-step instructions to solve specific problems. They assume the material covered in the tutorials.

3. **Discussion** explaining the *why* of design.

   * About aggregate distributions
   * Basic statistics of aggregate distributions
   * Examples
   * Computing quantiles
   * auditing quantiles, Awkward discreet examples
   * Purpose of the underwriter module
   * Aggregates versus portfolios
   * Pricing methodologies inspector missed messages

4. **Reference** describing everything that each Python object can do. Once you've got the basics, look here for the details.

5. **Agg Language Reference**.

6. **Technical resources** that provide actuarial, probability, and other non-programming background. Background on agg loss dists, SRMs, and useful references.

Prerequisites
-------------

The help assumes you know how to program in Python and are familiar with the basics of aggregate distributions. Awareness of the material covered in SOA exam STAM or CAS exam MAS I is helpful. Some topics (limits, reinsurance) are described in insurance terms.

Dependencies
------------

The Python library depends on ``numpy``, ``pandas``, ``matplotlib``, ``scipy`` and uses Python 3.6 or higher.

The parser uses ``sly``, a fantastic lex/yacc for Python, https://github.com/dabeaz/sly.


License
-------

[BSD 3](LICENSE)
