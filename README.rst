aggregate: a powerful aggregate loss modeling library for Python
================================================================

What is it?
-----------

**aggregate** is a Python package providing fast, accurate, and expressive data
structures designed to make working with probability distributions
easy and intuitive. Its primary aim is to be an educational tool, allowing
experimenation with complex, **real world** distributions. It has applications in
insurance, risk management, actuarial science and related areas.

Main Features
-------------

Here are just a few of the things that ``aggregate`` does well:

  - Output in tabular form using Pandas
  - Human readable persistence in YAML
  - Built in library of insurance severity curves for both catastrophe and non
    catastrophe lines
  - Built in parameterization for most major lines of insurance in the US, making it
    easy to build a "toy company" based on market share by line
  - Clear distinction between catastrophe and non-catastrohpe lines
  - Use of Fast Fourier Transforms throughout differentiates ``aggregate`` from
    tools based on simulation
  - Fast, accurate - no simulations!
  - Graphics and summaries following Pandas and Matplotlib syntax


Potential Applications
----------------------

  - Education
       * Building intuition around how loss distribtions convolve
       * Convergence to the central limit theorem
       * Generalized distributions
       * Compound Poisson distributions
       * Mixed distributiuons
       * Tail behavior based on frequency or severity tail
       * Log concavity properties
  - Pricing small insurance portfolios on a claim by claim basis
  - Analysis of default probabilities
  - Allocation of capital and risk charges
  - Detailed creation of marginal loss distributions that can then be
    sampled and used by other simulation software, e.g. to incorporate
    dependence structures, or in situations where it is necessary to
    track individual events, e.g. to compute gross, ceded and net bi-
    and trivariate distributions.

Missing Features
----------------

Here are some important things that ``aggregate`` does **not** do:

  - It is strictly univariate. It is impossible to model bivariate or multivariate distributions.
    As a result ``aggregate`` is fast and accurate
  - ``aggregate`` can model correlation between variables using shared mixing variables. This
    is adequate to build realistic distributions but would not be adequate for an industrial-
    strength insurance company model.

Documentation
-------------

http://www.mynl.com/aggregate/index.html


Where to get it
---------------

* The source code is currently hosted on GitHub at:
* https://github.com/mynl/aggregate


Installation
------------

pip install aggregate


Dependencies
------------

- [NumPy](https://www.numpy.org): 1.9.0 or higher
- [Pandas](https://github.com/pandas-dev/pandas): 0.23.0 or higher

License
-------

[BSD 3](LICENSE)

Contributing to aggregate
-------------------------

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome.

