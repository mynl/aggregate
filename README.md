# aggregate


aggregate - a powerful aggregate loss modeling library for Python
==================================================================

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

Missing Features
----------------

Here are some important things that ``aggregate`` does **not** do:

  - It is strictly univariate. It is impossible to model bivariate or multivariate distributions.
    As a result ``aggregate`` is fast and accurate
  - ``aggregate`` can model correlation between variables using shared mixing variables. This 
    is adequate to build realistic distributions but would not be adequate for an industrial-
    strength insurance company model.

Where to get it
---------------

* The source code is currently hosted on GitHub at:
* https://github.com/mynl/aggregate


Dependencies
------------

- [NumPy](https://www.numpy.org): 1.9.0 or higher
- [Pandas](https://github.com/pandas-dev/pandas): 0.23.0 or higher

License
-------

[BSD 3](LICENSE)


Background
----------

Work on ``aggregate`` started in 1997 at CNA Re in Matlab. 
[SCARE](http://www.mynl.com/wp/default.html) and [MALT](http://www.mynl.com/MALT/home.html) 
and SADCo in C++.

Contributing to aggregate
-------------------------

All contributions, bug reports, bug fixes, documentation improvements, 
enhancements and ideas are welcome.

Issues
======

Open 
----
- issue with mass and bodoff1 portfolio

Aug 30
------

* Examples params split into portfolios, lines and severity
* Examples is scriptable to return line or severity
* Portfolio tracks all update parameters to determine recalc
* Portfolio.update trim_df option
* User portfolios
* estimate bucket looks at limit and moments
* ability to save to YAML, optionally to user set
* Hack function to make line aggs for industry from IEE extract...very ROUGH

Aug 29
------

* removed great depenedence
* experimented with YAML persistence...tricky!
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
* removed adjust_axis_format - put in K, M, B into axis text
    - figure best format


Aug 28
-------

* added ability to recurse and use a portfolio as a severity
* deepcopy
* drawstyle='steps-post'
* distribution_factory, deleted old verison, combined in _ex
* added logging, read_log function as dataframe
* overrode + *
* removed junk from bottom:
   - old list of examples
   - qd = quick display
   - qdso with sort order on split _ index
   - qtab quick tabulate
   - cumintegralnew
   - cumintegraltest
   - pno - pre-iterator axes
   - defuzz - now in update
   - KEPT fit_distortion - ?see CAS talks, calibrate to a given - - distribution (? fit one transf to another?)
   
Aug 27: 
------

* repr and string
* uat function into CPortfolio
* insurability_triangle
* estimate function
