# aggregate


a powerful aggregate loss modeling library for Python
=====================================================

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

Technical Details
=================

Discretizing Severity Distributions
-----------------------------------

There are two simple ways to discretize a continuous distribution. 

1. Approximate the distribution with a purely discrete distribution supported at points $x_k=x_0+kb$, 
$k=0,1,\dots, N$. Call $b$ the bucket size. The discrete probabilities are 
$p_k=P(x_k - b/2 < X \le x_k+b/2)$. To create a rv_histogram variable from ``xs`` and corresponding 
 ``p`` values use:

        xss = np.sort(np.hstack((xs, xs + 1e-5)))
        pss = np.vstack((ps1, np.zeros_like(ps1))).reshape((-1,), order='F')[:-1]
        fz_discr = ss.rv_histogram((pss, xss))

The value 1e-5 just needs to be smaller than the resolution requested, i.e. do not "split the bucket". 
Generally histograms will be downsampled, not upsampled, so this is not a restriction.  
        
2. Approximate the distribution with a continuous "histogram" distribution
that is uniform on $(x_k, x_{k+1}]$. The discrete proababilities are $p_k=P(x_k < X \le x_{k+1})$. 
To create a rv_histogram variable is much easier, just use:

        xs2 = np.hstack((xs, xs[-1] + xs[1]))
        fz_cts = ss.rv_histogram((ps2, xs2))


The first methdo we call **discrete** and the second **histogram**. The discrete method is appropriate 
when the distribution will be used and interpreted as fully discrete, which is the assumption the FFT
method makes. The histogram method is useful if the distribution will be used to create a scipy.stats
rv_histogram variable. If the historgram method is interpreted as discrete and if the mean is computed
appropriately for a discrete variable as $\sum_i p_k x_k$, then the mean will be under-estimated by $b/2$. 



Plan, Progress and Issues
=========================

September 3
-----------

## Non programming Enhancements 
* Better sample of realistic severity curves
* Credit modeling: what is distortion implied by bond credit curve? By cat bond pricing? 

## Short term 
* Round trip to YAML
* Distortion that is the P/L convex envelope of a set of given points
* Errors with mass! Finite vs infinite supported distributions, lep vs ly and clin?!
* Understand output for collateral and priority!
* Fixed severity type
* Sev by name from examples in spec sev_a = name 
* Different freq dists and freq dist in exact mode 
* Integrate beta factory: Kent example should be built in: dist=beta, mean=, cv= (catch in shape from mean,cv) limit=?? 
* Exponential takes no shape 

## Medium Term
* Estimate Bucket function! Auto update
* Major paper examples?

## Nice to have enhancements 
* pf += to add a line? 
    - x = CPort(name)
    - x.add(line='home', premium=1200, lr=.5)?!  
* different ways of specfiying? .add method? add with (freq=, sev=) style
* More freq dists and shape scale specification
* occ and agg limit and attachment
* Label severity distributions to facilitate adding (user warrants labels are unique) 
* Better + function combining severity distributions 
* How to model two reinstatements?
* $N\mid N \ge n$ distribution?
* Split into subfiles and make a proper package
* sort beta factory 

## Educational Opportunities
* Uniform, triangular to normal
* Bernoulli to normal = life insurance
* $P(A>x)\sim \lambda P(X>x) \sim P(M>x)$ if thick tails
* Occ vs agg PMLs, body vs. tail. For non-cat lines it is all about correlation; for cat it is all about the tail
* Effron's theorem
* FFT exact for "making" Poisson, sum of normals is normal, expnentials is gamma etc.
* Slow convergence of truncated stable to normal
* Severity doesn't matter: difference between agg with sev and without for large claim count and stable severity
* Small large claim split approach...attrit for small; handling without correlation??
* Compound Poisson: CP(mixed sev) = sum CP(sev0 

## Practical Modeling Examples
* From limit profile
* Mixed severity 
* Modeling $N\mid N \ge n$
* How to model 2 reinstatements 

## Publication and Use 
* Interest from RMIR 

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
