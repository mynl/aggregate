aggregate: a powerful Python aggregate distribution modeling library
=====================================================================

What is it?
-----------

**aggregate** is a Python package providing an expressive language and fast,
  accurate computations to make working with aggregate (compound) probability
  distributions easy and intuitive. It allows students and practitioners to
  work with realistic real-world distributions that reflect the underlying
  frequency and severity generating processes. It has applications in
  insurance, risk management, actuarial science, and related areas.

Documentation
-------------

https://aggregate.readthedocs.io/


Where to get it
---------------

https://github.com/mynl/aggregate


Installation
------------

::

  pip install aggregate


Getting started
---------------

::

  from aggregate import build

  print(build)

>>> Underwriter   Rory
>>> Knowledge     138 programs
>>> log2          16
>>> update        True
>>> debug         False
>>> site_dir      C:\Users\steve\aggregate\databases
>>> default_dir   C:\Users\steve\documents\cloudstation\telos\python\aggregate_project\aggregate\agg

::

  # model of the roll of a single dice
  a = build('agg DICE dfreq [1] dsev [1:6]')
  print(a)

>>> Aggregate object         DICE
>>> Claim count              1.00
>>> Frequency distribution   empirical
>>> Severity distribution    dhistogram, unlimited.
>>> bs                       1
>>> log2                     4
>>> padding                  1
>>> sev_calc                 discrete
>>> normalize                True
>>> approximation            exact
>>> reinsurance              None
>>> occurrence reinsurance   No reinsurance
>>> aggregate reinsurance    No reinsurance
>>>
>>>       E[X] Est E[X]    Err E[X]   CV(X) Est CV(X)   Err CV(X) Skew(X) Est Skew(X)
>>> X
>>> Freq     1                            0
>>> Sev    3.5      3.5           0 0.48795   0.48795 -3.3307e-16       0  2.8529e-15
>>> Agg    3.5      3.5 -1.1102e-16 0.48795   0.48795  2.2204e-16       0  2.8529e-15

The DataFrame compares exact mean, CV and skewness with ``aggregate`` computation for the
frequency, severity, and aggregate components. The Tweedie distribution is a common error
term in GLM modeling. It is a compound Poisson aggregate with gamma severity. Users may be
surprised at the form of the density for small p.

::

  # a Tweedie distribution, mean 10, p=1.005, dispersion (phi, sigma^2)=4
  t = build('agg Tweedie tweedie 5 1.005 1')
  print(t)
  # check variance
  print(1 * 5**1.005, t.agg_var)

>>> Aggregate object         Tweedie
>>> Claim count              4.98
>>> Frequency distribution   poisson
>>> Severity distribution    gamma, unlimited.
>>> bs                       1/4096
>>> log2                     16
>>> padding                  1
>>> sev_calc                 discrete
>>> normalize                True
>>> approximation            exact
>>> reinsurance              None
>>> occurrence reinsurance   No reinsurance
>>> aggregate reinsurance    No reinsurance
>>>
>>>        E[X] Est E[X]    Err E[X]    CV(X) Est CV(X)   Err CV(X)  Skew(X) Est Skew(X)
>>> X
>>> Freq 4.9848                       0.44789                        0.44789
>>> Sev   1.003    1.003 -8.6819e-14 0.070888  0.070888  4.9123e-07  0.14178     0.14178
>>> Agg       5   4.9992 -0.00015419  0.44902   0.44885 -0.00036361  0.45126     0.44581
>>>
>>> 5.040398276097151 5.040398276098255

.. image:: tweedie.png
  :width: 400
  :alt: Alternative text

::

  # generic frequency and severity aggregate with Poisson frequency lognormal
  # severity mean 50 and cv 2
  a = build('agg Example 10 claims sev lognorm 50 cv 2 poisson')
  print(a)

>>> Aggregate object         Example
>>> Claim count              10.00
>>> Frequency distribution   poisson
>>> Severity distribution    lognorm, unlimited.
>>> bs                       1/16
>>> log2                     16
>>> padding                  1
>>> sev_calc                 discrete
>>> normalize                True
>>> approximation            exact
>>> reinsurance              None
>>> occurrence reinsurance   No reinsurance
>>> aggregate reinsurance    No reinsurance
>>>
>>>       E[X] Est E[X]   Err E[X]   CV(X) Est CV(X) Err CV(X)  Skew(X) Est Skew(X)
>>> X
>>> Freq    10                     0.31623                      0.31623
>>> Sev     50   49.888 -0.0022464       2    1.9314 -0.034314       14      9.1099
>>> Agg    500   498.27 -0.0034695 0.70711   0.68235 -0.035007   3.5355      2.2421


::

  # cdf and quantiles
  print(f'Pr(X<=500)={a.cdf(500)}\n0.99 quantile={a.q(0.99)}')

>>> Pr(X<=500)=0.6107533546345475
>>> 0.99 quantile=1727.125

See the documentation for more examples.

Dependencies
------------

See requirements.txt.

License
-------

[BSD 3](LICENSE)

Contributing to aggregate
-------------------------

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome.

