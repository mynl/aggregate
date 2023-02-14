|  |activity| |doc| |version|
|  |py-versions| |downloads|
|  |license| |packages|  |twitter|

.. |downloads| image:: https://img.shields.io/pypi/dm/aggregate.svg
    :target: https://pepy.tech/project/aggregate
    :alt: Downloads

.. |stars| image:: https://img.shields.io/github/stars/mynl/aggregate.svg
    :target: https://github.com/mynl/aggregate/stargazers
    :alt: Github stars

.. |forks| image:: https://img.shields.io/github/forks/mynl/aggregate.svg
    :target: https://github.com/mynl/aggregate/network/members
    :alt: Github forks

.. |contributors| image:: https://img.shields.io/github/contributors/mynl/aggregate.svg
    :target: https://github.com/mynl/aggregate/graphs/contributors
    :alt: Contributors

.. |version| image:: https://img.shields.io/pypi/v/aggregate.svg?label=pypi
    :target: https://pypi.org/project/aggregate
    :alt: Latest version

.. |activity| image:: https://img.shields.io/github/commit-activity/m/mynl/aggregate
   :target: https://github.com/mynl/aggregate
   :alt: Latest Version

.. |py-versions| image:: https://img.shields.io/pypi/pyversions/aggregate.svg
    :alt: Supported Python versions

.. |license| image:: https://img.shields.io/pypi/l/aggregate.svg
    :target: https://github.com/mynl/aggregate/blob/master/LICENSE
    :alt: License

.. |packages| image:: https://repology.org/badge/tiny-repos/python:aggregate.svg
    :target: https://repology.org/metapackage/python:aggregate/versions
    :alt: Binary packages

.. |doc| image:: https://readthedocs.org/projects/aggregate/badge/?version=latest
    :target: https://aggregate.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |twitter| image:: https://img.shields.io/twitter/follow/mynl.svg?label=follow&style=flat&logo=twitter&logoColor=4FADFF
    :target: https://twitter.com/SJ2Mi
    :alt: Twitter Follow

-----

aggregate: a powerful aggregate distribution modeling library in Python
========================================================================

Purpose
-----------

``aggregate`` solves insurance, risk management, and actuarial problems using realistic models that reflect underlying frequency and severity.
It delivers the speed and accuracy of parametric distributions to situations that usually require simulation, making it as easy to work with an aggregate (compound) probability distribution as the lognormal.
``aggregate`` includes an expressive language called DecL to describe aggregate distributions and is implemented in Python under an open source BSD-license.


Documentation
-------------

https://aggregate.readthedocs.io/


Where to get it
---------------

https://github.com/mynl/aggregate


Installation
------------

::

  pip install sly
  pip install aggregate


Getting started
---------------

To get started, import ``build``. It provides easy access to all functionality.

Here is a model of the sum of three dice rolls. The DataFrame ``describe`` compares exact mean, CV and skewness with the ``aggregate`` computation for the frequency, severity, and aggregate components. Common statistical functions like the cdf and quantile function are built-in. The whole probability distribution is available in ``a.density_df``.

::

  from aggregate import build, qd
  a = build('agg Dice dfreq [3] dsev [1:6]')
  qd(a)

>>>        E[X] Est E[X]    Err E[X]   CV(X) Est CV(X)   Err CV(X) Skew(X) Est Skew(X)
>>>  X                                                                                
>>>  Freq     3                            0                                          
>>>  Sev    3.5      3.5           0 0.48795   0.48795 -3.3307e-16       0  2.8529e-15
>>>  Agg   10.5     10.5 -3.3307e-16 0.28172   0.28172 -8.6597e-15       0 -1.5813e-13

::

  print(f'\nProbability sum < 12 = {a.cdf(12):.3f}\nMedian = {a.q(0.5):.0f}')

>>>  Probability sum < 12 = 0.741
>>>  Median = 10

The Tweedie is a common distribution used
in GLM modeling. It is a compound Poisson aggregate with gamma severity. Users may be
surprised at the form of the density for small p. Here is a Tweedie with
mean 10, p=1.005, dispersion (phi, sigma^2)=4

::

  t = build('agg Tweedie tweedie 5 1.005 1')
  qd(t)

>>>        E[X] Est E[X]    Err E[X]    CV(X) Est CV(X)   Err CV(X)  Skew(X) Est Skew(X)
>>> X
>>> Freq 4.9848                       0.44789                        0.44789
>>> Sev   1.003    1.003 -8.6819e-14 0.070888  0.070888  4.9123e-07  0.14178     0.14178
>>> Agg       5   4.9992 -0.00015419  0.44902   0.44885 -0.00036361  0.45126     0.44581

::

  # check variance
  print(1 * 5**1.005, t.agg_var)

>>> 5.04039827609 5.04039827609

.. image:: tweedie.png
  :width: 400
  :alt: Alternative text

``aggregate`` can use any ``scipy.stats`` continuous random variable as a severity, and
supports all common frequency distributions. Here is a compound-Poisson with lognormal
severity, mean 50 and cv 2.

::

  a = build('agg Example 10 claims sev lognorm 50 cv 2 poisson')
  qd(a)

>>>       E[X] Est E[X]   Err E[X]   CV(X) Est CV(X) Err CV(X)  Skew(X) Est Skew(X)
>>> X
>>> Freq    10                     0.31623                      0.31623
>>> Sev     50   49.888 -0.0022464       2    1.9314 -0.034314       14      9.1099
>>> Agg    500   498.27 -0.0034695 0.70711   0.68235 -0.035007   3.5355      2.2421

::

  # cdf and quantiles
  print(f'Pr(X<=500)={a.cdf(500):.3f}\n0.99 quantile={a.q(0.99)}')

>>> Pr(X<=500)=0.611
>>> 0.99 quantile=1727.125

See the documentation for more examples.

Dependencies
------------

See requirements.txt.

Install from source
--------------------
::

    git clone --no-single-branch --depth 50 https://github.com/mynl/aggregate.git .

    git checkout --force origin/master

    git clean -d -f -f

    python -mvirtualenv ./venv

    # ./venv/Scripts on Windows
    ./venv/bin/python -m pip install --exists-action=w --no-cache-dir -r requirements.txt

    # to create help files
    ./venv/bin/python -m pip install --upgrade --no-cache-dir pip setuptools<58.3.0

    ./venv/bin/python -m pip install --upgrade --no-cache-dir pillow mock==1.0.1 alabaster>=0.7,<0.8,!=0.7.5 commonmark==0.9.1 recommonmark==0.5.0 sphinx<2 sphinx-rtd-theme<0.5 readthedocs-sphinx-ext<2.3 jinja2<3.1.0

Note: options from readthedocs.org script.

License
-------

BSD 3 licence.


Versions
---------

Version numbers follow semantic versioning, MAJOR.MINOR.PATCH:

* MAJOR version changes with incompatible API changes.
* MINOR version changes with added functionality in a backwards compatible manner.
* PATCH version changes with backwards compatible bug fixes.

Help and contributions
-------------------------

Limited help available. Email me at help@aggregate.capital.

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome. Create a pull request on github and/or
email me.

Social media: https://www.reddit.com/r/AggregateDistribution/.

