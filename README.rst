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

  pip install aggregate



Version History
-----------------

0.15.0
~~~~~~~~~

* Added pygments lexer for decl (called agg, agregate, dec, or decl)
* Added to the documentation

0.14.1
~~~~~~~

* Added scripts.py for entry points
* Updated .readthedocs.yaml to build from toml not requirements.txt
* Fixes to documentation
* ``Portfolio.tvar_threshold`` updated to use ``scipy.optimize.bisect``
* Added ``kaplan_meier`` to ``utilities`` to compute product limit estimator survival
  function from censored data. This applies to a loss listing with open (censored)
  and closed claims.
* doc to docs []
* Enhanced ``make_var_tvar`` for cases where all probabilities are equal, using linspace rather
  than cumsum.

0.13.0 (June 4, 2023)
~~~~~~~~~~~~~~~~~~~~~~~

* Updated ``Portfolio.price`` to implement ``allocation='linear'`` and
  allow a dictionary of distortions
* ``ordered='strict'`` default for ``Portfolio.calibrate_distortions``
* Pentagon can return a namedtuple and solve does not return a dataframe (it has no return value)
* Added random.py module to hold random state. Incorporated into

    - Utilities: Iman Conover (ic_noise permuation) and rearrangement algorithms
    - ``Portfolio`` sample
    - ``Aggregate`` sample
    - Spectral ``bagged_distortion``

* ``Portfolio`` added ``n_units`` property
* ``Portfolio`` simplified ``__repr__``
* Added ``block_iman_conover``  to ``utilitiles``. Note tester code in the documentation. Very Nice! 😁😁😁
* New VaR, quantile and TVaR functions: 1000x speedup and more accurate. Builder function in ``utilities``.
* pyproject.toml project specification, updated build process, now creates whl file rather than egg file.

0.12.0 (May 2023)
~~~~~~~~~~~~~~~~~~~

* ``add_exa_sample`` becomes method of ``Portfolio``
* Added ``create_from_sample`` method to ``Portfolio``
* Added ``bodoff`` method to compute layer capital allocation to ``Portfolio``
* Improved validation error reporting
* ``extensions.samples`` module deleted
* Added ``spectral.approx_ccoc`` to create a ct approx to the CCoC distortion
* ``qdp`` moved to ``utilities`` (describe plus some quantiles)
* Added ``Pentagon`` class in ``extensions``

Earlier versions
~~~~~~~~~~~~~~~~~~

See github commit notes.

Version numbers follow semantic versioning, MAJOR.MINOR.PATCH:

* MAJOR version changes with incompatible API changes.
* MINOR version changes with added functionality in a backwards compatible manner.
* PATCH version changes with backwards compatible bug fixes.

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

Help and contributions
-------------------------

Limited help available. Email me at help@aggregate.capital.

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome. Create a pull request on github and/or
email me.

Social media: https://www.reddit.com/r/AggregateDistribution/.

