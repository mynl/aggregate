|  |activity| |doc| |version|
|  |py-versions| |downloads|
|  |license| |packages|  |twitter|

-----

aggregate: a powerful Python actuarial modeling library
========================================================

Purpose
-----------

``aggregate`` solves insurance, risk management, and actuarial problems using realistic models that reflect
underlying frequency and severity. It delivers the speed and accuracy of parametric distributions to situations
that usually require simulation, making it as easy to work with an aggregate (compound) probability distribution
as the lognormal. ``aggregate`` includes an expressive language called DecL to describe aggregate distributions
and is implemented in Python under an open source BSD-license.

White Paper (new July 2023)
----------------------------

The `White Paper <https://github.com/mynl/aggregate/blob/master/cheat-sheets/Aggregate_white_paper.pdf>`_ describes
the purpose, implementation, and use of the class ``aggregate.Aggregate`` that
handles the creation and manipulation of compound frequency-severity distributions.

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

0.21.2
~~~~~~~~

* Misc documentation updates.
* Experimental magic functions, allowing, eg. %agg [spec] to create an aggregate object (one-liner).
* 0.21.1 yanked from pypi due to error in pyproject.toml.

0.21.0
~~~~~~~~~

* Moved ``sly`` into the project for better control.  ``sly`` is a Python implementation of lex and yacc parsing tools.
  It is written by Dave Beazley. Per the sly repo on github:

  The SLY project is no longer making package-installable releases. It's fully functional, but if choose to use it,
  you should vendor the code into your application. SLY has zero-dependencies. Although I am semi-retiring the project,
  I will respond to bug reports and still may decide to make future changes to it depending on my mood.
  I'd like to thank everyone who has contributed to it over the years. --Dave

* Experimenting with a line/cell DecL magic interpreter in Jupyter Lab to obviate the
  need for ``build``.

0.20.2
~~~~~~~~~

* risk progression logic adjusted to exclude values with zero probability; graphs
  updated to use step drawstyle.

0.20.1
~~~~~~~

* Bug fix in parser interpretation of arrays with step size
* Added figures for AAS paper to extensions.ft and extensions.figures
* Validation "not unreasonable" flag set to 0
* Added aggregate_white_paper.pdf
* Colors in risk_progression

0.20.0
~~~~~~~

* ``sev_attachment``: changed default to ``None``; in that case gross losses equal
  ground-up losses, with no adjustment. But if layer is 10 xs 0 then losses
  become conditional on X > 0. That results in a different behaviour, e.g.,
  when using ``dsev[0:3]``. Ripple through effect in Aggregate (change default),
  Severity (change default, and change moment calculation; need to track the "attachment"
  of zero and the fact that it came from None, to track Pr attaching)
* dsev: check if any elements are < 0 and set to zero before computing moments
  in dhistogram
* same for dfreq; implemented in ``validate_discrete_distribution`` in distributions module
* Default ``recommend_p=0.99999`` set in constsants module.
* ``interpreter_test_suite`` renamed to ``run_test_suite`` and includes test
  to count and report if there are errors.
* Reason codes for failing validation; Aggregate.qt becomes Aggregte.explain_validation

0.19.0
~~~~~~~

* Fixed reinsurance description formatting
* Improved splice parsing to allow explicit entry of lb and ub; needed to
  model mixtures of mixtures (Albrecher et al. 2017)

0.18.0 (major update)
~~~~~~~~~~~~~~~~~~~~~~~

* Added ability to specify occ reinsurance after a built in agg; this
  allows you to alter a gross aggregate more easily.
* ``Underwriter.safe_lookup`` uses deepcopy rather than copy to avoid
  problems array elements.
* Clean up and improved Parser and grammar

    - atom -> term is much cleaner (removed power, factor; now
      managed with prcedence and assoicativity)
    - EXP and EXPONENT are right
      associative, division is not associative so 1/2/3 gives an error.
    - Still SR conflict from dfreq [ ] [  ] because it could be the
      probabilities clause or the start of a vectorized limit clause
    - Remaining SR conflicts are from NUMBER, which is used in many
      places. This is a problem with the grammar, not the parser.
    - Added more tests to the parser test suite
    - Severity weights clause must come after locations (more natural)
    - Added ability for unconditional dsev.
    - Support for splicing (see below)

* Cleanup of ``Aggregate`` class, concurrent with creating a cheat sheet

    - many documentation updates
    - ``plot_old`` deleted
    - deleted ``delbaen_haezendonck_density``; not used; not doing anything
      that isn't easy by hand. Includes dh_sev_density and dh_agg_density.
    - deleted ``fit`` as alternative name for ``approximate``
    - deleted unused fields

* Cleanup of ``Portfolio`` class, concurrent with creating a cheat sheet

    - deleted ``fit`` as alternative name for ``approximate``
    - deleted ``q_old_0_12_0`` (old quantile), ``q_temp``, ``tvar_old_0_12_0``
    - deleted ``plot_old``, ``last_a``, ``_(inverse)_tail_var(_2)``
    - deleted ``def get_stat(self, line='total', stat='EmpMean'): return self.audit_df.loc[line, stat]``
    - deleted ``resample``, was an alias for sample

* Management of knowledge in ``Underwriter`` changed to support loading
  a database after creation. Databases not loaded until needed - alas
  that includes printing the object. TODO: Consider a change?
* Frequency mfg renamed to freq_pgf to match other Frequency class methods and
  to accuractely describe the function as a probability generating function
  rather than a moment generating function.
* Added ``introspect`` function to Utilities. Used to create a cheat sheet
  for Aggregate.
* Added cheat sheets, completed for Aggregate
* Severity can now be conditional on being in a layer (see splice); managed
  adjustments to underlying frozen rv using decorators. No overhead if not
  used.
* Added "splice" option for Severity (see Albrecher et. al ch XX) and Aggregate,
  new arguments ``sev_lb`` and ``sev_ub``, each lists.
* ``Underwriter.build`` defaults update argument to None, which uses the object default.
* pretty printing: now returns a value, no tacit mode; added _html version to
  run through pygments, that looks good in Jupyter Lab.

0.17.1
~~~~~~~~

* Adjusted pyproject.toml
* pygments lexer tweaks
* Simplified grammar: % and inf now handled as part of resolving NUMBER; still 16 = 5 * 3 + 1 SR conflicts
* Reading databases on demand in Underwriter, resulting in faster object creation
* Creating and testing exsitance of subdirectories in Undewriter on demand using properties
* Creating directories moved into Extensions __init__.py
* lexer and parser as properties for Underwriter object creation
* Default ``recommend_p`` changed from 0.999 to 0.99999.
* ``recommend_bucket`` now uses ``p=max(p, 1-1e-8)`` if severity is unlimited.


0.17.0 (July 2023)
~~~~~~~~~~~~~~~~~~~~

* ``more`` added as a proper method
* Fixed debugfile in parser.py which stops installation if not None (need to
  enure the directory exists)
* Fixed build and MANIFEST to remove build warning
* parser: semicolon no longer mapped to newline; it is now used to provide hints
  notes
* ``recommend_bucket`` uses p=max(p, 1-1e-8) if limit=inf. Default increased from 0.999
  to 0.99999 based on examples; works well for limited severity but not well for unlimited severity.
* Implemented calculation hints in note strings. Format is k=v; pairs; k
  bs, log2, padding, recommend_p, normalize are recognized. If present they are used
  if no arguments are passed explicitly to ``build``.
* Added ``interpreter_test_suite()`` to ``Underwriter`` to run the test suite
* Added ``test_suite_file`` to ``Underwriter`` to return ``Path`` to ``test_suite.agg``` file
* Layers, attachments, and the reinsurance tower can now be ranges, ``[s:f:j]`` syntax

0.16.1 (July 2023)
~~~~~~~~~~~~~~~~~~~~

* IDs can now include dashes: Line-A is a legitimate date
* Include templates and test-cases.agg file in the distribution
* Fixed mixed severity / limit profile interaction. Mixtures now work with
  exposure defined by losses and premium (as opposed to just claim count),
  correctly account for excess layers (which requires re-weighting the
  mixture components). Involves fixing the ground up severity and using it
  to adjust weights first. Then, by layer, figure the severity and convert
  exposure to claim count if necessary. Cases where there is no loss in the
  layer (high layer from low mean / low vol componet) replace by zero. Use
  logging level 20 for more details.
* Added ``more`` function to ``Portfolio``, ``Aggregate`` and ``Underwriter`` classes.
  Given a regex it returns all methods and attributes matching. It tries to call a method
  with no arguments and reports the answer. ``more`` is defined in utilities
  and can be applied to any object.
* Moved work of ``qt`` from utilities into ``Aggregate``` (where it belongs).
  Retained ``qt`` for backwards compatibility.
* Parser: power <- atom ** factor to power <- factor ** factor to allow (1/2)**(3/4)
* ``random` module renamed `random_agg`` to avoid conflict with Python ``random``
* Implemented exact moments for exponential (special case of gamma) because
  MED is a common distribution and computing analytic moments is very time
  consuming for large mixtures.
* Added ZM and ZT examples to test_cases.agg; adjusted Portfolio examples to
  be on one line so they run through interpreter_file tests.

0.16.0 (June 2023)
~~~~~~~~~~~~~~~~~~~~

* Implemented ZM and ZT distributions using decorators!
* Added panjer_ab to Frequency, reports a and b values, p_k = (a + b / k) p_{k-1}. These values can be tested
  by computing implied a and b values from r_k = k p_k / p_{k-1} = ak + b; diff r_k = a and b is an easy
  computation.
* Added freq_dist(log2) option to Freq to return the frequency distribution stand-alone
* Added negbin frequency where freq_a equals the variance multiplier


0.15.0 (June 2023)
~~~~~~~~~~~~~~~~~~~~

* Added pygments lexer for decl (called agg, agregate, dec, or decl)
* Added to the documentation
* using pygments style in ``pprint_ex`` html mode
* removed old setup scripts and files and stack.md

0.14.1 (June 2023)
~~~~~~~~~~~~~~~~~~~~

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
* Added example use of the Pollaczeck-Khinchine formula, reproducing examples from
  the `actuar`` risk vignette to Ch 5 of the documentation.

Earlier versions
~~~~~~~~~~~~~~~~~~

See github commit notes.

Version numbers follow semantic versioning, MAJOR.MINOR.PATCH:

* MAJOR version changes with incompatible API changes.
* MINOR version changes with added functionality in a backwards compatible manner.
* PATCH version changes with backwards compatible bug fixes.

Issues and Todo
-----------------

* Treatment of zero lb is not consistent with attachment equals zero.
* Flag attempts to use fixed frequency with non-integer expected value.
* Flag attempts to use mixing with inconsistent frequency distribution.

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


.. substitutions

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