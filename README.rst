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

  display(build)

  # show some built-in distributions
  build.knowledge.head()

  # model of the roll of a single dice from the built-in library
  a = build.show('^A.*1')

  # a Tweedie distribution from the built-in library
  build.show('K.Tweedie2')

  # generic frequency and severity aggregate with Poisson frequency
  a = build('agg Example 10 claims sev lognorm 50 cv 2 poisson')
  display(a)
  a.plot()
  # cdf and quantiles
  a.cdf(500), a.q(0.99)

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

