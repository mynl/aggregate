Portfolio
=========

A :class:`~aggregate.Portfolio` is a collection of :class:`~aggregate.Aggregate`
units analysed jointly: it builds the portfolio loss distribution, allocates
capital, and prices the book and its units under a :class:`~aggregate.Distortion`.
:mod:`aggregate.portfolio` is a re-export façade; the class lives in
:mod:`aggregate._portfolio`.

Where the code lives
--------------------

The joint-distribution machinery is split by **how the joint law is built**
(Plan P4); the ``Portfolio`` methods are thin delegators:

- :mod:`aggregate._portfolio_density` — the density-based (independence) path:
  the ``add_exa`` / ``exeqa`` independent-sum kernel;
- :mod:`aggregate._portfolio_sample` — the sample-based (dependence) path:
  ``sample`` / ``add_exa_sample`` and the comonotonic / copula machinery;
- :mod:`aggregate._portfolio_common` — the FFT-vs-sample-agnostic exeqa
  numerics shared by both (capital allocation, Bodoff, convex-hull helpers).

Bucket/window sizing, validation, and pricing are shared with ``Aggregate``
through the same concern modules (:mod:`aggregate._bucket_window`,
:mod:`aggregate._validation`, :mod:`aggregate._pricing`). See
:doc:`3_x_Internal_Architecture`.

.. currentmodule:: aggregate.portfolio

.. autosummary::

   Portfolio
   make_awkward
   make_comonotonic_allocations
   swap_density_df

Portfolio class
---------------

.. autoclass:: aggregate.portfolio.Portfolio

Module functions
----------------

.. automodule:: aggregate.portfolio
   :exclude-members: Portfolio
