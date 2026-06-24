Distributions
=============

The computational core. :mod:`aggregate.distributions` is a thin re-export
**façade**: the classes live in kind-split submodules but are imported and used
through the stable public names (``aggregate.Aggregate`` /
``aggregate.distributions.Aggregate`` both resolve to the same class).

Class hierarchy
---------------

.. code-block:: text

   Frequency          base class for frequency distributions   (aggregate._frequency)
     └── Aggregate    compound distribution (frequency x sev)   (aggregate._aggregate)
           └── Portfolio   collection of Aggregate units        (aggregate._portfolio)

   Severity           scipy.stats wrapper + layers/limits/splicing  (aggregate._severity)

:class:`Severity` is a standalone wrapper around ``scipy.stats`` continuous RVs
and discrete empirical distributions, with support for layers, limits, and
spliced forms. :class:`Aggregate` combines a :class:`Frequency` and one or more
:class:`Severity` objects by FFT convolution.

Where the code lives
--------------------

Since 1.0.0a90–a95 the former monolith is split behind the façade:

- :mod:`aggregate._frequency`, :mod:`aggregate._severity`, :mod:`aggregate._aggregate`
  — the ``Frequency`` / ``Severity`` / ``Aggregate`` classes (kind subclasses use
  the ``Base<Kind>`` registry dispatch);
- :mod:`aggregate._fits` — the method-of-moments severity fits (``lognorm_fit``,
  ``sln_fit``, ``sgamma_fit``, …) re-exported here;
- the **concern modules** hold the orchestration that used to sit on
  ``Aggregate``: reinsurance (:mod:`aggregate._reinsurance`), bucket/window
  sizing (:mod:`aggregate._bucket_window`), validation
  (:mod:`aggregate._validation`), single-distribution pricing
  (:mod:`aggregate._pricing`), and the FFT convolution kernel
  (:mod:`aggregate._aggregate_compute`). The ``Aggregate`` methods are thin
  delegators to these. See :doc:`3_x_Internal_Architecture`.

.. currentmodule:: aggregate.distributions

.. autosummary::

   Frequency
   Severity
   Aggregate
   lognorm_fit
   sln_fit
   sgamma_fit
   gamma_fit
   beta_fit
   invgamma_fit
   invgauss_fit
   lognorm_lev
   lognorm_approx
   approximate_from_mcvsk

Frequency class
---------------

.. autoclass:: aggregate.distributions.Frequency

Severity class
--------------

.. autoclass:: aggregate.distributions.Severity

Aggregate class
---------------

.. autoclass:: aggregate.distributions.Aggregate

Severity fits and approximations
--------------------------------

.. automodule:: aggregate.distributions
   :exclude-members: Frequency, Severity, Aggregate
