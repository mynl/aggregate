.. _basics:

==============================
 Essential basic functionality
==============================

The two most important classes in the library are ``Aggregate``, which models a single unit (*unit* refers to a line of business, business unit, geography, operating division, etc.), and a ``Portfolio`` which models multiple units. The ``Portfolio`` broadly includes all the functionality in ``Aggregate`` and adds capabilities for price allocation and price functional calibration. Supporting these two are the ``Severity`` class, modeling size of loss, and the ``Underwriter`` class that keeps track of everything and acts as a helper.

``Aggregate`` and ``Portfolio`` objects both have the following attributes and functions.

* A ``density_df`` dataframe containing the relevant probability distributions and other information.
* A ``report_df`` providing a building block summary.
* A ``plot`` method to visualize the underlying distributions.
* An ``update`` method, to trigger the numerical calculation of probability distributions.
* A ``spec`` dictionary, containing the input information needed to recreate each object. For example, if ``ag`` is an ``Aggregate`` object, then ``Aggregate(**ag.spec)`` creates a new copy.
* ``log2`` and ``bs``


Second-level attributes:

* A ``statistics_df`` and ``statistics_total_df`` dataframes with theoretically derived statistical moments (mean, variance, CV, sknewness, etc.)
* An ``audit_df`` with information to check if the numerical approximations appear valid. Numerically computed statistics are prefaced ``emp_``.
