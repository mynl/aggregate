.. _2_x_aggregate:

Aggregate distributions
========================

* [QS2] Creating a simple aggregate distribution using the `agg` language (1, 2, or 3 outcomes; simple discrete severity) [cdf, pmf, sf, q, plot, describe, statistics]

* [QS3] Common design elements: More properties of `Aggregate` objects [audit_df, density_df, log2, bs, recommend_bucket, snap, update, tvar, var_dict]



``Aggregate`` and ``Portfolio`` objects both have the following attributes and functions.

* A ``density_df`` a dataframe containing the relevant probability distributions and other information.
* A ``report_df`` providing a building block summary.
* A ``plot`` method to visualize the underlying distributions.
* An ``update`` method, to trigger the numerical calculation of probability distributions.
* A ``spec`` dictionary, containing the input information needed to recreate each object. For example, if ``ag`` is an ``Aggregate`` object, then ``Aggregate(**ag.spec)`` creates a new copy.
* ``log2`` and ``bs``

Second-level attributes:

* A ``statistics_df`` and ``statistics_total_df`` dataframes with theoretically derived statistical moments (mean, variance, CV, sknewness, etc.)
* An ``audit_df`` with information to check if the numerical approximations appear valid. Numerically computed statistics are prefaced ``emp_``.

