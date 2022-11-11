.. _2_x_aggregate:

The :class:`Aggregate` Class
=============================

* [QS2] Creating a simple aggregate distribution using the `agg` language (1, 2, or 3 outcomes; simple discrete severity) [cdf, pmf, sf, q, plot, describe, statistics]

* [QS3] Common design elements: More properties of `Aggregate` objects [audit_df, density_df, log2, bs, recommend_bucket, snap, update, tvar, var_dict]


Creating an :class:`Aggregate` from the pre-loaded library

.. ipython:: python
    :okwarning:

    from aggregate import build
    a, d = build.show('^B.*1$')
