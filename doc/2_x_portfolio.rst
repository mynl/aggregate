.. _2_x_portfolio:

The :class:`Portfolio` Class
==============================



Viewing data
------------

.. See the :ref:`Basics section <basics>`.

Use :attr:`build.knowledge` and :meth:`build.qshow` to view the knowledge.

.. ipython:: python
    :okwarning:

   build.knowledge.head()
   build.qshow('^E\.')

.. note::

   :meth:`DataFrame.to_numpy` does *not* include the index or column labels in the output.



:class:`Aggregate` and :class:`Portfolio` objects both have the following attributes and functions.

* A ``density_df`` a dataframe containing the relevant probability distributions and other information.
* A ``report_df`` providing a building block summary.
* A ``plot`` method to visualize the underlying distributions.
* An ``update`` method, to trigger the numerical calculation of probability distributions.
* A ``spec`` dictionary, containing the input information needed to recreate each object. For example, if ``ag`` is an ``Aggregate`` object, then ``Aggregate(**ag.spec)`` creates a new copy.
* ``log2`` and ``bs``

Second-level attributes:

* A ``statistics_df`` and ``statistics_total_df`` dataframes with theoretically derived statistical moments (mean, variance, CV, sknewness, etc.)
* An ``audit_df`` with information to check if the numerical approximations appear valid. Numerically computed statistics are prefaced ``emp_``.


Example
--------

For example, a three-unit portfolio model of an account:

.. ipython:: python
    :okwarning:

    p = build('port Account'
            '\n\tagg UnitA 100 claims 100e3 xs 0 sev lognorm 30000 cv 1.25 poisson'
            '\n\tagg UnitB 150 claims 250e3 xs 5000 sev lognorm 50000 cv 0.9 poisson'
            '\n\tagg Cat 2 claims 1e8 xs 0 sev 500e3 * pareto 1.8 - 500e3 poisson')
    p

Notice the newline and tabs for each unit. The portfolio has three sublines, UnitA, UnitB and Cat.

* UnitA has 100 (expected) claims, each pulled from a lognormal distribution with mean of 30000 and coefficient of variation 1.25 within the layer 100000 xs 0 (i.e., losses are limited at 100000). The frequency distribution is Poisson.
* UnitB is similar.
* Cat is has expected frequency of 2 claims from the indicated limit, with severity given by a Pareto distribution with shape parameter 1.8, scale 500000, shifted left by 500000. This corresponds to the usual Pareto with survival function :math:`S(x) = (\lambda / (\lambda + x))^1.8` for :math:`x >= 0`.

The portfolio can be approximated using FFTs to convolve the aggregates and add the units. The severities are first discretized using a certain bucket-size (``bs``). The `port` object has a `port.recommend_bucket()` to suggest reasonable buckets:

.. ipython:: python
    :okwarning:

    print(p.recommend_bucket().iloc[:, [0,3,6,10]])
    p.best_bucket(16)

The column ``bsN`` correspond to discretizing with 2**N buckets. The rows show suggested bucket sizes for each unit and in total. For example with ``N=16` (i.e., 65,536 buckets) the suggestion is 1727. It is best the bucket size is a divisor of any limits or attachment points, so we select 2000.


:class:`Aggregate` objects act like a discrete probability distribution. There are properties for the mean, standard deviation, coefficient of variation (cv), and skewness.

.. ipython:: python
    :okwarning:

    a.agg_m, a.agg_sd, a.agg_cv, a.agg_skew

They have probability mass, cumulative distribution, survival, and quantile (inverse of distribution) functions.

.. ipython:: python
    :okwarning:

    a.pmf(6), a.cdf(5), a.sf(6), a.q(a.cdf(6)), a.q(0.5)

The portfolio object acts like a discrete probability distribution.

::

    bs = 10000
    pf.update(13, bs)
    pf.report('quick')
    pf.plot('density')
    pf.plot('density', logy=True)
    print(pf)

    Portfolio name           MyCompanyBook
    Theoretic expected loss     10,684,541.2
    Actual expected loss        10,657,381.1
    Error                          -0.002542
    Discretization size                   13
    Bucket size                     10000.00
    <aggregate.port.Portfolio object at 0x0000023950683CF8>

