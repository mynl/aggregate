.. _2_x_10mins:

10 minutes to aggregate
=========================

This is a test drive of aggregate. It is aimed at users who are programming interactively in Jupyter Lab or a similar REPL (read–eval–print loop) interface. Not everything will make sense the first time, but you will see what you can achieve with the package.

The most important classes in the library are

- :class:`Aggregate`, which models a single **unit** of business. The unit refers to a line, business unit, geography, operating division, etc.
- :class:`Portfolio`, which models multiple units. Broadly includes all the functionality in :class:`Aggregate` and adds pricing, calibration, and allocation methods.
- :class:`Severity`, which models a size of loss distribution
- :class:`Underwriter`, which keeps track of everything in its ``knowledge`` dataframe, interprets ``agg`` language programs, and acts as a helper.

To get started, import ``build``, a pre-configured :class:`Underwriter`.

.. ipython:: python
    :okwarning:

   from aggregate import build
   build

``build`` reports the size of its knowledge, the aggregates and portfolios it knows how to construct, and other basic information.

``build`` can create all other objects using the agg language. To make an ``aggregate`` with a Poisson frequency and lognormal severity with mean 10 and cv 1 simply run


.. ipython:: python
    :okwarning:

    a = build('agg Example1 5 claims sev lognorm 10 cv 1 poisson')
    a

To create a portfolio that combines two aggregates with gamma-mixed (negative binomial) frequency and gamma severities, run

.. ipython:: python
    :okwarning:

    p = build('''port Port.1
        agg Unit.A 10 claims sev lognorm 10 cv 1 mixed gamma .25
        agg Unit.B 4  claims sev lognorm 20 cv 1.5 mixed gamma .3''')
    p



``Aggregate`` and ``Portfolio`` objects both have the following methods and properties:

- ``density_df`` a dataframe containing the relevant probability distributions and other expected value information.
- ``statistics_df`` and ``statistics_total_df`` dataframes with theoretically derived statistical moments (mean, variance, CV, sknewness, etc.).
- ``audit_df`` a dataframe with information to check if the numerical approximations appear valid. Numerically estimated statistics are prefaced ``est_``. Quite similar to ``report_df``.
- ``describe`` a dataframe with key statistics that is printed with the object.


- ``spec`` a dictionary, containing the input information needed to recreate each object. For example, if ``a`` is an ``Aggregate`` object, then ``Aggregate(**a.spec)`` creates a new copy.
- ``spec_ex`` a dictionary that appends meta-information to ``spec``.
- ``log2`` and ``bs`` that control numerical calculations.
- ``program`` the ``agg`` program used to create the object. Blank if the object has been created directly.
- ``renamer`` a dictionary used to rename columns of member dataframes to be more human readable.

- ``plot`` method to visualize the underlying distributions.
- ``update`` method to run the numerical calculation of probability distributions.
- Statistical functions

    * ``pmf`` the probability mass function
    * ``pdf`` the probability density function
    * ``cdf`` the cumulative distribution function
    * ``sf`` the survival function
    * ``q`` the (left) inverse cdf, aka value at risk
    * ``tvar`` tail value at risk function
    * ``var_dict`` a dictionary of tail statistics by unit and in total

- ``recommend_bucket`` to recommend how to discretize the object.
- ``price`` to apply distortion (spectral) risk measure pricing rules with a variety of capital standards.
- ``snap`` to round an input number to the index of ``density_df``.






