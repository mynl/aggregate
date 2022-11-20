.. _2_x_aggregate:

The :class:`Aggregate` Class
=============================

**Objectives:** How the :class:`Aggregate` specifies and represents an aggregate distribution. Basic options and functionality.

**Audience:** User who wants to build an aggregate with parametric or discrete frequency and severity distributions.

**Prerequisites:** Familiar with building aggregates using ``build``. Understand probability theory behind aggregate distributions. Insurance and reinsurance terminology.

**See also:** :ref:`The aggregate language <2_x_agg_language>`.


How ``aggregate`` specifies and represents a distribution
----------------------------------------------------------

The objective of the ``aggregate`` package is to make working with aggregate probability distributions as straightforward as working with parametric distributions (e.g., those built into ``scipy.stats``), even though their densities rarely have a closed form expression. How should the aggregate be specified and represented?

The *specification* is an unambiguous definition of the distribution, analogous to "normal with mean 100 and standard deviation 20". It could be based on an insurance declaration (dec) page or reinsurance slip description. These are human-oriented business shorthand but are may be ambiguous and rarely explicitly specify the frequency and severity distributions, though they could be implicit in a risk classification (e.g., NY light/medium trucking).
:doc:`agg_language` remedies these deficiencies, providing a precise specification that is also human readable. Neither specification can be used directly for computation.

The *representation* is amenable to computation. It should provide a cumulative distribution function and other probability functions. These can be analytical, such as the normal cdf or Weibull distribution function. However, aggregates rarely have closed form expressions. Therefore we use a numerical approximation to the exact pdf or pmf.

There are two obvious ways to construct a numerical approximation:

#. As a discrete (arithmetic, lattice) distribution supported on :math:`0, b, 2b, \dots`.

#. As a continuous random variable with a piecewise linear distribution function.

The second approach assumes the aggregate is actually a continuous random variable, which is often not the case. For example, the Tweedie and all other compound Poisson distributions are mixed. When :math:`X` is mixed it is impossible to distinguish the jump and continuous parts when using a numerical approximation. The large jumps are obvious but the small ones are not.

We live in a discrete world. Monetary amounts are multiples of a smallest unit: the penny, cent, yen, satoshi;
at the same time, we can be prejudiced in favor of analytic solutions. Computers, however, definitely favor numerical ones.

For all of these reasons we use a discrete numerical approximation. To "know or compute an aggregate" means that we have a discrete approximation to its distribution function that is concentrated on integer multiples of a fixed bandwidth or bucket size :math:`b`. Concretely, this specifies the aggregate as the value :math:`b` and a vector of probabilities :math:`(p_0,p_1,\dots, p_{n-1})` with the interpretation

.. math:: \Pr(X=kb)=p_k.

All subsequent computations assume that this approximation **is** the aggregate distribution. Thus, moments can be estimated via

.. math:: \sum_k k^r p_i b

for example.

Specifying an Aggregate Distribution
-------------------------------------

As a precursor do describing the ``agg`` language, this section lays out the information needed to fully specify an aggregate loss distributions that typically occur in insurance. Here are some examples::

|    basic
|    basic with limit and ded
|    occ re
|    agg re
|
|    A trucking policy with a loss pick 4500, a limit of 1000, and a retention 50.
|
|    Coverage is 1M part of 3M xs 2M and 90% of 5M xs 5M and 20% of 10M xs 10M. (Excess or specific)
|
|    Covered losses are limited to 250K and subject to a 5M annual aggregate deductible. In no event shall the insurer |pay more than 15M. (aggregate cover)
|
|    Policy limit is 10M with an annual aggregate limit of 25M



Abstracting the details, the structure that emerges has seven parts

.. _seven clauses:

1. A label
2. The Exposure, optionally including occurrence limits and deductibles
3. The ground-up severity distribution
4. Occurrence reinsurance (optional)
5. The frequency distribution
6. Aggregate reinsurance (optional)
7. Additional notes (optional)

Creating an Aggregate Distribution
-------------------------------------

The ``agg`` language is the easiest way to create an :class:`Aggregate` object, via ``build``.

They can also be created directly using ``kwargs``, see :ref:`Aggregate Class`.

Basic Functionality
--------------------

An :class:`Aggregate` object has the following important methods and properties. See :ref:`Aggregate Class` for a full list.

.. most of these first mentioned in 10_mins.

- ``density_df`` a dataframe containing

    - the aggregate and severity pmf (called `p` and duplicated as `p_total` for consistency with ``Portfolio`` objects) log pmf, cdf and sf
    - the aggregate lev (duplicated as `exa`)
    - ``exlea`` (less than or equal to ``a``) which equals :math:`\mathsf E[X\mid X\le a]` as a function of ``loss``
    - ``exgta`` which equals :math:`\mathsf E[X\mid X\le a]`

- ``statistics_df`` and ``statistics_total_df`` dataframes with theoretically derived statistical moments

    - severity name, limit and attachment
    - ``freq1, freq2, freq3`` non-central frequency moments
    - ``sev1, sev2, sev3`` non-central severity moments
    - mean, cv and skewness

- ``audit_df`` a dataframe with information to check if the numerical approximations appear valid. Numerically estimated statistics are prefaced ``emp_`` (XXXX change to est) for empirical.
- ``describe`` a dataframe with key statistics that is printed with the object. Compares theoretical with gross estimated moments, providing a test of computational accuracy. It should always be reviewed after updating the object.


- ``spec`` a dictionary, containing the ``kwargs`` needed to recreate each object. For example, if ``a`` is an ``Aggregate`` object, then ``Aggregate(**a.spec)`` creates a new copy.
- ``spec_ex`` a dictionary that appends meta-information to ``spec`` including ``log2`` and ``bs``.
- ``log2`` and ``bs`` that control numerical calculations, see
- ``program`` the ``agg`` program used to create the object. Blank if the object has been created directly.
- ``renamer`` a dictionary used to rename columns of member dataframes to be more human readable.

- ``plot`` method to visualize the underlying distributions. Plots the pmf and log pmf functions and the quantile function. All the data is contained in ``density_df`` and the plots are created using ``pandas`` standard plotting commands.
- ``update`` method to update and the numerical calculation of probability distributions.
- Statistical functions

    * ``pmf`` the probability mass function
    * ``pdf`` the probability density function, given by the ``pmf`` divided by the bucket size
    * ``cdf`` the cumulative distribution function
    * ``sf`` the survival function
    * ``q`` the (left) inverse cdf, aka value at risk
    * ``tvar`` tail value at risk function
    * ``var_dict`` a dictionary of tail statistics by unit and in total

- ``recommend_bucket`` to recommend a bucket size for discretizing the distribution. Requires a second moment.
- ``price`` to apply distortion (spectral) risk measure pricing rules with a variety of capital standards, see XXXX.
- ``snap`` to round an input number to the index of ``density_df``.

