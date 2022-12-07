.. _2_x_10mins:

10 minutes to aggregate
=========================

**Objectives:** A whirlwind test drive of ``aggregate``.

**Audience:** A curious new user.

**Prerequisites:** Ignore what you don't understand. Not everything will make sense the first time, but you will see what you can achieve with the package.

**See also:** :doc:`2_User_Guides`.

Classes
--------

There are four important classes in the library.

#. :class:`Aggregate` that models a single unit of business, such as a line, business unit, geography, or operating division.
#. :class:`Portfolio` that models multiple units. Includes the functionality in :class:`Aggregate` and adds pricing, calibration, and allocation capabilities.
#. :class:`Severity` that models a size of loss distribution (a severity curve).
#. :class:`Underwriter` that keeps track of everything in its ``knowledge`` dataframe, interprets Dec language (DecL) programs, and acts as a helper.

The ``Underwriter``
---------------------

To get started, import ``build``, a pre-configured :class:`Underwriter` and the ``qd`` quick display helper function.

.. ipython:: python
    :okwarning:

   from aggregate import build, qd
   build

``build`` reports its name, the number of aggregates and portfolios it knows how to construct from its knowledge, and other information about defaults.

Creating an ``Aggregate`` and ``Portfolio``
--------------------------------------------

``build`` can create all other objects using DecL. To make an :class:`Aggregate` with a Poisson frequency and gamma severity with mean 10 and CV 1 simply run

.. ipython:: python
    :okwarning:

    a = build('agg Example1 5 claims sev gamma 10 cv 1 poisson')
    qd(a)

The quick display reports summary exact and estimated frequency, severity, and aggregate statistics. These make it easy to see if the numerical estimation appears valid. Look for a small error in the mean and close second (CV) and third (skew) moments.

To create a :class:`Portfolio` that combines two units with gamma-mixed (negative binomial) frequency and lognormal severities, run

.. ipython:: python
    :okwarning:

    p = build('''port Port.1
        agg Unit.A 10 claims sev lognorm 10 cv 1 mixed gamma .25
        agg Unit.B 4  claims sev lognorm 20 cv 1.5 mixed gamma .3''')
    qd(p)

The quick display reports the same summary statistics for each unit and the whole portfolio.

Common Methods and Properties
--------------------------------

:class:`Aggregate` and :class:`Portfolio` both have the following methods and properties.

- ``describe`` a dataframe with key statistics; printed with the object.
- ``density_df`` a dataframe containing the relevant probability distributions and other expected value information.
- ``statistics_df`` and ``statistics_total_df`` dataframes with analytically computed mean, variance, CV, and sknewness.
- ``audit_df`` and ``report_df`` are dataframes with information to check if the numerical approximations appear valid. Numerically estimated statistics are prefaced ``est_`` or ``empirical``.


- ``spec`` a dictionary containing the input information needed to recreate each object. For example, if ``a`` is an :class:`Aggregate`, then ``Aggregate(**a.spec)`` creates a new copy.
- ``spec_ex`` a dictionary that appends meta-information to ``spec``.
- ``log2`` and ``bs`` that control numerical calculations.
- ``program`` the DecL program used to create the object. Blank if the object was not created using DecL.
- ``renamer`` a dictionary used to rename columns of member dataframes to be more human readable.

- ``plot`` a method to visualize the underlying distributions.
- ``update`` a method to run the numerical calculation of probability distributions.
- Statistical functions

    * ``pmf`` the probability mass function
    * ``pdf`` the probability density function---broadly interpreted
    * ``cdf`` the cumulative distribution function
    * ``sf`` the survival function
    * ``q`` the (left) inverse cdf, aka value at risk
    * ``tvar`` tail value at risk function
    * ``var_dict`` a dictionary of tail statistics by unit and in total

- ``recommend_bucket`` to recommend the value of ``bs``.
- ``price`` to apply a distortion (spectral) risk measure pricing rule with a variety of capital standards.
- ``snap`` to round an input number to the index of ``density_df``.

Sum of Uniforms Is Triangular, With Reinsurance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python
    :okwarning:

    # gross, net of occurrence, and net
    bg = build('agg GROSS dfreq [2] dsev [1:10]')
    bno = build('agg NET_OCC dfreq [2] dsev [1:10] '
                'occurrence net of 3 x 7')
    bn = build('agg NET dfreq [2] dsev [1:10] '
               'occurrence net of 3 x 7 '
               'aggregate net of 3 x 10')
    for b in [bg, bno, bn]:
        qd(b)


Remaining code?

.. ipython:: python
    :okwarning:

    @savefig re_b.svg
    bg.plot()

    @savefig re_bno.svg
    bno.plot()

    @savefig re_bn.svg
    bn.plot()

    ml = bg.figure.axes[0].xaxis.get_major_locator(); \
    my = bg.figure.axes[0].yaxis.get_major_locator(); \
    yl = bg.figure.axes[2].get_ylim();
    for b in [bg, bno, bn]:
        for ax in b.figure.axes[:2]:
            ax.set(xlim=[0, 22])
            if b is not bg:
                ax.xaxis.set_major_locator(ml)
        if b is not bg:
            b.figure.axes[2].yaxis.set_major_locator(my)
            b.figure.axes[2].set(ylim=yl)

