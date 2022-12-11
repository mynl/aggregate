.. _2_x_10mins:

10 minutes to aggregate
=========================

**Objectives:** A whirlwind test drive of ``aggregate``.

**Audience:** A curious new user.

**Prerequisites:** Ignore what you don't understand. Not everything will make sense the first time, but you will see what you can achieve with the package.

**See also:** :doc:`../2_User_Guides`.

The objective of the ``aggregate`` package is to make working with aggregate probability distributions as straightforward as working with parametric distributions (e.g., those built into ``scipy.stats``), even though their densities rarely have a closed form expression.

Contents
---------

* :ref:`Classes`
* :ref:`10 min underwriter`

Classes
--------

``aggregate`` is built around four principal classes.

#. :class:`Aggregate` that models a single unit of business, such as a line, business unit, geography, or operating division.
#. :class:`Portfolio` that models multiple units. Includes the functionality in :class:`Aggregate` and adds pricing, calibration, and allocation capabilities.
#. :class:`Severity` that models a size of loss distribution (a severity curve).
#. :class:`Underwriter` that keeps track of everything in its ``knowledge`` dataframe, interprets Dec language (DecL) programs, and acts as a helper.

There is also a :class:`Frequency` class but it is rarely used standalone. :class:`Aggregate` derives from it.

.. _10 min underwriter:

The :class:`Underwriter`
------------------------

To get started, import ``build``, a pre-configured :class:`Underwriter` and ``qd`` (quick display), a helper function. Import the usual suspects too, for good measure.

.. ipython:: python
    :okwarning:

   from aggregate import build, qd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt


Printing ``build`` reports its name, the number of aggregates and portfolios it knows how to construct from its knowledge, and other information about hyper-parameter default values.

.. ipython:: python
    :okwarning:

   build

.. _10 min create:

Creating an :class:`Aggregate` and a :class:`Portfolio`
---------------------------------------------------------

``build`` can create all other objects using DecL. To make an :class:`Aggregate` with a Poisson frequency, mean 5, and gamma severity with mean 10 and CV 1 simply run ``build`` on a DecL program. The line breaks improve readability but are cosmetic.

.. ipython:: python
    :okwarning:

    a = build('agg Example1 '
              '5 claims '
              'sev gamma 10 cv 1 '
              'poisson')
    qd(a)

The quick display reports summary exact and estimated frequency, severity, and aggregate statistics. These make it easy to see if the numerical estimation appears valid. Look for a small error in the mean and close second (CV) and third (skew) moments. ``qd`` displays the dataframe ``a.describe``.

In this case, the aggregate mean error is too high because the discretization bucket size ``bs`` is too small (see REF). Run again with a larger bucket.

.. ipython:: python
    :okwarning:

    a = build('agg Example1 '
          '5 claims '
          'sev gamma 10 cv 1 '
          'poisson'
          , bs=1/128)
    qd(a)


.. warning::

    Always use bucket sizes that have an exact binary representation (integers, 1/2, 1/4, 1/8, etc.) **Never** use 0.1 or 0.2 or other numbers that do not have an exact float representation, see REF.

To create a :class:`Portfolio` that combines two units with gamma-mixed (negative binomial) frequency and lognormal severities, build another DecL program. Again, the line breaks are cosmetic.

.. _10mins qdp:

.. ipython:: python
    :okwarning:

    p = build('port Port.1 '
                'agg Unit.A '
                    '10 claims '
                    'sev lognorm 10 cv 1 '
                    'mixed gamma .25 '
                'agg Unit.B '
                    '4 claims '
                    'sev lognorm 20 cv 1.5 '
                    'mixed gamma .3',
                bs=1/16)
    qd(p)

The quick display reports the same summary statistics for each unit and the whole portfolio. The underlying dataframe is ``p.describe``.


Reinsurance
---------------

``aggregate`` can apply per occurrence and aggregate reinsurance. Here is a very simple example where it is easy to see what is going on.

**Gross:** A triangular aggregate created as the sum of two uniform distribution on 1, 2,..., 10.

.. ipython:: python
    :okwarning:

    bg = build('agg Gross dfreq [2] dsev [1:10]')
    qd(bg)


**Net of occurrence:** Apply 3 xs 7 occurrence reinsurance to cap individual losses at 7.

.. ipython:: python
    :okwarning:

    bno = build('agg NetOcc dfreq [2] dsev [1:10] '
                'occurrence net of 3 x 7')
    qd(bno)

.. warning::

   The ``describe`` dataframe always reports gross analytic statistics (``E[X]``, ``CV(X)``, ``Skew(X)``) and the requested net or ceded estimated statistics (``Est E[X]``, ``Est CV(X)``, ``Est Skew(X)``). Look at the gross portfolio first to check computational accuracy. Net and ceded "error" report the difference to analytic gross.

**Net:** Add an aggregate 4 xs 8 reinsurance to cover on the net of occurrence distribution.

.. ipython:: python
    :okwarning:

    bn = build('agg Net dfreq [2] dsev [1:10] '
               'occurrence net of 3 xs 7 '
               'aggregate net of 4 xs 8')
    qd(bn)


The ``describe`` Dataframe
---------------------------

The ``describe`` dataframe is a property. Printing with default settings shows what ``qd`` adds.

.. ipython:: python
    :okwarning:

    qd(bg.describe)
    with pd.option_context('display.max_columns', 15):
        print(bg.describe)

The ``density_df`` Dataframe
-----------------------------

The ``density_df`` dataframe contains a wealth of information. Start with the :class:`Aggregate` flavor. It has ``2**log2`` rows and is indexed by the outcomes, all multiples of ``bs``. Columns containing ``p`` are the probability mass function, of the aggregate or severity. ``p`` and ``p_total`` are identical, the latter included for consistency with :class:`Portfolio` output. ``F`` and ``S`` are the cdf and sf (survival function). ``lev`` is the limited expected value at the ``loss`` level; ``exa`` is identical. The other columns are explained below. Here are the first five rows.


.. ipython:: python
    :okwarning:

    print(bg.density_df.shape)
    print(bg.density_df.columns)
    with pd.option_context('display.max_columns', bg.density_df.shape[1]):
        print(bg.density_df.head())

The :class:`Portfolio` flavor is far more exhaustive. It includes a variety of columns for each unit, suffixed ``_unit``, and for the complement of each unit (sum of everything but that unit) suffixed ``_ημ_unit``. The totals are suffixed ``_total``. The most important columns are FILL IN. All the column names and a subset of ``density_df`` are shown next.

.. ipython:: python
    :okwarning:

    print(p.density_df.shape)
    print(p.density_df.columns)
    with pd.option_context('display.max_columns', p.density_df.shape[1]):
        print(p.density_df.filter(regex=r'[aipex012]_Unit\.A').head())


The ``statistics`` Series and Dataframe
------------------------------------------

The ``statistics`` series (for :class:`Aggregate`) and dataframe (for :class:`Portfolio`) objects shows analytically computed mean, variance, CV, and sknewness.
They apply to the **gross** portfolio when there is reinsurance, so the results for ``bg`` and ``bno`` are the same.

.. ipython:: python
    :okwarning:

    oco = ['display.width', 150, 'display.max_columns', 15,
            'display.float_format', lambda x: f'{x:.5g}']
    with pd.option_context(*oco):
        print(bg.statistics)
        print('\n')
        print(p.statistics)


The ``report_df`` Dataframe
-----------------------------------------------

The ``report_df`` dataframe combines information from ``statistics`` with estimated moments to check if the numerical approximations appear valid. It is an expanded version of ``describe``. Numerically estimated statistics are prefaced ``est_`` or ``empirical``.

.. ipython:: python
    :okwarning:

    with pd.option_context(*oco):
        print(bg.report_df)
        print('\n')
        print(p.report_df)

The ``report_df`` provides extra information when there is a mixed severity.

.. ipython:: python
    :okwarning:

    mix = build('agg Mix '
                '25 claims '
                'sev gamma [5 10 10] cv [0.5 0.75 1.5] '
                'mixed gamma 0.5'
               )
    mix.report_df

The dataframe shows statistics for each mixture component, columns ``0,1,2``, their sum if they are added independently and their sum if there is a shared mixing variable, as there is here. The common mixing induces correlation between the mix components, acting to increases the CV and skewness, often dramatically.

Accessing Severity in an :class:`Aggregate`
-------------------------------------------

The property ``mix.sevs`` is an array of the :class:`Severity` objects in the  :class:`Aggregate` ``mix``. It can be iterated over. Each :class:`Severity` object wraps a ``scipy.stats`` continuous random variable exposed as ``fz``. The ``args`` are the shape variable(s) and ``kwds`` the scale and location variables, see REF TYPES.

.. ipython:: python
    :okwarning:

    for s in mix.sevs:
        print(s.sev_name, s.fz.args, s.fz.kwds)

The property ``mix.sev`` is a ``namedtuple`` exposing the exact weighted pdf, cdf, and sf of the underlying ``fz`` objects.

.. ipython:: python
    :okwarning:

    mix.sev.pdf(4), mix.sev.cdf(4), mix.sev.sf(4)

The way discretization works means the following are equal.

.. ipython:: python
    :okwarning:

    mix.density_df.loc[4, 'F_sev'], mix.sev.cdf(4 + mix.bs/2)


Accessing Units in a :class:`Portfolio`
----------------------------------------

The units in a :class:`Portfolio` are called ``p.line_names`` (alas, named before I thought of calling lines units to be more inclusive). The :class:`Aggregate` objects can be iterated over.

.. ipython:: python
    :okwarning:

    for u in p:
        print(u.name, u.agg_m, u.est_m)


Hyper-parameters
------------------

``log2`` and ``bs`` control numerical calculations. ``log2`` equals the log to base 2 of the number of buckets used and ``bs`` equals the bucket size. These values are printed by ``qd``.

The ``spec`` Dictionary
-------------------------

The ``spec`` dictionary contains the input information needed to create each object. For example, if ``a`` is an :class:`Aggregate`, then ``Aggregate(**a.spec)`` creates a new copy.
``spec_ex`` appends meta-information to ``spec`` about hyper-parameters.

.. ipython:: python
    :okwarning:

    from pprint import pprint
    pprint(bg.spec)

Program
---------

``program`` returns the DecL program used to create the object. It is blank if the object was not created using DecL.

.. ipython:: python
    :okwarning:

    print(bn.program)
    print(p.program)


The ``plot`` Method
--------------------

The ``plot`` method provides basic visualization. Discrete :class:`Aggregate` objects are plotted differently than continuous ones.

The reinsurance examples show the discrete output format. The plots show the gross, net of occurrence, and net severity and aggregate pmf (left) and cdf (middle), and the quantile (Lee) plot (right). The property ``bg.figure`` returns the last figure made by the object as a convenience. You could also use ``plt.gcf()``.

.. ipython:: python
    :okwarning:

    bg.plot()
    @savefig 10min_gross.png
    bg.figure.suptitle('Gross - discrete format');

    bno.plot()
    @savefig 10min_no.png
    bno.figure.suptitle('Net of occurrence');

    bn.plot()
    @savefig 10min_noa.png
    bn.figure.suptitle('Net of occurrence and aggregate');


Continuous distribution substitute the log density for the distribution in the middle.

.. ipython:: python
    :okwarning:

    a.plot()
    @savefig 10min_cts.png
    a.figure.suptitle('Continuous format');


A :class:`Portfolio` just plots the density and log density of each unit and the total.

.. ipython:: python
    :okwarning:

    p.plot()
    @savefig 10min_p.png
    p.figure.suptitle('Portfolio plot');

The ``update`` Method
----------------------

After an :class:`Aggregate` or a :class:`Portfolio` object has been created it needs to be updated to populate its ``density_df`` dataframe. ``build`` automatically updates the objects it creates with default hyper-parameter values. Sometimes it is necessary to re-update with different hyper-parameters. The ``update`` method takes arguments ``log2=13``, ``bs=0``, and ``recommend_p=0.999``. The first two control the number and size of buckets. When ``bs==0`` it is estimated using the method ``recommend_bucket``, which uses a shifted lognormal method of moments fit to the aggregate and takes the ``recommend_p`` percentile as the right-hand end of the discretization. For thick tailed distributions it is often necessary to use a value closer to 1. If ``bs!=0`` then ``recommend_p`` is ignored.

Further control over updating is available, as described in REF.

The ``snap`` Method
--------------------

``snap`` rounds an input number to the index of ``density_df``.

Statistical Functions
-------------------------

:class:`Aggregate` and :class:`Portfolio` objects include basic statistics as properties:

* ``agg_m``, ``agg_cv``, ``agg_sd``, ``agg_var`` (variance), and ``agg_skew``.

:class:`Aggregate` objects include estimated numerical statistics as well:

* ``emp_m``, ``emp_cv``, ``emp_sd``, ``emp_var``, and ``emp_skew``.

These are just conveniences.

:class:`Aggregate` and :class:`Portfolio` objects act like ``scipy.stats`` (continuous) frozen random variable objects and include the following statistical functions.


* ``pmf`` the probability mass function
* ``pdf`` the probability density function---broadly interpreted---defined as the pmf divided by ``bs``
* ``cdf`` the cumulative distribution function
* ``sf`` the survival function
* ``q`` the quantile function (left inverse cdf, value at risk)
* ``tvar`` tail value at risk function
* ``var_dict`` a dictionary of tail statistics by unit and in total

We aren't picky about whether the density is technically a density when the aggregate is actually mixed or discrete.
The discrete output (``density_df.p_*``) is interpreted as the distribution, so none of the statistical functions is interpolated.
For example:

.. ipython:: python
    :okwarning:

    print(a.pmf(2), a.pmf(2.2), a.pmf(3), a.cdf(2), a.cdf(2.2))
    print(1 - a.cdf(2), a.sf(2))
    print(a.q(a.cdf(2)))

The last line illustrates that ``q`` and ``cdf`` are inverses. The ``var_dict`` function computes tail statistics for all units, return in a dictionary.

.. ipython:: python
    :okwarning:

    p.var_dict(0.99), p.var_dict(0.99, kind='tvar')


The ``price`` Method
---------------------

The ``price`` method computes the risk adjusted expected value (technical price net of expenses) of losses limited by capital at a specified VaR threshold. The risk adjustment is determined by a spectral risk measure corresponding to an input distortion function, see REF and PIR REF.

Distortions can be built using DecL. The plot shows :math:`g` and its dual.

.. ipython:: python
    :okwarning:

    g = build('distortion Pricer dual 3')
    @savefig 10min_g.png
    g.plot();
    qd(mix.q(0.999))

The last line computes the 99.9%ile outcome that can be used to specify regulatory assets :math:`a`. ``price`` applies to :math:`X\wedge a`.

.. ipython:: python
    :okwarning:

    qd(mix.price(0.999, g).T)

The ``price`` method output reports expected limited losses ``L``, the risk adjusted premium ``P``, the margin ``M = P - L``, the capital ``Q = a - P``, the loss ratio, leverage as premium to capital ``PQ``, and return on capital ``ROE``.

When ``price`` is applied to a :class:`Portfolio` it returns the total premium and its (lifted) natural allocation to each unit, see REF, along with all the other statistics in a dataframe. Losses are allocated by equal priority in default.

.. ipython:: python
    :okwarning:

    qd(p.price(0.999, g).df.T)

The ROE varies by unit, reflecting different consumption and cost of capital by layer. The less risky unit A runs at a higher loss ratio (cheaper insurance) but higher ROE than unit B because it consumes more expensive, equity-like lower layer capital but less capital overall (higher leverage).

Conditional Expected Values
----------------------------

:class:`Portfolio` objects include a slew of functions to allocate capital (please don't) or margin (please do). These all rely on what :cite:t:`Mildenhall2022a` call the :math:`\kappa` function, defined for a sum :math:`X=\sum_i X_i` as the conditional expectation

.. math::

    \kappa_i(x) = \mathsf E[X_i\mid X=x].

Notice that :math:`\sum_i \kappa_i(x)=x`, hinting at its allocation application.
See op. cite Chapter XX for an explanation of why :math:`\kappa` is so useful. In short, it shows which units contribute to bad overall outcomes. It is in ``density_df`` as the columns ``exeqa_unit``, read as the "expected value given X eq(uals) a".

Here are some values and graph for ``p``. Looking at its
:ref:`describe<10mins qdp>` dataframe shows that Unit.B is thicker tailed, confirmed by the log density plot on the right.

.. ipython:: python
    :okwarning:

    fig, axs = plt.subplots(1, 2, figsize=(2 * 3.5, 2.45)); \
    ax0, ax1 = axs.flat;
    lm = p.limits(); \
    bit = p.density_df.filter(regex='exeqa_[Ut]'); \
    bit.index.name = 'Loss'; \
    bit.plot(xlim=lm, ylim=lm, ax=ax0); \
    ax0.set(ylabel=r'$E[X_i\mid X]$', aspect='equal'); \
    ax0.axhline(bit['exeqa_Unit.A'].max(), lw=.5, c='C7');
    @savefig 10mins_exa.png
    p.density_df.filter(regex='p_[Ut]').plot(ax=ax1, xlim=lm, logy=True); \
    ax1.set(ylabel='Log density');
    bit['Pct A'] = bit['exeqa_Unit.A'] / bit.index
    qd(bit.loc[:lm[1]:1024])

The thin horizontal line at the maximum value of ``exeqa_Unit.A`` shows that :math:`\kappa_A` is not increasing. Unit A contributes more to moderately bad outcomes than B, but in the tail unit B dominates.

Using ``filter(regex=...)`` to select columns from ``density_df`` is a helpful idiom. The total column is labeled ``_total``. Using upper case for unit names makes them easier to select.

Summary of Common Methods and Properties
------------------------------------------

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


.. ipython:: python
    :suppress:

    plt.close('all')

