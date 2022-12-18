.. _2_x_student:

.. reviewed 2022-11-10

Student
==========

**Objectives:** Define and give examples of aggregate probability distributions; get started using ``aggregate``.

**Audience:** New user, with no knowledge of aggregate distributions or insurance terminology.

**Prerequisites:** Basic probability theory; Python and pandas programming.

**See also:** :doc:`../2_User_Guides`.

Contents
----------

* :ref:`st what is`
* :ref:`Formal Construction`
* :ref:`Simple Example`
* :ref:`Exercise - Test Your Understanding`
* :ref:`Dice Rolls`
* :ref:`student summary`


.. _st what is:

What Is an Aggregate Probability Distribution?
-----------------------------------------------

An **aggregate probability distribution** describes the sum of a random number of identically distributed outcome random variables. The distribution of the number called the **frequency** distribution and of the outcome the **severity** distribution.

**Examples.**

1. Total insurance claims from a portfolio: frequency equals the number of claims and the severity outcome is the amount of each claim.
2. Larvae per unit area (Neyman 1939): frequency is the number of egg clusters per unit area and severity is the number of larvae per egg cluster.
3. Number of vehicle occupants passing a point on the road: frequency is the number of vehicles passing the point and severity is the number of occupants per vehicle.
4. Total transaction value in an exchange: frequency is the number of transactions and severity is the amount of each transaction.

Aggregate distributions are used in many fields and go by different names, including compound distributions, generalized distributions, and stopped-sum distributions.


Formal Construction
-------------------

Let :math:`N` be a discrete random variable taking non-negative integer values. Its outcomes give the frequency (number) of events. Let :math:`X_i` be a series of iid severity random variables modeling an outcome. An **aggregate distribution** is the distribution of the random sum

.. math::

   A = X_1 + \cdots + X_N.

:math:`N` is called the frequency component of the aggregate and :math:`X` the severity.

An observation from :math:`A` is realized by:

1. Sample (or simulate) an outcome :math:`n` from :math:`N`
2. For :math:`i=1,\dots, n`, sample :math:`X_i`
3. Return :math:`A:=X_1 + \cdots + X_n`

It is usual to assume that the values of :math:`X` and :math:`N` are independent. Check this assumption is reasonable for your use case!

Simple Example
----------------

Frequency :math:`N` can equal 1, 2, or 3, with probabilities 1/2, 1/4, and 1/4.

Severity :math:`X` can equal 1, 2, or 4, with probabilities 5/8, 1/4, and 1/8.

Aggregate :math:`A = X_1 + \cdots + X_N`.

**Exercise.**

#. What are the expected value and CV of :math:`N`?
#. What are the expected value and CV of :math:`X`?
#. What are the expected value and CV of :math:`A`?
#. What possible values can :math:`A` take? What are the probabilities of each?

.. warning::

    Stop and solve the exercise!

The exercise is not difficult, but it requires careful bookkeeping and attention to detail. It would soon become impractical to solve by hand if there were more outcomes for frequency or severity. This is where ``aggregate`` comes in. It can solve exercise in the following few lines of code, which we now go through step-by-step.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd

The first line imports ``build`` and a helper "quick display" function ``qd``. You almost always want to start this way. The next three lines specify the aggregate using a Dec Language (DecL) program to describe its frequency and severity components.

.. ipython:: python
    :okwarning:

    a01 = build('agg Student:01 '
              'dfreq [1 2 3] [1/2 1/4 1/4] '
              'dsev [1 2 4] [5/8 1/4 1/8]')

The DecL program has three parts:

-  ``agg`` is a keyword and ``Simple`` is a user-selected name. This clause declares that  we are building an aggregate distribution.
-  ``dfreq`` is a keyword to specify the frequency distribution. The next two blocks of numbers are the outcomes ``[1 2 3]`` and their probabilities ``[1/2 1/4 1/4]``. Commas are optional in the lists and only division arithmetic is supported.
-  ``dsev`` is a keyword to specify the a discrete severity distribution. It has the same outcomes-probabilities form as ``dfreq``.

The program string is only one line long because Python automatically concatenates strings within parenthesis; it is split up for clarity. It is recommended that DecL programs be split in this way. Note the spaces at the end of each line.

Use ``qd`` to print a dataframe of statistics with answers the first three questions: the mean and CV for the frequency (``Freq``), severity (``Sev``) and aggregate (``Agg``) distributions.

.. ipython:: python
    :okwarning:

    qd(a01)

The columns ``E[X]``, ``CV(X)``, and ``Skew(X)`` report the mean, CV, and skewness for each component, and they are computed analytically.
The columns ``Est E[X]``, ``Est CV(X)``, and ``Est Skew(X)`` are computed numerically by ``aggregate``. For discrete models they equal the analytic answer because the only error introduced by ``aggregate`` comes from discretizing the severity distribution, and that is why there are no estimates for frequency. ``Err E[X]`` shows the absolute error in the mean. This handy dataframe can be accessed directly via the property ``a01.describe``. The note ``log2 = 5, bs = 1`` describe the inner workings, discussed in REF.

It remains to give the aggregate probability mass function. It is available in the dataframe ``a01.density_df``. Here are the probability masses, and distribution and survival functions evaluated for all possible aggregate outcomes.

.. ipython:: python
    :okwarning:

    qd(a01.density_df.query('p_total > 0')[['p_total', 'F', 'S']])

The possible outcomes range from 1 (frequency 1, outcome 1) to 12 (frequency 3, all outcomes 4). It is easy to check the reported probabilities are correct. It is impossible to obtain an outcome of 11.

For extra credit, here is a plot of the pmf, cdf, and the outcome Lee diagram, showing the severity and aggregate. These are produced automatically by ``a01.plot()`` from the ``density_df`` dataframe.

.. ipython:: python
    :okwarning:

    @savefig simple.png
    a01.plot()


Aggregate statistics: the mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mean of a sum equals the sum of the means. Let :math:`A = X_1 + \cdots + X_N`. If :math:`N=n` is fixed then :math:`\mathsf E[A] = n\mathsf E(X)`, because all :math:`\mathsf E[X_i]=\mathsf E[X]`. In general,

.. math::

    \mathsf E[A] = \mathsf E[X]\mathsf E[N]

by conditional probability.

Aggregate statistics: the variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The variance of a sum of independent random variables equals the sum of the variances.  If :math:`N=n` is fixed then :math:`\mathsf{Var}(A) = n\mathsf{Var}(X)` and :math:`\mathsf{Var}(N)=0`. If :math:`X=x` is fixed then :math:`\mathsf{Var}(A) = x^2\mathsf{Var}(N)` and :math:`\mathsf{Var}(X)=0`. Making the obvious associations :math:`n\leftrightarrow\mathsf E[N]`, :math:`x\leftrightarrow\mathsf E[X]` suggests

.. math::

    \mathsf{Var}(A) = \mathsf E[N]\mathsf{Var}(X) + \mathsf E[X]^2\mathsf{Var}(N).

Using conditional expectations and conditioning on the value of :math:`N` shows this  is the correct answer!

**Exercise.** Confirm the formulas for an aggregate mean and variance hold for the :ref:`Simple Example`.


Exercise - Test Your Understanding
--------------------------------------

Frequency: 1, 2 or 3 events; 50% chance of 1 event, 25% chance of 2, and 25% chance of 3. Severity: 1, 2, 4, 8 or 16, each with equal probability.

1. What is the average frequency?
2. What is the average severity?
3. What are the average aggregate?
4. What is the aggregate coefficient of variation?
5. Tabulate the probability of all possible aggregate outcomes.

Try by hand and using ``aggregate``.

Here is the ``aggregate`` solution. The probability clause in ``dsev`` can be omitted when all outcomes are equally likely.

.. ipython:: python
    :okwarning:

    a02 = build('agg Student:02 '
               'dfreq [1 2 3] [.5 .25 .25] '
               'dsev [1 2 4 8 16] ')
    qd(a02)
    qd(a02.density_df.query('p_total > 0')[['p_total', 'F', 'S']])
    @savefig less_simple.png
    a02.plot()


The largest outcome of 48 has probability 1/4 * (1/5)**3 = 1/500 = 0.002.


Dice Rolls
-------------

This section presents a series of examples involving dice rolls. The early examples are useful because you know the answer and can see ``aggregate`` is correct.

The DecL program for one dice roll. We write the simple DecL on one line.

.. ipython:: python
    :okwarning:

    one_dice = build('agg Student:01Dice dfreq [1] dsev [1:6]')
    one_dice.plot()
    @savefig student_onedice.png
    qd(one_dice)

The program for two dice rolls, yielding the triangular distribution.

.. ipython:: python
    :okwarning:

    two_dice = build('agg Student:02Dice dfreq [2] dsev [1:6]')
    two_dice.plot()
    @savefig student_twodice.png
    qd(two_dice)
    bit = two_dice.density_df.query('p_total > 0')[['p_total', 'F', 'S']]
    bit['p/12'] = bit.p_total * 12
    bit['p/12'] = bit['p/12'].astype(int)
    qd(bit)

The aggregate program for twelve dice rolls, which is much harder to compute by hand! The distribution is compared to a moment-matched normal approximation. ``fz`` is a ``scipy.stats`` normal random variable created using the ``approximate`` method.

.. ipython:: python
    :okwarning:

    import numpy as np
    import matplotlib.pyplot as plt

    twelve_dice = build('agg Student:12Dice dfreq [12] dsev [1:6]')
    qd(twelve_dice)

    fz = twelve_dice.approximate('norm')
    df = twelve_dice.density_df[['p_total', 'F', 'S']]
    df['normal'] = np.diff(fz.cdf(df.index + 0.5), prepend=0)
    qd(df.iloc[32:52])
    fig, axs = plt.subplots(1, 2, figsize=(2 * 3.5, 2.45), constrained_layout=True)
    ax0, ax1 = axs.flat
    df[['p_total', 'normal']].plot(xlim=[22, 64], ax=ax0);
    ax0.set(ylabel='pmf');
    df[['p_total', 'normal']].cumsum().plot(xlim=[22, 64], ax=ax1);
    @savefig student_norm12.png
    ax1.set(ylabel='Distribution');

The last two plots show very good convergence to the central limit theorem normal distribution.

Finally, a dice roll of dice rolls: throw a dice, then throw that many dice and add up the dots. The result ranges from 1 (throw 1 first, then 1 again) to 36 (throw 6 first, then 6 for each of the six die).

.. ipython:: python
    :okwarning:

    dd = build('agg Student:DD dfreq [1:6] dsev [1:6]')
    qd(dd)
    @savefig student_rollroll.png
    dd.plot()

The largest outcome of 36 has probability 6**-7. Here is a check of the accuracy.

.. ipython:: python
    :okwarning:

    import pandas as pd
    a, e = (1/6)**7, dd.density_df.loc[36, 'p_total']
    pd.DataFrame([a, e, e/a-1],
        index=['Actual worst', 'Computed worst', 'error'],
        columns=['value'])

We return to this example in Reinsurance Pricing, :ref:`re basic examples`.

.. _student summary:

Summary of Objects Created by DecL
-------------------------------------

Objects created by :meth:`build` in this guide.

.. ipython:: python
    :okwarning:
    :okexcept:

    from aggregate import pprint_ex
    for n, r in build.qshow('^Student:').iterrows():
        pprint_ex(r.program, split=20)


.. ipython:: python
    :suppress:

    plt.close('all')
