.. _2_x_student:

.. reviewed 2022-11-10

Student
==========

**Objectives:** Define and give examples of aggregate probability distributions; get started using the `aggregate` library.

**Audience:** New user without knowledge of aggregate distributions or insurance.

**Prerequisites:** Basic probability theory; Python and pandas programming.


What Is an Aggregate Distribution?
----------------------------------

**Aggregate distributions** are used to model outcomes generated as the
sum of an observable quantity over a random number of events.

1. Total insurance claims from a portfolio: number of claims and amount of each claim.
2. Larvae per unit area (Neyman 1939): number of egg clusters per unit area and number of larvae per egg cluster
3. Number of vehicle occupants passing a point on the road: number of vehicles passing the point and number of occupants per vehicle

Aggregate distributions are used in many fields and go by different names, including compound, generalized, and stopped-sum distributions.

In insurance terminology, the number of events is called the
**frequency** and the amount the **severity**. The help always uses that terminology.

Formal Construction
-------------------

Let the random variable :math:`N` equal the number of events and let :math:`X_i` be a series of iid random variables modeling an observable quantity. An **aggregate distribution** is the distribution of the sum of the observables

.. math::

   A = X_1 + \cdots + X_N.

An observation from :math:`A` is realized by:

1. Sampling (simulating) :math:`N`
2. For :math:`i=1,\dots, N`, sampling the number of occupants
   :math:`X_i`
3. Add the :math:`X_i`

It is usual to assume that the values of :math:`X` and :math:`N` are independent. Check this assumption is reasonable for your use case!

Parameters for the Simple Discrete Traffic Example
--------------------------------------------------

You observe traffic past a junction over sixteen 1-minute intervals.
There are 20 vehicles in total, or 20/16=1.25 per minute on average.
This gives the **frequency distribution** :math:`N`.


================== =================== =====================
Number of vehicles Number of intervals Probability
================== =================== =====================
0                  4                   4/16=0.25
1                  6                   6/16=0.375
2                  4                   4/16=0.25
3                  2                   2/16=0.125
Total intervals    16
================== =================== =====================

Separately, you also observe the number of occupants per vehicle for
sixteen vehicles, giving the **severity distribution** :math:`X`. The
ability to determine frequency and severity using separate data studies
is a strength of the aggregate method.

=================== ================== =====================
Number of occupants Number of vehicles Probability :math:`X`
=================== ================== =====================
1                   10                 10/16=0.625
2                   3                  3/16=0.1875
3                   0                  0
4                   3                  3/16=0.1875
Total vehicles      16
=================== ================== =====================

The average number of occupants per vehicle equals
1.75=(10+6+12)/16=28/16=1.75.

From here we can build an aggregate distribution :math:`A` of the number of occupants per minute. You should work out the entire distribution of :math:`A` by hand in a spreadsheet!

Aggregate Program for the Traffic Example
-----------------------------------------

Here is the ``agg`` language program for the traffic example. The program is one line long. Python automatically concatenates strings within parenthesis; it is split for clarity.

.. code:: agg

   a = build('agg Traffic dfreq [0 1 2 3] [4/16 6/16 4/16 2/16] '
             'dsev [1 2 4] [10/16 3/16 3/16]')

-  ``agg`` is a keyword
-  Traffic is a user-selected label
-  ``dfreq`` is a keyword to specify a discrete probability distribution for the frequency (the number of outcomes). It has the form ``[outcomes] [probabilities]``

   -  ``[0 1 2 3]`` are the outcomes
   -  ``[4/16 6/16 4/16 2/16]`` are the probabilities
   -  Commas are optional
   -  Only division arithmetic is supported

-  ``dsev`` is a keyword to specify the severity (the value of each outcome). It has the same form as ``dfreq``.

   -  ``[1 2 4]`` are the outcomes; there can be gaps
   -  The probabilities are ``[10/16 3/16 3/16]``.

Creating and printing the object yields:

.. ipython:: python
    :okwarning:

    from aggregate import build
    a = build('agg Traffic '
             'dfreq [0 1 2 3] [4/16 6/16 4/16 2/16] '
             'dsev [1 2 4] [10/16 3/16 3/16]')
    a


Aggregate statistics: the mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mean of a sum equals the sum of the means. Let :math:`A = X_1 + \cdots + X_N`. If :math:`N=n` is fixed then :math:`\mathsf E[A] = n\mathsf E(X)`, because all :math:`\mathsf E[X_i]=\mathsf E[X]`. In general,

.. math::

    \mathsf E[A] = \mathsf E[X]\mathsf E[N]

by conditional probability.

Aggregate statistics: the variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

or independent random variables, the variance of a sum equals the sum of the variances.  If :math:`N=n` is fixed then :math:`\mathsf{Var}(A) = n\mathsf{Var}(X)` and :math:`\mathsf{Var}(N)=0`. If :math:`X=x` is fixed then :math:`\mathsf{Var}(A) = x^2\mathsf{Var}(N)` and :math:`\mathsf{Var}(X)=0`. Making the obvious choices :math:`n=\mathsf E[N]`, :math:`x=\mathsf E[X]` and guessing gives

.. math::

    \mathsf{Var}(A) = \mathsf E[N]\mathsf{Var}(X) + \mathsf E[X]^2\mathsf{Var}(N)

which is the correct answer!



Exercises - Test Your Understanding
--------------------------------------

Simple Aggregate Model Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a given year there can be 1, 2 or 3 events. There is a 50% chance of
1 event, 25% chance of 2, and 25% chance of 3. Each event randomly
causes a loss of 5, 10 or 15, each with equal probability.

1. What is the average annual event frequency?
2. What is the average event severity?
3. What are the average losses each year?
4. What is the coefficient of variation of losses for each year?
5. Create a table showing all possible outcomes from the model
6. What is the probability of an annual loss of 5? How can it occur?
7. What is the probability of an annual loss of 10? How can it occur?
8. What is the highest amount of total losses that can occur in one
   year? What is the chances that occurs?



.. ipython:: python
    :okwarning:

    from aggregate import build
    import pandas as pd
    sam = build('agg SAM dfreq [1 2 3] [.5 .25 .25] dsev [5 10 15]')
    sam.plot()
    @savefig student_sam.png
    print(sam)
    sam.density_df.query('p_total > 0')[['p_total', 'p_sev']]


The largest outcome of 45 has probability 0.25 * (1/3)**3 (1/4) one for count, three outcomes of 50); check accuracy:

.. ipython:: python
    :okwarning:

    a, e = (1/4) * (1/3)**3, sam.pmf(45)
    pd.DataFrame([a, e, e/a-1],
        index=['Actual worst', 'Computed worst', 'error'], columns=['value'])


A More Complex Aggregate Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a given year there can be 1, 2, 3 or 20 events. There is a 45% chance
of 1 event, 25% chance of 2, 25% chance of 3, and 5% chance of 100
events. Each event randomly causes a loss of 5, 10 or 50, each with
equal probability.

1. What is the average annual event frequency?
2. What are the average losses each year?
3. What is the coefficient of variation of losses for each year?
4. What are the probabilities of each possible outcome?
5. What are the 99 and 99.6 percentiles of aggregate losses?
6. What is the probability of a maximum loss of 1000?

.. ipython:: python
    :okwarning:

    cam = build('agg CAM dfreq [1 2 3 20] [.45 .25 .25 0.05] '
                'dsev [5 10 50] [1/3 1/3 1/3]', log2=11, bs=1)
    cam.plot()
    @savefig student_cam.png
    print(cam)

    # percentiles
    cam.q(0.99), cam.q(0.996), cam.cdf(570)


The largest outcome of 1000 has probability 0.05 * (1/3)**20 (1/4 one for count, three outcomes of 50); check accuracy:

.. ipython:: python
    :okwarning:

    a, e = 0.05 * (1/3)**20, cam.pmf(1000)
    pd.DataFrame([a, e, e/a-1],
        index=['Actual worst', 'Computed worst', 'error'],
        columns=['value'])

Finally, show the density.

.. ipython:: python
    :okwarning:

    cam.density_df.query('p_total > 0')[['p_total', 'p_sev', 'F', 'S']]



More Aggregate Examples
-------------------------

The aggregate program for one dice roll.

.. ipython:: python
    :okwarning:

    one_dice = build('agg OneDice dfreq [1] dsev [1:6]')
    one_dice.plot()
    @savefig student_onedice.png
    print(one_dice)

The program for two dice rolls, yielding the triangular distribution.

.. ipython:: python
    :okwarning:

    two_dice = build('agg TwoDice dfreq [2] dsev [1:6]')
    two_dice.plot()
    @savefig student_twodice.png
    print(two_dice)
    print(two_dice.density_df.query('p_total > 0')[['loss', 'p_total', 'F']])

The aggregate program  for twelve dice rolls, which is much harder to do by hand! The answer is compared to a moment-matched normal approximation.

.. ipython:: python
    :okwarning:

    import numpy as np
    twelve_dice = build('agg TwelveDice dfreq [12] dsev [1:6]')
    print(twelve_dice)

    fz = twelve_dice.approximate('norm')
    # model dataframe and append normal approx
    df = twelve_dice.density_df[['loss', 'p_total']]
    df['normal'] = np.diff(fz.cdf(df.loss + 0.5), prepend=0)
    print(df) # .iloc[32:52])
    df.drop(columns=['loss']).plot(xlim=[22, 64])
    @savefig student_norm12.png
    pass


Finally, a dice roll of dice rolls: throw a dice, then throw that many die.

.. ipython:: python
    :okwarning:

    dice2 = build('agg Dice2 dfreq [1:6] dsev [1:6]')
    dice2.plot()
    @savefig student_rollroll.png
    dice2


The largest  outcome of 36 has probability 6**-7; check accuracy

.. ipython:: python
    :okwarning:

    a, e = (1/6)**7, dice2.density_df.loc[36, 'p_total']
    pd.DataFrame([a, e, e/a-1],
        index=['Actual worst', 'Computed worst', 'error'],
        columns=['value'])

Create the same distribution without shorthand notation and using more basic ``agg`` language.

.. code:: ipython3

    dice21 = build('agg Dice2b dfreq [1 2 3 4 5 6]  [1/6 1/6 1/6 1/6 1/6 1/6] '
                   ' sev dhistogram xps [1 2 3 4 5 6] [1/6 1/6 1/6 1/6 1/6 1/6]')


