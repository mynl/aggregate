.. _2_x_student:

===========================================
Student
===========================================


Simple Discrete Aggregate Distributions
---------------------------------------

Aggregate Frequency and Severity Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulation algorithm for insurance losses

::

       for i = 1 to 10000
           agg = 0
           simulate number of events n
           for j = 1 to n
               simulate loss amount X
               agg = agg + X
           output agg for simulation i

-  Write :math:`A = X_1 + \cdots X_N`, :math:`X_i` and :math:`N` random
   and independent, and :math:`X_i` identically distributed
-  Model insured losses via number of claims :math:`N` the **frequency**
   and the amount :math:`X_i` of each claim, the **severity**

Aggregate statistics: the mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Mean of sum = sum of means
-  :math:`A = X_1 + \cdots + X_N`
-  If :math:`N=n` is fixed then :math:`E[A] = nE(X)`, because all
   :math:`E[X_i]=E[X]`
-  In general, :math:`E[A] = E[X]E[N]` by conditional probability

Aggregate statistics: the variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  For independent random variables, variance of sum = sum of variances
-  :math:`A = X_1 + \cdots + X_N`
-  If :math:`N=n` is fixed then :math:`Var(A) = nVar(X)` and
   :math:`Var(N)=0`
-  If :math:`X=x` is fixed then :math:`Var(A) = x^2Var(N)` and
   :math:`Var(X)=0`
-  Obvious choices: :math:`n=E[N]`, :math:`x=E[X]`
-  Combine :math:`Var(A) = E[N]Var(X) + E[X]^2Var(N)`
-  Miraculously this is the correct answer!


Simple aggregate model example
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


The largest outcome of 45 has probability $0.25 * (1/3)**3 (1/4)$ one for count, three outcomes of 50); check accuracy:

.. ipython:: python
    :okwarning:

    a, e = (1/4) * (1/3)**3, sam.density_df.loc[45, 'p_total']
    pd.DataFrame([a, e, e/a-1],
        index=['Actual worst', 'Computed worst', 'error'], columns=['value'])


A more complex aggregate model
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

    a, e = 0.05 * (1/3)**20, cam.density_df.loc[1000, 'p_total']
    pd.DataFrame([a, e, e/a-1],
        index=['Actual worst', 'Computed worst', 'error'],
        columns=['value'])

Finally, show density.

.. ipython:: python
    :okwarning:

    cam.density_df.query('p_total > 0')[['p_total', 'p_sev', 'F', 'S']]



Dice-based aggregate examples
-----------------------------

Aggregates for one dice roll.

.. ipython:: python
    :okwarning:

    one_dice = build('agg OneDice dfreq [1] dsev [1:6]')
    one_dice.plot()
    @savefig student_onedice.png
    print(one_dice)

Aggregates for two dice rolls, the triangular distgibution.

.. ipython:: python
    :okwarning:

    two_dice = build('agg TwoDice dfreq [2] dsev [1:6]')
    two_dice.plot()
    @savefig student_twodice.png
    print(two_dice)
    print(two_dice.density_df.query('p_total > 0')[['loss', 'p_total', 'F']])

Aggregates for twelve dice rolls.
Compare twelve dice roll to a moment-matched normal approximation.

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


