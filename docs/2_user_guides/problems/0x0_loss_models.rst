Loss Models Book
--------------------

Examples from the text :cite:t:`LM`, Loss Models: from data to decisions. The Loss models book is used as a text for several actuarial society exams and many college courses. KPW is shorthand for Loss Models.

Contents
~~~~~~~~~

* :ref:`Example 9.3 and 4 <example 9_3 and 9_4>`
* :ref:`Example 9.5 and 6 <example 9_5 and 9_6>`
* :ref:`Exercise 9.19  <exercise 9_19>`
* :ref:`Exercise 9.23  <exercise 9_23>`
* :ref:`Exercise 9.24  <exercise 9_24>`
* :ref:`Exercise 9.31  <exercise 9_31>`
* :ref:`Exercise 9.34  <exercise 9_34>`
* :ref:`Exercise 9.35  <exercise 9_35>`
* :ref:`Exercise 9.36  <exercise 9_36>`
* :ref:`Example 9.9 and 10  <example 9_9 and 9_10>`
* :ref:`Exercise 9.39  <exercise 9_39>`
* :ref:`Exercise 9.40  <exercise 9_40>`
* :ref:`Example 9.11  <example 9_11>`
* :ref:`Example 9.12  <example 9_12>`
* :ref:`Exercise 9.45  <exercise 9_45>`
* :ref:`Exercise 9.57 and 58  <exercise 9_57 and 9_58>`
* :ref:`Exercise 9.59  <exercise 9_59>`
* :ref:`Exercise 9.60  <exercise 9_60>`
* :ref:`Example 9.14  <example 9_14>`
* :ref:`Exercise 9.63  <exercise 9_63>`
* :ref:`Example 9.15 and 18  <example 9_15 and 9_18>`
* :ref:`Example 9.16 and 17  <example 9_16 and 9_17>`
* :ref:`Exercise 9.73  <exercise 9_73>`
* :ref:`Exercise 9.74  <exercise 9_74>`


.. _example 9_3 and 9_4:

Method of Moments Approximations, Examples 9.3 and 9.4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The observed mean (and standard deviation) of the number of claims and the individual losses over the past 10 months are 6.7 (2.3) and 179,247 (52,141), respectively. Determine the mean and standard deviation of aggregate claims per month.


.. ipython:: python
    :okwarning:

    from aggregate import build, qd, mv, MomentAggregator, round_bucket
    import scipy.stats as ss
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    moms = MomentAggregator.agg_from_fs2(6.7, 2.3**2, 179247, 52141**2)
    moms


Using normal and lognormal distributions as approximating distributions for aggregate claims, calculate the probability that claims will exceed 140% of expected costs.

.. ipython:: python
    :okwarning:

    fzn = ss.norm(loc=moms.ex, scale=moms.sd)
    sigma = np.sqrt(np.log(moms.cv**2 + 1))
    fzl = ss.lognorm(sigma, scale=moms.ex*np.exp(-sigma**2/2))
    fzn.sf(1.4 * moms.ex), fzl.sf(1.4 * moms.ex)

**Notes.**

#. How to make the lognormal...

.. _example 9_5 and 9_6:

Group Dental Insurance, Examples 9.5, 9.6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under a group dental insurance plan covering employees and their families, the premium for each married employee is the same regardless of the number of family members. The insurance company has compiled statistics showing that the annual cost of dental care per person for the benefits provided by the plan has the distribution (given in units of 25) on the left.


Furthermore, the distribution of the number of persons per insurance certificate (i.e. per employee) receiving dental care in any year has the distribution on the right.

Determine the mean and standard deviation of total payments per employee.

.. math::
    \small
    \begin{matrix}
    \begin{array}{ccccc}\hline
        x & f(x) & \qquad & n & \Pr(N=n)\\ \hline
        1   &    0.150 & &      0  &    0.05 \\
        2   &    0.200 & &      1  &    0.10 \\
        3   &    0.250 & &      2  &    0.15 \\
        4   &    0.125 & &      3  &    0.20 \\
        5   &    0.075 & &      4  &    0.25 \\
        6   &    0.050 & &      5  &    0.15 \\
        7   &    0.050 & &      6  &    0.06 \\
        8   &    0.050 & &      7  &    0.03 \\
        9   &    0.025 & &      8  &    0.01 \\
        10  &    0.025 & &         &         \\ \hline
      \end{array}
    \end{matrix}


.. ipython:: python
    :okwarning:

    kpw_9_5 = build('agg KPW.95 '
                    'dfreq  [0:8] [0.05, 0.1, 0.15, 0.2, 0.25, 0.15, 0.06, 0.03, 0.01] '
                    'dsev [1:10] [0.15, 0.2, 0.25, 0.125, 0.075, 0.05, 0.05, 0.05, 0.025, 0.025]')

    qd(kpw_9_5)
    mv(kpw_9_5)

The probability distributions are in the ``density_df`` dataframe.

.. ipython:: python
    :okwarning:

    with pd.option_context('display.max_rows', 360, 'display.max_columns', 10,
                           'display.width', 150,
                           'display.float_format', lambda x: f'{x:.5g}'):
        print(kpw_9_5.density_df.query('p > 0')[['p', 'F', 'S']])

Aggregate stop loss premiums can be computed as tail integrals of the survival function. Multiply by the units, 25.

.. ipython:: python
    :okwarning:

    (kpw_9_5.density_df.S[::-1].cumsum()[::-1] * 25)[:8]

.. _exercise 9_19:

**Exercise 9.19.** An insurance portfolio produces N = 0, 1, 3 claims with probabilities 0.5, 0.4, 0.1.
Individual claim amounts are 1 or 10 with probability 0.9, 0.1.
Individual claim amounts and N are mutually independent. Calculate the probability
that the ratio of aggregate claims to expected claims will exceed 3.0.

.. ipython:: python
    :okwarning:

    kpw_9_19 = build('agg KPW.9.19 dfreq [0 1 3] [.5 .4 .1] '
                    'dsev [1 10] [.9 .1]')
    qd(kpw_9_19)
    m = kpw_9_19.agg_m
    print(f'mean        {m:.5g}\nprobability {kpw_9_19.sf(3 * m):.4g}')

.. _exercise 9_23:

**Exercise 9.23.** An individual loss distribution is normal with mean = 100 and variance = 9. The distribution
for the number of claims has outcomes 0, 1, 2, 3 with probabilities 0.5, 0.2, 0.2, 0.1.
Determine the probability that aggregate claims exceed 100.

.. ipython:: python
    :okwarning:

    kpw_9_23 = build('agg KPW.9.23 dfreq [0:3] [1/2 1/5 1/5 1/10] '
                    'sev 3 * norm + 100')
    qd(kpw_9_23)
    qd(kpw_9_23.density_df.loc[90:110:64, ['p', 'F', 'S']])

.. _exercise 9_24:

**Exercise 9.24.** An employer self-insures a life insurance program with the following two characteristics:

1. Given that a claim has occurred, the claim amount is 2,000 with probability 0.4 and 3,000 with probability 0.6.
2. The number of claims has outcomes 0, 1, 2, 3, 4 with probabilities 1/16, 1/4, 3/8, 1/4, 1/16.

The employer purchases aggregate stop-loss coverage that limits the employer’s annual claims cost to 5,000. The aggregate stop-loss coverage costs 1,472. Determine the employer’s expected annual cost of the program, including the cost of stop-loss coverage.

.. ipython:: python
    :okwarning:

    kpw_9_24 = build('agg KPW.9.24 dfreq [0:4] [1/16 1/4 3/8 1/4 1/16] '
                    'dsev [2 3] [0.4 0.6] '
                    'aggregate net of inf xs 5')
    qd(kpw_9_24)

    net = kpw_9_24.describe.iloc[-1, 1]
    print(f'\ngross loss    {kpw_9_24.agg_m:.5g}\nretained loss {net:.5g}\n'
          f'premium       {net + 1.472:.5g}')

Working in thousands.

.. _exercise 9_31:

**Exercise 9.31.** Medical and dental claims are assumed to be independent with compound Poisson
distributions as follows:

* Medical claims 2 expected claims, amounts uniform (0, 1000)
* Dental claims 3 expected claims, amounts uniform (0, 200)

Let X be the amount of a given claim under a policy that covers both medical and
dental claims. Determine E[(X − 100)+], the expected cost (in excess of 100) of any given
claim.


.. ipython:: python
    :okwarning:

    kpw_9_31 = build('agg KPW.9.31 [2 3] claims '
                     'sev [1000 200] * uniform '
                     'occurrence ceded to inf xs 100 '
                     'poisson')
    qd(kpw_9_31)
    qd(kpw_9_31.reinsurance_audit_df.stack(0).head(3))

Could also compute impact of aggregate reinsurance structures.

.. _exercise 9_34:

**Exercise 9.34.** A compound Poisson distribution has 5 expected claim and claim amount distribution p(100) = 0.80, p(500) = 0.16, and p(1,000) = 0.04. Determine the probability that aggregate claims will be exactly 600.

.. ipython:: python
    :okwarning:

    kpw_9_34 = build('agg KPW.9.34 5 claims '
                     'dsev [1 5 10] [.8 .16 .04] '
                     'poisson')
    qd(kpw_9_34)
    print(f'{kpw_9_34.pmf(6):.6g}')
    kpw_9_34.density_df.index = kpw_9_34.density_df.index.astype(int)
    qd(kpw_9_34.density_df.query('p > 0.001')[['p', 'F', 'S']], accuracy=5)

Work in hundreds. Convert index to integer to improve display. Show all outcomes with probability greater than 0.001.

.. _exercise 9_35:

**Exercise 9.35.** Aggregate payments have a compound distribution. The frequency distribution is negative binomial with :math:`r = 16` and :math:`\beta = 6`, and the severity distribution is uniform on the interval (0, 8). Use the normal approximation to determine the premium such that the probability is 5% that aggregate payments will exceed the premium.

The negative binomial has mean :math:`r\beta` and variance :math:`r\beta(1+\beta)`. Therefore the
gamma mixing variance equals :math:`c=1/r` (since :math:`r\beta(1+\beta)=n(1+cn)`.)
Hence the mixing cv equals 0.25. The premium is the 95%ile of the aggregate distribution.

.. ipython:: python
    :okwarning:

    kpw_9_35 = build('agg KPW.9.35 96 claims '
                     'sev 8 * uniform '
                     'mixed gamma 0.25')
    qd(kpw_9_35)
    mv(kpw_9_35)
    appx = kpw_9_35.approximate('all')
    ans = {k: v.isf(0.05) for k, v in appx.items()}
    ans['FFT'] = kpw_9_35.q(0.95)
    qd(pd.DataFrame(ans.values(),
                    index=pd.Index(ans.keys(), name='method'),
                    columns=['premium']).sort_values('premium'),
      accuracy=4)

The ``approximate`` method returns a dictionary with key the method, for normal and shifted and unshifted gamma and lognormal.

.. _exercise 9_36:

**Exercise 9.36.** The number of losses is Poisson with mean 3. Loss amounts have a Burr distribution
with :math:`\alpha = 3`, :math:`\theta = 2`, and :math:`\gamma = 1`. Determine the variance of aggregate losses.

A matter of converting parameterizations. This is the ``scipy.stats`` ``burr12`` distribution.  The shape parameters are ``c=gamma`` and ``d=alpha``. ``theta`` is a scale parameter.

.. ipython:: python
    :okwarning:

    kpw_9_36 = build('agg KPW.9.36 3 claims '
                     'sev 2 * burr12 1 3 '
                     'poisson')
    qd(kpw_9_36)
    mv(kpw_9_36)
    @savefig burr.png
    kpw_9_36.plot()

.. _example 9_9 and 9_10:

Compound Poisson, Example 9.9, 9.10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Policy A has a compound Poisson distribution with 2 expected claims and severity probabilities 0.6 on a payment of 1 and 0.4 on a payment of 2. Policy B has a compound Poisson distribution with 1 expected claim and probabilities 0.7 on a payment of 1 and 0.3 on a payment of 3.

Determine the probability that the total payment on the two policies will be 2.

Figure the weighted severity by hand.

.. ipython:: python
    :okwarning:

    kpw_9_9 = build('agg KPW.9.9 3 claims '
                     'dsev [1 2 3] [1.9/3 .8/3 .3/3] '
                     'poisson')
    qd(kpw_9_9)
    print(f'{kpw_9_9.pmf(2):.6g}')
    kpw_9_9.density_df.index = kpw_9_9.density_df.index.astype(int)
    bit = kpw_9_9.density_df.query('p > 0.001')[['p', 'F', 'S']]
    bit['p*exp(3)'] = bit.p * np.exp(3)
    qd(bit, accuracy=5)

The last column answers Example 9.10.

Alternatively, use the :class:`Portfolio` class.

.. ipython:: python
    :okwarning:

    p = build('port KPW.9.9.p '
              'agg A 2 claims dsev [1 2] [.6 .4] poisson '
               'agg B 1 claims dsev [1 3] [.7 .3] poisson')
    qd(p)

.. _exercise 9_39:

**Exercise 9.39.** For a compound distribution, frequency has a binomial distribution with parameters m = 3 and q = 0.4 and severity has an exponential distribution with a mean of 100. Calculate :math:`\Pr(A \le 300)`.

Assume 1.2 expected claims. Work in hundreds.

.. ipython:: python
    :okwarning:

    kpw_9_39 = build('agg KPW.9.39 1.2 claims '
                     'sev expon binomial 0.4')
    qd(kpw_9_39)
    print(f'probability = {kpw_9_39.cdf(3):.6g}')

.. _exercise 9_40:

**Exercise 9.40.**  A company sells three policies. For policy A, all claim payments are 10,000 and a single policy has a Poisson number of claims with mean 0.01. For policy B, all claim payments are 20,000 and a single policy has a Poisson number of claims with mean 0.02. For policy C, all claim payments are 40,000 and a single policy has a Poisson number of claims with mean 0.03. All policies are independent. For the coming year, there are 5,000, 3,000, and 1,000 of policies A, B, and C, respectively. Calculate the expected total payment, the standard deviation of total payment, and the probability that total payments will exceed 30,000.

Must use a :class:`Portfolio`. Work in thousands.

.. ipython:: python
    :okwarning:

    kpw_9_40 = build('port kpw_9_40\n'
                     '\tagg A 50 claims dsev [10] poisson\n'
                     '\tagg B 60 claims dsev [20] poisson\n'
                     '\tagg C 30 claims dsev [40] poisson\n')
    qd(kpw_9_40)
    qd(pd.Series({'expected payment': kpw_9_40.agg_m,
                 'sd payment': kpw_9_40.agg_sd,
                 'Pr > 3000': kpw_9_40.sf(3000)}).to_frame('value'),
                 accuracy=5)

.. _example 9_11:

ZM Binomial, Example 9.11
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A compound distribution has a zero-modified binomial distribution with 𝑚 = 3, :math:`q = 0.3`, and :math:`p_0^M = 0.4`. Individual payments are 0, 50, and 150, with probabilities 0.3, 0.5, and 0.2, respectively. Use the recursive formula to determine the probability distribution of :math:`S`.

.. todo::
    Implement ZM and ZT.

.. _example 9_12:

ETNB, Example 9.12
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The number of claims has a Poisson–ETNB distribution with Poisson parameter 𝜆 = 2 and ETNB parameters :math:`\beta = 3` and :math:`r = 0.2`. The claim size distribution has probabilities 0.3, 0.5, and 0.2 at 0, 10, and 20, respectively. Determine the total claims distribution recursively.


.. todo::
    Implement ZM and ZT.

.. _exercise 9_45:

**Exercise 9.45.** For a compound Poisson distribution, has 6 expected claims and individual losses take values 1, 2, 4 with equal probabilities. Determine the distribution of the aggregate.


.. ipython:: python
    :okwarning:

    kpw_9_45 = build('agg KPW.9.45 6 claims '
                     'dsev [1 2 4] poisson')
    qd(kpw_9_45)
    qd(kpw_9_45.density_df.query('p > 0.001')[['p', 'F', 'S']], accuracy=5)


**Exercise 9.47.** Aggregate claims are compound Poisson with 2 expected claims and severity outcomes 1, 2 with probability 1/4 and 3/4.
For a premium of 6, an insurer covers aggregate claims and agrees to pay a dividend (a refund of premium) equal to the excess, if any, of 75% of the premium over 100% of the claims. Determine the excess of premium over expected claims and dividends.

.. ipython:: python
    :okwarning:

    kpw_9_47 = build('agg KPW.9.47 2 claims '
                     'dsev [1 2] [1/4 3/4] poisson')
    qd(kpw_9_47)

    bit = kpw_9_47.density_df.query('p > 0')[['p', 'F', 'S']]
    bit['dividend'] = np.maximum(0.75 * 6 - bit.index, 0)
    qd(bit.head(10), accuracy=4)
    exp_div = (bit.dividend * bit.p).sum()
    print(f'prem      = {6:.5g}\n'
          f'exp loss  = {kpw_9_47.agg_m:.5g}\n'
          f'dividend  = {exp_div:.5g}\n'
          f'excess    = {6 - kpw_9_47.agg_m - exp_div:.5g}')

.. _exercise 9_57 and 9_58:

**Exercise 9.57, 9.58.** Aggregate losses have a compound Poisson claim distribution with 3 expected claims and individual claim amount distribution p(1) = 0.4, p(2) = 0.3, p(3) = 0.2, and p(4) = 0.1. Determine the probability that aggregate losses do not exceed 3.

Repeat the Exercise with a negative binomial frequency distribution with r = 6 and
:math:`\beta = 0.5`.

.. GO back and fix prior...
  exp = r b / (1+b) = 2
  CV = sqrt(1/6)

.. ipython:: python
    :okwarning:

    kpw_9_57 = build('agg KPW.9.57 3 claims '
                     'dsev [1:4] [.4 .3 .2 .1] poisson')
    qd(kpw_9_57)
    kpw_9_58 = build('agg KPW.9.58 3 claims '
                     'dsev [1:4] [.4 .3 .2 .1] mixed gamma 6**-0.5')
    qd(kpw_9_58)

    bit = pd.concat((kpw_9_57.density_df[['p', 'F', 'S']],
                     kpw_9_58.density_df[['p', 'F', 'S']]),
                    keys=('Po', 'NB'), axis=1)
    qd(bit.head(16), accuracy=5)

.. _exercise 9_59:

**Exercise 9.59.** A policy covers physical damage incurred by the trucks in a company’s fleet. The number of losses in a year has a Poisson distribution with expectation 5. The amount of a single loss has a gamma distribution with shape 0.5 and scale 2,500. The insurance contract pays a maximum annual benefit of 20,000. Determine the probability that the maximum benefit will be paid. Use a span of 100 and the method of rounding.

.. ipython:: python
    :okwarning:

    kpw_9_59 = build('agg KPW.9.59 5 claims '
                     'sev 2500 * gamma 0.5 '
                     'poisson')
    qd(kpw_9_59)
    print(f'pr(loss >= 20000) = {kpw_9_59.sf(20000):.6g}')


Repeated at the requested span of 100.

.. ipython:: python
    :okwarning:

    kpw_9_59.update(log2=10, bs=100)
    print(f'pr(loss >= 20000) = {kpw_9_59.sf(20000):.6g}')


.. _exercise 9_60:

**Exercise 9.60.** An individual has purchased health insurance, for which they pay 10 for each physician visit and 5 for each prescription. The probability that a payment will be 10 is 0.25, and the probability that it will be 5 is 0.75. The total number of payments per year has the Poisson–Poisson (Neyman Type A) distribution with primary mean 10 and secondary mean 4. Determine the probability that total payments in one year will exceed 400. Compare your answer to a normal approximation.

.. ipython:: python
    :okwarning:

    kpw_9_60 = build('agg KPW.9.60 40 claims '
                     'dsev [5 10] [3/4 1/4] '
                     'neyman 4')
    qd(kpw_9_60)

    fz = kpw_9_60.approximate('norm')
    print(f'FFT            {kpw_9_60.sf(400):.5g}\n'
          f'Normal approx  {fz.sf(400):.5g}')


.. _example 9_14:

Poisson Pareto, Example 9.14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The number of ground-up losses is Poisson distributed with mean 3. The individual loss distribution is Pareto with shape parameter :math:\alpha= 4` and scale parameter 10. An individual ordinary deductible of 6, coinsurance of 75%, and an individual loss limit of 24 (before application of the deductible and coinsurance) are all applied. Determine the mean, variance, and distribution of aggregate payments.

The covered layer is 18 xs 6, in which the insured pays 25% because of the coinsurance clause. The severity is unconditional.

.. ipython:: python
    :okwarning:

    kpw_9_14 = build('agg KPW.9.14 3 claims '
                     '18 xs 6 '
                     'sev 10 * pareto 4 - 10 ! '
                     'occurrence net of 0.25 so inf xs 0 '
                     'poisson')
    qd(kpw_9_14)
    print(f'variance = {kpw_9_14.describe.iloc[-1,[1, 4]].prod()**2:.6g}\ncomputed with bs=1/{1/kpw_9_14.bs:.0f} and log2={kpw_9_14.log2}')
    qd(kpw_9_14.density_df.loc[[0, 1, 2, 3], ['p', 'F', 'S']])
    @savefig kpw_9_14.png
    kpw_9_14.plot()


``describe`` returns gross under ``E[X]`` and the requested net or ceded under ``Est E[X]``. The print statement computes net variance from the product of estimated mean and cv. The spikes on the density corresponds to the possibility of only limit claims.

.. **TODO** harmonize with their answer for probabilities.

.. _exercise 9_63:

**Exercise 9.63.** A ground-up model of individual losses has a gamma distribution with shape parameter 2 and scale 100. The number of losses has a negative binomial distribution with :math:`r = 2` and :math:`\beta = 1.5`. An ordinary deductible of 50 and a loss limit of 175 (before imposition of the deductible) are applied to each individual loss.

* Determine the mean and variance of the aggregate payments on a per-loss basis.
* Determine the distribution of the number of payments.
* Determine the cumulative distribution function of the amount of a payment, given that a payment is made.
* Discretize the severity distribution using the method of rounding and a span of 40.
* Calculate the discretized distribution of aggregate payments up to a discretized amount paid of 120.

Negative binomial :math:`c=1/2` and hence mixing cv :math:`\sqrt{c}`, and the mean equals :math:`r\beta/(1+\beta)=1.4`. The cover is 125 xs 50. The severity is unconditional. First, the default calculation using ``bs=1/64``.

.. ipython:: python
    :okwarning:

    kpw_9_63 = build('agg KPW.9.63 1.4 claims '
                     '125 xs 50 '
                     'sev 100 * gamma 2 ! '
                     'mixed gamma 2**-0.5')
    qd(kpw_9_63)
    mv(kpw_9_63)
    qd(kpw_9_63.density_df.loc[:400:40*64,
        ['p', 'F', 'S', 'p_sev', 'F_sev', 'S_sev']],
        accuracy=5)

Next, calculations performed with the requested broader ``bs=40``.

.. ipython:: python
    :okwarning:

    kpw_9_63.update(log2=8, bs=40)
    qd(kpw_9_63)
    qd(kpw_9_63.density_df.loc[:400,
        ['p', 'F', 'S', 'p_sev', 'F_sev', 'S_sev']],
        accuracy=5)

The apparent difference in the severity distribution is caused by the rounding method. In the first case F(40) is almost exact whereas in the second it is actually F(60).

.. _example 9_15 and 9_18:

Group Life Individual Risk Model, Example 9.15, 9.18
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider a group life insurance contract with an accidental death benefit. Assume that for all members the probability of death in the next year is 0.01 and that 30% of deaths are accidental. For 50 employees, the benefit for an ordinary death is 50,000 and for an accidental death it is 100,000. For the remaining 25 employees, the benefits are 75,000 and 150,000, respectively. Develop an individual risk model and determine its mean and variance.

The :class:`Portfolio` solution, working in thousands.

.. ipython:: python
    :okwarning:

    kpw_9_15p = build('port KPW.9.15.p '
                      'agg A 0.5 claims '
                          'dsev [50 100] [0.7 0.3] '
                          'binomial 0.01 '
                      'agg B 0.25 claims '
                          'dsev [75 150] [0.7 0.3] '
                          'binomial 0.01 ')
    qd(kpw_9_15p)
    mv(kpw_9_15p)


The ``density_df`` dataframe contains the exact aggregate distribution, which is not easy to compute by other means. KPW says (emphasis added)

    With regard to calculating the probabilities, there are at least three options. One is to do an **exact calculation**, which involves **numerous convolutions** and **almost always requires more excessive computing time**. Recursive formulas have been developed, but they are cumbersome and are not presented here. For one such method, see De Pril [27]. One alternative is a parametric approximation as discussed for the collective risk model. Another alternative is to replace the individual risk model with a similar collective risk model and then do the calculations with that model. These two approaches are presented here.

The following solution attempts to commute convolution through the mixture. This works for a compound Poisson.
However, the sum of binomials is not binomial, and so the frequencies can't be independent binomial. They can be independent Poisson because it is additive.

.. ipython:: python
    :okwarning:

    kpw_9_15w = build('agg KPW.9.15.w '
                     '0.75 claims '
                     'dsev [50 75 100 150] '
                     '[0.35/0.75, 0.175/0.75, 0.15/0.75, 0.075/0.75] '
                     'binomial 0.01 ')
    qd(kpw_9_15w)
    mv(kpw_9_15w)

The compound Poisson approximation matches the mean but its variance is slightly off.

.. ipython:: python
    :okwarning:

    kpw_9_15cp = build('agg KPW.9.15.cp '
                     '0.75 claims '
                     'dsev [50 75 100 150] '
                     '[0.35/0.75, 0.175/0.75, 0.15/0.75, 0.075/0.75] '
                     'poisson ')
    qd(kpw_9_15cp)
    mv(kpw_9_15cp)

Comparing probabilities shows that all three distributions are very close.

.. ipython:: python
    :okwarning:

    bit = pd.concat((kpw_9_15p.density_df.loc[:400:128, ['p_total']].query('p_total > 1e-10'),
                     kpw_9_15cp.density_df.loc[:400, ['p_total']].query('p_total > 0'),
                     kpw_9_15w.density_df.loc[:400, ['p_total']].query('p_total > 0'),
                    ),
                    keys=('exact', 'compound Po', 'wrong'), axis=1).rename(columns={'p_total': 'p'})
    bit = bit.droplevel(1, axis=1)
    bit.index.name = 'loss'
    qd(bit, accuracy=5)

.. _example 9_16 and 9_17:

Group Life Individual Risk Model, Example 9.16, 9.17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A small manufacturing business has a group life insurance contract on its 14 permanent employees. The actuary for the insurer has selected a mortality table to represent the mortality of the group. Each employee is insured for the amount of his or her salary rounded up to the next 1,000. The group’s data are shown in the next table.

.. math::
    \small
    \begin{matrix}
    \begin{array}{cccrr} \hline
    \text{Employee} & \text{Age} & \text{Sex} & \text{Benefit} &       q \\ \hline
           1 &  20 &   M &  15,000 & 0.00149 \\
           2 &  23 &   M &  16,000 & 0.00142 \\
           3 &  27 &   M &  20,000 & 0.00128 \\
           4 &  30 &   M &  28,000 & 0.00122 \\
           5 &  31 &   M &  31,000 & 0.00123 \\
           6 &  46 &   M &  18,000 & 0.00353 \\
           7 &  47 &   M &  26,000 & 0.00394 \\
           8 &  49 &   M &  24,000 & 0.00484 \\
           9 &  64 &   M &  60,000 & 0.02182 \\
          10 &  17 &   F &  14,000 & 0.00050 \\
          11 &  22 &   F &  17,000 & 0.00050 \\
          12 &  26 &   F &  19,000 & 0.00054 \\
          13 &  37 &   F &  30,000 & 0.00103 \\
          14 &  55 &   F &  55,000 & 0.00479 \\ \hline
    \end{array}
    \end{matrix}

If the insurer adds a 45% relative loading to the net (pure) premium, what are the chances that it will lose money in a given year? Use the normal and lognormal approximations.

In order to make the answer self-contained, the code below includes the data munging to re-create the table, pasted from a pdf.

.. ipython:: python
    :okwarning:

    data = '''1
    20
    M
    15,000
    0.00149
    2
    23
    M
    16,000
    0.00142
    3
    27
    M
    20,000
    0.00128
    4
    30
    M
    28,000
    0.00122
    5
    31
    M
    31,000
    0.00123
    6
    46
    M
    18,000
    0.00353
    7
    47
    M
    26,000
    0.00394
    8
    49
    M
    24,000
    0.00484
    9
    64
    M
    60,000
    0.02182
    10
    17
    F
    14,000
    0.00050
    11
    22
    F
    17,000
    0.00050
    12
    26
    F
    19,000
    0.00054
    13
    37
    F
    30,000
    0.00103
    14
    55
    F
    55,000
    0.00479'''
    sdata = data.split('\n')
    df = pd.DataFrame(zip(*[iter(sdata)]*5),
                      columns=['Employee', 'Age', 'Sex', 'Benefit', 'q'])
    df.Benefit = df.Benefit.str.replace(',','').astype(float)
    df.q = df.q.astype(float)
    df = df.set_index('Employee')
    qd(df)
    print(f'expected claim count = {df.q.sum():.6g}')

Here are the FFT-exact, and various approximations to the required probability. Working in thousands. The ``dsev`` clauses enter the fixed benefit amount for each employee. Note the outsize impact of employee 9.

.. ipython:: python
    :okwarning:

    from aggregate import Portfolio
    a = [build(f'agg ee.{i} {r.q} claims '
               f'dsev [{r.Benefit / 1000}] '
               f'bernoulli')
             for i, r in df.iterrows()]

    kpw_9_16p = Portfolio('KPW.9.16p', a)
    kpw_9_16p.update(log2=8, bs=1, remove_fuzz=True)
    qd(kpw_9_16p)
    mv(kpw_9_16p)
    appx = kpw_9_16p.approximate('all')
    premium = 1.45 * kpw_9_16p.agg_m
    ans = {k: v.sf(premium) for k, v in appx.items()}
    ans['FFT'] = kpw_9_16p.sf(premium)
    qd(pd.DataFrame(ans.values(),
                    index=pd.Index(ans.keys(), name='method'),
                    columns=['premium']).sort_values('premium'),
      accuracy=5)


Here is a sample from the distribution and the mean-matched compound Poisson (for Exercise 9.18). The latter ``dsev`` clause works because all the benefit amounts are different. The temporary variable ``sev`` creates the severity curve. The log pmf graph reflects the irregular benefit amounts. Compare the cdf under ``comp Po`` with Table 9.17.

.. ipython:: python
    :okwarning:

    sev = df[['Benefit', 'q']]
    sev.q = sev.q / sev.q.sum()
    sev = sev.sort_values('Benefit')
    kpw_9_16cp = build('agg kpw_9_16.po '
                       f'{df.q.sum()} claims '
                       f'dsev {sev.Benefit.values /  1000} {sev.q.values} '
                       'poisson', bs=1, log2=10)
    qd(kpw_9_16cp)
    bit = pd.concat((kpw_9_16p.density_df.query('p_total > 0')[['p_total', 'F', 'S']],
                     kpw_9_16cp.density_df.query('p_total > 0')[['p_total', 'F', 'S']]),
                    keys=['exact', 'comp Po'], axis=1)
    bit.index = [f'{i:.0f}' for i in bit.index]
    bit.index.name = 'loss'
    with pd.option_context('display.max_rows', 360, 'display.max_columns', 10,
                           'display.width', 150, 'display.float_format', lambda x: f'{x:.7g}'):
        print(bit)
    fig, axs = plt.subplots(1,2, figsize=(3.5*2, 2.45), constrained_layout=True, squeeze=True)
    ax0, ax1 = axs.flat
    bit[('exact', 'p_total')].plot(marker='.', lw=.25, logy=True, ax=ax0, label='Portfolio');
    bit[('comp Po', 'p_total')].plot(marker='.', markerfacecolor='None', lw=.25, logy=True, ax=ax0, label='compound Po');
    (1-bit[('exact', 'p_total')].cumsum()).plot(ax=ax1);
    (1-bit[('comp Po', 'p_total')].cumsum()).plot(ax=ax1);
    ax0.legend();
    ax0.set(ylabel='log pmf');
    @savefig kpw_9_16.png
    ax1.set(ylabel='survival function');

.. _exercise 9_73:

**Exercise 9.73.**

An insurance company sold one-year term life insurance on a group of 2,300 independent lives as given in the next table.

.. math::
    \small
    \begin{matrix}
    \begin{array}{lrrr} \hline
    {} &   \text{Benefit} &         q &  \text{Number} \\
    \text{Class} &           &           &         \\ \hline
    1     &   100,000 &  0.1 &     500 \\
    2     &   200,000 & 0.02 &     500 \\
    3     &   300,000 & 0.02 &     500 \\
    4     &   200,000 &  0.1 &     300 \\
    5     &   200,000 &  0.1 &     500 \\ \hline
    \end{array}
    \end{matrix}

The insurance company reinsures amounts in excess of 100,000 on each life. The reinsurer wishes to charge a premium that is sufficient to guarantee that it will lose money 5% of the time on such groups. Obtain the appropriate premium by each of the following ways:

#. Using a normal approximation to the aggregate claims distribution.
#. Using a lognormal approximation.
#. Using a gamma approximation.
#. Using the compound Poisson approximation that matches the means.

In order to make the answer self-contained, the code below includes the data munging to re-create the table, pasted from a pdf.

.. ipython:: python
    :okwarning:

    data = '''1
    100,000
    0.10
    500
    2
    200,000
    0.02
    500
    3
    300,000
    0.02
    500
    4
    200,000
    0.10
    300
    5
    200,000
    0.10
    500'''
    sdata = data.split('\n')
    df = pd.DataFrame(zip(*[iter(sdata)]*4),
                      columns=['Class', 'Benefit', 'q', 'Number'])
    df.Benefit = df.Benefit.str.replace(',','').astype(float)
    df.q = df.q.astype(float)
    df.Number = df.Number.astype(int)
    df = df.set_index('Class')
    qd(df)


Next, build the exact solution for the gross book as a :class:`Portfolio` (extra credit).

.. ipython:: python
    :okwarning:

    a = [build(f'agg Class.{i} {r.q * r.Number} claims '
               f'dsev [{r.Benefit / 100000}] '
               f'binomial {r.q}')
             for i, r in df.iterrows()]

    p = Portfolio('KPW.9.73p', a)
    p.update(log2=10, bs=1, remove_fuzz=True)
    qd(p)

Build the reinsurer's loss distribution exactly, as ``p_ceded``, a :class:`Portfolio`, and the compound Poisson approximation ``cp_ceded``, an :class:`Aggregate`. The temporary variable ``bit`` is used to calculate the mixed severity distribution.

.. ipython:: python
    :okwarning:

    a_ceded = [build(f'agg Class.{i}.c {r.q * r.Number} claims '
               f'dsev [{r.Benefit / 100000 - 1}] '
               f'binomial {r.q}')
             for i, r in df.query('Benefit > 100000').iterrows()]

    p_ceded = Portfolio('KPW.9.73pc', a_ceded)
    p_ceded.update(log2=10, bs=1, remove_fuzz=True)
    qd(p_ceded)

    bit = df.query('Benefit > 100000')
    bit['Claims'] = bit.q * bit.Number
    bit.groupby('Benefit').Claims.sum()
    cp_ceded = build('agg CP.Approx '
                     f'{bit.Claims.sum()} claims '
                     f'dsev [1 2] [0.9 0.1] '
                     'poisson')
    qd(cp_ceded)

Compute the various estimated premiums, the 95%iles of the aggregate loss distribution.

.. ipython:: python
    :okwarning:

    prem_confidence = 0.95
    appx = p_ceded.approximate('all')
    ans = {k: v.ppf(prem_confidence) for k, v in appx.items()}
    ans['FFT'] = p_ceded.q(prem_confidence)
    ans['Comp Po'] = cp_ceded.q(prem_confidence)
    qd(pd.DataFrame(ans.values(),
                    index=pd.Index(ans.keys(), name='method'),
                    columns=['premium']).sort_values('premium'),
        accuracy=5)

.. _exercise 9_74:

**Exercise 9.74.** A group insurance contract covers 1,000 employees. An employee can have at most one claim per year. For 500 employees, there is a 0.02 probability of a claim, and when there is a claim, the amount has an exponential distribution with mean 500. For 250 other employees, there is a 0.03 probability of a claim and amounts are exponential with mean 750. For the remaining 250 employees, the probability is 0.04 and the mean is 1,000. Determine the exact mean and variance of total claims payments. Next, construct a compound Poisson model with the same mean and determine the variance of this model.

.. ipython:: python
    :okwarning:

    kpw_9_74p = build('port KPW.9.74p '
                      'agg A 10. claims sev  500 * expon binomial 0.02 '
                      'agg B 7.5 claims sev  750 * expon binomial 0.03 '
                      'agg C 10. claims sev 1000 * expon binomial 0.04 ')
    qd(kpw_9_74p)
    mv(kpw_9_74p)

Compound Poisson approximation is easy to construct as a mixture.

.. ipython:: python
    :okwarning:

    kpw_9_74cp = build('agg KPW.9.74.cp [10 7.5 10] claims sev [500 750 1000] * expon poisson')
    qd(kpw_9_74cp)
    mv(kpw_9_74cp)



.. ipython:: python
    :suppress:

    plt.close('all')
