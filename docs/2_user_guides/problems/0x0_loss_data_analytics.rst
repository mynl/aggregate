Loss Data Analytics Book
-----------------------------

Examples from Jed Frees' open source actuarial software. `Text <https://openacttexts.github.io/Loss-Data-Analytics/>`_ and `github source <https://github.com/OpenActTexts/Loss-Data-Analytics>`_ available on-line.

Contents
~~~~~~~~~~

* :ref:`Distribution examples <distribution examples>`

  - :ref:`Gamma <lda gamma>`
  - :ref:`Pareto <lda pareto>`
  - :ref:`Weibull <lda weibull>`

* :ref:`Mixture examples <mixture example>`
* :ref:`Coverage modifications: deductibles and limits <coverage modifications>`

  - :ref:`Deductible <lda deductible>`
  - :ref:`Limit <lda limits>`
  - :ref:`Limit and deductible <lda limits and deductible>`
  - :ref:`Reinsurance <lda reinsurance>`

* :ref:`Aggregate loss distributions <lda aggregate loss distributions>`

  - :ref:`Poisson-discrete <lda poisson discrete>`
  - :ref:`Discrete <lda discrete 532>`
  - :ref:`Geometric-discrete <lda geometric discrete>`
  - :ref:`Moments <lda moments 534>`
  - :ref:`Poisson-uniform <lda poisson uniform>`
  - :ref:`Geometric-discrete <lda geom discrete>`
  - :ref:`Zero-modified Poisson-Burr <lda zmpoisson burr>`
  - :ref:`Negative binomial <lda neg bin 555>`
  - :ref:`Poisson-exponential <lda poisson exponential>`

* :ref:`Portfolio management <portfolio management>`

  - :ref:`Discrete example <discrete example 1034>`
  - :ref:`Telecom example <telecom example>`


.. _distribution examples:

Distribution Examples
~~~~~~~~~~~~~~~~~~~~~~

.. _lda gamma:

Gamma distribution
"""""""""""""""""""

.. ipython:: python
    :okwarning:

    import scipy.stats as ss
    import numpy as np
    import matplotlib.pyplot as plt

    xs = np.linspace(0, 1000, 1001)

    fig, axs = plt.subplots(1, 2, figsize=(2 * 3.5, 2.45), constrained_layout=True, squeeze=True)
    ax0, ax1 = axs.flat

    for scale in [100, 150, 200, 250]:
        ax0.plot(xs, ss.gamma(2, scale=scale).pdf(xs), label=f'scale = {scale}')

    for shape in [2, 3, 4, 5]:
        ax1.plot(xs, ss.gamma(shape, scale=100).pdf(xs), label=f'shape = {shape}')

    @savefig lda_gamma.png scale=20
    for ax in axs.flat:
        ax.legend(loc='upper right')
        ax.set(ylabel='gamma density', xlabel='x')


.. _lda pareto:

Pareto distribution
""""""""""""""""""""""

.. ipython:: python
    :okwarning:

    xs = np.linspace(0, 3000, 3001)

    fig, axs = plt.subplots(1, 2, figsize=(2 * 3.5, 2.45), constrained_layout=True, squeeze=True)
    ax0, ax1 = axs.flat

    for scale in [2000, 2500, 3000, 3500]:
        ax0.plot(xs, ss.pareto(3, scale=scale, loc=-scale).pdf(xs), label=f'scale = {scale}')

    for shape in [1,2,3,4]:
        ax1.plot(xs, ss.pareto(shape, scale=2000, loc=-2000).pdf(xs), label=f'shape = {shape}')

    @savefig lda_pareto.png scale=20
    for ax in axs.flat:
        ax.legend(loc='upper right')
        ax.set(ylabel='Pareto density', xlabel='x')


.. _lda weibull:

Weibull distribution
"""""""""""""""""""""

``scipy.stats`` includes Weibull min (for positive :math:`x`) and Weibull max (for negative :math:`x`) distributions. We want the min version.

.. ipython:: python
    :okwarning:

    xs = np.linspace(0, 400, 401)

    fig, axs = plt.subplots(1, 2, figsize=(2 * 3.5, 2.45), constrained_layout=True, squeeze=True)

    ax0, ax1 = axs.flat

    for scale in [50, 100, 150, 200]:
        ax0.plot(xs, ss.weibull_min(3, scale=scale).pdf(xs), label=f'scale = {scale}')
    for shape in [1.5, 2, 2.5, 3]:
        ax1.plot(xs, ss.weibull_min(shape, scale=100).pdf(xs), label=f'shape = {shape}')
    @savefig lda_weibull.png scale=20
    for ax in axs.flat:
        ax.legend(loc='upper right')
        ax.set(ylabel='Weibull_min density', xlabel='x')

.. _mixture example:

Mixture Example (3.3.5)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. Link <https://openacttexts.github.io/Loss-Data-Analytics/ChapSeverity.html#MethodsCreation>`

A collection of insurance policies consists of two types. 25% of policies are Type 1 and 75% of policies are Type 2. For a policy of Type 1, the loss amount per year follows an exponential distribution with mean 200, and for a policy of Type 2, the loss amount per year follows a Pareto distribution with parameters :math:`\alpha=3` and :math:`\theta=200`. For a policy chosen at random from the entire collection of both types of policies, find the probability that the annual loss will be less than 100, and find the average loss.

**Solution.** The function ``pmv`` (print mean and variance) is a convenience.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd, mv
    def pmv(m, v):
        print(f'mean     = {m:.6g}\n'
              f'variance = {v:.7g}')

Create the :class:`Aggregate` object, display its ``describe`` dataframe and compare the cdf with the exact computation.

.. ipython:: python
    :okwarning:

    a = build('agg lda.3.3.5 '
              '1 claim '
              'sev [200 200] * [expon pareto] [1 3] + [0 -200] wts [.25 .75] '
              'fixed',
              normalize=False)
    qd(a)
    a.sev.cdf(100), 0.25 * (1 - np.exp(-0.5)) + 0.75 * (1 - (2/3)**3)

This example has a very thick tailed severity and it is best to specify ``normalized=False`` for the most accurate severity estimates. With default settings,
``aggregate`` suffers considerable discretization error, with an estimated mean well below the actual 125. The ``sev.cdf`` method exposes the actual underlying
severity distribution cdf functions and reproduces the requested probability exactly. The object cdf function relies on the discretization and so is shifted by
half a bucket size. (Also available: ``sev.sf`` and ``sev.pdf``.)

.. ipython:: python

    a.cdf(100), a.sev.cdf(100 + a.bs/2)

.. _coverage modifications:

Coverage Modifications
~~~~~~~~~~~~~~~~~~~~~~

.. _lda deductible:

**Deductible Example (3.4.1)**

A claim severity distribution is exponential with mean 1000. An insurance company will pay the amount of each claim in excess of a deductible of 100. Calculate the variance of the amount paid by the insurance company for one claim, including the possibility that the amount paid is 0.

**Solution.** In this case we must use unconditional severity to include the  possibility that the amount paid is 0. This is done by adding ``!`` at the end of the severity specification. The moments are computed exactly without updating.

.. ipython:: python
    :okwarning:

    import numpy as np
    a = build('agg lda.3.4.1 1 claim '
              'inf xs 100 sev 1000 * expon 1 ! '
              'fixed', update=False)
    qd(a)
    m = 1000 * np.exp(-0.1)
    mv(a)
    pmv(m, (2 * 1000**2 * np.exp(-0.1)) - m**2)

**Deductible Example (3.4.2)**

For an insurance:

-  Losses have a density function

   .. math::

      f_{X}\left( x \right) = \left\{ \begin{matrix}
       0.02x & 0 < x  < 10, \\
       0 & \text{elsewhere.} \\
       \end{matrix} \right.

-  The insurance has an ordinary deductible of 4 per loss.
-  :math:`Y^{P}` is the claim payment per payment random variable.

**Solution.** The trick here is to realize that :math:`X` is a beta variable with :math:`\alpha=2` and :math:`\beta=1`.

.. ipython:: python
    :okwarning:

    a = build('agg lda.3.4.2 1 claim 6 xs 4 sev 10 * beta 2 1 fixed')
    qd(a)
    mv(a)

.. _lda limits:

**Limit Example (3.4.4)**

Under a group insurance policy, an insurer agrees to pay 100% of the medical bills incurred during the year by employees of a small company, up to a maximum total of one million dollars. The total amount of bills incurred, :math:`X`, has *pdf*

.. math::
    f_{X}(x) = \left\{ \begin{matrix}
        \frac{x\left( 4 - x \right)}{9} & 0 < x < 3 \\
        0 & \text{elsewhere.} \\
        \end{matrix} \right.

where :math:`x` is measured in millions. Calculate the total amount, in millions of dollars, the insurer would expect to pay under this policy.

**Solution.** In this case the distribution has no obvious parametric form---though it is related to a beta. We can solve it in ``aggregate`` by using a custom empirical distribution.

.. ipython:: python
    :okwarning:

    xs = np.linspace(0, 4, 2**13, endpoint=False)
    F = np.where(xs<3,(xs * xs  * (2 - xs / 3)) / 9, 1)
    ps = np.diff(F, append=1)
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.45), constrained_layout=True, squeeze=True)
    @savefig lda_344.png scale=20
    ax.plot(xs, ps);

When the empirical distribution has many entries it is faster to build the ``Aggregate`` object directly, rather than use DecL. The moments of the severity and aggregate distribution are computed from the numerical approximation during creation. There is no need to update the object.

.. ipython:: python
    :okwarning:

    from aggregate import Aggregate

    a = Aggregate('Example', exp_en=1, sev_name='dhistogram', sev_xs=xs, sev_ps=ps,
                 exp_attachment=0, exp_limit=1, freq_name='fixed')
    print(a)

.. _lda limits and deductible:

**Limit and Deductible Example (3.4.5)**

The ground up loss random variable for a health insurance policy in 2006 is modeled with :math:`X`, a random variable with an exponential distribution having mean 1000. An insurance policy pays the loss above an ordinary deductible of 100, with a maximum annual payment of 500. The ground up loss random variable is expected to be 5% larger in 2007, but the insurance in 2007 has the same deductible and maximum payment as in 2006. Find the percentage increase in the expected cost per payment from 2006 to 2007.

**Solution.** Trend increases the ground-up severity distribution but not the limit and attachment. The calculation is performed exactly on creation; again, there is  no need to update the ``Aggregate`` object.

.. ipython:: python
    :okwarning:

    import pandas as pd

    a06 = build('agg X06 1 claim 500 xs 100 sev 1000 * expon fixed', update=False)
    a07 = build('agg X07 1 claim 500 xs 100 sev 1050 * expon fixed', update=False)
    ans = pd.concat((a06.describe, a07.describe), keys=['2006', '2007'])
    qd(ans)
    ans.iloc[5, 0] / ans.iloc[2, 0] - 1

.. _lda reinsurance:

**Reinsurance Example (3.4.6, modified)**

Losses arising in a certain portfolio have a two-parameter Pareto distribution with :math:`\alpha=5` and :math:`\theta=3,600`. A reinsurance arrangement has been made, under which (a) the reinsurer accepts 15% of losses up to :math:`u=5,000` and all amounts in excess of 5,000 and (b) the insurer pays for the remaining losses.

#.  Express the random variables for the reinsurer's and the insurer's payments as a function of :math:`X`, the portfolio losses.
#.  Calculate the mean amount paid on a single claim by the insurer.
#.  Calculate the standard deviation of the amount paid on a single claim by the insurer (retaining the 15% copayment).

**Solution.** The net position can be modeled as::

    agg insurer.net 1 claim
    sev 3600 * pareto 5 - 3600
    occurrence net of 0.15 so 5000 xs 0 and inf xs 5000
    fixed

but this involves the thick-tailed Pareto across its entire range. It is better to
recognize the severity is limited by the second excess layer and proceed as follows.

.. ipython:: python
    :okwarning:

    a = build('agg insurer.net 1 claim '
          '5000 xs 0 sev 3600 * pareto 5 - 3600 '
          'occurrence net of 0.15 so 5000 xs 0 '
          'fixed')
    qd(a)
    print('\n', a.agg_m, a.agg_sd)

.. _lda aggregate loss distributions:

Aggregate Loss Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _lda poisson discrete:

**Poisson/Discrete Example (5.3.1)**


The number of accidents follows a Poisson distribution with mean 12. Each accident generates 1, 2, or 3 claimants with probabilities 1/2, 1/3, and 1/6 respectively.

Calculate the variance in the total number of claimants.

**Solution.**

.. ipython:: python
    :okwarning:

    a = build('agg QU 12 claims dsev [1 2 3] [1/2 1/3 1/6] poisson')
    qd(a)
    mv(a)

As always, ``a`` contains the (exact) full distribution of outcomes. We could answer any question about it.

.. _lda discrete 532:

**Discrete Example (5.3.2)**

You are the producer of a television quiz show that gives cash prizes. The number of prizes, :math:`N`, and prize amount, :math:`X`, have the following distributions:

.. math::
    \small
    \begin{matrix}
    \begin{array}{ccccc}\hline
        n & \Pr(N=n) & & x & \Pr(X=x)\\ \hline
        1 & 0.8 & & 0 & 0.2 \\
        2 & 0.2 & & 100 & 0.7 \\
           &       & & 1000 & 0.1\\\hline
      \end{array}
    \end{matrix}

Your budget for prizes equals the expected aggregate cash prizes plus the standard deviation of aggregate cash prizes. Calculate your budget.

**Solution.** Just a matter of translating into DecL. No need to update the object.

.. ipython:: python
    :okwarning:

    a = build('agg lda.5.3.2 dfreq [1 2] [.8 .2] '
              'dsev [0 100 1000] [.2 .7 .1]', update=False)
    display(a)
    mv(a)
    a.agg_m + a.agg_sd

.. _lda geometric discrete:

**Geometric/Discrete Example (5.3.3 and 5.4.1)**

The number of claims in a period has a geometric distribution with mean :math:`4`. The amount of each claim :math:`X` follows :math:`\Pr(X=x) = 0.25, \ x=1,2,3,4`, i.e. a discrete uniform distribution on :math:`\{1,2,3,4\}`. The number of claims and the claim amounts are independent. Let :math:`S_N` denote the aggregate claim amount in the period. Calculate :math:`F_{S_N}(3)`.

**Solution.** We can compute the entire distribution. Here we show up to the 99th percentile. If the probability clause in ``dsev`` is omitted then all outcomes are treated as equally likely.

.. ipython:: python
    :okwarning:

    a = build('agg lda.5.3.3 4 claims dsev [1:4] geometric')
    qd(a, accuracy=4)
    b = a.density_df.loc[0:a.q(0.99), ['p_total', 'F']]
    b.index = b.index.astype(int)
    qd(b, accuracy=4)

.. _lda moments 534:

**Moments Example (5.3.4)**

You are given:

.. math::
    \small
    \begin{matrix}
      \begin{array}{ c | c  c }
        \hline
          & \text{Mean} & \text{Standard Deviation}\\ \hline
        \text{Number of Claims} & 8 & 3\\
        \text{Individual Losses} & 10,000 & 3,937\\
        \hline
      \end{array}
    \end{matrix}

As a benchmark, use the normal approximation to determine the probability that the aggregate loss will exceed 150% of the expected loss.

**Solution.** Use the ``MomentAggregator`` class to compute the moments of an aggregate from those of frequency and severity.


.. ipython:: python
    :okwarning:

    import scipy.stats as ss
    from aggregate import MomentAggregator
    mom = MomentAggregator.agg_from_fs2(8, 9, 10000, 3937**2)
    fz = ss.norm(loc=mom.ex, scale=mom.sd)
    mom['prob'] = fz.sf(1.5*mom.ex)
    qd(mom)

.. _lda poisson uniform:

**Poisson/Uniform Example (5.3.5 and 5.4.2)**

For an individual over 65:

#. The number of pharmacy claims is a Poisson random variable with mean 25.
#. The amount of each pharmacy claim is uniformly distributed between 5 and 95.
#. The amounts of the claims and the number of claims are mutually independent.

Estimate the probability that aggregate claims for this individual will exceed 2000 using the normal approximation.

**Solution.** Here is a close-to exact solution in addition to the normal approximation. Note that the uniform distribution has no shape parameter. The severity is made by shifting and scaling the base. Scaling is like multiplication and is applied before the location (addition) shift.

.. ipython:: python
    :okwarning:

    a = build('agg Pharma 25 claims sev 90 * uniform + 5 poisson')
    qd(a)

Here are the moments for the approximation. The ``approximate`` function returns a ``scipy.stats`` frozen normal object, which yields the approximation.

.. ipython:: python
    :okwarning:

    print(a.sf(2000), a.agg_m, a.agg_var)

    fz = a.approximate('norm')
    fz.sf(2000), a.sf(2000)

``approximate`` will also provide (shifted) gamma and lognormal fits.

.. ipython:: python
    :okwarning:

    approx = a.approximate('all')
    b = pd.DataFrame([[k, v.sf(2000)] for k, v in approx.items()],
                 columns=['approx', 'prob']).set_index('approx')
    b.loc['exact'] = a.sf(2000)
    b.sort_values('prob')

Here is a comparison of the FFT model with the normal approximation. Example 5.4.2 derives a similar probability using simulation.

.. ipython:: python
    :okwarning:

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.45), constrained_layout=True, squeeze=True)
    (a.density_df.p / a.bs).plot(label='Exact', ax=ax);
    ax.plot(a.xs, fz.pdf(a.xs), label='Normal approx');
    ax.set(xlim=[0, 3000], title='Normal approximation');
    @savefig lda_normal.png scale=20
    ax.legend(loc='upper right');

.. _lda geom discrete:

**Geometric/Discrete Example (5.3.6 and 5.3.7)**

In a given week, the number of projects that require you to work overtime has a geometric distribution with :math:`\beta=2`. For each project, the distribution of the number of overtime hours in the week, :math:`X`, is as follows:

.. math::
    \small
    \begin{matrix}
    \begin{array}{ccc} \hline
        x &  & f(x)\\ \hline
        5 &  & 0.2 \\
        10 & & 0.3 \\
        20 & & 0.5\\ \hline
      \end{array}
    \end{matrix}

The number of projects and the number of overtime hours are independent. You will get paid for overtime hours in excess of 15 hours in the week. Calculate the expected number of overtime hours for which you will get paid in the week.

**Solution.** This is a one-liner in ``aggregate``. Remember that aggregate reinsurance is specified after the frequency clause. The first column in the describe dataframe shows the analytic gross answer and the second the FFT-computed net.

.. ipython:: python
    :okwarning:

    a = build('agg Projects 2 claims '
              'dsev [5 10 20] [.2 .3 .5] geometric '
              'aggregate net of 15 xs 0')
    qd(a)

Example 5.3.7 uses a recursive calculation in steps of 5. We can replicate that using an aggregate tower. The ``reinsurance_audit_df`` provides ceded and net statistics by layer. Here we extract just the ceded part to get the excess (overtime).

.. ipython:: python
    :okwarning:

    a1 = build('agg Projects.1 2 claims '
              'dsev [5 10 20] [.2 .3 .5] geometric '
              'aggregate net of tower [0 5 10 15 inf]')
    b = a1.reinsurance_audit_df.xs('ceded', axis=1, level=0)
    b['cumul ex'] = b.ex[::-1].cumsum() - a.agg_m
    qd(b, accuracy=4)

.. _lda zmpoisson burr:

**Zero-Modified Poisson/Burr Example (5.5.4)**

Aggregate losses are modeled as follows:

#. The number of losses follows a zero-modified Poisson distribution with :math:`\lambda=3` and :math:`p_0^M = 0.5`.
#. The amount of each loss has a Burr distribution with :math:`\alpha=3, \theta=50, \gamma=1`.
#. There is a deductible of :math:`d=30` on each loss.
#. The number of losses and the amounts of the losses are mutually independent.

Calculate :math:`\mathsf{E}(N^P)` and :math:`\mathsf{Var}(N^P)`.

**Solution.**

.. todo::

    Implement ZT and ZM!

.. _lda neg bin 555:

**Negative Binomial Example (5.5.5 and 5.5.6 modified)**

A group dental policy has a negative binomial claim count distribution with mean 300 and variance 800. Ground-up severity is given by the following table:

.. math::
    \small
    \begin{matrix}
      \begin{array}{ c | c }
        \hline
          \text{Severity} & \text{Probability}\\ \hline
        40 & 0.25\\
        80 & 0.25\\
        120 & 0.25\\
        200 & 0.25\\
        \hline
      \end{array}
    \end{matrix}

You expect severity to increase 50% with no change in frequency. You decide to impose a per claim deductible of 100. Calculate the expected total claim payment :math:`S` after these changes. What is the variance of the total claim payment, :math:`\mathsf{Var}(S)`? (Modified:) Compare the aggregate distributions before and after the policy change.

**Solution.** A negative binomial with mean 300 and variance 800 has :math:`8/3 = 1 + 300c` and giving a mixing cv of :math:`\sqrt{c}=(5/900)^{0.5}=0.0745`.  Hence the aggregate program is

TODO: why is the answer not exact?

.. ipython:: python
    :okwarning:

    cv = ((8 / 3 - 1) / 300)**0.5
    a0 = build(f'agg Original 300 claims dsev [4 8 12 20] mixed gamma {cv}')
    a1 = build(f'agg Revised 300 claims inf xs 10 '
               f'sev dhistogram xps [{4*1.5} {8*1.5} {12*1.5} {20*1.5}] ! mixed gamma {cv}')
    qd(a0)
    qd(a1)
    mv(a0)
    mv(a1)

Here is a comparison of the two densities.

.. ipython:: python
    :okwarning:

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.45), constrained_layout=True, squeeze=True)
    a0.density_df.p_total.plot(ax=ax, label='Original');
    a1.density_df.p_total.plot(ax=ax, label='Adjusted');
    ax.set(xlim=[-10, 1.25 * a0.q(0.9999)]);
    @savefig lda_5_5_5.png scale=20
    ax.legend(loc='upper right');

.. _lda poisson exponential:

**Poisson/Exponential Coverage and Underwriting Modification (Example 5.5.7)**

A company insures a fleet of vehicles. Aggregate losses have a compound Poisson distribution. The expected number of losses is 20. Loss amounts, regardless of vehicle type, have exponential distribution with :math:`\theta=200`. To reduce the cost of the insurance, two modifications are to be made:

#. A certain type of vehicle will not be insured. It is estimated that this will reduce loss frequency by 20%.
#. A deductible of 100 per loss will be imposed.

Calculate the expected aggregate amount paid by the insurer after the modifications.

**Solution.** The ``!`` at the end of the severity clause indicates unconditional severity (including zero claims that fail to meet the deductible).

.. ipython:: python
    :okwarning:

    a = build(f'agg Auto {20 * 0.8} claims inf xs 100 sev 200 * expon ! poisson')
    qd(a)


If severity is conditional there are 16 claims in excess of the deductible, giving a much higher number. The mean severity is still 200 because of the exponential's memoryless property.

.. ipython:: python
    :okwarning:

    a = build(f'agg Auto {20 * 0.8} claims inf xs 100 sev 200 * expon poisson')
    qd(a)

.. _portfolio management:

Portfolio Management
~~~~~~~~~~~~~~~~~~~~~~~

.. _discrete example 1034:

**VaR for a Discrete Variable Example (10.3.4)**


Consider an insurance loss random variable with the following probability distribution:

.. math::
    \small
    \Pr[X=x] = \left\{
                      \begin{array}{ll}
                        0.75, & \text{for }x=1 \\
                        0.20, & \text{for }x=3 \\
                        0.05, & \text{for }x=4.
                      \end{array}
                    \right.

Determine the VaR at :math:`q = 0.6, 0.9, 0.95, 0.95001`.

**Solution.**

.. ipython:: python
    :okwarning:

    a = build('agg VaR 1 claim dsev [1 3 4] [.75 .2 .05] fixed')
    [a.q(i) for i in [.6, .9, .95, .9501]]

.. _telecom example:

**Multi-Unit Telecom Management Example (10.4.3.3)**

You are the Chief Risk Officer of a telecommunications firm. Your firm has several property and liability risks. We will consider:

- :math:`X_1`, buildings, modeled using a gamma distribution with mean 200 and scale parameter 100.
- :math:`X_2`, motor vehicles, modeled using a gamma distribution with mean 400 and scale parameter 200.
- :math:`X_3`, directors and executive officers risk, modeled using a Pareto distribution with mean 1000 and scale parameter 1000.
- :math:`X_4`, cyber risks, modeled using a Pareto distribution with mean 1000 and scale parameter 2000.

Denote the total risk as :math:`X = X_1 + X_2 + X_3 + X_4`. For simplicity, you assume that these risks are independent. (Later, we will consider the more complex case of dependence.)

To manage the risk, you seek some insurance protection. You wish to manage internally small building and motor vehicles amounts, up to :math:`M_1` and :math:`M_2`, respectively. You seek insurance to cover all other risks. Specifically, the insurer's portion is

.. math:: Y_{insurer} = (X_1 - M_1)_+ + (X_2 - M_2)_+ + X_3 + X_4,

so that your retained risk is :math:`Y_{retained}= X - Y_{insurer} = \min(X_1,M_1) +  \min(X_2,M_2)`. Using deductibles :math:`M_1=` 100 and :math:`M_2=` 200:

#. Determine the expected claim amount of (i) that retained, (ii) that accepted by the insurer, and (iii) the total overall amount.
#. Determine the 80th, 90th, 95th, and 99th percentiles for (i) that retained, (ii) that accepted by the insurer, and (iii) the total overall amount.
#. Compare the distributions by plotting the densities for (i) that retained, (ii) that accepted by the insurer, and (iii) the total overall amount.

**Solution.** Begin by figuring the gamma and Pareto parameters. For a gamma, the mean equals shape times scale, so shape equals 2 for building and motor. For a Pareto, the mean equals scale / (shape - 1), so shape equals 2 (no variance) for D&O and 3 for cyber (no third moment). We model the results using three :class:`Portfolio` objects, one for the retention, one for the insured amount, and one total. In each case the distribution gives total losses; the frequency component is trivial.

Since the insured and total aggregates have no variance it is hard to estimate an appropriate bucket size. The default method uses the standard deviation as a scale factor. We must use judgement (or trial and error), and select ``log2=18`` and ``bs=1`` to ensure there is enough "space". Checking the describe dataframe shows these values match the means well by unit and in total. The insured severity must be made unconditional.


.. ipython:: python
    :okwarning:

    from aggregate import build

    retained = build('''port retained
        agg building 1 claim 100 xs 0 sev 100 * gamma 2 fixed
        agg motor    1 claim 200 xs 0 sev 200 * gamma 2 fixed
    ''')
    qd(retained)

    insured = build('''port insured
        agg building 1 claim inf xs 100 sev 100 * gamma 2 ! fixed
        agg motor    1 claim inf xs 200 sev 200 * gamma 2 ! fixed
        agg d.and.o  1 claim            sev 1000 * pareto 2 - 1000 fixed
        agg cyber    1 claim            sev 2000 * pareto 3 - 2000 fixed
    ''', log2=18, bs=1)
    qd(insured)

    total = build('''port total
        agg building 1 claim sev 100 * gamma 2 fixed
        agg motor    1 claim sev 200 * gamma 2 fixed
        agg d.and.o  1 claim sev 1000 * pareto 2 - 1000 fixed
        agg cyber    1 claim sev 2000 * pareto 3 - 2000 fixed
    ''', log2=18, bs=1)
    qd(total)


The spacing in the agg programs is for clarity. We could also program using ``dfreq`` as ``agg motor dfreq [1] 100 xs 0 sev...``. Next, assemble the requested data elements.

.. ipython:: python
    :okwarning:

    pfs = [retained, insured, total]
    answers = pd.DataFrame(columns=['retained', 'insured', 'total'])
    answers.index.name = 'statistic'
    answers.loc['expected claim amount'] = [x.agg_m for x in pfs]
    for p in [.8, .9, .95, .99]:
        answers.loc[f'claim p_{p:.2f}'] = [x.q(p) for x in pfs]
    qd(answers)

Finally, plot the densities. Compared to the text plot, the FFT reveals a discontinuous distribution for retained loss, with a large mass at 300. This is clearer on the lower plots, which show the distribution functions.


.. ipython:: python
    :okwarning:

    fig, axs = plt.subplots(2, 3, figsize=(7.5, 3.5), constrained_layout=True, squeeze=True)
    xl = {}
    for ax, pf in zip(axs.flat, pfs):
        pf.density_df.p_total.plot(ax=ax)
        # compute and store a reasonable x range
        q = pf.q(0.99) * 1.1
        xl[hash(pf)] = [-q / 50, q]
        ax.set(xlim=xl[hash(pf)], title=pf.name.title() + ' density')
        if pf is retained:
            ax.set(ylabel='density')

    @savefig lda_10.png
    for ax, pf in zip(axs.flat[3:], pfs):
        pf.density_df.F.plot(ax=ax)
        ax.set(xlim=xl[hash(pf)], title=pf.name.title() + ' distribution', xlabel='loss')
        if pf is retained:
            ax.set(ylabel='density')

