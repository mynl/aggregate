Bahnemann Monograph
--------------------

Contents
~~~~~~~~~

Chapter 4: Aggregate Claims

* :ref:`Example 4.1 Simple Discrete-Discrete Aggregate <example 4_1>`
* :ref:`Example 4.2, Poisson-Gamma (Tweedie) Aggregate <example 4_2>`
* :ref:`Example 4.3, Tweedie Approximations <example 4_3>`
* :ref:`Example 4.4, Poisson-Discrete Approximation <example 4_4>`
* :ref:`Example 4.5, Poisson-Gamma Approximation <example 4_5>`
* :ref:`Example 4.15, Poisson-Lognormal With Policy Limit <example 4_15>`
* :ref:`Problems 4.7 and 13, Poisson-Gamma Distribution and Approximations <problem 4_7 and 4_13>`
* :ref:`Example 5.13, Poisson-Lognormal Layer Statistics <example 5_13>`
* :ref:`Example 6.3, Lognormal Increased Limits Factors (ILFs) <example 6_3>`
* :ref:`Example 6.4, Layer Premium <example 6_4>`
* :ref:`Example 6.5, Risk Loads <example 6_5>`
* :ref:`Example 6.6, Aggregate Premiums <example 6_6>`
* :ref:`Example 6.7, Deductible Credits <example 6_7>`
* :ref:`Summary of Created aggregate objects <bahn summary>`


Simple Discrete Aggregate, Example 4.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assume that n = 0, 1, 2 are the only possible numbers of
claims and they occur with probabilities 0.6, 0.3 and 0.1, and that there exist just three potential claim sizes: 1, 2, and 3 with probabilities 0.4, 0.5 and 0.1. (Note: the text uses claim sizes 100, 200 and 300.) Compute the distribution of possible outcomes and its mean and variance.

Imports and a convenience function.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd, mv
    import matplotlib.pyplot as plt

Build the aggregate and display key statistics.

.. ipython:: python
    :okwarning:

    a = build('agg Bahn.4.1 dfreq[0 1 2][.6 .3 .1] '
              'dsev[1 2 3][.4 .5 .1]')
    qd(a)
    mv(a)

Display all possible outcomes. Compare with the table on p. 107.

.. ipython:: python
    :okwarning:

    qd(a.density_df.query('p_total > 0') [['p_total', 'F']])


.. _example 4_2:

Poisson-Gamma (Tweedie) Aggregate, Example 4.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The text considers a Tweedie with expected claim count :math:`\lambda=2.5` and gammma shape 3 and scale 400. It computes the mean, variance and skewness, and uses the series expansion for the distribution to compute the CDF at various points (Table 4.1). These results can be replicated as follows.

.. ipython:: python
    :okwarning:

    a = build('agg Bahn.4.2 2.5 claims '
              'sev 400 * gamma 3 poisson')
    qd(a.describe)
    mv(a)

Extract various points of the pmf, cdf, and sf. The adjustment to the index is cosmetic. ``aggregate`` returns the entire distribution. The left plot shows the mixed density, with a mass at zero; right shows the cdf.

.. ipython:: python
    :okwarning:

    bit = a.density_df.loc[
        sorted(np.hstack((500, np.arange(0, 10000.5, 1000)))),
        ['p', 'F', 'S']]
    qd(bit, accuracy=4)
    fig, axs = plt.subplots(1, 2, figsize=(3.5*2, 2.45), constrained_layout=True, squeeze=True)
    ax0, ax1 = axs.flat
    (a.density_df.p / a.bs).plot(ylim=[0, 0.0002], xlim=[-100, 10000], lw=2, ax=ax0)
    ax0.set(title='Density')
    a.density_df.F.plot(ylim=[-0.05, 1.05], xlim=[-100, 10000], lw=2, ax=ax1)
    @savefig bahn1.png
    ax0.set(title='Mixed density');
    ax1.set(title='Distribution function');


.. _example 4_3:

Approximations to the Tweedie, Example 4.3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``aggregate`` largely circumvents the need for approximations, but it does support their creation. The following reproduces Table 4.3.


.. ipython:: python
    :okwarning:

    fz = a.approximate('all')
    bit['Normal'] = fz['norm'].cdf(bit.index)
    bit['Norm err'] = bit.Normal / bit.F - 1
    bit['sGamma'] = fz['sgamma'].cdf(bit.index)
    bit['sGamma err'] = bit.sGamma / bit.F - 1
    qd(bit, accuracy=4)

Here is Table 4.4. The FFT overstates :math:`F(0)` because of discretization error.

.. ipython:: python
    :okwarning:

    a2 = build('agg Bahn.4.2b 10 claims '
               'sev 6000 * gamma 0.05 poisson')
    qd(a2.describe)
    fz = a2.approximate('all')
    bit = a2.density_df.loc[
        sorted(np.hstack((500, np.arange(0, 20000, 2000)))),
        ['p', 'F', 'S']]
    bit['Normal'] = fz['norm'].cdf(bit.index)
    bit['Norm err'] = bit.Normal / bit.F - 1
    bit['sGamma'] = fz['sgamma'].cdf(bit.index)
    bit['sGamma err'] = bit.sGamma / bit.F - 1
    qd(bit, accuracy=4)

.. _example 4_4:

Poisson-Discrete Distribution, Example 4.4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The claim-count random variable is Poisson distributed with mean 1.75. Severity has a discrete distribution with outcomes 1, 2, 3, 4, 5 occurring with probabilities 0.2, 0.4, 0.2, 0.15, 0.05 respectively. Compute the aggregate distribution.

Here is Table 4.5.


.. ipython:: python
    :okwarning:

    a = build('agg Bahn.4.4 1.75 claims '
              'dsev [1 2 3 4 5] [.2 .4 .2 .15 .05] '
              'poisson')
    qd(a)
    qd(a.density_df.query('p > .001')[['p', 'F', 'S']], accuracy=4)


.. _example 4_5:

Poisson-Gamma Distribution, Example 4.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aggregate losses have Poisson frequency with mean 2.5 and gamma severity with shape 3 and scale 400. Hence the aggregate mean equals 1,200 and variance equals 480,000.
Now approximate the distribution function using FFT with a fine bucket size and the midpoint method for assigning claim-size probabilities and then ``bs=20`` and ``bs=100``.

Here is Table 4.6, comparing the distributions. The ``update`` method re-runs the FFT computation with different options, here altering ``bs``.

.. ipython:: python
    :okwarning:

    import numpy as np
    import pandas as pd
    a = build('agg Bahn.4.5 2.5 claims '
              'sev 400 * gamma 3 poisson')
    qd(a)
    xs = sorted(np.hstack((500, np.arange(0, 10001, 1000))))
    bit = a.density_df.loc[xs, ['F']]
    a.update(bs=100)
    bit100 = a.density_df.loc[xs,  ['F']]
    a.update(bs=20)
    bit20 = a.density_df.loc[xs,  ['F']]
    bit = pd.concat((bit, bit100, bit20), axis=1, keys=['h0.25', 'h100', 'h20'])
    bit[('h100', 'Rel Error')] = bit[('h100', 'F')] / bit[('h0.25', 'F')] - 1
    bit[('h20', 'Rel Error')] = bit[('h20', 'F')] / bit[('h0.25', 'F')] - 1
    bit = bit.sort_index(axis=1)
    qd(bit, accuracy=4)



.. _example 4_15:

Poisson-Lognormal Distribution With Limit, Example 4.15
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider an aggregate distribution with mean 3 Poisson frequency and lognormal claim size with parameters :math:`(\mu, \sigma) = (6, 1.5)`. Moreover, claim size is limited by a policy limit of 1,000. Graph the aggregate distribution.

The log density (left) shows the probability masses at outcomes consisting of only limit losses. The distribution (right) shows the corresponding jumps. Compare with Figure 4.4.

.. ipython:: python
    :okwarning:

    a = build('agg Bahn.4.15 '
              '3 claims '
              '1000 xs 0 '
              'sev exp(6) * lognorm 1.5 '
              'poisson')
    qd(a)

    fig, axs = plt.subplots(1, 2, figsize=(2*3.5, 2.45), constrained_layout=True)
    ax0, ax1 = axs.flat
    a.density_df.p.plot(ax=ax0, logy=True, label='FFT');
    a.density_df.F.plot(ax=ax1, label='FFT');
    ax0.set(ylabel='log density');
    ax0.set(ylabel='distribution', ylim=[0,1]);
    ax1.axvline(1000, c='C7', lw=.5);
    ax1.axvline(2000, c='C7', lw=.5);
    @savefig bahn4-15.png
    ax1.axvline(3000, c='C7', lw=.5);


.. _problem 4_7 and 4_13:

Poisson-Gamma Distribution and Approximations, Problems 4.7 and 13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An aggregate distribution has mean 8 Poisson frequency and gamma severity with shape 0.2 and scale 3750. Compute the distribution and compare with normal and shifted-gamma approximations.

.. ipython:: python
    :okwarning:

    a = build('agg Bahn.4.7 '
          '8 claims '
          'sev 3750 * gamma 0.2 '
          'poisson')
    qd(a)
    xs = np.arange(0, 30000,3000)
    qd(a.density_df.loc[xs, ['p', 'F','S']], accuracy=4)

``aggregate`` readily computes approximations and returns frozen ``scipy.stats`` objects.

.. ipython:: python
    :okwarning:

    fz = a.approximate('all')
    comp = pd.DataFrame({k: v.cdf(xs) for k, v in fz.items()}, index=xs)
    comp['agg'] = a.density_df.loc[xs, 'F',]
    comp.loc[:, [f'{k} err' for k in fz.keys()]] = comp.loc[:, fz.keys()].values / comp.loc[:, ['agg']].values - 1
    comp = comp.sort_index(axis=1)
    qd(comp, accuracy=4)

.. _example 5_13:

Poisson-Lognormal Layer Statistics, Example 5.13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider an aggregate distribution with mean 15 Poisson frequency and lognormal claim size with parameters :math:`(\mu, \sigma) = (5.9809, 1.8)`. What are the distribution characteristics for random variable S for claims in the layer 5,000 excess of 3,000?

The exact and FFT-estimated mean, cv, and skewness are reported in the ``describe`` dataframe, for frequency and severity. The values reported agree with the text, up to rounding.

 .. ipython:: python
    :okwarning:

    a = build('agg Bahn.5.13 '
          '15 claims 5000 xs 3000 '
          'sev exp(5.9809) * lognorm 1.8 ! '
          'poisson')
    qd(a)
    mv(a)

The exact severity can be accessed directly, as ``a.sevs[0].fz``, allowing us to compute the expected layer claim count. The aggregate can then be written in conditional form, producing the same statistics. The distribution function shows probability masses at multiples of the limit.

 .. ipython:: python
    :okwarning:

    xs = 15 * a.sevs[0].fz.sf(3000)
    print(f'excess claim count = {xs:.5f}')

    a = build('agg Bahn.5.13b '
              f'{xs} claims 5000 xs 3000 '
              'sev exp(5.9809) * lognorm 1.8 '
              'poisson')
    qd(a)
    fig, ax = plt.subplots(1,1,figsize=(3.5, 2.45))
    a.density_df.F.plot(ax=ax, label='FFT');
    fz = a.approximate('gamma')
    ax.plot(a.density_df.loss, fz.cdf(a.density_df.loss), c='C1', label='gamma approx.');
    ax.axvline(5000,  c='C7', lw=.5);
    ax.axvline(10000, c='C7', lw=.5);
    ax.axvline(15000, c='C7', lw=.5);
    ax.set(ylabel='cdf');
    @savefig bahn5-13.png
    ax.legend(loc='lower right');

.. _example 6_3:

Lognormal Increased Limits Factors (ILFs), Example 6.3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Indemnity losses for a portfolio of insurance policies have a lognormal claim-size distribution with parameters :math:`(\mu, \sigma) = (7, 2.4)`. The policy per-claim limit applies only to the indemnity portion of a claim, and the average per-claim loss adjustment expense is 2,200. Claim frequency for these policies is 0.0005 per exposure unit, and variable expenses equal 35% of premium.

A lognormal with :math:`\sigma = 2.4` has cv :math:`\sqrt{\exp(2.4^2)-1}=17.78` and is extremely thick-tailed, despite having moments of all orders. It is challenging to approximate numerically. Luckily, we only need to compute up to 5M. The ``aggregate`` parameters deliberately select a range that is too narrow for the entire distribution, but adequate for our purposes. Use ``log2=17`` and select ``bs`` greater than ``5e6 // 2**17 = 38``. We use ``bs=50``.
It is important to set ``normalize=False`` to avoid rescaling bucket probabilities to sum to one. These parameters are not a good model for the entire distribution; the mean error is too high.

The ``density_df`` dataframe includes limited expected values. Here is a sample.

.. ipython:: python
    :okwarning:

    a = build('agg Bahn.6.3 '
              '1 claim '
              'sev exp(7) * lognorm 2.4 '
              'fixed',
              bs=50, log2=17,
              normalize=False,
             )
    qd(a)
    xs = [1e5,  5e5, 7.5e5, 1e6, 2e6, 3e6, 4e6, 5e6]
    qd(a.density_df.loc[xs, ['F', 'S', 'lev']], accuracy=4)


The following reproduces Table 6.1. The ILF factors assume fixed (middle) and variable ALAE (right).

.. ipython:: python
    :okwarning:

    alae = 2200
    bit = a.density_df.loc[xs, ['lev']]
    bit['Fixed ALAE'] = (bit.lev + alae) / (bit.lev.iloc[0] + alae)
    bit['Prop ALAE'] = bit.lev / bit.lev.iloc[0]
    qd(bit, accuracy=4)


.. _example 6_4:

Layer Premium, Example 6.4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Continues :ref:`Example 6.3 <example 6_3>`.) Calculate the premium for successive excess layers of insurance for a policy with exposure equal 400. Use the ILFs under the assumption that the average per-claim ALAE payment is 2,200. Premium amounts for the successive million-dollar layers obtained from these layer factors applied to the basic-limit premium are displayed in Table 6.2 and reproduced below.

.. ipython:: python
    :okwarning:

    exposure = 400
    var_exp = 0.35
    frequency = 0.0005
    bit['Premium'] = exposure * frequency * (bit['lev'] + alae) / (1 - var_exp)
    bit['Layer Premium'] = np.diff(bit.Premium, prepend=0)
    qd(bit)


.. _example 6_5:

Risk Loads, Example 6.5
~~~~~~~~~~~~~~~~~~~~~~~~~

Example 6.5, computes risk loads as a percentage of standard deviation. ``aggregate`` can compute multiple limits at once, and the ``report_df`` dataframe returns individual severity and aggregate distribution statistics. The risk loads can be deduced from these. The risk load can be computed as ``k' * ex2`` or ``k * agg_cv`` (not shown).

The following code reproduces Table 6.3. First, define the controlling variables, and then set up the tower of limits within one object, using :doc:`2_x_vectorization`.

.. ipython:: python
    :okwarning:

    k_prime = 0.0277
    m = 400
    ϕ = 0.0005
    u = 0.2
    k = k_prime / np.sqrt(m * ϕ)

    limits = [1e5, 5e5, 1e6, 2e6, 3e6, 4e6, 5e6]
    bl = build('agg Bahn.6.5 '
               f'{m * ϕ} claims '
               f'{limits} xs 0 '
               'sev exp(7) * lognorm 2.4 '
               'poisson'
               , bs=50, log2=18)
    qd(bl.report_df.iloc[:, :-4], accuracy=4)

Next, extract the required columns from ``report_df`` and manipulate to compute the ILFs.

.. ipython:: python
    :okwarning:

    bit = bl.report_df.loc[['sev_m', 'sev_cv', 'agg_m', 'agg_cv']].iloc[:, :-4].T
    bit.index = limits
    bit.index.name = 'limit'
    bit['vx'] = (bit.sev_m * bit.sev_cv) ** 2
    bit['ex2'] = bit.vx + bit.sev_m**2
    bit['risk load'] = k_prime * bit.ex2 ** 0.5
    bit['lev'] = (1+u) * bit.sev_m
    bit['ILF w/o risk'] = bit['lev'] / bit.loc[100000, 'lev']
    bit['ILF with risk'] = (bit['lev'] + bit['risk load']) / (bit.loc[100000, 'lev'] + bit.loc[100000, 'risk load'])
    qd(bit, accuracy=4)


.. _example 6_6:

Aggregate Premiums, Example 6.6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Continues :ref:`Example 6.3 <example 6_3>`.) Compute expected losses across a variety of occurrence and aggregate limit combinations. Assume 20% ALAE outside the limits, expected claim count 1.2 with contagion parameter 0.1 (cv of mixing :math:`\sqrt{0.1}`), and lognormal severity :math:`(\mu, \sigma) = (7.6, 2.4)` (see errata).

The following code calculates Table 6.4 using FFT aggregate distributions. The last column, showing unlimited aggregate losses, agrees, but the other columns are slightly different because Bahnemann uses a shifted gamma approximation.

First, we compute all the aggregates.

.. ipython:: python
    :okwarning:

    b = {}
    for per_claim in [0.5e6, 1e6, 2e6, 3e6, 4e6, 5e6]:
        tower = np.array([0]  + [i for i in [0, 1e6, 2e6, 3e6, 4e6, 5e6, np.inf]
                if i >= per_claim])
        b[per_claim] = build('agg Bahn.6.6 1.2 claims '
               f'{per_claim} xs 0 '
               'sev exp(7.6) * lognorm 2.4 '
               f'mixed gamma {0.1}**.5 '
               f'aggregate ceded to tower {tower} '
               , bs=50, log2=18,
               normalize=False,
              )
    qd(pd.concat([i.describe[['E[X]', 'CV(X)', 'Skew(X)']] for i in b.values()],
        keys=b.keys(), names=['Occ limit', 'X']),
        accuracy=4)


Next, manipulate the output to determine layer loss costs using the ``reinsurance_audit_df`` dataframe. It tracks statistics for gross, ceded, and net loss across all requested layers, separately for occurrence and aggregate. In this case there are no occurrence layers. This step takes longer than computing the aggregates!

.. ipython:: python
    :okwarning:

    bit = pd.concat([i.reinsurance_audit_df['ceded'].iloc[:-1]
                    for i in b.values()], keys=b.keys(),
                    names=['Occ limit', 'kind', 'share', 'limit', 'attach'])
    bit['Agg limit'] = bit.index.get_level_values('limit') + bit.index.get_level_values('attach')
    bit = bit.droplevel(['kind', 'share', 'limit', 'attach'])
    bit = bit.set_index('Agg limit', append=True)
    bit = bit.groupby(level='Occ limit')[['ex']].cumsum()
    el = bit.unstack('Agg limit').droplevel(0, axis=1)
    table = pd.concat((el, el / el.loc[500000, np.inf]),
                      keys=['Loss', 'ILF'])
    qd(table.fillna(' - '), accuracy=4)

Here is a reconciliation to Table 6.4 of the 2M per claim and 2M aggregate limit expected loss, using the shifted gamma approximation. The limited aggregate loss is computed using the integral of the survival function ``fz.sf``.  ``quad`` is a general purpose numerical integration routine. It returns the integral and estimated error.

.. ipython:: python
    :okwarning:

    fz = b[2000000].approximate('sgamma')
    print(fz.stats())
    mv(b[2000000])
    from scipy.integrate import quad
    quad(fz.sf, 0, 2000000)


.. _example 6_7:

Deductible Credits, Example 6.7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Continues :ref:`Example 6.3 <example 6_3>`.) Consider a portfolio of policies for which the
ground-up indemnity claim size has a lognormal distribution with parameters
:math:`(\mu, \sigma) = (7.0, 2.4)` and allocated loss adjustment expense is 20% of the
indemnity amount. The basic limit is 100,000. Calculate the credit factors,
as well as the resulting frequency and severity, for six straight deductible options:
1,000; 2,000; 3,000; 4,000; 5,000; and 10,000. Base frequency equals 0.0005.

We can build all of the required distributions simultaneously using vectorization. Remember that the basic limit is ground up. The severity is unconditional, indicated by ``!`` at the end of the severity clause. The limit is eroded by the deductible.

.. ipython:: python
    :okwarning:

    deductibles = [0, 1e3, 2e3, 3e3, 4e3, 5e3, 10e3]
    limits = [100000 - i for i in deductibles]
    ϕ = 0.0005
    alae = 1.2
    bl = build('agg Bahn.6.7 '
               f'{ϕ} claims '
               f'{limits} xs {deductibles} '
               'sev exp(7) * lognorm 2.4 ! '
               'poisson'
               , bs=50, log2=18)
    qd(bl.report_df.iloc[:, :-4], accuracy=4)

Next, manipulate the ``report_df`` dataframe to compute the required quantities. The final exhibit replicates Table 6.5.

.. ipython:: python
    :okwarning:

    bit = bl.report_df.iloc[:, :-4].loc[['attachment', 'freq_m', 'sev_m', 'agg_m']].T
    bit = bit.rename(columns={'attachment': 'deductible'}).set_index('deductible')
    bit['F(d)'] = np.array([bl.sevs[0].fz.cdf(i) for i in bit.index])
    bit['freq_m'] = bit.loc[0, 'freq_m'] * (1 - bit['F(d)'])
    bit['E[X;d]'] = (bit.sev_m[0] - bit.sev_m)
    bit['C(d)'] = bit['E[X;d]'] / bit.sev_m[0]
    bit['sev_m'] = bit['sev_m'] / (1 - bit['F(d)']) * alae
    bit = bit.iloc[:, [-2, 3, -1, 0, 1]]
    bit['pure prem'] = bit.freq_m * bit.sev_m
    qd(bit, accuracy=4)


.. _bahn summary:

Summary
~~~~~~~~~~~

Here is a summary of all the objects created in this section.

.. ipython:: python
    :okwarning:

    build.qshow('^Bahn')
