.. _2_x_samples:

.. NEEDS WORK

Working With Samples
====================

**Objectives:** How to sample from :mod:`aggregate` and how to a build a :class:`Portfolio` from a sample. Inducing correlation in a sample using the Iman-Conover algorithm and determining the worst-VaR rearrangement using the rearrangement algorithm.

**Audience:** Planning and strategy, ERM, capital modeling, risk management actuaries.

**Prerequisites:** DecL, aggregate distributions, risk measures.

**See also:** :doc:`../5_technical_guides/5_x_samples`,  :doc:`../5_technical_guides/5_x_iman_conover`, :doc:`../5_technical_guides/5_x_rearrangement_algorithm`.

**Contents:**

#. :ref:`Helpful References`
#. :ref:`samp samp`
#. :ref:`samp ic`
#. :ref:`samp ra`
#. :ref:`samp summary`

Helpful References
--------------------

* :cite:t:`PIR` chapter 14 and 15
* :cite:t:`Puccetti2012`
* :cite:t:`Conover1999`
* :cite:t:`Mildenhall2005a`
* Vitale IC proof in dependency book


.. See examples in /TELOS/Blog/agg/examples/IC_and_rearrangement.ipynb.

Samples and Densities
-----------------------

Use case: make realistic marginal distributions with ``aggregate`` that reflect the underlying frequency and severity (rather than defaulting to a lognormal determined by a CV assumption) and then use a sample in your simulation model.

.. _samp samp:

Samples from :mod:`aggregate` Object
-------------------------------------

The method :meth:`sample` draws a sample from an :class:`Aggregate` or :class:`Portfolio`
class object. Both cases work by applying ``pandas.DataFrame.sample`` to the object's ``density_df`` dataframe.

**Examples.**

1. A sample from an :class:`Aggregate`. Set up a simple lognormal distribution, modeled as an aggregate with trivial frequency.

  .. ipython:: python
    :okwarning:

    from aggregate import build, qd
    a01 = build('agg Samp:01 '
              '1 claim '
              'sev lognorm 10 cv .4 '
              'fixed'
             , bs=1/512)
    qd(a01)

  Apply :meth:`sample` and display the results.

  .. ipython:: python
    :okwarning:

    df = a01.sample(10**5, random_state=102)
    fc = lambda x: f'{x:8.2f}'
    qd(df.head(), float_format=fc)

  The sample histogram and the computed pmf are close. The pmf is adjusted to
  the resolution of the histogram.

  .. ipython:: python
    :okwarning:

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.45), constrained_layout=True)
    xm = a01.q(0.999)
    df.hist(bins=np.arange(xm), ec='w', lw=.25, density=True,
        ax=ax, grid=False);
    (a01.density_df.loc[:xm, 'p_total'] / a01.bs).plot(ax=ax);
    @savefig samp_agg_hist.png scale=20
    ax.set(title='Sample and aggregate pmf', ylabel='pmf');


2. A sample from a :class:`Portfolio` produces a multivariate distribution. Setup a simple :class:`Portfolio` with three lognormal marginals.

  .. ipython:: python
    :okwarning:

    from aggregate.utilities import qdp
    from pandas.plotting import scatter_matrix
    p02 = build('port Samp:02 '
            'agg A 1 claim sev lognorm 10 cv .2 fixed '
            'agg B 1 claim sev lognorm 15 cv .5 fixed '
            'agg C 1 claim sev lognorm  5 cv .8 fixed '
           , bs=1/128)
    qd(p02)

  Apply :meth:`sample` to produce a sample with no correlation. Here are the first few values.

  .. ipython:: python
    :okwarning:

    df = p02.sample(10**4, random_state=101)
    qd(df.head(), float_format=fc)

  :meth:`qdp` prints the pandas ``describe`` statistics dataframe for a dataframe, and adds the CV.

  .. ipython:: python
    :okwarning:

    qdp(df)

  The sample is independent, with correlations close to zero, as expected.

  .. ipython:: python
    :okwarning:

    abc = ['A', 'B', 'C']
    qd(df[abc].corr())

  The scatterplot is consistent with independent marginals.

  .. ipython:: python
    :okwarning:

    @savefig sample_corr1.png scale=20
    scatter_matrix(df[abc], grid=False,
        figsize=(6, 6), diagonal='hist',
        hist_kwds={'density': True, 'bins': 25, 'lw': .25, 'ec': 'w'},
        s=1, marker='.');

3. Pass a correlation matrix to :meth:`sample` to draw a correlated sample. Correlation is induced using the Iman-Conover algorithm.

  The function :meth:`random_corr_matrix` creates a random correlation matrix using vines. The second parameter controls the average correlation. This example includes high positive correlation.

  .. ipython:: python
    :okwarning:

    from aggregate import random_corr_matrix
    rcm = random_corr_matrix(3, .6, True)
    rcm

  Re-sample with target correlation ``rcm``. The achieved correlation is reasonably close to the requested ``rcm``.

  .. ipython:: python
    :okwarning:

    df2 = p02.sample(10**4, random_state=102,
        desired_correlation=rcm)
    qd(df2.iloc[:, :3].corr('pearson'))

  The scatterplot now shows correlated marginals. The histograms are unchanged.

  .. ipython:: python
    :okwarning:

    df2['total'] = df2.sum(1)
    @savefig sample_corr2.png scale=20
    scatter_matrix(df2[abc], grid=False, figsize=(6, 6), diagonal='hist',
        hist_kwds={'density': True, 'bins': 25, 'lw': .25, 'ec': 'w'},
        s=1, marker='.');

  The sample uses a different random state and produces a different draw. Comparing ``qdp`` output is one way to see if 10000 simulations is adequate. In this case there is good agreement.

  .. ipython:: python
    :okwarning:

    qdp(df2)


.. _samp ic:

Applying the Iman-Conover Algorithm
---------------------------------------

The method :meth:`sample` automatically applies the Iman-Conover algorithm (described in :doc:`../5_technical_guides/5_x_iman_conover`). It is also easy to apply Iman-Conover to a dataframe using the method :meth:`aggregate.utilities.iman_conover`. It reorders the input dataframe to have the same rank correlation as a multivariate normal reference sample with the desired linear correlation. Optionally, a multivariate t-distribution can be used as the reference.

**Examples.**

Apply Iman-Conover to the sample ``df`` with target the correlation ``rcm``, reusing the variables created in the previous section. The achieved correlation is close to that requested, as shown in the last two blocks.

.. ipython:: python
    :okwarning:

    from aggregate import iman_conover
    import pandas as pd
    ans = iman_conover(df[abc], rcm, add_total=False)
    qd(pd.DataFrame(rcm, index=abc, columns=abc))
    qd(ans.corr())

Setting the argument ``dof`` uses a t-copula reference with ``dof`` degrees of freedom. The t-copula with low degrees of freedom can produce pinched multivariate distributions. Use with caution.

.. ipython:: python
    :okwarning:

    ans = iman_conover(df[abc], rcm, dof=2, add_total=False)
    qd(ans.corr())
    @savefig sample_corrt.png scale=20
    scatter_matrix(ans, grid=False, figsize=(6, 6), diagonal='hist',
        hist_kwds={'density': True, 'bins': 25, 'lw': .25, 'ec': 'w'},
        s=1, marker='.');

=====

See WP REF for ways to apply Iman-Conover with different reference distributions.

**Details.** Creating the independent scores for Iman-Conover is quite time consuming. They are cached for a given sample size. Second and subsequent calls are far quicker (an order of magnitude) than the first call.


.. _samp ra:

Applying the Re-Arrangement Algorithm
---------------------------------------

The method :meth:`rearrangement_algorithm_max_VaR` implements the re-arrangement algorithm described in :ref:`../5_technical_guides/5_x_rearrangement_algorithm`. It returns only the tail of the re-arrangement, since values below the requested percentile are irrelevant.

Apply to ``df`` and request 0.999-VaR. The marginals are the 10 largest values. The algorithm permutes them to balance large and small observations.

.. ipython:: python
    :okwarning:

    from aggregate import rearrangement_algorithm_max_VaR
    ans = rearrangement_algorithm_max_VaR(df.iloc[:, :3], .999)
    qd(ans, float_format=fc)

Here are the stand-alone ``sa`` VaRs by marginal, in total for ``df``, in total for the correlated ``df2``, and the re-arrangement solutions ``ra`` for a range of different percentiles. The column ``comon total`` shows VaR for the comonotonic sum of the marginals (which equals the largest TVaR and variance re-arrangement).

.. ipython:: python
    :okwarning:

    ps = [9000, 9500, 9900, 9960, 9990, 9999]

    sa = pd.concat([df[c].sort_values().reset_index(drop=True).iloc[ps] for c in df]
                    +[df2.rename(columns={'total':'corr total'})['corr total'].\
                      sort_values().reset_index(drop=True).iloc[ps]], axis=1)
    sa['comon total'] = sa[abc].sum(1)
    ra = pd.concat([rearrangement_algorithm_max_VaR(df.iloc[:, :3], p/10000).iloc[0]  for p in ps],
              axis=1, keys=ps).T
    exhibit = pd.concat([sa, ra], axis=1, keys=['stand-alone', 're-arrangement'])
    exhibit.index = [f'{x/10000:.2%}' for x in exhibit.index]
    exhibit.index.name = 'percentile'
    qd(exhibit, float_format=fc)

See also :ref:`ra worked example`.

.. _samp sample to portfolio:

Creating a :class:`Portfolio` From a Sample
---------------------------------------------

A :class:`Portfolio` can be created from an existing sample by passing in a dataframe rather than a list of aggregates. This approach is useful when another model has created the sample, but the user wants to access other ``aggregate`` functionality. Each marginal in the sample is created as a ``dsev`` with the sampled outcomes. The ``p_total`` column used to set scenario probabilities if its is input, otherwise each scenario is treated as equally likely. The :class:`Portfolio` ignores any the correlation structure of the sample; the marginals are treated as independent, but see :ref:`samp switcheroo` for a way around this assumption.

**Example.**

Create a simple discrete sample from a three unit portfolio.

.. ipython:: python
    :okwarning:

    sample = pd.DataFrame(
       {'A': [20, 22, 24, 6, 5, 6, 7, 8, 21, 3],
        'B': [20, 18, 16, 14, 12, 10, 8, 6, 4, 2],
        'C': [0, 0, 0, 0, 0, 0, 0, 0, 20, 40]})
    qd(sample)

Pass to :class:`Portfolio` to create with these marginals. In this case, treat the marginals as discrete and update with ``bs=1``.

.. ipython:: python
    :okwarning:

    from aggregate import Portfolio
    p03 = Portfolio('Samp:03', sample)
    p03.update(bs=1, log2=8)
    qd(p03)

The univariate statistics for each marginal are the same as the sample input, but because they added independently, the totals differ. The sample has negative correlation and a lower CV.

.. ipython:: python
    :okwarning:

    sample['total'] = sample.sum(1)
    qdp(sample)

The :class:`Portfolio` total is a convolution of the input marginals and includes all possible combinations added independently. The figure plots the distribution functions.

.. ipython:: python
    :okwarning:

    ax = p03.density_df.filter(regex='p_[ABCt]').cumsum().plot(
        drawstyle='steps-post', lw=1, figsize=(3.5, 2.45))
    ax.plot(np.hstack((0, sample.total.sort_values())), np.linspace(0, 1, 11),
        drawstyle='steps-post', lw=2, label='dependent');
    ax.set(xlim=[-2, 90]);
    @savefig samp_port_samp.png scale=20
    ax.legend(loc='lower right');


.. _samp switcheroo:

Using Samples and the Switcheroo Trick
---------------------------------------

:class:`Portfolio` objects created from a sample ignore the dependency structure; the ``aggregate`` convolution algorithm always assumes independence. It is highly desirable to retain the sample's dependency structure. Many calculations rely only on :math:`\mathsf E[X_i\mid X]` and not the input densities per se. Thus, we reflect dependency if we alter the values :math:`\mathsf E[X_i\mid X]` based on a sample and recompute everything that depends on them. The method :meth:`Portfolio.add_exa_sample` implements this idea.


**Example.**

``sample`` was chosen to have lots of ties - different ways of obtaining the same total outcome.

.. ipython:: python
    :okwarning:

    qd(sample)

Apply ``add_exa_sample`` to the ``sample`` dataframe and look at the outcomes with positive probability. When a total outcome can occur in multiple ways, ``exeqa_i`` gives the average value of unit ``i``.
The function is applied to a copy of the original :class:`Portfolio` object because it invalidates various internal states. The output dataframe is indexed by total loss. Notice that rows sum to the correct total.

.. ipython:: python
    :okwarning:

    p03sw = Portfolio('Samp:03sw', sample)
    p03sw.update(bs=1, log2=8)
    df = p03sw.add_exa_sample(sample)
    qd(df.query('p_total > 0').filter(regex='p_total|exeqa_[ABC]'))

Swap the ``density_df`` dataframe --- the **switcheroo trick**.

.. ipython:: python
    :okwarning:

    p03sw.density_df = df

See the function ``Portfolio.create_from_sample`` for a single step create from sample, update, add exa calc, and switcheroo.

Most :class:`Portfolio` spectral functions depend only on marginal conditional expectations. Applying these functions through ``p03sw`` reflects dependencies. Calibrate some distortions to a 15% return. The maximum loss is only 45, so use a 1-VaR, no default capital standard.

.. ipython:: python
    :okwarning:

    p03sw.calibrate_distortions(ROEs=[0.15], Ps=[1], strict='ordered');
    qd(p03sw.distortion_df)

Apply the PH and dual to the independent and dependent portfolios. Asset level 45 is the 0.861 percentile of the independent.

.. ipython:: python
    :okwarning:

    d1 = p03sw.dists['ph']; d2 = p03sw.dists['dual']
    for d in [d1, d2]:
        print(d.name)
        print('='*74)
        pr = p03.price(1, d)
        pr45 = p03.price(.861, d)
        prsw = p03sw.price(1, d)
        a = pd.concat((pr.df, pr45.df, prsw.df), keys=['pr', 'pr45', 'prsw'])
        qd(a, float_format=lambda x: f'{x:7.3f}')


.. There's a sneaky but effective way to add correlation. The idea is:

  * Make a portfolio with independent lines as usual
  * Pull a sample from each unit
  * Shuffle the sample to induce the correlation you want using Iman-Conover.
    You don't have to use a normal copula.
  * (Sneaky part): recompute :math:`\mathsf E[X_i \mid X]` functions with those
    from the sample.

  From there, you can compute everything you need to use the natural allocation
  because it works on the conditional expectations, not the actual sample. I
  call it the switcheroo operation.


.. _samp summary:

Summary of Objects Created by DecL
-------------------------------------

Objects created by :meth:`build` in this guide. Objects created directly by class constructors are not entered into the knowledge database.

.. ipython:: python
    :okwarning:
    :okexcept:

    from aggregate import pprint_ex
    for n, r in build.qlist('^Samp:').iterrows():
        pprint_ex(r.program, split=20)


.. ipython:: python
    :suppress:

    plt.close('all')
