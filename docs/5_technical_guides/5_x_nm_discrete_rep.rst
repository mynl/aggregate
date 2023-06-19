
.. _num how agg reps a dist:

Digital Representation of Distributions
----------------------------------------

    "We come now to reality. The truth is that the digital computer has totally defeated the analog computer. The input is a sequence of numbers and not a continuous function. The output is another sequence of numbers." :cite:p:`Strang1986am`

How ``aggregate`` Represents a Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. quote from index

``aggregate`` aims to deliver the speed and accuracy of parametric
distributions to aggregate probability distributions and make them
as easy to use as the lognormal. To achieve that, it needs a representation
of the underlying distribution amenable to computation.

There is no analytic expression for the cdf of most aggregate distributions. For example, there is no closed form expression for the distribution of the sum of two lognormals :cite:p:`Milevsky1998`. Therefore we must use a numerical approximation to the exact cdf.
There are two obvious ways to construct a numerical approximation to a cdf:

#. As a discrete (arithmetic, lattice) distribution supported on a discrete set of points.

#. As a continuous random variable with a piecewise linear distribution function.

The next two figures illustrate of the two approaches.
First, a discrete approximation, which results in a step-function, piecewise constant cdf, is shown left and the corresponding quantile function, right. The cdf is continuous from the right and the (lower) quantile function is continuous from the left. The distribution does not have a density function (pdf); it only has a probability mass function (pmf).

.. ipython:: python
    :okwarning:

    from aggregate.extensions.pir_figures import fig_4_5, fig_4_6
    @savefig num_discrete_approx.png scale=20
    fig_4_5()

Second, a piecewise linear continuous approximation, which results in a step-function pdf (not shown).

.. ipython:: python
    :okwarning:

    @savefig num_cts_approx.png scale=20
    fig_4_6()

The second approach assumes the aggregate has a continuous distribution, which is often not the case. For example, the Tweedie and all other compound Poisson distributions are mixed (they have a mass at zero). An aggregate whose severity has a limit will have a mass at multiples of the limit caused by the non-zero probability of limit-only claims. When :math:`X` is mixed it is impossible to distinguish the jump and continuous parts using a numerical approximation. The large jumps may be obvious but the small ones are not.

There are three other arguments in favor of discrete models. First, we live in a discrete world. Monetary amounts are multiples of a smallest unit: the penny, cent, yen, satoshi. Computers are inherently discrete. Second, probability theory is based on measure theory, which approximates distributions using simple functions that are piecewise constant. Third, the continuous model introduces unnecessary complexities in use, without any guaranteed gain in accuracy across all cases. See the complicated calculations in :cite:t:`Robertson1992`, for example.

For all of these reasons, we use a discrete approximation. Further, we assume that the distribution is known at integer multiples of a fixed bandwidth or bucket size. This assumption is forced by the use of FFTs and has some undesirable consequences. Ideally, we would use a stratified approach, sampling more points where the distribution changes shape and using larger gaps to capture the tail. However, the computational efficiency of FFTs make this a good trade-off.

Based on the above considerations, saying we have **computed an aggregate** means that we have a discrete approximation to its distribution function concentrated on integer multiples of a fixed bucket size :math:`b`. This specifies the approximation aggregate as

#. the value :math:`b` and
#. a vector of probabilities :math:`(p_0,p_1,\dots, p_{n-1})`

with the interpretation

.. math:: \Pr(X=kb)=p_k.

**All subsequent computations assume that the aggregate is approximated in this way.** There are several important consequences.

* The cdf is a step function with a jump of size :math:`p_k` at :math:`kb`.
* The cdf  is continuous from the right (it jumps up at :math:`kb`).
* The cdf be computed from the cumulative sum of :math:`(p_0,p_1,\dots, p_{n-1})`.
* The approximation has moments given by

  .. math:: \sum_k k^r p_i b.

* The limited expected value (the integral of the survival function), can be computed at the points :math:`kb` as :math:`b` times the cumulative sum of the survival function.
* The pdf, if it exists, can be approximated by :math:`p_i/b`.

All of these calculations are more straightforward than assuming a piecewise linear cdf.

.. _num robertson:

Sidebar: Continuous Discretization
"""""""""""""""""""""""""""""""""""""

  It is possible to *approximate* the continuous cdf approach in ``aggregate``. For example, the following code will reproduce the simple example in Section 4 of :cite:t:`Robertson1992`. Compare the output to his Table 4. Using ``bs=1/200`` approximates a continuous histogram. The use of a decimal bucket size is never recommended, but is used here to approximate Robertson's table values. We recommend against this approach. It is unnecessarily complicated and does not improve accuracy in any example we have encountered.

  .. ipython:: python
   :okwarning:

   from aggregate import build, qd

   s = build('agg Robertson '
             '5 claims '
             'sev chistogram xps [0 .2 .4 .6 .8 1] [.2 .2 .2 .2 .2] '
             'fixed'
             , bs=1/200, log2=12)
   qd(s.density_df.loc[0:6:40, ['F']], max_rows=100,
     float_format=lambda x: f'{x:.10f}')


.. _num discretization:

Discretizing the Severity Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section discusses ways to approximate a severity distribution with a discrete distribution. Severity distributions used by ``aggregate`` are supported on the non-negative real numbers; we allow a loss of zero, but not negative losses. However, the discretization process allows severity to be derived from a distribution supported on the whole real line---see the :ref:`note <num note>` below.

Let :math:`F` be a distribution function and :math:`q` the corresponding lower quantile function. It is convenient to be able to refer to a random variable with distribution :math:`F`, so let :math:`X=q(U)` where :math:`U(\omega)=\omega` is the standard uniform variable on the sample space :math:`\Omega=[0,1]`. :math:`X` has distribution :math:`F` :cite:p:`Follmer2016`.

We want  approximate  :math:`F` with a finite, purely discrete distribution supported at points :math:`x_k=kb`, :math:`k=0,1,\dots, m`, where :math:`b` is called the **bucket size** or the **bandwidth**. Split this problem into two: first create an infinite discretization on :math:`k=0,1,\dots`, and then truncate it.

The calculations described in this section are performed in :meth:`Aggregate.discretize`.

Infinite Discretization
""""""""""""""""""""""""""

There are four common methods to create an infinite discretization.

#. The **rounding** method assigns probability to the :math:`k` th bucket equal to

   .. math:: p_k &= \Pr((k - 1/2)b < X \le (k + 1/2)b) \\
                 &= F((k + 1/2)b) - F((k - 1/2)b) \\
             p_0 &= F(b/2).

#. The **forward** difference method assigns

   .. math:: p_k &= \Pr(kb < X \le (k+1)b ) \\
                 &= F((k + 1)b) - F(kb) \\
             p_0 &= F(b).

#. The **backward** difference method assigns

   .. math:: p_k &= \Pr((k-1)b < X \le kb ) \\
                 &= F(kb) - F((k - 1)b) \\
             p_0 &= F(0).

#. The **moment** difference method :cite:p:`LM` assigns

   .. math::

      p_k &= \frac{2\mathsf E[X \wedge kb] - \mathsf E[X \wedge (k-1)b] - \mathsf E[X \wedge (k+1)b]}{b} \\
      p_0 &= 1 - \frac{\mathsf E[X \wedge b]}{b}.

   The moment difference ensures the discretized distribution has the same first moment as the original distribution. This method can be extended to match more moments,  but the resulting weights are not guaranteed to be positive.


.. _num note:

.. note::
    Setting the first bucket to :math:`F(b/2)` for the rounding method (resp. :math:`F(b)`, :math:`F(0)`) allows the use of random variables with negative support. Any values :math:`\le 0` are included in the zero bucket. This behavior is useful because it allows the normal, Cauchy, and other similar distributions can be used as the basis for a severity.

Each of these methods produces a sequence :math:`p_k\ge 0` of probabilities that sum to 1 that can be interpreted as the pmf and distribution function :math:`F_b^{(d)}` of a discrete approximation random variable :math:`X_b^{(d)}`

.. math ::

    \Pr(X_b^{(d)} = kb) = p_k \\
    F_b^{(d)}(kb) = \sum_{i \le k} p_i

where superscript :math:`d=r,\ f,\ b,\ m` describes the discretization method and subscript :math:`b` the bucket size.

There is a disconnect between how the rounding method is defined and how it is interpreted. By definition, it corresponds to a distribution with jumps at :math:`(k+1/2)b`, not :math:`kb`. However, the approximation assumes the jumps are at :math:`kb` to simplify and harmonize subsequent calculations across the three discretization methods.

It is clear that :cite:p:`Embrechts2009a`

.. p. 499

.. math::

 &   F_b^{(b)}  \le F \le F_b^{(f)}  \\
 &   F_b^{(b)}  \le F_b^r \le F_b^{(f)}  \\
 &   X_b^{(b)}  \ge X \ge X_b^{(f)}  \\
 &   X_b^{(b)}  \ge X_b^r \ge X_b^{(f)}  \\
 &   X_b^{(b)} \uparrow X  \ \text{as}\  b\downarrow 0 \\
 &   X_b^{(f)} \downarrow X \ \text{as}\  b\downarrow 0

:math:`X_b`, :math:`X_r`, and :math:`X_f` converge weakly (in :math:`L^1`) to :math:`X` and the same holds for a compound distribution with severity :math:`X`. These inequalities are illustrated in the example below.

.. _num rounding default:

Rounding Method Used by Default
""""""""""""""""""""""""""""""""

``aggregate`` uses the **rounding** method by default and offers the forward and backwards methods to compute explicit bounds on the distribution approximation if required. We found that the rounding method performs well across all examples we have run.
These options are available in :meth:`update` through the ``sev_calc`` argument, which can take the values ``round``, ``forwards``, and ``backwards``.
This decision is based in part on the following observations about the moment method in :cite:t:`Embrechts2009a` (emphasis added):

   that both the forward/backward differences and the rounding method do not conserve any moments of the original distribution. In this light :cite:t:`Gerber1982` suggests a procedure that locally matches the first :math:`k` moments. Practically interesting is only the case :math:`k = 1`; for :math:`k \ge 2` the procedure is not well defined, potentially leading to negative probability mass on certain lattice points. The moment matching method is **much more involved than the rounding method** in terms of implementation; we need to calculate limited expected values. Apart from that, the **gain is rather modest**; moment matching only pays off for large bandwidths, and after all, **the rounding method is to be preferred**. This is further reinforced by the work of  :cite:t:`Grubel1999`: if the severity distribution is absolutely continuous with a sufficiently smooth density, the quantity :math:`f_{b,j} / b`, an approximation for the compound density, can be quadratically extrapolated.

.. LM on moment matching p. 182. careful here

:cite:t:`LM` report that :cite:t:`Panjer1983` found two moments were usually sufficient and that adding a third moment requirement adds only marginally to the accuracy. Furthermore, they report that the **rounding method and the first-moment method had similar errors**, while the second-moment method provided significant improvement but at the cost of no longer guaranteeing that the resulting probabilities are  **nonnegative**.

.. LM go on: The methods described here are qualitatively similar to numerical methods used to solve Volterra integral equations such as (9.26) developed in numerical analysis (see, e.g. Baker [10]).
  Ex 9.41 gives the formulas for weights in terms of LEVs.

Approximating the Density
"""""""""""""""""""""""""""
The pdf at $kb$ can be approximated as $p_k / b$. This suggests another approach to discretization. Using the rounding method

.. math::

    p_k &= F((k+1/2) b) - F((k- 1/2)b) \\
        &= \int_{(k-1/2)b}^{(k+1)b} f(x)dx \\
        &\approx f(kb) b.

Therefore we could rescale the vector :math:`(f(0), f(b), f(2b), \dots)` to have sum 1. This method works well for continuous distributions, but does not apply for mixed ones, e.g., when a policy limit applies.

Discretization Example
""""""""""""""""""""""""

.. note about negative needs to go elsewhere

This example illustrates the impact of different discretization methods on the severity and aggregate distributions. The example uses a severity that can take negative values. ``aggregate`` treats any negative values as a mass at zero. This approach allows for the use of the normal and other distributions supported on the whole real line. The severity has finite support, so truncation is not an issue, and it is discrete so it is easy to check the calculations are correct. The severity is shown first, discretized using ``bs=1, 1/2, 1/4, 1/8``. As expected, the rounding method (orange), lies between the forward (blue) and backwards (green) methods.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    dsev = [-1, 0, .25, .5, .75, 1, 1.5 + 1 / 16, 2, 2 + 1/4, 3]
    a01 = build(f'agg Num:01 1 claim dsev {dsev} fixed', update=False)
    fig, axs = plt.subplots(2, 2, figsize=(2 * 3.5, 2 * 2.45 + 0.1),
        constrained_layout=True)
    for bs, ax in zip([1, 1/2, 1/4, 1/8], axs.flat):
        for k in ['forward', 'round', 'backward']:
            a01.update(log2=10, bs=bs, sev_calc=k)
            a01.density_df.p_total.cumsum().\
                plot(xlim=[-.25, 3.25], lw=2 if  k=='round' else 1,
                drawstyle='steps-post', ls='--', label=k, ax=ax)
            ax.legend(loc='lower right')
            ax.set(title=f'Bucket size bs={bs}')
    axs[0,0].set(ylabel='distribution');
    axs[1,0].set(ylabel='distribution');
    @savefig num_ex1a.png scale=20
    fig.suptitle('Severity by discretization method for different bucket sizes');

Next, create aggregate distributions with a Poisson frequency,  mean 4 claims, shown for the same values of ``bs``.

.. ipython:: python
    :okwarning:

    a02 = build(f'agg Num:02 4 claims dsev {dsev} poisson', update=False)
    fig, axs = plt.subplots(2, 2, figsize=(2 * 3.5, 2 * 2.45 + 0.1),
        constrained_layout=True)
    for bs, ax in zip([1, 1/2, 1/4, 1/8], axs.flat):
        for k in ['forward', 'round', 'backward']:
            a02.update(log2=10, bs=bs, sev_calc=k)
            a02.density_df.p_total.cumsum().\
                plot(xlim=[-2, 27], lw=2 if  k=='round' else 1,
               drawstyle='steps-post', label=k, ax=ax)
            ax.legend(loc='lower right')
            ax.set(title=f'Bucket size bs={bs}')
    @savefig num_ex1b.png scale=20
    fig.suptitle('Aggregates by discretization method');


.. note::
    Setting ``drawstyle='steps-post'`` joins dots with a step function that jumps on the right (post=afterwards), making the result continuous from the right, appropriate for a distribution. Quantile functions are continuous from the left and should be rendered using  ``drawstyle='steps-pre'`` (before), which puts the jump on the left.

.. _num exact calculation:

Exact Calculation
""""""""""""""""""

The differences :math:`p_k=F((k + 1/2)b) - F((k - 1/2)b)` can be computed in three different ways, controlled by the ``discretization_calc`` option. The options are:

#. ``discretization_calc='distribution'`` takes differences of the sequence :math:`F((k + 1/2)b)`. This results in a potential loss of accuracy in the right tail where the distribution function increases to 1. The resulting probabilities can be no smaller than the smallest difference between 1 and a float. ``numpy`` reports this as ``numpy.finfo(float).epsneg``; it is of the order ``1e-16``.

#. ``discretization_calc='survival'`` takes the negative difference of the sequence :math:`S(k + 1/2)b)` of survival function values. This results in a potential loss of accuracy in the left tail where the survival function increases to 1. However, it provides better resolution in the right.

#. ``discretization_calc='both'`` attempts to make the best of both worlds, computing::

    np.maximum(np.diff(fz.cdf(adj_xs)), -np.diff(fz.sf(adj_xs)))

  This does double the work and is marginally slower.

The update default is ``survival``. The calculation method does not generally impact the aggregate distribution when FFTs are used because they compute to accuracy about ``1e-16`` (there is a 1 in each row and column of :math:`\mathsf F`, see :ref:`num fft`). However, the option can be helpful to create a pleasing graph of severity log density.

.. _num normalization:

Truncation and Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The discrete probabilities :math:`p_k` must be truncated into a finite-length vector to use in calculations. The number of buckets used is set by the ``log2`` variable, which inputs its base 2 logarithm. The default is ``log2=16`` corresponding to 65,536 buckets. There are two truncation options, controlled by the ``normalize`` variable.

#. ``normalize=False`` simply truncates, possibly resulting in a vector of probabilities that sums to less than 1.

#. ``normalize=True`` truncates and then normalizes, dividing the truncated vector by its sum, resulting in a vector of probabilities that does sums to 1 (approximately, see :ref:`floats <num floats>`).

The default is ``normalize=True``.

It is obviously desirable for the discrete probabilities to sum to 1. A third option, to put a mass at the maximum loss does not produce intuitive results---since the underlying distributions generally do not have a mass the graphs look wrong.

In general, it is best to use ``normalize=True`` in cases where the truncation error is immaterial, for example with a thin tailed severity. It is numerically cleaner and avoids issues with quantiles close to 1. When there will be an unavoidable truncation error, it is best to use  ``normalize=False``. The user needs to be aware that the extreme right tail is understated. The bucket size and number of buckets should be selected so that the tail is accurate where it is being relied upon. See :ref:`num error analysis` for more.

.. warning::
    Avoid using ``normalize=True`` for thick tail severities. It results in unreliable and hard to interpret estimated mean severity.

.. _num truncation example:

Truncation Example
"""""""""""""""""""

:cite:t:`Schaller2008` consider a Poisson-generalized Pareto model for operational risk. They assume an expected claim count equal to 18 and a generalized Pareto with shape 1, scale 12000 and location 7000. This distribution does not have a mean. They want to model the 90th percentile point. They compare using exponential tilting :cite:p:`Grubel1999` with padding, using up to 1 million ``log2=20`` buckets. They use a right-hand endpoint of 1 million on the severity. This example illustrates the impact of normalization and shows that padding and tilting have a similar effect.

Setup the base distribution without recomputing. Note infinite severity.

.. ipython:: python
   :okwarning:

   a = build('agg Schaller:Temnov '
             '18 claims '
             'sev 12000 * genpareto 1 + 7000 '
             'poisson'
             , update=False)
   qd(a)

Execute a variety of updates and assemble answer. Compare :cite:t:`Schaller2008`, Example 4.3.2, p. 197. They estimate the 90th percentile as 3,132,643. In this case, normalizing severity has a material impact; it acts to decrease the tail thickness and hence estimated percentiles.

.. ipython:: python
   :okwarning:

   import time
   import pandas as pd
   updates = {
       'a': dict(log2=17, bs=100, normalize=True, padding=0 , tilt_vector=None),
       'b': dict(log2=17, bs=100, normalize=False, padding=0, tilt_vector=None),
       'c': dict(log2=17, bs=100, normalize=False, padding=1, tilt_vector=None),
       'd': dict(log2=17, bs=100, normalize=False, padding=2, tilt_vector=None),
       'e': dict(log2=20, bs=25, normalize=True, padding=1 , tilt_vector=None),
       'f': dict(log2=20, bs=25, normalize=False, padding=1 , tilt_vector=None),
       'g': dict(log2=17, bs=100, normalize=False, padding=0, tilt_vector=20 / (1<<17))
       }
   ans = {}
   for k, v in updates.items():
       start_time_ns = time.time_ns()
       a.update(**v)
       end_time_ns = time.time_ns()
       ans[k] = [a.q(0.9), a.q(0.95), a.q(0.99),  (-start_time_ns + end_time_ns) / 1e6]
   df = pd.DataFrame(ans.values(), index=ans.keys(), columns=[.9, .95, .99, 'millisec'])
   for k, v in updates['a'].items():
       df[k] = [v[k] for v in updates.values()]
   df = df.replace(np.nan, 'None')
   df = df.set_index(['log2', 'bs', 'normalize', 'padding', 'tilt_vector'])
   df.columns.name = 'percentile'
   qd(df, float_format=lambda x: f'{x:12,.0f}', sparsify=False, col_space=4)


.. _num rec bucket:

Estimating the Bucket Size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

    Estimating the bucket size correctly is critical to obtaining accurate results from the FFT algorithm. This section is very important!

The bucket size is estimated as the :math:`p`-percentile of a moment matched fit to the aggregate. By default :math:`p=0.999`, but the user can selection another value using the ``recommend_p`` argument to ``update``.

On creation, :class:`Aggregate` automatically computes the theoretic mean, CV, and skewness :math:`\gamma` of the requested distribution. Using those values and :math:`p` the bucket size is estimated as follows.

#. If the CV is infinite the user must input :math:`b`. An ``ValueError`` is thrown if no value is provided. Without a standard deviation there is no way to gauge the scale of the distribution. Note that the CV is automatically infinite if the mean does not exist.
#. Else if the CV is finite and :math:`\gamma < 0`, fit a normal approximation (matching two moments). Most insurance applications have positive skewness.
#. Else if the CV is finite and :math:`0 < \gamma < \infty`, fit shifted lognormal and gamma distributions (matching three moments), and a normal distribution.
#. Else if the CV is finite but skewness is infinite, fit lognormal, gamma, and normal distributions (two moments).
#. Compute :math:`b'` as the greatest of any fit distribution :math:`p`-percentile (usually the lognormal).
#. If all severity components are limited, compute the maximum limit, :math:`m`, otherwise set :math:`m=0`.
#. Take :math:`b=\max(b', m)/n`, where :math:`n` is the number of buckets.
#. If :math:`b \ge 1` round up to 1, 2, 5, 10, 20, 100, 200, etc., and return. Else if :math:`b<1` return the smallest power of 2 greater than :math:`b` (e.g., 0.2 rounds to 0.25, 0.1 to 0.125).

Step 8 ensures that :math:`b \ge 1` is a reasonable looking round number and is an exact float when :math:`b \le 1`. The algorithm performs well in practice, though it can under-estimate :math:`b` for thick-tailed severities.
The user should always look at the diagnostics :ref:`10 min quick diagnostics`.

.. _num occ re and lp:

Occurrence Reinsurance and Loss Picks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If specific layer loss picks are selected, the adjustment occurs immediately after the gross severity is computed in Step 2.

Occurrence reinsurance is applied after loss pick adjustment and before step 3.

Aggregate reinsurance is applied at the very end of the algorithm.
