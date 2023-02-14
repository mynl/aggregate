.. _2_x_fft_convolution:

Numerical Methods and FFT Convolution
=======================================

**Objectives:**  Describe the numerical representation and FFT convolution algorithm that underlie all computations in ``aggregate``.

**Audience:** User who wants to understand the computation algorithms, potential errors, options and parameters.

**Prerequisites:** Probability theory and aggregate distributions; complex numbers and matrix multiplication; numerical analysis, especially numerical integration.

**See also:**  :doc:`../2_User_Guides`.

**Contents:**

* :ref:`num hr`
* :ref:`num overview`
* :ref:`num how agg reps a dist`
* :ref:`num discretization`
* :ref:`num algo steps`
* :ref:`num errors`
* :ref:`num parameters`
* :ref:`num error analysis`
* :ref:`num swot`
* :ref:`num pricing`
* :ref:`num floats`
* :ref:`num fft`

.. _num hr:

Helpful References
--------------------

* :cite:t:`LM`
* `Loss Data Analytics <https://openacttexts.github.io/Loss-Data-Analytics/>`_
* :cite:t:`Gerber1982`
* :cite:t:`Buhlmann1984`
* :cite:t:`Embrechts1993`
* :cite:t:`WangS1998`
* :cite:t:`Grubel1999`
* :cite:t:`Mildenhall2005a`
* :cite:t:`Embrechts2009a`


**Text to go somewhere.**

For thick tailed lognormal variables, it is best to truncate the severity distribution. Truncation does not impact PML estimates below the probability of truncation.  We select a truncation of USD 20T, about the size of the US economy. The unlimited models suggest there is less than a 1 in 10,000 chance of a model so large.

The FFT method is a miraculous technique for computing aggregate
distributions. It is especially effective when the expected claim count
is relatively small and the underlying severity distribution is bounded.
These assumptions are true for many excess of loss reinsurance treaties,
for example. Thus the FFT is very useful when quoting excess layers with
annual aggregate deductibles or other variable features. The FFT
provides a discrete approximation to the moment generating function.



The next step is magic in actuarial science. Remember that if :math:`N`
is a :math:`G`-mixed Poisson and :math:`A=X_1+\cdots+X_N` is an
aggregate distribution then

.. math:: M_A(z)=M_G(n(M_X(z)-1)).

Using FFTs you can replace the *function* :math:`M_X` with the discrete
approximation *vector* :math:`\hat{\mathsf{x}}` and compute

.. math:: \hat{\mathsf{a}}=M_G(n(\hat{\mathsf{x}} -1))

component-by-component to get an approximation vector to the function
:math:`M_A`. You can then use the inverse FFT to recover an discrete
approximation :math:`\mathsf a` of :math:`A` from :math:`\hat{\mathsf{a}}`! See (big) Wang
for more details.

Similar tricks are possible in two dimensions—see Press et al.,
and Homer and Clark for a discussion.

The FFT allows us to use the following very simple method to
qualitatively approximate the density of an aggregate of dependent
marginals :math:`X_1,\dots,X_n` given a correlation matrix
:math:`\Sigma`. First use the FFT method to compute the sum :math:`S'`
of the :math:`X_i` as though they were independent. Let
:math:`\mathsf{Var}(S')=\sigma^{'2}` and let :math:`\sigma^2` be the variance of
the sum of the :math:`X_i` implied by :math:`\Sigma`.

Next use the FFT to add a further “noise” random variable :math:`N`
to :math:`S'` with mean zero and variance :math:`\sigma^2-\sigma^{'2}`. Two
obvious choices for the distribution of :math:`N` are normal or shifted
lognormal. Then :math:`S'+N` has the same mean and variance as the sum of the
dependent variables :math:`X_i`. The range of possible choices for :math:`N`
highlights once again that knowing the marginals and correlation structure is
not enough to determine the whole multivariate distribution. It is an
interesting question whether all possible choices of :math:`N` correspond to
actual multivariate structures for the
:math:`X_i` and conversely whether all multivariate structures
correspond to an :math:`N`. (It is easy to use MGFs to deconvolve
:math:`N` from the true sum using Fourier methods; the question is
whether the resulting “distribution” is non-negative.)

Heckman and Meyers used Fourier transforms to compute aggregate
distributions by numerically integrating the characteristic function.
Direct inversion of the Fourier transform is also possible using FFTs.
The application of FFTs is not completely straight forward because of
certain aspects of the approximations involved. The details are very
clearly explained in Menn and Rachev. Their method allows the use of
FFTs to determine densities for distributions which have analytic MGFs
but not densities—notably the class of stable distributions.


.. _num overview:

Overview
----------

Numerical analysts face a trilemma

   **Fast, Accurate, or Flexible? Pick two!**

Simulation is flexible, but trades-off speed against accuracy. Fewer simulations is faster but provides lower accuracy; many simulations improves accuracy at the cost of speed.

``aggregate`` is based on the fast Fourier transform (FFT) convolution algorithm. It delivers the third trilemma option: it is fast and accurate but less flexible than simulation.

There are many use-cases where ``aggregate`` is unbeatable. It an compute the distribution of

* Aggregate losses of a portfolio with a complex limit and attachment profile, and a mixed severity distributions.
* Ceded or net outcomes net for an occurrence reinsurance program.
* Ceded or net outcomes for reinsurance contracts with variable features (sliding commissions, swing rated programs, profit commissions, aggregate limits, see :doc:`../2_user_guides/2_x_re_pricing`.)
* Retained loss net of specific and aggregate insurance, as found in a risk-retention group, see :doc:`../2_user_guides/2_x_ir_pricing`, including exact computation of Table L and Table M charges in US worker compensation ratemaking.
* The sum of independent units (line of business, business unit, geographic unit etc.). Other methods are available for dependent units.
* The sum of dependent units, where the dependence structure is driven by :ref:`common frequency mixing variables <5_x_probability>`.

It is particularly well-suited to compute aggregates with thick-tailed severity and where accuracy is important, such as catastrophe risk PMLs, AEP, and OEP points. Outcomes with low expected loss rates which are hard to simulate with sufficient accuracy.

In Finance, FFT methods are established as the go-to choice solution for convolution REF.

The basic FFT algorithm used in ``aggregate`` tracks only one variable at a time. Thus, it cannot model joint distributions such as ceded and net loss or quantities derived from it such as the total cession to a specific and aggregate cover, or the cession to an occurrence program with limited reinstatements. Both of these require a bivariate distribution. (It *can* model the net position after specific and aggregate cessions, and ceded losses to an occurrence program with an aggregate limit.) The strengths and weaknesses of the algorithm are discussed further in :ref:`num swot`.


.. _num how agg reps a dist:

How ``aggregate`` Represents a Distribution
--------------------------------------------

.. quote from index

``aggregate`` delivers the speed and accuracy of parametric distributions to situations that usually require simulation, making it as easy to work with an aggregate (compound) probability distribution as the lognormal. It can create an exact or very accurate approximation to the cumulative distribution function (cdf) of extremely complicated aggregate distributions---opening the way to calculate the pdf or pmf, sf, and many other actuarial and risk theoretic functions based on the distribution. It must start with a representation of the underlying distribution that is amenable to computation.

There is no analytic expression for the cdf of most aggregate distributions. For example, there is no closed form expression for the distribution of the sum of two lognormals :cite:p:`Milevsky1998`. Therefore we must use a numerical approximation to the exact cdf.

There are two obvious ways to construct a numerical approximation to a cdf:

#. As a discrete (arithmetic, lattice) distribution supported on a discrete set of points.

#. As a continuous random variable with a piecewise linear distribution function.

The next two figures illustrate of the two approaches.
First, a discrete approximation, which results in a step-function approximation. The cdf is shown left and the corresponding quantile function right. The cdf is continuous from the right and the (lower) quantile function from the left. The distribution does not have a density function (pdf); it only has a probability mass function (pmf).

.. ipython:: python
    :okwarning:

    from aggregate.extensions.pir_figures import fig_4_5, fig_4_6
    @savefig num_discrete_approx.png scale=20
    fig_4_5()

Second, a piecewise linear continuous approximation. This distribution has a step-function pdf (not shown).

.. ipython:: python
    :okwarning:

    @savefig num_cts_approx.png scale=20
    fig_4_6()


The second approach assumes the aggregate has a continuous distribution, which is often not the case. For example, the Tweedie and all other compound Poisson distributions are mixed (they have a mass at zero). An aggregate whose severity has a limit has a mass at multiples of the limit, caused by the non-zero probability of limit-only claims. When :math:`X` is mixed it is impossible to distinguish the jump and continuous parts from  a numerical approximation. The large jumps may be obvious but the small ones are not.

.. _num robertson:

Sidebar
"""""""""

It is possible to approximate the continuous cdf approach in ``aggregate``. For example, the following code will reproduce the simple example in Section 4 of :cite:t:`Robertson1992`. Compare the output to his Table 4. Using ``bs=1/200`` approximates a continuous histogram. The use of a decimal bucket size is never recommended, but is used here to approximate Robertson's table values.

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

----

There are three other arguments in favor of discrete models. First, we live in a discrete world. Monetary amounts are multiples of a smallest unit: the penny, cent, yen, satoshi. Computers are inherently discrete. Second, probability theory is based on measure theory, which approximates distributions using simple functions that are discrete. Third, the continuous model introduces unnecessary complexities in use, without any guaranteed gain in accuracy across all cases. See the complicated calculations in :cite:t:`Robertson1992`, for example.


.. a version of the following is in 10 mins

For these reasons, we use a discrete numerical approximation. Further, we assume that the distribution is known at integer multiples of a fixed bandwidth or bucket size. This assumption is forced by the use of FFTs and has some undesirable consequences, as we shall see. Ideally, we would use a stratified approach, sampling more points where the distribution changes shape and using larger gaps to capture the tail. However, the computational efficiency of FFTs make this a good trade-off.

Based on the above considerations, saying we have "computed an aggregate" means that we have a discrete approximation to its distribution function concentrated on integer multiples of a fixed bucket size :math:`b`. This specifies the approximation aggregate as

#. the value :math:`b` and
#. a vector of probabilities :math:`(p_0,p_1,\dots, p_{n-1})`

with the interpretation

.. math:: \Pr(X=kb)=p_k.

**All subsequent computations assume that the aggregate is approximated in this way.** There are several important consequences. The cdf is a step function with a jump of size :math:`p_k` at :math:`kb`. It is continuous from the right (it jumps up at :math:`kb`). Its moments are simply

.. math:: \sum_k k^r p_i b.

It can be computed as the cumulative sum of :math:`(p_0,p_1,\dots, p_{n-1})`. The limited expected value (the integral of the survival function), can be computed at the points :math:`kb` as :math:`b` times the cumulative sum of the survival function. The pdf, if it exists, can be approximated by :math:`p_i/b`. And so forth. All of these calculations are more straightforward than assuming a piecewise linear cdf.

.. _num discretization:

Discretizing the Severity Distribution
-----------------------------------------

This section discusses ways to approximate a severity distribution with a discrete distribution. Severity distributions used by ``aggregate`` are supported on the non-negative real numbers; we allow a loss of zero, but not negative losses. However, the discretization process allows severity to be derived from a distribution supported on the whole real line---see the :ref:`note <num note>` below.

Let :math:`F` be a distribution function and :math:`q` the corresponding lower quantile function. It is convenient to be able to refer to a random variable with distribution :math:`F`, so let :math:`X=q(U)` where :math:`U(\omega)=\omega` is the standard uniform variable on the sample space :math:`\Omega=[0,1]`. :math:`X` has distribution :math:`F` :cite:p:`Follmer2016`

We want  approximate  :math:`F` with a finite, purely discrete distribution supported at points :math:`x_k=kb`, :math:`k=0,1,\dots, m`, where :math:`b` is called the **bucket size** or the **bandwidth**. Split this problem into two: first create an infinite discretization on :math:`k=0,1,\dots`, and then truncate it.

The calculations described in this section are performed in :meth:`Aggregate.discretize`.

Infinite Discretization
~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Each of these methods produces a sequence :math:`p_k\ge 0` of probabilities that sum to 1 and can be interpreted as the pmf and distribution function :math:`F_b^{(d)}` of a discrete approximation random variable :math:`X_b^{(d)}`

.. math ::

    \Pr(X_b^{(d)} = kb) = p_k \\
    F_b^{(d)}(kb) = \sum_{i \le k} p_i

where superscript :math:`d=r,\ f,\ b,\ m` describes the discretization method and subscript :math:`b` the bucket size.

There is a disconnect between how the rounding method is defined and how it is interpreted. By definition, it corresponds to a distribution with jumps at :math:`(k+1/2)b`, not :math:`kb`. However, the approximation takes the latter interpretation to simplify and harmonize subsequent calculations across the three discretization methods.

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

Rounding Method Used by Default
""""""""""""""""""""""""""""""""


``aggregate`` uses the **rounding** method by default and offers the forward and backwards methods to compute explicit bounds on the distribution approximation if required.
These options are available in :meth:`update` through the ``sev_calc`` argument, which can take the values ``round``, ``forwards``, and ``backwards``.
This decision is based on the following observations and other independent testing.

:cite:t:`Embrechts2009a` comment on the moment method (emphasis added)

   that both the forward/backward differences and the rounding method do not conserve any moments of the original distribution. In this light :cite:t:`Gerber1982` suggests a procedure that locally matches the first :math:`k` moments. Practically interesting is only the case :math:`k = 1`; for :math:`k \ge 2` the procedure is not well defined, potentially leading to negative probability mass on certain lattice points. The moment matching method is **much more involved than the rounding method** in terms of implementation; we need to calculate limited expected values. Apart from that, the **gain is rather modest**; moment matching only pays off for large bandwidths, and after all, **the rounding method is to be preferred**. This is further reinforced by the work of  :cite:t:`Grubel1999`: if the severity distribution is absolutely continuous with a sufficiently smooth density, the quantity :math:`f_{b,j} / b`, an approximation for the compound density, can be quadratically extrapolated.

.. LM on moment matching p. 182. careful here

:cite:t:`LM` report that :cite:t:`Panjer1983` found two moments were usually sufficient and that adding a third moment requirement adds only marginally to the accuracy. Furthermore, they report that the **rounding method and the first-moment method had similar errors**, while the second-moment method provided significant improvement but at the cost of no longer guaranteeing that the resulting probabilities are  **nonnegative**.

.. LM go on: The methods described here are qualitatively similar to numerical methods used to solve Volterra integral equations such as (9.26) developed in numerical analysis (see, e.g. Baker [10]).
  Ex 9.41 gives the formulas for weights in terms of LEVs.

Discretization Example
""""""""""""""""""""""""

.. note about negative needs to go elsewhere

This example illustrates the impact of different discretization methods on the severity and aggregate distributions. The example uses a severity that can take negative values. ``aggregate`` treats any negative values as a mass at zero. This approach allows for the use of the normal and other distributions supported on the whole real line. It has finite support, so truncation is not an issue. And it is discrete so it is easy to check the calculations are correct. The severity is shown first, discretized using ``bs=1, 1/2, 1/4, 1/8``. The orange, rounded method, lies between the blue forward and green backwards lines.

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

.. TODO Right place for this?

.. note::
    Setting ``drawstyle='steps-post'`` joins dots with a step function that jumps on the right, making the result continuous from the right, appropriate for a distribution. Quantile functions are continuous from the left and should be rendered using  ``drawstyle='steps-pre'``, which puts the jump on the left.


.. TODO Right place for this?

Implementation Details
""""""""""""""""""""""""

Severities specified discretely, using the ``dsev`` keyword are implemented using a ``scipy.stats`` ``rv_historgram`` object, which is actually continuous. They work by concentrating the probability in small intervals just to the left of each knot point (to make the function right continuous). Given::

    dsev [xs] [ps]

where ``xs`` and ``ps`` are the vectors of outcomes and probabilities, internally ``aggregate`` creates::

   xss = np.sort(np.hstack((xs - 2 ** -30, xs)))
   pss = np.vstack((ps1, np.zeros_like(ps1))).reshape((-1,), order='F')[:-1]
   fz_discr = ss.rv_histogram((pss, xss))

The value 1e-30 needs to be smaller than the bucket size resolution, i.e. small enough not to “split the bucket”. The mass is to the left of the knot to make a right continuous function (the approximation ramps up before the knot). Generally histograms are downsampled, not upsampled, so this is not a restriction.

A ``dsev`` statement is translated into the more general::

    sev dhistorgram xps [xs] [ps]

where ``dhistrogram`` creates a discrete histogram (as above) and the ``xps`` keyword prefixes inputting the knots and probabilities. It is also possible to specify the input severity as a continuous histogram that is uniform on :math:`(x_k, x_{k+1}]`. The discrete probabilities are :math:`p_k=P(x_k < X \le x_{k+1})`. To create a rv_histogram variable is much easier, just use::

    sev chistorgram xps [xs] [ps]

which is translated into::

    xs2 = np.hstack((xs, xs[-1] + xs[1]))
    fz_cts = ss.rv_histogram((ps2, xs2))

The code adds an additional knot at the end to create enough differences (there are only two differences between three points). The `Robertson example<num robertson>`_ uses a ``chistogram``.

The discrete method is appropriate when the distribution will be used and interpreted as fully discrete, which is the assumption the FFT method makes and the default. The continuous method is useful if the distribution will be used to create a scipy.stats rv_histogram variable. If the continuous method is interpreted as discrete and if the mean is computed as :math:`\sum_i p_k x_k`, which is appropriate for a discrete variable, then it will be under-estimated by :math:`b/2`.

Exact Calculation
""""""""""""""""""

The differences :math:`p_k=F((k + 1/2)b) - F((k - 1/2)b)` can be computed in three different ways, controlled by the ``discretization_calc`` option. The options are:

#. ``discretization_calc='distribution'`` takes differences of the sequence :math:`F((k + 1/2)b)`. This results in a potential loss of accuracy in the right tail where the distribution function increases to 1. The resulting probabilities can be no smaller than the smallest difference between 1 and a float. ``numpy`` reports this as ``numpy.finfo(float).epsneg``; it is of the order ``1e-16``.

#. ``discretization_calc='survival'`` takes the negative difference of the sequence :math:`S(k + 1/2)b)` of survival function values. This results in a potential loss of accuracy in the left tail where the survival function increases to 1. However, it provides better resolution in the right.

#. ``discretization_calc='both'`` attempts to make the best of both worlds, computing::

    np.maximum(np.diff(fz.cdf(adj_xs)), -np.diff(fz.sf(adj_xs)))

  This does double the work and is marginally slower.

The update default is ``survival``. The calculation method does not generally impact the aggregate distribution when FFTs are used because they compute to accuracy about ``1e-16`` (there is a 1 in each row and column of :math:`\mathsf F`, see :ref:`num fft`). However, the option can be helpful to create a pleasing graph of severity log density.

Truncation
~~~~~~~~~~~

The discrete probabilities :math:`p_k` must be truncated into a finite-length vector to use in calculations. The number of buckets used is set by the ``log2`` variable, which inputs its base 2 logarithm. The default is ``log2=16`` corresponding to 65,536 buckets. There are two truncation options, controlled by the ``normalize`` variable.

#. ``normalize=False`` simply truncates, resulting in a vector of probabilities that sums to less than 1.

#. ``normalize=True`` truncates and then normalizes, dividing the truncated vector by its sum, resulting in a vector of probabilities that does sums to 1 (approximately, see :ref:`floats <num floats>`).

The default is ``normalize=True``.

It is obviously desirable for the discrete probabilities to sum to 1. A third option, to put a mass at the maximum loss does not produce intuitive results---since the underlying distributions generally do not have a mass the graphs look wrong.

In general, it is best to use ``normalize=True`` in cases where the truncation error is immaterial, for example with a thin tailed severity. It is numerically cleaner and avoids issues with quantiles close to 1. When there will be an unavoidable truncation error, it is best to use  ``normalize=False``. The user needs to be aware that the extreme right tail is understated. The bucket size and number of buckets should be selected so that the tail is accurate where it is being relied upon. See REF ERROR ANALYSIS for more.

.. warning::
    Be careful using ``normalize=True`` for thick tail severities. It results in unreliable and hard to interpret estimated mean severity.

**Example.** :cite:t:`Schaller2008` consider a Poisson-generalized Pareto model for operational risk. They assume an expected claim count equal to 18 and a generalized Pareto with shape 1, scale 12000 and location 7000. This distribution does not have a mean. They want to model the 90th percentile point. They compare using exponential tilting (see :cite:t:`Grubel1999`) with padding, using up to 1 million (log2=20) buckets. They use a right-hand endpoint of 1 million on the severity. This example illustrates the impact of normalization and shows that padding and tilting have a similar effect.

Setup the base distribution without recomputing. Note infinite severity.

.. ipython:: python
   :okwarning:

   a = build('agg Schaller:Temnov 18 claims sev 12000 * genpareto 1 + 7000 poisson', update=False)
   qd(a)

Execute a variety of updates and assemble answer. Compare Schaller and Temnov, Example 4.3.2, p. 197. They estimate the 90th percentile as 3,132,643. In this case, normalizing severity has a material impact; it acts to decrease the tail thickness and hence estimated percentiles.

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


.. _num algo steps:


HERE


.. _num errors:

Error Analysis
-----------------------------------

There are four sources of error in the algorithm:

#. Discretization: replacing the original distribution with a discretized approximation.
#. Truncation: shrinking the support of the distribution by right truncation.
#. Aliasing: working with only finitely many frequencies in the Fourier domain.
#. FFT algorithm: floating point issues, underflow and (rarely) overflow.


.. _num parameters:

Parameters and Their Selection
-------------------------------

**Parameters**

* bucket size
* number of buckets
* padding
* discretization calculation
* normalization
* severity calculation
* numerical *fuzz*
* tilt


.. _num error analysis:

Error Analysis
--------------------------------------

Exact Examples

* Frequency
* Uniform to triangular
* Gamma to sum; normal to sum
* Tweedie
* Levy convolution
* How many terms? Comparison with simulation? Errors in percentiles?

Explicit Error Analysis for a Tweedie
""""""""""""""""""""""""""""""""""""""""""""

There is a series expansion for the pdf of a Tweedie computed by conditioning on the number of claims and using that a convolution of gammas with the same scale parameter is again gamma. For a Tweedie with expected frequency :math:`\lambda`, gamma shape :math:`\alpha` and scale :math:`\beta`, it is given by

.. math::

    f(x) = \sum_{n \ge 1} e^{-\lambda}\frac{\lambda^n}{n!}\frac{x^{n\alpha-1}e^{-x/\beta}}{\Gamma(n\alpha)\beta^{{n\alpha}}}

for :math:`x>0` and :math:`f(x)=\exp(-\lambda)`. The exact function shows the FFT method is very accurate.

.. ipython:: python
    :okwarning:

    from aggregate import tweedie_convert, build, qd
    from scipy.special import loggamma
    import matplotlib.pyplot as plt
    import numpy as np
    from pandas import option_context

    a = build('agg Tw tweedie 10 1.01 1')
    qd(a.describe)
    @savefig tweedie_test_1.png
    a.plot()

A Tweedie with :math:`p` close to 1 approximates a Poisson. Its gamma severity is very peaked around its mean (high :math:`\alpha` and offsetting small :math:`\beta`).

The next function provides a transparent, if not maximally efficient, implementation of the Tweedie density.

.. ipython:: python
    :okwarning:

    def tweedie_density(x, mean, p, disp):
        pars = tweedie_convert(p=p, μ=mean, σ2=disp)
        λ = pars['λ']
        α = pars['α']
        β = pars['β']
        if x == 0:
            return np.exp(-λ)
        logl = np.log(λ)
        logx = np.log(x)
        logb = np.log(β)
        logbase = -λ
        log_term = 100
        const = -λ - x / β
        ans = 0.0
        for n in range(1, 2000): #while log_term > -20:
            log_term = (const  +
                        + n * logl  +
                        + (n * α - 1) * logx +
                        - loggamma(n+1) +
                        - loggamma(n * α) +
                        - n * α * logb)
            ans += np.exp(log_term)
            if n > 20 and log_term < -227:
                break
        return ans


The following graphs show that the FFT approximation is excellent, across a wide range, just as its good moment-matching performance suggests it would be.

.. ipython:: python
    :okwarning:

    bit = a.density_df.loc[5:a.q(0.99):256, ['p']]
    bit['exact'] = [tweedie_density(i, 10, 1.01, 1) for i in bit.index]
    bit['p'] /= a.bs

    fig, axs = plt.subplots(1, 2, figsize=(2 * 3.5, 2.45), constrained_layout=True, squeeze=True)
    ax0, ax1 = axs.flat

    bit.plot(ax=ax0);
    ax0.set(ylabel='density');
    bit['err'] = bit.p / bit.exact - 1
    bit.err.plot(ax=ax1);
    @savefig tweedie_test_2.png scale=20
    ax1.set(ylabel='relative error', ylim=[-1e-5, 1e-5]);


Approximations and Errors
"""""""""""""""""""""""""""""

* Simulation error of mean
* Simulation error of percentiles
* Moments of a lognormal (Mandlebrot)
* Implications for bs and log2

Based on an analysis of the relative error, select ``log2=18`` and ``bs=1/16``, see :ref:`../5_technical_guides/5_x_approximation_error`. The reported statistics are close to the theoretic numbers implied by the (limited) stochastic model.


:cite:t:`Brown1983`, Estimation of the variance of percentile estimates.

Compute the number of sims to model the mean to within tolerance a of actual with probability p, :math:`(z_{p/2}/a \nu)^2` where :math:`\nu` is the CV. (Usual normal approx to se of mean argument.) Eg for 90% conf z=1.644 and a=0.01 (FFT is generally much closer) you get 27,055 times :math:`\nu^2`. For cat like distributions :math:`\nu` can be in the range 50-100, leading to 67-270 million simulations. Thus FFT provides stunning accuracy.

OK, don't estimate mean. What about SE of percentiles (AEP)?

Finally, what about OEP? OEP is an adjusted quantile, so can use same argument on the severity with adjusted p values. See Brown and Wolf paper, prob JKK for percentile SEs.

BW says se is

.. math:: \frac{1}{f(x_p)}\left(\frac{p(1-p)}{n}\right)^{0.5}

What is the density? (Obs small because range so large)! Table and investigate...


Here's some code on the mean. ::

    import scipy.stats as ss

    z = ss.norm.isf
    phi = ss.norm.cdf

    def test_sample_mean(cv, p=0.99, a=0.01, simulate=False):
        """
        Test number of sims for p=99% certainty of a=1% accuracy when underlying
        variable is lognormal with given cv. Basic large sample, normal approximation
        to standard error of the mean.

        """
        zp = z((1-p)/2)
    n = int((zp / a * cv) ** 2)
    print(f'zp = {zp:.3f}, zp**2 = {zp*zp:.3f}\nformula = {(zp/a)**2:,.0f} * n**2\nn = {n:,.0f}')

    if n <= 100000 or simulate is True:
        mu, sig = mu_sigma_from_mean_cv(1, cv)
        fz = ss.lognorm(sig, scale=np.exp(mu))

        samps = [np.mean(fz.rvs(n)) for i in range(1000)]
        plt.hist(samps, lw=.25, ec='w', bins=10)
        samps = np.sort(samps)
        print(f'observed 99% ci equals ({samps[10]}, {samps[990]})')

    return zp, n

test_sample_mean(.2, .9, .01)

Stuff

The recommended bucket is too small because it is based on only the 99.9 percentile.


The contribution of the extreme tail to the mean of a distribution increases with the tail thickness. See results of
Mandlebrot - the higher moments of the lognormal are nonlocal and depend on different parts of the distribution. (Hence the problems with numerical integration!) [On-line](https://users.math.yale.edu/mandelbrot/web_pdfs/9_E9lognormalDistribution.pdf) *A case against the lognormal distribution* in Mandelbrot, Benoit B. "A case against the lognormal distribution." Fractals and scaling in finance. Springer, New York, NY, 1997. 252-269.

::

    # how much of the mean of a lognormal comes from the extreme tail?
    ans = []
    for sigma in [.5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4]:
        fz = ss.lognorm(sigma)
        for n in range(1,16):
            p = 1 - 10**-n
            q = fz.isf(1-p)
            m, v = fz.stats()
            cv = float(v**.5/m)
            lev = moms_analytic(fz, q, 0, 1)[1]
            ans.append([sigma, n, p, 10**-n, q, cv, float(m), lev])

    ans = pd.DataFrame(ans, columns=['sigma', 'n', 'p', 's', 'q(p)', 'cv', 'mean', 'lev'])
    ans['err'] = ans.lev / ans['mean'] - 1
    print(ans.to_string(formatters={'err': lambda x: f'{x:.1%}'}))

    x = ans.query('abs(err) < 0.001').groupby('sigma').apply(lambda x: x.iloc[0])
    x

    x.set_index('cv')['n'].plot()

Since bs is  based on the p999, it will fail when confronting and extreme tail.

Based on above graph we can come up with an (empirical) relationship between the CV and the required percentile for decent coverage.

More code... test different n for rec bucket, different methods.

::

    from aggregate import build, qd, Aggregate, Severity, round_bucket

    a = build('agg TEST 1 claim sev lognorm 1 cv 50 fixed', update=False)

    for n in range(3,11):
        a.update(recommend_p=n, log2=16)
        qd(a.describe)
        print(f'recommend n = {n}, bucket size = 1 / {1/a.bs}')
        print('-'*100)
        print()
    print(a.info)

    ans = {}
    for m in ['backward', 'round', 'forward']:
        a.update(bs=1/4, sev_calc=m, log2=16, normalize=False)
        print(m)
        qd(a.describe)
        print('-'*100)
        print()
        ans[m] = a.density_df[['p', 'F', 'S']]

    df = pd.concat(ans.values(), keys=ans.keys(), axis=1)

    df.xs('S', axis=1, level=1).plot(xlim=[-1, a.q(0.99)], logy=True, ylim=[1e-2, 1], lw=.5, figsize=(3.5,5))




.. _num swot:

Algorithm Strengths and Weaknesses
--------------------------------------------


* **Pros.**

    - Accurate (see discussion of se of mean and percentiles; exact shape of distribution; can't hope for histograms as sharp; can see what is going on [for his bucket size = 1 need... simulations])
    - Fast: both in absolute terms and especially relative to the accuracy achieved when compared with simulation approaches

        * Speed independent of the expected frequency!

* **Cons.**

    - Univariate: capture one variable at a time; can capture mixtures

        * Yes: mixture with common mixing between lines
        * No: ceded and net; specific and agg combined

      OTOH, so fast you can see the net and ceded distributions, just not
      as a bivariate distribution.

    - Need a small *b* to capture detail for small *x*
    - Need enough space, the range :math:`nb` (or *nb*) to capture the full range of outputs.

Flexibility can be improved using higher dimensional FFT methods, for example to track ceded and net positions simultaneously, but they soon run afoul of the limits of practical computation. See CAS WP ref for an example using 2-dimensional FFTs.

.. _num pricing:

Pricing Methods
----------------

Taken as read: a painful discussion that markets set prices, not actuaries and models. *Pricing* here means *valuing* according to some model. For actuaries valuation has another meaning (reserving for life actuaries). Pricing actuaries call it pricing, understanding that they are just determining a model value.

Several methods apply distortions.

#. :class:`Aggregate`: ``price``, ``apply_distortion``
#. :class:`Portfolio`: ``price``, ``apply_distortion,  (called by ``analyze distortion``)
#. :class:`Distortion`: ``price``, ``price2``
#. Work by hand using ``density_df.p_total``.

All of these methods use the same approach:

* Compute ``S`` as ``1 - p_total.cumsum()``
* Compute ``gS = d.g(S)``
* Compute ``(gS.loc[:a - bs] * np.diff(S.index)).sum()`` or ``.cumsum().iloc[-1]``

Using ``sum`` vs. ``cumsum`` is usually an O(1e-16) difference. These methods use the forward difference of dx and match against the unlagged values of ``S`` or ``gS`` (per PIR p. 272-3). The :class:`Aggregate` method prepends 0 and then computes a ``cumsum``, so the ``a`` index gives the right value.

When ``a`` is given, the series includes ``a`` (based on  ``.loc[:a]``) and the last value is dropped from the sum product.

The next block of code provides a reconciliation of methods.

.. ipython:: python
   :okwarning:

   from aggregate import Portfolio, build, qd
   import pandas as pd
   a = build('agg CommAuto 10 claims 10000 xs 0 sev lognorm 50 cv 4 poisson')
   qd(a)
   pa = Portfolio('test', [a])
   pa.update(log2=16, bs=1/4)
   qd(pa)
   pa.calibrate_distortions(ROEs=[0.1], Ps=[0.99], strict='ordered');
   d = pa.dists['dual']
   qd(pa.distortion_df)
   f"Exact value {pa.distortion_df.iloc[0, 2]:.15f}"
   dm = pa.price(.99, d)
   f'Exact value {dm.price:.15f}'
   bit = a.density_df[['loss', 'p_total', 'S']]
   bit['aS'] = 1 - bit.p_total.cumsum()
   bit['gS'] = d.g(bit.S)
   bit['gaS'] = d.g(bit.aS)
   test = pd.Series((d.price(bit.loc[:a.q(0.99), 'p_total'], kind='both')[-1],
                     d.price(a.density_df.p_total, a.q(0.99), kind='both')[-1],
                     d.price2(bit.p_total).loc[a.q(0.99)].ask, \
                     d.price2(bit.p_total, a.q(0.99)).ask,
                     a.price(0.99, d).iloc[0, 1],
                     dm.price,
                     bit.loc[:a.q(0.99)-a.bs, 'gS'].sum() * a.bs,
                     bit.loc[:a.q(0.99)-a.bs, 'gS'].cumsum().iloc[-1] * a.bs,
                     bit.loc[:a.q(0.99)-a.bs, 'gaS'].sum() * a.bs,
                     bit.loc[:a.q(0.99)-a.bs, 'gaS'].cumsum().iloc[-1] * a.bs),
             index=['distortion.price',
                    'distortion.price with a',
                    'distortion.price2, find a',
                    'distortion.price2(a)',
                    'Aggregate.price',
                    'Portfolio.price',
                    'bit sum',
                    'bit cumsum',
                    'bit sum alt S',
                    'bit cumsum alt S'
                   ])
   qd(test.sort_values(),
      float_format=lambda x: f'{x:.15f}')
   qd(test.sort_values() / test.sort_values().iloc[-1] - 1,
      float_format=lambda x: f'{x:.6e}')


.. _num floats:

Floating Point Arithmetic and Rounding Errors
-----------------------------------------------

Floating point arithmetic is not associative.

.. ipython:: python

   x = .1 + (0.6 + 0.3)
   y = (0.1 + 0.6) + 0.3
   x, x.as_integer_ratio(), y, y.as_integer_ratio()

Cumulative error. Knuth observations.

**Exercise Redux.**

Recall the exercise to compute  quantiles of a :ref:`dice roll <prob dice quantiles>`.
``aggregate`` produces the consistent results---if we look carefully and account for the foibles of floating point numbers. The case :math:`p=0.1` is easy. But the case :math:`p=1/6` appears wrong. There are two ways we can model the throw of a dice: with frequency 1 to 6 and fixed severity 1, or as fixed frequency 1 and severity 1 to 6. They give different answers. The lower quantile is wrong in the first case (it equals 1) and the upper quantile in the second (2).

.. ipython:: python
    :okwarning:

    from aggregate import build, qd
    import pandas as pd
    d = build('agg Dice dfreq [1:6] dsev [1]')
    print(d.q(0.1, 'lower'), d.q(0.1, 'upper'))
    print(d.q(1/6, 'lower'), d.q(1/6, 'upper'))
    d2 = build('agg Dice2 dfreq [1] dsev [1:6]')
    print(d2.q(1/6, 'lower'), d2.q(1/6, 'upper'))

These differences are irritating! The short answer is to adhere to

.. warning::
    Always use binary floats, that have an exact binary representation. They must have an exact binary representation as a fraction :math:`a/b` where :math:`b` is a power of two. 1/3, 1/5 and 1/10 are **not** binary floats.

Here's the long answer, if you want to know. Looking at the source shows that the quantile function is implemented as a previous or next look up on a dataframe of distinct values of the cumulative distribution function. These two dataframes are:

.. ipython:: python
    :okwarning:

    ff = lambda x: f'{x:.25g}'
    qd(d.density_df.query('p_total > 0')[['p', 'F']], float_format=ff)
    qd(d2.density_df.query('p_total > 0')[['p', 'F']], float_format=ff)
    print(f'\n{d.cdf(1):.25f} < {1/6:.25f} < 1/6 < {d2.cdf(1):.25f}')

Based on these numbers, the reported quantiles are correct. :math:`p=1/6` is strictly greater than ``d.cdf(1)`` and strictly less than ``d2.cdf(1)``, as shown in the last row! ``d`` and ``d2`` are different because the former runs through the FFT routine to convolve the trivial severity, whereas the latter does not.

----

**Exercise.** :math:`X` is a random variable defined on a sample space
with ten equally likely events. The event outcomes are
:math:`0,1,1,1,2,3, 4,8, 12, 25`. Compute :math:`\mathsf{VaR}_p(X)` for
all :math:`p`.

.. ipython:: python
    :okwarning:

    a = build('agg Ex.50 dfreq [1] '
              'dsev [0 1 2 3 4 8 12 25] [.1 .3 .1 .1 .1 .1 .1 .1]')
    @savefig quantile_a.png
    a.plot()
    print(a.q(0.05), a.q(0.1), a.q(0.2), a.q(0.4),
       a.q(0.4, 'upper'), a.q(0.41), a.q(0.5))
    qd(a.density_df.query('p_total > 0')[['p', 'F']],
        float_format=ff)

**Solution.** On the graph, fill in the vertical segments of the
distribution function. Draw a horizontal line at height :math:`p` and
find its intersection with the completed graph. There is a unique
solution for all :math:`p` except :math:`0.1, 0.4, 0.5,\dots, 0.9`.
Consider :math:`p=0.4`. Any :math:`x` satisfying
:math:`\mathsf{Pr}(X < x) \le 0.4 \le \mathsf{Pr}(X\le x)` is a :math:`0.4`-quantile. By
inspection the solutions are :math:`1\le x \le 2`. VaR is defined as the
lower quantile, :math:`x=1`. The :math:`0.41` quantile is :math:`x=2`.
VaRs are not interpolated in this problem specification. The loss 25 is
the :math:`p`-VaR for any :math:`p>0.9`. The apparently errant numbers for aggregate (the upper quantile at 0.1 equals 2) are explained by the float representations. The float representation of ``0.4`` is ``3602879701896397/9007199254740992`` which actually equals ``0.4000000000000000222044605``.


.. _num fft:

Fast Fourier Transforms
-----------------------

There are three things going on here:

#. **Fourier transform**
#. **Discrete** Fourier transform
#. **Fast** Fourier transform

**Fourier transforms** provide an alternative way to represent a distribution function. Given a distribution function :math:`F`, its Fourier transform (FT) is usually written :math:`\hat F`. The FT contains the same information as the distribution and there is a dictionary back and forth between the two, using the inverse FT.
Some computations with distributions are easier to perform using their FT, which is what makes them useful.

The FT is like exponentiation for distributions.
Exponentials and logs turn (difficult) multiplication into (easy) addition

.. math:: e^a \times e^b = e^{a+b}.

FTs turn difficult convolution of distributions (addition of the corresponding random variables) into easy multiplication. If :math:`X_i` are random variables and :math:`F_X` is the distribution of  :math:`X`, :math:`i=1,2`, and :math:`X=X_1+X_2` with distribution :math:`F` then

.. math:: \widehat{F_{X_1+X_2}} = \widehat{F_{X_1}} \times \widehat{F_{X_2}},

where the righthand side is a product of functions. Computing the distribution of a sum of random variables is complicated because you have to consider all different ways an outcome can be split. But it is easy using FTs:

* Take the FT of each distribution function
* Multiply the FTs
* Take the inverse FT

Of course, this depends on it being easy to compute the FT and its inverse---which is where FFTs come in.

**Discrete** Fourier transforms are a discrete approximation to continuous FTs, formed by sampling at finitely many points. The DFT is a vector, rather than a function. They retain the convolution property of FTs. They are sometimes called discrete cosine transforms (DCT).

The **Fast** Fourier transform generally refers to a very fast way to compute discrete FTs, but general usage blurs the distinction between discrete FTs and their computation and uses FFT as a catchall for both.

Here are some more details.
The FFT of the :math:`m\times 1` vector
:math:`\mathsf{x}=(x_0,\dots,x_{m-1})`, a discrete approximation to a distribution in our application, is another :math:`m\times 1` vector :math:`\hat{\mathsf{x}}`
whose :math:`j`\ th component is

.. math:: \sum_{k=0}^{m-1} x_k\exp(-2\pi ijk/m),

where :math:`i=\sqrt{-1}`. The coefficients of :math:`\hat{\mathsf{x}}` are complex numbers. It is
possible to express :math:`\hat{\mathsf{x}}=\mathsf{F}\mathsf{x}` where

.. math::

   \mathsf{F}=
   \begin{pmatrix}
   1 & 1 & \dots & 1 \\
   1 & w & \dots & w^{m-1} \\
   1 & w^2 & \dots & w^{2(m-1)} \\
   \vdots & & & \vdots \\
   1 & w^{m-1} & \dots & w^{(m-1)^2}
   \end{pmatrix}

is a matrix of complex roots of unity and  :math:`\exp(-2\pi i/m)`. This shows there is nothing
inherently mysterious about an FFT. The trick is that there is a very
efficient algorithm for computing the matrix multiplication :cite:p:`Press1992a`.  Rather than taking
time proportional to :math:`m^2`, as one would expect, it can be
computed in time proportional to :math:`m\log(m)`. For large values of :math:`m`, the difference
between :math:`m\log(m)` and :math:`m^2` time is the difference between
practically possible and practically impossible.

The inverse FFT to recovers :math:`\mathsf{x}` from its transform
:math:`\hat{\mathsf{x}}`. The inverse FFT is computed using the same equation as the FFT with :math:`\mathsf F^{-1}` (matrix inverse) in place of :math:`F`. The inverse equals


.. math::

   \mathsf{F}^{-1}=
   \frac{1}{m}
   \begin{pmatrix}
   1 & 1 & \dots & 1 \\
   1 & w^{-1} & \dots & w^{-(m-1)} \\
   1 & w^2 & \dots & w^{2(m-1)} \\
   \vdots & & & \vdots \\
   1 & w^{-(m-1)} & \dots & w^{-(m-1)^2}
   \end{pmatrix}


Because the equation is essentially the same, the inversion process can also be computed in :math:`m\log(m)` time.

In the convolution algorithm, the product of functions :math:`\widehat{F_{X_1}} \times \widehat{F_{X_2}}` is replaced by the component-by-component product of two vectors, which is easy to compute. Thus, to convolve two discrete distributions, represented as :math:`\mathsf p=(p_0,\dots,p_{m-1})` and :math:`\mathsf q=(q_0,\dots,q_{m-1})` simply

* Take the FFT of each vector, :math:`\hat{\mathsf p}=\mathsf F\mathsf p` and :math:`\hat{\mathsf q}=\mathsf F\mathsf q`
* Compute the component-by-component product :math:`\mathsf z = \hat{\mathsf p}\hat{\mathsf q}`
* Compute the inverse FFT :math:`\mathsf F^{-1}\mathsf z`.

The answer is the exact convolution of the two input distributions, except that sum values wrap around: the extreme right tail re-appears as probabilities around 0. This problem is called aliasing (the same as the wagon-wheel effect in old Westerns), but it can be addressed by padding the input vectors.


It is not necessary to understand the details of FTs to use ``aggregate`` although they are fascinating, see for example :cite:t:`Korner2022`. In probability, the moment generating functions and characteristic function are based on FTs. They are discussed in any serious probability text.

.. pi and agm?
