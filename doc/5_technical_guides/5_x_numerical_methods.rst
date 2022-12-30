.. _2_x_fft_convolution:

Numerical Methods and FFT Convolution
=======================================

**Objectives:**  Describe FFT numerical convolution, the basis of all computations in ``aggregate``.

**Audience:** User who wants to understand the computation options and parameters.

**Prerequisites:** Probability theory behind aggregate distributions. Basic complex numbers and matrix multiplication. Basics of numerical analysis, especially numerical integration.

**See also:**  :ref:`aggregate <2_x_aggregate>`.

For thick tailed lognormal variables, it is best to truncate the severity distribution. Truncation does not impact PML estimates below the probability of truncation.  We select a truncation of USD 20T, about the size of the US economy. The unlimited models suggest there is less than a 1 in 10,000 chance of a model so large.


Fast, Accurate, and Flexible
------------------------------

Numerical analysts face a trilemma

   **Fast, accurate, and flexible: pick two!**



Simulation methods are always flexible, but trade speed and accuracy. Fewer simulations is faster but provides lower accuracy; many simulations improves accuracy at the cost of speed.

FFT-based methods provide speed and accuracy but at some cost in flexibility. Their accuracy is other-worldy compared to simulation, see REF.

``aggregate`` relies on one-dimensional FFT methods which can **track only one variable** at a time. Thus, FFT-based methods can model

* Gross, ceded, or net position
* A sum of independent lines (other methods are available for correlation)
* A sum of dependent lines, where the dependence structure is driven by common mixing variables.

FFT-methods shine for thick tailed distributions where accuracy is important. For example:

* Catastrophe risk PMLs, AEP, OEP points
* Individual line pricing with underlying, e.g., large account retention with specific and aggregate covers.
* Covers with low expected loss rates which are hard to simulate with sufficient accuracy.

The downside is less flexibility. FFT-based methods cannot model the

* Joint distribution of ceded and net loss.
* Total cession to specific and aggregate covers because it requires the bivariate distribution. It can model the sum of the parts.

Flexibility can be improved using higher dimensional FFT methods, for example to track ceded and net positions simultaneously, but they soon run afoul of the limits of practical computation. See CAS WP ref for an example using 2-dimensional FFTs.


Floats
---------

Floating point arithmetic is not associative.

.. ipython:: python

   x,y = 4.41 + (2.36 + 1.53), (4.41 + 2.36) + 1.53
   x,y = .1 + (0.6 + 0.3), (0.1 + 0.6) + 0.3
   x, y, x.as_integer_ratio(), y.as_integer_ratio()




.. _num how agg reps a dist:

How ``aggregate`` represents a distribution
--------------------------------------------

The *representation* is amenable to computation. It should provide a cumulative distribution function and other probability functions. These can be analytical, such as the normal cdf or Weibull distribution function. However, aggregates rarely have closed form expressions. Therefore we use a numerical approximation to the exact pdf or pmf.

There are two obvious ways to construct a numerical approximation:

#. As a discrete (arithmetic, lattice) distribution supported on :math:`0, b, 2b, \dots`.

#. As a continuous random variable with a piecewise linear distribution function.

The second approach assumes the aggregate is actually a continuous random variable, which is often not the case. For example, the Tweedie and all other compound Poisson distributions are mixed. When :math:`X` is mixed it is impossible to distinguish the jump and continuous parts when using a numerical approximation. The large jumps are obvious but the small ones are not.

We live in a discrete world. Monetary amounts are multiples of a smallest unit: the penny, cent, yen, satoshi;
at the same time, we can be prejudiced in favor of analytic solutions. Computers, however, definitely favor numerical ones.

.. version of this in 10 mins

For all of these reasons we use a discrete numerical approximation. To "know or compute an aggregate" means that we have a discrete approximation to its distribution function that is concentrated on integer multiples of a fixed bandwidth or bucket size :math:`b`. Concretely, this specifies the aggregate as the value :math:`b` and a vector of probabilities :math:`(p_0,p_1,\dots, p_{n-1})` with the interpretation

.. math:: \Pr(X=kb)=p_k.

All subsequent computations assume that this approximation **is** the aggregate distribution. Thus, moments can be estimated via

.. math:: \sum_k k^r p_i b

for example.


Other notes
----------------

How FFT works is technical appendix. Using it to compute aggs goes here.

Emb Fr paper on moment matching

KPW on moment matching p. 182.

Panjer and Lutek [97] found that two moments were usually sufficient and that adding a third moment requirement adds only marginally to the accuracy. Furthermore, the **rounding method and the first-moment method (p = 1) had similar errors**, while the second-moment method (p = 2) provided significant improvement. The specific formulas for the method of rounding and the method of matching the first moment are given in Appendix E. A reason to favor matching zero or one moment is that the resulting probabilities will always be **nonnegative**. When matching two or more moments, this cannot be guaranteed.

The methods described here are qualitatively similar to numerical methods used to solve Volterra integral equations such as (9.26) developed in numerical analysis (see, e.g. Baker [10]).

Ex 9.41 gives the formulas for weights in terms of LEVs.

Parameters
-----------

* bucket size
* number of buckets
* padding
* discretization calculation
* normalization
* severity calculation
* numerical *fuzz*


.. _sev calc:

Discretizing the Severity Distribution
-------------------------------------------

Discretizing approximates the severity with a purely discrete distribution supported at points :math:`x_k=x_0+kb`, :math:`k=0,1,\dots, N`, where :math:`b` is called the **bucket size** or the **bandwidth**. The corresponding discrete probabilities can be computed in four ways.

#. The **round** or **discrete** method assigns probability

   .. math:: p_k = \Pr(x_k - b/2 < X \le x_k+b/2)

   to the :math:`k`th bucket.

#. The **forward** difference assigns

   .. math:: p_k = \Pr(x_k - b/2 < X \le x_{k+1} )

#. The **backward** difference assigns

   .. math:: p_k = \Pr(x_{k-1} - b/2 < X \le x_k )

   with (?) :math:`p_0=0`.

#. The **moment** difference (Gerber, KPW) assigns

   .. math::

      p_0 &= 1 - \frac{\mathsf E[X \wedge b]}{b} \\
      p_k &= \frac{2\mathsf E[X \wedge kb] - \mathsf E[X \wedge (k-1)b] - \mathsf E[X \wedge (k+1)b]}{b}

   It ensures the discretized distribution has the same first moment as the original distribution. This method can be extended to match more moments,  but the resulting weights are not guaranteed to be positive.

Call the discrete approximation :math:`X_b^d` where :math:`d=r,\ f,\ b,\ m` describes the discretization. It is clear that :math:`X_b` converges weakly (in :math:`L^1`) to :math:`X` and the same holds for a compound distribution using :math:`X` as severity for the rounding, forward and backward methods. Further, the rounding approximation is sandwiched between the forward and backwards methods (REF EF.p499).


EF comment on moment method:

   In this light, Gerber (1982) suggests a procedure that locally matches the first k moments. Practically interesting is only the case k = 1; for k ≥ 2 the procedure is not well defined, potentially leading to negative probability mass on certain lattice points. The moment matching method is much more involved than the rounding method in terms of implementation; we need to calculate limited expected values. Apart from that, the gain is rather modest; moment matching only pays off for large bandwidths, and after all, the rounding method is to be preferred. This is further reinforced by the work of Grübel and Hermesmeier (2000): if the severity distribution is absolutely continuous with a sufficiently smooth density, the quantity :math:`f_{h,j} / h`, an approximation for the compound density, can be quadratically extrapolated.

Need quad to work...bot not positive. Explore adjusting the first couple of buckets.

To create a rv_histogram variable from ``xs`` and corresponding ``p`` values use:

   ::

       xss = np.sort(np.hstack((xs, xs + 1e-5)))
       pss = np.vstack((ps1, np.zeros_like(ps1))).reshape((-1,), order='F')[:-1]
       fz_discr = ss.rv_histogram((pss, xss))

The value 1e-5 just needs to be smaller than the resolution requested, i.e. do not “split the bucket”. Generally histograms will be downsampled, not upsampled, so this is not a restriction.

Continuous Approximation to Severity (Ogive)
---------------------------------------------

Approximate the distribution with a continuous “histogram” distribution that is uniform on :math:`(x_k, x_{k+1}]`. The discrete proababilities are :math:`p_k=P(x_k < X \le x_{k+1})`. To create a rv_histogram variable is much easier, just use::

    xs2 = np.hstack((xs, xs[-1] + xs[1]))
    fz_cts = ss.rv_histogram((ps2, xs2))

The first method we call **discrete** and the second **histogram**. The discrete method is appropriate when the distribution will be used and interpreted as fully discrete, which is the assumption the FFT method makes. The histogram method is useful if the distribution will be used to create a scipy.stats rv_histogram variable. If the historgram method is interpreted as discrete and if the mean is computed appropriately for a discrete variable as :math:`\sum_i p_k x_k`, then the mean will be under-estimated by :math:`b/2`.



Fast Fourier Transforms
-----------------------

The FFT method is a miraculous technique for computing aggregate
distributions. It is especially effective when the expected claim count
is relatively small and the underlying severity distribution is bounded.
These assumptions are true for many excess of loss reinsurance treaties,
for example. Thus the FFT is very useful when quoting excess layers with
annual aggregate deductibles or other variable features. The FFT
provides a discrete approximation to the moment generating function.

To use the FFT method, first “bucket” (or quantize) the severity
distribution into a density vector :math:`\text{x}=(x_1,\dots,x_{m})`
whose length :math:`m` is a power of two :math:`m=2^n`. Here

.. math::

   \begin{gathered}
   x_i= \text{Pr}((i-1/2)b<X<(i+1/2)b)\\ x_1=\text{Pr}(X<b/2),\quad x_{m}=\text{Pr}(X>(m-1/2)b)\end{gathered}

for some fixed :math:`b`. We call :math:`b` the bucket size. Note
:math:`\sum_i x_i=1` by construction. The FFT of the :math:`m\times 1`
vector :math:`\text{x}` is another :math:`m\times 1` vector
:math:`\hat{\text{x}}` whose :math:`j`\ th component is

.. math::

   \sum_{k=0}^{2^n-1} x_k\exp(2\pi ijk/2^n).\label{fft}

The coefficients of :math:`\hat{\text{x}}` are complex numbers. It is
also possible to express :math:`\hat{\text{x}}=\text{F}\text{x}` where
:math:`\text{F}` is an appropriate matrix of complex roots of unity, so
there is nothing inherently mysterious about a FFT. The trick is that
there exists a very efficient algorithm for computing. Rather than
taking time proportional to :math:`m^2`, as one would expect, it can be
computed in time proportional to :math:`m\log(m)`. The difference
between :math:`m\log(m)` and :math:`m^2` time is the difference between
practically possible and practically impossible.

You can use the inverse FFT to recover :math:`\text{x}` from its
transform :math:`\hat{\text{x}}`. The inverse FFT is computed using the
same equation as the FFT except there is a minus sign in the exponent
and the result is divided by :math:`2^n`. Because the equation is
essentially the same, the inversion process can also be computed in
:math:`m\log(m)` time.

The next step is magic in actuarial science. Remember that if :math:`N`
is a :math:`G`-mixed Poisson and :math:`A=X_1+\cdots+X_N` is an
aggregate distribution then

.. math::

   M_A(\zeta)=M_G(n(M_X(\zeta)-1)).

Using FFTs you can replace the *function* :math:`M_X` with the discrete
approximation *vector* :math:`\hat{\text{x}}` and compute

.. math::

   \hat{\text{a}}=M_G(n(\hat{\text{x}} -1))

component-by-component to get an approximation vector to the function
:math:`M_A`. You can then use the inverse FFT to recover an discrete
approximation :math:`\text{a}` of :math:`A` from :math:`\hat{\text{a}}`!
See Wang for more details.

Similar tricks are possible in two dimensions—see Press et al. and Homer
and Clark for a discussion.

The FFT allows us to use the following very simple method to
qualitatively approximate the density of an aggregate of dependent
marginals :math:`X_1,\dots,X_n` given a correlation matrix
:math:`\Sigma`. First use the FFT method to compute the sum :math:`S'`
of the :math:`X_i` as though they were independent. Let
:math:`\text{Var}(S')=\sigma^{'2}` and let :math:`\sigma^2` be the
variance of the sum of the :math:`X_i` implied by :math:`\Sigma`. Next
use the FFT to add a further “noise” random variable :math:`N` to
:math:`S'` with mean zero and variance :math:`\sigma^2-\sigma^{'2}`. Two
obvious choices for the distribution of :math:`N` are normal or shifted
lognormal. Then :math:`S'+N` has the same mean and variance as the sum
of the dependent variables :math:`X_i`. The range of possible choices
for :math:`N` highlights once again that knowing the marginals and
correlation structure is not enough to determine the whole multivariate
distribution. It is an interesting question whether all possible choices
of :math:`N` correspond to actual multivariate structures for the
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

Fast Fourier Transforms
-----------------------

The FFT method is a miraculous technique for computing aggregate
distributions. It is especially effective when the expected claim count
is relatively small and the underlying severity distribution is bounded.
These assumptions are true for many excess of loss reinsurance treaties,
for example. Thus the FFT is very useful when quoting excess layers with
annual aggregate deductibles or other variable features. The FFT
provides a discrete approximation to the moment generating function.

To use the FFT method, first “bucket” (or quantize) the severity
distribution into a density vector :math:`\mathsf{x}=(x_1,\dots,x_{m})` whose
length :math:`m` is a power of two :math:`m=2^n`. Here

.. math::

   x_i=\mathsf{Pr}((i-1/2)b<X<(i+1/2)b)\\ x_1=\mathsf{Pr}(X<b/2),\quad x_{m}=\mathsf{Pr}(X>(m-1/2)b)

for some fixed :math:`b`. We call :math:`b` the bucket size. Note
:math:`\sum_i
x_i=1` by construction. The FFT of the :math:`m\times 1` vector
:math:`\mathsf{x}` is another :math:`m\times 1` vector :math:`\hat{\mathsf{x}}` whose :math:`j`\ th component is

.. math:: \sum_{k=0}^{2^n-1} x_k\exp(2\pi ijk/2^n).

The coefficients of :math:`\hat{\mathsf{x}}` are complex numbers. It is also
possible to express :math:`\hat{\mathsf{x}}=\mathsf{F}\mathsf{x}` where :math:`\mathsf{F}` is an
appropriate matrix of complex roots of unity, so there is nothing
inherently mysterious about a FFT. The trick is that there exists a very
efficient algorithm for computing (`[fft] <#fft>`__). Rather than taking
time proportional to :math:`m^2`, as one would expect, it can be
computed in time proportional to :math:`m\log(m)`. The difference
between :math:`m\log(m)` and :math:`m^2` time is the difference between
practically possible and practically impossible.

You can use the inverse FFT to recover :math:`\mathsf{x}` from its transform
:math:`\hat{\mathsf{x}}`. The inverse FFT is computed using the same equation
(`[fft] <#fft>`__) as the FFT except there is a minus sign in the
exponent and the result is divided by :math:`2^n`. Because the equation
is essentially the same, the inversion process can also be computed in
:math:`m\log(m)` time.

The next step is magic in actuarial science. Remember that if :math:`N`
is a :math:`G`-mixed Poisson and :math:`A=X_1+\cdots+X_N` is an
aggregate distribution then

.. math:: M_A(\zeta)=M_G(n(M_X(\zeta)-1)).

Using FFTs you can replace the *function* :math:`M_X` with the discrete
approximation *vector* :math:`\hat{\mathsf{x}}` and compute

.. math:: \hat{\mathsf{a}}=M_G(n(\hat{\mathsf{x}} -1))

component-by-component to get an approximation vector to the function
:math:`M_A`. You can then use the inverse FFT to recover an discrete
approximation :math:`\a` of :math:`A` from :math:`\hat{\mathsf{a}}`! See (big) Wang
for more details.

Similar tricks are possible in two dimensions—see Press et al.,
and Homer and Clark for a discussion.

The FFT allows us to use the following very simple method to
qualitatively approximate the density of an aggregate of dependent
marginals :math:`X_1,\dots,X_n` given a correlation matrix
:math:`\Sigma`. First use the FFT method to compute the sum :math:`S'`
of the :math:`X_i` as though they were independent. Let
:math:`\mathsf{Var}(S')=\sigma^{'2}` and let :math:`\sigma^2` be the variance of
the sum of the :math:`X_i` implied by :math:`\Sigma`. Next use the FFT
to add a further “noise” random variable :math:`N` to :math:`S'` with
mean zero and variance :math:`\sigma^2-\sigma^{'2}`. Two obvious choices
for the distribution of :math:`N` are normal or shifted lognormal. Then
:math:`S'+N` has the same mean and variance as the sum of the dependent
variables :math:`X_i`. The range of possible choices for :math:`N`
highlights once again that knowing the marginals and correlation
structure is not enough to determine the whole multivariate
distribution. It is an interesting question whether all possible choices
of :math:`N` correspond to actual multivariate structures for the
:math:`X_i` and conversely whether all multivariate structures
correspond to an :math:`N`. (It is easy to use MGFs to deconvolve
:math:`N` from the true sum using Fourier methods; the question is
whether the resulting “distribution” is non-negative.)

Heckman and Meyers used Fourier
transforms to compute aggregate distributions by numerically integrating
the characteristic function. Direct inversion of the Fourier transform
is also possible using FFTs. The application of FFTs is not completely
straight forward because of certain aspects of the approximations
involved. The details are very clearly explained in Menn and Rachev
:raw-latex:`\cite{mennrachev}`. Their method allows the use of FFTs to
determine densities for distributions which have analytic MGFs but not
densities—notably the class of stable distributions.



The FFT Approach
-----------------

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


Define the ? quantization or discretization error to be the difference between these two.

If *X* is abs continuous then L1 convergence. (AC implies uniformly continuous, so for ep exist delta, take bs less than delta and you are within ep.)


Approximations and Errors
-----------------------------

* Simulation error of mean
* Simulation error of percentiles
* Moments of a lognormal (Mandlebrot)
* Implications for bs and log2

Based on an analysis of the relative error, select ``log2=18`` and ``bs=1/16``, see :ref:`../5_technical_guides/5_x_approximation_error`. The reported statistics are close to the theoretic numbers implied by the (limited) stochastic model.


Estimation of the variance of
percentile estimates; Morton B. BROWN and Robert A. WOLFE

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
Mandlebrot - the higher moments of the lognormal are nonlocal and depend on different parts of the distribution. (Hence the problems with numerical integration!) (https://users.math.yale.edu/mandelbrot/web_pdfs/9_E9lognormalDistribution.pdf) *A case against the lognormal distribution* in Mandelbrot, Benoit B. "A case against the lognormal distribution." Fractals and scaling in finance. Springer, New York, NY, 1997. 252-269.

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


Explicit Error Quantification for a Tweedie
-----------------------------------------------

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

