======================
5. Technical Resources
======================


Actuarial, probability, and other non-programming backup.

Background on agg loss dists, SRMs, and useful references.



Probability and Risk Theory
===========================

<!-- from 5_Technical_Resources.md via pandoc -->

Discretizing Severity Distributions
-----------------------------------

There are two simple ways to discretize a continuous distribution.

1. Approximate the distribution with a purely discrete distribution
   supported at points :math:`x_k=x_0+kb`, :math:`k=0,1,\dots, N`. Call
   :math:`b` the bucket size. The discrete probabilities are
   :math:`p_k=P(x_k - b/2 < X \le x_k+b/2)`. To create a rv_histogram
   variable from ``xs`` and corresponding ``p`` values use:

   ::

       xss = np.sort(np.hstack((xs, xs + 1e-5)))
       pss = np.vstack((ps1, np.zeros_like(ps1))).reshape((-1,), order='F')[:-1]
       fz_discr = ss.rv_histogram((pss, xss))

The value 1e-5 just needs to be smaller than the resolution requested,
i.e. do not “split the bucket”. Generally histograms will be
downsampled, not upsampled, so this is not a restriction.

2. Approximate the distribution with a continuous “histogram”
   distribution that is uniform on :math:`(x_k, x_{k+1}]`. The discrete
   proababilities are :math:`p_k=P(x_k < X \le x_{k+1})`. To create a
   rv_histogram variable is much easier, just use:

   ::

       xs2 = np.hstack((xs, xs[-1] + xs[1]))
       fz_cts = ss.rv_histogram((ps2, xs2))

The first method we call **discrete** and the second **histogram**. The
discrete method is appropriate when the distribution will be used and
interpreted as fully discrete, which is the assumption the FFT method
makes. The histogram method is useful if the distribution will be used
to create a scipy.stats rv_histogram variable. If the historgram method
is interpreted as discrete and if the mean is computed appropriately for
a discrete variable as :math:`\sum_i p_k x_k`, then the mean will be
under-estimated by :math:`b/2`.

Generalized Distributions
-------------------------

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



Frequency Distributions
=======================

<!-- from 05.md -->

A random variable :math:`N` is :math:`G`-mixed Poisson if
:math:`N\mid G` has a Poisson :math:`nG` distribution for some fixed
non-negative :math:`n` and a non-negative mixing distribution :math:`G`
with :math:`\text{E}(G)=1`. Let :math:`\text{Var}(G)=c` (Glenn Meyers
calls :math:`c` the contagion) and let :math:`\text{E}(G^3)=g`.

The MGF of a :math:`G`-mixed Poisson is

.. math::

   \label{mgfi}
   M_N(\zeta)=\text{E}(e^{\zeta N})=\text{E}(\text{E}(e^{\zeta N} \mid G))=\text{E}(e^{n
     G(e^\zeta-1)})=M_G(n(e^\zeta-1))

since :math:`M_G(\zeta):=\text{E}(e^{\zeta G})` and the MGF of a Poisson
with mean :math:`n` is :math:`\exp(n(e^\zeta-1))`. Thus

.. math::


   \text{E}(N)=M_N'(0)=n M_G'(0)=n,

because :math:`\text{E}(G)=M_G'(0)=1`. Similarly

.. math::


   \text{E}(N^2)=M_N''(0)=n^2M_G''(0)+n M_G'(0)=n^2(1+c)+n

and so

.. math::


   \text{Var}(N)=n(1+cn).

Finally

.. math::


   \text{E}(N^3) = M_N'''(0) =n^3M_G'''(0)+3n^2M_G''(0)+n M_G'(0) = gn^3 + 3n^2(1+c) + n

and therefore the central moment

.. math::


   \text{E}(N-\text{E}(N))^3 = n^3(g -3c -1) + 3cn^2 + n.

We can also assume :math:`G` has mean :math:`n` and work directly with
:math:`G` rather than :math:`nG`, :math:`\text{E}(G)=1`. We will call
both forms mixing distributions.

Interpretation of the Coefficient of Variation of the Mixing Distribution
-------------------------------------------------------------------------

Per Actuarial Geometry, if :math:`\nu` is the CV of :math:`G` then the
:math:`\nu` equals the asymptotic coefficient of variation for any
:math:`G`-mixed compound Poisson distribution whose variance exists. The
variance will exist iff the variance of the severity term exists.

Gamma Mixing
------------

A negative binomial is a gamma-mixed Poisson: if :math:`N \mid G` is
distributed as a Poisson with mean :math:`G`, and :math:`G` has a gamma
distribution, then the unconditional distribution of :math:`N` is a
negative binomial. A gamma distribution has a shape parameter :math:`a`
and a scale parameter :math:`\theta` so that the density is proportional
to :math:`x^{a-1}e^{x/\theta}`, :math:`\text{E}(G)=a\theta` and
:math:`\text{Var}(G)=a\theta^2`.

Let :math:`c=\text{Var}(G)=\nu^2`, so :math:`\nu` is the coefficient of
variation of the mixing distribution. Then

-  :math:`a\theta=1` and :math:`a\theta^2=c`
-  :math:`\theta=c=\nu^2`, :math:`a=1/c`

The non-central moments of the gamma distribution are
:math:`\text{E}(G^r)=\theta^r\Gamma(a+r)/\Gamma(a)`. Therefore
:math:`Var(G) = a\theta^2` and :math:`E(G-E(G))^3 = 2a\theta^3`. The
skewness of :math:`G` is :math:`\gamma = 2/\sqrt(a) = 2\nu`.

Applying the general formula for the third central moment of :math:`N`
we get an expression for the skewness

.. math::


   \text{skew}(N) = \frac{n^3(\gamma -3c -1) + n^2(3c+2) + n}{(n(1+cn))^{3/2}}.

The corresponding MGF of the gamma is
:math:`M_G(\zeta) = (1-\theta\zeta)^{-a}`.

Shifted Mixing (General)
------------------------

We can adjust the skewness of mixing with shifting. In addition to a
target CV :math:`\nu` assume a proportion :math:`f` of claims are sure
to occur. Use a mixing distribution :math:`G=f+G'` such that

-  :math:`E(G)= f + E(G') = 1` and
-  :math:`CV(G) = SD(G') = \nu`.

As :math:`f` increases from 0 to 1 the skewness of :math:`G` will
increase. Delaporte first introduced this idea.

Since :math:`\text{skew}(G)=\text{skew}(G')` we have
:math:`g=\text{E}(G^3)=\nu^3 \text{skew}(G')+3c+1`.

Delaporte Mixing (Shifted Gamma)
--------------------------------

Inputs are target CV :math:`\nu` and proportion of certain claims
:math:`f`, :math:`0\leq f \leq 1`. Find parameters :math:`f`, :math:`a`
and :math:`\theta` for a shifted gamma :math:`G=f+G'` with
:math:`E(G')=1-f` and :math:`SD(G')=\nu` as

-  :math:`f` is input
-  mean :math:`a\theta=1-s` and :math:`CV=\nu=\sqrt{a}\theta` so
   :math:`a=(1-f)^2/\nu^2=(1-f)^2/c` and :math:`\theta=(1-f)/a`

The skewness of :math:`G` equals the skewness of :math:`G'` equals
:math:`2/\sqrt{a}= 2\nu/(1-f)`, which is then greater than the skewness
:math:`2\nu` when :math:`f=0`. The third non-central moment
:math:`g=2\nu^4/(1-f)+3c+1`

Poisson Inverse Gaussian Distribution
-------------------------------------

Bernoulli Distribution
----------------------

Binomial Distribution
---------------------

Fixed Distribution
------------------
