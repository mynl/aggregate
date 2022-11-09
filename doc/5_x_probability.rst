.. _probability:

Probability Background
======================

Severity Distributions
-----------------------


Computing moments
~~~~~~~~~~~~~~~~~~

Higher moments of a layer can be computed

.. math::

   \mathsf E[((X-a)^+ \wedge l)^n]
   &= \int_a^{a+l} (x-a)^n f(x)\,dx + l^nS(a+l) \\
   &= \sum_{k=0}^n (-1)^k \binom{n}{k} a^{n-k} \int_a^{a+l} x^k f(x)\,dx + l^nS(a+l) \\
   &= \sum_{k=0}^n (-1)^k \binom{n}{k} a^{n-k} \left(\mathsf E(k; a+l) - \mathsf E(k; a)\right)+ l^nS(a+l)

where

.. math::


   \mathsf E(k; a) = \int_0^a x^kf(x)\,dx

is the partial expectation function.

Lognormal
"""""""""

For the lognormal, the trick for higher moments is to observe that if
:math:`X` is :math:`\mathit{lognormal}(\mu,\sigma)` then :math:`X^k` is
:math:`\mathit{lognormal}(k\mu, k\sigma)`. The formula for partial
expectations of the lognormal is easy to compute by substitution, giving

.. math::

   \mathsf E(k, a) = \exp(k\mu + k^2\sigma^2/2)\Phi\left( \frac{\log x -\mu - k\sigma^2}{\sigma} \right)

Densities of the form :math:`f(x)=x^\alpha c(\alpha)g(x)`
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. math::

   \mathsf E(k, a)
   &= \int_0^a x^k x^\alpha c(\alpha)g(x) \,dx \\
   &= \frac{c(\alpha)}{c(n+\alpha)}\int_0^a x^{k+\alpha} c(k+\alpha)g(x) \,dx \\
   &= \frac{c(\alpha)}{c(n+\alpha)}F_{k+\alpha}(a)

are easy to express in terms of the distribution function. This is a broad class including the gamma, XXXX.

Pareto
"""""""

An easy integral computation, substitute :math:`y=\lambda + x` to express in powers of :math:`y`.

Frequency Distributions
------------------------

Mixed Frequency Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. from 05.md

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
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Per Actuarial Geometry, if :math:`\nu` is the CV of :math:`G` then the
:math:`\nu` equals the asymptotic coefficient of variation for any
:math:`G`-mixed compound Poisson distribution whose variance exists. The
variance will exist iff the variance of the severity term exists.

Gamma Mixing
"""""""""""""

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
"""""""""""""""""""""""""

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
"""""""""""""""""""""""""""""""""

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
""""""""""""""""""""""""""""""""""""""

Bernoulli Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

Binomial Distribution
~~~~~~~~~~~~~~~~~~~~~~

Fixed Distribution
~~~~~~~~~~~~~~~~~~~

The :math:`(a,b,0)` and :math:`(a,b,1)` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Aggregate Distributions
------------------------

aka compound distributions
