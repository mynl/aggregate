.. this doc is not part of the documentation - it is for future use only


Multivariate Methods
=======================

**Objectives:** Multivariate distributions from shared mixing variables; 2 dimensional FFT techniques.

**Audience:** Users interested in directly estimating multivariate distributions.

**Prerequisites:** DecL, general use of ``aggregate``, probability.

**See also:**

**Contents:**

* :ref:`Helpful References`
* :ref:`Two Shortcomings`
* :ref:`mv neg multi`
* :ref:`Correlated Aggregate Distributions`

Helpful References
--------------------

.. * PIR chapter 14, 15.

.. _strat margin alloc:

Two Shortcomings
-------------------

``aggregate`` methods are fundamentally univariate. Only one loss is tracked through each calculation. As a result, there are several programs it is impossible to model directly:

* The net of an occurrence cat program with an aggregate limit (need to track the net and cession to know when the aggregate limit is exhausted).
* The total cession from a specific and agg program (need to track two ceded results).

These are dual problems. It is possible to model them using two dimensional FFTs.

.. _mv neg multi:

Negative Multinomial Distribution and Related Frequency Distributions
-----------------------------------------------------------------------

When we consider mixed Poisson distributions we often regard :math:`G`
as carrying inter-risk correlation, or more evocatively “contagion”,
information about weather, the state of the economy and inflation, gas
prices etc. Hence if we have two related frequency variables :math:`N_1`
and :math:`N_2` we should expect to use the same :math:`G` and produce a
bivariate mixed Poisson where, conditional on :math:`G=g`, :math:`N_i`
has a Poisson distribution with mean :math:`n_i g` and :math:`N_1` and
:math:`N_2` are conditionally independent. The MGF of such a
distribution will be

.. math::

   M(\zeta_1,\zeta_2)
   &= \mathsf{E}(e^{\zeta_1N_1+\zeta_2N_2}) \notag \\
   &= \mathsf{E}(\mathsf{E}(e^{\zeta_1N_1+\zeta_2N_2}|G ) ) \notag \\
   &= \mathsf{E}_G(\mathsf{E}(e^{\zeta_1N_1}|G ) \mathsf{E}(e^{\zeta_2N_2}|G ) ) \notag \\
   &= \mathsf{E}_G(\exp(G(n_1(e^{\zeta_1}-1)+n_2(e^{\zeta_2}-1)))) \notag \\
   &= M_G(n_1(e^{\zeta_1}-1) +  n_2(e^{\zeta_2}-1)).

For example, if :math:`G` is a gamma random variable with MGF

.. math:: M_G(\zeta) = (1-\beta \zeta)^{-k}

(mean :math:`k\beta`, variance :math:`k\beta^2`) we get a bivariate
frequency distribution with MGF

.. math::

   M(\zeta_1,\zeta_2) &= [1- \beta(n_1(e^{\zeta_1}-1) +
     n_2(e^{\zeta_2}-1))]^{-k} \notag \\
   &=[1+\beta \sum_i n_i -\beta \sum_i n_ie^{\zeta_i}]^{-k}  \notag \\
   &=(Q -\sum_i P_ie^{\zeta_i})^{-k}

where :math:`P_i=\beta n_i`, :math:`P=\sum_i P_i` and :math:`Q=1+P`.
Equation (`[nmnmgf] <#nmnmgf>`__) is the moment generating function for
a negative multinomial distribution, as defined in Johnson, Kotz and
Kemp. The negative multinomial distribution
has positively correlated marginals as expected given its construction
with a common contagion :math:`G`.

The form of the moment generating function for negative multinomial
distribution can be generalized allowing us to construct multivariate
frequency distributions :math:`(N_1,\dots,N_t)` where

#. Each :math:`N_i` is a negative binomial.

#. The sum :math:`N_1+\cdots + N_t` under the multivariate distribution
   is also negative binomial. (In general, the sum of independent
   negative binomials will not be negative binomial.)

#. The :math:`N_i` are correlated.

We will call such multivariate frequencies, with common mixing
distributions, :math:`G`-mixed multivariate Poisson distributions.

Evolution of Claims Over Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is an application of the NMN distribution. If :math:`A` is an
aggregate distribution representing ultimate losses we may want to
determine a decomposition :math:`A=\sum_t D_t` into a sum of losses paid
at time :math:`t` for :math:`t=1,\dots,T`.

If :math:`A=X_1+\cdots +X_N` has a compound Poisson distribution then
such a decomposition is easy to arrange. Let :math:`\pi_t` be the
expected proportion of ultimate losses paid at time :math:`t`, so
:math:`\sum_{t=1}^{t=T} \pi_t=1`. By definition we mean

.. math:: \mathsf{E}(D_t) = \pi_t \mathsf{E}(A).\label{meandt}

(Equation (`[meandt] <#meandt>`__) is a different assumption to

.. math::

   \mathsf{E}(D_{t})=\pi_t\mathsf{E}(A|\text{information available at $t-1$})=\pi_t
   A_{t-1},

which is closer to the problem actually faced by the reserving actuary.
Our :math:`\pi_t`\ ’s are prior estimates assumed known at time 0. These
types of differences have interesting implications for actuarial methods
and they are explored further in Mack.) Now we
seek a decomposition

.. math:: A=D_1+D_2+\cdots+D_T\label{decomp}

but we know only (`[meandt] <#meandt>`__). The simplest approach is to
assume that severity :math:`X` is independent of time and that
:math:`\pi_t n` of the total :math:`n` claims are paid at time
:math:`t`. If we further assume that the number of claims paid at time
:math:`t` is also Poisson, then the moment generating function of
:math:`D_1+\cdots+D_T` is given by

.. math::

   M_{D_1+\cdots+D_T}(\zeta)
   &= \prod_t \exp( \pi_t n(M_X(\zeta)-1)) \notag \\
   & = \exp(n(\sum_t \pi_t M_X(\zeta)-1)) \notag \\
   &= \exp(n(M_X(\zeta)-1)) \notag \\
   & = M_A(\zeta).

Thus we have a very simple decomposition for (`[decomp] <#decomp>`__):
the individual :math:`D_t` are independent compound Poisson variables
with expected claim count :math:`\pi_t n` and severity distribution
:math:`X`.

Moving one step further, it is often observed in practice that average
severity increases with :math:`t` so the assumption that :math:`X` is
fixed for all :math:`t` is unrealistic. It may be better to assume that
losses which close at time :math:`t` are samples of a random variable
:math:`X_t`. As above, we assume that the expected number of such losses
is :math:`\pi_t' n` where :math:`n` is the expected ultimate number of
claims, and :math:`\pi_t'` adjusts the original :math:`\pi_t` for the
difference in average severity :math:`\mathsf{E}(X)` vs. :math:`\mathsf{E}(X_t)`. Now

.. math::

   M_{D_1+\cdots+D_T}(\zeta)
   &= \prod_t \exp( \pi_t' n(M_{X_t}(\zeta)-1)) \notag \\
   & = \exp(n(\sum_t \pi_t' M_{X_t}(\zeta)-1)) \notag \\
   &= \exp(n(M_{X'}(\zeta)-1)) \notag \\
   & = M_A(\zeta)

where :math:`X'` is a mixture of the :math:`X_t` with weights
:math:`\pi_t'`. Equation (`[CPDecomp2] <#CPDecomp2>`__) is a standard
result in actuarial science, see Bowers et al.

If we try to replicate the compound Poisson argument using a negative
binomial distribution for :math:`N` we will clearly fail. However if
:math:`X` is defined as a mixture of :math:`X_t` with weights
:math:`\pi_t`, as before, then we can write

.. math::

   M_{D_1,\dots,D_T}(\zeta_1,\dots,\zeta_T)
   = (Q-\sum_t P\pi_t M_{X_t}(\zeta_t))^{-k}\label{nmn1}

and so

.. math::

   M_A(\zeta) =M_{D_1,\dots,D_T}(\zeta,\dots,\zeta)
   = (Q-\sum_t P_t M_{X_t}(\zeta))^{-k}=(Q-PM_X(\zeta))^{-k}
   \label{nmn2}

where :math:`P_t:=\pi_t P`. Equation (`[nmn1] <#nmn1>`__) is the MGF for
a negative multinomial distribution, as defined in the previous section
and Johnson, Kotz and Kemp. As we have seen
the negative multinomial distribution has positively correlated
marginals, in line with our prior notions of liability dynamics. It
therefore provides a good model for the decomposition of ultimate losses
into losses paid each period.

Related Multivariate Frequency Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can use the same trick with other mixing distributions than the
gamma. The Poisson inverse Gaussian (PIG) distribution is an inverse
Gaussian mixture of Poissons, just as the negative binomial distribution
is a gamma mixture. The MGF is

.. math:: M(\zeta) = \exp(-\tau (\sqrt{1+\beta(e^{\zeta}-1)}-1)). \label{pig-pgf}

The mean is :math:`\tau\beta` and the variance is
:math:`\tau\beta(1+\beta)`. We can define a multivariate PIG (MPIG) by

.. math::

   M(\zeta_1,\dots,\zeta_T) =
   \exp(-\tau (\sqrt{1+\sum\beta_i(e^{\zeta_i}-1)}-1)). \label{mpig-pgf}

Sichel’s distribution is an generalized-inverse Gaussian mixture of
Poissons. The MGF is

.. math::

   M(\zeta) = \frac{K_{\gamma}(\omega\sqrt{1-2\beta(e^{\zeta}-1)})}{
   K_{\gamma}(\omega)(1-2\beta(e^{\zeta}-1))^{\gamma/2}}.

The mean and variance are given in Johnson, Kotz and Kemp
[page 456]. Clearly we can apply the same techniques to get another
multivariate frequency distribution.

The Poisson-Pascal distribution is a Poisson-stopped sum of negative
binomials. It has moment generating function

.. math:: M(\zeta) = \exp(\theta ((1-P(e^{\zeta}-1))^{-k}-1))

and so will also yield another multivariate family. The mean and
variance are given by

.. math::

     \mu = \theta kP \\
     \mu_2 = \theta kP(Q+kP).

.. _multiFreq:

Excess count interpretation of :math:`G`-mixed multivariate Poisson distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reader has probably realized that a :math:`G`-mixed multivariate
Poisson seems closely related to a single :math:`G`-mixed Poisson and a
series of indicator variables, combining results from the previous
sub-sections with Section `1.6 <#excessCounts>`__. Let :math:`N` be
:math:`G`-mixed Poisson with parameter :math:`n` and :math:`\mathsf{var}(G)=c`.
Let :math:`(N_1,N_2)` be :math:`G`-mixed bivariate Poisson with
parameters :math:`n_1` and :math:`n_2` and the same :math:`G`, so the
MGF of :math:`(N_1,N_2)` is

.. math::

   M_1(\zeta_1,\zeta_2)=M_G(n_1(e^{\zeta_1}-1)+n_2(e^{\zeta_2}-1)).

Finally let :math:`(I,J)` be a bivariate distribution supported on
:math:`\{0,1\}\times\{0,1\}` with

.. math::

   \mathsf{Pr}(I=0,J=0) &= p_{00} \\
   \mathsf{Pr}(I=1,J=0) &= p_{10} \\
   \mathsf{Pr}(I=0,J=1) &= p_{01} \\
   \mathsf{Pr}(I=1,J=1) &= p_{11}

and :math:`\sum p_{ij}=1`.

We can define a new bivariate distribution from :math:`(I,J)` and
:math:`N` as

.. math::

   (M_1,M_2)=(I_1,J_1)+\cdots +(I_N,J_N).

The MGF of :math:`(M_1,M_2)` is

.. math::

   M_2(\zeta_1,\zeta_2)=M_G(n(p_{11}e^{\zeta_1+\zeta_2} +
   p_{10}e^{\zeta_1} + p_{01}e^{\zeta_2} + p_{00}).

Thus, if :math:`p_{11}=0` we see the single-frequency sum of the
bivariate :math:`(M_1,M_2)` is actually a :math:`G`-mixed bivariate
Poisson. If :math:`p_{00}=0` then :math:`n=n_1+n_2`, otherwise
:math:`(1-p_{00})n=n_1+n_2` and there are some extraneous “zero” claims.
However, if :math:`p_{11}\not=0` then the single frequency sum is not a
:math:`G`-mixed bivariate Poisson.

Here is an interesting interpretation and application of :math:`(I,J)`.
We can regard :math:`I` as an indicator of whether a claim has been
reported at time :math:`t` and :math:`J` and indicator of whether the
claim is closed. Then

.. math::

   \mathsf{Pr}(I=0,J=0) &=\text{meaningless} \\
   \mathsf{Pr}(I=1,J=0) &=\text{reported claim which closes without payment} \\
   \mathsf{Pr}(I=0,J=1) &=\text{claim not yet reported which closes with payment} \\
   \mathsf{Pr}(I=1,J=1) &=\text{claim reported and closed with payment}.

Combining with a distribution :math:`N` of ultimate claims we can use
(`[singleFreq] <#singleFreq>`__) to produce
:math:`(M_1,M_2)=(I_1+\cdots+I_N,J_1+\cdots+J_N)`—a bivariate
distribution of (claims reported at time :math:`t`, ultimate number of
claims)! Note the value :math:`(0,0)` is a meaningless annoyance (it
scales :math:`n`) and we assume :math:`p_{00}=0`. The three other
parameters can easily be estimated using standard actuarial methods.

Given such a bivariate and a known number of claims reported we can
produce a posterior distribution of ultimate claims. Furthermore, in all
these techniques we can extend the simple count indicators :math:`(I,J)`
to be the distribution of case incurred losses and ultimate losses. Then
we would get a bivariate distribution of case incurred to date and
ultimate losses. I believe there is a lot of useful information that
could be wrought from these methods and that they deserve further study.
They naturually give confidence intervals on reserve ranges, for
example.

We end with a numerical example illustrating the theory we have
developed and showing another possible application. Rather than
interpreting :math:`p_{ij}` as reported and ultimate claims we could
interpret them as claims from line A and line B, where there is some
expectation these claim would be correlated. For example A could be auto
liability and B workers compensation for a trucking insured. Let
:math:`c=0.02` be the common contagion and :math:`n=250`. Then let

.. math::

   \mathsf{Pr}(I=0,J=0) &= 0 \\
   \mathsf{Pr}(I=1,J=0) &= 0.45 \\
   \mathsf{Pr}(I=0,J=1) &= 0.05 \\
   \mathsf{Pr}(I=1,J=1) &= 0.50.

We interpret :math:`I` as indicating a workers compensation claim and
:math:`J` as indicating an auto liability claim. The distribution says
that when there is an auto liability claim (:math:`J=1`) there is almost
always an injury to the driver, resulting in a workers compensation
claim (:math:`I=1`). However, there are many situations where the driver
is injured but there is no liability claim—such as back injuries.
Overall we expect :math:`250(0.45+0.50)=237.5` workers compensation
claims and :math:`250(0.05+0.5)=137.5` auto liability claims and 250
occurrences.

We will consider the single-frequency bivariate distribution and the
negative multinomial. We have seen that the negative multinomial
distribution will be slightly different because :math:`p_{11}\not=0`.
The appropriate parameters are :math:`n_1=250(p_{10}+p_{11})=237.5` and
:math:`n_1=250(p_{01}+p_{11})=137.5`. Figure `1.1 <#fig:NMNContours>`__
shows the negative multinomial bivariate (top plot) and the
single-frequency bivariate aggregate of :math:`(I,J)` (bottom plot).
Because of the correlation between :math:`I` and :math:`J`,
:math:`p_{11}=0.5`, the lower plot shows more correlation in aggregates
and the conditional distributions have less dispersion. Figure
`1.2 <#fig:NMNMarginals>`__ shows the two marginal distributions, which
are negative binomial :math:`c=0.02` and mean 237.5 and 137.5
respectively, the sum of these two variables assuming they are
independent (labelled “independent sum”), the sum assuming the negative
multinomial joint distribution (“NMN Sum”) which is identical to a
negative binomial with :math:`c=0.02` and mean :math:`350=237.5+137.5`,
the total number of claims from both lines, and finally, the sum with
dependent :math:`(I,J)` (“bivariate sum”). The last sum is not the same
as the negative binomial sum; it has a different MGF.

Figure `1.2 <#fig:NMNMarginals>`__ also shows the difference between the
sum of two independent negative binomials with means :math:`n_1` and
:math:`n_2` and contagion :math:`c` and a negative binomial with mean
:math:`n_1+n_2` and contagion :math:`c`. The difference is clearly very
material in the tails and is an object lesson to modelers who subdivide
their book into homogeneous parts but then add up those parts assuming
independence. Such an approach is *wrong* and must be avoided.

As the contagion :math:`c` increases the effects of :math:`G`-mixing
dominate and the difference between the two bivariate distributions
decreases, and conversely as :math:`c` decreases to zero the effect is
magnified. The value :math:`c=0.02` was selected to balance these two
effects.

.. figure:: C:/SteveBase/papers/CAS_WP/FinalICExhibits/NMNvsOtherFreqContours.pdf

   Comparison of negative multinomial (top) and single frequency
   bivariate claim count (bottom) bivariate distributions.


.. figure:: C:/SteveBase/papers/CAS_WP/FinalICExhibits/NMNvsOtherFreqMarginals.pdf

   Comparison of negative multinomial and single frequency bivariate
   claim count marginal and total distributions.




Correlated Aggregate Distributions
----------------------------------

Here we extend some of the ideas in Section `1.7.3 <#multiFreq>`__ from
plain frequency distributions to aggregate distributions. Begin with
bivariate aggregate distributions. There are two different situations
which commonly arise. First we could model a bivariate severity
distribution and a univariate count distribution:

.. math:: (A,B)=(X_1,Y_1)+\cdots+(X_N, Y_N).

Equation (`[modelone] <#modelone>`__) arises naturally as the
distribution of losses and allocated expense, ceded and retained losses,
reported and ultimate claims, and in many other situations. Secondly we
could model

.. math:: (A,B)=(X_1+\cdots+X_M, Y_1+\cdots+Y_N)

where :math:`X_i` and :math:`Y_j` are independent severities and
:math:`(M,N)` is a bivariate frequency distribution.
(`[modeltwo] <#modeltwo>`__) could be used to model losses in a clash
policy.

We will use the following notation. :math:`A=X_1+\cdots+X_M` and
:math:`B=Y_1+\cdots+Y_N` are two aggregate distributions, with
:math:`X_i` iid and :math:`Y_j` iid, but neither :math:`X` and :math:`Y`
nor :math:`M` and :math:`N` necessarily independent. Let :math:`\mathsf{E}(X)=x`
and :math:`\mathsf{E}(Y)=y`, :math:`\mathsf{var}(X)=v_x` and :math:`\mathsf{var}(Y)=v_y`. Let
:math:`\mathsf{E}(M)=m`, :math:`\mathsf{E}(N)=n`, :math:`c` be the contagion of
:math:`M` and :math:`d` that of :math:`N`. Hence :math:`\mathsf{var}(M)=m(1+cm)`
and :math:`\mathsf{var}(N)=n(1+dn)`.

Will now calculate the correlation coefficient between :math:`A` and
:math:`B` in four situations.

Correlated Severities, Single Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assume that the bivariate severity distribution :math:`(X,Y)` has moment
generating function :math:`M_{(X,Y)}(\zeta,\tau)`. Also assume that the
claim count distribution :math:`N` is a :math:`G`-mixed Poisson. Then,
just as for univariate aggregate distributions, the MGF of the bivariate
aggregate :math:`(A,B)` is

.. math:: M_{(A,B)}(\zeta,\tau)= M_G( n(M_{(X,Y)}(\zeta,\tau)-1)).\label{bivMGF}

Therefore, since :math:`\mathsf{E}(G)=1` and :math:`\mathsf{E}(G^2)=1+c`,

.. math::

   E(AB) &= \frac{\partial^2
     M_{(A,B)}}{\partial\zeta\partial\tau}\Big\vert_{(0,0)} \notag \\
   &= M_G''(0)n^2\frac{\partial M_{(X,Y)}}{\partial\zeta}
   \frac{\partial M_{(X,Y)}}{\partial\zeta} + M_G'(0)n
   \frac{\partial^2 M_{(X,Y)}}{\partial\zeta\partial\tau} \notag \\
   &=(1+c)n^2xy + n\mathsf{E}(XY) \notag \\
   &=(1+c)n^2xy + n\mathsf{cov}(X,Y) + nxy.

The value of :math:`\mathsf{cov}(X,Y)` will depend on the particular bivarate
severity distribution.

For example, suppose that :math:`Z` represents ground up losses,
:math:`X` represents a retention to :math:`a` and :math:`Y` losses
excess of :math:`a` (per ground up claim), so :math:`Z=X+Y`. Then
:math:`(X,Y)` is a bivariate severity distribution. Since :math:`Y` is
zero when :math:`Z\le a` we have :math:`\mathsf{cov}(X,Y)=(a-x)y`.

Bivariate Frequency
~~~~~~~~~~~~~~~~~~~

The second method for generating correlated aggregate distributions is
to use a bivariate frequency distribution. So, suppose :math:`(M,N)` has
a :math:`G`-mixed bivariate Poisson distribution. The variance of
:math:`A` is given by Equation (`[varAgg] <#varAgg>`__). To compute the
covariance of :math:`A` and :math:`B` write the bivariate MGF of
:math:`(A,B)` as

.. math::

   M_{(A,B)}(\zeta,\eta)=M(\zeta,\eta)=M_G(m(M_X(\zeta)-1)
   +n(M_Y(\eta)-1))=M_G(\psi(\zeta,\eta))

where the last equality defines :math:`\psi`. Then, evaluating at the
partial derivatives at zero, we get

.. math::

   \mathsf{E}(AB) &= \frac{\partial^2 M}{\partial\zeta\partial\eta} \notag \\
   &= \frac{\partial^2 M_G}{\partial t^2}
   \frac{\partial \psi}{\partial\zeta} \frac{\partial \psi}{\partial\eta}
   + \frac{\partial M_G}{\partial t}
   \frac{\partial^2 \psi}{\partial\zeta\partial\eta}  \notag \\
   &= (1+c)mxny.

Hence

.. math:: \mathsf{cov}(A,B)=\mathsf{E}(AB)-\mathsf{E}(A)\mathsf{E}(B)=cmnxy.




Parameter Uncertainty
~~~~~~~~~~~~~~~~~~~~~

It is common for actuaries to work with point estimates as though they
are certain. In reality there is a range around any point estimate. We
now work through one possible implication of such parameter uncertainty.
We will model :math:`\mathsf{E}[A]=R` and :math:`\mathsf{E}[B]=S` with :math:`R` and
:math:`S` correlated random variables, and :math:`A` and :math:`B`
conditionally independent given :math:`R` and :math:`S`. We will assume
for simplicity that the severities :math:`X` and :math:`Y` are fixed and
that the uncertainty all comes from claim counts. The reader can extend
the model to varying severities as an exercise. :math:`R` and :math:`S`
pick up uncertainty in items like the trend factor, tail factors and
other economic variables, as well as the natural correlation induced
through actuarial methods such as the Bornheutter-Ferguson.

Suppose :math:`\mathsf{E}[R]=r`, :math:`\mathsf{E}[S]=s`, :math:`\mathsf{var}(R)=v_r`,
:math:`\mathsf{var}(S)=v_s` and let :math:`\rho` be the correlation coefficient
between :math:`R` and :math:`S`.

By (`[varAgg] <#varAgg>`__) the conditional distribution of :math:`A \mid R`
is a mixed compound Poisson distribution with expected claim count
:math:`R/x` and contagion :math:`c`. Therefore the conditional variance
is

.. math::

   \mathsf{var}(A \mid R)
   &= \mathsf{E}[M \mid R]\mathsf{var}(X)+\mathsf{var}(M \mid R)\mathsf{E}[X]^2  \\
   &= R/x v_x + R/x(1+cR/x) x^2  \\
   &= xR(1+ v_x/x^2) + cR^2,

and the unconditional variance of :math:`A` is

.. math::

   \mathsf{var}(A)
   &= \mathsf{E}[\mathsf{var}(A \mid R)] + \mathsf{var}(\mathsf{E}[A \mid R])  \\
   &= \mathsf{E}[xR(v_x/x^2+1)+cR^2] + \mathsf{var}(R)  \\
   &=  xr(v_x/x^2+1)+c(v_r+r^2) + v_r.

Next, because :math:`A` and :math:`B` are conditionally independent
given :math:`R` and :math:`S`,

.. math::

   \mathsf{cov}(A,B)
   &= \mathsf{E}[\mathsf{cov}(A,B \mid R,S)] + \mathsf{cov}(\mathsf{E}[A \mid R], \mathsf{E}[B \mid S])  \\
   &= \mathsf{cov}(R, S).\label{simpleCov}

Note Equation (`[simpleCov] <#simpleCov>`__) is only true if we assume
:math:`A\not=B`.

Parameter Uncertainty and Bivariate Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, suppose :math:`\mathsf{E}[A]=R`, :math:`\mathsf{E}[B]=S` with :math:`R` and
:math:`S` correlated parameters *and* conditional on :math:`(R,S)`
suppose that :math:`(M,N)` has a :math:`G`-mixed bivariate Poisson
distribution. By (`[covMNM] <#covMNM>`__) :math:`\mathsf{cov}(A,B \mid R,S)=cRS`. The
unconditional variances are as given in (`[varA] <#varA>`__). The
covariance term is

.. math::

   \mathsf{cov}(A,B)
   &= \mathsf{E}[\mathsf{cov}(A,B \mid R,S)] + \mathsf{cov}(\mathsf{E}[A \mid R], \mathsf{E}[B \mid S])  \\
   &= c\mathsf{E}[RS]  + \mathsf{cov}(R,S)  \\
   &= (1+c)\mathsf{cov}(R,S) + crs  \\
   &= \rho \sqrt{v_rv_s}(1+c)+crs.
