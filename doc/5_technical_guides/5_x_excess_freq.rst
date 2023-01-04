.. _p xs freq:

Excess Frequency Distributions
------------------------------

Given a ground-up claim count distribution :math:`N`, what is the
distribution of the number of claims exceeding a certain threshold? We
assume that severities are independent and identically distributed and
that the probability of exceeding the threshold is :math:`q`. Define an
indicator variable :math:`I` which takes value 0 if the claim is below
the threshold and the value 1 if it exceeds the threshold. Thus
:math:`\mathsf{Pr}(I=0)=p=1-q` and :math:`\mathsf{Pr}(I=1)=q`. Let :math:`M_N` be the
moment generating function of :math:`N` and :math:`N'` is the number of
claims in excess of the threshold. By definition we can express
:math:`N'` as an aggregate

.. math:: N'=I_1 + \cdots + I_N.

Thus the moment generating function of :math:`N'` is

.. math::

   M_{N'}(\zeta) &=M_N(\log(M_I(\zeta)))  \\
   &=M_N(\log(p+qe^{\zeta}))

Using indicator variables :math:`I` is called :math:`p`-thinning by Grandell.

Here are some examples.

Let :math:`N` be Poisson with mean :math:`n`. Then

.. math:: M_{N'}(\zeta) = \exp(n(p+qe^{\zeta}-1)) =  \exp(qn(e^{\zeta}-1))

so :math:`N'` is also Poisson with mean :math:`qn`—the simplest possible
result.

Next let :math:`N` be a :math:`G`-mixed Poisson. Thus

.. math::

   M_{N'}(\zeta)
   &= M_N(\log(p+qe^{\zeta}))  \\
   &= M_G(n(p+qe^{\zeta}-1))  \\
   &= M_G(nq(e^{\zeta}-1)).

Hence :math:`N'` is also a :math:`G`-mixed Poisson with lower underlying
claim count :math:`nq` in place of :math:`n`.

In particular, if :math:`N` has a negative binomial with parameters
:math:`P` and :math:`c` (mean :math:`cP`, :math:`Q=1+P`, moment
generating function :math:`M_N(\zeta)=(Q-Pe^{\zeta})^{-1/c}`), then
:math:`N'` has parameters :math:`qP` and :math:`c`. If :math:`N` has a
Poisson-inverse Gaussian distribution with parameters :math:`\mu` and
:math:`\beta`, so

.. math:: M_N(\zeta)=\exp\left(-\mu(\sqrt{1+2\beta(e^{\zeta}-1)}-1)\right),

then :math:`N` is also Poisson inverse Gaussian with parameters
:math:`\mu q` and :math:`\beta q`.

In all cases the variance of :math:`N'` is lower than the variance of
:math:`N` and :math:`N'` is closer to Poisson than :math:`N` in the
sense that the variance to mean ratio has decreased. For the general
:math:`G`-mixed Poisson the ratio of variance to mean decreases from
:math:`1+cn` to :math:`1+cqn`. As :math:`q\to
0` the variance to mean ratio approaches :math:`1` and :math:`N'`
approaches a Poisson distribution. The fact that :math:`N'` becomes
Poisson is called the law of small numbers.

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
