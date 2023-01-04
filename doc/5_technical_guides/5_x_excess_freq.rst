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

Using indicator variables :math:`I` is called :math:`p`-thinning by
:cite:t:`Grandell1997`.

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
