.. _5_x_probability:

Probability Background
======================

**Objectives:** Statement and limited explanation of important probability concepts that underlie ``aggregate`` calculations.

**Audience:**

**Prerequisites:** Knowledge of probability and calculus (real analysis).

**See also:**

**Notation:** The variance of a random variable :math:`X` is :math:`\mathsf{var}(X)=\mathsf{E}[X^2]-\mathsf{E}[X]^2`. The standard deviation is :math:`\sigma(X)=\sqrt{\mathsf{var}(X)}`. The coefficient of variation (CV) of :math:`X` is :math:`\mathsf{CV}(X)=\sigma(X)/\mathsf{E}[X]`. The skewness of :math:`X` is :math:`\mathsf{E}[(X-\mathsf{E}[X])^3]/\sigma(X)^3`.

**Contents:**

* :ref:`p hr`
* :ref:`p types`
* :ref:`p mgfs`
* :ref:`p severity`
* :ref:`p frequency`
* :ref:`p aggregate`
* :ref:`p sln sg`
* :ref:`p xs freq`
* :ref:`p sev irrel`

.. _p hr:


Helpful References
--------------------

* :cite:t:`feller71`

.. _p types:

Types
------

.. todo::

  Documentation to follow.

.. _p mgfs:

Moment Generating Functions
---------------------------

The moment generating function of a random variable :math:`X` is defined
as

.. math:: M_X(\zeta)=\mathsf{E}[\exp(\zeta X)].

The moment generating function is related to the characteristic function
of :math:`X` which is defined as :math:`\phi_X(\zeta)=\mathsf{E}[\exp(i\zeta
X)]=M_X(i\zeta)`. :math:`\phi` is guaranteed to converge for all real
:math:`\zeta` and so is preferred in certain situations.

Moment generating functions get their name from the fundamental property
that

.. math:: \frac{\partial^n M_X}{\partial \zeta^n}\Big\vert_{\zeta=0}=\mathsf{E}[X^n]

for all positive integers :math:`n` provided the differential exists.

Let :math:`F` be the distribution function of :math:`X`. Feller
[Vol II Section XVII.2a] shows that if :math:`F` has
expectation :math:`\mu` then :math:`\phi`, the characteristic function
of :math:`F`, has a derivative :math:`\phi'` and :math:`\phi'(0)=i\mu`.
However the converse is false. Pitman proved that the following are equivalent.

#. :math:`\phi'(0)=i\mu`.

#. As :math:`t\to\infty`, :math:`t(1-F(t)+F(-t))\to 0` and

   .. math::

      \label{feller1}
      \int_t^{-t} xdF(x) \to \mu,

   where :math:`F(-t):=\lim F(s)` as :math:`s\uparrow t`.

#. The average :math:`(X_1+\cdots+X_n)/n` tends in probability to
   :math:`\mu`, that is
   :math:`\mathsf{Pr}(| (X_1+\cdots +X_n)/n-\mu|>\epsilon)\to 0` as :math:`n\to\infty`.

The condition for the limit in 2 to
exist is weaker than the requirement that :math:`\mathsf{E}[X]` exists if
:math:`X` is supported on the whole real line. For the expectation to
exist requires :math:`\int_{-\infty}^{\infty} xdF(x)` exists which means
:math:`\lim_{t\to-\infty}\lim_{s\to\infty}\int_t^s xdF(x)`.

The moment generating function of a bivariate distribution
:math:`(X_1,X_2)` is defined as

.. math:: M_{X_1,X_2}(\zeta_1,\zeta_2)=\mathsf{E}[\exp(\zeta_1 X_1+\zeta_2 X_2)].

It has the property that

.. math::

   \frac{\partial^{m+n} M_{X_1,X_2}}{\partial \zeta_1^m\partial
     \zeta_2^n}\Big\vert_{(0,0)} =\mathsf{E}[X_1^mX_2^n]

for all positive integers :math:`n,m`.

The MGF of a normal variable with mean :math:`\mu` and standard
deviation :math:`\sigma` is

.. math:: M(\zeta)=\exp(\mu\zeta+\sigma^2\zeta^2/2).

The MGF of a Poisson variable with mean :math:`n` is

.. math:: M(\zeta)=\exp(n(e^{\zeta}-1)).

See any standard text on probability for more information on moment
generating functions, characteristic functions and modes of convergence.

.. _p severity:

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

An easy integral computation, substitute :math:`y=\lambda + x` to express in powers of :math:`y`:

.. math::
  \mathsf E(k, a)
   &= \int_0^a \alpha  x^k \frac{\lambda^\alpha}{(\lambda + x)^{\alpha+1}}\,dx \\
   &= \int_\lambda^{\lambda + a} \alpha\lambda^\alpha \frac{(y-\lambda)^k}{y^{\alpha+1}}\,dy \\
   &= \sum_{i=0}^k (-1)^{k-i} \alpha\lambda^\alpha \binom{k}{i}   \int_\lambda^{\lambda + a}  y^{i-\alpha-1} \lambda^{k-i}\,dy \\
   &= \sum_{i=0}^k (-1)^{k-i} \alpha\lambda^{\alpha+k-i} \binom{k}{i}  \frac{y^{i-\alpha}}{i-\alpha}\big|_\lambda^{\lambda + a}.

.. _p frequency:

Frequency Distributions
------------------------

Bernoulli Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

Binomial Distribution
~~~~~~~~~~~~~~~~~~~~~~

Fixed Distribution
~~~~~~~~~~~~~~~~~~~

.. _mixed frequency distributions:

Mixed Frequency Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
:math:`Var(G) = a\theta^2` and :math:`E(G-E(G))^3 = 2a\theta^3`. The skewness of :math:`G` is :math:`\gamma = 2/\sqrt(a) = 2\nu`.

Applying the general formula for the third central moment of :math:`N`
we get an expression for the skewness

.. math::

   \text{skew}(N) = \frac{n^3(\gamma -3c -1) + n^2(3c+2) + n}{(n(1+cn))^{3/2}}.

The corresponding MGF of the gamma is
:math:`M_G(\zeta) = (1-\theta\zeta)^{-a}`.

The gamma and negative binomial occur in the literature with many different
parameterizations. The main ones are shown in the next three tables.

.. list-table:: Parameterizations of the Gamma Distribution
  :widths: 20 20 20 20 20
  :header-rows: 1

  * - **Model**
    - **Density**
    - **MGF**
    - **Mean**
    - **Var**
  * - (a) :math:`\alpha`, :math:`\beta`
    - :math:`\frac{\textstyle x^{\alpha-1}e^{-x/\beta}}{\textstyle\beta^{\alpha}\Gamma(\alpha)}`
    - :math:`(1-\beta t)^{-\alpha}`
    - :math:`\alpha\beta`
    - :math:`\alpha\beta^2`
  * - (b) :math:`\alpha`, :math:`\beta`
    - :math:`\frac{\textstyle x^{\alpha-1}\beta^{\alpha}e^{-x\beta}}{\textstyle\Gamma(\alpha)}`
    - :math:`(1-t/\beta)^{-\alpha}`
    - :math:`\alpha/\beta`
    - :math:`\alpha/\beta^2`
  * - (c) :math:`\alpha`, :math:`\theta`
    - :math:`\frac{\textstyle x^{\alpha-1}e^{-x/\theta}}{\textstyle \theta^{\alpha}\Gamma(\alpha)}`
    - :math:`(1-t\theta)^{-\alpha}`
    - :math:`\alpha\theta`
    - :math:`\alpha\theta^2`


Model (a) is used by Microsoft Excel, Wang, and Johnson et al. [Chapter 17]. Model (b) is used by Bowers et al. Model (c) is used by Klugman, Panjer and Willmot in the Loss Models text. Obviously model (c) is just model (a) with a change of notation.


.. list-table:: Parameterizations of the Negative Binomial Distribution
  :widths: 20 20 20 20 20
  :header-rows: 1

  * - **Model**
    - **Density**
    - **MGF**
    - **Mean**
    - **Var**
  * - (a) :math:`\alpha`, :math:`\beta`
    - :math:`\binom{\textstyle\alpha+x-1}{\textstyle x} \left(\frac{\textstyle \beta}{\textstyle 1+\beta}\right)^x \left(\frac{\textstyle 1}{\textstyle 1+\beta}\right)^{\alpha}`
    - :math:`(1-\beta(e^t-1))^{-\alpha}`
    - :math:`\alpha\beta`
    - :math:`\alpha\beta^2`
  * - (b) :math:`P`, :math:`k`
    - :math:`\binom{\textstyle k+x-1}{\textstyle x} \left(\frac{\textstyle P}{\textstyle Q}\right)^x \left(\frac{\textstyle Q-P}{\textstyle Q}\right)^k`
    - :math:`(Q-Pe^t)^{-k}`
    - :math:`kP`
    - :math:`kPQ`
  * - (c) :math:`p`, :math:`r>0`
    - :math:`\textstyle\binom{\textstyle r+x-1}{\textstyle x} p^rq^x`
    - :math:`\frac{\textstyle p^r}{\textstyle (1-qe^s)^r}`
    - :math:`rq/p`
    - :math:`rq/p^2`


Note that :math:`Q=P+1`, :math:`q=1-p`, :math:`0<p<1` and :math:`r>0`, and :math:`P=1/(\beta+1)`.


.. list-table:: Fitting the Negative Binomial Distribution
  :widths: 10 18 18 18 18 18
  :header-rows: 1

  * - **Model**
    - **Parameters**
    - **VM Scale**
    - **VM Shape**
    - **Ctg Scale**
    - **Ctg Shape**
  * - (a)
    - :math:`r`, :math:`\beta`
    - :math:`r=m/(v-1)`
    - :math:`\beta=v-1`
    - :math:`r=1/c`
    - :math:`\beta=cn`
  * - (b)
    - :math:`k`, :math:`P`
    - :math:`k=m/(v-1)`
    - :math:`P=v-1`
    - :math:`k=1/c`
    - :math:`P=cn`
  * - (c)
    - :math:`r`, :math:`p`
    - :math:`r=m/(v-1)`
    - :math:`p=1/v`
    - :math:`r=1/c`
    - :math:`p=1/(1+cn)`


Model (a) is used by Wang and Loss Models, (b) by Johnson et al. [Chapter 5]
and (c) by Bowers et al. and Excel. In model (c) the parameter :math:`r` need
not be an integer because the binomial coefficient can be computed as

.. math:: \binom{r+x-1}{x}=\frac{\Gamma(r+x)}{\Gamma(r)x!},

an expression which is valid for all :math:`r`. The cumulative
distribution function of the negative binomial can be computed using the
cumulative distribution of the beta distribution. Using the model (c)
parameterization, if :math:`N` is negative binomial :math:`p,r` then

.. math::

   \mathsf{Pr}(N\le k)=\text{BETADIST}(p;r,k+1):=\frac{1}{B(r,k+1)}\int_0^p
   u^{r-1} (1-u)^{k} du

where :math:`B` is the complete beta function. See Johnson, Kotz and
Kemp [Eqn. 5.31] for a derivation. BETADIST is
the Excel beta cumulative distribution function.

The name negative binomial comes from an analogy with the binomial. A
binomial variable has parameters :math:`n` and :math:`p`, mean
:math:`np` and variance :math:`npq`, where :math:`p+q=1`. It is a sum of
:math:`n` independent Bernoulli variables :math:`B` where
:math:`\mathsf{Pr}(B=1)=p` and :math:`\mathsf{Pr}(B=0)=q=1-p`. The MGF for a binomial is
:math:`(q+pe^{\zeta})^n` and the probabilities are derived from the
binomial expansion of the MGF. By analogy the negative binomial can be
defined in terms of the negative binomial expansion of
:math:`(Q-Pe^{\zeta})^{-k}` where :math:`Q=1+P`, :math:`P>0` and
:math:`k>0`.

For the actuary there are two distinct ways of looking at the negative
binomial which give very different results and it is important to
understand these two views. First there is the contagion view, where the
mixing distribution :math:`G` has mean :math:`n` and variance :math:`c`
producing a negative binomial with mean :math:`n` and variance
:math:`n(1+cn)`. (In fact :math:`G` is a gamma with model (a) parameters
:math:`\alpha=r` and :math:`\beta=1/r`.) The word contagion was used by
Heckman and Meyers and is supposed to
indicate a “contagion” of claim propensity driven by common shock
uncertainty, such as claim inflation, economic activity, or weather.
Here the variance grows with the square of :math:`n` and the coefficient
of variation tends to :math:`\sqrt{c}>0` as :math:`n\to\infty`.
Secondly, one can consider an over-dispersed family of Poisson variables
with mean :math:`n` and variance :math:`vn` for some :math:`v>1`. We
call :math:`v` the variance multiplier. Now the coefficient of variation
tends to :math:`0` as :math:`n\to\infty`. The notion of over-dispersion
and its application in modeling is discussed in Clark and Thayer, and Verrall.

.. _prob variance mult:

The Variance Multiplier
"""""""""""""""""""""""""

The variance of a mixed Poisson equals :math:`n(1+cn)` where :math:`c` equals the variance of the mixing distribution. Thus the variance equals :math:`v=1+cn` times the mean :math:`n`, where :math:`v` is called the **variance multiplier**. The variance multiplier specification is used by some US rating bureaus. The dictionary to variance and mix CV is

.. math::

  c = (v-1) / n \\
  \mathit{cv} = \sqrt{(v-1)/n}.

The frequency for an excess layer attaching at :math:`a` equals :math:`nS(a)`. For fixed :math:`c`, the implied variance multiplier :math:`v=1+cnS(a)` decreases and the excess claim count distribution converges to a Poisson. This is an example of the law of small numbers.

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

The :math:`(a,b,0)` and :math:`(a,b,1)` Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _p aggregate:

Aggregate Distributions
-----------------------

Let :math:`A=X_1+\cdots +X_N` be an aggregate distribution, where
:math:`N` is the **frequency** component and  :math:`X_i` are iid **severity**
random variables.


Aggregate statistics: the mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mean of a sum equals the sum of the means. Let :math:`A = X_1 + \cdots + X_N`. If :math:`N=n` is fixed then :math:`\mathsf E[A] = n\mathsf E(X)`, because all :math:`\mathsf E[X_i]=\mathsf E[X]`. In general,

.. math::

    \mathsf E[A] = \mathsf E[X]\mathsf E[N]

by conditional probability.

Aggregate statistics: the variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The variance of a sum of independent random variables equals the sum of the variances.  If :math:`N=n` is fixed then :math:`\mathsf{Var}(A) = n\mathsf{Var}(X)` and :math:`\mathsf{Var}(N)=0`. If :math:`X=x` is fixed then :math:`\mathsf{Var}(A) = x^2\mathsf{Var}(N)` and :math:`\mathsf{Var}(X)=0`. Making the obvious associations :math:`n\leftrightarrow\mathsf E[N]`, :math:`x\leftrightarrow\mathsf E[X]` suggests

.. math::

    \mathsf{Var}(A) = \mathsf E[N]\mathsf{Var}(X) + \mathsf E[X]^2\mathsf{Var}(N).

Using conditional expectations and conditioning on the value of :math:`N` shows this  is the correct answer!

**Exercise.** Confirm the formulas for an aggregate mean and variance hold for the :ref:`Simple Example`.

Moment Generating Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the tower property of conditional expectations and the independence of :math:`N` and :math:`X_i` gives

.. math::

   M_A(\zeta)
   &= \mathsf{E}[\exp(\zeta(X_1+\cdots X_N))] \\
   &= \mathsf{E}[\mathsf{E}[\exp(\zeta(X_1+\cdots X_N)) \mid N]] \\
   &= \mathsf{E}[\mathsf{E}[\exp(\zeta X_1)^N]] \\
   &= \mathsf{E}[\mathsf{E}[\exp(\zeta X_1)]^N] \\
   &= M_N(\log(M_X(\zeta)))

Differentiating and using XXs formula, yields the moments of :math:`A`, see below.

The last expression is very important and underlies the use of FFTs to compute aggregate distributions.

Next, specialize to the case where :math:`A=X_1+\cdots +X_N` is an aggregate distribution with
:math:`N` a :math:`G`-mixed Poisson. Then

.. math::

   M_A(\zeta)
   &= \mathsf{E}[\exp(\zeta(X_1+\cdots X_N))]  \\
   &= \mathsf{E}[\mathsf{E}[\exp(\zeta(X_1+\cdots X_N)) \mid N]]  \\
   &= \mathsf{E}[\mathsf{E}[\exp(\zeta X_1)^N]]  \\
   &= \mathsf{E}[\mathsf{E}[M_X(\zeta)^N \mid G]]  \\
   &= \mathsf{E}[\exp(nG(M_X(\zeta)-1))]  \\
   &= M_G(n(M_X(\zeta)-1))

Thus

.. math:: \mathsf{E}[A]=M_A'(0)=n M_G'(0)M_X'(0)=n \mathsf{E}[X]

and

.. math::

   \mathsf{E}[A^2] &=M_A''(0)  \\
           &=  n^2 M_G''(0)M_X'(0)^2+n M_G'(0)M_X''(0) \\
           &= n^2\mathsf{E}[G^2]\mathsf{E}[X]^2+n\mathsf{E}[X^2].

Hence, using the fact that :math:`\mathsf{E}[G^2]=1+c`,

we get

.. math::

   \mathsf{var}(A) &= n^2\mathsf{E}[G^2]\mathsf{E}[X]^2+n\mathsf{E}[X^2] -
   n^2\mathsf{E}[X]^2  \\
           &=  n^2 c \mathsf{E}[X]^2+ n \mathsf{E}[X^2]  \\
           &=  (\mathsf{var}(N)-\mathsf{E}[N])\mathsf{E}[X]^2+\mathsf{E}[N]\mathsf{E}[X^2]  \\
           &=  \mathsf{var}(N)\mathsf{E}[X]^2+\mathsf{E}[N]\mathsf{var}(X).

Continuing along the same vein we get

.. math::

   \mathsf{E}[A^3]= & \mathsf{E}[N]\mathsf{E}[X^3]+\mathsf{E}[N^3]\mathsf{E}[X]^3+3\mathsf{E}[N^2]\mathsf{E}[X]\mathsf{E}[X^2] \\
    &-3\mathsf{E}[N]\mathsf{E}[X]\mathsf{E}[X^2] -3\mathsf{E}[N^2]\mathsf{E}[X]^3+2\mathsf{E}[N]\mathsf{E}[X]^3.

and so we can compute the skewness of :math:`A`, remembering that

.. math:: \mathsf{E}[(A-\mathsf{E}[A])^3]=\mathsf{E}[A^3]-3\mathsf{E}[A^2]\mathsf{E}[A]+2\mathsf{E}[A]^3.

Further moments can be computed using derivatives of the moment generating function.

Having computed the mean, CV and skewness of the aggregate using these
equations we can use the method of moments to fit a shifted lognormal or
shifted gamma distribution. We turn next to a description of these handy
distributions.

.. _shiftedLN:
.. _p sln sg:

Shifted Gamma and Lognormal Distributions
-----------------------------------------

The shifted gamma and shifted lognormal distributions are versatile
three parameter distributions whose method of moments parameters can be
conveniently computed by closed formula. The examples below show that
they also provide a very good approximation to aggregate loss
distributions. The shifted gamma approximation to an aggregate is
discussed in Bowers et al. Properties of
the shifted gamma and lognormal distributions, including the method of
moments fit parameters, are also shown in Daykin et al. [Chapter 3].

Let :math:`L` have a lognormal distribution. Then :math:`S=s\pm L` is a
shifted lognormal, where :math:`s` is a real number. Since :math:`s` can
be positive or negative and since :math:`L` can equal :math:`s+L` or
:math:`s-L`, the shifted lognormal can model distributions which are
positively or negatively skewed, as well as distributions supported on
the negative reals. The key facts about the shifted lognormal are shown
in Table `1.4 <#shiftedDist>`__. The variable :math:`\eta` is a solution
to the cubic equation

.. math:: \eta^3 + 3\eta  - \gamma=0

where :math:`\gamma` is the skewness.

Let :math:`G` have a gamma distribution. Then :math:`T=s\pm G` is a
shifted gamma distribution, where :math:`s` is a real number. Table
`1.1 <#tab:gammaInfo>`__ shows some common parametric forms for the
gamma distribution. The key facts about the shifted gamma distribution
are also shown in Table `1.4 <#shiftedDist>`__.

The exponential is a special case of the gamma where :math:`\alpha=1`.
The :math:`\chi^2` is a special case where :math:`\alpha=k/2` and
:math:`\beta = 2` in the Excel parameterization. The Pareto is a mixture
of exponentials where the mixing distribution is gamma.

.. table:: Shifted Gamma and Lognormal Distributions

   +----------------------+-----------------------------------+------------------------------------------+
   | **Item**             | **Shifted Gamma**                 | **Shifted Lognormal**                    |
   +======================+===================================+==========================================+
   | Parameters           | :math:`s`,                        | :math:`s`,                               |
   |                      | :math:`\alpha`,                   | :math:`\mu`,                             |
   |                      | :math:`\theta`                    | :math:`\sigma`                           |
   +----------------------+-----------------------------------+------------------------------------------+
   | Mean :math:`m`       | :math:`s+\alpha\theta`            | :math:`s+\exp(\mu+\sigma^2/2)`           |
   +----------------------+-----------------------------------+------------------------------------------+
   | Variance             | :math:`\alpha\theta^2`            | :math:`m^2\exp(\sigma^2-1)`              |
   +----------------------+-----------------------------------+------------------------------------------+
   | CV, :math:`\nu`      | :math:`\sqrt{\alpha}\beta/\gamma` | :math:`\exp((\sigma^2-1)/2)`             |
   +----------------------+-----------------------------------+------------------------------------------+
   | Skewness,            | :math:`2/\sqrt{\alpha}`           | :math:`\gamma=\nu(\nu^2+3)`              |
   +----------------------+-----------------------------------+------------------------------------------+
   | **Method of Moments  |                                   |                                          |
   | Parameters**         |                                   |                                          |
   +----------------------+-----------------------------------+------------------------------------------+
   | :math:`\eta`         | n/a                               | :math:`\eta=u-1/u`                       |
   |                      |                                   | where                                    |
   +----------------------+-----------------------------------+------------------------------------------+
   |                      |                                   | :math:`u^3=\sqrt{\gamma^2+4}/2+\gamma/2` |
   +----------------------+-----------------------------------+------------------------------------------+
   | Shift variable,      | :math:`m-\alpha\beta`             | :math:`m(1-\nu\eta)`                     |
   | :math:`s`            |                                   |                                          |
   +----------------------+-----------------------------------+------------------------------------------+
   | :math:`\alpha` or    | :math:`4/\gamma^2`                | :math:`\sqrt{\ln(1+\eta^2)}`             |
   | :math:`\sigma`       |                                   |                                          |
   +----------------------+-----------------------------------+------------------------------------------+
   | :math:`\beta` or     | :math:`m\nu\gamma/2`              | :math:`\ln(m-s)-\sigma^2/2`              |
   | :math:`\mu`          |                                   |                                          |
   +----------------------+-----------------------------------+------------------------------------------+

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

.. _p sev irrel:

Severity is Irrelevant
----------------------

In some cases the actual form of the severity distribution is
essentially irrelevant to the shape of the aggregate distribution.
Consider an aggregate with a :math:`G`-mixed Poisson frequency
distribution. If the expected claim count :math:`n` is large and if the
severity is tame (roughly tame means “has a variance”; any severity from
a policy with a limit is tame; unlimited workers compensation may not be
tame) then particulars of the severity distribution diversify away in
the aggregate. Moreover the variability from the Poisson claim count
component also diversifies away and the shape of the aggregate
distribution converges to the shape of the frequency mixing distribution
:math:`G`. Another way of saying the same thing is that the normalized
distribution of aggregate losses (aggregate losses divided by expected
aggregate losses) converges in distribution to :math:`G`.

We can prove these assertions using moment generating functions. Let
:math:`X_n` be a sequence of random variables with distribution
functions :math:`F_n` and let :math:`X` another random variable with
distribution :math:`F`. If :math:`F_n(x)\to F(x)` as :math:`n\to\infty`
for every point of continuity of :math:`F` then we say :math:`F_n`
converges weakly to :math:`F` and that :math:`X_n` converges in
distribution to :math:`F`.

Convergence in distribution is a relatively weak form of convergence. A
stronger form is convergence in probability, which means for all
:math:`\epsilon>0` :math:`\mathsf{Pr}(|X_n-X|>\epsilon)\to 0` as
:math:`n\to\infty`. If :math:`X_n` converges to :math:`X` in probability
then :math:`X_n` also converges to :math:`X` in distribution. The
converse is false. For example, let :math:`X_n=Y` and :math:`X` be
binomial 0/1 random variables with :math:`\mathsf{Pr}(Y=1)=\mathsf{Pr}(X=1)=1/2`. Then
:math:`X_n` converges to :math:`X` in distribution. However, since
:math:`\mathsf{Pr}(|X-Y|=1)=1/2`, :math:`X_n` does not converge to :math:`X` in
probability.

It is a fact that :math:`X_n` converges to :math:`X` if the MGFs
:math:`M_n` of :math:`X_n` converge to the MFG of :math:`M` of :math:`X`
for all :math:`t`: :math:`M_n(t)\to M(t)` as :math:`n\to\infty`. See
Feller for more details. We can now prove the
following result.

.. container:: prop

   **Proposition.** Let :math:`N` be a :math:`G`-mixed Poisson distribution with mean
   :math:`n`, :math:`G` with mean 1 and variance :math:`c`, and let
   :math:`X` be an independent severity with mean :math:`x` and variance
   :math:`x(1+\gamma^2)`. Let :math:`A=X_1+\cdots+X_N` and :math:`a=nx`.
   Then :math:`A/a` converges in distribution to :math:`G`, so

   .. math:: \mathsf{Pr}(A/a < \alpha) \to \mathsf{Pr}(G < \alpha)

   as :math:`n\to\infty`. Hence

   .. math:: \sigma(A/a) = \sqrt{c + \frac{x(1+\gamma^2)}{a}}\to\sqrt{c}.

We know

.. math:: M_A(\zeta)=  M_G(n(M_X(\zeta)-1))

and so using Taylor’s expansion we can write

.. math::

   \lim_{n\to\infty} M_{A/a}(\zeta)
   &= \lim_{n\to\infty} M_A(\zeta/a)  \\
   &= \lim_{n\to\infty} M_G(n(M_X(\zeta/nx)-1))  \\
   &= \lim_{n\to\infty} M_G(n(M_X'(0)\zeta/nx+R(\zeta/nx)))  \\
   &= \lim_{n\to\infty} M_G(\zeta+nR(\zeta/nx)))  \\
   &= M_G(\zeta)

for some remainder function :math:`R(t)=O(t^2)`. Note that the
assumptions on the mean and variance of :math:`X` guarantee
:math:`M_X'(0)=x=\mathsf{E}[X]` and that the remainder term in Taylor’s
expansion actually is :math:`O(t^2)`. The second part is trivial.

The proposition implies that if the frequency distribution is actually a
Poisson, so the mixing distribution :math:`G=1` with
probability 1, then the loss ratio distribution of a very large book
will tend to the distribution concentrated at the expected, hence the
expression that “with no parameter risk the process risk completely
diversifies away.”

The next figure illustrate the proposition, showing how aggregates change
shape as expected counts increase.

.. ipython:: python
    :okwarning:

    from aggregate.extensions import mixing_convergence
    @savefig tr_prob_convg.png scale=20
    mixing_convergence(0.25, 0.5)

On the top, :math:`G=1` and the claim count is Poisson. Here the scaled
distributions get more and more concentrated about the expected value
(scaled to 1.0). Notice that the density peaks (left) are getting *further
apart* as the claim count increases. The distribution (right) is converging
to a Dirac delta step function at 1.

On the bottom, :math:`G` has a gamma distribution
with variance :math:`0.0625` (asymptotic CV of 25%). The density peaks are getting closer, converging to the mixing gamma. The scaled
aggregate distributions converge to :math:`G` (thick line, right).

It is also interesting to compute the correlation between :math:`A` and
:math:`G`. We have

.. math::

   \mathsf{cov}(A,G)
   &= \mathsf{E}[AG]-\mathsf{E}[A]\mathsf{E}[G]  \\
   &= \mathsf{E}\mathsf{E}[AG \mid G] - nx  \\
   &= \mathsf{E}[nxG^2] - nx  \\
   &= nxc,

and therefore

.. math:: \mathsf{corr}(A,G)=nxc/\sqrt{nx\gamma + n(1+cn)}\sqrt{c}\to 1

as :math:`n\to\infty`.

The proposition shows that in some situations severity is irrelevant to
large books of business. However, it is easy to think of examples where
severity is very important, even for large books of business. For
example, severity becomes important in excess of loss reinsurance when
it is not clear whether a loss distribution effectively exposes an
excess layer. There, the difference in severity curves can amount to the
difference between substantial loss exposure and none. The proposition
does *not* say that any uncertainty surrounding the severity
distribution diversifies away; it is only true when the severity
distribution is known with certainty. As is often the case with risk
management metrics, great care needs to be taken when applying general
statements to particular situations!
