.. _5_x_probability:

Probability Background
======================

**Objectives:** Statement and limited explanation of important probability concepts that underlie ``aggregate`` calculations.

**Audience:** Readers looking for a probability refresher.

**Prerequisites:** Knowledge of basic probability and calculus (real analysis).

**See also:** :doc:`5_x_insurance_probability`.

**Notation:** The variance of a random variable :math:`X` is :math:`\mathsf{var}(X)=\mathsf{E}[X^2]-\mathsf{E}[X]^2`. The standard deviation is :math:`\sigma(X)=\sqrt{\mathsf{var}(X)}`. The coefficient of variation (CV) of :math:`X` is :math:`\mathsf{CV}(X)=\sigma(X)/\mathsf{E}[X]`. The skewness of :math:`X` is :math:`\mathsf{E}[(X-\mathsf{E}[X])^3]/\sigma(X)^3`.

**Contents:**

* :ref:`p hr`
* :ref:`p types`
* :ref:`p severity`
* :ref:`p mgfs`
* :ref:`p frequency`
* :ref:`p aggregate`
* :ref:`p sln sg`
* :ref:`list of freq distributions`
* :ref:`list of distributions`

.. _p hr:


Helpful References
--------------------

* :cite:t:`LM`
* :cite:t:`Panjer1992`
* :cite:t:`Williams1991`
* :cite:t:`feller71`
* :cite:t:`Loeve2017`
* :cite:t:`JKK`
* :cite:t:`Mildenhall2017b`

.. _p types:

Types
------

.. todo::

  Documentation to follow.


.. _p severity:

Severity Distributions
-----------------------

Computing moments
~~~~~~~~~~~~~~~~~~

Higher moments of a layer with a limit :math:`y` excess of an attachment (deductible, retention) :math:`a` can be computed as

.. math::

   \mathsf E[((X-a)^+ \wedge l)^n]
   &= \int_a^{a+l} (x-a)^n f(x)\,dx + l^nS(a+l) \\
   &= \sum_{k=0}^n (-1)^k \binom{n}{k} a^{n-k} \int_a^{a+l} x^k f(x)\,dx + l^nS(a+l) \\
   &= \sum_{k=0}^n (-1)^k \binom{n}{k} a^{n-k} \left(\mathsf E[k; a+l] - \mathsf E[k; a]\right)+ l^nS(a+l)

where

.. math::


   \mathsf E[k; a] = \int_0^a x^kf(x)\,dx

is the partial expectation function.

Lognormal
"""""""""

For the lognormal, the trick for higher moments is to observe that if
:math:`X` is lognormal :math:`(\mu,\sigma)` then :math:`X^k` is lognormal :math:`(k\mu, k\sigma)`. The formula for partial
expectations of the lognormal is easy to compute by substitution, giving

.. math::

   \mathsf E[k, a] = \exp(k\mu + k^2\sigma^2/2)\Phi\left( \frac{\log x -\mu - k\sigma^2}{\sigma} \right)

Densities of the form :math:`f(x)=x^\alpha c(\alpha)g(x)`
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. math::

   \mathsf E[k, a]
   &= \int_0^a x^k x^\alpha c(\alpha)g(x) \,dx \\
   &= \frac{c(\alpha)}{c(n+\alpha)}\int_0^a x^{k+\alpha} c(k+\alpha)g(x) \,dx \\
   &= \frac{c(\alpha)}{c(n+\alpha)}F_{k+\alpha}(a)

are easy to express in terms of the distribution function. This is a broad class including the gamma.

Pareto
"""""""

An easy integral computation, substitute :math:`y=\lambda + x` to express in powers of :math:`y`:

.. math::
  \mathsf E[k, a]
   &= \int_0^a \alpha  x^k \frac{\lambda^\alpha}{(\lambda + x)^{\alpha+1}}\,dx \\
   &= \int_\lambda^{\lambda + a} \alpha\lambda^\alpha \frac{(y-\lambda)^k}{y^{\alpha+1}}\,dy \\
   &= \sum_{i=0}^k (-1)^{k-i} \alpha\lambda^\alpha \binom{k}{i}   \int_\lambda^{\lambda + a}  y^{i-\alpha-1} \lambda^{k-i}\,dy \\
   &= \sum_{i=0}^k (-1)^{k-i} \alpha\lambda^{\alpha+k-i} \binom{k}{i}  \frac{y^{i-\alpha}}{i-\alpha}\big|_\lambda^{\lambda + a}.



.. _p sev dist roster:

``scipy.stats`` Severity Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All zero, one, and two shape parameter ``scipy.stats`` continuous random variable classes can be used as severity distributions. See :ref:`list of distributions` for details about each available option.


.. _p frequency:

Frequency Distributions
------------------------

The following reference is from the ``scipy`` documentation.

Bernoulli Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

The probability mass function for `bernoulli` is:

    .. math::

       f(k) = \begin{cases}1-p  &\text{if } k = 0\\
                           p    &\text{if } k = 1\end{cases}

for :math:`k` in :math:`\{0, 1\}`, :math:`0 \leq p \leq 1`
`bernoulli` takes :math:`p` as shape parameter,
where :math:`p` is the probability of a single success
and :math:`1-p` is the probability of a single failure.



Binomial Distribution
~~~~~~~~~~~~~~~~~~~~~~

The probability mass function for `binom` is:

    .. math::

       f(k) = \binom{n}{k} p^k (1-p)^{n-k}

for :math:`k \in \{0, 1, \dots, n\}`, :math:`0 \leq p \leq 1`
`binom` takes :math:`n` and :math:`p` as shape parameters,
where :math:`p` is the probability of a single success
and :math:`1-p` is the probability of a single failure.


Geometric Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~

The probability mass function for `geom` is:

    .. math::

        f(k) = (1-p)^{k-1} p

for :math:`k \ge 1`, :math:`0 < p \leq 1`
`geom` takes :math:`p` as shape parameter,
where :math:`p` is the probability of a single success
and :math:`1-p` is the probability of a single failure.


Poisson Distribution
~~~~~~~~~~~~~~~~~~~~~~~

The probability mass function for `poisson` is:

    .. math::

        f(k) = \exp(-\mu) \frac{\mu^k}{k!}

for :math:`k \ge 0`.
`poisson` takes :math:`\mu \geq 0` as shape parameter.

Neyman (A) Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~

The Neyman distribution is a Poisson stopped-sum distribution of Poisson variables, see :cite:t:`JKK`.


Fixed Distribution
~~~~~~~~~~~~~~~~~~~

The fixed distribution takes a single value with probability one.

.. _p mgfs:

Moment Generating Functions
---------------------------

The moment generating function of a random variable :math:`X` is defined
as

.. math:: M_X(z)=\mathsf{E}[\exp(z X)].

The moment generating function is related to the characteristic function
of :math:`X` which is defined as :math:`\phi_X(z)=\mathsf{E}[\exp(iz
X)]=M_X(iz)`. :math:`\phi` is guaranteed to converge for all real
:math:`z` and so is preferred in certain situations.

Moment generating functions get their name from the fundamental property
that

.. math:: \frac{\partial^n M_X}{\partial z^n}\Big\vert_{z=0}=\mathsf{E}[X^n]

for all positive integers :math:`n` provided the differential exists.

Let :math:`F` be the distribution function of :math:`X`. :cite:t:`feller71` Section XVII.2a shows that if :math:`F` has
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

.. math:: M_{X_1,X_2}(z_1,z_2)=\mathsf{E}[\exp(z_1 X_1+z_2 X_2)].

It has the property that

.. math::

   \frac{\partial^{m+n} M_{X_1,X_2}}{\partial z_1^m\partial
     z_2^n}\Big\vert_{(0,0)} =\mathsf{E}[X_1^mX_2^n]

for all positive integers :math:`n,m`.

The MGF of a normal variable with mean :math:`\mu` and standard
deviation :math:`\sigma` is

.. math:: M(z)=\exp(\mu z\sigma^2 z^2/2).

The MGF of a Poisson variable with mean :math:`n` is

.. math:: M(z)=\exp(n(e^{z}-1)).

See any standard text on probability for more information on moment
generating functions, characteristic functions and modes of convergence.



.. _mixed frequency distributions:

Mixed Frequency Distributions
---------------------------------

A random variable :math:`N` is :math:`G`-mixed Poisson if
:math:`N\mid G` has a Poisson :math:`nG` distribution for some fixed
non-negative :math:`n` and a non-negative mixing distribution :math:`G`
with :math:`\mathsf E[G]=1`. Let :math:`\mathsf{var}(G)=c` and let :math:`\mathsf E[G^3]=g`. Glenn Meyers calls :math:`c` the contagion.

The MGF of a :math:`G`-mixed Poisson is

.. math::

   \label{mgfi}
   M_N(z)=\mathsf E[e^{z N}]=\mathsf E[\mathsf E[e^{z N} \mid G]]=\mathsf E[e^{n
     G(e^z-1]})=M_G(n(e^z-1))

since :math:`M_G(z):=\mathsf E[e^{z G}]` and the MGF of a Poisson
with mean :math:`n` is :math:`\exp(n(e^z-1))`. Thus

.. math::

   \mathsf E[N]=M_N'(0)=n M_G'(0)=n,

because :math:`\mathsf E[G]=M_G'(0)=1`. Similarly

.. math::

   \mathsf E[N^2]=M_N''(0)=n^2M_G''(0)+n M_G'(0)=n^2(1+c)+n

and so

.. math::

   \mathsf{var}(N)=n(1+cn).

Finally

.. math::

   \mathsf E[N^3] = M_N'''(0) =n^3M_G'''(0)+3n^2M_G''(0)+n M_G'(0) = gn^3 + 3n^2(1+c) + n

and therefore the central moment

.. math::


   \mathsf E(N-\mathsf E[N])^3 = n^3(g -3c -1) + 3cn^2 + n.

We can also assume :math:`G` has mean :math:`n` and work directly with
:math:`G` rather than :math:`nG`, :math:`\mathsf E[G]=1`. We will call
both forms mixing distributions.

Gamma Mixing
~~~~~~~~~~~~~~

A negative binomial is a gamma-mixed Poisson: if :math:`N \mid G` is
distributed as a Poisson with mean :math:`G`, and :math:`G` has a gamma
distribution, then the unconditional distribution of :math:`N` is a
negative binomial. A gamma distribution has a shape parameter :math:`a`
and a scale parameter :math:`\theta` so that the density is proportional
to :math:`x^{a-1}e^{x/\theta}`, :math:`\mathsf E[G]=a\theta` and
:math:`\mathsf{var}(G)=a\theta^2`.

Let :math:`c=\mathsf{var}(G)=\nu^2`, so :math:`\nu` is the coefficient of
variation of the mixing distribution. Then

-  :math:`a\theta=1` and :math:`a\theta^2=c`
-  :math:`\theta=c=\nu^2`, :math:`a=1/c`

The non-central moments of the gamma distribution are
:math:`\mathsf E[G^r]=\theta^r\Gamma(a+r)/\Gamma(a)`. Therefore
:math:`Var(G) = a\theta^2` and :math:`E(G-E(G))^3 = 2a\theta^3`. The skewness of :math:`G` is :math:`\gamma = 2/\sqrt(a) = 2\nu`.

Applying the general formula for the third central moment of :math:`N`
we get an expression for the skewness

.. math::

   \mathsf{skew}(N) = \frac{n^3(\gamma -3c -1) + n^2(3c+2) + n}{(n(1+cn))^{3/2}}.

The corresponding MGF of the gamma is
:math:`M_G(z) = (1-\theta z)^{-a}`.

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


Model (a) is used by Microsoft Excel, :cite:t:`WangS1998`, and :cite:t:`JKK` Chapter 17. Model (b) is used by :cite:t:`Bowers1997`. Model (c) is used by :cite:t:`KPW`. Obviously model (c) is just model (a) with a change of notation.


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


In model (c) the parameter :math:`r` need
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
:math:`(q+pe^{z})^n` and the probabilities are derived from the
binomial expansion of the MGF. By analogy the negative binomial can be
defined in terms of the negative binomial expansion of
:math:`(Q-Pe^{z})^{-k}` where :math:`Q=1+P`, :math:`P>0` and
:math:`k>0`.

The actuary can look at the negative
binomial in two different way, each of which gives different results. It is important to
understand these two views. First there is the **contagion view**, where the
mixing distribution :math:`G` has mean :math:`n` and variance :math:`c`
producing a negative binomial with mean :math:`n` and variance
:math:`n(1+cn)`. (In fact :math:`G` is a gamma with model (a) parameters
:math:`\alpha=r` and :math:`\beta=1/r`.) The word contagion is used by :cite:t:`Heckman1983` and is supposed to
indicate a “contagion” of claim propensity driven by common shock
uncertainty, such as claim inflation, economic activity, or weather.
Here the variance grows with the square of :math:`n` and the coefficient
of variation tends to :math:`\sqrt{c}>0` as :math:`n\to\infty`.
Secondly, one can consider an over-dispersed family of Poisson variables
with mean :math:`n` and variance :math:`vn` for some :math:`v>1`. We
call :math:`v` the **variance multiplier**. Now, the coefficient of variation
tends to :math:`0` as :math:`n\to\infty`. The notion of over-dispersion
and its application in modeling is discussed in :cite:t:`Clark2004` and :cite:t:`Verrall2004`.

.. _prob variance mult:

The Variance Multiplier
~~~~~~~~~~~~~~~~~~~~~~~~~~

The variance of a mixed Poisson equals :math:`n(1+cn)` where :math:`c` equals the variance of the mixing distribution. Thus the variance equals :math:`v=1+cn` times the mean :math:`n`, where :math:`v` is called the **variance multiplier**. The variance multiplier specification is used by some US rating bureaus. The dictionary to variance and mix CV is

.. math::

  c &= (v-1) / n \\
  \mathit{cv} &= \sqrt{(v-1)/n}.

The frequency for an excess layer attaching at :math:`a` equals :math:`nS(a)`. For fixed :math:`c`, the implied variance multiplier :math:`v=1+cnS(a)` decreases and the excess claim count distribution converges to a Poisson. This is an example of the law of small numbers.

Per :cite:t:`Mildenhall2017b`, if :math:`\nu` is the CV of :math:`G` then the
:math:`\nu` equals the asymptotic coefficient of variation for any
:math:`G`-mixed compound Poisson distribution whose variance exists. The
variance will exist iff the variance of the severity term exists. See :doc:`5_x_severity_irrelevant`.


Negative Binomial Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Negative binomial distribution describes a sequence of iid Bernoulli trials, repeated until a predefined, non-random number of successes occurs.

The probability mass function of the number of failures for `nbinom` is:

.. math::

   f(k) = \binom{k+n-1}{n-1} p^n (1-p)^k

for :math:`k \ge 0`, :math:`0 < p \leq 1`

`nbinom` takes :math:`n` and :math:`p` as shape parameters where n is the
number of successes, :math:`p` is the probability of a single success,
and :math:`1-p` is the probability of a single failure.

Another common parameterization of the negative binomial distribution is
in terms of the mean number of failures :math:`\mu` to achieve :math:`n`
successes. The mean :math:`\mu` is related to the probability of success
as

.. math::

   p = \frac{n}{n + \mu}

The number of successes :math:`n` may also be specified in terms of a
"dispersion", "heterogeneity", or "aggregation" parameter :math:`\alpha`,
which relates the mean :math:`\mu` to the variance :math:`\sigma^2`,
e.g. :math:`\sigma^2 = \mu + \alpha \mu^2`. Regardless of the convention
used for :math:`\alpha`,

.. math::

   p &= \frac{\mu}{\sigma^2} \\
   n &= \frac{\mu^2}{\sigma^2 - \mu}


Beta Binomial Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The beta-binomial distribution is a binomial distribution with a
probability of success `p` that follows a beta distribution.

The probability mass function for `betabinom` is:

.. math::

   f(k) = \binom{n}{k} \frac{B(k + a, n - k + b)}{B(a, b)}

for :math:`k \in \{0, 1, \dots, n\}`, :math:`n \geq 0`, :math:`a > 0`,
:math:`b > 0`, where :math:`B(a, b)` is the beta function.

`betabinom` takes :math:`n`, :math:`a`, and :math:`b` as shape parameters.


Shifted Mixing (General)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can adjust the skewness of mixing with shifting. In addition to a
target CV :math:`\nu` assume a proportion :math:`f` of claims are sure
to occur. Use a mixing distribution :math:`G=f+G'` such that

-  :math:`\mathsf E[G]= f + \mathsf E[G'] = 1` and
-  :math:`\mathsf{CV}(G) = \sigma(G') = \nu`.

As :math:`f` increases from 0 to 1 the skewness of :math:`G` will
increase. Delaporte first introduced this idea.

Since :math:`\mathsf{skew}(G)=\mathsf{skew}(G')` we have
:math:`g=\mathsf E[G^3]=\nu^3 \mathsf{skew}(G')+3c+1`.

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

See :cite:t:`LM`.


.. _p aggregate:

Aggregate Distributions
-----------------------

Let :math:`A=X_1+\cdots +X_N` be an aggregate distribution, where
:math:`N` is the **frequency** component and  :math:`X_i` are iid **severity**
random variables.


Aggregate Mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mean of a sum equals the sum of the means. Let :math:`A = X_1 + \cdots + X_N`. If :math:`N=n` is fixed then :math:`\mathsf E[A] = n\mathsf E[X]`, because all :math:`\mathsf E[X_i]=\mathsf E[X]`. In general,

.. math::

    \mathsf E[A] = \mathsf E[X]\mathsf E[N]

by conditional probability.

Aggregate Variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The variance of a sum of independent random variables equals the sum of the variances.  If :math:`N=n` is fixed then :math:`\mathsf{Var}(A) = n\mathsf{Var}(X)` and :math:`\mathsf{Var}(N)=0`. If :math:`X=x` is fixed then :math:`\mathsf{Var}(A) = x^2\mathsf{Var}(N)` and :math:`\mathsf{Var}(X)=0`. Making the obvious associations :math:`n\leftrightarrow\mathsf E[N]`, :math:`x\leftrightarrow\mathsf E[X]` suggests

.. math::

    \mathsf{Var}(A) = \mathsf E[N]\mathsf{Var}(X) + \mathsf E[X]^2\mathsf{Var}(N).

Using conditional expectations and conditioning on the value of :math:`N` shows this  is the correct answer!

**Exercise.** Confirm the formulas for an aggregate mean and variance hold for the :ref:`Simple Example`.

Aggregate Moment Generating Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the tower property of conditional expectations and the independence of :math:`N` and :math:`X_i` gives

.. math::

   M_A(z)
   &= \mathsf{E}[\exp(z(X_1+\cdots X_N))] \\
   &= \mathsf{E}[\mathsf{E}[\exp(z(X_1+\cdots X_N)) \mid N]] \\
   &= \mathsf{E}[\mathsf{E}[\exp(z X_1)^N]] \\
   &= \mathsf{E}[\mathsf{E}[\exp(z X_1)]^N] \\
   &= M_N(\log(M_X(z)))

Differentiating and using XXs formula, yields the moments of :math:`A`, see below.

The last expression is very important and underlies the use of FFTs to compute aggregate distributions.

Next, specialize to the case where :math:`A=X_1+\cdots +X_N` is an aggregate distribution with
:math:`N` a :math:`G`-mixed Poisson. Then

.. math::

   M_A(z)
   &= \mathsf{E}[\exp(z(X_1+\cdots X_N))]  \\
   &= \mathsf{E}[\mathsf{E}[\exp(z(X_1+\cdots X_N)) \mid N]]  \\
   &= \mathsf{E}[\mathsf{E}[\exp(z X_1)^N]]  \\
   &= \mathsf{E}[\mathsf{E}[M_X(z)^N \mid G]]  \\
   &= \mathsf{E}[\exp(nG(M_X(z)-1))]  \\
   &= M_G(n(M_X(z)-1))

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
discussed in :cite:t:`Bowers1997`. Properties of
the shifted gamma and lognormal distributions, including the method of
moments fit parameters, are also shown in :cite:t:`Daykin1993` chapter 3.

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


.. appendix: scipy sev dists

.. include:: 5_x_scipy_severity.rst
