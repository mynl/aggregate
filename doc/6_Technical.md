6. Technical Resources
======================


Probability and Risk Theory
============================



Discretizing Severity Distributions
-----------------------------------

There are two simple ways to discretize a continuous distribution.

1. Approximate the distribution with a purely discrete distribution supported at points $x_k=x_0+kb$,
$k=0,1,\dots, N$. Call $b$ the bucket size. The discrete probabilities are
$p_k=P(x_k - b/2 < X \le x_k+b/2)$. To create a rv_histogram variable from ``xs`` and corresponding
 ``p`` values use:

        xss = np.sort(np.hstack((xs, xs + 1e-5)))
        pss = np.vstack((ps1, np.zeros_like(ps1))).reshape((-1,), order='F')[:-1]
        fz_discr = ss.rv_histogram((pss, xss))

The value 1e-5 just needs to be smaller than the resolution requested, i.e. do not "split the bucket".
Generally histograms will be downsampled, not upsampled, so this is not a restriction.

2. Approximate the distribution with a continuous "histogram" distribution
that is uniform on $(x_k, x_{k+1}]$. The discrete proababilities are $p_k=P(x_k < X \le x_{k+1})$.
To create a rv_histogram variable is much easier, just use:

        xs2 = np.hstack((xs, xs[-1] + xs[1]))
        fz_cts = ss.rv_histogram((ps2, xs2))


The first method we call **discrete** and the second **histogram**. The discrete method is appropriate
when the distribution will be used and interpreted as fully discrete, which is the assumption the FFT
method makes. The histogram method is useful if the distribution will be used to create a scipy.stats
rv_histogram variable. If the historgram method is interpreted as discrete and if the mean is computed
appropriately for a discrete variable as $\sum_i p_k x_k$, then the mean will be under-estimated by $b/2$.

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

To use the FFT method, first "bucket" (or quantize) the severity
distribution into a density vector $\text{x}=(x_1,\dots,x_{m})$ whose length
$m$ is a power of two $m=2^n$. Here
$$
\begin{gathered}
x_i= \text{Pr}((i-1/2)b<X<(i+1/2)b)\\ x_1=\text{Pr}(X<b/2),\quad x_{m}=\text{Pr}(X>(m-1/2)b)\end{gathered}
$$

for some fixed $b$. We call $b$ the bucket size. Note $\sum_i
x_i=1$ by construction. The FFT of the $m\times 1$ vector $\text{x}$ is
another $m\times 1$ vector $\hat{\text{x}}$ whose $j$th component is
$$
\sum_{k=0}^{2^n-1} x_k\exp(2\pi ijk/2^n).\label{fft}
$$

The coefficients of $\hat{\text{x}}$ are complex numbers. It is also possible to
express $\hat{\text{x}}=\text{F}\text{x}$ where $\text{F}$ is an appropriate matrix of complex
roots of unity, so there is nothing inherently mysterious about a FFT.
The trick is that there exists a very efficient algorithm for computing. Rather than
taking time proportional to $m^2$, as one would expect, it can be
computed in time proportional to $m\log(m)$. The difference between
$m\log(m)$ and $m^2$ time is the difference between practically possible
and practically impossible.

You can use the inverse FFT to recover $\text{x}$ from its transform $\hat{\text{x}}$.
The inverse FFT is computed using the same equation as the FFT
except there is a minus sign in the exponent and the result is divided
by $2^n$. Because the equation is essentially the same, the inversion
process can also be computed in $m\log(m)$ time.

The next step is magic in actuarial science. Remember that if $N$ is a
$G$-mixed Poisson and $A=X_1+\cdots+X_N$ is an aggregate distribution
then
$$
M_A(\zeta)=M_G(n(M_X(\zeta)-1)).
$$

Using FFTs you can replace the *function* $M_X$ with the discrete approximation *vector* $\hat{\text{x}}$ and
compute
$$
\hat{\text{a}}=M_G(n(\hat{\text{x}} -1))
$$

component-by-component to get an
approximation vector to the function $M_A$. You can then use the inverse
FFT to recover an discrete approximation $\text{a}$ of $A$ from $\hat{\text{a}}$! See
Wang for more details.

Similar tricks are possible in two dimensions---see Press et al.
and Homer and Clark for a discussion.

The FFT allows us to use the following very simple method to
qualitatively approximate the density of an aggregate of dependent
marginals $X_1,\dots,X_n$ given a correlation matrix $\Sigma$. First use
the FFT method to compute the sum $S'$ of the $X_i$ as though they were
independent. Let $\text{Var}(S')=\sigma^{'2}$ and let $\sigma^2$ be the
variance of the sum of the $X_i$ implied by $\Sigma$. Next use the FFT
to add a further "noise" random variable $N$ to $S'$ with mean zero and
variance $\sigma^2-\sigma^{'2}$. Two obvious choices for the
distribution of $N$ are normal or shifted lognormal. Then $S'+N$ has the
same mean and variance as the sum of the dependent variables $X_i$. The
range of possible choices for $N$ highlights once again that knowing the
marginals and correlation structure is not enough to determine the whole
multivariate distribution. It is an interesting question whether all
possible choices of $N$ correspond to actual multivariate structures for
the $X_i$ and conversely whether all multivariate structures correspond
to an $N$. (It is easy to use MGFs to deconvolve $N$ from the true sum
using Fourier methods; the question is whether the resulting
"distribution" is non-negative.)

Heckman and Meyers   used Fourier transforms to compute
aggregate distributions by numerically integrating the characteristic
function. Direct inversion of the Fourier transform is also possible
using FFTs. The application of FFTs is not completely straight forward
because of certain aspects of the approximations involved. The details
are very clearly explained in Menn and Rachev. Their
method allows the use of FFTs to determine densities for distributions
which have analytic MGFs but not densities---notably the class of stable
distributions.


# Frequency Distributions

A random variable $N$ is $G$-mixed Poisson if $N\mid G$ has a Poisson $nG$
distribution for some fixed non-negative $n$ and a non-negative mixing distribution
$G$ with $\text{E}(G)=1$. Let $\text{Var}(G)=c$ (Glenn Meyers calls $c$ the contagion) and let $\text{E}(G^3)=g$.

The MGF of a $G$-mixed Poisson is
$$\label{mgfi}
M_N(\zeta)=\text{E}(e^{\zeta N})=\text{E}(\text{E}(e^{\zeta N} \mid G))=\text{E}(e^{n
  G(e^\zeta-1)})=M_G(n(e^\zeta-1))
$$

since $M_G(\zeta):=\text{E}(e^{\zeta G})$ and the MGF of a Poisson with mean $n$ is $\exp(n(e^\zeta-1))$.
Thus
$$
\text{E}(N)=M_N'(0)=n M_G'(0)=n,
$$

because $\text{E}(G)=M_G'(0)=1$. Similarly
$$
\text{E}(N^2)=M_N''(0)=n^2M_G''(0)+n M_G'(0)=n^2(1+c)+n
$$

and so
$$
\text{Var}(N)=n(1+cn).
$$

Finally
$$
\text{E}(N^3) = M_N'''(0) =n^3M_G'''(0)+3n^2M_G''(0)+n M_G'(0) = gn^3 + 3n^2(1+c) + n
$$

and therefore the central moment
$$
\text{E}(N-\text{E}(N))^3 = n^3(g -3c -1) + 3cn^2 + n.
$$

We can also assume $G$ has mean $n$ and work directly with $G$ rather
than $nG$, $\text{E}(G)=1$. We will call both forms mixing distributions.

## Interpretation of the Coefficient of Variation of the Mixing Distribution

Per Actuarial Geometry, if $\nu$ is the CV of $G$ then the $\nu$ equals the asymptotic coefficient
of variation for any $G$-mixed compound Poisson distribution whose variance exists. The variance
will exist iff the variance of the severity term exists.

## Gamma Mixing

A negative binomial is a gamma-mixed Poisson: if $N \mid G$ is
distributed as a Poisson with mean $G$, and $G$ has a gamma
distribution, then the unconditional distribution of $N$ is a negative
binomial. A gamma distribution has a shape parameter $a$ and a scale parameter $\theta$
so that the density is proportional to $x^{a-1}e^{x/\theta}$, $\text{E}(G)=a\theta$ and
$\text{Var}(G)=a\theta^2$.

Let $c=\text{Var}(G)=\nu^2$, so $\nu$ is the coefficient of variation of
the mixing distribution. Then

* $a\theta=1$ and $a\theta^2=c$
* $\theta=c=\nu^2$, $a=1/c$

The non-central moments of the gamma distribution are $\text{E}(G^r)=\theta^r\Gamma(a+r)/\Gamma(a)$. Therefore
$Var(G) = a\theta^2$ and $E(G-E(G))^3 = 2a\theta^3$.
The skewness of $G$ is $\gamma = 2/\sqrt(a) = 2\nu$.

Applying the general formula for the third central moment of $N$ we get an expression for the skewness
$$
\text{skew}(N) = \frac{n^3(\gamma -3c -1) + n^2(3c+2) + n}{(n(1+cn))^{3/2}}.
$$

The corresponding MGF of the gamma is $M_G(\zeta)  = (1-\theta\zeta)^{-a}$.

## Shifted Mixing (General)

We can adjust the skewness of mixing with shifting. In addition to a target CV $\nu$ assume a proportion $f$ of claims are sure to occur. Use a mixing distribution $G=f+G'$ such that

* $E(G)= f + E(G') = 1$ and
* $CV(G) = SD(G') = \nu$.

As $f$ increases from 0 to 1 the skewness of $G$ will increase. Delaporte first introduced this idea.

Since $\text{skew}(G)=\text{skew}(G')$ we have $g=\text{E}(G^3)=\nu^3 \text{skew}(G')+3c+1$.

## Delaporte Mixing (Shifted Gamma)

Inputs are target CV $\nu$ and proportion of certain claims $f$, $0\leq f \leq 1$. Find parameters $f$, $a$ and $\theta$ for a shifted gamma $G=f+G'$ with $E(G')=1-f$ and $SD(G')=\nu$ as

* $f$ is input
* mean $a\theta=1-s$ and $CV=\nu=\sqrt{a}\theta$ so $a=(1-f)^2/\nu^2=(1-f)^2/c$ and $\theta=(1-f)/a$

The skewness of $G$ equals the skewness of $G'$ equals $2/\sqrt{a}= 2\nu/(1-f)$, which is then greater than the skewness $2\nu$ when $f=0$. The third non-central moment $g=2\nu^4/(1-f)+3c+1$

## Poisson Inverse Gaussian Distribution



## Bernoulli Distribution

## Binomial Distribution

## Fixed Distribution

