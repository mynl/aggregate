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

