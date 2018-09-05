Probability and Risk Theory
===========================

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


The first methdo we call **discrete** and the second **histogram**. The discrete method is appropriate
when the distribution will be used and interpreted as fully discrete, which is the assumption the FFT
method makes. The histogram method is useful if the distribution will be used to create a scipy.stats
rv_histogram variable. If the historgram method is interpreted as discrete and if the mean is computed
appropriately for a discrete variable as $\sum_i p_k x_k$, then the mean will be under-estimated by $b/2$.

Generalized Distributions
-------------------------

Fast Fourier Transforms
-----------------------
