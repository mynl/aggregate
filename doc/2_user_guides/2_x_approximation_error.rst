.. _2_x_approximation_error:

Approximations and Errors
===========================

**Objectives.**  

**Audience.**

**Prerequisites.** 

**See also.**

* Simulation error of mean
* Simulation error of percentiles
* Moments of a lognormal (Mandlebrot)
* Implications for bs and log2

Estimation of the variance of
percentile estimates; Morton B. BROWN and Robert A. WOLFE

Compute the number of sims to model the mean to within tolerance a of actual with probability p, :math:`(z_{p/2}/a \nu)^2` where :math:`\nu` is the CV. (Usual normal approx to se of mean argument.) Eg for 90% conf z=1.644 and a=0.01 (FFT is generally much closer) you get 27,055 times :math:`\nu^2`. For cat like distributions :math:`\nu` can be in the range 50-100, leading to 67-270 million simulations. Thus FFT provides stunning accuracy.

OK, don't estimate mean. What about SE of percentiles (AEP)?

Finally, what about OEP? OEP is an adjusted quantile, so can use same argument on the severity with adjusted p values. See Brown and Wolf paper, prob JKK for percentile SEs.

BW says se is

.. math:: \frac{1}{f(x_p)}\left(\frac{p(1-p)}{n}\right)^{0.5}

What is the density? (Obs small because range so large)! Table and investigate...


Here's some code on the mean. ::

    import scipy.stats as ss

    z = ss.norm.isf
    phi = ss.norm.cdf

    def test_sample_mean(cv, p=0.99, a=0.01, simulate=False):
        """
        Test number of sims for p=99% certainty of a=1% accuracy when underlying
        variable is lognormal with given cv. Basic large sample, normal approximation
        to standard error of the mean.

        """
        zp = z((1-p)/2)
    n = int((zp / a * cv) ** 2)
    print(f'zp = {zp:.3f}, zp**2 = {zp*zp:.3f}\nformula = {(zp/a)**2:,.0f} * n**2\nn = {n:,.0f}')

    if n <= 100000 or simulate is True:
        mu, sig = mu_sigma_from_mean_cv(1, cv)
        fz = ss.lognorm(sig, scale=np.exp(mu))

        samps = [np.mean(fz.rvs(n)) for i in range(1000)]
        plt.hist(samps, lw=.25, ec='w', bins=10)
        samps = np.sort(samps)
        print(f'observed 99% ci equals ({samps[10]}, {samps[990]})')

    return zp, n

test_sample_mean(.2, .9, .01)

Stuff

The recommended bucket is too small because it is based on only the 99.9 percentile.


The contribution of the extreme tail to the mean of a distribution increases with the tail thickness. See results of
Mandlebrot - the higher moments of the lognormal are nonlocal and depend on different parts of the distribution. (Hence the problems with numerical integration!) (https://users.math.yale.edu/mandelbrot/web_pdfs/9_E9lognormalDistribution.pdf) *A case against the lognormal distribution* in Mandelbrot, Benoit B. "A case against the lognormal distribution." Fractals and scaling in finance. Springer, New York, NY, 1997. 252-269.

::

    # how much of the mean of a lognormal comes from the extreme tail?
    ans = []
    for sigma in [.5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4]:
        fz = ss.lognorm(sigma)
        for n in range(1,16):
            p = 1 - 10**-n
            q = fz.isf(1-p)
            m, v = fz.stats()
            cv = float(v**.5/m)
            lev = moms_analytic(fz, q, 0, 1)[1]
            ans.append([sigma, n, p, 10**-n, q, cv, float(m), lev])

    ans = pd.DataFrame(ans, columns=['sigma', 'n', 'p', 's', 'q(p)', 'cv', 'mean', 'lev'])
    ans['err'] = ans.lev / ans['mean'] - 1
    print(ans.to_string(formatters={'err': lambda x: f'{x:.1%}'}))

    x = ans.query('abs(err) < 0.001').groupby('sigma').apply(lambda x: x.iloc[0])
    x

    x.set_index('cv')['n'].plot()

Since bs is  based on the p999, it will fail when confronting and extreme tail.

Based on above graph we can come up with an (empirical) relationship between the CV and the required percentile for decent coverage.

