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

More code... test different n for rec bucket, different methods.

::

    from aggregate import build, qd, Aggregate, Severity, round_bucket

    a = build('agg TEST 1 claim sev lognorm 1 cv 50 fixed', update=False)

    for n in range(3,11):
        a.update(recommend_p=n, log2=16)
        qd(a.describe)
        print(f'recommend n = {n}, bucket size = 1 / {1/a.bs}')
        print('-'*100)
        print()
    print(a.info)

    ans = {}
    for m in ['backward', 'round', 'forward']:
        a.update(bs=1/4, sev_calc=m, log2=16, normalize=False)
        print(m)
        qd(a.describe)
        print('-'*100)
        print()
        ans[m] = a.density_df[['p', 'F', 'S']]

    df = pd.concat(ans.values(), keys=ans.keys(), axis=1)

    df.xs('S', axis=1, level=1).plot(xlim=[-1, a.q(0.99)], logy=True, ylim=[1e-2, 1], lw=.5, figsize=(3.5,5))


Explicit Error Quantification for a Tweedie
-----------------------------------------------

There is a series expansion for the pdf of a Tweedie computed by conditioning on the number of claims and using that a convolution of gammas with the same scale parameter is again gamma. For a Tweedie with expected frequency :math:`\lambda`, gamma shape :math:`\alpha` and scale :math:`\beta`, it is given by

.. math::

    f(x) = \sum_{n \ge 1} e^{-\lambda}\frac{\lambda^n}{n!}\frac{x^{n\alpha-1}e^{-x/\beta}}{\Gamma(n\alpha)\beta^{{n\alpha}}}

for :math:`x>0` and :math:`f(x)=\exp(-\lambda)`. The exact function shows the FFT method is very accurate.

.. ipython:: python
    :okwarning:

    from aggregate import tweedie_convert, build, qd
    from scipy.special import loggamma
    import matplotlib.pyplot as plt
    import numpy as np
    from pandas import option_context

    a = build('agg Tw tweedie 10 1.01 1')
    qd(a.describe)

    @savefig tweedie_test_1.png
    a.plot()

A Tweedie with :math:`p` close to 1 is approximates a Poisson. Its gamma severity is very peaked around its mean (high :math:`\alpha` and offsetting small :math:`\beta`).

The next function provides a transparent, if not maximally efficient, implementation of the Tweedie density.

.. ipython:: python
    :okwarning:

    def tweedie_density(x, mean, p, disp):
        pars = tweedie_convert(p=p, μ=mean, σ2=disp)
        λ = pars['λ']
        α = pars['α']
        β = pars['β']
        if x == 0:
            return np.exp(-λ)
        logl = np.log(λ)
        logx = np.log(x)
        logb = np.log(β)
        logbase = -λ
        log_term = 100
        const = -λ - x / β
        ans = 0.0
        for n in range(1, 2000): #while log_term > -20:
            log_term = (const  +
                        + n * logl  +
                        + (n * α - 1) * logx +
                        - loggamma(n+1) +
                        - loggamma(n * α) +
                        - n * α * logb)
            ans += np.exp(log_term)
            if n > 20 and log_term < -227:
                break
        return ans


The following graphs show that the FFT approximation is excellent, across a wide range, just as its good moment-matching performance suggests it would be.

.. ipython:: python
    :okwarning:

    bit = a.density_df.loc[5:a.q(0.99):256, ['p']]
    bit['exact'] = [tweedie_density(i, 10, 1.01, 1) for i in bit.index]
    bit['p'] /= a.bs

    fig, axs = plt.subplots(1, 2, figsize=(2 * 3.5, 2.45), constrained_layout=True, squeeze=True)
    ax0, ax1 = axs.flat

    bit.plot(ax=ax0);
    ax0.set(ylabel='density');
    bit['err'] = bit.p / bit.exact - 1
    bit.err.plot(ax=ax1);
    @savefig tweedie_test_2.png
    ax1.set(ylabel='relative error', ylim=[-1e-5, 1e-5]);

