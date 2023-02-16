.. _2_x_tweedie_keyword:

.. reviewed 2022-12-24

The ``tweedie`` Keyword
------------------------

**Prerequisites:**  Tweedie distribution from GLMs. Use of ``build``.

**See also:** :doc:`../../5_technical_guides/5_x_tweedie`.

The ``aggregate`` language keyword ``tweedie`` makes it easy to build Tweedie distributions.
It uses reproductive
parameters :math:`\mu, p, \sigma^2` (mean, power, and dispersion), since these are most natural for GLM modeling.

**Example.**

The keyword is used as follows to produce :math:`\mathsf{Tw}_{1.05}(2, 5)`, mean 2, :math:`p=1.05`, and dispersion 5.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd, mv
    a15 = build('agg DecL:15 tweedie 2 1.05 5')
    qd(a15)
    mv(a15)
    print(f'Expected variance = disp x mean ** p = {2**1.05 * 5:.5f}')

Inspecting the (non-trivial parts of the) specification shows the parser converts it into the additive form:

.. ipython:: python
    :okwarning:

    {k: v for k,v in a15.spec.items()
        if v!=0 and v is not None and v!=''}


The note shows the compound Poisson specification.

The helper function ``tweedie_convert`` translates between parameterizations. The scale (dispersion) parameter :math:`\sigma^2` has offsetting effects: higher :math:`\sigma^2` results in a lower claim count, a higher gamma mean, and a more skewed aggregate distribution with a bigger mass at zero.

**Example.**

The code below shows the three Tweedie representations, starting with the easiest to interpret.

.. ipython:: python
    :okwarning:

    from aggregate import tweedie_convert
    import pandas as pd

    p = 1.005; μ = 1; σ2 = 0.1;                    \
    m0 = tweedie_convert(p=p, μ=μ, σ2=σ2);         \
    λ = μ**(2-p) / ((2-p) * σ2);                   \
    α = (2 - p) / (p - 1);                         \
    β = μ / (λ * α);                               \
    tw_cv = σ2**.5 * μ**(p/2-1);                   \
    sev_m = α *  β;                                \
    sev_cv = α**-0.5;                              \
    m1 = tweedie_convert(λ=λ, m=sev_m, cv=sev_cv); \
    m2 = tweedie_convert(λ=λ, α=α, β=β);
    assert np.allclose(m0, m1, m2)
    temp = pd.concat((m0, m1, m2), axis=1); \
    temp.columns = ['mean p disp', 'lambda sev m cv', 'lambda shape scale'];
    with pd.option_context('display.float_format', lambda x: f'{x:.12g}'):
        print(temp)

Three different ways of specifying the same Tweedie distribution.

.. ipython:: python
    :okwarning:

    program = f'''
    agg DecL:16 {λ} claims sev gamma {sev_m:.8g} cv {sev_cv} poisson
    agg DecL:17 {λ} claims sev {β:.4g} * gamma {α:.4g} poisson
    agg DecL:18 tweedie {μ} {p} {σ2}
    '''
    tweedies = build(program)
    for a in tweedies:
        print(a.program)
        qd(a.object.describe)
        print()

Convert from reproductive form:

.. ipython:: python
    :okwarning:

    tweedie_convert(p=1.05, μ=2, σ2=5)

Convert from additive form:

.. ipython:: python
    :okwarning:

    tweedie_convert(λ=0.406710033, m=4.917508388, cv=0.229415734)

Build a Tweedie using reproductive parameters, ``p``, ``mu``, ``sigma2``.

.. ipython:: python
    :okwarning:

    a19 = build('agg DecL:19 tweedie 2 1.05 5')
    @savefig tweedie_a19.png
    a19.plot()
    qd(a19)
    print(a19.spec)
    print(a19.cdf(0), np.exp(-.40671))

**Example.**

When ``p`` is close to 1, the Tweedie approaches a Poisson. Here mean = 10 and sigma2 = 1, so the distribution is not over-dispersed.  The gamma severity has mean 1 and a very small CV; it acts like degenerate distribution at 1.

.. ipython:: python
    :okwarning:

    a20 = build('agg DecL:20 tweedie 10 1.0001 1')
    @savefig tweedie_a20.png
    a20.plot()
    qd(a20)
    tweedie_convert(p=1.0001, μ=10, σ2=1)

**Example.**

When ``p`` is close to 2, the Tweedie approaches a Gamma. Here mean = 10, and sigma2=0.04.
The variance equals ``sigma2 mu^2``, so CV = sigma = 0.2

.. ipython:: python
    :okwarning:

    a21 = build('agg DecL:21 tweedie 10 1.999 0.04', log2=16, bs=1/256)
    @savefig tweedie_a21.png
    a21.plot()
    qd(a21)

Build the same distribution explicitly from gamma severities. Here the gamma is built using mean and CV or shape and scale.

.. ipython:: python
    :okwarning:

    tc = tweedie_convert(p=1.9999, μ=10, σ2=.04)
    print(tc)
    m, cv = tc['μ'], tc['tw_cv']
    print(m, cv)
    g = build(f'sev g gamma {m} cv {cv}')
    sh = cv ** -2; sc = m / sh
    print(sc, sh)
    g2 = build(f'sev g2 {sc} * gamma {sh}')
    print(g2.stats(), g.stats())


Analytic Error Analysis
""""""""""""""""""""""""""

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
    qd(a)
    @savefig tweedie_test_1.png
    a.plot()

A Tweedie with :math:`p` close to 1 approximates a Poisson. Its gamma severity is very peaked around its mean (high :math:`\alpha` and offsetting small :math:`\beta`).

The next function provides a transparent, if inefficient, implementation of the Tweedie density.

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
    @savefig tweedie_test_2.png scale=20
    ax1.set(ylabel='relative error', ylim=[-1e-5, 1e-5]);


.. ipython:: python
    :suppress:

    import matplotlib.pyplot as plt
    plt.close('all')
