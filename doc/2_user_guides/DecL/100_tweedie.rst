.. _2_x_tweedie_keyword:

The ``tweedie`` Keyword
------------------------

**Prerequisites:**  Tweedie distribution from GLMs. Use of ``build``.

**See also:** :doc:`../../5_technical_guides/5_x_tweedie`.

The ``aggregate`` language keyword ``tweedie`` makes it easy to build Tweedie distributions.
It uses reproductive
parameters :math:`\mu, p, \sigma`, since these are most natural for GLM modeling. The keyword is used as follows to produce :math:`\mathsf{Tw}_{1.05}(2, 5)`, mean 2, :math:`p=1.05`, and dispersion 5.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd, mv
    tw1 = build('agg TW1 tweedie 2 1.05 5')
    qd(tw1)
    mv(tw1)
    print(f'Expected variance = disp x mean ** p = {2**1.05 * 5:.5f}')

Inspecting the (non-trivial parts of the) specification shows the parser converts it into the additive form:

.. ipython:: python
    :okwarning:

    {k: v for k,v in tw1.spec.items()
        if v!=0 and v is not None and v!=''}


The note shows the compound Poisson specification.

The helper function ``tweedie_convert`` translates between parameterizations. The scale parameter :math:`\sigma^2` has offsetting effects: higher :math:`\sigma^2` results in a lower claim count, a higher gamma mean, and a more skewed aggregate distribution with a bigger mass at zero. The code below shows the three representations, starting with the easiest to interpret.


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
    agg Tw0 {λ} claims sev gamma {sev_m:.8g} cv {sev_cv} poisson
    agg Tw1 {λ} claims sev {β:.4g} * gamma {α:.4g} poisson
    agg Tw1 tweedie {μ} {p} {σ2}
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

    tw1 = build('agg TW1 tweedie 2 1.05 5')
    @savefig tweedie_tw1.png
    tw1.plot()
    qd(tw1)
    print(tw1.spec)
    print(tw1.cdf(0), np.exp(-.40671))

When ``p`` is close to 1, the Tweedie approaches a Poisson. Here mean = 10 and sigma2 = 1, so the distribution is not over-dispersed.  The gamma severity has mean 1 and a very small CV; it acts like degenerate distribution at 1.

.. ipython:: python
    :okwarning:

    tw2 = build('agg TW2 tweedie 10 1.0001 1')
    @savefig tweedie_tw2.png
    tw2.plot()
    qd(tw2)
    tweedie_convert(p=1.0001, μ=10, σ2=1)

When ``p`` is close to 2, the Tweedie approaches a Gamma. Here mean = 10, and sigma2=0.04.
The variance equals ``sigma2 mu^2``, so CV = sigma = 0.2

.. ipython:: python
    :okwarning:

    tw3 = build('agg TW3 tweedie 10 1.999 0.04', log2=16, bs=1/256)
    @savefig tweedie_tw3.png
    tw3.plot()
    qd(tw3)

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

.. ipython:: python
    :suppress:

    import matplotlib.pyplot as plt
    plt.close('all')

