.. _2_x_tweedie:

=================================================
The Tweedie distribution for working actuaries
=================================================

.. Below is the code from the snippet.


Tweedie distributions are a suitable model for pure premiums and are used in GLMs. Tweedie distributions do not have a closed form density, but estimating the density is easy using ``aggregate``.


Theory
-------

The **Tweedie** family of distributions is a three-parameter exponential family. It is used as unit distribution in GLMs. A variable :math:`X \sim \mathrm{Tw}_p(\mu, \sigma^2)` when
:math:`\mathsf E[X] = \mu` and
:math:`\mathsf{Var}(X) = \sigma^2 \mu^p`, :math:`1 \le p \le 2`.
:math:`p` is a shape parameter and :math:`\sigma^2>0` is a scale   parameter called the dispersion.

A Tweedie with :math:`1<p<2` is a compound Poisson distribution with
gamma distributed severities. The limit when :math:`p=1` is an
over-dispersed Poisson and when :math:`p=2` is a gamma. More generally:
:math:`\mathsf{Tw}_0(\mu,\sigma^2)` is normal :math:`(\mu, \sigma^2)`,
:math:`\mathsf{Tw}_1(\mu, \sigma^2)` is over-dispersed Poisson
:math:`\sigma^2\mathsf{Po}(\mu/\sigma^2)`, and
:math:`\mathsf{Tw}_2(\mu,\sigma^2)` is a gamma with CV :math:`\sigma`.

Let :math:`\mathsf{Ga}(\alpha, \beta)` denote a gamma with shape
:math:`\alpha` and scale :math:`\beta`, with density
:math:`f(x;\alpha,\beta)=x^\alpha- e^{-x/\beta} / \beta^\alpha x\Gamma(\alpha)`.
It has mean :math:`\alpha\beta`, variance :math:`\alpha\beta^2`,
expected square :math:`\alpha(\alpha+1)\beta` and coefficient of
variation :math:`1/\sqrt\alpha`. We can define an alternative
parameterization
:math:`\mathsf{Tw}^*(\lambda, \alpha, \beta) = \mathsf{CP}(\lambda, \mathsf(Ga(\alpha,\beta))`
as a compound Poisson of gammas, with expected frequency
:math:`\lambda`.

The dictionary between the two parameterizations relies on the relation
between the two shape parameters :math:`\alpha` and :math:`p` given by

.. math::

   \alpha = \frac{2-p}{p-1}, \qquad
   p = \frac{2+\alpha}{1+\alpha}.

Starting from :math:`\mathrm{Tw}_p(\mu, \sigma^2)` \*
:math:`\lambda = \displaystyle\frac{\mu^{2-p}}{(2-p)\sigma^2}` \*
:math:`\beta = \displaystyle\frac{\mu^{1-p}}{(p-1)\sigma^2} = \mu /\lambda \alpha`

Starting from :math:`\mathsf{Tw}^*(\lambda, \alpha, \beta)` \*
:math:`\mu = \lambda \alpha \beta` \*
:math:`\sigma^2 = \lambda \alpha(\alpha + 1) / (\beta^2\mu^p)` by
equating expressions for the variance.

It is easy to convert from the gamma mean :math:`m` and CV :math:`\nu`
to :math:`\alpha=1/\nu^2` and :math:`\beta = m/\alpha`. Remember,
``scipy.stats`` scale equals :math:`\beta`.

Tweedie distributions are mixed: they have a probability mass of
:math:`p_0 =e^{-\lambda}` at 0 and are continuous on
:math:`(0, \infty)`.

Jørgensen calls :math:`\mathsf{Tw}(\lambda, \alpha, \beta)` the
**additive** form of the model because

.. math::


   \sum_i \mathsf{Tw}(\lambda_i, \alpha, \beta) =  \mathsf{Tw}\left(\sum_i \lambda_i, \alpha, \beta\right).

He calls :math:`\mathsf{Tw}_p(\mu, \sigma)` the **reproductive**
exponential dispersion model. If
:math:`X_i\sim \mathsf{Tw}_p(\mu, \sigma/w_i)` then

.. math::


   \frac{1}{w}\sum_i w_i X_i \sim \mathsf{Tw}_p\left(\mu, \frac{\sigma^2}{w}\right)

where :math:`w = \sum_i w_i`. The weights :math:`w_i` represents volume
in cell :math:`i` and :math:`X_i` represents the pure premium. The sum
on the left represents the total pure premium.

References
~~~~~~~~~~

Jørgensen, Bent. 1997. The theory of dispersion models. CRC Press.


Practice
--------

The ``aggregate`` language keyword ``tweedie`` uses reproductive
parameters :math:`\mu, p, \sigma`, since these are most natural for GLM modeling. The keyword is used as follows

.. ipython:: python
    :okwarning:

    from aggregate import build
    tw1 = build('agg TW1 tweedie 2 1.05 5')
    tw1

to produce :math:`\mathsf{Tw}_{1.05}(2, 5)`. Inspecting the (non-trivial parts of the) specification shows the parser converts it into the additive form:

.. ipython:: python
    :okwarning:

    {k: v for k,v in tw1.spec.items()
        if v!=0 and v is not None and v!=''}


The note shows the compound Poisson specification.

The helper function ``tweedie_convert`` translates between parameterizations. The scale parameter :math:`\sigma^2` has offsetting effects: higher :math:`\sigma^2` results in a lower claim count, a higher gamma mean, and a more skewed aggregate distribution with a bigger mass at zero.


.. ipython:: python
    :okwarning:

    from aggregate import tweedie_convert
    import pandas as pd
    pd.set_option("display.float_format", lambda x: f'{x:.12g}')

    # three representations, starting with easiest to interpret
    p = 1.005
    μ = 1
    σ2 = 0.1
    m0 = tweedie_convert(p=p, μ=μ, σ2=σ2)

    # magic numbers are
    λ = μ**(2-p) / ((2-p) * σ2)
    α = (2 - p) / (p - 1)
    β = μ / (λ * α)
    tw_cv = σ2**.5 * μ**(p/2-1)
    sev_m = α *  β
    sev_cv = α**-0.5

    m1 = tweedie_convert(λ=λ, m=sev_m, cv=sev_cv)
    m2 = tweedie_convert(λ=λ, α=α, β=β)
    assert np.allclose(m0, m1, m2)
    temp = pd.concat((m0, m1, m2), axis=1)
    temp.columns = ['mean p disp', 'lambda sev m cv', 'lambda shape scale']
    temp

asdf1

.. ipython:: python
    :okwarning:

    program = f'''
    agg Tw0 {λ} claims sev gamma {sev_m:.8g} cv {sev_cv} poisson
    agg Tw1 {λ} claims sev {β:.4g} * gamma {α:.4g} poisson
    agg Tw1 tweedie {μ} {p} {σ2}
    '''
    print(program)
    tweedies = build(program)

    pd.set_option("display.float_format", lambda x: f'{x:.8g}')

    for a in tweedies:
        a.object.plot()
        #plt.gcf().suptitle(a.program)
        #@savefig
        print(a.object)

asdf2

.. ipython:: python
    :okwarning:

    # from reproductive
    tweedie_convert(p=1.05, μ=2, σ2=5)

sdfd3

.. ipython:: python
    :okwarning:

    # from additive
    tweedie_convert(λ=0.406710033, m=4.917508388, cv=0.229415734)

sadf4


.. ipython:: python
    :okwarning:

    # build Tweedie using reproductive parameters, p, mu, sigma^2
    tw1 = build('agg TW1 tweedie 2 1.05 5')
    tw1.plot()
    @savefig tweedie_tw1.png
    print(tw1)
    print(tw1.spec)
    print(tw1.cdf(0), np.exp(-.40671))

asdf5

.. ipython:: python
    :okwarning:

    # when p close to 1 degenerates into Poisson, here mean = 10, sigma2 = 1, so not overdispersed
    tw2 = build('agg TW2 tweedie 10 1.0001 1')
    tw2.plot()
    @savefig tweedie_tw2.png
    print(tw2)

    # gamma has mean 1 and very small CV, acts like degenerate distribution at 1
    tweedie_convert(p=1.0001, μ=10, σ2=1)

asdf6

.. ipython:: python
    :okwarning:

    # when p close to 2 degenerates into Gamma, here mean = 10, and sigma2=0.04
    # variance of tweedie equals sigma2 mu^2, so CV = sigma = 0.2
    # note: this is computed as an approximation
    tw3 = build('agg TW3 tweedie 10 1.999 0.04', log2=16, bs=1/256)
    tw3.plot()
    @savefig tweedie_tw3.png
    print(tw3)



.. ipython:: python
    :okwarning:

    tc = tweedie_convert(p=1.9999, μ=10, σ2=.04)
    print(tc)

    # build explicitly as a gamma
    m, cv = tc['μ'], tc['tw_cv']
    print(m, cv)

    g = build(f'sev g gamma {m} cv {cv}')
    g.plot()
    @savefig tweedie_g.png
    pass

    # or using shape and scale
    sh = cv ** -2
    sc = m / sh
    print(sc, sh)

    g2 = build(f'sev g2 {sc} * gamma {sh}')
    g2.plot()
    @savefig tweedie_g2.png
    pass

    print(g2.stats(), g.stats())


The Power Variance Exponential Family of Distributions
------------------------------------------------------

.. ipython:: python
    :okwarning:

    from aggregate import power_variance_family
    power_variance_family()
    @savefig tweedie_powervariance.png
    pass


See the blog post `The Tweedie-Power Variance Function
Family <https://www.mynl.com/blog?id=c9a74f2055686bb2c250c4fc4f627a89>`__
for more details.


