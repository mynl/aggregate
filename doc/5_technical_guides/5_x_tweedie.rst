.. _2_x_tweedie:

The Tweedie Distribution: Theory
==================================

**Objectives:** Introduce aggregate probability distributions and the `aggregate` library for working with then in the context of real-world, but basic, actuarial problems illustrated using the Tweedie distribution from GLM modeling.

**Audience:** Actuaries at the Associate or Fellow level.

**Prerequisites:** GLM modeling, Tweedie distributions.

**See also:** :doc:`../2_user_guides/DecL/100_tweedie`.

**Contents:**

* :ref:`tw helpful refs`
* :ref:`tw theory`
* :ref:`tw pvf`

.. _tw helpful refs:

Helpful References
-------------------

* :cite:t:`Jorgensen1997`

.. _tw theory:

Theory
-------

Tweedie distributions are a suitable model for pure premiums and are used in GLMs. Tweedie distributions do not have a closed form density, but estimating the density is easy using ``aggregate``.

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

.. _tw pvf:

The Power Variance Exponential Family of Distributions
------------------------------------------------------

.. ipython:: python
    :okwarning:

    from aggregate import power_variance_family
    power_variance_family()
    @savefig tweedie_powervariance.png scale=20
    pass


See the blog post `The Tweedie-Power Variance Function
Family <https://www.mynl.com/blog?id=c9a74f2055686bb2c250c4fc4f627a89>`__
for more details.


