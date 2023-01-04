.. originally from ASTIN paper with John

.. _distortions:
.. _5_x_distortiona:

Distortions and Spectral Risk Measures
========================================

**Objectives:** Introduce distortion functions and spectral risk measures.

**Audience:** Readers looking for a deeper technical understanding.

**Prerequisites:** Knowledge of probability and calculus; insurance terminology.

**See also:** :doc:`5_x_distortion_calculations`.


**Contents:**

* :ref:`dist hr`
* :ref:`Distortion Function and Spectral Risk Measures`
* :ref:`Layer Densities`
* :ref:`The Equal Priority Default Rule`
* :ref:`Expected Loss Payments at Different Asset Levels`
* :ref:`The Natural Allocation Premium`
* :ref:`Properties of Alpha, Beta, and Kappa`
* :ref:`Properties of the Natural Allocation`
* :ref:`The Natural Allocation of Equity`
* :ref:`Appendix: Notation and Conventions`

.. _dist hr:

Helpful References
--------------------

* :cite:t:`PIR`
* The text in this section is derived from :cite:t:`Major2020`.
* :cite:t:`Mildenhall2022`

Distortion Function and Spectral Risk Measures
-------------------------------------------------

We define SRMs and recall results describing their different
representations. By :cite:t:`DeWaegenaere2003` SRMs are consistent with general
equilibrium and so it makes sense to consider them as pricing
functionals. The SRM is interpreted as the (ask) price for an
insurer-written risk transfer.

.. container:: definition

   **Definition.** A **distortion function** is an increasing concave function :math:`g:[0,1]\to [0,1]` satisfying :math:`g(0)=0` and :math:`g(1)=1`.

   A **spectral risk measure** :math:`\rho_g` associated with a distortion :math:`g` acts on a non-negative random variable :math:`X` as

   .. math::
       \rho_g(X) = \int_0^\infty g(S(x))dx.

The simplest distortion if the identity :math:`g(s)=s`. Then
:math:`\rho_g(X)=\mathsf E[X]` from the integration-by-parts identity

.. math::


   \int_0^\infty S(x)\,dx = \int_0^\infty xdF(x).

Other well-known distortions include the **proportional hazard**
:math:`g(s)=s^r` for :math:`0<r\le 1`, its **dual** :math:`g(s)=1-(1-s)^r`
for :math:`r\ge 1`, and the **Wang transform**
:math:`g(s)=\Phi(\Phi^{-1}(s)+\lambda)` for :math:`\lambda \ge 0`,
:cite:t:`Wang1995`.

Since :math:`g` is concave :math:`g(s)\ge 0g(0) + sg(1)=s` for all
:math:`0\le s\le 1`, showing :math:`\rho_g` adds a non-negative margin.

Going forward, :math:`g` is a distortion and :math:`\rho` is its
associated distortion risk measure. We interpret :math:`\rho` as a
pricing functional and refer to :math:`\rho(X)` as the price or premium
for insurance on :math:`X`.

SRMs are **translation invariant**, **monotonic**, **subadditive**, and **positive
homogeneous**, and hence **coherent**, :cite:t:`Acerbi2002b`. In addition, they are **law
invariant** and **comonotonic additive**. In fact, all such functionals are
SRMs. As well has having these properties, SRMs are powerful because we
have a complete understanding of their representation and structure,
which we summarize in the following theorem.

.. container:: theorem

   **Theorem.**
   Subject to :math:`\rho` satisfying certain continuity assumptions, the following are equivalent.

   #. :math:`\rho` is a law invariant, coherent, comonotonic additive risk measure.
   #. :math:`\rho=\rho_g` for a concave distortion :math:`g`.
   #. :math:`\rho` has a representation as a weighted average of TVaRs for a measure :math:`\mu` on :math:`[0,1]`:  :math:`\rho(X)=\int_0^1 \mathsf{TVaR}_p(X)\mu(dp)`.
   #. :math:`\rho(X)=\max_{\mathsf Q\in\mathscr{Q}} \mathsf E_{\mathsf Q}[X]` where :math:`\mathscr{Q}` is the set of (finitely) additive measures  with :math:`\mathsf Q(A)\le g(\mathsf P(A))` for all measurable :math:`A`.
   #. :math:`\rho(X)=\max_{\mathsf Z\in\mathscr{Z}} \mathsf E[XZ]` where :math:`\mathscr{Z}` is the set of positive functions on :math:`\Omega` satisfying :math:`\int_p^1 q_Z(t)dt \le g(1-p)`, and :math:`q_Z` is the quantile function of :math:`Z`.

The Theorem combines results from :cite:t:`Follmer2011` (4.79, 4.80, 4.93, 4.94,
4.95), :cite:t:`Delbaen2000`, :cite:t:`Kusuoka2001`, and :cite:t:`Carlier2003`. It
requires that :math:`\rho` is continuous from above to rule out the
possibility :math:`\rho=\sup`. In certain situations, the :math:`\sup`
risk measure applied to an unbounded random variable can only be
represented as a :math:`\sup` over a set of test measures and not a max.
Note that the roles of from above and below are swapped from
:cite:t:`Follmer2011` because they use the asset, negative is bad, sign
convention whereas we use the actuarial, positive is bad, convention.

The relationship between :math:`\mu` and :math:`g` is given by
:cite:t:`Follmer2011` 4.69 and 4.70. The elements of :math:`\mathscr Z` are the
Radon-Nikodym derivatives of the measures in :math:`\mathscr Q`.

The next four sections introduce the idea of layer densities and prove that SRM
premium can be allocated to policy in a natural and unique way.

Layer Densities
-----------------

Risk is often tranched into layers that are then insured and priced
separately. :cite:t:`Meyers1996` describes layering in the context of liability
increased limits factors and :cite:t:`Culp2009`, :cite:t:`Mango2013` in the context of
excess of loss reinsurance.

Define a layer :math:`y` excess :math:`x` by its payout function
:math:`I_{(x,x+y]}(X):=(X-x)^+\wedge y`. The expected layer loss is

.. math::

    \mathsf E[I_{(x,x+y]}(X)] &= \int_x^{x+y} (t-x)dF(t) + yS(x+y) \\
    &= \int_x^{x+y} t dF(t) + tS(t)\vert_x^{x+y} \\
    &= \int_x^{x+y} S(t)\, dt.

Based on this equation, :cite:t:`Wang1996` points out that
:math:`S` can be interpreted as the layer loss (net premium) density.
Specifically, :math:`S` is the layer loss density in the sense that
:math:`S(x)=d/dx(\mathsf E[I_{(0, x]}(X)])` is the marginal rate of increase in
expected losses in the layer at :math:`x`. We use density in this sense
to define premium, margin and equity densities, in addition to loss
density.

Clearly :math:`S(x)` equals the expected loss to the cover
:math:`1_{\{X>x\}}`. By scaling, :math:`S(x)dx` is the close to the
expected loss for :math:`I_{(x, x+dx]}` when :math:`dx` is very small;
:cite:t:`Bodoff2007` calls these infinitesimal layers.

:cite:t:`Wang1996` goes on to interpret

.. math::

   \int_x^{x+y} g(S(t))\,dt

as the layer premium and hence :math:`g(S(x))` as the layer premium
density. We write :math:`P(x):=g(S(x))` for the premium density.

We can decompose :math:`X` into a sum of thin layers. All these layers
are comonotonic with one another and with :math:`X`, resulting in an
additive decomposition of :math:`\rho(X)`, since :math:`\rho` is
comonotonic additive. The decomposition mirrors the definition of
:math:`\rho` as an integral.

The amount of assets :math:`a` available to pay claims determines the
quality of insurance, and premium and expected losses are functions of
:math:`a`. Premiums are well-known to be sensitive to the insurer’s
asset resources and solvency, :cite:t:`Phillips1998`. Assets may be infinite,
implying unlimited cover. When :math:`a` is finite there is usually some
chance of default. Using the layer density view, define expected loss
:math:`\bar S` and premium :math:`\bar P` functions as

.. math::

    \bar S(a) &= \mathsf E[X\wedge a]=\int_0^a S(x)\,dx \label{eq:sbar-def} \\
    \bar P(a) &= \rho(X\wedge a) = \int_0^\infty g(S_{X\wedge a}(x))\,dx \\
              &=\int_0^a g(S_{X}(x))dx. \label{eq:prem-def}


Margin is :math:`\bar M(a):=\bar P(a)-\bar S(a)` and margin density is
:math:`M(a)=d\bar M(a)/da`. Assets are funded by premium and equity
:math:`\bar Q(a):=a-\bar P(a)`. Again :math:`Q(a)=d\bar Q/da = 1-P(a)`.
Together :math:`S`, :math:`M`, and :math:`Q` give the split of layer
funding between expected loss, margin and equity. Layers up to :math:`a`
are, by definition, fully collateralized. Thus :math:`\rho(X\wedge a)`
is the premium for a defaultable cover on :math:`X` supported by assets
:math:`a`, whereas :math:`\rho(X)` is the premium for an unlimited,
default-free cover.

The layer density view is consistent with more standard approaches to
pricing. If :math:`X` is a Bernoulli risk with :math:`\Pr(X=1)=s` and
expected loss cost :math:`s`, then :math:`\rho(X)=g(s)` can be regarded
as pricing a unit width layer with attachment probability :math:`s`. In
an intermediated context, the funding constraint requires layers to be
fully collateralized by premium plus equity—without such funding the
insurance would not be credible since the insurer has no other source of
funds.

Given :math:`g` we can compute insurance market statistics for each
layer. The loss, premium, margin, and equity densities are :math:`s`,
:math:`g(s)`, :math:`g(s)-s` and :math:`1-g(s)`. The layer loss ratio is
:math:`s/g(s)` and :math:`(g(s)-s)/(1-g(s))` is the layer return on
equity. These quantities are illustrated in the next figure
for a typical distortion function. The corresponding statistics for
ground-up covers can be computed by integrating densities.

.. ipython:: python
    :okwarning:

    from aggregate.extensions.pir_figures import fig_10_3
    @savefig dist_g_fig.png scale=20
    fig_10_3()

For an insured risk, we regard the margin as compensation for
ambiguity aversion and associated winner’s curse drag. Both of these
effects are correlated with risk, so the margin is hard to distinguish
from a risk load, but the rationale is different. Again, recall,
although :math:`\rho` is non-additive and appears to charge for
diversifiable risk, :cite:t:`DeWaegenaere2003` assures us the pricing is
consistent with a general equilibrium.

The layer density is distinct from models that vary the volume of each
unit in a homogeneous portfolio model. Our portfolio is static. By
varying assets we are implicitly varying the quality of insurance.

The Equal Priority Default Rule
----------------------------------

If assets are finite and the provider has limited liability we need to
to determine policy-level cash flows in default states before we can
determine the fair market value of insurance. The most common way to do
this is using equal priority in default.

Under limited liability, total losses are split between provider
payments and provider default as

.. math::
   X = X\wedge a + (X-a)^+.

Next, actual payments :math:`X\wedge a` must be allocated to each
policy.

:math:`X_i` is the amount promised to :math:`i` by their insurance
contract. Promises are limited by policy provisions but are not limited
by provider assets. At the policy level, equal priority implies the
payments made to, and default borne by, policy :math:`i` are split as

.. math::

    X_i
    &= X_i \frac{X\wedge a}{X} + X_i \frac{(X-a)^+}{X} \\
    &= (\text{payments to policy $i$}) + (\text{default borne by policy $i$}).

Therefore the payments made to policy :math:`i` are given

.. math::

    X_i(a) := X_i \frac{X\wedge a}{X}
    = \begin{cases}
          X_i  & X \le a \\
          X_i\dfrac{a}{X} & X > a.
    \end{cases}\label{eq:equal-priority}

:math:`X_i(a)` is the amount actually paid to policy
:math:`i`. It depends on :math:`a`, :math:`X` and :math:`X_i`. The
dependence on :math:`X` is critical. It is responsible for almost all
the theoretical complexity of insurance pricing.

It is worth reiterating that with this definition
:math:`\sum_i X_i(a)=X\wedge a`.

**Example.**

Here is an example illustrating the effect of equal
priority. Consider a certain loss :math:`X_0=1000` and :math:`X_1` given
by a lognormal with mean 1000 and coefficient of variation 2.0. Prudence
requires losses be backed by assets equal to the 0.9 quantile. On a
stand-alone basis :math:`X_0` is backed by :math:`a_0=1000` and is
risk-free. :math:`X_1` is backed by :math:`a_1=2272` and the recovery is
subject to a considerable haircut, since
:math:`\mathsf E[X_1\wedge 2272] = 732.3`. If these risks are pooled, the pool
must hold :math:`a=a_0+a_1` for the same level of prudence. When
:math:`X_1\le a_1` both units are paid in full. But when
:math:`X_1 > a_1`, :math:`X_0` receives :math:`1000(a/(1000+X_1))` and
:math:`X_1` receives the remaining :math:`X_1(a/(1000+X_1))`. Payment to
both units is pro rated down by the same factor
:math:`a/(1000+X_1)`—hence the name *equal* priority. In the pooled
case, the expected recovery to :math:`X_0` is 967.5 and 764.8 to
:math:`X_1`. Pooling and equal priority result in a transfer of 32.5
from :math:`X_0` to :math:`X_1`. This example shows what can occur when
a thin tailed unit pools with a thick tailed one under a weak capital
standard with equal priority. We shall see how pricing compensates for
these loss payment transfers, with :math:`X_1` paying a positive margin
and :math:`X_0` a negative one. The calculations are performed in ``aggregate`` as follows. First, set up the :class:`Portfolio`:

.. ipython:: python
    :okwarning:

    from aggregate import build, qd

    port = build('port Dist:EqPri '
                 'agg A 1 claim dsev [1000] fixed '
                 'agg B 1 claim sev lognorm 1000 cv 2 fixed',
                bs=4)
    qd(port)

:meth:`var_dict` returns the 90th percentile points by unit and in total.

.. ipython:: python
    :okwarning:

    port.var_dict(.9)

Extract the relevant fields from ``density_df`` for the allocated loss recoveries.
The first block shows standalone, the second pooled.

.. ipython:: python
    :okwarning:

    qd(port.density_df.filter(regex='S|lev_[ABt]').loc[[port.B.q(0.9)]])
    qd(port.density_df.filter(regex='S|exa_[ABt]').loc[[port.q(0.9)]])

Expected Loss Payments at Different Asset Levels
---------------------------------------------------

Expected losses paid to policy :math:`i` are
:math:`\bar S_i(a) := \mathsf E[X_i(a)]`. :math:`\bar S_i(a)` can be computed,
conditioning on :math:`X`, as

.. math::

    \bar S_i(a) = \mathsf E[\mathsf E[X_i(a)\mid X]] = \mathsf E[X_i \mid X \le a]F(a) + a\mathsf E\left[ \frac{X_i}{X}\mid X>a \right]S(a).


Because of its importance in allocating losses, define

.. math::

    \alpha_i(a) := \mathsf E[X_i/X\mid X> a].

The value :math:`\alpha_i(x)` is the expected proportion
of recoveries by unit :math:`i` in the layer at :math:`x`. Since total
assets available to pay losses always equals the layer width, and the
chance the layer attaches is :math:`S(x)`, it is intuitively clear
:math:`\alpha_i(x)S(x)` is the loss density for unit :math:`i`, that is,
the derivative of :math:`\bar S_i(x)` with respect to :math:`x`. We now
show this rigorously.

.. container:: prop

    **Proposition.** Expected losses to policy :math:`i` under equal priority, when total losses are supported by assets :math:`a`, is given by

    .. math::
        \label{eq:alpha-S}
        \bar S_i(a) =\mathsf E[X_i(a)] = \int_0^a \alpha_i(x)S(x)dx

    and so the policy loss density at :math:`x` is :math:`S_i(x):=\alpha_i(x)S(x)`.

    *Proof.* By the definition of conditional expectation, :math:`\alpha_i(a)S(a)=\mathsf E[(X_i/X)1_{X>a}]`. Conditioning on :math:`X`, using the tower property, and taking out the functions of :math:`X` on the right shows

    .. math::
       \alpha_i(a)S(a)=\mathsf E[\mathsf E[(X_i/X) 1_{X>a}\mid X]]=\int_a^\infty \mathsf E[X_i \mid X=x]\dfrac{f(x)}{x}dx

    and therefore

    .. math::
       \frac{d}{da}(\alpha_i(a)S(a)) = -\mathsf E[X_i \mid X=a]\dfrac{f(a)}{a}.

    Now we can use integration by parts to compute

    .. math::
        \int_0^a \alpha_i(x)S(x)\,dx
        &= x\alpha_i(x)S(x)\Big\vert_0^a + \int_0^a x\,\mathsf E[X_i \mid X=x]\dfrac{f(x)}{x}\,dx\\
        &= a\alpha_i(a)S(a) + E[X_i \mid X\le a]F(a) \\
        &=  \bar S_i(a).

    Therefore the policy :math:`i` loss density in the asset layer at :math:`a`, i.e. the derivative of \cref{eq:eloss-main} with respect to :math:`a`, is :math:`S_{i}(a)=\alpha_i(a) S(a)` as required.

Note that :math:`S_i` is *not* the survival function of :math:`X_i(a)` nor of
:math:`X_i`.

The Natural Allocation Premium
--------------------------------

Premium under :math:`\rho` is given by :math:`\int_0^a g(S)`.
We can interpret :math:`g(S(a))` as the portfolio premium density in the
layer at :math:`a`. We now consider the premium and premium density for
each policy.

Using integration by parts we can express the price of an unlimited
cover on :math:`X` as

.. math::

    \label{eq:nat1}
    \rho(X)=\int_0^\infty g(S(x))\,dx = \int_0^\infty xg'(S(x))f(x)\,dx = \mathsf E[Xg'(S(X)))].

It is important that this integral is over all
:math:`x\ge 0` so the :math:`xg(S(x))\vert_0^a` term disappears.
The formula makes sense because a concave distortion is
continuous on :math:`(0,1]` and can have at most countably infinitely
many points where it is not differentiable (it has a kink). In total
these points have measure zero, :cite:t:`Borwein2010`, and we can ignore them in
the integral. For more details see :cite:t:`Dhaene2012b`.

Combining the integral  and the properties of a distortion
function, :math:`g'(S(X))` is the Radon-Nikodym derivative of a measure
:math:`\mathsf Q` with :math:`\rho(X)=\mathsf E_{\mathsf Q}[X]`. In fact,
:math:`\mathsf E_{\mathsf Q}[Y]=\mathsf E[Yg'(S(X))]` for all random variables
:math:`Y`. In general, any non-negative function :math:`Z` (measure
:math:`\mathsf Q`) with :math:`\mathsf E[Z]=1` and :math:`\rho(X)=\mathsf E[XZ]`
(:math:`=\mathsf E_{\mathsf Q}[X]`) is called a contact function (subgradient)
for :math:`\rho` at :math:`X`, see :cite:t:`Shapiro2009`. Thus :math:`g'(S(X))`
is a contact function for :math:`\rho` at :math:`X`. The name
subgradient comes from the fact that
:math:`\rho(X+Y)\ge \mathsf E_{\mathsf Q}[X+Y] = \rho(X) + \mathsf E_{\mathsf Q}[Y]`,
by the representation theorem.  The set of subgradients is called
the subdifferential of :math:`\rho` at :math:`X`. If there is a unique
subgradient then :math:`\rho` is differentiable. :cite:t:`Delbaen2000` Theorem 17
shows that subgradients are contact functions.

We can interpret :math:`g'(S(X))` as a state price density specific to
the :math:`X`, suggesting that :math:`\mathsf E[X_ig'(S(X))]` gives the value
of the cash flows to policy :math:`i`. This motivates the following
definition.

.. container:: def

   **Definition.** For :math:`X=\sum_i X_i` with :math:`\mathsf Q\in\mathcal Q` so that :math:`\rho(X)=\mathsf E_{\mathsf Q}[X]`, the **natural allocation premium** to policy :math:`X_j` as part of the portfolio :math:`X` is :math:`\mathsf E_{\mathsf Q}[X_j]`. It is denoted :math:`\rho_X(X_j)`.

The natural allocation premium is a standard approach, appearing in
:cite:t:`Delbaen2000`, :cite:t:`Venter2006` and :cite:t:`Tsanakas2003a` for example. It has many
desirable properties. Delbaen shows it is a fair allocation in the sense
of fuzzy games and that it has a directional derivative, marginal
interpretation when :math:`\rho` is differentiable. It is consistent
with :cite:t:`Jouini2001` and :cite:t:`Campi2013`, which show the rational price of
:math:`X` in a market with frictions must be computed by state prices
that are anti-comonotonic :math:`X`. In our application the signs are
reversed: :math:`g'(S(X))` and :math:`X` are comonotonic.

The choice :math:`g'(S(X))` is economically meaningful because it
weights the largest outcomes of :math:`X` the most, which is appropriate
from a social, regulatory and investor perspective. It is also the only
choice of weights that works for all levels of assets. Since investors
stand ready to write any layer at the price determined by :math:`g`,
their solution must work for all :math:`a`.

However, there are two technical issues with the proposed natural
allocation. First, unlike prior works, we are allocating the premium for
:math:`X\wedge a`, not :math:`X`, a problem also considered in
:cite:t:`Major2018`. And second, :math:`\mathsf Q` may not be unique. In general,
uniqueness fails at capped variables like :math:`X\wedge a`. Both issues
are surmountable for a SRM, resulting in a unique, well defined natural
allocation. For a non-comonotonic additive risk measure this is not the
case.

It is helpful to define the premium, risk adjusted, analog of the
:math:`\alpha_i` as

.. math::
    \label{eq:beta-def}
    \beta_i(a) := \mathsf E_{\mathsf Q}[(X_i/X) \mid X > a].

:math:`\beta_i(x)` is the value of the recoveries paid
to unit :math:`i` by a policy paying 1 in states :math:`\{ X>a \}`,
i.e. an allocation of the premium for :math:`1_{X>a}`. By the properties
of conditional expectations, we have

.. math::
    \label{eq:beta-cond}
    \beta_i(a) = \frac{\mathsf E[(X_i/X) Z\mid X > a]}{\mathsf E[Z\mid X > a]}.

The denominator equals
:math:`\mathsf Q(X>a)/\mathsf P(X>a)`. Remember that while
:math:`\mathsf E_{\mathsf Q}[X]=\mathsf E[XZ]`, for conditional expectations
:math:`\mathsf E_{\mathsf Q}[X\mid \mathcal F]=\mathsf E[XZ\mid \mathcal F]/\mathsf E[Z\mid \mathcal F]`,
see [:cite:t:`Follmer2011`, Proposition A.12].

To compute :math:`\alpha_i` and :math:`\beta_i` we use a third function,

.. math::
    \label{eq:kappa-def}
    \kappa_i(x):= \mathsf E[X_i \mid X=x],

the conditional expectation of loss by policy, given the
total loss.

.. main theorem

.. container:: theorem

   **Theorem.** Let :math:`\mathsf Q\in \mathcal Q` be the measure with Radon-Nikodym derivative :math:`Z=g'(S_X(X))`. Then:

   #. :math:`\mathsf E[X_i \mid X=x]=\mathsf E_{\mathsf Q}[X_i \mid X=x]`.
   #. :math:`\beta_i` can be computed from :math:`\kappa_i` as

   .. math::
       \beta_i(a)= \frac{1}{\mathsf Q(X>a)}\int_a^\infty \dfrac{\kappa_i(x)}{x} g'(S(x))f(x)\, dx. \label{eq:beta-easy}

   #. The natural allocation premium for policy :math:`i` under equal priority when total losses are supported by assets :math:`a`, :math:`\bar P_i(a):=\rho_{X\wedge a}(X_i(a))`, is given by

   .. math::
       \bar P_i(a) &=
        \mathsf E_{\mathsf Q}[X_i \mid {X\le a}](1-g(S(a))) + a\mathsf E_{\mathsf Q}[X_i/X  \mid {X > a}]g(S(a)) \label{eq:pibar-main} \\
        &=\mathsf E[X_iZ\mid X\le a](1-S(a)) + a\mathsf E[(X_i/X)Z\mid X>a]S(a).

   #. The policy :math:`i` premium density equals

   .. math::
       P_i(a)=\beta_i(a)g(S(a)).
       \label{eq:beta-gS}


It is an important to know when the natural allocation premium is unique. It is so when :math:`Z` is the only contact function (i.e., there are no others).
If :math:`X` has a strictly increasing quantile function or is injective then :math:`\mathsf Q` is unique and therefore given by :math:`g'S(X)` and hence :math:`X` measurable, see :cite:p:`Carlier2003` and :cite:t:`Marinacci2004b`. More generally, we can replace :math:`\mathsf Q` with its expectation given :math:`X` to make a canonical choice, resulting in the linear natural allocation :cite:p:`Cherny2011`.

The problem that can occur when :math:`\mathsf Q` is not unique, but
that can be circumvented when :math:`\rho` is a SRM, can be illustrated
as follows. Suppose :math:`\rho` is given by :math:`p`-TVaR. The measure
:math:`\mathsf{Q}` weights the worst :math:`1-p` proportion of outcomes
of :math:`X` by a factor of :math:`(1-p)^{-1}` and ignores the others.
Suppose :math:`a` is chosen as :math:`p'`-VaR for a lower threshold
:math:`p'<p`. Let :math:`X_a=X\wedge a` be capped insured losses and
:math:`C=\{X_a=a\}`. By definition :math:`\Pr(C)\ge 1-p'>1-p`. Pick any
:math:`A\subset C` of measure :math:`1-p` so that
:math:`\rho(X)=\mathsf E[X\mid A]`. Let :math:`\psi` be a measure preserving
transformation of :math:`\Omega` that acts non-trivially on :math:`C`
but trivially off :math:`C`. Then :math:`\mathsf{Q}'=\mathsf Q\psi` will
satisfy
:math:`\mathsf E_{\mathsf{Q}'}[X_a]=\mathsf E_{\mathsf{Q}}[X_a\psi^{-1}]=\rho(X_a)`
but in general :math:`\mathsf E_{\mathsf{Q}'}[X]<\rho(X)`. The natural
allocation with respect to :math:`\mathsf{Q}'` will be different from
that for :math:`\mathsf{Q}`. The theorem isolates a specific
:math:`\mathsf Q` to obtain a unique answer. The same idea applies to
:math:`\mathsf Q` from other, non-TVaR, :math:`\rho`: you can always
shuffle part of the contact function within :math:`C` to generate
non-unique allocations.
See :cite:t:`PIR` Example 239 for an illustration.

When :math:`\mathsf Q` is :math:`X` measurable, then
:math:`\mathsf E_{\mathsf Q}[X_i \mid X]=\mathsf E[X_i \mid X]`, which enables explicit calculation. In this case there is no risk adjusted version of :math:`\kappa_i`. If :math:`\mathsf Q` is not :math:`X` measurable, then there can be risk adjusted :math:`\kappa_i` because

.. math::

    \mathsf E[X_i Z \mid X] \not= \mathsf E[X_i \mid X] \mathsf E[Z \mid X].


.. this is wrong
.. There is no risk adjusted version of
    :math:`\kappa_i`. Intuitively, a law invariant risk measure cannot
    change probabilities within an event defined by :math:`X`: if it did
    then it would be distinguishing between events on information other than
    :math:`S(X)` whereas law invariance says this is all that can matter.
    It also identifies the premium density, giving an allocation of
    total premium and a premium analog of the loss allocation.
    It provides a clear and illuminating way
    to visualize risk by collapsing a multidimensional problem to one
    dimension.


The proof writes the price of a limited liability cover as the price of
default-free protection minus the value of the default put. This is the
standard starting point for allocation in a perfect competitive market
taken by :cite:t:`Phillips1998`, :cite:t:`Myers2001`, :cite:t:`Sherris2006a`, and :cite:t:`Ibragimov2010`.
They then allocate the default put rather than the value of insurance
payments directly.


To recap: the premium formulas  have been derived assuming
capital is provided at a cost :math:`g` and there is equal priority by
unit. The formulas are computationally tractable (see implementation in :doc:`5_x_portfolio_calculations`) and require only that :math:`X` have an increasing quantile function or that :math:`g'S(X)` be used as the risk adjustment, but make no other
assumptions. There is no need to assume the :math:`X_i` are independent.
They produce an entirely general, canonical determination of premium in
the presence of shared costly capital. This result extends :cite:t:`Grundl2007`,
who pointed out that with an additive pricing functional there is no
need to allocate capital in order to price, to the situation of a
non-additive SRM pricing functional.

Properties of Alpha, Beta, and Kappa
--------------------------------------

In this section we explore properties of :math:`\alpha_i`,
:math:`\beta_i`, and :math:`\kappa_i`, and show how they interact to
determine premiums by unit via the natural allocation.

For a measurable :math:`h`, :math:`\mathsf E[X_ih(X)]=\mathsf E[\kappa_i(X)h(X)]` by
the tower property. This simple observation results in huge
simplifications. In general, :math:`\mathsf E[X_ih(X)]` requires knowing the
full bivariate distribution of :math:`X_i` and :math:`X`. Using
:math:`\kappa_i` reduces it to a one dimensional problem. This is true
even if the :math:`X_i` are correlated. The :math:`\kappa_i` functions
can be estimated from data using regression and they provide an
alternative way to model correlations.

Despite their central role, the :math:`\kappa_i` functions are probably
unfamiliar so we begin by giving several examples to illustrate how they
behave. In general, they are non-linear and usually, but not always,
increasing.

Examples of :math:`\kappa` functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. If :math:`Y_i` are independent and identically distributed and
   :math:`X_n=Y_1+\cdots +Y_n` then
   :math:`\mathsf E[X_m\mid X_{m+n}=x]=mx/(m+n)` for :math:`m\ge 1, n\ge 0`.
   This is obvious when :math:`m=1` because the functions
   :math:`\mathsf E[Y_i\mid X_n]` are independent across :math:`i=1,\ldots,n`
   and sum to :math:`x`. The result follows because conditional
   expectations are linear. In this case :math:`\kappa_i(x)=mx/(m+n)` is
   a line through the origin.

2. If :math:`X_i` are multivariate normal then :math:`\kappa_i` are
   straight lines, given by the usual least-squares fits

   .. math::
      \kappa_i(x)= \mathsf E[X_i] + \frac{\mathsf{cov}(X_i,X)}{\mathsf{var}(X)}(x-\mathsf E[X]).

   This example is familiar from the securities market line and the CAPM
   analysis of stock returns. If :math:`X_i` are iid it reduces to the
   previous example because the slope is :math:`1/n`.

3. If :math:`X_i`, :math:`i=1,2`, are compound Poisson with the same
   severity distribution then :math:`\kappa_i` are again lines through
   the origin. Suppose :math:`X_i` has expected claim count
   :math:`\lambda_i`. Write the conditional expectation as an integral,
   expand the density of the compound Poisson by conditioning on the
   claim count, and then swap the sum and integral to see that
   :math:`\kappa_1(x)=\mathsf E[X_1\mid X_1 + X_2=x]=x\,\mathsf E[N(\lambda_1)/(N(\lambda_1)+N(\lambda_2))]`
   where :math:`N(\lambda)` are independent Poisson with mean
   :math:`\lambda`. This example generalizes the iid case. Further
   conditioning on a common mixing variable extends the result to mixed
   Poisson frequencies where each aggregate can have a separate or
   shared mixing distribution. The common severity is essential. The
   result means that if a line of business is defined to be a group of
   policies that shares the same severity distribution, then premiums
   for policies within the line will have rates proportional to their
   expected claim counts.

4. A theorem of Efron says that if :math:`X_i` are independent and have
   log-concave densities then all :math:`\kappa_i` are non-decreasing,
   :cite:t:`Saumard2014`. The multivariate normal example is a special case of
   Efron’s theorem.

:cite:t:`Denuit2012` define an ex post risk sharing rule called the conditional
mean risk allocation by taking :math:`\kappa_i(x)` to be the allocation
to policy :math:`i` when :math:`X=x`. A series of recent papers, see
:cite:t:`Denuit2020e` and references therein, considers the properties of the
conditional mean risk allocation focusing on its use in peer-to-peer
insurance and the case when :math:`\kappa_i(x)` is linear in :math:`x`.


Properties of the Natural Allocation
-----------------------------------------

We now explore margin, equity, and return in total and by policy. We
begin by considering them in total.

By definition the average return with assets :math:`a` is

.. math::
    \label{eq:avg-roe}
    \bar\iota(a) := \frac{\bar M(a)}{\bar Q(a)}

where margin :math:`\bar M` and equity :math:`\bar Q`
are the total margin and capital functions defined above.

The last formula has important implications. It tells us
the investor priced expected return varies with the level of assets. For
most distortions return decreases with increasing capital. In contrast,
the standard RAROC models use a fixed average cost of capital,
regardless of the overall asset level, :cite:t:`Tasche1999`. CAPM or the
Fama-French three factor model are often used to estimate the average
return, with a typical range of 7 to 20 percent, :cite:t:`Cummins2005`. A common
question of working actuaries performing capital allocation is about
so-called excess capital, if the balance sheet contains more capital
than is required by regulators, rating agencies, or managerial prudence.
Our model suggests that higher layers of capital are cheaper, but not
free, addressing this concern.

The varying returns may seem
inconsistent with Miller-Modigliani. But that says the cost of funding a
given amount of capital is independent of how it is split between debt
and equity; it does not say the average cost is constant as the amount
of capital varies.

No-Undercut and Positive Margin for Independent Risks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The natural allocation has two desirable properties. It is always less
than the stand-alone premium, meaning it satisfies the no-undercut
condition of :cite:t:`Denault2001`, and it produces non-negative margins for
independent risks.

.. container:: prop

   **Proposition.**
   Let :math:`X=\sum_{i=1}^n X_i`, :math:`X_i` non-negative and independent, and let :math:`g` be a distortion. Then

   #. the natural allocation is never greater than the stand-alone premium, and
   #. the natural allocation to every :math:`X_i` contains a non-negative margin.


Since :math:`\bar P_i = \mathsf E[\kappa_i(X)g'(S(X))]` we see the no-undercut
condition holds if :math:`\kappa_i(X)` and :math:`g'(S(X))` are
comonotonic, and hence if :math:`\kappa_i` is increasing, or if
:math:`\kappa_i(X)` and :math:`X` are positively correlated (recall
:math:`\mathsf E[g'(S(X))]=1`).  A policy
:math:`i^*` with increasing :math:`\kappa_{i^*}` is a capacity consuming line that always has a positive margin. However, it can occur that no :math:`\kappa_i` is increasing.

.. awkward sequence !

Policy Level Properties, Varying with Asset Level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We start with a corollary which gives a
nicely symmetric and computationally tractable expression for the
natural margin allocation in the case of finite assets.

.. container:: cor

    **Corollary.**
    The margin density for unit :math:`i` at asset level :math:`a` is given by

    .. math::
        \label{eq:coc-by-line}
        M_i(a) =\beta_i(a)g(S(a)) -  \alpha_i(a)S(a).

    *Proof.* We can compute margin
    :math:`\bar M_i(a)` in :math:`\bar P_i(a)` by line as

    .. math::

        \bar M_i(a)=& \bar P_i(a) - \bar L_i(a) \nonumber \\
        =& \int_0^a \beta_i(x)g(S(x)) -  \alpha_i(x)S(x)\,dx.  \label{eq:margin-by-line}

    Differentiating we get the margin density for unit
    :math:`i` at :math:`a` expressed in terms of :math:`\alpha_i` and
    :math:`\beta_i` as shown.

Margin in the current context is the cost of capital, thus this
is an important result. It allows us
to compute economic value by unit and to assess static portfolio
performance by unit—one of the motivations for performing capital
allocation in the first place. In many ways it is also a good place to
stop. Remember these results only assume we are using a distortion risk
measure and have equal priority in default. We are in a static model, so
questions of portfolio homogeneity are irrelevant. We are not assuming
:math:`X_i` are independent.

What can we say about by margins by
unit? Since :math:`g` is increasing and concave
:math:`P(a)=g(S(a))\ge S(a)` for all :math:`a\ge 0`. Thus all asset
layers contain a non-negative total margin density. It is a different
situation by unit, where we can see

.. math::
   M_i(a) \ge 0 \iff
   \beta_i(a)g(S(a)) - \alpha_i(a)S(a)\ge 0 \iff
   \frac{\beta_i(a)}{\alpha_i(a)} \ge \frac{S(a)}{g(S(a))}.

The unit layer margin density is positive when :math:`\beta_i/\alpha_i`
is greater than the all-unit layer loss ratio. Since the loss ratio is
:math:`\le 1` there must be a positive layer margin density whenever
:math:`\beta_i(a)/\alpha_i(a) > 1`. But when
:math:`\beta_i(a)/\alpha_i(a) < 1` it is possible the unit has a
negative margin density. How can that occur and why does it make sense?
To explore this we look at the shape of :math:`\alpha` and :math:`\beta`
in more detail.

It is important to remember why the Proposition does
not apply: it assumes unlimited cover, whereas here :math:`a<\infty`.
With finite capital there are potential transfers between units caused
by their behavior in default that overwhelm the positive margin implied
by the proposition. Also note the proposition cannot be applied to
:math:`X\wedge a=\sum_i X_i(a)` because the unit payments are no longer
independent.

In general we can make two predictions about margins.

**Prediction 1**: Lines where :math:`\alpha_i(x)` or
:math:`\kappa_i(x)/x` increase with :math:`x` will have always have a
positive margin.

**Prediction 2**: A log-concave (thin tailed) unit aggregated with a
non-log-concave (thick tailed) unit can have a negative margin,
especially for lower asset layers.

Prediction 1 follows because the risk adjustment puts more weight on
:math:`X_i/X` for larger :math:`X` and so
:math:`\beta_i(x)/\alpha_i(x)> 1 > S(x) / g(S(x))`. Recall the risk
adjustment is comonotonic with total losses :math:`X`.

A thin tailed unit aggregated with thick tailed units will have
:math:`\alpha_i(x)` decreasing with :math:`x`. Now the risk adjustment
will produce :math:`\beta_i(x)<\alpha_i(x)` and it is possible that
:math:`\beta_i(x)/\alpha_i(x)<S(x)/g(S(x))`. In most cases,
:math:`\alpha_i(x)` approaches :math:`\mathsf E[X_i]/x` and
:math:`\beta_i(x)/\alpha_i(x)` increases with :math:`x`, while the layer
loss ratio decreases—and margin increases—and the thin unit will
eventually get a positive margin. Whether or not the thin unit has a
positive total margin :math:`\bar M_i(a)>0` depends on the particulars
of the units and the level of assets :math:`a`. A negative margin is
more likely for less well capitalized insurers, which makes sense
because default states are more material and they have a lower overall
dollar cost of capital. In the independent case, as :math:`a\to\infty`
the proposition guarantees an eventually positive margins for all units.

These results are reasonable. Under limited liability, if assets and
liabilities are pooled then the thick tailed unit benefits from pooling
with the thin one because pooling increases the assets available to pay
losses when needed. Equal priority transfers wealth from thin to thick
in states of the world where thick has a bad event. But because thick
dominates the total, the total losses are bad when thick is bad. The
negative margin compensates the thin-tailed unit for transfers.

Another interesting situation occurs for asset levels within attritional
loss layers. Most realistic insured loss portfolios are quite skewed and
never experience very low loss ratios. For low loss layers, :math:`S(x)`
is close to 1 and the layer at :math:`x` is funded almost entirely by
expected losses; the margin and equity density components are nearly
zero. Since the sum of margin densities over component units equals the
total margin density, when the total is zero it necessarily follows that
either all unit margins are also zero or that some are positive and some
are negative. For the reasons noted above, thin tailed units get the
negative margin as thick tailed units compensate them for the improved
cover the thick tail units obtain by pooling.

In conclusion, the natural margin by unit reflects the relative
consumption of assets by layer, :cite:t:`Mango2005a`. Low layers are less
ambiguous to the provider and have a lower margin relative to expected
loss. Higher layers are more ambiguous and have lower loss ratios. High
risk units consume more higher layer assets and hence have a lower loss
ratio. For independent units with no default the margin is always
positive. But there is a confounding effect when default is possible.
Because more volatile units are more likely to cause default, there is a
wealth transfer to them. The natural premium allocation compensates low
risk policies for this transfer, which can result in negative margins in
some cases.

The Natural Allocation of Equity
------------------------------------

Although we have a margin by unit,
we cannot compute return by unit, or allocate frictional costs of
capital, because we still lack an equity allocation, a problem we now
address.

.. container:: def

   **Definition.**
   The **natural allocation of equity** to unit :math:`i` is given by

   .. math::
       Q_i(a) = \frac{\beta_i(a)g(S(a)) -  \alpha_i(x)S(a)}{g(S(a))- S(a)} \times (1-g(S(a))). \label{eq:main-alloc}

Why is this allocation natural? In total the layer return at :math:`a`
is

.. math::
   \iota(a) := \frac{M(a)}{Q(a)} = \frac{P(a) - S(a)}{1-P(a)} = \frac{g(S(a)) - S(a)}{1- g(S(a))}.

We claim that for a law invariant pricing measure the layer return *must
be the same for all units*. Law invariance implies the risk measure is
only concerned with the attachment probability of the layer at
:math:`a`, and not with the cause of loss within the layer. If return
*within a layer* varied by unit then the risk measure could not be law
invariant.

We can now compute capital by layer by unit, by solving for the unknown
equity density :math:`Q_i(a)` via

.. math::
   \iota(a) = \frac{M(a)}{Q(a)} = \frac{M_i(a)}{Q_i(a)}\implies Q_i(a) = \frac{M_i(a)}{\iota(a)}.

Substituting for layer return and unit margin gives the result.

Since :math:`1-g(S(a))` is the proportion of capital in the layer at
:math:`a`, the main allocation result says the allocation to unit
:math:`i` is given by the nicely symmetric expression

.. math::
    \label{eq:q-formula}
    \frac{\beta_i(a)g(S(a)) -  \alpha_i(x)S(a)}{g(S(a))- S(a)}.

To determine total capital by unit we integrate the
equity density

.. math::
   \bar Q_i(a) := \int_0^a Q_i(x) dx.

And finally we can determine the average return to unit :math:`i` at
asset level :math:`a`

.. math::
    \label{eq:avg-roe-by-unit}
    \bar\iota_i(a) = \frac{\bar M_i(a)}{\bar Q_i(a)}.


The average return will generally vary by unit and by
asset level :math:`a`. Although the return within each layer is the same
for all units, the margin, the proportion of capital, and the proportion
attributable to each unit all vary by :math:`a`. Therefore average
returns will vary by unit and :math:`a`. This is in stark contrast to
the standard industry approach, which uses the same return for each unit
and implicitly all :math:`a`. How these quantities vary by unit is
complicated. Academic approaches emphasized the possibility that returns
vary by unit, but struggled with parameterization, :cite:t:`Myers1987`.

This formula shows the average return by unit
is an :math:`M_i`-weighted harmonic mean of the layer returns given by
the distortion :math:`g`, viz

.. math::
   \frac{1}{\bar\iota_i(a)} = \int_0^a \frac{1}{\iota(x)}\frac{M_i(x)}{\bar M_i(a)}\,dx.

The harmonic mean solves the problem that the return for lower layers of
assets is potentially infinite (when :math:`g'(1)=0`). The infinities do
not matter: at lower asset layers there is little or no equity and the
layer is fully funded by the loss component of premium. When so funded,
there is no margin and so the infinite return gets zero weight. In this
instance, the sense of the problem dictates that
:math:`0\times\infty=0`: with no initial capital there is no final
capital regardless of the return.


Appendix: Notation and Conventions
-----------------------------------

An insurer has finite assets and limited
liability and is a one-period stock company. At :math:`t=0`
it sells its residual value to investors to raise equity. At time one it
pays claims up to the amount of assets available. If assets are
insufficient to pay claims it defaults. If there are excess assets they
are returned to investors.

Total insured loss, or total risk, is described by a random variable
:math:`X\ge 0`. :math:`X` reflects policy limits but is not limited by
provider assets. :math:`X=\sum_i X_i` describes the split of losses by
policy. :math:`F`, :math:`S`, :math:`f`, and :math:`q` are the
distribution, survival, density, and (lower) quantile functions of
:math:`X`. Subscripts are used to disambiguate, e.g., :math:`S_{X_i}` is
the survival function of :math:`X_i`. :math:`X\wedge a` denotes
:math:`\min(X,a)` and :math:`X^+=\max(X,0)`.

The letters :math:`S`, :math:`P`, :math:`M` and :math:`Q` refer to
expected loss, premium, margin and equity, and :math:`a` refers to
assets. The value of survival function :math:`S(x)` is the loss cost of
the insurance paying :math:`1_{\{X>x\}}`, so the two uses of :math:`S`
are consistent. Premium equals expected loss plus margin; assets equal
premium plus equity. All these quantities are functions of assets
underlying the insurance.

We use the actuarial sign convention: large positive values are bad. Our
concern is with quantiles :math:`q(p)` for :math:`p` near 1. Distortions
are usually reversed, with :math:`g(s)` for small :math:`s=1-p`
corresponding to bad outcomes. As far as possible we will use :math:`p`
in the context :math:`p` close to 1 is bad and :math:`s` when small
:math:`s` is bad.

Tail value at risk is defined for :math:`0\le p<1` by

.. math::


   \mathsf{TVaR}_p(X) = \frac{1}{1-p}\int_p^1 q(t)dt.

Prices exclude all expenses. The risk free interest rate is zero. These
are standard simplifying assumptions, e.g. :cite:t:`Ibragimov2010`.

The terminology describing risk measures is standard, and follows
:cite:t:`Follmer2011`. We work on a standard probability space, :cite:t:`Svindland2009`,
Appendix. It can be taken as :math:`\Omega=[0,1]`, with the Borel
sigma-algebra and :math:`\mathsf P` Lebesgue measure. The indicator
function on a set :math:`A` is :math:`1_A`, meaning :math:`1_A(x)=1` if
:math:`x\in A` and :math:`1_A(x)=0` otherwise.
