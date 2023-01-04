.. _quantiles:

.. from Ch 4 in PIR

Quantiles and Related Measures
==============================

**Objectives:** Definition and calculation of quantiles and related risk measures.

**Audience:** Readers interested in precise definitions of common risk measures.

**Prerequisites:** Risk measures, probability.

**See also:** :doc:`5_x_insurance_probability`.

**Contents:**

* :ref:`q hr`
* :ref:`quantiles`
* :ref:`Value at Risk`
* :ref:`q not sa`
* :ref:`q tvar`

.. _q hr:

Helpful References
--------------------

* :cite:t:`LM`
* :cite:t:`PIR`, Chapter 4
* :cite:t:`Hyndman1996`

..  Quantiles are the fundamental building block risk measure.
    Value at risk (VaR) = quantiles when used as a risk measure.
    Tail value at risk (TVaR).

Quantiles
---------

A quantile function is inverse to the distribution function
:math:`F(x):=\mathsf{Pr}(X\le x)`. For each :math:`0 < p < 1`, it solves
:math:`F(x)=p` for :math:`x`, answering the question,

   which :math:`x` has non-exceedance probability equal to :math:`p`?

Or, said another way,

   which :math:`x` has exceedance probability equal to :math:`1-p`?

When the distribution function is continuous and strictly increasing
there is a unique such :math:`x`. It is called the :math:`p`-quantile,
and is denoted :math:`q(p)`. The resulting function
:math:`q(p)=F^{-1}(p)` is called the quantile function; it satisfies
:math:`F(q(p))=p`.

Two issues arise when defining quantiles.

1. The equation :math:`F(x)=p` may fail to have a *unique* solution when
   :math:`F` is not strictly increasing. This can occur for any
   :math:`F`. Is corresponds to a range of *impossible* outcome values.

2. When :math:`F` is not continuous, the equation :math:`F(x)=p` may
   have *no solution*: :math:`F` can jump from below :math:`p` to above
   :math:`p`. Simulation and catastrophe models, and all discrete random
   variables have discontinuous distributions.

**Example.**

Here's an example of the problems that can occur.

.. ipython:: python
   :okwarning:

   from aggregate.extensions.pir_figures import fig_4_1
   @savefig quantiles2.png scale=20
   fig = fig_4_1()


The distribution :math:`F` has a flat spot between 0.9 and 1.5 at height
:math:`p=0.417`. At :math:`x=1.5` it jumps up to :math:`p=0.791`. The
“inverse” to :math:`F` at :math:`p=0.417` could be any value between 0.9
and 1.5—illustrated by the lower green horizontal dashed line. The inverse at
any value :math:`0.417 < p < 0.791` does not exist because there is no
:math:`p` so that :math:`F(p)=0.6`. However, any rational person looking
at the graph would agree that the answer must be :math:`x=1.5`, where
the black dashed line intersects the vertical line :math:`x=1.5`.

When :math:`F` is not continuous and :math:`F(x)=p` has no solution
because :math:`p` lies is within a jump, we can still find an :math:`x`
so that

.. math::

   \mathsf{Pr}(X < x)\le p \le \mathsf{Pr}(X\le x).

:math:`\mathsf{Pr}(X<x)` equals the height of :math:`F` at the
bottom of the jump and :math:`\mathsf{Pr}(X\le x)` at the top. Turning this
around, we can also say :math:`\mathsf{Pr}(X\ge x)\ge 1-p\ge \mathsf{Pr}(X> x)`. At a
:math:`p` with no jump, :math:`\mathsf{Pr}(X=x)=0`,
:math:`\mathsf{Pr}(X < x)=p=\mathsf{Pr}(X\le x)`, and we have a well defined inverse, as
the lower line at :math:`p=0.283` illustrates.

The vertical
segment at :math:`x=1.5` between :math:`p=0.417` and :math:`p=0.791` is
not strictly a part of :math:`F`\ ’s graph, because a function must
associate a *unique* value to each :math:`x` in its domain. However,
filling in the vertical segment makes it easier to locate inverse values
by finding the graph’s intersection with the horizontal line at
:math:`p` and is recommended in @Rockafellar2014b. Mentally, you should
always *fill in* jumps in this way, treating the added segment as part
of the graph.

===========

**Definition.** Let :math:`X` be a random variable with distribution function :math:`F`
and let :math:`0 < p < 1`. Any :math:`x` satisfying

.. math::

   \mathsf{Pr}(X < x)\le p\le \mathsf{Pr}(X\le x)

is a :math:`p` **quantile** of :math:`X`. Any function
:math:`q(p)` satisfying

.. math::

   \mathsf{Pr}(X < q(p))\le p\le \mathsf{Pr}(X\le q(p))

for :math:`0\ < p < 1` is a
**quantile function** of :math:`X`.

.. _prob dice quantiles:

**Exercise.** What are the :math:`0.1` and :math:`1/6` quantiles for the
outcomes of the fair roll of a 6-sided die?

**Solution.** There are six outcomes :math:`\{1,2,3,4,5,6\}` each with
probability :math:`1/6`. The distribution function jumps at each
outcome.

1. For :math:`p=0.1` we seek :math:`x` so that
   :math:`\mathsf{Pr}(X < x) \le 0.1 \le \mathsf{Pr}(X\le x)`. We know
   :math:`0=\mathsf{Pr}(X<1)<\mathsf{Pr}(X\le 1)=1/6` and therefore :math:`q(0.1)=1`. It
   is good to rule out other possible values. If :math:`x<1` then
   :math:`\mathsf{Pr}(X\le x)=0` and if :math:`x>1` then
   :math:`\mathsf{Pr}(X < x)\ge 1/6`, showing neither alternative satisfies the
   definition of a quantile.
2. For :math:`p=1/6` we seek :math:`x` so that
   :math:`\mathsf{Pr}(X < x) \le 1/6 \le \mathsf{Pr}(X\le x)`, which is satisfied by any
   :math:`1\le x \le 2`. If we pick :math:`x=1` then
   :math:`0=\mathsf{Pr}(X<1)<1/6=\mathsf{Pr}(X\le 1)`. If we pick :math:`1 < x < 2` then
   :math:`\mathsf{Pr}(X < x)=1/6=\mathsf{Pr}(X\le x)`. If :math:`x=2` then
   :math:`\mathsf{Pr}(X<2)=1/6<\mathsf{Pr}(X\le 2)=1/3`.

-----

Since the distribution and quantile functions are inverse, their graphs
are reflections of one another in a 45-degree line through the origin.
The distribution function is continuous from the right, hence the
location of the probability masses indicated by the circles.

Define

-  The **lower quantile** function
   :math:`q^-(p) := \sup\ \{x \mid F(x) < p \} = \inf\ \{ x \mid F(x) \ge p \}`,
   and
-  The **upper quantile** function
   :math:`q^+(p) := \sup\ \{x \mid F(x) \le p \} = \inf\ \{ x \mid F(x) > p \}`.

The lower and upper quantiles both satisfy the requirements to be a
quantile function. The lower quantile is left continuous. The upper
quantile is right continuous. When the quantile is not unique, it lies between the lower and upper values.

Value at Risk
---------------

When a quantile is used as a risk measure it is called **Value at Risk
(VaR)**: :math:`\mathsf{VaR}_p(X):=q^-(p) = \inf\ \{ x\mid F(x) \ge p\}`.

Thus :math:`l` is :math:`\mathsf{VaR}_p(X)` if it is the smallest loss
such that the probability :math:`X\le l` is :math:`\ge p`. This is
sometimes phrased: the smallest loss so that :math:`X\le l` with
confidence at least :math:`p`. *Smallest loss* allows for the case
:math:`F` is flat at :math:`p`. *Probability* :math:`\ge p` allows for
jumps in :math:`F`.

VaR has several advantages. It is simple to explain, can be estimated
robustly, and is always finite. It is widely used by regulators, rating
agencies, and companies in their internal risk management. Its principal
disadvantage is its failure to be subadditive.

.. _q not sa:

The Failure of VaR to be Subadditive
----------------------------------------

It is easy to create simple discrete examples where VaR fails to be subadditive. More interesting, 0.7-VaR applied to the sum of two independent exponential distributions is not subadditive, but 0.95-VaR is.

.. ipython:: python
   :okwarning:

   from aggregate import build, qd
   p = build('port NotSA '
             'agg A dfreq [1] sev 1 * expon '
             'agg B dfreq [1] sev 1 * expon')

   ans = p.var_dict(0.7)
   ans['sum'] = ans['A'] + ans['B']
   ans2 = p.var_dict(0.95)
   ans2['sum'] = ans2['A'] + ans2['B']

   pd.DataFrame([ans, ans2], index=pd.Index([0.7, 0.95], name='p'))

The function ``var_dict`` returns the VaR of each unit in ``p`` and the total. The total VaR is greater than the sum of the parts. Subadditivity requires total VaR be less than or equal to the sum of the parts.

.. _q tvar:

Tail VaR and Related Risk Measures
----------------------------------

Tail value at risk (TVaR) is the conditional average of the worst
:math:`1-p` outcomes. Let :math:`X` be a loss random variable and :math:`0 \le p<1`.
The :math:`p`-**Tail Value at Risk** is the conditional average of the
worst :math:`1-p` proportion of outcomes

.. math::

   \mathsf{TVaR}_p(X):=\dfrac{1}{1-p}\int_{p}^1 \mathsf{VaR}_s(X)\,ds=
   \dfrac{1}{1-p}\int_{p}^1 q^-(s)\,ds.

In particular :math:`\mathsf{TVaR}_0(X)=\mathsf{E}[X]`. When :math:`p=1`,
:math:`\mathsf{TVaR}_1(X)` is defined to be :math:`\sup(X)` if :math:`X` is unbounded.

TVaR is defined in terms of :math:`q^-`, that is, dual implicit events.
The actual sample space on which :math:`X` is defined is not used.
Recall, :math:`\mathsf{VaR}_p(X)` refers to the lower quantile
:math:`q^-(p)`.

TVaR is a well behaved function of :math:`p`. It is continuous,
differentiable almost everywhere, and equal to the integral of its
derivative (fundamental theorem of calculus). It takes every value
between :math:`\mathsf{E}[X]` and :math:`\sup X`. TVaR has a kink at
jumps in :math:`F` and is differentiable elsewhere.

Algorithm to Evaluate TVaR for a Discrete Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Algorithm Input:** :math:`X` is a discrete random variable, taking
:math:`N` equally likely values :math:`X_j\ge 0`,
:math:`j=0,\dots, N-1`. Probability level :math:`p`.

Follow these steps to determine :math:`\mathsf{TVaR}_p(X)`.

**Algorithm Steps**

(1) **Sort** outcomes into ascending order
    :math:`X_0 < \dots < X_{N-1}`.
(2) **Find** :math:`n` so that :math:`n \le pN < (n+1)`.
(3) **If** :math:`n+1=N` **then** :math:`\mathsf{TVaR}_p(X) := X_{N-1}`
    is the largest observation, exit;
(4) **Else** :math:`n < N-1` and continue.
(5) **Compute** :math:`T_1 := X_{n+1} + \cdots + X_{N-1}`.
(6) **Compute** :math:`T_2 := ((n+1)-pN)x_n`.
(7) **Compute** :math:`\mathsf{TVaR}_p(X) := (1-p)^{-1}(T_1+T_2)/N`.

These steps compute the average of the largest :math:`N(1-p)`
observations. Step (6) adds a pro-rata portion of the
:math:`\lfloor N(1-p)\rfloor` largest observation when :math:`N(1-p)` is
not an integer. For instance, if :math:`N=71` and :math:`p=0.95`, then
:math:`Np=67.45` and :math:`n=67`, giving
:math:`\mathsf{TVaR}_p = 20(0.55x_{67}+x_{68}+x_{69}+x_{70})/71`.

**Example.** Let :math:`X` be defined on
a sample space with ten equally likely events and outcomes
:math:`0,1,1,1,2,3, 4,8, 12, 25`. Compute :math:`\mathsf{TVaR}_p(X)` for
all :math:`p`. Is it a piecewise linear function?

**Solution.** For :math:`p \ge 0.9`, :math:`q(p)=25` and
:math:`\mathsf{TVaR}_p(X)=25`. For :math:`0.8 \ge p < 0.9`

.. math::

   (1-p)\mathsf{TVaR}_p(X) &= \int_p^1 q^-(s)ds = \int_p^{0.9}q^-(s)ds+ \int_{0.9}^1q^-(s)ds \\
   &= (0.9-p)\times 12 + (1-0.9)\times \mathsf{TVaR}_{0.9}(X),

for :math:`0.7 \ge p < 0.8`

.. math::

   (1-p)\mathsf{TVaR}_p(X) = (0.8-p)\times 8 + (1-0.8)\times \mathsf{TVaR}_{0.8}(X),

and so forth. The TVaR function is shown below.
TVaR is not piecewise linear. For
example, for :math:`0.8\le p<0.9`,
:math:`\mathsf{TVaR}_p(X)=(12(0.9-p) + 2.5)/(1-p)`.

The default aggregate TVaR function ignores this slight non-linearity and just interpolates. To get a more exact answer use ``kind='tail'``.

.. ipython:: python

   p = 0.73
   print(a.tvar(0.7), a.tvar(p), a.tvar(p, 'tail'),
      ((0.8-p) * 8 + 0.2 *a.tvar(0.8)) / (1-p))


CTE, and WCE: Alternatives to TVaR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two other risk measures (confusingly) similar to TVaR.

1. Tail value at risk (TVaR) is the conditional average of the worst
   :math:`1-p` outcomes.
2. **Conditional tail expectation** (CTE) refers to the conditional
   expectation of :math:`X` over :math:`X\ge \mathsf{VaR}_p(X)`.
3. **Worst conditional expectation** (WCE) refers to the greatest expected
   value of :math:`X` conditional on a set of probability :math:`>1-p`.

The formal definitions of CTE and WCE are as follows. Let :math:`X` be a loss random variable and :math:`0 \le p<1`.

- :math:`\mathsf{CTE}_p(X) := \mathsf{E}[X \mid X \ge \mathsf{VaR}_p(X)]` **(lower) conditional tail expectation** (TCE).

- The upper CTE equals :math:`\mathsf{E}[X \mid X \ge q^+(p)]`.

- :math:`\mathsf{WCE}_p(X) := \sup\ \{ \mathsf{E}[X \mid A] \mid \mathsf{Pr}(A) > 1-p \}` is the **worst conditional expectation**.

Like TVaR, CTE is defined in terms of quantiles, and the sample space on
which :math:`X` is defined is not used. In contrast, WCE works with the
original sample space and relies on its events. Some actuarial papers
refer to CTE as tail value at risk, e.g., :cite:t:`Bodoff2007`.

For continuous random variables TVaR, CTE, and WCE are all equal, and
they are easy to compute. The distinctions between them arise for
discrete and mixed variables when :math:`p` coincides with a mass point.

Expected Policyholder Deficit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **EPD ratio** is defined as the ratio of the EPD to expected losses.
It gives the proportion of losses that are unpaid when :math:`X` is
supported by assets :math:`a`.

**Example.**

We can use the EPD to define a tail risk measure that is
analogous to VaR and TVaR. Define the **EPD risk measure**
:math:`\mathsf{E}PD_s(X)` to be the amount of assets resulting in an EPD
ratio of :math:`0 < s < 1`, i.e., solving

.. math::

   \mathsf{E}[(X-\mathsf{E}PD_p(X))^+] = s\mathsf{E}[X].

The EPD risk measure is a stricter standard for smaller
:math:`s`. It accounts for the degree of default relative to promised
payments, making it attractive to regulators. It is used to set risk
based capital standards in :cite:t:`Butsic1994` and as a capital standard in
:cite:t:`Myers2001`.

EPD is available in aggregate as the ``epd`` column in ``density_df``.

