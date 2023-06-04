.. Originally pirc_O_bodoff.md in PIRC/Python.

Bodoff’s Percentile Layer Capital Method
=========================================

**Objectives:** Compare Bodoff with the natural allocation and show how to compute both in ``aggregate``. 

**Audience:** Those interested in current allocation methods and CAS Exam 9 candidates.

**Prerequisites:** Background on allocation and Bodoff's paper. 


**Contents:**

* :ref:`bodoff hr`
* :ref:`bodoff intro`
* :ref:`bodoff an`
* :ref:`bodoff 3`
* :ref:`bodoff pla`
* :ref:`bodoff te`
* :ref:`bodoff te1`
* :ref:`bodoff ex123`
* :ref:`bodoff ex4`
* :ref:`bodoff summary`
* :ref:`bodoff cas`


.. _bodoff hr:

Helpful References
--------------------

* :cite:t:`Bodoff2007`
* :cite:t:`PIR`

.. _bodoff intro:

Introduction
--------------

The abstract to :cite:t:`Bodoff2007`, Capital Allocation by Percentile Layer reads:

  This paper describes a new approach to capital allocation; the catalyst
  for this new approach is a new formulation of the meaning of holding
  Value at Risk (VaR) capital. This new formulation expresses the firm’s
  total capital as the sum of many granular pieces of capital, or
  “percentile layers of capital.” As a result, one must allocate capital
  separately on each layer and perform the capital allocation across all
  layers. The resulting capital allocation procedure, “capital allocation
  by percentile layer,” exhibits several salient features. First, **it
  allocates capital to all losses, rather than allocating capital only to
  extreme losses in the tail of the distribution**. Second, despite
  allocating capital to this broad range of loss events, **the proposed
  procedure does not allocate in proportion to average loss; rather, it
  allocates disproportionate capital to severe losses**. Third, **it
  allocates capital by relying neither upon esoteric parameters nor upon
  elusive risk preferences**. Ultimately, on the practical plane,
  **capital allocation by percentile layer produces allocations that are
  different from many other methods**. Concomitantly, on the theoretical
  plane, capital allocation by percentile layer leads to new continuous
  formulas for risk load and utility.

Bodoff’s paper is an important contribution to capital allocation and
actuarial science. Its key insight is that layers of capital respond to a
range of loss events and not just tail events and so it is not appropriate to focus
solely on default states when allocating capital. Bodoff takes capital to
mean total claims paying ability, comprised of equity and premium. Bodoff
allocates capital by considering loss outcomes and assumes that expected
loss, margin, premium, and equity all have the same allocation **within each
layer**.

Less favorably, Bodoff blurs the distinction between events and outcomes. He
allocates to identifiable **events** (wind-only loss, etc.) rather than
to **outcomes**. In examples, outcome amounts distinguish events. In the Lee
diagram, events are on the horizontal axis and outcomes on the vertical
axis.

.. _bodoff an:

Assumptions and Notation
--------------------------

The examples model two independent units :math:`X_1` and :math:`X_2`, usually ``wind``
and ``quake``, with total :math:`X = X_1 + X_2`.
:math:`F` and :math:`S` represent the distribution and survival function
of :math:`X` and :math:`q` its lower quantile function. The capital
(asset) requirement set equal to the (lower) :math:`a:=p=0.99`-VaR capital

.. _bodoff 3:

Three Possible Allocation Methods
----------------------------------

Consider three allocations:

1. **Conditional VaR**: ``coVaR``, method allocates using

   .. math:: a=\mathsf E[X\mid X=a] = \mathsf E[X_1\mid X=a] + \mathsf E[X_2\mid X=a]

2. **Alternative conditional VaR**: ``alt coVaR``, method allocates
   using

   .. math::

      a = a\,\mathsf E\left[\frac{X_1}{X}\mid X\ge a  \right] +
      a\,\mathsf E\left[\frac{X_2}{X}\mid X\ge a  \right]

3. **Naive conditional TVaR**: ``naive coTVaR``, method allocates
   :math:`a` proportional to :math:`\mathsf E[X_1\mid X \ge a]` and
   :math:`\mathsf E[X_2\mid X \ge a]`

Bodoff’s principal criticism of these methods is that they all ignore
the possibility of outcomes :math:`<a`.

* ``coVaR`` allocates based proportion of losses by unit on the events
  :math:`\{X=a\}` of exact size :math:`a`. It ignores other events near
  :math:`X=a` and all events :math:`X<a`, which seems unreasonable.
  The allocation is not numerically stable: in simulation output
  :math:`\{X=a\}` is often only a single event.

* ``alt coVaR`` allocates based proportion of losses by unit on the
  events :math:`\{X \ge a\}`. It still ignores all events :math:`<a`. It relies on the relationship

  .. math::
      a &= a\,\left(\mathsf E\left[\frac{X_1}{X}\mid X\ge a\right] + a\mathsf E\left[\frac{X_2}{X}\mid X\ge a\right]\right) \\
      &= a\,\alpha_1(a) + a\,\alpha_2(a)

* ``naive coTVaR`` resorts to a pro rata kludge because
  :math:`\mathsf E[X\mid X \ge x]\ge x` and is usually :math:`>x`.
  Pro rata adjustments signal the lack of a rigorous rationale and
  should be avoided. Note: what Bodoff calls TVaR is usually known as CTE.

* **Alternative conditional TVaR**: the ``coTVaR`` method (not considered
  by Bodoff but introduced by Mango, Venter, Kreps, Major) solves
  :math:`a=\mathsf{TVaR}(p^*)` for :math:`p^*\le p`
  (we shall see below we really need to use expected shortfall, not TVaR).
  Then determine :math:`a^*=q(p^*)`, the :math:`p^*`-VaR  and allocate
  using :math:`a=\mathsf E[X\mid X\ge a^*] =\mathsf E[X_1\mid X\ge a^*] + \mathsf E[X_2\mid X\ge a^*]`.


In addition, all methods can be criticized as actuarial allocation exercises
without an economic motivation. They do not consider premium: additional
assumptions needed to derive a premium from an asset or capital allocation,
such as a target return on allocated capital. They just provide an allocation
of premium plus capital, i.e., assets, and not a split between the two.


.. _bodoff pla:

Percentile Layer Allocation: Definition
---------------------------------------

Bodoff introduces the **percentile layer of capital**, ``plc``, allocation
method to address the criticism that methods 1-4 all ignore events causing
losses below the level of capital, whereas capital is certainly used to pay
such losses. It allocates capital in the same proportion as losses for each
layer.

In a one-dollar, all-or-nothing cover that attaches with probability
:math:`s=1-p` at :math:`x=q(p)` (:math:`=p`-:math:`\mathsf{VaR}`),
under equal priority unit
:math:`i` receives a proportion
:math:`\alpha_i(x):=\mathsf E\left[\dfrac{X_i}{X}\mid X > x\right]` of
assets, conditional on a loss.
Therefore, unconditional expected loss recoveries equal
:math:`\alpha_i(x)S(x)`, part of total layer losses :math:`S(x)`. Allocating
each layer of capital between 0 and :math:`a` in the same way gives
the **percentile layer of capital** ``plc`` allocation:

.. math:: a_i:=\int_0^a \alpha_i(x)\,dx = \int_0^a \mathsf E\left[ \frac{X_i}{X}\mid X >x \right]\,dx

By construction, :math:`\sum_i a_i=a`. The ``plc`` allocation can be
understood better by decomposing

.. math::
      a &= \int_0^a 1\, dx \\
      &= \int_0^a \alpha_1(x) + \alpha_2(x)\, dx \\
      &= \int_0^a \alpha_1(x)S(x) + \alpha_1(x)F(x)\, dx + \int_0^a \alpha_2(x)S(x) + \alpha_2(x)F(x)\, dx \\
      &= \left(\mathsf E[X_1(a)] + \int_0^a \alpha_1(x)F(x)\, dx\right) + \left(\mathsf E[X_2(a)] + \int_0^a \alpha_2(x)F(x)\, dx\right)

It splits unfunded assets (assets in excess of expected
losses) in the same proportion as losses in each asset layer, using
:math:`\alpha_i(x)`. ``plc`` says **nothing** about how to split the allocated
unfunded capital :math:`\int_0^a \alpha_2(x)F(x)\, dx` into margin
and equity. This is not surprising, since there are no pricing assumptions.
The natural allocation introduces a pricing distortion to compute an
allocation of premium, and hence margin.

There are six allocations considered by Bodoff, with the following
allocations of assets to unit 1.

#.  ``pct EX``:  :math:`\mathsf E[X_1] / \mathsf E[X]`
#. ``coVaR``:   :math:`\mathsf E[X_1\mid X=a]`
#. ``adj VaR``: :math:`a\,\mathsf E\left[\dfrac{X_1}{X}\mid X\ge a \right]`
#. ``naive coTVaR``: :math:`a\,\dfrac{\mathsf E[X_1\mid X \ge a]}{\mathsf E[X\mid X \ge a]}`
#.  ``coTVaR``:  :math:`\mathsf E[X_1\mid X > a^*]`, where :math:`a=\mathsf{TVaR}(p^*)`
#.  ``plc``:  :math:`\displaystyle \int_0^a \alpha_i(x)\,dx`, where   :math:`\alpha_i(x):=\mathsf E\left[\dfrac{X_i}{X}\mid X > x\right]`

.. _bodoff te:

Thought Experiments
---------------------

Bodoff introduces four thought experiments:

1. Wind and quake, wind losses 0 or 99,
   quake 0 or 100, 0.2 probability of a wind loss and 0.01 probability
   of a quake loss.

2. Wind and quake, wind 0 or 50, quake 0 or
   100, same probabilities.

3. Wind and quake, wind 0 or 5, quake 0 or
   100, same probabilities.

4. Bernoulli / exponential compound distribution (see :ref:`Bodoff Example 4`.)

The units are independent. The next block of code sets up and validates :class:`Portfolio`
objects for each. The Bodoff portfolios are part of the base library and can be extracted with
``build.qlist``.

.. ipython:: python
   :okwarning:

   import pandas as pd
   from collections import OrderedDict
   from aggregate import build, qd
   from aggregate.extensions import bodoff_exhibit
   bodoff = list(build.qlist('.*Bodoff').program)
   ports = OrderedDict()
   for s in bodoff:
       port = build(s)
       ports[port.name] = port
   for port in ports.values():
       if port.name != 'Bodoff:4':
           port.update(bs=1, log2=8, remove_fuzz=True, padding=1)
       else:
           port.update(bs=1/8, log2=16, remove_fuzz=True, padding=2)
       port.density_df = port.density_df.apply(lambda x: np.round(x, 14))
       qd(port)
       print(port.name)
       print('='*80 + '\n')

.. _bodoff te1:

Thought Experiment Number 1
----------------------------

There are four possible events :math:`\omega`, leading to the loss
outcomes :math:`X(\omega)` laid out next.

.. math::
    \small
    \begin{matrix}
    \begin{array}{lrrrrrr}\hline
      \text{Event,}\ \omega & X_1  & X_2  & X     & \Pr(\omega) & F & S \\ \hline
       \text{No loss}      &  0   & 0    &  0   &  0.76        & 0.76  &  0.24  \\
       \text{Wind   }      &  99  & 0    &  99  &  0.19        & 0.95  &  0.05  \\
       \text{Quake  }      &  0   & 100  &  100 &  0.04        & 0.99  &  0.01  \\
       \text{Both   }      &  99  & 100  &  199 &  0.01        & 1.00  &  0.00  \\ \hline
      \end{array}
    \end{matrix}


Compute the allocation using all the methods. In the next block, ``EX`` shows
expected unlimited loss by unit. ``sa VaR`` and ``sa TVaR`` show stand-alone
0.99 VaR and TVaR. The remaining rows display results for the methods
just described. The apparent issue with the ``coTVaR`` method is caused by
the probability mass at 100. A ``co ES`` allocation would re-scale the
``coTVaR`` allocation shown.

.. ipython:: python
   :okwarning:

   port = ports['Bodoff:1']
   reg_p = 0.99
   a = port.q(reg_p, 'lower')
   print(f'VaR assets = {a}')
   basic = bodoff_exhibit(port, reg_p)
   qd(basic, col_space=10)


.. ipython python
   :okwarning:

   pstar = port.tvar_threshold(reg_p, 'lower')
   unique_values = ', '.join([f'{x:.6g}' for x in np.unique(np.round(port.density_df.exgta_total, 5))[1:]])
   pstar, unique_values, port.tvar(0.99)


Graphs of the survival and allocation functions for Bodoff Example 1. Top row:
survival functions, bottom row: :math:`\alpha_i(x)` allocation functions. Left side
shows full range of :math:`0\le x\le 200` and right side highlights the functions
around the loss points, :math:`96\le x \le 103`.

.. ipython:: python
   :okwarning:

   fig, axs = plt.subplots(2, 2, figsize=(2 * 3.5, 2 * 2.45), constrained_layout=True)
   ax0, ax1, ax2, ax3 = axs.flat
   df = port.density_df
   for ax in axs.flat[:2]:
       (1 - df.query('(S>0 or p_total>0) and loss<=210').filter(regex='p_').cumsum()).\
           plot(drawstyle="steps-post", ax=ax, lw=1)
       ax.lines[1].set(lw=2, alpha=.5)
       ax.lines[2].set(lw=3, alpha=.5)
       ax.grid(lw=.25)
       ax.legend(loc='upper right')
   ax0.set(ylim=(-0.025, .25), xlim=(-.5, 210), xlabel='Loss', ylabel='Survival function');
   ax1.set(ylim=(-0.025, .3), xlim=[96,103], xlabel='Loss (zoom)', ylabel='Survival function');
   for ax in axs.flat[2:]:
       df.query('(S>0) and loss<=210').filter(regex='exi_xgta_X').plot(drawstyle="steps-post", lw=1, ax=ax)
       ax.lines[1].set(lw=2, alpha=.5)
       ax.grid(lw=.25)
       ax.legend(loc='upper right')
   ax2.set(ylim=(-0.025, 1.025), xlabel='Loss', ylabel='$E[X_i/X | X]$');
   @savefig bodoff_1.png scale=20
   ax3.set(ylim=(-0.025, 1.025), xlim=(96,103), xlabel='Loss (zoom)', ylabel='$E[X_i/X | X]$');

Expected Shortfall (usually called TVaR) differs from Bodoff's Tail Value at
Risk (generally called CTE) for a discrete distribution.  TVaR/CTE is a jump
function. ES is a continuous, increasing function taking all values between
the mean and maximum value of :math:`X`. The graph illustrates the functions for
Bodoff Example 1.

.. ipython:: python
   :okwarning:

   fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.45), constrained_layout=True)
   ps =np.linspace(0, 1, 101)
   tp = port.tvar(ps)
   ax.plot(ps, tp, lw=1, label='ES');
   ax.plot(df.F, port.density_df.exgta_total, lw=1, label='TVaR', drawstyle='steps-post');
   ax.plot([0, .76], [port.ex/.24, port.ex/.24, ], c='C1', lw=1, label=None);
   ax.grid();
   ax.legend();
   @savefig bodoff_2.png scale=20
   ax.set(ylim=[-5, 205], xlabel='p', ylabel='ES or TVaR/CTE');

.. _bodoff ex123:

Bodoff Examples 1-3
-----------------------

Example 2 illustrates that ``plc`` can produce an answer that is different
from expected losses. Example 3 it illustrates fungibility of pooled capital,
with losses from :math:`X_1` covered by the total premium. ``coTVaR`` suffers the
same issues in Examples 2 and 3 as it does in Example 1.

.. ipython:: python
   :okwarning:

   basic1 = bodoff_exhibit(ports['Bodoff:1'], reg_p)
   basic2 = bodoff_exhibit(ports['Bodoff:2'], reg_p)
   basic3 = bodoff_exhibit(ports['Bodoff:3'], reg_p)
   basic_all = pd.concat((basic1, basic2, basic3), axis=1,
      keys=[f'Ex {i}' for i in range(1,4)])
   qd(basic_all, col_space=7)

.. _bodoff ex4:

Bodoff Example 4
--------------------

The next table recreates the exhibit in Section 9.1 of Bodoff's paper. There are three units labelled ``a``, ``b``, and ``c``.
It shows the percent allocation of capital to each unit across different methods.
Breakeven percentile equals the percentile equal to expected losses. Bodoff's
calculation uses 10,000 simulations. The table shown here uses FFTs to obtain a close-to exact
answer. The exponential distribution is borderline thick tailed, and so is quite hard
to work with for both simulation methods and FFT methods.


.. ipython:: python
   :okwarning:

   p4 = ports['Bodoff:4']
   df91 = pd.DataFrame(columns=list('abc'), dtype=float)
   tv = p4.var_dict(.99, 'tvar')
   df91.loc['sa TVaR 0.99'] = np.array(list(tv.values())[:-1]) / sum(list(tv.values())[:-1])
   pbe = float(p4.cdf(p4.ex))
   for p in [.99, .95, .9, pbe]:
       tv = p4.cotvar(p)
       df91.loc[f'naive TVaR {p:.3g}'] = tv[:-1] / tv[-1]
   v = ((p4.density_df.filter(regex='exi_xgta_[abc]').
                       shift(1).cumsum() * p4.bs).loc[p4.q(.99)]).values
   df91.loc['plc'] = v / v.sum()
   df91.index.name = 'line'
   qd(df91, col_space=10, float_format=lambda x: f'{x:.1%}')


Pricing for Bodoff Example 4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bodoff Example 4 is based on a three unit portfolio. Each unit has a Bernoulli
0/1 frequency and exponential severity:

   * Unit ``a`` has a 0.25 probability of a claim and 4 severity
   * Unit ``b`` has a 0.05 probability of a claim and 20 severity
   * Unit ``c`` has a 0.01 probability of a claim and 100 severity

All units have unlimited expectation 1.0

Bodoff does not consider pricing per se. His allocation can be
considered as :math:`P_i+Q_i`, with no opinion on the split between
margin and equity. Making additional assumptions we can compare the ``plc`` capital
allocation with other methods. Assume total roe = 0.1 at 0.99-VaR capital standard.
Set up the target return, premium, and regulatory capital threshold (99% VaR):

.. ipython:: python
   :okwarning:

   roe = 0.1
   reg_p = 0.99
   v = 1 / (1 + roe)
   d = 1 - v
   port = ports['Bodoff:4']
   a = port.q(reg_p)
   el = port.density_df.at[a, 'lev_total']
   premium = v * el + d * a
   q = a - premium
   margin = premium - el
   roe, a, el, port.ex, premium, el / premium, q, margin / q

Calibrate pricing distortions to required return.

.. ipython:: python
   :okwarning:

   port.calibrate_distortions(ROEs=[roe], Ps=[reg_p], strict='ordered');
   qd(port.distortion_df)

Allocate premium plus equity to each unit across different pricing methods. All methods
except percentile layer capital calibrated to the same total premium and capital level.
Distortions that price tail loss will allocate the most to unit ``c``, the most volatile.
More bowed distortions will allocate most to ``a``. The three units have the same expected loss
(last row). ``covar`` is covariance method; ``coVaR`` is conditional VaR. ``agg`` corresponds to the PIR approach and ``bod`` to Bodoff’s
methods. Only additive methods are shown. ``method`` ordered by allocation
to unit ``a`` the least skewed; ``c`` is the most skewed.

.. ipython:: python
   :okwarning:

   ad_ans = port.analyze_distortions(p=reg_p, kind='lower')
   basic = bodoff_exhibit(port, reg_p)
   qd(basic, col_space=10)
   ans = pd.concat((ad_ans.comp_df.xs('P', 0, 1) + ad_ans.comp_df.xs('Q', 0, 1),
                    basic.rename(columns=dict(X='total')).iloc[3:]), keys=('agg', 'bod'))
   if port.name[-1] in list('123'):
       ans = ans.sort_values('X1')
       bit = ans.query(' abs(total - @a) < 1e-3 and abs(X1 + X2 - total) < 1e-3 ').dropna()
   if port.name[-1] not in list('123'):
       ans = ans.sort_values('a')
       bit = ans.query(' abs(total - @a) < 1e-2 and abs(a + b + c - total) < 1e-2 ')
   bit.index.names =['approach', 'method']
   qd(bit, col_space=10)

Premium for PIR and Bodoff methods, sorted by premium for ``a``.
All methods produce the same total premium by calibration.
Very considerable differences are evident across the methods.

.. ipython:: python
   :okwarning:

   basic.loc['EXa'] = \
   port.density_df.filter(regex='exa_[abct]').loc[a].rename(index=lambda x: x.replace('exa_', ''))
   premium_df = basic.drop(index=['EX', 'sa TVaR', 'coTVaR'])
   premium_df = premium_df.loc['EXa'] * v + d * premium_df
   ans = pd.concat((ad_ans.comp_df.xs('P', 0, 1), premium_df),
      keys=('agg', 'bod')).sort_values('a')
   bit = ans.query(' abs(total - @premium) < 1e-2 and abs(a + b + c - total) < 1e-2 ')
   bit.index.names =['approach', 'method']
   qd(bit, col_space=10, sparsify=False)

Corresponding loss ratios (remember, these are cat lines).

.. ipython:: python
   :okwarning:

   bit_lr = premium_df.loc['EXa'] / bit
   qd(bit_lr, col_space=10, sparsify=False,
      float_format=lambda x: f'{x:.1%}')

.. _bodoff summary:

Bodoff Summary
-----------------

Bodoff's methods allocate all capital like loss and do not distinguish expected loss,
margin and equity. It does not get to a price. It is event-centric, allocating to **events**,
but really allocating to **peril=lines**. Premium is not mentioned until Section 7 (of 10).
Then, it uses the basic CCoC formula :math:`P=vL + da` (eq. 8.2).

.. _bodoff cas:

CAS Exam Question: Spring 2018 Question 15
---------------------------------------------

An insurer has exposure to two independent perils, wind and earthquake:

-  Wind has a 15% chance of a $5 million loss, and an 85% chance of no
   loss.
-  Earthquake has a 1 % chance of a $15 million loss, and a 99% chance
   of no loss.

Using the capital allocation by percentile layer methodology with a
99.5% VaR capital requirement, determine how much capital should be
allocated to each peril.

**Solution.**

The last row gives the percentile layer capital.

.. ipython:: python
   :okwarning:

   cas15 = build('''
   port CASq15 
       agg X1 1 claim dsev [0,  5] [0.85, 0.15] fixed
       agg X2 1 claim dsev [0, 15] [0.99, 0.01] fixed
   ''')
   qd(cas15)
   # cas15.update(bs=1, log2=8, remove_fuzz=True, padding=1)
   cas15.density_df = cas15.density_df.apply(lambda x: np.round(x, 10))
   basic = bodoff_exhibit(cas15, reg_p=.995)
   qd(basic, col_space=10)
   df = cas15.density_df.query('S > 0 or p_total > 0')

The calculation of ``plc`` as the integral of :math:`\alpha` for unit 1 is simply:

.. ipython:: python
   :okwarning:

   df.exi_xgta_X1.shift(1, fill_value=0).cumsum().loc[15] * cas15.bs



