Denuit (2019)
-----------------

:cite:`Denuit2019` or
:cite:p:`Denuit2019` or
:cite:t:`Denuit2019`

Denuit, Michel: Size-biased transform and conditional mean risk sharing, with application to p2p insurance and tontines

Poisson/Discrete Example (6.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python
    :okwarning:

    from aggregate import build, qd

    p = build('''
    port Denuit6.1
        agg P1 0.08 claims dsev [1 2 3 4] [.1  .2  .4 .3] poisson
        agg P2 0.08 claims dsev [1 2 3 4] [.15 .25 .3 .3] poisson
        agg P3 0.10 claims dsev [1 2 3 4] [.1  .2  .4 .3] poisson
        agg P4 0.10 claims dsev [1 2 3 4] [.15 .25 .3 .3] poisson
    ''', bs=1, log2=10)
    qd(p)

Computation of :math:`\mathsf{E}[X_i\mid X=x]` and :math:`\mathsf{E}[X_i\mid X=x]/x` as a function of :math:`x`.
The first function, called :math:`\kappa_i(x)` in PIR, is computed automatically by the :class:`portfolio` class as
``exeqa_line`` (expectation given :math:`X` equals :math:`x`). The original figure is shown below.

.. ipython:: python

    bit = p.density_df.query('p_total > 0').iloc[1:]
    rat = bit.filter(regex='exeqa_P').apply(
        lambda x: x / bit.loss.to_numpy(), axis=0)
    ax = rat.plot.bar(ylim=[-0.05,1.05], stacked=True, figsize=(3.5, 2.45))
    ax.set(xlim=[-0.5, 15.5], ylim=[0,1])
    @savefig denuit_19.png scale=20
    ax.legend().set(visible=False);


.. image:: img/denuit_19_figure_1.png
  :width: 800
  :alt: Original paper table.



All the values are available as a table. These are consistent with numbers mentioned in the text.

.. ipython:: python

    from pandas import option_context
    b = bit.filter(regex='exeqa_P').apply(
        lambda x: x / bit.loss.to_numpy(), axis=0)
    b.index = b.index.astype(int)
    b.index.name = 'a'
    qd(b)

Proportion of expected loss by unit.

.. ipython:: python

    bb = p.describe.xs('Agg', axis=0, level=1)[['E[X]']]
    qd(bb / bb.iloc[-1,0])




Mortality Example and Figure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1. ASTIN 2022 Mortality Credits with Large Survivor Funds (D, Hieber, Roberts) [Fig 5 done]

Description...


.. ipython:: python
    :okwarning:

    import matplotlib.pyplot as plt; import pandas as pd
    wl = 0.6; wh = 1 - wl; ql = .1; qh = .2; al = 1; ah = 3
    ports = {}
    for n in (10, 20, 50, 100):
        ports[n] = build(f'port DR.4.3 '
              f'agg Low.q  {wl * n * ql} claims dsev [{al}] binomial {ql}'
              f'agg High.q {wh * n * qh} claims dsev [{ah}] binomial {qh}'
             , bs=1, log2=8)

    audit = pd.concat([i.describe for i in ports.values()], keys=ports.keys(), names=['n', 'unit', 'X'])
    qd(audit.xs('Agg', axis=0, level=2)['E[X]'].unstack(1))
    fig, axs = plt.subplots(2, 2, figsize=(2 * 3.5, 2 * 3.5), constrained_layout=True, squeeze=True)
    for ax, (n, port), mx, t in zip(axs.flat, ports.items(), [20, 25, 40, 60], [2, 5, 10, 10]):
        lm = [-1, mx]
        # lm = [-1, port.q(1)+1]
        port.density_df.query('p_total > 0').filter(regex='exeqa_[LHt]').plot(ax=ax, xlim=lm, ylim=lm)
        ax.set_xticks(range(0, mx, t))
        ax.set_yticks(range(0, mx, t))
        ax.grid(lw=.25, c='w')
        ax.set(title=f'{n} risks')
    @savefig denuit_45.png scale=20
    fig.suptitle('Denuit Figure 4.5')

