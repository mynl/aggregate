.. _2_x_re_pricing:

Reinsurance Pricing
==========================

**How to model 2 reinstatements??**

Reinsurance Pricing Applications
--------------------------------

*  Excess of loss exposure rating
*  Creation of a severity curve from a limit profile

Basic Reinsurance on Discrete Examples
--------------------------------------

This section presents a simple example where it is easy to see what is going on. The remaining sections deal with more realistic examples. It is based on the dice roll of dice rolls: throw a dice, then throw that many die) example from :doc:`2_x_student`.

.. ipython:: python
    :okwarning:

    from aggregate import build

    # dice of dice, gross for reference
    dice2 = build('agg DiceOfDice dfreq [1 2 3 4 5 6] dsev [1 2 3 4 5 6] ')
    dice2.plot()
    @savefig re_dice.png
    dice2

First, apply occurrence reinsurance.

.. ipython:: python
    :okwarning:

    dice_occ = build('agg DiceOcc dfreq [1 2 3 4 5 6] dsev [1 2 3 4 5 6] '
                     'occurrence net of 2 x 4')
    dice_occ.plot()
    @savefig re_dice_occ.png
    dice_occ

Then an aggregate cover:

.. ipython:: python
    :okwarning:

    dice_ag = build('agg Dice2 dfreq [1 2 3 4 5 6] dsev [1 2 3 4 5 6] '
                    'aggregate net of 12 x 24')
    dice_ag.plot()
    @savefig re_dice_ag.png
    dice_ag

Finally, both at once.

.. ipython:: python
    :okwarning:

    dice_re = build('agg DiceRe dfreq [1:6] dsev [1:6] '
                    'occurrence net of 2 x 4 aggregate net of 6 xs 18')
    dice_re.plot()
    @savefig re_dice_both.png
    dice_re


Sum of Uniforms Is Triangular, With Reinsurance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python
    :okwarning:


    # gross, net of occurrence, and net
    bg = build('agg GROSS dfreq [2] dsev [1:10]')
    bno = build('agg NET_OCC dfreq [2] dsev [1:10] '
                'occurrence net of 3 x 7')
    bn = build('agg NET dfreq [2] dsev [1:10] '
               'occurrence net of 3 x 7 '
               'aggregate net of 3 x 10')
    for b in [bg, bno, bn]:
        print(b)


Remaining code?

::

    bg.plot()
    @savefig re_b.png

    bno.plot()
    @savefig re_bno.png

    bn.plot()
    @savefig re_bn.png

    10 * 12


    # post process to rationalize the graphs
    ml = bg.figure.axes[0].xaxis.get_major_locator()
    my = bg.figure.axes[0].yaxis.get_major_locator()
    yl = bg.figure.axes[2].get_ylim()
    for b in [bg, bno, bn]:
        for ax in b.figure.axes[:2]:
            ax.set(xlim=[0, 22])
            if b is not bg:
                ax.xaxis.set_major_locator(ml)
        if b is not bg:
            b.figure.axes[2].yaxis.set_major_locator(my)
            b.figure.axes[2].set(ylim=yl)

.. _realistic examples:

Realistic Examples
----------------------

Here are models of the four examples from `2_agg_class_reinsurance_clause`_. TODO how does that show up?


    agg Trucking.Gross 5000 loss 1000 xs 0 sev lognorm 50 cv 1.75 poisson

    agg Trucking.Net   5000 loss 1000 xs 0 sev lognorm 50 cv 1.75 occurrence net of 750 xs 250 poisson

    agg Trucking.Ceded 5000 loss 1000 xs 0 sev lognorm 50 cv 1.75 occurrence ceded to 750 xs 250 poisson

    agg Trucking.Retention 5000 loss 1000 xs 0 \
    sev lognorm 50 cv 1.75 \
    occurrence net of 50% so 250 xs 250 and 500 xs 500 poisson \
    aggregate net of 250 po 1000 xs 4000 and 5000 xs 5000

    agg WorkComp.InsChrg 15000 loss 500 xs 0 sev lognorm 50 cv 1.75 poisson aggregate ceded to 50% so 2000 xs 15000


.. ipython:: python
    :okwarning:

    from aggregate import build, qd
    import matplotlib.pyplot as plt

    ans = build('''
    agg Trucking.Gross 5000 loss 1000 xs 0 sev lognorm 50 cv 1.75 poisson
    agg Trucking.Net   5000 loss 1000 xs 0 sev lognorm 50 cv 1.75 occurrence net of 750 xs 250 poisson
    agg Trucking.Ceded 5000 loss 1000 xs 0 sev lognorm 50 cv 1.75 occurrence ceded to 750 xs 250 poisson
    agg Trucking.Retention 5000 loss 1000 xs 0 \
    sev lognorm 50 cv 1.75 \
    occurrence net of 50% so 250 xs 250 and 500 xs 500 poisson \
    aggregate net of 250 po 1000 xs 4000 and 5000 xs 5000
    agg WorkComp.InsChrg 15000 loss 500 xs 0 sev lognorm 50 cv 1.75 poisson aggregate ceded to 50% so 2000 xs 15000
    ''', approximation='exact')

    for a in ans:
        qd(a.name)
        qd(a.object.describe)
        print('-'*100 + '\n')

These distributions have a high claim count, hence specify ``approximation='exact'``.

.. ipython:: python
    :okwarning:

    fig, axs = plt.subplots(1, 3, figsize=(3 * 3.5, 2.45), constrained_layout=True, squeeze=True)
    ax0, ax1, ax2 = axs.flat

    sc = 'linear'
    var = 'F'

    tg = ans[0].object
    tn = ans[1].object
    tc = ans[2].object
    tr = ans[3].object
    wc = ans[4].object

    tg.density_df[var].plot(ax=ax0, label=tg.name);
    tn.density_df[var].plot(ax=ax0, label=tn.name);
    tc.density_df[var].plot(ax=ax0, label=tc.name);
    ax0.legend() # loc='lower right');
    mx = tg.q(0.9995)
    xl = [-mx/50, mx]
    ax0.set(xlim=xl, yscale=sc, title='Trucking: gross, ceded, net');

    tr.density_df[var].plot(ax=ax1, label=tr.name);
    tg.density_df[var].plot(ax=ax1, label=tg.name);
    ax1.legend();
    ax1.set(xlim=xl, yscale=sc, title='Trucking: gross and retained');

    wc.density_df[var].plot(ax=ax2,label=wc.name);
    ax2.legend();
    xl2 = [-50, 1050]
    @savefig re_pricing_comps.png
    ax2.set(xlim=xl2, yscale=sc, ylim=ax0.get_ylim(), title='WC insurance charge distribution');


Pricing a Tower
-----------------

