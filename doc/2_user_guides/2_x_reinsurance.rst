.. _2_x_reinsurance:

Reinsurance
======================


**Objectives:** How to specify occurrence and aggregate reinsurance in ``agg``.

**Audience:** User who wants to build an aggregate with parametric or discrete frequency and severity distributions.

**Prerequisites:** Familiar with building aggregates using ``build``. Understand probability theory behind aggregate distributions.

**See also:** :doc:`2_x_frequency`, :doc:`2_x_severity`, :doc:`2_x_exposure`, :doc:`2_x_mixtures`, :doc:`2_x_limits`, :doc:`2_x_vectorization`, :doc:`../4_agg_Language_Reference`.



share of, part of, occ and acc.; net of, ceded to.

tower.


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
