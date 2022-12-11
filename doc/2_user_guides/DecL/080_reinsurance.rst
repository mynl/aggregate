.. _2_x_reinsurance:

DecL: Reinsurance
======================


**Objectives:** How to specify occurrence and aggregate reinsurance in DecL.

**Audience:** User who wants to build an aggregate with parametric or discrete frequency and severity distributions.

**Prerequisites:** Familiar with building aggregates using ``build``. Probability theory behind aggregate distributions. Reinsurance terminology.

**See also:** :doc:`2_x_frequency`, :doc:`2_x_severity`, :doc:`2_x_exposure`, :doc:`2_x_mixtures`, :doc:`2_x_limits`, :doc:`2_x_vectorization`, :doc:`../4_dec_Language_Reference`.



.. _2_agg_class_reinsurance_clause:

The ``reinsurance`` Clauses
----------------------------

Occurrence and aggregate reinsurance can be specified in the same way as limits and deductibles.
Both clauses are optional.
The ceded or net position can be output. Layers can be stacked and can include co-participations. For example, the three programs (the last displayed over four lines):

    agg Trucking 5000 loss 1000 xs 0 sev lognorm 50 cv 1.75 occurrence net of 750 xs 250 poisson

    agg WorkComp 15000 loss 500 xs 0 sev lognorm 50 cv 1.75 poisson aggregate ceded to 50% so 2000 xs 15000

    agg Trucking 5000 loss 1000 xs 0 \
    sev lognorm 50 cv 1.75 \
    occurrence net of 50% so 250 xs 250 and 500 xs 500 poisson \
    aggregate net of 250 po 1000 xs 4000 and 5000 xs 5000

specify the following:

1. The distribution of losses to the net position on the Trucking policy after a per occurrence cession of the 750 xs 250 layer. This net position could also be written without reinsurance as

    agg Trucking 4500 loss  250 xs 50 sev lognorm 50 1.75 poisson

  All occurrence reinsurance has free and unlimited reinstatements. Running

    agg Trucking 5000 loss 1000 xs 0 sev lognorm 50 cv 1.75 occurrence ceded to 750 xs 250 poisson

  would model ceded losses.

2. The distribution of losses to an aggregate protection for the 2000 xs 15000 layer of total losses, limited to 500. The underlying business could be an SIR on a large account Workers Compensation policy, and the aggregate is a part of the insurance charge (Table L, M).

3. Back to Trucking. Now we apply two occurrence layers. The first, 250 xs 250, is only 50% placed (so stands for share of), and the second is 100% of 500 xs 500. The net of these programs flows through to aggregate layers, 250 part of of 1000 xs 4000 (25% placement), and 100% of the 5000 xs 5000 aggregate layers. The modeled outcome is net of all four layers. In this case, it is not possible to write the net of occurrence using limits and attachments.

The distributions for these models are shown  in `realistic examples`_.

See :ref:`reinsurance pricing examples <2_x_re_pricing>` more examples, including an approach to reinstatements.


.. _realistic examples:

Basic Examples
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
    agg Trucking.Retention \
    5000 loss \
    1000 xs 0 \
    sev lognorm 50 cv 1.75 \
    occurrence net of 50% so 250 xs 250 and 500 xs 500 \
    poisson \
    aggregate net of 250 po 1000 xs 4000 and 5000 xs 5000
    agg WorkComp.InsChrg 15000 loss 500 xs 0 sev lognorm 50 cv 1.75 poisson \
    aggregate ceded to 50% so 2000 xs 15000
    ''', approximation='exact')

    for a in ans:
        qd(a.name)
        qd(a.object)
        print('-'*80 + '\n')

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
    ax0.legend()
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


