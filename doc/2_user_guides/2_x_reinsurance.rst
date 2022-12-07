.. _2_x_reinsurance:

DecL: Reinsurance
======================


**Objectives:** How to specify occurrence and aggregate reinsurance in DecL.

**Audience:** User who wants to build an aggregate with parametric or discrete frequency and severity distributions.

**Prerequisites:** Familiar with building aggregates using ``build``. Probability theory behind aggregate distributions. Reinsurance terminology.

**See also:** :doc:`2_x_frequency`, :doc:`2_x_severity`, :doc:`2_x_exposure`, :doc:`2_x_mixtures`, :doc:`2_x_limits`, :doc:`2_x_vectorization`, :doc:`../4_dec_Language_Reference`.


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
    @savefig re_pricing_comps.svg
    ax2.set(xlim=xl2, yscale=sc, ylim=ax0.get_ylim(), title='WC insurance charge distribution');
