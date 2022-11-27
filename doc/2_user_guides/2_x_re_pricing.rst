.. _2_x_re_pricing:

Reinsurance Pricing
==========================

**Objectives:** Solving practical reinsurance problems, including valuation of swings, slides, profit commissions, impact of aggregate limits and deductibles, surplus share, property risk rating, limits profile exposure rating, limited reinstatements.

**Audience:** User who wants to build an aggregate with parametric or discrete frequency and severity distributions.

**Prerequisites:** Familiar with building aggregates using ``build``. Understand probability theory behind aggregate distributions.

**See also:** :doc:`2_x_reinsurance`, :doc:`2_x_ir_pricing`, :doc:`problems/0x0_bahnemann`/


Reinsurance Pricing Applications
--------------------------------

David Clark *Basics of Reinsurance Pricing*, Actuarial Study Note, CAS (Arlington, VA) 2014 revised version.

* Excess of loss exposure rating
* Creation of a severity curve from a limit profile and exposure rating
* Surplus share
* AAD on excess or quota share
* Aggregate limit on excess or quota share
* Variable features

  - Swing rated
  - Slide (balanced)
  - Profit commission
  - Loss corridor

* Property curves
* Reinstatements



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

