Embrechts and Frei (2009)
-----------------------------

Poisson/Pareto Example
~~~~~~~~~~~~~~~~~~~~~~~

:cite:t:`Embrechts2009a`, Panjer recursion versus FFT for compound distributions.

Consider a :math:`\mathrm{Po}(\lambda)\vee\mathrm{Pareto}(\alpha, \beta)`, :math:`\alpha` is shape and :math:`\beta` is scale.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd
    from pandas import option_context
    import matplotlib.pyplot as plt

    alpha = 4
    beta = 3
    freq = 20
    a = build(f'agg EF.1 {freq} claims '
              f'sev {beta} * pareto {alpha} - {beta} '
               'poisson', bs=1/8, log2=8, padding=0, normalize=False)

    qd(a)

The last dataframe shows poor accuracy. Try different ways to compute the aggregate: padding, tilting, and more buckets.

.. ipython:: python
    :okwarning:

    from pandas import option_context

    df = a.density_df[['p_total']].rename(columns={'p_total': 'Pad 0, tilt 0'})

    a.update(bs=1/8, log2=8, padding=1, normalize=False)
    df['Pad 1, tilt 0'] = a.density_df.p_total

    a.update(bs=1/8, log2=8, padding=2, normalize=False)
    df['Pad 2, tilt 0'] = a.density_df.p_total

    a.update(bs=1/8, log2=8, padding=0, tilt_vector=0.01, normalize=False)
    df['Pad 0, tilt 0.01'] = a.density_df.p_total

    a.update(bs=1/32, log2=16, padding=1, normalize=False)
    bit = a.density_df[['p_total']].rename(columns={'p_total': 'log2 16, pad 1, tilt 0'})

    qd(a)

The last dataframe shows a good approximation.

The next figure (compare Figure 1 in the paper, shown below) shows that padding, as recommended in :cite:t:`WangS1998`, removes aliasing as effectively as padding, albeit at the expense of a longer FFT computation. The log density shows the aliasing is completely removed.

.. ipython:: python
    :okwarning:

    f, axs = plt.subplots(1, 2, figsize=(2 * 3.5, 2.45), constrained_layout=True, squeeze=True)
    ax0, ax1 = axs.flat
    df.plot(ax=ax0, logy=False)
    df.plot(ax=ax1, logy=True)
    ax0.legend(loc='upper left')
    @savefig ef_1.png scale=20
    ax1.legend(loc='lower right');


.. image:: img/ef_fig1.png
  :width: 800
  :alt: Original paper figure.

Clearly there is not enough *space* with only 2**8 buckets. Expanding to 2**16 and using a finer bucket covers a more realistic range. The log density plot shows a change in regime from Poisson body to Pareto tail. The extreme tail can be approximated by differentiating Feller's theorem, which says the survival function is converges to :math:`20\mathsf{Pr}(X>x)` where :math:`X` is the Pareto severity (right hand plot). The multiplication by four accounts for the different ``bs`` values.


.. ipython:: python
    :okwarning:

    f, axs = plt.subplots(1, 2, figsize=(2 * 3.5, 2.45), constrained_layout=True, squeeze=True)
    ax0, ax1 = axs.flat

    df.plot(ax=ax0, logy=False)
    (bit * 4).plot(ax=ax0, lw=3, alpha=.5);

    bit.plot(ax=ax1, logy=True);
    # density from tail, need to divide by bs
    ax1.plot(bit.index, (20*4/3*a.bs)*(3/(3+bit.index))**5, label='Feller approximation');
    ax0.set(xlim=[-5, a.q(0.99999)]);
    ax0.legend(loc='upper right');
    @savefig ef_2.png scale=20
    ax1.legend(loc='upper right');


Choice of Bandwidth (Bucket Size)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example replicates parts of Table 1. As well as the 99.9%ile it shows the 99.9999%ile.

.. ipython:: python
    :okwarning:

    import pandas as pd

    a = build('agg EF.2 50 claims sev expon poisson', update=False)
    ans = []
    for log2, bs in zip([10, 10, 10, 16, 16, 16, 16], [1, 1/2, 1/8, 1/8, 1/16, 1/64, 1/512]):
        a.update(log2=log2, bs=bs, padding=1)
        ans.append([log2, 1/bs, a.q(0.999), a.q(1-1e-6)])

    df = pd.DataFrame(ans, columns=['log2', '1/bs', 'p999', 'p999999'])
    qd(df, accuracy=4)


.. ipython:: python
    :suppress:

    plt.close('all')

