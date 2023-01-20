Other Papers
--------------------

Miscellaneous short examples from various texts.

Contents
~~~~~~~~~

* :ref:`other wang`
* :ref:`wang weather`
* :ref:`gerber stop loss`
* :ref:`richardson deferred`


.. _other wang:

Wang on the Wang Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Source paper**: :cite:t:`Wang2000`.

Pricing by Layer
"""""""""""""""""""

**Concepts**: Layer expected loss and risk adjusted layer technical premium with Wang and proportional spectral risk measures.

**Setup**: Ground-up Pareto risk, shape 1.2, scale 2000. Layer and compare Wang(0.1) and PH(0.9245) pricing.

**Source Exhibits**:

.. image:: img/Wang2000_table_1.png
    :width: 600
    :alt: Original paper table.

**Thanks**:  Zach Eisenstein of Aon.

**Code**:

Build the portfolio.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd
    layers = [0, 50e3, 100e3, 200e3, 300e3, 400e3, 500e3, 1000e3, 2000e3, 5000e3, 10000e3]
    a1 = build('agg Wang.t1 '
               '1 claim '
               '10000e3 xs 0 ' # limit the severity to avoid infinite variance
               'sev 2000 * pareto 1.2 - 2000 '
               f'occurrence ceded to tower {layers} '
               'fixed'
              )
    qd(a1)

Expected loss and other ceded statistics by layer.

.. ipython:: python
    :okwarning:

    ff = lambda x: x if type(x)==str else (
            f'{x/1e6:,.1f}M' if x >= 500000 else
            (f'{x:,.3f}' if x < 100 else f'{x:,.0f}'))
    fp = lambda x: f'{x:.1%}'
    qdl = lambda x: qd(x, index=False, line_width=200, formatters={'pct': fp},
                       float_format=ff, col_space=10)
    qdl(a1.reinsurance_occ_layer_df .xs('ceded', 1, 1).droplevel(0).reset_index(drop=False))


ZE provided function to make the exhibit table. The column ``pct`` shows the relative loading.

.. ipython:: python
    :okwarning:

    def make_table(agg, layers):
        agg_df = agg.density_df
        layer_df = agg_df.loc[layers, ['F', 'S', 'lev', 'gS', 'exag']]
        layer_df['layer_el'] = np.diff(layer_df.lev, prepend = 0)
        layer_df['premium'] = np.diff(layer_df.exag, prepend = 0)
        layer_df['pct'] = layer_df['premium'] / layer_df['layer_el'] - 1
        layer_df = layer_df.rename_axis("exhaust").reset_index()
        layer_df['attach'] = layer_df['exhaust'].shift(1).fillna(0)
        qdl(layer_df.loc[1:, ['attach', 'exhaust', 'layer_el', 'premium', 'pct']])

Make the distortions and apply. First, Wang.

.. ipython:: python
    :okwarning:

    d1 = build('distortion wang_d1 wang 0.1')
    a1.apply_distortion(d1)
    make_table(a1, layers)

Make the distortions and apply. Second, PH.

.. ipython:: python
    :okwarning:

    d2 = build('distortion wang_d2 ph 0.9245')
    a1.apply_distortion(d2)
    make_table(a1, layers)

It appears the layer 50000 xs 0 is reported incorrectly in the paper.


Satellite Pricing
"""""""""""""""""""

**Concepts**: The cost of a Bernoulli risk with a 5% probability of a total loss of $100m using Wang(0.1) distortion.

**Thanks**:  Zach Eisenstein of Aon.

**Code**:

Build the portfolio. Illustrates how to set up a Bernoulli.

.. ipython:: python
    :okwarning:

    a2 = build('agg wang2 0.05 claims dsev [100] bernoulli')
    qd(a2)

Check the distribution output.

.. ipython:: python
    :okwarning:

    qd(a2.density_df.query('p_total > 0')[['p_total', 'F', 'S']])

Build and apply the distortion. Use the :meth:`price` method. The first argument selects the 100% quantile for pricing, i.e., the limit is fully collateralized. First with the Wang from above. The relative loading reported is the complement of the loss ratio.

.. ipython:: python
    :okwarning:

    qd(a2.price(1, d1))

Try with a more severe distortion.

.. ipython:: python
    :okwarning:

    d3 = build('distortion wang_d2 wang 0.15')
    qd(a2.price(1, d2))


.. _wang weather:

Wang on Weather Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Source paper**: :cite:t:`Wang2002`.

**Concepts**: Applying Wang transform to empirical distribution via an :class:`Aggregate` object.

**Source Exhibits**:

.. image:: img/Wang2002_table_2.png
    :width: 600
    :alt: Original paper table.

**Data**:

Create a dataframe from the heating degree days (HDD) history laid out in Table 1.

.. ipython:: python
    :okwarning:

    d = '''Dec-79
    972.5
    Dec-87
    1018.5
    Dec-95
    1199.5
    Dec-80
    1147.0
    Dec-88
    1155.0
    Dec-96
    1156.0
    Dec-81
    1244.0
    Dec-89
    1474.5
    Dec-97
    1040.0
    Dec-82
    901.0
    Dec-90
    1129.5
    Dec-98
    940.5
    Dec-83
    1573.0
    Dec-91
    1077.5
    Dec-99
    1090.5
    Dec-84
    1055.0
    Dec-92
    1129.5
    Dec-00
    1517.5
    Dec-85
    1488.0
    Dec-93
    1090.5
    Dec-86
    1065.5
    Dec-94
    938.5'''
    import pandas as pd
    d = d.split('\n')
    df = pd.DataFrame(zip(d[::2], d[1::2]), columns=['month', 'hdd'])
    df['hdd'] = df.hdd.astype(float)
    qd(df.head())
    qd(df.describe())


**Code**:

Create an empirical aggregate based on the HDD data.

.. ipython:: python
    :okwarning:

    hdd = build(f'agg HDD 1 claim dsev {df.hdd.values} fixed'
                , bs=1/32, log2=16)
    qd(hdd)


Build the distortion and apply to call options at different strikes. Reproduces Table 2.

.. ipython:: python
    :okwarning:

    from aggregate.extensions import Formatter
    d1 = build('distortion w25 wang .25')
    ans = []
    strikes = np.arange(1250, 1501, 50)
    bit = hdd.density_df.query('p_total > 0')[['p_total']]
    for strike in strikes:
        ser = bit.groupby(by= lambda x: np.maximum(0, x - strike)).p_total.sum()
        ans.append(d1.price(ser, kind='both'))
    df = pd.DataFrame(ans, index=strikes,
                     columns=['bid', 'el', 'ask'])
    df.index.name = 'strike'
    df['loading'] = (df.ask - df.el) / df.el
    qd(df.T, float_format=Formatter(dp=2, w=8))

.. _gerber stop loss:

Gerber: Stop Loss Premiums
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Source papers**: :cite:t:`Gerber1982`.

**Concepts**: Stop loss and survival functions for Poisson uniform(1,3) aggregate. Claim count 1, 10, and 100.

**Source Exhibits**: Matches columns labeled exact in Tables 1-6.

**Code**:

Expected claim count 1.

.. ipython:: python
    :okwarning:

    gerber1 = build('agg Gerber1 1 claim sev 2 * uniform + 1 poisson'
                    , bs=1/1024)
    qd(gerber1)
    bit = gerber1.density_df.loc[0:21:2*1024, ['S', 'lev']]
    bit['stop_loss'] = gerber1.agg_m - bit.lev
    qd(bit)

Expected claim count 10.

.. ipython:: python
    :okwarning:

    gerber10 = build('agg Gerber10 10 claim sev 2 * uniform + 1 poisson'
                    , bs=1/128)
    qd(gerber10)
    bit = gerber10.density_df.loc[15:61:5*128, ['S', 'lev']]
    bit['stop_loss'] = gerber10.agg_m - bit.lev
    qd(bit)

Expected claim count 100.

.. ipython:: python
    :okwarning:

    gerber100 = build('agg Gerber100 100 claim sev 2 * uniform + 1 poisson'
                    , bs=1/16)
    qd(gerber100)
    bit = gerber100.density_df.loc[180:301:20*16, ['S', 'lev']]
    bit['stop_loss'] = gerber100.agg_m - bit.lev
    qd(bit)

.. _richardson deferred:

Richardson's Deferred Approach to the Limit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Source papers**: :cite:t:`Embrechts1993`, :cite:t:`Grubel2000`.

**Concepts**: Given estimators of an unknown quantity :math:`A^* = A(h) + ch^\alpha + O(h^\beta)` with :math:`\alpha < \beta`. Evaluate at :math:`h` and :math:`h/t`. Multiply the estimate at :math:`h/t` by :math:`t^\alpha`, subtract the original estimate, and rearrange to get

.. math::

    A^* = \frac{t^\alpha A(h/t) - A(h)}{t^\alpha - 1} + O(h^\beta).

The truncation error order of magnitude has decreased. The constant :math:`c` need not be known. Applying this approach to estimate the density as pmf divided by bucket size, :math:`f_{h} / h`, :cite:t:`Embrechts1993` report the following.

**Setup**: Poisson(20) exponential aggregate.

**Source Exhibits**: Figure 1.

The variable ``egp3`` is treated as the exact answer. It could also be approximated using the series expansion, but this has been shown already, in REF. Set up basic portfolios evaluated at different bucket sizes.

.. ipython:: python
    :okwarning:

    egp1 = build('agg EGP 20 claims sev expon 1 poisson',
                bs=1/16, log2=10)
    qd(egp1)
    egp2 = build('agg EGP 20 claims sev expon 1 poisson',
                bs=1/32, log2=11)
    qd(egp2)
    egp3 = build('agg EGP 20 claims sev expon 1 poisson',
                 log2=16)
    qd(egp3)

Concatenate and estimated densities from pmf. Compute errors to ``egp3``. Compute the Richardson extrapolation. It is indistinguishable from ``egp3``. The last table shows cases with the largest errors.

.. ipython:: python
    :okwarning:

    import pandas as pd
    df = pd.concat((egp1.density_df.p_total, egp2.density_df.p_total, egp3.density_df.p_total),
              axis=1, join='inner', keys=[1, 2, 3])
    df[1] = df[1] * 16;                   \
    df[2] = df[2] * 32;                   \
    df[3] = df[3] * (1<<10);              \
    df['rich'] = (4 * df[2] - df[1]) / 3; \
    df['diff_1'] = df[1] - df[3];         \
    df['diff_2'] = df[2] - df[3];         \
    m = df.diff_2.max() * .9;             \
    ax = df[['diff_1', 'diff_2']].plot(figsize=(3.5, 2.45));      \
    (df['rich'] - df[3]).plot(ax=ax, lw=.5, label='Richardson');
    @savefig rich.png scale=20
    ax.legend(loc='upper right')
    qd(df.query(f'diff_2 > {m}').iloc[::10])

