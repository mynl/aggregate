.. _2_x_cat:

Catastrophe Modeling Problems
==============================


Occ and Agg PMLs.



1851-2017

======== ===== =========
Category Count Frequency
======== ===== =========
1        116   0.69
2        75    0.45
3        76    0.46
4        20    0.12
5        3     0.02
======== ===== =========

Overall severity from RMI course lognormal (19.6, 2.58)

::

    cat = build('agg USWind [0.69 0.45 0.45 0.12 0.02] claims '
            'sev [exp(-1.7233),    exp(-1.5233),    exp(-1.3233),    exp(-1.1233),   exp(-0.92327)] * '
            'lognorm [2.18 2.28 2.38 2.45 2.58] poisson '
            'aggregate ceded to 50 x 0 and 50 x 50 and 100 x 100 and 100 x 200 and 100 x 300 and '
            '100 x 400 and 500 x 500 and 1e4 x 1000 '
            'note{losses in billions}',
            log2=18, bs=1/2**4, normalize=False)

    with pd.option_context('display.max_rows', 10, 'display.max_columns', 15, 'display.float_format', lambda x: f'{x:,.3f}', 'display.multi_sparse', False):
        display(cat.reins_audit_df)
