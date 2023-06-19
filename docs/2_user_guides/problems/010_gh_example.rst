Grübel and Hermesmeier (1999)
-------------------------------

Poisson/Levy Example
~~~~~~~~~~~~~~~~~~~~~~

Here is an example from :cite:t:`Grubel1999`.
The Levy distribution is a zero parameter distribution in ``scipy.stats``. The paper considers an aggregate with Poisson(20) claim count.
The Panjer recursion column can be replicated using more buckets and padding with ``bs=1``. The function ``exact`` uses conditional probability to
compute the aggregate probability of :math:`x-1/2 < X < x+1/2` exactly. The Levy is stable with index :math:`\alpha=1/2`, which means that

.. math::

    X_1 + \cdots + X_n =_d n^2X

for iid Levy variables.

The other models use ``log2=10``, no padding, and varying amounts of tilting.


.. ipython:: python
    :okwarning:

    from aggregate import build, qd
    from scipy.stats import levy

    a = build('agg L 20 claim sev levy poisson', update=False)

    bs = 1
    a.update(log2=16, bs=bs, padding=2, normalize=False, tilt_vector=None)
    df = a.density_df.loc[[1, 10, 100, 1000], ['p_total']] / a.bs
    df.columns = ['accurate']

    def exact(x):
        lam = 20
        n = 100
        # poisson freqs
        p = np.zeros(n)
        a = np.zeros(n)
        p[0] = np.exp(-lam)
        fz = levy()
        for i in range(1, n):
            p[i] = p[i-1] * lam / i
            a[i] = fz.cdf((x+0.5)/i**2) - fz.cdf((x-0.5)/i**2)
        return np.sum(p * a)

    df['exact'] = [exact(i) for i in df.index]

    # other models
    log2 = 10
    for tilt in [None, 1/1024, 5/1024, 25/1024]:
        a.update(log2=log2, bs=bs, padding=0, normalize=False, tilt_vector=tilt)
        if tilt is None:
            tilt = 0
        df[f'tilt {tilt:.4f}'] = a.density_df.loc[[1, 10, 100, 1000], ['p_total']]/a.bs

    qd(df.iloc[:, [1,0,2,3,4, 5]], accuracy=3)

This table is identical to the table shown in the paper.

.. image:: img/gh_table1.png
  :width: 800
  :alt: Original paper table.
