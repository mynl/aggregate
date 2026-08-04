.. _intro 1min:

``aggregate`` in One Minute
=============================

Insurance losses arrive as a random number of claims, each of a random size.
Every number an actuary is paid to produce, a premium, a reserve range, a
capital requirement, the value of a reinsurance program, is a statement about
the distribution of the total, :math:`S = X_1 + \cdots + X_N`. That distribution
almost never has a closed form, so practice falls back on matching a few
moments (wrong in the tail, which is where the money is) or simulation (slow,
noisy exactly at the quantiles that drive capital, and rerun from scratch for
every change of limit).

``aggregate`` computes the whole distribution, fast and essentially exactly,
from a one-line, plain-English description of the risk:

.. ipython:: python

    from aggregate import build, qd

    a = build('agg Casualty 250 claims 1000 xs 0 sev lognorm 100 cv 1.5 poisson')
    qd(a)

That is 250 expected claims, lognormal severity with mean 100 and CV 1.5, each
claim capped by a 1,000 per-occurrence limit, Poisson frequency. It built in a
few hundredths of a second, and the table is its own audit: moments computed
exactly from the specification, side by side with the distribution actually
produced. Now ask it anything:

.. ipython:: python

    print(f'mean              {a.est_m:10,.0f}')
    print(f'99.9th percentile {a.q(0.999):10,.0f}')
    print(f'99.9% TVaR        {a.tvar(0.999):10,.0f}')

No simulation, no seed, no noise in the tail, and when the limit changes,
rebuilding costs milliseconds, not an afternoon.

.. code-block:: bash

    pip install aggregate

Have five minutes? :doc:`intro-5min` shows layers, reinsurance, and how to check
the engine against answers you already know. Full documentation:
`aggregate.readthedocs.io <https://aggregate.readthedocs.io/>`_.
