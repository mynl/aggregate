.. _massive bivariate:

Massive (Disk-Backed) Bivariate Distributions
=============================================

**Prerequisites:** :class:`~aggregate.bivariate.BivariateAggregate` basics
(the ``bivariate`` / ``dbvsev`` / ``netceded`` DecL forms); familiarity with
``update``'s ``log2`` / ``bs`` sizing. Requires the optional ``zarr`` extra::

    pip install aggregate[massive]

The problem and the idea
------------------------

An in-core bivariate update is quadratic in the per-axis grid length: at
``log2 = (14, 14)`` the joint density alone is 2 GB, and the padded complex
FFT buffers several times that. The massive path removes the RAM ceiling
entirely: the separable 2-D compound ``iFFT2(pgf(FFT2(S)))`` runs as **three
streamed passes over disk-backed (zarr) stores**, with the transpose between
the FFT directions done *through disk* — pass 2 simply reads the pass-1
store column-wise, tiles in orthogonal order. Peak RAM is bounded by a band
(~0.5–1 GB at ``(16, 16)``) regardless of grid size, and the realized joint
density lives on disk in physical order. There is no dask and no cluster —
single process, threaded scipy FFTs.

All of the bv's windowing machinery — measured per-axis windows, signed
axes (``i0``), shifted output windows (``j0``), the decoupled FFT buffer
length — is reused unchanged; out-of-core the window relabel degenerates to
bookkeeping applied per band at write time.

Usage
-----

One new argument. Point ``update`` at a backing directory (prefer a fast
local drive with headroom ~2x the density for the transient staging store)::

    from aggregate import build

    mv = build('''bivariate MV 25 claims
        agg A dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
        agg B dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
        copula gumbel 0.4
        poisson''', update=False)
    mv.update(log2=(14, 14), store_dir='D:/scratch/bv1')

    bd = mv.bivariate          # a MassiveBivariateDistribution

``store_dir=None`` (the default) is the in-core path, byte-for-byte
untouched. On the massive path ``padding`` defaults to **0**: the measured
window already covers the support to ``10**-window_nines``, so padding would
quadruple disk and time for no useful protection — ``deficit`` is the guard.
All three modes route through the same kernel; only the per-claim severity
formation differs (and it is kept RAM-bounded: sparse scatters for
``netceded`` / ``dbvsev``, a support-trimmed copula rectangle mass).

The result surface
------------------

``marginals``, ``moments``, ``corr``, ``deficit``, ``info`` and
``summary_df`` on the bv all work from exact accumulators folded during the
final pass — no disk read. The container adds:

* ``bd.density`` — a **lazy zarr view**; ``bd.density[1000:1010, :]`` reads
  just those tiles (a full ``[:]`` read is you explicitly asking for the
  whole array).
* ``bd.marginal(i)`` — an exact axis marginal as a ``GridDistribution``.
* ``bd.slice(x=...)`` / ``bd.slice(y=...)`` — conditional laws, one
  row/column of tiles, instant at any size.
* ``bd.transformed_moments(f)`` — streamed exact moments of ``f(X, Y)``.
* ``MassiveBivariateDistribution.reopen(store_dir)`` — the store is
  self-describing; reconstruct the container in a later session without
  re-running the FFT.

Dict pushforward
----------------

The money application: reduce the on-disk joint to the 1-D laws the analyst
actually wants — several at once, in **one** pass over the density::

    out = bd.pushforward(
        {'net':     lambda x, y: x + 0.8 * y,
         'excess':  lambda x, y: np.maximum(x + y - 1000, 0),
         'premium': 1500.0},
        bs=1.0, bs_total=1.0)
    out['net'], out['excess'], out['premium'], out['total']

* With two or more functions the **total** ``t = Σ f_i`` is also returned
  (key ``'total'``), accumulated *pre-bucketing* — evaluated exactly per
  cell, bucketed once with its own required ``bs_total`` — so
  ``mean(total) == Σ mean(f_i)`` is exact, never a sum of bucketed errors.
* A single function skips the total (the usual, fast case).
* Constants (numbers, or callables detected constant on a probe lattice)
  never enter the sweep: a point mass, a scalar shift inside the total.
* Output windows are measured by a label-only pre-sweep (no disk); pin them
  per key with ``windows={'net': (lo, hi), ...}``.
* Every result carries the Est-vs-EX audit frame
  (``.pushforward_audit_df``) — the exact streamed moments cost nothing
  extra.

The same dict form works on the in-core
:meth:`~aggregate.bivariate.BivariateDistribution.pushforward` for API
uniformity.

Visualization
-------------

``bd.plot()`` is the exploration-grade static exhibit: constant cost at any
zoom, because every render reads only the level of a **sum/max/min
decimation pyramid** (built during the update) matching the pixel budget.
Log-scale color by default; the max channel adds a luminance boost where a
pixel hides sub-pixel structure, so one-cell-wide ridges (the netceded
comonotone filament) and atoms stay lit at any coarsening. Axis atom strips
and a ``P(0,0)`` badge surface the zero-inflation headline numbers instead
of letting them saturate the colormap. ``window=((x0, x1), (y0, y1))``
zooms; ``contours=True`` overlays log-density decades; ``exceedance=True``
overlays the joint tail ``P(X > x, Y > y)``; ``bd.plot_slice(x=...)`` draws
a conditional.

``bd.explore()`` (requires ``pip install aggregate[viz]``) launches the
Tier-2 interactive app in JupyterLab — holoviews + datashader + bokeh over
the same pyramid: Google-Maps-style pan/zoom with dynamic re-aggregation,
sum/max/min channel toggle, hover readout, linked marginal panels that
re-window to the viewport, and click-to-slice conditionals.

Budgets
-------

Complex128 staging + float64 density, ``padding=0``, uncompressed worst
case (zarr's zstd compression shrinks real densities substantially):

===============  =======  ============  ================  =================
grid (per axis)  cells    ``z2`` stage  ``density.zarr``  est. update wall
===============  =======  ============  ================  =================
2^13 = 8192      67 M     0.5 GB        0.5 GB            ~15 s
2^14 = 16384     268 M    2.1 GB        2.1 GB            ~1–3 min
2^15 = 32768     1.1 G    8.6 GB        8.6 GB            ~5–10 min
2^16 = 65536     4.3 G    34 GB         34 GB             ~15–30 min
===============  =======  ============  ================  =================

The staging store is deleted after the update by default
(``keep_transform=False``). RAM stays ~0.5–1 GB with the default 512-row
bands — constant in grid size. Pushforwards re-read the density (~1–3 min
at ``(16, 16)`` on NVMe) and are repeatable at will; the FFT never reruns.
