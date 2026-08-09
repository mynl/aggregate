"""Layer 2 compositor for the pricing-bounds objects in
:mod:`aggregate.bounds`.

Holds the bodies of ``Bounds.plot_weights`` (weight contour) and
``_HullEngine.plot`` (per-item convex-envelope curves with optional premium
slice). Each class keeps a one-line delegating stub.

``plot_bounds_envelope`` is gone: ``Bounds.plot_envelope`` draws the
document ``charts.chart_envelope`` emits, two panels where it drew three.
The weight contour and the hull view stay bespoke, the first because a
level set over ``(p_lo, p_hi)`` is a grid panel nobody has asked for yet
and the second because it reads private engine state (``_T``, ``_A``,
``_hulls``) with no public frame behind it.

These are single-consumer exhibits -- the envelope cloud, the weight contour
and the hull-bounds figure have no cross-class content reuse -- so each
content worker and its compositor collapse into this one module (the §1
single-consumer case).
"""

import numpy as np

from ._style import plt, mpl, FIG_W


def plot_bounds_weights(bounds, ax=None, *, levels=20, colorbar=True):
    """Contour plot of the bracketing weight as a function of ``(p_lo, p_hi)``.

    Parameters
    ----------
    bounds : Bounds
        The bounds object holding ``weight_df``.
    ax : Axes, optional
        Target axes; created if omitted.
    levels : int, default 20
        Contour levels.
    colorbar : bool, default True
        Attach a colorbar.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(FIG_W, FIG_W),
                             constrained_layout=True)
    bit = bounds.weight_df['weight'].unstack()
    img = ax.contourf(bit.columns, bit.index, bit,
                      cmap='viridis_r', levels=levels)
    ax.set(xlabel='p_upper', ylabel='p_lower',
           title='Weight for p_upper', aspect='equal')
    if colorbar:
        ax.get_figure().colorbar(img, ax=ax, shrink=.5, aspect=16,
                                 label='Weight to p_upper')
    return ax


def plot_hull_bounds(engine, items=None, P=None, axs=None, max_t=None):
    """Plot each item's curve, convex envelopes, and optional P-slice.

    Parameters
    ----------
    engine : _HullEngine
        The hull engine (``Bounds`` / ``AllocationBounds``) to plot.
    items : list of str, optional
        Default: all items.
    P : float, optional
        Draw the vertical slice at T = P and mark the bounds.
    axs : array of Axes, optional
        One per item; created if omitted.
    max_t : float, optional
        Truncate the T axis (the far tail compresses the picture).

    Returns
    -------
    array of Axes
    """
    if items is None:
        items = engine._y_names
    if axs is None:
        n = len(items)
        ncols = min(n, 3)
        nrows = -(-n // ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.8 * nrows),
                                constrained_layout=True, squeeze=False)
        axs = axs.flat
    for ax, u in zip(axs, items):
        j = engine._y_names.index(u) + 1
        t, y = engine._T, engine._A[:, j]
        if max_t is not None:
            mask = t <= max_t
            t, y = t[mask], y[mask]
        ax.plot(t, y, lw=0.75, c='C0', label=engine._curve_label)
        for side, c in (('lower', 'C2'), ('upper', 'C3')):
            h = engine._hulls[u][side]
            th, yh = engine._T[h], engine._A[h, j]
            if max_t is not None:
                m = th <= max_t
                th, yh = th[m], yh[m]
            ax.plot(th, yh, lw=1.25, c=c, ls='--', label=side)
        if P is not None:
            lo, *_ = engine._slice(u, 'lower', np.array([float(P)]))
            hi, *_ = engine._slice(u, 'upper', np.array([float(P)]))
            ax.axvline(P, lw=0.5, c='k')
            ax.plot([P, P], [lo[0], hi[0]], lw=2.5, c='k', solid_capstyle='butt')
            ax.plot([P, P], [lo[0], hi[0]], 'o', ms=4, c='k')
        ax.set(title=u, xlabel=engine._xlabel, ylabel=engine._ylabel)
        ax.legend(fontsize='x-small')
    return axs
