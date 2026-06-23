"""Layer 2 compositor for the pricing-bounds objects in
:mod:`aggregate.bounds`.

Holds the bodies of ``Bounds.plot_envelope`` (three-panel envelope cloud),
``Bounds.plot_weights`` (weight contour) and ``_HullEngine.plot`` (per-item
convex-envelope curves with optional premium slice). Each class keeps a
one-line delegating stub.

These are single-consumer exhibits -- the envelope cloud, the weight contour
and the hull-bounds figure have no cross-class content reuse -- so each
content worker and its compositor collapse into this one module (the §1
single-consumer case).
"""

from itertools import cycle

import numpy as np

from ._style import plt, mpl, FIG_W


def plot_bounds_envelope(bounds, *, axs=None, n_resamples=0, alpha=0.05,
                         distortions='ordered', title='',
                         lim=(-0.025, 1.025)):
    """Three-panel envelope figure (formerly ``cloud_view``).

    Panel 1: scatter of sampled cloud columns shaded by weight, plus the
    min/max envelope band. Panels 2-3: the calibrated distortions overlaid on
    the envelope band.

    Parameters
    ----------
    bounds : Bounds
        The bounds object holding ``cloud_df`` / ``weight_df``.
    axs : array of 3 Axes, optional
        If omitted, a new ``1 x 3`` figure is created.
    n_resamples : int, default 0
        If positive, draw this many bracket columns from ``cloud_df``,
        restricted to ``p_lo == 0`` (pricing distortions, those that pin the
        mean), and overplot them coloured by weight.
    alpha : float, default 0.05
        Opacity of the resampled curves.
    distortions : ``'ordered'``, list of dict, or ``'space'``
        What to overlay in panels 2-3. ``'ordered'`` only works for
        ``Portfolio`` objects with calibrated distortions.
    title : str, default ``''``
        Suptitle (applied to all panels).
    lim : tuple, default ``(-0.025, 1.025)``
        x and y axis limits.

    Returns
    -------
    fig, axs : matplotlib figure and array of three Axes.
    """
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(3 * FIG_W, FIG_W),
                                constrained_layout=True, squeeze=False)
        axs = axs[0]
    else:
        axs = np.atleast_1d(axs).flatten()
        fig = axs[0].get_figure()

    norm = mpl.colors.Normalize(0, 1)
    cm = mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r')
    mapper = cm.get_cmap()
    s_eval = np.linspace(0, 1, 1001)

    def _band(ax):
        ax.fill_between(bounds.cloud_df.index, bounds.cloud_df.min(1),
                        bounds.cloud_df.max(1), facecolor='C7', alpha=.15)
        bounds.cloud_df.min(1).plot(ax=ax, label='_nolegend_', lw=1, c='k')
        bounds.cloud_df.max(1).plot(ax=ax, label='_nolegend_', lw=1, c='k')

    if distortions == 'ordered':
        from ..portfolio import Portfolio
        if not isinstance(bounds._obj, Portfolio):
            raise ValueError("distortions='ordered' requires a Portfolio")
        distortions = [
            {k: bounds._obj.distortions[k] for k in ['ccoc', 'tvar']},
            {k: bounds._obj.distortions[k] for k in ['ph', 'wang', 'dual']},
        ]

    ax = axs[0]
    if n_resamples > 0:
        bit = bounds.weight_df.xs(0, drop_level=False) \
                              .sample(n=n_resamples, replace=True) \
                              .reset_index()
        for _, row in bit.iterrows():
            pl, pu = row['p_lower'], row['p_upper']
            w = row['weight']
            bounds.cloud_df[(pl, pu)].plot(ax=ax, lw=1, c=mapper(w),
                                           alpha=alpha, label=None)
        fig.colorbar(cm, ax=ax, shrink=.5, aspect=16,
                     label='Weight to upper threshold')
    _band(ax)
    ax.plot([0, 1], [0, 1], c='k', lw=.25)
    ax.set(xlim=lim, ylim=lim, aspect='equal')

    if isinstance(distortions, dict):
        distortions = [distortions]
    if isinstance(distortions, list):
        name_mapper = {'ccoc': 'CCoC', 'tvar': 'TVaR(p*)',
                       'ph': 'PH', 'wang': 'Wang', 'dual': 'Dual'}
        ls_cycle = list(mpl.lines.lineStyles.keys())
        for ax, dist_dict in zip(axs[1:], distortions):
            lssi = iter(cycle(ls_cycle))
            for k, d in dist_dict.items():
                ax.plot(s_eval, d.g(s_eval), lw=1, ls=next(lssi),
                        label=name_mapper.get(k, k))
            _band(ax)
            ax.plot([0, 1], [0, 1], c='k', lw=.25)
            ax.legend(loc='lower right', ncol=3, fontsize='large')
            ax.set(xlim=lim, ylim=lim, aspect='equal')
        # Average extreme overlay on the last panel
        bounds.cloud_df.mean(1).plot(ax=axs[-1], c=f'C{len(distortions[-1])}',
                                     ls='-.', lw=.5, label='Avg extreme')

    if title:
        for ax in axs:
            ax.set(title=title)

    return fig, axs


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
