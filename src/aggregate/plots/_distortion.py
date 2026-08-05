"""Layer 2 compositor for :class:`aggregate.spectral.Distortion`.

Holds the body of ``Distortion.plot`` -- the ``g`` / ``g_dual`` distortion
curves on either a linear ``[0, 1]^2`` panel or a log-log return-period panel.
The class keeps only a one-line delegating stub.

A distortion is a single-panel exhibit with no cross-class content reuse, so
its content worker and compositor collapse into this one module (the §1
single-consumer case).
"""

import numpy as np

from ..constants import DISTORTION_DUAL_TEX
from ._style import plt, mpl, FIG_W, FIG_H


def plot_distortion(dist, xs=None, n=101, both=True, ax=None, plot_points=True,
                    scale='linear', c=None, c_dual=None, size='small', **kwargs):
    """Plot a :class:`Distortion` ``g`` (and optionally its dual ``g_dual``).

    Parameters
    ----------
    dist : Distortion
        The distortion to plot.
    xs : array_like, optional
        x values; defaults to ``density_df.index`` (linear) or a log-spaced
        grid (return scale).
    n : int
        Grid size for ``scale='return'`` (ignored on linear scale, which uses
        the cached ``density_df`` grid).
    both : bool
        Also plot ``g_dual``.
    ax : matplotlib.axes.Axes, optional
        Existing Axes; if ``None`` a new figure is created.
    plot_points : bool
        Legacy flag (was used by the removed ``ConvexDistortion``).
    scale : {'linear', 'return'}
        Linear plot on ``[0, 1]^2`` or log-log return-period scale.
    size : str or float
        ``'small'`` / ``'large'`` figure preset or a numeric side length.
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    On linear scale the curve is read straight from ``density_df`` so the knot
    splicing (TVaR kink, BiTVaR/WtdTVaR knots, mass-at-0 epsilon) is reflected
    directly in the plot.
    """
    assert scale in ['linear', 'return']

    if scale == 'return':
        xs = 10 ** np.linspace(-10, 0, n)
        y1 = dist.g(xs)
        y2 = dist.g_dual(xs) if both else None
    else:
        if xs is None:
            df = dist.density_df
            xs = df.index.to_numpy()
            y1 = df['g'].to_numpy()
            y2 = df['g_dual'].to_numpy() if both else None
        else:
            y1 = dist.g(xs)
            y2 = dist.g_dual(xs) if both else None

    if ax is None:
        if size == 'small':
            sz = FIG_H
        elif isinstance(size, (float, int)):
            sz = size
        else:
            sz = FIG_W
        fig, ax = plt.subplots(1, 1, figsize=(sz, sz), layout="constrained")

    if c is None:
        c = 'C0'
    if c_dual is None:
        c_dual = 'C1'
    if scale == 'linear':
        ax.plot(xs, y1, c=c, label=dist.label, **kwargs)
        if both:
            ax.plot(xs, y2, c=c_dual, label=DISTORTION_DUAL_TEX, **kwargs)
        ax.plot(xs, xs, color='k', lw=0.5, alpha=0.5)
    elif scale == 'return':
        ax.plot(xs, y1, c=c, label=dist.label, **kwargs)
        if both:
            ax.plot(xs, y2, c=c_dual, label=DISTORTION_DUAL_TEX, **kwargs)
        ax.set(xscale='log', yscale='log',
               xlim=[1 / 5_000, 1], ylim=[1 / 5_000, 1])
        ax.plot(xs, xs, color='k', lw=0.5, alpha=0.5)

    # Axis labels: this was the one compositor in plots/ that drew none, and
    # it draws them now so it reads like the rest of the package and matches
    # what the ChartDoc renderer draws from the same axis labels.
    ax.set(title=dist.label, aspect='equal', xlabel='s', ylabel='g(s)')
    if scale == 'linear':
        # The unit square, drawn as the unit square: g maps [0, 1] to
        # [0, 1], so autoscale's 5% margin only adds white where no
        # distortion can go, and it puts the identity diagonal off the
        # corners it belongs in.
        ax.set(xlim=[0, 1], ylim=[0, 1],
               xticks=np.linspace(0, 1, 6),
               yticks=np.linspace(0, 1, 6))
    if both:
        ax.legend(loc='upper left', fontsize='x-small')
    return ax


def plot_distortion_affine(dist, ax=None, n_pts=101, cmap_name='viridis',
                           alpha=1., marker='o', marker_size=4):
    """Render the upper affine envelope of a ``wtdtvar`` distortion.

    Overlays the affine lines of the distortion's TVaR decomposition on the
    distortion curve (``plot_distortion(both=False)``), coloured along
    ``cmap_name`` and marked at their ``(s, g(s))`` support points.

    Parameters
    ----------
    dist : Distortion
        A ``wtdtvar``-family distortion exposing ``tvar_info_df``.
    ax : matplotlib.axes.Axes, optional
        Existing Axes; if ``None`` the base ``plot`` creates one.
    n_pts : int, default 101
        Number of points along ``[0, 1]`` for each affine line.
    cmap_name : str, default 'viridis'
        Colormap for the affine line family.
    alpha : float, default 1.0
        Line opacity.
    marker : str, default 'o'
        Support-point marker.
    marker_size : float, default 4
        Support-point marker size.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = dist.plot(both=False)
    ps = np.linspace(0, 1, n_pts)
    df = dist.tvar_info_df
    n_lines = len(df)
    cmap = mpl.colormaps.get_cmap(cmap_name)
    colors = [cmap(i / max(1, n_lines - 1)) for i in range(n_lines)]
    for c, (n, r) in zip(colors, df.iterrows()):
        if np.isnan(r.slope):
            continue
        line = r.intercept + r.slope * ps
        line = np.where((line >= 0) & (line <= 1), line, np.nan)
        ax.plot(ps, line, lw=0.5, color=c, alpha=alpha)
    if len(df) < 20:
        ax.scatter(df.s, df.gs, color=colors,
                   marker=marker, s=marker_size, zorder=3)
    return ax
