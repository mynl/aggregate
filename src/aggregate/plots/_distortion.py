"""What is left of the layer 2 compositor for :class:`Distortion`.

``plot_distortion`` is gone: ``Distortion.plot`` draws the document
``charts.chart_distortion`` emits, through ``plots._chartdoc``. The two
paths agreed pixel for pixel from ``1.0.0a209``, and keeping both in step by
hand for a picture they already agreed on was the duplication the chart IR
exists to remove.

The affine envelope stays bespoke: it overlays the TVaR decomposition's
affine lines on the curve, which the schema has no way to say, so it is
listed as bespoke rather than half-expressed.
"""

import numpy as np

from ._style import mpl


def plot_distortion_affine(dist, ax=None, n_pts=101, cmap_name='viridis',
                           alpha=1., marker='o', marker_size=4):
    """Render the upper affine envelope of a ``wtdtvar`` distortion.

    Overlays the affine lines of the distortion's TVaR decomposition on the
    distortion curve (``Distortion.plot(dual=False)``), coloured along
    ``cmap_name`` and marked at their ``(s, g(s))`` support points.

    Parameters
    ----------
    dist : Distortion
        A ``wtdtvar``-family distortion exposing ``tvar_info_df``.
    ax : matplotlib.axes.Axes, optional
        Accepted and unused: the base plot always makes its own figure and
        this draws into that. Kept so the signature does not change under
        callers while the drawing is bespoke.
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
    ax = dist.plot(dual=False).axes[0]
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
