"""The one generic ChartDoc renderer: matplotlib realizes the chart IR.

Grows exactly what each converted chart needs (``dev/plan-chart-ir.md``,
pass three). Today, from the surface pilot: the 'surface' and 'heatmap'
panel kinds. matplotlib has no faithful 3-D surface, so a 'surface' panel
renders as its honest 2-D reading, a ``pcolormesh`` projection of the z
grid with a contour overlay, and the title is stamped ``(projection)``;
``strict=True`` raises :class:`~aggregate.charts.ir.ChartCapabilityError`
instead. That is the capability-declaration pattern every later kind
follows: degrade honestly and say so, or refuse loudly, never approximate
silently.

Renderer-side decisions only live here: the sequential ramp (white to the
house primary, one-directional like the density it colors), figure sizing,
the colorbar. Everything semantic (grids, labels, scales, the log-toggle
meaningfulness flag) comes off the document.
"""

import numpy as np

from ..charts.ir import ChartCapabilityError
from ._style import plt, mpl, FIG_W, make_grid

__all__ = ['plot_chartdoc']

# The float-dust floor for log color scales, matching the app's LOG_FLOOR
# (theme.js:63) and the inventory's judgment call J5: below it a value is
# arithmetic noise, not signal.
_LOG_FLOOR = 1e-15

# Panel kinds this renderer realizes natively today. 'surface' is the
# declared degradation (projection); anything else raises until its
# conversion lands.
_NATIVE = {'heatmap'}
_DEGRADED = {'surface'}


def _house_ramp():
    """Sequential colormap, white to the first prop-cycle color.

    A joint density is one-directional, so the ramp is sequential (a
    diverging ramp would imply a midpoint that does not exist), mirroring
    the app's visualMap.
    """
    c0 = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    return mpl.colors.LinearSegmentedColormap.from_list(
        'aggregate_seq', ['#ffffff', c0])


def _render_grid_panel(ax, doc, panel, series, log_z):
    """Render one z-grid panel (surface projection or heatmap)."""
    surf = series.surface
    x = np.asarray(surf.x, dtype=float)
    y = np.asarray(surf.y, dtype=float)
    z = np.asarray(surf.z, dtype=float)
    axes = {a.id: a for a in doc.axes}
    if log_z:
        # One decade under the smallest mass actually present (ignoring
        # float dust), the app's floor-not-holes rule: a zero cell sits on
        # the floor rather than punching a hole in the field.
        pos = z[z > _LOG_FLOOR]
        floor = (10.0 ** np.floor(np.log10(pos.min()))
                 if pos.size else _LOG_FLOOR)
        norm = mpl.colors.LogNorm(vmin=floor, vmax=max(z.max(), floor * 10),
                                  clip=True)
    else:
        norm = mpl.colors.Normalize(vmin=0.0, vmax=z.max() or 1.0)
    mesh = ax.pcolormesh(x, y, z, shading='nearest', cmap=_house_ramp(),
                         norm=norm)
    if np.count_nonzero(z > 0) > 1 and z.max() > z.min():
        ax.contour(x, y, z, colors='w', linewidths=0.5, alpha=0.6)
    zlabel = axes[panel.z_axis].label
    ax.figure.colorbar(mesh, ax=ax, shrink=0.85,
                       label=f'log {zlabel}' if log_z else zlabel)
    ax.set(xlabel=axes[panel.x_axis].label, ylabel=axes[panel.y_axis].label)
    if panel.aspect == 'equal':
        ax.set_aspect('equal')


def plot_chartdoc(doc, ax=None, strict=False, log_z=False):
    """Render a chart document with matplotlib.

    Parameters
    ----------
    doc : ChartDoc
        The document to realize. Currently single-panel documents with a
        z-grid kind ('surface', 'heatmap'); further kinds arrive with
        their conversions.
    ax : matplotlib Axes, optional
        Draw into an existing axes; default makes a figure via the house
        canvas helpers.
    strict : bool
        Raise :class:`ChartCapabilityError` for any panel kind this
        renderer cannot realize faithfully ('surface'), instead of the
        declared degradation.
    log_z : bool
        Log color scale for the z grid. Only honored when the document
        declares ``meta['z_log_ok']``; meaningless otherwise and ignored.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn into.

    Raises
    ------
    ChartCapabilityError
        Under ``strict`` for degraded kinds, and always for kinds with no
        realization here yet.
    """
    for panel in doc.panels:
        if panel.kind in _NATIVE:
            continue
        if panel.kind in _DEGRADED:
            if strict:
                raise ChartCapabilityError(
                    f"panel {panel.id!r} kind {panel.kind!r}: matplotlib "
                    'has no faithful 3-D surface; non-strict renders the '
                    '2-D projection')
            continue
        raise ChartCapabilityError(
            f'panel {panel.id!r} kind {panel.kind!r} is not yet realized '
            'by the matplotlib renderer')
    if len(doc.panels) != 1:
        raise ChartCapabilityError(
            'multi-panel documents are not yet realized by the matplotlib '
            'renderer')

    panel = doc.panels[0]
    if ax is None:
        # Near-square: the z grid reads as a map, and the colorbar takes
        # the balance of the width.
        _, ax = make_grid(1, 1, figsize=(1.25 * FIG_W, FIG_W), squeeze=True)
    fig = ax.figure
    series = [s for s in doc.series if s.panel_id == panel.id]
    log_z = bool(log_z) and bool(doc.meta.get('z_log_ok'))
    _render_grid_panel(ax, doc, panel, series[0], log_z)
    title = doc.title or doc.name
    if panel.kind == 'surface':
        title = f'{title} (projection)'
    ax.set_title(title)
    return fig
