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
from ..constants import LOG_FLOOR
from ._style import plt, mpl, FIG_H, FIG_W, make_grid

__all__ = ['plot_chartdoc']

# Panel kinds this renderer realizes natively today. 'surface' is the
# declared degradation (projection); anything else raises until its
# conversion lands.
_NATIVE = {'heatmap', 'xy'}
_DEGRADED = {'surface'}


# The ladder for an atomic series, in atoms and in pixels. Both are
# renderer constants: how much room an atom gets is a property of the
# figure, never of the document.
#: At or under this many atoms in view, each one is drawn: a stem to its
#: value with a marker on the end. Mirrors the ``mx <= 60`` rule the
#: aggregate compositor has always used, counted in visible atoms rather
#: than in loss units so a cropped window is judged on what it shows.
LOLLIPOP_ATOMS = 40

#: Down to this many pixels per atom the steps are visible and are drawn.
#: Under it, steps and a line are the same picture and steps cost three
#: times the vertices.
STEP_PIXELS = 3.0


def _axes_width_px(ax):
    """Approximate drawing width of ``ax`` in pixels.

    Read off the figure rather than from a rendered bounding box, so the
    choice does not depend on a draw having happened. Constrained layout
    shifts this a little; a threshold does not care.
    """
    fig = ax.figure
    return ax.get_position().width * fig.get_size_inches()[0] * fig.dpi


def _atom_room(ax, window, x):
    """``(atoms in view, pixels per atom)`` for an atomic series."""
    lo, hi = window
    seen = int(np.count_nonzero((x >= lo) & (x <= hi)))
    return seen, (_axes_width_px(ax) / seen if seen else float('inf'))


def _draw_atomic(ax, x, y, label, y_axis, window):
    """Draw an atomic series at the honest density for the room available.

    Three drawings of one truth, chosen by how much room each atom gets,
    never by what the series is called.

    A **mass** lives *at* its atom, so where the atoms are far enough
    apart each is drawn as a stem with a marker: the lollipop the
    aggregate compositor has always used for a small book. Closer
    together the steps carry it, read as a bar at each bucket, and the
    point of them is the sharp vertical jump where a line would draw a
    slope the law does not have. Sub-pixel, steps and a line are the same
    picture, so no lie is told by taking the cheaper one.

    A **cumulative** function is different in kind and skips the lollipop
    rung entirely: F and S take a value at every x, not only at the
    atoms, so the honest drawing is a right-continuous step that jumps at
    the atom, however few atoms there are.

    Which of the two it is comes off the **axis**, not the series role: a
    reinsurance series is called gross or ceded in both panels, and only
    the axis knows that one of them carries mass and the other carries
    accumulated probability.
    """
    seen, pixels = _atom_room(ax, window, x)
    unit = getattr(y_axis, 'unit', None)
    if unit == 'probability':
        ax.plot(x, y, label=label,
                drawstyle='steps-post' if pixels >= STEP_PIXELS else 'default')
        return
    if unit == 'density' and seen <= LOLLIPOP_ATOMS:
        # Markers first so the series takes the next prop-cycle color, then
        # the stems in that same color.
        marker, = ax.plot(x, y, ls='none', marker='o', ms=2.5, label=label)
        ax.vlines(x, 0.0, y, color=marker.get_color(), lw=0.8)
        return
    ax.plot(x, y, label=label,
            drawstyle='steps-mid' if pixels >= STEP_PIXELS else 'default')


def _typeset(doc, text):
    """The typeset form of ``text`` if the document carries one.

    matplotlib can render mathtext, so it consults ``ChartDoc.tex``; a
    renderer that cannot typeset skips this and draws the plain string,
    which is why the plain form is the one the schema requires.
    """
    return doc.tex.get(text, text) if text else text


def _house_ramp():
    """Sequential colormap, white to the first prop-cycle color.

    A joint density is one-directional, so the ramp is sequential (a
    diverging ramp would imply a midpoint that does not exist), mirroring
    the app's visualMap.
    """
    c0 = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    return mpl.colors.LinearSegmentedColormap.from_list(
        'aggregate_seq', ['#ffffff', c0])


def _render_grid_panel(ax, doc, panel, series_list, log_z):
    """Render one z-grid panel (surface projection or heatmap).

    The panel's one surface series draws as the mesh; any x/y series on the
    same panel are overlays, drawn over it in document order (the iso-total
    diagonals of a joint density). Overlays are neutral and thin: the mesh
    is the subject, and a heavy line over a color field hides it.
    """
    surf = next(s for s in series_list if s.surface is not None).surface
    x = np.asarray(surf.x, dtype=float)
    y = np.asarray(surf.y, dtype=float)
    z = np.asarray(surf.z, dtype=float)
    axes = {a.id: a for a in doc.axes}
    if log_z:
        # One decade under the smallest mass actually present (ignoring
        # float dust), the app's floor-not-holes rule: a zero cell sits on
        # the floor rather than punching a hole in the field.
        pos = z[z > LOG_FLOOR]
        floor = (10.0 ** np.floor(np.log10(pos.min()))
                 if pos.size else LOG_FLOOR)
        norm = mpl.colors.LogNorm(vmin=floor, vmax=max(z.max(), floor * 10),
                                  clip=True)
    else:
        norm = mpl.colors.Normalize(vmin=0.0, vmax=z.max() or 1.0)
    mesh = ax.pcolormesh(x, y, z, shading='nearest', cmap=_house_ramp(),
                         norm=norm)
    if np.count_nonzero(z > 0) > 1 and z.max() > z.min():
        ax.contour(x, y, z, colors='w', linewidths=0.5, alpha=0.6)
    zlabel = _typeset(doc, axes[panel.z_axis].label)
    ax.figure.colorbar(mesh, ax=ax, shrink=0.85,
                       label=f'log {zlabel}' if log_z else zlabel)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for s in series_list:
        if s.surface is not None:
            continue
        xs = np.array([np.nan if v is None else v for v in s.x], dtype=float)
        ys = np.array([np.nan if v is None else v for v in s.y], dtype=float)
        ax.plot(xs, ys, color='k', lw=0.35, alpha=0.5)
    # An overlay states a relationship, not an extent: a family of iso-total
    # diagonals reaching past the mesh must not widen the window the grid set.
    ax.set(xlim=xlim, ylim=ylim,
           xlabel=_typeset(doc, axes[panel.x_axis].label),
           ylabel=_typeset(doc, axes[panel.y_axis].label))
    if panel.aspect == 'equal':
        ax.set_aspect('equal')


def _apply_axis(ax, which, axis):
    """Realize one ChartAxis on a matplotlib axis ('x' or 'y').

    The suggested range is the emitter's answer to which slice of the grid
    is worth looking at, and a heavy tail makes that the difference between
    a readable chart and every visible mass in a sliver at the origin. So
    the initial view honors it, exactly as :class:`ChartAxis` documents;
    panning and zooming afterwards is the reader's business.

    It is the range of the *data*, not of the frame, so a linear axis is
    then inset by matplotlib's own margin, exactly as autoscale would inset
    it. That is not cosmetic: a curve legitimately sitting at the end of
    its range (a distortion is 0 or 1 over whole stretches of s) would
    otherwise be drawn along the frame, where it cannot be read. A log
    range arrives as whole decades and is drawn as whole decades, because
    a decade gridline is how that axis is read.
    """
    if axis.scale == 'log':
        getattr(ax, f'set_{which}scale')('log')
    if axis.suggested_range is not None:
        lo, hi = axis.suggested_range
        if axis.scale == 'log':
            getattr(ax, f'set_{which}lim')(lo, hi)
        else:
            margin = plt.rcParams[f'axes.{which}margin'] * (hi - lo)
            getattr(ax, f'set_{which}lim')(lo - margin, hi + margin)
        # The unit interval draws with pinned round ticks: the reference
        # gridlines of a probability square are part of how it is read.
        if (lo, hi) == (0.0, 1.0) and axis.scale == 'linear':
            getattr(ax, f'set_{which}ticks')(np.linspace(0, 1, 6))


def _panel_window(panel, axes, series_list):
    """The x range the panel will show, before anything is drawn.

    The emitter's suggestion where there is one, else the data's own
    extent. Computed off the document rather than off the axes so the
    atom count does not depend on the order things are plotted in.
    """
    window = axes[panel.x_axis].suggested_range
    if window is not None:
        return window
    xs = [v for s in series_list if s.x for v in s.x if v is not None]
    return (min(xs), max(xs)) if xs else (0.0, 1.0)


def _render_xy_panel(ax, doc, panel, series_list):
    """Render one 'xy' panel: role-styled curves, gaps broken, marks drawn."""
    axes = {a.id: a for a in doc.axes}
    window = _panel_window(panel, axes, series_list)
    labeled = False
    for s in series_list:
        x = np.array([np.nan if v is None else v for v in s.x], dtype=float)
        y = np.array([np.nan if v is None else v for v in s.y], dtype=float)
        if s.role == 'identity':
            # The reference diagonal: neutral, thin, never in the legend.
            ax.plot(x, y, color='k', lw=0.5, alpha=0.5)
            continue
        if s.y2 is not None:
            y2 = np.array([np.nan if v is None else v for v in s.y2],
                          dtype=float)
            ax.fill_between(x, y, y2, alpha=0.15, label=_typeset(doc, s.name))
            labeled = True
            continue
        label = _typeset(doc, s.name)
        if s.support == 'continuous':
            # Samples of a function that exists between them: a line is the
            # truthful drawing and no ladder applies.
            ax.plot(x, y, label=label)
        else:
            _draw_atomic(ax, x, y, label, axes[panel.y_axis], window)
        labeled = True
    for m in doc.marks:
        if m.panel_id != panel.id:
            continue
        line = ax.axvline if m.orient == 'v' else ax.axhline
        line(m.at, lw=0.75 if not m.faint else 0.5, color='C7', ls='--',
             alpha=0.45 if m.faint else 1.0)
    _apply_axis(ax, 'x', axes[panel.x_axis])
    _apply_axis(ax, 'y', axes[panel.y_axis])
    # The document labels its axes and the renderer draws what it is given,
    # as the grid panels already do.
    ax.set(xlabel=_typeset(doc, axes[panel.x_axis].label),
           ylabel=_typeset(doc, axes[panel.y_axis].label))
    if panel.aspect == 'equal':
        ax.set_aspect('equal')
    if labeled and sum(s.role != 'identity' for s in series_list) > 1:
        ax.legend(loc='upper left', fontsize='x-small')


def plot_chartdoc(doc, ax=None, strict=False, log_z=False):
    """Render a chart document with matplotlib.

    Parameters
    ----------
    doc : ChartDoc
        The document to realize. Panels are laid out in one row, in
        document order; panels naming the same x axis share it.
    ax : matplotlib Axes, optional
        Draw into an existing axes; default makes a figure via the house
        canvas helpers. Single-panel documents only, since a shared axis
        is a property of the figure and not of one axes.
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
    if not doc.panels:
        raise ChartCapabilityError(f'document {doc.name!r} has no panels')
    if ax is not None and len(doc.panels) > 1:
        raise ChartCapabilityError(
            f'document {doc.name!r} has {len(doc.panels)} panels and cannot '
            'be drawn into one supplied axes; leave ax=None')

    title = doc.title or doc.name
    if len(doc.panels) == 1:
        panel = doc.panels[0]
        if ax is None:
            if panel.kind == 'xy':
                # An equal-aspect single panel is a square figure (the unit
                # square reads at the small preset, as the compositor does).
                size = ((FIG_H, FIG_H) if panel.aspect == 'equal'
                        else (FIG_W, FIG_H))
            else:
                # Near-square: the z grid reads as a map, and the colorbar
                # takes the balance of the width.
                size = (1.25 * FIG_W, FIG_W)
            _, ax = make_grid(1, 1, figsize=size, squeeze=True)
        axs = [ax]
    else:
        # Panels naming the same x axis share it: that is what makes a
        # density and its tail one reading rather than two pictures, and
        # zooming one must move the other.
        shared = len({p.x_axis for p in doc.panels}) == 1
        _, grid = make_grid(1, len(doc.panels), squeeze=False, sharex=shared)
        axs = list(grid[0])

    for panel, panel_ax in zip(doc.panels, axs):
        series = [s for s in doc.series if s.panel_id == panel.id]
        panel_title = panel.title or title
        if panel.kind == 'xy':
            _render_xy_panel(panel_ax, doc, panel, series)
        else:
            _render_grid_panel(panel_ax, doc, panel, series,
                               bool(log_z) and bool(doc.meta.get('z_log_ok')))
            if panel.kind == 'surface':
                panel_title = f'{panel_title} (projection)'
        panel_ax.set_title(_typeset(doc, panel_title))
    fig = axs[0].figure
    if len(doc.panels) > 1 and doc.title:
        # Each panel already says what it is, so the document's title is the
        # heading over them rather than a repeat inside each one.
        fig.suptitle(_typeset(doc, doc.title))
    return fig
