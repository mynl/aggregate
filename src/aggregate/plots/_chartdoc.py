"""The one generic ChartDoc renderer: matplotlib realizes the chart IR.

Grows exactly what each converted chart needs (``dev/plan-chart-ir.md``,
pass three). matplotlib has no faithful 3-D surface, so a 'surface' panel
that offers no other realization renders as its honest 2-D reading, a
``pcolormesh`` projection of the z grid with a contour overlay, and the
title is stamped ``(projection)``; ``strict=True`` raises
:class:`~aggregate.charts.ir.ChartCapabilityError` instead. A panel that
declares ``kinds`` this renderer *can* draw is drawn that way instead, with
no confession to make. That is the capability-declaration pattern every
later kind follows: prefer a declared realization, degrade honestly and say
so, or refuse loudly, never approximate silently.

The four switches on :func:`plot_chartdoc` are the matplotlib side of the
document's declared readings, and they follow the same surfacing rule as
the app's control strip: a switch acts on **every** axis or panel that
declares the reading and is ignored everywhere else, so one call draws one
coherent picture rather than a per-axis patchwork. Which readings exist is
the document's business; which one is on screen is the caller's.

Renderer-side decisions only live here: the sequential ramp (white to the
house primary, one-directional like the density it colors), figure sizing,
the colorbar, and the floor a log view puts under a window whose declared
low end is zero. Everything semantic (grids, labels, the scales an axis may
be read on, the forms a panel may take) comes off the document.
"""

import numpy as np

from ..charts.ir import ChartCapabilityError
from ..constants import LOG_FLOOR
from ._quantile import MAX_RETURN_PERIOD
from ._style import plt, mpl, FIG_H, FIG_W, make_grid

__all__ = ['plot_chartdoc']

# Panel kinds this renderer realizes natively today. 'surface' is the
# declared degradation (projection); anything else raises until its
# conversion lands.
_NATIVE = {'heatmap', 'xy'}
_DEGRADED = {'surface'}


def _realization(panel, requested):
    """The kind this renderer will draw ``panel`` as.

    A panel declares the realizations it supports; this picks one. An
    explicit request wins and is refused by name if the panel does not
    offer it, because silently drawing something else is the failure the
    capability pattern exists to prevent. Otherwise the panel's default
    kind is taken when this renderer draws it natively, then any other
    declared kind it draws natively (which is how a joint density asked
    for in relief arrives as a heatmap here and a surface in the browser),
    and only failing both does it fall through to the default for the
    caller to accept or refuse.
    """
    if requested is not None:
        if requested not in panel.kinds:
            raise ChartCapabilityError(
                f'panel {panel.id!r} was asked for kind {requested!r}, which '
                f'it does not declare; it offers {panel.kinds}')
        return requested
    if panel.kind in _NATIVE:
        return panel.kind
    for k in panel.kinds:
        if k in _NATIVE:
            return k
    return panel.kind


def _decade_floor(values):
    """The decade at or under the smallest value worth drawing, or ``None``.

    A log view of a window whose declared low end is zero needs a bottom,
    and the honest one is a round decade under the smallest thing actually
    drawn: gridlines land on powers of ten, which is how a log axis is
    read, and nothing real is cropped.

    Values at or under ``LOG_FLOOR`` are arithmetic noise rather than a
    tail, the same rule the grid panels floor a color scale by and the
    emitters cut a survival curve at. Without it one dust value at 1e-17
    would open six empty decades under a panel whose mass all sits in the
    top three.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > LOG_FLOOR)]
    if not v.size:
        return None
    return float(10.0 ** np.floor(np.log10(v.min())))


def _axis_scale(axis, log):
    """The scale this axis is drawn on: its default, or its log reading.

    ``log`` acts on every axis that declares a log reading and on no other,
    which is the surfacing rule the app's control strip follows: one switch,
    applied wherever the document says a log reading is meaningful.
    """
    return 'log' if (log and 'log' in axis.scales) else axis.scale


def _axis_window(axis, full):
    """The window this axis is drawn in: its suggestion, or its full extent.

    An axis offers the zoom-out only by carrying ``full_range``; where it
    does not, ``full`` finds nothing to act on and the suggestion stands,
    because a chart whose window is its meaning has no other reading.
    """
    if full and axis.full_range is not None:
        return axis.full_range
    return axis.suggested_range


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


def _draw_atomic(ax, x, y, label, x_axis, y_axis, window):
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
    the atom, however few atoms there are. A **quantile** function is that
    same cumulative read the other way round, probability on x and outcome
    on y, and the sideways step is left-continuous: it is the jump of F
    seen from the other axis, so it rises *before* its atom where F steps
    after it.

    Which of the three it is comes off the **axes**, not the series role:
    a reinsurance series is called gross or ceded in every panel it
    appears in, and only the axes know that one panel carries mass,
    another accumulated probability, and a third the same probability
    read back to an outcome.
    """
    seen, pixels = _atom_room(ax, window, x)
    unit = getattr(y_axis, 'unit', None)
    if unit == 'probability':
        ax.plot(x, y, label=label,
                drawstyle='steps-post' if pixels >= STEP_PIXELS else 'default')
        return
    if getattr(x_axis, 'unit', None) in ('probability', 'return_period'):
        ax.plot(x, y, label=label,
                drawstyle='steps-pre' if pixels >= STEP_PIXELS else 'default')
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
    """The typeset form of ``text``, which the document always carries.

    matplotlib can render mathtext, so it consults ``ChartDoc.tex``; a
    renderer that cannot typeset skips this and draws the plain string,
    which is why the plain form is the one the schema requires.

    The lookup is total (``complete_tex`` in the IR builds it that way and
    a test holds every emitter to it), so the fallback here is a net under
    an emitter bug rather than a state the schema licenses: a renderer that
    silently drew nothing, or drew the key, would turn a missing entry into
    a mystery instead of a plain string.
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


def _render_grid_panel(ax, doc, panel, series_list, log=False):
    """Render one z-grid panel (surface projection or heatmap).

    The panel's one surface series draws as the mesh; any x/y series on the
    same panel are overlays, drawn over it in document order (the iso-total
    diagonals of a joint density). Overlays are neutral and thin: the mesh
    is the subject, and a heavy line over a color field hides it.

    ``log`` acts on the z axis when the document declares a log reading of
    it, which for a joint density is where tail dependence lives, and on
    the plane axes on the same terms.
    """
    surf = next(s for s in series_list if s.surface is not None).surface
    x = np.asarray(surf.x, dtype=float)
    y = np.asarray(surf.y, dtype=float)
    z = np.asarray(surf.z, dtype=float)
    axes = {a.id: a for a in doc.axes}
    log_z = _axis_scale(axes[panel.z_axis], log) == 'log'
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
        xs = np.array([np.nan if v is None else v for v in s.x_values],
                      dtype=float)
        ys = np.array([np.nan if v is None else v for v in s.y_values],
                      dtype=float)
        ax.plot(xs, ys, color='k', lw=0.35, alpha=0.5)
    # An overlay states a relationship, not an extent: a family of iso-total
    # diagonals reaching past the mesh must not widen the window the grid set.
    ax.set(xlim=xlim, ylim=ylim,
           xlabel=_typeset(doc, axes[panel.x_axis].label),
           ylabel=_typeset(doc, axes[panel.y_axis].label))
    for which, axis_id in (('x', panel.x_axis), ('y', panel.y_axis)):
        if _axis_scale(axes[axis_id], log) == 'log':
            getattr(ax, f'set_{which}scale')('log')
    if panel.aspect == 'equal':
        ax.set_aspect('equal')


def _apply_axis(ax, which, axis, scale, window, floor=None):
    """Realize one ChartAxis on a matplotlib axis ('x' or 'y').

    The window is the emitter's answer to which slice of the grid is worth
    looking at, and a heavy tail makes that the difference between a
    readable chart and every visible mass in a sliver at the origin. So the
    initial view honors it, exactly as :class:`ChartAxis` documents;
    panning and zooming afterwards is the reader's business.

    It is the range of the *data*, not of the frame, so a linear axis is
    then inset by matplotlib's own margin, exactly as autoscale would inset
    it. That is not cosmetic: a curve legitimately sitting at the end of
    its range (a distortion is 0 or 1 over whole stretches of s) would
    otherwise be drawn along the frame, where it cannot be read. A log
    range arrives as whole decades and is drawn as whole decades, because
    a decade gridline is how that axis is read.

    A window read on log needs a positive bottom, and the low end of a loss
    or density window is routinely an exact zero. ``floor`` supplies the
    decade under the smallest positive value drawn: the emitter cannot
    compute it, since it does not know the reader will ask for log, and
    inventing a fixed epsilon here would crop or pad by orders of magnitude
    depending on the book.
    """
    if scale == 'log':
        getattr(ax, f'set_{which}scale')('log')
    if window is not None:
        lo, hi = window
        if scale == 'log':
            if lo <= 0:
                if floor is None:
                    return
                lo = floor
            getattr(ax, f'set_{which}lim')(lo, hi)
        else:
            margin = plt.rcParams[f'axes.{which}margin'] * (hi - lo)
            getattr(ax, f'set_{which}lim')(lo - margin, hi + margin)
        # The unit interval draws with pinned round ticks: the reference
        # gridlines of a probability square are part of how it is read.
        if (lo, hi) == (0.0, 1.0) and scale == 'linear':
            getattr(ax, f'set_{which}ticks')(np.linspace(0, 1, 6))


#: Where on the house ramp a value-carrying family starts. The ramp runs
#: from white, and a curve drawn in white is not drawn at all, so the family
#: uses the visible part of it and the lightest member still reads.
VALUE_RAMP_FLOOR = 0.3


def _value_norm(series_list):
    """Normalizer over the ``value`` a family of series carries.

    Normalized over what the panel actually holds rather than over a fixed
    range, because ``ChartSeries.value`` is any quantity the emitter says is
    a fact about the series, and only the emitter knows its scale.
    """
    seen = [s.value for s in series_list if s.value is not None]
    lo, hi = (min(seen), max(seen)) if seen else (0.0, 1.0)
    return mpl.colors.Normalize(lo, hi if hi > lo else lo + 1.0)


def _value_color(series_list, value):
    """The ramp color for one member of a value-carrying family."""
    at = _value_norm(series_list)(value)
    return _house_ramp()(VALUE_RAMP_FLOOR + (1 - VALUE_RAMP_FLOOR) * at)


def _paired_reading(doc, axis_id):
    """The paired return-period axis for ``axis_id``, if the document has one."""
    for a in doc.axes:
        if a.reciprocal_of == axis_id:
            return a
    return None


def _return_periods(values, how):
    """Return periods from the probabilities on a paired axis.

    ``T = 1 / v`` under the 'reciprocal' map, ``T = 1 / (1 - v)`` under
    'complement' (see :data:`~aggregate.charts.ir.RETURN_PERIOD_MAPS`).
    The quantile function saturates at its far end, where ``T`` diverges,
    so a non-finite or non-positive result becomes a gap: the curve stops
    where the grid stops knowing, rather than running out to an invented
    bound.
    """
    v = np.asarray(values, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = 1.0 / (1.0 - v) if how == 'complement' else 1.0 / v
    return np.where(np.isfinite(t) & (t > 0.0), t, np.nan)


def _legend_corner(drawn, window):
    """The upper corner with less curve under it: where a key hides nothing.

    That a legend exists is realization, but *where* it sits is not
    arbitrary, and it is not a per-chart setting either. A density family
    piles up on the left of its window and leaves the upper right free; a
    monotone family (a distortion, a cumulative) rises to the right and
    leaves the upper left free. So the corner is read off the drawn values,
    the taller half taking the legend away from itself, which is one rule
    for every chart and needs no instruction in the document.
    """
    mid = 0.5 * (window[0] + window[1])
    heights = [0.0, 0.0]
    for x, y in drawn:
        for half, keep in enumerate((x <= mid, x > mid)):
            seen = y[keep & np.isfinite(y)]
            if seen.size:
                heights[half] = max(heights[half], float(seen.max()))
    return 'upper right' if heights[0] >= heights[1] else 'upper left'


def _square_window(x_window, y_window, all_x, all_y):
    """One window for both axes of an equal-aspect panel.

    Equal aspect with two different ranges is a square box drawn over a
    rectangle of data, and it misreads: on a panel whose axes measure the
    same thing, a 45 degree line has to *be* at 45 degrees. So the two
    windows become one.

    The top is the higher of the two, so nothing an emitter declared is
    cropped. The bottom is where the data actually starts, because a panel
    whose curves begin well inside its window opens with an empty corner
    otherwise, and a loss window anchored at zero routinely does. It never
    widens past what the emitter asked for: the data can raise the floor,
    never lower it.
    """
    windows = [w for w in (x_window, y_window) if w is not None]
    if not windows:
        return None
    seen = [v[np.isfinite(v)] for v in (all_x, all_y) if v.size]
    seen = np.concatenate(seen) if seen else np.array([])
    lo = min(w[0] for w in windows)
    if seen.size:
        lo = max(lo, float(seen.min()))
    hi = max(w[1] for w in windows)
    return (lo, hi) if hi > lo else windows[0]


def _panel_window(window, values):
    """The range the panel will show, before anything is drawn.

    The emitter's window where there is one, else the data's own extent.
    Computed off the document rather than off the axes so the atom count
    does not depend on the order things are plotted in.
    """
    if window is not None:
        return window
    finite = values[np.isfinite(values)] if values.size else values
    return ((float(finite.min()), float(finite.max())) if finite.size
            else (0.0, 1.0))


def _render_xy_panel(ax, doc, panel, series_list, log=False, full=False,
                     return_period=False, invert=False):
    """Render one 'xy' panel: role-styled curves, gaps broken, marks drawn.

    ``return_period`` swaps a drawn probability axis for the paired reading
    the document offers on it, which is a change of coordinates and not of
    data: the same curve, interrogated at 1 in 200 rather than at 0.995.
    Panels the document offers no pairing for are untouched.

    ``invert`` exchanges the two axes of a panel that declares itself
    invertible, drawing the same pairs the other way round: a quantile
    function becomes the distribution function it inverts. Everything that
    follows reads the axes rather than the panel, so the exchange is the
    only thing that has to happen: the ladder picks a right-continuous step
    where it was picking a left-continuous one, the window and the labels
    follow their axes, and a paired return-period reading rides along on
    whichever axis it was attached to.
    """
    axes = {a.id: a for a in doc.axes}
    x_axis, y_axis = axes[panel.x_axis], axes[panel.y_axis]
    x_map = y_map = None
    if return_period:
        how = doc.meta.get('return_period_map', 'reciprocal')
        pair = _paired_reading(doc, panel.x_axis)
        if pair is not None:
            x_axis, x_map = pair, how
        pair = _paired_reading(doc, panel.y_axis)
        if pair is not None:
            y_axis, y_map = pair, how
    inverted = bool(invert) and panel.invertible
    if inverted:
        x_axis, y_axis = y_axis, x_axis
        x_map, y_map = y_map, x_map

    def coords(values, mapping):
        out = np.array([np.nan if v is None else v for v in values],
                       dtype=float)
        return _return_periods(out, mapping) if mapping else out

    drawn = []
    for s in series_list:
        # The document's x and y, mapped; the exchange happens after, so a
        # band's second edge stays with the coordinate it is an edge of.
        px = coords(s.x_values, y_map if inverted else x_map)
        py = coords(s.y_values, x_map if inverted else y_map)
        p2 = None if s.y2 is None else coords(s.y2,
                                              x_map if inverted else y_map)
        drawn.append((s, py, px, p2) if inverted else (s, px, py, p2))
    all_x = np.concatenate([x for _, x, _, _ in drawn]) if drawn else np.array([])
    all_y = np.concatenate([y for _, _, y, _ in drawn]) if drawn else np.array([])
    x_window = _panel_window(_axis_window(x_axis, full), all_x)
    if x_map and _axis_window(x_axis, full) is None:
        # Nothing declared a window for the return-period reading, and its
        # top is the saturating end of the quantile function, so the house
        # cap stands in: a billion-year event is past anyone's question.
        x_window = (x_window[0], min(x_window[1], MAX_RETURN_PERIOD))
    labeled = False
    for s, x, y, y2 in drawn:
        if s.role == 'identity':
            # The reference diagonal: neutral, thin, never in the legend.
            ax.plot(x, y, color='k', lw=0.5, alpha=0.5)
            continue
        if y2 is not None:
            # A band is the region between two edges of one coordinate, so
            # exchanged axes fill between them horizontally. Both edges are
            # stroked: a band whose boundary cannot be seen reads as vaguer
            # than the data, and each edge is a curve in its own right.
            band = _typeset(doc, s.name)
            if inverted:
                fill = ax.fill_betweenx(y, x, y2, alpha=0.15, label=band)
                edges = [(x, y), (y2, y)]
            else:
                fill = ax.fill_between(x, y, y2, alpha=0.15, label=band)
                edges = [(x, y), (x, y2)]
            edge_color = fill.get_facecolor()[0][:3]
            for ex, ey in edges:
                ax.plot(ex, ey, lw=0.6, color=edge_color)
            labeled = True
            continue
        if s.value is not None:
            # One of a family, labeled by a number rather than by a name:
            # the number is the reading, so it is encoded on the ramp and
            # the curve stays out of the legend. Forty legend entries would
            # be forty names nobody asked for.
            ax.plot(x, y, lw=0.75, alpha=0.55,
                    color=_value_color(series_list, s.value))
            continue
        label = _typeset(doc, s.name)
        if s.support == 'continuous':
            # Samples of a function that exists between them: a line is the
            # truthful drawing and no ladder applies.
            ax.plot(x, y, label=label)
        else:
            _draw_atomic(ax, x, y, label, x_axis, y_axis, x_window)
        labeled = True
    for m in doc.marks:
        if m.panel_id != panel.id:
            continue
        # A mark names the axis it sits on, so exchanged axes exchange it too.
        orient = m.orient if not inverted else ('h' if m.orient == 'v' else 'v')
        at, mapping = m.at, (x_map if orient == 'v' else y_map)
        if mapping:
            at = float(_return_periods(np.array([at]), mapping)[0])
            if not np.isfinite(at):
                continue
        line = ax.axvline if orient == 'v' else ax.axhline
        line(at, lw=0.75 if not m.faint else 0.5, color='C7', ls='--',
             alpha=0.45 if m.faint else 1.0)
    # A paired reading re-slices the panel: the deep tail a return-period
    # axis exists to show sits far outside the window computed for the
    # probability reading, so the companion axis follows the data instead.
    # The compositor's quantile worker does the same by relim-and-autoscale.
    x_scale, y_scale = _axis_scale(x_axis, log), _axis_scale(y_axis, log)
    y_window = None if x_map else _axis_window(y_axis, full)
    x_only = None if y_map else x_window
    if panel.aspect == 'equal' and x_scale == y_scale:
        x_only = y_window = _square_window(x_only, y_window, all_x, all_y)
    _apply_axis(ax, 'x', x_axis, x_scale, x_only, _decade_floor(all_x))
    _apply_axis(ax, 'y', y_axis, y_scale, y_window, _decade_floor(all_y))
    # The document labels its axes and the renderer draws what it is given,
    # as the grid panels already do.
    ax.set(xlabel=_typeset(doc, x_axis.label),
           ylabel=_typeset(doc, y_axis.label))
    if panel.aspect == 'equal':
        ax.set_aspect('equal')
    if labeled and sum(s.role != 'identity' for s in series_list) > 1:
        corner = _legend_corner([(x, y) for s, x, y, _ in drawn
                                 if s.role != 'identity'], x_window)
        ax.legend(loc=corner, fontsize='xx-small')
    value_label = doc.meta.get('value_label')
    if value_label and any(s.value is not None for s in series_list):
        # A family shaded by a number needs the number named, and only the
        # document can name it.
        ramp = mpl.colors.LinearSegmentedColormap.from_list(
            'aggregate_value',
            [_value_color(series_list, v) for v in
             np.linspace(*_value_norm(series_list).inverse((0., 1.)), 16)])
        ax.figure.colorbar(
            mpl.cm.ScalarMappable(_value_norm(series_list), ramp),
            ax=ax, shrink=0.7, aspect=18, label=_typeset(doc, value_label))


def plot_chartdoc(doc, ax=None, strict=False, log=False, full_range=False,
                  return_period=False, invert=False, kind=None):
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
        Raise :class:`ChartCapabilityError` for any panel this renderer
        can only degrade, instead of drawing the declared degradation.
    log : bool
        Read every axis that declares a log scale on log. An axis that
        declares one reading is untouched, so a document with nothing to
        say about log draws identically either way.
    full_range : bool
        Read every axis that carries a ``full_range`` at its full extent
        instead of at the window it suggests. Axes carrying only a
        suggestion are untouched.
    return_period : bool
        Draw a probability axis the document pairs with a return-period
        reading as that reading (:attr:`ChartAxis.reciprocal_of`), which
        spreads the rare tail so it can be read off directly. The
        transform comes from ``meta['return_period_map']``.
    invert : bool
        Exchange the two axes of every panel that declares itself
        invertible, which draws the same pairs the other way round: a Lee
        diagram becomes the distribution function it inverts. Panels that
        do not declare it are untouched.
    kind : str, optional
        Realize every panel that declares this kind as this kind, for a
        panel offering more than one (a joint density as 'heatmap' rather
        than 'surface'). ``None`` lets the renderer pick, preferring the
        panel's own default and then any declared kind it draws natively.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn into.

    Raises
    ------
    ChartCapabilityError
        For a requested kind a panel does not declare, under ``strict``
        for degraded kinds, and always for kinds with no realization here
        yet.

    Notes
    -----
    The four reading switches act on **every** axis or panel that declares
    the reading and on no other, which is the same surfacing rule the app
    applies to its control strip: declaring a reading on an axis asserts
    both that it is meaningful there and that it is reasonable for it to
    move when its siblings do. A document that declares nothing draws its
    one reading whatever is asked for, so a caller never has to know which
    chart it holds.
    """
    if not doc.panels:
        raise ChartCapabilityError(f'document {doc.name!r} has no panels')
    realized = [_realization(panel, kind) for panel in doc.panels]
    for panel, realization in zip(doc.panels, realized):
        if realization in _NATIVE:
            continue
        if realization in _DEGRADED:
            if strict:
                raise ChartCapabilityError(
                    f"panel {panel.id!r} kind {realization!r}: matplotlib "
                    'has no faithful 3-D surface; non-strict renders the '
                    '2-D projection')
            continue
        raise ChartCapabilityError(
            f'panel {panel.id!r} kind {realization!r} is not yet realized '
            'by the matplotlib renderer')
    if ax is not None and len(doc.panels) > 1:
        raise ChartCapabilityError(
            f'document {doc.name!r} has {len(doc.panels)} panels and cannot '
            'be drawn into one supplied axes; leave ax=None')

    title = doc.title or doc.name
    if len(doc.panels) == 1:
        panel = doc.panels[0]
        if ax is None:
            if realized[0] == 'xy':
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
        #
        # An equal-aspect panel cannot join in. Its window is settled by its
        # own squareness, so sharing lets it drag its neighbour's window
        # around to keep itself square, which is the tail wagging the dog:
        # the neighbour's window was computed from the data it draws. Equal
        # aspect is the stronger statement, so it wins and the axis is not
        # shared.
        square = any(p.aspect == 'equal' for p in doc.panels)
        shared = len({p.x_axis for p in doc.panels}) == 1 and not square
        # Equal-aspect panels are squares, and squares in a row need a
        # canvas that is as many squares wide, or constrained layout
        # collapses them to slivers trying to honor the aspect.
        all_square = all(p.aspect == 'equal' for p in doc.panels)
        size = (len(doc.panels) * FIG_H, FIG_H) if all_square else None
        _, grid = make_grid(1, len(doc.panels), squeeze=False, sharex=shared,
                            **({} if size is None else {'figsize': size}))
        axs = list(grid[0])

    for panel, realization, panel_ax in zip(doc.panels, realized, axs):
        series = [s for s in doc.series if s.panel_id == panel.id]
        panel_title = panel.title or title
        if invert and panel.invertible:
            # Exchanged axes draw a different picture, and the document
            # names it; with no name to use, say so rather than invent one.
            panel_title = (panel.inverse_title
                           or f'{panel_title}, inverted')
        if realization == 'xy':
            _render_xy_panel(panel_ax, doc, panel, series, log=log,
                             full=full_range, return_period=return_period,
                             invert=invert)
        else:
            _render_grid_panel(panel_ax, doc, panel, series, log=log)
            if realization in _DEGRADED:
                panel_title = f'{panel_title} (projection)'
        panel_ax.set_title(_typeset(doc, panel_title))
    fig = axs[0].figure
    if len(doc.panels) > 1 and doc.title:
        # Each panel already says what it is, so the document's title is the
        # heading over them rather than a repeat inside each one.
        fig.suptitle(_typeset(doc, doc.title))
    return fig
