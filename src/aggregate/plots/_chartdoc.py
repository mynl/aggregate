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

A 'tower' panel is drawn natively: labeled rectangles over a quantity axis,
a placement axis with no ticks because width is share, and the panel's
horizontal marks promoted to the quantity ticks, since a tower is read at
its breaks rather than on a continuous scale. A row holding one gets width
ratios, a tower being a strip and a curve beside it wanting room, and a
taller canvas than the house landscape cell.

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
from ._style import plt, mpl, FIG_H, FIG_W, FONT_SIZE, make_grid

__all__ = ['plot_chartdoc']

# Panel kinds this renderer realizes natively today. 'surface' is the
# declared degradation (projection); anything else raises until its
# conversion lands.
_NATIVE = {'heatmap', 'xy', 'tower', 'matrix'}
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
    """The decade strictly under the smallest value worth drawing, or ``None``.

    A log view of a window whose declared low end is zero needs a bottom,
    and the honest one is a round decade under the smallest thing actually
    drawn: gridlines land on powers of ten, which is how a log axis is
    read, and nothing real is cropped.

    Values at or under ``LOG_FLOOR`` are arithmetic noise rather than a
    tail, the same rule the grid panels floor a color scale by and the
    emitters cut a survival curve at. Without it one dust value at 1e-17
    would open six empty decades under a panel whose mass all sits in the
    top three.

    Notes
    -----
    **Strictly under, which matters when the smallest value is itself a
    round decade.** A tower panel holding one gross slab from 0 to 100 has
    exactly one positive coordinate, and a floor *at* 100 leaves the panel
    nothing to draw in: the rectangle collapses to the top of the frame.
    Rounding the exponent up and stepping one decade down gives 10 there
    and is unchanged wherever the smallest value is not an exact power of
    ten, which is every continuous series.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > LOG_FLOOR)]
    if not v.size:
        return None
    return float(10.0 ** (np.ceil(np.log10(v.min())) - 1.0))


def _tower_floors(doc, realized):
    """``axis_id -> decade floor`` for every quantity axis carrying a tower.

    Parameters
    ----------
    doc : ChartDoc
    realized : list of str
        The kind each panel is being drawn as, panel for panel.

    Returns
    -------
    dict
        Empty when the document draws no tower.

    Notes
    -----
    **The floor is a property of the axis, not of the panel.** A structure
    document puts the gross slab, the tower carving it up and the quantile
    curve beside it on one loss axis, and they are one reading: a boundary
    is meant to carry across. Computed per panel the three disagree, since
    each sees only its own content, and the panels are then drawn with
    different bottoms and different clamps, so a slab floats above the
    frame its neighbour fills. Pooling per axis is what makes them line up,
    and it is also what makes the clamp agree with the window matplotlib
    ends up with under ``sharey``, which takes the last panel's limits for
    all of them.

    **Pooled over the blocks, not over everything drawn against the axis.**
    A Lee curve on a loss axis runs down to the first positive grid point,
    three or four decades under the program, and a floor taken from that
    would open those decades under every tower and squeeze the bands back
    into slivers, which is the pathology the log reading exists to cure.
    The tower's breaks are the scale the picture is read at; the curve
    beside it simply runs off the bottom of the frame, as a curve on a log
    loss axis always does.
    """
    towers = {p.y_axis for p, r in zip(doc.panels, realized) if r == 'tower'}
    if not towers:
        return {}
    panels = {p.id: p for p in doc.panels}
    values = {}
    for block in doc.blocks:
        panel = panels.get(block.panel_id)
        if panel is not None and panel.y_axis in towers:
            values.setdefault(panel.y_axis, []).extend((block.y0, block.y1))
    return {axis_id: _decade_floor(vs) for axis_id, vs in values.items()}


def _axis_scale(axis, log):
    """The scale this axis is drawn on: its default, or its log reading.

    ``log`` here is the resolved flag for the one direction this axis is
    drawn in (see :func:`_log_directions`). It acts on an axis that
    declares a log reading and on no other, which is the surfacing rule
    the app's control strip follows: a switch applies wherever the
    document says a log reading is meaningful, and nowhere else.
    """
    return 'log' if (log and 'log' in axis.scales) else axis.scale


def _log_directions(log):
    """Resolve the ``log`` switch to its ``(x, y)`` direction pair.

    ``True`` asks for the log reading in both drawing directions, ``'x'``
    or ``'y'`` in that one, ``False`` in neither. The declaration rule is
    unchanged: a direction's flag still acts only on an axis that declares
    a log reading, so ``log='y'`` on a document whose ordinate declares
    none draws identically to the plain call.

    Notes
    -----
    Mirrors the app renderer's per-panel ``logX`` / ``logY`` controls,
    which exist because one log button could not draw a log ordinate over
    a linear loss axis, the reading wanted most often. The directions are
    the *drawn* directions: under ``invert`` the exchanged axes carry
    their flags with the direction they are drawn in, not the one the
    panel declared them under.
    """
    if log in (True, 'xy', 'yx'):
        return True, True
    if log in (False, None):
        return False, False
    if log == 'x':
        return True, False
    if log == 'y':
        return False, True
    raise ValueError(f"log must be a bool, 'x', 'y' or 'xy', got {log!r}")


def _surface_window(surf, x, y, z):
    """The part of a surface mesh its document calls the subject.

    A surface carries the whole reduced lattice and names a drawing range
    inside it through ``window`` (see :class:`~aggregate.charts.ir.SurfaceData`),
    so a heavy tail is served without being drawn. Two things here read it,
    and both matter:

    the **limits**, because a Lomax on a lattice wide enough to hold its
    tail is mostly empty, and drawing it at full width puts every visible
    mass in a sliver at the origin, which is the picture the window exists
    to prevent;

    and the **color normalization**, which is the less obvious half. The log
    floor is one decade under the smallest mass present, so taken over the
    whole mesh it is set by the far tail: five to six decades under the
    field the reader is looking at on the reference surfaces, which
    compresses that field into the top of the ramp and flattens exactly the
    structure the log reading exists to show.

    Returns
    -------
    (xlim, ylim, z_window) : tuple, tuple, ndarray
        Each limit is ``None`` where the document declares no window on that
        axis, which leaves the caller's own extent in force, and
        ``z_window`` is then ``z`` entire.
    """
    window = getattr(surf, 'window', None) or {}
    found = []
    for coords, key in ((x, 'x'), (y, 'y')):
        box = window.get(key)
        if box is None or len(box) < 2:
            found.append((None, None))
            continue
        lo, hi = float(min(box)), float(max(box))
        inside = (coords >= lo) & (coords <= hi)
        # A window naming nothing on the mesh is a document to draw whole
        # rather than an empty picture to draw.
        found.append(((lo, hi), inside) if inside.any() else (None, None))
    (xlim, in_x), (ylim, in_y) = found
    if in_y is not None:
        z = z[in_y, :]
    if in_x is not None:
        z = z[:, in_x]
    return xlim, ylim, z


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

# The tower panel's renderer-side decisions. What a block means is in the
# document (its role); how wide the strip is and how a role is shaded are
# not, and belong here.
#: Relative width of a tower panel against a curve panel beside it. A tower
#: is a strip, not a plot: its abscissa carries one unit of share and no
#: reading, so width past what the labels need is width taken from the
#: curve that does have a reading. 1:2 is the prior art's ratio.
TOWER_WIDTH = 1.0
CURVE_WIDTH = 2.0

#: Fill per block role: the placed layers in the house primary, everything
#: the cedent is left holding in grey, and an uncovered band as hatching
#: over nothing, because a gap is an absence and must not read as a thing
#: that was bought. Alternating alpha down the tower separates one layer
#: from the next without a rainbow, which is how a market slide draws it.
BLOCK_FILL = {
    'layer': ('C0', (0.80, 0.55)),
    'co_participation': ('C7', (0.16,)),
    'retention': ('C7', (0.30,)),
    'gap': ('none', (1.0,)),
    'gross': ('C1', (0.40,)),
}

#: Hatching per block role, for the two that are not solid things.
BLOCK_HATCH = {'co_participation': '///', 'gap': 'xx'}

#: Height of a tower figure as a multiple of the house panel height. A
#: tower is read up the page and each block's annotation stack needs rows,
#: so the default landscape cell is the wrong shape for it.
TOWER_HEIGHT = 1.8

#: Point size of a block's label text, a fraction of the house font. A
#: named number rather than matplotlib's 'xx-small', because the fits or
#: does not fit arithmetic below has to agree with what is drawn, and a
#: relative keyword leaves the renderer guessing what it asked for.
LABEL_POINTS = 0.65 * FONT_SIZE

#: Width of one character at the label font, as a fraction of the point
#: size. A crude average over a proportional face, which is all a fits or
#: does not fit test needs.
CHAR_WIDTH = 0.55

#: A label row's height as a multiple of the font size. A block shows as
#: many rows as fit at this pitch and no more, headline first: a stack
#: spilling over its own rectangle is worse than a stack cut short, because
#: the reader cannot tell which block the overflow belongs to.
ROW_PITCH = 1.5

#: Largest return period drawn when a document declares no window for its
#: paired reading. The quantile function saturates as its probability
#: reaches the end of the grid, where T diverges; a billion-year event is
#: past anyone's question and keeps both axes finite. Inherited from the
#: quantile worker this renderer replaced.
MAX_RETURN_PERIOD = 1e9


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
    the plane axes on the same terms. The z axis rides the ``y``
    direction of a directional ``log`` (the app's rule: the color field
    is an ordinate), so ``log='y'`` reads the mass on log over linear
    plane axes.
    """
    surf = next(s for s in series_list if s.surface is not None).surface
    x = np.asarray(surf.x, dtype=float)
    y = np.asarray(surf.y, dtype=float)
    z = np.asarray(surf.z, dtype=float)
    axes = {a.id: a for a in doc.axes}
    log_x, log_y = _log_directions(log)
    log_z = _axis_scale(axes[panel.z_axis], log_y) == 'log'
    # The mesh is drawn whole and the color scale is read off the window:
    # what is served beyond it is there to be panned to and to compute on,
    # not to set the scale for the part that is the subject.
    win_x, win_y, zw = _surface_window(surf, x, y, z)
    if log_z:
        # One decade under the smallest mass actually present (ignoring
        # float dust), the app's floor-not-holes rule: a zero cell sits on
        # the floor rather than punching a hole in the field.
        pos = zw[zw > LOG_FLOOR]
        floor = (10.0 ** np.floor(np.log10(pos.min()))
                 if pos.size else LOG_FLOOR)
        norm = mpl.colors.LogNorm(vmin=floor, vmax=max(zw.max(), floor * 10),
                                  clip=True)
    else:
        norm = mpl.colors.Normalize(vmin=0.0, vmax=zw.max() or 1.0)
    mesh = ax.pcolormesh(x, y, z, shading='nearest', cmap=_house_ramp(),
                         norm=norm)
    if np.count_nonzero(z > 0) > 1 and z.max() > z.min():
        ax.contour(x, y, z, colors='w', linewidths=0.5, alpha=0.6)
    zlabel = _typeset(doc, axes[panel.z_axis].label)
    ax.figure.colorbar(mesh, ax=ax, shrink=0.85,
                       label=f'log {zlabel}' if log_z else zlabel)
    xlim = win_x if win_x is not None else ax.get_xlim()
    ylim = win_y if win_y is not None else ax.get_ylim()
    for s in series_list:
        if s.surface is not None:
            continue
        xs = np.array([np.nan if v is None else v for v in s.x_values],
                      dtype=float)
        ys = np.array([np.nan if v is None else v for v in s.y_values],
                      dtype=float)
        ax.plot(xs, ys, color='k', lw=0.35, alpha=0.5)
    # An overlay states a relationship, not an extent: a family of iso-total
    # diagonals reaching past the mesh must not widen the window the document
    # set, and neither must the served tail beyond it. Panning out afterward
    # is the reader's business, and the mass is there to be found.
    ax.set(xlim=xlim, ylim=ylim,
           xlabel=_typeset(doc, axes[panel.x_axis].label),
           ylabel=_typeset(doc, axes[panel.y_axis].label))
    for which, axis_id, flag in (('x', panel.x_axis, log_x),
                                 ('y', panel.y_axis, log_y)):
        if _axis_scale(axes[axis_id], flag) == 'log':
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


def _paired_reading(doc, axis_id, attr='reciprocal_of'):
    """The paired axis declaring ``attr`` of ``axis_id``, if there is one.

    Two pointers name a paired reading, ``reciprocal_of`` (the return
    period) and ``complement_of`` (the reflection), and both are read the
    same way: an undrawn axis in ``doc.axes`` naming the drawn one it is an
    alternative reading of. The default keeps the older caller reading as
    it did.
    """
    for a in doc.axes:
        if getattr(a, attr) == axis_id:
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


def _reading_map(reflected, period):
    """The coordinate map for one axis, or ``None`` where it is the identity.

    Both paired readings are changes of coordinate on the same curve, so
    they compose into one callable applied wherever the document's numbers
    are read: the reflection first, then the return period, which is the
    order the two declarations are in. ``None`` for an axis asked for
    neither, so the common case allocates nothing and the callers that
    branch on "is there a map" keep reading as they did.
    """
    if not reflected and period is None:
        return None

    def apply(values):
        v = np.asarray(values, dtype=float)
        if reflected:
            v = 1.0 - v
        return v if period is None else _return_periods(v, period)

    return apply


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


def _axes_height_px(ax):
    """Approximate drawing height of ``ax`` in pixels.

    The companion of :func:`_axes_width_px`, read off the figure for the
    same reason: a label that fits must be decided before anything is
    drawn, and constrained layout only shifts the answer a little.
    """
    fig = ax.figure
    return ax.get_position().height * fig.get_size_inches()[1] * fig.dpi


def _block_reading(block, window, scale, floor):
    """``(height, span, mid)`` for one block in the coordinate it is drawn in.

    Parameters
    ----------
    block : TowerBlock
    window : tuple of float
        The drawn window on the quantity axis, before the floor is applied.
    scale : str
        'linear' or 'log', the reading the panel is drawn on.
    floor : float or None
        The decade floor from :func:`_decade_floor`, which is what a log
        axis puts in place of a window whose low end is an exact zero.

    Returns
    -------
    tuple of float
        The block's extent, the window's extent, and the block's center,
        all in the drawn coordinate. On log the first two are measured in
        decades and the third is the geometric center.

    Notes
    -----
    **A label is placed and sized in the coordinate the reader sees, not in
    loss units.** On a 1 to 100 log ordinate a ``15 xs 5`` band fills about
    43% of the drawn height while spanning 15% of the range, so measuring
    it linearly would drop labels it has room for and, on the retention
    above it, spill text across a neighbour. That matters most here because
    the log reading exists for exactly the program whose bands are slivers
    read linearly.

    A block sitting at or below the floor has no drawn height, so it
    answers zero and keeps its label rather than taking it somewhere the
    rectangle is not.
    """
    lo, hi = window
    linear = (block.y1 - block.y0, hi - lo, 0.5 * (block.y0 + block.y1))
    if scale != 'log' or floor is None:
        return linear
    base = max(lo, floor)
    y0, y1 = max(block.y0, base), max(block.y1, base)
    if not (y1 > y0 and hi > base):
        return 0.0, 1.0, y1
    log0, log1 = np.log10(y0), np.log10(y1)
    return log1 - log0, np.log10(hi) - np.log10(base), 10.0 ** (0.5 * (log0 + log1))


def _rows_that_fit(ax, height, span):
    """How many label rows a block of data height ``height`` has room for.

    Parameters
    ----------
    ax : matplotlib Axes
    height : float
        The block's extent on the quantity axis.
    span : float
        The drawn window's extent on that axis.

    Returns
    -------
    int
        Zero when the block is too thin for even its headline.
    """
    if not span > 0:
        return 0
    pixels = _axes_height_px(ax) * abs(height) / span
    row = ROW_PITCH * LABEL_POINTS * ax.figure.dpi / 72.0
    return int(pixels // row) if row > 0 else 0


def _draw_block(ax, block, shade, bottom=None):
    """Fill one block and draw the edges it is entitled to.

    ``shade`` picks among the role's alphas, counted over the blocks of that
    role rather than over all of them, so consecutive layers alternate and
    a retention between two layers does not break the alternation.

    An ``open_top`` block is drawn with its left, bottom and right edges
    and **no** top: the band continues past the frame, and a closed
    rectangle would assert a limit the contract has not got. The fill is
    laid down with no edge of its own so the three sides can be drawn
    deliberately.

    ``bottom`` is the lowest drawable coordinate, which a log reading needs
    and a linear one does not. A tower's two commonest bands start at an
    exact zero, the retention below the first attachment and the gross slab
    itself, and zero has no position on a log axis: sent there unclamped
    the rectangle's foot goes to negative infinity and the fill vanishes.
    Clamping to the panel's decade floor draws the band from the bottom of
    the frame, which is what the floor is for.
    """
    y0, y1 = block.y0, block.y1
    if bottom is not None:
        y0, y1 = max(y0, bottom), max(y1, bottom)
    color, alphas = BLOCK_FILL.get(block.role, ('C7', (0.3,)))
    alpha = alphas[shade % len(alphas)]
    ax.fill_betweenx([y0, y1], block.x0, block.x1,
                     facecolor=color, alpha=alpha, linewidth=0,
                     hatch=BLOCK_HATCH.get(block.role),
                     edgecolor='C7' if block.role in BLOCK_HATCH else 'none')
    edge = dict(color='C7', lw=0.6)
    ax.plot([block.x0, block.x0], [y0, y1], **edge)
    ax.plot([block.x1, block.x1], [y0, y1], **edge)
    ax.plot([block.x0, block.x1], [y0, y0], **edge)
    if not block.open_top:
        ax.plot([block.x0, block.x1], [y1, y1], **edge)


def _label_block(ax, doc, block, reading, x_span):
    """Draw as much of a block's label stack as its rectangle has room for.

    The headline comes first and the annotation lines follow in document
    order, which is the emitter's canonical order, so a block cut short
    loses its least important line rather than an arbitrary one.

    Both dimensions are tested, and a line that does not fit is **dropped
    rather than spilled**. Text running out of its own rectangle is worse
    than text missing: on a tower the rectangle is the reading, so a line
    lying across a neighbour asserts a term that block does not carry. A
    narrow layer therefore keeps its name and loses its terms, which the
    reader can still get from the panel beside it or from the frame.

    ``reading`` is the block's ``(height, span, mid)`` in the drawn
    coordinate, from :func:`_block_reading`, so the fit is judged and the
    text is centered where the rectangle actually is under either scale.
    """
    height, span, mid = reading
    rows = _rows_that_fit(ax, height, span)
    if rows < 1:
        return
    width = _axes_width_px(ax) * (block.x1 - block.x0) / x_span
    room = int(width / (CHAR_WIDTH * LABEL_POINTS * ax.figure.dpi / 72.0))
    headline = _typeset(doc, block.label) if block.label else ''
    if headline and len(headline) > room:
        # The headline names the block. An annotation floating in an
        # unnamed rectangle is worse than a blank one, so if the name will
        # not fit, nothing does.
        return
    lines = ([headline] if headline else []) + [
        line for line in (_typeset(doc, v) for v in block.label_lines)
        if len(line) <= room]
    lines = lines[:rows]
    if not lines:
        return
    ax.text(0.5 * (block.x0 + block.x1), mid,
            '\n'.join(lines), ha='center', va='center',
            fontsize=LABEL_POINTS, linespacing=1.3)


#: The diverging ramp a matrix is read on: favorable, neutral, unfavorable.
#: Green and red rather than a perceptual ramp because the reading is a verdict
#: and not a magnitude, and the neutral is a warm grey rather than white so a
#: cell inside the signal-free band is plainly a cell rather than a hole.
MATRIX_FAVORABLE = '#008300'
MATRIX_NEUTRAL = '#f0efec'
MATRIX_UNFAVORABLE = '#e34948'

#: Gap between two column bands, in cell widths. Wide enough to read as a break
#: and narrow enough that the matrix stays one object.
MATRIX_BAND_GAP = 0.35

#: Luminance below which a cell's text flips to white. The usual 0.5 leaves the
#: mid greens unreadable either way, so the threshold sits under it.
MATRIX_DARK_TEXT_LUMA = 0.45


def _matrix_offsets(groups, count):
    """Per-column x offsets that open a gap between column bands.

    Parameters
    ----------
    groups : tuple of str
        One group name per column, or empty for no grouping.
    count : int
        How many columns there are.

    Returns
    -------
    numpy.ndarray
        The offset to add to each column's left edge, cumulative so every band
        after the first is pushed clear of the one before it.
    """
    offsets = np.zeros(count)
    if not groups:
        return offsets
    gaps = 0.0
    for i in range(1, count):
        if groups[i] != groups[i - 1]:
            gaps += MATRIX_BAND_GAP
        offsets[i] = gaps
    return offsets


def _matrix_norm(signed, center, neutral):
    """The diverging colormap and norm for a matrix, with its neutral band.

    Parameters
    ----------
    signed : numpy.ndarray
        The polarity-applied departures from center, NaN where a cell carries
        no value or where the departure is not finite.
    center : float or None
        Where neutral sits. ``None`` means the values do not diverge.
    neutral : float
        Half-width of the signal-free band, in the values' own units.

    Returns
    -------
    (Colormap, Normalize) or (None, None)
        ``(None, None)`` where there is nothing to scale, meaning no center was
        declared or every departure is zero. The caller then draws text only.

    Notes
    -----
    The band is a **hard** stop rather than a soft midpoint: two segments of the
    ramp are pinned to the neutral color and the first step outside the band is
    unmistakably green or red. The document's claim is that a departure inside
    the band carries no signal, and a gradient running through it would show the
    reader a signal the emitter said was not there.
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    finite = signed[np.isfinite(signed)]
    if center is None or finite.size == 0:
        return None, None
    amplitude = float(np.abs(finite).max())
    if amplitude <= 0:
        return None, None
    # Where the band edges fall on a 0-to-1 ramp, clipped off the ends so a band
    # as wide as the data still leaves a sliver of each signal color.
    lo = float(np.clip(0.5 - neutral / (2 * amplitude), 0.02, 0.49))
    hi = float(np.clip(0.5 + neutral / (2 * amplitude), 0.51, 0.98))
    cmap = LinearSegmentedColormap.from_list('matrix_diverging', [
        (0.0, MATRIX_FAVORABLE), (lo, '#8fca8f'), (lo, MATRIX_NEUTRAL),
        (hi, MATRIX_NEUTRAL), (hi, '#f0a09f'), (1.0, MATRIX_UNFAVORABLE)])
    return cmap, TwoSlopeNorm(vmin=-amplitude, vcenter=0.0, vmax=amplitude)


def _render_matrix_panel(ax, doc, panel, series):
    """Draw a 'matrix' panel: named rows against named columns.

    Parameters
    ----------
    ax : matplotlib Axes
    doc : ChartDoc
    panel : Panel
        Kind 'matrix'. Its x and y axes are categorical; its z axis names what
        the values measure.
    series : list of ChartSeries
        The panel's series. Exactly one carries the matrix, which the document's
        own validation has already enforced.

    Notes
    -----
    The number printed in a cell is the raw value and the **color** is the
    polarity-applied departure from center. Keeping those apart is the point: a
    row read in the opposite direction shows the same multiple as its neighbors
    and colors it the other way, which is what ``row_polarity`` is for. Printing
    a negated number instead would be a lie about the quantity.

    A cell whose signed departure is not finite, which a ratio through a
    negative denominator produces, is drawn with no color and muted text rather
    than dropped. The number is real; it is the diverging scale that cannot read
    it.
    """
    from matplotlib.colors import to_rgb

    matrix = next(s.matrix for s in series if s.matrix is not None)
    axes = {a.id: a for a in doc.axes}
    nrow, ncol = len(matrix.rows), len(matrix.columns)
    values = np.array([[np.nan if v is None else float(v) for v in row]
                       for row in matrix.values], dtype=float)

    # The signed departure the color reads. With no center declared there is
    # nothing to depart from and the panel is text only.
    if matrix.center is None:
        signed = np.full_like(values, np.nan)
    else:
        polarity = np.array(matrix.polarity, dtype=float).reshape(-1, 1)
        with np.errstate(invalid='ignore'):
            signed = polarity * (values - matrix.center)
    cmap, norm = _matrix_norm(signed, matrix.center, matrix.neutral)

    offsets = _matrix_offsets(matrix.column_groups, ncol)
    # One quadmesh per column band: a band gap is a break in the x coordinate,
    # and a single mesh would stretch a cell across it.
    starts = [0] + [i for i in range(1, ncol)
                    if matrix.column_groups
                    and matrix.column_groups[i] != matrix.column_groups[i - 1]]
    bounds = starts + [ncol]
    if cmap is not None:
        for lo, hi in zip(bounds, bounds[1:]):
            ax.pcolormesh(np.arange(lo, hi + 1) + offsets[lo],
                          np.arange(nrow + 1), signed[:, lo:hi],
                          cmap=cmap, norm=norm, edgecolors='white',
                          linewidth=2)

    annotations = matrix.annotations
    for i in range(nrow):
        for j in range(ncol):
            x = j + 0.5 + offsets[j]
            if cmap is None or not np.isfinite(signed[i, j]):
                ink, faint = '#52514e', '#898781'
            elif (0.2126 * to_rgb(cmap(norm(signed[i, j])))[0]
                    + 0.7152 * to_rgb(cmap(norm(signed[i, j])))[1]
                    + 0.0722 * to_rgb(cmap(norm(signed[i, j])))[2]
                    <= MATRIX_DARK_TEXT_LUMA):
                ink, faint = 'white', '#ffffffd0'
            else:
                ink, faint = '#0b0b0b', '#52514e'
            if np.isfinite(values[i, j]):
                # A minus sign, not a hyphen: this is a number being read.
                text = f'{values[i, j]:.2f}×'.replace('-', '−')
                ax.text(x, i + (0.36 if annotations else 0.5), text,
                        ha='center', va='center', color=ink,
                        fontsize=FONT_SIZE + 1)
            if annotations and annotations[i][j]:
                ax.text(x, i + 0.72,
                        f'({annotations[i][j]})'.replace('-', '−'),
                        ha='center', va='center', color=faint,
                        fontsize=FONT_SIZE - 1.7)

    # Row band boundaries, drawn in the background color rather than as rules,
    # so a break reads as space between groups instead of another line.
    if matrix.row_groups:
        span = ncol + (offsets[-1] if ncol else 0.0)
        for i in range(1, nrow):
            if matrix.row_groups[i] != matrix.row_groups[i - 1]:
                ax.hlines(i, 0, span, color='white', linewidth=5, zorder=4)

    ax.set_xlim(0, ncol + (offsets[-1] if ncol else 0.0))
    ax.set_ylim(0, nrow)
    ax.set_xticks(np.arange(ncol) + 0.5 + offsets,
                  [_typeset(doc, c) for c in matrix.columns])
    ax.set_yticks(np.arange(nrow) + 0.5,
                  [_typeset(doc, r) for r in matrix.rows])
    # Rows read top to bottom, the order the document lists them in.
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.set(xlabel=_typeset(doc, axes[panel.x_axis].label),
           ylabel=_typeset(doc, axes[panel.y_axis].label))


def _render_tower_panel(ax, doc, panel, blocks, log=False, full=False,
                        floor=None):
    """Draw a 'tower' panel: labeled rectangles over a quantity axis.

    Parameters
    ----------
    ax : matplotlib Axes
    doc : ChartDoc
    panel : Panel
        Kind 'tower'. Its x axis is a placement axis in share and its y
        axis the quantity the blocks band.
    blocks : list of TowerBlock
        The panel's own blocks, in document order, which is bottom up.
    log : bool or {'x', 'y', 'xy'}
        Read the quantity axis on log where it declares one. The placement
        axis never does: a share is not a quantity anyone reads on log.
    full : bool
        Read the quantity axis at its full extent.
    floor : float or None
        The quantity axis' decade floor, from :func:`_tower_floors`,
        pooled over every panel that shares the axis. ``None`` falls back
        to this panel's own blocks, which is right only when it is the
        axis' only panel.

    Notes
    -----
    **The placement axis carries no ticks.** Width is share, and the reader
    takes it by comparison against the full strip beside it, not off a
    scale: a "0.6" tick on a rectangle whose own label says "60% po" is the
    same fact twice and invites the axis to be read as a quantity.

    **The quantity axis is ticked at the boundaries.** A tower is read at
    its breaks, so the panel's horizontal marks become the y ticks,
    labeled with the amounts the document put on them, in place of a
    continuous scale. That is the prior art's move and it is what makes the
    attachments legible without a label per block repeating them.

    **The log reading needs a bottom, and three things follow from it.** A
    balanced program is layered in a roughly geometric progression, whose
    bands are close to equal height on log and slivers on linear, so the
    reading earns its place. But the quantity axis routinely starts at an
    exact zero, which has no position on log: the panel's decade floor
    stands in for it, blocks are clamped to that floor so a rectangle
    starting at zero still has a foot, a boundary mark at zero is dropped
    rather than placed, and labels are sized and centered in decades so a
    band that is 43% of the drawn height is not judged as 15% of the range.
    """
    y_axis = {a.id: a for a in doc.axes}[panel.y_axis]
    x_axis = {a.id: a for a in doc.axes}[panel.x_axis]
    _, log_y = _log_directions(log)
    window = _axis_window(y_axis, full) or (
        min((b.y0 for b in blocks), default=0.0),
        max((b.y1 for b in blocks), default=1.0))
    scale = _axis_scale(y_axis, log_y)
    if floor is None:
        floor = _decade_floor([b.y0 for b in blocks] + [b.y1 for b in blocks])
    bottom = floor if scale == 'log' else None
    seen = {}
    for block in blocks:
        shade = seen.get(block.role, 0)
        seen[block.role] = shade + 1
        _draw_block(ax, block, shade, bottom)
    x_window = _axis_window(x_axis, full) or (0.0, 1.0)
    for block in blocks:
        _label_block(ax, doc, block, _block_reading(block, window, scale, floor),
                     x_window[1] - x_window[0])
    _apply_axis(ax, 'y', y_axis, scale, window, floor)
    ax.set_xlim(*x_window)
    ax.set_xticks([])
    ax.set(xlabel='', ylabel=_typeset(doc, y_axis.label))
    ticks = sorted({m.at for m in doc.marks if m.panel_id == panel.id})
    if scale == 'log':
        # A boundary at zero has no position on a log axis, and an
        # aggregate cover written ``20 xs 0`` emits one. Drop it rather
        # than hand matplotlib a tick it cannot place.
        ticks = [at for at in ticks if at > 0]
    if ticks:
        labels = {m.at: m.label for m in doc.marks
                  if m.panel_id == panel.id and m.label}
        ax.set_yticks(ticks)
        ax.set_yticklabels([_typeset(doc, labels.get(at, '')) or f'{at:,.0f}'
                            for at in ticks], fontsize=LABEL_POINTS)


def _render_xy_panel(ax, doc, panel, series_list, log=False, full=False,
                     return_period=False, invert=False, reflect=False,
                     y_floor=None):
    """Render one 'xy' panel: role-styled curves, gaps broken, marks drawn.

    ``return_period`` swaps a drawn probability axis for the paired reading
    the document offers on it, which is a change of coordinates and not of
    data: the same curve, interrogated at 1 in 200 rather than at 0.995.
    Panels the document offers no pairing for are untouched.

    ``reflect`` is the other change of coordinates on a probability axis,
    ``v`` to ``1 - v``, again where the document declares it
    (:attr:`~aggregate.charts.ir.ChartAxis.complement_of`). A non-exceeding
    probability becomes the exceedance, so the quantile function drawn
    against it is the survival function, and the unit square of a
    distortion reflected in both axes is its dual.

    ``invert`` exchanges the two axes of a panel that declares itself
    invertible, drawing the same pairs the other way round: a quantile
    function becomes the distribution function it inverts. Everything that
    follows reads the axes rather than the panel, so the exchange is the
    only thing that has to happen: the ladder picks a right-continuous step
    where it was picking a left-continuous one, the window and the labels
    follow their axes, and a paired reading rides along on whichever axis
    it was attached to.

    Notes
    -----
    **The atomic ladder needs no reflected case, and it looks like it
    should.** :func:`_draw_atomic` picks its step direction off the axis
    units, and a reflected probability axis is still a probability, so the
    same rung is chosen. Reflection reverses the direction the curve is
    monotone in, so a right-continuous step ought to become
    left-continuous, and it does, for free: matplotlib's step drawstyles
    are defined on the **order of the points given**, not on the direction
    of the axis. Points ``(x0, y0)`` and ``(x1, y1)`` with ``x0 < x1`` draw
    their corner at ``(x1, y0)``; reflected, the same two points draw it at
    ``(1 - x1, y0)``, which is that corner mirrored. The picture is the
    mirror of the picture, which is what was asked for. This is the same
    reason ``invert`` needed no explicit switch.
    """
    axes = {a.id: a for a in doc.axes}
    x_axis, y_axis = axes[panel.x_axis], axes[panel.y_axis]
    x_reflected = y_reflected = False
    x_period = y_period = None
    if reflect:
        pair = _paired_reading(doc, panel.x_axis, 'complement_of')
        if pair is not None:
            x_axis, x_reflected = pair, True
        pair = _paired_reading(doc, panel.y_axis, 'complement_of')
        if pair is not None:
            y_axis, y_reflected = pair, True
    if return_period:
        how = doc.meta.get('return_period_map', 'reciprocal')
        # Both readings on one axis: the return-period axis is the one
        # shown, and what is left to do to an already reflected axis is
        # the reciprocal, whatever the document's own map says. That is
        # not a special case but the definition, since 'complement' *is*
        # reflect-then-reciprocal (see RETURN_PERIOD_MAPS). On a loss it
        # redraws the curve return period drew alone; on a signed outcome
        # it reads the upside tail instead of the shortfall.
        pair = _paired_reading(doc, panel.x_axis)
        if pair is not None:
            x_axis, x_period = pair, 'reciprocal' if x_reflected else how
        pair = _paired_reading(doc, panel.y_axis)
        if pair is not None:
            y_axis, y_period = pair, 'reciprocal' if y_reflected else how
    x_map = _reading_map(x_reflected, x_period)
    y_map = _reading_map(y_reflected, y_period)
    inverted = bool(invert) and panel.invertible
    if inverted:
        x_axis, y_axis = y_axis, x_axis
        x_map, y_map = y_map, x_map
        # The period travels with its axis too: the window rules below are
        # keyed on it, and they read the axes after the exchange.
        x_period, y_period = y_period, x_period

    def coords(values, mapping):
        out = np.array([np.nan if v is None else v for v in values],
                       dtype=float)
        return out if mapping is None else mapping(out)

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
    if x_period and _axis_window(x_axis, full) is None:
        # Keyed on the return period specifically, never on "a map is
        # present": the cap exists because the quantile function saturates
        # and T diverges, and a reflected probability axis is bounded in
        # [0, 1] with nothing to cap.
        #
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
        if mapping is not None:
            at = float(mapping([at])[0])
            if not np.isfinite(at):
                continue
        line = ax.axvline if orient == 'v' else ax.axhline
        line(at, lw=0.75 if not m.faint else 0.5, color='C7', ls='--',
             alpha=0.45 if m.faint else 1.0)
    # A return-period reading re-slices the panel: the deep tail such an
    # axis exists to show sits far outside the window computed for the
    # probability reading, so the companion axis follows the data instead.
    # The compositor's quantile worker does the same by relim-and-autoscale.
    #
    # Keyed on the period and not on "a map is present", for the same
    # reason the cap above is. Reflection is a bijection of [0, 1] onto
    # itself, so it re-slices nothing and the companion window must stand;
    # and the reflected axis carries its own window from the emitter,
    # which is the whole point of declaring it as a paired axis.
    log_x, log_y = _log_directions(log)
    x_scale, y_scale = _axis_scale(x_axis, log_x), _axis_scale(y_axis, log_y)
    y_window = None if x_period else _axis_window(y_axis, full)
    x_only = None if y_period else x_window
    if panel.aspect == 'equal' and x_scale == y_scale:
        x_only = y_window = _square_window(x_only, y_window, all_x, all_y)
    _apply_axis(ax, 'x', x_axis, x_scale, x_only, _decade_floor(all_x))
    # A curve sharing a tower's quantity axis takes the tower's floor, so
    # the two panels are one reading; it keeps its own wherever the axis
    # was remapped to a paired reading, whose values are not that axis'.
    _apply_axis(ax, 'y', y_axis, y_scale, y_window,
                _decade_floor(all_y) if (y_floor is None or y_period)
                else y_floor)
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
                  reflect=False, return_period=False, invert=False,
                  kind=None):
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
        Raise :class:`~aggregate.charts.ir.ChartCapabilityError` for any
        panel this renderer can only degrade, instead of drawing the
        declared degradation.
    log : bool or {'x', 'y', 'xy'}
        ``True`` reads every axis that declares a log scale on log; ``'x'``
        or ``'y'`` confines the reading to that drawing direction, so
        ``log='y'`` draws a log ordinate over a linear abscissa (the
        reading the app's per-panel ``logX`` / ``logY`` controls exist
        for). An axis that declares one reading is untouched whichever is
        asked, so a document with nothing to say about log draws
        identically either way. On a grid panel the z (color) axis rides
        the ``'y'`` direction.
    full_range : bool
        Read every axis that carries a ``full_range`` at its full extent
        instead of at the window it suggests. Axes carrying only a
        suggestion are untouched.
    reflect : bool
        Read every probability axis the document pairs with its complement
        as that complement (:attr:`ChartAxis.complement_of`), the map ``v``
        to ``1 - v``. A non-exceeding probability becomes the exceedance,
        so a Lee panel draws the quantile against the exceedance and,
        inverted, draws the survival function, which is the reading a log
        axis exists for; the unit square of a distortion reflects in both
        axes and gives the dual. Axes with no complement declared are
        untouched.
    return_period : bool
        Draw a probability axis the document pairs with a return-period
        reading as that reading (:attr:`ChartAxis.reciprocal_of`), which
        spreads the rare tail so it can be read off directly. The
        transform comes from ``meta['return_period_map']``, except on an
        axis already read reflected, where what is left to do to it is the
        reciprocal by the definition of the two maps: on a loss that is
        the same curve this switch draws by itself, and on a signed
        outcome it is the upside tail rather than the shortfall.
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
    ~aggregate.charts.ir.ChartCapabilityError
        For a requested kind a panel does not declare, under ``strict``
        for degraded kinds, and always for kinds with no realization here
        yet.

    Notes
    -----
    The five reading switches act on **every** axis or panel that declares
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
        # A tower and the curve beside it are read off one quantity axis,
        # which runs up the figure rather than across it, so the sharing
        # that makes them one picture is in y. Every panel must name the
        # same one: a document with two cession stages carries a per-claim
        # axis and an annual one, which are different quantities and must
        # not be dragged onto one window.
        shared_y = (len({p.y_axis for p in doc.panels}) == 1
                    and any(r == 'tower' for r in realized) and not square)
        # A tower is a strip and a curve wants room, so the row is split by
        # what each panel has to show rather than evenly.
        ratios = ([TOWER_WIDTH if r == 'tower' else CURVE_WIDTH
                   for r in realized]
                  if any(r == 'tower' for r in realized) else None)
        # Equal-aspect panels are squares, and squares in a row need a
        # canvas that is as many squares wide, or constrained layout
        # collapses them to slivers trying to honor the aspect.
        all_square = all(p.aspect == 'equal' for p in doc.panels)
        size = (len(doc.panels) * FIG_H, FIG_H) if all_square else None
        if ratios is not None and size is None:
            # One tower's worth of width per unit of ratio, so a figure
            # holding five panels is not five full-width plots wide, and a
            # taller canvas than the house default, because a tower is read
            # up the page and every row of height is a row of label.
            size = (0.5 * FIG_W * sum(ratios), TOWER_HEIGHT * FIG_H)
        _, grid = make_grid(
            1, len(doc.panels), squeeze=False, sharex=shared,
            sharey=shared_y,
            **({} if ratios is None else
               {'gridspec_kw': {'width_ratios': ratios}}),
            **({} if size is None else {'figsize': size}))
        axs = list(grid[0])

    # One floor per quantity axis, computed before anything is drawn,
    # because panels sharing an axis must share its bottom.
    floors = _tower_floors(doc, realized)
    for panel, realization, panel_ax in zip(doc.panels, realized, axs):
        series = [s for s in doc.series if s.panel_id == panel.id]
        panel_title = panel.title or title
        if invert and panel.invertible:
            # Exchanged axes draw a different picture, and the document
            # names it; with no name to use, say so rather than invent one.
            panel_title = (panel.inverse_title
                           or f'{panel_title}, inverted')
        if realization == 'matrix':
            _render_matrix_panel(panel_ax, doc, panel, series)
        elif realization == 'tower':
            _render_tower_panel(
                panel_ax, doc, panel,
                [b for b in doc.blocks if b.panel_id == panel.id],
                log=log, full=full_range, floor=floors.get(panel.y_axis))
        elif realization == 'xy':
            _render_xy_panel(panel_ax, doc, panel, series, log=log,
                             full=full_range, reflect=reflect,
                             return_period=return_period, invert=invert,
                             y_floor=floors.get(panel.y_axis))
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
