"""Chart-document IR: frozen dataclasses, ``CHART_IR_VERSION`` 1.

The IR is the contract between the library's chart emitters and every
renderer (the matplotlib renderer in ``aggregate.plots._chartdoc``, the
app's generic ECharts adapter, and anything after them). It carries
*semantics only*: which series, on which axes, at which scales, with which
meaningful marks. Colors, fonts, hover, sizing and theming are the
renderer's business and have no fields here.

The boundary rule is semantics versus realization, not data versus display.
Log or linear is statistical meaning, so ``scale`` lives on the axis; a
sequential color ramp is presentation, so it does not. There is no
``extra_mpl_kwargs`` and no renderer passthrough of any kind: a need the
schema cannot express changes the schema visibly, or the chart stays
bespoke.

Determinism mirrors greater_tables (``greater_tables.ir`` and
``greater_tables.engine.hashing``): :func:`canonical_dict` gives
deterministic field presence (defaults omitted, NFC-normalized strings),
:func:`canonical_json` sorted-key compact UTF-8 bytes, :func:`doc_hash` the
first 12 hex characters of the sha256 of the hashless canonical form, and
:func:`stamp` writes the hash back without perturbing it. Same document,
same bytes, same hash, on any machine, any run.

Vocabularies (panel kinds, axis units, series roles, mark roles) are
documented strings, not enums, so version 1 can grow without schema churn.
A reader must ignore roles it does not know; a writer must not invent a
synonym for a role that already exists.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import unicodedata
from dataclasses import dataclass, field

__all__ = [
    'CHART_IR_VERSION', 'ChartAxis', 'ChartCapabilityError', 'ChartDoc',
    'ChartSeries', 'Mark', 'Panel', 'SurfaceData',
    'canonical_dict', 'canonical_json', 'doc_hash', 'stamp',
]

CHART_IR_VERSION = 1

#: Panel kinds. 'xy' is a family of curves over a shared pair of axes;
#: 'heatmap' and 'surface' carry one z grid each (a ``SurfaceData``), read
#: flat or in relief. A renderer that cannot realize a kind declares so
#: (see :class:`ChartCapabilityError`) rather than approximating silently.
PANEL_KINDS = ('xy', 'heatmap', 'surface')

#: Axis scales. Log or linear is statistical meaning (a heavy tail is
#: legible only on log), never styling.
AXIS_SCALES = ('linear', 'log')

#: Axis units, the working vocabulary (open; documented additions only):
#: 'currency' (a loss or outcome amount), 'probability', 'density'
#: (a mass or an ordinate), 'return_period', 'ratio' (a unitless share,
#: alpha and beta curves), 'index' (an ordinal position).
AXIS_UNITS = ('currency', 'probability', 'density', 'return_period',
              'ratio', 'index')

#: Series roles, the working vocabulary (open; documented additions only):
#: 'density', 'survival', 'cdf', 'identity' (the diagonal), 'distortion'
#: (a g(s) curve), 'gross', 'ceded', 'net', 'subject' (the aggregate
#: cover's subject when an occurrence program sits underneath), 'total',
#: 'unit' (one member of a portfolio family), 'joint' (a z grid),
#: 'iso_total' (a level set of x + y over a joint grid: every point on it
#: is one total loss, so it is the line the portfolio reader traces).
SERIES_ROLES = ('density', 'survival', 'cdf', 'identity', 'distortion',
                'gross', 'ceded', 'net', 'subject', 'total', 'unit', 'joint',
                'iso_total')

#: Mark roles: 'mean', 'break_even' (the zero of a signed outcome axis),
#: 'capital_anchor' (a return-period quantile such as 1-in-200).
MARK_ROLES = ('mean', 'break_even', 'capital_anchor')


class ChartCapabilityError(RuntimeError):
    """A renderer was asked (strictly) for a panel kind it cannot realize.

    Raised by a renderer running with ``strict=True`` when a panel's kind
    has no faithful realization there (matplotlib and a 3-D 'surface', for
    instance, where the honest non-strict answer is a labeled 2-D
    projection). Defined here, inside the IR, so emitter-side code and
    tests can name it without importing any renderer.
    """


# --------------------------------------------------------------------------
# The dataclasses. All frozen; sequence fields are coerced to tuples in
# ``__post_init__`` so documents are immutable all the way down and safe to
# share, cache and hash.
# --------------------------------------------------------------------------

def _freeze_seq(obj, name, converter=tuple):
    object.__setattr__(obj, name, converter(getattr(obj, name)))


@dataclass(frozen=True)
class SurfaceData:
    """One z grid: the data of a 'heatmap' or 'surface' panel.

    Parameters
    ----------
    x, y : tuple of float
        Grid cell centers along the panel's x and y axes.
    z : tuple of tuple of float
        Row-major values, ``z[i][j]`` at ``(x[j], y[i])``: ``len(y)`` rows
        of ``len(x)`` values, matplotlib's ``pcolormesh`` orientation.

    Notes
    -----
    What the values *are* (a mass per display cell, a density ordinate) is
    the emitting chart's semantics and is documented there; when the grid
    was reduced from a finer computational grid, the reduction must be mass
    preserving, because that reduction is meaning, not styling.
    """

    x: tuple
    y: tuple
    z: tuple

    def __post_init__(self):
        _freeze_seq(self, 'x')
        _freeze_seq(self, 'y')
        object.__setattr__(self, 'z', tuple(tuple(row) for row in self.z))
        for i, row in enumerate(self.z):
            if len(row) != len(self.x):
                raise ValueError(
                    f'SurfaceData row {i} has {len(row)} values, '
                    f'expected len(x) = {len(self.x)}')
        if len(self.z) != len(self.y):
            raise ValueError(
                f'SurfaceData has {len(self.z)} rows, '
                f'expected len(y) = {len(self.y)}')


@dataclass(frozen=True)
class ChartAxis:
    """One axis: a labeled, scaled reading of a quantity.

    Parameters
    ----------
    id : str
        Referenced by panels. Unique within the document.
    label : str
        The human reading ('loss', 'S(x)', a component's resolved label).
    scale : str
        'linear' or 'log'. Statistical meaning, never styling.
    suggested_range : tuple of float, optional
        The (lo, hi) window the emitter computed from the data (a quantile
        crop, a unit interval). A suggestion: renderers may pan or zoom,
        but the initial view honors it.
    kind : str
        'value' (continuous) or 'category' (ordinal positions).
    unit : str, optional
        One of :data:`AXIS_UNITS`; what the numbers measure.
    reciprocal_of : str, optional
        The id of another axis in the document of which this axis is the
        reciprocal reading (return period ``T = 1/S`` of a survival axis).
        The pairing is semantic: the two scales name the same curve, and a
        renderer may realize the pair as one axis with a twin. JUDGMENT
        CALL flagged in ``dev/chart-inventory.md``: the author confirms the
        twin is IR, not a renderer trick, at schema sign off.
    """

    id: str
    label: str
    scale: str = 'linear'
    suggested_range: tuple = None
    kind: str = 'value'
    unit: str = None
    reciprocal_of: str = None

    def __post_init__(self):
        if self.scale not in AXIS_SCALES:
            raise ValueError(f'unknown axis scale {self.scale!r}; '
                             f'expected one of {AXIS_SCALES}')
        if self.kind not in ('value', 'category'):
            raise ValueError(f'unknown axis kind {self.kind!r}; '
                             "expected 'value' or 'category'")
        if self.suggested_range is not None:
            _freeze_seq(self, 'suggested_range')
            if len(self.suggested_range) != 2:
                raise ValueError('suggested_range must be (lo, hi)')


@dataclass(frozen=True)
class Panel:
    """One subplot slot: a kind, its axes, and how it is read.

    Parameters
    ----------
    id : str
        Referenced by series and marks. Unique within the document.
    kind : str
        One of :data:`PANEL_KINDS`.
    x_axis, y_axis : str
        Axis ids. Two panels referencing the *same* axis id share that
        axis (the two-panel exhibits share one loss window this way).
    z_axis : str, optional
        The value axis of a z grid ('heatmap' color, 'surface' height).
        Required for those kinds, meaningless for 'xy'.
    read_axis : str
        'x' or 'y': which axis the reader interrogates. A density is read
        by loss ('x'); a tail panel is read probability to loss ('y'),
        because at a chosen survival the answer is the VaR.
    aspect : str, optional
        ``'equal'`` when equal data aspect is semantic (the g(s) unit
        square, the complex-plane Fourier disk), ``None`` otherwise. Never
        set for mere squareness of the box; anisotropic joint grids are
        deliberately free.
    title : str, optional
        The panel's own heading ('Density', 'Survival'). Semantic: it
        names what is drawn, not how.

    Notes
    -----
    A panel does not list its series; each series names its panel via
    ``panel_id`` and draw order is document order. One source of truth,
    checked in :meth:`ChartDoc.__post_init__`.
    """

    id: str
    kind: str
    x_axis: str
    y_axis: str
    z_axis: str = None
    read_axis: str = 'x'
    aspect: str = None
    title: str = None

    def __post_init__(self):
        if self.kind not in PANEL_KINDS:
            raise ValueError(f'unknown panel kind {self.kind!r}; '
                             f'expected one of {PANEL_KINDS}')
        if self.read_axis not in ('x', 'y'):
            raise ValueError("read_axis must be 'x' or 'y'")
        if self.aspect not in (None, 'equal'):
            raise ValueError("aspect must be 'equal' or None")
        if self.kind in ('heatmap', 'surface') and self.z_axis is None:
            raise ValueError(f"panel {self.id!r} kind {self.kind!r} "
                             'requires a z_axis')


@dataclass(frozen=True)
class ChartSeries:
    """One drawn thing: a named, role-carrying curve or grid.

    Parameters
    ----------
    name : str
        Legend identity. Series in different panels sharing a name are the
        same entity seen twice (the app links their legend toggles).
    role : str
        One of :data:`SERIES_ROLES` (open vocabulary): what the series
        *is*, which is what lets a renderer choose an honest default
        (steps for a discretized density, a dashed diagonal for identity)
        without per-chart styling in the IR.
    panel_id : str
        The panel this series draws in.
    x, y : tuple, optional
        Point coordinates for 'xy' panels, equal length. ``None`` entries
        are semantic gaps (a log axis cannot place an exact zero; survival
        values below the floating-point dust floor are noise, not tail)
        and renderers must break the line there, never bridge it.
    y2 : tuple, optional
        The second edge of a band series (the margin between F and gF, a
        min/max envelope): the series *is* the region between ``y`` and
        ``y2`` over ``x``. Same length as ``y``; only with an x/y payload.
    surface : SurfaceData, optional
        The z grid for 'heatmap' and 'surface' panels, instead of x/y.

    Notes
    -----
    An x/y series draws on **any** panel kind. On a grid panel it is an
    overlay, drawn over the mesh in document order: the iso-total diagonals
    of a joint density, a contour trace, a reference curve. The grid panel
    itself carries exactly one surface series; an 'xy' panel carries none.
    """

    name: str
    role: str
    panel_id: str
    x: tuple = None
    y: tuple = None
    y2: tuple = None
    surface: SurfaceData = None

    def __post_init__(self):
        if (self.surface is None) == (self.x is None and self.y is None):
            raise ValueError(
                f'series {self.name!r} must carry either x/y or surface')
        if self.surface is not None and self.y2 is not None:
            raise ValueError(f'series {self.name!r}: y2 needs an x/y payload')
        if self.surface is None:
            if self.x is None or self.y is None:
                raise ValueError(f'series {self.name!r} needs both x and y')
            _freeze_seq(self, 'x')
            _freeze_seq(self, 'y')
            if len(self.x) != len(self.y):
                raise ValueError(
                    f'series {self.name!r}: len(x) = {len(self.x)} '
                    f'!= len(y) = {len(self.y)}')
            if self.y2 is not None:
                _freeze_seq(self, 'y2')
                if len(self.y2) != len(self.y):
                    raise ValueError(
                        f'series {self.name!r}: len(y2) = {len(self.y2)} '
                        f'!= len(y) = {len(self.y)}')


@dataclass(frozen=True)
class Mark:
    """One meaningful annotation: a reference line a reader acts on.

    Parameters
    ----------
    panel_id : str
        The panel the mark draws in.
    orient : str
        'v' (a vertical at ``x = at``) or 'h' (a horizontal at ``y = at``).
    at : float
        Position on the orienting axis.
    label : str, optional
        The reading ('mean', '1-in-200', 'break even').
    role : str, optional
        One of :data:`MARK_ROLES` (open vocabulary).
    faint : bool
        A de-emphasized mark (the paired capital anchors on the tail
        panel) versus a full-weight one (the mean). Semantic emphasis, not
        a color choice.
    """

    panel_id: str
    orient: str
    at: float
    label: str = None
    role: str = None
    faint: bool = False

    def __post_init__(self):
        if self.orient not in ('v', 'h'):
            raise ValueError("orient must be 'v' or 'h'")


@dataclass(frozen=True)
class ChartDoc:
    """The chart document: the versioned IR a chart emitter returns.

    Parameters
    ----------
    name : str
        The chart's registry name ('joint_surface', 'distortion').
    title : str, optional
        The whole chart's heading, usually built from the emitting
        object's resolved label.
    panels : tuple of Panel
    axes : tuple of ChartAxis
    series : tuple of ChartSeries
        Draw order is document order (a total drawn after its units sits
        on top; that ordering is meaning).
    marks : tuple of Mark
    meta : dict
        Chart-level semantic facts that are not drawable objects
        (``z_log_ok``: the z grid spans enough orders of magnitude that a
        log height reading is meaningful). JSON-representable values only.
    ir_version : int
        Always :data:`CHART_IR_VERSION` for documents this build writes.
    generator : str, optional
        Producer tag, excluded from the content hash.
    hash : str, optional
        The 12-hex content hash, stamped by :func:`stamp`, excluded from
        the hashed form so stamping does not perturb the digest.
    """

    name: str
    title: str = None
    panels: tuple = ()
    axes: tuple = ()
    series: tuple = ()
    marks: tuple = ()
    meta: dict = field(default_factory=dict)
    ir_version: int = CHART_IR_VERSION
    generator: str = None
    hash: str = None

    def __post_init__(self):
        if self.ir_version != CHART_IR_VERSION:
            raise ValueError(
                f'unsupported ir_version {self.ir_version}; '
                f'this build reads {CHART_IR_VERSION}')
        _freeze_seq(self, 'panels')
        _freeze_seq(self, 'axes')
        _freeze_seq(self, 'series')
        _freeze_seq(self, 'marks')
        object.__setattr__(self, 'meta', dict(self.meta))
        panel_ids = [p.id for p in self.panels]
        axis_ids = [a.id for a in self.axes]
        for label, ids in (('panel', panel_ids), ('axis', axis_ids)):
            dupes = {i for i in ids if ids.count(i) > 1}
            if dupes:
                raise ValueError(f'duplicate {label} ids: {sorted(dupes)}')
        for p in self.panels:
            for ax in (p.x_axis, p.y_axis, p.z_axis):
                if ax is not None and ax not in axis_ids:
                    raise ValueError(
                        f'panel {p.id!r} references unknown axis {ax!r}')
        for a in self.axes:
            if a.reciprocal_of is not None and a.reciprocal_of not in axis_ids:
                raise ValueError(
                    f'axis {a.id!r} reciprocal_of unknown axis '
                    f'{a.reciprocal_of!r}')
        kinds = {p.id: p.kind for p in self.panels}
        # A grid panel carries exactly one surface, plus any number of x/y
        # overlays drawn over it (the iso-total diagonals of a joint
        # density). An 'xy' panel carries no surface at all.
        grid_kinds = ('heatmap', 'surface')
        surfaces = {p.id: 0 for p in self.panels if p.kind in grid_kinds}
        for s in self.series:
            if s.panel_id not in kinds:
                raise ValueError(
                    f'series {s.name!r} references unknown panel '
                    f'{s.panel_id!r}')
            if s.surface is None:
                continue
            if kinds[s.panel_id] not in grid_kinds:
                raise ValueError(
                    f'series {s.name!r} carries a surface but panel '
                    f'{s.panel_id!r} is kind {kinds[s.panel_id]!r}')
            surfaces[s.panel_id] += 1
        for pid, count in surfaces.items():
            if count != 1:
                raise ValueError(
                    f'panel {pid!r} carries {count} surface series, '
                    'expected exactly one')
        for m in self.marks:
            if m.panel_id not in kinds:
                raise ValueError(
                    f'mark {m.label!r} references unknown panel '
                    f'{m.panel_id!r}')


# --------------------------------------------------------------------------
# Canonical emission. Mirrors greater_tables: deterministic field presence
# (structural fields always, optional fields only when they differ from
# their defaults), NFC-normalized strings, sorted-key compact UTF-8 JSON,
# sha256 first 12 hex. ``hash`` and ``generator`` are excluded from the
# hashed form so stamping does not perturb the digest.
# --------------------------------------------------------------------------

# Fields always present in canonical form, per class: the structural core a
# reader keys on. Everything else appears only when it differs from its
# default, so adding an optional field never changes existing hashes.
_ALWAYS = {
    ChartDoc: ('ir_version', 'name'),
    Panel: ('id', 'kind', 'x_axis', 'y_axis'),
    ChartAxis: ('id', 'label'),
    ChartSeries: ('name', 'role', 'panel_id'),
    Mark: ('panel_id', 'orient', 'at'),
    SurfaceData: ('x', 'y', 'z'),
}


def _nfc(value):
    """NFC-normalize strings, recursively through tuples and dicts."""
    if isinstance(value, str):
        return unicodedata.normalize('NFC', value)
    if isinstance(value, tuple):
        return [_nfc(v) for v in value]
    if isinstance(value, dict):
        return {_nfc(k): _nfc(v) for k, v in value.items()}
    return value


def _default_of(f):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def _canonical(obj, skip=()):
    """One dataclass instance to its canonical dict, recursively."""
    always = _ALWAYS[type(obj)]
    out = {}
    for f in dataclasses.fields(obj):
        if f.name in skip:
            continue
        value = getattr(obj, f.name)
        if f.name not in always and value == _default_of(f):
            continue
        if isinstance(value, tuple) and value and dataclasses.is_dataclass(value[0]):
            out[f.name] = [_canonical(v) for v in value]
        elif dataclasses.is_dataclass(value):
            out[f.name] = _canonical(value)
        else:
            out[f.name] = _nfc(value)
    return out


def canonical_dict(doc, *, include_hash=True):
    """Return the canonical plain-dict form of a chart document.

    Parameters
    ----------
    doc : ChartDoc
        The document to canonicalize.
    include_hash : bool
        When False, ``hash`` and ``generator`` are omitted: the form the
        content hash operates on.

    Returns
    -------
    dict
        Deterministic field presence, NFC strings, tuples as lists. Key
        ordering is left to the JSON writer (sorted there).
    """
    skip = () if include_hash else ('hash', 'generator')
    return _canonical(doc, skip=skip)


def canonical_json(doc, *, include_hash=True):
    """Serialize a chart document to canonical UTF-8 JSON bytes.

    Sorted keys, compact separators, NFC strings, no NaN or Infinity: a
    value that is not JSON-representable is an emitter bug, surfaced here.

    Parameters
    ----------
    doc : ChartDoc
    include_hash : bool
        When False, omit ``hash`` and ``generator`` (the hashed form).

    Returns
    -------
    bytes
    """
    data = canonical_dict(doc, include_hash=include_hash)
    text = json.dumps(data, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)
    return text.encode('utf-8')


def doc_hash(doc):
    """Return the 12-hex-character sha256 content hash of a document."""
    return hashlib.sha256(
        canonical_json(doc, include_hash=False)).hexdigest()[:12]


def stamp(doc, generator=None):
    """Return a copy of ``doc`` with ``hash`` (and ``generator``) set."""
    changes = {'hash': doc_hash(doc)}
    if generator is not None:
        changes['generator'] = generator
    return dataclasses.replace(doc, **changes)
