"""Chart-document IR: frozen dataclasses, ``CHART_IR_VERSION`` 2.

.. warning::

   **Provisional module, in the sense of PEP 411**, along with the rest of
   :mod:`aggregate.charts`. The schema below is **not part of the 1.0 API
   contract** and may change in a minor release with no deprecation period.
   ``CHART_IR_VERSION`` is how a consumer detects that it has; pin it and
   check it. "Frozen" on the dataclasses below means immutable instances, and
   version 1 being closed to additions is a rule about how the schema changes
   in an orderly way, neither is a stability promise across releases. See
   :doc:`/3_reference/3_x_API_Stability`.

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

**The wire goes both ways.** :func:`load_chart_doc` rebuilds a document from
its canonical dict, so a consumer that fetched one can draw it through
:func:`aggregate.plots.plot_chartdoc` without knowing what emitted it. The
round trip is exact, ``doc_hash(load_chart_doc(canonical_dict(doc))) ==
doc.hash``, and the reader is the one place a wire document's ``ir_version``
is negotiated and this build's omitted defaults are restored.

Vocabularies (panel kinds, axis units, series roles, mark roles) are
documented strings, not enums, so version 1 can grow without schema churn.
A reader must ignore roles it does not know; a writer must not invent a
synonym for a role that already exists.

**Every human-facing string in a document is plain text**, never markup in
any renderer's language. ECharts has no TeX, so a mathtext series name
would already be broken on one of the two renderers that exist. Plain text
is also the legend identity renderers link series toggles by, so it is the
form that must stay stable.

**And every one of them carries both forms.** :attr:`ChartDoc.tex` is a
**total** lookup, not a partial one: the analogy is alt text in HTML, where
you write both because they serve different consumers and you do not make
one consumer guess. matplotlib reads the typeset form, ECharts reads the
plain one, and the emitter that writes a string writes both, a plain word
mapping to itself. A missing entry is therefore an emitter bug and not a
document saying "this string has no typeset form";
:func:`complete_tex` is how an emitter satisfies it without writing the
identities out by hand, and :func:`human_strings` is the set the contract
is checked over. Renderers keep a fallback to the plain string, as a net
under a bug rather than as a licensed state.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import unicodedata
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    'CHART_IR_VERSION', 'SUPPORT_KINDS', 'SURFACE_DTYPES', 'SURFACE_EDGES',
    'ChartAxis', 'ChartCapabilityError',
    'ChartDoc', 'ChartSeries', 'Mark', 'Panel', 'SurfaceData', 'SurfaceZBlock',
    'canonical_dict', 'canonical_json', 'complete_tex', 'decode_z_block',
    'doc_hash', 'encode_z_block',
    'human_strings', 'load_chart_doc', 'stamp',
]

#: The IR version. **When to bump it**, which is the question every
#: reopening of the schema asks: the version marks the point where a reader
#: that ignores what it does not know would draw something *wrong*. Adding
#: a field whose absence leaves the default reading correct and complete
#: does not qualify, however visible the field is, and however many hashes
#: it changes. ``support`` (1.0.0a214) and the declared readings
#: (``scales``, ``full_range``, ``kinds``) all landed at version 1 on that
#: rule: ignore them and you get the one reading the document already named
#: as its default. A field a reader must act on to draw the *default* right,
#: a changed meaning for an existing field, or a removal, all bump it.
#:
#: **Version 2** (1.0.0a238) is that rule's first real application. A series
#: may carry a coordinate as a lattice, ``(start, step, count)``, instead of
#: spelling every value out (:attr:`ChartSeries.x_lattice`). A reader that
#: does not know the field sees a series with no coordinates at all and can
#: draw nothing, which is exactly the "would draw something wrong" case, so
#: the version moves and an old reader refuses the document by name rather
#: than drawing an empty panel.
CHART_IR_VERSION = 2

#: Panel kinds. 'xy' is a family of curves over a shared pair of axes;
#: 'heatmap' and 'surface' carry one z grid each (a ``SurfaceData``), read
#: flat or in relief. A renderer that cannot realize a kind declares so
#: (see :class:`ChartCapabilityError`) rather than approximating silently.
PANEL_KINDS = ('xy', 'heatmap', 'surface')

#: Axis scales. Log or linear is statistical meaning (a heavy tail is
#: legible only on log), never styling.
AXIS_SCALES = ('linear', 'log')

#: How a paired return-period axis is computed from the probability axis it
#: points at (see :attr:`ChartAxis.reciprocal_of`), given the value ``v`` on
#: that axis. 'reciprocal': ``T = 1 / v``, the literal reading, and the
#: default when a document says nothing. It is the map for a survival axis
#: (``T = 1 / S``) and for the shortfall probability of a signed outcome,
#: where the adverse tail is the low one. 'complement': ``T = 1 / (1 - v)``,
#: the map for a non-exceedance axis of a loss, where the adverse tail is
#: the high one and ``1 - p`` is the exceedance the reader is asking about.
#: Carried in ``ChartDoc.meta['return_period_map']``, because it is one fact
#: about the whole document rather than a property of either axis.
#:
#: The two compose with the reflected reading
#: (:attr:`ChartAxis.complement_of`) without any special case, because
#: ``complement(v) = reciprocal(1 - v)``: 'complement' *is* "reflect, then
#: take the reciprocal". So an axis already read reflected takes
#: 'reciprocal' whatever the document declares, and a renderer asked for
#: both readings needs no lookup table. On a loss, whose map is
#: 'complement', that draws the return-period curve it already drew. On a
#: signed outcome, whose map is 'reciprocal' because the adverse tail is
#: the low one, it reads the *upside* tail's return period, which is a
#: picture unreachable any other way.
RETURN_PERIOD_MAPS = ('reciprocal', 'complement')

# The ``ChartAxis`` fields that declare a paired reading: an undrawn axis
# naming the drawn one it is an alternative reading of. Private, because a
# consumer reads the fields it knows by name; it exists so the validation
# that is the same for every pointer is written once.
_PAIRED_READINGS = ('reciprocal_of', 'complement_of')

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

#: What a series' x values *are*, which is a fact about the law and not
#: about the drawing. 'atomic': the points carry the whole distribution and
#: there is nothing between them, which in this library is the normal case,
#: because a discretized aggregate **is** the distribution rather than an
#: approximation to some continuous ideal. 'continuous': the points are
#: samples of a function that exists everywhere between them (a distortion
#: g(s), TVaR as a function of p, a frozen severity's pdf, none of which
#: are discretized at all).
#:
#: The default is 'atomic' because that is what this library computes with.
#: A renderer chooses the drawing from this plus the room it has: stems
#: where the atoms are far enough apart to see, steps where they are not
#: (sharp jumps, never a slope the law does not have), and a plain line
#: once a bucket is sub-pixel and the two are indistinguishable anyway.
SUPPORT_KINDS = ('atomic', 'continuous')

#: What a display-grid coordinate *names*, the fact whose absence made a
#: whole display bucket of bias invisible for four releases (see
#: :class:`SurfaceData`). 'left': the coordinate is the low edge of the cell
#: it labels, which spans ``[x_i, x_i + dx)``. 'mid': it is the cell's
#: midpoint. There is no 'right', because a right-edge convention labels a
#: cell with a coordinate no point in it reaches, which is how the bias got
#: in.
SURFACE_EDGES = ('left', 'mid')

#: How a :class:`SurfaceZBlock` carries its numbers. A closed vocabulary,
#: and the point of naming it in the document is that the default may change
#: without a format change.
#:
#: ================ ============ ====================== ===================
#: dtype            bytes/cell   worst relative error   when
#: ================ ============ ====================== ===================
#: ``f32b64``       4            1.2e-7                 the default
#: ``f64b64``       8            exact                  never by default
#: ``u16log12b64``  2            2.1e-4                 large grids
#: ================ ============ ====================== ===================
#:
#: **Do not default to float64.** The low mantissa bits of an FFT-built
#: density are genuine digits rather than fuzz, so no compressor touches
#: them: measured on a 128 x 128 display grid, float64 with a byte shuffle
#: and zstd came to 92.3 kB against 43.6 kB for float32 and 21.4 kB for the
#: quantized form, which makes a float64 pipeline twice the size of the JSON
#: text it replaces. Choosing the dtype *is* the compression decision.
#:
#: Bytes are little-endian in every case, so the wire does not depend on the
#: machine that wrote it, and base64 is applied last. There is no
#: in-payload compression: base64 costs 33% and the transport's
#: ``Content-Encoding`` gives back all but 0.5% of it, so compressing twice
#: would spend CPU on incompressible bytes.
SURFACE_DTYPES = ('f32b64', 'f64b64', 'u16log12b64')

#: Decades below the peak that ``u16log12b64`` spans, and the count of live
#: codes over them. Code 0 is reserved for an exact zero (a real joint
#: density is 14% to 59% exact zeros, and a reserved code is what keeps them
#: exact rather than decoding to a spurious ``peak * 1e-12``); codes 1 to
#: 65535 are log-uniform over the twelve decades, which resolves a value to
#: 2.1e-4 relative, four significant figures. A reader printing more digits
#: than that under this dtype is inventing precision the wire never carried.
U16_DECADES = 12
U16_LEVELS = 65535


class ChartCapabilityError(RuntimeError):
    """A renderer was asked (strictly) for a panel kind it cannot realize.

    Raised by a renderer running with ``strict=True`` when a panel's kind
    has no faithful realization there (matplotlib and a 3-D 'surface', for
    instance, where the honest non-strict answer is a labeled 2-D
    projection). Defined here, inside the IR, so emitter-side code and
    tests can name it without importing any renderer.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """


# --------------------------------------------------------------------------
# The dataclasses. All frozen; sequence fields are coerced to tuples in
# ``__post_init__`` so documents are immutable all the way down and safe to
# share, cache and hash.
# --------------------------------------------------------------------------

def _freeze_seq(obj, name, converter=tuple):
    object.__setattr__(obj, name, converter(getattr(obj, name)))


def _expand(values, lattice):
    """Coordinates from whichever form carries them.

    ``start + step * i`` reproduces the grid exactly, because that is the
    arithmetic the grid was built with; no accumulation, so no drift.
    """
    if lattice is None:
        return values
    start, step, count = lattice
    return tuple(start + step * i for i in range(int(count)))


def _deep_freeze(value):
    """Lists to tuples, recursively, through dict values.

    The dict-valued surface fields (``window``, ``marginals``, ``moments``)
    arrive as tuples from an emitter and as lists from
    :func:`load_chart_doc`, and a dataclass that stored them as they came
    would compare unequal across a round trip that hashes identically.
    """
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, dict):
        return {k: _deep_freeze(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class SurfaceZBlock:
    """The z values of a :class:`SurfaceData`, encoded.

    Parameters
    ----------
    dtype : str
        One of :data:`SURFACE_DTYPES`; how ``data`` decodes.
    data : str
        Base64 of the little-endian raw bytes, ``nx * ny`` values.
    order : str
        'yx', the only order: row major over ``(y, x)``, so value
        ``i * nx + j`` sits at ``(x0 + j * dx, y0 + i * dy)``. Named rather
        than assumed because an off-by-a-transpose on a non-square grid
        raises an error and on a square one draws a plausible lie.
    peak, decades : float, optional
        For ``u16log12b64`` only: the value code 65535 stands for, and how
        many decades below it the code range spans. Absent for the plain
        float dtypes, which need no scale.

    Notes
    -----
    Build one with :func:`encode_z_block` and read it with
    :func:`decode_z_block` rather than by hand, so the reserved zero code
    and the endianness are stated once.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """

    dtype: str
    data: str
    order: str = 'yx'
    peak: float = None
    decades: float = None

    def __post_init__(self):
        if self.dtype not in SURFACE_DTYPES:
            raise ValueError(f'unknown surface dtype {self.dtype!r}; '
                             f'expected one of {SURFACE_DTYPES}')
        if self.order != 'yx':
            raise ValueError(f"surface z order must be 'yx', "
                             f'got {self.order!r}')
        if (self.dtype == 'u16log12b64') and (self.peak is None
                                              or self.decades is None):
            raise ValueError(
                "u16log12b64 needs 'peak' and 'decades': without the scale "
                'the codes decode to nothing.')


def encode_z_block(values, dtype='f32b64', order='yx'):
    """Encode a flat sequence of z values as a :class:`SurfaceZBlock`.

    Parameters
    ----------
    values : array_like
        The grid, flattened in ``order``. Raveled here, so a 2-D array in
        the right orientation is accepted as it stands.
    dtype : str
        One of :data:`SURFACE_DTYPES`.
    order : str
        'yx'; carried through to the block.

    Returns
    -------
    SurfaceZBlock

    Notes
    -----
    Everything is written little-endian and base64'd last, so the bytes do
    not depend on the machine, which is what keeps :func:`canonical_json`
    deterministic across builds.

    For ``u16log12b64`` the peak is the largest value present and code 0 is
    reserved for an exact zero, so::

        value = 0                                              if code == 0
        value = peak * 10 ** ((code - 1) / 65534 * 12 - 12)     otherwise

    Anything more than twelve decades under the peak encodes as zero, which
    is a statement that the wire cannot carry it rather than a claim that it
    is absent: a consumer that needs that depth asks for a float dtype.
    """
    a = np.asarray(values, dtype=float).ravel()
    peak = decades = None
    if dtype == 'f32b64':
        raw = a.astype('<f4').tobytes()
    elif dtype == 'f64b64':
        raw = a.astype('<f8').tobytes()
    elif dtype == 'u16log12b64':
        decades = float(U16_DECADES)
        peak = float(a.max()) if a.size else 0.0
        if peak <= 0:
            codes = np.zeros(a.size, dtype=np.int64)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.log10(np.where(a > 0.0, a, np.nan) / peak)
            live = 1.0 + np.rint((rel + decades) / decades * (U16_LEVELS - 1))
            codes = np.where(np.isfinite(live), live, 0.0)
        raw = np.clip(codes, 0, U16_LEVELS).astype('<u2').tobytes()
    else:
        raise ValueError(f'unknown surface dtype {dtype!r}; '
                         f'expected one of {SURFACE_DTYPES}')
    return SurfaceZBlock(dtype=dtype, data=base64.b64encode(raw).decode('ascii'),
                         order=order, peak=peak, decades=decades)


def decode_z_block(block):
    """Decode a :class:`SurfaceZBlock` to a flat float array.

    Parameters
    ----------
    block : SurfaceZBlock

    Returns
    -------
    ndarray
        One dimension, ``nx * ny`` values in the block's ``order``. Reshape
        to ``(ny, nx)`` for 'yx'.

    Notes
    -----
    The inverse of :func:`encode_z_block`, exact for ``f64b64`` and to the
    dtype's declared error otherwise. Present here, beside the encoder, so
    the round trip is checkable in one place rather than only against a
    consumer written in another language.
    """
    raw = base64.b64decode(block.data)
    if block.dtype == 'f32b64':
        return np.frombuffer(raw, dtype='<f4').astype(float)
    if block.dtype == 'f64b64':
        return np.frombuffer(raw, dtype='<f8').astype(float)
    if block.dtype == 'u16log12b64':
        codes = np.frombuffer(raw, dtype='<u2').astype(float)
        out = block.peak * 10.0 ** ((codes - 1.0) / (U16_LEVELS - 1)
                                    * block.decades - block.decades)
        return np.where(codes == 0.0, 0.0, out)
    raise ValueError(f'unknown surface dtype {block.dtype!r}')


@dataclass(frozen=True)
class SurfaceData:
    """One z grid: the data of a 'heatmap' or 'surface' panel.

    Parameters
    ----------
    x, y : tuple of float
        The coordinate of each cell along the panel's x and y axes. What a
        coordinate *names* is :attr:`edge`; a grid that declares nothing is
        read as cell midpoints, which is what these were documented as
        before the lattice fields arrived.
    z : tuple of tuple of float
        Row-major values, ``z[i][j]`` at ``(x[j], y[i])``: ``len(y)`` rows
        of ``len(x)`` values, matplotlib's ``pcolormesh`` orientation.
    x0, dx, nx : float, float, int, optional
        The x lattice as an origin, a step and a count, in place of the
        array. See the Notes.
    y0, dy, ny : float, float, int, optional
        The y lattice. Independent of x in both step and count: one real
        joint comes out 64 wide against 128 deep with ``dx = 2`` against
        ``dy = 512``, so any consumer that assumes a square mesh, or reads a
        line of constant ``x + y`` off the index anti-diagonal, is wrong on
        real data.
    edge : str, optional
        One of :data:`SURFACE_EDGES`: what a coordinate names.
    bs : tuple of float, optional
        The **fine** bucket size each axis was reduced from, so
        ``dx == bs[0] * k[0]``.
    k : tuple of int, optional
        The block factor per axis. With ``bs`` this says what the grid is a
        reduction *of*, which is the difference between "the support starts
        at 508" and "the first display cell covers ``[0, 512)``".
    window : dict, optional
        ``{'p': depth, 'x': (lo, hi), 'y': (lo, hi), 'kept': fraction}``:
        the depth that was asked for, the resulting outer edges in data
        coordinates, and the share of the grid's mass inside them.
    marginals : dict, optional
        ``{'x': (...), 'y': (...)}``, each of length ``nx`` / ``ny``: the
        **exact** marginals on the display lattice, from the emitting object
        rather than integrated off the reduced and windowed joint, which is
        a different and worse curve. Same units as ``z``.
    moments : dict, optional
        ``{'mean': (mx, my)}`` and whatever else an emitter documents,
        computed on the **fine** lattice. A reference the consumer can check
        its own arithmetic against, and cheap: two numbers.
    deficit : float, optional
        Mass the construction never placed, already computed upstream.
    z_block : SurfaceZBlock, optional
        ``z`` again, encoded (see :func:`encode_z_block`). A new consumer
        prefers it; an old one reads ``z`` and is unaffected.

    Notes
    -----
    What the values *are* (a mass per display cell, a density ordinate) is
    the emitting chart's semantics and is documented there; when the grid
    was reduced from a finer computational grid, the reduction must be mass
    preserving, because that reduction is meaning, not styling.

    **Both lattices go as origin, step and count, and the reason is not
    size.** The grids *are* arithmetic sequences: an aggregate lives on a
    lattice by construction and a block reduction by a power-of-two factor
    leaves one. Everything a consumer does off the grid (a bilinear lookup,
    the line of constant total, a mean along a cut) divides by a constant
    step, and against arrays that division is an assumption the format
    permits the emitter to violate. Against ``x0``, ``dx``, ``nx`` there is
    nothing to violate. With ``bs`` and ``k`` alongside, the fine lattice
    comes free.

    The ``x``/``y``/``z`` arrays are the older form of the same grid, kept
    beside the lattice fields for one release so a consumer can move at its
    own pace. Dropping them is the breaking change and is what bumps
    :data:`CHART_IR_VERSION`.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """

    x: tuple
    y: tuple
    z: tuple
    x0: float = None
    dx: float = None
    nx: int = None
    y0: float = None
    dy: float = None
    ny: int = None
    edge: str = None
    bs: tuple = None
    k: tuple = None
    window: dict = None
    marginals: dict = None
    moments: dict = None
    deficit: float = None
    z_block: SurfaceZBlock = None

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
        if self.edge is not None and self.edge not in SURFACE_EDGES:
            raise ValueError(f'unknown surface edge {self.edge!r}; '
                             f'expected one of {SURFACE_EDGES}')
        for name in ('bs', 'k'):
            if getattr(self, name) is not None:
                _freeze_seq(self, name)
                if len(getattr(self, name)) != 2:
                    raise ValueError(f'SurfaceData {name} must be a pair, '
                                     f'got {getattr(self, name)!r}')
        for name in ('window', 'marginals', 'moments'):
            if getattr(self, name) is not None:
                object.__setattr__(self, name,
                                   _deep_freeze(getattr(self, name)))
        for side, values in (('x', self.x), ('y', self.y)):
            count = getattr(self, f'n{side}')
            if count is not None and int(count) != len(values):
                raise ValueError(
                    f'SurfaceData n{side} = {count} against '
                    f'len({side}) = {len(values)}')
            marginal = (self.marginals or {}).get(side)
            if marginal is not None and len(marginal) != len(values):
                raise ValueError(
                    f'SurfaceData marginals[{side!r}] has {len(marginal)} '
                    f'values, expected {len(values)}')


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
        'linear' or 'log', the reading this axis is drawn on by default.
        Statistical meaning, never styling.
    scales : tuple of str, optional
        Every scale this axis may be read on, ``scale`` among them. Omitted
        means the axis has one honest reading, and ``__post_init__`` fills
        ``(scale,)``, so a consumer never handles ``None``: a singleton is
        "fixed", anything longer is a reading the reader may choose. Which
        readings a quantity admits is a fact about the quantity, not about
        the drawing: a log reading of a heavy tail is meaningful, a log
        reading of a distortion's unit square is not.
    suggested_range : tuple of float, optional
        The (lo, hi) window the emitter computed from the data (a quantile
        crop, a unit interval). A suggestion: renderers may pan or zoom,
        but the initial view honors it.
    full_range : tuple of float, optional
        The whole data extent, offered as the alternative reading to
        ``suggested_range``, which it therefore requires. Presence is the
        declaration: an axis carrying both offers the zoom-out, one
        carrying only ``suggested_range`` has no other honest reading, and
        a chart whose window *is* its meaning (the unit square) sets only
        the one. Carried as numbers rather than as a flag because the
        extent of a log axis with an exact zero, or of a survival curve
        floored at ``LOG_FLOOR``, is not the naive min and max of the
        series, and working that out is emitter knowledge.
    kind : str
        'value' (continuous) or 'category' (ordinal positions).
    unit : str, optional
        One of :data:`AXIS_UNITS`; what the numbers measure.
    reciprocal_of : str, optional
        The id of the drawn probability axis this axis is the paired
        return-period reading of. Its presence is the declaration that the
        reading is on offer; the map from that axis' value to ``T`` is
        ``ChartDoc.meta['return_period_map']`` (see
        :data:`RETURN_PERIOD_MAPS`), which is 'reciprocal' unless a
        document says otherwise. The pairing is semantic: the two scales
        name the same curve, and a renderer may realize the pair as one
        axis with a twin, or as a control that redraws it. A paired axis
        sits in ``ChartDoc.axes`` and is **not** named by any panel, since
        it is an alternative reading of a drawn axis rather than a drawn
        axis of its own; both halves of that are checked in
        :meth:`ChartDoc.__post_init__`.
    complement_of : str, optional
        The id of the drawn probability axis this axis is the reflected
        reading of. Its presence is the declaration that the reading is on
        offer; the map is ``v`` to ``1 - v``, which needs no document-level
        instruction because there is only one complement. A non-exceeding
        probability reflected is the exceedance probability, so a quantile
        function read against it is the survival function, which is the
        reading a log axis exists for and the reason the pairing carries
        its own ``scales`` rather than borrowing the drawn axis'. Like
        ``reciprocal_of`` the paired axis sits in ``ChartDoc.axes`` and is
        **not** named by any panel, and an axis carries at most one of the
        two pointers: the two readings compose in the renderer, which is
        not the same thing as a chain of declarations.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """

    id: str
    label: str
    scale: str = 'linear'
    scales: tuple = None
    suggested_range: tuple = None
    full_range: tuple = None
    kind: str = 'value'
    unit: str = None
    reciprocal_of: str = None
    complement_of: str = None

    def __post_init__(self):
        if self.scale not in AXIS_SCALES:
            raise ValueError(f'unknown axis scale {self.scale!r}; '
                             f'expected one of {AXIS_SCALES}')
        if self.kind not in ('value', 'category'):
            raise ValueError(f'unknown axis kind {self.kind!r}; '
                             "expected 'value' or 'category'")
        if self.scales is None:
            object.__setattr__(self, 'scales', (self.scale,))
        else:
            _freeze_seq(self, 'scales')
            for s in self.scales:
                if s not in AXIS_SCALES:
                    raise ValueError(f'axis {self.id!r} declares unknown '
                                     f'scale {s!r}; expected one of '
                                     f'{AXIS_SCALES}')
            if self.scale not in self.scales:
                raise ValueError(
                    f'axis {self.id!r} is drawn on {self.scale!r}, which is '
                    f'not among the scales it declares, {self.scales}')
        for name in ('suggested_range', 'full_range'):
            if getattr(self, name) is not None:
                _freeze_seq(self, name)
                if len(getattr(self, name)) != 2:
                    raise ValueError(f'{name} must be (lo, hi)')
        if self.full_range is not None and self.suggested_range is None:
            # The full extent is the alternative to a window, so without a
            # window it declares a control that would do nothing.
            raise ValueError(
                f'axis {self.id!r} declares full_range with no '
                'suggested_range: the full extent is the alternative '
                'reading to a window, and is already the view without one')


@dataclass(frozen=True)
class Panel:
    """One subplot slot: a kind, its axes, and how it is read.

    Parameters
    ----------
    id : str
        Referenced by series and marks. Unique within the document.
    kind : str
        One of :data:`PANEL_KINDS`, the realization drawn by default.
    kinds : tuple of str, optional
        Every realization this panel supports, ``kind`` among them. Omitted
        means one, and ``__post_init__`` fills ``(kind,)``, exactly as
        :attr:`ChartAxis.scales` does. A z grid read flat or in relief is
        **one document declaring two realizations**, never two chart
        entries that must be kept in step by hand; a renderer picks one it
        can realize and only degrades when the panel offers none. 'xy' does
        not combine with the grid kinds: they take different payloads, so
        they are different charts and not two readings of one.
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
    invertible : bool
        The panel's two axes may be exchanged. Mechanically that is a
        transpose, but what it performs is an **inversion**: a Lee diagram
        exchanged is the distribution function, because a quantile function
        and a cdf are inverses and the drawn pairs are the same pairs.
        Declaring it says the exchange is meaningful, which is a fact about
        the quantities and not about the drawing: exchanging a density's
        axes says nothing, because mass against loss does not invert.
        A consumer that offers the reading also swaps ``read_axis``, since
        the reader still interrogates the same quantity.
    inverse_title : str, optional
        What the panel is called when its axes are exchanged, because that
        is a different picture with a name of its own ('Distribution
        function' for an inverted Lee diagram) and naming it is the
        library's job, not the renderer's. Only meaningful with
        ``invertible``; a renderer with no name to use says the title is
        inverted rather than inventing one.
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

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """

    id: str
    kind: str
    x_axis: str
    y_axis: str
    kinds: tuple = None
    z_axis: str = None
    read_axis: str = 'x'
    invertible: bool = False
    aspect: str = None
    title: str = None
    inverse_title: str = None

    def __post_init__(self):
        if self.kind not in PANEL_KINDS:
            raise ValueError(f'unknown panel kind {self.kind!r}; '
                             f'expected one of {PANEL_KINDS}')
        if self.kinds is None:
            object.__setattr__(self, 'kinds', (self.kind,))
        else:
            _freeze_seq(self, 'kinds')
            for k in self.kinds:
                if k not in PANEL_KINDS:
                    raise ValueError(f'panel {self.id!r} declares unknown '
                                     f'kind {k!r}; expected one of '
                                     f'{PANEL_KINDS}')
            if self.kind not in self.kinds:
                raise ValueError(
                    f'panel {self.id!r} is drawn as {self.kind!r}, which is '
                    f'not among the kinds it declares, {self.kinds}')
            if 'xy' in self.kinds and len(self.kinds) > 1:
                raise ValueError(
                    f'panel {self.id!r} declares {self.kinds}: an xy panel '
                    'carries curves and a grid panel carries a z grid, so '
                    'they are different charts, not two readings of one')
        if self.read_axis not in ('x', 'y'):
            raise ValueError("read_axis must be 'x' or 'y'")
        if self.aspect not in (None, 'equal'):
            raise ValueError("aspect must be 'equal' or None")
        if self.inverse_title is not None and not self.invertible:
            raise ValueError(
                f'panel {self.id!r} names an inverse_title but is not '
                'invertible, so nothing can ever use it')
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
    x_lattice, y_lattice : tuple, optional
        ``(start, step, count)`` in place of ``x`` or ``y``, for a
        coordinate that is an arithmetic progression, which in this library
        is nearly all of them: an aggregate is computed on a uniform bucket
        grid, so listing 65,536 evenly spaced numbers spells out a fact
        three numbers already state. Expand as ``start + step * i`` for
        ``i`` in ``range(count)``, which is how the grid was built, so the
        values round-trip exactly; :attr:`x_values` and :attr:`y_values` do
        it for you. Exactly one of ``x`` and ``x_lattice`` is set, and the
        same for y. An emitter uses the lattice form only where the values
        really are exactly arithmetic, so it is never an approximation.
    y2 : tuple, optional
        The second edge of a band series (the margin between F and gF, a
        min/max envelope): the series *is* the region between ``y`` and
        ``y2`` over ``x``. Same length as ``y``; only with an x/y payload.
    surface : SurfaceData, optional
        The z grid for 'heatmap' and 'surface' panels, instead of x/y.
    support : str
        One of :data:`SUPPORT_KINDS`, default 'atomic'. Whether the points
        are the whole law or samples of a function that lives between
        them. How that is *drawn* is the renderer's, and depends on how
        much room each atom gets.
    value : float, optional
        One number the series carries **as a whole**: the weight of the
        bracket a curve in an envelope cloud belongs to. It is a property
        of the series and not a styling instruction, so a renderer chooses
        how to encode it, a color ramp or an opacity or a legend entry, and
        one that has no use for it ignores it and is still correct. Reach
        for it only where the number is a fact a reader would ask about; it
        is not a channel for passing appearance through.

    Notes
    -----
    An x/y series draws on **any** panel kind. On a grid panel it is an
    overlay, drawn over the mesh in document order: the iso-total diagonals
    of a joint density, a contour trace, a reference curve. The grid panel
    itself carries exactly one surface series; an 'xy' panel carries none.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """

    name: str
    role: str
    panel_id: str
    x: tuple = None
    y: tuple = None
    x_lattice: tuple = None
    y_lattice: tuple = None
    y2: tuple = None
    surface: SurfaceData = None
    support: str = 'atomic'
    value: float = None

    @property
    def x_values(self):
        """The x payload, expanding a lattice if that is how it is carried."""
        return _expand(self.x, self.x_lattice)

    @property
    def y_values(self):
        """The y payload, expanding a lattice if that is how it is carried."""
        return _expand(self.y, self.y_lattice)

    def __post_init__(self):
        if self.support not in SUPPORT_KINDS:
            raise ValueError(f'unknown support {self.support!r}; '
                             f'expected one of {SUPPORT_KINDS}')
        carries_xy = not (self.x is None and self.y is None
                          and self.x_lattice is None and self.y_lattice is None)
        if (self.surface is None) == (not carries_xy):
            raise ValueError(
                f'series {self.name!r} must carry either x/y or surface')
        if self.surface is not None and self.y2 is not None:
            raise ValueError(f'series {self.name!r}: y2 needs an x/y payload')
        if self.surface is not None:
            return
        lengths = {}
        for side in ('x', 'y'):
            values, spec = getattr(self, side), getattr(self, f'{side}_lattice')
            if (values is None) == (spec is None):
                raise ValueError(
                    f'series {self.name!r} needs exactly one of {side} and '
                    f'{side}_lattice')
            if spec is None:
                _freeze_seq(self, side)
                lengths[side] = len(getattr(self, side))
            else:
                _freeze_seq(self, f'{side}_lattice')
                spec = getattr(self, f'{side}_lattice')
                if len(spec) != 3:
                    raise ValueError(f'series {self.name!r}: {side}_lattice '
                                     'must be (start, step, count)')
                if int(spec[2]) < 0:
                    raise ValueError(f'series {self.name!r}: {side}_lattice '
                                     'count must not be negative')
                lengths[side] = int(spec[2])
        if lengths['x'] != lengths['y']:
            raise ValueError(
                f"series {self.name!r}: len(x) = {lengths['x']} "
                f"!= len(y) = {lengths['y']}")
        if self.y2 is not None:
            _freeze_seq(self, 'y2')
            if len(self.y2) != lengths['y']:
                raise ValueError(
                    f'series {self.name!r}: len(y2) = {len(self.y2)} '
                    f"!= len(y) = {lengths['y']}")


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
        A de-emphasized mark, drawn as a scale to read the panel against
        rather than as an answer, versus a full-weight one (the mean).
        Semantic emphasis, not a color choice. No shipped emitter sets it
        at present, and a reader must still honor it.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
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
    r"""The chart document: the versioned IR a chart emitter returns.

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
        Chart-level semantic facts that are not drawable objects and belong
        to no single axis, panel or series: ``return_period_map`` (see
        :data:`RETURN_PERIOD_MAPS`), ``ordinate`` (whether a severity panel
        drew a pdf or a mass), ``basis`` and ``bases_available`` (which
        stage of a reinsurance program is drawn, and which exist).
        JSON-representable values only.
    tex : dict
        Plain string to its typeset form, for **every** human-facing string
        in the document: ``{'ǧ(s)': r'$\check g(s)$', 's': 's'}``. The
        lookup is total, a plain word mapping to itself, so a missing entry
        is an emitter bug rather than a statement that a string has no
        typeset form. Build it with :func:`complete_tex` rather than by
        hand. A renderer that can typeset looks a string up; one that
        cannot ignores the field and is still correct. Values are stored
        exactly as matplotlib consumes them, delimiters included, so a
        renderer never guesses where the math starts and an emitter can mix
        text and math in one string. Keyed by string value rather than by
        field, so one entry covers a name, an axis label and a title that
        read alike.
    ir_version : int
        Always :data:`CHART_IR_VERSION` for documents this build writes.
    generator : str, optional
        Producer tag, excluded from the content hash.
    hash : str, optional
        The 12-hex content hash, stamped by :func:`stamp`, excluded from
        the hashed form so stamping does not perturb the digest.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """

    name: str
    title: str = None
    panels: tuple = ()
    axes: tuple = ()
    series: tuple = ()
    marks: tuple = ()
    meta: dict = field(default_factory=dict)
    tex: dict = field(default_factory=dict)
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
        object.__setattr__(self, 'tex', dict(self.tex))
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
        drawn = {ax for p in self.panels
                 for ax in (p.x_axis, p.y_axis, p.z_axis) if ax is not None}
        # Both pointer fields name the same shape of thing, an undrawn axis
        # pointing at a drawn one, so the checks are written once over the
        # pair rather than twice down the file.
        seen_pairs = {}
        for a in self.axes:
            declared = [p for p in _PAIRED_READINGS
                        if getattr(a, p) is not None]
            if not declared:
                continue
            if len(declared) > 1:
                # Two pointers on one axis names a chained reading, and the
                # readings do not chain: they compose in the renderer, each
                # off the drawn axis (see RETURN_PERIOD_MAPS).
                raise ValueError(
                    f'axis {a.id!r} declares both '
                    f'{" and ".join(sorted(declared))}: an axis is one '
                    'alternative reading of one drawn axis, and the '
                    'readings compose in the renderer rather than by '
                    'chaining declarations')
            pointer = declared[0]
            target = getattr(a, pointer)
            if target not in axis_ids:
                raise ValueError(
                    f'axis {a.id!r} {pointer} unknown axis {target!r}')
            # A paired reading is an alternative to a drawn axis, so it
            # points at one and is not one itself. Both halves matter: a
            # pair drawn as its own panel axis would put the same curve on
            # screen twice, and a pair pointing at nothing on screen names
            # a reading of a quantity nobody can see.
            if a.id in drawn:
                raise ValueError(
                    f'axis {a.id!r} is the paired reading of '
                    f'{target!r} and must not be named by a panel: '
                    'it is an alternative reading of a drawn axis, not a '
                    'drawn axis of its own')
            if target not in drawn:
                raise ValueError(
                    f'axis {a.id!r} is the paired reading of '
                    f'{target!r}, which no panel draws')
            # A renderer looks a pairing up and takes the first match, so a
            # second one for the same axis and the same reading is a
            # declaration nobody can ever reach.
            if (pointer, target) in seen_pairs:
                raise ValueError(
                    f'axes {seen_pairs[(pointer, target)]!r} and {a.id!r} '
                    f'both declare {pointer}={target!r}: a renderer reads '
                    'the first, so the second is a reading nothing can '
                    'reach')
            seen_pairs[(pointer, target)] = a.id
        rp_map = self.meta.get('return_period_map')
        if rp_map is not None and rp_map not in RETURN_PERIOD_MAPS:
            raise ValueError(
                f'unknown return_period_map {rp_map!r}; expected one of '
                f'{RETURN_PERIOD_MAPS}')
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
# reader keys on, plus any field whose **default is an instruction**.
# Everything else appears only when it differs from its default, so adding a
# genuinely optional field never changes existing hashes.
#
# That last rule has one trap, and ``support`` fell into it (fixed 1.0.0a228).
# Omit-at-default is right when absent means the neutral thing: an absent
# ``scale`` is linear, an absent ``read_axis`` is x, an absent ``faint`` is
# full weight, and a reader that ignores all three still draws an honest
# picture. It is wrong when the default is the *active* case. ``support``
# defaults to ``'atomic'``, which tells a renderer to draw stems or steps and
# never a slope the law does not have, so omitting it shipped that instruction
# to nobody: ``'continuous'`` (a severity pdf, a distortion) survived
# serialization and every discretized density arrived bare, which reads as
# "nothing special" and draws as a plain line. The polarity was inverted
# against the meaning. Before adding a field here, ask which of its values a
# consumer must act on; if that value is the default, it belongs in this list.
#
# ``scales`` and ``kinds`` are here for the same reason, from the other
# side: their default is filled in rather than omitted, so a consumer reads
# "this axis has one reading" as a fact rather than inferring it from an
# absence, and the singleton case costs a handful of bytes.
_ALWAYS = {
    ChartDoc: ('ir_version', 'name'),
    Panel: ('id', 'kind', 'kinds', 'x_axis', 'y_axis'),
    ChartAxis: ('id', 'label', 'scales'),
    ChartSeries: ('name', 'role', 'panel_id', 'support'),
    Mark: ('panel_id', 'orient', 'at'),
    SurfaceData: ('x', 'y', 'z'),
    SurfaceZBlock: ('dtype', 'data', 'order'),
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

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    skip = () if include_hash else ('hash', 'generator')
    return _canonical(doc, skip=skip)


def _load_member(cls, data):
    """One canonical dict to its dataclass, with a readable error on a stray field.

    ``cls(**data)`` would report an unknown field as a bare ``TypeError``
    naming a keyword argument, which reads as a caller mistake rather than as
    what it is, a document this build cannot honor.
    """
    if not isinstance(data, dict):
        raise TypeError(
            f'{cls.__name__}: expected a dict, got {type(data).__name__}')
    unknown = sorted(set(data) - {f.name for f in dataclasses.fields(cls)})
    if unknown:
        raise ValueError(
            f'{cls.__name__} carries unknown field(s) {unknown}. The document '
            f'declares an ir_version this build reads ({CHART_IR_VERSION}), '
            'so it was written by a build that means something different by '
            'that number.')
    return cls(**data)


def _load_surface(data):
    """One canonical surface dict, rebuilding its nested z block if it has one."""
    if isinstance(data.get('z_block'), dict):
        data = {**data,
                'z_block': _load_member(SurfaceZBlock, data['z_block'])}
    return _load_member(SurfaceData, data)


def _load_series(data):
    """One canonical series dict, rebuilding its nested surface if it has one."""
    if isinstance(data, dict) and isinstance(data.get('surface'), dict):
        data = {**data, 'surface': _load_surface(data['surface'])}
    return _load_member(ChartSeries, data)


def load_chart_doc(d):
    """Rebuild a chart document from its canonical dict: the way back.

    The inverse of :func:`canonical_dict`, so a consumer holding a fetched
    document can hand it to :func:`aggregate.plots.plot_chartdoc`, which is
    generic over any :class:`ChartDoc` and needs to know nothing about the
    object that emitted it.

    Parameters
    ----------
    d : dict
        A canonical dict, or what ``json.loads`` makes of
        :func:`canonical_json` output. Panels, axes, series and marks arrive
        as plain dicts and are rebuilt as their dataclasses; tuples arrive as
        lists and are frozen by each class's ``__post_init__``.

    Returns
    -------
    ChartDoc
        Equal in content to the document that was written, and equal in hash.

    Raises
    ------
    ValueError
        Through the dataclasses' own validation (an ``ir_version`` this build
        does not read, a panel naming an axis the document does not carry, a
        series carrying neither an x/y payload nor a surface), or when a
        member carries a field this build does not know.

    Notes
    -----
    **The round trip is the contract**, and it is exact::

        doc_hash(load_chart_doc(canonical_dict(doc))) == doc.hash

    **Why this belongs to the library rather than to each client.**
    :func:`canonical_dict` drops any field equal to its default, so a reader
    is a statement about what those defaults are: panel ``read_axis`` and
    ``aspect`` never reach the wire, axis ``kind`` never reaches it, and
    ``invertible`` appears only where it is true. A client writing its own
    reader writes this build's default table down a second time, and a
    default that moves here then moves silently over there.

    ``ir_version`` is validated in :meth:`ChartDoc.__post_init__`, which makes
    the reader the one place a wire document's version is negotiated. A
    consumer assembling dataclasses by hand may pass the field through or may
    not, and the one that does not is the one that accepts a version it cannot
    read without noticing.

    :class:`SurfaceData`, and :class:`SurfaceZBlock` inside it, are the only
    nested payloads, so a reader that forgets them fails on exactly the
    bivariate documents, which are both the largest and the least often
    exercised.

    **The hash is what makes this checkable at all.** The ``agg`` chart runs to
    megabytes of canonical JSON, most of it explicit curve coordinates, and a
    quiet coercion in a hand written reader, a tuple that came back a list or
    a float that came back a string, is not something anyone catches by
    looking at the picture.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    if not isinstance(d, dict):
        raise TypeError(
            f'load_chart_doc: expected a canonical dict, got '
            f'{type(d).__name__}')
    rest = dict(d)
    members = {
        'panels': tuple(_load_member(Panel, p) for p in rest.pop('panels', ())),
        'axes': tuple(_load_member(ChartAxis, a) for a in rest.pop('axes', ())),
        'series': tuple(_load_series(s) for s in rest.pop('series', ())),
        'marks': tuple(_load_member(Mark, m) for m in rest.pop('marks', ())),
    }
    return _load_member(ChartDoc, {**rest, **members})


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

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    data = canonical_dict(doc, include_hash=include_hash)
    text = json.dumps(data, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)
    return text.encode('utf-8')


def doc_hash(doc):
    """Return the 12-hex-character sha256 content hash of a document.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    return hashlib.sha256(
        canonical_json(doc, include_hash=False)).hexdigest()[:12]


def human_strings(doc):
    """Every human-facing string a document exposes, in document order.

    The document's title, each panel's title, each axis label, each series
    name and each mark label: the strings a renderer puts in front of a
    reader, and therefore exactly the strings :attr:`ChartDoc.tex` must
    cover. Deduplicated, because the map is keyed by string value: one
    entry covers a name and an axis label that read alike.

    Parameters
    ----------
    doc : ChartDoc

    Returns
    -------
    tuple of str

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    out = []
    for text in ([doc.title] + [p.title for p in doc.panels]
                 + [p.inverse_title for p in doc.panels]
                 + [a.label for a in doc.axes]
                 + [s.name for s in doc.series]
                 + [m.label for m in doc.marks]):
        if text and text not in out:
            out.append(text)
    return tuple(out)


def complete_tex(doc, typeset=None):
    """Return ``doc`` with a total :attr:`ChartDoc.tex` map.

    The emitter supplies the strings that have a distinct typeset form and
    this fills the rest with themselves, which is what makes the lookup
    total without every emitter writing out a line of identities. A
    ``typeset`` key the document does not expose is an error rather than a
    harmless extra: it is a typo or a string that stopped being emitted,
    and either way a renderer would go on drawing the plain form while the
    map claimed otherwise.

    Parameters
    ----------
    doc : ChartDoc
        The document to complete. Any existing ``tex`` entries are kept.
    typeset : dict, optional
        Plain string to typeset form, for the strings that have one, stored
        exactly as matplotlib consumes them (delimiters included).

    Returns
    -------
    ChartDoc
        A copy whose ``tex`` covers :func:`human_strings` exactly.

    Raises
    ------
    ValueError
        For a ``typeset`` key the document does not expose.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    forms = dict(doc.tex)
    forms.update(typeset or {})
    strings = human_strings(doc)
    unknown = sorted(set(forms) - set(strings))
    if unknown:
        raise ValueError(
            f'typeset entries {unknown} name strings document {doc.name!r} '
            'does not expose; the map is keyed by the string a reader sees')
    return dataclasses.replace(
        doc, tex={text: forms.get(text, text) for text in strings})


def stamp(doc, generator=None):
    """Return a copy of ``doc`` with ``hash`` (and ``generator``) set.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    changes = {'hash': doc_hash(doc)}
    if generator is not None:
        changes['generator'] = generator
    return dataclasses.replace(doc, **changes)
