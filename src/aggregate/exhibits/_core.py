"""``aggregate.exhibits._core`` -- the exhibit machinery, class agnostic.

.. warning::

   **Provisional, in the sense of PEP 411**, along with the rest of
   :mod:`aggregate.exhibits`: not part of the 1.0 API contract, and subject to
   change in a minor release with no deprecation period. See
   :doc:`/3_reference/3_x_API_Stability`.

The base layer of the exhibits package, the analogue of ``plots._style``:
:class:`Perspective`, :class:`Exhibit`, the singledispatch registry, the
frame and IR stages, and the translation helpers that more than one class
module shares. It knows **nothing** about ``Aggregate``, ``PnL`` or any other
domain class, which is what keeps the package free of import cycles: the
per-class modules import this one, never the reverse.

To edit an exhibit you almost never come here. A class's treatment lives in
its own module (``_aggregate.py``, ``_portfolio.py``, ``_pnl.py``,
``_bivariate.py``, ``_distortion.py``), and the passthrough declarations are
the manifest at the foot of ``__init__.py``.

Design (see ``dev/plan-exhibits.md``):

* :class:`Perspective` names who is looking at the numbers: ``RAW`` (the
  underlying frame passed through), ``INSURED`` (buyer of insurance),
  ``INSURER`` (seller of insurance, buyer of reinsurance; the cedent), and
  ``REINSURER`` (seller of reinsurance). At 1.0 only ``RAW`` and ``INSURER``
  are implemented; ``INSURED`` and ``REINSURER`` are declared vocabulary with
  no registrations yet, so that adding them later does not churn the enum.
  Declared is not the same as promised: like everything in this package the
  enum is provisional, and the reinsurer semantics review may yet rename or
  respell a member.
* Each exhibit (``summary``, ``tail``, ``stats``, ``validation``, ``reins``,
  ``economic``, ``economic_ratios``, ``dependency``, ...) is a generic function
  dispatching on the object's type. Registration is open: app or user code
  may register new types with ``summary.register(MyType)``, supplying a
  *frames builder* ``f(obj) -> [(block_name, DataFrame, spec_kwargs), ...]``.
* The INSURER default rule: **INSURER equals RAW unless an override is
  registered** for the (exhibit, type) pair. The override hook
  (``summary.insurer.register(MyType)``) receives the raw blocks and returns
  translated blocks. Light overrides add a caption or row flags; heavy ones
  may rebuild the presentation entirely. The rule keeps the generic path
  total: a new exhibit is useful the moment its raw registration exists.
* The frame stage (:func:`exhibit_frames`) is pure pandas: testable, and
  usable on its own by a caller that wants the numbers rather than a table.
  Only :func:`build_exhibit` and :meth:`Exhibit.to_payload` touch
  greater_tables, through an import at the point of use, so the frame stage
  never pays for it.

All served frames pass through the host's ``LabeledMixin._relabel`` (honoring
``use_labels`` and ``renamer``); exhibit titles use ``_title_name``.
"""

import functools
import hashlib
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from ..constants import Validation

__all__ = [
    'Perspective', 'Exhibit', 'EXHIBITS',
    'available_exhibits', 'exhibit_frames', 'build_exhibit',
    'summary', 'tail', 'stats', 'validation', 'reins',
    'economic', 'economic_ratios', 'economic_waterfall', 'dependency',
    'pricing_calibrate', 'pricing_stand_alone', 'pricing_allocate',
    'pricing_evaluate', 'register_simple_exhibit',
]


#: Return periods emphasized as capital anchors in the ``tail`` exhibit:
#: 1 in 200 (99.5%, Solvency II) and 1 in 250 (99.6%, US capital adequacy).
CAPITAL_ANCHOR_PERIODS = (200.0, 250.0)

#: Raw noncentral moment measures dropped from the INSURER stats view (the
#: raw perspective keeps every row of the canonical 26 row store).
RAW_MOMENT_MEASURES = ('ex1', 'ex2', 'ex3')

#: Every served block carries its raw values beside its formatted strings.
#: The formatted string is this library's reading of a number and the raw
#: value is the number, and a document that ships only the reading has
#: destroyed information no consumer can recover: it cannot be sorted
#: numerically, downloaded at full precision, or rendered interactively at
#: all. See :func:`build_exhibit` for the measured payload cost. A library
#: default rather than a caller option, on the same reasoning that keeps
#: renderer passthrough out of the chart IR: presentation is the consumer's,
#: and it needs the numbers to do it.
INCLUDE_RAW = True

#: Rows a served block carries before it truncates, matching greater_tables'
#: own default. Unlike :data:`INCLUDE_RAW` this **is** the caller's to set
#: (``build_exhibit(..., max_rows=...)``, ``None`` for the whole frame),
#: because how much of a frame to ship is a question about the request and
#: not about the numbers: a preview pane and a download want different
#: answers to it, and neither is more correct. Truncating is not a silent
#: loss, since greater_tables records it in the block's ``notes`` as
#: "Showing first N of M rows", which is why a cap is honest where dropping
#: the raw values was not. No exhibit reaches it today; the longest block
#: measured is 17 rows.
MAX_ROWS = 200

#: Per measure formats for the INSURER views, as greater_tables format sugar
#: (author, 2026-08-05). Applies wherever a measure **is a column**: the
#: summary card and the P&L ledger. It cannot apply to the canonical moment
#: store, where measures run *down* a column, and that asymmetry is the open
#: question recorded in ``dev/plan-exhibits.md``.
#:
#: ``Skew`` is asked for as ``.3g``, three significant figures, which
#: greater_tables sugar does not express: its kinds are ``f`` / ``d`` / ``%``
#: / ``e`` / ``s``, with no ``g``. ``.3f`` is the nearest available and reads
#: the same for the skews actually seen (it differs only in trailing zeros on
#: large values). Switch the constant the day greater_tables grows a ``g``
#: kind; nothing else needs to change.
MEASURE_FORMATS = {
    'CV': '.1%',
    'Skew': '.3f',
}

#: Validation failure flags that emphasize the ``Sev`` row of a
#: Freq / Sev / Agg validation frame, and those that emphasize ``Agg``
#: (aliasing is an aggregate level symptom). ``Freq`` is PGF exact and
#: never flags.
_SEV_FAILURES = Validation.SEV_MEAN | Validation.SEV_CV | Validation.SEV_SKEW
_AGG_FAILURES = (Validation.AGG_MEAN | Validation.AGG_CV
                 | Validation.AGG_SKEW | Validation.ALIASING)


class Perspective(Enum):
    """Who is looking at the numbers.

    ``RAW`` serves the underlying frame with **no business translation**: no
    row emphasis, no dropped rows, no rearrangement. Since 1.0.0a226 it does
    carry a caption saying what the frame is, and the column formats for
    units the frame cannot carry itself (a ``CV`` reads as a percentage).
    Those describe the table rather than interpret it, and a table with no
    prose at all only pushes the writing of that prose to whoever displays
    it. ``INSURED`` is the policyholder (buyer of insurance). ``INSURER`` is
    the seller of insurance and buyer of reinsurance (the cedent, the object
    holder on the reins exhibits), and is where the business reading lives:
    what the frame *means*, over ``RAW``'s what it *is*. ``REINSURER`` is the
    seller of reinsurance, a sign and label flip relative to the insurer
    (retro, where the reinsurer buys, is parked).

    Only ``RAW`` and ``INSURER`` are implemented at 1.0. ``INSURED`` and
    ``REINSURER`` are declared here so the enum does not churn when their
    implementations arrive, which is a courtesy to readers rather than a
    promise: this module is provisional, and the reinsurer semantics review
    may yet rename or respell a member.

    Notes
    -----
    **The contract between the two implemented members.** A ``RAW`` block is
    exactly one public frame: one block per frame, named for the attribute
    holding it, in the frame's own orientation, no split and no dropped rows.
    ``INSURER`` is the only perspective that may restructure, and
    ``[Perspective-May-Restructure]`` says it may change the **block list**
    and not merely each block's content. So the served block names are a
    property of the (exhibit, perspective) pair: ``economic_ratios`` serves
    two blocks raw and three under insurer, and a client that assumed parity
    would be wrong about that today. Read the block list off
    :attr:`Exhibit.meta`'s ``blocks`` rather than assuming it.

    A ``RAW`` block with no public frame behind it is a **missing frame**,
    not a licensed exception: the exhibit layer is not a second place where
    frames are invented, since a frame invented here is one a notebook reader
    cannot reach.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """

    RAW = 'raw'
    INSURED = 'insured'
    INSURER = 'insurer'
    REINSURER = 'reinsurer'


#: Perspectives actually served at 1.0 (plan decision 5).
_IMPLEMENTED_PERSPECTIVES = (Perspective.RAW, Perspective.INSURER)


def _resolve_perspective(perspective):
    """Coerce a string or ``Perspective`` to the enum, or raise ``ValueError``."""
    if isinstance(perspective, Perspective):
        return perspective
    try:
        return Perspective(str(perspective).lower())
    except ValueError:
        values = ', '.join(p.value for p in Perspective)
        raise ValueError(
            f'unknown perspective {perspective!r}; expected one of: {values}') from None


@dataclass(frozen=True)
class Exhibit:
    """One built exhibit: IR blocks plus its envelope fields.

    Attributes
    ----------
    ir_blocks : tuple
        greater_tables ``TableDoc`` instances, one per block, hash stamped.
    name : str
        The exhibit registry name (``'summary'``, ``'tail'``, ...).
    title : str
        Display title, ``'<exhibit title>: <object _title_name>'``.
    perspective : Perspective
        The perspective the blocks were built under.
    meta : dict
        Envelope metadata: object identity (``kind``, ``object``, ``label``),
        block names, per block captions and source frame names.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """

    ir_blocks: tuple
    name: str
    title: str
    perspective: Perspective
    meta: dict

    @property
    def hash(self):
        """Deterministic 12 hex digest over the block ``doc_hash`` values.

        sha256 of the concatenated per block hashes, truncated to 12 hex
        characters; the app uses it as the exhibit ETag.
        """
        joined = ''.join(doc.hash or '' for doc in self.ir_blocks)
        return hashlib.sha256(joined.encode('ascii')).hexdigest()[:12]

    def to_payload(self):
        """Serialize to the app envelope: canonical dict per block.

        Returns
        -------
        dict
            ``{name, title, perspective, meta, blocks, hash}`` where
            ``blocks`` is a list of greater_tables ``canonical_dict`` forms
            (byte deterministic under a sorted key JSON writer).
        """
        gt = _import_greater_tables()
        return {
            'name': self.name,
            'title': self.title,
            'perspective': self.perspective.value,
            'meta': self.meta,
            'blocks': [gt.canonical_dict(doc) for doc in self.ir_blocks],
            'hash': self.hash,
        }


def _import_greater_tables():
    """Import greater_tables at the point of use.

    A plain dependency since 1.0.0a229, so this normally cannot fail. It stays
    lazy for two reasons that outlive the packaging question: the frame stage
    is pure pandas and must not pay the import, and a broken or partial
    install should say what is wrong here rather than at ``import aggregate``.
    """
    try:
        import greater_tables
    except ImportError as exc:
        raise ImportError(
            'greater_tables is required to build exhibit IR blocks. It is a '
            'dependency of aggregate (greater_tables>=6), so this means a '
            'broken environment: reinstall with '
            '`uv sync --all-extras` or `pip install -U aggregate`.') from exc
    return greater_tables


# --- generic exhibit machinery ----------------------------------------------

def _make_exhibit_function(name, title, doc):
    """Manufacture one generic exhibit function.

    The public callable ``f(obj, perspective=Perspective.RAW) -> Exhibit``
    wraps two ``singledispatch`` registries exposed as attributes:

    * ``f.frames`` and its alias ``f.register``: the per type raw frames
      builder ``builder(obj) -> [(block_name, DataFrame, spec_kwargs), ...]``.
      The base implementation raises ``NotImplementedError`` naming the type.
    * ``f.insurer``: the per type INSURER override hook
      ``hook(obj, blocks) -> blocks``, default identity (the INSURER default
      rule: INSURER equals RAW unless an override is registered).

    ``f.title`` carries the static display title.
    """

    @functools.singledispatch
    def frames(obj):
        raise NotImplementedError(
            f'exhibit {name!r} is not implemented for type {type(obj).__name__}')

    @functools.singledispatch
    def insurer(obj, blocks):
        return blocks

    def fn(obj, perspective=Perspective.RAW, *, max_rows=MAX_ROWS):
        return build_exhibit(obj, name, perspective, max_rows=max_rows)

    fn.__name__ = name
    fn.__qualname__ = name
    fn.__doc__ = doc
    fn.title = title
    fn.frames = frames
    fn.insurer = insurer
    fn.register = frames.register
    fn.dispatch = frames.dispatch
    return fn


def _has_frames(fn, obj):
    """True when ``obj``'s type (via MRO) has a registered frames builder."""
    return fn.frames.dispatch(type(obj)) is not fn.frames.registry[object]


def _updated(obj):
    """True when the object's realized grid exists (post ``update``).

    Checks the first marker the object carries and stops there:
    ``agg_density`` (Aggregate, a plain attribute that is None before
    ``update``) ahead of ``density_df``, because on an Aggregate the
    ``density_df`` *property* raises before ``update`` rather than
    returning None. Portfolio carries ``density_df`` as a None initialized
    attribute; ``BivariateAggregate.density_df`` also raises before
    ``update``, so a raising accessor reads as not updated.
    """
    sentinel = object()
    for attr in ('agg_density', 'density_df'):
        try:
            value = getattr(obj, attr, sentinel)
        except Exception:
            return False
        if value is not sentinel:
            return value is not None
    return False


def _perspectives_always(obj):
    """Perspectives for an exhibit with no structural gate."""
    return list(_IMPLEMENTED_PERSPECTIVES)


def _perspectives_updated(obj):
    """Perspectives for an exhibit whose source frame needs the realized grid."""
    return list(_IMPLEMENTED_PERSPECTIVES) if _updated(obj) else []


def _has_reinsurance(obj):
    """True when the object (or any unit of a portfolio) carries a cession."""
    if getattr(obj, 'occ_reins', None) is not None \
            or getattr(obj, 'agg_reins', None) is not None:
        return True
    agg_list = getattr(obj, 'agg_list', None)
    if agg_list is not None:
        return any(_has_reinsurance(a) for a in agg_list)
    return False


def _perspectives_reins(obj):
    """Perspectives for the reins exhibit: cession present and grid realized."""
    return list(_IMPLEMENTED_PERSPECTIVES) \
        if _has_reinsurance(obj) and _updated(obj) else []


def _perspectives_tower(obj):
    """Perspectives for an exhibit that needs a multi-step P&L walk.

    A single group P&L has one margin row and no walk to draw, so the
    exhibit reports itself unavailable and the app grays the chip out
    rather than rendering a one row table.
    """
    return list(_IMPLEMENTED_PERSPECTIVES) if getattr(obj, '_tower', False) \
        else []


def _perspectives_allocation(obj):
    """Perspectives for ``pricing.allocate``: is there one premium to split?

    Two shapes qualify, and the second has a basis condition the first does
    not. A ``Portfolio`` calibration always splits its target across the units
    of the book. An ``Aggregate`` calibration splits across the halves of an
    occurrence program, which needs the program to exist **and** the fit to
    have been struck on gross: a set calibrated on net has no gross premium to
    allocate, so the gate is structural rather than a preference.

    An aggregate with no cession has one distribution and nothing to split,
    which is not a degenerate allocation but the absence of one; its story is
    the stand-alone leaf.

    The occurrence arm is described here and not yet honored: the frame it
    would serve arrives with the natural allocation exhibit, and a predicate
    that says available before a builder can answer is worse than one that
    waits.
    """
    source = getattr(obj, '_source', None)
    if source is None:
        return []
    if getattr(source, 'agg_list', None) is not None:
        return list(_IMPLEMENTED_PERSPECTIVES)
    return []


def _perspectives_sharpen(obj):
    """Perspectives for the sharpen exhibit: a probe has run and left an audit.

    Unlike the other diagnostics this is not gated on ``update``: a probe
    implies one. The audit is absent until :meth:`Aggregate.sharpen` is
    called, and an object that has never been probed has nothing to show, so
    the exhibit reports itself unavailable rather than serving an empty grid.
    """
    return list(_IMPLEMENTED_PERSPECTIVES) \
        if getattr(obj, 'sharpen_df', None) is not None else []


def available_exhibits(obj):
    """List the exhibits this object can serve, with their perspectives.

    Derived from the singledispatch registries (an MRO hit means the type
    has a frames builder) plus each exhibit's availability predicate, so it
    cannot go stale as registrations are added. Needs no greater_tables
    import.

    Parameters
    ----------
    obj : object
        Any object; unregistered types return an empty list.

    Returns
    -------
    list of (str, list of Perspective)
        ``(name, perspectives)`` pairs in registry order; an exhibit whose
        predicate fails (for example ``reins`` without reinsurance, ``tail``
        before ``update``) is omitted.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    out = []
    for name, (fn, perspectives_fn) in EXHIBITS.items():
        if not _has_frames(fn, obj):
            continue
        perspectives = list(perspectives_fn(obj))
        if perspectives:
            out.append((name, perspectives))
    return out


def exhibit_frames(obj, name, perspective=Perspective.RAW):
    """The pure pandas frame stage of an exhibit: no greater_tables needed.

    Looks up the exhibit, dispatches the raw frames builder on ``type(obj)``,
    applies the INSURER override hook when asked for that perspective, and
    relabels every frame through the host's ``LabeledMixin._relabel`` so
    ``use_labels`` and ``renamer`` are honored.

    Parameters
    ----------
    obj : object
        A registered first class object (``Aggregate``, ``Portfolio``, ...).
    name : str
        Exhibit registry name; ``KeyError`` on an unknown name, listing the
        registry.
    perspective : Perspective or str, default Perspective.RAW
        ``NotImplementedError`` for the perspectives without a 1.0
        implementation (``insured``, ``reinsurer``).

    Returns
    -------
    list of (str, pandas.DataFrame, dict)
        ``(block_name, frame, spec_kwargs)`` triples; ``spec_kwargs`` are
        greater_tables ``TableSpec`` keyword arguments (caption, row_flags,
        formatters, ...), plain data so this stage stays GT free.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    try:
        fn, perspectives_fn = EXHIBITS[name]
    except KeyError:
        raise KeyError(
            f'unknown exhibit {name!r}; known exhibits: '
            + ', '.join(EXHIBITS)) from None
    perspective = _resolve_perspective(perspective)
    builder = fn.frames.dispatch(type(obj))
    if builder is fn.frames.registry[object]:
        builder(obj)  # raises NotImplementedError naming the type
    supported = perspectives_fn(obj)
    if not supported:
        raise ValueError(
            f'exhibit {name!r} is not available for {type(obj).__name__} '
            f'{getattr(obj, "name", "")!r}: its availability predicate failed '
            '(missing update() grid, or required structure such as reinsurance)')
    if perspective not in supported:
        implemented = ', '.join(p.value for p in supported)
        raise NotImplementedError(
            f'perspective {perspective.value!r} is not implemented for exhibit '
            f'{name!r} at 1.0; implemented: {implemented}')
    blocks = builder(obj)
    if perspective is Perspective.INSURER:
        blocks = fn.insurer.dispatch(type(obj))(obj, blocks)
    relabel = getattr(obj, '_relabel', None)
    if relabel is not None:
        blocks = [(bname, relabel(df), kw) for bname, df, kw in blocks]
    return blocks


def build_exhibit(obj, name, perspective=Perspective.RAW, *,
                  max_rows=MAX_ROWS):
    """Build an :class:`Exhibit`: frame stage, then greater_tables IR per block.

    The only greater_tables import site besides :meth:`Exhibit.to_payload`.

    Parameters
    ----------
    obj : object
        A registered first class object.
    name : str
        Exhibit registry name.
    perspective : Perspective or str, default Perspective.RAW
    max_rows : int or None, default :data:`MAX_ROWS`
        Rows each block carries before it truncates; ``None`` ships the whole
        frame. The one presentation question the caller answers rather than
        the frame stage, because how much of a frame to ship belongs to the
        request: a preview pane and a full download want different answers
        and neither is more correct. It wins over a block's own kwargs, which
        no block sets. A truncated block says so in its ``notes``.

    Returns
    -------
    Exhibit
        Frozen; ``ir_blocks`` hold one hash stamped ``TableDoc`` per block.

    Notes
    -----
    **Every block carries its raw values** (:data:`INCLUDE_RAW`), so a cell
    arrives as ``{'text': '17.50', 'raw': 17.5000001}`` rather than as the
    string alone. It is a library default and not a caller option, because a
    document that has dropped its numbers cannot be sorted, downloaded at
    full precision, or drawn by an interactive renderer at all, and no
    consumer can put back what the document threw away. The formatted string
    is this library's reading of the number; the raw value is the number, and
    a served document owes both. Alongside them each column already carries a
    machine readable format spec, so a client can render the reading, restate
    it, or ignore it.

    Measured across every exhibit on an ``Aggregate`` and a ``PnL``, carrying
    the numbers costs about 1.5x the payload (worst case 2x, on the 26 row
    moment store) on documents of a few kilobytes.

    A frame builder may still pass ``include_raw`` in its own block kwargs
    and win, since the default is applied under them rather than over them.
    Nothing does, and anything that did would be opting its consumers out of
    interactivity, so it wants a comment saying why.

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    perspective = _resolve_perspective(perspective)
    blocks = exhibit_frames(obj, name, perspective)
    gt = _import_greater_tables()
    fn, _ = EXHIBITS[name]
    ir_blocks = []
    captions = {}
    for block_name, df, kw in blocks:
        ir_blocks.append(gt.build(df, gt.TableSpec(
            **{'include_raw': INCLUDE_RAW, **kw, 'max_rows': max_rows})))
        if kw.get('caption'):
            captions[block_name] = kw['caption']
    title_name = getattr(obj, '_title_name', type(obj).__name__)
    meta = {
        'kind': type(obj).__name__,
        'object': getattr(obj, 'name', None),
        'label': getattr(obj, 'label', None),
        'blocks': [block_name for block_name, _, _ in blocks],
        'captions': captions,
    }
    return Exhibit(ir_blocks=tuple(ir_blocks), name=name,
                   title=f'{fn.title}: {title_name}',
                   perspective=perspective, meta=meta)


# --- the generic exhibit functions ------------------------------------------

summary = _make_exhibit_function(
    'summary', 'Summary',
    """At a glance risk view: moments and key percentiles.

    Source frame ``summary_df``, registered for all five first class
    classes. RAW is the frame passed through; on Aggregate and Portfolio
    INSURER adds the business caption and total / subtotal row flags, while
    PnL, Distortion and BivariateAggregate serve the raw frame under both
    perspectives (no override registered; the PnL card framing is the
    [Exhibits-PnL-Translation] phase, behind its author gate).

    Parameters
    ----------
    obj : object
        A registered first class object.
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)

tail = _make_exhibit_function(
    'tail', 'Return periods',
    """Return period / exceedance exhibit (VaR, TVaR, xsVaR, leverage).

    Source frame ``tail_df``; available only after ``update`` (the ladder
    reads the realized grid). INSURER emphasizes the 1 in 200 and 1 in 250
    capital anchor rows and flags the portfolio total block.

    Parameters
    ----------
    obj : object
        A registered first class object.
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)

stats = _make_exhibit_function(
    'stats', 'Statistics',
    """Canonical moment store exhibit. Source frame ``stats_df``.

    Registered for all five first class classes. On Aggregate and Portfolio
    the INSURER view drops the raw noncentral moment rows (measure in
    ``ex1`` / ``ex2`` / ``ex3``); RAW keeps the full store. The other
    classes serve their ``stats_df`` unchanged under both perspectives.

    Parameters
    ----------
    obj : object
        A registered first class object.
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)

validation = _make_exhibit_function(
    'validation', 'Validation',
    """Moment vs estimate QA exhibit. Source frame ``validation_df``.

    Registered for all five first class classes. The INSURER view
    emphasizes failing rows: moment failures from the object's
    ``Validation`` flags on Aggregate and Portfolio (``Sev`` / ``Agg`` rows,
    per unit and total on a portfolio), and ``Pass == False`` check rows on
    Distortion and BivariateAggregate. The PnL audit frame has no failure
    gate and passes through unchanged.

    Parameters
    ----------
    obj : object
        A registered first class object.
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)

reins = _make_exhibit_function(
    'reins', 'Reinsurance',
    """Reinsurance exhibit: ``reins_stats_df`` and ``reins_summary_df`` blocks.

    Available only when the object (or a portfolio unit) carries a cession
    and the grid is realized. Two blocks: the layering / end to end moment
    store and the per stage cession impact summary. The INSURER view drops
    the raw noncentral moment rows from the stats block, captions both
    blocks, and flags the portfolio total block on the summary.

    Parameters
    ----------
    obj : object
        A registered first class object.
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)

economic = _make_exhibit_function(
    'economic', 'Economics',
    """P&L ledger exhibit. Source frame ``PnL.economic_df``.

    The full ledger by (Side, Label), or (Step, Side, Label) on a tower, in
    currency units with the kappa scenario ladder. RAW passes the frame
    through; the INSURER business translation (captions, footing rules, Side
    sign presentation) lands in [Exhibits-Economic-Insurer], where this
    exhibit is also renamed to ``economic`` to mirror its frame.

    Parameters
    ----------
    obj : object
        A registered first class object.
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)

economic_ratios = _make_exhibit_function(
    'economic_ratios', 'Economic ratios',
    """P&L ratio exhibit. Source frames ``PnL.economic_ratios_df`` and ``legs_df``.

    Two blocks of raw materials: the per block amounts and LR / ER / CR
    ratios, and the itemized declared legs. RAW passes both frames through;
    the INSURER arrangement into the ratio card is behind the
    [Exhibits-PnL-Translation] author gate, so INSURER currently equals RAW
    per the default rule.

    Parameters
    ----------
    obj : object
        A registered first class object.
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)

economic_waterfall = _make_exhibit_function(
    'economic_waterfall', 'Economic waterfall',
    """The margin walk: gross, through what each layer cedes, to net.

    Two blocks. **walk** carries the currency amounts, the margin at each
    step and its 1-in-100 outcome on both bases. **evaluation** carries the
    dimensionless readings: how much of the gross premium and gross margin
    each step spends, the combined ratio, margin over its own standard
    deviation, and margin over required capital on each basis.

    The point of the exhibit is the pair of 1-in-100 columns. The
    **diversified** basis is the margin conditional on the whole book landing
    at its own 1-in-100, so it **foots down the walk exactly**; the
    **standalone** basis is each step's own 1-in-100, and tail measures do not
    add, so it does not. The gap between them is the diversification benefit,
    per layer, made visible.

    Available only on a P&L with a tower: a single group ledger has one
    margin row and no walk to draw. RAW and INSURER serve the same table,
    because here the exhibit *is* the translation; there is no underlying
    frame to pass through.

    Parameters
    ----------
    obj : object
        A ``PnL`` carrying a multi step walk.
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)

dependency = _make_exhibit_function(
    'dependency', 'Dependency',
    """Bivariate dependency exhibit: ``dependency_df`` and ``axis_support_df``.

    Two blocks: joint dependence (cov / corr / tau by level) and the per
    axis realized support of the marginals. No insurer override is
    registered (no business translation exists today), so INSURER equals
    RAW per the default rule.

    Parameters
    ----------
    obj : object
        A registered first class object.
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)


pricing_calibrate = _make_exhibit_function(
    'pricing.calibrate', 'Calibrated distortions',
    """The per family calibration receipt. Source frame ``distortion_df``.

    Registered on :class:`~aggregate.results.CalibrationResult`, the object
    ``calibrate_distortions`` returns, rather than on the book it was computed
    from: a calibration is a calculation and not stored state, and dispatching
    on its result is what makes it an ordinary exhibit
    (``[Pricing-Keyed-On-Result]``). Deliberately small, one block, identical
    under both perspectives: what the fit produced, with no adjustment.

    Parameters
    ----------
    obj : CalibrationResult
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)

pricing_stand_alone = _make_exhibit_function(
    'pricing.stand_alone', 'Stand-alone pricing',
    """The calibrated families applied to each part **as a price in its own
    right**. Also on :class:`~aggregate.results.CalibrationResult`.

    Stand-alone prices the parts alone; ``pricing.allocate`` splits the whole
    across them. That is the distinction the two leaves exist to keep apart
    (``[Standalone-Prices-The-Parts, Allocate-Splits-The-Whole]``, author,
    2026-08-14), and it is a real one: net priced as its own distribution, net
    as its share of the gross premium, and net calibrated directly are three
    different numbers.

    The block list is a property of the (exhibit, perspective) pair **and** of
    what the calibrated object was, which is the fullest exercise of
    ``[Perspective-May-Restructure]`` in the package. A reinsured
    ``Aggregate`` prices every view of its cession, a ``Portfolio`` prices
    every unit against the book priced whole, and an ``Aggregate`` with
    neither has one part, which is the whole, so its stand-alone story is the
    single calibration row.

    On the reinsured Aggregate this is also the first exhibit where RAW
    carries strictly **more** rows than INSURER: RAW serves every view the
    object can price, including ``ceded``, and INSURER drops the ceded rows
    because a ceded price is the seller's reading and this perspective is the
    buyer's (``[Difference-Is-A-Perspective]``).

    Parameters
    ----------
    obj : CalibrationResult
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)

pricing_allocate = _make_exhibit_function(
    'pricing.allocate', 'Allocated pricing',
    """One calibrated premium split across the parts, adding up. Also on
    :class:`~aggregate.results.CalibrationResult`.

    The additive decomposition, whatever the parts are: the units of a book,
    or the two halves of an occurrence program. Available only where there is
    something to split it over, which is why this exhibit carries a predicate
    while its stand-alone sibling does not
    (:func:`_perspectives_allocation`).

    Not to be read as a comparison of prices. Every row here is a share of one
    number, so the rows foot; the stand-alone leaf's rows are separate prices
    and do not.

    Parameters
    ----------
    obj : CalibrationResult
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)

pricing_evaluate = _make_exhibit_function(
    'pricing.evaluate', 'Breakeven acceptability',
    """The Cherny and Madan breakeven panel. Source frame ``evaluation_df``.

    Registered on :class:`~aggregate.results.EvaluationResult`. One block
    under both perspectives; INSURER replaces the caption with the business
    reading of ``gini_p`` and of the ``status`` column.

    Parameters
    ----------
    obj : EvaluationResult
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)


#: The exhibit registry: ``name -> (generic_fn, perspectives_fn)``.
#: ``perspectives_fn(obj)`` is the per object availability predicate; an
#: empty list means the exhibit is not available for that object.
EXHIBITS = {
    'summary': (summary, _perspectives_always),
    'tail': (tail, _perspectives_updated),
    'stats': (stats, _perspectives_always),
    'validation': (validation, _perspectives_always),
    'reins': (reins, _perspectives_reins),
    'economic': (economic, _perspectives_always),
    'economic_ratios': (economic_ratios, _perspectives_always),
    'economic_waterfall': (economic_waterfall, _perspectives_tower),
    'dependency': (dependency, _perspectives_updated),
    # Keyed on result objects, not on a built object: a successful call is
    # what produces one, so there is no partially available state to gate on.
    'pricing.calibrate': (pricing_calibrate, _perspectives_always),
    'pricing.stand_alone': (pricing_stand_alone, _perspectives_always),
    # the one pricing leaf with a structural gate: not every calibration has
    # parts to split its target across
    'pricing.allocate': (pricing_allocate, _perspectives_allocation),
    'pricing.evaluate': (pricing_evaluate, _perspectives_always),
}


def register_simple_exhibit(name, title, frame_attr, classes, *,
                            predicate=None, doc=None, caption=None,
                            formatters=None):
    """Declare a passthrough exhibit over one named frame, in one line.

    The common case: an exhibit that serves a single frame with no business
    translation. It registers **no insurer override**, so INSURER equals RAW
    by the default rule and both perspectives serve the same table, with no
    extra code. Register an override later (``<name>.insurer.register(Cls)``)
    and only that (exhibit, type) pair changes.

    Calling it twice for one ``name`` extends the existing exhibit to more
    classes rather than replacing it, so a class module may add itself to an
    exhibit the manifest already declared. That is also how one exhibit
    carries a **different caption per class**: one call per class group. A
    single sentence cannot describe ``summary_df`` on an ``Aggregate`` (count
    risk, severity, total loss) and on a ``PnL`` (consideration, obligation,
    margin) at once, and a caption that tries is worse than none.

    Parameters
    ----------
    name : str
        Registry key and the URL segment the app serves it under.
    title : str
        Display title, used as ``"<title>: <object>"``.
    frame_attr : str
        Attribute on the object holding the frame. Also the block name, so
        the served block is self-describing.
    classes : iterable of type
        The types this exhibit is registered for.
    predicate : callable, optional
        ``perspectives_fn(obj) -> list[Perspective]``, the availability gate.
        Defaults to :func:`_perspectives_always`.
    doc : str, optional
        Docstring for the generated exhibit function. A serviceable default
        is written from ``title`` and ``frame_attr``.
    caption : str, optional
        What the frame is, in a sentence or two, carried into the served
        block's ``TableSpec`` and lifted onto ``Exhibit.meta['captions']``.
        Without one a passthrough arrives with no prose at all, and a client
        that wants any has to write its own, which is how a frame's
        description ends up with three sources that can disagree.
    formatters : dict, optional
        Column formats, ``{column: format spec}``, for the columns whose
        units the frame itself does not carry (a ``CV`` reads as a
        percentage, a skewness does not).

    Returns
    -------
    callable
        The generic exhibit function, so a caller can hang an override on it.

    Examples
    --------
    ::

        register_simple_exhibit('bs_window', 'Grid sizing', 'bs_window_df',
                                [Aggregate, Portfolio, BivariateAggregate],
                                predicate=_perspectives_updated)

    .. versionadded:: 1.0
       Provisional, in the sense of PEP 411: not part of the 1.0 API
       contract. See :doc:`/3_reference/3_x_API_Stability`.
    """
    fn = EXHIBITS[name][0] if name in EXHIBITS else _make_exhibit_function(
        name, title,
        doc or f"""{title} exhibit. Source frame ``{frame_attr}``.

    A passthrough: RAW and INSURER serve the same table, since no insurer
    override is registered (the INSURER default rule).

    Parameters
    ----------
    obj : object
        A registered first class object.
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)
    if name not in EXHIBITS:
        EXHIBITS[name] = (fn, predicate or _perspectives_always)

    kw = {}
    if caption is not None:
        kw['caption'] = caption
    if formatters is not None:
        kw['formatters'] = formatters

    def _frames(obj, _attr=frame_attr, _kw=kw):
        # a fresh dict per call: the builders downstream do dict(kw, ...) but
        # an insurer override is free to mutate, and this one is shared
        return [(_attr, getattr(obj, _attr), dict(_kw))]

    _frames.__name__ = f'_{name}_frames'
    _frames.__doc__ = f'Serve ``{frame_attr}`` unchanged.'
    for cls in classes:
        fn.register(cls)(_frames)
    return fn


# --- shared translation builders --------------------------------------------

def _reins_frames(obj):
    """The two reinsurance blocks: the layering store and the stage summary.

    Shared by ``Aggregate`` and ``Portfolio``, whose frames differ in shape
    but not in which two frames make up the exhibit.
    """
    return [
        ('reins_stats_df', obj.reins_stats_df,
         {'caption': 'The program layer by layer, by view: the moments of '
                     'what is subject to each layer, what it cedes and what '
                     'is retained. Per layer columns are conditional on a '
                     'loss reaching the layer; the Ceded and Net totals are '
                     'not.'}),
        ('reins_summary_df', obj.reins_summary_df,
         {'caption': 'What each stage of the program does to the book, stage '
                     'by stage, in the eight column validation layout.'}),
    ]


# --- shared flag helpers ----------------------------------------------------


def _summary_flags(df):
    """Row flags for a summary frame: total and Agg subtotal rows.

    Positional (the frames stage relabels afterwards; positions survive a
    rename). On an ``Aggregate`` frame (plain ``X`` index) the ``Agg`` row is
    the total; on a ``Portfolio`` frame (``(unit, X)`` MultiIndex) the
    ``total`` block rows are totals and each unit's ``Agg`` row a subtotal.
    """
    flags = {}
    multi = isinstance(df.index, pd.MultiIndex)
    for i, key in enumerate(df.index):
        if multi:
            unit, x = key[0], key[-1]
            if unit == 'total':
                flags[i] = ('total',)
            elif x == 'Agg':
                flags[i] = ('subtotal',)
        elif key == 'Agg':
            flags[i] = ('total',)
    return flags


def _tail_flags(df):
    """Row flags for a tail frame: capital anchors and the total block.

    Emphasis on the :data:`CAPITAL_ANCHOR_PERIODS` rows (1 in 200 and
    1 in 250); the portfolio ``total`` block rows carry the total flag.

    The return period is read from the ``T`` column, not the index, which is
    the symmetric probability ladder ``P``. Each anchor therefore emphasizes
    two rows, its lower tail rung and its upper, one of which is the capital
    number under each sign convention.
    """
    flags = {}
    periods = df['T'] if 'T' in df.columns else None
    for i, key in enumerate(df.index):
        unit = key[0] if isinstance(key, tuple) else None
        row = []
        if unit == 'total':
            row.append('total')
        anchor = False
        if periods is not None:
            try:
                anchor = float(periods.iloc[i]) in CAPITAL_ANCHOR_PERIODS
            except (TypeError, ValueError):
                anchor = False
        if anchor:
            row.append('emphasis')
        if row:
            flags[i] = tuple(row)
    return flags


# --- shared translation helpers ---------------------------------------------
# Used by more than one class module; the registrations themselves live
# next to their class.

def _drop_raw_moment_rows(df):
    """Drop rows whose ``measure`` index level is a raw noncentral moment.

    The ``mean`` / ``cv`` / ``skew`` rows and any ``meta`` block stay
    (``ex1`` duplicates ``mean``, so nothing is lost). Shared by the stats
    and reins insurer overrides.

    A frame carrying no ``measure`` level passes through untouched: an empty
    moment store (a P&L with no stochastic engine) has nothing to drop.
    """
    if 'measure' not in (df.index.names or ()):
        return df
    keep = ~df.index.get_level_values('measure').isin(RAW_MOMENT_MEASURES)
    return df.loc[keep]


def _moment_validation_emphasis(df, valid_for_unit):
    """Row flags for a Freq / Sev / Agg validation frame from Validation flags.

    Parameters
    ----------
    df : pandas.DataFrame
        Plain ``X`` index (Aggregate) or ``(unit, X)`` MultiIndex
        (Portfolio, handle keyed: the override runs before relabeling).
    valid_for_unit : callable
        ``unit -> Validation flags`` (``unit`` is None on a plain frame,
        ``'total'`` for the portfolio total block); returning None or
        ``NOT_UPDATED`` skips the row.

    Returns
    -------
    dict
        Positional ``row_flags``: ``emphasis`` on a failing ``Sev`` / ``Agg``
        row (``Freq`` is PGF exact and never flags), plus ``total`` on the
        portfolio total block rows.
    """
    flags = {}
    for i, key in enumerate(df.index):
        unit, x = (key[0], key[-1]) if isinstance(key, tuple) else (None, key)
        verdict = valid_for_unit(unit)
        row = []
        if unit == 'total':
            row.append('total')
        if verdict is not None and not (verdict & Validation.NOT_UPDATED):
            if (x == 'Sev' and (verdict & _SEV_FAILURES)) \
                    or (x == 'Agg' and (verdict & _AGG_FAILURES)):
                row.append('emphasis')
        if row:
            flags[i] = tuple(row)
    return flags


def _check_table_emphasis(df):
    """Row flags for a check table with a boolean ``Pass`` column."""
    return {i: ('emphasis',) for i, ok in enumerate(df['Pass'])
            if not bool(ok)}


def _reins_summary_flags(df):
    """Row flags for a reins summary frame: the portfolio ``total`` block.

    The Aggregate frame (``(stage, view, component)`` index) has no total
    block and takes no flags; the Portfolio frame leads with a ``stage``
    level whose ``total`` block carries the end to end portfolio moments.
    """
    flags = {}
    for i, key in enumerate(df.index):
        if isinstance(key, tuple) and key[0] == 'total':
            flags[i] = ('total',)
    return flags


def _stats_insurer_moment_store(obj, blocks):
    """Drop the raw noncentral moment rows from the canonical store.

    The (component, measure) MultiIndex loses its ``ex1`` / ``ex2`` / ``ex3``
    rows; ``mean`` / ``cv`` / ``skew`` and the ``meta`` block stay. The raw
    perspective keeps all 26 rows.
    """
    block_name, df, kw = blocks[0]
    caption = ('Canonical moment store by component and measure across the '
               'computation views. Raw noncentral moments (ex1, ex2, ex3) '
               'are dropped from this view; the raw perspective keeps the '
               'full store.')
    return [(block_name, _drop_raw_moment_rows(df), dict(kw, caption=caption))]
