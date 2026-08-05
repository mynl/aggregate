"""Business exhibits: first class frames translated to greater_tables IR.

The library owns meaning, the app owns arrangement. An *exhibit* is a small
list of presentation ready tables (greater_tables ``TableDoc`` IR blocks)
derived from the first class citizen frames (``summary_df``, ``tail_df``,
``stats_df``, ...), carrying the business knowledge that would otherwise leak
into a client: captions, row emphasis, raw moment drops, relabeling. The test
for placement: if deleting the web app would destroy knowledge an actuary
would want in a notebook, that knowledge belongs here.

This module is purely additive. It imports from the core; the core never
imports it. It is deliberately NOT star imported in ``aggregate/__init__.py``
(the ``Tweedie`` / ``Pentagon`` precedent): reach it with
``from aggregate import exhibits``.

Design (see ``dev/plan-exhibits.md``):

* :class:`Perspective` names who is looking at the numbers: ``RAW`` (the
  underlying frame passed through), ``INSURED`` (buyer of insurance),
  ``INSURER`` (seller of insurance, buyer of reinsurance; the cedent), and
  ``REINSURER`` (seller of reinsurance). At 1.0 only ``RAW`` and ``INSURER``
  are implemented; ``INSURED`` and ``REINSURER`` are stable vocabulary with
  no registrations yet.
* Each exhibit (``summary``, ``tail``, ``stats``, ``validation``, ``reins``,
  ``pnl_ledger``, ``pnl_ratios``, ``dependency``) is a generic function
  dispatching on the object's type. Registration is open: app or user code
  may register new types with ``summary.register(MyType)``, supplying a
  *frames builder* ``f(obj) -> [(block_name, DataFrame, spec_kwargs), ...]``.
* The INSURER default rule: **INSURER equals RAW unless an override is
  registered** for the (exhibit, type) pair. The override hook
  (``summary.insurer.register(MyType)``) receives the raw blocks and returns
  translated blocks. Light overrides add a caption or row flags; heavy ones
  may rebuild the presentation entirely. The rule keeps the generic path
  total: a new exhibit is useful the moment its raw registration exists.
* The frame stage (:func:`exhibit_frames`) is pure pandas, testable and
  usable without greater_tables installed. Only :func:`build_exhibit` and
  :meth:`Exhibit.to_payload` touch greater_tables, through a lazy import
  that names the ``exhibits`` extra when the package is missing.

All served frames pass through the host's ``LabeledMixin._relabel`` (honoring
``use_labels`` and ``renamer``); exhibit titles use ``_title_name``.
"""

import functools
import hashlib
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from ._aggregate import Aggregate
from ._pnl import PnL
from ._portfolio import Portfolio
from .bivariate import BivariateAggregate
from .constants import Validation
from .spectral import Distortion

__all__ = [
    'Perspective', 'Exhibit', 'EXHIBITS',
    'available_exhibits', 'exhibit_frames', 'build_exhibit',
    'summary', 'tail', 'stats', 'validation', 'reins',
    'pnl_ledger', 'pnl_ratios', 'dependency',
]


#: Return periods emphasized as capital anchors in the ``tail`` exhibit:
#: 1 in 200 (99.5%, Solvency II) and 1 in 250 (99.6%, US capital adequacy).
CAPITAL_ANCHOR_PERIODS = (200.0, 250.0)

#: Raw noncentral moment measures dropped from the INSURER stats view (the
#: raw perspective keeps every row of the canonical 26 row store).
RAW_MOMENT_MEASURES = ('ex1', 'ex2', 'ex3')

#: Validation failure flags that emphasize the ``Sev`` row of a
#: Freq / Sev / Agg validation frame, and those that emphasize ``Agg``
#: (aliasing is an aggregate level symptom). ``Freq`` is PGF exact and
#: never flags.
_SEV_FAILURES = Validation.SEV_MEAN | Validation.SEV_CV | Validation.SEV_SKEW
_AGG_FAILURES = (Validation.AGG_MEAN | Validation.AGG_CV
                 | Validation.AGG_SKEW | Validation.ALIASING)


class Perspective(Enum):
    """Who is looking at the numbers.

    ``RAW`` passes the underlying frame through, essentially ``GT(df)`` to
    IR. ``INSURED`` is the policyholder (buyer of insurance). ``INSURER`` is
    the seller of insurance and buyer of reinsurance (the cedent, the object
    holder on the reins exhibits). ``REINSURER`` is the seller of
    reinsurance, a sign and label flip relative to the insurer (retro, where
    the reinsurer buys, is parked).

    Only ``RAW`` and ``INSURER`` are implemented at 1.0. ``INSURED`` and
    ``REINSURER`` are stable vocabulary so the enum does not churn when
    their implementations arrive.
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
    """Lazy import of greater_tables, naming the extra when missing."""
    try:
        import greater_tables
    except ImportError as exc:
        raise ImportError(
            'greater_tables is required to build exhibit IR blocks: install the '
            "exhibits extra (greater_tables>=6.0.0a8; until GT 6 publishes to "
            'PyPI this means a sibling checkout, see pyproject.toml)') from exc
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

    def fn(obj, perspective=Perspective.RAW):
        return build_exhibit(obj, name, perspective)

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


def build_exhibit(obj, name, perspective=Perspective.RAW):
    """Build an :class:`Exhibit`: frame stage, then greater_tables IR per block.

    The only greater_tables import site besides :meth:`Exhibit.to_payload`.

    Parameters
    ----------
    obj : object
        A registered first class object.
    name : str
        Exhibit registry name.
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
        Frozen; ``ir_blocks`` hold one hash stamped ``TableDoc`` per block.
    """
    perspective = _resolve_perspective(perspective)
    blocks = exhibit_frames(obj, name, perspective)
    gt = _import_greater_tables()
    fn, _ = EXHIBITS[name]
    ir_blocks = []
    captions = {}
    for block_name, df, kw in blocks:
        ir_blocks.append(gt.build(df, gt.TableSpec(**kw)))
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

pnl_ledger = _make_exhibit_function(
    'pnl_ledger', 'P&L ledger',
    """P&L ledger exhibit. Source frame ``PnL.stats_df``.

    Registrations land in the [Exhibits-PnL-Translation] phase.

    Parameters
    ----------
    obj : object
        A registered first class object.
    perspective : Perspective or str, default Perspective.RAW

    Returns
    -------
    Exhibit
    """)

pnl_ratios = _make_exhibit_function(
    'pnl_ratios', 'P&L ratios',
    """P&L ratio card exhibit. Source frames ``PnL.ratio_df`` and ``legs_df``.

    Registrations land in the [Exhibits-PnL-Translation] phase.

    Parameters
    ----------
    obj : object
        A registered first class object.
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


#: The exhibit registry: ``name -> (generic_fn, perspectives_fn)``.
#: ``perspectives_fn(obj)`` is the per object availability predicate; an
#: empty list means the exhibit is not available for that object.
EXHIBITS = {
    'summary': (summary, _perspectives_always),
    'tail': (tail, _perspectives_updated),
    'stats': (stats, _perspectives_always),
    'validation': (validation, _perspectives_always),
    'reins': (reins, _perspectives_reins),
    'pnl_ledger': (pnl_ledger, _perspectives_always),
    'pnl_ratios': (pnl_ratios, _perspectives_always),
    'dependency': (dependency, _perspectives_updated),
}


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
    """
    flags = {}
    for i, key in enumerate(df.index):
        unit, period = (key[0], key[-1]) if isinstance(key, tuple) else (None, key)
        row = []
        if unit == 'total':
            row.append('total')
        try:
            anchor = float(period) in CAPITAL_ANCHOR_PERIODS
        except (TypeError, ValueError):
            anchor = False
        if anchor:
            row.append('emphasis')
        if row:
            flags[i] = tuple(row)
    return flags


# --- summary registrations --------------------------------------------------

@summary.register(Aggregate)
def _summary_frames_aggregate(obj):
    return [('summary_df', obj.summary_df, {})]


@summary.register(Portfolio)
def _summary_frames_portfolio(obj):
    return [('summary_df', obj.summary_df, {})]


@summary.insurer.register(Aggregate)
def _summary_insurer_aggregate(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Moments and key percentiles by component (Freq, Sev, Agg). '
               'Percentiles are exact grid values. Frequency percentiles are '
               'blank by design: frequency enters through its PGF and no '
               'count distribution is materialized.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_summary_flags(df)))]


@summary.insurer.register(Portfolio)
def _summary_insurer_portfolio(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Moments and key percentiles per unit plus the portfolio '
               'total; the total block carries the Agg row only (a portfolio '
               'has no single Freq or Sev). Frequency percentiles are blank '
               'by design: frequency enters through its PGF and no count '
               'distribution is materialized.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_summary_flags(df)))]


# --- tail registrations -----------------------------------------------------

@tail.register(Aggregate)
def _tail_frames_aggregate(obj):
    return [('tail_df', obj.tail_df, {})]


@tail.register(Portfolio)
def _tail_frames_portfolio(obj):
    return [('tail_df', obj.tail_df, {})]


@tail.insurer.register(Aggregate)
def _tail_insurer_aggregate(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Return period ladder: VaR (the quoted number), TVaR (the '
               'priced number), excess VaR over the mean (capital), and '
               'VaR to mean leverage, exact from the FFT grid. The 1 in 200 '
               '(99.5%, Solvency II) and 1 in 250 (99.6%, US capital '
               'adequacy) anchors are emphasized.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_tail_flags(df)))]


@tail.insurer.register(Portfolio)
def _tail_insurer_portfolio(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Return period ladder per unit plus the portfolio total: VaR, '
               'TVaR, excess VaR over the mean (capital), and VaR to mean '
               'leverage, exact from the FFT grid. The 1 in 200 (99.5%, '
               'Solvency II) and 1 in 250 (99.6%, US capital adequacy) '
               'anchors are emphasized.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_tail_flags(df)))]


# --- summary registrations, remaining first class classes -------------------
# PnL / Distortion / BivariateAggregate serve their summary_df unchanged: no
# insurer override is registered yet (INSURER equals RAW per the default
# rule). The PnL business framing is the [Exhibits-PnL-Translation] phase,
# behind its author gate.

@summary.register(PnL)
@summary.register(Distortion)
@summary.register(BivariateAggregate)
def _summary_frames_generic(obj):
    return [('summary_df', obj.summary_df, {})]


# --- stats registrations ----------------------------------------------------

@stats.register(Aggregate)
@stats.register(Portfolio)
@stats.register(BivariateAggregate)
@stats.register(PnL)
@stats.register(Distortion)
def _stats_frames_generic(obj):
    return [('stats_df', obj.stats_df, {})]


def _drop_raw_moment_rows(df):
    """Drop rows whose ``measure`` index level is a raw noncentral moment.

    The ``mean`` / ``cv`` / ``skew`` rows and any ``meta`` block stay
    (``ex1`` duplicates ``mean``, so nothing is lost). Shared by the stats
    and reins insurer overrides.
    """
    keep = ~df.index.get_level_values('measure').isin(RAW_MOMENT_MEASURES)
    return df.loc[keep]


@stats.insurer.register(Aggregate)
@stats.insurer.register(Portfolio)
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


# --- validation registrations -----------------------------------------------

@validation.register(Aggregate)
@validation.register(Portfolio)
@validation.register(BivariateAggregate)
@validation.register(PnL)
@validation.register(Distortion)
def _validation_frames_generic(obj):
    return [('validation_df', obj.validation_df, {})]


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


@validation.insurer.register(Aggregate)
def _validation_insurer_aggregate(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Moment QA: reference vs realized FFT estimate with noise '
               'aware relative errors (the economic Gross / Net / Ceded view '
               'when reinsurance is present). Rows failing validation at the '
               'object\'s validation_eps gate are emphasized.')
    flags = _moment_validation_emphasis(df, lambda unit: obj.valid)
    return [(block_name, df, dict(kw, caption=caption, row_flags=flags))]


@validation.insurer.register(Portfolio)
def _validation_insurer_portfolio(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Moment QA per unit plus the portfolio total: reference vs '
               'realized FFT estimate with noise aware relative errors (the '
               'economic view when any unit cedes). Rows failing validation '
               'at each object\'s validation_eps gate are emphasized.')
    by_unit = {a.name: a.valid for a in obj}
    by_unit['total'] = obj.valid
    flags = _moment_validation_emphasis(df, by_unit.get)
    return [(block_name, df, dict(kw, caption=caption, row_flags=flags))]


def _check_table_emphasis(df):
    """Row flags for a check table with a boolean ``Pass`` column."""
    return {i: ('emphasis',) for i, ok in enumerate(df['Pass'])
            if not bool(ok)}


@validation.insurer.register(Distortion)
def _validation_insurer_distortion(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Structural identity checks with gates and verdicts; failing '
               'checks are emphasized.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_check_table_emphasis(df)))]


@validation.insurer.register(BivariateAggregate)
def _validation_insurer_bivariate(obj, blocks):
    block_name, df, kw = blocks[0]
    caption = ('Joint grid checks (per axis marginal means and the tail '
               'deficit) with gates and verdicts; failing checks are '
               'emphasized.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_check_table_emphasis(df)))]


# --- dependency registrations -----------------------------------------------

@dependency.register(BivariateAggregate)
def _dependency_frames_bivariate(obj):
    return [('dependency_df', obj.dependency_df, {}),
            ('axis_support_df', obj.axis_support_df, {})]


# --- reins registrations ([Exhibits-Reins-Insurer]) -------------------------

@reins.register(Aggregate)
@reins.register(Portfolio)
def _reins_frames(obj):
    return [('reins_stats_df', obj.reins_stats_df, {}),
            ('reins_summary_df', obj.reins_summary_df, {})]


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


@reins.insurer.register(Aggregate)
def _reins_insurer_aggregate(obj, blocks):
    (stats_name, stats_frame, stats_kw), \
        (summary_name, summary_frame, summary_kw) = blocks
    stats_caption = (
        'Layering analysis by view and layer: per layer columns are '
        'conditional on a loss reaching the layer, the Ceded and Net totals '
        'are unconditional, and the meta block carries attachment and '
        'exhaustion probabilities and loss on line. Raw noncentral moments '
        '(ex1, ex2, ex3) are dropped from this view; the raw perspective '
        'keeps the full store.')
    summary_caption = (
        'Per stage cession impact on the eight validation columns: Change '
        'reads as rebucketing error on the leading gross or subject row and '
        'as the percentage impact of the cession on the ceded and net rows.')
    return [
        (stats_name, _drop_raw_moment_rows(stats_frame),
         dict(stats_kw, caption=stats_caption)),
        (summary_name, summary_frame,
         dict(summary_kw, caption=summary_caption)),
    ]


@reins.insurer.register(Portfolio)
def _reins_insurer_portfolio(obj, blocks):
    (stats_name, stats_frame, stats_kw), \
        (summary_name, summary_frame, summary_kw) = blocks
    stats_caption = (
        'End to end gross, ceded and net aggregate moments per unit plus '
        'the convolved portfolio total. Raw noncentral moments (ex1, ex2, '
        'ex3) are dropped from this view; the raw perspective keeps the '
        'full store.')
    summary_caption = (
        'Per unit cession impact plus the end to end portfolio total: '
        'Change reads as the percentage impact of the reinsurance program '
        'on each moment (0 on the gross reference rows).')
    return [
        (stats_name, _drop_raw_moment_rows(stats_frame),
         dict(stats_kw, caption=stats_caption)),
        (summary_name, summary_frame,
         dict(summary_kw, caption=summary_caption,
              row_flags=_reins_summary_flags(summary_frame))),
    ]
