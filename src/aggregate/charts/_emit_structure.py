"""Structure chart emitter: a reinsurance program as a tower of layers.

The broker-slide picture. A gross slab on the left says what is being
protected, then one tower of rectangles per cession stage says how it is
carved up: each placed layer a block whose band is its attachment to its
exhaustion point and whose **width is its share**, with the unplaced
remainder beside it, the retention below and above, and any band nobody
covered drawn as a gap. Optionally each tower is joined to the matching
quantile (Lee) curve on the same loss axis, so every attachment reads off
as a return period.

This is the one chart in the library whose subject is a **contract** rather
than a law. Nothing lives between a block's corners, which is why a tower
carries :class:`~aggregate.charts.ir.TowerBlock` rectangles rather than
series, and why the chart is available before ``update()``: the program's
shape is declared, not computed.

**Three enrichment tiers, one ``annotate`` tuple.** A field whose source is
absent is silently omitted, so the same call works at every tier.

1. *Declared.* Geometry: the terms of each layer and its resolved name.
   Available on an un-updated object.
2. *Built.* The risk statistics off ``reins_stats_df``: expected loss, its
   standard deviation, loss on line, the attachment and exhaustion
   probabilities. Available on any updated :class:`~aggregate.Aggregate`,
   and so is ``lee=True``.
3. *Priced.* The economics: ceded premium, rate on line, loss ratio,
   ceding commission, reinstatements. **Available on a
   :class:`~aggregate.PnL` only**, and not by choice: a plain ``agg``
   strips its ``deposit`` / ``rol`` / ``rate`` / ``cede`` /
   ``reinstatements`` clauses with a warning, because it has no premium
   context to resolve them against. The resolved figures live on
   :attr:`~aggregate.PnL.economics`.

**The currency figures are annual on both towers.** A layer's premium,
expected loss and standard deviation are what it costs and pays over a
year, which is the basis a slip quotes and the only basis on which ``lr``,
``rol`` and ``lol`` mean anything. Its *geometry* is per occurrence on the
occurrence tower, so an expected loss several times the layer's limit is an
ordinary reading of a working layer and not a unit error.

**Premium is shown at 100% terms**, which is how a layer is quoted and
compared. The stored figures are at the placement share (all three DecL
forms are scaled by it), so the emitter divides by share exactly once, in
:func:`_layer_lines`. Expected loss and its standard deviation are divided
by the same share for the same reason, and the ratios (``lr``, ``rol``,
``lol``) are share-invariant either way.

**What the aggregate tower is read against.** Not ``p_agg_gross``: the
subject of an aggregate cover is the aggregate of the *requested occurrence
output*, which is ``p_agg_net_occ`` under an ``occurrence net of`` program
and ``p_agg_ceded_occ`` under an ``occurrence ceded to`` one. The
``p_agg_subject`` column of ``reins_density_df`` is exactly that, and it is
what ``reins_stats_df`` itself uses for every aggregate-stage attachment
probability. Naming either underlying column instead would be right for
half the programs and silently wrong for the other half.

Pure numpy and pandas; no matplotlib.
"""

import numpy as np

from .._aggregate import Aggregate
from .._grid_distribution import GridDistribution
from .._pnl import PnL
from . import register_chart, _emitter_base
from ._payload import lattice_payload
from ._two_panel import (RETURN_PERIOD_TOP, SURVIVAL_FLOOR, WINDOW_PAD,
                          loss_window, quantile_curve)
from .ir import (ChartAxis, ChartDoc, ChartSeries, Mark, Panel, TowerBlock,
                 complete_tex)

__all__ = ['chart_structure']

#: The annotation vocabulary, **in the order it is rendered**. An
#: ``annotate`` tuple selects from this; the order it is passed in is
#: ignored, so two callers asking for the same fields get the same picture.
#:
#: 'geometry' the terms line, 'premium' the ceded premium at 100% terms,
#: 'el' the expected ceded loss, 'lr' el over premium, 'rol' premium over
#: limit, 'lol' el over limit, 'sd' the standard deviation of the cession,
#: 'pr_attach' and 'pr_detach' the probabilities the layer is reached and
#: exhausted, 'reinstatements' the schedule, 'cede' the ceding commission
#: rate.
#:
#: 'premium', 'lr', 'rol', 'cede' and 'reinstatements' need resolved
#: economics and so a :class:`~aggregate.PnL`; the rest need only a built
#: :class:`~aggregate.Aggregate`, and 'geometry' needs nothing at all.
ANNOTATE_FIELDS = ('geometry', 'premium', 'el', 'lr', 'rol', 'lol', 'sd',
                   'pr_attach', 'pr_detach', 'reinstatements', 'cede')

#: The default selection: what the layer is, what it costs, what it is
#: expected to pay, and the ratio of the two.
DEFAULT_ANNOTATE = ('geometry', 'premium', 'el', 'lr')

#: Headroom above the highest attachment when the top layer is unlimited,
#: as a fraction of the window below it. An unlimited layer has no top to
#: draw, so the window supplies one and the block declares ``open_top``.
HEADROOM = 0.25

#: Both stage keys, in program order: an occurrence program inures to an
#: aggregate cover, so the occurrence tower is read first.
STAGES = ('occ', 'agg')

#: Human names for the two stages, used in panel titles and axis labels.
STAGE_TITLE = {'occ': 'Per occurrence', 'agg': 'In the aggregate'}
STAGE_LOSS = {'occ': 'Loss per claim', 'agg': 'Aggregate loss'}

chart_structure = _emitter_base('structure')


def _engine(obj):
    """The :class:`~aggregate.Aggregate` a host object carries, or None.

    A :class:`~aggregate.PnL` wraps one stochastic engine, which may be an
    aggregate, a bivariate, or nothing at all on a stitched ledger. Only
    the single-aggregate case has a program to draw.
    """
    engine = obj.engine if isinstance(obj, PnL) else obj
    return engine if isinstance(engine, Aggregate) else None


def _economics(obj):
    """The resolved per-layer economics, or an empty dict.

    Empty for an :class:`~aggregate.Aggregate`, which has none by
    construction, and for a :class:`~aggregate.PnL` whose cessions carry no
    premium clause.
    """
    return (obj.economics or {}) if isinstance(obj, PnL) else {}


def _has_program(obj):
    """Availability: a single aggregate carrying at least one cession."""
    agg = _engine(obj)
    if agg is None:
        return False
    return agg.occ_reins is not None or agg.agg_reins is not None


def _updated(agg):
    """True when the aggregate has a realized grid, so stats and curves exist."""
    return getattr(agg, 'agg_density', None) is not None


# ----------------------------------------------------------------- formatting

def _money(value):
    """A currency amount as a reader writes it.

    Integral amounts and anything at a thousand or above carry thousands
    separators and no decimals, which is how a slip is quoted. Below that,
    four significant digits, so a unit-scaled book (a layer of 1.5 xs 0.5)
    does not read as ``2 xs 0``. The thousand threshold is what keeps a
    computed mean of 10,000.000000001 out of scientific notation, which is
    not a form anyone quotes currency in.
    """
    value = float(value)
    if not np.isfinite(value):
        return 'unlimited'
    if abs(value) >= 1000 or value == int(value):
        return f'{value:,.0f}'
    return f'{value:,.4g}'


def _pct(value):
    """A rate or probability as a percentage, or None when it is not finite."""
    value = float(value)
    if not np.isfinite(value):
        return None
    return f'{value:.4g}%' if abs(value) < 0.001 else f'{value:,.1%}'


def _terms(share, limit, attach, stage):
    """The terms line for one layer: what a slip would say.

    ``share po limit xs attach``, in DecL's own spelling, dropping the
    placement clause for a full line. An unlimited layer attaching at zero
    on the **aggregate** stage is a quota share and is named one; the
    identical clause on the occurrence stage cedes the same thing but keeps
    its literal wording, because "quota share" is an aggregate-stage word
    and one name means one thing.
    """
    unlimited = not np.isfinite(limit)
    if stage == 'agg' and unlimited and attach == 0:
        return f'{_pct(share)} quota share'
    body = 'unlimited' if unlimited else _money(limit)
    head = '' if share >= 1.0 else f'{_pct(share)} po '
    return f'{head}{body} xs {_money(attach)}'


def _reinstatement_line(terms):
    """The reinstatement schedule as one line, or None."""
    rates = tuple(terms.rates)
    if not rates:
        return 'no reinstatements'
    return (f'{len(rates)} reinstatements at '
            + ', '.join(_pct(r) for r in rates))


# -------------------------------------------------------------- layer reading

def _meta(stats, stage, column, row):
    """One ``reins_stats_df`` meta value, or NaN when there is no frame."""
    if stats is None:
        return np.nan
    try:
        return float(stats.loc[('meta', row), (stage, column)])
    except KeyError:                        # pragma: no cover - defensive
        return np.nan


def _moment(stats, stage, column, measure):
    """One aggregate-component moment, or NaN when there is no frame."""
    if stats is None:
        return np.nan
    try:
        return float(stats.loc[('agg', measure), (stage, column)])
    except KeyError:                        # pragma: no cover - defensive
        return np.nan


def _by_layer(econ, stage, key, index):
    """One entry of a per-layer economics list, or NaN.

    The list is absent on a scalar-API economics dict, which carries only
    the side totals, so its absence is a normal reading and not an error.
    """
    values = econ.get(f'{key}_{stage}_by_layer') or []
    if index >= len(values):
        return np.nan
    return float(values[index])


def _layer_lines(agg, stats, econ, stage, index, share, limit, attach,
                 annotate, headline=None):
    """The selected annotation lines for one layer, in canonical order.

    Parameters
    ----------
    agg : Aggregate
        The engine, for the reinstatement schedule.
    stats : pandas.DataFrame or None
        ``reins_stats_df``, or None on an un-updated object.
    econ : dict
        Resolved per-layer economics; empty for an ``Aggregate``.
    stage : str
        'occ' or 'agg'.
    index : int
        Zero-based layer index; the frame's column is ``layer.{index + 1}``.
    share, limit, attach : float
        The layer's tuple.
    annotate : tuple of str
        The requested fields, any subset of :data:`ANNOTATE_FIELDS`.
    headline : str, optional
        The block's own label. When it is already the terms line, which is
        what an unnamed layer falls back to, the 'geometry' field is
        dropped: one fact belongs on the block once.

    Returns
    -------
    tuple of str
        One string per requested field whose source exists, in
        :data:`ANNOTATE_FIELDS` order.

    Notes
    -----
    Every currency figure is divided by ``share`` once, here, to put it on
    100% terms. That is exactly right for premium, commission, expected
    loss and its standard deviation, all four of which scale linearly in
    the placed fraction: the stored premium forms are quoted at 100% and
    multiplied by ``share``, and a layer's ceded density is built from a
    ceder that already carries the share. The ratios are share-invariant
    and so need no such care.
    """
    column = f'layer.{index + 1}'
    scale = share if share > 0 else np.nan
    premium = _by_layer(econ, stage, 'pc', index) / scale
    commission = _by_layer(econ, stage, 'c', index)
    el = _moment(stats, stage, column, 'mean') / scale
    cv = _moment(stats, stage, column, 'cv')
    values = {
        'geometry': _terms(share, limit, attach, stage),
        'premium': f'premium {_money(premium)}'
                   if np.isfinite(premium) else None,
        'el': f'el {_money(el)}' if np.isfinite(el) else None,
        'lr': f'lr {_pct(el / premium)}'
              if np.isfinite(el) and premium > 0 else None,
        'rol': f'rol {_pct(premium / limit)}'
               if np.isfinite(premium) and np.isfinite(limit) and limit > 0
               else None,
        'lol': f'lol {_pct(_meta(stats, stage, column, "lol"))}'
               if np.isfinite(_meta(stats, stage, column, 'lol')) else None,
        'sd': f'sd {_money(el * cv)}'
              if np.isfinite(el) and np.isfinite(cv) else None,
        'pr_attach': f'pr attach {_pct(_meta(stats, stage, column, "pr_attach"))}'
                     if np.isfinite(_meta(stats, stage, column, 'pr_attach'))
                     else None,
        'pr_detach': f'pr detach {_pct(_meta(stats, stage, column, "pr_detach"))}'
                     if np.isfinite(_meta(stats, stage, column, 'pr_detach'))
                     else None,
        'reinstatements': None,
        'cede': f'cede {_pct(commission / (premium * scale))}'
                if np.isfinite(commission) and commission > 0
                and np.isfinite(premium) and premium > 0 else None,
    }
    # The reinstatement schedule rides on the occurrence layer the DecL
    # clause decorated, which is the first one (the builder reads
    # ``occ_reins[0]``), so it annotates that layer and no other.
    terms = getattr(agg, 'reinstatement_terms', None)
    if terms is not None and stage == 'occ' and index == 0:
        values['reinstatements'] = _reinstatement_line(terms)
    if headline is not None and values['geometry'] == headline:
        values['geometry'] = None
    return tuple(values[f] for f in ANNOTATE_FIELDS
                 if f in annotate and values[f] is not None)


# --------------------------------------------------------------------- towers

def _layer_label(agg, stage, index, fallback):
    """A layer's headline: its resolved ``as`` name, else its terms.

    ``labels.occ_reins`` is a plain mapping from zero-based layer index to
    the declared name, so a layer with no ``as`` clause is simply absent
    from it.
    """
    view = getattr(agg.labels, f'{stage}_reins', None) or {}
    return view.get(index) or fallback


def _tower_blocks(agg, stats, econ, stage, panel_id, top, annotate):
    """Every block of one stage's tower, bottom up.

    Parameters
    ----------
    agg : Aggregate
    stats : pandas.DataFrame or None
    econ : dict
    stage : str
        'occ' or 'agg'.
    panel_id : str
        The panel the blocks draw in.
    top : float
        The drawn window's top, which is where an unlimited layer's block
        ends and what the highest retention reaches.
    annotate : tuple of str

    Returns
    -------
    tuple of TowerBlock

    Notes
    -----
    A gap has two spellings and both draw the same way: an explicit
    zero-share layer, which is the documented convention, and an implicit
    hole between consecutive layers, which ``_validate_reins_layers``
    equally permits. Neither is an error and a reader must see both, since
    an uncovered band inside a tower is the thing a program review is
    looking for.

    Above the tower the cedent is exposed again, so the band from the top
    layer to the window top is a retention and not empty space. It is
    omitted when the top layer is unlimited, which leaves nothing above it.
    """
    layers = getattr(agg, f'{stage}_reins') or []
    blocks = []
    cursor = 0.0
    for index, (share, limit, attach) in enumerate(layers):
        if attach > cursor:
            blocks.append(TowerBlock(
                panel_id=panel_id, x0=0.0, x1=1.0, y0=cursor, y1=attach,
                role='gap' if blocks else 'retention',
                label='Uncovered' if blocks else 'Retained',
                label_lines=(_terms(1.0, attach - cursor, cursor, stage),)))
        unlimited = not np.isfinite(limit)
        y1 = max(top, attach) if unlimited else attach + limit
        terms = _terms(share, limit, attach, stage)
        if share > 0:
            label = _layer_label(agg, stage, index, terms)
            blocks.append(TowerBlock(
                panel_id=panel_id, x0=0.0, x1=min(1.0, share),
                y0=attach, y1=y1, role='layer', label=label,
                label_lines=_layer_lines(agg, stats, econ, stage, index,
                                         share, limit, attach, annotate,
                                         headline=label),
                open_top=unlimited))
            if share < 1.0:
                blocks.append(TowerBlock(
                    panel_id=panel_id, x0=share, x1=1.0, y0=attach, y1=y1,
                    role='co_participation', label='Co-participation',
                    label_lines=(f'{_pct(1.0 - share)} unplaced',),
                    open_top=unlimited))
        else:
            # A zero-share layer is the documented way to declare a gap;
            # it cedes nothing, so it draws as one rather than as a layer
            # of no width.
            blocks.append(TowerBlock(
                panel_id=panel_id, x0=0.0, x1=1.0, y0=attach, y1=y1,
                role='gap', label='Uncovered', label_lines=(terms,),
                open_top=unlimited))
        cursor = y1
    if cursor < top:
        blocks.append(TowerBlock(
            panel_id=panel_id, x0=0.0, x1=1.0, y0=cursor, y1=top,
            role='retention', label='Retained',
            label_lines=(f'above {_money(cursor)}',)))
    return tuple(blocks)


def _tower_extent(agg, stage):
    """``(top, has_unlimited)`` of one stage's declared tower.

    ``top`` is the highest exhaustion point over the finite layers, or the
    highest attachment when every layer is unlimited. It is what the drawn
    window must contain: a window that crops the tower would draw a
    program the object does not have.
    """
    layers = getattr(agg, f'{stage}_reins') or []
    finite = [a + y for (_s, y, a) in layers if np.isfinite(y)]
    attaches = [a for (_s, _y, a) in layers]
    unlimited = any(not np.isfinite(y) for (_s, y, _a) in layers)
    top = max(finite) if finite else (max(attaches) if attaches else 0.0)
    return float(top), unlimited


def _support(agg, stage):
    """``(lo, hi)`` structural support of the quantity one stage's tower bands.

    Parameters
    ----------
    agg : Aggregate
        Need not be updated: the report this reads is built from the spec.
    stage : str
        'occ' or 'agg'.

    Returns
    -------
    tuple of float
        The bounds, with ``inf`` at an unbounded top. ``(0.0, inf)`` when
        the report or the row it wants is absent, which is the answer that
        leaves the window where it was before this rule existed.

    Notes
    -----
    Read off :attr:`~aggregate.Aggregate.tail_behavior_df`, which composes
    both of the two ways a severity can be bounded and is valid before
    :meth:`~aggregate.Aggregate.update`. The occurrence tower is read
    against the **gross** severity, so its bound is the union over the
    ``comp*`` rows: not the ``severity`` row, which is absent when there is
    a single component, and not ``severity (net occ)``, which is the net
    rather than the gross. Both spellings of a bounded severity land there,
    an explicit ``xs`` clause and a distribution bounded in itself such as
    ``40 * uniform``.

    The aggregate tower is the ``aggregate`` row, and it is ``inf`` under
    any unbounded frequency. That is the rule rather than a special case:
    an aggregate acquires a ceiling only when the frequency is bounded too,
    so ``dfreq [1 2 3] sev 40 * uniform`` is bounded at ``120`` and the
    same severity under Poisson is not.
    """
    try:
        df = agg.tail_behavior_df
    except (AttributeError, KeyError, ValueError):  # pragma: no cover
        # The chart draws whatever object it is handed, and a stage window
        # is not the place to fail over a report that will not build.
        return 0.0, np.inf
    if stage == 'occ':
        rows = [i for i in df.index if str(i).startswith('comp')]
    else:
        rows = [i for i in df.index if i == 'aggregate']
    if not rows or 'min' not in df or 'max' not in df:  # pragma: no cover
        return 0.0, np.inf
    frame = df.loc[rows]
    return float(frame['min'].min()), float(frame['max'].max())


def _stage_window(top, unlimited, reference, support):
    """The loss window for one stage: the contract, or the law beside it.

    Parameters
    ----------
    top : float
        The declared tower's top.
    unlimited : bool
        Whether the top layer is unbounded, in which case the window
        supplies a top the block is drawn to.
    reference : GridDistribution or None
        The distribution the tower is read against, when the object is
        built.
    support : tuple of float
        The stage quantity's structural bounds, from :func:`_support`.

    Returns
    -------
    tuple of float
        ``(lo, hi)``, **unpadded**: the extent the blocks are drawn to and
        the extent the axis declares. The caller pads the top only, for the
        axis alone, so a block never asserts cover above the contract.

    Notes
    -----
    **Where the support is bounded the contract answers and there is
    nothing to trade off.** The gross column is the policy, and a program
    review draws it whole: a ``100 xs 0`` policy cropped at a quantile of
    its own severity asserts less cover than was bought. A band this leaves
    as a sliver is what the axis' log reading is for.

    Where the support is unbounded the window is the **union** of the tower
    and the law's own interesting slice, never one or the other. Taking the
    law alone crops a high layer that rarely attaches out of its own
    picture; taking the tower alone puts a modest program against a book
    whose tail runs far past it. An unlimited top layer then has no
    exhaustion point to bound anything, so it gets fixed headroom above the
    highest thing that does.

    **The floor is the quantity's own lower bound and is never padded past
    it.** A loss cannot be negative, so an axis labeled ``-2,000`` says
    something false about the quantity; the tower itself starts at zero, so
    a positive lower bound (``dsev [10 20 30]``) must not crop the
    retention band below it either. Hence ``min(0, support_lo)``, which is
    ``0`` for every loss and honors a genuinely signed severity.
    """
    lo = min(0.0, support[0])
    if np.isfinite(support[1]):
        # max, not the support alone: a tower may be written above its own
        # severity's ceiling, and a window that cropped it would draw a
        # program the object does not have.
        return lo, max(support[1], top, lo + 1.0)
    hi = top
    if reference is not None:
        window = loss_window(reference.q, 0.0)
        if window is not None:
            hi = max(hi, window[1])
    if unlimited:
        hi = hi * (1.0 + HEADROOM) if hi > 0 else 1.0
    return lo, max(hi, lo + 1.0)


def _reference(agg, stage):
    """The distribution one stage's tower is read against, or None.

    The occurrence tower is read against the gross severity, the aggregate
    tower against ``p_agg_subject``: the aggregate of the requested
    occurrence output, which is what the aggregate cover actually sees.
    """
    if not _updated(agg):
        return None
    df = agg.reins_density_df
    column = 'p_sev_gross' if stage == 'occ' else 'p_agg_subject'
    return GridDistribution(df['loss'].to_numpy(dtype=float),
                            df[column].to_numpy(dtype=float),
                            bs=agg.bs, name=column)


def _gross_block(agg, stats, stage, panel_id, top, support_hi=np.inf):
    """The gross slab: what the first tower is carving up.

    On an occurrence program that is the policy layer each claim is written
    on, annotated with the gross severity's moments; on an aggregate-only
    program it is the subject aggregate over the drawn window, annotated
    with its own. One slab either way, because the point of the panel is
    the comparison of heights.

    Parameters
    ----------
    agg : Aggregate
    stats : pandas.DataFrame or None
        ``reins_stats_df`` on a built object, for the moment lines.
    stage : str
        'occ' or 'agg', which decides what the slab is.
    panel_id : str
    top : float
        The drawn window's top, unpadded.
    support_hi : float
        The subject quantity's own ceiling, from :func:`_support`.

    Notes
    -----
    **Two things can close the slab and either one is enough.** A written
    limit closes it because that is the cover bought; the severity's own
    support closes it because no claim can exceed it. ``sev 40 * uniform``
    with no ``xs`` clause is the second case on its own, and reading
    ``exp_limit`` alone would draw it open topped at 40 while asserting
    cover above a loss that cannot happen. The slab is open only when
    nothing bounds it, or when what bounds it sits above the drawn window.
    """
    if stage == 'occ':
        limit = agg.spec.get('exp_limit', np.inf)
        attach = agg.spec.get('exp_attachment', 0.0) or 0.0
        limit = float(max(np.atleast_1d(limit)))
        attach = float(min(np.atleast_1d(attach)))
        component = 'sev'
        headline = _terms(1.0, limit, attach, 'occ')
    else:
        limit, attach = np.inf, 0.0
        component = 'agg'
        headline = 'Subject'
    # Either bound closes it, so the ceiling is the lower of the two. A
    # ceiling inside the window is drawn and the slab closes on it; one
    # above the window (an unbounded subject whose window is a quantile
    # slice) is cropped, and cropping is right there. Saying nothing about
    # the crop is not: the block declares an open top exactly as an
    # unlimited layer does rather than asserting a ceiling the book has
    # not got.
    ceiling = min(attach + limit, support_hi)
    cropped = np.isfinite(ceiling) and ceiling > top
    unlimited = not np.isfinite(ceiling) or cropped
    y1 = top if unlimited else ceiling
    lines = [headline]
    if stats is not None:
        try:
            mean = float(stats.loc[(component, 'mean'), ('occ', 'Gross')])
            cv = float(stats.loc[(component, 'cv'), ('occ', 'Gross')])
        except KeyError:                    # pragma: no cover - defensive
            mean = cv = np.nan
        if np.isfinite(mean):
            lines.append(f'mean {_money(mean)}')
        if np.isfinite(mean) and np.isfinite(cv):
            lines.append(f'sd {_money(mean * cv)}')
            lines.append(f'cv {cv:,.3f}')
    return TowerBlock(panel_id=panel_id, x0=0.0, x1=1.0, y0=attach, y1=y1,
                      role='gross', label=agg.label, open_top=unlimited,
                      label_lines=tuple(lines))


# ---------------------------------------------------------------------- emit

def _lee_series(agg, stage, panel_id):
    """The quantile curve of a stage's reference distribution, as a series."""
    df = agg.reins_density_df
    column = 'p_sev_gross' if stage == 'occ' else 'p_agg_subject'
    p, outcome = quantile_curve(df['loss'].to_numpy(dtype=float),
                                df[column].to_numpy(dtype=float))
    name = 'Gross claim' if stage == 'occ' else 'Subject'
    return ChartSeries(name=name, role='gross' if stage == 'occ' else 'subject',
                       panel_id=panel_id, x=tuple(float(v) for v in p),
                       **lattice_payload(outcome, agg.bs, 'y'))


def _structure(obj, annotate=DEFAULT_ANNOTATE, lee=False):
    """Emit the reinsurance structure diagram.

    Parameters
    ----------
    obj : Aggregate or PnL
        Must carry at least one cession;
        :func:`~aggregate.charts.available_charts` answers ``'structure'``
        exactly when it does. A ``PnL`` must wrap a single ``Aggregate``.
    annotate : tuple of str
        Which annotation fields to render beside each layer, any subset of
        :data:`ANNOTATE_FIELDS`. Rendered in that constant's order whatever
        order they are passed in. A field whose source is absent is omitted
        rather than blanked, so one tuple serves an un-updated object, a
        built one and a priced one. Pass ``()`` for bare rectangles.
    lee : bool
        Also emit, beside each tower, the quantile curve of the
        distribution that tower is read against, sharing its loss axis, so
        every boundary reads off as a return period. Requires a built
        object.

    Returns
    -------
    ChartDoc
        A 'tower' panel for the gross slab, one for each cession stage
        present, and with ``lee`` one 'xy' panel beside each tower. Each
        tower shares its loss axis with its own Lee panel; the two stages
        do **not** share one, because a per-claim loss and an annual
        aggregate are not the same quantity.

    Raises
    ------
    ValueError
        For an unknown ``annotate`` field, or for ``lee=True`` on an
        object that has not been updated.

    Notes
    -----
    Every layer boundary is also a :class:`~aggregate.charts.ir.Mark` on
    the tower's loss axis, labeled with the amount, so a renderer can put
    the boundaries where the ticks would be: a tower is read at its
    breaks, not on a continuous scale. The same marks repeat on the Lee
    panel with ``faint`` set, which is what carries each boundary across to
    the curve and turns it into a return period.
    """
    agg = _engine(obj)
    unknown = sorted(set(annotate) - set(ANNOTATE_FIELDS))
    if unknown:
        raise ValueError(
            f'unknown annotate field(s) {unknown}; expected a subset of '
            f'{ANNOTATE_FIELDS}')
    built = _updated(agg)
    if lee and not built:
        raise ValueError(
            f'chart_structure: lee=True needs the realized grid: call '
            f'{agg.name}.update() first, or take the geometry-only chart '
            'the default gives.')
    annotate = tuple(annotate)
    econ = _economics(obj)
    stats = agg.reins_stats_df if built else None

    stages = [s for s in STAGES if getattr(agg, f'{s}_reins') is not None]
    axes, panels, blocks, marks, series = [], [], [], [], []
    for stage in stages:
        top, unlimited = _tower_extent(agg, stage)
        reference = _reference(agg, stage) if built else None
        support = _support(agg, stage)
        lo, drawn = _stage_window(top, unlimited, reference, support)
        loss_id = f'{stage}_loss'
        # The axis takes the padded top so a band is not drawn hard against
        # the frame; the blocks below take ``drawn``, the bare number, since
        # a retention ending two percent above the policy limit asserts
        # cover that does not exist. A consumer clamps the suggestion back
        # to the declared extent, which lands the drawn axis on the limit
        # itself rather than on a number near it.
        axes.append(ChartAxis(
            id=loss_id, label=STAGE_LOSS[stage], unit='currency',
            scales=('linear', 'log'),
            suggested_range=(lo, drawn + WINDOW_PAD * (drawn - lo)),
            full_range=(lo, drawn) if np.isfinite(support[1]) else None))
        # The placement axis is a share, so it is a 'ratio' and it is read
        # as a width rather than interrogated. One per stage, because two
        # panels sharing an axis id share the axis itself and the towers
        # are independent pictures.
        place_id = f'{stage}_place'
        axes.append(ChartAxis(id=place_id, label='Placement', unit='ratio',
                              suggested_range=(0.0, 1.0),
                              full_range=(0.0, 1.0)))
        if stage == stages[0]:
            gross_place = 'gross_place'
            axes.append(ChartAxis(id=gross_place, label='Placement',
                                  unit='ratio', suggested_range=(0.0, 1.0),
                                  full_range=(0.0, 1.0)))
            panels.append(Panel(id='gross', kind='tower', x_axis=gross_place,
                                y_axis=loss_id, read_axis='y',
                                title='Gross'))
            blocks.append(_gross_block(agg, stats, stage, 'gross', drawn,
                                       support[1]))
        panels.append(Panel(id=stage, kind='tower', x_axis=place_id,
                            y_axis=loss_id, read_axis='y',
                            title=STAGE_TITLE[stage]))
        stage_blocks = _tower_blocks(agg, stats, econ, stage, stage,
                                     drawn, annotate)
        blocks.extend(stage_blocks)
        boundaries = sorted({b.y0 for b in stage_blocks if b.role == 'layer'}
                            | {b.y1 for b in stage_blocks
                               if b.role == 'layer' and not b.open_top})
        for at in boundaries:
            marks.append(Mark(panel_id=stage, orient='h', at=at,
                              label=_money(at)))
        if lee:
            lee_id = f'{stage}_lee'
            p_id = f'{stage}_p'
            axes.append(ChartAxis(
                id=p_id, label='Non-exceeding probability', unit='probability',
                suggested_range=(0.0, 1.0)))
            axes.append(ChartAxis(
                id=f'{stage}_rp', label='Return period', unit='return_period',
                scales=('linear', 'log'), reciprocal_of=p_id,
                suggested_range=(1.0, RETURN_PERIOD_TOP),
                full_range=(1.0, float(round(1.0 / SURVIVAL_FLOOR)))))
            panels.append(Panel(
                id=lee_id, kind='xy', x_axis=p_id, y_axis=loss_id,
                read_axis='y', invertible=True,
                title=f'{STAGE_TITLE[stage]}, return period',
                inverse_title='Distribution function'))
            series.append(_lee_series(agg, stage, lee_id))
            # The joining rules: the same boundaries, faint, so the curve
            # is read against the program rather than annotated by it.
            for at in boundaries:
                marks.append(Mark(panel_id=lee_id, orient='h', at=at,
                                  label=_money(at), faint=True))

    return complete_tex(ChartDoc(
        name='structure',
        title=f'{agg.label}: reinsurance structure',
        axes=tuple(axes),
        panels=tuple(panels),
        series=tuple(series),
        marks=tuple(marks),
        blocks=tuple(blocks),
        meta={'return_period_map': 'complement',
              'stages': list(stages),
              'tiers': ['declared']
                       + (['built'] if built else [])
                       + (['priced'] if econ.get('pc_occ_by_layer')
                          or econ.get('pc_agg_by_layer') else [])},
    ))


chart_structure.register(Aggregate)(_structure)
chart_structure.register(PnL)(_structure)

register_chart('structure', chart_structure, predicate=_has_program,
               primary=None)
