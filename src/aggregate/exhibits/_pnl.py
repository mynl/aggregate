"""Exhibit treatments for :class:`~aggregate._pnl.PnL`.

Two families, and [Overview-Engine] is the ruling that keeps them apart. The
**generic** exhibit names (``summary``, ``tail``, ``stats``, ``validation``)
describe the wrapped **book**: each serves an ``engine_*`` frame, so a reader
moving between them reads one object throughout. The **accounting** family
(``economic``, ``economic_ratios``, ``economic_waterfall``, ``economic_tail``)
describes the **ledger**, which is the P&L's own story.

Before that ruling the generic names were mixed: ``stats`` and ``validation``
already looked through to the engine while ``summary`` and ``tail`` served the
ledger card and the closing margin's ladder, so one group of leaves described
two different distributions. The ledger card (``PnL.summary_df``) keeps its
name, ``qd`` and its notebook repr and simply no longer answers to an exhibit;
the closing margin's ladder moved to ``economic_tail``, where its payoff
orientation sits beside the rest of the ledger's readings.
"""

import numpy as np
import pandas as pd

from .._pnl import (
    PERCENTILE_LADDER, PnL, WATERFALL_RETURN_PERIOD, _kappa_label, _pct_label,
)
from ._core import (
    economic, economic_ratios, economic_tail, economic_waterfall, reins, stats,
    summary, tail, validation,
    _reins_frames, _stats_insurer_moment_store, _summary_flags, _tail_flags,
)

#: Ledger row kind to greater_tables row flag ([Exhibits-Economic-Insurer]).
#: The grand result is the bottom line; each group's and each tier's own
#: result, and the grand side totals, are subtotals; a running net is a
#: cumulative reading aid rather than a booked line, so it is muted, and the
#: net-of-tier result ([Ledger-Net-Of-Tier]) is the same position under a
#: tier-level name, so it mutes with it. Legs, the net-of-tier side totals,
#: and ``total_impact`` are unflagged, the latter because it is a difference
#: between two positions rather than one of them.
LEDGER_ROW_FLAGS = {
    'grand_result': ('total',),
    'group_result': ('subtotal',),
    'tier_result': ('subtotal',),
    'grand_total': ('subtotal',),
    'running_net': ('muted',),
    'net_result': ('muted',),
}


# --- the generic family: the wrapped book ([Overview-Engine]) ---------------

#: Said once, on every block that describes the wrapped book rather than the
#: ledger. A reader arriving from a P&L has the ledger in mind and these
#: frames are not about it, so without the sentence the two readings are
#: indistinguishable on the page and they differ by the whole program.
#:
#: The second clause is the one that earns its keep. The engine reports the
#: distribution **its own grid realizes**, so a book declaring ``net of``
#: reports net and one declaring ``ceded to`` reports the cession; neither is
#: the ledger's ``Gross`` row, which is the unreinsured subject. Naming the
#: engine's view as gross would be wrong on exactly the programs a P&L exists
#: to describe.
_ENGINE_RIDER = (' Figures for the wrapped book as its own grid realizes it, '
                 'so a book that declares a program reports that program\'s '
                 'view and the ledger\'s Gross row is the unreinsured '
                 'reading. The accounting exhibits are where the P&L reads '
                 'itself.')

#: Served in place of an engine frame on a hand-built kernel P&L. The same
#: treatment :func:`_stats_insurer_pnl` gives the absent moment store: an
#: empty frame with a caption saying what happened beats a blank table with
#: no explanation, and beats a predicate in the class-agnostic registry,
#: which gates per exhibit name and so cannot gray one class alone.
_NO_ENGINE = ('This P&L was built from grids rather than from a declared '
              'book, so it carries no stochastic engine and there is no book '
              'to describe. Its own readings are the economic exhibits.')


@summary.register(PnL)
def _summary_frames(obj):
    """The wrapped book's headline card ([Overview-Engine]).

    Serves :attr:`PnL.engine_summary_df`, exactly as
    :func:`_validation_frames` serves ``engine_validation_df``: the RAW
    invariant wants one public frame per block, so the delegation lives on
    the class and this function only names it.
    """
    df = obj.engine_summary_df
    if df.empty:
        return [('engine_summary_df', df, {'caption': _NO_ENGINE})]
    return [('engine_summary_df', df, {'caption': (
        'Headline moments and key percentiles for the book this P&L is a '
        'ledger over, by component: count risk (Freq), single claim severity '
        '(Sev) and total loss (Agg), one block per unit on a portfolio. '
        'Percentiles are exact grid values. Frequency percentiles are blank '
        'by design, because frequency enters through its PGF and no count '
        'distribution is ever materialized.' + _ENGINE_RIDER)})]


@summary.insurer.register(PnL)
def _summary_insurer_pnl(obj, blocks):
    """The same card with the business caption and the Agg row flags."""
    block_name, df, kw = blocks[0]
    if df.empty:
        return [(block_name, df, dict(kw, caption=_NO_ENGINE))]
    caption = ('Moments and key percentiles of the wrapped book by component '
               '(Freq, Sev, Agg). Percentiles are exact grid values. '
               'Frequency percentiles are blank by design: frequency enters '
               'through its PGF and no count distribution is materialized.'
               + _ENGINE_RIDER)
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_summary_flags(df)))]


@tail.register(PnL)
def _tail_frames(obj):
    """The wrapped book's return period ladder ([Overview-Engine]).

    Serves :attr:`PnL.engine_tail_df`, which is a **loss** distribution in
    loss orientation. The closing margin's ladder is a payoff and is the
    ``economic_tail`` exhibit; the two differ in which tail is the adverse
    one, so they are deliberately not the same leaf.
    """
    df = obj.engine_tail_df
    if df.empty:
        return [('engine_tail_df', df, {'caption': _NO_ENGINE})]
    return [('engine_tail_df', df, {'caption': (
        'Return period ladder for the book this P&L is a ledger over, read '
        'off the realized grid: VaR (the quoted number), TVaR (the priced '
        'number), excess VaR over the mean (the capital), and VaR to mean '
        'leverage. A loss distribution, so the adverse tail is the high one; '
        'the closing margin is a payoff and reads off the other half of the '
        'ladder, under the economic exhibits.' + _ENGINE_RIDER)})]


@tail.insurer.register(PnL)
def _tail_insurer_pnl(obj, blocks):
    """The same ladder with the capital anchors emphasized."""
    block_name, df, kw = blocks[0]
    if df.empty:
        return [(block_name, df, dict(kw, caption=_NO_ENGINE))]
    caption = ('Return period ladder for the wrapped book: VaR (the quoted '
               'number), TVaR (the priced number), excess VaR over the mean '
               '(capital), and VaR to mean leverage, exact from the FFT '
               'grid. A loss distribution, so the adverse tail is the high '
               'one, and the 1 in 200 (99.5%, Solvency II) and 1 in 250 '
               '(99.6%, US capital adequacy) anchors are emphasized.'
               + _ENGINE_RIDER)
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_tail_flags(df)))]


@stats.insurer.register(PnL)
def _stats_insurer_pnl(obj, blocks):
    """The engine's moment store, or an explanation of why there is none.

    Since [PnL-Economic-Frames] a P&L's ``stats_df`` is its engine's moment
    store, so the treatment is the ordinary one. A hand-built kernel P&L
    carries no engine and the frame is empty; rather than serve a blank
    table with no explanation, the caption says what happened. The ledger
    itself is the ``economic`` exhibit.
    """
    block_name, df, kw = blocks[0]
    if df.empty:
        caption = ('This P&L was built from grids rather than from a '
                   'declared book, so it carries no stochastic engine and '
                   'has no moment store to report. Its accounting view is '
                   'the economic exhibit.')
        return [(block_name, df, dict(kw, caption=caption))]
    return _stats_insurer_moment_store(obj, blocks)


# The reins exhibit serves through the engine ([PnL-Reins-Passthrough]):
# the delegating ``PnL.reins_stats_df`` / ``reins_summary_df`` properties
# make the shared ``_reins_frames`` builder work unchanged, and the insurer
# translation is the Aggregate one (it reads only the served frames), so it
# is registered for PnL where it lives, in ``exhibits._aggregate``.
# Availability looks through ``obj.engine`` (``_has_reinsurance``).
reins.register(PnL)(_reins_frames)


@validation.register(PnL)
def _validation_frames(obj):
    """The engine's moment QA first, the ledger audit only when it has rows.

    [PnL-Overview-Punchups]: every DecL-built P&L wraps a live engine whose
    ``validation_df`` is the moment QA a Validation tab wants, while the
    P&L's own ``validation_df`` is a per-leg **rebucketing audit** that is
    empty on every DecL build (no builder passes ``bs=`` to a ``Leg``).
    Serving the empty audit alone made the tab blank; serving the engine's
    frame first fixes that, and the audit stays as a second block exactly
    when it has something to say.
    """
    out = [('engine_validation_df', obj.engine_validation_df, {'caption': (
        'Moment QA for the wrapped book: the reference moment against the '
        'realized FFT estimate, with noise aware relative errors, for the '
        'frequency, severity and aggregate. Errors of this size are '
        'discretization, not model error. Empty on a hand-built kernel '
        'P&L, which carries no engine.')})]
    audit = obj.validation_df
    if not audit.empty:
        out.append(('validation_df', audit, {'caption': (
            'Ledger QA: each declared amount against the realized estimate, '
            'with absolute and relative error. Served only for legs carrying '
            'their own rebucketing grid.')}))
    return out


# ``economic`` RAW serves the two ladders side by side since
# [Ledger-Both-Ladders]; its insurer translation (below) keeps reading the
# first block only, so INSURER stays the abbreviated single sheet.
# ``economic_ratios`` carries two blocks, so it needs a builder either way.

@economic.register(PnL)
def _economic_frames(obj):
    """The ledger, on both ladders when they differ ([Ledger-Both-Ladders]).

    Two readings of one sheet. :attr:`PnL.economic_df` carries the scenario
    (``κ``) ladder, "the state the book is in": each cell a conditional
    mean, footing down the sheet. :attr:`PnL.economic_marginal_df` carries
    each row's own quantiles under plain ``P`` headers, "each row on its
    own": always available, never footing. When ``economic_df`` is itself
    marginal (an ineligible build: no shared atoms and no Palm ladder) the
    two sheets coincide and only the first is served, with a caption that
    says which regime is in force.
    """
    df = obj.economic_df
    scenario = any(str(c).startswith('κ') for c in df.columns)
    if not scenario:
        return [('economic_df', df, {'caption': (
            'The full ledger by side and label in currency units. This '
            'ledger shares no atoms across its rows and has no scenario '
            'ladder, so the P columns are each row\'s own marginal '
            'quantiles: the state the book is in is not computable here, '
            'and the ladder does not foot.')})]
    return [
        ('economic_df', df, {'caption': (
            'The full ledger by side and label in currency units, then the '
            'kappa columns: what each line comes to when the book as a '
            'whole lands at that percentile, the state the book is in. '
            'Conditional means add, so every column foots down the sheet.')}),
        ('economic_marginal_df', obj.economic_marginal_df, {'caption': (
            'The same ledger with each row on its own: the P columns are '
            'per row marginal quantiles, read off each row\'s own '
            'distribution. Quantiles never add, so this ladder does not '
            'foot; the one row the two sheets share is the grand result, '
            'whose scenario cells are its own quantiles.')}),
    ]

def _ledger_row_flags(obj, df):
    """Positional row flags for a ledger sheet, from the ledger plan.

    ``obj._plan`` and the frame's rows are built in the same order and are
    1:1 by construction, but this is presentation code reading a private
    attribute, so a length mismatch declines to flag rather than mislabeling
    rows: a wrong ``total`` on the wrong line is worse than no emphasis.
    """
    plan = getattr(obj, '_plan', None)
    if plan is None or len(plan) != len(df):
        return {}
    flags = {i: LEDGER_ROW_FLAGS[kind]
             for i, (_label, kind, _payload) in enumerate(plan)
             if kind in LEDGER_ROW_FLAGS}
    # A single group ledger has no ``grand_result``: its one group result IS
    # the bottom line, so nothing would carry ``total`` and the sheet would
    # read as all subtotals. Promote the last result row in that case.
    if flags and not any('total' in f for f in flags.values()):
        last = max(i for i, f in flags.items() if 'subtotal' in f)
        flags[last] = ('total',)
    return flags


#: The moment columns the abbreviated insurer ledger keeps
#: ([Ledger-Insurer-Abbreviated]). ``Skew`` comes off: the reading a ledger
#: is scanned for is level, spread and one tail, and a third moment on every
#: line is width the reader pays for and rarely spends.
LEDGER_MOMENTS = ('EX', 'SD', 'CV')

#: The single ladder point the abbreviated insurer ledger keeps: the **first**
#: rung, which is the adverse one. P&Ls are in payoff sign convention, left
#: tail bad, so ``κ01`` is the bad state and ``κ99`` is the benign one; an
#: abbreviation ending at the top of the ladder would report the good news and
#: read as the bad. The full ladder stays one perspective away, on RAW.
LEDGER_TAIL_Q = PERCENTILE_LADDER[0]


def _ledger_columns(df, scenario):
    """The abbreviated insurer ledger's columns, in sheet order.

    Returns the labels actually present, so a ledger built by some future
    route that names its ladder differently loses a column rather than
    raising on a reindex: this is presentation code, and the same declining
    rule :func:`_ledger_row_flags` follows.
    """
    tail = (_kappa_label if scenario else _pct_label)(LEDGER_TAIL_Q)
    return [c for c in (*LEDGER_MOMENTS, tail) if c in df.columns]


@economic.insurer.register(PnL)
def _economic_insurer(obj, blocks):
    """The abbreviated ledger, with its footing rules and the kappa semantics
    said aloud.

    RAW is the whole sheet, every moment and the full percentile ladder.
    INSURER is the reading version ([Ledger-Insurer-Abbreviated]): ``EX``,
    ``SD``, ``CV`` and the adverse tail state, four columns wide, because
    thirteen columns of ladder is a frame to slice rather than a sheet to
    read, and the app has the RAW toggle for that.

    The one thing a reader must not get wrong about the tail column is what it
    means: it is a **scenario state**, not a per row quantile, so it foots down
    the sheet, and ``κ01`` is the adverse state under the payoff convention.
    When the ledger has no shared atoms the header falls back to plain ``P``
    and no conditioning happened, which changes how the column reads entirely,
    so the caption says which regime is in force.
    """
    block_name, df, kw = blocks[0]
    scenario = any(str(c).startswith('κ') for c in df.columns)
    df = df[_ledger_columns(df, scenario)]
    caption = (
        'The ledger in currency units: declared legs, side totals and '
        'results, in ledger order. Signed as booked, so every column adds '
        'down the sheet. EX, SD and CV are marginal row properties.')
    if scenario:
        caption += (
            ' The kappa column holds scenario states, not per row quantiles: '
            'kappa-01 is the state in which the grand result lands at its 1% '
            'quantile, and each cell is the conditional mean of that row in '
            'that state, so the column foots exactly. Direction is uniform in '
            'the outcome, so kappa-01 is the adverse state (payoff '
            'convention, left tail bad) and a loss sensitive premium '
            'correctly reads high there. Skew and the rest of the percentile '
            'ladder are on the raw view.')
    else:
        caption += (
            ' This ledger shares no atoms across its rows (a one sweep or '
            'stitched route), so the ladder is marginal under a plain P '
            'header: P01 is that row\'s own 1% quantile, no conditioning '
            'happened, and the column does not foot. Skew and the rest of the '
            'percentile ladder are on the raw view.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_ledger_row_flags(obj, df)))]


@economic_ratios.register(PnL)
def _economic_ratios_frames(obj):
    """The per block amounts and ratios, plus the itemized declared legs."""
    return [
        ('economic_ratios_df', obj.economic_ratios_df,
         {'caption': 'Raw materials: the amounts each block contributes and '
                     'the loss, expense and combined ratios they imply, in '
                     'one frame to slice. Currency and ratio columns sit '
                     'side by side here; the insurer view separates them.'}),
        ('legs_df', obj.legs_df,
         {'caption': 'The declared legs, one row each, as written into the '
                     'ledger.'}),
    ]


#: The ratio frame splits into pure blocks under INSURER, per the reporting
#: guideline that a column carries one unit: currency amounts, then the
#: dimensionless ratios, then the itemized legs.
_AMOUNT_COLS = ('P', 'L', 'E', 'M')
_RATIO_COLS = ('LR', 'ER', 'CR', 'E_LR', 'E_ER', 'E_CR', 'P_share', 'M_share')


def _ratio_row_flags(obj, df):
    """Total on the last row, and the net-of-tier rows muted.

    The same treatment the ledger gives them (:data:`LEDGER_ROW_FLAGS`): a
    net-of-tier row is the running position through the tier, a cumulative
    reading aid rather than a block of its own, so it reads quiet. Matched by
    label against the P&L's own net-span labels, with the ledger exhibit's
    declining rule: presentation code never guesses a row.
    """
    nets = set(getattr(obj, '_net_span_labels', {}).values())
    flags = {i: ('muted',) for i, label in enumerate(df.index)
             if label in nets}
    if len(df) > 1:
        flags[len(df) - 1] = ('total',)
    return flags


@economic_ratios.insurer.register(PnL)
def _economic_ratios_insurer(obj, blocks):
    """Split amounts from ratios, so no column mixes two units.

    The raw frame is deliberately mixed: it is raw materials, the frame to
    slice and pivot. Presented, it wants the reporting rule applied, one unit
    per column, which means two blocks rather than one wide one. The itemized
    legs block stays on RAW (author's ruling, 2026-10-01): the insurer view is a
    client-facing exhibit, and a third table by leg confused more than it
    itemized.
    """
    (ratios_name, ratios_df, ratios_kw), _legs = blocks
    amounts = [c for c in _AMOUNT_COLS if c in ratios_df.columns]
    ratios = [c for c in _RATIO_COLS if c in ratios_df.columns]
    flags = _ratio_row_flags(obj, ratios_df)
    out = []
    if amounts:
        out.append((
            'amounts', ratios_df[amounts],
            dict(ratios_kw, row_flags=flags, caption=(
                'Premium, loss and expense per block, signed in the gross '
                'direction so they add across blocks and the margin identity '
                'M = P - L - E holds exactly. Loss absorbs cession recoveries '
                'and any unclassified obligation leg; expense absorbs ceding '
                'commission, a contra expense. A net of tier row is the '
                'running position through that tier and sits outside the '
                'sum, which is why it reads muted. The raw view itemizes '
                'the declared legs.'))))
    if ratios:
        # no ratio_cols here: every one of these labels points at the `ratio`
        # style in the format sheets, which stamps greater_tables' own ratio
        # column tag as well as the reading ([Format-Sheets] decision 6)
        out.append((
            'ratios', ratios_df[ratios],
            dict(ratios_kw, row_flags=flags,
                 caption=(
                'LR, ER and CR are ratios of means, the convention of a rate '
                'filing, re-derived from each block\'s own amounts and never '
                'averaged from the blocks below. The E_ columns are the same '
                'three as means of ratios. The pairs agree identically when '
                'premium is deterministic and part company exactly when it is '
                'random and correlated with loss, which is what a retro, a '
                'swing, a slide or a profit commission is. Share columns are '
                'against the first (gross) block.'))))
    return out


# --- the waterfall ([Exhibits-Waterfall]) -----------------------------------

@economic_waterfall.register(PnL)
def _economic_waterfall_frames(obj):
    """Serve the two walk frames the P&L publishes, with their captions.

    The arithmetic moved onto :attr:`PnL.walk_df` and
    :attr:`PnL.evaluation_df` at ``1.0.0a253``: a RAW block is exactly one
    public frame, and this exhibit was the one place left where the exhibit
    layer invented the frame it served.
    """
    walk_df, evaluation_df = obj.walk_df, obj.evaluation_df
    idx = walk_df.index
    t = WATERFALL_RETURN_PERIOD
    # the walk's own heading for the 1-in-t state, derived the same way the
    # frame derives it, so the caption and the columns cannot drift apart
    m = f'M{100 // t:02d}'
    net_available = not walk_df[f'{m} div net'].isna().all()
    gross_col = walk_df[f'{m} div gross']
    gross_available = not gross_col.isna().all()
    gross_truncated = gross_available and gross_col.isna().any()

    # Total on the closing net; the net-of-tier rows muted, the ledger's own
    # treatment: each is the running position through its tier, a cumulative
    # reading aid that sits outside the walk's sum.
    nets = set(getattr(obj, '_net_span_labels', {}).values())
    total_row = {i: ('muted',) for i, step in enumerate(idx) if step in nets}
    if len(idx) > 1:
        total_row[len(idx) - 1] = ('total',)
    walk_caption = (
        f'The margin walk in currency: gross, what each layer cedes, and the '
        f'closing net. Margin is the expected result and {m} is the result in '
        f'the 1-in-{t} state. The div columns are one conditioning each and '
        f'answer different questions: div net conditions on the whole book '
        f'landing at its own 1-in-{t}, the return on the capital the firm '
        f'actually holds, while div gross conditions on the gross result '
        f'landing at its own 1-in-{t}, the program\'s performance in the '
        f'gross stress state. Each is a ladder of conditional means and '
        f'foots down the walk where complete. The standalone column is '
        f'two-sided by role: a risk '
        f'bearing step reads its own left tail, the state that calls for '
        f'capital, while a ceded step reads its own right tail, the '
        f'writer\'s 1-in-{t}, the state in which the cover pays most and '
        f'the capital the writer of that cover would hold, so the cell is '
        f'positive there and the sign flip marks which reading a row takes. '
        f'Tail measures do not add, so standalone does not foot, and its '
        f'gap against the div columns is the diversification benefit. A '
        f'muted net of tier row is the running position through that tier, '
        f'the book once that tier\'s program has worked; it restates the '
        f'rows above it, so the footing runs over the unmuted steps alone.')
    if not net_available and not gross_available:
        walk_caption += (
            ' Both div columns are blank here: this ledger shares no '
            'atoms across its rows and carries no Palm ladder, so no '
            'conditioning was possible.')
    elif gross_truncated:
        walk_caption += (
            ' The div gross column ends at the aggregate tier: the gross '
            'basis cannot see through a nonlinear aggregate transform, so '
            'the aggregate cover and everything downstream of it are blank '
            'and the column foots only over the sub-ledger it serves.')
    evaluation_caption = (
        f'The same walk read as ratios: the rubric for the program. Margin '
        f'ratio is 1 less CR, the margin per unit of premium. MSD is '
        f'margin over its own standard deviation, a multiple. The CoC '
        f'columns are one quotient, M over the negated {m}, on the '
        f'standalone, net diversified and gross diversified capital bases: '
        f'net reads as the return on held capital, gross as the step\'s '
        f'performance in the gross stress state. The sign convention '
        f'carries the reading: gross and net rows have margin above zero '
        f'and {m} below, so the ratio is the return earned on the capital '
        f'that state calls for; a ceded row has both reversed (margin '
        f'below zero, the writer\'s right-tail {m} above), so the same '
        f'arithmetic reads as a cost, the margin given up per unit of '
        f'capital. The reinsurance test is whether a ceded row\'s CoC '
        f'comes in under the return the risk-bearing rows earn.')

    return [
        ('walk_df', walk_df, dict(caption=walk_caption, row_flags=total_row)),
        # 'Premium spent', 'Margin spent' and 'CR' carry the `ratio` style in
        # the format sheets, which tags them as ratios wherever they appear
        ('evaluation_df', evaluation_df,
         dict(caption=evaluation_caption, row_flags=total_row)),
    ]


# --- the closing margin's ladder ([Overview-Engine]) ------------------------

#: Return period ladder column the INSURER reading of ``economic_tail`` drops.
#: ``GridDistribution.tvar`` is the upper tail measure ``E[X | X > VaR(p)]`` at
#: every ``p``, with no orientation flip. A P&L is a payoff, read off the lower
#: half of the ladder, so on exactly the rungs this exhibit exists for the
#: column averages almost the whole distribution and sits near the mean: it
#: pairs a downside ``VaR`` with the other side's conditional mean. The
#: matching lower measure ``E[X | X <= VaR(p)]`` is not computed today, so the
#: column comes off the sheet that gets read and stays on RAW until it is
#: (author's ruling, 2026-10-06). See the warning on
#: :meth:`PnL.tail_periods_df`, which records the same defect on
#: ``Aggregate.tail_df`` and ``Portfolio.tail_df``.
_PAYOFF_TVAR = 'TVaR'


@economic_tail.register(PnL)
def _economic_tail_frames(obj):
    """The closing margin's return period ladder, whole."""
    return [('tail_df', obj.tail_df, {'caption': (
        'Return period ladder over the closing margin, in payoff '
        'orientation: the adverse tail is the low one, so the ladder walks '
        'into the losses and the 1 in 200 year is the rung that goes '
        '200-to-1 against you. VaR is the quoted number, excess VaR over the '
        'mean the capital that rung calls for, and VaR to mean the leverage. '
        'Read the TVaR column with care: it is the measure above each rung, '
        'not below, so on a payoff it pairs a downside VaR with the other '
        'side\'s conditional mean. The insurer view drops it for that '
        'reason.')})]


@economic_tail.insurer.register(PnL)
def _economic_tail_insurer(obj, blocks):
    """The same ladder without ``TVaR``, and with the capital anchors flagged.

    The one INSURER restructure here is a dropped column, on the same
    reasoning that abbreviates the ledger ([Ledger-Insurer-Abbreviated]): a
    reading sheet carries what is read, and what is misleading on this
    orientation stays one perspective away rather than on the page. See
    :data:`_PAYOFF_TVAR`.
    """
    block_name, df, kw = blocks[0]
    df = df[[c for c in df.columns if c != _PAYOFF_TVAR]]
    caption = (
        'Return period ladder over the closing margin. A payoff, so the '
        'adverse tail is the LOW one: the 1 in 200 year is the rung that '
        'goes 200-to-1 against you, and the ladder walks into the losses as '
        'you read up. VaR is the quoted result at that rung, excess VaR its '
        'shortfall against expectation (negative on the downside, which is '
        'the capital the state calls for), and VaR to mean the leverage, '
        'blank near break-even where the ratio has no content. The 1 in 200 '
        '(99.5%, Solvency II) and 1 in 250 (99.6%, US capital adequacy) '
        'anchors are emphasized on both sides of the ladder. TVaR is on the '
        'raw view only: the library computes the measure above a rung, which '
        'on a payoff is the wrong side, so it would read as a tail average '
        'and sit near the mean.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_tail_flags(df)))]
