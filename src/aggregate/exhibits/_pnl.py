"""Exhibit treatments for :class:`~aggregate._pnl.PnL`.

The accounting family: ``economic`` (the ledger sheet) and
``economic_ratios``, plus the ``stats`` override that explains an absent
engine. Since [PnL-Economic-Frames] a P&L's ``stats_df`` is its engine's
moment store, so ``stats`` is the ordinary treatment rather than a special
case; the ledger has its own exhibit under its own name.
"""

import numpy as np
import pandas as pd

from .._pnl import (
    PERCENTILE_LADDER, PnL, WATERFALL_RETURN_PERIOD, _kappa_label, _pct_label,
)
from ._core import (
    economic, economic_ratios, economic_waterfall, reins, stats, validation,
    _reins_frames, _stats_insurer_moment_store,
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


@economic_ratios.insurer.register(PnL)
def _economic_ratios_insurer(obj, blocks):
    """Split amounts from ratios, so no column mixes two units.

    The raw frame is deliberately mixed: it is raw materials, the frame to
    slice and pivot. Presented, it wants the reporting rule applied, one unit
    per column, which means two blocks rather than one wide one. The legs
    block passes through with a caption noting what it omits.
    """
    (ratios_name, ratios_df, ratios_kw), (legs_name, legs_df, legs_kw) = blocks
    amounts = [c for c in _AMOUNT_COLS if c in ratios_df.columns]
    ratios = [c for c in _RATIO_COLS if c in ratios_df.columns]
    total_row = {len(ratios_df) - 1: ('total',)} if len(ratios_df) > 1 else {}
    out = []
    if amounts:
        out.append((
            'amounts', ratios_df[amounts],
            dict(ratios_kw, row_flags=total_row, caption=(
                'Premium, loss and expense per block, signed in the gross '
                'direction so they add across blocks and the margin identity '
                'M = P - L - E holds exactly. Loss absorbs cession recoveries '
                'and any unclassified obligation leg; expense absorbs ceding '
                'commission, a contra expense. The legs block below itemizes '
                'both.'))))
    if ratios:
        # no ratio_cols here: every one of these labels points at the `ratio`
        # style in the format sheets, which stamps greater_tables' own ratio
        # column tag as well as the reading ([Format-Sheets] decision 6)
        out.append((
            'ratios', ratios_df[ratios],
            dict(ratios_kw, row_flags=total_row,
                 caption=(
                'LR, ER and CR are ratios of means, the convention of a rate '
                'filing, re-derived from each block\'s own amounts and never '
                'averaged from the blocks below. The E_ columns are the same '
                'three as means of ratios. The pairs agree identically when '
                'premium is deterministic and part company exactly when it is '
                'random and correlated with loss, which is what a retro, a '
                'swing, a slide or a profit commission is. Share columns are '
                'against the first (gross) block.'))))
    out.append(('legs', legs_df, dict(legs_kw, caption=(
        'One row per declared leg, the itemized companion. Derived rows '
        '(totals, results, running nets) are absent by design: they are sums '
        'of these.'))))
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

    total_row = {len(idx) - 1: ('total',)} if len(idx) > 1 else {}
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
        f'gap against the div columns is the diversification benefit.')
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
        f'The same walk read as ratios: the rubric for the program. MSD is '
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
