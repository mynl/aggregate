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
    PERCENTILE_LADDER, PnL, WATERFALL_RETURN_PERIOD, _WATERFALL_SPLIT,
    _kappa_label, _pct_label,
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
#: ([Ledger-Insurer-Abbreviated]). Level, spread, spread relative to level, and
#: the third moment.
#:
#: ``Skew`` came off at a305, on the ground that a third moment on every line is
#: width the reader pays for and rarely spends. It is back at a391 because the
#: ledger **inherited** it: the P&L's own summary card was the one presented
#: frame that carried ``Skew``, and [Overview-Engine] retired that card's leaf,
#: so dropping the column here would have put the third moment one perspective
#: away on every sheet a reader sees. The a305 reasoning was about redundancy
#: and the redundancy is gone.
LEDGER_MOMENTS = ('EX', 'SD', 'CV', 'Skew')

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


#: The moment columns a delta ledger row has no distribution behind
#: ([Delta-Row-Marked]). ``EX`` is absent on purpose: the mean of a difference
#: is the difference of the means, exact by linearity, so it is the one cell on
#: such a row that means what it says.
DELTA_BLANK_MOMENTS = ('SD', 'CV', 'Skew')


def _blank_delta_cells(df, scenario):
    """Blank the cells a per-statistic delta ledger row cannot support.

    Parameters
    ----------
    df : pandas.DataFrame
        A ledger sheet, carrying the rows to treat in ``.attrs['delta_rows']``
        (:meth:`~aggregate._pnl.PnL._delta_row_entries`).
    scenario : bool
        Whether the sheet's ladder is the conditional (``κ``) one.

    Returns
    -------
    pandas.DataFrame
        A copy with those cells blank, or `df` unchanged when the sheet carries
        no delta row, which is every route that keeps real atoms.

    Notes
    -----
    The row this exists for is the stitched tower's total impact, the grand
    result less the first step's. Those two rows ride different marginals, so
    their difference has no distribution and every statistic on the row but the
    mean is a difference of the two rows' statistics. Before the mark reached
    the sheet a reader saw the consequence directly: a **negative standard
    deviation**, with nothing on the line to say it was a delta of statistics.

    Why the ladder treatment turns on `scenario`. A ``κ`` cell on a delta row is
    the difference of two conditional means under the **same** conditioning
    event, so by linearity it is the conditional mean of the difference: exact,
    and it foots down its column like every other cell. It stays. A plain ``P``
    cell is a difference of quantiles, which is not a quantile of anything
    (:class:`~aggregate._pnl._DeltaGD` says so in as many words), so it goes.

    RAW keeps the numbers, this being presentation code: a column slice or a
    blanked cell is not a public frame, and a caller who wants the raw deltas
    reads the frame off the class and the mark beside it.
    """
    rows = [r for r in df.attrs.get('delta_rows', ()) if r in df.index]
    if not rows:
        return df
    cols = [c for c in df.columns
            if c in DELTA_BLANK_MOMENTS
            or (not scenario and str(c).startswith('P'))]
    if not cols:
        return df
    out = df.copy()
    out.loc[rows, cols] = float('nan')
    # `.copy()` carries `.attrs` across, which is what a consumer downstream
    # (a caption, a second treatment) needs to know the row is still a delta.
    return out


@economic.insurer.register(PnL)
def _economic_insurer(obj, blocks):
    """The abbreviated ledger, with its footing rules and the kappa semantics
    said aloud.

    RAW is the whole sheet, every moment and the full percentile ladder.
    INSURER is the reading version ([Ledger-Insurer-Abbreviated]): the four
    moments of :data:`LEDGER_MOMENTS` and the adverse tail state, five columns
    wide, because thirteen columns of ladder is a frame to slice rather than a
    sheet to read, and the app has the RAW toggle for that.

    The one thing a reader must not get wrong about the tail column is what it
    means: it is a **scenario state**, not a per row quantile, so it foots down
    the sheet, and ``κ01`` is the adverse state under the payoff convention.
    When the ledger has no shared atoms the header falls back to plain ``P``
    and no conditioning happened, which changes how the column reads entirely,
    so the caption says which regime is in force.

    A stitched tower's impact row is treated by :func:`_blank_delta_cells`
    ([Delta-Row-Marked]): its spread cells go blank rather than printing a
    difference of statistics, which is how a negative standard deviation used
    to reach this sheet.
    """
    block_name, df, kw = blocks[0]
    scenario = any(str(c).startswith('κ') for c in df.columns)
    df = df[_ledger_columns(df, scenario)]
    deltas = bool(df.attrs.get('delta_rows'))
    df = _blank_delta_cells(df, scenario)
    caption = (
        'The ledger in currency units: declared legs, side totals and '
        'results, in ledger order. Signed as booked, so every column adds '
        'down the sheet. EX, SD, CV and Skew are marginal row properties.')
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
    if deltas:
        caption += (
            ' The impact row is a difference of two positions that ride '
            'different marginals, so it has no distribution of its own: its '
            'mean is exact by linearity and every spread cell is blank, '
            'because a difference of standard deviations is not the standard '
            'deviation of the difference.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_ledger_row_flags(obj, df)))]


@economic_ratios.register(PnL)
def _economic_ratios_frames(obj):
    """The per block amounts and ratios, plus the itemized declared legs."""
    return [
        ('economic_ratios_df', obj.economic_ratios_df,
         {'caption': 'Raw materials: the amounts each block contributes, the '
                     'standard deviation of its result, and the loss, expense '
                     'and combined ratios the amounts imply, in one frame to '
                     'slice. Currency, a spread and ratio columns sit side by '
                     'side here; the insurer view keeps the walk a reader '
                     'reads across and moves the expected-ratio columns and '
                     'the share columns off it.'}),
        ('legs_df', obj.legs_df,
         {'caption': 'The declared legs, one row each, as written into the '
                     'ledger.'}),
    ]


#: The walk the INSURER view of ``economic_ratios`` serves, in reading order:
#: what each block wrote, what it cost, what is left, how volatile that is, and
#: the three plain ratios of those amounts ([PnL-Summary-One-Table], a389).
#:
#: **This reverses the reporting rule that a column carries one unit**, which is
#: what split the view into an amounts block and a ratios block through a388.
#: Reversed deliberately, and here in the treatment rather than worked around by
#: whoever presents it: the thing an underwriter does with this table is read
#: **across a row**, and the money and the ratio of that money belong beside
#: each other when that is the motion. Two tables made the reader hold four
#: numbers in their head to cross a single block. ``greater_tables`` resolves
#: formats per column name, so nothing about the mixed block is hard to render.
#:
#: The columns left off are not deleted, they are on RAW. ``E_LR`` / ``E_ER`` /
#: ``E_CR`` are means of ratios and answer a question nobody asks of the sheet
#: that gets read every time, but they are the one thing that tells you a retro,
#: a swing or a profit-commission cession is behaving as advertised, so the
#: perspective toggle is exactly the right mechanism for them (author's ruling,
#: 2026-10-06). ``P_share`` / ``M_share`` go with them; the walk reports both
#: shares in its own first two columns.
_SUMMARY_COLS = ('P', 'L', 'E', 'M', 'SD', 'LR', 'ER', 'CR')


def _ratio_row_flags(obj, df):
    """Total on the last row; the gross row and the net-of-tier rows subtotal.

    **The three rows a reader looks for are bold.** The gross block is row 0 in
    every builder, which is the fact the waterfall docstring already relies on;
    the net-of-tier rows are the book once each tier's program has worked; and
    the closing row is the net position. Those are the positions, and everything
    between them is a step that gets from one to the next.

    This reverses the muting of a388, where a net-of-tier row read quiet because
    it restates the rows above it. That reason has not changed and is not what
    the emphasis was saying: a cumulative row still sits outside the footing, and
    the **caption** is where that is stated. Reading quiet and reading important
    are not opposites on a frame whose rows are not all the same kind of thing.

    ``subtotal`` and ``total`` both render at ``font-weight: 600`` in
    ``gt.css``, the latter with a rule above it, so the closing row still reads
    as the end of the walk rather than as one more position.

    Matched by label against the P&L's own net-span labels, with the ledger
    exhibit's declining rule: presentation code never guesses a row.
    """
    nets = set(getattr(obj, '_net_span_labels', {}).values())
    flags = {i: ('subtotal',) for i, label in enumerate(df.index)
             if label in nets}
    if len(df):
        flags[0] = ('subtotal',)
    if len(df) > 1:
        flags[len(df) - 1] = ('total',)
    return flags


@economic_ratios.insurer.register(PnL)
def _economic_ratios_insurer(obj, blocks):
    """One block: the walk, amounts and the ratios of those amounts together.

    The raw frame is deliberately mixed, being raw materials, the frame to slice
    and pivot. Through a388 this view applied the reporting rule that a column
    carries one unit and split it in two. [PnL-Summary-One-Table] reverses that
    here, for the reason written out on :data:`_SUMMARY_COLS`: the table is read
    across a row, so an amount and the ratio of that amount belong beside each
    other.

    The itemized legs block stays on RAW (author's ruling, 2026-10-01), as do the
    ``E_`` columns and the two share columns (2026-10-06).
    """
    (ratios_name, ratios_df, ratios_kw), _legs = blocks
    cols = [c for c in _SUMMARY_COLS if c in ratios_df.columns]
    if not cols:
        return []
    # no ratio_cols here: LR, ER and CR each point at the `ratio` style in the
    # format sheets, which stamps greater_tables' own ratio column tag as well
    # as the reading ([Format-Sheets] decision 6), and the amounts beside them
    # take `money` by the same route, so one mixed block formats itself
    return [(
        'walk', ratios_df[cols],
        dict(ratios_kw, row_flags=_ratio_row_flags(obj, ratios_df), caption=(
            'One row per block, read across: premium written, loss, expense, '
            'the margin left, the standard deviation of that margin, and the '
            'three ratios the amounts imply. P, L, E and M are signed in the '
            'gross direction, so they add across blocks and the identity '
            'M = P - L - E holds exactly; loss absorbs cession recoveries and '
            'any unclassified obligation leg, and expense absorbs ceding '
            'commission, a contra expense. SD is a marginal row property and '
            'does not add. LR, ER and CR are ratios of means, the convention '
            'of a rate filing, re-derived from each block\'s own amounts and '
            'never averaged from the blocks below, so CR satisfies '
            '1 - CR = M / P on every row. The three bold rows are the '
            'positions: gross, the book net of each tier, and the closing net. '
            'A net of tier row is the running position through that tier and '
            'sits outside the sum, so the footing runs over the steps between '
            'them alone. The raw view itemizes the declared legs and adds '
            'the expected-ratio columns, which are the same three read as '
            'means of ratios and part company with these exactly when premium '
            'is random and correlated with loss, the signature of a retro, a '
            'swing, a slide or a profit commission.')))]


# --- the waterfall ([Exhibits-Waterfall]) -----------------------------------

@economic_waterfall.register(PnL)
def _economic_waterfall_frames(obj):
    """The merged walk, whole: one frame, one block.

    The arithmetic moved onto the class at ``1.0.0a253``, a RAW block being
    exactly one public frame and this exhibit having been the one place left
    where the exhibit layer invented what it served. [Waterfall-Capital] merged
    the two frames it published into :attr:`PnL.waterfall_df`, so RAW is now an
    ordinary passthrough and the two-block reading is the INSURER restructure it
    always should have been.
    """
    return [('waterfall_df', obj.waterfall_df, {'caption': (
        'The margin walk and the capital behind it, one row per step that '
        'books a result: what each step spends, what it earns, the capital it '
        'calls for on three bases and what that capital costs on each. '
        'Capital is signed as capital, so a cession reads negative: capital '
        'released. The level it is struck at is in the frame\'s '
        'return_period attribute. Those amounts are notional, with nothing in '
        'the ledger truncated at them. The insurer view splits this into the '
        'walk and the capital bases and says how to read each.')})]


@economic_waterfall.insurer.register(PnL)
def _economic_waterfall_insurer(obj, blocks):
    """Two blocks, split after ``MSD``, each with its reading.

    **The split is by question rather than by unit**, which is the point of
    [Waterfall-Capital]. The first block is the walk: what each step spends and
    what it earns. The second is three capital bases, each beside its own cost
    of capital, which is the comparison the frame exists to support and which
    through a389 meant reading across two tables. Splitting one frame into
    several blocks is exactly the liberty ``[Perspective-May-Restructure]``
    grants this perspective and no other.
    """
    _name, df, kw = blocks[0]
    idx = df.index
    t = df.attrs.get('return_period', WATERFALL_RETURN_PERIOD)
    net_available = not df['Capital net'].isna().all()
    gross_col = df['Capital gross']
    gross_available = not gross_col.isna().all()
    gross_truncated = gross_available and gross_col.isna().any()

    # Total on the closing net; the gross row and the net-of-tier rows
    # subtotal, which is a397's reversal of a388's muting. The three are the
    # positions a reader of a margin walk looks for, the steps between them get
    # from one to the next, and the gross block is row 0 in every builder. That
    # a cumulative row sits outside the walk's sum has not changed and is the
    # caption's to say; see `_ratio_row_flags`.
    nets = set(getattr(obj, '_net_span_labels', {}).values())
    total_row = {i: ('subtotal',) for i, step in enumerate(idx) if step in nets}
    if len(idx):
        total_row[0] = ('subtotal',)
    if len(idx) > 1:
        total_row[len(idx) - 1] = ('total',)

    split = list(df.columns).index(_WATERFALL_SPLIT) + 1
    walk_cols, capital_cols = df.columns[:split], df.columns[split:]

    walk_caption = (
        'The margin walk: gross, what each layer cedes, and the closing net. '
        'Premium spent and Margin spent are the step against the gross block, '
        'Margin ratio is the margin per unit of premium, Margin is the '
        'expected result, and MSD is that margin over its own standard '
        'deviation, a multiple. The bold rows are the positions: gross, the '
        'book net of each tier, and the closing net. A net of tier row is the '
        'running position through that tier, the book once that tier\'s '
        'program has worked; it restates the rows above it, so the footing '
        'runs over the steps between the positions alone. The capital each '
        'step calls for, '
        'and what that capital costs, are in the table below.')
    capital_caption = (
        f'Three capital bases for the same walk, each beside its own cost of '
        f'capital, all struck at the 1-in-{t} state. Capital is the negated '
        f'result in that state, so a risk-bearing row reads the capital it '
        f'holds and a ceded row reads a negative number: the capital the cover '
        f'releases, which is what a cover does. The CoC columns are margin '
        f'over the capital beside them, on every row, so they read as a return '
        f'where capital is held and as the price paid per unit released where '
        f'it is given back, with no change of arithmetic. The reinsurance test '
        f'is whether a ceded row\'s CoC comes in under the return the '
        f'risk-bearing rows earn. Standalone is each step\'s own 1-in-{t}, '
        f'two-sided by role: a risk-bearing step reads its own adverse tail, a '
        f'ceded step the writer\'s, the state in which the cover pays most. '
        f'Tail measures do not add, so standalone does not foot, and its gap '
        f'against the other two is the diversification benefit. Net conditions '
        f'on the whole book landing at its own 1-in-{t}, which decomposes the '
        f'capital the firm actually holds and is the basis for attribution; '
        f'gross conditions on the gross result landing at its own, which reads '
        f'the program as a stress test. Both are ladders of conditional means '
        f'and foot down the walk where complete. These capital amounts are '
        f'NOTIONAL: nothing in the ledger is truncated at them, no default is '
        f'modeled and no loss is limited by them. They are the capital the '
        f'state calls for, read off a quantile.')
    if not net_available and not gross_available:
        capital_caption += (
            ' Both conditional columns are blank here: this ledger shares no '
            'atoms across its rows and carries no Palm ladder, so no '
            'conditioning was possible.')
    elif gross_truncated:
        capital_caption += (
            ' The gross column ends at the aggregate tier: the gross basis '
            'cannot see through a nonlinear aggregate transform, so the '
            'aggregate cover and everything downstream of it are blank and '
            'the column foots only over the sub-ledger it serves.')

    # `walk` and `capital`, not `capital at 1-in-100`: block names are
    # identifiers, in snake case like every other one, and the level in a name
    # is what [Waterfall-Capital] just took out of the columns. The plan's third
    # home for it, a visible block heading, does not exist; a consumer draws the
    # table and its caption, so the level rides in the caption and in
    # `.attrs['return_period']`.
    #
    # No ratio_cols on either block: 'Premium spent', 'Margin spent',
    # 'Margin ratio' and the three CoC columns each point at the `ratio` style
    # in the format sheets, which stamps greater_tables' own tag wherever they
    # appear.
    return [
        ('walk', df[walk_cols],
         dict(kw, caption=walk_caption, row_flags=total_row)),
        ('capital', df[capital_cols],
         dict(kw, caption=capital_caption, row_flags=total_row)),
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
