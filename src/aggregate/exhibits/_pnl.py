"""Exhibit treatments for :class:`~aggregate._pnl.PnL`.

The accounting family: ``economic`` (the ledger sheet) and
``economic_ratios``, plus the ``stats`` override that explains an absent
engine. Since [PnL-Economic-Frames] a P&L's ``stats_df`` is its engine's
moment store, so ``stats`` is the ordinary treatment rather than a special
case; the ledger has its own exhibit under its own name.
"""

import numpy as np
import pandas as pd

from .._pnl import PnL
from ._core import (
    MEASURE_FORMATS, economic, economic_ratios, economic_waterfall, stats,
    _stats_insurer_moment_store,
)

#: Return period the waterfall evaluates capital at. A business choice, not a
#: law, and deliberately a constant here rather than a kwarg: the exhibit is
#: the place to change it (author, 2026-08-05, declining to build a
#: configurable capital-level framework).
WATERFALL_RETURN_PERIOD = 100

#: Ledger row kinds that ARE a step's own result. A running net is excluded:
#: it is the cumulative position after the step, not the step itself.
_RESULT_KINDS = ('group_result', 'tier_result', 'grand_result')

#: Ledger row kind to greater_tables row flag ([Exhibits-Economic-Insurer]).
#: The grand result is the bottom line; each group's and each tier's own
#: result, and the grand side totals, are subtotals; a running net is a
#: cumulative reading aid rather than a booked line, so it is muted. Legs and
#: ``total_impact`` are unflagged, the latter because it is a difference
#: between two positions rather than one of them.
LEDGER_ROW_FLAGS = {
    'grand_result': ('total',),
    'group_result': ('subtotal',),
    'tier_result': ('subtotal',),
    'grand_total': ('subtotal',),
    'running_net': ('muted',),
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


# ``economic`` itself is a plain passthrough over ``economic_df``, declared in
# the manifest; its insurer translation lives here. ``economic_ratios`` carries
# two blocks, so it needs a builder either way.

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


@economic.insurer.register(PnL)
def _economic_insurer(obj, blocks):
    """The ledger with its footing rules and the kappa semantics said aloud.

    The one thing a reader must not get wrong about this sheet is what the
    ladder columns mean: they are **scenario states**, not per row quantiles,
    so they foot down the sheet, and ``κ01`` is the adverse state under the
    payoff convention. When the ledger has no shared atoms the headers fall
    back to plain ``P`` and no conditioning happened, which changes how the
    columns read entirely, so the caption says which regime is in force.
    """
    block_name, df, kw = blocks[0]
    scenario = any(str(c).startswith('κ') for c in df.columns)
    caption = (
        'The ledger in currency units: declared legs, side totals and '
        'results, in ledger order. Signed as booked, so every column adds '
        'down the sheet. EX, SD, CV and Skew are marginal row properties.')
    if scenario:
        caption += (
            ' The kappa columns are scenario states, not per row quantiles: '
            'column kappa-q is the state in which the grand result lands at '
            'its q quantile, and each cell is the conditional mean of that '
            'row in that state, so every kappa column foots exactly. '
            'Direction is uniform in the outcome, so kappa-01 is the adverse '
            'state (payoff convention, left tail bad) and a loss sensitive '
            'premium correctly reads high there.')
    else:
        caption += (
            ' This ledger shares no atoms across its rows (a one sweep or '
            'stitched route), so the ladder is marginal under plain P '
            'headers: each cell is that row\'s own quantile, no conditioning '
            'happened, and the ladder columns do not foot.')
    return [(block_name, df,
             dict(kw, caption=caption, row_flags=_ledger_row_flags(obj, df),
                  formatters=MEASURE_FORMATS))]


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
_AMOUNT_COLS = ('P', 'L', 'E', 'C', 'M')
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
                'Premium, loss, expense and commission per block, signed in '
                'the gross direction so they add across blocks and the margin '
                'identity M = P - L - E - C holds exactly. Loss absorbs '
                'cession recoveries and any unclassified obligation leg.'))))
    if ratios:
        out.append((
            'ratios', ratios_df[ratios],
            dict(ratios_kw, ratio_cols=list(ratios), row_flags=total_row,
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

def _step_result_rows(obj):
    """Map each walk step to its own result row: ``{step: (position, label)}``.

    Reads the ledger plan rather than pattern matching the index, because a
    step's own result and the running net *after* it both sit under
    ``Margin`` and only the plan distinguishes them. The plan label doubles
    as the :attr:`PnL.density_df` key, which is how the standalone quantile
    is reached. A later result for the same step wins, so a tier subtotal
    supersedes the groups it spans.
    """
    plan = getattr(obj, '_plan', None)
    index = obj.economic_df.index
    if plan is None or len(plan) != len(index):
        return {}
    out = {}
    for i, (idx, (label, kind, _payload)) in enumerate(zip(index, plan)):
        if kind in _RESULT_KINDS and isinstance(idx, tuple):
            out[idx[0]] = (i, label)
    return out


def _capital_ratio(margin, bad_outcome):
    """``M / -M_100``: margin over the capital that outcome would call for.

    ``bad_outcome`` is the margin in the 1-in-100 state and is normally
    negative, so ``-bad_outcome`` is the capital you would have to inject and
    the ratio reads as a return on it. ``NaN`` when no capital is called for
    (a non-negative outcome, so nothing to inject and no denominator) or when
    either input is missing.
    """
    if not (np.isfinite(margin) and np.isfinite(bad_outcome)):
        return np.nan
    capital = -bad_outcome
    return margin / capital if capital > 0 else np.nan


@economic_waterfall.register(PnL)
def _economic_waterfall_frames(obj):
    """Build the walk and its evaluation from the ledger and the ratio frame.

    Arithmetic over numbers the P&L has already computed: the margin and its
    SD off :attr:`PnL.economic_df`, the diversified 1-in-100 off that frame's
    kappa column, the standalone 1-in-100 off each result row's own
    :class:`GridDistribution`, and the shares and combined ratio off
    :attr:`PnL.economic_ratios_df`. Nothing new is estimated here, which is
    why the exhibit layer is the right home for it.
    """
    ledger, ratios = obj.economic_df, obj.economic_ratios_df
    steps = _step_result_rows(obj)
    p = 1.0 / WATERFALL_RETURN_PERIOD
    kappa = f'κ{int(round(p * 100)):02d}'
    diversified_available = kappa in ledger.columns
    densities = obj.density_df

    walk, evaluation, index = [], [], []
    for step in ratios.index:
        if step not in steps:
            continue
        pos, label = steps[step]
        row = ledger.iloc[pos]
        margin, sd = float(row['EX']), float(row['SD'])
        gd = densities.get(label)
        standalone = float(gd.q(p)) if gd is not None else np.nan
        divers = float(row[kappa]) if diversified_available else np.nan
        r = ratios.loc[step]
        index.append(step)
        walk.append([margin, standalone, divers])
        evaluation.append([
            float(r['P_share']), float(r['M_share']), float(r['CR']),
            margin / sd if sd > 0 else np.nan,
            _capital_ratio(margin, standalone),
            _capital_ratio(margin, divers),
        ])

    idx = pd.Index(index, name='Step')
    t = WATERFALL_RETURN_PERIOD
    walk_df = pd.DataFrame(
        walk, index=idx,
        columns=['M', f'M @ 1-in-{t} standalone', f'M @ 1-in-{t} diversified'])
    evaluation_df = pd.DataFrame(
        evaluation, index=idx,
        columns=['Premium spent', 'Margin spent', 'CR', 'M / SD',
                 'M / capital standalone', 'M / capital diversified'])

    total_row = {len(idx) - 1: ('total',)} if len(idx) > 1 else {}
    walk_caption = (
        f'The margin walk in currency: gross, what each layer cedes, and the '
        f'closing net, each shown at its expected value and in its 1-in-{t} '
        f'state. The diversified column conditions on the whole book landing '
        f'at its own 1-in-{t}, so it foots down the walk exactly. The '
        f'standalone column is each step\'s own 1-in-{t}; tail measures do '
        f'not add, so it does not foot, and the gap between the two columns '
        f'is the diversification benefit.')
    if not diversified_available:
        walk_caption += (
            ' The diversified column is blank here: this ledger shares no '
            'atoms across its rows, so no conditioning was possible.')
    evaluation_caption = (
        f'The same walk read as ratios. Premium and margin spent are against '
        f'the gross block. Capital is the injection a 1-in-{t} outcome would '
        f'call for, so M / capital is a return on it; it is blank where the '
        f'step calls for no capital, which is what a purchased layer does in '
        f'the adverse state, and there the diversified column is the more '
        f'meaningful of the two.')

    return [
        ('walk', walk_df, dict(caption=walk_caption, row_flags=total_row)),
        ('evaluation', evaluation_df,
         dict(caption=evaluation_caption, row_flags=total_row,
              ratio_cols=['Premium spent', 'Margin spent', 'CR'])),
    ]
