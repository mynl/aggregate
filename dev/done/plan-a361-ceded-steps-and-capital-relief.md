# plan-a361: ceded steps in the margin walk, and the capital they release

**Status:** drafted 2026-09-29, awaiting the author's ruling on one naming
question in [relief-column]. Not yet implemented.
**Repo:** `V:\worktrees\aggregate_REFACTOR`, the `aggregate` library, bumping
`1.0.0a360` to `1.0.0a361`. No change in `aggregate_api`.

## Goal

Two defects in the `economic_waterfall` exhibit, both about ledger rows that
**hedge** rather than **bear risk**. A cession is not a risk unit, and the walk
currently asks it two questions that only a risk unit can answer, then declines
to ask it the one question it can.

1. `M01 standalone` is not a capital measure on a ceded step. It is blanked
   there, along with the ratio built on it.
2. `M / M01 diversified` is blanked on exactly the rows where it carries the
   most information. The guard is corrected and the column is split in two, so
   that no column mixes two meanings.

Nothing about the risk-bearing rows changes. **The `M01 diversified` walk
column is correct as it stands and is not touched**: its cells are correctly
signed and the column foots. Only the ratio derived from it moves.

## Background: what the walk is for

`PnL.walk_df` and `PnL.evaluation_df` (`src/aggregate/_pnl.py:2100`, built in
one pass by `_waterfall_frames`) are served as the two blocks of the
`economic_waterfall` exhibit (`src/aggregate/exhibits/_pnl.py:226`). The walk
runs gross, then each cover, then the closing net, one row per step that books a
result of its own.

Between them the two frames answer three questions, and today only two of the
answers are sound.

**What does each step contribute to the expected result?** `Margin`. Foots by
linearity. Sound, and untouched by this plan.

**What does each step contribute to the capital the book's 1-in-100 calls
for?** `M01 diversified`, which is `E[X_i | X = x_q]` off the ledger's kappa
column, the Euler allocation of VaR, which is why it foots exactly. Sound for
every row type, cessions included: on a cession the cell is positive and is the
capital relief the cover buys. Untouched by this plan.

**How much of that is pooling benefit rather than the step's own risk?**
`M01 standalone`. This is the one that breaks, because the counterfactual it
rests on is "stand this row up as its own company". A risk unit can be stood up
alone. A hedge cannot.

`WATERFALL_RETURN_PERIOD` is `100` (`_pnl.py:784`) and binds `t` in both files.
`M01` reads as the margin at the 1st percentile: a P&L is payoff oriented
(`PNL_IS_LOSS_VALUE = False`, `_pnl.py:127`), so the adverse tail is the low
one.

## Current behavior, with evidence

The worked example throughout is the author's capstone program, which produces
a two-cover tower with a computable kappa ladder. It builds by the per-atom
route, not the stitched one, so the diversified column is populated:
`xpnl peel (bottom-up): 2-layer per-atom tower (Gross -> XOL layer -> QS ->
All) over the occurrence (gross, ceded) joint; scenario (kappa) ladder`.

```
xpnl Capstone.PnL
  derive premium
  less
    agg Capstone.FullProgram
      [2000 5000 10000 3000 1000] premium as GWP at [0.95 0.95 0.9 0.85 0.75] lr
      [500 1000 5000 10000 5000] xs [0 0 0 0 1000]
      sev 151.63266492815836 * lognorm 1
        picks [500 1000 2000 5000 10000] [14900 2250 1250 500 50]
      occurrence net of
        90% po 4500 xs 500 deposit 6000 as "XOL layer"
      mixed gamma 0.125
      aggregate net of
        75% po inf xs 0 rate 1 cede 0.275 as QS
  less
    0.25 premium expenses as "G&A"
  peel bottom-up
```

It serves, today:

```
            Margin  M01 standalone  M01 diversified
Gross      2050.00       -10300.00         -8789.82
XOL layer -1800.00        -5200.00          1289.82
QS         -776.25        -5538.75          5036.25
All        -526.25        -2463.75         -2463.75

           Premium spent  Margin spent      CR   M / SD  M / M01 sa  M / M01 div
Gross           1.000000      1.000000  0.9268   0.4508      0.1990       0.2332
XOL layer      -0.192857     -0.878049  0.6667  -0.8265     -0.3462          NaN
QS             -0.605357     -0.378659  0.9542  -0.3409     -0.1401          NaN
All             0.201786     -0.256707  1.0931  -0.6933     -0.2136      -0.2136
```

Two sanity readings worth keeping in mind. The diversified column foots:
`-8789.82 + 1289.82 + 5036.25 = -2463.75`. And on the closing row standalone
and diversified agree exactly, always, because `E[result | result = x] = x`
makes the grand-result cell its own marginal quantile.

### Defect 1: standalone is not a capital measure on a cession

A purchased cover's own low tail is the state in which it **was not needed**.
For a layer remote enough that it attaches beyond the 1-in-100, the quantile
lands inside the no-recovery atom and the cell reports the layer's premium, a
deterministic input the frame already carries elsewhere. For a quota share on a
high-frequency book the cell is not exactly the premium, because the recovery
has little mass at zero, but it is the same wrong tail and a less obvious tell.

The number is exact. It is precisely computed and precisely the wrong quantity,
so "unreliable" would misdescribe it. The question "what capital would this row
call for alone" has no answer for a hedge, and the code asks it of every row
anyway.

Consequence: the standalone column's diversification reading, `sum of the
standalone cells against the diversified total`, is a statement about pooling
risk units. Cessions were never part of that sum and their presence in it is
noise.

### Defect 2: the capital ratio blanks released capital

`_capital_ratio` (`_pnl.py:791`) is

```python
capital = -bad_outcome
return margin / capital if capital > 0 else np.nan
```

The guard conflates two different situations. `capital == 0` genuinely has no
denominator and `NaN` is right. `capital < 0` means capital was **released**,
and the docstring's "no denominator" is false there: the quotient is perfectly
well defined, and it is the most informative cell in the frame.

On the capstone tower the two suppressed cells are

- **XOL layer**, `-1800 / -1289.82 = +139.6%`. 1800 of expected margin given up
  to buy 1289.82 of capital relief.
- **QS**, `-776.25 / -5036.25 = +15.4%`. 776.25 given up to release 5036.25.

Set against the gross row earning **23.3%** on its allocated capital, that is
the reinsurance purchasing decision on one line. The QS buys capital back more
cheaply than the capital earns and is accretive. The XOL buys it at 139.6% and
is not, which is why the book goes from +23.3% on 8789.82 of capital to -21.4%
on 2463.75. Blended program cost `2576.25 / 6326.07 = 40.7%` against 23.3%
earned, and the total impact row already carries both of those numbers.

**Why un-guarding alone is not the fix.** The sign is the same on both kinds of
row but the polarity of "good" flips. On a risk row a high ratio is capital
earning well. On a ceded row a high ratio is relief bought expensively. One
column carrying both would have a reader scanning for large positive numbers
read 139.6% on the XOL as the best line in the program, which is worse than the
blank it replaces. So the column splits.

## The change

### [step-role] teach the step map which rows hedge

`_pnl.py:2080`, `_step_result_rows`. It already holds each row's plan tuple,
whose payload identifies the group, so the role lookup is free here and needs
no new plumbing. `Group.role` is `'sell'` or `'buy'` (`_pnl.py:247`), and
`_ledger_plan` payloads (`_pnl.py:636`) are the group index `gi` for
`group_result`, the span `(lo, hi)` for `tier_result`, and `None` for
`grand_result` (`_pnl.py:701`).

The return shape becomes `{step: (position, label, ceded)}`. There is exactly
one call site, `_pnl.py:2108`, so the widening is contained.

```python
out[idx[0]] = (i, label, self._step_is_ceded(kind, payload))
```

with a new private helper beside it:

```python
def _step_is_ceded(self, kind, payload):
    """Does this walk step hedge rather than bear risk?

    A ``buy`` group is a hedge: it pays in the states the book is short of
    capital, so "the capital it would call for alone" is not a question it
    can answer. The grand result is the net position, which bears risk, and
    a tier span is treated as ceded when it contains any cover, since a
    mixed span's own low tail is no more interpretable than a cession's.
    """
    if kind == 'group_result':
        return self._group_specs[payload].role == 'buy'
    if kind == 'tier_result':
        lo, hi = payload
        return any(g.role == 'buy' for g in self._group_specs[lo:hi])
    return False
```

### [standalone-ceded] blank the standalone quantile on ceded steps

`_pnl.py:2118-2122`, inside `_waterfall_frames`:

```python
pos, label, ceded = steps[step]
...
# A ceded step hedges rather than bears risk, so its own 1-in-100 is the
# state in which it paid nothing and the quantile reports its premium.
# Blank, rather than serve an exact number that is not capital.
standalone = np.nan if ceded or gd is None else float(gd.q(p))
```

The standalone **ratio** follows with no further work: `_capital_ratio` already
returns `NaN` when either input is not finite, so `M / M01 standalone` blanks on
the same rows automatically.

### [relief-ratio] let the capital ratio see released capital

`_pnl.py:791`. The formula is unchanged; only the guard and the docstring move.

```python
def _capital_ratio(margin, bad_outcome):
    """``M / -M01``: margin over the capital that state calls for, or releases.

    ``bad_outcome`` is the margin in the 1-in-100 state. On a risk-bearing row
    it is negative, so ``-bad_outcome`` is the capital you would have to inject
    and the ratio reads as a return on it. On a ceded row it is positive, so
    ``-bad_outcome`` is negative: the cover **releases** capital, and the ratio
    reads as the price paid per unit released. Both are the same arithmetic;
    the caller separates them into two columns, because a high number is good
    on the first reading and bad on the second.

    ``NaN`` where either input is missing, or where the capital is zero to
    within :data:`VALIDATION_NOISE`, which is the one case with genuinely no
    denominator.
    """
    if not (np.isfinite(margin) and np.isfinite(bad_outcome)):
        return np.nan
    capital = -bad_outcome
    if abs(capital) <= VALIDATION_NOISE:
        return np.nan
    return margin / capital
```

`VALIDATION_NOISE` is already imported (`_pnl.py:107`). The tolerance replaces
a bare `> 0` test, so a near-zero capital no longer produces an exploding
ratio in either direction.

### [relief-column] split the diversified ratio in two

`_pnl.py:2127-2143`. One evaluation of the quotient, routed to one of two
columns by the row's role, so every cell in every column has one meaning.

```python
divers_ratio = _capital_ratio(margin, divers)
evaluation.append([
    float(r['P_share']), float(r['M_share']), float(r['CR']),
    margin / sd if sd > 0 else np.nan,
    _capital_ratio(margin, standalone),
    np.nan if ceded else divers_ratio,
    divers_ratio if ceded else np.nan,
])
...
evaluation_df = pd.DataFrame(
    evaluation, index=idx,
    columns=['Premium spent', 'Margin spent', 'CR', 'M / SD',
             f'M / {m} standalone', f'M / {m} diversified',
             'Cost of relief'])
```

The resulting occupancy, which is the whole point of the step:

| column | risk rows | ceded rows |
|---|---|---|
| `M / M01 standalone` | return on standalone capital | blank |
| `M / M01 diversified` | return on allocated capital | blank |
| `Cost of relief` | blank | price per unit of capital released |

**OPEN QUESTION, the only one in this plan.** The new column is named
`Cost of relief` throughout the draft. It is deliberately outside the
`M / ...` idiom, because it is a different quantity in meaning even though it is
the same arithmetic, and reusing the idiom would invite the reader to scan the
three columns as one family. Alternatives if the author prefers the parallel
form: `Relief cost`, `M / M01 relief`, `Cost of M01 relief`. **Author to rule
before implementation**; everything else here is settled.

### [formats] declare the new reading

`src/aggregate/formats/formats-raw.yaml`, beside the two entries added at a358
in the P&L ratios block near `'Premium spent'`:

```yaml
  'Cost of relief': ratio
```

A price per unit of capital reads as a percentage, the same as the two return
columns it sits beside. `formats-insurer.yaml` does not redefine `ratio`, so
one entry covers both perspectives.

### [captions] both footnotes

`src/aggregate/exhibits/_pnl.py:241-262`. Two changes of substance, and both
captions stay one paragraph.

The **walk** caption gains the reason standalone is blank on ceded steps. It
currently ends "The standalone column is each step's own 1-in-100, the capital
it would call for alone", which becomes true as written once the rows where it
is false stop being populated, so the addition is a sentence saying so:

> ... it is blank on a ceded step, whose own 1-in-100 is the state in which the
> cover paid nothing, so the quantile would report its premium rather than any
> capital.

The **evaluation** caption is rewritten around the three ratio columns, naming
the polarity flip explicitly, because that is the one thing a reader cannot
infer from the headings:

> The same walk read as ratios. Premium and margin spent are against the gross
> block. The two `M / M01` columns are the return earned on the capital a
> 1-in-100 outcome calls for, on each of the walk's two readings of that state,
> and they are served only on the steps that bear risk. A cover does not call
> for capital, it releases it, so a ceded step reads instead under cost of
> relief: the margin given up per unit of capital the cover hands back, where a
> **low** number is the good one, and the test is whether it comes in under the
> return the risk-bearing rows earn.

The existing `if not diversified_available:` rider is unchanged.

### [docstrings] the two properties

`_pnl.py`, the `Returns` sections of `walk_df` (around 2160) and
`evaluation_df` (around 2209). `walk_df` records that `M01 standalone` is blank
on ceded steps and why. `evaluation_df` gains the `Cost of relief` entry and
states the polarity flip in its `Notes`. `_capital_ratio`'s own docstring keeps
the exact `M / -M01` statement, which is where a reader chasing the arithmetic
lands.

### [tests] the library's assertions

All in `V:\worktrees\aggregate_REFACTOR\tests\test_exhibits.py` unless noted.

- **`test_waterfall_capital_ratio_definition`** (line 969) currently loops both
  bases and asserts `NaN` wherever `capital <= 0`. That assertion is now wrong
  for the diversified basis on a ceded row. Rework it into two: the standalone
  basis, asserted only on risk rows; and the diversified quotient, asserted to
  land in `M / M01 diversified` on risk rows and in `Cost of relief` on ceded
  rows, with the same `margin / -m100` identity in both.
- **`test_waterfall_diversified_foots_and_standalone_does_not`** (line 953)
  sums `walk['M01 standalone']`. With `NaN` now in that column on ceded steps,
  pandas will skip them and the sum changes meaning. Inspect the `tower`
  fixture's composition at implementation time and rewrite the assertion
  against the risk rows explicitly rather than relying on the skip.
- **`test_waterfall_blanks_diversified_without_shared_atoms`** (line 988)
  asserts `walk['M01 standalone'].notna().any()` on the `peel` fixture. A
  guaranteed-cost peel carries a gross step, so this should still hold, but
  confirm rather than assume.
- **New:** a ceded step has `NaN` in both `M01 standalone` and
  `M / M01 standalone`, and a finite `Cost of relief`; a risk step is the
  mirror image. One test over the `tower` fixture.
- **New:** the closing row's standalone equals its diversified exactly, the
  `E[result | result = x] = x` identity, which is a cheap guard against the
  anchor silently moving.
- **`PENDING_VOCABULARY`** (line 251) needs no new entry, since
  `Cost of relief` is declared in the format sheet by [formats]. The gate will
  say so if [formats] is missed.
- **`tests/data/exhibit_snapshots.json`** pins captions and column labels for
  every case. **Do not hand-edit.** Regenerate with
  `uv run python tests/capture_exhibit_snapshots.py` and read the diff: it
  should carry the two captions, the new column, and blanked cells on ceded
  rows, and nothing else.

### [features] the feature register

`dev/FEATURES.csv` rows 24 and 25 describe `walk_df` and `evaluation_df`
including every column. **Note for the implementer, learned at a358:**
`dev/regen_features.py` is an **auditor over the live member surface**, not a
generator of the hand-written `notes` prose. It reports `OK` and writes nothing
when no member name has moved, which is the case here. Edit the two `notes`
cells by hand and move `# table-version` to `1.0.0a361`. Run the auditor anyway
to confirm it stays green.

## Acceptance checks

1. `uv run --no-sync pytest tests/test_exhibits.py tests/test_exhibit_formats.py`
   green.
2. `uv run --no-sync pytest` green, the full fast suite, once at the commit
   boundary.
3. The snapshot diff carries only the two captions, the new `Cost of relief`
   column, and blanked standalone cells on ceded rows.
4. The capstone program above serves exactly:

```
            Margin  M01 standalone  M01 diversified
Gross      2050.00       -10300.00         -8789.82
XOL layer -1800.00             NaN          1289.82
QS         -776.25             NaN          5036.25
All        -526.25        -2463.75         -2463.75

           M / M01 standalone  M / M01 diversified  Cost of relief
Gross                  0.1990               0.2332             NaN
XOL layer                 NaN                  NaN          1.3956
QS                        NaN                  NaN          0.1541
All                   -0.2136              -0.2136             NaN
```

   Read as percentages under either perspective: XOL relief costs 139.6%, QS
   relief costs 15.4%, gross earns 23.3%.
5. `M01 diversified` still foots: `-8789.82 + 1289.82 + 5036.25 = -2463.75`.
   This plan must not move it.

## Bookkeeping

**`aggregate`, `1.0.0a360` to `1.0.0a361`.** One commit carrying the source
change, the format sheet, the tests, the regenerated snapshot,
`dev/FEATURES.csv`, `pyproject.toml`, the `CHANGELOG.md` section, and this plan
moved to `dev/done/`. One line, no body, no trailers, house format:

```
[Waterfall-Ceded-Steps] a361: blank standalone on cessions, add cost of relief
```

The CHANGELOG entry needs a paragraph for each defect, and must state plainly
that `evaluation_df` gains a column and that cells which previously carried
numbers on ceded rows are now blank, since both are breaking for a caller
indexing the frame.

**No `aggregate_api` change.** The SPA renders exhibits generically off the
`/v1` exhibit route, so the new column and the blanks arrive without a bump
there. The lede warning discussed on 2026-09-29 is **dropped as superseded**:
it existed to flag a number this plan removes.

Not pushed. The author pushes.

## Execution log, 2026-09-29

Ruled and executed 2026-09-29: the author accepted `Cost of relief` as the
column name, closing the plan's one open question. Implemented as specified,
with three divergences, all small.

- **[acceptance-rounding]** The acceptance table's XOL `Cost of relief` cell
  reads `1.3956`; the computed value is `1800 / 1289.82 = 1.39554`, which
  prints `1.3955` at four decimals. A rounding slip in the plan, not in the
  code; the 139.6% reading at one decimal is as stated.
- **[test-import]** The new mirror-image test imports numpy locally inside the
  test function, matching the file's existing style (`test_exhibits.py` keeps
  numpy out of its module imports).
- **[no-todo-item]** `dev/TODO.md` carries no tracked item for this plan, so
  there is nothing to tick there.

Snapshot regenerated; the diff carries exactly the two captions, the new
`Cost of relief` column, and blanked cells on ceded rows (Tower and Peel
waterfall cases only). `dev/regen_features.py` green after the hand edit of
the two notes cells. `uv.lock` was already modified before the session (a
project-version re-lock) and is left uncommitted for the author, matching the
precedent that bump commits do not carry it.
