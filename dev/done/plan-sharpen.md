# Plan: [Sharpen-Grid-Probe]

Target version: **1.0.0a192**. Home: `src/aggregate/_bucket_window.py` (author
DECIDED: it joins `bs_describe` / `bs_explain` there; the open
**[bs_describe-Wart]** TODO item is *not* settled by this plan).

## 1. What ships

A `sharpen` probe that answers "is this grid actually well chosen?" by
re-updating the object on the eight neighbouring `(bs, log2)` cells, scoring each
against the analytic moments, and moving to a better cell when the win is large.

```python
a = build('agg X 100 claims sev lognorm 100 cv 2 poisson')
a.sharpen()          # probes, moves only on a big win, returns a
a.sharpen_df         # tidy 9-row frame, one row per cell
a.sharpen_description  # one line: what it found, what it did
a.sharpen_explanation  # the long form
a.update(log2=16, bs=1, sharpen=True)   # opt-in auto-sharpen after an update
```

Scope: `Aggregate` and `Portfolio`. **Not** `BivariateAggregate` (author
DECIDED: quadratic cost, no sharpen). **No** `PnL.sharpen` method: a `PnL`
snapshots its source atoms at construction (`_pnl.py:898`), so sharpening a
*built* `PnL` would leave its ledger on a stale grid. It does not need one --
`build_many` updates the deferred engine as an `Aggregate` (`underwriter.py:1832`)
and only then snapshots the `PnL` (line 1880), so `build(prog, sharpen=True)`
reaches a `pnl` for free, pre-construction.

## 2. The score

Read `stats_df['error']`, the canonical noise-aware relative error already
computed at the end of every update (`moments.py:548`). Do not recompute
moments.

Six terms, each divided by **its own** validation tolerance, so the number is in
tolerance units:

| term | tolerance |
|---|---|
| `('sev', 'mean')`, `('agg', 'mean')` | `eps` |
| `('sev', 'cv')`, `('agg', 'cv')` | `10 * eps` |
| `('sev', 'skew')`, `('agg', 'skew')` | `100 * eps` |

`eps = ob.validation_eps`. Those multipliers are exactly the ones
`valid_aggregate` / `valid_portfolio` apply, so **`score <= 1` means "passes
validation at this eps"** and `score = 1` is the pass boundary. The score does
not move when `validation_eps` is changed.

```
u_i   = |error_i| / tol_i
score = ( mean_i u_i**power ) ** (1/power)        power finite
score = max_i u_i                                 power = inf
```

The `mean` (not `sum`) keeps cells comparable when terms drop out. A term is
**live** iff its *theoretical* value is finite and `abs(theo) > VALIDATION_NOISE`
-- the same test `valid_*` uses, and the theoretical does not depend on `bs`, so
the live set is identical across all nine cells. A non-finite *empirical* scores
`inf` (a genuine failure, not a dropped term). Theoretical column is `mixed` for
`Aggregate`, `total` for `Portfolio`; detected by presence, no class import.

Under reinsurance `error` is the SUBJECT (gross) comparison, which is the right
thing: `sharpen` audits the grid, and the gross grid is what the grid choice
controls.

Recorded but **not** scored: the aliasing ratio `agg_err_mean / sev_err_mean`.
It is the specific fingerprint of `bs` too small and no sum of errors captures a
ratio; it is a column and a tiebreak, not a term. Deficit is dropped entirely
(author: too hard to interpret once normalized).

## 3. The probe

Cells are `bs * 2**i` for `i` in `(-1, 0, 1)` and `log2 + j` for `j` in
`(-1, 0, 1)`. `bs` is always positive so halving and doubling are unconditionally
safe. Strict x2 / /2 rather than `round_bucket` ladder rungs, because the
geometry is the point: with extent `W = bs * 2**(log2+i+j)`, rows are constant
resolution and **anti-diagonals are constant extent**, so the three same-extent
cells isolate the pure resolution effect. Ladder rungs (1.25x, 1.6x from `bs=5`)
are too fine to move the error and break that alignment.

Guards: drop the `log2+1` column when `log2 >= log2_cap` (default 24); drop the
`log2-1` column when `log2-1 < LOG2_FLOOR` (8).

**The centre cell is free** -- it is the object's current state, already
computed, so its score is read in place. Eight probe updates, then one final
update (the winner, or the centre to restore). At most 9 updates, never 10.

## 4. Selection: "helps a lot", not "helps a little"

Two gates off one number, `good_enough` (default **0.5**):

1. **Probe gate.** If the centre score is `<= good_enough`, return immediately.
   Nothing runs, `_sharpen_df` holds the single centre row.
   `good_enough=0` therefore means "always probe" (a score is never negative),
   which is why there is no separate `force` kwarg.
2. **Move gate.** Among cells scoring `<= good_enough`, sort by `(log2, score)`
   and take the first: **the smallest `log2` that meets the target**, never
   grow the grid unless it is the only way. If no cell meets the target, take
   the best available step provided it beats the centre by `min_gain`.

**Deviation found in execution.** `min_gain` shipped at **2.0**, not the 10.0
this plan first specified, and the fallback branch gained a parsimony tiebreak
(`SHARPEN_FALLBACK_SLACK = 1.25`: among cells within 25% of the best score,
prefer the smallest `log2`).

Testing on a deliberately starved grid, `bs = 1/8` at `log2 = 14`, showed the
10x bar was wrong. The best neighbour scored 908 against the centre's 3494, a
3.8x win on an object that *fails validation outright*, and the 10x bar rejected
it. Worse, it rejected it permanently: re-running `sharpen` re-centres on the
same cell and refuses identically, so the descent could never start. The probe
gate already guarantees this branch is only reached by a grid that has failed
the target, so the severe bar was protecting against nothing and blocking the
one case the probe exists for. At 2.0 the same object walks
`(1/8, 14) -> (1/4, 15) -> (1/2, 16) -> (1, 17)` over three calls, reaching
`not unreasonable`, and converges on the fourth. The slack tiebreak keeps the
original protection: a marginally better score still never buys a doubled grid.

In a 3x3 every non-centre cell is on the probe edge, so a "boundary" flag would
just restate "it moved". Instead, when it moves, the description says so and
notes that re-running `sharpen` re-centres and continues the descent.

## 5. Locking the update settings

`update_work` stamps `sev_calc`, `discretization_calc`, `normalize`, `padding`
**from its arguments, whose defaults are `'discrete'` / `'survival'` / `True` /
`1`**. A bare `ob.update(log2=L0, bs=bs0)` therefore restores the grid but
silently *resets* those four. So `sharpen` harvests the current state once, up
front, and passes it to every probe cell and to the final update. That is both
the fidelity fix and the apples-to-apples fix: all nine cells differ in `bs` and
`log2` and in nothing else.

Locked for `Aggregate`: `sev_calc`, `discretization_calc`, `normalize`,
`padding`, `reins_bucket`, `dsev_bucket`.
Locked for `Portfolio`: `sev_calc`, `discretization_calc`, `normalize`,
`padding`, `remove_fuzz` (stamped as `_remove_fuzz`, `_portfolio.py:2096`).

Not stamped anywhere, so handled by rule rather than harvest:

- `force_severity` -- probe with `False` (cheap), finish with `True` (safe
  superset, matches what `build` passes).
- `add_exa` (Portfolio) -- probe with **`False`**. It is the dominant cost of a
  Portfolio update and contributes nothing to the moments; the final update
  runs `True`.
- `bucket_sizing_p` -- irrelevant, every cell passes an explicit `bs`.
- `x_min` -- the final restore passes `x_min=self.xs[0]` so a signed grid comes
  back on its original origin. Probe cells let `_bs_window` re-pick, which is
  genuinely what you would get if you asked for that cell; the result is a
  column.

## 6. The frame

Tidy/long, one row per cell, on `_sharpen_df`, exposed as the `sharpen_df`
property (returns a copy). Columns:

`d_bs, d_log2, bs, log2, extent, x_min, score, u_sev_mean, u_sev_cv,
u_sev_skew, u_agg_mean, u_agg_cv, u_agg_skew, aliasing, validation, warnings,
seconds, selected, note`

`d_bs` / `d_log2` are the integer offsets, so the 3x3 picture is one `pivot`.
`selected` marks the winner (mirrors `bs_window_df`). `warnings` is a count, not
the kinds (author: there are plenty already). Every cell runs inside its own
`try`, recording `NaN` plus the exception text in `note` rather than aborting
the sweep.

## 7. Narrative

`sharpen_description` (short, one line) and `sharpen_explanation` (long), the
house pair, mirroring `bs_description` / `bs_explanation`. Both are new members
of the description/explanation pairs rule, and both halves ship together so
`FCC_UNPAIRED_NARRATIVES` stays empty.

The explanation carries: what the score units mean, the extent-vs-resolution
reading of the anti-diagonals, **why it refused to grow `log2`**, what moved, the
re-run hint, and the Portfolio unit caveat below.

## 8. Portfolio caveat, stated not hidden

The Portfolio score reads `port.stats_df['error']` for the **total only** (author
DECIDED: same logic as `Aggregate`, easy start). `valid_portfolio` short-circuits
on the first bad unit, so a total-only score can call a portfolio clean while one
unit is poorly resolved. Units are not sharpened individually and then combined:
they would each land on a different `bs` and the combine has to reconcile them
anyway (`best_window` already does the SBJ look-through). The limitation is
stated in `sharpen_explanation`, not left silent.

## 9. `update(..., sharpen=False)`

New bool kwarg on `Aggregate.update` and `Portfolio.update`, default `False`,
applied after the update work completes. `build_many` does **not** turn it on
(author: potentially too much work); a user can still opt in per build because
`kwargs` pass through, so the `BivariateAggregate` branch must
`kwargs.pop('sharpen', None)` (its `update` does not take it).

Recursion guard: the probe updates pass `sharpen=False` explicitly.

## 10. `Aggregate.focus` -> `center_window`

`focus(p)` is the no-recompute central-window re-slicer (`_aggregate.py:5490`).
The name is wrong for what it does and it blocks the good name. Rename to
`center_window`; the only call sites are two lines in
`tests/test_balanced_window.py` plus one `:meth:` cross-reference in
`utilities.py:404`. Still alpha, so a straight rename, no deprecation shim.

## 11. Work items

1. **[Center-Window-Rename]** `focus` -> `center_window`; fix the two test call
   sites and the `utilities.py` cross-reference.
2. **[Sharpen-Worker]** `_bucket_window.py`: `_SHARPEN_TERMS`, `sharpen_score`,
   `sharpen`, `sharpen_describe`, `sharpen_explain`. One implementation, duck
   dispatch on `hasattr(ob, 'agg_list')`, no import of `_aggregate` /
   `_portfolio` (the module's leaf property is preserved).
3. **[Sharpen-Surface]** `Aggregate` / `Portfolio`: `_sharpen_df = None` in
   `__init__`, `sharpen()` method, `sharpen_df` / `sharpen_description` /
   `sharpen_explanation` properties.
4. **[Sharpen-Update-Hook]** `sharpen=False` kwarg on both `update`s; the
   `kwargs.pop` in the `BivariateAggregate` branch of `build_many`.
5. **[Sharpen-Tests]** `tests/test_sharpen.py`.
6. **[Sharpen-Release]** version bump to 1.0.0a192, `CHANGELOG.md`,
   `dev/TODO.md`, `dev/FEATURES.csv` via `dev/regen_features.py`, move this plan
   to `dev/done/`. One commit.

## 12. Tests

- score of a well-sized agg is small; `sharpen()` returns without probing
  (`sharpen_df` has exactly 1 row, `selected` on the centre).
- `good_enough=0` forces the probe on that same object (9 rows).
- a deliberately starved grid (explicit tiny `bs`) scores badly, and `sharpen()`
  moves: `bs`/`log2` change, the new score is better, `selected` is not the
  centre.
- `execute=False` leaves `bs`, `log2` **and** the locked settings unchanged
  (updated with a non-default `sev_calc` / `padding` first, so a reset would
  show).
- `log2_cap` equal to the current `log2` drops the `+1` column (6 rows).
- log2 parsimony: when two cells both clear the target, the smaller `log2` wins.
- a cell that raises is recorded as `NaN` + `note`, and the sweep completes.
- `Portfolio.sharpen()` runs end to end.
- `update(..., sharpen=True)` reaches the same state as `update(...)` then
  `sharpen()`.
- the description / explanation pair is non-empty in all three outcomes (not
  run, moved, no easy win).

---

## Follow-up punch-ups, shipped 1.0.0a193

Author review of the shipped `a192` probe raised five, all executed:

1. **`validation_score` as a property.** The score is a handy statistic on its
   own, so it is exposed on `Aggregate` and `Portfolio` at `power=2`. That made
   `_bucket_window.sharpen_score` a second name for one concept, so the score
   **moved to `_validation.py`** with the rest of the validation family:
   `SCORE_TERMS`, `validation_score_terms(obj)` (power-free per-term detail),
   `combine_score_terms(terms, power)` and `validation_score(obj, power=2)`.
   `_bucket_window` imports them.
2. **A wider bucket reach, and then a better idea.** The author first asked for a
   fixed `bs/4 .. 4*bs` row, then proposed expanding outward until the score
   worsens. The second is strictly better and subsumes the first. Shipped as a
   **line search per `log2` row**: double until the score stops improving, halve
   likewise, capped by `bs_limit` (default 16, must be a power of two).

   Chosen over the two cheaper variants (search at the current `log2` only, then
   check the neighbours; or search once and probe only near the winner) because
   keeping all three rows complete preserves the constant-extent anti-diagonal
   reading that makes the frame interpretable, and costs nothing on a grid near
   its optimum: the search stops at the first cell that fails to improve, so the
   common case is still eight evaluations. Worst case is 26.

   Stopping at the first worse cell assumes the score is single-troughed in `bs`
   at fixed `log2`, which holds because a larger bucket trades resolution for
   extent. The exception is a discrete severity whose atoms land on grid points
   at some buckets and not others, which could dip again past the turn;
   `bs_window` sizes those by its exact-discrete method, so they rarely reach a
   probe. Documented in the `sharpen` Notes rather than guarded against, since a
   patience parameter would be a seventh knob for a case the estimator already
   handles.
3. **`(d_bs, d_log2)` is the index**, not two columns, so
   `sharpen_df.score.unstack('d_log2')` is the picture. The line search makes the
   rows ragged; unvisited cells come back `NaN`.
4. **Sub-unit `bs` renders as a binary fraction** (`1/8`, not `0.125`) in both
   narrative halves, via `_fmt_bs`. Anything that is not a unit fraction falls
   back to plain formatting.
5. **`sharpen_description` opens capitalized.** Found alongside it: a
   bucket-only move reported `log2 16 to 16`, which reads as a bug rather than
   as "unchanged", so `_move_phrase` now names only what actually changed.

