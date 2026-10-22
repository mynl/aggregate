# Plan: κ columns condition on the whole level set, not on exact float equality

`[Kappa-Level-Sets]`, one step, one version bump (a fix, so a patch). Drafted 2026-10-22 at 1.7.1
(`3f342d3`). **Status: reviewed and executed 2026-10-22 as 1.10.1, on Steve's `/execute-plan`.**
See the Execution log at the foot.

## Goal

Every `κq` cell of `PnL.economic_df`, and every number derived from those cells (the waterfall's
diversified capital columns), is the conditional mean of its row over **all** the atoms where the
conditioning row equals its `q` quantile. Today it is the conditional mean over the subset of those
atoms whose floating-point value happens to match the quantile bit for bit, and that subset can be
a small and arbitrary fraction of the level set.

## Background: what κ is

`PnL.economic_df` reports, for each ladder point `q` in `PERCENTILE_LADDER`, a scenario column
`κq`. Each cell is `E[row | result == x_q]`, where `result` is the ledger's grand result (the net
margin) and `x_q` its `q` quantile ([Kappa-Scenario-Percentiles]). The columns foot by linearity of
conditional expectation. `PnL._gross_scenario_ladder` is the twin anchored on the first group's
result, the gross margin ([Waterfall-Gross-Basis]); `waterfall_df` reads `κ01` from both to build
`M01 div net`, `M01 div gross`, `Div CoC net` and `Div CoC gross`.

On the per-atom route (`self._probs` is set), every ledger row is an array of signed values, one
per source atom (1-D grid cell, or 2-D joint cell for an occurrence program), computed by evaluating
the leg functions and summing them in group order.

## Current behavior

`src/aggregate/_pnl.py`, `PnL._scenario_ladder` (line 1972) and `PnL._gross_scenario_ladder`
(line 2004) both form the conditioning event as

```python
mask = res_vals == float(gd.q(q))
```

where `gd` is the row's exact irregular `GridDistribution`, built in `_EvaluatedLeg.gd` (line 491)
by `pd.Series(probs, index=values).groupby(level=0).sum()`, which groups atoms by exact value.

Both steps assume that atoms with the same nominal result carry bit-identical floats. They do not
whenever the result is reached by different arithmetic paths. On a 2-D occurrence joint the net
margin `2000 - P + r - l` is computed from many `(l, r)` pairs with the same `l - r`, and the
rounding differs from pair to pair. A nonlinear leg (a corridor's `phi(r / P) * P`, a retro collar)
adds further noise. The level set `{result = x_q}` then splits into several floats a few ulps apart.
The exhibit groups them into separate atoms, `gd.q(q)` returns one of them, and the mask catches only
the atoms carrying that exact float.

Footing does not detect the defect: linearity holds on any subset of atoms, so a column computed on
the wrong subset still adds down the page. That is why no existing test fails. The κ tests in
`tests/` check column names, ordering and footing, never the slice's mass.

The same exact-equality assumption sits in `PnL.prob_eq_0` (line 1581, `self._probs[v == 0]`), so a
break-even atom split by noise is undercounted.

## Evidence

Measured 2026-10-22 on 1.7.1. For each program, at the worst ladder point: the probability mass the
exact slice catches as a fraction of the level set `|result - x_q| <= 1e-9 * S` (S the ledger's
largest absolute row value), the number of distinct floats the level set splits into, and the
largest absolute change in any κ cell across the ladder.

| Program | Atoms | Exact slice / level set | Split | Max κ error |
|---|---:|---:|---:|---:|
| `Bear1990.BN.TreatyII` (free reinstatements) | 1,048,576 | 1.000 | 1 | 0 |
| `Bear1990.BN.TreatyIIp` (paid reinstatements) | 1,048,576 | 1.000 | 1 | 0 |
| `Bear1990.BN.TreatyIII` (corridor, 1-D) | 65,536 | 1.000 | 1 | 0 |
| `Bear1990.BN.TreatyIV` (retro `pnl`) | 65,536 | 0.381 | 2 | 289 |
| `Bear1990.BN.TreatyV`, `TreatyVI` | 65,536 | 1.000 | 1 | 0 |
| GC occurrence walk, `occurrence net of 750 xs 250 deposit 361.83` | 131,072 | 0.163 | 2 | 698 |
| Same, plus `aggregate net of 100000 xs 0 deposit 1000 corridor 100% po 5% xs 0%` | 131,072 | 0.019 | 16 | 1,309 |

The occurrence programs share the gross book
`2000 premium at 70% lr 1000 xs 0 sev 120 * lomax 2.5 mixed gamma .185` under an `xpnl` with
`2000 premium`. The Bear1990 programs are the entries of `src/aggregate/agg/reproductions.agg`.
A hand-built `PnL` over the same occurrence joint with an annual aggregate deductible written as a
corridor on the recovery leg (`D:\projects\hacks\individual-pieces\occurrence-layer-aggregate-deductible.qmd`,
section 5) gave a `κ01` layer recovery of 334 under numpy 2.4.5 and 544 under numpy 2.5.1, against
577 on the full level set: the exact slice caught 21 of 1,024 cells in one environment and 434 in
the other. So the defect also makes κ depend on the numpy build.

The measurement script, for the acceptance check below:

```python
import numpy as np
from aggregate._pnl import PERCENTILE_LADDER

def kappa_shortfall(pnl, rtol=1e-9):
    """Worst exact-slice mass fraction and worst κ error over the ladder."""
    v, pr, gd = pnl._grand_result.values, pnl._probs, pnl._grand_result.gd
    tol = rtol * max(float(np.abs(r.values).max()) for r in pnl._rows.values())
    frac, err = 1.0, 0.0
    for q in PERCENTILE_LADDER:
        xq = float(gd.q(q))
        ex, nr = v == xq, np.abs(v - xq) <= tol
        frac = min(frac, pr[ex].sum() / pr[nr].sum())
        for row in pnl._rows.values():
            a = (row.values[ex] * pr[ex]).sum() / pr[ex].sum()
            b = (row.values[nr] * pr[nr]).sum() / pr[nr].sum()
            err = max(err, abs(a - b))
    return frac, err
```

## The change

One idea, applied in four places: **atoms whose values agree to within a ledger-wide tolerance are
one outcome.** Group them once, and slice and count by group, never by float equality.

1. **The tolerance.** Add `level_set_rtol: float = 1e-9` to `ValidationSettings` in
   `src/aggregate/config.py`, with a docstring entry and an environment override in the existing
   mapping (`AGGREGATE_VALIDATION_LEVEL_SET_RTOL`). The ledger's absolute tolerance is
   `tol = level_set_rtol * S`, with `S` the largest absolute value of any evaluated row over the
   atoms (at least 1), computed once per `PnL` evaluation and stored as `self._level_tol`. Rounding
   noise from a handful of float operations is of order `1e-13 * S`; genuinely distinct outcomes on
   any grid the library builds differ by far more than `1e-9 * S`. Values closer than that are below
   anything the exhibits display, so merging them changes nothing a reader can see except the κ
   cells, which become correct.

2. **The grouping helper.** Add `_atom_groups(values, tol)` to `_pnl.py`: sort the values (stable),
   start a new group wherever the gap to the previous sorted value exceeds `tol`, and return the
   group index of every atom. Gap clustering, not rounding to a grid of width `tol`, because
   rounding splits any cluster that happens to straddle a grid boundary. A group's value is the
   probability-weighted mean of its atoms' values, or its smallest value when its mass is zero (a
   joint has zero-mass cells, which the prototype tripped over). Measured cost: 3 ms at 131,072
   atoms, 19 ms at 1,048,576.

3. **The exact GD.** `_EvaluatedLeg.gd`, `bs == 0` branch: build the irregular `GridDistribution`
   from the groups (group values, summed group probabilities) instead of the exact-value `groupby`.
   `_EvaluatedLeg` needs the tolerance, so `PnL` passes `self._level_tol` when it builds the rows.
   Each row's group index is cached on the row (`_EvaluatedLeg._group_index`) because the ladder
   needs it for the conditioning rows.

4. **The ladders and the break-even probability.** Fold `_scenario_ladder` and
   `_gross_scenario_ladder`, which are identical apart from the anchor row, into one private
   `_kappa_ladder(anchor)`. For each `q`, find the GD atom `x_q = gd.q(q)`, take its group `g`
   (`np.searchsorted(gd.x, x_q)`, exact because `x_q` is an element of `gd.x`), and slice
   `mask = anchor._group_index == g`. `PnL.prob_eq_0` sums `probs` over `|v| <= tol` on the
   per-atom route. The sweep-backed (massive) branch is a regular grid and stays as it is.

Docstrings: `_scenario_ladder`'s replacement, `economic_df` and `prob_eq_0` state the tolerance and
why it exists (a `Notes` paragraph on rounding noise across arithmetic paths), and the `economic_df`
text "the level set `{result == x_q}`" becomes "the level set, up to `level_set_rtol`".

Out of scope, checked and unaffected: the stitched Palm ladder (`_palm_scenario_ladder` in
`_pnl_builders.py`), which conditions on bucket indices; the massive one-sweep route; the
`bs > 0` legs, which rebucket onto a regular grid; `_moments_of`'s constant-leg test, since a
constant leg evaluates through the same arithmetic on every atom.

## Files

- `src/aggregate/config.py`: `ValidationSettings.level_set_rtol`, docstring, environment mapping.
- `src/aggregate/_pnl.py`: `_atom_groups`, `_EvaluatedLeg.gd` and `_group_index`, the tolerance
  on `PnL`, `_kappa_ladder` replacing the two ladder methods (update the two call sites, lines 2245
  and 2685), `prob_eq_0`, docstrings.
- `tests/test_pnl_kappa_level_sets.py` (new), see below.
- `CHANGELOG.md`, `src/aggregate/__init__.py` (`__version__`), `dev/TODO.md`.
- Docs: `docs/3_reference/3_x_PnL.rst` and the three overview pages that show κ
  (`features.rst`, `pipeline-exhibits-and-charts.rst`, `pipeline-pnl.rst`) for any wording on
  exact equality. Printed κ values in executed examples may move; the docs are pending a rebuild
  (not built in the verification cycle).

## Tests

New file `tests/test_pnl_kappa_level_sets.py`, fast tier:

- **A forced split.** A 1-D `PnL` over a hand-built `(values, probs)` source with two legs whose
  sum reaches the same nominal result through different float paths (for example atoms giving
  `0.1 + 0.2` and `0.3`). The κ cell at that quantile equals the conditional mean over both atoms,
  and the GD has one atom there, not two.
- **The occurrence walk.** The GC occurrence walk from the evidence table, on a small grid. For
  every ladder point, the slice mass equals the level-set mass and every κ cell equals the
  brute-force tolerant conditional mean (the `kappa_shortfall` function above returns `(1.0, ~0)`).
- **Separation.** No two atoms of any row's exact GD are within `tol` of each other, and the GD's
  total mass is 1.
- **Break-even.** `prob_eq_0` on a ledger whose break-even atoms are split by noise returns the
  level-set mass.
- **Footing.** The existing `_assert_column_foots` pattern still holds on the walk.

## Acceptance

- `kappa_shortfall` returns a fraction of 1.0 and an error below `1e-9 * S` for every program in
  the evidence table.
- The hand-built AAD ledger in the qmd above gives the same `κ01` (577 to the cent) under the
  `.venv` (numpy 2.4.5) and `aggregate-reproductions` (numpy 2.5.1) environments.
- Tier 2 (`uv run pytest`), tier 3 (`uv run pytest -m 'slow or not slow'`) and the numerics gate
  (`-W error::RuntimeWarning`) are green. Any test whose pinned κ or waterfall value moves is a
  finding to read, not to re-pin blindly: the expected movers are ledgers whose result is reached by
  more than one arithmetic path (occurrence walks, retro, corridors and other nonlinear legs).
- `economic_df` on `Bear1990.BN.TreatyII` (1,048,576 atoms) is no more than 10% slower than at
  1.7.1.

## CHANGELOG entry (one paragraph, at the bump)

Lede `**[Kappa-Level-Sets] κ columns condition on the whole level set**`, then the facts: the κ
cells of `economic_df` and the waterfall's diversified columns move on ledgers whose result is
computed through rounding-sensitive arithmetic (occurrence walks, retro, corridors), previously
off by up to several hundred on the measured examples; new setting `validation.level_set_rtol`
(default `1e-9`); `prob_eq_0` counts break-even atoms split by rounding.

## Execution log

**1.10.1, 2026-10-22**, on `dev` after 1.10.0 (`67eb0cb`). The plan carried no open question; its status read drafted, and Steve's `/execute-plan` invocation stood as the review. Every cited file, symbol and line number (1972, 2004, 1581, 491, 2245, 2685) still matched at 1.10.0: `_pnl.py` and `_grid_distribution.py` were untouched since 1.7.1.

Divergences and findings:

1. **Grouping by distinct values, not by sorting the atoms.** Measured on TreatyII (1,048,576 atoms, 1,374 distinct result values): the exact `groupby` took 13.2 ms of a 24 ms cold `economic_df`, and a stable argsort with gap clustering 17.3 ms, which alone would have broken the 10% budget. `_atom_groups(values, probs, tol)` instead hashes the atoms to their distinct values (`pd.factorize`, one pass, as the `groupby` did), gap-clusters the sorted distinct values, and forms each group's weighted mean from distinct value times its mass, so no second pass over the atoms. Same groups as the plan's algorithm. Cold `economic_df` on TreatyII went from 24 ms to 22 ms.
2. **Group bounds, not a per-atom group index.** The plan cached `_EvaluatedLeg._group_index`. The row instead keeps `_group_bounds = (lo, hi)` per GD atom, and `_EvaluatedLeg.level_set(x)` slices by `values == lo` when the group holds one distinct value (the common case, today's cost) or by `lo <= values <= hi` when the tolerance merged several. Groups are maximal runs more than `tol` apart, so the interval holds exactly the group's atoms.
3. **A lone distinct value keeps its float exactly**, and a merged group's weighted mean is clipped to `[lo, hi]`. Without that, `v * m / m` can move a value by an ulp, which at zero tolerance could break the strict ordering of the GD atoms. With `level_set_rtol = 0` the GD is the old exact one.
4. **The tolerance is read at evaluation** (`get_settings()` in `_assemble_rows`), not frozen in a module constant, so a reload or the environment variable takes effect. Rows built outside `_assemble_rows` (`_zero_row`, the stitched and sweep routes) keep `level_tol = 0`, exact grouping. The scan for `S` costs 2.0 ms on TreatyII's 0.95 s build.
5. **Acceptance item 2 ran on a different ledger.** The qmd's hand-built AAD ledger (a corridor on the recovery leg) was replaced at 1.9.0 by the native `aad` clause (`occurrence net of 750 xs 250 aad 50 deposit 326.39`). On that walk the exact slice caught 0.934 of the level set at 1.10.0, with a κ error of 74, and `κ01` layer recovery read 525.48 under both numpy 2.4.5 (`.venv`) and numpy 2.5.1 (the reproductions book's venv, editable on this worktree). After the fix it reads 599.55 under both, with zero error against the brute-force level set. The plan's 577 belonged to the retired hand-built ledger and grid and is not reproduced.
6. **TreatyIIp** lives in the reproductions book's `reproductions.agg` (`D:/projects/aggregate/reproductions`), not in the shipped one; the evidence script read it from there.
7. **Tests beyond the plan's list**: `_atom_groups` keeps a lone value exactly and gives a massless group its smallest value; the walk fixture is shown to split its level sets; the gross-anchored ladder is checked against brute force too. The brute force uses a fixed `1e-9`, not the setting, and with `AGGREGATE_VALIDATION_LEVEL_SET_RTOL=0` the split, level-set and break-even tests fail, so they detect the defect. Footing reuses `tests.test_pnl_peel._assert_column_foots`.
8. **No pinned value moved.** Neither the exhibit snapshots (Tower and Peel walks included) nor any other test changed.
9. **Docs**: two sentences in `docs/2_aggregate_overview/pipeline-pnl.rst` (the `bs=0` leg, and "tested with float equality and no tolerance"). `3_x_PnL.rst`, `features.rst` and `pipeline-exhibits-and-charts.rst` hold no exact-equality wording. The config template gains a commented `level_set_rtol` entry. Not built.

Measured after the fix, `kappa_check.py` (the plan's measurement, comparing each `economic_df` κ cell with the brute-force tolerant mean):

| Program | Atoms | Max κ error before | After | Worst merged group span |
|---|---:|---:|---:|---:|
| `Bear1990.BN.TreatyII` | 1,048,576 | 0 | 0 | 0 |
| `Bear1990.BN.TreatyIIp` | 1,048,576 | 0 | 0 | 1.8e-12 |
| `Bear1990.BN.TreatyIII` | 65,536 | 1.8e-17 | 1.8e-17 | 1.8e-12 |
| `Bear1990.BN.TreatyIV` | 65,536 | 289.4 | 0 | 3.0e-08 |
| `Bear1990.BN.TreatyV`, `TreatyVI` | 65,536 | 0 | 0 | 0 |
| GC occurrence walk | 131,072 | 698.2 | 0 | 5.7e-14 |
| Same plus the corridor | 131,072 | 1,427.8 | 0 | 3.6e-12 |
| Same walk with `aad 50` | 131,072 | 74.1 | 0 | 9.1e-13 |

The widest merged group spans under 1% of the tolerance (TreatyIV, tolerance 4.1e-06), and the smallest gap between distinct GD atoms on any of these ledgers is 0.02, so the default sits orders of magnitude from both and no chaining occurs.

Tests: tier 1 on the P&L, config and exhibit files, 737 passed. Tier 3 `UV_PROJECT_ENVIRONMENT=.venv uv run pytest -m 'slow or not slow'`, 6496 passed. Numerics gate, the same with `-W error::RuntimeWarning`, 6496 passed.
