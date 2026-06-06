# Sub-project E — `stats_df` consolidation on Portfolio

**Part of the Portfolio refactor.** Master plan: `portfolio-refactor-planning.md`. Prior: Sub-projects A, B, C, D must be landed.

## Goal

Mirror Aggregate's Stage 1c+ work on Portfolio: introduce `stats_df` as the single source of stats truth; delete the overlapping report/statistics/audit_df family. Public stats surface on Portfolio becomes exactly three things — `info`, `describe`, `stats_df` — same shape as Aggregate.

## Pre-conditions

- Sub-projects A, B, C, D landed and committed. Hash: f3a6236702c99be66854e2d266aa7acbd63be80d
- `Aggregate` already has `stats_df` (Stage 1c+; done in a prior session).
- 415/415 pytest green.

## Touches

- `aggregate/portfolio.py` (heavy)
- `aggregate/extensions/portfolio_pir.py` (light: any moved exhibit code that reads the old frames needs flipping to read `stats_df`)
- `tests/`
- `docs/` (some `.rst` files reference `port.statistics`, `port.report_df` patterns and need migration — analogous to the Aggregate doc-sweep we did in Stage 1c+)
- `README.rst`, `pyproject.toml`

## Background — what Aggregate did

The Aggregate Stage 1c+ collapsed six overlapping moment DataFrames (`statistics_df`, `_statistics_total_df`, `report_ser`, `_audit_df`, `report_df`, `statistics`) into a single canonical `stats_df` with:

- **MultiIndex rows on `(component, measure)`** — components `meta`, `freq`, `sev`, `agg`; measures like `mean`, `cv`, `skew`, `ex1`, `ex2`, `ex3`, plus meta keys (`name`, `limit`, etc.).
- **Columns per-mixture-component + `mixed` + `independent` + `empirical` + `error`** — the `mixed`/`independent` distinction matters when frequency mixing introduces between-component correlation.
- **Two-phase build:** theoretical content in `__init__`, empirical content appended in `update_work`.

Replicate that structure on Portfolio, adapting columns to be per-unit instead of per-mixture-component.

## Frames being collapsed on Portfolio

| current frame                | replacement                                                          |
|:-----------------------------|:---------------------------------------------------------------------|
| `statistics_df` (MultiIndex) | `stats_df.drop(columns=['mixed','independent','empirical','error'])` |
| `statistics` (alias)         | gone — use `stats_df`                                                |
| `report_df` (flat stat names)| gone — use `stats_df`                                                |
| `report` (display HTML)      | gone — `qd(stats_df)` or `stats_df.style`                            |
| `audit_df` (empirical+error) | `stats_df[['empirical','error']]`                                    |
| `make_audit_df` (helper)     | gone — folded into stats_df construction in `update`                 |

## Design — Portfolio's stats_df

**Rows:** Same shape as Aggregate's `_STATS_ROW_INDEX`. May share the constant — decide during implementation whether to import `_STATS_ROW_INDEX` from `aggregate.distributions` or define a parallel `_PORT_STATS_ROW_INDEX`. (Recommend: define `_PORT_STATS_ROW_INDEX` in `portfolio.py`. If they happen to be identical, fine; if Portfolio needs extra rows in the future, no entanglement with Aggregate.)

**Columns:** one per unit + `mixed` + `independent` + `empirical` + `error`. The per-unit columns hold each `Aggregate.stats_df['mixed']` (or appropriate column) — i.e., the unit's view of its own moments. The Portfolio-level `mixed`/`independent`/`empirical`/`error` columns hold portfolio-aggregate moments.

**Build phases:**

1. **`Portfolio.__init__`** (after the existing `statistics_df` build, in parallel for now): construct the empty stats_df shell with `_PORT_STATS_ROW_INDEX` rows and per-unit + total columns; fill per-unit cells from each `Aggregate.stats_df['mixed']`; fill `mixed`/`independent` from the `MomentAggregator`'s totals.
2. **`Portfolio.update`** (after the existing audit_df build, in parallel for now): fill the `empirical` column from `xsden_to_meancvskew` results on the FFT output; fill `error = empirical / mixed - 1`.

Same recipe Aggregate followed.

## Tasks (execution order)

This sub-project follows the **parallel-write → flip readers → delete old** pattern that worked on Aggregate.

1. **Define `_PORT_STATS_ROW_INDEX`** in `portfolio.py`. Either import from `aggregate.distributions._STATS_ROW_INDEX` or define a fresh constant with the same shape. Decide based on whether Portfolio needs any extra rows now (probably not).

2. **Build `stats_df` in `__init__` alongside the existing frames.** Don't delete anything yet — parallel-write. Verify the cell values match the old frames' values for a representative test case (e.g., a two-unit Portfolio with a known calibrated structure).

3. **Build empirical + error columns in `update`.** Same parallel-write principle: keep the existing `make_audit_df`/`audit_df` build alive; add the stats_df update alongside.

4. **Verify numerical equivalence.** Run pytest. The new stats_df cells should match the corresponding old frames' cells.

5. **Flip `Portfolio.describe` to read from stats_df.** Mirror what Aggregate's describe does now: source theoretical moments from `stats_df['mixed']`, empirical from `stats_df['empirical']`, error from `stats_df['error']`. Build the 3-row Freq/Sev/Agg × E[X]/CV/Skew/Est/Err table.

6. **Flip `Portfolio.info` if it reads old frames.** Check `info`'s body; if it reads `statistics_df` for any reason, flip to `stats_df`.

7. **Flip the moved PIR-exhibit code in `portfolio_pir.py`.** Any function that reads `port.audit_df`, `port.statistics_df`, etc. — flip to `port.stats_df`. Particularly check `accounting_economic_balance_sheet` (reads `audit_df['Mean']`).

8. **Delete the old frames.** Once all readers are flipped:
    - Delete `statistics_df`, `statistics`, `report_df`, `report`, `audit_df`, `make_audit_df` from `Portfolio`.
    - Delete `_audit_df` if present.
    - Delete state init lines in `__init__` for these attrs.
    - Delete any `MomentAggregator.stats_series` call (if Aggregate Stage 1c+ retired it; otherwise leave alone).

9. **Doc sweep.** `rg "\\.statistics_df|\\.report_df|\\.audit_df|\\.statistics\\b" docs/ -t rst`. For each Portfolio-object reference (vs. Aggregate-object reference, which is already migrated), rewrite to `stats_df` syntax. Mirror exactly what we did in the Aggregate Stage 1c+ doc migration. Roughly:
    - `port.statistics_df.loc[('agg', 'mean'), 'total']` → `port.stats_df.loc[('agg', 'mean'), 'mixed']`
    - `port.report_df.loc['agg_m']` → `port.stats_df.loc[('agg', 'mean')]`
    - `port.audit_df['EmpMean']` → `port.stats_df.loc[('agg', 'mean'), 'empirical']`
    - `print(port.statistics)` → `print(port.stats_df)`

10. **Run full pytest.** 415 passed.

11. **README + version bump.** Bullet below. Bump alpha (likely the version that ends the Portfolio refactor — could even be `1.0.0b1` if you want to mark the milestone, your call).

## Verification

- `uv run pytest` → 415 + 6 PEG regression tests passed.
- **PEG regression check:** `uv run pytest tests/test_portfolio_peg_regression.py -v` → 6 passed. The PEG baseline numbers (calibration shapes, portfolio moments, pricing readouts) must remain identical — the stats consolidation is a presentation refactor, not a numerical one.
- Visual `test_suite` HTML report green.
- `port.stats_df`, `port.describe`, `port.info` all produce correct outputs on a multi-unit example. Spot-check: for PEG, `port.stats_df.loc[('agg', 'mean'), 'mixed']` matches `BASELINE['portfolio_moments']['agg_m']`.
- Doc grep clean: `rg "\\.report_df|\\.statistics_df|\\.audit_df|\\.statistics\\b" docs/ -t rst` returns nothing on Portfolio object access.
- PMIR unaffected: PMIR doesn't read these old frames (uses `density_df`, `q`, `distortions`, `agg_m`, etc. — all survive intact).

## Commit

One commit at sub-project end after human review.

**Commit message draft:**

```
Portfolio refactor E: stats consolidation — stats_df is the single source of truth

Mirror of Aggregate's Stage 1c+ work. Six overlapping stats frames
(statistics_df, statistics, report_df, report, audit_df, make_audit_df)
collapsed into a single canonical stats_df.

- stats_df: MultiIndex on (component, measure) × per-unit + mixed +
  independent + empirical + error columns.
- describe and info now read from stats_df.
- portfolio_pir exhibit functions updated to read stats_df.
- Old frames removed.
- Docs migrated to stats_df syntax (mirrors prior Aggregate migration).

Public stats surface on Portfolio is now exactly three things: info,
describe, stats_df — same shape as Aggregate.
```

## README bullet draft

```rst
- ``Portfolio`` stats consolidation: ``statistics`` / ``statistics_df``
  / ``report`` / ``report_df`` / ``audit_df`` / ``make_audit_df``
  collapsed into a single canonical ``stats_df`` (MultiIndex on
  ``(component, measure)`` × per-unit + ``mixed`` + ``independent`` +
  ``empirical`` + ``error``). Public stats surface on ``Portfolio`` is
  now exactly three things: ``info``, ``describe``, ``stats_df`` — same
  shape as ``Aggregate``.
```

## Post-conditions

Portfolio's stats surface is parallel to Aggregate's. The Portfolio refactor as a whole is complete. The combined Aggregate + Portfolio refactor produces a coherent two-class API: `info` / `describe` / `stats_df` on both, with consistent moment representations.

## Cross-cutting reminders

- `UV_LINK_MODE=copy` is set. Don't override.
- Run `uv run pytest`, not bare `pytest`.
- PowerShell shell: use `rg` directly via Bash; no awk/sed.
- Don't prefix Bash commands with `cd`.
- Update `README.rst` at sub-project close. Bump alpha version.
- Don't build docs in a verification cycle — doc audits via `rg` only.
- Do NOT commit unless the user explicitly says so.
- Pre-existing visual `test_suite` failure: `Cc.Freq20`. Ignore.
