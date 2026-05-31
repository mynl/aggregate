# reins-reporting — rationalize reinsurance reporting (Aggregate + Portfolio)

**Target version: 1.0.0a19** (execute *after* `reins-buckets.md`).

## Context

The reinsurance reporting surface is fragmented across overlapping,
inconsistently-shaped DataFrames (`reinsurance_df`, `reinsurance_audit_df`,
`reinsurance_report_df`, `reinsurance_occ_layer_df`) and there is **no
Portfolio-level reinsurance reporting at all**. This is the same situation the
`Aggregate`/`Portfolio` stats reporting was in before the report
rationalization. Crucially, almost all of the underlying numbers are *already
computed* (see "Data already computed") — this is a re-assembly, not new math,
except for the new Portfolio gross/ceded/net densities.

## Goal

Rationalize into three coherent objects, defined at the `Aggregate` level and
combined at the `Portfolio` level (all new at the Portfolio level):

- **`reins_density_df`** — all gross/ceded/net (gcn) densities (rename of
  `reinsurance_df`).
- **`reins_stats_df`** — single source of truth for reins stats; variable
  columns by what reinsurance is applied; feeds `reins_describe`. Subsumes
  `reinsurance_audit_df`, `reinsurance_report_df`, `reinsurance_occ_layer_df`.
- **`reins_describe`** — fixed-layout daily-driver, `unit / gcn / fsa` rows ×
  `EX | Est | Change` columns.

## Inventory → fate (author's list 1–11)

| # | item | fate |
|---|------|------|
| 1 | `reinsurance_description` | unchanged |
| 2 | `reinsurance_kinds` | unchanged |
| 3 | `agg_reins` tuples | unchanged |
| 4 | `occ_reins` tuples | unchanged |
| 5 | `occ_reins_df` | keep, make private `_occ_reins_df` |
| 6 | `agg_reins_df` | keep, make private `_agg_reins_df` |
| 7 | `reinsurance_df` | **rename** → `reins_density_df` |
| 8 | `reinsurance_audit_df` | **remove** → folded into `reins_stats_df` |
| 9 | `reinsurance_occ_layer_df` | **remove** → folded into `reins_stats_df` |
| 10 | `reinsurance_report_df` | **remove** → folded into `reins_stats_df` |
| 11 | `reinsurance_occ_plot` | unchanged |

## "Subject" vs "Gross" terminology

Use **Subject** for the input to a stage that may itself be ceded output (agg
covers act on the requested occ output, so "Gross" would be wrong there). Reserve
**Gross** for the very top — what enters before any occurrence cover. At the
Portfolio total, "Gross" = sum of unit subjects-before-occ.

## Data already computed (confirmed — re-assembly, not new math)

- `stats_df` columns: `mixed` (subject theoretical), `gross_empirical` (subject
  empirical), `after_occ` (post-occ empirical), `empirical` (final modeled),
  `occ_impact`, `agg_impact` — see `distributions.py:2496` (`_init_stats_df`)
  and the meta.4 staged-reins block in `update_work` (`distributions.py:2966`).
- `reins_density_df` (= `reinsurance_df`, `distributions.py:1802`):
  `p_sev_{gross,ceded,net}`, `p_agg_{gross,ceded,net}_occ`,
  `p_agg_{gross,ceded,net}`.
- Per-layer stats: `_reins_audit_df_work` (`distributions.py:1951`) applies each
  layer alone via `_apply_reins_work` → per-layer mean/cv/skew (occ on severity,
  agg on the aggregate density). `reinsurance_occ_layer_df`
  (`distributions.py:1847`) adds expected counts by layer (`n * sev.sf(attach)`).

## Design

### A. `reins_density_df` (rename)

Pure rename of `reinsurance_df` and its backing `_reinsurance_df`. Update the
one internal use site (`reinsurance_occ_plot`, `reinsurance_report_df` logic)
and all docs. No behavior change.

### B. `reins_stats_df` (Aggregate)

Single source of truth, shaped like `stats_df`:

- **Rows**: MultiIndex `(component, measure)` — `meta` rows (numbers only) plus
  `('freq'|'sev'|'agg', 'ex1'|'ex2'|'ex3'|'mean'|'cv'|'skew')`.
- **Columns**: hierarchical `(stage, view)`, variable by what's applied:
  - occ stage (if `occ_reins`): one column per occ layer keyed by
    `(share, limit, attach)`; plus `('occ','ceded total')`,
    `('occ','net total')`, `('occ','subject')`.
  - agg stage (if `agg_reins`): one column per agg layer; plus
    `('agg','ceded total')`, `('agg','net total')`, `('agg','subject')`.
- **Sources**: per-layer columns from `_reins_audit_df_work` (kept as the
  per-layer engine); totals from `reins_density_df` moments
  (`xsden_to_mwrangler`) for sev/agg; freq impact (expected ceded counts) from
  the existing `n * sev.sf(attach)` logic; agg fsa totals reuse the `stats_df`
  staged columns (`mixed` / `after_occ` / `empirical`).
- Computed lazily, cached in `_reins_stats_df`, invalidated by the
  `reins_bucket` setter (from `reins-buckets.md`) and on `update`.

### C. `reins_describe` (Aggregate) — fixed layout

- **Rows**: MultiIndex `('Gross'|'Ceded'|'Net', 'Freq'|'Sev'|'Agg')`,
  rectangular — reads as three stacked mini-`describe` blocks.
- **Cols**: `EX` (analytic / most-accurate available) | `Est` (post-rebucketing
  empirical) | `Change` (`(Est − EX) / EX`, same arithmetic as `describe`'s
  `Err`).
- **Cell sourcing** (all from §B / `reins_density_df` / `stats_df`):
  - **Gross**: Freq = `E[N]`; Sev = gross severity mean/cv/skew; Agg =
    **subject** (the occ output the agg program sees,
    `stats_df['mixed']` / `gross_empirical`).
  - **Ceded**: Freq = expected ceded count (`n · P(X > attach)`); Sev/Agg =
    ceded moments.
  - **Net**: Sev/Agg = net moments. **Net/Freq is degenerate** (= gross count) →
    rendered `NaN` with a documented convention. Under-defined cells follow the
    same rule; the full occ-vs-agg, layer-by-layer detail lives in
    `reins_stats_df`.
  - `EX` uses analytic / pre-rebucketing values where available (most accurate);
    `Est` uses the post-rebucketing empirical density.

### D. Portfolio-level (all new)

- **`Portfolio.reins_density_df`**: build portfolio gross/ceded/net *aggregate*
  densities by convolving the per-unit gcn aggregate densities (same
  independent-FFT machinery the portfolio already uses for its total), assembled
  from each unit's `reins_density_df`. Units without reinsurance contribute
  gross = ceded = net = modeled. Docstring notes the independence assumption
  (consistent with existing portfolio totals).
- **`Portfolio.reins_stats_df`**: combine per-unit `reins_stats_df` (per-unit
  columns) plus a portfolio `total` derived from `reins_density_df`, mirroring
  how `Portfolio.stats_df` aggregates units.
- **`Portfolio.reins_describe`**:
  ```python
  pd.concat([u.reins_describe for u in self] + [total_block],
            keys=names + ['total'], names=['unit', 'view', 'X'])
  ```
  — the exact assembly pattern of `Portfolio.describe` (`portfolio.py:909`).
  Total block: means sum across units per view; cv/skew computed from the
  portfolio gcn densities in `reins_density_df`.
- Gate all three on "any unit has reinsurance" (reuse the
  `Portfolio._reins_after_label` helper added in the describe fix); return
  `None` when no unit cedes.

## Files

- `src/aggregate/distributions.py` — rename `reinsurance_df` →
  `reins_density_df`; add `reins_stats_df`, `reins_describe`; privatize
  `occ_reins_df` / `agg_reins_df`; delete `reinsurance_audit_df`,
  `reinsurance_report_df`, `reinsurance_occ_layer_df` (folding their math into
  `reins_stats_df`); keep `_reins_audit_df_work` as the per-layer engine.
- `src/aggregate/portfolio.py` — `reins_density_df`, `reins_stats_df`,
  `reins_describe`; reuse `_reins_after_label`.
- Docs (lockstep, **do not build** per CLAUDE.md): rewrite
  `docs/2_user_guides/2_x_re_pricing.rst` and `docs/2_user_guides/2_x_cat.rst`
  to the new API; grep `docs/` for stale `:meth:` / `:attr:` references to the
  removed names.

## Verification

- New `tests/test_reins_reporting.py`: for occ-only, agg-only, and both-programs
  single-`Aggregate` cases assert
  - `reins_describe` shape (gcn × fsa, rectangular);
  - Gross/Agg `EX` matches `describe`'s subject; Ceded + Net mean == Gross mean
    (sev and agg);
  - `reins_stats_df` columns match the applied layers.
- Portfolio cases from the original prompt (`ONEre`, `BOTHre`, `TEST`, plus a
  gross-only control): assert `reins_describe` aligns (`unit/gcn/fsa` + total),
  total means == sum of unit means per view, and the gross-only port returns
  `None`.
- `uv run pytest` green (set `UV_LINK_MODE=copy`); sync any DecL programs into
  `src/aggregate/agg/test_decl.agg`.
- Smoke-render `reins_describe` / `reins_stats_df` via `qd(...)` for the prompt's
  examples.

## Close-out

- Bump `pyproject.toml` to `1.0.0a19`.
- README.rst bullets: three new reins objects, removed legacy DataFrames,
  Portfolio reins reporting, and a note that the doc rewrite is pending a manual
  rebuild (per CLAUDE.md).

## Open / watch

- The agg-stage "subject" column depends on the requested occ view
  (`net of` / `ceded to`); confirm `reins_stats_df` / `reins_describe` label it
  consistently with `reinsurance_description`.
- Portfolio gcn convolution assumes unit independence (consistent with existing
  portfolio totals) — note in the docstring.
