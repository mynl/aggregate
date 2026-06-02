# reins-reporting — rationalize reinsurance reporting (Aggregate + Portfolio)

**Target version: 1.0.0a19** (execute *after* `reins-buckets.md`).

*Revised 2026-06-01 with author decisions — see "Decisions (2026-06-01)" below;
they supersede the first-draft `reins_stats_df` (per-layer) and `reins_describe`
(single 3-row gcn block) shapes.*

## Context

The reinsurance reporting surface is fragmented across overlapping,
inconsistently-shaped DataFrames (`reinsurance_df`, `reinsurance_audit_df`,
`reinsurance_report_df`, `reinsurance_occ_layer_df`) and there is **no
Portfolio-level reinsurance reporting at all**. Crucially, almost all of the
underlying numbers are *already computed* (see "Data already computed") — this
is a re-assembly, not new math, except for the new Portfolio gross/ceded/net
densities. See `dev/pipeline-reinsurance.rst` for the object-by-object anatomy
of the current surface.

## Decisions (2026-06-01)

These are locked and drive the design below.

1. **Naming convention (gross vs subject).**
   - **Gross** = what enters at the *top of step 1*, before any cover.
   - **Subject** = what flows *into* a stage's cover. For occurrence the
     subject *is* gross; for aggregate the subject is the *occurrence output*
     (already net/ceded of occ).
   - `gcn` = gross/ceded/net; `fsa` = freq/sev/agg.
   - In `reins_describe` the **occurrence block leads with Gross**, the
     **aggregate block leads with Subject**. Never label a top row "Subject"
     where "Gross" is meant, and never call the agg-cover input "Gross".

2. **Exact vs rebucketed — stop muddling them in one frame.** The per-stage
   engine produces two *different* things:
   - **rebucketed densities** (`p_net` / `p_ceded`) — mass on the model grid
     (`loss = k·bs`); the empirical, FFT-ready objects. These slot **straight
     into `reins_density_df`** (no separate `_occ_reins_df` / `_agg_reins_df`).
   - **exact loss images** (`loss_net = netter(x)`, `loss_ceded = ceder(x)`) —
     exact real values, off-grid. Used **only** in moment calcs, so they are
     **not stored**; they are computed on demand from the `ceder` / `netter`
     functions already retained on the object (`occ_ceder`/`occ_netter`,
     `agg_ceder`/`agg_netter`).

   Hence two bases for every stat:
   - **EX** ("theoretic") = exact moments — `Σ ceder(xs[k])^j · p_subject[k]`
     (and the netter analogue), **no rebucketing scatter**. Pre-bucket truth.
   - **Est** ("empirical") = moments of the **rebucketed** density (`linear` or
     `nearest`) read from `reins_density_df`.
   - **Change** = `(Est − EX) / EX` makes the rebucketing error visible: ≈ 0 for
     `linear` on the mean (mass-split preserves E exactly); ≤ `bs/2` positional
     bias for `nearest`. This is the natural home for validating `reins-buckets`.

3. **Drop the `F_` columns** (`F_subject`/`F_net`/`F_ceded`) from the engine
   frame. They are vestigial from the old `interp1d`-of-grouped-CDF algorithm
   (which built `F_net`/`F_ceded` then `np.diff`'d to *get* the densities). The
   scatter computes `p_net`/`p_ceded` directly; the only remaining reader is the
   debug-plot CDF panel, which `cumsum`s inline.

4. **Drop the by-layer detail.** No persistent per-layer object. By-layer can be
   recomputed on the fly if a future need arises, but the core stats are
   **whole-structure**: exact (pre-bucket) "theoretic" numbers compared against
   the `linear`/`nearest` bucketed empirical. This removes `reinsurance_audit_df`,
   `reinsurance_occ_layer_df`, and the `_reins_audit_df_work` engine.

5. **Aggregate is treated analogously to occurrence throughout** — same engine,
   same gross/subject/ceded/net views, same EX-vs-Est bases; only the subject
   differs (gross severity for occ; occ-output aggregate for agg).

6. **`reins_density_df` has consistent columns** regardless of which programs are
   present. A missing stage contributes the no-cession values (ceded = 0, net =
   subject) rather than `None`/absent columns.

7. **`reins_describe` is per-stage** (not a single 3-row gcn block): one block
   per applicable stage, each a `view × fsa` table with `EX | Est | Change`
   columns. The occ/agg split lives here, in the daily driver.

## Inventory → fate (revised)

| # | item | fate |
|---|------|------|
| 1 | `reinsurance_description` | unchanged |
| 2 | `reinsurance_kinds` | unchanged |
| 3 | `agg_reins` tuples | unchanged |
| 4 | `occ_reins` tuples | unchanged |
| 5 | `occ_reins_df` | **remove** — densities live on `sev_density_{gross,ceded,net}` + `reins_density_df`; exact values on demand via `occ_ceder`/`occ_netter` |
| 6 | `agg_reins_df` | **remove** — ditto via `agg_ceder`/`agg_netter` |
| 7 | `reinsurance_df` | **rename** → `reins_density_df`; consistent columns; `p_agg_gross_occ → p_agg_gross`, `p_agg_gross → p_agg_subject` |
| 8 | `reinsurance_audit_df` | **remove** (by-layer dropped) |
| 9 | `reinsurance_occ_layer_df` | **remove** (by-layer dropped) |
| 10 | `reinsurance_report_df` | **remove** → folded into `reins_stats_df` / `reins_describe` |
| 11 | `reinsurance_occ_plot` | unchanged (re-point reads to the density members / `reins_density_df`) |
| – | `_reins_audit_df_work` | **remove** (per-layer engine no longer needed) |
| – | `F_*` engine columns | **remove** (vestigial) |

## Data already computed (confirmed — re-assembly, not new math)

- `stats_df` staged columns: `mixed` (subject theoretical), `gross_empirical`
  (subject empirical), `after_occ` (post-occ empirical), `empirical` (final
  modeled), `occ_impact`, `agg_impact` — `distributions.py` `_init_stats_df` and
  the meta.4 staged-reins block in `update_work`.
- gcn severity densities on `sev_density_{gross,ceded,net}`; gcn aggregate
  densities on `agg_density_{gross,ceded,net}`; the FFT helper `_fft_aggregate`
  (honours the fixed-1 / zero-risk shortcuts) for aggregates of any severity
  view.
- `ceder`/`netter` step functions retained per stage → the exact (EX) moments
  are one vectorised `Σ ceder(xs)^j · p_subject` away, no rebucketing.
- Expected ceded counts `n · P(X > attach)` (the old
  `reinsurance_occ_layer_df` `en`).

## Design

### A. `reins_density_df` (rename + consistent columns)

One row per grid bucket. Columns (always present; missing-stage = no-cession):

- **Severity (occurrence level)**: `p_sev_gross`, `p_sev_ceded`, `p_sev_net`.
- **Aggregate of each occ severity view**: `p_agg_gross` (= `_fft_aggregate`
  of gross severity — the true gross aggregate), `p_agg_ceded_occ`,
  `p_agg_net_occ`.
- **Aggregate cover**: `p_agg_subject` (= aggregate of the *requested* occ
  output = the agg cover's input; equals `p_agg_gross` when no occ),
  `p_agg_ceded`, `p_agg_net`.

Renames vs today: `p_agg_gross_occ → p_agg_gross`; `p_agg_gross → p_agg_subject`
(the author's "third-to-last column" fix). All aggregate columns come from
`_fft_aggregate` so the shortcuts are consistent. Re-point `reinsurance_occ_plot`
and delete the old `None`-filled branches.

### B. `reins_stats_df` (per-stage, both bases) — single source of truth

Shaped like `stats_df`, but stage/view/basis instead of components:

- **Rows**: MultiIndex `(component, measure)` — `('freq'|'sev'|'agg',
  'ex1'|'ex2'|'ex3'|'mean'|'cv'|'skew')`.
- **Columns**: hierarchical `(stage, view, basis)`, variable by what's applied:
  - **occ** (if `occ_reins`): view ∈ `gross | ceded | net`.
  - **agg** (if `agg_reins`): view ∈ `subject | ceded | net`.
  - **basis** ∈ `EX | Est`.
- **EX** sourcing: severity/aggregate moments from the exact
  `Σ ceder(xs)^j · p_subject` (and netter) — pre-bucket; freq ceded =
  `n · P(X > attach)`, freq gross/subject = `E[N]`, freq net = NaN (degenerate).
- **Est** sourcing: moments of the rebucketed densities in `reins_density_df`
  (via `xsden_to_mwrangler`); agg fsa totals can reuse the `stats_df` staged
  columns (`mixed` / `after_occ` / `empirical`) where they coincide.
- Lazy, cached in `_reins_stats_df`, invalidated by the `reins_bucket` setter
  (from `reins-buckets.md`) and on `update`.

### C. `reins_describe` (per-stage daily driver) — fixed layout

One block per applicable stage; each block is `view × fsa` rows ×
`EX | Est | Change` columns:

- **Occurrence block** (leads with **Gross**): rows `(Gross|Ceded|Net) ×
  (Freq|Sev|Agg)`. The `Agg` row of each view = aggregate of that severity view
  (`p_agg_gross` / `p_agg_ceded_occ` / `p_agg_net_occ`).
- **Aggregate block** (leads with **Subject**): rows `(Subject|Ceded|Net) ×
  (Agg)` (Sev not applicable; Freq degenerate → NaN). Views map to
  `p_agg_subject` / `p_agg_ceded` / `p_agg_net`.
- **Columns**: `EX` (exact pre-bucket), `Est` (rebucketed), `Change`
  (`(Est − EX)/EX`, same arithmetic as `describe`'s `Err`).
- Degenerate cells (`Net.Freq`) render `NaN` with a documented convention.
- Index a MultiIndex `(stage, view, component)`; rectangular within each block.
- Derived entirely from `reins_stats_df` (which is derived from
  `reins_density_df` + the exact ceder/netter + `stats_df`).

### D. Portfolio-level (all new) — end-to-end gcn

Per-stage detail is only meaningful per unit (units may carry different
programs), so the **Portfolio objects are end-to-end** (final gross / ceded /
net of the portfolio aggregate); the per-stage breakdown stays in each unit's
`reins_describe`.

- **`Portfolio.reins_density_df`**: portfolio gross/ceded/net *aggregate*
  densities by convolving the per-unit gcn aggregate densities under the same
  independent-FFT machinery the total already uses, assembled from each unit's
  `reins_density_df`. Units without reinsurance contribute gross = ceded = net =
  modeled. Consistent columns. Docstring notes the independence assumption.
  *(This convolution is the one genuinely new computation.)*
- **`Portfolio.reins_stats_df`**: per-unit columns + a portfolio `total` derived
  from `reins_density_df`, mirroring `Portfolio.stats_df`. End-to-end gcn views
  (no per-stage split at the portfolio level).
- **`Portfolio.reins_describe`**:
  ```python
  pd.concat([u.reins_describe for u in self] + [total_block],
            keys=names + ['total'], names=['unit', ...])
  ```
  — the `Portfolio.describe` assembly pattern. Total block is end-to-end gcn:
  means sum across units per view; cv/skew from the portfolio gcn densities.
- Gate all three on "any unit cedes" (reuse `_reins_after_label`); return `None`
  when no unit cedes.

## Files

- `src/aggregate/distributions.py` — rename `reinsurance_df` →
  `reins_density_df` (consistent columns; `p_agg_gross_occ → p_agg_gross`,
  `p_agg_gross → p_agg_subject`); add `reins_stats_df`, `reins_describe`; delete
  `occ_reins_df` / `agg_reins_df`, `reinsurance_audit_df`,
  `reinsurance_report_df`, `reinsurance_occ_layer_df`, `_reins_audit_df_work`;
  drop the `F_*` columns from `_apply_reins_work` (rebuild the debug CDF panel
  via `cumsum`); ensure `occ_ceder/occ_netter` and `agg_ceder/agg_netter` are
  retained for the exact (EX) path.
- `src/aggregate/portfolio.py` — `reins_density_df`, `reins_stats_df`,
  `reins_describe`; reuse `_reins_after_label`.
- Docs (lockstep, **do not build** per CLAUDE.md): rewrite
  `docs/2_user_guides/2_x_re_pricing.rst` and `docs/2_user_guides/2_x_cat.rst`
  to the new API; refresh `dev/pipeline-reinsurance.rst` Part B; grep `docs/`
  for stale `:meth:` / `:attr:` references to the removed names.

## Verification

- New `tests/test_reins_reporting.py`: for occ-only, agg-only, and both-programs
  single-`Aggregate` cases assert
  - `reins_density_df` has the **same columns** regardless of which programs are
    present (consistent-columns invariant); `p_agg_subject` present;
  - `reins_describe` per-stage shape (occ block leads Gross; agg block leads
    Subject; `view × fsa` × `EX|Est|Change`);
  - **EX vs Est**: `linear` ⇒ `Change` on every mean ≈ 0 (≤ `10·VALIDATION_NOISE`);
    `nearest` ⇒ `|Est − EX| ≤ bs/2` on the mean; both conserve mass;
  - `reins_stats_df` columns match the applied stages/views.
- Portfolio cases (`ONEre`, `BOTHre`, `TEST`, + a gross-only control): assert
  `reins_describe` aligns (`unit/...` + `total`), total means == sum of unit
  means per view (end-to-end gcn), and the gross-only port returns `None`.
- `uv run pytest` green (`UV_LINK_MODE=copy`); regenerate any moved baselines
  deliberately; sync any DecL into `src/aggregate/agg/test_decl.agg`.
- Smoke-render `reins_describe` / `reins_stats_df` via `qd(...)`.

## Close-out

- Bump `pyproject.toml` to `1.0.0a19`.
- README.rst bullets: three new reins objects (density/stats/describe) with the
  exact-vs-rebucketed `EX|Est|Change` view, removed legacy DataFrames + `F_`
  columns, Portfolio reins reporting; note the doc rewrite is pending a manual
  rebuild (per CLAUDE.md).

## Open / watch

- The `reinsurance_occ_layer_df` `severity` column used a `sf(attach/share)`
  gross-up (likely a bug). It is being removed; if the conditional-severity
  number is wanted in `reins_stats_df`, use `sf(attach)` and confirm.
- "Subject" labelling for the agg stage depends on the requested occ view
  (`net of` / `ceded to`); keep `reins_stats_df` / `reins_describe` consistent
  with `reinsurance_description`.
- Portfolio gcn convolution assumes unit independence (consistent with existing
  portfolio totals) — note in the docstring.
- EX for the agg stage is exact *relative to the post-occ subject aggregate*,
  which itself already carries the occ rebucketing. The per-stage EX/Est/Change
  isolates each stage's rebucketing error; document that the agg `Change` is the
  agg-stage contribution, not cumulative.
