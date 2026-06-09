# Plan numerics-1 — unit-density decoupling

> Part of the numerics program; see `plan-numerics-0-meta.md` for the target
> architecture and steering rules. **First executable plan.** Low risk:
> pure-additive accessors plus mechanical migration of *display* readers off the
> `p_{unit}` columns. **No compute change, no distortion surface touched.**

## Goal

Establish the mechanism for sourcing unit pmfs from the owning `Aggregate`
objects, and move every *non-kappa* consumer onto it, so that by the end the
**only** remaining reader of `Portfolio.density_df['p_{unit}']` is the kappa
construction in `add_exa`. That last reader is removed in
`plan-numerics-2-objective` (when kappa goes shifted-support), at which point the
`p_{unit}` write is dropped. This plan keeps writing `p_{unit}` (legacy), so it is
behavior-preserving.

Rationale (meta §2): splitting this out shrinks the blast radius of the objective
rewrite to *only* the kappa compute, and these accessors are independently testable.

## Background facts (verified in current code)

- `Aggregate` already models the native-grid split: `density_df` (aggregate on
  `self.xs`) vs `sev_density_df` (severity on `self.xs_sev`). We mirror that one
  level up for the portfolio.
- `Portfolio.update` writes `p_{unit}` in **both** paths: non-signed
  (`portfolio.py:2068`, `= agg.agg_density`) and signed (`:2050`, a `np.roll` of the
  unit's density onto the total origin). The signed-path column is the misleading
  one (a rolled presentation, not a native pmf).
- Current `p_{unit}` readers to migrate (grep `p_{` / `filter(regex='p_')` /
  `f'p_{...}'`):
  - `portfolio.py`: `percentiles` (`:1656`), `_limits` (`:2425`), `plot` (`:2501`),
    `reins_density_df` display (`:1149`), `sample_density_compare` (`:615`),
    `var_dict` unit quantiles (`:3239`), `trim_df` (`:2395`).
  - `pedagogy.py`: density panels and `plot_twelve` (`:674, :1221, :1315, :1337,
    :1339, :1464`).
  - `add_exa` (`:2617–2658`) — kappa + stand-alone `lev_/e_`: **left for
    numerics-2** (this is the deliberate last reader).

## Deliverables

### 1. Accessors (additive, on `Portfolio`)

```
unit_density(unit, view='agg') -> pandas.Series      # native unit pmf, index = unit loss
unit_density_df(view='agg')    -> pandas.DataFrame   # long form, index (unit, loss)
aligned_unit_density_df(grid='total'|'union'|'zero', *, allow_window_mismatch=False)
```

- `unit_density` / `unit_density_df` read each `agg.density_df.p_total` on the unit's
  **own** grid (concat for the long form). Columns for the long form:
  `unit, loss, p, F, S, bs, x_min, x_max, mass` (the window-audit metadata the
  windowed-world note §Requirements item 9 asks for).
- `aligned_unit_density_df` is the **explicitly-named display adapter**: reindex /
  scatter unit pmfs onto a requested grid. For the legacy zero-origin book,
  `grid='total'` reproduces today's `p_{unit}` columns exactly. For a genuinely
  windowed book it **warns** unless `allow_window_mismatch=True` (steering rule 3:
  any unit-pmf-on-total-grid view is a labelled artifact, never compute input).
  `bs` is forced common — the `Portfolio` dictates `bs` to its `Aggregate`s — so
  `'union'` is well defined; assert equal `bs` anyway and fail loudly if a unit was
  re-updated off-grid.
- (Maybe) a thin `Aggregate.pmf_series(view='agg')` if `density_df.p_total` proves
  too heavy as the source — decide during step 1; default is to just read
  `density_df`.

### 2. Migrate display readers

Point `percentiles`, `_limits`, `plot`, `reins_density_df` display,
`sample_density_compare`, `var_dict` unit quantiles, and `trim_df` at the accessors.
None of these are compute-critical; they are display / range / reporting.

- Quantile facts (verified): `Aggregate.q` is the exact step-function quantile
  (`make_var_tvar`, lower/upper inf-definition — no interpolation);
  `Aggregate.cdf` uses `interp1d(kind='previous')`, i.e. a proper step function.
  The outlier is `percentiles` (`:1670`): `interp1d(kind='linear',
  fill_value='extrapolate')` on `p_{line}.cumsum()` — deliberately *interpolated*
  display percentiles (per its docstring). **Preserve that behavior**: re-source
  the cumulative from the native unit pmf via the accessor, keep `kind='linear'`
  — behavior-identical, no numeric drift. (Unifying `percentiles` onto exact step
  `q` is a separate one-line decision, not taken here.)
- `var_dict`: unit quantiles delegate to the unit `Aggregate` (`agg.q` — same
  step machinery as `Portfolio.q`, so consistency is exact); total stays on
  `Portfolio.density_df`.
- `plot` / `_limits`: total range from `p_total`; unit overlays from
  `unit_density_df()`.
- `reins_density_df`: the *display* frame uses native gross/ceded/net unit views;
  the reinsurance *combine* rework that needs native origins is deferred (it rides
  with numerics-2/4, windowed-world note §Reinsurance) — here only de-couple the
  display read.

### 3. Pedagogy density panels

`plot_twelve` and the other pedagogy consumers that read `p_{unit}` for **density /
bivariate** panels switch to `unit_density_df()` / native axes
(`Z = outer(unit_2.p, unit_1.p)`). The *allocation* panels (kappa/alpha/S) are
untouched here — they ride with numerics-3's `allocation_diagnostics` frame. (This
is the first slice of "plot_twelve is not the boss"; the rest is numerics-3.)

### 4. Stale pointer fix

The signed-path warning at `portfolio.py:2106` still references the deleted
`dev/plan-portfolio-neg-x-pricing.md`. Repoint the message at
`dev/plan-numerics-2-objective.md` (which removes the warning entirely when the
signed objective path lands).

## Step 0 — inventory (do before editing)

Produce the reader table: every site that reads `density_df['p_{unit}']` /
`filter(regex='p_')`, classified **display** (migrate now) vs **kappa/stand-alone**
(leave for numerics-2). This is the small-scale version of the meta's
"measure, don't guess" rule and guards against missing a hidden reader.

## Tests (`tests/test_portfolio*.py`)

- `unit_density(u).sum()` == the unit's represented mass; index == unit's native loss.
- Two-unit **disjoint-window** book: `unit_density_df` rows equal each unit's own
  `density_df` p; the accessor does not depend on total-grid overlap.
- **Legacy parity:** for a standard zero-origin book,
  `aligned_unit_density_df(grid='total')` reproduces the current `p_{unit}` columns
  exactly (this is the behavior-preservation gate).
- `aligned_unit_density_df(grid='total')` on a windowed book **warns** without
  `allow_window_mismatch=True`.
- `Portfolio.plot` and the migrated reports render on a book without reading
  `p_{unit}` as compute (assert via a guard / spy that the migrated paths call the
  accessor).
- Full `pytest` green; baselines unchanged (no numeric output changed).

## Files

- `src/aggregate/portfolio.py` — new accessors; migrate the seven display readers.
- `src/aggregate/distributions.py` — (optional) `Aggregate.pmf_series`.
- `src/aggregate/pedagogy.py` — density/bivariate panels onto native unit pmfs.
- `tests/` — accessor + parity + windowed-warn tests.

## Out of scope (explicit)

- Kappa / `add_exa` (numerics-2). The `p_{unit}` **write** stays.
- Any distortion column, `T.*`/`M.*`, allocation diagnostics (numerics-3).
- Reinsurance native-grid *combine* (numerics-2/4).

## Housekeeping

Version bump (own `1.0.0a*`); CHANGELOG entry (new accessors; note `p_{unit}` is now
legacy/scheduled-for-removal); `dev/TODO.md` N2 row gains a sub-bullet; move to
`dev/done/` on landing. US spelling; `reins` canonical.
