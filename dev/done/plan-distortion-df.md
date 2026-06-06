# plan-distortion-df — de-cruft the calibration summary

> **STATUS: DONE — shipped in 1.0.0a34 (2026-06-06).** All three phases landed:
> `Distortion.standard_shape` → `gini_p`; `distortion_df` rebuilt to the
> per-distortion receipt (`param_name, param, gini_p, area, error`, index
> `distortion`); new one-row `calibration_df` (inputs `coc, p` + pentagon octet,
> `ROE == coc` self-check). Docs (`2_x_10mins`, `5_x_distortions`) updated; three
> regression tests added. Full suite green (1023 passed); `src` ruff clean. Maps
> to TODO **F10**. The design notes below are kept as the as-built record.

## At a glance

`Portfolio.calibrate_distortions(coc, p|a)` calibrates the standard set
`[ccoc, ph, wang, dual, tvar]` to one premium target at one asset level. Its
audit, `distortion_df`, currently mixes the *per-distortion result* (what
varies) with the *calibration target* (constant across rows) in one wide frame
with a part-vestigial `(a, LR, method)` MultiIndex.

After this change:

- **`distortion_df`** is the per-distortion receipt only — index `distortion`
  (ordered categorical), columns `param_name, param, gini_p, area, error`.
- **`calibration_df`** (new) is the calibration *target*, shown once: a one-row
  canonical pentagon octet (`L, M, P, Q, a, LR, PQ, ROE`) for the point the set
  was calibrated to.
- **`Distortion.standard_shape` → `Distortion.gini_p`** (attribute rename): the
  normalised, cross-family-comparable shape `= 2∫g−1 = p_equiv`.

## The problem

`distortion_df` today:

- **columns**: `S, L, P, PQ, Q, COC, param, std_param, error`
- **index**: MultiIndex `(a, LR, method)`, 5 rows (one per distortion).

Only `param`, `std_param`, `error` vary by row. `S, L, P, PQ, Q, COC` and the
index levels `a, LR` are the calibration *target* — identical across all five
rows by construction (every distortion is calibrated to the same `P` at the
same `a`). So ≈⅔ of the frame is constant repetition. The `(a, LR)` index and
the per-row target columns are leftovers from the **removed batch API**
(`calibrate_distortions(LRs=, COCs=, As=, Ps=)`); calibration is now strictly
one-point, and **no batch mode is wanted** (confirmed — never used).

Secondary: `std_param` is an unhelpful name (it is the comparable
TVaR-equivalent / Gini level), and `method` is vestigial (it is just the
distortion family name).

## Decisions (confirmed with author, 2026-06-06)

- **`distortion_df` becomes per-distortion only.** Index `distortion` (drop the
  constant `a`/`LR` levels); columns `param_name, param, gini_p, area, error`.
- **`param_name`** (new) — the parameter's name for that family (`r`, `shape`,
  `p`, …), since `param` means different things across families.
- **`gini_p`** — the renamed `std_param` (`= 2∫g−1 = p_equiv`); see the verified
  identity below.
- **`area`** (new) `= (gini_p + 1) / 2 = ∫₀¹ g` — the area under the distortion
  (`= loading + ½`).
- **`calibration_df`** (new attribute) — the calibration target as a **one-row
  frame: the calibration inputs `coc, p` lead, then the canonical pentagon
  octet** (`L, M, P, Q, a, LR, PQ, ROE`), built with `complete_pentagon`. Stored
  on `self.calibration_df`. Self-contained: you see the inputs and can confirm
  `ROE == coc` at a glance. (`a` is the octet's `a`, not duplicated as a lead
  column; `p` is the probability, given or computed as `cdf(a)`.)
- **No batch mode** — calibration stays one-point; we just stop pretending via
  the index.
- **Rename the attribute** `Distortion.standard_shape` → `Distortion.gini_p`
  (not just the column) — one consistent name end to end.

## Why `gini_p` is honest (verified identity)

`standard_shape` equals `p_equiv = 2∫₀¹ g − 1` for every calibrated family —
the same quantity as the `gini`/`p_equiv` rows already in
`Distortion._compute_stats_df`:

| family | `g` | `∫g` | `2∫g−1` | current `standard_shape` |
|---|---|---|---|---|
| tvar | `min(x/(1−p),1)` | `(1+p)/2` | `p` | `p` ✓ |
| ph | `x^θ` | `1/(θ+1)` | `(1−θ)/(1+θ)` | `(1−s)/(1+s)` ✓ |
| dual | `1−(1−x)^θ` | `θ/(θ+1)` | `(θ−1)/(θ+1)` | `(s−1)/(s+1)` ✓ |
| wang | `Φ(Φ⁻¹(x)+λ)` | `Φ(λ/√2)` | `2Φ(λ/√2)−1` | `2Φ(s/√2)−1` ✓ |
| ccoc | `δ+(1−δ)x` | `(1+δ)/2` | `δ` | `δ` ✓ |

So `gini_p` is exact, and `area = (gini_p+1)/2 = ∫g` follows by definition.

## Design

### D1 — `Distortion.gini_p` (rename `standard_shape`)
Rename the attribute in `spectral.py`: `_common_init` (default `np.nan`) and the
five subclass assignments (`ccoc`, `ph`, `wang`, `dual`, `tvar`), plus the two
docstring mentions. `gini_p` is set to the same value as today. (Leave the
`gini` / `p_equiv` rows in `_compute_stats_df` as they are — same number,
different reader-facing framing; no need to churn them.)

### D2 — `distortion_df` schema
In `calibrate_distortions`, build:

- **index**: `pd.CategoricalIndex(d_list, dtype=DISTORTION_DTYPE, name='distortion')`
  (keeps the canonical `ccoc, ph, wang, dual, tvar` sort).
- **columns** (per distortion):
  - `param_name` — the family's parameter name. Source: `subclass.param_name`
    (the `cll/clin/lep/ly` fallback map already in `calibrate_distortion`),
    `'r'` for `ccoc`.
  - `param` — `dist.shape`.
  - `gini_p` — `dist.gini_p`.
  - `area` — `(dist.gini_p + 1) / 2`.
  - `error` — `dist.error` (premium miss).

Drop `S, L, P, PQ, Q, COC` and the `(a, LR)` index levels (now in
`calibration_df`).

### D3 — `calibration_df` (the target, once)
Add `self.calibration_df = None` next to `self.distortion_df = None`. In
`calibrate_distortions`, after computing the target `L (=exa), P, Q (=a−P)`,
build a one-row pentagon:

```
cal = complete_pentagon(
    pd.DataFrame([[exa, P - exa, P, a - P]],
                 columns=['L', 'M', 'P', 'Q'],
                 index=pd.Index(['calibration'], name='line')))
# lead with the inputs: coc, p (a is already the octet's a)
cal.insert(0, 'p', p_val)        # p_val = p if given else self.cdf(a)
cal.insert(0, 'coc', coc)
calibration_df = cal
```

`a, LR, PQ, ROE` fill in automatically; `ROE == coc` is the self-check. `S` is
dropped (recoverable as `1 − self.cdf(a)`). The octet still trails
(`calibration_df.iloc[:, -8:]`). Store on `self.calibration_df`;
`calibrate_distortions` keeps returning `distortion_df` (the receipt).

## Blast radius

- **Code**: `distortion_df` is built in exactly one place
  (`portfolio.py:calibrate_distortions`) and **read nowhere** in the codebase
  (only the `self.distortion_df = None` init). `standard_shape` is set in
  `spectral.py` and read only at `portfolio.py:2727`. Tiny, self-contained.
- **Tests**: none reference `distortion_df` or `standard_shape`. We **add** a
  regression test (none to fix).
- **Docs**: 7 sites. Six are display-only (`calibrate_distortions(...);
  qd(p.distortion_df)`) and just re-render. Two real edits:
  - `5_technical_guides/5_x_distortions.rst:284` reads `distortion_df.iloc[0,2]`
    (was column `P`) → repoint to `calibration_df` (`P` is there now).
  - `2_user_guides/2_x_10mins.rst:1339` prose ("returned in the `param` column…
    last column gives the error") → update to the new columns and point at
    `calibration_df` for the target.

## Phases
Small enough to land in one pass; listed for ordering.

- **Phase 1 — `Distortion.gini_p` (D1).** Rename the attribute + docstrings.
- **Phase 2 — `distortion_df` + `calibration_df` (D2, D3).** Rebuild the frame;
  add `calibration_df`; update the `calibrate_distortions` docstring
  (Returns/Notes) to describe both.
- **Phase 3 — docs + test.** Fix the two doc edits; add the regression test.

## Files
- `src/aggregate/spectral.py` — `standard_shape` → `gini_p` (`_common_init`,
  five subclasses, docstrings).
- `src/aggregate/portfolio.py` — `calibrate_distortions` rebuild;
  `self.calibration_df` init + populate; `calibrate_distortion` may expose
  `param_name` for the row (or derive it inline).
- `docs/5_technical_guides/5_x_distortions.rst` — positional read → `calibration_df`.
- `docs/2_user_guides/2_x_10mins.rst` — prose around the calibration table.
- `tests/test_distortion_quartet.py` (or a new `tests/test_calibrate_distortions.py`)
  — pin the new schema.

## Verification
- **`distortion_df` schema**: index name `'distortion'`, ordered categorical in
  canonical order; columns exactly `['param_name', 'param', 'gini_p', 'area',
  'error']`; 5 rows.
- **Identity**: for each row `area == (gini_p + 1)/2`; `gini_p` matches the
  `p_equiv` row of that distortion's `_compute_stats_df` (`rtol≈1e-6`); `error`
  small (`< 1e-5`).
- **`calibration_df`**: one row, canonical pentagon octet; `ROE ≈ coc`
  (the calibration target); `a` equals the snapped asset level.
- **`gini_p` rename**: `Distortion(...).gini_p` exists; `standard_shape` is gone
  (no stragglers — grep `spectral.py`/`portfolio.py`).
- `uv run pytest` green; `uv run ruff check src` clean. Version bump per the
  standing rule (plan-based code change) → next `1.0.0a*`; CHANGELOG entry noting
  the **breaking** schema/attribute change.

## Notes
- `calibrate_distortions` remains one-point; the index no longer implies a
  batch that does not exist.
- `area = loading + ½` ties to the existing `loading` stat (`∫g − ½`); both are
  views of the same Gini quantity, keeping the vocabulary consistent.
