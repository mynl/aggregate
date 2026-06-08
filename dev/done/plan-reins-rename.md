# Plan: rename public `reinsurance_*` methods → `reins_*` (surface consistency)

## Goal

Make the **public reinsurance surface uniformly abbreviated** (`reins_`). The
codebase already uses `reins_` everywhere — reporting attributes
(`reins_describe`, `reins_density_df`, `reins_stats_df`, `reins_audit_df`,
`reins_df`), the layer attributes (`occ_reins`, `agg_reins`), the config key
(`reins_bucket`), the parser/Lark names — and sits in a deliberately
abbreviation-friendly house style (`sev`, `occ`, `agg`, `freq`, `cv`, `bs`,
`log2`). The **only** spelled-out public symbols are three `Aggregate` methods.
Bring them into line. This is the inverse of the (deleted) big "spell-everything-
out" plan: we canonize `reins` rather than fight it, for a one-file diff with no
behaviour, no spec-key, no grammar, no config, and no snapshot exposure.

**Decision recorded:** `reins` is the accepted canonical short form for
"reinsurance" in identifiers. Do not flip this back.

## Scope (settled)

- **Rename exactly three `Aggregate` methods** (all in `distributions.py`):
  | Current | → |
  |---|---|
  | `reinsurance_occ_plot` | `reins_occ_plot` |
  | `reinsurance_description` | `reins_description` |
  | `reinsurance_kinds` | `reins_kinds` |
- **Hard rename, no deprecation alias** — consistent with every other rename
  a32–a40, and there is **zero test usage** to cushion (so nothing regresses).
- Update the five internal call sites (`describe`/info report + a validation
  check, all in `distributions.py`).
- Update the doc references (RST source only — never `docs/_build/`).

### Out of scope (do NOT touch)

- All `reins_*` symbols already abbreviated (no change needed — that's the point).
- `occ` / `agg` abbreviations.
- The DecL grammar, parser transformer methods, spec keys, config/env keys,
  `REINSURANCE*` constants — none are involved.
- The legacy migration `.. note::` in `2_x_re_pricing.rst` (lines ~210–216),
  which references *removed* by-layer names (`reinsurance_audit_df`,
  `reinsurance_occ_layer_df`, `reinsurance_df`, `reinsurance_report_df`). Those
  are historical, not the three methods. **Flag separately** — that note looks
  stale (it lists `reinsurance_audit_df` as removed, but `reins_audit_df` is a
  live symbol) — but fixing it is a different task; leave it untouched here.

## Exact edits

**`src/aggregate/distributions.py`** (8 occurrences):

| Line | Change |
|---|---|
| 2265 | `def reinsurance_occ_plot(` → `def reins_occ_plot(` |
| 2271 | warning string `'reinsurance_occ_plot called …'` → `'reins_occ_plot called …'` |
| 3728 | `self.reinsurance_kinds()` → `self.reins_kinds()` |
| 3729 | `self.reinsurance_description("occ")` → `self.reins_description("occ")` |
| 3730 | `self.reinsurance_description("agg")` → `self.reins_description("agg")` |
| 4799 | `self.reinsurance_kinds()` → `self.reins_kinds()` |
| 5154 | `def reinsurance_description(` → `def reins_description(` |
| 5202 | `def reinsurance_kinds(` → `def reins_kinds(` |

**`docs/2_user_guides/2_x_re_pricing.rst`** (6 occurrences): lines 201, 202, 203
(`:meth:` refs) and 235, 236, 244 (`a.<method>()` ipython calls) —
`reinsurance_kinds/description/occ_plot` → `reins_kinds/description/occ_plot`.

**Reference page** `docs/3_reference/3_x_Distribution.rst` uses autodoc
`:members:`, so the renamed methods re-document automatically on the next build —
no manual edit. (Docs are NOT rebuilt in the verification loop per CLAUDE.md;
note pending rebuild.)

## Verification

1. `uv run pytest` — full suite green. The `describe`/info path and the
   validation flag call the renamed methods, so existing describe/validation
   tests exercise them indirectly.
2. Smoke: build an agg with occ+agg reinsurance; assert `a.reins_kinds()`,
   `a.reins_description()` return text and `a.reins_occ_plot()` runs; assert the
   old names raise `AttributeError`.
3. Grep guard: `rg "reinsurance_(kinds|description|occ_plot)" src/ docs/`
   (excluding `docs/_build/`) returns **nothing**.

## Housekeeping (standing rules)

- Bump `pyproject.toml` `1.0.0a40 → a41`.
- `CHANGELOG.md`: new section — public `reinsurance_*` methods → `reins_*`
  (breaking, no alias; surface-consistency).
- `dev/TODO.md`: one-line under Track **H** (Hygiene), marked done with the version.
- Move this plan to `dev/done/plan-reins-rename.md` on landing.
- No `test_decl.agg` change (no DecL programs added).
