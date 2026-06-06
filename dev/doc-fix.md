# doc-fix — executed-cell errors in the docs build

> Catalogue of `*Error` failures surfaced in the rendered docs (text build at
> `T:\doc-diff\agg-doc-diff\text`, HTML at `docs/_build/html`). Grouped by
> **fundamental cause**, most-impactful first, with the proposed fix or
> **fix unclear**. Line refs are into the built `.txt` (locate the matching cell
> in the doc *source* under `docs/…` — `.rst` / notebook — to edit). ~55 raised
> errors collapse to ~11 root causes; many later errors in a page are
> **cascades** of the first failed cell.

Captured 2026-06-05 against 1.0.0a30.

## Status (2026-06-06) — first fix pass

**Fixed in doc source** (no version bump; the author committed first so the diff
is clean): **G1, G3, G9, G10**. Each new expression was validated at runtime.

- **G1 ✅** — every `density_df.<sev col>` → `sev_density_df`; the three
  mixed agg+sev selections split across the two frames (`density_df` for
  `p/F/S`, `sev_density_df` joined/reindexed for `p_sev/F_sev/S_sev`). Includes
  `2_x_re_pricing.rst:1167-1170` which is the same root cause but did **not**
  surface as a captured traceback (a `@savefig` cell).
- **G3 ✅** — `ans.comp_df` → `ans.pricing_df`;
  `a2.pricing.unstack(1).droplevel(0, axis=0).T` → `a2.pricing_df.T` (verified
  this yields the intended stat×line table).
- **G9 ✅** — restored `fp`/`qdl` in the first setup cell of `0x0_other_misc.rst`
  (`qd` passes `**kwargs` to `to_string`, so `index/line_width/formatters` work).
- **G10 ✅** — `5_x_pk.rst` `PZTest` build: `'poisson'` → `'poisson '` (the
  missing space that concatenated to `poissonagg`). **This also clears the G11
  cascades** `'Limit1' is not in list` and `KeyError: 'A'` (the port now builds
  both units).

**Pended (per request / not clear):**
- **G2** — `tilt_vector` removed → rewrite under TODO **F2**.
- **G4** — `stand_alone_pricing` removed, no in-code replacement (fix unclear).
- **G6** — mixed-severity `comp_*` columns / `('meta','name')` keys (fix unclear).
- **G7 / G8** — `describe`/reins length-mismatch + `Gross`→`Subject` labels:
  **assessed not-clear** — entangled with the undecided **F4** relabel and the
  **D7** case-study rebuild (and the bahnemann `KeyError: 100000` rows are
  cascades). Pended.
- **G11 remainder** — `KeyError: 'Est'` (loss_data_analytics) still open; the
  `gh_example` str+float bug folds into the G2/F2 rewrite.

Per-group detail below is the original catalogue; the ✅ groups are now done.

---

## G1 — severity columns moved to `sev_density_df`  ✅ easy
**Cause.** `p_sev`, `F_sev`, `S_sev`, `log_p_sev` no longer live on
`density_df`; they moved to their own native grid in `sev_density_df`
(`distributions.py:2122`, cols `p_sev / F_sev / S_sev`).
**Fix.** Replace `a.density_df.F_sev` / `a.density_df[['p_sev','F_sev','S_sev']]`
→ `a.sev_density_df.F_sev` / `a.sev_density_df[[...]]`. (The user's first find,
`a03.density_df.F_sev`, is exactly this.)
**Instances (~10):** `2_x_cat.txt:303, 407, 594, 756`; `2_x_10mins.txt:970,
1014`; `problems/0x0_loss_models.txt:1113, 1201`; `5_technical_guides/5_x_pk.txt:246`.

## G2 — `tilt_vector` / FFT tilting removed  ⛓ ties to TODO **F2**
**Cause.** The Grübel–Hermesmeier exponential tilt was removed (commit
`6de20f2`): `Aggregate.update_work(..., tilt_vector=)` and the `tilt_vector`
density column no longer exist. The whole `010_gh_example` page is the old
tilting demo; its later errors (`ValueError: Update Aggregate before asking for
density_df`, the length-mismatch, `IndexError out-of-bounds`) are **cascades**
of the failed `update(tilt_vector=…)`.
**Fix.** Rewrite the tilting examples as the DIY illustration scoped in TODO
**F2** (or delete them) — not a mechanical swap. **Fix pending F2.**
**Instances (~6 root + cascades):** `problems/010_gh_example.txt:43, 153`
(+ cascades `60, 101, 128, 191`); `5_technical_guides/5_x_nm_discrete_rep.txt:589`;
`5_technical_guides/5_x_numerical_methods.txt:984`.

## G3 — pricing result-object attribute renames  ✅ easy
**Cause.** `analyze_distortions` → `AnalyzeDistortionsResult` with `pricing_df`
(was `comp_df`); `analyze_distortion` → `AnalyzeDistortionResult` with
`pricing_df` (was `pricing`) (`results.py`).
**Fix.** `ans.comp_df` → `ans.pricing_df`; `a2.pricing` → `a2.pricing_df`. The
downstream `.xs('LR', axis=0, level=1)` still applies (rows are
`(distortion, stat)`).
**Instances (~3):** `2_x_10mins.txt:2274` (`comp_df`), `2473, 2566` (`pricing`).

## G4 — `Portfolio.stand_alone_pricing` removed  ❓ fix unclear
**Cause.** Method deleted; **no in-code replacement**. Downstream
`a.iloc[:8]` then fails because `a` keeps an earlier `Aggregate` binding
(`'Aggregate' object has no attribute 'iloc'` — cascade).
**Fix.** **Fix unclear** — needs an example rewrite. Likely re-expressed via
`price` / `analyze_distortions` per unit, but not a 1:1 swap; confirm the
intended stand-alone (vs allocated) pricing path first.
**Instances (~2 + cascades):** `2_x_10mins.txt:2449, 2546` (+ `.iloc` cascades
`2457, 2554`).

## G5 — `twelve_plot` → `pedagogy.plot_twelve`, needs `distortion_name`  ✅ easy-ish
**Cause.** `Portfolio.twelve_plot` removed; the figure now lives at
`aggregate.pedagogy.plot_twelve(port, fig, axs, distortion_name, …)`
(`pedagogy.py:1269`, provenance note confirms the rename).
**Fix.** `from aggregate.pedagogy import plot_twelve` and call with the required
`distortion_name` (and the `fig, axs` it expects).
**Instances (~2):** `2_x_10mins.txt:2532` (`twelve_plot`), `2342`
(`plot_twelve() missing … 'distortion_name'`).

## G6 — mixed-severity component columns / spec keys renamed  ❓ fix unclear
**Cause.** Mixed-severity `stats_df` no longer has per-component columns
`comp_0, comp_1, …`, and the spec/knowledge access by `'name'` /
`('meta','name')` changed shape.
**Fix.** **Fix unclear** — determine the current mixed-severity component
column scheme and spec-dict keys, then update the readouts. (Investigate
alongside the mixed-severity work.)
**Instances (~10):** `2_x_ir_pricing.txt:168` & `2_x_re_pricing.txt:1989`
(`comp_0…comp_3`); `DecL/060_mixed_severity.txt:627, 642, 654, 708, 734, 749,
761, 815` (`KeyError 'name'` / `('meta','name')`).

## G7 — `df.columns = [...]` length mismatch (describe/reins shape changed)  ⛓ ties to **D7/F4**
**Cause.** `reins_stats_df` / `describe` now return a different column count, so
doc cells that hard-assign a fixed column-name list raise `Length mismatch`.
**Fix.** Update the hard-coded column lists to the current describe output;
coordinate with the reins-describe relabel (TODO **F4** Gross→Subject) and the
case-study rebuild (**D7**).
**Instances (~2):** `problems/010_gh_example.txt:101` (5 vs 1, also a G2
cascade); `problems/0x0_bahnemann.txt:831` (11 vs 7).

## G8 — reins describe label rename (`Gross`→`Subject`)  ⛓ ties to **F4/D7**
**Cause.** Example expects `'Subject EX' / 'Subject CV' / 'Subject Sk'` columns
that the current `describe` doesn't emit (label set changed).
**Fix.** Align with current describe labels (decide F4 first). The subsequent
`KeyError: 100000` rows are **cascades** (a `.loc[100000]` after the failed
build).
**Instances (~1 + cascades):** `problems/0x0_bahnemann.txt:1039`
(+ cascades `857, 896, 912, 951`).

## G9 — `NameError` cascades + removed `qdl` helper  ⚠ mostly cascade
**Cause.** Downstream cells reference names a failed earlier cell never bound
(`ilw`, `params`, `a`). Separately, `qdl(...)` is a **removed** display helper
(not in the codebase).
**Fix.** Most resolve once the upstream root cell is fixed. For `qdl`: replace
with `qd` (confirm it was the long/list variant of `qd`).
**Instances (~6):** `2_x_cat.txt:777, 805` (`ilw`), `813, 821` (`params`);
`problems/0x0_other_misc.txt:111, 130` (`qdl`).

## G10 — DecL parse error `poissonagg`  ✅ easy
**Cause.** A doc DecL program reads `…poissonagg…` (missing space / stale
syntax): `ValueError: DecL parse error … Unexpected 'poissonagg'`.
**Fix.** Correct the program text (separate `poisson` from the next `agg`
clause); re-check the source cell.
**Instances (1):** `5_technical_guides/5_x_pk.txt:405`.

## G11 — assorted one-offs  ❓ investigate per doc
Individual column/label renames or example bugs, several themselves cascades of
G2/G7:
- `5_x_pk.txt:462` `ValueError: 'Limit1' is not in list` — a removed/renamed
  layer label. **fix unclear.**
- `5_x_pk.txt:552` `KeyError: 'A'` — stale index/column key. **fix unclear.**
- `problems/0x0_loss_data_analytics.txt:719, 803` `KeyError: 'Est'` — renamed
  output column. **fix unclear** (find the new name).
- `problems/010_gh_example.txt:128` `TypeError: can only concatenate str (not
  "float") to str` — an f-string/format bug in the (G2) example; folds into the
  F2 rewrite.

---

## Suggested order of attack
1. **G1, G3, G5, G10** — mechanical, do now (no design decisions).
2. **G9 `qdl`→`qd`** and re-run pages to clear NameError cascades.
3. **G7/G8** — after the reins-describe label decision (F4), then rebuild the
   case studies (D7).
4. **G2** — fold into the F2 tilting-DIY rewrite.
5. **G4, G6, G11** — **fix unclear**: need an API-mapping decision
   (`stand_alone_pricing` replacement; mixed-severity component scheme) before
   editing.

**Note.** Edits land in the doc *source* (`docs/…` `.rst` / notebooks); the
`.txt`/HTML are build artefacts. Don't rebuild in the iteration loop (per
CLAUDE.md) — the author rebuilds manually.
