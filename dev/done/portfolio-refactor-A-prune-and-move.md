# Sub-project A — Pure deletions + PIR move

**Part of the Portfolio refactor.** Master plan: `portfolio-refactor-planning.md` (table of all methods with K/M/D verdicts; cross-cutting design decisions; sub-project sequencing). This file is self-contained for the A iteration.

## Goal

Shrink `portfolio.py` from ~6,133 LOC to ~3,500 LOC by deleting dead code and moving PIR-exhibit machinery to a new `aggregate/extensions/portfolio_pir.py`. No semantic change to the surviving surface.

## Pre-conditions

- Branch `REFACTOR` at the head of recent Stage 1c++ work (last commit on branch should be the P99.9e/stats_df work or its successor).
- 415/415 pytest green.
- Visual `aggregate.extensions.test_suite` HTML report builds cleanly, modulo the pre-existing `Cc.Freq20` ZT-Poisson `brentq` crash.

## Touches

- `aggregate/portfolio.py` (heavy)
- `aggregate/extensions/portfolio_pir.py` (new file)
- `aggregate/extensions/case_studies.py` (import updates — heaviest consumer of moved code)
- `aggregate/extensions/bodoff.py` (light touch: replace one `self.cotvar(...)` line with a manual `exgta_` lookup since `cotvar` is being deleted)
- `tests/` (any test referencing a deleted/moved method)
- `README.rst` (bullet at end)
- `pyproject.toml` (alpha bump if appropriate)

## Deletions (Bucket D — no consumers in docs, PMIR, or aggregate-internal except other Bucket-D methods)

PMIR audit was done in master plan (see counts in the function table); the methods below have zero PMIR-pkg, zero current-notebook, and zero docs/rst references.

- `gradient` — ~196 LOC; the numerical-derivative-of-stats family.
- Non-spectral allocation: `merton_perold`, `cotvar`, `equal_risk_var_tvar`, `equal_risk_epd`. Together ~80 LOC.
- EPD / priority / collateral: `analysis_priority`, `analysis_collateral`, `priority_capital_df`, `priority_analysis_df`, `epd_2_assets`, `assets_2_epd`, plus `_epd_2_assets`, `_assets_2_epd` backing attrs. Together ~250 LOC.
- `var_dict` — keep the method, drop the `kind='epd'` branch (raise `ValueError` for unknown `kind`, or just remove the branch silently). Other `kind` values (`'lower'`, `'upper'`, `'tvar'`) stay.
- UAT trio: `uat`, `uat_differential`, `uat_interpolation_functions`. Together ~100 LOC.
- `collapse` (already flagged deprecated in-code).
- `audits` (diagnostic-only plot; ~30 LOC; easily rerolled by hand if anyone misses it).
- `stat_renamer` (returns `dict('')` — bogus stub).
- Inner branches only on `analyze_distortion`: delete `analyze_distortion_add_comps` helper (~170 LOC) and `analyze_distortion_plots` helper (~300 LOC). Keep `analyze_distortion`'s `add_comps=False` and `plot=False` defaults so existing call sites pass through; the parameters themselves go away in Sub-project D.

## Moves (Bucket M — to `aggregate/extensions/portfolio_pir.py`)

**Pattern to use: free functions** taking `port: Portfolio` as the first arg. Not monkey-patched methods. Call sites in `case_studies.py` become `from aggregate.extensions.portfolio_pir import premium_capital; premium_capital(port, a=…)` instead of `port.premium_capital(a=…)`. Reasons:

1. Clearer at the call site that this is extension code, not core.
2. No surprise method appears on `Portfolio` when `portfolio_pir` is imported.
3. Easier to test in isolation.

If a future tool/utility makes monkey-patching cleaner we can revisit; for now, free functions.

**To move:**

- Premium-capital exhibits: `premium_capital`, `multi_premium_capital`, `accounting_economic_balance_sheet`, `make_all`, `show_enhanced_exhibits`, `set_a_p` (only consumers are the above). The `EX_premium_capital`/`EX_multi_premium_capital`/`EX_accounting_economic_balance_sheet` state attrs that these methods populate: initialize lazily in the free functions, attach to the `Portfolio` instance on the fly via `port.EX_premium_capital = ...` rather than declaring them in `Portfolio.__init__`.
- PIR figures: `profit_segment_plot`, `natural_profit_segment_plot`, `biv_contour_plot` (+ its `density_sample` helper), `twelve_plot` (+ its `short_renamer` helper), `gamma`.
- PIR pricing: `stand_alone_pricing`, `stand_alone_pricing_work`, `calibrate_blends` (+ module-level helpers `check01`, `make_array`, `convex_points` from the bottom of `portfolio.py`).
- Bulk constructors: `from_DataFrame`, `from_Excel`, `from_dict_of_aggs` (all `@staticmethod`s on `Portfolio` today; become module-level functions in `portfolio_pir.py`).
- Renamers: `renamer` (the BIG presentation dict property — ~150 LOC), `premium_capital_renamer` (small dict class attribute).

## Tasks (execution order)

1. **Create `aggregate/extensions/portfolio_pir.py`** with a module docstring explaining its purpose ("PIR-book exhibit machinery and related helpers extracted from Portfolio; free functions taking a Portfolio as first arg") and the necessary imports (`numpy`, `pandas`, `matplotlib.pyplot`, etc., plus `from aggregate.portfolio import Portfolio` if needed for type hints — careful with circular imports; if a circle appears, fall back to `from aggregate import Portfolio` inside function bodies, or no type hint).

2. **Move each Bucket-M function.** Convert `def method(self, ...)` to `def method(port, ...)`. Update internal `self.` references to `port.`. Keep numerical behavior identical — this is mechanical translation, no logic changes. Order them in the new file roughly by use category (premium-capital exhibits, figures, stand-alone pricing, blends, constructors, renamers).

3. **Audit `case_studies.py` for call sites.** Use `rg "\\.(premium_capital|multi_premium_capital|accounting_economic_balance_sheet|make_all|show_enhanced_exhibits|set_a_p|profit_segment_plot|natural_profit_segment_plot|biv_contour_plot|density_sample|twelve_plot|short_renamer|gamma|stand_alone_pricing(_work)?|calibrate_blends|from_DataFrame|from_Excel|from_dict_of_aggs)" aggregate/extensions/case_studies.py -n`. For each call site, change `obj.method(...)` to `method(obj, ...)` with the appropriate import.

4. **Update `bodoff.py`:** line 40 is `basic.loc['coTVaR'] = self.cotvar(pt)`. Since `cotvar` is being deleted, replace with the equivalent manual lookup: `basic.loc['coTVaR'] = self.density_df.loc[self.q(pt), [f'exgta_{l}' for l in self.line_names_ex]].values`. (That's literally what `cotvar` did; just inlined.)

5. **Delete Bucket-D methods and attrs from `portfolio.py`.** Be careful with shared infrastructure:
   - `gradient` deletion: doesn't affect `np.gradient` — different function. Keep the `numpy as np` import; spot-check `rg "np\\.gradient" aggregate/portfolio.py` returns nothing after deletion.
   - `epd_2_assets`/`assets_2_epd` deletion: removes the backing attrs `_epd_2_assets`/`_assets_2_epd` from `__init__` too. Also remove the `_x = self.assets_2_epd` lazy-build line in `apply_distortion(efficient=False)`. After EPD goes, `efficient=False` may collapse to a no-op or be dropped entirely — but defer that to Sub-project D where `apply_distortion` is rewritten.
   - `var_dict(kind='epd')`: the `elif kind=='epd':` block goes (lines around 3425-3429). Other branches stay.
   - `analysis_priority`, `analysis_collateral`, `priority_capital_df`: their consumers (`equal_risk_epd`, `analysis_*`, themselves) are all going, so the deletion is hermetic.
   - `analyze_distortion_add_comps`, `analyze_distortion_plots`: delete the bodies. Leave the `add_comps`/`plot` parameters on `analyze_distortion` with `False` defaults so PMIR's `analyze_distortion(p=..., add_comps=False)` call sites still parse. The parameters themselves disappear in Sub-project D.

6. **Test pass.** `uv run pytest`. Touch any test that references a deleted/moved method. Expected: most tests don't reference these (they're operating on the surviving surface), so few touches needed.

7. **Visual smoke.** `uv run python -m aggregate.extensions.test_suite`. HTML report should build. Compare to the pre-A capture if you have one.

8. **README + version bump.** Add the bullet (below) to the current alpha-version block in `README.rst`. Bump `pyproject.toml` to the next alpha (e.g., `1.0.0a4` → `1.0.0a5`). Per `feedback_readme_after_iteration`, don't defer this.

## Verification

- `uv run pytest` → 415 + 3 PEG regression tests passed (Sub-project 0 added the regression test before A).
- **PEG regression check:** `uv run pytest tests/test_portfolio_peg_regression.py -v` → 3 passed. Numerical baseline reproduced exactly.
- `uv run python -m aggregate.extensions.test_suite` builds the HTML report (known exception: `Cc.Freq20` ZT-Poisson `brentq` crash — pre-existing).
- Confirm `case_studies.py` constructs and runs a small case: `from aggregate.extensions import case_studies as cs; cs.CaseStudy()` succeeds.
- Confirm PMIR is unaffected: verified in master plan; no Bucket-M function is referenced from `pmir_package/src/`.
- Quick eyeball: `wc -l aggregate/portfolio.py` shows ~3,500 LOC (down from 6,133).

## Commit

One commit at the end of the sub-project, after human review. Do not commit unless the user explicitly says so.

**Commit message draft** (heredoc with the Co-Authored-By footer per repo convention):

```
Portfolio refactor A: prune dead code + extract PIR exhibits

- Delete gradient, non-spectral allocations (merton_perold, cotvar,
  equal_risk_var_tvar, equal_risk_epd), EPD/priority/collateral family
  (analysis_priority, analysis_collateral, priority_capital_df,
  priority_analysis_df, epd_2_assets, assets_2_epd), uat trio, collapse,
  audits, stat_renamer, var_dict(kind='epd') branch. ~700 LOC.
- Strip analyze_distortion_add_comps and analyze_distortion_plots bodies
  (~470 LOC). Parameters retained as no-op defaults for now.
- Move PIR-exhibit machinery to aggregate/extensions/portfolio_pir.py:
  premium_capital + AEBS family, profit_segment plots, biv_contour,
  twelve_plot + short_renamer, gamma, stand_alone_pricing,
  calibrate_blends + module helpers, from_DataFrame/Excel/dict_of_aggs,
  BIG renamer + premium_capital_renamer. ~1,800 LOC.
- portfolio.py shrinks from 6,133 → ~3,500 LOC.

415/415 pytest green.
```

## README bullet draft

```rst
- ``Portfolio`` refactor pass A: pruned ~700 LOC of dead code (gradient,
  non-spectral allocations, EPD/priority family, uat trio, collapse,
  audits) and moved ~1,800 LOC of PIR-exhibit machinery to
  ``aggregate.extensions.portfolio_pir`` (premium_capital + AEBS family,
  profit_segment / natural / biv_contour plots, twelve_plot +
  short_renamer, gamma, stand_alone_pricing, calibrate_blends,
  from_DataFrame/Excel/dict_of_aggs). ``portfolio.py`` shrinks from 6,133
  → ~3,500 LOC.
```

## Post-conditions

Portfolio is roughly half its starting size. Subsequent sub-projects land cleaner diffs against this leaner base. Sub-project B (approximation + tilting drop) is unblocked.

## Cross-cutting reminders

- `UV_LINK_MODE=copy` is set in `.claude/settings.local.json`. Don't override.
- Run `uv run pytest`, not bare `pytest`.
- PowerShell shell: no awk/sed; use `rg` directly via Bash (per `feedback_use_rg`, `feedback_powershell_no_unix`).
- Don't prefix Bash commands with `cd T:/...` (per `feedback_no_cd_prefix`) — shell already starts at the project root.
- Update `README.rst` at the close of this sub-project (per `feedback_readme_after_iteration`). Bump alpha version.
- Don't build docs as part of a verification cycle (per `CLAUDE.md`). Doc audits via `rg` are fine.
- Do NOT commit unless the user explicitly says so.
- The visual `test_suite` has one pre-existing failure (`Cc.Freq20` ZT-Poisson `brentq` crash). Ignore unless a new failure appears.
