# Portfolio refactor planning

This is the master plan. Five companion files break the work into iteration-sized sub-projects, each self-contained and readable cold (after a compact). Iterate one at a time, compacting between sub-projects.

| sub | file                                                       | scope                                          |
|:---:|:-----------------------------------------------------------|:-----------------------------------------------|
|  0  | `portfolio-refactor-0-baseline.md`                         | Pre-flight: PEG regression baseline + test     |
|  A  | `portfolio-refactor-A-prune-and-move.md`                   | Pure deletions + move PIR exhibits             |
|  B  | `portfolio-refactor-B-approx-tilt.md`                      | Drop approximation + tilting paths             |
|  C  | `portfolio-refactor-C-distortion-calibration.md`           | Distortion owns its own calibration            |
|  D  | `portfolio-refactor-D-pricing-redesign.md`                 | Cache, analyze collapse, dataclasses, renames  |
|  E  | `portfolio-refactor-E-stats-consolidation.md`              | `stats_df` mirror of Aggregate Stage 1c+       |

**Sub-project 0** captures a numerical baseline for a two-unit Portfolio (`PEG`) before any refactoring begins. The baseline is pinned as JSON in `tests/data/peg_baseline.json` and verified by `tests/test_portfolio_peg_regression.py`. Every subsequent sub-project (A–E) must keep the regression test green. Sub-project D updates the test to use the new API after the rename but the baseline numbers don't change.

The rest of this file is the canonical reference: the per-method table with KMD verdicts, the design decisions, and the surfaces table. Read it once for context, then work the sub-project files.

Counts are rough sums of: docs `.rst` references + PMIR package references + `aggregate/`-internal references. **KMD** = **K**eep in core, **M**ove (to `extensions/portfolio_pir.py` unless noted), **D**elete.

| function                             | calls |   KMD    | notes and determination                                                                                          |
|:-------------------------------------|------:|:--------:|:-----------------------------------------------------------------------------------------------------------------|
| EX_accounting_economic_balance_sheet |     0 |    M     | state for the AEBS exhibit                                                                                       |
| EX_multi_premium_capital             |     0 |    M     | state for multi_premium_capital exhibit                                                                          |
| EX_premium_capital                   |     0 |    M     | state for premium_capital exhibit                                                                                |
| accounting_economic_balance_sheet    |     0 |    M     | PIR balance-sheet exhibit                                                                                        |
| add_eta_mu                           |     1 |    K     | FFT internal; PMIR pokes it directly                                                                             |
| add_exa                              |    ~5 |    K     | core FFT path — populates exeqa/exlea/exgta/exa columns                                                          |
| add_exa_details                      |    ~3 |    K     | extra columns after add_exa; the `eta_mu=True` branch (which built EPD splines) simplifies when EPD family goes  |
| add_exa_sample                       |     1 |    K     | switcheroo internal; called by create_from_sample                                                                |
| agg_cv                               |     0 |    K     | scalar mirror of stats_df, set in __init__                                                                       |
| agg_list                             |    ~5 |    K     | list of constituent Aggregate objects                                                                            |
| agg_m                                |     5 |    K     | scalar mirror; PMIR uses for LR setup                                                                            |
| agg_sd                               |     0 |    K     | scalar mirror                                                                                                    |
| agg_skew                             |     0 |    K     | scalar mirror                                                                                                    |
| agg_var                              |     0 |    K     | scalar mirror                                                                                                    |
| analysis_collateral                  |     0 |    D     | EPD collateral exhibit; non-spectral                                                                             |
| analysis_priority                    |     0 |    D     | EPD priority exhibit; non-spectral                                                                               |
| analyze_distortion                   |     7 |    K     | **redesign**: drop add_comps + plots; single-distortion, single-p; calls `apply_distortion`                      |
| analyze_distortion_add_comps         |     0 |    D     | non-spectral comparison rows; PMIR always passes add_comps=False                                                 |
| analyze_distortion_plots             |     0 |    D     | ~300 LOC of plot output; never invoked                                                                           |
| analyze_distortions                  |     8 |    K     | **replace with `analyze_distortions2` API**: single `p`, optional `distortions` dict                             |
| analyze_distortions2                 |     1 |    K     | this becomes THE `analyze_distortions`; existing list-based version gone                                         |
| apply_distortion                     |     7 |    K     | **redesign**: lazy-eval cache; method takes a distortion and returns the augmented_df                            |
| apply_distortions                    |     2 |    D     | absorbed into `analyze_distortions` after cache redesign (no longer needs a separate plural runner)              |
| approx_freq_ge                       |     0 |    D     | auto-fallback threshold; goes with the whole approximation path (reaches into Aggregate.update_work)             |
| approx_type                          |     0 |    D     | auto-fallback slognorm/sgamma toggle; goes with the approximation path                                           |
| approximate                          |    17 |    K     | on-demand method-of-moments fit (user-facing); separate machinery, stays                                         |
| as_severity                          |    ~1 |    K     | wrap Portfolio as Severity for nesting                                                                           |
| assets_2_epd                         |     0 |    D     | only consumer is the now-dead EPD family; confirmed                                                              |
| audit_df                             |    ~5 |   D-P3   | Phase 3: subsumed by stats_df (mirror Aggregate Stage 1c+)                                                       |
| audit_percentiles                    |     0 |    K     | config (percentile list for make_audit_df)                                                                       |
| audits                               |     0 |    D     | only exeqa-sum-error diagnostic; 5-line manual reroll                                                            |
| augmented_df                         |    ~6 |    K     | **redesign**: lazy-eval method `augmented_df(distortion)` over an `_augmented_dfs` cache; see Design 1           |
| best_bucket                          |     4 |    K     | rounded-bucket recommender; used in update                                                                       |
| biv_contour_plot                     |     0 |    M     | bivariate density contour — book figure                                                                          |
| bodoff                               |     1 |    K     | Bodoff layer allocation; dedicated doc page; ~25 LOC                                                             |
| bs                                   |     - |    K     | config (bucket size) — central FFT param                                                                         |
| calibrate_blends                     |     0 |    M     | IME-paper blend distortion; case_studies consumer                                                                |
| calibrate_distortion                 |     7 |    K     | **redesign**: thin wrapper; the per-name Newton iterations move to `Distortion` subclasses                       |
| calibrate_distortions                |    13 |    K     | **redesign**: replace with `calibrate_distortions2` API: single `coc`, single `p`, calibrates `self.distortions` |
| calibrate_distortions2               |     1 |    K     | this becomes THE `calibrate_distortions`; existing LRs/COCs/ROEs/As/Ps lists gone                                |
| cdf                                  |    60 |    K     | distribution function                                                                                            |
| collapse                             |     0 |    D     | flagged deprecated already                                                                                       |
| cotvar                               |     0 |    D     | non-spectral co-TVaR allocation                                                                                  |
| create_from_sample                   |     7 |    K     | switcheroo entry point; PMIR-critical                                                                            |
| density_df                           |   50+ |    K     | THE core output DataFrame                                                                                        |
| density_sample                       |     0 |    M     | helper for biv_contour_plot; moves with it                                                                       |
| describe                             |    22 |    K     | theoretical-vs-empirical moment table                                                                            |
| discretization_calc                  |     0 |    K     | config (survival vs distribution)                                                                                |
| dist_ans                             |    ~2 | K-rename | **rename → `distortion_df`** (drop some columns); see Renames                                                    |
| distortion                           |    ~3 |    K     | property returning _distortion (last-applied)                                                                    |
| distortion_df                        |    12 | K-rename | the new name for renamed/trimmed `dist_ans`                                                                      |
| dists                                |    22 | K-rename | **rename → `distortions`** (dict of calibrated distortions)                                                      |
| epd_2_assets                         |     0 |    D     | only consumer is the now-dead EPD family; confirmed                                                              |
| equal_risk_epd                       |     0 |    D     | non-spectral allocation                                                                                          |
| equal_risk_var_tvar                  |     0 |    D     | non-spectral allocation                                                                                          |
| est_cv                               |     0 |    K     | empirical CV, set in update                                                                                      |
| est_m                                |    ~2 |    K     | empirical mean, set in update                                                                                    |
| est_sd                               |     0 |    K     | empirical sd, set in update                                                                                      |
| est_skew                             |     0 |    K     | empirical skew, set in update                                                                                    |
| est_var                              |     0 |    K     | empirical var, set in update                                                                                     |
| ex                                   |     0 |    K     | alias for empirical mean; set in update                                                                          |
| explain_validation                   |    ~3 |    K     | summary of validation flags                                                                                      |
| figure                               |     0 |    K     | set by plot for figure handle                                                                                    |
| from_DataFrame                       |     1 |    M     | lightweight; move and reconsider later if needed                                                                 |
| from_Excel                           |     0 |    M     | lightweight                                                                                                      |
| from_dict_of_aggs                    |     0 |    M     | lightweight                                                                                                      |
| ft                                   |    ~3 |    K     | FFT wrapper with padding/tilt; used by add_exa                                                                   |
| gamma                                |     2 |    M     | coverage-effectiveness (~150 LOC, mixed num+plot); no PMIR use                                                   |
| gradient                             |     0 |    D     | ~196 LOC; flagged for removal                                                                                    |
| hash_rep_at_last_update              |     0 |    K     | dedup guard in update()                                                                                          |
| help                                 |    ~1 |    K     | regex lookup over methods/properties                                                                             |
| ift                                  |    ~3 |    K     | inverse FFT wrapper (sibling to ft)                                                                              |
| independent_audit_df                 |     0 |    K     | switcheroo state — pre-switch snapshot                                                                           |
| independent_density_df               |    ~2 |    K     | switcheroo state — pre-switch snapshot                                                                           |
| info                                 |     2 |    K     | one-screen object summary                                                                                        |
| json                                 |    ~1 |    K     | serialization; used by save()                                                                                    |
| last_update                          |     0 |    K     | timestamp of last update()                                                                                       |
| limits                               |    ~2 | K-rename | **→ `_limits` (private, like Aggregate)**                                                                        |
| line_name_pipe                       |     0 |    K     | regex-friendly `\|`-joined names (internal)                                                                      |
| line_names                           |   20+ |    K     | the canonical unit list                                                                                          |
| line_names_ex                        |   15+ |    K     | line_names + ['total']                                                                                           |
| line_renamer                         |    ~5 |    K     | LaTeX-style line-name pretty-printer                                                                             |
| log2                                 |     - |    K     | config (FFT log2 grid size)                                                                                      |
| make_all                             |     1 |    M     | orchestrates premium_capital + multi + AEBS exhibits                                                             |
| make_audit_df                        |     2 | K→D-P3   | needed until Phase 3 stats_df lands                                                                              |
| make_comonotonic_allocations         |     0 |    K     | useful Denuit's algorithm tool                                                                                   |
| merton_perold                        |     0 |    D     | non-spectral allocation                                                                                          |
| multi_premium_capital                |     0 |    M     | multi-asset version of premium_capital                                                                           |
| n_units                              |     2 |    K     | len(line_names)                                                                                                  |
| name                                 |    ~5 |    K     | portfolio name                                                                                                   |
| natural_profit_segment_plot          |     0 |    M     | PIR figure                                                                                                       |
| nice_program                         |    ~1 |    K     | wrap+pretty-print the parser program                                                                             |
| normalize                            |     0 |    K     | config (normalize severities)                                                                                    |
| padding                              |     - |    K     | config (FFT padding factor)                                                                                      |
| pdf                                  |    22 |    K     | probability density (continuous interp)                                                                          |
| percentiles                          |    ~1 |    K     | percentile table by line; useful diagnostic                                                                      |
| plot                                 |   144 |    K     | default density+log-density plot                                                                                 |
| pmf                                  |    16 |    K     | probability mass (discrete)                                                                                      |
| pprogram                             |    ~1 |    K     | pretty-printed parser program                                                                                    |
| pprogram_html                        |    ~1 |    K     | HTML version                                                                                                     |
| premium_capital                      |     3 |    M     | PIR pricing-summary exhibit                                                                                      |
| premium_capital_renamer              |     0 |    M     | rename dict for AEBS/premium-capital exhibits                                                                    |
| price                                |    13 |    K     | top-level pricing entry; **single `p` only**                                                                     |
| price_ccoc                           |     0 |    K     | constant-CoC convenience wrapper; ~15 LOC                                                                        |
| pricing_bounds                       |     1 |    K     | IME 2022 paper bounds                                                                                            |
| priority_analysis_df                 |     0 |    D     | only used by analysis_priority                                                                                   |
| priority_capital_df                  |     0 |    D     | EPD priority spline scaffolding                                                                                  |
| profit_segment_plot                  |     0 |    M     | PIR figure                                                                                                       |
| program                              |    ~3 |    K     | source DecL program string (set by parser)                                                                       |
| q                                    |    90 |    K     | quantile function                                                                                                |
| recommend_bucket                     |     6 |    K     | per-line bucket recommendation table                                                                             |
| remove_fuzz                          |    ~3 |    K     | post-FFT noise scrub                                                                                             |
| renamer                              |    ~5 |    M     | the BIG presentation dict; only consumers are moved/deleted plot helpers                                         |
| report                               |     0 |   D-P3   | Phase 3                                                                                                          |
| report_df                            |     0 |   D-P3   | Phase 3 — replaced by stats_df                                                                                   |
| sample                               |     6 |    K     | multivariate sample with optional Iman-Conover                                                                   |
| sample_compare                       |     0 |    K     | keep for now — small but useful                                                                                  |
| sample_density_compare               |     0 |    K     | keep for now — small but useful                                                                                  |
| sample_df                            |     - |    K     | input attribute when Portfolio built from a DataFrame                                                            |
| save                                 |    ~1 |    K     | persist json to file                                                                                             |
| scatter                              |     0 |    K     | pandas scatter_matrix on exeqa cols; harmless to keep                                                            |
| set_a_p                              |     3 |    M     | argument-coalescer for assets/p; only consumers are the PIR exhibits                                             |
| sev_calc                             |     0 |    K     | config (discrete/continuous)                                                                                     |
| sf                                   |    40 |    K     | survival function                                                                                                |
| short_renamer                        |    ~3 |    M     | only consumer is twelve_plot; moves with it                                                                      |
| show_enhanced_exhibits               |     0 |    M     | iterates EX_* attrs and displays HTML                                                                            |
| snap                                 |     3 |    K     | snap a value to the bs-grid                                                                                      |
| spec                                 |    ~2 |    K     | dict spec (for round-trip)                                                                                       |
| spec_ex                              |    ~1 |    K     | extended spec with bs/log2 metadata                                                                              |
| stand_alone_pricing                  |     2 |    M     | PIR exhibit; the "2 doc hits" are PMIR's own same-named method                                                   |
| stand_alone_pricing_work             |     2 |    M     | engine for stand_alone_pricing                                                                                   |
| stat_renamer                         |     0 |    D     | returns `dict('')` — bogus stub                                                                                  |
| statistics                           |     2 |   D-P3   | Phase 3 — alias for statistics_df                                                                                |
| statistics_df                        |   ~10 |   D-P3   | Phase 3                                                                                                          |
| swap_density_df                      |    ~1 |    K     | the switcheroo core; called by create_from_sample                                                                |
| tilt_amount                          |     0 |    D     | tilting goes away early — reaches into Aggregate.update_work                                                     |
| tm_renamer                           |     1 |    K     | exa→T.L, exag→T.P; needed by analyze_distortion numeric core                                                     |
| trim_df                              |     1 |    K     | drop unwanted columns post-update                                                                                |
| tvar                                 |     9 |    K     | tail value at risk                                                                                               |
| tvar_threshold                       |     2 |    K     | find p such that TVaR(p)=VaR(p₀)                                                                                 |
| twelve_plot                          |     2 |    M     | not generic — moves to extensions                                                                                |
| uat                                  |     0 |    D     | user-acceptance scratchpad                                                                                       |
| uat_differential                     |     0 |    D     | user-acceptance helper                                                                                           |
| uat_interpolation_functions          |     0 |    D     | user-acceptance helper                                                                                           |
| unit_names                           |    13 |    K     | property alias for line_names (PMIR-canonical name)                                                              |
| unit_names_ex                        |    ~3 |    K     | property alias for line_names_ex                                                                                 |
| update                               |    30 |    K     | FFT entry point; **signature shrinks** when approx + tilt paths are dropped (lose `approx_freq_ge`, `approx_type`, `tilt_amount`, `approximation`) |
| valid                                |    ~3 |    K     | validation status                                                                                                |
| validation_eps                       |     0 |    K     | config (validation tolerance)                                                                                    |
| var                                  |    ~2 |    K     | alias for q (VaR)                                                                                                |
| var_dict                             |     6 |    K     | per-line VaR/TVaR dict; **drop kind='epd' branch when epd_2_assets goes**                                        |

## Design decisions

### 1. `apply_distortion` becomes a lazy-eval cache

**Decision.** `apply_distortion(distortion) → DataFrame` becomes a method (no longer dual-purpose). Internally it consults `self._augmented_dfs[distortion.name]`; on cache miss it builds the frame and stores it, then returns. The current side-effect-on-`self` and the wrapping `Answer` both go away.

```
self._augmented_dfs : dict[str, pd.DataFrame]    # cache, keyed by distortion name

# unified entry: lazy-eval, returns the cached frame
def augmented_df(self, distortion) -> pd.DataFrame:
    name = distortion.name if isinstance(distortion, Distortion) else str(distortion)
    if name not in self._augmented_dfs:
        self._augmented_dfs[name] = self._build_augmented(distortion)
    return self._augmented_dfs[name]

# pricing-summary extraction at a single asset (or probability) level — now trivial
def pricing_at(self, p, distortion) -> dict | pd.DataFrame:
    aug = self.augmented_df(distortion)
    a = self.q(p) if p <= 1 else self.snap(p)
    return aug.loc[[a]].filter(regex=...).T  # the row-extraction logic currently in `price` and `analyze_distortion`
```

`apply_distortion(distortion)` may stay as a thin alias for `augmented_df(distortion)` if existing call sites are worth preserving, or be deleted in favor of the property/method.

**Memory math.** 5 distortions × 2^16 rows × ~30 columns × 8 bytes ≈ 80 MB. On a 32 GB machine this is noise. If a future workflow builds a dozen distortions and we want the option to flush, `self._augmented_dfs.clear()` does it; no need to invent eviction logic now.

**Consequence — `apply_distortions` (plural) goes away.** It was the per-distortion runner over a dict. After the cache, `analyze_distortions(p, distortions=None)` just iterates `self.distortions` (or the passed dict) and calls `pricing_at(p, d)` for each — which warms the cache transparently. The single plural entry point becomes superfluous; row dropped to D.

**Consequence — the whole `analyze_distortion(s)` Answer family collapses to trivial DataFrame work.** Once `pricing_at(p, dist)` exists, the rest is concat-and-reshape:

- `analyze_distortion(dist, p)` → `pricing_at(p, dist)` (plus light metadata if needed).
- `analyze_distortions(p, distortions=None)` → for each `d` in `(distortions or self.distortions)`, ensure `augmented_df(d)` (warms cache), then concat `pricing_at(p, d)` results keyed by distortion name.
- `comp_df` (compare-across-distortions) → reshape of the same concat — one row per distortion, columns are the L/LR/M/P/PQ/Q/ROE statistics.
- First call to `analyze_distortions` builds all augmented_dfs (once each). Subsequent calls at different `p` values are cache hits — only the row-extraction runs.

Net effect: the bulky logic in today's `analyze_distortion` body (~250 LOC after the add_comps + plot surgery) reduces to a couple of dozen lines, with the heavy lifting moved into the cache and `pricing_at`. The dataclass return types in §4 wrap thin frames, not big computations.

### 2. EPD-extended exa family is safely deleted

Confirmed grep: only the dead Bucket-D methods consume `epd_2_assets`/`assets_2_epd` (`analysis_priority`, `analysis_collateral`, `priority_capital_df`, `equal_risk_epd`, `var_dict(kind='epd')`, `analyze_distortion_add_comps`). Neither Bounds nor twelve_plot touches them. PMIR doesn't touch them. The one tangent: `apply_distortion(efficient=False)` lazily *builds* the EPD splines as a side effect but doesn't consume the build; that branch collapses for free with the EPD removal. The `efficient` parameter may also simplify or disappear.

### 3. Distortion owns its calibration (do this early)

`Portfolio.calibrate_distortion(name, ...)` currently embeds ~240 LOC of per-name Newton iterations and closed-form `f`, `f'` for `ph`, `wang`, `ly`, `clin`, `roe`/`ccoc`, `lep`, `tt`, `cll`, `dual`, `tvar`, `wtdtvar` — a giant if/elif by string. That belongs on the `Distortion` subclass: PH knows how to calibrate itself, given an S vector + a premium target.

After: `Portfolio.calibrate_distortion` reduces to "extract S, ask the named Distortion to calibrate, store" (~30 lines). `Portfolio.calibrate_distortions` is a small loop over the standard set.

Discrete and easily-testable; queue it early in the sequence.

### 4. `Answer` class → named dataclasses

`utilities.Answer` is a generic-bag class; meaningful field names exist by convention only. Mirror the `ParsedProgram` refactor with a small dataclass per return type:

- `AnalyzeDistortionResult(distortion, augmented_df, pricing_df, exhibit, audit_df)` — single distortion
- `AnalyzeDistortionsResult(distortions, augmented_dfs, pricing_df)` — multi-distortion
- `PricingBoundsResult(bounds, allocs, stats, p_star)`
- `GradientResult(...)`, `BlendResult(...)` etc. for residual users

`apply_distortion` returns a `DataFrame` directly (after the cache refactor), so it doesn't need a wrapping type at all.

### 5. Ordered categoricals for canonical column / index orders

Pandas `CategoricalDtype(categories=[...], ordered=True)` lets us bake "the right order" into the data. Use it in two places:

**(a) Distortion order.** Defined once in `aggregate/spectral.py` (next to `Distortion`):

```python
DISTORTION_ORDER = ['ccoc', 'ph', 'wang', 'dual', 'tvar', 'wtdtvar', 'lep', 'ly', 'clin', 'tt', 'cll', 'bitvar', 'blend']
DISTORTION_DTYPE = pd.CategoricalDtype(categories=DISTORTION_ORDER, ordered=True)
```

`distortion_df.index` (formerly `dist_ans.index`) gets cast to `DISTORTION_DTYPE`, and any other frame indexed by distortion name follows suit. Sort becomes correct automatically.

**(b) Pricing-statistic column order.** Defined alongside the pricing functions:

```python
PRICING_STAT_ORDER = ['L', 'LR', 'M', 'P', 'PQ', 'Q', 'ROE']  # loss, loss ratio, margin, premium, P:Q, capital, ROE
PRICING_STAT_DTYPE = pd.CategoricalDtype(categories=PRICING_STAT_ORDER, ordered=True)
```

Apply at frame construction in `price`, `analyze_distortion`, `analyze_distortions`. Stops the "did I list these in the right order this time?" gotcha. (Side note on Q vs equity: yes, those terms drift in the literature; sticking with `Q` as capital/equity is the lesser evil since it's used throughout PIR.)

## Renames

| current                | new                                                                | reason                                          |
|:-----------------------|:-------------------------------------------------------------------|:------------------------------------------------|
| `self.dists`           | `self.distortions`                                                 | dict of calibrated `Distortion` objects         |
| `self.dist_ans`        | `self.distortion_df`                                               | tidy DataFrame; subsumes current property       |
| `self.limits` (method) | `self._limits`                                                     | matches Aggregate; private to plot helpers      |
| `self.augmented_df`    | `self.augmented_df(distortion)` — method over `_augmented_dfs` cache | lazy eval; signature change but PMIR-friendly   |

## API simplifications (lists → single value)

| current                                                    | new                                          |
|:-----------------------------------------------------------|:---------------------------------------------|
| `calibrate_distortions(LRs=, COCs=, ROEs=, As=, Ps=, ...)` | `calibrate_distortions(coc, p)`              |
| `analyze_distortions(As=, Ps=, ...)`                       | `analyze_distortions(p, distortions=None)`   |
| `apply_distortions(dist_dict, As=, Ps=, ...)`              | gone (absorbed into the cache + `analyze_distortions`) |
| `price(p, ...)`                                            | unchanged (already single-p)                 |

The `2`-suffixed methods in code today (`calibrate_distortions2`, `analyze_distortions2`) **are** the canonical signatures. They get renamed to the unsuffixed names; the list-based versions are removed. PMIR's `dmc_p2p.py:69` call `port.analyze_distortions2(p=1, dists=dists)` becomes `port.analyze_distortions(p=1, distortions=dists)`.

### 6. Drop the auto-approximation path

`Portfolio.update` currently chooses, per unit, between FFT (`'exact'`) and a method-of-moments shortcut (`approx_type ∈ {'slognorm','sgamma'}`) using the per-unit threshold `approx_freq_ge` — the "if the frequency is high, don't bother with the FFT" path. In practice this is never a problem ("just use more buckets"). Drop the whole path.

**Surface affected.**

- `Portfolio.update` signature: drop `approx_freq_ge`, `approx_type`, `approximation` parameters; drop the `'exact' if agg.n < approx_freq_ge else approx_type` ternary; always pass `'exact'` down.
- `Portfolio.__init__` state: drop `approx_freq_ge`, `approx_type` attributes (and any `info`/`__str__`/`json` lines that read them).
- `Aggregate.update_work`: drop the slognorm/sgamma branch in the inner switch (it lives one level down).
- `Portfolio.approximate(approx_type=...)` — stays. This is the user-facing on-demand fit; it's a different code path that takes the approx_type as a parameter and doesn't read `self.approx_type`.

**Reaches into Aggregate.** That's why this should land early — same Aggregate-side cleanup as tilting.

### 7. Drop tilting

`tilt_amount`, the `tilt_vector` constructed inside `update`, and the FFT-tilt path through `ft`/`ift`/`Aggregate.update_work` all go ("just use more buckets"). Mostly an Aggregate concern, small Portfolio surface (`tilt_amount` attr + the `if self.tilt_amount != 0` branch in `update`, the `tilt=None` parameter on `ft`/`ift`).

Same shape as approximation drop, same reason to do it early: the Aggregate-side touch is easier to land while both files are still in active refactor scope.

## Summary

- **Keep (K):** lean Portfolio — distributional surface, FFT internals, the distortion pricing engine (after redesign), the switcheroo, `plot`, `bodoff`, `sample`, `sample_compare`/`sample_density_compare`, `make_comonotonic_allocations`, plus the meta/state/info layer.
- **Move (M):** `extensions/portfolio_pir.py` — premium_capital family + AEBS + make_all + show_enhanced + EX_* state + set_a_p, profit_segment + natural_profit + biv_contour + density_sample, **twelve_plot** + `short_renamer`, gamma, stand_alone_pricing(_work), calibrate_blends + module helpers, from_DataFrame/Excel/dict_of_aggs, the BIG `renamer` + `premium_capital_renamer` dicts.
- **Delete (D):** gradient, non-spectral allocation family (merton_perold, cotvar, equal_risk_epd, equal_risk_var_tvar), EPD/priority/collateral (analysis_priority, analysis_collateral, priority_capital_df, priority_analysis_df, epd_2_assets, assets_2_epd), uat trio, collapse, audits, stat_renamer, **approx_freq_ge + approx_type + auto-approximation path**, **tilt_amount + tilt path**, **apply_distortions plural** (absorbed by cache), inner branches of `analyze_distortion` (add_comps + plots).
- **Rename:** `dists`→`distortions`, `dist_ans`→`distortion_df`, `limits`→`_limits`.
- **Redesign:** `apply_distortion`/`augmented_df` → lazy-eval cache; `Distortion` owns calibration; `Answer` → named dataclasses; lists-API → single-value API; ordered categoricals for distortion + pricing-stat orders.
- **Phase 3:** statistics, statistics_df, report, report_df, audit_df, make_audit_df → consolidated into `stats_df` parallel to Aggregate.

## Sub-projects

Five sub-projects, each ending with a human-review/commit checkpoint. **Compact between sub-projects** — each one is written to be readable cold by a fresh agent who has access to (a) this file, (b) the codebase, and (c) `CLAUDE.md`. Pre-conditions name the prior sub-projects so you can tell what state the tree should already be in.

Verification target throughout: `uv run pytest` → 415 passed. Visual `uv run python -m aggregate.extensions.test_suite` should produce its HTML report cleanly (modulo the pre-existing `Cc.Freq20` ZT-Poisson `brentq` crash). Doc builds are out-of-cycle per CLAUDE.md.

PowerShell working environment, `UV_LINK_MODE=copy` set in `.claude/settings.local.json`, branch is `REFACTOR`, work is committed only when explicitly approved.

### Sub-project A — Pure deletions + PIR move (planning steps 1, 2)

**Goal.** Shrink `portfolio.py` from ~6,133 LOC to ~3,500 LOC by deleting dead code and moving PIR-exhibit machinery to a new `extensions/portfolio_pir.py`. No semantic change to the surviving surface.

**Pre-conditions.** Branch `REFACTOR` at the merge base of this planning doc.

**Touches.** `aggregate/portfolio.py` (heavy), `aggregate/extensions/portfolio_pir.py` (new file), `aggregate/extensions/case_studies.py` (import updates), `aggregate/extensions/bodoff.py` (may need light touch), `tests/`.

**Deletions** (no consumers in docs, PMIR, or aggregate-internal except other Bucket-D methods):

- `gradient` and the `np.gradient` import line if it becomes unused (it isn't — check `np.gradient` calls before deleting the `import`).
- Non-spectral allocation: `merton_perold`, `cotvar`, `equal_risk_var_tvar`, `equal_risk_epd`.
- EPD / priority / collateral: `analysis_priority`, `analysis_collateral`, `priority_capital_df`, `priority_analysis_df`, `epd_2_assets`, `assets_2_epd`, plus `_epd_2_assets`, `_assets_2_epd` backing attrs.
- `var_dict` — keep the method, drop the `kind='epd'` branch (raise `ValueError` for unknown kind).
- UAT trio: `uat`, `uat_differential`, `uat_interpolation_functions`.
- `collapse`, `audits`, `stat_renamer`.
- Inner branches only: `analyze_distortion_add_comps` body (delete the helper), `analyze_distortion_plots` body (delete the helper). Keep `analyze_distortion`'s `add_comps=False` and `plot=False` defaults for parameter compat with PMIR; parameters themselves go in Sub-project D.

**Moves to `aggregate/extensions/portfolio_pir.py`** (recommend: free functions taking `port: Portfolio` as first arg, not monkey-patched methods — clearer at the call site):

- Premium-capital exhibits: `premium_capital`, `multi_premium_capital`, `accounting_economic_balance_sheet`, `make_all`, `show_enhanced_exhibits`, `set_a_p` (only consumers are the above), `EX_premium_capital`/`EX_multi_premium_capital`/`EX_accounting_economic_balance_sheet` state attrs (init in portfolio_pir, attached to the Portfolio instance on demand).
- PIR figures: `profit_segment_plot`, `natural_profit_segment_plot`, `biv_contour_plot` + `density_sample` helper, `twelve_plot` + `short_renamer` (twelve_plot is its only consumer), `gamma`.
- PIR pricing: `stand_alone_pricing`, `stand_alone_pricing_work`, `calibrate_blends` + module-level helpers `check01`, `make_array`, `convex_points`.
- Bulk constructors: `from_DataFrame`, `from_Excel`, `from_dict_of_aggs`. Lightweight; case_studies and a few PIR uses.
- Renamers: `renamer` (the BIG presentation dict), `premium_capital_renamer`.

**Tasks.**

1. Create `aggregate/extensions/portfolio_pir.py` with module docstring and imports.
2. Pick the move pattern: free functions `do_thing(port, ...)`. Code change at call sites in case_studies is `port.do_thing(...)` → `from aggregate.extensions.portfolio_pir import do_thing; do_thing(port, ...)`.
3. Move each Bucket-M function. Keep numerical behavior identical.
4. Update `case_studies.py` imports — list of touched call sites: `gross.calibrate_blends`, `port.stand_alone_pricing`, `port.analyze_distortions` (this stays on Portfolio — it's K, not M), `port.premium_capital`, `port.make_all`, etc. Audit before commit.
5. Update `bodoff.py` if it touches any moved bits (it shouldn't — uses `var_dict`, `tvar_threshold`, `cotvar` — and `cotvar` is being deleted, so `bodoff.py:40` line `basic.loc['coTVaR'] = self.cotvar(pt)` needs to be either deleted or rewritten as a manual `exgta_` lookup).
6. Delete Bucket-D code from `portfolio.py`.
7. Strip `analyze_distortion_add_comps` and `analyze_distortion_plots` bodies; keep call-site stubs that no-op when called (so old call sites with `add_comps=False, plot=False` defaults still work).
8. Drop `var_dict(kind='epd')` branch.
9. Audit tests/ — anything that references a deleted/moved method needs touch.
10. `uv run pytest` — chase to 415 green.
11. `uv run python -m aggregate.extensions.test_suite` — confirm visual report still builds.

**Verification.**
- `uv run pytest` → 415 passed.
- Visual `test_suite` HTML builds.
- `from aggregate.extensions import case_studies; cs.CaseStudy()` constructs without error.
- PMIR uses unaffected (audited earlier: no Bucket-M function is referenced by PMIR pkg).

**Commit.** One commit at the end of the sub-project after human review.

**README bullet (1.0.0b? section — bump after a4):**
```
- Portfolio: pruned dead code (gradient, non-spectral allocations,
  EPD/priority/collateral family, uat trio, collapse, audits — ~700 LOC)
  and moved PIR-exhibit machinery to ``aggregate.extensions.portfolio_pir``
  (premium_capital + AEBS family, profit_segment/natural/biv_contour
  plots, twelve_plot + short_renamer, gamma, stand_alone_pricing,
  calibrate_blends, from_DataFrame/Excel/dict_of_aggs, ~1,800 LOC).
  ``portfolio.py`` shrinks from 6,133 → ~3,500 LOC.
```

**Post-conditions.** Portfolio is roughly half its starting size. Subsequent sub-projects land cleaner diffs.

---

### Sub-project B — Drop approximation + tilting paths (planning steps 3, 4)

**Goal.** Remove the auto-fallback-to-method-of-moments approximation path and the FFT tilting path. Both reach into Aggregate, so do them together while we're touching `Aggregate.update_work`.

**Pre-conditions.** Sub-project A landed. Portfolio at ~3,500 LOC.

**Touches.** `aggregate/portfolio.py`, `aggregate/distributions.py` (Aggregate.update_work), `aggregate/utilities.py` (ft/ift wrappers if they carry `tilt=`).

**Approximation drop.**

- Drop `approx_freq_ge` and `approx_type` attrs from `Portfolio.__init__`.
- Drop `approx_freq_ge`, `approx_type`, `approximation` kwargs from `Portfolio.update`.
- Drop the `'exact' if agg.n < approx_freq_ge else approx_type` ternary; always pass `'exact'` (or rename the param to something clearer at the call site to `Aggregate.update_work`).
- Drop any `info`/`__str__`/`json` lines that read the dropped attrs.
- In `Aggregate.update_work`: drop the slognorm/sgamma branch keyed on `approx_type`. The moment-matching shortcut goes; FFT is the only path.
- **Untouched:** `Portfolio.approximate(approx_type='slognorm')` and `Aggregate.approximate(approx_type='slognorm')`. These are user-facing on-demand method-of-moments fits; they take `approx_type` as a parameter and don't read `self.approx_type`. Separate machinery.

**Tilting drop.**

- Drop `tilt_amount` attr from `Portfolio.__init__`.
- Drop the `if self.tilt_amount != 0: tilt_vector = np.exp(...)` block in `Portfolio.update`.
- Stop propagating `tilt_vector` from `Portfolio.update` → `Aggregate.update_work`.
- Drop the `tilt=None` parameter on `Portfolio.ft`/`Portfolio.ift` (and corresponding `aggregate/utilities.py:ft`/`ift` wrappers).
- In `Aggregate.update_work`: drop the tilt branch in the inner FFT call.

**Tasks.**

1. Grep `rg "approx_freq_ge|approx_type|approximation\\s*=" aggregate/` and list every read site.
2. Drop reads + parameter declarations in `Portfolio.update`, `__init__`, info, json (and any test that references these as kwargs).
3. Drop the slognorm/sgamma branch in `Aggregate.update_work`. Note: this affects the inner FFT step in `Aggregate.update_work` only — `Severity` discretization and the outer `__init__` logic should be untouched.
4. Grep `rg "tilt_amount|tilt_vector|tilt\\s*=" aggregate/` and list every read.
5. Drop tilt construction + propagation in `Portfolio.update`.
6. Drop `tilt=` from `ft`/`ift` in `portfolio.py` and `utilities.py`.
7. Drop tilt branch in `Aggregate.update_work`.
8. `uv run pytest`.
9. Manual sanity: `build('agg Dice dfreq [3] dsev [1:6]')` and a multi-line port both update green with no surprises.

**Verification.**
- `uv run pytest` → 415 passed.
- Visual `test_suite` green.
- Manual `print(port.info)` shows no broken references to dropped attrs.

**Commit.** One commit at sub-project end after review.

**README bullet:**
```
- Dropped the auto-approximation path (slognorm/sgamma fallback when
  frequency was high) from ``Portfolio.update`` and ``Aggregate.update_work``;
  always use FFT. ``Portfolio.approximate`` / ``Aggregate.approximate``
  (on-demand method-of-moments fit) untouched.
- Dropped FFT tilting (``tilt_amount`` attr, ``tilt_vector`` construction,
  ``tilt=`` parameter on ``ft``/``ift``). Use more buckets instead.
```

**Post-conditions.** `Portfolio.update` and `Aggregate.update_work` have leaner signatures. The Aggregate-side mechanical touch is done.

---

### Sub-project C — Distortion owns its calibration (planning step 5)

**Goal.** Move per-name Newton iterations out of `Portfolio.calibrate_distortion` (currently ~240 LOC of if/elif by distortion name) into the appropriate `Distortion` subclasses in `aggregate/spectral.py`. Each subclass calibrates itself.

**Pre-conditions.** Sub-projects A, B landed.

**Touches.** `aggregate/spectral.py`, `aggregate/portfolio.py`, `tests/`.

**Design.** Each pricing distortion subclass gains a method:

```python
def calibrate(self, S: np.ndarray, bs: float, premium_target: float,
              ess_sup: float = 0.0, **kwargs) -> "Distortion":
    """Fit shape parameter to hit premium_target via numerical integration of g(S) over bs grid.
    Mutates self.shape and self.error; returns self for chaining."""
```

The Portfolio side reduces to:

```python
def calibrate_distortion(self, name, ...):
    # extract S, ess_sup, premium_target
    dist = Distortion(name=name, ...)
    dist.calibrate(S, bs=self.bs, premium_target=premium_target, ess_sup=ess_sup)
    return dist
```

`Portfolio.calibrate_distortions` (after Sub-project D simplifies it to single-coc/single-p) is a thin loop over the standard list of distortion names.

**Tasks.**

1. Identify each `name == '<X>'` branch in `Portfolio.calibrate_distortion` (currently at portfolio.py:2630-2762): `ph`, `wang`, `ly`, `clin`, `roe`/`ccoc`, `lep`, `tt`, `cll`, `dual`, `tvar`, `wtdtvar`. Note: `roe`/`ccoc` shares a branch.
2. For each, add a `calibrate(S, bs, premium_target, ...)` method to the corresponding `Distortion<Kind>` subclass in `spectral.py`. Faithful port; same Newton iteration, same convergence criteria.
3. Refactor `Portfolio.calibrate_distortion` to extract S + targets, then delegate.
4. Add tests in `tests/`: for each subclass, construct with a known `S` vector and `premium_target`, assert that `calibrate` recovers the expected shape parameter.
5. `uv run pytest`. Old test cases that hit `Portfolio.calibrate_distortion` should still pass with identical numerics.

**Verification.**
- `uv run pytest` → 415 + new per-distortion calibration tests passed.
- Visual `test_suite` green.
- Numerical check: `port.calibrate_distortion('ph', ...)` produces the same `shape` value before vs after.

**Commit.** One commit at sub-project end.

**README bullet:**
```
- Distortion calibration moved from ``Portfolio.calibrate_distortion``
  (~240 LOC if/elif by name) into the ``Distortion`` subclasses
  (``DistortionPH.calibrate``, ``DistortionWang.calibrate``, etc.).
  ``Portfolio.calibrate_distortion`` shrinks to ~30 LOC of dispatch.
  Each distortion is now testable in isolation.
```

**Post-conditions.** Distortion subclasses own their own numerics. Portfolio's calibration layer is thin.

---

### Sub-project D — Distortion pricing redesign (planning steps 6–11)

**Goal.** Six related changes that together produce a clean distortion-pricing pipeline. Order within the sub-project matters because each step feeds the next: cache lands first, then the analyze methods collapse onto it, then the dataclasses wrap the small results, then categoricals + renames clean up.

**Pre-conditions.** Sub-projects A, B, C landed. `Distortion` subclasses know how to calibrate themselves.

**Touches.** `aggregate/portfolio.py` (heavy), `aggregate/spectral.py` (ordered categorical), `aggregate/utilities.py` (Answer → dataclasses), PMIR call sites (`pmir_package/src/pmir/*.py` and current notebooks) for the API + rename sweep.

**Order of operations within Sub-project D.**

1. **`augmented_df` lazy-eval cache** (planning step 9 — done first because everything else depends on it).
   - Introduce `self._augmented_dfs: dict[str, pd.DataFrame]` on Portfolio. Initialize empty in `__init__`. Reset in `update`.
   - Replace `apply_distortion(d)`'s body with: `name = d.name; if name not in self._augmented_dfs: self._augmented_dfs[name] = self._build_augmented(d); return self._augmented_dfs[name]`. Keep the function name for now (rename in step 5 of this sub-project, below).
   - Move the current `apply_distortion` body (the actual augmented_df construction) into `_build_augmented(self, distortion) -> pd.DataFrame`. Pure: no side effects, returns the frame.
   - Delete `apply_distortions` (plural).
   - Add `pricing_at(self, p, distortion) -> pd.DataFrame`: pull the L/LR/M/P/PQ/Q/ROE row from the cached augmented_df at the asset level corresponding to `p` (or `a` if `p > 1`).
   - Keep the current `self.augmented_df` attribute working temporarily as "most-recently-applied" (so `Portfolio.gradient`-style users — wait, gradient is deleted in A — so this might just be unnecessary). Decide: drop the attr entirely, make `self.augmented_df` a method `(distortion)` over the cache. PMIR's `ans.augmented_dfs[dname]` reads survive because they come from `analyze_distortions` return, not from the Portfolio attr.

2. **`analyze_distortion(s)` collapse** (planning step 6 — surgery + step 7 — API simplification, combined).
   - `analyze_distortion(self, distortion, p)`: returns `pricing_at(p, distortion)` plus light metadata (e.g., calibration audit values). Drop the `add_comps` and `plot` parameters entirely (they were already no-op after Sub-project A).
   - `analyze_distortions(self, p, distortions=None)`: takes single `p`, optional distortions dict (defaults to `self.distortions`). For each `d`, ensure `augmented_df(d)` (cache warm), then concat `pricing_at(p, d)` keyed by distortion name. Replaces the current list-based `analyze_distortions` AND `analyze_distortions2`.
   - `calibrate_distortions(self, coc, p)`: takes single `coc`, single `p`. Calibrates the standard distortion set (ccoc, ph, wang, dual, tvar) into `self.distortions` (the new name — see step 6). Replaces the current LRs/COCs/ROEs/As/Ps list-based version AND `calibrate_distortions2`.
   - Update PMIR call sites:
     - `pmir_package/src/pmir/dmc_p2p.py:69`: `port.analyze_distortions2(p=1, dists=dists)` → `port.analyze_distortions(p=1, distortions=dists)`.
     - Other `analyze_distortions(p=..., add_comps=False)` call sites: drop the `add_comps=False` kwarg.
     - `calibrate_distortions(COCs=[coc], Ps=[reg_p])` patterns → `calibrate_distortions(coc, reg_p)`.

3. **`Answer` → named dataclasses** (planning step 8).
   - Define in `aggregate/utilities.py` (or new `aggregate/results.py`):
     - `AnalyzeDistortionResult(distortion, pricing_df, audit_df)`
     - `AnalyzeDistortionsResult(distortions, pricing_df, augmented_dfs)`
     - `PricingResult(df, price, price_dict, a_reg, reg_p)` (for `price()` — replaces today's namedtuple)
     - `PricingBoundsResult(bounds, allocs, stats, comp, allocs_slow, p_star)`
     - `GradientResult(...)`, `BlendResult(...)` if any residual consumers — but gradient is deleted, so probably just blend
   - Sweep `aggregate/` for `Answer(...)` and replace.
   - PMIR call sites that destructure Answer fields are touched: `ans.augmented_dfs[dname]` reads stay (now on `AnalyzeDistortionsResult`), `ans.exhibit` becomes `result.exhibit`, etc.

4. **Ordered categoricals** (planning step 10).
   - In `aggregate/spectral.py`:
     ```python
     DISTORTION_ORDER = ['ccoc', 'ph', 'wang', 'dual', 'tvar', 'wtdtvar', 'lep', 'ly', 'clin', 'tt', 'cll', 'bitvar', 'blend']
     DISTORTION_DTYPE = pd.CategoricalDtype(categories=DISTORTION_ORDER, ordered=True)
     ```
   - In `aggregate/portfolio.py`:
     ```python
     PRICING_STAT_ORDER = ['L', 'LR', 'M', 'P', 'PQ', 'Q', 'ROE']
     PRICING_STAT_DTYPE = pd.CategoricalDtype(categories=PRICING_STAT_ORDER, ordered=True)
     ```
   - Cast index of `distortion_df` (the renamed `dist_ans` — see step 5) to `DISTORTION_DTYPE`.
   - Cast columns of `pricing_at`, `analyze_distortion`, `analyze_distortions` result frames to `PRICING_STAT_DTYPE`.

5. **Renames** (planning step 11).
   - `self.dists` → `self.distortions` (dict). Sweep `aggregate/`, `tests/`, and PMIR.
   - `self.dist_ans` → `self.distortion_df` (renamed AND column-trimmed; merge with the existing `distortion_df` property that produces a tidy view).
   - `self.limits()` → `self._limits()`. Internal-only; consumers are `plot`, `analyze_distortion`, etc.

**Tasks.** Roughly the same order as above; one logical commit per major step within the sub-project, all squashed at end if preferred, or kept as 5 commits for review granularity (recommend keeping separate during review, squash before merge).

**Verification.**
- `uv run pytest` → 415 passed (touched tests for renames).
- Visual `test_suite` green.
- PMIR smoke test: open `pricing-from-the-insureds-perspective.ipynb`, run the cells that touch Portfolio. Numerical results match prior run.
- Before/after numerical check on `port.analyze_distortions(p=1)` for a representative case-study Portfolio — results unchanged.

**Commit.** Either one squashed commit at sub-project end, or 5 step-commits per the order above. User preference.

**README bullet:**
```
- Portfolio distortion pricing pipeline rewritten:
    - ``augmented_df(distortion)`` is now a lazy-eval method backed by an
      internal cache keyed by distortion name. ``apply_distortion``
      becomes the cache-warmer; ``apply_distortions`` (plural) gone. New
      ``pricing_at(p, distortion)`` extractor pulls the L/LR/M/P/PQ/Q/ROE
      row from the cached frame.
    - ``analyze_distortion`` and ``analyze_distortions`` collapse to
      ``pricing_at`` + concat over the distortion dict.
    - ``calibrate_distortions(coc, p)`` and ``analyze_distortions(p, distortions=None)``
      replace the prior LRs/COCs/ROEs/As/Ps list-based signatures.
    - Generic ``Answer`` return type replaced with named dataclasses
      (``AnalyzeDistortionResult``, ``AnalyzeDistortionsResult``,
      ``PricingResult``, ``PricingBoundsResult``).
    - ``DISTORTION_ORDER`` and ``PRICING_STAT_ORDER`` ordered categoricals
      ensure consistent index/column ordering in pricing exhibits.
    - Renamed ``dists`` → ``distortions``, ``dist_ans`` → ``distortion_df``,
      ``limits`` → ``_limits``.
```

**Post-conditions.** Portfolio's distortion-pricing surface is clean: one cache, one extractor, one dataclass family, one canonical signature pattern. PMIR happy. Ready for stats consolidation.

---

### Sub-project E — `stats_df` consolidation on Portfolio (planning step 12)

**Goal.** Mirror Aggregate's Stage 1c+ work on Portfolio: introduce `stats_df` as the single source of stats truth on Portfolio; delete the overlapping report/statistics/audit_df family. Public stats surface becomes exactly three things on Portfolio: `info`, `describe`, `stats_df` — same shape as Aggregate.

**Pre-conditions.** Sub-projects A-D landed. Aggregate already has `stats_df` (Stage 1c+). Portfolio still has the old report/statistics/audit_df family. This sub-project is the Portfolio analog of what we just did on Aggregate.

**Touches.** `aggregate/portfolio.py`, `tests/`, `docs/` (some doc files reference `port.statistics`/`port.report_df` patterns and need migration analogous to the Aggregate doc migration).

**Frames being collapsed.**

| current frame                | replacement                                                  |
|:-----------------------------|:-------------------------------------------------------------|
| `statistics_df` (MultiIndex) | `stats_df.drop(['mixed','independent','empirical','error'])` |
| `statistics` (alias)         | gone — use `stats_df`                                        |
| `report_df` (flat stat names)| gone — use `stats_df`                                        |
| `report` (display HTML)      | gone — `qd(stats_df)`                                        |
| `audit_df` (empirical+error) | `stats_df[['empirical','error']]`                            |
| `make_audit_df` (helper)     | gone — folded into `update`'s stats_df construction          |

**Tasks.** Follow the recipe used for Aggregate Stage 1c+ (`docs/stage1c+` checklist if useful; the planning lives in `refactor-plan.md` parking lot):

1. Define `_PORT_STATS_ROW_INDEX` MultiIndex on `(component, measure)`. Likely same shape as Aggregate's `_STATS_ROW_INDEX` (meta + freq + sev + agg sections). Decide whether to share the constant with Aggregate or duplicate.
2. Build `stats_df` in `Portfolio.__init__` after the existing statistics_df / audit_df construction — parallel-write.
3. Add empirical + error columns to stats_df in `Portfolio.update` after the existing audit_df construction.
4. Rewrite `Portfolio.describe` to source from `stats_df['mixed']` and `stats_df['empirical']` (mirror Aggregate's describe).
5. Rewrite `Portfolio.info` if it reads old frames.
6. Verify: pytest green, numerical outputs unchanged.
7. Flip consumers in `aggregate/extensions/portfolio_pir.py` to read from stats_df.
8. Sweep `docs/` for `port.statistics`, `port.report_df`, `port.audit_df`, `port.statistics_df` references; rewrite to `stats_df` syntax (mirror the migration we did in Stage 1c+ for Aggregate).
9. Delete: `statistics_df`, `statistics`, `report_df`, `report`, `audit_df`, `make_audit_df`, `_audit_df` if present, plus init lines for these attrs.
10. `uv run pytest` → 415 green.

**Verification.**
- `uv run pytest` → 415 passed.
- Visual `test_suite` green.
- Manual: `port.stats_df`, `port.describe`, `port.info` produce expected outputs on a multi-unit example.
- Doc grep clean: `rg "report_df|statistics_df|audit_df|\.statistics\\b" docs/ -t rst` returns nothing on Portfolio object access.

**Commit.** One commit at sub-project end after review.

**README bullet:**
```
- Portfolio stats consolidation: ``statistics`` / ``statistics_df`` /
  ``report`` / ``report_df`` / ``audit_df`` / ``make_audit_df`` collapsed
  into a single canonical ``stats_df`` (MultiIndex on (component, measure)
  × per-unit + mixed + independent + empirical + error). Public stats
  surface on ``Portfolio`` is now exactly three things: ``info``,
  ``describe``, ``stats_df`` — same shape as ``Aggregate``.
```

**Post-conditions.** Portfolio's stats surface is parallel to Aggregate's. The Portfolio refactor is complete.

---

## Cross-cutting reminders for every sub-project

- `UV_LINK_MODE=copy` is set in `.claude/settings.local.json`. Don't override.
- Run `uv run pytest`, not bare `pytest`.
- PowerShell shell: no awk/sed; use `rg` directly for grep, Read/Edit/Write for files (per `feedback_use_rg`, `feedback_powershell_no_unix`).
- Don't prefix Bash commands with `cd T:/...` (per `feedback_no_cd_prefix`).
- Update `README.rst` at the close of each sub-project with the bullet draft above (per `feedback_readme_after_iteration`). Bump the alpha version in `pyproject.toml` as appropriate.
- Don't build docs as part of a verification cycle (CLAUDE.md). Doc audits via `rg` are fine.
- Do NOT commit unless the user explicitly says so. Each sub-project ends with a human review step before the commit lands.
- The visual `test_suite` has one pre-existing failure (`Cc.Freq20` ZT-Poisson `brentq` crash). Treat it as known and ignore unless a new failure appears.
