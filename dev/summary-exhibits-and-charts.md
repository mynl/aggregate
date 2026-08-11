# Who publishes what: exhibits and charts

> Reference snapshot of the two registries as they stand in the working tree, 1.0.0a238 era. Every fact below was read off the code or introspected from a live registry, not recalled. Design rationale lives in `dev/exhibits-and-charts.md`, `dev/plan-exhibits.md` and `dev/plan-chart-ir.md`; this file answers only "what is published, by whom, with what options".

Both surfaces work the same way. A registry maps a name to a `singledispatch` generic plus an availability predicate, capability is derived from the registry rather than declared a second time (`exhibits.available_exhibits(obj)`, `charts.available_charts(obj)`), and one module function builds the document (`exhibits.build_exhibit(obj, name, perspective)`, `charts.build_chart_doc(obj, name, **options)`). Both packages are provisional in the PEP 411 sense and outside the 1.0 API contract.

## 1. Exhibits

An exhibit is a titled envelope of one or more blocks, each block a greater_tables `TableDoc` over one frame with its captions, formats and row flags. **Perspective** is who is reading. The enum has four members, `RAW`, `INSURED`, `INSURER`, `REINSURER`, and exactly two are implemented: `RAW` and `INSURER`. `INSURED` and `REINSURER` are declared vocabulary so the enum does not churn later; asking for either raises rather than guessing.

The governing rule is that **INSURER equals RAW unless an override is registered for that (exhibit, class) pair**. So the "INSURER changes" column below is the complete list of business translation that exists today, and a blank means the two perspectives serve identical bytes.

| Exhibit | Title | Published by | Blocks (source frames) | Available when | INSURER changes |
|---|---|---|---|---|---|
| `summary` | Summary | Aggregate, Portfolio, PnL, Distortion, BivariateAggregate | 1: `summary_df` | always | Aggregate, Portfolio: caption, total/subtotal row flags, measure formats |
| `tail` | Return periods | Aggregate, Portfolio | 1: `tail_df` | updated | both: caption, 1-in-200 and 1-in-250 row emphasis |
| `stats` | Statistics | Aggregate, Portfolio, PnL, Distortion, BivariateAggregate | 1: `stats_df` | always | Aggregate, Portfolio, PnL: drops the raw noncentral rows `ex1`, `ex2`, `ex3` (26 rows becomes 23), caption |
| `validation` | Validation | Aggregate, Portfolio, PnL, Distortion, BivariateAggregate | 1: `validation_df` | always | Aggregate, Portfolio, Distortion, BivariateAggregate: caption, emphasis on rows failing the object's `validation_eps` gate |
| `reins` | Reinsurance | Aggregate, Portfolio | RAW 2: `reins_stats_df`, `reins_summary_df` | cedes and updated | Aggregate **restructures into 3 blocks**: `reins_layer_terms` and `reins_layer_moments`, layers down the rows, then the summary. Portfolio (no layer axis) drops the raw noncentral rows and flags the summary rows |
| `economic` | Economics | PnL | 1: `economic_df` | always | caption switches on whether the ladder is a kappa scenario or a marginal `P` ladder, ledger row flags, measure formats |
| `economic_ratios` | Economic ratios | PnL | RAW 2: `economic_ratios_df`, `legs_df` | always | **restructures into 3 blocks**: `amounts` (P, L, E, C, M), `ratios` (LR, ER, CR, E_*, shares), `legs`, so no column mixes two units |
| `economic_waterfall` | Economic waterfall | PnL | 2: `walk_df`, `evaluation_df` | multi-step walk (`_tower`) | none |
| `dependency` | Dependency | BivariateAggregate | 2: `dependency_df`, `axis_support_df` | updated | none |
| `bs_window` | Grid sizing | Aggregate, Portfolio, BivariateAggregate | 1: `bs_window_df` | updated | none |
| `sharpen` | Grid probe | Aggregate, Portfolio | RAW 1: `sharpen_df` | probed (`sharpen_df` present) | **restructures into 2 blocks**: `score_grid` (`score` unstacked by `d_log2`) then the full walk |
| `tail_behavior` | Tail behavior | Aggregate, Portfolio | 1: `tail_behavior_df` | updated | none |

Read by class, which is the question a landing page asks:

* **Aggregate**: `summary`, `tail`, `stats`, `validation`, `bs_window`, `tail_behavior`, plus `reins` when it cedes and `sharpen` once the grid probe has run. Six to eight.
* **Portfolio**: the same, per unit plus the total.
* **PnL**: `summary`, `stats`, `validation`, `economic`, `economic_ratios`, plus `economic_waterfall` on a walk.
* **BivariateAggregate**: `summary`, `stats`, `validation`, `dependency`, `bs_window`.
* **Distortion**: `summary`, `stats`, `validation`.
* **Severity**: none. It publishes a chart but no exhibit, which is a gap rather than a decision.

## 2. Charts

A chart emitter returns a `ChartDoc`: axes, panels, series, marks, and chart-level `meta`, carrying semantics only. Where an exhibit has a perspective, a chart has **declared readings**: an axis names every scale it may honestly be read on (`scales`) and whether a zoom-out exists (`full_range` alongside `suggested_range`); a probability axis may name a paired return-period axis (`reciprocal_of`); a panel may declare itself `invertible` and may offer more than one realization (`kinds`). The renderer's switches act on every axis or panel that declares the reading and on nothing else, so a caller never needs to know which chart it is holding.

Chart-level facts, one line each. Options are semantic arguments to the emitter, never renderer settings.

* **`agg`** on Aggregate, its primary chart. Available when updated. Option `xmax`. Entry point `Aggregate.plot(xmax, log, full_range, return_period, invert)`.
* **`port`** on Portfolio, primary. Updated. Option `xmax`. Entry point `Portfolio.plot(xmax, log, full_range)`.
* **`pnl`** on PnL, primary. Needs `result`. No options. Entry point `PnL.plot(log, full_range, return_period, invert)`.
* **`severity`** on Severity, primary. Always. Option `n` (default 512 quantile-spaced points). Entry point `Severity.plot(n, log, full_range, return_period, invert)`.
* **`reins`** on Aggregate, primary for nothing (it is a view of a book, not the book's own picture). Needs an occurrence program. No options. Entry point `Aggregate.reins_occ_plot(log, full_range, return_period, invert)`.
* **`distortion`** on Distortion, primary. Always. Option `dual`. Entry point `Distortion.plot(dual, ax)`.
* **`envelope`** on Bounds, primary. Always. Options `n_resamples` (bracketing curves inside the band, each carrying its weight as `ChartSeries.value`), `n`. Entry point `Bounds.plot_envelope(n_resamples)`.
* **`joint_surface`** on BivariateAggregate, primary. Needs the in-memory joint density. Options `window` (default 4, the marginal quantile depth kept, measured on the fine lattice **before** the reduction; 0 = whole grid), `detail` (default 128 cells per axis, a ceiling reached by a power-of-two block sum, mass preservingly) and `encoding` (default `f32b64`, also `f64b64` / `u16log12b64` / `json`). The surface block carries `x0/dx/nx`, `y0/dy/ny`, `edge='left'`, `bs`, `k`, `window`, the exact `marginals`, fine-lattice `moments` and `deficit`. **No class method yet**: reach it through `charts.build_chart_doc` and `plots.plot_chartdoc`.

| Chart / panel | x axis | y axis | Series | Marks | Readings offered |
|---|---|---|---|---|---|
| `agg` / density | `outcome`, Loss, currency, linear or log, window `q(0.001) or 0 .. q(0.999)` padded 2%, full = whole grid | `mass`, Probability mass, density, linear or log, `(0, peak)`, no zoom-out | Aggregate and Severity, role `density`, atomic | mean, 1-in-200, both full weight | log x, log y, full x |
| `agg` / lee | `p`, Non-exceeding probability, `(0, 1)`, paired with `return_period` (log, `1 .. 1e9`) | `outcome`, shared with the density panel | Aggregate and Severity, role `cdf` | 1-in-100 and 1-in-250, faint | log y, full y, return period, invert to "Distribution function" |
| `port` / density | `outcome`, Loss, currency, linear or log, window, full | `mass`, linear or log, `(0, top)` | one per unit (role `unit`) then Total (role `total`) last, so the book draws on top; each on its own native grid | mean, 1-in-200 | log x, log y, full x |
| `port` / kappa | `outcome`, shared | `kappa`, `E[Xi \| X = x]`, currency, linear or log, window = the loss window, full to the last kept point | same names again, role `unit` / `total`, support continuous; the Total curve **is** the diagonal | 1-in-200, faint | log x, log y, full x, full y; `aspect='equal'` is semantic |
| `pnl` / density | `outcome`, P&L, currency, **linear only** (signed), window not anchored at 0, full | `mass`, linear or log | one series, the result name | break even at 0, mean | log y, full x |
| `pnl` / lee | `p`, paired with `return_period` | `outcome`, shared, linear only, full | the same series, role `cdf` | break even (horizontal), 1-in-100 and 1-in-250 faint | full y, return period, invert |
| `severity` / density | `loss`, currency, linear or log, `0 .. isf(0.001)` padded, full | `pdf` (or Probability mass for a law with no density), linear or log, no window | one series; continuous unless the law is atomic, in which case the ordinate is read from the jumps of the cdf | none, deliberately | log x, log y, full x |
| `severity` / lee | `p`, paired with `return_period` | `loss`, shared, linear or log, full | the same series, role `cdf` | none | log y, full y, return period, invert |
| `reins` / occurrence | `claim`, Loss per claim, currency, **linear only**, window = the occurrence limit padded 2%, full = grid | `sev_density`, Occurrence density, **log only**, no window | Gross, Ceded, Net, roles `gross` / `ceded` / `net`, net drawn last | none | full x only |
| `reins` / aggregate | `p`, paired with `return_period` | `annual`, Aggregate loss, currency, linear or log, window from the gross curve, full | Gross, Ceded, Net again, as quantile curves | none | log y, full y, return period, invert |
| `distortion` / square | `s`, probability, linear, `(0, 1)` | `g(s)`, probability, linear, `(0, 1)` | the distortion, the dual (optional), then identity | none | none: the unit square **is** the window, and `aspect='equal'` is the whole point |
| `envelope` / cloud | `s`, `(0, 1)` | `g(s)`, `(0, 1)` | Envelope (a `y2` band series, so the series is the region), `n_resamples` BiTVaR curves each carrying its weight, identity | none | none; equal aspect |
| `envelope` / calibrated | `s`, shared | `g(s)`, shared | the band again, then CCoC, TVaR(p\*), PH, Wang, Dual, then Avg extreme, then identity. **Panel omitted** where nothing is calibrated | none | none; equal aspect |
| `joint_surface` / joint | `x0`, resolved component label, currency | `x1`, resolved component label, currency; z axis `z`, density, linear or log | one `SurfaceData`, role `joint`, values are display-cell masses | none | log z. `kinds` declares only `'surface'` today, so the renderer's `kind` switch has nothing to choose between |

### Things worth knowing that no column holds

**Two panels, one axis, except once.** On `agg`, `pnl` and `severity` the outcome axis is a single `ChartAxis` referenced as the density panel's x and the Lee panel's y, so a window set on it moves both. On `port` both panels take it as x. `reins` is the deliberate exception: its two panels share nothing, because a per-claim loss and an annual aggregate are different quantities and one window across both would claim they were the same.

**Three floors, all measured rather than chosen.** `LOG_FLOOR = 1e-15` turns float dust into `None` gaps that a renderer must break the line at, never bridge. `SURVIVAL_FLOOR = 1e-9` is the deepest survival worth a panel and is what puts the return-period axis top at 1e9. `KAPPA_FLOOR = 1e-14` on the portfolio kappa panel is a decade above the dust floor because kappa divides by `p_total`, and the residual of the sum-to-diagonal identity is what measured the cliff.

**Support is a fact about the law.** `support='atomic'` says the points carry the whole distribution and there is nothing between them, which in this library is the normal case; `'continuous'` says they sample a function that exists everywhere between them, which is a distortion, a kappa curve, and a severity that has a density. The renderer picks stems, steps or a line from that plus the room it has.

**Three panels became two, twice.** The old aggregate compositor drew density, log density and Lee; the log density is not a third reading of the book, so it became a declared reading of the first, and the cdf panel came back as the Lee panel's declared inversion. The bounds compositor split five calibrated distortions across two panels by order of addition; all five now sit on one band.

**What is missing.** Severity publishes no exhibit. `joint_surface` has no class method. `INSURED` and `REINSURER` are vocabulary with no implementation. The joint panel declares one realization where the schema and the renderer both support a heatmap and surface toggle.
