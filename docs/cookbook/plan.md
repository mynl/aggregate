# [Cookbook] — a runnable feature cookbook for `aggregate`

**Status:** framing (skeletons + 2 exemplar pages). Calibration of the house examples is the author's; the structure and the checks are the design work here.

## [Cookbook-Purpose]

The 20-minute demo predates a wall of new features (PnL/xPnL, variable rating, reinstatements, placement scaling, labels-everywhere, bivariate, massive bivariate). This cookbook is the single place to (a) **relearn the syntax**, (b) **see the flow** on realistically-calibrated examples, and (c) **watch the invariants hold with your own eyes** — so the pytest suite guards what you already eyeballed once, and you can trust it.

It is *not* the API reference (that is the Sphinx `.rst` tree) and *not* a tutorial from zero. It is a chef's cookbook: one recipe per feature, each on the same rhythm, each ending in a visible check.

## [Cookbook-Mechanics] — how the pages compose

- **One master page, `cookbook.qmd`**, carries the only YAML frontmatter (title, kernel, toc, bibliography). It narrates and stitches the feature pages together with Quarto `{{< include _NN_name.qmd >}}` shortcodes.
- **Feature pages `_NN_name.qmd` are frontmatter-less fragments.** Quarto's `include` does a raw text splice *before* render, and its one hard rule is that included files must not carry their own YAML. So the fragments have none.
- **Shared setup is Python, not a qmd fragment: `_setup.py`.** Every page's first cell is `from _setup import *`. The import is idempotent, so the master re-running it once per include is free; and each page therefore **runs standalone in JupyterLab** (jupytext auto-recognizes `.qmd` as quarto markdown; kernel defaults to `python3`). This resolves the "compose into one page *and* open each alone in JL" tension without duplicating helper code.
- **The house book.** `_setup.py` defines *one* realistically-calibrated book (premium, plan LR, severity, frequency) reused across pages, so cross-page comparison means something and calibration is done once. **Calibration is PROVISIONAL — the author tunes it.**
- **Execution policy.** The bivariate and massive-bivariate pages are **never run automatically**: their expensive cells carry `#| eval: false` in the composed render (author runs them by hand in JL, or we cache captured output). Every other page executes on render.

## [Cookbook-Recipe-Shape] — the rhythm of every page

Each recipe follows the same four parts, borrowed from @Beazley2013. They are *implicit*: the parts are carried by the order of the material, not by labels on the page, so a hand-written page and a generated one read alike.

1. **Problem** — what you are trying to do, and why you would. One or two lines of opening prose.
2. **Solution** — the code that does it, self-contained enough to copy straight out of the page: the DecL (pretty-printed with `pp`) and the `build`.
3. **Discussion** — why it works and what to watch for: the common FCC surface (`.valid`, `.validation_explanation`), `summary_df` / `stats_df` / `.plot()` / the relevant `*_explanation`. One or two exhibits, not five.
4. **The check** — the visible invariant, inside a `::: {.callout-note collapse="true" title="The check"}` div so it collapses by default. **This is the hard part and the point of the cookbook.** See the check catalog below.

The generator (`aggregate.cookbook`) emits exactly this shape from a library entry's `doc{{{…}}}`, which carries `## Problem` / `## Solution` / `## Discussion` / `## Check` headings that are consumed into fields rather than rendered.

## [Cookbook-Check-Catalog] — the toolkit for the check

The check is where "trust the tests" becomes "see it yourself." Six archetypes; each page is tagged with the ones it uses, as an HTML comment inside the callout.

- **[Check-Reconciliation]** things add up: `net + ceded = gross`; walk means foot down the sheet; allocations sum to the total; consideration = `gross − ceded premium + commission`.
- **[Check-Scaling-Sweep]** vary one knob, watch the output move as predicted — a small table or a line. The placement page *is* this: sweep the share 25/50/75/100% and see ceded premium trace a straight line.
- **[Check-Independent-Oracle]** `est_*` (empirical/FFT) vs theoretic moments; or a quick Monte Carlo for the exotic terms with no closed form.
- **[Check-Limiting-Case]** sanity bounds: `TVaR ≥ VaR`; premium ≥ EL; a distortion has `g(0)=0, g(1)=1`, concave; a swing collar clamps at min/max; zero placement ⇒ gross.
- **[Check-Round-Trip]** `format_program(build(x))` round-trips; `pnl` consolidated == `xpnl` walk to grid accuracy; two equivalent spellings agree (`rol` vs the matching `deposit`).
- **[Check-Cross-Object]** an aggregate priced alone == its allocation in a one-unit portfolio; bivariate marginals recover the univariates.

## [Cookbook-Numbering] — sections own a number

To stop the "forever renumbering" churn, the scheme is **two-level and section-owned**: `_SS_MM_name.qmd`, where `SS` is a stable section (01…09) and `MM` is the recipe within it (`00` = section landing, `01+` = recipes). Adding an example is a new `MM` — it never renumbers a neighbor; a new section is rare. Single-recipe sections (Distortions, Bounds) skip the `MM` level.

Heading + TOC convention (so the TOC shows sections and recipes, nothing else):
- **section landing `_SS_00`** → `# Title` (H1, numbered → *N*).
- **recipe `_SS_MM`** (MM≥01) → `## Title` (H2, numbered → *N.M*).
- **single-recipe section** → `# Title` (H1).
- **the four parts** carry no labels at all, so they never clutter the TOC; the check is a collapsed callout div, not a heading. A long recipe may use `###` sub-headings for genuinely separate movements.
- **anchors** are semantic and number-free (`{#sec-placement}`), so cross-links (`@sec-placement`) survive any renumber.

## [Cookbook-Pages] — the recipe list

`How to read` is the master's unnumbered preamble. Distortions lead (simple, used everywhere) and double as the gentle intro to the common FCC surface.

| § | files | recipes | checks |
|---|-------|---------|---------------|
| 1 | `_01_distortions` | **gentle intro** — menu; calibrate/plot; *first meeting with `valid`/`summary`/`stats`/`plot`/`*_explanation`* | Limiting-Case, Round-Trip |
| 2 | `_02_00_severity` + `_01_discrete`, `_02_scipy_continuous` | the `sev` clause; discrete/empirical, then named continuous (leveraging the `.agg` actuarial examples); slognorm-vs-sgamma | Independent-Oracle |
| 3 | `_03_00_frequency` + `_01_discrete`, `_02_mixed_poisson`, `_03_other` | fixed/dfreq; poisson & mixed-gamma (contagion); binomial/geometric | Independent-Oracle |
| 4 | `_04_00_aggregate` + `_01_discrete`, `_02_insurance`, `_03_mixed_severity`, `_04_limit_profile`, `_05_pricing` | discrete (Dice); the house **insurance** book; mixed severity; limit profile; distortion pricing | Reconciliation, Independent-Oracle, Limiting-Case |
| 5 | `_05_00_reinsurance` + `_01_occurrence`, `_02_aggregate`, `_03_reinstatements`, `_04_variable_rating`, `_05_placement` | occ & agg layers; reinstatements; swing/slide/pc/corridor; **placement sweep** (check-heavy exemplar) | Reconciliation, Limiting-Case, Scaling-Sweep |
| 6 | `_06_00_pnl` + `_01_consolidated`, `_02_walk` | consolidated card; the gross→cover→Total walk; κ vs P | Reconciliation, Round-Trip |
| 7 | `_07_bounds` | `Bounds` / `AllocationBounds` / `PricingBounds` (IME 2022) | Limiting-Case |
| 8 | `_08_00_portfolio` + `_01_construction`, `_02_pricing`, `_03_samples` | multi-unit build; distortion pricing + allocation; the **switcheroo** | Reconciliation, Cross-Object, Independent-Oracle |
| 9 | `_09_00_bivariate` + `_01_massive` | occ view-pairs (`netceded`/…); disk-backed zarr — **both not auto-run** | Cross-Object, Reconciliation |

Aggregate moment attributes verified on the house book (`actual_m`, `est_m`, `sev_m`, `n`, `actual_cv`, `est_cv`) reconcile: `actual_m` = 7500 = 10000 × 75% lr, `n × sev_m` ≈ 7500. Used live in the `_04_02_insurance` check.

## [Cookbook-Decisions] (resolved 2026-07-07)

1. **Render target — Quarto *website* HTML.** A `_quarto.yml` project (`type: website`, `render: [cookbook.qmd]`, `toc-location: right`, `theme: cosmo`, `freeze: auto`) renders the single composed page with site chrome and a section TOC into `_site/`. Not wired into the Sphinx tree. **Context:** the docs are RST today but the author prefers QMD and is weighing a QMD *monograph* (publisher interest) — so the cookbook is deliberately Quarto-native, a possible seed for that. Keep it self-contained and portable.
2. **Tables — `greater_tables.GT`.** `_setup.py` defines the generic display verb `qd` wired to `GT` (the author's crisp-HTML table lib), degrading to a plain `display` if `GT` is absent. Pages call `qd(df)`. NB: the installed `GT` takes styling via `config`/`overrides`, *not* the `hrule_widths`/`vrule_widths` kwargs used in the older `hacks/pnl-testing.qmd` `qd2`.
3. **House-book calibration — author owns it; `_setup.py` is the source of truth.** Confirmed: `_setup.py` holds the parameters (`HOUSE_*`) and the calibrated DecL snippet library (`house()`, `OCC_LAYER`, `AGG_LAYER`, `SWING`, `REINST`, `pnl()`); pages import and compose them, and the Solution pretty-prints the resolved program so nothing is hidden. Calibration remains PROVISIONAL.

## [Cookbook-Build-Order]

1. `_setup.py` + `cookbook.qmd` master + all page skeletons. ← this pass
2. Flesh exemplars: `_01_distortions` (gentle intro), `_11_placement` (check-heavy). ← this pass
3. Author calibrates the house book; we flesh the remaining pages page-by-page, author reacting to each ("know it when I see it").
4. Wire the checks (the real work) using the catalog above.
