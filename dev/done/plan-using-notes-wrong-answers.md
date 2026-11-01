# Plan: five wrong numbers found writing the Using pages

`[Using-Notes-Wrong-Answers]`. Drafted 2026-10-31 against aggregate 1.43.1 (`03aaec6`). Two
minor bumps: `[Bounds-Envelopes]` (the two pricing-bounds formulas) and
`[Reinsurance-Bivariate-Structure-Numbers]` (the other three). Agreed by the author on
2026-10-30 (`D:\Projects\Using\bugs\aggregate-review-answers-1.43.qmd`, item 11, "the formulas
need your review"): each step states the formula it implements so the review can be of the
formula, not the code. It runs fourth, after `[Severity-Parameter-Count]` and independent of the
grid plans.

## Goal

Five functions return a wrong number with no flag today. After this plan:

| Item | Today | After |
|:--|:--|:--|
| `PricingBounds` upper bound with a finite cap | `[0.75, 0.75]` on the smallest example; the vertical segment at the cap is dropped | `[0.75, 1.25]`; the capped book's upper bound rises from 126.11 to 152.58 at `P = 1.2 E` |
| `convex_distortion` | a mass of 0.3 at 0 and $g(1/4) = 0.55$ on (0,0), (1/2,0.8), (1,1) | no mass, $g(1/4) = 0.4$ |
| `cede` beside `pc` on one layer | the commission is recorded in `economics` and never booked; the result equals `pc` alone | refused at parse time, as `slide` with `cede` is |
| Bivariate axis sizing with a limited component | the axis is sized on the unlimited marginal, mean 625 against 484.7, with a misleading warning | sized on the marginal after its layer; no warning |
| Structure chart return period, occurrence stage | $1/S(x)$ per claim: 12.1 at an attachment that is a 1.21-year event | $1/(n\,S(x))$, the annual occurrence exceedance |

## Steps

### `[Bounds-Envelopes]`

**1. `PricingBounds` keeps tied vertices.** `bounds.py` 1697 to 1708 sorts the union of the X
and Y breakpoints by $T_X$ and keeps only vertices where $T_X$ strictly increases
(`ok = T > runmax`). Above X's saturation every vertex has $T_X = \operatorname{ess\,sup} X$,
so only the first survives (smallest $p$, smallest $T_Y$) and the vertical segment at the cap,
with the top point $(\max X, \mathrm{TVaR}_1(Y))$ that anchors the upper hull, is lost. With a
finite cap `_extract_pmf` (737 to 745) collapses the tail into an atom at $a$, so X always
saturates while a capped Y keeps rising: every capped call is affected, the lower hull is not.

*Formula.* At a tie in $T_X$ the upper hull takes the vertex with the largest $T_Y$ and the
lower hull the vertex with the smallest, per Y column. The hull code needs strictly increasing
$x$ (`_init_engine` 916 to 928, `_monotone_hull` 617 to 657, `_slice` 971 to 992), so the
selection is made per column before the hull is built, not by a single filter on the shared
vertex list. `AllocationBounds` has the same filter at 1331 to 1337, where ties arise only from
floating-point wobble because the vertices are X's own atoms; it is left as it is, with a test
that a wobble tie does not change its answer.

*Acceptance.* `PricingBounds(X, {'Y': Y})` with $X = \{0: 1/2, 2: 1/2\}$, $Y = \{0: 3/4, 2: 1/4\}$:
`curve_df` holds the vertex at $p = 3/4$, $(2, 2)$, and `bounds(1.5)` is `[0.75, 1.25]`. On the
`cts_port` fixture at `p = 0.99` and `P = 1.2 E = 239.15`, the lower bound stays 119.72 and the
upper is 152.58. The `discrete_pb` fixture (`tests/test_pricing_bounds.py` 59 to 64) is
unchanged, since Y's breakpoint sits below X's saturation; a second fixture with Y's breakpoint
above it is added. `tests/test_fcc_surface.py` 178 and `tests/test_bounds.py` 294 to 302 pass;
the docs example at `features.rst` 2295 is rerun and its numbers recorded.

**2. `convex_distortion` without the spurious mass.** `spectral.py` 4234. The corners are
appended at 4272 to 4278, the knots are `sorted(set(hull.simplices.flatten()))` at 4283, and
the weights solve `lstsq` on `Distortion.tvar_terms(ps)` at 4297 to 4310. The $(0, 0)$ knot
becomes a $p = 1$ column (`tvar_terms`, 986 to 994, evaluates the $s = 0$ column as 0), which
is identical on every knot to the column of the smallest positive knot, so $M$ is rank
deficient and `lstsq` returns the minimum-norm solution, splitting the weight between the two.
That split is the mass. A second defect: the knot set takes both sides of the hull, so
`convex_distortion([0, .3, .5, 1], [0, .1, .8, 1])` makes 0.3 a knot and returns $g(0.5) = 0.897$
instead of 0.8.

*Formula.* The distortion is the upper concave envelope of the points through $(0,0)$ and
$(1,1)$. Its knots are the upper hull's vertices only. The TVaR representation
$g = \sum_i w_i\, \mathrm{TVaR}_{p_i}$ over the knots $s_i > 0$ has $p_i = 1 - s_i$ and weights
determined by the slopes: with slopes $k_0 > k_1 > \dots > k_m = $ the last segment's, the weight
of the knot at $s_i$ is $(k_{i-1} - k_i)\, s_i$, and the $p = 1$ term (a mass at 0) appears only
when the envelope's first segment has a positive intercept, which cannot happen with $(0,0)$ a
vertex. No least squares: the weights are read off the hull. `Bounds.min_envelope` (`bounds.py`
445 to 458), `bagged_distortion` (`spectral.py` 4347, 4350) and `convex_example` (4379, 4397)
follow.

*Acceptance.* The toy gives $g(1/4) = 0.4$ and $g(10^{-9}) < 10^{-8}$; the four-point case gives
$g(0.5) = 0.8$; on the BDD portfolio `min_envelope`'s $g(10^{-12})$ is within a factor of 10 of
the cloud minimum at `JUMP_EPS` (about $2\times10^{-12}$, today $5.9\times10^{-4}$); the two
`convex_example` masses are recorded before and after. `tests/test_bounds.py` 75 to 83 and 213
to 222 pass; new direct tests of `convex_distortion` on the two cases.

**Files.** `src/aggregate/bounds.py`, `src/aggregate/spectral.py`, `tests/test_pricing_bounds.py`,
`tests/test_bounds.py`, a new `tests/test_convex_distortion.py`.

### `[Reinsurance-Bivariate-Structure-Numbers]`

**3. `cede` with `pc` is refused.** `parser.py` `reins_clause` (2348 to 2409) refuses `slide`
with `cede` at 2405 to 2408 (`'slide' replaces the fixed 'cede' commission; give one or the
other`). `pc` with `cede` parses, the Underwriter stores `c_agg` (1622 to 1624, 1964), and the
`expense` branches of `_pnl_builders.py` replace the fixed commission instead of adding it
(`_feature_maps` 1717 to 1720, `_build_variable_consolidated` 1871 to 1875, the two walks at
1918 to 1925 and 1974 to 1981, the occurrence-terms path at 2041 to 2042, 2222 to 2228 and 2307
to 2312). Measured on the test head program with `deposit 1500`: `cede 10% pc 25% after 10%`
and `pc 25% after 10%` both give `est_m` 3940.306; `cede 10%` alone gives 4088.699. The ruling
(review item 11) is to refuse: booking it is a feature. The refusal joins the slide one, with
the same shape: `'pc' replaces the fixed 'cede' commission; give one or the other`. The
structure chart's `cede` annotation (`charts/_emit_structure.py` 287, 311) then never shows a
commission that is not booked. `swing` with `cede` keeps its own message in
`[Using-Notes-Guards-And-Messages]`.

*Acceptance.* The program above raises `ValueError` at parse time with that message;
`test_pc_build` (`tests/test_variable_rating_decl.py` 121) and `test_slide_with_cede_rejected`
(277 to 280) pass; a `test_pc_with_cede_rejected` is added beside them. `decl-testers.agg`
gains the refused line, commented, in the variable-rating section.

**4. Bivariate axes are sized on the layered marginal.** `bivariate.py` `_standalone_marginal`
(1603 to 1605) rebuilds each sizing marginal from the unit spec keeping only `sev_*`, `name` and
`note`, which drops `exp_limit` and `exp_attachment`; `_measure_marginal_window` (1617 to 1671)
then measures the unlimited law while `_marginal_moments` (1552 to 1563) uses the layered
analytic moments, and the guard at 1662 to 1669 warns about the disagreement it created. The
joint itself is right (realized marginal mean 484.75 against the analytic 484.74 on `MV.Indep`
with `100 xs 0` on A); only the sizing is off. The fix is the spec filter: keep `exp_limit`,
`exp_attachment` and the other layer keys (`exp_retention_end` if present).

*Acceptance.* `MV.Indep` with `100 xs 0` on A builds with no `bv-marginal-mean-mismatch`
warning and an axis-0 window no wider than before; the unlimited program keeps its grid. The
guard test (`tests/test_bivariate.py` 1050 to 1060) still fires on its own fixture; a limited
component test is added beside 1025 to 1047.

> **Superseded by the author's ruling of 2026-11-01; see the execution log.** The formula and the change below were not implemented. The return period landed as $1/(1 - P_N(F(x)))$, on a redrawn curve, with no IR change.

**5. The structure chart's occurrence return period is annual.** `charts/_emit_structure.py`:
the occurrence Lee curve is `p_sev_gross` (637 to 646) and the return-period axis is
`reciprocal_of` the probability axis under the document's `return_period_map = 'complement'`
(765 to 769, 790), which the renderers apply as $1/(1 - v)$ (`plots/_chartdoc.py` 702 to 715
and 1333 to 1347; in the API `web/src/charts/reading-map.js` 61 to 66 and
`chartdoc-to-echarts.js` 1484). On the occurrence panel that is $1/S_{\text{sev}}(x)$, one over
the per-claim exceedance. Measured on `agg RPT 10 claims 1000 xs 0 sev lognorm 100 cv 1.5
occurrence net of 750 xs 250 poisson`: $S(250) = 0.08268$, axis 12.10; $n S = 0.8268$, annual
return period 1.21. The aggregate panel (`p_agg_subject`) is annual already.

*Formula.* On the occurrence stage the return period of $x$ is $1/(n\,S_{\text{sev}}(x))$, the
reciprocal of the expected annual number of claims exceeding $x$; `utilities.oep`
(`utilities.py` 767) is the Poisson form of the same quantity. It can be below one year when
$n S > 1$ (several such claims a year), so the axis starts at $1/n$, not 1.

*Change.* `return_period_map` is one value per document (`charts/ir.py` 151 to 172), so the
occurrence axis cannot get a different map without an IR change. The change is per axis: the
map and its scale move onto `ChartAxis` (`return_period_map='annual_occurrence', scale=n` on the
occurrence axis, `'complement'` on the aggregate one), the document meta keeps the aggregate
value for old readers, and the two renderers read the axis first. The API renderer is a
downstream chore in the same cycle, listed in the API's `dev/TODO.md`.

*Acceptance.* On `RPT` the occurrence return period at 250 reads 1.21 and at the aggregate
panel nothing moves; `tests/test_chart_structure.py` 441 to 481 pass and a test pins the
occurrence values at two attachments; the library renderer draws the axis from $1/n$. The
Using page `structure-chart` is rerendered and its sentence on return periods checked.

**Files.** `src/aggregate/parser.py`, `src/aggregate/bivariate.py`,
`src/aggregate/charts/_emit_structure.py`, `src/aggregate/charts/ir.py`,
`src/aggregate/plots/_chartdoc.py`, `src/aggregate/agg/decl-testers.agg`,
`tests/test_variable_rating_decl.py`, `tests/test_bivariate.py`, `tests/test_chart_structure.py`.

## Not in this plan

- Booking `cede` beside `pc` (a feature, ruled out).
- The `AllocationBounds` tie filter beyond a test that it is not affected.
- The API renderer's per-axis return-period map, which follows in the API repository.

## Execution log

### 1.52.0 `[Bounds-Envelopes]`, 2026-11-01

**Names.** One new private helper, `_column_envelope(t, y, side)` in `bounds.py`, which reduces each tied run to its one reachable vertex per column and per side and then calls `_monotone_hull`. `_init_engine` calls it, so `AllocationBounds` passes through it with no ties to reduce. No public name was added or changed. The first choice, `_envelope`, is already a private function in `charts/_emit_bounds.py`, hence the longer name.

**Formula check.** Both formulas were verified by hand before coding. The weight formula needs one convention the plan left implicit: $k = 0$ above the last knot, so the knot at $s = 1$ has weight equal to the last slope. On the toy this gives weights 0.6 at $s = 1/2$ and 0.4 at $s = 1$, and $g(1/4) = 0.4$.

**Recorded numbers.**

| Quantity | Before | After |
|:--|--:|--:|
| Hand example, `bounds(1.5)` | `[0.75, 0.75]` | `[0.75, 1.25]` |
| `cts_port`, `p = 0.99`, lower at `P = 239.15` | 119.72 | 119.72 |
| `cts_port`, `p = 0.99`, upper at `P = 239.15` | 126.11 | 152.58 |
| `cts_port` vertices, distinct $T_X$ | 6,147, 6,147 | 8,165, 6,147 |
| `features.rst` 2295 example, `P = 21,899.22`, lower | 10,799.44 | 10,799.44 |
| `features.rst` 2295 example, `P = 21,899.22`, upper | 10,897.51 | 12,941.39 |
| Toy, mass, $g(1/4)$ | 0.3, 0.55 | 0, 0.4 |
| Four-point case, $g(0.5)$ | 0.897 | 0.8 |
| `convex_example('bond')`, mass | 0.003058 | 0 |
| `convex_example('cat')`, mass | 0.022958 | 0 |
| BDD `min_envelope`, $g(10^{-12})$ against the cloud minimum $2\times10^{-12}$ | $5.9\times10^{-4}$ | $2\times10^{-12}$ |

The `features.rst` "before" comes from re-applying the old filter to the new vertex table. Every `check` error on the capped book is below $10^{-11}$, on the new upper bound as on the old.

**Gates.** `uv run pytest -m 'slow or not slow'` and the same with `-W error::RuntimeWarning`, in `.venv`: 10,408 passed, 1 skipped (`plotly` not installed), each in about 2:47.

**Divergences.**

- **A point outside the unit square is refused.** The plan says nothing on this. The new weights would be negative on such a point, and the old code clipped and rescaled them, which is another silent wrong number. `convex_distortion` raises `ValueError` (with $10^{-9}$ of room on `gs`, since an average of values at most 1 can round above 1).
- **The corners are always included.** The plan's formula says "through $(0,0)$ and $(1,1)$"; the code used to add a corner only when that `s` was absent, then rescale. A supplied $(1, g)$ with $g < 1$ is now overridden by $(1,1)$. A supplied $(0, g_0)$ with $g_0 > 0$ stands and is a true mass of $g_0$, the plan's "positive intercept" case: `[0, .5, 1], [.1, .8, 1]` has mass 0.1, was 0.3.
- **Qhull is gone from `spectral.py`.** The knots come from the existing `_monotone_hull`, imported inside the function because `bounds.py` imports `spectral.py`. Collinear points now return the identity, where Qhull raised a precision error.
- **The `AllocationBounds` test inserts the tie.** A floating-point wobble cannot be produced on demand, so the test asserts `T` strictly increasing on a real book and then hands the engine a duplicate vertex.

**Findings, not fixed.**

- **The plan overstates "every capped call is affected".** A risk that saturates at or below the reference has no tied vertex above it, so its bound does not move. The examples on the Using `pricing-bounds` page are of this kind and keep their numbers (14.098, 18.497, 3.810, 11.811).
- **`uniform` as Y has the same missing segment.** It supplies no breakpoints, so no vertex sits above a discrete X's saturation: on the hand X, `PricingBounds(X, {'U': 'uniform'}).bounds(1.5)` is `[0.625, 0.625]` where the exact upper is 0.75. Recorded as `[Pricing-Bounds-Callable-Y]` in `dev/TODO.md`.

### 1.53.0 `[Reinsurance-Bivariate-Structure-Numbers]`, 2026-11-01

**Rulings by the author, 2026-11-01, asked at review.**

- **The occurrence return period is the annual maximum, on a redrawn curve.** The plan's formula, $1/(n\,S(x))$, is a claim rate. The review's "1.2-year event" and the Using page are $1/P(\text{largest claim in a year} > x)$, and the plan's statement that `utilities.oep` is "the Poisson form of the same quantity" is false. The two read alike (1.21 and 1.22) only because they came from different books: on the page's book at 250 the rate is 0.58 and the annual maximum 1.22, and on `RPT` at 250 they are 1.21 and 1.78. Three routes were offered: the annual maximum on a redrawn curve, the claim rate as planned, and the annual maximum in Poisson form through the plan's per-axis map. The first was ruled.
- **The refusal covers the whole aggregate program.** The plan refuses `cede` beside `pc` on one layer. A `cede` on another aggregate layer is dropped the same way, by `slide` as well as by `pc`: 3,443.66 with or without `cede 10%` on the first layer, with `c_agg` 50 in `economics`.

**Formula, step 5 as ruled.** With $N$ independent claims of distribution $F$, the largest is at most $x$ with probability $P_N(F(x))$, for $P_N$ the frequency pgf, and a year with no claim has a largest claim of zero. The return period of $x$ is $1/(1 - P_N(F(x)))$ years, which is $1/(1 - e^{-nS(x)})$ for Poisson. The largest claim above $x$ implies the total above $x$, so this return period is never shorter than the gross aggregate's at the same $x$. It is at least $1/(1 - P_N(0))$, so at least one year.

**Change, step 5 as ruled.** `_lee_series` draws the occurrence stage from `_largest_claim_mass`, and the document's existing 'complement' map gives the reciprocal. No field, map or renderer changes, `CHART_IR_VERSION` stays 4, and the API needs no chore: its renderer draws the new curve as it stands. A reader sees two changes: the series name is `Largest gross claim in a year`, was `Gross claim`, and `occ_p` is labeled `Annual non-exceeding probability`. The cost the ruling accepted: the probability axis of that panel is annual too, so the area under the curve is no longer per-claim layer loss.

**Names.** New private `_largest_claim_mass` and `STAGE_PROBABILITY` in `charts/_emit_structure.py`. No public name changed. The plan's `scale=n` collided with `ChartAxis.scale`; the ruling removed the field.

**Recorded numbers.**

| Quantity | Before | After |
|:--|--:|--:|
| `cede 10% pc 25% after 10%`, `deposit 1500`, `est_m` | 3,940.306, same as `pc` alone | `ValueError` |
| `cede` alone, `corridor` beside `cede`, for comparison | 4,088.699, 3,864.104 | unchanged |
| Occurrence `cede` beside aggregate `pc`, the effect of `cede` | +50 | +50 |
| `MV.IndepLimited`, sizing marginal mean, against analytic 484.738 | 625.03 | 484.7 |
| `MV.IndepLimited`, axis 0 top, `bs`, buckets | 20,470, 10, 2,048 | 2,046, 2, 1,024 |
| `MV.IndepLimited`, realized marginal mean | 484.750 | 484.745 |
| `MV.IndepExcess`, realized marginal mean, against analytic 581.652 | 580.661 | 581.493 |
| Unlimited program, axis 0 top, `bs`, buckets | 20,470, 10, 2,048 | unchanged |
| `RPT`, return period at 250 and 500 | 12.10, 46.71 per claim | 1.78, 5.19 years |
| The Using page's book, return period at 250, 500, 1,000 | 11.57, 29.07, 87.61 per claim | 1.22, 2.01, 4.90 years |

**Divergences.**

- **Step 5 follows the ruling, not the plan**, as above. The API `dev/TODO.md` chore and the per-axis map under "Not in this plan" are void.
- **The refusal has a second check**, in `_split_reins`, for the other layers; the one the plan names sits with the slide check in `reins_clause`, which the plan cites at 2348 to 2409 and now sits at 2390 to 2453. The `slide` refusal widened with it.
- **The refused lines in `decl-testers.agg` are three**, the plan's plus `slide` and the other layer.
- **The `MV.Indep` programs in the plan are named `MV.IndepLimited` and `MV.IndepExcess`** in `decl-testers.agg`. The excess case is an addition, since the plan's example attaches at 0.
- **The zero-modified test uses `zm 0.2 !`.** The pgf takes `base_mean`, not `n`. Without `!` the count of 10 is the realized mean and the base mean is 12.5; with it the base mean is 10 and the realized mean 8.

- **Two render baselines moved, which the plan did not list.** `test_chartdoc_matches_baseline` failed on `structure` and `structure_log`, as it should. Rendered to scratch first: `distortion`, `agg` and `matrix` are identical, and in both structure pictures only the occurrence Lee panel differs (its curve and its axis label). The two images were looked at and regenerated with `--regen`. They are the author's approved pictures, so they are for the author to confirm.

**Gates.** `uv run pytest -m 'slow or not slow'` and the same with `-W error::RuntimeWarning`, in `.venv`: the first run failed only on the two baselines above; after regenerating them, 10,432 passed, 1 skipped (`plotly` not installed), each in about 2:50. `tests/capture_spec_snapshot.py` reran with no change to `expected_specs.json`.

**Findings, not fixed.** The stale open item in `D:/Projects/Using/CLAUDE.md` (dated 2026-10-30) still reports the `PricingBounds` and `convex_distortion` defects as open, and `dev/issues-arising-from-using-notes.md` lists items 1, 2, 3 and 6 as open.
