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

(filled at execution: versions, the recorded numbers, divergences)
