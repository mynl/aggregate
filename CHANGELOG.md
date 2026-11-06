# Changelog

## API stability, and how to read these notes

**This section is standing policy, not a release entry.** It applies to every version below and to 1.0 itself.

The public surface has two tiers.

**Stable.** `Aggregate`, `Portfolio`, `PnL`, `Severity`, `Frequency`, `Distortion`, `BivariateAggregate`, `Underwriter`, `build`, `qd`, and the DecL grammar in `decl.lark`. From 1.0 onward a documented name here keeps its meaning: a breaking change waits for a major release and is preceded by a deprecation period. Breaking changes recorded in the alpha entries below happened *before* that promise took effect, which is what the alpha series was for.

**Provisional, in the sense of [PEP 411](https://peps.python.org/pep-0411/): `aggregate.charts` and `aggregate.exhibits`.** These two modules are **not part of the 1.0 API contract**. Their APIs may change in a *minor* release, with no deprecation period. That covers the chart IR schema and its document hash, the chart registry and the set of charts that exist, and on the exhibits side the exhibit names, block structure, captions, row flags and the `Perspective` vocabulary.

They are additive side projects to the 1.0 release rather than part of it. Their dependencies point inward, so both import from the core and the core does not import them, and nothing in either module touches an existing class. They therefore cannot destabilize or delay 1.0, and 1.0 ships whether or not either is finished. The one edge into pre-existing code is per-chart conversion of a bespoke plot to the emitter-plus-renderer form, each conversion gated by a before-and-after image diff and deferrable past 1.0. Explicitly post-1.0: conversion of the charts the app does not use, full convergence on matplotlib rendering the IR, the `INSURED` and `REINSURER` perspectives, and any exhibit meta-language.

They are public and not underscore prefixed on purpose. Use them, and report what does not fit: that feedback is how a provisional module graduates to stable in a later minor release. Full statement in `docs/3_reference/3_x_API_Stability.rst`.

**A note on `greater_tables`.** It became a plain dependency at `1.0.0a229` rather than an optional extra, so the exhibit surface never raises `ImportError` on a supported interpreter. That is a fact about installation and not a stability promise. It does not move `aggregate.exhibits` into the 1.0 contract.

**Numbers are a separate question from names.** A correctness fix changes results within any release and is called out in the entry that makes it. A number that was wrong is not an interface to be preserved.

---

## 2.1.0a2 - 2026-11-06

**[Gini-Rename] `Distortion.gini_p` is renamed `gini`, with no alias.** The property, the `gini_p` column of `distortion_df` and of every `evaluation_df`, the `gini_p` block of the evaluate exhibit and its captions, the `formats-*.yaml` keys and the row of `Distortion.info` all read `gini`, and `gini_p` raises `AttributeError`. This breaks the stable `Distortion` surface inside a minor series, deliberately and with no deprecation period, because nothing outside the author's own work reads the name yet. Code that reads `gini_p` changes to `gini`; no number moves.

## 2.1.0a1 - 2026-11-06

**[Reference-Label-Leak] A DecL reference no longer overrides the label and tags of the statement that refers to it.** `agg B as "b" agg.A` has label `b`, and `agg C agg.A` has label `C` (it was `A`'s); neither takes `A`'s tags. The same holds for `2 @ agg.A`, for a pnl engine written `agg E agg.A` (the loss leg is `E`'s), for an aggregate whose severity is `sev.S`, and for `sev S2 sev.S`. The severity inside an aggregate written with `sev sev.S` or `sev.S` takes `S`'s stored label as `labels.severity` unless the clause has its own `as`. `agg.A` with its own reinsurance, `less agg.A` in a pnl, and a portfolio unit written `agg.A` are `A` and keep its label. No `library.agg` entry changes label or tags.

## 2.0.0 - 2026-11-03

**Release 2.0.0, the first since 1.11.0, covers 1.11.1 to 1.63.2.** New: event and year loss tables (`elt`, `yelt`) as direct Aggregate input, inline or from CSV, TSV, Parquet or Feather, with secondary uncertainty quantized by the Rust `eltq` through `pip install aggregate[elt]` (1.12.0 to 1.21.0); `chfagg` and `AggregateCHF`, a law computed directly from its characteristic function, covering stable (with tilting), Tweedie, normal inverse Gaussian, variance gamma and tempered stable, with pricing, sharpen, and use as a severity through `sev chfagg.NAME` (1.27.0 to 1.41.0); `stable` and `tweedie` statements that translate parameterizations and store named laws, with arithmetic on stable laws (1.22.0, 1.23.0, 1.42.0, 1.43.0); exact dyadic lattice buckets for discrete severities, Portfolios and bivariate axes (1.24.0 to 1.26.0); closed-form layered moments for beta, Lomax and a shifted Pareto (1.17.0); `library.agg` examples of event and year loss tables and two more characteristic function laws (1.63.2). Changed: `bs` and `log2` move for discrete books on a lattice, and `density` values scale with `bs`, with probabilities at the atoms unchanged. Thanks to @KasraBashirioskooei for two pull requests: the deductible form in the layer phrase (#23) and the layer pdf at the limit (#24).

**Fixed: a correctness pass (1.43.1 to 1.63.1), and numbers move.** Reinsured and layered discrete programs that returned the right mean and the wrong distribution now return the right one (1.45.0 to 1.48.0). A `sev` clause takes exactly the numbers its family takes, so `norm 100 cv 0.3` is N(100, 30), was N(100, 1) (1.51.0). Layer moments are correct far in the tail, and a shifted law `c * law + b` takes its moments from the law with the shift applied after, so its skewness is right (1.49.0, 1.50.0). `PricingBounds` upper bounds rise where the reference saturates, and `convex_distortion` no longer puts a false mass at zero (1.52.0). Validation checks each reinsured view, and a build that fails emits one `ValidationWarning` where it was silent before (1.56.0 to 1.58.0). A DecL list of any length parses (1.43.1). Three `pedagogy.py` functions and `ClassicalPremium.calibrate` run again, and the Fischer premium takes `fischer_power` (1.63.0). Earlier fixes: a signed book of lattice units holds its mass (1.25.0); `Tweedie.fz` returns a law for p > 2, p < 0 and p = inf (1.31.0); a quoted label keeps `#` (1.11.2).

**Breaking, hence 2.0.** New reserved words: `elt`, `yelt`, `no2u`, `unit`, `stable` and `chfagg`, and a name can no longer begin `stable.`, `tweedie.` or `chfagg.`. A `sev` clause refuses too many or too few shape parameters for its family: write `expon` for `expon 1`, and `[lognorm expon] [0.5 _]` for `[lognorm expon] [0.5 1]` (1.51.0). `price` takes exactly one of `p` and `a`, `p > 1` raises, and `Distortion.ccoc` is keyword-only (1.59.0). `cede` beside `pc` or `slide` raises (1.53.0), and so does `cede` beside `swing` (1.59.0). `dsev` probabilities that miss 1 by more than a relative 1e-5 raise (1.44.0). `ValidationWarning` is an error under a warnings-as-errors filter, and `Validation.passes` is false for a failing reinsured view (1.56.0, 1.58.0). `cll` leaves the default `available_distortions()` (1.62.0). `TEST_SUITE_FILENAME` and `Underwriter.test_suite_file` are removed, and `Underwriter.interpret_file` requires a filename (1.61.0). Message text changed throughout, so code that matches on message strings needs the new wording (1.60.0). `aggregate.ft.FourierTools` is deprecated in favor of `AggregateCHF` (1.34.0).

## 1.63.2 - 2026-11-02

**[Library-Loss-Tables-And-Laws] `library.agg` gains six entries: its first event and year loss tables, and two more characteristic function laws.** A new section, Event and year loss tables, holds `EventLossTable` (eight events, `[id rate mean max]`), `EventLossTable.SecondaryUncertainty` (the six column form, which needs `pip install aggregate[elt]`), `EventLossTable.SharedEvents` (two units under an occurrence layer) and `YearLossTable` (an annual aggregate cover), all four tagged with the new `topic:elt`. `NormalLaw` (`30 * norm + 20`) and `GammaLaw` (`100 * gamma 4 - 200`, support from -200) join `StableLaw` under Numerical character. No existing entry changes.

## 1.63.1 - 2026-11-02

**[Deductible-Forms-Follow-Ups] A layer's `pdf` marks the point mass at the limit with `inf`, conditional and unconditional.** The conditional form returned 0 at the limit. The unconditional form put the marker at the detachment `a + y`, a ground-up loss, and returned a nonzero density between the limit and the detachment, where it now returns 0. `severity_error_analysis` sets a marker to 0 in its quadratic term, so `h2_adj` is finite when a grid point lands on the limit (it was `nan` for a franchise) and unchanged for an ordinary layer. The density series of `chart_severity` for an ordinary layer carries one `inf` at the limit, as a franchise's already did. Contributed by @KasraBashirioskooei (#24).

## 1.63.0 - 2026-11-02

**[Pedagogy-Repairs] Three broken `pedagogy.py` functions run again, and `ClassicalPremium` takes a keyword-only `fischer_power`, default 2.** `similar_risks_example()` raised `TypeError` from `Bounds.plot_envelope(axs=...)`; it now draws the two-panel envelope and returns its frame. `plot_similar_risks_graphs(..., p_reg=0.9)` raised `AttributeError`; it now prices the new portfolio by $\mathrm{TVaR}_p(X \wedge a)$ at $a = q_{p_{reg}}(X)$, and `p_reg=1`, the default, gives the same numbers as before. `ClassicalPremium.calibrate` and `price(method='Fischer')` raised `AttributeError` for any input; the Fischer premium is $\mathsf{P}X + c\,(\mathsf{P}[(X - \mathsf{P}X)_+^r])^{1/r}$ with $r$ = `fischer_power`, and power 1 equals the `Dutch` premium. `decl-testers.agg` has a new section PED with three programs.

## 1.62.0 - 2026-11-02

**[Research-Distortions] `cll` leaves the default `Distortion.available_distortions()`, which is now `('ccoc', 'ph', 'wang', 'dual', 'tvar')`, equal to `DEFAULT_CALIBRATION_DISTORTIONS`.** No calibration or table moves, since calibration already defaulted to those five. Code that relied on the default list holding `cll` must pass `pricing=True, strict=False`, and `Distortion.distortions_from_params` at its defaults no longer reads a `cll` row. `available_distortions(pricing=True, strict=False)`, `available_distortions(pricing=False)`, the DecL distortion words, `Distortion('cll', ...)` and `calibrate_distortions(..., names=['cll'])` are unchanged. The research families `cll`, `clin`, `lep` and `ly` are described under a new heading, Research distortion families, on the Distortion reference page and in the `Distortion` docstring; the docs are pending a rebuild.

## 1.61.0 - 2026-11-01

**[Scaffold-Retirement] The SLY-parity corpora `_test_suite.agg` and `_test_suite2.agg` are deleted, and the test suite parses, round-trips and snapshots `decl-testers.agg` instead.** No number or default moves, since `build` already loaded `library`. `tests/data/expected_specs.json` has 647 entries, up from 209, keyed by kind word and name rather than statement text. Section TS of `decl-testers.agg` holds the 23 programs tests still need, including the only exercise of six grammar rules. `library.agg` joins the unparser corpus. `tests/data/bucket_baseline_*.csv` are deleted, and the maintenance scripts default to `library`.

**Breaking.** `aggregate.config.TEST_SUITE_FILENAME` and `Underwriter.test_suite_file` are removed. `Underwriter.interpret_file` requires its `filename`; a bundled file is `build.default_dir / 'library.agg'`. `Underwriter(databases='_test_suite')` warns and loads nothing, and `load('_test_suite')` raises `FileNotFoundError`. **In this repository, a program appended to `decl-testers.agg` fails `test_spec_matches_snapshot` until `uv run python tests/capture_spec_snapshot.py` records it.**

## 1.60.1 - 2026-11-01

**[Gini-P-Negative-Return] `gini_p`, and so `calibrate_distortions`, works on a `ccoc` with a negative return, which 1.59.0 broke.** A premium below expected loss calibrates `ccoc` to `r < 0`, and from 1.59.0 `calibrate_distortions(coc=-0.5, p=0.99)` raised `TypeError` from a complex standard deviation. `gini_p` is `r / (1 + r)` again, -1 here. `stats_df` on such a distortion also stops raising, and a negative closed-form variance gives a `nan` standard deviation. The `dual` calibration at such a premium, which has no solution and gives a `nan` row, no longer emits a `RuntimeWarning`. No other number moves.

## 1.60.0 - 2026-11-01

**[Using-Notes-Messages] Message text loses its dashes, internal labels and history; `grossnet` and `grossceded` refusals use their own word.** No number moves. No raised, warned, logged or explanation string holds a dash as punctuation, including `bs_explain`, `tail_explanation`, the P&L construction explanation and `IgnoredDecLClauseWarning`, whose text is now `V1: ignoring the aad clause: a plain 'agg' ...`. Refusals drop `[reins-premium]`, `[reins-single-layer]`, `[reins-one-clause]`, `decision 0` and `the engine ceiling`, and the refusal of a reinstated or annual-term layer on a tower now gives the remedy `Write one occurrence layer`. `p=1 does not name an asset level` is now `p=1 does not fix an asset level`, and `unit rows carry different rates` is now `have`. Code matching these strings needs the new text. Docstrings corrected: `PnL.tvar` is the upper tail, the `_pricing` module has `calibrate_distortions`. Docs corrected: the Bounds classes are top-level names, and `so` and `of` are not `po` synonyms.

## 1.59.0 - 2026-11-01

**[Using-Notes-Guards] Inputs that gave a wrong answer or a raw error are refused with a message, `price` takes `a=`, `ccoc` prices at a finite `p`, every distortion has a `gini_p`, and `distortion` takes `as`.** Now `ValueError`: `zm` outside [0, 1) (`zm -0.2` was a mass of -0.2 reading `not unreasonable`); `pollaczeck_khinchine` at `rho <= 0`; a `po` share above 100% (`6 po 5`, `150% po 100`); `cede` beside `swing` on one layer; a `dbvsev` atom off a pinned grid, with the `log2` it needs; an unknown severity family, with close names in `err.report.suggestions`; an unmatched bracket, with its position in `err.report`. `a.price(0.99, Distortion('ccoc', r=0.1))` prices at $P = (L + ra)/(1 + r)$, while `p = 1` and a bare `apply_distortion` on an unbounded law still refuse. New: `distortion D as "my distortion" ph 0.6`, with `name` the family and the text in `label`; `Distortion.gini_p` is a property, finite for every family, where `bitvar`, `wtdtvar`, `beta`, `minimum`, `mixture` and the rest were `nan`. `help(values=...)` calls no method, and the default evaluates no property. `repr` and `info` of a `dfreq` count work. Numbers move only in `lep`'s closed form, which ignored the cap on `g`: `Distortion('lep', r0=0.03, r=0.15)` has `gini_p` 0.10866, was 0.10869, and its closed-form `var`, `std` and `cv` are blank.

**Breaking.** `Aggregate.price`, `AggregateCHF.price` and `Portfolio.price` take exactly one of `p` and the new keyword `a`. `p > 1` raises: `Aggregate.price` priced it at the top of the grid and `Portfolio.price` read it as assets, so write `port.price(distortion=d, a=assets)`. `p = 1` on an unbounded risk raises on all three, and an `a` above an unbounded risk's grid raises. `Distortion.ccoc` is keyword-only, `Distortion.ccoc(d=...)` or `r=...`. An unmatched bracket raises `ValueError` from `UnderwritingLexer.preprocess`, was a bare `AssertionError`, so the API drops the catch in `split_statements` and updates `test_an_unbalanced_bracket_is_a_422_not_a_500`.

## 1.58.0 - 2026-11-01

**[Validation-Warning] A build whose verdict fails emits one `ValidationWarning` with the verdict.** New `constants.ValidationWarning` and `constants.validation_warning_suppressed`; silence it with `silence_warnings(ValidationWarning)`. `agg L 3 claims 1500 xs 250 dsev [1000 2000 3000] fixed` at `bs` 1000 now warns `L: fails sev mean, agg mean`, once per session for a name and its flags. A Portfolio warns once for itself and its units, and `sharpen` once for the grid it settles on. Code that treats any warning as an error will see this one: 56 of the 338 entries in `library.agg` and `reproductions.agg` earn it, all failures that were silent before. No number moves. The verdict costs about 1 ms an update, with no FFT.

## 1.57.0 - 2026-11-01

**[Verdict-Moment-Floors] A skewness of rounding is not failed, and an analytic moment the library could not compute is reported.** New `moments.third_central_moment` and `moments.skewness_is_testable`, and new flag `Validation.ANALYTIC_UNDEFINED`, read as `analytic cv, skew undefined; not unreasonable` with `passes` false. Numbers move only where the third central moment is within 1e-15 of the terms subtracted, where it is now 0: `1000 claims dsev [999 1000 1001] fixed` had a skewness of 0.015 and `100 * gamma 2 + 100000000` one of 94.9, and no grid moves on 1,690 programs. Below 1e-12 the skewness stays reported and is not tested, so `1 claims dsev [115 120 125] fixed` and `DISC.RoundedProbs` read `not unreasonable`, and `WindowedSimple` leaves the known failures. A `nan` from a zero mean, a zero variance or an infinite moment is not flagged.

## 1.56.0 - 2026-11-01

**[Verdict-Reinsured-Views] The verdict checks the reinsured view against the exact image of the subject on its grid, and `Validation.passes` stops treating reinsurance as a pass.** Every reinsured verdict changes: a clean one reads `reinsurance; subject not unreasonable; net not unreasonable`, was `reinsurance; subject not unreasonable`, and `agg X 1 claims dsev [1000 2000 3000] occurrence net of 1500 xs 250 fixed` at `bs` 1000 reads `reinsurance; subject not unreasonable; net fails sev cv, agg cv`. New flag `Validation.VIEW`, and `explain_validation` takes `view=`. `passes` is now false for a failing subject or view, so seven library entries move into the known failures, among them `CommAuto.Exposure` and `USXOLTower`. A Portfolio with a reinsured unit checks its total against the sum of its units' cumulants instead of skipping it. No number moves.

The API must follow: `validationState` in `web/src/main.js` compares the text after `reinsurance; subject ` with `not unreasonable`, so it grades every clean reinsured object as a warning until it reads the second clause.

## 1.55.0 - 2026-11-01

**[Stable-Completed-Mean] A tilted stable law with a mean reads `not unreasonable`, judged on its completed mean.** The verdict changes: `chfagg X stable 1.5 1`, `2 * stable 1.5 1 + 10` and `stable 1.1 1` read `fails agg mean` before and `not unreasonable` now. The `error` of the mean row in `stats_df`, the `Err EX` of `validation_df`, the `info` row and `validation_score` now use the window mean plus the closed-form tail, with errors from 1e-7 to 4e-6; `empirical` and `est_m` stay the window mean. An untilted grid (`hints{tilt=False}`) still fails, and `tvar` and `lev` beyond the window stay on `[CHF-Tail-Completion]`.

## 1.54.0 - 2026-11-01

**[Structure-Occurrence-Lee-Per-Claim] The occurrence curve on the structure chart is the per-claim severity again, reversing that part of 1.53.0.** Numbers move back. `chart_structure(lee=True)` gives a return period in claims that does not depend on the claim count: 11.6 at 250 on the Using page's book, was 1.22 years. The series is `Gross claim`, `occ_p` is `Non-exceeding probability`, and the `structure` and `structure_log` render baselines return to their earlier images. The `cede` refusal and the bivariate axis sizing of 1.53.0 stand.

## 1.53.0 - 2026-11-01

**[Reinsurance-Bivariate-Structure-Numbers] A fixed commission beside `pc` or `slide` is refused, a bivariate axis is sized on the marginal after its layer, and the occurrence return period on the structure chart is annual.** Numbers move. `chart_structure(lee=True)` draws the occurrence stage as the largest gross claim in a year, $P(M \le x) = P_N(F(x))$, so its return period is $1/(1 - P_N(F(x)))$ years for every count family: on 20 Poisson claims an attachment reading 11.6 now reads 1.22. The series is now `Largest gross claim in a year` and `occ_p` is `Annual non-exceeding probability`; `CHART_IR_VERSION` stays 4, the API needs no change, and the `structure` and `structure_log` render baselines move. A limited bivariate component no longer warns of a mean mismatch, and its grid is finer: `100 xs 0` on `MV.Indep` has axis 0 to 2,046 at `bs` 2, was 20,470 at `bs` 10; the unlimited program's grid is unchanged.

**Breaking.** `cede` beside `pc` raises at parse time, and so does a `cede` on another layer of an aggregate program carrying `pc` or `slide`. These programs parsed and dropped the commission with no flag. A `cede` on the occurrence stage, or beside `corridor` or `swing`, is unaffected.

## 1.52.0 - 2026-11-01

**[Bounds-Envelopes] `PricingBounds` keeps the vertical segment where the reference saturates, and `convex_distortion` returns the upper concave envelope with no mass at zero.** Numbers move. A `PricingBounds` risk that saturates above the reference, usual with a finite cap, has a higher upper bound: `[0.75, 0.75]` becomes `[0.75, 1.25]` on the smallest example, and 126.11 becomes 152.58 on a capped book at `P = 1.2 E`; lower bounds do not move, and `curve_df` gains the tied rows. `convex_distortion` on (0, 0), (1/2, 0.8), (1, 1) has mass 0, was 0.3, and $g(1/4) = 0.4$, was 0.55; a point below the envelope no longer moves the curve. The false mass also leaves `Bounds.min_envelope`, `bagged_distortion` and `convex_example` (`bond` 0.0031 and `cat` 0.0230, both now 0). A point outside the unit square now raises `ValueError`, and collinear points return the identity rather than a Qhull error.

## 1.51.0 - 2026-11-01

**[Severity-Parameter-Count] A `sev` clause gives a family exactly the numbers it takes, and the `mean cv` form works on a family with no shape.** Positional numbers are the family's shapes; scale and location are `c *` and `+ b`. Numbers move: `norm 100 cv 0.3` is N(100, 30), was N(100, 1); `expon 10 cv 1` has mean 10, was 11; `uniform 10 cv 0.2` has mean 10 and standard deviation 2, was mean 10.5. New DecL: `_` holds the place of a family with no shape in a vector of families, `[lognorm norm] [0.5 _]` and `[lognorm beta] [0.5 2] [_ 3]`. A `mean cv` the family cannot reach raises at build where it gave `nan` or the wrong law: `norm -100 cv 0.3`, `norm 100 cv 0`, `beta 2 cv 0.2`, `pareto 100 cv 0.5`. `Severity` raises on `sev_a` for a family with no shape, `sev_b` for a family of at most one, an unset shape on a two-shape family, and a mean or cv alone on a family with no shape.

**Breaking.** `norm 100 30`, `expon 5`, `gamma 4 25`, `beta 2`, `burr`, `gamma`, `burr 100 cv 0.5` and `cauchy 10 cv 1` raise at parse, each with the spelling to use where there is one. They used to drop the number or return `nan`. Rewrite `expon 1` as `expon`, and `[lognorm expon] [0.5 1]` as `[lognorm expon] [0.5 _]`; rewritten programs keep their density bucket for bucket. The published paper's `[expon pareto] [1 2.5]` is one of these.

## 1.50.0 - 2026-11-01

**[Shift-Removed-Moments] A shifted severity `c * law + b` has the moments of the law, with the shift applied after.** Numbers move: `100 * gamma 2 + 10000` has skewness 1.41421, was 1.387; `+ 1000000` 1.41430, was `nan`; `100 * weibull_min 2 + 100000` 0.63111, was 2,602; `100 * norm + 500` a mean of 500.00000535, was 499.99995. Of the 30 shifted programs measured, 13 failed and 6 had a `nan` skewness; 26 now read `not unreasonable` and none is `nan`. The four that fail are at a shift of 1,000,000, the rounding limit of about epsilon `(b / sd)^3`. No `bs` or `log2` moves on any of the 327 corpus programs, and two false failures clear. A layer, franchise or disappearing deductible on a shifted law takes the same route. `feasible_bucket` and `size_biased_cdf` no longer read a shifted law as an unshifted one, so `InfeasibleGridWarning` moves: `100 * lognorm 2 + 1000` stops warning and `100 * lognorm 3 + 1000` still warns. New attribute: `Severity.fz0`, the law without its shift, or `None`. Splices, reflected and signed laws, `cantor`, `SeverityElt` and `SeverityCopy` keep their own paths.

## 1.49.0 - 2026-11-01

**[Analytic-Layer-Tail-Cancellation] Layer moments on `lognorm`, `gamma`, `expon`, `pareto`, `lomax` and `beta` are right wherever the layer attaches.** Numbers move: `10 xs 1000` on `lognorm 1` goes from conditional moments 9.64, 111.2 and -14,784 to 9.657, 95.44 and 948.8, and `10 xs 1e5` on `100 * lognorm 1` from 8.28, 171,386 and -1.7e10 to 9.996, 99.95 and 999.5 and now reads `not unreasonable`. A far-tail franchise or disappearing deductible gets the same fix (`100 xs 1000 to 1100 disappearing` on `lognorm 1`: third moment -6.48e6 to 953,304). A layer at or below the median with attachment up to 16 times its width is unchanged to the bit. Thin layers move below the median as well (third moment by 1e-8 at attachment 17 times the width, 4e-3 at 1,000 times), and other layers past it by up to 6e-9 for a survival probability of at least 1e-3. `moms()` with a zero limit returns zeros rather than raising.

## 1.48.1 - 2026-11-01

**[Correctness-Oracle] Discrete programs are now checked in the test suite against an oracle that shares no code with the sizer.** The new `tests/lattice_oracle.py` enumerates the law of a program and gives the smallest `log2` that holds it, with `nttagg` as the reference where a program is in its scope. `tests/test_lattice_oracle.py` runs about 120 programs in the fast suite, and the full 858 and a 718-program comparison of each unit alone and as a one-unit Portfolio under `slow`. Seven reinsured programs that need a window are named as known residuals, so any new wrong answer fails. No library behavior changes.

## 1.48.0 - 2026-11-01

**[Lattice-Infeasible-Warning] A lattice book whose exact grid does not fit the `log2` cap warns, and the verdict says so.** `agg AWK dfreq [1:100] dsev [1000 1001]` still builds at `bs` 2, `log2` 16, and now raises the new `LatticeGridWarning` with the `log2` to pass and reads `grid infeasible for this lattice (needs log2 17 at bs 1); fails sev cv, agg skew`; `build(..., log2=17)` is exact with no warning. A pinned `bs` and a reference severity have no reading. `bs_description`, `bs_explanation`, `validation_explanation` and `bs_window_df` carry the same numbers, on a Portfolio as well. The lattice search now goes to `max(16, log2)` levels, so `dsev [1 200000]` at `log2` 18 finds its step. No shipped corpus program moves.

## 1.47.0 - 2026-10-31

**[Portfolio-Follows-Unit-Grid] A one-unit Portfolio takes its unit's grid verbatim, and a larger book's grid holds its largest severity.** `port P agg A dfreq [1] dsev [3]` moves from `bs` 1, `log2` 1, mass 0 and `fails pmf deficit 1` to `bs` 1, `log2` 2 and the exact law, the same object as `agg A dfreq [1] dsev [3]`, and this holds for signed, reinsured and continuous units. A book with a reinsured unit is no longer windowed. A lattice book that fits nothing at the moment fit is tried again at the sum of its unit windows before the lattice is given up. Grids move on 3 of the 699 shipped corpus programs, all one-unit Portfolios, none larger.

## 1.46.0 - 2026-10-31

**[Window-Holds-Severity] A grid at a lattice bucket spans a claim as well as the aggregate band, and a book under an aggregate cover is not windowed.** `100 claims dsev [1000 1001] fixed` moves from `bs` 2, `log2` 16 from 0, which split the atom at 1001 and read `fails sev cv, agg cv`, to `bs` 1, `log2` 10 from 99,640 with the exact law. `3 claims dsev [8 10] fixed aggregate ceded to 2 xs 4` moves from returning 12 to returning 2. On the study's Aggregates there are no silent wrong answers left. A severity with no lattice sizes as before, and the `windowed` row's note now holds `sev_hull=`, the new `Aggregate._severity_hull`, in place of `sev_hi=`.

## 1.45.0 - 2026-10-31

**[Lattice-Views] A discrete severity's lattice step is taken over every view it has: gross after any layer, ceded and net of each occurrence layer, and an aggregate cover's points.** The new `Aggregate._severity_views()` feeds `_severity_atoms`, so `_severity_lattice`, `_pooled_lattice` and `sharpen` follow. Reinsured and layered discrete programs that returned the right mean and the wrong law now return the right law: `1 claims dsev [1000 2000 3000] occurrence net of 1500 xs 250 fixed` moves from `bs` 1000 with outcomes 0, 1000, 2000 to `bs` 50 with 250, 500, 1500. On the study's 858 programs, silent wrong answers fall from 68 to 5. Grids and numbers move on 13 of the 688 shipped corpus programs, all reinsured, layered or signed; a share off the dyadic lattice, such as `30% po` of integers, now has no lattice and takes the continuous estimator.

## 1.44.0 - 2026-10-31

**[Dsev-Probability-Normalization] `dsev` and `xps` probabilities must sum to 1.** DecL refuses a sum outside a relative 1e-5 with a `ValueError` giving the sum, so `dsev [1 2] [0.3 0.3]`, which built before, now stops. A sum inside that tolerance is normalized at construction, so `100 claims dsev [999 1000 1001] [0.333333 0.222222 0.444444] fixed` moves from sd 13.29 and skew -42.6 to the correct 8.749 and -0.0213. `format_program` still shows the probabilities as written. A direct `Severity('dhistogram', ...)` outside the tolerance logs an error and keeps its sum, as before, which is the route a reference with `hints{normalize=False}` uses.

## 1.43.1 - 2026-10-30

**[Long-DecL-Lists] A DecL list of any length parses: `UnderwritingTransformer` derives from `lark.visitors.Transformer_NonRecursive`.** A `dsev` vector of about 500 numbers, or a list of comparable length under any other left recursive rule such as inline `elt` rows, raised `RecursionError`. A vector of 40,000 numbers now parses in about 8 seconds. Specs and results are unchanged.

## 1.43.0 - 2026-10-30

**[Stable-Arithmetic] `stable.NAME` references take `+`, `-`, `*`, `/`, unary minus and `@`, and `stable NAME <expression>` stores the result.** An expression alone returns an anonymous `Stable`, has kind `conversion` and stores nothing. `stable z <expression>` stores the four computed numbers, bound at parse time. Each occurrence of a reference is an independent copy: `stable.x + stable.x` has scale `2 ** (1 / alpha)`, `2 * stable.x` has scale 2, and `t @ stable.x` is the sum of `t` copies. New in Python: `Stable.__truediv__` and `Stable.__rmatmul__`. A `RecipeNotFound` for a name holding a minus sign, of any kind, now gives the spaced spelling, and a parse error no longer suggests a `builtin_*` terminal name. See `dev/done/plan-stable-expressions.md`.

## 1.42.0 - 2026-10-30

**[Named-Conversion-Statements] `stable NAME ...` and `tweedie NAME ...` store a law in the recipe base, referenced as `stable.NAME` and `tweedie.NAME`.** Both take the conversion forms with an optional `as` label and the trailer, and store canonical numbers under the new kinds `stable` and `tweedie`: the S1 `alpha`, `beta`, `scale`, `loc`, and `p`, `mean`, `dispersion`. `Stable` and `TweedieConversion` gain `name`, `label`, `program` and `from_spec`, and the string of a named law leads with its name. `library.agg` gains `stable Cauchy`, `Levy`, `Landau` and `Holtsmark`. A bare name held under two kinds now raises the `KeyError` listing its dotted spellings, in `build` and `preview`, where a parse error appeared before. `TweedieConversion` no longer has `__slots__`. See `dev/done/plan-stable-expressions.md`.

Breaking: the prefixes `stable.` and `tweedie.` can no longer begin a name. aggregate-api needs `[aLL-Named-Laws]`: its `test_examples` fails on the two new statement heads until then.

## 1.41.0 - 2026-10-30

**[CHF-Loss-Lab-Ripple] A `chfagg` has the `sharpen` exhibit once a probe has run and a read-only `exp_premium` of 0, and a calibration on a survival function that is not positive and decreasing raises `ValueError` rather than a bare `AssertionError`.** The exhibit reads `sharpen_df`, with the score grid first under the insurer perspective and captions for a law's probe of at most two cells. `exp_premium` is `Aggregate`'s reading of an exposure stating no premium. The refusal applies to every kind and reaches `evaluate(P)` with no anchor on a ringing grid, such as `tweedie 1.5 100 2`; anchored at `p=` that law answers. All three are what the Loss Lab needs: its Sharpen leaf serves the exhibit, its Evaluate form asks for a premium when `exp_premium` is 0, and its Evaluate route turns a `ValueError` into a 400.

## 1.40.0 - 2026-10-30

**[CHF-Accessor-Parity] A `chfagg` has `Aggregate`'s cheap members: `value_type`, `prob_eq_0`, `spec`, `json`, `density`, `sample`, `price`, `price_ccoc`, `price_pentagon_ex`, `evaluate`, `apply_distortion` and `__str__`.** Each is `Aggregate`'s thin delegation to `_pricing` or `density_df`, without reinsurance views; `evaluate` needs the premium `P=`, since a `chfagg` carries none, and `value_type` is read-only `'loss'`. `density_df` gains `log_p`, `exlea`, `e`, `exgta` and `exeqa` with `Aggregate`'s guards, in `Aggregate`'s column order, and `var_dict` takes the `var` and `tvar` kinds `price` passes. On one grid `25 * gamma 4` and `agg dfreq [1] sev 25 * gamma 4` price, evaluate and complete the pentagon to 3e-11. A distortion is refused on a grid whose frequency truncation rings, as `choquet_weights` refuses negative masses: `25 * gamma 4` at `hints{log2=12; bs=1/4}`, and `tweedie 1.5 100 2`, whose density jumps at 0+. Last of six steps of `dev/done/plan-characteristic-function-aggregate-parity.md`; the Loss Lab's capability test for a `chfagg` (six exhibits, `can_sharpen` and `can_evaluate` now true) and its `gamma 4 25` fixtures need updating.

## 1.39.0 - 2026-10-30

**[CHF-Grid-And-Tail-Reports] A `chfagg` has `Aggregate`'s grid sizing and tail reports: `bs_window_df`, `bs_description`, `bs_explanation`, `tail_class`, `tail_behavior_df`, `tail_description` and `tail_explanation`.** `bs_window_df` has `Aggregate`'s columns and dtypes, indexed by `method`, with rows `rule A` (edges from the closed-form tails at the budget), `cap` (the tilted bucket cap, when it binds), `given` (rule C), `recommended` and `used`; `None` before `update`. `tail_behavior_df` has `Aggregate`'s columns, rows `frequency` and `severity` empty with the note `none: a characteristic function law`, and `aggregate`, each side classed from the family protocol: `power-law` where the side has a tail constant, `bounded` at a finite support end, `super-exponential` where the cumulant generating function is finite on the whole side (the normal), else `exponential`. `tail_class` is `TailClasses(None, None, <heavier side>)`; `bounded` derives from it. The three tail lines of `info` now come from the report. The `bs_window` and `tail_behavior` exhibits register `AggregateCHF`, so a `chfagg` lists six exhibits; the snapshot gains their four entries.

## 1.38.0 - 2026-10-30

**[CHF-Sharpen] A `chfagg` sharpens: `sharpen`, `sharpen_df`, `sharpen_description`, `sharpen_explanation` and `sharpen_program`, with `Aggregate`'s signature and outcome.** The score is the power mean, in tolerance units, of the aggregate moment terms (the mean dropped for a power tail) and four terms of the inversion, `u_coverage`, `u_edge`, `u_truncation` and `u_resolution`, appended to `Aggregate`'s `sharpen_df` columns. A center at or under `good_enough` whose outside mass is within the budget is confirmed and nothing runs; otherwise the probe compares the center with the `recommended_grid` cell, its log2 capped at one above the center, picks by `_pick_cell` with a cell sound when its outside mass is within the budget, and pins the result through `pin_sharpen` (`hints{log2=...; bs=...}`, or `note{sharpen: ...}`). `bs_limit` is accepted and inert. `gamma 100 cv 0.5`, `2 * stable 1.5 1 + 10` and `stable 1.5 0.5` are confirmed at log2 16; `stable 1.1 1` reaches log2 19 in three calls. `recommended_grid` of a two-sided stable law now estimates its outside mass: `stable 1.5 0.5` is advised log2 16 where it was log2 8 with 1.8% outside. Closes `[CHF-Sharpen]`; the Loss Lab's Sharpen leaf lights for a `chfagg` once its tests expect it.

## 1.37.0 - 2026-10-30

**[CHF-Validation-Verdict] A `chfagg` has `Aggregate`'s verdict: `valid`, `validation_score`, `validation_eps`, and its words and opening.** `valid` comes from the new `_validation.valid_characteristic_function`: `AGG_MEAN`, `AGG_CV` and `AGG_SKEW` by `valid_aggregate`'s rules off the moment store, `DEFECTIVE` when the deficit exceeds `DEFICIT_MATERIALITY`, `ALIASING` for an untilted grid that wraps, `INFEASIBLE` when no grid up to log2 24 meets coverage, truncation and resolution, and `TRUNCATION`, a new `Validation` member `Aggregate` never sets. `validation_description` is `explain_validation`, which gains the `TRUNCATION` phrase and a `subject` keyword (`grid infeasible for this law`, and wrap worded for a law). `validation_explanation` opens with the helper `_validation.validation_opening`, now shared with `Aggregate` and `Portfolio` (their text unchanged on all 157 corpus programs), then writes out the inversion's checks; for a tilted stable law with a mean it completes the window mean with the closed-form tail (`FamilyStable.tail_mean_above`), within 1e-5 of the scale. Verdicts moved: `tweedie 1.5 100 2` is `not unreasonable` (it flagged truncation), the stable laws read `fails agg mean`, `stable 0.7 1` reads `grid infeasible for this law; fails pmf deficit`. The validation exhibit emphasizes a failing `Agg` row.

**A number moved.** `truncation_error` of a law with an atom is now that of the part the inversion transforms, `|phi(pi/h) - w exp(i pi x_a / h)|`: `tweedie 1.5 100 2` goes from 4.54e-5 to 1.13e-7, its `recommended_grid` from (24, 1/32768) to (13, 1/16), and the rule A bucket walk uses the same measure, so `tweedie 1.8 1000 5` keeps bs 1/4 where it took 1/8.

## 1.36.0 - 2026-10-30

**[CHF-Aggregate-Frames] A `chfagg`'s moment store, summary and validation frames take `Aggregate`'s shapes, and its theoretical moments are `actual_*`.** `stats_df` is the 26 row `(component, measure)` store with `Aggregate`'s nine columns, built at construction and refreshed by `update`: the law in the `agg` block's theoretical columns, the grid's raw and central moments in `empirical` and `gross_empirical`, `error` between them, the `freq` and `sev` blocks `NaN`. `summary_df` has rows `Freq`, `Sev`, `Agg` (the first two `NaN`) with CV through `Aggregate._cv_or_nan`. `validation_df` is `Aggregate`'s three rows by `EX | CV | Sk` before `update` and the eight columns after, with the SD trio for a law whose support reaches below 0. The old check table is gone from the public surface: its rows are lines of `info`, which now lists `Aggregate`'s rows in order (`n/a` where they do not apply) and then the inversion's, ending in a `checks` line such as `7 ok, 1 flag: outside mass, use hints{...}`. `agg_m`, `agg_cv` and `agg_skew` are renamed `actual_m`, `actual_cv` and `actual_skew`, with `actual_sd`, `actual_var` and `est_var` added and no alias; `est_cv` blanks at a mean indistinguishable from 0. The summary exhibit flags the `Agg` row and the insurer stats view drops the raw moment rows, as for an `Aggregate`; the exhibit snapshot moves only the six `AggregateCHF` entries.

## 1.35.0 - 2026-10-30

**[CHF-Arithmetic-Form] A `chfagg` takes its family's shapes only, with scale and location written as arithmetic around the law, `c * FAMILY shapes + b`, as on an `agg` severity.** The shapes are scipy's where scipy has the law: `stable alpha beta [s0|s1]`, `norm` (none), `gamma shape`, `invgauss mu`, `norminvgauss a b`; `variancegamma sigma nu theta` and `temperedstable C G M Y` keep their published parameters less the location; `tweedie` and the `mean cv` forms are unchanged and take the arithmetic on top (`2 * gamma 100 cv 0.5` has mean 200). The spec carries `chf_scale` and `chf_loc`, also new keywords of `AggregateCHF`, which wraps the family in the new `FamilyAffine`; full parameters from Python (`AggregateCHF('G', 'gamma', shape=4, scale=25)`) are unchanged. `FAMILY_PARAMETERS` and `FAMILY_REQUIRED` list shapes, `LEGACY_PARAMETERS` and `arithmetic_spelling` give the old reading, and `Family.to_decl` renders the arithmetic form. A negative multiplier is refused; a minus glued to a number is its sign (`variancegamma 0.2 0.5 -0.1`) and a shift is written with a space (`- 10`); a shifted Tweedie's atom stays a grid point. Corpora, language reference, decl-help card and grammar listing use the new form; the spec snapshot moves only the `chfagg` lines.

**Breaking, for `chfagg` declarations written since 1.30.0.** A number the family does not take is refused, and the message gives the arithmetic spelling of the same law, which rebuilds the 1.34.0 masses to 1e-12: `gamma 4 25` is `25 * gamma 4`, `norm 100 30` is `30 * norm + 100`, `invgauss 100 400` is `400 * invgauss 0.25`, `stable 1.5 1 2 10` is `2 * stable 1.5 1 + 10`, `norminvgauss 2 0.5 3 1` is `3 * norminvgauss 6 1.5 + 1`, and a trailing `mu` is `+ mu`. At `alpha = 1` in S1 the shift gains `(2/pi) beta gamma log gamma`: `stable 1 0.5 2 3` is `2 * stable 1 0.5 + 3.44127120031`. The Loss Lab's `chfagg` probe and fixtures spell `gamma 4 25` and need `25 * gamma 4`.

## 1.34.0 - 2026-10-29

**[Characteristic-Function-Pedagogy] `FourierTools` folds into the engine behind a shim, and the pedagogy figures draw an `AggregateCHF`.** `aggregate.ft` stays importable as a forwarding shim: `FourierTools` keeps its body and its constructor raises a `DeprecationWarning` naming `AggregateCHF.from_chf` and `from_family`; `poisson_example`, `fft_wrapping_illustration`, `recentering_convolution_example`, `recentering_convolution` and `stats` are moved unchanged to `aggregate.pedagogy` and forwarded through a module `__getattr__` that warns once per name; `make_levy_chf` returns `Stable(alpha, beta).chf` with a warning. Nothing in the library imports the shim, which a test checks. New in `pedagogy`: `chf_simpson_inversion` (Simpson's rule on the Fourier integral by three complex FFTs, the successor of `invert_simpson`), `chf_exact_comparison` (the grid against the family's reference, the successor of `compute_exact`), and the figures `chf_inversion_figure`, `chf_wrapping_figure`, `chf_simpson_figure` and `chf_fourier3d_figure`, with bodies `plot_chf_inversion`, `plot_chf_wraps` and `plot_chf_simpson` in `aggregate.plots`. `Tweedie.chf_law` is the law as an `AggregateCHF`; `Tweedie.test` computes it at the grid given and draws the new figures, `audit` and `stats` read it, and `Tweedie.ft` returns the legacy object with a warning. `dev/FEATURES.csv` gains the `AggregateCHF` column; `[FourierTools-Shim-Removal]`, `[CHF-Tail-Completion]` and `[CHF-Sharpen]` join `dev/TODO.md`; the auxiliary docs page documents `aggregate.characteristic_function` and keeps one sentence on the shim. Eighth and last step of `dev/done/plan-characteristic-function-engine.md`.

## 1.33.0 - 2026-10-29

**[Characteristic-Function-Severity] A `chfagg` serves as a severity, `sev chfagg.NAME` and `ssev chfagg.NAME`, under the hygiene rule an `agg` reference follows.** The token `BUILTIN_CHFAGG` joins `sev_ref_leaf`, `chfagg.` joins the `ID` prefix lookahead (and `agg.sublime-syntax`), and the underwriter resolves the reference by building the inner at its pinned `hints{log2=...; bs=...}`, which `AggregateCHF.with_hints()` writes with `tilt` carried through; the hygiene error on an unpinned `chfagg` gives the grid its own window rule would choose. `_dhistogram_from_object` reads one hook, `_output_xs_ps`, now carried by `Aggregate`, `Portfolio` and `AggregateCHF`, in place of its class switch; `output_support_max` answers a `chfagg` from its declared support. On one grid `agg X dfreq [2] sev chfagg.G` and `agg X dfreq [2] sev 25 * gamma 4` agree to 1e-13; `ssev` keeps a signed law's negative atoms and `sev` clamps them with the existing warning; a tilted heavy-tailed reference carries a deficit of `outside_above`, which the deficit warning reports. `with_hints` pins `normalize` only on an object that has the setting. New corpus lines `CHF.RefGamma`, `CHF.UsesRef` and `CFE.*Ref*`, the spec snapshot gains two entries, the grammar reference is regenerated. Seventh of eight steps of `dev/done/plan-characteristic-function-engine.md`; the web app's `decl-keywords.json` needs the `chfagg.` prefix.

## 1.32.0 - 2026-10-29

**[Characteristic-Function-Pricing] A `chfagg` prices, serves the four generic exhibits and has a chart.** `AggregateCHF.price_pentagon` and `calibrate_distortions` are thin wrappers over `_pricing`, with no `reins_view`; on one grid `chfagg G gamma 4 25` and `agg G dfreq [1] sev 25 * gamma 4` calibrate to the same shapes (8e-12) and the same pentagon (2e-9), and `p = 1` is refused since every family is unbounded. `exhibits._characteristic_function` registers `summary`, `tail`, `stats` and `validation` as passthroughs with their own captions, so `available_exhibits` lists those four and nothing else; the format sheet declares the `validation` frame's `value` and `target` columns, scoped to that exhibit. `charts._emit_characteristic_function` registers the `chfagg` chart, the mass and Lee document over the grid with the family's reference law drawn alongside wherever scipy has a cheap one (not `levy_stable`), and it is the class's primary chart. New fixtures: `AggregateCHF` in the exhibit snapshot (eight entries, nothing else moved) and a `chfagg` in the chart sweep. Sixth of eight steps of `dev/done/plan-characteristic-function-engine.md`; the Loss Lab's kind can now be admitted.

## 1.31.0 - 2026-10-29

**[Characteristic-Function-Families] `chfagg` reaches the whole Tweedie family and the normal inverse Gaussian, variance gamma and tempered stable laws, and `Tweedie.fz` is repaired.** `chfagg T tweedie p mean dispersion` and `tweedie p mean cv c` build every member with `p <= 0` or `p > 1` through `FamilyTweedie` over `aggregate.tweedie.Tweedie`'s cumulant function; for `1 < p < 2` the atom `exp(-lambda)` at 0 is split out of the characteristic function before inversion and put back in the bucket at 0, and the law agrees with the compound engine's on one grid; `p = 1`, `0 < p < 1` and `p = inf` are refused with pointers (`agg ... poisson dsev [phi]`, no law, `stable 1 1`). `norminvgauss alpha beta delta mu`, `variancegamma sigma nu theta mu` and `temperedstable C G M Y [mu]` are `FamilyNormalInverseGaussian`, `FamilyVarianceGamma` and `FamilyTemperedStable`, with closed-form cumulants, scipy references where scipy has the law (`norminvgauss`; `laplace` for the variance gamma at `theta = 0`, `nu = 1`), and `temperedstable`'s `mu` optional (`FAMILY_REQUIRED`). `Tweedie.fz` returns an updated `AggregateCHF` for `p > 2`, `p < 0` and `p = inf`, where it returned `None` or an untilted scipy law (`Tweedie(2.5, 1, 1).fz` was `None`; `Tweedie(-1, 1, 1).fz.est_m` is 1), and the reproductive `p = 0` law now carries its dispersion (it was `norm(scale=1)`). An untilted light law whose rule A bucket lets truncation through, such as `tweedie 2.5 100 2` with its tail of rate 3e-4, now takes the bucket on the ladder that minimizes the larger of the truncation and the Chernoff bound on the wrapped mass, and every untilted grid reports that bound as `outside_below` and `outside_above` (wrapped in, not left out); a law rule A already resolves keeps rule A's bucket, and `recommended_grid` also requires truncation at roundoff. Closes `[Power-Variance-Family]`. New corpus lines `CHF.Tweedie*`, `CHF.NIG`, `CHF.VG`, `CHF.CGMY` and `CFE.*`; the spec snapshot gains five entries and moves none. Fifth of eight steps of `dev/done/plan-characteristic-function-engine.md`.

## 1.30.0 - 2026-10-29

**[Characteristic-Function-Statement] The DecL statement `chfagg` declares a law computed from its characteristic function, and `build` returns it updated.** `chfagg NAME [as "label"] <body> <trailer>` with the body `stable alpha beta [scale [loc]] [s0|s1]`, `tweedie p mean dispersion`, `tweedie p mean cv c`, or a family word and its parameters, `gamma 4 25`, `gamma 100 cv 0.5`, `norm 100 30`, `norm 100 cv 0.3`, `invgauss 100 400`, `invgauss 100 cv 0.5`; `stable` and `tweedie` are keywords, every other family word is an identifier the transformer validates, and `chfagg` is the one new reserved word, so `agg.sublime-syntax` and `decl_words` carry it (the web app's `decl-keywords.json` needs the same edit). The spec holds the parameters as written, keyed by name, with `family` and `form`, and `decl_writer` renders it back exactly. `hints{log2=...; bs=...; tilt=...}` reach `update`, and `tilt` is dropped before an `agg`, `port` or `bivariate` update, where it means nothing. A `chfagg` is not a portfolio unit. The `tweedie` body parses but is refused at build until `[Characteristic-Function-Families]`. New corpus lines `CHF.*` in `_test_suite.agg` (the spec snapshot gains eight entries and moves no other), `CFE.*` in `decl-testers.agg`, the shipped `chfagg StableLaw` in `library.agg`, a section in the DecL reference beside `nttagg`, the grammar reference regenerated, and the help cards. Fourth of eight steps of `dev/done/plan-characteristic-function-engine.md`.

## 1.29.0 - 2026-10-29

**[Characteristic-Function-Two-Sided] A stable law with two heavy tails is split into its two totally skewed halves, each tilted toward its own light side, and assembled by a linear convolution so that nothing wraps.** `FamilyStable.halves()` gives `X_+ ~ S(alpha, 1, gamma_+, loc)` and `X_- ~ S(alpha, -1, gamma_-, 0)` with `gamma_± = gamma ((1 ± beta) / 2) ** (1 / alpha)`, exact for every `alpha` including 1. `AggregateCHF.update(tilt=True)` on `|beta| < 1` inverts each half on twice the grid with its own `tau*`, moves a normal law from the wide half to the narrow one where the narrow half's truncation would not reach roundoff (`transfer_sigma`, nonzero near `|beta| = 1`), applies the bucket factor once, and keeps the `2 ** log2` buckets that leave equal mass below and above, so `x_min` is an output: `grid_window` returns `None` for it. The bucket cap is `pi gamma / (104 log 2) ** (1 / alpha)`, `tilt` and `tilt_theta` are `(tau_minus, tau_plus)`, `edge_error` is `max(E_L, E_R, outside_mass)` with the one-sided formula on the whole law per side, and `info` and `validation_df` show `transfer sigma`. Measured at log2 16, unit scale, against the untilted kernel on a 64x window: worst relative bucket error 3e-5 for `alpha = 1.5` at `beta` 0, 0.5 and 0.99 (4e-6 at 0.99, where `sigma / h = 2.5`), 2e-4 at `alpha = 1`, every grid positive, the outside masses balanced to the digit, 30 to 45 ms. A number for `tilt` tilts both halves alike; `tilt=False` wraps as before. For `alpha < 1` the window is coverage limited and the coverage row says so. Third of eight steps of `dev/done/plan-characteristic-function-engine.md`.

## 1.28.0 - 2026-10-29

**[Characteristic-Function-Tilting] A totally skewed stable law is inverted along a shifted contour, with the tilt chosen in closed form, and the mass beyond the window is reported rather than wrapped.** `AggregateCHF.update(..., tilt=True)` (default) tilts a `stable` law with `beta = 1` by `tau*` e-folds across the window, the value that balances the damped wrap against the amplified roundoff, caps the bucket at `min(gamma / 10, pi gamma / (52 log 2) ** (1 / alpha))` rounded down to the ladder (`utilities._round_bucket_down`), and puts the left edge at the support bound or the 1e-40 Chernoff edge; `beta = -1` is the reflection. `tilt=False` inverts on the real line and wraps, for teaching; a number is the tilt in e-folds, and is a `ValueError` on a law with no power tail. New attributes `tilt` and `tilt_theta` (pairs, left and right), `outside_below`, `outside_above`, `outside_mass`, `edge_error` (the modeled worst relative bucket error) and `transfer_sigma`; the mass below the window sits in the first bucket, so `cdf`, `q` and `sf` are right throughout it. `grid_window` returns `(bs, log2, x_min, tilt)`, `recommended_grid` accounts for the cap, and `validation_df` gains the total mass against `1 - outside_above`, outside mass against the budget with paste-ready `hints{log2=...; bs=...}`, edge error and tilt rows. `FamilyStable.chf` is continued onto the upper half plane for `beta = 1`, and `tail_mass_above` gives the closed-form tail: the series below `alpha = 1`, its asymptotic form above, and at `alpha = 1` the leading term with its logarithmic second term `a ** 2 (log v + gamma_E - 1) / v ** 2`. Measured at log2 16, unit scale: relative bucket error 1.5e-5 at mid window and 4e-5 at the right edge for `alpha = 1.5` (0.1 and 0.34 untilted), 7e-7 and 1.4e-6 for `alpha = 0.7`, grid totals within 1e-8 of `1 - outside_above`, 5 ms. `tilt` joins `_HINT_KEYS`, inert on every other kind. A law with `|beta| < 1` is still inverted untilted, with a logged warning, until `[Characteristic-Function-Two-Sided]`.

## 1.27.0 - 2026-10-29

**[Characteristic-Function-Object] `aggregate.characteristic_function.AggregateCHF` computes a distribution by inverting its characteristic function to bucket masses on a self-sized grid.** A first-class peer of `Aggregate` (not a subclass; the shared surface is duck typed) over a family object with closed forms: `stable`, `norm`, `gamma` and `invgauss`, each in its positional form and, where there is one, a `mean cv` form. `update(log2, bs, bucket_sizing_p)` sizes the window from the law's tails (Chernoff edges on a light side, the power tail on a stable side, a finite support bound where it is within 5% of the window) and rounds the bucket with `round_bucket`, with the rounding slack split between the sides; a given `bs` is used as is. The masses are the round discretization's, so a `chfagg` and an `agg` on one grid agree bucket for bucket to 1e-10, and the light-tailed families match scipy to 1e-9 at the default grid. `from_family` wraps a `Stable`; `from_chf` inverts any characteristic function on an explicit grid, untilted. `q`, `tvar`, `cdf`, `sf`, `pdf`, `pmf`, `snap`, `prob_loss_assets`, `var_dict`, `density_df`, `summary_df`, `tail_df`, `stats_df`, `validation_df`, `info`, `plot` and the trailer members carry `Aggregate`'s names and shapes; `truncation_error` is `|phi(pi / bs)|`, and `recommended_grid` is the smallest grid resolving the body. `AggregateCHF` joins `FIRST_CLASS_CLASSES`, and `qd` prints it. Untilted, a stable law wraps; the shifted contour is `[Characteristic-Function-Tilting]`, the next bump. First of eight steps of `dev/done/plan-characteristic-function-engine.md`.

## 1.26.0 - 2026-10-28

**[Bivariate-Dyadic-Lattice] A discrete bivariate axis sizes by the dyadic lattice rule, and an axis with no lattice is sized freely and scattered.** `bivariate._lattice_bs` is the dyadic lattice step and the floating-point `_float_gcd` is gone. An axis with a step takes the first lattice candidate whose grid covers its measured width within the axis cap (`total_log2 - min_axis_log2`, 16 at the defaults). An axis with no lattice, no fitting candidate, or a pinned `bs` that misses it is measured like a copula axis, and the joint matrix is placed on it with the units' `dsev_bucket` scheme, so the margins of the severity still equal `g0` and `g1`. `_netceded_exact_bs` uses the same routine and rule, with its budget test as the fit. Numbers move only for decimal or noisy axes: `[0 0.1 0.2]` gets a power-of-two `bs` (was 0.1), and `[0 0.1 0.3000001]` builds in a fraction of a second (was a `bs` near `1e-7` that did not finish). Every `MV.DBV*` tester and every `bvagg` program in the shipped corpora sizes as before.

## 1.25.0 - 2026-10-28

**[Portfolio-Lattice-Bucket] A Portfolio whose units are all discrete on one pooled dyadic lattice takes its bucket from the lattice rule, and `log2` shrinks to the need, signed books included.** `port_best_window` takes the step of every unit's atoms together and picks the first lattice candidate whose grid fits the cap, ahead of `round_bucket(max(min_k bs_k, W / N))`, which stays for every other book. A signed book's need is measured from the origin `update` realizes, the sum of the unit origins, not from the window. `Portfolio.bs_explanation` and the `used` row of `bs_window_df` name the lattice.

Numbers move: Poisson units on steps 2 and 5 get `bs` 1 (was 2, which put the atoms at 5 and 15 between grid points), a signed book of quarter-point units keeps 0.25 at `log2` 8 (was 16), and `DiscretePortfolio` keeps `bs` 1 at `log2` 6 (was 16). (Corrected at 1.26.0: this entry first credited 1.25.0 with moves that 1.24.0 made through the units' buckets, steps 3 and 5 from `bs` 4 to 1 and the quarter-point book from 2^-10 to 0.25.) A signed book of many lattice units now holds its mass: 1000 units on `[-1.75 .25 .5]` held 17.7% of it at 1.24.0 and hold all of it at `bs` 0.25, `log2` 13. That bug remains for signed books without a lattice (`[Signed-Portfolio-Origin]` in `dev/TODO.md`).

## 1.24.0 - 2026-10-28

**[Severity-Lattice-Fractional] A discrete severity on a fractional or signed dyadic lattice gets an exact, small grid, by one bucket rule for every lattice.** When every severity atom is a multiple of a common dyadic step `h` (0.25, 0.75, 1.5, 3, 200000), each sizing row takes the largest nice divisor of `h` (a power of two below 1, the `round_bucket` ladder from 1 up) if its window fits `log2`, else `h`, else the standard estimator; the exact discrete window now covers fractional and signed lattices. `dsev [0.25 0.5 1.75]` under 3 fixed claims goes from `bs` 2^-13 at `log2` 16 to 0.25 at 5, and `[0.75 1.5 2.25]` gets 0.25. Decimal atoms and thirds have no dyadic lattice and size as before. New private names in `aggregate.utilities`: `LATTICE_RTOL`, `LATTICE_DEPTH`, `_dyadic_lattice_step`, `_lattice_candidates`. Lattice rows of `bs_window_df` note the step, the residual and whether `bs` is on it (a pinned `bs` off the lattice is noted, not warned), and an atom that noise moved off its lattice point lands whole in one bucket under `dsev_bucket='linear'`.

Numbers move, to the exact answer: a fixed or `dfreq` count on an integer lattice no longer takes a hard-coded `bs = 1` (`dsev [5 10 15]` gets 5, `[10 20 30]` 10), a random count on a step off the ladder takes its ladder divisor (step 3 gets 1, step 25 gets 5), and Portfolios move with their units' buckets. In the three shipped corpora 12 of 329 programs move, listed in `dev/done/plan-fractional-lattice-bucket.md`. Probabilities at the atoms are unchanged, so `density` values scale with `bs`.

## 1.23.0 - 2026-10-28

**[Stable-Statement] `aggregate.stable.Stable` holds alpha-stable parameters, and a bare `stable` statement returns one.** `Stable(alpha, beta, scale=1, loc=0, parameterization='S1')` stores Nolan's S1 parameters, reports `loc0` in S0 (`s0_to_s1` and `s1_to_s0` convert), adds laws with a common alpha, scales, negates and shifts (`sum()` works), and gives `chf`, `mean`, `prob_above_loc`, `tail_constant` and `left_tail_constant` in closed form, with `pdf`, `cdf`, `sf`, `ppf`, `isf`, `rvs` and `plot` from `scipy.stats.levy_stable` (reliable in the body, not in the far tail). `Stable.from_compound_pareto` is the generalized central limit theorem for a Poisson count and a Pareto tail with `1 < alpha < 2`. The statement is a `'conversion'`: `stable 1.5 1 2 0`, `stable 1.5 1 2 -2 s0` and `stable 1000 claims sev 100 * lomax 1.5` each return a `Stable` whose string lists both parameterizations, the mean, the probability above it and the power tails. Submodule access only, `from aggregate.stable import Stable`; it replaces PMIR's `pmir_legacy.stable.Stable`, which stays for backwards compatibility.

Breaking: `stable` is a reserved word, so a program that used it as a name no longer parses. Downstream: `aggregate-api` regenerates `web/src/decl-words.json` (`python -m aggregate.decl_words`), which its `tests/test_decl.py` checks against the installed library, and shows the answer once `[Status-Bar-Text-Answers]` lands.

## 1.22.0 - 2026-10-28

**[Tweedie-Statement] a bare `tweedie` statement converts between Tweedie parameterizations, and the `agg` clause gains a `cv` form.** `tweedie 1.5 100 2` (p, mean, dispersion), `tweedie 1.5 100 cv 0.447214` (p, mean, CV) and `tweedie 10 claims sev gamma 10 cv 1` (Poisson count, gamma severity, written as in an `agg`) each return the new top-level `TweedieConversion`, a named tuple with fields `p`, `mean`, `dispersion`, `cv`, `frequency`, `sev_m`, `sev_cv` and `p0` and constructors `from_dispersion`, `from_cv` and `from_compound`, whose string lists every spelling joined by ` = ` and `Pr(X=0)`, to six significant figures. The statement is a new answer kind, `'conversion'`, beside `'expr'` (`underwriter.ANSWER_KINDS`): it has no name or trailer and is never stored in the recipe base. Only a plain gamma is accepted in the compound form, and every refusal is a `ValueError` naming the form. `agg X tweedie 1.5 100 cv 0.447214` builds and renders back as written, including as a `pnl` engine, through the private `_tweedie_cv` key; existing spec hashes are unchanged.

Downstream: `aggregate-api` refuses a `conversion` statement as an unsupported kind until `[Status-Bar-Text-Answers]` lands.

## 1.21.0 - 2026-10-28

**[ELT-Reporting-Punchup] an `elt` or `yelt` aggregate reports through `elt_description`, `elt_explanation` and `elt_df`.** `elt_description` is a few aligned lines: the table, the event rate and mean annual loss (or the annual outcomes), the secondary uncertainty, the grid, and the checks counted by status with each warning named. `elt_explanation` is three paragraphs: the model in words, each source with its provenance, and every check with its value, tolerance and status, warnings first. `elt_df` is the table the model uses, one row per positive-rate event (`rate`, `mean`, then `max`, `sdi` and `sdc` as the form has them, and `units`, the input rows behind a rolled-up shared event) or per year of positive probability (`prob`, `loss`). For any other aggregate the strings are empty and the frame is empty, so the Narrative pane of aLL gains an `elt` section for loss tables only, with no edit there.

Removed: `loss_table_summary_df`, `loss_table_stats_df` and `loss_table_validation_df` (1.14.0, never published), with no deprecation; their facts and checks are in the new pair.

## 1.20.0 - 2026-10-28

**[YELT-Keyword] the DecL clause for an annual loss table is `yelt`, and `ylt` is gone.** `yelt [0 0 100 300]` (or a file with only a loss column), `yelt [[year loss]]` and `yelt [[year loss prob]]` declare the forms `YELT1`, `YELT2` and `YELT3`, with the semantics of the old `ylt` unchanged. `yelt` is a reserved word and `ylt` an ordinary identifier, with no alias. The spec key is `yelt_sources`, the audit records the table as `yelt`, `info` shows `YELT2, 4 years, ...`, and the error messages, the grammar reference, the Sublime colorer and the corpus follow. `ylt` was never published (the last PyPI release is 1.11.0), so there is no deprecation.

Downstream: `aggregate-api` replaces `ylt` with `yelt` in `web/src/decl-words.json`, `src/aggregate_api/loss_table_files.py` and `tests/test_loss_table_files.py`.

## 1.19.0 - 2026-10-28

**[ELT-Shared-Event-Rollup] a shared ELT6 event is rolled up into one row, and the table updates.** When an event has rows for two or more units, the rows become one row before the model is built: means, maximums and correlated SDs add, independent SDs add in quadrature, at the event's rate (the convention of Homer and Li and of Mitchell-Wallace et al., eltq's `coupling='homer'`). The rolled row is classified, quantized and tail checked as a one-row event, so `update` and a `Portfolio` unit work where 1.15.0 to 1.18.2 raised `NotImplementedError`. The model moments of a shared table move from Burke's coupling to the roll-up (the `LT.Comonotone` severity second moment from 11,000 to 14,000, `LT.Independent` from 9,000 to 12,100), and the row classification counts rolled rows. `info` shows `{k} shared events rolled up`, `loss_table_summary_df` gains the row `('table', 'shared events')` with the input rows and how many have inadmissible variance, the `moment integration` validation row goes, and `SeverityElt` gains `grid_supported`. Tables without a shared event are unchanged, digest included. The Burke route stays in the code behind the private `_loss_table.SHARED_EVENT_RULE` (`[ELT-Burke-Total]`, pended).

## 1.18.2 - 2026-10-28

**[Sharpen-Grid-Score-Defective] the insurer `score_grid` of the sharpen exhibit shows score and the defective flag side by side.** Rows are the realized bucket sizes (`1/64` rather than `0.015625`) and columns the realized log2, in place of the step offsets `d_bs` and `d_log2`, under two column groups, `score` and `defective`. `**` prefixes the starting grid on both axes and wraps the defective flag of the selected cell, so a better score passed over as defective shows beside the cell that was chosen. The raw `sharpen_df` is unchanged.

## 1.18.1 - 2026-10-28

**[Format-Matrix-Rows] the spread layout puts each row of a nested array on its own line.** `format_program` and `pprogram` now open an `elt` / `ylt` matrix or a `dbvsev` probability matrix on the clause line, indent each row one level deeper and close the bracket at the clause's level. A one-row matrix stays on one line, and the terse layout (`spec_to_decl`, `to_agg`) is unchanged. `BivariateDiscreteJoint` in `library.agg` is rewritten in the new layout.

## 1.18.0 - 2026-10-27

**[ELT-Eltq-Extra] eltq is the optional `elt` extra, also in `all`.** `pip install aggregate[elt]` installs eltq 2.13 or later from PyPI for secondary uncertainty in an ELT6 source; it is no longer installed by hand. In a dev checkout `uv sync --all-extras` now replaces a local eltq build with the published wheel, so re-install a checkout after syncing to test an unreleased eltq. The ELT6 tests still skip where eltq is absent, like the other extras' tests.

## 1.17.0 - 2026-10-27

**[Beta-Lomax-Exact-Moments] Beta and Lomax severities, and a Pareto shifted by its scale, compute layered moments in closed form.** `Severity.moms()` no longer uses quadrature for `beta`, `lomax` or `sev s * pareto a - s`, so their theoretical moments, and the `stats_df` errors computed from them, move to the exact values (beta by up to about `2e-5` relative on skewed shapes). Skewed ELT betas such as shapes `1e-5 1` or `1e-6 10` at scale `1e8` now return finite moments where quadrature returned nan or raised `ZeroDivisionError`, and a shifted Pareto with an integer shape no longer returns nan for a finite layer. The Type-I Pareto closed form no longer overflows at large scale and shape. Bucket feasibility readings are unchanged: `lomax` and `beta` still report none (`[Feasibility-Lomax-Beta]`). The `Port.Bounded` regression baseline was re-captured.

## 1.16.0 - 2026-10-27

**[ELT-Single-Event-Uncertainty] An ELT6 with secondary uncertainty updates: eltq quantizes the event severity on the grid, and the sizer checks the severity tail.** With an automatic `bs`, a grid whose edge leaves more than `1e-12` of annual event rate, or more than `validation_eps / 10` of the first three cumulants, above it is widened by growing `bs` at the same `log2`; a given `bs` is honored with a `DefectiveDistributionWarning` naming the `bs` that would pass. `SeverityElt.tail_measures`, `tail_report` and `last_quantization` expose the measurement, and the summary and validation frames gain the severity edge, the tail rate and moment shares, the quantized mass and eltq's mean error. A table sharing an event across units still refuses `update`. A loss table CV or skewness within `1e-12` of zero now gets an absolute estimate error, which removes a false warning on symmetric tables. Docs await a rebuild.

## 1.15.0 - 2026-10-27

**[ELT-Uncertainty-Moments] An ELT6 source without `no2u` builds with secondary uncertainty and reports eltq's model moments before any grid.** Each row is a beta on `[0, max]` with SD `sdi + sdc`, a two-point law on `{0, max}` where that variance is inadmissible, or an atom at zero SD; rows sharing an event across units sum under Burke's coupling. `stats_df`, `info` (`secondary uncertainty`) and the three `loss_table_*_df` frames carry the model, with new summary rows for the row laws, `largest event support`, the moment method and the eltq version. `build(..., update=False)` works; `update` raises `NotImplementedError` until the next step. This needs eltq 2.13 or later, installed by hand (`uv pip install` of a local eltq checkout, then sync with `--inexact`), and raises `ImportError` with that guidance without it; ELT3, ELT4, `no2u` and YLT tables never import it.

## 1.14.0 - 2026-10-26

**[Loss-Table-Reports] `Aggregate.loss_table_summary_df`, `loss_table_stats_df` and `loss_table_validation_df` report an `elt` or `ylt` declaration.** The summary lists what was read, combined and per source, with each file's resolved path and SHA-256 digest; the stats frame sets the severity and annual mean, CV and skewness computed from the table beside the model's and the discretized estimate; the validation frame lists each check with value, tolerance and a `pass`, `warning` or `not computed` status. All three are empty for any other aggregate. This completes the deterministic part of `[DecL-Loss-Tables]`.

## 1.13.0 - 2026-10-26

**[Loss-Table-Files] An `elt` or `ylt` source can be a quoted CSV, TSV, Parquet or Feather path.** Columns are matched through one alias table after ignoring case, spaces and underscores, a `unit` column splits a long file into units, and identifiers stay exact (a floating point identifier column raises). A relative path resolves against the declaring `.agg` file's directory or the session's working directory at registration, and `to_agg` writes it absolute. Each source's audit records the resolved path, size, modification time and SHA-256 digest. Parquet and Feather need `pyarrow`; Excel is not read.

## 1.12.0 - 2026-10-26

**[Loss-Table-Inline] `agg NAME elt ...` and `agg NAME ylt ...` declare an aggregate from an inline event or year loss table.** An `elt` source holds rows `[id rate mean]`, `[id rate mean max]`, or `[id rate mean max sdi sdc]` under `no2u` (mean only). Rows sharing an event identifier across `unit NAME` sources are one occurrence, and the body lowers to compound Poisson. A `ylt` source holds annual losses, `[year loss]` or `[year loss prob]`; units sum within aligned years, the body lowers to `dfreq [1] dsev`, and occurrence terms raise. An ELT6 source without `no2u`, and file sources, raise `NotImplementedError` until later steps of `dev/done/plan-elt.md`. `elt`, `ylt`, `no2u` and `unit` are new reserved words, so a program using one as a name must rename it. `Aggregate.info` gains a `loss table` row, and the DecL reference has an "Event and Year Loss Tables" section. Docs await a rebuild.

## 1.11.2 - 2026-10-26

**[Quote-Safe-Statements] A quoted string is text when a program is split into statements: `as "Cat #1"` keeps its `#1` and `as "a[1]"` keeps its spacing.** Before, the comment stripper cut a label at `#` or `//` (so the statement failed to lex) and the bracket pass respaced `[` and `]` inside it. Both `UnderwritingLexer.preprocess` and `raw_statements` now lift quoted strings the way they already lifted `note{}` bodies. The first step of `[DecL-Loss-Tables]` (`dev/done/plan-elt.md`), which needs quoted file paths.

## 1.11.1 - 2026-10-25

**[Deductible-Forms-Follow-Ups] The layer phrase shows the deductible form: `1,000 xs 50 franchise` or `1,000 xs 50 to 80 disappearing`, where it showed `1,000 xs 50` for every form.** The phrase is the severity in the `qd` one-liner, the `layer` row of `Severity.info`, and the `Declared support` sentence of `tail_explanation`; an ordinary layer is unchanged. Contributed by @KasraBashirioskooei (#23).

## 1.11.0 - 2026-10-22

**Release 1.11.0, the first since 1.6.0, covers 1.6.1 to 1.11.0.** New: the deductible forms on the layers clause, `franchise` (1.7.0) and `to d2 disappearing` (1.11.0); an annual aggregate deductible `aad` and aggregate limit `agglimit` on an occurrence layer, in DecL (1.8.0) and in `pnl` / `xpnl` (1.9.0); `qd` wraps printed strings at 72 characters (1.7.1); and `scripts/check_wheel.py` for release builds (1.6.1). Changed: `aad`, `agglimit`, `franchise` and `disappearing` are new reserved words, so a program that uses one as a name must rename it; exhibits require `greater_tables>=8.1.0`, and some tables render with different decimals (1.10.0). Fixed: κ columns condition on the whole level set, so κ numbers move on rounding-sensitive ledgers (1.10.1); a `fixed` count or a `dfreq` outcome that is not a non-negative whole number now raises instead of returning negative probabilities (1.10.2). Thanks to @KasraBashirioskooei for four pull requests: the wheel check (#17), both deductible forms (#15, #20) and the input guards (#19).

**[Disappearing-Deductible] `y xs d1 to d2 disappearing` on the layers clause: the retention falls linearly from `d1` at a loss of `d1` to zero at a loss of `d2`, and the layer pays `min(X - r(X), y)`.** The `d1 < d2` member of the 1.7.0 retention family: `d2` is stored as `exp_retention_end`, `to d1 disappearing` equals `franchise`, and `d1 <= d2` is checked per layer. A vector `to [...]` has one entry per layer, counted after the limits and attachments broadcast; a scalar applies to every layer. `decl_writer` renders `to d2 disappearing` for any retention end other than the attachment. New reserved word `disappearing`; the DecL reference "Deductible Forms" section covers both modifiers, the Twenty Minutes page compares the three forms, `SevLayer.Disappearing` is in `library.agg`, and `_test_suite.agg` gains `E.TSev20` and `E.TSev21`. Docs await a rebuild. Contributed by @KasraBashirioskooei (#20) from the rulings on #9.

## 1.10.2 - 2026-10-22

**[Input-Guards] A `fixed` claim count that is not a whole number, and a `dfreq` outcome that is negative or not whole, raise `ValueError`.** Both built silently before: the fractional forms returned negative probabilities and a total mass other than one, and a negative outcome was clamped onto 0, moving the mean. The tolerance is a relative `1e-9`. The `fixed` message names the count and its route (stated claims, expected loss over mean severity, or premium times loss ratio over mean severity) and suggests a two point `dfreq` with the same mean, or `poisson`. The `dfreq` check is in the empirical frequency constructor, so `Frequency('empirical', ...)` raises too. Contributed by @KasraBashirioskooei (#19) from the ruling on #18.

## 1.10.1 - 2026-10-22

**[Kappa-Level-Sets] κ columns condition on the whole level set.** A `κq` cell of `PnL.economic_df`, and the waterfall's diversified columns built from `κ01`, is now the mean over every atom whose result agrees with `x_q` to within `validation.level_set_rtol` (new, default `1e-9`, env `AGGREGATE_VALIDATION_LEVEL_SET_RTOL`) times the ledger's largest absolute row value; it was the mean over the atoms whose float matched `x_q` bit for bit. **Numbers move** on ledgers whose result is reached by rounding-sensitive arithmetic (occurrence walks, retro, corridors): on the measured cases the largest κ cell change was 289 on `Bear1990.BN.TreatyIV`, 698 on a 750 xs 250 occurrence walk, 1,428 on the same walk with a corridor, and the native `aad 50` walk's `κ01` layer recovery went from 525.48 to 599.55. The exact distribution of every `bs = 0` ledger row merges atoms within the same tolerance, and `prob_eq_0` counts break-even atoms split by rounding. Docs await a rebuild.

## 1.10.0 - 2026-10-22

**[Format-Sheet-Block-Filter] Exhibits build without greater_tables' `formatters key ... matches no column` warning: the format sheets send each block only the entries for columns it has.** `FormatSheet.block(exhibit, rename, labels)` keeps an exact entry and its tag only when the displayed label (after `rename`) is in `labels`. The `labels` default changes from `()` to `None`, which returns the whole sheet; an empty list returns nothing. A block's own `formatters` are not filtered, so a misspelled key there still warns. Before this, greater_tables 7.4 and later raised one warning per sheet entry a block lacked, and one aggregate-api pricing request returned 538.

**Requires `greater_tables>=8.1.0`** (was `>=6`), and the development lock moves from 6.0.0 to 8.1.0. Tables can render differently from greater_tables 6 and 7; its CHANGELOG marks each such release **Output changes**. Here a float column with no sheet reading takes the decimals its smallest value needs, or SI past five (greater_tables 7.4.0): several `approximation` columns gain a decimal, and `Change CV` on the `reins` summary now reads in SI like `Change EX`. Exhibit snapshots re-captured (8 of 166 cases).

## 1.9.0 - 2026-10-22

**[Annual-Terms-PnL] `pnl` and `xpnl` book the cedant's net under `aad` / `agglimit`: gross loss less the layer recovery `min((R - d)+, m)`, on the same `(L, R)` joint as reinstatements.** New `contract_terms.AnnualLayerTerms(limit, annual_deductible, annual_limit, deposit, share)`, amounts after share, with `recovery`, a zero `reinstatement_premium`, `phi` equal to the recovery, and `decl_clause`; the underwriter attaches it as `pnl.engine.reinstatement_terms` and both faces use the reinstatement builders, so `agglimit 1500` equals `reinstatements 1 free` and `agglimit 750` equals `no reinstatements` leg for leg. The premium clause is optional (zero premium with `ZeroPremiumCessionWarning`); `cede` and one aggregate-tier feature compose as with reinstatements; an `agg.NAME` reference over a `net of` program applies the stored terms. An undeclared layer's step is named with its terms (`occ 750 xs 250 aad 50`) and the structure chart annotates the layer with them. `CoarseJointGridWarning` now also checks a finite `agglimit` (not the `aad`) and opens `coarse occurrence joint:`. The occurrence-layer note's hand ledger is reproduced: net margin 460.70, SD 474.20, recovery 187.08 on a bucket of 5 (460.80 and 187.19 on the default joint). Docs await a rebuild.

## 1.8.0 - 2026-10-22

**[Annual-Terms-DecL] `aad d` and `agglimit m` on an occurrence layer: an annual aggregate deductible and an aggregate limit on the layer's annual recovery.** Written directly after the layer (`occurrence net of 90% po 750 xs 250 aad 50 agglimit 500 deposit 100`), either or both, amounts for 100% of the layer with the share applied after; the reinsurer pays `min((R - s d)+, s m)`. New reserved words `aad` and `agglimit`; spec keys `occ_reins_aad` (default `0`) and `occ_reins_agglimit` (default `inf`), which round trip through `decl_writer`. Refused on an aggregate layer, on more than one occurrence layer, with `reinstatements`, for a negative `aad`, and for a zero or explicit infinite `agglimit`. A plain `agg` honors the terms after `occurrence ceded to`, as the aggregate stage `aggregate ceded to s m xs s d` (an `aggregate` clause alongside raises), and ignores them after `net of` with an `IgnoredDecLClauseWarning`, the stored program keeping them; `bvagg` components follow the same two routes, and a `port` unit raises the same `TypeError` as every other economics clause there. A `pnl` / `xpnl` with the terms raises `NotImplementedError` until the next step. Grammar reference and DecL cheat sheet updated; docs await a rebuild.

## 1.7.1 - 2026-10-22

**[Qd-Wrap-Strings] `qd` wraps the strings it prints at 72 characters.** The `Aggregate` and `Portfolio` info blob, the `PnL` and `BivariateAggregate` repr, and a bare `str` argument now print as wrapped lines instead of one long row in markdown output. Wrapping is per line, so embedded newlines are kept, and long tokens are not split. Tables, numbers and other objects print as before.

## 1.7.0 - 2026-10-20

**[Franchise-Deductible] `y xs a franchise` on the layers clause: a franchise deductible, built as the retention family with `d2` as its parameter.** Once a loss exceeds the attachment the whole loss is paid, subject to the limit: the payment is `min(X, y)` for `X > a`, deductible first and limit second (the IRMI definition and *Loss Models*, first edition), so `limit` stays the largest payment in every form and `detachment` becomes the loss at which the payment first reaches it, `max(a, y)` for the franchise. Engine: `make_layer_retention_cdf`, `_sf`, `_pdf`, `_ppf` and `_isf` beside the `make_layer_attachment_*` wrappers, moments from the existing partial expectations (exact atom sums for a discrete law), and `Severity(..., exp_retention_end=inf)` / `Aggregate(..., exp_retention_end=None)` carrying `d2` (`inf` the ordinary deductible, `a` the franchise); the spec key round trips through `decl_writer` and homogeneous scaling, and vectorizes over `[...] xs [...]`. `feasible_bucket` and `size_biased_cdf` return their no-reading values under a finite retention end. New reserved word `franchise`; DecL reference section "Deductible Forms", a comparison in the Twenty Minutes page, and `SevLayer.Franchise` in `library.agg`. Docs await a rebuild. Contributed by @KasraBashirioskooei (#15) from the rulings on #9; the disappearing deductible, the `d2 > a` member, follows in #20.

## 1.6.1 - 2026-10-20

**[Wheel-Smoke-Check] `scripts/check_wheel.py` builds the wheel, lists it against `src` and smoke tests it in a throwaway venv.** Three stages, every check reported and a nonzero exit on any failure: `uv build --wheel` (or `--wheel <path>` for an existing artifact); a listing check that every package, module and package-data file under `src/aggregate` is in the wheel; and a smoke run in a fresh venv from outside the repository, which confirms the import came from the venv and not `src`, then the version, `build` on a known mean, `qd`, `plot`, every subpackage and a `library.agg` recipe. Standard library plus `uv` only, a release-time command like the numerics gate and not part of the suite; usage in `scripts/README.md`. Contributed by @KasraBashirioskooei (#17).

## 1.6.0 - 2026-10-19

**Release 1.6.0, the first since 1.2.0, rolls up 1.2.1 to 1.6.0a3 below.** New: exact arithmetic in nttagg probabilities and the `exact_tail` chart (1.4.0, 1.5.0), `aggregate.ntt.engine_status()` (1.3.0), a version written once and read live (1.2.1), thread safe `Underwriter` loading (1.5.2), and the DecL word list, `fit_page` and the house type in library figures (1.6.0). Fixed: exact zeros for an identically zero cession (1.5.1), and unit capital above the augmented frame (1.5.3). Thanks to @KasraBashirioskooei for five pull requests: both fixes, the gold standard allocation and bounds tests (#10, #14), and the `AggregateDeductible` library entry (#16).

**[House-Style-Integration] one DecL word list for every colorizer, and library figures that fit a document's page in the house type.** Rolls up 1.6.0a1 to 1.6.0a3 below. `aggregate.decl_words.decl_words()` (and `python -m aggregate.decl_words` for JSON) is the single DecL word list, derived from `decl.lark` and the distortion registry, and `AggLexer` builds from it. `aggregate.plots.fit_page(width, max_height)` fits every library figure to a page, off by default. `aggregate.mplstyle` takes the house type and face (STIXGeneral, 9pt labels, 8pt ticks and legend), and no library code names a type size. aggregate-api, aggregate-asp and house-style each pick these up in their own plans once this is on PyPI.

## 1.6.0a3 - 2026-10-19

**[Library-Figures-House-Type] library figures take the house type and face, and no library code names a type size.** `aggregate.mplstyle` now matches house-style's `house.mplstyle` (kit 1.11.0) on type: STIXGeneral with STIX math, labels and titles at 9pt, ticks and legend at 8pt, bold titles (STIXGeneral has no semibold), tighter legend spacing, and `pdf.fonttype: 42`; colors, grid, line widths, `figure.figsize` and dpi are unchanged. Every size in library code (tower block and tick labels, matrix cell text, legends, the massive bivariate annotations) is read from the active style, so a document's own style now reaches them; without `aggregate.style.use()` legends take matplotlib's default size where they were forced small. Tower block labels grow from 5.85pt to 8pt, so a thin band may lose a label that no longer fits. `FONT_SIZE` and `LEGEND_FONT` stay exported but are unused. aLL's chart base text moves from 10.5px to 11px, because `/v1/meta/style` serves `font.size` 9 where it was 8.5. The chartdoc pixel baselines are regenerated. Docs figures change face at the next build. Plan `dev/done/plan-library-figures.md`.

## 1.6.0a2 - 2026-10-19

**[Library-Figures-Fit-Page] `aggregate.plots.fit_page(width, max_height)` fits every library figure to a document's page.** A document calls it once with its text width in inches, and every figure aggregate draws afterwards, `Aggregate.plot()`, `Portfolio.plot()`, the `make_grid` and `make_mosaic` canvases and the rest, is drawn at a size that fits: a figure wider than the page shrinks to it, its height falls with it but not below 2in, and no figure is taller than `max_height`. A narrower figure keeps its size; nothing is ever enlarged. On a 6.3in page a two panel `Aggregate.plot()` is 6.3 x 2.2in where it was 7.0 x 2.45in, drawn at that size so its type prints at the style's size. `fit_page()` clears the page. Off by default, so nothing changes for anyone who does not call it. `Portfolio.scatter` now honors its `figsize`, `marker`, `s`, `alpha` and `diagonal`, which it ignored. Docs await a rebuild. Plan `dev/done/plan-library-figures.md`; house-style calls `fit_page` in its own plan once this is on PyPI.

## 1.6.0a1 - 2026-10-19

**[DecL-Word-List] one DecL word list, `aggregate.decl_words.decl_words()`, for every program that colors DecL.** It returns the words by group: `reserved`, `frequencies`, `clauses` and `constants` read from `decl.lark`, `distortions` from `Distortion.available_distortions(pricing=False)`, and `declarations`, `severities` (name to shape parameter count) and `mixing` kept in the module and checked by `tests/test_decl_words.py`. `python -m aggregate.decl_words` prints it as JSON with the package version, for colorizers outside Python. `AggLexer` now builds its keyword rules from the list, and its tokens over the four shipped corpora are unchanged; the severity names `gilbrat` and `trapz`, which scipy no longer has, are gone, and `gibrat` is added. Submodule access only. aggregate-api, aggregate-asp and house-style pick the list up in their own `[DecL-Word-List]` plans once this is on PyPI. Docs await a rebuild. Plan `dev/done/plan-decl-word-list.md`.

## 1.5.3 - 2026-10-19

**[Unit-Capital-Above-Frame] unit capital above the augmented frame uses the last row, and far tail unit capital always warns.** `unit_capital_at`, behind `pricing_at`, `price` and `pentagon_at`, now treats an asset level above the last row of the augmented frame as `pricing_at` does: it uses the last row and warns, where it raised a numpy broadcast `ValueError`. `pentagon_at` now warns when it falls back to the last row. A new warning fires whenever the asset level reaches layers with `S <= VALIDATION_NOISE`, whichever units are requested, so a single unit `pentagon_at(unit=...)` in the far tail is no longer silent. The split is left as computed and total capital stays exact; nothing moves at ordinary asset levels. Contributed by @KasraBashirioskooei (#13, from #12).

## 1.5.2 - 2026-10-14

**[Thread-Safe-Load] `Underwriter.load()`, `fork()` and `reload()` are thread safe.** A thread that looks up a recipe or takes a fork while another thread is reading the configured databases now waits for the read to finish. Before, it found the base already marked loaded part way through the read, and got `RecipeNotFound` or, worse, a fork that kept the partial base for good: this is what broke a freshly started aLL server, and any multi-threaded host could hit it. A configured read that raises is now rolled back and retried on the next access, so a broken file in the configured request raises every time until it is fixed, where before later calls quietly used whatever had been read. Underwriters, `build` included, still pickle and deep-copy. Concurrent `build()` calls on one underwriter remain unsafe; give each thread a `fork()`. Plan `dev/done/plan-1.5.2-thread-safe-load.md`.

## 1.5.1 - 2026-10-14

**[Zero-Share-Layer-Moments] a cession that is identically zero reports exact zeros.** A layer, or a whole program, whose every layer has zero share or zero limit now gets the exact point mass at zero, at the occurrence and the aggregate stage alike: raw moments and mean exactly 0, `cv` and `skew` `nan`, `pr_loss` 0, and the same for the realized `est_*` of a `ceded to` program. Previously a genuine truncation deficit in the subject was relocated through the zero layer, so `aggregate net of 0% po 200 xs 400 and 100 xs 600` over `sev lognorm 50 cv 1.5` reported a layer mean of `1.19e-06` with `cv` and `skew` of `1.66e+05`. `CS.Gapped` in `tests/test_chart_structure.py` is now checked on every tier. Contributed by @KasraBashirioskooei (#11).

## 1.5.0 - 2026-10-14

**[NTT-Exact-Tail-Chart] a new chart, `exact_tail`, draws an nttagg's exact answer as base-10 logarithms down to the last bucket of the support.** Two panels over one loss axis, `mass` and `survival`, each with an `Exact` series computed from the integers and a `Float FFT` series beside it, so a bucket at 1e-462 reads where the float answer has long since gone to zero or noise. Exact zeros are omitted. Available for an updated `AggregateNTT` only and never primary, so `primary_chart` still answers `agg`; draw it with `plot_chartdoc(build_chart_doc(a, 'exact_tail'))`. The chart IR gains two axis units whose values are exponents, `log10_density` (drawn as markers only) and `log10_probability` (drawn as `probability` is), and `plot_chartdoc` draws both; `CHART_IR_VERSION` stays at 4. New `charts.chart_exact_tail`. Docs await a rebuild. Plan `dev/done/plan-1.4.0-nttagg-arithmetic-exact-chart.md`; the aLL leaf is `aggregate-api/dev/plan-1.13.0-exact-tail-leaf.md`.

## 1.4.0 - 2026-10-14

**[NTT-Rational-Arithmetic] nttagg probability lists accept exact arithmetic.** A probability may be parenthesized arithmetic (`+ - * /`, and `**` or `^` with a whole-number power), evaluated exactly as a `Fraction` with no float anywhere: `dsev [1:3] [1/2 (1/2-1/(13*11*(19**3))) (1/(13*11*(19**3)))]` is exactly `[1/2 980835/1961674 1/980837]`, and `(0.1+0.2)` is exactly `3/10`. As in `agg`, a bare item sees only literals, `/`, `**` and parentheses, so `+`, `-` and `*` need parentheses and a probability still cannot start with a sign. A fractional power raises, as does a power whose result would exceed `NTT_EXPONENT_MAX_BITS` (new in `constants.py`, `2**20` bits). `a/b` now accepts decimal operands, so the message "nttagg a/b probabilities take integers" is gone; `1/0` reads as before. Reformatting prints the reduced fractions, not the expression. `_test_suite.agg` gains `NTT.Arithmetic` and the spec snapshot its entry. Docs await a rebuild. Plan `dev/done/plan-1.4.0-nttagg-arithmetic-exact-chart.md`.

## 1.3.0 - 2026-10-14

**[NTT-Engine-Status] `aggregate.ntt.engine_status()` says whether the compiled engine would be used, and why not.** It returns a dict: `engine` (`'rust'` or `'python'`, what `ntt_engine='auto'` resolves to), `installed`, `engine_version`, `engine_protocol`, `library_protocol`, and `reason` (`None` when the engine is used, else `'not installed'` or `'protocol 1, library speaks 2'`). It never warns and never raises, and it leaves the once-per-process protocol-mismatch warning for the first computation, so a caller such as an About panel can tell an engine installed and used from one installed and ignored without reaching for private names. Submodule access only, like the rest of `aggregate.ntt`. Plan `dev/done/plan-1.3.0-ntt-engine-status.md`.

## 1.2.1 - 2026-10-14

**[Version-Source] the version is written once and read live.**

- `aggregate.__version__` is a plain literal in `src/aggregate/__init__.py`, and `pyproject.toml` reads it (`dynamic = ["version"]`, through `[tool.setuptools.dynamic]`). It no longer comes from `importlib.metadata`, which in an editable install keeps the number recorded at install time until a reinstall.
- A plugin's `LoadedPlugin.version` is the `__version__` of its top-level package when it declares one, falling back to distribution metadata. An editable plugin now reports the version in its source.
- `[tool.uv] cache-keys` adds `src/aggregate/__init__.py`, so uv rebuilds the editable install when the version changes.
- A bump edits one line and needs no sync. Plan: `dev/done/plan-1.2.1-version-source.md`, shared with aggregate-api, aggregate-relativity and aggregate-ntt.

## 1.2.0 - 2026-10-13

**[NTT-Exact-Aggregate] exact compound distributions through the new `nttagg` DecL kind, the `reproductions.agg` corpus, and `decl_help()`.** The first release since 1.0.1, rolling up 1.1.0 and 1.2.0a1 to 1.2.0a15 below. `build('nttagg DiceOfDice dfreq [1:100] dsev [1:100]')` returns `aggregate.ntt.AggregateNTT`, an `Aggregate` that runs the usual FFT pipeline and computes the exact rational distribution beside it with the number-theoretic transform: `a.p_exact` (reduced Fractions, built on first access), `a.p_exact_at(loss)` for chosen buckets, `density_df` columns `p_exact_float` and `p_exact_log10` (finite far below the smallest double), per-bucket `accuracy_df`, the `ntt_audit` record, and an exact `stats_df` column that validation measures against. Probabilities are text-exact (`0.3` is `3/10`), outcomes are nonnegative integers, the grid sizes itself to the support, and the kind cannot appear in a portfolio, pnl or bivariate. The 1000 x 1000 uniform dice of dice runs in 216 s in pure Python, or 10.7 s with the optional compiled engine `aggregate-ntt` installed from <https://github.com/mynl/aggregate-ntt> (1.0.0a3 or later, never a dependency); `ntt_engine=` forces either path and answers are bit-identical. The wheel also ships `reproductions.agg`, 140 DecL programs rebuilding published actuarial and numerical results, read with `build.load("reproductions.agg")` (see <https://blog.mynl.com/posts/notes/2026-10-08-Aggregate-Reproductions/>); `library.agg` gains `ExactDiceOfDice` and drops the broken `PnLBook` recipe. New `decl_help()` returns one page of DecL help as HTML that renders in Jupyter: the anatomy of a statement by default, or a one-page reference card with `decl_help('card')`.

**Numbers move.** `est_cv`, `est_skew` and the empirical rows of `stats_df` no longer relocate rounding noise and deleted tail mass to the top of the grid; only a deficit above 1e-12 moves (`[Moment-Deficit-Genuine-Only]`, 1.2.0a14). Results change on thick-tailed laws and tall grids: ten dice on a 2**20 grid read skewness 0, the true value, where they read 4.875. Two library validation baselines move with it: `RawLognorm` clears `SEV_SKEW` and `ThickThickPortfolio` gains `AGG_SKEW`.

## 1.2.0a15 - 2026-10-13

**[DecL-Help-Pages] `decl_help()` returns one page of DecL help as HTML**, for Jupyter and for the aggregate Loss Lab to serve at `GET /v1/decl/help`. `decl_help()` is the anatomy (one `agg` statement taken apart slot by slot, a portfolio and a P&L built on it, and the alternative forms for each slot); `decl_help('card')` is a ten-box reference card that prints to one Letter landscape page. The return is a `str` (`DeclHelp`) with `_repr_html_`; the forms are listed in `DECL_HELP_FORMS`; the sources are package data in `data/decl-help/`. Every program in both pages builds, and every snippet lexes clean, under `tests/test_decl_help.py`. Plan `dev/done/plan-1.2.0a15-decl-help.md`.

## 1.2.0a14 - 2026-10-12

**[Moment-Deficit-Genuine-Only] `est_skew`, `est_cv` and the empirical rows
of `stats_df` stop parking deleted tail and rounding at the top of the grid.**
The moment sites in `Aggregate.update_work` (and its stage helper),
`Portfolio.update` and `create_from_sample` now clean the density with
`remove_fuzz(raw, eps=noise_floor(raw))`, where the new
`utilities.noise_floor` is the density's largest negative magnitude capped at
machine epsilon, and `xsden_to_mwrangler(xs, den, deficit=None)` gains the
`deficit` keyword and relocates to the grid top only a deficit above
`VALIDATION_NOISE` (1e-12), measured on the raw array; negative and
rounding-level deficits no longer relocate for any caller, severity and
reinsurance views included. Numbers move on thick-tailed laws and tall grids:
GEMAct's Table 2 model at `log2=22` reads 7.282 against theory 7.284 where it
read 11.916, ten dice on a 2**20 grid read 0 where they read 4.875, and
`J.Re14` at its default grid reads -7.691 where it read -6.060. The heavy
tailed `Mata2005.M.v3`/`.v4`, `G.Mixed10`/`11` and `Mack2003.S.v9` read lower
and steadier; `Bruno2006.B7` is the one case that moves further from theory,
0.0039446 to 0.0039976. Two library validation baselines move: `RawLognorm`
clears `SEV_SKEW` (the stage column no longer disagrees with the main site)
and `ThickThickPortfolio` gains `AGG_SKEW` on a unit whose grid cannot hold
its tail. The baseline parquet, spotchecks and exhibit snapshots are
recaptured; `scripts/moment_rule_sweep.py` compares the rules per object.

## 1.2.0a13 - 2026-10-10

**[NTT-Grid-Cap] an `nttagg` sizes its grid to its support instead of filling
the underwriter's cap.** `log2` on `AggregateNTT.update` is now a cap in the
same sense `bs_window` uses the word: any value at or above the required
`ceil(log2(N + 1))`, where `N = max(freq) * max(sev)`, resolves to exactly
that required value, and so does `0`. `build('nttagg ... dfreq [1:6] dsev
[1:6]')` returns `log2=6` and 64 buckets where it returned `log2=16` and
65,536, taking `density_df` from 7.9 MB to 7.7 KB and `accuracy_df` from
3.1 MB to 3.1 KB; the 100 x 100 case moves from 16 to 14. An explicit larger
`log2` is trimmed as well, because with `bs` pinned to 1 and the support
exactly 0..N there is one correct grid and the extra buckets are exact zeros.
A cap below the required value still raises and names the value to pass,
unchanged, and `build(..., log2=0)`, which could never reach this kind,
is no longer needed for anything.

## 1.2.0a12

**[NTT-Exact-At] new `AggregateNTT.p_exact_at(loss)`, the exact probability
at the buckets you name.** A scalar loss gives one reduced
`Fraction`, a sequence gives a Series of them, both identical to the
corresponding entries of `p_exact`, and `Fraction(0)` beyond the support. A
loss that is not an integer on the grid raises rather than being rounded.
Measured on a 500 x 500 case: **12 microseconds against 5.65 s** for the
full `p_exact`, and reading buckets never populates or needs that cache.

`p_exact` itself is unchanged, still the full Series of reduced Fractions,
still lazy and cached. Its docstring now says plainly that it is the
expensive view: one gcd per bucket against a 9,976-bit denominator, about
44 s on a 1000 x 1000 case, and nothing else in the class touches it, since
`density_df`, `accuracy_df` and `ntt_audit` all work from the raw integer
numerators. Also recorded there: `p_exact.sum() == 1` never needs checking,
because `sum(numerators) == denominator` is one of the three decisive
checks `exact_compound` makes before returning.

## 1.2.0a11

**[NTT-Accuracy-Integer] `accuracy_df` drops its Fractions for an identity
and is 16x faster.** A double is exactly `m / 2**k` and the exact
probability is `H / D`, so with `excess = m * D - H * 2**k` the signed error
is `excess / (2**k * D)` and the relative error is `excess / (2**k * H)`,
dividing by `H / D` having cancelled `D` outright. Two integer true
divisions, each correctly rounded, in place of two `Fraction` operations
whose gcd against a 9,976-bit denominator was the entire cost. **Values are
bit-for-bit identical**, which a new test pins against the route it
replaced.

Measured on first access: 500 x 500 from 14.66 s to **0.92 s**,
1000 x 1000 from about 127 s to **5.34 s**, the ratio growing with the
denominator. This frame had become the most expensive part of an exact
computation by a wide margin, 20x the transform and reconstruction
together; getting everything at 500 x 500 (build, `density_df`,
`accuracy_df`, `ntt_audit`) is now 3.43 s against 17.3 s.

`_safe_float` is removed, having no caller left; `_safe_divide` replaces it
one level lower, taking the two integers instead of the `Fraction` they
would have formed.

## 1.2.0a10

**[Rust-Engine-Phases] the compiled engine now owns both heavy phases, and
the 1000 x 1000 case takes 10.7 s.** With `aggregate-ntt` 1.0.0a4 installed,
the per-prime transform runs there as well as the reconstruction, drawing
63-bit primes instead of 32-bit and so needing about half as many. Measured,
answers bit-identical to the pure-Python path: 500 x 500 from 16.4 s to
**1.11 s**, the headline 1000 x 1000 uniform dice from 212.3 s to **10.71 s**
(159 primes rather than 322, transform 33.8 s of thread time over four
workers, reconstruction 5.74 s, limb decode 0.94 s). Nothing changes without
the engine.

**`_ENGINE_PROTOCOL` is now 2**, matching the engine's own bump when its
transform signature landed. An `aggregate-ntt` older than 1.0.0a3 is refused
rather than called wrongly: `ntt_engine='rust'` raises and `'auto'` falls
back with one warning. Upgrade both together.

Which route a computation took is reported two ways. `ntt_audit` and `meta`
carry `engine`, `engine_version` and `engine_protocol` as before. New: the
`aggregate.ntt` logger emits one `INFO` line before the work naming the
engine and version, the prime count and width, and the worker count, and one
after it with the time each phase took.

**`primes_used` and `prime_bits_*` now differ between the two engines, and
both are correct**: the reconstructed integers are determined by the
mathematics, not by which primes carried them. Three tests that asserted a
literal 32-bit prime count were rewritten to assert the relationship
instead; any downstream check of those two audit rows against a fixed number
needs the same treatment.

## 1.2.0a9

**[Rust-Engine-Prime-Width] `ntt_primes` draws primes of a chosen width, and
the root no longer needs a factorization.** New `max_bits` parameter,
defaulting to 32, which is the only width `ntt` can transform with:
`ntt` now **raises** on a prime at or above `2**32`, because its uint64
products overflow silently and return plausible nonsense. `max_bits=64`
halves the prime count for a given bound (159 against 322 on the 1000 x 1000
case) and lifts the supply ceiling that bounds the exact method's envelope,
for a consumer with wider arithmetic. `exact_compound` does not yet pass it:
the compiled engine's `Reconstructor` takes 32-bit primes as of
`aggregate-ntt` 1.0.0a2, so wide primes have no consumer until the engine
widens both of its kernels and the seam protocol bumps on both sides.

Roots now come from `_root_of_unity`, which uses the fact that the transform
length is a power of two instead of factoring `p - 1`. Returned roots
therefore differ from earlier versions and **answers do not**: any primitive
length-th root of unity gives the same transform, since forward and inverse
use the same one. `_prime_factors` and `_find_generator` remain as the
reference it is pinned against. The reason is the 64-bit case, where
factoring `c = (p - 1) / length` is a lottery on the shape of `c`: a
balanced-semiprime cofactor costs about four minutes for a single prime
against 20 microseconds for the new route, which is indifferent to it.

A supply exhaustion at 32 bits now names the real remedy, the compiled
engine, rather than leaving the reader to suspect the transform length.

## 1.2.0a8

**[NTT-Rust-Integration] the optional compiled engine is used when it is
installed.** `aggregate.ntt` discovers the separate `aggregate-ntt` package
at import time and runs the Chinese remainder reconstruction there, falling
back to the pure-Python path when it is absent, which is the normal case and
needs no attention: a user who has never heard of the package sees no change.
New keyword `ntt_engine='auto' | 'python' | 'rust'` on `exact_compound` and
`build(..., ntt_engine=...)`, where `'auto'` is the default, `'python'`
always forces the reference path, and `'rust'` raises rather than quietly
downgrading when the engine is missing. `ntt_audit` and the `meta` record
gain `engine`, `engine_version`, `engine_protocol` and
`limb_decode_seconds`, so a saved record says what produced it. The seam is
versioned and a protocol mismatch never computes: it raises under `'rust'`
and falls back with one `UserWarning` under `'auto'`. Measured on this
machine with the engine present, both answers bit-identical: 500 x 500 from
18.3 s to 4.3 s, the headline 1000 x 1000 uniform dice case from 212.3 s to
39.2 s, its reconstruction phase from 187 s to 7.5 s, and limb decoding
0.9 s.

`aggregate-ntt` is **not** a dependency, optional extra or otherwise, and
will not become one. Install it by hand to use it. The pure-Python path is
unchanged, stays the reference the engine is tested against, and is fast
enough for any ordinary problem.

## 1.2.0a7 - 2026-10-09

**[NTT-Exact-Log10-Lazy] `density_df` gains `p_exact_log10`, which can see the tail, and stops building a million Fractions to hide it.** On a large `nttagg` the exact probabilities run far below the smallest double, so `p_exact_float` is `0.0` across most of the tail and the exact answer was invisible in the frame. `p_exact_log10` is the base-10 logarithm formed as `log10(numerator) - log10(denominator)`, which is a float for any coefficient however small: on a 200 x 200 uniform dice of dice the deepest bucket carrying mass reads **-462.5** where `p_exact_float` reads `0.0`. It is `-inf` only where the probability is exactly zero, and it is accurate to about 1e-10 in the implied probability, which in the denormal range makes it strictly more accurate than taking the logarithm of `p_exact_float`.

**Breaking: `p_exact` is no longer a `density_df` column. It is now a property, `a.p_exact`,** returning the same reduced Fractions as a Series aligned the same way, built on first access and cached. Migration is `a.density_df.p_exact` to `a.p_exact`. The column was built eagerly whenever the frame was touched, at one gcd per bucket against a 9,976 bit denominator, which measured **77.7 s on a 1000 x 1000 case** where every other column is effectively free. First access to `density_df` on a 500 x 500 case now costs 0.4 s against 5.8 s. `p_exact_float` is computed by integer division rather than through a Fraction, which is bit-identical and 730x cheaper.

## 1.2.0a6 - 2026-10-09

**[NTT-Accuracy-Overflow] `ntt_audit` and `accuracy_df` no longer raise on a large `nttagg`, and `max_rel_err` reports a number that means something.** Both properties raised `OverflowError` whenever the exact tail reached below about 1e-328, because the per-bucket relative error divides by an exact probability far smaller than any double. That is every uniform dice of dice from roughly 150 x 150 up, so the headline diagnostic was unusable on exactly the cases the exact engine exists for. The ratio now reports a signed infinity instead of raising.

`max_rel_err` is now taken over *resolved* buckets only, those whose exact probability is nonzero and strictly above `noise_floor`, which is the measured amplitude of the float pipeline's own numerical noise. Unrestricted it reported the ratio of noise to an unrepresentable number: 2.1e183 on a 100 x 100 case and 7.2e304 on a 200 x 200. It now reads 0.31 and 0.078. **This number has moved, and the old one was meaningless rather than merely imprecise.**

`ntt_audit` gains three rows: `rel_err_buckets` (how many buckets the maximum covers), `unresolved_buckets` (nonzero exact probability at or below the noise floor, so the float pipeline never had the signal) and `underflow_buckets` (nonzero exact probability that underflows to zero as a double). On a 200 x 200 case those read 26,302, 13,698 and 277. It also stops rebuilding the whole reduced-Fraction vector a second time to find the exact zeros, which halves the audit's cost on a large case.

## 1.2.0a5 - 2026-10-09

**[NTT-Workers-Default] the `nttagg` transform thread count defaults to `min(4, os.cpu_count())` instead of `min(8, ...)`.** What threads buy in this engine is overlap with the serial reconstruction rather than parallel speedup, so once the product tree shortened the reconstruction at 1.2.0a4 the eighth thread stopped paying for itself. Measured on the 1000 x 1000 uniform dice case: four threads give 216.4 s against 212.3 s for eight, so 2 percent of wall clock for twice the CPU, 709 transform thread-seconds against 1356. Both are 1.45x the 312.9 s serial run. Pass `ntt_workers=n` to `build` or `workers=n` to `exact_compound` to choose explicitly; `1` still forces serial. Answers are unaffected either way.

## 1.2.0a4 - 2026-10-09

**[NTT-CRT-ProductTree] the `nttagg` Chinese remainder reconstruction is a product tree instead of a linear fold, and large exact cases run about 2.8x faster.** The reconstruction folded one prime at a time into a running answer, multiplying by a modulus that grew with every prime, so the phase was quadratic in the prime count and had become 99.3 percent of the wall clock on the largest cases. It is now a balanced product tree, built online as a merge stack so that no more than `log2(n)` partial reconstructions are ever held and each prime's residues are still consumed as they arrive. The headline 1000 x 1000 uniform dice case, support 10^6 at transform length 2**20 over 321 primes, runs in 212 s against 594 s at 1.2.0a3, with the reconstruction itself falling from 590 s to 198 s; the 500 x 500 case runs in 16.6 s against 35.2 s. Answers are bit identical, pinned by a test that the tree and the linear fold reconstruct the same integers, and the three end-to-end checks are unchanged.

The transform phase speeds up as a side effect, because the old reconstruction held the GIL in stretches of up to several seconds and starved the worker threads. Transform thread-seconds on the 500 x 500 case fall from 266 to 119 for the same arithmetic, and the per-prime log lines no longer climb through a run.

Peak memory on the largest cases roughly doubles, since the merge stack holds partial reconstructions alongside the answer. `NTT_MAX_BYTES` still estimates the answer only, so a case near the cap now needs about twice the headroom it did.

## 1.2.0a3 - 2026-10-09

**[NTT-Composability] `AggregateNTT` no longer carries the two methods it refuses, and `library.agg` gains an `nttagg` entry.** `pnl_program` and `reins_program` are now absent on an `nttagg` rather than present and raising at call time, so `hasattr(o, 'pnl_program')` answers False, which is how this library already states composability: `PnL` carries neither, `Portfolio` carries `pnl_program` and not `reins_program`. Reading either member still prints the full explanation, now as `AttributeError` rather than `ValueError`, and `dir()` still lists the name. Consumers that derive navigation from object capability no longer need a hard-coded list of kinds. `library.agg` gains `ExactDiceOfDice` in the Numerical character section, tagged `topic:numerics, role:intermediate`, so the kind is reachable from `build.recipes`. Details and divergences: `dev/done/plan-nttagg-composability.md`.

## 1.2.0a2 - 2026-10-09

**[NTT-Inplace-Kernels] the `nttagg` transform writes through preallocated buffers, and the transform phase now threads by default.** Every hot array expression in `aggregate.ntt` goes through `out=` against scratch allocated once per call: the butterflies in `ntt`, the twiddle table doubling, the inverse scaling, the binary exponentiations and the power updates in both spectral walks. Same integer arithmetic in the same prime fields, so answers are bit identical; what goes away is a full-length temporary per modular multiply, worth 1.65x on the transform phase of the dense 400 x 400 case (12.7 s to 7.7 s) and 1.5x on the spectral walk at transform length 2**20.

`ntt_workers` now defaults to `min(8, os.cpu_count())` instead of serial; pass `ntt_workers=1` to force serial. `ntt_audit` gains a `workers` row, which `meta` already carried unreported. The dense 400 x 400 case runs in 13.8 s at the default against 25.1 s serial at 1.2.0a1 and 119 s at 1.1.0; the headline 1000 x 1000 uniform dice case, support 10^6 at transform length 2**20 over 321 primes, now runs in 594 s against about an hour serial and about 25 minutes at eight workers on 1.1.0.

## 1.2.0a1 - 2026-10-09

**[NTT-Uniform-Geometric] the `nttagg` spectral walk sums equal-weight outcome runs in closed form.** The frequency outcomes are decomposed once into maximal runs of consecutive integers sharing one cleared weight, and a run of eight or more is evaluated as a geometric series in the spectral variable rather than one modular array multiply per outcome, costing `log2(run_length)` array multiplies instead of `run_length`. A uniform claim count is a single such run, so the dominant phase of a dense case collapses: measured 37.5x on the spectral walk of the 1000 x 1000 uniform dice shape (8.62 s to 0.23 s per prime at transform length 2**20), taking the dense 400 x 400 case end to end from 119 s to 25.1 s serial. Shorter runs and irregular outcome sets keep the per-outcome walk and pay nothing new. Answers are unchanged: same integer arithmetic in the same prime fields, pinned by the polynomial oracle and by a direct test that the closed form and the forced per-outcome walk agree bit for bit.

One correctness fix rides along, found while reviewing the algebra. The geometric path folds the severity denominator `Ds` into the spectral point through `Ds**(-1) mod p`, which does not exist on a prime dividing `Ds`. Such a prime is now routed to the plain walk. Reaching it needs a severity denominator carrying a prime factor of the form `c * L + 1` just below `2**32`, so no previously computed answer moves, but the case is constructible and is now covered by a test.

## 1.1.0 - 2026-10-09

**[Reproductions-Corpus] the wheel ships `reproductions.agg` and `library.agg` drops the broken `PnLBook` recipe.** `src/aggregate/agg/reproductions.agg` carries every DecL program from *Reproductions*, the book of worked examples rebuilding published actuarial and numerical results with `aggregate`: 124 canonical-form programs from 27 chapters, each noting its source paper, named `<CitationKey>.<name>`, generated from the book by `make_agg.py` (read back with `build.load("reproductions.agg")`). The broken `PnLBook` recipe (book-level P&L, a deliberate NotImplementedError demonstration) left `library.agg` in the same cleanup, and `tests/test_library_entries.py` `CANNOT_BUILD` no longer lists it.

**[NTT-Exact-Aggregate] new DecL kind `nttagg`: exact compound distributions via the number-theoretic transform.** `build('nttagg DiceOfDice dfreq [1:100] dsev [1:100]')` returns `aggregate.ntt.AggregateNTT` (submodule access only), an `Aggregate` that runs the usual float FFT pipeline and, beside it, the exact rational answer computed over 32-bit prime fields with CRT reconstruction: `density_df` columns `p_exact` / `p_exact_float`, per-bucket `accuracy_df`, audit record `ntt_audit`, exact `stats_df` moments that validation measures against, guardrails for monster cases (`ntt_override=True`), and a threaded transform phase (`ntt_workers=`). Inputs are text-exact (`0.3` is `3/10`, `a/b`, exact uniform default); outcomes are nonnegative integers; the kind cannot appear in a portfolio, pnl, or bivariate. Rolls up the 1.1.0a1 to 1.1.0a5 entries below; language reference and `aggregate.ntt` API pages added, docs rebuild pending the author's run. Design, rulings and measurements: `dev/done/plan-nttagg.md`.

## 1.1.0a5 - 2026-10-09

**[NTT-Parallel] `exact_compound(..., workers=n)` threads the per-prime transform phase; `build(..., ntt_workers=n)` forwards it.** Measurement drove the design per the plan: the transform phase is over 90 percent of a dense case, so the per-prime pipeline (now `_residues_mod_prime`) runs on a `ThreadPoolExecutor` (numpy uint64 ufuncs release the GIL; the bigint CRT stays serial). The square-and-multiply walk also stops squaring past the last bit of each gap. Dense 400 x 400: 164s serial before, 119s after the walk fix, 47s with `workers=8`. Serial remains the default and the answer is bit-identical either way; `meta` / `ntt_audit` inputs gain a `workers` entry.

## 1.1.0a4 - 2026-10-09

**[NTT-Exact-Moments] `AggregateNTT.stats_df` gains an `exact` column and validation runs against it.** Frequency and severity raw moments come from the input rationals and aggregate raw moments from the reconstructed numerators (`sum n**k H_n / D`), all exact; `cv` and `skew` take one float square root at the end, so every entry is the correctly rounded float of an exact quantity. The same values overwrite the `mixed` theoretical rows, so `validation_df`, `explain_validation` and the score terms compare the FFT estimate against the exact baseline with no change to `_validation.py`.

## 1.1.0a3 - 2026-10-09

**[NTT-Class] `aggregate.ntt.AggregateNTT`: `build('nttagg ...')` returns the exact compound.** An ordinary `Aggregate` (the float FFT pipeline runs as usual, driven bit-identically from the exact spec) that also computes the distribution exactly: `density_df` gains `p_exact` (reduced `Fraction` per bucket) and `p_exact_float`; new `accuracy_df` (per-bucket `p_exact_float`, `p_fft` from the raw `agg_density`, exact-before-rounding `err`, `abs_err`, `rel_err`) and `ntt_audit` (sizes, primes, timings, error summaries). `update` pins `bs=1`, resolves `log2` from the support, and raises naming the required `log2` when the cap (16 via `build`) is too small; soft guardrails `NTT_MAX_SUPPORT` / `NTT_MAX_BYTES` in `constants.py` refuse monster cases with a full size estimate unless `ntt_override=True`; the 32-bit prime supply is a hard limit. The 100 x 100 dice-of-dice builds in about a second (21 primes, 672-bit denominator) and reproduces the exact endpoint `1/100^101` against a float noise floor near `6e-19`.

## 1.1.0a2 - 2026-10-09

**[NTT-DecL] the `nttagg` DecL kind parses: grammar, transformer, writer, corpus.** `nttagg NAME dfreq [...] [...] dsev [...] [...] trailer` declares an exact bounded discrete compound: probabilities are text-exact (`0.3` is exactly `3/10`, `a/b` is `Fraction(a, b)`, scientific literals are exact, an omitted list is the exact uniform `1/m`), outcomes are nonnegative integers, duplicates combine and sort as in `dhistogram`. New grammar terminals `NTTAGG` and `NTT_NUMBER`; `nttagg` joins the reserved words (ID exclusion, Pygments, Sublime, parse-error labels). `spec_to_decl` renders the new kind with exact `a/b` probabilities and round-trips exactly. Four `NTT.*` lines join `_test_suite.agg` and the spec snapshot (pure additions); the snapshot machinery serializes `Fraction` as `"a/b"`. `build('nttagg ...')` raises NotImplementedError until the `[NTT-Class]` stage wires `AggregateNTT`.

## 1.1.0a1 - 2026-10-09

**[NTT-Engine] new module `aggregate.ntt`: exact bounded discrete compounds via the number-theoretic transform.** Engine functions only, first stage of `dev/plan-nttagg.md`: `exact_compound` (vectorized numpy NTT over 32-bit prime fields with incremental CRT reconstruction, returning exact integer numerators and denominator), `exact_compound_poly` (the slow direct bigint polynomial reference, the tests' independent oracle), `ntt`, `ntt_primes`, and `is_prime_u64` (deterministic Miller-Rabin). Submodule access only, no top-level re-export. New `tests/test_ntt.py` covers primality, prime supply, transform round trips, convolution against numpy, the polynomial oracle, and closed-form rational probabilities of the 100 x 100 dice-of-dice case. The DecL `nttagg` kind and `AggregateNTT` class arrive in later stages.

## 1.0.1 - 2026-10-07

**[Packaging-Subpackages] the wheel ships `aggregate.charts`, `aggregate.plots` and `aggregate.exhibits`, which 1.0.0 omitted.** `[tool.setuptools]` declared `packages = ["aggregate"]`, which names only the top level package: setuptools does not recurse, so 34 of 85 modules were left out of the built artifact. In that wheel `a.plot()` raised `ModuleNotFoundError: No module named 'aggregate.charts'`. Packaging now uses automatic discovery (`[tool.setuptools.packages.find]`, `where = ["src"]`, `include = ["aggregate*"]`). No library code changed.

**1.0.0 was tagged but never published**, and 1.0.1 is the first release on PyPI; there is nothing to upgrade from. The defect was invisible to the test suite by construction, because the suite runs against the editable install where every module is present on disk, so it was found by installing the built wheel into a clean environment. That check is now part of the release procedure.

## 1.0.0 - 2026-10-07

**The 1.0 release.** No code changes from `1.0.0a401`. What changes is the promise: the API stability policy in the preamble above takes effect, and the package classifier moves from `Development Status :: 4 - Beta` to `5 - Production/Stable`. From here a documented name in the stable set keeps its meaning, a breaking change waits for a major release and is preceded by a deprecation period, and the breaking changes recorded in the alpha entries below are what the alpha series was for. `aggregate.charts` and `aggregate.exhibits` stay provisional in the PEP 411 sense and are **not** covered. Requires Python 3.12 to 3.14.

One thing to act on: the published *Pricing Insurance Risk* exhibits do **not** reproduce on 1.0. Much of the machinery they used was removed during the alpha series because it compared recommended spectral methods against legacy ones that are not recommended. Use `pip install aggregate==0.30.1` in an isolated environment, as the README's "Pricing Insurance Risk Examples" section records.

---

Entries before 1.0.0 are in `dev/done/CHANGELOG-before-1.0.md`.
