# TODO

> **Live items only, post-1.0.** Started 2026-10-07 at `1.0.0a399`, when the
> 1,706-line predecessor was retired to `dev/done/TODO-2026-10-07.md`. That
> archive is the full historical backlog, including every item deliberately not
> carried here; consult it before concluding something was never considered.
>
> The old file was organized around an alpha/beta split for a `1.0.0b1` that was
> never cut. **There is no phase vocabulary here.** Everything in this file is
> post-1.0 unless an entry says otherwise. The 1.0 release items themselves live
> in `dev/done/plan-a400-v1-loose-ends.md` and `V:\worktrees\BETA-MERGE.md`.
>
> **Labels, not codes**, per `CLAUDE.md`. GitHub issue numbers stay in
> parentheses as the stable external cross-reference. Source comments cite
> entries here by label, so **renaming an entry means grepping for it.**
>
> **Keep it short.** The predecessor became unreviewable, which is why it was
> retired. One tight paragraph per entry. If an entry needs more room it wants a
> `dev/plan-*.md`, not more lines here.

---

## The strategic one

- **[Portfolio-Shared-Mixing-Dependence]** — the one item on this list that
  changes what the library *is*, and the number one finding of the objective
  library review taken at `1.0.0a89`. `Portfolio` units combine by
  **independent convolution** (`_portfolio_density`, the independent-sum FFT
  combine). The dependence that does exist is scattered across three mechanisms
  that **do not compose**: the two-peril copula in `bivariate.py`, sample-only
  Iman Conover, and allocation-only comonotonic. For a capital-allocation tool
  the review called this "the gap users will find first", and 1.0 ships with it.
  Proposed clean fix: a **mixing variable shared across units**, a common shock,
  composing with the existing PGF machinery to give principled n-unit
  dependence without copulas. The author's annotation on the review: this is
  where Iman Conover and the switcheroo come in. **Design before code.**
  Knock-on to record while designing: `bivariate.py` is roughly 2k LOC serving
  exactly two perils, and its cost against benefit is worth rereading once a
  shared-mixing story exists. The author closed the front-page scope question
  on 2026-10-07: the README and the ~30 worked reproductions already say what
  the library does, so no scope-disclaimer section is owed.

## In flight

- **[NTT-Tail-Visibility]** (`dev/plan-ntt-tail-visibility.md`) — phase 1
  `[NTT-Accuracy-Overflow]` landed at **1.2.0a6**: `ntt_audit` and
  `accuracy_df` no longer raise on any case whose exact tail falls below the
  double range (every uniform dice from about 150 x 150 up), and
  `max_rel_err` is restricted to buckets the float pipeline can actually
  resolve, so it reads 0.078 rather than 7.2e304 on a 200 x 200 case. Phase 2
  `[NTT-Exact-Log10-Lazy]` landed at **1.2.0a7**: `density_df` carries
  `p_exact_log10`, which reads -462.5 on a bucket whose `p_exact_float` has
  underflowed to zero, and `p_exact` moved from an eagerly built column to
  the cached `a.p_exact` property, taking first access to `density_df` on a
  500 x 500 case from 5.8 s to 0.4 s. That mattered more than it looked:
  the Fractions would have been 74 percent of the wall clock once the Rust
  engine lands. **Both phases are done; the plan is retired to
  `dev/done/`.**


- **[NTT-Exact-Object-Layer]** (**DONE**, **1.2.0a11** and **1.2.0a12**) —
  the Python object layer around the exact answer, which became the
  bottleneck once the compiled engine made the computation cheap. Two
  fixes, both measured on a 500 x 500 case. `accuracy_df` lost its
  Fractions for an exact integer identity, 14.66 s to 0.92 s, bit-identical
  (`[NTT-Accuracy-Integer]`). And `p_exact_at(loss)` answers for named
  buckets in 12 microseconds against 5.65 s for the full `p_exact`, which
  stays as the opt-in Series it already was (`[NTT-Exact-At]`). Getting
  everything on a 500 x 500 case went from 22.8 s to 3.4 s, of which the
  exact computation is 0.7 s. The remaining cost is intrinsic: a full
  `p_exact` is one gcd per bucket against a 9,976-bit denominator, and the
  ruling of 2026-10-10 is that the caller who wants a million reduced
  rationals is the right one to pay for it. A compiled query object
  answering moments and tail probabilities without materializing
  coefficients is `[Rust-Query-Object]` in the sibling repo, deferred there
  and not needed by anything here.

- **[NTT-Rust-Integration]** (**DONE**,
  `dev/done/plan-ntt-rust-integration.md`, phases landed at **1.2.0a8**,
  **1.2.0a9** and **1.2.0a10**) — use the compiled engine in
  `V:\dev\aggregate-ntt` when it is installed, and behave exactly as today
  when it is not. The engine is a separate optional Rust package, discovered
  at import time and never declared as a dependency, which keeps this library
  pure Python, keeps its BSD license clear of GMP, and lets the author build
  for one machine. This library keeps input validation, prime selection, the
  guardrails, the audit and the three end-to-end checks, so the engine is
  verified rather than trusted; it gets the per-prime transform and the CRT
  reconstruction. New keyword `ntt_engine='auto'|'python'|'rust'`, provenance
  rows in `ntt_audit`, and a versioned seam that **refuses** on protocol
  mismatch rather than computing. The **reconstruction phase landed at
  1.2.0a8** against the engine's 1.0.0a2, bit-identical to the Python path
  and taking the headline 1000 x 1000 case from 212.3 s to 39.2 s. **What
  **Both phases are integrated.** Reconstruction at 1.2.0a8, the `max_bits`
  prime-width contract at 1.2.0a9 (`[Rust-Engine-Prime-Width]`), and the
  transform at 1.2.0a10 against engine 1.0.0a4, which took the headline
  1000 x 1000 case to 10.71 s from 212.3 s with bit-identical answers. The
  seam sits at protocol 2 and proved itself in a real drift event: a
  1.0.0a4 engine met a protocol 1 library and fell back with a warning
  rather than computing. **One item left for the author**, no longer
  pressing: the `ntt_workers` default of 4 could be 8. Measured at 1.2.0a8,
  when the Python transform dominated, four threads ran 3.57 s against
  1.92 s for sixteen, which looked urgent; with the Rust transform the whole
  phase is 18x faster and the headroom is 25 percent (500 x 500: 1.12 s at
  four, 0.84 s at eight, 0.80 s at sixteen; 1000 x 1000: 11.10 s at four,
  8.30 s at eight). Raising it would also move the pure-Python path, where 4
  is right, so it wants a per-engine default if it is worth doing at all.
  Its reflected half is `dev/plan-rust-engine.md` in that repo.

- **[NTT-CRT]** (**CLOSED**, `dev/done/plan-ntt-crt.md`) — stage 1
  `[NTT-CRT-ProductTree]` landed at **1.2.0a4**: the reconstruction is a
  balanced product tree built as an online merge stack, taking the headline
  1000 x 1000 case from 594 s to 212 s and its reconstruction phase from
  590 s to 198 s, and unstarving the transform threads as a side effect
  (500 x 500 transform thread-seconds 266 to 119). **Two calls now sit with
  the author.** Stage 2 `[NTT-CRT-ProcessPool]`: the plan set about 100 s on
  the headline case as the bar for dropping it, and stage 1 landed at 212 s,
  so it is neither clearly wanted nor clearly dropped. Weigh it against
  pickling hundreds of megabytes of bigints across a process boundary, and
  against the remaining shape of the cost, which is now 198 s of
  reconstruction against 105 s of serial transform; **superseded** by
  `[NTT-Rust-Integration]` above, since threads without a GIL are what it
  tried to approximate with processes. Stage 3 `[NTT-CRT-Gmpy2]` is
  specified and measured at 2.0x on the reconstruction (a hybrid keeping
  CPython integers at the bottom of the product tree and promoting to `mpz`
  at 16 primes, since GMP is 7x on the wide merges and slower on the narrow
  ones), and is **SHELVED** as the fallback if the Rust engine stalls. Do not
  implement it without the author saying so. The `ntt_workers` default was
  settled at **1.2.0a5**,
  `[NTT-Workers-Default]`, narrowing from `min(8, ...)` to `min(4, ...)`
  on the measurement that four threads match eight for half the CPU.

## Outside contributions

> Work an outside contributor is doing, and what we promised them in return.
> Rulings and promises were posted on GitHub on 2026-10-13. Contributors leave
> the version, `CHANGELOG.md` and this file alone; we add all three at merge.

- **[Deductible-Vocabulary]** (#9): franchise **landed at 1.7.0** (#15), the
  aggregate deductible library entry at 1.6.0 (#16), disappearing pending in
  #20 with the owed corpus items (a `_test_suite.agg` line, the unparser
  corpus and the snapshot recapture, for both forms). Revised ruling on #15:
  the payment is `min(x - r(x), y)`, deductible first and limit second, so
  `limit` is the largest payment and the only constraint is `d1 <= d2`. One
  transform on the continuous severity with `d2` (`retention_end`) as its
  parameter: `inf` ordinary, `d1` franchise, `d1 < d2` disappearing.

- **[Deductible-Forms-Follow-Ups]** (from the #15 review): (1) the old
  `make_layer_attachment_pdf` tests `x >= limit` before `x == limit`, so it
  never returns the point-mass marker at the limit (`pdf(1000)` is 0 for
  `1000 xs 50` on a lognormal); the retention wrapper tests the mass first.
  (2) `feasible_bucket` and `size_biased_cdf` return their no-reading values
  under a finite `retention_end`, so bucket sizing has no feasibility reading
  for a franchise or disappearing deductible. (3) The `qd` one-liner prints
  `1,000 xs 50` with no modifier for a franchise.

- **[Snapshot-Portability]** (promised in #8): untouched `master` at 1.0.1
  fails 56 fast tests on macOS (Apple Silicon): 44 exhibit snapshots, chart
  renders, `test_baseline`, and the renewal and peg regressions, apparently
  last-digit float differences from snapshots captured on this machine. Make
  the comparisons tolerant enough that a clean checkout passes on macOS and
  Linux without losing their value as change detectors. Promised, so owed.
  Two more from #11, failing by tolerance alone:
  `test_moment_deficit.py::test_signed_window_deficit_is_genuine_and_still_relocates`
  compares a skew near zero at `rtol=1e-6`, and
  `test_moments.py::test_deterministic_wait_realized_cv_is_zero` asserts
  `est_cv == 0.0` exactly where macOS gets `2.4e-8`.

## Recently landed

- **[Library-Figures]** DONE at **1.6.0a2** and **1.6.0a3** (2026-10-19,
  `dev/done/plan-library-figures.md`, the library half of house-style's plan
  of the same name): `aggregate.plots.fit_page(width, max_height)` fits every
  library figure to a document's page, and `aggregate.mplstyle` takes the
  house type and face with no type size left in library code. house-style
  calls `fit_page` from its setup once 1.6.0 is on PyPI.

- **[DecL-Word-List]** DONE at **1.6.0a1** (2026-10-19,
  `dev/done/plan-decl-word-list.md`): `aggregate.decl_words.decl_words()` is
  the one DecL word list, derived from `decl.lark` and the distortion
  registry where they hold the words, and `AggLexer` builds from it. The
  consumers are the other repositories' plans of the same name: aLL serves it
  at `GET /v1/decl/words` and retires `decl-keywords.json`
  (`[Keyword-Mirror-Gaps]` there), aSP drops its inline regex, house-style
  generates `decl.xml` from it. `_TERMINAL_LABELS` and `agg.sublime-syntax`
  stay hand kept and test checked.

- **[Thread-Safe-Load]** DONE at **1.5.2** (2026-10-14,
  `dev/done/plan-1.5.2-thread-safe-load.md`, raised from aLL): `load`, `fork`
  and `reload` run under a per-instance `RLock`, and `_loaded` turns True only
  once the read finishes, so no thread can look up in or fork a half-read
  base; a failed configured read rolls back and retries. aLL's startup read
  (api 1.13.4) stays, since it keeps the read off the first request, but its
  comment in `create_app` describing the unlocked load goes stale once the
  api's floor reaches 1.5.2.

- **[Zero-Share-Layer-Moments]** DONE at **1.5.1** (2026-10-14, #11, the
  contributor's fix): 1.2.0a14 cleaned the occurrence reproducer only because
  its subject deficit was below `VALIDATION_NOISE`, and an aggregate stage
  `0% po` layer over a subject with a genuine deficit still reported a mean of
  `1.2e-6`. `_reinsurance.cedes_nothing` now gives any zero cession the exact
  point mass at zero (`zero_cession_density`), and the `CS.Gapped` tier
  exclusion in `tests/test_chart_structure.py` is lifted.

- **[Allocation-Bounds-Gold-Standard-Tests]** DONE (2026-10-14, #10,
  squash-merged as `35a0624`, no version bump): the contributor's 21 tests in
  `tests/test_allocation_gold_standard.py`, Poisson thinning identities under
  PH, Wang, TVaR and dual, common scale gamma, a constant unit, and
  `hypothesis` invariants. `hypothesis` joined the `dev` extra, so sync with
  `uv sync --all-extras --inexact`. No snapshots moved and #8 is closed, so
  nothing promised is still owed.

- **[NTT-Exact-Tail-Chart]** DONE at **1.5.0** (2026-10-14,
  `dev/done/plan-1.4.0-nttagg-arithmetic-exact-chart.md` step 2): the `exact_tail`
  chart draws an nttagg's exact log10 mass and survival beside the float FFT,
  with the new axis units `log10_density` and `log10_probability`. The aLL
  leaf is `aggregate-api/dev/plan-1.13.0-exact-tail-leaf.md`.

- **[NTT-Rational-Arithmetic]** DONE at **1.4.0** (2026-10-14,
  `dev/done/plan-1.4.0-nttagg-arithmetic-exact-chart.md` step 1): nttagg
  probabilities accept parenthesized arithmetic and `**` with a whole-number
  power, evaluated exactly as `Fraction`.

- **[DecL-Help-Pages]** DONE at **1.2.0a15** (2026-10-13,
  `dev/done/plan-1.2.0a15-decl-help.md`):
  `decl_help()` and `decl_help('card')` ship the anatomy and the one-page card
  as HTML fragments from `data/decl-help/`, every program built by
  `tests/test_decl_help.py`. The api half is
  `aggregate-api/dev/plan-1.7.0-decl-help.md`. Divergences and the
  measurements are in the plan's execution log.

- **[NTT-Speed]** DONE at **1.2.0a1** and **1.2.0a2** (2026-10-09,
  `dev/done/plan-ntt-speed.md`): geometric spectral runs for equal-weight
  outcomes and `out=` kernels throughout took the 1000 x 1000 uniform dice case
  from about an hour to 594 s, and the dense 400 x 400 case from 119 s to
  13.8 s. `ntt_workers` now defaults to `min(8, os.cpu_count())` and
  `ntt_audit` reports it. Stage 3, a numba Montgomery transform engine, was
  **dropped** on the measurement: the transform phase is 0.7 percent of the
  headline wall clock. The CRT work it points at is `[NTT-CRT]` above.

- **[NTT-Composability]** DONE at **1.2.0a3** (2026-10-09,
  `dev/done/plan-nttagg-composability.md`): `pnl_program` and `reins_program`
  are absent on an `nttagg` instead of present and refusing, so `hasattr` is
  the honest capability probe a consumer can navigate by, and `library.agg`
  carries `ExactDiceOfDice`. Raised from `aggregate_api`, where the defect was
  visible. One divergence: the generic kind guards in `_program.py` stay, the
  plan's premise for deleting them (an `'nttagg'` arm in each) not matching the
  code.

- **[NTT-Exact-Aggregate]** DONE at **1.1.0** (2026-10-09,
  `dev/done/plan-nttagg.md`): the `nttagg` DecL kind and
  `aggregate.ntt.AggregateNTT` compute bounded discrete compounds exactly
  (NTT over prime fields plus CRT) alongside the float pipeline, with
  `accuracy_df`, `ntt_audit`, exact `stats_df` moments feeding validation,
  and a measured thread-pool transform phase. Staged 1.1.0a1 to a5 then
  1.1.0; divergences and measurements in the plan's execution log.

## Correctness

- **[Unit-Capital-Above-Frame]** DONE at **1.5.3** (2026-10-19, #13, from
  #12; was `[Unit-Capital-Far-Tail]`): above the augmented frame,
  `unit_capital_at` and `pentagon_at` use the last row and warn, and a warning
  fires whenever `a` reaches layers with `S <= VALIDATION_NOISE`, whichever
  units are requested. The far tail half stays declined: no renormalizing by
  the sum of unit margins and no share inheritance, since the warning is the
  intended signal.
- **[Fuzz-Removal-Phantom-Tail-Mass]** (was `[Far-Tail-Raw-Moment-Inflation]`;
  diagnosed 2026-10-07, second occurrence 2026-10-11, full account in
  `dev/done/plan-a400-v1-loose-ends.md` phase 3 and
  `dev/done/plan-1.2.0a14-moment-deficit.md`): **landed 1.2.0a14** as
  `[Moment-Deficit-Genuine-Only]`: the moment sites clean at the density's
  own noise floor (`utilities.noise_floor`, capped at machine epsilon) and
  `xsden_to_mwrangler` relocates only a positive deficit above
  `VALIDATION_NOISE`, measured on the raw array. The plan and its note move to
  `dev/done/` when the author says so. **Residual:** a positive rounding
  deficit above 1e-12 still relocates (`Bruno2006.B7`'s mirror image, `J.Re14` at
  `log2` 18 and 20), today's behavior. The principled fix is
  `[Padding-Discard-Measurement]`, next.
- **[Padding-Discard-Measurement]** (proposed 2026-10-12, the residual of
  `[Moment-Deficit-Genuine-Only]`): when the padded inverse transform slices
  its result to the grid, remember how much probability the slice discarded
  and pass that, not the raw-sum shortfall, to `xsden_to_mwrangler` as the
  deficit. The choke point is `utilities.ift`, not the convolution: every
  path, Aggregate and Portfolio alike, takes its inverse through `ift`, and
  `temp[0:len(temp) >> padding]` is the one line that drops mass, so if `ift`
  reports `temp[n:].real.sum()` beside the density (a second return or an
  attribute on the result) both paths get the measurement for free and the
  Portfolio caveat (`p_total` is a product of transforms) disappears.
  Relocation then becomes "put back exactly what the padding dropped", so the
  one-sided 1e-12 gate in the wrangler, which exists only because the raw-sum
  deficit mixes truncation with rounding, retires, and `J.Re14` at tall grids
  and `Bruno2006.B7` fall out as consequences. With `padding=0` the
  measurement reads 0, which is right: nothing was dropped, the tail wrapped
  instead, and aliasing is the separate problem the existing checks report.
  Acceptance: the `test_moment_deficit.py` residual test flips from
  documenting the old value to matching the no-relocation reading, and the
  sweep in `scripts/moment_rule_sweep.py` shows no case worse than theory
  by more than today's.

- **[Matrix-Verdict-Colorblind]** (measured 2026-10-07) — the `matrix` panel
  encodes its verdict on a red/green diverging ramp whose two ends have a
  luminance contrast ratio of **1.25:1** (`MATRIX_FAVORABLE` `#008300`,
  relative luminance 0.1623; `MATRIX_UNFAVORABLE` `#e34948`, 0.2156). They
  differ almost purely in hue, on the red/green axis, so for a deuteranope or
  protanope (~8% of men) and in greyscale print the two verdicts are
  indistinguishable. That matters more here than on an ordinary chart: the
  numbers are printed in every cell, so magnitude survives, but **which
  direction is favorable is row-dependent and carried by color alone** (
  `row_polarity`), and `tests/test_chartdoc_matrix.py` says in as many words
  that it "is the one thing a reader cannot check by eye against the numbers".
  The ramp was a deliberate choice (see the comment above the constants in
  `plots/_chartdoc.py`: a verdict rather than a magnitude), so this is a
  revisit, not a bug report. Three ways out: keep the hues but separate their
  luminance, switch to a colorblind-safe diverging pair, or add a redundant
  non-color cue per cell. **Blocks nothing**, but it should be settled before
  the matrix baseline image is blessed, since blessing freezes the picture.

- **[Wheel-Smoke-Check]** (**DONE**, **1.6.1**, #17): `scripts/check_wheel.py`
  builds the wheel, checks its listing against `src/aggregate` (packages,
  modules, package data) and smoke tests it in a throwaway venv from outside
  the repository. A release-time command, not part of the suite. It replaces
  the manual clean-venv install step; run it at every release.

## Tests and the parity scaffold

- **[Rationalize-Tests]** (#51), then **[Scaffold-Retirement]** — **in that
  order**, and both deliberately deferred past 1.0. The `_test_suite.agg` /
  `_test_suite2.agg` SLY-parity scaffold has done its job, but retiring it is
  not the small deletion the old TODO described: **eleven** test files read it,
  four of them collecting 1,614 cases, and `test_decl_parser` reads it and
  nothing else. Five non-test consumers too (`config.TEST_SUITE_FILENAME`,
  `Underwriter.test_suite_file` and `interpret_file`'s default, the shipped
  `config.default.toml`, and both `scripts/` baseline scripts). The evidence and
  the argument for deferring are in `dev/done/plan-a400-v1-loose-ends.md` phase 4.
  `[Rationalize-Tests]` is the prerequisite: rework how the suite *consumes* the
  single library (the `conftest` parametrization, the spec snapshot) without
  losing coverage, and then the scaffold can go.

## Reporting

- **[Reporting-Guidelines]** — the half that remains is what the reports
  **contain**. The "what is a first-class citizen" half shipped at `1.0.0a170`
  (`[FCC-Contract]`): the membership rule and required surface are declared in
  `constants.FIRST_CLASS_CLASSES` / `FCC_REQUIRED`, audited by
  `dev/regen_features.py`, asserted by `tests/test_fcc_surface.py`, and written
  up in `docs/2_aggregate_overview/features.rst`. Still open: rows are fixed,
  columns are pure (one unit per column, currency and ratios never mixed),
  headings are presentation-ready, and the `summary` / `validation` /
  `reins_<flavor>` split is settled. Plan: `dev/reporting-guidelines.md`.
  **Gates `[Accounting-Summary-DF]`**; scope the two together.
- **[Accounting-Summary-DF]** (pended 2026-07-04, unblocked since `1.0.0a136`) —
  the gross/ceded-**split** consolidated P&L card, `accounting_summary_df`:
  Consideration gross premium / ceded premium / total, Obligation gross loss /
  ceded loss / total, Margin total. Strictly correct, since consideration and
  obligation cannot be netted, and it supports GAAP / STAT / IFRS shapes. Plain
  `summary_df` stays the simple net card; this is the detail view between it and
  the full `xpnl` walk.

## Grammar, parser and unparser

- **[Bivariate-DecL-Label]** — `BivariateAggregate` became a `LabeledMixin` host
  at `1.0.0a171`, but its **object-level** label has no DecL spelling: the nine
  `bv_out` productions in `decl.lark` carry no `as_label`, so
  `bivariate Cat as "..."` does not parse and the label is set with `label=`.
  Component labels already work, each unit being an ordinary `agg`. Closing it
  is a grammar change plus the `decl_writer` round-trip, the grammar-reference
  regen, and the `test_decl_unparser` / `test_grammar_sync` corpora.
- **[Unparser-Reference-Gaps]** (surfaced by `[Library-Canonical-Layout]`, a178)
  — `decl_writer` cannot render some constructs back to what was written, so a
  set of `library.agg` entries are exempt from the canonical layout and
  hand-written. The list is `UNPARSER_EXEMPT` in `tests/test_agg_libraries.py`;
  **shrinking it is the progress measure.** `a320` taught `_render_splice` the
  compact one-list form, and the `a327` wave grew the list into a catalogue of
  every source spelling the unparser cannot recover.
- **[Session-Build-Clobbers-The-Trailer]** (rewritten for the general case a300)
  — a session `build(...)` **re-registers** its entry in the shared recipe base,
  so building a program carrying less trailer than the library entry of the same
  name silently replaces that entry's `note{}` and `tags{}` with nothing. The
  concrete trigger that found it is gone (the recipe-run path went at a300), but
  **the overwrite behavior survives** for a user who shadows a library name.
- **[Power-Variance-Family]** (from `dev/done/plan-tweedie.md`) — the `tweedie`
  clause covers only the `1 < p < 2` slice, while the `Tweedie` class already
  spans the whole power variance family: Gaussian at 0, Poisson at 1, gamma at
  2, inverse Gaussian and Lévy at 3, extreme stable outside. Extend the DecL
  surface to the rest. Depends on `[Tweedie-Round-Trip]` first; the naming, the
  clause shape, and which members deserve a keyword are all open.

## P&L and bivariate

- **[Massive-Kappa-Second-Sweep]** (from `[PnL-Punchups-01]`, `1.0.0a134`) —
  bring the kappa scenario percentiles to the massive one-sweep P&L route.
  Conditioning needs the joint per atom *and* the grand-result quantiles before
  indicator-weighted means can accumulate, so it is a second band sweep. Until
  then the massive `stats_df` keeps **marginal** ladders, visibly so since
  `a136` (plain `P01…P99` headers against the in-memory scenario `κ` columns).
  The 2-D follow-up's `[Gross-Anchored-Kappa-Insight]` may largely dissolve
  this.
- **[Waterfall-Limited-Margin]** — from
  `dev/deferred/plan-pnl-evaluate-anchor-DEFERRED.md`, which carries the
  analysis.

## From the a89 objective review

> The surviving strategic items from the objective library review of
> 2026-06-21, taken around `1.0.0a89`. Its verdict: the engine and the pricing /
> allocation science are distinctive and ready; the gap was n-unit dependence
> (above) plus a finished experience layer. The review page itself
> (`dev/REVIEW.md`) and the verbatim source it condensed
> (`dev/beta-review-2026-06-21.md`) are both **deleted** and recoverable from
> git history; this section is the record.

- **[Esscher-Exponential-Premium]** — the one classical premium principle the
  distortion framework does not subsume. Exists only inside `pedagogy.py`'s
  premium-principle comparison figure, not on the pricing surface.
- **[Distortion-Production-vs-Research]** — separate the production distortions
  from the research zoo (CLL, CLin, LEP, LY) in the docs, and possibly in the
  namespace, so a new user meets the handful they should reach for.
- **[Two-Surfaces-Blessed-Path]** (rider on the docs work) — there are two
  expressive surfaces, DecL and the objects. Document the blessed path per
  task, DecL to construct and objects to analyze, so users do not infer it.

## Docs

> None of these has a code dependency. Full detail for each is in
> `dev/done/TODO-2026-10-07.md` under *Docs & packaging*.

- **[v1-Journey-Philosophy]** (#15) — the intro / "Journey" page and the
  statements of philosophy (the user manages logging, warnings and matplotlib;
  the distribution **is** `pᵢ` at `xᵢ`, no jump detection; `qd` is the doc-only
  fixed-font exception). The warnings half is settled by `[Warning-Policy]`
  (`1.0.0a219`) and needs only writing up.
- **[Tail-Descriptor-Docs-Tests]** (#41, #42) — bounded, log-concave, super-exp,
  exp and sub-exp descriptors for frequency **and** severity, plus the
  bounded/unbounded indicator, with tests.
- **[Doc-Gaps]** (#58 to #62) — custom errors; syntax checker and better error
  reporting; stale "site" database references; ZT/ZM zero-truncation and
  modification; splice examples.
- **[API-Docstring-Coverage]** and **[Docstring-Sweep-NumPy]** (#18) —
  docstring *quality*, not missing pages: page coverage finished at a196
  (`[Reference-Chapter-Audit]`). Sphinx `:param:` to NumPy style in
  `iman_conover.py` / `moments.py` and pockets elsewhere, public surface first.
- **[Doc-Fix]** — clear the executed-cell `*Error`s in the docs build. **Last**,
  after the code has settled, since it churns with every API change. Plan:
  `dev/done/doc-fix.md`.
- **[Reinsurance-Case-Study-Docs]**, **[Pedagogy-Docs-Punchup]**,
  **[Reinsurance-Structure-Diagrams]**, **[Cheat-Sheet-Tweaks]** — the smaller
  docs items, detailed in the archive under *Docs*.

## Polish

> Carried as a list, not as paragraphs. Each is detailed in the archive.

- **[Plot-Argument-Warts]** (found at 1.6.0a2): `Distortion.plot_affine(ax=)`
  accepts `ax` and ignores it (`plots/_distortion.py`); and
  `pedagogy.similar_risks_example` raises `TypeError`, calling
  `Portfolio.plot(figsize=)` and `Bounds.plot_envelope(axs=)`, neither of
  which takes that argument.
- **[Display-Surface-Punchups]** — `_repr_html_` is on the five first-class
  classes only; `_html_info_blob` exists on `Aggregate` alone while four other
  classes build the same intro inline (the obvious fourth mixin); and the P&L
  builders pass `label=name`, so `LabeledMixin._title_name` would render
  `'PL (PL)'` if anything on `PnL` called it.
- **[Plotting-Punchups]** — lead item: a `pnl` aggregate should plot the
  aggregate **only**, with no loss-convention severity overlaid on a payoff.
  Gate on `_agg_affine_active()` / `_signed()` in `plots/_aggregate.py`.
- **[Validation-Calc-Review]** (#49) — audit the validation algorithm against
  the published *Aggregate* paper and make the docs match. The switches-to-config
  half is done; the aliasing half shipped as `[Validation-Punchup]`.
- **[Input-Guards]** (#18, ruled 2026-10-19): a **fixed** frequency whose
  count is not whole (directly, or derived from loss or premium) builds
  silently with negative probabilities; it will raise `ValueError` within a
  small tolerance, pointing at `dfreq` and `poisson`. Contributor PR to
  follow. The old "zero `lb`" and "mixing with an inconsistent frequency"
  notes are dropped: neither could be reconstructed, and the grammar already
  rejects mixing with a second frequency.
- **[Sdist-Prune-Tests]** The sdist ships setuptools' default
  `tests/test_*.py` (139 files since 1.0.1) without `conftest.py` or
  `tests/data`, so the tests cannot run from it. Add `prune tests` to
  `MANIFEST.in` at the next release, then drop the matching exception in the
  leak check of `release-aggregate.ps1`.
- **[Library-Round-Two]**, **[Showcase-Examples-Tune]** — library curation
  calls, each needing an author ruling rather than a rule.
- **[bs_describe-Wart]**, **[DecL-Colorizer-Resync]**, **[Colorizer-Style-Choice]**,
  **[Switcheroo-Sample-Regression]**,
  **[Pedagogy-Migrations]**, **[Bivariate-Gate-Flake]**,
  **[Plot-Severity-Outside-Window]** — small hygiene items.

## Ideas, not commitments

> The archive's *Post-v1.0 ideas* section holds seventeen of these with full
> rationale. Named here so they are not reinvented:
> `[Recipe-Doc-Signing]`, `[Bounded-Residual-Lump]`, `[Natural-Lattice-Snap]`,
> `[Multi-Resolution-Portfolio-Combine]`, `[Premium-Loss-Algebra-DecL]`,
> `[Named-Cherny-Madan-Families]`, `[PnL-Density-DF-Running-Nets]`,
> `[Range-Sugar-Round-Trip]`, `[Rate-Based-Reins-Clauses]`,
> `[Reinstatement-Event-Date-Terms]`, `[Paper-Reproductions]`,
> `[Joint-Padding-Window-Tradeoff]`, `[Multi-Window-Splice-Conditional]`,
> `[Data-To-Dsev-Preprocessors]`, `[Parameter-Risk-Predictive]`,
> `[Individual-Risk-Model-Census]`.
>
> Rejected, and recorded as such in
> `dev/done/plans-considered-and-rejected.md`: `[Display-Mode]`, and the
> never-Panjer ruling.
