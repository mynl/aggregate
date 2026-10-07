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

## Correctness

- **[Zero-Share-Layer-Moments]** (found a352) — a `0% po` layer cedes nothing,
  so all its ceded moments are pure floating-point noise, and every ratio
  derived from them is meaningless. The defect has **two faces** depending on
  which way the noise falls. When the second moment lands just below zero,
  `moments.mcvsk` square-roots it and `reins_stats_df` reports `cv` and `skew`
  as `nan` beside a mean of `-1.8e-11`. When it lands just above, there is no
  `nan` and no warning, and the ratios explode instead: measured 2026-10-07 on
  `agg Z 10 claims sev lognorm 50 cv 1.5 occurrence net of 20 xs 20 and 0% po
  20 xs 40 and inf xs 60 poisson`, the zero-share layer reports `agg` `cv` and
  `skew` of **2.51e+06** against a mean of `5.2e-09`, and `sev` `cv` / `skew`
  of `nan`. The second face is the worse one: a plausible-looking number in a
  reported frame with nothing flagging it. Fix is not an `np.errstate` guard:
  recognize a zero-share layer and report it as such (zeros or blanks across
  the column, the author's call) rather than publishing ratios computed from
  noise. Partly masked because the gapped program is exercised on the declared
  tier only; see the `_EVERY_TIER` comment in `tests/test_chart_structure.py`.
  **Proposed for 1.0** in `dev/done/plan-a400-v1-loose-ends.md`; if it does not make
  the cut it lands here.
- **[Fuzz-Removal-Phantom-Tail-Mass]** (was `[Far-Tail-Raw-Moment-Inflation]`;
  **diagnosed 2026-10-07**, full account in
  `dev/done/plan-a400-v1-loose-ends.md` phase 3) — on a tall grid
  `utilities.remove_fuzz` zeroes *genuine* far-tail density, and
  `moments.xsden_to_mwrangler` then relocates the shortfall to the top of the
  grid, where `x**3` turns it into a large number. On `lognorm 200 cv 2` at
  `bs=6.427`, `log2=21`: genuine far-tail density is ~`1e-26`, well under
  `remove_fuzz`'s **absolute** `eps=2.22e-16`, so it zeroes 1,977,176 of
  2,097,152 buckets and removes `4.632e-12` of real mass; the wrangler reads
  that as a truncation deficit and adds `pg * (xs[-1]+bs)**3 = 4.632e-12 *
  2.449e21 = +1.134e10` to an `ex3` of `3.0e10`. Result: central third moment
  +113%, `est_skew` **7.537** against an exact **3.5355**, and a spurious
  `AGG_SKEW` validation flag. The **raw** density gives 3.5354, correct, so the
  FFT is innocent. The original entry blamed `1 - cdf` and the raw third
  moment; both are wrong, and the raw third moment is accurate to ~1e-5 at
  every grid.
  **Fix needs a ruling, because all three candidates move `est_skew` and can
  move validation flags corpus-wide.** Recommended: make `remove_fuzz`'s
  threshold relative to the density's own scale instead of an absolute epsilon
  of 1.0, which is the root error, as its own plan with a blast-radius pass.
  Cheap interim: stop the wrangler relocating a fuzz-attributable shortfall to
  the grid top. Not recommended without more work: dropping the `remove_fuzz`
  call in `_aggregate.py` (measured better at every grid tested, but that call
  exists *because* fuzz once corrupted the skew, and that case was not
  re-found).
  **Owed either way:** two comments assert things now measured to be false.
  `remove_fuzz`'s docstring says "the exact aggregate has no genuine density
  below machine epsilon", and the comment above the call in `_aggregate.py`
  adds that the aggregate "has no negative density even under aliasing" (795,744
  buckets are negative at `log2=21`).

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

- **[Wheel-Smoke-Check]** (found the hard way at 1.0.1) — a packaging defect is
  **invisible to pytest by construction**: the suite runs against the editable
  install, where every module is present on disk, so `packages = ["aggregate"]`
  silently dropped `aggregate.charts`, `.plots` and `.exhibits` from the 1.0.0
  wheel and `a.plot()` raised `ModuleNotFoundError` for anyone who installed
  it. Nothing in the repo would ever have said so. Wanted: a
  `scripts/check_wheel.py` that runs `uv build`, installs the wheel into a
  throwaway venv, and asserts the smoke set (import, `build(...)`, `.plot()`,
  each subpackage, a `library.agg` recipe, `qd`), plus a listing check that
  every `src/aggregate/**/__init__.py` has a counterpart in the artifact. A
  release-time command like the numerics gate, **not** part of the suite, since
  it needs a built artifact. Until it exists, the clean-venv install is a
  manual step in `V:\worktrees\BETA-MERGE.md` and must not be skipped.

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

- **[Deductible-Vocabulary]** — franchise and disappearing deductibles, and an
  annual aggregate deductible, as first-class constructs. Corridor landed on
  the reinsurance side, so these are what remains of the review's
  richer-deductibles ask. Nothing in the grammar or engine matches `franchise`,
  `disappearing` or an aggregate deductible today.
- **[Esscher-Exponential-Premium]** — the one classical premium principle the
  distortion framework does not subsume. Exists only inside `pedagogy.py`'s
  premium-principle comparison figure, not on the pricing surface.
- **[Distortion-Production-vs-Research]** — separate the production distortions
  from the research zoo (CLL, CLin, LEP, LY) in the docs, and possibly in the
  namespace, so a new user meets the handful they should reach for.
- **[Allocation-Bounds-Gold-Standard-Tests]** — extend the gold-standard
  testing style past the engine into allocation and bounds: closed-form checks
  wherever an analytic allocation exists, plus `hypothesis` invariants (`q`
  monotone, TVaR at least VaR, allocation sums to the total, net at most
  gross).
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
- **[Input-Guards]** — zero `lb` not consistent with attachment equals zero;
  flag **fixed** frequency with a non-integer expected value; flag **mixing**
  with an inconsistent frequency distribution.
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
