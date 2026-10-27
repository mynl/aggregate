# TODO

> **Open items only.** A finished item is deleted, not ticked: its CHANGELOG
> lede, its `dev/done/plan-*.md` and the git log are the record. A partly
> finished item is rewritten to what remains; a declined one moves to
> `dev/done/plans-considered-and-rejected.md` with its reason.
>
> **An entry is a reminder, not an argument.** Two or three lines; analysis
> belongs in a `dev/plan-*.md`. The longer entries trimmed on 2026-10-25 are in
> full at `git show 1c2b97b:dev/TODO.md`, and the backlog before 2026-10-07 is
> `dev/done/TODO-2026-10-07.md`. Check both before concluding an item was never
> considered.
>
> **Labels, not codes.** Labels appear in source comments, so renaming or
> deleting an entry means grepping for its label. Numbers in parentheses are
> GitHub issues and pull requests on `mynl/aggregate`.

---

## Strategic

- **[Portfolio-Shared-Mixing-Dependence]**: `Portfolio` units combine by
  independent convolution, and the dependence that exists (the bivariate
  copula, sample-only Iman Conover, allocation-only comonotonic) does not
  compose. Proposed: a mixing variable shared across units, a common shock
  through the PGF machinery. Number one finding of the a89 review. Design
  before code (where Iman Conover and the switcheroo come in), then reweigh
  `bivariate.py`, about 2k LOC for two perils.
- **[DecL-Loss-Tables]**: secondary uncertainty for `elt` (ELT6 without
  `no2u`), executing as `dev/done/plan-elt-uncertainty.md`: model moments with
  no grid landed at 1.15.0 (`[ELT-Uncertainty-Moments]`), then
  `[ELT-Single-Event-Uncertainty]`. `[ELT-Shared-Event-Uncertainty]` stays
  open: a shared event builds with model moments and refuses `update` until
  eltq decides `[Eltq-Windowed-Shared-Events]`. No ELT grid budget
  (`elt_max_log2` was dropped, ruling 18). The deterministic import landed at
  1.11.2 to 1.14.0.
- **[ELT-Eltq-Extra]**: once eltq 2.13 or later is on PyPI, declare the
  optional `elt` extra (`eltq>=` that version) in `pyproject.toml`, add it to
  `all`, and drop the `importorskip` in the ELT6 tests. Until then eltq is
  installed by hand, like `aggregate-ntt`.

## Outside contributions

- **[Snapshot-Portability]** (promised in #8): a clean checkout fails 56 fast
  tests on macOS, mostly last-digit float drift against snapshots captured
  here, plus two tolerance failures from #11 (a near-zero skew at `rtol=1e-6`
  in `test_moment_deficit.py`, an exact `est_cv == 0.0` in `test_moments.py`).
  Make the comparisons portable and keep them as change detectors.
- **[Deductible-Forms-Follow-Ups]** (from the #15 review): (1)
  `make_layer_attachment_pdf` tests `x >= limit` before `x == limit`, so it
  never returns the point mass at the limit; (2) `feasible_bucket` and
  `size_biased_cdf` give no reading under a finite `retention_end`.

## Correctness

- **[Padding-Discard-Measurement]**: have `utilities.ift` report the
  probability its slice to the grid discards, and pass that to
  `xsden_to_mwrangler` as the deficit in place of the raw-sum shortfall. Retires
  the 1e-12 relocation gate and fixes `J.Re14` at tall grids and
  `Bruno2006.B7`, the residual of `[Moment-Deficit-Genuine-Only]`.

## Tests

- **[Rationalize-Tests]**, then **[Scaffold-Retirement]**, in that order:
  rework how the suite consumes the library (the `conftest` parametrization,
  the spec snapshot) without losing coverage, then retire the
  `_test_suite.agg` scaffold, still read by eleven test files and five non-test
  consumers. Evidence in `dev/done/plan-a400-v1-loose-ends.md` phase 4.

## Reporting

## Grammar, parser and unparser

- **[Bivariate-DecL-Label]**: `bivariate Cat as "..."` does not parse, since
  the `bv_out` productions in `decl.lark` have no `as_label`. Grammar change
  plus the `decl_writer` round trip, the reference regen and the corpora.
- **[Unparser-Reference-Gaps]**: shrink `UNPARSER_EXEMPT` in
  `tests/test_agg_libraries.py`, the `library.agg` entries `decl_writer`
  cannot render back as written.

## P&L and bivariate

- **[Massive-Kappa-Second-Sweep]**: kappa scenario percentiles on the massive
  one-sweep P&L route, which needs a second band sweep; until then its
  `stats_df` ladders are marginal. `[Gross-Anchored-Kappa-Insight]` may
  dissolve it.
- **[Waterfall-Limited-Margin]**: analysis in
  `dev/deferred/plan-pnl-evaluate-anchor-DEFERRED.md`.
- **[Reinstatement-Cap-On-Ceded-Agg]**: a plain `agg` after `occurrence ceded
  to` honors `agglimit` but ignores the reinstatement cap `(m+1) s y`; honor it
  as an aggregate stage, as the annual terms are.
- **[Reins-Economics-On-Port-Unit]**: any economics clause in a `port` unit
  raises a raw `TypeError` from `Aggregate.__init__`. Extend the strip and the
  `ceded to` annual rewrite to port units, with a remedy message of their own.

## From the a89 objective review

> Surviving items from the objective library review of 2026-06-21; the review
> itself is in git history.

- **[Esscher-Exponential-Premium]**: the one classical premium principle the
  distortions do not subsume; today only in a `pedagogy.py` figure.
- **[Distortion-Production-vs-Research]**: separate the production distortions
  from the research zoo (CLL, CLin, LEP, LY) in the docs, perhaps in the
  namespace.
- **[Two-Surfaces-Blessed-Path]**: document the blessed path per task: DecL to
  construct, objects to analyze.

## Docs

> No code dependencies. Detail in `dev/done/TODO-2026-10-07.md` under *Docs &
> packaging*.

- **[v1-Journey-Philosophy]**: the intro page and the philosophy statements
  (the user manages logging, warnings and matplotlib; the distribution is
  `pᵢ` at `xᵢ`; `qd` as the fixed-font exception).
- **[Tail-Descriptor-Docs-Tests]**: bounded, log-concave, super-exp, exp and
  sub-exp descriptors for frequency and severity, with tests.
- **[Doc-Gaps]**: custom errors; syntax checker and error reporting; stale
  "site" database references; ZT/ZM; splice examples.
- **[API-Docstring-Coverage]**, **[Docstring-Sweep-NumPy]**: docstring quality,
  Sphinx `:param:` to NumPy style, `iman_conover.py` and `moments.py` first.
- **[Doc-Fix]**: clear the executed-cell `*Error`s in the docs build. Last,
  once the code settles. Plan: `dev/done/doc-fix.md`.
- **[Reinsurance-Case-Study-Docs]**, **[Pedagogy-Docs-Punchup]**,
  **[Reinsurance-Structure-Diagrams]**, **[Cheat-Sheet-Tweaks]**: smaller docs
  items, in the archive.

## Polish

- **[Plot-Argument-Warts]**: `Distortion.plot_affine(ax=)` ignores `ax`;
  `pedagogy.similar_risks_example` raises `TypeError`, passing arguments that
  `Portfolio.plot` and `Bounds.plot_envelope` do not take.
- **[Display-Surface-Punchups]**: `_repr_html_` is on the five first-class
  classes only; `_html_info_blob` is on `Aggregate` alone (a fourth mixin?);
  the P&L builders' `label=name` would render `'PL (PL)'`.
- **[Plotting-Punchups]**: lead item, a `pnl` aggregate plots the aggregate
  only, with no severity overlaid on a payoff.
- **[Validation-Calc-Review]**: audit the validation algorithm against the
  published *Aggregate* paper and make the docs match.
- **[Sdist-Prune-Tests]**: the sdist ships `tests/test_*.py` without
  `conftest.py` or `tests/data`. Add `prune tests` to `MANIFEST.in` at the next
  release and drop the matching exception in the leak check of
  `release-aggregate.ps1`.
- **[NTT-Workers-Per-Engine-Default]**: `ntt_workers` defaults to 4; eight is
  about 25 percent faster on the Rust engine, but 4 is right for pure Python,
  so a change wants a per-engine default.
- **[Library-Round-Two]**, **[Showcase-Examples-Tune]**: library curation
  calls, each needing an author ruling.
- Small hygiene, detailed in the archive: **[bs_describe-Wart]**,
  **[Colorizer-Style-Choice]**, **[Switcheroo-Sample-Regression]**,
  **[Pedagogy-Migrations]**, **[Bivariate-Gate-Flake]**,
  **[Plot-Severity-Outside-Window]**.

## Ideas, not commitments

> Named so they are not reinvented; the archive's *Post-v1.0 ideas* section has
> the rationale for each.

`[Recipe-Doc-Signing]`, `[Bounded-Residual-Lump]`, `[Natural-Lattice-Snap]`,
`[Multi-Resolution-Portfolio-Combine]`, `[Premium-Loss-Algebra-DecL]`,
`[Named-Cherny-Madan-Families]`, `[PnL-Density-DF-Running-Nets]`,
`[Range-Sugar-Round-Trip]`, `[Rate-Based-Reins-Clauses]`,
`[Reinstatement-Event-Date-Terms]`, `[Paper-Reproductions]`,
`[Joint-Padding-Window-Tradeoff]`, `[Multi-Window-Splice-Conditional]`,
`[Data-To-Dsev-Preprocessors]` (its ELT and YLT limb is `[DecL-Loss-Tables]`), `[Parameter-Risk-Predictive]`,
`[Individual-Risk-Model-Census]`.
