# TODO

> **Live items only.** What has landed is in `CHANGELOG.md` and the git log, and
> is **removed from here**. The master narrative — the six steps to v1.0, the
> beta-tag checklist, the show-off example, the monograph/cookbook plan — is
> `../plan-for-v1.md`; this file carries the item-level detail behind it.
>
> **Phase.** `alpha` = must finish before cutting `1.0.0b1`; `beta` = fine just
> after the alpha→beta cut. Release mechanics (version/CHANGELOG/LICENSE
> agreement, the `uv run pytest -m 'slow or not slow'` green run, fresh-env
> install, tagging, the branch merge) live in `plan-for-v1.md` §1 and are not
> repeated here.
>
> **Labels, not codes.** Every item has a descriptive `[Bracket-Label]` (CLAUDE.md,
> Naming conventions). GitHub issue numbers are kept in parentheses as the stable
> external cross-reference.
>
> **Where plans live.** `dev/` = live · `dev/deferred/` = parked past the beta,
> not closed · `dev/done/` = closed (shipped, `-REJECTED`, or `-SUPERSEDED`).
>
> **Last updated: 2026-07-27** — rebuilt from scratch. The previous file, with the
> full `1.0.0a122`–`a151` done-history, is archived at
> `dev/done/TODO-2026-07-27.md`. Struck: `[Display-Mode]` (rejected — see
> `dev/done/plans-considered-and-rejected.md`). Added the seven beta-gate items
> that `plan-for-v1.md` §1 named but this file never carried, plus
> `[PnL-Repr-HTML]` from the superseded first-class-PnL plan.

---

## Beta gate — blocks `1.0.0b1`

### Interface & reporting

- **[Reporting-Guidelines]** — *define what "first-class citizen" means* for a
  reporting object: a report's **rows are fixed** (it does not morph as the
  object gains properties), columns are **pure** (one unit per column — currency
  and ratios never mixed), headings are **presentation-ready**, and the `summary`
  (what *is* this?) vs `validation` (is it calculating right?) vs `reins_<flavor>`
  (only when reinsurance present) split is settled. Plan:
  `dev/reporting-guidelines.md`. Gates `[Aggregate-Summary-DF-Useless]` and
  `[Accounting-Summary-DF]` — scope the three together before redesigning any of
  them.
- **[FCC-Surface-Sweep]** — passes 1 and 2 shipped (`1.0.0a149`–`a151`;
  `dev/FEATURES.csv` now reports **zero** undocumented capabilities, and the
  executable half is `tests/test_fcc_surface.py`). **What is left is five author
  decisions**, all flagged in the matrix:
  1. the `BivariateAggregate.tail_df` name collision — a per-axis support frame
     under the name `a149` made a return-period property elsewhere
     (`bivariate.py:2483` vs `_portfolio.py:1347`); suggested `axis_support_df`;
  2. `BivariateAggregate` as a `LabeledMixin` host;
  3. `info` on `GridDistribution`;
  4. the `PnL.gd` / `PnL.result` alias (`_pnl.py:1172`, `:1177`) — one canonical
     name;
  5. whether `var_dict` follows the VaR-is-`q` rule and becomes `q_dict`
     (`_portfolio.py:3025`).
- **[PnL-Repr-HTML]** — `PnL` is the **only** first-class citizen with no
  `_repr_html_`, so it does not render in Jupyter (`Aggregate`, `Portfolio` and
  the three bivariate classes all define one). The residue of
  `dev/done/plan-pnl-first-class-SUPERSEDED.md`; land it with
  `[Reporting-Guidelines]` so the card it renders is the settled one.
- **[Display-Surface-Incidentals]** — four unrelated small defects found while
  surveying the duplicated presentation surface for `[Program-Mixin]`
  (`1.0.0a154`). None is urgent; none has a visible symptom today. Grouped so
  they are not re-discovered:
  1. `tweedie.py:850` defines **`__repr_html__`** — the wrong dunder (IPython
     looks for `_repr_html_`), so the method is unreachable. No symptom only
     because `_repr_mimebundle_` (`:866`) serves the same HTML — i.e. it is
     silently dead code, not a broken display.
  2. `Copula` (`copula.py:64`) is the only class carrying `LabeledMixin` but
     **not `HelpMixin`** — the one asymmetric mixin host.
  3. `Underwriter.__repr__` (`underwriter.py:906-929`) **hand-rolls the `info`
     layout** — 13 label/value rows padded to a hardcoded 19 columns instead of
     `info_row` / `INFO_LABEL_WIDTH = 25`. Invisible to both the matrix and
     `test_fcc_surface.py` because `Underwriter` is not a FEATURES column and
     has no `info`.
  4. `regen_features.py:200` skips every `_`-prefixed name, so the matrix is
     **structurally blind to `__repr__` / `_repr_html_` / `_text_info_blob`** —
     exactly the most-duplicated part of the presentation surface. Widening
     that filter is the prerequisite for auditing display consistency at all.
  *(A fifth suspect was checked and cleared: `Distortion.program` **does**
  survive pickling — the class has no `__reduce__`, so default `__dict__`
  pickling carries it.)*
- **[Plotting-Punchups]** — plotting polish; lead item: a `pnl` aggregate plots
  the aggregate **only** (no loss-convention severity overlaid on a payoff —
  a wrong-sign distraction for the UW/finance audience). Gate on
  `_agg_affine_active()` / `_signed()` in `plots/_aggregate.py`, both the
  discrete and continuous branches. Plan: `dev/plan-plotting-punchups.md`.
- **[Aggregate-Summary-DF-Useless]** (author verdict 2026-07-05, the CatBook
  example) — `Aggregate.summary_df` on a decorated engine judged "USELESS and
  needs improving". Scope with `[Reporting-Guidelines]` /
  `[Accounting-Summary-DF]`.

### Correctness & bugs

- **[Signed-Bounded-Window]** — robustness: kill the `int(inf)` `OverflowError`
  in the bucket/window sizer and resolve the silently half-applied layer on a
  signed severity. **Needs the author's D1 pick** (F3-A implement the clamp /
  F3-B reject the contradictory clause (the plan's lean) / F3-C honest metadata)
  before it can execute. Independent of everything else — do whenever; the
  regression bar is "ordinary aggregates byte-for-byte unchanged". The repro
  still crashes today. Plan: `dev/plan-signed-bounded-window-overflow.md`.
- **[Validation-Calc-Review]** (#49) — audit the validation algorithm against the
  published *Aggregate* paper and make the docs match the actual algo. The
  "all switches → config" sub-goal is done (`eps`/`noise`, `aliasing_ratio`,
  `exeqa_noise_floor`, `deficit_materiality`); remaining: fix the false-positive
  *agg-mean-error ≫ sev-error / aliasing* failure (try larger `bs`; revisit the
  too-tight tolerance, now an `aliasing_ratio` config edit).
- **[Input-Guards]** (ported from README) — three correctness/guard items: zero
  `lb` not consistent with attachment equals zero; flag **fixed** frequency with
  a non-integer expected value; flag **mixing** with an inconsistent frequency
  distribution.
- **[ZT-ZM-Frequency-Fix]** — zero-truncated / zero-modified frequency is broken
  (`poisson zt` → NaN solver for every parameterization; `zm` builds but the
  semantics are wrong — it inverts a post-modification mean). Redesign: the user
  inputs the **un-truncated/un-modified base mean** and we apply the shift
  **forward** (no solver); ship documented shift helpers (both directions). Two
  examples are commented out in `examples.agg` until then. Named in
  `plan-for-v1.md` §1; pairs with `[Doc-Gaps]`.
- **[Joint-Padding-Window-Tradeoff]** (logged 2026-07-07) — the in-core
  occurrence joint already *computes* a padded transform 4× the retained grid
  (`padding = 1` doubles each axis; `build_netceded_joint` inherits the engine's
  padding, `bivariate.py:868`). Flipping to `padding = 0` spends the same flops
  on **retained** cells: half the `bs`, or twice the window, per axis — the
  massive path's settled design. The catch: with `padding = 0`, clipped tail mass
  **wraps onto the body and is invisible to the deficit** (the a126 caution), so
  the flip requires (a) the massive window discipline on the in-core netceded
  sizing (the `balanced_window` measurement already exists — raise its nines
  target), and (b) the decisive cheap guard: compare the joint's marginal means
  against the engine's exact 1-D marginal means at build time (wrap hides from
  the deficit, never from the means). Tail-thickness caveat: thin/moderate tails
  win cleanly; a `pareto 1.2` cat book needs an astronomical window to the nines,
  where `padding = 1`'s honest clip stays competitive — so keep it a knob, not a
  silent default flip. Plumbing: `occ_bivariate` does not expose `padding=` (add
  a pass-through).
  **Phase 2, its own investigation — NOT a rider:** mixed-radix axis lengths
  `3 * 2**k` (scipy FFT handles radix 3 within ~10–20% of a power of two) give a
  1.5× window at unchanged binary `bs` — the "√2 step" — but `log2` is stored as
  an *exponent* throughout the library (`1 << log2` arithmetic, window sizing,
  the massive chunker, `bs_window_df` audits), so the blast radius is large;
  author flagged this explicitly (2026-07-07). Scope it standalone before
  touching anything.

### Tests & example libraries

- **[Agg-Library-Build-Check]** (from `plan-for-v1.md` §1 — *"the one change that
  makes step 1.2 real"*) — `tests/test_agg_libraries.py` today only checks each
  example **parses**. Extend it to check the example **builds**, that the result
  carries the surface it should (`valid`, `validation_explanation`, `summary_df`,
  `stats_df`, `plot`), and that its check passes. This is what turns "the
  interface is fixed" from an assertion into a test.
- **[Cookbook-Feature-Coverage]** (from `plan-for-v1.md` §1) — `cookbook.agg`
  covers every feature being frozen, and `decl-testers.agg` still fails exactly
  where it is meant to fail. The two files are for **testing**; `examples.agg` is
  for **showing** — keep the split.
- **[Showcase-Examples-Tune]** (from `plan-for-v1.md` §1) — the section-A showcase
  examples are still marked draft; tune them, extend `examples.agg` notes with
  tags / keywords / purpose, and make `aggregate_api/examples.py` read them.
  Feeds the playground dropdown and the 5-minute intro from one source.
- **[Rationalize-Tests]** (#51) — needed vs no-longer-needed; untangle and re-wire
  how the suite *consumes* the single test library (the `conftest`
  parametrization of every `test_suite.agg` line, the SLY snapshot regression)
  without losing coverage. Overlaps `[Scaffold-Retirement]` — do them in that
  order.

### Docs & packaging

> None of these has a code dependency — ready whenever the docs cycle opens.

- **[README-Stable-Body]** (#39, #13, #14) — rewrite the README body for the
  stable-v1.0 audience (what / who / install / one-liner DecL); the `README.md` +
  `CHANGELOG.md` split is done, only the body remains. One known copy fix
  (`plan-for-v1.md`): the opening line says the library builds
  **approximations**, which fights the *exact, not approximate* claim — the
  honest version is exact **compared to simulation**.
- **[v1-Journey-Philosophy]** (#15) — the v1.0 intro / "Journey" page + statements
  of philosophy (user manages logging / warnings / matplotlib; the distribution
  **is** `pᵢ` at `xᵢ`, no jump detection; `qd` is the doc-only fixed-font
  exception); cover the v1.0 shifts (linear allocation default, bounded
  detection, forwards-`S`, pentagon columns, `DefectiveDistributionWarning`).
- **[Grammar-Reference-From-Lark]** (#17) — regenerate
  `docs/4_agg_language_reference/` from `decl.lark` via `grammar(add_to_doc=True)`
  (it still describes the SLY-era grammar). The generator writes straight to the
  real `docs/` path — just run it and commit.
- **[Tail-Descriptor-Docs-Tests]** (#41, #42) — bounded / log-concave / super-exp
  / exp / sub-exp descriptors for freq **and** sev, plus the bounded/unbounded
  indicator; with tests.
- **[Doc-Gaps]** (#58–#62) — custom errors; syntax checker / better error
  reporting; stale "site" database refs; ZT/ZM zero-truncation/modification;
  splice examples.
- **[API-Docstring-Coverage]** — every public function/class carries a NumPy-style
  docstring that renders in the API reference. The doc side of
  `[Docstring-Sweep-NumPy]`.
- **[Docstring-Sweep-NumPy]** (#18) — Sphinx `:param:` → NumPy style in
  `iman_conover.py` / `moments.py` (and pockets elsewhere); public surface first.

### At the cut

- **[Scaffold-Retirement]** (do at the `1.0.0b1` cut) — the `.agg` libraries were
  split (a71) into a **shipped** set (`examples`, `actuarial-severity-curves`,
  `decl-testers`, `cookbook`) and a temporary SLY-parity scaffold
  (`_test_suite.agg`, `_test_suite2.agg`). The scaffold has done its job (proving
  the Lark parser matches the retired SLY parser). Delete both files and retire
  their dependents: the SLY snapshot (`tests/data/expected_specs.json` +
  `capture_sly_snapshot.py`), `test_decl_parser.py`, `test_splice_suite.py`, the
  `conftest` `test_suite_lines` / `underwriter` fixtures, `config.py`
  `TEST_SUITE_FILENAME` + `Underwriter.test_suite_file` + `interpret_file`'s
  default, the `freeze_knowledge.py` / `bucket_baseline.py` `DEFAULT_DATABASES`,
  the default toml `_test_suite` line, and the docs "Test Suite Programs"
  `literalinclude`. Then fix the Testing section of `CLAUDE.md`, which still
  calls the snapshot the main test. Confirm `[Agg-Library-Build-Check]` covers
  the shipped libraries; decide whether `decl-testers.agg` needs its own
  permanent parse harness.

---

## After the cut — `beta`

### Reporting

- **[Accounting-Summary-DF]** (pended 2026-07-04) — the gross/ceded-**split**
  consolidated-P&L card, `accounting_summary_df`: Consideration gross premium /
  ceded premium / total; Obligation gross loss / ceded loss / total; Margin
  total. Strictly correct (consideration and obligation cannot be netted) and
  supports GAAP / STAT / IFRS reporting shapes. The plain `summary_df` stays the
  simple net card ("summary" means summary); this is the detail view between it
  and the full `xpnl` walk. Unblocked since `1.0.0a136`.
- **[Walk-Step-Default-Labels]** (from `dev/done/plan-pnl-faces-punchlist.md`,
  2026-07-05) — undeclared cover steps in the walks default to `'ceded occ'` /
  `'ceded agg'` (`_pnl_builders.py:573`, `:904`, `:1147`); the author asked for
  better. Proposal: the DecL layer descriptor when the side has exactly one layer
  (`'occ 4750 xs 250'`, `'agg 95% po 100 xs 100'`), the generic name otherwise.
  **Needs the author's format pick** — it renames Step index keys and the derived
  plan rows (`'<label> result'`, `'net through <label>'`) in every undeclared
  program, so it should land deliberately, with the test churn in one sweep.
- **[Doc-Fix]** — clear the executed-cell `*Error`s in the docs build. **Last —
  after the code has settled** (it churns with every API change). Plan:
  `dev/doc-fix.md`.

### Numerics & pricing core

- **[Massive-Kappa-Second-Sweep]** (from `[PnL-Punchups-01]`, `1.0.0a134`) — bring
  the kappa scenario percentiles to the massive one-sweep P&L route.
  Conditioning needs the joint per atom *and* the grand-result quantiles before
  indicator-weighted means can accumulate — a second band sweep. Until then the
  massive `stats_df` keeps **marginal** ladders — since `a136` visibly so: plain
  `P01…P99` headers vs the in-memory scenario `κ` columns
  ([Decision-Kappa-Shared-Source-Rule]). Note the 2-D follow-up's
  [Gross-Anchored-Kappa-Insight] (G-slices are axis-aligned row averages,
  one-pass even on the massive route) may largely dissolve this for the
  variable-feature exhibits.
- **[Walk-Validation-DF]** (logged 2026-07-07) — joint-sourced walks (GC occ
  `xpnl`, the composed feature walks) have no attached exact-vs-realized audit;
  their only runtime guards are the joint's deficit bookkeeping and
  `CoarseJointGridWarning`. Add a `validation_df` for joint-sourced walks: each
  row's realized EX against the engine's exact `reins_density_df` marginal mean —
  the per-program version of what `tests/test_pnl_consolidated_walk.py` asserts,
  and the same computation as `[Joint-Padding-Window-Tradeoff]`'s means guard
  (build once, serve both).
- **[Consolidated-LAE-Off-Source]** (logged 2026-07-05) — the guaranteed-cost
  consolidated `pnl` books loss-basis LAE deterministically (`rate * E[gross
  loss]`): its source is the net marginal and the gross loss is not measurable
  there. The reinstatement consolidated face does NOT have this problem (axis 0
  of the joint carries the gross loss — LAE stochastic). Candidate fixes: a
  (gross, net) bivariate source for the GC consolidated face, or a stitched extra
  row off the gross marginal.

### Hygiene & tests

> The god-module refactor (the `GridDistribution` value type, the `plots/`
> subsystem + matplotlib defer, the `distributions.py` / `portfolio.py` splits,
> and the shared-concern modules `_validation` / `_bucket_window` / `_pricing` /
> `_reinsurance`) is **complete** — shipped `1.0.0a90`–`a95`. Conditional
> leftovers: the plots visual refresh, `ReinsuranceProgram` composition, and the
> sample-subsystem (correlation / switcheroo) review.

- **[bs_describe-Wart]** — investigate the `bs_describe` / `bs_explain` module
  workers (the `color=` workers behind `bs_description` / `bs_explanation`):
  purpose, the local-`line` accumulator, and whether they earn their place or
  want reshaping. **Reconcile with `dev/done/plan-consistent-naming.md` §3** (the
  deferred rename of the verb workers that shadow the noun properties by one
  letter, e.g. → `_format_bs_grid`). Do the two together. *Standing reminder —
  surface periodically until scoped.*
- **[Pedagogy-Migrations]** (#19) — move figure generators out of `ft.py` /
  `tweedie.py` into `pedagogy.py` so those stay API-focused. Feeds
  `[Pedagogy-Docs-Punchup]`.
- **[Switcheroo-Sample-Regression]** (#12) — a `Port.Sample` regression case
  guarding the kappa-replacement path.

### Docs

- **[Reinsurance-Case-Study-Docs]** (#16) — rebuild `bahnemann` / `enterprise
  risk` / `other_misc` per-layer exhibits from `reins_stats_df`, verify against
  published (numerics now stable).
- **[Pedagogy-Docs-Punchup]** (#40) — punch up `pedagogy` and integrate with docs;
  possible minor renames. Needs `[Pedagogy-Migrations]`.
- **[Reinsurance-Structure-Diagrams]** (#45) — under-specified; confirm source /
  scope (PMIR code?).
- **[Cheat-Sheet-Tweaks]** — at the alpha→beta cut, re-run `introspect` per class,
  reconcile any renames/removals, and apply pending wording/layout tweaks (incl.
  whether to densify DecL pages 2–3). Held until first beta.
- **[Plot-Severity-Outside-Window]** (#8) — plot severity when its grid doesn't
  overlap the aggregate window (inset, broken axis, or separate figure; `info`
  already warns). Approach undecided.

---

## Post-v1.0 ideas

- **[Multi-Resolution-Portfolio-Combine]** (#20) — compute each unit on its own
  `bs`, decimate onto the shared grid before the Fourier product (the real fix
  for the coarse shared-`bs` deficit). Deficit accepted / surfaced for now.
- **[Premium-Loss-Algebra-DecL]** (#21, v2.0) — constant aggregates and full
  aggregate arithmetic (`agg.A - agg.B`, `agg.A + c`); `pnl` covers the common
  case for v1.0.
- **[Named-Cherny-Madan-Families]** (#23) — add the MINMAXVAR / MAXVAR / MAXMINVAR
  distortion kinds so `PnL.evaluate` can surface the *named* indices (`dual`
  already *is* MINVAR). `@Cherny2009a`.
- **[Rate-Based-Reins-Clauses]** — extend reinsurance clauses to accept e.g.
  `net of 50% of 500 xs 500 at .3 rol or 3000 ceded or .25 ros` (rate on subject
  = quota share).
- **[Reinstatement-Event-Date-Terms]** — reinstatement terms depending on event
  date (pro-rata as to time); from `dev/done/pre-plan-reinstatements.md`.
- **[Portfolio-of-PnL]** (loss-sensitive half) — the constant-consideration case
  shipped at `1.0.0a125`; what remains is net-then-combine for loss-sensitive
  considerations, and `pnl` units *inside* a `port` (still a
  `NotImplementedError` gate). Parked design:
  `dev/deferred/plan-pnl-portfolio-DEFERRED.md`.
- **[Paper-Reproductions]** — reproduce published worked examples as cookbook
  recipes with beat 4 = *"matches the published value"*: the strongest available
  evidence the engine is right. Ranked candidate assessment (18 papers, three
  tiers, parameters transcribed) in `dev/plan-examples.md`; first targets
  Venter1983 and Mack2003, then the exact benchmarks Bruno2006 / Jin2016. Six
  papers' severity curves are already harvested into
  `actuarial-severity-curves.agg` §F, and all citation keys are confirmed present
  in `uber-library.bib`.

> *Rejected, so it is not re-proposed cold:* DecL colorization (aesthetic-only,
> structurally weak) and the `dev`/`user` **display mode** `ReprMixin` (not worth
> the effort — both views are already one attribute away). Reasoning:
> `dev/done/plans-considered-and-rejected.md`.
