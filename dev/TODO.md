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
> **Last updated: 2026-07-29** — the author's review annotations worked through
> (`1.0.0a170` `[FCC-Contract]`, `a171` `[FCC-Surface-Decisions]`). Closed and
> removed: `[ZT-ZM-Frequency-Fix]` (shipped `a152`),
> `[Aggregate-Summary-DF-Useless]` (the gross/net smell is fixed),
> `[FCC-Surface-Sweep]` and `[PnL-Repr-HTML]` (both shipped `a171`), and
> `[FCC-Contract-Gaps]` (shipped `a172`, both excuse lists now empty), and
> `[Display-Surface-Incidentals]` (shipped `a174`). Deferred to post-v1.0, last:
> `[Joint-Padding-Window-Tradeoff]`. Closed with no change:
> `Underwriter.__repr__`. Added: `[Bivariate-DecL-Label]` and
> `[Display-Surface-Punchups]` (what the widened matrix filter exposed).
> `[Plotting-Punchups]` is its own task, not part of the reporting cluster.
>
> **Previous rebuild, 2026-07-27.** The file before that, with the full
> `1.0.0a122`–`a151` done-history, is archived at `dev/done/TODO-2026-07-27.md`.
> Struck then: `[Display-Mode]` (rejected — see
> `dev/done/plans-considered-and-rejected.md`).

---

## Beta gate — blocks `1.0.0b1`

### Interface & reporting

- **[Reporting-Guidelines]** — *define what "first-class citizen" means* for a
  reporting object: a report's **rows are fixed** (it does not morph as the
  object gains properties), columns are **pure** (one unit per column — currency
  and ratios never mixed), headings are **presentation-ready**, and the `summary`
  (what *is* this?) vs `validation` (is it calculating right?) vs `reins_<flavor>`
  (only when reinsurance present) split is settled. Plan:
  `dev/reporting-guidelines.md`. Gates `[Accounting-Summary-DF]` — scope the two
  together before redesigning either.
  **The "what is a first-class citizen" half is DONE** (`1.0.0a170`
  `[FCC-Contract]`): the membership rule and the required surface are declared in
  `constants.FIRST_CLASS_CLASSES` / `FCC_REQUIRED`, audited by
  `dev/regen_features.py`, asserted by `tests/test_fcc_surface.py`, and written up
  as §0 of `dev/reporting-guidelines.md`. What remains here is the other half:
  what those reports **contain**.
- **[Bivariate-DecL-Label]** — `BivariateAggregate` became a `LabeledMixin` host
  at `1.0.0a171`, but its **object-level** label has no DecL spelling: the nine
  `bv_out` productions in `decl.lark` (copula / discrete / view-pair, each in
  three frequency flavors) carry no `as_label`, so `bivariate Cat as "..."` does
  not parse and the label is set with `label=`. The *component* labels already
  work, because each unit is an ordinary `agg`. Closing this is a grammar change
  plus the `decl_writer` round-trip, the grammar-reference regen and the
  `test_decl_unparser` / `test_grammar_sync` corpora, so it is its own item
  rather than a rider.
- **[Display-Surface-Punchups]** — what the `1.0.0a174` widening exposed, and one
  loose end. The original four-item `[Display-Surface-Incidentals]` list is
  **done** (`a174`: the tweedie dunder fixed, `Copula` given `HelpMixin`, the
  matrix's `_`-prefix filter widened to a `DISPLAY_MEMBERS` allowlist and the
  four display rows curated, `Frequency` given the `__repr__` it never had;
  `Underwriter.__repr__` closed with no change, since it is not a first-class
  citizen). What the new `display` group in `dev/FEATURES.csv` now makes visible:
  1. **`_repr_html_` is on the five first-class classes only.** `Severity`,
     `Frequency`, `GridDistribution` and the three `Bounds` classes render as
     plain text in a notebook. Deliberate or not, it is now a decision.
  2. **`_html_info_blob` exists on `Aggregate` alone**, while `Portfolio`,
     `BivariateAggregate`, `PnL` and `Distortion` each build the same intro
     paragraph inline in `_repr_html_`. The same shape of duplication
     `HelpMixin` / `ProgramMixin` collapsed, and the obvious fourth mixin.
  3. **Found at `a171`:** the P&L builders pass `label=name`, so a `PnL`'s
     private `_label` is never `None` and `LabeledMixin._title_name` renders the
     name twice (`'PL (PL)'`). No symptom today only because nothing on `PnL`
     calls `_title_name`: `_repr_html_` deliberately uses `label`, as `__repr__`
     already did. Either stop defaulting `label` to `name` in
     `_pnl_builders.py`, or accept that `PnL` does not use the `label (handle)`
     form and say so on the class.
  *(Checked and cleared during the original survey: `Distortion.program` **does**
  survive pickling — the class has no `__reduce__`, so default `__dict__`
  pickling carries it.)*
- **[Plotting-Punchups]** — plotting polish; lead item: a `pnl` aggregate plots
  the aggregate **only** (no loss-convention severity overlaid on a payoff —
  a wrong-sign distraction for the UW/finance audience). Gate on
  `_agg_affine_active()` / `_signed()` in `plots/_aggregate.py`, both the
  discrete and continuous branches. **Its own task** (author, 2026-07-29): it is
  independent of the reporting cluster above, so do not scope it with them.
  Plan: `dev/plan-plotting-punchups.md`.

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
- **[Showcase-Examples-Tune]** (from `plan-for-v1.md` §1) — **mostly done
  a157–a159.** `tags{...}` exists and every one of `library.agg`'s 186 entries
  carries tags, so the letter prefixes are retired and `discover(tags=...)`
  works. **Still open:** tune the `hero` entries themselves (they were marked
  draft), and rewrite `aggregate_api/examples.py` against the new library —
  it re-lexed the old `FORMAT` header and the entry names have all changed.
- **[Recipe-Library]** (`dev/done/plan-meta-data.md`; `dev/done/plan-recipes.md`) —
  notes-driven describe / test / audit. Phase 1 (the `tags{}` / `doc{{{}}}`
  trailer clauses, distortion trailer, ambiguity guards) landed in **a157**;
  phase 2 (the `aggregate.recipe` runtime, `Underwriter.recipe()` / `.recipes`)
  in **a158**; phase 3 (the merged 186-entry `library.agg` with unique names and
  tags, `discover(tags=)`) in **a159**; phase 4 (the
  `tests/test_library_recipes.py` harness + the first four recipes) in
  **a160**; tag namespacing in **a161**; `<<decl>>` substitution in **a163**;
  `[Recipe-Is-The-Entry]` (one `Recipe` class, `knowledge` → `recipes`
  throughout) in **a164**; `[Cookbook-Generate]` (the recipe-page generator,
  native Quarto cells, `_setup.recipe()` retired) in **a165**.
  **Remaining: phase 5** — convert the hand-written five-beat stub pages to
  generated recipes, page by page with author reaction. The mechanism is done;
  what is left is writing the `doc{{{}}}` for the entries each page teaches and
  pruning that page's `{{< include >}}`. Start from
  `build.recipes.query('doc')` (4 entries, 3 pages) against `[Cookbook-Pages]`
  in `docs/cookbook/plan.md`.
  **There is no phase 6 backlog.** A `note{}` is the norm and is all `discover`
  and the object dropdown need; a `doc{{{}}}` is for the cookbook-worthy few,
  so `build.recipes.query('not doc')` is a **directory**, not a worklist. What
  *is* worth hunting is `build.recipes.query('doc and n_asserts == 0')` — a
  recipe that describes without testing. When writing one, use `<<decl>>`
  rather than retyping the program, and plain ` ```python ` fences — both
  documented in the `library.agg` header — then re-run the generator.
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

- **[Recipe-Doc-Signing]** — optionally authenticate a `doc{{{...}}}` body with a
  trailing `<!-- hash: ... -->` computed over the body and salted from a private
  environment variable, so `Recipe.run()` can refuse an unsigned or tampered
  recipe. Deliberately *not* done for v1.0: running a recipe from a `.agg` file
  executes the code in it, and the shipped answer is a documented warning —
  treat a third-party `.agg` like a third-party Python file. See
  `dev/done/plan-meta-data.md` *Resolved questions* 2.
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
- **[Joint-Padding-Window-Tradeoff]** (logged 2026-07-07; **deferred past v1.0**
  by the author 2026-07-29, and deliberately last) — the in-core occurrence joint
  already *computes* a padded transform 4× the retained grid (`padding = 1`
  doubles each axis; `build_netceded_joint` inherits the engine's padding,
  `bivariate.py:868`). Flipping to `padding = 0` spends the same flops on
  **retained** cells: half the `bs`, or twice the window, per axis — the massive
  path's settled design. The catch: with `padding = 0`, clipped tail mass **wraps
  onto the body and is invisible to the deficit** (the a126 caution), so the flip
  requires (a) the massive window discipline on the in-core netceded sizing (the
  `balanced_window` measurement already exists — raise its nines target), and (b)
  the decisive cheap guard: compare the joint's marginal means against the
  engine's exact 1-D marginal means at build time (wrap hides from the deficit,
  never from the means). Tail-thickness caveat: thin/moderate tails win cleanly; a
  `pareto 1.2` cat book needs an astronomical window to the nines, where
  `padding = 1`'s honest clip stays competitive — so keep it a knob, not a silent
  default flip. Plumbing: `occ_bivariate` does not expose `padding=` (add a
  pass-through).
  **Phase 2, its own investigation — NOT a rider:** mixed-radix axis lengths
  `3 * 2**k` (scipy FFT handles radix 3 within ~10–20% of a power of two) give a
  1.5× window at unchanged binary `bs` — the "√2 step" — but `log2` is stored as
  an *exponent* throughout the library (`1 << log2` arithmetic, window sizing,
  the massive chunker, `bs_window_df` audits), so the blast radius is large;
  author flagged this explicitly (2026-07-07). Scope it standalone before
  touching anything.

> *Rejected, so it is not re-proposed cold:* DecL colorization (aesthetic-only,
> structurally weak) and the `dev`/`user` **display mode** `ReprMixin` (not worth
> the effort — both views are already one attribute away). Reasoning:
> `dev/done/plans-considered-and-rejected.md`.
