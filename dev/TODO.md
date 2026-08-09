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
> **Three sections, and only the first one gates.** *Beta gate* blocks
> `1.0.0b1`. *Provisional modules* (`aggregate.charts`, `aggregate.exhibits`)
> run in parallel and **never** block it: they are additive, PEP 411
> provisional, and ship in whatever state they are in. *After the cut* is
> everything deliberately deferred. When triaging, the question is which of the
> three an item belongs in, and the default for anything touching only
> `charts/` or `exhibits/` is the second.
>
> **Labels, not codes.** Every item has a descriptive `[Bracket-Label]` (CLAUDE.md,
> Naming conventions). GitHub issue numbers are kept in parentheses as the stable
> external cross-reference.
>
> **Where plans live.** `dev/` = live · `dev/deferred/` = parked past the beta,
> not closed · `dev/done/` = closed (shipped, `-REJECTED`, or `-SUPERSEDED`).
>
> **Last updated: 2026-08-09.** `[Exhibits-Module]` and `[Chart-IR]` moved out
> of the beta gate into their own **Provisional modules** section. They were
> filed under the gate when they were added on 2026-08-05, which was a
> transcription error against the original design: both are additive side
> projects, ship marked provisional in the PEP 411 sense, and do not gate
> `1.0.0b1`. The status is now recorded in the module docstrings, in
> `docs/3_reference/3_x_API_Stability.rst` (plus a warning admonition on each
> module page and `versionadded:: 1.0` on the public objects), and in the
> `CHANGELOG.md` preamble.
>
> **Previous update, 2026-08-05.** Added `[Exhibits-Module]` and `[Chart-IR]`:
> the aggregate to aLL interface work from the author's design notes (business
> exhibits over greater_tables IR with raw/insured/insurer/reinsurer
> perspectives, 1.0 implementing raw and insurer, and a chart intermediate
> representation with a 3-D surface
> pilot, the twelve plot inventoried per panel). Both approved, plans in
> `dev/plan-exhibits.md` and `dev/plan-chart-ir.md`.
>
> **Previous update, 2026-07-29** — the author's review annotations worked through
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

- **[Loss-Lab-Round-3]** — **DONE** (`1.0.0a223`–`a227`), plan in
  `dev/done/plan-loss-lab-round-3.md`; the api half is
  `aggregate_api/dev/plan-ui-round-3.md`. All five phases landed: `reins_view=`
  and `reins_views` on the pricing surface, `reins_price_df`, the Portfolio
  reinsurance chart, exhibit captions and the window prose, and the `writer` /
  `width` / `kinds` housekeeping. **Two findings left open** for their own
  decisions, both recorded in the plan's header: (1) `GridDistribution.tvar`
  has no orientation flip, so on a **payoff** a return-period row's `VaR` reads
  the downside while its `TVaR` averages the other side, which affects
  `Aggregate.tail_df` / `Portfolio.tail_df` on any payoff object and predates
  the plan; (2) a `pnl` recipe stores its *engine aggregate's* spec under the
  P&L's name, so `spec_to_decl` cannot render it and `to_agg` falls back to the
  stored program with a warning.
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
  Smaller blast radius since a230 (`[Reflected-Loss-Severity]`): a bounded
  reflection like `10 - lognorm 1.5 splice [0 10]` used to need `ssev`, and with
  it the identity layering; it can now be declared with plain `sev`, where the
  layer is a real layer. The half-applied layer remains for genuinely signed
  severities, and is still silent.
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
- **[Unparse-Dense-Spec-Guard]** — `decl_writer.spec_to_decl` silently emits
  *wrong* DecL when handed the dense `Aggregate.spec` instead of the sparse
  parser spec it documents. Found while checking whether an object built with
  the constructor could be decompiled (a218). The dense dict spells "unset" as
  `0`/`None` while the parser simply omits the key, and `0` is legitimate for
  `exp_premium` and `sev_scale`, so `13.7376 claims` renders as
  `0 premium at 0 lr`, the severity picks up a `0 *` scale, and a spurious
  `poisson 0 0 loss` appears. The result re-parses and builds to `est_m = 0`,
  `est_cv = nan` with no error anywhere. First it raises `AttributeError` on
  `label_map` (present-but-`None` defeats the `.get('label_map', {})` default),
  which is the only reason this is not already biting. Fix is a guard, not a
  decompiler: detect the dense shape and raise pointing at the parser spec.
  A real object-to-DecL decompiler needs a per-key inverse of the constructor's
  defaulting and is a separate, larger question — do not conflate them.
### Tests & example libraries

- **[Unparser-Reference-Gaps]** (surfaced by `[Library-Canonical-Layout]`, a178)
  — `decl_writer` cannot render three constructs back to what was written, so 13
  `library.agg` entries are exempt from the canonical layout and hand-written.
  The list is `UNPARSER_EXEMPT` in `tests/test_agg_libraries.py`; shrinking it
  is progress. The `tweedie` clause was the fourth and is **done** (`a231`,
  `dev/done/plan-tweedie.md`): it carries a `_tweedie` provenance key, renders
  its clause back, and no longer overwrites the author's `note{}`. The pattern
  it established, record what was declared and render from that, is what item 1
  wants.
  1. **Named object reference** (9 entries) — `sev.UnitSeverity` is resolved and
     inlined at parse time, and nothing on the spec records that a reference was
     written, so it renders as `dsev [1]`. Fixing it means carrying the
     reference on the spec (a `sev_ref` key, say) and rendering from that. This
     is the one worth doing: named-severity reuse is a documented feature that
     would otherwise appear nowhere in the shipped library.
  2. **Distortion combinator** (1) — `minimum` / `mixture` drop their child
     distortion names, so `format_program` raises rather than rendering.
  3. **`ssev <c> - <dist>`** (3) — renders in the general affine form
     `-1 * <dist> + <c>`, whose leading `-1 *` parses two ways
     (`tests/test_grammar_ambiguity.py`, `KNOWN_AMBIGUOUS`). Either teach the
     writer the compact spelling when the scale is exactly `-1`, or resolve the
     grammar ambiguity. `dev/done/reflow_library.py` refuses any rendering more
     ambiguous than its source, so this cannot regress silently.
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
  The warnings half of that philosophy is now settled and just needs writing up:
  `[Warning-Policy]` (`1.0.0a219`, `dev/done/plan-warning-policy.md`) says a
  condition is announced once per session at the level where it can change an
  answer, the per-object verdict lives in `valid` / `validation_explanation`,
  and `silence_warnings` / `reset_warn_once` are the two user controls.
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
  `[Docstring-Sweep-NumPy]`. **Page coverage is done** as of a196
  (`[Reference-Chapter-Audit]`): every public module now has a home in
  `docs/3_Reference.rst`, every autodoc target and cross-reference resolves, and
  the `autosummary` lists match each module's `__all__`. What remains here is
  docstring *quality*, not missing pages. Re-run the three checks after any
  public surface change.
- **[Docstring-Sweep-NumPy]** (#18) — Sphinx `:param:` → NumPy style in
  `iman_conover.py` / `moments.py` (and pockets elsewhere); public surface first.

### At the cut

- **[Scaffold-Retirement]** (do at the `1.0.0b1` cut) — the `.agg` libraries were
  split (a71) into a **shipped** set (`examples`, `actuarial-severity-curves`,
  `decl-testers`, `cookbook`) and a temporary SLY-parity scaffold
  (`_test_suite.agg`, `_test_suite2.agg`). The scaffold has done its job (proving
  the Lark parser matches the retired SLY parser). Delete both files and retire
  their dependents: the spec snapshot (`tests/data/expected_specs.json` +
  `capture_spec_snapshot.py`), `test_decl_parser.py`, `test_splice_suite.py`, the
  `conftest` `test_suite_lines` / `underwriter` fixtures, `config.py`
  `TEST_SUITE_FILENAME` + `Underwriter.test_suite_file` + `interpret_file`'s
  default, the `freeze_knowledge.py` / `bucket_baseline.py` `DEFAULT_DATABASES`,
  the default toml `_test_suite` line, and the docs "Test Suite Programs"
  `literalinclude`. Then fix the Testing section of `CLAUDE.md`, which still
  calls the snapshot the main test. Confirm `[Agg-Library-Build-Check]` covers
  the shipped libraries; decide whether `decl-testers.agg` needs its own
  permanent parse harness.

---

## Provisional modules: parallel to the release, they do **NOT** gate `1.0.0b1`

> **These two items are additive side projects, not part of the 1.0 contract.**
> `aggregate.charts` and `aggregate.exhibits` ship marked **provisional in the
> sense of PEP 411**: public, encouraged, and free to change in a minor release
> with no deprecation period. Their dependencies point **inward** (they import
> the core, the core does not import them), so no amount of churn here reaches
> an existing class, and **1.0 ships whether or not either is finished**.
>
> The one edge into pre-existing code is the per-chart conversion of a bespoke
> plot to emitter-plus-renderer, gated by before-and-after image diffs
> (`tests/data/chartdoc_baselines/`) and deferrable **chart by chart** past 1.0.
>
> Work them as attention allows. Nothing below is a reason to hold the tag.
> Explicitly post-1.0: conversion of the charts the app does not use, full
> matplotlib-renders-the-IR convergence, the `INSURED` / `REINSURER`
> perspectives, and any exhibit meta-language.
>
> Recorded in three places, per the design: the module docstrings, the Sphinx
> docs (`docs/3_reference/3_x_API_Stability.rst` plus a warning admonition on
> each module page and `versionadded:: 1.0` on the public objects), and the
> `CHANGELOG.md` preamble.

- **[Exhibits-Module]**: new `aggregate/exhibits.py` translating the raw FCC
  stats frames into greater_tables IR envelopes. `Perspective` enum
  (raw/insured/insurer/reinsurer), singledispatch generics (summary, tail, stats,
  validation, reins, pnl_ledger, pnl_ratios, dependency), a registry derived
  `available_exhibits(obj)` capability query, a new `exhibits` optional extra
  (lazy GT import), and one generic aLL endpoint pair (capability plus
  envelope with ETag). Migrates the app's ROW_FLAGS/FORMATS/caption knowledge
  into the library. Purely additive, provisional at 1.0. Six phases; author
  gate on the PnL business framing. 1.0 implements RAW and INSURER only,
  with INSURER = RAW unless a per (exhibit, type) override is registered;
  overrides are custom per exhibit (identity, thin renames, or an extensive
  reshape for the xpnl tower ledger), settled case by case with the author
  once the infrastructure lands. INSURED and REINSURER are
  enum vocabulary, implementations deferred with the reinsurer semantics
  review as that work's opening gate. Approved 2026-08-04; perspectives
  renamed and scoped 2026-08-05 (buyer/seller were relative and confusing,
  the insurer both sells and buys). Plan: `dev/plan-exhibits.md`.
  **Progress:** `[Exhibits-Scaffold]` and `[Exhibits-Stats-Validation]`
  landed together at `a200` (module, registry, summary and tail for
  Aggregate/Portfolio; stats and validation across the five FCCs with the
  raw moment drop and failing row emphasis; dependency for bvagg; snapshot
  tests, lazy GT boundary; the `exhibits` extra is documented but commented
  in `pyproject.toml` until greater_tables 6 publishes to PyPI, since an
  active unresolvable extra would break `uv sync --all-extras`).
  `[Exhibits-Reins-Insurer]` landed `a201` (reins exhibit, two blocks,
  cession gated, insurer moment drop plus captions and total flags).
  `[PnL-Economic-Frames]` landed `a204` (BREAKING: the PnL ledger is
  `economic_df`, `ratio_df` is `economic_ratios_df`, and `stats_df` delegates
  to the engine so the name means one thing across the contract).
  `[Exhibits-Package-Split]` landed `a205` (`exhibits/` package mirroring
  `plots/`, `register_simple_exhibit`, `bs_window` and `tail_behavior`,
  `economic` / `economic_ratios` renames). `[Exhibits-Economic-Insurer]`
  landed `a206` (ledger captions stating the kappa regime, ledger row flags,
  the ratio frame split into pure-unit blocks, `MEASURE_FORMATS`).
  `[Exhibits-Waterfall]` landed `a207` (`economic_waterfall`: the margin walk
  in two blocks, tower gated, capital as `M / -M_100`, the diversified column
  footing where the standalone one cannot). **Its author gate is still open**,
  along with the CV-on-the-ledger question; both are in the plan's open list.
  Next: the app consolidation (`[Exhibits-App-Consolidation]`), which is the
  SPA reading envelopes and the migrated app knowledge finally being deleted.
  `[Exhibits-PnL-Translation]` raw stage landed `a203` (pnl_ledger and
  pnl_ratios as RAW passthroughs); the INSURER framing draft awaits the
  author gate (captions, footing rules, Side sign presentation, the tower
  reshape; see the draft section appended to `dev/plan-exhibits.md`). Open with the author: per measure formats for the stats
  insurer view (greater_tables formats are per column, the store mixes
  measures down a column, so the app's measure formats have no TableSpec
  home yet) and whether the PnL validation audit ever gets a failure gate.
- **[Chart-IR]**: a minimal versioned chart IR in a new `aggregate/charts/`
  package (frozen dataclasses, no pydantic; ChartDoc/ChartSeries/ChartAxis/
  Panel/Mark), with the one generic mpl renderer at `plots/_chartdoc.py` and
  matplotlib-renders-the-IR as the committed post 1.0 convergence. Three
  independently landing passes: inventory (`dev/chart-inventory.md`, review
  gate), schema (sign off gate), conversion (one chart per commit, image diff
  and fixture diff acceptance). Pilot: the bvagg 3-D surface (emitter block
  sums the joint; ECharts true 3-D; mpl renders the 2-D projection or raises
  under `strict=True`). 1.0 scope is the 8 app facing charts plus schema
  representability (not conversion) of the `plot_twelve` panels, inventoried
  per panel: the kappa and two unit independence panels are expected future
  aLL charts, deliberately not developed before the IR, then born as
  emitters. Emitters read the numerics-2 accessors (`unit_density_df`,
  `allocation_diagnostics`), never legacy `density_df` `p_<unit>` columns.
  Approved 2026-08-04, panel scope added 2026-08-05.
  Plan: `dev/plan-chart-ir.md`. **Progress:** inventory landed a197
  (`dev/chart-inventory.md`, six judgment calls awaiting author picks);
  schema v1 landed a198 (`charts/ir.py` plus `tests/test_charts_ir.py`,
  field list awaiting sign off; the charts boundary assertion lives in
  `test_charts_ir.py` for now because `test_plots_boundary.py` was
  mid-edit in the parallel exhibits workstream); pilot library side landed
  a199 (`chart_joint_surface` emitter with the migrated surfaceGrid
  reduction, `plots/_chartdoc.py` renderer with the capability pattern);
  pilot app side landed as aggregate_api a38 (`/chart/{name}` route with
  doc-hash ETag, generic `chartdoc-to-echarts.js` adapter plus
  `surfaceOverrides` chrome dict, `surfaceGrid`/`surfaceOption` deleted,
  fixtures and node smoke updated); first conversion, distortion g(s),
  landed a202 (emitter, xy renderer growth, and the image gate:
  compositor-generated baseline, measured conversion residual RMS 0.05,
  pins mpl 3.10.9 / tolerance 2.0). **Sign-off gate closed 2026-08-05**:
  all six judgment calls agreed, the gate pins confirmed, and the schema
  changes they imply executed from `dev/plan-chart-schema-signoff.md`.
  `[Chart-Grid-Overlays]` landed a208 (grid panels carry one surface plus
  any number of x/y overlays, the `iso_total` role, the twelve-plot
  bivariate panel representability test, and `LOG_FLOOR = 1e-15` in
  `constants.py` settling J5). `[Chart-Plain-Text-Names]` with
  `[Chart-Axis-Labels]` landed a209 (the plain-text naming rule and the
  `ChartDoc.tex` lookup, the dual settled on `ǧ(s)` in `constants.py`, the
  renderer drawing the axis labels the document carries, `plot_distortion`
  gaining the labels it never had, and the baseline regenerated: the
  conversion residual is now 0, pixel for pixel). Chart IR version 1 is
  **signed off and closed to additions**: a chart the schema cannot express
  changes it by a fresh author decision or stays bespoke. Second conversion,
  the reins triple, landed a210 (`chart_reins` with the basis semantic
  option and the cession predicate, survival through `GridDistribution.sf`
  exactly matching the app's accumulation; the renderer grew multi-panel
  layout with shared x axes, and now honors `suggested_range`, which it had
  never applied; corrected at a211, where the suggested range became the
  extent of the *data*, inset by the renderer's own margin, since a
  distortion legitimately sits at 0 or 1 and must not be drawn along the
  frame). Third conversion, severity, landed a212 (`chart_severity`
  absorbing the app's log-spaced sf inversion, and reading probability
  mass where a law has no density instead of drawing a flat zero;
  `charts/_two_panel.py` now holds the window and survival-floor
  semantics the five two-panel charts share). `[Chart-Atomic-Support]`
  landed a214, the one deliberate reopening of the frozen schema (author
  decision, 2026-08-05): `ChartSeries.support` is `'atomic'` or
  `'continuous'` and defaults to atomic, because a discretized
  distribution **is** the distribution here; the renderer owns the ladder
  from that flag plus the room each atom gets (stems, then steps read as
  bars, then a plain line once a bucket is sub-pixel), counted in visible
  atoms so a cropped window is judged on what it shows, and cumulative
  functions step right-continuously off the *axis* rather than the series
  role. **Part two of the plan is the rest of pass four restated as one
  job per first-class class. Drafted and APPROVED 2026-08-09**, six open
  gates settled the same day. It changes two things the original did not
  anticipate: `Object.plot()` moves onto the IR **now** rather than post
  1.0, with the compositor deleted in the same commit, and the schema
  reopens a third time (after `support` at a214) to carry the view
  toggles, as `[Chart-Declared-Readings]`: `ChartAxis.scales` and
  `full_range`, the first use of `reciprocal_of`, and `Panel.kinds`
  answering `heatmap | surface` with one document declaring two
  realizations, `CHART_IR_VERSION` staying 1. Both follow from the app
  going purist about IR charts, which makes `chart_agg` the critical path
  for the whole app rather than one item in a queue. Nine jobs in order:
  `[Chart-Declared-Readings]`, `[Chart-Aggregate]`, `[Chart-Distortion]`,
  `[Chart-PnL]`, `[Chart-Severity]`, `[Chart-Bounds]`,
  `[Chart-Bivariate]`, then `[Chart-Portfolio]` and `[Chart-Reins]`
  **last**, both blocked on author design work (kappas for the portfolio,
  a fine-tuning pass for reinsurance) so they are not written twice. The
  severity drawing is **also** expected to change, so `[Chart-Severity]`
  is a redesign and not a rewire, and keeps its slot only while its
  design is ready.
  The settled gates: `charts.build_chart_doc` / `primary_chart` as module
  functions mirroring `build_exhibit` (no new method on any class);
  **panel arrangement is renderer-side**, the document says which panels
  in what order and order is only a hint, so no `ChartDoc.layout`;
  `chart_agg` is **the density and the Lee panel**, with the old log
  panel becoming log x plus log y declared on the density panel, which
  changes what `Aggregate.plot()` draws and needs a fresh baseline;
  `ChartDoc.tex` is **total**, both forms always written like alt text,
  which is stronger than today's docstring and gains a test.
  `[Chart-Bounds]` is two panels, the weighted cloud and all five
  calibrated distortions on one band, consolidating today's three.
  **REMINDER the author asked for: 3-D plot punchups are wanted and not
  yet written down. Prompt for them when `[Chart-Bivariate]` starts,
  before the emitter.** The app's written ask is
  `dev/note-all-chart-asks.md`; its half is
  `aggregate_api/dev/plan-plot-ir-api.md`. Note `chart_reins` **is**
  already registered for `Portfolio`, so that ask in the note is stale.
  **Part two progress:** job 1 `[Chart-Declared-Readings]` landed a233
  (`ChartAxis.scales` / `full_range`, `Panel.kinds`, the stated
  `reciprocal_of` contract, `meta['return_period_map']` carrying the
  `1/v` versus `1/(1-v)` branch, `plot_chartdoc`'s four switches with
  `log_z` deleted, `meta['z_log_ok']` retired, every hash changed,
  `CHART_IR_VERSION` still 1 with the bump rule now written down).
  Two author decisions taken before execution, 2026-08-09: the renderer
  selects a reading through four named switches rather than a `reading=`
  mapping or per-axis overrides; and the Lee panel keeps **non-exceedance
  p** on its probability axis, so `reciprocal_of` is widened by
  `meta['return_period_map']` rather than the axis being changed to carry
  the interrogated tail. Job 1b `[Chart-Tex-Totality]` landed a234
  (`complete_tex` / `human_strings` in `ir.py`, all four emitters
  returning through it, a sweep asserting the set difference is empty,
  the renderer's fallback restated as a net under a bug). It landed as
  its own bump rather than inside job 1, because totality touches every
  emitter and the declared readings do not. Job 2 `[Chart-Aggregate]`
  landed a235: `chart_agg`, `Aggregate.plot` rewired and
  `plot_aggregate` deleted, `build_chart_doc` / `primary_chart` /
  `register_chart(primary=)` / `ChartEntry`, and the renderer growing
  the sideways step, the paired-reading re-slice and the legend restyle
  (smaller, in the emptier upper corner) the author asked for on review.
  **The author reviewed the new figure before the baseline was cut**, as
  the plan requires; `quantile_x='return'` is now `return_period=True`
  and `axd` / `figsize` / `max_return_period` are gone from
  `Aggregate.plot`. Job 3 `[Chart-Distortion]` landed a236: the rewire
  and the deletion of `plot_distortion`, licensed by a re-measured RMS 0;
  `both=` becomes `dual=` (five pedagogy call sites and
  `plot_distortion_affine` follow), and `scale='return'` is **removed**
  rather than relocated, on the plan's own statement that a log reading
  of a distortion's unit square is not meaningful. Nothing called it; it
  returns as `scales=('linear', 'log')` the day it is wanted.
  Job 4 `[Chart-PnL]` landed a237: `chart_pnl` over the shared
  `outcome_doc` builder, `PnL.plot` rewired and `plot_pnl` deleted; the
  signed window is the two-sided crop, the outcome axis declares
  `('linear',)` only, `return_period_map` is `reciprocal` so the anchors
  land on 100 and 250 exactly, and break even at 0 is a mark in both
  panels. **Jobs 1 to 4 are done (a233, a234, a235, a236, a237).**
  **Next: job 5 `[Chart-Severity]`**, which is a redesign and not a
  rewire, so it opens by asking what the severity chart should be and
  moves to the back with `[Chart-Portfolio]` and `[Chart-Reins]` if that
  design is not ready. Job 6 `[Chart-Bounds]` is settled (two panels,
  the cloud and all five calibrated distortions on one band) and could
  be taken first if the severity design is not.

---

## After the cut — `beta`

### Reporting

- ~~**[Layer-Peeling-Shorthand]**~~ **DONE `1.0.0a183`**
  (`dev/done/plan-layer-peeling.md`; the placeholder was
  `dev/done/plan-yapnl.md`) — `xpnl ... peel top-down|bottom-up` books one group
  per reinsurance **layer** instead of one per tier. Aggregate layers peel
  per-atom (disjoint intervals on one subject, so single-layer ceders sum to the
  cumulative ceder identically) and keep the κ ladder; two or more occurrence
  layers route through the kernel's `stitched_rows` seam, where EX foots exactly
  by linearity but the dispersion columns are marginal.
- ~~**[Tier-Subtotal-Rows]**~~ **DONE `1.0.0a184`**
  (`dev/done/plan-tier-subtotals.md`) — a layer-peeled `xpnl` shows the whole
  occurrence and whole aggregate program again: a tier peeling into two or more
  steps gains an `'All occurrence'` / `'All aggregate'` block after its last
  step. `_ledger_plan` gained `tier_spans` plus the `'tier_total'` /
  `'tier_result'` kinds; per-atom it is a partial sum over a group span (κ ladder
  intact), stitched it costs one FFT for the occurrence tier and none for the
  aggregate. Also fixed three bare `else` row-kind fallthroughs that silently
  booked an unknown kind as the impact row, and `PnL.__add__` dropping new
  constructor arguments on composition.
- **[Peel-Aggregate-Tier-Only]** (logged 2026-07-30, from
  `[Layer-Peeling-Shorthand]`) — the stitched route is all-or-nothing for a
  `PnL`, so peeling the occurrence layers of a program that *also* carries
  aggregate layers pulls the aggregate steps onto the marginal ladder too, even
  though they would foot per-atom on their own. A way to peel **one tier** and
  leave the other lumped would keep the κ ladder for the common
  "several occurrence layers, one aggregate cover" shape. The deferred explicit
  peel-order form (a permutation of layer indices rather than a direction) is
  the natural surface for it; note that form is also the only one that
  manufactures layer gaps, so it needs `[Ceder-Gap-Knot]` (fixed in `a182`).
- ~~**[PnL-Ratio-Frame]**~~ **DONE `1.0.0a185`**
  (`dev/done/plan-pnl-ratio-frame.md`) — **breaking**: the `Scaled` column,
  `scaled_stats_df`, the `scale` property and the `scale=` kwarg are gone;
  ratios live in `PnL.economic_ratios_df` (named `ratio_df` until `a204`), with
  `PnL.legs_df` as the itemized companion.
  `Scaled` divided every cell by one number (net premium on a walk), so it read
  1.43 for gross premium and -0.571 for a 0.40 gross loss ratio. `Leg` gained a
  validated `kind` (`LEG_KINDS`), which is what makes an expense ratio possible;
  `stats_df` unchanged. Amounts are signed in the gross direction so they add
  across blocks, `M == P - L - E - C` holds identically, and a cession's LR
  reads positive. `LR` (ratio of means) and `E_LR` (mean of ratio) are reported
  separately because a correlated premium makes them different numbers.
- **[Ratio-Distribution]** (logged 2026-07-30, from `[PnL-Ratio-Frame]`) — a
  per-atom P&L holds the joint of loss and premium, so the loss **ratio** is
  available as a `GridDistribution`, not just its two scalar means. That is what
  a sliding commission actually needs: `contract_terms.SlideTerms.phi` maps a
  *realized* loss ratio to a commission, so the commission is `E[phi(LR)]`,
  which no moment of `LR` determines. Shape: a `ratio_distribution(ratio='LR',
  step=...)` accessor returning the GD, refusing on the stitched / massive
  routes where there are no shared atoms AND premium is random (exactly where
  `E_LR` is `nan` after the `a186` regate). Care needed at atoms with vanishing
  premium, where the ratio is undefined.
- **[Accounting-Summary-DF]** (pended 2026-07-04) — the gross/ceded-**split**
  consolidated-P&L card, `accounting_summary_df`: Consideration gross premium /
  ceded premium / total; Obligation gross loss / ceded loss / total; Margin
  total. Strictly correct (consideration and obligation cannot be netted) and
  supports GAAP / STAT / IFRS reporting shapes. The plain `summary_df` stays the
  simple net card ("summary" means summary); this is the detail view between it
  and the full `xpnl` walk. Unblocked since `1.0.0a136`.
- ~~**[Walk-Step-Default-Labels]**~~ **DONE `1.0.0a183`** (with
  `[Layer-Peeling-Shorthand]`, `dev/done/plan-layer-peeling.md`) — an undeclared
  cover step whose tier holds exactly **one** layer is now named by that layer's
  DecL descriptor (`'occ 4750 xs 250'`, `'agg 85% so 1500 xs 7000'`) via
  `_tier_label`; a multi-layer tier keeps the generic `'ceded occ'` / `'ceded
  agg'`, because it has no single descriptor and `peel` is how you see those
  layers separately. Peeling forced the pick: `_ledger_plan` raises on duplicate
  row labels, so one group per layer cannot share one generic name. Test churn
  landed in the same sweep (six modules).
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
  surface periodically until scoped.* **Now three pairs, not one:** `1.0.0a192`
  added `sharpen_describe` / `sharpen_explain` to `_bucket_window.py` on the
  same shape (deliberately, to match the neighbours rather than invent a fourth
  convention), so whatever the rename lands on applies to them too.
- **[Pedagogy-Migrations]** (#19) — move figure generators out of `ft.py` /
  `tweedie.py` into `pedagogy.py` so those stay API-focused. Feeds
  `[Pedagogy-Docs-Punchup]`.
- **[Switcheroo-Sample-Regression]** (#12) — a `Port.Sample` regression case
  guarding the kappa-replacement path.
- **[DecL-Colorizer-Resync]** — **done `1.0.0a175`**
  (`dev/done/plan-colorizer-resync.md`). `AggLexer` re-derived from `decl.lark`
  (quoted labels, `tags{}`, `doc{{{}}}`, `//`, `@`, `_` separators, all four
  builtin prefixes); `agg.sublime-syntax` resynced. The guard now tokenizes all
  four shipped corpora asserting zero `Token.Error` and derives the brace
  clauses and operator literals from the grammar, so the reserved-word walk is
  no longer the only check.
- **[Bivariate-Gate-Flake]** — `tests/test_bivariate.py::test_mv_explain_flags_clipped_book`
  failed under `pytest -m 'slow or not slow' -W error::RuntimeWarning` in two
  full-suite runs out of four (`1.0.0a220`), and would not reproduce: the test
  passes alone, the module passes alone, and all three bivariate modules pass
  together under the same flag. That suite is the one already carrying
  `xdist_group` for memory pressure (several 2-D grids allocated concurrently
  once produced a numpy allocation failure in a test whose own grid was 64x64),
  so worker load is the first suspect, and the runs that failed were back to
  back with other full-suite runs. Until it is understood, `-W error::RuntimeWarning`
  stays a release-time command rather than going into `addopts`. Reproduce by
  running the whole suite under the flag repeatedly; capture the traceback
  (`--tb=long -rf` to a file) the first time it fires.
- **[Colorizer-Style-Choice]** — `decl_writer._colorize` hard-codes
  `style='friendly'` for html, ansi and latex alike. A `style=` axis on
  `format_program`, and a dark-background default for the terminal path, is a
  small separate item.

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
  already *is* MINVAR). `@Cherny2009a`. Independent of
  `[Margin-Acceptability-Evaluate]` (`1.0.0a187`), which fixed *what* is solved;
  this adds families to solve it over.
- **[PnL-Density-DF-Running-Nets]** — `PnL.density_df` yields a GD for the legs,
  group results, tier results and the grand result, but **not** the
  `running_net` rows, which `PnL.evaluate` does report (`1.0.0a187`). Two views
  disagreeing about what a ledger row is. `evaluate` reads `_by_kind` directly
  so nothing depends on the gap; closing it means deciding whether `density_df`
  is "rows with a law" (add them) or "declared plus subtotal rows" (leave it and
  say so in the docstring).
- **[Range-Sugar-Round-Trip]** (logged 2026-08-09; **post-v1.0, do not start
  before the cut**) — range sugar is expanded at parse time and never rebuilt,
  so `dsev [1:6]` renders back as `dsev [1 2 3 4 5 6]` and `[0:10:2]` as
  `[0 2 4 6 8 10]`. **The wanted behavior is to NOT expand them**: a range
  should round-trip as the range that was written. `decl_writer._fmt_seq`
  currently says so in as many words, that re-folding "would be cosmetic only",
  which is the judgment this item overturns. It is cosmetic for a six-element
  vector and not at all cosmetic for `[1:1000]`.
  Four grammar productions are involved, the two-part and three-part forms of
  each of `numbers_range` / `numbers_range_step` (`decl.lark:625`) and
  `doutcomes_range` / `doutcomes_range_step` (`:491`).
  Same shape of fix as `[Tweedie-Round-Trip]` (`a231`): record what was
  declared and render from that, rather than reconstructing it from the
  expansion. Harder in one specific way, though. Tweedie's provenance is one
  key on one object; a range is a property of an *individual array-valued spec
  entry*, and vectors appear in `dfreq`, `dsev`, exposure and limit profiles,
  weights and layer lists. So the record has to hang off the array, not the
  spec, which is the design question to settle first. Detecting an arithmetic
  progression after the fact is the wrong answer: it would re-fold a list the
  author wrote out longhand, exactly the canonicalizing-with-opinions failure
  the tweedie work was careful to avoid.
- **[Power-Variance-Family]** (from `dev/done/plan-tweedie.md`, 2026-08-09) — the
  `tweedie` clause covers only the `1 < p < 2` slice of the power variance
  family, but the `Tweedie` class already spans the whole p range: Gaussian at
  0, Poisson at 1, gamma at 2, inverse Gaussian and Lévy at 3, extreme stable
  outside. Extend the DecL surface to the rest of the family. Depends on
  `[Tweedie-Round-Trip]` landing first, since that is what makes a declared
  member of the family survive the parse at all; the naming, the clause shape
  and which members are worth a keyword are all open.
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
