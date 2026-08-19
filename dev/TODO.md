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
- ~~**[Cede-Contra-Expense]**~~ **DONE `1.0.0a304`**
  (`dev/done/plan-cede-expenses.md`): ceding commission folds into `E` as
  contra expense (author ruling 2026-08-18), the same treatment recoveries
  already get in `L`, and the ratio frame's separate `C` column is gone.
  `CR` and `E_CR` are numerically invariant, `ER` becomes net of commission,
  `legs_df` keeps the itemization. API side was a no op, as scoped. The
  exhibit snapshot and `features.rst` were the two surfaces the plan did not
  name: the first regenerated, the second deliberately left for the next
  `dev/task-features.md` run.

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
- ~~**[Dfreq-One-Claim-Shortcut]**~~ **DONE `1.0.0a303`**
  (`dev/done/plan-dfreq-one-claim-shortcut.md`): `dfreq [1]` takes the same
  exact severity-copy path as `1 claim ... fixed`. New read-only
  `Aggregate.one_claim` property (support test, never the mean) feeds both
  gates: `freq_sev_convolution`, whose `en` / `freq_name` parameters became a
  caller-computed `one_claim` boolean, and the `bivariate.py` netceded joint.
  Baseline regenerated for the one case that moves, `Base.DfreqOne`, at 1e-11
  relative; the other nine are byte identical.
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
- **[Windowed-Cdf-Nan]**, **needs the author's ruling**: diagnosed a289 and
  deliberately left unfixed. On a windowed grid `cdf` returns `nan` below the
  window instead of 0: on the `[Windowed-Grid-Breaks-Calibration]` repro,
  `a.cdf(0)` and `a.cdf(10000)` are both `nan` where the distribution has no
  mass below `x_min` by construction. `sf = 1 - cdf` inherits it; `q` is
  unaffected. Cause: `Aggregate.cdf` (`_aggregate.py:6357`) and
  `Portfolio.cdf` (`_portfolio.py:2073`) both build
  `interp1d(..., kind='previous', bounds_error=False, fill_value='extrapolate')`,
  and scipy's `previous` kind has no previous knot below the first, so it fills
  `nan`. Confirmed in isolation on scipy 1.17.1. The one-line fix is
  `fill_value=(0.0, 1.0)`, verified identical on every in-window point. **The
  ruling needed is the upper fill**: `1.0` is exactly right mathematically but
  today the above-grid value is the cumsum top (`1 - 5e-12` on the repro), so
  `sf` above the grid goes from `5e-12` to exactly `0`, and anything dividing
  by `sf` in the far tail sees that. `fill_value=(0.0, cum[-1])` preserves
  current behavior above and fixes only below. Same root as
  `dev/done/plan-windowed-grid-calibration.md`, different site and a much wider
  blast radius, which is why it was not folded into that bump.
### Tests & example libraries

- **[Library-Round-Two]** (from the a297 tidy) — round one deleted 28 exact
  duplicate programs, fixed the flatly wrong labels and re-sectioned the file.
  What it deliberately left, because each needs an author call rather than a
  rule:
  1. **The Bodoff family is four deep on one experiment.** `sev dhistogram xps
     [...]` and `dsev [...]` build byte-identical objects, so `BodoffOne`,
     `BodoffOneNet`, `BodoffWindQuake` and `NumericsBodoff` are one law under
     four names, differing only in unit names and spelling; `BodoffTwo` and
     `BodoffTwoNet` likewise. **What does the `Net` suffix mean?** Nothing in
     any of those programs is net of anything. a297 repaired `BodoffThreeNet`,
     which had been pasted as a byte-for-byte copy of `BodoffThree`, to use
     `dsev` like its siblings; whether the siblings should exist at all is the
     open question.
  2. **`PIRTame` and `ThinThinPortfolio` are the same program**, differing in
     unit names and in `PIRTame`'s `hints`. Two purposes, one law.
  3. **The Tweedie names read backwards.** `TweedieDirect` is claim count plus
     gamma **mean and cv**, `TweedieFromMoments` is claim count plus gamma
     **scale and shape**. Moments are the first pair.
  4. **`ExactGamma` has no gamma in it.** It is the exact counterpart of
     `ApproximateGamma`, which is what the name is reaching for, and
     `ApproximateRightSlognorm` keeps a `Right` that no longer contrasts with
     anything.
  5. **`Simple` is a filing word**, not a description: `PoissonSimple`,
     `DiscreteSimple`, `LimitProfileSimple`, `PHDistortionSimple`,
     `WindowedSimple`. Same for `Pair` on `BivariateCatPair` /
     `BivariateNetCededPair` / `SignedPortfolioPair`, where
     `BivariateNetCeded` and `BivariateNetCededPair` are near-identical names
     over substantially different programs.
  6. **Range notation cannot appear in the library.** `format_program` expands
     `dsev [1:6]` to `dsev [1 2 3 4 5 6]`, so no entry can demonstrate the
     range spelling, and `ThreeDice`'s own doc describes `dsev [1:6]` beside a
     `<<decl>>` that renders the expansion. Either record the range on the spec
     (the `_tweedie` pattern again) or stop naming it in prose.
  7. **The `sev` clause-form names are muddled**: `SevScaled` is scaled *and*
     shifted, `SevShifted` is the signed one, `SevOneParameterScaled` has no
     scale factor, `SevReversed` is a lognormal by scale and shape.
  8. **Inner engine names in the P&L block follow no rule**: `PnLprem.Vec_e`,
     `PnLsigned.Asym_e`, `E.PNL_e`, `APX.PnL_e`.
  9. **Sixteen entries carry tags but no `note{}`.** A note is the norm and is
     the SPA dropdown blurb, so each is a small gap: `ApproximateAggReins`,
     `ApproximateLeftReflect`, `ApproximatePnL`, `ApproximateRightSlognorm`,
     `BivariateClaytonMixed`, `BivariateIndependent`, `BivariateNormal`,
     `BivariatePnLAxis`, `CCoCDistortion`, `MinimumDistortion`,
     `PHDistortionSimple`, `PnLBook`, `SignedPortfolioMixed`, `TVaRDistortion`,
     `UnitSeverity`, `WindowContinuous`. Regenerate the list with
     `build.recipes.query('not note')`.
- **[Aliasing-Test-Misfires-On-A-Reference-Severity]** (found writing
  `SplitLimitPolicy`, a297) — `valid_aggregate`'s aliasing test asks whether the
  aggregate mean relative error exceeds `ALIASING_RATIO` times the **severity**
  mean relative error. A `sev agg.NAME` reference on a matched grid is exact, so
  that severity error collapses to `1e-13`, the ratio loses all headroom, and an
  aggregate accurate to `1.5e-11` is reported as `fails agg mean error >> sev,
  possible aliasing; try larger bs`. Raising `log2` makes it worse, since the
  aggregate error grows with the point count while the severity error does not.
  The `VALIDATION_NOISE` silencer at `1e-12` is the only thing standing between
  a correct model and a wrong verdict, and `SplitLimitPolicy` only lands under
  it because `bs=1/4` keeps the grid small. Either exempt a reference severity
  from the ratio test or floor the denominator.
- **[Session-Build-Clobbers-The-Trailer]** (was `[Recipe-Run-Clobbers-The-Trailer]`,
  found running the a297 recipes; **rewritten for the general case a300**) — a
  session `build(...)` **re-registers** its entry in the shared recipe base, so
  building a program that carries less trailer than the library entry of the
  same name silently replaces that entry's `note{}` and `tags{}` with nothing.
  The concrete trigger that found it is gone: it was a recipe's Solution
  rebuilding itself from `<<decl>>`, which by design carries `hints{}` and
  nothing else, and `dev/plan-decommission-docs.md` removed the whole recipe-run
  path at a300. **The overwrite behavior itself survives**, and a user who
  builds `agg LimitProfile ...` of their own in a session still loses the
  shipped entry's trailer for the rest of it. Author ruled 2026-08-17 to keep
  this open on those terms. Either a session build should not silently
  overwrite a **library** entry (warn, or namespace the session base), or the
  overwrite should merge the trailer rather than replace it.
  **The "namespace the session base" arm is now available**, a302
  (`[Session-Isolation]`, `dev/done/plan-session-isolation.md` phase L1):
  `Underwriter.fork()` gives each caller a private recipe base over the shared
  parsed entries, so a build in a fork rebinds the fork's key and leaves the
  library entry it shadows untouched. That is how a multi-user host contains
  this bug, and `tests/test_session_isolation.py` pins the containment. It does
  **not** close the item: a single-process user, the Jupyter case, still
  overwrites the trailer in the one base they have, and the other two arms
  (warn, or merge the trailer) are still the open question there.
- **[Unparser-Reference-Gaps]** (surfaced by `[Library-Canonical-Layout]`, a178)
  — `decl_writer` cannot render three constructs back to what was written, so 13
  `library.agg` entries are exempt from the canonical layout and hand-written.
  The list is `UNPARSER_EXEMPT` in `tests/test_agg_libraries.py`; shrinking it
  is progress. The `tweedie` clause was the fourth and is **done** (`a231`,
  `dev/done/plan-tweedie.md`): it carries a `_tweedie` provenance key, renders
  its clause back, and no longer overwrites the author's `note{}`. The pattern
  it established, record what was declared and render from that, is what item 1
  wants.
  1. **Named object reference** (8 entries) — `sev.UnitSeverity` is resolved and
     inlined at parse time, and nothing on the spec records that a reference was
     written, so it renders as `dsev [1]`. Fixing it means carrying the
     reference on the spec (a `sev_ref` key, say) and rendering from that. This
     is the one worth doing: named-severity reuse is a documented feature that
     would otherwise appear nowhere in the shipped library.
     **The pattern now exists and the key is taken.** `a291`
     (`[Agg-As-Severity]`, `dev/done/plan-agg-port-as-sev.md`) added
     `sev agg.NAME` / `sev port.NAME` carrying exactly this `sev_ref` key, the
     verbatim dotted id, and `decl_writer._render_sev_clause` renders it back as
     a reference from day one. Those are the first dotted references in DecL
     that round-trip as references. `sev.NAME` can join them by recording
     `sev_ref` alongside the inlined spec, since that reference *is* resolved at
     parse time and the writer branch would need to prefer it over the inlined
     keys rather than being their only source.
     **Author ruled 2026-08-18: declined.** "It **is** different and is ok to
     keep it so. If a user wants the other treatment they wrap it in a
     `dfreq[1]` agg." So `sev.NAME` keeps inlining, reference semantics are
     spelled `agg Wrapper dfreq[1] sev sev.NAME` then `sev agg.Wrapper`, and
     the 8 exempt entries stay exempt. Ruling 6 of
     `dev/done/plan-session-isolation.md`, where it was asked because a cache keyed
     on program identity would have depended on the answer; the design that
     shipped at a302 does not, because it qualifies on what the parse
     *resolved*, never on how the writer renders.
  2. **Distortion combinator** (1) — `minimum` / `mixture` drop their child
     distortion names, so `format_program` raises rather than rendering.
  3. **`ssev <c> - <dist>`** (3) — renders in the general affine form
     `-1 * <dist> + <c>`, whose leading `-1 *` parses two ways
     (`tests/test_grammar_ambiguity.py`, `KNOWN_AMBIGUOUS`). Either teach the
     writer the compact spelling when the scale is exactly `-1`, or resolve the
     grammar ambiguity. `dev/done/reflow_library.py` refuses any rendering more
     ambiguous than its source, so this cannot regress silently.
- **[Agg-Library-Build-Check]** (from `plan-for-v1.md` §1 — *"the one change that
  makes step 1.2 real"*) — **DONE a299**, `tests/test_library_entries.py`
  (phase C of `dev/plan-decommission-docs.md`). Every one of the 168 entries
  builds as its own case, `agg` and `port` carrying `valid`,
  `validation_explanation`, `summary_df` and `stats_df`. Two baselines make a
  new failure a finding: `CANNOT_BUILD` (six entries that deliberately refuse,
  pinned by exception type) and `VALIDATION_BASELINE` (26 entries that do not
  clear validation, pinned by exact flags and grouped by cause), both asserted
  in **both** directions so an entry that starts passing is a finding too. Six
  entries over a second are `slow`; the fast set runs in about 20 s.
- **[Cookbook-Feature-Coverage]** — **WITHDRAWN a298** (`dev/plan-decommission-docs.md`).
  It was written against a three file split (`cookbook.agg`, `examples.agg`,
  `actuarial-severity-curves.agg`) that the `library.agg` merge already
  retired, and the cookbook it names is gone. The live part of the intent,
  that every entry in the shipped library actually builds, is
  `[Agg-Library-Build-Check]` above, which lands as phase C of the same plan.
  `decl-testers.agg` keeps its own job and is unaffected.
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
  **Phase 5 is WITHDRAWN a298** (`dev/plan-decommission-docs.md`). It was the
  conversion of the stub pages to generated recipes, and the cookbook it
  generated into is gone. The seven doc bodies became notes in
  `aggregate-presentations`; the invariants they asserted become explicit
  pytest at phase C of the same plan. What survives of `[Recipe-Library]` is
  the `Recipe` record itself, `note{}` / `tags{}` / `hints{}`, `recipe()` /
  `recipes`, and `discover(tags=)`, none of which this plan touches.
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
  `[Ledger-Insurer-Abbreviated]` landed `a305`: the INSURER ledger narrows to
  `EX`, `SD`, `CV` and the adverse tail state (`κ01`, or `P01` on a marginal
  ladder), four columns rather than thirteen, with RAW keeping the whole
  sheet. That makes punch list item 1, CV blanking on `Margin` rows per
  decision 13, load bearing rather than cosmetic: `CV` is now one column in
  four, and every result row of a ledger is a margin row, so the sheet's
  worst cells are a quarter of what a reader sees.
  Next: the app consolidation (`[Exhibits-App-Consolidation]`), which is the
  SPA reading envelopes and the migrated app knowledge finally being deleted.
  `[Exhibits-PnL-Translation]` raw stage landed `a203` (pnl_ledger and
  pnl_ratios as RAW passthroughs); the INSURER framing draft awaits the
  author gate (captions, footing rules, Side sign presentation, the tower
  reshape; see the draft section appended to `dev/plan-exhibits.md`). Open with the author: per measure formats for the stats
  insurer view (greater_tables formats are per column, the store mixes
  measures down a column, so the app's measure formats have no TableSpec
  home yet) and whether the PnL validation audit ever gets a failure gate.
- **[Format-Sheets]**: the column formats move out of seven module level
  dicts and into two YAML sheets shipped as package data,
  `aggregate/formats/formats-raw.yaml` (the default reading of every named
  column) and `formats-insurer.yaml` (an overlay holding only the entries
  that differ). Overridable from `~/.aggregate` and the working directory,
  nearest winning, the same rule a user `.agg` database follows. The sheet
  doubles as a **registry of the column vocabulary**: an entry asserts that
  a label means one thing across the package, and the sweep makes naming
  drift a test failure. Plan: `dev/plan-formats.md`, drafted and ruled
  2026-08-14. **Progress:** `[Format-Sheet-Files]` landed a286 (the two
  sheets, `exhibits/_formats.py`, the three stop search path, styles and
  the four greater_tables tag styles, load time validation, `pyyaml>=6.0`
  declared). `[Format-Sheet-Application]` landed a287 (wired into
  `build_exhibit` post relabel, the seven dicts and both `ratio_cols` call
  sites deleted, 124 exhibit snapshots regenerated). `[Format-Sheet-Enforcement]`
  landed a288 (the served column sweep, two structural exemptions, and
  `PENDING_VOCABULARY`, the 43 label punch list the sweep produced). **All
  three phases done; execution record in the plan's section 9.**
  **Follow up, `[Format-Sheet-Patterns]` a295**: a `patterns:` section keyed
  on a regex (whole match, file order, expanded in the loader against the
  block's own labels so greater_tables still sees exact words only), the
  scoped `exhibits:` section restructured to carry `columns:` and
  `patterns:`, and the shipped `e[0-9]+\.m[0-9]+` entry for the moment
  store's mixture components. A scoped `'.*'` is a per-exhibit default,
  which retires the idea that the sheets need a `float_format` section. The
  summary and validation half of the punch list is closed in the same
  version, so the pending list is down to 31.
  **Open with the author:** the rest of the punch list, and the naming drift
  it caught (four spellings of a mean, three of a standard deviation, three
  of a skewness). Declined 2026-08-16: bounding the SI ladder to a window,
  which would need new `FormatSpec` fields plus csv-grid work.
- **[Exhibit-Official-Channels]**: closes
  `dev/note-from-aggregate-api-round-6.md`, the round in which the app stopped
  building table documents out of frames it fetched, so every gap the library
  left became a live regression on screen rather than something the app papered
  over. Plan: `dev/plan-exhibit-official-channels.md`, which also installs the
  invariant that a RAW block is exactly one public frame in its own
  orientation and that INSURER is the only perspective that may restructure.
  Scope boundary, author 2026-08-11: the framework is what these phases are
  about, complete, used and working; whether a given exhibit reads the business
  right is a separate conversation. **Progress:**
  `[PnL-Consideration-Rounding]` landed a251 (note item 5);
  `[Chart-Doc-Reader]` landed a252 (note item 6, `load_chart_doc`);
  `[Exhibit-Perspective-Contract]` landed a253 (the RAW invariant written and
  swept, `PnL.walk_df` / `PnL.evaluation_df` promoted out of the exhibit
  layer per `[Waterfall-Frames-Are-Owed]`); `[BS-Window-Diagnostics]` landed
  a254 (note item 2, `W` and `coverage` published, and four served frames
  given honest index names); `[Sharpen-Exhibit]` landed a255 (note item 3,
  the twelfth exhibit, RAW one block and INSURER two);
  `[Reins-Insurer-Orientation]` landed a256 (note item 1, the Aggregate
  layering analysis turned over and split into contract and consequence).
  **All six note items are closed**: item 4, the computed pricing exhibits,
  closed through the pricing plan below; 1, 2, 3, 5 and 6 at a251 to a256 as
  listed above. Phase 7 struck, its bug fixed at a250. Phases 8 to 10, the
  Pricing pane redesign keyed on result objects, were superseded 2026-08-12
  by `plan-pricing-exhibits.md` (canonical in the API repo's `dev/done/`,
  pointer here in `dev/done/`), which carried the pricing leaves end to end.
  The `REINSURER` question stays deferred (INSURER drops the
  `ceded` rows; the seller's reading waits).
  **All five LIB phases landed 2026-08-12, a259 to a263**:
  `[Pricing-Result-Objects]` a259 (`CalibrationResult` / `EvaluationResult`,
  breaking return types, lazy allocation frames, `SourcedMixin`);
  `[Unbounded-Anchor-Guard]` a260 (`p=1` refused on an unbounded risk at five
  entry points); `[Evaluate-Asset-Anchor]` a261 (the round trip closes, `ccoc`
  joins the anchored panel); `[Reins-View-Pricing]` a262 (`reins_price_df`
  adopts the pentagon octet and drops `bid`, `reins_view=` on both pentagon
  front doors, `lr=` on `calibrate_distortions`); `[Pricing-Exhibits]` a263
  (the three `pricing.*` exhibits, exhibit count 12 to 15). Review notes and
  the nine places the code and the plan disagree are in
  `dev/done/plan-pricing-exhibits-LIB.md`. **The three API phases landed at
  `aggregate_api` a83 to a85**, and the app's Sharpen leaf moved onto the a255
  `sharpen` exhibit at a94, emptying its `tables.FORMATS`; `bs_window` gained
  its column formats here at a267 (`[BS-Window-Formats]`). Both plans moved
  to `dev/done/` 2026-08-13. **The one open remainder, the Bounds
  registration** (phase 10's bounds half), moved to the author's master list
  in `dev-files.md` ("Getting to 1.0"). **Owed:** `dev/FEATURES.csv` regen
  once the in-flight refresh lands.
  **Follow up from the first app-side use of the pane:**
  `[Allocation-Default-Linear]` a265, `dev/done/plan-fix-unbounded-ccoc.md`.
  The pentagon surface hardcoded `allocation='lifted'` and never read
  `allocation_method`, so Calibrate at `p < 1` on an unbounded book served an
  Allocate subtab with no `ccoc` row. All six call sites now resolve `None` to
  the member. Per unit capital splits move on any default-path readout, which
  is the a17 intent arriving late. **Owed on the API side:** re-pin
  `test_pricing_exhibits.py::test_a_skipped_distortion_is_a_warning_and_not_a_failure`
  and rewrite `run_calibration`'s docstring Notes, both after the sync.
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
  Plan: `dev/done/plan-chart-ir.md`, **closed to done 2026-08-12**: the 1.0
  scope landed (eight app emitters by a244, first-class `.plot()` through
  `plot_chartdoc`); the bivariate tail rides in `dev/plan-3d-plot.md` and the
  remaining conversions are post-1.0 by the plan's own charter. **Progress:** inventory landed a197
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
  `[Chart-Payload-Weight]` landed a238, outside the nine jobs and at the
  author's request: a series carries a regular grid as a lattice
  (`x_lattice` / `y_lattice`, `(start, step, count)`, taken only where it
  is exact) and an empty run as its endpoints, which took the aggregate
  document from 7.4 MB to 5.4 MB and a lattice book's from 1.6 MB to
  0.05 MB. **`CHART_IR_VERSION` is 2**, the first application of the
  version rule written down at a233: a reader that ignores `x_lattice`
  sees a series with no coordinates at all. Job 5 `[Chart-Severity]`
  landed a239, on the author's design: four panels to two, because the
  log density is a reading of the density and the distribution is the
  Lee diagram transposed; the series is `continuous` unless the law has
  no density.
  **Next: job 6 `[Chart-Bounds]`** (settled: two panels, the weighted
  cloud and all five calibrated distortions on one band), then
  `[Chart-Bivariate]`, then `[Chart-Portfolio]` and `[Chart-Reins]`
  last, both still blocked on author design work.
  `[Chart-Invertible-Lee]` landed a240 (author's call: keep the named
  switches, they are discoverable and documented and there will not be
  many more): `Panel.invertible` plus `inverse_title`, and
  `plot(invert=True)` on the aggregate, P&L and severity draws the
  distribution function their Lee panels invert to. The ladder and the
  return-period pairing both followed the exchange with no special case,
  which is what reading them off the axes bought. Job 6
  `[Chart-Bounds]` landed a241: two panels, the cloud shaded by weight
  and all five calibrated distortions on one band, `y2` used for the
  first time, `ChartSeries.value` added for the bracket weight.
  Job 8 `[Chart-Portfolio]` landed a242, on the author's design: density
  (with log declared) plus the **kappa** panel, `exeqa_*` filtered to
  `p_total > KAPPA_FLOOR = 1e-14`. The floor was **measured**, not
  chosen: the unit kappas sum to `x` by construction, so that identity's
  residual is kappa's own error, and it runs (median, worst) 4.6e-9 /
  3.2e-4 at 1e-14 against 9.9e-5 / **0.52** with no floor, with the same
  numbers at log2 16 and 18. The kappa axis takes the *loss* window,
  since the curves sum to the diagonal.
  `[Chart-Equal-Aspect]` landed a243 (author's ask): an equal-aspect
  panel gets **one window for both axes**, topped by the higher of the
  two and floored where the data starts, and it no longer shares an axis,
  since squareness would otherwise drag a neighbour's window about. The
  kappa panel is square. Job 9 `[Chart-Reins]` landed a244, on the
  author's spec: the document is the occurrence plot (per claim on log
  only, the year as a Lee diagram carrying log, return period and
  inversion), **Aggregate only at 1.0**, so the `Portfolio` registration
  and the `basis` option are gone and an aggregate-cover-only object has
  no reinsurance chart. `plots/_aggregate.py` and `plots/_quantile.py`
  are deleted with it; `MAX_RETURN_PERIOD` moved to the renderer.
  **Only job 7 `[Chart-Bivariate]` is left**, and the 3-D punchups the
  author wanted are now written down: `dev/plan-3d-plot.md` (a symlink to
  the canonical copy in `aggregate_api/dev/`), a three-party plan over
  LIB, API and the SPA.
  **The LIB half of that plan is done**, in two bumps, with review notes
  and the five points where the code and the plan disagree recorded in
  `dev/plan-3d-plot-LIB.md`. a257 `[Joint-Density-Clip]`: both 2-D FFT
  de-fuzz sites route through one helper that warns on a large negative
  rather than preserving it, on a floor relative to the mass the grid
  carries. a258 `[Joint-Surface-Contract]`: the display coordinate is the
  block's **first** fine cell rather than its last (a support reported as
  starting at 508 on a distribution supported from 0, and a mean biased up
  by close to a whole display bucket); the window is measured on the fine
  lattice and cropped **before** the block factor is chosen, which is
  worth a factor of fourteen in resolution; and the surface block carries
  both lattices as origin, step and count, `bs` and `k`, the realized
  window, the exact marginals, the fine-lattice means, the deficit, and
  `z` again as a declared-dtype base64 block. Additive, so
  `CHART_IR_VERSION` stays 2; **phase two, dropping the `x` / `y` / `z`
  arrays, is the breaking change and bumps it to 3**, and waits on the SPA
  having moved. Still open on the LIB side and out of that plan's scope:
  a `Portfolio` to bivariate route (post 1.0), and the massive disk-backed
  joint, which has no surface chart.
  Then `[Chart-Portfolio]` and
  `[Chart-Reins]`, both still blocked on author design work.
  Not a job but noted at a241: `AllocationBounds` and `PricingBounds`
  are `_HullEngine` subclasses on a `Portfolio`, **not** built around a
  `Bounds`, so they carry no `cloud_df` and the envelope chart does not
  serve them; `plot_hull_bounds` stays bespoke and would need its own
  emitter.
  **`[Chart-Reflected-Reading]` landed a269**, outside the nine jobs: a
  sixth declared reading, `ChartAxis.complement_of` and the renderer's
  `reflect=True`, the map `v` to `1 - v`. The Lee panels of `agg`, `pnl`,
  `severity` and `reins` gain a `survival` axis carrying the log reading
  the non-exceeding probability cannot offer, and `distortion` and
  `envelope` reflect both axes of the unit square to the dual.
  `CHART_IR_VERSION` stays 2; the six documents that gain an axis move
  their hashes. The two probability readings compose with no lookup
  table, since `complement(v) = reciprocal(1 - v)`, which corrects the
  plan's ruling 4 as drafted (author, 2026-08-13). Plan in
  `dev/done/plan-chart-reflect.md`; the app half is planned in the API
  repo (`aggregate_api/dev/plan-chart-reflect.md`) and is **not started**.
  **The survival window was raised at review and ruled on** (author,
  2026-08-13): the reflected axis declares `(0.0, 1.0)`, so a log reading
  of it opens to `LOG_FLOOR` at 1e-15 rather than to `SURVIVAL_FLOOR` at
  1e-9. Flooring it at 1e-9 trims too much and the reading can go
  deeper, so `(0.0, 1.0)` stands and the renderer's decade floor is the
  right answer. That leaves `charts/_two_panel.py:survival_window()`
  exported, documented and called by nothing, which is now a deliberate
  state rather than a gap. One follow-up the work surfaced, small and
  not blocking: `_emit_portfolio.py` sets
  `meta['return_period_map'] = 'complement'` on a document with no
  probability axis, so it pairs with nothing and does nothing.

---

## After the cut — `beta`

### Reporting

- **[Agg-As-Severity-Result-Cache]** (from `dev/done/plan-agg-port-as-sev.md`
  §11) — every build of an aggregate whose severity is a `sev agg.NAME`
  reference rebuilds and re-updates the referenced object, so a program that
  both defines and uses an inner builds it twice, and iterating on an outer pays
  the inner's FFT each time. The cost is milliseconds to tens of milliseconds at
  typical resolutions, and the mandatory `hints{log2=…; bs=…}` make the rebuild
  exactly reproducible, so correctness does not depend on caching. Cache a built
  inner on its stored recipe **only if profiling shows it matters**; the reason
  it was not done at a291 is that a cache introduces staleness questions
  (redefinition, hint edits) that the fresh re-resolution simply does not have.
- ~~**[Reference-Severity-Zero-Atom-Default]**~~ **CLOSED by author ruling
  `1.0.0a294`, no change to the default** — a layers clause on a reference
  severity conditions on exceeding the attachment, exactly as on a
  hand-written `dsev`, and stays that way. The observation behind the item was
  that a zero-truncated inner has no zero atom in theory but does in the
  materialized `dsev` (any severity with positive density at the origin
  discretizes mass into the first bucket, so `gamma 50 cv 2` leaves about 7%
  there), so the default conditioning moves the split-limit answer by 6%. The
  ruling: that mass is what it is, there is no reason to reach for `!`, and it
  is the user's call. The a291 warning kept its number and lost its
  recommendation.

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
  DecL descriptor (`'occ 4750 xs 250'`, `'agg 85% po 1500 xs 7000'`, respelled
  from `so` at `1.0.0a249`) via
  `_tier_label`; a multi-layer tier keeps the generic `'ceded occ'` / `'ceded
  agg'`, because it has no single descriptor and `peel` is how you see those
  layers separately. Peeling forced the pick: `_ledger_plan` raises on duplicate
  row labels, so one group per layer cannot share one generic name. Test churn
  landed in the same sweep (six modules).
- **[Doc-Fix]** — clear the executed-cell `*Error`s in the docs build. **Last —
  after the code has settled** (it churns with every API change). Plan:
  `dev/doc-fix.md`.
- **[Bucket-Baseline-Script-Rot]** (found at `1.0.0a249`) —
  `scripts/bucket_baseline.py` no longer runs: it reads `uw.knowledge`, which
  the `Underwriter` has not had for a long time, and raises `AttributeError` on
  a clean tree. Its output `tests/data/bucket_baseline_summary.csv` was last
  written at `a49` and the script last touched at `a71`, so the committed
  baseline is ~200 versions stale and quietly wrong (it still shows the retired
  `so` spelling). No test reads it, which is why the rot went unnoticed. Decide
  between fixing the script and regenerating, or deleting both: a review
  artifact nothing checks and nobody can regenerate is worse than no artifact.

### Numerics & pricing core

- **[Bivariate-Total-Exeqa]** (phase 3 of
  `dev/done/plan-natural-allocation-to-occurrence-net-ceded.md`, deferred at
  execution `1.0.0a273`/`a274`, **author gate**) — the generic conditional
  mean for any bivariate: `E[X | X + Y = s]` and `E[Y | X + Y = s]` on the
  **total's** own grid, as against `exeqa_df`'s conditioning on an axis. It
  serves copula-mode pairs (allocating a dependent two-unit total, which no
  Portfolio machinery can do), it is the same quantity the 3D surface's total
  cut and kappa dots read client side today, and it would make the two-joint
  additivity check of `tests/test_bivariate_exeqa.py` a single-object one.
  Where the two axes share a `bs` the anti-diagonal is lattice aligned and the
  sums are exact; where they differ, route value-weighted mass and plain mass
  through `_scatter_1d` onto the total grid and take the ratio. Deferred
  because the netceded ask does not need it; recorded because it unifies three
  consumers and should be designed once. Still deferred after
  `dev/notes-net-natural-allocation.md` and `dev/plan-pricing-natural-allocation.md`
  (`a277` to `a285`): the kappa band conditions on an **axis**, which is the
  easy case and the one the netceded question asks, so nothing there touches
  this. What did land next to it is `JointBandsMixin._row_bands`, the row-wise
  iterator this would fold over on the massive route.
- **[Massive-Kappa-Second-Sweep]** (from `[PnL-Punchups-01]`, `1.0.0a134`) — bring
  the kappa scenario percentiles to the massive one-sweep P&L route.
  Conditioning needs the joint per atom *and* the grand-result quantiles before
  indicator-weighted means can accumulate — a second band sweep. Until then the
  massive `stats_df` keeps **marginal** ladders — since `a136` visibly so: plain
  `P01…P99` headers vs the in-memory scenario `κ` columns
  ([Decision-Kappa-Shared-Source-Rule]). Note the 2-D follow-up's
  [Gross-Anchored-Kappa-Insight] (G-slices are axis-aligned row averages,
  one-pass even on the massive route) may largely dissolve this for the
  variable-feature exhibits. **Half of that dissolution landed at `a278` to
  `a279`**: `JointBandsMixin._row_bands` is the iterator, and
  `exeqa_df(levels=...)` is a worked example of per-row conditional quantiles
  over it, in core and on disk alike, which is the shape this item needs. What
  remains here is the P&L side, which conditions on the grand result rather
  than on an axis. The app surface waiting on it is the Pricing pane's greyed
  `Massive joint` toggle (`aggregate_api`, plan-pricing-natural-allocation
  phase B2).
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
  **Second instance, a301:** `tests/test_reins_bivariate.py::test_netceded_refuses_a_pin_it_cannot_honor`
  failed once under a plain `pytest -m 'slow or not slow'` (no
  `-W error::RuntimeWarning`), then passed alone and passed on an immediate
  re-run of the same full gate. Same profile as the original: a bivariate
  suite, memory-heavy, load-dependent, unreproducible in isolation. Two
  different tests in two different bivariate modules now points at the shared
  cause rather than at either test, which strengthens the worker-load
  hypothesis and widens the symptom past the RuntimeWarning flag.
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

- **[Recipe-Doc-Signing]** — **MOOT a301** (`dev/done/plan-decommission-docs.md`).
  It proposed authenticating a `doc{{{...}}}` body with a trailing
  `<!-- hash: ... -->` so `Recipe.run()` could refuse an unsigned or tampered
  recipe. Nothing in the library executes code out of a `.agg` file any more:
  the clause, `Recipe.run` and the whole doc half of `Recipe` are gone, so a
  `.agg` file is data again and there is no payload to sign. Recorded rather
  than deleted because the underlying question, whether a recipe base from
  elsewhere can be trusted, is worth remembering if executable content is ever
  proposed again. The answer then was a documented warning; the answer now is
  that there is nothing to execute.
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
  `net of 50% po 500 xs 500 at .3 rol or 3000 ceded or .25 ros` (rate on subject
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

---

## From the beta-gate review, folded in 2026-07-27

> These are the surviving live items from `dev/REVIEW.md`, the condensed
> objective review of the library taken on 2026-06-21 around `1.0.0a89`. That
> page is now deleted and this section is the record. Its verdict in one line:
> the engine and the pricing / allocation science are beta ready and genuinely
> distinctive, and the gap to a *confident* beta is n-unit portfolio dependence
> plus a finished experience layer (guides, examples, docstrings, API freeze).
> The experience layer is already tracked above, in *Docs & packaging* and
> *Docs*; the dependence item is the first entry here and was the review's own
> number one priority.
>
> **Closed since the review, so do not reopen:** the `distributions.py` monolith
> split (shipped `1.0.0a90` to `a95`, *Hygiene & tests* preamble),
> `[ZT-ZM-Frequency-Fix]` (`a152`), the published API stability policy
> (`docs/3_reference/3_x_API_Stability.rst` plus the `CHANGELOG.md` preamble),
> and reinstatements, now full grammar and engine with corridor cessions
> alongside them. The verbatim source the review page condensed,
> `dev/beta-review-2026-06-21.md`, is recoverable from git history.

- **[Portfolio-Shared-Mixing-Dependence]**: the one strategic item on this list,
  and the review's most material functional limitation. Portfolio units combine
  by **independent convolution** (`_portfolio_density`, the independent-sum FFT
  combine), and the dependence that does exist is scattered across three places
  that do not compose: the two-peril copula in `bivariate.py`, sample-only Iman
  Conover, and allocation-only comonotonic. For a capital-allocation tool that
  is the gap users will find first. Proposed clean fix: a **mixing variable
  shared across units**, a common shock, composing with the existing PGF
  machinery to give principled n-unit dependence without copulas. The author's
  annotation on the review page: this is where Iman Conover and the switcheroo
  come in. Design before code. Knock-on to record while designing:
  `bivariate.py` is roughly 2k LOC serving exactly two perils, and its cost
  against benefit is worth rereading once a shared-mixing story exists.
- **[Deductible-Vocabulary]**: franchise / disappearing deductibles and an
  annual aggregate deductible as first-class constructs. Corridor landed on the
  reinsurance side (`corridor <share> po <width> xs <attachment>` in
  `decl.lark`), so these two are what remains of the review's richer-deductibles
  ask. Nothing in the grammar or the engine matches `franchise`, `disappearing`
  or an aggregate deductible today.
- **[Esscher-Exponential-Premium]**: the one classical premium principle the
  distortion framework does not subsume. Today it exists only inside
  `pedagogy.py`'s premium-principle comparison figure, not on the pricing
  surface.
- **[Distortion-Production-vs-Research]**: separate the production distortions
  from the research zoo (CLL, CLin, LEP, LY) in the docs, and possibly in the
  namespace, so a new user meets the handful they should reach for rather than
  the whole catalogue.
- **[Allocation-Bounds-Gold-Standard-Tests]**: extend the gold-standard testing
  style past the engine into allocation and bounds. Closed-form checks wherever
  an analytic allocation exists, plus `hypothesis` invariants: `q` monotone,
  TVaR at least VaR, allocation sums to the total, net at most gross.
- **[README-Scope-Statement]** (rider on `[README-Stable-Body]`): say what the
  library is **not** on the front page. Loss development / IBNR / triangles,
  stochastic reserving, credibility, GLM and experience rating, multi-year
  dynamics, inflation and trend, and cat-model internals are all correctly out
  of scope, and stating so is the best defense against unfair missing-feature
  critiques. Name the seam too: cat-model output enters cleanly as empirical
  `dsev` / histogram severities.
- **[PIR-Reproduction-Documented]**: "the published PIR exhibits reproduce only
  under `pip install aggregate==0.30.1` in an isolated environment" is tribal
  knowledge, recorded in `CLAUDE.md` and nowhere a book reader looks. Readers
  who try the current release and fail will distrust the library. Make it loud
  in the README and the docs.
- **[Two-Surfaces-Blessed-Path]** (rider on `[v1-Journey-Philosophy]`): there are
  two expressive surfaces, DecL and the objects. Document the blessed path per
  task, DecL to construct and objects to analyze, so users do not have to infer
  it.
