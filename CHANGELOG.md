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

## 1.0.0a329

**[Summary-Validation-Swap] the bivariate frames take their first-class names: `summary_df` is the at-a-glance headline, `validation_df` the moment audit, and the gate table goes private.** `BivariateAggregate.summary_df` is now `Mean | SD | CV | Skew | P01 | Median | P99` over the two marginals (by resolved label) and the realized dependent `total`, matching `Aggregate` and `Portfolio`; the old audit body (Freq / Sev / Agg blocks, eight columns) moved verbatim to `validation_df`; the old check table (`Est | Ref | Err | Gate | Pass`) became the private `_gate_checks()`, still feeding `info`, `validation_description` and `validation_explanation` unchanged, and the `validation.insurer` exhibit now serves the moment audit with gate-derived row emphasis. `qd(bv)` prints the new headline frame. `[Signed-Spread-Fix]` rides along: `_signed()` now asks each component (an `ssev` or negative-atom `dsev` component flips the audit to SD columns, as does a signed netceded source), where it previously saw only `pnl` affines.

Breaking for readers of the old frames: code indexing `summary_df` by `(component, part)` tuples must read `validation_df`, and the check table is no longer public (only `Distortion` still serves a check-shaped `validation_df`; a later `[FCC-Validation-Uniformity]` pass is logged in `dev/TODO.md`).

## 1.0.0a328

**[Bivariate-Punchup] the probability accessors land: `marginal`, `conditional` and `total` on `BivariateAggregate` and both joint containers.** `marginal(axis)` returns the axis marginal as a `GridDistribution` (axis by index, `'x'`/`'y'`, or component name, case insensitive; a `pnl` axis comes back payoff oriented); `conditional(kind, value, report=None)` returns the full conditional law for `kind` in `x | y | x+y | x-y` (the diagonal kinds condition on the total-grid bucket containing `value`, and `report=` picks which axis's law is returned); `total` is a cached property holding the realized law of `X + Y` (exact anti-diagonal fold on a shared `bs`, mean-preserving scatter otherwise). All three live on `JointBandsMixin`, so the in-core and massive containers expose one identical probability surface (`MassiveBivariateDistribution.marginal(i)` is unchanged in behavior but now served by the mixin); full `pushforward` signature unification stays deferred as `[Bivariate-Pushforward-Parity]` in `dev/TODO.md`. From `dev/plan-bivariate-punchup.md`, first of three bumps.

## 1.0.0a327

**[Library-New-Examples] 43 worked examples join `library.agg`, the capstone chain among them.** From `dev/done/new-examples-added.md`: an eleven-entry capstone section under the new `topic:capstone` tag (a complete reinsurance pricing workflow: a gross book entered from a limit and attachment profile, exposure rating, layer loss picks, an XOL program, quota share variants with a sliding scale commission, P&L and grossnet views, each link pulling the previous one in by reference); the renewal frequency group (`years` with `wait` / `dwait`, exponential through defective); the reinsurance economics group (`rate` and `cede`, reinstatements in treaty language and number words, swing, retro, slide, profit commission and corridor over a shared `VariableFeature.Base`); the bivariate group (`clash`, dense and sparse `dbvsev`, and the three view prefixes rebuilt DRY over `BivariateOccReBase`, replacing the inline `BivariateNetCeded` it duplicated); the P&L expense trio; and the small forms (`ExposureRatedPolicy`, `TowerLimitProfile`, `EqualWeightMixture`, `SplicedDisjointSegments` rewritten to a genuine two-segment splice, `UnconditionalDiscreteSeverity`, `PascalFrequency`, `DelaporteMixedFrequency`, `PayoffPrimitive`, `MixtureDistortion`). Carries the in-flight tag vocabulary migration (`topic:spectral`, `topic:economics`, the `check:` namespace retired) and its two new guard tests. The library's first nested `dbvsev [[...]]` flips `preprocess` step 3 onto its depth-aware path, which does not pad brackets, so the canonical-layout test now compares whitespace-normalized text; `UNPARSER_EXEMPT` grew into a catalogue of every source spelling the unparser cannot recover, covering the 27 pre-existing offenders that padding had been hiding. `MED.*` renamed `CommAuto.*` for the filing-prefix rule; `SLOW_ENTRIES` re-measured (nineteen new entries over a second) and `VALIDATION_BASELINE` gains `Capstone.SelectedLosses` (picks) and `DelaporteMixedFrequency` (a 1.1e-4 count-tail deficit the discrete update path cannot be told to widen away). Divergences from the source file, and two findings for later (the improper one-component two-list splice, the discrete path ignoring `log2`), are recorded in the execution notes of `dev/done/new-examples-added.md`.

## 1.0.0a326

**[Parse-Error-Hints] `ErrorReport` gains a `hint`, and the first rule explains `and` after an expense label.** `500 fixed expense as FE and 15% premium expense as Comm` reported a bare `Unexpected 'and'`, accurate and silent on the cause; the report now adds a sentence saying `as` closes an expense group, so either the `and` goes (separate groups, one leg each) or the label moves after the last `and`-joined term (one combined leg). `ErrorReport.hint` is `str | None`, defaults to `None`, adds a key to `to_dict()`, renders as a trailing `Hint:` line, and appends to `summary`, so `str(e)` carries it. Rules live in `parser_errors._HINT_RULES`, each matching the source text immediately before the error position. Consumers that build their own display from `to_dict()` rather than `render()` or `summary` need to read the new key to show it.

## 1.0.0a325

**[ZT-ZM-Recalibrate-Default] `zt` and `zm` now deliver the claim count you asked for, and `!` opts out.** `4 claims ... poisson zt` used to give 4.0746 claims and `4 claims ... poisson zm 0.5` gave 2.0373, the exposure clause setting the un-modified (base) mean of the (a, b, 1) construction and the reweighting shifting the realized `E[N]` off it. The bare clause now states the realized mean and solves for the base mean behind it; the trailing `!` selects the old textbook reading. `Aggregate.freq_pin_mean` changes default from `False` to `True` to match, so the constructor and DecL read the same way. `Frequency.solve_base_mean` is now on the common path, so its refusal is reachable from ordinary programs: a zero-truncated count cannot average below one, and `0.5 claims ... zt` that used to build now raises naming `!` as the escape. `ZeroModifiedExposureWarning` fires only on the `!` path, a monetary target being pinned by default. `create_frequency` was applying the modification twice under the new reading and is fixed; it reported 142.857 against a `zm 0.3` parent's own 100.

**Breaking, both tiers.** Direct `Aggregate(freq_zm=True, ...)` callers who relied on the old default get a solved base mean where they used to get the textbook parameterization; pass `freq_pin_mean=False` to keep it. Every `zt` / `zm` line in `library.agg`, `decl-testers.agg` and `_test_suite.agg` had its marker toggled so all nineteen affected entries build to identical numbers, verified entry by entry against `a324`. `expected_specs.json` regenerates, eight lines.

## 1.0.0a324

**[Format-Program-Picks-Line] `picks` gets its own line in the spread layout.** `_render_sev_clause` returns a `_Block` when the severity carries picks: the distribution heads the clause and `picks [attachments] [losses]` is its child, one level deeper. `_render_dist` gains a `split=` argument returning `(head, picks fragment)`; its other two callers are unchanged. The terse form does not move a byte, so `spec_to_decl`, the `to_agg` export and the round-trip corpus are untouched: the trailing `!` of an unconditional severity and an interior severity label both close the whole clause, so both ride at the end of the picks fragment. The `clash` renderer flattens the clause with `_render_terse`, a clash component being one line by construction.

App ask, recorded here per the grammar ripple agreement: `web/src/decl-keywords.json` has no `picks` entry, so the editor does not color it. LIB already does (`decl_pygments`). One line, on the app's own next bump.

## 1.0.0a323

**[Reins-Insurer-Terms-Block] the reins exhibit's contract block serves `loss` in place of `pr_loss`, and `output` as an integer.** Under the insurer perspective `reins_layer_terms` now reads share, limit, attach, pr_attach, pr_detach, **loss**, lol, output. `loss` is the row's expected aggregate loss, read off the store's `('agg', 'mean')`; no frame row is added and `reins_stats_df` keeps `pr_loss`, which the raw perspective still serves. It pairs with `lol`: `loss` is the placed figure, so a half placed layer shows half the dollars, while `lol` divides by share times limit and is share independent. `output` casts to `int64` in the view, so the served TableDoc carries an integer dtype. `tests/data/exhibit_snapshots.json` regenerates, one key of 124.

## 1.0.0a322

**[Reins-Insurer-Moments-Block] the reins exhibit's layer moments block gains the cover and narrows frequency to its mean.** Under the insurer perspective the `reins_layer_moments` block now reads cover | freq | sev | agg over share limit attach | mean | mean cv skew | mean cv skew. The three cover columns are the `meta` share, limit and attachment restated under a `cover` component group, repeated from the contract block above so this one stands alone; the frequency cv and skew are dropped. Layer frequency is the ground up count thinned by the probability a loss reaches the layer, so freq mean times sev mean is agg mean on every row, now asserted by `test_reins_layer_frequency_is_the_thinned_count`. View level only: `reins_stats_df` does not move and the raw perspective still serves every dropped column. `tests/data/exhibit_snapshots.json` regenerates, one key of 124.

## 1.0.0a321

**[Reins-Economics-On-Bvagg-Ignore-Warn] a reinsurance economics clause on a bivariate component builds instead of raising.** `deposit` / `rol` / `rate`, `cede`, `reinstatements`, the variable rating features and `retro` parse anywhere the shared agg body parses, so they reach a `bvagg` on its unit specs, where both unit construction routes splat the spec into `Aggregate` and raised `TypeError: Aggregate.__init__() got an unexpected keyword argument 'occ_reins_premium'`. Every view pair prefix (`netceded`, `grossceded`, `grossnet`) was affected, so the app's GCN control failed on any priced occurrence program, as did any copula component carrying a clause. The factory now filters each unit spec through the new shared `underwriter._strip_ignored_economics`, which the plain `agg` branch also uses, warns once per unit with `IgnoredDecLClauseWarning` naming the unit, and keeps the cession layers the netceded joint needs. `ignored_clauses_message` takes a third argument, `context`, choosing the remedy sentence between `'agg'` and `'bvagg'`; a joint carries loss against loss, so the `pnl` / `xpnl` recipe the agg message offers is not repeated there. The stored recipe keeps the full declaration on both routes.

## 1.0.0a320

**[Recipe-Seq-As-Read] a recipe records where it was read and how it was written.** `Recipe` gains `seq`, the zero-based position the entry was read in (counted across the whole recipe base, so a second `.agg` file continues the count), and `as_read`, the entry's DecL laid out as it appears in its source file, with comments and the terminating `;` removed and the trailer kept. Both are new columns on the `recipes` frame, which still sorts alphabetically: `recipes.sort_values('seq')` is reading order. `as_read` is `''` for a session build, and a rebuild of an existing name keeps that name's place. New `UnderwritingLexer.raw_statements` splits a program into statements without flattening them. `as_read` is what still says `dsev [1:6]`, `ph 2/3` and `ceded to tower [...]`, since the parser expands or evaluates each of those and keeps only the result.

**A single-band `splice` renders in the compact one-list form.** `_render_splice` emits `splice [8 12]` rather than `splice [8] [12]` whenever the bands are contiguous, which is what the library writes, so `SevSpliced` and `BivariateCatPair` now match their source byte for byte.

## 1.0.0a319

**[Picks-Off-Grid-Error] a picks attachment that misses the grid raises a ValueError naming a bucket that would work.** `_picks_work` validates every attachment against the realized grid before any frame work: off grid, above the top of the window, or infinite are all refused, where an off grid attachment previously died with a raw pandas `KeyError` from the survival lookup and an infinite one took the same route. The message names each offender, the realized `bs` and window top, and the largest halving of the bucket that divides every attachment, with `hints{bs=...}` given as the DecL spelling. Attachments are not snapped: a layer boundary inside a bucket cannot divide that bucket's mass between the layer below and the layer above. The three survival lookups behind the layer integrals are now positional rather than exact float labels, which is bit for bit identical on every grid that built before.

Nothing that built before changes. `build('agg X 1 claim sev lognorm 100 cv 2 picks [100 200 500] [45 20 25] fixed')` now raises instead of crashing, and builds on the suggested `bs=4`.

## 1.0.0a318

**[Reference-Trailer-Preservation] a builtin reference keeps the referenced entry's stored trailer.** `agg_out_builtin` merges only the trailer keys that carry a value, so an absent outer clause no longer overwrites the stored `note` and `hints` with the empty strings `trailer()` seeds; an outer clause still wins wherever it is written, and `tags` is unchanged, having never had a seeded default. `agg.MED.WithPicks` builds again, where it died with a raw `KeyError` from the off grid picks adjustment. A reference build also stops writing the blanked trailer back into the recipe base, which had left the entry on the wrong grid for the rest of the session.

Numbers move: a program referencing a hinted library entry by `agg.NAME` now builds on the entry's pinned grid rather than the auto sized one, and sixteen `library.agg` entries carry `hints{}`. The rename form, `agg NEW agg.OLD`, still drops the stored trailer and is left alone here.

## 1.0.0a317

**[Agg-Magic-Validation] the `%%agg` magic gains `-v`, `--validation`**, which `qd`s each built object's `validation_df`, the moment vs estimate audit, in place of the object summary. An object without the frame (a recipe stub, an `expr` value) displays itself as before, and the volume flags apply unchanged. `magics.py` only.

## 1.0.0a316

**[Mixture-Thinning-Moments] a severity mixture splits the claim count by thinning the frequency, not by scaling its mean.** New `MomentAggregator.thin_moments(wt, m1, m2, m3)` returns the moments of `Binomial(N, wt)` given the parent's; `Aggregate` resolves each exposure row's frequency once and thins it per component. `MomentAggregator.add_f1s` is deleted, `_record_component` takes a frequency moment triple rather than a count, and `Frequency.carries_own_count` is new (true only for the empirical family). `dfreq` with a severity mixture was wrong in the answer as well as the report: `dfreq[1] sev [2764 24548 275654 1917469 10000000] * expon wts [...]` now gives a severity mean of 13,990, against 12,220,435 theoretic and 2,442,979 from the FFT before. A mixture under a `logarithmic` frequency no longer raises. `reins_stats_df` reports the excess claim count as the thinned count, so `dfreq [1 2] dsev [10 20 30]` with a `10 xs 10` layer shows 1.0 claims reaching the layer, not 1.5.

Numbers move and one program now raises. Per-component `stats_df` columns and the `independent` total change for fixed, binomial, empirical (`dfreq`), renewal (`years`), Neyman A, Pascal and any zero-modified frequency; `mixed`, the FFT answer, and every Poisson or mixed Poisson program are unchanged. An exposure profile of more than one row under `dfreq` or `years` now raises `ValueError` instead of returning a `nan` variance and half the correct mean. Downstream: the two `reins` exhibit snapshots are re-captured.

## 1.0.0a315

**[Notebook-Magic] the `%%agg` cell magic: a DecL program can be the cell.** New module `aggregate.magics`, loaded on request with `%load_ext aggregate.magics`, supplying one cell magic. `%%agg` followed by a DecL program builds it, binds it, and `qd`s it, which is what `a = build('''...''')` did with the language buried inside a Python string literal. Out of the quotes, an editor sees DecL and highlights it as DecL. A version of this shipped as `agg_magics.py` before 0.21 and was lost in the reorganization; this is that idea rebuilt on the current surface, at the author's request.

**Explicit load, on the house preference for less magic.** `import aggregate` does not register it. A magic that arrives unasked is a name in the notebook nobody declared, and the module imports IPython at module scope, which `aggregate.utilities` goes to some trouble to keep off the `import aggregate` path (it costs about a second). Keeping `magics.py` off that path too is the other half of that care.

**One magic, not two.** `Underwriter.build` is `Underwriter.build_many` plus an unwrap and a count check, so the magic always calls the plural form and unwraps when exactly one output comes back. One statement or twenty is the same cell, with nothing to detect and no second spelling to remember.

**What gets bound.** A single output binds the target name, `a` unless one is given, and also the declared DecL name when it is a legal Python identifier, so `agg Dice ...` leaves both `a` and `Dice`. Several outputs bind the target to a `{decl_name: object}` dict plus each legal identifier on its own. A name DecL allows and Python does not, `EV.Peel`, is reachable through the dict. A recipe that cannot stand alone, a named mixture severity, binds the `Recipe` itself rather than being dropped, so its spec is still in reach.

**Flags.** `-q` builds and reports what it bound without the `qd` display, for a cell declaring a dozen objects; `-s` prints nothing at all; `-p` also calls `.plot()`, for a cell declaring one object, and says it declined rather than putting several unlabeled figures under one cell. `-p` is independent of the volume, so `-s -p` draws and says nothing. `--log2` and `--bs` set the grid, `--bs` evaluated in the notebook namespace so `1/32` and a variable both work; a `hints{}` clause in the program does the same job and travels with the declaration.

**Documented where a new user will meet it**: a section at the foot of Getting Started with a pointer note beside the first `build` example, a section in `README.md`, and the module on the Auxiliary Modules reference page. `tests/test_magics.py` covers binding, the three volumes, the grid arguments, both plot paths and a parse error, and runs against a bare `Underwriter` that has read no `.agg` file, so a broken entry in the shipped library cannot fail it.

---

## 1.0.0a314

**[Chart-2D-Punchups] three axes stop deciding for the reader.** The library half, in full, of the paired plan `dev/done/plan-2d-punchup-requirements.md` (the app half is `aggregate_api/dev/plan-2d-punchups.md`, which is canonical for the whole change and runs after this). Nothing in the wire format moves: no new field, no `CHART_IR_VERSION` bump, no reader change. Three existing `ChartAxis` fields take different values on four declarations, and every `agg`, `pnl`, `sev` and `reins` document hash changes with them.

**The return period becomes an ordinary axis.** It was declared `scale='log'` over `1 .. 1e9`, which is the axis choosing the reading and the window on the reader's behalf: read linearly nine decades pin the curve to the left edge, and read on log most of what they show is the float dust past `SURVIVAL_FLOOR`. It now declares `scales=('linear', 'log')`, a suggested ladder of `1 .. 1e4` and a `full_range` of `1 .. 1e9`, so the axis follows the button. Left alone it draws the ladder; `full_range` opens the deep tail; `log` makes the opened tail readable. The ladder top is the new `charts._two_panel.RETURN_PERIOD_TOP`, and 1-in-10,000 clears the 1-in-100 and 1-in-200 anchors a panel marks by two decades (author ruling 2026-08-21).

**This changes what `.plot(return_period=True)` draws.** The renderer takes an axis' drawn scale from its declaration, so the library's own return-period picture is now linear on the ladder rather than log over nine decades, and `a.plot(return_period=True, log=True, full_range=True)` is what recovers the old one. That is the intended reading and not a side effect: the document declares what is honest and the reader chooses among the declarations. Both readings were always drawable; only the default moved.

**All three emitters that declare the axis moved together** (`_emit_aggregate.outcome_doc`, which serves `agg` and `pnl`, plus `_emit_reins` and `_emit_severity`). The plan named two; the severity emitter carried the same declaration character for character, and leaving it behind would have left one document reading differently from the other three with no way to fix it downstream. Author ruling 2026-08-21.

**The `reins` occurrence panel gets two honest axes.** `sev_density` keeps log as its drawn scale and now declares `scales=('linear', 'log')`: a layered severity read linearly is usually a spike and nothing else, but that is still a reading, the reader can see for themselves that it is a spike, and declaring one scale removed the control rather than the temptation. `claim` gains the log reading and, more to the point, an unconditional full extent. An **unlimited** program has no occurrence limit to crop to, so `_claim_window` declined, and because `full_range` requires a `suggested_range` the panel lost the zoom out and the log reading as well, though its extent was knowable either way. The extent now stands in as the suggestion (author ruling 2026-08-21), so the zoom out is a button that changes nothing, which says "there is no crop to undo" more clearly than a control that is missing. Closes item 12 of `aggregate_api/dev/api-punchlist.md`.

**The mass axis declares its full extent.** `outcome_doc` crops the ordinate to the subject's own peak whenever a severity companion overtops it by more than `COMPANION_HEADROOM`, then declined to publish the extent it had already computed, on the reasoning that `(0, the peak)` is the whole of it. True of the aggregate alone and false exactly when the crop bites, which is when the reader most wants the companion's head back. The extent is now read off the drawn series inside `outcome_doc`, so the suggestion and the extent cannot drift apart. Where nothing was clipped the two windows coincide, on the same "present and idle" reading as the unlimited `claim` axis. This was L3 of the plan, recorded there as open; ruled in on 2026-08-21.

**Downstream.** The API must run `uv sync --extra dev` with its server stopped before any of this is believed, or `importlib.metadata` keeps reporting the old `aggregate` and the moved axes never arrive. It then re-captures `dev/fixtures/charts.json`, since every chart ETag moved.

---

## 1.0.0a313

**[Validation-Infeasible-Flag] the feasibility reading becomes a validation member and a warning.** Part three, the last, of `dev/done/plan-validation-punchup.md`. a312 computed the reading and reported it through the grid narrative; this puts it where a user meets it, in `valid` and in a warning at build time.

**`Validation.INFEASIBLE` is a new validation type, not a failure** (author ruling 2026-08-21). It is a statement about the grid rather than about the outcome, so it fires whether or not the moments happen to fail. `Validation.passes` therefore treats it as **transparent**: the expression masks it out of the first term, so an infeasible object whose moments fail still fails, and one whose moments hold still passes, carrying the reading. It deliberately does **not** join the `REINSURANCE` arm, which makes a reinsured object with failing moments pass, because doing so would let the flag hide the very problem it names. `USXOLTower` is exactly that case in the shipped library: reinsured, infeasible, and still passing.

**It leads the explanation, and sits outside the failure list.** `DEFECTIVE` leads the moment flags because missing mass makes every moment comparison under it uninformative; `INFEASIBLE` leads even that, because the grid could not have reproduced the severity whatever else happened. The plan put the clause inside the `fails ...` list, which reads "fails grid infeasible for this severity"; it is instead placed ahead of the list, on the `reinsurance` idiom, so the short form reads `grid infeasible for this severity; fails sev mean, agg mean`, and for an object whose moments hold, `grid infeasible for this severity; not unreasonable`. The long `validation_explanation` carries the reasoning: what the size biased distribution is, why the mean and the tail want different resolutions, why a finer `bs` makes it worse, and why an occurrence limit is the fix. `exhibits/_core.py` puts it in `_SEV_FAILURES`, since it is a property of the severity against the grid.

**`InfeasibleGridWarning`** fires once per session per severity and grid, through `warn_once`, on the `ReflectedSeverityClampWarning` pattern; the key carries `log2`, so a genuinely different grid still speaks up. The message is long on purpose. The author's framing is that bucket selection is one of the largest hurdles FFT methods put in front of a user, and this failure does not look numerical from the outside, it looks like a mean that is simply wrong. So it names the bucket the body needs, the share of the mean the first half bucket takes and where that mass is placed, the reach the mean is not complete without, the `log2` that would hold both, and the one action that resolves it:

```
Cat: severity 'lognorm' cannot be reproduced on this grid. The body needs
bs <= 0.0302749 (46.22% of the mean is supplied below the first half bucket,
and is placed at 0); the mean is not complete until the grid reaches
1,025,760, so matching it to 0.0001 needs log2 = 25, and log2 = 16 was used.
A thick unlimited severity has this problem at any bucket size, because its
mean is furnished far above its median and a finer bs only shortens the
reach. Add an occurrence limit, or accept the reported moment errors.
```

**Measured on the corpus: four programs of 255 set it, and every one was already failing its severity mean.** `CurvePareto` (needs 16.99 against 16, marginal and real), `GrossCatXOL` (25.01), `HeavyTailValidation` (46.55), `USXOLTower` (25.01). Zero programs gain the flag on a grid that works, which is the invariant the plan asked to be asserted as a count. `HeavyTailValidation` is the one worth noting: it is the library's worked example of what a failed validation looks like, and it now says **why** it failed rather than only that it did. 132 of the 255 carry a reading at all; the rest are severity kinds the closed forms cannot read exactly, and they claim nothing in either direction. `VALIDATION_BASELINE` gains `INFEASIBLE` on those four plus `ThickThickPortfolio`, which inherits it from a unit.

**Docs.** `docs/2_aggregate_overview/pipeline-aggregate.rst` gains the flag in its `valid` list with the transparency rule spelled out, and `bucket-selection.rst`'s feasibility section gains the flag, the warning and the corpus count. Between them the two sections now carry the whole account, which is what the author asked for.

With this the plan is complete and moves to `dev/done/`.

---

## 1.0.0a312

**[Severity-Feasibility-Reading] the library now says when the grid cannot reproduce the severity's mean at any bucket size.** Part two of `dev/done/plan-validation-punchup.md`, the computation and its narrative. No flag yet, so the blast radius is prose; `Validation.INFEASIBLE` follows in part three.

**The problem, stated exactly.** `library.agg` ships `GrossCatXOL`, a US hurricane ILW model, `1.74 claims sev lognorm 8.501 cv 14.624 poisson` in billions. It is a real model, not a constructed pathology. Its severity mean is wrong by 41.6% and validation says "fails sev mean, agg mean", which tells the user nothing they can act on. What is happening: the lognormal has sigma 2.317 and median 0.580, so the median claim is a fifteenth of the mean; the sizer reaches to 8.3e6 to cover the aggregate tail, which at `log2=16` forces `bs=200`, so the median claim is one three hundred and forty fifth of a single bucket. Bucket zero collects everything below `bs/2` and places it at exactly zero, and the share of the **mean** supplied there is 46.2%. That single effect is the whole error.

**The reading is a ratio of two quantiles of the size biased law.** Writing `P_1` for the size biased distribution of the severity, `dP_1/dP = y / E[Y]`, the discretized mean loses `P_1(Y <= bs/2)` at the bottom and `P_1(Y > n bs)` at the top, and those are the only first order losses. So a mean accurate to relative `delta` needs `bs/2` below the `delta/2` quantile of `P_1` and `n bs` above its `1 - delta/2` quantile. Dividing one by the other eliminates `bs` and leaves a requirement on the bucket count alone. The size biased cdf is `F_1(t) = (LEV(t) - t S(t)) / E[Y]`, both pieces of which the library already computes exactly through `_moms_analytic`, so no new quadrature is introduced. New `size_biased_cdf` and `feasible_bucket` in `_severity.py`, and `severity_feasibility` in `_bucket_window.py`, which stores the reading on the aggregate as `_bs_feasibility` beside `_bs_clip` and `_bs_snap`.

**For an unlimited lognormal the mean cancels and only sigma survives**, `log2 >= 2 z sigma / log 2 - 1`, about `11.23 sigma - 1` at `delta = 1e-4`. The general numerical inversion agrees with that closed form to a tenth of an exponent across the four reference severities: `lognorm 200 cv 2` reads 13.24 against 13.2, `cv 5` reads 19.26 against 19.3, `cv 10` reads 23.12 against 23.1, and the cat severity reads 25.01 against 25.0. Against a working `log2=16` the practical boundary sits near sigma 1.51, a CV around 3: below it an unlimited lognormal is routine, above it no bucket size works and a finer one is worse.

**`log2_required` is measured against the severity's own reach, not the realized grid top.** The plan drafted it as `log2(grid_top / bs_max)`, on the argument that this accounts for the reach the sizer actually bought. Its own reference table, its message text and its tests all say `log2(reach / bs_max)`, and the control case settles it: `N.US.Hurricane` is `1e12 xs 0 sev exp(19.595) * lognorm 2.581`, sigma 2.581, **thicker** than the cat model and perfectly feasible, because the limit truncates `P_1` and the requirement is 15.7 against a realized 16. Its chosen `bs` is still ten times coarser than its body wants, which is why its mean is off by 0.089%, but that is a **coarse** grid rather than an impossible one, and only the second is what a feasibility reading should report. The grid based form reads 15.6 for `lognorm 200 cv 2` against a tabulated 13.2 and would fail the plan's own test.

**Read out through the narrative surfaces**, which is where a user already looks: `bs_description` gains a clause naming the shortfall, `bs_explanation` the full account, both fired when `log2_required` exceeds the realized `log2` by more than half an exponent (`FEASIBILITY_SLACK`, half an exponent because `log2_required` is continuous and the grid is a power of two). The long form names the three numbers that make the situation legible and the one action that resolves it: the bucket the body needs, the share of the mean the first half bucket takes, the reach the mean is not complete without, and that the fix is an occurrence limit rather than a finer grid. No new public member.

**Exact or absent, never estimated.** The reading is available for the severity kinds whose partial expected values `_partial_e` computes in closed form, `lognorm`, `gamma`, `pareto` and `expon`, which covers mixed exponentials. It reports nothing for a discrete or histogram severity (exact on its own lattice, so the question does not arise), a signed or reflected law (whose `fz` is patched, and whose closed forms answer for the unreflected base), and any other continuous family. The plan allowed a quadrature fallback for those; it is declined here, because quadrature inside a bisection at sizing time is both slow and fragile on exactly the heavy tails the reading is for. `grid_is_infeasible` is `False` on a missing reading: not knowing is not the same as knowing the grid is fine, and a flag raised on ignorance is worse than no flag. For a severity mixture the binding component, the smallest `bs_max`, is the one reported.

**Nothing about grid selection changes.** The reading is computed after selection and chooses nothing. Two things the plan explicitly ruled out stay out: the sizer is not capped at the resolution requirement, because buying reach is the decision and it stays (and capping would have moved `GrossCatXOL` only from 41.6% to about 8.6%); and one moment local moment matching for a gross severity, which would make the discretized mean exact at any `bs`, is refused because it would repair the reported number while leaving the model's mean furnished by a region no one has an opinion about, which is worse than an obvious error because it is silent. The existing mean preserving scatter for reinsurance rebucketing stands, where the grid is forced and the means have to work.

**`docs/2_aggregate_overview/bucket-selection.rst` carries the whole account**, at the author's request. Its "Two failure modes" section becomes three, the third being a severity no grid in the budget can resolve, and a new "Feasibility: when no bucket size works" section derives the size biased requirement, tabulates the four reference severities against their realized errors, works the `N.US.Hurricane` control, and records the two deliberate rejections. The same section corrects the aliasing paragraph for a311: wrap conserves mass and is `ALIASING`, clipping drops it and is the pmf deficit and `DEFECTIVE`.

---

## 1.0.0a311

**[Aliasing-Direct-Measure] the `ALIASING` flag now measures aliasing.** Part one of `dev/done/plan-validation-punchup.md`, the plan that came out of diagnosing `[Signed-Bounded-Window]`. The flag had six firings in the 257 program corpus and six false positives, and the cause was structural rather than a badly chosen threshold.

**What was wrong.** The test was a bare ratio: fire when the aggregate mean relative error exceeds `ALIASING_RATIO` (10) times the severity mean relative error, floored on the numerator only, at `VALIDATION_NOISE` (1e-12). Any severity that discretizes essentially exactly puts near zero in the denominator and the ratio explodes on nothing. That covers a discrete law, an integer lattice, a reinsurance layer landing on bucket edges, and the `sev agg.NAME` reference that produced `[Aliasing-Test-Misfires-On-A-Reference-Severity]`. Three of the six firings had `2.2e-16` in the denominator, and the largest aggregate mean error among all six was 3.5e-7, three hundred times **below** the tolerance at which anything is called a failure. Raising the ratio does not fix a ratio whose denominator is machine epsilon.

**What replaces it.** New `convolution_residual(en, sev_mean, agg_mean)` in `_validation.py` returns `|E[A] - E[N] E[X]| / |E[N] E[X]|`, with the severity and aggregate means both read off the **discretized** law. Severity discretization error therefore cancels exactly rather than sitting in a denominator, and what is left is the error the convolution step itself introduced, which is what the flag's own explanation always claimed to be about. No new data is read: `stats_df` already carries all three inputs. `validation.aliasing_ratio` retires in favor of `validation.aliasing_eps`, default `1e-5`, one order of magnitude inside `eps`, which is the band where a convolution error is real but has not yet failed the mean outright. The module constants `ALIASING_RATIO` become `ALIASING_EPS` in `_validation.py` and `_portfolio.py`, with their re-exports through `distributions.py` and `portfolio.py`.

**The flag now means wrap, and nothing else.** Wrap conserves mass, relocating it, so a genuine wrap moves the mean while the pmf deficit stays at exactly zero. Truncation drops mass, so it always carries a measurable deficit, and `DEFECTIVE` and `AGG_MEAN` already own that case with an explanation that already ends "Raise log2, or widen the grid". `ALIASING` therefore sets only when the residual exceeds `aliasing_eps` **and** the deficit is arithmetic dust. The plan proposed gating at `deficit_materiality` (1e-4); measurement at a310 moved that to `VALIDATION_NOISE` on the author's ruling, because five corpus programs lose the mean to truncation with deficits between 1.5e-5 and 9.6e-5, every one under materiality, and three of them pass validation today. Gating at materiality would have made them fail for wrap they do not have, which is the old false positive wearing new clothes.

**Measured outcome.** Zero firings across the 257 program corpus, against six before. The forced case separates the two mechanisms cleanly: `100 claims sev gamma 100 cv 1 poisson` at `log2=14, bs=1` gives residual 5.40e-5 with deficit 0 at `padding=0` (wrap, flagged) and residual 1.36e-6 with deficit 3.29e-5 at `padding=1` (truncation, not flagged); the same book at `log2=12` is `DEFECTIVE` with no `ALIASING`. Note the default `padding=1` runs the FFT on a doubled grid and discards the top half, so true wrap needs the aggregate to reach past **twice** the grid. Real aliasing is rare by construction, and a flag that almost never fires is the honest outcome rather than a sign the change went too far.

**One true positive survives in the shipped library**, and it is the right one. `SignedPortfolioPair` loses 94% of its total mean with a pmf deficit of 1.2e-13: mass conserved and relocated, the definition of wrap, in the entry whose section exists to demonstrate exactly that. Its three former companions in `VALIDATION_BASELINE`, `SignedPremiumMinusLoss`, `SignedPortfolioMixed` and `ThinThinPortfolio`, leave the list with residuals of 3.5e-7, 2.5e-10 and 1.1e-12. They were never aliasing.

**The portfolio takes the same change with the same threshold.** The portfolio step is the convolution of its units, and every unit's own FFT is validated before this code is reached, so the predictor is the sum of the units' realized aggregate means. A unit carrying reinsurance sets `REINSURANCE`, which trips the early return, so the units' `empirical` and `gross_empirical` columns coincide for every unit that gets here.

**Wording.** The short form becomes "agg mean lost in the convolution, possible FFT wrap; raise log2". The long form's leading clause moves off the ratio and onto the measurement: the aggregate mean is not `E[N]` times the discretized severity mean, and no mass is missing. `docs/2_aggregate_overview/pipeline-aggregate.rst` is updated in step with the code.

---

## 1.0.0a310

**[US-Spelling-Bucket-Window] `_bucket_window.py` normalized to US spelling, which moves one served string and two private state keys.** House rule is US spelling everywhere, normalizing a whole file when it is touched rather than sweeping the repo. a309 touched this file and left its 44 British spellings alone; this is that pass, taken on its own so the bug fix stayed readable.

**One string a consumer can see.** The Portfolio sizer's realized-grid row read `realised portfolio grid (resolution=..., extent=...)` while the Aggregate sizer's read `realized grid (...)`, so the two halves of the same reporting surface disagreed with each other. The Portfolio spelling moves. It reaches the `bs_window` exhibit's `used` row, `bs_window_df`, and `bs_description` / `bs_explanation`. `tests/data/exhibit_snapshots.json` is regenerated, with exactly those four strings changing and nothing else. Nothing app side is owed: the note travels as served content.

**Two private state keys renamed**, `_sharpen_state['centre_score']` and `['centre_defective']`, to `center_score` and `center_defective`, along with the `sharpen_df` note `centre (current grid)`. `_sharpen_state` is private, is read in this repo only by `_program.py` (which reads neither key) and one test assertion, and is not part of any exhibit or chart payload.

The rest is prose: `honoured`, `realised`, `discretises`, `neighbouring`, `colour`, `centre` and their inflections, in docstrings and comments. `tests/test_sharpen.py` was normalized in the same pass, having been touched for the key rename, which renames `test_explicit_centre_is_honoured` to `test_explicit_center_is_honored`. `tests/data/bucket_baseline_windows.csv` is deliberately left alone: it is a one-time reference artifact from `dev/done/plan-bucket-combine.md`, read by no test, and it records what the code said when it was captured.

---

## 1.0.0a309

**[Signed-Bounded-Window] a layer clause on a signed severity was half applied, and the bucket sizer called `int(inf)` on the window that lie produced.** Executed from `dev/done/plan-signed-bounded-window-overflow.md`, rewritten and confirmed with the author on 2026-08-21. The reproduction, `build('agg NT 50 claims 25000 xs 0 ssev -lognorm 200 cv 10 + 180 mixed ig .4', bs=5)`, raised `OverflowError: cannot convert float infinity to integer` out of `_bucket_window._size`. Three fixes, at three layers, in root-cause order.

**The root cause is a clause that was recorded but never applied.** `Severity.__init__` stores `limit`, `attachment` and `detachment` from the layer clause before `_build` runs, then dispatches on `signed`: a signed severity takes `_apply_signed`, which installs identity layering with no `x < 0 -> 0` clamp and no attachment or limit transform. The bookkeeping stayed. So `sev.limit` read 25,000 for a law that never met a limit, `bounded` and `tail_class` reported `BOUNDED`, and nothing told the user that the clause they wrote had been ignored. The mathematics says it could not have been honored as written anyway: `y xs a` is `min(y, max(X - a, 0))`, and on a signed `X` the outer `max` annihilates the negative half, which is the entire reason for writing `ssev`.

**`Severity._drop_layer_clause` now resets the three fields and warns once**, through `IgnoredDecLClauseWarning`, the class a pure `agg` already uses for the ceded premium, reinstatement and variable rating clauses it has no premium context to activate. The message names the dropped clause and the two real alternatives, drop the layer, or use plain `sev` if the clamp was what was wanted. The unlayered `limit` is `inf` for a continuous law and the atom support max for a histogram kind, whose `_build` sets `limit = min(exp_limit, xs.max())` and so reports the support top when there is no clause: resetting a signed `dsev [-2 5]` to `inf` would have replaced one lie with another. Detecting the clause reads a new private `_exp_limit`, the limit **as declared**, because `self.limit` may already carry that histogram truncation and can no longer answer the question. Layering a signed base properly, with an arbitrary and possibly negative attachment, is a real feature and stays deferred as the successor to `dev/plan-negative-x-agg.md` section 6. A **negative** attachment is the meaningful general case there.

**`Aggregate._severity_high_estimate` stops capping a signed severity's reach by the same clause.** `Aggregate.limit` records the layer as declared, which is right, but that estimate uses it as a real bound on the law, and on a signed book it sizes the grid. Author ruling 2026-08-21, taken with the confirm above. Only the buggy combination is affected: an unlayered signed aggregate already carries `limit = inf`.

**`_bounded_severity_window` returns `None` when a computed edge is not finite.** The structural `_severity_bounded` test reads the spec; the window reads the support. They can disagree, and did: the reflect patch gives `fz.support() = (-inf, 180)`, so the method returned `(-inf, 4.98e6)`, a bounded window with an unbounded edge. The documented contract already tolerates `None`, and this is the lower edge, signed severity twin of the upper edge guard the splice path has carried since a230.

**The sizer no longer computes a number it will not read, and no longer reaches `int()` on one it cannot.** `_size` skipped straight past `need` when `bs` is pinned: the user owns the grid, `log2` is honored verbatim, and the crash was in arithmetic whose result was discarded three lines later. The remaining computation moved into `_need_log2`, which returns `inf` for a non-finite span, step or ratio, reading as "more than any cap" at every branch. The old guard, `if span > 0`, did not catch the case, because `inf > 0` is `True`. Finally `_row` records a method whose window has an infinite edge as inapplicable rather than sizing it, the `exact_discrete` unreachable-support precedent, so selection falls through instead of raising, and an inapplicable `bounded_small` row can no longer be selected.

**Ordinary aggregates are byte for byte unchanged**, which was the plan's regression bar. Every new branch is reached only by a signed severity carrying a layer clause, or by a window with a non-finite edge, and after the first fix the reported program never builds the bounded row at all. The reproduction now builds and reports itself **defective**: the raw `180 - lognormal` at `cv 10` has sigma about 2.15 and reaches roughly 14.5 million buckets over 50 claims, so the PMF deficit is 0.977 and the negative reach is clipped. That is honest, it is what the existing warnings are for, and it is the subject of `dev/done/plan-validation-punchup.md`, not of this fix.

---

## 1.0.0a308

**[Joint-Surface-Whole-Grid] the surface stops being the window: the whole reduced lattice travels, and `window` becomes the drawing range inside it.** The second of the two library edits in `dev/plan-3d-plot-LIB.md`, its sections 4 and 5, specified as section 5.8 of the canonical `dev/done/plan-3d-plot.md` and ruled by the author on 2026-08-12. With it that plan's library half is complete.

**The defect is what a consumer could not compute.** The emitter cropped before it emitted, so the served grid **was** the window, `window.x` and `window.y` came back equal to the lattice bounds, and there was nothing outside them to work with. Every conditional formed off that grid is then normalized by the **visible** mass, which is a different and less interesting object whose mean moves whenever the window does. Measured on the prototype at a useful depth, a third of a cut at constant total is off screen and the conditional expectation comes out 4.9 to 7.8 percent wrong, with the error changing sign along the total, so it bends the shape of the curve the chart exists to show.

**The window now selects, and no longer subsets.** It is still measured on the fine lattice and the block factor is still chosen from the cropped extent, which is what makes `detail` mean something and is worth a factor of fourteen in resolution on the reference Lomax. What changed is that the reduction is applied to the whole fine lattice at that factor. The blocking anchors at index 0, so a law supported from the origin is drawn from the origin whatever the window did. `window` keeps its shape and its four keys, and `kept` keeps its meaning to the letter, the share of the mass inside the box, now a sum over a sub-rectangle rather than over the whole array.

**`detail` reads on the window, not on the axis, and that is the one thing to know when reading a document.** It bounds the cells **across the window**; the emitted axis runs the whole lattice at that step and so is longer, by the ratio of the lattice to the window. Concretely, at the default depth `Indep` returns a y axis of 8192 cells whose window names 116 of them, where it used to return a 116 cell axis whose window was the whole of it.

**What it costs, and the ruling on it.** The reduced grid at the `k` the window chose: `Indep` 64 x 8192 against 31 x 116, `IndepSigned` 128 x 256 against 66 x 126, the Gumbel joint 512 x 1024 against 126 x 113. `Indep` is the honest worst case, a Lomax on a lattice wide enough to hold its tail, at 524k cells and 2.1 MB as float32 before transport compression. The author ruled on 2026-08-12 that size is not the constraint here and `detail` is the lever if it becomes one. If it does, the knob to reach for is **not** a crop: it is a `context` parameter saying how far beyond the window to carry, with the whole grid as its default, so the choice is stated in the document rather than taken silently by the emitter.

**A ragged tail pads rather than dropping.** `_pad_to_blocks` zero-extends an axis to a whole multiple of `k` before the reduction. The alternative was dropping the last partial block, which loses mass silently, or keeping it short, which breaks the uniform step every interpolation downstream divides by. The pad sits beyond the axis' support, so the block sums stay exact. Belt and braces on today's objects, whose axes are FFT grids blocked by powers of two, and cheaper written now than diagnosed later on the first axis that is not.

**The marginals move onto the whole lattice with the grid they label**, which is not optional: `SurfaceData` refuses a marginal whose length differs from its axis. They are still the object's own exact marginals rather than row sums of a windowed joint, which would be the marginals of a truncated distribution. One consequence worth having: the x marginal no longer moves when the y depth does.

**The library's own renderer had to learn the window, and this is the part the canonical plan does not cover**, having been written from the app's side, where `windowRange` already read the window as a sub-rectangle. `plots/_chartdoc.py` read neither `window` nor `edge`: it drew `surf.x`, `surf.y`, `surf.z` entire and pinned the limits to the mesh. On the day this edit lands that draws `Indep` as 8192 cells of mostly empty tail with the subject a sliver at the origin, which is the picture the window exists to prevent. `_surface_window` now reads the box, on the conservative renderer-only route the plan directs absent a ruling: zero wire change, and no effect on any other consumer. The document-level alternative, putting the box on `ChartAxis.suggested_range`, would change the app's flat heatmap reading too and is not ours to take unilaterally.

**The color scale reads the window as well, which is the half of that fix nobody had noticed.** The log floor is one decade under the smallest mass present, so taken over the whole mesh it is set by the far tail: five decades under the field the reader is looking at on `Indep`, six on `IndepSigned`, which compresses that field into the top of the ramp and flattens exactly the structure the log reading exists to show. It is now read over the drawn window. The mesh is still drawn whole, so panning out finds the tail.

**`pcolormesh` is called with `shading='nearest'`, which reads coordinates as cell centers**, so a308 lands on top of a307 having made that assumption true rather than merely close. Landing the two in the other order would have put a half display bucket of displacement into the library's own picture at the moment the mesh grew.

**Additive, so `CHART_IR_VERSION` stays 2.** No field is added, removed or retyped: `window` keeps its shape and its meaning while its relationship to the lattice changes. Phase two, dropping the `x`, `y` and `z` arrays in favor of the lattice fields and the encoded block, is the breaking change that bumps it to 3, and it waits on the SPA having moved. App side the chore is a re-capture of `dev/fixtures/charts.json` and a re-run of `smoke-charts.mjs`; no code moves, the decode having read `mid` and treated the window as a sub-rectangle since before either edit.

---

## 1.0.0a307

**[Joint-Surface-Representative-Point] a display cell is filed under the point its mass actually sits at, and `edge` becomes literally true.** The first of the two library edits in `dev/plan-3d-plot-LIB.md`, its section 3, specified as section 5.1.2 of the canonical `dev/done/plan-3d-plot.md` and ruled by the author on 2026-08-12.

**The defect this closes is a declaration, not an arithmetic slip.** A display cell covers fine atoms at `a, a + bs, ..., a + (k - 1) * bs`. a258 labeled it `a` and declared `edge='left'`, which is a true statement about a cell spanning `[a, a + dx)` and an invitation to recover its middle by adding `dx / 2`. The middle of the atoms is `(k - 1) * bs / 2` above `a`, so a consumer that accepts the invitation lands half a fine bucket past the answer, and one that does not lands half a display bucket short of it. Both readings were available and neither was right.

**The coordinate is now the block's representative point**, the mean of the fine coordinates it covers, and the document says `edge='mid'`, a value `SURFACE_EDGES` already carried. `window` follows it: the box is still the outer edges of the outer cells, now half a step outside the outer coordinates rather than starting on the first one.

**What the choice buys is a bound, not a better number in every case.** A block's conditional mean lies somewhere in `[a, a + (k - 1) * bs]`, so filing the block at the middle of that span holds the display mean's error under `dx / 2` whatever the density does inside the block, and no other single coordinate does: the low edge is one-sided and its bound is a whole bucket, reached by a block whose mass sits at the far end. Measured on the reference Lomax surface at the default window, where a block is two atoms, the residual is **-0.0150 display buckets against -0.2649 for the low edge and +0.2350 for a cell midpoint**, seventeen times better, and what is left there is the window's own truncation rather than the convention. Being a bound rather than a tendency, it does not promise to win every case: reduce that same axis 128 to 1 and the Lomax puts each block's mass hard against its low end, where this convention reads 0.45 buckets high and the low edge, flattered by the shape, reads 0.05 low. Both are inside the bound only one of them has, and neither is a mean a consumer should be reading off the picture when `moments` carries the exact one.

**The residual changed character, which is the part worth reading.** It used to be a convention, one-sided and bounded by the reduction. It is now the deviation of the within-block mass from uniform: second order, of either sign, and shrinking as the density flattens across a block rather than toward a fixed side.

**Nothing about the grid moved**, and that is the check that this edit touched the coordinate and only the coordinate. Verified against the previous release over both reference surfaces at four window depths and three detail targets: `nx`, `ny`, `k`, `dx`, `dy`, `z`, the encoded block byte for byte, both marginals, `deficit`, `moments` and `window.kept` are identical, and every coordinate moved by exactly `(k - 1) * bs / 2`.

**One consequence to state rather than leave to be discovered.** The first display cell of a law supported from zero now extends to `-bs / 2`, which looks like support below the origin and is not. The fine lattice already does this and `pcolormesh` already draws it for any centered grid in this library, so it is consistent rather than new. In the other direction the library's own renderer improves: `_render_grid_panel` draws with `shading='nearest'`, which reads coordinates as cell centers, so this edit makes an assumption that was already there true rather than merely close.

**Additive, so `CHART_IR_VERSION` stays 2.** No field is added, removed or retyped. Consumers holding a captured document re-capture it, the document hash having moved; app side that is one refresh of `dev/fixtures/charts.json`.

---

## 1.0.0a306

**[PnL-Reinsurance-Pricing] `pnl_program` learns to price the reinsurance: `net_combined_ratio` builds the premium from the bottom up, net technical premium plus the cost of each cover, grossed up once for expenses.** `dev/done/plan-pnl-reinsurance-pricing.md`, all four phases. The origin is a hole the library itself pointed at: wrap a reinsured engine in a P&L and the cession books at zero ceded premium with a `ZeroPremiumCessionWarning` saying "price the cover to silence this", and nothing in the library would price it. Now something does.

**The construction, and the identity that chose it.** With `E_net` the net expected loss, `E_j` layer `j`'s expected ceded loss and `c` the combined ratios: `T_net = E_net / c_net`, `Q_j = E_j / c_j`, `TP = T_net + sum_j Q_j`, and `P = TP / (1 - ER)` through the existing `derive_consideration`. The point is what falls out: the underwriting margin is `TP - sum_j Q_j - E_net = T_net - E_net`, exactly the net margin, so it **does not move when a reinsurance price moves**. Raising a layer's price raises the booked premium by exactly the extra cost and leaves the book's own result alone, which is what a reader turning these knobs should see. An earlier draft sized the premium straight off a loss ratio and had to ask whether the loss was gross or net; the question was wrong, and building upward dissolves it.

**Combined ratio, not loss ratio, on purpose.** On a technical premium there is no expense load, so on the net it *is* a loss ratio. The name is chosen for what the argument becomes when real market rates arrive, and for the slot a distortion's implied technical premium ratio will eventually fill.

**Per layer, because that is how market rates arrive.** `occ_combined_ratio` and `agg_combined_ratio` each take `None` (meaning `net_combined_ratio`, so one number prices the whole tower), a scalar for the tier, or **one value per layer** in declaration order, which is ascending attachment since `_validate_reins_layers` refuses any other. A length mismatch names both counts rather than recycling or truncating.

**Every cover is written as a `deposit`, and that is the load-bearing choice.** A first draft chose between a `rate` (against the P&L premium) and a `rol` (against the layer limit) by the layer's attachment probability. Review killed both halves. The `rate` form makes the booked premium depend on itself wherever a layer is already priced, and the probability the switch read is a **per claim** number on the occurrence tier and a per year one on the aggregate tier, so one threshold on it inverts the classification on a realistic tower: a `900 xs 600` occurrence layer reads `pr_attach` 0.0203 while paying in 18.3 percent of years, against a `500 xs 1500` aggregate layer at 0.1247 paying in 12.5 percent. A deposit is `share x amount`, so it references neither the premium nor the limit, and there is no threshold left to misread. The `working_attach` argument the switch needed does not exist.

**The share algebra, the one place this is easy to get wrong.** `reins_stats_df`'s ceded loss is **share adjusted**, the loss of the fraction actually placed, while a DecL premium clause is quoted at **100 percent placement** and scaled down by `share` at resolution. The deposit written is therefore `d_j = Q_j / s_j`, which resolves back to `Q_j` exactly. Writing `Q_j` itself would charge a half placed layer twice for its share. The quotient is also the number a reader would read off a quote sheet, being share invariant: a `50% po 100 xs 100` layer and the same layer at full line write the identical deposit, and pay half and all of it respectively.

**One rounding rule for every currency figure.** Deposits round through `_round_consideration`, the function the consideration already used: whole units above 100, cents at or below. Each deposit rounds **before** the technical premium is formed and `P` rounds last, so the emitted program is internally consistent to within the rounding of `P` alone and the residual on the margin identity is the closed form `P x (1 - ER) - sum_j s_j d_j - E_net` rather than merely something small. The tests assert equality, not a tolerance.

**A layer the author already priced is left untouched**, and still paid for: a `deposit` or `rol` clause resolves without reference to `P`, so its cost enters the total at the amount it will resolve to and the identity still holds. A layer priced on `rate` **raises**, naming the layer and the two bases that are known in time. The loop does have a closed-form solution, `P = A / (1 - ER - sum_j s_j r_j)`, but adopting it would put back exactly the self-reference the deposit form was chosen to remove. The case is not hypothetical: a bare `agg` accepts a `rate` clause, warns that it is ignoring it, and keeps it in the program text, so `pnl_program` was already copying it into a program where it *does* resolve against `P`.

**Two seams worth recording.** The clauses are read off the **re-parsed** spec `_require_program` returns, not `ob.spec`, because a ceded-premium clause on a bare `agg` is stripped from the built object (a plain aggregate has no premium context) while surviving in its text. And the resolved per layer economics a test needs ride at `pnl.engine._pnl_recipe['econ']`; `pnl._pnl_recipe` is `None`.

**Scope.** An `Aggregate` engine, both tiers, any number of layers, partial placements. A `Portfolio` engine raises: a unit's cession would have to be quoted against the whole book's premium, which wants its own ruling, and since the library does not warn about unpriced unit cessions today nothing is left half done. Ceding commission, reinstatements and variable features are out, each having its own premium interaction.

**Provably inert by default.** `net_combined_ratio=None` is the function exactly as it was, for a stated premium engine and a sized one, with and without reinsurance; the tier arguments are read only through the net one, so passing them alone cannot move a byte. No existing caller's text changes and nothing needs a deprecation. With a stated premium and **no** cover to pay for, the program still says `derive premium`, so the linkage to the engine's own exposure clause survives; ceded premium is a cost the sentinel cannot know, so once there is any, the number is written out.

**Where it lives.** Three new functions in `_program.py`: `_pnl_layer_ratios` (the `None` / scalar / sequence resolution and its length check), `_pnl_layer_premium` (what an existing clause already resolves to, and the `rate` refusal) and `_pnl_technical` (the ladder). `Aggregate.pnl_program` and `Portfolio.pnl_program` grow the same three keyword-only arguments and forward.

**A stale docstring corrected while passing through.** `Portfolio.pnl_program` claimed it emitted `less port.NAME` and that "the grammar has no inline portfolio engine". Both have been false since `1.0.0a216`, which added the portfolio twin of the inline agg engine precisely so the text would build outside the session that wrote it; the units have been written out inline ever since.

**Tests.** Sixteen new cases in `tests/test_derived_programs.py`: the default proved inert on three engines, the plan's worked example reproduced byte for byte and digit for digit, the deposit rounding on both sides of the 100 joint, the margin identity exactly for a sized premium and a stated one, the identity holding under a moved layer ratio while the booked premium absorbs the whole extra cost, the per layer sequence and both wrong lengths, an unusable net ratio (checked before the per layer resolution, since an engine with no cession reaches the division without passing through it), the share invariance both ways, an existing `deposit` and an existing `rol` kept and paid for, the `rate` refusal, the portfolio refusal, and the `ZeroPremiumCessionWarning` counted at one before and zero after. The identity assertions are relative to `1e-12`; the separate check that the P&L's booked loss matches the engine's `est_m` carries `1e-9`, which is where the grid arithmetic actually lives. New corpus programs in `src/aggregate/agg/decl-testers.agg` (DP block), inputs and emitted outputs both.

## 1.0.0a305

**[Ledger-Insurer-Abbreviated] the Economics ledger's INSURER view narrows from thirteen columns to four: `EX`, `SD`, `CV` and the adverse tail state.** Author request, 2026-08-19. The ledger sheet is the one exhibit a reader scans line by line, and `Skew` plus a nine rung percentile ladder made it a frame to slice rather than a sheet to read. The four that stay are the reading a ledger is opened for: level, spread, spread relative to level, and one tail.

**The tail rung is the bottom of the ladder, not the top.** A P&L is in payoff sign convention, left tail bad, so `κ01` is the adverse state and `κ99` is the benign one. The request named the top rung; the sheet says otherwise, and on the Tower fixture the bottom line reads `EX = 251` against `κ01 = -1380` and `κ99 = +860`. An abbreviation ending at `κ99` would have put the good news in the slot the eye reads as the bad. Ruled to `κ01` by the author on that basis. Ledgers with no shared atoms keep their marginal ladder and take `P01` under the same rule.

**RAW is untouched, and is the escape hatch.** `PnL.economic_df` still carries every moment and the full `PERCENTILE_LADDER`, and `exhibits.economic(pnl)` still serves all thirteen columns. Only the INSURER translation narrows, so the app's Raw toggle reaches the whole sheet in one click. This is the two perspective split working as designed rather than a capability being removed.

**Where it lives.** One hook, `exhibits._pnl._economic_insurer`, plus `_ledger_columns` and the two module constants naming the choice, `LEDGER_MOMENTS` and `LEDGER_TAIL_Q`. Column selection is by label and declines rather than raises on a label it cannot find, matching the rule `_ledger_row_flags` already follows. The captions are rewritten: they no longer promise `Skew`, they speak of one tail column rather than a ladder, and both regimes now close by saying where the dropped columns went.

**Formats and row flags are unaffected.** The format sheets key on the displayed label and ignore labels a block does not carry, so the survivors keep their readings; row flags are positional over rows, which the narrowing does not touch.

**The API side is a no op.** Under the purist ruling the app draws what it is served: the Economics, Ledger leaf loads the exhibit envelope and the narrower block flows through. No client-side reference to these columns exists, so there is no round-note ask and no API phase.

**Tests.** Three exhibit snapshots move, `economic/insurer` for the PnL, Tower and Peel fixtures, and only those three of the 124. `test_economic_raw` now asserts INSURER is a value-identical column subset of RAW rather than the whole frame; `test_measure_formats_where_measures_are_columns` names per case which measures its block still carries, since the ledger no longer has `Skew` to check. New: `test_economic_insurer_is_abbreviated` (the exact four columns in both ladder regimes, and that the tail column really is the adverse one, read off the `total` row flag rather than the last row, which on a walk is `Impact`) and `test_economic_raw_keeps_the_whole_sheet`.

**One stale line corrected while passing through.** The exhibit table in `docs/2_aggregate_overview/pipeline-exhibits-and-charts.rst` still listed the ratios amounts block as `(P, L, E, C, M)`; `C` went at `1.0.0a304`.

## 1.0.0a304

**[Cede-Contra-Expense] BREAKING, stable tier: ceding commission folds into the expense column and the `C` column of `PnL.economic_ratios_df` is removed.** `dev/plan-cede-expenses.md`, all five phases, on the author's 2026-08-18 ruling. `PnL` is a stable-tier class, so this is an alpha-series break of the kind the preamble reserves the alpha line for; the exhibit half rides the provisional `aggregate.exhibits` tier and needs no such notice. Ceding commission received is a **contra expense**, netted against acquisition expense exactly the way a cession recovery is netted against loss. The ratio frame had it both ways: `_RATIO_BUCKET` sent `'recovery'` into `L` and gave `'commission'` its own `C`, so loss and expense answered the same question two different ways.

**The posited justification was searched for and not found.** The originating plan (`dev/done/plan-pnl-ratio-frame.md`, a185) records no economic reason for a separate `C`. The closest things on record are a naming remark and the format sheet's "a column carries one unit" comment, which does not apply since both are money. The only rationale that can be reconstructed is keeping a gross, un-netted expense ratio, and it does not hold up: commission legs arise only on cession blocks, so the Gross row's `ER` is identical either way and the gross reading survives where it belongs. The library's own earlier design agrees with the ruling: the a116 GCN waterfall booked per-tier commission credits in the GCN **Expense** section, so the separate `C` introduced at a185 was the anomaly, not the tradition.

**What moves, and what provably does not.** `M == P - L - E` is the row identity, replacing `M == P - L - E - C`. `CR` and `E_CR` are **numerically invariant**: they summed `(L + E + C) / P` and now sum `(L + E) / P` over the folded `E`, which is the same money. `LR`, `E_LR`, `P`, `L`, `M`, `P_share` and `M_share` are unchanged everywhere. `ER` and `E_ER` change meaning wherever commission exists: on a cession block `ER` reads as the commission rate with conventional sign (a `cede 0.2` layer reports `ER = 0.2`, both numerator and denominator being negative), and on the `All` row it becomes the statutory expense ratio net of commission. Verified on a two-tier peeled walk: of the eight blocks only the one carrying the commission leg, its tier subtotal and `All` move at all.

**No information is lost.** `legs_df` keeps every commission leg itemized under `kind='commission'`, and that frame is the documented place to recover a split the ratio frame does not carry. `LEG_KINDS` is unchanged, so `'commission'` remains a declarable kind; only where it accumulates changed.

**A hand-built `sell` group leg tagged `'commission'`** (agent commission paid, rather than ceding commission received) folds into `E` as acquisition expense, which is also the correct reading.

**A comment corrected while passing through.** The `LEG_KINDS` note called `'commission'` a consideration-side flow on a `buy` group. Every builder declares commission legs on the **obligation** side, all thirteen of them, which is exactly why the amount enters negated and reads as a credit. The note now says so.

**Exhibits and formats follow.** `exhibits._pnl._AMOUNT_COLS` drops `'C'`, so the Economics, Ratios amounts block is four columns, and its caption states the shorter identity and names the two contras. `formats/formats-raw.yaml` drops the `C: money` entry; no other frame in the library carried a bare `C` column, and no other format sheet defined one.

**The API side is a no op.** Under the purist ruling the app draws what it is served: the Economics, Ratios pane loads the exhibit envelope and the narrower amounts block flows through. No client-side reference to the `C` column exists, so there is no round-note ask and no API phase.

**Tests.** Eight assertion sites moved to the shorter identity and column roster, across `test_create_pnl.py`, `test_exhibits.py`, `test_pnl_peel.py` and `test_exhibit_formats.py` (whose money-vocabulary list carried `C`). One new case, `test_ratio_df_commission_folds_into_expense_as_a_contra`, pins the fold end to end on a peeled walk that cedes on exactly one layer: that block's `E` is minus its commission, its `ER` is the cede rate, every uncommissioned block still reports `E == 0`, `All` carries the credit as its net expense ratio, and `CR` equals its pre-fold value. `tests/data/exhibit_snapshots.json` regenerated: the diff is the `C` column and its cells leaving, the trailing column keys renumbering, and the new caption.

**Docs.** `docs/2_aggregate_overview/pipeline-pnl.rst` carries the new roster, the shorter identity and the contra reading. `docs/2_aggregate_overview/features.rst` was deliberately **not** touched: its a185 section is release history, governed by `dev/task-features.md` as additive and repair-only, and it still spells the frame `ratio_df` from before the a204 rename. This change belongs in that page as a new section on the next task run.

## 1.0.0a303

**[Dfreq-One-Claim-Shortcut] `dfreq [1]` takes the same exact path as `1 claim ... fixed`.** `dev/plan-dfreq-one-claim-shortcut.md`, all four phases. With exactly one claim the aggregate **is** the (post occurrence reinsurance) severity, so the FFT round trip is an identity executed numerically. The convolution has always known that, but it recognized only one spelling of the fact: the gate read `sum(en) == 1 and freq_name == 'fixed'`, so `1 claim ... fixed` returned the discretized severity bit for bit while `dfreq [1]`, which builds an empirical frequency, ran `ift(freq_pgf(ft(sev)))` and came back carrying machine epsilon dust (measured 2.8e-17 per bucket, which compounds to about 1e-11 relative on the third moment). Two ways of saying the same thing, two different answers. Now both copy.

**`Aggregate.one_claim`, a new read-only property, is the single predicate.** True when the claim count is identically one, however the count was spelled: a `FrequencyFixed` whose component counts sum to one (a limit profile splitting that single claim over several components still qualifies, because `sev_density` is then the corresponding severity mixture), or a `FrequencyEmpirical` carrying all its mass on the outcome one. `FrequencyRenewal` is an empirical frequency post build, so a renewal count that degenerates to one claim qualifies too, at no extra cost.

**The empirical test is on the support, never on the mean.** `dfreq [0 2] [.5 .5]` has mean one and is emphatically not one claim; it keeps the convolution. Zero probability atoms are ignored, since `validate_discrete_distribution` makes the outcomes distinct and ascending but never drops a zero mass, so a declared but impossible outcome such as `dfreq [0 1] [0 1]` does not defeat the test.

**The kernel stops sniffing frequency semantics.** `_aggregate_compute.freq_sev_convolution` traded its `en` and `freq_name` parameters for a single `one_claim=False` boolean that the caller computes. It is a pure internal function reached from `Aggregate._fft_aggregate` and the direct kernel tests, so nothing public moved, and the change is the point: deciding what the frequency means is the object's job, and the kernel's job is the transform.

**One predicate, two gates, so they cannot drift.** `bivariate.netceded_joint_density` carried its own copy of the old inline test for the 2-D netceded joint. It now reads `agg.one_claim`, which is also how the 2-D path picked up the `dfreq [1]` case for free.

**`ftagg_density` is still computed on the shortcut path.** A `Portfolio` multiplies unit transforms to combine them under the independence copula, so the transform is needed even when the density is a copy. Only the inverse transform is skipped.

**The signed and windowed branch is unchanged.** It has no one-claim shortcut for either spelling, and keeps that parity; adding one there is a separate question about relabeling an array that was never convolved.

**Baseline moved by exactly one case.** `tests/baseline/` regenerated: `Base.DfreqOne` (`agg Base.DfreqOne dfreq [1] sev lognorm 100 cv 0.5`) moves in `density_df`, `stats_df` and `describe` at a maximum relative 1.0e-11, concentrated in the tail rows and the third moment, which is the dust coming out. The other nine cases are byte identical parquet, which is the regression bar the plan set: the FFT path itself was not touched, the shortcut only widened.

**Tests.** `tests/test_aggregate_compute.py` gains eight cases: four spellings of one claim (`1 claim ... fixed` and `dfreq [1]`, over both a discrete and a continuous severity) asserting `np.array_equal(agg_density, sev_density)`, three negative controls (a fixed count of two, the mean one `dfreq [0 2] [.5 .5]`, and a Poisson mean of one) asserting the predicate is False and the density is not the severity, and the zero probability atom case. The two byte-for-byte kernel tests migrate to the new signature.

**Docs.** `docs/2_aggregate_overview/pipeline-aggregate.rst` described the old asymmetry as behavior, including the sentence saying `dfreq [1]` carries sub-eps FFT fuzz that the shortcut path does not. It now describes the shared gate and records when the two spellings converged.

## 1.0.0a302

**[Session-Isolation] one recipe base, many users: `Underwriter.fork()`, `Underwriter.preview()`, and `RecipeNotFound`.** The library half of `dev/plan-session-isolation.md`, phases L1 to L4, executed while the application executed its own. Everything here stands on its own product merit, which was the author's condition for it landing in the library at all: a scratch base for a notebook, an honest report of what a program depends on, and a named exception for a name that is gone.

**`Underwriter.fork()` returns an isolated copy sharing the parsed recipes.** The recipe base is read and parsed once, and each caller who needs a base of their own takes a fork: a copy whose recipe dict is a fresh `dict()` over the same `Recipe` objects. Measured at about 7 microseconds per fork over 100, against roughly 3.2 seconds for a fresh load. Declarations built in a fork land in the fork and are invisible to the parent and to every sibling, in both directions; entries the parent already held stay visible to all of them. Shallow is right for the dict because `add_recipe` always rebinds a key to a brand-new `Recipe` and `recipe()` hands out `replace()` copies, so a shared entry is never mutated in place: overwriting a name in a fork rebinds the fork's key and leaves the parent's entry alone.

**Resetting the parser is the substance of the method.** The `parser` property builds `UnderwritingParser(self._safe_lookup, ...)` and the wrapper holds that bound method, so a plain copy would keep the *parent's* callback: the fork would parse against the parent's base while registering into its own. Split brain, silent, and in exactly the direction a fork exists to prevent. So `_parser` and `_lexer` are reset to `None`, which is nearly free (the Lark grammar is a module singleton and the wrapper is two attribute stores), and `_sev_ref_stack`, the `sev agg.NAME` cycle guard, gets its own list because per-instance mutable state cannot be shared. `databases` is copied, and `_request` is copied when it is a list, so a later `load()` on either side cannot rewrite the other's request. The parent is loaded first: a fork of an unloaded underwriter would inherit an empty dict and read the databases again on its own first access, once per fork, which is the cost the method exists to avoid.

**A fork is a snapshot, not a subscription.** Entries the parent gains afterwards do not appear in it. `config.reload_settings` mutates the module-level `build` in place, so forks taken before a reload keep the base they were taken from, and the module alias `build_many` stays bound to the singleton's method, never to a fork's.

**`interpret_file` is written on `fork()`.** It hand-rolled exactly this scratch copy, the fix that let a file reference a name it declared further up without the validation polluting the base doing the validating. One implementation now, in the public method, with the private copy deleted.

**`Underwriter.preview(program)` reports what a program resolves to, without building anything.** It returns a `ProgramPreview`: the `route` it took, the parsed `statements` as ordinary `Recipe` records, and every entry the program `resolved` as `ResolvedReference(kind, name, source)` triples. It registers nothing and writes no instance attribute, so two threads may preview at once. The cost is one parse, tens of milliseconds against builds measured in hundreds, which a caller that previews before building pays twice on the build path.

**Why the resolved list is complete by construction.** Every reference form funnels through `_safe_lookup`: the dotted forms are resolved there and inlined into the spec, and the one deferred form, `sev agg.NAME`, still calls it as an existence check before recording its symbolic `sev_ref`. Recording what that one callback saw therefore catches every reference form the language has, including any added later, with no text scanning and no spec walking. The bare name is the one program shape that never reaches the parser, so it is looked up first, in the order `_build_work` tries it, and reported the same way: a bare name is a reference.

**Two things the plan left to implementation, both recorded because they change what the report says.** First, the deferred chain is **followed**. An inlined reference brings the referent's spec with it, so what the program stands on is already in hand, but a `sev agg.NAME` does not, and the entry it names may reference a third. So the deferred targets are walked to exhaustion with a per-call cycle guard, never the instance `_sev_ref_stack`, since this runs outside any build and possibly beside one. Without the walk a program two steps from a redefined entry would report as leaning on nothing but the library, which is the one way the report can be wrong in the direction that matters. Second, the parse runs against a **fork**, so a multi-statement program that declares a name and uses it further down resolves the way `_interpret_program` allows, while provenance is read from the previewing underwriter, so a name the program declares itself is reported only if that underwriter also holds it. What a program depends on is what it did not bring.

**`RecipeNotFound` names the lookup miss, and is a `KeyError`.** Raised from the two not-found sites in `Underwriter.recipe()`, the one place lookup is implemented, so it reaches the subscript form `uw[name]`, the parser's `_safe_lookup` callback and the bare-name branch of `build` in one edit. It carries `kind` and `name` as attributes and overrides `__str__`, because plain `KeyError` renders `repr(args[0])` and would wrap a sentence of diagnosis in quotes. Subclassing `KeyError` keeps every existing `except KeyError` and `except LookupError` site behaving as before. What the named type adds is the ability to tell "that name is gone" from every other lookup failure, which is what a host has to do to answer an expired reference with "rebuild it" rather than "no such thing". An **ambiguous** name, the same name under two kinds, stays a plain `KeyError`: the entry is there, the question was not answerable.

**Behavior change, narrow: a deferred reference whose referent has vanished now raises `RecipeNotFound` rather than `ValueError`.** `_resolve_sev_ref`'s "it parsed, so it was there when the program was read; it has been removed or renamed since" diagnosis keeps its text and becomes the same exception type as every other missing name. A caller that wrapped a build in `except ValueError` to catch it now needs `except (ValueError, LookupError)`. Nothing in the library or the test suite caught it, and telling a vanished name from a bad program is the point. A reference **cycle** is still a `ValueError`: that is a bad program, not a missing name.

**One canonical `sev_ref` walker.** `_carries_sev_ref`, the boolean that refuses a reference written where it cannot be resolved, is now `bool(_collect_sev_refs(spec))` over one recursive walk that returns the dotted targets it found. The preview needs the targets, the refusal needs only whether there are any, and there is one implementation of "where can a `sev_ref` hide in a spec" (top level, a portfolio's unit dicts, a bivariate's `units` tuples).

**`_recipes_frame` snapshots the dict before iterating.** A fork can be registering a build on one thread while another reads the frame, and iterating the live dict would raise "dictionary changed size during iteration". One pass over a few hundred keys.

**Additive on the stable tier.** `Underwriter` gains two methods and the module gains three names, `RecipeNotFound`, `ProgramPreview` and `ResolvedReference`, all exported and on the `docs/3_reference/3_x_Underwriter.rst` autosummary. Nothing existing changed shape.

**Tests.** `tests/test_session_isolation.py`, 30 cases: the fork cost pinned two orders of magnitude above the measurement, isolation in both directions, the library clobber contained, the parser bound to the fork and parsing against the fork's own declarations, the cycle guard and lists reset, the parent loaded first, `interpret_file` leaving no residue; the preview on a self-contained program, an inlined reference, the same text reading `session` in one fork and the library file in another, the depth-two deferred chain, the bare-name route agreeing with `build`, registering nothing, a program's own declarations, an `expr` statement, the ordinary parse error with its `ErrorReport`, and eight concurrent previews equal to their serial baselines; `RecipeNotFound` as a `KeyError` and a `LookupError`, its attributes, its plain message, ambiguity staying a plain `KeyError`, the vanished deferred referent, and the cycle still a `ValueError`.

**`dev/TODO.md`, phase L4.** `[Session-Build-Clobbers-The-Trailer]` records that `fork()` supplies the "namespace the session base" arm of the author's 2026-08-17 ruling, and that it does not close the item: a single-process Jupyter user still overwrites the trailer in the one base they have. `[Unparser-Reference-Gaps]` item 1 records the author's 2026-08-18 ruling, declined: `sev.NAME` keeps inlining, reference semantics are spelled `agg Wrapper dfreq[1] sev sev.NAME` then `sev agg.Wrapper`, and the 8 exempt library entries stay exempt. The question was asked because a cache keyed on program identity would have depended on the answer; the design that shipped does not, because it qualifies on what the parse resolved rather than on how the writer renders.

## 1.0.0a301

**[Doc-Clause-Out-Of-The-Grammar] BREAKING: the DecL `doc{{{...}}}` trailer clause is removed.** Phases E and F of `dev/done/plan-decommission-docs.md`, and the reason the whole plan had a deadline. `decl.lark` is in the stable tier: from 1.0 a documented name keeps its meaning, and a breaking change waits for a major release with a deprecation period ahead of it. Keeping `doc` would have promised, for the life of the major version, a declaration language with a clause whose job is to hold a book. This was a pre-1.0 decision or a 2.0 decision, and the author took it now.

**A program carrying `doc{{{...}}}` no longer parses.** The trailer is three order-free clauses, `note{...}` / `tags{...}` / `hints{...}`, each stating a fact about the entry. The error names the three that remain. Nothing in the shipped library or in `decl-testers.agg` used the clause after a300, so no bundled program changed meaning.

**Removed, end to end.** `decl.lark` loses the `DOC` terminal and `trailer_item_doc`. `parser.py` loses `_DOC_FENCE_RE`, `_encode_doc`, `_decode_doc`, preprocess step 0, the `DOC` token callback and the `base64` import that existed only for them; step 0b keeps its name, since renumbering seven cross-referenced steps to close a gap buys nothing. `decl_writer.py` loses `doc` from `TRAILER_ITEMS` and its emit branch. `decl_pygments.py` loses `_doc_fence`, both `doc{{{` patterns and the deferred Markdown import. `parser_errors.py` loses the `"DOC"` label. `underwriter.py` loses `doc` from `TRAILER_KEYS` and the seven doc-derived columns from `recipes`, which is now `note`, `tags`, `source`, `program`, `spec`. `recipe.py` goes from 466 lines to 150: the record, its three trailer properties and `decl` survive; `parse_doc`, `SECTIONS`, `DECL_PLACEHOLDER`, the four section properties, `solution_code`, `check_code`, `markdown`, `namespace`, `n_asserts`, `is_runnable` and `run` are gone.

**BREAKING, stable tier: `Aggregate`, `Portfolio`, `Severity` and `BivariateAggregate` lose the `doc` constructor keyword and the `doc` attribute.** This was missing from the plan as drafted and was added by author ruling on review. Nothing can set the attribute once the clause is gone, so keeping it would have shipped four permanently empty public attributes on frozen classes. `doc` also leaves `FCC_REQUIRED` in `constants.py`, so the first-class-class contract is `info`, `help`, `note`, `hints`, `tags`, `program`, `pprogram`, the DataFrame quartet and `plot`.

**`Recipe.decl` is untouched**, which is the only part of this surface a downstream consumer reads.

**Tests.** `tests/test_doc_clause.py` deleted, 331 lines. `tests/test_recipe.py` rewritten around the record: identity, copy semantics, the derived trailer, `decl` (canonical, hints-only, `''` when the entry cannot be unparsed, cache dropped by `dataclasses.replace`), and the frame. `tests/test_grammar_sync.py` asserts three brace clauses and drops the doc-fence markdown lexing case; its Markdown-import guard stays, because the 110 ms cost is what matters and a future module-level import would still land on every user.

**Regenerated.** `docs/4_agg_language_reference/ref_include.rst`. The captured spec snapshot did **not** move: it is taken from `test_suite.agg`, which never carried a doc, so `tests/data/expected_specs.json` is byte-identical.

**Prose (phase F, riding this commit).** `docs/2_aggregate_overview/underwriter.rst` loses the trailer table's fourth row and the whole doc write-up including the two `ipython` blocks that called `r.sections` / `r.problem` / `r.solution_code` / `r.check_code` / `r.run()`, replaced by the `Recipe` record and `decl`. Also `docs/1_Getting_Started.rst`, `docs/3_reference/3_x_Underwriter.rst`, `cheat-sheets/Underwriter_Cheat_Sheet.tex`, `src/aggregate/config.py`, and `docs/2_aggregate_overview/features.rst`, where the a157 and a165 sections stand as history under a note recording the withdrawal, and the one executable block that read the retired columns is repaired.

**`[Recipe-Doc-Signing]` is moot** and marked so rather than deleted: it proposed signing a doc body so `Recipe.run` could refuse a tampered one, and there is now nothing to execute in a `.agg` file.

## 1.0.0a300

**[Doc-Bodies-Out-Of-The-Library] the seven long-form write-ups leave `library.agg`.** Phase D of `dev/plan-decommission-docs.md`. The file loses 311 lines and keeps every entry: `ThreeDice`, `PHDistortion`, `LayerPicks`, `LimitProfile`, `OccurrenceXOL`, `NeymanInnerOuter` and `SplitLimitPolicy` are unchanged as declarations and keep their `note{}`, `tags{}` and `hints{}`. Only the `doc{{{...}}}` bodies go. The material is not lost: it became notes in `aggregate-presentations` at phase A, and the invariants it asserted became `tests/test_library_entries.py` at a299, which is why this phase comes third rather than first.

**The header says what a trailer is for.** The `WRITING A doc` section, the `<<decl>>` convention, the plain-fence convention and the `HOW MUCH DOCUMENTATION` paragraph are replaced by the rule they were working around: a trailer holds facts about an entry, and a document is not a fact about an entry. A note that wants a library program prints it, `print(build.recipe('LayerPicks').decl)`, which reads the live library and cannot go stale. The six `[Check-*]` archetypes that arrived in the header at a298 stay.

**`tests/test_library_recipes.py` deleted**, 107 lines. It ran each doc's Solution and Check through `Recipe.run`; a299 replaced it with ordinary asserts that a traceback can point at.

**`tests/test_recipe.py` keeps testing the clause, against its own fixture.** Three tests read `<<decl>>` expansion out of `library.agg`, which no longer has a doc to read. They now build a synthetic `DocDemo` entry, following the pattern the neighboring hints test already used. This matters because **phase D is a stable resting place**: the grammar still accepts `doc`, the machinery still works and is still covered, and nothing in the shipped library uses it. Phase E is the breaking change and it is separate.

**`[Recipe-Run-Clobbers-The-Trailer]` becomes `[Session-Build-Clobbers-The-Trailer]`** in `dev/TODO.md`. Its trigger is gone with the recipe-run path, but the underlying behavior is not: a session `build` still silently replaces a library entry's trailer with whatever the session program carries. Author ruled to keep it open on the general terms.

## 1.0.0a299

**[Library-Entries-Build-Check] every shipped entry builds, and its stated invariants are ordinary pytest.** Phase C of `dev/plan-decommission-docs.md`, and it closes `[Agg-Library-Build-Check]` in `dev/TODO.md`. New `tests/test_library_entries.py`, 177 cases. Before this the shipped library had one assertion against it, that the file loaded; nothing in it was ever built.

**Every entry builds, as its own case.** All 168 entries, with `agg` and `port` additionally asserting they carry `valid`, `validation_explanation`, `summary_df` and `stats_df`. Six entries that cost more than a second are `slow` marked (`BivariateNormal` alone is twelve seconds); the fast set runs in about 20 seconds.

**Two baselines, each asserted in both directions.** `CANNOT_BUILD` names the six entries that deliberately refuse and the exception each raises: `DefectivePareto` (infinite variance, so no bucket size can be estimated), `ISOMixedExponential` and `MixedExponentialSev` (a mixture severity has no standalone meaning), and `BivariatePnLAxis`, `NumericsPayPair` and `PnLBook` (joint and book-level P&L are deferred). `VALIDATION_BASELINE` names the 26 that build but do not clear validation, pinned by exact flags and grouped by cause: heavy-tail reference curves whose higher moments no practical grid reproduces, the two failure demonstrations, `LayerPicks` where picking is the decision to leave the declared moments, the signed-window aliasing group, and the deliberately coarse or thick entries. Checking both directions matters: a list of known failures nobody checks for staleness stops being a baseline and becomes a comment, so an entry that starts **passing** fails the suite too, as does a name that no longer exists.

**The seven invariants, transcribed.** `ThreeDice` is exact rather than accurate, the PH distortion is a concave probability map, `LayerPicks` reproduces every pick with nothing moving above the tower, `LimitProfile` derives its claim count, `OccurrenceXOL` cedes without changing the count, `NeymanInnerOuter` matches the `neymana` keyword to `1e-13` including the atom at zero, and the 100/300 split limit prices its per accident limit at four thousandths of one percent. Unchanged in substance, one test function each.

**One correction.** The `OccurrenceXOL` frequency check passed vacuously as written: it compared `reins_summary_df['EX']` across views, but that column carries the theoretic **gross** value in all three, so it compared a number to itself. The real content is in `Est EX`, where the gross frequency is `NaN` (a gross count is an input, not an estimate) and ceded and net both come back as the declared count. The transcribed test asserts that form.

## 1.0.0a298

**[Cookbook-Removal] the Quarto cookbook leaves the repository.** Phase B of `dev/plan-decommission-docs.md`, the plan that retires the `doc{{{...}}}` clause and the cookbook together. The author's ruling: right idea, wrong place. Long-form write-ups belong in the presentations and monograph staging ground, not in a DecL trailer and not in a generated book inside the library.

**Deleted.** `docs/cookbook/` entire, source tree and build artifacts; `src/aggregate/cookbook.py`, the 336 line renderer; `dev/generate_cookbook.py`, its thin caller; `tests/test_cookbook_generate.py`. Nothing imported `aggregate.cookbook` except those two, so this phase is self contained. `docs/conf.py` loses the `cookbook/plan.md` exclude pattern and `docs/3_reference/3_x_Underwriter.rst` loses its Cookbook section and `automodule` directive.

**Kept, in the place that defines it.** The six `[Check-*]` archetypes are the `check:` tag namespace in `library.agg`, so their definitions move from `docs/cookbook/plan.md` into the `library.agg` header beside the vocabulary they describe: reconciliation, scaling-sweep, independent-oracle, limiting-case, round-trip, cross-object, each with the example it was defined by. The tag pool is unchanged; only its documentation moved.

**Where the material went.** The seven documented entries became notes in `aggregate-presentations`, written before anything was deleted: `ph-distortion`, `three-dice` (widened to exact discrete aggregates), `limit-profile`, `occurrence-xol` and `split-limit` as new pages, with `LayerPicks` folded into the existing `picks-mix-exp` and `NeymanInnerOuter` into `compound-frequencies`, since both notes already covered that ground at greater length. Every one renders clean.

**Not in this phase.** The `doc{{{...}}}` clause still parses and the seven doc bodies are still in `library.agg`; phases C, D and E take the tests, the bodies and the grammar in that order, so coverage never dips and the shipped library never stops parsing.

## 1.0.0a297

**[Library-Tidy-Round-One] three new recipes, and the shipped library loses 28 duplicate programs and gains a reading order.** `library.agg` goes from 196 statements to 168. Nothing here changes library code: the entries, their notes and their filing are the deliverable, and the one source edit is a docstring typo.

**The ISO mixed exponential, and a layer picks recipe.** `sev ISOMixedExponential` is the five component commercial auto curve on the 2008 cost level, untrended, mean 13,990, with the Actuarial Review citation for the parameters in its note. `agg LayerPicks` is the cookbook version of it: a four layer tower to 1M, selected losses of 6,700 / 1,700 / 1,600 / 1,000 against curve losses of 7,495 / 1,462 / 1,310 / 1,127, and a generated exhibit of curve, pick, adjustment and achieved. Every pick is reproduced to floating point, the adjusted mean is the picks plus the untouched loss above the tower, and above 1M the survival function does not move. The Discussion states the two things the clause will not tell you: the first bracket holds layer **tops**, not widths, and **a picked aggregate reports a failed validation by design**, since validation compares the estimate against the analytic moments of the declared severity and picking is the decision to leave them.

**A Neyman A built by hand.** `agg NeymanCluster` is one cluster, a Poisson(4) count on a unit severity; `agg NeymanInnerOuter` draws a Poisson(2.5) number of them through `sev agg.NeymanCluster`. The recipe builds `10 claims dsev [1] neymana 4` beside it and asserts the two probability mass functions agree to `1e-13` across the support, atom at zero included. `agg NeymanAFrequency` joins the frequency section, which had no Neyman entry. The point the doc makes is that a two level count is a stopped sum, so the construction reaches any of them and the keyword exists only because the two level Poisson case earns a closed form pgf.

**The split limit, which is what `[Agg-As-Severity]` was built for.** `agg SplitLimitClaimant` is one US auto 100/300 accident: a zero truncated Poisson number of claimants held at 1.25 by the frequency `!`, each capped at 100. `agg SplitLimitPolicy` writes 100 such policies with the 300 per accident limit as an ordinary occurrence limit on the reference. The recipe reports the actuarial result rather than asserting it: an accident needs three claimants before the 300 can bind at all, `P(policy loss > 300)` is about 7 in 100,000, and the 300 is worth 0.0019 of expected loss against a per policy mean of 43.92. In a 100/300 policy the per claimant limit does essentially all the work.

Two practical findings came out of writing it, both now in the doc. The inner and the outer pin the **same** `bs`: a layered reference is rebucketed by differencing its survival function, exact when the grids agree and a few parts in ten thousand out when they do not. And the reference carries a severity side `!`, because at any usable resolution a severity with positive density at the origin discretizes a wisp of every claim into the first bucket, about 1.6 in 10,000 here; those are small losses, not zero losses, and the default conditional reading would drop them and rescale.

**Twenty eight duplicate programs deleted**, each byte identical in spec to the entry that survives it, found mechanically rather than by eye. Survivor in parentheses: `ApproximateRightSgamma` (`ApproximateGamma`), `BasicXOL` (`OccurrenceXOL`, which carries the doc), `BinomialSimple` (`BinomialFrequency`; its note said n = 10 for a Binomial(20, 1/2)), `BivariateGumbelPair` and `CopulaWindFlood` (`BivariateGumbel`, which inherits `role:hero`), `CededLayer` (`ReinsuranceLayered`; a singular name over a four layer tower), `DualDistortionSimple` (`DualDistortion`), `ExposureByPremium` (`ExposureByLoss`; it was declared `500 loss`), `MixedSeverityAndFrequency` (`LimitProfileSimple`), `NetCededJoint` (`BivariateNetCededPair`), `NetCededShowcase` (`BivariateNetCeded`), `NumericsDice` (`DiceOfDice`, which inherits `topic:numerics`), `PnLMixed` (`PnLVector`), `SevAttached` (`SevLayered`), `SevBlend` (`MixedSeverityProfile`), `SevExponential` (`CurvePareto`), `SevGammaCV` (`CurveInverseGamma`), `SevPareto` (`CurveWeibull`), `SevVersusSsev` (`SignedPortfolioPair`), `SevWeibull` (`CurveInverseGaussian`), `SignedFixed` and `SignedSeveritySimple` (`SignedSeverity`), `ThreeDiceSum` (`ThreeDice`), `TweedieCompound` and `TweedieSimple` (`TweedieDispersion`), `WindowDiscrete` (`WindowedGrid`), `ZMPoissonSimple` (`ZMPoissonFrequency`), `ZTPoissonSimple` (`ZTPoissonFrequency`).

Four of those are worth calling out because the name, not the program, was the defect. `SevExponential` declared a shifted Pareto, `SevPareto` a Weibull, `SevWeibull` an inverse Gaussian and `SevGammaCV` an inverse gamma with no cv in sight, a shift by one down the block. Each spelling they demonstrate is already in the severity clause forms and each family is already in the curve reference, so they went rather than being renamed. `SevAttached` attached at zero, where the severity side `!` is a no op on a positive severity: verified identical to `SevLayered` bucket for bucket.

**Four entries renamed**, each because the name contradicted the program: `BasicPoissonSev` to `BasicDfreqWeighted` (there is no Poisson in it), `DiceThreeDice` to `DiceFiveDice` (`dfreq [5]`), `ExposureByLossVector` to `ExposureByPremium` (nothing is a vector; it is premium and a loss ratio), and `SevUnconditional` to `SevLayeredConditional`, which carries no `!` and whose own note already said conditional.

**Two programs were wrong and are fixed rather than deleted.** `WangDistortion` was declared `ph 0.7`, a second copy of `PHDistortion` under a different measure's name, and is now `wang 0.3`, so the library has a Wang transform for the first time. `BodoffThreeNet` was a byte for byte copy of `BodoffThree`, and now spells its severity `dsev` like `BodoffOneNet` and `BodoffTwoNet` do.

Smaller repairs: `TweedieDispersion`'s note read `variance = dispersion xs mean**p`; `BasicDfreq`'s claimed range notation the statement does not contain, since the writer expands `[1:3]`; `NumericsUnbounded`'s carried a bare `(G6)`; `GrossCatXOL` was tagged `topic:severity` and is a gross aggregate; `InverseGaussianMixed` was tagged `topic:aggregate` and is a frequency mixture. `SignedGotcha` and `SignedPortfolioPair` gained notes naming them as the two halves of the `sev` against `ssev` contrast, which is the reading the deleted `SevVersusSsev` was gesturing at.

**The file is re-sectioned into teaching order**: starter examples, distortions, severity clause forms, the severity curve reference, frequency, aggregates, aggregates with reinsurance, one aggregate as the severity of another, bivariate, profit and loss, numerics, portfolios, papers. Every severity entry is now contiguous, where the curve reference used to sit at the far end of the file with the `Curve*` families interleaved alphabetically among the citekeys. **In three places the order is load bearing**, because a builtin reference resolves by sequential load: `sev.UnitSeverity` leads the severity section ahead of everything that uses it, `dist.PHDistortion` and `dist.DualDistortion` lead the distortions ahead of `MinimumDistortion`, and `agg.USXOLTower` precedes the `xpnl` that wraps it, as each inner precedes its outer. The header says so.

**`sev agg.NAME` leaves the hand written set.** `dev/done/reflow_library.py` was holding back every statement containing a dotted reference, which since a293 is too broad: the writer renders a severity reference, because the spec records the reference rather than inlining the object. The script now strips `s?sev <agg|port>.NAME` before testing, so the two new outer entries are laid out by machine and the held back set is 13, exactly `UNPARSER_EXEMPT`.

**The cookbook gains two pages**, `_recipes_02_severity.qmd` and `_recipes_03_frequency.qmd`, and `cookbook.qmd` includes them. Seven documented recipes over five pages, up from four over three.

`Frequency`'s class docstring listed the Neyman A frequency as `negymana`, which is not a keyword. It reads `neymana` now.

Two findings that are not fixed here are recorded in `dev/TODO.md` with reproductions: `[Aliasing-Test-Misfires-On-A-Reference-Severity]`, where an exact reference severity collapses the denominator of the validator's aliasing ratio and a model accurate to `1.5e-11` is reported as aliasing, and `[Recipe-Run-Clobbers-The-Trailer]`, where a recipe's `<<decl>>` Solution re-registers its own entry without `note{}` or `tags{}` in the shared session recipe base. `[Library-Round-Two]` collects the nine judgment calls this pass deliberately left, the Bodoff family's fourfold redundancy at the top of the list.

## 1.0.0a296

**[Moment-Store-Default-Reading] the one exhibit that cannot name its columns states a reading instead.** `stats` gains a scoped catch-all, `exhibits: {stats: {patterns: {'.*': ',.7g'}}}`, so every column of the moment store with no entry of its own reads the same way. It is the case the a295 mechanism was built for: those columns are computation views (`mixed`, `empirical`, `after_occ`) and **unit names**, which are the user's words rather than the library's, so no list of exact entries can cover them and inference was answering column by column, differing between books and reaching for SI suffixes on the wide ones. `error` keeps `.5g`, because an exact entry still beats a pattern.

**The mixture components are repeated inside the scoped section**, above the catch-all. A scoped pattern is tried before a global one, so without that line `'.*'` would shadow the global `e[0-9]+\.m[0-9]+` rule inside this exhibit. Worth knowing generally: scoping a broad pattern to an exhibit hides the narrow global ones there.

The moment store now reads `2e+10`, `1.25e+17` and `6.092922e+10` where it read `20.000G`, `125.000P` and `60.929G`, uniformly across views, units and the analytic components. Exhibit snapshots regenerated. Two format sheet tests now assert the shape rather than the literal reading, since which format a family wears is the author's to edit in the sheet while the mechanism is what the tests are for.

## 1.0.0a295

**[Format-Sheet-Patterns] a format sheet can say how a family reads.** A new `patterns:` section keys on a regular expression instead of a label, matched **whole** against the displayed column label, tried in the order written with the first match winning. It answers the labels no list can enumerate: the moment store's analytic mixture components are `e0.m0`, `e0.m1`, `e0.m2`, one per component, so how many exist is a property of the program rather than of the vocabulary. The shipped sheet declares them `si`, which is a reading they used to get from inference on a wide book and not on a narrow one.

**greater_tables is untouched, and could not do this anyway.** Its format lookup is by exact label. The expansion happens in the loader, against the block's own columns at the moment the `formatters` mapping is built, so GT still only ever receives words it will match. Nothing moves in the IR, no schema version changes, and no consumer needs to learn anything.

**Precedence, high to low**: a scoped exact entry, a global exact entry, a scoped pattern, a global pattern, then GT's inference. Exact beats pattern because a pattern is a rule about a family while an entry is a statement about one word, and the word is the more specific of the two. A pattern also carries a style, so `'.* CV': ratio` stamps the ratio tag on everything it matches.

**BREAKING, in the provisional sense, and small: the scoped `exhibits:` section is now structured.** It carries `columns:` and `patterns:` rather than column entries directly, so a scoped section is a sheet in miniature and learning one teaches the other:

```yaml
exhibits:
  tail:
    columns:
      P: probability
```

The old flat shape raises at load naming the file, the exhibit and the fix. Only the shipped sheet used it.

**A scoped `'.*'` is a per-exhibit default**, which is worth stating because it was the one thing the sheets could not express before and the reason a `float_format` section looked necessary: `exhibits: {stats: {patterns: {'.*': 'si'}}}` says everything in that exhibit with no entry of its own reads in SI, exact entries still winning. Not shipped, since `stats_df`'s remaining columns are unit names and the author's call, but available.

**Sheet entries added, and the readings that move with them.** The round one gap the a288 sweep reported is closed for `summary` and `validation`: `Mean`, `Median`, `SD`, `EX`, `Est EX`, `P01` and `P99` read as money; `Err`, `Err EX`, `Err CV`, `abs_err`, `rel_err` and `Gate` read `.5g` like `error`; `Est CV` joins `CV` as a ratio and `Sk` / `Est Sk` join `Skew`. Those columns were reaching for engineering notation on any realistic book, since a relative error spans more than the `1e6` ratio that trips GT's inference and a loss column sits outside its `[1e-3, 1e6]` magnitude window. Sixteen labels leave inference, twelve leave `PENDING_VOCABULARY`, and the committed exhibit snapshots move with them.

**Not done, deliberately**: bounding the SI ladder to a window (suffixes from `m` to `T`, exponent form outside) would need new `FormatSpec` fields in greater_tables plus matching work in its JS bridge and in csv-grid, whose ladder runs `n` to `T` and clamps rather than falling back. The author's ruling is to keep the whole ladder and learn the prefixes.

## 1.0.0a294

**[Reference-Severity-Reports-Its-Source] a reference to an unbounded aggregate reports unbounded, everywhere.** Author ruling, 2026-08-16, closing the one question `dev/done/plan-agg-port-as-sev.md` left open. Section 4.6 pinned the tail descriptor's consumers at `tail_behavior_df`, which left `bounded` reading the materialized atoms: `agg X dfreq [2] sev agg.Unbounded` showed an infinite max in the frame and `X.bounded == True` beside it. `_severity_bounded`, `classify_severity`, `_combine_severities` and `aggregate_tail_info` now take the same `reference=` flag `build_tail_rows` already had, and every reporting surface passes it: `Aggregate.bounded`, `Aggregate.tail_class`, `Severity.bounded`, `Severity.tail_class`, `Severity.tail_description` / `tail_explanation`, and the `info` blocks that read them.

**With it goes the `p = 1` pricing guard**, which is the point of having the two agree. `_pricing` refuses `p = 1` on an unbounded law because it resolves to the top of a grid that moves with `log2` rather than with the risk; a reference whose source is unbounded is exactly that case, and it used to slip through on the finiteness of its atoms.

**The bucket sizer keeps the other reading, unchanged.** `Aggregate._bounded_severity_window` now calls `tail._severity_bounded` directly, without the flag, because the grid is chosen for the atoms actually convolved. That is the plan's ruling 17: the source's own theoretical tail already drove the *source's* window choice, the certified `dsev` encodes it, and the consuming aggregate sizes from it as it would from a hand-written one. So `bounded_small` still applies to a reference severity and no grid moves.

**Both routes into a reference now report identically.** `SeverityMeta` (the `Severity(agg)` / `as_severity` path) stamps the same `reference_id` / `reference_support_max` / `reference_bs` the DecL resolver stamps, so the programmatic and declarative halves answer the same question the same way, and the special-case `meta` branch in `_severity_bounded` folds into the histogram one. A `copy` severity keeps its own branch.

**Also: the conditioning notice drops its recommendation.** Author ruling, same date: the incidental mass at a materialized reference's zero atom "is what it is", there is no reason to reach for `!`, and **the default conditioning behavior is unchanged** for a reference exactly as for a hand-written `dsev`. The a291 warning said the user had "almost certainly" meant `!`, which overstepped. It now reports the number and names the option without recommending either, and the docs and the corpus write the split-limit book in its plain conditioning form.

## 1.0.0a293

**[Agg-As-Severity-Port-Units] a portfolio unit can take its severity from a reference.** Phase C, the last of `dev/done/plan-agg-port-as-sev.md`, and the smallest: the `port` branch of `_factory` resolves any unit spec carrying `sev_ref` before `Portfolio` builds its `Aggregate`s, because the unit specs go straight into `Aggregate(**unit)`, which cannot see the key. The provenance stamp is applied per unit afterwards, where the `Aggregate` objects exist.

```
port Book
    agg Auto  500 claims sev agg.SplitPolicy poisson
    agg Prop   10 claims sev lognorm 50 cv 1 poisson
```

Everything else follows from phase B unchanged, because it is the same resolver: the hygiene rule applies per unit, the cycle guard spans units, the tail descriptor is transitive through a unit, and `format_program` renders the unit back as a reference. A unit carrying no reference is untouched.

The plan's section 4.7 note about more than one lattice reaching a sizing becomes reachable here in principle, since two units may reference sources on different grids. It does not fire in practice: each unit is sized on its own, so each sees one `d`. The multiplicity warning stays as the guard it was written to be.

A corpus line in `_test_suite.agg` section Z; `expected_specs.json` recaptured, additions only. Four new tests.

## 1.0.0a292

**[Agg-As-Severity-Commensurable-Grids] the outer grid learns about the reference's lattice.** Phase B2 of `dev/done/plan-agg-port-as-sev.md`, in its own bump so the `_bucket_window.py` diff stays readable. A severity built from a `sev agg.NAME` reference carries the referenced object's own bucket size `d` on `reference_bs`, set exactly on the resolution path and never inferred from `np.diff` of the atoms. `snap_bs_to_reference` asks one question at the end of bucket selection: is the chosen grid commensurable with `d`? If not, and the bucket was estimated rather than pinned, it snaps to `d * 2**m` and the winning row is re-derived through `_size`, so origin, span and the log2 need all still come from the one kernel. Nothing else in the sizer changes: no new candidate row, no change to `_size`, no change to the priority logic.

**The common case costs one generator expression and does nothing**, which is the design. An aggregate with no reference severity exposes no `d` and returns immediately. More than that: **a dyadic `d` is already commensurable with every bucket the estimator can pick**, since `round_bucket` returns a power of two below 1 and an integer at or above it, so the usual `hints{bs=1/32}` reaches the snap and finds nothing to do. What actually triggers it is a lattice that is not a power of two, `hints{bs=1/3}`, where an estimate of `1/512` snaps up to `1/3` itself: the aggregate of a lattice-valued severity lives on that lattice, so the coarser grid is the *exact* one, which is the same argument `_severity_lattice` already makes for integer atoms, generalized to a non-integer step.

**The decision table**, in order, with integer tests at a relative tolerance of `1e-9`: no `d`, unchanged; `d / b0` an integer (the outer grid is finer, so inner atoms land on grid points), unchanged; `b0 / d` an integer, unchanged, and deliberately not forced to a power of two, since the estimator or the user landed on an exact multiple and there is nothing to fix; otherwise auto-sized, snap to `d * 2**m` with `m = max(0, ceil(log2(b0 / d)))`, never below `d`; otherwise pinned, honored as written with a warning naming both values and the nearest commensurable bucket on each side. No back doors around an explicit `bs`.

**Reported, not silent.** The winning row's `note` in `bs_window_df` gains the before and after, and `bs_explanation` gains a sentence naming the source, its lattice, the two bucket sizes and the reason. The structured record is `Aggregate._bs_snap`, the sibling of `_bs_clip`, cleared at the top of every sizing.

The v1 grammar admits at most one reference per aggregate, so at most one `d` reaches a sizing; the function nonetheless takes a set, snaps to the coarsest and warns if it ever sees more, which is future-proofing for portfolio unit sizing. Origins are multiples of the outer `bs`, hence of `d`, so inner atoms stay on the `d` sublattice, and a signed reference is covered by the existing origin flooring. Eight new tests in `tests/test_agg_as_severity.py`.

## 1.0.0a291

**[Agg-As-Severity] an aggregate can be a severity, in DecL.** Phase B of `dev/done/plan-agg-port-as-sev.md`. `sev agg.NAME` and `sev port.NAME`, with the `ssev` and unconditional `!` forms and an optional `as` label, let a declaration already in the recipe base serve as the severity of another aggregate. The motivating shape is a US personal auto split limit, a 100/300 policy:

```
agg SplitPolicy 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson zt ! hints{log2=16; bs=1/32}
agg SplitBook   5000 claims 300 xs 0 sev agg.SplitPolicy ! mixed gamma .2
```

**No new terminals and no new keywords.** `BUILTIN_AGG` and `BUILTIN_PORT` already existed, the `ID` lookahead already excluded both prefixes, and all four dotted terminals were already in `_TERMINAL_LABELS`, the Pygments lexer and the sublime syntax. So the grammar gains two `sev_clause` alternatives and one leaf rule and nothing else, `test_grammar_ambiguity`'s `KNOWN_AMBIGUOUS` is unchanged, and **nothing is owed to the app side `decl-keywords.json` mirror**.

**DecL's first deferred reference.** Every other dotted name is resolved and inlined by the parser (`_safe_lookup` returns a deepcopy of the stored spec). This one cannot be: the severity is the referenced object's *computed output*, `xs` and `agg_density`, which exists only after an update. So the parse records a symbolic `sev_ref` key holding the verbatim dotted id, and `Underwriter._resolve_sev_ref` resolves it at the top of `_factory`, before `Aggregate.__init__` builds its `Severity` components. Two things follow that are worth stating plainly. The inner may exist **only as a recipe** and never have been built, which is the normal case for a library entry, so this is a named reference rather than object sharing. And it is not textual injection: the inner is fully parsed on its own, which is exactly why the language allows calling an inner by name and never inline, and why a doubled `!` cannot arise.

`sev_ref` is also the first spec key that **round-trips as a reference**: `decl_writer` renders `sev agg.SL`, `ssev agg.SL` and the trailing `!` back from it. That is the pattern `[Unparser-Reference-Gaps]` in `dev/TODO.md` asks for, now established; the note there records what `sev.NAME` would need to join it.

**A reference is a certified, fully formed `dsev`.** The inner's output pmf becomes a discrete severity exactly as though the user had transcribed `dsev [xs] [ps]`: exact discrete moments (reported as the outer's theoretical severity moments, because that is what the object outputs), exact `layer_moments`, `support_atoms` for lattice detection, and the mean-preserving linear rebucket onto the outer grid. The query is nullary, so nothing flows in from the outer, `normalize` included. Materialization goes through the same `_dhistogram_from_object` the a290 programmatic path uses, so the declarative and programmatic halves cannot drift.

**Referenced inners must pin their own grid** (author ruling). A reference to a declaration carrying no `hints{log2=…; bs=…}` raises, because the severity *is* whatever the inner computes and the inner has to say at what resolution or it follows ambient defaults instead of the model. The error runs the inner's own bucket sizer (spec-only, no FFT) so the suggestion can be pasted rather than guessed at, and names the new **`with_hints`** on `Aggregate` and `Portfolio`: it returns the object's program with its realized `log2` / `bs` / `normalize` merged into the existing `hints{}` clause, so `build(inner.with_hints())` re-registers a candidate inner as a certified one. The merge is key by key, so a declared `padding` survives.

**Every build re-resolves.** There is no built-object cache in the recipe base, so the inner is rebuilt from its recipe each time. With the hints mandatory that is exactly reproducible, correct under last-write-wins redefinition, and holds no hidden state. Depth is unlimited (resolution recurses through `_factory`), guarded by `Underwriter._sev_ref_stack`, which names the chain: `severity reference cycle: agg.A -> agg.B -> agg.A`. Caching is recorded as `[Agg-As-Severity-Result-Cache]`, to be taken up only if profiling ever hurts.

**Signed sources follow the `sev` / `ssev` matrix.** `ssev` over a signed inner keeps the negative atoms; plain `sev` over one clamps them onto the zero atom and warns, since that combination usually means a P&L-shaped source is being read in a loss context. `ssev` over a nonnegative inner is silent.

**Tail reporting is the one place a reference is not read as a `dsev`, deliberately.** A materialized reference is a finite atom set and always looks bounded; the object behind it may not be. `tail_behavior_df` (and the narrative pair) now report the *source's theoretical* tail, derived from its count support, its components' structural support and its reinsurance structure via the new `tail.output_support_max`, with the aggregate row's note naming the source and saying it was sized on the atoms. **Numerics are untouched**: `_loss_tail_classes` asks `_tail_rows(reference=False)`, so the bucket sizer sees the atoms it actually convolves. That split is the design, not an oversight: the inner's theoretical tail already drove the *inner's* window choice, and the certified `dsev` encodes it, so the outer legitimately sizes from it as bounded. Descriptor precedence is absolute, which is what makes a depth 2 chain report unbounded rather than stopping at the first materialized link. `Aggregate.bounded` and `Severity.bounded` are unchanged (they read the family classifier, which the sizer also reads); the plan pins the descriptor's consumers at the tail report.

**Fixed while executing the plan: `hints{}` leaked between statements of one program.** `build_many` rebound `log2` / `bs` / `bucket_sizing_p` / `kwargs` in place as it walked its outputs, so `_resolve_hints` on statement 1 became the caller default for statement 2. A program whose first statement carried `hints{log2=16; bs=1/32}` built **every later statement on that grid**: a 5,000-claim book that should have sized to `bs=2` came out at `bs=1/32` with half its mass missing. The loop now holds the caller's arguments fixed and copies `kwargs` per output. This is a pre-existing bug of its own, and it sits directly in this feature's path, because the hygiene rule guarantees a referenced inner carries hints and the single-program define-then-use flow is the headline case.

**Also found: the plan's headline program needed a `!`, and now says so.** A layers clause conditions on exceeding the attachment by default, and the plan reasoned that a zero-truncated inner has no zero atom so none was needed. That holds for the theoretical law and not for the materialized one: any severity with positive density at the origin discretizes mass into the first bucket, and `gamma 50 cv 2` is shape 0.25, so about 10% of a single claim lands below `bs / 2` and the zero-truncated per-policy aggregate materializes with about 7% at its zero atom. Conditioning rescales that away and lifts the severity mean by the same proportion, moving the split-limit answer by 6%. The behavior is exactly `dsev`'s and is unchanged; what is new is that the resolver **warns**, and only where the reading cannot have been intended, namely when the source's claim count is never zero so the mass is certainly discretization. A plain Poisson inner really does have `P(S = 0) = e**-lambda` and conditioning it away is the documented alternative model, so that case stays silent. Whether the *default* should flip for a reference is a language-semantics question left to the author, recorded as `[Reference-Severity-Zero-Atom-Default]`.

**Also fixed: `interpret_file` could not validate a file that declares a name and then uses it.** It parsed statement by statement without registering anything, so any internal dotted reference raised `KeyError` out of `_safe_lookup` and aborted the whole run, which is precisely the outcome a method that exists to *collect* per-statement errors must not have. It now parses against a scratch underwriter seeded from this one and registers each statement as it goes, so definition-before-use resolves; an unresolvable reference is reported as an ordinary error row (`LookupError` joins the caught set) rather than ending the run. As a side benefit, validating a file no longer adds its contents, or the hand-seeded `sev One`, to the calling underwriter's recipe base.

**Refused rather than silently missed:** a reference inside a `bivariate` or `clash` body. Those parse the shared `sev_clause`, so one reaches the underwriter syntactically; the joint grid is a different problem, and a silent miss would build a bivariate whose severity spec is a bare string. Also refused: `2 * agg.Ref` and `agg.Ref + 10`, whose algebra rewrites `sev_mean` / `sev_scale` / `sev_loc`, none of which a reference has. `@` is frequency only and stays legal.

New tests in `tests/test_agg_as_severity.py`, 45 cases: exactness against a hand-rolled self-convolution, the three split-limit representations and the zero-truncation identity, hygiene and `with_hints`, depth 2, cycles and self-reference, the signed matrix, seven tail-descriptor cases including the transitivity pin and the sizer isolation, iterated builds, the round trip, and smoke cases for `approximate`, `create_frequency` and a P&L engine. Corpus lines in `_test_suite.agg` (new section Z) and `decl-testers.agg` (new section ASV); `expected_specs.json` recaptured, additions only. `test_decl_unparser` now preloads the whole three-file corpus rather than `_test_suite.agg` alone, so a reference defined in one corpus file resolves from another. Grammar reference regenerated; `dev/FEATURES.csv` gains `with_hints` and the `Aggregate.as_severity` cell.

## 1.0.0a290

**[SeverityMeta-Afresh] a compound distribution can be a severity again, and this time it is a dsev.** Phase A of `dev/done/plan-agg-port-as-sev.md`. `SeverityMeta`, the class behind `Severity(some_aggregate)` and `Portfolio.as_severity()`, is rebuilt on the `SeverityDHistogram` foundation instead of the pre-1.0 `rv_histogram` hybrid. It is now a `SeverityDHistogram` subclass: the source's output pmf becomes the atoms of a discrete severity, exactly as though the user had transcribed `dsev [xs] [ps]` by hand. Exact first three moments summed off the atoms, `support_atoms` for lattice detection, an honest `_DiscreteRV` with exact `layer_moments`, and the mean-preserving linear rebucket onto the consuming grid. The old construction pinned the mass at zero in a `bs * 1e-7` sliver and read the rest as continuous-uniform, so it had none of those.

**It was broken in practice, not merely dated.** `Portfolio.as_severity()` raised `IndexError: string index out of range` from inside scipy, before any severity logic ran. `Severity.__init__` passed `name=''` for every object-valued `sev_name`, and scipy indexes `name[0]` to pick the article for its generated docstring. The empty string is now the class name, which is non-empty by construction. The `meta` and `copy` paths both went through that line, so both were unreachable. **This changes `Portfolio.as_severity` numerics** (hybrid histogram to exact atoms); no test covered the old behavior, and it could not be called.

**The query is nullary, and the side effect is gone.** The source is asked what distribution it outputs and answers from its current computed state: nothing is passed in, nothing is recomputed, and the source is not touched. The old code repurposed the `sev_a` / `sev_b` spec slots as `log2` / `bs` and **updated the source in place** to match, through a call to `easy_update`, a routing alias removed long enough ago that the branch could only ever have raised `AttributeError`. That side effect was flagged for review at the refactor; this is the review outcome. `sev_a` / `sev_b` are accepted when they merely restate the source's current grid, which is the shape `Portfolio.as_severity` used to call with, and raise when they contradict it: update the source at the resolution you want, then convert. The stale "aliased as easy_update" line in `Aggregate.update`'s docstring goes with it.

**`Aggregate.as_severity` is the new twin.** `as_severity` existed only on `Portfolio`, so the more natural case, a per-policy aggregate serving as the severity of a book of policies, had no method. Both now carry it with the same `(limit, attachment, conditional)` signature. That shape is a US personal auto split limit: 100 per claimant on the inner, 300 per policy on the outer.

**One shared materialization.** `aggregate._severity._dhistogram_from_object` is the single implementation of "read this object's output law as atoms", built here and consumed by phase B's DecL route so the declarative and programmatic halves cannot drift. It clips FFT fuzz below zero (`validate_discrete_distribution` makes the support distinct and ascending but says nothing about probabilities, so the clip happens here or not at all), drops the zero-probability buckets, which typically takes 65,536 grid points down to a few thousand atoms, and honors the source's own `normalize`. A residual deficit above `1e-6` warns as a `DefectiveDistributionWarning` naming the remedy, a larger `hints{log2=...}` on the source. That floor is deliberately tighter than the house `DEFICIT_MATERIALITY` of `1e-4`: this is a *severity* deficit and the outer frequency multiplies it, so a per-claim `1e-6` over 5,000 claims costs the outer aggregate half a percent.

**Signed sources follow the `sev` / `ssev` rule.** Under `ssev` the negative atoms pass through and the severity auto-signs through the existing negative-atom rule. Under plain `sev` the negative mass moves onto the zero atom and warns, naming the mass and pointing at `ssev`: the same clamp-at-0 convention that separates `sev 100 - lognorm` from `ssev 100 - lognorm`, and usually a sign that a P&L-shaped source is being read in a loss context.

New tests in `tests/test_reference_severity.py`, 14 cases, including the author's `Portfolio.as_severity()` repro. The `meta.` DecL keyword is **not** reinstated and never will be; the declarative surface is `sev agg.NAME`, landing in phase B.

## 1.0.0a289

**[Windowed-Grid-Breaks-Calibration] the pricing calibration learns about the window.** `_pricing.py` was written against a zero-based grid and never told about `_bucket_window`: the word `x_min` did not appear in the module. On any build whose grid windows, so that the first bucket sits at `x0 > 0` rather than at the origin, `calibrate_distortions` returned wrong distortion shapes, and the error was always exactly the window offset. Fixed in the classic branch. **This changes numbers on every windowed calibration**, which is the point: they were wrong.

**The layer integral needs the origin, and the window does not have it.** `∫₀^a g(S(x)) dx` runs over `[0, a]`, and on a windowed grid the region `[0, x0)` is simply absent. `S == 1` there, so `g(S) == 1` for every distortion, and the missing rectangle has area exactly `x0`. `_limited_ev` lost the same rectangle from the expected loss. Both functions were internally consistent and both said "0-based" in their docstrings; what was wrong is that `a` and `P` were out of frame while `exa` and `S` were in frame, and the cost of capital inversion `P = ν·exa + δ·a` mixed them. A frame mix, not a missing term, which is why patching only `exa` made it worse.

**Two failure modes, and the quiet one was the `Aggregate`.** An `Aggregate` lost both legs and they partly masked each other: the premium target was computed from the understated expected loss and matched against an integral understated by the same rectangle, so the families converged on plausible-looking numbers with small reported errors and nothing warned. On the reported program the expected loss came back 10,732.41 against a true 24,491.92, short by 13,759.50, which is `x_min` to the last digit. A `Portfolio` lost only the second leg, since `add_exa`'s cached `exa_total` is window aware, so the target was right and the integral was capped: a negative PH index, infinite Wang and Dual parameters, and a TVaR at a probability of 220.

**Slide the window, do not pad it.** A spectral risk measure is translation equivariant, so calibrating `X' = x_min + X` to a premium `P` is calibrating `X` to `P - x_min`. The density slides onto the 0-based frame, the target goes in as `P - x0`, and the receipt comes back out. Padding `S` with `x0 / bs` leading ones would give the same answer, but that count is unbounded (a `Po(1e9)` at `bs=1` windows a billion buckets off the origin) where the shift is O(1). This is the transform the signed and payoff branch has always applied with a positive `c`; the classic branch never reached it because `transform` triggers on `index.min() < 0` and the shift is clamped by `c = max(0, -min(support))`. Widening that trigger instead was considered and rejected: it would start refusing legitimate windowed reinsured aggregates at the `reins_view` guard, key the shift off `VALIDATION_NOISE`-trimmed support rather than the exact `x_min`, and lose the `kind` argument, which is honored only on the classic path.

**`lr=` resolves out of frame here, in frame on the signed path, deliberately.** A loss ratio is scale free but not shift free, so the frame is a choice. A canonical shift stands for genuinely negative outcomes and the in-frame premium is the one that means anything; a window is an artifact of the grid, so a reader who writes `lr=0.7` means the loss ratio on the premium they are shown. Both docstrings now say so. A consequence worth knowing is that the admissible range differs by frame: `_coc_from_lr` refuses when `P >= a`, which out of frame is `lr > L/a`, both ends being numbers the reader is shown.

**`M`, `Q` and the cost of capital never moved.** They are shift invariant, so only `L`, `P` and `a` slide. A grid already at the origin takes `x0 = 0`, where the density is not copied and the arithmetic is an exact no-op, so the legacy path is byte-for-byte unchanged.

**Docstrings now state 0-based as a contract on the caller** rather than as an assumption about the grid, in `_limited_ev` and `_calibration_survival`. Stating it the other way is what let this sit unnoticed.

New tests in `tests/test_windowed_calibrate.py`, 11 cases. There was no test asserting a windowed calibration before, which is how an infinite Wang parameter survived. Nothing is owed on the app side: `aggregate_api` calls `calibrate_distortions` and renders what it is served, so its Calibrate, Stand-alone, Allocate and Evaluate leaves were reporting these numbers faithfully and are correct once it syncs. Written up in `dev/done/plan-windowed-grid-calibration.md`. One related symptom was diagnosed and deliberately left unfixed, `cdf` returning `nan` below the window where the answer is 0: same root, different site, much wider blast radius, tracked as `[Windowed-Cdf-Nan]` in `dev/TODO.md` pending a ruling on what the *upper* fill should be.

## 1.0.0a288

**[Format-Sheet-Enforcement] the vocabulary is swept.** The third and last phase of `[Format-Sheets]`. A sweep over every served block under both perspectives reports any **float data column** with no declared reading, which is a new word entering the column vocabulary unannounced. Tests only; no served exhibit changes.

**Float only, deliberately.** An int, bool, string or date column is typed by the IR and reads correctly with no help. A float is exactly the column whose digit count cannot be inferred honestly, which is why the sheets exist and what the sweep is about.

**Two structural exemptions, and they are the interesting part.** A block whose **column axis is named** is skipped: its labels are values of that axis (units, views, layers, probe steps), so the label is data and the reading belongs to the row rather than to the word. Two blocks are the same case without being able to say so, `stats_df` and `reins_stats_df`, whose columns are computation views, unit names, or the two axes of a bivariate, over an unnamed column axis. They are listed by name with a reason. Naming that axis would retire the list, which is the a254 fix applied to the other axis and is worth doing upstream rather than in a test.

**The open list.** `PENDING_VOCABULARY` names the 43 labels served today with a reading nobody has declared, for the author to rule on one at a time: a sheet entry, an exemption with a reason, or a rename onto a word the sheet already carries. It is a ratchet in both directions, so a new undeclared label fails and a label that stops being served has to come out, and it cannot rot into a blanket exemption.

**What the registry caught on its first pass**, which is the argument for having one: the moment vocabulary drifted before it existed. `EX`, `SD` and `Sk` are served beside the declared `CV` and `Skew`, and `mean`, `sd`, `skew` and `cv` appear again in lower case on the bivariate and tail behavior frames. Four spellings of a mean, three of a standard deviation and three of a skewness, none of them wrong in place and no two of them the same word. That is a naming question rather than a formatting one, which is what the sheet was supposed to surface.

## 1.0.0a287

**[Format-Sheet-Application] the sheets are wired, the seven dicts are gone.** `build_exhibit` reads the format sheets and lays them under each block's own kwargs, and `MEASURE_FORMATS`, `BS_WINDOW_FORMATS`, `SHARPEN_FORMATS`, `PENTAGON_FORMATS`, `CALIBRATION_FORMATS`, `DISTORTION_FORMATS` and `EVALUATION_FORMATS` are deleted along with their exports. `STAT_SLICE_FORMATS`, `STAT_SLICES`, `STAT_SLICE_TITLES` and the `score_grid` literal stay: those are properties of a block's shape rather than of a column's meaning, which is the one deliberate exception.

**Applied after relabeling, which fixes a latent bug.** greater_tables keys formats on the *displayed* column label, and `exhibit_frames` relabels every frame through the host's `_relabel` after the builder returns. The old dicts were attached by the builder, keyed on pre-relabel names, so a `renamer` that touched a formatted column silently detached its format. The sheet's keys now make the same trip the frame's columns made, read off `_relabel` itself rather than off `renamer`, so the `use_labels` gate cannot come apart from it either.

**Precedence**, low to high: greater_tables' dtype and tag inference, `formats-raw.yaml`, `formats-insurer.yaml` under that perspective, the same file names in `~/.aggregate` and the working directory, then a block's own `formatters` entry, which always wins. A block that declares a tag selector keeps its own and takes none from the sheet, since a selector may be a regex or `'all'` and those do not merge with a list.

**The two `ratio_cols` call sites are gone.** The eight ratio columns of `economic_ratios_df` and the three of `evaluation_df` point at the `ratio` style, which stamps greater_tables' ratio tag wherever those labels appear. A ratio is now tagged on every block that serves one rather than on the two that remembered to say so.

**Readings that changed, and they did change.** This is the version where the drafted vocabulary meets the served exhibits, so the committed exhibit snapshots move (124 of them, regenerated). What moved, in one list: every ratio column gains the `ratio` tag and its `.1%` reading, including `CV` on the reins and validation frames and the whole `economic_ratios` ratio block, which were reading as inferred decimals; the pentagon amounts and the VaR ladder read as `,.7g` under RAW and `,.2f` under INSURER, where the ladder had been inference; `Skew` reads `.3g` under RAW, three significant figures, which greater_tables 6 parses and 5.x could not, so the `_core` comment saying the `g` kind does not exist comes out; `error`, `param`, `gini_p` and `area` read `.5g` under RAW and `.5f` under INSURER, replacing `.2e` and the inferred SI notation `error` picked up on `stats_df`; `x_min`, `x_max` and `W` gain a decimal; `bs` goes to six significant figures; `T` reads `,.1f`; `VaR/Mean` reads `.3f`, matching `PQ`, both being multiples.

**The `P` collision is real and is handled by the scoped section.** `P` is premium as a data column in twenty served blocks and the probability ladder as `tail_df`'s index in ten, so the tail exhibit carries `P: probability` in the sheet's `exhibits:` section. Without it the ladder would read as an amount, and under the insurer money format the 1 in 1000 and the 1 in 10000 rungs would both print `1.00`.

Nothing is owed on the app side: `aggregate_api`'s own `tables.FORMATS` has been empty since its a94, and it never imported these names.

## 1.0.0a286

**[Format-Sheet-Files] the column formats become a file.** Two YAML **format sheets** ship as package data in the new `aggregate/formats/`, the sibling of `aggregate/agg/` and for the same reason. `formats-raw.yaml` holds the library's default reading of every named column, keyed by column label; `formats-insurer.yaml` is an **overlay** holding only the entries where the business reading differs. Absent means same. Two full sheets would drift apart, a delta cannot.

Nothing is wired yet, so no served exhibit changes in this version: this is the vocabulary and the loader, and `[Format-Sheet-Application]` is the version that applies them and deletes the seven module level dicts they replace.

**A sheet is also a registry of the column vocabulary.** An entry asserts that a column label means one thing across the package, which is a statement the old dicts could not make: the same fact, a CV reads as a percentage, was declared wherever someone remembered to declare it. The enforcement sweep arrives with `[Format-Sheet-Enforcement]`; the file is what makes it possible.

**Overrides are the `.agg` rule, not a new mechanism.** The loader looks for the same two file names in the shipped directory, then `~/.aggregate`, then the working directory, nearest winning, merging per key rather than per file, so a one line local sheet changes one reading and inherits the rest. Overriding a shipped sheet is the same act as overriding a shipped DecL database, so it is the same search path.

**Styles.** A sheet's `styles:` section names a reading once (`money`, `ratio`, `probability`, `residual`) and every column that wears it points at the name, so the house ratio precision is one line rather than fourteen. Styles merge across the layers **before** column entries resolve against them, which is what lets the insurer overlay redefine `money` in a single line and move `L`, `M`, `P`, `Q`, `a`, `E`, `C` and the whole VaR ladder with it. Four style names are special because greater_tables owns them as semantic column tags (`ratio`, `year`, `date`, `raw`): a column pointing at one of those is stamped with the tag as well as the format, so a consumer learns the column's kind and not only its reading. That is what retires the two hand written `ratio_cols` call sites next version.

**Scoped entries ship from day one**, and the collision they exist for was already there: `P` is premium as a data column in twenty served blocks and the **probability ladder** as `tail_df`'s index in ten. `exhibits: {tail: {P: probability}}` keeps the ladder reading as a probability; without it the insurer money format would print the 1 in 1000 and the 1 in 10000 rungs both as `1.00`.

**Wire safe by construction.** Every value goes through greater_tables' `parse_sugar` at load, so a bad string raises once, naming the file and the key, rather than per cell. A value can be sugar, an int, a mapping of `FormatSpec` fields (the only way to reach `scale`, `prefix`, `suffix`, `negative: paren`) or a style name, and it can never be a callable. That is the feature: a callable never reaches the IR, so a sheet can only say things a client can re-render.

`pyyaml>=6.0` is now a declared dependency. greater_tables already required it, so this costs nothing to resolve; it is declared because this package imports it, and a transitive dependency that vanished when greater_tables changed its YAML library would break the sheets a long way from the cause. Parsing is `yaml.safe_load` only.

## 1.0.0a285

**[Kappa-Chart-Surfaces] one chart name, three sources.** `kappa` now draws for a `Portfolio` and for an `Aggregate` carrying an occurrence program, alongside the `BivariateAggregate` it launched on. The question is the same one in all three cases, what each part contributes given the whole, so it is one name.

**On a `Portfolio`**: the kappa panel the `port` overview already draws on its right hand side, served alone, in an equal-aspect single-panel document. The panel builder is **extracted and shared** rather than duplicated, so the two documents cannot drift; `chart_port` is otherwise unchanged. No band here, unlike the occurrence version: a book's unit kappas come off the independence trick in `density_df` rather than off a stored joint, so a conditional band would be new machinery with no session behind it, and the honest picture is the mean curves.

**On an `Aggregate`**: a thin delegate that reads `occ_joint` and emits the band chart. The curves are a property of the cession rather than of a calibration, which is why this is a chart on the built object and not a fourth pricing call; and because the joint is held, drawing after an allocation costs a lookup rather than a second 2-D FFT.

The predicate is duck-typed over the three shapes (a book has units, a joint has a mode, an aggregate has a program), so the chart module needs no import of every class it draws for.

**One consequence worth stating**: `available_charts` on a `Portfolio` now returns two names rather than one, `['kappa', 'port']` in registration order. `primary_chart` still answers `'port'`, which is the function that exists to say which one to draw when nothing else was asked for, and a caller reading the first available chart as "the object's own picture" should read that instead.

## 1.0.0a284

**[Natural-Allocation-Exhibit] the occurrence branch of `pricing.allocate`.** `pricing.allocate` on a gross-calibrated `Aggregate` calibration serves `natural_allocation_df`: gross, ceded and net for every family, footing exactly, gross constant at the calibrated premium. The predicate widens to match, so `available_exhibits` on a gross-calibrated occurrence result now lists `pricing.calibrate`, `pricing.stand_alone` and `pricing.allocate`, and a **net**-calibrated one still lists the first two only.

**RAW equals INSURER here**, which is the default rule doing its job rather than an omission (plan decision 8). The allocation is already the cedent's one-basis reading: the ceded row is what the cession costs the cedent out of its own premium rather than what a reinsurer would charge for the layer, so there is nothing to drop and nothing to star, and there are no difference rows because the whole table is a decomposition. Contrast the stand-alone leaf, where INSURER is a real restructure. Revisit only if a `REINSURER` perspective ever arrives, which would read the ceded row differently.

**The caption carries the grid the split was priced on**: the joint's bucket size or the fact that the exact lattice was taken, its cell count, its deficit, and the worst `rho_gap` across the families. A priced exhibit should not be readable without the grid behind it.

Two things worth recording, both surfaced by the fixtures and both true of the mathematics rather than of the code. On a program whose worst gross year is also its worst ceded year, `ccoc`'s stand-alone net price and its allocated net share **coincide exactly**, because the mass sits on the essential supremum and the two suprema are the same outcome. And the concave families separate the two readings by a few tenths of a percent on the reference cession, which is the size of the effect the two leaves exist to show.

## 1.0.0a283

**[Calibration-Natural-Allocation-Frame] the calibrated gross premium, split across an occurrence program.** `CalibrationResult.natural_allocation_df`: `(distortion, view)` rows over gross, ceded and net, the pentagon octet across, ceded plus net footing to gross exactly.

The third question about a cession, and the one neither row of `reins_price_df` answers. That frame prices the three views as separate distributions, so each is a price in its own right, and the difference between two of them is the cedent's allowance for reinsurance (`[Difference-Is-A-Perspective]`). This is neither: **one** premium decomposed, on one consistent basis, adding up. Each family's distorted view of the gross distribution sets the weights and the kappa curve off the joint says what ceded and net each earn under them.

**`P` is the shared calibrated premium**, not each family's own fitted one (author's decision, plan §9.1). There is one gross premium in the market; the families differ in the fractions they imply, and the gross row reading constant down the table is the visible statement of that.

**The fractions come off the joint and the level does not.** A distortion is calibrated on the aggregate's fine 1-D gross density while the joint's gross marginal is a coarser rebucketed cousin, so the two do not price to the bit. `natural_allocation` computes shares on the joint's grid and applies them to the stated premium; both readings and their difference ride in `.attrs` as `rho_gap`, per family, alongside the joint's realized sizing (`bs`, per-axis `log2`, whether the exact lattice was taken, the deficit). A priced exhibit should not be readable without the grid it was priced on. On the reference program the concave families gap by under 0.06% of the premium and `ccoc` by 8%, which is the a274 pathology surfacing rather than hiding: a mass-at-zero family read unlimited charges the top grid bucket.

**Two structural guards**, each naming what to do. No occurrence program: there are no halves to split across. A calibration struck on any basis but gross: there is no gross premium here to allocate, so recalibrate with `reins_view='gross'`. The reading is unlimited (`a` infinite, `Q`, `PQ` and `ROE` blank), exactly as on an unlimited `reins_price_df` quote.

**`Aggregate.occ_joint(views=('gross', 'ceded'), **sizing)`**, `occ_bivariate` behind a memo keyed on the sizing (author agreed, plan decision 6). Two surfaces want the same joint of one object, this frame and the kappa chart, and a 2-D FFT is not something to pay for twice because two callers asked the same question. Cleared by `update_work`, so a re-updated object never answers off a joint built on its old grid; a `store_dir` build is not held, since a disk-backed joint owns a directory whose lifetime is the caller's.

## 1.0.0a282

**[Portfolio-Standalone-Frame] every unit of a book priced alone, against the book priced whole.** `CalibrationResult.stand_alone_df`, and the `Portfolio` branch of `pricing.stand_alone` that serves it. The counterpart to `pricing_df`: that frame splits one premium across the units so its rows foot, and this one prices each unit as its own distribution with the same fitted families so they do not. The gap between `sum of parts` and `total` is what pooling is worth under that family.

**The anchor is the total's, once** (author, 2026-08-14). A portfolio calibration resolves its anchor on the portfolio total, and that one asset level is the level of the whole exercise: every row prices `min(X, a)` at the same `a`, the `a` column is constant down the frame, and a `p` calibration and an `a` calibration that resolve to the same level produce the same frame. Nothing re-anchors per unit. This is deliberately not the `reins_price_df` rule, where `p=` lets each view find its own capital: views of one program are alternative wholes and it is right to ask what each needs, while units are parts of one whole whose anchor is the book's. The per-unit `q_i(p)` alternative was considered and rejected on that ground; the unlimited alternative was rejected too, since a mass-at-zero family priced unlimited charges the top grid bucket and tracks `log2`.

**The derived rows** (author's ruling, 2026-08-14). `sum of parts` adds `L`, `M` and `P` and takes `Q = a - sum(P)`, so it is a pentagon at the same asset level as every other row rather than at `n` times it. `total` is the book priced whole there, which ties exactly to the family's fitted premium, target plus its `error`. Sub-additivity puts the sum at or above the total for every concave family: `min(X, a) <= sum_i min(X_i, a)` pointwise, so monotonicity and sub-additivity compose.

**INSURER appends `sum of parts less total` per family**, amounts differenced and ratios re-derived through `complete_pentagon`, interleaved beside its family rather than pooled at the foot. Sum and total are both facts and ride in RAW; what the gap between them means belongs to a perspective (`[Difference-Is-A-Perspective]`, one book up from the cession case). Its capital column is the mirror of its premium column by construction, both rows standing behind the same assets, so the reading is the premium and the margin.

Worth knowing, and now pinned by a test: **`ccoc` books no diversification benefit at all on a bounded book.** A mass-at-zero family charges the essential supremum, and on a bounded book both the supremum and the mean are additive across independent units, so its margin is exactly additive and its benefit row is a real zero rather than a missing number.

## 1.0.0a281

**[Standalone-Rename] stand-alone prices the parts; allocate splits the whole.** Ruling `[Standalone-Prices-The-Parts, Allocate-Splits-The-Whole]` (author, 2026-08-14). The pricing pane's second leaf was doing two different jobs under one name, and only one of them was an allocation.

A `Portfolio`'s `pricing_df` is a genuine allocation: `analyze_distortions` documents its `allocation=` parameter as the tail-share choice for the per-unit premium split, the rows foot to the total, and it was correctly named all along. The aggregate side was not. `reins_price_df` prices gross, ceded and net as three **separate distributions**, each a price in its own right rather than a share of one, and says so in its own Notes. That is stand-alone pricing, and it now says so.

**The split.** `pricing.allocate` keeps the `Portfolio` branch **unchanged**: a book reader sees no difference at all, and the served blocks are byte-identical (112 of the 116 snapshots are unmoved; the four that move are the aggregate-side blocks, and they move content for content). A new `pricing.stand_alone` carries the two `Aggregate` branches, the reinsured one serving `reins_price_df` with its INSURER restructure (ceded dropped, the calibrated view starred, the `less` difference rows appended, `[Difference-Is-A-Perspective]` intact) and the plain one serving the single calibration row, which is the degenerate case of one part that is the whole.

**`pricing.allocate` gains a predicate**, `_perspectives_allocation`, and is the one pricing leaf with a structural gate: a calibration must have parts to split its target across. A `Portfolio` always does. An `Aggregate` does where an occurrence program exists **and** the fit was struck on gross, since a set calibrated on net has no gross premium to allocate; that arm lands with the frame that serves it. An aggregate with no cession has one distribution, which is not a degenerate allocation but the absence of one, and its story is the stand-alone leaf.

**Wire-visible, with exactly one consumer.** The old key's aggregate meaning is gone rather than aliased: an alias would let a client keep asking the wrong question and get a plausible answer. `available_exhibits` on an aggregate calibration now reports `pricing.calibrate` and `pricing.stand_alone`. Plan: `dev/plan-pricing-natural-allocation.md`, phase N1.

## 1.0.0a280

**[Chart-Kappa-Band] the conditional cession, drawn as a curve with a band.** A new registered chart, `kappa`, on a `netceded` `BivariateAggregate` carrying a gross axis. Two panels over one gross outcome axis, and it is the picture the band columns were built for.

**Left, the cession.** `E[ceded | gross]` and `E[net | gross]` with their percentile bands, over the identity. The net band is the ceded band reflected in the diagonal, `[g - q99, g - q01]`, because `N = G - C` pointwise: two shaded regions of equal width, one hugging zero and one hugging the diagonal, which is the conservation statement in one picture. Equal aspect is semantic there, as on the portfolio kappa panel, since both axes are losses and the reading is each curve's slope against 45 degrees. Bands ride as `y2` series, so a band **is** the region between two edges rather than two curves a reader has to associate.

**Right, the share**, the same reading divided by the outcome (quantiles commute with `c -> c / g` at fixed `g`, so the share band is the value band divided by the index and costs no second pass), against the **deterministic ceiling**: for a single layer `limit xs attach` at placement `share`, the most that could be ceded with a gross total of `g` comes from splitting it into claims of exactly `attach + limit`, so the ceiling is a comb with teeth every `attach + limit` and a maximum share of `share * limit / (attach + limit)`. Drawn next to the upper band edge it says how much of the theoretically available cession the program actually delivers, and the answer is typically nowhere near: getting several claims to land exactly at the top of the layer is a lot to ask. Only for a single layer; a tower has no such simple envelope, so the panel has one fewer curve rather than a wrong one.

**Three decisions the emitter makes rather than the caller.** The plotted range is a probability window on the gross marginal (`1e-3` to `0.999`) and not a mass floor, because a raw threshold like `p > 1e-4` means different things at different bucket sizes while a CDF range means the same thing on every grid. There is no smoothing: the band edges step by whole buckets because they are quantiles of a lattice law, and the structure that a rolling mean would tidy away is the comb of the ceiling, which is mechanism rather than noise. And the legend says **percentile band**, never "confidence interval": nothing here is an estimate with sampling error, the joint is the law.

The predicate accepts a **disk-backed** joint, unlike `joint_surface`, which needs the array in memory. Surviving the massive route is the point of a row-wise band. `SERIES_ROLES` gains `'ceiling'` (a deterministic bound on a curve, not a reading of the law).

## 1.0.0a279

**[Kappa-Band-Columns] the kappa curve gains a band, and stops refusing a disk-backed joint.** `BivariateAggregate.exeqa_df(axis=0, levels=None, cdf_range=None)`. Both new keywords are additive: with neither, the frame is exactly what it was.

**Why a mean is the wrong summary here.** The kappa curve is the conditional mean cession given the gross outcome, and `natural_allocation` prices with it. But the gross outcome does not determine the cession: the same 500 can arrive as one claim of 500, ceding 50, or as five claims of 100, ceding 250. The curve averages that away by construction and the allocation inherits the averaging, which is correct as pricing and silent as description. Measured on `agg Demo 10 claims sev lognorm 50 cv 1.5 occurrence net of 50 xs 50 poisson`, at a gross outcome of 500 the mean cession is 103.6 while the 1st and 99th percentiles are 48.0 and 172.5, so the realized share of that outcome ceded runs from under a tenth to over a third. At `g = 200` the band runs from 0.0 to 67.0, meaning a year of that size can cede nothing at all.

**`levels`** adds one `q<pp>_<other axis>` column per probability (`q01_Ceded`, `q99_Ceded`, and `q0.5_Ceded` for a fractional level, so two near levels cannot collide on one name). Each is a quantile of that row's own normalized conditional law, routed through `GridDistribution` so the probability vocabulary stays the library's single implementation and a lattice law's atoms are handled the way every other quantile in the package handles them. A row with no mass yields `NaN`, matching the existing treatment of a conditional mean given a null event.

Worth knowing, and recorded in the tests: a percentile band is **not** guaranteed to contain the mean. In the far left tail the conditional cession is a spike at zero carrying a vanishing chance of a full limit recovery, so `q99` is 0 while the mean is not. That is a true statement about a very skewed conditional law, and the argument for drawing the band rather than the curve alone.

**`cdf_range`** crops to a probability window on the conditioning marginal and computes the quantile columns only there. The quantiles are the expensive part (42 s over a 65,536 row grid, against 0.8 s to build the joint they read), and a plotted range is a few thousand rows rather than sixty five thousand: on that grid a `(1e-3, 0.999)` window is 6,823 rows and 5.1 s. It crops rather than blanks, so a `NaN` keeps meaning "no mass here" and never "not measured here"; `F` and `S` stay the whole distribution's. A probability window also means the same thing on every grid, which a raw mass floor does not.

**The massive refusal is gone**, from `exeqa_df` and therefore from `natural_allocation`, which only ever inherited it. The sweep runs over `_row_bands`, so a disk-backed joint answers at bounded memory. That was the one hole in an otherwise first class massive surface: everything else (`summary_df`, `stats_df`, `marginals`, `moments`, `corr`, `pushforward`, `plot`) already worked on disk.

## 1.0.0a278

**[Joint-Row-Bands] one band iterator, and one conditional probe, either side of the disk boundary.** `JointBandsMixin` on both `BivariateDistribution` and `MassiveBivariateDistribution`. Everything the kappa band needs is a row-wise fold of the joint (`d @ y` against `d.sum(1)`, and a quantile per row), and a fold should not have to know whether the density is a numpy array or a zarr store. This is the small phase that makes the next one small.

**`_row_bands(axis=0, band_rows=None)`** yields `(r0, r1, block)` over the joint. In core it yields exactly **one** band covering the whole grid, so a consumer written against it costs nothing there: no copy, no chunk arithmetic, one pass either way. On the massive route the default band is the store's own row chunk, which is the read size the chunking was chosen for, and peak memory is `band_rows * n_other * 8` bytes regardless of grid size. `axis=1` yields bands of the **transpose**, so a consumer always folds along rows and the transpose is not a special case for the caller.

**`slice(x=)` / `slice(y=)` reaches the in-core container.** The analyst's probe for one conditional law existed only on the massive container, which is backwards: the in-core case does it in two lines, and that is exactly why it should be the same two lines under the same name on both. The massive implementation is deleted rather than copied; both now read the mixin, addressing axes through the role accessors (`axis0` / `axis1` / `bs0` / `bs1` / `axis_names`) rather than the positional `.ceded` / `.net` names, which lie for a `('gross', 'ceded')` joint. `BivariateDistribution` gains `bs0` / `bs1` role properties to complete that surface.

No behavior changes: `slice` on a massive joint answers exactly as it did.

## 1.0.0a277

**[Sizing-And-Passthrough] the netceded joint sizes itself honestly, and says what it chose.** Three defects reported together in `dev/notes-net-natural-allocation.md` §7, fixed together because they are one story: a caller could not reach the grid they wanted, and the grid they got instead answered anyway. The measured case is `a.occ_bivariate(views=('gross', 'ceded'), bs=0.5)` on an unbounded lognormal, which returned a 512 x 2,048 joint carrying a **deficit of 0.535** and a correlation of **-0.2225** for a comonotone pair, after a single warning.

**A pin that cannot be honored raises.** When a pinned `bs` or `log2_x` / `log2_y` needs more than `2**total_log2` cells, the sizing used to clip the wider axis and report a "tail deficit". It is not a tail loss: with both axes pinned equal the rule cut axis 0 to the 16 bucket floor, and a 16 bucket gross axis is a different distribution rather than a truncated one. The caller has stated numbers that cannot all be honored, so `ValueError` now names what was pinned, how many cells the windows actually need, and the two escapes (`total_log2=`, with `store_dir=` at that size, or relax the pin). Nothing pinned still coarsens `bs` until the pair fits, so the default build never raises.

**`occ_bivariate` passes the whole grid through.** It took `bs`, `log2_x` and `log2_y` and nothing else, so it could reach neither `total_log2` nor `store_dir`, and those two are coupled: without the first no grid is large enough to want the second. The massive netceded path existed and worked but was unreachable from the public method, which is why the session behind the notes drove it by constructing `BivariateAggregate(mode='netceded', ...)` by hand. The signature now carries `total_log2`, `store_dir`, `row_chunk`, `col_chunk` and `keep_transform`.

**`update(log2=...)` on a netceded joint does something.** It was a no-op on the `occ_bivariate` route (sizing always read the constructor's `_nc_kwargs`), while the over-budget warning recommended it by name. The two keywords now size whichever grid the object owns: the inner aggregate's on the DecL prefix route, where `build('netceded agg ...', bs=1, log2=16)` means what it means for every other DecL form, and the **joint's** on the `occ_bivariate` route, where the inner aggregate arrives already updated. A scalar `log2` is the 2-D budget, a `(log2_x, log2_y)` pair pins the axes, and `bs` is the common bucket size; a `bs` pair is refused, since one aggregate split two ways carries one lattice.

**The default grid prefers the exact lattice.** The netceded joint forces one `bs` on both axes because the third view is read as a subtraction on the index, which is what makes `kappa_C + kappa_N` the identity exactly and `A_C + A_N = P` exact by construction. When that common lattice can also carry every per-claim view image, the comonotone scatter never splits a point and the conditional (kappa) curve is exact rather than accurate to the smear. The default now measures that lattice (`_netceded_exact_bs`, folding `_lattice_bs`'s float gcd over the images and the gross `bs`) and takes it when the budget affords it, falling back to the budget-derived `bs` otherwise. Exactness is a property of excess of loss layers on an aligned lattice and not a general one: a share cession multiplies by a non-integer factor and usually has no usable common lattice at any size, so the rule is prefer the exact lattice when affordable, report when not, never pretend.

**What was chosen is reported.** `BivariateAggregate.bs_explanation` gains a sentence per netceded joint saying whether the scatter fires, and if not, why: no affordable lattice, or an affordable one the caller pinned past. The realized decision rides on the object as a `NetcededSizing` record, which is what the pricing exhibits will carry in their `attrs`. `build_netceded_joint` returns that record in the slot where `clipped` sat; a netceded joint no longer clips.

## 1.0.0a276

**[Tail-Symmetric-Ladder] the return period table is indexed by probability and carries both tails.** `tail_df` and `tail_periods_df` on `Aggregate`, `Portfolio`, `PnL` and `BivariateAggregate` are now indexed by the non-exceedance probability `P`, which runs from `0.001` to `0.999` on the default ladder, with the return period `T` as the first column. Every rung contributes **both** of its probabilities, the lower tail `1 / T` and the upper `1 - 1 / T`, so the index is symmetric about the median and one table serves both sign conventions: a loss is read off the high rows, a payoff off the low ones.

**What the old frame could not do.** It was indexed by `T` and picked one probability per rung from the object's `is_loss_value`, a loss taking `p = 1 - 1/T` and a payoff `p = 1/T`. That is one side of the distribution, chosen for the reader, and it left the other side unreported. A signed position has an adverse left tail and an informative right one at the same time, and the caller, not the frame, is the one who knows which question is being asked. The frame therefore now states no orientation at all, and `return_period_frame` loses its `is_loss_value` argument: the signature is `(q, tvar, mean, periods=None)`. Its two branches still come from `period_to_p`, taken together rather than chosen between.

**The `T` column is a reading, not one formula in `P`.** It is `1 / P` below the median and `1 / (1 - P)` above it, which is the rung the row came from and the interpretation that matters at that probability. So a 1-in-200 loss year and a 1-in-200 shortfall year both label `200`, on opposite sides of the table. There is no monotone map from `P` to `T` here, and the convenience is deliberate.

**Breaking, in the alpha sense.** `tail_df.loc[200]` becomes `tail_df.loc[0.995]` (or `.loc[0.005]` read as a payoff); the `p` column is gone, being the index; the `Portfolio` and `BivariateAggregate` MultiIndex level renames from `T` to `P`; and the default frame is 19 rows rather than 10. The 1-in-200 (99.5%, Solvency II) and 1-in-250 (99.6%, US capital adequacy) anchors are still emphasized in the `tail` exhibit, now matched on the `T` column instead of the index, so each anchor emphasizes its two rungs and the capital row is highlighted under either convention. Eight of the 116 exhibit snapshots move, all of them `tail`, and no other exhibit changes.

## 1.0.0a275

**[BS-Window-Dtypes] the grid sizing frames carry their own dtypes.** Every column of `Aggregate.bs_window_df` was `object`: `applies`, `x_min`, `x_max`, `W`, `bs`, `log2`, `coverage`, `note`. The frame was built by transposing a method-per-column block (`pd.DataFrame(rows).T`), and because the rows mix bool, float, int and str, every column of that block was mixed and typed `object`; a transpose carries the dtype across wholesale rather than re-inferring per column. The tell was which columns worked: `selected`, `log2_need` and `clipped` are the three assigned *after* the transpose, and were the only three correctly typed.

It showed in the served table. `bs_window` is a passthrough exhibit, so an object column reaches the IR as `dtype='string'`, left aligned, wrapped, and carrying **no raw values**, which is what an interactive grid sorts and filters on. `log2` had no `BS_WINDOW_FORMATS` entry, having never needed one as an integer, so it rendered with no format at all.

The sizer now builds index-oriented (`pd.DataFrame.from_dict(rows, orient='index')`), which infers per column. That is what the Portfolio sizer has always done, building from a list of row dicts and never transposing, which is why its frame was already correct. `applies` is bool, the window and grid columns are float, `coverage` and `note` are string.

Both `log2` columns, aggregate and portfolio, are now nullable `Int64`. An exponent reads as an integer, and both `log2` and `log2_need` can be genuinely absent (a non-applies `sbj` row records no grid, and `log2_need` is NaN on a degenerate window), so plain `int64` will not hold them; without the cast the published dtype would depend on the book. `log2_need` consequently reads `6` rather than `6.000`.

`BivariateAggregate.bs_window_df` and `axis_support_df` had the identical transpose and are fixed the same way. The first feeds the same exhibit, so fixing only the aggregate would have left that leaf broken; the second was rendering unformatted float repr (`19.499999999999996`) because a string column bypasses formatting, and now reads `19.50`.

Twelve of the 116 exhibit snapshots move, all in `bs_window` and `dependency`, all of them column dtype, alignment and raw values plus the two text improvements above. Captions, notes, heads, feet and level counts are untouched, and no keys are added or removed. Every consumer already wrapped these cells in `float()` / `int()` / `bool()`, so nothing downstream changes. Plan and the deliberate snapshot read in `dev/done/plan-bs-window-dtypes.md`.

Known and not fixed here: the bivariate `clipped` column is a bool flag while the aggregate `clipped` is an estimated mass, and `BS_WINDOW_FORMATS` carries one `'.2e'` for the name, so the bivariate flag renders `0.00e+00` where it should read `False`. Pre-existing, unchanged by this work, and fixing it means renaming one of the two columns or scoping the format per class.

## 1.0.0a274

**[NetCeded-Natural-Allocation] a gross premium splits across an occurrence program.** `BivariateAggregate.natural_allocation(distortion, P=None)` allocates a gross distorted premium to the occurrence ceded and net components of a netceded joint. Rows `gross` / `ceded` / `net`, columns the pentagon octet, ceded plus net footing to gross exactly.

**The question neither existing row answers.** `reins_price_df` prices gross, ceded and net as three separate distributions and says so: views are not a decomposition, and gross less net is the cedent's allowance for reinsurance rather than a reinsurer's price ([Difference-Is-A-Perspective], author 2026-08-11). This is the third question. Given a premium for the gross book, what does each half of the program earn on one consistent basis, adding up. It needs the joint because an occurrence program splits `G = C + N` claim by claim, so at the aggregate level `N` is not a comonotone function of `G`: the random claim count decouples them, and no 1-D calculation reaches it. Under an *aggregate* program it would be comonotone and the split would be arithmetic.

**The increment form, which is why no derivative of `g` is evaluated.** The natural allocation `A_C = E[C g'(S_G(G))]` becomes, on the lattice, `A_C = sum_i kappa_C(g_i) Delta_gS_i` with `Delta_gS_i = g(S(g_{i-1})) - g(S(g_i))` off the joint's own gross marginal. That is the Lebesgue-Stieltjes statement directly: the usual conditions on `g'(S)` do not arise, atoms are handled exactly, and `A_C + A_N = rho_g` of the marginal to floating point by construction, because `kappa_C + kappa_N` is the identity. The weights come from `choquet_weights`, the same convention `Distortion.price` uses, so the gross row **is** that function's answer on the same grid. The kappa curve comes from `exeqa_df` (a273).

**Fractions, not levels, and both readings reported.** A distortion is calibrated on the aggregate's fine 1-D gross density; the joint's gross marginal is a coarser rebucketed cousin with its own deficit, so `rho_g` of it does not hit the calibrated premium to the bit. The method therefore computes allocation **fractions** on the joint's grid and applies them to the caller's `P`, which is robust to discretization and keeps additivity exact whatever `P` is. Rather than absorb the difference, `.attrs` carries `rho_joint` (the joint's marginal), `rho_fine` (the source aggregate's fine density) and `rho_gap`, their difference: a large gap says the joint's grid is too coarse to price on, which is the caller's judgment to make. Measured on an ordinary excess layer at `bs=4`, the gap is 2e-3 relative under `ph 0.7`.

**Accepts either view pair, in either axis order.** `('gross', 'ceded')` and `('gross', 'net')` both work, and gross may sit on either axis; whichever component is on the joint's second axis is read from the kappa curve and the remaining one is `g` less that curve, taken on the index. A definition rather than a second measurement, which is what makes the rows foot exactly. A `(net, ceded)` joint is refused, having no gross axis to condition on, as is copula mode, which wants the deferred [Bivariate-Total-Exeqa] work. Both refusals name what to do instead, and both fire before the call-update check so a caller holding the wrong shape of object is told that rather than told about its density.

**Two behaviors documented rather than engineered away.** Under a grid deficit the identity distortion prices **above** the mean by `deficit * top_value`, because `choquet_weights` runs forwards and parks unrepresented mass at the largest represented outcome; a test pins that quantity so it is never mistaken for an allocation error. And the reading is unlimited, so a distortion with a mass at zero (`ccoc`) charges the largest outcome the grid happens to represent and its premium moves with `log2` rather than with the risk. The fractions, being ratios on one grid, are far steadier than the level; pass a finite-`a` price in as `P` when the level matters.

New file `tests/test_natural_allocation.py`, fast tier: the identity distortion recovers the component means on a deficit-free lattice program; ceded plus net foots to gross at 1e-12 under four distortion families; a lattice-aligned 50% share allocates exactly half under every one of them; the gross row ties to `Distortion.price`; a caller supplied `P` is split by the same fractions and totals exactly; `rho_gap` is small, reported, and does not move with `P`; the two view pairs and the two axis orders agree.

Plan `dev/done/plan-natural-allocation-to-occurrence-net-ceded.md`, executed with nine recorded divergences at its foot, including a correction to one of its own tests. Phase 3 [Bivariate-Total-Exeqa] stays deferred behind its author gate and is now tracked in `dev/TODO.md`.

## 1.0.0a273

**[Bivariate-Exeqa] the kappa curve comes off the joint.** `BivariateAggregate.exeqa_df(axis=0)` returns the conditional mean of one axis given the other, over the whole conditioning grid: index the conditioning axis, columns `p`, `F`, `S`, `exeqa_<conditioning axis>` (the identity) and `exeqa_<other axis>` (the curve). It is one matrix vector product per direction, `d @ y` against `d.sum(1)`, and it adds no state.

**Why it could not come from the existing machinery.** Portfolio computes `exeqa_*` by the FFT trick, which assumes its units are independent. The two axes of a netceded joint are dependent by construction, sharing a claim count and a comonotone per-claim cession, so that route is unavailable and the answer has to come out of the joint the library already builds. This is the piece phase 1 of `dev/done/plan-natural-allocation-to-occurrence-net-ceded.md` identified as missing; nothing in `bivariate.py` computed a conditional mean before.

**What it is exact about.** The mass weighted mean of the kappa column reproduces the other axis's marginal mean to floating point, both being the same sum taken in a different order. Pointwise the curve carries the rebucketing scatter: `scatter_bivariate` splits each per-claim point bilinearly over up to four cells, which preserves both marginal means exactly but smears a conditional one. Measured on an ordinary excess layer at `bs=4`, the smear is **under one bucket** everywhere. Where the cession lands on the joint lattice there is no split and the curve is exact, which is the shape of the first test.

**Zero-mass rows are `NaN`, not zero,** in both `exeqa` columns, with `p` saying why: a conditional expectation given a null event has no value. The joint is de-fuzzed at construction, so `p > 0` is an exact test rather than a threshold. A disk-backed (massive) joint is refused rather than read row by row, and pointed at `MassiveBivariateDistribution.slice`.

`F` and `S` are read through `GridDistribution`, so the probability vocabulary stays the library's one implementation; under a grid deficit `S` ends at the deficit rather than at zero.

New file `tests/test_bivariate_exeqa.py`, in the fast tier: a lattice-aligned share cession pins signs, axis order and grid alignment at once (exact to 1e-12); the continuous version is within a bucket; the mass weighted tie to the marginal holds at 1e-12; the pointwise value agrees with the normalized row read as a `GridDistribution`; and the two-joint decomposition `E[C | G] + E[N | G] = g`, taken across the `(gross, ceded)` and `(gross, net)` pairs built on one pinned `bs`, closes to within a bucket.

**A correction to the plan, recorded rather than quietly fixed.** Its phase 1 test 4 asked that `exeqa_self + exeqa_other` equal the index. Under axis conditioning that is vacuous, `exeqa_self` being the index by definition; the wording came from Portfolio, where conditioning is on the **total** and the units genuinely sum to it. The two-joint form above is the statement that carries the intended content, and it is the one that shows the scatter.

## 1.0.0a272

**[Picks-Robustness] infeasible picks warn instead of failing silently, and the picks debug audit checks `quad`'s error relatively.** Two hardening changes in `_picks_work`, surfaced by writing the picks illustration note. A layer loss pick below the full limit losses implied by the layers above it forces a negative adjustment weight, the adjusted survival function then increases across the layer, and the returned severity carries negative probabilities; that used to happen with no signal at all. Now a warning names the offending layers at the moment a weight goes negative, and a catch-all warning fires whenever the final adjusted density is negative anywhere by any route (interactions with the cap at one and the bottom layer rebuild included). The density is still returned so the caller can inspect it.

**The debug path asserted `quad`'s absolute error estimate below a fixed `1e-6`.** The estimate grows with the integration range while the answer stays accurate, so `Aggregate.picks(..., debug=True)` raised `AssertionError` on ordinary curves (a lognormal mean 500 cv 2 integrated to 10,000, for instance). The check is now relative to the integral's value.

New tests in `tests/test_picks.py`: feasible picks are hit exactly on the grid, the debug audit runs on the case that used to raise, and infeasible picks warn and still return the (negative) density.

## 1.0.0a271

**[Chart-Marks-Mean-Only] the percentile lines come off every chart, and the mean stays.** A mark is a line the document asserts permanently, and after this exactly two readings earn one: the mean, on the mass panel of `agg`, `port` and `pnl`, and break even at zero, in both panels of `pnl`. Everything marked at a return period is gone. `agg` loses the full-weight 1-in-200 from its density panel and the faint 1-in-100 and 1-in-250 from its Lee panel, `port` loses the 1-in-200 from both of its panels, `pnl` loses the two faint Lee anchors. `CAPITAL_ANCHOR` and `LEE_ANCHORS` are deleted from `charts/_emit_aggregate.py` and the two emitters that imported them follow.

**The reading arrived somewhere better, which is the whole argument.** A percentile is a point on a curve the chart already draws, and a reader who wants it hovers: the app's readout strip writes every series value at the hovered coordinate, and a return-period axis prints there as `1-in-N`. A permanent dashed vertical asserting the same number costs a label collision rule, a side rule, a faint weight and a place in every punch up round, and returns a number that pointing at the curve gives for free. The mean and break even stay because neither is a point a hover can reach: the mean is a property of the whole distribution rather than of any coordinate, and zero is where the sign of a signed outcome changes.

**What it costs, recorded rather than discovered later.** The full-weight 1-in-200 goes with the rest, and on the `port` kappa panel that line had a reading attached: read up from capital and each unit's share there is its share of the loss. That reading is now a hover on the kappa curves. The `agg` Lee panel and the `port` kappa panel are left carrying no marks at all, and nothing replaces them. The change is purely subtractive: no emitter gains a mark anywhere.

**`CHART_IR_VERSION` stays 2**, by the rule the constant documents. Emitting fewer instances of a record every reader already handles is not a field a reader must act on, not a changed meaning and not a removal. Every two-panel document's hash does move, so ETags keyed on `agg`, `port` and `pnl` invalidate once, and that case is named explicitly as not qualifying.

**The mechanism is untouched.** `Mark`, `ChartDoc.marks` and `MARK_ROLES` are as they were, `'capital_anchor'` included: that tuple documents what a mark *may* say, not what the shipped emitters happen to say, and a future reader wanting an anchor should find the word already spelled. `Mark.faint` stays a field, set by no shipped emitter now, because removing a field is the one edit here that would have moved the IR version; its docstring loses the anchor example. The matplotlib compositor loops `doc.marks` generically and simply draws fewer lines, and the app draws what it is served, so the lines leave the browser with no app-side deletion at all. `tail_periods_df` keeps its `tail` exhibit caller and stays public, and `exhibits._core.CAPITAL_ANCHOR_PERIODS` is untouched: the tail **table** has room for both rows where a panel does not.

**A negative guard, `tests/test_chart_marks.py`,** holds over every registered chart on a built `agg`, `port` and `pnl` at once: no mark carries `role='capital_anchor'`, no role outside `{'mean', 'break_even'}` appears, and the mean is marked once where it is drawn. A guard rather than three edited tests alone, because the anchors could otherwise come back one emitter at a time. The `agg` rendered baseline is regenerated on the pinned matplotlib 3.10.9; `distortion.png` is unaffected, that chart emitting no marks.

**Supersedes round 5 ask 4** (author, 2026-08-10), which asked the density panel to carry mean, 1-in-100 and 1-in-200 with only 1-in-250 coming off. It was never executed, so there is nothing to unwind.

Plan: `dev/done/plan-no-reference-lines.md`, canonical in the API repo. The app half is one tooltip and two stale comments, and lands there.

## 1.0.0a270

**[Derived-Premium] `derive premium`, the fourth premium head: the engine premium grossed up for the expense clause.** `inherit premium` copies the engine's technical premium T, and the `less` clause then deducts expenses from it, so the expenses eat the risk load. `derive premium` reads T as a technical, risk loaded premium and books the unique gross premium whose own expenses leave exactly T behind: with fixed expense total F and premium expense ratio total r, `P = (T + F) / (1 - r)`. Multiple fixed terms add; multiple premium ratios add, combined ratio style; grouping and `as` labels change nothing. The expected underwriting result then carries the engine risk load and nothing else. This automates the hand written gross up the a268 paren arithmetic entry used as its motivating case, `(100_000/(1-.25)) premium`, and folds the fixed expense numerator in besides.

**Three refusals, all build time `ValueError`s that name the fix.** A **loss basis** expense: losses are not reliably known by inspection, so the gross up is undefined (use fixed or premium expenses, or `inherit premium` with the loss expense left in place). An engine with **no premium**: nothing to derive from, the inherit error's twin, for agg and port engines alike. Premium ratios totalling **one or more**: no finite premium grosses that up. An expense free head is legal and equals `inherit premium`; port engines derive from their accumulated premium exactly as they inherit; `xpnl` shares the head; `retro` remains mutually exclusive by grammar. The sentinel is `aggregate.parser.DERIVE_PREMIUM`, resolved in the underwriter through the new `_pnl_builders.derive_consideration`, which is the fixed point of `resolve_expense` restricted to the deterministic bases and sits beside it.

**`derive` leaves the identifier space. Breaking, narrowly.** The keyword joins the `ID` exclusion, `_TERMINAL_LABELS`, the Pygments colorizer and `agg.sublime-syntax`, as `test_grammar_sync` requires. No shipped corpus used `derive` as a name (checked, per the a249 `[Single-Placement-Keyword]` model), so nothing round trips differently: the spec snapshot re-captured with zero moved entries. The web app's `decl-keywords.json`, which lives in a separate repository, carries no pnl vocabulary at all today; the API's `dev/plan-pnl-button.md` adds the pnl group with `derive` in it.

**`pnl_program` writes `derive premium` now. Behavior change.** `_pnl_consideration` returns the derive sentinel instead of the inherit one whenever the engine carries premium, so the program `pnl_program` writes (and the app's PnL button echoes) grosses up: with the default `expense_ratio=0.25` the booked premium is T/0.75 where it used to be T, and premium net of expenses is the technical premium exactly. The loss ratio sized head for an engine without premium, and its rounding, are untouched, and a derived premium is never rounded, exactly as an inherited one never was.

New DecL cases in `decl-testers.agg` (`DRV.` block), tests in `tests/test_pnl_derive_premium.py`, the grammar reference regenerated, the cheat sheet and the P&L pipeline and features pages updated. Plan: `dev/done/plan-derived-premium.md`; the app half is `dev/plan-pnl-button.md` in the API repo.

## 1.0.0a269

**[Chart-Reflected-Reading] a probability axis declares its reflection, and a Lee panel reads the survival function.** The chart IR offered five renderer switches, `log`, `full_range`, `return_period`, `invert` and `kind`, each following the one rule `[Chart-Declared-Readings]` set: the document declares which readings a quantity honestly admits, the switch acts wherever the declaration exists and nowhere else. One reading was missing. A Lee panel draws the quantile function against `p`, the non-exceeding probability, and inverted it draws `F(x)`. What nobody could ask for was `S(x) = 1 - F(x)`, the reading an actuary reaches for most often and the one a log axis was invented for. `reflect=True` is that reading, the map `v` to `1 - v` on every axis that declares it.

**The declaration is a paired axis, `ChartAxis.complement_of`,** mirroring `reciprocal_of` in every respect: an undrawn axis sitting in `doc.axes`, carrying its own label, its own `scales` and its own window, named by no panel. A boolean plus a label field on the drawn axis was refused because it has nowhere to say that `S(x)` is log readable while `F(x)` is not, which is the main reason to want the reading at all. One boolean, not a `reflect_x` / `reflect_y` pair, which `invert` would make ambiguous.

**Six documents gain one.** `agg` (and through it `pnl`), `severity` and `reins` gain a `survival` axis, Exceeding probability, `scales=('linear', 'log')`, paired to `p`. `distortion` and `envelope`, the two unit-square documents, gain `s_complement` and `g_complement`, labeled `1 - s` and `1 - g(s)`: the literal coordinates rather than the dual's name, because the reflected point is `(1 - s, 1 - g(s))` for every series on the panel where naming it the dual asserts an identity that holds only of the `g` curve. `port` gains nothing, having no probability axis at all. Reflecting both axes of a distortion draws the dual, so under `dual=True` the reflection exchanges which curve each legend entry traces: the picture is right and the names are stale, and `dual=False` reads clean.

**The two probability readings compose with no special case,** because `complement(v) = reciprocal(1 - v)`. The 'complement' map *is* reflect-then-reciprocal, so an axis already read reflected takes the plain reciprocal whatever the document declares. On a loss that redraws the curve `return_period=True` draws by itself. On a signed `PnL`, whose map is 'reciprocal' because the adverse tail is the low one, `reflect=True, return_period=True` reads the *upside* tail's return period, which is a picture unreachable any other way.

**Two renderer behaviors are now keyed on the return period specifically rather than on "a map is present",** which is the trap in the change and both sites carry a comment saying so. The `MAX_RETURN_PERIOD` cap exists because the quantile function saturates and `T` diverges, and a reflected probability axis is bounded in `[0, 1]`. The companion window release exists because a return-period reading re-slices the panel into the deep tail, and reflection is a bijection of `[0, 1]` onto itself that re-slices nothing.

**The atomic ladder needed no reflected case,** and `_render_xy_panel` gains a Notes paragraph recording why, because it looks like a bug until someone works it out: matplotlib's step drawstyles are defined on the order of the points given, not on the direction of the axis, so a right-continuous step drawn over the reflected point sequence is exactly the mirror of the picture it drew before. The same reason `invert` needed no explicit switch.

**Validation tightened for both pointers.** `ChartDoc.__post_init__` now loops over `reciprocal_of` and `complement_of` together, so the three existing checks (names an existing axis, is not itself drawn, points at something drawn) serve both and the message names which pointer failed. Two checks fall out: an axis carries at most one pointer, since two would name a chained reading and the readings compose instead; and at most one paired axis exists per (pointer, target), since a renderer takes the first match and a second would be silently unreachable. That second check tightens `reciprocal_of` too, which had the same latent looseness.

**`CHART_IR_VERSION` stays 2**, by the rule the constant documents: a reader that ignores `complement_of` sees an extra axis in `doc.axes` that no panel names, which is exactly what `return_period` already looks like to it, and draws the default reading correctly and completely. `complement_of` is omitted at its default, so no existing document's hash moves for the field itself. **The six documents that gain an axis do move**, so their hashes and any ETag keyed on them invalidate once, which is correct and harmless.

**Version skew, for the other consumer.** An older `aggregate` build calling `load_chart_doc` on a new document raises `ChartAxis carries unknown field(s) ['complement_of']`. That is the standing consequence of every additive field and the fix is the standing one: sync the library before serving.

Entry points gaining `reflect=False`: `Aggregate.plot`, `Aggregate.reins_occ_plot`, `PnL.plot`, `Severity.plot`, `Distortion.plot` and `Bounds.plot_envelope`. The last two took no reading switches at all and gain this one only, because it is the only reading a unit square declares. `Portfolio.plot` is unchanged.

Plan in `dev/done/plan-chart-reflect.md`; the app half is planned in the API repo and is not executed here.

## 1.0.0a268

**[DecL-Paren-Arithmetic] `+`, `-` and `*` join the DecL expression sub-language, legal inside parentheses.** Every slot that takes a number has always accepted an expression (`/`, `**`, `^`, `exp`, parentheses), evaluated eagerly in the transformer. What was missing were the three operators that let a user write a computed exposure, a grossed-up premium, or a severity normalizer inline. The author's target program now builds:

```
agg TEST (4 + 3*2) claims (100_000/(1-.25)) premium 1000 xs 0
    sev (exp(-1 * .4**2/2)) * lognorm .4 poisson
```

Ten claims, a premium grossed up for a 25% expense ratio, and the lognormal mean normalizer `exp(-sigma^2/2)` that makes the severity mean 1. The exposure head takes it, so does the informational premium suffix (`[FYI-Premium-Exposure-Head]`, a266), and so do layers, frequency parameters, severity scales and shifts, reinsurance clauses, collars, and ranges (`dfreq [(1+0):(3*2)]`).

**The parenthesis requirement is the design, not a limitation.** Outside parentheses those three characters already mean something: `*` is the severity scale operator and the portfolio homogeneous multiplier, `+` is the severity shift and the portfolio sum, `-` is severity negation, and `NUMBER` absorbs a glued leading minus so `[1 -2]` lexes as a two-element vector. Admitting the operators into bare expressions would make `2 * 3 * agg.X` genuinely ambiguous and would destabilize vector lexing. Inside parentheses none of that exists, because a parenthesis holds exactly one expression and never a list, so the only viable reading of `(1-.25)` is the subtraction. Grammar-wise this is a paren island: a new `sum` / `product` ladder reachable only from `atom`'s parenthesized alternative, with three one-line transformer methods beside the existing `atom_*` group. Precedence runs `()`, `exp`, `**` and `^`, `*` and `/`, `+` and `-`; `**` stays right-associative (`(2**3**2)` is 512), everything else left-associative.

**Three behaviors are documented rather than changed.** A minus glued to a literal is part of that literal and binds tighter than `**`, so `(-.4**2)` is +0.16 where Python's `-.4**2` is -0.16; negate a computed value with `(-1 * x)` or `(0 - x)`, which is what the normalizer above does. Parentheses hold one expression, never a list, so `(1 -2)` is -1 while `[1 -2]` keeps its two-element reading and `[1 - 2]` stays a parse error. A percentage literal keeps its percent reading only when used literally, so `(50% + 1)` is 1.5 and a computed value in the `po` placement position reads as an absolute amount. A unary-minus production was considered and rejected: it would make `(-3)` lex two ways and make the value of `(-3**2)` depend on which lexing won, exactly what the ambiguity guard exists to keep out.

**Round trip: expressions evaluate at parse time, so the canonical text shows the evaluated literal.** `spec_to_decl` and `format_program` are unchanged, because the writer only ever sees floats. `(4 + 3*2) claims` decompiles as `10 claims` and `(100_000/(1-.25)) premium` as `133333.33333333334 premium`. That is the long-standing canonical, not verbatim, contract, the same one that collapses `exp(.5)` to its float. The consequence worth naming: running `format_program` over user source replaces a formula with its value. Formula-preserving reformatting would be a token-level text tool, not a spec change.

**Nothing existing moved.** No new terminals: `PLUS`, `MINUS`, `TIMES` and `EXP` were already in the grammar and already carried `_TERMINAL_LABELS` entries, so `test_grammar_sync` passes with zero edits to the Pygments lexer, `agg.sublime-syntax`, or the labels, and the web app's `decl-keywords.json` is owed nothing (`+ - * ( )` are punctuation to a highlighter). The spec snapshot regen moved only the two new corpus lines, every pre-existing program parsing byte for byte as before. The ambiguity sweep gained twenty paren-math programs, each asserted to have exactly one parse; a new `tests/test_paren_arithmetic.py` pins the evaluation ladder, the gate negatives (bare `4 + 3*2 claims`, `[1 - 2]`, `(1 2)`, `(1 +)` all still parse errors) and the unparser collapse. Corpus: `_test_suite.agg` sections E and F, `decl-testers.agg` section PA.

**Docs edited, rebuild pending** (house rule keeps the Sphinx build outside the verification loop): a new Numeric Expressions section in `docs/4_dec_Language_Reference.rst` stating the operator set, the precedence ladder, the paren gate and the three documented behaviors; the grammar listing regenerated; and the stale note in `features.rst` that said `1 + 2` does not parse corrected to the paren form.

---

## 1.0.0a267

**[BS-Window-Formats] the grid-sizing exhibit prints its window like an amount, not like a float's repr.** `bs_window_df` gained `W` and `coverage` at a254 (`[BS-Window-Widen]`, round 6 ask 2) and the widening carried no formats, so the exhibit's deciding columns printed as raw float text (`31.30829289650644` for a window edge), which is exactly the curation hazard that ruling warned about, one door down. The registration now carries `BS_WINDOW_FORMATS` (exported alongside `SHARPEN_FORMATS`): the window edges `x_min` / `x_max` and the width `W` print as amounts (`,.0f`, matching how `SHARPEN_FORMATS` treats the same loss-axis quantities), `bs` in significant digits (`,.4g`, since a bucket size can be dyadic-fractional), and `clipped` in scientific notation (`.2e`, an estimated far-tail mass where only the exponent separates a good row from a perfect one). `coverage` is a string upstream (`'1-1e-12'`) and needs nothing.

**The caption now says what the frame shows**: each method's window is [x_min, x_max] with width W = x_max - x_min, and coverage is the fraction of the distribution the window holds. Raised in the app's testing notes (`aggregate_api/dev/api-punchlist.md`, the window formatting bullet); the app has drawn this leaf through the exhibit route since its a71, so it picks the polish up with no change on its side.

**Test surface.** The exhibit snapshot regen moved exactly the ten `bs_window/*` entries (five kinds by two perspectives), verified key by key against the prior snapshot; `test_simple_exhibits_are_passthroughs` now expects the `formatters` kwarg on `bs_window` and caption-only on `tail_behavior`.

---

## 1.0.0a266

**[FYI-Premium-Exposure-Head] the claims and loss sizing heads carry an informational premium: `5 claims 20000 premium`, `500 loss 650 premium`.** The head sizes the law exactly as it did without the suffix; the premium books `exp_premium` and the loss ratio back-fills from the realized expected loss, landing in the `('meta', 'prem')` and `('meta', 'lr')` rows of `stats_df`. This is the natural spelling for a book rated one way and modeled another: a quote sheet carries a claim count or a loss pick plus a premium, and previously the premium had nowhere to go short of hand-converting the head to `premium at lr`. Author request, 2026-08-12.

**The engine already knew how.** `Aggregate.__init__` has always reconciled in this order: claim count trumps loss, then a premium with no loss ratio back-fills the ratio from the realized expected loss (`if _pr > 0: _lr = _el / _pr`). `Aggregate(name, exp_en=5, exp_premium=20000)` worked before this change; the DecL surface was the only missing piece. The Sparre-Andersen renewal head `T years at r rate` established the informational-premium concept (a premium that never sizes the count); this extends it to the ordinary heads. Two free synergies fall out: `pnl … inherit premium` reads an engine's FYI premium, and the `@` / `*` builtin scaling operators already scale `exp_premium`.

**Grammar: a new optional `fyi_premium` suffix on the claims and loss alternatives of `exposures`, and no new keywords.** No `and` joins the clause, a considered decision: `and` in DecL combines terms into one item (reinsurance layers in one cession, expense terms in one group, reinstatement groups) and nothing is combined here, while comma-as-whitespace already gives the natural reading `5 claims, 20000 premium`. Earley-safe: PREMIUM anchors the clause and the only competitor after a claims or loss head that begins with numbers is a layers clause, whose own anchor is XS (pinned by the ambiguity sweep over the new corpus lines). The sizing head `premium at lr` is untouched, and combining the two readings (`5 claims 20000 premium at .65 lr`) is a deliberate parse error at the `at`. Zero new terminals means nothing is owed to `_TERMINAL_LABELS`, the Pygments or Sublime colorers, or the web app's `decl-keywords.json`.

**The invariant: an informational clause never changes the law.** Both amounts must be scalar, validated in the transformer with specific messages: a vector premium against a scalar head would broadcast into extra components and change the distribution; a scalar premium against a vector head would repeat per component and misreport the total. `test_fyi_premium.py` pins the invariant directly (`agg_density` identical with and without the suffix) alongside the reconciliation, the layer-clause disambiguation, the comma spelling, `inherit premium`, both refusals, and the writer round-trip.

**Surface details.** The suffix takes its own `as` label, landing on the new `premium` site of `label_map` (an open dict; `_INTERIOR_LABEL_KEYS` gains `_premium_label`). The writer renders the suffix after the head (`5 claims 20000 premium as booked`, byte-exact round-trip); the premium *sizing* form is now detected by `exp_premium` together with `exp_lr` rather than `exp_premium` alone. A bivariate reuses the shared `exposures` production and `BivariateAggregate` already accepts `exp_premium`, so the suffix rides through it without special-casing.

**Docs and corpus.** `features.rst` gains an [FYI premium] subsection under New DecL elements; the grammar reference regenerated; the `Aggregate.__init__` docstring for `exp_premium` no longer claims a loss ratio is required. Corpus: `_test_suite.agg` F.Expos04 / F.Expos05 (spec snapshot recaptured, purely additive), `decl-testers.agg` section FYI (seven stress lines including the pnl-engine interplay).

---

## 1.0.0a265

**[Allocation-Default-Linear] the pentagon surface honors `allocation_method`, and `ccoc` allocates on an unbounded book at a finite anchor.** Reported from the app's Pricing pane: Calibrate at `p < 1` on an unbounded book warned `analyze_distortions: skipping ccoc` and served an Allocate subtab with no `ccoc` row, while Calibrate and Evaluate both carried it. `dev/done/plan-fix-unbounded-ccoc.md`.

**The ruling being enforced is not new.** Linear became the default natural allocation at `1.0.0a17`, which flipped `Portfolio.price` to it and created `Portfolio.allocation_method` (default `'linear'`, validating setter, clears the augmented cache) as the source of truth. `price` honored the member. The rest of the pentagon-era surface, built later, hardcoded `allocation='lifted'` as a keyword default and never consulted it, so `apply_distortion`, `augmented_df`, `pricing_at`, `pentagon_at`, `analyze_distortion`, `analyze_distortions`, and therefore `CalibrationResult.pricing_df` and the `pricing.allocate` exhibit, all read the other surface. The `allocation_method` docstring already claimed it drove "`price` (and downstream readouts)"; the downstream readouts were never wired.

Each of those now takes `allocation=None` and resolves the sentinel to `allocation_method`. `apply_distortion` resolves first, so the cache key records the method the frame was actually built under and never the sentinel. `analyze_distortions` resolves once at the top, so its skip test, its sweep and its cache snapshot agree on one answer. Author ruling 2026-08-12: linear everywhere, with no per-distortion special casing, and lifted available by explicit request only.

**Why the skip was right and is now unnecessary.** The lifted tail share integrates the distorted law beyond the anchor all the way to the essential supremum, where a mass distortion's atom lives. On an unbounded support that mass collapses onto the FFT truncation row, so at a `p = 0.99` anchor with `r = 10%` roughly ninety percent of `g(S(a))` is the artifact and the split reads the truncation row rather than the book (numerics-3 G6). A finite anchor trumps unboundedness for every total level number, which is what a260 and a261 encoded, but not for beta. The linear split has no such term: alpha is the objective tail share, the builder already blanks the beta columns under mass plus unbounded, and the alpha based margin feeding `unit_capital_at` is stable. So the family flows through the sweep like any other, and the skip and its `UserWarning` now fire only under an explicit `allocation='lifted'`, with wording that says so.

**Visible number changes, on the default path.** Totals do not move: the two allocations differ only in the tail share splitting the last layer, so `exag_total`, `price_pentagon`, `evaluate`, `calibrate_distortions` and `reins_price_df` are identical before and after. Per unit cells do move on any readout that used to inherit the lifted default: the premium split wherever the anchor leaves a tail to divide, and the capital split generally, since `_unit_capital_at` integrates the layer margin across the whole range. On a bounded book that is the a17 intent arriving late rather than a fix. The committed exhibit snapshots for `pricing.allocate` on the calibration of the two unit `EX.Port` fixture were regenerated for exactly this reason and no other: `L`, `M` and `P` per unit are unchanged at that anchor, `Q`, `a`, `PQ` and `ROE` move, and the two `CalibrationPortfolio` keys are the only ones in the file that differ.

**What deliberately does not change.** The lifted refusal in `build_augmented` and the unconditional raise in `Aggregate.apply_distortion` (one distribution, nothing to allocate) both stand. `guard_unbounded_anchor` still refuses `p = 1` on an unbounded risk. `allocation_diagnostics` keeps its `surface='lifted'` default: it is a diagnostic about layer curves and the surface is the subject, not a default to be inherited. `price_stand_alone` is untouched and, despite the total leg now building, still raises on a mass distortion plus an unbounded book: it prices each unit's own `Aggregate` stand alone first, and that raise is unconditional.

**Tests.** Everything captured on the lifted surface now asks for it by name, so no baseline was regenerated: the PEG regression fixture and `capture_peg_baseline.py`, the numerics-3 pre-change lock (`pricing_at` and `pentagon_at` legs), and the `tests/baseline` corpus capture and check (the `augmented__*` and `pricing_at__*` legs; its `price()` legs always passed a method explicitly). New coverage names the symptom: a mass family prices through the default sweep with no warning and finite per unit cells; the `pricing.allocate` exhibit serves `ccoc` in every stat slice on an unbounded book (new fixture `EX.UnbPort`, mirrored in `decl-testers.agg`); the allocated `ccoc` total column closes on the shared `calibration_df` octet; and setting `allocation_method = 'lifted'` puts the skip back, so the member drives the default in both directions. `test_cache_keys_coexist` now pins that the sentinel and the name it resolves to are one cache entry rather than two.

**Downstream.** The API's `test_pricing_exhibits.py::test_a_skipped_distortion_is_a_warning_and_not_a_failure` pins today's skip and re-pins to `ccoc` present with no warning; `run_calibration`'s docstring describes the skip as expected. Both belong to the API's own version bump after it syncs.

---

## 1.0.0a264

**[Bounds-Envelope-Jump] the maximum envelope jumps at the origin, and the grid now carries the jump instead of smearing it.** A display bug, reported from the app's Bounds leaf: the drawn upper edge of the envelope sat below the `ccoc` curve near `s = 0`, which is a visible contradiction, because `ccoc` is a member of the admissible set and the maximum envelope has to lie above every member of it.

**Nothing was wrong with the arithmetic.** `cloud_df` was right, `min` and `max` over it were right at every grid point, and `max >= min` held everywhere it was evaluated. The defect was one missing sample point.

`p_knots` includes `p = 1`, where `TVaR_1(min(X, a)) = a`, so the cloud legitimately contains brackets `(p_lo, 1)` whose distortion carries an **atom at zero**, exactly as `ccoc` does. Every such column steps from `0` to its weight the instant `s` leaves zero, so the maximum envelope is genuinely discontinuous at the origin, rising to the largest of those weights. The minimum envelope is continuous there, since brackets with `p_hi < 1` carry no atom. `s_grid` was a bare `linspace`, so the first cell held no point, and any consumer joining `(0, 0)` to the first grid value drew a ramp where the admissible set has a cliff. That understates the upper edge across the whole first cell by the full height of the jump. Sub-pixel in a three inch figure, which is why it never showed in a notebook, and the whole picture in a zoomable one.

`Bounds.s_grid` now splices one point at `JUMP_EPS = 1e-12`, immediately right of zero, and is `n_s + 1` long. This is the discipline `Distortion._build_grid` already applies to a distortion's own grid, with the same constant and for the same reason ("ensure a knot at eps so trapz captures the jump"); the cloud is a set of biTVaRs, so it wants the same treatment. The band, `max_envelope` and the `charts.chart_envelope` document all read the grid, so one point fixes all three.

The jump height is now exact and is **not** a tolerance: it equals the mass of the `ccoc` calibrated to the same premium and anchor, both being `M / (a - L)` by the pentagon identity. The `(p_lo = 0, p_hi = 1)` bracket *is* the CCoC distortion, and it is the admissible distortion that loads the first infinitesimal of probability hardest.

**Bracket weights are clipped to `[0, 1]`,** folded in because the new knot surfaced it. A weight outside the interval is not a convex combination and so does not name a distortion. `p_star` is a root found to `xtol = 2**-17` and is itself spliced into `p_knots`, so brackets whose `p_lo` **is** `p_star` came out a few thousandths negative; unclipped they extrapolate away from `p_hi` rather than interpolating toward it, which dragged the minimum envelope below the curve at `p_lo`, and at the jump knot below zero. Clipping to zero leaves exactly the `TVaR_{p_lo}` curve, which prices to the premium to the same root tolerance. Envelope effect away from the origin is about `1e-6`.

Shapes: `s_grid`, `tvar_hinges`, `cloud_df` and `min_envelope_hinges` all gain one row against `n_s`, and the docstrings and one shape pin say so.

Tests: the jump knot exists and is sorted; the jump height equals the `ccoc` mass; weights and cloud values lie in `[0, 1]`; `max_envelope >= min_envelope` on 20,001 points, which used to fail inside the first cell because a linear interpolation was being compared against a fitted concave distortion. On the chart side, the band leaves the origin vertically, and **every calibrated distortion lies inside the band as a renderer draws it**, checked against the polyline rather than against the grid values, since that is the difference this release is about. That last test needed a fixture holding the calibration and the bounds at one asset level; `test_chart_bounds.py`'s existing fixture calibrates at `q(1)` and lets `Bounds` default, so the two disagree about `a` and its five curves are not the admissible set of its band.

Not addressed, and unchanged: with no asset cap the minimum envelope collapses onto the risk neutral diagonal to within FFT noise, since for an unbounded risk there is always a bracket arbitrarily close to the identity. The band is then a tautology rather than a bound, and `ccoc` still sits outside it. That wants a ruling on whether `Bounds` refuses an unbounded anchor the way `[Unbounded-Anchor-Guard]` refuses `p = 1`, or labels it.

---

## 1.0.0a263

**[Pricing-Exhibits] the Pricing pane's three tables are library exhibits, dispatched on the calculation that produced them.** Phase L5 of `dev/plan-pricing-exhibits.md`, the last of the LIB half. It closes round 6 item 4 for the pricing leaves.

Three new exhibits, in a new class module `exhibits/_pricing.py`:

- **`pricing.calibrate`** on `CalibrationResult`: one block, `distortion_df`, identical under both perspectives. The per family receipt with no adjustment, deliberately small.
- **`pricing.allocate`** on `CalibrationResult`: the calibrated target spread over whatever the source has to spread it over. A `Portfolio` spreads across units (`calibration_df` then `pricing_df`), a reinsured `Aggregate` across the views of its cession (`reins_price_df`), and an `Aggregate` with neither has one distribution and nothing to spread, so its allocation story is the single calibration row.
- **`pricing.evaluate`** on `EvaluationResult`: one block, `evaluation_df`, with the business reading of `gini_p` and of the `status` column under INSURER.

**Registration needed no framework change.** Dispatch is plain `singledispatch` on `type(obj)`, so giving a calibration a type is the whole of what was required to make it an ordinary exhibit: ruling `[Pricing-Keyed-On-Result]`. `available_exhibits` on a **built** object is untouched, so a client's capability payload does not move because these exist.

**The block list is a property of the source as well as the perspective,** which is the fullest exercise of `[Perspective-May-Restructure]` in the package. Two readings are worth calling out.

On a book, INSURER replaces the tall raw `pricing_df` with four blocks, one statistic at a time, units across: `LR`, `P`, `PQ` and `ROE`. The raw frame stacks eight statistics for five distortions into one honest, unreadable table; a reader asking which distortion to use compares one statistic at a time.

On a cession, INSURER is **narrower** than RAW, the first exhibit where that is true. RAW serves every view the object can price, `ceded` among them. INSURER drops the ceded rows, stars the calibrated basis in the index (`gross` reads `gross*`), and appends a `basis less view` difference row per distortion with its ratios recomputed on the differenced amounts. A ceded price is what the layer is worth to whoever writes it, a reinsurer's reading; the cedent's reading of the same cession is the difference between two of its own programs, which is the allowance for reinsurance in the rate. Ruling `[Difference-Is-A-Perspective]`.

**Formats and captions moved upstream from the app.** `PENTAGON_FORMATS`, `CALIBRATION_FORMATS`, `DISTORTION_FORMATS`, `EVALUATION_FORMATS` and the stat slice formats and titles are now library constants beside `SHARPEN_FORMATS`. They are statements about what the library's own numbers mean and how they read.

**The RAW invariant holds over a calculation,** which was the thing worth checking. A RAW block still names an attribute on the dispatched object that returns a real public frame; that the frame is computed on first access rather than stored is an efficiency question and not a contract one. The a253 invariant sweep extends over the result fixtures with no change of wording.

Tests: the result fixtures join `tests/test_exhibits.py`'s shared object set, so they inherit the RAW invariant sweep, the caption sweep, the named index sweep and the committed `canonical_dict` snapshots (14 new keys, no existing snapshot moved). New alongside them, **the standing envelope contract is now asserted on this side of the wire**: every served block, over every object, exhibit and perspective, reconstructs through `gt.TableDoc.model_validate` to the same `hash`. An ETag that did not survive the round trip would make every cached table on the other end unverifiable.

---

## 1.0.0a262

**[Reins-View-Pricing] the layered quote speaks the same octet as every other price in the library.** Phase L4 of `dev/plan-pricing-exhibits.md`.

**Breaking: `reins_price_df` returns the canonical pentagon octet.** Its columns were `a, el, bid, ask, margin`; they are now `L, M, P, Q, a, LR, PQ, ROE` (`aggregate.pentagon.PENTAGON_STATS`). `el` is `L`, `ask` is `P`, `margin` is `M`, and `Q` and the three ratios are completed through `complete_pentagon`, the one implementation every frame emitter routes through. These tables sit on a screen beside the pentagon and the per unit allocation, and a number must not change name between two tables a reader compares.

**`bid` is dropped** (author, 2026-08-12: too confusing). Two adjacent columns whose difference is a bid ask spread invited reading the spread as the answer, and the answer is the ask. It is no longer computed.

**An unlimited quote has no capital, and now says so.** With neither `p` nor `a` the price is unlimited, which is the natural quote for a cession already bounded by its own terms. `a` reads infinite, which is what was asked for, and `Q`, `PQ` and `ROE` are blank rather than reading a 0% return on infinite capital.

**`price_pentagon` and `price_pentagon_ex` gain `reins_view=`** on both `Aggregate` and `Portfolio`. Both legs of the anchoring triple come off the named view: its own quantile and its own limited expected loss, matching what `calibrate_distortions(reins_view=...)` already did. A preview line can now answer on the basis a reinsured reader chose rather than on whichever view the program happened to hold.

**`calibrate_distortions` accepts `lr=` as the alternative target,** exactly one of `coc` or `lr`. The loss ratio is converted to a cost of capital through the pentagon at the resolved anchor, which is the only place the conversion can be made: it needs `L` and `a`, the two numbers only the distribution knows. A client that converts for itself is holding a copy of the library's accounting.

**A loss ratio that leaves no capital is refused,** which a cost of capital target cannot do. `coc` puts the premium between the expected loss and the assets by construction; `lr` can put it above the assets, and then the implied cost of capital is negative and the calibration downstream chases a target above the essential supremum and reports shapes that did not converge. The `ValueError` names the implied premium, the assets and the expected loss.

**One naming fix, on the buyer's side of a cession.** `reins_price_df` and `Portfolio._reins_view_density` both said that differencing two view prices is not the price of the cession, which is true and was the whole story. Per `[Difference-Is-A-Perspective]` they now also say what the difference **is**: the allowance for reinsurance in the rate, the cedent's own reading, computed deliberately. The `ceded` row is the seller's price for the same layer, and the gap between the two is the negotiation.

---

## 1.0.0a261

**[Evaluate-Asset-Anchor] the round trip closes: calibrate, price, evaluate at the same anchor recovers the calibration.** Phase L3 of `dev/plan-pricing-exhibits.md`, and the plan's own acceptance criterion.

`Aggregate.evaluate` and `Portfolio.evaluate` accept at most one of `p=` or `a=`, resolved on the distribution being evaluated (so under a `reins_view` it is that view's quantile, not the object's own). With an anchor the position measured is `premium - min(X, a)`, an obligation with assets behind it. Without one, nothing changes: the position is measured against its whole distribution, which stays the default and is the unlimited reading.

**Why it did not close before.** `calibrate_distortions` solves `rho_g(min(X, a)) = P`: its layer integral runs over `[0, a)` and stops there. `evaluate` integrated to the top of the FFT grid instead, so the two were solving different equations, and the distance between their answers was whatever the tail beyond `a` was worth. Capping the loss makes the two equations one equation, and the shapes now agree to solver tolerance on all three of the author's reference programs.

The cap places the tail rather than dropping it: the mass above `a` moves to an atom at `a`, which is where a position with `a` behind it settles. Dropping it would renormalize and change every moment.

**`ccoc` joins the acceptability panel when an anchor is supplied** (ruled 2026-08-12). It was excluded because its closed form needs an asset level the acceptability question did not supply, and with an anchor that reason is gone. The unanchored default keeps today's four, since the stated reason still holds there. `EVAL_FAMILIES_ANCHORED` is the new default set. `ccoc` recovers the cost of capital it was calibrated to exactly, because the closed form is shift invariant and so is the frame the solve runs in.

**One naming fix, exposed by `ccoc` now appearing in both receipts.** A family with no declared `param_name` read `r` in the calibration receipt and `param` in the acceptability panel, which is two names for one thing. Both now go through `_param_name` and read `r`. The four families that declare a name (`a`, `lam`, `b`, `p`) are unaffected.

The `p = 1` guard from a260 reaches `evaluate` in this phase, as the plan schedules it.

`PnL.evaluate` is deliberately unchanged (decision 4): each ledger row is its own position and the right anchor semantics there deserve their own discussion.

---

## 1.0.0a260

**[Unbounded-Anchor-Guard] `p = 1` is a statement about a maximum, and an unbounded risk has none.** Phase L2 of `dev/plan-pricing-exhibits.md`.

`p = 1` used to resolve, silently, to the last grid point carrying mass. On a bounded law that is the maximum loss and is exactly right. On an unbounded one it is an artifact of `log2`: double the grid and the asset level, the premium, the capital and every ratio built on them all move, while the risk has not changed. There was no guard anywhere in the chain.

`guard_unbounded_anchor` in `_pricing.py` now runs at every entry point that turns a probability into a capital level: `calibrate_distortions`, `price_pentagon`, `price_pentagon_ex` and `reins_price_df`, with `evaluate` joining them at a261. `p == 1` on an object reporting `bounded` False raises `ValueError` with a message written to be read on the wire, since the app shows it to a reader as a sentence.

**The test is the tail classification, not the density.** `bounded` comes from `tail.TailClass`, which is why it can tell a bounded law from an unbounded one that ran out of grid; a density frame cannot, and that confusion is what the guard exists to end. On a `Portfolio` it is the worst-of over units, which is the right test there. The `total` row of a portfolio `tail_behavior_df` shows realized grid extent and is not it.

**Only the exact spelling is refused.** A `p` below 1 that happens to land on the top bucket is a statement about that grid and the honest answer is a wider one. `p = 1` is the only value that cannot mean anything other than "the maximum".

Three ways past the guard, each saying something different: `a=` names the level (`a=obj.q(1)` reproduces the old number, with the caller having asked for it), a `p` below 1 asks a question the distribution can answer, and `obj.bounded = True` certifies a support the heuristic could not prove.

**Two existing callers moved, both deliberate and both in the Bounds fixtures.** The IME 2022 bounds work anchors on the top of the realized grid on purpose, and wrote it `p=1` on Poisson books. They now write `a=port.q(1)`, which is the same number with the choice made visible. Nothing in the library itself called `p=1`.

---

## 1.0.0a259

**[Pricing-Result-Objects] a calibration is a receipt, and a receipt knows what it was written about.** Phase L1 of `dev/plan-pricing-exhibits.md`, the joint plan with `aggregate_api`. Review notes for the whole LIB half are in `dev/plan-pricing-exhibits-LIB.md`.

**Breaking: `calibrate_distortions` returns a `CalibrationResult` and `evaluate` returns an `EvaluationResult`,** on `Aggregate`, `Portfolio` and (for `evaluate`) `PnL`. Both were bare `DataFrame`s. The frames are unchanged and are the first attribute of each result: `.distortion_df` and `.evaluation_df`. The stored attributes did not move either, so `obj.calibrate_distortions(...)` followed by `obj.distortion_df` reads exactly as before and only code that used the *return value* as a frame has to change.

The reason for the break is that a frame cannot be dispatched on. The pricing exhibits (`pricing.calibrate`, `pricing.allocate`, `pricing.evaluate`) are `singledispatch` registrations like every other exhibit, and what they register on is the result: ruling `[Pricing-Keyed-On-Result]`. Without a type there would have to be a second channel through the exhibit machinery for calculations that are not stored on an object, and there is now none.

**A result carries the position, not just the shapes.** A calibration knows its `coc`, its resolved `p` and `a`, which of the two the caller fixed (`anchor`), the `kind`, the `names` requested and the `reins_view` it was fitted on. An evaluation knows the `premium` it was measured against, the view, and (from phase L3) the asset anchor. A panel of breakeven shapes read without its premium is a table with nothing to hold it to.

**The allocation of the target is a lazy public frame on the result.** `CalibrationResult.pricing_df` allocates across the units of a book (`analyze_distortions` at the calibration anchor with the result's own distortions) and `CalibrationResult.reins_price_df` allocates across the views of a cession; each is computed on first access and cached, and each raises `AttributeError` naming the reason when the source has nothing to spread the target over. This is what keeps the RAW exhibit invariant intact over an exhibit built on a calculation: a RAW block still names an attribute on the dispatched object that returns a real public frame, and that the frame is computed on demand is an efficiency question rather than a contract one.

**Every result borrows its source's identity.** The five result classes (the two new ones and `AnalyzeDistortionResult`, `AnalyzeDistortionsResult`, `PricingResult`) gain `_source` and, through `SourcedMixin`, the four names an exhibit reads off a dispatched object: `name`, `label`, `_title_name` and `_relabel`. So a served pricing exhibit is titled `Calibrated distortions: BasicBook` rather than `Calibration: CalibrationResult`. A result with no source answers `None` and relabels to the identity rather than raising.

`PricingResult` also gains the `_relabel` at construction that its two siblings already applied, so `Portfolio.price` returns a display copy with unit labels like `analyze_distortion` and `analyze_distortions` do.

`qd` grew a branch for both results, printing the frames they carry rather than a dataclass repr.

---

## 1.0.0a258

**[Joint-Surface-Contract] the joint surface says what grid it is, what it is a reduction of, and what it left out.** The LIB half of `dev/plan-3d-plot.md`, sections 5.1, 5.2 and 5.4 to 5.6. Review notes, and the five places the code and the plan disagree, are in `dev/plan-3d-plot-LIB.md`.

**The coordinate was the block's last fine cell, and is now its first.** The display reduction filed a block covering `[a, a + k * bs)` under `a + (k - 1) * bs`. Two consequences, and the first is not an approximation: a distribution supported on `[0, inf)` was **reported as starting at `(k - 1) * bs`**, which on one real joint, whose y axis reduces 128 fine cells into a single 512-wide bucket, meant a support starting at 508 and a mean reading 508 against a true 23.8. The coordinate is now the block's first fine coordinate, its low edge, the convention the fine lattice is already in, and the document declares it through `edge`.

A residual bias survives and is intrinsic, not a defect: a display cell holds `k` atoms and labeling it with any single coordinate loses their spread, so a mean taken against the display lattice sits up to one bucket **low** where it used to sit up to one bucket high. That is what `moments` is for, and a consumer that wants a mean reads it rather than integrating the picture.

**The window is chosen before the reduction, because cropping cannot recover resolution that was already averaged away.** `window` (default 4: keep `q(1e-4)` to `q(1 - 1e-4)` of each marginal) is measured on the fine lattice through `GridDistribution`, the crop is taken, and only then is the block factor chosen from the cropped extent. Order matters by a factor of fourteen: the same window on that Lomax axis is 232 fine cells taken first and a handful of 512-wide display cells taken last. The factor is a power of two **per axis, chosen independently**, and the crop is aligned outward to whole blocks, because an uneven last block breaks the uniform step every interpolation downstream divides by and puts the wide cell exactly where the tail is. The low edge snaps to a reachable zero within 5% of the window's width, and never on a signed axis, whose lower bound is real. A floor of 8 cells per axis keeps a grid there when a heavy tail puts both window edges in one bucket.

`detail` (default 128) is a **ceiling**, not a target to straddle, since it is also the knob a deployment caps to bound a payload. One exception, found by testing: the cell floor outranks it, because powers of two do not reach every count and a 528-cell window offers 9 cells or 5 when 8 is asked for. The overshoot is bounded by 15 cells whatever the target.

**What the block now carries.** Both lattices as origin, step and count rather than as arrays, which is normative: the grids **are** arithmetic sequences, so sending `x0 + i * dx` sends a derived quantity and invites a reader to wonder whether it might not be uniform this time. Beside them `bs` and `k`, which say what the grid is a reduction *of*; `window` with the depth, the outer edges and the mass kept; the **exact** marginals on the display lattice, from the object rather than integrated off a windowed joint, which is a different and worse curve; `moments.mean` from the fine lattice; and `deficit`. Every value is a mass per display cell, marginals included: mass is exact, is what a block sum preserves, and lets a consumer form either reading with one division. Mixing the two is not cosmetic. A prototype that integrated a marginal as if it were a density while normalizing a conditional into a real one had them disagree by a factor of 1024.

**`z` again, encoded.** `SurfaceZBlock` carries base64 of little-endian bytes under a declared `dtype`, so the default can change without a format change: `f32b64` (the default, exact to seven figures), `f64b64`, and `u16log12b64` at two bytes a cell and four significant figures. Not float64 by default, and the intuition points the wrong way here: the low mantissa bits of an FFT-built density are genuine digits that no compressor touches, so a float64 pipeline measures **twice the size of the JSON text it replaces**. Code 0 of the quantized form is reserved for an exact zero, which a real joint is 14% to 59% of; spreading all 65,536 codes over the twelve decades would put a floor of spurious mass under most of the grid. `encoding='json'` emits no block at all and leaves the plain arrays as the payload.

**Additive, so `CHART_IR_VERSION` stays 2.** The `x`, `y`, `z` arrays are emitted beside the lattice form for one release; dropping them is the breaking change and is what bumps to 3. `load_chart_doc` learned the new fields in the same commit that emits them, so the round trip contract holds over the nested block, hash and object both. `display_log2` is replaced by `detail`; nothing passed it.

---

## 1.0.0a257

**[Joint-Density-Clip] small in magnitude is noise; large and negative is not.** Both 2-D FFT paths in `bivariate.py` finish by zeroing round-off dust, and both did it with `density[abs(density) < 1e-15] = 0`. That predicate came from `utilities.remove_fuzz`, whose docstring justifies its two-sidedness explicitly for the signed P&L columns of a frame, and it does not transfer. A joint density is non-negative even where its **support** is signed, which is the ordinary case for a P&L axis, so the `abs` was zeroing the harmless negatives and **preserving** exactly the ones that mean something. A large negative cell is a broken construction, not fuzz.

Both sites now route through `_clip_density_fuzz`, which clips the dust, then warns naming the count and the worst cell before clipping any survivor. The warning is the point; the clipping is housekeeping around it. **The deficit is a free second detector**: both callers take `1 - density.sum()` afterward, so clipping a genuine negative raises the sum and drives the deficit below zero, which every validation frame already shows.

**The floor is now a fraction of the mass the grid carries** rather than an absolute constant, because the depth an absolute constant reaches moves with the grid: the same `1e-15` sits 12.4 decades under the peak on one joint and 10.7 on another. A 2-D FFT accumulates round-off in proportion to the total it sums, not to the tallest cell it produced, so the sum is what the floor is anchored to. On a normalized joint that reproduces the absolute constant exactly, which is why no number in the suite moves.

Robustness, not a live bug: measured at 1024², 2048² and 4096² nothing trips it. The tests therefore **inject** a negative rather than wait for one, since a test that waits for nature passes forever without testing anything.

The first of the LIB items in `dev/plan-3d-plot.md` (its section 5.3), and the one independent of the rest.

---

## 1.0.0a256

**[Reins-Insurer-Orientation] the layering analysis turns over, and splits in two.** `reins_stats_df` is built measures down the stub and layers across the columns, which is right for the library: a layer is a natural column of an analysis, and the frame is built once for every consumer. It is not how a reinsurance reader reads. Under INSURER on an `Aggregate` the layering analysis is now two blocks with **layers down the rows**, gross then the layer then ceded then net, because that is the comparison and the eye makes it down a column rather than across a row.

**Two blocks, because they are two kinds of thing.** `reins_layer_terms` is the contract: the share of a limit over an attachment, the chance a loss reaches it and exhausts it, and the loss on line. `reins_layer_moments` is the consequence: what the layer does to the frequency, severity and aggregate, each with its mean, CV and skewness. Read as one table those two make a column header change meaning half way down. An aggregate stage leaves the frequency and severity columns blank, having changed neither.

Three blocks under insurer against RAW's two, per `[Perspective-May-Restructure]`. Neither new block has a frame behind it, which is what an INSURER block is for: both are readings of `reins_stats_df`, which RAW still serves whole in its own orientation, noncentral moments included.

**The `Portfolio` reading is unchanged**, and deliberately: a book's `reins_stats_df` is indexed by view and measure with units across and has no layer axis at all, so there is no layering to turn over. It keeps its noncentral moment drop and its total row flags.

**Closes round 6 item 1** (`dev/note-from-aggregate-api-round-6.md`), the last of the four table asks and a live regression in the app's Reinsurance / Stats pane until it syncs. The app deleted `_reins_stats_transposed` at a71 ahead of this landing, so nothing waits on it there except the reading. One snapshot moves.

---

## 1.0.0a255

**[Sharpen-Exhibit] the grid probe says what it scored and why.** The bucket probe left an audit that no exhibit served, so the one leaf that showed it wrote its own prose about the library's own search, which is the kind of second opinion this whole line of work is removing. `sharpen` is the twelfth registered exhibit, on `Aggregate` and `Portfolio`, available once `sharpen_df` is present. Not gated on `update`: a probe implies one.

**One block raw, two under insurer**, which is the smallest exercise of `[Perspective-May-Restructure]` and deliberately so. RAW is `sharpen_df`, every grid tried with its working. INSURER leads with `score_grid`, steps in bucket size down and steps in log2 across, then the same walk. Ruling `[Sharpen-Grid-Is-A-Reading]`: `score` is a **column** on the audit used for deciding, not a second fact, so the grid is a reading of a published frame and no new frame is owed. The unstack is the whole translation, and it is the right one: a reader comparing grids wants the two step axes on the two axes of a table, not twenty columns of working with the deciding number buried among them.

The formats come with it, in `exhibits.SHARPEN_FORMATS`. The six `u_` columns are relative errors against the analytic moments and run from about 1e-7 to a few percent, so they read in scientific notation: at a fixed `.4f` a good cell and a perfect cell both print `0.0000`, which is exactly the comparison the table exists to support.

**Closes round 6 item 3** (`dev/note-from-aggregate-api-round-6.md`), and with it the last two entries in the app's `tables.FORMATS` that describe a library frame. Deliberately **not** snapshotted: the audit carries a `seconds` column, so a captured document would differ on every run.

---

## 1.0.0a254

**[BS-Window-Diagnostics] the bucket window frame carries the two columns it exists for.** `bs_window_df` published the candidate windows and withheld `W`, the window's width, and `coverage`, the fraction of the distribution it holds. The leaf's whole question is "is this grid big enough", and those two are the answer, so a reader had a list of candidate windows and no way to compare them. Both were curated onto the private `_bs_window_df` as expert material, which was the wrong call about which columns carry the meaning. Ruling `[BS-Window-Widen]`: widen the published frame rather than register the exhibit against the private one, because if the curation dropped what answers the question then the curation was wrong.

Published columns are now `applies`, `selected`, `x_min`, `x_max`, `W`, `bs`, `log2`, `log2_need`, `coverage`, `clipped`, `note` on an `Aggregate`, and the same list without the two method flags on a `Portfolio`. `coverage` stays the **string** it is (`'1-1e-12'`, `'E[N]-adj 1-1e-12'`), carrying a precision a float cannot and saying which of two things was held to it.

**The stray `level_0` was an unnamed index, and it was not alone.** `pd.DataFrame(rows).T` leaves `index.name` as `None`, which reaches a served table as a column headed `level_0`. Named at construction so the private frame and the published one agree: `method` on an Aggregate. The sweep for the same bug across every served block found two more. `Portfolio.tail_behavior_df` was unnamed where its `Aggregate` twin has been `component` all along, and is now `unit`. `PnL.legs_df` had a bare positional index and is now named `leg`, the index staying positional because legs are ordered. `Portfolio.bs_window_df` had the opposite fault, an index named `unit` that then had five rows appended to it which are not units, four combine candidates and the realized grid; it is now `source`, which is what every row answers. A standing test asserts no served block carries an unnamed index level, under either perspective.

**Closes round 6 item 2** (`dev/note-from-aggregate-api-round-6.md`), which is a live regression in the app's More / Window pane until it syncs. Exhibit snapshots re-captured: eighteen entries move, the eight `bs_window` ones gaining two columns and the ten others gaining a stub header.

---

## 1.0.0a253

**[Exhibit-Perspective-Contract] a RAW block is exactly one public frame, and the exhibit layer stops inventing frames.** The invariant is now written into the `aggregate.exhibits` docstring and onto `Perspective`, and swept in `tests/test_exhibits.py`: every RAW block of every exhibit available on every fixture names an attribute on the object and serves that frame, in the frame's own orientation, with no split and no dropped rows. `INSURER` is the only perspective that may restructure, and ruling `[Perspective-May-Restructure]` says it may change the **block list** and not merely each block's content, so `meta['blocks']` is a property of the (exhibit, perspective) pair. `economic_ratios` already exercises it, two blocks raw and three under insurer, and a client assuming parity would be wrong about that today.

**`PnL.walk_df` and `PnL.evaluation_df` are new public frames.** The invariant is a forcing function before it is a contract, and it found one violation in shipped code on the day it was written: `economic_waterfall` computed its two tables inside the exhibit layer and no public frame stood behind them. Ruling `[Waterfall-Frames-Are-Owed]`: promote, rather than write the invariant with an exemption in it. The margin walk in currency is `walk_df` (the expected result, the step's own 1-in-100 state, and the state conditional on the whole book's, so the reader can see the diversification benefit as the gap between the two) and the same walk as ratios is `evaluation_df` (premium and margin spent, combined ratio, margin over standard deviation, and return on the capital each 1-in-100 reading calls for). A notebook reader can now reach what only the app could see.

Nothing is newly estimated: both frames are arithmetic over quantities the P&L had already computed, which is what made the exhibit layer look like a reasonable home for them. `WATERFALL_RETURN_PERIOD` moves to `_pnl.py` with the frames it describes, unchanged at 100 and still a constant rather than a keyword. The two blocks are renamed `walk` to `walk_df` and `evaluation` to `evaluation_df`, so a block still says which frame it is, which is the convention every other exhibit already followed.

Part of `[Exhibit-Official-Channels]` (`dev/plan-exhibit-official-channels.md`). Exhibit snapshots are untouched: block names live in the envelope's `meta`, not in the `TableDoc` a snapshot captures.

---

## 1.0.0a252

**[Chart-Doc-Reader] a served chart document has a way home.** The two registries were the same shape end to end except at one point: `greater_tables.TableDoc` is a pydantic model, so an exhibit block reconstructs from its own wire form and redraws without knowing an `Aggregate` ever existed, while `canonical_dict` went out and nothing read it back. The obvious guess failed, `ChartDoc(**canonical_dict(doc))` raising `AttributeError: 'dict' object has no attribute 'id'`, because panels, axes, series and marks arrive as plain dicts and `__post_init__` validates them as dataclasses. `load_chart_doc(d)` closes it, in `charts/ir.py` beside `canonical_dict` and exported from `aggregate.charts`.

**The round trip is the contract**: `doc_hash(load_chart_doc(canonical_dict(doc))) == doc.hash`, checked over every document the registry emits, each through a genuine `json.loads(json.dumps(...))`. That covers all three payload shapes, explicit coordinates, the lattice form, and the nested `SurfaceData` a bivariate document carries. Reconstruction is equal to the original as a value, not merely equal in hash.

**Why the library owns the reader rather than each client.** `canonical_dict` drops any field equal to its default, so a reader is a statement about what those defaults are: panel `read_axis` and `aspect` never reach the wire, axis `kind` never reaches it, and `invertible` appears only where it is true. A client writing its own is writing this build's default table down a second time, and a default that moves here then moves silently over there. `ir_version` is validated in `ChartDoc.__post_init__`, which makes the reader the one place a wire document's version is negotiated, and a hand rolled reader that passes the field through by accident is the one that accepts a version it cannot read. `SurfaceData` is the only nested payload, so a reader that forgets it fails on exactly the largest and least exercised documents. A member carrying a field this build does not know is refused by name rather than as a bare `TypeError` about a keyword argument.

**Closes round 6 item 6** (`dev/note-from-aggregate-api-round-6.md`). Nothing changes app side when it lands, and nothing about what is emitted changes: `canonical_dict` was already complete and already deterministic, only the way back was missing. Urgency came from the 3D plan, whose documents are the largest chart payloads yet, and where a quiet coercion in a hand written reader, a tuple that came back a list, is not something anyone catches by looking at the picture.

---

## 1.0.0a251

**[PnL-Consideration-Rounding] a sized P&L premium is a number someone would write down.** `pnl_program` sizes an uninherited premium as expected loss over the target loss ratio and wrote the quotient out at full precision, so the derivation dropped `1428.5840984231345 premium` into a program the reader is meant to read, keep and edit: sixteen digits derived from an input of "about 70 percent". `_pnl_consideration` now rounds where the number is produced. No decimals above 100, two at or below, which is the author's rule and has one joint in it on purpose: above 100 the cents are noise against the quote, at or below they are the number, and a rule with more joints stops being predictable from the outside.

**Only a sized premium rounds.** An inherited one is a number the program already stated, and restating it differently would make the wrap disagree with the exposure clause it wrapped. Rounding at the source rather than in the writer keeps the value and its printed form the same fact: `decl_writer._fmt_num` drops the trailing zero of an integral float, so the program reads `1429 premium` and the spec carries 1429.

**Closes round 6 item 5** (`dev/note-from-aggregate-api-round-6.md`), the last of round 5's asks. `aggregate_api` rounded this downstream in `post_pnl` as a string rewrite of DecL in a service, which is not where it belongs; that pass is idempotent, applying the rule to an already round number, so it can be deleted on the sync that picks this up. The rounded premium moves the realized loss ratio by at most half a currency unit's worth, which is the price of a program a reader can keep.

The three derived P&L programs mirrored in `agg/decl-testers.agg` re-render (`80.00000000000009` to `80`, `14285.739238292414` to `14286`, `11238.555678761779` to `11239`), and `DP.OddPremium` joins the DP block as the case that must **not** round, an inherited premium stated to the cent.

---

## 1.0.0a250

**[Reins-Density-Fuzz] `reins_density_df` removes its FFT fuzz, like every other density frame. Bug fix.** `calibrate_distortions(reins_view=...)` raised a bare `AssertionError` with no message, and past that `reins_price_df` returned `NaN`. Measured over six program shapes and every view each carries: four of five shapes failed to calibrate on at least one view, `net` failed on three of five, and pricing lost four of the five calibrated families. All of it is fixed by one line.

**One frame was swept and its sibling was not.** `density_df` has called `remove_fuzz` since it was written (`_aggregate.py`, "remove the fuzz, same method as `Portfolio.remove_fuzz`"), so its minimum is exactly `0.0`. `reins_density_df` never did, so the gross / ceded / net columns kept the inverse FFT's sub-epsilon negatives: worst measured `-9.1e-17` in a bucket and `9.2e-14` summed across a view. Nothing read those columns until `reins_view=` landed at `a223`, and then everything downstream of a cumulative sum broke at once, because negative mass makes `1 - cumsum` tick back **up**. A survival that increases is not a survival, so `_calibration_survival` stopped on the exactness assertion it has carried since the initial commit in 2018, and past that an `S` above 1 sent `(1 - S) ** shape` to `NaN` for every fractional shape family (`ph`, `wang`, `dual`) and put a mass distortion's weight on a negative atom (`ccoc`, reported as an atom weight of `-9.1e-02`). Only `tvar` came through, being piecewise linear.

**The assertion was correct and is untouched.** It asserts a property that follows from "this is a pmf", and the fix restores that property at the one place it had stopped holding, rather than relaxing the test that noticed. The default calibration path never read the unswept frame and never failed, in this release or any before it.

**Numbers move in the last digits, and only where they were already zero.** De-fuzzing changes every reinsurance moment at the ulp level, so `tests/data/exhibit_snapshots.json` is re-captured: the four `reins` entries shift their raw values, and ten displayed strings change, every one of them at pico or femto scale (a skew reading `118.177f` now reads `112.038f`). No quantity a reader would act on moved. This is the direction `remove_fuzz` exists for, since far-tail fuzz is weighted by `x**k` and corrupts exactly these moments.

---

## 1.0.0a249

**[Single-Placement-Keyword] there is one way to write a partial placement, and it is `po`. Breaking.** DecL offered three keywords for the same layer: `50% so 100 xs 0`, `50% po 100 xs 0` and `50% of 100 xs 0`. `so` and `of` are retired. `po` (*part of*) is the survivor, unchanged in meaning: the leading quantity picks the reading, a percentage is the share directly and a bare number is an absolute amount whose share is `amount / limit`.

The three rules computed the identical `(share, limit, attach)` tuple, so nothing is lost. `so` is an ordinary English word and therefore a poor reserved word; `of` forked the Earley parse against the `net of` that precedes it, and deleting it removes that ambiguity outright. `po` was kept because `corridor <share> po <width> xs <attach>` already speaks it, so the language now says "part of" in one voice. Retiring `so` returns it to the identifier space: `agg so ...` and `sev so ...` build, pinned by new cases in `decl-testers.agg` and `tests/test_hygiene3.py`.

**The canonical output form moved with it, which is the visible half.** `decl_writer` renders every partial placement as `<pct>% po <limit> xs <attach>` where it used to render `<pct>% so ...`. That string is what `pprogram` shows back, and `_pnl_builders._layer_descriptor` calls the same renderer, so an undeclared cover step in a P&L walk now reads `agg 85% po 1500 xs 7000`. Code that indexes an `xpnl` frame by the old literal will raise `KeyError`. A declared `as` label still wins and is unaffected.

Nothing records how a placement was written. `_PercentNumber` carries the `%` marker only as far as the clause rule, which returns a resolved fraction; the spec stores that fraction and nothing else. So the percentage form is the only form the library can emit, and a placement entered by amount round-trips to a percentage: `5 po 15 xs 5` renders `33.3333% po 15 xs 5`. Same layer, canonical spelling.

**The `po` small-share warning now names the fix.** It read "Did you mean share of?", which was useless advice even before the retirement, since a bare number under `so` did exactly what it does under `po`. It now shows both readings and the percentage form that expresses the other one. It earned its keep immediately: it found a live monograph program, `occurrence net of 0.25 so inf xs 0`, whose intended 25% coinsurance was silently a 0% placement, because a bare amount over an `inf` limit is zero.

Corpora reformatted and derived artifacts regenerated: `library.agg`, `_test_suite.agg` and `decl-testers.agg` (whose four `J.Re18*` cases collapse to two, and whose two `HY3.Of*` cases become `HY3.Po*`), the spec snapshot, `ref_include.rst`, and the generated cookbook reinsurance page. **No spec value moved**: the snapshot re-keys on the new program text and every `(share, limit, attach)` is identical, which is the check this change turns on. `agg.sublime-syntax` and `decl_pygments` drop `so` in step with the grammar, as `test_grammar_sync` requires. The web app's `decl-keywords.json`, which lives in a separate repository, needs the same edit.

Plan: `dev/done/plan-single-placement-keyword.md`.

---

## 1.0.0a248

**[PnL-Value-Type] a P&L says which sign convention it is read on, instead of leaving consumers to infer it from the class name.** `PnL.value_type` joins `Aggregate.value_type` and `Portfolio.value_type`, returning the payoff label through the same `value_type_label` helper, so a `[labels]` relabel moves all three together. `PnL.info` gains the matching row, in the same position the other two carry it.

A P&L is the one first class kind whose convention is fixed by what it is: profit is positive and loss is negative, which the name states, so the adverse tail is the low one. `Aggregate` declares its convention per object in DecL and `Portfolio` derives one from unanimous units; a P&L has nothing to declare or derive, so `_is_loss_value` is a **class attribute** rather than an instance one and every instance agrees by construction.

The fact was already true and was stated six times. `is_loss_value=False` appeared as a literal at four sites in `_pnl.py` and two in `_pnl_builders.py`, so a consumer could see the orientation on `pnl.result` but not on the P&L, and the app was left asserting a fact about a library class from outside it. All six now read the new module constant `PNL_IS_LOSS_VALUE`, which is where the convention is stated and the only place it can be read from.

`dev/FEATURES.csv` corrects the row that recorded `value_type` as absent from `PnL`, and `docs/2_aggregate_overview/info-strings.rst` gains the new row.

---

## 1.0.0a247

**[Exhibit-Row-Cap] how many rows a served block carries is the caller's question.** `build_exhibit` and the eleven convenience functions (`exhibits.summary(obj, 'insurer')` and its siblings) take a keyword-only `max_rows`, defaulting to `exhibits.MAX_ROWS` (200, `greater_tables`' own default, so nothing changes by default) and accepting `None` for the whole frame.

The counterpart to `include_raw` at a246, and deliberately the opposite call. Carrying the numbers is not negotiable, because a document that drops them destroys what no consumer can recover. Row extent is negotiable, because a preview pane and a full download want different answers and neither is more correct, and because truncating loses nothing silently: `greater_tables` records it in the block's `notes` as `Showing first N of M rows`. So `INCLUDE_RAW` is a library constant and `max_rows` is a parameter, and the line between them is whether the document can say what it did.

`max_rows` wins over a block's own kwargs, which no block sets. No exhibit is near the cap today: the longest block measured across every exhibit on an `Aggregate` and a `PnL` is 17 rows.

This is the last of the three carriage requests from `aggregate_api`'s round 5 note. The third, a `formatters={}` plus wide `float_format` passthrough for a full precision reading, is **not** implemented and is not needed: with raw values travelling since a246, a client formats from the numbers rather than asking the library to stop formatting them.

---

## 1.0.0a246

**[Exhibit-Raw-Values] a served exhibit block carries its numbers, not only its formatted strings.** `build_exhibit` now sets `include_raw` on every `TableSpec` it builds (`exhibits._core.INCLUDE_RAW`), so a body cell arrives as `{'text': '17.50', 'raw': 17.5000001}` rather than as `'17.50'` alone.

The default was `False`, which meant the library formatted each number, shipped the string and threw the number away. A consumer cannot put back what the document destroyed: it cannot sort numerically, download at full precision, or render interactively at all, since `greater_tables`' own `irToGridInput` needs a raw value for every non-string data column. Downstream that pushed clients back onto `exhibit_frames()` and pandas to fetch the numbers a second time, which forks the contract instead of using it. The formatted string is this library's *reading* of a number and the raw value is the number, and a served document owes both; each column already carried a machine readable format spec beside them, so a client can render the reading, restate it, or ignore it.

A library default rather than a caller option, on the same reasoning that keeps renderer passthrough out of the chart IR: presentation belongs to the consumer, and the consumer needs the numbers to do it. A frame builder may still pass `include_raw` in its own block kwargs and win, since the default is applied under them; nothing does.

**Nothing about formatting changed.** Every format spec, every formatted string and every caption is byte identical; `tests/data/exhibit_snapshots.json` is re-captured and the diff is provably additive, in that stripping the new raw carriage reproduces the previous file exactly across all 102 cases. Payload cost measured over every exhibit on an `Aggregate` and a `PnL` is about 1.5x, worst case 2x on the 26 row moment store, on documents of a few kilobytes.

**Breaking for anyone pinning an exhibit hash.** `TableDoc.hash`, `Exhibit.hash` and therefore the ETag a client revalidates against all move once, for every exhibit and both perspectives. `aggregate.exhibits` is provisional, so this is a minor-release change by policy.

---

## 1.0.0a245

**[View-Pair-Spread] the view-pair keyword gets its own line in the spread layout.** `netceded`, `grossceded` and `grossnet` used to be glued onto the aggregate's head line, so a spread render opened `grossceded agg Property` and put the clauses one level in. The keyword is now the block head with the whole aggregate as its single child, which reads as what it is: a view pair taken *of* an aggregate.

```
grossceded
  agg Property
    80 claims
    500 xs 0
    sev lognorm 50 cv 1.2
    occurrence net of
      50% so 375 xs 126
    poisson
```

Terse output is byte-for-byte unchanged (`_render_terse` space-joins a head back onto its children, so the single-line form still reads `grossceded agg Property 80 claims ...`), and so are round trips. `src/aggregate/agg/library.agg` still writes its four `netceded` entries in the old flat form, which parses the same and is left alone.

---

## 1.0.0a244

**[Chart-Reins] the reinsurance document is redrawn as the occurrence plot, and the last compositor goes.** `chart_reins` is rewritten to the shape `Aggregate.reins_occ_plot` draws, that method now renders it, and `plots/_aggregate.py` and `plots/_quantile.py` are both **deleted**: the occurrence plot was the quantile worker's last caller. This is the ninth and last job of `dev/plan-chart-ir.md` part two.

**Two panels that share nothing, not even a loss axis.** A per-claim loss and an annual aggregate are different quantities, and one window across both would say they were the same. The left is the gross, ceded and net severity as the treaty sees each claim, windowed by the occurrence limit because a cession is bounded by its limit. The right is the same three for the year, as a **Lee diagram**, where the old chart drew survival: a chosen probability now reads off as a loss.

**Every reading lives on the right panel**, as the author specified: log on both axes, the paired return-period reading of its probability axis, and the inversion that turns it into the distribution function. The occurrence panel reads on log and declares **no linear alternative**, because a layered severity is a spike and a tail and the linear reading of it is a spike and nothing else, so offering the control would offer a worse picture.

**Aggregate only at 1.0** (author's decision), which is a deliberate narrowing. `chart_reins` was registered for `Portfolio` since `a225` and carried a `basis` option choosing among four triples; both are gone. A book's units cede on different stages, so a book-level triple would have to pretend they cede on the same one, and an **aggregate cover is a separate contract with a separate picture**, so an aggregate-cover-only object now has no reinsurance chart rather than a misleading one. Both are restorable in a few lines and neither is guessed at here.

`reins_occ_plot(log=False, full_range=False, return_period=False, invert=False)`, returning the figure and stashing it on `self.figure`; `axs` and the worker passthrough are gone. `MAX_RETURN_PERIOD` moves into `plots/_chartdoc.py`, where it is now a renderer constant like the rest of the drawing ladder, and it stands in only where a document declares no window for its paired reading.

**`plots/` now holds no two-panel compositor at all.** What remains is bespoke by design: the scatter matrix, the sample comparison, the bounds weight contour and hull view, the affine envelope, the bivariate contours and the massive pyramid, plus `pedagogy` and `ft`.

---

## 1.0.0a243

**[Chart-Equal-Aspect] the matplotlib renderer stops treating equal aspect as a box shape and treats it as a reading.** `Panel.aspect` has been in the IR since v1 and the distortion and envelope charts have always set it, but honoring it meant no more than `set_aspect('equal')`, which squares the *box* while leaving the two axes on different ranges. That is a square drawn over a rectangle of data, and on a panel whose axes measure the same thing it misreads: a 45 degree line has to *be* at 45 degrees.

So an equal-aspect panel now gets **one window for both axes**. The top is the higher of the two, so nothing an emitter declared is cropped; the bottom is where the data actually starts, so a panel whose curves begin well inside its window no longer opens with an empty corner, which a loss window anchored at zero routinely gives. It never widens past what the emitter asked for: the data can raise the floor, never lower it. A renderer that has no use for the hint may still ignore it; this one does not.

**A square panel no longer shares its x axis.** Sharing let squareness drag a neighbour's window around to keep itself square, which is the tail wagging the dog, since the neighbour's window was computed from the data it draws. Equal aspect is the stronger statement, so it wins and the axis goes unshared.

**The portfolio's kappa panel is now square**, which is what prompted this: both its axes are losses and the total's curve is the diagonal, so the reading is each unit's slope against 45 degrees.

No baseline moves. The distortion's window is the unit square on both axes already, so merging them changes nothing, which the image gate confirms.

---

## 1.0.0a242

**[Chart-Portfolio] a portfolio emits total, units and the kappa reading.** `Portfolio.plot` draws the document `charts.chart_port` emits and `plots._portfolio.plot_portfolio` is deleted. The density and the log density were one quantity read two ways, so log is a declared reading of the one panel, and the panel that frees up is the **kappa** panel, which is the author's design and the reason a book is not an aggregate with more curves on it.

**`exeqa_i` is `E[X_i | X = x]`**: what each unit contributes when the book as a whole lands at `x`. Read up from a total loss and you get the split that produced it, and the vertical at the 1-in-200 is then the natural allocation at capital, which is why the same anchor sits on both panels. The unit curves sum to the diagonal by construction, and the total's curve is that diagonal exactly (`exeqa_total` is `E[X | X = x] = x`), so it is drawn: seeing the parts sum to it is the reading.

**The floor on that panel is measured, not chosen.** A conditional expectation *divides* by `p_total`, so where that mass is arithmetic dust the quotient is noise rather than allocation. Because the unit curves must sum to `x` exactly, the residual of that identity measures kappa's own error, and on a two-lognormal book it runs, as (median, worst): `1e-12` → (4.6e-11, 4.1e-07); **`1e-14` → (4.6e-09, 3.2e-04)**; `1e-15` → (7.3e-05, 1.2e-03); no floor → (9.9e-05, **0.52**). The cliff is real rather than gradual and `KAPPA_FLOOR = 1e-14` sits a decade above it, which is also a decade above `LOG_FLOOR`, the dust floor for a mass that is only *displayed*. The same numbers come back at `log2` 16 and 18, so the floor does not need to scale with the grid, and on that book kappa stays trustworthy out to ten times `q(0.999)`: inside the default window the floor never bites, and what it protects is the zoomed-out reading. The test re-runs the measurement rather than asserting the constant.

The kappa axis declares its window as the **loss** window, because the curves sum to the diagonal and so the tallest thing on the panel sits at the window's right edge. Without saying so the panel scales to kappa at losses far off the right of the shared axis and squashes every curve a reader can see into the bottom eighth.

Draw order is meaning: units first, **total last**, so the book sits on top of the parts it is made of, which reverses what the compositor did. Each unit draws on its **own native grid** through `unit_density`, which a windowed book does not share with the portfolio's. Units are named by their resolved label, so a client links a unit's legend toggle across both panels by name. `plot(xmax=None, log=False, full_range=False)`; `axd` and `figsize` are gone. `scatter` and `sample_compare` stay bespoke.

---

## 1.0.0a241

**[Chart-Bounds] the envelope is two panels, the cloud and all five calibrated distortions on one band.** `Bounds.plot_envelope` draws the document `charts.chart_envelope` emits and `plots._bounds.plot_bounds_envelope` is deleted. The compositor drew **three** panels and split the five distortions across the last two, `['ccoc', 'tvar']` on one and `['ph', 'wang', 'dual']` on the other. That was an accident of the order they were added rather than a reading anyone wants: the question is how the five compare, and five curves on one band answer it (author's decision, `dev/plan-chart-ir.md`).

**The band is one series, not two curves**, which is the first real use of `ChartSeries.y2`, carried since v1 and never emitted: the series *is* the region between the extremes, which is what an envelope means, rather than two curves a reader has to associate. The renderer strokes both edges as well as filling between them, because a band whose boundary cannot be seen reads as vaguer than the data.

**`ChartSeries.value` carries one number a series has as a whole**, and the cloud is what it was added for: each curve is one bracketing BiTVaR and its weight is a fact a reader asks about, so it travels as data and the renderer decides how to encode it (the house ramp, and a colorbar named by `meta['value_label']`). It is a property of the series and **not** a channel for passing appearance through: a renderer with no use for it ignores it and is still correct. A series carrying one stays out of the legend, because forty legend entries are forty names nobody asked for. Omitted at default, so `CHART_IR_VERSION` stays 2.

Both panels are equal-aspect unit squares and neither axis declares a log or a full-range reading: the unit square *is* the window, so a zoom-out on it would do nothing and a log reading of it says nothing. The renderer learned to size a row of equal-aspect panels as a row of squares, which constrained layout was otherwise collapsing to slivers. The second panel is **omitted** rather than drawn empty where the priced object carries nothing calibrated, which a `Bounds` built on a bare `Aggregate` does not.

`plot_envelope(n_resamples=0)`, returning the figure and stashing it on `self.figure`. `axs`, `alpha`, `lim` and `title` belonged to the drawing and are the renderer's now; `distortions=` goes with the split it selected between. `plot_weights` and the hull view stay bespoke, the first because a level set over `(p_lo, p_hi)` is a grid panel nobody has asked for and the second because it reads private engine state with no public frame behind it.

**Note for `AllocationBounds` and `PricingBounds`:** they are `_HullEngine` subclasses built on a `Portfolio`, not on a `Bounds`, and carry a vertex table (`_T`, `_A`, `_hulls`) rather than a `cloud_df`, so this chart does not serve them. Their drawing is `plot_hull_bounds`, still bespoke, and it needs an emitter of its own.

---

## 1.0.0a240

**[Chart-Invertible-Lee] a panel may declare that its axes exchange, and a Lee diagram does.** Mechanically an exchange is a transpose; what it *performs* is an inversion, which is the name it takes: a quantile function and a distribution function are inverses, so a Lee diagram read the other way round **is** the cdf, drawn from the same pairs. `Panel.invertible` declares it, `plot_chartdoc(invert=True)` and `plot(invert=True)` do it, and the aggregate, P&L and severity Lee panels all declare it. It is a fact about the quantities and not about the drawing: exchanging a density's axes says nothing, because mass against loss does not invert.

**Two things fell out rather than being written**, and both say the earlier rules went in the right place. The renderer's ladder reads its drawing off the **axes** rather than the series role, so an inverted quantile function picks up the right-continuous step of a cdf on its own, where it had been drawing the left-continuous step of a quantile. And the paired return-period reading is resolved by axis id, so it rides along on whichever axis it was attached to. The only real work was a band, which fills between two edges of one coordinate and so turns with it, and a mark, which names the axis it sits on.

**`Panel.inverse_title` names the other picture**, because it is a different picture and naming it is the library's job: the Lee panels carry 'Distribution function'. A renderer with no name to use says the title is inverted rather than inventing one, and naming an inverse on a panel that is not invertible is refused, since nothing could ever use it. It joins `human_strings`, so `tex` stays total over it.

`CHART_IR_VERSION` stays 2: a reader ignoring `invertible` draws the panel in the orientation the document already named as its default, which is correct and complete.

---

## 1.0.0a239

**[Chart-Severity] a severity emits its density and its Lee diagram, and the compositor goes.** `Severity.plot` draws the document `charts.chart_severity` emits and `plots/_severity.py` is deleted. Four panels become two, and **neither collapse loses anything**, which is why this was a redesign rather than a rewire.

The density and the log density were one quantity read two ways, so the density panel declares both scales and the reader picks, exactly as on the aggregate. The distribution and the Lee diagram are **inverses**: the same curve with its axes exchanged, so neither carries information the other does not, and only one of them needs drawing. The Lee orientation is the one kept, because it is the one that pairs with a return-period reading, which is how a tail is actually quoted. The curve costs nothing to build and is exact: the grid is *already* a quantile grid, inverted from log-spaced exceedance probabilities, so `(F, loss)` is the pair the grid was computed from rather than an accumulation of it.

**A severity says it is continuous**, because it is: it is the one genuinely continuous law this library holds, a frozen scipy variable with no `xs` and no discretization, which happens in `Aggregate` and not here. So the renderer draws a line and no rung of the atomic ladder applies. A **discrete** severity has no density at all, so it reads as probability mass, says so in `meta['ordinate']`, and is atomic; the two readings are never mixed in one document.

`Severity.plot(n=None, log=False, full_range=False, return_period=False)`, returning the figure and stashing it on `self.figure`. `axd`, `figsize` and `layout` are gone with the mosaic they described, and `quantile_x='return'` is `return_period=True`. The loss axis gained a `full_range` out to the 1-in-100,000 loss, where the window still crops at the 0.1% exceedance.

---

## 1.0.0a238

**[Chart-Payload-Weight] a document stops writing out what it can state, and `CHART_IR_VERSION` moves to 2.** Two facts about this library's data, neither of them compression: both are statements about the data that happen to save a great deal of space. The aggregate document for a `log2 = 16` book falls from **7.4 MB to 5.4 MB**, and for a lattice book from **1.6 MB to 0.05 MB**, a 32-fold cut. Nothing rounds, thins or samples: every value a document carried before, it carries still.

**A computed grid is a lattice.** `ChartSeries.x_lattice` and `y_lattice` carry `(start, step, count)` in place of a list of evenly spaced numbers. An aggregate lives on `k * bs`, so four series over one loss grid were four copies of it, 2 MB of a 7.4 MB document. The lattice form is taken **only where `start + step * i` reproduces the values exactly**, checked rather than assumed, so a trimmed, collapsed or genuinely irregular coordinate (a severity's quantile grid, a cumulative probability) falls back to the explicit form on its own and no caller has to reason about it. `ChartSeries.x_values` and `y_values` expand whichever form is present, and the renderer reads those.

**An empty stretch of that grid is one fact, not thousands.** A lattice severity leaves most buckets with no probability at all, and a run of them says nothing beyond where it starts and stops, so the interior goes. The endpoints **stay**: delete a run outright and a stepped or straight line bridges the hole and draws mass where the law has none, so keeping one zero either side makes the collapse exact under every rung of the renderer's ladder rather than only under stems. Collapsing is applied exactly when it pays, and the arithmetic says when rather than a threshold someone picked: an untouched lattice ships `count` masses plus three numbers, a collapsed one ships `k` masses **and** `k` coordinates, so it pays when `2k < count`. On a smooth book that is never, and on a lattice book it is always.

**The version moves to 2, and this is the first real application of the rule** written into `ir.py` at `a233`: it marks the point where a reader that ignores what it does not know would draw something *wrong*. A reader that does not know `x_lattice` sees a series with no coordinates at all, so it must refuse the document rather than draw an empty panel, which `ChartDoc` now makes it do by name. Every hash changes with it.

---

## 1.0.0a237

**[Chart-PnL] a P&L delegates to the aggregate emitter over its signed result, read from the low tail.** `PnL.plot` draws the document `charts.chart_pnl` emits and `plots._aggregate.plot_pnl` is deleted. A P&L's result is a `GridDistribution` like any other, so the chart is the aggregate's: the same two panels, the same one outcome axis read by both, the same quantile curve. The emitter is a delegation, not a second drawing, over the shared `outcome_doc` builder that `chart_agg` now also goes through.

**Four things differ, and every one is a fact about what the outcome means.** The outcome axis is **signed**, so its window is the two-sided quantile crop rather than being anchored at zero: a loss window reads from zero because the mass at and near zero is real, and half a P&L's outcomes are on the other side of it. For the same reason it declares `scales=('linear',)` and offers no log reading at all, which is a declaration doing exactly its job; the ordinate still declares one. The adverse tail is the **low** end, so `meta['return_period_map']` is `reciprocal` and the reader interrogates the shortfall, the same branch `PnL.tail_periods_df` already takes: ask for the return-period reading and the 1-in-100 and 1-in-250 anchors land on 100 and 250 exactly. And break even at zero is a **reading**, not decoration, so it is a mark in both panels, vertical where the outcome is on x and horizontal where it is on y.

There is no severity companion, a P&L being an accounting result rather than a compound of one. The panels gain the Lee diagram the compositor never drew: it drew a mass panel and a cdf panel, and the cdf is the Lee curve seen from the other axis, so nothing is lost and the tail is now readable at a chosen probability.

`PnL.plot(log=False, full_range=False, return_period=False)`, returning the figure and still stashing it on `self.figure`; `axd` and the canvas `**kwargs` are gone, as on `Aggregate.plot`.

---

## 1.0.0a236

**[Chart-Distortion] a distortion draws through its own document and the compositor goes.** `Distortion.plot` now renders the document `charts.chart_distortion` emits, and `plots._distortion.plot_distortion` is deleted. The deletion is licensed by a measurement rather than an argument: the two paths had been pixel identical since `a209` and were re-measured at RMS 0 immediately before, so nothing about the picture changed here.

**The argument list was decided one argument at a time**, which is what a conversion is for. `both` survives as the semantic option it always was, under the emitter's spelling `dual`, because whether to draw the dual is a statement about the pricing and not about the drawing. `ax` survives. `xs`, `n`, `c`, `c_dual`, `size`, `plot_points` and the `**kwargs` passthrough are realization and are gone; `plot_points` had been dead since `ConvexDistortion` was removed. The method returns the figure rather than the axes, matching the other converted plots.

**`scale='return'` is gone with them**, and this is the one capability removed rather than relocated. The log-log unit square is a *reading*, so under `[Chart-Declared-Readings]` its home would be `scales=('linear', 'log')` on both axes, and the plan's own statement of the principle uses this exact case as the counterexample: a log reading of a heavy tail is meaningful, a log reading of a distortion's unit square is not. Nothing in the library, the docs or the test suite called it. It comes back as two declared scales the day that reading is wanted, and the picture will then be the same one.

`plot_distortion_affine` stays bespoke, as planned: it overlays the TVaR decomposition's affine lines on the curve, which the schema cannot express, so it is listed as bespoke rather than half-expressed. It and the five `pedagogy` call sites move to `dual=`; the pedagogy bounds figure loses a `lw=1` it was passing through to matplotlib.

---

## 1.0.0a235

**[Chart-Aggregate] an aggregate emits density and Lee, and the log panel becomes a reading of the first.** The first conversion that *replaces* a compositor rather than shipping alongside one. `Aggregate.plot` now draws the document `charts.chart_agg` emits, through `plots.plot_chartdoc`, and `plots._aggregate.plot_aggregate` is deleted. One set of semantic decisions, two renderers, no third path.

**Breaking, and deliberate: `Aggregate.plot()` draws a different picture.** Three panels become two. The old middle panel was never a third reading of the book, it was the first panel read on log, so it is now a declared reading of the density panel, whose axes both offer log. The Lee diagram keeps its panel, because interrogating a probability and reading back a loss is a different question and not a rescaling. The author reviewed the new figure before the baseline was generated from it.

**The signature changes with it.** `plot(xmax=None, log=False, full_range=False, return_period=False)`, returning the figure and still stashing it on `self.figure`. `quantile_x='return'` is now `return_period=True`, the reading the document declares rather than an argument threaded down to a drawing worker; `axd`, `figsize`, `max_return_period` and the `**kwargs` ride-through are gone, since a document is realized by the renderer's own argument list. `reins_occ_plot` is untouched and still forwards to the quantile worker.

**The ordinate is a probability mass, not a density.** The continuous branch used to divide by `bs`. A discretized aggregate *is* the distribution here, which is what `support='atomic'` says and what the renderer's ladder draws, so the ordinate is the mass at each atom, `meta['ordinate']` says so, and the app already drew it that way. The y numbers therefore scale with `bs`.

**One rule replaces each of the old branches.** Discrete or continuous is gone as a branch: the renderer's ladder picks stems, steps or a line from the declared support and the room on screen, which is what `a214` built it for. The compositor's zero-mass anchor row is gone, a drawing artifact that once shipped a bug of its own; instead the emitter trims the quantile curve to the support at **both** ends, so a signed book's Lee curve starts at the worst outcome that can actually happen rather than dropping to the grid's left edge at `p = 0`. The ordinate window is the aggregate's own peak, widened for a companion within `COMPANION_HEADROOM` of it and clipping one that dwarfs it, which reproduces what the compositor did in its two separate branches.

The document carries four marks with their labels, the mean and the 1-in-200 at full weight on the density panel and the 1-in-100 and 1-in-250 faint on the Lee panel, all read from `tail_periods_df`, so a browser draws the same lines from the same numbers instead of inventing them. The loss axis is **one** axis, the density panel's x and the Lee panel's y, so a window moves both.

**`charts.build_chart_doc(obj, name, **options)` and `charts.primary_chart(obj)`**, module functions mirroring `exhibits.build_exhibit`, so no first-class class gains a method and the `self.approximate` shadowing hazard cannot recur. `build_chart_doc` resolves the registry entry, checks availability, dispatches and stamps the content hash, which stops being every emitter's job. `primary_chart` answers which chart is an object's *own* picture, a question `available_charts` cannot: an aggregate's severity is a component of it and its reinsurance is a view of it, and a landing page needs to know which one is the aggregate. `register_chart` gained `primary=`, and registry values are now a `ChartEntry` named tuple rather than a bare pair.

**Renderer growth, all of it in terms the schema already had.** A quantile function draws as a left-continuous step, because it is the cumulative seen from the other axis, and the renderer reads that off the axes rather than the series role, so it needs no per-chart instruction. A paired reading re-slices its panel, so the companion axis follows the data instead of holding a window computed for the other reading, which is the relim-and-autoscale the quantile worker always did. A log view floors at the decade under the smallest value above `LOG_FLOOR`, so one dust value at 1e-17 no longer opens six empty decades.

**Legends restyled, at the author's request:** smaller, and placed in the emptier upper corner rather than always upper left. The corner is computed from the drawn values, the taller half of the window pushing the key away from itself, so a density family takes the upper right and a monotone family the upper left, one rule for every chart. This moves the committed `distortion` baseline, which had been pixel identical since `a209` and was re-measured at RMS 0 immediately before; the image gate now covers `agg` too.

---

## 1.0.0a234

**[Chart-Tex-Totality] every human-facing string in a chart document carries both forms.** `ChartDoc.tex` was documented as a partial lookup with a fallback. It is now **total**: the plain string and its typeset form are both written, a plain word mapping to itself, and a missing entry is an emitter bug rather than a document saying "this string has no typeset form". The analogy is alt text in HTML. You write both because they serve different consumers, matplotlib reading one and ECharts the other, and you do not make either consumer guess or derive.

**`complete_tex(doc, typeset=None)` is how an emitter satisfies it** without writing a line of identities by hand: it fills every string the document exposes, using the typeset forms supplied and the string itself elsewhere. A `typeset` key the document does not expose raises, because it is either a typo or a string that stopped being emitted, and either way a renderer would go on drawing the plain form while the map claimed otherwise. **`human_strings(doc)`** is the set the contract is checked over, the document title, each panel title, each axis label, each series name and each mark label, deduplicated because the map is keyed by the string a reader sees.

All four emitters now return through `complete_tex`, and a test sweeps every registered chart and asserts the set difference is empty. The renderer keeps its fallback to the plain string, restated in the docstring as a net under a bug rather than a licensed state.

**Nothing drawn changed**, and the distortion image gate confirms it at 0, pixel for pixel: an identity entry renders the string it already rendered. Document hashes change again, since `tex` is content. The related point stands and costs nothing: the plain form does not have to be ASCII, and `ǧ(s)` remains the working example.

---

## 1.0.0a233

**[Chart-Declared-Readings] an axis declares the scales and ranges it may be read on, and a panel the forms it may take.** The third deliberate reopening of the signed-off chart IR, after `support` at `a214`, and the first job of part two of `dev/plan-chart-ir.md`. The principle is the one `scale` already stated, widened from one reading to the set of them: **which readings a quantity admits is a fact about the quantity, not about the drawing.** A log reading of a heavy tail is meaningful; a log reading of a distortion's unit square is not. So the document says which readings exist and a caller picks one, rather than every consumer inventing a per-chart list of buttons.

Four declarations. `ChartAxis.scales` is the scales an axis may be read on, with `scale` staying the default; `ChartAxis.full_range` is the whole extent, offered as the alternative to `suggested_range`, which it therefore requires, and carried as numbers rather than as a flag because the extent of a log axis with an exact zero, or of a survival curve floored at `LOG_FLOOR`, is not the naive min and max of the series and working that out is emitter knowledge. `ChartAxis.reciprocal_of` was drafted at schema time and never emitted; it now has a stated contract, that the paired axis sits in `doc.axes`, is not named by any panel, points at an axis some panel does draw, and by its presence offers the return-period reading. `Panel.kinds` is the realizations a panel supports, so a joint density read flat or in relief is **one document declaring two realizations** rather than two registry entries to keep in step by hand forever.

`scales` and `kinds` fill themselves with `(scale,)` and `(kind,)` when omitted, so a consumer never handles `None`, and both are always serialized: a singleton must read as "fixed" rather than as an absence to interpret. That is the polarity rule the `support` bug taught, applied from the other side.

**`meta['return_period_map']` says how a paired axis is computed**, `reciprocal` for `T = 1 / v` and `complement` for `T = 1 / (1 - v)`, defaulting to the literal reciprocal the field name promises. It is needed because the return period of a *non-exceedance* axis of a loss is `1 / (1 - p)` and not `1 / p`, while a survival axis, and the shortfall probability of a signed P&L, both take the reciprocal directly. One fact about the document, so it lives in `meta` rather than on either axis.

**`plot_chartdoc` gained the switches that select among the declarations**: `log`, `full_range`, `return_period` and `kind`, in the canonical control order the app also follows. Each acts on **every** axis or panel that declares the reading and on no other, so a document that declares nothing draws its one reading whatever it is asked for, and a caller never has to know which chart it is holding. `log_z` is deleted and `log` covers it. A log view of a window whose low end is an exact zero drops to the decade under the smallest positive value drawn, which is renderer knowledge: the emitter cannot know a reader will ask for log, and a fixed epsilon would crop or pad by orders of magnitude depending on the book.

**The capability pattern improves.** A panel that declares a realization matplotlib draws natively gets it, rather than the 2-D projection with `(projection)` stamped on the title as a confession, and `strict=True` stops raising for it. Degradation is now what happens when a panel offers nothing this renderer can draw, which is what it always should have meant.

`meta['z_log_ok']` retires: the surface pilot's z axis declares `scales=('linear', 'log')` instead, which is the same fact in the place that owns it. **Every document hash changes**, since `scales` and `kinds` are always serialized. That is a content address changing when content changes, and clients re-capture fixtures. `CHART_IR_VERSION` **stays 1**, and the rule for when it would move is now recorded in `ir.py` next to the constant: the version marks the point where a reader that ignores what it does not know would draw something *wrong*. A reader ignoring all four of these draws the default reading, which is correct and complete.

No baseline moves: nothing about any default reading changed, and the distortion image gate still lands at 0, pixel for pixel.

---

## 1.0.0a232

**[Tweedie-Live-Object] `Aggregate.as_tweedie()` reports an aggregate's reproductive Tweedie parameters.** Returns a `TweedieParameters` named tuple, `(p, mean, dispersion)`, satisfying `variance = dispersion * mean ** p`, or `None`.

**It recognizes, it does not merely remember.** A Tweedie with `1 < p < 2` *is* a compound Poisson distribution with gamma severity, so any aggregate of that shape has reproductive parameters whether or not anyone typed `tweedie`. `10.05 claims sev gamma 0.0995 cv 0.0709 poisson` and `10.05 claims sev 0.0005 * gamma 199 poisson`, the two long-hand entries in `library.agg`, both report the same `(1.005, 1, 0.1)` as the keyword does. A declared triple is returned verbatim; anything else is derived from the engine by running `tweedie_convert` in its `(lambda, alpha, beta)` to `(p, mu, sigma^2)` direction.

**This is the deliberate opposite of what the unparser does**, and the split is the point. `as_tweedie` is generous because generosity there costs nothing. The writer renders from provenance alone, so those two long-hand programs still read back long-hand: rewriting an author's program into a spelling they did not choose is not its job.

`None` is returned for anything that is no longer a bare compound Poisson-gamma: reinsurance or a layer at either level, a limit or attachment, a weighted or mixed severity, a location shift or splice, a reflected severity, a limit profile, a zero-modified or truncated frequency, and any method-of-moments `approximate` fit, whose engine is a fitted single severity on a fixed count rather than the requested compound at all.

The named tuple splats into the analytic class, which is where the exact series density, the characteristic function and the dual live:

```python
from aggregate.tweedie import Tweedie
tw = Tweedie(*a.as_tweedie())
```

**No display surface changed.** `qd` and `_repr_html_` do not name a distribution family for any other object and there was no reason to make Tweedie the exception.

The monograph's `tweedie` section is rewritten around the keyword surviving rather than around inspecting its expansion, and picked up a repair while it was open: its three-statement `build_many` program had one DecL statement per line with no `;` and no blank line, which stopped being legal when `[DecL-Newline]` landed. Plan, with its execution record, in `dev/done/plan-tweedie.md`.

---

## 1.0.0a231

**[Tweedie-Round-Trip] the `tweedie` keyword survives the parse.** It used to be a one-way rewrite. `agg A tweedie ...` was expanded at parse time into its compound-Poisson-gamma equivalent and nothing downstream learned a Tweedie had been declared, so `pprogram` printed `10.050251256281404 claims / sev 0.0004999999999999894 * gamma 199.00000000000426 / poisson` and the word `tweedie` was gone. The declaration is now recorded on the spec and renders back as written.

**Breaking: the argument order is now `tweedie <p> <mean> <dispersion>`.** It read `<mean> <p> <dispersion>`. `p` is the shape parameter, it selects the member of the family, and the standard notation is `Tw_p(mean, dispersion)`. Everything else in the module was already p-first, `Tweedie.__init__`, `tweedie_convert`, `__repr__` and `to_series`, so this makes one order out of two. An old program does not silently build a different distribution: the clause now rejects `p` outside the open interval `(1, 2)`, which is the only range where the compound Poisson-gamma representation exists, with a message naming the order. `tweedie 1 1.005 0.1` reads `p=1` and is refused; before, the same numbers reached `tweedie_convert` and raised a bare `ZeroDivisionError` from `alpha = (2-p)/(p-1)`. The one case that can still misread is an old program whose *mean* happened to lie in `(1, 2)`.

**The author's `note{}` is no longer destroyed.** The transformer synthesized a note of its own, `Tw(p=1.005, mu=1.0, ...) --> CP(...)`, and overwrote whatever was written; `tags`, `hints` and `doc` survived, only `note` did not. Every `tweedie` entry in `library.agg` had lost its authored sentence this way, so `build.recipe('TweedieSimple').note` returned machine text. The synthetic note is deleted outright, not repaired. It carried a defect of its own, an unbalanced `CP(`, the gamma scale printed twice under two names, `lambda` formatted to a different precision than `alpha` and `beta`, and it was the only Aggregate-facing string containing Greek, so reading `a.note` raised `UnicodeEncodeError` on a cp1252 Windows console.

**How it works.** The parser records the declared parameters under a private `_tweedie` key, a `TweedieParameters` named tuple, beside the expansion it already produced; the engine is unchanged and still sees an ordinary poisson x gamma. `decl_writer` renders the clause from that key. This is provenance, not recognition: an aggregate written the long way carries no `_tweedie` and still renders as its author wrote it, because the unparser is the inverse of the parser and not a canonicalizer that rewrites people's programs into a spelling they did not choose. `_tweedie` is a private `Aggregate.__init__` argument because a parsed spec is splatted straight into that constructor at eleven call sites with no choke point to intercept it at; it is dropped from `_spec` when unset, so `_spec_hash` is unchanged for every object that is not a Tweedie.

**`Tweedie.__init__` takes its reproductive parameters positionally.** The keyword-only marker is gone, so `Tweedie(*params)` splats. The additive pair stays keyword by convention. Purely additive: every existing call site already passed by name.

Three `library.agg` entries and one `_test_suite.agg` line leave their round-trip exemption lists, so `UNPARSER_EXEMPT` drops from seventeen to fourteen and `_FIDELITY_EXEMPT` is now empty. Closes `[Unparser-Reference-Gaps]` item 2 in `dev/TODO.md`, both halves. `TweedieParameters` is exported from `aggregate.tweedie`; `Tweedie` itself stays submodule-only. Plan in `dev/done/plan-tweedie.md`; its second phase, `[Tweedie-Live-Object]`, follows at a232. No grammar change, so the DecL reference is unaffected.

---

## 1.0.0a230

**[Reflected-Loss-Severity] a reflected severity is now legal under plain `sev`, and clamps at zero like every other negative-support severity.** `sev 10 - lognorm 1.5 splice [0 10]` used to be rejected outright, with a message directing the user to `ssev`. That was inconsistent: the library already accepts `sev 10 * norm + 5` and `sev lognorm 5 cv 1 - 10`, both of which reach below zero, builds them as ordinary non-signed losses, clamps the negative part through the layered-loss transform, and returns correct clamped moments. Reflection was the one shape singled out.

The rule is now one rule. Under `sev` a reflected body is built like any other severity whose support reaches below zero: `x < 0` clamps to `0`, the severity stays non-signed, and layers, attachments, and occurrence reinsurance work normally. Under `ssev` nothing changes; it still keeps the negative support.

**This matters most where the reflection never goes negative.** `10 - (X | 0 <= X <= 10)` lives on `[0, 10]`, because the splice applies to the underlying law and the reflection composes on top of it. Declaring that with `ssev` was the only route before, and it dragged in the whole signed code path: identity layering, so an occurrence layer was silently ignored, plus the two-sided grid and the signed `Portfolio` combine. It is now an ordinary bounded loss.

**A reflection that does reach below zero warns.** `ReflectedSeverityClampWarning` names the clamped mass, so a 0.01% tail and a wholesale truncation read differently, and points at `ssev`. It fires only when both conditions hold, reflected *and* actually below zero, so the bounded case above stays silent. In the limit the whole law clamps: `sev -lognorm 1.5` has support `(-inf, 0]` and is a point mass at zero, with a warning. Reported through `warn_once`, keyed on the severity name, shift, and splice window.

**Breaking, narrowly.** A program that relied on `sev <reflected>` raising now builds instead. Nothing that parsed before parses differently.

**[Splice-Moment-Defect] a spliced signed severity reported unspliced moments.** `_apply_lb_ub` swaps `cdf`, `sf`, `pdf`, `ppf`, `isf`, and `support` on the frozen RV but not `moment`, and both `_apply_reflect` and `_apply_signed` read `moment` off it. So `ssev 10 - lognorm 1.5 splice [0 10]` reported a severity mean of 6.9198, which is 10 less the *unspliced* 3.0802, where the truth is 8.3115. The FFT answer was right throughout; the analytic moments were not, which failed validation on a correct build and sized the automatic bucket window from the wrong numbers. The same defect applied without a reflection, to any spliced `ssev`.

Raw moments now come from `Severity._raw_moments`, which keeps the exact closed-form `fz.moment` when there is no splice and integrates the patched `isf` in quantile space when there is. Unspliced results are bit-identical; spliced ones change, from wrong to right.

**The `approximate sgamma` / `slognorm` left-skew fit sets `sev_signed` explicitly.** It fits the reflected aggregate and maps back through `sev_reflect`, and had been relying on reflection implying signedness. It now declares the flag through the same low-quantile test the other branches use, which is what its docstring already promised.

**The unparser keys the severity keyword off `sev_signed` alone.** It used to emit `ssev` whenever `sev_reflect` was set. With both forms legal that would rewrite `sev 100 - lognorm` as `ssev 100 - lognorm` and silently un-clamp the declaration.

Plan in `dev/done/plan-reflected-loss-severity.md`. No grammar change, so the DecL reference is unaffected.

---

## 1.0.0a229

**[Greater-Tables-Dependency] `greater_tables` is a plain dependency, and the Python floor rises to 3.12.** GT 6.0.0 published to PyPI on 2026-08-07, so the workaround it forced can go: the `exhibits` extra had stayed commented out because an active extra naming an unpublished package would make `uv sync --all-extras` unresolvable, and the install instruction was a sibling checkout.

It lands in `dependencies` rather than as an extra. The exhibit surface is part of what the library is for, and an extra bought nothing once the package resolved normally.

**Three floors move with it**, because GT declares them and inheriting them silently would be worse than stating them:

| | was | now | why |
|---|---|---|---|
| `requires-python` | `>=3.11` | `>=3.12` | GT 6.0.0 is `>=3.12` |
| `numpy` | `>=1.26` | `>=2.0` | GT's floor |
| `pandas` | `>=2.1` | `>=2.2` | GT's floor |

The 3.11 classifier is dropped, and `pydantic` and `pyyaml` join the tree transitively. The alternative, a `python_version >= '3.12'` marker on the dependency, was rejected: it would have left the exhibit surface raising `ImportError` on 3.11, which is the optionality this change exists to end, only now silent and version dependent.

**The import stays at the point of use** in `_import_greater_tables`. That was never really about the packaging: the frame stage (`exhibit_frames`) is pure pandas and must not pay the import, and a broken install should say so where the tables are built rather than at `import aggregate`. Its error message no longer names an extra, because there is not one.


## 1.0.0a228

**[Chart-Support-Serialized] `ChartSeries.support` reaches the client, so a discretized density can be drawn as one.** The IR always carried the fact and the library's own renderer always honored it; the serializer deleted it in exactly the case that carries an instruction.

`SUPPORT_KINDS` states the contract (`charts/ir.py:88-102`): ``'atomic'`` means the points **are** the law and there is nothing between them, so a renderer draws stems where the atoms are far enough apart to see, steps where they are not, and a plain line only once a bucket is sub-pixel. `plots/_chartdoc.py` implements exactly that ladder off `series.support`, which is why `plot_chartdoc` has always drawn lollipops and steps.

`_canonical` omits any field equal to its dataclass default unless the class lists it in `_ALWAYS`, and `support` defaults to `'atomic'`. So the payload carried `support` on a severity pdf and a distortion curve (`'continuous'`, the exception) and carried **nothing** on every aggregate density, survival and reinsurance triple. The polarity was inverted against the meaning: the value that says *draw steps or stems* was the one deleted, and the value that says *a plain line is right* was the one that survived. A client reading that payload sees an absent field, which reads as "nothing special", and draws a line. It also has nothing to threshold a stem-versus-step switch on.

`support` joins `_ALWAYS[ChartSeries]`. **Every chart document hash changes**, once and deliberately: the `_ALWAYS` comment's promise that "adding an optional field never changes existing hashes" is what caused this, and it now carries the rule that produced the bug. Omit-at-default is right when absent means the neutral thing (an absent `scale` is linear, an absent `read_axis` is x, an absent `faint` is full weight, and a reader ignoring all three still draws an honest picture). It is wrong when the **default is the active case**. Before adding a field to a canonical form, ask which of its values a consumer must act on; if that value is the default, it belongs in the always list. Checked: `support` was the only field with the inverted polarity.

A sweep test now asserts that every series every live emitter produces declares its support in `canonical_dict`, so it cannot silently vanish again. The values were verified semantically, not just for presence: reinsurance densities and survivals `atomic`, the severity pdf and the distortion curves `continuous`.

**Consumers must read the new field.** A client that special-cased the field's absence needs to switch on `support` instead; absent should be treated as `'atomic'`, not as a line.


## 1.0.0a227

**[Loss-Lab-Round-3] Phase E: three housekeeping items, and the plan closes.**

**`writer`.** The api had reimplemented `to_agg`'s dependency-ordered walk because the library offered the export only as a *file*, and the two orders had drifted apart. `Underwriter.format_agg(pattern, kind, source, *, layout)` returns the same selection as a string; `to_agg` calls it and adds the header, so there is one selection and one ordering in the world. `to_agg` gains the same `layout=`: `'terse'` is the historical one-statement-per-line `.agg` form, `'spread'` puts each clause on its own line. Both re-parse to the same spec, so a spread export re-loads exactly as a terse one does.

`_KIND_WRITE_ORDER` gains `'pnl'` and `'xpnl'`, after `port`. They previously sorted last by the 99 fallback, which was the right answer for the wrong reason and gave anything copying the order nothing to copy.

**Found while doing it: `to_agg` failed outright on any session that had built a P&L.** The recipe stores the P&L's *engine aggregate* spec under the P&L's name, with no `consideration` key, so `spec_to_decl` dispatches on the kind and raises `TypeError` reaching for one. `_entry_to_decl` caught only `NotImplementedError` (the combinator-distortion case), so the exception escaped and took the export with it. It now warns, naming the entry, and writes the recipe's stored `program`, which re-loads by construction. The underlying spec-shape question is a separate item.

**`width`.** `format_program(spec_or_text, *, fmt, layout, trailer, width)` accepted `width` and ignored it, documented as reserved. Removed. `layout` is structural, one clause per line rather than width driven, nothing ever needed it, and a signature that takes an argument and discards it is worse than one that does not take it.

**`kinds`, the deferred half of `dev/done/plan-summary-tail-tables.md`.** The a113 rename executed for `Aggregate` and `Portfolio` and left `PnL` and `BivariateAggregate` without a `tail_df` at all, so a consumer holding four kinds had to dispatch on kind to know whether the frame existed. Both now carry `tail_df` (property) and `tail_periods_df(periods=)` (worker), the same pair and the same five columns. Their `summary_df` is **unchanged**: the PnL card (Consideration / Obligation / Margin) and the bivariate summary are right for their objects, which is what the a113 deferral was protecting.

* **`PnL.tail_df`** is the grand result read on the **downside**, because a P&L is a payoff: `T` maps to `p = 1/T`, not `1 - 1/T`, so a 1 in 200 year is one that goes 200-to-1 against you. The orientation is read from the result's `is_loss_value` rather than assumed.
* **`BivariateAggregate.tail_df`** is one block per axis, `MultiIndex (axis, T)`, off the **realized marginals** of the joint grid through a `GridDistribution` per axis. These are the axis *aggregate* distributions compounded under the shared frequency, so they are not the distributions of the `units` the program names, a unit there being the per claim component. **There is no total block:** the two axes are sized independently and routinely carry different bucket sizes (`bs` is a list, one per axis), so the sum has no common lattice and forming one would mean a rebucketing choice this class has never made. A dependent sum is also not a portfolio total, which assumes independence.

**A defect found and recorded, not silently changed.** `GridDistribution.tvar(p)` is the upper tail measure `E[X | X > VaR(p)]` at every `p`, with **no orientation flip**, despite `PnL.tvar`'s docstring claiming otherwise. On a payoff ladder, where `p` is small, that averages nearly the whole distribution and lands near the mean: the row's `VaR` reads the downside while its `TVaR` reads the other side. The matching measure would be the lower `E[X | X <= VaR(p)]`, which the library does not compute. This predates the new frames and already affects `Aggregate.tail_df` and `Portfolio.tail_df` on any payoff object, so it is documented on `PnL.tail_periods_df` and left for its own decision rather than changed at the end of this plan.

`dev/plan-loss-lab-round-3.md` moves to `dev/done/`.


## 1.0.0a226

**[Loss-Lab-Round-3] Phase D: every exhibit block says what it is, and a window prints as a window.**

**Captions on the passthrough exhibits.** `register_simple_exhibit` returned `{}` for its frame kwargs, and captions are lifted from exactly those, so every passthrough (`summary`, `stats`, `validation`, `tail`, `economic`, `bs_window`, `tail_behavior`, plus the two multi block builders `reins` under RAW, `economic_ratios` and `dependency`) arrived as a bare table. The insurer builders did carry captions, which is what made the gap easy to miss. The consequence was downstream: a client that wanted prose wrote its own, so a frame's description came to have three possible sources that could disagree.

`register_simple_exhibit` gains `caption=` and `formatters=`. **One frame does not have one description across five classes**, so an exhibit registered for several is now declared once per class group, which the existing "calling it twice extends" behavior already supported. `summary_df` is the clear case: count risk / severity / total loss on an `Aggregate`, the three ledger rows on a `PnL`, moments of `g` and its dual on a `Distortion`. A caption true of all three would say nothing.

RAW captions say what the frame **is**; the insurer overrides say what it **means** for the business, and replace them where they exist. `tests/test_exhibits.py` now sweeps every (object, exhibit, perspective) and fails if any block ships without prose, so a new passthrough cannot reopen the gap.

**The `Perspective.RAW` contract is restated** rather than quietly broken. RAW is still no business translation, no row emphasis, no dropped rows, no rearrangement. It now carries a caption and the column formats for units the frame cannot carry itself, both of which describe the table rather than interpret it. `summary` under RAW gains `MEASURE_FORMATS`, so a CV reads as a percentage exactly as the insurer view already showed it, which is the only snapshot change beyond captions.

**The window prose.** `x_min`, `x_max` and `W` formatted with `:g`, so a grid top of 12,182,881 printed as `1.21829e+07`, and nothing on the page said the three numbers were one fact. Two helpers in `_bucket_window.py`, used at every site in `bs_describe` / `bs_explain` and `Portfolio.bs_description` / `bs_explanation`:

* **`fmt_amount`** groups thousands and drops the fractional part above 1,000, where it is noise against the bucket size. `nan` reads `n/a` and the infinities read `infinite`, so a missing bound reads as missing rather than as a number.
* **`fmt_window`** writes the window as the interval it is: `[0, 131,072] of width 131,072`.

`bs` now renders through the existing `_fmt_bs`, so a sub-unit bucket reads `1/65536` rather than `1.52588e-05`.

Writing the interval and the width together exposed a real inconsistency the loose numbers had hidden: on an `Aggregate` the realized grid `[x_min, x_max]` and the winning method's recommended `W` are **different numbers**, the grid being the larger. The narrative now reports the window's width from its own ends, and leaves the method's recommended width in the sentence about methods, where it belongs.


## 1.0.0a225

**[Loss-Lab-Round-3] Phase C: the reinsurance chart draws a book, not just an aggregate.** `chart_reins` carried `@chart_reins.register(Aggregate)` and nothing else, and its availability gate read `occ_reins` / `agg_reins` off the object, which a `Portfolio` does not have in that form. So `available_charts` never answered `'reins'` for a reinsured book. The visible casualty was the app's Reinsurance Plot leaf, the only one gated on the chart registry rather than the exhibit registry, which greyed out and read as unbuilt. It was built.

**A `Portfolio` emitter, and a fourth basis, `'total'`.** A book's `reins_density_df` convolves each unit's **end-to-end** view, so there is no book-wide occurrence stage to draw and no `p_agg_subject`: units cede on different stages, and a book-level `'occ'` triple would have to pretend they cede on the same one. `'total'` is its own key rather than a reuse of `'agg'` because the first series really is gross here, where on `'agg'` it is the **subject**, which equals true gross only when no occurrence program sits underneath. Reusing the key would have put two meanings under one label, which is what the module docstring already warns against.

`meta['bases_available']` is `('total',)`, so a client offers the buttons that exist rather than assuming three, and any other basis is refused by name rather than quietly drawing the one that does exist.

**`_cession_stages` is now dispatched, not sniffed**, because the answer is a different fact about each class: an aggregate's stages come from its own two reinsurance slots, a book's from whether any unit cedes at all, which `reins_views` (a223) already answers. The document body is class agnostic and shared, reading only `reins_density_df`, `bs` and `label`, which both classes carry.

The three portfolio series are separate distributions and do not satisfy `gross = net (+) ceded`. The chart draws three laws on one grid, which is what it should show; the panel is not a decomposition and must not be read as one.

Tests use the existing `RR.Port` corpus program, whose units cede on *different* stages (A occurrence, B aggregate, C not at all), which is precisely why a book has no stage of its own.


## 1.0.0a224

**[Loss-Lab-Round-3] Phase B: `reins_price_df`, what a stated risk measure says a cession is worth.** `grep -n 'distortion' src/aggregate/_reinsurance.py` returned nothing. The library computed cessions thoroughly and priced ceded premium from the DecL clause that declared it, and `Distortion.price` took any pmf, but nothing walked one through the other. So a ceded premium could only be a price someone agreed. This asks the other question.

`Aggregate.reins_price_df(distortion=None, *, p=None, a=None, views=None)`, with the `Portfolio` twin, since both classes gained `reins_views` in `a223` and the implementation is then identical. Rows are `(distortion, view)`, columns `a` / `el` / `bid` / `ask` / `margin`, with `margin = ask - el`, what the risk measure charges over the expected loss.

`distortion=` takes a `Distortion`, a name in `obj.distortions`, a `{name: Distortion}` mapping, or `None` for the whole calibrated set, which is the usual call after `calibrate_distortions`. At most one of `p=` (each view resolves **its own** `a = q(p)`, holding the threshold fixed rather than the capital) or `a=`; with neither the price is unlimited, the natural quote for a cession, a layer already bounded by its own terms.

The stage selection is model knowledge, which is why this is here: which views exist for a given program is the judgment `reins_view_columns` encodes, and an api reproducing it would duplicate that judgment.

**What ties and what does not.** The unlimited `el` of the `ceded` view is the ceded mean, so it ties to `reins_stats_df` exactly. It does **not** tie to a DecL-declared ceded premium, and should not: one is a price agreed, the other a price implied, and the gap is the reading worth having. Views are separate distributions rather than a decomposition, so the `ceded` row is the price of the cession and a gross price less a net price is not.

**A distortion with a mass wants a finite asset level.** `ccoc` weights the essential supremum, so at the default `a = inf` over an unbounded support it charges the largest outcome the grid happens to represent and the quote moves with `log2` rather than with the risk. An aggregate cession is unbounded whenever the frequency is, since a bounded per-occurrence layer times an unbounded claim count still has no ceiling, so this bites on ordinary programs. Documented on the method; the same fact already makes `Portfolio.analyze_distortions` skip mass families on an unbounded book.

**Also, while in the file: `reins_audit_df` is gone from the docstrings.** The frame has not existed for some time, but the name survived at `_reinsurance.py:183`, `:270`, `:306` and `_aggregate.py:4082`, `:4097`, `:4110`, where anything grepping for it, including a future agent, would conclude it was there. Repointed to `reins_stats_df`, which is the live per-layer frame.


## 1.0.0a223

**[Loss-Lab-Round-3] Phase A: `reins_view=` names which of a cession's distributions to price on.** A reinsured object holds one distribution and the whole pricing surface read it with no way to say otherwise. Which one it holds is a property of how the program was written, not of what the caller wants: a `net of` program holds its net, a `ceded to` program holds its **ceded**. So an entry point reading `density_df` and calling it "the net" is right half the time, silently.

**The keyword.** `reins_view=` on `calibrate_distortions` (both classes), `evaluate` (both classes) and `analyze_distortions` (Portfolio). The default `None` is the object's own density, so every existing call is unchanged, byte for byte: the cached `exa_total` shortcut still fires on the untouched path, and the signed / payoff branch is reached exactly as before.

The name is neither `basis` nor `view`, both of which are taken. `basis` is the `EX` / `Est` level of `reins_stats_df` and the triple selector of `chart_reins`. `view` is worse: it is already a kwarg on the very methods this touches, meaning **bid / ask** (`Portfolio.price`, both `apply_distortion` fronts, `build_augmented`, `unit_capital_at`, `Distortion.effective_g`), with a third meaning (`agg` / `sev`) on `Portfolio.unit_density`. `analyze_distortions` reaches `apply_distortion(view='ask')` internally, so a bare `view=` would have carried two meanings one call apart.

**The vocabulary** is the `view` level of `reins_stats_df`, extended by the stage word where a program has two stages:

| `reins_view` | Aggregate | Portfolio |
|---|---|---|
| `gross` | `p_agg_gross` | `p_agg_gross` |
| `ceded` | `p_agg_ceded` with an aggregate cover, else `p_agg_ceded_occ` | `p_agg_ceded` |
| `net` | `p_agg_net` with an aggregate cover, else `p_agg_net_occ` | `p_agg_net` |
| `ceded occ` | `p_agg_ceded_occ`, both stages only | not offered |
| `net occ` | `p_agg_net_occ`, both stages only | not offered |

`ceded` and `net` resolve **end to end**, by the rule `Portfolio._reins_unit_views` already encodes, reused rather than rewritten. Naming the raw column instead would be the silent-wrong-answer this phase exists to close: with no aggregate cover `p_agg_ceded` carries the no-cession value, so a caller asking an occurrence-only program for its `ceded` would have been handed a point mass at 0. The `occ` pair appears only when **both** stages are present, since with one stage it duplicates `ceded` / `net` exactly and a reader offered five views assumes five answers. `p_agg_subject` is deliberately unnamed: it equals `net occ` or `ceded occ` according to `occ_kind`.

**`reins_views`, new, on `Aggregate` and `Portfolio`.** The accepted list, `[]` with no cession. The keyword refuses what an object cannot answer, so a caller has to be able to ask, and the accepted set is object dependent. A Portfolio's is shorter by design: a book convolves each unit's end-to-end view, so there is no book-wide occurrence stage to name, and units cede on different stages. Mirrors `available_charts` / `available_exhibits`.

**A Portfolio's own total already IS its net view** (`p_agg_net` equals `density_df['p_total']` exactly). So `analyze_distortions(reins_view='net')` is the existing path under a name, per-unit allocation included, and `'gross'` / `'ceded'` raise `NotImplementedError`: allocating either needs a twin portfolio of gross (or ceded) units, which the library does not build. Accepting the keyword and refusing the two it cannot honor is the point, since a caller sweeping `reins_views` gets an error rather than three identical net answers under three labels.

**Provenance.** `distortion_df` and `calibration_df` carry `attrs['reins_view']`. Without it a gross-calibrated set stored on the object is indistinguishable from an own-density one.

**`evaluate` does not adjust `P` with the view**, and says so. Evaluating the gross distribution against the net premium asks what stress the position would survive if the cover failed to respond, which is worth being able to ask deliberately and is a wrong answer to ask by accident. The `Step` label is suffixed by the view, so panels for several views concatenate. A Portfolio passes the view down to each named unit, and a unit that does not cede refuses **by name**.

A `reins_view` is refused on the signed / payoff path: that branch's canonical-frame bookkeeping is written against the object's own support, and a cession of a signed outcome has no settled meaning to hold it to.

New in `_reinsurance.py`: `REINS_VIEWS`, `reins_view_columns`, `resolve_reins_view` (the one place the refusal is worded, shared by both classes) and `reins_view_density`. Quantiles on a chosen view come from a `GridDistribution` built over it, so nothing here hand-rolls a search. Tests in `tests/test_reins_view_pricing.py`, fixtures mirrored in `decl-testers.agg` section RV.

This deletes the reason for the api's `_BasisView` shim, and closes `alloc`: the app lost every per-unit allocation table on a reinsured portfolio because its reinsurance pricing path could not reach `analyze_distortions` at all.


## 1.0.0a222

**[Undefined-Moment-Reporting] A moment that does not exist is reported as `inf` or `0`, whichever it is, and never as `nan` from a subtraction that cancelled.** Two sites, one theme: a value the arithmetic could not represent was reaching a user-visible CV or skewness.

**The variance cancellation floor is now relative.** `var = ex2 - ex1**2` subtracts two numbers that are *equal* when the variance is zero, so a degenerate component (a severity capped to a point mass) reaches it as pure cancellation and lands a few ulp either side of 0. The guard was `np.allclose(var, 0)`, which carries numpy's default **absolute** `atol = 1e-8`: a sensible size for a variance of order 1, and far too tight for one whose moments are of order 5e7. Mack2003's spliced Pareto hit exactly that, a point mass at 6961.69 giving

```
var = 48465132.98318529 - 6961.6903826**2 = -3.725e-08
```

which is `7.7e-16` relative, one machine epsilon. It survived the guard, logged `weird var < 0`, took the square root of a negative number and reported a `nan` CV and skewness for that component. The floor is now relative to the magnitudes being subtracted, `VALIDATION_NOISE * max(|ex2|, ex1**2)`, which is what the arithmetic actually says: the cancellation error is of order `eps` times the scale. The `isfinite` test in that condition is load-bearing, since `inf <= inf` would otherwise snap a genuinely infinite variance to exactly 0.

A *materially* negative variance still logs `weird var < 0`; it means the moments are mutually inconsistent, which is an error and not cancellation. The `sqrt` below it is guarded, because the logger has already said it.

**An unlimited layer over a heavy tail reports an infinite moment.** `_moms_analytic` expands `E[((X-a)^+ ^ l)^m]` binomially in the partial expectations, and that expansion cannot survive an infinite term: at attachment 0 the low-order coefficients are 0, so a term reads `0 * inf`; above 0 the alternating signs give `inf - inf`. Both produce `nan` for a layer moment that is simply infinite. A Pareto with `alpha = 1.5` has a mean and nothing above it, so an unlimited layer on it returned `[1, 3, inf, nan]` where `[1, 3, inf, inf]` is the truth.

Short-circuited instead. A layer capped at a finite limit is bounded by `limit**m`, so every partial expectation to a finite detachment is finite and the sum is always safe; only an unlimited layer can diverge, and the `m`-th moment of an unlimited excess layer exists exactly when `E[X**m]` does. That is one `isfinite` test, and the sum below it is then only ever evaluated on finite terms.

This was latent, and `[Pareto-Type-I-Analytic]` (`1.0.0a221`) surfaced it: the analytic branch reports a missing moment as `inf`, where the quadrature it replaced had returned a large finite number that hid the `inf - inf`. Both answers were wrong; this one is right.

**Effect on the reproductions book**, which is where the whole sequence started: the seven chapters that carried 226 stderr lines now carry **one**, the genuine 96% grid deficit on Mack's infinite-mean Pareto, which is the chapter's own subject.

No API change.

## 1.0.0a221

**[Pareto-Type-I-Analytic] The single-parameter Pareto gets closed-form partial moments, so `sev {xm} * pareto {alpha}` stops integrating a heavy tail to infinity on every build.** `_partial_e` had an analytic branch for the *shifted* (Lomax) form, `scale = lam, loc = -lam`, and sent everything else to quadrature. The Type-I form, `scale = xm, loc = 0`, support `[xm, inf)`, is what DecL's `sev {xm} * pareto {alpha}` builds and what the rare-event literature uses, and it took the fallback every time: a logged warning, an `IntegrationWarning` reporting the integral as probably divergent, and a numerical answer where an exact one exists. The reproductions book's `Liu2026`, which is Type-I Pareto throughout, carried 80 stderr lines from this one gap.

For `S(x) = (xm/x)^alpha`:

```
int_xm^a x^k f(x) dx = alpha * xm^alpha * (a^(k-alpha) - xm^(k-alpha)) / (k - alpha)
```

with `alpha * xm^alpha * log(a/xm)` at the removable case `k = alpha`. At `a = inf` and `k < alpha` this is `alpha * xm^k / (alpha - k)`, so `k = 1` is the textbook mean `alpha*xm/(alpha-1)`; at `k >= alpha` it is `inf`, which is correct, and the moment machinery downstream already reports a non-existent moment as `nan` rather than complaining (`[RuntimeWarning-Census]`, `1.0.0a220`).

**More accurate, not just quieter.** It agrees with the quadrature path it replaces to 8e-16 relative, and beats scipy's own `moment(3)`: for `alpha = 3.5, xm = 250` the closed form gives exactly `109375000.0` where `scipy` gives `109375000.00679`. No baseline moved, because nothing in the suite was reading a Type-I Pareto moment precisely enough to notice the quadrature error.

**The fallback is hardened too**, for the genuinely shifted Paretos that still reach it. It now integrates from the distribution's own support lower bound rather than from 0: quadrature over a dead region where the density is identically zero is exactly what made `quad` conclude "probably divergent", since it samples the flat part and finds nothing. `IntegrationWarning` is suppressed there because the caller already tests the returned absolute error estimate, which is the honest convergence check. The `Pareto not shifted to x>0 range` and `Potential convergence issues` lines drop from `logger.warning` to `logger.debug`: taking the fallback is a routine code path, not a problem, and it was 41 uncontrolled stderr lines in the rendered book.

**Measured.** The five Type-I Pareto builds behind `Liu2026`'s cells, run standalone: 21 stderr lines before (10 `IntegrationWarning`, 10 logger lines, 1 defective), 0 after.

No API change. `_partial_e_pareto_type_1` is a new private helper beside `_partial_e_numeric`, which stays as the auditing reference its docstring describes and is now the test oracle.

## 1.0.0a220

**[RuntimeWarning-Census] The benign numpy boundary noise is guarded, and four sites that were quietly returning `nan` into results are fixed.** Closes the `dev/TODO.md` item of the same name, opened when `[TVaR-Endpoint-Noise]` (`1.0.0a179`) cleared the first three sites and left the rest for case-by-case review. `pytest -W error::RuntimeWarning` over the full suite went from 75 failures to 0; the census itself, the count of distinct source lines emitting a `RuntimeWarning`, went from 14 to none.

The item was filed as benign noise. Ten of the fourteen sites were, and the other four were bugs.

**Guarded, because the value never reaches the answer.** `np.errstate` plus a `Notes` paragraph saying why, following the a179 pattern. `moments.py` `agg_from_fs`, `cumulate_moments` and `static_moments_to_mcvsk`: a severity with no finite second or third moment (a Pareto with `alpha <= 3`, a Lévy) makes the raw moments infinite, so the central combinations evaluate `inf - inf` and `0 * inf`. `nan` is the correct report for a moment that does not exist. `PHDistortion.g_prime`: `np.where` evaluates both branches, so `0 ** (rho - 1)` is computed at `x = 0` and then discarded in favour of the explicit `inf`. `Distortion._kusuoka_density`: the next line discards every non-finite value. `pedagogy`'s two `log10(Z)` calls: the next two lines mask `-inf` out of the figure.

**Fixed, because the `nan` was the answer.**

`WangDistortion.g_prime` returned `nan` at **both** endpoints. It evaluated `phi(z + lambda) / phi(z)` with `z = Phi^-1(x)`, which is `0 / 0` where `z` is infinite and the normal density underflows. Cancelling the exponentials first removes the indeterminate form:

```
phi(z + lam) / phi(z) = exp(-lam * z - lam^2 / 2)
```

giving `+inf` at 0 and `0` at 1, the correct one-sided limits, at half the cost (one `ppf` and one `exp` rather than two `ppf` and two `pdf`). The identity case `lambda = 0` is separated out, since `-0 * inf` is `nan`.

The base-class `g_prime` central difference evaluated `g` **outside `[0, 1]`**, where a distortion is not defined. At `x = 0` it asked for `g(-1e-6)`: the CLL family takes a fractional power of a negative number there and the LEP family a square root of a negative product, so both returned `nan` for a slope that exists. The stencil is now clamped into the interval, giving a forward difference at 0 and a backward difference at 1. **The interior is bit-identical on purpose:** the clamp only engages within `h` of an endpoint and the divisor stays the exact literal `2e-6` everywhere else, so no existing slope moves. This fixes every distortion that inherits the default, not just the two that showed up.

`LEPDistortion.g_inv` took the square root of a discriminant that rounding can push a few ulp negative for a `y` sitting exactly on the mass level or on the saturation level. Both branches are already resolved by the enclosing `where` / `maximum`, so the radicand is floored at 0.

`_aggregate_compute.discretize` divided by a zero normalizer. A severity whose support lies entirely off the grid discretizes to all zeros: `build('agg HF 1 claim sev gamma 50 cv 0.1 fixed hints{bs=1/64; log2=10}')` puts a mean-50 severity on a grid topping out at 16. `appx / 0` then replaced a truthful "no mass here" with `nan` in every bucket, and **`nan` survives the FFT**, so the entire aggregate came back `nan` with nothing said anywhere. The zeros are now left alone, which makes the deficit exactly 1 and hands the report to `[Warning-Policy]`'s machinery:

```
>>> a.validation_description
'fails pmf deficit 1.000e+00, sev mean, agg mean'
```

plus a `DefectiveDistributionWarning` naming the fix. A `logger.warning` records which component was empty.

**Not made a standing gate.** `-W error::RuntimeWarning` is clean on the full suite, but `tests/test_bivariate.py::test_mv_explain_flags_clipped_book` failed under it in two runs out of four and could not be reproduced: the module passes alone, and all three bivariate modules pass together. That suite is the one already carrying `xdist_group` for memory pressure, so load is the first suspect. Tracked in `dev/TODO.md` as `[Bivariate-Gate-Flake]`; the gate is documented as a release-time command rather than added to `addopts` until it is understood.

No API change.

## 1.0.0a219

**[Warning-Policy] A defective distribution is announced once, at the level where it can change a price, and again when a defective law is actually priced.** Found by reading the rendered reproductions book, which carried about 250 stderr lines across 17 pages. They were not spread thin: 208 of them came from nine source lines, and two chapters produced 72% of the total. `Mata2005` printed the *same two messages* 64 times. The genuinely interesting facts, that Mack's Pareto splice has no finite mean and that BenRached's Lévy severity loses 12.6% of its mass, were buried in the repetition rather than surfaced by it.

Three things were wrong, and they compounded.

**The threshold was three orders of magnitude too tight.** `DefectiveDistributionWarning` fired on any deficit above `validation.noise` (1e-12), a floor that measures arithmetic dust. Meanwhile the library's own pricer already treats `validation.deficit_materiality` (1e-4) as the line between FFT truncation and an economic problem. Sixteen of the book's 95 defective warnings were below 1e-4, three below 1e-7, and one was 1.6e-12: real, in the sense that the number was not zero, and useless, in the sense that no reader could act on it. Construction now warns at the materiality floor, so the two agree on what "defective" means.

**Nothing deduplicated.** The message embeds the measured deficit, and Python's default filter keys on `(text, category, lineno)`, so every distinct value counted as a new warning. Jupyter then clears `__warningregistry__` between cells, so even identical messages repeat once per cell. New in `constants`:

```python
warn_once(message, category, *, key=None, stacklevel=3)
reset_warn_once()          # re-arm; public
warn_once_isolated()       # context manager, library machinery
```

`warn_once` keys on the *condition*, not the text, so the first occurrence carries its numbers and the rest stay silent behind one appended sentence saying so. A sweep that rebuilds the same shape 64 times reports one fault. `warn_once_isolated` both bypasses the registry and restores it on exit, which is what the library's two warning-provoking internals need: the portfolio window pre-pass must not *spend* the session's one emission on a warning it then filters away, and `sharpen`'s probe **counts** warnings per candidate cell, so it needs every cell to warn.

**The one moment it mattered was silent.** A deficit is a grid property until someone prices through it; then forwards `S` parks the missing mass at the top atom, backwards parks it at the bottom, and the same book returns two defensible numbers differing by exactly the deficit. `choquet_weights` is the single gate every pricing route passes (`Distortion.price`, `Portfolio.price`, `Aggregate.apply_distortion`), and a material deficit with `allow_deficit=True` only wrote `logger.debug` there. It now warns, once, on its own key, so neither channel swallows the other.

**Every object carries its own verdict.** New `Validation.DEFECTIVE`, set from the measured deficit and deliberately **not** in the passing set: a defective object reports `valid = False` and says why.

```
>>> a.validation_description
'fails pmf deficit 4.141e-01, sev mean, agg mean'
```

The deficit leads the failure list, on the same argument that already puts mean before CV: mass missing from the realized law makes every moment comparison below it uninformative. `validation_explanation` gains the long form. The magnitude is recorded on the object at update time by `_validation.pmf_deficit`, the one definition of `1 - sum(p)`, which replaces the identical private `_sharpen_deficit`.

**Also fixed, same cause, different family.** Two `ss.rv_histogram` calls left `density` at its `None` default, so scipy emitted `RuntimeWarning: Bin widths are not constant` (30 lines in the book) for every unequally spaced `chistogram` and for **every** `meta` severity, whose bins are `bs*1e-7, bs/2, bs, bs, ...` and so can never be constant. Passing `density=True` states what scipy already assumed: output is bit-identical, checked.

**Deliberately unchanged.** The `sharpen` grid probe still gates on `VALIDATION_NOISE`, now explicitly tighter than the warning. Choosing among candidate grids, losing no mass at all is free to insist on; interrupting the reader is not free, so that waits for a deficit large enough to move a price. The docs that asserted the two thresholds were the same are corrected.

**Newly loud, on purpose.** A massive/bivariate update whose pinned `(bs, log2)` cannot cover the measured window reported it through `logger.warning`, which is silent by default. That clip is not a sliver: a 64x64 pin loses a measured 0.9999 of the joint. It is now a `DefectiveDistributionWarning`.

**Measured.** `Mata2005`'s two heaviest cells, run standalone: 32 warnings before, 1 after, and the survivor is a real 3.4e-04 deficit. The library is not warning-free; the remaining `RuntimeWarning` noise is `[RuntimeWarning-Census]` in `dev/TODO.md`.

**Opting out** is unchanged and needs nothing new: `silence_warnings(DefectiveDistributionWarning)` covers both channels, and `reset_warn_once()` is the inverse.

**Breaking, narrowly.** `Validation` gains a member and `valid` now returns `False` for an object with a material deficit that previously passed. Test suites that use `pytest.warns` on library warnings need the registry reset between tests; the aggregate suite does this with an autouse `conftest` fixture. `_bucket_window._sharpen_deficit` is gone, replaced by `_validation.pmf_deficit`.

## 1.0.0a218

**[Reproductions-Highlights] The reproductions book opens with a short version: five papers, one page each, each page a DecL program beside the exhibit it reproduces.** The detail chapters are the point of the exercise but they are not the way in. Each of the fifteen carries its transcription decisions, its inferred readings and its grid caveats, and a reader wanting to know whether the library reproduces published work had to wade through Python to find out. The new first chapter answers that in ten minutes: @Venter1983 for accuracy, @Bear1990 for the language, @Homer2003 for reach, @Bruno2006 for speed, @Mata2005 for the reproduction auditing the paper rather than the reverse. One paragraph of context, the program, the answer, one number.

The program shown on each page is read back off the object that produced the table beneath it, so the two cannot drift, and every other line of Python is hidden. Grid settings ride inside the program in `hints{}` rather than arriving as arguments to `update`, which is what makes the page's claim literally true: the declaration is the whole program.

**Two chapters stop using the constructor.** `Venter1983.qmd` and `Mack2003.qmd` were the only reproductions built through `Aggregate(...)` plus `update()` rather than through `build`, which meant the two best examples in the book had no DecL to show. Both convert, and neither changes a number.

Venter's severity now ships as **`sev Venter1983.Piecewise`**, the eleventh named curve in `library.agg`, so his whole model is one statement: `agg Venter 13.7376 claims 250000 xs 0 sev.Venter1983.Piecewise poisson hints{log2=16; bs=500; normalize=False; padding=1}`. That reproduces his Exhibit 3 recursive column to 5.1e-05 over all 34 rows, and switching `bs=500` to `bs=5` lands on his characteristic function column to 7.3e-05 instead. **The curve carries a usage constraint**, recorded in its `note{}` and in `severity-curves.qmd`: the limit mass rides in one wide bin above 250,000 and needs a `250000 xs 0` layer to fold it back onto the limit, so used unlimited it spreads 2.41% of probability over an interval Venter never wrote. Same shape of constraint as `Mack2003.RiebesellPareto`.

Mack's spliced lognormal-into-Pareto was a **120,000 point `chistogram` cut at 1e12** with the residual swept onto the last bin. It is now a two component `splice` clause, which is the same shape as Theorem 2's own statement: two conditioning windows, two weights, and the survival function continuous at the join by construction. No cutoff and no residual. Limited expected values agree with the construction they replace to within four parts in a billion, the Riebesell loss elimination ratio still matches to nine significant figures above the threshold, and doubling the sum insured still gives exactly 1.200000. One benign diagnostic follows from the Pareto's missing moments: at some limits its component is capped to a point mass whose variance is zero to rounding, so `aggregate` reports a tiny negative variance and a `nan` skewness. It reaches the moment diagnostics only.

**Also.** `severity-curves.qmd` gains the Venter provenance entry, and its summary cell now tolerates a tabulated curve: a `chistogram`'s `fz` is an unfrozen `rv_histogram` with no `.args`, no `.kwds['scale']` and no `.dist.name`, and it reads a positional `stats('mv')` as a shape parameter, so the call is now `stats(moments='mv')`. `_quarto.yml` listed `Homer2003.qmd` twice and rendered it as two chapters; the duplicate is gone.

No API change. `library.agg` gains one entry, so `build.discover(kind='sev', tags='role:paper')` returns eleven rows rather than ten.

## 1.0.0a217

**[Consolidated-Density-Columns] `add_exa` attaches its columns in one concat,
so a wide book no longer fragments `density_df`.** Found while clearing the
errors out of the monograph's published-problems page: every portfolio built
there printed a wall of pandas `PerformanceWarning: DataFrame is highly
fragmented` into the rendered output, 53 of them on one page. The source was
the library, not the documents. `add_exa` assigned each of its columns to
`density_df` individually, twelve per unit plus the totals, and pandas inserts
a block per assignment. Past a hundred blocks pandas warns, and the frame it
leaves behind is slow to read column-wise for the rest of its life.

The columns are now accumulated in a dict and attached in a single
`pd.concat`. A twelve-unit book goes from 133 blocks to 6 and warns not at
all. Output is unchanged: same columns, same order, bit-for-bit identical
values, checked against the previous implementation.

**Breaking, narrowly.** `add_exa` no longer extends its frame in place; it
returns a new one, because a consolidated frame cannot be produced in place.
Both internal call sites (`Portfolio.update` and
`_portfolio_sample.swap_density_df`) already assigned to `density_df` and were
updated. Code that called `port.add_exa(df, state)` for the side effect and
then read `df` must now take the return value:

```python
df = port.add_exa(df, unit_state)
```

## 1.0.0a216

**[Inline-Port-Engine] a P&L can write its portfolio engine out, so a
portfolio-backed program travels.** Raised by the author and the `aggregate_api`
agent against `[Derived-Programs]`: an aggregate engine inlined its body and a
portfolio engine emitted `less port.NAME`, and doing the two kinds differently
is arbitrary. Worse, it does not work where it matters. `port.NAME` resolves
only against the underwriter that holds NAME, so the text builds in the session
that wrote it and nowhere else, and a shared multi-user server would have to
write every user's book into one knowledge base to make it resolve. That is the
same argument that made `reins_program` self-contained; it applies here and was
not followed through.

The grammar gains the portfolio twin of the inline aggregate engine:

```
agg_source: AGG name as_label agg_body   -> agg_source_inline
          | PORT name as_label agg_list  -> agg_source_inline_port
          | builtin_agg                  -> agg_source_ref_agg
          | BUILTIN_PORT                 -> agg_source_ref_port
```

reusing the shared `agg_list`, minus the trailer, exactly as the inline
aggregate reuses `agg_body`. The unit list ends where the P&L's second `less`
begins: `less` cannot start an `agg_out`, so the greedy list has one parse, and
`test_grammar_ambiguity` agrees.

```
pnl Book_PnL
  11238.555678761779 premium
  less
    port Book
      agg Motor
        50 claims
        sev lognorm 100 cv 2
        poisson
      agg Liability
        20 claims
        sev lognorm 200 cv 1
        poisson
  less
    0.25 premium expenses
```

`Portfolio.pnl_program` emits that, and it builds against an `Underwriter()`
with an empty knowledge base. The portfolio's own trailer is dropped rather than
carried up: it describes the book, not the P&L over it.

*One build path, two render forms.* The parser already resolved `port.NAME` to
its spec at parse time and then kept only the name, so `_build_pnl_from_port`
went back to the store for what the parser had already had in hand. Both source
forms now carry the resolved spec on `_engine_port_spec`, and the underwriter
performs **no knowledge-base lookup** for either. `_engine_port` survives as the
referenced name alone, set for the reference form, so `port.NAME` still
round-trips as the reference the author wrote instead of being expanded into the
units it resolved to. The source kinds are `'port'` and `'port.ref'`.

Nothing about the reference form changes for existing programs. Grammar
reference regenerated; corpus entries added to `decl-testers.agg`, which can now
round-trip a portfolio-backed P&L in isolation, having had no way to before.

## 1.0.0a215

**[Sharpen-Pin] `sharpen()` writes its outcome onto the object's own program,
so `program` means the program that *builds* this object.** Author-found, and
the second half of `[Derived-Programs]`. `a.sharpen()` moved the grid and left
`a.hints` empty, which is surprising on its own; the sharper form is a
declaration that pins one axis. `hints{log2=17}` sharpened on the bucket axis
left `hints` still reading `log2=17`, silently *incomplete* rather than merely
stale, so `build(a.program)` came back on a different bucket while `hints` read
as a complete record of the grid.

`sharpen()` now writes `program`, `note` and `hints` **together**, through
`_program.pin_sharpen`, so the three never disagree and `build(ob.program)`
reproduces the sharpened object. A moved grid becomes `hints{log2=...; bs=...}`;
a confirmed one a `note{sharpen: grid confirmed, no change}` and deliberately no
hints, for the reason `[Derived-Programs]` gave. `program` is stamped the way
`build` stamps it, through the preprocessor, so it stays one line with any
`doc{{{...}}}` body encoded, and only the trailer changes.

*Sharpen's sentence is replaceable, not cumulative.* The note prefix
`'sharpen: '` is namespaced, and a new verdict removes the old one, so probing
three times leaves one verdict rather than three. A confirmation is also dropped
when a later probe moves the grid, since it would sit beside fresh hints
contradicting them. Your own prose never matches the prefix and is never
touched. This closes a latent defect in a213, where repeated build-sharpen
cycles would have accumulated notes.

`sharpen_program` is now that pinned program rendered to read, the `spread`
layout with the trailer left in, rather than a re-derivation. Four things to
reach for, one each: `sharpen_program` for text to read or share, `program` for
the one-line stamp, `hints` for the settings, `note` for the verdict. Pinning is
a no-op for an object with no program, or one whose program cannot be re-parsed;
neither is worth failing a probe over, and `sharpen_description` still reports
the move.

**The P&L spread layout reads as the subtraction it is.** Author request. The
premium head and both `less` keywords were glued into the block head, so a
`pnl` rendered its first three ideas on one line and its expense clause on
another. Each `less` now heads a sub-block over what it takes away:

```
pnl Book_PnL
  15384.6422566226 premium
  less
    agg Book
      100 claims
      sev lognorm 100 cv 2
      poisson
  less
    0.25 premium expenses
  note{motor book, 2026 plan}
```

`terse` is **byte-identical** to what it has always been, because
`_render_terse` space-joins a head back onto its children, so `spec_to_decl`,
the `to_agg` exporter and the whole round-trip corpus are untouched by the
layout. `xpnl`, `peel` and a `port.NAME` engine follow.

## 1.0.0a214

**[Chart-Atomic-Support] the discretization is the distribution, and the
renderer decides how much of it you can see.** Author decision, taken
against the schema freeze rather than around it: `ChartSeries.support`,
one of `'atomic'` or `'continuous'`, defaulting to **atomic**. The default
is the point. This library computes with gridded discrete laws and treats
them as the distribution rather than as an approximation to some
continuous ideal, so `'continuous'` is the exception a series has to
claim. Today that is a distortion (g is a function of s, defined
everywhere, sampled on the knot-spliced grid) and a frozen severity (build
one and it is a scipy variable with no `xs` and no discrete density,
because discretization happens in `Aggregate` and not there). A discrete
severity is atomic, and every aggregate grid is.

The document says only what the law **is**. How densely to draw it is a
property of the figure, which no document can know, so the renderer owns
the ladder:

| room per atom | drawing |
|---|---|
| at most `LOLLIPOP_ATOMS` (40) in view | stems with markers, each atom drawn |
| at least `STEP_PIXELS` (3) per atom | steps, read as a bar at each bucket |
| below that | a plain line, indistinguishable from steps anyway |

This is not new house style, it is the rule `plots/_aggregate.py` has
always followed (`stem` under `mx <= 60`, `steps-mid` above it,
`steps-post` for F) lifted into the renderer so every chart gets it.
Counted in **visible** atoms rather than in `log2` or loss units, because
charts crop: a 65,536-point grid showing 80 buckets should draw its steps,
and the same grid showing 18,000 should not.

Cumulative functions skip the stem rung and step **right-continuously**:
F and S take a value at every x, not only at the atoms. Which of the two a
series is comes off the **axis**, not the series role, and that is
load-bearing rather than stylistic: a reinsurance series is called `gross`
in both panels, and only the axis knows that one carries mass and the
other accumulated probability. The first cut read the role and stepped the
reins survival panel as though it were a density.

No image moves: the distortion is continuous and drew as a line before and
after.

## 1.0.0a213

**[Derived-Programs] the program that reproduces an object you arrived at.**
The round-trip surface already said what an object *is*: `program` is the
statement the parser received, `pprogram` is what the parser understood. It
said nothing about objects you *arrived at*. Three ways of arriving are common
enough to deserve text, and each is a small piece of grammar knowledge that
every caller was otherwise re-deriving.

**`sharpen_program`**, a property on `Aggregate` and `Portfolio`, is the fourth
thing a probe produces, beside `sharpen_df`, `sharpen_description` and
`sharpen_explanation`. `sharpen()` moves the grid and the knowledge of which
grid then lives only in the live object; reopen the notebook tomorrow and the
rebuild is back on the automatic choice with the audit still to run. The grid
moved, so the program comes back carrying `hints{log2=...; bs=...}` and nothing
else, the hints being the record. The grid was confirmed, so it carries
`note{sharpen: grid confirmed, no change}` and deliberately **no** hints:
pinning a grid the automatic selector would have picked anyway adds noise to a
program someone is going to read and share, and implies the selector is not
trusted. The note is the record that the audit ran, which is what saves running
it twice. A probe run under `execute=False` that recommends a move it did not
take records the recommendation, since the object still sits on its original
grid and pinning the winner would describe an object that does not exist.
Empty before `sharpen()` has run, exactly as `sharpen_df` is.

**`pnl_program(loss_ratio=0.70, expense_ratio=0.25)`**, a method on both,
returns `pnl NAME_PnL <premium> less <engine> less <expense>`. An `Aggregate`
**inlines** its own body, stripped of the trailer, which the wrapping `pnl` now
owns, and the aggregate's own `as` label becomes the engine label, which is
what names the P&L's loss leg. A `Portfolio` **references** (`less port.NAME`),
because the grammar has no inline portfolio engine. The premium is `inherit
premium` when the exposure states one, and otherwise expected loss over
`loss_ratio`, read off the computed density so the realized ratio is exactly
the one asked for. `expense_ratio=0` omits the clause rather than writing a
zero.

**`reins_program(cession)`**, a method on `Aggregate`, takes one cession clause
per tier and returns a **self-contained** program with the clause in its
correct slot. The slot is the whole point: an occurrence cession sits *before*
the frequency clause and an aggregate cession *after* it, so neither can be
appended to the text, and anyone splicing strings rather than specs gets it
wrong. The clause replaces its own tier and leaves the other alone. Being
self-contained is the load-bearing choice: `agg NEW agg.OLD occurrence net of
...` is grammatical and is the obvious first idea, but `agg.OLD` resolves only
against an underwriter's knowledge base, so the returned text would build in
the session that made it and nowhere else, and a shared server would be writing
every user's builds into one store.

*Two constraints shaping all three.* **The trailer is merged, never appended.**
A spec holds one `note` and one `hints`, so a second clause would silently win
or lose depending on the transformer; each mutation replaces the settings it
owns in place and leaves every other chunk of the clause, recognized or not,
exactly where the author put it. **The render asks for the trailer.**
`format_program` defaults to `trailer=False` on the reasoning that formatting a
program is usually about the math rather than the metadata; these three are the
exception, and for `sharpen_program` the trailer is the entire payload.

Shared machinery in `_program.py`, thin delegations on the host classes, which
is the pattern `sharpen` itself already follows. Functions rather than
`ProgramMixin` members, because the mixin's six hosts include `Severity`,
`Distortion` and `BivariateAggregate`, none of which can answer any of the
three, and a member that raises on four of six hosts is a member in the wrong
place. All three parse the stored program back to its spec, mutate the spec and
re-render through the writer: never string surgery. `tests/test_derived_programs.py`
asserts the round trip for each, the cases a naive implementation breaks on
included (a program already carrying `note{}` / `hints{}` / both, an engine
with no premium through the P&L wrap, both reinsurance slots, and a sharpen
that finds no improvement). Programs and the derived outputs both mirrored into
`decl-testers.agg` under a `DP` block.

## 1.0.0a212

**[Chart-IR] pass three, [Chart-Conversions]: severity, the third
conversion.** `charts/_emit_severity.py`: `chart_severity(sev, n=512)`
synthesizes the grid a `Severity` does not have (it is a look-through onto
a frozen scipy variable, not a compute result) by absorbing the algorithm
the app's server had been carrying: invert the survival over log-spaced
exceedance probabilities in two half runs meeting at the median, then
`unique` for monotonicity. A severity is routinely heavy tailed and often
unbounded, so a linear grid either truncates the tail or spends nearly
every point on it. Quantile spacing puts the points where the probability
is, and that is meaning, which is why it belongs in an emitter.

The ordinate is a **pdf**, not a mass, and the axis says so: an aggregate's
`p_total` is probability per bucket and sums to one, this does not and must
never be summed.

*A defect inherited with the algorithm.* A discrete severity has no
density, so its pdf is identically zero and the panel drew a flat line
along the axis and called it a distribution. Where the pdf is zero
everywhere on the grid the emitter now reads the jumps of the step cdf
instead, which are exactly the atoms, labels the axis `Probability mass`,
and records which reading it gave in `meta['ordinate']`. The test is the
symptom rather than the severity's kind, so a wrapper around a discrete
law is caught as surely as the discrete law itself. The two readings are
never mixed in one document.

*Shared two-panel semantics.* `charts/_two_panel.py` holds the parts five
charts (reins, sev, agg, port, pnl) have in common and that are meaning
rather than styling: the window pad, the survival floor and its round
decade, and the rule that float dust becomes a gap. Lifted out now, with
the third of the five, rather than after four copies had drifted. The
reins emitter moves onto it; `SURVIVAL_FLOOR` and `WINDOW_PAD` now live
there.

No image gate for this chart either: `plot_severity` exists but draws a
different chart (four panels including a Lee diagram), so there is nothing
to compare pixel for pixel. Corpus entries added as `CH.Sev*` in
`decl-testers.agg`, one per branch the emitter has to get right.

## 1.0.0a211

**[Chart-Conversions] the suggested range is the range of the data, not of
the frame.** Author correction to a210: pinning `Distortion.plot` to
`[0, 1]` was wrong, and deliberately so in the original. A distortion is
legitimately 0 or 1 over whole stretches of s, and an axis pinned to the
unit interval draws those stretches along the frame, where they cannot be
read. The `[0, 1]` pin is reverted and the compositor is back to
autoscale's margin, with a comment recording why the margin is deliberate
so it is not "tidied" again.

The renderer keeps honoring `suggested_range`, which it must (the reins
window is the difference between a readable chart and a sliver at the
origin), but now treats it as the extent of the *data* and insets a linear
axis by matplotlib's own `axes.xmargin` / `axes.ymargin`, exactly as
autoscale would. A log range arrives as whole decades and is drawn as whole
decades, because a decade gridline is how that axis is read. The distortion
baseline regenerates **byte-identical to its a209 form**, so the correction
is exact rather than approximate, and the conversion residual stays 0.

## 1.0.0a210

**[Chart-IR] pass three, [Chart-Conversions]: the reinsurance triple, and
the renderer learns to lay out panels.** `charts/_emit_reins.py`:
`chart_reins(agg, basis=None)` reads `reins_density_df` and emits the
two-panel exhibit, a density panel and a log-survival panel over one shared
loss window, three series each. Registered behind a cession predicate, so
`available_charts` answers `['reins']` exactly when some stage of the
program cedes something.

Which of the frame's three triples is drawn is a **semantic option** to the
emitter and not renderer view state: `'sev'` is the occurrence program seen
per claim, `'occ'` the aggregate before, ceded by and after that program,
`'agg'` the aggregate cover's subject, cession and net. They answer
different questions of different contracts. The default is `'occ'` where
the occurrence program cedes and `'agg'` otherwise, so it is always a
triple that carries something; naming a basis whose stage cedes nothing is
a `ValueError` listing what is available. On the `'agg'` triple the first
series is **subject**, never relabeled gross, since it equals true gross
only with no occurrence program underneath it.

Survival is accumulated in the emitter through `GridDistribution.sf`
rather than client side, and agrees with the app's `1 - cumsum` **exactly**
(asserted, not assumed: each column is a pmf on one grid). Values at or
under `LOG_FLOOR` emit as gaps, because a log axis cannot place float dust
and drawing it at the floor reads as tail that is not there. Series names
reuse the existing `REINS_LABEL_*` constants, so the triple is called the
same thing here as everywhere else in the library.

*Renderer.* Panels now lay out in one row, one axes per panel, and panels
naming the same x axis **share** it, which is what makes a density and its
tail one reading rather than two pictures. The single-panel path is
untouched. A multi-panel document refuses a caller-supplied `ax`, since a
shared axis is a property of the figure.

*A defect the conversion exposed.* `ChartAxis.suggested_range` documents
that the initial view honors it, and the renderer read the field only to
detect the unit interval for tick pinning: it never set the limits. Benign
on the distortion, whose data spans its whole range, and badly wrong on a
heavy tail, where the whole point of the window is that the visible mass
would otherwise be a sliver at the origin. The renderer now honors it.
`plot_distortion` pins `[0, 1]` on both axes with it, so the two sides of
the seam stay on one baseline image, and because a unit square should draw
as the unit square rather than with autoscale's 5% of white where no
distortion can go. The baseline regenerates; the conversion residual stays
**0**.

No image gate for this chart: there is no `plots/` compositor for it, so
the before side is the app's client-side builder and that comparison
belongs to the paired app commit.

## 1.0.0a209

**[Chart-IR] pass two closes its sign-off gate, part two:
[Chart-Plain-Text-Names] with [Chart-Axis-Labels].** Plan:
`dev/plan-chart-schema-signoff.md`. Both halves land together because both
move the same pinned baseline image, and one regeneration is honest where
two would be noise.

*Names.* A schema rule, not a label fix: **every human-facing string in a
chart document is plain text**, never markup in any renderer's language.
ECharts has no TeX, so the dual distortion's `$g\check$` legend string was
already broken on one of the two renderers that exist (and malformed as
mathtext besides: `\check` takes an argument). `ChartDoc.tex` is the new
optional plain-to-TeX lookup a renderer consults only if it can typeset;
one that cannot ignores it and is still correct. It is keyed by string
value rather than by field, so one entry covers a name, an axis label and a
title that read alike, and it defaults empty and is omitted from the
canonical form, so **no existing document hash moves**. The dual is now
`ǧ(s)` everywhere, in `constants.py` as `DISTORTION_DUAL_LABEL` beside its
`DISTORTION_DUAL_TEX` form; the compositor had been calling the same curve
two different things (`$g\check$` on the linear branch, `Dual {label}` on
the return branch) and now calls it one.

*Axis labels.* The renderer draws the axis labels the document already
carries, on 'xy' panels as the grid panels always did. `plot_distortion`
gains `s` and `g(s)` with it: it was the one compositor in `plots/` that
drew no axis labels at all, so this brings it into line with the rest of
the package and keeps both sides of the seam pinned to a single baseline.
A visible change to `Distortion.plot`.

The baseline regenerates for both changes, and the measured conversion
residual is now **0**: the rendered ChartDoc reproduces the compositor
pixel for pixel, where it was RMS 0.05 at a202. The gate pins are confirmed
by the author and no longer provisional (matplotlib 3.10.9, RMS tolerance
2.0). The image-gate modules now force the Agg backend, which `--regen`
always did: without it they inherited whatever interactive backend was
active and failed intermittently on an unrelated Tk toolkit error.

Chart IR version 1 is now signed off and **closed to additions**.

## 1.0.0a208

**[Chart-IR] pass two closes its sign-off gate, part one:
[Chart-Grid-Overlays].** The author's picks on the six inventory judgment
calls are in (`dev/chart-inventory.md` section E, all six agreed) and the
schema changes they imply land here. Plan: `dev/plan-chart-schema-signoff.md`.

The representability check the plan promised found a real gap: the twelve-plot
bivariate panel (`pedagogy.plot_twelve` panel (1,3), a contoured joint density
with the `x + y = c` iso-total lines drawn over it) could not be expressed,
because a grid panel refused to carry anything but its surface. `ChartDoc`
now validates the honest rule: a 'heatmap' or 'surface' panel carries
**exactly one** surface series **plus any number of x/y overlays**, drawn over
the mesh in document order; an 'xy' panel still refuses a surface. The
diagonals ride as ordinary two-point line series with the new `iso_total`
role, so no new geometry concept and no new `Mark` orientation enter the
schema. The matplotlib renderer draws overlays neutral and thin over the
mesh, and pins the window the grid set so a diagonal reaching past the data
cannot widen it. `tests/test_charts_ir.py` builds the panel and asserts it
validates: representability only, per the 1.0 scope, with no emitter
scheduled.

Judgment call J5 settled with it, and it is a constant rather than a look
change: `LOG_FLOOR = 1e-15` joins `constants.py`, the one float-dust floor for
anything drawn on a log scale, and the renderer's private `_LOG_FLOOR`
collapses onto it. It lives in `constants.py` rather than in `charts/` because
the `plots/` compositors need the same value and are not chart-IR code, so the
one definition is shared without touching the charts-plots dependency
direction. Emitters adopt it as their gap floor as each conversion lands and
the compositors converge chart by chart under the image gate, so nothing
outside the charts lane changes appearance here.

With this and the naming pass that follows, chart IR version 1 is **signed off
and closed to additions**: a chart the schema cannot express either changes the
schema through a fresh author decision or stays bespoke and is listed as
bespoke.

## 1.0.0a207

**[Exhibits-Waterfall]: `economic_waterfall`, the flagship business exhibit.**

The margin walk, gross through what each layer cedes to net, with the margin
evaluated at every stop. Available only on a P&L with a tower, since a single
group ledger has one margin row and no walk to draw. RAW and INSURER serve the
same table: here the exhibit **is** the translation, so there is no underlying
frame to pass through.

Two blocks, keeping one unit per column. **walk** carries currency: the margin
at each step and its 1-in-100 outcome on each basis. **evaluation** carries the
dimensionless readings: premium and margin spent against the gross block, the
combined ratio, margin over its own standard deviation, and margin over
required capital on each basis.

The point of the exhibit is the pair of 1-in-100 columns, and the arithmetic
bears it out on the shipped fixture. The **diversified** basis is each step's
margin conditional on the whole book landing at its own 1-in-100, read
straight off the ledger's kappa column, so it **foots**: `-1802.99 + 422.99 =
-1380.00`. The **standalone** basis is each step's own 1-in-100 from its own
grid distribution, and tail measures do not add, so it does not:
`-1840 + -100 = -1940`, against a closing `-1380`. The gap between the columns
is the diversification benefit, per layer, made visible. Both are asserted in
the tests rather than left as prose.

Capital is `M / -M_100`: the 1-in-100 outcome is negative, so its negation is
the injection required and the ratio reads as a return on it. On the fixture
the story lands in one number: buying the layer spends 10% of gross premium
and 16.3% of gross margin, and lifts return on capital from `0.166` to
`0.182`. The cell is **blank** where a step calls for no capital, which is what
a purchased layer does in the adverse state, where it releases capital rather
than consuming it; the caption says so and points at the diversified column as
the meaningful one there.

The return period is a module constant, `WATERFALL_RETURN_PERIOD = 100`, not a
kwarg: which capital level to evaluate at is a business choice, and the exhibit
is the place to change it rather than a configuration surface to build.

When a ledger shares no atoms (the stitched and one sweep routes) no
conditioning was possible, so the diversified column blanks and the caption
says why. Step discovery reads the ledger plan rather than pattern matching the
index, because a step's own result and the running net after it both sit under
`Margin` and only the plan tells them apart; that also picks up tier subtotals
on a multi layer peel for free.

Fixtures `EX.Tower` (shares atoms, kappa ladder) and `EX.Peel` (stitched,
marginal ladder) drive both the tests and the snapshot corpus, now 102 entries.
Both join `decl-testers.agg` section EX. A latent defect from `a206` is fixed
in passing: the `EX.Tower` corpus line carried no `;` terminator, harmless
while it was the last line in the file and a parse error the moment one
followed it.

## 1.0.0a206

**[Exhibits-Economic-Insurer]: the P&L accounting exhibits get their business
reading.**

`economic` (the ledger) keeps its shape under INSURER and gains the three
things a reader needs in order not to misread it. A caption that says the
ladder semantics **aloud**: the kappa columns are scenario states, not per row
quantiles, so they foot down the sheet, and `κ01` is the adverse state under
the payoff convention, which is why a loss sensitive premium correctly reads
high there. The caption is regime aware: a ledger with no shared atoms (the
one sweep and stitched routes) falls back to plain `P` headers where no
conditioning happened and the ladder does **not** foot, and the caption says
that instead. Row flags read off the ledger plan: the grand result is the
bottom line (`total`), each group's and each tier's own result and the grand
side totals are `subtotal`, and a running net is `muted`, being a cumulative
reading aid rather than a booked line. Legs are never flagged. A single group
ledger has no grand result, so its one group result is promoted from subtotal
to total rather than leaving a sheet of subtotals with no bottom line.

`economic_ratios` splits into three blocks under INSURER, per the reporting
rule that a column carries one unit: `amounts` (P, L, E, C, M, currency,
signed in the gross direction so the margin identity holds exactly), `ratios`
(LR, ER, CR, the three `E_` means of ratios, and the two share columns,
declared as `ratio_cols` so they render as percentages), and `legs`. The
captions carry the two facts that are easy to get wrong: ratios are re-derived
from each block's own amounts and never averaged from the blocks below, and
the plain and `E_` pairs agree identically for a deterministic premium and
part company exactly when premium is random and correlated with loss, which is
what a retro, a swing, a slide or a profit commission is.

**Per measure formats, where a measure is a column.** `MEASURE_FORMATS`
applies `CV` as `.1%` and a fixed format to `Skew`, on the summary card and on
the ledger, both of which carry measures across the columns. It cannot apply
to the canonical moment store, where measures run *down* a column; that
asymmetry stays the open question in the plan. One deviation to flag: `Skew`
was asked for as `.3g`, three significant figures, which greater_tables sugar
does not express (its kinds are `f`, `d`, `%`, `e` and `s`, with no `g`).
`.3f` is the nearest available and reads the same for the skews actually seen,
differing only in trailing zeros on large values. Switch the constant the day
greater_tables grows a `g` kind; nothing else needs to change.

Snapshot corpus stays at 78 entries with new content. `EX.Tower`, a two step
walk carrying group, running net and grand rows plus a kappa ladder, joins
`decl-testers.agg` section EX and round-trips through both corpus suites.

## 1.0.0a205

**[Exhibits-Package-Split]: `exhibits.py` becomes `exhibits/`, and passthrough
exhibits become one line each.**

The module reached 950 lines, past the 800-line trigger the plan set for
itself, so the split landed before the economic work and the waterfall add
more. It mirrors `plots/` and the public import path is unchanged:
`from aggregate import exhibits` still works exactly as before.

| module | holds |
|---|---|
| `_core.py` | the machinery, class agnostic: `Perspective`, `Exhibit`, the registry, the frame and IR stages, the shared translation helpers |
| `_aggregate.py`, `_portfolio.py`, `_pnl.py`, `_bivariate.py`, `_distortion.py` | one per class, holding **that class's business translation**, which is where you go to edit how an exhibit reads |
| `__init__.py` | the public surface, plus the **passthrough manifest** at its foot |

`_core.py` imports no domain class, which is what keeps the package free of
import cycles: the per-class modules import it, never the reverse.

**`register_simple_exhibit(name, title, frame_attr, classes, predicate=None)`**
declares a passthrough exhibit over one frame in a single line. It registers
no insurer override, so INSURER equals RAW by the default rule and both
perspectives serve the same table with no extra code; registering an override
later changes only that (exhibit, type) pair. Calling it twice for one name
extends the exhibit to more classes rather than replacing it. Seven
hand-written passthrough builders collapsed into five manifest lines.

**Two diagnostics join**, both through the new helper and both gated on the
realized grid: `bs_window` (`bs_window_df`, on Aggregate, Portfolio and
BivariateAggregate) and `tail_behavior` (`tail_behavior_df`, on Aggregate and
Portfolio). They are the app's "More" material.

**`pnl_ledger` and `pnl_ratios` are renamed `economic` and
`economic_ratios`**, restoring the property that an exhibit name mirrors its
frame, which now holds everywhere: `summary_df` to `summary`, `economic_df` to
`economic`, `dependency_df` to `dependency`. The exhibit registry is ten
entries.

Snapshot corpus grows to 78 (the two renames, the two new diagnostics across
their kinds). The app's `check-exhibits.py` now reads the exhibit list from
the library rather than a literal, so a new exhibit joins the sweep with no
edit there; it reports clean, and `stats` on a P&L reads 26 rows raw and 17
insurer through the API, confirming the `a204` engine delegation end to end.

## 1.0.0a204

**[PnL-Economic-Frames]: the P&L accounting frames get their own names, and
`stats_df` finally means one thing. BREAKING.**

Found during the exhibits review: on a `PnL`, the `stats` and `pnl_ledger`
exhibits emitted **byte-identical documents**, same hash, same single block.
Two exhibit names for one frame is a synonym. The root cause was deeper than
the exhibit layer. `stats_df` meant four different things across the
first-class contract: the `(component, measure)` by view moment store on
`Aggregate` and `Portfolio`, a column of `D_g` statistics on `Distortion`, a
`(basis, stat)` by axis table on `BivariateAggregate`, and on `PnL` not a
statistics frame at all but the ledger sheet with the kappa scenario ladder.
The FCC audit can only check that a name exists, never that it means the same
thing, so this drifted quietly.

| was | is | note |
|---|---|---|
| `PnL.stats_df` (the ledger) | `PnL.economic_df` | it is an accounting view, and now says so |
| `PnL.ratio_df` | `PnL.economic_ratios_df` | joins the accounting family |
| absent | `PnL.stats_df` delegating to `self.engine.stats_df` | one meaning everywhere: the moment store of a book |
| `PnL.validation_df` | unchanged | its leg rebucketing audit is a real check on the P&L's own construction and does not delegate |

`economic_ratios_df` is **not** a view of `economic_df`, which is why it keeps
its own frame: splitting expense from commission needs `Leg.kind`, and the
`E_LR` / `E_ER` / `E_CR` columns need the per-atom vectors, neither of which
survives into the ledger sheet.

`PnL.engine` is `None` on a hand-built kernel P&L, so `stats_df` returns an
**empty** `DataFrame` there rather than `None` or a raise: the FCC contract
says the member exists, callers reach it defensively, and greater_tables
renders an empty frame cleanly (verified: a valid 0 by 0 document with a
hash). The `stats` exhibit's insurer view says so in words instead of serving
a blank table.

Consequently the `stats` exhibit on a P&L now serves the engine's moment store
and takes the ordinary insurer treatment, the raw noncentral moment drop, so
it is a genuinely different document from the ledger. `_drop_raw_moment_rows`
gained an empty-frame guard.

Moved in the same commit, per the one-coherent-unit rule: `_pnl.py` (both
properties plus 29 docstring references), the `qd` comment, the exhibit
registrations, `dev/FEATURES.csv` (the `stats_df` note rewritten, an
`economic_df` row added, `ratio_df` renamed; the auditor passes),
`docs/2_aggregate_overview/pipeline-pnl.rst`, `docs/3_reference/3_x_PnL.rst`,
two lines of `features.rst`, 148 references across twelve test files, and the
app's `_CSV_FRAMES`, which gains `economic_df` and `economic_ratios_df` entries
so the ledger stays reachable until the economics tab lands.

Docs are pending a rebuild.

## 1.0.0a203

**[Exhibits-Module] phase four opens, [Exhibits-PnL-Translation], raw stage.**
`pnl_ledger` (source `PnL.stats_df`: the full ledger by (Side, Label), or
(Step, Side, Label) on a tower, with the kappa scenario ladder) and
`pnl_ratios` (two blocks: `ratio_df`, the per block amounts and LR / ER / CR
ratios, and `legs_df`, the itemized declared legs) registered for PnL as RAW
passthroughs. No insurer override is registered: INSURER equals RAW by the
default rule, deliberately. The flagship INSURER business framing (captions,
footing rules, Side sign presentation, the xpnl tower ledger reshape) is the
plan's author gate and lands only after that review; a written draft with
proposed captions, flag rules and the ratio card reshape is appended to
`dev/plan-exhibits.md` for it (left uncommitted for the author). Snapshot
corpus grows to 60; PnL now serves five exhibits.

## 1.0.0a202

**[Chart-IR] pass three continues, [Chart-Conversions]: distortion g(s),
the first conversion.** `charts/_emit_distortion.py`:
`chart_distortion(dist, dual=True)` reads the knot-spliced `density_df`
grid verbatim (splicing is meaning, exactly why the compositor reads the
frame rather than re-evaluating `g`) and emits one equal-aspect 'xy' panel:
the distortion curve, optionally its dual, and the identity diagonal as a
role-carrying series. Registered, so `available_charts` on a `Distortion`
answers `['distortion']`.

The renderer grew its 'xy' realization: role-styled curves ('identity'
draws neutral, thin, legendless; band series fill between `y` and `y2`),
gaps broken at `None`, mark verticals and horizontals, pinned round ticks
on a probability unit interval, equal aspect from the panel, the compact
square figure for an equal-aspect single panel, and the legend rule
(upper left, x-small, only with more than one named series). Axis *labels*
are deliberately not drawn yet: the converted compositor never drew them,
and the gate below is pixel parity; whether the mpl look should gain them
is an author call recorded with the gate pins.

The acceptance gate landed with the conversion:
`tests/test_chartdoc_render.py` compares the rendered ChartDoc against a
committed baseline PNG generated from the *compositor* (the before side),
plus a second test pinning the compositor itself to the baseline so
neither side of the seam drifts silently. Measured conversion residual at
setup: RMS 0.05 on the 0-255 scale, sub-perceptual antialiasing only.
PROVISIONAL PINS awaiting author confirmation per the plan's gate-setup
step: matplotlib 3.10.9 (tests skip on other versions rather than fail on
font differences) and RMS tolerance 2.0. `--regen` on the test module
rewrites baselines.

The paired app commit (the app distortion exhibit consuming
`chart/distortion` through the adapter, which needs the adapter's 'xy'
realization) is the next app-side step and is not part of this commit.

## 1.0.0a201

**[Exhibits-Module] phase three, [Exhibits-Reins-Insurer].** The `reins`
exhibit registered for Aggregate and Portfolio: two blocks,
`reins_stats_df` (the layering / end to end moment store) and
`reins_summary_df` (the per stage cession impact on the eight validation
columns). Availability is gated on a cession being present (any unit, on a
portfolio) and the grid realized; the predicate reads `occ_reins` /
`agg_reins` directly, so it costs nothing. The INSURER view drops the raw
noncentral moment rows (ex1/ex2/ex3) from the stats block through the same
`_drop_raw_moment_rows` helper the stats exhibit now shares, captions both
blocks (conditional per layer columns vs unconditional totals on an
Aggregate; end to end gross / ceded / net per unit plus the convolved total
on a Portfolio; Change as rebucketing error on the leading row, percentage
cession impact on ceded / net), and flags the portfolio summary's `total`
block. RAW passes both frames through untouched.

Tests: ceding Aggregate and Portfolio fixtures join `tests/test_exhibits.py`
(availability, the two block shape, the moment drop, total flags, and the
not available path without a cession) and the snapshot corpus, now 56
committed canonical_dict snapshots. Fixture programs (bv and pnl fixtures
included, previously missing) added to `decl-testers.agg` section EX; both
DecL corpus suites round-trip them.

## 1.0.0a200

**[Exhibits-Module] phases one and two, [Exhibits-Scaffold] plus
[Exhibits-Stats-Validation].** New `aggregate/exhibits.py`
(`dev/plan-exhibits.md`, approved 2026-08-04): business exhibits translating
the first class frames to greater_tables table-document IR. The library owns
meaning, the app owns arrangement; the app's row flag, caption and raw moment
drop knowledge starts migrating in here. Purely additive: the core never
imports it, it is not star exported (`from aggregate import exhibits`), and
`import aggregate.exhibits` imports neither matplotlib nor greater_tables
(two new guards in `tests/test_plots_boundary.py`).

The surface: `Perspective` enum (raw / insured / insurer / reinsurer; 1.0
implements raw and insurer, the other two are stable vocabulary), frozen
`Exhibit` dataclass (`ir_blocks`, sha256 `hash` over the block doc hashes for
ETags, `to_payload()` canonical dict envelope), eight generic singledispatch
exhibit functions (`summary`, `tail`, `stats`, `validation`, `reins`,
`pnl_ledger`, `pnl_ratios`, `dependency`) with open registration, the
`EXHIBITS` registry with per object availability predicates,
`available_exhibits(obj)` derived from the registries so it cannot go stale,
`exhibit_frames` (the pure pandas stage, fully testable without GT), and
`build_exhibit` (the only lazy greater_tables import, naming the extra when
missing). INSURER equals RAW unless a per (exhibit, type) override hook is
registered; every override is an explicit reviewable delta on the raw frame.
All served frames pass through `LabeledMixin._relabel`; titles use
`_title_name`.

Registered at this version: `summary` for all five first class classes with
insurer captions and total / subtotal row flags on Aggregate and Portfolio
(frequency percentiles blank by design noted in the caption); `tail` for
Aggregate and Portfolio with the 1 in 200 / 1 in 250 capital anchors
emphasized and the portfolio total block flagged; `stats` for all five, the
insurer view dropping the raw noncentral moment rows (ex1/ex2/ex3, 26 rows
to 17) on Aggregate and Portfolio; `validation` for all five, the insurer
view emphasizing failing rows (moment failures via the object's `Validation`
flags on Aggregate and Portfolio, `Pass == False` check rows on Distortion
and BivariateAggregate; the PnL audit has no gate and passes through); and
`dependency` for BivariateAggregate (`dependency_df` plus `axis_support_df`,
no override, none needed today).

Tests: `tests/test_exhibits.py` (65 cases: availability by kind, frame stage
structure, relabeling honored, error paths, and committed
`canonical_dict` snapshots per (exhibit, perspective, kind) via
`tests/capture_exhibit_snapshots.py`, 36 snapshots guarding both the
business translation and IR drift; the snapshot file drives the case list).
Fixture programs added to `decl-testers.agg` section EX; the exhibits
surface documented as a comment row in `dev/FEATURES.csv`.

**The `exhibits` extra is documented but commented in `pyproject.toml`:**
greater_tables 6 is not yet on PyPI (latest published is 5.3.0), and an
active unresolvable extra would break `uv sync --all-extras`. It activates
verbatim when GT 6 publishes; until then the sibling checkout is path
installed, as `aggregate_api` already does.

Open with the author: per measure formats for the stats insurer view
(greater_tables formats are per column, the canonical store mixes measures
down a column), and whether the PnL validation audit gets a failure gate.

## 1.0.0a199

**[Chart-IR] pass three opens, [Chart-Surface-Pilot], library side.** The
first emitter and the generic renderer, end to end on the chart that forces
the IR to be designed from semantics rather than matplotlib's vocabulary.

`charts/_emit_bivariate.py`: `chart_joint_surface(bv, display_log2=None)`
pulls the joint matrix off a `BivariateAggregate`, block-sums it to the
display grid inside the emitter (the app's `surfaceGrid` mass-preserving
reduction, migrated: fixed prefix blocks, short final block, non-finite as
zero, clamped right-edge labels per inventory judgment call J3; verified in
tests against a transliteration of the surface.js loop), labels the axes
from the resolved component labels (ending the app's `axisNames` stats_df
hack), and returns one 'surface' panel with z linear and
`meta['z_log_ok'] = True` declaring the log-height toggle meaningful.
Registered in the `CHARTS` registry with an in-memory-density predicate, so
`available_charts(bv)` now answers `['joint_surface']` (the massive
disk-backed pyramid stays bespoke). Values are display-cell masses; the sum
of the emitted grid equals the joint's mass to 1e-10 in the end-to-end test.

`plots/_chartdoc.py`: `plot_chartdoc(doc, ax=None, strict=False,
log_z=False)`, the one generic mpl renderer, inside the mpl boundary,
establishing the capability-declaration pattern: matplotlib has no faithful
3-D surface, so a 'surface' panel renders as its honest 2-D reading (a
`pcolormesh` projection with a contour overlay) and the title is stamped
"(projection)"; `strict=True` raises `ChartCapabilityError` instead;
'heatmap' renders natively; unrealized kinds refuse loudly. Renderer-side
choices only (the sequential white-to-house-primary ramp, the colorbar the
massive heatmap never had, figure sizing); `log_z` is honored only when the
document declares `z_log_ok`, with the app's floor-not-holes rule (one
decade under the smallest mass present). Exported as
`aggregate.plots.plot_chartdoc`.

Tests: `tests/test_chart_surface_pilot.py` (reduction equivalence to the
app algorithm on an awkward grid with a NaN cell, mass conservation,
passthrough, document shape and orientation, byte determinism, registry
dispatch, projection stamp, strict refusal). The app side of the pilot (the
`/chart/joint_surface` route, the generic `chartdoc-to-echarts.js` adapter,
and the surface.js reduction deletion) follows as one aggregate_api commit.

## 1.0.0a198

**[Chart-IR] pass two, [Chart-Schema].** Chart IR schema v1: new
`aggregate/charts/` package. `charts/ir.py` holds the frozen dataclasses
(`ChartDoc`, `Panel`, `ChartAxis`, `ChartSeries`, `Mark`, `SurfaceData`),
`CHART_IR_VERSION = 1`, and hand-rolled `canonical_dict` / `canonical_json` /
`doc_hash` / `stamp` mirroring greater_tables' determinism contract exactly:
deterministic field presence (structural fields always, optional fields only
when they differ from their defaults), NFC-normalized strings, sorted-key
compact UTF-8 JSON with NaN forbidden, sha256 truncated to 12 hex, and `hash`
/ `generator` excluded from the hashed form so stamping does not perturb the
digest. No pydantic; aggregate's dependency set is unchanged. Documents
validate on construction (reference integrity across panel / axis / series
ids, payload-kind agreement, xy length agreement) and are immutable all the
way down.

Vocabularies (panel kinds, axis units, series roles, mark roles) are
documented strings, not enums. Two fields flagged in the inventory's judgment
calls are drafted in and are one-line removals if vetoed:
`ChartAxis.reciprocal_of` (the return-period twin as a semantic pairing) and
`ChartSeries.y2` (band series); `Panel.aspect` carries the semantic
equal-aspect cases (unit square, complex plane). There is deliberately no
`extra_mpl_kwargs` and no renderer passthrough of any kind.

`charts/__init__.py` is the public surface (`from aggregate import charts`;
nothing star exported): the IR names plus the `CHARTS` registry,
`register_chart`, and `available_charts(obj)` derived from singledispatch
registries plus per-chart predicates, mirroring the exhibits capability
mechanics. No emitters yet; the registry fills from the pilot onward.

Representability, checked on paper against the twelve `plot_twelve` panels
per the plan: the density / log density pair and the kappa, alpha, beta
families are multi-series xy panels off `density_df` `exeqa_*` and
`allocation_diagnostics` columns with roles carrying the objective vs
distorted distinction; the per-unit S / gS families are xy with a backdrop
role; the margin panels and the Lee-orientation stand-alone / natural M
panels use `read_axis='y'` plus `y2` band series; the bivariate density panel
is a `SurfaceData` grid (its sparse-scatter branch is a renderer realization,
not IR). Nothing in the twelve panels, the bounds plots, or the ft
illustrations demands a fourth panel kind.

Tests: `tests/test_charts_ir.py` (structure, validation, determinism,
capability, and the boundary check that `import aggregate.charts` does not
load matplotlib). The plan puts that assertion in `test_plots_boundary.py`;
it lives in the charts test module for now because `test_plots_boundary.py`
was mid-edit in the parallel `[Exhibits-Module]` workstream, and migrating it
later is a two-line move.

The schema sign-off gate stands: the author reviews the field list before
emitters multiply beyond the pilot.

## 1.0.0a197

**[Chart-IR] pass one, [Chart-Inventory].** `dev/chart-inventory.md`: one row
per drawing across the 8 app charts, all 20 `plots/` layer-2 compositors,
`pedagogy.py` (with `plot_twelve` broken out per panel, twelve rows), and the
`ft.py` illustrations, each split into semantic columns (data inputs, panels
and shared axes, scales, reference marks, window logic, series naming) and the
incidental column that stays renderer-side. Compiled from the app's
`twoPanelData` / `surfaceGrid` / serializer sources and the compositor bodies,
with anchors throughout. Six judgment calls are collected for author review
with a recommendation each: the twin return-period axis (recommend IR, as an
axis-level reciprocal pairing), reference-line toggles (marks in IR,
visibility renderer-side), the surface display-grid label convention
(right-edge, matching the app; the center-vs-edge divergence from the server's
`bin_density` recorded for later reconciliation), the surface default camera
(renderer override dict, angle preserved), one log floor (1e-15), and band
series (a `y2` field). The plan's review gate stands: the semantic vs
incidental split awaits author sign off; anything vetoed is a doc edit plus,
where drafted into the schema, a field removal before emitters multiply.

**Housekeeping.** `dev/plan-chart-ir.md` (approved 2026-08-04) is committed
alongside its first executed pass.

## 1.0.0a196

**[Reference-Chapter-Audit]** The API reference chapter audited against the live
package. Three checks were run over `docs/3_Reference.rst` and its leaf pages: every
autodoc target resolved, every narrative cross-reference resolved, and the
`autosummary` lists were reconciled against each module's `__all__`. Autodoc
targets went from 128 to 153.

### `aggregate.balanced_window` is now exported

`balanced_window` was defined in `utilities.py` but missing from its `__all__`.
Autodoc honours `__all__`, so the function was never documented, and the seven
places that cite it as `` :func:`~aggregate.utilities.balanced_window` `` (in
`_aggregate.py`, `bivariate.py`, `config.py`, and the reference chapter itself)
were all dead links. It is now in `__all__`, so `aggregate.balanced_window`
joins the top-level surface and those references resolve.

The same docstring pointed at `aggregate.distributions.estimate_agg_window`,
which has lived in `aggregate._bucket_window` since the a90 to a95 split. Path
corrected.

### `PnL` gets a public page

`PnL` is one of the five `FIRST_CLASS_CLASSES` and is re-exported at the top
level, but its only autodoc lived on the Internal Architecture page, which the
chapter introduces as documenting things that are *not* part of the supported
API. It was also being rendered a second time under the Distribution page's
"Severity fits and approximations" heading, because `distributions.__all__`
re-exports it.

New page `3_reference/3_x_PnL.rst`, sitting after Portfolio: the `pnl` / `xpnl`
faces, how to read the card and the sheet (marginal percentiles versus the `κ`
scenario ladder), and the `PnL` / `Leg` / `Group` / `stack_marginal_pnls`
surface. `aggregate._pnl` is documented there instead of on Internal
Architecture, which keeps `_pnl_builders` as the internal half. The Distribution
page excludes `PnL` and points at the new page.

### Newly documented

- **`aggregate.contract_terms`**, previously absent from the chapter entirely:
  the `ContractTerms` taxonomy and its six single-leg features (retro, swing,
  slide, profit commission, corridor, and the two-map `ReinstatementTerms`).
  Documented on the new P&L page, since a contract term fills one P&L leg.
- **`aggregate.decl_pygments.AggLexer`**, reached at the top level through the
  star import but undocumented. Now a "Syntax highlighting" section on the
  Parser page.
- **`aggregate.parser.INHERIT_PREMIUM`**, the `inherit premium` sentinel: in
  `parser.__all__`, but the Parser page has no `automodule`, so nothing rendered
  it. Now an explicit `autodata`.
- **`aggregate.utilities.explain_validation`**, which the Internal Architecture
  page promises is "documented on the Utilities page". It was, by `automodule`,
  but was missing from the curated list a reader scans after following that
  pointer.
- **`aggregate.tail`**: the `autosummary` listed 12 of 20 exported names. The
  eight missing (`TailClasses`, `TailInfo`, `TailRow`, `tail_class_label`,
  `thickness_label`, `occ_net_severity_row`, `describe_rows`,
  `CONCENTRATION_CV`) are added. The two underscore-private bounded-support
  tables are exported and cited by `Aggregate.bounded` / `Severity.bounded`, but
  autodoc skips private names even when listed in `__all__`, so the page now
  names them under `:private-members:`.

### Housekeeping

`BivariateAggregate`'s first-class status is stated on the Auxiliary page, along
with the reason it stays a submodule import. `aggregate.style` remains
undocumented on purpose: it is a backward-compatibility shim over
`aggregate.plots`. Dash-as-punctuation cleared from all nine leaf pages and the
chapter index.

## 1.0.0a195

**[Sharpen-Grid-Probe]** A grid that loses mass off its top end is disqualified,
however well it scores. `sharpen_df` records the deficit for every cell.

### The problem

```
agg CatXOLTower as "US Hurricane Reinsurance"
    1.74 claims
    sev lognorm 8.501 cv 14.624 splice [0 500]
    poisson
```

`a194` sharpened this to `bs = 1/64`, which scored `0.044` and **lost 2.1e-08 of
its mass off the top of the grid**. `bs = 1/32` scored `0.177` and held all of
it. The probe took the better score.

A moment score cannot see this and never will. At `bs = 1/64` the six terms are

```
u_sev_mean 0.0759   u_sev_cv 0.0081   u_sev_skew 0.0001
u_agg_mean 0.0773   u_agg_cv 0.0063   u_agg_skew 0.0014     score 0.0444
```

`u_agg_mean` sits 13x *inside* tolerance: the lost mass is out at ~1024 and moves
a mean of 10.75 by about 2e-06 in relative terms. Meanwhile the deficit is 2.1e04
times the noise floor, and it is not cosmetic. Forwards `S = 1 - cumsum` and
backwards `S` differ by exactly the missing mass, so two correct-looking pricing
routes disagree, on a cat tower, in the tail, which is the whole point of the
object.

### The fix: a gate, not a penalty

A cell carrying a genuine deficit is **disqualified while any clean cell
survives**. Not a term in the score, for three reasons: every score term divides
by its own validation tolerance and a deficit has none to divide by; the only
defensible divisor gives `2.1e04`, which would swamp the norm and so is a gate
wearing a number that means nothing; and the cost is qualitative rather than a
matter of degree.

The threshold is the one the library already uses, the level at which
`DefectiveDistributionWarning` fires, so **a cell rejected here is exactly one
that would warn when you used it**. No new constant.

```
Sharpen: 11 cells in 0.30s, best score 0.177 vs 2.12 at the centre, target 0.5.
1 better-scoring cell rejected as defective (mass off the end of the grid).
Moved: bs 1/8 to 1/32, at log2 16.
```

Soundness sits outside the existing thrift ordering, so it composes: because the
"does an affordable cell meet the target?" test now runs over *clean* cells, **a
deficit becomes a reason to grow `log2`**, which is exactly its cure. Tighten the
target on the tower and you get the fine bucket on a grid wide enough to hold it:

```python
a.sharpen(good_enough=0.1)      # bs 1/64 at log2 17, deficit 9e-13
```

### The probe gate was waving defective grids through

`sharpen` skipped the probe entirely whenever the current grid scored at or under
target. A grid with good moments and lost mass, which is exactly the `bs = 1/64`
cell above, was therefore left alone. The gate is now **at target *and* sound**.

This bites more often than it sounds. An ordinary `100 claims sev lognorm 100
cv 2 poisson` clips ~8e-09 of its tail on the auto-sized grid, so it is no longer
waved through; `sharpen` probes it and finds a grid that holds everything.

### Frame columns

Three new, none removed:

* **`deficit`** — `1 - sum(p)`, the mass the grid failed to hold.
* **`defective`** — the gate the picker applied, so its decision is visible
  rather than re-derived from a threshold.
* **`warns`** — the warning class names that fired, comma-joined.

`warnings` (the count) stays. `sharpen_description` reports how many
better-scoring cells the gate threw out, and `sharpen_explanation` says why,
since otherwise the frame reads as though the picker ignored its own minimum.

## 1.0.0a194

**[Sharpen-Grid-Probe]** The selection rule changes, `min_gain` is retired, and
a discrete severity no longer has its exact bucket searched.

### The rule: never pay more than you already are

`sharpen` now takes **the best score among the cells that do not grow `log2`**.
It grows by one only when nothing at the current grid size or smaller reaches
`good_enough`, and something at the larger size does.

The old rule, "the smallest `log2` that meets the target", was written for a
fixed 3x3 where each row held one arbitrary sample of the bucket axis. Once
`a193` made every row search to its own trough, that rule started taking a much
worse score in exchange for a grid saving nobody asked for:

```
d_log2 │    -4       -3       -2       -1      0      1
   -1  │     —  2.06428  0.17708  0.64241  2.122  6.386
    0  │ 2.043  0.04443  0.17671  0.64241  2.122  6.386
    1  │ 0.010  0.04405  0.17671  0.64241  2.122  6.386
```

`a193` picked `0.177` at `log2 - 1`, the cheapest cell clearing the bar.
`a194` picks `0.0444` at the current `log2`. Reducing `log2` is a bonus, not a
goal: it is now picked up by the tie rule (cells within 25% of the best score
count as tied, and ties break on `log2`, then on how far the bucket moved, so a
competitive centre wins and the grid is not churned for nothing).

**`min_gain` is gone.** Its job was "is the best cell enough better than the
centre to bother", which the rule above answers directly. `good_enough` is now
the only judgment knob, with two uses: the probe gate, and the growth trigger.
`good_enough=0` means "probe everything, never grow".

### The `log2 + 1` row is computed lazily

It costs twice as much per cell as the current row and, under the rule above, is
only ever consulted when nothing affordable reaches the target. So it is no
longer searched up front. On the table above it was pure waste: rows `0` and `+1`
agree at every bucket except one, because that book is resolution-limited, not
extent-limited.

### Discrete severity: the bucket is already exact

When every severity atom is a whole number of buckets there is nothing to search
on the bucket axis. A coarser bucket scatters the atoms off their own values; a
finer one only wastes grid. So the bucket is **pinned** and only `log2` is
probed, which is what a failing discrete object actually needs, its problem being
extent.

```
d = build('agg Dice dfreq [3] dsev [1:6]')
d.sharpen(good_enough=0)
# Sharpen: 2 cells in 0.02s (discrete: bucket pinned, grid size only), ...
```

Detection reuses `Aggregate._severity_lattice`, the gcd of the integer atoms,
taking the gcd across units for a `Portfolio`. A bucket that is exact but finer
than it needs to be gets a note naming the coarsest exact one: *"bs 1 is finer
than it needs to be: the atoms sit on a lattice of 5."*

### Also

* **`log2_cap` defaults to 20**, was 24.
* A walk that runs out of `bs_limit` **while still improving** says so in its
  `note`, which is a different fact from turning, and `sharpen_description`
  surfaces it when the winner sits there. The stop reason is appended to a
  cell's note rather than overwriting it, so a failed cell keeps its exception
  text.
* The current-`log2` row is always searched. It used to be dropped along with
  the row below it when `log2` sat under `SHARPEN_LOG2_FLOOR`, which left a
  small-grid object with nothing probed at all.

## 1.0.0a193

**[Sharpen-Grid-Probe]** Punch-ups to `a192`. The score is now a first-class
statistic in its own right, and the probe walks instead of taking one step.

### `validation_score`, a property

```python
a = build('agg X 100 claims sev lognorm 100 cv 2 poisson')
a.validation_score      # 0.391
```

The quantity `sharpen` minimizes, exposed on `Aggregate` and `Portfolio` because
it is worth watching on its own. **In units of the validation tolerance**, so
`<= 1` means the object passes at its own `validation_eps` and `1` is exactly
the pass boundary. Where `valid` says whether a line was crossed, this says by
how far, which is what makes it comparable across grids.

It lives in `_validation.py` with the rest of the validation family, not in the
grid code: `validation_score(obj, power=2)` for the other powers,
`validation_score_terms(obj)` for the per-term detail, `SCORE_TERMS` for the
tolerance multiples. The `_bucket_window.sharpen_score` of `a192` is gone; it
was a second name for this.

### The probe walks

Still three rows at `log2 - 1` / `log2` / `log2 + 1`, but each row is now a
**line search out from the current bucket**: `bs` is doubled until the score
stops improving, then halved likewise, capped by the new `bs_limit` (default
`16`, four doublings each way, must be a power of two).

This is a large practical difference on a badly sized grid. Forcing the book
above onto `bs=1/8, log2=14`:

```
a192:  3494 -> 908 -> 5.49 -> 0.391       four calls
a193:  3494 -> 1.61 -> 0.463              two calls, and it validates
```

The line search is well posed because the score has a single trough in `bs` at
fixed `log2`: a larger bucket buys extent and loses resolution, so the two error
families trade off. A severity whose atoms land on grid points at some buckets
and not others could in principle dip again past the turn and be missed;
`bs_window` sizes those by its exact-discrete method, so they rarely reach a
probe. Cost is unchanged on a grid near its optimum, eight evaluations, because
the search stops at the first cell that fails to improve.

### `sharpen_df` is indexed by the offsets

```python
a.sharpen_df.score.unstack('d_log2')

d_log2        -1         0         1
d_bs
-1      5.486182  1.605396  0.390593
 0      1.605666  0.391127  0.074544
 1      0.395764  0.091433  0.107531
 2      0.251204  0.239852       NaN
 3      5.213805       NaN       NaN
```

`(d_bs, d_log2)` moved from columns to the index, so the picture is one unstack
away. The rows are ragged, since each searched to its own turning point, and
cells never visited come back `NaN`. Reading down a column is constant grid
size; reading an anti-diagonal is constant extent.

That table also shows the parsimony rule earning its place: the outright best
cell is `0.0745` at `log2 17`, and `sharpen` picks `0.0914` at `log2 16`, half
the memory for a score 23% worse and still far inside the target.

### Narrative

* A sub-unit `bs` reads as **the binary fraction it is**: `bs 1/8 to 2`, not
  `bs 0.125 to 2`.
* A move names **only what changed**. A bucket-only move said `log2 16 to 16`,
  which reads as a bug rather than as "unchanged".
* `sharpen_description` opens with `Sharpen`, capitalized.

## 1.0.0a192

**[Sharpen-Grid-Probe]** A new `sharpen()` audits the FFT grid the bucket
estimator chose, and moves to a better one when the win is large.

```python
a = build('agg X 100 claims sev lognorm 100 cv 2 poisson')
a.update(log2=14, bs=1/8)     # a grid far too small to hold the book
a.sharpen()
print(a.sharpen_description)
# sharpen: 9 cells in 0.22s, best score 908 vs 3.49e+03 at the centre,
# target 0.5. Moved: bs 0.125 to 0.25, log2 14 to 15.
```

`update` *chooses* a grid from the analytic moments before any FFT runs.
`sharpen` *audits* that choice afterwards: it re-updates the object on the eight
neighbouring cells (half, same, double `bs` by one step down, same, up in
`log2`), scores each against the analytic moments, and moves only on a large
win. Available on `Aggregate` and `Portfolio`.

### The score

Six terms, severity and aggregate mean, CV and skewness, read from the canonical
`stats_df['error']` and each divided by **its own** validation tolerance (`eps`
for a mean, `10 eps` for a CV, `100 eps` for a skewness). Those are the
multipliers `valid_aggregate` already applies, so the units are tolerance:
**`score <= 1` means the object passes validation**, and `1` is exactly the pass
boundary. The score does not move when `validation_eps` changes.

Combined as a power mean, `(mean_i u_i ** power) ** (1 / power)`; `power=1`,
`2` (default) and `numpy.inf` all land on the same scale. A term whose
theoretical value is zero or infinite drops out, as it does in validation; a
non-finite empirical value scores infinite.

### Reading the frame

`sharpen_df` is tidy, one row per cell, carrying the offsets `d_bs` / `d_log2`,
the realized grid, the score and its six terms, the aliasing ratio, the
validation verdict, timings and a `note`. Extent is `bs * 2**log2`, so cells on
an anti-diagonal share an extent and differ only in resolution. Reading them
together says whether a grid is **extent-limited**, widen it, or
**resolution-limited**, refine it. `sharpen_description` and
`sharpen_explanation` say it in prose.

### Two gates, one number

`good_enough` (default `0.5`) sets both. If the current grid already scores at
or under it, **nothing is run at all**; `good_enough=0` never clears, which is
how a probe is forced. Among cells that meet it, the winner is the one with the
**smallest `log2`**, then the best score: more grid almost always helps a
little, and a plain argmin would grow `log2` on nearly every object and double
everyone's runtime for a negligible gain.

When no cell reaches the target the best available step is taken anyway, if it
beats the centre by `min_gain` (default `2`). Re-running `sharpen` re-centres
the probe and continues, so a badly starved grid walks back to a valid one over
a few calls.

`execute=False` makes the probe pure diagnosis: the original grid is restored,
along with the `sev_calc` / `discretization_calc` / `normalize` / `padding`
settings that a bare re-update would silently reset to their defaults. Those
same settings are locked across every cell, so the nine differ in `bs` and
`log2` and in nothing else.

### Also

* **`update(..., sharpen=True)`** opts a single update into auto-sharpening.
  Off by default and **not** turned on by `build`, because a probe costs eight
  extra updates. `BivariateAggregate` has no sharpen: the probe is quadratic in
  the joint grid.
* A `pnl` gets it through its engine. `build_many` updates the deferred engine
  and only then snapshots the P&L, so `build(prog, sharpen=True)` reaches it
  pre-construction. There is deliberately no `PnL.sharpen`, which would leave a
  built ledger on a stale grid.
* Portfolio probe cells run `add_exa=False`, the dominant cost of a portfolio
  update and irrelevant to the moments; the chosen cell is then updated in full.
  The portfolio score reads the **total only**, so a well resolved total can
  still hide a poorly resolved unit. `sharpen_explanation` says so.

### Breaking

* **`Aggregate.focus` is now `Aggregate.center_window`.** The name was wrong for
  what it does, a no-recompute re-slicer returning the central window of
  `density_df` holding `1 - p` of the mass, and it blocked the good name. No
  deprecation shim.

## 1.0.0a191

**[First-Step-Label]** The direct block of a walk now reads the labels you
declared. **Breaking for any exhibit that names the first step or the default
premium leg.**

Given

```
xpnl Deal as "ABC" 33333.33 premium as "XYZ" less
  agg Deal_e as "LLL" ...
```

the first block of `stats_df` reads

| Step | Side | Label |
|---|---|---|
| `ABC` | `Consideration` | `XYZ` |
| | `Obligation` | `LLL` |
| | `Margin` | `LLL` |

### What changed

* **The first `Step` is the P&L's own `as` label**, defaulting to `Gross`. It
  used to be the *engine's* `as` label, which put the subject business's name
  in the step column and left the deal's own name nowhere. `xpnl` and every
  variant (guaranteed cost, peel, variable rating, reinstatements) agree.
* **The direct block's margin row takes the engine's `as` label too**,
  defaulting to `Gross`. The subject business names its own margin the way it
  already names its loss leg. This replaces `Direct`, introduced two versions
  ago in `a189`, which never survived contact with a labelled sheet.
* **The default consideration leg label is `Premium`**, capitalized, matching
  `Loss`. It was `premium`.

The `sell` group's margin label is carried by the new **`Group.margin_label`**,
so a hand-built ledger can name it directly. It is read only on the direct
block of a ledger that buys something, which is the `a189` gate unchanged:
a plain single-group `pnl` and a ledger merging two sold books keep `Total`
throughout, since neither has anything to be direct *of*.

### Migration

* First-step keys move: `('Gross', ...)` becomes `('<your as label>', ...)`,
  and stays `('Gross', ...)` only when the P&L carries no `as` clause.
* `('Gross', 'Margin', 'Direct')` becomes `('Gross', 'Margin', 'Gross')`, or
  your engine's `as` label.
* `('Consideration', 'premium')` becomes `('Consideration', 'Premium')`, and
  likewise the `density_df` key. `Leg.kind` values and expense-basis strings
  are unchanged and stay lowercase: only the default *label* moved.
* A step's own result is still the first `Margin` row of its block in plan
  order, whatever the label reads.

## 1.0.0a190

**[Evaluate-Positions-Only]** Two corrections to the `a188` acceptability
panel, both about what a row *is*.

### `total impact` is no longer evaluated

The impact row is the grand result less the first group's, so it measures what
the ledger's purchases did to the bottom line. That makes it a **difference
between two positions, not a position**: nobody holds it, so the breakeven
stress it survives is not a question with an answer. It is now excluded from
`evaluate` outright, whether or not it happens to carry a law (it does on an
unpeeled walk; on a stitched peel it is a delta of two statistics whose sides
ride different marginals and has none).

The ceded program **as a position** is unaffected and still reported, under the
tier subtotal rows (`All occurrence result`, `All aggregate result`).

`_MARGIN_KINDS` drops `'total_impact'`. That made the panel's
no-distribution branch unreachable, since the impact row was the only ledger
row ever built as a `_DeltaRow`, so `aggregate._pricing.no_distribution_panel`
is **deleted**. `total impact` remains an ordinary ledger row on `stats_df`,
`summary_df` and `density_df`; only `evaluate` skips it.

### A netted row reads `role == 'net'`

A running net and the grand result net buying against selling. That is neither
side, so calling them `sell` was wrong. `EVAL_SIGN` gains `'net': 1.0`: they
are already in payoff orientation and read as booked, exactly as before, but
the panel now names them for what they are.

| row | `role` |
|---|---|
| a `sell` group's own result | `sell` |
| a cession's own result | `buy` |
| a tier subtotal | its span's shared role, or `net` if the span mixes |
| a running net, the grand result | `net` |

Numbers are unchanged: `net` and `sell` carry the same sign.

### Breaking

* `evaluate` no longer returns a `total impact` block. Code indexing that step
  raises `KeyError`.
* Rows that reported `role == 'sell'` for a running net or the grand result now
  report `'net'`. `Aggregate.evaluate` and `Portfolio.evaluate` still report
  `'sell'`, being obligations written rather than netted ledgers.
* `aggregate._pricing.no_distribution_panel` is gone.

## 1.0.0a189

**[Ledger-Side-Label-Levels]** The `PnL` sheets rename their index levels and
stop overloading `Total`. **Breaking for anything that names a level or a
margin row key.**

### The levels

`View` becomes **`Side`** and `Line` becomes **`Label`**, on `stats_df`
(`(Side, Label)` single-group, `(Step, Side, Label)` on a tower), on
`summary_df` (`Side` / `(Step, Side)`), and as `legs_df` columns
(`Step` / `Side` / `Label` / `kind` / `EX` / `SD`).

`Line` assumed a line of business, which the ledger does not otherwise assume,
and the level holds presentation labels: declared leg names, or `Total` on a
derived row. `Side` was already the internal word (`_pnl.py` passes
`side='cons'|'obl'`); this promotes it and widens it to cover `Margin`.

`Side` was chosen over `Leg` deliberately. `Leg` is the public class for an
individual declared cash flow, and those now sit under `Label`, so a level
named `Leg` would be the one level with no leg names in it.

### Direct and Net

`Total` was doing three jobs: a within-step subtotal, the direct result, and
the grand result. On a ledger that **buys** something they separate:

| row | before | after |
|---|---|---|
| a `sell` group's own result | `Total` | **`Direct`** |
| a cession's own result | `Total` | `Total` |
| the grand consideration / obligation / result | `Total` | **`Net`** |
| a running net, the impact, every subtotal | unchanged | unchanged |

The pair is gated on the ledger actually containing a `buy` group, because only
then is there something to be direct *of*. A plain single-group `pnl` is one
`sell` group whose legs are already net, and a ledger merging two sold books
has no net to take: both keep `Total` throughout, so calling either `Direct`
would be a lie. The per-step `net through <g>` row keeps its `Net` label
either way, being a running total rather than half of this contrast.

### Migration

* `df.rename(..., level='View')` becomes `level='Side'`; `xs(..., level='Line')`
  becomes `level='Label'`.
* `('All', 'Margin', 'Total')` becomes `('All', 'Margin', 'Net')`, and likewise
  for the grand consideration and obligation rows.
* `('Gross', 'Margin', 'Total')` becomes `('Gross', 'Margin', 'Direct')` for
  whatever the first, sold step is called.
* Code that filtered legs out with `Label not in ('Total', 'Net', 'Impact')`
  needs `'Direct'` in that set.
* A step's own result, whatever its label, is the first `Margin` row of its
  block in plan order.

`PnL._view_index` is now `PnL._side_index` and `_VIEW_DEFAULTS` is
`_SIDE_DEFAULTS`; both are private.

## 1.0.0a188

**[Counterparty-Margin-Evaluate]** `evaluate` reads a **bought** position from
the seller's side, so every ceded layer in a tower now prices instead of
reporting `NaN`. This completes `a187`: evaluating every margin row is only
useful if the reinsurance rows produce an answer.

### The problem

A cession's margin to the buyer is negative by construction, because you pay
for cover. The `E[M] > 0` guard therefore fired on every reinsurance row of
every tower. On a two-tier program (two occurrence layers, two aggregate
layers, peeled) six of the thirteen margin rows came back `NaN` reading
`E[M] <= 0`. Arithmetically right, analytically useless: the panel said nothing
at all about the cover.

The question worth asking about a purchased layer is what stress the
**seller's** position survives, and that is the buyer's margin negated.

### What changed

`PnL._row_role` classifies each margin row and `evaluate` passes the result
through to the solve:

* a **group result** takes its own group's `role`, so a cession is a `buy`;
* a **tier subtotal** and the **total impact** read `buy` only when *every*
  group they cover is a `buy`, so a mixed span stays as booked;
* a **running net** and the **grand result** are the holder's own net position
  and are always `sell`.

The flip is keyed off `Group.role`, not off a row label. On a builder-produced
ledger the two rules agree exactly; they diverge only on a hand-built ledger
with two `sell` groups, where a label rule would wrongly flip the second book.

The panel gains a **`role`** column (`sell` / `buy`), first in `EVAL_COLS`,
naming whose position the parameters describe. `Aggregate.evaluate` and
`Portfolio.evaluate` always report `sell`: an aggregate is an obligation
written. `aggregate._pricing.EVAL_SIGN` is the one-line map, and
`evaluate_margin` / `evaluate_constant_premium` / `no_distribution_panel` all
take `role=`, raising on anything else rather than defaulting.

### What this buys you

The panel becomes a **buy decision**. A layer whose `gini_p` sits above the
running net immediately over it is priced above the holder's own acceptability,
so buying it lowers the net; one below it raises the net. On the two-tier
program above (`ph`), the direct book reads 0.329, the two occurrence layers
0.586 and 0.535, and the running net falls 0.329 to 0.255 to 0.172 as they are
bought. The cover is dear and the sheet says so.

`DegenerateEvaluationWarning` becomes rare and meaningful: it now fires on a
layer priced below its own expected recovery, a real finding, rather than on
the routine fact that cover costs money.

### Breaking

* `evaluate` gains a leading `role` column. Code selecting columns positionally
  or asserting the exact column list needs updating; `EVAL_COLS` is the
  canonical order.
* A ceded-layer row that returned `NaN` on `a187` now returns solved
  parameters. Constant-premium and running-net answers are **unchanged**, which
  is the regression anchor.

## 1.0.0a187

**[Margin-Acceptability-Evaluate]** `evaluate` now solves `rho_g(margin) = 0`
rather than `rho_g(obligation) = E[consideration]`, applies to **every margin
row of a ledger**, and gains `Aggregate` and `Portfolio` faces. This is a
**correction, not an enhancement**: a variable-premium position evaluated on
`a186` returned a number that answered the wrong question.

### Why the old form was wrong

Distortion risk measures are translation-equivariant, so with a **constant**
premium `P` the statements `rho_g(P - L) = 0` and `rho_g(L) = P` are the same,
and pricing the obligation against the premium was legitimate. Once the
consideration is random (swing rating, slide, profit commission, reinstatement
premium, corridor) that equivalence fails: `rho_g` is comonotone-additive but
not additive, and the margin `M = P - L` is generally not monotone in `L`, so
`rho_g(P) - rho_g(L)` is not `rho_g(P - L)`. Only the margin's own pushforward
answers the question.

The symptom was visible: a loss-sensitive P&L with `E[P] < E[L]` used to
calibrate happily to a "premium" below its own expected loss. It now reports,
correctly, that the position is acceptable at no stress at all.

**Constant-premium answers are unchanged.** With `M = P - L` the canonical
shift is `c = P - min(L)` and `Z = L - min(L)`, so the target is `P - min(L)`
and the solve is `rho_g(L) = P`, exactly the classic form. This is the
regression anchor, pinned by
`test_evaluate_matches_the_constant_premium_price_form`.

### Every margin row, not just the grand result

`PnL.evaluate` evaluates each `group_result` / `running_net` / `tier_result` /
`grand_result` / `total_impact` row: the gross deal, each reinsurance layer as
a position in its own right, and the running net after each purchase. Reading
a `gini_p` column down the `net through ...` rows is watching the deal improve
as cover is bought. The **single-obligation-leg restriction is gone** (a margin
is one random variable however many legs feed it, so expense ledgers evaluate),
and so is the regular-grid restriction. A **stitched peel** now evaluates too;
its `total impact` row is the one exception, being a delta of two statistics
whose sides ride different marginals, and it reports `NaN` with that reason.

### Grid-agnostic quadrature

A margin pushforward generally lands on an **irregular** support, so
`Distortion.calibrate` / `calibrate_set` take `dx` where they took `bs`: a
scalar bucket size, or a per-node width vector. One new helper,
`Distortion._quad(v, dx)`, carries the split. The scalar branch keeps the exact
`np.sum(v) * dx` summation order, so the classic lattice pricing path
(`calibrate_distortions`) is **bit-for-bit unchanged**; the vector branch is
*exact* rather than an approximation, since the survival function of an atomic
law is a step function and the layer integral is a finite sum of rectangles.
Rebucketing a margin onto a lattice was rejected as the alternative: it is
mean-preserving, so it looks right while misreading the tail the index reads.

### The new faces

- `Aggregate.evaluate(P=None, names=None)` and
  `Portfolio.evaluate(P=None, unit='total', names=None)`. `P` defaults to
  `exp_premium` and **raises** when unset, since a premium of 0 would silently
  make every position unacceptable. `Portfolio` accepts a list of units, giving
  the book's acceptability profile in one frame.
- All three faces return the same tidy frame, so panels concatenate.

### The panel

Tidy (long) form: `MultiIndex` rows `(Step, distortion)`, columns
`param_name` / `param` / `gini_p` / `error` / `status`. `dev/reporting-guidelines.md`
rule 2 decides the orientation: a wide frame would put a PH exponent, a Wang
`lam`, a Dual `b` and a TVaR `p` under one `param` heading, four units in one
column. The wide comparison view is `.unstack('distortion')`. `area` is
dropped, being exactly `(gini_p + 1) / 2`.

A degenerate position reports `NaN`, never `0` or `inf`, with a new
`DegenerateEvaluationWarning` raised **once per call** naming the affected
steps. Two cases: `E[M] <= 0` (acceptable at no stress) and `M >= 0` almost
surely (an arbitrage, acceptable at every stress). The limiting parameter
differs by family and says nothing about the position, so a number there would
invite meaningless comparisons; the `status` column names which case fired. A
reinsurance program booked as its own step reads `E[M] <= 0` correctly, since
you pay for cover.

### `E_consideration` retired

Property and `info` row both. It was the same idea as the `Scaled` column
retired in `a185`: one committed ledger-wide premium number. Its only two
consumers in `src` were `evaluate` (which no longer targets a premium) and one
`info` row. The `a185` CHANGELOG and `dev/done/plan-pnl-ratio-frame.md` claim
it is "the `P` denominator `ratio_df` needs"; that was **wrong even then**.
`ratio_df` accumulates `P` per block from the `LEG_KINDS` buckets, and reading
`ratio_df['P']` is the replacement.

Plan: `dev/done/plan-margin-acceptability-evaluate.md`.

## 1.0.0a186

**[PnL-Ratio-Frame]** Two corrections to the ratio frame shipped in `a185`.

**Renamed** `EX_LR` / `EX_ER` / `EX_CR` to `E_LR` / `E_ER` / `E_CR`. `EX_LR`
read as "the `EX` column, of `LR`", which is not what it is; `E_LR` reads as
`E[LR]`, which is.

**The availability gate was wrong.** It asked whether a joint existed when the
question is whether the **denominator is random**. A constant premium factors
straight out of `E[X / P]`, so the mean of the ratio is exact on any route and
simply repeats the plain ratio. Reporting `nan` there withheld a number that was
never in doubt.

It bit hardest exactly where it was least warranted. A layer-peeled `xpnl` with
two or more occurrence layers takes the stitched route, and peeling is
guaranteed-cost by construction: `peel` is refused on the variable-rating and
reinstatement recipes, and every consideration row is a resolved
`deposit` / `rol` / `rate`. So a peeled ledger's premium is *always* constant,
and its `E_` columns were always blank when they were always exactly computable.

The gate is now three-way, on the denominator:

- constant premium: `E_x` is the plain ratio, exact, on every route;
- random premium with atoms: average over them, as before;
- random premium without atoms: `nan`, genuinely unknowable, never a silent
  fallback to the ratio of the means.

`nan` also still stands where an atom carrying probability has a vanishing
premium, the ratio being undefined there. The third case needs a route with no
shared atoms *and* a variable feature, which no current builder produces; it
would arise if `[Peel-Aggregate-Tier-Only]` let a variable-rated cession into a
peeled ledger. `_block_amounts` reports the constant-denominator test off the
per-row exact standard deviations, which every route carries, rather than off
atoms that may not exist.

The retro case is unchanged and still the reason both columns exist:
`LR = 0.2427` against `E_LR = 0.2256`.

## 1.0.0a185

**[PnL-Ratio-Frame]** **Breaking.** The `Scaled` column is gone, and with it
`scaled_stats_df`, the `scale` property and the `scale=` constructor argument.
Ratios now live in `ratio_df`, their own table, joined by an itemized `legs_df`.
Plan: `dev/done/plan-pnl-ratio-frame.md`.

`Scaled` divided every cell by one committed number, `E[grand total
consideration]`, which on a walk is gross premium *minus every cession*. On a
two-group tower (premium 100, loss 40, ceded premium 30, recovery 8) the gross
premium cell read **1.43** and the gross "loss ratio" read **-0.571** against a
true 0.40. The combined-ratio reading was an accident of the single-group case,
where the divisor happens to be that block's own premium; the original design
called the column *"just a unitless, comparable number"*, explicitly not an
accounting ratio. `[Reporting-Guidelines]` settles the direction: rule 1 permits
column changes, and rule 2 says ratios live in their own columns or tables.
`E_consideration` survives the cull, being the `P` denominator `ratio_df` needs
and the premium target `evaluate` calibrates to; the `info` block's `scale` row
becomes `E[consideration]`.

`Leg` gains an optional `kind`, one of the new `LEG_KINDS` (`'premium'`,
`'loss'`, `'expense'`, `'recovery'`, `'commission'`). Every builder now sets it,
so `_resolve_expense_split` no longer discards the one fact that separates
`'LAE'` from `'Loss'`. This is what makes an expense ratio possible at all:
nothing in `stats_df` distinguishes a loss leg from an expense leg but the label
text. `stats_df` gains no index level, so no existing exhibit changes shape. A
leg left unclassified (the dict shorthand, a hand-built ledger) folds into `L` as
the residual, so a ledger that declares no expense legs reports `E = 0` rather
than guessing.

`PnL.ratio_df` is indexed by `Step`, one row per block: each group, each tier
subtotal, and `All`. Columns are the amounts `P` / `L` / `E` / `C`, the block's
signed `M`, then `LR` / `ER` / `CR` and `EX_LR` / `EX_ER` / `EX_CR`, then
`P_share` / `M_share` against the gross block. `L`, `M`, `P` and `LR` keep the
`PENTAGON_STATS` spelling so a P&L ratio frame concatenates and diffs against a
pricing frame.

The amounts are signed **in the gross direction**: consideration as booked,
obligations negated. A cession's ceded premium and recovery are therefore both
negative, which buys three things at once. The amounts add across blocks, so
layers sum into their tier and tiers into `All`. `M == P - L - E - C` holds
identically, being the signed row sum, with `1 - CR == M / P` as its other
reading. And every ratio comes out with its conventional sign, because numerator
and denominator flip together: a cover that paid back three times its premium
reads `LR = 3.10`, not `-3.10`. Ratios are re-derived from each row's own
amounts, never averaged from the blocks below, following `pricing_df`.

**`LR` and `EX_LR` are the ratio of the means and the mean of the ratio, and
they are reported separately because they are different numbers.** When premium
is random and correlated with loss (a retro-rated account, a swing / slide /
profit-commission cession, reinstatement ceded premium `D + h(R)`),
`E[L/P] != E[L]/E[P]`. Measured on `pnl R retro basic 3000 lcm 1.1 min 3500 max
8000 premium less agg R_e 1000 loss sev lognorm 100 cv 2 poisson`: `LR = 0.2426`
against `EX_LR = 0.2256`, a 7% relative gap, and in the direction the retro
implies, since premium rises with loss and damps the per-atom ratio. With a
deterministic premium the two agree to the last bit.

`EX_*` needs the joint of loss and premium, so it exists only on a per-atom
route. It is `nan` on a route with no shared atoms (the stitched peel, the
massive sweep) and `nan` when some atom carrying probability has a vanishing
premium, rather than silently falling back on the ratio of the means.

`PnL.legs_df` is the itemized companion: one row per **declared** leg with its
`Step` / `View` / `Line` / `kind` / `EX` / `SD`. Derived rows are absent by
design, being sums of these. It is the only place `Leg.kind` surfaces, and the
frame to pivot when the ratio you want is not one `ratio_df` carries.

Both frames are **raw materials**, in the author's sense: unformatted, and
deliberately absent from `qd` and the notebook repr, which keep rendering
`summary_df`, the presentation-ready layer. Transpose for the stat-down-the-side
orientation, matching the documented `pricing_df.T` convention.

Follow-on logged as `[Ratio-Distribution]`: a per-atom route holds the joint, so
the loss ratio is available as a `GridDistribution`, which is what `E[phi(LR)]`
needs for a sliding commission. That is strictly more than either scalar column
gives.

## 1.0.0a184

**[Tier-Subtotal-Rows]** A layer-peeled `xpnl` now also shows the whole
occurrence and whole aggregate program. Peeling gave the per-layer detail but
lost the tier lines the plain walk carried, so a five-layer tower had no row for
the program as a whole. Plan: `dev/done/plan-tier-subtotals.md`.

A tier that peels into **two or more** steps gains its own three-row block after
the last step it spans:

```
Gross            Consideration | Obligation | Margin
occ 300 xs 200   ...
occ 100 xs 100   ...
All occurrence   Consideration | Obligation | Margin   <- subtotal
agg 150 xs 300   ...
All              Consideration | Obligation | Margin | Impact
```

A tier that peels into one step already *is* its own subtotal and gets nothing,
which is why the plain tier walk and a one-layer-per-tier peel are byte
identical to `a183`. There is no `Net` row on a tier block: the running net
through the tier is already the last layer's `Net`.

`_ledger_plan` takes a `tier_spans` argument, `(label, lo, hi)` triples naming
contiguous group spans with `hi` exclusive, and emits two new row kinds,
`'tier_total'` `(lo, hi, side)` and `'tier_result'` `(lo, hi)`. `Line` reads
`'Total'`, like the group and grand totals the block sits between, so exhibits
that filter `Line` on `'Total'` already treat it as the summary row it is.

The three evaluation routes cost little. On the **per-atom** route a subtotal is
the grand total restricted to a group span, so it is a partial sum over the same
atoms: the scenario `κ` ladder survives and every column still foots exactly. On
the **massive** route it is one extra sweep key, not a second sweep. On the
**stitched** route the kernel needs nothing, since that route is keyed by row
label, but the builder needs the tier's *whole* cession as its own marginal:
summing the per-layer recovery vectors would add densities rather than
variables, and the tier recovery is not an affine of any cumulative net already
in hand. That costs one further FFT for the occurrence tier (the cumulative
ceder from its last step) and no FFT at all for the aggregate tier, which is a
pushforward of the net-of-occurrence aggregate.

The subtotal is the cross-check the peel had been missing: on a two-layer
occurrence tower the peel's `All occurrence` block reaches the tier total by
summing its own per-layer rows, while the plain walk reaches it as one lumped
group off the occurrence joint. Two independent code paths, and they agree to
1.6e-7 (the walk is the looser, riding the coarser 2-D joint).

Three latent kernel problems fixed while in there. `_assemble_rows`,
`_init_massive` and `_view_index` each closed their row-kind dispatch with a
bare `else` that *meant* `'total_impact'`, so an unhandled kind was silently
booked as the impact row; all three now name `'total_impact'` explicitly and
raise on anything unknown. `PnL.__add__` reconstructed from `_group_specs` /
`_scale_arg` / `result_name` only, so any new constructor argument silently
dropped on composition; it now carries `tier_spans`, shifting the right-hand
operand's spans by the left-hand group count, since `+` renumbers the groups.

`summary_df` gains the matching tier blocks and `density_df` the tier result
rows. `tests/test_pnl_peel.py` grows to 53 cases; `decl-testers.agg` section AI
gains `AI.Both`, two layers in each tier.

## 1.0.0a183

**[Layer-Peeling-Shorthand]** New DecL clause `peel`: `xpnl ... peel top-down`
or `peel bottom-up` books one group per reinsurance **layer** instead of one per
tier, so every layer reports its own ceded premium, ceding commission and
marginal impact, and `net through <layer>` builds the program up a layer at a
time. Plan: `dev/done/plan-layer-peeling.md`; the placeholder was in
`dev/done/plan-yapnl.md`.

```
xpnl Book 1000 premium less
  agg Book_e 1000 premium at 70% lr
    sev lognorm 100 cv 2
    occurrence net of 100 xs 100 deposit 60 and 300 xs 200 deposit 40
    poisson
  peel top-down
```

walks `Gross`, `occ 300 xs 200`, `occ 100 xs 100`, `All`, where the tier walk
had a single lumped `ceded occ` step. Occurrence layers are peeled before
aggregate ones, preserving the tier walk's step order; within a tier `top-down`
introduces the highest-attaching layer first and `bottom-up` the lowest.
Zero-share gap fillers are structural, not cessions, and get no step. The clause
is a modifier on an existing statement rather than a new object kind, because
peeling is a way of building, not a new type ([Decision-XPnL-Is-A-Recipe]).

The two tiers are not symmetric, and that picks the route. **Aggregate layers**
are disjoint intervals on the one aggregate subject, so single-layer ceders sum
to the cumulative ceder identically and each layer is simply another per-atom
leg on the shared source: the scenario `κ` ladder survives and every column
foots. **Occurrence layers** are not, because the ceded-occurrence aggregate is
not a function of the gross aggregate (the random claim count decouples them),
which is why the tier walk reaches for the 2-D `occ_bivariate` in the first
place. Peeling `m` occurrence layers per-atom would need an `(m+1)`-axis joint,
which [One-2D-Source] forbids, and the total ceded axis cannot be split after
the fact because the sum of per-claim cessions does not determine the per-layer
allocation.

Two or more peeled occurrence layers therefore route through the kernel's
`stitched_rows` seam, kept at `a141` as the designated no-joint assembly seam
and now with a second consumer. The caveat is softer than it sounds: every row
of an occurrence peel is the compound of a **deterministic per-claim severity
transform**, so each row's marginal is exactly one FFT of a transformed
severity, the aggregate tier rides the net-of-occurrence aggregate as a plain
pushforward, and the `EX` column **foots exactly** by linearity. Only the
dispersion columns are marginal, so the ladder carries plain `P01…P99` headers
rather than `κ`, `evaluate` and `+` are unavailable, and loss-basis LAE degrades
to the deterministic `rate * E[gross loss]` (the same trade the consolidated
face makes, [Consolidated-LAE-Off-Source]). Cost is `2m` one-dimensional FFTs on
the existing grid, against the two `reins_density_df` already pays.

The route is selected by how many occurrence layers are actually peeled, so a
program with one layer per tier reproduces the tier walk byte for byte in either
direction. On the two-layer example above the peeled recoveries sum to the
engine's own `E[ceded occ]` to 1.25e-10, and the closing margin matches the
consolidated `pnl` to the digit; the lumped tier walk is the figure that differs
slightly, because it rides the coarser 2-D joint.

`Underwriter._resolve_reins_economics` now reports `pc_occ_by_layer` /
`c_occ_by_layer` and the `agg` pair beside the unchanged per-side totals: the
loop already computed them layer by layer and only the aggregation was lossy.
`peel` is rejected with a message naming the reason on a consolidated `pnl`
(no steps to peel), a `port.NAME` engine (no layer structure), and every
non-guaranteed-cost recipe: a plain engine has no reinsurance, a reinstated
program is a single occurrence layer, a variable-rating feature decorates a
single aggregate layer.

**[Walk-Step-Default-Labels]** landed with it, closing the item from
`dev/done/plan-pnl-faces-punchlist.md`. **Breaking** for anyone indexing walk
`stats_df` by step name. An undeclared cover step whose tier holds exactly
**one** layer is now named by that layer's DecL descriptor, so
`occurrence net of 500 xs 500` gives a step `'occ 500 xs 500'` where it used to
give `'ceded occ'`, and a partial share reads `'agg 85% so 1500 xs 7000'`. A
multi-layer tier keeps the generic `'ceded occ'` / `'ceded agg'`, because it has
no single descriptor and `peel` is how you see those layers separately. The
descriptor comes from the writer's own cession renderer, so it is exactly the
DecL the layer round-trips to, and descriptors cannot collide because the layer
validator rejects overlapping cessions.

Peeling forced the pick rather than merely benefiting from it: `_ledger_plan`
raises on duplicate ledger row labels, so one group per layer could not share
one generic name. A layer's declared `as` label still wins, read from the
`{layer_index: label}` map pooled into `label_map` since `a132`. Test churn from
the rename landed in the same sweep across `test_decl_labels`, `test_pnl`,
`test_pnl_ceded_premium`, `test_pnl_engine_source` and
`test_reinstatement_decl`.

New `tests/test_pnl_peel.py` (40 cases) pins the degenerate anchor, both routes,
column footing, agreement with the engine's exact marginals, the label rules,
gap-filler skipping, every rejection and the writer round-trip; seven programs
mirror into `decl-testers.agg` section AI. `src/aggregate/_pnl.py` needed no
structural change, `_ledger_plan` having been m-group generic already.

## 1.0.0a182

**[Ceder-Gap-Knot]** Bug fix. `make_ceder_netter` mis-ceded on any reinsurance
program with three or more layers and two or more genuine gaps, returning a
cession part way up the gap instead of holding flat across it.

The ceder is built by walking the layers and emitting knot points. A layer that
attaches above the running top of the program sits over a gap, so it needs an
explicit left-hand knot to hold the ceded amount flat from the previous layer's
top up to its own attachment. The running top was tracked by accumulation
(`base += (a + y)`) rather than assignment, so it ran ahead of the truth and the
`a > base` test that emits that knot silently failed from the second gap onward.
The ceder then interpolated straight across the gap. On
`[(1, 100, 0), (1, 100, 200), (1, 100, 400)]` a subject loss of 400 ceded 250
where the correct answer is 200, and the missing knot at `(400, 200)` is visible
in the `debug=True` knot list.

Only programs with two or more gaps were affected. A gap written the documented
way, as a zero-share filler layer `0 po L xs A`, keeps the attachments
contiguous and so never triggered it, which is why this survived. Contiguous
towers, single gaps, unlimited top layers and the docstring's own worked example
are all unchanged, and the accumulating value's second use, the
`base < INF` test that decides whether to close the ceder flat at infinity, is
if anything more correct now that `base` really is the top of the last layer.

`tests/test_reins_buckets.py` gains a ceder-knot section: the two-gap
regression, a parametrized check of the interpolated ceder against the
closed-form sum of each layer's own payout across five layer shapes, and a pin
on the docstring's knot table.

## 1.0.0a181

**[OEP-Curve]** New utility `aggregate.oep(agg, p, *, freq=0)`, the occurrence
exceeding probability curve, and the exact severity inverse it needed.

`Aggregate.sev` gains `ppf` and `isf`. It already carried the exact continuous
`cdf` / `sf` / `pdf` of the weighted severity mixture, the documented look
through past the discretization, but it had no inverse, so there was no way to
get an exact severity quantile out of an `Aggregate` at all. `q_sev` reads the
bucketed `sev_density_df` and can only return a lattice point: on
`agg T1 2 claims sev lognorm 1000 cv 1.31 poisson` at `bs = 8`,
`sev.ppf(0.99) = 6207.853` against `q_sev(0.99) = 6208`. A single component
delegates to the `Severity` `rv_continuous` methods, which respect limits,
attachments and splices. A mixture has no closed-form inverse and is solved by
`_mixture_inverse`, which brackets the root with the component quantiles, a
bracket the monotonicity of the components guarantees, then calls
`scipy.optimize.brentq`. The `SevFunctions` namedtuple is now defined at module
level alongside `RuinFunction` instead of being rebuilt inside the property.

`oep(agg, p)` answers "there is a probability `p` that one or more occurrences
in a year exceed `x`, what is `x`?" It inverts
`p = 1 - exp(-lam * Pr(L > x))`, returning a DataFrame indexed by `p` with the
loss, the severity probabilities `S_sev` / `F_sev` of that loss, the achieved
`oep`, and both return periods: the occurrence one `1 / (lam * S_sev)`, which
can be shorter than a year, and the annual one `1 / oep`, which cannot. It
reproduces Table 8 of *Return Period Confusions Clarified* and the
`q(1 + log(1 - 1/n) / lam)` formula in the catastrophe modeling user guide, and
it agrees bit for bit with `scipy.stats.lognorm(1, scale=1000).isf(...)` on the
post's example.

Three points worth knowing:

* The loss is computed as `isf(-log1p(-p) / lam)`, not `ppf(1 + log1p(-p) /
  lam)`. The second form cancels toward 1 as `lam` grows and the two already
  differ by 6e-12 relative at `lam = 100`, `p = 1e-4`.
* There is a ceiling, `p < 1 - exp(-lam)`, which is 0.8646647 at `lam = 2`. A
  year with no occurrence has no largest loss, so above it nothing is exceeded.
  Raises rather than returning `nan`.
* Poisson frequency is required, and zero-modified Poisson is refused: the
  thinning argument is what makes `1 - exp(-lam S)` hold. Passing `freq` only
  rescales `lam`, it does not waive the requirement.

`S_sev` is read back from the loss rather than reused from the target, so where
the severity has an atom the achieved `oep` visibly falls below the requested
`p` instead of the table silently repeating the same loss. On a limited
severity the quantile pins to the limit, `S_sev` is 0, and both return periods
are infinite.

Also in this release: the stale `:meth:`sev_q`` / `:meth:`sev_tvar``
cross-reference in `_sev_grid_distribution` corrected to `q_sev` / `tvar_sev`,
and `balanced_window` added to the utilities autosummary, which had never
listed it. Docs are edited but **not rebuilt**; a rebuild is pending.

## 1.0.0a180

**[Help-Default-Regex]** `regex` now defaults to `'.*'` on `HelpMixin.help` and
on `utilities.agg_help`, so a bare `a.help()` lists the whole surface instead of
raising `TypeError`. Two filters decide what appears, and only two: `regex`, and
the leading-underscore skip governed by `private` (default `False`). Inherited
names are not filtered, so `Severity.help()` reports the
`scipy.stats.rv_continuous` methods alongside its own; that is documented in the
`agg_help` Notes.

## 1.0.0a179

**[TVaR-Endpoint-Noise]** `Bounds(port, premium).weight_df` emitted
`RuntimeWarning: invalid value encountered in multiply`. The numbers were
always right; only the reporting was noisy. Three sites now carry the
`np.errstate(divide='ignore', invalid='ignore')` guard the rest of the library
already uses.

`Bounds.p_knots` deliberately includes `p = 1`. The quantile kernel
`make_var_tvar` pads its tail arrays with `inf` sentinels and selects with
`np.where`, which evaluates *both* branches, so at the endpoint the discarded
branch computes `0 * inf` and a `1 / 0`. The comment in that function already
noted `np.where` does not short circuit; the guard was simply missing.

This was never specific to `Bounds`. `make_var_tvar` is the kernel every
`q` / `var` / `tvar` call routes through, so a bare
`port.tvar(np.array([0.5, 1.0]))` warned too. The scalar branch was always
clean because it short circuits before the padded arrays.

Guarded:

- `_grid_distribution.py`, the vectorized branch of `make_var_tvar`'s `tvar`.
- `_grid_distribution.py`, `GridDistribution.tvar_of_limited`, whose false
  branch divides by `1 - p` even where the answer taken is the cap `a`.
- `spectral.py`, `Distortion.tvar_terms`, a nested `np.where` over
  `min(s / (1 - p), 1)`. This is what fired on `Bounds.cloud_df` and
  `min_envelope`.

New regression cases in `tests/test_grid_distribution.py` and
`tests/test_bounds.py` assert both the silence and the values, so a future
refactor that lets the NaN reach the result also fails. `tests/test_bounds.py`
had its own copy of the knot function dividing by `1 - p` at `p = 1`; it now
calls a small `_tvar_g` helper that writes the endpoint out.

### Scope

This does not make the library warning free. Measured with
`pytest -W error::RuntimeWarning`: 71 failed and 4 errors before, 52 failed and
0 errors after. The remainder are other benign boundary sites, mostly elsewhere
in `spectral.py` plus `moments.py`, `pedagogy.py`, and `_aggregate_compute.py`.
They need case by case review and are tracked as **[RuntimeWarning-Census]** in
`dev/TODO.md`. Nothing in the package promotes warnings to errors, so this
matters only to users whose own kernel does.

## 1.0.0a178

**[Library-Canonical-Layout]** `agg/library.agg` is rewritten in the canonical
`spread` layout: one clause per line at a two-space indent, `;` closing each
statement, a blank line between them. Median statement length was 153
characters and 40 statements ran past 200; nothing but a note body is now over
100. It is the layout `build.recipe('X').decl` prints and every generated
cookbook page already showed, so the source and the docs finally read alike.

```
agg ExposureRating
  [100 200 1000 50] premium at [0.9 0.85 0.9 0.8] lr
  [250 500 1000 2000] xs 0
  sev lognorm 120 cv 12
  occurrence ceded to
    750 xs 750
  mixed gamma 0.2
  note{Exposure rating on a small portfolio with limits profile, amounts in 000s.}
  tags{role:hero, topic:aggregate};
```

### How

`dev/reflow_library.py` renders each entry through
`format_program(..., layout='spread')` and refuses to write unless every
statement in the file still parses to the same `(kind, name, spec)`. Whitespace
between tokens is insignificant to the lexer, so that check is complete: where
the breaks fall is taste, whether the meaning moved is not. 169 entries are
canonical; 16 are held back and hand-written in the same style, because the
writer cannot render them back to what was written.

Comments pass through verbatim, so the file header and every section divider
are untouched.

### Two defects the layout exposed

**Four PIR case studies were building at the wrong resolution.** `PIRTame`,
`PIRDiscrete`, `PIRCatNonCatGross` and `PIRCatNonCatNet` each carried
`hints{bs=1/64; log2=16; padding=1}` after their last unit, where a `port`
trailer does not reach: `port_out` places the trailer *before* `agg_list`, so
the portfolio's slot has closed by the time a unit is read. The hints bound to
the unit and were ignored. `PIRDiscrete` built at `log2=11, bs=1` against the
`log2=8, bs=1` it asked for; `PIRTame` at `bs=1/256` against `bs=1/64`. Moved
to the port header line, all four now build as specified. Invisible on one long
line; obvious once the clause is indented under the unit it was attached to.

**`ssev <c> - <dist>` renders as an ambiguous form.** The writer emits the
general affine spelling `ssev -1 * <dist> + <c>`, whose leading `-1 *` parses
two ways (`scaled(-1, X)` or `negate(scaled(1, X))`, algebraically identical).
Three entries are held back rather than grow the shipped library's ambiguous
set for a cosmetic gain, and the reflow script now refuses any rendering more
ambiguous than its source.

### Also

- `decl_writer` renders a unit claim count as `1 claim`, not `1 claims`.
  46 single-claim severity entries read as English again.
- `agg/decl-testers.agg`: 17 `port` entries had a `note{}` describing the
  *portfolio* bound to their last unit. Moved to the header line.
  `AE.Trailer.Port` and `FCC.NoteP` keep theirs, being the fixtures for the
  positional rule and for per-unit notes.
- `tests/test_agg_libraries.py` gains two checks: every non-exempt entry is
  byte-identical to what `format_program` renders (so regeneration is a no-op,
  not a diff), and no library `port` binds a trailer to a unit.

### Known unparser gaps

Recorded as `[Unparser-Reference-Gaps]` in `dev/TODO.md`; the exempt list lives
in `tests/test_agg_libraries.py`.

| gap | entries | what the writer emits |
|---|---|---|
| named object reference | 9 | `sev.UnitSeverity` inlines to `dsev [1]`; the spec keeps no record a reference was written |
| `tweedie` clause | 3 | expands to the equivalent compound-Poisson-gamma, and the transformer has already replaced the author's `note{}` with a conversion string |
| distortion combinator | 1 | `minimum dist.A dist.B` raises: the child names are not retained |
| `ssev <c> - <dist>` | 3 | the general affine form, which parses two ways |

## 1.0.0a177

**[Trailer-Body-Inert]** A `note{...}` is prose, but the preprocessor read it as
DecL. Three characters were claimed that a human writing a note has every right
to use, and all three are now free.

| written | stored before | stored now |
|---|---|---|
| `note{E[loss]=85, margin 15}` | `E [loss] =85, margin 15` | `E[loss]=85, margin 15` |
| `note{layer is 5# of limit}` | parse error, several clauses upstream | `layer is 5# of limit` |
| `note{net // ceded}` | parse error, several clauses upstream | `net // ceded` |

`#` and `//` are the comment openers stripped by step 2, so the statement was
cut at the note and the parser reported an unexpected token well before it.
`[` and `]` drive step 3's vector collapse, which pads them with spaces. The
padding was **cumulative**: regenerating a file from its own specs added
another space every pass.

Twenty-four shipped notes were affected, 8 in `agg/library.agg` and 16 in
`agg/decl-testers.agg`. They now read exactly as written.

### The fix

`UnderwritingLexer.preprocess` gains step 0b, the doc clause's treatment one
size down. Each single-line `note{}` / `tags{}` / `hints{}` body is lifted
behind an indexed placeholder before the comment and bracket steps and restored
verbatim in step 7, once the text has been split into statements. An index
rather than base64, because the substitution is undone inside `preprocess`: a
real note body could imitate a base64 payload, but nothing can imitate a
placeholder written by the same call that reads it. A body spelled across a
line break is deliberately not lifted, so it keeps the behavior it had.

`decl_writer._split_statements` was a near-copy of `preprocess`, forked to omit
the bracket step for exactly this reason. It treated the symptom and left the
`#` / `//` half in place. With the cause fixed, the copy is deleted and the
function delegates, so writer and reader agree on what a statement is by
construction.

### What a note still cannot hold

A `}` (the terminal ends at the first one) and a line break. That is what
`doc{{{...}}}` is for.

New fixtures: section AG of `agg/decl-testers.agg` and the free-text cases in
`tests/test_doc_clause.py`, including an idempotence check that pins the
regeneration drift.

## 1.0.0a176

**[Interpret-File-Whole-Text]** `Underwriter.interpret_file` reported parse
errors that were not there: 67 on `agg/library.agg`, 13 on `agg/decl-testers.agg`.
Both counts are now zero, and neither file had anything wrong with it.

### The bug

`interpret_file` split the raw file text on newlines and parsed each **physical
line**. That model predates multi-line statements and the `doc{{{...}}}`
trailer. A doc body is markdown, so every one of its lines became a bogus
"statement" and every one of those failed to parse. The production path
(`Underwriter.load` to `UnderwritingLexer.preprocess`) has always split on
statements instead, which is why the same files load without complaint.

The `.agg` branch now calls `self.lexer.preprocess(txt)` on the whole file, the
identical split `_read_file` uses: statements separated by a blank line or a
line-final `;`, comments transparent, doc bodies lifted out first.

### Surface changes

- The `preprocessed program` and `program` columns collapse into one `program`
  column holding the statement. The old pair compared a preprocessed line
  against its raw source, a distinction that no longer exists when the unit of
  work is the statement.
- Rows are collected positionally rather than in a dict keyed on the entry name,
  so two entries sharing a name (`decl-testers.agg` reuses names across kinds)
  no longer silently overwrite each other.
- `.csv` files are unchanged: a cell is free-form, so it is still preprocessed
  in place and may still report `multiline` or `blank`.

Docs are pending a rebuild: the per-line wording in
`docs/new_material/underwriter.rst`, `docs/2_user_guides/2_x_10mins.rst` and
`docs/2_user_guides/dm-claude.qmd` was updated in place.

## 1.0.0a175

**[DecL-Colorizer-Resync]** The Pygments colorer was a hand-written mirror of
`decl.lark` that had not kept pace with the language. It is now re-derived from
the grammar, and the drift guard has been widened so the next addition cannot
slip past.

### The visible symptom: red boxes in Jupyter

`AggLexer` had no rule for the quoted display label that
`[DecL-Labels-Everywhere]` introduced, so `as "My Book"` lexed both quotes as
`Token.Error`. The `friendly` style, which `decl_writer._colorize` hard-codes
for all three markup formats, draws `Token.Error` with a red border. Every
labeled object's `pprogram_html` therefore showed a red box around each quote.

`Token.Error` counts over the shipped corpora, before and after:

| corpus | before | after |
|---|---|---|
| `agg/library.agg` | 170 | 0 |
| `agg/decl-testers.agg` | 77 | 0 |
| `agg/_test_suite.agg` | 0 | 0 |
| `agg/_test_suite2.agg` | 0 | 0 |

### What the colorer had missed

Thirteen defects, each reproduced against real DecL before being fixed.

- **`tags{...}`** and **`doc{{{...}}}`** had no rules at all. The trailer word
  fell through to a bare identifier and the body was lexed as DecL, which is
  where most of `library.agg`'s errors came from: a markdown doc body is full of
  backticks and quotes.
- **`//` comments** were lexed as two division operators. Only `#` was handled,
  though `UnderwritingLexer.preprocess` has always stripped both.
- **`@`**, the inhomogeneous-multiply operator, was an error.
- **Underscore group separators** were not accepted: `1_000_000` came out as
  three numbers and two errors.
- **`port.X`, `dist.X` and `distortion.X`** were split into three tokens. Only
  `agg.` and `sev.` were recognized as builtin references.
- **`mixed` at the end of a line** was uncolored. The rule was the literal
  string `'mixed '`, trailing space required, so the legal line break between
  `mixed` and its mixing distribution broke it.
- **`sichel.gamma` and `sichel.ig`** were not single tokens.
- **`dfreq`** sat with the frequency *distribution* names rather than with
  `dsev` / `dbvsev` / `dwait`, which is what it actually is. Conversely
  **`dhistogram` and `chistogram`** sat with the declaration keywords although
  they are severity distribution names reached through `sev dhistogram xps ...`.
- **`Name.Type`**, used for the `note{` / `hints{` / `wts` markers, has no entry
  in the `friendly` style, so those markers rendered as plain black text.

The root cause of a whole class of these is one line: every keyword rule used
`suffix=r'\b'`, and `\b` is not DecL's word boundary. The grammar makes `.`,
`_`, `:`, `~` and `-` name characters, so `\b` peeled keywords off the front of
longer names: `loss-ratio` came out as `loss`, `-`, `ratio`, and `dist.A` as
`dist`, `.`, `A`. Every rule now uses the grammar's own lookahead,
`(?![a-zA-Z0-9._:~\-])`.

### What is now colored

`tags{}` slugs read as `Name.Tag`; `hints{}` keys as `Name.Attribute` with
their values as numbers, constants or names; `note{}` stays prose. A
`doc{{{...}}}` body is handed to Pygments' Markdown lexer, so headings, inline
code and fenced code blocks highlight properly. That lexer costs roughly 110 ms
to import, so it is imported inside the fence callback rather than at module
scope, and a test pins that `import aggregate` does not pull it in.

Distortion kinds, copula kinds and approximation kinds are deliberately left as
plain identifiers: they are ordinary `ID` to the grammar, and coloring them
would mean a third hand-maintained vocabulary to keep in sync with
`spectral.py` and `copula.py`.

### Dead Python-lexer inheritance removed

The module began life as a copy of the Python lexer, and said so. It carried
`!=`, `==`, `<<`, `>>`, `:=`, the `in` / `is` / `or` / `not` operator words, and
backslash line-continuation rules. DecL has none of these. The continuation
rules were actively harmful: `\`-continuation was removed from the language, so
a stray backslash is now a real lexer error, and those rules hid it.

There is deliberately **no catch-all rule**. `Token.Error` is the drift signal,
and swallowing it would make the corpus test below vacuous.

### The guard

`tests/test_grammar_sync.py` walked the 88 reserved words in the grammar's `ID`
exclusion list. That walk structurally could not see any of the defects above:
`note`, `tags`, `hints` and `doc` are absent from the list on purpose, since
their terminals include the opening brace, and neither `STRING` nor the
operator terminals carry the priority tag the test greps for. It also probed
each word with a trailing space, which is exactly what hid the `mixed` defect.

The suite now also tokenizes all four shipped corpora asserting zero
`Token.Error`, checks the token values concatenate back to the source byte for
byte, derives the brace-clause words and the operator literals from the grammar
so a fifth trailer clause fails the suite, probes each reserved word against
five different following characters, and pins both the lazy Markdown import and
the stray backslash staying an error. 169 cases became 570.

`agg.sublime-syntax`, the second hand-written mirror, had drifted further: 33
missing reserved words and 2 stale ones (`multivariate` / `mv`, gone from the
grammar). It is resynced, gained the `tags` / `doc` / `//` / `port.` handling,
and is now covered by the same guard.

Docs are not rebuilt here. `docs/4_dec_Language_Reference.rst` reaches the lexer
through the `pygments.lexers` entry point, so its `literalinclude
:language: agg` picks this up on the next build with no edit.

## 1.0.0a174

**[Display-Surface-Incidentals]** The grouped small defects, plus the change that
made the group findable in the first place.

### The matrix can finally see the display surface

`dev/regen_features.py` dropped every `_`-prefixed name from its introspection,
which made `dev/FEATURES.csv` **structurally blind** to `__repr__`,
`_repr_html_`, `_text_info_blob` and `_html_info_blob`: the most duplicated part
of the library, and the part this item is about auditing. An explicit
`DISPLAY_MEMBERS` allowlist now lets those through. Deliberately not a blanket
widening, which would bury the UNDOCUMENTED report under internals and make it
useless.

It paid immediately. `Frequency` had **no `__repr__`** and printed
`<aggregate._frequency.FrequencyPoisson object at 0x...>`. It now reports its
family, its shape parameters and `E[N]`, omitting the mean when no owning
`Aggregate` has stamped one rather than inventing a number:

```
Frequency(poisson, E[N]=10)
Frequency(gamma, a=0.25, E[N]=10)
Frequency(poisson, zm p0=0.5, E[N]=2.03731)
Frequency(poisson)
```

The four display members are now a `display` group in the matrix, with the
remaining gaps written down rather than rediscovered: `_repr_html_` is on the
five first-class classes only, so `Severity`, `Frequency`, `GridDistribution`
and the three `Bounds` classes still render as plain text in a notebook; and
`_html_info_blob` exists on `Aggregate` alone, while four other classes build
the same intro paragraph inline. Neither is fixed here. The point of the row is
that closing them becomes a decision rather than a discovery.

### `tweedie.Tweedie._repr_html_`

Was spelled `__repr_html__`, which is not a dunder IPython looks for, so the
method was unreachable. No visible symptom, because `_repr_mimebundle_` serves
the same HTML and takes precedence in a notebook: dead code, not a broken
display. Both are kept and both render `to_frame()`, so they cannot disagree.

### `Copula` gains `HelpMixin`

It was the one class carrying `LabeledMixin` without `HelpMixin`, so `.help()`
reached every other labeled object and not this one. Neither mixin defines an
`__init__`, so this is a base-list change only.

### Closed with no change

`Underwriter.__repr__` hand-rolls its label/value layout at a hardcoded 19
columns instead of `info_row` / `INFO_LABEL_WIDTH`. Left alone by author
decision: `Underwriter` is not a first-class citizen, has no `info`, and is not a
column in the matrix, so the shared layout does not apply to it.

## 1.0.0a173

**[Discover-Kwarg-Guard]** `build.discover(tag='role:hero')` returned every one
of the 186 entries in the recipe base. A filter that silently matches everything
is the worst possible answer: it looks like a successful query, and it denies
that the tag vocabulary exists.

The tag filter was never wrong. The parameter is `tags`, plural, and `discover`
accepts `**kwargs` so it can forward build options to `build()`. The singular
spelling therefore bound into `kwargs`, and the lightweight directory path
returns without ever reading them. Nothing was filtered because nothing was
asked.

The same typo did fail on the `plot` / `describe` path, where `kwargs` reaches
`build()` and eventually `update()`, which is what made it so easy to miss: the
call that builds blows up, the call that lists does not.

### Two rules, both checked before any work

A kwarg whose name is close to one of `discover`'s own parameters is a mistyped
filter, and is rejected on either path with the house `Did you mean:` hint,
`get_close_matches` at the same `n=3, cutoff=0.6` the DecL parser uses. That
covers `kinds=` and `describ=` as well as the `tag=` that prompted this.

Anything left over is a build option, which is forwarded only when `plot`,
`describe`, or `return_objects` asks for a build. Passed on the directory path
it would be dropped, so it raises instead. `discover(log2=16)` was previously a
silent no-op returning the whole base; `discover('Dice', log2=16, describe=True)`
still forwards exactly as before.

```
>>> build.discover(tag='role:hero')
TypeError: discover() got an unexpected keyword argument 'tag'. Did you mean: tags?
>>> build.discover(log2=16)
TypeError: discover() ignores log2: build options are only forwarded when plot,
describe, or return_objects asks for a build.
```

### An empty filter intersection no longer raises

Separately, `discover('NoSuchName', tags='role:hero')` raised
`KeyError: "['program'] not in index"`. When `regex` or `kind` had already left
zero rows, the tag mask was an empty list, and pandas does not read one as a
boolean mask: `is_bool_indexer` guards its list branch on `len > 0`, so `df[[]]`
selected zero *columns*, and the `program` lookup downstream failed. The mask is
now an explicitly bool-dtype `Series`, which indexes correctly when empty. Subset
semantics are unchanged.

Three regression tests in `tests/test_underwriter.py` pin all of it: the singular
spelling failing identically on both paths, build options rejected when unused
and still forwarded when used, and the empty intersection returning an empty
frame that keeps its `program` column.

## 1.0.0a172

**[FCC-Contract-Gaps]** The four holes the declared contract exposed at `a170`,
closed. `FCC_CONTRACT_EXCEPTIONS` and `FCC_UNPAIRED_NARRATIVES` are now empty,
which is what finishes the item: the contract is satisfied, not excused.

### `Portfolio` keeps its own DecL trailer

The parser has always merged the whole trailer into a `port` spec, and
`_resolve_hints` has always applied it to the build. The underwriter passed only
`label` and `note` to the constructor, so `tags`, `doc` and `hints` were dropped:
a portfolio's own trailer was write-only. All three are wired now. Retention
only, so nothing about how a portfolio builds changes.

Worth knowing, because it is easy to get wrong: the port trailer goes **right
after the name** (`port_out: PORT name as_label trailer agg_list`). Written after
the last unit it binds to *that unit*, which is what it means, not a portfolio
note in the wrong place.

### `validation_df` on `BivariateAggregate` and `Distortion`

Both are **check tables**, `Est | Ref | Err | Gate | Pass`, carrying only what
can fail. That is the point: a reader who wants the verdict should not have to
know which of twenty numbers in `summary_df` is load-bearing.

The bivariate's rows are one per axis (each marginal must reproduce its
standalone aggregate) plus the tail deficit. It is now the **one** computation
behind the `validation` row of `info`, `validation_description` and
`validation_explanation`, which previously each recomputed it.

The distortion's rows are the four structural identities, and they need **two
tolerance regimes**. `g(0) = 0` and `g(1) = 1` are evaluated directly, must hold
exactly, and are gated at the config noise floor. `E[D_g] + E[D_g_inv] = 1` and
`g(g_inv(0.5)) = 0.5` come off a trapezoidal integral on 101 points, so they
carry a genuine `O(h²)` discretization term; gating those at the noise floor
would fail every kind for the crime of being a numerical integral. They are gated
at `10 h²`, which tightens automatically if the grid is refined. Measured across
the canonical five plus `bitvar`, the worst realized error is 1.9e-4 against a
1e-3 gate, and a real break is `O(1)`.

The moment rows of `Distortion.stats_df` are deliberately **not** validated. For
an atomic kind (`tvar`, `bitvar`) the grid-vs-closed-form error is legitimately
large, because a trapezoid cannot see a Dirac atom. That is a property of the
grid, so it stays a reported number rather than a pass/fail. `summary_df` keeps
its rows exactly as they were.

### The narrative pairs are complete

**Breaking, in a narrow way.** `validation_explanation` was never a long form: it
returned `'not unreasonable'` or `'fails agg cv'`. That terse text now lives on
`validation_description`, the name that describes it, and every terse consumer
(the `info` row, the one-line intro, the `Underwriter` `valid` column) reads it,
so **what they print is unchanged**. Code that read `validation_explanation` for
a short phrase gets a paragraph instead and should move to
`validation_description`.

`validation_explanation` became the long form it always claimed to be. It names
what was checked (the analytic first three moments of severity and aggregate
against the realized grid, at the object's own `validation_eps`), why only the
lowest-order failure is reported, and what to do about aliasing or the
reinsurance caveat when they apply.

`reins_explanation` is new, on `Aggregate` and `Portfolio`. `reins_description`
says what the program *declares*; this adds what the cession *does*, off
`reins_summary_df`: expected loss gross, ceded as a share, and net. Reported per
**stage**, because the aggregate cover attaches to the occurrence net rather than
the gross, and one "ceded" number across both stages double counts. It answers
`'No reinsurance.'` on a clean book, so no caller has to check `reins_kinds`
first.

## 1.0.0a171

**[FCC-Surface-Decisions]** The five open `[FCC-Surface-Sweep]` decisions,
executed, plus `[PnL-Repr-HTML]`.

### `BivariateAggregate.tail_df` is `axis_support_df`

**Breaking.** It was a collision, not an analogy. `Aggregate.tail_df` and
`Portfolio.tail_df` are return-period ladders (p, VaR, TVaR, xsVaR by exceedance
probability). The bivariate frame under the same name reported something else
entirely: where the realized mass sits on each axis. Two different reports
answering to one name is how a reader gets the wrong one.

`tail_description` and `tail_explanation` keep their names, because what they
describe really is the tail.

### `BivariateAggregate` joins the label surface

**Breaking only in the sense that a class gained members.** It was the last class
carrying neither half of the label surface, so `label`, `label_map`, `labels`,
`renamer` and `use_labels` now work on a bivariate, and the renamer resolves each
axis to its component `Aggregate`'s label:

```python
b = build('bivariate Cat 25 claims '
          'agg Wind as "Windstorm" dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2 '
          'agg Flood as "Flooding" dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5 '
          'copula gumbel 0.4 poisson')
b.axis_support_df.index          # ['Windstorm', 'Flooding']
b.use_labels = False
b.axis_support_df.index          # ['Wind', 'Flood']
```

Applied at serve time in `axis_support_df`, `bs_window_df` and `stats_df`, which
are the frames keyed by axis. The component labels come through DecL because each
unit is an ordinary `agg`; an **object-level** label has no DecL spelling yet
(the `bv_out` productions carry no `as_label`), so it is set with `label=`. That
grammar gap is logged as `[Bivariate-DecL-Label]`.

### `GridDistribution.info`

Not contractual: a `GridDistribution` cannot be declared in DecL, so it is not a
first-class citizen. It is here because the grid a quantile came off is exactly
what you want to see when a number looks wrong, and the rows are the grid rather
than the risk:

```
grid object name         Simple
value type               loss
buckets                  65,536
support                  0 to 131,070
bs                       2
total mass               1
E[X]                     10,000
```

Realized total mass is the row that earns its place. It is `1` for a complete
distribution and less when the holder handed over a clipped or conditional slice,
and every accessor works off the realized cumulative either way.

### `PnL.result` is the one name

**Breaking.** `PnL.gd` is retired. `result` was already the documented name and
what the tests used. `gd` stays the *internal* vocabulary on ledger rows, where a
leg's or a group's `.gd` reads correctly; a first-class object's own distribution
reads as its `result`.

### `var_dict` stays `var_dict`

No code. `q_dict` was considered under the VaR-is-`q` rule and **rejected**: too
cryptic. The rule stops at the scalar accessors, and this is recorded so it is not
re-proposed cold.

### `PnL` and `Distortion` render in Jupyter

`PnL` was thought to be the only first-class class without a `_repr_html_`.
Making the check executable rather than eyeballing it found a second, `Distortion`,
which had `__repr__` only. Both now render the same two pieces as everywhere else,
an intro paragraph and the headline frame.

The P&L intro says out loud that its percentile columns are marginal and do not
foot, with `stats_df` named as the footing sheet. That was already true of the
fixed card settled at `a134`; leaving the reader to discover it from a column that
does not add up was the wrong place to learn it.

## 1.0.0a170

**[FCC-Contract]** "First-class citizen" stops being a phrase in a plan and
becomes a declaration the build checks.

### The contract

`aggregate.constants` gains four names. They are names only, no class imports,
because that module is the import-graph leaf and both the dev audit and the test
suite have to read the one copy:

```python
FIRST_CLASS_CLASSES = ('Aggregate', 'Portfolio', 'BivariateAggregate',
                       'PnL', 'Distortion')
NEAR_FIRST_CLASS    = ('Severity',)
FCC_REQUIRED = ('info', 'help', 'note', 'hints', 'tags', 'doc',
                'program', 'pprogram',
                'summary_df', 'validation_df', 'stats_df', 'density_df',
                'plot')
```

Two criteria put a class on the list, and both must hold: it **can be created in
DecL** (which is why `Bounds`, `Frequency` and `GridDistribution` are out: they
are reached from an object, never declared), and it **flows through to the
`aggregate_api` SPA**, which calls exactly those members on whatever it is
handed. `Severity` is DecL-creatable and near-first-class, but it is a
look-through onto a frozen scipy random variable rather than a compute result, so
it is exempt from the DataFrame quartet and listed separately rather than
silently omitted.

Everything outside `FCC_REQUIRED` is optional and callers reach it with
`getattr`. The `*_description` (short) / `*_explanation` (long) narratives are the
main such family, and optional does not mean unconstrained: wherever one half is
present, the other must be too.

### Two checks, one source

`dev/regen_features.py` grows an `## FCC CONTRACT` section and a
`## NARRATIVE PAIRS` section, both counting toward its exit status.
`tests/test_fcc_surface.py` runs the same two checks on live objects, so the
contract cannot break without a red test even when nobody runs the dev script.
Membership is read from `dir`, never `hasattr`: `hasattr` swallows exceptions and
would report a raising property as a missing member.

### What the check found

Four holes nobody was tracking, now carried in the declaration itself:

| Class | Missing |
|---|---|
| `Portfolio` | `hints` (the parser produces it, the class never stores it, the same shape as the `note` gap fixed at `a154`) |
| `BivariateAggregate` | `validation_df` (its validation lives inside `summary_df`) |
| `Distortion` | `validation_df` |

plus two unpaired narrative stems: `validation_explanation` has no short form and
`reins_description` no long one. They are declared as
`FCC_CONTRACT_EXCEPTIONS` and `FCC_UNPAIRED_NARRATIVES`, and both **must be empty
by `1.0.0b1`**. An excused member that is later added fails the check too, so an
excuse cannot outlive its hole.

### Also

`ZMPoissonSimple` and `ZTPoissonSimple` in `library.agg` had their programs
crossed: the entry named `ZM` declared `poisson zt` and the entry named `ZT`
declared `poisson zm .5`. The notes matched the programs, so only the two names
were wrong. Swapped.

The first-class-citizen definition is written up as §0 of
`dev/reporting-guidelines.md`. `dev/TODO.md` is reconciled against the author's
review: `[ZT-ZM-Frequency-Fix]` (shipped `a152`) and
`[Aggregate-Summary-DF-Useless]` are closed and removed,
`[Joint-Padding-Window-Tradeoff]` is deferred past v1.0, the five
`[FCC-Surface-Sweep]` decisions and `[PnL-Repr-HTML]` are settled, and
`[FCC-Contract-Gaps]` is added.

## 1.0.0a169

**[Build-Output-Dispatch]** `build_many` now finishes every kind the parser can
produce, under either setting of `update`.

### A bivariate built with `update=False` warned about itself

```python
build('bivariate BV 25 claims '
      'agg A dfreq [0 1] [.5 .5] sev lognorm 50 cv 1.5 '
      'agg B dfreq [0 1] [.5 .5] sev gamma 50 cv 1.0 poisson', update=False)
```
logged `Unexpected: output kind is <class 'aggregate.bivariate.BivariateAggregate'>.
(expr/number?)`. Nothing was broken: the object came back correct and
un-updated, exactly as asked. The message was noise.

`build_many` ran a post-construction `isinstance` chain in which the
`BivariateAggregate` branch was gated on `update is True`, while the no-op
escape hatch for `update is False` listed only `(Aggregate, Portfolio)`.
`BivariateAggregate` is not an `Aggregate` subclass, and that escape hatch
predates bivariates, so every bivariate entry point fell through to the
catch-all: `bivariate` / `bv`, `netceded`, `grossceded`, `grossnet`, `clash`,
and the `dbvsev` discrete forms. `build(<bivariate>, update=False)` had no test,
which is how it survived. `pnl` was unaffected only by accident: a P&L is still
its deferred inner `Aggregate` when the loop runs, and is snapshotted afterwards.

The chain is now guard clauses over two tuples, `no_update` and `updatable`,
declared at the head of the loop. A new output type is registered in one place
instead of two. The catch-all survives as a defensive net, without its stale
`(expr/number?)` hint, which is no longer where an expression lands.

### `update` is honored by truthiness

The branches tested `update is True` and `update is False` by identity, so
`update=1` matched neither: it skipped the update *and* tripped the catch-all
warning. `update` is now normalized with `bool()` where it is resolved.

### `build('3')` evaluates to `3.0`

**Behavior change.** The grammar has always had a top-level `expr` production,
but `_factory` had no branch for it, so a bare expression raised
`ValueError: Cannot build expr objects`. Worse, `_interpret_program` had already
written an `('expr', '3.0')` entry into the recipe base, and that orphan then
broke `to_agg`, whose renderer has no `expr` case and raises a `ValueError` that
the export loop does not catch.

An expression is an answer, not a declaration: it now evaluates to its value and
is never stored. `build('3')` is `3.0`, `build('2 ** 3')` is `8.0`,
`build('exp(1)')` is `e`. Note that the top-level production accepts a subset of
arithmetic: `1 + 2` and a trailing `2 * 3` still do not parse there, because `*`
is the severity scale operator. That is a separate grammar question, untouched
here.

### Also

Seven docstrings carried seven different, incomplete lists of what a *kind* can
be. The four that describe the parser's own output (`UnderwritingParser` and its
`parse`, the `Underwriter` class docstring, `add_recipe`) now agree with
`_factory`: `agg`, `sev`, `port`, `bvagg`, `pnl`, `xpnl`, `distortion`, `expr`.

`tests/test_underwriter.py` gains a parametrized sweep over one program per
output kind, asserting `update=False` returns the right class and logs nothing,
plus the `expr` cases. The bivariate programs are the `MV.*` lines already in
`decl-testers.agg`, reused verbatim.

## 1.0.0a168

**[Recipe-Canonical-Lookup]** plus a documentation sweep: one lookup
implementation instead of two, and the reference chapters realigned with the
package as it actually is.

### `recipe()` is the canonical lookup

Since a164, `build[x]` and `build.recipe(x)` returned the same class and did the
same work, arrived at by two independent code paths. That is a synonym, which
the house rule forbids, and the two paths had drifted: the subscript raised
`Item Nope not found.` and, on an ambiguous name, `Error: no unique object found
matching Nope. Found 2 objects.`, neither of which tells you the remedy.
`recipe()` raised `no recipe named 'Nope'` and named the kinds plus the `kind=`
fix.

`Underwriter.recipe` is now the single implementation. `__getitem__` is a
four-line delegator: `uw[name]` is `uw.recipe(name)`, `uw[kind, name]` is
`uw.recipe(name, kind)`. Consequences:

* Both spellings raise the **better** message, because there is only one that
  can be raised.
* A tuple subscript of the wrong length now raises a clear `ValueError` instead
  of a confusing `KeyError`.
* `recipe(name, kind)` takes the direct `(kind, name)` dict hit when `kind` is
  supplied, so the parser's `sev.X` resolution path stays O(1) rather than
  scanning the store.

No deprecation and no behavior change for correct code: the subscript keeps
working and keeps taking both forms. The docs demote it from a peer to a
parenthetical, so the guide teaches **two** ideas (construct, or look up)
instead of three.

### Reference chapters realigned

The grammar listing was rendered wrong. `grammar(add_to_doc=True)` wraps
`decl.lark` in a `code-block` directive precisely so the page can `include` it,
but chapter 4 used `literalinclude`, so the directive itself appeared as
literal text and the whole grammar shipped as an unhighlighted block indented
four extra spaces. Fixed to `include`. The emitted language changed from `lark`
to `text` because Pygments has no `lark` lexer and an unknown name is a build
warning; `ref_include.rst` is now excluded from the toctree, since it is
included rather than built (it was shipping a stray orphan page).

Also in chapter 4: the test-suite `literalinclude` pointed at
`../aggregate/agg/_test_suite.agg`, which has not existed since the `src/`
layout.

Modules that had no reference page at all now have one: `decl_writer` (the
unparser, whose `format_program` and `spec_to_decl` are re-exported at the top
level) and `parser_errors` join the Parser page; `copula` joins `bivariate` on
Auxiliary; `results` joins Portfolio; and the mixins (`_program`, `_labeled`,
`_help`) plus `_renewal` and `_aggregate_compute_massive` join the internal
architecture map. Three modules are deliberately still unlisted pending a call
on where they belong: `contract_terms`, `decl_pygments`, `style`.

### Cross-references

`autosectionlabel_prefix_document = True` means a section's implicit label is
`<docname>:<Title>`, so a bare `` :ref:`Title` `` never resolves. Every
"Contents" list at the head of the user and technical guides was written bare,
so those lists rendered as dead plain text. 26 refs across 13 files now carry
the document prefix, and 10 more had simply the wrong target: `10 mins
formatting` for `10 min formatting`, `10mins create from knowledge` for `10 min
create from knowledge`, `For Portfolio Objects` for `10 min port bucket`,
`Aggregate Class` / `Portfolio Class` for the reference-chapter sections, and a
`:ref:` that should have been a `:doc:`. Six broken refs remain whose target
does not exist anywhere; they are listed in the run notes rather than guessed
at.

`aggregate.utilities.iman_conover` and
`aggregate.distributions.DEFAULT_RETURN_PERIODS` both moved long ago; their
cross-references now point at `aggregate.iman_conover.iman_conover` and
`aggregate._aggregate.DEFAULT_RETURN_PERIODS`.

### Corrections

The claim that `pprogram` canonicalizes `50% so` to `5 so` was wrong: it
round-trips unchanged. Replaced in both `2_x_10mins.rst` and `dm-claude.qmd`
with five verified examples (`[1:6]` expands, `.3` becomes `0.3`, `50 po`
becomes `500% so`, `exp(.5)` evaluates, `sev.X` resolves inline).

`decl.lark`'s `pnl` comment still described `xpnl` as returning a `PnLTower`,
retired at a129; both `pnl` and `xpnl` return a `PnL`.

## 1.0.0a167

**[Cookbook-Promote]** — the cookbook generator becomes package code, and
`library.agg`'s header stops describing a design that was replaced two versions
ago.

### `aggregate.cookbook`

`dev/generate_cookbook.py` was the odd one out. `recipe.py` promises "one
source, three consumers: render, run, audit", and two of the three already
lived in the package (`Recipe.run`, `build.recipes`). The renderer sitting in
`dev/` was an accident of when it was written, not a decision. Its test loaded
it with `importlib.util.spec_from_file_location`, which is the standard smell.

More to the point, rendering a library is a **capability any DecL library
should have**, not a chore private to this repo:

```python
from aggregate import Underwriter
from aggregate.cookbook import write_cookbook

uw = Underwriter(databases='my_structures')
uw.load()
write_cookbook('book', uw)
```

or from a shell:

```
python -m aggregate.cookbook docs/cookbook [--check]
```

Public surface, submodule access only (no top-level re-export, matching
`bounds` / `ft` / `tweedie`):

| name | what |
|---|---|
| `render_recipe(recipe, slug=, level=, titles=)` | one recipe to Quarto markdown |
| `cookbook_pages(uw=None, sections=, titles=, setup=)` | a whole library to `{filename: text}`; pure, no filesystem |
| `write_cookbook(out_dir, uw=None, check=False)` | write or check; returns `(stale, pages)` |
| `TOPIC_SECTIONS`, `TITLE_OVERRIDES`, `DEFAULT_SETUP` | the defaults, all overridable |

`dev/generate_cookbook.py` survives as a ten-line caller supplying this repo's
library and output directory. An unplaceable recipe now logs a warning through
`logging` rather than printing, since library code should not print.

### `library.agg` header

Two blocks still described the world as it was before a165 and a166:

- `<<decl>>` was documented as keeping `note` / `tags` / `hints`. Since a166 it
  carries **`hints` and nothing else**, and the header now says why: hints
  change how the object *builds*, while the note is the recipe's Problem, the
  tags are the page it sits on, and the doc is the page itself.
- the plain-fence rule was justified by the retired runtime `recipe()` verb
  "emitting them through IPython display". It now says what actually happens:
  a doc body is **data**, and `aggregate.cookbook` rewrites each plain fence
  into a `{python}` Quarto cell on the way out. Since the code inside is passed
  through untouched, a fence may open with a cell option such as
  `#| fig-cap: "..."`, which Quarto honors and `Recipe.run` sees as a comment.
  That is documented now rather than left to be discovered.

Also: "the default `build` knowledge base" became "recipe base" (missed in
a164).

### Also in this commit

Two doc-only pieces that carry no behaviour change. They would not have earned
a bump on their own, but `recipe.py` is touched by both them and the promotion
above, so splitting the commit would have split a file.

**`program` is not "the raw input".** `ProgramMixin.pprogram` said "For the raw
input as supplied to `build` use `program`", and four other places agreed.
What is actually stored is `UnderwritingLexer.preprocess` output: folded onto
one line, comments stripped, extra spaces from the bracket step, and any `doc`
body base64-encoded. Nobody keeps the keystrokes. Corrected in `_program.py`
(module docstring, the `program` attribute, `pprogram`), both user guides,
`Recipe.program`, `add_recipe`, and the `program` / `pprogram` rows of
`dev/FEATURES.csv`. The attribute doc now spells out all four differences and
says the base64 is deliberate, since that is the part that reads as a bug when
you first hit it. The distinction itself stands: `program` is what the parser
was handed, `pprogram` is what it understood, and the gap between them shows
the canonicalization.

**`dev/underwriter.rst`**, a new guide to the Underwriter and the recipe
surface, covering what the 10-minute guide covers plus recipes, tags,
`discover`, the `build(x)` / `build[x]` / `build.recipe(x)` distinction, and
`program` versus `pprogram`. Draft, in `dev/` pending a home in the built docs.

## 1.0.0a166

**[Trailer-Layout]** — the trailer gets its own lines, and formatting drops it
by default.

### Each clause on its own line

`_render_trailer` now returns one fragment per clause instead of a joined
string, so `note` / `tags` / `hints` / `doc` each become a child of the
enclosing block. In `spread` that puts each on its own line at the statement's
indentation level:

```
agg Demo
  10 claims
  1000 xs 0
  sev lognorm 50 cv 1.5
  poisson
  note{a stored note}
  tags{topic:aggregate, role:intro}
  hints{log2=16}
```

`terse` space-joins them back, so the single-line form, and therefore
`spec_to_decl` and `to_agg`, is unchanged. `sev` and `distortion` became
`_Block`s to get the same treatment (declaration as the head, trailer clauses
as children); their terse output is byte-identical. A portfolio's own trailer
moved out of the head line and in front of its units, which is where the
grammar binds it; both layouts re-parse to the same spec.

Only a fragment's first line is indented, so a `doc` body keeps its own column
positions and its fenced code blocks survive.

### `trailer=False` is the default

**Breaking.** `format_program(...)` and `obj.format_program(...)` now default to
`trailer=False`, so `pprogram` and `pprogram_html` show the bare declaration.
Formatting a program is nearly always about the math and the insurance, not the
metadata around it. Pass `trailer=True` for all four clauses, or an iterable
such as `trailer=('hints',)` for a subset.

`spec_to_decl` is untouched and still always emits the full trailer, so
`to_agg` export and round-trip remain exact.

### `<<decl>>` substitutes hints only

`Recipe.decl` renders with `trailer=('hints',)`, down from
`('note', 'tags', 'hints')`. `hints` stays because it changes how the object
*builds*: a copy-pasteable program without it would not reproduce the recipe.
The rest is redundant inside a recipe, where the note is the Problem, the tags
are the page, and the doc is the page itself. Generated cookbook pages are
correspondingly leaner:

```python
a = build('''agg ThreeDice
  dfreq [3]
  dsev [1 2 3 4 5 6]''')
```

## 1.0.0a165

**[Cookbook-Generate]** — cookbook recipe pages are generated from
`library.agg` into native Quarto cells. `_setup.recipe()` is retired.

`recipe('X')` (a158) rendered a recipe by `display(Markdown(...))`-ing its prose
and `exec`-ing its code, all inside **one** cell. That reimplements a slice of
Quarto, and badly:

- **per-block cell options cannot exist** — `#| echo`, `#| fig-cap`,
  `#| warning` are Quarto's, and there were no Quarto blocks to put them on;
- **figures landed in the wrong place** — the inline backend flushes a figure
  that was merely *created* at **cell end**, so a plotting recipe drew its plot
  after the Discussion instead of inside the Solution.

`dev/generate_cookbook.py` emits real ` ```{python} ` cells instead. Quarto then
does the work it is for: figures land where they are created, `#|` options work
per block, a failure is a Quarto cell error, and `freeze` caches per cell.

The transform is deliberately thin. Prose passes through verbatim; each fenced
python block has its fence rewritten from ` ```python ` to ` ```{python} `.
Nothing else. Because the code is untouched, **the page and the pytest harness
execute byte-identical text** — `tests/test_cookbook_generate.py` asserts that
fence by fence, and it is the whole reason for generating rather than
hand-writing. It also means a doc can open a fence with `#| fig-cap: "..."`,
which Quarto honours and `Recipe.run` sees as a comment.

Verified in a real render, not assumed: in `_site/cookbook.html` the PH
distortion recipe emits heading → Problem → solution table → lead-in prose →
**figure** → caption → Discussion → collapsed check, in that order.

- **Pages are grouped by `topic:` tag** into `_recipes_NN_<slug>.qmd`, committed
  (diffable; a docs build should not have to run a generator first) and
  idempotent. `--check` exits 1 if any is stale.
- **Additive, not a rewrite.** Generated fragments are included at the end of
  their section under their own `#sec-recipe-…` anchors, so they cannot collide
  with a hand-written page. Retiring the five-beat stubs is phase 5 of
  `dev/plan-meta-data.md`, page by page with author reaction — not done here.
- **A documented entry whose topic maps to no section is logged, not dropped.**
- `_setup.py`'s star-import now supplies exactly the names
  `Recipe.namespace()` seeds (`aggregate`, `build`, `qd`, `np`, `pd`), so the
  equivalence above is structural rather than coincidental. `recipe()` is gone;
  `qd` / `cbqd` / `pp` / `show` and the calibrated house book stay.
- Fixed while rendering: `_05_05_placement.qmd` called `PnL.valid`, which does
  not exist and broke the build — validation belongs to the engine a PnL wraps
  (`p.engine.valid`). Also the stray `C` in `_01_distortions.qmd` and the stray
  `ads` in `cookbook.qmd`.

## 1.0.0a164

**[Recipe-Is-The-Entry]** — one class, one registry. `knowledge` is gone.

Two objects described one library entry: a `ParsedProgram` holding its kind /
name / spec / program / source / built object, and a separate `Recipe` holding
that same entry's parsed `doc{{{...}}}`. Two registries, two names, one thing —
"which do I use?" had no good answer, and the overlap existed only because
`Recipe` was bolted on in a158 rather than replacing anything.

`ParsedProgram` is deleted. **`Recipe` is the entry**, whole:

```python
r = build.recipe('LimitProfile')     # or build['LimitProfile']
r.kind, r.name, r.spec, r.program, r.source, r.object
r.note, r.tags, r.hints, r.doc                        # the DecL trailer
r.problem, r.solution, r.discussion, r.check          # the parsed doc
r.decl                                                # doc-free declaration
```

The documentation surface is **derived from `spec`**, not stored alongside it,
so there is exactly one copy of every fact. Doc sections parse **lazily**, on
first access: four of 186 shipped entries carry a doc, and `load()` sits on
`build`'s import path.

### Breaking

No alias, no deprecation shim — this is the v1.0 moment to take the old name
out.

| removed | use |
|---|---|
| `Underwriter.knowledge` | `Underwriter.recipes` |
| `Underwriter.add_entry(...)` | `Underwriter.add_recipe(...)` |
| `Underwriter._knowledge` | `Underwriter._recipes` |
| `aggregate.underwriter.ParsedProgram` | `aggregate.recipe.Recipe` |
| `Recipe.program` (a158–a163: the doc-free rendering) | `Recipe.decl` |
| `parse_doc(..., program=)` | `parse_doc(..., decl=)` |

`Recipe.program` now means what it means everywhere else in the package — the
DecL source line, verbatim, doc payload and all. The doc-free canonical
re-rendering that `<<decl>>` expands to is `Recipe.decl`.

### One frame

`build.recipes` merges the two frames a158 left behind. Indexed `(kind, name)`;
identity and audit flags first, the wide payload last so it reads at a
terminal:

```
note  tags  doc  problem  solution  discussion  check  n_asserts  source  program  spec
```

`build.recipes.iloc[:, :9]` is the readable slice.
`build.recipes.query('doc and n_asserts == 0')` still finds a recipe that
describes an invariant without testing it.

`discover()` now reads that frame and filters tags from the `tags` column
rather than digging into each `spec`. Its build path (`plot=` / `describe=`)
trims to `source` / `program` before appending the eleven summary statistics —
carrying the audit flags and `spec` alongside them made an unreadable frame.
Its directory view also stops leaking the base64 `doc{{{...}}}` payload into
the displayed program: the cleaner matched a multi-line fence, but a *stored*
program carries the preprocessor's encoded one-liner.

### Framing correction

A `doc{{{...}}}` is for **cookbook-worthy entries only**. Most entries carry a
`note{}` and nothing more — enough for `discover` and the object dropdown. The
`library.agg` header and the `dev/TODO.md` `[Recipe-Library]` entry both read
as though 182 entries were awaiting docs; they are not.
`build.recipes.query('not doc')` is a **directory**, not a backlog.

## 1.0.0a163

**[Recipe-Library]** — `<<decl>>`: a recipe never retypes the program it
documents.

The four recipes shipped in a160 each **copied their own declaration** into
their Solution block. Two copies of the same program inside one statement, and
nothing tying them together — guaranteed to drift the first time either was
edited. A doc writes this instead:

````
```python
a = build('''<<decl>>''')
```
````

and the entry's own declaration is substituted in.

### The recursion guard

The substituted declaration is rendered **without its doc**, keeping `note` /
`tags` / `hints` — `hints` matters, because it changes how the object builds and
a copy-pasteable program without it would not reproduce the recipe. Dropping
the doc is what stops the doc from quoting itself.

That needed `format_program` to suppress *one* trailer item rather than all of
them, so **`trailer` now accepts a collection as well as a bool**:

```python
format_program(x, trailer=True)                        # all four (default)
format_program(x, trailer=False)                       # none
format_program(x, trailer=('note', 'tags', 'hints'))   # everything but the doc
```

An unknown item name raises rather than silently rendering nothing.
`Recipe.program` is now this doc-free declaration (it previously held the raw
stored program, base64 doc payload and all).

### Canonical form is not a change

The substitution renders **canonical** DecL, which can differ cosmetically from
what is typed in the library: `50% po 2500 xs 2500` comes back as
`50% so 2500 xs 2500` (the writer normalises every partial placement to the
percentage `so` form; both mean half the layer). The `OccurrenceXOL` recipe's
Discussion has been corrected — it explained `po`, a token the reader could no
longer see — and now explains the normalisation itself.

### Fence convention, documented

Plain ` ```python `, **not** ` ```{python} `. Quarto never sees these as source
cells: `recipe()` emits them at runtime through IPython display, and
`Recipe.run` executes them. A ` ```{python} ` fence is Quarto *source* syntax
and would render as a literal label. A fence tagged anything else (` ```text `)
is prose and is not executed — use it to show output.

Both conventions are written into the `library.agg` header, where a recipe
author will actually meet them.

### Also

`tests/test_recipe.py`'s library-dependent tests now use their own
`Underwriter` instead of the global `build` singleton — shared mutable state
that any other test in the session can perturb, which showed up once as a
parallel-only failure.

## 1.0.0a162

**[Test-Loop]** — test-impact analysis for the edit loop, and a scheduling fix
for the one flaky test.

### `pytest-testmon` (new `dev` dependency)

Reruns only the tests your edit actually touched. Measured here: a 3-file scope
went **10.9 s → 0.16 s** with nothing changed, and a real edit to
`src/aggregate/recipe.py` selected **6 of 55** tests in 0.71 s.

The working command is **`pytest -n0 --dist no --testmon-forceselect`**, and
every flag earns its place:

- **`--testmon-forceselect`, not plain `--testmon`.** `addopts` carries
  `-m 'not slow'`, and testmon *silently* downgrades to `--testmon-noselect`
  (reorder, deselect nothing) whenever `-m` / `-k` / `--lf` / `::test_name` is
  present. Plain `--testmon` therefore looks like it works — it writes
  `.testmondata`, prints no warning — while running the entire suite. This cost
  a diagnosis; it is now documented so it costs nobody else one.
- **`-n0 --dist no`** to override `-n auto --dist loadgroup`; testmon traces
  coverage in-process and xdist breaks it. (`-p no:xdist` does *not* work — it
  makes those `addopts` unparseable.)

`.testmondata` is gitignored. `.pytest_cache` needs no entry: pytest writes its
own `.pytest_cache/.gitignore`; testmon does not.

### `--dist loadgroup` and `xdist_group`

`addopts` gains `--dist loadgroup`; ungrouped tests distribute as before, but
tests sharing an `xdist_group` name go to one worker and run sequentially
against each other. `test_bivariate.py` is now
`pytestmark = [slow, xdist_group('bivariate')]`: each case allocates a 2-D FFT
grid, and several concurrently once exhausted memory — a numpy allocation
failure in a test whose own grid is 64x64, so the pressure was its neighbours.
Costs ~10% on the gate (337 s → 373 s), which buys away a flake class that
previously cost ~9 minutes to re-diagnose.

Use `xdist_group` when tests are individually fine but collectively too large;
use `slow` when a test is simply long.

### CLAUDE.md

Test guidance restructured into three explicit tiers (testmon edit loop / full
fast suite pre-commit / everything at a version bump), with the measured
baseline recorded — **2,562 cases in ~105 s, ~41 ms per test** — so "is the
suite bloated?" has a number attached. It is not; the three parametrized corpus
files are 37% of cases and ~29% of wall clock.

Also documents a real trap: **there are two virtualenvs**, `.venv` (development)
and `.doc-venv` (docs build only), and an ambient
`UV_PROJECT_ENVIRONMENT=.doc-venv` makes a bare `uv run` resolve to the latter.
Both are editable onto the same `src/`, so results agree — but a package
installed in one is invisible to the other.

## 1.0.0a161

**[Tag-Namespace]** — library tags are namespaced, and **no longer restate the
object's type**.

The a159 vocabulary was wrong: of 186 entries, **103 carried a tag that merely
repeated their own kind**, and 80 of those uses carried no information at all —
`aggregate` appeared on 31 `agg` entries and nowhere else, `portfolio` on 29
`port` entries and nowhere else, and likewise `bivariate` and `distortion`.
`discover(kind=...)` already filters by type, so those tags were dead weight
that invited exactly the question *"why does type appear twice, and which do I
use?"*

They were not simply deletable, though: **42 of 55 `severity` tags sit on `agg`
entries** (the severity zoo, the curve reference), where the word names the
*subject*, not the type. So the fix is a namespace, not a cull:

| | |
|---|---|
| `topic:X` | what the recipe is **about** — `severity` `frequency` `aggregate` `reinsurance` `pnl` `portfolio` `distortion` `bivariate` `bounds` `ruin` `numerics` |
| `role:X` | where it stands — `hero` `intro` `reference` `paper` |
| `check:X` | which invariant its Check asserts (unchanged) |
| `slow` | the one deliberately bare tag — it names a **pytest marker**, not a property of the subject |

A namespaced tag cannot be mistaken for a kind, so `sev UnitSeverity
tags{topic:severity}` is unambiguous and legal. All 186 entries were re-tagged
mechanically: **275 tag uses before, 275 after**, identical modulo the prefix.

### `kind` and `tags` are independent axes

```python
build.discover(kind='sev')                          # by TYPE
build.discover(tags='topic:severity')               # by SUBJECT
build.discover(kind='agg', tags='topic:severity')   # both
```

`discover(kind=)` gains no code, only documentation saying it *is* the type
filter. Explicitly rejected: a "look-through" tag synthesising the kind, and
auto-appending kind to tags — both recreate the two-ways-to-say-it problem this
removes.

### Guarded

Two new tests in `tests/test_agg_libraries.py`: every tag must carry a known
namespace (or be in a one-item bare allow-list), and separately, no tag may
name its own entry's kind. The first is the mechanism, the second is the
reason — stated apart so a future bare tag still has to satisfy it. Verified
against a synthetic library that a reintroduced `tags{aggregate}` fails.

## 1.0.0a160

**[Recipe-Library]** — phase 4 of `dev/plan-meta-data.md`: **the library tests
itself.** New `tests/test_library_recipes.py` turns every documented entry into
a test case — the Solution block executes, then the Check block runs in the same
namespace and its assertions must hold.

Before this, the shipped example library had exactly **one** assertion against
it: that the file loaded. Nothing in it was ever `build()`-ed, so an entry could
parse cleanly and still produce garbage moments or fail validation with no test
signal. The library's own stated invariants are now the test.

Entries tagged `slow` are quarantined behind the `slow` marker, matching the
suite-wide fast-by-default policy. A doc with no runnable ```python block, or a
Check section with no `assert`, fails — a recipe that cannot be executed cannot
be trusted, and an invariant that is stated but not tested is the exact failure
mode the mechanism exists to prevent.

### First four recipes

`library.agg` gains real `doc{{{...}}}` bodies on four entries, one per check
archetype, 15 assertions between them:

- **`ThreeDice`** *(check:independent-oracle)* — the discrete case where the FFT
  is exact, not approximate. Asserts `E[A] = 10.5` to 1e-12 and both extreme
  probabilities equal `(1/6)^3`.
- **`LimitProfile`** *(check:reconciliation)* — the premium x limit x loss-ratio
  table. Asserts the aggregate mean *is* the premium-weighted expected loss
  (`1000x0.8 + 2000x0.7 + 500x0.5 = 2450`) exactly, and that `E[N] x E[X]`
  still reconciles across the blended profile.
- **`PHDistortion`** *(check:limiting-case)* — asserts `g(0)=0`, `g(1)=1`,
  increasing, **concave** (the coherence property), and `g(s) >= s` — a load,
  never a discount.
- **`OccurrenceXOL`** *(check:reconciliation)* — two-layer partial placement.
  Asserts `net + ceded = gross` on both severity and aggregate, and that
  occurrence cover leaves the claim count alone.

That last one is worth a note: the count check reads the *theoretic* `EX`
column, because gross frequency has no `Est EX` (it is an input, so there is
nothing to estimate and the cell is NaN). Writing the check found that.

## 1.0.0a159

**[Recipe-Library]** — phase 3 of `dev/plan-meta-data.md`: **three shipped
libraries become one.** `examples.agg` (38 entries), `cookbook.agg` (118) and
`actuarial-severity-curves.agg` (28) are replaced by a single
**`library.agg`** with **186 entries**, descriptive globally-unique names, and
a tag on every entry.

### Breaking

- **`databases` now defaults to `('library',)`.** The out-of-the-box knowledge
  base grows from 38 entries to 186 — `cookbook.agg` and
  `actuarial-severity-curves.agg` were shipped but *not* loaded by default, so
  most of this was invisible unless you asked for it. Intended, not a side
  effect.
- **The single-letter filing prefixes are gone.** `E.LimitProfile` is
  `LimitProfile`, `K.PH` is `PHDistortion`, `A.CatXOLTower` is `CatXOLTower`.
  Grouping is what `tags{}` is for; the letters were a comment convention that
  no code ever read. Names leading with a **citekey** keep it —
  `Mack2003.Lognorm` is a citation, not a filing code.
- **Names are unique across kinds**, enforced at load by
  `Underwriter._check_library_names_unique` (scoped to `library.agg`; the
  regression corpora reuse names deliberately). This is what lets
  `build('X')` and `build.recipe('X')` always mean the same entry, with no
  `kind=` disambiguator anywhere.

### Two entries recovered

`cookbook.agg` defined `J.Re01` **three times**, so it declared 120 statements
but loaded 118 — the last definition silently won and two examples were
unreachable. They are now `ReinsuranceOccurrenceTower`,
`ReinsuranceAggregateLayer` and `ReinsuranceOccurrenceWithAggLimit`.

### `discover(tags=...)`

```python
build.discover(tags='hero')                # the landing-page heroes
build.discover(tags='severity, reference') # BOTH tags -- tags narrow
```

Tag vocabulary (documented in the library header): domain (`severity`,
`frequency`, `aggregate`, `reinsurance`, `pnl`, `portfolio`, `distortion`,
`bivariate`, `numerics`), role (`hero`, `intro`, `reference`, `paper`), check
archetype (`check:*`), and cost (`slow`).

### Fixed

- **View-pair statements now lift `tags` / `doc`, not just `note`.** A
  `netceded agg X ... tags{...}` has no trailer of its own (the grammar is
  `NETCEDED agg_out`), so the trailer lands on the inner agg; the parser lifted
  only `note` onto the bivariate. Caught by the new "every entry is tagged"
  invariant.

### Notes on the merge

Statements were transformed at text level rather than regenerated from specs,
so programs and their `note{}` text survive byte-for-byte. Two things that bit
during the merge and are worth knowing:

- **`UnderwritingLexer.preprocess` is not safe for this.** Its bracket step
  inserts a space after every `]`, rewriting `E[N]` inside a note to `E [N] `.
  `decl_writer._split_statements` is the same splitter minus that step, written
  for exactly this reason.
- **A `port`'s tags go on the header line.** Appending them at the end of a
  folded portfolio statement binds them to the last *unit* — the positional
  rule `tests/test_trailer_attachment.py` pins.

Every one of the 183 carried-over entries was verified to keep its original
spec and note; the only differences are the intended rename and the added tags.

## 1.0.0a158

**[Recipe-Library]** — phase 2 of `dev/plan-meta-data.md`: the runtime that
turns a `doc{{{...}}}` body into something you can render, run and audit. New
module **`aggregate.recipe`**.

### `Recipe` — one source, three consumers

`parse_doc(text)` splits a doc body on its level-2 headings into the
*Python Cookbook* sections plus a check:

- **`## Problem`** — what you are trying to do.
- **`## Solution`** — the code. Written self-contained, so a reader can
  copy-paste it off the page.
- **`## Discussion`** — why it works.
- **`## Check`** — pure `assert`s. **Runs in the namespace the Solution left
  behind**, so it can assert against the objects that code just built. That is
  what makes a recipe self-*testing* rather than merely self-describing.

Anything under a non-canonical heading (`## References`, say) is preserved
verbatim in `Recipe.extra` rather than being glued onto the preceding section,
and headings are hunted only *outside* fenced code — so a `## banner` comment
at the start of a line inside a ```python block stays code.

`Recipe.run(ns=None)` executes Solution then Check in one namespace seeded with
`build`, `qd`, `np`, `pd` and `aggregate`; a failure is re-raised naming the
recipe and the block. `Recipe.markdown()` reassembles the prose in canonical
order. `Recipe.n_asserts` is the audit metric that matters — a Check section
with zero asserts states an invariant without testing it.

### `Underwriter.recipe(name)` and `Underwriter.recipes`

- **`build.recipe('X')`** returns the parsed `Recipe`, resolving **by name
  alone**. The knowledge base is keyed `(kind, name)`, but a recipe is
  addressed the way a reader says it; a name that exists under two kinds raises
  rather than silently picking, and `kind=` disambiguates. An entry with no doc
  yields an empty Recipe — absence of documentation is a fact to audit, not an
  exception.
- **`build.recipes`** is the audit frame, one row per entry indexed
  `(kind, name)`, with `tags` / `note` / `doc` / `problem` / `solution` /
  `discussion` / `check` / `n_asserts` / `source`. It exists to answer two
  questions:

  ```python
  build.recipes.query('not doc')                  # what is undocumented?
  build.recipes.query('doc and n_asserts == 0')   # documented but unchecked?
  ```

### Cookbook

`docs/cookbook/_setup.py` gains **`recipe(name)`**, the one cookbook verb: a
recipe page becomes a heading plus one call, and the page and the pytest
harness read the same source so they cannot drift. Output goes through
`IPython.display` rather than Quarto's `#| output: asis` — asis text and rich
display output do not interleave reliably in one cell, and a recipe needs its
tables and plots to land *between* its prose sections.

`Recipe.run` executes code carried in a `.agg` file. For the shipped library
that is the same trust level as the rest of the package; a third-party `.agg`
deserves the same reading as a third-party Python module. Signing is recorded
as post-v1.0 `[Recipe-Doc-Signing]`.

## 1.0.0a157

**[Recipe-Library]** — phase 1 of `dev/plan-meta-data.md`: the DecL trailer
grows from two clauses to four, so a library entry can carry its own
description, grouping and executable recipe. Phases 2–6 (the recipe runtime,
the merged `library.agg`, the recipe test harness, and the Problem / Solution /
Discussion cookbook) follow.

### `tags{...}` — grouping and selection

A comma- and/or space-separated slug list, decomposed to an ordered tuple with
duplicates dropped: `tags{severity, heavy-tail}` and
`tags{severity heavy-tail}` are the same thing. Lands on `spec['tags']` and on
the object as `.tags`. This is the machine-readable replacement for the
`# A. Showcase` letter-prefix convention, which only ever existed as a comment
that nothing in `src/aggregate` read.

### `doc{{{...}}}` — a long-form markdown recipe, in the language

```
agg LimitProfile
    [1000 2000 500] prem at [.8 .7 .5] lr
    ...
    note{one-line abstract}
    tags{aggregate, intro}
    doc{{{
## Problem
...
## Solution
```python
a = build('LimitProfile')
```
}}}
```

A doc body needs everything DecL preprocessing destroys — `#` headings (step 1
strips to end of line), blank lines (steps 4–5 split statements on them), fenced
code, braces, semicolons. So **the body never reaches the lexer as text**: a new
**step 0** in `UnderwritingLexer.preprocess` lifts it out and substitutes
URL-safe base64, whose alphabet (`A-Za-z0-9-_=`) contains no `#`, `//`, `}`,
`[`, `]`, `;` or whitespace and therefore survives steps 1–6 byte-for-byte. The
`DOC` terminal decodes it again; `decl_writer` re-emits the raw body between
real fences, so `format_program` output re-parses.

The opening fence ends its line and **the closing fence must be alone on its
line**. That anchor makes an inline `}}}` inside Python — `{'a': {'b': {'c':
1}}}` — harmless; a line that *is* `}}}` is the single forbidden body content.

### Breaking / behavioural

- **`distortion` gains a trailer.** `dist` statements previously accepted no
  `note{}` at all (the shipped library headers said so); they now take the full
  trailer like every other statement, and `Distortion` carries `.note` /
  `.tags` / `.hints` / `.doc`. New `Distortion.from_spec(spec)` replaces
  `Distortion(**spec)` at the three construction sites — the kind subclasses
  take strict natural-parameter signatures and reject unknown keywords, so
  trailer metadata is applied after construction.
- **`PnL` gains `.note` / `.tags` / `.hints` / `.doc`**, which it never had.
  They come from the statement's build recipe via `PnL._adopt_engine`, not from
  the wrapped engine — for a `port.NAME` engine the Portfolio's own metadata is
  not the P&L's.
- **`tags` and `doc` are *conditional* spec keys**, present only when written.
  `note`/`hints` remain unconditional. The captured spec snapshot compares key
  sets exactly, so unconditional keys would have failed all 163 cases.
- `Underwriter.discover` strips `tags{}` and `doc{{{}}}` from its program column
  alongside `note{}`/`hints{}`.

### Grammar

`trailer` is now a repetition rather than five spelled-out alternatives:

```lark
trailer: trailer_item*
trailer_item: NOTE -> ... | TAGS -> ... | HINTS -> ... | DOC -> ...
```

The original spelled-out form existed because the natural two-item spelling
gives the *empty* trailer two parses. A star has exactly one empty parse, and it
scales — four order-free optional items written out would be 65 alternatives.
Repeating a clause is a clear `ValueError` from the transformer.

### Two new guards

The parser runs with Lark's default `ambiguity='resolve'`, which silently picks
one parse and never warns — so a grammar edit could change behaviour invisibly.
Two tests close that hole, both written *before* the grammar changed so they
record the pre-existing behaviour:

- **`tests/test_grammar_ambiguity.py`** builds a second Lark with
  `ambiguity='explicit'` and sweeps all 600+ shipped statements, asserting the
  ambiguous set equals a pinned allow-list. Two entries are accepted:
  `ssev -3 * lognorm ...` (parses as `scale(-3)` or `negate(scale(3))` —
  algebraically identical, pre-existing) and the deliberate
  `AE.Trailer.BivariateBare` fixture below.
- **`tests/test_trailer_attachment.py`** pins *which* parse wins. A trailing
  trailer binds to the **outer** object. For `port` that is forced positionally
  (`port_out` puts its trailer before `agg_list`); for a `bivariate` with
  neither a copula clause nor an outer frequency it is a genuine ambiguity —
  the last component's trailer and the bivariate's are adjacent and both
  nullable — resolved to the bivariate. Round-trip tests cannot catch a flipped
  binding, because the flipped parse is still a fixed point.

## 1.0.0a156

**[Ruin-Example-Punchups]** — legend on the `ruin_example` psi(u) panel: the
linear curve, the log-scale dashed curve and the `(u0, psi(u0))` marker are
now labeled, with the twin axis's handles merged into one `fontsize='x-small'`
legend (upper right). Folds in the author's figure-proportion tweaks
(`FIG_W * 2` x `FIG_H`, equal panel widths, x-small paths legend).

## 1.0.0a155

**[Ruin-Example-Punchups]** — `pedagogy.ruin_example` figure polish, straight
after the a153 ship.

- **Ruin-time rug on by default** — `show_default_times` (the `'|'` rug below
  the zero line marking the ruin time of *every* simulated path that dies, not
  just the drawn ones) now defaults `True`; it shows at a glance where the
  full simulation falls.
- **psi(u) panel, always on** — the figure is now two panels: sample paths on
  the left (the wide hero panel, unchanged), psi against initial surplus on
  the right in the `5_x_pk.rst` / PIR-Fig-9.1 style — linear solid plus a
  dashed log-scale twin (a straight line under the Lundberg regime), marker
  and crosshair at `(u0, psi(u0))`, x-range auto-capped where psi falls below
  1e-5 rather than showing the whole `2**log2` grid.
- **House figure conventions** — size from the `FIG_W` / `FIG_H` constants
  (`(FIG_W * 3, FIG_H * 2)`), created with `layout='constrained'` (the
  author's edit, folded in); no `tight_layout` anywhere. Standing rule going
  forward: figures are created `layout='constrained'`, never
  `fig.tight_layout()`.

## 1.0.0a154

**[Program-Mixin]** — the DecL round-trip surface (`program` / `pprogram` /
`pprogram_html`) collapses into the codebase's **third mixin**, and gains the
render axis that motivated the work: `format_program(trailer=False)` renders a
declaration **without its `note{...}` / `hints{...}` trailer**. Plan:
`dev/done/plan-program-mixin.md`.

### `trailer=` — the bare declaration

```python
>>> a = build('agg X 10 claims sev lognorm 50 cv 1 poisson note{a stored note}')
>>> print(a.format_program(layout='terse', trailer=False))
agg X 10 claims sev lognorm 50 cv 1 poisson
```

One flag, not two: `note{...}` and `hints{...}` are one grammar construct
(`decl.lark`) emitted by one function, so one switch governs both. It applies
through the whole tree — a portfolio's units and a bivariate's components lose
their trailers too, not just the head line. This is the form to print in a
paper, a docstring or an exhibit, where a stored note or a build hint is noise.

`trailer` is the one render axis that is **not** round-trip safe: `fmt` and
`layout` re-parse to the identical spec, `trailer=False` re-parses to the same
spec with `note` and `hints` blanked. Nothing else moves — in particular the
semantic `!` markers (unconditional severity, the zero-modified mean pin,
defective `dwait`) are clause syntax rather than trailer and always survive.
`spec_to_decl` is untouched, so `to_agg` and the round-trip snapshot contract
are unaffected.

### `ProgramMixin` (`src/aggregate/_program.py`)

`pprogram` / `pprogram_html` were **ten near-identical properties across five
classes and five files**, each a three-line delegation to
`decl_writer.format_program`; `pprogram_html` was byte-identical on
`Aggregate` / `Portfolio` and again on `PnL` / `BivariateAggregate`. They now
live once. Same idiom as `LabeledMixin` and `HelpMixin` — no `__init__`, so it
stays transparent to each host's `super()` chain, which matters because the
hosts range from `object` to `scipy.stats.rv_continuous`.

The duplication had stopped being tidiness and started blocking capability:
every host hard-coded the render options in its own property body, so a new
axis cost an edit per class. That is why `trailer` ships with the mixin and not
before it.

Hosts: `Aggregate`, `Portfolio`, `PnL`, `Severity`, `BivariateAggregate`, and
— the hole this closed — **`Distortion`**, which is DecL-creatable (the writer
has always had a `'distortion'` kind renderer, and `build` has always stamped
`obj.program`) but declared neither the attribute nor `pprogram`. Exactly the
gap `HelpMixin` closed for `Frequency` / `GridDistribution` at `a150`.

The mixin is deliberately **narrower** than `HelpMixin`: `help` is on all
eleven matrix columns, a DecL declaration on six. `Frequency`,
`GridDistribution` and the three `Bounds` classes are not hosts and must never
gain `pprogram`; `tests/test_fcc_surface.py` asserts that in both directions.

An object-bound `format_program(fmt=, layout=, trailer=)` is the parametrized
worker — the twin of the free function, binding `self.program`, mirroring
`HelpMixin.help` over `utilities.agg_help`. `pprogram` and `pprogram_html` are
that method at fixed defaults and are unchanged.

**Considered and rejected: an `InfoMixin`.** Ten `info` properties with the
same five-sentence docstring look like the bigger prize, but per class the
genuinely shared code is two lines — the `info_row` comprehension and the
`'\n'.join`. Everything else is payload, and even the footers diverge. It would
relocate ≈20 lines and force ten hosts through a hook to do it. What `info`
shares is its *contract*, and that already has two homes: `dev/info-strings.rst`
and `tests/test_fcc_surface.py`.

### Also

- **`Portfolio.nice_program` retired.** A `textwrap.fill` over the *raw*
  program — method not property, non-NumPy docstring, zero call sites in
  `src/`, `tests/` or `docs/`. The last surviving non-`decl_writer` program
  printer. Use `format_program(layout='terse')` for a compact form.
- **`Portfolio.note`.** The parser produced `note` for a `port` spec and the
  writer rendered it, but `Portfolio` never stored one — so a portfolio note
  could be written and never read back. `Aggregate`, `Severity` and
  `BivariateAggregate` all retained theirs. `Portfolio.__init__` takes `note=`
  and `build` passes it.
- **`dev/FEATURES.csv`**: `pprogram` / `pprogram_html` / `program` gain their
  `Distortion` column; new `format_program` row; `nice_program` row deleted;
  `note` and `hints` promoted from undocumented attributes to documented rows
  (undocumented attributes 90 → 88, undocumented capabilities still **zero**).

**No behaviour change at defaults** — verified byte-for-byte over all 1204
renders of every statement in every shipped `.agg` file, in both layouts.

## 1.0.0a153

**[Ruin-Wiener-Hopf]** — eventual-ruin probabilities for renewal (Sparre-
Andersen) aggregates, and a strict Poisson guard on the classical solver.
Plan: `dev/done/plan-ruin-wiener-hopf.md`.

### `Aggregate.wiener_hopf(rho, kind='index', log2=None)`

The renewal counterpart of `pollaczeck_khinchine`: for a `years ... wait ...`
frequency, computes `psi(u)` = probability of eventual ruin as a function of
initial surplus, via cepstral Wiener-Hopf factorization of the per-claim
random walk `Y = X - cW` (severity minus premium accrued over the wait,
`c = (1 + rho) E[X] / E[W]` so `rho` keeps its PK margin-to-loss meaning).
The kernel `ruin_cepstral` lives in `_renewal.py` (four complex FFTs:
Spitzer identity, support separation in the cepstrum, `z = 1` regularization
by dividing out `1 - z^{-1}`), ported from the author's `ruin-probabilities`
notes. Severity rides the existing discretized `sev_density_df.p_sev`; the
wait mixture is discretized on the money grid by the same rounding scheme
(`_discretize_wait_pmf`, following `wait_count_pmf`). psi is returned on the
severity u-grid (`log2` widens the circle for heavy tails; a warning fires
when the top-of-grid psi exceeds 1e-6). Guards: renewal frequency only,
non-defective wait law, nonnegative severity grid, net profit condition
checked on the grid. For exponential waits it agrees with PK up to
discretization (WH is O(bs^2)-accurate, PK O(bs)-biased); with exponential
severity it reproduces the Lundberg form `psi(u) = psi(0) exp(-beta
(1 - psi(0)) u)` for any wait law — both are pinned by `tests/test_ruin.py`.

### `pollaczeck_khinchine` requires Poisson

**BREAKING:** the docstring always said "assumes frequency is Poisson"; now
it is enforced — any other frequency (including fixed and the mixed Poissons
gamma/delaporte/...) raises `ValueError`, with a pointer to `wiener_hopf`
for renewal frequencies. The Pareto example in `5_x_pk.rst` switches its
carrier frequency from `fixed` to `poisson` (identical PK output — the
method reads only severity).

### Shared `RuinFunction` named tuple

Both solvers now return `RuinFunction(ruin, find_u, mean, density)`
(clearing a long-standing TODO): psi as a pd.Series, the capital-lookup
closure, the discretized severity mean, and the method's u-grid density
vector — the integrated-severity (equilibrium) density for PK, the pmf of
the all-time maximum for WH. A namedtuple is a tuple, so existing positional
unpacking is unchanged. The `find_u` closure construction is factored into
`_ruin_find_u`, shared by both.

### Pedagogy

- **`ruin_example(agg, rho, u0, ...)`** — general, single-unit successor to
  `plot_ruin_surplus_paths`, ported from the notes' example builder: exact
  psi (auto-dispatch poisson→PK, renewal→WH), Monte Carlo validation
  simulating the walk at claim instants (severity and renewal waits sampled
  from the *same discretized model* the solver prices, so sim-vs-exact is
  apples-to-apples), sample-path plot with expected trend, law-of-the-
  iterated-logarithm funnel and ruin-time markers, and a summary DataFrame
  (exact vs simulated psi, safety loading, LIL variance rate, horizons, grid
  diagnostics). Horizons are auto-derived from the exact psi curve.
- **`_ruin_function`** dispatch helper routes `ClassicalPremium.illustrate`,
  `plot_ruin_surplus_paths` and `natural_scale` by frequency kind (poisson →
  PK, renewal → WH, else `ValueError`). `plot_ruin_surplus_paths` no longer
  computes PK twice per unit (psi computed once at `padding=2`, capital
  passed to `illustrate` as `K` — numerically identical to before), and its
  docstring now documents the required `PZTest` portfolio shape.

Docs pending rebuild (`5_x_pk.rst` example edit). Tests: `tests/test_ruin.py`
(13 cases); DecL programs mirrored in `decl-testers.agg` (AD.*).

## 1.0.0a152

**[ZT-ZM-Frequency-Fix]** — zero-truncated / zero-modified frequency
reparameterized to the textbook form, and fixed. `zt` previously raised
`ValueError: function value at x=0.0 is NaN` for **every** parameterization,
and so did every `zm` that *reduced* the zero mass; only zero-inflation worked.

### The rule

**The exposure clause states the un-modified (base) mean; the modification
moves it.** The `(a, b, 1)` construction holds the base distribution fixed and
rescales its positive-count probabilities, so `E[N]` is an *output*:

```
p_k^M = (1 - p0M)/(1 - p0) p_k,  k >= 1        G^M(z) = (1 - c) + c G(z)
```

This is the parameterization of Klugman-Panjer-Willmot (2012) §6.6, Loss Data
Analytics ch. 2, and R's `actuar` (`dzmpois(x, lambda, p0)` takes the base
`lambda`), so textbook problems now transcribe directly. It is also always
feasible — every `p0M` in `[0, 1)` is admissible — and closed form, where the
old mean-matching solve had a non-rectangular feasible region (a ZT count can
never average below 1) and could invert to absurd base means (a mean-4
aggregate with 95% zeros needed a base Poisson of mean **80**).

**BREAKING:** `4 claims ... poisson zm 0.5` now has `E[N] = 2.0373`, not 4.

### `!` pins the mean

Append `!` to the `zm` / `zt` clause to opt back in to the old behaviour:
`aggregate` solves for the base mean whose realized `E[N]` equals the exposure
clause. `!` is the existing DecL *unconditional* marker (`sev !`, `dsev !`,
`dwait !`), and the realized count mean is the unconditional one.

Use it when the exposure clause states **money**. `1000 loss`,
`1000 premium at 0.65 lr` and `100 exposure at 0.05 rate` state a target a
shifted mean would silently miss, so those forms now raise a
**`ZeroModifiedExposureWarning`** naming the shortfall and the fix. The plain
`n claims` form stays silent — count in, shifted count out is the default.

### Added

- **`Frequency.base_mean`** (attribute) — the un-modified mean, what
  `freq_moms` / `freq_pgf` consume. Renames `unmodified_mean`.
- **`Frequency.modify_mean(base_mean=None)`** — forward shift, closed form.
- **`Frequency.solve_base_mean(target_mean)`** — the inverse; the only
  surviving solve, reached only under `!`. Raises with the attainable bound
  (`E[N] > 1 - p0M`) when a target is unreachable.
- **`Frequency.apply_deductible(survival)`** — Loss Models §8.6: the payment
  count `N^P` implied by a loss count under a deductible. Base parameter scales
  by `v = S(d)`, zero mass becomes `P_{N^L}(1 - v)`; a zero-*truncated* loss
  count yields a zero-*modified* payment count. Poisson / geometric / negbin /
  binomial.
- **`Aggregate.base_mean`** (property) — `n` for an unmodified frequency, the
  base mean under `zm` / `zt`. Every PGF evaluation now routes through it.

### Fixed

- `Frequency.prob_eq_0` returned `_prob_eq_0(en)` for a zero-modified
  frequency; under `zm` / `zt` the answer is `freq_p0` by construction.
- `_bounded_severity_window` fed the realized mean back into `freq_moms`,
  applying the modification twice and sizing a **4-bucket** grid for a
  zero-modified Poisson(4) body. It now uses the base moments, whose reach is
  what the aggregate support actually needs.
- `MomentAggregator.get_fsa_stats(remix=True)` re-entered `freq_moms` with the
  accumulated total, double-applying the shift; it now carries
  `tot_freq_base`.
- `create_frequency()` rebuilt the count program from the realized `n` while
  preserving the `zm` clause — a third double-application.

### Docs

Three `.. todo:: Implement ZT and ZM!` blocks replaced with executed
solutions: **Loss Data Analytics 5.5.4** (ZM Poisson/Burr under a deductible,
cross-checked against the elementary thinning identities), **Loss Models
Example 9.11** (ZM binomial, exact to machine precision), and the empty
`DecL/050_frequency.rst` section. Loss Models 9.12 keeps its todo, now stating
the real gap — a compound frequency with a zero-truncated *secondary*, which is
unrelated to this change. Two `examples.agg` programs uncommented.

## 1.0.0a151

**[FCC-Alias-Retirement]** — the breaking half of pass 2 of
`[FCC-Surface-Sweep]`, and the end of that sweep. One name per concept: the
matrix read turned up six live alias pairs, and one row (`var`) covering two
incompatible meanings. **All removals, no deprecation shims** (pre-beta).

### The rule

**`var` means VARIANCE, always. VaR is `q`, always. There is no `ppf`.**
`var` had been VaR on `Portfolio` / `PnL` / `GridDistribution`, *variance* on
`Severity` (scipy), and deliberately absent on `Aggregate` — that absence is
what made the collision visible.

### Removed

- **`Portfolio.var`, `PnL.var`, `GridDistribution.var`** (all VaR aliases of
  `q`). Use `q`.
- **`Aggregate.ppf`** (alias of `q`).
- **`Aggregate.pla`, `Portfolio.pla`, `GridDistribution.pla`** — the canonical
  name is `prob_loss_assets`.
- **`Aggregate.cramer_lundberg`** — `pollaczeck_khinchine` was always the
  definition and is now the only name.
- **`Portfolio.unit_renamer`** — deprecated alias of `renamer` since `a128`;
  the tail of `a133 [Label-Canonical]`.

**Kept, deliberately:** `Severity.ppf` and `Severity.var`. `Severity` wraps a
frozen `scipy.stats` rv and inherits its surface — that is where the odd naming
comes from (`mean()`, `var()` = variance, `ppf`, `rvs` vs `sample`, `sev_mean`
vs `Aggregate.sev_m`), and it is not this library's to rename. `dev/FEATURES.csv`
records it in a `# severity-scipy` legend row so it stops reading as drift.
This also closes the old open question of whether `Aggregate.sev_*` should
follow `actual_*`: it should not.

### Renamed

- **`Frequency.prn_eq_0(n)` → `prob_eq_0`**, a **zero-argument property**
  evaluated at `en` (the unconditional expected count the owning `Aggregate`
  stamps at construction, as `freq_df` already used). It joins the `prob_eq_0`
  family — `P(N = 0)` is the same question one level down from `P(X = 0)` on
  `Aggregate` / `Portfolio` / `PnL` — and raises if `en` is unset or the kind
  has no closed form. The parametrized worker survives as the private
  `_prob_eq_0(n)`, which `Aggregate` calls per mixture component and the
  zero-modification solver inverts at trial means; the split mirrors `a149`'s
  `tail_df` (property) vs `tail_periods_df(periods=)` (worker).

### Call sites

`pedagogy.py` (Cramér–Lundberg ruin figures) and
`docs/5_technical_guides/5_x_pk.rst` move to `pollaczeck_khinchine`; four test
modules move off the retired aliases and now assert they are gone.
`dev/FEATURES.csv` drops the `pla` and `prn_eq_0` rows and re-stamps — its
**undocumented-capability count is now zero**. Docs are edited in lockstep but
a rebuild is pending.

## 1.0.0a150

**[FCC-Help-Mixin]** — the additive half of pass 2 of `[FCC-Surface-Sweep]`.
A hand read of `dev/FEATURES.csv` found the auditor's blind spots: it checks
only the presence grid, never the `notes` prose, the `kind` column, or the
undocumented plain attributes. This release closes the surface gaps that read
turned up; the alias retirements it also turned up follow in `a151`. Purely
additive — nothing is renamed or removed.

- **`HelpMixin` (`src/aggregate/_help.py`)** — `.help(regex)` was a one-line
  delegation to `utilities.agg_help` **copy-pasted into nine classes across
  eight modules**, each with its own near-identical twelve-line docstring. It
  now lives once, in the codebase's **second mixin** (after `LabeledMixin`;
  same idiom — no `__init__`, so it stays transparent to each host's `super()`
  chain, which matters because the hosts range from `object` to
  `scipy.stats.rv_continuous`). Hosts: `Aggregate`, `Portfolio`, `PnL`,
  `Severity`, `BivariateAggregate`, `Bounds`, `_HullEngine` (so
  `AllocationBounds` / `PricingBounds`), `Distortion`, `Underwriter`, and — the
  two that had no `help` at all — **`Frequency`** and **`GridDistribution`**.
  `agg_help` is imported *inside* the method: `utilities` imports
  `_grid_distribution`, so a module-level import would close the cycle.
- **`PnL.tvar`** — a P&L delegated `q` / `var` / `cdf` / `sf` to its result
  `GridDistribution` but not `tvar`, so the library's flagship risk measure was
  unavailable on the newest first-class class. One delegation; the result GD
  already carries the payoff orientation, so it reads the correct tail.
- **`Severity.pprogram` / `.pprogram_html`** — `a149` claimed every
  DecL-creatable class round-trips its declaration, but `Severity` was missed:
  `build('sev X lognorm 100 cv 2')` stamps `program` yet could not render the
  canonical form. An inline `sev` clause returns `''` — the enclosing
  `Aggregate` owns the text.
- **`Portfolio.n_units`** gained the docstring it never had.

### `dev/FEATURES.csv` and its auditor

The matrix is rewritten and widened; `dev/regen_features.py` grew three checks.
No version-visible behaviour, but the table is the plan of record for the rest
of the sweep.

- **Two new columns: `GridDistribution` and `Distortion`** (nine → eleven).
  `GridDistribution` is the engine every `q` / `var` / `tvar` / `cdf` / `sf`
  routes through and the type of `PnL.result`; `Distortion` already carried
  `info`, `help`, `plot` and the whole `LabeledMixin` surface — more of the FCC
  surface than `Frequency` has — while three notes said "not a column here".
- **A three-token cell vocabulary, with a legend row.** `Y*` ("same concept,
  different shape") was already in use with nothing documenting it. Added `~`
  for a **name collision**: `Distortion.tvar` / `.mean` / `.max` are static
  *constructors* of a distortion, not risk measures, so a presence-only audit
  would have demanded a misleading `Y` in the `tvar` row.
- **Stale prose fixed.** Eight rows still described `ReinstatementAnalysis`,
  `VariableRatingAnalysis` and `pnl.analysis` — deleted at `a144` — as live,
  including a `WRINKLE` on `density` documenting a name collision that no
  longer exists.
- **Two rows had an unquoted comma in `notes`**, silently truncating the field
  for any `DictReader` consumer (`na_grid`, `actual_cv`). Merged and quoted.
- **~70 new rows**: the `gd-api` and `distortion-api` / `distortion-ctor`
  groups, the `Bounds` lazy cloud surface, the Portfolio density/allocation
  working set, and the incomplete stat families (`sev_var`, `est_sev_sd` /
  `_cv` / `_skew` / `_var` — invisible before because the auditor counted
  undocumented attributes rather than naming them). Undocumented capabilities
  went from 65 to 2, and both remainders are the aliases `a151` deletes.
- **`regen_features.py`**: `member_kind` now reports `functools.cached_property`
  as a `property` (it is neither a `property` instance nor `callable`, so it
  fell through to `classattr` — mislabelling `freq_df`, `Distortion.info` and
  the whole lazy `Bounds` surface); the `kind` column is audited as a new
  failure class; undocumented attributes are listed by name; the scipy skip set
  covers `rv_continuous.__init__`'s *instance* attributes; and `~` cells get
  their own informational COLLISION section.

## 1.0.0a149

**[FCC-Surface-Sweep]** — the first pass of step 1 of `plan-for-v1.md`: work
`dev/FEATURES.csv` class by class and close the gaps and inconsistencies in the
first-class-citizen surface. Nine classes are in scope — `Aggregate`,
`Portfolio`, `BivariateAggregate`, `PnL`, `Severity`, `Frequency`, `Bounds`,
`AllocationBounds`, `PricingBounds`. **Several renames are breaking**; there
are no deprecation aliases (pre-beta).

### Breaking renames

- **`agg_m` / `agg_sd` / `agg_cv` / `agg_skew` / `agg_var` → `actual_*`** on
  `Aggregate` and `Portfolio`. The moment surface is now two symmetric
  families — **`actual_*`** (theoretical / analytic) and **`est_*`** (realised
  FFT grid) — instead of one prefix that read like the class name. The
  `Underwriter` knowledge-base summary frame renames its columns to match
  (`actual_m`, `actual_cv`, `actual_sd`, `actual_skew`); `tests/data/peg_baseline.json`
  keys renamed with it (values unchanged).
- **`PnL.mean` / `.sd` / `.cv` / `.skew` → `est_m` / `est_sd` / `est_cv` /
  `est_skew`.** `est_`, not `actual_`: a P&L is evaluated per-atom over the
  *source's realised grid*, so every moment inherits that grid's
  discretization. (Within the ledger those per-atom values are exact — the
  `EX` basis `validation_df` audits a rebucketed `bs > 0` leg against — but
  relative to the generating `Aggregate`'s analytic moments they are
  estimates.)
- **`PnL.prob_loss` → `prob_eq_0`, with new semantics `P(result == 0)`**, and
  the property added to `Aggregate` and `Portfolio`. `prob_loss` could not
  travel: "the probability of a loss" and "the probability of losing money"
  are opposite tails of the same number depending on `value_type`.
  `prob_eq_0` is sign-neutral — *no loss* on a loss object, *exactly break
  even* on a payoff. On demand from the realised grid, `None` before `update`.
  On a continuous severity the zero bucket also absorbs everything that
  discretizes to zero, so the reading is `P(N = 0)` plus that sliver.
  The `Aggregate.info` row `P(loss)` (which was always `n/a`) becomes `P(X=0)`;
  `Portfolio.info` gains the same row.
- **`Aggregate.tail_df` / `Portfolio.tail_df` are now properties.** The
  first-class form takes no arguments and uses the standard
  `DEFAULT_RETURN_PERIODS` ladder; the parametrized worker is the new
  **`tail_periods_df(periods=None)`**. `BivariateAggregate.tail_df` was already
  a property, so `tail_df` is uniform across the matrix.
- **`Severity.support_description` deleted.** Its content — the declared layer
  form, atom listing, or signed support — now reads out through
  `Severity.tail_description` (appended when it says something the interval
  does not) and the new `tail_explanation`, i.e. through the narrative pair
  every class carries rather than a Severity-only name.

### Added

- **`info` on every first-class class.** Was `Aggregate` / `Portfolio` /
  `BivariateAggregate` (plus `Distortion`); now also `PnL`, `Severity`,
  `Frequency`, `Bounds`, `AllocationBounds`, `PricingBounds` — terse, but the
  same fixed-layout contract (`info_row`, no conditional rows, `n/a`
  placeholder). `AllocationBounds` and `PricingBounds` share the slice-geometry
  block via `_HullEngine._hull_info_rows`. Row catalogues in
  `dev/info-strings.rst`.
- **`pprogram` / `pprogram_html` on `BivariateAggregate` and `PnL`** — every
  DecL-creatable class now round-trips its declaration. `PnL.program` is
  stamped by `build` and falls back to `engine.program`.
- **`Severity.actual_m` / `actual_sd` / `actual_cv` / `actual_var` /
  `actual_skew`** — the analytic post-layer moments, cached off `moms()`.
  There is no `est_*` counterpart: a standalone `Severity` is never
  discretized. scipy's inherited `mean()` / `std()` / `var()` / `stats()` are
  untouched. Note `actual_cv` is the *achieved* CV; `sev_cv` is the *declared*
  one, and they differ under a layer, splice or shift.
- **`tail_explanation` on `Severity` and `Frequency`**, completing the
  short/long narrative pair wherever `tail_description` exists. Severity walks
  support → declared layer → what the tail class means for the moments (the
  power-law index and the first infinite moment). Frequency walks family →
  zero-modification → `E[N]` / `SD(N)` / dispersion vs Poisson → whether the
  count tail can set the aggregate tail on its own.
- **`bs_explanation`, `tail_explanation` and `validation_explanation` on
  `BivariateAggregate`** — per-axis grid and joint cell count with the deficit
  gate; per-axis realised tails plus correlation and copula tau; and the long
  form of the one-line `info` validation row.
- **`Frequency.name`** — read-only alias of `freq_name`, so every first-class
  class answers to `name`.
- **`Portfolio.reins_description` / `Portfolio.reins_kinds`** — look-throughs
  over the units (`Unit A: net of 500 xs 500 per occurrence. Unit B: no
  reinsurance.`), naming non-ceding units too so the sentence covers the whole
  book; both collapse to the aggregate's own clean-book answer when no unit
  cedes. `Portfolio.info` gains a `reinsurance` row.

### Changed

- **`Aggregate._text_info_blob` / `Portfolio._text_info_blob` are one line**
  (space-joined sentences, not stacked) and close with
  `Validation: {validation_explanation}.` — exactly what `_repr_html_`
  already did. `qd` no longer prints its own separate validation block, so
  text and HTML now say the same thing once.

### Notes

- `tests/test_fcc_surface.py` is new: the executable half of
  `dev/FEATURES.csv`, checking the shared surfaces exist and agree across the
  nine classes.
- `uv run python dev/regen_features.py` reports 0 mismatches / 0 stale.
- Docs (`.rst` / `.qmd` sources) updated for the `agg_*` → `actual_*` rename;
  the doc build is pending.

## 1.0.0a148

**[Resolved-En-Plus-Freq-Df]** — two small frequency-surface items.

- **`Aggregate.en` now holds the resolved per-component claim count** for
  empirical and renewal frequencies. The limit-profile broadcast arm never
  wrote the resolved count back, so `a.en` leaked the `-1` spec sentinel (for
  any `dfreq` body, predating the renewal work) while `a.n` and the stats were
  correct. The arm now mirrors the mixture-product arm's write-back;
  `Aggregate.freq_pmf` and the `en`-consuming paths see the true count.
- **`Frequency.freq_df`** — new on-demand cached property (computed only on
  first access): a comparison table with index `n`, columns `p` (the count
  pmf) and `po_p` (the mean-matched Poisson pmf) — an eyeball diagnostic for
  dispersion, cluster fatness, and renewal regularity (`wait expon` shows
  `p ≡ po_p` to the kernel noise floor; `mixed gamma ν` shows var/mean
  = 1 + ν²n exactly). Works for **every** frequency kind: the Aggregate
  stamps its resolved unconditional claim count onto the new
  `Frequency.en` attribute at construction, and parametric kinds invert
  their own pgf on an FFT grid sized by their moments (a standalone
  frequency raises until `en` is set); the empirical family (`dfreq` /
  `renewal`) uses its exact materialized pmf and intrinsic mean. Guards:
  mean ≤ 1000 (a toy, small-count diagnostic); integer support for the
  empirical path.

## 1.0.0a147

**[Wait-Clause-Layers]** — the severity layer transform `y xs a` is now
available on the renewal wait clause: `wait y xs a <dist> [!]` (layer term
before the distribution, mirroring the exposures layer clause).

- **Semantics.** Conditional by default — `W' = ((W − a) | W > a) ∧ y`, short
  raw waits conditioned away, cap atom at `y`. With `!` the unconditional
  transform `W' = min((W − a)+, y)`: raw waits ≤ `a` collapse to a zero-wait
  atom (mass `P(W ≤ a)`) that rides the existing geometric-batch machinery as
  **simultaneous-claim clusters**. Exact anchor (memorylessness):
  `wait inf xs a expon !` ≡ `geometric_batch_compose(Poisson counts, 1 − e^{−a/μ})`.
- **Plumbing.** New `Aggregate` kwargs `wait_attachment` / `wait_limit`
  (broadcast like every other `wait_*` term — vector layers give a mixture of
  differently layered waits). The layer lives entirely inside the component
  `Severity` (`exp_attachment`/`exp_limit`); the kernel consumes the atom at 0
  via the existing `p0` extraction with no changes. Splice + layer on one wait
  is rejected (`ValueError`); layers on `dwait` are not supported (write the
  clamped outcomes directly).
- **Hard-atom grid snap.** A finite cap `y` is a genuine atom in an otherwise
  continuous wait law. When `{y, T}` are commensurable, `wait_grid` refines the
  continuous bucket size to `step / 2^j` (new `hard_atoms` argument, new
  `hard_atom_snap` row in `_renewal_bs_df`) so the atom sits exactly on the
  lattice, and the kernel switches to the **closed-interval readout**: sums
  `S_k` landing exactly at `T` count in full (the half-bucket convention would
  halve that atom mass — e.g. `min(W, 0.5)` waits at `T = 10` gave `EN = 19.5`
  instead of 20). Incommensurable caps fall back to the continuous grid with a
  documented O(h/2) placement smear. Diagnostics: `FrequencyRenewal.wait_snapped`;
  `convergence_check()` keeps the closed readout at h/2 on snapped grids
  (O(h) continuous convergence, delta ≈ 2× remaining error).
- Writer round-trips the layer term; corpus lines `Y.Renewal.Layer*` /
  `AB.Renewal.Layer*`; spec snapshot re-captured (additive).

## 1.0.0a146

**[Renewal-Frequency-Wait-Clause]** — the claim-generation process can now be a
general Sparre-Andersen renewal process: any iid waiting-time law, not just
exponential ⇒ Poisson. New DecL surface: a `T years [at r rate]` exposure head
strictly paired (at the grammar level) with a `wait <severity-expression>` /
`dwait [outcomes] [probs] [!]` clause in the frequency slot:

```
agg Ren 10 years sev lognorm 100 cv 1 wait expon        # ≡ 10 claims … poisson
agg Det 3 years dsev [1] dwait [1]                      # N ≡ 3
agg Clu 2 years dsev [1] dwait [0 1] [.5 .5]            # geometric claim clusters
agg Ter 2 years dsev [1] wait expon splice [0 1.1] !    # terminating (defective)
```

- **Engine** (`_renewal.py`, ported from the author's `sparre` notes library):
  the count pmf comes from `{N(T) ≥ k} = {S_k ≤ T}` via a Plancherel /
  no-inverse-FFT method — two forward rffts, then one vector multiply + one dot
  per k, with exponential tilting (`θL = 20`, the float64 optimum) damping
  circular wrap. Discretization REUSES the severity kernel
  (`discretize_severities`, extracted verbatim from `Aggregate.discretize` into
  `_aggregate_compute.py`). Grid sizing (shape / accuracy / coverage rules,
  exact-lattice override for commensurable `dwait`, log2 window [16, 24]) is
  recorded in `_renewal_bs_df`, mirroring the `_bs_window_df` idiom.
- **Zero waits = claim clusters**: `P(W ≤ 0)` (atoms at 0 plus collapsed
  negative mass, warned) is factored out exactly and recomposed as geometric
  batches — `N = Σ_{i≤M+1} G_i − 1`, including the boundary batch riding at the
  last epoch ≤ T (the plan's original composition dropped it; caught in review
  against the `dwait [0 1] [.5 .5]` enumeration `P(N=m) = m/2^{m+1}`).
  Defective waits (splice `!` windows, `dwait` probs summing < 1) terminate the
  process; the count pmf stays proper.
- **`wait` reuses the severity mini-language** (scaling, mixtures, splice, `!`,
  `sev.NAME`); flat `wait_*` spec keys mirror `sev_*` as `Aggregate.__init__`
  kwargs (+ `exp_years`, `exp_rate`). `at r rate` books informational premium
  `T·r` (feeds PnL `inherit premium`); the count comes solely from the wait law.
- **Post-build the renewal frequency IS an empirical frequency**
  (`FrequencyRenewal(FrequencyEmpirical)`, `freq_a = 0..kmax`, `freq_b = pN`):
  moments, pgf, count support, `freq_pmf`, validation and grid sizing all run
  the ordinary dfreq machinery; `create_frequency()` emits the realized
  `dfreq [0:kmax] [pN…]` with no kernel recomputation. `Frequency.convergence_check()`
  is the explicit-opt-in Richardson h/2 diagnostic. Writer round-trips the wait
  clause (`_render_wait` via a `sev_*` view through `_render_dist`).
- Anchors verified in tests: `10 years wait expon` ≡ `10 claims poisson`
  (same grid, densities to 1e-8, identical q(0.99)); gamma → `gammainc(ka, λT)`;
  uniform(0,1) T=1 → `k/(k+1)!`; inverse Gaussian closed form; O(h²) Richardson;
  cluster/defective enumerations; mixture mean hits the second-order renewal
  expansion `T/μ + (σ²−μ²)/2μ²`.

**[Empirical-PGF-Horner-Dispatch]** — `FrequencyEmpirical.freq_pgf` no longer
materializes the `n_atoms × len(z)` complex matrix. `evaluate_pgf_polynomial`
(`_aggregate_compute.py`) dispatches on an operation-count model: dense supports
→ Horner (one fused multiply-add per degree, single accumulator); sparse
supports → sorted-gap square-and-multiply with an explicit power-of-two cache
(never complex `np.power` on large exponents). Fractional/negative outcomes keep
the legacy matrix path verbatim. Baselines re-captured: summation reordering
moves densities ≤ 2e-20/bucket, but `Port.Bounded`'s lifted/mass pricing
surfaces read the numerical sup inside the bounded book's noise plateau, which
relocated a few buckets (`test_baseline`, spotchecks, `numerics3_precapture`).

Also: the `_en < 0` empirical-count sentinel now resolves *before* the
premium/lr reconciliation in the limit-profile broadcast arm, so a
premium-carrying empirical/renewal exposure records per-component `lr`
correctly (no legacy path paired the two).

## 1.0.0a145

**[Exact-Discrete-Reachability-Guard]** — the `exact_discrete` grid-sizing method
no longer has *unconditional* top priority; it must now clear a reachability
guard. A fully-discrete `dfreq`/`fixed` × `dsev` aggregate has a finite,
exactly-computable combinatorial support `[N·s_min, N·s_max]`, but for a large
count that support is a gross *overstatement*: each extreme corner needs *every*
one of `N` claims to land on the same extreme atom, an astronomically improbable
event, so the mass sits nowhere near it (a near-normal aggregate whose support is
vast but whose spread the CLT concentrates). Sizing the grid to the fictional
support then coarsened `bs` far past the lattice step and **aliased the
severity** — e.g. `agg L dfreq[1000000] dsev[-1000 20] [0.019 .981]` coarsened
`bs` from `1` to `20000` and the estimated `sd` came out 6.3× too large, tripping
`ALIASING` / `AGG_CV` / `AGG_SKEW`.

- Each support corner now carries a `log10` *attainment probability*
  `log10 P(N=N_ach) + N_ach·log10 P(X=s_ext)`, where `N_ach` is the count that
  realizes that corner (`N_max` for the outer extreme, `N_min` for the inner one;
  a corner pinned at `0` is always reachable). `exact_discrete` keeps top priority
  only when at least one corner is reachable; when **both** fall below
  `discretization.exact_discrete_reach_logp` (new setting, default `-20`) the row
  is recorded in `_bs_window_df` for inspection (`coverage = 'support unreachable
  (rejected)'`, both `logp` in `note`) but not selected, and the sizer falls
  through to `bounded_small` / `moment`. The reported life-insurance case now
  sizes on the moment window (`bs=40`, mean exact to 1e-10, `sd` within 1%).
- Genuine small-count discrete books (dice, coins) have corners near
  `10**-3`..`10**-6`, well above the floor — unchanged and byte-stable. An
  asymmetric book with one reachable corner keeps its exact support.
- `Aggregate._exact_discrete_window` now returns `(A_lo, A_hi, bs_lattice,
  logp_lo, logp_hi)`. Design doc `dev/bucket-selection.rst` updated in lockstep.

## 1.0.0a144

**[Decommission-Analysis-Classes]** — the `ReinstatementAnalysis` and
`VariableRatingAnalysis` drill-down classes are removed. The generic P&L (a
signed group ledger over `GridDistribution`s, assembled in `_pnl_builders.py`)
made them redundant: the domain math already lives on the *terms* objects and
`Aggregate.occ_bivariate`, and the builders read that directly. No public API
break — neither class was ever re-exported (submodule-access only).

- **`variable_rating.py` deleted** along with `Aggregate.variable_rating_analysis`.
  The `kind == 'var'` DecL dispatch never read the analysis; the builder was
  always the engine.
- **`reinstatement.py` deleted** along with `Aggregate.reinstatement_analysis`.
  The three load-bearing pieces relocated: `ReinstatementTerms` folded into
  `contract_terms.py` (alongside its `ContractTerms` siblings — import it from
  there now); `check_joint_grid_adequacy` + `JOINT_KINK_MIN_BUCKETS` and the new
  `agg_tier_maps` / `build_reinstatement_source` helpers moved to
  `_pnl_builders.py`. `build_reinstatement_pnl` now takes the joint source,
  terms and economics as explicit arguments instead of an analysis object.
- **`pnl.analysis` attribute removed** — the wrapped engine is reachable via
  `pnl.engine` for drill-down; the terms live on `pnl.engine.reinstatement_terms`
  / `pnl.engine.variable_terms`. `plot_reinstatement` and the `qd`
  `ReinstatementAnalysis` branch are gone.
- **Numbers unchanged.** The builder paths are untouched; every P&L exhibit
  (`stats_df`, `economics`, leg means, `pnl`↔`xpnl` agreement) is identical. The
  decl regression suites were repointed onto the PnL's own surface (leg SDs off
  `stats_df`, exact means off the `(L, R)` joint `p._source`), not weakened.
- Docs pending a rebuild (no `.rst` referenced the removed submodule symbols).

## 1.0.0a143

**[Reins-Premium-Placement-Scaling]** — reinsurance premiums in a `pnl` / `xpnl`
are quoted at **100% placement** and scaled down by the fraction actually placed
(the layer's `share of` / `part of`).

- **`deposit` and `rate` now scale by the placement share.** In
  `Underwriter._resolve_reins_economics`, a `deposit` resolves to `share ×
  amount` and a `rate` to `share × rate × gross_premium` (previously both used
  the full 100% figure). `rol` was already correct — `share × rol × limit` is
  the placed premium — and is unchanged (adding another factor would
  double-count). `cede` is a fraction of the *placed* premium, so the ceding
  commission scales automatically. Reinstatements read the resolved occurrence
  premium (`econ['pc_occ']`), so their base premium scales too. A 100% placement
  (`… xs …`, share = 1) is unaffected.
- **`swing` collar currency terms scale by share.** `basic` / `min` / `max`
  (entered at 100%) are multiplied by the decorated layer's placement share
  before building `SwingTerms`; `lcm` (a dimensionless loss multiplier applied
  to the already-placed ceded loss) is left unchanged, so the ceded premium
  `clip(share·basic + lcm·A, share·min, share·max)` is the placed figure.
- **Breaking (intended):** any `pnl` / `xpnl` with a non-100% `deposit` / `rate`
  cession, or a placed `swing`, now books a smaller (correctly-placed) ceded
  premium. Example (`50% so 2000 xs 3000`): `deposit 1500` → 750, `rate 30%` of
  5000 → 750, `deposit 1500 cede 25%` commission → 187.5. `rol` figures are
  unchanged.

## 1.0.0a142

**[Help-Everywhere] + [Summary-Computed-Moments] + [Repr-Trim]** — a batch of
short display punchups.

- **`.help` on every first-class citizen.** Added the `.help(regex, ...)`
  lookup to `Distortion`, `Bounds`, `PnL`, `Severity`, and the two bounds
  engines `AllocationBounds` / `PricingBounds` (via their shared `_HullEngine`
  base) — it was already on `Aggregate`, `Portfolio`, `BivariateAggregate`,
  `Underwriter`. The backing `agg_help` gained a **`private=False`** axis (skip
  `_`-prefixed names unless asked) and its defaults changed to **`lod='terse',
  values='none', private=False`** — so a bare `.help('regex')` is now a clean
  public-name listing rather than dumping docstrings, values, and private
  members. The `.help` methods on all classes default to the same.
- **`Validation.passes` predicate.** The passing test (clean
  `NOT_UNREASONABLE`, or only `REINSURANCE`) moved onto the `Validation` flag as
  a `.passes` property; the duplicated private `Aggregate._validation_passes` /
  `Portfolio._validation_passes` helpers are gone (`qd` now reads
  `x.valid.passes`).
- **`summary_df` reports computed moments.** `Aggregate.summary_df` and
  `Portfolio.summary_df` now show the **realised (FFT-grid) `est_*`** moments
  for the `Sev` / `Agg` / `total` rows — the same values validation audits
  against — not the analytic (theoretical) moments. The `Freq` row stays
  PGF-exact (the engine estimates no count distribution); before `update()` the
  whole frame falls back to theoretical. The `E[X]` column is renamed
  **`Mean`** (friendlier), and the percentile columns are renamed **`P01` /
  `Median` / `P99`** (matching the `PnL` card headers), from the old `p0.01 /
  p0.50 / p0.99`.
- **`_repr_html_` trimmed and validation always shown.** `Aggregate` and
  `Portfolio` HTML reprs now render just the intro + `summary_df` headline (the
  `tail_df` return-period table is dropped from the inline repr — still
  available via `.tail_df()`). The intro always closes with the validation
  result inline (`Validation: {validation_explanation}.`), replacing the
  fail-only red block.
- **`make_grid` / `make_mosaic` default to house sizing.** When `figsize` is
  omitted, `aggregate.plots.make_grid(nrows, ncols)` now sizes the figure as
  `(ncols * FIG_W, nrows * FIG_H)` (one panel-sized cell per grid position)
  instead of falling back to matplotlib's global `figure.figsize` — so
  `make_grid(1, 3)` is three panels wide out of the box. `make_mosaic` gets the
  same default, reading the grid shape from its `layout`. Explicit `figsize=`
  still wins. The internal callers already passed exactly this size, so their
  now-redundant `figsize=` arguments were removed (KISS); only direct
  interactive `make_grid`/`make_mosaic` calls change.

## 1.0.0a141

**[Kappa-Walks] + [Single-Block-One-Step-Walk] + [All-Gross-Loss-Renames]** —
author feedback on a140 (three items, `dev/done/plan-pnl-faces-punchlist.md` addendum).

- **Every DecL `xpnl` walk is now per-atom with a footing scenario (κ)
  ladder** — the a136 marginal stitch ([GC-Tower-Marginal-Stitch]) is retired
  from the walks. The source is picked by the occurrence tier: no occ program
  → the engine's exact gross marginal (1-D atoms; an aggregate cover's legs
  are functions `g(x)`); occ program → the occurrence `(gross, ceded)` joint
  from `occ_bivariate` (the ceded-occ aggregate is NOT a function of the
  gross aggregate, so no 1-D source can carry a footing walk). Consequences:
  every column — EX and each κ — **foots exactly** down the sheet; the
  impact row is a true per-atom difference (its SD/percentiles are of the
  difference distribution, not per-stat deltas); loss-basis LAE books
  stochastic `rate·l` on all walks (the stitch was all-marginal); running
  nets carry real covariance. **Trade-off:** occ-bearing walks now run on
  the joint's budget-sized common bucket size, so their EX values are
  joint-grid accurate (~1e-3 rel typical; kinked agg-tier maps worse —
  `CoarseJointGridWarning` now guards these joints too) rather than
  marginal-exact; the consolidated `pnl` keeps the exact marginals, and
  builds are slower by one 2-D FFT. The kernel's `stitched_rows` mode stays
  (the designated no-joint assembly seam, e.g. a future massive-source
  xpnl; covered by a direct kernel test) but the stitched builder helpers
  are gone.
- **One-step walk is a single block** — `xpnl` over a plain engine now
  renders just its `Gross` block: no duplicated grand rows, no zero impact
  (`force_tower` is presentation-only; `_ledger_plan(force_grand=)`
  removed).
- **Renames (author picks):** the walk base step fallback is **`Gross`**
  (capital G); the closing grand step's index key is **`All`** (too many
  things were already called Total) on stats_df, summary_df and every tower
  — including the massive route; the default loss leg label is **`Loss`**
  (and `Loss (net)` on the consolidated faces). **Breaking** for code
  addressing `('Total', 'Margin', …)` tuples, the `'gross'` step or the
  `'loss'` row default; declared `as` labels are untouched.

## 1.0.0a140

**[Engine-Reference-On-PnL] + [Reinst-Joint-Grid-Adequacy]** — Phases 3–4 of
`dev/done/plan-pnl-faces-punchlist.md`.

- **`PnL.engine`** — every DecL-assembled P&L (and `make_pnl`) now keeps a
  reference to the wrapped stochastic engine (the inner
  `Aggregate`/`Portfolio`); `None` on hand-built kernel P&Ls. The `source`
  stays the *simplest sufficient object*: the plain face now normalizes an
  engine source to its density as a `GridDistribution` (uniform with the
  reinsurance faces — "GD in, engine ref kept").
- **One canonical premium-leg default** — the plain `pnl`'s consideration leg
  default is now `'premium'` (matching the walk's gross step);
  `'consideration'` retired as the DecL default. **Breaking** for code
  addressing the default row label.
- **`CoarseJointGridWarning`** (new in `constants`) — the reinstatement joint
  runs on a budget-sized common bucket size; a kinked treaty map across too
  few buckets carries a Jensen-type O(bs) bias the internal audits cannot
  see (they compare on the same grid). `reinstatement_analysis` now warns
  when a kink region — the occurrence fill width, an aggregate cover's layer
  width — spans fewer than `reinstatement.JOINT_KINK_MIN_BUCKETS` (= 20)
  buckets, naming the `bs=`/`log2_x=`/`log2_y=` knobs as the remedy.
- Deferred pending an author format pick: [Walk-Step-Default-Labels]
  (undeclared cover steps still read `'ceded occ'`/`'ceded agg'`; proposal
  in `dev/done/plan-pnl-faces-punchlist.md` is the layer descriptor, e.g. `'occ 4750 xs 250'`).

## 1.0.0a139

**[Consolidated-Reinstatement-PnL]** — Phase 2 of `dev/done/plan-pnl-faces-punchlist.md`: `pnl` over
a reinstatements program is now the **consolidated single-group net view**
([Decision-PnL-Is-Consolidated]), closing the [2D-Deferred] carve-out. One
sell group of 2-D legs over the same (L, R) joint the walk uses: a stochastic
**net premium** `P_G − D − h(R) − P_agg(L,R) + commissions` (the reinstatement
premium — and a swing/slide/pc-rated aggregate tier — ride along) against
**loss (net)** `−(L − A(R) − REC_agg(L,R))`, plus expenses. Loss-basis LAE
stays stochastic `rate·l` (axis 0 carries the gross loss — nothing off-source
here, cf. [Consolidated-LAE-Off-Source]). Because both faces are pushforwards
of the ONE joint, `pnl.mean` equals the `xpnl` walk's grand result **exactly**
(no engine-drift tolerance). Scenario (κ) ladder; committed scale stays
`gross − deposit − pc_agg`; `economics` now attached on both reinstatement
faces. **Breaking:** the step tower is `xpnl`-only — reinstatement programs
wanting the tower under `pnl` must switch to `xpnl` (the
`test_reinstatement_decl.py` ledger tests did exactly that).

## 1.0.0a138

**[One-Classifier-Fix]** — Phase 1 of `dev/done/plan-pnl-faces-punchlist.md` ([PnL-Faces-Punchlist]):
the pnl/xpnl assembly now classifies the **occurrence tier**
{none | gc | reinstatements} and the **aggregate tier** {none | gc | feature}
independently and dispatches on the pair. The old single-kind elif chain let a
feature branch shadow the reinstatements clause and mis-scope an inuring
occurrence program; every cell of the composition matrix is now correct (see
`tests/test_composition_matrix.py`).

- **(GC occ, feature agg)** — previously silently wrong in both faces
  ([Var-Feature-Composed-With-Occ-Program]): the walk booked the net-of-occ
  loss under the gross label and dropped the occ step; the consolidated net
  premium omitted `- pc_occ + c_occ`. Now: the consolidated face folds the
  inuring program's constants into the net premium (source = the net-of-occ
  atoms, the feature's true subject; scenario κ ladder); the walk is the
  stitched four-block tower gross → ceded occ → feature cover → Total, the
  feature rows exact pushforwards of the net-of-occ marginal
  (`_fn_marginal_entry`; marginal P ladder; loss-basis LAE deterministic).
- **(reinstatements occ, feature agg)** — previously the feature branch won
  and the reinstatements clause was **silently dropped** (annual cap gone,
  net-as-gross, occ economics lost; margins could flip sign)
  ([Reinstatements-Dropped-By-Feature-Branch]). Now the reinstatement branch
  wins and the feature rides the same (L, R) joint via the tier maps:
  `ReinstatementAnalysis(agg_feature_terms=)` swaps exactly one map — swing
  premium `phi(g_rec)`, slide/pc commission `phi(g_rec/P_C)*P_C` (a real
  stochastic leg, folded into the uw columns; the scalar commission shift
  stays 0), corridor-adjusted recovery. Both faces remain the 2-D tower
  ([2D-Deferred]).
- **[XPnL-Zero-Premium-Cessions]** — reinsurance presence, not economics
  presence, now drives the face: a cession side with no
  `deposit`/`rol`/`rate` books at **zero ceded premium** with one
  `ZeroPremiumCessionWarning` (new in `constants`). `pnl` over an unpriced
  cession is the consolidated net view (was: the plain face with
  `consideration`/`loss` labels); `xpnl` walks it (was: NotImplementedError).
  Feature-decorated sides and reinstated occ layers are exempt (they own /
  require their premium clause).
- **[Decision-XPnL-Plain-Is-One-Step-Walk]** — `xpnl` over a plain engine
  returns the trivial one-step walk (gross → Total; grand rows duplicate the
  step, impact identically zero) instead of erroring. Kernel:
  `PnL(force_tower=True)` presents a single-group ledger in the tower shape
  (`_ledger_plan(force_grand=)`); `build_plain_pnl(walk=)`. `xpnl` over a
  `port` stays rejected (the total hides its units).
- Internals: the guaranteed-cost walk's row descriptions are now tagged
  (`('const', b)` / `('affine', persp, a, b)` / `('fn', persp, f)`) and
  assembled by the shared `_stitch_ledger`; `build_variable_pnl` gained
  `econ=`; `Aggregate.reinstatement_analysis` gained `agg_feature_terms=`.
- Tests: new `tests/test_composition_matrix.py` (hand-pushforward exactness
  for the composed cells, the reinstatement-cap survival, matrix routing);
  `test_pnl_ceded_premium` zero-premium flips; the plain-xpnl test asserts
  the one-step walk.

## 1.0.0a137

**[PnL-Engine-Name-Roundtrip]** — bug fix. `format_program`/`pprogram` on a
`pnl` (or `xpnl`) program no longer rewrites the backing loss engine's name:
the parser preserves the source engine name under `spec['engine_name']`
(alongside the existing `engine_note`/`engine_label` carve-outs), the writer
renders it instead of always synthesizing `NAME_e`, and the underwriter pops it
before the inner Aggregate build. `pnl NetOcc … agg A:NetOcc …` now round-trips
as `agg A:NetOcc` rather than `agg NetOcc_e`. The engine name remains cosmetic
(discarded at build); `NAME_e` stays as the fallback for specs built without an
engine name.

## 1.0.0a136

**[PnL-Consolidated-XPnL-Walk]** — Phases 2–5 of
`dev/done/plan-pnl-consolidated-xpnl-walk.md`: two objects, two questions, no
conditional shapes.

- **[Decision-PnL-Is-Consolidated]** — **breaking**: `build('pnl …')` always
  returns a **single-group** consolidated net view: `Consideration` = net
  premium (gross − ceded premiums + commissions, one constant leg for
  guaranteed cost), `Obligation` = net loss (the deepest `reins_density_df`
  net marginal) + the pnl's own expenses, `Margin`. The a125/a129
  premium-clause promotion to the Gross/Ceded/Net group ledger is removed
  from the pnl route. `Aggregate.make_pnl(gross=, ceded=)` follows
  (`build_gcn_pnl` → `build_consolidated_pnl`). Leg labels per
  [Flag-Net-Premium-Leg-Label]: `'net premium'` / `'loss (net)'`, a declared
  label qualifying as `'<label> (net)'`. Variable-rating programs (retro /
  swing / slide / pc / corridor — 1-D) consolidate too: the feature's map
  folds into the net premium / net loss legs
  ([Decision-2D-Is-Computation-Only]); reinstatements keep the per-atom 2-D
  tower for now ([2D-Deferred] transitional carve-out). `pnl.economics` and
  `pnl.analysis` ride along (both now always-present attributes, `None` when
  absent). The gross/ceded-split card is pended as
  **[Accounting-Summary-DF]** in `dev/TODO.md`.
- **[Decision-XPnL-Is-A-Recipe]** — **breaking return type**:
  `build('xpnl …')` returns a plain **multi-group `PnL`** (the walk: gross →
  each cover → Total with per-step results, running nets, and the closing
  impact), replacing the bare 4-row `stack_marginal_pnls` DataFrame. No
  `XPnL` class. Guaranteed-cost walks are **marginal-stitched**
  ([GC-Tower-Marginal-Stitch]): every row is an affine transform of one
  exact engine marginal, derived rows read the engine's own net marginals
  (never cross-row sums), the EX column foots exactly by linearity, SDs /
  percentiles are exact per row, loss-basis LAE books deterministic
  (all-marginal, no hybrid), and the `total impact` row is a per-statistic
  delta. Variable-rating `xpnl` = the two-group per-atom ledger (scenario
  ladder — one shared source); reinstatement `xpnl` = the 2-D tower
  re-homed. Plain and retro engines have nothing to step through and error.
  Step labels per [Decision-Total-Step-Stays-Total] (base step = engine
  label, fallback `'gross'`). `build_xpnl_stack` retired
  (`stack_marginal_pnls` stays — generic public no-joint assembler);
  massive-source `xpnl` out of scope. Kernel: internal **stitched
  construction mode** (`stitched_rows=` gd-backed plan rows, an extension of
  the sweep-backed path; no `+` composition, no `evaluate`).
- **[Decision-Ladder-Column-Names]** — the card `P1` → **`P01`**
  (zero-padded pair with `P99`); the `stats_df` / `scaled_stats_df`
  **scenario** columns `P01…P99` → **`κ01…κ99`** — they are kappas
  (conditional means given the result at its percentile), and the header now
  *is* the [Decision-Kappa-Shared-Source-Rule] flag: marginal ladders (the
  stitched walk, the massive route, `stack_marginal_pnls`) keep plain
  `P01…P99` headers. Direction unchanged ([Decision-Kappa-Outcome-Direction]:
  payoff convention, left tail bad — a pure relabel, no reversal).
- **[Construction-Introspection]** — `PnL.construction_description` (one
  paragraph: route / source / groups) and `PnL.construction_explanation`
  (the full story: engine and clauses, economics resolution, per-row source,
  booking signs, scenario-vs-marginal ladder and why, closing executable
  replay block), recorded by the builders at construction; hand-built kernel
  P&Ls get a generic structural narrative.
- **Docs & tests** — `_pnl.py` module docstring gains the pnl-vs-xpnl
  ("position vs walk") and kappa shared-source paragraphs; the reins pnl
  suites migrated to consolidated + walk asserts; new
  `tests/test_pnl_consolidated_walk.py` pins the motivating `Cat` acceptance
  case (card `Consideration = 9000`, walk steps
  `Gross Book1 / Occ Cover / Agg Cover / Total`, rows = engine marginals,
  EX foots, ladder marginal + flagged) plus stitched correctness and the
  construction smoke; `decl-testers.agg` notes updated; `dev/FEATURES.csv`
  re-audited (`construction_*` / `economics` rows). Docs pending a manual
  rebuild (standing rule).

## 1.0.0a135

**[Reins-Economics-On-Agg-Ignore-Warn]** — Phase 1 of
`dev/done/plan-pnl-consolidated-xpnl-walk.md`: a pure aggregate ignores what it
cannot use and says so.

- **Plain `agg`s accept every reinsurance decoration** — ceded-premium
  clauses (`deposit` / `rol` / `rate`), ceding commissions (`cede`),
  `reinstatements`, and the variable-rating features (`swing` / `slide` /
  `pc` / `corridor`) no longer hard-error on a bare `agg`. The factory builds
  the loss structure only (byte-identical to the undecorated build) and emits
  **one `IgnoredDecLClauseWarning`** (new, `aggregate.constants`) naming the
  ignored clauses and the remedy.
- **Knowledge injection**: the stored spec keeps the full decorated program,
  so `pnl X <premium> less agg.NAME` re-injects and resolves the economics
  (`deposit` / `rol` / `cede` verbatim; `rate` against the *pnl's* premium —
  exactly why the bare agg must ignore it). Build the agg standalone, get it
  working, then fold it into a `pnl` / `xpnl` by reference.
- The factory filters a **copy** of the spec (the parsed spec aliases the
  knowledge entry — popping would destroy the knowledge the feature exists to
  retain). New corpus section `Z.` in `decl-testers.agg`; new suite
  `tests/test_agg_ignored_clauses.py`;
  `test_premium_clause_on_plain_agg_errors` and
  `test_standalone_agg_cede_still_errors` flipped raises → warns.

## 1.0.0a134

**[PnL-Punchups-01]** — kappa scenario percentiles, the fixed summary card, and
the three-level tower stats index (`dev/done/plan-pnl-punchups-01.md`; executes
[Kappa-Scenario-Percentiles] / [Stats-Tower-Step-Level] / [Summary-Fixed-Card]
/ [Docs-And-Education] as one bump).

- **[Kappa-Scenario-Percentiles]** — the `PnL.stats_df` /
  `scaled_stats_df` percentile columns are now **scenario states, not per-row
  quantiles**: column `Pq` conditions on the exact grand-result slice
  `result == gd.q(q)` and each cell is the conditional mean
  `E[row | result == x_q]` — the library's kappa function applied to the
  ledger. Columns **foot exactly** (legs → totals → result add down the sheet
  to `x_q`); direction is uniform (small `p` is bad for the holder — the
  motivating retro bug, a premium-low cell next to a loss-high cell, is an
  impossible state and gone); the grand-result row remains its own marginal
  quantile automatically. `EX / SD / CV / Skew` stay marginal. Non-monotone
  results ("switcheroo") give exact level-set means — documented, not
  engineered away. The **massive one-sweep route keeps marginal ladders**
  (conditioning needs a second sweep — tracked as
  **[Massive-Kappa-Second-Sweep]** in `dev/TODO.md`);
  `stack_marginal_pnls` is marginal by construction.
- **[Stats-Tower-Step-Level]** — multi-group `stats_df` / `scaled_stats_df`
  rows are three-level `(Step, View, Line)` (step = group label, grand block
  under step `'Total'`); the a132 qualified-string lines (`'base total'`,
  `'Net through cover'`, `'total impact'`) became levels:
  `(step, view, 'Total')` / `(step, 'Margin', 'Net')` /
  `('Total', 'Margin', 'Impact')`. Single-group sheets stay two-level
  `(View, Line)`.
- **[Summary-Fixed-Card]** — **breaking reshape**: `PnL.summary_df` is now the
  fixed headline card, not the ledger dump. Single-group: a flat three-row
  `Consideration / Obligation / Margin` card (mirror of the flat
  `Aggregate.summary_df`); tower: one `(Step, View)` block per step + a
  closing `Total` block with `Net` / `Impact` rows (mirror of the per-unit
  `Portfolio.summary_df` blocks). Rows scale with steps, never legs. Columns
  unchanged (`EX / Scaled / SD / CV / Skew / P1 / Median / P99`); the card
  percentiles are **marginal** quantiles of each row's own distribution
  ([Decision-Card-Percentiles-Stay-Marginal]) and deliberately do **not**
  foot — the footing sheet is `stats_df`. With scale = premium the `Scaled`
  column reads as a combined-ratio decomposition. Leg-level detail moved to
  `stats_df` (tests migrated wholesale). On the **massive route** the card is
  complete with no extra sweep keys: a side total is the `group_total` row
  (>1 leg), the leg row itself (1 leg), or a constant zero (0 legs) — the
  single-group grand-reference `SimpleNamespace` stubs are gone.
- **[Docs-And-Education]** — "Reading the P&L sheets" section in the
  `aggregate._pnl` module docstring (rendered via the Internal Architecture
  autodoc page): card vs sheet (range vs alignment), why marginal percentiles
  don't add, the switcheroo caveat, the serve-time view-rename recipe
  (`df.rename({'Obligation': 'Loss & LAE'}, level='View')` — view labels
  stay defaults-only per [Decision-View-Labels-Scope]). No DecL / grammar /
  snapshot changes.

## 1.0.0a133

**[Label-Canonical]** — one human-facing `label`, no `display_` twins. The
`LabeledMixin` surface collapsed its two near-synonym public names onto a single
resolved `label` property (`_label` → derived default → `name`, never blank);
the stored slot is now the private `_label`. The old public
`display_name` (resolved property) and `display_label` (stored attribute /
constructor kwarg / spec key) are **gone** — every host constructor
(`Aggregate`, `Portfolio`, `Severity`, `PnL`, every `Distortion*`) now takes
`label=` instead of `display_label=`, and reads back through `.label`.

- **Grammar:** the wrapper rule `display_label` → `as_label` (aliases
  `as_label_some` / `as_label_none`); the inner `label: ID` text-capture rule is
  unchanged. The DecL `as "…"` clause is untouched at the source level.
- **Spec keys:** `display_label` → `label`, `engine_display_label` →
  `engine_label`; the transient parser-internal `_premium_label` → `_label`
  (the emitted `consideration_label` key is unchanged).
- **Copula folded onto the mixin:** `copula.Copula` / `CopulaShuffle` now
  subclass `LabeledMixin` (dropping their hand-rolled `display_name`
  empty-string sentinel); their kind handle is exposed as `name` and a
  kind-based `_label_default()` keeps `label` non-blank.
- **Carve-out:** `_pnl.Leg` / `Group` keep their single `self.label` string
  (handle *and* label in one — no name/label split to model); left untouched.
- Breaking rename, pre-1.0 (acceptable). Docs `ref_include.rst` pending a
  grammar regen.

## 1.0.0a132

**[Labels-Reins-Into-Namespace]** — the per-layer cession labels join the
`labels` namespace, making it the complete interior-label surface. The
reins-clause `as` labels now pool into `label_map` as sparse
`{layer_index: label}` dicts — read `a.labels.occ_reins[0]` /
`a.labels.agg_reins` (the shape `_LabelView` always documented) — and the
parallel `occ_reins_label` / `agg_reins_label` **attributes are gone** (one
home, no synonyms). The DecL spec keys and the unparser round-trip are
unchanged; the P&L builders read the namespace. Object-level labels stay on
`display_label` / `display_name` (each node's label lives on the node; only
classless clause sites pool into the owning object's map — the deliberate
tree rule, reaffirmed).

**[PnL-Stats-View-MultiIndex]** — `PnL.stats_df` / `scaled_stats_df` rows now
carry a two-level `(View, Line)` MultiIndex: `View` groups the sheet into
`Consideration` / `Obligation` / `Margin` (result rows, running nets, total
impact); `Line` is the presentation label — leg labels as declared, total rows
read `Total` (qualified `'<group> total'` per group on a multi-group sheet),
group results the group label, the grand result `('Margin', 'Total')`.
Presentation only: the flat ledger labels stay the canonical row keys on
`summary_df` / `density_df` / `validation_df` and the sweep results. No
version bump (author call).

## 1.0.0a131

**[PnL-Generic-Final] phase [Xpnl-Onion-2D]** — the massive-source route and
the 2-D closeout.

- A `MassiveBivariateDistribution` source now evaluates the **whole ledger in
  one pushforward band sweep**: every leg needs an explicit `bs > 0`, each
  derived row (totals, group results, running nets, grand rows) is pushed as
  its own signed-sum function — never a sum of bucketed legs — so
  `mean(result) == sum(signed leg means)` exactly; exact means / sds come
  back from the sweep's streamed audit and feed `validation_df` (every leg
  audited). The ledger row template is now a single shared plan
  (`_ledger_plan`) consumed by both evaluation routes, so they cannot drift.
- The `xpnl` marginal stack, the reinstatement / subsequent-agg joint
  ledgers, and the 2-D guards (an `is2d` leg over a 1-D source errors; a
  second occurrence-level feature is unreachable — `[reins-one-clause]` and
  the aggregate-basis-only variable-rating rule reject it at parse /
  validation time) landed in a129–a130 and are covered by tests.

## 1.0.0a130

**[PnL-Generic-Final] phase [Builders-Variable-Features]** — the six variable
features become one-changed-leg ledger builders; the analyses demote to
drill-down objects.

- `build_variable_pnl` (retro / swing / slide / pc / corridor) and
  `build_reinstatement_pnl` in `_pnl_builders.py`: every feature program now
  returns the **cash-flow-parts group ledger** (a feature never changes the
  machinery — it changes one leg's function): retro = a stochastic premium
  leg over the gross density (same ledger shape as a plain book — the
  acceptance pair, now a test); swing = a stochastic cession premium; slide /
  pc = a stochastic commission leg on the cession; corridor = the adjusted
  recovery; reinstatements = the two-group (three with a subsequent agg
  cover) ledger over the one shared `(L, R)` joint, ceded premium
  `D + h(R)` genuinely stochastic, LAE stochastic `rate·l` off axis 0,
  committed `Scaled` denominator `gross − deposit − pc_agg`.
- **Analyses demoted** (`ReinstatementAnalysis` / `VariableRatingAnalysis`):
  their bespoke `summary_df` / `stats_df` / `distributions` / `gcn_df` /
  `as_pnl` / per-leg `density_df` are gone — the exhibit IS the returned
  PnL's ledger. Kept as domain extras: `terms`, the recovery / ceder maps,
  `validation_df` (reinstatement Est-vs-EX audit), `tail_df`, `plot`, and
  narratives; the exact-moment engine survives privately behind them.
  `qd(analysis)` / `_repr_html_` show the treaty intro + `tail_df`.

## 1.0.0a129

**[PnL-Generic-Final] phases [Kernel-Group-Ledger] + [Builders-Plain-GCN-Port]**
— the P&L kernel rewritten as a **source plus signed group ledger**
(`dev/done/plan-yapnl.md`); insurance semantics moved out to a new
`_pnl_builders.py`. **Breaking (pre-beta, deliberate):**

- New kernel surface: `Leg(label, func, bs=0, is2d=False)`,
  `Group(label, role, consideration, obligation)`, the public
  `PnL(*, name, source, groups=…)` constructor (single-group
  `role=/consideration=/obligation=` sugar; `{label: func}` shorthand;
  `pnl_a + pnl_b` concatenates same-source ledgers), and
  `stack_marginal_pnls` (the no-joint assembler).
- **Signed exhibits throughout**: the group `role` books each leg
  (`sell` → `+consideration, −obligation`; `buy` → the contra), so the `EX`
  column adds down the sheet and every row's distribution sits on the signed
  values (percentile orientation automatic). Rows = the ledger, columns =
  metrics: `summary_df` (headline), `stats_df` (full ladder, currency),
  `scaled_stats_df` (the stats of `X / scale`; the `_UNSCALABLE_STATS` NaN
  machinery is gone), `density_df` (`{row: GridDistribution}`), and
  `validation_df` (Est-vs-EX audit for `bs>0` legs — exact irregular
  distributions are the default, `bs>0` rebuckets via the shared pushforward
  machinery).
- **Retired**: `create_pnl`, `create_pnl_tower`, `PnLTower`, `PnL.margin_df`,
  `PnL.tower`, `PnL.stochastic_engine` (now `PnL.source`), the
  `make_pnl(net=…)` override (net is a result, never a construction
  primitive), and the old stats-as-rows `stats_df` orientation.
- **A reinsurance `pnl` now returns the group ledger**: an aggregate-only
  cession is a real `buy` group over the gross marginal (rows
  `ceded agg premium / recovery / result`, running net, grand rows); an
  occurrence guaranteed-cost program books over the net-of-occ marginal with
  the occ economics as constant legs. Resolved economics ride as
  `pnl.economics`. **`xpnl` returns the marginal perspective stack** as a
  perspectives × stats DataFrame (was a `PnLTower`).
- Label plumbing fixed: the engine `as` label now names the **loss leg**
  (was silently discarded), expense-group `as` labels name one obligation leg
  per group in every path, reins-layer `as` labels name cession groups /
  legs. Port-sourced P&Ls support expense clauses (un-NYI).
- The reinstatement / variable-rating analyses build their faces and
  waterfalls on the new kernel (`.analysis` is now a plain attribute); their
  bespoke exhibits are unchanged this version and demote next
  ([Builders-Variable-Features]).

## 1.0.0a128

**[DecL-Labels-Everywhere]** — broaden DecL `as` labels to interior sub-object
sites, route Portfolio exhibits through them, and give every labelable class one
shared label surface. Follow-on to the a124 `[DecL-Labels]` object-level pass.
Pure presentation — **no computed value changes**; labels land in attributes,
dict keys, and rendered text, never in the FFT. (Phases 0–4 of
`dev/done/plan-labels.md`; the S4/S5 mixture-component and frequency labels stay
deferred.)

- **One `LabeledMixin`** (`_labeled.py`, the repo's first mixin) now carries the
  whole label surface — `display_label`, the resolved `display_name` property, the
  interior `label_map` + read-only `labels` namespace view, the `renamer` property,
  and the per-object `use_labels` switch (default `True`; the setter invalidates the
  cached renamer). Mixed into `Aggregate`, `Portfolio`, `PnL`, `Severity`, and
  `Distortion` via an explicit `self._init_labels(...)` (no cooperative `super()` —
  `Severity` sits on a heavy scipy base). The four a124 classes lost their
  duplicated `display_label`/`display_name`/`_title_name` copies.
- **Interior labels** — the `as "..."` clause now names three sub-object sites
  inside an `agg` body, gathered into `Aggregate.label_map` and read back via the
  `labels` namespace (`a.labels.exposure`, `a.labels.layer`, `a.labels.severity`):
  - **exposure** (S1) — `10000 premium as "GWP 2026" at 0.65 lr` (mid-clause, right
    after the amount; end-clause for bare `claims` / `loss`).
  - **inline severity clause** (S3) — `sev lognorm 100 cv 2 as "ISO ME B"`.
  - **occurrence layer** (S2) — `1000 xs 500 as "Working Layer"` (guarded by an
    explicit ambiguity test vs. the following severity clause).
  Round-trips through `decl_writer`. Bivariate / clash reuse the shared productions
  but are out of scope, so their temp label keys are stripped.
- **Portfolio exhibits route through labels (Phase 4).** Each object now exposes a
  `renamer` (`{handle: display}`) and a `use_labels` switch (default on); the mixin
  applies the renamer as a final `df.rename()` on a **copy** at the display
  boundary, so canonical (handle-keyed) frames and any compute/join that keys off
  the handle are untouched (D2). For a **Portfolio** the axis is its member units
  (+ `total`), each mapped to its member Aggregate's `display_name`; a labeled unit
  shows its label in `summary_df`, `unit_density_df` /
  `aligned_unit_density_df`, the `analyze_distortion(s)` pricing frames, and the
  `plot` legend, while an **unlabeled portfolio is unchanged** (identity renamer).
  Ordering always keys off the handle — the label rides in via the order-preserving
  `rename`, so relabeling can never reorder an exhibit (D4). `Portfolio.unit_renamer`
  is now a thin alias for `renamer`; its old name-guessing heuristic (`.`/`:`
  title-casing, `X1` → TeX subscripting) is **removed** (it had no internal callers).
- **Distortion naming realigned (D6, breaking for direct attribute users).** The
  old inverted scheme (`name` property returning the label, a mandatory
  `display_name` **attribute** holding it) is gone. Now `name` = the kind handle
  (`'ph'`), `display_label` = the optional explicit label, and `display_name` =
  the resolved property (`display_label` → auto-pretty `'PH(0.9)'` derived default
  → handle). `str`/`repr` show the pretty form. The ~20 factory shortcuts no longer
  pass `display_name=`; the pretty strings moved into per-subclass
  `_display_default()`. Construction keyword `display_name=` → `display_label=`
  across `Distortion` and the `approx_ccoc` / `convex_distortion` / `bagged_distortion`
  helpers. Exhibit/plot/cache-key readers of a distortion's old `.name` were
  repointed at `.display_name` (notably the augmented-frame cache key, so
  `TVaR(0.9)` and `TVaR(0.99)` stay distinct). Directly-constructed distortions now
  get the auto-pretty label too, so a bare `PHDistortion(a=0.7)` and
  `Distortion.ph(0.7)` share an `id()` (previously differed).

## 1.0.0a127

**[Test-Speed-SOP]** — parallel-by-default test suite and a fast local loop.
No library behavior change; test infrastructure and developer docs only.

- **`pytest-xdist`** added to the `dev` extra; `addopts` now carries `-n auto`,
  so `uv run pytest` fans the (FFT/numpy-bound) suite across all cores. Disable
  with `-n0` for `pdb`/deterministic ordering.
- **`slow` marker** registered and applied at module level to the three
  bleeding-edge bivariate suites (`test_bivariate.py`, `test_massive_bivariate.py`,
  `test_reins_bivariate.py`) — the 159 heaviest cases, whose two multi-minute
  monsters set the whole suite's wall-clock floor. `addopts` defaults to
  `-m 'not slow'`, so the everyday loop skips them and stays fast; run the full
  gate/CI suite with `uv run pytest -m 'slow or not slow'`.
- **CLAUDE.md** gains a "running the suite efficiently" SOP (parallel default,
  `--lf`/`-k`/`-x` edit loop, full run at the gate, the `slow`-marker recipe) and
  a corrected sync rule: sync a dev checkout with **`uv sync --all-extras`**, never
  a single `--extra` — `uv sync` is exact, so a subset prunes the other extras'
  packages (`uv sync --extra dev` deletes the `massive`/`viz`/`numba` stack and
  breaks the bivariate suites).
- **`numba` is now a declared opt-in extra** (`pip install aggregate[numba]`),
  joining the existing `massive` (`zarr`) and `viz` (`holoviews`/`datashader`/
  `bokeh`) extras — the numba-compiled TVaR / biTVaR paths in `utilities.py` are
  optional (pure-numpy fallbacks exist). Installed via `uv sync --all-extras`.

## 1.0.0a126

**[Massive-Bivariate]** — massive (disk-backed) bivariate distributions:
out-of-core update, dict pushforwards, exploration-grade visualization.
Executes `dev/done/plan-bv.md` (techniques proven in the bigbiv exploration
repo). A `bv` now updates at `log2 = (14, 14)` and beyond with the joint
density living **on disk only** (zarr) and RAM bounded by a band.

- **`update(store_dir=...)`** on `BivariateAggregate` routes all three modes
  (copula, discrete `dbvsev`, netceded) through the new out-of-core kernel
  `_aggregate_compute_massive.massive_bivariate_convolution` — a 3-pass
  zarr corner-turn (row-FFT → column-band FFT+pgf → inverse row-FFT with the
  `i0`/`j0` window relabel folded in at write time). `store_dir=None` is the
  in-core path, untouched. Massive default `padding=0` (the measured window
  is the guard; `deficit` reports any wrap). Sparse severities where dense
  would not fit: the netceded comonotone scatter goes to scipy.sparse
  (`scatter_bivariate_sparse`), the copula rectangle mass is trimmed to the
  per-event support, the `dbvsev` lattice scatters sparse. The `_size_axes`
  *measurement* grid is capped at 2^20 on the massive path only.
- **`MassiveBivariateDistribution`** — the disk-backed sibling of
  `BivariateDistribution`: lazy `.density` zarr view, exact accumulator
  marginals/moments/corr (no disk read), `marginal(i)`, `slice(x=/y=)`
  conditionals, streamed `transformed_moments`, and **`reopen(store_dir)`**
  (the store is self-describing — no FFT re-run in a later session).
  `marginals` / `moments` / `corr` / `deficit` / `info` / `summary_df` on the
  parent bv all work from the pass-3 accumulators.
- **Dict pushforward** — `pushforward({key: f(x, y) or constant}, bs,
  bs_total=, windows=)` streams every function *and their pre-bucket
  **total*** in ONE pass over the density; a single function skips the
  total; constants short-circuit to point masses before the sweep; a per-key
  Est-vs-EX audit (`.pushforward_audit_df`) folds in the same sweep with
  `mean(total) == Σ mean(f_i)` exact. Also available on the in-core
  `BivariateDistribution.pushforward` (dict form) for API uniformity.
- **Visualization** — a sum/max/min decimation pyramid built during pass 3
  (`pyramid.zarr`); Tier-1 matplotlib exhibit
  (`plots/_bivariate_massive.py`): constant-cost pan/zoom (only the pyramid
  level matching the pixel budget is read), log10 color by default,
  max-channel luminance boost so sub-pixel ridges/atoms glow, axis atom
  strips + origin badge, exact marginal panels, decade and joint-exceedance
  contours, `plot_slice`. Tier-2 **`explore()`**: a holoviews + datashader +
  bokeh app (pan/zoom re-aggregation, channel toggle, hover readout, linked
  marginals, click-to-slice) behind the new `aggregate[viz]` extra.
- **Packaging** — new optional extras: `aggregate[massive]` (`zarr>=3`) and
  `aggregate[viz]` (datashader, holoviews, bokeh). Lazy imports with
  actionable errors; no dask (single-process banded loops + threaded scipy
  FFTs, per the measured bigbiv comparison).
- Measured at `(14, 14)` (16384², dev machine): update ~142 s wall,
  deficit 2e-8, density ~0.8 GB compressed on disk, 2-function pushforward
  sweep ~43 s, `reopen` 0.02 s, full-view and deep-zoom plots 0.7 s / 0.35 s,
  peak RSS 2.4 GB.

## 1.0.0a125

**[PnL-Engine-Source]** — a P&L wraps a **complete stochastic engine** (no more
"half-baked agg inside a pnl"). Executes `dev/done/plan-pnl-engine-source.md`.
**Breaking `pnl` syntax.** Also subsumes the `[Portfolio-of-PnL]` TODO: point a
`pnl` at a `port` rather than build a bespoke `PortPnL`.

- **New `pnl` grammar.** `pnl NAME <premium> less <engine> [less <expenses>]`,
  where `<engine>` is a complete aggregate: an inline `agg NAME <body>` (any agg
  form — full / dfreq / tweedie / rename), a stored `agg.NAME`, or a stored
  `port.NAME`. The old forked body (`pnl NAME <P> premium less <lr> lr sev …`) and
  its bare-`lr` `pnl_exposures` / `_pnl_lr` machinery are **removed**. A second
  `less` is the hard anchor introducing the (optional) expense clause, so the
  embedded engine's tail can never cross into the expenses.
- **Shared `agg_body`.** The aggregate body is factored out of `agg_out` and
  reused verbatim by the embedded engine, so a `pnl` engine is byte-for-byte the
  same aggregate a standalone `agg` builds (parse/shape snapshot unchanged).
- **`inherit premium`.** A new premium head copying the engine's technical
  premium — `Aggregate.exp_premium` (now retained on the instance) or the
  accumulated `Portfolio.exp_premium`. A build error if the engine has none.
- **Two independent premiums, on purpose.** The engine's `premium at lr` is a
  sizing/exposure input; the P&L's premium is the booked consideration. Booking
  `12000` over a book sized at `10000` is rate adequacy, not a bug.
- **`xpnl` (exploded).** Same syntax; returns the Gross/net-occ/net-agg
  `PnLTower` instead of the collapsed `PnL`. Requires a wrapped engine with
  reinsurance economics; `xpnl` over a plain engine, or over a `port`, raises
  `NotImplementedError` (the port total hides its units — nothing to explode).
- **`port.NAME` sourcing + `Portfolio.exp_premium`.** A portfolio now accumulates
  its units' premium (a plain sum, no distribution) and exposes the total-loss
  density as a `pnl` source; a port-sourced P&L is the plain net-net-book case.
- **Economics unchanged.** `deposit` / `cede` / `rol` / `rate` / reinstatements /
  variable rating / `retro` stay in the engine's reins clauses and are resolved by
  the wrapping `pnl`/`xpnl` exactly as before; the standalone-`agg` guard (a bare
  `agg … cede …` still errors) is unchanged. `retro` over a *reinsured* engine
  remains `NotImplementedError`.

Migration: every `pnl X <P> premium less <exposure> <body>` becomes `pnl X <P>
premium less agg X_e <exposure> <body>` (a bare `<lr> lr` exposure becomes `<P>
premium at <lr> lr`); a trailing expense clause moves after a second `less`. All
shipped `.agg` databases and the test suite were swept.

## 1.0.0a124

**[DecL-Labels]** — human display labels, quoted names, and expense grouping.
Additive and presentation-only: no computed value changes, labels land in dict
keys, column headers, and repr / exhibit titles. Executes
`dev/done/plan-decl-labels.md`. (Companion to the pending `[PnL-Engine-Source]`
refactor; landed first so that plan's program sweep is written once against final
syntax.)

- **`as` display-label clause.** A new reserved word `as` introduces an optional
  human label on `agg` / `pnl` / `sev` / `port` objects — a bareword skips the
  quote tax (`as lae`) or a `STRING` carries spaces (`as "Gross Book P&L"`). The
  bareword `name` stays the identity / reference handle; the label is presentation
  only, surfaced as the new `display_label` attribute and preferred over `name` by
  the new `display_name` property and by repr / exhibit titles (`label (name)`).
- **`STRING` terminal.** A fresh quoted-string lexical class `/"[^"\n]*"/` (no
  embedded newlines, no escapes in v1); DecL was quote-free, so it cannot collide
  with keywords / `ID` / numbers.
- **Premium label** — `<n> premium as "GWP"` names the consideration leg (the
  P&L's consideration dict key / `summary_df` row).
- **Expense grouping (two-level).** `and`-joined expense terms **combine** into one
  reported obligation leg; **juxtaposed** groups (no `and`) stay **separate** legs
  — the same combine-vs-separate distinction the rest of DecL draws. Each group
  takes an optional `as` label; the default leg name is the basis (`premium
  expense` / `loss expense` / `fixed expense`) or `expense` for a mixed / lone
  group (backward compatible: today's `and`-joined expenses remain one `expense`
  leg). `_resolve_expense_split` now returns one `(name, scalar, loss_rate)` per
  group; `make_pnl` builds one leg per group.
- **Reinsurance-cession label** — `... 100 xs 200 deposit 50 as "Cat XL"` renames
  that basis's **cession** column in `PnL.margin_df` (the Gross/Ceded/Net
  waterfall); the `net` columns keep their structural names.
- **Round-trip.** The DecL unparser (`decl_writer`) renders every label form, so
  labeled programs survive `spec → DecL → spec`.
- **Deferred:** per-component labels inside a *mixture* severity (would need
  invasive changes to the severity mini-language spec/engine for marginal value;
  the object-level `sev` label is delivered). Tracked in `dev/TODO.md`.

## 1.0.0a123

**[PnL-Exhibits]** — the `PnL` value object becomes **generic and
self-describing**: fixed-shape exhibits driven only by the constructor args, and
**`build('pnl …')` always returns a `PnL`** (never a `PnLTower` /
`ReinstatementAnalysis` / `VariableRatingAnalysis` — domain-specific return types
were removed). Executes `dev/done/plan-pnl-exhibits.md`.

- **Always-`PnL` routing.** Every `pnl` program returns the **net** `PnL`. The
  Gross/Ceded/Net waterfall / reinstatement / variable-rating machinery is now
  **attached**, not returned: `pnl.tower` (a `PnLTower`, for a plain cession) or
  `pnl.analysis` (a `ReinstatementAnalysis` / `VariableRatingAnalysis`, the
  drill-down home for the treaty maps, `validation_df`, `tail_df`, `plot`).
  **Breaking:** `build('pnl … reinstatements …')` / `… swing …` / `… deposit …`
  now yield a `PnL`; reach the old object via `.analysis` / `.tower`.
- **New `PnL.margin_df`** — the Gross/Ceded/Net **margin waterfall** (stat rows ×
  `gross`/`ceded`/`net`/benefit columns), forwarded from the attached
  tower/analysis. A plain P&L has no cession, so it raises. Replaces the old
  top-level `gcn_df` on a P&L (which no longer exists — `gcn_df` is domain-specific).
- **`summary_df` fixed shape.** Columns are always `EX / Scaled / SD / CV / Skew /
  P1 / Median / P99` (renamed `% Consid` → **`Scaled`**, `P01/P99` → `P1/P99`);
  rows vary only with the number of legs. A **constant** leg (a fixed premium) now
  reports `SD = 0`, `CV = 0`, `Skew = NaN` exactly — probabilities are renormalized
  to sum to 1, so a clipped source tail no longer leaks spurious variance; residual
  float dust is snapped (`moments._snap_noise`).
- **`stats_df` is now a property** (was a method `stats_df(scale=…)`). One fixed
  layout: stat rows (`EX/SD/CV/Skew` + `P1…P99`) × `(leg, {value, Scaled})`
  columns. The **scale is committed at construction** (`create_pnl(scale=…)`;
  default = expected total consideration; the reinstatement net uses
  `gross − deposit`). `CV` / `Skew` do not scale → their `Scaled` cells are `NaN`;
  CV is guarded to `NaN` at a break-even (near-zero) mean.
- **`PnL.stochastic_engine`** — a P&L now retains an opaque reference to the source
  it was built over (an `Aggregate`, `GridDistribution`, …) for drill-down, though
  it never *depends* on it (the exhibits are read off the leg value arrays).
- **`loss`-basis expense is stochastic** in the plain path: `20% loss expenses` is
  now `0.20 × actual loss` per atom (LAE scales with loss), not a point mass at
  `0.20 × E[loss]`. `fixed` / `premium` terms stay deterministic. (The GCN
  commission split still uses the scalar `E[loss]` form.)

## 1.0.0a122

**[PnL-API]** — a domain-agnostic **`create_pnl`** and a reshaped, engine-free
`PnL` value object. A P&L is *money in minus money out* over a random state:
group each component map's values over the source atoms by output value and sum
probability, and you get three **exact** `GridDistribution` legs (consideration,
obligation, result). Executes `dev/done/plan-pnl-api.md`; **supersedes** the a121 leg
kernel (`legs.py` / `_insurance_view.py`), which are removed. Insurance becomes
one *caller* of the general API, not a special case.

- **`create_pnl(source, *, consideration, obligation, role='sell', …)` (new,
  public).** The general entry. `source` is an opaque slot — a
  `GridDistribution`, an `Aggregate`, a `BivariateDistribution`, or a bare
  `(values, probs)` pair; the component maps decide what it means. Magnitudes are
  non-negative and the **`role`** (`'sell'`/`'buy'`) supplies the sign, so the
  cession sign-flip falls out of the role, never a hand edit. Labels are **data**
  (the dict keys name the legs) — no built-in perspective/category taxonomy.
- **`PnL` reshaped into a lightweight value object.** It **consumes and discards**
  its stochastic engine: no `.agg`, no `update()`, no `validation_df` (the legs
  are exact group-bys — nothing of its own to validate). The four FCC reports:
  `summary_df` (headline, columns `EX / % Consid / SD / CV / Skew / P01 / Median /
  P99`), `stats_df(scale=)` (detailed stats×legs, each paired with a
  `%-of-scale`; a stochastic scale warns), `density_df` (an ordered
  `{leg: GridDistribution}`, no lossy shared staple), and `plot` (net density +
  CDF). `q` / `cdf` / `sf` / `evaluate` / `prob_loss` kept. **Breaking:** the old
  `pnl_df`, `.agg`, `update()`, `value_type`, and the fixed
  Consideration/Obligation/Expense/Margin `summary_df` are gone.
- **`create_pnl_tower` / `PnLTower` (new).** Stacks legs into the inuring
  waterfall — each leg, the running **net** after it, each one-step **benefit**
  delta, and a total benefit — as `gcn_df` (stats rows × waterfall columns; means
  add, SDs don't). The plain Gross/Ceded/Net view is a tower of per-perspective
  legs (`gcn_tower_from_aggregate`), with the resolved `deposit`/`rol`/`rate`/
  `cede` economics on `.economics`.
- **Insurance rewired onto the tower.** `ReinstatementAnalysis` and
  `VariableRatingAnalysis` are now `create_pnl` tower builders over their joint /
  degenerate sources; `build('pnl …')` returns the analysis (reinstatement /
  swing / slide / pc / corridor / retro) or a `PnLTower` (Gross/Ceded/Net)
  directly. Their `summary_df` / `tail_df` / `validation_df` keep their shapes;
  `VariableRatingAnalysis` **gains** `summary_df` / `tail_df`. `gcn_df` adopts the
  new-canonical stats×waterfall schema (the legacy `Mean`/`Ratio`/`Volatility`/
  `UW %ile` multi-section exhibit is retired).
- **Removed:** `aggregate.legs` (`Leg` / `LegSet` / `GraphSource`),
  `aggregate._insurance_view` (`InsuranceView` and the perspective/category
  vocabulary), and `gcn_assemble_column` / the hand-rolled `_gcn_*`. The exact
  reinstatement / variable audit is preserved via the *kept* pushforward
  primitives (`BivariateDistribution.pushforward` / `pushforward_1d`), not the
  removed View. `GridDistribution` gains `to_series()`.
- **DecL: 0 changes.** The grammar stays insurance-only; the cross-domain vehicle
  is the Python `create_pnl` API.

## 1.0.0a121

P4 — the **bivariate leg kernel**: a domain-free engine for the P&L leg model,
with insurance as a View on top. Collapses the ad-hoc leg machinery that
`VariableRatingAnalysis` and `ReinstatementAnalysis` each carried onto one
shared core, so "1-D vs 2-D" is now just a source swap and a feature only fills
a leg. Executes `dev/done/plan-bivariate-legs.md` (moved to `dev/done/`); behavior is
preserved except the intended `summary_df` no-morph change.

- **`aggregate.legs` (new, domain-free kernel).** A `Leg` is one cash-flow
  stream — a vectorized signed map `f(X, Y)` of the two coordinates of a
  bivariate law, pushed forward onto its own `GridDistribution`. No insurance
  vocabulary; signs live in the maps, so cash-flow algebra (`net = X − Y`, a
  margin) is plain summation (`Leg.__add__` / `__sub__` / `__neg__` /
  `Leg.combine`). `LegSet` evaluates an ordered, named set over a **source**:
  `distributions(source)` (per-leg pushforwards) and `stats_df(source)` (exact
  per-leg moments — the "EX" column, means add with no rebucketing).
- **Source protocol + `GraphSource`.** A source is anything exposing
  `pushforward(fn, …)` + `transformed_moments(fn, …)`; `BivariateDistribution`
  already satisfies it (the full 2-D path). The new `GraphSource` is the
  *degenerate* case — all mass on the curve `Y = κ(X)` over a 1-D grid — taking
  the cheap `pushforward_1d` fast path. Dimensionality lives in the source, not
  the leg; every leg keeps the universal `f(X, Y)` signature.
- **`aggregate._insurance_view` (new).** The insurance vocabulary, as a
  composition over a `LegSet` (never a mixin on `Leg`): the
  `name → (perspective, category)` label map and the `category → kind` rollup
  (`premium → consideration`; `loss`/`expense → obligation`;
  `underwriting → margin`), plus an `InsuranceView` that owns the one cached
  kernel evaluation (with fixed legs kept as exact point masses).
- **`VariableRatingAnalysis` and `ReinstatementAnalysis` rebuilt on the kernel.**
  Both now assemble a `LegSet` and read `distributions` / `stats_df` from an
  `InsuranceView` (VR over a `GraphSource` with `κ =` the base ceder; the
  reinstatement analysis over the `(L, R)` joint with `gross_premium` as a point
  mass). The GCN / summary / tail / validation exhibits and all bespoke extras
  are unchanged; the full variable-rating and reinstatement suites stay green.
- **`PnL.summary_df` no longer morphs into the GCN waterfall.** It is *always*
  the fixed small `Consideration / Obligation / Expense / Margin` table, even for
  a Gross/Ceded/Net position (`dev/reporting-guidelines.md` guideline 1). The GCN
  waterfall remains the separate `PnL.gcn_df` exhibit (and
  `pnl.reinstatement_analysis.summary_df` for the richer headline). **Breaking
  for callers that read `gcn`-shaped columns off `summary_df`** — read `gcn_df`.
- **Tests.** New `tests/test_legs.py` (synthetic-source kernel: degenerate-vs-full
  agreement, many-to-one binning, leg algebra, net as a fresh pushforward) and
  `tests/test_insurance_view.py` (label vocabulary + the cached composition).
- **Docs pending a rebuild:** the §3 "how a leg is computed" material graduates
  to a bivariate-leg-model docs page (generic `X` / `Y` / `κ` notation); not
  built in the iteration loop.

## 1.0.0a120

Phase 3 variable rating — **retro** (the fifth feature) gets its DecL surface, so
all five now run through `build('...')`.

- **`retro` rating clause** in the `pnl` premium head: `retro <collar> premium`
  substitutes for the fixed `<num> premium`, varying the **gross** premium as a
  collared affine map of net account loss. The grammar factors the head into
  `pnl_premium: numbers PREMIUM | RETRO collar PREMIUM`, so the trailing
  `premium` keeps the `premium less` divider and reuses the existing head
  structure. The collar is keyword-first (`basic <b> lcm <m> [min <lo>] [max
  <hi>]`), matching `swing`:

  ```
  pnl R retro basic 3000 lcm 1.1 min 3500 max 8000 premium less 1000 loss sev lognorm 100 cv 2 poisson
  ```
- Underwriter builds `RetroTerms` and a `VariableRatingAnalysis` over the gross
  density (`variable_layer=None`); `PnL.gcn_df` delegates as for the other
  features. Retro currently requires **no inuring reinsurance** (the clean 1-D
  net-account-loss = gross-loss case); retro + reinsurance is a follow-up and
  raises a clear error. Unparser renders the retro head (shared `_render_collar`,
  also now used by `swing`).
- Tests: retro cases in `tests/test_variable_rating_decl.py`; `VR.Retro` line in
  `decl-testers.agg`. Design note: `swing` deliberately takes **no** trailing
  `premium` (it decorates a reins layer, with no `less` to pair with).

## 1.0.0a119

Phase 3 variable rating — the `[analysis]` engine and the `[decl]` surface for the
four aggregate-basis features (`dev/done/plan-variable-rating.md`). The four now run
end-to-end through `build('...')`; retro stays programmatic pending a rating-clause
syntax decision, and occurrence-basis (2-D) variable rating is a follow-up.

- **`VariableRatingAnalysis`** (`aggregate.variable_rating`, submodule access only)
  — a feature-agnostic Gross / Ceded / Net engine driven by any `ContractTerms`.
  The feature's `target_leg` selects the one stochastic leg (gross premium / ceded
  premium / expense / ceded loss); every leg is written over `(l, a)` and pushed
  forward over the gross density via `pushforward_1d` (the 1-D aggregate basis,
  decision 0: one variable feature per program, no stacking). `gcn_df` reuses the
  shared `gcn_assemble_column`; means add `gross + ceded = net`. Takes raw arrays,
  so it is unit-testable independent of `Aggregate`.
- **DecL surface** for the four reins-layer features, each decorating one
  `aggregate net of` layer: `swing basic <b> lcm <m> [min <lo>] [max <hi>]`
  (replaces deposit/rol/rate), `slide <c> at <lr> and …` (replaces `cede`),
  `pc <share> after <allowance>`, `corridor <share> po <width> xs <attach>`. New
  grammar terminals + transformer rules emit the locked spec keys
  `agg_reins_swing` / `_slide` / `_pc` / `_corridor`; per-layer validation enforces
  the swing/premium and slide/cede exclusions, the LR-denominator requirement, and
  decision 0 (at most one feature; aggregate basis only this release).
- **`Aggregate.variable_rating_analysis()`** builds the analysis from the gross
  density + the decorated layer; **`PnL.gcn_df` delegates** to it when the
  aggregate carries a DecL variable feature (mirrors the reinstatement path).
- **Grammar-sync mirrors** (`decl_pygments.AggLexer`, `parser_errors._TERMINAL_LABELS`)
  updated for the new keywords — and for the Phase 2 `reinstatements` / `free` /
  `no` keywords that had drifted (TODO D10), so `test_grammar_sync` is green.
- Tests: `tests/test_variable_rating_analysis.py` (hand-checked GCN identities +
  Monte-Carlo), `tests/test_variable_rating_decl.py` (build-through + negative
  cases); DecL lines added to `src/aggregate/agg/decl-testers.agg`.

## 1.0.0a118

Starts **Phase 3 (variable rating)** with the `[terms]` workstream — the pure,
engine-free contract-terms layer (`dev/done/plan-variable-rating.md`). No analysis or
DecL wiring yet; that follows.

- **New `ContractTerms` taxonomy** (`aggregate.contract_terms`, submodule access
  only). A *contract term* is a deterministic vectorized map `phi` of a realized
  loss quantity that fills one P&L leg (the legs model, appendix section 1). The
  thin base owns the `target_leg` / `loss_basis` metadata, the abstract `phi`, and
  the shared finite / nonnegative / monotone validation probe
  `_check_vectorized` (generalized from the reinstatement callable validator).
- **Five frozen-dataclass features**, each with a hand-checked sign-correct worked
  example: `RetroTerms` (gross premium, collared affine in net account loss),
  `SwingTerms` (ceded premium, collared affine in ceded loss — shares the collar
  machinery via `_CollaredAffineTerms`), `SlideTerms` (sliding-scale ceding
  commission, decreasing PWL of ceded LR from `(commission, loss_ratio)` anchors,
  flat outside the ends), `ProfitCommissionTerms` (`share·(1 − LR − allowance)₊`),
  and `CorridorTerms` (loss-ratio band the cedant retains). The ratio features
  (slide / pc / corridor) read the **ceded loss ratio** so `phi` stays a pure
  single-argument map; the analysis layer (next workstream) binds the premium.
- **`ReinstatementTerms` refactored under `ContractTerms`** — sets the metadata,
  exposes `reinstatement_premium` as `phi` (the premium decorator; `recovery`
  remains the annual-cap loss transform, making reinstatement the one two-map
  feature), and routes its callable validator through the shared probe.
  Behavior-preserving: the full reinstatement suite stays green.
- Spec-key naming for the layer-decorating features locked to the shipped
  `{which}_reins_*` convention (`occ_reins_swing` / `agg_reins_swing`, etc.);
  account-level `retro_terms` stays flat (the one exception). DecL / analysis
  wiring is pending.

## 1.0.0a117

Completes **Phase 2 (reinstatement premiums)** — see `1.0.0a116` for the core
feature. This release adds the first-class display surface, the decision-3
subsequent aggregate cover, and deterministic expense / commission flow-through.

- **First-class display surface** for `ReinstatementAnalysis`: `info`,
  `_repr_html_`, `validation_explanation`, `density_df(leg=…)` (per-leg `p`/`F`/`S`
  frame), the reused `bs_window_df` / `bs_description` / `bs_explanation`
  joint-grid sizing audit (delegated to the `BivariateAggregate` holder), `qd`
  support, and a four-panel `plot()` mosaic — joint `(L, R)` log density with the
  recovery breakpoints / cap; the `A(R)` / `h(R)` / `D+h` / `D+h−A` maps; gross
  vs net underwriting return-period; and the cession-impact curve. A
  reinstatement-backed `PnL`'s `plot()` delegates to this mosaic.
- **Subsequent aggregate cover (decision 3).** A reinstated occurrence layer may
  carry a genuine subsequent `aggregate net of …` cover. Its recovery
  `g(L − A(R))` is a deterministic pushforward of the *same* `(L, R)` joint (via
  the net-of-occurrence loss), so `gcn_df` extends to the five-column inuring
  waterfall `gross | ceded occ | net occ | ceded agg | net agg` with the means
  adding tier by tier; `summary_df` collapses to Gross / total-cession / final-net;
  `tail_df` and the plot read the net-of-everything result. The agg ceded premium
  is threaded from the `pnl` layer clause. `R` stays unlimited — the reinstatement
  cap lives only in `ReinstatementTerms`.
- **Deterministic expenses & ceding commissions** now flow through the
  reinstatement waterfall: a reinstatement `pnl` with `… expense` / `cede` books
  the gross expense and the per-tier commission credits in the GCN **Expense**
  section (means add down `Premium + Loss + Expense = UW` and across the split),
  and `summary_df` / `tail_df` / `plot` report the underwriting result net of
  expense — exactly as a plain `pnl` does. The reinstatement premium `h(R)` is
  non-commissionable, so the commission stays deterministic; the *stochastic*
  commission features (`slide` / profit commission) are Phase 3. The leg
  distributions stay pure underwriting — expense is a deterministic presentation
  shift, mirroring `PnL._gcn_perspective_rows`.

## 1.0.0a116

### Property-cat reinstatement premiums (stochastic ceded premium)

Occurrence reinsurance with paid/free **reinstatements**, where the ceded
premium `D + h(R)` is *stochastic* (the reinstatement premium `h(R)` is a
function of the unlimited annual occurrence recovery `R`). One declarative `pnl`
block does gross volume, the cat layer, the reinstatement schedule, and the
gross/ceded/net underwriting exhibit:

```
pnl Cat
    10000 premium less 85% lr
    sev lognorm 50 cv 3
    occurrence net of
        95% po 100 xs 100
            rol 18%
            reinstatements 1 free and 1 at 50% and 2 at 100%
    poisson
```

- **DecL** — a `reinstatements` clause decorates a single occurrence layer. Two
  surface forms: the treaty-language group chain `<count> free and <count> at
  <p>%` (counts as digits **or** the number-words `one`…`five`, which stay
  ordinary identifiers everywhere else) and the explicit list `[a1 ... am]`
  (escape hatch). `free` = `at 0%`; `at p%` is a multiplier of the base rate on
  line `r = base_premium / limit`, where the base premium is the layer's
  `deposit | rol | rate` clause. Three "how much annual cover" cases: **omit**
  the clause = free + unlimited reinstatements (existing behavior); `reinstatements
  …` = m paid/free reinstatements ((m+1)·y annual capacity); and **`no
  reinstatements`** = zero reinstatements, a single annual limit y (m=0, recovery
  capped at y, deterministic ceded premium). New spec key `occ_reins_reinst`
  (rate tuple, or the empty tuple for `no reinstatements`); `decl_writer`
  round-trips all forms.
- **`build()` returns a `PnL`** whose `gcn_df` / `summary_df` are backed by a
  lazily built `ReinstatementAnalysis` (the stochastic ceded premium makes the
  ceded/net **CV Premium** rows nonzero, read off the `(L, R)` pushforward).
  `pnl.reinstatement_analysis` exposes the engine; `Aggregate.reinstatement_analysis()`
  is the arg-free programmatic entry point.
- **Validation (locked)** — `[reins-premium]` (a reinstatements clause needs a
  base premium clause); `[reins-one-clause]` (at most one occurrence layer may
  carry the clause); `[reins-single-layer]` (a reinstated layer ⇒ a single
  occurrence layer — the 2-D `(L, R)` engine ceiling). A subsequent
  `aggregate net of` cover is allowed (it stays a deterministic pushforward over
  the same joint).
- **Engine** (landed earlier this feature): `BivariateDistribution.pushforward`
  / `pushforward_1d` / `transformed_moments` (public); `ReinstatementTerms`
  (recovery `A(R)`, reinstatement premium `h(R)`, annual cap `(m+1)y`);
  `ReinstatementAnalysis` (eleven leg distributions, GCN/summary/validation/tail
  exhibits); the shared `gcn_assemble_column` waterfall builder.

### Fixes

- **`PnL.plot()` dropped the gross / ceded legs for an occurrence-only GCN
  P&L.** The plot read the gross / net-of-occurrence aggregate densities from the
  `agg_density_*` attributes, which only **aggregate** reinsurance populates; for
  an occurrence-only treaty they are `None`, so those legs drew as invisible
  zero lines. The GCN overlay now reads the retained waterfall from
  `reins_density_df` (`Gross → [Net occ] → Net`), positioned by the same
  per-perspective premium / expense the GCN table uses (factored into the shared
  `PnL._gcn_magnitudes`). occ-only → Gross/Net; occ+agg → Gross/Net occ/Net.
- **`PnL.q` / `cdf` / `sf` were hand-rolled and scalar-only** (`pnl.q([.001,
  .99])` raised `TypeError`). They now delegate to a cached net
  `GridDistribution` (`pnl.gd`) — the single home for `q`/`var`/`cdf`/`sf` every
  other class already uses — so they vectorize, gain `var`, and match the
  canonical kernel. (`tvar` is deliberately reached via `pnl.gd.tvar`, not a bare
  `pnl.tvar`: GD's `tvar` is the upper-tail/loss-sense shortfall, wrong-signed for
  a payoff.)

## 1.0.0a115

### Multiple `pnl` expense terms (`and`-joined)

A `pnl` expense clause is now a **list of terms joined by `and`** (the
reinsurance-list precedent), so a book can carry variable **and** fixed expenses
at once:

```
pnl Book 1000 premium less 8 claims sev lognorm 50 cv 1 poisson
    25% premium expense and 1000 fixed expense
```

- `expense_clause` → `expense_list` of `expense_term`s; the terms **sum** into the
  gross expense. `expense_spec` is now a list of `(basis, value)` pairs (a bare
  tuple is still accepted via the Python API). The inter-term `and` sits after all
  reinsurance, so it never competes with a cession `and`.
- `decl_writer` renders the term list (`… and …`); round-trip preserved. No
  change to the GCN exhibit — it reads the summed `_gross_expense()`.
- High-level shape: `pnl LABEL <premium> premium less <loss clause> <expense
  term> [and <expense term> …]`.

## 1.0.0a114

### P&L expenses, ceded premium / ceding commission, and the GCN waterfall exhibit (`dev/done/plan-pnl-expenses-ceded-premium.md`)

Phase 1 of three (Phase 2 reinstatements, Phase 3 variable rating). Adds gross
**expenses**, per-layer **ceded premium**, and **ceding commission** to a `pnl`,
all deterministic, and rebuilds the Gross/Ceded/Net exhibit as a multi-section
reinsurance **waterfall**.

- **`pnl` gross expenses** — `25% premium expenses` / `25% loss expenses` /
  `2000 fixed expenses` (the base is explicit; `expense`/`expenses` both
  accepted; optional, absent ⇒ 0). Books a gross obligation; reduces the margin
  and drives a combined-ratio line. Expense clauses attach to `pnl` only.
- **Ceded premium per reinsurance layer** — `deposit <amount>` | `rol <frac>` |
  `rate <frac>` (mutually exclusive), plus an optional `cede <frac>` commission.
  `deposit` is currency, `rol` = share × rol × limit, `rate` = rate × gross
  premium; the commission books in the **expense** column as a credit, so net
  expense = gross expense − commission. **Any** premium clause promotes a `pnl`
  to the Gross/Ceded/Net view (the gross premium is the stated `pnl` premium).
- **`PnL.gcn_df` rebuilt** as a waterfall-column, multi-section frame: columns
  `gross | ceded occ | net occ | occ impact | ceded agg | net agg | agg impact |
  impact` (inuring occurrence → aggregate; columns shown only for configured
  sides; the `*impact` columns are percent change, percentage-point on ratios);
  row sections **Mean** (Premium/Loss/Expense/UW — signed, additive down to UW
  and across the GCN split), **Ratio** (LR/ER/CR), **Volatility** (SD & Skew of
  LR/CR), and **UW percentiles** (payoff convention; regulatory anchors). Each
  column reads its own exact aggregate marginal from `reins_density_df`; only the
  Mean section adds across columns (means add, SDs don't). The single-leg
  `summary_df` keeps `Consideration | Obligation | Margin`, now with an Expense
  row and a combined-ratio line.
- **`pnl` separator is now `less`, not `-`** (hard switch). `pnl NAME <premium>
  premium less <loss body>` — a dedicated keyword so the P&L split never collides
  with severity arithmetic (`ssev 20 - lognorm`). The `-` separator is removed;
  the corpus and tests are migrated. (Multi-line indented authoring already works
  — newlines/indentation inside a statement are whitespace.)
- **GCN UW percentiles are loss-severity aligned** — the cession columns are
  reversed (report the `1 − p` quantile) so each percentile row reads as one
  scenario direction (worst-loss row: gross deeply negative *and* recovery large).
- **Grammar / unparser** — new terminals `EXPENSES`, `FIXED`, `DEPOSIT`, `ROL`,
  `CEDE`, `LESS` (reusing `RATE`); `reins_clause` refactored into `reins_layer` +
  optional premium + optional `cede` (the loss-structure path is unchanged). Parse
  errors: `cede` without a premium, `rol` without a finite limit, ceded-premium
  clauses on a plain `agg`. `decl_writer` renders all new clauses (round-trip
  preserved).
- New tests `tests/test_pnl_expenses.py`, `tests/test_pnl_ceded_premium.py`;
  corpus section `EXP` in `decl-testers.agg`. Note: when a leg becomes
  loss-sensitive (Phase 3 swing/slide on an XOL) the net-distribution derivation
  swaps the comonotone relabel for the pushforward engine — recorded as a design
  risk, not yet built.

## 1.0.0a113

### User-facing `summary_df` + `tail_df`; QA frames renamed (`dev/done/plan-summary-tail-tables.md`)

Turns the daily-driver display frames from *validation artifacts* into *risk
views* a practitioner reads at a glance, and frees the two best names for them.
**Breaking** on `Aggregate` and `Portfolio` (hard cut, no deprecated aliases).

Name map:

| Name | Now | Was |
|---|---|---|
| `summary_df` | at-a-glance moments + key percentiles (Freq/Sev/Agg) | the moment-error table |
| `tail_df` | return-period / exceedance table (VaR, TVaR, …) | the tail-behavior classifier |
| `validation_df` | the moment-vs-estimate error table | old `summary_df` |
| `tail_behavior_df` | support + per-side tail class + concentration | old `tail_df` |

- **`summary_df`** (property): `E[X] | SD | CV | Skew | p0.01 | p0.50 | p0.99`,
  indexed `Freq` / `Sev` / `Agg`. `SD` and `CV` are both always present (stable
  layout); `CV = SD/E[X]` blanks when `|E[X]|` is ~0 relative to `SD` (signed /
  near-break-even). Percentiles come from the FFT grid (exact, not simulated):
  `Agg` via `q`, `Sev` via `q_sev`; **Freq percentiles are blank by design**
  (frequency is a PGF, never materialized — use `create_frequency()` for the
  count distribution). Moments are analytic, so `Freq × Sev = Agg` mean is exact.
- **`tail_df`** (now a method, `tail_df(periods=…)`): the return-period table
  `p | VaR | TVaR | xsVaR | VaR/Mean`, indexed by return period `T` (default
  ladder `2…1000`, incl. the 1-in-200 / 1-in-250 capital anchors). Aggregate-only;
  the Portfolio form adds a leading `unit` index level plus a `total` block.
  Honors the loss/payoff convention via the new `period_to_p` (inverse of
  `return_period_map`). `None` before `update()`.
- **`validation_df`** / **`tail_behavior_df`**: the old `summary_df` /
  `tail_df` payloads, unchanged, under names that say what they are. The
  validation engine was already reading `stats_df['error']` directly, so the
  rename is display-only (`self.valid` is untouched).
- **`qd` / `_repr_html_`** now lead with `summary_df`, then `tail_df`, and flag
  validation **only on failure** (silent on a pass); both open with the short
  text / HTML intro.
- Internal callers migrated (`_bucket_window`, `Portfolio.bs_explanation` and
  `tail_behavior_df`); all stale `summary_df` / `tail_df` docrefs repointed.
- **Deferred:** `PnL` and `BivariateAggregate` keep their existing
  `summary_df` / `tail_df` (different, already user-facing meanings) — a separate
  follow-up. Row-highlighting of the SII/250 rows in HTML is a pending polish.

## 1.0.0a112

### Lee/quantile worker consumes a `GridDistribution` (return-period revisit)

Completes the return-period plot revisit deferred from a111: the Layer-1 Lee
worker `plot_quantile(ax, gd, quantile_x=…, max_return_period=…)` now takes a
`GridDistribution` instead of `(p, loss, is_loss_value=…)` arrays. The worker
derives the curve from the GD's own `(cumsum(p), x)`, trims the saturating top of
the quantile function itself, and reads orientation off `gd.return_period(p)` —
so the `is_loss_value=` flag is gone from the compositors (`plot_aggregate`,
`plot_severity`, `plot_reins_occ`); each passes a self-describing object instead.

- `Aggregate._sev_grid_distribution()` now carries the **aggregate's** role
  (`is_loss_value=self._is_loss_value`), not an intrinsic loss role: the severity
  curve shares the aggregate's Lee panel and must spread the same tail, so a
  payoff aggregate draws its severity with the payoff convention. Its objective
  consumers (`sev_q`, `sev_tvar`) are sign-agnostic, so only `return_period` is
  affected; the `value_type` setter already resets the `_sev_dist` cache.
- `reins_occ_plot` wraps each gross/ceded/net aggregate PMF as a cheap GD (the
  old hand-rolled survival + 1e-15 de-fuzz is gone; the worker's saturating-top
  trim handles the flat tail). The reins Lee curves now draw as a continuous
  staircase rather than a NaN-broken survival line.
- The continuous-`Aggregate` Lee panel is now trimmed at the saturating top too
  (previously a standing "to do" — only the discrete panel was trimmed).
- A standalone `Severity` (no discrete grid of its own) wraps its plot-grid
  `cdf` as a throwaway GD; the discrete `Aggregate` panel wraps the *anchored*
  `df` so the steps-pre Lee line still rises from the baseline at its own signed
  index (signed-aggregate regression preserved).

No behavior change to the `'linear'` default beyond the continuous-panel trim
and the reins staircase cosmetics. `plot_quantile`'s old array call shape is
removed — direct callers pass a GD (see `tests/test_plot_return_period.py`).

## 1.0.0a111

### `GridDistribution` knows its sign (loss vs payoff)

Orientation — whether `X = 1` means "I pay 1" (**loss**) or "I receive 1"
(**payoff**) — is now an intrinsic, immutable property of a `GridDistribution`
(GD) rather than a side-channel flag threaded through every orientation-aware
call. `GridDistribution(x, p, ..., is_loss_value=True)` and
`GridDistribution.from_series(..., is_loss_value=True)` accept the role, expose
it as the read-only `gd.is_loss_value`, surface it in `__repr__`
(`GridDistribution 'name' (n=…, bs=…, loss|payoff)`), and propagate it through
`gd.cap(a)`. The default is `True`, so sign-agnostic callers (including the
pricing GDs in `bounds.py`) are unaffected.

The **objective kernel is untouched**: `q`, `var`, `tvar`, `tvar_threshold`,
`cdf`, `sf`, `pmf`, `mean`, `lev`, `tvar_of_limited` never read `is_loss_value`
— they describe the random variable and are identical for a loss or a payoff
(existing numeric tests pass byte-for-byte). Orientation is consulted only by
the "which side is bad?" operations.

The loss/payoff return-period map now lives on the GD: `gd.return_period(p)`
returns `T = 1/(1−p)` for a loss and `T = 1/p` for a payoff, delegating to a new
module-level `return_period_map(p, is_loss_value)` so the Lee/quantile plot
worker and the upcoming summary `tail_df` share **one** implementation instead
of each re-deriving the branch. The Lee worker (`plot_quantile`) keeps its
array-based signature but now reads this shared map and caps directly on `T`
(behavior-preserving); the `Aggregate` plot compositors source the panel
orientation from the aggregate's own GD (`agg._grid_distribution().is_loss_value`),
with the severity curve following the aggregate role (the panel convention).

Holders thread their role into the GD they build (`Aggregate` /
`Portfolio._grid_distribution()`), and the `Aggregate.value_type` setter now
resets the GD caches (`_dist`, `_sev_dist`) so a post-build role change rebuilds
with the new orientation. Pricing call sites that already thread
`is_loss_value=` into `effective_g` are left as a defaulted shim for now.

## 1.0.0a110

### Feature: return-period x-axis for quantile (Lee) plots (`quantile_x='return'`)

The Lee/quantile panel can now be drawn against **log return period** instead of
the non-exceedance probability `p`. Pass `quantile_x='return'` (default
`'linear'`, today's `x = p` behavior) to `Aggregate.plot`, `Severity.plot`, or
`Aggregate.reins_occ_plot` — the three plots that own a Lee panel. The transform
branches on the value-type role (`_is_loss_value`): a **loss** maps the
large-`p` tail to large `T` via `T = 1/(1−p)`, a **payoff** maps the small-`p`
tail via `T = 1/p`, so in both cases log-x spreads the rare *bad* tail where it
can be read directly. The axis is log-scaled with decade ticks and labeled
"Return period". Drawing is **capped at `max_return_period` (default 1e9)**: the
saturating endpoint (`p→1` loss / `p→0` payoff, where `T` diverges) is dropped,
and the outcome (y) axis is rescaled to the deepest *plotted* point so it tracks
the cap rather than running off to the saturating tail. Only the Lee panel is
affected — density and distribution panels are unchanged.

The transform and **all** of its axis handling live in one Layer-1 worker
(`plots._quantile.plot_quantile`). The option is **not** named in the Layer-2
compositors or the class `.plot()` stubs — it rides through their `**kwargs`, so
the layered plotting infrastructure stays thin and future Lee-panel options
(e.g. `max_return_period`) flow through without touching `_aggregate.py` /
`_severity.py`. `PnL` and `Portfolio` are unaffected: their `.plot()` exhibits
have no Lee panel.

## 1.0.0a109

### Feature: bare unary minus on a severity (`ssev -lognorm …`)

DecL now accepts a **bare unary minus** in front of a severity as sugar for the
working `0 - X` reflection: `ssev -lognorm 10 cv 0.5` ≡ `ssev 0 - lognorm 10 cv
0.5`. Unary minus binds *tighter* than the additive shift (standard math
precedence), so `-lognorm 2 + 5` parses as `(-X) + 5 == 5 - X`, **not**
`-(X + 5)`. A negative *number* multiplier (`-3 * lognorm`) still lexes as one
`NUMBER` and stays on the existing scaled-reflection path. The new grammar
alternative is `sev1: MINUS sev1 -> sev1_negate` (Earley + dynamic lexer, no
ambiguity).

### Breaking: a reflected severity now requires `ssev`, not `sev`

A reflected severity carries negative support, so it is rejected under the
clamped `sev` clause with a clear message (*"a reflected (signed) severity needs
'ssev', not 'sev'"*). This applies to both the new `-X` form and the existing
`0 - X` (`rsub`) form. Reflection already routed through the signed (no-clamp)
build path regardless of the keyword, so this is a surface-consistency change,
not a numerical one; `decl_writer` now emits `ssev` whenever `sev_reflect` is
set so specs still round-trip. The one corpus line using `sev 100 - lognorm`
(`SBJ.Signed`) is restated as `ssev` (behavior-identical).

## 1.0.0a108

### Fix: point-mass severity blocked the windowed grid (textbook concentration case)

A high-mean, highly concentrated aggregate with a **point-mass severity**
(`dsev [k]`) — e.g. `agg Test 1000000 claims dsev [1] poisson`, the count
distribution `create_frequency` materializes — silently fell back to the coarse
0-based grid instead of the tight two-sided window its `_bs_window_df` had
already computed.

Root cause: a point mass has zero spread, so its skewness is `0/0 = NaN` and
`Aggregate._severity_high_estimate` (the windowed severity-fit guard's input)
returned NaN from the method-of-moments fit. The guard's `np.isfinite(sev_hi)`
test then failed, marking the `windowed` row inapplicable so selection fell
through to `bounded_small` / `moment` (`bs≈20`, `x_min=0`, ~99% empty buckets).

`_severity_high_estimate` now short-circuits a degenerate severity (standard
deviation ≤ `VALIDATION_NOISE`) to its atom location (the mean), so the windowed
method applies and is selected: `bs=1` on the integer lattice with a non-zero
origin. Non-degenerate severities are unchanged. No API change. This directly
fixes `create_frequency` count distributions at high frequency.

## 1.0.0a107

### `create_frequency()` — materialize the claim-count distribution

The engine carries frequency only as a PGF (applied in the Fourier domain), so
there was no `q` / `tvar` / `cdf` / percentiles for the claim *count*.
`Aggregate.create_frequency()` now returns a first-class `Aggregate` whose
aggregate density **is** the count distribution `P(N = k)` — built through the
normal `build` front door via the `dsev [1]` point-mass trick (N unit claims sum
to N). Every inherited method (`q`, `tvar`, `cdf`, `sf`, `pmf`, `plot`,
`density_df`, `summary_df`, …) then works on the count.

```python
fa = a.create_frequency()    # an Aggregate that IS the count distribution
fa.q([0.01, 0.5, 0.99])      # count percentiles
fa.tvar(0.99)                # tail count
```

`Portfolio.create_frequency()` returns a `Portfolio` of one count-distribution
unit per constituent aggregate; the portfolio total is the **total claim count
across all units**.

The rendered program keeps only the frequency clause (family + mixing /
contagion + `zm`/`zt`) and collapses exposure to the resolved expected count
`self.n claims`; severity, layers, and both occurrence and aggregate
reinsurance are dropped — they reshape severity-per-claim or the aggregate
total but never the *number* of claims (carrying `occurrence net of 50 xs 0`
through a `dsev [1]` severity would have netted every unit to 0). An empirical
(`dfreq`) frequency is kept as the count distribution directly. The result is a
*snapshot* — rebuild if the parent changes. Built objects with no DecL program
raise a clear error.

## 1.0.0a106

### Reinsurance-aware Gross / Ceded / Net `PnL` view (Stage E of `PnL`)

`PnL` is reinsurance-aware. On an **aggregate-reinsurance**-bearing risky leg:

- `agg_re.make_pnl(consideration=P)` is a **net-only** P&L against the net loss
  (what comes out of the aggregate) — unchanged single-leg behaviour;
- `agg_re.make_pnl(gross=Pg, ceded=Pc[, net=Pn])` builds the **Gross / Ceded /
  Net** view. `PnL.gcn_df` (also returned by `summary_df`) is a **doubly
  additive** 3×3 exhibit: rows add (`Net = Gross + Ceded`, the legs are
  comonotone — 1-D, no joint model) and columns add (`Margin = Consideration +
  Obligation`). The ceded leg is literally negative: you pay the ceded premium
  (`−Pc`) and receive the recovery (`+E[R]`). `net=` overrides the retained
  premium (default `Pg − Pc`).
- **Net is the headline** — it drives the net `pnl_df`, moments, `evaluate`, and
  `plot()` (which overlays the three legs' margins, Net the heavy line).

Constructed via the Python API (no DecL surface). Requires aggregate
reinsurance; both `gross=` and `ceded=` are required, and are mutually exclusive
with `consideration=`. (Function-valued consideration already shipped in a103;
the DecL swing/slide builder and book-level `PortPnL` remain deferred.)

## 1.0.0a105

### `PnL.evaluate()` — the Cherny–Madan breakeven acceptability panel (Stage D of `PnL`)

A `PnL` is **evaluated, not priced**. `PnL.evaluate()` finds, per distortion
family, the breakeven stress that drives the risk-adjusted net to zero — i.e. it
calibrates `g(loss-version of the risky leg) = the held consideration` over the
**full** support (no asset cap, no cost-of-capital inversion), reusing
`Distortion.calibrate_set`.

- Returns an **acceptability panel**: one row per family (`ph`, `wang`, `dual`,
  `tvar` — `ccoc` excluded, it needs an asset level), with the family-specific
  breakeven `param`, the calibration `error`, and the family-agnostic
  **`gini_p`** (`= 2∫g − 1`) acceptability index (`@Cherny2009a`), plus
  `area = (gini_p+1)/2`.
- `gini_p` is **monotone in loading**: a more profitable position survives a
  larger stress and scores a larger `gini_p`. (We report the index in full —
  never abbreviated "AI".)
- Constant consideration only; function-valued (loss-sensitive) evaluation is
  deferred (raises a clear `NotImplementedError`).

## 1.0.0a104

### Signed additive `PnL.summary_df` and `PnL.plot()` (Stage C of `PnL`)

- **`PnL.summary_df`** — a signed, **additive** three-row P&L table where the
  `EX` column adds: **Consideration + Obligation = Margin**. Each row is its
  signed contribution to the net (`+` received / `-` paid for Consideration;
  `-` a loss borne / `+` a payoff held for Obligation), reported with the **SD**
  spread (not CV — the margin sits near break-even). A constant consideration is
  certain (SD 0); a callable (loss-sensitive) consideration carries a real
  SD/Sk. Buying flips both signs (Consideration `< 0` *and* Obligation `> 0`).
  Freq/Sev/Agg detail stays on `pnl.agg.summary_df`.
- **`PnL.plot()`** — its own two-panel exhibit (Margin density + distribution)
  with the break-even line at 0; **no severity panel** (a P&L is an affine of its
  aggregate, not a compound of a severity — plot the bare leg via
  `pnl.agg.plot()`).

## 1.0.0a103

### First-class `PnL` veneer; the in-place `pnl` affine removed (Stage B of `PnL`)

`pnl` is now a first-class **`PnL`** composition over a *pure-loss* `Aggregate`,
replacing the old in-place affine on `Aggregate`. `build('pnl ...')` returns a
`PnL` (was an `Aggregate`-with-affine). **Breaking** for the new `pnl` surface
(nothing external depends on it).

- **`PnL`** (composition): `pnl.agg` is the untouched loss obligation (honest
  density / moments / plot in loss terms); the net is the derived `pnl.pnl_df`
  (`consideration - X` for a loss leg, `consideration + X` for a payoff leg) — a
  cheap 1-D relabel, no FFT/window/dropped-mass machinery. Exposes `mean`, `sd`,
  `var`, `skew`, `prob_loss`, `q`/`cdf`/`sf`, and `value_type == 'payoff'`
  (a P&L is always payoff). Consideration may be a **signed scalar/vector**
  (sums to one book amount) **or a callable** `f(x)` (loss-sensitive
  swing/slide, passed by hand).
- **`Aggregate.make_pnl(consideration)`** — the canonical constructor (no `sign`
  argument; the combine sign is read from `value_type`). `build('pnl NAME C prem
  - <body>')` == `build('agg NAME <body>').make_pnl(consideration=C)`.
- **The in-place affine is gone.** `Aggregate` no longer carries
  `agg_premium`/`agg_reflect`/`agg_shift` or `_apply_agg_affine`/`_pnl_window`/
  `_agg_affine_active`; `update()` no longer reflects/shifts. `Aggregate` is a
  pure loss (or payoff-oriented) distribution again. The signed-*severity*
  (`ssev` / negative-`dsev`) path is unchanged.
- **Portfolios and bivariates of `pnl` units are deferred** (book-level / joint
  P&L: a loss-sensitive consideration must be netted per unit *before*
  combining, which the shared combine does not preserve). They raise a clear
  `NotImplementedError`. A **payoff book** is now expressed with payoff-oriented
  aggregates (`agg ... payoff`), not `pnl` units.

The signed additive `summary_df` (Consideration / Obligation / Margin),
`PnL.plot()`, the Cherny–Madan `evaluate` panel, the reinsurance-aware GCN view,
and a DecL swing/slide builder land in later stages.

## 1.0.0a102

### DecL `payoff` / `loss` orientation suffix on `agg` (Stage A of `PnL`)

First stage of the first-class `PnL` work (`dev/done/plan-pnl.md`). An `agg`
declaration may now carry a trailing **`payoff`** or **`loss`** keyword that sets
the variable's sign-convention role (`value_type`) — *pure orientation*, with
**no reflect/shift** (that affine remains the `pnl` wrapper):

```
agg AssetReturn 100 claims sev lognorm 10 cv 1 poisson payoff
```

- `payoff` marks an asset-return / direct-payoff primitive ("more is better"); it
  prices through the **dual** distortion via the existing `_is_loss_value` path
  (`_canonical_loss_frame` reverses the support — no new pricing machinery).
  `loss` is the explicit default; omitting the suffix is unchanged.
- The suffix sits **last, immediately before the `note`/`hints` trailer** (after
  freq / reinsurance / `approximate`), so it cannot collide with the `loss`
  *exposure* head keyword. Available on both the `freq` and `dfreq` agg forms.
- The DecL unparser (`decl_writer`) round-trips the suffix; a default `agg` is
  unaffected.

No behavior change for existing programs (the default is `loss`). The `pnl`
veneer, signed summary, `evaluate` panel, plotting, and reinsurance-aware GCN
views land in later stages.

## 1.0.0a101

### `.help` render targets + `output` → `values` rename (**breaking**)

`.help` (and the `agg_help` worker backing it on `Aggregate`, `Portfolio`,
`Underwriter`, and the bivariate class) gains an `fmt` axis that picks the render
target, so it reads well in a plain terminal / REPL — not only in Jupyter, where
it previously rendered through `IPython.display` and printed ugly object reprs
everywhere else.

- `fmt='auto'` (default) resolves to **ANSI** under a Jupyter kernel and **plain
  text** in a terminal. It never auto-selects `html` (ANSI color with a
  consistent monospace font is preferred even in JupyterLab); reach `html`
  explicitly. `fmt='text'` / `'ansi'` are dependency-free (no IPython import);
  `fmt='html'` is the original rich Markdown path.
- Jupyter detection probes `sys.modules` and never forces the ~1s IPython import.

**Breaking:** the a98 `output` parameter is renamed to **`values`** (`none` /
`short` / `all` — how much of each name's value / call result to show), removing
the `output`/`fmt` overlap. `lod`, `values`, and `fmt` are now three orthogonal
axes: docstring detail, value detail, render target. (The unrelated
`Aggregate.approximate(output=...)` / `Portfolio.approximate` keep their own
`output` — different method, different meaning.)

Tests: `tests/test_help.py` (text/ansi/auto/html targets, the rename, bad-value
guards); `tests/test_bivariate.py::test_mv_help_runs` updated to `values=`.

## 1.0.0a100

### `format_program` spread layout (default multiline)

`format_program` gains a `layout` axis, orthogonal to the existing `fmt`
(markup) axis. `layout='spread'` — now the **default** — renders DecL multiline:
each clause on its own two-space-indented line, with reinsurance cessions and a
portfolio's / bivariate's sub-aggregates nested one level deeper. `layout='terse'`
is the historical single-line-per-statement form (a portfolio keeps its
tab-indented units) and is byte-for-byte what `spec_to_decl` / `to_agg` produce.

So `Aggregate.pprogram` / `.pprogram_html` and `Portfolio.pprogram` /
`.pprogram_html` now render spread by default (and the doc examples that print
them will re-render multiline on the next docs build). Pass
`format_program(obj.program, layout='terse')` for the old one-line form.

Internals: both layouts render from one intermediate `_Block` tree (head line +
indented children), so the clause ordering lives in a single place. `spec_to_decl`
stays terse-only — the round-trip / idempotence contract and the `.agg` snapshot
are unchanged. Both layouts re-parse to the same spec (the preprocessor collapses
intra-statement newlines + indentation to a single space). Top-level statements
in a multi-statement program are now joined with a blank line so they re-parse as
distinct statements.

Docs: the printed DecL examples in the user guides now render spread; the author
rebuilds the doc tree out-of-loop.

## 1.0.0a99

### Distortion calibration on signed and payoff supports

`calibrate_distortions` (on `Aggregate` and `Portfolio`) now calibrates
correctly for **every** support — non-negative loss, signed loss (straddles 0,
via `ssev` or a negative `dsev` atom), and payoff (`pnl`, more-is-better) — not
just `X >= 0`. The layer / Lee integral `∫₀^∞ g(S) dx` that each subclass
`calibrate` evaluates is valid only on a non-negative axis, so the caller now
maps the object onto a **canonical non-negative loss frame** `Z`:

- **reverse** a payoff (`X -> -X`) so its bad tail lands on the right where a
  concave `g` loads it — the matched half of the `g_dual` flip `effective_g`
  already applies when *pricing* a payoff (the two cancel on price);
- **shift** by `c = max(0, -min(support))` onto `[0, ∞)`.

By translation-equivariance the shift is exact, not an approximation: the
calibrated distortion *shapes* are frame-free and identical to the unshifted
law's. The per-kind subclass math is **unchanged** (still pure and 0-based); all
of the new bookkeeping lives in `_pricing.calibrate_distortions` /
`_pricing._canonical_loss_frame`. The classic `X >= 0` path (`c = 0`, no
reverse) is byte-for-byte unchanged and provably never enters the transform
branch.

The `calibration_df` receipt is reported in the caller's loss convention
(un-shifted): `M`, `Q`, `coc` are shift-invariant; only `L`, `P`, `a` slide by
`-c`, going negative *together* exactly when the position is net-beneficial (a
payoff that is really a profit — "Loss" then reads as a negative number). The
accounting identities `P = L + M` and `a = P + Q` always hold with `M, Q >= 0`.

`calibrate_distortions` also gains a `names=` argument (both classes) to select
the distortion families to calibrate; default is the standard set.

Tests: `tests/test_signed_calibrate.py` (shift-exactness, payoff dual
round-trip, per-kind sweep incl. mass-at-zero `ly`/`clin`/`lep`, classic
no-op); DecL mirrored as `SC.*` in `decl-testers.agg`.

## 1.0.0a98

### `agg_help` / `.help` — finer control over detail (`lod`, `output`)

`agg_help` (and the `.help(regex)` method on `Aggregate`, `Portfolio`,
`Underwriter`, and the bivariate classes) gains two orthogonal knobs:

- **`lod`** (`'terse'|'short'|'all'`, default `'short'`) — level of
  *documentation* detail: `'terse'` shows the name and method signature only,
  `'short'` the first few lines of the docstring, `'all'` the full docstring.
- **`output`** (`'none'|'short'|'all'`, default `'short'`) — how much of each
  *value* (attribute value or no-argument call result) to show: `'none'` shows
  none, `'short'` shows values but truncates a `DataFrame`/`Series` to
  `.head(5)`, `'all'` shows them in full.

Methods now display their bound signature in the header, properties show their
docstring, and invalid `lod`/`output` values raise `ValueError`. The default
(`lod='short', output='short'`) gives a compact, scannable readout in place of
the former full-docstring-and-full-value dump.

## 1.0.0a97

### `prob_loss_assets` / `pla` — free choice of capital anchor; `price_pentagon_ex` (Plan: plan-pla)

A new distributional primitive and a full-power pentagon front door built on it.

- **`prob_loss_assets` (alias `pla`)** on `GridDistribution`, with thin
  delegators on `Aggregate` and `Portfolio`. Given any **one** of the VaR
  probability `p`, the limited expected loss `L = E[min(X, a)]`, or the asset
  level `a`, it returns the consistent grid-snapped triple `(p, L, a)` — three
  views of the same point on the distribution. The `L`-anchor inverts `lev(a) =
  L` by a safeguarded Newton step (`a ← a − (lev(a) − L)/S(a)`) with a bisection
  fallback for the ill-conditioned far tail (`S(a) → 0`); a feasibility guard
  rejects `L ≥ E[X]`. Returns the module-level `ProbLossAssets` namedtuple.

- **`price_pentagon_ex`** on `Aggregate` / `Portfolio` — the full-power front
  door over `price_pentagon`. Accepts the full pentagon vocabulary
  (`{p, a, L, M, P, Q, LR, PQ, ROE}`), free over the capital anchor (now
  including the expected loss `L`). `Pentagon.solve` remains the gatekeeper
  (insoluble configs such as `{PQ, ROE, LR}` raise); the distribution supplies
  the single extra equation `L = lev(a)` via `pla`, injected only when the
  accounting is one equation short. A uniform post-check warns
  (`UserWarning`, a pricing-time advisory, **not** routed through
  `explain_validation`) when an accounting-determined `L` does not reconcile
  with `E[min(X, a)]` at the solved assets. Output is one `'total'` row with a
  leading `p` column followed by the eight canonical pentagon stats.
  `price_pentagon` / `solve_obj` / `price_ccoc` are unchanged underneath.

- **Fixed `GridDistribution.lev` on the cached grid view.** `Aggregate` /
  `Portfolio` `_grid_distribution()` now builds on the **full** contiguous `bs`
  grid instead of the `p_total > 0` subset. The var/tvar kernel already filters
  to the positive-mass subset internally, so `q` / `tvar` / `mean` are
  byte-identical (frozen baseline unmoved), but the width-summing `lev` / `cdf`
  / `sf` need the full grid: the subset dropped the empty low buckets (where
  `S == 1`), which made `lev` silently undercount `E[min(X, a)]` by the missing
  slab. `lev` now matches the `exa` / `exa_total` / `add_exa` datum to
  floating-point dust — which is what lets `pla` and `price_pentagon_ex` use
  `GridDistribution.lev` as the single LEV source.

## 1.0.0a96

### Rationalize the `xsden_*` / `ser_to_mwrangler` moment helpers

The `aggregate.moments` discretized-density helpers were a core function plus
four thin wrappers — too many near-identical options. Two had **zero** source
callers and appeared in no public example, so they are removed (breaking, but
dead surface):

- **Removed `xsden_to_noncentral`** — the raw-moments need is met by
  `xsden_to_mwrangler(xs, den).noncentral` (what the live callers already use).
- **Removed `ser_to_mwrangler`** — a one-line Series adapter no caller used;
  `xsden_to_mwrangler(ser.index.to_numpy(), ser.to_numpy())` is the direct form.

Kept: `xsden_to_mwrangler` (the core; returns a `MomentWrangler` with every
view) and the two reductions with real callers, `xsden_to_meancv` (→ `(m, cv)`)
and `xsden_to_meancvskew` (→ `(m, cv, skew)`), both also demonstrated in the
reinsurance-pricing user guide. `xsden_to_mwrangler`'s docstring now spells out
when to prefer the reductions (single view) vs. the core (more than one view).
`bivariate.py` was switched from the inline `xsden_to_mwrangler(...).mcvsk`
idiom to `xsden_to_meancvskew(...)` so there is one obvious way.

## 1.0.0a95

### Concern modules filled in; legacy bucket sizers retired (Plan: finish shared concerns)

The closing step of the god-module refactor. P3/P4 birthed three concern
modules — `_reinsurance.py`, `_validation.py`, `_bucket_window.py` — but left
them as thin shells: only the leaf math + constants had moved, while the
per-class orchestration bodies stayed on `Aggregate` / `Portfolio`. This
iteration moves those bodies in, so each concern lives in one place, and retires
the legacy `recommend_bucket` / `best_bucket` sizers. **Behaviour-frozen:** no
numbers change; the relocations are byte-identical and the frozen baseline
(`test_baseline.py`) is unmoved.

- **Reinsurance bodies → `_reinsurance.py`** (Agg-only). The apply engine
  (`apply_reins_work` and its occ/agg drivers), the reporting frames
  (`reins_density_df`, `reins_stats_df`, `reins_summary_df`, the view-stats and
  moment kernels), and the narrative (`reins_description`, `reins_kinds`,
  `reins_after_label`, the describe block) are now free functions taking the
  `Aggregate`; the methods/properties delegate. `reins_occ_plot` continues to
  route through `plots/`.
- **Bucket/window bodies → `_bucket_window.py`.** `Aggregate._bs_window` and the
  Portfolio sizers (`best_window`, `_bs_window`, `bs_window_df`,
  `_single_big_jump_window`, `_build_bs_window_df`) are now free functions
  (`bs_window`, `port_best_window`, …) sitting beside their
  `estimate_agg_window` / `_estimate_agg_percentile` leaves; the methods
  delegate. `value_type_role` moved to `utilities.py` (a leaf) so both
  `_aggregate` and `_bucket_window` can share it without a cycle.
- **Validation bodies → `_validation.py`.** `Aggregate.valid` /
  `Portfolio.valid` become `valid_aggregate` / `valid_portfolio` (similar but
  not identical — kept side by side, not merged); the two `validation_explanation`
  properties share one free function. The classes' properties delegate.
- **Breaking: `recommend_bucket` and `best_bucket` removed** (both `Aggregate`
  and `Portfolio`). They were off the live path since 1.0.0a49 — `update`
  routes `bs==0` through `_bs_window` / `best_window`. `aggregate_error_analysis`
  now seeds its starting `bs` from `estimate_agg_window` instead. Docs and the
  user guide updated to the `best_window` / `bs_window_df` surface.

## 1.0.0a94

### `portfolio.py` split into a façade + three subsystems (Plan P4, Phase 4A)

The ~4,750-line `portfolio.py` god module is split behind a re-export façade,
mirroring the P3 `distributions.py` split. `Portfolio` stays one public class;
its body is divided along **how the joint loss distribution is built**. Every
import path is unchanged (`aggregate.Portfolio`,
`aggregate.portfolio.Portfolio`, `from aggregate.portfolio import …`).

- **`portfolio.py` is now a thin façade.** The `Portfolio` class lives in
  `aggregate._portfolio`; the façade re-exports the historical public surface
  (`Portfolio`, `make_awkward`, `make_comonotonic_allocations`,
  `swap_density_df`) plus the qualified-path attributes
  (`check01` / `make_array` / `convex_points`,
  `VALIDATION_NOISE` / `ALIASING_RATIO` / `EXEQA_NOISE_FLOOR`).
- **`aggregate._portfolio_density`** — the density-based (independence) path:
  the `add_exa` / `exeqa_*` independent-sum kernel and the `_ft_nots`
  spectral-division helper, lifted to free functions taking `port`.
  `Portfolio.add_exa` is now a thin wrapper.
- **`aggregate._portfolio_common`** — the common exeqa numerics, agnostic to
  FFT-vs-sample origin (the switcheroo invariant): `build_augmented` (the
  apply-distortion / linear-vs-lifted allocation engine), `unit_capital_at`
  (layer-ROE capital allocation), `allocation_diagnostics`, `bodoff`, and the
  convex-hull helpers. The corresponding `Portfolio` methods delegate.
- **`aggregate._portfolio_sample`** — the sample-based (dependence) path:
  `sample` (Iman–Conover), `add_exa_sample` (the switcheroo's exeqa-from-sample
  kernel), `swap_density_df`, and the `make_comonotonic_allocations_work`
  majorization. **Behaviour-frozen relocation** — the substantive review of the
  dependence machinery is a later plan.
- **Consumes the P3 shared concerns:** `Portfolio.price_pentagon` now delegates
  to `aggregate._pricing.price_pentagon` (was a duplicated body), joining
  `calibrate_distortions` / `price_ccoc` which already routed through `_pricing`
  (a93).
- **New `tests/test_portfolio_subsystems.py`** exercises the extracted kernels
  directly (reachable now without a full `update()` / `sample()`); the full
  baseline is unmoved (behaviour-frozen extraction, numbers identical).

**Breaking (pre-1.0): `Portfolio.percentiles` removed.** The interpolated
per-unit percentile table (no `Aggregate` analogue) is superseded by the exact
vector-valued `q`. Use `port.q(p)` / `port['unit'].q(p)` for quantiles.

### Deferred within P4

- The intricate, author-sensitive **bucket/window sizers**
  (`recommend_bucket` / `best_bucket` / `best_window` / `bs_window_df` and
  helpers) and the **validation** bodies (`valid` / `validation_explanation`)
  stay on `Portfolio` for now rather than being dropped into the shared
  `_bucket_window` / `_validation` modules — matching P3 Phase 1b, which
  likewise deferred extracting the same-shaped `Aggregate.valid` body. They land
  alongside the Aggregate-side extraction so the two sizers/validators can be
  read side by side safely.
- The **sample-subsystem review** and the **4B composition** seam remain
  post-beta, conditional, each its own later plan.

## 1.0.0a93

### Distortion calibration on a single distribution — `Aggregate`/`Portfolio` parity

Phase 1c of the `distributions.py` split (`dev/done/plan-split-distributions.md`).
The pricing-distortion *family calibration loop* moves onto `Distortion`, and an
`Aggregate` can now calibrate a distortion set directly — no more wrapping it in
a one-unit `Portfolio`.

- **New `Distortion.calibrate_set(...)` classmethod** — the permanent home for
  the family loop. Given a survival vector and a premium target it constructs
  and calibrates each kind in `names` (default `('ccoc', 'ph', 'wang', 'dual',
  'tvar')`) and returns `{name: calibrated Distortion}`. Pure `Distortion`
  knowledge (the family registry / per-kind initial shape); the caller hands in
  the GD-derived data (**GD → Distortion**; `GridDistribution` never imports
  `Distortion`). Extracted faithfully from the per-name dispatch that lived
  inline in `Portfolio.calibrate_distortion`.
- **New `Aggregate.calibrate_distortions(coc, *, p=None, a=None, kind='lower')`**
  — the `Aggregate` counterpart of `Portfolio.calibrate_distortions`, same
  receipts (`distortion_df` / `calibration_df` / `distortions`). Calibrates to
  the aggregate's own distribution; per-unit allocation stays a `Portfolio`
  concern. Verified bit-for-bit against the legacy one-unit-`Portfolio` path on a
  shared grid.
- **New `Aggregate.price_ccoc(ccoc, *, p)`** — parity with
  `Portfolio.price_ccoc`; the cost-of-capital alias of `price_pentagon`.
- **Breaking (internal): `Portfolio.calibrate_distortion` (singular) removed.**
  It was only ever called by the plural `calibrate_distortions`; the plural is
  unchanged in signature and numerics but now resolves the survival datum once
  and drives `Distortion.calibrate_set`. The singular's unused
  `S_column`/`S_calc`/`r0`/`kind` options (the plural always used defaults) are
  gone. No public caller existed.
- The single-distribution pricing glue (pentagon-target → `calibrate_set`, plus
  `price`/`price_pentagon`/`price_ccoc`) now lives in the shared
  `aggregate._pricing` module born in the structural split below.

### Aggregate FFT convolution core extracted as a pure function (Phase 2A)

- **New `aggregate._aggregate_compute.freq_sev_convolution(sev_density,
  freq_pgf, n, *, N, bs, i0, x_min, en, freq_name, padding)`** — the
  FFT-PGF-iFFT kernel (`iFFT(freq_pgf(n, FFT(sev)))`) lifted out of
  `Aggregate._fft_aggregate`, which is now a thin wrapper passing its array /
  scalar state explicitly. The kernel is reachable and testable **without** a
  full `Aggregate.update()` and is notebook-inspectable. Behavior is unchanged
  (byte-for-byte: the wrapper reproduces the prior `agg_density`).
- New `tests/test_aggregate_compute.py` drives the kernel directly: fixed count
  vs repeated `np.convolve`, the zero-risk point mass, compound-Poisson mean /
  variance identities, and byte-for-byte parity with a built `Aggregate`
  (fixed and Poisson frequency).
- `Aggregate.discretize` was **not** extracted — it orchestrates the per-
  component `Severity` objects (`sev.cdf`/`sf`/`fz`, `_rebucket_to_grid`) rather
  than transforming plain arrays, so it stays a method.

### `distributions.py` split into kind modules + shared concerns (structural)

Phases 1 and 1b of the same plan — pure code relocation, **no behavior change**
(full suite matches the pre-split baseline at each step), so not separately
version-bumped; recorded here for orientation.

- `distributions.py` is now a thin re-export **façade** over `_fits` /
  `_frequency` / `_severity` / `_aggregate`. Every historical import path
  (`aggregate.Aggregate`, `aggregate.distributions.X`,
  `from aggregate.distributions import …`) is unchanged.
- The cross-cutting concerns are born as leaf/near-leaf modules reusable by both
  `Aggregate` and `Portfolio`: `_bucket_window` (grid sizing), `_reinsurance`
  (Agg-only ceder/netter), `_validation`, and `_pricing`.
- **`explain_validation` relocated** from `utilities.py` to `_validation.py`,
  with a back-compat re-import kept in `utilities` (mirrors the P1 `make_var_tvar`
  move); all existing import paths keep working.

## 1.0.0a92

### Plotting subsystem — single matplotlib boundary (Pass A)

The library-wide plotting refactor (`dev/done/plan-plots-subsystem.md`), **Pass A**:
a behavior-preserving port of every plot body into a new `aggregate.plots`
subpackage organized as canvas × content × class. No version bump (pure moves
plus a behavior-adjacent import-timing change); figure output is unchanged.

- **New `aggregate/plots/` subpackage** — the single matplotlib entry point for
  the whole library. Three layers: `_style.py` (Layer 0 — canvas creators
  `make_mosaic`/`make_grid`, the house style absorbed from the old `style.py`,
  and the shared figure constants); `_quantile.py` (Layer 1 — the shared Lee /
  quantile content worker); and per-class Layer-2 compositors `_aggregate`,
  `_severity`, `_distortion`, `_portfolio`, `_bounds`, `_bivariate`, `_fourier`.
- **`import aggregate` no longer imports matplotlib.** Each class `.plot()` is
  now a one-line stub delegating to its compositor via a function-local import,
  so matplotlib is loaded only on the first plot. Confirmed by
  `tests/test_plots_boundary.py`, which also asserts no module outside `plots/`
  imports matplotlib at top level (the figure-generator module `pedagogy.py` is
  a documented exemption; the paper-figure helpers in `ft.py`/`tweedie.py` use
  function-local imports).
- **`aggregate.style` is now a thin backward-compat shim** re-exporting `use` /
  `context` / `rc_params` from `aggregate.plots._style`; `import aggregate.style`
  and the docs/apiweb callers are unchanged.
- Moved bodies: `Aggregate.plot` / `reins_occ_plot`, `Severity.plot`,
  `Distortion.plot` / `plot_affine`, `Portfolio.plot` / `scatter` /
  `sample_compare`, `Bounds.plot_envelope` / `plot_weights` /
  `_HullEngine.plot`, `BivariateAggregate.plot` / `BivariateDistribution.contour`,
  and the matplotlib `FourierTools` plots. The public method signatures are
  unchanged. (Pass B — the deliberate visual refresh — is tracked separately.)

### Docs bibliography re-sourced from the author's master library

The Sphinx bibliography is now generated from the author's master BibTeX
library (`uber-library.bib`, ~7,100 entries) instead of two ad-hoc local files.

- **New** `docs/update_extract_bib.py` — scans every `:cite:` key in
  `docs/**/*.rst`, pulls the matching entries from the master library by
  brace-balanced extraction, and writes `docs/extract.bib`. It exits non-zero
  and names the offenders if any cited key is missing from both the master
  library and `manual.bib`, so broken citations are caught before commit. The
  library path is configurable (`--uber` / `UBER_LIBRARY`) and only ever read.
- **New** `docs/manual.bib` — hand-maintained entries for the few academic
  works absent from the master library (Bertram1983, Panjer1992, Lukacs1970bk,
  McKean2014bk, Bertram1981) plus software citations (Python, SciPy, pandas,
  matplotlib, SLY, ...).
- **New** `docs/README.md` — documents the workflow.
- **Retired** `docs/books.bib`; `extract.bib` is now generated, not hand-rolled.
  `conf.py` lists `['extract.bib', 'manual.bib']`; `docs/bib.bat` runs the new
  script.
- **Cite-key renames** across ~28 `.rst` files to canonical `AuthorYYYY` keys
  (e.g. `PIR` → `Mildenhall2022a`, `LM`/`KPW`/`kpw5` → `Klugman2019`,
  `JKK` → `Johnson2005`, `feller71` → `Feller1971`, `WangS1998` → `Wang1998a`).
  The previously cited-but-undefined `KPW` is now resolved.

Docs need a manual rebuild (not run in the iteration loop).

### Landing-page intro simplified

The `index.rst` introduction replaced its six-panel `sphinx_design` card grid
(with per-section graphics) with a plain numbered list mirroring the chapter
numbers (1–6). The six `_static/*.png` card images were removed, and the now
unused `sphinx_design` extension was dropped from `conf.py` and the `dev`
dependencies.

## 1.0.0a91

### `GridDistribution` adopted by Aggregate, Portfolio, and Bounds (P1, Phases 2–5)

The duplicated, drifted var/tvar plumbing is replaced by the `GridDistribution`
value type from 1.0.0a90. Same kernel ⇒ numerically identical (the frozen
`test_baseline.py` does not move) — **except one called-out bug fix**:

- **`Aggregate`** — `_var_tvar_function` / `_sev_var_tvar_function` caches and the
  `_make_var_tvar` wrapper are gone; `q` / `tvar` delegate to a lazily-built
  `GridDistribution` over the aggregate grid, and `q_sev` / `tvar_sev` to one over
  the severity grid (`sev_density_df`).
  - **Bug fix:** `tvar_sev(p)` previously read the *aggregate* tvar cache
    (`_var_tvar_function['tvar']` built from `p_total`) rather than the severity
    grid, so it returned the aggregate TVaR. It now correctly returns the
    severity-grid TVaR — **this number changes** (e.g. for a lognormal-severity
    book it dropped from the aggregate's ~141 to the severity's ~17).
- **`Portfolio`** — same swap; the mutate-vs-return `_make_var_tvar` divergence is
  gone, and `tvar_threshold` now delegates to `GridDistribution.tvar_threshold`
  (the scattered `_var_tvar_function = None` resets become `_dist = None`).
- **`Bounds`** — the hand-rolled `make_var_tvar(pd.Series(...))` calls in
  `_resolve_obj` and `_RiskSource` are replaced by `GridDistribution`; `_resolve_obj`
  now hands `Bounds` a `GridDistribution` (reusing the `Aggregate` / `Portfolio`
  object's own view), and the capped-TVaR formula `TVaR_p(min(X, a))` moved onto
  the value type as `GridDistribution.tvar_of_limited(p, a)` — `Bounds._tvar_x_a`
  delegates to it (O(1), no grid rebuild, so the `p_star` root-find stays cheap).

`cdf` / `sf` / `pdf` / `pmf` on `Aggregate` / `Portfolio` are left on their existing
`interp1d` mechanism for now (those objects are used directly by the plotting code
as `interp1d` callables, so they are not pure var/tvar plumbing). **Phase 6
(Bivariate)** is deferred: it is a purely *additive* exposure (Bivariate has no
existing `q` / `tvar`), the joint-vs-marginal quantile semantics are a design
question, and Bivariate's structural treatment is already deferred per
`dev/done/plan-README.md`.

## 1.0.0a90

### `GridDistribution` — the shared discrete-grid distribution value type (P1, Phase 1)

New leaf module `aggregate._grid_distribution` introduces `GridDistribution`, a
small read-only value type holding a probability vector `p` over a loss index
`x` with an optional bucket size `bs`. It is the single home for the
marginal-vector accessors that `Aggregate`, `Portfolio`, `Severity`, `Bounds`
and `Bivariate` all need:

- **Spacing-agnostic probability accessors** (no equal-spacing assumption):
  `q`/`var` (lower & upper quantile), `tvar`, `tvar_threshold`, `cdf`, `sf`,
  `pmf`, `mean`, and the new `lev(a)` (limited expected value `E[min(X, a)]`,
  matching the `Portfolio.add_exa` `bs · Σ_{x<a} S` convention) and
  `tvar_of_limited(p, a)` (the analytic `TVaR_p(min(X, a))` composite, O(1) — no
  grid rebuild).
- **Width-dependent ops** (`pdf` = mass / `bs`, `snap` to the regular grid)
  require `bs` and raise a clear error when it is `None`.
- `cap(a)` returns a fresh `GridDistribution` for `min(X, a)`.

The `make_var_tvar` kernel **relocated** from `aggregate.utilities` into the new
module (it is no longer in `utilities.__all__`); `balanced_window` and all
internal callers import it from its new home. Per the pre-1.0 no-deprecated-alias
rule this is a clean move with no shim.

`_DiscreteRV` (the discrete-severity frozen RV) is now a thin scipy-naming
adapter over a held `GridDistribution`: it shares the cumulative-step core
(`cdf`/`sf`/`mean`/lower-quantile `ppf`/`isf`) and keeps only the
severity-specific bits (`pdf ≡ 0`, raw `moment`, `stats`, `var`, `rvs`,
`layer_moments`). Behaviour is identical (same kernel) — `test_discrete_severity`
is unchanged.

New `tests/test_grid_distribution.py` brute-force / analytic-checks every
accessor (fair-die and skewed grids, a non-uniform grid, and cross-checks of
`tvar_of_limited` against `cap(a).tvar` and `Bounds._tvar_x_a`).

This is a **pure addition**: no consumer behaviour changes yet. Per-consumer
adoption (Severity → Bounds → Aggregate → Portfolio → Bivariate) follows in
later, behaviour-guarded phases.

## 1.0.0a89

### Pedagogy figures renamed off the legacy PIR `fig_<ch>_<num>` names

The five remaining book-figure helpers in `aggregate.pedagogy` carried opaque
`fig_4_1`-style names tied to *Pricing Insurance Risk* chapter/figure numbers.
They now have descriptive names; the original name and PIR figure number are
recorded in each docstring:

- `fig_4_1` → `plot_quantile_illustration`
- `fig_4_5` → `plot_discrete_distribution_quantile`
- `fig_4_6` → `plot_continuous_distribution_quantile`
- `fig_4_8` → `plot_tvar_quantile`
- `fig_9_1` → `plot_ruin_surplus_paths`

**Breaking:** the old names are gone (submodule-only helpers, never core API).
The citing technical-guide docs (`5_x_quantiles`, `5_x_nm_discrete_rep`,
`5_x_pk`, `2_x_10mins`) now import the new names. `natural_scale` is unchanged.

## 1.0.0a88

### DecL syntax colourer + error labels resynced with the grammar (D10)

The Pygments lexer (`decl_pygments.AggLexer`) and the parse-error terminal
labels (`parser_errors._TERMINAL_LABELS`) had drifted from `decl.lark` as
keywords were added. Both are now up to date and a new test keeps them honest:

- **Colourer** now recognises the keywords added since the last sync —
  `pnl`, `ssev`, `approximate`/`approx`, `bivariate`/`bv`, `clash`, `copula`,
  `dbvsev`, `netceded`, `grossceded`, `grossnet` — and drops four stale words
  that are no longer DecL keywords (`unlimited`, `unlim`, `wt`, `x`).
- **Error labels** gained friendly entries for the same ten new terminals
  (`APPROXIMATE`, `BIVARIATE`, `NETCEDED`, `GROSSCEDED`, `GROSSNET`, `CLASH`,
  `COPULA`, `DBVSEV`, `SSEV`, `PNL`).
- **`tests/test_grammar_sync.py`** derives the canonical keyword set directly
  from `decl.lark` (the `ID` terminal's exclusion list) and the priority-tagged
  terminal names, then asserts the colourer colours every reserved word and the
  label table covers every keyword terminal. Adding a keyword to the grammar
  now fails this test until the mirrors are updated.

The web app's `decl-keywords.json` lives in a separate repo and is out of scope
here; the remaining `D10` work is to drive that and the two now-synced mirrors
from a single generated source.

## 1.0.0a87

### Infinite-variance aggregates now error without an explicit `bs`

Sizing the FFT grid relies on a method-of-moments tail estimate, which needs a
finite variance. An aggregate whose severity has no finite second moment (a
power law such as `pareto` with shape `alpha <= 2`) gives no basis to guess
`bs`, so building one without an explicit `bs` now raises the new
`InfiniteVarianceError` (a `ValueError` subclass, exported from
`aggregate.constants`) instead of silently sizing a "reachable bulk" and
warning. Pass an explicit `bs` to build these:

```python
build('agg IMP 3 claims sev 100 * pareto 1.5 - 100 poisson', bs=1)
```

This reverts the earlier reachable-bulk fallback (and removes the internal
`Aggregate._reachable_bulk_high` helper). Finite-variance heavy-tailed
aggregates (e.g. `pareto` shape `> 2`, or any layered/limited severity) are
unaffected and still auto-size.

### `knowledge` DataFrame shows readable `source` provenance

The `source` column of `build.knowledge` no longer prints the full resolved
path of every database. It now collapses by location: a built-in database
shows just its name (no directory, no `.agg` suffix, e.g. `test_suite`); a user
database under `~/.aggregate` shows `~/<name>` with the suffix dropped and any
sub-directory kept (e.g. `~/mylib`, `~/sub/mylib`); any other loaded file shows
its full path; the `session` sentinel passes through unchanged.

## 1.0.0a86

### Discrete bivariate severity (`dbvsev`) + discrete-frequency `bv` forms

New DecL feature: a `bivariate` (`bv`) object can be declared **directly from
discrete data** — a shared discrete frequency (`dfreq`) and/or a discrete
bivariate severity (`dbvsev`) — the 2-D analogue of `agg NAME dfreq [...] dsev
[...]`. One new keyword (`dbvsev`), a new `mode='discrete'` on
`BivariateAggregate`, and no new dependencies.

- **`dbvsev` keyword** — the joint per-claim probability matrix given directly on
  an explicit lattice (`S[i][j] = P(X=xs[i], Y=ys[j])`, rows = X, columns = Y).
  Three auto-detected surface forms:
  - dense contingency table `dbvsev [xs] [ys] [[row] [row] ...]`;
  - dense with the matrix omitted ⇒ **uniform** over the lattice;
  - sparse triples `dbvsev [[x y p] [x y p] ...]` (collisions summed).

  Ranges (`dbvsev [0:2] [0:2]`) are accepted; probabilities are renormalised to
  sum 1 (with a warning) and validated for shape / non-negativity.
- **Four `bv` forms** now build, the 2×2 of {`exposures … freq`, `dfreq`} ×
  {two `agg`/`pnl` + copula, `dbvsev`}:
  - `bv N dfreq [...] [...] dbvsev [...]` (headline);
  - `bv N <count> claims dbvsev [...] <freq>`;
  - `bv N dfreq [...] [...] agg A … agg B … copula …`;
  - the existing `bv N <count> claims agg … agg … copula …`.
- **`mode='discrete'`** reuses the whole copula 2-D compound FFT path; only the
  formation of the joint per-claim matrix `S` changes (given directly instead of
  built from a copula). Per-axis sizing uses the lattice gcd as the bucket size,
  so `S` scatters onto the grid with no rebucketing and the marginals reproduce
  the standalone discrete compounds **exactly**. Loss/loss only — a `pnl` axis
  raises a clear error.
- **Reporting** — a discrete `bv` reports `kind = discrete (dbvsev)`, `copula tau
  = n/a`; `summary_df` / `stats_df` show exact marginal reproduction;
  `dependency_df` reports the per-claim and aggregate cov / corr (tau is `nan`,
  no copula).

Supporting changes:

- **Nesting-aware preprocessor** — `UnderwritingLexer.preprocess` now collapses
  newlines inside `[ ]` with a depth counter so the `dbvsev` `[[ ... ]]` matrices
  survive; the non-nested path is unchanged (byte-for-byte, so the parser
  snapshot is unaffected).
- **Clean transformer errors** — a `ValueError` raised in the transformer (e.g.
  `dbvsev` validation) now surfaces directly instead of wrapped in Lark's
  `VisitError`.
- The unparser (`decl_writer`) renders discrete `bv` objects back to the
  canonical dense `dbvsev` form.

## 1.0.0a85

### Accessor-name rationalization + bivariate reporting redesign (breaking)

Finishes the `<noun>_df` / noun-property convention sweep across the four main
classes and rebuilds the `BivariateAggregate` reporting surface. **No deprecated
aliases** — clean breaks (pre-1.0).

Renames / decorator changes:

- **`reins_describe` → `reins_summary_df`** on `Aggregate` and `Portfolio` (the
  last `describe`-verb property left after a84; it returns a DataFrame). Private
  workers `_reins_describe` / `_reins_describe_block` keep their names.
- **`Aggregate.reins_kinds()` → property `reins_kinds`** (a no-arg narrative
  accessor; now a noun property like every other narrative member).
- **`Distortion.tvar_info_df()` → `cached_property tvar_info_df`** (base and the
  `DistortionWtdTVaR` override), joining the `info` / `summary_df` / `stats_df` /
  `density_df` cached-property quartet; added to the cache-invalidation list.
- **`Portfolio.trim_df()` → `trim_density_df()`** and the `update(trim_df=…)`
  kwarg → `update(trim_density_df=…)` in lockstep (it is a mutator returning
  `None`, so the `_df` suffix was misleading).
- **`BivariateAggregate.corr()` / `marginals()` → properties** `corr` /
  `marginals` (no-arg accessors; `moments(max_order)` stays a method). The
  lower-level `BivariateDistribution` methods are unchanged.
- Removed the internal `_BOUNDED_FREQS` / `_BOUNDED_SCIPY_SEVS` re-export from
  `distributions.py`; import them from `aggregate.tail` (the single source).

Bivariate reporting redesign:

- **`BivariateAggregate.explain` deleted.** Its per-axis theory-vs-empirical
  validation is now carried by the `Agg` rows of `summary_df`.
- **`summary_df` rebuilt to the `Portfolio.summary_df` shape**: a shared `Freq`
  block, a `Sev` / `Agg` block per component, and a `total` `Sev` / `Agg` block
  (the genuine `X + Y` aggregate), with the eight-column validation view
  (`EX | Est EX | Err EX | <spread> | … | Sk | Est Sk`) and the same CV→SD switch
  when a component is a signed (`pnl`) axis. `Est` is populated only where it is
  observable from the joint density (the `Agg` rows); the `total Agg` empirical
  comes from the realised mixed moments.
- **New `dependency_df` property** owning the joint dependence: `cov` / `corr` /
  `tau` at `Sev` (per-claim joint severity) and `Agg` (realised aggregate) level.
  `tau` is the input copula's Kendall tau (a per-claim property).
- **`stats_df` slimmed** to pure per-component marginal moments; its old `joint`
  dependence block (`cov` / `corr` / `copula_tau` / `E[A0 A1]`) moved to
  `dependency_df`.

Docs pending a manual rebuild (per `CLAUDE.md`).

## 1.0.0a84

### `describe` → `summary_df`; deprecated alias `explain_validation` removed (breaking)

Two naming corrections for the v1.0 surface. **No deprecated aliases** — clean
breaks, since pre-1.0 is the one chance to make the names right.

- **`describe` property renamed to `summary_df`** on `Aggregate`, `Portfolio`,
  `Distortion`, and `BivariateAggregate`. The old name was the lone exception to
  the universal `<noun>_df` convention for DataFrame-returning members
  (`stats_df`, `density_df`, `tail_df`, `bs_window_df`, `reins_stats_df`,
  `tvar_info_df`, …) **and** it shadowed pandas' famous `DataFrame.describe()` —
  worse, as a *property* (no parens), so `a.describe()` raised `TypeError`.
  `summary_df` joins the `_df` family and removes the collision. The daily-driver
  table shown by `qd(obj)` / `_repr_html_` is unchanged; only the attribute name
  moves. Private workers (`_describe`, `_describe_signed`, `_compute_describe`)
  keep their names (internal). **Breaking**: `obj.describe` → `obj.summary_df`.
  *(`FourierTools.describe()` — a string method, not a DataFrame — is unrelated
  and unchanged.)*
- **`Aggregate.explain_validation()` / `Portfolio.explain_validation()` removed.**
  These were deprecated thin aliases (added in a82) for the
  `validation_explanation` property; per the no-alias policy they are deleted.
  Use the **`validation_explanation`** property. The module-level worker
  `aggregate.utilities.explain_validation(flag)` (takes a `Validation` flag) is a
  different function and is unchanged.
- Docs updated throughout (user guides, problem sets) to `summary_df`; the
  pandas `df.describe()` / `ft_obj.describe()` examples are untouched.

## 1.0.0a83

### config Phase 2 — numerics floors + stranded sizing knobs (breaking)

Completes the `dev/done/plan-config.md` work: the last hard-coded numerics floors and
a few half-migrated sizing knobs move out of `constants.py` into
`aggregate.config`, and one dead constant is removed. Wiring follows the
established pattern — a module-level `UPPERCASE` constant captured from
`get_settings()` at import — so call sites are unchanged; only the *source* of
each value moves.

- **`FT_NOISE_FLOOR` removed.** It had no live call site: the experimental
  `min|ft| < FT_NOISE_FLOOR` switch was abandoned during the numerics review with
  zero accuracy benefit (`dev/done/audit-numerics-2-findings.md`). Dropped from
  `constants.py` (constant + `__all__`), not migrated.
- **`[validation]` gains `aliasing_ratio` (10), `exeqa_noise_floor` (1e-4),
  `deficit_materiality` (1e-4).** Formerly `constants.ALIASING_RATIO` /
  `EXEQA_NOISE_FLOOR` / `DEFICIT_MATERIALITY`. The `deficit_materiality` 1e-4
  level is a judgment call still flagged for review (numerics-3); it is now a
  config edit rather than a code change.
- **`[discretization]` gains `window_log2_growth` (4), `window_slack_thick`
  (0.75), `concentration_cv` (0.1).** These window-sizing knobs were literals
  interleaved between fields already in `[discretization]` (`distributions.py`,
  `tail.py`); gathered for consistency.
- **`[bivariate]` gains `min_axis_log2` (4).** Formerly `bivariate._MIN_AXIS_LOG2`,
  now a first-class sibling of `total_log2` / `window_nines`.
- **Left module-internal (deliberately not config):** `copula._PPF_CLIP` (masked
  scratch guard) and `spectral._DISTORTION_DENSITY_N` (plot resolution) —
  implementation details, not user knobs.
- **Breaking:** `aggregate.constants` no longer exposes `FT_NOISE_FLOOR`,
  `ALIASING_RATIO`, `EXEQA_NOISE_FLOOR`, or `DEFICIT_MATERIALITY` (no re-export).
  Read them from `aggregate.config.get_settings().validation.*`. The annotated
  `config.default.toml` template documents all new keys; `tests/test_config.py`
  covers the new defaults, an override per touched section, and the constants
  removal.

`constants.py` is now down to the plotting figure defaults (permanently here),
the `Validation` / `DefectiveDistribution*` types, the structural `REINS_LABEL_*`
keys, and the `INFO_*` display convention.

## 1.0.0a82

### Consistent naming on the narrative / reporting surface — partly breaking

The narrative and reporting surface on `Aggregate` and `Portfolio` now follows
one rule: **`<item>_<aspect>`, the aspect is a noun, and every short/long
narrative is a property returning a string** (the `tail_class` /
`tail_description` / `tail_explanation` / `tail_df` family is the template).

- **Validation narrative** `explain_validation()` (verb-first, backwards from
  every other family) → **`validation_explanation` property**. The old
  `explain_validation()` method is **retained as a deprecated thin alias**
  returning `validation_explanation`, so existing callers keep working; prefer
  the property. The module-level worker `aggregate.utilities.explain_validation(rv)`
  (takes a `Validation` flag) is unchanged. On both `Aggregate` and `Portfolio`.
- **Reinsurance narrative** `reins_description(kind, width)` was a *method* with
  arguments → now a bare **`reins_description` property** (str, the
  `kind='both', width=0` text) for the consistent surface; the parameterized
  worker moved to the private `_reins_description(kind, width)`. **Breaking** for
  any external `a.reins_description('occ')` call (the property takes no args).
- **`concentration_p` → `cv`** (Aggregate **and** Portfolio `tail_df`). The
  opaque `concentration_p = Phi(mean/sd)` diagnostic (which saturated at ~1 for
  any real book) is replaced by the directly interpretable **coefficient of
  variation `cv = sd / mean`** (`inf` when `mean == 0`). The `tail_df` column,
  the `TailRow.cv` field, and `tail.concentration()`'s second return value all
  rename. The `concentrated` flag and its gate are unchanged. **Breaking**:
  public `tail_df` column rename; the tail narratives now read `… (cv=…)` /
  `It is concentrated: cv ~ …`.
- **`Portfolio.tail_df` `total` row completed.** `min` / `max` are now filled
  from the realised combine grid (`bs_window_df` `used` row) once sized (left as
  `n/a` before `update`), and the per-side tail classes are computed **per side**
  as the worst-of over the unit rows — so a non-negative book correctly reports a
  `bounded` **left** tail (it previously inherited the overall worst-of label on
  both sides, mislabelling the bounded floor).
- **`top=` → `x_max=`** in the `bs_description` / `bs_explanation` one-liners and
  prose (matching the `x_max` column already on `bs_window_df`). The internal
  `_bs_grid_top` helper and `top` locals are unchanged.
- **`bs_explanation` rewritten** to a single reporting template on both classes,
  with **"window width" replacing "span"** throughout: per-side tail classes and
  log2; (portfolio) per-unit tails and the candidate window widths (method of
  moments / RMS / sum / single big jump); the raw→dyadic `bs` rounding
  (portfolio only — the per-method aggregate sizer has no single pre-round `bs`);
  natural support bounds and concentration; the realised `x_min` / `x_max`; and a
  closing "increase log2" suggestion when the tail clipped.

## 1.0.0a81

### Rename portfolio sub-component `line` → `unit` — **breaking**

The oldest naming wart in the library is gone. *Pricing Insurance Risk*
(Mildenhall & Major, 2022) settled on **unit** as the generic term for a
portfolio sub-component (a line of business, geography, segment, account, or
reinsurance layer all read naturally as a "unit"). The half-renamed `unit_* →
line_*` pass-throughs are deleted and `unit` is now canonical throughout; the
`line_*` names are removed outright (no deprecation alias — this is a pre-release).

- **`Portfolio` storage / accessors** `line_names`, `line_names_ex`,
  `line_name_pipe`, `line_renamer` → **`unit_names`, `unit_names_ex`,
  `unit_name_pipe`, `unit_renamer`**. Accessing `line_names` now raises
  `AttributeError`. The thin `unit_names`/`unit_names_ex` pass-through properties
  were deleted (the names now belong to the real attributes); `n_units` stays.
- **Output-frame index/column label** `name='line'` → **`name='unit'`** on every
  pricing/quantile frame (`pricing_at`, `pentagon_at`, `price`, `var_dict`,
  `Pentagon.as_frame`, the single-`Aggregate` price frame). Any downstream
  `groupby('line')` / `.loc['line']` / `index.name == 'line'` must move to
  `'unit'`.
- **Keyword arguments** `line=` / `lines=` → **`unit=` / `units=`**:
  `Portfolio.pentagon_at(unit='total')`, `Bounds(unit='total')`,
  `Pentagon.as_frame(unit=)` / `from_row(unit=)`, and the private
  `_line_capital_at` → `_unit_capital_at(units=)`.
- **`BivariateAggregate`** follows suit: ctor `lines=` → `units=`, attributes
  `line_names`/`lines`/`_line_specs` → `unit_names`/`units`/`_unit_specs`, and the
  internal DecL spec key `'lines'` → `'units'` (parser producer + `decl_writer`
  round-trip moved together).
- Swept `pedagogy.py`, `results.py`, `bounds.py`, `pentagon.py`, and the test
  suite. Matplotlib (`linewidth`, `ax.lines`), plotly line specs, the actuarial
  *rate-on-line* / *loss on line* terms, optimisation *line search*, and
  source-text *line* machinery are deliberately untouched.
- Docs reference no renamed symbols, so no `:attr:`/`:meth:` cross-refs broke;
  LOB prose in `docs/` ("line of business" → "unit") is a pending author doc pass.

## 1.0.0a80

### Rename multivariate → bivariate (MV-7) — **breaking**

The final stage of the bivariate firm-up: the code is, and will remain, strictly
two-axis, so the public surface now says so. For three or more correlated lines
the path is independent components coupled by Iman–Conover and read back as a
sample (the "switcheroo"), not a native shared-frequency `rfftn` convolution.

- **DecL keyword** `multivariate` / `mv` → **`bivariate` / `bv`**, dropped
  outright (no deprecation alias). The old words now raise a parse error with a
  "Did you mean: bivariate?" suggestion.
- **Class** `MultivariateAggregate` → **`BivariateAggregate`**;
  `__repr__` / `info` say *bivariate*. `BivariateDistribution` keeps its name.
- **Module** `aggregate.multivariate` → **`aggregate.bivariate`** (reach it as
  `from aggregate.bivariate import BivariateAggregate`); internal transformer
  kind string `mvagg` → `bvagg`; grammar rules `mv_out`/`mv_body` → `bv_out`/
  `bv_body`.
- **Config** section `[multivariate]` → **`[bivariate]`**; `MultivariateSettings`
  → `BivariateSettings`; `get_settings().multivariate` → `.bivariate`.
- Swept `decl-testers.agg` / `cookbook.agg` / `examples.agg`, the reinsurance
  user guide, `dev/info-strings.rst`, the DecL cheat sheet, the Sublime syntax,
  and the regenerated grammar reference. The `5_x_multivariate.rst` technical
  guide is **unchanged** — it documents genuinely multivariate (t-dimensional)
  *frequency* theory, not the bivariate aggregate class.
- `BivariateDistribution` and the `occ_bivariate` engine were already named
  bivariate; no change there.

This is the last stage before the `1.0.0b1` candidate: the beta's public API
*is* the v1.0 bivariate API.

## 1.0.0a79

### Bivariate modelling features: shuffle-of-Min copula + clash statement (MV-6)

Stage MV-6 of the bivariate firm-up — the two modelling wins.

- **Shuffle-of-Min copula** (`aggregate.copula.ShuffleOfMin` +
  `CopulaShuffle`). A singular copula built by cutting the unit square into `n`
  equal vertical strips, permuting them, and optionally reflecting some — the
  graph of a measure-preserving bijection. Shuffles of Min are *dense* in the
  space of copulas, so they double as a flexible non-parametric dependence
  stress-test. **Programmatic-only** (no DecL keyword): build it as
  `CopulaShuffle(perm=[...], flip=[...])` and hand it to a bivariate
  (`mv.copula = CopulaShuffle(...); mv.update()`). Its `cdf` is exactly the
  `Copula.rectangle_pmf` interface, so marginals reproduce like any copula.
  Kendall's `tau` is computed **exactly** from the permutation/flips (`n=1`
  recovers M, `tau=1`, or W, `tau=-1`); a `sample` method gives exact draws.
- **`clash` statement** — `clash NAME na nb nc claims <A limit+sev> <B limit+sev>
  <freq>`. The natural cat-clash baseline: a shared event drives two perils, each
  triggered by an independent per-event Bernoulli, and `nc` is the expected count
  of joint-trigger (clash) events. `solve_clash_model(na, nb, nc)`
  (`aggregate.multivariate`) closes the independent-trigger 2×2 table
  (`n0 = na·nb/nc`) to derive the shared event count `n` and the two triggers
  `pa = (na+nc)/n`, `pb = (nb+nc)/n`; the two components become
  `dfreq [0 1] [1-p p]` factories under the **independent** copula on the shared
  frequency. New `CLASH.2` terminal (+ `ID` exclusion); round-trips through the
  DecL unparser as the `clash` form.

Note: a clash with **heavy** components (e.g. cv 2–3 lognormals) at a **large**
shared count can exceed the 2-D memory budget — one common `bs` per axis must
resolve both the severity and the much wider aggregate, so a coarse `bs` can
under-resolve the severity. This is the existing budget tension (not a clash
bug); the MV-4 `validation` row flags it (`check: marginal mean`). Raise
`update(log2=…)` or use lighter/bounded severities.

## 1.0.0a78

### Netceded view-pairs (MV-5)

Stage MV-5 of the bivariate firm-up: the occurrence netceded decomposition
generalises from the single `(ceded, net)` pair to **any two of {gross, ceded,
net}**. The three views satisfy `ceded + net = gross`, so exactly three
unordered pairs exist, each named by one DecL prefix.

- **Two new DecL prefixes** — `grossceded` and `grossnet`, siblings of the
  existing `netceded` (priority-2 terminals with the word-boundary lookahead,
  added to the `ID` exclusion list). Each takes one ordinary `agg` carrying
  occurrence reinsurance and builds the joint per-occurrence aggregate of the
  named pair.
- **`Aggregate.occ_bivariate(views=…)`** — grows a `views=('net', 'ceded')`
  parameter (each entry one of `'gross'` / `'ceded'` / `'net'`); the override
  signature is now `occ_bivariate(views, bs, log2_x, log2_y)` (one common `bs`
  + per-axis log2), replacing the view-named `bs_ceded`/`bs_net`/`log2_ceded`/
  `log2_net`.
- **Axis-order convention fixed.** The keyword names the pair **x-then-y**, so
  `netceded` → axis 0 = Net, axis 1 = Ceded; `grossceded` → (Gross, Ceded);
  `grossnet` → (Gross, Net). **Breaking:** the previous `netceded` /
  `occ_bivariate` axis 0 was Ceded; it is now Net (gross leads when present).
- **`build_netceded_joint(views=…)`** parameterised by the view pair — `gross`
  uses the identity image map (the gross loss itself), `ceded`/`net` use
  `occ_ceder`/`occ_netter`; all three rebucket onto the common-`bs` grid via the
  linear scatter (mean-preserving). `_netceded_theory` reads the matching
  `Gross` / `Ceded` / `Net` occurrence columns of `reins_stats_df`. `info`,
  `describe`, axis labels, and the contour plot follow the chosen pair.

The validation invariant holds for every pair: each marginal reproduces the
named standalone occurrence aggregate (means exact), and `gross − net` recovers
`ceded`.

Also: docs / syntax artifacts updated for the new keywords (the language
reference `ref_include.rst`, the reinsurance user guide, `dev/info-strings.rst`,
the Sublime syntax, `decl-testers.agg`), and a latent path bug in
`parser.grammar(add_to_doc=True)` fixed (it wrote `ref_include.rst` to
`src/docs/` instead of the repo-root `docs/` under the `src/` layout).

## 1.0.0a77

### Bivariate reporting surface (MV-4)

Stage MV-4 of the bivariate firm-up: the `MultivariateAggregate` reporting now
reads like `Aggregate` / `Portfolio`.

- **`info` rebuilt to the shared catalogue.** Dropped the bespoke f-string blob
  for the `aggregate.constants.info_row` / `INFO_NA` convention: a fixed row
  catalogue, every row always present in the same order, `n/a` before `update`.
  Mirrors the Agg/Port layout (name, mode, components, copula, shared frequency,
  claim count, padding, a per-axis block — name/kind/bs/log2/x_min/x_max — then
  correlation, copula tau, tail deficit, validation, id). Documented in a new
  **Bivariate** section of `dev/info-strings.rst`.
- **`explain`** (new) — the showpiece invariant: per-axis marginal moments
  (theoretical vs realized `mean`/`cv` with relative error) and the joint tail
  deficit. The `info` `validation` row is its one-line headline (deficit gate +
  a loose marginal-mean sanity; the exact, resolution-dependent errors live in
  `explain`).
- **`bs_window_df` / `bs_description`** (new) — the realized per-axis grid
  (`kind`, `bs`, `log2`, window, `clipped`), a two-row summary (the bv *measures*
  its grid, so there is no 1-D method ladder to report).
- **`tail_df` / `tail_description`** (new) — per-axis realized support + moments
  + a `right_heavy` flag.

`describe` / `stats_df` keep their per-component moment block (theoretical vs
empirical) and joint dependence footer (correlation, copula tau).

## 1.0.0a76

### Netceded axis sizing routes through `balanced_window` (MV-3)

Stage MV-3 of the bivariate firm-up. The netceded ``(ceded, net)`` joint is now
sized by the same *measure-don't-guess* primitive as the copula axes — the
second private sizer is gone, so both regimes share one path.

- **Deleted `multivariate.size_axis`** (the per-axis moment-quantile guesser,
  `cap_log2=14`). Netceded axes are sized by `balanced_window` on the realized
  occurrence margins (`reins_density_df['p_agg_ceded_occ']` / `['p_agg_net_occ']`,
  produced in one gross pass) — pure window selection, ceded/net being
  non-negative so the grids are 0-based.
- **One common `bs`, sized from the budget.** The comonotone `(c, n)` curve
  couples the axes, so they share a single `bs`; the linear scatter rebuckets
  the gross-sampled points onto it. The common `bs` is sized to fit
  `2**total_log2` (`~ sqrt(hi_c·hi_n)/2**(total_log2/2)`), **not pinned to the
  gross `bs`** — the gross grid auto-sizes fine to resolve the cession layer
  (e.g. 0.125), which is far too fine for the 2-D grid (it blew the budget and
  lost ~55% of the mass). Sizing from the budget also makes `Aggregate.occ_bivariate`
  and the DecL `netceded` form agree regardless of the gross grid each was built
  on. When a caller pins `bs`/`log2` past the budget the wider axis is clipped
  (a warned, reported deficit).

Both private sizers (`_size_axis`, `size_axis`) are now gone; both bivariate
regimes route through `balanced_window`.

## 1.0.0a75

### Bivariate windowing: honest, centred axis measurement

Three fixes so a symmetric axis windows symmetrically (motivating case: the
mean-0 `ssev` axis of `DISCRETE.2`, which came out as `[-29, +99]`).

- **Measure each marginal on an honest grid, decoupled from the loss/payoff
  trim.** The 1-D sizer protects one tail and trims the other (a *pricing*
  convention) — for a signed axis that clipped the cheap tail, biasing the
  equal-tail measurement up. A signed marginal is now rebuilt on a *centred*
  grid (slack both sides, sized from the SBJ-aware first build's realized extent
  so a heavy tail is still covered) before `balanced_window` reads it.
  Non-negative axes are unchanged (`MultivariateAggregate._measure_marginal_window`).
- **Back the measurement depth off the FFT noise floor:** `[multivariate].window_nines`
  12 → **9**. A 2-D marginal's far tail is numerical dust below ~`1e-10`, so
  measuring equal-tail quantiles deeper read noise (and skewed a symmetric axis).
  Per-axis deficit stays ~`1e-9`, far below target.
- **Centre the measured window in the (power-of-two) grid.** The grid width is
  quantised to a power of two, so a window leaves unavoidable slack; that slack
  is now split either side instead of piled above (which left a symmetric axis
  off-centre). A non-negative axis stays clamped at 0.

Net: `DISCRETE.2` axis B is now `[-64, +64]` centred on 0 (deficit ~3e-11). The
bivariate fit rounds `bs` *up* (`round_bucket`) for guaranteed coverage, never
to nearest: the grid length is a power of two, so a window lands on a
power-of-two-wide grid whatever `bs` is — rounding `bs` down can't tighten that,
it only clips or forces a larger `log2`. The dead space is split by the centred
placement instead.

## 1.0.0a74

### `round_bucket` ladder: no more 2.5x jumps

`round_bucket` (the library-wide "nice bucket size" rounder) used a 1-2-5 decade
ladder, so a raw `bs` of 3.4 jumped to **5** (a 2.5x overshoot; the `2→5` and
`20→50` gaps). It now rounds **up** to the denser ladder `{1, 2, 4, 5, 8} * 10**k`
for `bs ≥ 1` — every consecutive gap is ≤ 2x, so 3.4 → **4**, 5.5 → 8, 9 → 10.
`bs < 1` keeps the powers-of-two ladder (binary-exact for the FFT grid; already
≤ 2x). This is a blast-radius-wide change: any auto-sized `bs ≥ 1` may now land
on a finer/closer value (4 and 8 are reachable; the overshoot is bounded < 2x).

### Bivariate: per-axis `log2` / `bs` via `(x, y)` tuples

`MultivariateAggregate.update` (and therefore `build(..., log2=…, bs=…)`) now
accepts a 2-tuple to pin the axes independently:

- `log2=(log2_x, log2_y)` pins the per-axis grid `log2` (budget = their sum);
  a scalar remains the *total* budget split automatically.
- `bs=(bs_x, bs_y)` pins the per-axis bucket size; a scalar applies to both.

This lets you explore the split the auto-sizer doesn't — e.g. for the signed
`DISCRETE.2` book at a 2²⁰ budget, the auto split `(11, 9)` (which equalizes
`bs`) leaves the hard mean-0 axis under-resolved; `build(prog, log2=(9, 11))`
gives that axis the finer grid and its sd error drops ~3x at identical memory.
Tuples pass straight through `build`; they are bivariate-only (a tuple on a 1-D
`agg`/`port` is unsupported).

## 1.0.0a73

### Bivariate windowing: use the measured lower edge (no artificial 0-pin)

Follow-up to MV-2. `_size_axes` was pinning every non-negative axis origin to
`x_min = 0`, discarding the lower edge `balanced_window` had measured. For a book
whose mass lives far from 0 (low CV — e.g. a 500-claim compound of `10 * uniform`,
mean 2500, sd 129) that stranded the mass in the upper third of the grid and
wasted ~⅔ of each axis (≈89% of the 2-D cells).

- The axis origin is now the **measured** lower edge (snapped down to `bs`):
  negative on a signed axis, positive when the mass genuinely lives far from 0,
  and 0 only when the mass reaches the origin. The sole deliberate 0-pin is a
  `pnl` axis, whose loss has a known lower bound of 0 *and* whose `_affine_axis`
  relabel assumes a 0-based loss grid (the affine owns the tight P&L window).
- The FFT working buffer length `M` is now **decoupled** from the output length
  `N`: a compound is anchored at physical 0 (non-negative severity) or wraps its
  negatives (signed), so `M` is sized to reach from `min(0, x_min)` up to the
  window top, independent of the stored window `N`. This lets a tight far-from-0
  window shrink the stored grid without aliasing the upper tail.
- Example: the `10 * uniform` axis window tightens from `[0, 5115]` to
  `[1716, 3762]` with `bs` 5 → 2 (2.5× finer resolution at the same memory),
  deficit ~1e-10. The MV-2 signed bug book is unchanged.

## 1.0.0a72

### Bivariate axis sizing: measure, don't guess (MV-2)

Stage MV-2 of the bivariate firm-up (`dev/done/plan-mv.md`). Fixes the motivating
aliasing bug — a signed (`ssev`) bivariate book that lost **54% of its mass** to
wrap-around because each axis was sized for its *single-event* severity, not its
*marginal* support.

- **Axis sizing is now measured, not guessed.** `MultivariateAggregate._size_axis`
  (the old moment-window guess, `cap_log2=11`) is gone. Each component's
  standalone loss marginal is run first, an equal-tail `balanced_window` (MV-1)
  reads its support straight off the realized pmf, and the per-axis
  `(bs, log2, x_min)` is read off the measured window. The total grid budget is
  a single `update(log2=...)` input (default **20** = 2²⁰ cells, config
  `[multivariate].total_log2`); the per-axis split *falls out* of the measured
  supports (allocated so the two bucket sizes come out comparable). The measured
  window always covers the deep tail, so a smaller budget coarsens `bs` rather
  than clipping support — mass is conserved regardless of budget.
- **Signed axes no longer wrap.** `update_work` now lifts the 1-D
  `_fft_aggregate` `i0`/`j0` machinery to 2-D: the per-claim severity is laid
  into the padded buffer with physical 0 at index 0 (each axis's negative
  severity buckets wrapped to the top), the shared frequency applied
  elementwise, and the finished density rolled back onto each axis's output
  window. A signed marginal gets a centred two-sided window; the all-non-negative
  grid is byte-for-byte the original zero-pad path.
- The motivating book's per-axis deficit drops from **0.54 → <1e-6**; each
  marginal mean reproduces its standalone aggregate, and the marginal sd
  converges to the standalone as the budget grows (resolution-limited, not
  biased). New `[multivariate].total_log2` config knob (default 20).

This stage touches the **copula** sizing path only; `netceded` axis sizing
(`size_axis`) is unchanged here and is rerouted in MV-3.

## 1.0.0a71

### `.agg` library rationalization

Split the bundled `src/aggregate/agg/*.agg` files into a clear shipped set and a
temporary test-scaffold set, ahead of the alpha→beta cleanup.

**Shipped at v1.0 (4):**
- `examples.agg` — the curated default `build` library (also the 20-min intro and SPA dropdown). Unchanged.
- `actuarial-severity-curves.agg` — **renamed** from `other-distributions.agg`; the severity-curve reference, cited in docs.
- `decl-testers.agg` — **promoted** from `test_decl.agg`; the DecL *language*-stress corpus (kept in sync with the pytest tree).
- `cookbook.agg` — **renamed** from `testers.agg`; the broad insurance-useful worked-examples library (DRAFT; still to be de-duplicated).

**Temporary migration scaffolding (deleted before beta), now `_`-prefixed:**
- `_test_suite.agg` (from `test_suite.agg`) — the SLY-parity regression corpus + snapshot.
- `_test_suite2.agg` (from `test_suite2.agg`) — the splice/mixed-severity extension.

**Deleted:** `spa_examples.agg` (superseded by `examples.agg` for the SPA; its
content harvested into `cookbook.agg`) and `spa_examples-old.agg` (the legacy
walkthrough that originally seeded `cookbook`).

References updated across `tests/`, `src/aggregate/config.py`
(`TEST_SUITE_FILENAME`), `config.default.toml`, and the `scripts/` defaults. New
`tests/test_agg_libraries.py` is the permanent net that every shipped *user-facing*
library loads (parses + cross-resolves), so the `_`-prefixed scaffolding can
retire safely. Docs pending a rebuild.

## 1.0.0a70

### `balanced_window` + `Aggregate.focus` (bivariate sizing foundation)

Stage MV-1 of the bivariate firm-up (`dev/done/plan-mv.md`). A *measure-don't-guess*
windowing primitive, pure 1-D — the foundation the bivariate axis sizing (MV-2/3)
is built on. No bivariate behaviour changes yet.

- **`aggregate.utilities.balanced_window(ser, p, bs=None)`** — given a realized
  pmf series (`index = xs`, `values = ps`) and a discarded tail mass `p`, returns
  the equal-tail window `[q(p/2), q(1 - p/2)]`, optionally snapped to `bs`. The
  *post-calc* analogue of `estimate_agg_window`: it measures the window from an
  already-computed marginal rather than guessing from moments. **Balanced** means
  equal *probability* trimmed off each tail, so a signed P&L or skewed marginal
  stays centred on its mass. Reuses `make_var_tvar` for the quantiles so the
  convention matches `Aggregate.q` (`kind='lower'`). `p` is the literal discarded
  mass (e.g. `1e-6`), not a coverage — no `>1 → nines` reading.
- **`Aggregate.focus(p=1e-6)`** — a thin, no-recompute re-slicer: runs
  `balanced_window` on the realized `p_total` and returns the central
  `density_df` slice holding `1 - p` of the mass. Does not mutate the aggregate.

## 1.0.0a69

### DecL statement separation: blank line or `;`, no more `\`

How a DecL *program* splits into individual *statements* changed. The old rule —
"one statement per line; an indented or `\`-continued line folds onto the
previous one" — is replaced by a markdown/Python hybrid:

- **A blank line** (empty or whitespace-only) separates two statements (the
  markdown paragraph model). A statement may now span as many lines, with
  whatever indentation, as you like — a multi-line `port` just needs its units
  in one paragraph (no blank line between them).
- **A `;` at the end of a line** also ends a statement (the Python model), so
  dense one-statement-per-line lists stay legal. The `;` inside
  `hints{key=value;}` / `note{...}` is untouched (those end a line with `}`).
- **Comments are transparent** — a `#` / `//` comment never separates
  statements, so you can comment out or annotate a clause inside a multi-line
  statement (e.g. a reinsurance line) and the statement stays intact. The
  corollary: a comment alone no longer separates two statements; use a blank
  line or a `;`.

**Breaking changes:**

- **The `\` line-continuation is removed.** A statement spans multiple lines for
  free, so a stray backslash is now a **lexer error** (dropped from the
  `decl.lark` `%ignore` class) rather than silently ignored — it surfaces the
  copy/paste confusion the old behavior hid.
- Two statements on adjacent lines with **neither** a blank line **nor** a `;`
  between them now fold into one statement (usually a loud parse error). Insert a
  blank line or a trailing `;`.
- `Underwriter.to_agg` now writes entries blank-line separated; files written by
  older versions that packed statements one-per-line with no separator must be
  re-exported or hand-separated to re-load.

The single owner of the rule is `UnderwritingLexer.preprocess`
(`aggregate.parser`); `decl_writer._split_statements` mirrors it. Comments are
stripped before the vector-bracket step, so a stray bracket in a comment no
longer unbalances preprocessing. The bundled `.agg` libraries and the DecL
documentation were migrated. Docs pending a rebuild.

## 1.0.0a68

### `approximate()`: one fit core, symmetric guard, honest reflected-fit errors

`Aggregate.approximate()` / `Portfolio.approximate()` and the `approximate` DecL
keyword had **two** moment-fit implementations. The method path
(`approximate_from_mcvsk`) was the unguarded one: a symmetric or left-skewed
distribution hit `sln_fit`/`sgamma_fit` with non-positive skew and got a
degenerate `(-inf, inf, 0)` → a silent `nan` distribution (e.g. a 12-dice sum,
`skew = 0`, with the default `slognorm`).

- **Single fit core.** `_approximate_sev_kwargs` is now the one place the family
  fits and guards live (generalized to all five families: `norm` / `lognorm` /
  `gamma` / `sgamma` / `slognorm`). `approximate_from_mcvsk` is a thin **output
  adapter** over it (`scipy` / `sev_kwargs` / `sev_decl` / `agg_decl` /
  `Aggregate`), so the two `approximate` surfaces and the DecL keyword share one
  implementation. No second path to drift.
- **Symmetric → normal, with a warning.** A *shifted* family (`slognorm` /
  `sgamma`) requested for a (near-)symmetric distribution now returns its normal
  limit (the mathematically correct answer) and emits a `UserWarning` from the
  interactive `.approximate()` method (pass `approx_type='norm'` to select it
  explicitly). The declarative DecL/constructor path keeps degrading silently.
- **Reflected (left-skew) fits error honestly where they can't be drawn.** A
  left-skewed fit reflects (`sev_reflect`); that has no native frozen `scipy` or
  one-line DecL form, so `output='scipy'` / `'sev_decl'` / `'agg_decl'` raise a
  clear `ValueError` pointing to `output='sev_kwargs'` / the default `Aggregate`
  object (which do reflect) or `approx_type='norm'`.
- **`approximate('all')`** stays quiet about degeneration and skips any family it
  can't represent for the given distribution/output (e.g. reflected + `scipy`),
  returning the admissible subset.

## 1.0.0a67

### Fix: `Aggregate.approximate()` method was shadowed by a same-named attribute

The `approximate=` constructor keyword added in `dev/done/plan-approximate.md`
stored its value as `self.approximate`, which **shadowed** the existing
`Aggregate.approximate()` method (the method-of-moments surrogate factory, the
parity-partner of `Portfolio.approximate`) on every instance — `agg.approximate`
was the string `'exact'`, and calling it raised `'str' object is not callable`.

- **The build-time choice is now the attribute `Aggregate.approximation`** (a
  noun), leaving `approximate()` callable as before. It is **falsey (`''`) for an
  exact freq×sev convolution** and the fit kind (`'sgamma'` / `'slognorm'`)
  otherwise, so `if a.approximation:` reads as "is this object a moment-match
  surrogate?". The `approximate` DecL keyword, the `approximate=` kwarg, and the
  spec key are unchanged; `describe` still shows the `approximate` row.
- `Portfolio` was never affected (no `approximate` attribute) and is unchanged —
  the two classes again expose the same `approximate()` method.
- **Naming-vetting rule** added to `CLAUDE.md`: a new instance attribute set in
  `__init__` silently shadows a method of the same name, so new public
  method/attribute/kwarg names must be `rg`-checked against the existing surface
  at planning time (noun for stored value, verb for action).

## 1.0.0a66

### Portfolio windowed combine 1P — Portfolio MM + single-big-jump look-through

Final task of `dev/done/plan-bucket-window-2.md` (Round 3). The portfolio
combine grid is reconciled with the a62–a65 univariate (`Aggregate`) sizer. The
old `best_window` sized the shared grid from a **linear sum of per-unit window
widths**, which overstates the bulk by `sqrt(k)` for `k` iid units (it ignores
diversification) and double-counts every unit's far-tail allowance. The combine
now mirrors the per-aggregate *bulk / extent* split:

- **Bulk from Portfolio MM.** The shared `bs`/span is sized from the **exact
  total moments** (`agg_m`, `agg_sd`, `agg_skew`; cumulants add under
  independence) fed straight into the same `estimate_agg_window` the single
  aggregate uses — never a per-unit width combine. A diversified-iid book now
  sizes `~sqrt(k)` tighter than before (the headline win).
- **One portfolio single-big-jump extent floor** (`Portfolio._single_big_jump_window`):
  a *look-through* to the units, `sbj_hi_port = agg_m + max_k(sbj_hi_k − ES_k)`
  — the heaviest unit's one big claim on the combined bulk (**max**, not sum, so
  the per-unit a59 extents are not double-counted). Self-activating: a thin /
  well-diversified total leaves the grid unmoved.
- **Resolution floor** stays `min_k bs_k` (a unit's lattice must survive), and
  `bs` is `round_bucket`-ed **once** at the top (carry raw, round once).
- **Windowed non-signed origin (Plan B).** A *concentrated* non-signed total
  whose mass clears 0 (high-frequency / tiny-cv, e.g. `Poisson(100000)`) is now
  **windowed** — the shared grid starts at `x_min > 0`, routed through the same
  roll-combine path as a signed book — instead of wasting the whole lower grid
  on a forced 0-based placement. The per-aggregate heavy-severity "Regime B"
  limitation does **not** bind the combine (each unit keeps its own 0-based
  severity grid; only the convolved *total* is relabelled).
- **`x_min` policy:** the combine ships with the algorithm's `x_min`; back-compat
  with old published grids is **not** a ship gate. The numerics-2/3
  origin-invariance (`sum_i kappa_i(x) == x`, moments, mass) holds on whatever
  grid is picked (verified in the 1P tests).
- **Reporting parity (`[bs-reporting]`).** `Portfolio.bs_window_df` gains four
  inspectable combine-candidate rows — `mm` (the live MM bulk), `rms`
  (RMS-of-windows reference; the `mm − rms` gap reads as the
  skewness/diversification adjustment), `sbj` (the look-through), and `sum` (the
  legacy linear bound) — plus the `log2_need` / `clipped` columns (parity with
  `Aggregate`). New `Portfolio.bs_explanation` (verbose grid prose) and
  `Portfolio.tail_df` (per-unit + worst-of `total`). A windowed/signed combine
  that clips the tail records `Portfolio._bs_clip` and warns once (the
  speculative phase-1 pre-pass is silenced).

**As-built note.** The review's `mm ≤ rms ≤ sum` ordering holds robustly for a
diversified light-tailed book but can legitimately invert (`mm > rms`) for a
*concentrated subexponential* total, where MM is tail/skew-aware while the RMS
reference is symmetric-normal; the tests assert the ordering only in the clean
regime, and `bs_explanation` flags an inversion rather than treating it as a bug.
The multi-driver pooled root-find (`sum_k E[N_k](1−F_k(x)) = 1−p_star`) is left
as a documented refinement; the `max_k` look-through is a safe dominant-unit
bound that the FFT doubling-padding absorbs.

## 1.0.0a65

### Bucket-selection 1A-bucket — making the grid choice legible (`[bs-reporting]`)

Fourth task of `dev/done/plan-univariate-bucket.md`. The bucket-grid decision is the
#1 numerical choice; this surfaces it. Pure reporting -- no change to the grid.

- **`_bs_window_df` enriched** with two derived columns: `log2_need` (the log2 a
  method's window needs at its own `bs` -- a row with `log2_need > log2` was
  capped) and `clipped` (the estimated far-tail mass dropped, on the `used` row).
- **`bs_window_df`** -- a curated, read-only public property on **`Aggregate`**
  (method / window / grid / applies / selected / note) and **`Portfolio`** (one
  row per unit + the realised `used` grid). Folds in TODO **H10**; the private
  `_bs_window_df` keeps the expert extras (`coverage`, `W`).
- **`bs_description` / `bs_explanation`** -- short and verbose narratives of the
  grid choice (`Aggregate`; `bs_description` also on `Portfolio`). The short line
  is the winning method + `(bs, log2, x_min)` + grid top + any clip; the verbose
  prose adds the aggregate tail one-liner, which methods applied and why the
  winner won, and how to widen a clipped tail. The ANSI-coloured variants are the
  module functions `aggregate.distributions.bs_describe` / `bs_explain`
  (`color=True`), mirroring the tail narrative.
- **No more double warning.** A book whose far tail is clipped at sizing time
  (item 6's `DefectiveDistributionWarning` + structured `_bs_clip`) no longer
  *also* emits the generic update-time "PMF deficit" warning -- the sizing
  warning is the same mass with actionable advice (the exact `log2` to raise to),
  so the deficit warning is suppressed when `_bs_clip` is set. The most common
  heavy book (`100 claims lognorm cv 2`) now warns once, not twice.

## 1.0.0a64

### Bucket-selection 1A-bucket — wiring the tail report into sizing (`[use-selection]`)

Third task of `dev/done/plan-univariate-bucket.md`. The bucket sizer (`_bs_window`)
now consults the layered tail report (`_loss_tail_classes`, `concentration`)
instead of ad-hoc geometric proxies. Six changes, each byte-stability-gated
against the full suite and `test_bucket_sizing.py`:

1. **Thickness-gated single-big-jump floor.** The SBJ extent floor now fires
   only for a genuinely **thick** (subexponential-or-heavier) tail -- the loss
   right tail for a positive severity, the reflected left tail for a signed one
   (`is_thick`). A no-op for thin tails (the MoM window already covers them),
   now explicit and cheaper.
2. **Power-law / infinite-variance: honest truncation.** An infinite-variance
   (power-law) aggregate has no finite deep quantile to size to. The old
   `recommend_bucket` fallback **re-raised** on infinite cv (a crash for an
   unlimited Pareto); it is replaced by `_reachable_bulk_high`, which sizes the
   reachable bulk to a moderate `bucket_sizing_p` coverage from the severity's
   actual quantile, **warns**, and accepts the far tail as a reported deficit --
   exact below the truncation, never normalized back in, no `alpha`-quantile
   chase.
3. **Thin-left-gated windowed left-lift (the asymmetric window).** A
   concentrated heavy-severity book (the former "Regime B", which stayed 0-based
   and clipped the tail) is now **reclaimed**: the windowed upper edge is floored
   by the single-big-jump reach, which grows the grid -- and its severity
   discretisation extent -- enough that a single heavy occurrence fits and the
   thick right tail is captured. Lifting `x_min` off 0 is gated on a thin left
   tail. Selection is relaxed so a (possibly coarser) windowed grid wins when it
   captures a reach the 0-based pick clips.
4. **Tail-aware padding / slack.** The fixed `window_pad_skew` split is replaced
   by a tail-driven one: an **asymmetric** band puts ~3/4 of the power-of-2 slack
   on the thick side (`WINDOW_SLACK_THICK`); a **symmetric** band centres, with
   the loss/payoff convention demoted to a tie-breaker. (The per-edge window
   *coverage* still follows the convention.)
5. **Concentration from the report.** The windowed-eligibility gate is now the
   conservative `concentrated` flag (`agg_cv < CONCENTRATION_CV`, ~0.1), the
   single source of truth, replacing the looser geometric `w_lo > 0` (~0.21).
   Borderline books (`cv` in ~[0.10, 0.21]) revert to the 0-based grid.
6. **Far-tail clip → warning.** The positive-tail clip is promoted from a silent
   `logger.info` to a visible `DefectiveDistributionWarning`, with a structured
   `Aggregate._bs_clip` field (reach, grid top, `log2` needed, estimated clipped
   mass via `_clipped_mass_estimate`) for the validation / bs report.

Also: an informational **`severity (net occ)` overlay row** in `tail_df`
(`occ_net_severity_row`) reporting how occurrence reinsurance reshapes the
retained per-occurrence tail (bounded when a top layer cedes 100% to infinity,
else the gross tail) -- the sizer still works on the gross severity. The
signed-padding placement was verified (a two-sided signed book's negative and
positive reaches coexist in the FFT buffer without collision).

`Aggregate.tail_report` (added experimentally in a63, same-day) is **removed**:
the ANSI `color=` option lives only on `aggregate.tail.describe_rows` /
`explain_rows`, the future terminal/HTML hook; the plain `tail_description` /
`tail_explanation` properties are the public surface.

## 1.0.0a63

### Comprehensive scipy severity tail tables (the family classifier)

Populates `aggregate.tail`'s family classifier from a reconciled survey of every
`scipy.stats` continuous distribution's tail behavior (two independent
derivations cross-checked, `C:/s/AI/notes/2026-06-17-probability-distribution-tails/integrated.md`).
A user severity from any standard scipy family now classifies exactly instead of
falling back to `UNKNOWN`:

- **`SCIPY_SEV_TAIL`** expanded to ~45 fixed-class families (super-exponential:
  `chi`, `maxwell`, `rayleigh`, `nakagami`, `rice`, `halfnorm`, `foldnorm`,
  `gompertz`, `kstwobign`, `exponpow`, …; exponential: `chi2`, `erlang`,
  `fatiguelife`, `wald`, `invgauss`, `genexpon`, `geninvgauss`, `ncx2`,
  `recipinvgauss`, `halflogistic`, `hypsecant`, `dgamma`, `genlogistic`,
  `norminvgauss`, …; subexponential: `gibrat`, `johnsonsu`, `powerlognorm`).
- **`_POWER_LAW_ALPHA`** expanded with the correct shape-slot α for `loglaplace`,
  `nct`, `halfcauchy`, `foldcauchy`, `skewcauchy`, `kappa3`, `mielke`,
  `betaprime`, `f`, `jf_skew_t`, `levy`, `alpha`, `rel_breitwigner`, `landau`.
- **Parameter-aware** families added to the classifier: `gengamma` (Weibull
  exponent `c`; `c<0` → power-law), `exponweib`, `gennorm` / `halfgennorm`
  (`β`), `dweibull`, `tukeylambda` (`λ>0` bounded / `=0` exp / `<0` power-law),
  `levy_stable` (`α<2` power-law). The shared `_weibull_shape` /
  `_family_right_class` / `_family_sides` helpers are now the single source for
  `classify_severity` and the `tail_df` per-side classes (no parallel table).
- **`_SEV_LEFT_CLASS`** records the asymmetric two-sided families whose left tail
  differs from the right (`gumbel_r`, `gumbel_l`, `loggamma`, `moyal`,
  `exponnorm`, `landau`, `crystalball`) so a signed/two-sided severity reports a
  correct per-side `tail_df`.
- **`bounded` is now robust to any finite scipy support**: `_severity_bounded`
  falls back to `fz.support()` finiteness (spec-only), so finite-support families
  not in `_BOUNDED_SCIPY_SEVS` (`argus`, `bradford`, `gausshyper`, `johnsonsb`,
  `irwinhall`, `powerlaw`, `loguniform`, `genhalflogistic`, `tukeylambda` λ>0, …)
  classify as bounded without enumeration.

The four reconciled discrepancies between the two source views (`exponpow`
right-tail = super-exponential; `genhalflogistic` bounded; `studentized_range`;
`truncnorm`) are documented in `integrated.md`. Bucket selection does not read
the tail classifier yet, so this remains report-only.

## 1.0.0a62

### Bucket-selection 1A-bucket — the narrative tail report (`[tail-narrative]`)

Second task of `dev/done/plan-univariate-bucket.md`. The `tail_description` (short,
aligned) and `tail_explanation` (verbose) properties now narrate the **layered**
a61 `tail_df` — support and per-side tail class, not the old single-rung
sentence. Built from one shared `Aggregate._tail_rows()` (the same `TailRow`
list behind `tail_df`), so frame and prose never drift.

- **`tail_description`** — three aligned lines (frequency / severity /
  aggregate), e.g. `aggregate tail   [0, inf), subexponential right tail; not
  concentrated (P>0=1.00)`. Also feeds `Aggregate.info()`.
- **`tail_explanation`** — bottom-up prose: the per-component severity
  breakdown and blend, the single-big-jump mechanism (or the frequency driver)
  for a thick right tail, the power-law moment failure, the heavy-left
  (signed / `pnl`) sizing note, and the concentration.
- **`Severity.tail_description`** — one line (`lognorm, [0, inf), subexponential
  right tail`), from the same `severity_tail_row`. **`Frequency.tail_description`**
  — the family count class (support depends on exposure, so the full count
  support shows only in the aggregate's `tail_df`).
- **ANSI option** — `describe_rows` / `explain_rows` take `color=True` to
  emphasise thick (subexponential-or-heavier) tail classes in bold-red for a
  TTY; the properties stay plain (so `info()` is plain).

`aggregate.tail` text builders rebuilt around the rows: `describe_row`,
`describe_rows`, `explain_rows` replace the old `TailInfo`-based `describe_lines`
/ `explain` (which carried the now-dropped log-concave / single-rung phrasing).
**Byte-stable** — narrative only; selection still does not read the report.

## 1.0.0a61

### Curated `examples.agg` example library + `build` default

A new `src/aggregate/agg/examples.agg` (Version 1) is the curated, public-facing
DecL example set: ~32 hand-picked programs (2 severities, 2 distortions, ~21
aggregates, 8 portfolios) organized A–J by what each illustrates, spanning the
DecL-capability and numerical-character axes. It is the single source for the
default `build` knowledge base, the twenty-minute intro, and the `aggregate_api`
SPA examples dropdown.

- **`build` default database changed** from `test_suite` to `examples`
  (`config.py` `BuildSettings.databases`, `config.default.toml`). Out of the
  box, `build` now loads the curated set rather than the historical reference
  suite. Set `[build] databases = ["test_suite"]` to restore the old default.
- **New `testers.agg`** — the back-room comprehensive-coverage companion (WIP
  dumping ground), seeded from the legacy feature walkthrough; pending merge
  with the `test_suite` family (TODO T1).
- **Tests:** `test_decl_unparser` now parses its `test_suite` corpus through a
  dedicated `Underwriter(databases='test_suite')` rather than the `build`
  singleton (decoupled from the default-database choice); `test_config` asserts
  the new default.
- **TODO B4** logged: zero-truncated/zero-modified frequency is broken
  (`poisson zt` raises; `zm` semantics inverted) — redesign to take the base
  mean and ship forward shift helpers. The two ZM/ZT examples are commented out
  in `examples.agg` meanwhile.
- **Companion (`aggregate_api`):** the examples route reads the bundled
  `agg/examples.agg` and folds `\`-line continuations, so multi-line `port`
  programs are captured whole.

### `tail_df` schema revision — support + per-side tail class

Reworks the a60 `tail_df` to report **support and tail shape** (and nothing
that belongs to grid selection). Per author review:

- **`min` / `max` are now the structural support** (smallest / largest
  *attainable* value; `-inf` / `inf` at an unbounded end), not a
  method-of-moments reach. So `bounded` is the self-consistent `min` and `max`
  both finite, the aggregate of a fixed 3 × dice `[1..6]` reads exactly
  `[3, 18]`, a signed `dsev` book reads its exact two-sided support, and a `pnl`
  book reads `[-inf, premium]`. The numeric grid *reach* moves to the bucket
  report (`bs_window_df`, a later task).
- **`left` / `right` thick-thin become `left_tail` / `right_tail` full tail
  classes** (`bounded` / `super-exponential` / `exponential` /
  `subexponential` / `power-law`): a finite support end is `bounded` (a hard
  boundary, no tail), an infinite end carries the family decay rung. So a
  lognorm reads `left_tail = bounded`, `right_tail = subexponential`; a `pnl`
  book reads `right_tail = bounded` (premium cap), `left_tail = subexponential`
  (the loss right tail, reflected). The sizer's thick/thin is the derived
  `is_thick(right_tail)`.
- **`concentration_p` is now `Phi(mean / sd)`** — the normal-approximation
  probability the aggregate is positive (the band clears 0), a genuine p-value
  in `(0, 1)` — replacing the a60 `1 / cv` sd-count. The conservative
  `concentrated` gate (`cv < CONCENTRATION_CV = 0.1`) is unchanged.
- **Dropped `tail_class`, `alpha`, `log_concave` columns.** None drives
  selection; the actionable power-law fact moves into `note` as
  `"power-law, alpha=1.5, infinite variance"`. Final columns: `family, min,
  max, left_tail, right_tail, bounded, concentrated, concentration_p, note`.

`aggregate.tail` API updated accordingly (`TailRow` fields; `concentration(m,
sd)`; `build_tail_rows` takes `agg_m` / `agg_sd` / `agg_reflect` / `agg_shift`).
Still **byte-stable** — selection does not read the report yet.

## 1.0.0a60

### Bucket-selection 1A-bucket — the layered thick/thin tail report (`tail_df`)

First task of `dev/done/plan-univariate-bucket.md` (`[tail-report]`). A new
first-class, **spec-only** report of tail shape, built bottom-up across the
layers that determine grid choice. `Aggregate.tail_df` returns a DataFrame with
one row per layer — `frequency`; one per severity mix component (`comp0` …); the
combined effective `severity` (when there is more than one component); and the
`aggregate` — carrying the structural fields the sizer reasons about:

- `min` / `max` reach (claim-space layered-loss support per component;
  method-of-moments reach for the aggregate, two-sided for a signed book, `nan`
  for an infinite-variance power-law);
- `bounded`, and `left` / `right` **thick-thin** labels (thick ⇔
  subexponential-or-heavier; a non-negative layer is thin-left; `UNKNOWN` is
  conservatively thick);
- the `tail_class` rung, power-law `alpha`, and `log_concave`;
- (aggregate row only) the conservative `concentrated` flag and its
  `concentration_p` sd-margin, using the tighter `CONCENTRATION_CV = 0.1` cut.

The report carries **two facts** for a capped heavy family — its base-family
thickness and its structural bound — so a thick base capped by a finite `limit`
/ splice is reported as effective-bounded with a `"subexponential base, capped
at L"` note. No numeric tail estimator: classification is family-lookup plus
structure (the grid the estimate would need is the very thing being chosen).

New surfaces in `aggregate.tail`: `TailRow`, `is_thick`, `thickness_label`,
`severity_support`, `concentration`, `build_tail_rows`, `tail_frame`,
`CONCENTRATION_CV`. **Pure addition — byte-stable**: nothing in selection reads
the report yet (that is the next task, `[use-selection]`).

## 1.0.0a59

### Bucket-window 1A-fix — single-big-jump extent floor (heavy / signed severities)

Second part of `dev/done/plan-bucket-window-2.md` (§1A-fix). The 3-moment
method-of-moments output window is blind to a tail the first three moments do
not capture. Two failure faces, one root cause, are addressed by flooring the
selected window's *extent* (not its resolution) by a single big claim on an
otherwise typical bulk — for a subexponential severity the aggregate's far tail
is `P(S>x) ≈ E[N]·P(X>x)`, so the severity is probed at the `E[N]`-adjusted
level `p** = 1 - (1-p*)/E[N]` and the extent floored at `ES - μ_X + q_X(p**)`.

- **Signed severities — correctness (catastrophic case fixed).** A signed
  severity (e.g. `100 - lognorm 10 cv 2.5`) can have positive aggregate skew
  while its reflected tail reaches far below 0. The MoM window then misses the
  reach entirely and the severity *wraps the FFT buffer* (aliasing), losing
  ~47% of the mass and returning a garbage law. The grid now always covers the
  single-big-jump reach `[sbj_lo, sbj_hi]` (width ≥ severity reach), keeping the
  bulk `bs` when the log2 budget allows and coarsening `bs` within the log2 cap
  otherwise — aliasing is corrected at any log2 (mass recovered to 1).
- **Positive heavy severities — refinement.** A heavy unlimited severity's MoM
  window under-reaches the true right tail (e.g. `5000 claims lognorm 100 cv 2`
  clips ~3.9e-7 of the priced tail at the default grid). The window now extends
  up to the single big jump **when it fits at the bulk `bs` within the requested
  `log2`** (so a larger `log2` is captured fully and finely); at a constrained
  `log2` the MoM window is kept (clipping a tiny far tail beats coarsening the
  bulk to uselessness — e.g. a 5-claim, mean-50 book whose tail reaches 47k).
  The existing `DefectiveDistribution` warning still flags a material clip.
- **`log2` honored, no silent memory growth.** The single-big-jump floor never
  grows `log2` past an explicit / hinted / default request and never coarsens a
  pinned `bs`; light / thin / bounded / concentrated and windowed books are
  byte-stable (the floor's `max`/`min` are no-ops). New
  `[discretization] sbj_tail_floor` (default `1e-14`) caps how deep the severity
  is probed (guards `q_X(p**) -> inf` for a large `E[N]` on an unbounded sev).
- **Inspectability.** `_bs_window_df` gains an `sbj` row — the grid the
  single-big-jump extent implies (origin / `bs` / `log2`, sized like every other
  method row, so it never reads NaN); the selected method's `note` records when
  the floor binds. When a positive heavy tail is clipped at a constrained
  `log2`, a `logger.info` reports the reach and suggests the `log2` that would
  capture it.

New helpers `Aggregate._single_big_jump_window` and `._severity_low_estimate`.
Byte-stability of the `test_suite.agg` snapshot and the numerics-2/3 regression
gates is preserved.

Deferred: the planned signed-only kurtosis *diagnostic* is dropped — the
true-law compound kurtosis of the motivating signed case is modest (~4.8, not
the ~737 the plan cited, which was the empirical kurtosis of the already-aliased
distribution), so a kurtosis-vs-fit test does not fire. The aliasing it was
meant to surface is now fixed at source, and a material positive-tail clip is
already flagged by `DefectiveDistribution`.

## 1.0.0a58

### Bucket-window 1A — convention-aware aggregate output windowing

First step of `dev/done/plan-bucket-window-2.md` (Step 1, part 1A: the
`Aggregate` sizer; part 1P, the `Portfolio` combine, follows). The automatic
output window for a concentrated aggregate (mass band clears 0) is now
symmetric in its estimator and oriented by the sign convention, and the band
is placed sensibly in the grid rather than jammed against the floor.

- **Per-edge window coverage** (`estimate_agg_window` gains `p_lo` / `p_hi`).
  A windowed book covers its *protected* edge deep (anti-clip) and trims its
  *cheap* edge shallow (anti-waste): for a loss the upper (priced right) tail
  is protected and the lower trimmed; a payoff mirrors. New
  `[discretization] window_nines_trim` (default 6) sets the trim depth;
  `window_nines` (12) remains the protected depth. Backward compatible —
  callers passing only `p` are unchanged.
- **Balanced padding** (`[discretization] window_pad_skew`, default 0.1).
  The power-of-2 slack around a windowed band is split `f = 0.5 -/+ skew`
  below the band (loss -> more room on the right, payoff -> mirror) instead of
  all above. Only the *windowed* row is rebalanced; ordinary, exact-discrete
  and bounded books keep their band-bottom origin.
- **Convention from `value_type`, with override.** The windowed skew branches
  on `_is_loss_value` (never the label string); `update(window_convention=...)`
  overrides per call. Defaults derive from the aggregate's `value_type`.
- **Relaxed windowed-selection gate** (`<` -> `<=`). A band that clears 0 now
  wins on *placement* -- reclaiming the empty `[0, x_lo)` region and balancing
  the slack -- even when `bs` is unchanged, not only when strictly finer. The
  severity-fit guard is unchanged, so the change cannot select a windowed grid
  the severity does not fit.
- **Explicit Regime-B (heavy severity) branch.** When the mass band clears 0
  but a single severity overflows the windowed extent (the benign FFT wrap is
  invalid), the book keeps the 0-based grid and a `logger.info` explains why
  (expected, not defective -- no warning). This replaces the previous silent
  fall-back.

Byte-stability: ordinary aggregates (`agg_cv > 1/z`) and heavy-severity
Regime-B books are unchanged -- the full `test_suite.agg` snapshot and the
numerics-2/3 regression gates pass untouched. The only grids that move are
genuinely windowed (Regime-A) books, which gain a centred placement.

Deferred (documented in `dev/TODO.md`): Regime-B *clip remediation* (deepening
upper coverage / growing `log2` to capture the heavy right tail, e.g. the
`5000 claims cv 2` 3.9e-7 top-bucket clip). It needs a waste/clip threshold to
separate genuinely-wasteful Regime-B books from ordinary ones that merely have
`w_lo > 0`, and so deserves its own validated pass rather than risking the 1A
byte-stability guarantee.

## 1.0.0a57

### Numerics-3 — distortion spine (one Choquet engine; linear/lifted unified)

Third plan of the numerics program
(`dev/done/plan-numerics-3-distortion.md`). All distorted pricing routes
through one exact-discrete Choquet helper; the linear and lifted
allocations become one builder; the `T.*`/`M.*` column families are gone.
Step-0 audit with measured verdicts in `dev/done/audit-numerics-3-findings.md`.

- **One Choquet helper.** `spectral.choquet_weights(x, p, g)` computes the
  exact distorted atom weights `gp = g(T) − g(S)` (`T = P(X ≥ x)`, the
  strict shift of `S = P(X > x)`; a pmf on a clean law) and is the *only*
  place they are computed: `Distortion.price` (both `method='dx'` and
  `'ds'` now return the identical `Σ min(x,a)·gp` value), `make_q`,
  `Aggregate.apply_distortion` and `Portfolio._build_augmented` all route
  through it. Choquet values are capped dot products carrying the grid
  origin — exact on signed, shifted and nonuniform supports. The layer
  form `x0 + Σ g(S)·Δx` survives only as an internal reconciliation
  assert.
- **Deficit policy** (new `DefectiveDistributionError`,
  `constants.DEFICIT_MATERIALITY = 1e-4`): pmf deficits at the validation
  noise floor are renormalized away; small FFT-truncation losses (already
  advertised by `DefectiveDistributionWarning`) are parked per
  `S_calculation` (forwards: top atom, backwards: bottom atom — the two
  agree to tolerance on a clean law, asserted); material deficits raise
  unless `allow_deficit=True` is passed explicitly.
- **`view × value_type` pricing axis.** `Distortion.effective_g(view,
  is_loss_value=...)` resolves the 2×2 by XOR (dual iff `bid` XOR payoff
  role), branching on the canonical `_is_loss_value` flag, never the
  configurable label strings. A payoff-role object prices as the dual of
  the matching loss object, surviving label reconfiguration.
- **Unified linear/lifted builder.** `apply_distortion(distortion, *,
  view, S_calculation, allocation='lifted', allow_deficit=False)` builds
  one column schema for both methods in one O(n) sweep across all asset
  levels: `exag_i = Σ_{k≤a} κ_i·gp + a·g(S(a))·TAIL_i` with `TAIL` = beta
  (`exi_xgtag`, lifted) or alpha (`exi_xgta`, linear). **Breaking:** the
  separate `_collapsed_exeqa` linear pricing engine is deleted;
  `Portfolio.price` reads rows of the unified frame for both methods.
  Linear **totals** are unchanged (≤ 2e-12 vs the a56 capture) but linear
  **per-line** values move: the unified formula keeps the `X = a` state at
  its true `κ(a)` and splits only the strict tail by alpha (the old
  engine merged `X ≥ a` into the collapsed atom), and per-line capital now
  uses the same layer-ROE construction as lifted (the old separate
  `rcoc` engine differed structurally). Lifted surfaces reproduce a56 to
  1e-14 (exact books) / 1e-11 (64k-row FFT books, fp order-of-ops drift);
  locked by `tests/data/numerics3_precapture.json` +
  `tests/test_numerics3_distortion.py`; the corpus baselines were
  recaptured.
- **Breaking: `T.*`/`M.*` columns removed** (`T.L/T.P/T.M/T.Q/T.LR/...`,
  `M.L/M.P/M.M/M.Q/...`) along with the `tm_renamer` property. Pricing
  readers use explicit `L = exa`, `P = exag`, `M = P − L`; per-line
  capital `Q_i(a)` is computed **on demand** by the layer-ROE
  construction (line layer margin ÷ total layer ROE, integrated; the
  layer margin is the exact first difference of `exag_i − exa_i`) inside
  `pricing_at` / `pentagon_at` / `price`, with `Σ_i Q_i(a) = a −
  exag_total(a)` reconciled. Zero-total-margin layers contribute zero
  capital (under the identity distortion per-line `Q` is 0 — there is no
  margin to allocate); fully-loss-funded layers (`gS = 1`) use the
  L'Hôpital ROE limit.
- **Breaking: mass-on-unbounded guard moved into the builder** (G6).
  `apply_distortion` / `Aggregate.apply_distortion` refuse a mass
  distortion (e.g. `ccoc`) on an unbounded support for the lifted frame —
  previously only `price(allocation='lifted')` refused, so `exag_total`
  could still build an unstable frame. `allocation='linear'` remains
  available (the collapsed default atom is bounded by construction; the
  unstable beta columns are blanked). `analyze_distortions` skips such
  members of a sweep with a `UserWarning` instead of failing the exhibit.
- **Breaking: `efficient` removed entirely** from `apply_distortion` /
  `price`; one frame shape. The diagnostic layer curves moved to the new
  explicit `Portfolio.allocation_diagnostics(distortion,
  surface='lifted'|'linear')` frame (`layer_loss/premium/margin/capital`,
  `cum_margin/cum_capital`, `layer_roe_total`, plus kappa/alpha/beta and
  `F/gF/S/gS/gp_total`); `pedagogy.plot_twelve` consumes it (the
  efficient→full cache-pop hack is gone).
- **Breaking: `apply_distortion` cache key widened** from the distortion
  name to `(name, view, role-flag, S_calculation, allocation)`, so
  bid/ask, loss/payoff, forwards/backwards and linear/lifted frames
  coexist (previously a second call with different options returned the
  stale first frame). `augmented_dfs` is keyed accordingly.
- **Aggregate surface aligned.** `Aggregate.apply_distortion(dist, *,
  view, S_calculation, allow_deficit)` writes `gS`, `gp_total` and the
  exact `exag = ρ_g(X ∧ a)`; the six surfaces (`Distortion.price` dx/ds,
  Portfolio `exag_total` / `price` both methods, Aggregate `exag` /
  `price`) agree on the total premium to machine precision.
- **Signed (P&L) books price.** The numerics-2 `NotImplementedError` is
  gone: total distorted columns (`gS/gp_total/exag_total`) are exact on
  signed windows and the signed total prices via `dot(κ, gp)`
  (additive across units); the equal-priority per-line distorted columns
  are NaN (not a recovery share on a signed grid). Homogeneous payoff
  books price through the dual automatically; mixed books still raise at
  construction (hygiene-4).
- `AllocationBounds` owns its bounded-total collapse directly (built from
  the `exi_xgta_*` columns); reproduces its baseline unchanged.
- `Pentagon.from_row` reads `L/P` and derives `M` (per-line `Q` needs the
  layer integral — use `Portfolio.pentagon_at`).
- Docs: `5_x_distortions.rst` / `5_x_portfolio_calculations.rst` /
  `2_x_10mins.rst` updated to the exact-discrete formulation (doc build
  pending, run manually).

## 1.0.0a56

### Numerics-2 — objective spine (shifted-support kappa + direct sums)

Second plan of the numerics program
(`dev/done/plan-numerics-2-objective.md`). `Portfolio.add_exa` and the
Aggregate objective columns rewritten on the exact-discrete, origin-carrying
footing; signed (P&L) books now get the objective allocation columns. Step-0
audit with measured verdicts in `dev/done/audit-numerics-2-findings.md`.

- **Shifted-support kappa.** `exeqa_{line}` is computed from each unit's
  **native** pmf: the first-moment density `x·p_i(x)` is built from true
  physical values and scattered into the physical-zero FFT buffer
  (first moments, unlike probabilities, cannot be recovered from a rolled
  vector), then `ift(ft_xp_i · ft_not_i) / p_total`, relabelled onto the
  output window by the same roll as the combine. Exact on negative and
  nonzero origins (brute-force-convolution tested). Per-unit FT state is
  transient within `update` (D6) — only scalars and native pmfs persist.
- **`ft_nots` single owner.** Per-line "not-line" FT products live in one
  helper: spectral division when the line's spectrum has no exact zero bins
  (measured per-bin well-conditioned even on underflowed spectra),
  prefix/suffix partial products otherwise — `O(m·M)`, replacing the legacy
  `O(m²·M)` rebuild.
- **Direct sums carrying the origin.** `exa_total/lev_total` =
  `Σ_{x≤a} x·p + a·S(a)`; `exlea/exgta/exi_xlea/exi_xgta/exa_{line}` from
  forward/reverse direct sums of `kappa·p_total`; Aggregate `lev/exa/exlea/
  exgta` likewise. `cumsum(S)·bs` and the `loss_max` / `mult ∈ {1,10,100}`
  blanking heuristic are gone; ratio denominators carry explicit
  `F/S ≤ validation-noise` guards (NaN where the conditioning event is
  unresolvable; previously unguarded division could emit `-inf`).
- **Stand-alone unit quantities from native pmfs.** `lev_{line}` is the
  exact capped native sum `Σ_{x≤a} x·p_i + a·(1−F_i(a))` and `e_{line}` the
  native mean — valid whether or not the unit window overlaps the total
  window.
- **Signed (P&L) books**: `update(add_exa=True)` now computes the objective
  columns (the warn + F/S-only fallback is removed). The equal-priority
  share `kappa/x` is not a recovery share on a signed grid (steering 6), so
  `exi_x*_{line}` and `exa_{line}` are NaN there; `apply_distortion` /
  pricing on signed books raises `NotImplementedError` until numerics-3.
- **Breaking: `p_{unit}` columns removed from `Portfolio.density_df`**
  (both combine paths). Unit pmfs live on the Aggregates — read them via
  `unit_density` / `unit_density_df` / `aligned_unit_density_df`
  (numerics-1). The sampling/switcheroo cluster (`sample`,
  `add_exa_sample`, `swap_density_df`, `make_awkward`) is mechanically
  re-sourced onto the accessors (redesign deferred to its own plan);
  `swap_density_df` still accepts a user `p_{line}` frame.
- **Breaking: EPD family removed** — `add_exa_details` (`epd_0_*`,
  `epd_1_*`, `e1xi_1gta_*`), and `Aggregate.density_df['epd']` (no
  consumers). Stand-alone EPD is the one-liner `(e − lev) / e`.
- **`add_exa` signature changed**: takes the per-unit native state
  (`{name: dict(xs, p, ft_p)}`) instead of pre-built `ft_nots`. The
  `Portfolio.ft` / `Portfolio.ift` padding-bound wrappers (only used by the
  old `add_exa`) are removed — use `aggregate.utilities.ft/ift` directly.
- Regression: key columns byte-stable (baseline harness; recaptured for the
  removed `p_{unit}` columns and the ≤2.8e-12 Aggregate `lev` drift);
  derived columns gated by new pre-change spot-checks
  (`tests/test_baseline_spotchecks.py`, measured drift ≤7.5e-12 on the
  `(e−cum)/S` cancellation, ≤2.5e-14 elsewhere). New invariant suite
  `tests/test_numerics2_objective.py` (Σκ(x)=x, Σ exa_i = exa_total,
  brute-force kappa incl. negative/positive origins, zero-spectrum
  prefix/suffix, native lev, signed exeqa vs Monte Carlo). Docs updated in
  lockstep (`5_x_portfolio_calculations.rst`, quantiles, student guide,
  10mins, samples); doc build pending.

## 1.0.0a55

### Numerics-1 — unit-density decoupling

First plan of the numerics program (`dev/done/plan-numerics-1-unit-density.md`;
target architecture in `dev/done/plan-numerics-0-meta.md`). Pure-additive accessors
plus migration of the display readers off the legacy
`Portfolio.density_df['p_{unit}']` columns. No compute change; no distortion
surface touched.

- **New accessors on `Portfolio`** sourcing unit pmfs from the owning
  `Aggregate` objects on their native grids:
  - `unit_density(unit, view='agg')` — one unit's pmf (`view='sev'` for the
    discretized severity), indexed by the unit's own loss grid.
  - `unit_density_df(view='agg')` — long form, `(unit, loss)` MultiIndex, with
    window-audit metadata (`bs`, `x_min`, `x_max`, represented `mass`).
  - `aligned_unit_density_df(grid='total'|'union'|'zero', *,
    allow_window_mismatch=False)` — the explicitly-named **display adapter**
    scattering unit pmfs onto a common grid (bucket-number alignment). On a
    legacy zero-origin book `grid='total'` reproduces the `p_{unit}` columns
    exactly; on a windowed book it warns that the view is clipped unless
    acknowledged. Raises if a unit was re-updated off the portfolio `bs`.
- **Display readers migrated** off `density_df['p_{unit}']`:
  `Portfolio.percentiles` (still deliberately interpolated), `_limits`, and
  `plot` (total now always plotted first — Book standard — previously only
  guaranteed for two-unit books); `pedagogy.ClassicalPremium.distribution`,
  `pedagogy.plot_bivariate`, and the density / bivariate / stand-alone-M
  panels of `pedagogy.plot_twelve` (allocation panels ride with numerics-3).
- **`p_{unit}` columns are now legacy.** The write remains (kappa in `add_exa`
  and the sampling/switcheroo cluster still read them) and is dropped in
  numerics-2 when kappa goes shifted-support. Do not write new readers.
- **Fix:** `ClassicalPremium.distribution` referenced the removed
  `Portfolio.audit_df`; empirical moments now computed directly from the pmf.
- Stale plan pointer in the signed-path `update` warning repointed to
  `dev/done/plan-numerics-2-objective.md`.
- Tests: `tests/test_unit_density.py` (native-grid accessors, disjoint-support
  signed book, exact legacy parity gate, windowed-clip warning, and
  stripped-frame proofs that the migrated readers no longer need `p_{unit}`).
  New DecL programs mirrored in `test_decl.agg` (section UD).

## 1.0.0a54

### Hygiene 4 — value_type, fixed-layout info strings, pnl prem/lr meta

One batch, four items (`dev/done/plan-hygiene-4.md`):

- **`Portfolio.value_type` derived from its units.** Read-only property: the
  unanimous `value_type` of the constituent aggregates. A mixed loss/payoff
  book is rejected at construction with a `ValueError` naming the offending
  units (no coherent sign convention); an empty portfolio defaults to loss.
- **Fixed-layout `info` strings** across `Aggregate` / `Portfolio` /
  `Distortion`. Every row is always present, in the same order, for every
  instance — no conditional rows; unavailable values render as `n/a`. All
  three classes share one label/value convention
  (`aggregate.constants.info_row`, 25-col label, no colon); `Distortion` was
  rewritten onto it (was indent+colon style). Row changes: Aggregate gains
  `value_type`-near-top, `x_min`/`x_max` (replacing the conditional
  `window`/`signed severity`/`severity window` block), always-present
  `premium`/`expected loss`/`loss ratio`/`P(loss)` (the `E[margin]` row is
  dropped — derivable), `bounded` and `id` footer rows; the `approximate`
  continuation line is dropped (detail stays in the note). Portfolio gains
  `value_type`, `x_min`/`x_max` (replacing `signed window`), premium rows;
  `tail`/`bounded` move to the footer; `hash` is relabelled `id`. Distortion
  drops `display name`/`strict-pricing`, renames `mu({0})`/`mu({1})` to
  `weights mean`/`weights max`, adds `kind name`/`shape`/`shape name`/
  `other params`/`area`. The full row catalogue and value enumerations are
  documented in `dev/info-strings.rst` (destined for docs; **docs pending
  rebuild**).
- **`stats_df` `('meta','prem')`/`('meta','lr')` backfilled for `pnl`.** The
  `pnl X prem - ...` form routes premium through `agg_premium`, which never
  reached the meta rows; they now backfill from it when the exposure clause
  supplied no premium. GROSS basis: under reinsurance `prem`/`lr` are the
  theoretical pre-reinsurance figures (`lr = gross el / prem`). The frozen
  numeric baseline is unaffected (no `pnl` programs in the corpus); no
  density / risk-measure numbers move.
- **`value_type` labels configurable.** New `[labels]` config section
  (`loss = "loss"`, `payoff = "payoff"`; env `AGGREGATE_VALUE_TYPE_LOSS` /
  `_PAYOFF`). Objects store the role as a private boolean `_is_loss_value`
  (loss is the anchor pole); the label is resolved at the display/parse
  boundaries only, so a relabel renames the printed word without moving any
  object's role, and future pricing code branches on the boolean, never the
  label text.

Breaking (display-level): code pinning the old `Distortion.info` format
(`Distortion: {name}`, colon rows), the Portfolio `hash` label, the Aggregate
`E[margin]` / `signed window` / `severity window` info rows, or the
conditional presence of `dsev_bucket` must be updated. Constructing a
`Portfolio` mixing loss and payoff units is now an error. The `Aggregate`
constructor and `value_type` setter now raise on an invalid `value_type`
(previously the constructor silently coerced to `'loss'`).

## 1.0.0a53

### DecL unparser + program formatter (`decl_writer`)

New `aggregate.decl_writer` module — the structural inverse of the parser
(`dev/done/plan-decl-unparser.md`). It renders a parsed spec back to canonical
DecL text instead of pretty-printing by regex, so a single function backs program
display, the `to_agg` exporter, and any future web `format` endpoint.

- **`spec_to_decl(spec, kind, name)`** — the unparser. Pure function from a raw
  transformer spec (`parsed.spec` / a knowledge entry's `pp.spec`) to canonical
  DecL. Built from clause renderers that mirror the transformer rules one-for-one
  (exposure, layers, severity incl. scale/reflect/`mean cv`/mixtures/splice/
  `dsev`/`xps`/`picks`, frequency incl. `mixed`/`zm`/`zt`, reinsurance, `pnl`,
  `port`, `multivariate`/`copula`/`netceded`, `approximate`, distortions, note/
  hints trailer).
- **`format_program(spec_or_text, *, fmt='text'|'html'|'ansi'|'latex')`** — the
  public entry. Accepts a spec, a `(kind, name, spec)` tuple, or a program string
  (which it parses first). Pure: returns a `str`, never prints. Colorization
  reuses the existing `decl_pygments.AggLexer` (no second keyword list).
- **Contract:** idempotence one step removed — `f(f(f(x))) == f(x)` with
  `f = spec_to_decl`. The whole reference corpus (`test_suite.agg` +
  `test_suite2.agg` + `test_decl.agg`) round-trips, verified by
  `tests/test_decl_unparser.py` with a numpy/inf/object-aware spec comparator.

**Breaking.** `utilities.decl_pprint` is **removed** (along with its
`pygments`/IPython plumbing in `utilities`). `Aggregate.pprogram` /
`pprogram_html` and `Portfolio.pprogram` / `pprogram_html` now render the
**canonical** form via `format_program(self.program)` (was the verbatim text with
notes stripped); `self.program` still holds the raw input. `Underwriter.to_agg`
now emits `spec_to_decl(spec)` per entry, so exported `.agg` files are canonical
(it falls back to the stored program only for `minimum`/`mixture` combinator
distortions, whose child references cannot round-trip). Docs that imported
`decl_pprint` now use `format_program`.

Also: fixed two `test_decl.agg` notes that carried `{...}` braces inside
`note{...}` (the `NOTE` terminal cannot represent `}`); they never parsed
standalone (`test_decl.agg` is a reference corpus, not runtime-loaded).

## 1.0.0a52

### Hygiene-3 batch (grammar + robustness nits)

Three small, independent nits, one version bump (`dev/done/plan-hygiene-3.md`).
No movement of the numeric baseline.

- **Underscore digit separators in DecL numbers.** The `NUMBER` terminal now
  accepts Python-style `_` group separators — `agg BIG 10_000_000 claims …` —
  with leading / trailing / doubled underscores (`_1`, `1_`, `1__0`) rejected by
  the lexer, exactly as Python's `float()` / `int()` behave. Grammar-only: the
  transformer already coerces via `float`, which strips the underscores.
- **`of` as a share synonym in reinsurance.** A reinsurance clause now accepts
  `of` alongside `so` / `po`, so `occurrence net of 90% of 6000 xs 4000` reads
  naturally. `of` is treated as *share of* (`so`): a literal percentage is the
  share directly, a bare amount is `amount / limit`. No new terminal (the `OF`
  token already existed); a single new `reins_clause` alternative.
- **Fixed an array-ambiguous truth test.** `Aggregate._sev_label` used
  `if not self.sevs:`, which raised *"truth value of an array … is ambiguous"*
  for a multi-component (weighted) severity, where `self.sevs` is an ndarray.
  Replaced with the explicit `self.sevs is None or len(self.sevs) == 0` idiom; a
  sweep of `distributions` / `portfolio` / `spectral` found no other array-valued
  truthiness tests.

Deferred: the "every public DataFrame member present (`None`) before compute"
item was moved to `dev/TODO.md` Track H (**H9**) — review found it largely
already-satisfied or aimed at members that don't exist; the genuine narrow
version needs separate scoping.

## 1.0.0a51

### Non-zero aggregate output window for high-mean / thin-tail aggregates (Plan B)

A concentrated aggregate — one whose coefficient of variation is small enough
(`agg_cv < 1/z`, `z = norm.isf(1e-WINDOW_NINES) ≈ 7`) that its whole probability
mass sits a long way above 0 — is now computed on a **two-sided output window**
far from 0 instead of the wasteful `[0, x_max]` grid. The headline case

```
agg Window 10000000 claims dsev [1 2] poisson
```

(mean 15,000,000, sd ≈ 5,000) used to build at `bs ≈ several hundred`, spending
almost all of its resolution on the empty `[0, 14.97M]`; it now resolves at
**`bs = 1`** on the window `[≈14.96M, ≈15.04M]`, with matching moments and mass.

**How it works.** This reuses the existing benign-FFT-wrap machinery built for
the negative-x / `pnl` work: the severity is laid into the period-`M·bs` FFT
buffer and the finished aggregate is relabelled onto the window by a single
modular `np.roll` of `round(x_min/bs)`. Relabelling a finished, exact array
carries no `N·s` shift term, so it is correct for random as well as fixed
frequency; the large roll and any period straddle are handled automatically by
`np.roll`'s modular semantics. The compute path was already in place — the new
work is purely the **sizing** decision in `Aggregate._bs_window`.

- **New `windowed` sizing method** (a row in the inspectable `_bs_window_df`).
  It sizes `bs` from the realised band *width* (`estimate_agg_window`), not from
  `x_max`, and is selected only when **strictly finer** than the 0-based pick —
  a self-tuning, self-limiting rule. It can only be finer when the band clears 0
  (`agg_cv < 1/z`), so **ordinary aggregates are byte-for-byte unchanged** (their
  window would include 0; the candidate never qualifies).
- **Integer lattice preserved.** For a discrete `dsev` the windowed method keeps
  the exact lattice `bs` and grows `log2` a small, bounded amount past the cap
  (`WINDOW_LOG2_GROWTH = 4`) rather than coarsening below the lattice and
  mis-placing the atoms — so the headline case lands at `bs = 1`, `log2 = 17`.
- **Severity-fit guard.** Windowing relabels only the aggregate; the severity is
  still discretised on `[0, N·bs]`. A single occurrence must fit that extent, so
  a `fixed`-1 / `approximate` object (whose one severity already sits at the
  aggregate mean) is **not** windowed — it falls back quietly to the 0-based
  grid. The `windowed` row is still recorded (marked `applies=False`) for
  inspection.
- **Occurrence reinsurance suppresses windowing.** The occ-reins severity
  rebucketing and `reins_density_df` carry the severity on the *output* grid
  (`xs == xs_sev`), which a non-zero window origin would break. A book with occ
  reins keeps the 0-based grid; **aggregate** reinsurance is unaffected
  (it operates on the aggregate, on the windowed `xs`/`x_min`).

**Behavioural note (intended).** `q` / `F` / quantiles / plots of a windowed
aggregate are defined on the window `[x_min, x_max]`, not from 0. The window
covers the probability mass to `1 − 1e-WINDOW_NINES` per edge; what is given up
is the `[0, x_min)` *axis* region (sub-tolerance far tail, low-attachment layer
losses, the from-0 severity overlay). Pass **`x_min=0` to `update`** to force the
legacy 0-based grid back. The footprint is broader than the single headline
example: **any** non-reinsured aggregate concentrated enough that its mass clears
0 (e.g. large claim counts) now resolves on a finer, non-zero-origin window.

## 1.0.0a50

### Hygiene: consistent `info`, self-describing `approximate` note, window-aware plots

Three small display/consistency fixes (no numeric baseline moves):

- **`Aggregate.info` always emits the `approximate` line.** It is now a
  permanent header line positioned *after* severity (`freq → sev → approximate`)
  and shows `exact` for an ordinary aggregate, instead of appearing only when a
  fit was active and *before* severity. The scaffold no longer reorders or drops
  the line by state. (`Portfolio.info` and `Distortion.info` audited — their
  conditionals are all feature/content driven, left as-is.)

- **The `approximate` note records the original program and fitted params.**
  Previously the note was assembled inside `Aggregate.__init__` **before**
  `self.program` was set by the build path, so it could only ever carry the fit
  moments — never *what* was approximated. The note is now kept as the user's
  pure note at construction; the fit is captured in a structured
  `self._approx_fit`, and a new `_approx_description()` renders
  `"<program>  approximated by <kind>: <family>(params), m=.. cv=.. skew=.."`
  lazily — shown indented under the `info` `approximate` line and folded into the
  note once `program` is available. Round-tripping rides on `program` (re-parsed
  on load), not the note, so the note never compounds across re-exports.

- **`Aggregate`/`Portfolio` plots are window-aware.** The linear x-limits
  (`_limits(stat='range')` and the discrete-plot left edge) are keyed on the grid
  origin: an ordinary 0-based aggregate is unchanged and a signed P&L window
  keeps its two-sided range, but a **thin-tailed output window starting above 0**
  now anchors the left edge at the realised support minimum instead of forcing 0
  (no empty `[0, x_min]` band). Existing ordinary/signed plots are unchanged; the
  new branch prepares the axes for non-zero-origin output windows. The
  severity-overlay-vs-window question is deferred (see `dev/TODO.md`).

## 1.0.0a49

### Fixed: Portfolio combine grid — `best_window` replaces the RMS combine

The portfolio auto-sizer combined its units' per-unit window choices into one
shared `(bs, log2)` grid by **root-sum-square** (`Portfolio.best_bucket`). That
scaled the wrong way — *k* identical units gave `round_bucket(b·√k)`, so
**adding units coarsened the grid** — and it ignored the integer lattice
entirely, so an all-integer discrete book got a fine continuous `bs` (e.g.
`1/4096`) spread over the full `log2=16` cap.

New `Portfolio.best_window(log2, bs_in, bucket_sizing_p)` implements the correct
**resolution + span** rule, the *max* of two independent constraints:

```
bs = round_bucket(max(min_k bs_k, W_tot / N))
```

- **resolution** `min_k bs_k` — the finest bucket any unit needs (from each
  unit's own `_bs_window`), captured in a phase-1 analytic pre-pass;
- **span** `W_tot / N` — the no-wrap floor, `W_tot = Σ_k W_k` over the
  *selected-method* support-window widths (not the padded `used`-row extent).

For a **non-signed** book the grid keeps origin 0 and `log2` is now **shrunk**
to just hold the summed support — a tiny discrete port no longer inflates to the
`log2` cap. The **signed** (P&L) path keeps its analytic origin estimate and the
`log2` cap (tight signed `log2`/origin is deferred to the output-window work).
`best_bucket` is **retained but deprecated** (`DELETE BEFORE BETA`) as a
side-by-side comparison aid; it is no longer on the live path.

**Effect on the knowledge base (9 of 146 objects move; no single aggregate
changes — the bug was purely the combine):**

- *Wins.* Discrete books size correctly: the Bodoff portfolios and
  `PIR.1.Discrete` move from an absurd fine `bs` (≈0.03–0.0001 over 65536
  buckets) to the lattice-correct `bs=1` with a shrunk `log2`.
- *Neutral.* The continuous CNC ports re-resolve to each unit's natural `bs`
  (e.g. 0.03125 → 0.125) with mean fidelity unchanged.
- *Known limitation (surfaced, not introduced).* The two heavy-tailed HuSCS
  catastrophe ports coarsen further (`bs` 20000 → 300000). This is **not** a
  combine defect: the `Hu` unit's *own standalone* `_bs_window` already sizes at
  `bs=300000`, because its `exp()·lognorm` cat severity has a `1−1e-12` window
  ~1.9e10 wide. The span faithfully refuses to wrap that mass (the old `bs=20000`
  silently truncated it). A combined-moment span gives the identical bucket, so
  there is no combine-level fix — the fix belongs to the per-unit window
  *coverage* policy for heavy tails, which is out of scope here (see
  `dev/TODO.md`). These cat books need an explicit `bs` for production use today
  regardless.

Regression tests in `tests/test_bucket_sizing.py`; the bucket-sizing decisions
for the whole knowledge base are snapshotted to `tests/data/bucket_baseline_*.csv`
(a review reference, not a hard gate) via `scripts/bucket_baseline.py`.

## 1.0.0a48

### Fixed: spliced unbounded severities crashed window sizing

Building an aggregate whose severity splices an **unbounded** base family (e.g.
`sev lognorm 40 cv .65 splice [1 100]`) crashed in grid sizing with
`ValueError: Inadmissible value passed to round_bucket, inf`.

`splice [lb ub]` records the cap in `sev_lb`/`sev_ub` and conditions
`fz.cdf/sf/isf/ppf/pdf`, so the severity correctly reports `bounded == True` — but
`fz.support()` still returned the underlying family's `(0, inf)`. The bounded-window
sizer (`_bounded_severity_window`), invoked precisely *because* the severity is
bounded, then read an infinite upper edge. `_apply_lb_ub` now also patches
`fz.support()` to the honest `[sev_lb, sev_ub]` (mirroring the reflect-shift support
patch in `_apply_reflect`). The window of existing uniform-splice aggregates
tightens slightly to the true support — a strict accuracy gain (no grid wasted over
zero-mass regions). Only the splice path is affected; unspliced severities short-
circuit before the patch.

## 1.0.0a47

### Added: `approximate` DecL keyword — method-of-moments aggregates

A new design-time directive replaces the freq × sev FFT convolution with a single
continuous severity fitted to the aggregate's first three moments — the fast
shortcut for very-high-frequency books where the exact convolution is overkill.

```
agg Big 1e6 claims sev lognorm 100 cv 2 poisson approximate sgamma
```

- `approximate exact | sgamma | slognorm` (`exact` is the inert default; omitting
  the clause is the same). `sgamma`/`slognorm` fit a **shifted gamma / shifted
  lognormal**; a **normal** is used as the symmetric (skew ≈ 0) limit, and a
  **reflected** fit handles genuinely left-skewed aggregates (verified to match
  the exact aggregate's mean/CV/skew to ~7 significant figures across all three
  skew regimes).
- **No special compute path.** The substitution happens in `Aggregate.__init__`:
  the object is rewritten as an ordinary fixed-1-claim aggregate of the fitted
  severity, so `density_df`, validation, the `pnl` affine, and the `Portfolio`
  combine all work with zero special-casing. The original program round-trips
  (it is preserved on `self.program`); the fit is summarised in `note` and shown
  in `info`.
- **Incompatible with occurrence reinsurance** (which acts pre-convolution, so the
  method-of-moments fit has nothing to bite on) — rejected with a clear error at
  parse time and in the constructor. **Aggregate reinsurance rides along**
  unchanged. Works on `pnl` too: the loss part is fitted and the premium affine
  rides along.
- Available on the `agg … claims …`, `agg … dfreq …`, and both `pnl` forms.

No core/`freeze_knowledge` impact — no existing program uses `approximate`, so all
146 knowledge-base objects are unchanged to 1e-12. Grammar changed →
`ref_include.rst` regenerated; **doc rebuild pending**.

## 1.0.0a46

### Added: exponential-tilting pedagogy (Grübel–Hermesmeier illustration)

Exponential tilting for FFT aliasing control returns as a **pedagogy helper**, not
a core feature. The production convolution stays tilt-free — padding remains the
operational aliasing control (the `tilt`/`tilt_vector` arguments dropped from the
core `ft`/`ift`/`update_work` in the 1.0 refactor are **not** restored).

- New `aggregate.pedagogy.tilted_aggregate_density(agg, *, log2, bs, padding=0,
  tilt=None, normalize=False)` runs a single tilted convolution
  (`z·e^{-θk} → rfft → freq_pgf → irfft → ·e^{+θk}`) entirely locally. With
  `tilt=None` it reproduces the ordinary untilted convolution byte-for-byte.
- New `tilt_vector(theta, n)` helper and `gh_tilting_exhibit(...)`, which
  assembles the full Grübel–Hermesmeier (1999) Poisson/Levy comparison table
  (accurate + closed-form exact + tilt sweep) in one call.
- Rewired `docs/2_user_guides/problems/010_gh_example.rst` to the new helpers
  (the old `tilt_vector=` kwarg no longer existed). **Doc rebuild pending.**

No core/`freeze_knowledge` impact — the production path is untouched (all 146
knowledge-base objects unchanged).

## 1.0.0a45

### Fixed: `pnl` with a signed loss severity (`dsev` negative atom / `ssev`)

A `pnl` (premium-minus-loss) aggregate whose **loss severity is itself signed**
— a `dsev` with a negative atom, or an `ssev` — now convolves the loss on its
genuine signed grid before the affine relabel onto the P&L window. Previously the
affine path hard-coded a **0-based** loss grid, so the loss's negative atoms
wrapped to the top of the FFT buffer: roughly half the mass was silently dropped
and the empirical moments read ±2¹⁵ grid-index garbage (e.g.
`pnl GP 5 premium - dfreq[3] dsev[-1 1]` reported `Est EX = -32763.75` instead of
`5`). It now yields the correct `P&L ∈ {2,4,6,8}` with mass 1, mean 5, sd √3.
This is a **bug fix, not a breaking change** — every ordinary (non-negative-loss)
`pnl` is byte-for-byte unchanged (all 146 frozen knowledge-base objects match to
1e-12).

Implementation: `_bs_window` already sized a correct signed loss window; it now
hands that loss origin back to `update` for the affine case (0 for an ordinary
pnl, preserving the legacy grid), and `update` builds the loss grid uniformly
from it. `_apply_agg_affine`, already origin-agnostic, relabels the signed loss
onto the tight P&L window. It also now warns (`DefectiveDistributionWarning`) when
the reverse-and-roll drops more than dust off the P&L window — surfacing the
genuinely-unrepresentable far-tail case that was previously silent.

## 1.0.0a44

### Hygiene: module organization & dependencies

- **Relocated `make_ceder_netter`** and its layer-order validator
  `_validate_reins_layers` from `utilities` to `distributions`, their only
  consumer (the occurrence/aggregate reinsurance application). Hard move, no
  deprecation alias. Direct importers should use
  `from aggregate.distributions import make_ceder_netter, _validate_reins_layers`.
- **Dropped unused runtime dependencies** `cycler`, `psutil`, `ipykernel`, and
  `jinja2` — none were imported anywhere in `src/aggregate` (`cycler` is still
  provided transitively by matplotlib; `bounds` uses stdlib `itertools.cycle`).
- **Deferred `IPython` to lazy imports** inside the two display helpers that use
  it (`decl_pprint`, `agg_help`), removing it from the top of `utilities`. Since
  `utilities` sits on the `import aggregate` path, this cuts roughly a second off
  cold import time; `IPython` is pulled in only when a display helper is actually
  called. (`Pygments` is left eager — it is ~0 ms to import and is loaded anyway
  by the `decl_pygments` lexer.)
- Confirmed **no var/tvar duplication**: `utilities.make_var_tvar` is the single
  implementation; the per-instance `Aggregate`/`Portfolio._make_var_tvar` are
  thin wrappers.

## 1.0.0a43

### Fixed: `Portfolio.describe` spread column for signed portfolios

`Portfolio.describe` reports the spread as **CV** normally and **SD** when the
portfolio is signed (any unit is a P&L / negative-support `ssev`/`dsev` unit),
mirroring `Aggregate.describe`. The choice is now made once, portfolio-wide:
CV and SD cannot be mixed in one frame, so if *any* unit is signed the whole
table — every unit block and the total — uses SD, with unsigned units forced
via the new `Aggregate._describe(force_sd=...)`. The total SD is read robustly
from the second moment (`sqrt(ex2 - mean²)`), never `mean × cv`. Previously a
mixed signed/unsigned portfolio produced a ragged frame (some unit blocks in CV,
others in SD) and the mean-zero total showed a blown-up `Est CV`. All-unsigned
portfolios are unaffected (byte-for-byte identical output).

### Added: `price_pentagon` on `Aggregate` and `Portfolio`

Complete the eight-stat pricing octet (`L, M, P, Q, a, LR, PQ, ROE`) from a
capital level plus one target — no distortion involved, pure accounting
completion against expected loss at that level:

```python
port.price_pentagon(p=0.99, ROE=0.10)   # VaR capital + cost of capital
agg.price_pentagon(a=250, LR=0.70)      # asset level + loss ratio
```

- Fix the capital level with exactly one of `p` (VaR probability) or `a` (asset
  level, snapped to the grid).
- Supply exactly one pricing target: premium `P`, cost of capital `ROE`
  (a.k.a. CoC), loss ratio `LR` — also `M`, `Q`, `PQ`. The target keywords match
  the canonical stat names (`PENTAGON_STATS`). Clear `ValueError` if not exactly
  one capital input and one target.
- Returns the canonical one-row `'total'` pentagon DataFrame
  (`PENTAGON_STATS` columns), matching the rest of the pricing family.

Thin wrapper over the existing `Pentagon` machinery — no new pricing math.
`Pentagon.solve_obj` now accepts `a=` as well as `p=` (and `p` becomes
keyword-only; it has no other callers). `Portfolio.price_ccoc` is now a thin
alias for `price_pentagon(p=p, ROE=ccoc)` (the cost-of-capital special case);
its output is unchanged.

## 1.0.0a42

### Consolidated fuzz removal into one vectorized utility

The scattered FFT round-off de-fuzz idioms are unified on a single helper,
`utilities.remove_fuzz(data, eps=None)` — accepts an ndarray or a DataFrame,
two-sided (`|x| < eps → 0`, large negatives preserved, so it is correct on
signed/P&L densities), defaulting to machine epsilon.

- Replaces the per-cell `DataFrame.map(lambda x: 0 if abs(x) < eps else x)` in
  `Portfolio.remove_fuzz` and `Aggregate.density_df` (the former now writes the
  float columns back in place; the latter reassigns) — faster, vectorized.
- Replaces four duplicated `np.where(np.abs(x) < eps, 0.0, x)` array copies
  (two in `Portfolio`, two in `Aggregate`) that fed `xsden_to_mwrangler`.
- `ft.recentering_convolution` keeps its looser `2*eps` tolerance via the
  explicit `eps=` argument.

**Numerically inert**: a freeze/check over all 146 test-suite objects matched
within `atol=1e-12`. The one intended change is the moment-fit (MMSE) path,
whose threshold tightens from a stray `1e-16` to machine `eps` (~2.22e-16); it
is not part of the frozen `describe`/`density_df` surface.

**Deliberate carve-outs** (not routed through the utility): the one-sided
`Frequency.pmf` clip (a frequency pmf has no legitimate negatives) and the
plot-cosmetic `1e-15` clip in the reinsurance occurrence plot (looser threshold
plus a `0 → nan` step). Both now carry a comment marking them as such.

## 1.0.0a41

### Renamed: public `reinsurance_*` methods → `reins_*`

The three spelled-out `Aggregate` reinsurance methods now use the `reins_`
abbreviation, matching the rest of the surface (`reins_describe`,
`reins_density_df`, `reins_stats_df`, `reins_audit_df`, `reins_df`, the
`occ_reins`/`agg_reins` layer attributes, the `reins_bucket` config key) and the
house abbreviation style (`sev`, `occ`, `agg`, `freq`, `cv`, `bs`). `reins` is
now the canonical short form for "reinsurance" in identifiers.

**Breaking, no alias** (alpha):

| Old | New |
|---|---|
| `Aggregate.reinsurance_kinds()` | `Aggregate.reins_kinds()` |
| `Aggregate.reinsurance_description()` | `Aggregate.reins_description()` |
| `Aggregate.reinsurance_occ_plot()` | `Aggregate.reins_occ_plot()` |

Pure surface rename — no behaviour, numbers, columns, grammar, spec keys, or
config/env keys change. Docs reference the new names (reference page
auto-regenerates on the next Sphinx build).

## 1.0.0a40

### Fixed: SD/variance for zero-mean signed aggregates

`describe` reported `NaN` for the standard deviation of a mean-zero signed
(P&L) aggregate — e.g. `agg A2 dfreq [3] dsev [-1 1]`, where the severity SD is
1 and the aggregate SD is √3. The stored moments were always correct
(`stats_df` carries `ex2`); only the SD/variance *derivation* was wrong. It
reconstructed `SD = mean × CV`, and `CV = SD/mean` is `NaN` at mean 0, so
`SD = 0 × NaN = NaN`. The irony: `_describe_signed` exists precisely to dodge
unstable CV at mean ≈ 0, but the SD it printed was itself built from CV.

The fix derives variance **directly from the second moment** at all four
computation sites in `distributions.py` — `var = ex2 − mean²` (theoretical) or
`MomentWrangler.central[1]` (empirical FFT), with a `max(var, 0.0)` clamp for
fp dust before `sqrt`. For any positive-mean object `ex2 − mean² == (mean·cv)²`
to fp, so **all normal aggregates are bit-for-bit unchanged**; the behavioural
change is confined to mean ≈ 0 signed objects, where SD goes `NaN → correct`.
`MomentWrangler` (whose `NaN`-at-mean-0 CV is correct) is untouched.

## 1.0.0a39

### Ergonomic tweaks: keyword-only `Underwriter`, signed Lee plot, `density` accessor

Three small quality-of-life fixes, no behavioural change to the numerics:

- **`Underwriter` is now keyword-only** (`Underwriter(*, name=, databases=, …)`).
  This blocks the easy slip of `Underwriter('test_suite')`, which used to bind
  the first positional to `name` and silently *name* the underwriter after the
  database you meant to load. Use `Underwriter(databases='test_suite')`. The
  `repr` now also reports a **`requested`** line (the load *request*
  `self._request`) directly under `knowledge`, distinct from the resolved
  `databases` actually read. All existing call sites already used keywords, so
  blast radius is zero.
- **Signed (P&L) Lee plot fix.** `Aggregate.plot`'s discrete branch anchors a
  zero-mass row just left of the support; it set `loss=0` on that row, which made
  the quantile (Lee) panel draw a spurious vertical segment from `(F=0, loss=0)`
  down to the first point on signed support. The anchor's `loss` now equals its
  own index, so the Lee plot starts cleanly at the true minimum. `Portfolio.plot`
  was unaffected (its limits are already two-sided on signed support).
- **New `density` property** on `Aggregate` and `Portfolio`:
  `density_df.query('p_total > 0')` — the "live" support of the distribution
  with the grid's leading/trailing zero-probability buckets dropped. This is
  usually what you want to inspect. Plain property (recomputed per access) so it
  never goes stale against an `update`/resample.

## 1.0.0a38

### New: knowledge-freeze regression harness (`scripts/freeze_knowledge.py`)

A standalone, dependency-free tool to snapshot and verify knowledge-base
outputs across refactors. `freeze` builds every agg/port program with default
parameters and writes each object's `describe` and filtered `density_df`
(`p_*`/`exeqa_*` columns) to parquet, plus a `_manifest.json` recording the
exact program text, database list, and library version. `check` rebuilds from
the manifest and verifies the recomputed frames match the snapshot to a tight
absolute tolerance (default 1e-12). Importable `freeze_knowledge()` /
`check_knowledge()` functions for Jupyter; argparse CLI for the shell. Default
output goes to an OS-temp dir; pass `--root` for durable storage. Output parquet
is never committed. Additive tooling only — no library code changed.

## 1.0.0a37

### New: `PricingBounds` — cross-pricing ranges and the Gini lens

`bounds.py` gains **`PricingBounds`**: given that a reference risk `X` is
priced to P by *some* distortion, the range of the price of another risk `Y`
over the whole consistent family `G_P = {g : rho_g(X) = P}` (similar-risks
paper). A distortion prices any risk by `rho_mu(Z) = int TVaR_p(Z) mu(dp)`, so
the pricing constraint is one affine condition and the extreme measures are
biTVaRs; the price range of `Y` is the vertical slice at `T_X = P` through the
convex hull of the curve `(TVaR_p(X), TVaR_p(Y))`.

- **Unmatched p-grids resolved** by the *union of breakpoints*: within the
  intersection of an X-atom and a Y-atom both TVaRs are affine in `1/(1-p)`,
  so evaluating at every union breakpoint gives the exact piecewise-linear
  curve — no resampling.
- **Shared engine.** The hull/slice query core (`bounds`, `bitvars`,
  `distortion`, `p_star`, `plot`, `check`) was factored out of
  `AllocationBounds` into a `_HullEngine` base keyed purely on the vertex
  table; both classes are now thin front ends. `AllocationBounds` behaviour is
  unchanged (its test module passes verbatim).
- **TVaR-source adapters.** Each axis is a TVaR source — a discrete risk
  (`Aggregate`/`Portfolio`/pmf `Series`, with exact breakpoints and
  first-principles repricing) or a closed-form `(T, T_inv)` pair. The uniform
  reference `uniform_source()` (`TVaR_p = (1+p)/2`) is the **Gini lens**: as
  the X-source the constraint collapses to the mean-Kusuoka-level condition
  `E_mu[p] = 2P - 1` (`PricingBounds.mean_kusuoka_level`), reading the
  envelope gap of `Y`'s own TVaR curve. Sources are symmetric across both
  axes.
- Entry point: **`Portfolio.pricing_bounds(y_sources, *, a=0, p=0,
  s_floor=1e-14, n_grid=1024)`** — `y_sources` is one risk or a list/dict;
  `a`/`p` cap `min(X, a)` and `min(Y, a)`. Query with `bounds(P)`,
  `bitvars(P)`, `distortion(P, risk, bound)`, `check(P)` (independent
  survival-hinge repricing of each `Y`, plus `total` = price of `X` = P).
- **Conditioning.** Bounded (finite `a`) is exact and well-conditioned —
  `check` reprices to ~1e-12. Unbounded, the upper price bound of a
  heavy-tailed `Y` is genuinely tail-driven and the deep-tail FFT vertices
  (`1-p` below ~1e-7) are discretization noise; cap or raise `s_floor` for
  stable answers (documented on the class).
- The module-docstring comparison table now spans all three classes
  (`Bounds` / `AllocationBounds` / `PricingBounds`).

## 1.0.0a36

### New: `AllocationBounds` — natural-allocation pricing ranges (TODO N5)

`bounds.py` gains **`AllocationBounds`**: given that the portfolio total is
priced to P, the range of natural-allocation premiums to each unit over all
consistent distortions `G_P = {g : rho_g(X) = P}` (similar-risks paper). The
extreme allocations are attained at biTVaRs; each unit's range is the vertical
slice at `T = P` through the convex hull of the curve `(TVaR_p(X), a_i(p))`.
On the discrete FFT grid that curve is *exactly piecewise linear* with
vertices at CDF breakpoints (both coordinates are affine in `1/(1-p)` within
an atom), so the hulls — built from conditional tail expectations via reverse
cumsums — are exact, O(n) per unit, and **P-independent**: construct once,
slice for any premium.

- Entry point: **`Portfolio.allocation_bounds(*, a=0, p=0, units=None,
  s_floor=1e-14)`** returns the `AllocationBounds` object; query with
  `bounds(P)`, `bitvars(P)` (achieving `(p0, p1, w1)`), `p_star(P)` (exact
  TVaR inversion), `distortion(P, unit, bound)`, `check(P)`
  (first-principles repricing audit), `na_grid(p_grid)`, `plot(P=...)`.
- **Bounded totals** via `a=` (asset level, snapped to the grid) or `p=`
  (resolves `a = q(p)`): prices `X ∧ a` with the default states `X >= a`
  collapsed to one atom carrying the *linear* natural allocation
  `a·E[X_i/X | X >= a]` (lifted NA not offered); feasible premium range
  becomes `[E[X ∧ a], a]`. The delicate tail-collapse idiom was factored
  out of `price(allocation='linear')` into a single owner,
  **`Portfolio._collapsed_exeqa(a)`**, now shared by both callers —
  `price` outputs verified byte-identical before/after the refactor.
- The `bounds.py` module docstring now contrasts `Bounds` (IME 2022:
  distortion envelopes for the total, premium fixed at construction) with
  `AllocationBounds` (allocation ranges, premium at call time).
- Diagnostics: `additivity_error` surfaces the `exeqa` noise floor inherited
  from `density_df`; `s_floor` truncates noise-dominated tail vertices.

**Removed:** the long-broken `Portfolio.pricing_bounds`
(`NotImplementedError` since 1.0.0a11) and its `PricingBoundsResult`
dataclass.

## 1.0.0a35

### A bare `Underwriter()` loads no databases by default

**Behavior change.** `Underwriter(databases=...)` now defaults to `None` —
a bare `Underwriter()` starts **empty** (loads nothing) rather than pulling the
configured `build.databases`. This makes "give me an empty underwriter" the
trivial default and removes the surprise of an ad-hoc instance silently loading
the bundled `test_suite`.

- The module-level **`build` still loads the configured `build.databases`**
  (`test_suite` by default) — it now passes that request explicitly, so
  `config.toml`'s `[build].databases` continues to control what `build` knows.
  `discover()` and the built-in examples are unaffected.
- `databases` is therefore no longer config-driven at the *constructor* level
  (only `build` reads config); `log2` / `update` still default from config.
- Nicer help: the "use the configured default" sentinel now renders as
  `<config default>` in signatures (Jupyter `?`, `inspect.signature`) instead of
  `<object object at 0x…>`.

### Fixed: `to_agg` writes entries in dependency order

`to_agg` wrote entries sorted by `(kind, name)`, so severities (`'sev'`) landed
**after** the aggregates that reference them (`'agg' < 'sev'`). Because `.agg`
files load sequentially and named references (`sev.X` / `agg.X` / `dist.X`) must
resolve as each line is parsed, a saved file containing a named-reference would
fail to re-load — breaking the advertised round-trip. Entries are now written in
dependency order (severities and distortions, then aggregates, then portfolios),
so cross-referenced books round-trip. (Deeply chained *combo* distortions that
reference each other by name may still need a manual reorder.)

### `to_agg` gains a write `mode` (`x` / `w` / `a`)

`to_agg(..., mode=...)` mirrors Python's open modes so an existing file is no
longer silently clobbered:

- `'x'` (**new default**, safe) — create a new file, raising `FileExistsError`
  if it already exists.
- `'w'` — overwrite (logged).
- `'a'` — append the selection as a new block at the end, preceded by a dated
  `# added <timestamp>` comment; the block is written in dependency order. (On
  a missing file `'a'` behaves like `'w'`.)

## 1.0.0a34

### De-crufted calibration summary (`distortion_df` / `calibration_df`)

`Portfolio.calibrate_distortions` produced a wide `distortion_df` that mixed the
*per-distortion result* with the *calibration target* — the latter constant
across all five rows, with a part-vestigial `(a, LR, method)` MultiIndex left
over from a removed batch API. Split into two clean frames.

- **`distortion_df`** is now the per-distortion receipt only: index
  `distortion` (ordered categorical, canonical `ccoc, ph, wang, dual, tvar`),
  columns `param_name, param, error, gini_p, area`. `param_name` names the
  family's parameter (`r, a, lam, b, p`); `error` is the premium miss; `gini_p`
  is the comparable normalised shape `= 2∫g−1 = p_equiv`; `area = (gini_p+1)/2
  = ∫g`.
- **`calibration_df`** (new attribute) holds the shared target once: a one-row
  frame leading with the inputs `coc, p, F(a)`, then the canonical pentagon
  octet `L, M, P, Q, a, LR, PQ, ROE`. `ROE` equals the requested `coc` — a
  built-in self-check.
- **Breaking — `Distortion.standard_shape` renamed to `Distortion.gini_p`**
  (attribute, end to end). Verified `= 2∫g−1` for every calibrated family.
- Calibration is one-point (no batch mode); the index no longer implies a batch
  that does not exist. Docs (`2_x_10mins`, `5_x_distortions`) updated to read
  `P` from `calibration_df` and describe the new columns.

## 1.0.0a33

### `Portfolio.price_stand_alone` restored (modernised)

Reinstated `Portfolio.price_stand_alone(dist, p)` — used by the "10 minutes"
guide (its absence was breaking that build). It prices every unit on a
**stand-alone** basis (each backed by its own VaR(`p`) capital, the distortion
applied to its own loss distribution) and contrasts that with the diversified
whole.

- Built on the existing pricing primitives: each unit's column comes from
  `Aggregate.price` (a unit is just an `Aggregate`), the `total` column from
  `Portfolio.pricing_at` — so stand-alone and allocated pricing share one code
  path rather than re-deriving the integral by hand.
- Every row routes through the canonical pentagon
  (`complete_pentagon`), so the readout carries the full octet
  `L, M, P, Q, a, LR, PQ, ROE` in one consistent order. The `sum` row adds the
  amounts across units and re-derives the ratios.
- Canonical orientation, matching `pricing_at`: the eight stats are the
  columns, one row per entity (units, `total`, `sum`) under a `(method, unit)`
  MultiIndex. Transpose (`a.T`) for the traditional stat-down-the-side exhibit.
- Argument checking: `p` must be a probability in `(0, 1)`; `dist` must be a
  `Distortion` or the name of a calibrated one (clear `TypeError` / `ValueError`
  / `KeyError` otherwise). NumPy-style docstring added.

## 1.0.0a32

### Legible `Underwriter` database loading

The `Underwriter` knowledge-base loading surface — historically one overloaded
`databases` attribute plus the near-identical `read_database` / `read_databases`
methods, a fragile `len(knowledge)==0` lazy-load proxy, and a DataFrame-backed
store — was rebuilt around one explicit pipeline. The resolved knowledge is
unchanged (`build` loads the same `test_suite`); only the plumbing changed.

- **Dict-backed store with provenance.** The knowledge base is now a flat
  `{(kind, name): ParsedProgram}` dict; the `(kind, name)`-indexed DataFrame is
  built on demand for `knowledge` / `discover` (now with a `source` column).
  Each entry carries a `source` tag — the originating file `Path`, or
  `'session'` for an in-session `build(...)` — answering "where did this come
  from?".
- **One request, one resolver.** The constructor `databases=` argument is the
  *request* (stored privately); the public `databases` attribute now reports the
  resolved file `Path`s **actually loaded**. A single glob-aware resolver
  handles it: `'default'` / `'user'` / `'all'` are predefined globs, anything
  else is a path or glob resolved across cwd → `~/.aggregate` → bundled
  (literals first-match-wins; globs union across all three). `databases=['cat_*']`
  now works. The removed `'site'` token is no longer special-cased.
- **Honest lazy load + clear verbs.** An explicit `_loaded` flag replaces the
  entry-count proxy. New/renamed methods: `load(request=None)` (the one load
  verb; configured-once when `None`, additive otherwise), `reload()` (reset to
  as-created and re-read), `resolve_databases(request)` (preview without
  reading), `available_databases()` (discover `.agg` files on disk).
  **Breaking:** `read_database` / `read_databases` are renamed outright to
  `load` (no deprecated aliases) — `build` was effectively the only caller.
- **Consistent error policy.** A literal file named in an explicit `load(path)`
  that is missing raises `FileNotFoundError`; the configured request and any
  empty glob only warn.
- **Save / export.** New `to_agg(path, pattern='.*', kind='all',
  source='session')` writes selected entries' DecL back to a `.agg` file
  (round-trip re-loadable), to `~/.aggregate` unless the path is absolute. The
  default `uw.to_agg('mybook')` saves everything built this session — making the
  class docstring's "persist to and from `.agg` files" claim true.
- **One preprocessing owner.** `read_database`'s ad-hoc whitespace regexes are
  gone; `UnderwritingLexer.preprocess` now owns continuation/indent folding (a
  newline followed by any tab or spaces is a continuation), covering the tabbed
  and space-indented Portfolio layouts alike.

## 1.0.0a31

### One canonical pricing readout (the "pentagon")

Every pricing method emits the same eight accounting quantities — the amounts
`L` (loss), `M` (margin), `P` (premium `= L + M`), `Q` (capital),
`a` (assets `= P + Q`) and the ratios `LR = L/P`, `PQ = P/Q`,
`ROE = M/Q`. These used to be built independently by each method, disagreeing
on naming, order, completeness and dtype. They are now a single canonical
contract owned by `aggregate.pentagon`:

- **One name, one order.** `M/Q` is always `ROE` (with `CoC` documented as
  the synonym); the canonical order is `[L, M, P, Q, a, LR, PQ, ROE]`
  (`pentagon.PENTAGON_STATS` / `PENTAGON_DTYPE`). `Portfolio.pricing_at`,
  `price`, `price_ccoc`, `analyze_distortion`, `analyze_distortions` and
  `Aggregate.price` all route through one `complete_pentagon` helper, so the
  derivation `a = P + Q; LR = L/P; …` lives in exactly one place.
- **Consistent orientation.** Stats are always columns, one row per priced
  entity; any descriptor columns lead and the pentagon octet is the trailing
  eight (`df.iloc[:, -8:]`).
- **``analyze_distortion` audit fixed.** Its `audit_df` is now a one-row frame
  in that shape — `dname`/`dshape` lead, the full octet trails. *(Shape
  change: previously a column-oriented frame that omitted ``PQ` and mixed the
  metadata into the stat index.)* `price_ccoc` now emits `ROE` instead of
  `COC`.
- **Pentagon objects (additive).** New `Portfolio.pentagon_at(distortion, p|a, line)` returns a single-row `Pentagon` — an eight-vector with named
  attributes and provenance that completes any soluble partial input via
  `Pentagon.solve` (e.g. give it `P`, `L` and `a` or `Q`, get the
  rest). No existing method changed its return type.

## 1.0.0a30

### User-editable configuration file

The "secret bits" that used to be hard-coded literals — the default `log2`,
the default database, the reinsurance / discrete-severity bucketing scheme, the
bucket-sizing percentile, the output-window coverage, and the validation
tolerances — are now read from a single, optional, hand-editable **TOML** file
at `~/.aggregate/config.toml`. Values layer lowest-to-highest as
**built-in defaults → config file → ``AGGREGATE_*` environment variables →
explicit ``build(...)`` / ``Underwriter(...)` keyword arguments**, so a call
argument always wins and two fresh installs with no file behave identically.

New surface (all on the module-level `build` and any `Underwriter`):

- `build.write_default_config()` writes an annotated, **fully-commented**
  template to `~/.aggregate/config.toml` (inert until you uncomment a key);
- `build.show_settings()` prints every setting **and its source**
  (`default` / `config` / `env`);
- `build.reload_settings()` re-reads the file/environment after an edit and
  refreshes the module `build` in place;
- `aggregate.get_settings()` returns the resolved, immutable `Settings`
  snapshot (read once per session);
- `repr(build)` / `Underwriter` info gains a `config` line reporting the
  active file and how many settings are overridden;
- escape hatches: `AGGREGATE_CONFIG=/path` relocates the file,
  `AGGREGATE_CONFIG=none` ignores it (reproducible runs). Unknown keys,
  sections, and `AGGREGATE_*` variables warn loudly rather than silently
  no-op.

The tunable defaults and the path names now live in the new leaf module
`aggregate.config`; `aggregate.constants` is slimmed to the `Validation`
flag enum, `DefectiveDistributionWarning`, and the structural reinsurance
column labels.

**Breaking changes**

- **Minimum Python is now 3.11** (the config reader uses the standard-library
  `tomllib`; no new third-party dependency). The 3.10 classifier is dropped.
- **``recommend_p`` is renamed to ``bucket_sizing_p`** everywhere — the
  `build` / `build_many` / `update` keyword, the `hints{...}` key, and
  the underlying constant. There is **no alias**; update any call sites.
- The tunable names that used to live in `aggregate.constants` (e.g.
  `VALIDATION_EPS`, `VALIDATION_NOISE`, `RECOMMEND_P`,
  `REINS_BUCKET_DEFAULT`, `DSEV_BUCKET_DEFAULT`) moved to
  `aggregate.config` and are **not** re-exported; read them from
  `get_settings()` (e.g. `get_settings().validation.noise`).
- A bare `Underwriter()` now takes its `log2` / `databases` / `update`
  defaults from the configured `[build]` section (this unifies the old
  10-vs-16 `log2` split with the module `build`); pass `databases=None`
  to load nothing.

Scope: this is Phase 1 — the `[build]`, `[discretization]`,
`[validation].eps` / `.noise`, and `[multivariate].window_nines` settings.
Plot styling (`[plotting]` / `.mplstyle` override) and the numerics-pending
validation floors (`aliasing_ratio`, `exeqa_noise_floor`, `ft_noise_floor`)
land in a later phase.

## 1.0.0a29

### Tail-thickness classification for aggregates and portfolios

Every `Aggregate` and `Portfolio` now reports an ordered tail-thickness
class on a five-rung scale — `bounded` \< `super-exponential` \<
`exponential` \< `subexponential` \< `power-law` — plus a separate
log-concavity flag. The class is derived by deterministic family lookup keyed on
the frequency and (scipy) severity families, with the aggregate rung given by
the heavier of frequency and severity (`max`): for a subexponential-or-heavier
severity the single-big-jump principle makes the aggregate inherit the severity
class; for light severity the heavier decay rate wins. New surface:

- `Aggregate.tail_class` → a `(freq, sev, agg)` named triple of
  `TailClass` rungs; `Severity.tail_class` → the component rung;
- `Aggregate.tail_description` (three aligned lines) and
  `tail_explanation` (one sentence, with the power-law tail index `alpha`
  and infinite-variance / infinite-mean flags) — both also shown in `info`;
- `Portfolio.tail_class` / `tail_description` / `tail_explanation` report
  the **worst-of** unit aggregate (correct under independence) and name the
  driving unit(s).

`bounded` is now a **derived** view of the classifier (`bounded` iff the
aggregate tail class is `BOUNDED`) on `Aggregate`, `Severity`, and
`Portfolio`; the certify setter (`obj.bounded = True`) and the lifted
natural-allocation admissibility guard are unchanged. The bounded-support tables
moved to the new leaf module `aggregate.tail` (re-exported from
`distributions` for back-compat). `tail.py` is the single source of truth.

Scope: this is Phase 1 — exact, deterministic, spec-only (`bounded` resolves
before `update()`). Unrecognised or numeric-only families (histogram, meta,
spliced) classify as `undetermined`; the numeric density-tail estimator that
will fill those in (and set the aggregate's log-concavity) is deferred to a
later Phase 2.

## 1.0.0a28

### Mean-preserving (`linear`) bucketing for discrete severities

A new `dsev_bucket` setting controls how discrete-severity atoms (`dsev` /
`dhistogram` / `fixed`) are placed onto the model grid during
discretization, mirroring the existing reinsurance `reins_bucket`:

- `'linear'` (the **default**) splits each off-grid atom's mass across its two
  bracketing buckets, so the discretized first moment equals `Σ xₖ pₖ`
  exactly (mean-preserving);
- `'nearest'` snaps each atom to its closest bucket (the historical
  behaviour), biasing the discretized mean by up to `bs/2` per atom.

This matters when atoms are off-grid — e.g. a severity given as a sample of
empirical losses with a non-integer `bs`. The common integer-atom, `bs = 1`
case (a die, fixed losses) is on-grid, where the two schemes coincide, so it is
unchanged. Pass it as a `build` / `update` keyword:

    a = build('agg Off dfreq [1] dsev [0.3 1.7 2.4] [.5 .3 .2]', bs=0.5)
    # a.dsev_bucket == 'linear'; discretized mean == 0.3*.5 + 1.7*.3 + 2.4*.2

Scope: Phase 1 covers *unlayered* discrete severities (the empirical-sample use
case). A *layered* discrete severity discretizes via the cdf-difference and so
behaves as `'nearest'` regardless of the setting. `info` shows
`dsev_bucket` when a discrete component is present. The two discrete cases in
the baseline corpus (`Sym.Dice`, `Port.Bodoff`) shift at the floating-point
floor toward exact mass placement and were re-captured.

## 1.0.0a27

### Distortion DecL syntax is a flat number list; the parser stops knowing kinds

The DecL distortion form is now uniformly `distortion NAME kind n1 n2 ...` — a
flat list of the kind's parameters, no brackets:

    distortion D ph 0.9
    distortion D bitvar 0.9 0.99 0.5      # p0 p1 w1
    distortion D power 0.01 1.0 2         # x0 x1 alpha

Previously the parser carried a hand-maintained `_distortion_spec` table that
re-encoded every kind's parameter names (duplicating what the `Distortion`
subclasses already declare) and reached into `spectral` for domain facts. That
table is gone. Each subclass now declares its DecL parameter order in a
`decl_params` class attribute, and a single `Distortion.decl_spec` maps the
number list onto the kind's natural keyword arguments — one source of truth,
and adding a distortion kind no longer touches the parser.

- `ccoc` takes the return `r` (`distortion D ccoc 0.25`), not the discount.
- `wtdtvar` (parameter *vectors*) and the `minimum` / `mixture` combinators
  (which take distortion *references*) have no flat-number form and raise a clear
  error if written that way; construct them in Python or via the combinator
  syntax. The bracketed `kind shape [list]` form is removed.

## 1.0.0a26

### Honest discrete severity — exact moments, no more `rv_histogram` hack

A truly discrete severity (`dsev`, `fixed`) used to be represented by
*abusing* `scipy.stats.rv_histogram` — a continuous, piecewise-linear-CDF
object — forced to mimic a step function by pouring each atom's mass into a
tiny `2**-d`-wide sliver to its left (sized by a float-resolution helper,
`max_log2`). It worked, but it was a representation lie: it produced quantile
artifacts (`ppf(0.5) = 149.9999999992` instead of `150`) and — because the
sliver width *scales with atom magnitude* — it quietly degraded the **moments**
of large-valued discrete books.

`SeverityDHistogram` / `SeverityFixed` now back `self.fz` with a small,
honest `_DiscreteRV`: exact right-continuous step `cdf`/`sf`, `pdf = 0`,
and exact `ppf`/`isf`/`support` (no trailing-9s artifacts; `rvs` returns
exact atoms).

- **All discrete moments are now exact**, computed as finite sums over the
  atoms — unlimited, limited, *and* layered. Previously every discrete moment
  (even the unlimited mean of a fair die) was computed by numerical
  isf-integration and came back as `3.4999999995` rather than `3.5`; layered
  discrete moments integrated a step function by quadrature, which was both
  inexact and fragile. A discrete severity never routes its moments through the
  numerical path anymore.
- **Aggregate density is unchanged** — the FFT samples `cdf`/`sf` at
  half-bucket edges, which never coincide with an atom, so the discretised
  density is bit-for-bit identical to before. The improvement is confined to
  reported moments and quantiles, which become *more* correct (the baseline /
  golden regression snapshots were re-captured to record the exact values).
- `max_log2` is now unused (kept for one release; slated for removal).

See `dev/done/plan-discrete-severity-fz.md`.

## 1.0.0a25

### `note{}` is now pure text; build settings move to `hints{}`

`note{...}` used to do double duty: free-text annotation *and* a
`key=value;` side-channel for build settings (`log2`, `bs`, …). That
overloading meant any `=` in note prose — e.g. `note{... needs x_min<=-6}` —
was mis-read as a keyword argument and crashed the build. Notes are now **pure
annotation**; a dedicated `hints{...}` clause carries build settings.

- **Syntax.** `hints{key=value; key=value}`, e.g.
  `agg A 5 claims sev lognorm 10 cv 2 poisson hints{log2=18; bs=1/64}`.
  `note{}` and `hints{}` are both optional and order-free (at most one of
  each); `hints` is allowed everywhere `note` is (agg, sev, port).
- **Caller always wins.** Explicit `build(...)` keyword arguments override
  in-program `hints` uniformly — including `recommend_p` (fixing the old
  quirk where a note's `recommend_p` overrode the caller).
- **Forgiving.** Values are inferred generically (int / float / `a/b`
  fraction / `True`/`False` / str). Unknown keys warn and are dropped;
  a duplicate key warns and the last value wins; a malformed clause warns and is
  skipped — a bad hint never crashes the build.
- **Deprecation.** A `note{}` that still looks like it carries `key=value`
  settings emits a one-time warning (the note is treated as pure text).
- **Migration.** The bundled `test_suite.agg` / `test_decl.agg` corpora moved
  their settings-in-notes into `hints{}`; built grids are unchanged. See
  `dev/done/plan-note-parse.md`.

## 1.0.0a24

### The `multivariate` keyword — copula-coupled bivariate aggregates

A single event can drive two correlated perils — wind *and* flood, attritional
*and* large — and you want the **joint** law of the two aggregates, not just two
marginals. `multivariate` makes that a first-class object: two component
`agg` / `pnl` severity factories whose per-claim severities are coupled by a
**copula**, accumulated by a **shared** outer frequency through a 2D FFT. It
subsumes the `1.0.0a20` `occ_bivariate` backbone into a modelled,
DecL-declared facility. See `dev/done/plan-multivariate.md`.

- **Syntax.** :

      multivariate Cat 25 claims
          agg Wind  dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
          agg Flood dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
          copula gumbel 0.4          # Kendall tau = 0.4
          mixed gamma .2             # shared mixing -> common shock

  The shared count (`25 claims`) and trailing frequency own the event count;
  each component's `dfreq [0 1] [p0 p1]` is the per-event trigger probability,
  so its *aggregate* is the per-event severity `g_i = (1−p_i)δ₀ + p_i f_i`.
  The `copula` clause is **optional** — omitted (or `copula independent`)
  means the independence copula, where the only dependence is the shared count.

- **Copulas, à la ``Distortion`** (new `aggregate.copula.Copula`). A registry
  / factory hierarchy — `Copula('gumbel', 0.4)` dispatches on the name — with
  **normal** (Pearson ρ), **gumbel** (Kendall τ, upper tail), **clayton**
  (Kendall τ, lower tail), **fgm** (Spearman ρ_s), and **independent**. Each
  takes its *natural* dependence parameter and converts internally; `t` (the
  two-parameter kind) is deferred.

- **Discrete Sklar construction.** The joint per-claim severity is the copula
  rectangle mass `S[i,j] = C(G1[i],G2[j]) − C(G1[i−1],G2[j]) − C(G1[i],G2[j−1]) + C(G1[i−1],G2[j−1])` over the marginal CDF breakpoints; `S` has the
  component severities as exact marginals (atoms handled as jumps in `G`). The
  joint aggregate is `iFFT2(freq_pgf(N, FFT2(S)))` — the ordinary compound FFT
  with 1D transforms replaced by 2D, valid because `freq_pgf` is elementwise.
  Marginalising one axis reproduces that component's standalone aggregate.

- **``pnl` axes in v1.** A `pnl` component contributes its *loss* severity to
  the copula+FFT, then its premium becomes a **per-axis affine** (reflect +
  shift) applied to that tensor axis *after* the FFT — the 1D `_apply_agg_affine`
  relabel lifted to one axis. The affine commutes with marginalisation (marginal
  = the standalone `pnl`) and flips the loss-loss copula dependence to the
  correct profit-loss sign.

- **Reporting.** `MultivariateAggregate` exposes `marginals` / `moments`
  (`E[A0ⁱ A1ʲ]`) / `corr` and the properties `density_df` / `stats_df` /
  `describe` / `info`, a two-panel `plot` (joint per-claim **severity** on
  the left, joint **aggregate** on the right), and a `help` introspector. The
  realised output correlation is reported
  **alongside** the copula τ — compounding attenuates per-claim dependence, and a
  shared *mixing* frequency adds common-shock dependence on top (so even the
  independence copula gives a positive baseline correlation from the shared
  count).

- **Net/ceded as a first-class mode.** The joint per-occurrence (ceded, net)
  law of a *reinsured* aggregate is now a `MultivariateAggregate` in
  **``netceded` mode** — same 2D-FFT engine, a different (comonotone)
  per-claim severity builder. Two entry points: the DecL `netceded <agg with occurrence reinsurance>` statement, and `Aggregate.occ_bivariate()`, which
  now **returns** that object (so it gets the full `describe` / `stats_df` /
  `info` / `plot` / `help` surface, not a bare container). The axes are
  `Ceded` / `Net`; marginalising reproduces the univariate occurrence
  ceded / net aggregates, and `E[Ceded] + E[Net]` equals the gross mean.

- **Module move.** `BivariateDistribution` (the internal 2D density
  container) and the `size_axis` / `scatter_bivariate` / `build_netceded_joint`
  helpers live in `aggregate.multivariate`; the old `aggregate.bivariate`
  module is removed (import from `aggregate.multivariate`).

- New `tests/test_multivariate.py` (37 cases); DecL mirrored in
  `test_decl.agg` (section MV). The `t` copula, ≥3-variate `rfftn` path,
  and a `MultivariatePortfolio` are scoped as later stages in the plan.

## 1.0.0a23

### The `pnl` keyword — premium-minus-loss aggregates

A profit is premium minus loss, and the premium is collected **once for the
book**, not once per claim. `pnl` makes that a first-class object — a sibling
of `agg` that builds an ordinary loss aggregate and applies an
aggregate-level affine wrapper `PnL = premium − A`. It is the natural producer
of payoff-typed objects and goes anywhere an `agg` goes, so a `port` of
`pnl` lines is a book-level underwriting-result distribution (via the signed
combine landed in `1.0.0a22`). See `dev/done/plan-pnl-premium.md`.

- **Syntax.** `pnl NAME <premium> prem - <loss-exposure> <sev> <freq> …`.
  Three exposure heads: `70% lr` (binds to the stated premium,
  `E[loss] = premium·lr`), `10 claims` (frequency-driven), `85 loss`
  (expected-loss-driven). The premium **vectorises** like an `agg` exposure —
  `pnl X [100 200 100] prem - .8 lr [1000 2000 5000] xs 0 sev …` shifts by
  `Σ premium` and reports the total P&L only.
- **Once for the book, not per claim.** A constant inside `sev`/`dsev`/
  `ssev` is multiplied by the claim count; the `pnl` premium is a single
  deterministic shift. `pnl P 100 prem - 5 claims …` (mean `100 − E[A]`) is
  deliberately **not** `agg 5 claims ssev 100 - …` (mean `5·(100 − E[X])`).
- **No new numerics.** The loss FFT, its validation, and every ordinary
  aggregate are byte-for-byte unchanged (gated behind `agg_reflect` /
  `agg_shift` defaults). The affine is a pure grid relabel of the finished
  density: `mean → premium − E[A]`, `sd` unchanged, `skew → −skew`;
  `ftagg_density` is rebuilt in the combine convention so a book of `pnl`
  units convolves with no combine-side change. The P&L window is a tight,
  mass-centred two-sided window (`estimate_agg_window` on the affine moments).
- **Signed-aware ``describe`: SD instead of CV.** `CV = sd/mean` is
  meaningless as the mean → 0 (a P&L straddling break-even), so for **any**
  signed object — a `pnl` *or* a `ssev` / negative-`dsev` aggregate — the
  moment table now shows an **SD trio** (`SD | Est SD | Err SD`) instead of CV.
  This also cleans up the `1.0.0a22` signed-portfolio `describe`. Non-signed
  output is unchanged.
- **P&L readout.** `info` reports `premium` / `E[loss]` / `E[margin]` /
  `loss ratio` / `P(loss)` (`= P(PnL < 0)`, read straight off the signed
  density). `value_type` is set to `payoff` — finally giving that member a
  job (consumed by the pricing plan).
- New `tests/test_pnl.py` (15 cases); DecL mirrored in `test_decl.agg`
  (section PnLprem). Reinsurance gross/ceded-premium P&L and general aggregate
  algebra are split out to `dev/TODO-Remember.md` (items 6c / 6b).

## 1.0.0a22

### Portfolio combine on signed (profit/loss) support

Second half (`Portfolio` scope) of the negative-x work in
`dev/done/plan-negative-x-port.md` — the *combine*. A portfolio of independent
signed (P&L) units now aggregates correctly onto a shared signed grid, so a
book that straddles 0 is a first-class object alongside the single-unit P&L
landed in `1.0.0a21`.

- **Window-aware combine.** Each unit keeps its **own** optimal signed window
  `[x_min_k, x_max_k)`; the portfolio insists only on a shared `bs` /
  `log2` / `padding`. The FFT product is origin-at-0 because each unit's
  `ftagg_density` is independent of that unit's `x_min` (the output roll
  hits the density, never the transform), so the units multiply correctly and
  the total is placed on `[x_min_tot, ...)` by a single F2 `np.roll`. This
  replaces the old truncating `ift` (which silently dropped the wrapped
  negative tail) — `p_total` now conserves mass on signed support.
- **Driven on own grids.** Units are updated on their own signed windows
  (not a 0-based grid), so each unit object stays internally correct (no
  spurious deficit warning, right moments / `describe` / `plot`) — a strict
  improvement in instrumentation over a shared-origin drive.
- **Coarsen-to-fit bucket.** New signed-aware `Portfolio._bs_window` (a thin
  wrapper on `best_bucket`, recorded in a unit-indexed
  `Portfolio._bs_window_df`) sizes the shared grid: the summed support is
  wider than any unit's but `2**log2` is capped, so `bs` is the *coarser*
  of the RMS recommendation and the fit floor `W_tot / N` (buy the space,
  avoid aliasing). A fine-lattice unit coarsened by the shared grid surfaces
  its own per-unit deficit warning rather than failing silently.
- **density_df** `loss` / `p_total` / `p_{line}` / `F` / `S` are
  correct on signed support, and hence so are `q` / `var` / `tvar` (the
  index-agnostic `make_var_tvar` needs no change). `plot` is signed-aware
  (`_limits('range')` returns a two-sided window so the negative tail is no
  longer clipped), and `info` reports the realised signed window.
- **Pricing deferred.** Pricing / allocation columns (`add_exa` and
  everything it writes, distortion pricing, `value_type` consumption) assume
  a `loss ≥ 0` axis and are split out to
  `dev/plan-portfolio-neg-x-pricing.md`. A signed portfolio routes through
  the `add_exa=False` branch (F/S only); passing `add_exa=True` warns and
  falls back rather than emitting wrong numbers.
- The non-negative path is **byte-for-byte unchanged** — every signed path is
  gated behind `Portfolio._signed()`. The `build_many` Portfolio branch no
  longer pre-computes `best_bucket` (no back doors: `update` routes
  `bs=0` through `_bs_window` itself, mirroring the Aggregate fix). New
  `tests/test_negative_x_port.py` (14 cases); DecL mirrored in
  `test_decl.agg` (section PortPnL).
- **DecL: ``shift - dist` severity (premium minus loss).** A constant minus a
  distribution now parses as the natural P&L reading, e.g.
  `ssev 100 - lognorm 80 cv .2` — premium `100` minus a lognormal loss
  (severity mean `20`). It is exactly `-1 * X + 100` (reflect the
  distribution, then shift), composes with a scale (`100 - 2 * lognorm ...`),
  and like all reflection needs `ssev` to keep the signed support (plain
  `sev` clamps the sub-zero tail). One grammar rule (`numbers MINUS sev1`)
  \+ transformer; the unambiguous whitespace-separated minus means no existing
  program changes.

## 1.0.0a21

### Negative-support (profit/loss) severity and the output window

First half (`Aggregate` scope) of the negative-x work in
`dev/done/plan-negative-x-agg.md`. A *profit is a negative loss*, so an aggregate
can now live on a signed grid, making profit/loss (P&L) distributions a
first-class object.

- **Signed severity (F1).** Severity may take negative values -- a profit is a
  negative loss. The DecL severity family is now:

  - `sev` -- continuous, **clamps** its sub-zero tail at 0 (unchanged);
  - `dsev` -- discrete, **never clamps**; a negative atom (e.g.
    `dsev [-2 5] [.5 .5]`) auto-signs the aggregate, so
    `build('agg PnL 1e6 claims dsev [-1 10] [15/16 1/16] poisson')` works
    directly;
  - `ssev` -- **new**: continuous, **never clamps** -- the signed / P&L
    sibling of `sev` (e.g. `ssev 50 * norm + 10`).

  Signedness is a property of the severity declaration (`Severity.signed`),
  recorded at parse time -- which is what lets the analytic moments (and hence
  the automatic window) be correct before any FFT. The `update(..., signed=)`
  argument remains as an override. The separate `value_type` member
  (`'loss'`/`'payoff'`, default `'loss'`) records the *pricing* sign
  convention; it is **orthogonal** to `ssev` (signed does not imply payoff),
  inert for the distribution, and consumed only at the pricing layer.

- **Output window (F2).** `update(x_min=...)` places the aggregate on a window
  `[x_min, x_min + (2**log2)*bs)`; `x_min` may be negative. The default
  `x_min='auto'` resolves to `0` for an ordinary aggregate and to an
  automatic two-sided window for a signed one, so a P&L just works from
  `build` with no extra argument. The window is estimated from the analytic
  moments (new `estimate_agg_window(m, sd, skew, p)` -- reflected
  shifted-lognormal / -gamma fits with a symmetric/normal fallback; takes the
  standard deviation directly so the mean-zero case works), so a tight far-from-0
  lump (e.g. a Poisson(10^6) P&L concentrated near a *negative* mean) uses a
  small `bs` over a narrow window rather than paying for `[0, mean]`. The
  placement is a relabelling (single `np.roll` on the padded FFT buffer),
  exact for random as well as fixed frequency.

- **Bucket + window estimator.** `update` now runs up to three sizing methods
  and records them in an expert-inspectable `Aggregate._bs_window_df`, then
  selects: **exact_discrete** (a `dfreq`/`fixed` × `dsev` on an integer
  lattice has exact finite support -- `bs=1` and a minimal `log2`) \>
  **bounded_small** (a bounded severity with a small claim count -- window
  `[0, N_hi·s_max]` from a high frequency quantile) \> **moment** (the legacy
  3-moment sizing, reproduced exactly for non-negative aggregates;
  `estimate_agg_window` for signed). `log2` is a cap; pinning `bs` lets you
  keep full control of the grid.

- **Severity reporting moved to** `Aggregate.sev_density_df` (its own grid
  `xs_sev`). A windowed/signed aggregate and its severity no longer share a
  grid, so `p_sev`/`F_sev`/`S_sev`/`log_p_sev` left `density_df` for
  the new frame; `plot`, `q_sev`, `tvar_sev` and the error analysis are
  re-sourced. `q`/`tvar`/`var` work on signed support; `plot` is
  signed-aware (axes span the negative support). `info` renders discrete
  severities by their support (`atoms [-2 5]`, shortened for many) instead of a
  spurious `5 xs 0` layer, shows `window` / `value_type` / `signed severity`, and warns when the severity falls outside the output window. The
  default 0-based, non-negative path is byte-for-byte unchanged.

- Internals: `validate_discrete_distribution` gains `allow_negative`
  (`dfreq` still clamps claim counts; signed `dsev` preserves negatives);
  `SeverityDHistogram` places negative atoms correctly; signed severities use
  identity layering (no `x<0 -> 0` clamp) and raw moments. Signedness is the
  declaration only -- the `ssev` keyword is the **only** DecL change, and there
  is no `signed=` argument. New `tests/test_negative_x.py` (28 cases).

- **Deferred to the Portfolio half** (`dev/done/plan-negative-x-port.md`):
  portfolio combine on signed support, the full `Portfolio.density_df` column
  audit (esp. the price column / `add_exa`), and distortion/pricing
  consumption of `value_type`. The `ft.py` recentering helpers are not yet
  refactored to call the core path (follow-up).

## 1.0.0a20

### Joint (ceded, net) occurrence distribution via 2D FFT

- New `Aggregate.occ_bivariate(...)` returns a `BivariateDistribution`
  (new submodule `aggregate.bivariate`; submodule access only) holding the
  **joint** law of the aggregate occurrence ceded `C` and net `N` losses
  under an occurrence reinsurance program. The two margins are already
  available individually (`reins_density_df['p_agg_ceded_occ' | 'p_agg_net_occ']`);
  the joint law — their correlation, co-moments, reinsurer-vs-cedent
  dependency — was not, and the random claim count means it does not factor.
- The mathematics is the ordinary compound-distribution FFT with the 1-D
  transforms replaced by 2-D transforms: per claim, `(c(X), n(X))` lies on
  the line `c + n = X`, so placing the gross severity mass there builds a
  bivariate severity `S` and the joint aggregate density is
  `iFFT2(freq_pgf(n, FFT2(S)))` — valid because `freq_pgf(n, z)` is
  elementwise in `z`. Occurrence only (the aggregate-cover bivariate is
  degenerate). Per-axis bucket / window sizing is auto-derived from the
  univariate margins (with `bs_ceded` / `bs_net` / `log2_ceded` /
  `log2_net` overrides) and the net/ceded mass is scattered onto the 2-D grid
  by the active `reins_bucket` scheme.
- `BivariateDistribution` provides `.marginals()`, `.moments(max_order)`
  (mixed raw moments `E[C^i N^j]`), `.corr()` (Pearson; positive — a random
  count couples ceded and net), `.contour()`, and rich reprs. The marginals
  reproduce the univariate occurrence aggregates and the anti-diagonal `C+N`
  reproduces the gross aggregate, giving exact validation targets; auto-sizing
  generally yields a *finer* (more accurate) ceded grid than the model grid.
- New `tests/test_reins_bivariate.py` (31 cases); DecL cases added to
  `test_decl.agg` (section Z). An experimental docs subsection is pending a
  manual rebuild.

## 1.0.0a19

### Rationalized reinsurance reporting (Aggregate + Portfolio)

- Three new reinsurance objects on `Aggregate` replace the old fragmented
  surface:
  - `reins_density_df` — per-bucket gross/ceded/net densities with
    **consistent columns** regardless of which stages are configured (a
    missing stage contributes the no-cession values). Renamed from the legacy
    `reinsurance_df`: `p_agg_gross_occ → p_agg_gross` (the true gross
    aggregate) and the old `p_agg_gross` → `p_agg_subject` (the
    aggregate-cover input).
  - `reins_stats_df` — a per-layer layering summary (empirical, model-grid).
    Columns `(view, layer)` with `view` ∈ `occ|agg`: `Gross` (always),
    then per occurrence `layer.1` … and the `Ceded` / `Net` totals, then
    the aggregate layers and their `Ceded` / `Net` (no `Subject` column —
    it is the column flagged `output` below). **Occurrence layers are
    conditional** on reaching the layer: frequency is the penetrating count
    `n·P(X>attach)` and severity is the unconditional layer severity divided
    by `P(X>attach)` (so the layer aggregate mean is unchanged and aggregate
    layer means sum to `Ceded`); the `agg` row is the layer's actual FFT
    aggregate. `Ceded` / `Net` totals are unconditional (`Ceded` sev +
    `Net` sev = `Gross` sev). The aggregate block leaves `freq` / `sev`
    NaN (they don't combine). Meta rows: `share` / `limit` / `attach`
    (`Gross` = claim-count-weighted policy terms, share 1; occ `Ceded` =
    share-placed sum of limits, min attachment), `pr_attach` / `pr_detach`
    (ground-up exposure probabilities that the underlying loss attaches /
    exhausts the view — from the underlying severity `fz`, since the modeled
    severity is conditional and reads 0 at the policy cap), `pr_loss`
    (P aggregate \> 0), `lol` (loss on line = layer agg mean / placed limit),
    and `output` (0/1, marks each stage's output view). Plus
    `(freq|sev|agg, ex1|ex2|ex3|mean|cv|skew)` (`ex1` duplicates `mean`
    for `filter(regex=...)`).
  - `reins_describe` — the daily-driver per-stage summary, sharing the same
    **eight columns as** `describe` (`EX | Est EX | Change EX | CV | Est CV | Change CV | Sk | Est Sk`) and mirroring its **economic view**: `EX` /
    `CV` / `Sk` hold the *theoretic reference* — the leading view's exact
    pre-bucket moments (`Gross` for the occurrence block, `Subject` for the
    aggregate block) — held constant down each component; `Est *` is the
    per-view model output; and `Change = (Est − reference) / reference` reads
    two ways off one arithmetic: on the leading (Gross/Subject) row it is the
    numerical validation / rebucketing error (~0 under `linear`), and on the
    ceded / net rows it is the % impact of the cession on that moment. Follows
    the gross/subject convention — the occurrence block leads with **Gross**,
    the aggregate block leads with **Subject**. Frequency is reported
    *unconditionally* on the `Est` basis (mean `E[N]` only, so `freq × sev == agg` per view; cv / skew `NaN`) — consistent with `reins_stats_df`,
    whose conditional basis is confined to the per-layer `layer.k` columns; the
    leading `gross` row's `Est` frequency is left `NaN` to mirror
    `describe`. The `view` / `component` index labels are lower-case to
    match the other frames.
- New **Portfolio** reinsurance reporting (previously absent):
  `reins_density_df` / `reins_stats_df` / `reins_describe` give the
  end-to-end gross/ceded/net of the portfolio aggregate, convolving the
  per-unit gcn aggregate marginals under the existing independent-FFT
  machinery (means add: portfolio total = sum of unit means per view). All
  three return `None` when no unit cedes.
- Removed the redundant/confusing legacy objects: `reinsurance_df` (renamed),
  `reinsurance_audit_df`, `reinsurance_report_df`,
  `reinsurance_occ_layer_df`, the persistent `occ_reins_df` /
  `agg_reins_df` members, and the per-layer `_reins_audit_df_work` engine.
  The vestigial `F_*` (CDF) columns are dropped from the per-stage engine
  frame (the debug plot cumsums inline). `reinsurance_occ_plot` now reads
  from `reins_density_df`; the `occ_ceder` / `occ_netter` /
  `agg_ceder` / `agg_netter` step functions are retained for the exact
  (EX) path.
- Reinsurance reporting **labels are centralised constants** in
  `constants.py` (`REINS_LABEL_GROSS` / `SUBJECT` / `NET` / `CEDED` /
  `OUTPUT`). `describe`'s reinsurance view now leads with **Gross** (was
  "Subject") and labels the model-output column **Net** / **Ceded** / **Output**
  (the last for a mixed program, e.g. occ net of + agg ceded to — replacing the
  old "After"). The occurrence/aggregate ordering and the gross/ceded/net view
  order are canonical throughout (no longer alphabetical).
- New `tests/test_reins_reporting.py`; DecL cases added to `test_decl.agg`
  (section Y). Docs (`2_x_re_pricing.rst`, `2_x_cat.rst`) rewritten to the
  new API — pending a manual docs rebuild. `Re.Both` describe baseline
  regenerated for the Gross/Output relabel.

## 1.0.0a18

### Reinsurance rebucketing switch + layer-order validation

- New `Aggregate.reins_bucket` switch (`'linear'` default, or
  `'nearest'`) controls how net/ceded distributions are rebucketed onto
  the model grid. `'linear'` splits each off-grid value's mass across its
  two bracketing buckets, preserving the first moment **exactly**;
  `'nearest'` rounds to the closest bucket (≤ `bs/2` positional bias).
  Property + validating setter mirror `Portfolio.allocation_method` (clears
  cached reins frames on change); a new module constant
  `REINS_BUCKET_DEFAULT` and an `update`/`update_work`
  `reins_bucket=` kwarg thread it through. Reinsurance is baked in at
  `update`, so a post-build change needs a re-`update()`.
- `Aggregate._apply_reins_work` rebucket core rewritten: the old
  `groupby` → `interp1d` CDF-interpolation → `np.diff` scheme (an
  undocumented third method that did not cleanly preserve the mean, plus two
  `len(...)==1` special cases) is replaced by a vectorized `np.add.at`
  scatter (new `_rebucket_to_grid` helper). Same `reins_df` columns; the
  degenerate "all ceded → net is 0" case falls out naturally. Top-of-grid
  overflow piles into the last bucket (same mode as an aggregate deficit).
- `make_ceder_netter` now hard-errors on out-of-order or overlapping
  reinsurance layers via a new `_validate_reins_layers` check at its single
  choke point: attachments must be non-decreasing and layers must not overlap.
  Gaps are allowed — express one with a zero-share layer `0 po L xs A`.
- Baseline `Re.Both` snapshots regenerated (the only case affected; drift
  ~1e-5 relative, reflecting the more accurate mass-preserving rebucket).
  New `tests/test_reins_buckets.py`; DecL case `ReBucket` added to
  `test_decl.agg`.

## 1.0.0a17

### Refactor harness + Copy-on-Write opt-in

- New `tests/baseline/` characterisation harness for the v1.0 core-compute
  refactor: 10 deterministic cases (7 aggregates, 3 portfolios) snapshot
  `stats_df` / `describe` / `density_df` plus per-distortion
  `augmented_df` / `pricing_at` / `price()` to parquet at
  `rtol=1e-12`, with a pinned manifest recording versions + commit SHA.
  `tests/test_baseline.py` runs every case before reporting, collecting
  all divergences into one summary (see `dev/done/plan-baseline-harness.md`).
  Adds `pyarrow>=15` to dev extras.
- Pandas Copy-on-Write is now opted in at package import for pandas 2.x
  (pandas \>= 3.0 has CoW on as the default, so the option-setter is a
  conditional no-op there to avoid the deprecated-option warning).

Parser so/po disambiguation + mixture-arm perf guard
\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~~

- `so` (share-of) and `po` (part-of) reinsurance keywords are now
  true synonyms; the **number** sets the meaning. A literal percentage
  (`50% so 200 xs 100`, `50% po 200 xs 100`) is the share directly;
  a bare number (`5 so 10 xs 0`, `5 po 10 xs 0`) is an absolute
  amount and the share is `amount / limit`. Previously `50% po`
  divided the percentage by the limit (silent factor-of-200 error)
  and `5 so` returned the bare value as the share (out-of-range).
  Implementation: parser tracks the `%` suffix through a tiny
  `_PercentNumber` float subclass; arithmetic strips it, so an
  expression like `25 * 2` falls through as an absolute amount.
- New corpus cases `J.Re18a`..`J.Re18d` cover all four
  (keyword, percent-or-absolute) combinations and assert they collapse
  to the same `(share, limit, attach)` tuple.
- Mixture-arm `Aggregate.__init__` skips the ground-up-mixture
  `Severity` constructions when no exposure row carries a positive
  attachment — saves one Severity per mixture component on the common
  no-excess path. They were only needed for the `sf(attach)`
  re-weighting under excess covers.

### Portfolio cleanup (add_exa_details slim, swap_density_df, comments)

- `Portfolio.add_exa_details` slimmed to the still-meaningful EPD +
  reimbursement diagnostic columns (`epd_0_total`, `epd_0_{line}`,
  `epd_1_{line}`, `e1xi_1gta_*`). The legacy eta-mu /
  second-priority surface (`ημ_*`, `exeqa_ημ_*`, `e2pri_*`,
  `lev_ημ_*`, `exlea_ημ_*`, `exi_xgta_ημ_*`, `exa_ημ_*`,
  `epd_2_*`, `epd_0_ημ_*`, `epd_1_ημ_*`) and the
  `add_eta_mu()` companion method removed — they were
  `plot_twelve`-only defensive scaffolding, and `plot_twelve`
  doesn't actually read them.
- `Portfolio._build_augmented(efficient=False)` no longer computes
  `exi_xgtag_ημ_*` / `exag_ημ_*` (no consumers). `pedagogy.plot_twelve`
  no longer warms `add_exa_details(eta_mu=True)`.
- `swap_density_df` promoted from experimental method to standalone
  function in `aggregate.portfolio` (the method is now a thin shim).
  The function recomputes empirical stats via `xsden_to_mwrangler`;
  a swapped portfolio has no `mixed`/`independent` decomposition
  so those stats_df columns are left blank by design.
- `Portfolio.add_exa` / `Portfolio.update` journey-of-discovery
  comments scrubbed: commented-out alternative implementations,
  T.S. Eliot quote, `# TODO What is this crap?` markers, `Doh`
  asides, and dead chained-assignment-workaround blocks gone.
  The `ft_nots` argument of `add_exa` is now required (the
  `None`/`ημ_<line>`-fallback branch was dead since the eta-mu
  removal).

### Portfolio pricing & allocation (pentagon, linear default, ROE fix)

- `Portfolio.price` default flips to `allocation='linear'` (was
  `'lifted'`). Lifted natural allocation reads from the risk-adjusted
  `augmented_df` and is unstable on the right edge for a mass
  distortion on an unbounded support — essentially all the distortion
  weight lands on the last bucket. Linear collapses tail states with
  objective probabilities and stays bounded.
- New `Portfolio.allocation_method` member (`'linear'` /
  `'lifted'`) is the source of truth; the setter clears the
  `augmented_df` cache so the next `apply_distortion` rebuilds.
  Shown in `info`. `price(allocation=…)` still overrides on a
  one-off basis.
- New `Aggregate.bounded` / `Portfolio.bounded` property: `True`
  iff the frequency *and* every severity component is bounded
  (`fixed` / `bernoulli` / `binomial` / `empirical` frequencies
  and finite-support / layer-capped / splice-capped severities).
  Conservative — defaults to `False` whenever it cannot be proved
  `True`. Certify with `obj.bounded = True` (escape hatch for
  cases the heuristic misses).
- `Portfolio.price(allocation='lifted')` now **refuses** when the
  portfolio is unbounded and any requested distortion carries a mass
  (e.g. CCoC on Port.CNC); the error points the caller at
  `allocation='linear'` or the `bounded` override. Bounded
  portfolios (Bodoff, beta mixtures) still take lifted+CCoC unchanged.
- `Portfolio._build_augmented` de-duplicated: the total-level block
  (`exag_total`, `M.M_total`, `M.Q_total`, `M.ROE_total`,
  `roe_zero`) is now computed once with the correct L'Hôpital ROE
  fallback `ROE(1) = 1/g'(1) − 1`. Previously the `efficient=True`
  branch (the default) used `g'(1)` and disagreed with the
  `efficient=False` branch on the right edge — surfaces as numerical
  shifts on mass-distortion + tail cells (the baseline harness moves on
  Port.Bounded and PEG CCoC; non-mass distortions are unaffected).
  When `g'(1) = 0` (TVaR beyond the threshold) the limit is `+∞`
  and `M.Q_{line}/∞ = 0` falls through cleanly.
- `pricing_at` returns the pentagon order `L M P Q a | LR PQ ROE`
  (amounts then ratios; `a = P + Q` is now a first-class column,
  not a post-hoc decoration in `analyze_distortions`). The lifted
  and linear branches of `price` emit the same column shape.
  `PRICING_STAT_ORDER` / `PRICING_STAT_DTYPE` updated to match.
- Linear-branch `price`: the distortion-independent `exp_loss`
  integral (and the tail-collapse on `exeqa`) is hoisted out of the
  per-distortion loop. With *k* distortions the per-call cost drops
  from *k* full reverse-cumsum sweeps to one.
- Journey-of-discovery comments in `_build_augmented` and the linear
  `price` branch deleted; the surviving comments are short, current,
  and point at the equation numbers in PIR §14 where useful.

### Aggregate cleanups + forwards-S unification

- `Distortion.price` now defaults to `S_calculation='forwards'`
  (`S = 1 − cumsum`). Backwards is still available via the same kwarg.
  Forwards is the conservative, mass-preserving choice: under a genuine
  PMF deficit it carries the missing mass as a tail blob rather than
  silently dropping it. Aligns `Distortion.price` with the four other
  sites (`add_exa`, `_build_augmented`, `add_exa_sample`,
  `density_df.S`) that already use forwards by default.
- New `DefectiveDistributionWarning(UserWarning)` in
  `aggregate.constants`, emitted once per `update_work` when the
  aggregate PMF deficit `1 − Σp_agg` exceeds `VALIDATION_NOISE`
  (forwards and backwards `S` diverge by exactly the deficit, so the
  warning advertises the divergence at construction time rather than
  letting it surface silently in downstream pricing).
- New private `Aggregate._fft_aggregate` helper is now the single source
  of truth for the FFT-PGF-iFFT core. `_freq_sev_convolution`,
  `reinsurance_df`, and the subject-aggregate hook in `update_work`
  all delegate to it; the zero-risk and fixed-1 shortcuts live in one
  place.
- Redundant `est_*` moment writes inside `apply_occ_reins` and
  `apply_agg_reins` removed: `update_work` overwrites those fields
  immediately from the same densities using the de-fuzzed
  `xsden_to_mwrangler` worker. The unused
  `Aggregate.aggregate_keys` class attribute is also gone.

### Aggregate reinsurance reporting (Subject / Net / Change)

- `Aggregate.describe` becomes an economic view under reinsurance:
  columns are `Subject EX | <label> EX | Change EX | Subject CV | <label> CV | Change CV | Subject Sk | <label> Sk`, where `<label>`
  is `Net` (every cession passes the net), `Ceded` (every cession
  passes the ceded), or `After` (occ and agg pass different kinds).
  `Change = (after − subject) / subject` is the same column arithmetic
  as the legacy `Err` and reads either as the validation eyeball (no
  reins) or as the cession impact (under reins).
- Headings switch to the denser `EX` / `CV` / `Sk` form on both
  `Aggregate.describe` and `Portfolio.describe` (legacy
  `E[X]` / `CV(X)` / `Skew(X)` retired).
- `Portfolio.describe` now picks its column layout at the **portfolio**
  level so the unit blocks and the `total` block always agree. If
  **any** unit carries reinsurance the whole table flips to the economic
  Subject / `<label>` / Change view (the `total` Subject is the gross
  theoretical, `<label>` the realised after-reins); units with no
  cession are rendered in that layout too. With no reinsurance anywhere
  the table keeps the plain theory/empirical validation view. Previously
  the `total` block stayed in validation headings while ceding units
  used economic headings, so the columns misaligned under `pd.concat`.
  New `Portfolio._reins_after_label` aggregates the per-unit labels
  (one kind → that label, mixed → `After`); `Aggregate.describe` is
  refactored onto `Aggregate._describe(force_reins_label=...)` so the
  portfolio can impose one shared label on every unit.
- The scaffold `stats_df` columns from the previous iteration are now
  populated: `after_occ` (post-occ-reins moments, pre-agg-reins),
  `occ_impact` (after_occ / mixed), `agg_impact` (empirical /
  after_occ), and `gross_empirical` (subject empirical, via one extra
  FFT of `sev_density_gross` when occ-reins is present, free
  otherwise).
- `stats_df['error']` is now the subject-validation column:
  `gross_empirical` vs `mixed`. Under no reinsurance
  `gross_empirical == empirical` and this is exactly the legacy
  theoretical-vs-empirical column. Under reinsurance it is the only
  apples-to-apples check available (the after-reins object has no
  independent theoretical to validate against).
- `Aggregate.valid` now validates the SUBJECT under the hood and ORs
  in `Validation.REINSURANCE` when reins is present. `info` /
  `explain_validation` surface this as `reinsurance; subject not unreasonable` (or `reinsurance; subject fails ...`) so the user
  can tell whether the gross object is sound.

### Shared stats hygiene across Aggregate and Portfolio

- `stats_df` is now an all-`float64` frame on both `Aggregate` and
  `Portfolio`; the legacy `('meta','name')` string row has been removed
  (the name lives on `self.name`). All the `.astype(float)` casts at
  consumer sites are gone.
- Per-component columns are renamed from flat `comp_<i>` to the 2-D
  `e{e}.m{m}` form (exposure component × severity-mixture component).
  The limit-profile arm uses `m=0`; the mixture-product arm carries both
  indices.
- `stats_df` gains scaffold columns for the upcoming reinsurance
  reporting redesign: `after_occ`, `occ_impact`, `agg_impact`,
  `gross_empirical` (NaN-filled for now, populated when reinsurance
  reporting lands).
- `Aggregate.valid` and `Portfolio.valid` now read mean / aliasing
  signals straight off `stats_df['error']` -- single source of truth, no
  detour through `describe`. The hard-coded `eps**3` floor and `10×`
  aliasing ratio are replaced with named constants `VALIDATION_NOISE` and
  `ALIASING_RATIO` in `aggregate.constants`.
- `Portfolio.update` and `Portfolio.create_from_sample` now compute
  empirical aggregate moments via the de-fuzzed `xsden_to_mwrangler`
  worker -- the same convention `Aggregate.update_work` already uses
  (small absolute shift in `est_m`/`est_cv`/`est_skew` and downstream
  pricing -- the PEG / harness baselines move at ~1e-9 relative and are
  recaptured in this iteration).
- `Portfolio._write_empirical_stats` no longer inverts each unit's
  empirical severity `(mean, cv, skew)` back into raw moments via
  `MomentWrangler`; it reads `Aggregate.stats_df['empirical']` raw
  moments directly.
- Two new floor constants in `aggregate.constants` replace the bare
  numerics in `add_exa` / `add_exa_details`: `EXEQA_NOISE_FLOOR`
  (the `exeqa` decomposition-error truncation threshold, was `1e-4`)
  and `FT_NOISE_FLOOR` (the "build up the product" guard, was `1e-10`).

### Noise-aware validation, denoised `describe`, empirical raw moments

- `Aggregate.valid` / `Portfolio.valid` now test CV and skewness with
  `np.isclose` against a definite noise floor (`VALIDATION_NOISE = 1e-12`) instead of a relative error guarded only by `> 0`. This fixes
  spurious skew/CV failures for symmetric or low-skew distributions whose
  analytic value is exactly 0 but computes as floating-point dust -- e.g.
  `dsev [1:6]` (a fair die) no longer reports `fails sev skew, agg skew`.
- `describe` no longer displays floating-point dust: near-zero moment
  cells are snapped to 0, and the error columns fall back to absolute error
  where the theoretical value is ~0.
- `stats_df['error']` is now noise-aware (same fallback). The raw
  `empirical` and `mixed` / `total` columns retain their exact values.
- Empirical raw moments `ex1` / `ex2` / `ex3` are now populated in
  `Aggregate.stats_df['empirical']` for the `sev` and `agg` rows
  (`Portfolio` already did this).
- Empirical aggregate moments are now computed from a de-fuzzed *copy* of
  the FFT density (sub-machine-epsilon fp noise zeroed, the same
  `remove_fuzz` threshold `density_df` uses), so the stored higher
  moments -- notably skew -- are clean and grid-independent instead of
  picking up `x**3`-amplified far-tail noise. `agg_density` itself is
  left untouched as the raw FFT output.
- New `moments.ser_to_mwrangler(ser)` builds a `MomentWrangler` from a
  Series whose index is the support (e.g. `density_df.p_total`).
- `utilities.silence_warnings` now takes optional `category` / `message`
  / `module` arguments to scope what is suppressed.
- The `xsden_to_*` moment helpers now share a single public entry point
  `xsden_to_mwrangler` (returns a `MomentWrangler`), resolving the
  previous `meancv` / `meancvskew` tail-mass inconsistency and avoiding
  redundant moment passes where both raw and standardized moments are
  needed; a definitely defective distribution
  (`sum(p) < 1 - VALIDATION_NOISE`) is now logged at INFO.

## 1.0.0a16

### Distortion: atom-row stats_df, Kusuoka summary in describe

`describe` now ends with three Kusuoka-summary rows for every kind:
`mean_mass` (atom of $`\mu`$ at `p=0`), `max_mass` (atom at
`p=1`), and `interior_atoms` (boolean). The unambiguous names
replace the earlier `mass_at_0` / `mass_at_1` labelling.

`stats_df` drops those three rows and instead carries a variable-length
**atoms section** -- one row per Dirac atom of $`\mu`$, indexed
`mu_<p:.3f>`. The `closed_form` column holds the `p` value;
`D_g` holds the atom mass.

`MixtureDistortion._kusuoka_atoms` merges duplicate `p` across
members. `MinimumDistortion._kusuoka_atoms` detects atoms via two
sources: `brentq`-refined active-member transitions (mass via the
slope-jump identity $`m = s^* (g_i'(s^*) - g_j'(s^*))`$) and
boundary atoms inherited from the member active near `s = 0` /
`s = 1`.

### Portfolio.calibrate_distortion(s) cleanup

- `calibrate_distortion`: dropped the unused `df` parameter; default
  `r0` changed `0.0 → 0.05`. Docstring clarifies that `r0` is
  consumed only by the mass-at-zero kinds (`cll`, `clin`, `lep`,
  `ly`) and ignored otherwise.
- `calibrate_distortions`: dropped `r0` and `df` -- both were
  dead, since the calibrated-kind list is fixed to
  `[ccoc, ph, wang, dual, tvar]` (none take `r0`; the legacy `tt`
  kind that consumed `df` was removed earlier).
- Updated 7 `.rst` doc call sites from the legacy
  `calibrate_distortions(ROEs=[r], Ps=[p], strict='ordered')` to the
  current `calibrate_distortions(coc=r, p=p)`. Also fixed
  `port.dists[…]` → `port.distortions[…]` and `dist_ans` →
  `distortion_df` in the 10-min walkthrough prose.

### plot_twelve self-warming and bound-method fix

`pedagogy.plot_twelve` was silently relying on two preconditions the
user had to set up by hand. Now self-sufficient:

- Detects when the cached `augmented_df` is the lean
  (`efficient=True`) build (no per-line `M.M_<line>` columns), pops
  the cache entry, warms the eta-mu derivatives via
  `add_exa_details(eta_mu=True)` if needed, and rebuilds with
  `apply_distortion(distortion_name, efficient=False)`.
- Two stale `port.augmented_df.loc` / `.query` accesses (treating
  `augmented_df` as a property -- it's been a method since the
  `apply_distortion` refactor) now use the local `aug_df` variable.

### Package surface housekeeping

Each submodule declares its own `__all__`; the package `__init__.py`
is now a stack of `from .module import *` lines. Single source of
truth -- change what's public at the top level by editing the source
module, not `__init__`.

The `warnings.simplefilter('ignore')` block formerly run on package
import is gone. Library code should not mutate global state at import
time. The replacement is an explicit, opt-in helper:

``` python
from aggregate import silence_warnings
silence_warnings()    # mute warnings globally; user choice, not the
                      # library's
```

The four remaining `from .constants import *` lines (in
`distributions`, `utilities`, `spectral`, `portfolio`) were
replaced with explicit imports listing only the constants each module
actually uses.

### aggregate.parser_errors: structured DecL parse-error reports

New `aggregate.parser_errors` module turns Lark's terse parse
exceptions into structured `ErrorReport` dataclasses with line and
column, source-line echo, caret marker, friendly terminal labels, and
"did you mean..." suggestions via `difflib.get_close_matches`. With
Earley + dynamic lexer, almost every DecL parse failure surfaces as
`UnexpectedCharacters`; the formatter recovers the full mistyped word
by scanning forward through the DecL identifier character class, then
compares it against the parser-state's allowed terminal set.

The report is attached to every `build()` parse failure as
`e.report` (and `e.report.render()` gives the multi-line text
form). The wrapping `ValueError`'s `args[0]` is now a one-line
human-readable summary, so `str(e)` at the traceback tail reads
e.g. `DecL parse error at line 1, column 9: Unexpected 'cliams'. Did you mean: claims?` rather than the historical
`namespace(type='?', value='c', index=8)`. The Lark cause chain is
suppressed (`raise … from None`) so notebook tracebacks don't dump
Lark's internal `UnexpectedCharacters` frame; `e.report` carries
forward everything users actually need from it. Three opt-in recipes
(notebook print-then-raise, programmatic suggestion read, IPython
traceback hook) are documented in the "Reading Parse Errors" section
of the DecL language reference.

Long source lines are windowed around the caret with word-boundary
snap and ellipsis markers (`...` / `...`) so the marker stays
visible on a single terminal row. The rendered block uses a tight
layout: the "Did you mean" suggestion and the "Expected" list both
appear inline on the same line as the "Unexpected ..." message, with
no blank breaks.

### decl.lark: keyword terminals now require word boundaries

Every DecL keyword terminal (`AGG`, `SEV`, `PORT`, `CLAIMS`,
`MIXED`, `DISTORTION`, `FREQ`, …) now carries a negative
lookahead `(?![a-zA-Z0-9._:~\-])` mirroring the ID-continuation
character class. Without this, Lark's dynamic lexer would peel a
keyword off the front of a typo like `aggx` and continue parsing as
if the user had written `agg x`, surfacing the error several tokens
downstream at the wrong column. The lookahead forces keywords to
match only on word boundaries — same trick Python's tokenizer uses
for `def` vs `define`. Typos like `aggx Re:MFV41 …` now report
`Unexpected 'aggx'. Did you mean: agg?` at column 1 instead of a
misleading column-6 error about `Re:MFV41`.

## 1.0.0a15

### `Distortion` info / describe / stats_df / density_df quartet

`Distortion` now exposes the same four-property quartet as `Aggregate`
and `Portfolio`: `info` (multi-line summary string), `describe`
(small `(D_g, D_g_inv)` DataFrame with checks block), `stats_df`
(single-column `D_g` table with closed-form and error columns), and
`density_df` (full grid: `g, g_inv, g_dual, g_dual_inv, g_prime, g_dual_prime, kusuoka`).

All four are lazy `cached_property` — zero cost if never accessed.
Parameter setters (`d.a = 0.5`) and calibration (`_finalize_calibration`)
both route through `_build()` which invalidates the cache, so
calibrate-then-read returns fresh tables. The cache survives pickling.

Closed-form moments are surfaced where available: `ph`, `wang`,
`dual`, `tvar`, `ccoc`, `beta`, `bitvar`, `wtdtvar`, `cll`,
`clin`, `lep`. Multi-knot kinds (`minimum`, `mixture`) and the
remaining kinds (`power`, `ly`) leave the `closed_form` column as
`NaN` and rely on numeric values. The `error` column gives an
instant readout of trapezoidal-grid accuracy.

A new `_density_knots()` hook is overridden on kinked kinds so the
grid splices in TVaR/BiTVaR/WtdTVaR kinks (and the cap points for
CLL/CLin); `Distortion.plot()` now reads from `density_df`, so
the plotted curve is consistent with the tables and benefits from
the same knot splicing.

A `_kusuoka_summary()` hook returns three rows surfaced in
`stats_df` — the atoms of the Kusuoka spectral measure $`\mu`$
at `p=0` and `p=1`, and a boolean flag for interior atoms in
`\mu` (True for `tvar`, `bitvar`, `wtdtvar`, and combinations
of these).

## 1.0.0a14

### `aggregate` matplotlib house-style

New `aggregate.style` module — single source of truth for plot styling,
shared by the docs build and any forthcoming server / notebook use. The
underlying style is shipped as `aggregate/data/aggregate.mplstyle`
(color, serif, `figure.figsize = 3.5, 2.45`, `figure.dpi = 300`,
constrained layout).

To use the style in JupyterLab, at the top of any notebook:

    import aggregate.style
    aggregate.style.use()

That mutates `matplotlib.rcParams` globally for the kernel and also
sets `pd.options.display.width = 120`. Any plots from that point on
use the house style.

Variants:

    # leave pandas alone (e.g. you've already configured display.width):
    aggregate.style.use(pandas=False)

    # scoped — only for one figure, restores prior rcParams on exit:
    with aggregate.style.context():
        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.show()

    # scoped with overrides — bigger figure for a screen demo:
    with aggregate.style.context(**{"figure.figsize": (7, 4),
                                    "figure.dpi": 100}):
        fig, ax = plt.subplots()

**One gotcha:** matplotlib's inline backend in Jupyter has its own
`figure.dpi` / `figure.figsize` defaults that it applies *after*
import. If cells imported `matplotlib` before you called `use()`,
the inline backend's defaults can sneak back. Safest pattern is to put
`import aggregate.style` / `aggregate.style.use()` as the **first**
matplotlib-touching lines in the notebook.

If figures still look wrong after that, force the inline backend's own
dpi explicitly:

    %config InlineBackend.figure_format = 'retina'   # or 'png'
    %config InlineBackend.rc = {'figure.dpi': 100}   # override for screen

Replaces `knobble_fonts` (formerly inlined in `docs/conf.py`); the
docs build now calls `aggregate.style.use()`. The B&W branch is
dropped (paperless commitment). `rc_params()` exposes the parsed
style as a dict for inspection / composition.

## 1.0.0a13

`Distortion` constructors take natural, kind-specific parameter names
instead of the generic `(name, shape, r0, df, col_x, col_y)` slots.

- New signature per kind (positional or kwarg):
  - `Distortion('ph', a=)`, `Distortion('wang', lam=)`,
    `Distortion('dual', b=)`, `Distortion('tvar', p=)`.
  - `Distortion('ccoc', d=)` *or* `Distortion('ccoc', r=)` —
    keyword-only; passing exactly one of `d` or `r` is required;
    positional `Distortion('ccoc', x)` raises `TypeError` (explicit
    over implicit). `Distortion.ccoc(d)` static factory unchanged.
  - `Distortion('bitvar', p0=, p1=, w1=)` — `w1` is the weight on the
    upper threshold `p1`.
  - `Distortion('wtdtvar', ps=, wts=)` — `ps` and `wts` are equal
    length; `wts` summing close to 1 is normalised silently, otherwise
    `ValueError`.
  - `Distortion('cll', r0=, b=)`, `Distortion('clin', r0=, slope=)`,
    `Distortion('lep', r0=, r=)`, `Distortion('ly', r0=, r=)`.
  - `Distortion('beta', a=, b=)`, `Distortion('power', x0=, x1=, alpha=)`.
  - `Distortion('minimum', distortions=)`,
    `Distortion('mixture', distortions=, wts=)`.
- Each scalar-shape subclass exposes its natural name as a read/write
  property (e.g. `d.a`, `d.p`); the writer re-runs `_build` so
  downstream cached state stays consistent.
- DecL grammar unchanged; `parser.py` translates the legacy
  `kind shape [df]` tuple to natural kwargs via a new
  `_distortion_spec` helper. Existing `distortion d1 ph 0.5`,
  `distortion d2 bitvar 0.5 [0.95 0.99]`, etc. all still parse.
- `ConvexDistortion` removed (it was a constructor, not a kind).
  Replaced by module-level `aggregate.spectral.convex_distortion(s, gs, *, display_name='')` that takes two raw arrays and returns a
  `WtdTVaRDistortion` whose piecewise-linear `g` matches the upper
  convex envelope. Companions `bagged_distortion` and
  `convex_example` are likewise module-level (not staticmethods).
- Removed: `Distortion.average_distortion`,
  `Distortion.bagged_distortion`, `Distortion.s_gs_distortion`,
  `Distortion.convex_example` staticmethods; `_plot_decorations`
  hook (presentation concern, not core behaviour).
- `power` is no longer calibratable through `Portfolio.calibrate_distortion`
  (`_calibration_init_shape` dropped; `strict_pricing=False`).
- Snapshot regression: `tests/data/distortion_g_snapshot.csv` pins
  `g` and `g_inv` at canonical parameter sets for every documented
  kind to `rtol=1e-10`.

## 1.0.0a12

`extensions/` package removed. Optional/auxiliary code consolidated into
top-level modules or migrated out:

- New: `aggregate.pedagogy` absorbs all doc-cited figure helpers
  (`adjusting_layer_losses`, `savings_charge`, `mixing_convergence`,
  `power_variance_family`, `plot_quantile_illustration` (was `fig_4_1`),
  `plot_discrete_distribution_quantile` (was `fig_4_5`),
  `plot_continuous_distribution_quantile` (was `fig_4_6`),
  `plot_tvar_quantile` (was `fig_4_8`),
  `plot_ruin_surplus_paths` (was `fig_9_1`), `natural_scale`) plus four
  curated, renamed PIR figures: `plot_distortion_and_ins_stats` (was
  `fig_10_3`), `plot_spectral_three_panel` (was `fig_10_5`), `plot_twelve`
  (was `twelve_plot`), `plot_bivariate` (was `biv_contour_plot`). Also
  `bodoff_exhibit` (now takes `port` as first arg, not `self`).
  `ClassicalPremium` pulled in to keep `plot_ruin_surplus_paths` working.
- New: `aggregate.pentagon` (was `extensions.pentagon`). Class plus
  the `mapper` / `make_possible_pentagons` helpers. Not re-exported
  from top-level `aggregate`; reach as
  `from aggregate.pentagon import Pentagon`.
- Deleted: `basic.py`, `samples.py`, `test_suite.py` (visual
  reporter; pytest now drives the test_suite.agg coverage),
  `bodoff.py`, `risk_progression.py`, `case_studies.py`,
  `portfolio_pir.py`, `pir_figures.py` and `figures.py`
  (cherry-picked into `pedagogy.py`; the rest deleted),
  `cnc.py` / `discrete.py` / `hs.py` / `tame.py` (PIR
  case-study runner scripts), and the entire `templates/` folder
  (HTML/Markdown scaffolding for the deleted exhibit pipeline; the
  package-data entry in `pyproject.toml` was dropped to match).
- PIR case-study reproduction: install `aggregate==0.30.1` in an
  isolated environment to get the legacy `CaseStudy` workflow. PMIR
  is a separate forward-looking project and does **not** reproduce PIR
  exhibits.
- Doc imports updated to point at `aggregate.pedagogy` (technical
  guides) and `aggregate.ft` / `aggregate.tweedie` (reference page).
  Case-studies user-guide page replaced with a redirect note. The stale
  `aggregate.extensions.ft_invert` examples in
  `5_x_nm_ft_conv_algo.rst` were removed.
- No backwards-compat shim. `from aggregate.extensions import ...` is
  gone; `from aggregate.pedagogy import ...` is the new path.

## 1.0.0a11

`Bounds` redesigned. The IME 2022 pricing-bounds class is now one-shot:
`Bounds(obj, premium, *, a=np.inf, line='total', n_p=257, n_s=513)`
runs the full computation at construction. Access `p_star`, `min_envelope`
(a coherent `Distortion`), `max_envelope` (a callable; not a Distortion
because max-of-concaves isn't concave in general), `min_envelope_hinges`
(the active `(p_lo, p_hi)` bracket at each `s`), `cloud_df`,
`weight_df` and `tvar_df` as properties.

- Accepted input types broadened: `Portfolio` (with `line=`),
  `Aggregate`, `pd.Series`, `pd.DataFrame` (first column = pmf).
- `p_star` solved with `scipy.optimize.brentq` after a dyadic coarse
  bracket on `k/256`. Adaptive p-knots densify the grid at
  `p_star ± 2^{-k}` for `k = 8..11` so the kink between the CCoC
  and TVaR regimes resolves cleanly.
- `cloud_view` → `plot_envelope`. `weight_image` → `plot_weights`.
- Renaming internal arrays to clarify the math:
  `p_knots` (TVaR thresholds, shape `(n_p,)`),
  `s_grid` (distortion eval points, shape `(n_s,)`),
  `tvar_x_p` (`TVaR_p(min(X, a))` at each knot),
  `tvar_hinges` (`min(1, s/(1-p))`, shape `(n_p, n_s)`).
- Removed: `principal_extreme_distortion_analysis`, `ped_distortion`,
  `quick_price` (uncalled), `t_mode` getter/setter and Gauss-Legendre
  branch, `add_one` flag (locked True), `make_tvar_function` (folded
  into the bounded TVaR cache), `tvar_with_bound` (ditto).
- Pedagogy helpers `similar_risks_graphs_sa`, `similar_risks_example`,
  module-level `plot_max_min`, and `plot_lee` moved to a new
  `aggregate.pedagogy` module. Not exported from top-level
  `aggregate`. `plot_max_min` and `plot_lee` previously exported
  from top-level — those exports dropped per the no-shim policy.
- `Portfolio.pricing_bounds` now raises `NotImplementedError` —
  pending rewrite against the new Bounds API. The matmul-shape bug
  reported on PEG (33977 vs 512) was a symptom of the legacy
  `Bounds.tvar_cloud` accepting a free-form `s` array; the new
  `Bounds` always uses a 513-point binary `s_grid`.
- `tests/test_bounds.py` (9 cases). Closed-form analytical pins:
  brackets `(p_star, p_hi)` at `premium = TVaR_{p_star}` carry
  weight zero, so the resulting cloud columns equal the
  `TVaR_{p_star}` distortion exactly. Arbitrary bracket reproduces
  the weighted-combination formula to `1e-10`.

## 1.0.0a10

`ft` consolidation. `FourierTools` and friends promoted from
`aggregate.extensions.ft` to top-level `aggregate.ft`. Reach for
the class via `from aggregate.ft import FourierTools` (submodule
access only, no top-level re-export — same treatment as `Tweedie`).

- The legacy procedural `ft_invert` function (~140 LOC) deleted.
  Its functionality is fully covered by the `FourierTools` class,
  which the module's own docstring already documented as the
  preferred replacement. Docs that reference `ft_invert` are stale
  and will be swept separately.
- Paper-figure helpers (`poisson_example`, `fft_wrapping_illustration`,
  `recentering_convolution`, `recentering_convolution_example`)
  retained in `aggregate.ft` for now. A future `aggregate.pedagogy`
  module will consolidate figure-generators from across the codebase
  (see CLAUDE.md TODO).
- `make_levy_chf` retained.
- Reach-back imports inside `ft.py` (`from .. import build, qd, Aggregate`)
  replaced by direct module imports — eliminates the
  partially-loaded-package fragility that drove tweedie's load-order
  dance in 1.0.0a9.
- `aggregate.tweedie`'s `FourierTools` import repathed
  (`from .extensions.ft` → `from .ft`).
- Light tidy: `FourierTools(object)` → `FourierTools`; stale
  `ft_invert` references in docstrings / assert messages / dead
  commented debug lines cleaned up.
- New `tests/test_ft.py` — small in-regression case asserting that
  `FourierTools` against a closed-form distribution
  (`scipy.stats.norm`) inverts to the analytic pdf.
- Old `from aggregate.extensions.ft import ...` will break — no
  shim per the no-backcompat policy in `CLAUDE.md`.

## 1.0.0a9 (in progress)

Tweedie consolidation. `Tweedie` class promoted from
`aggregate.extensions.tweedie` to top-level `aggregate.tweedie`.
`tweedie_convert` and `tweedie_density` moved out of
`utilities.py` into the same module; their public re-exports from
`aggregate` are unchanged. Public import path for the class:
`from aggregate.tweedie import Tweedie`.

- `Mode`, `Tweedie`, `tweedie_illustration` are NOT re-exported
  at top level — submodule access is the only path. `Tweedie` gets
  the same treatment as `Bounds` (peripheral-but-public).
- `make_test_suite` and `run_test` (interactive notebook scaffolds
  that read a CSV and `IPython.display` audit frames) deleted.
  Replaced by a small in-regression pytest module
  `tests/test_tweedie.py` covering `tweedie_convert` round-trip,
  `tweedie_density` at the mass-at-zero point, Tweedie class moments
  matching V(μ)=disp·μ^p, and the additive↔reproductive duality.
- Light tidy in the moved file: `Tweedie(object)` → `Tweedie`;
  dead commented imports / unused `Path` / `IPython.display`
  removed; `# noqa` annotation on `Aggregate` import dropped.
- `parser.py`'s lazy `tweedie_convert` import repathed from
  `.utilities` to `.tweedie`.
- Old `from aggregate.extensions.tweedie import ...` will break —
  no shim per the no-backcompat policy in `CLAUDE.md`.
- Docs (`docs/2_user_guides/DecL/100_tweedie.rst`, technical guide)
  still reference the old import path — pending a separate docs
  sweep.

## 1.0.0a8 (in progress)

Portfolio refactor sub-project E — stats consolidation. Six overlapping
`Portfolio` stats frames (`statistics_df`, `statistics`,
`report_df`, `report`, `audit_df`, `make_audit_df`) collapsed
into a single canonical `stats_df`. Public stats surface on
`Portfolio` is now exactly three things: `info`, `describe`,
`stats_df` — same shape as `Aggregate`.

- **``stats_df`** is a `DataFrame` with MultiIndex on
  `(component, measure)` rows (`meta` + `freq` + `sev` + `agg`
  blocks) and columns one-per-unit + `total` + `empirical` + `error`.
  Per-unit columns hold each `Aggregate.stats_df['mixed']` (the
  unit's own view); `total` is the portfolio-aggregate theoretical
  view (sum of each unit's `mixed`); `empirical` is the post-FFT
  combined view; `error = empirical / total - 1`.
- Column is `total` rather than Aggregate's `mixed`: at the
  Portfolio level there is no mixed-vs-independent distinction
  (mixed-vs-independent is an Aggregate-only concept that strips a
  single agg's freq mixing distribution).
- **Empirical column is fully populated**:
  - `('agg', *)` rows — raw moments `ex1` / `ex2` / `ex3` plus
    `mean` / `cv` / `skew`, computed straight from the
    portfolio-total FFT density (plain summation, no tail-mass
    correction — matches the PEG baseline numerics).
  - `('sev', *)` rows — raw moments and central moments,
    re-aggregated from each unit's empirical sev mean/cv/skew via a
    fresh `MomentAggregator`. `Aggregate.stats_df` stores only
    empirical mean/cv/skew for sev, so the raw moments are inverted
    via `MomentWrangler` before being fed to the aggregator.
  - `('meta', *)` rows for `limit` / `attachment` / `el` /
    `prem` / `lr` — copied across from `total` with implied
    `error = 0` (these are factual or sums of expected values,
    no FFT analog).
  - `('freq', *)` rows stay `NaN` in `empirical` — frequency is
    exact (no convolution operates on it); same convention as
    `Aggregate.stats_df`.
- **Meta totals tightened**:
  - `total[('meta', 'attachment')]` = `0` when every unit attaches
    at 0 (previously `NaN`); `NaN` only when units disagree.
  - `total[('meta', 'limit')]` = `max` across units (legacy
    convention preserved).
  - `total[('meta', 'lr')]` = `el / prem` when `prem > 0` else
    `NaN`.
- **``('agg', 'P99.9e')` row dropped** — Aggregate dropped the
  estimated-99.9th-percentile row in Stage 1c+; Portfolio follows
  suit. Percentile access via `port.q(p)` / `port.var_dict(p)`
  remains.
- `describe` and the headline `agg_m` / `agg_cv` / `agg_skew` /
  `est_m` / `est_cv` / `est_skew` now read from `stats_df`.
  `describe` total row surfaces empirical sev mean/cv/skew (was
  blank before — sev empirical only existed per-unit).
- `extensions.portfolio_pir.accounting_economic_balance_sheet` and
  `extensions.bodoff` updated to read `stats_df`. The remaining
  `case_studies` exhibit code keeps its old `audit_df` references —
  those extensions are slated for removal at 1.0 per the master plan.
- PEG regression baseline unchanged (numbers reproduce bit-identically
  at `rtol=1e-10`).

Housekeeping in the same release block:

- `extensions.portfolio_pir.gamma` (the ~136-LOC conditional layer
  effectiveness γ exhibit) and its `GammaResult` dataclass deleted —
  both were orphaned: not called from `make_all`, not exercised by
  any test, not referenced in any rendered doc.
- `Underwriter.__repr__` gains a one-line usage hint pointing at
  `.discover(regex)` — fills the discoverability gap left when
  `qshow` / `qlist` / `show` were removed in 1.0.0a1.
- Eight stale comments and docstrings across `distributions.py` /
  `utilities.py` / `portfolio.py` that still mentioned
  `statistics_df` / `report_ser` / `audit_df` (in their
  stats-consolidation sense) refreshed. Distinct
  `reinsurance_audit_df` / `reinsurance_report_df` attributes
  and the `audit_df` field on `AnalyzeDistortionResult` are
  unrelated and unchanged.

Utilities refactor — `aggregate/utilities.py` shrinks from 3,753 to
~700 LOC. The grab-bag module is reduced to a focused set of
cross-cutting helpers; dead code is removed; themed code moves into
new modules or back to its only caller.

- **Deletes (~1,300 LOC):**
  - `frequency_examples` / `axiter_factory` / `AxisManager`
    (~530 LOC of pedagogical scaffolding with no consumers).
  - `MomentAggregator.stats_series` (retired by Stage 1c+).
  - `test_var_tvar` (internal scaffold).
  - Plotting / formatting subsystem: `FigureManager`,
    `make_mosaic_figure`, `easy_formatter`, `knobble_fonts`,
    `style_df`, `friendly`, `GreatFormatter`, `sEngFormatter`,
    `show_fig` (~630 LOC). `aggregate` no longer touches the
    user's matplotlib settings.
  - Dead-import / alias cleanup: `html_title`, `suptitle_and_tight`,
    `ln_fit` alias.
  - Dead public helpers: `mv`, `qdp`, `introspect`,
    `get_fmts`, `sensible_jump`, `GCN` namedtuple.
  - Timer cruft and the commented `knobble_fonts(True)` call.
- **New modules:**
  - `aggregate/moments.py` — `MomentAggregator`, `MomentWrangler`,
    `xsden_to_meancv`, `xsden_to_meancvskew`.
  - `aggregate/iman_conover.py` — `iman_conover` + `ic_*`,
    `block_iman_conover`, `make_corr_matrix`,
    `random_corr_matrix`, `rearrangement_algorithm_max_VaR`.
- **Public ``*_fit`` family in ``distributions.py`:** symmetric
  `(m, cv[, skew])` → distribution-parameter cluster, all importable
  from `aggregate`: `lognorm_fit` (renamed from
  `mu_sigma_from_mean_cv`), `gamma_fit`, `beta_fit`,
  `invgamma_fit`, `invgauss_fit`, `sln_fit`, `sgamma_fit`.
  Plus `approximate_from_mcvsk` (renamed from `approximate_work`),
  `lognorm_approx`, `lognorm_lev`. The `ln_fit` alias is
  dropped — `lognorm_fit` is canonical.
  `approximate_from_mcvsk`'s gamma branch now calls
  `gamma_fit(m, cv)` — symmetric with the lognorm branch using
  `lognorm_fit`.
- **Private single-module helpers moved + privatised** (no public
  surface change beyond the underscore): `_estimate_agg_percentile`,
  `_picks_work`, `_moms_analytic` + `_partial_e` +
  `_partial_e_numeric`, `_integral_by_doubling`,
  `_logarithmic_theta` all moved into `distributions.py`.
  `_parse_note` (the merge of `parse_note` + `parse_note_ex`)
  moved into `underwriter.py`. `_short_hash` moved into
  `spectral.py`.
- `make_comonotonic_allocations` moved to `portfolio.py` as a
  public module-level function (paired with the `Portfolio`
  method of the same name). Named locally
  `make_comonotonic_allocations_work` to avoid clashing with the
  method; re-exported cleanly as `make_comonotonic_allocations`.
- `Aggregate.plot` and the `bounds.py` `FigureManager` call
  site rewritten to plain matplotlib (`plt.subplot_mosaic`,
  `plt.subplots`). `extensions/case_studies.py` mpl call sites
  converted likewise.
- `pprint` renamed to `decl_pprint` (the DecL syntax-highlighter
  helper; avoids stdlib name collision).
- Documentation: `mu_sigma_from_mean_cv` → `lognorm_fit` updated
  in `2_x_actuary_student.rst`, `2_x_re_pricing.rst`, and
  `5_x_rearrangement_algorithm.rst`. Other rst pages pending a
  full sweep.
- PEG regression baseline unchanged (numbers reproduce bit-identically
  at `rtol=1e-10`); 430 pytest cases pass.

Packaging: src/ layout. The package source moved from
`aggregate/` to `src/aggregate/`. The src layout prevents accidental
imports from the source tree when CWD is the repo root — the only way
to `import aggregate` is now via the installed (editable) package,
which makes editable installs behave identically to wheel installs.
`pyproject.toml` gains `package-dir = {"" = "src"}`; `MANIFEST.in`
grafts repathed; `docs/conf.py` `sys.path` insert updated to
`../src`; four test/capture files repathed to
`src/aggregate/agg/test_suite{,2}.agg`. No public API change; 430
pytest cases still pass.

## 1.0.0a7

Portfolio refactor sub-project D — distortion-pricing pipeline redesign.
Six related changes that together collapse ~500 LOC of pricing code into
a small cache + a single signature convention:

- **D.1 — augmented_df lazy-eval cache.** `Portfolio.apply_distortion`
  becomes a thin cache lookup-or-build keyed on distortion name; the
  construction logic lives in a private `_build_augmented`. Second
  calls return the cached frame (`frame_a is frame_b`).
  `port.augmented_dfs` is a dict view of the cache; `port.augmented_df`
  is the clean read-side accessor (also routes through the cache).
  `apply_distortion` drops the `df_in=` (gradient path, gone),
  `create_augmented=` (the cache replaces it), and `plots=`
  (uninvoked) kwargs. `apply_distortions` (plural) deleted. New
  `Portfolio.pricing_at(distortion, *, p=None, a=None)` consolidates
  the row-extraction logic that previously lived in `price` and
  `analyze_distortion`.
- **D.2 — analyze_distortion(s) and calibrate_distortions collapse onto
  the cache.** Each former 100-250 LOC method becomes ~25 LOC.
  `analyze_distortion(distortion, *, p=None, a=None)` returns an
  `AnalyzeDistortionResult` dataclass with `pricing_df` and
  `audit_df`. `analyze_distortions(*, p=None, a=None, distortions=None)` returns `AnalyzeDistortionsResult` with the
  multi-distortion exhibit (MultiIndex `(distortion, stat)`) and a
  cache snapshot. `analyze_distortions2` and the list-based
  `calibrate_distortions(LRs=, COCs=, ROEs=, As=, Ps=, …)` deleted in
  favour of single-coc / single-p forms.
- **Explicit ``p=`` / ``a=` convention** across the pricing surface
  (`pricing_at`, `analyze_distortion`, `analyze_distortions`,
  `calibrate_distortions`). The legacy implicit `p > 1 → asset`
  threshold is gone; callers state intent. Each method raises
  `ValueError` if both or neither is supplied.
- **D.3 — Answer → typed dataclasses.** The legacy `Answer` dict
  class is deleted. `aggregate.results` defines `PricingResult`,
  `PricingBoundsResult`, `AnalyzeDistortionResult`,
  `AnalyzeDistortionsResult`, and `GammaResult` (the last used by
  `extensions.portfolio_pir`). Inline `namedtuple` definitions in
  `Portfolio.price` and `Portfolio.pricing_bounds` promoted to the
  same module.
- **D.4 — ordered categoricals.** `aggregate.spectral.DISTORTION_ORDER`
  / `DISTORTION_DTYPE` (`ccoc, ph, wang, dual, tvar, wtdtvar, lep, ly, clin, tt, cll, bitvar, blend`) and
  `aggregate.portfolio.PRICING_STAT_ORDER` / `PRICING_STAT_DTYPE`
  (`L, LR, M, P, PQ, Q, ROE`) bake the canonical order into the data.
  `Portfolio.distortion_df` `method` index level, `pricing_at`
  columns, and `analyze_distortions` pricing_df `distortion` level
  are typed categoricals -- `sort_index()` produces the canonical
  order without ad-hoc reordering.
- **D.5 — renames.** `Portfolio.dists` → `Portfolio.distortions`;
  `Portfolio.dist_ans` and the `distortion_df` property merged into
  a single `Portfolio.distortion_df` attribute with the trimmed 9-col
  layout (`S, L, P, PQ, Q, COC, param, std_param, error`) and index
  names `('a', 'LR', 'method')`; `Portfolio.limits` →
  `Portfolio._limits` (internal helper).
- PEG regression baseline unchanged -- `test_pricing` at `rtol=1e-8`
  reproduces the legacy `analyze_distortions2` exhibit bit-identically.
  The new pipeline is mathematically the same; only the API surface changed.

## 1.0.0a6

Portfolio refactor sub-project C — distortion calibration moves to the
`Distortion` subclasses themselves:

- `Portfolio.calibrate_distortion` was ~240 LOC of per-name Newton
  iterations in a giant `if name == 'ph': ... elif name == 'wang': ...`
  switch. Each branch defined a local `f(shape) → (residual, derivative)`
  closure and ran a hand-rolled Newton loop. That code now lives on the
  `Distortion` subclasses — each pricing-distortion class owns its own
  `calibrate(S, bs, premium_target, *, ess_sup, assets, el, **kwargs)`
  method: `PHDistortion`, `WangDistortion`, `DualDistortion`,
  `TVaRDistortion` (`max_iter=200`), `CCoCDistortion` (closed-form,
  no iteration), `LYDistortion`, `CLinDistortion`, `LEPDistortion`,
  `CLLDistortion`.
- `Portfolio.calibrate_distortion` shrinks to ~100 LOC — about half
  asset/S resolution (unchanged), about half dispatch to the subclass via
  `Distortion._registry`. The `tt` (Wang-t) branch is gone — there is
  no `TtDistortion` subclass to host it and the branch was dead code.
  `wtdtvar` calibration is also dropped from the dispatcher (the
  parametrisation overload between calibration form `(w, [p0, p1])` and
  the standard form `(ps, wts)` was already broken in the constructor;
  pick a pricing distortion that calibrates cleanly instead).
- New `Distortion` base-class methods `_newton_iterate(f, shape, *, max_iter, tol)` and `_finalize_calibration(shape, fx, prem, assets)`
  factor the Newton loop and the post-iteration bookkeeping (write
  `shape` / `error` / `premium_target` / `assets`, log on
  non-convergence, re-run `_build` to refresh cached state) out of the
  per-subclass methods.
- Class attribute `Distortion._calibration_init_shape` is the
  per-kind starting shape used both to construct the uncalibrated
  distortion and as the Newton iteration's starting point. `None` on
  the base means "not calibratable through the Portfolio dispatch."
- Each subclass is now testable in isolation. New
  `tests/test_distortion_calibrate.py` (12 cases) exercises every
  migrated kind directly on a synthetic `S` vector and asserts the
  achieved premium matches the target.
- PEG regression baseline unchanged — the new subclass-based Newton
  iteration reproduces bit-identical Newton convergence.

## 1.0.0a5

Portfolio refactor sub-project B — drop approximation and tilting paths
from `Portfolio.update` and `Aggregate.update_work`:

- Removed the auto-fallback method-of-moments approximation path. The
  `approx_freq_ge` / `approx_type` / `approximation` kwargs are gone
  from `Portfolio.update`; the matching `approx_type` /
  `approx_freq_ge` attrs are gone from `Portfolio.__init__`,
  `Portfolio.json`, and `Portfolio.__repr__`. The
  `'exact' if agg.n < approx_freq_ge else approx_type` ternary is gone;
  callers always get the FFT path. The slognorm / sgamma branch in
  `Aggregate.update_work` (and the `approximation` attribute on
  `Aggregate`) is deleted. `Portfolio.approximate` /
  `Aggregate.approximate` (the user-facing on-demand
  method-of-moments fit returning a `scipy.stats` frozen RV or a DecL
  spec) are unchanged.
- Removed FFT tilting (Grübel/Hermesmeier 1999) from the update pipeline:
  the `tilt_amount` attr is gone from `Portfolio.__init__`, the
  `tilt_vector` construction block is gone from `Portfolio.update`,
  and the `tilt=` parameter is removed from the `ft` / `ift`
  module-level helpers in `aggregate.utilities` and the matching
  `Portfolio.ft` / `Portfolio.ift` wrappers. The tilt branches inside
  `Aggregate.update_work`, `Aggregate._freq_sev_convolution`, and
  `Aggregate.apply_agg_reins` are gone. Use more buckets if aliasing
  shows up — per author's standing preference.
- `aggregate.extensions.figures.gh_example` was the only consumer of
  tilting in the visualisation layer; it now compares the padded FFT
  result against the exact compound probability without the
  tilt-comparison loop.
- PEG regression baseline (`tests/data/peg_baseline.json`) re-captured
  against the exact FFT path. The previous baseline incidentally
  exercised slognorm — PEG's two units (n=100 and n=150) tripped the
  default `approx_freq_ge=100` threshold. The drift is ~5e-6 on
  `est_m` and ~2e-5 on pricing cells; the new contract is the
  exact-FFT result.

Portfolio refactor sub-project A — pure deletions + PIR move
(`portfolio.py` shrinks from 6,133 → 3,707 LOC):

- Deleted ~700 LOC of dead code from `Portfolio`: `gradient` (~196 LOC),
  non-spectral allocations (`merton_perold`, `cotvar`,
  `equal_risk_var_tvar`, `equal_risk_epd`), the EPD / priority /
  collateral family (`analysis_priority`, `analysis_collateral`,
  `priority_capital_df`, `epd_2_assets`, `assets_2_epd` properties
  plus their backing attrs), the `uat` / `uat_differential` /
  `uat_interpolation_functions` trio, `collapse`, `audits`,
  `stat_renamer`, and the `var_dict(kind='epd')` branch.
- Stripped `analyze_distortion_add_comps` and
  `analyze_distortion_plots` (~470 LOC) — both consumed the deleted
  allocation methods. `analyze_distortion` keeps `add_comps` and
  `plot` parameters as no-op defaults (`add_comps=False` now).
- Moved ~1,800 LOC of PIR-exhibit machinery to the new
  `aggregate.extensions.portfolio_pir` module as free functions taking
  a `Portfolio` as the first argument: `premium_capital`,
  `multi_premium_capital`, `accounting_economic_balance_sheet`,
  `make_all`, `show_enhanced_exhibits`, `set_a_p`,
  `profit_segment_plot`, `natural_profit_segment_plot`,
  `density_sample`, `biv_contour_plot`, `twelve_plot`,
  `short_renamer`, `gamma`, `stand_alone_pricing`,
  `stand_alone_pricing_work`, `calibrate_blends` (with helpers
  `check01` / `make_array` / `convex_points`), the bulk
  constructors `from_DataFrame` / `from_Excel` /
  `from_dict_of_aggs`, and the big `renamer` plus
  `premium_capital_renamer`.
- `aggregate.extensions.case_studies` updated to call the moved
  functions as free functions; `aggregate.extensions.bodoff` inlines
  the deleted `cotvar` lookup.

## 1.0.0a4

Portfolio refactor sub-project 0 — PEG regression baseline:

- New regression fixture `tests/peg.py` exposes `build_peg` which
  constructs the canonical two-unit `port PEG` Portfolio (limit-and-attachment severity, three-component lognormal severity mixture per
  unit, gamma frequency mixing with different mixing CVs per unit).
- New capture script `tests/capture_peg_baseline.py` runs PEG through
  `calibrate_distortions(COCs=[.15], Ps=[.995])` and
  `analyze_distortions2(.995)` for the five-distortion suite
  (`ccoc`, `ph`, `wang`, `dual`, `tvar`) and writes the
  numerical baseline to `tests/data/peg_baseline.json`.
- New test module `tests/test_portfolio_peg_regression.py` pins
  portfolio moments (`rtol=1e-10`), per-distortion calibration shapes
  (`rtol=1e-8`, `|error| < 1e-5`), and every cell of the
  `analyze_distortions2` exhibit (120 values, `rtol=1e-8`).
- Every subsequent Portfolio refactor sub-project (A through E) must
  reproduce these baseline numbers; the JSON is the contract.

`Aggregate` stats consolidation — finish the job: eliminate the
`_statistics_df` / `_statistics_total_df` scratch frames so `stats_df`
is the only theoretical-moment DataFrame the class holds:

- `Aggregate.__init__` now pre-creates an empty `stats_df` (canonical
  `MultiIndex` rows, NaN-filled) right after `n_components` is known in
  each broadcasting arm, via a new `_init_stats_df` helper.
- `_record_component` writes a column of `stats_df` directly (no more
  intermediate row in `_statistics_df`).
- The post-loop totals block writes `mixed` / `independent` /
  `('meta', 'wt')` directly into `stats_df` columns.
- `('agg', 'P99.9e')` row dropped — it had only two populated cells
  (`mixed` and `independent`), was read in one spot (`_limits`
  fallback when `agg_density` is `None`), and is cheaply rebuildable
  on demand via `estimate_agg_percentile`. That one read site now
  computes on the fly.
- All readers migrated: `avg_limit` / `avg_attach` / `tot_prem` /
  `tot_loss`, `self.agg_m` / `agg_cv` / `agg_skew` / `sev_*`,
  `update_work` severity weights, `severity_error_analysis` weights,
  `info` / `_html_info_blob` component count.
- `_statistics_df`, `_statistics_total_df`, and the
  `_build_stats_df` method are gone.
- Side benefit: `stats_df` row layout is now cleaner — all `meta` rows
  together at the top (`mix_cv` and `wt` previously trailed at the
  bottom because of how the legacy scratch frames were ordered).

## 1.0.0a3

`Aggregate` stats consolidation: six overlapping moment DataFrames → one
`stats_df` (breaking changes; v1.0 cleanup):

- New canonical `Aggregate.stats_df`: single source of truth for moment
  statistics. `MultiIndex (component, measure)` rows (`component` ∈
  `{meta, freq, sev, agg}`; `measure` ∈ `{mean, cv, skew, ex1, ex2, ex3, …}`); columns are per-component (`comp_0`, …), `mixed`,
  `independent`, `empirical`, and `error`. Built in two phases:
  theoretical content in `__init__`, `empirical` and `error` appended
  in `update_work` after the FFT. Empty cells are `NaN` where
  meaningful (e.g. `('freq', *) × empirical` is undefined — the FFT
  produces one combined empirical distribution, not per-component
  empirical moments).
- Naming convention unified: `ex1` / `ex2` / `ex3` for raw moments
  and `mean` / `cv` / `skew` for derived. The legacy `_1` / `_m`
  flat-column convention is gone.
- The Aggregate "stats surface" is now exactly three things — `info` (text
  about the Aggregate), `describe` (the daily-driver moment audit), and
  `stats_df`. Removed: `report_df`, `report_ser`, `statistics`,
  `audit_df`. Privatised: `statistics_df` → `_statistics_df`,
  `statistics_total_df` → `_statistics_total_df`.
- `Aggregate.describe` rewritten to source from `stats_df`; output
  byte-identical.
- `Portfolio` migrated to read `a.stats_df['mixed']` instead of
  `a.report_ser` (three lines in `portfolio.py`). Portfolio's own
  `statistics_df` / `audit_df` / `report_df` are unaffected — they
  live on Portfolio, not Aggregate, and will be rationalised in Stage 2.
- Docs migrated: ~30 references to `report_df` / `statistics` /
  `statistics_df` across nine tutorial pages rewritten to use
  `stats_df` with explicit row / column accessors.

## 1.0.0a2

Aggregate surface rationalization (breaking changes; v1.0 cleanup):

- Visible layer structure: file-level section dividers in `distributions.py` and a public-API block in the `Aggregate` class docstring document the integration surface (`report_ser`, `statistics_df`, `update_work`, `agg_density`, `ftagg_density`, `density_df`, plus the risk-measure surface `q` / `tvar` / `cdf` / `sf` / …) that `Portfolio` and `Bounds` consume.
- FFT five-line core extracted to `Aggregate._freq_sev_convolution`; docstring references the four-step algorithm in §2.2 of the paper. `update_work` reads top-to-bottom as compute-severity → occurrence reinsurance → convolution → aggregate reinsurance → audit.
- Shared inner-block of `__init__`'s two broadcasting arms factored into `Aggregate._record_component` (centralises `statistics_df` column ordering across the limit-profile arm and the mixture-product arm).
- `__init__` state initialization regrouped into labelled blocks: spec passthroughs, grid + runtime config, exposure outputs, computed densities, empirical moment estimates, cached lazy functions, reinsurance state, theoretical moment tables.
- `density_df` property docstring expanded with a column-by-column reference table (set-by / read-by for each of 17 columns) — no behavior change.
- Aggregate methods privatised (leading underscore): `audit_df` → `_audit_df`, `statistics_total_df` → `_statistics_total_df`, `limits` → `_limits`, `html_info_blob` → `_html_info_blob`. `aggregate/extensions/figures.py` and `aggregate/extensions/test_suite.py` updated for the renames.
- `Aggregate.more`, `Portfolio.more`, `Underwriter.more` renamed to `.help`. Backing free function in `utilities.py` renamed `more` → `agg_help` (prefixed so it doesn't shadow Python's builtin `help` at module / package level).
- `pprogram` / `pprogram_html` collapsed: dropped the `split=20` line-magic and the `show=True` side-effect print. Methods preserved — cheat sheets and Underwriter consume them.
- Historical-comment sweep across the `Aggregate` class: stale `# TODO` / `# WHOA! WTF` markers and a commented-out spec-dict block removed.
- Logger calls in `distributions.py` converted to lazy `%s`-style formatting (extends the earlier `utilities.py` cleanup).
- Public surface intentionally retained after a docs audit revealed heavy tutorial usage: `statistics`, `statistics_df`, `report_df`, `report_ser`, `info`, `describe`, `snap`, `unwrap`, `picks`, `recommend_bucket`.

`Underwriter.build()` return contract uniform:

- `Underwriter.build()` now raises `CannotBuild` (subclass of `ValueError`) when a parsed spec produces no top-level object — previously returned a `ParsedProgram` with `object=None` in the named-mixed-severity edge case. The contract is now uniform: `build → object` always (or raises), `build_many → list[ParsedProgram]` always. `CannotBuild` is exported from the `aggregate` package.
- `Underwriter.discover()` catches `CannotBuild` and skips the row with a `logger.warning` (mirrors today's `NotImplementedError` handling).

Tooling:

- New `doc-test-uv.ps1` script: uv-managed doc build that replaces the clone-to-tmp dance in `doc-test.ps1`. Builds in place, uses a dedicated `.doc-venv` (set via `UV_PROJECT_ENVIRONMENT`) so doc builds don't disturb the main development `.venv`. Supports any Python via `--python X.Y` (uv auto-downloads if needed).

## 1.0.0a1

Underwriter surface rationalization (breaking changes; v1.0 cleanup):

- `Underwriter.discover(regex, kind='', plot=False, describe=False, return_objects=False, **kwargs)` replaces `show` / `qshow` / `qlist` (all three removed). Default behavior is the lightweight directory view (matches today's `qshow`); pass `plot=True` or `describe=True` to build each match.
- `Underwriter.build_many(program, ...)` is the explicit-batch counterpart to `build`; `build` now raises `ValueError` when its program produces 0 or \>1 top-level outputs (directing the user to `build_many`).
- `Underwriter.interpret_file(filename=None, where='')` replaces `interpret_test_file` and absorbs `run_test_suite`; with no arguments it runs the bundled test suite. Fixes a `KeyError: 0` bug from the pandas iterrows path.
- Directory rationalization: `site_dir`, `case_dir`, `template_dir` properties removed. Single new `user_dir` (`~/.aggregate`). `default_dir` is now located via `importlib.resources.files`.
- Base data directory moved from `~/aggregate` to `~/.aggregate` (dotted convention). No fallback — existing users must `mv ~/aggregate ~/.aggregate`.
- Constructor magic strings: `databases='all'` now expands to `['default', 'user']`; `databases='site'` raises `ValueError` directing users to `'user'`.
- Methods privatized (now leading underscore): `write` → `_build_work`, `factory` → `_factory`, `safe_lookup` → `_safe_lookup`, `interpret_program` → `_interpret_program`. `write_from_file`, `dir`, `test_suite()` method, `run_test_suite` deleted (all unused).
- Portfolio and case_studies internal callers switched from `uw.write(spec)` to `uw.build_many(spec, update=False)` (equivalent — same `ParsedProgram` list, no smart-update).
- `ParsedProgram` (dataclass) replaces `Answer` for the Underwriter parse-output type. `Answer` itself remains in `utilities.py` and continues to be used by `Portfolio`.
- `Underwriter.__repr__` clarified: shows `0 loaded (access .knowledge to read configured database(s))` when knowledge is pending; no I/O side effect.
- Several bug fixes: `factory` `ValueError` is now actually raised; the buggy "1 port among many" return path is gone; `__getitem__` `TypeError` → `KeyError` chain preserved with `from e`; `read_database` narrows to `OSError` and uses `logger.exception`.
- Three new constants in `constants.py`: `USER_DIR_NAME`, `PACKAGE_DATA_DIR`, `TEST_SUITE_FILENAME`.
- Internal cleanup: lazy `%s`-formatted logger calls throughout; ~130 lines of stale commented-out code removed from `utilities.py`.

## 0.30.1

- Confirmed support for Python 3.13 and 3.14

## 0.30.0

- Added `comonotonic_allocations` to `Portfolio` to implement the method of Denuit, Michel, et al. "Comonotonicity and Pareto optimality, with application to collaborative insurance." Insurance: Mathematics and Economics 120 (2025): 1-16. This uses numba if available. Warning: it can be very slow without numba!

## 0.29.0

- Portfolio analyze_distortions2 to iron out annoyances with current function but retain it for backwards compatibility.
- Portfolio calibrate_distortions2 for same reasons, args coc and reg_p.
- Spectral tvar_info_df and plot_affine for working with weighted TVaR distortions.
- Changed behavior of Distortion.random_distortion so that input number of knots *includes* mass and mean if present.
- Added random_distortion_ex(n=1, random_state=None) in Distortion class to simulate across types, extending random_distortion which is only a wtdtvar.

## 0.28.1

- `applymap` to `map` per Pandas update.

## 0.28.0

- Added `standard_shape` to Distortion and added to distortion_df created by Portfolio.calibrate_distortions.
- Updated dependencies and imports for doc build.
- Added `spectral.consistent_distortions` to create consistent family of representative distortions.

## 0.27.1

- Fixed a bug with recommend unit in a portfolio with all fixed components.
- Adjusted line styles in twelve plot and clarified use in doc string.
- Corrected ROE calculation of natural allocation premium when g(s) = 1.

### 0.27.0

- Removed control over logging and just use `logger = logging.getLogger(__name__)` in all modules. Removed `log_test` function and `LoggerManager` class.
- Removed `numba` as a requirement - huge library, hardly used. Only occurs in spectral module.
- Replaced build_docs batch file with doc-test which mirrors readthedocs process more closely.

### 0.26.0

- `extensions` no longer sets `pd.float_format` to Engineering.
- Added `tweedie.Tweedie` class to `extensions` to compute the Tweedie class distributions for
  all valid $`p`$. (Dangling jax dependence.)

### 0.25.0

- Tweak `extensions.ft.FourierTools`: added `invert_simpson` method using Simpson's rule,
  better for stable distributions. This is the method used by `scipy.stats`.
- Bumped to 0.25 which should have done in 0.24.2 because it added new functionality
- Tidied docs
- `knobble_fonts` uses serif font by default in matplotlib, and sets up
  in color mode by default.

### 0.24.2

- Added `Distortion.make_q` to return the risk adjusted probabilities used
  in pricing. Same logic as `price_ex`. Makes it easy to compute the natural
  allocation from a distortion.
- Added `extensions.ft.FourierTools` class, which performs direct inversion of a (continuous) Fourier transform (characteristic function)
  using FFTs. This is particularly useful for stable distributions, where the Fourier transform is known but the density is not. See examples in Section 5 of the documentation.
- Added `make_levy_chf` to `extensions` to compute the characteristic function of a Levy stable distribution.

### 0.24.1

- Added script to build the documentation from a local clone of the repository.
- Added `Aggregate.unwrap` to adjust aggregates computed with too few buckets
  but enough space. It unwraps the computed aggregate by adjusting the index. This
  reverses the "wagon-wheel" effect, whereby FFTs wrap-around the end of the array.
- Vectorized `ultilities.estimate_agg_percentile` for use in `Aggregate.unwrap`

### 0.24.0

- Added state to Distortions so they can be pickled. Involved separating part of `Distortion.__init__`
  into a new method, `Distortion._complete_init`. This is called from `__init__` and `__setstate__`.
  Ensured `_complete_init` refers to arguments as self.argname, not argname and set self
  variables in class `__init__` method.
- Fixed mixture g functions to handle input multidimensional arrays.
- Simplified `Distortion.__repr__` and `Distortion.__str__`.
- Added `Distortion.id` to generate a unique ID depending on `__dict__` argument elements.
- Corrected `g_prime` for minimum distortion.
- Fixed biTVaR distortion to handle p1==1 by including the mass explicitly.
- Added `Distortion.price_ex` to combine best of price and price2 methods and improve flexibility. It sorts and summarizes if needed. Optional return formats.
- Added four numba compiled functions to Distortion for fast computation of
  g.g(1-ps.cumsum()) and g.price( kind='ask'). These are tvar_gS, bitvar_gS,
  tvar_ra (for risk adjusted expected value) and bitvar_ra. In each case the
  values are computed without any copies of the original data, making them
  far more memory efficient for very large input arrays. At the extreme,
  bitvar_ra results in a speed up of the order of 2000x in realistic
  situations, even with small (100s) input vectors. The functions are static
  members of Distortion (numba requirement). They are not parallelized
  because of the cumulative computation of S. See the file
  PyWork/Distortion-price-tester.ipynb for tests (TODO: integraete into the
  documentation.) This addition results in numba being a required package.
- Removed dependency on `titlecase` package.
- Removed `Distortion.calibrate` method, which was not used and never tested. It lives with `Portfolio`.

### 0.23.0

- Added `sample_df` dataframe to `Portfolio` when created from a sample
  to store the sample. Original sample is needed in various applications.
- Added `swap_density_df(self, new_df, padding=1)` to `Portfolio`.
- Fixed errors in Case Studies caused by changes in Pandas.
- Added ability to create Markdown case output, rather than HTML.
- Added beta distortion (generalizes the PH and dual)
- Updated `np.alltrue` to `np.all`; updated `NoConverge` in `scipy.optimize`.
- Added `Distortion.calibrate` to calibrate to a pricing target from input `density_df` (TODO: needs testing).
- Added `wtdtvar`` to ``Distortion` to compute the weighted TVaR from p values and weights,
  masses and mean components.
- Added `minimum` to `Distortion` to create a new `Distortion` as the minimum of a list of input Distortions. The list is passed as shape.
- Added `random_distortion` to `Distortions` to compute a random distortion, useful
  for testing!
- Fixed `tvar` distortion to allow p=1 (max)
- Simplified `Distortion.__repr__` and `Distortion.__str__`.
- Added `Distortion.ph``, ``.wang`, ..., methods for common distortions, with better
  hints for parameters. All are static methods that delegate to the constructor.
- Fixed documentation build errors.

### 0.22.0

- Created version 0.22.0, "convolation" for AAS submission

### 0.21.4

- Updated requirement using `pipreqs` recommendations
- Color graphics in documentation
- Added `expected_shift_reduce = 16  # Set this to the number of expected shift/reduce conflicts` to `parser.py`
  to avoid warnings. The conflicts are resolved in the correct way for the grammar to work.
- Issues: there is a difference between `dfreq[1]` and `1 claim ... fixed`, e.g.,
  when using spliced severities. These should not occur.

### 0.21.3

- Risk progression, defaults to linear allocation.
- Added `g_insurance_statistics` to `extensions` to plot insurance statistics from a distortion `g`.
- Added `g_risk_appetite` to `extensions` to plot risk appetite from a distortion `g` (value, loss ratio,
  return on capital, VaR and TVaR weights).
- Corrected Wang distortion derivative.
- Vectorized `Distortion.g_prime` calculation for proportional hazard
- Added `tvar_weights` function to `spectral` to compute the TVaR weights of a distortion. (Work in progress)
- Updated dependencies in pyproject.toml file.

### 0.21.2

- Misc documentation updates.
- Experimental magic functions, allowing, eg. %agg \[spec\] to create an aggregate object (one-liner).
- 0.21.1 yanked from pypi due to error in pyproject.toml.

### 0.21.0

- Moved `sly` into the project for better control. `sly` is a Python implementation of lex and yacc parsing tools.
  It is written by Dave Beazley. Per the sly repo on github:

  The SLY project is no longer making package-installable releases. It's fully functional, but if choose to use it,
  you should vendor the code into your application. SLY has zero-dependencies. Although I am semi-retiring the project,
  I will respond to bug reports and still may decide to make future changes to it depending on my mood.
  I'd like to thank everyone who has contributed to it over the years. --Dave

- Experimenting with a line/cell DecL magic interpreter in Jupyter Lab to obviate the
  need for `build`.

### 0.20.2

- risk progression logic adjusted to exclude values with zero probability; graphs
  updated to use step drawstyle.

### 0.20.1

- Bug fix in parser interpretation of arrays with step size
- Added figures for AAS paper to extensions.ft and extensions.figures
- Validation "not unreasonable" flag set to 0
- Added aggregate_white_paper.pdf
- Colors in risk_progression

### 0.20.0

- `sev_attachment`: changed default to `None`; in that case gross losses equal
  ground-up losses, with no adjustment. But if layer is 10 xs 0 then losses
  become conditional on X \> 0. That results in a different behaviour, e.g.,
  when using `dsev[0:3]`. Ripple through effect in Aggregate (change default),
  Severity (change default, and change moment calculation; need to track the "attachment"
  of zero and the fact that it came from None, to track Pr attaching)
- dsev: check if any elements are \< 0 and set to zero before computing moments
  in dhistogram
- same for dfreq; implemented in `validate_discrete_distribution` in distributions module
- Default `recommend_p=0.99999` set in constsants module.
- `interpreter_test_suite` renamed to `run_test_suite` and includes test
  to count and report if there are errors.
- Reason codes for failing validation; Aggregate.qt becomes Aggregte.explain_validation

### 0.19.0

- Fixed reinsurance description formatting
- Improved splice parsing to allow explicit entry of lb and ub; needed to
  model mixtures of mixtures (Albrecher et al. 2017)

### 0.18.0 (major update)

- Added ability to specify occ reinsurance after a built in agg; this
  allows you to alter a gross aggregate more easily.

- `Underwriter.safe_lookup` uses deepcopy rather than copy to avoid
  problems array elements.

- Clean up and improved Parser and grammar

  > - atom -\> term is much cleaner (removed power, factor; now
  >   managed with prcedence and assoicativity)
  > - EXP and EXPONENT are right
  >   associative, division is not associative so 1/2/3 gives an error.
  > - Still SR conflict from dfreq \[ \] \[ \] because it could be the
  >   probabilities clause or the start of a vectorized limit clause
  > - Remaining SR conflicts are from NUMBER, which is used in many
  >   places. This is a problem with the grammar, not the parser.
  > - Added more tests to the parser test suite
  > - Severity weights clause must come after locations (more natural)
  > - Added ability for unconditional dsev.
  > - Support for splicing (see below)

- Cleanup of `Aggregate` class, concurrent with creating a cheat sheet

  > - many documentation updates
  > - `plot_old` deleted
  > - deleted `delbaen_haezendonck_density`; not used; not doing anything
  >   that isn't easy by hand. Includes dh_sev_density and dh_agg_density.
  > - deleted `fit` as alternative name for `approximate`
  > - deleted unused fields

- Cleanup of `Portfolio` class, concurrent with creating a cheat sheet

  > - deleted `fit` as alternative name for `approximate`
  > - deleted `q_old_0_12_0` (old quantile), `q_temp`, `tvar_old_0_12_0`
  > - deleted `plot_old`, `last_a`, `_(inverse)_tail_var(_2)`
  > - deleted `def get_stat(self, line='total', stat='EmpMean'): return self.audit_df.loc[line, stat]`
  > - deleted `resample`, was an alias for sample

- Management of knowledge in `Underwriter` changed to support loading
  a database after creation. Databases not loaded until needed - alas
  that includes printing the object. TODO: Consider a change?

- Frequency mfg renamed to freq_pgf to match other Frequency class methods and
  to accuractely describe the function as a probability generating function
  rather than a moment generating function.

- Added `introspect` function to Utilities. Used to create a cheat sheet
  for Aggregate.

- Added cheat sheets, completed for Aggregate

- Severity can now be conditional on being in a layer (see splice); managed
  adjustments to underlying frozen rv using decorators. No overhead if not
  used.

- Added "splice" option for Severity (see Albrecher et. al ch XX) and Aggregate,
  new arguments `sev_lb` and `sev_ub`, each lists.

- `Underwriter.build` defaults update argument to None, which uses the object default.

- pretty printing: now returns a value, no tacit mode; added `html` version to
  run through pygments, that looks good in Jupyter Lab.

### 0.17.1

- Adjusted pyproject.toml
- pygments lexer tweaks
- Simplified grammar: % and inf now handled as part of resolving NUMBER; still 16 = 5 \* 3 + 1 SR conflicts
- Reading databases on demand in Underwriter, resulting in faster object creation
- Creating and testing exsitance of subdirectories in Undewriter on demand using properties
- Creating directories moved into Extensions \_\_init\_\_.py
- lexer and parser as properties for Underwriter object creation
- Default `recommend_p` changed from 0.999 to 0.99999.
- `recommend_bucket` now uses `p=max(p, 1-1e-8)` if severity is unlimited.

### 0.17.0 (July 2023)

- `more` added as a proper method
- Fixed debugfile in parser.py which stops installation if not None (need to
  enure the directory exists)
- Fixed build and MANIFEST to remove build warning
- parser: semicolon no longer mapped to newline; it is now used to provide hints
  notes
- `recommend_bucket` uses p=max(p, 1-1e-8) if limit=inf. Default increased from 0.999
  to 0.99999 based on examples; works well for limited severity but not well for unlimited severity.
- Implemented calculation hints in note strings. Format is k=v; pairs; k
  bs, log2, padding, recommend_p, normalize are recognized. If present they are used
  if no arguments are passed explicitly to `build`.
- Added `interpreter_test_suite()` to `Underwriter` to run the test suite
- Added `test_suite_file` to `Underwriter` to return `Path` to `test_suite.agg` file
- Layers, attachments, and the reinsurance tower can now be ranges, `[s:f:j]` syntax

### 0.16.1 (July 2023)

- IDs can now include dashes: Line-A is a legitimate date
- Include templates and test-cases.agg file in the distribution
- Fixed mixed severity / limit profile interaction. Mixtures now work with
  exposure defined by losses and premium (as opposed to just claim count),
  correctly account for excess layers (which requires re-weighting the
  mixture components). Involves fixing the ground up severity and using it
  to adjust weights first. Then, by layer, figure the severity and convert
  exposure to claim count if necessary. Cases where there is no loss in the
  layer (high layer from low mean / low vol componet) replace by zero. Use
  logging level 20 for more details.
- Added `more` function to `Portfolio`, `Aggregate` and `Underwriter` classes.
  Given a regex it returns all methods and attributes matching. It tries to call a method
  with no arguments and reports the answer. `more` is defined in utilities
  and can be applied to any object.
- Moved work of `qt` from utilities into `Aggregate` (where it belongs).
  Retained `qt` for backwards compatibility.
- Parser: power \<- atom \*\* factor to power \<- factor \*\* factor to allow (1/2)\*\*(3/4)
- `` random` module renamed `random_agg `` to avoid conflict with Python `random`
- Implemented exact moments for exponential (special case of gamma) because
  MED is a common distribution and computing analytic moments is very time
  consuming for large mixtures.
- Added ZM and ZT examples to test_cases.agg; adjusted Portfolio examples to
  be on one line so they run through interpreter_file tests.

### 0.16.0 (June 2023)

- Implemented ZM and ZT distributions using decorators!
- Added panjer_ab to Frequency, reports a and b values, p_k = (a + b / k) [p](){k-1}. These values can be tested
  by computing implied a and b values from r_k = k p_k / [p](){k-1} = ak + b; diff r_k = a and b is an easy
  computation.
- Added freq_dist(log2) option to Freq to return the frequency distribution stand-alone
- Added negbin frequency where freq_a equals the variance multiplier

### 0.15.0 (June 2023)

- Added pygments lexer for decl (called agg, agregate, dec, or decl)
- Added to the documentation
- using pygments style in `decl_pprint` html mode
- removed old setup scripts and files and stack.md

### 0.14.1 (June 2023)

- Added scripts.py for entry points
- Updated .readthedocs.yaml to build from toml not requirements.txt
- Fixes to documentation
- `Portfolio.tvar_threshold` updated to use `scipy.optimize.bisect`
- Added `kaplan_meier` to `utilities` to compute product limit estimator survival
  function from censored data. This applies to a loss listing with open (censored)
  and closed claims.
- doc to docs \[\]
- Enhanced `make_var_tvar` for cases where all probabilities are equal, using linspace rather
  than cumsum.

### 0.13.0 (June 4, 2023)

- Updated `Portfolio.price` to implement `allocation='linear'` and
  allow a dictionary of distortions

- `ordered='strict'` default for `Portfolio.calibrate_distortions`

- Pentagon can return a namedtuple and solve does not return a dataframe (it has no return value)

- Added random.py module to hold random state. Incorporated into

  > - Utilities: Iman Conover (ic_noise permuation) and rearrangement algorithms
  > - `Portfolio` sample
  > - `Aggregate` sample
  > - Spectral `bagged_distortion`

- `Portfolio` added `n_units` property

- `Portfolio` simplified `__repr__`

- Added `block_iman_conover` to `utilitiles`. Note tester code in the documentation. Very Nice! 😁😁😁

- New VaR, quantile and TVaR functions: 1000x speedup and more accurate. Builder function in `utilities`.

- pyproject.toml project specification, updated build process, now creates whl file rather than egg file.

### 0.12.0 (May 2023)

- `add_exa_sample` becomes method of `Portfolio`
- Added `create_from_sample` method to `Portfolio`
- Added `bodoff` method to compute layer capital allocation to `Portfolio`
- Improved validation error reporting
- `extensions.samples` module deleted
- Added `spectral.approx_ccoc` to create a ct approx to the CCoC distortion
- `qdp` moved to `utilities` (describe plus some quantiles)
- Added `Pentagon` class in `extensions`
- Added example use of the Pollaczeck-Khinchine formula, reproducing examples from
  the `actuar` risk vignette to Ch 5 of the documentation.

### Earlier versions

See github commit notes.

Version numbers follow semantic versioning, MAJOR.MINOR.PATCH:

- MAJOR version changes with incompatible API changes.
- MINOR version changes with added functionality in a backwards compatible manner.
- PATCH version changes with backwards compatible bug fixes.
