# plan-tweedie: make the `tweedie` keyword survive the parse

**Label:** `[Tweedie-Round-Trip]`
**Status:** **DONE**, executed 2026-08-09. Phase 1 landed at `1.0.0a231`, Phase 2 at `1.0.0a232`.
**Closes:** `dev/TODO.md` `[Unparser-Reference-Gaps]` item 2 (both halves: the unparser gap and the "live second bug" note clobber).
**Breaking:** the clause argument order changed to `tweedie <p> <mean> <dispersion>`.

## Execution record

Both phases landed as planned, with three things worth recording.

**The `[Spec-Snapshot-Rename]` tidy went first**, so the re-capture in 1.6 ran under the new name. `tests/capture_sly_snapshot.py` is now `tests/capture_spec_snapshot.py`, and the false "IF the SLY parser is restored from git history" sentence in `CLAUDE.md` is replaced by the re-capture command plus a statement of what the snapshot proves.

**`dev/reflow_library.py` needed a change the plan missed.** It carries its own `NON_INVERTIBLE` regex, independent of `UNPARSER_EXEMPT` in the test, and that regex matched `\btweedie\b`. Removing the three entries from the test list was therefore not enough: the reflow script still held them back, and the count printed 17 rather than 14. The regex and the script's docstring are updated too.

**The monograph page carried a pre-existing break**, unrelated to Tweedie: its three-statement `build_many` program put one DecL statement per line with no `;` and no blank line, which stopped being legal when `[DecL-Newline]` landed. Repaired with `;` terminators while the page was open. Every `{python}` block on the page now executes.

Two things were checked and deliberately not done. `docs/2_aggregate_overview/features.rst` is maintained by the repeatable `dev/task-features.md` run, which computes its gap from the `CHANGELOG.md` feed, so a231 and a232 are picked up by construction and hand-patching the ledger would have fought the process. And `dev/check_features_rst.py` fails at block 80 (a `pnl` economics block, `features.rst` line 1763), which was verified to fail identically before this work: pre-existing drift, left alone.

## The problem

`agg A tweedie 1 1.005 0.1` is expanded at parse time into its compound-Poisson-gamma equivalent and nothing downstream ever learns a Tweedie was declared. `parser.agg_body_tweedie` (`src/aggregate/parser.py:604`) calls `tweedie_convert` and returns `exp_en=λ, freq_name='poisson', sev_name='gamma', sev_a=α, sev_scale=β`, plus a synthetic `_engine_note` string that `agg_out_named` (`parser.py:641`) promotes to the spec's `note`.

Three consequences, all live today.

1. **The keyword does not round-trip.** `a.pprogram` re-parses `a.program` and renders the expansion, so `tweedie 1 1.005 0.1` comes back as `10.050251256281404 claims / sev 0.0004999999999999894 * gamma 199.00000000000426 / poisson`. Three `library.agg` entries and one test-suite line are hand-exempted from the round-trip checks because of it.
2. **The synthetic note destroys the author's `note{}`.** `build.recipe('TweedieSimple').note` returns the machine string, not the sentence written in `library.agg`. `tags{}`, `hints{}` and `doc{}` all survive; only `note` is overwritten.
3. **The synthetic note is defective anyway.** `Tw(p=1.005, μ=1.0, σ^2=0.1) --> CP(λ= 10.0503, ga(α=199, β=0.0005), scale=0.0005` has an unbalanced `CP(`, prints β twice (once as `β`, once as `scale`, they are the same number), formats λ with `:8g` (width 8) where α and β use `:.8g` (precision 8), and is the only Aggregate-facing string containing `μ σ λ α β`, so reading `a.note` raises `UnicodeEncodeError` on a cp1252 Windows console.

## The design

Two mechanisms, deliberately kept apart, because they answer different questions.

**Provenance** answers "was this written as a `tweedie` clause?" and drives the *text*. It is a private spec key, `_tweedie`, holding the three declared parameters. Only the unparser reads it. An aggregate written the long way (`K.Tweedie0`) has no `_tweedie` and therefore still renders the way its author wrote it, which is the point: the unparser is the inverse of the parser, not a canonicalizer that rewrites people's programs into a spelling they did not choose.

**Recognition** answers "is this object a Tweedie?" and drives the *object*. It is a method, `Aggregate.as_tweedie()`, which reads `_tweedie` when it is set and otherwise derives the triple from the engine whenever the aggregate is an unlayered poisson × gamma. `tweedie_convert` already runs in that direction (`tweedie.py:53-60`, the `λ, α, β` branch). So `K.Tweedie0` and `K.Tweedie1`, both written the long way, answer too, and the feature stops being about the keyword and becomes about the mathematics.

### Why `_tweedie` is a constructor argument

The obvious alternative, popping `_tweedie` off the spec in the underwriter before construction, does not survive contact with the code: `Aggregate(**spec)` is called at six sites in `underwriter.py` (1115, 1251, 1277, 1308, 1322 and the bivariate sibling at 1367), two in `_portfolio.py` (211, 214) and three in `bivariate.py` (1162, 1217, 1381). There is no choke point, and `program` is a plain class attribute on `ProgramMixin` (`_program.py:130`), not a property with a setter, so there is nothing to hook. Eleven sprinkled pops is a bug waiting for a twelfth site.

One private argument, `_tweedie=None`, at the end of a signature that already carries around sixty, makes all eleven correct without touching any of them. The underscore marks it internal. It never appears in DecL, in `qd`, or in the reference docs.

Riding the provenance inside an existing free-form field was considered and rejected. The `note{}` route has real precedent (`_merge_note` with `replace_prefix`, `_program.py:359`, is exactly how `sharpen` stores its own machine sentence), but it means the unparser scrapes numbers back out of a human-facing prose field that the user can edit at will, and `decl_writer`'s own module docstring exists partly to say it never does regex surgery. `label_map`, `hints` and `tags` fail the same test for the same reason.

### The clause argument order changes

`tweedie <mean> <p> <dispersion>` becomes `tweedie <p> <mean> <dispersion>`. `p` is the shape parameter, it selects the member of the family, and it belongs first: the standard notation is `Tw_p(mean, dispersion)`, with `p` as the subscript. The rest of the module already agrees, `Tweedie.__init__`, `tweedie_convert`, `Tweedie.__repr__` and `to_series` are all p-first, and it was only the DecL clause that disagreed. Making them match is what lets one named tuple serve both.

This breaks existing programs, which the author has accepted on the grounds that there are very few in the wild. Mitigation is to make the break loud rather than silent, since a misread old program produces a *different valid-looking distribution* rather than an error.

Add an explicit range check in `agg_body_tweedie`: the compound Poisson-gamma representation exists only for `1 < p < 2` strictly, so anything outside that range is a DecL error naming the argument order. Most old programs die on it naturally, because their first argument was a mean: `tweedie 1 1.005 0.1` reads `p=1`, `tweedie 2 1.05 5` reads `p=2`, `tweedie 10 1.999 0.04` reads `p=10`, all rejected. Today those cases reach `tweedie_convert` and raise `ZeroDivisionError` from `α = (2 - p) / (p - 1)` with no explanation, so the check is an improvement on its own terms. It leaves one residual hazard, an old program whose *mean* happened to lie in `(1, 2)`, which will silently reinterpret. Call it out in `CHANGELOG.md` under a breaking-change heading.

The `1 < p < 2` restriction is exactly what `[Power-Variance-Family]` (`dev/TODO.md`, post-v1.0) would eventually lift, and it cannot be lifted here: outside that range there is no frequency × severity decomposition for the engine to build on.

### Why the `*` comes out of `Tweedie.__init__`

`Tweedie.__init__(self, p, *, mean=None, dispersion=1, theta=None, index=1)` is a dual-mode constructor: `(mean, dispersion)` selects reproductive, `(theta, index)` selects additive. Dropping the `*` gives `(self, p, mean=None, dispersion=1, theta=None, index=1)`, so `Tweedie(1.5, 100, 0.5)` reaches reproductive mode positionally and the additive path keeps its keywords. This is purely additive: every existing call site (`tweedie.py:288, 290, 306, 356`, `tests/test_tweedie.py:50, 58`, `tests/test_fcc_surface.py:820`) passes `mean` / `dispersion` / `theta` / `index` by name and is unaffected. Calls that previously raised `TypeError` now succeed; nothing that worked stops working.

## Phase 1: `[Tweedie-Provenance-Spec]` and `[Tweedie-Writer-Clause]`

These land together in one version bump. Phase 1 alone changes the parser spec without delivering the round-trip that justifies the change.

### 1.1 The named tuple

Add to `src/aggregate/tweedie.py`, beside `tweedie_convert`, following the `RuinFunction` precedent from `[Ruin-Wiener-Hopf]`:

```python
TweedieParameters = namedtuple('TweedieParameters', ['p', 'mean', 'dispersion'])
```

Field order matches `Tweedie.__init__` so `Tweedie(*params)` splats after 1.2, and after the reorder in 1.3 it matches the DecL clause too. Add it to `tweedie.py`'s `__all__` alongside `tweedie_convert` and `tweedie_density`. The `Tweedie` class itself stays out of `__all__`, unchanged.

### 1.2 `Tweedie.__init__` positional

Remove the `*`. Update the docstring sentence "All parameters except p, which is required must be given by name", which becomes false. Add a `Notes` line recording that reproductive `(p, mean, dispersion)` is positional so a `TweedieParameters` splats, and that the additive pair stays keyword by convention.

### 1.3 The parser: reorder, validate, stop clobbering

`parser.agg_body_tweedie` unpacks `(_tw, pp, mu, sig2)` in the new order, rejects `p` outside `(1, 2)` with a message naming the order, returns the same engine keys as today plus `"_tweedie": TweedieParameters(p=pp, mean=mu, dispersion=sig2)`, and drops `_engine_note` entirely.

`agg_out_named` (`parser.py:637`) loses the `body.pop("_engine_note", ...)` line and its `or trailer["note"]` fallback; `note` comes from the trailer like every other body form. `agg_source_inline` (`parser.py:682`) loses its matching pop. `_tweedie` is left in the fragment and flows through both wrappers untouched, so a tweedie embedded in a `pnl` or used as a `port` unit carries its provenance the same way.

The grammar rule `TWEEDIE expr expr expr` is unchanged in shape, so `decl.lark` needs no edit and `docs/4_agg_language_reference/ref_include.rst` needs no regeneration on account of the reorder. Check whether the rule carries an explanatory comment naming the old order.

Keep the derivation in the transformer. Moving it into the constructor was considered (it would match the `sev_mean` / `sev_cv` precedent, where the parser stores the declared parameterization and `__init__` derives the engine one) but it widens the blast radius to everything that reads a parsed-but-unbuilt spec, for no gain the writer can see.

### 1.4 `Aggregate.__init__`

Add `_tweedie=None` as the final parameter, after `doc=''`. Store it as `self._tweedie`. Document it in the parameter list as private provenance, set only by the parser, read by the unparser and `as_tweedie()`, and explicitly not part of the public constructor contract.

`self._spec` is captured wholesale from the constructor frame (`_aggregate.py:1953`) and pops only `frame`, `get_value`, `self`, so a new argument would change `_spec_hash()` (`:2852`) for every Aggregate in the library. Extend the pop list to drop `_tweedie` **when it is `None`**: hashes stay stable for everything that is not a Tweedie, and a real Tweedie keeps its provenance in `a.spec` so `Aggregate(**a.spec)` preserves it. Confirm nothing in `tests/data/` pins a spec hash.

### 1.5 The unparser branch

In `decl_writer`, the aggregate-body renderer gains an early branch: when the spec carries a truthy `_tweedie`, emit `tweedie <p> <mean> <dispersion>` through `_fmt_num` and skip the exposure, severity and frequency renderers entirely. `_fmt_num` (`decl_writer.py:93`) already renders integral floats without the `.0` and everything else through `repr`, which is the shortest string that round-trips, so `tweedie 1.005 1 0.1` comes back verbatim.

The trailer renders normally, so the author's `note{}` now appears. No suppression logic is needed anywhere, because nothing machine-generated is being hidden.

Update the "Canonical, not verbatim" section of the module docstring: Tweedie is no longer in the list of deliberately lossy rules.

### 1.6 Corpora, tests and data

Reorder the four in-repo clause occurrences, all currently `tweedie 1 1.005 0.1`, to `tweedie 1.005 1 0.1`:

- `src/aggregate/agg/_test_suite.agg:207` (`K.Tweedie2`). Its `note{}` already says "mean, p, and dispersion" and must be reworded.
- `src/aggregate/agg/library.agg:786, 798, 810` (`TweedieCompound`, `TweedieDispersion`, `TweedieSimple`). Their notes name the order too.
- `Tweedie.to_decl` (`tweedie.py:846`) emits the clause and must swap `{self.mean}` and `{self.p}`.

Then:

- `tests/test_decl_unparser.py`: delete `_FIDELITY_EXEMPT = {'K.Tweedie2'}` and the comment explaining it.
- `tests/test_agg_libraries.py`: remove `TweedieCompound`, `TweedieDispersion`, `TweedieSimple` from `UNPARSER_EXEMPT`. That test compares with `trailer=True`, so it only passes once the notes stop being clobbered; the two halves of this phase verify each other.
- `dev/done/reflow_library.py`: re-run. The three entries become machine-laid-out and the header comment in `library.agg` (lines 130 to 138) drops from seventeen hand-written exemptions to fourteen, with the `tweedie` sentence removed.
- `tests/data/expected_specs.json`: re-capture with `uv run python tests/capture_sly_snapshot.py`. The keys are the program text, so the four reordered lines change key as well as value. Diff the result and confirm only the Tweedie lines moved; the point of the snapshot is that an unexpected line moving is a finding.
- `src/aggregate/agg/decl-testers.agg`: add the reordered clause under the matching section if a tweedie case is added to the pytest suite, per the standing sync rule.

### 1.7 New tests

- Round-trip: `build('agg TW tweedie 1.005 1 0.1').pprogram` contains the `tweedie` clause and re-parses to an equal spec.
- Note survival: `build('agg TW tweedie 1.005 1 0.1 note{mine}').note == 'mine'`.
- Fixed point: `f(f(f(x))) == f(x)` for a tweedie statement, the invariant the `decl_writer` module docstring states.
- Order guard: `tweedie 1 1.005 0.1` (the old spelling) raises a DecL error naming the argument order, not `ZeroDivisionError`.
- Embedded: a tweedie inside a `pnl` and as a `port` unit both render the clause.
- No stray Unicode: assert the built object's user-facing strings encode to cp1252, guarding the crash class the old note introduced.
- Hash stability: a non-Tweedie aggregate's `_spec_hash()` is unchanged by this work.

## Phase 2: `[Tweedie-Live-Object]`

Separate version bump. Nothing here is needed for the round-trip.

### 2.1 `Aggregate.as_tweedie()`

Returns a `TweedieParameters`, or `None`. Reads `self._tweedie` when set. Otherwise recognizes: returns the derived triple when the aggregate is a plain unlayered compound Poisson-gamma, gated on `freq_name == 'poisson'`, `sev_name == 'gamma'`, a single unweighted severity component, no occurrence or aggregate reinsurance, no limit or attachment, no zero-modification or truncation, and `approximate == 'exact'`. Derivation is `tweedie_convert(λ=self.n, α=..., β=...)` read back as `(p, μ, σ2)`.

`as_` rather than `to_`: the house `to_*` verbs (`to_agg`, `to_frame`, `to_decl`, `to_series`) all export a whole object, so `as_tweedie()` returning parameters reads correctly and leaves `to_tweedie()` free should a live `Tweedie` ever be wanted. Name verified free on `Aggregate`, `Portfolio` and `Severity`.

Docstring must show the one-liner to the rich object, `Tweedie(*a.as_tweedie())`, which is what the `*` removal in 1.2 buys.

No change to `qd`, `_repr_html_`, or any other display surface. Those stay generic: they do not name a distribution family for any other object and there is no reason Tweedie should be the exception.

### 2.2 Documentation

- `docs/2_aggregate_overview/features.rst`, via `dev/task-features.md` with `dev/check_features_rst.py` as the gate: there is currently **no row at all** for the `tweedie` DecL clause. Its four existing Tweedie mentions are under-the-hood notes about the class and a dunder fix, so a reader scanning the feature matrix does not learn the keyword exists.
- `dev/FEATURES.csv` via `dev/regen_features.py`.
- `dev/TODO.md`: close `[Unparser-Reference-Gaps]` item 2 and renumber the surviving three.
- `CHANGELOG.md`: the argument-order break gets its own call-out, not a buried bullet.

### 2.3 The monograph page

`posts/3_user_guides/DecL/_100_tweedie.qmd` at `C:/s/AI/aggregate-monograph`, which is source of truth. Never edit the RST.

Five `build(...)` call sites need reordering (lines 17, 90, 102, 113, 149), and the surrounding prose names the old order in several places. More substantially, the page currently teaches the leak: it says "The note shows the compound Poisson specification" and builds its worked example around inspecting the converted spec to reveal the additive form. That framing inverts once the clause round-trips. Rewrite it around the declaration surviving, `as_tweedie()`, and the exact-density comparison, keeping the three-parameterizations table, which is still the clearest thing on the page. Re-render to confirm the executable blocks still run.

### 2.4 The cookbook page

The payoff, and the reason any of this has marketing reach. Tweedie is the one family in the library with a closed-form density, so `tweedie_density`'s series expansion can be plotted against the FFT aggregate. That is the accuracy claim demonstrated against an exact answer rather than asserted. Problem / Solution / Discussion / Check, per `[Cookbook-And-Docs-Fork]`.

## Resolved decisions

Taken by the author 2026-08-09.

1. **Clause order changes to `tweedie <p> <mean> <dispersion>`.** `p` is a shape parameter and comes first; `Tw_p(mean, disp)` is how people write it. Breaking existing programs is accepted, few exist. Loud failure via the `1 < p < 2` check is the mitigation.
2. **The named tuple is `(p, mean, dispersion)`,** now consistent with the class, the module and the clause.
3. **`expected_specs.json` is re-captured, not hand-patched.** See the correction below.
4. **No display changes.** `qd` stays generic and no "Declared as" line is added to `Aggregate._repr_html_`.
5. **`_tweedie` is popped from `_spec` when `None`,** so hashes are stable for every non-Tweedie object.
6. **The monograph page is in scope** and is kept current as part of this work.

## Correction: the spec snapshot is not frozen

`CLAUDE.md` states that `tests/data/expected_specs.json` "can be regenerated with `uv run python tests/capture_sly_snapshot.py` IF the SLY parser is restored from git history; otherwise treat it as a frozen reference". That is false, and it produced a wrong recommendation in the first draft of this plan.

`tests/capture_sly_snapshot.py` imports `aggregate.parser.UnderwritingLexer` and `aggregate.underwriter.Underwriter`, the **current Lark parser**. There is no SLY anywhere in the tree and the script does not want any. It has been re-run at least twice since the migration, at `d3d834f` (`[Renewal-Frequency-Wait-Clause]`) and `3c8885b` (`[Wait-Clause-Layers]`), both commit subjects saying "snapshot re-captured". Half a dozen plans in `dev/done/` treat re-baselining as routine.

So the snapshot is not a SLY artifact and has not been one for a long time. What it actually is: a **change detector** over the 326-line `_test_suite.agg` corpus, regenerated from the parser it tests. It cannot prove the parser correct, but it makes any grammar or transformer edit show its full blast radius as a diff you must look at and accept deliberately. `dev/done/refactor-plan.md` calls it the "spec-shape canary", which is the right description.

### `[Spec-Snapshot-Rename]`

A standalone tidy, requested by the author 2026-08-09. Pure renaming and documentation correction, no behavior change, so no version bump and nothing for Claude to commit.

**The name.** `capture_spec_snapshot.py` is recommended over `capture_parser_snapshot.py` on one-canonical-name grounds: the artifact is already called a spec snapshot everywhere it is referred to. The data file is `expected_specs.json`, the test is `test_spec_matches_snapshot`, and `dev/done/refactor-plan.md` calls it the "spec-shape canary". `parser` would name the producer rather than the artifact and introduce a second noun for one concept. The difference is marginal and the author's call.

**The docstring matters more than the name.** Two claims in it are false, independent of what the file is called. It opens "Capture (kind, name, spec) output from the current SLY parser" when it imports the Lark parser, and it says "This script is a migration artifact... the script's only future use is regenerating the snapshot if `_test_suite.agg` changes meaningfully", when in practice re-capture is routine after any grammar or transformer change and the corpus is the least likely trigger. Replace both with what the snapshot is for: a change detector over the corpus, regenerated deliberately, whose diff is the blast radius of a parser edit.

**Every live reference to the script name**, verified by grep:

| File | Line | What it needs |
|---|---|---|
| `tests/capture_sly_snapshot.py` | 1 to 11 | the rename, plus the docstring rewrite above |
| `CLAUDE.md` | 214 | the false "IF the SLY parser is restored from git history" sentence replaced |
| `README.md` | 196 | command reference in the maintenance block |
| `dev/TODO.md` | 292 | named in the dependents list of an existing item |
| `tests/test_decl_parser.py` | 29 | a comment naming the script as the source of the JSON sentinels |

Five files and the script itself. Earlier drafts of this plan named `test_doc_clause.py` and `test_reinstatement_decl.py`; both reference `expected_specs.json` but neither names the script, so they need no edit.

**Leave the historical record alone.** "SLY" also appears in `CHANGELOG.md`, `src/aggregate/parser.py`, `src/aggregate/decl.lark`, `docs/manual.bib` (a real citation to Beazley's SLY), and a dozen plans in `dev/done/`. Those are accurate statements about what happened, not stale instructions, and `CHANGELOG.md` is a standing rewrite carve-out. The distinction to apply throughout: correct anything that tells a reader what to *do*, preserve anything that records what was *done*.

## Deferred, not in this plan

**`[Tweedie-Reinsurance-Clauses]`.** The grammar rule is `TWEEDIE expr expr expr` with no reinsurance, layer, approximation or orientation slots, so `agg X tweedie 1.005 1 0.1 occurrence net of 0.5 xs 0.5` is a syntax error and the rename form is the only route. A user who thinks of `tweedie` as first-class will hit this. Small to add once `_tweedie` is on the spec, but it is a grammar change and belongs on its own.

**`[Tweedie-Named-Parameters]`.** Even p-first, `tweedie 1.005 1 0.1` gives no clue which number is which. A keyword spelling would read better and would be purely additive to the grammar. Not proposed, noted.

**`[Distortion-Declared-As-Asymmetry]`.** `Distortion._repr_html_` emits `Declared as <code>...</code>` (`spectral.py:1063`); no other first-class class does. Having ruled out adding one to `Aggregate`, the asymmetry now points the other way, at whether `Distortion`'s should go. Purely cosmetic, worth a line in `dev/TODO.md` at most.

**`[Power-Variance-Family]`.** Tracked in `dev/TODO.md`. The `1 < p < 2` check added in 1.3 is precisely the boundary that work would push on.
