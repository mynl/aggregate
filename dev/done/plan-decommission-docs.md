# [Decommission-Docs] Plan: retire the `doc{{{...}}}` clause and the cookbook

Status: APPROVED and EXECUTING, written 2026-08-17 against `1.0.0a297`. File and
line references were verified against the working tree at a297. The author
dispositioned every open question on 2026-08-17; the rulings are in section 9
and are folded into the sections they affect.

Author ruling, 2026-08-17: **right idea, wrong place.** The
`doc{{{...}}}` clause is too complex for the library, it mixes apples and
oranges, and long-form write-ups belong in
`C:/s/AI/notes/aggregate-presentations`, the staging ground for material that
ends up in the monograph. Testing that is worth keeping becomes explicit
pytest. At the same time the cookbook leaves the repo entirely: whatever is
left in `docs/cookbook` moves to the same staging ground and the directory
goes.

## 1. Why now, and why not later

**The grammar freezes at 1.0.** The CHANGELOG preamble puts `decl.lark` in the
stable tier: from 1.0 a documented name keeps its meaning and a breaking change
waits for a major release with a deprecation period ahead of it. `doc` is a DecL
keyword. Ship 1.0 and the actuarial DSL has promised, for the life of the major
version, to carry a keyword whose job is to hold a book. This is a pre 1.0b
decision or it is a 2.0 decision.

**The one argument that could have saved it does not hold.** If the app needed
the long-form write-up over the wire, the doc would have to live in the recipe
base. It does not. `aggregate_api/src/aggregate_api/examples.py` says so in a
comment and enforces it in code:

```python
# ``tags`` are carried in their own fields, and ``doc`` is never served. The doc
# pattern runs first and is non-greedy over newlines: in a *stored* program the
# body is the preprocessor's base64 one-liner, not the readable text, so it must
# never reach the editor.
_STRIP_CLAUSES = (
    re.compile(r"\s*doc\{\{\{.*?\}\}\}", re.S),
    ...
)
```

The app does not merely decline the doc. It has to defend against it, because
the stored form is a base64 blob that would otherwise reach a user's editor.
That is the clause paying rent in a downstream repo for a service nobody
consumes.

**The rest of the case, briefly.** The base64 lift exists because the lexer
cannot carry markdown in a slot built for a one line string, which is the
container telling you the payload is wrong. The Python inside is invisible to
ruff, to editors, to refactoring and to any useful traceback: a failing check
reports `<recipe LayerPicks:check>` with no line into a real file.
`library.agg` has become one file with three jobs, which is the same fault that
forced the `examples` / `cookbook` / `actuarial-severity-curves` merge, rotated.
And `[Recipe-Run-Clobbers-The-Trailer]` in `dev/TODO.md` exists only because a
recipe runs through the shared recipe base and re-registers itself over its own
entry.

**What is worth keeping** is the outcome, not the mechanism: before this
machinery the shipped library had exactly one assertion against it, that the
file loaded, and nothing in it was ever built. Section 6 keeps that, in
pytest, where it always belonged.

## 2. Scope: what is removed

Line counts are the a297 working tree.

| What | Where | Lines | Disposition |
|---|---|---|---|
| Doc bodies, 7 entries | `src/aggregate/agg/library.agg` | ~330 | to notes, then deleted |
| Doc fixtures, 6 statements | `src/aggregate/agg/decl-testers.agg` | ~20 | deleted |
| `DOC` terminal, `trailer_item_doc` | `src/aggregate/decl.lark` | ~25 | deleted |
| base64 lift and decode, `_encode_doc` / `_decode_doc` / `_DOC_FENCE_RE`, preprocess step 0, the `DOC` callback | `src/aggregate/parser.py` | ~60 | deleted |
| Doc half of `Recipe`: `doc`, four section properties, `extra`, `solution_code`, `check_code`, `sections`, `n_asserts`, `is_runnable`, `markdown`, `namespace`, `run`, `parse_doc`, `SECTIONS`, `DECL_PLACEHOLDER`, `_split_sections`, `_code_of`, `_headings_outside_fences` | `src/aggregate/recipe.py` | ~320 of 469 | deleted |
| Doc branch of the trailer writer | `src/aggregate/decl_writer.py` | ~15 | deleted |
| Doc fence lexing, markdown delegation | `src/aggregate/decl_pygments.py` | ~25 | deleted |
| `doc` in `TRAILER_KEYS`, the doc derived `_RECIPE_COLUMNS` (`doc`, `problem`, `solution`, `discussion`, `check`, `sections`, `n_asserts`) | `src/aggregate/underwriter.py` | ~40 | deleted |
| `"DOC"` label | `src/aggregate/parser_errors.py` | 1 | deleted |
| **`doc` kwarg and public attribute on the first-class classes**: `doc=''` in `__init__` and `self.doc = doc` on `Aggregate` (2010, 2293), `Portfolio` (111, 312), `Severity` (1103, 1225), `BivariateAggregate` (1270, 1289), plus the `doc=spec.get('doc', '')` passthrough at `underwriter.py:1480` | `_aggregate.py`, `_portfolio.py`, `_severity.py`, `bivariate.py`, `underwriter.py` | ~15 | deleted |
| Doc references in the `ProgramMixin` docstrings: the base64 paragraph on `program`, the `trailer` kwarg of `format_program` ("emits all four"), the `pprogram` metadata sentence, the `pin_sharpen` note | `src/aggregate/_program.py` | ~10 | rewritten |
| The whole renderer | `src/aggregate/cookbook.py` | 336 | deleted |
| Thin caller | `dev/generate_cookbook.py` | 30 | deleted |
| Cookbook source tree | `docs/cookbook/` | 620 plus `_freeze/`, `_site/`, `.quarto/`, `cookbook_files/` | to notes, then deleted |
| Tests | `tests/test_doc_clause.py` 331, `tests/test_cookbook_generate.py` 174, `tests/test_library_recipes.py` 107 | 612 | deleted |
| Doc parts of `tests/test_recipe.py` and `tests/test_grammar_sync.py` | | ~250 | trimmed |
| Prose | `docs/2_aggregate_overview/underwriter.rst`, `docs/3_reference/3_x_Underwriter.rst`, `docs/1_Getting_Started.rst`, `docs/2_aggregate_overview/features.rst`, `cheat-sheets/Underwriter_Cheat_Sheet.tex`, `src/aggregate/config.py`, `docs/conf.py`, the `library.agg` header | | ~80 | rewritten |

Roughly 1,600 lines of code and test go, plus the cookbook tree.

## 3. Scope: what survives, and must not regress

* **`Recipe` itself.** `kind`, `name`, `spec`, `program`, `object`, `source`,
  `note`, `tags`, `hints`, `decl`, `_render_decl`, `__repr__`. It stays the
  record the recipe base stores and `build_many` returns.
* **`Recipe.decl`.** Load bearing downstream: `aggregate_api` `_decl_of()` uses
  it to fill the editor, falling back to the stored program only when the
  unparser cannot render a spec. Do not touch it.
* **`note{}`, `tags{}`, `hints{}`.** These are metadata about an entry and
  belong in a trailer. They are not in scope and the distinction is the whole
  point: a note is a fact about the entry, a doc is a document.
* **`build.recipes`** as an audit frame, minus the doc derived columns.
* **`library.agg` as the one shipped library**, the default recipe base, and the
  source the app reads for its Examples menu. Untouched apart from losing seven
  doc bodies.
* **Every library entry.** Nothing is deleted from the library by this plan.
  a297 already did that pass.

## 4. Destination: one note per recipe

Notes go to `C:/s/AI/notes/aggregate-presentations`, whose convention is a
standalone `.qmd` per topic with YAML front matter, the project `_quarto.yml`
supplying `bibliography: C:/s/TELOS/Biblio/uber-library.bib` and the
journal-of-risk-and-uncertainty CSL. Model on `picks-mix-exp.qmd`: inputs in the
first cell, prose, then the numerical check as visible code rather than as a
hidden assert.

| Library entry | Current page | Proposed note | Note |
|---|---|---|---|
| `PHDistortion` | `_recipes_01_distortions.qmd` | `ph-distortion.qmd` | |
| `LayerPicks` | `_recipes_02_severity.qmd` | **fold into `picks-mix-exp.qmd`** | that note is already this material at length; do not create a third picks note |
| `SplitLimitPolicy` | `_recipes_02_severity.qmd` | `split-limit.qmd` | |
| `NeymanInnerOuter` | `_recipes_03_frequency.qmd` | `neyman-inner-outer.qmd` | check overlap with the existing `compound-frequencies.qmd` first; may fold |
| `LimitProfile` | `_recipes_04_aggregate.qmd` | `limit-profile.qmd` | |
| `ThreeDice` | `_recipes_04_aggregate.qmd` | `three-dice.qmd` | short; could open a wider "exact discrete aggregates" note |
| `OccurrenceXOL` | `_recipes_05_reinsurance.qmd` | `occurrence-xol.qmd` | |

Two more files need a decision rather than a move:

* **`docs/cookbook/_setup.py`** (185 lines) calls itself the single source of
  truth for example calibration and supplies exactly the names
  `Recipe.namespace` seeds. In the notes repo each note is self contained and
  does its own imports, so this is either a shared `_setup.py` there or it is
  dropped and its calibration constants inlined into the notes that use them.
  **Author call.** Recommendation: drop it, and let each note carry its inputs
  in its first cell the way `picks-mix-exp.qmd` already does. The centralization
  was solving a problem that only existed because five pages were generated into
  one book.
* **`docs/cookbook/plan.md`** carries the `[Check-*]` archetype vocabulary
  (reconciliation, scaling sweep, independent oracle, limiting case, round trip,
  cross object). That vocabulary is worth keeping: it is also the `check:` tag
  namespace in `library.agg` and is documented in the library header. Move the
  six definitions into the library header, then delete the file.
* `__prefob.qmd` is a local writedown preview shim with an absolute path to this
  worktree's venv. Delete, do not move.

## 5. Phases

Each lettered phase is one version bump and one commit, per the house rule.
Phase A touches a different repo and bumps nothing here.

**Order is load bearing in one place: C before D before E.** The explicit tests
must exist before the doc bodies are deleted, so coverage never dips, and the
doc bodies must be gone before the grammar stops accepting them, or the shipped
library will not parse.

### Phase A [Notes-Extraction], no bump, notes repo only, DONE 2026-08-17

Landed in `C:/s/AI/notes/aggregate-presentations`, five new notes and two folds,
every one rendered clean:

| Library entry | Landed as |
|---|---|
| `PHDistortion` | `ph-distortion.qmd` |
| `ThreeDice` | `three-dice.qmd`, widened to "Exact Discrete Aggregates" |
| `LimitProfile` | `limit-profile.qmd` |
| `OccurrenceXOL` | `occurrence-xol.qmd` |
| `SplitLimitPolicy` | `split-limit.qmd` |
| `LayerPicks` | folded into `picks-mix-exp.qmd` |
| `NeymanInnerOuter` | folded into `compound-frequencies.qmd` |

Three findings came out of the transcription, all recorded here because they
change what phase C should assert:

* **The `OccurrenceXOL` frequency check is vacuous as written.** It compares
  `reins_summary_df['EX']` across views, but that column carries the theoretic
  **gross** value in all three views, so it compares a number to itself. The
  real content is in `Est EX`, where ceded and net frequency both come back as
  the declared count and gross is `NaN` (a gross count is an input, not an
  estimate). Phase C asserts the `Est EX` form.
* **The `SplitLimitPolicy` doc overstates the first-bucket wisp** by a factor of
  ten: it says "about 1.6 in 10,000", measured is 1.71e-05, about 1.7 in
  100,000. The note computes it live so it cannot go stale again.
* **The `lev` route beats the note's quadrature by seven orders of magnitude**
  on the picks reconciliation (6e-12 against ~1e-4), so the fold brought
  `density_df['lev']` into `picks-mix-exp.qmd` as the exact reading.

The original phase A text follows.

Write the seven notes per the table in section 4, from the doc bodies in
`library.agg` and the rendered pages in `docs/cookbook/`. Nothing is deleted
here. This phase exists first so that no prose is ever only in a file that a
later phase removes.

Each note is prose, not a test: turn each `## Check` block into a visible
numerical demonstration with printed output, the way `picks-mix-exp.qmd` ends
with a table of layer integrals against the picks. The assertions themselves go
to pytest in Phase C, not into the note.

Verify: every note renders (`quarto render <name>.qmd`), and every one cites
through the project bibliography if it cites at all.

### Phase B [Cookbook-Removal], bump

Delete `docs/cookbook/` entire, `src/aggregate/cookbook.py`,
`dev/generate_cookbook.py`, `tests/test_cookbook_generate.py`. Remove the
`'cookbook/plan.md'` line from `docs/conf.py` exclude patterns and the
`Cookbook` section plus `.. automodule:: aggregate.cookbook` from
`docs/3_reference/3_x_Underwriter.rst`. Move the `[Check-*]` archetypes from
`docs/cookbook/plan.md` into the `library.agg` header, beside the `check:` tag
vocabulary they define.

Nothing imports `aggregate.cookbook` except its own test and
`dev/generate_cookbook.py`, so this phase is self contained and could equally
run last. It is placed early because it is the largest deletion and the least
entangled.

### Phase C [Library-Entries-Build-Check], bump

Lands `[Agg-Library-Build-Check]` from `dev/TODO.md`, which wants exactly this
and is the natural replacement: today `tests/test_agg_libraries.py` checks only
that each entry **parses**. New `tests/test_library_entries.py`:

1. **Every entry builds.** Parametrized over `library.agg`, one case per entry,
   `slow` marked where the grid is large (`LayerPicks` at `log2=20`,
   `SplitLimitPolicy` at 17, the `bs=2` hurricane entries). Assert the result
   carries the surface it should: `valid`, `validation_explanation`,
   `summary_df`, `stats_df`.
2. **The seven invariants, written out as ordinary asserts**, one test function
   each, transcribed from the `## Check` blocks. They are already correct and
   already passing; this is a move, not new work. Name each for its entry, for
   example `test_layer_picks_reproduces_every_pick`.
3. **A validation allow list.** Two entries fail validation deliberately and by
   design: `HeavyTailValidation` (infinite variance, the validator refuses to
   guess) and `LayerPicks` (picking is the decision to leave the declared
   severity's analytic moments). The test must name them and say why, so that a
   *new* failure is a finding rather than noise.

Watch: this test builds the whole library, so measure it and mark generously.
The a297 numbers say the three doc carrying entries build in about 1.6 s
together, and most entries are far cheaper.

### Phase D [Doc-Bodies-Out-Of-The-Library], bump

Strip the seven `doc{{{...}}}` bodies from `library.agg` and the six from
`decl-testers.agg`. Delete `tests/test_library_recipes.py`. Rewrite the
`library.agg` header: the `WRITING A doc` section, the `<<decl>>` convention,
the ` ```python ` fence convention, and the `HOW MUCH DOCUMENTATION` paragraph
all go, replaced by a pointer to the notes repo. Re-run
`dev/done/reflow_library.py` and re-run the a297 canonical layout test.

At the end of this phase the grammar still accepts `doc`, and nothing uses it.
That is the safe intermediate state, and it is where to stop if the author wants
to reconsider the grammar half.

### Phase E [Doc-Clause-Out-Of-The-Grammar], bump

The one breaking change, and the reason for the whole plan's timing.

1. `decl.lark`: delete the `DOC` terminal and `trailer_item_doc`, and rewrite
   the trailer comment block, which currently explains why four order-free
   optional items are a repetition rather than 65 spelled out alternatives.
   Three items keep that argument intact.
2. `parser.py`: delete `_DOC_FENCE_RE`, `_encode_doc`, `_decode_doc`, preprocess
   step 0, the `DOC` token callback, and `"doc"` from the trailer skip lists at
   lines 785, 1140 and the duplicate-trailer error message at 2199.
3. `recipe.py`: reduce to the record plus `note` / `tags` / `hints` / `decl`.
   Rewrite the module docstring, which is currently mostly about the doc
   contract, and delete the `.. warning::` about `Recipe.run` executing code,
   which no longer has a referent.
4. `decl_writer.py`: `TRAILER_ITEMS` loses `doc`; delete the emit branch and the
   comment about the doc fence spanning lines.
5. `decl_pygments.py`: delete `_doc_fence` and both `doc{{{` patterns.
6. `underwriter.py`: `TRAILER_KEYS` loses `doc`; `_RECIPE_COLUMNS` loses the
   seven doc derived columns; fix the `recipes` docstring examples, which
   currently show `build.recipes.query('doc')`.
7. `parser_errors.py`: delete the `"DOC"` label.
7a. The first-class classes lose the `doc` kwarg and attribute: `_aggregate.py`,
    `_portfolio.py`, `_severity.py`, `bivariate.py`, and the
    `underwriter.py:1480` passthrough that fills them. This is a **stable-tier**
    constructor signature change, which is the sharpest form of the plan's own
    timing argument: nothing can set the attribute once the clause is gone, so
    leaving it would ship a permanently empty public attribute on four frozen
    classes. Author ruled 2026-08-17 that it goes with phase E.
7b. `_program.py`: rewrite the doc-carrying docstrings. The `program` attribute
    loses its base64 paragraph (nothing is encoded any more), `format_program`'s
    `trailer` kwarg documents three clauses rather than four, `pprogram` drops
    `.doc` from its metadata sentence, and `pin_sharpen`'s Notes lose the
    base64 clause.
8. `tests/test_grammar_sync.py`: `grammar_brace_clauses() == ["hints", "note",
   "tags"]`, and delete the doc fence markdown lexing tests near line 296.
9. `tests/test_recipe.py`: trim to what survives.
10. `tests/test_doc_clause.py`: delete.
11. Regenerate the grammar reference: `grammar(add_to_doc=True)` writes
    `docs/4_agg_language_reference/ref_include.rst` directly.
12. Re-capture the spec snapshot if it moves:
    `uv run python tests/capture_spec_snapshot.py`. Read the diff deliberately;
    only the `decl-testers.agg` doc statements should change.

### Phase F [Docs-Catch-Up], rides with E

Doc-only edits do not bump on their own, so they land inside Phase E's commit as
release hygiene:

* `docs/2_aggregate_overview/underwriter.rst` lines 205 to 255, the trailer table
  and the whole `doc` write-up including the two `ipython` blocks that call
  `r.sections`, `r.problem`, `r.solution_code`, `r.check_code` and `r.run()`.
  Replace with a sentence pointing at the notes repo.
* `docs/1_Getting_Started.rst` line 150, "whether it has a full cookbook
  ``doc``".
* `cheat-sheets/Underwriter_Cheat_Sheet.tex` line 223, the `Recipe` field list.
* `src/aggregate/config.py` line 100, which names the old trio.
* `docs/2_aggregate_overview/features.rst`: this file is generated from
  `dev/task-features.md` and gated by `dev/check_features_rst.py`, so edit the
  source and regenerate rather than the `.rst`. Rows a165 and a167 record the
  cookbook generator as a feature; they become history, so the honest edit is a
  new row saying it was withdrawn before 1.0, not a deletion of the old ones.

## 6. Ripple checks against the app

Per the `T:/worktrees/CLAUDE.md` agreement that grammar changes ripple:

* **`web/src/decl-keywords.json` does not list `doc`.** Verified at a297: its
  `structural` pool is `agg`, `port`, `sev`, `distortion`, `note`. Nothing owed.
* **`parser_errors._TERMINAL_LABELS` loses `"DOC"`** (line 168). The app mirrors
  that file by hand in `decl-keywords.json`, so confirm again at execution time
  that nothing was added in between.
* **`_STRIP_CLAUSES` in `aggregate_api/examples.py` keeps its doc pattern** for
  one release, harmlessly: a regex that matches nothing costs nothing, and
  removing it is the app's call on its own schedule. Raise it in the next round
  note rather than assuming.
* **`Recipe.decl` is untouched**, which is the only part of this surface the app
  actually uses.

## 7. The accepted downside

Examples in the library and examples in the monograph can drift, because nothing
will any longer force them to agree. The author's ruling is that this is
acceptable: it is not hard to keep in sync and it is not mission critical.

Two cheap mitigations, both optional:

1. A note that wants a library program can print it rather than retype it:
   `print(build.recipe('LayerPicks').decl)`. That is `<<decl>>` without the
   templating layer, and it cannot go stale because it reads the live library.
   Recommend it as the house idiom in the notes repo.
2. Phase C's explicit tests pin the invariants regardless of what any note says,
   so drift can produce a stale explanation but never a wrong library.

## 8. Verification

At each bump:

* `uv run pytest` green.
* At Phase D and E, `python dev/done/reflow_library.py --check` reports 13 held
  back, matching `UNPARSER_EXEMPT`.
* At Phase E, `uv run pytest -m 'slow or not slow'` (the full gate) plus a
  deliberate read of the `expected_specs.json` diff.
* `rg -n "doc\{\{\{|DECL_PLACEHOLDER|<<decl>>|aggregate.cookbook" .` returns
  nothing outside `CHANGELOG.md` and `dev/done/`.
* Note that `tests/test_massive_bivariate.py::test_massive_pnl_one_sweep_ledger`
  fails at a297 for unrelated reasons (`TypeError: Index must be a MultiIndex`
  in `stats_df.xs(..., level='Label')`). Do not read it as this plan's damage.

## 9. Author rulings, 2026-08-17

All five questions were put to the author and answered before execution began.

1. **`_setup.py`: drop it**, and let each note carry its inputs in its first
   cell the way `picks-mix-exp.qmd` already does. The centralization was solving
   a problem that only existed because five pages were generated into one book.
2. **`LayerPicks` and `NeymanInnerOuter`: fold.** `picks-mix-exp.qmd` and
   `compound-frequencies.qmd` gain the material rather than the notes repo
   gaining two near duplicates. Confirmed correct on reading both: the Neyman
   section of `compound-frequencies.qmd` already builds the same inner and outer
   against `neymana` and checks it harder.
3. **Run all phases, A through F**, the grammar removal included. The DecL
   stable-tier promise makes `doc` a 2.0 decision after 1.0, so it goes now.
4. **The `doc` kwarg and attribute on the four first-class classes goes with
   phase E** (section 2 and step 7a). This was missing from the draft and is the
   plan's only stable-tier signature change.
5. **`[Recipe-Run-Clobbers-The-Trailer]` stays open**, rewritten for the general
   case. Phase D removes its only current trigger, but the underlying behavior,
   a session build silently overwriting a library entry, survives this plan and
   is worth tracking on its own terms.

## 10. What this plan does not do

* It does not delete or rename any library entry. `[Library-Round-Two]` in
  `dev/TODO.md` is separate and still open.
* It does not touch `note{}`, `tags{}` or `hints{}`.
* It does not change how `build`, `discover` or the app's Examples menu work.
* It does not address `[Aliasing-Test-Misfires-On-A-Reference-Severity]`, also
  in `dev/TODO.md`.
