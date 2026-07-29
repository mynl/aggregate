# [Recipe-Library] — notes-driven describe / test / audit

> **Done (2026-07-29).** Work completed in a157 (phase 1, the `tags{}` /
> `doc{{{}}}` trailer clauses), a158 (phase 2, the `aggregate.recipe` runtime),
> a159 (phase 3, the merged 186-entry `library.agg`), a160 (phase 4, the
> `tests/test_library_recipes.py` harness), and a161, a163, a164, a165 with
> follow-ons in a166, a167, a168 (tag namespacing, `<<decl>>` substitution,
> `[Recipe-Is-The-Entry]`, `[Cookbook-Generate]`, trailer layout,
> `aggregate.cookbook`, canonical `recipe()` lookup), *except* for the Cookbook
> implementation. Considered completed.
>
> What phase 5 leaves behind is cookbook work, and it is tracked where it
> belongs: `dev/TODO.md` `[Recipe-Library]` for the page-by-page conversion, and
> `docs/cookbook/plan.md` `[Cookbook-Pages]` for the page list. Four loose ends
> named in phase 5 above are still open: the `plan.md` rename of
> `[Cookbook-Five-Beats]`, the `_setup.py` calibration move into `library.agg`,
> the two `.md` orphans (`_03_04_modifications`, `_10_01_PK_WH`), and the
> missing `cookbook/` entry in `docs/conf.py` `exclude_patterns`. Phase 6 was
> closed by rescoping, not by execution: a `note{}` is the norm and a
> `doc{{{}}}` is for the cookbook-worthy few.

**Status at design time (2026-07-28):** out for peer review. Nothing implemented.
Two review rounds incorporated: round 1 found three blocking issues (all folded
into §1a / §1a′ / §3.4); round 2 closed five open questions and added measured
evidence for the grammar-ambiguity decision. See *Resolved questions* and
*Open questions for review* at the end for what is settled and what is not.

## Context

The cookbook (`docs/cookbook/`) is 33 fragments on a **five-beat** rhythm, of which
6 are complete, 11 have beat 1 only, and 7 are bare templates. The beats don't
survive contact with the benchmark: *Python Cookbook* (Beazley & Jones) uses
**Problem / Solution / Discussion**, and that is the frame the author wants.

Underneath sits a worse problem. There are three shipped `.agg` libraries with
overlapping purpose (`examples.agg` 36 entries, `cookbook.agg` 120,
`actuarial-severity-curves.agg` 28) and no machine-readable grouping: the
`# A. Showcase` TOC convention is normative **only in `examples.agg`** (its header
spells it out as a format spec read by the external `aggregate_api/examples.py`);
`cookbook.agg` has no TOC and no banners, and its letter prefixes are inherited
debris plus a re-started A–F block harvested from the deleted `spa_examples.agg`.
Nothing in `src/aggregate` reads those banners — `parser.py` preprocessing step 1
deletes comments outright, and the knowledge frame carries only
`(kind, name) → program, spec, source`.

And the test gap: `_test_suite.agg` / `decl-testers.agg` get ~300 parametrized
pytest cases. `cookbook.agg` gets **one** — that it loads. Nothing in it is ever
`build()`-ed, let alone checked.

**The outcome.** One library, `library.agg`, whose entries carry their own
Problem / Solution / Discussion / Check as structured DecL. The cookbook renders
from it, pytest runs it, and an audit frame reports what is documented and
checked. Adding a DecL feature then means adding one library entry that
self-describes, self-demonstrates and self-tests.

**Decisions taken (author):** new `doc{{{…}}}` clause (not an extended `note{}`);
one merged file under a *new* name `library.agg`; Problem/Solution/Discussion/
**Check**; thin `.qmd` pages calling `recipe('Name')`; `tags{…}` clause **and**
retire the letter prefixes now.

---

## Names (vetted against the existing surface — review these)

| New name | Kind | Collision check |
|---|---|---|
| `doc{{{ … }}}` | DecL clause (trailer item) | no `DOC` terminal today |
| `tags{ … }` | DecL clause (trailer item) | no `TAGS` terminal today |
| `spec['doc']`, `spec['tags']` | spec keys | free |
| `.doc` (str), `.tags` (tuple[str,…]) | attributes on `Aggregate`, `Portfolio`, `Severity`, `PnL`, `BivariateAggregate`, `Distortion` | `rg "self\.doc\|def doc\|self\.tags\|def tags"` over `src/aggregate` → **no hits** |
| `aggregate/recipe.py`, class `Recipe` | module + class | `Recipe` appears only inside bracketed labels in `_pnl_builders.py` comments |
| `Recipe.run()`, `.parts`, `.markdown()` | methods | free |
| `Underwriter.recipe(name)` → `Recipe` | method | free. Resolves by **name alone** — see the uniqueness rule below |
| `Underwriter.recipes` → DataFrame | property (the audit view) | free. **Deliberately not `audit()`** — `Tweedie.audit` already owns that word for a different concept |
| `recipe(name)` in `docs/cookbook/_setup.py` | the one cookbook verb | free |
| `library.agg` | file | free |

**Names in `library.agg` are globally unique across kinds** (author's decision).
The knowledge base is keyed `(kind, name)`, so `sev Pareto` and `agg Pareto` can
legally coexist — and do today, which is one of the 14 collisions Phase 3 has to
resolve. Requiring uniqueness makes `recipe('X')` and `build('X')` refer to the
same thing forever, with no `kind=` disambiguator anywhere. Enforced at load
(§3.2); the constraint is on `library.agg` only — `decl-testers.agg` and
`_test_suite.agg` are unaffected.

---

## Phase 1 — [Recipe-Doc-Clause]: the language change

Files: `src/aggregate/decl.lark`, `parser.py`, `decl_writer.py`, `_aggregate.py`,
`_portfolio.py`, `_severity.py`, `_pnl.py`, `bivariate.py`, `spectral.py`.

**1a. Refactor the trailer to a repetition.** Today (`decl.lark:495-499`) it is
five spelled-out alternatives, because the obvious `note hints | hints note`
form gave the *empty* trailer two parses. A star does not have that problem —
zero items is exactly one parse:

```lark
trailer: trailer_item*
trailer_item: NOTE  -> trailer_item_note
            | HINTS -> trailer_item_hints
            | TAGS  -> trailer_item_tags
            | DOC   -> trailer_item_doc
```

The transformer's `trailer` method folds the list into a dict, raising a clear
parse error on a repeated item. All 15 call sites of `trailer` in the grammar are
untouched.

**`tags` and `doc` must be *conditional* spec keys.** `note` and `hints` are
written unconditionally today and every snapshot entry carries them; the snapshot
comparison is exact-key-set (`tests/test_decl_parser.py:82`,
`assert set(actual) == set(expected)`). Adding two more unconditional keys fails
**all 163 cases** with `Extra={'tags','doc'}`. So: emit `spec['tags']` /
`spec['doc']` only when non-empty, have the six hosts read them with
`spec.get(...)`, and have `_render_trailer` do the same. `note`/`hints` keep
their unconditional behavior — this is a pure addition.

**1a′. The ambiguity risk is real, and it is not observable by default.** The
parser is built with Lark's default `ambiguity='resolve'` (`parser.py:2115-2121`,
no `ambiguity=` argument), so it silently picks a parse and never warns — "check
for an ambiguity warning" would be a vacuous test. And a genuine ambiguity
already exists in the bivariate rules: `copula_clause` is nullable
(`decl.lark:253`) and `bv_body` ends in a `bv_item` = `agg_out | pnl_out`
(`decl.lark:240-246`), **each of which ends in its own `trailer`**. In
`bv_out_copula_nofreq` (`decl.lark:203`) a trailing `note{...}` can therefore
attach to the inner agg *or* to the bivariate, and `resolve` picks silently
today. The star rewrite does not create this — but it reshuffles the parse forest,
so `resolve` may flip the choice. The spec snapshot gives **no** cover: its 163
entries are 151 `agg` / 10 `port` / 2 `sev`, and `_test_suite.agg` has zero
`bivariate`/`clash`/`netceded` lines.

**Measured, not assumed.** A sweep of all 606 shipped DecL statements
(`_test_suite`, `_test_suite2`, `decl-testers`, `examples`, `cookbook`,
`actuarial-severity-curves`) under `ambiguity='explicit'` gives:

| Case | `_ambig` nodes |
|---|---|
| `port P note{a} agg A … note{b}` | **0** |
| `port P agg A … note{trailing}` | **0** |
| `bivariate B … copula gumbel 0.4 poisson note{t}` | **0** |
| `bivariate B … agg X … agg Y … note{t}` (no copula, no freq) | **1 node, 2 parses** |
| Whole 606-line shipped corpus | **1** — and it is *not* a trailer case |

So the trailer ambiguity is real but **narrow**: it needs a `bivariate` with no
copula clause *and* no outer frequency *and* a trailing trailer. Anything that
separates the last component from the trailer (a `copula …`, a `poisson`) kills
it. **No shipped program triggers it.** And `resolve` currently picks the
*semantically desirable* parse — the note lands on the bivariate, not on the last
component (verified: `.note == 'trailing'`, both units `''`).

The one real corpus ambiguity is unrelated and benign:
`agg UM.Scaled 10 claims ssev -3 * lognorm 2 cv 0.5 poisson` parses as either
`sev1_scaled(-3, …)` or `sev1_negate(sev1_scaled(3, …))` — algebraically the same
distribution. It goes into the baseline as accepted.

**Decision: baseline and guard, do not re-engineer the grammar** (see
*Resolved question 1* below for why). Procedure, in this order:

1. **Before touching the grammar**, add bivariate / clash / netceded / port
   programs carrying a trailing `note{...}` to `decl-testers.agg`, plus tests
   asserting *where the note landed* (`b.note` vs `b.units[-1].note`). Round-trip
   alone is not enough — a flipped attachment can still be a fixed point.
2. Add `tests/test_grammar_ambiguity.py`: a module-scoped second `Lark` built with
   `ambiguity='explicit'`, swept over every shipped `.agg`, asserting the set of
   ambiguous statements equals a small pinned allow-list (today: the one `ssev`
   case). **Fail on any new one.** This is cheap, fast, and is a durable asset
   beyond this plan — the grammar has never had such a check.
3. Only then do the star rewrite from 1a, and re-run 1 and 2 unchanged.
4. **Specify the behavior in the grammar comment and the DecL reference**: a
   trailing trailer on a `bivariate` with neither a copula clause nor an outer
   frequency binds to the *bivariate*; to annotate the final component instead,
   place the copula or frequency clause after it. Turning the accident into
   documented behavior is the actual fix.

If the star does widen the ambiguity, the fallback is a **single `TRAILER`
terminal** matching a run of items, decomposed in the transformer — strictly less
ambiguous than today (one token, not two). The spelled-out form is not a fallback:
four order-free optional items is 65 alternatives.

**Portfolio is a different mechanism and is not at risk.** `port_out` places its
trailer *before* `agg_list` (`decl.lark:51`, `PORT name as_label trailer
agg_list`), so a port note is positional and unambiguous — measured 0 `_ambig`
above, with or without a trailing agg note. This is what `examples.agg:29-31`
means by "a trailing note attaches to the last `agg`, not the portfolio": the port
slot has already closed. Verified end to end:

```python
p = build('''
port TESTPORT note{at the port level}
    agg RuRe 1 year sev gamma 2 wait 0.25 * uniform note{on the agg}
''')
assert p.note == 'at the port level'          # passes
assert p.agg_list[0].note == 'on the agg'     # passes
```

Because the port trailer is positional, `tags{}` and `doc{{{}}}` land on a
portfolio exactly the same way — on the `port` header line, before the units.
Worth a line in the `library.agg` header, since it is the one placement rule an
author has to remember.

**1b. New terminals** (priority 3, beside `NOTE`/`HINTS` at `decl.lark:687-694`):

```lark
TAGS.3: /tags\{[^}]*\}/
DOC.3:  /doc\{\{\{[A-Za-z0-9_=-]*\}\}\}/
```

`DOC` matches only an **encoded placeholder** — see 1c.

**1c. `UnderwritingLexer.preprocess` gains a step 0** (`parser.py:158`, before the
comment strip). Scan `doc\{\{\{[ \t]*\n(.*?)\n[ \t]*\}\}\}` with `re.S` — **the
closing fence must be alone on its line** — and replace each body with its
**URL-safe base64**. Requiring the fence on its own line matters more here than
the `}`-in-`note{}` precedent suggests: notes are prose, but doc bodies carry
Python, where `}}}` is an ordinary line ending (`{'a': {'b': {'c': 1}}}`). With
the line anchor, inline `}}}` in code is harmless. This is the whole trick, and it
is why the design works:

- the urlsafe alphabet is `A-Za-z0-9-_=` — no `#`, no `//`, no `}`, no `[`/`]`,
  no `;`, no whitespace — so the encoded token passes through preprocessing
  steps 1–6 **byte-for-byte**;
- `#` markdown headings survive because extraction happens *before* the
  comment stripper (`parser.py:165`), which is the exact reason the obvious
  markdown-in-`note{}` design cannot work today;
- blank lines survive because extraction happens *before* the paragraph split
  (`parser.py:212`);
- `preprocess` keeps its `list[str]` signature — no side table.

The `DOC` terminal callback base64-decodes and `.strip('\n')`s. The `TAGS`
callback splits on commas/whitespace into a tuple of slugs.

The one forbidden sequence inside a doc body is a line consisting only of `}}}`
(optionally indented). Document it in the grammar comment, as the `}`-in-`note{}`
rule is documented today. The writer emits the fence the same way, so the form
round-trips.

**1d. Give `distortion` a trailer.** `distortion_out` (`decl.lark:40-42`) has none,
so `dist` entries cannot carry a note *or* a doc — which blocks cookbook §1
(Distortions), whose entries are `dist K.PH` / `K.Dual` / `K.Min`. Append
`trailer` to the three alternatives and thread it through
`distortion_out_params` / `_combo` / `_combo_wtd` (`parser.py:321,355,365`).
`decl_writer._render_distortion` already accepts and ignores a `trailer` arg
(`decl_writer.py:966-976`) — wire it up. Remove the "`note{}` is NOT legal on
`dist` lines" caveat from the library headers (it appears in all three, e.g.
`examples.agg:32`). `Distortion` then carries `.note`, `.tags` and `.doc` like
every other host, so distortions are first-class in describe/test/audit — which
is the point, and cookbook §1 depends on it.

**1e. Writer.** `_render_trailer` (`decl_writer.py:589`) emits, in order,
`note{…} tags{…} hints{…}` then `doc{{{\n…\n}}}` last (bulkiest, and a
`_Block` line of its own). `trailer=False` continues to drop **all** of them —
one flag, one construct, unchanged contract.

**1f. Round-trip.** `format_program(build(x))` must re-parse to the same spec:
the writer emits a raw fenced body, preprocess re-encodes it. Add doc/tags
fixtures to `decl-testers.agg` (house rule) and to `tests/test_decl_unparser.py`'s
corpus, plus `tests/test_hints.py`-style unit tests for the `#`/blank-line/`}`
cases that motivated the design.

**1g. Attributes.** `.doc` and `.tags` on the six hosts, alongside `.note`.
`Underwriter.discover`'s program-cleaning regexes (`underwriter.py:2068-2075`)
must also strip `tags{…}` and `doc{{{…}}}`.

**Bump + commit at the end of Phase 1.**

---

## Phase 2 — [Recipe-Runtime]: notes → code

New module `src/aggregate/recipe.py`.

```
Recipe(name, kind, note, tags, problem, solution, discussion, check,
       solution_code, check_code, extra)
```

- **`parse_doc(text) -> Recipe`** — splits the markdown on
  `^##\s+(Problem|Solution|Discussion|Check)\b` (case-insensitive); anything else
  is kept in `extra` and rendered verbatim. Extracts ` ```python ` fences from
  Solution and Check.
- **`Recipe.run(ns=None) -> dict`** — execs Solution then Check **in one
  namespace**, seeded with `build`, `qd`, `np`, `pd`. Solution code is written
  **self-contained** (`a = build('LimitProfile')` …) so a reader can copy-paste
  it; Check continues in the same namespace and is pure `assert`s. Raises on
  failure. Carries the trust warning of *Resolved question 2*.
- **`Recipe.markdown()`** — the prose, reassembled.
- **`Underwriter.recipe(name)`** → `Recipe`; **`Underwriter.recipes`** → the audit
  DataFrame (§4).

**Rendering into Quarto.** `docs/cookbook/_setup.py` gains the one cookbook verb
`recipe(name)`, which emits through **`IPython.display.display()`** —
`display(Markdown(...))` for prose, then the Solution code echoed as a fenced
block, then its rich outputs (`greater_tables.GT`, matplotlib figures), then
Discussion, then Check in a collapsed callout. Deliberately *not*
`#| output: asis`: asis text and `display()` output do not interleave reliably
within one cell, and recipes need tables and plots inline.

**Bump + commit at the end of Phase 2** (Phases 1+2 can share one bump if they
land together — one bump, one commit, never batched afterwards).

---

## Phase 3 — [Library-Consolidation]: three files → `library.agg`

The expensive phase; it is curation, not a script.

1. **Create `src/aggregate/agg/library.agg`** from `examples.agg` +
   `cookbook.agg` + `actuarial-severity-curves.agg` (~184 entries).
2. **Retire the letter prefixes.** Verified: only **one** reference to a shipped
   dotted name exists outside the `.agg` files —
   `docs/cookbook/_04_04_limit_profile.qmd:23` (`build('E.LimitProfile')`). So the
   rename is nearly free *externally*. Internally, mechanical prefix-stripping
   collides on 14 names — `Basic`, `Binomial`, `Discrete`, `PmL`, `Poisson`,
   `Re01`, `Signed`, `ThreeDice`, `Tweedie`, `Windowed`, `Bodoff`, `Book`,
   `Lognorm`, `Pareto` — several of which are `cookbook.agg`'s **own** internal
   duplicates (`J.Re01` is defined three times, `cookbook.agg:103-105`). Each
   collision is resolved by hand: merge, or split into distinct recipes with
   distinct descriptive names. **Names must end up unique across kinds, not just
   within one** — `sev Pareto` / `agg Pareto` is a collision under this rule even
   though `(kind, name)` keys it fine today. Add a load-time check that raises on
   a duplicate bare name so the invariant can't silently rot; scope it to
   `library.agg`.
3. **Tag vocabulary**, documented in the file header (which replaces the old
   `FORMAT` block). **Namespaced, and it never restates the object's type** —
   revised at a161 after review; the first cut used bare words and 103 of 186
   entries ended up carrying a tag that merely repeated their own kind:
   - `topic:X` — what the recipe is *about*: `severity`, `frequency`,
     `aggregate`, `reinsurance`, `pnl`, `portfolio`, `distortion`,
     `bivariate`, `bounds`, `ruin`, `numerics`. Independent of type, so an
     `agg` entry demonstrating a severity form is `topic:severity` and that is
     not a restatement.
   - `role:X` — `hero`, `intro`, `reference`, `paper`
   - `check:X` — `reconciliation`, `scaling-sweep`, `independent-oracle`,
     `limiting-case`, `round-trip`, `cross-object` (the beat-4 catalog,
     re-homed)
   - `slow` — the one deliberately bare tag; it names a pytest marker, not a
     property of the subject.

   **Type is `discover(kind=...)`, not a tag.** Enforced by
   `tests/test_agg_libraries.py`. Explicitly rejected: a "look-through" tag
   synthesising the kind, and auto-appending kind to tags — both recreate the
   two-ways-to-say-it problem.
4. **Wiring**: `config.py:105` → `databases = ('library',)` (and its docstring at
   `:92-105`); `tests/test_agg_libraries.py:22` → `['library']`. Delete the three
   old files.
   **Stated decision, not a side effect:** `databases` is `('examples',)` today,
   so `cookbook.agg` and `actuarial-severity-curves.agg` are **not** loaded by
   default. The merge therefore grows the out-of-the-box knowledge base from 36
   to ~184 entries. Author has confirmed this is intended; it gets its own
   CHANGELOG line.
5. **Coordination flag, not a task here:** `aggregate_api/examples.py` lives in a
   *different repo* and re-lexes the old `FORMAT` header to build the SPA
   dropdown. Per the author it will be rewritten against whatever this plan
   adopts; note it in the CHANGELOG and the run summary. The `/v1/examples` URL
   need not change.

**Bump + commit.**

---

## Phase 4 — [Recipe-Tests-Audit]

- **`tests/test_library_recipes.py`** — parametrize over every KB entry with a
  `doc`; `Recipe.run()` it headless (`matplotlib.use('Agg')`, `plt.close('all')`
  teardown). `tags{slow}` maps to `@pytest.mark.slow`, per the existing
  fast-by-default policy. This converts ~184 library entries from *one* load
  assertion into executed, asserted cases — the single biggest coverage win here.
- **`Underwriter.recipes`** — DataFrame indexed `(kind, name)` with columns
  `tags`, `has_note`, `has_doc`, `problem`/`solution`/`discussion`/`check`
  (bool), `n_asserts`, `source`. That is the audit: what is documented, what is
  demonstrated, what is checked.
- **Coverage cross-check** against `dev/FEATURES.csv` (the maintained capability
  matrix — regenerate via `dev/regen_features.py`): every DecL feature should
  name at least one tagged recipe.
- **`tests/test_grammar_ambiguity.py`** (built in Phase 1, §1a′ step 2) extends to
  cover `library.agg` once it exists. The allow-list should still be the single
  `ssev` case; anything else the merge introduces is a finding.
- **`Recipe.run()` carries the trust warning** in its docstring and in the
  cookbook's "How to read": running a recipe executes the code in the `.agg`
  file, so treat a third-party library like third-party Python.

**Bump + commit.**

---

## Phase 5 — [Cookbook-Problem-Solution-Discussion]

- **Rewrite `docs/cookbook/plan.md`**: `[Cookbook-Five-Beats]` →
  `[Cookbook-Problem-Solution-Discussion]`, citing Beazley & Jones as the
  benchmark. `[Cookbook-Beat-Four-Catalog]` is **kept**, re-homed as the
  `check:*` tag vocabulary. `[Cookbook-Numbering]`, the heading/TOC rules and the
  semantic anchors are good — keep verbatim.
- **Rewrite `cookbook.qmd`'s "How to read"** (lines 10-25) to the new frame; fix
  the stray `ads` (line 37) and the stray `C` in `_01_distortions.qmd:11`.
- **`_setup.py`** — the calibrated snippet library (`HOUSE_PREM`, `HOUSE_SEV`,
  `HOUSE_FREQ`, `house()`, `OCC_LAYER`, `AGG_LAYER`, `SWING`, `REINST`,
  `pnl()`) **moves into `library.agg`** as named entries. This resolves the live
  tension flagged by `[Cookbook-Decisions]` #3 ("`_setup.py` is the single
  source of truth for calibration") versus `_04_04_limit_profile.qmd` already
  reaching for a library entry. After the move the **library** is the single
  source of truth and `_setup.py` is just display helpers (`qd` / `pp` /
  `show`).
- **Page shape.** ~~A thin page calling a runtime `recipe('X')` verb~~ —
  **superseded**, see `[Cookbook-Generate]` in `dev/plan-recipes.md`. The
  runtime verb `exec`'d the code itself inside one cell, which meant no
  per-block `#|` options and a figure flushed by the inline backend at *cell
  end*, i.e. after the prose rather than inside the Solution. Pages are instead
  **generated** from `library.agg` into native ` ```{python} ` Quarto cells by
  `aggregate.cookbook`; `_setup.recipe()` is retired. `Recipe.run()` stays
  — it is the pytest harness, and keeping both consumers on the same doc is what
  makes "the page and the test run the same program" true by construction.

  Essay pages (`_06_03_ir_modeling.qmd` — the signed-severity trap — and
  `_02_02_scipy_continuous.qmd`) keep their hand-written prose and cite recipes
  alongside it. `_06_03`'s `[Check-Reparameterization]` is a seventh archetype
  not in the catalog; per the author it simply folds into `check:round-trip`.
- **Fix the two orphans**: `_03_04_modifications.md` and `_10_01_PK_WH.md` are
  `.md`, are absent from `cookbook.qmd`'s include list, and use non-standard
  vocabulary. Convert to `.qmd`, wire in (with a `_10_00` landing for §10), and
  normalize.
- **`docs/conf.py:142`** — `exclude_patterns` does not exclude `cookbook/`, so
  those `.md` files are picked up as orphan documents by a Sphinx build. Add the
  exclude.

**Bump + commit.**

---

## Phase 6 — [Recipe-Population]: fill the empty pages

**Scope corrected (a164).** This was written as though all 186 entries awaited a
doc. They do not: a `note{}` is the norm and is all `discover` and the object
dropdown need, and a `doc{{{}}}` is for the entries the cookbook actually
teaches from. `build.recipes.query('not doc')` is a **directory**, not a
worklist. What follows is about the *cookbook pages*, and the recipes each one
needs.

Not a guess-list — the first task is to **produce the mapping** from
`build.recipes` / `build.discover()` once tags exist: page → recipe name(s), with
gaps visible. Verified starting points:

| Page | Recipe(s) | Status |
|---|---|---|
| `_04_04_limit_profile` | `LimitProfile` (was `E.LimitProfile`, `examples.agg:217-222`) | exists, needs a doc |
| `_01_distortions` | `PH`, `Dual`, `Min` (was `K.*`, `examples.agg:386-392`) | needs Phase 1d first |
| `_10_01_PK_WH` | the `AD.Ruin*` family in `decl-testers.agg` | promote into `library.agg` |

For the remaining 11 partial + 7 template pages: harvest candidates from the
merged library first, then from `decl-testers.agg` and `tests/`, promoting each
harvested program into `library.agg` **with a full doc**. Work page-by-page with
author reaction — the existing `[Cookbook-Build-Order]` step 3 rhythm.

**Bump + commit per batch of pages.**

---

## Comments in `.agg` files are no longer load-bearing — confirmed

After this plan, **all machine-readable metadata comes from the four trailer
clauses**; no tool reads a comment. Concretely:

- `tags{}` replaces the `# A. Showcase` TOC and the `# ---` letter banners as the
  grouping mechanism. Nothing in `src/aggregate` ever read them anyway —
  `parser.py:158-162` deletes comments before the lexer runs; the only consumer
  was the external `aggregate_api/examples.py`, re-lexing the raw text, and that
  is being rewritten against the new scheme.
- `note{}` is the short description, `doc{{{}}}` the recipe, `hints{}` the build
  settings. `library.agg`'s header replaces the old `FORMAT` block and documents
  the tag vocabulary.

Comments keep exactly two non-metadata jobs, both for humans:

1. **The file header** — purpose, format, tag vocabulary. Nowhere else to put it.
2. **Commenting out a clause mid-statement** — a genuine language feature:
   preprocessing step 1 makes a full-line comment *transparent*, so a
   commented-out reinsurance line inside a multi-line `agg` folds away without
   splitting the statement. No clause can replace that.

Section banners may survive as plain visual dividers in a 184-entry file, but
they carry no meaning to any tool. Author's call during the Phase 3 curation.

## Resolved questions

1. **Bivariate trailer ambiguity — baseline and guard, do not re-engineer.**
   The evidence in §1a′ decides it: zero of 606 shipped statements trigger it;
   `resolve` already picks the desirable parse (note → bivariate); and the
   trigger needs three coincident conditions. The "proper fix" — making the final
   `bv_item` use a trailer-less variant — is not cheap (`bv_body` is left-recursive
   at `decl.lark:240-241`, so "final" is not syntactically distinguished) and it
   would **remove the ability to annotate the second component at all**. Paying a
   capability to close a hole nothing falls into is the wrong trade. Instead:
   pin it with `tests/test_grammar_ambiguity.py`, and *document the binding rule*
   so it is specified behavior rather than an accident.
2. **`Recipe.run()` trust — warn, don't gate** (author). Ship it with a clear
   warning in the docstring and the cookbook: running a recipe from a `.agg` file
   executes the code in it, so treat a third-party `.agg` like a third-party
   Python file. No signing, no opt-in kwarg, no source gating in v1.0. A
   doc-signing scheme (trailing `<!-- hash: … -->` over the doc body, salted from
   a private env var) goes to `dev/TODO.md` as a **post-v1.0 idea**, explicitly
   judged not worth the machinery now. Separately, `aggregate_api` will not
   execute any user-supplied code — the same posture as its existing `hints{}`
   capping of `log2`.
3. **`distortion` gets a trailer** (author): no reason for the restriction, and
   the describe/test/audit process should cover `Distortion` like everything
   else. §1d proceeds; `.doc`/`.tags`/`.note` all land on `Distortion`.
4. **SPA** (author): `aggregate_api` will be rewritten against whatever this plan
   adopts. §3.5 stays a coordination note, not a constraint on the design.
5. **`[Check-Reparameterization]`** (author): ignore — no seventh archetype. The
   `_06_03_ir_modeling.qmd` tag folds into `check:round-trip`.

## Open questions for review

*Round 1 found three blocking issues — conditional spec keys, the vacuous
ambiguity check plus the real bivariate trailer ambiguity, and the unstated
36 → 184 default-KB change — all folded into §1a / §1a′ / §3.4. Round 2 closed the
five questions above. Nothing is currently blocking; the items below are opinions
worth having, not decisions to make.*

- The `ssev -3 * lognorm` ambiguity (§1a′) is pre-existing, benign, and unrelated
  to this work, but it is now *visible*. Worth its own look someday — flag only.
- Whether `tests/test_grammar_ambiguity.py` should also gate the *user* library
  directory (`~/.aggregate/*.agg`), or only shipped files. Shipped-only is the
  proposed default.

## Verification

1. **Grammar, first.** `uv run pytest tests/test_decl_parser.py
   tests/test_decl_unparser.py tests/test_hints.py -n0`. Three independent things
   must hold, and the spec snapshot covers only one:
   - *Keys*: `expected_specs.json` is keyed on preprocessed line **text**;
     `_test_suite.agg` gains no docs, so the preprocessed text must be
     byte-identical — if it moves, step 0 leaked.
   - *Values*: the spec key set must be unchanged, which is what conditional
     `tags`/`doc` buys (§1a). A run with unconditional keys fails all 163 cases;
     that is the fast confirmation the conditional path is actually taken.
   - *Attachment*: the pre-change trailer-attachment baseline and the
     `ambiguity='explicit'` sweep from §1a′, which the snapshot cannot see at all.
     Baseline to reproduce before any edit: **606 statements, exactly 1 ambiguous**
     (`agg UM.Scaled … ssev -3 * lognorm 2 cv 0.5 poisson`). After the rewrite the
     count must be unchanged, and the bivariate trailing note must still land on
     the bivariate (`b.note == 'trailing'`, both units `''`).
2. **Round-trip** — `format_program(build(prog))` re-parses to the same spec for a
   doc carrying `#` headings, blank lines, ` ``` ` fences, `}` and inline `}}}`
   inside code, and a `;`. These are the cases that break today.
3. **Runtime** — `uv run pytest tests/test_library_recipes.py`; then the full gate
   `uv run pytest -m 'slow or not slow'`.
4. **Smoke** — `build('LimitProfile')`, `build.recipe('LimitProfile').run()`,
   `build.recipes` (audit frame), `build.discover(tags='hero')`.
5. **Render** — `cd docs/cookbook && uv run quarto render`, then open
   `_site/cookbook.html`. Note this has **never been rendered in this worktree**
   (no `_site/`, no `_freeze/`), so expect first-render friction independent of
   these changes. Do **not** run the Sphinx build in the loop (house rule).

## Housekeeping per house rules

Each phase: version bump in `pyproject.toml` + its own `CHANGELOG.md` section +
`dev/TODO.md` update, committed together as one coherent unit with a one-line
subject `[Recipe-Library] aNNN: …`. `dev/FEATURES.csv` regenerated when the
public surface moves (Phases 1, 2, 4). This plan lives at `dev/plan-meta-data.md`
and moves to `dev/done/` at the end.
Closes the `plan-for-v1.md:82` item ("Extend `examples.agg` notes with tags,
keywords, and purpose") and `dev/TODO.md:157-158`.

**New `dev/TODO.md` entry, post-v1.0:** `[Recipe-Doc-Signing]` — optionally
authenticate a `doc{{{}}}` body with a trailing `<!-- hash: … -->` computed over
the body and salted from a private environment variable, so `Recipe.run()` can
refuse an unsigned or tampered recipe. Explicitly judged not worth the machinery
for v1.0 (*Resolved questions* 2); recorded so the idea is not lost.
