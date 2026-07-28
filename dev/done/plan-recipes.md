# [Recipe-Is-The-Entry] + [Cookbook-Generate]

**Status: DONE 2026-07-28.** Part A + Part C shipped as **a164**, Part B as
**a165**. Follows `dev/plan-meta-data.md` phases 1–4 (**a157–a163**), whose
phases 5–6 remain open — see `[Recipe-Library]` in `dev/TODO.md`.

Executed as written, with three notes:

- **A1 name collision, resolved.** `Recipe.program` (a158) meant the *doc-free*
  rendering while `ParsedProgram.program` meant the *verbatim source line*.
  Merging the classes forced a choice: `program` keeps the verbatim meaning
  (what every other caller assumes) and the doc-free canonical rendering became
  the new `Recipe.decl` — the name `<<decl>>` already implied. `parse_doc`'s
  `program=` kwarg became `decl=`.
- **A3 frame width.** Resolved by column order — identity and audit flags
  first, `program` / `spec` last; `build.recipes.iloc[:, :9]` is the readable
  slice, and the docs use it.
- **B: additive, not a rewrite.** Generated fragments are included at the end of
  their section under their own `#sec-recipe-…` anchors rather than replacing
  the hand-written five-beat stubs. Retiring those is phase 5, page by page with
  author reaction — doing it here would have silently discarded the author's
  in-page TODO notes. `_setup.recipe()` turned out to be dead code: no page ever
  called it.

---

## Where things stand (what already shipped)

| version | label | what landed |
|---|---|---|
| a157 | [Recipe-Library] | `tags{}` + `doc{{{}}}` trailer clauses; `distortion` gained a trailer; `tests/test_grammar_ambiguity.py` + `tests/test_trailer_attachment.py` |
| a158 | [Recipe-Library] | `aggregate/recipe.py`: `Recipe` / `parse_doc`; `Underwriter.recipe()` + `.recipes`; cookbook `recipe()` verb |
| a159 | [Recipe-Library] | three libraries → one `library.agg`, 186 entries, unique names, tags everywhere; `discover(tags=)` |
| a160 | [Recipe-Library] | `tests/test_library_recipes.py` runs every documented entry's Solution + Check; first four recipes |
| a161 | [Tag-Namespace] | tags namespaced `topic:` / `role:` / `check:` (+ bare `slow`); type is `discover(kind=)`, never a tag |
| a162 | [Test-Loop] | `pytest-testmon`; `--dist loadgroup` + `xdist_group('bivariate')`; three-tier test guidance |
| a163 | [Recipe-Library] | `<<decl>>` expands to the entry's own doc-free declaration; `format_program(trailer=)` takes an item collection |

**Conventions established, do not relitigate:**

- A recipe's Solution writes ` ```python ` + `a = build('''<<decl>>''')`. It
  **never retypes the program**. `<<decl>>` expands to the entry's declaration
  rendered *without* its doc (note/tags/hints kept — hints change how the object
  builds). Dropping the doc is the recursion guard.
- Fences are plain ` ```python `, **not** ` ```{python} `, *inside a doc body* —
  the doc is data, and `Recipe.run` executes it. (Part B below emits
  ` ```{python} ` into generated `.qmd`, which is a different thing: that is
  Quarto source.)
- Tags never restate an entry's kind. Enforced by `tests/test_agg_libraries.py`.
- **A `doc{{{}}}` is for cookbook-worthy entries only.** Most of the 186 carry a
  `note{}` and nothing more — enough for the SPA dropdown and `discover`.

---

## Context — the two seams this closes

**1. Two objects describe one entry.** `Underwriter` keeps a *knowledge base* of
`ParsedProgram(kind, name, spec, program, object, source)`
(`underwriter.py:325`), while `recipe.py` keeps a `Recipe` holding the same
entry's parsed doc. Two registries, two names, one thing. The overlap exists
only because `Recipe` was bolted on in a158 instead of replacing anything.
`knowledge` was a London-cabbie joke; `recipes` is shorter and better, and v1.0
is the moment to rip the old name out — **no alias, no deprecation shim**
(author's explicit call).

**2. Cookbook rendering reimplements a slice of Quarto.** `_setup.recipe('X')`
emits one cell of `display(Markdown(...))` and `exec()`s the code itself. Two
consequences: per-block cell options (`#| echo`, `#| fig-cap`, `#| warning`)
cannot exist, and a matplotlib figure that is merely *created* is flushed by the
inline backend at **cell end** — so a plotting recipe renders its figure after
the prose rather than inside the Solution. (Expected from inline-backend
behaviour; **not** measured — the cookbook has never been rendered in this
worktree.) Generating real ` ```{python} ` cells hands all of it back to Quarto.

**Measured blast radius for Part A:** `_knowledge` 23 src / 32 tests;
`ParsedProgram` 22 src / 2 tests / 3 docs; `.knowledge` 4 src / 8 tests / 7
docs; `add_entry` 2 src / 2 tests. ≈100 references, mechanical.

---

## Part A — [Recipe-Is-The-Entry]

Files: `src/aggregate/recipe.py`, `src/aggregate/underwriter.py` (the bulk),
`docs/1_Getting_Started.rst`, `docs/2_user_guides/2_x_10mins.rst`,
`docs/2_user_guides/dm-claude.qmd`, `docs/3_reference/3_x_Underwriter.rst`,
and ~44 test references.

**A1. `Recipe` absorbs `ParsedProgram`.** One class, living in `recipe.py`,
carrying identity and content together:

```
Recipe(kind, name, spec, program, source, object=None,
       note, tags, hints, doc, ...)
```

plus the doc surface already present — `problem` / `solution` / `discussion` /
`check`, `solution_code`, `check_code`, `n_asserts`, `sections`, `run()`,
`markdown()`.

- **Parse the doc lazily**, on first access to a section. Only a handful of the
  186 entries have a doc, and `load()` is on `build`'s import path.
- Keep `object` populated by the factory *after* construction — that is what
  `build()` returns from. Same for `source` (`Path` of the `.agg` file, or the
  `'session'` sentinel; it backs `to_agg`'s source filter).
- `ParsedProgram` is deleted outright.

**A2. Rename the registry.**

| old | new |
|---|---|
| `Underwriter._knowledge` (dict) | `_recipes` |
| `Underwriter.knowledge` (DataFrame property) | `recipes` |
| `Underwriter._knowledge_frame()` | `_recipes_frame()` |
| `Underwriter.add_entry(...)` | `add_recipe(...)` |
| `ParsedProgram` | `Recipe` |

`Underwriter._check_library_names_unique` and `LIBRARY_FILENAME` (a159) refer to
`_knowledge`; `Underwriter.recipe()` / `.recipes` (a158) already exist and
become the merged surface rather than a second one.

**A3. One frame.** `build.recipes` merges today's two: entry columns
(`program`, `spec`, `source`) plus the doc columns from a158 (`tags`, `note`,
`doc`, `problem`/`solution`/`discussion`/`check`, `n_asserts`). Indexed
`(kind, name)` — the "add kind/type to replicate the (kind, name) match" the
author asked for is already the index; nothing new needed.
`Underwriter.recipe(name)` keeps returning **one** `Recipe`, resolving by name
alone (`kind=` disambiguates; library names are unique across kinds).

**Watch the frame width.** A dozen columns including an object-valued `spec` and
a possibly-long `doc` can be unusable at the terminal — `build.knowledge` was
already awkward to print. Put identity + tags + flags first and the heavy
columns last; check both `qd(build.recipes)` and the plain repr.

**A4. Docs.** Four files describe "the knowledge base" in prose *and* call
`build.knowledge` in **executed** code blocks:

- `docs/1_Getting_Started.rst:154-160`
- `docs/2_user_guides/2_x_10mins.rst:188-193, 246`
- `docs/2_user_guides/dm-claude.qmd:227-231, 296`
- `docs/3_reference/3_x_Underwriter.rst:5` — leads with "It owns the **knowledge
  base**", which sets the mental model; rewrite the sentence, don't patch it.

**Bump + commit (a164).**

---

## Part B — [Cookbook-Generate]

Files: new `dev/generate_cookbook.py`, `docs/cookbook/_setup.py`, the
`docs/cookbook/_SS_MM_*.qmd` fragments, `docs/cookbook/plan.md`,
`docs/cookbook/cookbook.qmd`.

**B1. The generator.** Reads `library.agg`; for every entry with a doc, emits a
`.qmd` fragment with **native Quarto cells**:

```markdown
<!-- generated from library.agg by dev/generate_cookbook.py -- do not hand-edit -->

## Limit profile {#sec-limit-profile}

<Problem prose>

```{python}
a = build('''<expanded decl>''')
qd(a.summary_df)
```

<Discussion prose>

::: {.callout-note collapse="true" title="The check"}
```{python}
assert abs(a.actual_m - 2450.0) < 1e-9
```
:::
```

`<<decl>>` is expanded at generation time — reuse `Underwriter.recipe`, whose
`Recipe.program` is already the doc-free declaration (a163). Quarto then
executes natively: figures land where created, `#|` options work per block,
errors report as Quarto cell errors, `freeze` caches at cell granularity.

**B2. Generated files are build artifacts** — header comment says so, generator
is idempotent. **Commit them**: diffable, and the docs build should not require
running a generator first.

**B3. Retire the runtime path.** `_setup.recipe()` goes — its exec/`display`
machinery is exactly what this replaces. `_setup.py` keeps `qd` / `pp` / `show`
and its imports.

**`Recipe.run()` STAYS.** It is the pytest harness
(`tests/test_library_recipes.py`) — a different consumer with different needs,
and keeping it is what guarantees the cookbook and the test suite execute the
*same* code from the *same* source. That property is the point of the whole
mechanism; do not remove it along with the renderer.

**B4. Page mapping.** `topic:*` tags already mirror the cookbook sections, so
the generator groups by topic tag. The `_SS_MM` numbering, heading levels and
semantic anchors from `[Cookbook-Numbering]` (`docs/cookbook/plan.md`) still
apply. Section landing pages (`_SS_00`) stay hand-written, as do the essay pages
(`_06_03_ir_modeling.qmd`, `_02_02_scipy_continuous.qmd`), which cite recipes
alongside their own prose.

**Bump + commit (a165).**

---

## Part C — correct the "182 to go" framing (ships with Part A)

`library.agg`'s header STATUS line and the `dev/TODO.md` `[Recipe-Library]`
entry both read as though all 186 entries await a doc. They do not: a `note{}`
is the norm and is sufficient for the SPA and `discover`; a `doc{{{}}}` is for
the entries the cookbook teaches. `build.recipes.query('not doc')` is a
**directory**, not a worklist. Reword both.

Also update `dev/plan-meta-data.md`: its phase 5 still specifies the retired
thin-page `recipe()` approach and must point at the generator instead.

---

## Verification

1. **Nothing lost in the rename** — capture `(kind, name, spec)` for all 186
   entries before Part A and compare after. The equivalent check on the a159
   merge caught two real bugs; do not skip it.
2. **Test loop** — `uv run pytest -n0 --dist no --testmon-forceselect` while
   editing (plain `--testmon` silently does nothing here: `addopts` carries
   `-m 'not slow'`, which forces no-select). Full `pytest` before each commit,
   `-m 'slow or not slow'` before each bump.
3. **`build.recipes` reads well** — eyeball `qd(build.recipes.head())` and the
   plain repr.
4. **Docs snippets still execute** — the four A4 files contain executed blocks.
   Run the snippets directly; do **not** run the full Sphinx build in the loop
   (house rule).
5. **Cookbook renders** — `cd docs/cookbook && uv run quarto render`, open
   `_site/cookbook.html`, and confirm a plotting recipe puts its figure *inside*
   the Solution. This is the whole point of Part B. Never rendered in this
   worktree, so expect first-render friction unrelated to these changes.
6. `uv run python dev/regen_features.py` clean after Part A (public surface
   moves).

## Housekeeping

Two bumps, two commits, house one-line subjects; a `CHANGELOG.md` section each;
`dev/TODO.md` and `dev/plan-meta-data.md` updated. Environment note: `.venv` is
the dev environment, `.doc-venv` is docs-only, and an ambient
`UV_PROJECT_ENVIRONMENT=.doc-venv` makes a bare `uv run` resolve to the latter —
be explicit when it matters.
