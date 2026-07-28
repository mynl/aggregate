# Changelog

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
