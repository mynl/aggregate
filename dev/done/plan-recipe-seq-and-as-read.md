# Plan: `Recipe.seq` and `Recipe.as_read`

Status: drafted 2026-08-24.
Label: `[Recipe-Seq-As-Read]`.
Consumer: `aggregate_api` `dev/plan-examples-dropdown.md`, which cannot present
the library in reading order or show the DecL the author wrote without these.

## Why

`library.agg` is written as a reading order, and the entries in a section build
on one another. Two things currently destroy that on the way out.

**Order.** `Underwriter._recipes_frame` ends in `.sort_index()`, so
`build.recipes` is alphabetical by `(kind, name)`. The as-read order survives
only in the private `_recipes` dict, which is insertion ordered because
`_read_file` walks the file top to bottom.

**Text.** `Recipe.program` is the statement as the parser received it: one line,
comments stripped, whitespace collapsed. `Recipe.decl` is the canonical
re-render. Neither is what the author typed, so a consumer wanting to show a
library entry has to pick between a one-liner and a re-render in which
`ph 2/3` has become `ph 0.6666666666666666`, `ceded to tower [0 25 50 75 100
125]` has become five and-chained layers, and `dsev [1:6]` has become
`dsev [1 2 3 4 5 6]`.

Sections are deliberately **not** part of this. The `# ---` banners in
`library.agg` are comments and stay comments; nothing in `src/aggregate` parses
them, which is what the file header already promises. Reading order is carried
by `seq` alone, and a consumer that wants headings can invent its own or, better,
do without.

## The two fields

Both land on `Recipe` (a dataclass, so they are two more fields) and as two more
columns on the `recipes` frame.

### `seq : int`

The zero-based order in which the entry was read, assigned by `_read_file` as it
walks a database. Unique within a load of the whole recipe base, so an entry
read from the second `.agg` file continues the count rather than restarting.

A session-built entry gets the next number at registration, so a user's own
programs sort after the library, which is where a consumer would want them.

### `as_read : str`

The entry's DecL exactly as it appears in the source file: multi-line, indented
as written, comments stripped, trailer included. `''` for a session-built entry,
which never had a file to come from.

The name pairs with `decl` (the canonical render) and does not collide with
`source`, which already means the file path or `'session'`. `source_text` is the
alternative if `as_read` reads too clever.

The preprocessor already splits the file into statements before the lexer
touches them, so the raw span is in hand at exactly the moment the `Recipe` is
constructed; this is a matter of keeping it rather than fetching it.

## Work

1. `Recipe`: add `seq: int = -1` and `as_read: str = ''`, and document both in
   the class docstring's Parameters block.
2. `Underwriter._read_file`: keep a counter, keep the raw statement text, pass
   both through to the `Recipe`.
3. Session builds: assign `seq` at registration from the same counter.
4. `_recipes_frame`: emit `seq` and `as_read` columns, and add `seq` to
   `_RECIPE_COLUMNS`. **Keep the trailing `.sort_index()`**: the frame's default
   presentation stays alphabetical, which is what a person reading it at a
   prompt wants, and a consumer that wants file order says
   `recipes.sort_values('seq')`. Additive, so nothing that reads the frame today
   changes.
5. `discover` gains nothing; it filters the frame and inherits both columns.

## Tests

* `tests/test_agg_libraries.py`: `seq` is a permutation of `range(len(recipes))`,
  and sorting the shipped library by it reproduces the order the statements
  appear in `library.agg`.
* `as_read` re-parses to the same spec for every entry, which is the real
  contract: it is DecL, not a comment.
* `as_read` is `''` for a session build and non-empty for every library entry.
* A session-built entry gets a `seq` greater than every library entry's.

## Not in this plan

**Provenance for desugared clauses.** `as_read` gives a consumer the source
text, which is enough to *show* an entry correctly. It does not help
`format_program`, which still cannot invert `2/3`, `ceded to tower [...]`,
`[1:6]` or `1_000` because the parser evaluated or expanded each one and put
only the result on the spec. That work is the `_tweedie` provenance pattern from
a231 generalized, tracked under `[Unparser-Reference-Gaps]` in `dev/TODO.md`,
and it is a separate decision.

**One exception worth taking now, because it needs no provenance at all.**
`splice [a b]` and `splice [a] [b]` parse to the same spec, and
`splice_one` is a contiguous-band spelling: from breakpoints `[b0 b1 ... bn]` it
builds `sev_lb = [b0 ... b(n-1)]`, `sev_ub = [b1 ... bn]`. So the compact form is
recoverable by inspection: emit `splice [lb[0], *ub]` whenever
`lb[1:] == ub[:-1]`, and the two-list form otherwise. Verified: `splice [5 25]`
and `splice [5] [25]` both give `lb=[5.0], ub=[25.0]`, while
`splice [0 10] [5 50]` gives `lb=[0, 10], ub=[5, 50]` and fails the test. A
single-band splice always takes the compact form, which is what the library
writes. One edit to `_render_splice`, and `SevSpliced` leaves the non-canonical
list.

## Execution log

Executed 2026-08-24 at `1.0.0a320`, in one bump, against a working tree carrying the author's in-flight `library.agg` reorganization.

### Divergences

1. **`as_read` is recovered by a dedicated splitter, not by keeping a span.** The plan says the preprocessor "already splits the file into statements before the lexer touches them, so the raw span is in hand". It is not. `UnderwritingLexer.preprocess` is a single pass of string rewrites, and by the time it splits (step 5) the text has already lost its comment lines (step 1), had newlines inside brackets turned to spaces (step 3) and had every line-final `;` rewritten to a blank line (step 4). There is no surviving span to keep. The new `UnderwritingLexer.raw_statements` therefore mirrors the *separation* rules on the original text, line at a time, and shares only step 0b's trailer lift with `preprocess`.

2. **The trailer lift is not optional, and finding out why is the reason this took a prototype.** A first version stripped comments with a plain regex and merged `AG.Note.Slashes`, `AG.Note.Semicolon` and `AG.Note.WithRest` into their neighbours: those three fixtures exist precisely to prove that a `#`, a `//` or a `;` inside `note{...}` is prose. Lifting trailer bodies behind placeholders first, exactly as step 0b does, took `decl-testers.agg` from 412 blocks against 415 statements to 415 against 415.

3. **`as_read` is not text-comparable to `program`, and the tests do not compare them.** `preprocess` reformats around brackets on its non-nested path, so a source `dfreq[1]` flattens to `dfreq [1]`; eight `library.agg` entries differ from their source on exactly that. Worse, the path is chosen by whether the *whole file* contains a nested `[[...]]`, so the same statement flattens one way inside `decl-testers.agg` and another way standing alone. The contract pinned instead is the plan's own test 2: `as_read` parses to the same spec. Measured across all four corpora before writing any code: 717 entries, zero spec differences, zero parse errors.

4. **`as_read` is filled only for entries read from a file.** The plan says `''` for a session build, which this keeps, but it also means the raw split runs once per file read rather than on every `build(...)` call. The pairing is positional and guarded: if the two splitters disagree on count the source text is dropped for that file with a logged warning, because attaching it to the wrong entry is far worse than not having it.

5. **`as_read` joins `_RECIPE_COLUMNS`, which the plan's item 4 does not say.** Item 4 asks for both columns but names only `seq` for the tuple. `_recipes_frame` ends in `df[list(_RECIPE_COLUMNS)]`, which *selects*, so a column outside the tuple would have been computed and then dropped. Order is `seq` first (one integer, and the thing a caller sorts on) and `as_read` last, behind `spec`, per the tuple's own comment about keeping the frame readable at a terminal.

6. **`seq` is assigned inside `add_recipe`, not passed in.** It is the entry's position in the recipe base, which only the store knows. A monotonic `_take_seq` counter backs it rather than `len(self._recipes)`, since entries are overwritten and a fork shares the parsed dict, so size is not a position.

7. **Rebuilding an existing name keeps its `seq`.** The plan rules that a session build takes the next number, which holds for a new name. It is silent on an overwrite. Keeping the place makes reading order stable under a rebuild, which is the property a consumer presenting the library actually needs; the alternative would silently move an entry to the end of the list because the user re-ran it.

8. **Tests went to `tests/test_recipe.py`, not `tests/test_agg_libraries.py`.** They are about `Recipe` field semantics, and `test_agg_libraries.py` carried the author's uncommitted edits throughout. `test_recipe.py`'s existing exact-column assertion was updated in the same edit, since it pins the frame's column list.

9. **The plan's splice example is contradicted by the code, and the change is simpler than described.** It says `splice [0 10] [5 50]` "gives `lb=[0, 10]`, `ub=[5, 50]` and fails the test". In fact `Severity` refuses it: `Splice bound must be a scalar or length-1 sequence ... Multi-segment splice is not implemented`. Every splice that parses today is a single band, so the `lb[1:] == ub[:-1]` guard is always satisfied. The guard is kept anyway, because the refusal is a current limitation rather than a statement about the grammar, and a future multi-band splice would need it. `SevSpliced` and `BivariateCatPair`, the two splice bearing library entries, both render canonically now.

### Not done

`SevSpliced` was already absent from `UNPARSER_EXEMPT`, so no exemption had to be retired.
