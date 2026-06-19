# Plan: blank-line + `;` statement separation in DecL

Status: **design settled (decisions in §9); ready to implement on approval.**

## 1. What is being asked

Change the rule that splits a DecL *program* (a blob of text) into individual
*statements*.

- **Today:** a statement is "one line". A column-0 line begins a new statement;
  an **indented** line or a line ending in a **`\` backslash** is folded onto the
  previous statement. Indentation is the *only* thing that makes a multi-line
  `port` work.

- **New rule (settled):** two ways to end a statement, either alone or mixed:
  1. a **blank line** (`\n\n`; blank = empty **or whitespace-only**) — the
     markdown paragraph model; and
  2. a **`;` at end of line** (`;\s*$`, tested *after* comment-stripping) — the
     Python model, so dense one-per-line lists stay legal.

  Any newline *within* a paragraph that is not a blank line and not preceded by a
  `;` is just a space, so a statement may be laid out freely. The `\` continuation
  is **removed** and a stray `\` becomes a hard error (§6.4).

## 2. Where the rule lives (single owner)

`UnderwritingLexer.preprocess(program) -> list[str]` in
`src/aggregate/parser.py` (currently ~lines 91–135). **Every** path funnels
through it:

- `Underwriter._interpret_program` (`underwriter.py:1070`) — loops the list, parses each.
- `Underwriter._read_file` (`underwriter.py:621`) — reads a `.agg` file, hands the whole text to `_interpret_program`.
- `Underwriter.interpreter_file/_test` (`underwriter.py:1386`+) — diagnostic path; branches on `len(preprocessed) == 1` vs `> 1`.
- Tests: `tests/conftest.py:17`, `tests/test_decl_parser.py:37`, `tests/capture_sly_snapshot.py:53` all call `preprocess` to get the per-statement corpus — so they pick up `;` / blank-line handling for free.

Because the `;` is **consumed by `preprocess`** before the lexer runs, the token
stream handed to the parser for a statement is identical with or without a
trailing `;`. Confirmed: `;` is not a grammar token in `decl.lark` (it appears
only in comments and in the `hints{key=value;}` description), so **no grammar
change is needed for `;`.**

## 3. Why blank-line / `;` is the right model (the `agg` ambiguity)

The cheaper-looking alternative — "a new statement begins at a column-0 line
whose first word is a top-level keyword" — **does not work**: `agg` and `pnl` are
both top-level statement keywords *and* the sub-unit keywords inside `port` /
`multivariate`, and the current grammar only distinguishes a port's units from
standalone aggs **by indentation**. A keyword splitter would shred every
portfolio into its units.

The blank-line rule resolves this cleanly: a `port` and its units are **one
paragraph** (no internal blank line, no `;` between units); the next statement
starts after a blank line or a line-final `;`. This is the only model that makes
layout indentation-insensitive while keeping `agg` units nested in `port`. Adding
`;` on top restores dense lists without reintroducing indentation significance.

## 4. Assessment — is it a good idea?

**Yes — adopt it.** The `;` decision removes the one serious objection.

Advantages
- Indentation carries **no** semantic weight → robust against editors that
  reflow/strip whitespace (the stated motivation).
- Removes `\` continuation, whose worst trait today is that a leftover `\` is
  *silently ignored* (it sits in `decl.lark`'s `%ignore` class) — invisible
  confusion, especially next to Python's free string-concatenation inside `()`,
  which makes a broken multi-line DecL string *look* right. Making `\` an error
  (§6.4) turns the parser into a finder for these.
- One familiar model: markdown paragraphs **or** Python `;`, freely mixed.
- Dense one-per-line lists survive via `;` → migration is "append `;`", which is
  **spec-neutral** (the statement text the parser sees is unchanged), so
  `expected_specs.json` / the SLY snapshot do not drift.

Residual risks (manageable)
- Two adjacent statements with **neither** a blank line **nor** a `;` between
  them glue into one paragraph. This is almost always a loud parse error (no
  valid `answer` production), not a silent misparse — but the error column can be
  confusing. Small residual chance a glued pair accidentally parses.
- Migration touches every `.agg` file and the DecL doc blocks (mechanical,
  scriptable — §8).
- A few preprocessing corners (§6) must be exact: blank = whitespace-only;
  comment-strip must not inject a paragraph break; `;` detected after
  comment-stripping and outside `[ ]`.

## 5. Blast radius — inventory

### 5.1 Code (small)
- `src/aggregate/parser.py` — rewrite `preprocess` + docstring (the core change).
- `src/aggregate/decl.lark` — drop `\\` from `%ignore` (line 427:
  `%ignore /[ \t,|]+/`) so a stray `\` errors loudly. No other grammar change.
- `src/aggregate/underwriter.py` — review the "preprocesses to N lines" wording
  in `interpreter_file/_test`; N now counts paragraphs. No logic change expected.

### 5.2 Corpus `.agg` files (mechanical, spec-neutral)
`src/aggregate/agg/`: `test_suite.agg` (274 lines; 136 statement lines sit in 23
comment-free dense runs), `test_suite2.agg`, `test_decl.agg`, `spa_examples.agg`,
`spa_examples-old.agg`, `other-distributions.agg`, `testers.agg`, and
`examples.agg` (also delete its 18 trailing `\`).

Migration choice (recommended): **append `;` to the end of every top-level
statement**, keeping the dense one-per-line layout. For multi-line `port`/`mv`
blocks, the `;` goes only at the **end of the whole block** (units stay
unterminated so they remain one paragraph); delete any `\`. Because `preprocess`
strips the `;`, the parsed specs are unchanged → snapshot stable.

### 5.3 Tests — see §7.

### 5.4 Docs — a dedicated, careful sweep of `./docs` (`.rst`)

This is the largest mechanical piece **and the noisiest**, so it gets its own
work item rather than a one-liner. The `\$` grep hits ~546 lines across 33 files,
but the **overwhelming majority are Python `\` continuations inside
`.. ipython::` / matplotlib blocks and LaTeX `\\` row-breaks in math and tables
— those MUST be left untouched.** Only DecL is in scope. Concretely, three
categories of edit:

**(a) Prose that documents the old rule — rewrite.**
- `docs/4_dec_Language_Reference.rst:20–31` — the five-step preprocess list
  ("Programs are processed one line at a time", "Map backslash newline (Python
  line continuations)", "Replace \\n\\t … tabbed indented Portfolio layout") **and**
  the "following characters are ignored … may be used freely" list. Rewrite to
  the paragraph (blank-line) + line-final `;` model; the ignored-char list loses
  `\` (now an error, §6.4) and should note that a trailing `;` ends a statement.
- `docs/1_Getting_Started.rst:92` — "The Python line continuation ``\`` is used to
  create compact input." → rewrite to the `;` / blank-line compact-input story.
- Re-grep `docs/2_user_guides/2_x_dec_language.rst` and the `DecL/` subsection
  intros for any "one line" / "continuation" / "indent" prose (current grep finds
  none in `2_x_dec_language.rst`, but re-verify after the code change lands).

**(b) DecL multi-line code blocks using `\` — de-chatter.** Remove the trailing
`\`; the lines then join as one paragraph, or split them with a blank line / `;`
to match the new idiom. Known DecL-`\` files (counts approximate, hand-verify):
- `docs/2_user_guides/DecL/080_reinsurance.rst` (~10),
  `070_vectorization.rst` (~4), `010_Aggregate.rst` (~3),
  `065_limit_profiles.rst` (~2), `060_mixed_severity.rst`, `100_tweedie.rst`;
- `docs/2_user_guides/2_x_10mins.rst`, `2_x_re_pricing.rst`, `2_x_cat.rst`;
- `docs/5_technical_guides/5_x_multivariate.rst`.
  **These files interleave DecL and Python blocks** — e.g. `2_x_10mins.rst` has
  Python `\` at lines 561–567 and 1305–1312 (matplotlib) that **stay**, alongside
  DecL `\` elsewhere that **goes**. So this is a per-block judgement, not a
  file-level find/replace.

**(c) DecL embedded in Python strings.** `build('…')` / `build(r'''…''')` calls
whose string spans lines with `\`, plus any `.. parsed-literal` / inline DecL.
Grep `build\(` blocks for embedded `\`.

**Method / guardrails:**
- Discriminator is **context**: a backslash inside a DecL program (a line
  carrying `agg`/`port`/`sev`/`pnl`/`dfreq`/`dsev`/`claims`/`xs`/`mixed`/`cv`…,
  or inside a block known to be DecL) → remove. A backslash in `.. ipython::`,
  `.. code-block:: python`, matplotlib, or math (`\\`, `\mid`, `\Pr`, `\alpha`) →
  leave.
- **Do not mass-`sed`.** Go file-by-file; for each DecL block decide
  join-vs-blank-line-vs-`;`.
- After edits, re-grep for DecL-context `\` to confirm none remain, and
  `rg "one line|line continuation|tabbed indented|backslash newline"` to confirm
  the old-rule prose is gone.
- Per CLAUDE.md: keep `.rst` edits in lockstep with the code change, **do not
  rebuild docs in the loop** (the author rebuilds the 500+ page tree manually),
  and note "docs pending rebuild" in the commit/PR.
- Out of repo: the author's Annals paper DecL listings use the same `\` idiom —
  list them for the author to fix (we do not edit them here).

## 6. Preprocessing algorithm and corners

Proposed order inside the new `preprocess`:

1. **Collapse `[ ]` vector newlines → space** (unchanged; a numpy-formatted
   vector may contain newlines and must not be read as a paragraph break). This
   also pulls any `;`/whitespace inside a vector off the line end.
2. **Strip `#` / `//` comments to *empty string*** (not to `\n`). Leaving the
   line's own newline means a trailing comment (`agg A ... ;  # note`) does not
   inject a blank line that would split a multi-line paragraph; a full-line
   comment collapses to a genuinely blank line, which correctly separates.
   Stripping before step 3 also exposes a `;` hidden behind a trailing comment.
3. **Turn a line-final `;` into a paragraph break:**
   `re.sub(r';[ \t]*(\r?\n|$)', '\n\n', text)`. Only a `;` at end of (stripped)
   line fires; `hints{...}`/`note{...}` always end a line with `}`, so their
   internal `;` is never line-final and is safe.
4. **Split into paragraphs on blank-line runs:** `re.split(r'\n\s*\n', text)`
   (`\s` covers whitespace-only lines and runs of blanks).
5. **Flatten each paragraph:** `' '.join(p.split())` collapses intra-paragraph
   newlines, indentation, and repeated spaces to single spaces; drop empties.

Corners:
- 6.1 **Blank = whitespace-only** — handled by `\s` in step 4 and `p.split()` in 5.
- 6.2 **`;` only at end of line** (`;\s*$`) — protects hints/notes; mid-line `;`
  outside `hints{}` reaches the lexer and errors (acceptable; it is a syntax
  error). Matches the author's `;$` intent.
- 6.3 **Order matters** — brackets (1) before comments (2) before `;` (3) before
  paragraph split (4).
- 6.4 **`\` is now an error** — removed from `decl.lark`'s `%ignore` so a stray
  backslash raises a clear lexer error instead of vanishing. The migration
  deletes every `\`; any that survive are surfaced loudly. (A `\`-continued pair
  with the `\` simply deleted becomes two adjacent lines → one paragraph → joins
  correctly, which is the desired result.)

## 7. Test-harness changes

- `tests/conftest.py`, `tests/test_decl_parser.py`, `tests/capture_sly_snapshot.py`
  — already route through `preprocess`; they gain `;`/blank-line handling for
  free. Verify nothing asserts a *fixed* case count. Re-run the snapshot capture
  (or assert the Lark specs still match the frozen `expected_specs.json`): specs
  should be **identical** since the per-statement tokens don't change.
- `tests/test_decl_unparser.py`:
  - `_corpus_lines` (line ~50) currently iterates `read_text().splitlines()` and
    treats each **physical line** as a program — it bypasses `preprocess`, so it
    would choke on a trailing `;` and on any genuinely multi-line statement.
    **Rework it to call `UnderwritingLexer.preprocess(file_text)` and iterate the
    returned statements** (derive the id from the statement's second token, as
    now). This is strictly more robust and makes the dense-`;` corpus work.
  - `test_format_port_is_multiline` (line 224) asserts the unparser emits
    `port P\n\tagg ...` (tab-indented). A tab-indented, blank-free block is one
    paragraph under the new rule → it round-trips. **No unparser change needed.**
- Add dedicated cases to `tests/` + `src/aggregate/agg/test_decl.agg` (keep them
  in sync, per house rule) covering: `;`-terminated dense list; blank-line
  separation; multi-line `port` with no `\`; a wrapped single statement; a stray
  `\` now raising; `;` inside `hints{}`/`note{}` *not* splitting.

## 8. Migration approach (safe + reviewable)

One-shot helper in `dev/` (not shipped): for each `.agg` file, delete trailing
`\`, then append `;` to each top-level statement end (detected with the *current*
column-0 rule; indented `agg`/`pnl` lines are port units and are left
unterminated, with the `;` placed after the block). Run it, eyeball the diff,
`uv run pytest`. Because the change is spec-neutral, a green suite + an unchanged
`expected_specs.json` is strong evidence the migration is faithful. Repeat the
same mechanical pass on the docs DecL blocks.

## 9. Decisions (settled with author)

1. **Statement separators = blank line *and* line-final `;` (`;\s*$`).** Keeps
   dense lists (Python-style `;`) alongside markdown paragraphs. `;` is handled
   entirely in `preprocess`; no grammar token.
2. **Clean break — no compatibility shim.** Old `\`/indent folding is removed
   outright; the corpus and docs are migrated in the same change.
3. **`\` is hunted down and removed, and made a hard error** (dropped from the
   `%ignore` class) so leftover backslashes surface loudly rather than vanish —
   addressing the real-world confusion in the docs and the Annals paper.

## 10. Version / housekeeping (per CLAUDE.md)

Code change from a plan → bump `1.0.0a*` in `pyproject.toml`; add a `CHANGELOG.md`
entry (new blank-line + `;` separation; `\` removed and now an error; breaking
change note); update `dev/TODO.md`; move this plan to `dev/done/` on completion.
Docs `.rst` edited in lockstep, **not** rebuilt in the loop — flag "docs pending
rebuild" and list the Annals paper for the author.
