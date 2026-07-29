# [DecL-Colorizer-Resync]

Bring `AggLexer` (and the Sublime mirror) back in line with `decl.lark`.

## Context

`src/aggregate/decl_pygments.py` is a hand-written mirror of the DecL grammar.
It was authored against the 0.15-era language and has not kept pace: the
trailer grew `tags{...}` and `doc{{{...}}}`, labels introduced a quoted
`STRING` class, `//` joined `#` as a comment marker, `@` became the
inhomogeneous-multiply operator, `port.X` / `dist.X` joined `agg.X` / `sev.X`
as builtin references, and `NUMBER` gained Python underscore separators.

The drift is visible, not theoretical. `Token.Error` renders in the `friendly`
style as a **red border box**, and `_colorize` hard-codes `friendly`, so every
labeled object's `pprogram_html` currently shows a red box around each quote in
Jupyter. Measured `Token.Error` counts over the shipped corpora:

| file | Token.Error today | target |
|---|---|---|
| `src/aggregate/agg/library.agg` | 170 | 0 |
| `src/aggregate/agg/decl-testers.agg` | 77 | 0 |
| `src/aggregate/agg/_test_suite.agg` | 0 | 0 |

The existing guard, `tests/test_grammar_sync.py`, walks the 88 reserved words in
the `ID:` negative-lookahead list and asserts each gets a type other than
`Token.Name` / `Token.Error`. It structurally **cannot** catch this class of
drift, because `note` / `tags` / `hints` / `doc` are deliberately absent from
that list (their terminals include the brace, so the bare word is a legal `ID`),
and neither `STRING` nor the operator terminals carry the `.N` priority tag the
test greps for. Widening the guard is as much the point of this work as the
lexer fix.

Outcome: the shipped corpora tokenize with zero `Token.Error`, the four trailer
clauses each get their own treatment, and a corpus-level test makes the next
grammar addition fail loudly instead of rotting quietly.

## Confirmed defects

All thirteen were reproduced by tokenizing real DecL, not inferred.

1. **`tags{...}` unhandled.** `tags` falls to bare `Name`, `{` to `Punctuation`,
   body lexed as DecL, stray `}` to `Name.Type`.
2. **`doc{{{...}}}` unhandled.** No rule at all. The markdown body is lexed as
   DecL; its backticks and quotes become `Token.Error`, and the first `}` in the
   body terminates.
3. **`STRING` unhandled.** `as "My Label"` gives `Token.Error` on each quote.
   Highest-visibility defect (see table above).
4. **`//` comments unhandled.** Only `#.*$` exists. `//` becomes two `Operator`
   slashes and the comment text is lexed as DecL.
5. **`@` unhandled.** `2 @ agg.Base` gives `Token.Error` on `@`.
6. **Underscore separators unhandled.** `1_000_000` gives
   `Number '1'`, `Error '_'`, `Number '000'`, `Error '_'`, `Number '000'`.
7. **`port.X` and `dist.X` / `distortion.X` builtins unhandled.** Only
   `(sev|agg)\.` is matched; the rest split into `Keyword` + `Operator` + `Name`.
8. **`\b` boundaries shatter hyphenated identifiers.** The grammar makes
   `. _ : ~ -` name characters; `\b` does not. So `loss-ratio` tokenizes as
   `Keyword('loss')` + `Operator('-')` + `Name('ratio')`, and likewise
   `no-claims`, `min-premium`, `to-date`.
9. **`mixed` before a newline is uncolored.** The root rule is the literal
   `r'mixed '` with a required trailing space, and `mixed` is absent from the
   keyword list, so `mixed\ngamma` leaves `mixed` a bare `Name`.
   `test_grammar_sync` misses this because it probes with a trailing space.
10. **`Name.Type` is unstyled in `friendly`.** It inherits `Token.Name`, i.e.
    default foreground, so today's `note{` / `hints{` / `}` / `wts` markers are
    invisible.
11. **`dfreq` miscategorized.** Grouped with the FREQ *distribution* names
    (`Name.Function`) while its siblings `dsev` / `dbvsev` / `dwait` are
    `Name.Label`.
12. **Dead Python cruft.** The `expr` state carries `!=|==|<<|>>|:=` and
    `(in|is|and|or|not)\b`; `root` carries `\\\n` and `\\` continuation rules.
    None exist in DecL. The module docstring says "Based on Python lexer", which
    is the root cause.
13. **`sichel.gamma` / `sichel.ig` are not units** in the `mixed_freq` state.

## Design decisions (settled)

- **`doc` body: nested Markdown, lazily imported.** Use Pygments' documented
  `(regex, callback)` rule form so `pygments.lexers.markup` (measured at 113 ms
  first import) stays off the `import aggregate` path and loads only when a doc
  body is actually encountered. `import aggregate` currently pays roughly 0 ms
  for Pygments and must keep doing so (`dev/done/plan-hygiene-1.md` records the
  decision to leave Pygments eager on exactly that measurement).
- **`tags` / `hints` bodies: structured.** Tag slugs get `Name.Tag`; hint keys
  get `Name.Attribute`, `=` and `;` `Punctuation`, values `Number` / `Name`.
  `note` stays prose as `Comment`.
- **No vocabulary coloring.** Distortion kinds, copula kinds and
  `approximate` kinds stay plain identifiers. Only grammar terminals get color,
  so there is nothing to keep in sync with `spectral.py` / `copula.py`.
- **Keyword lists stay hand-written in the module.** Deriving them from
  `decl.lark` at import time would add file I/O to `import aggregate`, and the
  grammar's flat `ID:` list cannot express the per-role coloring (FREQ names as
  `Name.Function`, one- and two-parameter scipy severities distinguished). The
  *test* becomes the source-of-truth guard instead. This matches the house
  preference for explicit over magic.
- **Boundary lookahead replaces `\b` throughout.** Every `words(..., suffix=...)`
  and every keyword rule uses the grammar's own
  `(?![a-zA-Z0-9._:~\-])`, so the lexer's notion of a word boundary is
  identical to `decl.lark`'s by construction.

## Files

| file | change |
|---|---|
| `src/aggregate/decl_pygments.py` | the rewrite (primary) |
| `tests/test_grammar_sync.py` | widen the guard |
| `agg.sublime-syntax` | resync the second mirror |
| `pyproject.toml` | version bump |
| `CHANGELOG.md` | new section |
| `dev/TODO.md` | note the item and close it |

## Work

### 1. `src/aggregate/decl_pygments.py`

Rewrite the module docstring: state that this mirrors `decl.lark` by hand, that
`tests/test_grammar_sync.py` is the guard, and drop the "Based on Python lexer"
lineage that seeded the cruft. NumPy-style, no dashes as punctuation.

Define one module constant for the boundary so it reads once and is used
everywhere:

```python
#: The grammar's own word boundary (decl.lark ``ID``). ``.``, ``_``, ``:``,
#: ``~`` and ``-`` are name characters in DecL, so ``\b`` is wrong: it splits
#: ``loss-ratio`` into ``loss``, ``-``, ``ratio``.
_KW = r'(?![a-zA-Z0-9._:~\-])'
```

**`root` state, in this order.** Order is load-bearing; the notes say why.

1. `doc` fenced form, callback rule. Match
   `(doc\{\{\{[ \t]*\n)([\s\S]*?)(\n[ \t]*\}\}\})`. `[\s\S]*?` rather than
   `.*?` because `RegexLexer` runs under `re.MULTILINE` only, so `.` does not
   cross newlines; non-greedy so the first closing fence wins, matching
   `parser.py::_DOC_FENCE_RE`. The callback yields the fences as
   `Comment.Preproc` and delegates the body to a lazily imported
   `MarkdownLexer` via `get_tokens_unprocessed`, offsetting each index by
   `match.start(2)`.
2. `doc` base64 form, `(doc\{\{\{)([A-Za-z0-9_=-]*)(\}\}\})`, fences
   `Comment.Preproc`, body `Comment`. Mirrors `DOC.3` exactly, so a raw stored
   `.program` string colorizes sanely too.
3. `//.*$` and `#.*$` as `Comment.Single`. Both markers, `//` first so it beats
   the `/` operator.
4. `note\{` push `note`; `tags\{` push `tags`; `hints\{` push `hints`. All three
   fence keywords as `Comment.Preproc` (which *is* styled in `friendly`, unlike
   today's `Name.Type`).
5. `STRING`: `"[^"\n]*"` as `String`. No escape handling and no embedded
   newline, exactly as `decl.lark` line 729.
6. Builtins, all four prefixes, `Name.Builtin`:
   `(?:agg|sev|port|distortion|dist)\.[a-zA-Z][a-zA-Z0-9._:~\-]*`. Note the
   `-` added to the tail class to match `BUILTIN_*.3`.
7. `mixed` + `_KW` as `Keyword`, push `mixed_freq`. No trailing-space
   requirement.
8. `!` as `Operator` (the semantic marker: unconditional severity, mean pin,
   defective `dwait`). Today it is `Generic.Heading`, which is a heading color;
   `Operator` is the honest type.
9. `include('numbers')`, then `include('keywords')`, then `include('operators')`.
   Numbers before operators so a leading `-` is absorbed into the token, which is
   what `NUMBER.2` at priority 2 does.
10. `<[A-Z_0-9*]+>` as `Generic.Heading`, kept: the help renderer emits
    `<PLACEHOLDER>` metavariables.
11. ID catch-all, mirroring `decl.lark` exactly and **without** the trailing
    `\b`: `[a-zA-Z][\._:~a-zA-Z0-9\-]*`.

**Sub-states.**

- `note`: `[^}]*` as `Comment`, then `\}` as `Comment.Preproc` and `#pop`.
  (Today the body rule pops and the closing brace is picked up by a stray
  root-level `\}` rule; that stray rule goes away.)
- `tags`: `[a-zA-Z][\._:~a-zA-Z0-9\-]*` as `Name.Tag`, `[,\s]+` as `Text`,
  `\}` as `Comment.Preproc` and `#pop`.
- `hints`: key `[a-zA-Z][\._:~a-zA-Z0-9\-]*(?=\s*=)` as `Name.Attribute`;
  `[=;]` as `Punctuation`; `include('numbers')`; identifier as `Name`
  (values like `False`, `round`); `\}` as `Comment.Preproc` and `#pop`.
- `mixed_freq`: whitespace as `Text`; then `sichel\.(?:gamma|ig)`, then the bare
  names `gamma delaporte ig sig beta sichel` and the `<DISTRIBUTION>`
  placeholder, all `Name.Function` with `#pop`; finish with
  `default('#pop')` (from `pygments.lexer`) so an unrecognized follower cannot
  strand the state.

**`numbers`.** Collapse the four overlapping rules to one that mirrors
`NUMBER.2` verbatim, plus an integer rule for the `Number.Integer` distinction:

```python
'numbers': [
    (r'-?(?:\d(?:_?\d)*\.?(?:\d(?:_?\d)*)?|\.\d(?:_?\d)*)'
     r'(?:[eE][+\-]?\d(?:_?\d)*)?%?' + _KW, Number),
    (r'-?inf' + _KW, Number),
],
```

**`keywords`.** Same five `words()` groups as today (FREQ names, zero- /
one- / two-parameter scipy severities, the discrete declaration keywords, the
reserved-word list), with three changes: every `suffix=r'\b'` becomes
`suffix=_KW`; `dfreq` moves out of the FREQ-distribution group into the
`Name.Label` group beside `dsev` / `dbvsev` / `dwait`; the builtin rule moves up
to `root` (item 6 above). Add `splice`, `and` and `wts`, currently reachable
only through special-cased `root` rules that emit heading and type colors; as
ordinary reserved words they belong in the list and the special cases go away.
Do **not** add `mixed`: root item 7 already gives it `Keyword`, and a second
entry here would be unreachable.

> **Ordering trap, verified in a prototype.** If the `mixed` push rule is placed
> after `include('keywords')`, the keyword rule wins, the push never fires, and
> the following mixing distribution silently loses its color. Root item 7 must
> stay ahead of item 9.

**`operators`** (renaming today's `expr` state, whose name was a Python
leftover). Delete `!=|==|<<|>>|:=` and `(in|is|and|or|not)\b`; delete the `\\`
continuation rules from `root`. Keep exactly the grammar's operator set:

```python
'operators': [
    (r'[^\S\n]+', Text),
    (r'\*\*|\^', Operator),           # EXPONENT
    (r'[-+*/@=]', Operator),          # MINUS PLUS TIMES DIVIDE INHOMOG_MULTIPLY EQUAL_WEIGHT
    (r'[][():]', Punctuation),        # brackets, parens, RANGE
    (r'[,|]', Text),                  # %ignore treats these as whitespace
    (r';', Punctuation),              # statement separator
],
```

### 2. `tests/test_grammar_sync.py`

Keep both existing tests. Add:

- **`test_corpus_has_no_error_tokens`**, parametrized over `library.agg`,
  `_test_suite.agg` and `decl-testers.agg` (locate them relative to
  `aggregate.__file__`, as the existing suite does). Tokenize each file whole
  and assert zero `Token.Error`, reporting the offending values on failure.
  This is the guard that would have caught every one of the thirteen defects,
  and it costs three lexer passes.
- **`test_construct_colored`**, a table of `(snippet, substring, expected token
  prefix)` covering the constructs the reserved-word walk cannot reach:
  `tags{...}`, `doc{{{...}}}` in both fenced and base64 form, a quoted label,
  `//` and `#` comments, `@`, `1_000_000`, `port.X`, `dist.X`, `loss-ratio` as
  one `Name`, and `mixed` followed by a newline.
- **`test_sublime_syntax_keywords_match_grammar`**: extract the alternations
  from `agg.sublime-syntax` (the `{{keyword_boundary}}` groups and the ID
  exclusion list) and assert set equality with `grammar_reserved_words()`. Today
  that diff is 33 missing and 2 stale; after the resync it is empty. Update the
  module docstring, which currently says the Sublime file is unchecked.

Fix the docstring's `TODO D10` reference to a descriptive label per CLAUDE.md.

### 3. `agg.sublime-syntax`

Same content changes, in Sublime's idiom. Measured diff against the grammar's
88 reserved words:

- **33 missing**: `after approx approximate as basic bivariate bv cede corridor
  dbvsev deposit dwait expense expenses free inherit lcm less max min no payoff
  pc reinstatement reinstatements retro rol slide swing wait xpnl year years`
- **2 stale**: `multivariate mv` (no longer in the grammar)

Also: add `tags` and `doc` contexts alongside `note_hints`; add `//` to the
`comments` context; add `port\.` to `builtins`; drop `\\` from the separator
class (continuation was removed from the language); tighten `inside_string` to
reject newlines and drop the `\\.` escape rule, since `STRING` supports
neither; regenerate the ID exclusion list from the grammar's.

### 4. Release hygiene

One commit, one coherent unit: bump `1.0.0a174` to `1.0.0a175` in
`pyproject.toml`, add the matching `## 1.0.0a175` section to `CHANGELOG.md`
(house format, `**[DecL-Colorizer-Resync]**` then prose), record the item under
`dev/TODO.md`'s "Hygiene & tests" heading as done, and move this plan to
`dev/done/plan-colorizer-resync.md`. Commit subject, one line, no body:

```
[DecL-Colorizer-Resync] a175: tags/doc/labels/// colored, zero Token.Error across the corpora
```

Docs are not rebuilt as part of this (house rule); note in the CHANGELOG that
`docs/4_dec_Language_Reference.rst`'s `literalinclude :language: agg` will pick
the change up on the author's next build.

## Prototype evidence

The four mechanisms this plan leans on were each run before being written down,
so the executor is not discovering them fresh:

- `words(..., suffix=_KW)` turns `loss-ratio`, `no-claims` and `min-premium`
  into a single `Name` each, where `\b` splits all three today.
- The single `numbers` regex accepts `1_000_000`, `25%`, `-inf`, `.5`, `10e6`,
  `1.2343e2` and `-3` as one `Number` apiece.
- The doc callback yields `Comment.Preproc` fences around a body that
  `MarkdownLexer` resolves into `Generic.Heading`, `String.Backtick` and
  `Generic.Strong`.
- `default('#pop')` cleanly exits `mixed_freq` on an unrecognized follower.

## Verification

Before and after, the headline number:

```powershell
.venv/Scripts/python.exe -c "from pathlib import Path; import aggregate; from pygments.token import Token; from aggregate.decl_pygments import AggLexer; d=Path(aggregate.__file__).parent/'agg'; [print(f, sum(1 for t,_ in AggLexer().get_tokens((d/f).read_text(encoding='utf-8')) if t is Token.Error)) for f in ('library.agg','_test_suite.agg','decl-testers.agg')]"
```

Expect `170 / 0 / 77` before and `0 / 0 / 0` after.

Then:

1. **Edit loop.** `uv run pytest -n0 --dist no --testmon-forceselect` while
   iterating on the lexer.
2. **Targeted.** `uv run pytest tests/test_grammar_sync.py tests/test_decl_unparser.py tests/test_fcc_surface.py -q`
   (the unparser file holds the html / ansi / latex smoke tests;
   `test_fcc_surface.py:281` asserts the `pprogram_html` contract).
3. **Gate.** `uv run pytest` before declaring done.
4. **Bump.** `uv run pytest -m 'slow or not slow'` once, at the commit boundary.
5. **Eyeball the real output.** Render a feature-complete statement and confirm
   no `class="err"` survives and the doc body highlights as markdown:

   ```python
   from aggregate import build
   a = build('agg TT as "My Book" 10 premium as "GWP" at 0.65 lr '
             'sev lognorm 10 cv 2 poisson '
             'note{a note} tags{role:hero, topic:aggregate} hints{log2=12}')
   html = a.format_program(fmt='html', trailer=True)
   assert 'class="err"' not in html
   print(a.format_program(fmt='ansi', trailer=True))
   ```

6. **Import cost unchanged.** `python -X importtime -c "import aggregate"` must
   not show `pygments.lexers.markup`; it may only appear after a doc body is
   colorized.

## As executed, `1.0.0a175`

Landed as planned, with five corrections found while running it.

1. **No trailing guard on the numeric rule.** Appending `_KW` to the number
   regex looked right and was wrong: `:` is a name character, so the guard
   rejected `1` in `[1:6]` and `[10:50:10]`, taking `_test_suite.agg` from 0 to
   36 errors. The grammar's `NUMBER.2` carries no lookahead for exactly this
   reason. Only the `inf` spelling keeps one, so `infinity` stays one
   identifier.
2. **`dhistogram` and `chistogram` moved the other way.** The plan grouped them
   with `dsev` / `dbvsev` / `dwait` as declaration keywords, inheriting an
   existing miscategorization. They are severity distribution *names*, reached
   through the grammar's `ids` rule as `sev dhistogram xps [0 99] [.8 .2]`, so
   they now sit with the scipy severities. `dfreq` still moves the other way,
   as planned.
3. **`mixed_freq` needs `\s+`, not `[^\S\n]+`.** With horizontal whitespace
   only, a newline between `mixed` and its mixing distribution fell to
   `default('#pop')` and the distribution lost its color. That is the same
   defect the plan set out to fix, one layer down.
4. **`hints{}` gained `Keyword.Constant`** for `True` / `False` / `None`, which
   appear in real hint bodies (`normalize=False`).
5. **Four corpora, not three.** `_test_suite2.agg` also ships and is now
   covered.

The severity-name recoloring and the distortion / copula / approximation
vocabulary that a design pass proposed were both declined: the first is a
visual redesign rather than a resync, and the second was ruled out at planning
time to avoid a third hand-maintained vocabulary. `[Colorizer-Style-Choice]`
went to `dev/TODO.md` as the one genuinely open follow-up.

Result: 0 `Token.Error` and a lossless round trip on all four corpora, 3039
fast tests green, `test_grammar_sync.py` from 169 cases to 570.
