# Plan — program-display punch-ups

Two independent display/formatting punch-ups, each its own version bump. They do
not share code; collected here because both are "make the interactive display
nicer" work.

- **Punch-up 1 — `format_program` spread layout** (below): a multiline, indented
  default layout for rendered DecL.
- **Punch-up 2 — `help` / `agg_help` `fmt` rendering** (end of file): a render-target
  axis (`auto` / `text` / `ansi` / `html`) so `.help()` reads well in a terminal,
  not just Jupyter.

`format_program` and `help`/`agg_help` are unrelated functions; **punch-up 2 does
not touch `format_program` or `pprogram`.**

---

# Punch-up 1 — `format_program` spread layout

**Goal.** Add a multiline, indented "spread" layout to `format_program` (and the
`pprogram` / `pprogram_html` properties that back it), and make spread the
**default**. The current single-line form becomes the opt-in `'terse'` layout.

**Decisions (confirmed with author):**

- **Parameter name:** `layout`, values `'spread'` (default) and `'terse'`. Sits
  alongside the existing `fmt=` (markup: text/html/ansi/latex) and `width=`
  (reserved for wrapping). New axis = line layout, orthogonal to color.
- **Default:** spread, everywhere `format_program` is called with defaults —
  including `Aggregate.pprogram` / `.pprogram_html` and `Portfolio.pprogram` /
  `.pprogram_html`, hence the doc examples that print them.
- **Indent unit:** two spaces per nesting level.
- **`spec_to_decl` stays terse-only.** The `to_agg` exporter and the
  round-trip/idempotence tests go through `spec_to_decl`; leaving it terse keeps
  `.agg` files and the snapshot contract unchanged. Spread lives entirely in the
  rendering layer.

## Target output

Standalone aggregate (spread):

```
agg A
  10 claims
  1000 xs 0
  sev 10 * lognorm 0.3 splice [0 100]
  occurrence net of
    75% so 100 xs 200 and
    50% so 100 xs 300
  mixed gamma 0.5
  aggregate net of
    100 xs 0
```

Portfolio (spread) — the agg head sits one level in, its clauses one level
deeper, reinsurance cessions one level deeper again:

```
port ONE
  agg A
    10 claims
    1000 xs 0
    sev 10 * lognorm 0.3 splice [0 100]
    occurrence net of
      75% so 100 xs 200 and
      ...
    mixed gamma 0.5
  agg B
    ...
```

Terse (`layout='terse'`) reproduces today's output **byte-for-byte** — single
line per agg/pnl/sev/bvagg/distortion, and the existing tab-indented-units form
for `port`. (Terse is the regression anchor; see Testing.)

## Why this re-parses safely

`UnderwritingLexer.preprocess` (parser.py:93) flattens every newline +
indentation *within* a statement to a single space; only a **blank line** or a
line-final `;` starts a new statement. So an arbitrarily-indented multiline
statement collapses back to the same token stream the terse form produces — the
spec is identical. The portfolio renderer already relies on exactly this (its
tab-indented units fold back into one statement). Spread just extends the
indentation that is already proven to round-trip.

**Correctness corollary — statement separation.** `format_program` joins
top-level statements with `'\n'` today (decl_writer.py:783). With single-line
statements that is already fragile (a lone `\n` is *not* a statement separator,
so two statements would fold into one on re-parse); with multiline spread output
it is definitely wrong. Fix: join top-level statements with a **blank line**
(`'\n\n'`) in both layouts. In practice `obj.program` is a single statement, so
this only hardens the rarely-hit multi-statement path — but it must be correct.

## Design

The top-level renderers already build an ordered list of clause fragments and
hand it to `_join` (single-space join). Spread needs (a) those fragments on their
own indented lines and (b) two places where a fragment itself has indented
children: reinsurance cessions, and a portfolio's / bivariate's sub-aggregates.
Introduce a minimal tree node so both layouts render from one structure — no
duplicated clause-ordering logic, no drift.

### A small intermediate node

```python
class _Block:
    """A head line plus indented children, rendered terse (flattened to one
    line) or spread (head, then each child one indent deeper)."""
    __slots__ = ('head', 'children', 'sep')
    def __init__(self, head, children, sep=''):
        self.head = head          # str: the line that introduces the block
        self.children = children  # list[str | _Block]
        self.sep = sep            # connective between children: '' or 'and'
```

A clause fragment is either a plain `str` (renders the same on its own line in
both modes) or a `_Block`. Two renderers below produce identical *terse* output
to today and the *spread* output above:

```python
def _render_terse(node, _depth=0):
    if isinstance(node, str):
        return node
    join = f' {node.sep} ' if node.sep else ' '
    body = join.join(t for c in node.children if (t := _render_terse(c)))
    return _join([node.head, body])     # _join drops empties, single-spaces

def _render_spread(node, depth=0, indent='  '):
    pad = indent * depth
    if isinstance(node, str):
        return pad + node
    lines = [pad + node.head] if node.head else []
    last = len(node.children) - 1
    for i, c in enumerate(node.children):
        child = _render_spread(c, depth + 1, indent)
        if node.sep and i != last:
            child += f' {node.sep}'         # trailing 'and' on the line
        lines.append(child)
    return '\n'.join(lines)
```

- terse `_Block('occurrence net of', [c1, c2], sep='and')`
  → `occurrence net of c1 and c2` (unchanged from today).
- spread → `occurrence net of` / `  c1 and` / `  c2`.

### Renderer changes (build nodes instead of pre-joined strings)

Refactor the four structural renderers to return a `_Block` (or a plain `str`
when there is no nesting) rather than calling `_join` themselves. The clause-level
helpers (`_render_exposure`, `_render_layers`, `_render_sev_clause`,
`_render_freq`, `_render_approx`, `_render_trailer`) are unchanged — they still
return single fragments.

1. **`_render_reins`** — return `_Block(f'{prefix} {kind}', [cession strings],
   sep='and')` instead of the `' and '.join(...)` string. Empty when absent
   (return `''`).
2. **`_render_agg` / `_render_pnl`** — collect the same ordered fragment list,
   return `_Block(head, fragments)` (head = `agg NAME` / `pnl NAME …`). The pnl
   `premium -` prefix stays part of the head line.
3. **`_render_port`** — `_Block('port NAME [trailer]', [sub-agg _Blocks])`.
   Terse render of this must reproduce the **current tab-indented** form, so port
   keeps a small bespoke terse path (head line, then each unit `'\t' + terse(unit)`)
   rather than the generic single-line flatten. Spread uses the generic walker.
4. **`_render_bvagg`** — its `units` / discrete / clash / netceded branches build
   `_Block`s the same way; components become children.
5. **`_render_sev_out`, `_render_distortion`** — no sub-clause structure; return a
   plain `str` (identical in both layouts). `dbvsev` stays one line.

### Wiring `layout` through

- `spec_to_decl(spec, kind, name)` — **unchanged signature, terse-only.** It calls
  the kind renderer to get the node, then `_render_terse(node)`. `to_agg` and the
  round-trip tests keep their exact current output.
- Add an internal `_spec_to_node(spec, kind, name)` (the renderer dispatch that
  returns the `_Block`/str) so both `spec_to_decl` (terse) and `format_program`
  (either layout) share one dispatch.
- `format_program(spec_or_text, *, fmt='text', layout='spread', width=None)`:
  1. Build node(s): from a spec/tuple directly; from a string, parse each
     statement (`_render_statement` returns a node now, with the same
     parse-failure verbatim fallback wrapped as a plain `str`).
  2. Render each node with `_render_terse` or `_render_spread` per `layout`.
  3. Join top-level statements with `'\n\n'`.
  4. `_colorize(text, fmt)` as today — unchanged; it lexes whatever text it gets.
- Validate `layout` ∈ {`'spread'`, `'terse'`} with a `ValueError` mirroring the
  `fmt` guard.

### Colorization interaction

`_colorize` lexes the full string with `AggLexer` through a Pygments formatter.
Indentation and newlines are plain `Text` tokens to the lexer, and the HTML/LaTeX
formatters emit inside a `<pre>` / verbatim block, so multiline structure
survives. **Verify** (don't assume) that `format_program(prog, fmt='html')` (now
spread by default) still contains `<span>`s and preserves the newlines — add to
the smoke tests below.

## Files touched

- `src/aggregate/decl_writer.py` — the `_Block` node, two render walkers,
  `_spec_to_node`, refactor of the four structural renderers, new `layout` param
  and `'\n\n'` statement join. Update the module docstring (Layer 3) and the
  `format_program` / `spec_to_decl` docstrings to document `layout` and state that
  `spec_to_decl` is terse-only.
- `src/aggregate/_aggregate.py` (~3650–3666) — `pprogram` / `pprogram_html`
  docstrings note the spread default; **no signature change** (defaults flow
  through). Same for `src/aggregate/_portfolio.py` (~2116–2126).
- `tests/test_decl_unparser.py` — update the two layout-sensitive smoke tests
  (below) and add spread coverage.
- `CHANGELOG.md`, `pyproject.toml` version bump, `dev/TODO.md`, move this plan to
  `dev/done/` at close.

## Testing

The round-trip suite (`test_roundtrip`) goes through `spec_to_decl` → **unchanged
and must stay green** (terse is the byte-for-byte regression anchor; idempotence +
fidelity unaffected).

Smoke tests to update / add in `tests/test_decl_unparser.py`:

- `test_format_text_is_plain` — `_SMOKE` now renders multiline; assert it starts
  with `agg X` and that the severity clause is on its **own indented line**
  (`'\n  sev '` substring). Add a `layout='terse'` variant asserting the old
  single-line string is recoverable.
- `test_format_accepts_spec_tuple` — currently asserts the single-line
  `'agg Y dfreq [1 2 3 4 5 6] dsev [1]'`. Split: keep that exact string under
  `layout='terse'`; add a spread assertion (head `agg Y`, two indented clause
  lines).
- `test_format_port_is_multiline` — under spread, assert `port P`, `  agg A`,
  `  agg B` (two-space, not tab); add a `layout='terse'` case asserting the tab
  form still holds.
- New `test_format_reins_cessions_indent` — an agg with a two-cession occurrence
  clause: assert `occurrence net of` on its own line, first cession line ends
  with ` and`, cessions indented one level past the clause keyword.
- New `test_format_html_spread_preserves_newlines` — `fmt='html'` default-spread
  output contains `<span` **and** newlines between clause lines.
- `test_format_bad_layout_raises` — `layout='zigzag'` → `ValueError`.

Run: `uv run pytest tests/test_decl_unparser.py` (set `UV_LINK_MODE=copy`).

## Docs

`format_program` / `pprogram` appear in ~10 `.rst` user-guide pages
(`docs/2_user_guides/...`) that print the result. They will re-render multiline
once the default flips — desirable, but a visible change. Per CLAUDE.md, do **not**
build the doc tree in the loop; grep confirms no `.rst` asserts a *terse* string,
so no `.rst` edits are required — just note in the PR/commit that the printed
DecL examples now render spread and the author should rebuild docs out-of-loop.

## Open / watch items

- **Empty-head blocks.** A `_Block` with an empty `head` must not emit a blank
  line in spread (guarded above) — shouldn't occur with current renderers, but
  the guard makes it safe.
- **pnl head line.** Keep `pnl NAME <premium> premium -` intact as the head so the
  affine wrapper re-parses; the loss-head fragment (`claims`/`lr`/`loss`) is the
  first child clause.
- **`width`** stays reserved/ignored; spread is structural, width would later wrap
  individual long lines (e.g. a long `dsev [...]`). Out of scope here.

---

# Punch-up 2 — `help` / `agg_help` `fmt` rendering + `output`→`values` rename

**Goal.** `.help()` currently renders **only** through IPython
(`display(Markdown(...))`), so it looks right in Jupyter but prints ugly object
reprs in a plain terminal/REPL. Add an `fmt` axis that picks the render target,
defaulting to `auto` (Jupyter → colorized ANSI; terminal → plain text). At the
same time, rename the existing `output` param to `values` to kill the
"output-ish" overlap with `fmt`.

**Decisions (confirmed with author):**

- **New axis `fmt`**, a *third* orthogonal axis next to `lod` (docstring detail)
  and `values` (value detail, renamed below). Values:
  `auto` (default) / `text` / `ansi` / `html`.
- **Rename `output` → `values`** (the existing a98 param: `none` / `short` /
  `all`, governing how much of each name's value / call-result to show). Breaking,
  but pre-v1.0 alpha and the whole point is to remove the `output`/`fmt`
  confusion. `values='none'` reads as "names + docstrings only". NOTE: the
  unrelated `Aggregate.approximate(output='scipy')` / `Portfolio.approximate`
  keep their own `output` — different meaning (which *representation* to return),
  different method; out of scope and now no longer overloaded against help.
- **Spelling is `text`** (not `txt`) — matches the already-shipped
  `format_program(fmt='text')`; one canonical spelling library-wide.
- **`auto` resolves to `ansi` in Jupyter, `text` otherwise.** The author prefers
  ANSI colors + consistent monospace font over HTML *even in JupyterLab*, so
  `auto` deliberately does **not** pick `html`. `html` is reachable only by
  asking for it explicitly. An explicit `fmt=` always wins over `auto`.

## Current surface → new surface

- Worker: `agg_help(self, regex, lod='short', output='short')` — `utilities.py:521`.
- Four thin wrappers, all `help(self, regex, lod='short', output='short')`:
  `Aggregate` (`_aggregate.py:2041`), `Portfolio` (`_portfolio.py:284`),
  `Underwriter` (`underwriter.py:1463`), bivariate (`bivariate.py:1829`).

All five rename `output` → `values` and gain `fmt='auto'` (last arg); the wrappers
pass both straight through to `agg_help`.

```
agg_help(self, regex, lod='short', values='short', fmt='auto')
help(self, regex, lod='short', values='short', fmt='auto')   # ×4 wrappers
```

## Design

The worker walks `dir(self)` and, per matched name, emits three pieces: a
**header** (`Callable/Attribute: name(sig)`), an optional **docstring** (governed
by `lod`), and an optional **value / call-result** (governed by `values`). Today
each piece is pushed through `display(Markdown(...))` / `display(value)`. The
change is purely the `output`→`values` rename plus *how those three pieces are
emitted*; the `dir` walk, classification, `lod` truncation, and the
`values` `.head(5)` logic are untouched.

### Resolve `auto` once, up front

```python
def _help_target(fmt):
    """Resolve the help render target: 'text' | 'ansi' | 'html'."""
    if fmt not in ('auto', 'text', 'ansi', 'html'):
        raise ValueError(
            f"fmt must be 'auto', 'text', 'ansi', or 'html'; got {fmt!r}")
    if fmt != 'auto':
        return fmt
    return 'ansi' if _in_jupyter() else 'text'
```

Jupyter detection must be **cheap and not force an IPython import** (the ~1s cost
the module already guards against). If IPython was never imported, we are
certainly not in a notebook, so probe `sys.modules` rather than importing:

```python
def _in_jupyter():
    """True iff running under a Jupyter (ZMQ) kernel, without importing IPython."""
    mod = sys.modules.get('IPython')
    if mod is None:
        return False
    shell = mod.get_ipython()
    # ZMQInteractiveShell = notebook / lab / qtconsole (all render ANSI);
    # TerminalInteractiveShell / None = not Jupyter.
    return shell is not None and type(shell).__name__ == 'ZMQInteractiveShell'
```

(ANSI escapes render as color in JupyterLab stream output and in a color
terminal, so `ansi` is correct for the Jupyter case the author wants.)

### Emit per target

Three render branches, each consuming the same (header, doc, value) pieces:

- **`text`** — `print()` plain. Header as a plain line (e.g. `name(sig)` with a
  light rule, no Markdown `###`); docstring verbatim; value via `print(value)`
  (a `DataFrame`/`Series` already prints as a clean text table; `values='short'`
  still heads it to 5 rows). No IPython import on this path.
- **`ansi`** — same as `text` but the **header line is colorized** (ANSI bold +
  one accent color for the name, a dim tag for `Callable`/`Attribute`); docstring
  and values stay plain (no Markdown→ANSI engine — keep it dependency-free with a
  tiny set of escape constants). This is the "colors + consistent font" the
  author wants. No IPython import on this path.
- **`html`** — today's exact behavior: lazy `from IPython.display import Markdown,
  display`, `display(Markdown(...))` for header/doc, `display(value)` for values.

Implementation shape: factor the per-match emit so the branch on target is in one
place (e.g. a small local `emit_header / emit_doc / emit_value` trio selected
once before the loop), so the three pieces are not re-branched at every call site.
Keep the function side-effecting (prints/displays, returns `None`) — same contract
as today.

### Naming / collision check (per CLAUDE.md)

`fmt` does not collide with any attribute or method on `Aggregate`, `Portfolio`,
`Underwriter`, or the bivariate class (verified: the only `fmt` in the package is
the `format_program` free-function kwarg). It sits cleanly beside `lod` /
`values`. The `fmt` values reuse the established `text` / `html` / `ansi`
vocabulary; `values` is a fresh name (no existing `values` attribute on these
classes — the only `values` in the package is `pandas`/`dict` `.values`, never a
kwarg here).

## Files touched

- `src/aggregate/utilities.py` — `agg_help`: rename `output`→`values`, add
  `fmt='auto'`, the `_help_target` / `_in_jupyter` helpers, the three emit
  branches; move the IPython import into the `html` branch; rewrite the docstring
  (`values` semantics + `fmt` as the third orthogonal axis). Add `import sys` if
  not already present.
- `src/aggregate/_aggregate.py`, `_portfolio.py`, `underwriter.py`,
  `bivariate.py` — rename `output`→`values` and add `fmt='auto'` on each `help`
  wrapper, pass both through, update the docstring.
- `tests/test_bivariate.py` (~344–353) — the only existing caller of help's
  `output=`: rename to `values=`, and the bad-value case (`output='lots'`)
  becomes `values='lots'`.
- `CHANGELOG.md` (call out the **breaking** `output`→`values` rename) +
  `pyproject.toml` version bump (separate bump from punch-up 1); `dev/TODO.md` if
  tracked.

## Testing

`.help` is a side-effecting display helper, so test via `capsys` and by
monkeypatching the Jupyter probe:

- `test_help_text_is_plain` — `a.help('mean', fmt='text')`; captured stdout
  contains the matched name and a docstring fragment, and has **no** `\x1b[`
  escape and no literal `<span`/Markdown object repr.
- `test_help_values_renamed` — `a.help('mean', values='none')` works and a
  stray `output='none'` now raises `TypeError` (the rename is real, not aliased).
- `test_help_ansi_has_escape` — `fmt='ansi'`; captured stdout contains `\x1b[`.
- `test_help_auto_resolves` — monkeypatch `_in_jupyter` → `True` then `False`;
  assert `auto` produces ANSI escapes in the first case and plain text in the
  second (i.e. `auto` delegates to the resolver, explicit wins).
- `test_help_bad_fmt_raises` — `fmt='xml'` → `ValueError`.
- `test_help_html_runs` — `fmt='html'` executes without error (skip if IPython
  is unavailable in the test env).

Run: `uv run pytest tests/` (set `UV_LINK_MODE=copy`); target the new help tests
specifically while iterating.

## Open / watch items

- **Three orthogonal axes, documented as such.** The rename resolves the
  `output`/`fmt` overlap; make the orthogonality explicit in the worker docstring:
  `lod ⊥ values ⊥ fmt` — `lod` = docstring detail, `values` = how much of the
  value/call-result, `fmt` = render target. `lod='terse', values='none'` is a
  bare name listing.
- **ANSI in redirected output.** ANSI codes leak as escapes if `help(fmt='ansi')`
  is piped to a file. That is acceptable: `ansi` is only auto-selected inside
  Jupyter (which renders it); a terminal user who redirects can pass `fmt='text'`.
  No TTY-sniffing — keep it explicit.
```

