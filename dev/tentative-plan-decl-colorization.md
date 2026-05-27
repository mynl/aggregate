# Plan — unified DecL colorization

**Status:** DEFERRED 2026-05-27. Plan is complete and ready to execute,
but parked pending a concrete pull. The headline payoff most readers
assume — colorized errors in JupyterLab cell tracebacks — does not
happen automatically: IPython's traceback formatter calls
``__str__`` on the exception, not ``_repr_html_``. The real wins are
(a) Sphinx docs DecL code blocks picking up the project palette, and
(b) ANSI-colored ``e.report.render()`` in a TTY. Both are aesthetic
rather than load-bearing. Resurrect when the docs identity becomes
a stated priority or a user explicitly asks for the Jupyter HTML box.

**Drift to fix on resurrection:** ``ErrorReport.render()`` was rewritten
in the parser-errors-promotion work to use a single concatenated
message line (message + "Did you mean…" + "Expected…"). The
``render(color=...)`` sketch in §"Renderer 2 — ANSI terminal" still
shows the old multi-``lines.append`` form. Color the three
sub-strings individually before concatenating, then assemble the
``lines`` list — do not restructure the tight layout.

**Depends on:** parser_errors promotion (so there's a Jupyter HTML
consumer to motivate the HTML renderer; not strictly required).
**Unblocks:** consistent DecL visual identity across docs (Sphinx
code blocks), Jupyter (`_repr_html_` for ErrorReport), terminals
(ANSI in `e.report.render()`), and matplotlib (existing).

## Why

Three current/potential DecL renderers exist or will exist:

1. **Sphinx code blocks** — driven by Pygments via
   `aggregate.decl_pygments.AggLexer`. Currently classifies tokens
   but uses whatever Pygments style the doc theme picks. Colors are
   inconsistent with the rest of the project identity.

2. **Terminal output of `ErrorReport.render()`** — plain text today.
   In a TTY we could colorize keywords, the caret, the "did you
   mean" line.

3. **Jupyter HTML output of `ErrorReport._repr_html_()`** — flagged
   v1.1 in the parser_errors plan. Needs a styled HTML template
   when it lands.

4. **matplotlib plots** — already styled via `aggregate.style` from
   `data/aggregate.mplstyle`.

Each currently picks its own colors (or none). Result: code in the
docs is a different shade of blue than... whatever the next consumer
becomes. The deleted web SPA had a coherent palette
(steelblue keyword, copper number, forest green string, grey italic
comment, mynl red code-value, violet atom for distortion / sev
names) deliberately rhymed with the matplotlib palette. That work
shouldn't go to waste.

**Goal: one Python-side palette dict, four renderers consuming it.**

## Tree

`aggregate.style` is currently a single module (`src/aggregate/style.py`,
~150 lines, matplotlib-focused). Promote it to a package:

```
src/aggregate/
    style/
        __init__.py        ← re-exports use, context, rc_params (back-compat)
        mpl.py             ← current style.py guts: use, context, rc_params
        palette.py         ← NEW: the canonical color dict (single source of truth)
        pygments_style.py  ← NEW: Pygments Style class consuming palette
        ansi.py            ← NEW: terminal ANSI renderer
        html.py            ← NEW: HTML span renderer
    data/
        aggregate.mplstyle  ← unchanged
    decl_pygments.py        ← unchanged (the lexer)
```

`from aggregate.style import use, context` keeps working — `__init__.py`
re-exports. No breakage for existing notebooks / `docs/conf.py`.

## The palette

`aggregate/style/palette.py`:

```python
"""The aggregate visual identity, in one dict.

Each renderer (matplotlib, Pygments, ANSI, HTML) reads from
:data:`PALETTE` and translates into its native form. To change a
color across all surfaces, edit one entry here.
"""

PALETTE = {
    # Semantic token colors -- the DecL identity.
    'keyword':     '#1f4e8a',   # steelblue, bold     -- agg, sev, mixed, ...
    'atom':        '#6a3eaf',   # violet              -- distortion / sev names
    'builtin':     '#1f4e8a',   # same as keyword     -- sev.X, agg.X
    'number':      '#b87333',   # copper              -- 100, 1.5, 1e-3
    'string':      '#2e7d32',   # forest green        -- 'x', note{...}
    'comment':     '#8a8f99',   # grey, italic        -- # to EOL
    'operator':    '#555555',   # dark grey           -- + - * /
    'punctuation': '#1a1a1a',   # near-black          -- [ ] ( ) ,
    'heading':     '#a81313',   # mynl red            -- !, <DISTRIBUTION>
    # Diagnostic colors -- caret, suggestion, error frame.
    'caret':       '#a81313',   # mynl red, bold
    'suggestion':  '#2e7d32',   # forest -- "did you mean..."
    'error_frame': '#d4a017',   # warm amber border for error blocks
    # Background accents (light theme).
    'soft':        '#f0f6fb',   # aliceblue-ish, table hover
    'rule':        '#1a1a1a',   # table rules, near-black
}

# Bold/italic flags per token class. Pygments and HTML use these;
# ANSI applies SGR codes 1 and 3.
WEIGHTS = {
    'keyword': {'bold': True},
    'caret':   {'bold': True},
    'comment': {'italic': True},
    'suggestion': {'italic': True},
}

# Dark-mode variant (RTD dark theme support). Future work.
DARK_PALETTE = None  # filled in when we add a dark theme
```

Notes:

- These are the colors from the deleted `web/src/styles/cm6.css`,
  carried over so the visual identity is preserved.
- All hex codes; renderers downsample to xterm-256 / xterm-16 as
  needed (`ansi.py` handles that).
- Light theme only for v0. Dark theme deferred — needs separate
  palette tuning, not a 1:1 inversion.

## Renderer 1 — Pygments style

`aggregate/style/pygments_style.py`:

```python
"""Pygments Style class for DecL code blocks.

Registered as a Pygments style entry point; Sphinx picks it up by
name when `pygments_style = 'aggregate'` is set in conf.py.
"""

from pygments.style import Style
from pygments.token import (
    Comment, Generic, Keyword, Name, Number, Operator, Punctuation,
    Text, String,
)
from .palette import PALETTE, WEIGHTS


def _styled(color, weights=None):
    parts = [color]
    if weights and weights.get('bold'):   parts.append('bold')
    if weights and weights.get('italic'): parts.append('italic')
    return ' '.join(parts)


class AggregateStyle(Style):
    """Pygments style mapping AggLexer tokens onto aggregate's palette."""

    background_color = '#fdfdfd'
    default_style    = ''

    styles = {
        Comment:          _styled(PALETTE['comment'],  WEIGHTS.get('comment')),
        Keyword:          _styled(PALETTE['keyword'],  WEIGHTS.get('keyword')),
        Name:             PALETTE['punctuation'],
        Name.Function:    PALETTE['keyword'],
        Name.Class:       PALETTE['atom'],
        Name.Namespace:   PALETTE['atom'],
        Name.Builtin:     PALETTE['builtin'],
        Name.Label:       PALETTE['atom'],
        Name.Type:        PALETTE['string'],
        Number:           PALETTE['number'],
        Number.Integer:   PALETTE['number'],
        Operator:         PALETTE['operator'],
        Operator.Word:    _styled(PALETTE['keyword'], WEIGHTS.get('keyword')),
        Punctuation:      PALETTE['punctuation'],
        String:           PALETTE['string'],
        Generic.Heading:  PALETTE['heading'],
        Text:             '',
    }
```

Register as a Pygments style entry point in `pyproject.toml`:

```toml
[project.entry-points."pygments.styles"]
aggregate = "aggregate.style.pygments_style:AggregateStyle"
```

`docs/conf.py`: set `pygments_style = "aggregate"`. Sphinx then
loads it via the entry point, no import needed.

The pre-existing `pygments.lexers` entry point (`AggLexer`) keeps
working — lexer and style are independent.

## Renderer 2 — ANSI terminal

`aggregate/style/ansi.py`:

```python
"""ANSI 256-color SGR rendering for DecL text and ErrorReport pieces.

Used by ErrorReport.render() when stdout is a TTY. Falls back to
plain text otherwise.
"""

import os
import re
import sys
from .palette import PALETTE


def supports_color(stream=sys.stdout) -> bool:
    """True if `stream` is a TTY and the env opts in to color."""
    if os.environ.get('NO_COLOR'): return False
    if os.environ.get('FORCE_COLOR'): return True
    return hasattr(stream, 'isatty') and stream.isatty()


def _hex_to_xterm256(hex_color: str) -> int:
    """Map a #rrggbb hex to the nearest xterm-256 palette index."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    # Standard 6x6x6 cube quantization.
    return 16 + 36 * (r * 5 // 255) + 6 * (g * 5 // 255) + (b * 5 // 255)


def sgr(text: str, palette_key: str, *, bold=False, italic=False) -> str:
    """Wrap `text` in ANSI SGR codes for the named palette color."""
    color = _hex_to_xterm256(PALETTE[palette_key])
    parts = [f'\x1b[38;5;{color}m']
    if bold: parts.append('\x1b[1m')
    if italic: parts.append('\x1b[3m')
    parts.append(text)
    parts.append('\x1b[0m')
    return ''.join(parts)


# Token-stream renderer for full DecL source (used by future CLI
# pretty-printer; not used by ErrorReport).
def colorize_decl(text: str) -> str:
    """Render DecL `text` with ANSI color codes via the AggLexer."""
    from pygments import highlight
    from pygments.formatters import Terminal256Formatter
    from aggregate.decl_pygments import AggLexer
    from .pygments_style import AggregateStyle
    return highlight(text, AggLexer(), Terminal256Formatter(style=AggregateStyle))
```

`ErrorReport.render()` upgrade (in `parser_errors.py`):

```python
def render(self, color: bool = None) -> str:
    """Multi-line text rendering.

    Parameters
    ----------
    color : bool | None
        Force-enable / force-disable ANSI coloring. ``None`` (default)
        auto-detects via ``aggregate.style.ansi.supports_color``.
    """
    from aggregate.style.ansi import sgr, supports_color
    use_color = supports_color() if color is None else color
    def c(text, key, **kw):
        return sgr(text, key, **kw) if use_color else text
    lines = [
        f"DecL parse error at line {self.line}, column {self.column}:",
        "",
        f"  {self.source_line}",
        f"  {c(self.caret, 'caret', bold=True)}",
        "",
        self.message,
    ]
    if self.suggestions:
        lines.append("")
        lines.append(c(f"Did you mean: {', '.join(self.suggestions)}?",
                       'suggestion', italic=True))
    # ... expected line unchanged
    return "\n".join(lines)
```

`supports_color` honors the de-facto standards: `NO_COLOR`,
`FORCE_COLOR`, TTY detection. Pipes get plain text, terminals get
color. Jupyter terminals get color (their stdout is a TTY-shaped
pty); the IPython display layer gets HTML, not text — see next.

## Renderer 3 — HTML

`aggregate/style/html.py`:

```python
"""Inline HTML span rendering for use in Jupyter `_repr_html_`."""

from html import escape
from .palette import PALETTE, WEIGHTS


def span(text: str, palette_key: str) -> str:
    """Return a `<span style="...">text</span>` styled per palette."""
    style = [f'color: {PALETTE[palette_key]}']
    w = WEIGHTS.get(palette_key, {})
    if w.get('bold'):   style.append('font-weight: bold')
    if w.get('italic'): style.append('font-style: italic')
    return f'<span style="{"; ".join(style)}">{escape(text)}</span>'


def colorize_decl_html(text: str) -> str:
    """Full DecL → HTML colorization via AggLexer + Pygments HtmlFormatter."""
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from aggregate.decl_pygments import AggLexer
    from .pygments_style import AggregateStyle
    return highlight(text, AggLexer(),
                     HtmlFormatter(style=AggregateStyle, nowrap=True))
```

`ErrorReport._repr_html_()` (added in parser_errors plan v1.1 step,
or here — your call):

```python
def _repr_html_(self):
    from aggregate.style.html import span, colorize_decl_html
    parts = [
        '<div style="border-left: 4px solid ', PALETTE['error_frame'],
        '; padding: 0.5em 0.75em; background: #fffbea; font-family: serif;">',
        f'<div style="color: #555; font-size: 0.9em;">'
        f'DecL parse error at line {self.line}, column {self.column}.</div>',
        '<pre style="margin: 0.5em 0; font-family: monospace;">',
        colorize_decl_html(self.source_line),
        '<br>',
        span(self.caret, 'caret'),
        '</pre>',
        f'<div>{escape(self.message)}</div>',
    ]
    if self.suggestions:
        parts.append(
            f'<div>{span("Did you mean: " + ", ".join(self.suggestions) + "?", "suggestion")}</div>'
        )
    parts.append('</div>')
    return ''.join(parts)
```

Two design choices here worth noting:

a) **Inline styles, not a `<style>` block or class names.** Jupyter
notebooks sometimes strip `<style>` tags from non-trusted output;
inline `style=` attributes survive. Tradeoff: the HTML is repetitive
and chunky. Acceptable because parse errors are rare and the HTML
isn't user-facing source.

b) **Pygments handles the source-line colorization.** We don't
hand-roll token coloring; we delegate to the lexer we already have.
The caret and suggestion are hand-built spans because they're
diagnostic, not DecL source.

## Renderer 4 — matplotlib (existing)

`aggregate/style/mpl.py` is the current `style.py` moved into the
package. No functional change. The matplotlib palette comes from
`data/aggregate.mplstyle`, not from `palette.py` — they evolved
separately. **Open question: unify them?**

Pros: one place to change the project's accent color.
Cons: matplotlib styles cover dozens of rcParams beyond just colors
(font sizes, figure dpi, grid behavior). Mixing those into
`palette.py` muddies the simple "what color is X" semantics.

**Recommendation: do not unify in v0.** Leave the mplstyle file
alone. Add a docstring note in `palette.py` linking out to it for
plot-line colors. Revisit if we add a second matplotlib style
(dark, print).

## Back-compat surface

- `from aggregate.style import use, context, rc_params` keeps working
  (re-exported from the package `__init__`).
- `import aggregate.style; aggregate.style.use()` keeps working.
- The Pygments lexer entry point is unchanged.
- New entry point added: `[project.entry-points."pygments.styles"]`.
  Style picks up automatically when installed.
- `ErrorReport.render(color=None)` keeps the signature compatible
  with `render()` (positional/keyword optional). Existing callers
  in `tests/test_parser_errors.py` keep passing.

## Tests

New file: `tests/test_style_palette.py` covering:

- Palette dict has all required keys (smoke).
- Pygments style class instantiates and serves all token types
  encountered by `AggLexer` on `test_suite.agg`.
- `sgr(text, key)` produces the expected SGR escape sequence for
  a known palette entry.
- `supports_color` honors `NO_COLOR=1` and `FORCE_COLOR=1` env vars.
- `colorize_decl_html('agg X dfreq [3]')` round-trips through
  Pygments without raising and contains the expected color hex.
- `ErrorReport.render(color=False)` is byte-identical to the
  pre-promotion output (regression guard).
- `ErrorReport.render(color=True)` includes `\x1b[` somewhere.

~8 tests. None of them slow.

Plus one Sphinx-side check: a `docs/conf.py` change to
`pygments_style = "aggregate"`, then verify a small `.. code-block::
agg` snippet in an existing docs page renders with the new colors.
This is a manual verification (the doc build is slow, per CLAUDE.md).

## Docs

A short page at `docs/4_agg_language_reference/`: "DecL code
appearance" — shows the palette swatches, links the matplotlib
style, mentions the `aggregate` Pygments style name for users who
embed DecL in their own Sphinx projects.

## Open knobs

1. **Color depth for ANSI?** Plan ships xterm-256 quantization. An
   xterm-16 fallback for very limited terminals is easy to add but
   probably unused on developer hardware. Skip.
2. **Dark-theme palette?** Out of scope. RTD's dark mode is unsigned;
   doc theme handling would need its own design.
3. **CSS class names instead of inline styles in HTML output?**
   Skipped per the trust-in-Jupyter-output argument above. Could
   add a `colorize_decl_html_cls()` variant later.
4. **Single matplotlib palette unification?** Recommended **no** for
   v0 — see above.
5. **CLI tool `agg-pretty <file.agg>` that ANSI-colorizes a DecL
   file?** Cute, three lines around `colorize_decl()`. Worth adding
   if the user wants to demo the colors at the terminal. Otherwise
   the colors only appear in error output, which is OK.

## Out of scope

- LSP / editor protocol support. The deleted web editor is gone;
  if you ever want a VS Code or Neovim plugin, the palette here
  feeds it but the plugin itself is separate work.
- Localization of token names.
- Per-user color customization. The palette is hardcoded; users
  who want different colors install a different Pygments style.

## Verification

1. `uv run pytest tests/test_style_palette.py -q` — green.
2. `uv run pytest -q` — full suite still green.
3. `uv run python -c "from aggregate.style.ansi import colorize_decl;
   print(colorize_decl('agg Dice dfreq [3] dsev [1:6]'))"` — terminal
   shows colored output.
4. `uv run python -c "from aggregate import build;
   try: build('agg X 100 claims sev lognorm 100 cv 2 mixd poisson 0.5')
   except ValueError as e: print(e.report.render())"` — in a TTY,
   caret is red, suggestion green; piped to a file, plain text.
5. Manual: rebuild docs locally, confirm a `.. code-block:: agg`
   snippet uses the new colors. (User runs this — slow.)
6. Manual (after `_repr_html_` ships): trigger a parse error in a
   Jupyter cell, surface `e.report`, see the styled HTML.

## Recovery / rollback

- Style package: `git revert` the move; `__init__.py` re-exports
  preserve all old imports either way.
- Pygments entry point: removing from `pyproject.toml` reverts
  Sphinx to its previous style.
- `ErrorReport.render(color=...)` is additive; defaulting `color=False`
  restores pre-coloration output exactly.

## File-by-file checklist

1. Promote `src/aggregate/style.py` → `src/aggregate/style/`
   package. Move the current contents into `mpl.py`. `__init__.py`
   re-exports `use`, `context`, `rc_params`.
2. New file: `src/aggregate/style/palette.py` — the canonical dict.
3. New file: `src/aggregate/style/pygments_style.py` —
   `AggregateStyle`.
4. New file: `src/aggregate/style/ansi.py` — `sgr`,
   `supports_color`, `colorize_decl`.
5. New file: `src/aggregate/style/html.py` — `span`,
   `colorize_decl_html`.
6. `pyproject.toml`: add `[project.entry-points."pygments.styles"]`
   block; update `[tool.setuptools] packages` to include
   `aggregate.style`.
7. `src/aggregate/parser_errors.py`: upgrade `ErrorReport.render`
   to optional `color` flag; add `_repr_html_` if doing v1.1 step
   here (otherwise leave for the parser_errors v1.1 step).
8. `docs/conf.py`: set `pygments_style = "aggregate"`.
9. `docs/4_agg_language_reference/`: new "DecL code appearance"
   page with palette swatches.
10. `tests/test_style_palette.py`: ~8 tests.
11. `README.rst`: one bullet under the current 1.0.0a16 work.
