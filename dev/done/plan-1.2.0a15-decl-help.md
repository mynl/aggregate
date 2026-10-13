# Plan 1.2.0a15: DecL help pages, an anatomy and a one-page card, shipped with the library

Status: **executed 2026-10-13 at 1.2.0a15**, retired to `dev/done/` by the author the same day.
See "Execution log" at the end. Drafted
2026-10-13, revised the same day after the author's review, cross-reviewed with the api's
`V:\dev\aggregate-api\dev\plan-1.7.0-decl-help.md`. One bump,
**[DecL-Help-Pages]**. The help rolls into the 1.2.0 release: it lands as 1.2.0a15 first, and
the 1.2.0 release note is written after it (the draft roll-up was deleted 2026-10-13 for that
reason).

## Goal

Ship one page of DecL help, in two forms, as part of the library:

* **anatomy** ("DecL at a glance"): one `agg` statement taken apart slot by slot, a portfolio
  and a P&L as two ways to use it, and a table of alternative forms for each slot. The aLL
  opens it from a `?` in the program box, as a popover in front of the output.
* **card** ("DecL on one page"): a ten-box reference card in the PDF cheat-sheet style that
  prints to exactly one Letter landscape page. The aLL opens it from the hamburger menu.

Both are static HTML fragments that render identically in the Loss Lab and in a Jupyter cell.
The library owns them because it owns the language: the help then follows the grammar's version,
and a test builds every example in it, so the help cannot drift the way
`cheat-sheets/DecL_Cheat_Sheet.tex` did (it kept teaching `so` and `of` after they were
retired at 1.0.0a249, until they were removed from the sheet on 2026-10-13, outside this plan).
The api serves what the library returns and adds the Lab-only behavior (Load buttons, where the
pages open). That half is the api's plan.

## How we got here

Design ran in three rounds on 2026-10-13. The mockups, the comparison, and the round notes are in
`X:\decl-help\` (`index.html`; round 1 in `round-1\`). The author chose concept B for the `?`
popover and concept C for the menu card, parked concept A (a four-step walkthrough), and then
hand-edited both pages. **The hand-edited `concept-b-anatomy.html` and `concept-c-card.html` are
the content source for this plan.** Everything between the `THE FRAGMENT` and `END FRAGMENT`
comments in each file is what ships, after the mechanical changes listed under "From mockup to
source" below. Everything outside those comments (the fake Loss Lab, the fake JupyterLab cell,
`mock.js`, `mock.css`) is mock chrome and does not ship.

Decisions already made, not to be reopened here:

* HTML fragments, not markdown. The SPA has no markdown renderer, and adding one for one page is
  not worth it.
* The library writes the content; the api serves it at `GET /v1/decl/help`; Jupyter shows it.
* Ordering of the anatomy: a plain aggregate first, then reinsurance in place, then portfolio
  and P&L. Ordering of the alternative forms: most used in `library.agg` and
  `reproductions.agg` first (lognorm by mean and cv, then gamma; Poisson, mixed gamma, fixed).
* Alternatives are never written as `(a|b|c)`. In DecL, `|` is whitespace to the lexer and
  parentheses are arithmetic, so a reader who copies that form gets a confusing parse error. The
  pages show alternatives as a comment on a runnable line (`38000 premium  # or inherit premium
  or derive premium`) or as separate entries.

## Current behavior

* **No help exists in the library.** `HelpMixin.help` (`_help.py`) lists an object's attributes;
  it is unrelated. The DecL reference is `docs/4_dec_Language_Reference.rst` (grammar listing,
  pre-processing, parse errors) and the four-page `cheat-sheets/DecL_Cheat_Sheet.tex`.
* **The lexer exists.** `decl_pygments.AggLexer` is a hand mirror of `decl.lark`, guarded by
  `tests/test_grammar_sync.py` (every reserved word, and zero `Token.Error` over the shipped
  corpora). Its token types: `Keyword` (reserved words), `Name.Function` (frequency families,
  zero-shape severities, mixing laws), `Name.Class` and `Name.Namespace` (one- and two-shape
  severities), `Name.Label` (`dfreq`, `dsev`, `dbvsev`, `dwait`, `nttagg`), `Name.Builtin`
  (`agg.X`, `sev.X`, `port.X`, `dist.X`), `Number`, `String` (quoted labels), `Comment.Single`
  (`#`, `//`), and inside trailers `Comment.Preproc`, `Comment`, `Name.Tag`, `Name.Attribute`.
* **The existing colorizer does not fit.** `decl_writer._colorize(text, 'html')` runs Pygments'
  `HtmlFormatter(style='friendly')`, whose output uses Pygments' own short classes (`k`, `m`,
  `s`) and needs Pygments' stylesheet. A fragment must carry its own classes and colors.
* **Package data** (`pyproject.toml`, `[tool.setuptools.package-data]`): `*.lark`, `agg/*.agg`,
  `data/*.mplstyle`, `data/*.toml`, `formats/*.yaml`. Pygments is already a core dependency.
* **Top-level exports**: `__init__.py` re-exports each submodule's `__all__` with
  `from .module import *`; `__version__` comes from `importlib.metadata`.
* **The api** pins `aggregate>=1.0.1,<2` from PyPI, which has no help. Its plan feature-detects
  the help, so neither side needs a floor change to land.

## The change

### Content files

Two new files, `src/aggregate/data/decl-help/anatomy.html` and
`src/aggregate/data/decl-help/card.html`. Each holds exactly one fragment:

```html
<section class="decl-help dh-anatomy" data-decl-version="">
<style> ... every rule scoped under .decl-help.dh-anatomy ... </style>
... body ...
</section>
```

They are the author's to edit, by hand, as HTML. The rules that keep them renderable and
testable, each one enforced by a test in `tests/test_decl_help.py`:

1. **Plain DecL inside code.** The text of every `<pre class="dh-code">` and every
   `<code class="dh">` is plain DecL: HTML entities are allowed (`&amp;`), tags are not. The
   renderer colors these; nothing else is touched.
2. **`...` means elided.** A chip such as `agg Book ...` or `sev ... !` may elide with three
   dots. The renderer leaves them as text; the lex test removes them before lexing.
3. **Loadable programs are marked, not copied.** A bare `data-load` attribute goes either on a
   `pre.dh-code` (the program is its text) or on a container whose program lines are
   `<code class="dh dh-l">` cells, joined with newlines (the anatomy table). The renderer writes
   the program into the attribute's value. The author never types a program twice.
4. **Every CSS rule is scoped** under `.decl-help.dh-anatomy` or `.decl-help.dh-card`,
   including the token colors and the rules inside `@container` blocks. Two fragments in one
   document (a notebook showing both, or the Lab opening one after the other) then cannot
   restyle each other. This answers the api plan's Q2.
5. **Sizes in `em`**, off one base size per form: `font-size: var(--dh-size, 13.5px)` on the
   anatomy, `var(--dh-size, 11px)` on the card. A host scales the whole page by setting
   `--dh-size` (for a projector, say) and nothing else changes proportion. `rem` is avoided
   because it refers to the host document's root size, which differs between the Lab and
   JupyterLab.
6. **Colors from custom properties** (`--dh-ink`, `--dh-mut`, `--dh-line`, `--dh-accent`, and the
   token colors `--dh-kw`, `--dh-num`, `--dh-str`, `--dh-com`), declared on the section, so a
   host can retheme (JupyterLab dark, for instance) without editing the fragment.
7. **Self-contained.** No `<script>`, `<link>`, `<img>`, or `url(...)`. The only links are to
   `aggregate.readthedocs.io`. The brace beside "exposure" stays an inline `<svg>`.
8. **Host hooks**: `.dh-lab` content shows only when the host adds `in-lab` to the section;
   `.dh-py` content shows everywhere else. The fragment carries both rules.
9. **Version**: `data-decl-version=""` on the section, and `{{version}}` anywhere in the text
   (the card's header), are filled by the renderer.
10. **Printing**: the card sets `print-color-adjust: exact` on itself, so its box fills survive
    printing. It carries no `@page` rule; page size is the host's business.

### From mockup to source

Mechanical changes when copying each fragment out of `X:\decl-help\`, in order:

1. Rename the form classes: `dh-b` becomes `dh-anatomy`, `dh-c` becomes `dh-card`. Concept
   letters are not names.
2. Scope every rule (rule 4), convert `px`/`rem` sizes to `em` (rule 5), add
   `print-color-adjust` to the card (rule 10).
3. Strip any token spans (the mockups color in the browser, so their source has none) and
   leave plain text in code (rule 1).
4. Set `data-decl-version=""` and put `{{version}}` in the card's `dh-meta` line.

### Content decisions

Three problems found while checking the hand-edited pages, decided by the author on 2026-10-13
and **already applied to the mockups**, so the copy in "From mockup to source" carries them:

1. **The quota share chip did not parse.** `aggregate net of 66% inf xs 0` with `cede 30%` reads
   "Unexpected 'inf'": a partial placement needs `po`, and `cede` needs a ceded-premium clause on
   the same layer. Now `aggregate net of 66% po inf xs 0 rate 100% cede 30%`.
2. **`sev 50 * invgamma 1.2` had no finite variance**, so typed with no limit it stopped with
   "cannot estimate a bucket size". Now `sev 50 * invgamma 2.5`, which builds unlimited.
3. **One term per slot on both pages.** The first slot is a *label* on both (the card's
   `# name`, `agg NAME`, `xpnl NAME`, `agg.Name`, `port.Name` and "Names start with a letter"
   became label forms, as did the anatomy footer). Exposure coverage is "limit xs deductible";
   reinsurance layers are "limit xs attachment" (the card's layers row and the anatomy's
   aggregate layers chip now say so).

All other examples build: the anatomy program with `note{main book}` and
`hints{bs=1;log2=16;}`, the portfolio, the peeled `xpnl` (margin 1,869 at 38,000 premium), the
card's four programs, `[100 200 50] claims`, `sev ... !` in its full form, `1000 exposure at 25
rate`, `picks [250 1000] [20000 4500]`, `sev 40 * lomax 2.5 + 40`, `sev 40 * expon + 10`.

### The renderer

New module `src/aggregate/decl_help.py`, re-exported at top level through its `__all__`:

```python
__all__ = ['decl_help', 'DECL_HELP_FORMS', 'DeclHelp']

DECL_HELP_FORMS = ('anatomy', 'card')

class DeclHelp(str):
    """The rendered fragment. A str, so a web server returns it as is; displays in Jupyter."""
    form: str
    def _repr_html_(self) -> str: ...      # returns str(self)
    def __repr__(self) -> str: ...         # "DeclHelp('anatomy', aggregate 1.2.0a15, 21 kB)"

def decl_help(form: str = 'anatomy') -> DeclHelp: ...
```

Names vetted on 2026-10-13 with `rg` over `src/aggregate` and `tests`: no `decl_help`,
`DeclHelp`, `DECL_HELP_FORMS` or `decl-help` exists. `decl_help` is a function, distinct from
the `help` method `HelpMixin` puts on objects. An unknown form raises `ValueError` naming the
valid ones.

`DeclHelp` is a `str` subclass so that one call serves both hosts: the api passes it straight to
an `HTMLResponse`, and Jupyter calls `_repr_html_`. The custom `__repr__` stops a plain Python
prompt printing 20 kB of HTML. The alternative, two functions (one returning `str`, one
returning a display object), doubles the surface for no gain.

`decl_help(form)` reads `data/decl-help/{form}.html` with `importlib.resources`, renders it,
and caches the result (`functools.lru_cache` on a private `_render(form)`; a library upgrade
restarts any process that holds the cache). Rendering is three passes over the source text,
each a regular expression, which is safe because the source rules above forbid tags inside
code:

1. **Fill `data-load`.** For each `pre.dh-code` carrying a bare `data-load`, the program is its
   unescaped text. For each other element carrying one, the program is its `code.dh-l` texts
   joined with `\n`. Write `data-load="<program, attribute-escaped>"`. This answers the api
   plan's Q3: the host reads the attribute and never the markup.
2. **Color.** For each `pre.dh-code` and `code.dh`, unescape the text, lex it with `AggLexer`,
   and emit escaped text with these spans:

   | Token types | Class | Matches the editor's |
   |---|---|---|
   | `Keyword`, `Keyword.Constant`, `Name.Function`, `Name.Class`, `Name.Namespace`, `Name.Label`, `Name.Builtin` | `t-kw` | keyword (blue, bold) |
   | `Number` | `t-num` | number (copper) |
   | `String`, `Comment.Preproc`, `Comment`, `Name.Tag`, `Name.Attribute` | `t-str` | string, including `note{}`, `tags{}`, `hints{}` (green) |
   | `Comment.Single` | `t-com` | comment (grey italic) |
   | everything else | no span | plain |

   The editor's `t-atom` (distortion names) has no lexer category, since `ph` and `wang` are
   ordinary names to `AggLexer`, so the fragments do not use it. Token types are matched
   exactly, not by Pygments' subtype rule, because `Comment.Single` is a subtype of `Comment`
   and the two map differently.
3. **Stamp** `data-decl-version="{aggregate.__version__}"` and replace `{{version}}`.

The colors are the aLL editor's (`aggregate-api/web/src/styles/cm6.css`), so an example in the
help looks like the same example in the program box.

### Jupyter

`from aggregate import decl_help; decl_help()` shows the anatomy; `decl_help('card')` shows the
card. Output from a cell run in the current session is trusted, so JupyterLab keeps the
`<style>` and the inline `<svg>`; the hover linking is pure CSS (`:has`), so nothing needs
script. A notebook reopened from disk is untrusted until its cells are run again, and
JupyterLab's sanitizer may then drop parts of the fragment; rerunning the cell restores it. This
is how every HTML output behaves, not something to engineer around. To be confirmed at
acceptance, not assumed.

### Tests

New `tests/test_decl_help.py`, all in the fast suite:

* `DECL_HELP_FORMS == ('anatomy', 'card')`; `decl_help()` is the anatomy; an unknown form
  raises `ValueError`; the return is a `str` and has `_repr_html_`.
* **Every program builds.** Parametrized over each `pre.dh-code` in both forms and each
  `data-load` value: `build(program)` succeeds. Seven programs today, a few seconds in all.
* **Every snippet lexes clean.** Parametrized over every `pre.dh-code` and `code.dh` text, with
  `...` removed: `AggLexer` yields no `Token.Error`. This is weaker than building, since a chip
  such as `cede 30%` is not a statement, and it is what catches a misspelled keyword or a
  retired one. It would not have caught correction 1 above, which lexes and does not parse;
  that one is caught only by building, so a chip that teaches a whole clause should be checked
  by hand when it is added.
* **Source rules hold**: no tag inside any `pre.dh-code` or `code.dh` (rule 1); every selector
  in each `<style>` starts with `.decl-help.dh-<form>` (rule 4); no `<script`, `<link`, `<img`
  or `url(` anywhere, and every `href` points at `aggregate.readthedocs.io` (rule 7).
* The rendered section carries `data-decl-version` equal to `aggregate.__version__`, and no
  `{{version}}` survives.
* `data/decl-help/*.html` is in `[tool.setuptools.package-data]`, so the files reach the wheel.

### Docs

`docs/4_dec_Language_Reference.rst` gains a short section near the top, "Help in Jupyter and the
Loss Lab", saying what `decl_help()` and `decl_help('card')` show. Docs are pending a rebuild,
per the house rule; no build in the verification loop.

### Files

* `src/aggregate/decl_help.py` (new), `src/aggregate/__init__.py` (one import line).
* `src/aggregate/data/decl-help/anatomy.html`, `src/aggregate/data/decl-help/card.html` (new).
* `pyproject.toml`: package-data gains `"data/decl-help/*.html"`; version `1.2.0a15`.
* `tests/test_decl_help.py` (new).
* `docs/4_dec_Language_Reference.rst`.
* `CHANGELOG.md`, `dev/TODO.md`, and this plan moved to `dev/done/` when the author says it is
  done.

## Answers to the api plan's questions

**Q1, entry point.** `aggregate.decl_help(form='anatomy') -> DeclHelp`, a `str`. The forms are
`aggregate.DECL_HELP_FORMS`. The api's `available_forms()` is
`list(getattr(aggregate, 'DECL_HELP_FORMS', ()))` and `fragment(form)` is
`str(aggregate.decl_help(form))`.

**Q2, colliding base rules.** Accepted in full: every rule is scoped to its form (rule 4), and a
test enforces it. The SPA's "one fragment in the DOM at a time" becomes ordinary hygiene rather
than a correctness requirement.

**Q3, Load reads an attribute.** Accepted: `data-load` carries the program verbatim (renderer
pass 1). It stays the attribute name, valued rather than bare in the rendered output.

**Q4, Load loads but does not build.** The author's ruling; the library is indifferent. The
anatomy has no text that depends on the answer.

**Q5, type size.** The fragment keeps the mockup's scale but expresses it in `em` off
`--dh-size` (rule 5). The api can set `--dh-size: 15px` on the popover if the forms table proves
small on a projector, without touching the library.

**Q6, the `?` placement.** SPA only. The fragment has no opinion about where it opens; the caret
and the accent edge are the popover's chrome, not the fragment's.

**Q7, keyword list and comment.** Agreed: separate api TODO items.

Two contract changes the api plan needs, made there alongside this draft: the form classes are
`dh-anatomy` and `dh-card` (not `dh-b` and `dh-c`), and the token classes are `t-kw`, `t-num`,
`t-str`, `t-com` (no `t-atom`).

## Decisions

Made by the author on 2026-10-13:

1. **Release.** The help rolls into 1.2.0, as 1.2.0a15. The draft 1.2.0 roll-up was deleted from
   the CHANGELOG and is written at release, after this lands. The api's disabled-`?` text reads
   "DecL help arrives with aggregate 1.2.0".
2. **Content.** The three decisions under "Content decisions", applied to the mockups.

Still open:

3. **The api's Q4** (Load builds or not). The library is indifferent.

## Acceptance

* `uv run pytest tests/test_decl_help.py`, then the fast suite (`uv run pytest`), then the full
  suite at the bump (`uv run pytest -m 'slow or not slow'`).
* `uv build --wheel` and the wheel lists `aggregate/data/decl-help/anatomy.html` and `card.html`.
* In JupyterLab, `decl_help()` and `decl_help('card')` render as the mockups do, the hover
  linking works, the two shown in one notebook keep their own type sizes, and setting
  `--dh-size` on one scales only that one.
* Against the api: `uv pip install -e V:/worktrees/aggregate_REFACTOR` in the api checkout,
  then the api plan's own contract run (`uv run --no-sync pytest`).
* The card, saved as PDF from Chrome or Edge with the api's print path, is one Letter
  landscape page.

## CHANGELOG (draft)

```
## 1.2.0a15 - 2026-10-xx

**[DecL-Help-Pages] `decl_help()` returns one page of DecL help as HTML**, for Jupyter and for the aggregate Loss Lab, which serves it at `GET /v1/decl/help`. `decl_help()` is the anatomy (one `agg` statement taken apart slot by slot, a portfolio and a P&L built on it, and the alternative forms for each slot); `decl_help('card')` is a ten-box reference card that prints to one Letter landscape page. The return is a `str` (`DeclHelp`) with `_repr_html_`; the forms are listed in `DECL_HELP_FORMS`. Every example in both pages builds under the test suite. Plan `dev/plan-1.2.0a15-decl-help.md`.
```

The 1.2.0 release note, written at release, names `decl_help()` among the release's headline
additions.

Commit subject: `[DecL-Help-Pages] 1.2.0a15: decl_help ships the anatomy and card DecL pages`

## Not in this plan

* **The api and SPA half**: `V:\dev\aggregate-api\dev\plan-1.7.0-decl-help.md`.
* **Concept A**, the four-step walkthrough: parked.
* **The PDF cheat sheet.** Its retired `so` and `of` were removed on 2026-10-13, outside this
  plan; the card does not replace it.
* **Embedding the card in the RTD docs** (a `raw:: html` block fed by `decl_help('card')`). Easy
  once this lands; a separate decision.

## Alternatives considered

* **Markdown source.** Ruled out by the author: the SPA has no renderer.
* **`HtmlFormatter` from Pygments** (what `_colorize` uses). Its classes need Pygments' own
  stylesheet and clash with any page that already styles `.highlight .k` (Sphinx, JupyterLab).
  The fragment needs its own six classes, which is a twenty-line token walk.
* **Pre-render at build time and commit the colored HTML.** Removes the render step, and doubles
  the files the author edits by hand, or makes them edit generated output. Rendering at call time
  costs milliseconds once per process.
* **A shared `base.css` composed into both fragments.** Removes about twenty duplicated lines,
  and adds a composition step and a third file. With every rule scoped, duplicated lines cannot
  interfere, so the duplication is harmless.
* **An HTML parser instead of regular expressions.** The source rules make the regular
  expressions safe, and the tests check those rules, so a parser buys nothing.

## Execution log

Executed 2026-10-13 as one bump, 1.2.0a14 to **1.2.0a15**, on the author's `/execute-plan`.

### Review

* The status line read "draft, for a final cross-review". The api plan records that cross-review as done on 2026-10-13, and the one open item (the api's Q4) is the api's, so the invocation was taken as the go-ahead.
* Every path, symbol and token type in "Current behavior" checked out against 1.2.0a14. `decl_help`, `DeclHelp` and `DECL_HELP_FORMS` collide with nothing in `src/` or `tests/`.
* The content source was read, not written, from `X:\decl-help\`, a drive outside the usual working set, because this plan names it as the source.
* No `dev/TODO.md` entry existed for [DecL-Help-Pages]; one was added under "Recently landed".

### Divergences

1. **[Anatomy-Right-Column] `.dh-part { min-width: 0 }` added to the anatomy.** In the mockup the right column overran the section at any width between the 820px breakpoint and about 1,300px: the portfolio's longest line (73 characters) set the grid item's minimum width, so `overflow: hidden` on the section clipped the prose and the P&L equation. Measured at 1,100px, 521px of content sat in a 453px column, identically in the mockup and in the straight copy. With the rule, the prose wraps and the long line scrolls inside its `pre`, as that `pre`'s `overflow-x: auto` intends. Breaking the portfolio's unit lines in two, as the card does, would remove the scroll too; that is a content choice left to the author.
2. **[Card-Header-Code] `.dh-head p code { font-size: .896em }` added to the card.** The base `code` rule converts to `.936em` of an 11px parent, but the header paragraph is 11.5px, so without this rule the header's inline code would grow from 10.3px to 10.8px. With it, the conversion is exact.
3. **Dead rules dropped**: `.ln` (used only by the mockups' in-browser colorer), `.t-atom` and `--dh-atom` (the plan drops `t-atom`), and the card's duplicate `.tpl { margin-bottom }`.
4. **What "sizes in em" covers.** Font sizes, padding, margins, gaps, offsets, widths, the gradient stripes in optional rows, and the container breakpoints are in `em`. Border widths, radii and the brace's stroke stay in `px`, so hairlines stay crisp when a host scales `--dh-size`. A container breakpoint in `em` is evaluated against the section's own font size, so the layout collapses in proportion when `--dh-size` changes. Rule 5 in the module docstring says this.
5. **A `pre`'s `data-load` value drops the newline after `<pre>` and the one before `</pre>`.** HTML drops the first anyway, and the second would leave a blank last line in the editor.
6. **`decl-testers.agg` is not extended**, although the project's `execute-plan` skill asks for every DecL program used in a pytest case to be mirrored there. The help sources are themselves the corpus, and the test reads them directly; a mirror would be the second copy this plan exists to avoid.
7. **Docs: an `automodule` entry for `aggregate.decl_help`** in `docs/3_reference/3_x_Parser.rst`, beside `decl_pygments`. It is not in the plan's file list; without it the new module would be the only DecL module missing from the reference chapter. Docs are pending a rebuild.
8. **The plan stays in `dev/`.** The project's `version-bump` skill moves a finished plan at the bump; this plan and the house rule say it moves when the author says it is done.

### Measurements

* `tests/test_decl_help.py`: 141 cases, about 4 s. The seven programs build (three in the anatomy, four on the card), and all 126 code snippets lex with no `Token.Error`. The tests also check that the rendered code, with its spans removed, is the source text exactly.
* The rendered fragments were compared with the mockups in headless Edge, measuring every heading, table, row, cell, box, list item and `pre`. That covers 73 elements in the anatomy (at 1,100px and at 700px, where the container query collapses it to one column) and 178 on the card (at 1,056px and 800px). All agree within 0.4px, apart from divergence 1, which is intended.
* The card at 1,056px is 812.8px tall, against a page height of 816px. That leaves **3px of headroom**: one more row on the card spills onto a second page.
* The card was printed from headless Edge into an 11in sheet under `@page { size: letter landscape; margin: 0 }`. With and without `in-lab`, the result is one page of 792 by 612pt, with the box fills and token colors intact.
* `uv build --wheel` lists `aggregate/data/decl-help/anatomy.html` and `card.html`.
* Rendering both forms takes 6ms, once per process.

### Left for the author

* JupyterLab acceptance (render, hover linking, two fragments at their own sizes, `--dh-size` on one) and the api's contract run against this worktree. Neither was run here.
* **A ripple for the api.** Every fragment rule now carries `.decl-help.dh-{form}`, so each rule outranks the unscoped mockup rule it replaces by one class, and the `in-lab` rules are (0,4,0). A host rule meant to override the fragment must outrank that. The mockups' print CSS, `body .decl-help.in-lab .dh-lab { display: none }`, is (0,3,1) and would no longer hide the Lab notes in print. The api plan's print path overrides no fragment rule, so nothing in it breaks.
