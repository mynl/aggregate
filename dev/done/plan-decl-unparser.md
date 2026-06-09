# Plan — DecL unparser + program formatter (`decl_writer`)

> **Status: ready to execute.** Larger than the hygiene items — this builds a new
> capability (the inverse of the parser). Bumps `1.0.0a*`. Author (prod-claude)
> executes and commits. **Breaking by design and by agreement**: `decl_pprint` is
> removed, programs render in a canonical form (not the user's verbatim text), and
> `to_agg` output becomes canonical. Fix docs/callers behind it.

## Background / decisions from the pre-execution review

Settled before writing the module (read this first — it corrects a premise in the
original draft):

- **There are two different "spec" dicts, and `spec_to_decl` inverts only one.**
  - The **raw transformer spec** — `parsed.spec`, the knowledge entry's `pp.spec`
    — is sparse (only grammar-produced keys) and carries `_PercentNumber`,
    `Copula` objects, reins tuples, `np.inf`. **This is the unparser's input.**
  - `Aggregate._spec` is a *different* object: `distributions.py:3380` captures it
    as `dict(inspect.getargvalues(frame).locals)`, i.e. the full `__init__`
    keyword set with every default filled in (dense). It is **not** the parse
    spec, and `spec_to_decl` does **not** consume it.
- **`pprogram` re-parses.** `Aggregate.pprogram` / `Portfolio.pprogram` become
  `format_program(self.program)` — re-parse the verbatim text the object already
  stores → raw spec → render. Consequence (accepted): an object built
  *programmatically* (`Aggregate(**spec)`, `Portfolio` arithmetic) has
  `self.program == ''` and so has no canonical `pprogram` — the same practical
  result as today. We do **not** add a dense-`_spec` fallback path.
- **There is no unique inverse from `_spec` back to a program**: many distinct
  programs resolve to the same `_spec` (and to the same parse spec). So the right
  invariant is **idempotence one step removed**: with `f = spec_to_decl`,
  `f(f(f(x))) == f(x)`. The original corpus text `x` need not equal `f(x)` — it
  is only *functionally equivalent* (builds the same object). The fixed point is
  reached after one application, and that fixed point is what the round-trip test
  pins. See the corrected Contract section below.
- **Spec equality in the test is numpy/object-aware, not `==`.** Specs hold
  ndarrays, `np.inf`, `Copula`, `_PercentNumber`, nested tuples — a bare `==`
  raises the array-truthiness error (the very bug hygiene-3 item 3 fixed). The
  round-trip test ships a deep, inf-aware, tolerant comparator.
- **These transforms are lossy / normalizing** (the spec carries no trace of the
  surface form, so the canonical render differs from the user's text — all still
  idempotent, because the loss already happened at first parse):
  - `agg_out_tweedie` discards the user note and bakes a CP-gamma spec → renders
    as the expanded `poisson … gamma` form, never as `tweedie …`.
  - `builtin_agg_*` resolve the builtin and mangle the name
    (`_i_scaled`/`_homog_scaled`/`_shifted`) → render expanded, not `2 * agg.Foo`.
  - `sev1_scaled` (negative multiplier) and `sev2_rsub` fold sign+factor into
    `sev_reflect` + scaled params → recoverable only by emitting `-1 * …`.
- **Range re-folding (`[1:6]`) is cosmetic, not load-bearing.** Emitting the
  explicit list round-trips identically, so re-folding is best-effort and never
  gates the round-trip test.

## Why

There is no `spec → DecL` converter anywhere in the library. `to_agg` round-trips
by writing the **verbatim** stored `program` text; `decl_pprint`
(`utilities.py:76`) pretty-prints by **regex surgery** on that text, duplicating
keyword knowledge that already lives in `decl.lark` (and again in
`decl_pygments.py`, `parser_errors._TERMINAL_LABELS`, and the web app's
`decl-keywords.json` — five hand-maintained mirrors total). The split regex
(`utilities.py:98`) doesn't even know the post-2024 keywords (`pnl`,
`multivariate`, `copula`, `netceded`, `ssev`, `splice`, `approximate`), so long
modern programs don't lay out well, and the `html=True` branch silently ignores
note-stripping and splitting (a latent bug).

The right fix is to make the printer the **inverse of the parser**: render from the
parsed spec, never from regex. That one new function then backs display,
`to_agg`, and a future web `format` endpoint, and its round-trip test doubles as a
proof that the spec dictionary fully captures the language.

## Design — three layers in a new module `src/aggregate/decl_writer.py`

(Name mirrors `parser.py`/`parser_errors.py`; the *writer* is the inverse of the
*reader*. Rename freely.)

### Layer 1 — `spec_to_decl(spec) -> str`  (the unparser; the keystone)

Pure function: a **raw transformer spec** `dict` (`parsed.spec` / `pp.spec`, *not*
the dense `Aggregate._spec` — see Background) → canonical DecL text. It is the
structural inverse of the `UnderwritingParser`
transformer, so build it as a set of small **clause renderers** that mirror the
transformer rules one-for-one:

| Renderer | Inverse of | Emits |
|---|---|---|
| `_render_exposure` | `exposures_*` / `pnl_exp_*` | `N claims` / `E loss` / `P premium at LR lr` / `X exposure at R rate`; vectors → `[…]` |
| `_render_layers` | `layers_*` | `limit xs attach`, or `tower …` |
| `_render_sev` | `sev0/1/2_*`, `sev_weighted` | family + params (param form vs `mean cv` form), `scale * dist + shift`, `wts […]` mixtures, `splice [lb ub]`, `dsev`/`dhistogram` atoms, `ssev` for signed |
| `_render_freq` | `freq_*` | `poisson` / `fixed` / `dfreq […]` / `mixed <kind> …` / `binomial …` etc., `zm`/`zt` |
| `_render_reins` | `occ_reins_*` / `agg_reins_*` / `reins_clause_*` | `occurrence|aggregate net of|ceded to <clauses>` (`so`/`po`, `xs`, towers) |
| `_render_approx` | `approx_clause` | `approximate sgamma|slognorm` (omit for `exact`) |
| `_render_trailer` | `trailer_*` | `note{…}` and `hints{…}` — **preserved, never stripped** |

Top-level dispatch by kind, mirroring the `*_out` rules:
- **`agg`**: `agg NAME <exposure> [<layers>] <sev> [<occ reins>] <freq> [<agg reins>] [<approx>] [<trailer>]`
- **`pnl`**: `pnl NAME <premium> premium - <exposure> …` (rest as `agg`)
- **`sev`** (standalone): `sev NAME <sev>`
- **`port`**: `port NAME [<trailer>]` then indented sub-`agg` lines
- **`multivariate`**: `multivariate NAME <count> claims` then indented component aggs, optional `copula <kind> <p>`, shared `mixed …` (and the `netceded <agg>` form)
- **`distortion`**: `distortion NAME <kind> n1 n2 …` (flat number list, per a27)

**Canonicalization niceties** (acceptable losses vs the user's text, plus a couple
of re-sugarings worth doing):
- Re-collapse arithmetic integer vectors to range form where exact — emit `[1:6]`
  / `[lo:hi:step]` rather than a 1001-element list for `dsev [1:1001]`. (The
  transformer expands ranges; detect and re-fold.)
- One clause per line, fixed clause order, fixed indentation. No `split=` knob —
  layout is structural.

### Layer 2 — colorization via the existing lexer

`spec_to_decl` output is plain text; colorize by lexing it with the **existing**
`decl_pygments.AggLexer` and rendering through a Pygments *formatter* selected by
`fmt`. No new lexer, no new keyword list:
- `text` → no color (plain passthrough),
- `html` → `HtmlFormatter` (web, Sphinx),
- `ansi` → `Terminal256Formatter` (REPL / Jupyter console color — the case
  `_repr_html_` can't serve; see [[project_colorization_deferred]]),
- `latex` → `LatexFormatter` (papers / book).

### Layer 3 — `format_program(spec_or_text, *, fmt='text', width=None) -> str`

The public entry. Accepts a spec dict (preferred) or a program string (parses it
first, so doc snippets that pass `obj.program` keep working). **Pure: returns a
`str`, never prints, stable return type** (an `HTML`/ANSI string is still a
`str`). Printing is the caller's job — mirrors stdlib `pformat` vs `pprint`. A
one-line `print`-wrapper may be added if desired, but is not required.

## Contract (becomes the test gate)

The achievable invariant is **idempotence one step removed** (see Background — the
inverse is not unique, so we do not require `parse(spec_to_decl(spec)) == spec`
verbatim against the user's text). With `f = spec_to_decl ∘ parse` applied to a
corpus line's text:

- **Fidelity from canonical spec:** `parse(spec_to_decl(s)) ≅ s`, where `s` is the
  *parse spec* of the corpus line (already the post-transform canonical form) and
  `≅` is numpy/inf/object-aware equality (the deep comparator, not `==`). This is
  the per-category gate.
- **Idempotence:** `spec_to_decl(parse(spec_to_decl(s))) == spec_to_decl(s)`
  (string-equal — the canonical text is a fixed point after one application).

The round-trip over the whole corpus is also a **spec-completeness proof**: any
line that fails to close has found a construct the parse spec does not faithfully
carry — fix the unparser, or (if genuinely missing) record the spec gap as a
separate bug. Do **not** paper over a gap by special-casing verbatim text.

## Execution stages (gate round-trip per category as it accumulates)

1. **Scaffold** `decl_writer.py`; dispatch skeleton; the trivial `agg N claims
   <one-param sev> <simple freq>`. Round-trip the simplest test_suite lines.
2. **Severity surface** — all param forms, `mean cv`, `scale*dist+shift`, `wts`
   mixtures, `splice`, `ssev`/signed, `dsev`/`dhistogram` atoms, range re-folding.
3. **Frequency surface** — `mixed` kinds + params, `zm`/`zt`, `dfreq`, all families.
4. **Exposure + layers** — `loss`/`premium at lr`/`exposure at rate`, vectors,
   `xs`, towers.
5. **Reinsurance** — `occ`/`agg`, `net of`/`ceded to`, `so`/`po`(/`of` if
   hygiene-3 item 4 has landed), clause lists, towers.
6. **Compound kinds** — `port`, `pnl`, `multivariate`/`copula`/`netceded`,
   `approximate`, `distortion`.
7. **Trailer** — `note{}`/`hints{}` preserved through round-trip.

After each stage the corpus round-trip set widens; full corpus green is the
overall done-gate.

## Migration / removal (the breaking part)

- **Delete `decl_pprint`** from `utilities.py` and its `__all__` entry; remove the
  now-dead regex/IPython-HTML block.
- **Export** `format_program` (and `spec_to_decl`) from the package; update
  `__init__`.
- **Rewire callers:** `Aggregate.pprogram` / `pprogram_html` (distributions.py
  ~5862–5872) and `Portfolio.pprogram` / `pprogram_html` (portfolio.py ~2408) →
  `format_program(self.program, fmt='text'|'html')` (**re-parse the stored text**,
  per Background — not `self._spec`). Keep `self.program` as the **verbatim user
  input** (provenance); `pprogram` is now the derived canonical form. An object
  with empty `self.program` (built programmatically) yields an empty `pprogram`.
- **`to_agg`** (underwriter.py) → emit `spec_to_decl(spec)` per entry instead of
  the stored `program` line, so saved `.agg` files are canonical. (Also removes
  `to_agg`'s dependence on a populated verbatim `program`.)
- **Docs:** the ~7 user-guide `.rst` files that do
  `from aggregate import decl_pprint; decl_pprint(r.program, split=20)` →
  `from aggregate import format_program; print(format_program(r.program))`.
  Docs pending rebuild (per CLAUDE.md; do not build in the loop).

## Out of scope (note, don't do here)

The **single keyword source of truth** (deriving `AggLexer` /
`_TERMINAL_LABELS` / the web `decl-keywords.json` from the grammar terminals) is
the deeper fix for the five-mirrors problem, but it is independent of this plan —
Layer 2 reuses `AggLexer` as-is. Flag it as a follow-on and connect it to
[[project_colorization_deferred]].

## Tests

- New `tests/test_decl_unparser.py`: corpus **fidelity** + **idempotence** over
  `test_suite.agg` (+ `test_suite2.agg`, `test_decl.agg`); per-category
  parametrization so failures localize.
- `format_program` smoke per `fmt`: each returns a non-empty `str`; `html`
  contains `<span`; `ansi` contains an ESC (`\x1b[`) sequence; `text` is plain.
- Add any DecL lines needed to exercise an under-covered construct to
  `test_decl.agg` (keep in sync, per [[feedback_test_decl_sync]]).

## Housekeeping

- Bump `1.0.0a*` in `pyproject.toml`.
- `CHANGELOG.md` section: new `decl_writer` module — `spec_to_decl` unparser and
  `format_program(fmt=text|html|ansi|latex)`; **breaking**: `decl_pprint` removed,
  programs/`to_agg` now render canonical DecL, `pprogram` is derived (raw input
  stays on `program`).
- `dev/TODO.md`: mark landed; add the out-of-scope "single keyword source" item.
- Move this plan to `dev/done/` when complete.
