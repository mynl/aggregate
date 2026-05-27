# Plan — promote `parser_errors` into the default DecL parse path

**Status:** ready to execute on your green light.
**Depends on:** nothing (module already exists, fully tested at `tests/test_parser_errors.py` — 32 tests passing).
**Unblocks:** every notebook user who mistypes DecL gets a useful error.

## Why

`aggregate.parser_errors.format_error(source, exc)` turns Lark's
terse `UnexpectedCharacters` / `UnexpectedToken` / `UnexpectedEOF`
into an `ErrorReport` with:

- line + column
- source-line echo + caret marker
- friendly terminal labels (`'mixed'`, `'occurrence'` …)
- `"Did you mean: mixed?"` suggestions via `difflib.get_close_matches`
- `.render()` — multi-line text for terminals/REPL
- `.to_dict()` — JSON-friendly (was the api shape; now incidental)

Today this module is **not wired into the default parse path**.
`UnderwritingParser.parse` raises `ValueError(SimpleNamespace(...))`
(legacy SLY token shape, see `parser.py:920-937`); the
`Underwriter._interpret_program` callers at `underwriter.py:539-550`
and `underwriter.py:856-…` log a `'Parse error in input "<text with
>>> insertion>"'` line and re-raise.

So `build('agg X 100 claims sev lognorm 100 cv 2 mixd poisson 0.5')`
in a notebook today emits:

```
ValueError: namespace(type='?', value='m', index=38)
```

After this plan, the same call emits something like:

```
DecL parse error at line 1, column 39:

  agg X 100 claims sev lognorm 100 cv 2 mixd poisson 0.5
                                        ^^^^

Unexpected 'mixd'.

Did you mean: mixed?

Expected: 'mixed', 'occurrence', '/', '+', '-', …
```

That is the whole win. Notebooks, scripts, the `qd(build(...))`
one-liner — every call site picks it up.

## Tree (after promotion)

No new modules. Two existing files edited:

```
src/aggregate/
    parser.py          ← attach .report to the ValueError on the way out
    parser_errors.py   ← (already exists, unchanged)
    underwriter.py     ← upgrade the two except-ValueError blocks
tests/
    test_parser_errors.py    ← (already exists, 32 tests)
    test_parser_errors_integration.py   ← NEW: end-to-end via build()
```

## Design

Three small changes plus a back-compat decision.

### 1. `parser.py`: attach an `ErrorReport` to the wrapping `ValueError`

The current wrapping at `parser.py:920-937` raises three flavors of
`ValueError(SimpleNamespace(type, value, index))`. **Keep that
exactly as-is** — external code may already unpack `e.args[0].type`
etc. Attach the report as an attribute instead:

```python
# In parser.py, after computing `text` and entering the try:
except UnexpectedToken as e:
    tok = e.token
    err = ValueError(
        SimpleNamespace(
            type=getattr(tok, "type", "?"),
            value=str(tok),
            index=getattr(tok, "start_pos", 0) or 0,
        )
    )
    err.report = format_error(text, e)
    raise err from e
```

…and the same `.report = format_error(text, e)` attach in the two
sibling branches (`UnexpectedCharacters`, `UnexpectedInput`).

Why an attribute, not the args:

- `args` is part of the exception's public unpacking contract.
  Adding/changing positions there is a breaking change.
- `e.report` is discoverable via tab-complete in IPython and
  doesn't disturb any existing isinstance/match-args/repr usage.
- `format_error` is already pure and side-effect-free, so the
  cost is one dataclass construction per failed parse (cheap;
  parse failures are rare and never on the hot path).

**Decision: do NOT override `__str__`.** Leave `str(e)` rendering
the legacy `args[0]` SimpleNamespace alone. Callers that want the
rich form opt in via `e.report.render()`. Reasons:

- `ValueError.__str__` is C-level; overriding requires a subclass,
  which adds a new public class to the API surface.
- The error object would wear two faces (`args[0]` legacy,
  `e.report` structured). `str(e)` ambiguating between them is
  user-hostile.
- Opt-in is roll-back-cheap. The reverse — flip the default,
  realize logs got noisy, walk it back — is expensive.

The docs section below shows the explicit opt-in patterns.

### 2. `underwriter.py`: log the rendered report instead of the `>>>` line

Two sites: `underwriter.py:539-550` (`_interpret_program`) and the
near-identical block around `underwriter.py:856`. Both currently do:

```python
except ValueError as e:
    if isinstance(e.args[0], str):
        logger.error(e)
        raise
    t = e.args[0].type
    v = e.args[0].value
    i = e.args[0].index
    txt2 = program_line[0:i] + '>>>' + program_line[i:]
    logger.error('Parse error in input "%s"\nValue %s of type %s not expected',
                 txt2, v, t)
    raise
```

Replace the bottom half with:

```python
except ValueError as e:
    if isinstance(e.args[0], str):
        logger.error(e)
        raise
    report = getattr(e, "report", None)
    if report is not None:
        logger.error("DecL parse error:\n%s", report.render())
    else:
        # Legacy fallback path -- a ValueError that didn't come
        # through our parser wrapping. Shouldn't happen in practice
        # but kept so external test doubles still work.
        t = e.args[0].type
        v = e.args[0].value
        i = e.args[0].index
        txt2 = program_line[0:i] + '>>>' + program_line[i:]
        logger.error('Parse error in input "%s"\nValue %s of type %s not expected',
                     txt2, v, t)
    raise
```

The two sites differ slightly in surrounding context; the change
is the same shape in both.

### 3. `parser_errors.py`: trim the api-era references in the docstring

Lines 13-14 of the module docstring currently say:

> The intended consumer is :mod:`aggregate.api` (the FastAPI service
> in Plan C), but the formatter is pure…

After the api deletion that's wrong. Replace with:

> The intended consumer is the default :class:`UnderwritingParser.parse`
> error path (every `build(...)` call hitting a DecL typo), plus any
> CLI / REPL / debugger that wants a richer error story.

No code change — pure docstring sweep. Catch the same kind of stale
"plan C / plan D / api boundary" prose elsewhere in the module while
we're there.

## Back-compat surface

Listing this explicitly so review can blow it up if something is
wrong:

| Caller of `parser.parse` | Breaks? |
|---|---|
| `e.args[0].type / .value / .index` consumers | **No** — unchanged. |
| `str(e)` consumers (e.g. logging) | **No** — `__str__` unchanged. |
| Tracebacks in pytest | **No** — `ValueError` class unchanged. |
| `isinstance(e, ValueError)` | **No** — class unchanged. |
| New `.report` attribute readers | New capability; no existing readers. |

The `Underwriter` log message changes shape (the `>>>` insertion
goes away in the common case). Anyone grepping logs for the literal
`Parse error in input` string will need to update — but that's
internal tooling, and the new line carries strictly more info.

## Tests

New file: `tests/test_parser_errors_integration.py`. Lightweight —
the heavy unit coverage already exists.

```python
# Asserts:
# - build('agg X 100 claims sev lognorm 100 cv 2 mixd poisson 0.5')
#   raises ValueError with a .report attribute
# - report.suggestions includes 'mixed'
# - report.line == 1, column points at the typo
# - report.render() contains the caret line
# - the legacy ValueError(args[0]) shape is unchanged
# - logger.error called with rendered text (use caplog)
```

Plus one regression test that a successful `build('agg Dice ...')`
still works — the wrapping only activates on failure.

Estimated ~6 tests, all under one pytest second.

## Decided knobs

| Knob | Decision | Note |
|---|---|---|
| Override `__str__`? | **No** | Explicit opt-in via `e.report.render()`. See docs section. |
| `_repr_html_` on ErrorReport? | **Defer to v1.1** | Pure polish; styling depends on the colorization plan landing. |
| `logger.error` vs `warning`? | **error** | A failed parse is the user's program being wrong. |
| Surface `.report` to `Underwriter` callers? | **Automatic** | The re-raise propagates `.report` for free. |

## Documenting the opt-in

Add a short "Reading parse errors" section to the DecL language
reference (`docs/4_agg_language_reference/`). Three patterns,
each two lines:

**1. Notebook / REPL — show the formatted error before the traceback.**

```python
try:
    build('agg X 100 claims sev lognorm 100 cv 2 mixd poisson 0.5')
except ValueError as e:
    print(e.report.render())     # nice caret + "did you mean"
    raise                         # re-raise to keep the traceback
```

**2. Script that wants the suggestion programmatically.**

```python
try:
    build(text)
except ValueError as e:
    if getattr(e, "report", None) and e.report.suggestions:
        print(f"Did you mean: {e.report.suggestions[0]}?")
    raise
```

**3. IPython traceback hook — auto-format every parse error in a session.**

```python
# In ~/.ipython/profile_default/startup/decl_errors.py:
from IPython import get_ipython
def _showtb(self, etype, evalue, tb, **kw):
    report = getattr(evalue, "report", None)
    if report is not None:
        print(report.render())
    return self._showtraceback_original(etype, evalue, tb, **kw)
ip = get_ipython()
ip._showtraceback_original = ip.showtraceback
ip.showtraceback = _showtb.__get__(ip)
```

Pattern (3) is power-user; not bundled into the library, just shown
as a recipe. The library does not install IPython hooks on import.

These three patterns are what the docs page exists to teach. Each
sample compiles and runs as written; the snippets go in a
`.. testcode::` block so the doc build catches drift.

## Out of scope

- Updating the parser to emit suggestions for *semantic* errors
  (e.g. unknown distortion name in the lookup phase). Those raise
  `KeyError`/`ValueError` from the transformer, not the parser.
  Separate plan.
- Translating Lark's "Expected one of: …" terminal sets into
  full English phrases (e.g. "expected a frequency clause").
  The current `_label` map is the right granularity for v0.
- Localization. English only; not worth the abstraction yet.

## Verification

1. `uv run pytest tests/test_parser_errors.py tests/test_parser_errors_integration.py -q`
   — both green.
2. `uv run pytest -q` — full suite green (no regression on the
   667 existing tests).
3. Smoke test in a one-liner:
   ```
   uv run python -c "from aggregate import build; build('agg X 100 claims sev lognorm 100 cv 2 mixd poisson 0.5')"
   ```
   Expected: traceback ends in `ValueError`, immediately above which
   the logger has printed the formatted DecL parse error with caret
   and "Did you mean: mixed?".
4. Same in a Jupyter cell: same logger output, same traceback shape.

## Recovery / rollback

Three files touched; revert is one `git revert`. The parser_errors
module is unchanged in behavior — only the docstring sweep is
visible to users.

## File-by-file checklist

1. `src/aggregate/parser.py`: attach `.report = format_error(text, e)`
   in the three `except` branches at lines 920-937. Three lines added,
   no logic change.
2. `src/aggregate/underwriter.py`: upgrade the two
   `_interpret_program` `except ValueError` blocks (lines 540-550
   and ~856) to log `e.report.render()` when present, falling back
   to the legacy `>>>` insertion if absent.
3. `src/aggregate/parser_errors.py`: docstring sweep — strip Plan C
   / api / "intended consumer is aggregate.api" prose. Replace with
   "intended consumer is the default `build()` error path; reusable
   from any CLI / REPL / debugger".
4. `tests/test_parser_errors_integration.py`: new file, ~6 tests
   covering: `.report` attached on UnexpectedCharacters, on
   UnexpectedToken, on UnexpectedEOF; legacy `args[0]` shape unchanged;
   `caplog` sees the rendered output via `_interpret_program`;
   successful builds untouched.
5. `docs/4_agg_language_reference/`: new short page "Reading parse
   errors" with the three opt-in recipes above. Wire into the
   section toctree.
6. `README.rst`: replace the standalone "promotion is pending"
   sentence in the parser_errors bullet with a present-tense
   description of the new default behavior.
