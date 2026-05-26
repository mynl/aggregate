# Plan B — DecL error formatter

**Status:** ready to execute (after Plan A, but not blocked on it).
**Depends on:** nothing. (Reads `decl.lark` and Lark's exception classes; both already in the project.)
**Unblocks:** apiweb error pane (Plan C/D), and gives the docs/CLI a richer DecL error story for free.

## Goal

Turn Lark's parse exceptions into structured, human-friendly error reports. The current `UnderwritingParser` catches Lark exceptions and wraps them as `ValueError(SimpleNamespace(type, value, index))` — enough for the legacy callers, useless for a user trying to fix their DecL.

Plan B introduces a pure formatter module that the apiweb consumes for its error pane, and that any future CLI / REPL can reuse. It does *not* change the existing `UnderwritingParser` error path (which has many callers).

## Deliverables

- New: `src/aggregate/parser_errors.py` — the formatter module.
- New: `tests/test_parser_errors.py` — unit tests, no apiweb dependencies.
- No changes to `parser.py`, `underwriter.py`, or any caller. The legacy `ValueError(SimpleNamespace)` path stays exactly as-is.

## Module API

```python
# src/aggregate/parser_errors.py
"""
Structured DecL parse-error reports.

Lark's ``UnexpectedToken`` / ``UnexpectedCharacters`` / ``UnexpectedEOF``
exceptions carry useful information (line, column, expected terminal set)
but render as terse one-liners by default. This module converts them into
``ErrorReport`` objects suitable for IDE-style display: caret-annotated
source line, friendly terminal labels, "did you mean" suggestions.

Used by ``aggregate.apiweb`` for its error pane. Reusable in any tool
that calls Lark directly.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from difflib import get_close_matches
from typing import Iterable

from lark.exceptions import (
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedInput,
    UnexpectedToken,
)


@dataclass
class ErrorReport:
    """Structured DecL parse-error report.

    Attributes
    ----------
    line, column : int
        1-indexed position of the offending token.
    source : str
        The original DecL text (echoed for context).
    source_line : str
        The single line of ``source`` containing the error.
    caret : str
        A line of spaces and ``^`` characters that aligns with
        ``source_line`` to mark the bad span. Width matches the bad
        token (1 char minimum).
    got : str | None
        The unexpected token's literal value, if any.
    got_type : str | None
        The unexpected token's terminal name, if any.
    expected : list[str]
        Human-friendly labels for terminals that would have been
        accepted at this position (e.g. ``"'mixed'"``, ``"a number"``).
    expected_terminals : list[str]
        Raw Lark terminal names — for tooling, not display.
    suggestions : list[str]
        ``"Did you mean..."`` candidates derived from ``got`` against
        the literal terminals in ``expected_terminals``.
    message : str
        One-line human-readable summary.
    """
    line: int
    column: int
    source: str
    source_line: str
    caret: str
    got: str | None
    got_type: str | None
    expected: list[str]
    expected_terminals: list[str]
    suggestions: list[str]
    message: str

    def to_dict(self) -> dict:
        """JSON-serializable form for the apiweb response body."""
        return asdict(self)

    def render(self) -> str:
        """Multi-line text rendering for terminals / REPL."""
        lines = [
            f"DecL parse error at line {self.line}, column {self.column}:",
            "",
            f"  {self.source_line}",
            f"  {self.caret}",
            "",
            self.message,
        ]
        if self.suggestions:
            lines.append("")
            lines.append("Did you mean: " + ", ".join(self.suggestions) + "?")
        if self.expected:
            lines.append("")
            lines.append("Expected: " + ", ".join(self.expected[:10]) +
                         ("..." if len(self.expected) > 10 else ""))
        return "\n".join(lines)


def format_error(source: str, exc: BaseException) -> ErrorReport:
    """Convert a Lark parse exception into an ``ErrorReport``.

    Accepts ``UnexpectedToken``, ``UnexpectedCharacters``,
    ``UnexpectedEOF``, or any other ``UnexpectedInput`` subclass. If the
    exception was wrapped by ``UnderwritingParser.parse`` into a
    ``ValueError``, the original Lark exception is unwrapped from
    ``__cause__``.

    Parameters
    ----------
    source : str
        The DecL text that was parsed.
    exc : BaseException
        The exception raised. Either a Lark ``UnexpectedInput`` subclass
        or a ``ValueError`` wrapping one via ``raise ... from``.

    Returns
    -------
    ErrorReport
    """
    # Unwrap UnderwritingParser's ValueError wrapping if present.
    if not isinstance(exc, UnexpectedInput) and exc.__cause__ is not None:
        if isinstance(exc.__cause__, UnexpectedInput):
            exc = exc.__cause__

    if isinstance(exc, UnexpectedToken):
        return _from_token(source, exc)
    if isinstance(exc, UnexpectedCharacters):
        return _from_chars(source, exc)
    if isinstance(exc, UnexpectedEOF):
        return _from_eof(source, exc)
    if isinstance(exc, UnexpectedInput):
        return _from_generic(source, exc)
    # Anything else: minimal report.
    return _fallback(source, exc)
```

Per-exception helpers (private):

```python
def _from_token(source: str, exc: UnexpectedToken) -> ErrorReport:
    line, col = exc.line, exc.column
    tok = exc.token
    got = str(tok)
    got_type = getattr(tok, "type", None)
    expected_raw = sorted(exc.expected)
    expected_labels = [_label(t) for t in expected_raw]
    suggestions = _did_you_mean(got, expected_raw)
    return ErrorReport(
        line=line,
        column=col,
        source=source,
        source_line=_source_line(source, line),
        caret=_caret(col, max(1, len(got))),
        got=got,
        got_type=got_type,
        expected=expected_labels,
        expected_terminals=expected_raw,
        suggestions=suggestions,
        message=f"Unexpected {_label(got_type)} {got!r}.",
    )


def _from_chars(source: str, exc: UnexpectedCharacters) -> ErrorReport:
    line, col = exc.line, exc.column
    char = source[exc.pos_in_stream] if exc.pos_in_stream < len(source) else ""
    expected_raw = sorted(getattr(exc, "allowed", set()) or set())
    return ErrorReport(
        line=line,
        column=col,
        source=source,
        source_line=_source_line(source, line),
        caret=_caret(col, 1),
        got=char,
        got_type=None,
        expected=[_label(t) for t in expected_raw],
        expected_terminals=expected_raw,
        suggestions=[],
        message=f"Unexpected character {char!r}.",
    )


def _from_eof(source: str, exc: UnexpectedEOF) -> ErrorReport:
    line = source.count("\n") + 1
    last_line = _source_line(source, line)
    col = len(last_line) + 1
    expected_raw = sorted(exc.expected)
    return ErrorReport(
        line=line,
        column=col,
        source=source,
        source_line=last_line,
        caret=_caret(col, 1),
        got=None,
        got_type=None,
        expected=[_label(t) for t in expected_raw],
        expected_terminals=expected_raw,
        suggestions=[],
        message="Unexpected end of input.",
    )


def _from_generic(source, exc): ...   # minimal report from .line/.column if present
def _fallback(source, exc): ...       # last-resort report
```

Utilities:

```python
def _source_line(source: str, line: int) -> str:
    lines = source.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1]
    return ""


def _caret(column: int, width: int) -> str:
    """Build a caret line for column ``column`` (1-indexed), ``width`` cols wide."""
    return " " * max(0, column - 1) + "^" * max(1, width)


# Hand-curated mapping of common Lark terminal names → human labels.
# Anonymous / unknown terminals fall through to the raw name.
_TERMINAL_LABELS: dict[str, str] = {
    "INT": "an integer",
    "SIGNED_INT": "an integer",
    "NUMBER": "a number",
    "SIGNED_NUMBER": "a number",
    "FLOAT": "a number",
    "NAME": "an identifier",
    "CNAME": "an identifier",
    "STRING": "a quoted string",
    "ESCAPED_STRING": "a quoted string",
    "NEWLINE": "end of line",
    "LBRACKET": "'['",
    "RBRACKET": "']'",
    "LPAREN": "'('",
    "RPAREN": "')'",
    "COMMA": "','",
    "COLON": "':'",
    # DecL-specific terminals filled in once we audit decl.lark for the
    # public keywords. Anything unrecognized falls through to the raw name.
}


def _label(terminal: str | None) -> str:
    if terminal is None:
        return "token"
    if terminal in _TERMINAL_LABELS:
        return _TERMINAL_LABELS[terminal]
    # Anonymous Lark terminals are like __ANON_0; skip those.
    if terminal.startswith("__"):
        return "token"
    # Keyword-style terminals (uppercased names matching their literal)
    # → wrap in quotes.
    return f"'{terminal.lower()}'"


def _did_you_mean(got: str, expected_terminals: Iterable[str]) -> list[str]:
    """Suggest close matches between the bad token and expected literals."""
    if not got:
        return []
    # Only consider expected terminals that look like keywords (uppercase ASCII).
    candidates = [t.lower() for t in expected_terminals
                  if t.isupper() and t.isascii() and not t.startswith("__")]
    return get_close_matches(got.lower(), candidates, n=3, cutoff=0.6)
```

## Design notes

- **Pure formatter, no parsing.** The module never invokes Lark; it interprets exception instances. This keeps it cheap and testable.
- **Doesn't replace the existing wrapping.** The `UnderwritingParser.parse` ValueError wrapper has many callers across the codebase; touching it is out of scope for Plan B. The apiweb calls Lark at a level where the original exception is available (Plan C), or unwraps via `__cause__`.
- **Terminal labeling is hand-curated.** A more general approach would parse `decl.lark` to extract each terminal's literal value, but that's complex (Lark terminals can be regexes, references, alternations). The hand-curated table covers the common keywords; everything else falls back to the raw terminal name. Plan B ships the seed table; subsequent passes can expand it as real errors expose gaps.
- **"Did you mean" is `difflib.get_close_matches`.** Stdlib only, well-tested, cutoff 0.6 is conservative.
- **No grammar excerpt (yet).** Earlier brainstorm mentioned showing the relevant grammar rule. Implementing this well requires walking the Lark parse-table to find which rule was active — non-trivial. Deferred to a future enhancement; the structured `expected` list already does most of the work.
- **Doc-link field also deferred.** Same reasoning — needs a stable rule-name → anchor mapping that doesn't exist yet.

## Verification steps

1. `uv run pytest tests/test_parser_errors.py -v` — all tests pass.
2. Manual REPL spot-check:
   ```python
   from aggregate.parser_errors import format_error
   from lark.exceptions import UnexpectedToken
   from aggregate.parser import UnderwritingParser
   p = UnderwritingParser(lambda x: x)
   bad = "agg X 100 claims sev lognorm 100 cv 2 mixedd poisson"
   try:
       p.parse(bad)
   except ValueError as e:
       report = format_error(bad, e)
       print(report.render())
   ```
   Expect a multi-line report with caret pointing at `mixedd` and a suggestion of `mixed`.
3. Existing parser tests still pass: `uv run pytest tests/ -k parser` (regression sanity — Plan B doesn't modify `parser.py`, but worth running).

## Test sketch (`tests/test_parser_errors.py`)

```python
import pytest
from lark.exceptions import UnexpectedToken, UnexpectedCharacters

from aggregate.parser import UnderwritingParser
from aggregate.parser_errors import ErrorReport, format_error, _label, _did_you_mean


@pytest.fixture
def parser():
    return UnderwritingParser(lambda x: x)


def _try_parse(parser, text):
    """Helper: parse and return the raised exception."""
    with pytest.raises(Exception) as info:
        parser.parse(text)
    return info.value


# ---- format_error against real parser output ----

def test_unexpected_token_basic(parser):
    bad = "agg X 100 claims sev lognorm 100 cv 2 mixedd poisson"
    exc = _try_parse(parser, bad)
    report = format_error(bad, exc)
    assert isinstance(report, ErrorReport)
    assert report.line == 1
    assert report.column > 0
    assert report.got is not None
    assert "mixedd" in report.got
    assert report.suggestions  # something close to 'mixed'

def test_did_you_mean_finds_mixed():
    # Pure unit test of the suggestion function.
    assert "mixed" in _did_you_mean("mixedd", ["MIXED", "NET", "CEDED"])

def test_did_you_mean_empty_for_nonsense():
    assert _did_you_mean("zzzzzz", ["MIXED", "NET", "CEDED"]) == []


# ---- label mapping ----

def test_label_known_terminal():
    assert _label("NUMBER") == "a number"
    assert _label("NAME") == "an identifier"

def test_label_anonymous_skipped():
    assert _label("__ANON_0") == "token"

def test_label_keyword_quoted():
    assert _label("MIXED") == "'mixed'"


# ---- rendering ----

def test_render_contains_caret(parser):
    bad = "agg X 100 zzz"
    exc = _try_parse(parser, bad)
    rendered = format_error(bad, exc).render()
    assert "^" in rendered
    assert "line" in rendered.lower()


# ---- to_dict serializable ----

def test_to_dict_json_safe(parser):
    import json
    bad = "agg X 100 zzz"
    exc = _try_parse(parser, bad)
    d = format_error(bad, exc).to_dict()
    json.dumps(d)  # must not raise
```

## Open knobs for execution

- **Seed `_TERMINAL_LABELS` table.** I've listed the structural / generic Lark terminals; the DecL-specific keyword terminals (`MIXED`, `SEV`, `OCCURRENCE`, `CEDED`, etc.) should be enumerated from `decl.lark` as part of execution. Roughly 30–40 entries. Worth a short pass through the grammar.
- **`get_close_matches` cutoff.** 0.6 is the default; 0.7 reduces noise, 0.5 finds more. Default is sensible; adjust based on test feel.
- **`render()` truncation.** I cap "Expected: ..." at 10 items. The expected-set for some DecL positions is large (top-level can accept many keywords). Cap is arbitrary — confirm or change.

## Out of scope (explicit non-goals)

- Modifying `parser.py` / `UnderwritingParser.parse` to raise richer exceptions natively. Many callers depend on the current `ValueError(SimpleNamespace)` shape; changing it is a separate, larger refactor.
- Grammar excerpt / "this rule was being matched" feature. Deferred; the expected-set already covers most of the user value.
- Doc deep-links. Deferred until a stable rule-anchor map exists.
- CLI integration (an `aggregate-lint` command or similar). Plan B is library code only; CLI would be its own plan.

## File-by-file checklist (for execution)

1. Read `src/aggregate/decl.lark` and extract keyword terminal names. Add to `_TERMINAL_LABELS` seed table.
2. Create `src/aggregate/parser_errors.py` with the API above.
3. Create `tests/test_parser_errors.py` with the test sketch.
4. Run `uv run pytest tests/test_parser_errors.py -v` until green.
5. Run `uv run pytest tests/ -k parser` to confirm no regressions.
6. Manual REPL check per "Verification steps" item 2.

## Recovery / rollback

No existing code modified — plan B is pure addition. To roll back: delete the two new files. Nothing to revert in `parser.py`, `underwriter.py`, or any other module.
