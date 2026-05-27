"""
Structured DecL parse-error reports.
====================================

Lark's :class:`~lark.exceptions.UnexpectedToken`,
:class:`~lark.exceptions.UnexpectedCharacters`, and
:class:`~lark.exceptions.UnexpectedEOF` exceptions carry useful
information (line, column, expected terminal set) but render as terse
one-liners. This module converts them into :class:`ErrorReport` objects
suitable for IDE-style display: caret-annotated source line, friendly
terminal labels, and "did you mean" suggestions.

The intended consumer is the default :class:`UnderwritingParser.parse`
error path (every ``build(...)`` call hitting a DecL typo), plus any
CLI / REPL / debugger that wants a richer DecL error story. The
formatter is pure: it accepts a Lark exception (or a
:class:`ValueError` wrapping one) and returns a dataclass.

``UnderwritingParser.parse`` attaches an :class:`ErrorReport` to the
wrapping :class:`ValueError` as ``.report``; this module's
:func:`format_error` also unwraps the Lark exception via
``__cause__`` if a raw ``ValueError`` is passed in.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from difflib import get_close_matches
from typing import Iterable

from lark.exceptions import (
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedInput,
    UnexpectedToken,
)

# Word-boundary scanner for UnexpectedCharacters. Lark only hands us the
# single offending character; we look forward through identifier-like
# characters (matching the DecL ID regex character class) to recover the
# full mistyped token. This is what makes "Did you mean..." useful for
# the Earley + dynamic-lexer setup, where bad input almost always
# surfaces as UnexpectedCharacters rather than UnexpectedToken.
_WORD_CHARS = re.compile(r"[A-Za-z0-9._:~\-]+")

__all__ = ["ErrorReport", "format_error"]


# ----------------------------------------------------------------------
# Terminal -> human label mapping.
# Hand-curated seed table covering generic Lark terminals plus the
# DecL keyword terminals declared in ``decl.lark``. Anonymous and
# unrecognized terminals fall through to ``_label``'s defaults.
# ----------------------------------------------------------------------
_TERMINAL_LABELS: dict[str, str] = {
    # Generic Lark terminals.
    "INT": "an integer",
    "SIGNED_INT": "an integer",
    "NUMBER": "a number",
    "SIGNED_NUMBER": "a number",
    "FLOAT": "a number",
    "NAME": "an identifier",
    "CNAME": "an identifier",
    "ID": "an identifier",
    "STRING": "a quoted string",
    "ESCAPED_STRING": "a quoted string",
    "NEWLINE": "end of line",
    # Punctuation / operators (DecL-specific).
    "LBRACKET": "'['",
    "RBRACKET": "']'",
    "LPAREN": "'('",
    "RPAREN": "')'",
    "COMMA": "','",
    "COLON": "':'",
    "EXPONENT": "'**' or '^'",
    "PLUS": "'+'",
    "MINUS": "'-'",
    "TIMES": "'*'",
    "DIVIDE": "'/'",
    "INHOMOG_MULTIPLY": "'@'",
    "EQUAL_WEIGHT": "'='",
    "RANGE": "':'",
    # DecL keyword terminals (decl.lark).
    "OCCURRENCE": "'occurrence'",
    "AGGREGATE": "'aggregate'",
    "EXPOSURE": "'exposure'",
    "TWEEDIE": "'tweedie'",
    "PREMIUM": "'premium'",
    "TOWER": "'tower'",
    "MIXED": "'mixed'",
    "PICKS": "'picks'",
    "CLAIMS": "'claims'",
    "SPLICE": "'splice'",
    "CEDED": "'ceded'",
    "DFREQ": "'dfreq'",
    "DSEV": "'dsev'",
    "LOSS": "'loss'",
    "PORT": "'port'",
    "RATE": "'rate'",
    "NET": "'net'",
    "SEV": "'sev'",
    "AGG": "'agg'",
    "XPS": "'xps'",
    "WEIGHTS": "'wts'",
    "AND": "'and'",
    "EXP": "'exp'",
    "AT": "'at'",
    "CV": "'cv'",
    "LR": "'lr'",
    "XS": "'xs'",
    "OF": "'of'",
    "TO": "'to'",
    "PART_OF": "'po'",
    "SHARE_OF": "'so'",
    "ZM": "'zm'",
    "ZT": "'zt'",
    "DISTORTION": "'distortion'",
    # FREQ accepts a set of distribution-name literals; show the menu.
    "FREQ": "a frequency name (poisson, binomial, ...)",
    # Builtin-dotted forms.
    "BUILTIN_AGG": "a builtin aggregate (agg.X)",
    "BUILTIN_SEV": "a builtin severity (sev.X)",
    "BUILTIN_DIST": "a builtin distortion (dist.X)",
    "NOTE": "a note clause (note{...})",
}


# Truncation cap for the rendered "Expected: ..." list. Some DecL
# positions can accept 30+ terminals; capping keeps the rendered error
# readable. ``ErrorReport.expected`` itself is not truncated.
_EXPECTED_RENDER_CAP = 10


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
        Raw Lark terminal names -- for tooling, not display.
    suggestions : list[str]
        "Did you mean..." candidates derived from ``got`` against the
        keyword literals in ``expected_terminals``.
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
        """JSON-serializable dict form (for tooling, logging, or transport)."""
        return asdict(self)

    @property
    def summary(self) -> str:
        """One-line summary: location + message + suggestion (no caret block).

        Used as ``args[0]`` of the wrapping :class:`ValueError`, so
        ``str(e)`` and the default Python traceback footer carry the
        useful single-line form.
        """
        head = (
            f"DecL parse error at line {self.line}, "
            f"column {self.column}: {self.message}"
        )
        if self.suggestions:
            head += " Did you mean: " + ", ".join(self.suggestions) + "?"
        return head

    def render(self) -> str:
        """Multi-line text rendering for terminals / REPL.

        Tight layout: the "Did you mean" suggestion appends to the message
        line (no blank break between them), and the "Expected" list sits
        on the same line (no blank break before it). Long source lines are
        windowed around the caret so the marker stays visible on one
        terminal row.
        """
        message = self.message
        if self.suggestions:
            message += " Did you mean: " + ", ".join(self.suggestions) + "?"
        if self.expected:
            shown = self.expected[:_EXPECTED_RENDER_CAP]
            tail = "..." if len(self.expected) > _EXPECTED_RENDER_CAP else ""
            message += " Expected: " + ", ".join(shown) + tail
        display_line, display_caret = _window_source(self.source_line, self.caret)
        lines = [
            f"DecL parse error at line {self.line}, column {self.column}:",
            "",
            f"  {display_line}",
            f"  {display_caret}",
            "",
            message,
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def format_error(source: str, exc: BaseException) -> ErrorReport:
    """Convert a Lark parse exception into an :class:`ErrorReport`.

    Accepts :class:`~lark.exceptions.UnexpectedToken`,
    :class:`~lark.exceptions.UnexpectedCharacters`,
    :class:`~lark.exceptions.UnexpectedEOF`, or any
    :class:`~lark.exceptions.UnexpectedInput` subclass. If the
    exception was wrapped by ``UnderwritingParser.parse`` into a
    :class:`ValueError` via ``raise ... from``, the original Lark
    exception is unwrapped from ``__cause__``.

    Parameters
    ----------
    source : str
        The DecL text that was parsed.
    exc : BaseException
        The exception raised. Either a Lark ``UnexpectedInput`` subclass
        or a ``ValueError`` wrapping one.

    Returns
    -------
    ErrorReport
        Structured report. For non-parse exceptions (anything that is
        not an ``UnexpectedInput``, has no ``UnexpectedInput`` in its
        ``__cause__`` chain, and carries no pre-computed ``.report``
        attribute), a minimal fallback report is returned.
    """
    # Fast path: ``UnderwritingParser.parse`` attaches the report at the
    # failure site so the rich form survives even when the cause chain
    # is suppressed with ``raise ... from None``. Honour it.
    cached = getattr(exc, "report", None)
    if isinstance(cached, ErrorReport):
        return cached
    exc = _unwrap(exc)
    if isinstance(exc, UnexpectedToken):
        return _from_token(source, exc)
    if isinstance(exc, UnexpectedCharacters):
        return _from_chars(source, exc)
    if isinstance(exc, UnexpectedEOF):
        return _from_eof(source, exc)
    if isinstance(exc, UnexpectedInput):
        return _from_generic(source, exc)
    return _fallback(source, exc)


# ----------------------------------------------------------------------
# Per-exception builders (private)
# ----------------------------------------------------------------------

def _from_token(source: str, exc: UnexpectedToken) -> ErrorReport:
    line = getattr(exc, "line", 1) or 1
    col = getattr(exc, "column", 1) or 1
    tok = exc.token
    got = str(tok)
    got_type = getattr(tok, "type", None)
    expected_raw = sorted(getattr(exc, "expected", set()) or set())
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
    line = getattr(exc, "line", 1) or 1
    col = getattr(exc, "column", 1) or 1
    pos = getattr(exc, "pos_in_stream", None)
    if pos is not None and 0 <= pos < len(source):
        char = source[pos]
        word = _extract_word(source, pos)
    else:
        char = ""
        word = ""
    expected_raw = sorted(getattr(exc, "allowed", set()) or set())
    # If we recovered an identifier-shaped word, use it for the
    # suggestion search; otherwise fall back to the raw char.
    got = word or char
    suggestions = _did_you_mean(word, expected_raw) if word else []
    width = max(1, len(word) if word else 1)
    if word and word != char:
        message = f"Unexpected {word!r}."
    else:
        message = f"Unexpected character {char!r}."
    return ErrorReport(
        line=line,
        column=col,
        source=source,
        source_line=_source_line(source, line),
        caret=_caret(col, width),
        got=got,
        got_type=None,
        expected=[_label(t) for t in expected_raw],
        expected_terminals=expected_raw,
        suggestions=suggestions,
        message=message,
    )


def _extract_word(source: str, pos: int) -> str:
    """Recover the identifier-shaped run starting at ``pos`` in ``source``.

    Empty string if the character at ``pos`` is not identifier-like
    (a punctuation mark or whitespace).
    """
    m = _WORD_CHARS.match(source, pos)
    return m.group(0) if m else ""


def _from_eof(source: str, exc: UnexpectedEOF) -> ErrorReport:
    line = source.count("\n") + 1 if source else 1
    last_line = _source_line(source, line)
    col = len(last_line) + 1
    expected_raw = sorted(getattr(exc, "expected", set()) or set())
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


def _from_generic(source: str, exc: UnexpectedInput) -> ErrorReport:
    line = getattr(exc, "line", 1) or 1
    col = getattr(exc, "column", 1) or 1
    return ErrorReport(
        line=line,
        column=col,
        source=source,
        source_line=_source_line(source, line),
        caret=_caret(col, 1),
        got=None,
        got_type=None,
        expected=[],
        expected_terminals=[],
        suggestions=[],
        message=f"Parse error: {exc}",
    )


def _fallback(source: str, exc: BaseException) -> ErrorReport:
    return ErrorReport(
        line=1,
        column=1,
        source=source,
        source_line=_source_line(source, 1),
        caret=_caret(1, 1),
        got=None,
        got_type=None,
        expected=[],
        expected_terminals=[],
        suggestions=[],
        message=f"{type(exc).__name__}: {exc}",
    )


# ----------------------------------------------------------------------
# Utilities (private)
# ----------------------------------------------------------------------

def _unwrap(exc: BaseException) -> BaseException:
    """Follow ``__cause__`` to find an ``UnexpectedInput`` if present.

    ``UnderwritingParser.parse`` wraps Lark exceptions in ``ValueError``
    via ``raise ValueError(...) from e``, so the Lark exception lives
    at ``exc.__cause__``. Walk a short chain to be safe against double
    wrapping.
    """
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, UnexpectedInput):
            return cur
        seen.add(id(cur))
        cur = cur.__cause__
    return exc


def _source_line(source: str, line: int) -> str:
    """Return the 1-indexed ``line`` from ``source``, or ``""`` if out of range."""
    lines = source.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1]
    return ""


def _caret(column: int, width: int) -> str:
    """Build a caret line for column ``column`` (1-indexed), ``width`` cols wide."""
    return " " * max(0, column - 1) + "^" * max(1, width)


# Default visible width for the windowed source line. 76 leaves room for
# the 2-space indent in ``render()`` and a small margin below an 80-col
# terminal. The "snap budget" is how many characters of window we're
# willing to give up to land the cut on a word boundary.
_WINDOW_WIDTH = 76
_SNAP_BUDGET = 15


def _window_source(line: str, caret: str, width: int = _WINDOW_WIDTH) -> tuple[str, str]:
    """Truncate a long ``line`` to a window of ``width`` chars around the caret.

    Returns a ``(line, caret)`` pair where the line fits in ``width``
    characters (plus possible ellipsis markers) and the caret is
    re-aligned to point at the same token. Truncation points snap to
    word boundaries (spaces) when possible so tokens stay intact;
    ``"... "`` / ``" ..."`` markers indicate where the line was cut.

    Short lines are returned unchanged.
    """
    if len(line) <= width:
        return line, caret
    n_carets = caret.count("^")
    caret_start = len(caret) - len(caret.lstrip(" "))
    caret_end = caret_start + n_carets
    # Center the window on the caret span, then redistribute any leftover
    # budget if we bump up against the left or right edge of the line.
    slack = max(0, width - n_carets)
    left = slack // 2
    start = caret_start - left
    end = start + width
    if start < 0:
        end -= start
        start = 0
    if end > len(line):
        start -= end - len(line)
        end = len(line)
    start = max(0, start)
    # Word-boundary snap. Don't snap if it would cost more than
    # ``_SNAP_BUDGET`` chars of context, or push the cut into the caret.
    if start > 0:
        sp = line.find(" ", start, min(start + _SNAP_BUDGET, caret_start))
        if sp != -1:
            start = sp + 1
    if end < len(line):
        sp = line.rfind(" ", max(end - _SNAP_BUDGET, caret_end), end)
        if sp != -1:
            end = sp
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(line) else ""
    windowed_line = prefix + line[start:end] + suffix
    new_caret_start = max(0, (caret_start - start) + len(prefix))
    windowed_caret = " " * new_caret_start + "^" * n_carets
    return windowed_line, windowed_caret


def _label(terminal: str | None) -> str:
    """Map a Lark terminal name to a human-friendly label.

    Known terminals come from :data:`_TERMINAL_LABELS`. Anonymous
    terminals (``__ANON_*``) collapse to ``"token"``. Anything else --
    typically a custom keyword terminal -- is rendered as its lowercase
    form in single quotes (so ``MIXED`` -> ``'mixed'``).
    """
    if terminal is None:
        return "token"
    if terminal in _TERMINAL_LABELS:
        return _TERMINAL_LABELS[terminal]
    if terminal.startswith("__"):
        return "token"
    return f"'{terminal.lower()}'"


def _did_you_mean(got: str, expected_terminals: Iterable[str]) -> list[str]:
    """Suggest close matches between the bad token and expected keywords.

    Only keyword-style terminals (all-uppercase ASCII, non-anonymous)
    are considered as candidates. Matching is case-insensitive against
    the lowercased terminal name -- a reasonable proxy for the
    underlying literal for the DecL keyword terminals in
    ``decl.lark`` (e.g. ``MIXED`` -> ``"mixed"``).
    """
    if not got:
        return []
    candidates = [
        t.lower()
        for t in expected_terminals
        if t.isupper() and t.isascii() and not t.startswith("__")
    ]
    return get_close_matches(got.lower(), candidates, n=3, cutoff=0.6)
