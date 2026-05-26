"""DecL completion + lex helpers for the editor pane.

Two endpoints feed off this module:

* ``POST /v1/decl/complete`` -- given DecL text and a cursor
  position, return the set of terminals that could legally appear
  next (keyword completions for the SPA's CodeMirror integration).
* ``POST /v1/decl/lex`` -- given DecL text, return the token stream
  produced by Lark's lexer (used by syntax-highlighting UIs that
  prefer server-side tokenization).

Both go through Lark's :meth:`Lark.parse_interactive` API, which
returns an :class:`~lark.parsers.lalr_interactive_parser.InteractiveParser`
that exposes the live parse state. ``.accepts()`` returns the set
of terminal names that would succeed at the current position --
that's the completion menu.

V1 ships keyword/terminal completions only. Knowledge-base
identifier completions (severity names, calibrated distortions)
are flagged in the plan as a v1.1 enhancement.
"""

from __future__ import annotations

from lark.exceptions import UnexpectedInput

# Reuse the parser instance already constructed by ``parser.py`` so
# we don't duplicate the (somewhat expensive) Lark.open() call.
# (Currently only used by the lex endpoint -- see notes on
# ``complete`` for why interactive parsing isn't available.)
from aggregate.parser import _PARSER

# Reuse the curated terminal -> label table from Plan B so suggestion
# labels are consistent with parse-error displays.
from aggregate.parser_errors import _TERMINAL_LABELS


# ----------------------------------------------------------------------
# Completion classification
# ----------------------------------------------------------------------
# Terminals that represent generic value categories rather than fixed
# strings. Used to tag a Completion's ``kind`` so the editor can render
# (e.g.) keyword pills differently from "expects a number" hints.
_LITERAL_TERMINALS = {
    "NUMBER", "SIGNED_NUMBER", "INT", "SIGNED_INT", "FLOAT",
    "STRING", "ESCAPED_STRING",
}
_IDENTIFIER_TERMINALS = {"NAME", "CNAME", "ID"}


def _classify(terminal: str) -> str:
    """Map a terminal name to ``'keyword' | 'identifier' | 'literal'``.

    Anything not in the explicit identifier/literal sets is treated
    as a keyword (it's either a curated DecL keyword from
    ``decl.lark`` or an anonymous terminal -- the editor renders
    both as "keyword" tokens).
    """
    if terminal in _LITERAL_TERMINALS:
        return "literal"
    if terminal in _IDENTIFIER_TERMINALS:
        return "identifier"
    return "keyword"


def _label(terminal: str) -> str:
    """Human label for ``terminal``.

    Pulled from Plan B's curated table when available, otherwise
    the terminal's lowercased name in single quotes. Anonymous
    terminals (``__ANON_*``) collapse to the empty string -- the
    editor filters those out.
    """
    if terminal in _TERMINAL_LABELS:
        # Strip the surrounding quotes on keyword labels because the
        # editor wants the bare token to insert at the cursor; the
        # label is mostly for tooltip/menu display.
        return _TERMINAL_LABELS[terminal].strip("'")
    if terminal.startswith("__"):
        return ""
    return terminal.lower()


# Lark's ``parse_interactive`` is LALR-only; DecL uses Earley + dynamic
# lexer, so we can't ask the live parser "what accepts next?". V1 ships
# a static keyword set sourced from Plan B's terminal-label table --
# good enough for the SPA's "show me available keywords" dropdown but
# not context-aware. Grammar-driven completions are flagged in the plan
# as a v1.1 enhancement (would require building a parallel LALR
# grammar or stepping through token-by-token).
_STATIC_KEYWORDS = sorted(
    {
        t for t in _TERMINAL_LABELS
        if t.isupper() and t.isascii() and not t.startswith("__")
    }
)


def complete(decl: str, cursor: int) -> list[dict]:
    """Return completion candidates for the given cursor position.

    V1 implementation: filter the static keyword pool against the
    identifier-shaped word ending at ``cursor`` (case-insensitive
    prefix match). When the cursor sits on whitespace / start of
    input, the full keyword pool is returned.

    Parameters
    ----------
    decl : str
        Current editor content.
    cursor : int
        Zero-indexed character position.

    Returns
    -------
    list[dict]
        Sorted list of ``{label, terminal, kind}`` dicts. Empty only
        when no keyword starts with the current prefix.
    """
    prefix = decl[:cursor]
    # Identify the word the cursor is currently inside (or at the
    # end of). Walk backwards while we still see identifier chars.
    i = len(prefix)
    while i > 0 and (prefix[i - 1].isalnum() or prefix[i - 1] in "._-:~"):
        i -= 1
    word = prefix[i:].lower()

    out: list[dict] = []
    for term in _STATIC_KEYWORDS:
        label = _label(term)
        if not label:
            continue
        if word and not label.lower().startswith(word):
            continue
        out.append({"label": label, "terminal": term, "kind": _classify(term)})
    return out


def lex(decl: str) -> list[dict]:
    """Tokenize ``decl`` via Lark's lexer and return token records.

    Returns one dict per token::

        {"type": "MIXED", "value": "mixed", "start": 7, "end": 12,
         "line": 1, "column": 8}

    A failure to tokenize (e.g. an unexpected character partway
    through) surfaces an empty list -- callers should hit
    ``/v1/objects`` for a proper :class:`ErrorReport`.
    """
    try:
        tokens = list(_PARSER.lex(decl))
    except UnexpectedInput:
        return []
    except Exception:
        return []
    out: list[dict] = []
    for tok in tokens:
        out.append({
            "type": tok.type,
            "value": str(tok),
            "start": tok.start_pos or 0,
            "end": tok.end_pos or 0,
            "line": tok.line or 1,
            "column": tok.column or 1,
        })
    return out
