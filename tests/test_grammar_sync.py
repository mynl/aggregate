"""Keep the hand-maintained DecL mirrors in sync with ``decl.lark``.

``decl.lark`` is the single source of truth for the DecL keywords / terminals
(CLAUDE.md: "The DecL grammar — single source of truth for the language").
Several artefacts mirror that keyword set by hand and silently drift when a new
keyword is added to the grammar (this is TODO ``D10``):

- :class:`aggregate.decl_pygments.AggLexer` — the Pygments syntax colourer.
- :data:`aggregate.parser_errors._TERMINAL_LABELS` — the human-readable
  terminal labels used in parse-error messages.

These tests derive the canonical keyword set directly from the grammar and fail
if either mirror falls behind, so adding a keyword to ``decl.lark`` forces the
mirrors to be updated in the same change.

The web app's ``decl-keywords.json`` lives in a separate repository and is not
checked here.
"""
from __future__ import annotations

import re

import pytest
from pygments.token import Token

from aggregate.parser import GRAMMAR_FILE
from aggregate.decl_pygments import AggLexer
from aggregate.parser_errors import _TERMINAL_LABELS


def _grammar_text() -> str:
    return GRAMMAR_FILE.read_text(encoding="utf-8")


def grammar_reserved_words() -> list[str]:
    """Every reserved word, read from the ``ID`` terminal's exclusion list.

    The ``ID`` terminal in ``decl.lark`` rejects bare keywords via a single
    negative-lookahead alternation — the authoritative flat list of every
    reserved word in the language::

        ID: /(?!agg\\.|...)(?!(?:agg|aggregate|...|zt)(?![namechar])).../
    """
    id_line = next(
        line for line in _grammar_text().splitlines() if line.lstrip().startswith("ID:")
    )
    m = re.search(r"\(\?!\(\?:([a-zA-Z0-9_|]+)\)\(\?!", id_line)
    assert m, "could not find the ID-exclusion keyword list in decl.lark"
    return sorted(set(m.group(1).split("|")))


def grammar_keyword_terminals() -> list[str]:
    """Every priority-tagged terminal name (``NAME.2:`` / ``NAME.3:``).

    These are the keyword / builtin / NUMBER / NOTE / HINTS terminals — the
    ones a parse error can name and therefore wants a friendly label for.
    """
    names = re.findall(r"^([A-Z][A-Z0-9_]*)\.\d+\s*:", _grammar_text(), re.MULTILINE)
    return sorted(set(names))


def test_grammar_reserved_words_extraction_sane():
    """Guard the extraction itself: a few known keywords must be present."""
    words = set(grammar_reserved_words())
    for expected in ("agg", "poisson", "bivariate", "dbvsev", "pnl", "zt"):
        assert expected in words, f"{expected!r} missing — extraction likely broke"


@pytest.mark.parametrize("kw", grammar_reserved_words())
def test_colorizer_colours_every_grammar_keyword(kw):
    """Every grammar reserved word is coloured by :class:`AggLexer`.

    A word the lexer does not recognise falls through to the catch-all ID rule
    (bare ``Token.Name``) or fails to match (``Token.Error``). Either means the
    colourer has drifted from the grammar — add the keyword to ``AggLexer``.
    """
    # Trailing space so word-boundary / state-switch rules (e.g. ``mixed ``)
    # match exactly as they do mid-program.
    first_type, _value = next(iter(AggLexer().get_tokens(kw + " ")))
    assert first_type not in (Token.Name, Token.Error), (
        f"{kw!r} is not coloured by AggLexer (got {first_type}); "
        f"add it to aggregate/decl_pygments.py"
    )


@pytest.mark.parametrize("terminal", grammar_keyword_terminals())
def test_terminal_labels_cover_every_keyword_terminal(terminal):
    """Every priority-tagged grammar terminal has an error-message label."""
    assert terminal in _TERMINAL_LABELS, (
        f"{terminal!r} has no entry in parser_errors._TERMINAL_LABELS; "
        f"add a human-readable label"
    )
