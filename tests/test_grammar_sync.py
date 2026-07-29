"""Keep the hand-maintained DecL mirrors in sync with ``decl.lark``.

``decl.lark`` is the single source of truth for the DecL keywords / terminals
(CLAUDE.md: "The DecL grammar — single source of truth for the language").
Several artefacts mirror that keyword set by hand and silently drift when a new
keyword is added to the grammar:

- :class:`aggregate.decl_pygments.AggLexer` — the Pygments syntax colorer.
- ``agg.sublime-syntax`` — the Sublime Text colorer at the repo root.
- :data:`aggregate.parser_errors._TERMINAL_LABELS` — the human-readable
  terminal labels used in parse-error messages.

These tests derive the canonical keyword set directly from the grammar and fail
if any mirror falls behind, so adding a keyword to ``decl.lark`` forces the
mirrors to be updated in the same change.

Reserved words are only half the surface, and the smaller half. The trailer
clauses (``note`` / ``tags`` / ``hints`` / ``doc``) are deliberately **absent**
from the grammar's ``ID`` exclusion list, because their terminals include the
opening brace and so the bare word is a legal identifier. Neither ``STRING``
nor the operator terminals carry the ``.N`` priority tag either. A reserved-word
walk therefore cannot see any of them, which is how ``tags{...}`` and
``doc{{{...}}}`` went uncolored for several releases while this suite stayed
green. The corpus tests below close that hole: they tokenize every shipped
``.agg`` file and assert zero ``Token.Error``, which no construct can escape.

The web app's ``decl-keywords.json`` lives in a separate repository and is not
checked here.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from pygments.token import Token

from aggregate.parser import GRAMMAR_FILE
from aggregate.decl_pygments import AggLexer
from aggregate.parser_errors import _TERMINAL_LABELS

#: The shipped DecL corpora, tokenized whole by the corpus tests below.
CORPORA = ('library.agg', '_test_suite.agg', '_test_suite2.agg',
           'decl-testers.agg')

#: The Sublime Text colorer, the second hand-maintained mirror.
SUBLIME_FILE = Path(__file__).parents[1] / 'agg.sublime-syntax'


def _grammar_text() -> str:
    return GRAMMAR_FILE.read_text(encoding="utf-8")


def _corpus_text(name: str) -> str:
    return (GRAMMAR_FILE.parent / 'agg' / name).read_text(encoding="utf-8")


def _tokens(source: str) -> list:
    """Tokenize ``source`` with a fresh lexer, returning ``(type, value)``."""
    return list(AggLexer().get_tokens(source))


def _significant(source: str) -> list:
    """Tokens for ``source``, dropping pure whitespace. For readable asserts."""
    return [(t, v) for t, v in _tokens(source) if v.strip()]


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


def grammar_brace_clauses() -> list[str]:
    """The trailer clause words, read from the ``word\\{`` terminals.

    Derived rather than hard-coded so that a *fifth* brace clause added to
    ``decl.lark`` fails this suite instead of silently going uncolored, which
    is exactly how ``tags`` and ``doc`` were missed.
    """
    found = re.findall(r"^([A-Z]+)\.\d+:\s*/([a-z]+)\\\{", _grammar_text(),
                       re.MULTILINE)
    return sorted({word for _terminal, word in found})


def grammar_operator_literals() -> list[str]:
    """Every quoted operator literal, e.g. ``**``, ``^``, ``@``, ``:``.

    Reads the untagged string-alternation terminals (``EXPONENT: "**" | "^"``),
    which carry no ``.N`` priority and so are invisible to
    :func:`grammar_keyword_terminals`.
    """
    literals = []
    for line in _grammar_text().splitlines():
        m = re.match(r'^[A-Z][A-Z0-9_]*:\s+((?:"[^"]*"\s*\|?\s*)+)$', line)
        if m:
            literals.extend(re.findall(r'"([^"]*)"', m.group(1)))
    return sorted(set(literals))


def test_grammar_reserved_words_extraction_sane():
    """Guard the extraction itself: a few known keywords must be present."""
    words = set(grammar_reserved_words())
    for expected in ("agg", "poisson", "bivariate", "dbvsev", "pnl", "zt"):
        assert expected in words, f"{expected!r} missing — extraction likely broke"


def test_grammar_brace_clause_extraction_sane():
    """The grammar has exactly four trailer clauses today."""
    assert grammar_brace_clauses() == ["doc", "hints", "note", "tags"]


def test_grammar_operator_extraction_sane():
    """The operator literals include the ones with no priority tag."""
    literals = set(grammar_operator_literals())
    for expected in ("**", "^", "@", "=", ":", "+", "-", "*", "/"):
        assert expected in literals, f"{expected!r} missing — extraction likely broke"


@pytest.mark.parametrize(
    "tail", ["", " ", "\n", " x", ","],
    ids=["eof", "space", "newline", "word", "comma"],
)
@pytest.mark.parametrize("kw", grammar_reserved_words())
def test_colorizer_colours_every_grammar_keyword(kw, tail):
    """Every grammar reserved word is colored by :class:`AggLexer`.

    A word the lexer does not recognise falls through to the catch-all ID rule
    (bare ``Token.Name``) or fails to match (``Token.Error``). Either means the
    colorer has drifted from the grammar — add the keyword to ``AggLexer``.

    Parametrized over what follows the word. An earlier version of this test
    probed ``kw + " "`` only, which hid a real defect: the lexer's rule for
    ``mixed`` was the literal string ``'mixed '``, so ``mixed`` at the end of a
    line (legal, since a newline inside a statement is whitespace) fell through
    to ``Token.Name`` while this test stayed green.
    """
    first_type, _value = next(iter(AggLexer().get_tokens(kw + tail)))
    assert first_type not in (Token.Name, Token.Error), (
        f"{kw!r} followed by {tail!r} is not colored by AggLexer "
        f"(got {first_type}); add it to aggregate/decl_pygments.py"
    )


@pytest.mark.parametrize("terminal", grammar_keyword_terminals())
def test_terminal_labels_cover_every_keyword_terminal(terminal):
    """Every priority-tagged grammar terminal has an error-message label."""
    assert terminal in _TERMINAL_LABELS, (
        f"{terminal!r} has no entry in parser_errors._TERMINAL_LABELS; "
        f"add a human-readable label"
    )


@pytest.mark.parametrize("corpus", CORPORA)
def test_corpus_has_no_error_token(corpus):
    """No shipped DecL corpus produces a single ``Token.Error``.

    The strongest guard available: it exercises every construct the language
    actually ships with, in situ, and so catches the whole class of drift the
    reserved-word walk structurally cannot see. ``Token.Error`` is left live
    rather than swallowed by a catch-all rule precisely so that it can act as
    this signal; the ``friendly`` style draws it with a red border.
    """
    source = _corpus_text(corpus)
    errors = [(i, v) for i, (t, v) in enumerate(_tokens(source))
              if t is Token.Error]
    assert not errors, (
        f"{corpus}: {len(errors)} Token.Error tokens, first few "
        f"{errors[:8]}; AggLexer has drifted from decl.lark"
    )


@pytest.mark.parametrize("corpus", CORPORA)
def test_corpus_round_trips_losslessly(corpus):
    """Concatenating the token values reproduces the source byte for byte.

    A lexer that drops or duplicates text is broken whatever its colors, and a
    callback rule (the ``doc{{{...}}}`` fence delegates its body to the
    Markdown lexer) is exactly where that bug hides.
    """
    source = _corpus_text(corpus)
    rendered = ''.join(v for _t, v in _tokens(source))
    # get_tokens normalizes the trailing newline; compare without it.
    assert rendered.rstrip('\n') == source.rstrip('\n'), (
        f"{corpus}: AggLexer did not round-trip the source"
    )


@pytest.mark.parametrize("word", grammar_brace_clauses())
def test_colorizer_handles_every_brace_clause(word):
    """Each ``word{...}`` trailer clause has a rule in :class:`AggLexer`.

    Derived from the grammar, so a fifth clause fails here. Without this the
    only signal is the corpus test, and only once a corpus actually uses it.
    """
    snippet = 'doc{{{abc}}}' if word == 'doc' else f'{word}{{x}}'
    first_type, _value = next(iter(AggLexer().get_tokens(snippet)))
    assert first_type is Token.Comment.Preproc, (
        f"{word}{{...}} opens with {first_type}, not Comment.Preproc; "
        f"add a rule to aggregate/decl_pygments.py"
    )


@pytest.mark.parametrize("literal", grammar_operator_literals())
def test_colorizer_colours_every_operator_literal(literal):
    """Every quoted operator terminal lexes to one non-error token."""
    tokens = _significant(literal)
    assert len(tokens) == 1, f"{literal!r} lexed to {tokens}, expected one token"
    token_type, _value = tokens[0]
    assert token_type in (Token.Operator, Token.Punctuation), (
        f"{literal!r} lexed as {token_type}, expected Operator or Punctuation"
    )


#: ``(source, expected substring-to-token mapping)`` for the constructs neither
#: the reserved-word walk nor the grammar-derived extractions can express.
CONSTRUCTS = [
    ('as "My Label"', '"My Label"', Token.Literal.String),
    ('tags{role:hero}', 'role:hero', Token.Name.Tag),
    ('hints{bs=1/64}', 'bs', Token.Name.Attribute),
    ('hints{normalize=False}', 'False', Token.Keyword.Constant),
    ('# a comment', '# a comment', Token.Comment.Single),
    ('// a comment', '// a comment', Token.Comment.Single),
    ('2 @ agg.Base', '@', Token.Operator),
    ('2 @ agg.Base', 'agg.Base', Token.Name.Builtin),
    ('dist.A', 'dist.A', Token.Name.Builtin),
    ('distortion.B', 'distortion.B', Token.Name.Builtin),
    ('port.P', 'port.P', Token.Name.Builtin),
    ('1_000_000', '1_000_000', Token.Literal.Number),
    ('50%', '50%', Token.Literal.Number),
    ('-inf', '-inf', Token.Literal.Number),
    ('loss-ratio', 'loss-ratio', Token.Name),
    ('no-claims', 'no-claims', Token.Name),
    ('infinity', 'infinity', Token.Name),
    ('agg X 10 claims mixed\ngamma 0.5', 'mixed', Token.Keyword),
    ('mixed sichel.gamma 0.5', 'sichel.gamma', Token.Name.Function),
    ('dfreq [1 2]', 'dfreq', Token.Name.Label),
    ('dsev [1 2]', 'dsev', Token.Name.Label),
    ('sev dhistogram xps [0 99]', 'dhistogram', Token.Name.Class),
]


@pytest.mark.parametrize(
    "source,value,expected", CONSTRUCTS,
    ids=[f"{v}-{str(e).replace('Token.', '')}" for _s, v, e in CONSTRUCTS],
)
def test_construct_is_coloured(source, value, expected):
    """Named constructs get the intended token, and none of them errors."""
    tokens = _significant(source)
    assert not any(t is Token.Error for t, _v in tokens), (
        f"{source!r} produced Token.Error: {tokens}"
    )
    assert (expected, value) in tokens, (
        f"{value!r} in {source!r} did not lex as {expected}; got {tokens}"
    )


def test_a_stray_backslash_is_still_an_error():
    """Backslash continuation is gone from the language, so ``\\`` must error.

    Pins the deliberate absence of a catch-all rule in ``AggLexer``. If one is
    ever added, :func:`test_corpus_has_no_error_token` becomes vacuous and this
    is the test that notices.
    """
    first_type, _value = next(iter(AggLexer().get_tokens('\\ x')))
    assert first_type is Token.Error


def test_markdown_lexer_is_not_imported_by_aggregate():
    """``import aggregate`` must not pull in ``pygments.lexers.markup``.

    That module costs roughly 110 ms to import and is needed only to color a
    ``doc{{{...}}}`` body, so ``decl_pygments`` defers it into the fence
    callback. ``decl_pygments`` is star-imported by ``aggregate/__init__.py``,
    so a module-level import would put the cost on every user. Run in a
    subprocess because pytest's own imports pollute ``sys.modules``.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import aggregate, sys; print('pygments.lexers.markup' in sys.modules)"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "False", (
        "importing aggregate pulled in pygments.lexers.markup; keep the "
        "Markdown import inside the doc-fence callback"
    )


def test_doc_fence_body_is_lexed_as_markdown():
    """A ``doc{{{...}}}`` body highlights as markdown, not as DecL."""
    source = 'doc{{{\n# Heading\nUse `build()` here.\n}}}'
    tokens = _significant(source)
    assert (Token.Generic.Heading, '# Heading') in tokens
    assert (Token.Literal.String.Backtick, '`build()`') in tokens


def sublime_reserved_words() -> set[str]:
    """Every reserved word named in ``agg.sublime-syntax``.

    Reads both the ``{{keyword_boundary}}`` alternations and the ID-exclusion
    lookahead, which together are that file's copy of the grammar's word list.
    """
    text = SUBLIME_FILE.read_text(encoding="utf-8")
    words: set[str] = set()
    for m in re.finditer(r"\(\?:([a-zA-Z0-9_|]+)\)\{\{keyword_boundary\}\}", text):
        words |= set(m.group(1).split("|"))
    for m in re.finditer(r"\(\?!\(\?:([a-zA-Z0-9_|]+)\)\(\?!", text):
        words |= set(m.group(1).split("|"))
    return words


def test_sublime_syntax_matches_grammar_reserved_words():
    """``agg.sublime-syntax`` names exactly the grammar's reserved words.

    The second hand-maintained mirror, and the one with no other guard. It had
    drifted by 33 missing and 2 stale words before this test existed.
    """
    grammar = set(grammar_reserved_words())
    sublime = sublime_reserved_words()
    missing = sorted(grammar - sublime)
    stale = sorted(sublime - grammar)
    assert not missing and not stale, (
        f"agg.sublime-syntax has drifted from decl.lark; "
        f"missing {missing}, stale {stale}"
    )
