"""
Pygments lexer for the DecL language.

``AggLexer`` mirrors ``decl.lark``, which is the single source of truth for the
language. Pygments resolves it globally through the ``pygments.lexers`` entry
point in ``pyproject.toml``, so ``:language: agg`` in the docs, ``pygmentize``
on a ``.agg`` file, and :func:`aggregate.decl_writer._colorize` (behind
``format_program`` and ``pprogram_html``) all reach the same class.

Its words come from :func:`aggregate.decl_words.decl_words`, the one DecL word
list every colorizer reads, so a keyword added to the grammar is colored here
with no edit. The rest of the rule set (comments, the trailer clause bodies,
labels, numbers, operators) is written by hand and can drift from the grammar,
and silently: an unmatched construct simply falls through to the catch-all ID
rule or emits ``Token.Error``. ``tests/test_grammar_sync.py`` is the guard. It
walks every reserved word in the grammar's ``ID`` exclusion list, checks the
constructs that list cannot reach (the trailer clauses, quoted labels,
comments, operators), and tokenizes the whole shipped ``.agg`` corpus asserting
zero ``Token.Error``.

Notes
-----
Two structural facts about DecL drive the rule set and are easy to get wrong.

First, the word boundary. ``.``, ``_``, ``:``, ``~`` and ``-`` are all name
characters in the grammar's ``ID`` terminal, so ``\\b`` is the wrong boundary:
it splits ``loss-ratio`` into ``loss``, ``-``, ``ratio``. Every keyword rule
here uses :data:`_KW`, the grammar's own negative lookahead, instead.

Second, comments never reach the Lark lexer in source form:
``UnderwritingLexer.preprocess`` strips ``#`` and ``//`` before parsing. This
lexer runs on the text a reader sees, so it handles both comment markers
itself.
"""

from pygments.lexer import RegexLexer, default, include, words
from pygments.token import (Comment, Generic, Keyword, Name, Number, Operator,
                            Punctuation, String, Text)

from .decl_words import decl_words

__all__ = ['AggLexer']


#: The grammar's own word boundary, copied from the ``ID`` terminal in
#: ``decl.lark``. Used as the ``words()`` suffix everywhere a keyword is
#: matched, so this lexer and the parser agree by construction on where a word
#: ends.
_KW = r'(?![a-zA-Z0-9._:~\-])'

#: An identifier, mirroring the tail of the grammar's ``ID`` terminal.
_ID = r'[a-zA-Z][\._:~a-zA-Z0-9\-]*'

_WORDS = decl_words()

#: The token type for a severity name, by its shape parameter count. ``None``
#: is the empirical ``dhistogram`` and ``chistogram``, colored with the one
#: shape names as they always have been.
_SEVERITY_TOKENS = {0: Name.Function, 1: Name.Class, 2: Name.Namespace,
                    None: Name.Class}


def _severity_rule(count):
    """The ``words()`` rule for the severity names with ``count`` shapes."""
    return (words(tuple(name for name, n in _WORDS['severities'].items()
                        if n == count), suffix=_KW),
            _SEVERITY_TOKENS[count])


#: Every reserved word without a color of its own. ``mixed`` is absent because
#: the root state colors it and opens the mixing distribution state; the
#: frequency names and the declaration keywords have their own rules.
_KEYWORDS = tuple(sorted(set(_WORDS['reserved']) - set(_WORDS['frequencies'])
                         - set(_WORDS['declarations']) - {'mixed'}))


class AggLexer(RegexLexer):
    """Syntax highlighter for DecL, the aggregate declaration language.

    Mirrors the terminals of ``decl.lark``. See the module docstring for the
    drift guard and for the two structural traps (the word boundary, and the
    constructs the preprocessor rewrites before parsing).
    """

    name = 'Aggregate'
    aliases = ['aggregate', 'agg', 'decl', 'dec']
    filenames = ['*.agg', '*.dec', '*.decl']
    url = 'http://www.github.com/mynl/aggregate'

    mimetypes = ['text/x-agg', 'text/x-aggregate', 'text/x-decl',
                 'text/x-dec']

    tokens = {

        'root': [
            (r'\n', Text),

            # Both comment markers. `//` before the operators, which own `/`.
            (r'//.*$', Comment.Single),
            (r'#.*$', Comment.Single),

            # The rest of the trailer. Each body has its own state because the
            # three read differently: prose, slugs, key=value settings.
            (r'note\{', Comment.Preproc, 'note'),
            (r'tags\{', Comment.Preproc, 'tags'),
            (r'hints\{', Comment.Preproc, 'hints'),

            # A quoted display label. No escapes and no embedded newline, as in
            # the grammar's STRING terminal.
            (r'"[^"\n]*"', String),

            # agg.X / sev.X / port.X / dist.X / distortion.X.
            (r'(?:agg|sev|port|distortion|dist)\.[a-zA-Z][a-zA-Z0-9._:~\-]*',
             Name.Builtin),

            # `mixed` opens the mixing-distribution state. This must precede
            # the keyword list: put it after and the plain keyword rule wins,
            # the state never opens, and the mixing distribution loses its
            # color.
            (r'mixed' + _KW, Keyword, 'mixed_freq'),

            # The semantic marker: unconditional severity, zero-modified mean
            # pin, defective dwait.
            (r'!', Operator),

            # Metavariable placeholders emitted by the help renderer.
            (r'<[A-Z_0-9*]+>', Generic.Heading),

            # Numbers before operators, so a leading minus is absorbed into the
            # token exactly as the grammar's priority-2 NUMBER does.
            include('numbers'),
            include('keywords'),
            include('operators'),

            (_ID, Name),
        ],

        # Prose. `[^}]+` rather than `[^}]*` so an empty note{} falls through
        # to the closing brace instead of matching zero width forever.
        'note': [
            (r'[^}]+', Comment),
            (r'\}', Comment.Preproc, '#pop'),
        ],

        # Slugs, split by the parser on [,\s]+, so anything that is not a
        # separator or the closing brace is one slug.
        'tags': [
            (r'[^\s,}]+', Name.Tag),
            (r'[,\s]+', Text),
            (r'\}', Comment.Preproc, '#pop'),
        ],

        # `key=value;` build settings, e.g. hints{bs=1/64; log2=10}. A value
        # may be a per-axis pair, e.g. hints{log2=(9,12)}, hence the comma.
        'hints': [
            (r'\}', Comment.Preproc, '#pop'),
            (_ID + r'(?=\s*=)', Name.Attribute),
            (r'[=;,]', Punctuation),
            (r'(?:True|False|None)' + _KW, Keyword.Constant),
            include('numbers'),
            (r'[-+*/()]', Operator),
            (r'[^\S\n]+', Text),
            (_ID, Name),
        ],

        'mixed_freq': [
            # `\s+`, not `[^\S\n]+`: every newline that is not a paragraph
            # break is whitespace inside a statement, so `mixed` and its
            # mixing distribution may sit on separate lines.
            (r'\s+', Text),
            # The mixing distributions, and the help renderer's placeholder.
            (words(_WORDS['mixing'] + ('<DISTRIBUTION>',), suffix=_KW),
             Name.Function, '#pop'),
            # Never strand the state on an unrecognized follower.
            default('#pop'),
        ],

        # One rule per NUMBER alternative in the grammar. Digit runs are
        # \d(?:_?\d)* so Python underscore group separators (1_000_000) stay a
        # single token; the trailing % marks a share rather than an amount.
        #
        # The numeric form carries no trailing guard, matching the grammar: a
        # number is legitimately followed by a name character in `[1:6]` and
        # `[10:50:10]`, where `:` is the RANGE operator. Only the `inf` spelling
        # needs one, so that `infinity` stays a single identifier.
        'numbers': [
            (r'-?(?:\d(?:_?\d)*\.?(?:\d(?:_?\d)*)?|\.\d(?:_?\d)*)'
             r'(?:[eE][+\-]?\d(?:_?\d)*)?%?', Number),
            (r'-?inf' + _KW, Number),
        ],

        'keywords': [
            # Frequency distributions (the grammar's FREQ terminal).
            (words(_WORDS['frequencies'], suffix=_KW), Name.Function),

            # Named severity distributions, colored by shape parameter count.
            # They arrive through the grammar's `ids` rule as a plain ID, so
            # the empirical `dhistogram` and `chistogram` are names too, not
            # declaration keywords.
            _severity_rule(0),
            _severity_rule(1),
            _severity_rule(2),
            _severity_rule(None),

            # The discrete declaration keywords. `dfreq` belongs here with its
            # siblings rather than with the FREQ distribution names: it
            # declares an outcome-and-probability pair, it is not a named
            # distribution.
            (words(_WORDS['declarations'], suffix=_KW), Name.Label),

            # Every remaining reserved word. `inf` is not reserved; the number
            # rules claim it, as the grammar does.
            (words(_KEYWORDS, suffix=_KW), Keyword),
        ],

        # Exactly the grammar's operator and punctuation set. Comma and pipe
        # are whitespace to the parser (%ignore), so they are Text here.
        'operators': [
            (r'[^\S\n]+', Text),
            (r'\*\*|\^', Operator),
            (r'[-+*/@=]', Operator),
            (r'[][():]', Punctuation),
            (r'[,|]', Text),
            (r';', Punctuation),
        ],
    }
