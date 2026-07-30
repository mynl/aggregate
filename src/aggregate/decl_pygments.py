"""
Pygments lexer for the DecL language.

``AggLexer`` is a hand-written mirror of ``decl.lark``, which is the single
source of truth for the language. Pygments resolves it globally through the
``pygments.lexers`` entry point in ``pyproject.toml``, so ``:language: agg`` in
the docs, ``pygmentize`` on a ``.agg`` file, and
:func:`aggregate.decl_writer._colorize` (behind ``format_program`` and
``pprogram_html``) all reach the same class.

Because the mirror is hand-written it can drift from the grammar, and silently:
an unmatched construct simply falls through to the catch-all ID rule or emits
``Token.Error``. ``tests/test_grammar_sync.py`` is the guard. It walks every
reserved word in the grammar's ``ID`` exclusion list, checks the constructs that
list cannot reach (the trailer clauses, quoted labels, comments, operators), and
tokenizes the whole shipped ``.agg`` corpus asserting zero ``Token.Error``. Add
a keyword or a clause to ``decl.lark`` and that suite fails until this file
follows.

Notes
-----
Two structural facts about DecL drive the rule set and are easy to get wrong.

First, the word boundary. ``.``, ``_``, ``:``, ``~`` and ``-`` are all name
characters in the grammar's ``ID`` terminal, so ``\\b`` is the wrong boundary:
it splits ``loss-ratio`` into ``loss``, ``-``, ``ratio``. Every keyword rule
here uses :data:`_KW`, the grammar's own negative lookahead, instead.

Second, comments and the ``doc`` fence never reach the Lark lexer in source
form. ``UnderwritingLexer.preprocess`` strips ``#`` and ``//`` comments and
base64-encodes ``doc{{{...}}}`` bodies before parsing. This lexer runs on the
text a reader sees, so it handles both comment markers and both doc forms: the
multi-line fence carrying readable markdown (what a ``.agg`` file holds, and
what ``format_program(trailer=True)`` emits) and the encoded single-line form
(what ``.program`` stores).
"""

from pygments.lexer import RegexLexer, bygroups, default, include, words
from pygments.token import (Comment, Generic, Keyword, Name, Number, Operator,
                            Punctuation, String, Text)

__all__ = ['AggLexer']


#: The grammar's own word boundary, copied from the ``ID`` terminal in
#: ``decl.lark``. Used as the ``words()`` suffix everywhere a keyword is
#: matched, so this lexer and the parser agree by construction on where a word
#: ends.
_KW = r'(?![a-zA-Z0-9._:~\-])'

#: An identifier, mirroring the tail of the grammar's ``ID`` terminal.
_ID = r'[a-zA-Z][\._:~a-zA-Z0-9\-]*'


def _doc_fence(lexer, match):
    """Tokenize a ``doc{{{ ... }}}`` fence, highlighting the body as markdown.

    A doc body is long-form prose: headings, blank lines, inline code and
    fenced code blocks. Delegating it to Pygments' Markdown lexer reads far
    better than one flat comment, and it costs nothing at import time because
    ``pygments.lexers.markup`` is imported here rather than at module scope. It
    takes roughly 110 ms to load, and ``import aggregate`` pulls this module in
    eagerly, so the import stays inside the callback and fires only when a
    program actually carries a doc body.

    Parameters
    ----------
    lexer : RegexLexer
        The lexer instance, unused. Part of the Pygments callback signature.
    match : re.Match
        Groups are the opening fence, the body, and the closing fence.

    Yields
    ------
    tuple of (int, TokenType, str)
        Absolute index, token type, value. The nested lexer numbers its tokens
        from zero, so each index is offset by the body's start.
    """
    from pygments.lexers.markup import MarkdownLexer

    yield match.start(1), Comment.Preproc, match.group(1)
    for index, token, value in MarkdownLexer().get_tokens_unprocessed(match.group(2)):
        yield match.start(2) + index, token, value
    yield match.start(3), Comment.Preproc, match.group(3)


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

            # `doc{{{...}}}`, both forms. The fenced form spans lines, so the
            # body class is [\s\S] rather than `.`: RegexLexer runs under
            # re.MULTILINE only, where `.` stops at a newline. Non-greedy so
            # the first closing fence wins, matching parser._DOC_FENCE_RE.
            # These come first: the body may open with a `#` heading, which the
            # comment rule below would otherwise eat.
            (r'(doc\{\{\{[ \t]*\n)([\s\S]*?)(\n[ \t]*\}\}\})', _doc_fence),
            (r'(doc\{\{\{)([A-Za-z0-9_=-]*)(\}\}\})',
             bygroups(Comment.Preproc, Comment, Comment.Preproc)),

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

        # `key=value;` build settings, e.g. hints{bs=1/64; log2=10}.
        'hints': [
            (r'\}', Comment.Preproc, '#pop'),
            (_ID + r'(?=\s*=)', Name.Attribute),
            (r'[=;]', Punctuation),
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
            (r'sichel\.(?:gamma|ig)' + _KW, Name.Function, '#pop'),
            (words(('gamma', 'delaporte', 'ig', 'sig', 'beta', 'sichel',
                    '<DISTRIBUTION>'),
                   suffix=_KW), Name.Function, '#pop'),
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
            (words(
                ('binomial', 'pascal', 'poisson', 'bernoulli', 'geometric',
                 'fixed', 'neyman', 'neymana', 'neymanA', 'logarithmic',
                 'negbin'),
                suffix=_KW
            ), Name.Function),

            # scipy.stats severity distributions, by shape-parameter count.
            # Zero parameter.
            (words((
                'anglit', 'arcsine', 'cauchy', 'cosine', 'expon', 'gilbrat',
                'gumbel_l', 'gumbel_r', 'halfcauchy', 'halflogistic',
                'halfnorm', 'hypsecant', 'kstwobign', 'laplace', 'levy',
                'levy_l', 'logistic', 'maxwell', 'moyal', 'norm', 'rayleigh',
                'semicircular', 'uniform', 'wald'
            ), suffix=_KW), Name.Function),
            # One parameter.
            (words((
                'alpha', 'argus', 'bradford', 'chi', 'chi2', 'dgamma', 'dweibull', 'erlang',
                'exponnorm', 'exponpow', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm',
                'gamma', 'genextreme', 'genhalflogistic', 'genlogistic', 'gennorm', 'genpareto',
                'gompertz', 'halfgennorm', 'invgamma', 'invgauss', 'invweibull', 'kappa3', 'ksone',
                'kstwo', 'laplace_asymmetric', 'loggamma', 'loglaplace', 'lognorm', 'lomax',
                'nakagami', 'pareto', 'pearson3', 'powerlaw', 'powernorm', 'rdist', 'recipinvgauss',
                'rice', 'skewcauchy', 'skewnorm', 't', 'triang', 'truncexpon', 'tukeylambda',
                'vonmises', 'vonmises_line', 'weibull_max', 'weibull_min', 'wrapcauchy'
            ), suffix=_KW), Name.Class),
            # Two parameter.
            (words((
                'beta', 'betaprime', 'burr', 'burr12', 'crystalball', 'exponweib', 'f', 'gengamma',
                'geninvgauss', 'johnsonsb', 'johnsonsu', 'kappa4', 'levy_stable', 'loguniform',
                'mielke', 'nct', 'ncx2', 'norminvgauss', 'powerlognorm', 'reciprocal',
                'studentized_range', 'trapezoid', 'trapz', 'truncnorm'
            ), suffix=_KW), Name.Namespace),
            # Empirical severities. These are distribution names, not
            # declaration keywords: they arrive through the grammar's `ids`
            # rule, as in `sev dhistogram xps [0 99] [.8 .2]`.
            (words(('dhistogram', 'chistogram'), suffix=_KW), Name.Class),

            # The discrete declaration keywords. `dfreq` belongs here with its
            # siblings rather than with the FREQ distribution names: it
            # declares an outcome-and-probability pair, it is not a named
            # distribution.
            (words(('dfreq', 'dsev', 'dbvsev', 'dwait'),
                   suffix=_KW), Name.Label),

            # Every remaining reserved word in the grammar's ID exclusion list.
            # `mixed` is absent on purpose: the root state colors it and opens
            # the mixing-distribution state. `inf` is absent because the number
            # rules claim it, as the grammar does.
            (words(
                ('occurrence', 'aggregate', 'distortion', 'exposure', 'tweedie',
                 'premium', 'tower', 'picks', 'prem', 'pnl', 'xpnl', 'peel',
                 'inherit',
                 'bivariate', 'bv', 'clash', 'copula',
                 'netceded', 'grossceded', 'grossnet',
                 'approximate', 'approx', 'ssev', 'splice',
                 'claims', 'ceded', 'claim', 'loss', 'payoff', 'dist',
                 'expense', 'expenses', 'cede', 'deposit', 'rol', 'less',
                 'port', 'rate', 'net', 'sev', 'agg', 'xps', 'wts',
                 'and', 'as', 'exp', 'at', 'cv', 'lr', 'xs',
                 'of', 'to', 'po', 'so', 'zm', 'zt',
                 # reinstatements
                 'reinstatements', 'reinstatement', 'free', 'no',
                 # variable rating
                 'swing', 'slide', 'retro', 'corridor', 'basic', 'lcm',
                 'min', 'max', 'pc', 'after',
                 # renewal frequency
                 'wait', 'years', 'year', ),
                suffix=_KW
            ), Keyword),
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
