"""
One page of DecL help, as an HTML fragment, for Jupyter and the aggregate Loss Lab.

``decl_help()`` returns the *anatomy* ("DecL at a glance": one ``agg`` statement
taken apart slot by slot, a portfolio and a P&L built on it, and the alternative
forms for each slot). ``decl_help('card')`` returns the *card* ("DecL on one
page": a ten-box reference card that prints to one Letter landscape page). The
return is a :class:`DeclHelp`, a ``str`` that displays itself in Jupyter, so a
web server can send it as is.

The sources are hand-edited HTML in ``data/decl-help/{form}.html``. The library
owns them because it owns the language: the help follows the grammar's version,
and ``tests/test_decl_help.py`` builds every program in it, so it cannot drift the
way ``cheat-sheets/DecL_Cheat_Sheet.tex`` did.

Notes
-----
**Source rules.** Each source file holds exactly one
``<section class="decl-help dh-{form}" data-decl-version="">`` with one
``<style>`` and no script. The rules below keep it renderable and testable, and
``tests/test_decl_help.py`` enforces each one.

1. The text of every ``<pre class="dh-code">`` and ``<code class="dh">`` is plain
   DecL. HTML entities are allowed (``&amp;``); tags are not. The renderer colors
   these elements and touches nothing else.
2. ``...`` marks an elision (``agg Book ...``, ``sev ... !``). It renders as
   text; the lex test removes it before lexing.
3. A bare ``data-load`` attribute marks a loadable program, on a
   ``pre.dh-code`` (the program is its text) or on a container whose program
   lines are ``<code class="dh dh-l">`` cells (joined with newlines). The
   renderer writes the program into the attribute, so it is never typed twice
   and a host reads the attribute, never the markup.
4. Every CSS rule is scoped under ``.decl-help.dh-{form}``, including those
   inside ``@container`` blocks, so two fragments in one document cannot
   restyle each other.
5. Sizes are in ``em`` off one base, ``font-size: var(--dh-size, ...)`` on the
   section, so a host scales the whole page by setting ``--dh-size``. ``rem`` is
   avoided because it follows the host document's root size. Border widths and
   radii stay in ``px``.
6. Colors come from ``--dh-*`` custom properties declared on the section, so a
   host can retheme without editing the fragment.
7. Self-contained: no ``<script>``, ``<link>``, ``<img>`` or ``url(...)``, and
   every link points at ``aggregate.readthedocs.io``.
8. ``.dh-lab`` content shows only when the host adds ``in-lab`` to the section;
   ``.dh-py`` content shows everywhere else.
9. ``data-decl-version=""`` and ``{{version}}`` are filled with the library
   version.
10. The card sets ``print-color-adjust: exact`` on itself and carries no
    ``@page`` rule; page size is the host's business.

**Rendering** is three regular-expression passes over the source: fill
``data-load``, color the code, stamp the version. Regular expressions are safe
here, rather than an HTML parser, because rule 1 forbids tags inside code and
the tests check it. Coloring lexes with :class:`~aggregate.decl_pygments.AggLexer`
and maps token types *exactly* (not by Pygments' subtype rule, since
``Comment.Single`` is a subtype of ``Comment`` and the two color differently) to
the four classes of the Loss Lab editor: ``t-kw``, ``t-num``, ``t-str``,
``t-com``. Pygments' own ``HtmlFormatter`` is not used: its short classes need
Pygments' stylesheet and clash with any page that already styles
``.highlight .k``.
"""

import functools
import html
import re
from importlib.resources import files

from pygments.token import Comment, Keyword, Name, Number, String

from .decl_pygments import AggLexer

__all__ = ['decl_help', 'DECL_HELP_FORMS', 'DeclHelp']

#: The forms ``decl_help`` offers, the first being the default.
DECL_HELP_FORMS = ('anatomy', 'card')

#: Exact token type to the editor class it colors as. Anything absent is plain.
_TOKEN_CLASS = {
    Keyword: 't-kw', Keyword.Constant: 't-kw', Name.Function: 't-kw',
    Name.Class: 't-kw', Name.Namespace: 't-kw', Name.Label: 't-kw',
    Name.Builtin: 't-kw',
    Number: 't-num',
    String: 't-str', Comment.Preproc: 't-str', Comment: 't-str',
    Name.Tag: 't-str', Name.Attribute: 't-str',
    Comment.Single: 't-com',
}

# An opening tag carrying a bare (valueless) data-load attribute.
_OPEN_LOAD = re.compile(
    r'<(?P<tag>[a-z][a-z0-9]*)\b(?P<before>[^>]*?)\sdata-load(?=[\s/>])(?P<after>[^>]*)>')
# A pre or code element and its text; rule 1 makes the non-greedy body safe.
_CODE_ELEMENT = re.compile(r'(<(pre|code)\b([^>]*)>)(.*?)(</\2>)', re.S)
_CLASS_ATTR = re.compile(r'\bclass="([^"]*)"')


class DeclHelp(str):
    """A rendered DecL help fragment.

    A ``str``, so a web server returns it as is, with ``_repr_html_`` so a
    Jupyter cell displays it and a short ``repr`` so a plain Python prompt
    does not print 20 kB of HTML.

    Parameters
    ----------
    text : str
        The rendered fragment.
    form : str
        Which form it is, one of :data:`DECL_HELP_FORMS`. Defaulted only so
        that pickling and copying, which rebuild a ``str`` subclass from its
        text alone, still work; the attribute is restored afterwards.
    """

    def __new__(cls, text, form=''):
        obj = super().__new__(cls, text)
        obj.form = form
        return obj

    def _repr_html_(self):
        """Return the fragment as a plain ``str`` for Jupyter's HTML display."""
        return str(self)

    def __repr__(self):
        """Name the form, the library version and the size, not the HTML."""
        from . import __version__
        size = len(self.encode('utf-8')) / 1000
        return f'DeclHelp({self.form!r}, aggregate {__version__}, {size:.0f} kB)'


def decl_help(form='anatomy'):
    """Return one page of DecL help as an HTML fragment.

    Parameters
    ----------
    form : str, default ``'anatomy'``
        ``'anatomy'`` ("DecL at a glance", one statement taken apart slot by
        slot, with a portfolio, a P&L and the alternative forms of each slot)
        or ``'card'`` ("DecL on one page", a ten-box reference card that prints
        to one Letter landscape page). :data:`DECL_HELP_FORMS` lists them.

    Returns
    -------
    DeclHelp
        The fragment, a ``str``: one ``<section class="decl-help dh-{form}">``
        with its own scoped ``<style>`` and no script. It displays itself in a
        Jupyter cell.

    Raises
    ------
    ValueError
        If ``form`` is not one of :data:`DECL_HELP_FORMS`.

    Notes
    -----
    The render is cached per process, so a library upgrade needs a restart to
    show new help. The module docstring sets out the source rules and the
    rendering passes.

    Examples
    --------
    >>> from aggregate import decl_help
    >>> decl_help()                        # doctest: +SKIP
    >>> decl_help('card')                  # doctest: +SKIP
    """
    if form not in DECL_HELP_FORMS:
        raise ValueError(f'Unknown DecL help form {form!r}; '
                         f'the forms are {", ".join(map(repr, DECL_HELP_FORMS))}.')
    return DeclHelp(_render(form), form)


@functools.lru_cache(maxsize=None)
def _render(form):
    """Read ``data/decl-help/{form}.html`` and render it, once per process.

    Parameters
    ----------
    form : str
        A member of :data:`DECL_HELP_FORMS`, already validated.

    Returns
    -------
    str
        The fragment with ``data-load`` filled, code colored and the version
        stamped.
    """
    from . import __version__
    source = _source(form)
    text = _color_code(_fill_loads(source))
    version = html.escape(__version__)
    return (text.replace('data-decl-version=""', f'data-decl-version="{version}"', 1)
                .replace('{{version}}', version))


def _source(form):
    """Return the unrendered source of one form, read as package data."""
    return files('aggregate').joinpath('data', 'decl-help', f'{form}.html').read_text(encoding='utf-8')


def _classes(attrs):
    """Return the set of class names in an opening tag's attribute text."""
    m = _CLASS_ATTR.search(attrs)
    return set(m.group(1).split()) if m else set()


def _code_texts(fragment):
    """Yield ``(tag, classes, text)`` for every ``pre`` and ``code`` element.

    ``text`` is the element's body exactly as written, entities unexpanded.
    """
    for m in _CODE_ELEMENT.finditer(fragment):
        yield m.group(2), _classes(m.group(3)), m.group(4)


def _element_end(source, tag, start):
    """Return the index of the tag closing the ``tag`` element opened before ``start``.

    Counts nested elements of the same name, which is all the balance a
    ``data-load`` container needs.
    """
    depth = 1
    for m in re.finditer(rf'<(/?){tag}\b[^>]*>', source[start:]):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            return start + m.start()
    raise ValueError(f'Unclosed <{tag} data-load> in the DecL help source.')


def _fill_loads(source):
    """Pass 1: write each loadable program into its bare ``data-load`` attribute.

    On a ``pre`` the program is the element's text, without the newline that
    follows ``<pre>`` (which HTML drops) or the one before ``</pre>``. On any
    other element it is the text of the ``code.dh-l`` cells inside, joined with
    newlines.
    """
    def fill(m):
        tag = m['tag']
        body = source[m.end():_element_end(source, tag, m.end())]
        if tag == 'pre':
            program = html.unescape(body).strip('\n')
        else:
            program = '\n'.join(html.unescape(text) for t, classes, text in _code_texts(body)
                                if t == 'code' and 'dh-l' in classes)
        return f'<{tag}{m["before"]} data-load="{html.escape(program)}"{m["after"]}>'

    return _OPEN_LOAD.sub(fill, source)


def _color_code(fragment):
    """Pass 2: color the text of every ``pre.dh-code`` and ``code.dh``."""
    def color(m):
        tag, classes = m.group(2), _classes(m.group(3))
        if (tag == 'pre' and 'dh-code' in classes) or (tag == 'code' and 'dh' in classes):
            return m.group(1) + _color_tokens(html.unescape(m.group(4))) + m.group(5)
        return m.group(0)

    return _CODE_ELEMENT.sub(color, fragment)


def _color_tokens(text):
    """Return DecL ``text`` as escaped HTML with ``t-kw``/``t-num``/``t-str``/``t-com`` spans.

    Parameters
    ----------
    text : str
        Plain DecL, entities already expanded.

    Returns
    -------
    str
        The same text, escaped, with each colored token wrapped in a span.
        Removing the spans and unescaping gives back ``text`` exactly: the
        lexer runs with ``stripnl`` and ``ensurenl`` off, so no newline is
        added or dropped. An ``...`` elision lexes as error tokens, which are
        uncolored.
    """
    out = []
    for ttype, value in AggLexer(stripnl=False, ensurenl=False).get_tokens(text):
        escaped = html.escape(value, quote=False)
        cls = _TOKEN_CLASS.get(ttype)
        out.append(f'<span class="{cls}">{escaped}</span>' if cls else escaped)
    return ''.join(out)
