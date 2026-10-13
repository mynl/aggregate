"""The DecL help pages, ``decl_help()``: every example builds and the source rules hold.

The help is hand-edited HTML in ``src/aggregate/data/decl-help/``. These tests are
what stop it drifting from the grammar, and what keep it renderable in a host page.
The source rules they enforce are numbered in the docstring of
``aggregate/decl_help.py``.
"""
from __future__ import annotations

import html
import re
import tomllib
from pathlib import Path

import pytest
from pygments.token import Token

import aggregate
from aggregate import DECL_HELP_FORMS, DeclHelp, build, decl_help
from aggregate.decl_help import _code_texts, _source
from aggregate.decl_pygments import AggLexer

PYPROJECT = Path(__file__).parents[1] / 'pyproject.toml'


def _code_snippets():
    """``(form, text)`` for every ``pre.dh-code`` and ``code.dh`` in the sources, unescaped."""
    out = []
    for form in DECL_HELP_FORMS:
        for tag, classes, text in _code_texts(_source(form)):
            if (tag == 'pre' and 'dh-code' in classes) or (tag == 'code' and 'dh' in classes):
                out.append((form, html.unescape(text)))
    return out


def _programs():
    """Every program the help shows whole: each ``pre.dh-code`` and each rendered ``data-load``."""
    programs = {}
    for form in DECL_HELP_FORMS:
        for tag, classes, text in _code_texts(_source(form)):
            if tag == 'pre' and 'dh-code' in classes:
                programs.setdefault(html.unescape(text).strip('\n'), form)
        for value in re.findall(r'data-load="([^"]*)"', decl_help(form)):
            programs.setdefault(html.unescape(value), form)
    return list(programs.items())


def _selectors(css):
    """Yield every selector in a stylesheet, descending into ``@container`` and ``@media``."""
    css = re.sub(r'/\*.*?\*/', '', css, flags=re.S)
    prelude, i = '', 0
    while i < len(css):
        ch = css[i]
        if ch == '{':
            head, prelude = prelude.strip(), ''
            if not head.startswith('@'):
                # a style rule: split its selector list at top-level commas
                depth, start = 0, 0
                for j, c in enumerate(head):
                    depth += (c == '(') - (c == ')')
                    if c == ',' and depth == 0:
                        yield head[start:j].strip()
                        start = j + 1
                yield head[start:].strip()
                i = css.index('}', i)
        elif ch == '}':
            prelude = ''
        else:
            prelude += ch
        i += 1


PROGRAMS = _programs()
SNIPPETS = _code_snippets()


def test_forms_and_entry_point():
    assert DECL_HELP_FORMS == ('anatomy', 'card')
    default = decl_help()
    assert isinstance(default, DeclHelp) and isinstance(default, str)
    assert default.form == 'anatomy' and default == decl_help('anatomy')
    assert default._repr_html_() == str(default) and type(default._repr_html_()) is str
    assert repr(default).startswith("DeclHelp('anatomy', aggregate ")
    with pytest.raises(ValueError, match='card'):
        decl_help('glossary')


def test_program_count():
    """Seven programs at 1.2.0a15: three in the anatomy, four on the card."""
    assert len(PROGRAMS) == 7


@pytest.mark.parametrize('form, program', [(f, p) for p, f in PROGRAMS],
                         ids=[f'{f}:{p.splitlines()[0]}' for p, f in PROGRAMS])
def test_program_builds(form, program):
    assert build(program) is not None


@pytest.mark.parametrize('form, text', SNIPPETS,
                         ids=[f'{f}:{t.strip()[:40]}' for f, t in SNIPPETS])
def test_snippet_lexes_clean(form, text):
    """Weaker than building, since a chip is not a statement: catches a misspelled or retired keyword."""
    errors = [v for t, v in AggLexer().get_tokens(text.replace('...', '')) if t in Token.Error]
    assert errors == []


@pytest.mark.parametrize('form', DECL_HELP_FORMS)
def test_source_rules(form):
    source = _source(form)
    # one fragment, one style
    assert source.startswith(f'<section class="decl-help dh-{form}" data-decl-version="">')
    assert source.count('<section') == 1 and source.count('<style>') == 1
    # rule 1: plain DecL inside code
    for tag, classes, text in _code_texts(source):
        if (tag == 'pre' and 'dh-code' in classes) or (tag == 'code' and 'dh' in classes):
            assert '<' not in text, text
    # rule 4: every selector scoped to its form
    css = re.search(r'<style>(.*?)</style>', source, re.S).group(1)
    selectors = list(_selectors(css))
    assert len(selectors) > 20
    for selector in selectors:
        assert re.match(rf'\.decl-help\.dh-{form}(?![\w-])', selector), selector
    # rule 5: no rem, which would follow the host's root size
    assert not re.search(r'\d(?:\.\d+)?rem\b', css)
    # rule 7: self-contained, and links only to the docs
    for banned in ('<script', '<link', '<img', 'url('):
        assert banned not in source
    hrefs = re.findall(r'href="([^"]*)"', source)
    assert hrefs and all(h.startswith('https://aggregate.readthedocs.io/') for h in hrefs)


@pytest.mark.parametrize('form', DECL_HELP_FORMS)
def test_rendered(form):
    rendered = decl_help(form)
    assert f'data-decl-version="{aggregate.__version__}"' in rendered
    assert '{{version}}' not in rendered
    # every data-load is valued
    assert not re.search(r'\sdata-load(?=[\s/>])', rendered)
    # coloring is lossless: strip the spans and the source's code comes back
    source_code = [t for _, _, t in _code_texts(_source(form))]
    rendered_code = [html.unescape(re.sub(r'</?span[^>]*>', '', t)) for _, _, t in _code_texts(rendered)]
    assert rendered_code == [html.unescape(t) for t in source_code]
    assert '<span class="t-kw">' in rendered


def test_anatomy_loads():
    """The anatomy offers three Load targets: the slot table, the portfolio and the P&L."""
    loads = [html.unescape(v) for v in re.findall(r'data-load="([^"]*)"', decl_help('anatomy'))]
    assert [p.split()[0] for p in loads] == ['agg', 'port', 'xpnl']
    assert loads[0].endswith('  hints{bs=1;log2=16;}')
    assert 'data-load' not in decl_help('card')


def test_package_data():
    """The sources reach the wheel."""
    with PYPROJECT.open('rb') as f:
        package_data = tomllib.load(f)['tool']['setuptools']['package-data']['aggregate']
    assert 'data/decl-help/*.html' in package_data
