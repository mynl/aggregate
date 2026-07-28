"""Tests for the ``tags{...}`` and ``doc{{{...}}}`` trailer clauses (1.0.0a157).

``note{...}`` is a one-line abstract, ``tags{...}`` a slug list for grouping and
selection, ``hints{...}`` build settings, and ``doc{{{...}}}`` a long-form
markdown recipe that may carry fenced code.

The interesting one is ``doc``. Everything it needs to contain -- ``#``
headings, blank lines, fenced code blocks, braces, semicolons -- is something
DecL preprocessing would otherwise destroy: step 1 strips anything after a
``#``, step 4/5 split statements on blank lines, and a bare ``}`` closes a
note-style clause at the first occurrence. So the body never reaches the lexer
as text: step 0 of :meth:`UnderwritingLexer.preprocess` lifts it out and
substitutes URL-safe base64, whose alphabet is inert through every later step.

The one forbidden body content is a line consisting solely of ``}}}``; an
inline ``}}}`` (a nested dict literal, say) is harmless because the closing
fence must be alone on its line.

The DecL programs exercised here are mirrored in
``src/aggregate/agg/decl-testers.agg`` under the AF section.
"""

import pytest

from aggregate import build
from aggregate.decl_writer import format_program
from aggregate.parser import UnderwritingLexer, UnderwritingParser

PARSER = UnderwritingParser(lambda x: {})
BASE = 'agg A 5 claims sev lognorm 10 cv 2 poisson'


def _spec(program):
    """Preprocess and parse one DecL program, returning its spec dict."""
    statements = UnderwritingLexer.preprocess(program)
    assert len(statements) == 1, f'expected 1 statement, got {len(statements)}'
    _kind, _name, spec = PARSER.parse(statements[0])
    return spec


# ----------------------------------------------------------------------
# tags: decomposition
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    'clause, expected',
    [
        ('tags{severity}', ('severity',)),
        ('tags{severity frequency}', ('severity', 'frequency')),
        ('tags{severity, frequency}', ('severity', 'frequency')),
        ('tags{severity,frequency  aggregate}', ('severity', 'frequency', 'aggregate')),
        ('tags{check:round-trip, hero}', ('check:round-trip', 'hero')),
        # order preserved, duplicates dropped
        ('tags{b, a, b}', ('b', 'a')),
        ('tags{}', ()),
    ],
)
def test_tags_decompose_to_a_tuple(clause, expected):
    """``tags{...}`` splits on commas and/or whitespace into an ordered tuple."""
    assert _spec(f'{BASE} {clause}').get('tags', ()) == expected


# ----------------------------------------------------------------------
# Conditional spec keys
# ----------------------------------------------------------------------
def test_note_and_hints_are_always_present():
    """Every spec carries ``note`` and ``hints``, as it has since 1.0.0a25."""
    spec = _spec(BASE)
    assert spec['note'] == ''
    assert spec['hints'] == ''


def test_tags_and_doc_are_absent_when_not_written():
    """``tags``/``doc`` are conditional keys.

    The captured spec snapshot (``tests/data/expected_specs.json``) compares key
    sets exactly, so emitting these unconditionally would fail every one of its
    163 cases. Hosts read them with ``spec.get(...)``.
    """
    spec = _spec(BASE)
    assert 'tags' not in spec
    assert 'doc' not in spec


# ----------------------------------------------------------------------
# doc: the content DecL would otherwise destroy
# ----------------------------------------------------------------------
HARD_DOC = """## Problem

A `#` heading above, a blank line, and code below.

## Solution

```python
d = {'a': {'b': {'c': 1}}}
x = 1; y = 2      # trailing comment with a semicolon
url = 'http://example.com'   // not a comment either
```
"""

PROGRAM_WITH_DOC = f'{BASE}\n    doc{{{{{{\n{HARD_DOC}}}}}}}'


def test_doc_survives_markdown_and_code():
    """Headings, blank lines, fences, nested braces, ``;`` and ``//`` all survive."""
    doc = _spec(PROGRAM_WITH_DOC)['doc']
    assert doc.startswith('## Problem')
    assert '\n\n' in doc, 'blank lines must survive the paragraph split'
    assert '# heading above' in doc or '`#` heading' in doc
    assert "{'a': {'b': {'c': 1}}}" in doc, 'inline }}} must not close the fence'
    assert 'x = 1; y = 2' in doc
    assert '// not a comment either' in doc
    assert '```python' in doc


def test_doc_body_does_not_split_the_statement():
    """A doc's blank lines must not break one statement into several."""
    assert len(UnderwritingLexer.preprocess(PROGRAM_WITH_DOC)) == 1


def test_inline_closing_fence_does_not_terminate_the_doc():
    """Only a line that *is* ``}}}`` closes the block."""
    doc = _spec(PROGRAM_WITH_DOC)['doc']
    # The nested dict ends with }}} mid-line; everything after it survived.
    assert doc.index("{'c': 1}}}") < doc.index('// not a comment either')


def test_empty_doc_body_is_allowed():
    """A doc with nothing in it parses and yields an empty string."""
    assert _spec(f'{BASE}\n    doc{{{{{{\n\n}}}}}}').get('doc', '') == ''


# ----------------------------------------------------------------------
# Order-free, at most one of each
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    'clauses',
    [
        'note{n} tags{t} hints{log2=16}',
        'hints{log2=16} note{n} tags{t}',
        'tags{t} hints{log2=16} note{n}',
        'tags{t} note{n} hints{log2=16}',
    ],
)
def test_trailer_items_are_order_free(clauses):
    """Any order of the trailer items gives the same spec."""
    spec = _spec(f'{BASE} {clauses}')
    assert spec['note'] == 'n'
    assert spec['tags'] == ('t',)
    assert spec['hints'] == 'log2=16'


@pytest.mark.parametrize('clause', ['note{a} note{b}', 'tags{a} tags{b}',
                                    'hints{log2=16} hints{log2=17}'])
def test_repeated_trailer_item_is_an_error(clause):
    """At most one of each; a repeat is a clear parse error, not a silent win."""
    with pytest.raises(ValueError, match='repeated'):
        PARSER.parse(f'{BASE} {clause}')


# ----------------------------------------------------------------------
# Round trip
# ----------------------------------------------------------------------
def test_doc_and_tags_round_trip_through_format_program():
    """``format_program`` output re-parses to the same spec.

    The writer emits a raw markdown body between real fences; re-parsing simply
    re-encodes it. This is the property that makes the doc clause safe to store
    in a ``.agg`` library and re-export.
    """
    program = f'{BASE} note{{abstract}} tags{{a, b}}\n    doc{{{{{{\n{HARD_DOC}}}}}}}'
    spec1 = _spec(program)
    rendered = format_program(program, fmt='text')
    spec2 = _spec(rendered)
    assert spec2['doc'] == spec1['doc']
    assert spec2['tags'] == spec1['tags'] == ('a', 'b')
    assert spec2['note'] == spec1['note'] == 'abstract'


def test_trailer_false_drops_all_four_clauses():
    """One flag governs the whole trailer, doc and tags included."""
    a = build(f'agg TrailerOff 5 claims sev lognorm 10 cv 2 poisson '
              f'note{{n}} tags{{t}} hints{{log2=16}}', update=False)
    bare = a.format_program(layout='terse', trailer=False)
    for clause in ('note{', 'tags{', 'hints{', 'doc{{{'):
        assert clause not in bare
    full = a.format_program(layout='terse')
    assert 'note{n}' in full and 'tags{t}' in full


# ----------------------------------------------------------------------
# Hosts
# ----------------------------------------------------------------------
def test_distortion_carries_a_trailer():
    """Distortions gained a trailer in 1.0.0a157; they had none before.

    This is what lets the distortion recipes be described, tagged and tested
    like every other library entry.
    """
    d = build('dist DocDist ph 0.7 note{proportional hazard} tags{distortion, hero}')
    assert d.note == 'proportional hazard'
    assert d.tags == ('distortion', 'hero')
    assert d.doc == ''
    assert 'note{proportional hazard}' in d.pprogram
    assert 'tags{distortion, hero}' in d.pprogram


def test_aggregate_carries_tags_and_doc():
    """The attributes land on the built object, not just the spec."""
    a = build(f'agg DocAgg 5 claims sev lognorm 10 cv 2 poisson tags{{aggregate, intro}}'
              f'\n    doc{{{{{{\n## Problem\n\nHi.\n}}}}}}', update=False)
    assert a.tags == ('aggregate', 'intro')
    assert a.doc == '## Problem\n\nHi.'


def test_pnl_carries_its_own_trailer_not_the_engines():
    """A ``pnl``'s metadata is the statement's, taken from its build recipe."""
    p = build('pnl DocPnL 100 premium less '
              'agg DocPnL_e 5 claims sev lognorm 10 cv 1 poisson '
              'note{the pnl note} tags{pnl}')
    assert p.note == 'the pnl note'
    assert p.tags == ('pnl',)
