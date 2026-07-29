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
# note/tags/hints bodies are free text (1.0.0a177)
# ----------------------------------------------------------------------
# A note is prose, so it may contain the three characters DecL preprocessing
# would otherwise claim: ``#`` and ``//`` (comment openers, step 2) and ``[``
# / ``]`` (the vector collapse, step 3). Before 1.0.0a177 the first two
# truncated the note into a confusing parse error and the third silently
# padded the text, so ``E[loss]=85`` was stored as ``E [loss] =85``. Step 0b
# lifts the body out the way step 0 lifts a doc body.
@pytest.mark.parametrize(
    'body',
    [
        'E[loss]=85, margin 15',            # brackets, unpadded
        'Beta on [0, 100000] -- bounded',   # brackets with a space inside
        'layer is 5# of limit',             # a hash mid-note
        'a // b',                           # a double slash mid-note
        'trailing hash #',                  # a hash at the very end
        'alpha = [0 .5 1 1]; and more',     # brackets and a semicolon
    ],
)
def test_note_body_is_free_text(body):
    """The note reaches the spec byte-for-byte, whatever punctuation it holds."""
    assert _spec(f'{BASE} note{{{body}}}')['note'] == body


def test_hash_in_a_note_does_not_truncate_the_statement():
    """The clause after a ``#``-bearing note still parses.

    The old failure was not just a mangled note: step 2 cut the line at the
    ``#``, so everything after it vanished and the parser reported an
    unexpected token several clauses earlier.
    """
    spec = _spec(f'{BASE} note{{5# of limit}} tags{{topic:x}} hints{{log2=16}}')
    assert spec['note'] == '5# of limit'
    assert spec['tags'] == ('topic:x',)
    assert spec['hints'] == 'log2=16'


def test_trailer_bodies_are_lifted_independently():
    """Three bodies on one statement restore to the right three clauses."""
    spec = _spec(f'{BASE} note{{n[1]}} tags{{t}} hints{{log2=16}}')
    assert (spec['note'], spec['tags'], spec['hints']) == ('n[1]', ('t',), 'log2=16')


def test_a_full_line_comment_holding_a_note_still_vanishes():
    """Lifting bodies must not resurrect a commented-out statement."""
    program = f'# {BASE} note{{commented out}}\n{BASE} note{{live}}'
    statements = UnderwritingLexer.preprocess(program)
    assert len(statements) == 1
    assert 'commented out' not in statements[0]
    assert statements[0].endswith('note{live}')


def test_bracketed_note_round_trips_through_format_program():
    """Unparse and re-parse leaves the note alone, so regeneration is stable.

    The padding was cumulative: each pass through the preprocessor added
    another space, so a file regenerated from its own specs drifted a little
    further every time.
    """
    program = f'{BASE} note{{E[loss]=85}}'
    once = format_program(_spec_kind_name(program), fmt='text', trailer=True)
    twice = format_program(_spec_kind_name(once), fmt='text', trailer=True)
    assert once == twice
    assert 'E[loss]=85' in once


def _spec_kind_name(program):
    """``(kind, name, spec)`` for one DecL program, the shape format_program takes."""
    statements = UnderwritingLexer.preprocess(program)
    assert len(statements) == 1
    return PARSER.parse(statements[0])


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
    # trailer=True is explicit: the default is False (the bare declaration is
    # what you want nearly every time you format one), and this test is about
    # the trailer surviving the round trip.
    rendered = format_program(program, fmt='text', trailer=True)
    spec2 = _spec(rendered)
    assert spec2['doc'] == spec1['doc']
    assert spec2['tags'] == spec1['tags'] == ('a', 'b')
    assert spec2['note'] == spec1['note'] == 'abstract'


def test_trailer_governs_all_four_clauses_and_is_off_by_default():
    """One flag governs the whole trailer, doc and tags included.

    ``False`` is the default as of 1.0.0a166: formatting a program is almost
    always about the math and the insurance, not the metadata around it.
    """
    a = build(f'agg TrailerOff 5 claims sev lognorm 10 cv 2 poisson '
              f'note{{n}} tags{{t}} hints{{log2=16}}', update=False)
    for bare in (a.format_program(layout='terse', trailer=False),
                 a.format_program(layout='terse')):          # same thing
        for clause in ('note{', 'tags{', 'hints{', 'doc{{{'):
            assert clause not in bare
    full = a.format_program(layout='terse', trailer=True)
    assert 'note{n}' in full and 'tags{t}' in full
    # and an explicit subset keeps only what it names
    hinted = a.format_program(layout='terse', trailer=('hints',))
    assert 'hints{log2=16}' in hinted
    assert 'note{' not in hinted and 'tags{' not in hinted


def test_spread_puts_each_trailer_clause_on_its_own_line():
    """Each of note / tags / hints / doc is its own clause, so its own line."""
    a = build('agg TrailerLines 5 claims sev lognorm 10 cv 2 poisson '
              'note{n} tags{t} hints{log2=16}', update=False)
    lines = a.format_program(trailer=True).split('\n')
    assert '  note{n}' in lines
    assert '  tags{t}' in lines
    assert '  hints{log2=16}' in lines


def test_sev_and_distortion_indent_their_trailer_too():
    """A flat statement still spreads its trailer under the declaration."""
    for prog, head in (
            ('sev TrailerSev lognorm 10 cv 1 note{s} tags{topic:severity}',
             'sev TrailerSev lognorm 10 cv 1'),
            ('dist TrailerDist ph 0.7 note{d} tags{topic:distortion}',
             'distortion TrailerDist ph 0.7')):
        lines = format_program(prog, trailer=True).split('\n')
        assert lines[0] == head
        assert lines[1].startswith('  note{')
        assert lines[2].startswith('  tags{')


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
    full = d.format_program(trailer=True)
    assert 'note{proportional hazard}' in full
    assert 'tags{distortion, hero}' in full


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
