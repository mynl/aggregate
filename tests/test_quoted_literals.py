"""Quoted strings survive statement splitting verbatim ([Quote-Safe-Statements]).

Both splitters, :meth:`UnderwritingLexer.preprocess` and
:meth:`UnderwritingLexer.raw_statements`, lift quoted strings behind
placeholders before stripping comments or touching brackets, the treatment
``note{}`` bodies already had. Before 1.11.2 a label ``as "Cat #1"`` lost its
tail to the comment stripper and ``as "a[1]"`` came back respaced.
"""

import pytest

from aggregate import build
from aggregate.parser import UnderwritingLexer

L = UnderwritingLexer


@pytest.mark.parametrize('label', [
    'Cat #1',
    'a // b',
    'a[1]',
    'E[X] = 85',
    'x; y',
    'note{q}',
    'two  spaces',
])
def test_label_text_is_verbatim(label):
    """Every DecL punctuation mark inside a quoted label is text."""
    program = f'agg X as "{label}" dfreq [1] dsev [1 2]'
    for split in (L.preprocess, L.raw_statements):
        statements = split(program)
        assert len(statements) == 1
        assert f'"{label}"' in statements[0]


def test_label_survives_build():
    a = build('agg QuoteSafe as "Cat #1 [gross]" dfreq [1] dsev [1 2]',
              update=False)
    assert a.label == 'Cat #1 [gross]'


def test_quote_in_note_is_prose():
    """A lone quote inside a note does not open a string."""
    program = 'agg X dfreq [1] dsev [1 2] note{he said "x # y}'
    for split in (L.preprocess, L.raw_statements):
        assert split(program)[0].endswith('note{he said "x # y}')


def test_comments_with_quotes_still_strip():
    program = ('# a "quoted" comment\n'
               'agg X as "Y" dfreq [1] dsev [1 2] // 3" bolts\n'
               '\n'
               'agg Z dfreq [1] dsev [[1 2]]')
    for split in (L.preprocess, L.raw_statements):
        statements = split(program)
        assert len(statements) == 2
        assert 'bolts' not in statements[0]
        assert 'quoted' not in statements[0]
        assert '"Y"' in statements[0]


def test_brackets_in_string_do_not_unbalance_statement():
    """A ``[`` inside a string must not hold a statement open over a blank line."""
    program = 'agg X as "open [" dfreq [1] dsev [1 2]\n\nagg Z dfreq [1] dsev [3]'
    for split in (L.preprocess, L.raw_statements):
        assert len(split(program)) == 2
