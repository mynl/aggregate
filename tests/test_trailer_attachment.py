"""Where a trailing ``note{...}`` binds when a statement nests a trailer-bearing object.

The rule, in one line: **a trailing trailer binds to the OUTER object**; to
annotate an inner component, put its trailer on that component.

This matters because several statements nest an ``agg_out`` -- which ends in its
own ``trailer`` -- immediately before their own ``trailer``. In one shape it is
genuinely ambiguous to the grammar:

``bivariate NAME <exposure> <agg> <agg> note{...}``

``copula_clause`` is nullable (``decl.lark``) and ``bv_body`` ends in an
``agg_out``, so with neither a copula clause nor an outer frequency to separate
them, the ``note`` can parse two ways. Earley's default ``ambiguity='resolve'``
picks the outer binding. These tests pin that choice so a grammar edit cannot
flip it silently -- a round-trip test alone would not catch a flip, because the
flipped parse is still a fixed point of parse -> render -> parse.

``port`` is *not* ambiguous: ``port_out`` places its trailer BEFORE ``agg_list``,
so the portfolio's slot has already closed by the time a unit is read. That is
why a note written after the last unit lands on the unit, not the portfolio --
the behaviour the shipped library headers warn about.

The DecL programs exercised here are mirrored in
``src/aggregate/agg/decl-testers.agg`` under the AE section.
"""

import warnings

import pytest

from aggregate import build
from aggregate.parser import UnderwritingParser

PARSER = UnderwritingParser(lambda x: {})

BV_HEAD = 'bivariate T 25 claims'
WIND = 'agg Wind dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2'
FLOOD = 'agg Flood dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5'

# (id, program, expected outer note, expected per-unit notes)
CASES = [
    (
        'bivariate-bare',
        f'{BV_HEAD} {WIND} {FLOOD} note{{outer}}',
        'outer',
        ['', ''],
    ),
    (
        'bivariate-separated',
        f'{BV_HEAD} {WIND} {FLOOD} copula gumbel 0.4 poisson note{{outer}}',
        'outer',
        ['', ''],
    ),
    (
        'bivariate-inner-and-outer',
        f'{BV_HEAD} {WIND} note{{inner wind}} {FLOOD} note{{inner flood}} '
        f'copula gumbel 0.4 poisson note{{outer}}',
        'outer',
        ['inner wind', 'inner flood'],
    ),
    (
        'clash',
        'clash T 8 5 2 claims sev lognorm 50 cv 1.2 sev lognorm 60 cv 1.5 '
        'mixed gamma 0.2 note{outer}',
        'outer',
        ['', ''],
    ),
    (
        # A view-pair wraps ONE agg, and the transformer lifts that agg's note
        # onto the bivariate spec -- so it legitimately appears in both places.
        'netceded',
        'netceded agg T 8 claims sev 300 * beta 2 3 '
        'occurrence net of 0.7 po 60 xs 40 poisson note{outer}',
        'outer',
        ['outer'],
    ),
]


@pytest.mark.parametrize(
    'program, outer, units', [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_trailing_note_binds_to_outer_object(program, outer, units):
    """A trailing note lands on the bivariate/clash/view-pair, not a component."""
    _kind, _name, spec = PARSER.parse(program)
    assert spec['note'] == outer
    assert [u[2].get('note', '') for u in spec['units']] == units


def test_port_trailer_is_positional():
    """``port`` puts its trailer before the units, so both notes land as written.

    This is the shape the shipped library headers warn about: a note written
    after the last unit belongs to that unit, because the portfolio's trailer
    slot closed before ``agg_list`` began.
    """
    program = ('port PP note{on the portfolio} '
               'agg U1 5 claims sev lognorm 10 cv 1 poisson note{on the last unit}')
    _kind, _name, spec = PARSER.parse(program)
    assert spec['note'] == 'on the portfolio'
    assert [u[2].get('note', '') for u in spec['spec']] == ['on the last unit']


def test_port_trailer_survives_to_the_built_objects():
    """The positional split holds end to end, not just in the spec."""
    p = build('''
port TESTPORT note{at the port level}
    agg RuRe 1 year sev gamma 2 wait 0.25 * uniform note{on the agg}
''')
    assert p.note == 'at the port level'
    assert p.agg_list[0].note == 'on the agg'


def test_bare_bivariate_note_survives_to_the_built_object():
    """The ambiguous shape resolves to the bivariate on a real build, too."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        b = build(f'{BV_HEAD} {WIND} {FLOOD} note{{outer}}', update=False)
    assert b.note == 'outer'
    assert [u.note for u in b.units] == ['', '']
