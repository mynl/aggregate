"""DecL human labels, quoted names, and expense grouping ([DecL-Labels]).

Covers the ``as`` display-label clause on objects (``agg`` / ``pnl`` / ``sev`` /
``port``), on the premium head, on reinsurance cessions, and the two-level
expense grammar (``and`` combines into one leg; juxtaposition makes separate
legs). No computed value changes -- labels land in dict keys, column headers, and
repr / exhibit titles. See ``dev/plan-decl-labels.md``.

The DecL programs here are mirrored in ``src/aggregate/agg/decl-testers.agg``
section Y (this module is the canonical source).
"""
import pytest

from aggregate import Underwriter, build
from aggregate.decl_writer import spec_to_decl

TOL = 1e-6

_PNL_BASE = ('pnl B 1000 premium less 850 loss sev lognorm 100 cv 1 poisson')


@pytest.fixture(scope="module")
def uw():
    return Underwriter()


def _parse(uw, program):
    return uw.parser.parse(uw.lexer.tokenize(program))


# ----------------------------------------------------------------------
# Object display labels
# ----------------------------------------------------------------------
def test_no_label_display_name_falls_back_to_name():
    a = build('agg Plain 100 claims sev lognorm 100 cv 2 poisson')
    assert a.display_label is None
    assert a.display_name == a.name == 'Plain'


def test_quoted_object_label():
    a = build('agg GrossBook as "Gross Book P&L" 100 claims sev lognorm 100 cv 2 poisson')
    assert a.name == 'GrossBook'                    # identity handle unchanged
    assert a.display_label == 'Gross Book P&L'      # spaces carried by quotes
    assert a.display_name == 'Gross Book P&L'
    assert 'Gross Book P&L' in repr(a)


def test_bareword_object_label_skips_quotes():
    a = build('agg Lae as lae 100 claims sev lognorm 100 cv 2 poisson')
    assert a.display_label == 'lae'
    assert a.name == 'Lae'


def test_sev_object_label():
    s = build('sev MySev as "My Severity" lognorm 100 cv 2')
    assert s.name == 'MySev'
    assert s.display_label == 'My Severity'
    assert s.display_name == 'My Severity'


def test_pnl_object_label():
    p = build(_PNL_BASE.replace('pnl B', 'pnl B as "My Book"'))
    assert p.name == 'B'
    assert p.display_label == 'My Book'
    assert 'My Book' in repr(p)


def test_port_object_label():
    p = build('port MyPort as "My Portfolio" '
              'agg A 50 claims sev lognorm 100 cv 2 poisson '
              'agg B 50 claims sev lognorm 100 cv 2 poisson')
    assert p.name == 'MyPort'
    assert p.display_label == 'My Portfolio'
    assert p.display_name == 'My Portfolio'


def test_title_name_shows_label_and_identity():
    a = build('agg GrossBook as "Gross Book" 100 claims sev lognorm 100 cv 2 poisson')
    assert a._title_name == 'Gross Book (GrossBook)'


# ----------------------------------------------------------------------
# `as` is a reserved word
# ----------------------------------------------------------------------
def test_as_is_reserved(uw):
    # a bare `as` where an identifier is expected must not lex as an ID
    with pytest.raises(Exception):
        _parse(uw, 'agg as 100 claims sev lognorm 100 cv 2 poisson')


# ----------------------------------------------------------------------
# Premium label -> consideration leg key
# ----------------------------------------------------------------------
def test_premium_label_names_consideration_leg():
    p = build(_PNL_BASE.replace('1000 premium', '1000 premium as "GWP"'))
    assert 'GWP' in p.summary_df.index
    assert 'consideration' not in p.summary_df.index
    assert p.summary_df.loc['GWP', 'EX'] == pytest.approx(1000.0)


def test_premium_no_label_keeps_default_leg():
    p = build(_PNL_BASE)
    assert 'consideration' in p.summary_df.index


# ----------------------------------------------------------------------
# Expense grouping: `and` combines, juxtaposition separates
# ----------------------------------------------------------------------
def test_single_group_and_joined_is_one_leg():
    # backward compatible: and-joined terms sum into one leg named 'expense'
    p = build(_PNL_BASE + ' 25% premium expense and 200 fixed expense')
    assert 'expense' in p.summary_df.index
    assert p.summary_df.loc['expense', 'EX'] == pytest.approx(0.25 * 1000 + 200)


def test_juxtaposed_groups_are_separate_legs():
    p = build(_PNL_BASE + ' 25% premium expense 200 fixed expense')
    idx = p.summary_df.index
    # two separate legs, default basis-derived names
    assert 'premium expense' in idx
    assert 'fixed expense' in idx
    assert 'expense' not in idx
    assert p.summary_df.loc['premium expense', 'EX'] == pytest.approx(250.0)
    assert p.summary_df.loc['fixed expense', 'EX'] == pytest.approx(200.0)


def test_labeled_expense_groups():
    p = build(_PNL_BASE + ' 25% premium expense as acq 30% loss expense as lae')
    idx = p.summary_df.index
    assert 'acq' in idx and 'lae' in idx
    assert p.summary_df.loc['acq', 'EX'] == pytest.approx(250.0)


def test_combine_vs_separate_total_matches():
    # combined (and) vs separate (juxtaposition) must give the same total expense
    combined = build(_PNL_BASE + ' 25% premium expense and 200 fixed expense')
    separate = build(_PNL_BASE + ' 25% premium expense 200 fixed expense')
    tot_c = combined.summary_df.loc['Total obligation', 'EX']
    tot_s = separate.summary_df.loc['Total obligation', 'EX']
    assert tot_c == pytest.approx(tot_s, rel=TOL)


def test_labeled_group_combines_multiple_terms():
    p = build(_PNL_BASE + ' 25% premium expense and 200 fixed expense as "acquisition"')
    assert 'acquisition' in p.summary_df.index
    assert p.summary_df.loc['acquisition', 'EX'] == pytest.approx(0.25 * 1000 + 200)


# ----------------------------------------------------------------------
# Reins-clause label -> margin_df cession column
# ----------------------------------------------------------------------
def test_reins_label_names_margin_column():
    p = build('pnl RP 1000 premium less 850 loss sev lognorm 100 cv 1 '
              'occurrence net of 100 xs 200 deposit 50 as "Cat XL" poisson')
    assert p.tower is not None
    assert 'Cat XL' in p.margin_df.columns
    assert 'ceded' not in p.margin_df.columns


def test_reins_no_label_keeps_structural_column():
    p = build('pnl RP 1000 premium less 850 loss sev lognorm 100 cv 1 '
              'occurrence net of 100 xs 200 deposit 50 poisson')
    assert 'ceded' in p.margin_df.columns


# ----------------------------------------------------------------------
# Round-trip: labels survive spec -> DecL -> spec
# ----------------------------------------------------------------------
@pytest.mark.parametrize("program", [
    'agg GrossBook as "Gross Book" 100 claims sev lognorm 100 cv 2 poisson',
    'sev MySev as lae lognorm 100 cv 2',
    'pnl Book as "My Book" 1000 premium as GWP less 850 loss '
    'sev lognorm 100 cv 1 poisson 25% premium expense as acq 30% loss expense as lae',
    'pnl RP 1000 premium less 850 loss sev lognorm 100 cv 1 '
    'occurrence net of 100 xs 200 deposit 50 as "Cat XL" poisson',
])
def test_label_roundtrip(uw, program):
    kind, name, spec = _parse(uw, program)
    rendered = spec_to_decl(spec, kind, name).replace('\n', ' ')
    _kind2, _name2, spec2 = _parse(uw, rendered)
    assert spec == spec2
