"""[Occurrence-Annual-Terms]: an annual aggregate deductible and an aggregate
limit on an occurrence layer.

``aad d`` / ``agglimit m`` decorate one occurrence layer, directly after the
layer, so the reinsurer pays ``min((R - d)+, m)`` on the layer's annual
recovery ``R``. Amounts are for 100% of the layer and the placement share
applies after. Step ``[Annual-Terms-DecL]`` covers the grammar, the spec keys,
the refusals and the plain ``agg``: on ``occurrence ceded to`` the terms are
honored as the aggregate stage, exactly; on ``net of`` they are ignored with an
:class:`~aggregate.constants.IgnoredDecLClauseWarning` and the stored program
keeps them for a ``pnl`` / ``xpnl``. See
``dev/done/plan-occurrence-annual-deductible-limit.md``.

These DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg``
under the ``OAT.*`` section.
"""

import numpy as np
import pytest

from aggregate import Underwriter
from aggregate.constants import IgnoredDecLClauseWarning

#: A small book; the layer clause is spliced in.
_BOOK = 'agg {name} 10 claims sev lognorm 100 cv 2 occurrence {kind} {layer} poisson {tail}'


def _prog(name, layer, kind='net of', tail=''):
    """One plain ``agg`` program on the small book."""
    return _BOOK.format(name=name, kind=kind, layer=layer, tail=tail).strip()


@pytest.fixture
def uw():
    """A clean underwriter (no databases) with eager update."""
    return Underwriter(databases=None, update=True)


def _occ_keys(uw, program):
    """The occurrence spec keys of one parsed program."""
    _, _, spec = uw.parser.parse(program)
    return {k: v for k, v in spec.items() if k.startswith('occ_')}


# ----------------------------------------------------------------------
# grammar / transformer: the three forms, with and without a share
# ----------------------------------------------------------------------
@pytest.mark.parametrize('layer, share, aad, agglimit', [
    ('90% po 100 xs 100 aad 50', 0.9, 50.0, np.inf),
    ('750 xs 250 agglimit 500', 1.0, 0.0, 500.0),
    ('650 xs 350 aad 50 agglimit 500', 1.0, 50.0, 500.0),
    ('675 po 750 xs 250 aad 50 agglimit 500', 0.9, 50.0, 500.0),
])
def test_forms_parse_to_the_spec_keys(uw, layer, share, aad, agglimit):
    occ = _occ_keys(uw, _prog('T', layer))
    assert occ['occ_reins'][0][0] == pytest.approx(share)
    assert occ['occ_reins_aad'] == aad
    assert occ['occ_reins_agglimit'] == agglimit


def test_terms_sit_before_the_premium(uw):
    occ = _occ_keys(uw, _prog(
        'T', '750 xs 250 aad 50 agglimit 500 deposit 100 cede 10% as Layer'))
    assert occ['occ_reins_aad'] == 50.0
    assert occ['occ_reins_agglimit'] == 500.0
    assert occ['occ_reins_premium'] == [('deposit', 100.0)]
    assert occ['occ_reins_cede'] == [0.1]
    assert occ['occ_reins_label'] == ['Layer']


def test_undecorated_layer_has_no_annual_keys(uw):
    occ = _occ_keys(uw, _prog('T', '750 xs 250'))
    assert 'occ_reins_aad' not in occ
    assert 'occ_reins_agglimit' not in occ


# ----------------------------------------------------------------------
# refusals
# ----------------------------------------------------------------------
@pytest.mark.parametrize('program, match', [
    ('agg T 10 claims sev lognorm 100 cv 2 poisson aggregate net of 750 xs 250 aad 50',
     'aggregate layer is itself'),
    (_prog('T', '750 xs 250 aad 50 and 1000 xs 1000'),
     'single occurrence layer'),
    (_prog('T', '750 xs 250 aad 50 deposit 100 reinstatements 1 free'),
     "cannot combine with 'reinstatements'"),
    (_prog('T', '750 xs 250 aad -5'), "'aad' must be finite and >= 0"),
    (_prog('T', '750 xs 250 agglimit 0'), "'agglimit' must be finite and > 0"),
    (_prog('T', '750 xs 250 agglimit inf'), "'agglimit' must be finite and > 0"),
    (_prog('T', '750 xs 250 aad 50 aad 60'), 'parse error'),
])
def test_refusals(uw, program, match):
    with pytest.raises(ValueError, match=match):
        uw.parser.parse(program)


# ----------------------------------------------------------------------
# plain agg, ``ceded to``: honored as the aggregate stage, exactly
# ----------------------------------------------------------------------
def test_ceded_to_equals_the_hand_written_aggregate_stage(uw):
    a = uw(_prog('A', '90% po 750 xs 250 aad 50 agglimit 500', 'ceded to'))
    b = uw(_prog('B', '90% po 750 xs 250', 'ceded to',
                 'aggregate ceded to 450 xs 45'))
    np.testing.assert_array_equal(a.agg_density, b.agg_density)
    assert a.agg_reins == [(1.0, 450.0, 45.0)]
    assert a.agg_kind == 'ceded to'
    assert a.labels.agg_reins == {0: 'AAD 50, agglimit 500'}


def test_ceded_to_aad_alone_is_unlimited(uw):
    a = uw(_prog('A', '750 xs 250 aad 50', 'ceded to'))
    assert a.agg_reins == [(1.0, np.inf, 50.0)]
    assert a.labels.agg_reins == {0: 'AAD 50'}


def test_ceded_to_with_an_aggregate_clause_raises(uw):
    with pytest.raises(ValueError, match="cannot also carry an 'aggregate'"):
        uw(_prog('A', '750 xs 250 aad 50', 'ceded to',
                 'aggregate ceded to 300 xs 0'))


def test_ceded_to_keeps_the_declaration(uw):
    uw(_prog('A', '750 xs 250 aad 50 agglimit 500', 'ceded to'))
    stored = uw[('agg', 'A')].spec
    assert stored['occ_reins_aad'] == 50.0
    assert stored['occ_reins_agglimit'] == 500.0
    assert 'agg_reins' not in stored


# ----------------------------------------------------------------------
# plain agg, ``net of``: ignored with one warning, kept for a P&L
# ----------------------------------------------------------------------
def test_net_of_is_ignored_with_one_warning(uw):
    with pytest.warns(IgnoredDecLClauseWarning) as rec:
        a = uw(_prog('N', '750 xs 250 aad 50 agglimit 500'))
    ours = [w for w in rec
            if issubclass(w.category, IgnoredDecLClauseWarning)]
    assert len(ours) == 1
    assert 'aad / agglimit' in str(ours[0].message)
    plain = uw(_prog('P', '750 xs 250'))
    np.testing.assert_array_equal(a.agg_density, plain.agg_density)


def test_net_of_keeps_the_declaration(uw):
    with pytest.warns(IgnoredDecLClauseWarning):
        a = uw(_prog('N', '750 xs 250 aad 50 agglimit 500'))
    stored = uw[('agg', 'N')].spec
    assert stored['occ_reins_aad'] == 50.0
    assert stored['occ_reins_agglimit'] == 500.0
    assert 'aad 50 agglimit 500' in a.program


# ----------------------------------------------------------------------
# the writer renders the terms back as written
# ----------------------------------------------------------------------
@pytest.mark.parametrize('layer', [
    '90% po 100 xs 100 aad 50',
    '750 xs 250 agglimit 500',
    '750 xs 250 aad 50 agglimit 500 deposit 100 cede 0.1',
    '750 xs 250 aad 0',
])
def test_writer_round_trips(uw, layer):
    from aggregate.decl_writer import format_program
    program = _prog('W', layer)
    _, _, spec = uw.parser.parse(program)
    again = format_program(program, layout='terse')
    _, _, spec2 = uw.parser.parse(again)
    assert spec2 == spec
    assert layer.split(' xs ')[1].split(' ', 1)[1] in again


# ----------------------------------------------------------------------
# bvagg components follow the same two routes
# ----------------------------------------------------------------------
@pytest.fixture
def uwb():
    """A clean underwriter with lazy update: the routes act in ``__init__``."""
    return Underwriter(databases=None, update=False)


def test_bvagg_net_of_is_ignored_naming_the_unit(uwb):
    with pytest.warns(IgnoredDecLClauseWarning, match='V1: ignoring the aad'):
        b = uwb('grossceded agg V1 dfreq [3] dsev [2 4 6 8 10 12] '
                'occurrence net of 6 xs 6 aad 2')
    assert b._nc_agg.occ_reins is not None


def test_bvagg_ceded_to_becomes_the_aggregate_stage(uwb):
    b = uwb('bivariate V2 5 claims '
            'agg V2a dfreq [0 1] [.5 .5] dsev [2 4 6 8 10 12] '
            'occurrence ceded to 6 xs 6 aad 2 agglimit 3 '
            'agg V2b dfreq [0 1] [.5 .5] dsev [1 2 3 4] copula normal 0.5')
    assert b.units[0].agg_reins == [(1.0, 3.0, 2.0)]
    assert b.units[0].agg_kind == 'ceded to'
