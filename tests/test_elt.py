"""Event loss tables in DecL, the deterministic forms ([DecL-Loss-Tables]).

``agg NAME elt ...`` lowers an ELT3 or ELT4 table, or an ELT6 under ``no2u``, to
compound Poisson: one count per physical event, the event's unit means summed
into one occurrence loss, equal losses coalesced with their rates. The oracles
are the ``[ELT-Probability-Oracles]`` table of ``dev/done/plan-elt.md``; the
DecL programs are mirrored in ``decl-testers.agg`` section LT.
"""

import warnings

import numpy as np
import pytest

from aggregate import Aggregate, build
from aggregate._loss_table import LossTableError
from aggregate.decl_writer import spec_to_decl
from aggregate.parser import UnderwritingLexer


BASIC = 'agg LT.Basic as "My easy example" elt [[1 .1 100] [2 .1 40] [3 .1 80]]'


def _moments(program):
    """Theoretical annual mean and variance, before any grid is built."""
    a = build(program, update=False)
    return a.n, a.actual_m, a.actual_var


@pytest.mark.parametrize('program, rate, mean, var', [
    (BASIC, 0.3, 22, 1800),
    ('agg LT.Max elt [[1 .1 100 1000] [2 .1 40 600] [3 .1 80 2000]]',
     0.3, 22, 1800),
    ('agg LT.Shared elt unit A [[1 .1 100]] elt unit B [[1 .1 40]]',
     0.1, 14, 1960),
    ('agg LT.Distinct elt unit A [[1 .1 100]] elt unit B [[2 .1 40]]',
     0.2, 14, 1160),
    ('agg LT.MeanOnly elt no2u [[1 .1 50 100 10 10]]', 0.1, 5, 250),
])
def test_elt_oracles(program, rate, mean, var):
    n, m, v = _moments(program)
    assert n == pytest.approx(rate, rel=1e-12)
    assert m == pytest.approx(mean, rel=1e-12)
    assert v == pytest.approx(var, rel=1e-12)


def test_shared_event_is_one_occurrence():
    """A shared event sums its unit losses into one atom, counted once."""
    a = build('agg LT.Shared elt unit A [[1 .1 100]] elt unit B [[1 .1 40]]',
              update=False)
    np.testing.assert_allclose(a.spec['sev_xs'], [140])
    assert a._loss_table_meta['events'] == 1
    assert a._loss_table_meta['units'] == ['A', 'B']


def test_elt4_maximums_are_reporting_only():
    """ELT4 and ELT3 of the same means give the same law on the same grid."""
    a = build(BASIC, log2=10, bs=1)
    b = build('agg LT.Max elt [[1 .1 100 1000] [2 .1 40 600] [3 .1 80 2000]]',
              log2=10, bs=1)
    np.testing.assert_array_equal(a.agg_density, b.agg_density)
    assert a._loss_table_meta['max_event_exposure'] is None
    assert b._loss_table_meta['form'] == 'ELT4'
    assert b._loss_table_meta['max_event_exposure'] == 2000


def test_elt3_with_elt4_drops_maximums():
    a = build('agg LT.Mixed elt unit A [[1 .1 100 1000]] elt unit B [[2 .1 40]]',
              update=False)
    meta = a._loss_table_meta
    assert meta['form'] == 'ELT3'
    assert meta['max_event_exposure'] is None


def test_no2u_projection_combines_with_elt4():
    a = build('agg LT.Proj elt no2u unit A [[1 .1 50 100 10 10]] '
              'elt unit B [[2 .1 40 400]]', update=False)
    meta = a._loss_table_meta
    assert meta['form'] == 'ELT4'
    assert [s['effective_form'] for s in meta['sources']] == ['ELT4', 'ELT4']
    assert meta['sources'][0]['form'] == 'ELT6'
    assert meta['max_event_exposure'] == 400


def test_zero_loss_event_keeps_its_rate():
    """P(A = 0) = exp(-rate (1 - p0)), not exp(-rate)."""
    a = build('agg LT.Zero elt [[1 .2 0] [2 .2 100]]', log2=10, bs=1)
    assert a.n == pytest.approx(0.4)
    assert a.agg_density[0] == pytest.approx(np.exp(-0.4 * 0.5), rel=1e-12)


def test_all_zero_table_is_degenerate_at_zero():
    a = build('agg LT.AllZero elt [[1 .1 0] [2 .3 0]]')
    assert a.est_m == 0
    assert a.agg_density[0] == pytest.approx(1.0)


def test_zero_rate_events_are_excluded_and_counted():
    a = build('agg LT.ZeroRate elt [[1 0 100] [2 .1 40]]', update=False)
    assert a.n == pytest.approx(0.1)
    assert a.actual_m == pytest.approx(4)
    assert a._loss_table_meta['events_zero_rate'] == 1


def test_large_integer_ids_stay_distinct():
    """2**53 and 2**53 + 1 are one float but two events."""
    a = build('agg LT.BigIds elt unit A [[9007199254740992 .1 100]] '
              'elt unit B [[9007199254740993 .1 40]]', update=False)
    assert a.n == pytest.approx(0.2)
    assert a._loss_table_meta['events'] == 2


def test_integer_and_string_ids():
    """7 and "7" are one event; "007" is another."""
    a = build('agg LT.Ids elt unit A [[7 .1 100] ["007" .1 5]] '
              'elt unit B [["7" .1 40]]', update=False)
    assert a._loss_table_meta['events'] == 2
    assert a.actual_m == pytest.approx(0.1 * 140 + 0.1 * 5)


def test_row_order_and_partition_preserve_the_law():
    a = build('agg LT.P1 elt [[1 .1 100] [2 .2 40] [3 .1 80]]', log2=10, bs=1)
    b = build('agg LT.P2 elt [[3 .1 80]] elt [[2 .2 40] [1 .1 100]]',
              log2=10, bs=1)
    np.testing.assert_allclose(a.agg_density, b.agg_density, atol=1e-16)


def test_matches_hand_written_dsev_with_treaties():
    """The lowering is the ordinary compound Poisson dsev, treaties included."""
    tail = 'occurrence net of 100 xs 50 aggregate net of 200 xs 100'
    a = build(f'agg LT.T elt unit A [[1 .1 60] [2 .2 30]] elt unit B [[1 .1 60]] {tail}',
              log2=12, bs=1)
    b = build('agg LT.H 0.3 claims dsev [120 30] [1/3 2/3] '
              'occurrence net of 100 xs 50 poisson aggregate net of 200 xs 100',
              log2=12, bs=1)
    np.testing.assert_allclose(a.agg_density, b.agg_density, atol=1e-15)


def test_occurrence_treaty_sees_the_event_total():
    """Not each unit separately: 60 + 60 under 100 xs 50 nets 50, not 100."""
    a = build('agg LT.Occ elt unit A [[1 .1 60]] elt unit B [[1 .1 60]] '
              'occurrence net of 100 xs 50', log2=10, bs=1)
    assert a.est_m == pytest.approx(0.1 * 50, rel=1e-9)


def test_model_moments_before_update():
    a = build(BASIC, update=False)
    assert a.bs == 0
    assert a.actual_m == pytest.approx(22)
    assert 'ELT3, 3 events, 1 unit' in a.info


def test_reconstruction_keeps_the_audit():
    a = build(BASIC)
    b = Aggregate(**a.spec)
    assert b._loss_table_meta == a._loss_table_meta
    assert b.actual_m == pytest.approx(a.actual_m)


def test_ordinary_spec_has_no_audit_key():
    """Unset, the key is dropped, so ordinary spec hashes do not move."""
    a = build('agg LT.Plain 1 claim dsev [1 2] fixed', update=False)
    assert '_loss_table_meta' not in a.spec
    assert a._loss_table_meta is None


@pytest.mark.parametrize('program', [
    BASIC,
    'agg LT.S as "Shared" elt no2u unit Personal as "Personal lines" '
    '[[1 0.1 300 3000 400 300] [2 0.1 100 1000 371 267]] '
    'elt unit Commercial [[1 0.1 200 1000] ["007" 0.1 200 4000]] '
    'occurrence net of 1000 xs 500 aggregate net of 2000 xs 1000 hints{log2=12}',
    'agg LT.Path elt unit Wind "data dir/wind #1 [v2].csv" '
    'elt unit Quake "//server/quake.parquet"',
    'pnl LT.PnL 30 premium less agg LT.PnL_e elt [[1 0.1 100] [2 0.1 40]] '
    'occurrence net of 50 xs 50',
])
def test_round_trip(program, underwriter):
    kind, name, spec = underwriter.parser.parse(program)
    text = spec_to_decl(spec, kind=kind, name=name)
    statement = UnderwritingLexer.preprocess(text)[0]
    assert underwriter.parser.parse(statement)[2] == spec


def test_quoted_path_is_verbatim(underwriter):
    path = 'data dir/wind #1 [v2]//x.csv'
    spec = underwriter.parser.parse(
        UnderwritingLexer.preprocess(f'agg LT.Q elt "{path}"')[0])[2]
    assert spec['elt_sources'][0]['path'] == path


def test_portfolio_and_pnl_embedding():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = build('port LT.Port agg LT.U1 elt [[1 .1 100]] '
                  'agg LT.U2 4 claims dsev [10] poisson')
        assert p.agg_list[0].actual_m == pytest.approx(10)
        assert p.agg_list[0]._loss_table_meta['form'] == 'ELT3'
        pl = build('pnl LT.PnL 30 premium less agg LT.PnL_e '
                   'elt [[1 .1 100] [2 .1 40]] occurrence net of 50 xs 50',
                   log2=10, bs=1)
        hand = build('pnl LT.Hand 30 premium less agg LT.Hand_e 0.2 claims '
                     'dsev [100 40] occurrence net of 50 xs 50 poisson',
                     log2=10, bs=1)
    assert pl.est_m == pytest.approx(hand.est_m, rel=1e-12)


def test_reference_takes_occurrence_terms():
    build('agg LT.Ref elt [[1 .1 100] [2 .1 40]]', update=False)
    a = build('agg LT.RefNet agg.LT.Ref occurrence net of 50 xs 50',
              log2=10, bs=1)
    assert a.est_m == pytest.approx(0.1 * 50 + 0.1 * 40, rel=1e-9)


@pytest.mark.parametrize('program', [
    'agg LT.Bad 2 @ agg.LT.Ref',
    'agg LT.Bad 2 * agg.LT.Ref',
    'agg LT.Bad agg.LT.Ref + 10',
])
def test_algebra_on_a_table_is_refused(program):
    build('agg LT.Ref elt [[1 .1 100] [2 .1 40]]', update=False)
    with pytest.raises(ValueError, match='declared from an elt'):
        build(program, update=False)


def test_bivariate_body_is_refused():
    with pytest.raises(ValueError, match='elt or ylt body is not supported'):
        build('bivariate LT.BV 1 claim agg LT.B1 elt [[1 .1 100]] '
              'agg LT.B2 1 claim dsev [1] fixed copula normal 0.5', update=False)


@pytest.mark.parametrize('program, error, match', [
    ('agg LT.E elt [[1 .1 50 100 10 10]]', NotImplementedError,
     'secondary uncertainty'),
    ('agg LT.E elt [[1 .1 50 100 10 10]] elt no2u [[2 .1 50 100 10 10]]',
     LossTableError, 'Add no2u to elt source 1'),
    ('agg LT.E elt [[1 .1 50 100 10 10]] elt [[2 .1 50]]',
     LossTableError, 'cannot be combined'),
    ('agg LT.E elt [[1 .1 50]] elt [[1 .1 60]]', LossTableError,
     'more than once'),
    ('agg LT.E elt unit A [[1 .1 50]] elt unit B [[1 .2 60]]',
     LossTableError, 'different rates'),
    ('agg LT.E elt [[1 .1 50 40]]', LossTableError, 'mean exceeds max'),
    ('agg LT.E elt [[1 -.1 50]]', LossTableError, 'rate must be finite'),
    ('agg LT.E elt no2u [[1 .1 50 100 -1 10]]', LossTableError,
     'sdi must be finite'),
    ('agg LT.E elt [[1 0 50]]', LossTableError, 'no event with a positive rate'),
    ('agg LT.E elt unit A as x [[1 .1 1]] elt unit A as y [[2 .1 1]]',
     LossTableError, 'labeled both'),
])
def test_invalid_tables_fail_before_building(program, error, match):
    with pytest.raises(error, match=match):
        build(program, update=False)


@pytest.mark.parametrize('program, match', [
    ('agg LT.E elt [[1.5 .1 100]]', 'parse error'),
    ('agg LT.E elt [[1 .1 100] [2 .1]]', 'mixed widths'),
    ('agg LT.E elt [[1 .1 100 4 5]]', 'has 5 fields'),
    ('agg LT.E elt [[1 .1 100]] approximate gamma', 'approximate'),
    ('agg LT.E elt [[1 .1 100]] 2 claims', 'parse error'),
    ('agg LT.E elt [["" .1 100]]', 'identifier is empty'),
])
def test_invalid_declarations_fail_to_parse(program, match):
    with pytest.raises(ValueError, match=match):
        build(program, update=False)
