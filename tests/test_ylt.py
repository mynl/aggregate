"""Year loss tables in DecL ([DecL-Loss-Tables], ``[YLT-Probability-Oracles]``).

``agg NAME ylt ...`` lowers YLT1 (losses), YLT2 (``[year loss]``) and YLT3
(``[year loss prob]``) to ``dfreq [1] dsev``: each year is a whole annual
outcome, units are summed within a year before equal totals coalesce, and a
year's probability is counted once. The DecL programs are mirrored in
``decl-testers.agg`` section LT.
"""

import warnings

import numpy as np
import pytest

from aggregate import build
from aggregate._loss_table import LossTableError
from aggregate.decl_writer import spec_to_decl
from aggregate.parser import UnderwritingLexer


def _law(program):
    """The lowered annual atoms and probabilities, and the theoretical moments."""
    a = build(program, update=False)
    return a.spec['sev_xs'], a.spec['sev_ps'], a.actual_m, a.actual_var


@pytest.mark.parametrize('program, xs, ps, mean, var', [
    ('agg LT.Y1 ylt [0 0 100 300]', [0, 100, 300], [.5, .25, .25], 100, 15000),
    ('agg LT.Y2 ylt [[1 0] [2 0] [3 100] [4 300]]',
     [0, 100, 300], [.5, .25, .25], 100, 15000),
    ('agg LT.Y3u ylt [[1 0 .25] [2 0 .25] [3 100 .25] [4 300 .25]]',
     [0, 100, 300], [.5, .25, .25], 100, 15000),
    ('agg LT.Y3 ylt [[1 0 .5] [2 100 .3] [3 300 .2]]',
     [0, 100, 300], [.5, .3, .2], 90, 12900),
    ('agg LT.Coalesce ylt [[1 0 .5] [2 100 .3] [3 100 .2]]',
     [0, 100], [.5, .5], 50, 2500),
    ('agg LT.Aligned ylt unit A [[1 0] [2 100]] ylt unit B [[1 100] [2 0]]',
     [100], [1], 100, 0),
    ('agg LT.One ylt [75]', [75], [1], 75, 0),
    ('agg LT.Zeros ylt [0 0 0]', [0], [1], 0, 0),
    ('agg LT.ZeroProb ylt [[1 0 .5] [2 100 .5] [3 1e12 0]]',
     [0, 100], [.5, .5], 50, 2500),
])
def test_ylt_oracles(program, xs, ps, mean, var):
    x, p, m, v = _law(program)
    np.testing.assert_allclose(x, xs)
    np.testing.assert_allclose(p, ps, rtol=1e-12)
    assert m == pytest.approx(mean, rel=1e-12, abs=1e-12)
    assert v == pytest.approx(var, rel=1e-12, abs=1e-9)


@pytest.mark.parametrize('program, mean, var', [
    ('agg LT.Net ylt [0 0 100 300] aggregate net of 150 xs 50', 50, 3750),
    ('agg LT.Net3 ylt [[1 0 .5] [2 100 .3] [3 300 .2]] '
     'aggregate net of 150 xs 50', 45, 3225),
])
def test_annual_reinsurance(program, mean, var):
    a = build(program, log2=10, bs=1)
    assert a.est_m == pytest.approx(mean, rel=1e-12)
    assert a.est_var == pytest.approx(var, rel=1e-9)


def test_equals_dfreq_one_dsev_on_the_same_grid():
    a = build('agg LT.Y ylt [0 0 100 300]', log2=10, bs=1)
    b = build('agg LT.D dfreq [1] dsev [0 0 100 300]', log2=10, bs=1)
    assert a.one_claim
    np.testing.assert_array_equal(a.agg_density, b.agg_density)


def test_units_are_summed_within_years_not_convolved():
    """Independent convolution of the two units would give variance 5,000."""
    a = build('agg LT.Aligned ylt unit A [[1 0] [2 100]] '
              'ylt unit B [[1 100] [2 0]]', log2=10, bs=1)
    assert a.est_var == pytest.approx(0, abs=1e-9)
    assert a._loss_table_meta['alignment'] == 'year id'


def test_nonuniform_probability_counted_once():
    x, p, m, _ = _law('agg LT.Two ylt unit A [[1 10 .2] [2 20 .8]] '
                      'ylt unit B [[2 5 .8] [1 1 .2]]')
    np.testing.assert_allclose(x, [11, 25])
    np.testing.assert_allclose(p, [.2, .8])
    assert m == pytest.approx(.2 * 11 + .8 * 25)


def test_reordered_year_ids_give_the_same_law():
    a = _law('agg LT.O1 ylt unit A [[1 0] [2 100] [3 50]] '
             'ylt unit B [[1 5] [2 0] [3 7]]')
    b = _law('agg LT.O2 ylt unit A [[3 50] [1 0] [2 100]] '
             'ylt unit B [[2 0] [3 7] [1 5]]')
    for u, v in zip(a, b):
        np.testing.assert_allclose(u, v)


def test_positional_vectors_pair_by_position():
    x, p, _, _ = _law('agg LT.Pos ylt unit A [0 100] ylt unit B [100 0]')
    np.testing.assert_allclose(x, [100])


def test_uniform_ylt3_mixes_with_ylt1():
    x, p, _, _ = _law('agg LT.Mix ylt unit A [0 100] '
                      'ylt unit B [[1 1 .5] [2 2 .5]]')
    np.testing.assert_allclose(x, [1, 102])


def test_audit_counts():
    a = build('agg LT.Audit ylt [[1 0 .5] [2 0 0] [3 300 .5]]', update=False)
    meta = a._loss_table_meta
    assert meta['form'] == 'YLT3'
    assert meta['years'] == 3
    assert meta['years_zero_probability'] == 1
    assert meta['zero_years'] == 1
    assert meta['zero_probability'] == pytest.approx(.5)
    assert 'YLT3, 3 years, 1 unit, supplied year probabilities' in a.info


def test_roundoff_in_probabilities_is_normalized_and_recorded():
    a = build('agg LT.Round ylt [[1 0 (1/3)] [2 1 (1/3)] [3 2 (1/3)]]',
              update=False)
    assert a.spec['sev_ps'].sum() == pytest.approx(1, abs=1e-15)
    assert abs(a._loss_table_meta['probability_residual']) < 1e-12


@pytest.mark.parametrize('program', [
    'agg LT.Y ylt [0 0 100 300] aggregate net of 150 xs 50 note{four years}',
    'agg LT.Y3 as "Weighted" ylt [[1 0 0.5] [2 100 0.3] [3 300 0.2]]',
    'agg LT.U ylt unit Personal as "Personal lines" [[1 0] [2 100]] '
    'ylt unit Commercial [["a" 100] ["b" 0]]',
    'agg LT.F ylt unit Personal "personal-years.csv"',
    'pnl LT.PnL 120 premium less agg LT.PnL_e ylt [0 0 100 300] '
    'aggregate net of 150 xs 50',
])
def test_round_trip(program, underwriter):
    kind, name, spec = underwriter.parser.parse(program)
    text = spec_to_decl(spec, kind=kind, name=name)
    statement = UnderwritingLexer.preprocess(text)[0]
    assert underwriter.parser.parse(statement)[2] == spec


def test_pnl_with_annual_terms():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('pnl LT.PnL 120 premium less agg LT.PnL_e ylt [0 0 100 300] '
                  'aggregate net of 150 xs 50', log2=10, bs=1)
        b = build('pnl LT.Hand 120 premium less agg LT.Hand_e dfreq [1] '
                  'dsev [0 0 100 300] aggregate net of 150 xs 50', log2=10, bs=1)
    assert a.est_m == pytest.approx(b.est_m, rel=1e-12)


@pytest.mark.parametrize('program', [
    'agg LT.Bad agg.LT.YRef occurrence net of 50 xs 50',
    'pnl LT.BadP 100 premium less agg LT.BadP_e agg.LT.YRef '
    'occurrence net of 50 xs 50',
])
def test_occurrence_terms_refused_through_a_reference(program):
    build('agg LT.YRef ylt [0 0 100 300]', update=False)
    with pytest.raises(LossTableError, match='annual outcomes'):
        build(program, update=False)


@pytest.mark.parametrize('program, match', [
    ('agg LT.E ylt [0 100] occurrence net of 50 xs 50', 'annual outcomes'),
    ('agg LT.E ylt no2u [0 100]', 'parse error'),
    ('agg LT.E ylt [0 100] elt [[1 .1 1]]', 'parse error'),
    ('agg LT.E ylt [[1 0] [2 1 .5]]', 'mixed widths'),
    ('agg LT.E ylt [[1 0 .5 7]]', 'has 4 fields'),
    ('agg LT.E ylt [0 1] approximate gamma', 'approximate'),
])
def test_invalid_declarations(program, match):
    with pytest.raises(ValueError, match=match):
        build(program, update=False)


@pytest.mark.parametrize('program, match', [
    ('agg LT.E ylt unit A [[1 0] [2 100]] ylt unit B [[1 100] [3 0]]',
     'same complete set of years'),
    ('agg LT.E ylt unit A [[1 0] [2 100]] ylt unit B [[1 100]]',
     'same complete set of years'),
    ('agg LT.E ylt unit A [0 100 5] ylt unit B [100 0]',
     'same complete set of years'),
    ('agg LT.E ylt [[1 0] [1 100]]', 'more than once'),
    ('agg LT.E ylt unit A [0 1] ylt unit A [1 0]', 'two ylt sources'),
    ('agg LT.E ylt [0 -1]', 'losses must be finite'),
    ('agg LT.E ylt [[1 0 .5] [2 100 .3] [3 300 .3]]', 'sum to'),
    ('agg LT.E ylt [[1 0 0] [2 100 0]]', 'every probability is zero'),
    ('agg LT.E ylt [[1 0 -.5] [2 100 1.5]]', 'probabilities must be finite'),
    ('agg LT.E ylt unit A [[1 0 .4] [2 100 .6]] ylt unit B [[1 0 .5] [2 3 .5]]',
     'different probability'),
    ('agg LT.E ylt unit A [[1 0 .4] [2 100 .6]] ylt unit B [[1 0] [2 3]]',
     'equally likely'),
])
def test_invalid_tables_fail_before_building(program, match):
    with pytest.raises(LossTableError, match=match):
        build(program, update=False)
