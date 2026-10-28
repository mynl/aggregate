"""The loss table reports ([DecL-Loss-Tables], ``[ELT-Reporting-Punchup]``).

``elt_description`` and ``elt_explanation`` on an ``Aggregate``, the short and
long narratives of an ``elt`` or ``yelt`` declaration, and ``elt_df``, the
table the model uses. The facts and checks behind both narratives are the
private ``_summary_rows`` and ``_check_rows``, which the tests inspect
directly. Empty strings and an empty frame for any other aggregate.
"""

from pathlib import Path
import warnings

import numpy as np
import pytest

from aggregate import Aggregate, _loss_table, build
from aggregate._loss_table import _check_rows, _moment_rows, _summary_rows
from aggregate.constants import DefectiveDistributionWarning, info_row

DATA = Path(__file__).parent / 'data'
BASIC = 'agg LT.Basic elt [[1 .1 100] [2 .1 40] [3 .1 80]]'
NO2U = ('agg LT.S elt no2u unit A as "Alpha" [[1 .1 50 100 10 10]] '
        'elt unit B [[1 .1 60 300] [2 .2 40 500]]')
YELT = 'agg LT.Y3 yelt [[1 0 .5] [2 100 .3] [3 300 .2]]'
BETA = 'agg LT.Beta elt [[1 .1 50 100 10 10]]'
DISJOINT = ('agg LT.Disjoint elt unit Wind [[1 .1 50 100 10 10]] '
            'elt unit Quake [[2 .2 30 200 5 20]]')
ROLLUP = ('agg LT.Rollup elt unit A [[1 .1 30 300 20 10]] '
          'elt unit B [[1 .1 20 200 15 10]]')
LARGE_1 = 'agg LT.Large1 elt [[1 .1 20000 1e9 60000 0]]'
RETIRED = ('loss_table_summary_df', 'loss_table_stats_df',
           'loss_table_validation_df')


@pytest.fixture
def eltq():
    """The eltq package, or skip the test."""
    return pytest.importorskip('eltq', minversion='2.13')


def _lines(*rows):
    """The expected ``elt_description``, one ``info_row`` per pair."""
    return '\n'.join(info_row(label, value) for label, value in rows)


def _statuses(a):
    """The status of each loss table check of ``a``, by name."""
    return {c.check: c.status for c in _check_rows(a)}


def _quiet_build(program, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return build(program, **kwargs)


# ----------------------------------------------------------------------
# Not a loss table, and the retired frames
# ----------------------------------------------------------------------

def test_empty_for_an_ordinary_aggregate():
    a = build('agg A 1 claim dsev [1:6] poisson')
    assert a.elt_description == ''
    assert a.elt_explanation == ''
    assert a.elt_df.empty
    assert _summary_rows(a) == {} and _check_rows(a) == []


@pytest.mark.parametrize('name', RETIRED)
def test_retired_frames_are_gone(name):
    assert not hasattr(Aggregate, name)
    assert not hasattr(build(BASIC, update=False), name)


# ----------------------------------------------------------------------
# [Loss-Table-Description]
# ----------------------------------------------------------------------

def test_elt3_description_before_and_after_update():
    a = build(BASIC, update=False)
    assert a.elt_description == _lines(
        ('loss table', 'ELT3, 3 events from 3 rows, 1 unit'),
        ('event rate', '0.3, mean annual loss 22'),
        ('grid', 'n/a'),
        ('checks', '7 pass, 6 not computed'))
    a.update(bs=1, log2=10)
    assert a.elt_description.splitlines()[-2:] == [
        info_row('grid', 'bs 1, log2 10'), info_row('checks', '13 pass')]


def test_elt4_no2u_description():
    a = build(NO2U, update=False)
    assert a.elt_description == _lines(
        ('loss table', 'ELT4, 2 events from 3 rows, 2 units: A (Alpha), B'),
        ('event rate', '0.3, mean annual loss 19'),
        ('secondary uncertainty', 'removed by no2u'),
        ('grid', 'n/a'),
        ('checks', '7 pass, 6 not computed'))


def test_zero_rate_events_are_counted():
    a = build('agg LT.ZeroRate elt [[1 .1 100] [2 .1 40] [3 0 80]]', update=False)
    assert a.elt_description.splitlines()[0] == info_row(
        'loss table', 'ELT3, 2 events from 3 rows, 1 unit; 1 zero-rate event '
        'excluded')
    assert '1 event with zero rate is excluded.' in a.elt_explanation


def test_yelt_description_before_and_after_update():
    a = build(YELT, update=False)
    assert a.elt_description == _lines(
        ('loss table', 'YELT3, 3 years, 1 unit, supplied year probabilities'),
        ('annual outcomes', '3 outcomes, probability of zero 0.5, mean annual '
         'loss 90'),
        ('grid', 'n/a'),
        ('checks', '5 pass, 6 not computed'))
    a.update(bs=1, log2=10)
    assert a.elt_description.splitlines()[-2:] == [
        info_row('grid', 'bs 1, log2 10'), info_row('checks', '11 pass')]


@pytest.mark.parametrize('program, table, rate, laws', [
    (BETA, 'ELT6, 1 event from 1 row, 1 unit', '0.1, mean annual loss 5',
     '1 beta, 0 two-point, 0 atom rows'),
    (DISJOINT, 'ELT6, 2 events from 2 rows, 2 units: Wind, Quake',
     '0.3, mean annual loss 11', '2 beta, 0 two-point, 0 atom rows'),
    (ROLLUP, 'ELT6, 1 event from 2 rows, 2 units: A, B',
     '0.1, mean annual loss 5',
     '1 beta, 0 two-point, 0 atom rows; 1 shared event rolled up'),
])
def test_elt6_description_before_and_after_update(eltq, program, table, rate,
                                                  laws):
    a = build(program, update=False)
    assert a.elt_description == _lines(
        ('loss table', table), ('event rate', rate),
        ('secondary uncertainty', laws), ('grid', 'n/a'),
        ('checks', '7 pass, 12 not computed'))
    a.update(bs=1 / 64, log2=16)
    assert a.elt_description.splitlines()[-2:] == [
        info_row('grid', 'bs 1/64, log2 16, severity edge 1,024'),
        info_row('checks', '19 pass')]


# ----------------------------------------------------------------------
# [Loss-Table-Explanation]
# ----------------------------------------------------------------------

def _assert_every_check_once(a):
    """Three paragraphs, and each check of _check_rows stated once."""
    paragraphs = a.elt_explanation.split('\n\n')
    assert len(paragraphs) == 3
    for c in _check_rows(a):
        name = c.check[:1].upper() + c.check[1:]
        assert paragraphs[2].count(f'{name}: {c.status}') == 1, c.check
    return paragraphs


@pytest.mark.parametrize('program', [BASIC, NO2U, YELT])
def test_explanation_states_every_check_once(program):
    a = build(program, update=False)
    paragraphs = _assert_every_check_once(a)
    assert 'Severity mean estimate: not computed: needs update.' in paragraphs[2]
    a.update(bs=1, log2=10)
    _assert_every_check_once(a)


@pytest.mark.parametrize('program', [BETA, DISJOINT, ROLLUP])
def test_elt6_explanation_states_every_check_once(eltq, program):
    a = build(program, update=False)
    paragraphs = _assert_every_check_once(a)
    assert 'Tail rate above edge: not computed: needs update.' in paragraphs[2]
    assert 'computes the model moments in closed form' in paragraphs[0]
    a.update(bs=1 / 64, log2=16)
    _assert_every_check_once(a)


def test_explanation_of_the_rollup(eltq):
    model = build(ROLLUP, update=False).elt_explanation.split('\n\n')[0]
    assert 'rolled up into one row each, here 1 event from 2 rows' in model
    assert 'Of those rows, 0 have inadmissible variance.' in model


def test_warning_is_named_and_listed_first(eltq):
    """A pinned bs that clips the tail: the warnings lead the checks."""
    with pytest.warns(DefectiveDistributionWarning, match='clips the severity tail'):
        a = build(LARGE_1, bs=40, log2=16)
    warned = [c.check for c in _check_rows(a) if c.status == 'warning']
    assert {'tail rate above edge', 'tail third moment share'} <= set(warned)
    checks = a.elt_description.splitlines()[-1]
    assert f"warnings ({', '.join(warned)})" in checks
    para = _assert_every_check_once(a)[2]
    assert ', warnings first.' in para
    first = max(para.index(f'{w[:1].upper() + w[1:]}: warning') for w in warned)
    assert first < para.index('Form projection: pass')


def test_explanation_sources_of_a_file():
    a = build(f'agg LT.YF yelt "{(DATA / "yelt/weighted.tsv").as_posix()}"')
    s = _summary_rows(a)
    part = 'source 1, unit Total'
    sources = a.elt_explanation.split('\n\n')[1]
    assert sources.startswith('Source 1, unit Total: file ')
    assert s[(part, 'sha256')][:12] in sources
    assert f"{s[(part, 'bytes')]:,d} bytes" in sources
    assert 'YELT3, 3 rows.' in sources


def test_explanation_sources_under_no2u():
    sources = build(NO2U, update=False).elt_explanation.split('\n\n')[1]
    assert sources == ('Source 1, unit A (Alpha): inline, ELT6 read as ELT4 '
                       'under no2u, 1 row. Source 2, unit B: inline, ELT4, '
                       '2 rows.')


# ----------------------------------------------------------------------
# [Loss-Table-Df]
# ----------------------------------------------------------------------

def test_elt3_df_excludes_zero_rate_events():
    df = build('agg LT.ZeroRate elt [[1 .1 100] [2 .1 40] [3 0 80]]',
               update=False).elt_df
    assert df.index.name == 'event'
    assert list(df.index) == ['1', '2']
    assert list(df.columns) == ['rate', 'mean', 'units']
    assert df['mean'].tolist() == [100, 40]


def test_elt4_df_sums_units_by_hand():
    df = build('agg LT.UnitsMax elt unit A [[1 .1 60 300] [2 .2 40 500]] '
               'elt unit B [[3 .3 7 9] [1 .1 50 100]]', update=False).elt_df
    assert list(df.columns) == ['rate', 'mean', 'max', 'units']
    assert list(df.index) == ['1', '2', '3']
    assert df['rate'].tolist() == [.1, .2, .3]
    assert df['mean'].tolist() == [60 + 50, 40, 7]
    assert df['max'].tolist() == [300 + 100, 500, 9]
    assert df['units'].tolist() == [2, 1, 1]


def test_no2u_df_is_elt4():
    df = build(NO2U, update=False).elt_df
    assert list(df.columns) == ['rate', 'mean', 'max', 'units']
    assert df.loc['1', 'mean'] == 110 and df.loc['1', 'units'] == 2


def test_yelt_df():
    df = build('agg LT.ZeroProb yelt [[1 0 .5] [2 100 .5] [3 1e12 0]]',
               update=False).elt_df
    assert df.index.name == 'year'
    assert list(df.columns) == ['prob', 'loss']
    assert list(df.index) == ['1', '2']
    assert df['prob'].sum() == pytest.approx(1, abs=1e-15)
    aligned = build('agg LT.Aligned yelt unit A [[1 0] [2 100]] '
                    'yelt unit B [[1 100] [2 0]]', update=False).elt_df
    assert aligned['loss'].tolist() == [100, 100]
    assert aligned['prob'].tolist() == [.5, .5]


@pytest.mark.parametrize('program, ids, units', [
    (BETA, ['1'], [1]),
    (DISJOINT, ['1', '2'], [1, 1]),
    (ROLLUP, ['1'], [2]),
])
def test_elt6_df_is_what_eltq_receives(eltq, program, ids, units):
    a = build(program, update=False)
    df = a.elt_df
    assert list(df.columns) == ['rate', 'mean', 'max', 'sdi', 'sdc', 'units']
    assert list(df.index) == ids and df['units'].tolist() == units
    cols = a.sevs[0].model.columns()
    for column, key in (('rate', 'rate'), ('mean', 'mean_loss'),
                        ('max', 'max_loss'), ('sdi', 'sd_indep'),
                        ('sdc', 'sd_corr')):
        np.testing.assert_array_equal(df[column].to_numpy(), cols[key])


def test_rolled_row(eltq):
    """[30 300 20 10] and [20 200 15 10] roll to [50 500 25 20]."""
    row = build(ROLLUP, update=False).elt_df.loc['1']
    assert row[['rate', 'mean', 'max', 'sdi', 'sdc', 'units']].tolist() == [
        .1, 50, 500, 25, 20, 2]


def test_burke_rule_df_shows_the_input_rows(eltq, monkeypatch):
    monkeypatch.setattr(_loss_table, 'SHARED_EVENT_RULE', 'burke')
    df = build(ROLLUP, update=False).elt_df
    assert list(df.index) == ['1', '1']
    assert df['units'].tolist() == [1, 1]
    assert df['mean'].tolist() == [30, 20]


def test_elt_df_is_a_fresh_frame():
    a = build(BASIC, update=False)
    df = a.elt_df
    df.loc['1', 'mean'] = -1
    assert a.elt_df.loc['1', 'mean'] == 100


# ----------------------------------------------------------------------
# [Loss-Table-Private-Helpers]: the facts and checks behind the strings
# ----------------------------------------------------------------------

def test_elt_summary():
    s = _summary_rows(build(NO2U, update=False))
    assert s[('table', 'form')] == 'ELT4'
    assert s[('table', 'units')] == 'A, B'
    assert s[('table', 'unit labels')] == 'A: Alpha'
    assert s[('table', 'events')] == 2
    assert s[('table', 'event rate')] == pytest.approx(0.3)
    assert s[('table', 'mean annual loss')] == pytest.approx(0.1 * 110 + 0.2 * 40)
    assert s[('table', 'largest event exposure')] == 500
    assert s[('table', 'secondary uncertainty')] == 'removed by no2u'
    assert s[('grid', 'bs')] == 'n/a'
    assert s[('source 1, unit A', 'form')] == 'ELT6'
    assert s[('source 1, unit A', 'effective form')] == 'ELT4'


def test_elt3_combination_reports_no_exposure():
    a = build('agg LT.M elt unit A [[1 .1 100 1000]] elt unit B [[2 .1 40]]',
              update=False)
    assert _summary_rows(a)[('table', 'largest event exposure')] == 'n/a'
    assert list(a.elt_df.columns) == ['rate', 'mean', 'units']


def test_yelt_summary_and_file_record():
    a = build(f'agg LT.Y yelt "{(DATA / "yelt/weighted.tsv").as_posix()}"')
    s = _summary_rows(a)
    assert s[('table', 'kind')] == 'yelt'
    assert s[('table', 'years')] == 3
    assert s[('table', 'zero-loss years')] == 1
    assert s[('table', 'probability of zero')] == pytest.approx(0.5)
    assert s[('table', 'year probabilities')] == 'supplied'
    assert s[('grid', 'log2')] == a.log2
    part = 'source 1, unit Total'
    assert len(s[(part, 'sha256')]) == 64
    assert s[(part, 'ignored columns')] == 'none'


def test_moments_before_and_after_update():
    m = _moment_rows(build(BASIC, update=False))
    assert m[('annual', 'mean')]['table'] == pytest.approx(22)
    assert m[('annual', 'mean')]['model'] == pytest.approx(22)
    assert all(np.isnan(r['estimate']) for r in m.values())
    b = build(BASIC)
    m = _moment_rows(b)
    np.testing.assert_allclose([r['table'] for r in m.values()],
                               [r['model'] for r in m.values()], rtol=1e-12)
    assert all(abs(r['error']) < b.validation_eps for r in m.values())


def test_symmetric_skew_compares_absolutely():
    m = _moment_rows(build('agg LT.Sym elt [[1 .1 100] [2 .1 40]]'))
    assert m[('severity', 'skew')]['table'] == 0
    assert np.isfinite(m[('severity', 'skew')]['error'])


def test_check_statuses():
    v = _statuses(build(BASIC, update=False))
    assert v['annual mean estimate'] == 'not computed'
    assert v['severity mass'] == 'pass'
    assert set(_statuses(build(BASIC)).values()) == {'pass'}


def test_degenerate_yelt_marks_undefined_measures():
    checks = {c.check: c for c in _check_rows(build('agg LT.One yelt [75]'))}
    assert checks['annual mean estimate'].status == 'pass'
    assert checks['annual skew estimate'].status == 'not computed'
    assert 'undefined' in checks['annual skew estimate'].detail
