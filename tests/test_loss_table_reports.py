"""The three loss table report frames ([DecL-Loss-Tables], ``[Loss-Table-Reports]``).

``loss_table_summary_df``, ``loss_table_stats_df`` and
``loss_table_validation_df`` on an ``Aggregate``: what the declaration read,
the table's moments against the model and the discretized estimate, and the
checks with their status. Empty frames, with the same columns, for any other
aggregate.
"""

from pathlib import Path

import numpy as np
import pytest

from aggregate import build

DATA = Path(__file__).parent / 'data'
BASIC = 'agg LT.Basic elt [[1 .1 100] [2 .1 40] [3 .1 80]]'


def test_empty_for_an_ordinary_aggregate():
    a = build('agg LT.Plain 2 claims dsev [1 2] poisson', update=False)
    s, t, v = (a.loss_table_summary_df, a.loss_table_stats_df,
               a.loss_table_validation_df)
    assert s.empty and list(s.columns) == ['value']
    assert list(s.index.names) == ['part', 'item']
    assert t.empty and list(t.columns) == ['table', 'model', 'estimate', 'error']
    assert list(t.index.names) == ['component', 'measure']
    assert v.empty and list(v.columns) == [
        'check', 'value', 'tolerance', 'status', 'detail']


def test_elt_summary():
    a = build('agg LT.S elt no2u unit A as "Alpha" [[1 .1 50 100 10 10]] '
              'elt unit B [[1 .1 60 300] [2 .2 40 500]]', update=False)
    s = a.loss_table_summary_df['value']
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
    assert a.loss_table_summary_df.loc[('table', 'largest event exposure'),
                                       'value'] == 'n/a'


def test_yelt_summary_and_file_record():
    a = build(f'agg LT.Y yelt "{(DATA / "yelt/weighted.tsv").as_posix()}"')
    s = a.loss_table_summary_df['value']
    assert s[('table', 'kind')] == 'yelt'
    assert s[('table', 'years')] == 3
    assert s[('table', 'zero-loss years')] == 1
    assert s[('table', 'probability of zero')] == pytest.approx(0.5)
    assert s[('table', 'year probabilities')] == 'supplied'
    assert s[('grid', 'log2')] == a.log2
    part = 'source 1, unit Total'
    assert len(s[(part, 'sha256')]) == 64
    assert s[(part, 'ignored columns')] == 'none'


def test_stats_before_and_after_update():
    a = build(BASIC, update=False)
    t = a.loss_table_stats_df
    assert t.loc[('annual', 'mean'), 'table'] == pytest.approx(22)
    assert t.loc[('annual', 'mean'), 'model'] == pytest.approx(22)
    assert t['estimate'].isna().all()
    b = build(BASIC)
    t = b.loss_table_stats_df
    np.testing.assert_allclose(t['table'], t['model'], rtol=1e-12)
    assert (t['error'].abs() < b.validation_eps).all()


def test_symmetric_skew_compares_absolutely():
    a = build('agg LT.Sym elt [[1 .1 100] [2 .1 40]]')
    t = a.loss_table_stats_df
    assert t.loc[('severity', 'skew'), 'table'] == 0
    assert np.isfinite(t.loc[('severity', 'skew'), 'error'])


def test_validation_statuses():
    a = build(BASIC, update=False)
    v = a.loss_table_validation_df.set_index('check')
    assert v.loc['annual mean estimate', 'status'] == 'not computed'
    assert v.loc['severity mass', 'status'] == 'pass'
    b = build(BASIC)
    v = b.loss_table_validation_df
    assert (v['status'] == 'pass').all(), v


def test_degenerate_yelt_marks_undefined_measures():
    v = build('agg LT.One yelt [75]').loss_table_validation_df.set_index('check')
    assert v.loc['annual mean estimate', 'status'] == 'pass'
    assert v.loc['annual skew estimate', 'status'] == 'not computed'
    assert 'undefined' in v.loc['annual skew estimate', 'detail']


def test_frames_are_fresh_copies():
    a = build(BASIC, update=False)
    s = a.loss_table_summary_df
    s.loc[('table', 'form'), 'value'] = 'changed'
    assert a.loss_table_summary_df.loc[('table', 'form'), 'value'] == 'ELT3'
