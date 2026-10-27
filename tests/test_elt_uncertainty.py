"""Secondary uncertainty for ELT aggregates ([ELT-Uncertainty]).

An ELT6 ``[id rate mean max sdi sdc]`` without ``no2u`` builds an ordinary
``Aggregate`` whose severity is the rate-weighted mixture of each event's
total, beta, two-point or atom per row as eltq classifies it, summed under
Burke's coupling for a shared event. The model moments come from eltq before
any grid exists. Plan: ``dev/done/plan-elt-uncertainty.md``; the DecL programs
are mirrored in ``decl-testers.agg`` section LT.

The cases that compute need eltq 2.13 or later and skip without it; the
refusal without eltq runs everywhere.
"""

import sys

import numpy as np
import pandas as pd
import pytest

from aggregate import Aggregate, build

BETA = 'agg LT.Beta elt [[1 .1 50 100 10 10]]'
TWO_POINT = 'agg LT.TwoPoint elt [[1 .1 50 100 30 30]]'
ATOM = 'agg LT.Atom elt [[1 .1 50 100 0 0]]'
COMONOTONE = ('agg LT.Comonotone elt unit A [[1 .1 50 100 0 60]] '
              'elt unit B [[1 .1 20 100 0 60]]')
INDEPENDENT = ('agg LT.Independent elt unit A [[1 .1 50 100 60 0]] '
               'elt unit B [[1 .1 20 100 60 0]]')


@pytest.fixture
def eltq():
    """The eltq package, or skip the test."""
    return pytest.importorskip('eltq', minversion='2.13')


def _annual(a):
    """Annual mean, variance and third cumulant from the model ``stats_df``."""
    e1, e2, e3 = (float(a.stats_df.loc[('agg', k), 'mixed'])
                  for k in ('ex1', 'ex2', 'ex3'))
    return e1, e2 - e1 ** 2, e3 - 3 * e1 * e2 + 2 * e1 ** 3


# ----------------------------------------------------------------------
# [ELT-Uncertainty-Moments]: model moments with no grid
# ----------------------------------------------------------------------

def test_beta_row_moments(eltq):
    """beta(2.625, 2.625) on [0, 100]: annual mean 5, variance 290, C3 18,500."""
    a = build(BETA, update=False)
    assert type(a.sevs[0]).__name__ == 'SeverityElt'
    m, v, k3 = _annual(a)
    assert m == pytest.approx(5, rel=1e-12)
    assert v == pytest.approx(290, rel=1e-12)
    assert k3 == pytest.approx(18_500, rel=1e-12)
    stats = a.loss_table_stats_df
    assert stats.loc[('annual', 'mean'), 'table'] == pytest.approx(5)
    assert stats.loc[('annual', 'cv'), 'table'] == pytest.approx(290 ** 0.5 / 5)
    np.testing.assert_allclose(stats['table'], stats['model'], rtol=1e-12,
                               atol=1e-12)
    secondary = a._loss_table_meta['secondary']
    assert (secondary['beta_rows'], secondary['two_point_rows'],
            secondary['atom_rows']) == (1, 0, 0)


def test_two_point_row(eltq):
    """SDs summing to 60 exceed the beta bound: two points on {0, 100}."""
    a = build(TWO_POINT, update=False)
    m, v, _ = _annual(a)
    assert (m, v) == (pytest.approx(5), pytest.approx(500))
    secondary = a._loss_table_meta['secondary']
    assert secondary['rate_at_zero'] == pytest.approx(0.05)
    v = a.loss_table_validation_df.set_index('check')
    assert v.loc['row classification', 'value'] == 1
    assert v.loc['row classification', 'status'] == 'pass'


def test_atom_row_matches_no2u(eltq):
    a = build(ATOM, update=False)
    b = build('agg LT.AtomNo2u elt no2u [[1 .1 50 100 0 0]]', update=False)
    assert _annual(a) == pytest.approx(_annual(b), rel=1e-12)
    assert a.actual_var == pytest.approx(250)
    assert a._loss_table_meta['secondary']['atom_rows'] == 1


def test_zero_mean_event_keeps_its_rate(eltq):
    a = build('agg LT.ZeroMean elt [[1 .2 0 100 0 0] [2 .2 50 100 10 10]]',
              update=False)
    assert a.n == pytest.approx(0.4)
    assert a.actual_m == pytest.approx(10)


def test_all_zero_table(eltq):
    a = build('agg LT.AllZero6 elt [[1 .1 0 0 0 0] [2 .3 0 100 0 0]]',
              update=False)
    meta = a._loss_table_meta
    assert a.n == pytest.approx(0.4)
    assert a.actual_m == 0
    assert meta['events'] == 2
    assert meta['aal'] == 0
    assert meta['secondary']['atom_rows'] == 2


def test_disjoint_units_are_one_row_events(eltq):
    """Wind and quake with distinct ids: the same moments as one unit."""
    a = build('agg LT.Disjoint elt unit Wind [[1 .1 50 100 10 10]] '
              'elt unit Quake [[2 .2 30 200 5 20]]', update=False)
    b = build('agg LT.OneUnit elt [[1 .1 50 100 10 10] [2 .2 30 200 5 20]]',
              update=False)
    assert not a.sevs[0].shared
    assert _annual(a) == pytest.approx(_annual(b), rel=1e-12)
    assert a._loss_table_meta['units'] == ['Wind', 'Quake']


@pytest.mark.parametrize('program, e2, e3, method', [
    (COMONOTONE, 11_000, 1_900_000, 'factor_integration'),
    (INDEPENDENT, 9_000, 1_300_000, 'closed_form'),
])
def test_shared_event_coupling(eltq, program, e2, e3, method):
    """Two two-point units, means 50 and 20 on M = 100: Burke's loading 1
    (all correlated SD) is comonotone, loading 0 independent, both exact."""
    a = build(program, update=False)
    sev = a.sevs[0]
    assert sev.shared
    assert sev.moms() == pytest.approx((70, e2, e3), rel=1e-8)
    assert a.n == pytest.approx(0.1)
    meta = a._loss_table_meta
    assert meta['secondary']['method'] == method
    assert meta['secondary']['converged']
    assert 'shared events (model moments only)' in a.info
    if method == 'factor_integration':
        v = a.loss_table_validation_df.set_index('check')
        assert v.loc['moment integration', 'status'] == 'pass'


def test_update_refuses_until_the_grid_lands(eltq):
    a = build(BETA, update=False)
    with pytest.raises(NotImplementedError,
                       match=r'\[ELT-Single-Event-Uncertainty\]'):
        a.update()
    with pytest.raises(NotImplementedError, match='update=False'):
        build(BETA)


def test_shared_refusal_names_the_event(eltq):
    a = build(COMONOTONE, update=False)
    with pytest.raises(NotImplementedError,
                       match=r'event 1 has rows for units A, B.*'
                             r'\[ELT-Shared-Event-Uncertainty\]'):
        a.update()


def test_shared_refusal_as_portfolio_unit(eltq):
    with pytest.raises(NotImplementedError,
                       match=r'\[ELT-Shared-Event-Uncertainty\]'):
        build('port LT.SharedPort agg LT.SharedUnit elt unit A '
              '[[1 .1 50 100 0 60]] elt unit B [[1 .1 20 100 0 60]] '
              'agg LT.Other 2 claims dsev [10] poisson')


def test_rebuild_keeps_the_model_without_the_file(eltq, tmp_path):
    path = tmp_path / 'elt6.csv'
    pd.DataFrame({'id': [1, 2], 'rate': [.1, .2], 'mean': [50, 30],
                  'max': [100, 200], 'sdi': [10, 5], 'sdc': [10, 20]}
                 ).to_csv(path, index=False)
    a = build(f'agg LT.File6 elt "{path.as_posix()}"', update=False)
    path.unlink()
    b = Aggregate(**a.spec)
    assert b.sevs[0].model is a.sevs[0].model
    assert b.actual_var == pytest.approx(a.actual_var, rel=1e-15)
    assert 'sha256=' in repr(a.sevs[0].model)


def test_spec_hash_is_content_addressed(eltq):
    first = build(BETA, update=False)._spec_hash()
    assert build(BETA, update=False)._spec_hash() == first
    changed = build(BETA.replace('10 10', '10 11'), update=False)._spec_hash()
    assert changed != first


def test_summary_and_info_rows(eltq):
    a = build(BETA, update=False)
    s = a.loss_table_summary_df['value']
    assert s[('table', 'form')] == 'ELT6'
    assert s[('table', 'secondary uncertainty')] == '1 beta, 0 two-point, 0 atom'
    assert s[('table', 'largest event support')] == 100
    assert s[('table', 'model moments')] == 'closed form'
    assert s[('table', 'eltq')].startswith(eltq.__version__)
    assert 'ELT6, 1 events, 1 unit, secondary uncertainty' in a.info


# ----------------------------------------------------------------------
# Without eltq: refused with guidance, everything else unchanged
# ----------------------------------------------------------------------

def test_without_eltq(monkeypatch):
    monkeypatch.setitem(sys.modules, 'eltq', None)
    with pytest.raises(ImportError, match='no2u'):
        build('agg LT.NoEltq elt [[1 .1 50 100 10 10]]', update=False)
    build('agg LT.NoEltq3 elt [[1 .1 50]]', update=False)
    build('agg LT.NoEltq4 elt [[1 .1 50 100]]', update=False)
    build('agg LT.NoEltqNo2u elt no2u [[1 .1 50 100 10 10]]', update=False)
    build('agg LT.NoEltqY ylt [0 100]', update=False)
