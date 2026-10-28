"""Secondary uncertainty for ELT aggregates ([ELT-Uncertainty]).

An ELT6 ``[id rate mean max sdi sdc]`` without ``no2u`` builds an ordinary
``Aggregate`` whose severity is the rate-weighted mixture of each event's
total, beta, two-point or atom per row as eltq classifies it, a shared event's
rows rolled up into one row first. The model moments come from eltq before
any grid exists. Plans: ``dev/done/plan-elt-uncertainty.md`` and
``dev/done/plan-elt-shared-rollup.md``; the DecL programs are mirrored in
``decl-testers.agg`` section LT.

The cases that compute need eltq 2.13 or later and skip without it; the
refusal without eltq runs everywhere.
"""

import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from aggregate import Aggregate, _loss_table, build
from aggregate._loss_table import _check_rows, _moment_rows, _summary_rows
from aggregate.constants import DefectiveDistributionWarning

BETA = 'agg LT.Beta elt [[1 .1 50 100 10 10]]'
LARGE_1 = 'agg LT.Large1 elt [[1 .1 20000 1e9 60000 0]]'
LARGE_2 = 'agg LT.Large2 elt [[1 .1 100 1e9 10000 0]]'
REMOTE = 'agg LT.Remote elt [[1 .1 50 100 10 10] [2 5e-7 10 1e7 0 20000]]'
EPS = 1e-4
RUNG = 2 ** 0.25
TWO_POINT = 'agg LT.TwoPoint elt [[1 .1 50 100 30 30]]'
ATOM = 'agg LT.Atom elt [[1 .1 50 100 0 0]]'
COMONOTONE = ('agg LT.Comonotone elt unit A [[1 .1 50 100 0 60]] '
              'elt unit B [[1 .1 20 100 0 60]]')
INDEPENDENT = ('agg LT.Independent elt unit A [[1 .1 50 100 60 0]] '
               'elt unit B [[1 .1 20 100 60 0]]')
ROLLUP_A = 'elt unit A [[1 .1 30 300 20 10]]'
ROLLUP_B = 'elt unit B [[1 .1 20 200 15 10]]'
ROLLUP = f'agg LT.Rollup {ROLLUP_A} {ROLLUP_B}'
ROLLUP_THREE = ('agg LT.RollupThree '
                'elt unit A [[1 .1 30 300 20 10] [2 .2 40 100 5 5]] '
                'elt unit B [[1 .1 20 200 15 10] [3 .3 7 9 0 0]] '
                'elt unit C [[1 .1 10 50 0 0]]')
ROLLUP_LARGE = ('agg LT.RollupLarge elt unit A [[1 .1 10000 1e9 30000 0]] '
                'elt unit B [[1 .1 10000 1e9 30000 0]]')
#: Digests of one-row tables at 1.18.2, before the roll-up landed.
DIGESTS_1_18_2 = {
    'LT.Beta': 'bbc5b3d3c38fffa9817ce369ee1eb65bf79b0d82b9bdfb01bea2ad9b25266257',
    'LT.Disjoint': 'e60db2fda3a3eab5a79970e973cfa621577ee280f1a5512982604f2635d44886',
    'LT.Large1': 'dbdc47b5e60b8ad17b2ba7747fadc129e819b3fdb620ed1c811750b71088d7da',
}


@pytest.fixture
def eltq():
    """The eltq package, or skip the test."""
    return pytest.importorskip('eltq', minversion='2.13')


@pytest.fixture
def burke(monkeypatch):
    """The pended Burke total rule in place of the roll-up."""
    monkeypatch.setattr(_loss_table, 'SHARED_EVENT_RULE', 'burke')


def _homer(eltq, model):
    """eltq's own roll-up of the input rows, an independent oracle."""
    return eltq.multiunit_moments(
        event_id=model.event_ids,
        unit_id=np.asarray(model.units, dtype=object)[model.unit_code],
        rate=model.rate, mean_loss=model.mean, sd_indep=model.sdi,
        sd_corr=model.sdc, max_loss=model.max, coupling='homer',
        variance_policy='bernoulli')


def _annual(a):
    """Annual mean, variance and third cumulant from the model ``stats_df``."""
    e1, e2, e3 = (float(a.stats_df.loc[('agg', k), 'mixed'])
                  for k in ('ex1', 'ex2', 'ex3'))
    return e1, e2 - e1 ** 2, e3 - 3 * e1 * e2 + 2 * e1 ** 3


def _checks(a):
    """The loss table checks of ``a`` by name."""
    return {c.check: c for c in _check_rows(a)}


def _statuses(a):
    """The status of each loss table check of ``a``, by name."""
    return {c.check: c.status for c in _check_rows(a)}


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
    moments = _moment_rows(a)
    assert moments[('annual', 'mean')]['table'] == pytest.approx(5)
    assert moments[('annual', 'cv')]['table'] == pytest.approx(290 ** 0.5 / 5)
    np.testing.assert_allclose([r['table'] for r in moments.values()],
                               [r['model'] for r in moments.values()],
                               rtol=1e-12, atol=1e-12)
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
    v = _checks(a)
    assert v['row classification'].value == 1
    assert v['row classification'].status == 'pass'


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
def test_shared_event_coupling(eltq, burke, program, e2, e3, method):
    """Under the Burke rule, two two-point units, means 50 and 20 on M =
    100: Burke's loading 1 (all correlated SD) is comonotone, loading 0
    independent, both exact."""
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
        assert _checks(a)['moment integration'].status == 'pass'


def test_shared_refusal_names_the_event(eltq, burke):
    a = build(COMONOTONE, update=False)
    with pytest.raises(NotImplementedError,
                       match=r'event 1 has rows for units A, B.*'
                             r'\[ELT-Burke-Total\].*default roll-up'):
        a.update()


def test_shared_refusal_as_portfolio_unit(eltq, burke):
    with pytest.raises(NotImplementedError, match=r'\[ELT-Burke-Total\]'):
        build('port LT.SharedPort agg LT.SharedUnit elt unit A '
              '[[1 .1 50 100 0 60]] elt unit B [[1 .1 20 100 0 60]] '
              'agg LT.Other 2 claims dsev [10] poisson')


def test_burke_rule_moves_the_digest(eltq, monkeypatch):
    """The rule enters a shared table's digest; a bad rule is refused."""
    rollup = build(COMONOTONE, update=False).sevs[0].model
    monkeypatch.setattr(_loss_table, 'SHARED_EVENT_RULE', 'burke')
    burke = build(COMONOTONE, update=False).sevs[0].model
    assert (rollup.shared_rule, burke.shared_rule) == ('rollup', 'burke')
    assert rollup.digest != burke.digest
    assert burke.rolled_columns is None
    monkeypatch.setattr(_loss_table, 'SHARED_EVENT_RULE', 'matched')
    with pytest.raises(ValueError, match='SHARED_EVENT_RULE'):
        build(COMONOTONE, update=False)


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
    s = _summary_rows(a)
    assert s[('table', 'form')] == 'ELT6'
    assert s[('table', 'secondary uncertainty')] == '1 beta, 0 two-point, 0 atom'
    assert s[('table', 'largest event support')] == 100
    assert s[('table', 'model moments')] == 'closed form'
    assert s[('table', 'eltq')].startswith(eltq.__version__)
    assert 'ELT6, 1 events, 1 unit, secondary uncertainty' in a.info


# ----------------------------------------------------------------------
# [ELT-Single-Event-Uncertainty]: the severity on a grid
# ----------------------------------------------------------------------

def _quiet_build(program, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return build(program, **kwargs)


@pytest.mark.parametrize('program', [BETA, TWO_POINT, ATOM])
def test_fine_grid_oracles(eltq, program):
    """Quantization error separated from FFT error: every estimate within eps."""
    a = _quiet_build(program, bs=1 / 64, log2=16)
    errors = [r['error'] for r in _moment_rows(a).values()
              if np.isfinite(r['error'])]
    assert len(errors) >= 4
    assert all(abs(e) <= EPS for e in errors), errors
    q = a.sevs[0].last_quantization
    assert q['passed']
    assert q['mass'] == pytest.approx(0.1, rel=1e-12)
    v = _statuses(a)
    assert v['quantized mass against event rate'] == 'pass'
    assert v['secondary mean error'] == 'pass'


def test_zero_sd_matches_no2u_on_the_lattice(eltq):
    rows = '[[1 .1 50 100 0 0] [2 .2 30 80 0 0] [3 .05 7 9 0 0]]'
    a = _quiet_build(f'agg LT.Lattice6 elt {rows}', bs=1, log2=10)
    b = _quiet_build(f'agg LT.Lattice4 elt no2u {rows}', bs=1, log2=10)
    np.testing.assert_allclose(a.sev_density, b.sev_density, atol=1e-15)
    np.testing.assert_allclose(a.agg_density, b.agg_density, atol=1e-15)


def test_moments_fixed_across_grids(eltq):
    a = _quiet_build(BETA, bs=1 / 16, log2=12)
    model = a.stats_df['mixed'].copy()
    est = a.stats_df['gross_empirical'].copy()
    a.update(bs=1 / 4, log2=12)
    pd.testing.assert_series_equal(a.stats_df['mixed'], model)
    assert not np.allclose(a.stats_df['gross_empirical'].dropna(),
                           est.dropna(), rtol=1e-12, atol=0)


def test_no_quantization_before_update(eltq, monkeypatch):
    calls = []
    original = eltq.quantize_arrays

    def spy(*args, **kwargs):
        calls.append(kwargs.get('bucket_size'))
        return original(*args, **kwargs)

    monkeypatch.setattr(eltq, 'quantize_arrays', spy)
    a = build(BETA, update=False)
    assert calls == []
    a.update(bs=1 / 16, log2=12)
    assert calls == [1 / 16]
    a.update()
    assert len(calls) == 2


@pytest.mark.parametrize('program, maximum', [(LARGE_1, 1e9), (LARGE_2, 1e9)])
def test_large_exposure_automatic_grid(eltq, program, maximum):
    """No hints: the grid passes the tail check and its top is far below M."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', DefectiveDistributionWarning)
        a = build(program)
    assert a.log2 == 16
    q = a.sevs[0].last_quantization
    assert q['passed']
    assert a.bs * 2 ** a.log2 < maximum / 5


@pytest.mark.parametrize('program, mu, sd, passing', [
    (LARGE_1, 20_000, 60_000, 4.31e6),
    (LARGE_2, 100, 10_000, 15.6e6),
])
def test_ladder_edge_from_forty_sd(eltq, program, mu, sd, passing):
    """From an edge at mu + 40 sd the ladder passes within a rung of the
    plan's measurement (eltq 2.14.0, 2026-10-27)."""
    sev = build(program, update=False).sevs[0]
    n = 2 ** 16
    edge = mu + 40 * sd
    assert not sev.tail_report(edge, edge * (n - 1) / (n - 0.5))['passed'][0]
    t_k, _ = sev.passing_edge(edge, n)
    assert passing / RUNG <= t_k <= passing * RUNG


def test_forty_sd_pinned_grids_warn(eltq):
    """bs 40 and 6 put the 2**16 edge near mu + 40 sd: honored, warned,
    reported."""
    with pytest.warns(DefectiveDistributionWarning, match='clips the severity tail'):
        a = build(LARGE_1, bs=40, log2=16)
    v = _statuses(a)
    assert v['tail rate above edge'] == 'warning'
    assert v['tail third moment share'] == 'warning'
    assert a.bs == 40
    with pytest.warns(DefectiveDistributionWarning, match='clips the severity tail'):
        b = build(LARGE_2, bs=6, log2=16)
    v = _checks(b)
    for check in ('tail mean share', 'tail second moment share',
                  'tail third moment share'):
        assert v[check].status == 'warning'
    assert v['tail mean share'].value == pytest.approx(0.38, abs=0.02)


def test_remote_two_point_row(eltq):
    """A rate 5e-13 atom at M = 1e7 carries 15% of C2 and nearly all of C3
    while R and D1 pass below M: only the moment shares find it."""
    sev = build(REMOTE, update=False).sevs[0]
    c = np.asarray(sev.model.moments.raw_moments)
    assert 5e-13 * 1e14 / c[1] == pytest.approx(0.147, abs=0.01)
    assert 5e-13 * 1e21 / c[2] > 0.99
    report = sev.tail_report(5e6, 5e6)
    assert report['rate_above'][0] <= report['rate_target']
    assert report['shares'][0][0] < 1e-5
    assert not report['passed'][0]
    # automatic: the edge reaches M
    a = _quiet_build(REMOTE)
    q = a.sevs[0].last_quantization
    assert q['passed'] and q['edge'] >= 1e7
    assert a._bs_elt_tail['ladder_edge'] == pytest.approx(1e7)
    # an edge exactly at M keeps the atom on the grid
    n = 2 ** 10
    bs = 1e7 / (n - 0.5)
    while (n - 0.5) * bs < 1e7:
        bs = np.nextafter(bs, np.inf)
    pmf = sev.quantize(np.arange(n) * bs, bs)
    q = sev.last_quantization
    assert q['rate_above'] == 0
    assert q['top_half_rate'] == pytest.approx(5e-13, rel=1e-9)
    assert pmf[-1] * q['rate'] == pytest.approx(5e-13, rel=1e-9)
    # a pinned edge below M warns on the second and third moment shares
    with pytest.warns(DefectiveDistributionWarning, match='clips the severity tail'):
        b = build(REMOTE, bs=100, log2=16)
    v = _statuses(b)
    assert v['tail rate above edge'] == 'pass'
    assert v['tail mean share'] == 'pass'
    assert v['tail second moment share'] == 'warning'
    assert v['tail third moment share'] == 'warning'


def test_treaties_match_the_quantized_dsev(eltq):
    """Reinsurance acts on the discretized severity like any other."""
    grid = dict(bs=1 / 16, log2=12)
    gross = _quiet_build(BETA, **grid)
    keep = gross.sev_density > 0
    terms = 'occurrence net of 30 xs 20 aggregate net of 50 xs 10'
    a = _quiet_build(f'agg LT.BetaNet elt [[1 .1 50 100 10 10]] {terms}', **grid)
    # the same spec, treaties included, with the quantized severity as atoms
    spec = {**a.spec, 'name': 'LT.BetaHand', 'sev_name': 'dhistogram',
            'sev_xs': gross.xs[keep], 'sev_ps': gross.sev_density[keep]}
    spec.pop('_loss_table_meta')
    b = Aggregate(**spec)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        b.update(**grid)
    assert a.est_m < gross.est_m
    np.testing.assert_allclose(a.agg_density, b.agg_density, atol=1e-14)


def test_composition(eltq):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = build('port LT.Port6 agg LT.U6 elt [[1 .1 50 100 10 10]] '
                  'agg LT.U6b 4 claims dsev [10] poisson')
        assert p.agg_list[0].sevs[0].last_quantization is not None
        assert p.est_m == pytest.approx(5 + 40, rel=1e-4)
        pl = build('pnl LT.PnL6 30 premium less agg LT.PnL6_e '
                   'elt [[1 .1 50 100 10 10]] occurrence net of 50 xs 20')
        assert np.isfinite(pl.est_m)
        build('agg LT.Ref6 elt [[1 .1 50 100 10 10]]', update=False)
        r = build('agg LT.Ref6Net agg.LT.Ref6 occurrence net of 50 xs 20',
                  bs=1 / 16, log2=12)
        d = build('agg LT.Ref6Dir elt [[1 .1 50 100 10 10]] '
                  'occurrence net of 50 xs 20', bs=1 / 16, log2=12)
        np.testing.assert_allclose(r.agg_density, d.agg_density, atol=1e-15)
        a = build(BETA)
        model = a.stats_df['mixed'].copy()
        a.sharpen()
        pd.testing.assert_series_equal(a.stats_df['mixed'], model)


def test_disjoint_units_update_like_one_unit(eltq):
    grid = dict(bs=1 / 8, log2=12)
    a = _quiet_build('agg LT.Disjoint elt unit Wind [[1 .1 50 100 10 10]] '
                     'elt unit Quake [[2 .2 30 200 5 20]]', **grid)
    b = _quiet_build('agg LT.OneUnit elt [[1 .1 50 100 10 10] '
                     '[2 .2 30 200 5 20]]', **grid)
    np.testing.assert_allclose(a.agg_density, b.agg_density, atol=1e-15)


def test_grid_rows_before_and_after_update(eltq):
    a = build(BETA, update=False)
    v = _statuses(a)
    assert v['tail rate above edge'] == 'not computed'
    assert v['secondary mean error'] == 'not computed'
    assert _summary_rows(a)[('grid', 'severity edge')] == 'n/a'
    a.update(bs=1 / 16, log2=12)
    s = _summary_rows(a)
    assert s[('grid', 'severity edge')] == pytest.approx((2 ** 12 - 0.5) / 16)
    assert s[('grid', 'tail rate above edge')] == 0


# ----------------------------------------------------------------------
# [ELT-Shared-Event-Rollup]: a shared event rolled up into one row
# ----------------------------------------------------------------------

def test_rollup_oracle(eltq):
    """[30 300 20 10] and [20 200 15 10] roll to [50 500 25 20], a
    beta(1.01111, 9.1) with SD 45; eltq's homer coupling agrees."""
    a = _quiet_build(ROLLUP)
    sev = a.sevs[0]
    assert sev.shared and sev.grid_supported
    np.testing.assert_allclose(sev.moms(), (50, 4525, 562511.4678899),
                               rtol=1e-12)
    np.testing.assert_allclose(sev.moms(),
                               _homer(eltq, sev.model).severity_moments,
                               rtol=1e-12)
    assert abs(a.est_m / 5 - 1) <= a.validation_eps
    assert sev.last_quantization['passed']
    assert '1 shared events rolled up' in a.info
    secondary = a._loss_table_meta['secondary']
    assert secondary['shared_rule'] == 'rollup'
    assert (secondary['shared_events'], secondary['shared_rows'],
            secondary['shared_inadmissible_rows']) == (1, 2, 0)
    assert (secondary['beta_rows'], secondary['two_point_rows']) == (1, 0)
    s = _summary_rows(a)
    assert s[('table', 'shared events')] == (
        '1 events from 2 rows, rolled up; 0 of the rows have inadmissible '
        'variance')
    assert 'moment integration' not in _checks(a)


@pytest.mark.parametrize('program, e2, e3, laws', [
    (COMONOTONE, 14_000, 2_800_000, (0, 1)),
    (INDEPENDENT, 12_100, 2_236_644.1717791, (1, 0)),
])
def test_rollup_corpus_regressions(eltq, program, e2, e3, laws):
    """Two inadmissible unit rows each: comonotone rolls to the two-point
    [70 200 0 120], independent to the beta [70 200 84.853 0]."""
    a = _quiet_build(program)
    sev = a.sevs[0]
    np.testing.assert_allclose(sev.moms(), (70, e2, e3), rtol=1e-12)
    assert sev.last_quantization is not None
    assert a.est_m == pytest.approx(7, rel=1e-3)
    secondary = a._loss_table_meta['secondary']
    assert (secondary['beta_rows'], secondary['two_point_rows']) == laws
    assert _summary_rows(a)[('table', 'shared events')] \
        == '1 events from 2 rows, rolled up; 2 of the rows have inadmissible variance'


def test_rollup_portfolio_unit_updates(eltq):
    p = _quiet_build(f'port LT.RollupPort agg LT.RollupUnit {ROLLUP_A} '
                     f'{ROLLUP_B} agg LT.RollupOther 2 claims dsev [10] poisson')
    assert p.agg_list[0].sevs[0].last_quantization is not None
    assert p.est_m == pytest.approx(5 + 20, rel=1e-4)


def test_rollup_three_units_and_atoms(eltq):
    """Event 1 has three units, one an atom, beside two one-row events: the
    rolled row by hand, and the one-row events bitwise."""
    model = build(ROLLUP_THREE, update=False).sevs[0].model
    cols = model.columns()
    assert list(model.event_ids) == ['1', '1', '1', '2', '3']
    for key, expect in (('rate', [.1, .2, .3]), ('mean_loss', [60, 40, 7]),
                        ('max_loss', [550, 100, 9]), ('sd_indep', [25, 5, 0]),
                        ('sd_corr', [20, 5, 0])):
        assert cols[key].tolist() == expect, key
        assert not cols[key].flags.writeable
    hand = build('agg LT.RollupHand elt [[1 .1 60 550 25 20] [2 .2 40 100 5 5] '
                 '[3 .3 7 9 0 0]]', update=False).sevs[0].model
    assert list(model.moments.raw_moments) == list(hand.moments.raw_moments)
    np.testing.assert_allclose(model.moments.raw_moments,
                               _homer(eltq, model).raw_moments, rtol=1e-12)
    assert model.shared_audit() == (1, 3, 0)
    assert model.rows == 5 and model.events == 3


@pytest.mark.parametrize('program', [BETA, LARGE_1,
                                     'agg LT.Disjoint elt unit Wind '
                                     '[[1 .1 50 100 10 10]] elt unit Quake '
                                     '[[2 .2 30 200 5 20]]'])
def test_one_row_tables_unchanged(eltq, program):
    a = build(program, update=False)
    model = a.sevs[0].model
    assert model.digest == DIGESTS_1_18_2[a.name]
    assert model.shared_rule is None and model.rolled_columns is None
    assert ('table', 'shared events') not in _summary_rows(a)
    assert 'rolled up' not in a.info


def test_rollup_feasibility(eltq):
    """Admissible unit rows, two to five per event, always roll into an
    admissible beta (Cauchy-Schwarz)."""
    rng = np.random.default_rng(20261028)
    units = rng.integers(2, 6, size=100)
    clauses = []
    for u in range(5):
        rows = []
        for e in np.flatnonzero(units > u):
            m = rng.uniform(10, 1000)
            mu = rng.uniform(0.05, 0.95) * m
            sd = rng.uniform(0.01, 0.99) * (mu * (m - mu)) ** 0.5
            w = rng.uniform()
            rows.append(f'[{e + 1} .01 {mu:.12g} {m:.12g} {w * sd:.12g} '
                        f'{(1 - w) * sd:.12g}]')
        clauses.append(f"elt unit U{u} [{' '.join(rows)}]")
    a = build(f"agg LT.RollupRandom {' '.join(clauses)}", update=False)
    model = a.sevs[0].model
    assert model.shared_audit() == (100, int(units.sum()), 0)
    cols = model.columns()
    sd = cols['sd_indep'] + cols['sd_corr']
    room = cols['mean_loss'] * (cols['max_loss'] - cols['mean_loss'])
    assert (sd * sd < room).all()
    secondary = a._loss_table_meta['secondary']
    assert (secondary['beta_rows'], secondary['two_point_rows']) == (100, 0)


def test_rollup_tail_check_runs(eltq):
    """Two [1 .1 10000 1e9 30000 0] rows roll to M = 2e9; the tail check
    runs and passes far below it, as LT.Large1 does."""
    from aggregate._bucket_window import _elt_severity
    with warnings.catch_warnings():
        warnings.simplefilter('error', DefectiveDistributionWarning)
        a = build(ROLLUP_LARGE)
    assert _elt_severity(a) is a.sevs[0]
    assert a.sevs[0].model.moments.max_support == 2e9
    assert a.sevs[0].last_quantization['passed']
    assert a.bs * 2 ** a.log2 < 2e9 / 5


@pytest.mark.parametrize('program, swapped', [
    (ROLLUP, f'agg LT.RollupSwap {ROLLUP_B} {ROLLUP_A}'),
    (ROLLUP_THREE, 'agg LT.RollupThreeSwap elt unit C [[1 .1 10 50 0 0]] '
                   'elt unit B [[1 .1 20 200 15 10] [3 .3 7 9 0 0]] '
                   'elt unit A [[1 .1 30 300 20 10] [2 .2 40 100 5 5]]'),
])
def test_rollup_clause_order(eltq, program, swapped):
    """Summation order within an event can move the last bit, no more."""
    a = build(program, update=False).sevs[0].model.moments
    b = build(swapped, update=False).sevs[0].model.moments
    np.testing.assert_allclose(a.raw_moments, b.raw_moments, rtol=1e-15)


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
    build('agg LT.NoEltqY yelt [0 100]', update=False)
