"""Ceded premium (``deposit`` / ``rol`` / ``rate``) and ceding commission
(``cede``) on a ``pnl`` (Phase 1, decision 3).

A premium clause on any reinsurance layer promotes the ``pnl`` to the
Gross/Ceded/Net exhibit (any-clause -> GCN). The premium resolves to currency
(``deposit`` an amount, ``rol`` = share x rol x limit, ``rate`` = rate x gross
premium); the commission ``cede x ceded_premium`` books as an expense credit.
See ``dev/done/plan-pnl-expenses-ceded-premium.md``.
"""
import pytest

from aggregate import build

TOL = 1e-3
_BASE = 'pnl T 5000 prem less 100 claims sev lognorm 50 cv 1.5 '


def test_deposit_rol_rate_resolution():
    assert build(_BASE + 'poisson aggregate net of 2000 xs 3000 deposit 1500'
                 )._gcn_econ['pc_agg'] == pytest.approx(1500.0)
    # rol: share (1.0) x 8% x limit (2000)
    assert build(_BASE + 'poisson aggregate net of 2000 xs 3000 rol 8%'
                 )._gcn_econ['pc_agg'] == pytest.approx(0.08 * 2000)
    # rate: 30% of the gross premium (5000)
    assert build(_BASE + 'poisson aggregate net of 2000 xs 3000 rate 30%'
                 )._gcn_econ['pc_agg'] == pytest.approx(0.30 * 5000)


def test_deposit_and_rol_coincide_when_equal():
    rol = build(_BASE + 'poisson aggregate net of 2000 xs 3000 rol 8%')
    dep = build(_BASE + 'poisson aggregate net of 2000 xs 3000 deposit 160')
    assert rol._gcn_econ['ceded'] == pytest.approx(dep._gcn_econ['ceded'])


def test_cede_books_commission_credit_and_nets_expense():
    p = build(_BASE + 'poisson aggregate net of 2000 xs 3000 rol 8% cede 25%')
    assert p._gcn_econ['c_agg'] == pytest.approx(0.25 * 160)
    m = p.gcn_df.xs('Mean')
    assert m.loc['Expense', 'ceded'] == pytest.approx(40.0)    # commission credit (+)
    # net expense = gross expense - commission = 0 - 40 (a net credit)
    assert p._net_expense() == pytest.approx(-40.0)


def test_any_premium_clause_promotes_to_gcn():
    # no premium clause -> ordinary single-leg (net-only) pnl
    assert build(_BASE + 'poisson aggregate net of 2000 xs 3000')._gcn is None
    # a premium clause -> GCN
    assert build(_BASE + 'poisson aggregate net of 2000 xs 3000 deposit 1500'
                 )._gcn is not None


def test_occurrence_only_gcn_columns():
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 10% poisson')
    assert list(p.gcn_df.columns) == ['gross', 'ceded', 'net', 'impact']
    assert p._gcn_econ['pc_occ'] == pytest.approx(0.10 * 100)
    assert p._gcn_econ['pc_agg'] == 0.0


def test_both_sides_split_and_full_waterfall():
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 5% poisson '
              'aggregate net of 2000 xs 3000 rol 8% cede 20%')
    e = p._gcn_econ
    assert e['pc_occ'] == pytest.approx(0.05 * 100)
    assert e['pc_agg'] == pytest.approx(0.08 * 2000)
    assert e['c_agg'] == pytest.approx(0.20 * 160)
    assert list(p.gcn_df.columns) == [
        'gross', 'ceded occ', 'net occ', 'occ impact',
        'ceded agg', 'net agg', 'agg impact', 'impact']


def test_gcn_means_add_across_waterfall():
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 5% poisson '
              'aggregate net of 2000 xs 3000 rol 8% cede 20%')
    m = p.gcn_df.xs('Mean')
    for r in ['Premium', 'Loss', 'Expense', 'UW']:
        assert m.loc[r, 'gross'] + m.loc[r, 'ceded occ'] == pytest.approx(
            m.loc[r, 'net occ'], abs=1e-2)
        assert m.loc[r, 'net occ'] + m.loc[r, 'ceded agg'] == pytest.approx(
            m.loc[r, 'net agg'], abs=1e-2)
    # and rows add down to UW within each retained column
    for col in ['gross', 'net occ', 'net agg']:
        s = m[col]
        assert s['UW'] == pytest.approx(
            s['Premium'] + s['Loss'] + s['Expense'], abs=1e-2)


def test_cede_without_premium_errors():
    with pytest.raises(ValueError, match='cede'):
        build(_BASE + 'poisson aggregate net of 2000 xs 3000 cede 25%')


def test_rol_without_finite_limit_errors():
    with pytest.raises(ValueError, match='finite limit'):
        build(_BASE + 'poisson aggregate ceded to inf xs 3000 rol 8%')


def test_premium_clause_on_plain_agg_errors():
    with pytest.raises(ValueError, match='pnl'):
        build('agg Z 100 claims sev lognorm 50 cv 1.5 poisson '
              'aggregate net of 2000 xs 3000 rol 8%')
