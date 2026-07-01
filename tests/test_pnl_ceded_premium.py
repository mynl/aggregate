"""Ceded premium (``deposit`` / ``rol`` / ``rate``) and ceding commission
(``cede``) on a ``pnl`` (Phase 1, decision 3).

A premium clause on any reinsurance layer promotes the ``pnl`` to a
:class:`PnLTower` (the Gross/Ceded/Net exhibit). The premium resolves to currency
(``deposit`` an amount, ``rol`` = share x rol x limit, ``rate`` = rate x gross
premium); the commission ``cede x ceded_premium`` books as an expense credit.
The resolved economics are exposed on :attr:`PnLTower.economics` (keys
``pc_occ`` / ``pc_agg`` = occ / agg ceded premium, ``c_occ`` / ``c_agg`` =
commissions, ``gross`` / ``ceded`` = totals). See
``dev/done/plan-pnl-expenses-ceded-premium.md``.
"""
import pytest

from aggregate import build, PnL, PnLTower

TOL = 1e-3
_BASE = 'pnl T 5000 prem less 100 claims sev lognorm 50 cv 1.5 '


def test_deposit_rol_rate_resolution():
    assert build(_BASE + 'poisson aggregate net of 2000 xs 3000 deposit 1500'
                 ).economics['pc_agg'] == pytest.approx(1500.0)
    # rol: share (1.0) x 8% x limit (2000)
    assert build(_BASE + 'poisson aggregate net of 2000 xs 3000 rol 8%'
                 ).economics['pc_agg'] == pytest.approx(0.08 * 2000)
    # rate: 30% of the gross premium (5000)
    assert build(_BASE + 'poisson aggregate net of 2000 xs 3000 rate 30%'
                 ).economics['pc_agg'] == pytest.approx(0.30 * 5000)


def test_deposit_and_rol_coincide_when_equal():
    rol = build(_BASE + 'poisson aggregate net of 2000 xs 3000 rol 8%')
    dep = build(_BASE + 'poisson aggregate net of 2000 xs 3000 deposit 160')
    assert rol.economics['ceded'] == pytest.approx(dep.economics['ceded'])


def test_cede_books_commission_credit_and_nets_expense():
    p = build(_BASE + 'poisson aggregate net of 2000 xs 3000 rol 8% cede 25%')
    assert p.economics['c_agg'] == pytest.approx(0.25 * 160)     # commission credit
    # net expense = gross expense - commission = 0 - 40 (a net credit) shows on the
    # net perspective's expense leg
    assert p.summary_df.loc['expense', 'EX'] == pytest.approx(-40.0)


def test_any_premium_clause_promotes_to_gcn():
    # no premium clause -> ordinary single-leg (net-only) plain pnl
    assert isinstance(build(_BASE + 'poisson aggregate net of 2000 xs 3000'), PnL)
    # a premium clause -> a Gross/Ceded/Net tower
    assert isinstance(
        build(_BASE + 'poisson aggregate net of 2000 xs 3000 deposit 1500'),
        PnLTower)


def test_occurrence_only_gcn_columns():
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 10% poisson')
    assert list(p.gcn_df.columns) == ['gross', 'ceded', 'net', 'impact']
    assert p.economics['pc_occ'] == pytest.approx(0.10 * 100)
    assert p.economics['pc_agg'] == 0.0


def test_both_sides_split_and_full_waterfall():
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 5% poisson '
              'aggregate net of 2000 xs 3000 rol 8% cede 20%')
    e = p.economics
    assert e['pc_occ'] == pytest.approx(0.05 * 100)
    assert e['pc_agg'] == pytest.approx(0.08 * 2000)
    assert e['c_agg'] == pytest.approx(0.20 * 160)
    assert list(p.gcn_df.columns) == [
        'gross', 'ceded occ', 'net occ', 'ceded agg', 'net agg',
        'occ impact', 'agg impact', 'impact']


def test_gcn_means_add_across_waterfall():
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 5% poisson '
              'aggregate net of 2000 xs 3000 rol 8% cede 20%')
    g = p.gcn_df
    # means add across the split on the EX row: gross + ceded occ = net occ, then
    # net occ + ceded agg = net agg (the covariance carried per atom)
    assert g.loc['EX', 'gross'] + g.loc['EX', 'ceded occ'] == pytest.approx(
        g.loc['EX', 'net occ'], abs=1e-2)
    assert g.loc['EX', 'net occ'] + g.loc['EX', 'ceded agg'] == pytest.approx(
        g.loc['EX', 'net agg'], abs=1e-2)
    # the impact columns are the running deltas off the retained legs
    assert g.loc['EX', 'occ impact'] == pytest.approx(
        g.loc['EX', 'net occ'] - g.loc['EX', 'gross'], abs=1e-2)
    assert g.loc['EX', 'agg impact'] == pytest.approx(
        g.loc['EX', 'net agg'] - g.loc['EX', 'net occ'], abs=1e-2)


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
