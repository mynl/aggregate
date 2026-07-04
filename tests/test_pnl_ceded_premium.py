"""Ceded premium (``deposit`` / ``rol`` / ``rate``) and ceding commission
(``cede``) on a ``pnl`` (signed group-ledger form).

A premium clause on any reinsurance layer promotes the ``pnl`` to the
Gross/Ceded/Net **group ledger** (``dev/plan-yapnl.md``): an aggregate cession
is a real ``buy`` group over the gross marginal (its rows book contra:
``-premium / +recovery / +commission``); an occurrence guaranteed-cost program
books over the net-of-occ marginal with the occ ceded premium as a constant
leg (the occ risk transfer is the ``xpnl`` exhibit). The premium resolves to
currency (``deposit`` an amount, ``rol`` = share x rol x limit, ``rate`` =
rate x gross premium); the commission ``cede x ceded_premium`` books as a
received leg on the cession. The resolved economics ride on
:attr:`PnL.economics` (keys ``pc_occ`` / ``pc_agg`` = occ / agg ceded premium,
``c_occ`` / ``c_agg`` = commissions, ``gross`` / ``ceded`` = totals).
"""
import pytest

from aggregate import build, PnL

TOL = 1e-3
_BASE = 'pnl T 5000 prem less agg T_e 100 claims sev lognorm 50 cv 1.5 '


def _leg(pnl, label):
    """One declared leg's stats_df row, by Line label."""
    return pnl.stats_df.xs(label, level='Line').iloc[0]


def _lines(pnl):
    return list(pnl.stats_df.index.get_level_values('Line'))


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


def test_cede_books_commission_leg_on_the_cession():
    p = build(_BASE + 'poisson aggregate net of 2000 xs 3000 rol 8% cede 25%')
    assert p.economics['c_agg'] == pytest.approx(0.25 * 160)
    # the commission is a received leg on the buy group: +40 in the ledger
    assert _leg(p, 'ceded agg commission')['EX'] == pytest.approx(40.0)


def test_any_premium_clause_promotes_to_ledger():
    # no premium clause -> ordinary single-group (net-only) plain pnl
    plain = build(_BASE + 'poisson aggregate net of 2000 xs 3000')
    assert isinstance(plain, PnL)
    assert not hasattr(plain, 'economics')
    assert list(plain.stats_df.index.names) == ['View', 'Line']
    # a premium clause -> the two-group ledger with a real cession group
    p = build(_BASE + 'poisson aggregate net of 2000 xs 3000 deposit 1500')
    assert isinstance(p, PnL)
    s = p.stats_df
    for row in (('ceded agg', 'Consideration', 'ceded agg premium'),
                ('ceded agg', 'Obligation', 'ceded agg recovery'),
                ('ceded agg', 'Margin', 'Total'),
                ('ceded agg', 'Margin', 'Net'),
                ('Total', 'Margin', 'Total'),
                ('Total', 'Margin', 'Impact')):
        assert row in s.index, row
    # the cession books contra and its result is the step delta
    assert s.loc[('ceded agg', 'Consideration', 'ceded agg premium'), 'EX'] \
        == pytest.approx(-1500.0)
    assert s.loc[('ceded agg', 'Margin', 'Total'), 'EX'] == pytest.approx(
        s.loc[('Total', 'Margin', 'Impact'), 'EX'], abs=1e-9)


def test_occurrence_only_books_constants_over_net_occ():
    """An occ guaranteed-cost program books over the net-of-occ marginal:
    the occ ceded premium is a constant leg (the occ risk transfer is the
    ``xpnl`` exhibit)."""
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 10% poisson')
    assert _leg(p, 'ceded occ premium')['EX'] == pytest.approx(-(0.10 * 100))
    assert 'loss (net occ)' in _lines(p)
    assert p.economics['pc_occ'] == pytest.approx(0.10 * 100)
    assert p.economics['pc_agg'] == 0.0
    # single-group ledger (the occ cession is not measurable per atom here)
    assert list(p.stats_df.index.names) == ['View', 'Line']


def test_both_sides_split_and_full_ledger():
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 5% poisson '
              'aggregate net of 2000 xs 3000 rol 8% cede 20%')
    e = p.economics
    assert e['pc_occ'] == pytest.approx(0.05 * 100)
    assert e['pc_agg'] == pytest.approx(0.08 * 2000)
    assert e['c_agg'] == pytest.approx(0.20 * 160)
    # net-of-occ sell group + real agg cession buy group
    lines = _lines(p)
    for row in ('premium', 'loss (net occ)', 'ceded occ premium',
                'ceded agg premium', 'ceded agg recovery',
                'ceded agg commission'):
        assert row in lines, row
    steps = list(p.stats_df.index.get_level_values('Step'))
    assert 'net occ' in steps and 'ceded agg' in steps and 'Total' in steps


def test_ledger_means_add_down_the_sheet():
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 5% poisson '
              'aggregate net of 2000 xs 3000 rol 8% cede 20%')
    s = p.stats_df
    # group results are per-atom sums of their signed legs
    assert s.loc[('net occ', 'Margin', 'Total'), 'EX'] == pytest.approx(
        _leg(p, 'premium')['EX'] + _leg(p, 'loss (net occ)')['EX']
        + _leg(p, 'ceded occ premium')['EX'], abs=1e-2)
    assert s.loc[('ceded agg', 'Margin', 'Total'), 'EX'] == pytest.approx(
        _leg(p, 'ceded agg premium')['EX']
        + _leg(p, 'ceded agg recovery')['EX']
        + _leg(p, 'ceded agg commission')['EX'], abs=1e-2)
    # the grand result sums the group results
    assert s.loc[('Total', 'Margin', 'Total'), 'EX'] == pytest.approx(
        s.loc[('net occ', 'Margin', 'Total'), 'EX']
        + s.loc[('ceded agg', 'Margin', 'Total'), 'EX'], abs=1e-2)


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
