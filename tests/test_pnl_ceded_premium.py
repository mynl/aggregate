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
    assert p.summary_df.loc['ceded agg commission', 'EX'] == pytest.approx(40.0)


def test_any_premium_clause_promotes_to_ledger():
    # no premium clause -> ordinary single-group (net-only) plain pnl
    plain = build(_BASE + 'poisson aggregate net of 2000 xs 3000')
    assert isinstance(plain, PnL)
    assert not hasattr(plain, 'economics')
    assert 'ceded agg result' not in plain.summary_df.index
    # a premium clause -> the two-group ledger with a real cession group
    p = build(_BASE + 'poisson aggregate net of 2000 xs 3000 deposit 1500')
    assert isinstance(p, PnL)
    s = p.summary_df
    for row in ('ceded agg premium', 'ceded agg recovery', 'ceded agg result',
                'net through ceded agg', 'margin', 'total impact'):
        assert row in s.index
    # the cession books contra and its result is the step delta
    assert s.loc['ceded agg premium', 'EX'] == pytest.approx(-1500.0)
    assert s.loc['ceded agg result', 'EX'] == pytest.approx(
        s.loc['total impact', 'EX'], abs=1e-9)


def test_occurrence_only_books_constants_over_net_occ():
    """An occ guaranteed-cost program books over the net-of-occ marginal:
    the occ ceded premium is a constant leg (the occ risk transfer is the
    ``xpnl`` exhibit)."""
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 10% poisson')
    s = p.summary_df
    assert s.loc['ceded occ premium', 'EX'] == pytest.approx(-(0.10 * 100))
    assert 'loss (net occ)' in s.index
    assert p.economics['pc_occ'] == pytest.approx(0.10 * 100)
    assert p.economics['pc_agg'] == 0.0
    # single-group ledger (the occ cession is not measurable per atom here)
    assert 'total impact' not in s.index


def test_both_sides_split_and_full_ledger():
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 5% poisson '
              'aggregate net of 2000 xs 3000 rol 8% cede 20%')
    e = p.economics
    assert e['pc_occ'] == pytest.approx(0.05 * 100)
    assert e['pc_agg'] == pytest.approx(0.08 * 2000)
    assert e['c_agg'] == pytest.approx(0.20 * 160)
    s = p.summary_df
    # net-of-occ sell group + real agg cession buy group
    for row in ('premium', 'loss (net occ)', 'ceded occ premium',
                'net occ result', 'ceded agg premium', 'ceded agg recovery',
                'ceded agg commission', 'ceded agg result',
                'net through ceded agg', 'margin'):
        assert row in s.index, row


def test_ledger_means_add_down_the_sheet():
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 5% poisson '
              'aggregate net of 2000 xs 3000 rol 8% cede 20%')
    s = p.summary_df
    # group results are per-atom sums of their signed legs
    assert s.loc['net occ result', 'EX'] == pytest.approx(
        s.loc['premium', 'EX'] + s.loc['loss (net occ)', 'EX']
        + s.loc['ceded occ premium', 'EX'], abs=1e-2)
    assert s.loc['ceded agg result', 'EX'] == pytest.approx(
        s.loc['ceded agg premium', 'EX'] + s.loc['ceded agg recovery', 'EX']
        + s.loc['ceded agg commission', 'EX'], abs=1e-2)
    # the grand result sums the group results
    assert s.loc['margin', 'EX'] == pytest.approx(
        s.loc['net occ result', 'EX'] + s.loc['ceded agg result', 'EX'],
        abs=1e-2)


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
