"""Ceded premium (``deposit`` / ``rol`` / ``rate``) and ceding commission
(``cede``) on a ``pnl`` / ``xpnl``.

``pnl`` is the **consolidated** net view ([Decision-PnL-Is-Consolidated],
``dev/plan-pnl-consolidated-xpnl-walk.md``): always one group -- consideration
= net premium (gross - ceded premiums), obligation = net loss + a commission
leg for the ceding commissions received ([Commission-Obligation-Side]) + own
expenses. The per-step split is the ``xpnl`` **walk** (gross -> each
cover -> Total), whose cession groups book contra (``-premium / +recovery /
+commission``). The premium resolves to currency (``deposit`` an amount,
``rol`` = share x rol x limit, ``rate`` = rate x the side's subject premium:
the stated gross on the occurrence side, gross - pc_occ on the aggregate side
[Agg-Rate-Nets-Inuring-Occ]); the commission is ``cede x ceded_premium`` per
layer. The resolved economics ride
on :attr:`PnL.economics` (keys ``pc_occ`` / ``pc_agg`` = occ / agg ceded
premium, ``c_occ`` / ``c_agg`` = commissions, ``gross`` / ``ceded`` =
totals).
"""
import pytest

from aggregate import build, PnL
from aggregate.constants import IgnoredDecLClauseWarning

TOL = 1e-3
_BASE = 'pnl T 5000 prem less agg T_e 100 claims sev lognorm 50 cv 1.5 '


def _leg(pnl, label):
    """One declared leg's economic_df row, by Line label."""
    return pnl.economic_df.xs(label, level='Label').iloc[0]


def _lines(pnl):
    return list(pnl.economic_df.index.get_level_values('Label'))


def test_deposit_rol_rate_resolution():
    assert build(_BASE + 'poisson aggregate net of 2000 xs 3000 deposit 1500'
                 ).economics['pc_agg'] == pytest.approx(1500.0)
    # rol: share (1.0) x 8% x limit (2000)
    assert build(_BASE + 'poisson aggregate net of 2000 xs 3000 rol 8%'
                 ).economics['pc_agg'] == pytest.approx(0.08 * 2000)
    # rate: 30% of the gross premium (5000)
    assert build(_BASE + 'poisson aggregate net of 2000 xs 3000 rate 30%'
                 ).economics['pc_agg'] == pytest.approx(0.30 * 5000)


def test_premium_scales_by_placement_share():
    # All premium forms are quoted at 100% placement and scaled by the share
    # placed. A 50% placement (``50% po 2000 xs 3000``) halves deposit and rate;
    # rol already carries the share (``share x rol x limit``) so it is
    # unchanged by the explicit factor but still tracks the share.
    dep = build(_BASE + 'poisson aggregate net of 50% po 2000 xs 3000 deposit 1500')
    assert dep.economics['pc_agg'] == pytest.approx(0.5 * 1500)          # 750
    rate = build(_BASE + 'poisson aggregate net of 50% po 2000 xs 3000 rate 30%')
    assert rate.economics['pc_agg'] == pytest.approx(0.5 * 0.30 * 5000)  # 750
    rol = build(_BASE + 'poisson aggregate net of 50% po 2000 xs 3000 rol 8%')
    assert rol.economics['pc_agg'] == pytest.approx(0.5 * 0.08 * 2000)   # 80
    # cede is a fraction of the *placed* premium, so it follows the scaling
    ced = build(_BASE + 'poisson aggregate net of 50% po 2000 xs 3000 '
                'deposit 1500 cede 25%')
    assert ced.economics['c_agg'] == pytest.approx(0.25 * 0.5 * 1500)    # 187.5
    # a 100% placement is unchanged (share = 1)
    full = build(_BASE + 'poisson aggregate net of 2000 xs 3000 deposit 1500')
    assert full.economics['pc_agg'] == pytest.approx(1500.0)


def test_deposit_and_rol_coincide_when_equal():
    rol = build(_BASE + 'poisson aggregate net of 2000 xs 3000 rol 8%')
    dep = build(_BASE + 'poisson aggregate net of 2000 xs 3000 deposit 160')
    assert rol.economics['ceded'] == pytest.approx(dep.economics['ceded'])


def test_cede_books_commission_as_obligation_leg():
    # [Commission-Obligation-Side]: the commission received is not premium;
    # it books as an obligation-side commission leg on the consolidated face,
    # displaying positive, matching the walk convention
    p = build(_BASE + 'poisson aggregate net of 2000 xs 3000 rol 8% cede 25%')
    assert p.economics['c_agg'] == pytest.approx(0.25 * 160)
    assert _leg(p, 'net premium')['EX'] == pytest.approx(5000 - 160)
    assert _leg(p, 'commission')['EX'] == pytest.approx(40.0)
    # ...and stays a visible leg on the xpnl walk
    x = build(_BASE.replace('pnl T', 'xpnl TX', 1)
              + 'poisson aggregate net of 2000 xs 3000 rol 8% cede 25%')
    assert _leg(x, 'agg 2000 xs 3000 commission')['EX'] == pytest.approx(40.0)


def test_any_reinsurance_yields_consolidated_pnl():
    # reinsurance presence (not economics presence) drives the face
    # ([XPnL-Zero-Premium-Cessions]): no premium clause -> the SAME
    # consolidated net view at zero ceded premium, with one warning
    from aggregate.constants import ZeroPremiumCessionWarning
    with pytest.warns(ZeroPremiumCessionWarning, match='aggregate cession'):
        plain = build(_BASE + 'poisson aggregate net of 2000 xs 3000')
    assert isinstance(plain, PnL)
    assert plain.economics['pc_agg'] == 0.0
    assert list(plain.economic_df.index.names) == ['Side', 'Label']
    # zero ceded premium: net premium = the full gross premium
    assert _leg(plain, 'net premium')['EX'] == pytest.approx(5000.0)
    assert 'Loss (net)' in _lines(plain)
    # a premium clause -> the same consolidated face, priced
    p = build(_BASE + 'poisson aggregate net of 2000 xs 3000 deposit 1500')
    assert isinstance(p, PnL)
    assert list(p.economic_df.index.names) == ['Side', 'Label']
    assert list(p.summary_df.index) == ['Consideration', 'Obligation',
                                        'Margin']
    # consideration = net premium: 5000 - 1500
    assert _leg(p, 'net premium')['EX'] == pytest.approx(3500.0)
    assert p.economics['pc_agg'] == pytest.approx(1500.0)
    # the two faces differ ONLY by the ceded premium constant
    assert _leg(plain, 'net premium')['EX'] - 1500.0 == pytest.approx(
        _leg(p, 'net premium')['EX'])


def test_zero_premium_cession_walks():
    """The zero-premium cession is walkable: xpnl books the cover at zero
    premium with the real recovery ([XPnL-Zero-Premium-Cessions])."""
    from aggregate.constants import ZeroPremiumCessionWarning
    with pytest.warns(ZeroPremiumCessionWarning):
        x = build(_BASE.replace('pnl T', 'xpnl TX', 1)
                  + 'poisson aggregate net of 2000 xs 3000')
    s = x.economic_df
    steps = list(dict.fromkeys(s.index.get_level_values('Step')))
    assert steps == ['Gross', 'agg 2000 xs 3000', 'All']
    assert _leg(x, 'agg 2000 xs 3000 premium')['EX'] == 0.0
    assert _leg(x, 'agg 2000 xs 3000 recovery')['EX'] > 0


def test_occurrence_only_consolidates_over_net_occ():
    """An occ guaranteed-cost program consolidates over the net-of-occ
    marginal: one net-premium leg, the net loss; the occ risk transfer is
    the ``xpnl`` walk."""
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 10% poisson')
    assert _leg(p, 'net premium')['EX'] == pytest.approx(5000 - 0.10 * 100)
    assert 'Loss (net)' in _lines(p)
    assert p.economics['pc_occ'] == pytest.approx(0.10 * 100)
    assert p.economics['pc_agg'] == 0.0
    assert list(p.economic_df.index.names) == ['Side', 'Label']


def test_both_sides_split_and_walk_ledger():
    # the pnl consolidates; the per-side split shows on the economics and,
    # step by step, on the xpnl walk
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 5% poisson '
              'aggregate net of 2000 xs 3000 rol 8% cede 20%')
    e = p.economics
    assert e['pc_occ'] == pytest.approx(0.05 * 100)
    assert e['pc_agg'] == pytest.approx(0.08 * 2000)
    assert e['c_agg'] == pytest.approx(0.20 * 160)
    assert _leg(p, 'net premium')['EX'] == pytest.approx(5000 - 5 - 160)
    assert _leg(p, 'commission')['EX'] == pytest.approx(32.0)
    x = build(_BASE.replace('pnl T', 'xpnl TX', 1)
              + 'occurrence net of 100 xs 200 rol 5% poisson '
              'aggregate net of 2000 xs 3000 rol 8% cede 20%')
    lines = _lines(x)
    for row in ('Premium', 'Loss', 'occ 100 xs 200 premium', 'occ 100 xs 200 recovery',
                'agg 2000 xs 3000 premium', 'agg 2000 xs 3000 recovery',
                'agg 2000 xs 3000 commission'):
        assert row in lines, row
    steps = list(x.economic_df.index.get_level_values('Step'))
    assert 'Gross' in steps and 'occ 100 xs 200' in steps \
        and 'agg 2000 xs 3000' in steps and 'All' in steps


def test_walk_means_add_down_the_sheet():
    x = build(_BASE.replace('pnl T', 'xpnl TX', 1)
              + 'occurrence net of 100 xs 200 rol 5% poisson '
              'aggregate net of 2000 xs 3000 rol 8% cede 20%')
    s = x.economic_df
    # step results foot to their signed legs (means add by linearity)
    assert s.loc[('Gross', 'Margin', 'Gross'), 'EX'] == pytest.approx(
        _leg(x, 'Premium')['EX'] + _leg(x, 'Loss')['EX'], abs=1e-9)
    assert s.loc[('agg 2000 xs 3000', 'Margin', 'Total'), 'EX'] == pytest.approx(
        _leg(x, 'agg 2000 xs 3000 premium')['EX']
        + _leg(x, 'agg 2000 xs 3000 recovery')['EX']
        + _leg(x, 'agg 2000 xs 3000 commission')['EX'], abs=1e-9)
    # the grand result sums the step results exactly
    assert s.loc[('All', 'Margin', 'Net'), 'EX'] == pytest.approx(
        s.loc[('Gross', 'Margin', 'Gross'), 'EX']
        + s.loc[('occ 100 xs 200', 'Margin', 'Total'), 'EX']
        + s.loc[('agg 2000 xs 3000', 'Margin', 'Total'), 'EX'], abs=1e-9)
    # running nets read the engine's own net marginals
    assert s.loc[('agg 2000 xs 3000', 'Margin', 'Net'), 'EX'] == pytest.approx(
        s.loc[('All', 'Margin', 'Net'), 'EX'], abs=1e-9)
    # the consolidated pnl's margin equals the walk's grand result -- to
    # joint-grid accuracy: the walk rides the occurrence (gross, ceded)
    # joint (budget-sized common bs), the consolidated margin reads the
    # engine's exact net marginal
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 5% poisson '
              'aggregate net of 2000 xs 3000 rol 8% cede 20%')
    assert p.est_m == pytest.approx(
        s.loc[('All', 'Margin', 'Net'), 'EX'], rel=5e-3)


def test_faces_agree_on_consideration_and_obligation_totals():
    """[Commission-Obligation-Side] cross-face invariant: the consolidated
    summary rows match the walk's All row, not just the margin. The
    consideration totals are constants and agree exactly; the obligation
    totals and margins agree to joint-grid accuracy when an occurrence
    program makes the walk ride the (gross, ceded) joint, and exactly when
    only an aggregate cover exists (the walk rides the exact gross
    marginal)."""
    decl = ('poisson aggregate net of 2000 xs 3000 rol 8% cede 20%')
    p = build(_BASE + decl)
    x = build(_BASE.replace('pnl T', 'xpnl TX', 1) + decl)
    ps, xs = p.summary_df, x.economic_df
    assert ps.loc['Consideration', 'EX'] == pytest.approx(
        xs.loc[('All', 'Consideration', 'Net'), 'EX'], abs=1e-9)
    assert ps.loc['Obligation', 'EX'] == pytest.approx(
        xs.loc[('All', 'Obligation', 'Net'), 'EX'], abs=1e-9)
    assert ps.loc['Margin', 'EX'] == pytest.approx(
        xs.loc[('All', 'Margin', 'Net'), 'EX'], abs=1e-9)
    both = ('occurrence net of 100 xs 200 rol 5% cede 10% poisson '
            'aggregate net of 2000 xs 3000 rol 8% cede 20%')
    p2 = build(_BASE + both)
    x2 = build(_BASE.replace('pnl T', 'xpnl TX', 1) + both)
    p2s, x2s = p2.summary_df, x2.economic_df
    assert p2s.loc['Consideration', 'EX'] == pytest.approx(
        x2s.loc[('All', 'Consideration', 'Net'), 'EX'], abs=1e-9)
    assert p2s.loc['Obligation', 'EX'] == pytest.approx(
        x2s.loc[('All', 'Obligation', 'Net'), 'EX'], rel=5e-3)
    assert p2s.loc['Margin', 'EX'] == pytest.approx(
        x2s.loc[('All', 'Margin', 'Net'), 'EX'], rel=5e-3)


def test_agg_rate_nets_inuring_occ_premium():
    # [Agg-Rate-Nets-Inuring-Occ]: the aggregate-side ``rate`` base is the
    # subject premium net of the inuring occurrence cession, not the stated
    # gross; the occurrence commission does not add back
    p = build(_BASE + 'occurrence net of 100 xs 200 rol 10% cede 20% poisson '
              'aggregate net of 50% po 2000 xs 3000 rate 30% cede 25%')
    e = p.economics
    assert e['pc_occ'] == pytest.approx(0.10 * 100)                     # 10
    assert e['pc_agg'] == pytest.approx(0.5 * 0.30 * (5000 - 10))       # 748.5
    assert e['c_agg'] == pytest.approx(0.25 * 0.5 * 0.30 * (5000 - 10))
    # the occurrence side rates off the stated gross (occ is first in the
    # tower), regardless of its own cession
    occ = build(_BASE + 'occurrence net of 50% po 100 xs 200 rate 2% poisson')
    assert occ.economics['pc_occ'] == pytest.approx(0.5 * 0.02 * 5000)


def test_agg_deposit_and_rol_ignore_inuring_occ_premium():
    # companion to [Agg-Rate-Nets-Inuring-Occ]: ``deposit`` and ``rol``
    # aggregate clauses are absolute quotes, unchanged by an occ program
    dep = build(_BASE + 'occurrence net of 100 xs 200 rol 10% poisson '
                'aggregate net of 2000 xs 3000 deposit 1500')
    assert dep.economics['pc_agg'] == pytest.approx(1500.0)
    rol = build(_BASE + 'occurrence net of 100 xs 200 rol 10% poisson '
                'aggregate net of 2000 xs 3000 rol 8%')
    assert rol.economics['pc_agg'] == pytest.approx(0.08 * 2000)


def test_cede_without_premium_errors():
    with pytest.raises(ValueError, match='cede'):
        build(_BASE + 'poisson aggregate net of 2000 xs 3000 cede 25%')


def test_rol_without_finite_limit_errors():
    with pytest.raises(ValueError, match='finite limit'):
        build(_BASE + 'poisson aggregate ceded to inf xs 3000 rol 8%')


def test_premium_clause_on_plain_agg_warns_and_builds():
    # [Reins-Economics-On-Agg-Ignore-Warn]: a pure aggregate ignores what it
    # cannot use and says so -- the economics need a pnl / xpnl to activate.
    with pytest.warns(IgnoredDecLClauseWarning, match='pnl'):
        a = build('agg Z 100 claims sev lognorm 50 cv 1.5 poisson '
                  'aggregate net of 2000 xs 3000 rol 8%')
    assert type(a).__name__ == 'Aggregate'
