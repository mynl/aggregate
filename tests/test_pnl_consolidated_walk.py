"""[PnL-Consolidated-XPnL-Walk] acceptance and per-atom walk correctness.

Pins the plan's motivating example (the ``Cat`` occ-tower program of
2026-07-04): the consolidated ``pnl`` card reads net premium against net loss
-- never "gross premium next to higher loss" -- and the ``xpnl`` walk steps
gross -> Occ Cover -> Agg Cover -> All as a per-atom tower over the
occurrence (gross, ceded) joint (a141: the marginal stitch retired in favor
of a footing scenario-κ ladder; walk means are joint-grid accurate). Plus
the [Construction-Introspection] smoke tests. See
``dev/plan-pnl-consolidated-xpnl-walk.md`` and ``dev/PLAN-A.md``.
"""

import numpy as np
import pytest

from aggregate import PnL, build

# the acceptance Cat program: 12000 premium, occ + agg covers with deposits,
# three expense groups
_ENGINE = ('agg Cat_e as "Gross Book1" 12000 prem at 85% lr sev lognorm 50 cv 3 '
           'occurrence net of 75% po 2750 xs 250 deposit 2000 as "Occ Cover" '
           'poisson '
           'aggregate net of 80% po 1000 xs 7000 deposit 1000 as "Agg Cover"')
_TAIL = ' less 500 fixed expense 5% premium expense 2% loss expense'


@pytest.fixture(scope='module')
def cat():
    p = build(f'pnl Cat 12000 premium less {_ENGINE}{_TAIL}')
    x = build(f'xpnl CatX 12000 premium less {_ENGINE}{_TAIL}')
    a = build(_ENGINE.replace(' deposit 2000 as "Occ Cover"', '')
              .replace(' deposit 1000 as "Agg Cover"', ''))
    return p, x, a


# ----------------------------------------------------------------------
# the consolidated pnl: position, not walk
# ----------------------------------------------------------------------
def test_acceptance_pnl_consolidated_card(cat):
    p, _x, _a = cat
    df = p.summary_df
    # always the flat three-row card, whatever the program
    assert list(df.index) == ['Consideration', 'Obligation', 'Margin']
    # Consideration = net premium 12000 - 2000 - 1000; no gross premium
    # booked against a higher-than-net loss
    assert df.loc['Consideration', 'EX'] == pytest.approx(9000.0)
    # the card foots on EX
    assert df.loc['Margin', 'EX'] == pytest.approx(
        df.loc['Consideration', 'EX'] + df.loc['Obligation', 'EX'], abs=1e-9)
    # single source -> the stats ladder is the scenario (κ) pass
    assert 'κ01' in p.stats_df.columns


def test_acceptance_pnl_books_net_loss(cat):
    p, _x, a = cat
    rd = a.reins_density_df
    xs = rd['loss'].to_numpy()
    pn = rd['p_agg_net'].to_numpy()
    e_net = float((xs * pn).sum() / pn.sum())
    s = p.stats_df
    # the engine's declared label names the loss leg, '(net)'-qualified
    # ([Flag-Net-Premium-Leg-Label] precedent)
    assert s.loc[('Obligation', 'Gross Book1 (net)'), 'EX'] == \
        pytest.approx(-e_net, rel=1e-9)


# ----------------------------------------------------------------------
# the xpnl walk: steps, engine marginals, footing, flags
# ----------------------------------------------------------------------
def test_acceptance_walk_steps(cat):
    _p, x, _a = cat
    assert isinstance(x, PnL)
    # base step = the engine's declared label; cover steps = the reins
    # ``as`` labels; the grand step key is 'All' (a140 rename)
    steps = list(dict.fromkeys(
        x.stats_df.index.get_level_values('Step')))
    assert steps == ['Gross Book1', 'Occ Cover', 'Agg Cover', 'All']


def test_walk_rows_read_engine_marginals(cat):
    _p, x, a = cat
    rd = a.reins_density_df
    xs = rd['loss'].to_numpy()

    def marg_mean(col):
        pv = rd[col].to_numpy()
        return float((xs * pv).sum() / pv.sum())

    s = x.stats_df
    # the walk rides the occurrence (gross, ceded) joint, so its rows agree
    # with the engine's exact 1-D marginals to JOINT-GRID accuracy (the
    # joint runs on a budget-sized common bucket size): tight for linear
    # rows, ~1e-2 for the kinked agg-tier map
    assert s.loc[('Occ Cover', 'Obligation', 'Occ Cover recovery'), 'EX'] \
        == pytest.approx(marg_mean('p_agg_ceded_occ'), rel=1e-3)
    assert s.loc[('Agg Cover', 'Obligation', 'Agg Cover recovery'), 'EX'] \
        == pytest.approx(marg_mean('p_agg_ceded'), rel=2e-2)
    def marg_sd(col):
        pv = rd[col].to_numpy()
        pv = pv / pv.sum()
        m = float((xs * pv).sum())
        return float(np.sqrt((xs * xs * pv).sum() - m * m))

    assert s.loc[('Occ Cover', 'Obligation', 'Occ Cover recovery'), 'SD'] \
        == pytest.approx(marg_sd('p_agg_ceded_occ'), rel=1e-2)


def test_walk_running_nets_and_footing(cat):
    _p, x, a = cat
    s = x.stats_df
    ex = s['EX']
    # the EX column foots exactly (means add by linearity)
    legs = [i for i in s.index if i[2] not in ('Total', 'Net', 'Impact')]
    assert ex.loc[legs].sum() == pytest.approx(
        ex.loc[('All', 'Margin', 'Total')], abs=1e-9)
    # running nets are cumulative step results
    assert ex.loc[('Occ Cover', 'Margin', 'Net')] == pytest.approx(
        ex.loc[('Gross Book1', 'Margin', 'Total')]
        + ex.loc[('Occ Cover', 'Margin', 'Total')], abs=1e-9)
    assert ex.loc[('Agg Cover', 'Margin', 'Net')] == pytest.approx(
        ex.loc[('All', 'Margin', 'Total')], abs=1e-9)
    # ...and every κ column foots too (per-atom conditional means)
    k = [c for c in s.columns if c.startswith('κ')][0]
    assert s[k].loc[legs].sum() == pytest.approx(
        s.loc[('All', 'Margin', 'Total'), k], abs=1e-6)


def test_walk_ladder_scenario_and_flagged(cat):
    _p, x, _a = cat
    s = x.stats_df
    # one shared joint -> the scenario (κ) ladder
    # ([Decision-Kappa-Shared-Source-Rule])
    assert 'κ01' in s.columns and 'P01' not in s.columns
    # ...and flagged in the narrative
    assert 'κ' in x.construction_description


def test_walk_impact_row_is_per_atom_delta(cat):
    _p, x, _a = cat
    s = x.stats_df
    # impact = grand result - gross step result: a TRUE per-atom difference
    # (its SD / percentiles are of the difference distribution, not deltas
    # of statistics -- richer than the retired stitched per-stat delta)
    assert s.loc[('All', 'Margin', 'Impact'), 'EX'] == pytest.approx(
        s.loc[('All', 'Margin', 'Total'), 'EX']
        - s.loc[('Gross Book1', 'Margin', 'Total'), 'EX'], abs=1e-9)
    assert s.loc[('All', 'Margin', 'Impact'), 'SD'] >= 0



def test_walk_shares_atoms():
    # the per-atom walk restored shared atoms (the stitched tower had none):
    # a per-atom probability vector exists and the ledger is not stitched
    a = build('agg WR 100 claims sev lognorm 50 cv 1.5 poisson '
              'aggregate net of 2000 xs 3000')
    from aggregate._pnl_builders import build_xpnl_walk
    x = build_xpnl_walk(a, gross=5500, ceded=1800)
    assert not x._stitched
    assert x._probs is not None and x._probs.sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------
# [Construction-Introspection] smoke
# ----------------------------------------------------------------------
def test_construction_narratives_present(cat):
    p, x, _a = cat
    assert 'Consolidated pnl' in p.construction_description
    assert 'net premium 9000' in p.construction_description
    assert 'xpnl walk' in x.construction_description
    # the explanation names the route, the source columns, the ladder rule,
    # and closes with the executable replay
    for text, needles in ((p.construction_explanation,
                           ('p_agg_net', 'scenario', 'Replay', 'PnL(name=')),
                          (x.construction_explanation,
                           ('occ_bivariate', 'foot', 'Replay',
                            'PnL(name='))):
        for needle in needles:
            assert needle in text, needle


def test_construction_generic_fallback():
    # a hand-built kernel P&L is never without a narrative
    from aggregate import Leg
    p = PnL(name='hand', source=(np.array([0.0, 10.0]),
                                 np.array([0.5, 0.5])),
            role='sell', consideration=6.0,
            obligation=[Leg('loss', lambda v: v)])
    assert 'Hand-built' in p.construction_description
    assert 'PnL(name=' in p.construction_explanation


def test_construction_narrative_on_plain_and_var():
    p = build('pnl B 1000 premium less agg B_e 850 loss '
              'sev lognorm 100 cv 1 poisson')
    assert 'Plain pnl' in p.construction_description
    v = build('pnl V 10000 premium less agg V_e 10000 prem at 85% lr '
              'sev lognorm 50 cv 3 poisson aggregate net of 5000 xs 4000 '
              'swing basic 500 lcm 0.5')
    assert 'SwingTerms' in v.construction_description
    assert 'consolidated' in v.construction_description
