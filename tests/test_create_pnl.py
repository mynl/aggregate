"""The domain-agnostic ``create_pnl`` API (``dev/plan-pnl-api.md`` Part B).

A P&L is *money in minus money out* over a random state: group each component
map's values over the source atoms by output value, sum probability, and you get
three exact :class:`GridDistribution` legs (consideration, obligation, result).
These tests exercise the cross-domain core -- a raw ``(values, probs)`` slot, a
:class:`GridDistribution` slot, and a coupled bivariate joint -- the genuinely
new capability that is *not* insurance-shaped. The insurance presets
(``gcn_df`` / reinstatement / variable rating) are covered by their own suites.
"""
import warnings

import numpy as np
import pytest

from aggregate import create_pnl, create_pnl_tower
from aggregate._grid_distribution import GridDistribution
from aggregate.bivariate import BivariateDistribution

TOL = 1e-12

# a tiny discrete loss for the 1-D cases
_VALS = np.array([0.0, 10.0, 20.0, 30.0])
_PROBS = np.array([0.4, 0.3, 0.2, 0.1])


def test_raw_pair_and_gd_slots_agree():
    """The same ``(values, probs)`` via a raw pair and a GD slot coincide."""
    a = create_pnl((_VALS, _PROBS), consideration=15.0, obligation=lambda x: x)
    gd = GridDistribution(_VALS, _PROBS)
    b = create_pnl(gd, consideration=15.0, obligation=lambda x: x)
    np.testing.assert_allclose(a.summary_df.to_numpy(), b.summary_df.to_numpy())


def test_exact_moments_no_rebucketing():
    """A leg is an exact group-by: moments equal the closed-form values."""
    p = create_pnl((_VALS, _PROBS), consideration=15.0, obligation=lambda x: x)
    # E[loss] = 0*.4 + 10*.3 + 20*.2 + 30*.1 = 10; Var = E[X^2]-100 = 200-100 = 100
    s = p.summary_df
    assert s.loc['obligation', 'EX'] == pytest.approx(10.0, abs=TOL)
    assert s.loc['obligation', 'SD'] == pytest.approx(10.0, abs=1e-9)
    # result = 15 - loss: mean 5, SD unchanged (a constant shift), and the
    # percentile ladder is exact (P99 of result = 15 - q_low(0.01) of loss)
    assert s.loc['result', 'EX'] == pytest.approx(5.0, abs=TOL)
    assert s.loc['result', 'SD'] == pytest.approx(10.0, abs=1e-9)


def test_buy_role_flips_result_sign():
    """``role='buy'`` negates the result vs ``'sell'`` (same magnitudes)."""
    sell = create_pnl((_VALS, _PROBS), consideration=4.0,
                      obligation=lambda x: np.minimum(x, 20) * 0.5, role='sell')
    buy = create_pnl((_VALS, _PROBS), consideration=4.0,
                     obligation=lambda x: np.minimum(x, 20) * 0.5, role='buy')
    assert sell.mean == pytest.approx(-buy.mean, abs=TOL)


def test_multicomponent_totals_rows():
    """A multi-part obligation surfaces a 'Total obligation' row; a single does not."""
    two = create_pnl((_VALS, _PROBS), consideration={'premium': 15.0},
                     obligation={'loss': lambda x: x, 'expense': 3.0})
    assert 'Total obligation' in two.summary_df.index
    assert two.summary_df.loc['Total obligation', 'EX'] == pytest.approx(13.0)
    one = create_pnl((_VALS, _PROBS), consideration=15.0, obligation=lambda x: x)
    assert 'Total obligation' not in one.summary_df.index
    assert 'Total consideration' not in one.summary_df.index


def test_means_add_sds_dont_in_tower():
    """The running net adds means but not SDs -- the per-atom diff carries cov."""
    base = create_pnl((_VALS, _PROBS), consideration=15.0,
                      obligation=lambda x: x, name='base')
    cover = create_pnl((_VALS, _PROBS), consideration=2.0, role='buy',
                       obligation=lambda x: np.maximum(x - 20, 0), name='cover')
    tower = create_pnl_tower([base, cover], delta_names=['cover benefit'])
    g = tower.gcn_df
    # means add across the leg + net columns
    assert g.loc['EX', 'net'] == pytest.approx(
        g.loc['EX', 'base'] + g.loc['EX', 'cover'], abs=1e-9)
    # SDs do not add (cover reduces the tail, so net SD < base SD)
    assert g.loc['SD', 'net'] < g.loc['SD', 'base']
    # the one-step delta equals the benefit (net - base)
    assert g.loc['EX', 'cover benefit'] == pytest.approx(
        g.loc['EX', 'net'] - g.loc['EX', 'base'], abs=1e-9)


def test_stats_df_scale_default_and_stochastic_warns():
    """Default scale is Total consideration; a stochastic scale warns and uses EX."""
    p = create_pnl((_VALS, _PROBS), consideration={'premium': 20.0},
                   obligation={'loss': lambda x: x})
    sdf = p.stats_df()
    # premium is the (deterministic) scale: its '% of Total' EX is exactly 1
    assert sdf.loc['EX', ('premium', '% of Total')] == pytest.approx(1.0)
    # a stochastic consideration scale warns
    q = create_pnl((_VALS, _PROBS),
                   consideration={'variable': lambda x: 20.0 + x},
                   obligation={'loss': lambda x: x})
    with pytest.warns(UserWarning, match='stochastic'):
        q.stats_df(scale='variable')


def test_density_df_is_ordered_dict_of_gds():
    """density_df is {leg: GridDistribution} -- consideration(s), obligation(s), result."""
    p = create_pnl((_VALS, _PROBS), consideration={'premium': 15.0},
                   obligation={'loss': lambda x: x, 'expense': 3.0},
                   result_name='margin')
    dd = p.density_df
    assert list(dd) == ['premium', 'loss', 'expense', 'margin']
    assert all(isinstance(v, GridDistribution) for v in dd.values())
    # each GD round-trips to a Series on its own grid
    assert dd['loss'].to_series().sum() == pytest.approx(1.0)


def _independent_joint(y, py, p, pp):
    """A product joint of two 1-D laws -> a BivariateDistribution."""
    dens = np.outer(py, pp)
    return BivariateDistribution(dens, y, p, bs_ceded=1.0, bs_net=1.0)


def test_crop_revenue_negative_dependence_lowers_risk():
    """B1.1: the natural hedge -- a negatively-coupled (Y, P) joint cuts put risk.

    Crop revenue insurance pays ``max(guarantee - Y*P, 0)``. With yield Y and
    price P negatively dependent (a short crop is dear), revenue ``Y*P`` is less
    volatile than under independence, so the shortfall put has a *thinner* tail.
    A marginal-only view (independence) overstates the risk.
    """
    y = np.array([80.0, 100.0, 120.0])          # yield
    p = np.array([8.0, 10.0, 12.0])             # price
    guarantee = 1000.0
    obl = lambda yy, pp: np.maximum(guarantee - yy * pp, 0.0)

    # negative dependence: high yield <-> low price on the anti-diagonal
    neg = np.zeros((3, 3))
    neg[0, 2] = neg[1, 1] = neg[2, 0] = 1 / 3   # (80,12),(100,10),(120,8)
    coupled = BivariateDistribution(neg, y, p, bs_ceded=1.0, bs_net=1.0)
    indep = _independent_joint(y, np.full(3, 1 / 3), p, np.full(3, 1 / 3))

    pc = create_pnl(coupled, consideration=60.0, obligation=obl, role='sell')
    pi = create_pnl(indep, consideration=60.0, obligation=obl, role='sell')
    # the coupled obligation is less volatile (the hedge) -- marginals overstate
    assert pc.summary_df.loc['obligation', 'SD'] < \
        pi.summary_df.loc['obligation', 'SD']


def test_count_axis_fixed_plus_variable_cost():
    """B3: a bivariate (N, X) with f(N, X) = STARTUP*N + FUEL*X mixes the axes."""
    n = np.array([0.0, 1.0, 2.0])               # event count N
    x = np.array([0.0, 100.0, 200.0])           # amount used X
    dens = np.full((3, 3), 1 / 9)
    biv = BivariateDistribution(dens, n, x, bs_ceded=1.0, bs_net=1.0)
    plant = create_pnl(
        biv, consideration=400.0,
        obligation=lambda nn, xx: 50.0 * nn + 1.5 * xx,
        name='Plant', result_name='margin')
    # E[cost] = 50*E[N] + 1.5*E[X] = 50*1 + 1.5*100 = 200
    assert plant.summary_df.loc['obligation', 'EX'] == pytest.approx(200.0)
    assert plant.mean == pytest.approx(400.0 - 200.0)
