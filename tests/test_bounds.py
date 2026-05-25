"""
Regression tests for :mod:`aggregate.bounds`.

The pin uses the bounded BDD portfolio at ``premium = TVaR_0.5(total)``.
At that premium, brackets with ``p_lo = p_star`` carry weight zero, so the
corresponding cloud columns are exactly the ``TVaR_{p_star}`` distortion.
That's the closed-form anchor for the regression suite.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build, Bounds, Distortion


PROGRAM = """
port BDD
    agg A 1 claim sev 10 * beta 2 3 fixed
    agg B 1 claim sev 15 * beta 4 2 fixed
"""


@pytest.fixture(scope='module')
def bdd_at_tvar50():
    """BDD portfolio calibrated so that premium = TVaR(0.5) of the total."""
    port = build(PROGRAM)
    a = port.q(1)
    prem = float(port.tvar(0.5))
    capital = a - prem
    loss = port.agg_m
    margin = prem - loss
    coc = margin / capital
    port.calibrate_distortions(coc, p=1)
    bd = Bounds(port, premium=prem)
    return port, prem, bd


def test_p_star_matches_calibration(bdd_at_tvar50):
    """At premium = TVaR_0.5, p_star = 0.5."""
    _, _, bd = bdd_at_tvar50
    assert abs(bd.p_star - 0.5) < 1e-6


def test_p_star_column_equals_tvar_distortion(bdd_at_tvar50):
    """Brackets (p_star, p_hi) carry weight zero, so the column equals TVaR_{p_star}."""
    _, _, bd = bdd_at_tvar50
    s = bd.s_grid
    cols_at_p_star = bd.cloud_df.xs(bd.p_star, level='p_lower', axis=1)
    expected = np.minimum(1.0, s / (1.0 - bd.p_star))
    # every (p_star, p_hi) column should be identical and equal to TVaR_p*
    for col in cols_at_p_star.columns:
        np.testing.assert_allclose(
            cols_at_p_star[col].values, expected, atol=1e-12)


def test_min_envelope_is_distortion(bdd_at_tvar50):
    """min_envelope is a coherent Distortion: g(0)=0, g(1)=1, monotone."""
    _, _, bd = bdd_at_tvar50
    assert isinstance(bd.min_envelope, Distortion)
    s = np.linspace(0, 1, 101)
    g = bd.min_envelope.g(s)
    assert abs(g[0]) < 1e-10
    assert abs(g[-1] - 1.0) < 1e-10
    assert np.all(np.diff(g) >= -1e-10)


def test_max_envelope_is_callable(bdd_at_tvar50):
    """max_envelope is a callable interp1d (NOT a Distortion)."""
    _, _, bd = bdd_at_tvar50
    assert not isinstance(bd.max_envelope, Distortion)
    s = np.linspace(0, 1, 11)
    g = bd.max_envelope(s)
    assert g.shape == s.shape
    # bounds: 0 <= g <= 1
    assert g.min() >= -1e-12
    assert g.max() <= 1.0 + 1e-12


def test_min_envelope_hinges_records_active_pair(bdd_at_tvar50):
    """min_envelope_hinges records s, the (p_lo, p_hi) bracket and weight active at each s."""
    _, _, bd = bdd_at_tvar50
    hinges = bd.min_envelope_hinges
    assert hinges.shape == (513, 4)
    assert list(hinges.columns) == ['s', 'p_lo', 'p_hi', 'weight']
    np.testing.assert_array_equal(hinges['s'].values, bd.s_grid)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(hinges), 3, replace=False):
        row = hinges.iloc[i]
        pl, pu, w = row['p_lo'], row['p_hi'], row['weight']
        # recorded weight matches weight_df
        assert np.isclose(w, bd.weight_df.at[(pl, pu), 'weight'], atol=1e-12)
        # the BiTVaR closed-form at s == cloud_df value == envelope minimum
        s = row['s']
        bitvar_g = (1 - w) * min(1, s / (1 - pl)) + w * min(1, s / (1 - pu))
        recorded = bd.cloud_df.iloc[i][(pl, pu)]
        actual_min = bd.cloud_df.iloc[i].min()
        assert np.isclose(recorded, actual_min, atol=1e-12)
        assert np.isclose(recorded, bitvar_g, atol=1e-12)


def test_arbitrary_bracket_matches_closed_form(bdd_at_tvar50):
    """A non-degenerate bracket reproduces the closed-form weighted combination."""
    port, prem, bd = bdd_at_tvar50
    # pick p_lo just below p_star and p_hi just above; both in p_knots
    ps = bd.p_knots
    p_lo = ps[ps < bd.p_star][-2]   # second-to-last below p_star
    p_hi = ps[ps > bd.p_star][2]    # third above p_star
    # closed-form weight and resulting g
    t_lo, t_hi = float(port.tvar(p_lo)), float(port.tvar(p_hi))
    w = (prem - t_lo) / (t_hi - t_lo)
    s = bd.s_grid
    expected = (1 - w) * np.minimum(1, s / (1 - p_lo)) + \
               w * np.minimum(1, s / (1 - p_hi))
    got = bd.cloud_df[(p_lo, p_hi)].values
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_distortion_lookup(bdd_at_tvar50):
    """Bounds.distortion(pl, pu) returns the matching BiTVaR."""
    _, _, bd = bdd_at_tvar50
    pl, pu = bd.weight_df.index[10]   # arbitrary valid pair
    d = bd.distortion(pl, pu)
    assert isinstance(d, Distortion)
    # spot-check: at s=1, g(1)=1 for a BiTVaR
    assert np.isclose(d.g(1.0), 1.0, atol=1e-10)


def test_premium_below_mean_raises():
    port = build(PROGRAM)
    with pytest.raises(ValueError, match='below mean'):
        Bounds(port, premium=10.0)   # mean is 14.0


def test_aggregate_input():
    """Bounds accepts an Aggregate as input."""
    port = build(PROGRAM)
    agg = port.A
    prem = float(agg.tvar(0.5))
    bd = Bounds(agg, premium=prem)
    assert abs(bd.p_star - 0.5) < 1e-6
