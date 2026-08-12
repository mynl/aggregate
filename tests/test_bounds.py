"""
Regression tests for :mod:`aggregate.bounds`.

The pin uses the bounded BDD portfolio at ``premium = TVaR_0.5(total)``.
At that premium, brackets with ``p_lo = p_star`` carry weight zero, so the
corresponding cloud columns are exactly the ``TVaR_{p_star}`` distortion.
That's the closed-form anchor for the regression suite.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from aggregate import build, Bounds, Distortion
from aggregate.bounds import JUMP_EPS


PROGRAM = """
port BDD
    agg A 1 claim sev 10 * beta 2 3 fixed
    agg B 1 claim sev 15 * beta 4 2 fixed
"""


def _tvar_g(s, p):
    """The TVaR_p knot function ``min(1, s / (1 - p))``, endpoint included.

    Written out rather than divided, because ``p = 1`` is a real knot in
    ``Bounds.p_knots``. There the TVaR-1 distortion is ``g(s) = 1`` for
    ``s > 0`` and ``g(0) = 0``, matching :meth:`Distortion.tvar_terms`.
    """
    if p >= 1.0:
        return 1.0 if s > 0 else 0.0
    return min(1.0, s / (1.0 - p))


@pytest.fixture(scope='module')
def bdd_at_tvar50():
    """BDD portfolio calibrated so that premium = TVaR(0.5) of the total."""
    port = build(PROGRAM)
    a = port.q(1)
    prem = float(port.tvar(0.5))
    capital = a - prem
    loss = port.actual_m
    margin = prem - loss
    coc = margin / capital
    # a=, not p=1: the Bounds methodology deliberately takes the top of the
    # realized grid as the asset level, and on an unbounded book p=1 is
    # refused because it hides that choice ([Unbounded-Anchor-Guard], a260)
    port.calibrate_distortions(coc, a=a)
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
    # n_s + 1: the uniform grid plus the jump knot ([Bounds-Envelope-Jump])
    assert hinges.shape == (bd.n_s + 1, 4) == (514, 4)
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
        bitvar_g = (1 - w) * _tvar_g(s, pl) + w * _tvar_g(s, pu)
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


def test_s_grid_carries_the_jump_knot(bdd_at_tvar50):
    """``[Bounds-Envelope-Jump]`` the grid samples just right of zero.

    ``p_knots`` includes ``p = 1``, so the cloud contains biTVaRs with an atom
    at zero and the **maximum** envelope is discontinuous at the origin. A bare
    uniform grid holds no point in the first cell, so any consumer joining
    ``(0, 0)`` to the first grid value draws a ramp where there is a cliff.
    """
    _, _, bd = bdd_at_tvar50
    assert len(bd.s_grid) == bd.n_s + 1
    assert bd.s_grid[0] == 0.0
    assert bd.s_grid[1] == JUMP_EPS
    assert np.all(np.diff(bd.s_grid) > 0)          # still sorted and unique


def test_the_maximum_envelope_jumps_to_the_ccoc_mass(bdd_at_tvar50):
    """The jump height is exactly the mass of the calibrated ``ccoc``.

    Both are ``M / (a - L)`` by the pentagon identity, so this is an identity
    and not a tolerance: the ``(p_lo = 0, p_hi = 1)`` bracket **is** the CCoC
    distortion, and it is the admissible distortion that loads the first
    infinitesimal of probability hardest. It is why the maximum envelope has to
    lie above ``ccoc`` everywhere, and why a grid that smears the jump puts the
    drawn upper edge below a curve it must contain.
    """
    port, prem, bd = bdd_at_tvar50
    hi = bd.cloud_df.max(axis=1).to_numpy()
    assert hi[0] == 0.0
    # the anchor the fixture calibrated at, so the two agree about a
    bd_at_a = Bounds(port, premium=prem, a=float(port.q(1)))
    jump = float(bd_at_a.cloud_df.max(axis=1).to_numpy()[1])
    assert abs(jump - float(port.distortions['ccoc'].mass)) < 1e-9


def test_bracket_weights_are_convex(bdd_at_tvar50):
    """A weight outside ``[0, 1]`` is not a convex combination.

    ``p_star`` is a root found to ``xtol = 2**-17`` and is itself spliced into
    ``p_knots``, so the brackets whose ``p_lo`` is ``p_star`` used to come out a
    few thousandths negative and dragged the minimum envelope below zero at the
    jump knot.
    """
    _, _, bd = bdd_at_tvar50
    w = bd.weight_df['weight']
    assert w.min() >= 0.0 and w.max() <= 1.0
    cloud = bd.cloud_df.to_numpy()
    assert cloud.min() >= 0.0 and cloud.max() <= 1.0


def test_the_two_envelopes_do_not_cross(bdd_at_tvar50):
    """``max_envelope`` above ``min_envelope`` off the grid as well as on it.

    The two are built by different machinery, a linear interpolation against a
    fitted concave distortion, so they can only be compared where both are
    evaluated. Inside the first cell they used to cross, because the
    interpolation ramped up from zero while the fit left the origin steeply.
    """
    _, _, bd = bdd_at_tvar50
    s = np.linspace(0, 1, 20001)
    assert (bd.max_envelope(s) - bd.min_envelope.g(s) >= -1e-9).all()


def test_frames_emit_no_runtime_warning():
    """``weight_df`` / ``cloud_df`` must build without numpy noise.

    ``p_knots`` deliberately includes ``p = 1``, which used to trip the eager
    ``np.where`` branches in the tvar kernel and in ``Distortion.tvar_terms``.
    The reported NaN was always discarded, so this pins both the silence and
    the absence of NaN in the results. Builds its own ``Bounds`` rather than
    using the module fixture, whose cached properties may already be warm.
    """
    port = build(PROGRAM)
    prem = float(port.tvar(0.5))
    bd = Bounds(port, premium=prem)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        wdf = bd.weight_df
        cdf = bd.cloud_df
    assert not wdf.isna().any().any()
    assert not cdf.isna().any().any()
