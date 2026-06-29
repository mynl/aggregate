"""Unit tests for the domain-free bivariate leg kernel (``aggregate.legs``).

Exercises the kernel on **synthetic** sources -- no insurance vocabulary -- per
``dev/plan-bivariate-legs.md`` (Tests / regression bar): degenerate-vs-full
agreement, many-to-one binning, leg algebra, and ``net`` as a fresh pushforward
(means add, SDs come from the joint).
"""

import numpy as np
import pytest

from aggregate.legs import Leg, LegSet, GraphSource
from aggregate.bivariate import BivariateDistribution


# ----------------------------------------------------------------------
# fixtures: a 1-D grid and the two sources over it
# ----------------------------------------------------------------------
def _grid_density():
    """A small normalized 1-D source ``(X, p)`` (bs = 1, 11 atoms)."""
    x = np.arange(0.0, 10.0001, 1.0)
    p = np.array([0.05, 0.10, 0.15, 0.15, 0.12, 0.10, 0.10, 0.08, 0.07, 0.05, 0.03])
    return x, p / p.sum()


def _comonotone_full(kappa_slope=0.5):
    """A *full* bivariate whose mass sits exactly on ``Y = kappa_slope * X``.

    Built so the degenerate :class:`GraphSource` and this 2-D source describe the
    same law -- the agreement test then checks the kernel cannot tell them apart.
    """
    x, p = _grid_density()
    y = np.arange(0.0, kappa_slope * 10.0 + 1e-9, kappa_slope)
    dens = np.zeros((len(x), len(y)))
    for i, xi in enumerate(x):
        j = int(round(kappa_slope * xi / kappa_slope))
        dens[i, j] = p[i]
    bv = BivariateDistribution(dens, x, y, 1.0, kappa_slope)
    gs = GraphSource(x, p, kappa=lambda v: kappa_slope * np.asarray(v, dtype=float))
    return gs, bv


# ----------------------------------------------------------------------
# Leg: identity, immutability, algebra
# ----------------------------------------------------------------------
def test_leg_is_frozen_and_hides_map_in_repr():
    leg = Leg('ceded_loss', lambda x, y: y)
    assert leg.name == 'ceded_loss' and leg.is_value is True
    assert 'map' not in repr(leg)            # field(repr=False)
    with pytest.raises(Exception):
        leg.name = 'other'                   # frozen


def test_leg_algebra_signs_are_summation():
    """net = gross - ceded; combine() sums; signs live in the maps."""
    x = np.arange(5.0)
    gross = Leg('gross', lambda a, b: a)
    ceded = Leg('ceded', lambda a, b: b)
    net = gross - ceded
    assert np.allclose(net.map(x, 0.4 * x), x - 0.4 * x)
    # combine([gross, -ceded]) is the same margin
    margin = Leg.combine('margin', [gross, -ceded])
    assert np.allclose(margin.map(x, 0.4 * x), x - 0.4 * x)
    # scaling and negation
    assert np.allclose((gross.scaled(0.5)).map(x, 0.0), 0.5 * x)
    assert np.allclose((-gross).map(x, 0.0), -x)


# ----------------------------------------------------------------------
# Source protocol: degenerate vs full agreement
# ----------------------------------------------------------------------
def test_degenerate_and_full_sources_agree():
    """A leg pushed over GraphSource and over the equivalent full bivariate match."""
    gs, bv = _comonotone_full(0.5)
    ceded = Leg('ceded_loss', lambda x, y: y)
    # exact moments agree to machine precision (means add, no rebucketing)
    sg = gs.transformed_moments(ceded.map)
    sb = bv.transformed_moments(ceded.map)
    assert sg['mean'] == pytest.approx(sb['mean'], rel=1e-12)
    assert sg['sd'] == pytest.approx(sb['sd'], rel=1e-9)
    # the pushforward laws agree on a shared grid
    gd_g = gs.pushforward(ceded.map, bs=0.5, window=(0.0, 5.0))
    gd_b = bv.pushforward(ceded.map, bs=0.5, window=(0.0, 5.0))
    for lvl in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert gd_g.q(lvl) == pytest.approx(gd_b.q(lvl), abs=1e-9)


def test_graph_source_factors_a_two_arg_map_through_x():
    """On a degenerate source, net loss X - kappa(X) is a 1-D pushforward."""
    x, p = _grid_density()
    gs = GraphSource(x, p, kappa=lambda v: 0.5 * np.asarray(v, dtype=float))
    net = Leg('net_loss', lambda a, b: a - b)
    s = gs.transformed_moments(net.map)
    assert s['mean'] == pytest.approx(float(np.sum(p * (x - 0.5 * x))))


# ----------------------------------------------------------------------
# Many-to-one binning: a saturating map accumulates, it does not relabel
# ----------------------------------------------------------------------
def test_many_to_one_map_accumulates_mass_at_the_cap():
    """A saturating leg min(X, cap) piles all the X >= cap mass onto one atom."""
    x, p = _grid_density()
    cap = 6.0
    gs = GraphSource(x, p)                    # kappa = 0; leg ignores Y
    capped = Leg('capped', lambda a, b: np.minimum(a, cap))
    gd = gs.pushforward(capped.map, bs=1.0, window=(0.0, cap))
    expected_top = float(p[x >= cap].sum())   # all mass at or above the cap
    top = gd.p[np.argmin(np.abs(gd.x - cap))]
    assert top == pytest.approx(expected_top, abs=1e-9)
    # exact mean of the capped variable, straight off the source
    assert gd.mean() == pytest.approx(float(np.sum(p * np.minimum(x, cap))),
                                      abs=1e-9)


# ----------------------------------------------------------------------
# Net is a fresh pushforward: means add, SDs come from the joint
# ----------------------------------------------------------------------
def test_net_means_add_but_sd_is_from_the_joint():
    """For an independent joint, E[X-Y] = E[X]-E[Y] yet Var(X-Y)=VarX+VarY."""
    x, px = _grid_density()
    y = np.arange(0.0, 5.0001, 1.0)
    py = np.array([0.3, 0.25, 0.2, 0.15, 0.07, 0.03])
    py = py / py.sum()
    dens = np.outer(px, py)                   # independent margins
    bv = BivariateDistribution(dens, x, y, 1.0, 1.0)
    gross = Leg('gross_loss', lambda a, b: a)
    ceded = Leg('ceded_loss', lambda a, b: b)
    net = gross - ceded
    legs = LegSet([gross, ceded, net])
    stats = legs.stats_df(bv)
    # means add exactly
    assert stats.loc['gross_loss-ceded_loss', 'mean'] == pytest.approx(
        stats.loc['gross_loss', 'mean'] - stats.loc['ceded_loss', 'mean'],
        rel=1e-12)
    # variance is the joint's: VarX + VarY (zero covariance for independence),
    # which is NOT (sd_X - sd_Y) ** 2 -- "SDs don't subtract"
    var_x = stats.loc['gross_loss', 'sd'] ** 2
    var_y = stats.loc['ceded_loss', 'sd'] ** 2
    var_net = stats.loc['gross_loss-ceded_loss', 'sd'] ** 2
    assert var_net == pytest.approx(var_x + var_y, rel=1e-9)
    assert var_net != pytest.approx((stats.loc['gross_loss', 'sd']
                                     - stats.loc['ceded_loss', 'sd']) ** 2)


# ----------------------------------------------------------------------
# LegSet: ordered evaluation over a source
# ----------------------------------------------------------------------
def test_legset_distributions_and_stats_cover_every_leg():
    gs, _ = _comonotone_full(0.4)
    legs = LegSet([
        Leg('gross_loss', lambda x, y: x),
        Leg('ceded_loss', lambda x, y: y),
        Leg('net_loss', lambda x, y: x - y),
    ])
    assert legs.names == ['gross_loss', 'ceded_loss', 'net_loss']
    dists = legs.distributions(gs)
    assert set(dists) == set(legs.names)
    stats = legs.stats_df(gs)
    # net mean = gross mean - ceded mean (additivity through the kernel)
    assert stats.loc['net_loss', 'mean'] == pytest.approx(
        stats.loc['gross_loss', 'mean'] - stats.loc['ceded_loss', 'mean'],
        rel=1e-12)
