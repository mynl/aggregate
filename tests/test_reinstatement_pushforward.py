"""Tests for the generic pushforward engine on ``BivariateDistribution``.

Covers the ``[engine]`` workstream of ``dev/plan-reinstatements.md`` (pre-plan
sections 9-10, 15): the pushforward of a joint (or 1-D marginal) density through a
vectorized scalar map, returning a ``GridDistribution``, and the exact
source-grid moments used as the audit's "EX" column. These tests stand alone
from any reinstatement contract logic -- they validate the engine itself.
"""

import numpy as np
import pytest

from aggregate.bivariate import (BivariateDistribution, pushforward_1d,
                                  _pushforward_grid, _scatter_1d)
from aggregate._grid_distribution import GridDistribution


def _toy_joint():
    """A small hand-built joint of ``(L, R)`` on integer grids, ``R <= L``."""
    ax0 = np.array([0., 1., 2., 3.])          # L
    ax1 = np.array([0., 1., 2.])              # R
    dens = np.array([
        [0.10, 0.05, 0.00],
        [0.05, 0.20, 0.05],
        [0.00, 0.10, 0.15],
        [0.00, 0.05, 0.20],
    ])
    dens = dens / dens.sum()
    return BivariateDistribution(dens, ax0, ax1, 1.0, 1.0,
                                 meta={'axis_names': ('L', 'R')})


def _direct_moment(bd, f, k):
    """Brute-force ``E[f(L,R)^k]`` straight off the joint grid."""
    L = bd.axis0[:, None]
    R = bd.axis1[None, :]
    return float((bd.density * f(L, R) ** k).sum())


# ----------------------------------------------------------------------
# transformed_moments: exact on the source grid
# ----------------------------------------------------------------------
def test_transformed_moments_mean_exact():
    bd = _toy_joint()
    f = lambda l, r: l - r                    # signed net loss
    em = bd.transformed_moments(f, max_order=2)
    assert em['mean'] == pytest.approx(_direct_moment(bd, f, 1))
    var = _direct_moment(bd, f, 2) - _direct_moment(bd, f, 1) ** 2
    assert em['var'] == pytest.approx(var)
    assert em['sd'] == pytest.approx(np.sqrt(var))


def test_transformed_moments_nonlinear():
    bd = _toy_joint()
    f = lambda l, r: np.minimum(r, 1.5)       # a cap -- nonlinear in R
    em = bd.transformed_moments(f, max_order=3)
    assert em['mean'] == pytest.approx(_direct_moment(bd, f, 1))


# ----------------------------------------------------------------------
# pushforward (2-D): mass, mean, signed grid, EX==Est
# ----------------------------------------------------------------------
def test_pushforward_preserves_mass_and_mean():
    bd = _toy_joint()
    f = lambda l, r: l - r
    gd = bd.pushforward(f, name='net', is_loss_value=True)
    assert isinstance(gd, GridDistribution)
    assert gd.p.sum() == pytest.approx(1.0)
    # Est (rebucketed) mean equals EX (exact source-grid) mean
    assert gd.mean() == pytest.approx(bd.transformed_moments(f)['mean'])
    assert gd.clipped_mass == pytest.approx(0.0)


def test_pushforward_signed_grid():
    bd = _toy_joint()
    # net underwriting result: P_G - L + (something) -> straddles zero
    gd = bd.pushforward(lambda l, r: 1.5 - l + r, name='uw', is_loss_value=False)
    assert gd.x.min() < 0 < gd.x.max()
    assert gd.is_loss_value is False


def test_pushforward_nonlinear_mean_matches_exact():
    bd = _toy_joint()
    f = lambda l, r: np.minimum(r, 1.5)
    gd = bd.pushforward(f)
    assert gd.mean() == pytest.approx(bd.transformed_moments(f)['mean'])


# ----------------------------------------------------------------------
# scheme, chunking, grid invariance
# ----------------------------------------------------------------------
def test_linear_and_nearest_agree_on_grid_aligned_values():
    # integer-valued map onto an integer grid: nearest and linear coincide
    bd = _toy_joint()
    f = lambda l, r: l + r
    g_lin = bd.pushforward(f, bs=1.0, scheme='linear')
    g_near = bd.pushforward(f, bs=1.0, scheme='nearest')
    assert g_lin.mean() == pytest.approx(g_near.mean())
    np.testing.assert_allclose(g_lin.p, g_near.p, atol=1e-12)


def test_chunking_matches_unchunked():
    bd = _toy_joint()
    f = lambda l, r: l - 0.5 * r
    g_whole = bd.pushforward(f, bs=0.25)
    g_chunk = bd.pushforward(f, bs=0.25, chunk_size=1)
    np.testing.assert_allclose(g_whole.p, g_chunk.p, atol=1e-12)
    assert g_whole.mean() == pytest.approx(g_chunk.mean())


def test_mean_invariant_across_bucket_size():
    bd = _toy_joint()
    f = lambda l, r: l - r
    means = [bd.pushforward(f, bs=bs).mean() for bs in (0.5, 0.25, 0.1, 0.05)]
    exact = bd.transformed_moments(f)['mean']
    for m in means:
        assert m == pytest.approx(exact, abs=1e-9)


# ----------------------------------------------------------------------
# clipping report
# ----------------------------------------------------------------------
def test_clipping_reported_when_window_too_narrow():
    bd = _toy_joint()
    f = lambda l, r: l - r                     # range is [-2, 3]
    with pytest.warns(Warning):
        gd = bd.pushforward(f, window=(0.0, 1.0))
    assert gd.clipped_mass > 0
    # mass is retained (piled on edges), not lost
    assert gd.p.sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------
# 1-D path
# ----------------------------------------------------------------------
def test_pushforward_1d_identity_recovers_mean():
    x = np.array([0., 1., 2., 3., 4.])
    p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    gd = pushforward_1d(x, p, lambda z: 2.0 * z, name='double')
    assert gd.mean() == pytest.approx(float((2.0 * x * p).sum()))
    assert gd.p.sum() == pytest.approx(1.0)


def test_pushforward_1d_matches_2d_on_axis0_only_map():
    # a map depending only on L should give the same law via the 1-D path on
    # the L-marginal as via the 2-D joint
    bd = _toy_joint()
    f1 = lambda l: np.minimum(l, 2.0)
    f2 = lambda l, r: np.minimum(l, 2.0)
    l_marg, _ = bd.marginals()
    g1 = pushforward_1d(bd.axis0, l_marg, f1, bs=0.5)
    g2 = bd.pushforward(f2, bs=0.5)
    assert g1.mean() == pytest.approx(g2.mean())


# ----------------------------------------------------------------------
# low-level helpers
# ----------------------------------------------------------------------
def test_scatter_1d_linear_splits_mean_preserving():
    # one unit of mass at value 0.3 on a unit grid -> 0.7 at 0, 0.3 at 1
    mass, clipped = _scatter_1d(np.array([0.3]), np.array([1.0]),
                                z0=0.0, bs=1.0, n_out=4, scheme='linear')
    np.testing.assert_allclose(mass, [0.7, 0.3, 0.0, 0.0])
    assert clipped == 0.0
    # mean preserved
    assert float((np.arange(4) * mass).sum()) == pytest.approx(0.3)


def test_pushforward_grid_aligns_zero_for_signed_range():
    z0, bs, n_out = _pushforward_grid(-2.0, 3.0, bs=1.0)
    assert z0 == pytest.approx(-2.0)
    grid = z0 + bs * np.arange(n_out)
    assert 0.0 in grid                          # zero lands on the grid
    assert grid[-1] >= 3.0
