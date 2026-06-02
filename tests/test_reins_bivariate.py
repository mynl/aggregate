"""Tests for the joint (ceded, net) occurrence bivariate distribution (1.0.0a20).

Covers ``dev/done/reins-bivariate.md``:

- ``Aggregate.occ_bivariate`` -- 2D-FFT joint law of aggregate occurrence ceded
  ``C`` and net ``N``; preconditions; auto / explicit per-axis sizing.
- ``BivariateDistribution`` -- marginals, mixed moments, correlation, contour.

The central validation is that the two marginals reproduce the univariate
occurrence aggregates already reported in ``reins_density_df`` /
``reins_stats_df`` / ``reins_describe``, and the anti-diagonal ``C + N``
reproduces the gross aggregate -- exact targets from the same densities.

The DecL programs are mirrored in ``src/aggregate/agg/test_decl.agg``
(section Z).
"""
from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')  # headless contour smoke test
import matplotlib.pyplot as plt  # noqa: E402

from aggregate import build  # noqa: E402
from aggregate.bivariate import (  # noqa: E402
    BivariateDistribution, size_axis, scatter_bivariate)


# Lognormal occurrence cover: the general (deficit-carrying) path.
OCC = (
    'agg BV.Occ 10 claims sev lognorm 50 cv 1.5 '
    'occurrence net of 50 xs 50 poisson'
)
# Bounded severity: negligible grid deficit -> tightest marginal match.
OCC_BOUNDED = (
    'agg BV.OccB 8 claims sev 300 * beta 2 3 '
    'occurrence net of 0.7 so 60 xs 40 poisson'
)
# Fixed claim count (count > 1 exercises the general 2D path with z**n PGF).
OCC_FIXED = (
    'agg BV.Fix dfreq [4] dsev [1:20] '
    'occurrence net of 6 xs 6'
)
# A "ceded to" output view -- occ_bivariate ignores the requested view and
# always reports the joint per-occurrence (ceded, net) aggregate.
OCC_CEDED = (
    'agg BV.Ced 8 claims sev 300 * beta 2 3 '
    'occurrence ceded to 0.7 so 60 xs 40 poisson'
)


def _build(prog, **kw):
    kw.setdefault('bs', 1)
    kw.setdefault('log2', 16)
    return build(prog, **kw)


def _marginal_moments(grid, density):
    """Return ``(mean, cv, skew)`` of a 1D marginal density on ``grid``."""
    tot = density.sum()
    m1 = (density * grid).sum() / tot
    m2 = (density * grid ** 2).sum() / tot
    m3 = (density * grid ** 3).sum() / tot
    var = m2 - m1 ** 2
    sd = np.sqrt(var)
    cv = sd / m1 if m1 else np.nan
    skew = (m3 - 3 * m1 * var - m1 ** 3) / sd ** 3 if sd > 0 else np.nan
    return m1, cv, skew


def _cover_log2(margin, xs, bs):
    """Smallest log2 grid length at bucket ``bs`` covering a margin's support."""
    cdf = np.cumsum(margin)
    cdf = cdf / cdf[-1]
    vmax = xs[int(np.searchsorted(cdf, 1 - 1e-9))]
    return max(6, int(np.ceil(np.log2(vmax / bs + 1))))


# ----------------------------------------------------------------------------
# Preconditions
# ----------------------------------------------------------------------------

def test_requires_occ_reins():
    """Gross book (no occurrence cover) -> ValueError."""
    a = _build('agg BV.Gross 10 claims sev lognorm 50 cv 1.5 poisson')
    with pytest.raises(ValueError, match='occurrence reinsurance'):
        a.occ_bivariate()


def test_requires_updated():
    """Un-updated object -> ValueError."""
    a = _build(OCC, update=False)
    with pytest.raises(ValueError, match='updated'):
        a.occ_bivariate()


# ----------------------------------------------------------------------------
# Marginals reproduce the univariate occurrence aggregates
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('prog', [OCC, OCC_BOUNDED, OCC_FIXED, OCC_CEDED])
def test_marginals_sum_to_one(prog):
    a = _build(prog)
    b = a.occ_bivariate()
    cd, nd = b.marginals()
    assert cd.sum() == pytest.approx(1.0, abs=1e-6)
    assert nd.sum() == pytest.approx(1.0, abs=1e-6)
    assert b.meta['deficit'] < 1e-6


@pytest.mark.parametrize('prog', [OCC, OCC_BOUNDED, OCC_FIXED])
def test_marginal_means_match_stats_df(prog):
    """Ceded / net marginal means match reins_stats_df occ totals.

    Means are reproduced exactly (the linear scatter preserves the first
    moment), independent of the auto-chosen bucket sizes.
    """
    a = _build(prog)
    b = a.occ_bivariate()
    cd, nd = b.marginals()
    rs = a.reins_stats_df
    c_mean, _, _ = _marginal_moments(b.ceded, cd)
    n_mean, _, _ = _marginal_moments(b.net, nd)
    assert c_mean == pytest.approx(rs.loc[('agg', 'mean'), ('occ', 'Ceded')], rel=2e-3)
    assert n_mean == pytest.approx(rs.loc[('agg', 'mean'), ('occ', 'Net')], rel=2e-3)


@pytest.mark.parametrize('prog', [OCC_BOUNDED, OCC_FIXED])
def test_matched_grid_reproduces_univariate_cv(prog):
    """On the model grid (``bs_ceded == bs_net == self.bs``) the marginals
    reproduce the univariate occurrence aggregate cv, not just the mean.

    Auto-sizing picks a *finer* ceded bucket than the model grid, so the
    auto-sized cv is more accurate than (and legitimately differs from) the
    coarse ``reins_stats_df`` value -- the rigorous identity holds only when
    the bivariate grid matches the model grid. Bounded / fixed books are used
    so the FFT deficit is negligible.
    """
    a = _build(prog)
    rd = a.reins_density_df
    l2c = _cover_log2(rd['p_agg_ceded_occ'].to_numpy(), a.xs, a.bs)
    l2n = _cover_log2(rd['p_agg_net_occ'].to_numpy(), a.xs, a.bs)
    b = a.occ_bivariate(bs_ceded=a.bs, bs_net=a.bs, log2_ceded=l2c, log2_net=l2n)
    cd, nd = b.marginals()
    rs = a.reins_stats_df
    c_mean, c_cv, _ = _marginal_moments(b.ceded, cd)
    n_mean, n_cv, _ = _marginal_moments(b.net, nd)
    assert c_mean == pytest.approx(rs.loc[('agg', 'mean'), ('occ', 'Ceded')], rel=1e-3)
    assert n_mean == pytest.approx(rs.loc[('agg', 'mean'), ('occ', 'Net')], rel=1e-3)
    assert c_cv == pytest.approx(rs.loc[('agg', 'cv'), ('occ', 'Ceded')], rel=1e-3)
    assert n_cv == pytest.approx(rs.loc[('agg', 'cv'), ('occ', 'Net')], rel=1e-3)


@pytest.mark.parametrize('prog', [OCC, OCC_BOUNDED])
def test_marginals_match_describe(prog):
    """Ceded / net marginal means match the reins_describe occ Est cells."""
    a = _build(prog)
    b = a.occ_bivariate()
    cd, nd = b.marginals()
    rd = a.reins_describe
    c_mean, _, _ = _marginal_moments(b.ceded, cd)
    n_mean, _, _ = _marginal_moments(b.net, nd)
    assert c_mean == pytest.approx(rd.loc[('occ', 'ceded', 'agg'), 'Est EX'], rel=2e-3)
    assert n_mean == pytest.approx(rd.loc[('occ', 'net', 'agg'), 'Est EX'], rel=2e-3)


# ----------------------------------------------------------------------------
# Anti-diagonal C + N reproduces the gross aggregate; additivity
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('prog', [OCC, OCC_BOUNDED, OCC_FIXED])
def test_additivity_mean(prog):
    """E[C] + E[N] == E[gross aggregate]."""
    a = _build(prog)
    b = a.occ_bivariate()
    m = b.moments(1).to_numpy()
    tot = m[0, 0]
    e_c, e_n = m[1, 0] / tot, m[0, 1] / tot
    gross_mean = a.reins_stats_df.loc[('agg', 'mean'), ('occ', 'Gross')]
    assert e_c + e_n == pytest.approx(gross_mean, rel=2e-3)


@pytest.mark.parametrize('prog', [OCC_BOUNDED, OCC_FIXED])
def test_anti_diagonal_variance(prog):
    """Var(C + N) == Var(gross): the joint law respects the comonotone sum.

    Var(C+N) = Var C + Var N + 2 Cov(C, N) from the mixed moments.
    """
    a = _build(prog)
    b = a.occ_bivariate()
    m = b.moments(2).to_numpy()
    tot = m[0, 0]
    e_c, e_n = m[1, 0] / tot, m[0, 1] / tot
    var_c = m[2, 0] / tot - e_c ** 2
    var_n = m[0, 2] / tot - e_n ** 2
    cov = m[1, 1] / tot - e_c * e_n
    var_sum = var_c + var_n + 2 * cov
    gross_cv = a.reins_stats_df.loc[('agg', 'cv'), ('occ', 'Gross')]
    gross_mean = a.reins_stats_df.loc[('agg', 'mean'), ('occ', 'Gross')]
    gross_var = (gross_cv * gross_mean) ** 2
    assert var_sum == pytest.approx(gross_var, rel=1e-2)


# ----------------------------------------------------------------------------
# Correlation
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('prog', [OCC, OCC_BOUNDED, OCC_FIXED])
def test_corr_in_range(prog):
    a = _build(prog)
    rho = a.occ_bivariate().corr()
    assert np.isfinite(rho)
    assert -1.0 <= rho <= 1.0


def test_corr_positive_for_random_count():
    """Random (Poisson) count couples ceded and net -> positive correlation."""
    a = _build(OCC_BOUNDED)
    assert a.occ_bivariate().corr() > 0


def test_poisson_count_increases_correlation():
    """A random claim count adds a common-shock coupling on top of the
    per-claim dependence, so Poisson frequency gives a higher ceded/net
    correlation than the same book with a fixed count.
    """
    pois = _build('agg BV.CP 8 claims sev 300 * beta 2 3 '
                  'occurrence net of 0.7 so 60 xs 40 poisson')
    fixed = _build('agg BV.CF dfreq [8] sev 300 * beta 2 3 '
                   'occurrence net of 0.7 so 60 xs 40')
    assert pois.occ_bivariate().corr() > fixed.occ_bivariate().corr() > 0


# ----------------------------------------------------------------------------
# Sizing and overrides
# ----------------------------------------------------------------------------

def test_explicit_overrides_respected():
    a = _build(OCC_BOUNDED)
    b = a.occ_bivariate(bs_ceded=2, bs_net=2, log2_ceded=9, log2_net=10)
    assert b.bs_ceded == 2
    assert b.bs_net == 2
    assert b.density.shape == (1 << 9, 1 << 10)


def test_size_axis_covers_support():
    """size_axis returns a grid whose top covers the margin's effective max."""
    a = _build(OCC)
    rd = a.reins_density_df
    cdf = np.cumsum(rd['p_agg_ceded_occ'].to_numpy())
    cdf /= cdf[-1]
    vmax = a.xs[int(np.searchsorted(cdf, 1 - 1e-9))]
    bs, log2 = size_axis(rd['p_agg_ceded_occ'].to_numpy(), a.xs, a.bs)
    assert bs * ((1 << log2) - 1) >= vmax


def test_size_axis_both_overrides_bypass():
    bs, log2 = size_axis(np.array([1.0]), np.array([0.0, 1.0]), 1.0,
                         bs=3.0, log2=7)
    assert (bs, log2) == (3.0, 7)


# ----------------------------------------------------------------------------
# Scatter helper: mass and (linear) marginal-mean preservation
# ----------------------------------------------------------------------------

def test_scatter_conserves_mass():
    cv = np.array([0.0, 3.7, 9.2])
    nv = np.array([0.0, 1.4, 6.6])
    mass = np.array([0.2, 0.5, 0.3])
    for scheme in ('linear', 'nearest'):
        s = scatter_bivariate(cv, nv, mass, 1.0, 1.0, 16, 16, scheme=scheme)
        assert s.sum() == pytest.approx(mass.sum())


def test_scatter_linear_preserves_marginal_means():
    cv = np.array([0.0, 3.7, 9.2])
    nv = np.array([0.0, 1.4, 6.6])
    mass = np.array([0.2, 0.5, 0.3])
    s = scatter_bivariate(cv, nv, mass, 1.0, 1.0, 16, 16, scheme='linear')
    gc = np.arange(16)
    e_c = (s.sum(axis=1) * gc).sum()
    e_n = (s.sum(axis=0) * gc).sum()
    assert e_c == pytest.approx((cv * mass).sum())
    assert e_n == pytest.approx((nv * mass).sum())


# ----------------------------------------------------------------------------
# Moments table, repr, contour smoke
# ----------------------------------------------------------------------------

def test_moments_table_shape_and_total():
    a = _build(OCC_BOUNDED)
    m = a.occ_bivariate().moments(3)
    assert m.shape == (4, 4)
    assert m.iloc[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_repr_and_html():
    a = _build(OCC_BOUNDED)
    b = a.occ_bivariate()
    assert 'BivariateDistribution' in repr(b)
    assert '<table' in b._repr_html_()


def test_contour_smoke():
    a = _build(OCC_BOUNDED)
    b = a.occ_bivariate()
    fig, ax = plt.subplots()
    out = b.contour(ax=ax)
    assert out is ax
    out2 = b.contour(log=True)
    assert out2 is not None
    plt.close('all')
