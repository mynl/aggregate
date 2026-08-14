"""Tests for the joint (ceded, net) occurrence bivariate distribution (1.0.0a20).

Covers ``dev/done/reins-bivariate.md``:

- ``Aggregate.occ_bivariate`` -- 2D-FFT joint law of aggregate occurrence ceded
  ``C`` and net ``N``; preconditions; auto / explicit per-axis sizing.
- ``BivariateDistribution`` -- marginals, mixed moments, correlation, contour.

The central validation is that the two marginals reproduce the univariate
occurrence aggregates already reported in ``reins_density_df`` /
``reins_stats_df`` / ``reins_summary_df``, and the anti-diagonal ``C + N``
reproduces the gross aggregate -- exact targets from the same densities.

The DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg``
(section Z).
"""
from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')  # headless contour smoke test
import matplotlib.pyplot as plt  # noqa: E402

from aggregate import build  # noqa: E402
from aggregate.constants import DefectiveDistributionWarning  # noqa: E402
from aggregate.bivariate import (  # noqa: E402
    BivariateDistribution, scatter_bivariate)

# Bleeding-edge bivariate machinery and among the heaviest cases in the suite;
# quarantined from the fast local loop (`-m 'not slow'`), still run in full/CI.
pytestmark = pytest.mark.slow


# Lognormal occurrence cover: the general (deficit-carrying) path.
OCC = (
    'agg BV.Occ 10 claims sev lognorm 50 cv 1.5 '
    'occurrence net of 50 xs 50 poisson'
)
# Bounded severity: negligible grid deficit -> tightest marginal match.
OCC_BOUNDED = (
    'agg BV.OccB 8 claims sev 300 * beta 2 3 '
    'occurrence net of 0.7 po 60 xs 40 poisson'
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
    'occurrence ceded to 0.7 po 60 xs 40 poisson'
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
    cd, nd = b.marginals
    assert cd.sum() == pytest.approx(1.0, abs=1e-6)
    assert nd.sum() == pytest.approx(1.0, abs=1e-6)
    assert b.deficit < 1e-6


@pytest.mark.parametrize('prog', [OCC, OCC_BOUNDED, OCC_FIXED])
def test_marginal_means_match_stats_df(prog):
    """Net / ceded marginal means match reins_stats_df occ totals.

    Axis order is the (x=net, y=ceded) convention, so axis 0 is Net and axis 1
    is Ceded. Means are reproduced exactly (the linear scatter preserves the
    first moment), independent of the auto-chosen bucket sizes.
    """
    a = _build(prog)
    b = a.occ_bivariate()
    net_d, ced_d = b.marginals
    rs = a.reins_stats_df
    net_mean, _, _ = _marginal_moments(b.axis_xs[0], net_d)
    ced_mean, _, _ = _marginal_moments(b.axis_xs[1], ced_d)
    assert net_mean == pytest.approx(rs.loc[('agg', 'mean'), ('occ', 'Net')], rel=2e-3)
    assert ced_mean == pytest.approx(rs.loc[('agg', 'mean'), ('occ', 'Ceded')], rel=2e-3)


@pytest.mark.parametrize('prog', [OCC_BOUNDED, OCC_FIXED])
def test_matched_grid_reproduces_univariate_cv(prog):
    """On the model grid (``bs == self.bs``) the marginals reproduce the
    univariate occurrence aggregate cv, not just the mean.

    Auto-sizing picks a *coarser* common bucket than the model grid, so the
    auto-sized cv legitimately differs from the model-grid ``reins_stats_df``
    value -- the rigorous identity holds only when the bivariate grid matches
    the model grid. Axis 0 is Net, axis 1 is Ceded (x=net, y=ceded), so the net
    cover sizes ``log2_x`` and the ceded cover sizes ``log2_y``. Bounded / fixed
    books are used so the FFT deficit is negligible.
    """
    a = _build(prog)
    rd = a.reins_density_df
    l2c = _cover_log2(rd['p_agg_ceded_occ'].to_numpy(), a.xs, a.bs)
    l2n = _cover_log2(rd['p_agg_net_occ'].to_numpy(), a.xs, a.bs)
    b = a.occ_bivariate(bs=a.bs, log2_x=l2n, log2_y=l2c)
    net_d, ced_d = b.marginals
    rs = a.reins_stats_df
    net_mean, net_cv, _ = _marginal_moments(b.axis_xs[0], net_d)
    ced_mean, ced_cv, _ = _marginal_moments(b.axis_xs[1], ced_d)
    assert net_mean == pytest.approx(rs.loc[('agg', 'mean'), ('occ', 'Net')], rel=1e-3)
    assert ced_mean == pytest.approx(rs.loc[('agg', 'mean'), ('occ', 'Ceded')], rel=1e-3)
    assert net_cv == pytest.approx(rs.loc[('agg', 'cv'), ('occ', 'Net')], rel=1e-3)
    assert ced_cv == pytest.approx(rs.loc[('agg', 'cv'), ('occ', 'Ceded')], rel=1e-3)


@pytest.mark.parametrize('prog', [OCC, OCC_BOUNDED])
def test_marginals_match_describe(prog):
    """Net / ceded marginal means match the reins_summary_df occ Est cells.

    Axis 0 is Net, axis 1 is Ceded (x=net, y=ceded convention).
    """
    a = _build(prog)
    b = a.occ_bivariate()
    net_d, ced_d = b.marginals
    rd = a.reins_summary_df
    net_mean, _, _ = _marginal_moments(b.axis_xs[0], net_d)
    ced_mean, _, _ = _marginal_moments(b.axis_xs[1], ced_d)
    assert net_mean == pytest.approx(rd.loc[('occ', 'net', 'agg'), 'Est EX'], rel=2e-3)
    assert ced_mean == pytest.approx(rd.loc[('occ', 'ceded', 'agg'), 'Est EX'], rel=2e-3)


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
    rho = a.occ_bivariate().corr
    assert np.isfinite(rho)
    assert -1.0 <= rho <= 1.0


def test_corr_positive_for_random_count():
    """Random (Poisson) count couples ceded and net -> positive correlation."""
    a = _build(OCC_BOUNDED)
    assert a.occ_bivariate().corr > 0


def test_poisson_count_increases_correlation():
    """A random claim count adds a common-shock coupling on top of the
    per-claim dependence, so Poisson frequency gives a higher ceded/net
    correlation than the same book with a fixed count.
    """
    pois = _build('agg BV.CP 8 claims sev 300 * beta 2 3 '
                  'occurrence net of 0.7 po 60 xs 40 poisson')
    fixed = _build('agg BV.CF dfreq [8] sev 300 * beta 2 3 '
                   'occurrence net of 0.7 po 60 xs 40')
    assert pois.occ_bivariate().corr > fixed.occ_bivariate().corr > 0


# ----------------------------------------------------------------------------
# Sizing and overrides
# ----------------------------------------------------------------------------

def test_explicit_overrides_respected():
    a = _build(OCC_BOUNDED)
    b = a.occ_bivariate(bs=2, log2_x=9, log2_y=10)
    assert b.bs[0] == 2
    assert b.bs[1] == 2
    assert b.density.shape == (1 << 9, 1 << 10)


def test_netceded_one_common_bs_no_finer_than_gross():
    """MV-3: both axes share one bs, coarsened from gross to fit (never finer).

    The legacy ``size_axis`` (per-axis moment-quantile sizer) is gone; netceded
    now measures each occ margin with ``balanced_window`` and uses a single
    common bs >= the gross bs.
    """
    a = _build(OCC)
    b = a.occ_bivariate()
    assert b.bs[0] == b.bs[1]              # one common bs
    assert b.bs[0] >= a.bs - 1e-12         # never finer than gross
    # the grid covers the margins -- mass is conserved
    cd, nd = b.marginals
    assert float(cd.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(nd.sum()) == pytest.approx(1.0, abs=1e-6)


def test_netceded_marginals_match_occ_views():
    """Each marginal reproduces the corresponding occurrence reins aggregate.

    Axis 0 is Net, axis 1 is Ceded (x=net, y=ceded convention).
    """
    a = _build(OCC)
    b = a.occ_bivariate()
    net_d, ced_d = b.marginals
    rs = a.reins_stats_df
    assert float((net_d * b.axis_xs[0]).sum()) == pytest.approx(
        rs.loc[('agg', 'mean'), ('occ', 'Net')], rel=2e-3)
    assert float((ced_d * b.axis_xs[1]).sum()) == pytest.approx(
        rs.loc[('agg', 'mean'), ('occ', 'Ceded')], rel=2e-3)


def test_netceded_refuses_a_pin_it_cannot_honor():
    """Pinning bs/log2 past the budget raises rather than clipping an axis.

    It warned and clipped the wider axis through 1.0.0a276, which is not a
    tail loss: with both axes pinned equal the rule cut axis 0 to the 16 bucket
    floor and the object then answered questions off a joint carrying half its
    mass ([Budget-Clip-Destroys-Axis]). The caller stated numbers that cannot
    all be honored, so the sizing says which and stops.
    """
    a = _build(OCC)
    with pytest.raises(ValueError, match='over the budget'):
        a.occ_bivariate(bs=a.bs, log2_x=14, log2_y=14)      # 14+14 > budget 20
    # and the escape it names works
    b = a.occ_bivariate(bs=a.bs, log2_x=14, log2_y=14, total_log2=28)
    assert not b._clipped
    assert [len(x) for x in b.axis_xs] == [1 << 14, 1 << 14]


# ----------------------------------------------------------------------------
# View-pairs (MV-5): netceded / grossceded / grossnet via occ_bivariate(views=)
# ----------------------------------------------------------------------------

# (view name in reins_stats_df, axis index, occ_bivariate views tuple)
_VIEW_PAIRS = {
    'netceded':   (('net', 'ceded'),   ('Net', 'Ceded')),
    'grossceded': (('gross', 'ceded'), ('Gross', 'Ceded')),
    'grossnet':   (('gross', 'net'),   ('Gross', 'Net')),
}


@pytest.mark.parametrize('views,labels', list(_VIEW_PAIRS.values()),
                         ids=list(_VIEW_PAIRS))
def test_view_pair_marginals_match_named_views(views, labels):
    """occ_bivariate(views=...) marginals reproduce the named occ aggregates.

    Axis 0 is ``views[0]``, axis 1 is ``views[1]`` (keyword names x-then-y).
    Means are exact (linear scatter preserves the first moment).
    """
    a = _build(OCC)
    b = a.occ_bivariate(views=views)
    assert b.unit_names == list(labels)
    m0, m1 = b.marginals
    rs = a.reins_stats_df
    assert float((m0 * b.axis_xs[0]).sum()) == pytest.approx(
        rs.loc[('agg', 'mean'), ('occ', labels[0])], rel=2e-3)
    assert float((m1 * b.axis_xs[1]).sum()) == pytest.approx(
        rs.loc[('agg', 'mean'), ('occ', labels[1])], rel=2e-3)
    assert m0.sum() == pytest.approx(1.0, abs=1e-6)
    assert m1.sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize('views', [p[0] for p in _VIEW_PAIRS.values()],
                         ids=list(_VIEW_PAIRS))
def test_view_pair_axis_order_x_then_y(views):
    """The keyword names the pair x-then-y: axis kinds follow ``views``."""
    a = _build(OCC)
    b = a.occ_bivariate(views=views)
    assert (b._axis_kind(0), b._axis_kind(1)) == views


def test_grossnet_anti_diagonal_is_ceded():
    """gross - net == ceded: E[Gross] - E[Net] reproduces E[Ceded]."""
    a = _build(OCC)
    b = a.occ_bivariate(views=('gross', 'net'))
    m0, m1 = b.marginals
    e_g = float((m0 * b.axis_xs[0]).sum())
    e_n = float((m1 * b.axis_xs[1]).sum())
    e_ceded = a.reins_stats_df.loc[('agg', 'mean'), ('occ', 'Ceded')]
    assert e_g - e_n == pytest.approx(e_ceded, rel=2e-2)


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
    assert 'BivariateAggregate' in repr(b)
    assert b.mode == 'netceded'
    assert '<table' in b._repr_html_()


def test_plot_smoke():
    a = _build(OCC_BOUNDED)
    b = a.occ_bivariate()
    fig, axs = plt.subplots(1, 2)
    out = b.plot(axs=axs)
    assert out is None                      # returns None (no double-render)
    assert axs.flat[0].get_title() == 'severity'
    assert axs.flat[1].get_title() == 'aggregate'
    b.plot(log=True)
    assert len(b.figure.axes) == 2
    plt.close('all')


def test_describe_and_info_netceded():
    a = _build(OCC_BOUNDED)
    b = a.occ_bivariate()
    df = b.summary_df
    assert {('Ceded', 'Agg'), ('Net', 'Agg'),
            ('total', 'Agg')}.issubset(set(df.index))
    assert np.isclose(float(df.loc[('total', 'Agg'), 'EX']),
                      float(df.loc[('Ceded', 'Agg'), 'EX'])
                      + float(df.loc[('Net', 'Agg'), 'EX']))
    assert np.isclose(float(b.dependency_df.loc['Agg', 'corr']), b.corr)
    assert 'netceded' in b.info
