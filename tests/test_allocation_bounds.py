"""
Tests for ``aggregate.bounds.AllocationBounds`` / ``Portfolio.allocation_bounds``.

Two layers:

1. A tiny discrete portfolio where every quantity — curve vertices, hulls,
   bounds, achieving biTVaRs, p_star — is computed by hand and asserted
   exactly (to float tolerance).
2. Self-consistency audits on a continuous (lognormal/gamma) portfolio:
   additivity, first-principles repricing of the bounds, width monotone in
   the premium, feasibility errors, and the degenerate-distortion fallback.

Hand calculation for the discrete case
--------------------------------------

Units A ~ dsev [0, 8], B ~ dsev [0, 2], each with one certain claim,
independent.  Total X has four outcomes, each probability 1/4:

    x:          0     2     8     10
    E[A|X=x]:   0     0     8     8
    E[B|X=x]:   0     2     0     2

Curve vertices (T_m, a_m) at p = 0, .25, .5, .75 (= F breakpoints):

    m  p     T = E[X|X>=x_m]   a_A = E[A|X>=x_m]   a_B
    0  0     5                 4                   1
    1  .25   20/3              16/3                4/3
    2  .5    9                 8                   1
    3  .75   10                8                   2

Unit A: vertices (5,4), (20/3,16/3), (9,8), (10,8).  The first three lie on
the line of slope 0.8 through (5,4) — so the lower hull is the single chord
(5,4)-(10,8) (collinear interior vertices removed; (9,8) lies above it).
Upper hull: slopes 1 then 0 — (5,4), (9,8), (10,8).

Unit B: vertices (5,1), (20/3,4/3), (9,1), (10,2).  Lower hull (5,1), (9,1),
(10,2); upper hull is the chord (5,1)-(10,2) of slope 0.2 (the (20/3,4/3)
vertex is exactly collinear with it).

At P = 7:
    lower A = 4 + 0.8(7-5)  = 5.6   biTVaR (0, .75), w = 2/5
    upper A = 4 + 1.0(7-5)  = 6.0   biTVaR (0, .50), w = 1/2
    lower B = 1             = 1.0   biTVaR (0, .50), w = 1/2
    upper B = 1 + 0.2(7-5)  = 1.4   biTVaR (0, .75), w = 2/5

p_star(7): 7 lies between T_1 = 20/3 and T_2 = 9, atom at x = 2:
    1 - p* = S_2 (T_2 - 2)/(P - 2) = 0.5 * 7/5 = 0.7,  p* = 0.3.
"""

import numpy as np
import pytest

from aggregate import build
from aggregate.spectral import Distortion


TOL = 1e-12


@pytest.fixture(scope='module')
def discrete_ab():
    """The hand-checkable portfolio and its AllocationBounds."""
    port = build("""port SmallAB
        agg A dfreq [1] dsev [0 8]
        agg B dfreq [1] dsev [0 2]
    """, bs=1, log2=8)
    return port.allocation_bounds()


@pytest.fixture(scope='module')
def cts_ab():
    """A continuous two-unit portfolio for self-consistency audits."""
    port = build("""port CtsAB
        agg A 10 claims sev lognorm 10 cv 1.25 poisson
        agg B  4 claims sev gamma 25 cv 0.8 mixed gamma 0.6
    """)
    return port.allocation_bounds()


# ---------------------------------------------------------------------------
# Exact hand-checked values on the discrete portfolio
# ---------------------------------------------------------------------------

def test_curve_vertices_exact(discrete_ab):
    cd = discrete_ab.curve_df
    assert np.allclose(cd.index, [0, .25, .5, .75], atol=TOL)
    assert np.allclose(cd['exeqa_total'], [5, 20 / 3, 9, 10], atol=TOL)
    assert np.allclose(cd['exeqa_A'], [4, 16 / 3, 8, 8], atol=TOL)
    assert np.allclose(cd['exeqa_B'], [1, 4 / 3, 1, 2], atol=TOL)


def test_premium_range(discrete_ab):
    lo, hi = discrete_ab.premium_range
    assert lo == pytest.approx(5, abs=TOL)      # E[X]
    assert hi == pytest.approx(10, abs=TOL)     # ess sup


def test_bounds_at_7(discrete_ab):
    b = discrete_ab.bounds(7.0)
    assert b.loc[(7.0, 'A'), 'lower'] == pytest.approx(5.6, abs=TOL)
    assert b.loc[(7.0, 'A'), 'upper'] == pytest.approx(6.0, abs=TOL)
    assert b.loc[(7.0, 'B'), 'lower'] == pytest.approx(1.0, abs=TOL)
    assert b.loc[(7.0, 'B'), 'upper'] == pytest.approx(1.4, abs=TOL)


def test_bitvars_at_7(discrete_ab):
    bv = discrete_ab.bitvars(7.0)
    cd = discrete_ab.curve_df
    p_star = 0.3

    # Where the hull edge is unambiguous the biTVaR is pinned exactly.
    # upper A: chord (p=0) -> (p=.5), w = (7-5)/(9-5); lower B identical.
    r = bv.loc[(7.0, 'A', 'upper')]
    assert (r['p0'], r['p1']) == (0.0, 0.5)
    assert r['w1'] == pytest.approx(0.5, abs=TOL)
    r = bv.loc[(7.0, 'B', 'lower')]
    assert (r['p0'], r['p1']) == (0.0, 0.5)
    assert r['w1'] == pytest.approx(0.5, abs=TOL)

    # lower A / upper B run along an exactly-collinear stretch of the curve
    # (vertices at p = 0, .25 and the chord to .75 share slope), so the
    # optimizer is non-unique and float noise decides which edge is
    # reported.  Assert the invariants instead: the biTVaR straddles
    # p_star, has a proper weight, and its chord reproduces the bound.
    for (u, side, value) in [('A', 'lower', 5.6), ('B', 'upper', 1.4)]:
        r = bv.loc[(7.0, u, side)]
        assert r['p0'] <= p_star <= r['p1']
        assert 0.0 <= r['w1'] <= 1.0
        chord = ((1 - r['w1']) * cd.loc[r['p0'], f'exeqa_{u}']
                 + r['w1'] * cd.loc[r['p1'], f'exeqa_{u}'])
        assert chord == pytest.approx(value, abs=1e-10)
        assert r['value'] == pytest.approx(value, abs=1e-10)


def test_p_star_exact(discrete_ab):
    assert discrete_ab.p_star(7.0) == pytest.approx(0.3, abs=TOL)
    # At a vertex T value, p_star returns the vertex p.
    assert discrete_ab.p_star(9.0) == pytest.approx(0.5, abs=TOL)
    assert discrete_ab.p_star(5.0) == pytest.approx(0.0, abs=TOL)


def test_check_reprices_exactly(discrete_ab):
    chk = discrete_ab.check(7.0)
    assert np.abs(chk['err']).max() < 1e-12
    assert np.abs(chk['total_err']).max() < 1e-12


def test_endpoint_bounds_collapse_to_point(discrete_ab):
    # At P = E[X] only the mean distortion qualifies; at P = ess sup only
    # the max distortion: the allocation range collapses (to E[X_i] and
    # kappa_i(x_max) respectively).
    b = discrete_ab.bounds(5.0)
    assert np.abs(b['width']).max() < TOL
    assert b.loc[(5.0, 'A'), 'lower'] == pytest.approx(4.0, abs=TOL)
    b = discrete_ab.bounds(10.0)
    assert np.abs(b['width']).max() < TOL
    assert b.loc[(10.0, 'A'), 'lower'] == pytest.approx(8.0, abs=TOL)
    assert b.loc[(10.0, 'B'), 'lower'] == pytest.approx(2.0, abs=TOL)


def test_distortion_objects(discrete_ab):
    d = discrete_ab.distortion(7.0, 'A', 'upper')
    assert isinstance(d, Distortion)
    assert d.kind == 'bitvar'
    # Degenerate slice at a vertex -> pure TVaR fallback.
    d = discrete_ab.distortion(5.0, 'A', 'lower')
    assert d.kind == 'tvar'


def test_infeasible_premium_raises(discrete_ab):
    with pytest.raises(ValueError, match='feasible premium range'):
        discrete_ab.bounds(4.0)
    with pytest.raises(ValueError, match='feasible premium range'):
        discrete_ab.bounds(10.5)


def test_unknown_unit_raises(discrete_ab):
    with pytest.raises(ValueError, match='missing columns'):
        discrete_ab.port.allocation_bounds(units=['A', 'Z'])


# ---------------------------------------------------------------------------
# Self-consistency audits on a continuous portfolio
# ---------------------------------------------------------------------------

def test_cts_additivity(cts_ab):
    # Unit curves must sum to the total curve up to the exeqa noise floor
    # inherited from density_df (zeroed exeqa below cut_eps + FFT noise).
    assert cts_ab.additivity_error < 1e-4 * cts_ab.premium_range[0]


def test_cts_check_reprices(cts_ab):
    P = cts_ab.premium_range[0] * 1.1
    chk = cts_ab.check(P)
    # Lower bounds avoid the deep tail and reprice essentially exactly;
    # all bounds reprice within the additivity noise.
    assert np.abs(chk['err']).max() < 1e-9 * P
    assert np.abs(chk['total_err']).max() < 1e-9 * P


def test_cts_width_monotone(cts_ab):
    # The allocation image G_P -> A(G_P) widens as P moves away from E[X]
    # (more two-point mixtures qualify) over the moderate premium range.
    lo = cts_ab.premium_range[0]
    Ps = lo * np.array([1.05, 1.1, 1.2, 1.4])
    b = cts_ab.bounds(Ps)
    for u in cts_ab.units:
        w = b.xs(u, level='unit')['width'].to_numpy()
        assert np.all(np.diff(w) > 0)


def test_cts_bounds_bracket_pstar_allocation(cts_ab):
    # The pure TVaR_{p*} allocation is one feasible natural allocation, so
    # it must lie within [lower, upper] for every unit.
    P = cts_ab.premium_range[0] * 1.15
    ps = cts_ab.p_star(P)
    na = cts_ab.na_grid([ps])
    b = cts_ab.bounds(P)
    for u in cts_ab.units:
        v = na[f'exeqa_{u}'].iloc[0]
        assert b.loc[(P, u), 'lower'] <= v + 1e-9
        assert v <= b.loc[(P, u), 'upper'] + 1e-9


def test_cts_pstar_inverts_tvar(cts_ab):
    P = cts_ab.premium_range[0] * 1.25
    ps = cts_ab.p_star(P)
    # Cross-check against the Portfolio's own TVaR function.
    assert cts_ab.port.tvar(ps) == pytest.approx(P, rel=1e-9)
