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

Bounded at assets a = 9 (linear natural allocation)
---------------------------------------------------

Default states X >= 9 = {10} collapse to a single atom at 9 with mass .25
and conditional allocations 9 * (8/10, 2/10) = (7.2, 1.8) (sum = 9).
Vertices:

    m  p     T = E[X^9|X^9>=x_m]   a_A       a_B
    0  0     4.75                  3.8       0.95
    1  .25   19/3                  76/15     19/15
    2  .5    8.5                   7.6       0.9
    3  .75   9                     7.2       1.8

Hulls for A: lower = chord (4.75,3.8)-(9,7.2), slope 0.8 (vertices 1, 2
above it; vertex 1 exactly collinear); upper = (4.75,3.8), (8.5,7.6),
(9,7.2).  For B: lower = (4.75,.95), (8.5,.9), (9,1.8); upper = chord
(4.75,.95)-(9,1.8), slope 0.2 (vertex 1 exactly collinear).  At P = 7:

    lower A = 3.8 + 0.8(7-4.75)        = 5.6
    upper A = 3.8 + (3.8/3.75)(7-4.75) = 6.08
    lower B = 0.95 - (0.05/3.75)(7-4.75) = 0.92
    upper B = 0.95 + 0.2(7-4.75)       = 1.4

The p = 0 vertex (4.75, 3.8, 0.95) must equal density_df
(exa_total, exa_A, exa_B) at loss = 9 — the PIR alpha-S integral.
"""

import numpy as np
import pytest

from aggregate import build
from aggregate.spectral import Distortion


TOL = 1e-12


@pytest.fixture(scope='module')
def small_port():
    """The hand-checkable discrete portfolio."""
    return build("""port SmallAB
        agg A dfreq [1] dsev [0 8]
        agg B dfreq [1] dsev [0 2]
    """, bs=1, log2=8)


@pytest.fixture(scope='module')
def discrete_ab(small_port):
    """Unbounded AllocationBounds on the discrete portfolio."""
    return small_port.allocation_bounds()


@pytest.fixture(scope='module')
def discrete_ab9(small_port):
    """Bounded at assets a = 9: tail {10} collapses to an atom at 9."""
    return small_port.allocation_bounds(a=9)


@pytest.fixture(scope='module')
def cts_port():
    """A continuous two-unit portfolio for self-consistency audits.

    The grid is pinned (``bs``/``log2``) so these audits exercise the
    ``AllocationBounds`` geometry on a *fixed, fine* grid, independent of the
    portfolio auto-sizer -- mirroring the discrete fixtures, which also pin.
    ``test_cts_pstar_inverts_tvar`` cross-checks a TVaR inversion round-trip at
    ``rel=1e-9``; that round-trip is grid-resolution-limited, so it needs the
    finer ``bs=0.0625`` (the value the legacy ``best_bucket`` RMS combine
    happened to pick). The resolution + span ``best_window`` combine
    (1.0.0a49) correctly sizes this book at each unit's natural ``bs=0.125``
    -- coarser, and below what the ``1e-9`` round-trip tolerates -- so the
    pin keeps the audit meaningful rather than coupling it to the sizer.
    """
    return build("""port CtsAB
        agg A 10 claims sev lognorm 10 cv 1.25 poisson
        agg B  4 claims sev gamma 25 cv 0.8 mixed gamma 0.6
    """, bs=0.0625, log2=16)


@pytest.fixture(scope='module')
def cts_ab(cts_port):
    return cts_port.allocation_bounds()


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


# ---------------------------------------------------------------------------
# Bounded totals (asset cap, linear natural allocation)
# ---------------------------------------------------------------------------

def test_bounded_curve_vertices_exact(discrete_ab9):
    cd = discrete_ab9.curve_df
    assert np.allclose(cd.index, [0, .25, .5, .75], atol=TOL)
    assert np.allclose(cd['exeqa_total'], [4.75, 19 / 3, 8.5, 9], atol=TOL)
    assert np.allclose(cd['exeqa_A'], [3.8, 76 / 15, 7.6, 7.2], atol=TOL)
    assert np.allclose(cd['exeqa_B'], [0.95, 19 / 15, 0.9, 1.8], atol=TOL)


def test_bounded_premium_range(discrete_ab9):
    lo, hi = discrete_ab9.premium_range
    assert lo == pytest.approx(4.75, abs=TOL)   # E[X ∧ 9]
    assert hi == pytest.approx(9.0, abs=TOL)    # the cap


def test_bounded_bounds_at_7(discrete_ab9):
    b = discrete_ab9.bounds(7.0)
    assert b.loc[(7.0, 'A'), 'lower'] == pytest.approx(5.6, abs=1e-10)
    assert b.loc[(7.0, 'A'), 'upper'] == pytest.approx(6.08, abs=1e-10)
    assert b.loc[(7.0, 'B'), 'lower'] == pytest.approx(0.92, abs=1e-10)
    assert b.loc[(7.0, 'B'), 'upper'] == pytest.approx(1.4, abs=1e-10)


def test_bounded_collapse_at_cap(discrete_ab9):
    # At P = a only the max distortion qualifies; allocations are the
    # default-state linear shares a * E[X_i/X | X >= a].
    b = discrete_ab9.bounds(9.0)
    assert np.abs(b['width']).max() < TOL
    assert b.loc[(9.0, 'A'), 'lower'] == pytest.approx(7.2, abs=TOL)
    assert b.loc[(9.0, 'B'), 'lower'] == pytest.approx(1.8, abs=TOL)


def test_bounded_p0_vertex_equals_exa(discrete_ab9):
    # Independent cross-check: the p = 0 vertex is the linear-NA expected
    # loss, which add_exa precomputes as exa_i(a) = int_0^a S exi_xgta dx.
    exa = discrete_ab9.port.density_df.loc[9.0, ['exa_total', 'exa_A', 'exa_B']]
    assert np.allclose(discrete_ab9.curve_df.iloc[0], exa, atol=TOL)


def test_bounded_check_reprices(discrete_ab9):
    chk = discrete_ab9.check(7.0)
    assert np.abs(chk['err']).max() < 1e-12
    assert np.abs(chk['total_err']).max() < 1e-12


def test_bounded_additivity(discrete_ab9):
    # Collapsed kappas sum to a by construction; whole curve additive.
    assert discrete_ab9.additivity_error < 1e-12


def test_p_resolves_to_assets(small_port):
    # a via p: q(0.75) = 8, so p=0.75 and a=8 must agree.
    ab_p = small_port.allocation_bounds(p=0.75)
    ab_a = small_port.allocation_bounds(a=8)
    assert ab_p.a == ab_a.a == 8.0
    pd_idx = ab_p.curve_df.index
    assert np.allclose(pd_idx, ab_a.curve_df.index, atol=TOL)
    assert np.allclose(ab_p.curve_df, ab_a.curve_df, atol=TOL)


def test_cap_beyond_ess_sup_matches_unbounded(small_port, discrete_ab):
    # a at the essential sup: the "collapse" is a no-op and the bounded
    # object reproduces the unbounded curve.
    ab10 = small_port.allocation_bounds(a=10)
    assert np.allclose(ab10.curve_df.index, discrete_ab.curve_df.index, atol=1e-12)
    assert np.allclose(ab10.curve_df, discrete_ab.curve_df, atol=1e-12)
    b1, b2 = ab10.bounds(7.0), discrete_ab.bounds(7.0)
    assert np.allclose(b1, b2, atol=1e-12)


def test_cts_bounded_audits(cts_port):
    ab = cts_port.allocation_bounds(p=0.995)
    a = ab.a
    assert ab.premium_range[1] == pytest.approx(a, abs=1e-10)
    P = ab.premium_range[0] * 1.05
    chk = ab.check(P)
    assert np.abs(chk['err']).max() < 1e-9 * P
    assert np.abs(chk['total_err']).max() < 1e-9 * P
    # Bounded additivity: the collapsed atom sums to a exactly; residual
    # only from the body exeqa noise.
    assert ab.additivity_error < 1e-4 * ab.premium_range[0]
