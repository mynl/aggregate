"""
Tests for ``aggregate.bounds.PricingBounds`` / ``Portfolio.pricing_bounds``.

Three layers:

1. A tiny discrete cross-pricing case (X and Y supplied as exact pmf
   ``Series`` so there is no FFT noise) where the union-of-breakpoints vertex
   table, hulls, bounds, achieving biTVaRs, and ``p_star`` are computed by
   hand and asserted exactly.
2. Structural invariants: ``Y = X`` collapses the range to ``[P, P]``; the
   uniform reference exposes the Gini mean-Kusuoka level.
3. Self-consistency audits on continuous risks (bounded, the
   well-conditioned regime): ``check`` reprices Y from first principles, the
   achieving distortion prices X to P, and the feasible range is sensible.

Hand calculation for the discrete case
--------------------------------------

Reference ``X`` has outcomes ``{0, 8}`` each probability 1/2, so
``TVaR_p(X) = 4/(1-p)`` for ``p < 1/2`` and ``8`` for ``p >= 1/2``; its only
interior breakpoint is ``p = 1/2``.

Target ``Y`` has outcomes ``{0, 3}`` with probabilities ``{1/4, 3/4}``, so
``TVaR_0(Y) = 2.25``, ``TVaR_p(Y) = 3`` for ``p >= 1/4``; its breakpoint is
``p = 1/4``.

Union breakpoints ``{1/4, 1/2}`` plus ``p = 0`` give vertices

    p      T_X = TVaR_p(X)      T_Y = TVaR_p(Y)
    0      4                    2.25
    1/4    16/3 = 5.3333...     3
    1/2    8                    3

The middle vertex ``(16/3, 3)`` lies *above* the chord ``(4, 2.25)-(8, 3)``
(which passes through ``(16/3, 2.5)``), so:

- lower hull = the chord ``(4, 2.25)-(8, 3)`` (slope 3/16);
- upper hull = ``(4, 2.25), (16/3, 3), (8, 3)``.

At ``P = 6`` (``T_X = 6``):

    lower Y = 2.25 + (3/16)(6 - 4) = 2.625   biTVaR (0, 1/2),  w1 = 1/2
    upper Y = 3                              biTVaR (1/4, 1/2), w1 = 1/4

``p_star(6)``: ``4/(1-p) = 6`` gives ``p = 1/3``.
"""

import numpy as np
import pandas as pd
import pytest

from aggregate import build
from aggregate.bounds import PricingBounds, uniform_source


TOL = 1e-12


@pytest.fixture(scope='module')
def discrete_pb():
    """Exact discrete cross-pricing (pmf Series, no FFT noise)."""
    X = pd.Series([0.5, 0.5], index=[0.0, 8.0])
    Y = pd.Series([0.25, 0.75], index=[0.0, 3.0])
    return PricingBounds(X, {'Y': Y})


@pytest.fixture(scope='module')
def cts_port():
    """A continuous portfolio used as the reference risk X."""
    return build("""port PBRef
        agg A 10 claims sev lognorm 10 cv 1.25 poisson
        agg B  4 claims sev gamma 25 cv 0.8 mixed gamma 0.6
    """)


# ---------------------------------------------------------------------------
# Exact hand-checked discrete cross-pricing
# ---------------------------------------------------------------------------

def test_curve_vertices_exact(discrete_pb):
    cd = discrete_pb.curve_df
    assert np.allclose(cd.index, [0, 0.25, 0.5], atol=TOL)
    assert np.allclose(cd['tvar_X'], [4, 16 / 3, 8], atol=TOL)
    assert np.allclose(cd['price_Y'], [2.25, 3, 3], atol=TOL)


def test_premium_range(discrete_pb):
    lo, hi = discrete_pb.premium_range
    assert lo == pytest.approx(4.0, abs=TOL)     # E[X]
    assert hi == pytest.approx(8.0, abs=TOL)     # ess sup X


def test_bounds_at_6(discrete_pb):
    b = discrete_pb.bounds(6.0)
    assert b.loc[(6.0, 'Y'), 'lower'] == pytest.approx(2.625, abs=TOL)
    assert b.loc[(6.0, 'Y'), 'upper'] == pytest.approx(3.0, abs=TOL)
    assert b.loc[(6.0, 'Y'), 'width'] == pytest.approx(0.375, abs=TOL)


def test_bitvars_at_6(discrete_pb):
    bv = discrete_pb.bitvars(6.0)
    r = bv.loc[(6.0, 'Y', 'lower')]
    assert (r['p0'], r['p1']) == (0.0, 0.5)
    assert r['w1'] == pytest.approx(0.5, abs=TOL)
    r = bv.loc[(6.0, 'Y', 'upper')]
    assert (r['p0'], r['p1']) == (0.25, 0.5)
    assert r['w1'] == pytest.approx(0.25, abs=TOL)


def test_p_star_exact(discrete_pb):
    assert discrete_pb.p_star(6.0) == pytest.approx(1 / 3, abs=TOL)
    assert discrete_pb.p_star(8.0) == pytest.approx(0.5, abs=TOL)
    assert discrete_pb.p_star(4.0) == pytest.approx(0.0, abs=TOL)


def test_check_reprices_exactly(discrete_pb):
    chk = discrete_pb.check(6.0)
    assert np.abs(chk['err']).max() < TOL
    assert np.abs(chk['total_err']).max() < TOL


def test_endpoint_collapses_to_point(discrete_pb):
    # At P = E[X] only the mean distortion qualifies (price Y = E[Y]); at
    # P = ess sup X only the max distortion (price Y = TVaR_1(Y)).
    b = discrete_pb.bounds(4.0)
    assert b.loc[(4.0, 'Y'), 'width'] < TOL
    assert b.loc[(4.0, 'Y'), 'lower'] == pytest.approx(2.25, abs=TOL)
    b = discrete_pb.bounds(8.0)
    assert b.loc[(8.0, 'Y'), 'width'] < TOL
    assert b.loc[(8.0, 'Y'), 'lower'] == pytest.approx(3.0, abs=TOL)


def test_infeasible_premium_raises(discrete_pb):
    with pytest.raises(ValueError, match='feasible premium range'):
        discrete_pb.bounds(3.0)
    with pytest.raises(ValueError, match='feasible premium range'):
        discrete_pb.bounds(8.5)


def test_multiple_y_and_names(discrete_pb):
    # Several Y sources priced at once; explicit dict names are preserved.
    Y2 = pd.Series([0.5, 0.5], index=[0.0, 10.0])
    pb = PricingBounds(pd.Series([0.5, 0.5], index=[0.0, 8.0]),
                       {'Y': pd.Series([0.25, 0.75], index=[0.0, 3.0]),
                        'Z': Y2})
    b = pb.bounds(6.0)
    assert set(b.index.get_level_values('risk')) == {'Y', 'Z'}


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

def test_y_equals_x_collapses(cts_port):
    # Every distortion pricing X to P prices X (= Y) to P: range is [P, P].
    pb = cts_port.pricing_bounds({'Xcopy': cts_port}, p=0.99)
    P = pb.premium_range[0] * 1.15
    b = pb.bounds(P)
    assert b.loc[(P, 'Xcopy'), 'lower'] == pytest.approx(P, rel=1e-9)
    assert b.loc[(P, 'Xcopy'), 'upper'] == pytest.approx(P, rel=1e-9)
    assert b.loc[(P, 'Xcopy'), 'width'] < 1e-7 * P


def test_uniform_gini_level():
    Y = build('agg Y 5 claims sev lognorm 20 cv 0.6 poisson')
    pb = PricingBounds('uniform', {'Y': Y}, a=Y.q(0.99))
    # P is the price of U[0,1] in [0.5, 1]; the mean Kusuoka level is 2P-1.
    assert pb.premium_range[0] == pytest.approx(0.5, abs=1e-9)
    assert pb.mean_kusuoka_level(0.7) == pytest.approx(0.4, abs=TOL)
    b = pb.bounds(0.7)
    # Non-degenerate dispersion reading: lower < upper.
    assert b.loc[(0.7, 'Y'), 'width'] > 0
    # gini_level is reported at the left end of the feasible range.
    assert pb.gini_level == pytest.approx(0.0, abs=1e-9)


def test_uniform_source_roundtrip():
    u = uniform_source()
    p = np.array([0.0, 0.3, 1.0])
    assert np.allclose(u.tvar(p), (1 + p) / 2, atol=TOL)
    assert u.tvar_inv(0.75) == pytest.approx(0.5, abs=TOL)


# ---------------------------------------------------------------------------
# Continuous, bounded self-consistency audits (the exact regime)
# ---------------------------------------------------------------------------

def test_bounded_check_reprices(cts_port):
    pb = cts_port.pricing_bounds({'A': cts_port.A, 'B': cts_port.B}, p=0.99)
    P = pb.premium_range[0] * 1.1
    chk = pb.check(P)
    assert np.abs(chk['err']).max() < 1e-9 * P
    assert np.abs(chk['total_err']).max() < 1e-9 * P


def test_bounded_distortion_prices_x(cts_port):
    # The achieving biTVaR must price the reference X to P (independent of the
    # Y it bounds): its total repricing equals P.
    pb = cts_port.pricing_bounds({'A': cts_port.A}, p=0.99)
    P = pb.premium_range[0] * 1.2
    chk = pb.check(P)
    assert np.abs(chk['total_err']).max() < 1e-9 * P
    d = pb.distortion(P, 'A', 'upper')
    assert d.kind in ('bitvar', 'tvar')


def test_bounded_width_monotone(cts_port):
    # Wider as P moves away from E[X] (more mixtures qualify).
    pb = cts_port.pricing_bounds({'A': cts_port.A}, p=0.99)
    lo = pb.premium_range[0]
    Ps = lo * np.array([1.02, 1.05, 1.1, 1.2])
    w = pb.bounds(Ps).xs('A', level='risk')['width'].to_numpy()
    assert np.all(np.diff(w) > 0)


def test_cap_resolves_via_p_and_a(cts_port):
    # a via p and a via direct a must agree.
    a = cts_port.q(0.99)
    pb_p = cts_port.pricing_bounds({'A': cts_port.A}, p=0.99)
    pb_a = cts_port.pricing_bounds({'A': cts_port.A}, a=a)
    assert pb_p.a == pb_a.a
    assert np.allclose(pb_p.curve_df.index, pb_a.curve_df.index, atol=TOL)
    assert np.allclose(pb_p.bounds(pb_p.premium_range[0] * 1.1),
                       pb_a.bounds(pb_a.premium_range[0] * 1.1), atol=1e-9)


def test_pstar_inverts_reference_tvar(cts_port):
    pb = cts_port.pricing_bounds({'A': cts_port.A}, p=0.99)
    P = pb.premium_range[0] * 1.15
    ps = pb.p_star(P)
    assert 0.0 < ps < 1.0
    # p_star inverts the (capped) reference TVaR: TVaR_{p_star}(X ∧ a) == P.
    tvar_at = float(pb._x_source.tvar(np.array([ps]))[0])
    assert tvar_at == pytest.approx(P, rel=1e-7)
