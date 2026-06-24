"""Tests for ``prob_loss_assets`` / ``pla`` and ``price_pentagon_ex``.

``prob_loss_assets`` (plan ``dev/done/plan-pla.md``) turns any one of the three
mutually-determining views ``{p, L, a}`` -- VaR probability, limited expected
loss ``E[min(X, a)]``, asset level -- into the consistent grid-snapped triple.
``price_pentagon_ex`` is the full-power pentagon front door built on it: free
over the capital anchor, gatekept by ``Pentagon.solve``, and warning when an
accounting-determined ``L`` does not reconcile with ``lev(a)``.

These tests pin the round-trip consistency, the ``L``-anchor root-find accuracy
(Newton vs. an independent bisection), the feasibility / arity guards, the far
tail ill-conditioning, and the four ``price_pentagon_ex`` behaviours
(anchor-equivalent silent solves, accounting solves that warn, insoluble
configs, and the uniform post-check). Both ``Aggregate`` and ``Portfolio`` are
covered.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import bisect

from aggregate import build
from aggregate._grid_distribution import ProbLossAssets


@pytest.fixture(scope="module")
def port():
    return build('port PP.Test '
                 'agg A 1 claim sev lognorm 10 cv .3 fixed '
                 'agg B 1 claim sev lognorm 8 cv .2 fixed')


@pytest.fixture(scope="module")
def agg():
    return build('agg PP.Agg 100 claims sev lognorm 10 cv 0.5 poisson')


# ---------------------------------------------------------------- pla: round trips

@pytest.mark.parametrize('obj_name', ['port', 'agg'])
@pytest.mark.parametrize('p0', [0.5, 0.8, 0.95, 0.99])
def test_pla_round_trip(request, obj_name, p0):
    """a -> (p, L, a); then anchoring on that p and that L returns the *same*
    grid-snapped triple."""
    obj = request.getfixturevalue(obj_name)
    a0 = float(obj.q(p0))
    via_a = obj.prob_loss_assets(a=a0)
    via_p = obj.prob_loss_assets(p=via_a.p)
    via_L = obj.prob_loss_assets(L=via_a.L)
    assert isinstance(via_a, ProbLossAssets)
    assert np.isclose(via_a.a, via_p.a) and np.isclose(via_a.a, via_L.a)
    assert np.isclose(via_a.p, via_p.p) and np.isclose(via_a.p, via_L.p)
    assert np.isclose(via_a.L, via_p.L) and np.isclose(via_a.L, via_L.L)


def test_pla_is_grid_consistent(agg):
    """The returned triple satisfies L == lev(a) and p == cdf(a) exactly."""
    gd = agg._grid_distribution()
    t = agg.prob_loss_assets(p=0.9)
    assert np.isclose(t.L, gd.lev(t.a))
    assert np.isclose(t.p, float(gd.cdf(t.a)))


def test_pla_matches_exa_column(agg):
    """L from pla equals the stored ``exa`` (the add_exa / E[min(X,a)] datum)."""
    t = agg.prob_loss_assets(p=0.95)
    assert np.isclose(t.L, agg.density_df.loc[t.a, 'exa'])


def test_pla_alias(agg):
    """``pla`` is the same callable as ``prob_loss_assets``."""
    assert agg.pla(p=0.9) == agg.prob_loss_assets(p=0.9)


# ---------------------------------------------------------------- pla: L-anchor accuracy

@pytest.mark.parametrize('p0', [0.6, 0.9, 0.99])
def test_pla_L_anchor_matches_bisection(agg, p0):
    """The safeguarded-Newton solve agrees with an independent bisection on the
    discrete ``lev``, and reproduces L to bucket tolerance."""
    gd = agg._grid_distribution()
    a0 = float(gd.q(p0))
    L = gd.lev(a0)
    res = gd.prob_loss_assets(L=L)
    # independent bisection on lev(a) - L, then snap
    a_bis = bisect(lambda aa: gd.lev(aa) - L, 0.0, float(gd.q(1)))
    a_bis = float(gd.snap(a_bis))
    assert np.isclose(res.a, a_bis, atol=gd.bs)
    assert abs(gd.lev(res.a) - L) <= gd.bs


# ---------------------------------------------------------------- pla: guards

def test_pla_L_above_mean_raises(agg):
    gd = agg._grid_distribution()
    with pytest.raises(ValueError, match='infeasible'):
        gd.prob_loss_assets(L=gd.mean() * 1.01)


@pytest.mark.parametrize('kwargs', [
    {},                         # zero anchors
    {'p': 0.9, 'a': 100.0},     # two anchors
    {'p': 0.9, 'L': 5.0, 'a': 100.0},  # three anchors
])
def test_pla_arity_guard(agg, kwargs):
    with pytest.raises(ValueError, match='exactly one'):
        agg.prob_loss_assets(**kwargs)


def test_pla_ill_conditioned_far_tail(agg):
    """L very close to E[X] (far tail, S(a) -> 0): the solve must not blow up --
    the bisection fallback carries it -- and the returned L stays <= mean."""
    gd = agg._grid_distribution()
    m = gd.mean()
    L = gd.lev(float(gd.q(0.999)))   # deep in the tail but strictly feasible
    assert L < m
    res = gd.prob_loss_assets(L=L)
    assert np.isfinite(res.a) and res.a <= gd.q(1)
    assert abs(gd.lev(res.a) - L) <= gd.bs


# ---------------------------------------------------------------- price_pentagon_ex

PENTAGON_STATS = ['L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE']


def _octet(df):
    """The eight pentagon stats as floats (drops the leading ``p`` descriptor)."""
    return df[PENTAGON_STATS].iloc[0].to_numpy(dtype=float)


@pytest.mark.parametrize('obj_name', ['port', 'agg'])
def test_ex_anchor_equivalence_silent(request, obj_name):
    """Anchoring on L, the equivalent a, and the equivalent p give the same
    octet, the returned L equals lev(a), and no warning fires."""
    obj = request.getfixturevalue(obj_name)
    gd = obj._grid_distribution()
    p0, r = 0.99, 0.1
    a0 = float(obj.q(p0))
    L0 = gd.lev(a0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # any warning becomes an error
        via_a = obj.price_pentagon_ex(a=a0, ROE=r)
        via_p = obj.price_pentagon_ex(p=p0, ROE=r)
        via_L = obj.price_pentagon_ex(L=L0, ROE=r)
    assert np.allclose(_octet(via_a), _octet(via_p))
    assert np.allclose(_octet(via_a), _octet(via_L))
    assert np.isclose(via_a['L'].iloc[0], gd.lev(via_a['a'].iloc[0]))


@pytest.mark.parametrize('obj_name', ['port', 'agg'])
def test_ex_reports_p_and_columns(request, obj_name):
    """Output is one 'total' row with a leading ``p`` then the eight stats, and
    ``p`` reconciles with ``cdf(a)``."""
    obj = request.getfixturevalue(obj_name)
    gd = obj._grid_distribution()
    df = obj.price_pentagon_ex(p=0.99, ROE=0.1)
    assert list(df.columns) == ['p'] + PENTAGON_STATS
    assert df.index.tolist() == ['total'] and df.index.name == 'unit'
    assert np.isclose(df['p'].iloc[0], float(gd.cdf(df['a'].iloc[0])))


def test_ex_accounting_solve_warns(port):
    """{P, M, ROE} (no p/a/L) solves by accounting, deduces a = P + M/ROE,
    reports p = cdf(a), and warns when L = P - M disagrees with lev(a)."""
    with pytest.warns(UserWarning, match='accounting identities'):
        df = port.price_pentagon_ex(P=12.0, M=2.0, ROE=0.1)
    a_sol = df['a'].iloc[0]
    assert np.isclose(a_sol, 12.0 + 2.0 / 0.1)             # P + M/ROE
    assert np.isclose(df['p'].iloc[0], float(port._grid_distribution().cdf(a_sol)))


def test_ex_accounting_solve_consistent_silent(port):
    """Construct {P, M, ROE} whose P - M happens to equal lev(a_solved): the
    same accounting path is then silent (the post-check is on L vs lev(a))."""
    gd = port._grid_distribution()
    r, M_ = 0.1, 1.0
    # a is fixed by accounting at a = P + M/r = (L + M) + M/r with L = P - M;
    # solve a = lev(a) self-consistently for L so that P - M == lev(a).
    # iterate: pick a, L = lev(a), P = L + M, check a == P + M/r.
    # easier: choose a on the grid, set L = lev(a), M, then P = L + M and ROE so
    # that Q = M/ROE = a - P  ==>  ROE = M / (a - P).
    a0 = float(port.q(0.99))
    L0 = gd.lev(a0)
    P_ = L0 + M_
    Q_ = a0 - P_
    roe = M_ / Q_
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        df = port.price_pentagon_ex(P=P_, M=M_, ROE=roe)
    assert np.isclose(df['a'].iloc[0], a0)
    assert np.isclose(df['L'].iloc[0], L0)


def test_ex_post_check_over_determined(port):
    """{a, P, M} with P - M == lev(a) is silent; the same with P - M != lev(a)
    warns -- the check is on L_solved vs lev(a_solved), not on input mode."""
    gd = port._grid_distribution()
    a0 = float(port.q(0.99))
    lev_a = gd.lev(a0)
    M_ = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        df = port.price_pentagon_ex(a=a0, P=lev_a + M_, M=M_)
    assert np.isclose(df['L'].iloc[0], lev_a)
    with pytest.warns(UserWarning, match='accounting identities'):
        port.price_pentagon_ex(a=a0, P=lev_a + M_ + 5.0, M=M_)


def test_ex_insoluble_config_raises(port):
    """Three scale-free ratios {PQ, ROE, LR} cannot pin the level: Pentagon.solve
    raises and _ex propagates it cleanly."""
    with pytest.raises(ValueError, match='Insoluble'):
        port.price_pentagon_ex(PQ=1.5, ROE=0.1, LR=0.7)


def test_ex_rejects_p_and_a(port):
    with pytest.raises(ValueError, match='at most one of p='):
        port.price_pentagon_ex(p=0.99, a=100.0, ROE=0.1)


def test_ex_matches_price_pentagon_octet(port):
    """On a shared anchor+target, _ex reproduces the price_pentagon octet (both
    read the limited expected loss off the same grid)."""
    p0, r = 0.99, 0.1
    ex = port.price_pentagon_ex(p=p0, ROE=r)
    base = port.price_pentagon(p=p0, ROE=r)
    assert np.allclose(_octet(ex), base[PENTAGON_STATS].iloc[0].to_numpy(dtype=float))
