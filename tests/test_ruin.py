"""Eventual-ruin solvers: PK Poisson guard, ``wiener_hopf``, pedagogy dispatch.

The Poisson guard on ``pollaczeck_khinchine``, the cepstral Wiener-Hopf
solver ``Aggregate.wiener_hopf`` for renewal (wait-clause) frequencies, the
shared ``RuinFunction`` named tuple, the ``pedagogy._ruin_function``
dispatch and the ``pedagogy.ruin_example`` figure builder. DecL programs
mirrored in ``src/aggregate/agg/decl-testers.agg`` (AD.*).
See dev/done/plan-ruin-wiener-hopf.md [Ruin-Wiener-Hopf].
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aggregate import build
from aggregate._aggregate import RuinFunction


@pytest.fixture(autouse=True)
def _quiet():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


# ---------------------------------------------------------------------------
# shared builds -- module scope keeps the suite fast; the two fixtures are
# the SAME model: 10 claims poisson == 10 years wait expon
# (see tests/test_renewal_agg.py::test_three_way_poisson_equivalence)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def po_agg():
    return build('agg RuPo 10 claims sev gamma 2 poisson', bs=1 / 16, log2=12)


@pytest.fixture(scope='module')
def re_agg():
    return build('agg RuRe 10 years sev gamma 2 wait expon',
                 bs=1 / 16, log2=12)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_pk_guard(po_agg):
    assert po_agg.pollaczeck_khinchine(0.2) is not None
    for prog in ('agg RuFix 10 claims sev gamma 2 fixed',
                 'agg RuBin 10 claims sev gamma 2 binomial 0.5'):
        a = build(prog, bs=1 / 16, log2=10)
        with pytest.raises(ValueError, match='assumes a Poisson'):
            a.pollaczeck_khinchine(0.2)


def test_pk_guard_points_to_wh(re_agg):
    with pytest.raises(ValueError, match='wiener_hopf'):
        re_agg.pollaczeck_khinchine(0.2)


def test_wh_guard_poisson(po_agg):
    with pytest.raises(ValueError, match='pollaczeck_khinchine'):
        po_agg.wiener_hopf(0.2)


def test_wh_guard_defective():
    a = build('agg RuDef 2 years dsev [1] dwait [1 2] [.4 .5] !')
    with pytest.raises(ValueError, match='defective'):
        a.wiener_hopf(0.2)


def test_wh_guard_log2(re_agg):
    with pytest.raises(ValueError, match='truncate'):
        re_agg.wiener_hopf(0.2, log2=re_agg.log2 - 1)


def test_wh_net_profit_violation(re_agg):
    with pytest.raises(ValueError, match='net profit'):
        re_agg.wiener_hopf(-0.2)


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------

def test_wh_matches_pk_exponential_waits(po_agg, re_agg):
    # exponential waits => compound Poisson: WH and PK price the same
    # model on the same bs = 1/16 grid. PK's integrated-distribution
    # build is O(bs)-biased, WH's rounding O(bs^2); observed max gap
    # 6.7e-3 here, 1.7e-3 at bs = 1/64 (converging), so abs = 1e-2.
    rho = 0.2
    pk = po_agg.pollaczeck_khinchine(rho)
    wh = re_agg.wiener_hopf(rho)
    assert np.array_equal(pk.ruin.index, wh.ruin.index)
    for u in [0.0, 4.0, 8.0, 16.0, 32.0]:
        assert wh.ruin.loc[u] == pytest.approx(pk.ruin.loc[u], abs=1e-2)
    # psi(0) = 1/(1+rho) exactly in the continuum (Poisson magic);
    # observed grid errors 2.2e-3 (WH) and 4.5e-3 (PK)
    assert wh.ruin.iloc[0] == pytest.approx(1 / (1 + rho), abs=5e-3)
    assert pk.ruin.iloc[0] == pytest.approx(1 / (1 + rho), abs=1e-2)


def test_wh_exponential_severity_closed_form():
    # Exponential severity under ANY wait law has exponential ladder
    # heights, so psi(u) = psi(0) exp(-R u) with R = beta (1 - psi(0)),
    # beta = 1/mean severity -- a genuinely non-Poisson exact benchmark
    # (gamma-2 waits here). On the grid the ladder heights are exactly
    # geometric, so log-linearity holds to float precision (observed
    # 4e-13); the slope matches the continuum R to O(bs) (observed
    # 8.7e-3 at bs = 1/4).
    a = build('agg RuExp 5 years sev 10 * expon wait gamma 2',
              bs=1 / 4, log2=12)
    wh = a.wiener_hopf(0.3)
    psi = wh.ruin
    beta = 1 / 10
    R = beta * (1 - psi.iloc[0])
    u1, u2, u3 = 20.0, 60.0, 100.0
    s12 = -(np.log(psi.loc[u2]) - np.log(psi.loc[u1])) / (u2 - u1)
    s23 = -(np.log(psi.loc[u3]) - np.log(psi.loc[u2])) / (u3 - u2)
    assert s12 == pytest.approx(s23, rel=1e-9)
    assert s12 == pytest.approx(R, rel=2e-2)


def test_wh_mixture_wait():
    # hyperexponential (mixture) wait exercises the multi-component path
    # of _discretize_wait_pmf; exponential severity keeps the Lundberg
    # closed form exact regardless of the wait law (observed: log-linear
    # to 2e-11, slope vs R to 9.6e-3 at bs = 1/4)
    a = build('agg RuMix 5 years sev 10 * expon '
              'wait [.6 .4] * expon wts [.5 .5]', bs=1 / 4, log2=12)
    wh = a.wiener_hopf(0.3)
    psi = wh.ruin
    R = (1 / 10) * (1 - psi.iloc[0])
    u1, u2, u3 = 20.0, 60.0, 100.0
    s12 = -(np.log(psi.loc[u2]) - np.log(psi.loc[u1])) / (u2 - u1)
    s23 = -(np.log(psi.loc[u3]) - np.log(psi.loc[u2])) / (u3 - u2)
    assert s12 == pytest.approx(s23, rel=1e-9)
    assert s12 == pytest.approx(R, rel=2e-2)


def test_wh_wider_grid(re_agg):
    # log2 override: wider circle, same bs; the shared prefix agrees
    wh1 = re_agg.wiener_hopf(0.2)
    wh2 = re_agg.wiener_hopf(0.2, log2=re_agg.log2 + 1)
    assert len(wh2.ruin) == 2 * len(wh1.ruin)
    n = len(wh1.ruin)
    assert np.allclose(wh1.ruin.to_numpy()[:n // 2],
                       wh2.ruin.to_numpy()[:n // 2], atol=1e-9)


# ---------------------------------------------------------------------------
# return contract
# ---------------------------------------------------------------------------

def test_ruin_function_named_tuple(po_agg, re_agg):
    pk = po_agg.pollaczeck_khinchine(0.2)
    wh = re_agg.wiener_hopf(0.2)
    for rf in (pk, wh):
        assert isinstance(rf, RuinFunction)
        ruin, find_u, mean, density = rf          # positional unpack
        assert ruin is rf.ruin and find_u is rf.find_u
        assert mean == rf.mean and density is rf.density
        assert callable(rf.find_u)
        u = rf.find_u(0.05)
        assert rf.ruin.index[0] <= u <= rf.ruin.index[-1]
        assert rf.mean == pytest.approx(2.0, abs=1e-6)   # gamma-2 severity
    # interpolate kind returns off-grid capital
    wh_i = re_agg.wiener_hopf(0.2, kind='interpolate')
    assert wh_i.find_u(0.05) == pytest.approx(wh.find_u(0.05), abs=re_agg.bs)


# ---------------------------------------------------------------------------
# pedagogy
# ---------------------------------------------------------------------------

def test_pedagogy_dispatch(po_agg, re_agg):
    from aggregate.pedagogy import _ruin_function
    pk = _ruin_function(po_agg, 0.2, kind='index')
    wh = _ruin_function(re_agg, 0.2, kind='index')
    assert isinstance(pk, RuinFunction) and isinstance(wh, RuinFunction)
    a = build('agg RuFix 10 claims sev gamma 2 fixed', bs=1 / 16, log2=10)
    with pytest.raises(ValueError, match='fixed'):
        _ruin_function(a, 0.2, kind='index')


def test_ruin_example_smoke(re_agg):
    from aggregate.pedagogy import ruin_example
    summary, fig = ruin_example(re_agg, 0.2, 8.0, n_sims=2000, n_plot=10,
                                seed=42)
    try:
        v = summary['value']
        assert v['frequency kind'] == 'renewal'
        for key in ('premium rate c', 'psi(0) exact', 'psi(u0) exact',
                    'psi(u0) simulated', 'sim std error', 'LIL variance '
                    'rate sigma2', 'mass beyond grid (psi at top)'):
            assert key in v.index
        # simulation runs the SAME discretized model as the exact solver:
        # agreement within 4 standard errors
        assert abs(v['psi(u0) simulated'] - v['psi(u0) exact']) \
            <= 4 * v['sim std error']
    finally:
        plt.close(fig)


def test_ruin_example_guards(po_agg, re_agg):
    from aggregate.pedagogy import ruin_example
    with pytest.raises(ValueError, match='rho > 0'):
        ruin_example(re_agg, -0.1, 5.0)
    with pytest.raises(ValueError, match='renewal'):
        ruin_example(po_agg, 0.2, 5.0, log2=14)
    with pytest.raises(ValueError, match='beyond the represented'):
        ruin_example(re_agg, 0.2, 1e9)
