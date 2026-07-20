"""Stage-0 unit tests for the Sparre-Andersen renewal kernel (`_renewal.py`).

Pure-function tests against closed-form renewal benchmarks -- no DecL, no
``build()``. See dev/plan-sparre-a.md [Renewal-Frequency-Wait-Clause].
"""

from math import factorial

import numpy as np
import pytest
import scipy.special as sp
import scipy.stats as ss

from aggregate import Severity
from aggregate._renewal import (geometric_batch_compose, renewal_count_pmf,
                                wait_count_pmf, wait_grid)


def _round_discretize(fz, h, m, T):
    """Rounding-discretize a scipy frozen rv onto arange(m)*h, truncated at T."""
    n1 = int(round(T / h))
    xs = np.arange(n1 + 1) * h
    pm = np.zeros(m)
    pm[:n1 + 1] = np.diff(fz.cdf(np.hstack((-np.inf, xs + h / 2))))
    return pm


def _one(sev):
    """Single-component wait: the (components, weights) pair."""
    return [(sev, 0.0, np.inf, True)], [1.0]


# ---------------------------------------------------------------------------
# renewal_count_pmf -- kernel level, hand-discretized pm
# ---------------------------------------------------------------------------

def test_kernel_expon_poisson():
    # W ~ expon(1) => N(T) ~ Poisson(T)
    T, log2 = 3.0, 13
    m = 1 << log2
    h = T / int(0.75 * m)
    pm = _round_discretize(ss.expon(), h, m, T)
    k, pN = renewal_count_pmf(pm, h, T)
    assert np.abs(pN - ss.poisson(T).pmf(k)).max() < 2e-8


def test_kernel_richardson_h2():
    # halving h shrinks the error ~4x (O(h^2) with the half-bucket readout)
    T = 3.0
    errs = []
    for log2 in (12, 13):
        m = 1 << log2
        h = T / int(0.75 * m)
        pm = _round_discretize(ss.expon(), h, m, T)
        k, pN = renewal_count_pmf(pm, h, T)
        errs.append(np.abs(pN - ss.poisson(T).pmf(k)).max())
    assert errs[0] / errs[1] > 2.5


def test_kernel_invgauss_closed_form():
    # IG is closed under convolution: S_k ~ IG(k*mu, k^2*lam), so
    # P(N >= k) = invgauss(mu/(k*lam), scale=k^2*lam).cdf(T)
    mu, lam, T, log2 = 0.5, 1.5, 2.0, 14
    m = 1 << log2
    h = T / int(0.75 * m)
    pm = _round_discretize(ss.invgauss(mu / lam, scale=lam), h, m, T)
    k, pN = renewal_count_pmf(pm, h, T)

    def sk_cdf(kk):
        return ss.invgauss(mu / (kk * lam), scale=kk * kk * lam).cdf(T)

    sf = np.array([1.0] + [sk_cdf(kk) for kk in k[1:]])
    exact = -np.diff(np.append(sf, sk_cdf(len(k))))
    assert np.abs(pN - exact).max() < 2e-8


def test_kernel_deterministic_lattice_boundary():
    # W == 1, T = 3: arrivals at 1, 2, 3; the atom exactly at T counts
    pm = np.zeros(8)
    pm[1] = 1.0
    k, pN = renewal_count_pmf(pm, 1.0, 3.0, lattice=True, kmax=6)
    assert pN[3] == pytest.approx(1.0, abs=1e-12)
    assert np.delete(pN, 3).max() < 1e-12


def test_kernel_defective_terminating():
    # defective expon: each draw survives w.p. q, so
    # P(N >= k) = q^k * gammainc(k, T); pmf still proper
    q, T, log2 = 0.6, 2.0, 13
    m = 1 << log2
    h = T / int(0.75 * m)
    pm = q * _round_discretize(ss.expon(), h, m, T)
    k, pN = renewal_count_pmf(pm, h, T)
    sf = np.array([1.0] + [q ** kk * sp.gammainc(kk, T) for kk in k[1:]])
    exact = -np.diff(np.append(
        sf, q ** len(k) * sp.gammainc(len(k), T)))
    assert np.abs(pN - exact).max() < 2e-8
    assert pN.sum() == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# geometric_batch_compose
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('p0', [0.3, 0.5])
def test_batch_compose_enumeration(p0):
    # W in {0, 1}, T integer: S_k = #ones among first k, so the brute force
    # is P(N >= k) = P(Binomial(k, 1-p0) <= T); M == T a.s.
    T = 2
    pmf_M = np.zeros(T + 1)
    pmf_M[T] = 1.0
    pN = geometric_batch_compose(pmf_M, p0)
    sf = np.array([1.0] + [ss.binom(kk, 1 - p0).cdf(T)
                           for kk in range(1, len(pN) + 1)])
    exact = -np.diff(sf)
    assert np.abs(pN[:len(exact)] - exact).max() < 1e-13
    # closed form: P(N=n) = C(n, T) (1-p0)^(T+1) p0^(n-T)
    n = np.arange(len(pN))
    closed = sp.comb(n, T) * (1 - p0) ** (T + 1) * p0 ** (n - T)
    closed[:T] = 0.0
    assert np.abs(pN - closed).max() < 1e-13


def test_batch_compose_p0_zero_identity():
    pmf_M = np.array([0.1, 0.4, 0.5])
    assert np.array_equal(geometric_batch_compose(pmf_M, 0.0), pmf_M)


def test_batch_compose_rejects_bad_p0():
    with pytest.raises(ValueError):
        geometric_batch_compose(np.array([1.0]), 1.0)
    with pytest.raises(ValueError):
        geometric_batch_compose(np.array([1.0]), -0.1)


# ---------------------------------------------------------------------------
# wait_count_pmf -- orchestrator with real Severity components
# ---------------------------------------------------------------------------

def test_orchestrator_expon_poisson():
    k, pN, info = wait_count_pmf(*_one(Severity('expon', sev_scale=1.0)), 3.0)
    assert np.abs(pN - ss.poisson(3.0).pmf(k)).max() < 1e-8
    assert not info['lattice']
    assert info['p0'] == 0.0


def test_orchestrator_gamma_closed_form():
    # gamma(a=2, rate 1) waits: P(N >= k) = gammainc(2k, T)
    T = 5.0
    k, pN, info = wait_count_pmf(
        *_one(Severity('gamma', sev_a=2.0, sev_scale=1.0)), T)
    sf = np.array([1.0] + [sp.gammainc(2 * kk, T) for kk in k[1:]])
    exact = -np.diff(np.append(sf, sp.gammainc(2 * (k[-1] + 1), T)))
    assert np.abs(pN - exact).max() < 1e-8


def test_orchestrator_uniform_factorial():
    # uniform(0,1) waits, T=1: P(S_k <= 1) = 1/k! so P(N=k) = k/(k+1)!.
    # The uniform density JUMPS exactly at T=1, breaking the smoothness
    # assumption of the half-bucket endpoint correction for k <= 1 only
    # (f^{*1} discontinuous at T): those two entries carry an O(h/4)
    # artifact; all higher convolutions are smooth enough for ~1e-8.
    k, pN, info = wait_count_pmf(*_one(Severity('uniform', sev_scale=1.0)), 1.0)
    exact = np.array([kk / factorial(kk + 1) for kk in k])
    assert np.abs(pN - exact).max() < 5e-6
    assert np.abs(pN[2:] - exact[2:]).max() < 1e-8


def test_orchestrator_deterministic_dwait():
    # dwait [1], T=3 => N == 3 on the exact lattice
    # tail junk is bounded by the kernel's designed noise floor: the tilt
    # tilt_total=20 is the float64 optimum of exp(-t) wrap leakage vs
    # eps*exp(0.75 t) dot-product noise, floor ~ exp(-20) ~ 2e-9
    k, pN, info = wait_count_pmf(
        *_one(Severity('dhistogram', sev_xs=[1], sev_ps=[1])), 3.0)
    assert info['lattice']
    assert pN[3] == pytest.approx(1.0, abs=1e-8)
    assert np.delete(pN, 3).max() < 1e-8


def test_orchestrator_two_point_lattice():
    # W in {1, 2} each 1/2, T=2: N=2 iff W1=W2=1 (p 1/4), else N=1
    k, pN, info = wait_count_pmf(
        *_one(Severity('dhistogram', sev_xs=[1, 2], sev_ps=[.5, .5])), 2.0)
    assert info['lattice']
    assert pN[1] == pytest.approx(0.75, abs=1e-9)
    assert pN[2] == pytest.approx(0.25, abs=1e-9)


def test_orchestrator_mixture_lattice():
    # 50/50 mixture of deterministic waits 1 and 2 == dwait [1 2] [.5 .5]
    comps = [(Severity('dhistogram', sev_xs=[1], sev_ps=[1]), 0.0, np.inf, True),
             (Severity('dhistogram', sev_xs=[2], sev_ps=[1]), 0.0, np.inf, True)]
    k, pN, info = wait_count_pmf(comps, [0.5, 0.5], 2.0)
    assert info['lattice']
    assert pN[1] == pytest.approx(0.75, abs=1e-9)
    assert pN[2] == pytest.approx(0.25, abs=1e-9)


def test_orchestrator_cluster_binomial():
    # dwait [0 1] [.5 .5], T=2: P(N >= k) = P(Binomial(k, 1/2) <= 2).
    # This is the counterexample that caught the boundary-batch bug: the
    # zero-run riding at the last epoch <= T must count.
    T = 2.0
    k, pN, info = wait_count_pmf(
        *_one(Severity('dhistogram', sev_xs=[0, 1], sev_ps=[.5, .5])), T)
    assert info['p0'] == pytest.approx(0.5, abs=1e-15)
    sf = np.array([1.0] + [ss.binom(kk, 0.5).cdf(2) for kk in range(1, len(k) + 1)])
    exact = -np.diff(sf)
    assert np.abs(pN - exact).max() < 1e-9


def test_orchestrator_direct_vs_factored():
    # the 0-atom convolves exactly at lattice index 0, so running the kernel
    # WITH the atom left in must agree with the factored composition
    T = 2.0
    k_f, pN_f, info = wait_count_pmf(
        *_one(Severity('dhistogram', sev_xs=[0, 1], sev_ps=[.5, .5])), T)
    pm = np.zeros(8)
    pm[0] = pm[1] = 0.5
    k_d, pN_d = renewal_count_pmf(pm, 1.0, T, lattice=True, kmax=80)
    n = min(len(pN_f), len(pN_d))
    assert np.abs(pN_f[:n] - pN_d[:n]).max() < 1e-9


def test_orchestrator_negative_mass_collapses_with_warning():
    # negative atoms collapse into the zero-wait batch: W' in {0: .25, 2: .75}
    # so S_k = 2 * #twos and P(N >= k) = P(Binomial(k, .75) <= 1) for T=2
    T = 2.0
    with pytest.warns(UserWarning, match='negative waiting-time mass'):
        k, pN, info = wait_count_pmf(
            *_one(Severity('dhistogram', sev_xs=[-1, 2], sev_ps=[.25, .75])), T)
    assert info['p0'] == pytest.approx(0.25, abs=1e-15)
    sf = np.array([1.0] + [ss.binom(kk, 0.75).cdf(1) for kk in range(1, len(k) + 1)])
    exact = -np.diff(sf)
    assert np.abs(pN[:len(exact)] - exact).max() < 1e-9


def test_orchestrator_all_zero_mass_errors():
    with pytest.raises(ValueError, match='at or below 0'):
        wait_count_pmf(
            *_one(Severity('dhistogram', sev_xs=[0], sev_ps=[1])), 1.0)


def test_orchestrator_defective_weights():
    # weights summing to q < 1 = terminating process (the dwait ... ! form):
    # P(N >= k) = q^k gammainc(k, T) for defective expon waits
    q, T = 0.6, 2.0
    k, pN, info = wait_count_pmf(
        [(Severity('expon', sev_scale=1.0), 0.0, np.inf, True)], [q], T)
    assert info['defect'] == pytest.approx(1 - q, abs=1e-12)
    sf = np.array([1.0] + [q ** kk * sp.gammainc(kk, T) for kk in k[1:]])
    exact = -np.diff(np.append(
        sf, q ** (len(k) + 1) * sp.gammainc(len(k) + 1, T)))
    assert np.abs(pN - exact).max() < 1e-8
    assert pN.sum() == pytest.approx(1.0, abs=1e-8)


def test_orchestrator_defective_window():
    # unconditional splice window [0, 1.1]: waits beyond 1.1 terminate.
    # P(N = 0) = P(W escapes) = exp(-1.1); defect reported exactly.
    T = 2.0
    k, pN, info = wait_count_pmf(
        [(Severity('expon', sev_scale=1.0), 0.0, 1.1, False)], [1.0], T)
    assert info['defect'] == pytest.approx(np.exp(-1.1), abs=1e-9)
    assert pN[0] == pytest.approx(np.exp(-1.1), abs=1e-6)
    assert pN.sum() == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# wait_grid sizing diagnostics
# ---------------------------------------------------------------------------

def test_wait_grid_bs_df_structure():
    bs, log2, lattice, bs_df = wait_grid(1.0, 1.0, 10.0)
    assert list(bs_df.index) == ['coverage', 'shape', 'accuracy',
                                 'exact_lattice', 'hard_atom_snap',
                                 'log2_cap']
    assert bs_df.selected.sum() == 1
    assert not lattice
    assert int(round(10.0 / bs)) * bs == pytest.approx(10.0, rel=1e-12)
    for key in ('bs', 'log2', 'n1', 'lattice', 'kmax_est'):
        assert key in bs_df.attrs


def test_wait_grid_exact_lattice_selected():
    bs, log2, lattice, bs_df = wait_grid(1.5, 0.5, 3.0, atoms=[1.0, 2.0])
    assert lattice
    assert bs == pytest.approx(1.0)
    assert bs_df.loc['exact_lattice', 'selected']
    assert bs_df.selected.sum() == 1


def test_wait_grid_incommensurable_falls_back():
    bs, log2, lattice, bs_df = wait_grid(
        1.0, 0.5, 1.0, atoms=[np.sqrt(0.1), 1.0])
    assert not lattice
    assert bs_df.selected.sum() == 1
