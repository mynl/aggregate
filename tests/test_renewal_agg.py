"""Stage-3 end-to-end ``build()`` tests for the renewal frequency.

The wait clause drives a full FFT aggregate: equivalence anchors vs poisson,
closed-form count laws, cluster/defective processes, premium bookkeeping,
portfolio units, the MoM-approximate path, the realized-dfreq conversion and
the sizing diagnostics. DecL programs mirrored in
``src/aggregate/agg/decl-testers.agg`` (AB.*).
See dev/plan-sparre-a.md [Renewal-Frequency-Wait-Clause].
"""

from math import factorial

import numpy as np
import pytest
import scipy.special as sp
import scipy.stats as ss

from aggregate import build
from aggregate.constants import Validation
from aggregate._frequency import FrequencyRenewal


@pytest.fixture(autouse=True)
def _quiet():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


# ---------------------------------------------------------------------------
# equivalence anchors
# ---------------------------------------------------------------------------

def test_three_way_poisson_equivalence():
    # 10 claims poisson == 10 years wait expon == 1 year wait 0.1*expon
    p = build('agg RP 10 claims sev lognorm 100 cv 1 poisson')
    r1 = build('agg R1 10 years sev lognorm 100 cv 1 wait expon')
    r2 = build('agg R2 1 year sev lognorm 100 cv 1 wait 0.1 * expon')
    assert isinstance(r1.frequency, FrequencyRenewal)
    assert r1.n == pytest.approx(10.0, abs=1e-7)
    assert r2.n == pytest.approx(10.0, abs=1e-7)
    assert (r1.bs, r1.log2) == (p.bs, p.log2)
    for r in (r1, r2):
        assert np.abs(r.density_df.p_total.values
                      - p.density_df.p_total.values).max() < 1e-8
    assert r1.q(0.99) == p.q(0.99)


def test_gamma_wait_count_sf():
    # gamma(2, rate 1) waits: P(N >= k) = gammainc(2k, T)
    T = 5
    a = build(f'agg RG {T} years dsev [1] wait gamma 2')
    pN = a.frequency.freq_b
    k = a.frequency.freq_a.astype(int)
    sf_exact = np.array([1.0] + [sp.gammainc(2 * kk, T) for kk in k[1:]])
    exact = -np.diff(np.append(sf_exact, sp.gammainc(2 * (k[-1] + 1), T)))
    assert np.abs(pN - exact).max() < 1e-8


def test_uniform_wait_factorial():
    # uniform(0,1) waits, T=1: P(N=k) = k/(k+1)!; k <= 1 carries the O(h/4)
    # endpoint artifact (density jump exactly at T), higher k are smooth
    a = build('agg RU 1 year dsev [1] wait uniform')
    pN = a.frequency.freq_b
    exact = np.array([kk / factorial(kk + 1)
                      for kk in range(len(pN))])
    assert np.abs(pN - exact).max() < 5e-6
    assert np.abs(pN[2:] - exact[2:]).max() < 1e-8


# ---------------------------------------------------------------------------
# discrete lattice / cluster / defective processes
# ---------------------------------------------------------------------------

def test_deterministic_dwait():
    a = build('agg RD 3 years dsev [1] dwait [1]')
    assert a.frequency.wait_lattice
    assert a.density_df.loc[3, 'p_total'] == pytest.approx(1.0, abs=1e-8)
    # the count mean folds in the kernel's exp(-20) wrap-leakage tail junk
    # (~2e-9 per entry, k-weighted), so 1e-7 not 1e-8
    assert a.n == pytest.approx(3.0, abs=1e-7)


def test_two_point_dwait_full_bucket():
    # W in {1, 2} each 1/2, T=2: the atom exactly at T counts (full-bucket
    # readout): N=2 iff both waits are 1 (prob 1/4), else N=1
    a = build('agg R2P 2 years dsev [1] dwait [1 2]')
    pN = a.frequency.freq_b
    assert pN[1] == pytest.approx(0.75, abs=1e-9)
    assert pN[2] == pytest.approx(0.25, abs=1e-9)


def test_cluster_dwait_enumeration():
    # dwait [0 1]: zero waits arrive in geometric clusters;
    # P(N >= k) = P(Binomial(k, 1/2) <= T) by S_k = #ones
    a = build('agg RC 2 years dsev [1] dwait [0 1] [.5 .5]')
    assert a.frequency.wait_p0 == pytest.approx(0.5, abs=1e-15)
    pN = a.frequency.freq_b
    kk = np.arange(len(pN) + 1)
    sf = np.array([1.0] + [ss.binom(int(k), 0.5).cdf(2) for k in kk[1:]])
    exact = -np.diff(sf)
    assert np.abs(pN - exact).max() < 1e-9


def test_defective_dwait():
    a = build('agg RF 2 years dsev [1] dwait [1 2] [.4 .5] !')
    assert a.frequency.wait_defect == pytest.approx(0.1, abs=1e-12)
    assert a.density_df.loc[0, 'p_total'] == pytest.approx(0.1, abs=1e-9)
    assert a.frequency.freq_b.sum() == pytest.approx(1.0, abs=1e-9)


def test_defective_splice_window():
    # unconditional splice: waits beyond 1.1 terminate; P(N=0) = exp(-1.1)
    a = build('agg RS 2 years dsev [1] wait expon splice [0 1.1] !')
    assert a.frequency.wait_defect == pytest.approx(np.exp(-1.1), abs=1e-9)
    assert a.density_df.loc[0, 'p_total'] == pytest.approx(
        np.exp(-1.1), abs=1e-6)


def test_mixture_wait_second_order_mean():
    # hyperexponential wait: E N(T) = T/mu + (sigma^2 - mu^2)/(2 mu^2) + o(1)
    a = build('agg RM 5 years dsev [1] wait [.6 .4] * expon wts [.5 .5]')
    mu = 0.5 * 0.6 + 0.5 * 0.4
    ex2 = 0.5 * 2 * 0.36 + 0.5 * 2 * 0.16
    second_order = 5 / mu + (ex2 - 2 * mu * mu) / (2 * mu * mu)
    assert a.n == pytest.approx(second_order, abs=2e-3)


def test_wait_sev_lookup():
    from aggregate import build as _b
    _b('sev RWW expon 1')
    a = build('agg RSN 3 years dsev [1] wait sev.RWW')
    assert a.n == pytest.approx(3.0, abs=1e-7)


# ---------------------------------------------------------------------------
# premium bookkeeping / composition
# ---------------------------------------------------------------------------

def test_years_at_rate_premium_and_lr():
    a = build('agg RR 1 year at 500 rate sev lognorm 100 cv 1 '
              'wait 0.1 * expon')
    assert a.exp_premium == 500.0
    assert a.exp_rate == 500.0
    assert a.n == pytest.approx(10.0, abs=1e-7)
    # loss ratio = el / premium (the sentinel-before-premium reorder fix)
    assert a.agg_m / a.exp_premium == pytest.approx(2.0, abs=1e-6)


def test_pnl_inherit_premium_smoke():
    p = build('pnl RPL inherit premium less agg RPLe 1 year at 500 rate '
              'dsev [100] wait 0.1 * expon')
    assert p.E_consideration == 500.0
    assert p.mean == pytest.approx(-500.0, abs=1e-5)


def test_renewal_unit_in_port():
    p = build('port RPT agg U1 3 years dsev [1] dwait [1] '
              'agg U2 dfreq [1] dsev [2]')
    assert p.agg_m == pytest.approx(5.0, abs=1e-7)


def test_approximate_mom_path():
    a = build('agg RAP 10 years sev lognorm 100 cv 1 wait expon '
              'approximate sgamma')
    assert a.approximation == 'sgamma'
    assert a.agg_m == pytest.approx(1000.0, rel=1e-6)


# ---------------------------------------------------------------------------
# realized dfreq conversion + diagnostics
# ---------------------------------------------------------------------------

def test_create_frequency_realized_dfreq():
    a = build('agg RCF 1 year at 500 rate dsev [1] wait 0.1 * expon')
    prog = a._frequency_program('RCF.freq')
    assert 'dfreq' in prog and 'wait' not in prog and 'years' not in prog
    fa = a.create_frequency()
    got = fa.density_df.p_total.values[:len(a.frequency.freq_b)]
    assert np.abs(got - a.frequency.freq_b).max() < 1e-14
    assert fa.agg_m == pytest.approx(a.n, abs=1e-12)


def test_renewal_bs_df_structure():
    a = build('agg RBS 10 years dsev [1] wait expon')
    df = a._renewal_bs_df
    assert list(df.index) == ['coverage', 'shape', 'accuracy',
                              'exact_lattice', 'hard_atom_snap',
                              'log2_cap']
    assert df.selected.sum() == 1
    for key in ('bs', 'log2', 'n1', 'kmax', 'p0', 'defect',
                'est_count_error'):
        assert key in df.attrs
    # exact-lattice row selected for a commensurable dwait
    b = build('agg RBS2 3 years dsev [1] dwait [1 2]')
    assert b._renewal_bs_df.loc['exact_lattice', 'selected']
    assert b._renewal_bs_df.selected.sum() == 1


def test_validation_clean():
    # renewal theoretical freq moments are the realized pmf's own, so the
    # moment audit is exact-to-discretization
    a = build('agg RV 10 years sev lognorm 100 cv 1 wait expon')
    assert a.valid == Validation.NOT_UNREASONABLE


def test_convergence_check_diagnostic():
    a = build('agg RCC 3 years dsev [1] wait expon')
    delta = a.frequency.convergence_check()
    assert 0 < delta < 1e-7
    # exact lattice: no discretization error, check short-circuits
    b = build('agg RCC2 3 years dsev [1] dwait [1]')
    assert b.frequency.convergence_check() == 0.0


def test_strict_pairing_programmatic():
    from aggregate import Aggregate
    with pytest.raises(ValueError, match='renewal'):
        Aggregate('BadR', exp_en=10, freq_name='poisson',
                  sev_name='lognorm', sev_mean=100, sev_cv=1,
                  wait_name='expon', wait_scale=1.0)
    with pytest.raises(ValueError, match='exp_years'):
        Aggregate('BadR2', freq_name='renewal', exp_en=-1,
                  sev_name='lognorm', sev_mean=100, sev_cv=1,
                  wait_name='expon', wait_scale=1.0)
