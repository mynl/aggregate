"""Tests for negative-support (signed) severity and the output window (1.0.0a21).

Covers ``dev/plan-negative-x-agg.md`` -- the Aggregate-scope half of the
negative-x work:

- **F1 negative-support severity**: a profit is a negative loss; ``dsev`` with
  negative atoms auto-signs, continuous severities use the ``ssev`` keyword.
  Negatives wrap to the top of the padded FFT buffer via the input offset
  ``i0``.
- **F2 output window**: the aggregate is relabelled onto a window
  ``[x_min, x_min + N*bs)`` by a single ``np.roll`` -- correct for random as
  well as fixed frequency. ``x_min=None`` estimates a two-sided window from the
  analytic moments.
- **value_type** sign-convention member (inert for the distribution).

The DecL programs are mirrored in ``src/aggregate/agg/test_decl.agg``
(section P&L).
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build
from aggregate.constants import DefectiveDistributionWarning
from aggregate.distributions import (
    estimate_agg_window, validate_discrete_distribution)


# ---------------------------------------------------------------------------
# Identity: the default (non-negative, 0-based) path is unchanged.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('program', [
    'agg Id.A 10 claims sev lognorm 100 cv 2 poisson',
    'agg Id.B dfreq [3] dsev [1:6]',
    'agg Id.C 5 claims sev gamma 50 cv 1.5 mixed gamma 0.4',
])
def test_default_path_identity(program):
    """``x_min=0`` reproduces the default update bit-for-bit (unsigned sev)."""
    a = build(program, update=False)
    a.update(log2=12, bs=1)
    base = a.agg_density.copy()

    b = build(program, update=False)
    b.update(log2=12, bs=1, x_min=0)
    assert np.array_equal(base, b.agg_density)
    assert b.i0 == 0 and b.x_min == 0.0 and not b._signed_sev


# ---------------------------------------------------------------------------
# Fixed-N exactness against a hand-computed convolution.
# ---------------------------------------------------------------------------

def test_fixed_n_closed_form():
    """Sum of 3 iid X in {-2, 5} (p .5): support {-6,1,8,15} p {1,3,3,1}/8."""
    a = build('agg PnL.Fix dfreq [3] dsev [-2 5] [.5 .5]', update=False)
    a.update(log2=6, bs=1, x_min=-8)
    assert a._signed_sev          # dsev with negative atom auto-signs
    df = a.density_df
    got = {round(float(x)): round(float(p), 10)
           for x, p in zip(df.index, df.p_total) if abs(p) > 1e-9}
    assert got == {-6: 0.125, 1: 0.375, 8: 0.375, 15: 0.125}
    assert abs(a.agg_density.sum() - 1.0) < 1e-12
    assert abs(a.est_m - 4.5) < 1e-9       # 3 * E[X], E[X] = 1.5


# ---------------------------------------------------------------------------
# Symmetric P&L: random frequency, mean 0, symmetric to machine precision.
# ---------------------------------------------------------------------------

def test_symmetric_poisson_pnl():
    """X in {-1, 1} p .5, N ~ Poisson(20): E[A]=0, Var=E[N]=20, skew 0."""
    a = build('agg PnL.Sym 20 claims dsev [-1 1] [.5 .5] poisson', update=False)
    a.update(log2=8, bs=1, x_min=-128)
    assert a._signed_sev
    assert abs(a.agg_density.sum() - 1.0) < 1e-12
    assert abs(a.est_m) < 1e-9
    assert abs(a.est_var - 20.0) < 1e-6
    assert abs(a.est_skew) < 1e-6
    assert a.q(0.5) == 0.0
    # symmetry of the density about 0
    df = a.density_df
    for k in (1.0, 3.0, 5.0, 10.0):
        assert abs(float(df.p_total.get(k, 0)) -
                   float(df.p_total.get(-k, 0))) < 1e-12


# ---------------------------------------------------------------------------
# Binary P&L lump with a *negative* mean, far from 0, via the auto window.
# ---------------------------------------------------------------------------

def test_binary_pnl_negative_mean_auto_window():
    """X in {-1,10} p {15/16,1/16}, N~Poisson(1e6).

    E[X] = -5/16; E[A] = -312500; E[X^2] = 115/16; Var = 1e6*115/16; sd ~ 2681.9.
    The auto window must place a tight band around the *negative* mean, not at 0.
    """
    a = build('agg PnL.Lump 1e6 claims dsev [-1 10] [15/16 1/16] poisson',
              update=False)
    # analytic checks
    assert abs(a.agg_m - (-312500.0)) < 1.0
    sd = abs(a.agg_m * a.agg_cv)
    assert abs(sd - np.sqrt(1e6 * 115 / 16)) < 1.0
    a.update(log2=16, x_min=None)
    assert a._signed_sev
    # window is a band around the negative mean (not anchored at 0): the grid
    # brackets the mean and stays a few tens of sd from it, far below 0.
    assert a.x_min < -312500 < a.x_max < 0
    assert -312500 - 40 * sd < a.x_min and a.x_max < -312500 + 40 * sd
    assert abs(a.agg_density.sum() - 1.0) < 1e-6   # 12 nines -> full coverage
    assert abs(a.est_m - (-312500.0)) < 5.0
    assert abs(a.est_sd - sd) < 5.0
    assert a.q(0.5) < 0                       # median is negative


# ---------------------------------------------------------------------------
# ssev: continuous signed severity via DecL; auto-windows from build.
# ---------------------------------------------------------------------------

def test_ssev_continuous_signed():
    """``ssev`` declares a never-clamp continuous severity (a P&L)."""
    a = build('agg P 1 claim ssev 10 * norm fixed', update=False)
    assert a.sevs[0].signed                    # severity carries the flag
    assert abs(a.agg_m) < 1e-9                  # norm(0, 10): mean 0
    assert a._signed()                          # aggregate derives signed
    a.update(log2=12)                           # x_min='auto' -> auto window
    assert a.i0 > 0 and a.x_min < 0             # signed grid, brackets 0
    assert abs(a.agg_density.sum() - 1.0) < 1e-4
    assert abs(a.est_m) < 1e-2                  # ~ 0
    assert abs(a.est_sd - 10.0) < 0.1           # raw norm sd, NOT clamped
    assert a.q(0.5) == 0.0 or abs(a.q(0.5)) <= a.bs


def test_ssev_not_clamped_vs_sev():
    """``ssev norm`` keeps its negative tail; plain ``sev norm`` clamps it."""
    signed = build('agg S 1 claim ssev 10 * norm fixed', update=False)
    signed.update(log2=12)
    clamped = build('agg C 1 claim sev 10 * norm fixed', update=False)
    clamped.update(log2=12)
    assert signed.sevs[0].signed and not clamped.sevs[0].signed
    assert abs(signed.est_m) < 1e-2            # signed mean ~ 0
    assert clamped.est_m > 3.0                 # clamp at 0 lifts the mean


def test_build_auto_windows_signed_dsev():
    """A signed dsev aggregate auto-windows straight from build/update."""
    a = build('agg D 5 claims dsev [-3 4] [.5 .5] poisson', update=False)
    a.update(log2=12)                           # no x_min -> auto for signed
    assert a._signed_sev and a.x_min < a.q(0.001)
    assert abs(a.agg_density.sum() - 1.0) < 1e-3
    # E[X] = 0.5 -> E[A] = 5 * 0.5 = 2.5
    assert abs(a.est_m - 2.5) < 0.05


# ---------------------------------------------------------------------------
# A too-narrow forced window leaks mass -> deficit warning (W >= N*bs).
# ---------------------------------------------------------------------------

def test_narrow_window_deficit_warns():
    """Forcing a window narrower than the support raises the defect warning."""
    a = build('agg PnL.Narrow 20 claims dsev [-1 1] [.5 .5] poisson',
              update=False)
    with pytest.warns(DefectiveDistributionWarning):
        a.update(log2=4, bs=1, x_min=-2)     # window [-2, 13], support ~ +/-15


# ---------------------------------------------------------------------------
# value_type sign-convention member.
# ---------------------------------------------------------------------------

def test_value_type_member():
    a = build('agg VT 5 claims sev lognorm 100 cv 1 poisson', update=False)
    a.update(log2=10, bs=2)
    assert a.value_type == 'loss'             # default
    base = a.agg_density.copy()
    a.value_type = 'payoff'                   # inert for the distribution
    assert a.value_type == 'payoff'
    assert np.array_equal(base, a.agg_density)
    with pytest.raises(ValueError):
        a.value_type = 'nonsense'


# ---------------------------------------------------------------------------
# estimate_agg_window: symmetric (normal), right-skew, reflected left-skew.
# ---------------------------------------------------------------------------

def test_estimate_agg_window_symmetric():
    """Skew 0 -> normal window, centred, width 2*z*sd. Takes sd directly."""
    m, sd = 100.0, 10.0
    lo, hi, w = estimate_agg_window(m, sd, 0.0, p=0.999)
    z = abs((hi - m) / sd)
    assert abs(lo - (m - z * sd)) < 1e-9
    assert abs((hi - m) - (m - lo)) < 1e-9    # symmetric about the mean
    assert abs(w - (hi - lo)) < 1e-9


def test_estimate_agg_window_mean_zero():
    """Mean 0 (cv undefined) still works because sd is passed directly."""
    lo, hi, w = estimate_agg_window(0.0, 10.0, 0.0, p=0.999)
    assert lo < 0 < hi
    assert abs(hi + lo) < 1e-9                 # symmetric about 0
    assert w > 0


def test_estimate_agg_window_negative_mean():
    """A negative-mean, near-symmetric aggregate brackets the negative mean."""
    lo, hi, w = estimate_agg_window(-312500.0, 2681.0, 0.003, p=0.999)
    assert lo < -312500 < hi < 0
    assert w > 0


def test_estimate_agg_window_reflection():
    """Left-skew window is the mirror of the right-skew window of -A."""
    m, sd, skew = 100.0, 50.0, 1.2
    lo_r, hi_r, w_r = estimate_agg_window(m, sd, skew, p=0.999)
    lo_l, hi_l, w_l = estimate_agg_window(-m, sd, -skew, p=0.999)
    assert abs(lo_l - (-hi_r)) < 1e-6
    assert abs(hi_l - (-lo_r)) < 1e-6
    assert abs(w_l - w_r) < 1e-6


# ---------------------------------------------------------------------------
# Unified bucket+window estimator (_bs_window_df).
# ---------------------------------------------------------------------------

def test_bs_window_exact_discrete_selected():
    """dfreq x dsev (integer lattice) -> exact_discrete, minimal log2, bs=1."""
    a = build('agg Dice dfreq [3] dsev [1:6]', update=False)
    a.update()                                    # log2=16 cap, bs auto
    df = a._bs_window_df
    assert bool(df.loc['exact_discrete', 'selected'])
    assert a.bs == 1.0 and a.log2 < 16            # minimal exact grid
    assert a.q(0) == 3.0 and a.q(1) == 18.0       # support [3,18]


def test_bs_window_pinned_bs_keeps_log2():
    """Pinning bs means the user controls the grid: log2 is honoured, not shrunk."""
    a = build('agg Dice dfreq [3] dsev [1:6]', update=False)
    a.update(log2=10, bs=1)
    assert a.log2 == 10                            # not reduced to the exact 5


def test_bs_window_bounded_small_selected():
    """Small claim count + bounded severity -> bounded_small beats moment."""
    a = build('agg C 3 claims sev 500 * beta 2 3 poisson', update=False)
    a.update()
    df = a._bs_window_df
    assert bool(df.loc['bounded_small', 'selected'])
    # permissive selection: a hard support bound is accepted up to 1.5x moment
    assert df.loc['bounded_small', 'W'] <= 1.5 * df.loc['moment', 'W']


def test_bs_window_large_count_uses_moment():
    """Large claim count: LLN concentrates; moment window beats bounded."""
    a = build('agg C 5000 claims sev 500 * beta 2 3 poisson', update=False)
    a.update()
    assert bool(a._bs_window_df.loc['moment', 'selected'])


# ---------------------------------------------------------------------------
# sev_density_df: severity on its own grid; density_df has no sev columns.
# ---------------------------------------------------------------------------

def test_sev_density_df_off_window():
    """Severity off the aggregate window is still correct in sev_density_df.

    A high claim count pushes the aggregate window (12-nine coverage) well above
    the per-claim severity support, so the severity is entirely off-window.
    """
    a = build('agg PnL 2000 claims dsev [-2 5] [.5 .5] poisson', update=False)
    a.update(log2=16, bs=1)
    assert 'p_sev' not in a.density_df.columns     # moved out
    sdf = a.sev_density_df
    nz = sdf[sdf.p_sev > 1e-9]
    assert set(np.round(nz.loss.values).astype(int)) == {-2, 5}
    assert a.q_sev(0.5) == -2.0                    # severity quantile still works
    assert not a._severity_in_window()             # flagged off-window


def test_info_discrete_support_and_warning():
    """info renders discrete support (not 'L xs 0') and warns when sev off-window."""
    a = build('agg PnL 2000 claims dsev [-2 5] [.5 .5] poisson', update=False)
    a.update(log2=16, bs=1)
    info = a.info
    assert 'atoms [-2 5]' in info and 'xs 0' not in info
    assert 'OUTSIDE output window' in info
    # many-atom dsev is shortened
    b = build('agg Big dfreq [2] dsev [1:1001]', update=False)
    b.update(log2=12, bs=1)
    assert '1001 atoms' in b.info and '...' in b.info


# ---------------------------------------------------------------------------
# validate_discrete_distribution: dfreq clamps, dsev preserves negatives.
# ---------------------------------------------------------------------------

def test_validate_discrete_clamp_vs_preserve():
    xs, ps = validate_discrete_distribution(
        np.array([-2.0, 5.0]), np.array([0.5, 0.5]), allow_negative=False)
    assert list(xs) == [0.0, 5.0]             # negative clamped to 0 (dfreq)
    xs, ps = validate_discrete_distribution(
        np.array([5.0, -2.0]), np.array([0.5, 0.5]), allow_negative=True)
    assert list(xs) == [-2.0, 5.0]            # preserved and sorted (dsev)
    assert list(ps) == [0.5, 0.5]
