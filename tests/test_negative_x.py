"""Tests for negative-support (signed) severity and the output window (1.0.0a21).

Covers ``dev/plan-negative-x-agg.md`` -- the Aggregate-scope half of the
negative-x work:

- **F1 negative-support severity**: a profit is a negative loss; ``dsev`` with
  negative atoms auto-enables signed mode, continuous severities opt in with
  ``signed=True``. Negatives wrap to the top of the padded FFT buffer via the
  input offset ``i0``.
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
    """``x_min=0`` / ``signed=False`` reproduce the default update bit-for-bit."""
    a = build(program, update=False)
    a.update(log2=12, bs=1)
    base = a.agg_density.copy()

    b = build(program, update=False)
    b.update(log2=12, bs=1, x_min=0, signed=False)
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
    # window is a tight band around the negative mean (not anchored at 0)
    assert a.x_min < -300000 < a.x_max < 0
    assert (a.x_max - a.x_min) < 5e4        # tight, << |mean|
    assert abs(a.est_m - (-312500.0)) < 5.0
    assert abs(a.est_sd - sd) < 5.0
    assert a.q(0.5) < 0                       # median is negative


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
    """Skew 0 -> normal window, centred, width 2*z*sd."""
    m, cv = 100.0, 0.1
    lo, hi, w = estimate_agg_window(m, cv, 0.0, p=0.999)
    sd = cv * m
    z = abs((hi - m) / sd)
    assert abs(lo - (m - z * sd)) < 1e-9
    assert abs((hi - m) - (m - lo)) < 1e-9    # symmetric about the mean
    assert abs(w - (hi - lo)) < 1e-9


def test_estimate_agg_window_negative_mean():
    """A negative-mean, near-symmetric aggregate brackets the negative mean."""
    lo, hi, w = estimate_agg_window(-312500.0, -0.00858, 0.003, p=0.999)
    assert lo < -312500 < hi < 0
    assert w > 0


def test_estimate_agg_window_reflection():
    """Left-skew window is the mirror of the right-skew window of -A."""
    m, cv, skew = 100.0, 0.5, 1.2
    lo_r, hi_r, w_r = estimate_agg_window(m, cv, skew, p=0.999)
    lo_l, hi_l, w_l = estimate_agg_window(-m, -cv, -skew, p=0.999)
    assert abs(lo_l - (-hi_r)) < 1e-6
    assert abs(hi_l - (-lo_r)) < 1e-6
    assert abs(w_l - w_r) < 1e-6


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
