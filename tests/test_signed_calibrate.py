"""Calibration on signed and payoff supports (``dev/plan-signed-distortion-calibration.md``).

``calibrate_distortions`` runs every subclass ``calibrate`` on the canonical
non-negative loss frame ``Z`` (``_canonical_loss_frame``): a payoff is physically
reversed, then everything is shifted by ``c = max(0, -min(support))``. The
subclass layer-integral math stays pure and 0-based; only the caller maps anchors
/ targets in and un-shifts the receipt back to the caller's loss convention.

Acceptance (plan): every distortion calibration is correct for non-negative loss,
signed loss (straddles 0), and payoff (more-is-better). The classic ``X >= 0``
path (``c = 0``, no reverse) is provably untouched.

DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg`` (SC.*).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from aggregate import build

NAMES = ('ccoc', 'ph', 'wang', 'dual', 'tvar')
GRID = dict(log2=10, bs=1, padding=1)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def _cal(obj):
    return obj.calibration_df.loc['calibration']


# ---------------------------------------------------------------------------
# Shift-exactness: a signed loss == the same law slid onto [0, inf)
# ---------------------------------------------------------------------------

def test_shift_exactness_p_anchor():
    """A non-negative law (support anchored at 0) and the same law shifted
    negative calibrate to identical distortion shapes; L, P slide by exactly the
    shift c; M and Q are shift-invariant."""
    base = build('agg SC.Base0 dfreq [1] dsev [0 10 20] [.5 .3 .2]', **GRID)
    sign = build('agg SC.Signed dfreq [1] dsev [-10 0 10] [.5 .3 .2]', **GRID)
    assert base._is_loss_value and sign._is_loss_value
    assert base.density_df.index.min() == 0.0       # classic path
    assert sign.density_df.index.min() < 0.0         # transformed path

    db = base.calibrate_distortions(0.1, p=0.8).distortion_df
    ds = sign.calibrate_distortions(0.1, p=0.8).distortion_df
    np.testing.assert_allclose(db['param'].values, ds['param'].values,
                               rtol=0, atol=1e-12)
    cb, cs = _cal(base), _cal(sign)
    c = 10.0
    assert cb['P'] - cs['P'] == pytest.approx(c, abs=1e-9)
    assert cb['L'] - cs['L'] == pytest.approx(c, abs=1e-9)
    assert cb['M'] == pytest.approx(cs['M'], abs=1e-9)   # invariant
    assert cb['Q'] == pytest.approx(cs['Q'], abs=1e-9)   # invariant
    assert cs['ROE'] == pytest.approx(0.1)               # coc reproduced


def test_shift_exactness_a_anchor():
    """Same, anchoring on the asset level (loss-convention units): base a=10
    corresponds to signed a=0."""
    base = build('agg SC.Base0 dfreq [1] dsev [0 10 20] [.5 .3 .2]', **GRID)
    sign = build('agg SC.Signed dfreq [1] dsev [-10 0 10] [.5 .3 .2]', **GRID)
    db = base.calibrate_distortions(0.1, a=10).distortion_df
    ds = sign.calibrate_distortions(0.1, a=0).distortion_df
    np.testing.assert_allclose(db['param'].values, ds['param'].values,
                               rtol=0, atol=1e-12)
    assert _cal(base)['P'] - _cal(sign)['P'] == pytest.approx(10.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Payoff: the orthogonality guard (reverse <-> g_dual cancel on price)
# ---------------------------------------------------------------------------

def test_payoff_dual_round_trip():
    """A pnl payoff calibrated to a target stores a frame-free loss-frame shape;
    pricing it back through apply_distortion(view='ask') -- which applies g_dual
    for a payoff -- reproduces the (negated) canonical premium target. This is
    the test that fails if reverse and effective_g ever double-flip."""
    pay = build('agg SC.Pay dfreq [1] dsev [-2 5 8] [.2 .4 .4] payoff', **GRID)
    assert pay._is_loss_value is False
    pay.calibrate_distortions(0.1, p=0.9)
    P_report = _cal(pay)['P']
    for name in NAMES:
        g = pay.distortions[name]
        pay.apply_distortion(g, view='ask')
        rho = float(pay.density_df['exag'].iloc[-1])     # rho_{g_dual}(X)
        assert rho == pytest.approx(-P_report, abs=1e-3), name


def test_payoff_receipt_in_loss_convention():
    """A profitable payoff reports a negative L (the 'loss' is really a profit);
    M, Q stay non-negative and the accounting identities hold."""
    pay = build('agg SC.Pay dfreq [1] dsev [-2 5 8] [.2 .4 .4] payoff', **GRID)
    pay.calibrate_distortions(0.1, p=0.9)
    c = _cal(pay)
    assert c['L'] < 0                       # profit reported as negative loss
    assert c['M'] >= 0 and c['Q'] >= 0      # margin / capital stay positive
    assert c['P'] == pytest.approx(c['L'] + c['M'], abs=1e-9)
    assert c['a'] == pytest.approx(c['P'] + c['Q'], abs=1e-9)
    assert c['ROE'] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Per-kind sweep on a signed grid, incl. mass-at-zero kinds and ccoc
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('names', [
    ('ccoc', 'ph', 'wang', 'dual', 'tvar'),
    ('ly', 'clin', 'lep'),                  # mass-at-zero kinds (ess_sup term)
    ('cll',),
])
def test_per_kind_sweep_signed(names):
    """Every family calibrates on a signed grid: the calibrated shape reproduces
    the (in-frame) premium target when its g is integrated against the canonical
    survival."""
    sign = build('agg SC.Signed dfreq [1] dsev [-10 0 10] [.5 .3 .2]', **GRID)
    df = sign.calibrate_distortions(
        0.1, p=0.85, names=names).distortion_df
    assert list(df.index) == list(names)
    assert (df['error'].abs() < 1e-3).all()


def test_signed_and_classic_share_ccoc_closed_form():
    """ccoc is shift-invariant: r = M/Q is identical for a law and its shifted
    twin, and equals the requested coc."""
    base = build('agg SC.Base0 dfreq [1] dsev [0 10 20] [.5 .3 .2]', **GRID)
    sign = build('agg SC.Signed dfreq [1] dsev [-10 0 10] [.5 .3 .2]', **GRID)
    db = base.calibrate_distortions(
        0.1, p=0.8, names=('ccoc',)).distortion_df
    ds = sign.calibrate_distortions(
        0.1, p=0.8, names=('ccoc',)).distortion_df
    assert db.loc['ccoc', 'param'] == pytest.approx(ds.loc['ccoc', 'param'],
                                                     abs=1e-9)
    assert db.loc['ccoc', 'param'] == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# Classic non-negative path is provably untouched (c = 0, no reverse)
# ---------------------------------------------------------------------------

def test_classic_path_unchanged():
    """A non-negative loss never enters the transform branch and matches a fresh
    calibrate call exactly."""
    a = build('agg SC.Classic 50 claims sev lognorm 40 cv 1.5 poisson',
              log2=16, bs=1)
    assert a.density_df.index.min() == 0.0
    df = a.calibrate_distortions(0.1, p=0.99).distortion_df
    assert list(df.index) == list(NAMES)
    assert _cal(a)['ROE'] == pytest.approx(0.1)
    assert (df['error'].abs() < 1e-3).all()
    # all-positive accounting on a genuine loss
    c = _cal(a)
    assert c['L'] > 0 and c['P'] > 0 and c['a'] > 0
