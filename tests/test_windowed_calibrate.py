"""Calibration on a windowed grid (``dev/done/plan-windowed-grid-calibration.md``).

A grid produced by ``_bucket_window`` starts at ``x0 > 0`` rather than at the
origin, deliberately, so that ``q`` / ``F`` / plots are defined on the window.
The layer integral ``∫₀^a g(S(x)) dx`` is not defined there: it needs the region
``[0, x0)``, over which ``S == 1`` and so ``g(S) == 1`` for every distortion,
contributing exactly ``x0``. ``calibrate_distortions`` therefore slides the
density onto the 0-based frame, calibrates against the target ``P - x0``, and
brings the receipt back out, the same transform the signed branch applies with a
positive ``c``.

Acceptance (plan): a windowed build and the same program forced zero-based with
``x_min=0`` calibrate to the same shapes and the same pentagon. Before the fix
they did not, quietly on an ``Aggregate`` (expected loss short by ``x0``, the
integral short by the same rectangle, so the families converged on plausible
numbers) and loudly on a ``Portfolio`` (cached ``exa_total`` is window aware, so
only the integral was short: a negative PH index, infinite Wang and Dual).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from aggregate import build

NAMES = ('ccoc', 'ph', 'wang', 'dual', 'tvar')

# The reported program: a limit profile whose default grid windows hard.
AGG = """agg WC.LimitProfile
  [10000 20000 5000] premium at [0.8 0.7 0.5] lr
  [1000 2000 5000] xs 0
  sev lognorm 50 cv 1.5
  poisson"""

PORT = """port WC.WindowTest
  agg U1 500 claims sev lognorm 50 cv 1.5 poisson
  agg U2 400 claims sev lognorm 60 cv 1.4 poisson"""

PENTAGON = ['L', 'M', 'P', 'a', 'Q', 'LR']


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def _cal(obj):
    return obj.calibration_df.loc['calibration']


@pytest.fixture(scope='module')
def windowed_agg():
    return build(AGG)


@pytest.fixture(scope='module')
def windowed_port():
    return build(PORT)


# ---------------------------------------------------------------------------
# The window really is off the origin (guards the fixtures themselves)
# ---------------------------------------------------------------------------

def test_fixtures_are_windowed(windowed_agg, windowed_port):
    """Both fixtures must actually window, or every test below is vacuous."""
    assert float(windowed_agg.density_df.index[0]) > 0
    assert float(windowed_port.density_df.index[0]) > 0


# ---------------------------------------------------------------------------
# Windowed == zero based, on both arms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('program, zero_based', [
    (AGG, dict(x_min=0)),
    # ``x_min`` is an Aggregate.update kwarg: Portfolio.update rejects it, so
    # the whole-book reference is pinned with an explicit grid instead
    (PORT, dict(log2=18, bs=2)),
], ids=['aggregate', 'portfolio'])
def test_windowed_matches_zero_based(program, zero_based):
    """The same program windowed and forced onto the legacy zero-based grid
    calibrates to the same shapes and the same pentagon. This is the regression
    the plan asks for: build twice, once plain and once zero based."""
    win = build(program)
    ref = build(program, **zero_based)
    assert float(win.density_df.index[0]) > 0
    assert float(ref.density_df.index[0]) == 0.0

    dw = win.calibrate_distortions(0.1, p=0.99).distortion_df
    dr = ref.calibrate_distortions(0.1, p=0.99).distortion_df
    # shapes agree to discretization noise, not to floating-point dust: the two
    # builds sit on different bucket sizes, so the reference is itself only an
    # approximation of the same law
    np.testing.assert_allclose(dw['param'].values, dr['param'].values,
                               rtol=1e-4, atol=0)
    cw, cr = _cal(win), _cal(ref)
    for col in PENTAGON:
        assert cw[col] == pytest.approx(cr[col], rel=1e-4), col


@pytest.mark.parametrize('program', [AGG, PORT], ids=['aggregate', 'portfolio'])
def test_windowed_calibration_converges(program):
    """Every family fits on a windowed grid. Before the fix the ``Portfolio``
    arm returned a negative PH index and infinite Wang / Dual parameters, and
    no test asserted otherwise, which is how they survived."""
    obj = build(program)
    df = obj.calibrate_distortions(0.1, p=0.99, names=NAMES).distortion_df
    assert list(df.index) == list(NAMES)
    assert np.isfinite(df['param'].values).all()
    assert (df['error'].abs() < 1e-3).all()
    assert df.loc['ph', 'param'] > 0
    assert df.loc['tvar', 'param'] < 1


def test_expected_loss_is_the_limited_expectation(windowed_agg):
    """The reported ``L`` is ``E[min(X, a)]`` read straight off the pmf. The
    old failure was short by exactly ``x0``, the area of the ``[0, x0)``
    rectangle that the windowed survival sum omits."""
    a = windowed_agg.q(0.99, 'lower')
    d = windowed_agg.density_df
    x = d.index.values
    S = np.maximum(1.0 - d.p_total.cumsum().values, 0.0)
    x0 = float(x[0])
    # ∫₀^a S = the [0, x0) rectangle (S == 1 there) plus the windowed sum
    lev = x0 + float(S[x < a].sum() * windowed_agg.bs)

    windowed_agg.calibrate_distortions(0.1, a=a)
    assert _cal(windowed_agg)['L'] == pytest.approx(lev, rel=1e-9)


# ---------------------------------------------------------------------------
# The shift is covariant: only L, P, a move
# ---------------------------------------------------------------------------

def test_margin_and_capital_are_shift_invariant(windowed_agg):
    """``M``, ``Q`` and the cost of capital do not see the window: the
    accounting identities close on the reported, out of frame numbers."""
    res = windowed_agg.calibrate_distortions(0.1, p=0.99)
    c = _cal(windowed_agg)
    assert c['P'] == pytest.approx(c['L'] + c['M'], rel=1e-9)
    assert c['a'] == pytest.approx(c['P'] + c['Q'], rel=1e-9)
    assert c['M'] / c['Q'] == pytest.approx(0.1, rel=1e-6)
    assert c['ROE'] == pytest.approx(0.1)
    assert res.coc == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# lr= resolves out of frame (author ruling, 1.0.0a289)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('lr', [0.90, 0.95])
def test_lr_resolves_out_of_frame(windowed_agg, lr):
    """A reader who writes ``lr=`` on a windowed grid means the loss ratio on
    the premium they are shown, so the requested ratio comes back on the
    reported ``L`` and ``P``. The signed branch resolves in frame instead, and
    that divergence is deliberate: a canonical shift stands for genuinely
    negative outcomes, a window is an artifact of the grid."""
    res = windowed_agg.calibrate_distortions(lr=lr, p=0.99)
    c = _cal(windowed_agg)
    assert c['LR'] == pytest.approx(lr, rel=1e-9)
    assert c['L'] / c['P'] == pytest.approx(lr, rel=1e-9)
    assert res.coc > 0
    # the in-frame reading would have been a different number entirely
    x0 = float(windowed_agg.density_df.index[0])
    assert (c['L'] - x0) / (c['P'] - x0) != pytest.approx(lr, rel=1e-3)


# ---------------------------------------------------------------------------
# A grid already at the origin takes x0 = 0 and is untouched
# ---------------------------------------------------------------------------

def test_zero_based_grid_untouched():
    """``x0 == 0`` makes the slide a no-op (the density is not even copied), so
    the legacy path is byte-for-byte what it was."""
    a = build('agg WC.Classic 50 claims sev lognorm 40 cv 1.5 poisson',
              log2=16, bs=1)
    assert float(a.density_df.index[0]) == 0.0
    df = a.calibrate_distortions(0.1, p=0.99).distortion_df
    assert list(df.index) == list(NAMES)
    assert (df['error'].abs() < 1e-3).all()
    c = _cal(a)
    assert c['ROE'] == pytest.approx(0.1)
    assert c['L'] > 0 and c['P'] > 0 and c['a'] > 0


# ---------------------------------------------------------------------------
# The a= anchor takes the same route as p=
# ---------------------------------------------------------------------------

def test_asset_anchor_windowed(windowed_agg):
    """Anchoring on ``a`` rather than ``p`` shifts identically: the caller's
    ``a`` is out of frame and is snapped on the object's own grid."""
    a = float(windowed_agg.q(0.995, 'lower'))
    by_a = windowed_agg.calibrate_distortions(0.1, a=a).distortion_df
    by_p = windowed_agg.calibrate_distortions(0.1, p=0.995).distortion_df
    np.testing.assert_allclose(by_a['param'].values, by_p['param'].values,
                               rtol=1e-9, atol=0)
