"""Stage MV-1 regressions: ``balanced_window`` + ``Aggregate.focus``.

``balanced_window(ser, p)`` is the *measure-don't-guess* primitive the bivariate
axis sizing is built on (``dev/plan-mv.md`` §5.1): given a realized pmf and a
discarded tail mass ``p``, it returns the equal-tail window ``[q(p/2),
q(1 - p/2)]`` snapped to ``bs``. ``Aggregate.focus`` is the thin public
re-slicer over a computed ``density_df`` (§5.4). These pin the contract both
must deliver: the window keeps ``1 - p`` of the mass, trims equal *probability*
off each tail (so a signed/skewed margin stays centred on its mass), snaps to
the grid, and ``focus`` round-trips the mass without recompute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats as ss

from aggregate import build
from aggregate.utilities import balanced_window


def _normal_series(loc=250.0, scale=40.0, bs=0.5, n=2000):
    """A finely-bucketed, normalized, symmetric pmf for direct primitive tests."""
    xs = np.arange(n) * bs
    ps = ss.norm.pdf(xs, loc=loc, scale=scale)
    ps = ps / ps.sum()
    return pd.Series(ps, index=xs)


def test_window_holds_one_minus_p():
    """The kept window contains at least ``1 - p`` of the mass."""
    ser = _normal_series()
    for p in (1e-2, 1e-3, 1e-6):
        lo, hi = balanced_window(ser, p)
        kept = float(ser[(ser.index >= lo) & (ser.index <= hi)].sum())
        assert kept >= 1.0 - p - 1e-12, f'p={p}: kept {kept} < {1 - p}'


def test_equal_tail_balance():
    """Each discarded tail carries ~``p/2`` of the mass (balanced = equal-prob)."""
    ser = _normal_series()
    p = 1e-3
    lo, hi = balanced_window(ser, p)
    below = float(ser[ser.index < lo].sum())
    above = float(ser[ser.index > hi].sum())
    # Each tail is at most p/2, and the two are close on a symmetric pmf.
    assert below <= p / 2 + 1e-12
    assert above <= p / 2 + 1e-12
    assert below == pytest.approx(above, abs=1e-4)


def test_symmetric_window_is_centred():
    """A symmetric pmf gets a window centred on its mean."""
    loc = 250.0
    ser = _normal_series(loc=loc)
    lo, hi = balanced_window(ser, 1e-3)
    assert 0.5 * (lo + hi) == pytest.approx(loc, abs=2.0)


def test_snapping_to_bs():
    """With ``bs`` given the edges floor/ceil to grid multiples and bracket the raw window."""
    ser = _normal_series(bs=0.5)
    p = 1e-3
    raw_lo, raw_hi = balanced_window(ser, p)              # already grid points here
    bs = 7.0  # deliberately coarser than the series spacing to force snapping
    lo, hi = balanced_window(ser, p, bs=bs)
    assert lo % bs == pytest.approx(0.0, abs=1e-9)
    assert hi % bs == pytest.approx(0.0, abs=1e-9)
    assert lo <= raw_lo and hi >= raw_hi                 # snapped window contains the raw one


def test_p_out_of_range_raises():
    ser = _normal_series()
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError):
            balanced_window(ser, bad)


def test_focus_round_trips_mass():
    """``focus(p)`` returns a slice holding ``>= 1 - p`` of the total mass, no recompute."""
    a = build('agg FocusN 100 claims sev lognorm 100 cv 0.8 poisson')
    before = a.density_df.copy()
    p = 1e-4
    win = a.focus(p)
    assert float(win['p_total'].sum()) >= 1.0 - p - 1e-9
    # focus does not mutate the aggregate's density_df
    pd.testing.assert_frame_equal(a.density_df, before)
    # window is a contiguous central slice of the full grid
    assert win.index.min() >= a.density_df.index.min()
    assert win.index.max() <= a.density_df.index.max()


def test_focus_matches_balanced_window():
    """``focus`` is exactly ``balanced_window`` on ``p_total`` re-sliced on the grid."""
    a = build('agg FocusN2 50 claims sev lognorm 100 cv 1.2 poisson')
    p = 1e-5
    ser = a.density_df.query('p_total > 0').p_total
    lo, hi = balanced_window(ser, p, bs=a.bs)
    win = a.focus(p)
    assert win.index.min() == pytest.approx(lo)
    assert win.index.max() == pytest.approx(hi)


def test_signed_margin_stays_centred():
    """A signed (P&L) severity keeps mass both sides of zero -- the window isn't clipped to 0.

    The motivating failure mode for the bivariate sizing: an ``ssev`` axis with
    mass below zero must not have its lower tail dropped. ``balanced_window``
    trims equal probability each side, so the window straddles the mean.
    """
    a = build('agg SignedPnL dfreq [1] ssev uniform - .3')
    lo, hi = balanced_window(a.density_df.query('p_total > 0').p_total, 1e-3, bs=a.bs)
    assert lo < 0.0 < hi, f'expected a window straddling 0, got [{lo}, {hi}]'
    mean = a.est_m
    assert lo < mean < hi
