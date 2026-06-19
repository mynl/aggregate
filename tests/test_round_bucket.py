"""Regressions for :func:`aggregate.utilities.round_bucket` (a74).

The bucket rounder picks a "nice" grid step that *covers* the requested ``bs``.
The ladder is ``{1, 2, 4, 5, 8} * 10**k`` for ``bs >= 1`` (rounding up) and
powers of two for ``bs < 1`` (binary-exact). Both have <= 2x gaps, so the old
``2 -> 5`` / ``20 -> 50`` 2.5x jumps (and the resulting bucket overshoot) are
gone. These pin that contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from aggregate.utilities import round_bucket


@pytest.mark.parametrize('bs,expected', [
    (1, 1.0), (1.1, 2.0), (2, 2.0), (2.5, 4.0), (3.39, 4.0), (4, 4.0),
    (4.5, 5.0), (5, 5.0), (5.5, 8.0), (8, 8.0), (8.7, 10.0), (9.9, 10.0),
    (10, 10.0), (13, 20.0), (20, 20.0), (25, 40.0), (50, 50.0), (100, 100.0),
    (250, 400.0), (457, 500.0), (2412, 4000.0),
])
def test_ladder_values_ge_1(bs, expected):
    assert round_bucket(bs) == expected


def test_3_4_rounds_to_4_not_5():
    """The motivating case: 3.4 used to jump to 5 (2.5x ladder); now -> 4."""
    assert round_bucket(3.4) == 4.0


def test_rounds_up_covers_bs():
    """``round_bucket`` never rounds *down* -- the bucket must cover the support."""
    for bs in np.geomspace(0.001, 1e9, 4000):
        assert round_bucket(bs) >= bs - 1e-12


def test_no_jump_bigger_than_2x():
    """Consecutive distinct outputs never differ by more than 2x."""
    vals = sorted(set(round_bucket(x) for x in np.geomspace(1e-3, 1e9, 8000)))
    ratios = [b / a for a, b in zip(vals, vals[1:])]
    assert max(ratios) <= 2.0 + 1e-9


def test_sub_unit_buckets_are_powers_of_two():
    """``bs < 1`` stays binary-exact (divides the FFT grid cleanly)."""
    for bs in [0.9, 0.6, 0.5, 0.4, 0.3, 0.125, 0.1, 1 / 64, 0.001]:
        rb = round_bucket(bs)
        assert rb >= bs
        log2 = np.log2(rb)
        assert log2 == pytest.approx(round(log2)), f'{bs} -> {rb} not a power of two'


def test_inadmissible_raises():
    with pytest.raises(ValueError):
        round_bucket(0)
    with pytest.raises(ValueError):
        round_bucket(np.inf)
