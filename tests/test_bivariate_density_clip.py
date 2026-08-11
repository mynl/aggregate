"""[Joint-Density-Clip]: the de-fuzz on a finished joint density.

The two 2-D FFT paths in ``bivariate.py`` end by zeroing round-off dust.
Until now they did it with ``abs(density) < 1e-15``, a predicate inherited
from ``utilities.remove_fuzz``, whose two-sidedness is justified in its own
docstring for the signed columns of a P&L frame and does not transfer here: a
joint density is non-negative even where its **support** is signed, so a
large negative cell is a broken construction rather than dust, and the old
predicate preserved exactly those while zeroing the harmless ones.

Nothing currently trips it. Measured at 1024², 2048² and 4096² there are no
surviving negatives, the smallest positive is the floor itself, and 14% to
59% of cells are exact zeros. So these tests **inject** rather than wait for
nature: a test that waits passes forever without testing anything.
"""

import logging

import numpy as np

from aggregate.bivariate import _DENSITY_FUZZ, _clip_density_fuzz


def test_dust_is_zeroed_at_the_relative_floor():
    density = np.zeros((4, 4))
    density[0, 0] = 0.5
    density[1, 1] = 0.5
    dust = _DENSITY_FUZZ * 0.1
    density[2, 2] = dust
    density[3, 3] = -dust
    _clip_density_fuzz(density, 'test')
    assert density[2, 2] == 0.0 and density[3, 3] == 0.0
    assert density.sum() == 1.0


def test_the_floor_tracks_the_mass_not_the_peak():
    """A grid ten times as fine has a peak ten times lower and the same noise.

    An absolute constant would sit a decade deeper on the finer grid and a
    decade shallower on the coarser one, which is the depth drifting with the
    resolution. Anchored to the sum it holds still: the same relative dust is
    cleared from both.
    """
    for cells, peak in ((16, 1.0 / 16), (1600, 1.0 / 1600)):
        density = np.full(cells, peak)
        density[0] = peak + _DENSITY_FUZZ * 0.5
        density[1] = _DENSITY_FUZZ * 0.5
        _clip_density_fuzz(density, 'test')
        assert density[1] == 0.0
        assert density[0] > 0.0


def test_a_large_negative_warns_and_is_clipped(caplog):
    """The case the two-sided predicate let through in silence."""
    density = np.zeros((4, 4))
    density[0, 0] = 0.8
    density[1, 1] = 0.3
    density[2, 2] = -0.1        # far above any floor: not dust
    with caplog.at_level(logging.WARNING, logger='aggregate.bivariate'):
        _clip_density_fuzz(density, "bivariate 'Broken'")
    assert density[2, 2] == 0.0
    assert density.min() == 0.0
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert '1 negative cells' in message
    assert '-1.000e-01' in message
    assert "bivariate 'Broken'" in message


def test_clipping_a_negative_drives_the_deficit_negative(caplog):
    """The free detector: both callers take ``1 - sum`` after this runs.

    Zeroing a genuine negative raises the sum, so a deficit below zero is the
    signal that something was clipped upward and this warning should have
    fired. That makes the deficit a second, independent way to notice, which
    matters because a log line is easy to miss and the deficit is already on
    every validation frame.
    """
    density = np.array([[0.9, 0.2], [0.0, -0.1]])
    assert 1.0 - density.sum() == 0.0
    with caplog.at_level(logging.WARNING, logger='aggregate.bivariate'):
        _clip_density_fuzz(density, 'test')
    assert 1.0 - density.sum() < 0.0


def test_a_clean_density_is_left_alone_and_silent(caplog):
    rng = np.random.default_rng(11)
    density = rng.random((32, 32))
    density /= density.sum()
    before = density.copy()
    with caplog.at_level(logging.WARNING, logger='aggregate.bivariate'):
        _clip_density_fuzz(density, 'test')
    np.testing.assert_array_equal(density, before)
    assert not caplog.records


def test_an_all_zero_grid_does_not_divide_by_its_own_emptiness():
    density = np.zeros((3, 3))
    _clip_density_fuzz(density, 'test')
    assert density.sum() == 0.0
