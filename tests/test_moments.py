"""Tests for the discretized-density moment helpers in ``aggregate.moments``.

These helpers (``xsden_to_meancv`` / ``xsden_to_meancvskew`` /
``xsden_to_noncentral``) are used on every ``update`` of an ``Aggregate`` or
``Portfolio``, so they get a dedicated regression suite. The shared worker
``_xsden_work`` also handles defective distributions (``sum(p) < 1``) by
placing the missing mass at the implied maximum loss.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from aggregate.constants import VALIDATION_NOISE
from aggregate.moments import (
    xsden_to_mwrangler,
    ser_to_mwrangler,
    xsden_to_meancv,
    xsden_to_meancvskew,
    xsden_to_noncentral,
    _noise_aware_rel_error,
    _snap_noise,
)

# A fair die: discrete uniform on {1, ..., 6}.
DIE_XS = np.arange(1.0, 7.0)
DIE_PS = np.full(6, 1.0 / 6.0)
# Exact moments of the fair die.
DIE_EX1 = 3.5
DIE_EX2 = 91.0 / 6.0
DIE_EX3 = 441.0 / 6.0
DIE_SD = np.sqrt(35.0 / 12.0)
DIE_CV = DIE_SD / DIE_EX1


def test_noncentral_matches_die():
    """``xsden_to_noncentral`` returns the exact raw moments of a fair die."""
    ex1, ex2, ex3 = xsden_to_noncentral(DIE_XS, DIE_PS)
    assert np.isclose(ex1, DIE_EX1, rtol=1e-12)
    assert np.isclose(ex2, DIE_EX2, rtol=1e-12)
    assert np.isclose(ex3, DIE_EX3, rtol=1e-12)


def test_meancvskew_die_symmetric():
    """The fair die is symmetric: mean/cv exact, skew is fp dust (< noise)."""
    m, cv, skew = xsden_to_meancvskew(DIE_XS, DIE_PS)
    assert np.isclose(m, DIE_EX1, rtol=1e-12)
    assert np.isclose(cv, DIE_CV, rtol=1e-12)
    assert abs(skew) < VALIDATION_NOISE


def test_helpers_are_consistent():
    """All three wrappers share ``xsden_to_mwrangler`` so they agree on mean/cv."""
    m1, cv1 = xsden_to_meancv(DIE_XS, DIE_PS)
    m2, cv2, _ = xsden_to_meancvskew(DIE_XS, DIE_PS)
    assert (m1, cv1) == (m2, cv2)


def test_ser_to_mwrangler_matches_xsden():
    """The Series wrapper (index=xs, values=density) matches the xs/den form."""
    ser = pd.Series(DIE_PS, index=DIE_XS)
    m, cv, skew = ser_to_mwrangler(ser).mcvsk
    assert np.isclose(m, DIE_EX1, rtol=1e-12)
    assert np.isclose(cv, DIE_CV, rtol=1e-12)
    assert abs(skew) < VALIDATION_NOISE
    assert ser_to_mwrangler(ser).noncentral == xsden_to_mwrangler(DIE_XS, DIE_PS).noncentral


def test_defective_distribution_places_mass_at_implied_max():
    """Deficit mass goes to the implied max loss xs[-1] + bs."""
    ps = np.full(6, 0.15)  # sums to 0.90 -> deficit 0.10
    ex1, _, _ = xsden_to_noncentral(DIE_XS, ps)
    # implied max = 6 + bs(=1) = 7; ex1 = sum(xs*ps) + 0.10 * 7
    expected = float(np.sum(DIE_XS * ps)) + 0.10 * 7.0
    assert np.isclose(ex1, expected, rtol=1e-12)


def test_defective_logs_info(caplog):
    """A genuinely defective distribution logs at INFO."""
    ps = np.full(6, 0.15)  # deficit 0.10 >> VALIDATION_NOISE
    with caplog.at_level(logging.INFO, logger="aggregate.moments"):
        xsden_to_noncentral(DIE_XS, ps)
    assert "defective" in caplog.text.lower()


def test_proper_distribution_no_defective_log(caplog):
    """A proper distribution does not log a defective warning."""
    with caplog.at_level(logging.INFO, logger="aggregate.moments"):
        xsden_to_noncentral(DIE_XS, DIE_PS)
    assert "defective" not in caplog.text.lower()


def test_zero_mean_gives_nan_cv():
    """A mean-zero distribution yields cv = nan (no division by zero)."""
    xs = np.array([-1.0, 0.0, 1.0])
    ps = np.array([0.5, 0.0, 0.5])
    m, cv, _ = xsden_to_meancvskew(xs, ps)
    assert m == 0.0
    assert np.isnan(cv)


def test_noise_aware_rel_error_relative_branch():
    """When the reference is well away from 0, ordinary relative error."""
    assert np.isclose(_noise_aware_rel_error(1.5, 1.0), 0.5)


def test_noise_aware_rel_error_absolute_branch():
    """When the reference is ~0, fall back to absolute error."""
    # ref below the noise floor -> absolute error est - ref
    assert np.isclose(_noise_aware_rel_error(1e-15, 1e-14), 1e-15 - 1e-14)
    assert _noise_aware_rel_error(2.0, 0.0) == 2.0


def test_noise_aware_rel_error_series():
    """Series in, Series out, with the index preserved and per-row logic."""
    est = pd.Series([1.5, 5e-15], index=["a", "b"])
    ref = pd.Series([1.0, 1e-14], index=["a", "b"])
    out = _noise_aware_rel_error(est, ref)
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["a", "b"]
    assert np.isclose(out["a"], 0.5)             # relative
    assert np.isclose(out["b"], 5e-15 - 1e-14)   # absolute fallback


def test_snap_noise():
    """Dust snaps to 0; genuine values and NaN are preserved."""
    assert _snap_noise(1e-14) == 0.0
    assert _snap_noise(1e-6) == 1e-6
    assert np.isnan(_snap_noise(np.nan))
    out = _snap_noise(pd.Series([1e-14, 1e-6], index=["x", "y"]))
    assert isinstance(out, pd.Series)
    assert out["x"] == 0.0 and out["y"] == 1e-6
