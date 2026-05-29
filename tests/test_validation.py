"""Validation behaviour for symmetric / low-skew distributions.

Regression guard for the bug where a symmetric distribution (analytic skew
exactly 0, but fp dust in practice) spuriously failed ``valid`` because the
skew check used a relative error guarded only by ``> 0``. The fix skips the
CV/skew test when the *theoretical* value is at or below ``VALIDATION_NOISE``.
Also covers the denoised ``describe`` display and the newly-populated
empirical raw moments.
"""
from __future__ import annotations

import numpy as np

from aggregate import build, Validation


def test_symmetric_die_validates():
    """A fair die (skew 0) is not flagged for skew/CV."""
    a = build("agg Die dfreq [1] dsev [1:6]")
    assert a.valid == Validation.NOT_UNREASONABLE


def test_symmetric_die_validates_across_grids():
    """Symmetric skew is skipped regardless of the empirical FFT noise.

    The empirical skew of a symmetric distribution is grid-dependent noise
    that can be far larger than the analytic dust; validation must not flag
    it at any reasonable grid.
    """
    for log2 in (8, 13, 16):
        a = build("agg Die dfreq [1] dsev [1:6]", bs=1, log2=log2)
        assert a.valid == Validation.NOT_UNREASONABLE, f"failed at log2={log2}"


def test_describe_snaps_skew_dust_to_zero():
    """``describe`` shows exactly 0 for the (symmetric) skew, not fp dust."""
    a = build("agg Die dfreq [1] dsev [1:6]")
    d = a.describe
    assert d.loc["Sev", "Sk"] == 0.0
    assert d.loc["Agg", "Sk"] == 0.0
    assert d.loc["Sev", "Est Sk"] == 0.0
    assert d.loc["Agg", "Est Sk"] == 0.0


def test_empirical_raw_moments_populated():
    """Empirical ex1/ex2/ex3 are populated for sev and agg."""
    a = build("agg Die dfreq [1] dsev [1:6]")
    emp = a.stats_df["empirical"]
    for comp in ("sev", "agg"):
        for k in ("ex1", "ex2", "ex3"):
            assert not np.isnan(float(emp[(comp, k)])), f"{comp} {k} is NaN"
    # exact die raw moments
    assert np.isclose(float(emp[("sev", "ex1")]), 3.5, rtol=1e-9)
    assert np.isclose(float(emp[("sev", "ex2")]), 91.0 / 6.0, rtol=1e-9)
    assert np.isclose(float(emp[("sev", "ex3")]), 441.0 / 6.0, rtol=1e-9)


def test_empirical_skew_clean_on_wide_grid():
    """Empirical moments are taken from a de-fuzzed copy, so the stored skew
    stays clean on a wide grid where x**3-amplified FFT fuzz would otherwise
    corrupt it. (self.agg_density itself is left as the raw FFT output.)"""
    from aggregate.constants import VALIDATION_NOISE
    a = build("agg Die dfreq [1] dsev [1:6]", bs=1, log2=16)
    assert abs(float(a.stats_df["empirical"][("agg", "skew")])) < VALIDATION_NOISE
    assert np.isclose(float(a.stats_df["empirical"][("agg", "ex3")]), 441.0 / 6.0, rtol=1e-9)


def test_error_column_noise_aware():
    """The skew error row is a tiny absolute value, not the ~-0.83 from the
    old relative-vs-dust computation."""
    a = build("agg Die dfreq [1] dsev [1:6]")
    assert abs(float(a.stats_df["error"][("agg", "skew")])) < 1e-6


def test_symmetric_portfolio_validates():
    """A portfolio whose total is symmetric is not flagged for skew."""
    p = build(
        """port TestSym
            agg A dfreq [1] dsev [1:6]
            agg B dfreq [1] dsev [1:6]""",
        bs=1,
        log2=10,
    )
    assert p.valid == Validation.NOT_UNREASONABLE


def test_skewed_model_still_validates_and_keeps_skew():
    """A genuinely skewed model still validates and its skew is not snapped."""
    b = build("agg LN 25 claims sev lognorm 50 cv 0.75 poisson")
    assert b.valid == Validation.NOT_UNREASONABLE
    # real, non-trivial skew preserved (guards against over-snapping)
    assert float(b.describe.loc["Agg", "Sk"]) > 0.1
