"""Validation behaviour for symmetric / low-skew distributions.

Regression guard for the bug where a symmetric distribution (analytic skew
exactly 0, but fp dust in practice) spuriously failed ``valid`` because the
skew check used a relative error guarded only by ``> 0``. The fix skips the
CV/skew test when the *theoretical* value is at or below ``VALIDATION_NOISE``.
Also covers the denoised ``describe`` display and the newly-populated
empirical raw moments.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from aggregate import build, Validation
from aggregate import _validation


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
    d = a.validation_df
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
    from aggregate.config import get_settings
    VALIDATION_NOISE = get_settings().validation.noise
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
    assert float(b.validation_df.loc["Agg", "Sk"]) > 0.1


def test_valid_relocation_to_validation_module():
    """Phase C relocation guard: the moved free functions
    ``_validation.valid_aggregate`` / ``valid_portfolio`` /
    ``validation_explanation`` reproduce the class-property results exactly
    (clean, reinsurance, and portfolio cases). A relocation bug surfaces as a
    flag/explanation mismatch independent of the rest of the suite.
    """
    from aggregate import _validation

    a = build("agg RC 10 claims sev lognorm 100 cv 2 poisson")
    v_prop = a.valid
    assert v_prop == Validation.NOT_UNREASONABLE
    a._valid = None
    assert _validation.valid_aggregate(a) == v_prop
    assert a.validation_explanation == _validation.validation_explanation(a)

    r = build("agg RR 10 claims sev lognorm 100 cv 2 "
              "occurrence net of 50 xs 50 poisson")
    assert r.valid & Validation.REINSURANCE
    assert "reinsurance" in r.validation_explanation
    r._valid = None
    assert _validation.valid_aggregate(r) == r.valid

    p = build("port RP agg A 10 claims sev lognorm 100 cv 2 poisson "
              "agg B 5 claims sev gamma 50 cv 1 poisson")
    v_pprop = p.valid
    p._valid = None
    assert _validation.valid_portfolio(p) == v_pprop
    assert p.validation_explanation == _validation.validation_explanation(p)


# ---------------------------------------------------------------------------
# [Aliasing-Direct-Measure]: the ALIASING flag measures the convolution step
# ---------------------------------------------------------------------------
# Through 1.0.0a310 the flag was a ratio of the aggregate mean error to the
# severity mean error, floored on the numerator only, at the arithmetic dust
# floor. Any severity that discretized essentially exactly put near zero in
# the denominator and the ratio exploded on nothing: six firings in the 257
# program corpus, six false positives. See
# ``dev/done/plan-validation-punchup.md``.

_FORCED = 'agg AL 100 claims sev gamma 100 cv 1 poisson'


def _forced(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return build(_FORCED, bs=1, **kw)


def test_convolution_residual_cancels_severity_discretization():
    """The predictor uses the DISCRETIZED severity mean, so its error cancels.

    That is the whole design: there is no severity error in the denominator to
    blow up, and what the residual measures is the convolution step alone.
    """
    a = _forced(log2=16)
    en = float(a.stats_df['mixed'][('freq', 'mean')])
    sev = float(a.stats_df['gross_empirical'][('sev', 'mean')])
    agg = float(a.stats_df['gross_empirical'][('agg', 'mean')])
    assert _validation.convolution_residual(en, sev, agg) == pytest.approx(
        abs(agg - en * sev) / (en * sev), rel=1e-12)


@pytest.mark.parametrize('en, sev, agg', [
    (0.0, 100.0, 0.0),              # mean-zero predictor
    (100.0, 0.0, 0.0),              # mean-zero severity
    (np.nan, 100.0, 10000.0),       # not-finite input
    (100.0, 100.0, np.inf),
])
def test_convolution_residual_is_nan_where_it_has_no_meaning(en, sev, agg):
    """No predictor, no residual. ``nan`` never trips the flag."""
    assert np.isnan(_validation.convolution_residual(en, sev, agg))


def test_convolution_residual_takes_absolute_values():
    """A signed aggregate works: the sign of the miss is not the question."""
    assert _validation.convolution_residual(1.0, -100.0, -90.0) == \
        pytest.approx(_validation.convolution_residual(1.0, 100.0, 110.0))


def test_wrap_sets_aliasing_and_not_defective():
    """``padding=0`` runs the FFT on the bare grid, so the tail wraps.

    Wrap CONSERVES mass, so the deficit is exactly zero while the mean moves.
    That is the one shape the flag is for.
    """
    a = _forced(log2=14, padding=0)
    assert a.valid & Validation.ALIASING
    assert not (a.valid & Validation.DEFECTIVE)
    assert a._deficit == 0.0


def test_the_same_grid_padded_sets_neither():
    """``padding=1`` doubles the grid and discards the top half, so no wrap."""
    a = _forced(log2=14, padding=1)
    assert not (a.valid & Validation.ALIASING)
    assert not (a.valid & Validation.DEFECTIVE)


def test_truncation_sets_defective_and_not_aliasing():
    """[Aliasing-Under-Defective]: mass DROPPED is not mass wrapped.

    ``DEFECTIVE`` already owns this case and its explanation already ends
    "Raise log2, or widen the grid"; a second flag saying the same thing in
    different words is noise.
    """
    a = _forced(log2=12, padding=1)
    assert a.valid & Validation.DEFECTIVE
    assert not (a.valid & Validation.ALIASING)


def test_sub_material_truncation_does_not_read_as_wrap():
    """The gate is the dust floor, not ``deficit_materiality``.

    Measured on the corpus, five programs lose the mean to truncation with
    deficits between 1.5e-5 and 9.6e-5, every one under materiality. Gating at
    materiality would call those wrap, which is the old false positive wearing
    new clothes (author ruling 2026-08-21). ``Cc.Freq25.Negbin`` is one of
    them, reproduced here as its own program.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('agg NB 20 claims sev lognorm 100 cv 2 negbin 1.5',
                  log2=13, bs=8)
    assert 0 < a._deficit < 1e-4                   # sub-material truncation
    assert not (a.valid & Validation.ALIASING)


def test_exact_severity_no_longer_explodes_the_test():
    """The regression for ``[Aliasing-Test-Misfires-On-A-Reference-Severity]``.

    A severity that discretizes exactly used to put ~2.2e-16 in the ratio's
    denominator. There is no denominator now.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        inner = build('agg RefInner dfreq [1] dsev [1:6] hints{log2=3; bs=1}')
        a = build('agg RefOuter 5 claims sev agg.RefInner poisson')
    assert not (inner.valid & Validation.ALIASING)
    assert not (a.valid & Validation.ALIASING)


def test_reinsurance_layer_on_bucket_edges_is_clean():
    """The other exact-discretization shape: a layer landing on bucket edges.

    Three of the six corpus false positives were reinsurance towers whose
    severity mean error was one ulp.
    """
    r = build('agg RT 10 claims 1000 xs 0 sev lognorm 100 cv 2 '
              'occurrence net of 500 xs 500 poisson')
    assert not (r.valid & Validation.ALIASING)


def test_portfolio_aliasing_parity_clean():
    """[Portfolio-Aliasing-Parity]: an ordinary portfolio is not flagged."""
    p = build('port PAL agg A 10 claims sev lognorm 100 cv 2 poisson '
              'agg B 5 claims sev gamma 50 cv 1 poisson')
    assert not (p.valid & Validation.ALIASING)


def test_portfolio_predictor_is_the_sum_of_the_units():
    """The portfolio step is the convolution of the units, so that is the
    predictor: their own realized aggregate means, added."""
    p = build('port PAL2 agg A 10 claims sev lognorm 100 cv 2 poisson '
              'agg B 5 claims sev gamma 50 cv 1 poisson')
    predicted = sum(float(a.stats_df['empirical'][('agg', 'mean')])
                    for a in p.agg_list)
    total = float(p.stats_df['empirical'][('agg', 'mean')])
    assert _validation.convolution_residual(1.0, predicted, total) < 1e-5


def test_aliasing_wording_names_the_measurement():
    """Short and long form both describe the convolution, not a ratio."""
    a = _forced(log2=14, padding=0)
    assert 'convolution' in a.validation_description
    assert 'wrap' in a.validation_description
    assert 'discretized severity mean' in a.validation_explanation
    assert 'no mass is missing' in a.validation_explanation


def test_aliasing_ratio_config_key_is_retired():
    """``aliasing_ratio`` is gone; ``aliasing_eps`` replaces it."""
    from aggregate import config
    s = config.get_settings()
    assert not hasattr(s.validation, 'aliasing_ratio')
    assert s.validation.aliasing_eps == pytest.approx(1e-5)
