"""Tests for the ``approximate`` keyword -- method-of-moments aggregates.

``approximate sgamma | slognorm`` replaces the freq x sev FFT convolution, at
construction, with a single continuous severity fitted to the aggregate's first
three moments (shifted gamma / shifted lognormal, normal as the symmetric limit,
a reflected fit for left skew) carried on a fixed frequency of 1 claim. The
substitution happens in ``Aggregate.__init__`` so the resulting object is an
ordinary 1-claim aggregate: ``density_df``, validation, the ``pnl`` affine, and
the ``Portfolio`` combine all work with no special-casing. See
dev/done/plan-approximate.md.

Covers: moment fidelity vs the exact aggregate across all three skew regimes
(right / symmetric / left, the last via the reflect path), the negative-loc but
positive-mass case (must NOT be marked signed), occurrence-reinsurance rejection
(parse-time and direct constructor) with aggregate reinsurance allowed, the
``pnl`` combination, the Portfolio combine, and the unknown-kind error.

The DecL programs are mirrored in ``src/aggregate/agg/test_decl.agg`` (section P).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from aggregate import Aggregate, build

# The approximate object's empirical moments are the fit's own moments computed
# on a well-resolved grid, so they reproduce the exact aggregate's analytic
# (m, cv, skew) very tightly. MoM matches the first three moments by design.
REL = 1e-3


def _theory(a):
    return np.array([a.agg_m, a.agg_cv, a.agg_skew])


def _empirical(a):
    return np.array([a.est_m, a.est_cv, a.est_skew])


# ----------------------------------------------------------------------
# Moment fidelity across skew regimes
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["sgamma", "slognorm"])
def test_right_skew_matches_exact(kind):
    """A right-skewed aggregate is reproduced to its first three moments."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = build("agg E 5000 claims sev lognorm 100 cv 2 poisson")
        approx = build(f"agg A 5000 claims sev lognorm 100 cv 2 poisson approximate {kind}")
    assert approx.approximate == kind
    assert approx.frequency.freq_name == "fixed"
    assert approx.n == 1
    # not signed: a right-skewed high-mean fit lives entirely on the positive axis
    assert approx._signed_sev is False
    np.testing.assert_allclose(_empirical(approx), _theory(exact), rtol=REL)


def test_negative_loc_positive_mass_not_signed():
    """A shifted-gamma fit with very negative loc but all-positive mass.

    The deciding test for signing is the low quantile, not the loc: here the
    fit's ``loc`` is hugely negative yet the mass sits near +15,873, so the
    ordinary 0-based grid must be used (signed False) and the moments still
    match.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = build("agg EN 200 claims sev 100 * beta 5 1.3 poisson")
        approx = build("agg AN 200 claims sev 100 * beta 5 1.3 poisson approximate sgamma")
    assert approx._signed_sev is False
    np.testing.assert_allclose(_empirical(approx), _theory(exact), rtol=REL)


@pytest.mark.parametrize("kind", ["sgamma", "slognorm"])
def test_left_skew_reflect_matches_exact(kind):
    """A genuinely left-skewed aggregate uses the reflect path and still matches.

    Fixed frequency + a left-skewed severity (beta(5, 1.3) is left-skewed) gives
    a negative aggregate skew, so the fit is performed on the reflected (right-
    skewed) aggregate and mapped back via ``sev_reflect``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = build("agg EL 1 claim sev 100 * beta 5 1.3 fixed")
        approx = build(f"agg AL 1 claim sev 100 * beta 5 1.3 fixed approximate {kind}")
    assert exact.agg_skew < 0
    assert approx.sevs[0].sev_reflect is True
    assert approx._signed_sev is True
    np.testing.assert_allclose(_empirical(approx), _theory(exact), rtol=REL)


def test_symmetric_uses_normal_limit():
    """A symmetric aggregate (skew ~ 0) degenerates to the normal limit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = build("agg ES dfreq [3] dsev [1 2 3]")
        approx = build("agg AS dfreq [3] dsev [1 2 3] approximate sgamma")
    assert abs(exact.agg_skew) < 1e-9
    assert approx.sevs[0].sev_name == "norm"
    np.testing.assert_allclose(_empirical(approx)[:2], _theory(exact)[:2], rtol=REL)
    assert abs(approx.est_skew) < 1e-6


# ----------------------------------------------------------------------
# Reinsurance interaction
# ----------------------------------------------------------------------
def test_occ_reins_rejected_parse():
    """Occurrence reinsurance + approximate is a parse-time error."""
    with pytest.raises(Exception, match="occurrence reinsurance"):
        build("agg R 100 claims sev lognorm 100 cv 2 occurrence net of 50 xs 0 "
              "poisson approximate sgamma")


def test_occ_reins_rejected_direct_constructor():
    """The constructor rejects occ-reins too (direct-call belt-and-suspenders)."""
    with pytest.raises(ValueError, match="occurrence reinsurance"):
        Aggregate(name="Z", exp_en=100, sev_name="lognorm", sev_a=1.5,
                  sev_scale=100, freq_name="poisson",
                  occ_reins=[(1.0, 50, 0)], occ_kind="net of",
                  approximate="sgamma")


def test_agg_reins_allowed():
    """Aggregate reinsurance rides along on the approximated object."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = build("agg AR 100 claims sev lognorm 100 cv 2 poisson "
                  "aggregate net of 100 xs 0 approximate sgamma")
    assert a.approximate == "sgamma"


def test_unknown_kind_rejected():
    """An unrecognised approximate kind is a clear parse-time error."""
    with pytest.raises(Exception, match="approximate"):
        build("agg U 100 claims sev lognorm 100 cv 2 poisson approximate wibble")


# ----------------------------------------------------------------------
# pnl + approximate, portfolio combine, surfacing
# ----------------------------------------------------------------------
def test_pnl_approximate_loss_part_fitted():
    """``pnl ... approximate`` fits the loss part; the premium affine rides along."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        approx = build("pnl PA 600000 prem - 5000 claims sev lognorm 100 cv 2 "
                       "poisson approximate sgamma")
    assert approx.approximate == "sgamma"
    assert approx._agg_reflect is True
    assert approx._agg_shift == 600000.0
    # E[margin] = premium - E[loss] = 600000 - 500000
    assert approx.est_m == pytest.approx(100000.0, rel=1e-3)
    assert approx.agg_density.sum() == pytest.approx(1.0, abs=1e-6)


def test_portfolio_combine_conserves_mass():
    """A portfolio with an approximate member updates and conserves mass."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = build("port PE\n"
                      "  agg U1 50 claims sev lognorm 100 cv 1 poisson\n"
                      "  agg U2 40 claims sev lognorm 80 cv 1.2 poisson")
        approx = build("port PA\n"
                       "  agg U1 50 claims sev lognorm 100 cv 1 poisson approximate sgamma\n"
                       "  agg U2 40 claims sev lognorm 80 cv 1.2 poisson")
    assert approx.agg_list[0].approximate == "sgamma"
    # each member conserves its own mass inside the portfolio
    assert approx.agg_list[0].agg_density.sum() == pytest.approx(1.0, abs=1e-6)
    # total mass and mean track the exact-member portfolio
    assert approx.density_df.p_total.sum() == pytest.approx(
        exact.density_df.p_total.sum(), abs=1e-6)
    assert approx.agg_m == pytest.approx(exact.agg_m, rel=1e-4)


def test_exact_default_is_inert():
    """``approximate exact`` (and the omitted clause) leave an ordinary aggregate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = build("agg P 100 claims sev lognorm 100 cv 2 poisson")
        explicit = build("agg P 100 claims sev lognorm 100 cv 2 poisson approximate exact")
    assert plain.approximate == "exact"
    assert explicit.approximate == "exact"
    assert explicit.frequency.freq_name == "poisson"
    np.testing.assert_allclose(_theory(explicit), _theory(plain), rtol=0, atol=0)


def test_note_and_info_surface_approximation():
    """The approximation is visible in ``note`` and ``info``.

    The note (and the indented ``info`` line) records *what* was approximated --
    the original program -- and *how* -- the fitted family and parameters.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = build("agg A 5000 claims sev lognorm 100 cv 2 poisson approximate sgamma")
    # note: non-empty, names the fit, and preserves the original program text
    assert a.note
    assert "sgamma" in a.note
    assert "approximated by" in a.note
    assert "lognorm" in a.note            # the original program is preserved
    # info: the always-present marker line plus the program-aware description
    assert "approximate              sgamma" in a.info
    assert "sgamma" in a.info
    assert "lognorm" in a.info            # description carries the program


def test_info_always_shows_approximate_marker_for_exact():
    """An ordinary (``exact``) aggregate still emits the ``approximate`` line.

    Item 1: the marker is a permanent header line (freq -> sev -> approximate),
    positioned after the severity line, shown as ``exact`` when no fit is active.
    """
    a = build("agg X 5 claims sev lognorm 100 cv 2 poisson")
    assert "approximate              exact" in a.info
    # ordering: the approximate line follows the severity line
    info = a.info
    assert info.index("severity distribution") < info.index("approximate")
    # no description continuation for an exact aggregate
    assert a._approx_description() == ""


def test_approximate_note_round_trips_via_program():
    """Re-building from ``a.program`` reproduces the approximation.

    The round-trip rides on ``program`` (re-parsed), not the note, so the
    rebuilt aggregate is itself approximated and self-describing, and the note
    does not compound across the round-trip.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = build("agg A 5000 claims sev lognorm 100 cv 2 poisson approximate sgamma")
        b = build(a.program)
    assert a.approximate == b.approximate == "sgamma"
    assert b.note and "sgamma" in b.note
    # note length is stable across the round-trip (no recursive growth)
    assert len(b.note) == len(a.note)
