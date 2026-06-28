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

The DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg`` (section P).
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
    assert approx.approximation == kind
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
    assert a.approximation == "sgamma"


def test_unknown_kind_rejected():
    """An unrecognised approximate kind is a clear parse-time error."""
    with pytest.raises(Exception, match="approximate"):
        build("agg U 100 claims sev lognorm 100 cv 2 poisson approximate wibble")


# ----------------------------------------------------------------------
# pnl + approximate, portfolio combine, surfacing
# ----------------------------------------------------------------------
def test_pnl_approximate_loss_part_fitted():
    """``pnl ... approximate`` fits the loss leg; the PnL nets the consideration."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        approx = build("pnl PA 600000 prem less 5000 claims sev lognorm 100 cv 2 "
                       "poisson approximate sgamma")
    # the approximation lives on the (loss) risky leg
    assert approx.agg.approximation == "sgamma"
    # E[margin] = consideration - E[loss] = 600000 - 500000
    assert approx.mean == pytest.approx(100000.0, rel=1e-3)
    assert approx.pnl_df.p_total.sum() == pytest.approx(1.0, abs=1e-6)


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
    assert approx.agg_list[0].approximation == "sgamma"
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
    # the attribute is falsey for an exact convolution (``if a.approximation:``)
    assert plain.approximation == ""
    assert explicit.approximation == ""
    assert not plain.approximation
    assert explicit.frequency.freq_name == "poisson"
    np.testing.assert_allclose(_theory(explicit), _theory(plain), rtol=0, atol=0)


def test_approximate_method_not_shadowed_by_attribute():
    """``Aggregate.approximate()`` (the MoM-surrogate *method*) stays callable.

    Regression: the ``approximate=`` constructor kwarg is stored as the
    ``approximation`` *attribute* so it does not shadow the same-named method
    (the parity-partner of ``Portfolio.approximate``). The method must remain
    callable on an instance and return a frozen scipy distribution.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = build("agg M 100 claims sev lognorm 100 cv 2 poisson")
    assert callable(a.approximate)                       # not shadowed by a str
    fz = a.approximate("slognorm")                       # the MoM surrogate
    assert hasattr(fz, "cdf") and hasattr(fz, "ppf")     # frozen scipy rv
    # mean of the fit tracks the aggregate mean
    assert fz.mean() == pytest.approx(a.agg_m, rel=1e-3)
    # ``approximate('all')`` returns the dict of five fits
    allfits = a.approximate("all")
    assert set(allfits) == {"norm", "gamma", "lognorm", "sgamma", "slognorm"}


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
    # info: the always-present marker line (one row, no continuation -- the
    # fitted family + program detail lives in the note / _approx_description)
    assert "approximate              sgamma" in a.info
    assert "lognorm" in a._approx_description()


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
    assert a.approximation == b.approximation == "sgamma"
    assert b.note and "sgamma" in b.note
    # note length is stable across the round-trip (no recursive growth)
    assert len(b.note) == len(a.note)


# ----------------------------------------------------------------------
# .approximate() method (a68): one fit core for both surfaces, symmetric
# guard + warning, reflected-fit representability errors. See
# dev/done/plan-approximate.md and the a68 CHANGELOG entry.
# ----------------------------------------------------------------------
def _symmetric_dice():
    """12-die sum: symmetric (skew=0), the motivating nonsense-fit case."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build("agg SymDice dfreq [12] dsev [1:6]")


def test_method_symmetric_shifted_degenerates_to_normal_with_warning():
    """``slognorm`` on a symmetric book returns the normal limit + a UserWarning.

    Regression: previously ``sln_fit(m, cv, 0) -> (-inf, inf, 0)`` produced a
    ``nan`` scipy distribution silently. Now the shifted fit degenerates to its
    normal limit (a sensible answer) and says so.
    """
    a = _symmetric_dice()
    with pytest.warns(UserWarning, match="symmetric"):
        fz = a.approximate("slognorm", output="scipy")
    # a real normal, not nonsense
    assert np.isfinite(fz.mean()) and fz.mean() == pytest.approx(a.agg_m, rel=1e-6)
    assert np.isfinite(fz.cdf(40))
    # the explicit normal is identical and silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning would fail
        fzn = a.approximate("norm", output="scipy")
    assert fzn.cdf(40) == pytest.approx(fz.cdf(40), rel=1e-12)


def test_method_all_symmetric_is_quiet_and_admissible():
    """``approximate('all')`` on a symmetric book is quiet and degrades cleanly."""
    a = _symmetric_dice()
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # 'all' must not warn
        allfits = a.approximate("all")
    assert {"norm", "gamma", "lognorm", "sgamma", "slognorm"} == set(allfits)
    # the shifted families degenerated to the normal limit
    assert allfits["slognorm"].mean() == pytest.approx(a.agg_m, rel=1e-6)


def test_method_reflected_fit_representability():
    """A left-skewed fit reflects: sev_kwargs works, frozen-scipy/decl error.

    Driven straight through the fit core / adapter with a negative skew so the
    test does not depend on a particular signed-severity DecL program.
    """
    from aggregate.distributions import (approximate_from_mcvsk,
                                          _approximate_sev_kwargs)
    sev = _approximate_sev_kwargs(100.0, 0.3, -0.8, "slognorm")
    assert sev.get("sev_reflect") is True              # reflected fit
    # no native frozen scipy / one-line DecL form -> explicit error
    for out in ("scipy", "sev_decl", "agg_decl"):
        with pytest.raises(ValueError, match="reflected"):
            approximate_from_mcvsk(100.0, 0.3, -0.8, "n", "agg n 1 claim sev ",
                                   "note", "slognorm", out)
    # but the kwargs / Aggregate-object surfaces represent it fine
    assert approximate_from_mcvsk(100.0, 0.3, -0.8, "n", "a", "nt",
                                  "slognorm", "sev_kwargs").get("sev_reflect")


def test_method_one_fit_core_shared():
    """The method adapter and the construction core agree (one implementation).

    The positive-skew shifted fit from ``approximate_from_mcvsk(output='scipy')``
    is built from the very ``sev_kwargs`` the constructor path uses, so their
    parameters match exactly.
    """
    from aggregate.distributions import (approximate_from_mcvsk,
                                          _approximate_sev_kwargs)
    m, cv, skew = 5000.0, 0.8, 1.3
    sev = _approximate_sev_kwargs(m, cv, skew, "slognorm")
    fz = approximate_from_mcvsk(m, cv, skew, "n", "a", "nt", "slognorm", "scipy")
    # frozen lognorm built from the same (shape, loc, scale)
    assert fz.kwds["loc"] == pytest.approx(sev["sev_loc"])
    assert fz.kwds["scale"] == pytest.approx(sev["sev_scale"])
    assert fz.args[0] == pytest.approx(sev["sev_a"])


def test_portfolio_method_symmetric_warns_too():
    """``Portfolio.approximate`` shares the same guard (symmetric -> normal + warn)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = build("port SymPort agg A dfreq [12] dsev [1:6]")
    with pytest.warns(UserWarning, match="symmetric"):
        fz = p.approximate("sgamma", output="scipy")
    assert np.isfinite(fz.mean())
