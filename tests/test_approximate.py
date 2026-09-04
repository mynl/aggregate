"""Tests for the ``approximate`` keyword -- method-of-moments aggregates.

``approximate norm | lognorm | gamma | sgamma | slognorm`` replaces the freq x
sev FFT convolution, at construction, with a single continuous severity fitted
to the aggregate's moments (the shifted pair match mean, cv and skew, with
normal as the symmetric limit and a reflected fit for left skew; the unshifted
three match mean and cv) carried on a fixed frequency of 1 claim. The
substitution happens in ``Aggregate.__init__`` so the resulting object is an
ordinary 1-claim aggregate: ``density_df``, validation, the ``pnl`` affine, and
the ``Portfolio`` combine all work with no special-casing. The emitted severity
type mirrors the input (``sev`` clamps at 0, ``ssev`` stays signed). See
dev/done/plan-approximate.md and dev/done/plan-approximate-punchup.md.

Covers: moment fidelity vs the exact aggregate across all three skew regimes
(right / symmetric / left, the last via the reflect path, clamped under a plain
``sev`` input and exact under ``ssev``), the negative-loc but
positive-mass case (must NOT be marked signed), occurrence-reinsurance rejection
(parse-time and direct constructor) with aggregate reinsurance allowed, the
``pnl`` combination, the Portfolio combine, the unknown-kind error, the DecL
reach of the unshifted families, the parser-born object mode, and the survey.

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
    return np.array([a.actual_m, a.actual_cv, a.actual_skew])


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
def test_left_skew_reflect_clamps_under_sev(kind):
    """A left-skewed plain ``sev`` aggregate reflects, then clamps at 0.

    Fixed frequency + a left-skewed severity (beta(5, 1.3) is left-skewed)
    gives a negative aggregate skew, so the fit reflects. The emitted severity
    type mirrors the INPUT (author ruling 2026-09-04,
    dev/done/plan-approximate-punchup.md): a plain ``sev`` input stays unsigned, so
    the reflected fit's sub-zero tail is clamped to an atom at 0 and the
    matched moments drift by the clamp mass, visibly (the point of the
    ruling). The ``ssev`` twin keeps the fit exact; see
    ``test_ssev_input_mirrors_ssev_and_is_exact``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = build("agg EL 1 claim sev 100 * beta 5 1.3 fixed")
        approx = build(f"agg AL 1 claim sev 100 * beta 5 1.3 fixed approximate {kind}")
    assert exact.actual_skew < 0
    assert approx.sevs[0].sev_reflect is True
    # mirrors the unsigned input: NOT signed, despite the reflected fit
    assert approx._signed_sev is False
    # the clamp is an atom at 0 (mass a few e-4 for this fixture)
    assert float(approx.density_df.p_total.iloc[0]) > 1e-5
    # moments drift by the clamp: mean stays tight, cv and skew degrade in
    # order (measured ~5e-5 / ~2e-3 / ~3e-2 relative on this fixture)
    t, e = _theory(exact), _empirical(approx)
    assert e[0] == pytest.approx(t[0], rel=1e-3)
    assert e[1] == pytest.approx(t[1], rel=1e-2)
    assert e[2] == pytest.approx(t[2], rel=1e-1)


@pytest.mark.parametrize("kind", ["sgamma", "slognorm"])
def test_ssev_input_mirrors_ssev_and_is_exact(kind):
    """An ``ssev`` (signed) input mirrors to ``ssev`` and reproduces exactly.

    The signed twin of ``test_left_skew_reflect_clamps_under_sev``: with the
    input declared signed the reflected fit is carried unclamped, so the
    matched moments reproduce to fit tolerance.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = build("agg ELS 1 claim ssev 100 * beta 5 1.3 fixed")
        approx = build(f"agg ALS 1 claim ssev 100 * beta 5 1.3 fixed approximate {kind}")
    assert approx.sevs[0].sev_reflect is True
    assert approx._signed_sev is True
    assert approx._signed() is True
    np.testing.assert_allclose(_empirical(approx), _theory(exact), rtol=REL)


def test_symmetric_uses_normal_limit():
    """A symmetric aggregate (skew ~ 0) degenerates to the normal limit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = build("agg ES dfreq [3] dsev [1 2 3]")
        approx = build("agg AS dfreq [3] dsev [1 2 3] approximate sgamma")
    assert abs(exact.actual_skew) < 1e-9
    assert approx.sevs[0].sev_name == "norm"
    np.testing.assert_allclose(_empirical(approx)[:2], _theory(exact)[:2], rtol=REL)
    # the unsigned input clamps the normal's tiny sub-zero tail (~1e-5 mass)
    # to an atom at 0, which lifts the realized skew off exact zero
    # (dev/done/plan-approximate-punchup.md mirroring ruling); measured ~1.3e-4
    assert abs(approx.est_skew) < 1e-3


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
        approx = build("pnl PA 600000 prem less agg PA_e 5000 claims sev lognorm 100 cv 2 "
                       "poisson approximate sgamma")
    # the approximation is applied to the loss leg before the PnL snapshots it;
    # observably, E[margin] = consideration - E[loss] = 600000 - 500000 and the
    # net result mass is conserved (the eager group-by loses no mass).
    assert approx.est_m == pytest.approx(100000.0, rel=1e-3)
    assert approx.result.to_series().sum() == pytest.approx(1.0, abs=1e-6)


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
    assert approx.actual_m == pytest.approx(exact.actual_m, rel=1e-4)


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
    assert fz.mean() == pytest.approx(a.actual_m, rel=1e-3)
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
    assert np.isfinite(fz.mean()) and fz.mean() == pytest.approx(a.actual_m, rel=1e-6)
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
    assert allfits["slognorm"].mean() == pytest.approx(a.actual_m, rel=1e-6)


def test_method_reflected_fit_representability():
    """A left-skewed fit reflects: every output serves it except scipy.

    Driven straight through the fit core / adapter with a negative skew so the
    test does not depend on a particular signed-severity DecL program. The
    reflected fit renders in DecL through the ordinary reflection syntax
    ``loc - scale * name shape`` (dev/done/plan-approximate-punchup.md); only
    ``output='scipy'`` still raises, because scipy has no frozen reflected rv.
    """
    from aggregate.distributions import (approximate_from_mcvsk,
                                          _approximate_sev_kwargs)
    sev = _approximate_sev_kwargs(100.0, 0.3, -0.8, "slognorm")
    assert sev.get("sev_reflect") is True              # reflected fit
    with pytest.raises(ValueError, match="reflected"):
        approximate_from_mcvsk(100.0, 0.3, -0.8, "n", "agg n 1 claim ",
                               "note", "slognorm", "scipy")
    # the DecL outputs render the reflection as the rsub form and build;
    # a signed input carries the fit unclamped, so the moments reproduce
    frag = approximate_from_mcvsk(100.0, 0.3, -0.8, "n", "agg n 1 claim ",
                                  "note", "slognorm", "sev_decl",
                                  signed_input=True)
    assert " - " in frag and "lognorm" in frag
    prog = approximate_from_mcvsk(100.0, 0.3, -0.8, "n", "agg n 1 claim ",
                                  "note", "slognorm", "agg_decl",
                                  signed_input=True)
    assert prog.startswith("agg n 1 claim ssev ")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = build(prog)
    np.testing.assert_allclose(_empirical(r), [100.0, 0.3, -0.8], rtol=REL)
    # the kwargs surface mirrors signedness through sev_signed
    kw_signed = approximate_from_mcvsk(100.0, 0.3, -0.8, "n", "a", "nt",
                                       "slognorm", "sev_kwargs",
                                       signed_input=True)
    assert kw_signed.get("sev_reflect") and kw_signed.get("sev_signed")
    kw_plain = approximate_from_mcvsk(100.0, 0.3, -0.8, "n", "a", "nt",
                                      "slognorm", "sev_kwargs")
    assert kw_plain.get("sev_reflect") and "sev_signed" not in kw_plain


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


# ----------------------------------------------------------------------
# Punchup (dev/done/plan-approximate-punchup.md): all five families in DecL,
# the norm DecL fragment fix, sev/ssev mirroring, the parser-born object
# mode, and the survey after reflected fits render.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["norm", "lognorm", "gamma"])
def test_decl_unshifted_families_build(kind):
    """``approximate norm | lognorm | gamma`` build from DecL and match m, cv.

    The unshifted families are two-moment fits: the declared aggregate's skew
    is not reproduced (``norm`` targets zero skew).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = build("agg EU 100 claims sev lognorm 50 cv 1 poisson")
        approx = build(f"agg AU 100 claims sev lognorm 50 cv 1 poisson approximate {kind}")
    assert approx.approximation == kind
    assert approx.frequency.freq_name == "fixed"
    t, e = _theory(exact), _empirical(approx)
    assert e[0] == pytest.approx(t[0], rel=1e-3)
    assert e[1] == pytest.approx(t[1], rel=1e-2)


def test_unknown_kind_error_names_all_six():
    """The parse-time kind error enumerates every accepted kind."""
    with pytest.raises(Exception) as exc_info:
        build("agg U6 100 claims sev lognorm 100 cv 2 poisson approximate wibble")
    msg = str(exc_info.value)
    for kind in ("exact", "norm", "lognorm", "gamma", "sgamma", "slognorm"):
        assert kind in msg


def test_norm_agg_decl_parses_and_builds():
    """``approximate('norm', output='agg_decl')`` speaks today's DecL (the bug).

    The norm branch used to emit the pre-refactor ``{scale} @ norm 1 # {loc}``
    spelling, which the current grammar rejects. The modern form is
    ``{scale} * norm + {loc}``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = build("agg NB 10 claims sev lognorm 50 cv 1 poisson")
        prog = a.approximate("norm", output="agg_decl")
        assert "@" not in prog and "#" not in prog and "* norm +" in prog
        surrogate = build(prog)
    # mean survives the clamp to ~the clamp mass; cv to grid tolerance
    assert surrogate.est_m == pytest.approx(a.est_m, rel=1e-2)


def test_sev_type_mirrors_input_on_method_outputs():
    """``sev`` in, ``sev`` out; ``ssev`` in, ``ssev`` out (all method outputs)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = build("agg MP 10 claims sev lognorm 50 cv 1 poisson")
        signed = build("agg MS 10 claims ssev -lognorm 10 cv 0.5 poisson")
    prog_plain = plain.approximate("norm", output="agg_decl")
    assert " sev " in prog_plain and " ssev " not in prog_plain
    prog_signed = signed.approximate("slognorm", output="agg_decl")
    assert " ssev " in prog_signed
    # sev_kwargs mirrors through sev_signed
    assert "sev_signed" not in plain.approximate("norm", output="sev_kwargs")
    assert signed.approximate("slognorm", output="sev_kwargs").get("sev_signed") is True
    # the built surrogates carry the mirrored signedness
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = build(prog_plain)
        ss = build(prog_signed)
    assert sp._signed() is False
    # the clamped normal carries an atom at 0
    assert float(sp.density_df.p_total.iloc[0]) > 0
    assert ss._signed() is True


def test_object_mode_is_parser_born():
    """``output='agg'`` returns a ``build``-born object carrying a program."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = build("agg OB 10 claims sev lognorm 50 cv 1 poisson")
        ob = a.approximate("slognorm", output="agg")
        assert ob.program
        assert "note{" in ob.program
        rb = build(ob.program)
    assert rb.est_m == pytest.approx(ob.est_m, rel=1e-9)
    assert rb.est_cv == pytest.approx(ob.est_cv, rel=1e-9)


def test_object_mode_reflected_fixture():
    """The object mode serves a left-skew (reflected) fit too."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        left = build("agg OL 1 claim ssev 100 * beta 5 1.3 fixed")
        ob = left.approximate("slognorm", output="agg")
        assert ob.program and " ssev " in ob.program
        rb = build(ob.program)
    np.testing.assert_allclose(_empirical(rb), _theory(left), rtol=REL)


def test_all_survey_object_and_scipy_outputs():
    """``approximate('all')``: five families on 'agg'; scipy skips reflection only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        right = build("agg SR 100 claims sev lognorm 50 cv 1 poisson")
        allobj = right.approximate("all", output="agg")
        assert set(allobj) == {"norm", "gamma", "lognorm", "sgamma", "slognorm"}
        assert all(ob.program for ob in allobj.values())
        left = build("agg SL 1 claim sev 100 * beta 5 1.3 fixed")
        # scipy: the shifted fits reflect and are skipped, the rest serve
        assert set(left.approximate("all", output="scipy")) == \
            {"norm", "gamma", "lognorm"}
        # agg_decl: everything serves once the reflection renders
        assert set(left.approximate("all", output="agg_decl")) == \
            {"norm", "gamma", "lognorm", "sgamma", "slognorm"}


def test_portfolio_object_mode_parser_born():
    """``Portfolio.approximate`` object mode is ``build``-born likewise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = build("port POB agg U1 50 claims sev lognorm 100 cv 1 poisson "
                  "agg U2 40 claims sev lognorm 80 cv 1.2 poisson")
        ob = p.approximate("slognorm", output="agg")
        assert ob.program
        rb = build(ob.program)
    assert rb.est_m == pytest.approx(ob.est_m, rel=1e-9)


# ----------------------------------------------------------------------
# Second scope (dev/done/plan-approximate-punchup.md): the approximation frames.
# ----------------------------------------------------------------------
_FRAME_COLUMNS = ["exact", "norm", "gamma", "lognorm", "sgamma", "slognorm"]


def _right_skew_fixture():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build("agg FR 10 claims sev lognorm 50 cv 1 poisson")


def test_approximation_df_shape():
    """Six family columns over the meta / stats / quantiles row blocks."""
    a = _right_skew_fixture()
    df = a.approximation_df
    assert list(df.columns) == _FRAME_COLUMNS
    assert df.columns.name == "approximation"
    assert df.index.names == ["component", "measure"]
    assert list(dict.fromkeys(df.index.get_level_values(0))) == [
        "meta", "stats", "quantiles"]
    assert list(df.loc["meta"].index) == ["distribution", "shape", "loc",
                                          "scale"]
    assert list(df.loc["stats"].index) == ["mean", "cv", "skew", "ks"]
    # the quantile rows ride the tail_df ladder
    ps = df.loc["quantiles"].index.to_list()
    assert ps == a.tail_df.index.to_list()


def test_approximation_df_exact_column_and_ks():
    """The exact column reads the grid; ks is 0 there, positive elsewhere,
    and smaller for the shifted three-moment families than for norm."""
    a = _right_skew_fixture()
    df = a.approximation_df
    st = df.loc["stats"]
    assert st.loc["mean", "exact"] == pytest.approx(a.est_m)
    assert st.loc["cv", "exact"] == pytest.approx(a.est_cv)
    assert st.loc["ks", "exact"] == 0.0
    for kind in _FRAME_COLUMNS[1:]:
        assert st.loc["ks", kind] > 0
    assert st.loc["ks", "norm"] > st.loc["ks", "sgamma"]
    assert st.loc["ks", "norm"] > st.loc["ks", "slognorm"]
    # the exact quantiles read the grid
    q = df.loc["quantiles"]
    for p in (0.5, 0.99):
        assert q.loc[p, "exact"] == pytest.approx(a.q(p))


def test_approximation_df_clamp_shows_and_fragment_builds():
    """On a plain ``sev`` fixture the normal fit's clamp lift is displayed
    and the meta fragment builds under ``sev``."""
    a = _right_skew_fixture()
    df = a.approximation_df
    # the clamped normal's achieved mean sits above the exact mean by the
    # clamp mass at 0 (the ruling: displayed, not hidden)
    assert df.loc[("stats", "mean"), "norm"] > df.loc[("stats", "mean"),
                                                      "exact"]
    frag = df.loc[("meta", "distribution"), "norm"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surrogate = build(f"agg FRN 1 claim sev {frag} fixed")
    assert surrogate.est_m == pytest.approx(
        df.loc[("stats", "mean"), "norm"], rel=1e-3)
    # the clamped normal's low quantiles floor at 0
    assert df.loc[("quantiles", 0.001), "norm"] == 0.0


def test_approximation_df_ssev_no_clamp_matches():
    """On an ``ssev`` fixture the shifted columns reproduce the exact
    moments to fit tolerance while the two-parameter columns miss skew."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = build("agg FS 1 claim ssev 100 * beta 5 1.3 fixed")
    df = a.approximation_df
    st = df.loc["stats"]
    for kind in ("sgamma", "slognorm"):
        assert st.loc["mean", kind] == pytest.approx(st.loc["mean", "exact"],
                                                     rel=1e-3)
        assert st.loc["cv", kind] == pytest.approx(st.loc["cv", "exact"],
                                                   rel=1e-2)
        assert st.loc["skew", kind] == pytest.approx(st.loc["skew", "exact"],
                                                     rel=5e-2)
    # left skew: the two-parameter families cannot go negative
    assert st.loc["skew", "exact"] < 0
    assert st.loc["skew", "gamma"] > 0
    assert st.loc["skew", "lognorm"] > 0
    # the reflected fragments spell the rsub form
    assert " - " in df.loc[("meta", "distribution"), "slognorm"]


def test_approximation_density_df_columns_sum_to_one():
    """Each density column sums to ~1 (clamped and reflected included)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        right = build("agg FDR 10 claims sev lognorm 50 cv 1 poisson")
        left = build("agg FDL 1 claim sev 100 * beta 5 1.3 fixed")
    for a in (right, left):
        d = a.approximation_density_df
        assert list(d.columns) == _FRAME_COLUMNS
        assert d.index.name == "loss"
        sums = d.sum()
        # a clamped family misses only its atom at 0, a few percent at worst
        # on these fixtures; exact sums to 1 by construction
        assert sums["exact"] == pytest.approx(1.0, abs=1e-6)
        for kind in _FRAME_COLUMNS[1:]:
            assert sums[kind] == pytest.approx(1.0, abs=0.05)


def test_portfolio_approximation_df_reads_the_total():
    """The Portfolio frame reads the total's grid and moments."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = build("port FP agg U1 5 claims sev lognorm 100 cv 1 poisson "
                  "agg U2 4 claims sev lognorm 80 cv 1.2 poisson")
    df = p.approximation_df
    assert list(df.columns) == _FRAME_COLUMNS
    assert df.loc[("stats", "mean"), "exact"] == pytest.approx(p.est_m)
    d = p.approximation_density_df
    assert d["exact"].sum() == pytest.approx(1.0, abs=1e-6)


def test_approximation_df_no_half_bucket_bias():
    """Family moments are read under the round convention, no +bs/2 shift.

    Regression for [Approximation-Centered-Mass] (a333): reading a family
    cumulative at the grid points assigns each bucket's mass to its right
    endpoint, shifting the achieved mean by +bs/2 against the exact
    column, glaring on a coarse grid (a mean-6 dice book at bs = 1 read
    its normal fit as 6.5). The frame now reads G(x + bs/2), the round
    discretization of the emitted law, so the fit's true mean survives.
    """
    a = build("agg FB dfreq [3] dsev [1 2 3]")
    assert a.bs == 1.0
    st = a.approximation_df.loc["stats"]
    # the normal fit of a symmetric book has mean exactly E[X]; before the
    # fix this row read 6.5
    assert st.loc["mean", "norm"] == pytest.approx(st.loc["mean", "exact"],
                                                   rel=1e-6)
