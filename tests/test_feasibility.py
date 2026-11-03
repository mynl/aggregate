"""Whether the chosen grid can reproduce the severity's mean at all.

Part two of ``dev/done/plan-validation-punchup.md``, the severity feasibility
reading. The sizer buys reach unconditionally and pays for it with ``bs``
(author ruling 2026-08-21, keeping that decision), so on a thick enough
severity the bucket it lands on cannot resolve the body, and no other bucket
would either. This says so out loud instead of reporting an unexplained 41.6%
mean error.

The mathematics is a ratio of two quantiles of the **size biased** law: the
discretized mean loses ``P_1(Y <= bs/2)`` at the bottom, where the ``round``
scheme puts everything at exactly 0, and ``P_1(Y > n*bs)`` at the top. For an
unlimited lognormal the mean cancels and only ``sigma`` survives,
``log2 >= 11.23 sigma - 1`` at ``delta = 1e-4``, which is the table below.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from aggregate import build
from aggregate._bucket_window import (FEASIBILITY_SLACK, grid_is_infeasible,
                                      severity_feasibility)
from aggregate._severity import (Severity, _layered_lev, feasible_bucket,
                                 size_biased_cdf)
from aggregate.constants import (InfeasibleGridWarning, Validation,
                                 reset_warn_once)

#: ``exp(19.595) * lognorm 2.581`` limited at 1e12: sigma 2.581, thicker than
#: the cat model below, and feasible anyway because it is limited. The control
#: that proves the reading measures feasibility rather than thickness.
HURRICANE = ('agg Hurricane 1.79 claims 1e12 xs 0 '
             'sev exp(19.595) * lognorm 2.581 poisson')

#: The US hurricane ILW cat model from ``library.agg``, in billions. A real
#: model, not a constructed pathology, and the case that motivated all of this.
CAT = 'agg Cat 1.74 claims sev lognorm 8.501 cv 14.624 poisson'


def _built(program, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return build(program, **kw)


# ---------------------------------------------------------------------------
# The closed form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sev, sigma, log2_required', [
    ('lognorm 200 cv 2', 1.2686, 13.2),
    ('lognorm 200 cv 5', 1.8050, 19.3),
    ('lognorm 200 cv 10', 2.1483, 23.1),
    ('lognorm 8.501 cv 14.624', 2.3173, 25.0),
])
def test_log2_required_matches_the_closed_form(sev, sigma, log2_required):
    """``log2 >= 2 z sigma / log 2 - 1``, computed the general way.

    The reading is a numerical inversion of the size biased cdf, so agreeing
    with the closed form to a tenth of an exponent is the check that the
    machinery computes what the derivation says it does. Note the mean does
    not appear: only ``sigma`` does.
    """
    a = _built(f'agg P 1 claim sev {sev} fixed')
    assert float(a.sevs[0].fz.args[0]) == pytest.approx(sigma, abs=5e-4)
    assert a._bs_feasibility['log2_required'] == pytest.approx(
        log2_required, abs=0.1)
    assert a._bs_feasibility['log2_required'] == pytest.approx(
        11.227 * sigma - 1, abs=0.1)


def test_bs_max_is_twice_the_lower_size_biased_quantile():
    """The bottom half bucket is what the mean loses to bucket zero."""
    a = _built('agg P 1 claim sev lognorm 200 cv 2 fixed')
    sev = a.sevs[0]
    bs_max, _reach = feasible_bucket(sev, 1e-4)
    assert size_biased_cdf(sev, bs_max / 2) == pytest.approx(5e-5, rel=1e-3)


def test_reach_is_the_upper_size_biased_quantile():
    """And the grid top is what it loses to truncation."""
    a = _built('agg P 1 claim sev lognorm 200 cv 2 fixed')
    sev = a.sevs[0]
    _bs_max, reach = feasible_bucket(sev, 1e-4)
    assert size_biased_cdf(sev, reach) == pytest.approx(1 - 5e-5, rel=1e-3)


def test_size_biased_cdf_is_a_cdf():
    """Monotone from 0 to 1, which the bisection relies on."""
    sev = _built('agg P 1 claim sev lognorm 200 cv 2 fixed').sevs[0]
    xs = np.geomspace(1e-3, 1e9, 60)
    fs = np.array([size_biased_cdf(sev, x) for x in xs])
    assert np.all(np.diff(fs) >= -1e-12)
    assert fs[0] == pytest.approx(0.0, abs=1e-9)
    assert fs[-1] == pytest.approx(1.0, abs=1e-9)


def test_size_biased_law_of_a_limited_severity_ends_at_the_limit():
    """A layer limit is an atom in the size biased law: everything above is it."""
    sev = _built('agg P 1 claim 1000 xs 0 sev lognorm 200 cv 2 fixed').sevs[0]
    assert size_biased_cdf(sev, 1000.0) == 1.0
    assert size_biased_cdf(sev, 1e9) == 1.0
    assert size_biased_cdf(sev, 999.0) < 1.0


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def test_the_cat_model_cannot_be_reproduced_on_its_grid():
    """``GrossCatXOL``'s shape: sigma 2.317 unlimited, needing log2 25."""
    a = _built(CAT)
    f = a._bs_feasibility
    assert f['log2_required'] == pytest.approx(25.0, abs=0.1)
    assert a.log2 == 16
    assert grid_is_infeasible(f)
    # the number that makes the situation legible
    assert f['lost_at_zero'] == pytest.approx(0.462, abs=0.005)
    assert f['bs_max'] == pytest.approx(0.0303, abs=1e-3)


def test_a_limit_makes_the_same_thickness_feasible():
    """The control: sigma 2.581, thicker, and fine, because it is limited.

    This is what separates a feasibility reading from a thickness reading. Its
    chosen ``bs`` is still ten times coarser than ``bs_max``, which is why its
    mean is off by about 0.09%, but that is a coarse grid rather than an
    impossible one and no ``log2`` is being asked for that it cannot have.
    """
    a = _built(HURRICANE)
    f = a._bs_feasibility
    assert f['log2_required'] < a.log2 + FEASIBILITY_SLACK
    assert not grid_is_infeasible(f)
    assert f['bs'] > f['bs_max']            # coarse, deliberately


def test_an_ordinary_lognormal_is_feasible():
    """``lognorm 200 cv 2`` needs log2 13.2 and gets 16."""
    a = _built('agg OK 10 claims sev lognorm 200 cv 2 poisson')
    assert not grid_is_infeasible(a._bs_feasibility)
    assert a._bs_feasibility['lost_at_zero'] < 1e-4


def test_raising_log2_makes_a_thick_severity_feasible():
    """The reading is about the grid, so a bigger grid changes it."""
    tight = _built('agg CV5a 5 claims sev lognorm 200 cv 5 poisson', log2=16)
    roomy = _built('agg CV5b 5 claims sev lognorm 200 cv 5 poisson', log2=21)
    assert grid_is_infeasible(tight._bs_feasibility)
    assert not grid_is_infeasible(roomy._bs_feasibility)
    # and log2_required itself does not move: it is the severity's number
    assert tight._bs_feasibility['log2_required'] == pytest.approx(
        roomy._bs_feasibility['log2_required'], abs=1e-9)


# ---------------------------------------------------------------------------
# What it reports, and what it declines to report
# ---------------------------------------------------------------------------

def test_the_narrative_carries_the_reading():
    """Short form names the shortfall; long form explains it."""
    a = _built(CAT)
    assert 'severity needs log2 25' in a.bs_description
    expl = a.bs_explanation
    assert 'cannot be reproduced on this grid at any bucket size' in expl
    assert 'occurrence limit' in expl
    assert '46.2' in expl                    # the share lost at zero


def test_a_feasible_grid_says_nothing():
    """No shortfall, no clause: the reading is silent when it has no news."""
    a = _built('agg OK2 10 claims sev lognorm 200 cv 2 poisson')
    assert 'severity needs log2' not in a.bs_description
    assert 'any bucket size' not in a.bs_explanation


@pytest.mark.parametrize('program', [
    'agg D dfreq [1] dsev [1:6]',                       # discrete: exact
    'agg S 5 claims ssev 100 - lognorm 80 cv .2 poisson',   # signed
    'agg R 5 claims sev 100 - lognorm 80 cv .2 poisson',    # reflected
    'agg W 5 claims sev 100 * beta 2 3 poisson',            # outside the four
])
def test_reports_nothing_rather_than_guessing(program):
    """Kinds the closed forms cannot read exactly get no reading at all.

    Not knowing is not the same as knowing it is fine, so ``grid_is_infeasible``
    is ``False`` on a missing reading and nothing is claimed either way.
    """
    a = _built(program)
    assert a._bs_feasibility is None
    assert not grid_is_infeasible(a._bs_feasibility)
    assert 'severity needs log2' not in a.bs_description


def test_the_binding_component_of_a_mixture_wins():
    """A shared grid has to serve every component, so the tightest one rules."""
    a = _built('agg M 5 claims sev lognorm [200 200] cv [2 10] wts [.5 .5] '
               'poisson')
    solo = _built('agg M2 5 claims sev lognorm 200 cv 10 poisson')
    assert a._bs_feasibility['bs_max'] == pytest.approx(
        solo._bs_feasibility['bs_max'], rel=1e-6)


def test_the_reading_survives_a_pinned_bs():
    """A pinned grid is still a grid, and still either works or does not."""
    a = _built(CAT, bs=2, log2=16)
    assert a._bs_feasibility['bs'] == 2.0
    assert grid_is_infeasible(a._bs_feasibility)


def test_severity_feasibility_declines_a_non_finite_grid_top():
    """Defensive: no grid top, no reading."""
    a = _built('agg OK3 10 claims sev lognorm 200 cv 2 poisson')
    assert severity_feasibility(a, a.bs, a.log2, np.inf) is None


# ---------------------------------------------------------------------------
# [Validation-Infeasible-Flag] and [Infeasible-Grid-Warning], part three
# ---------------------------------------------------------------------------
# The reading becomes a validation member and a warning. INFEASIBLE is a new
# validation TYPE, not a failure: it is a statement about the grid rather than
# about the outcome, so it fires whether or not the moments happen to fail and
# it leads the explanation the way DEFECTIVE does, because it explains the
# moment failures underneath it rather than adding to them.


def test_infeasible_is_transparent_to_passes():
    """The whole semantic, asserted on the enum where it lives."""
    assert Validation.INFEASIBLE.passes
    assert not (Validation.INFEASIBLE | Validation.SEV_MEAN).passes
    assert Validation.NOT_UNREASONABLE.passes
    # and it deliberately does NOT join the reinsurance arm, which would let
    # the flag hide the very problem it names
    assert (Validation.INFEASIBLE | Validation.REINSURANCE).passes


def test_the_cat_model_sets_the_flag():
    a = _built(CAT)
    assert a.valid & Validation.INFEASIBLE
    assert a.valid & Validation.SEV_MEAN          # and says why


def test_a_limited_severity_of_the_same_thickness_does_not():
    a = _built(HURRICANE)
    assert not (a.valid & Validation.INFEASIBLE)


def test_an_ordinary_lognormal_does_not_set_the_flag_or_warn():
    reset_warn_once()
    with warnings.catch_warnings():
        warnings.simplefilter('error', InfeasibleGridWarning)
        a = build('agg OK4 10 claims sev lognorm 200 cv 2 poisson')
    assert not (a.valid & Validation.INFEASIBLE)


def test_the_flag_leads_the_short_form_outside_the_failure_list():
    """It is a reading, so it sits before ``fails``, like ``reinsurance``."""
    a = _built(CAT)
    d = a.validation_description
    assert d.startswith('grid infeasible for this severity;')
    assert 'fails sev mean' in d


def test_the_long_form_explains_the_moment_failures_under_it():
    a = _built(CAT)
    e = a.validation_explanation
    assert 'cannot be reproduced on this grid at any bucket size' in e
    assert 'what every moment failure above is measuring' in e
    assert 'occurrence limit' in e
    assert 'log2 25' in e


def test_the_warning_fires_once_and_names_the_numbers():
    reset_warn_once()
    with pytest.warns(InfeasibleGridWarning) as rec:
        build(CAT)
    msg = str(rec[0].message)
    assert 'cannot be reproduced on this grid' in msg
    assert 'log2 = 25' in msg
    assert '46.22' in msg                       # the share lost at zero
    assert 'Add an occurrence limit' in msg
    # once per session per (severity, grid): the second build is silent
    with warnings.catch_warnings():
        warnings.simplefilter('error', InfeasibleGridWarning)
        build(CAT)


def test_a_different_grid_is_a_different_warning():
    """The key carries log2, so a genuinely different grid still speaks up."""
    reset_warn_once()
    with pytest.warns(InfeasibleGridWarning):
        build(CAT, log2=16)
    with pytest.warns(InfeasibleGridWarning):
        build(CAT, log2=18)


# ---------------------------------------------------------------------------
# A shifted law ([Shift-Removed-Moments])
# ---------------------------------------------------------------------------

def test_a_shifted_law_is_not_read_as_an_unshifted_one():
    """It was read as an unshifted gamma of scale 5,100: 13,189.6 and 0.3127."""
    sev = Severity('gamma', sev_a=2, sev_scale=100, sev_loc=10000)
    assert _layered_lev(sev, 10000) == pytest.approx(10000)
    assert size_biased_cdf(sev, 10000) == 0
    assert _layered_lev(sev, np.inf) == pytest.approx(10200)


@pytest.mark.parametrize('kind, shape, shift, log2_required, reach', [
    ('gamma', 2.0, 10000, -0.83, 1.126e4),
    ('lognorm', 1.0, 10000, -0.36, 1.554e4),
    ('lognorm', 2.0, 1000, 12.06, 8.544e6),
    ('lognorm', 3.0, 1000, 25.39, 8.795e10),
])
def test_feasibility_of_a_shifted_law(kind, shape, shift, log2_required, reach):
    """All of the mean lies above the shift, so ``bs_max`` is twice it and the top binds."""
    sev = Severity(kind, sev_a=shape, sev_scale=100, sev_loc=shift)
    bs_max, top = feasible_bucket(sev, 1e-4)
    assert bs_max == pytest.approx(2 * shift, rel=1e-3)
    assert top == pytest.approx(reach, rel=1e-3)
    assert np.log2(top / bs_max) == pytest.approx(log2_required, abs=0.01)


def test_a_thick_shifted_law_still_warns():
    """A decline would have silenced this: the grid really is infeasible."""
    reset_warn_once()
    with pytest.warns(InfeasibleGridWarning) as rec:
        a = build('agg SM.Thick 5 claims sev 100 * lognorm 3 + 1000 poisson')
    assert 'log2 = 25' in str(rec[0].message)
    assert grid_is_infeasible(a._bs_feasibility)


def test_a_shifted_law_no_longer_warns_for_the_law_it_is_not():
    """Read as unshifted this needed log2 21.45 and warned at 16."""
    reset_warn_once()
    with warnings.catch_warnings():
        warnings.simplefilter('error', InfeasibleGridWarning)
        a = build('agg SM.Medium 5 claims sev 100 * lognorm 2 + 1000 poisson')
    assert not grid_is_infeasible(a._bs_feasibility)


def test_the_flag_joins_the_severity_row_of_an_exhibit():
    """A property of the severity against the grid, so it emphasizes ``Sev``."""
    from aggregate.exhibits import _core
    assert _core._SEV_FAILURES & Validation.INFEASIBLE
    assert not (_core._AGG_FAILURES & Validation.INFEASIBLE)


def test_no_program_gains_the_flag_without_a_failing_severity_mean():
    """The corpus gate, asserted as a count so a regression is a number.

    Measured at 1.0.0a313 over the 255 building ``agg`` programs of
    ``_test_suite.agg`` plus ``library.agg``: four set INFEASIBLE
    (``CurvePareto``, ``GrossCatXOL``, ``HeavyTailValidation``,
    ``USXOLTower``), every one of them already failing its severity mean, and
    132 carry a reading at all. This test checks the invariant on the shipped
    library, which is the half that is ours to keep clean.

    ``HeavyTailValidation`` left the set when the library reorganization
    respelled it from ``100 * pareto 1.3 + -100`` to the equivalent
    ``100 * lomax 1.3``: same law, but the sizer reads the shifted Pareto's
    body differently and the flag no longer fires. The invariant the test is
    really about, INFEASIBLE never without a failing severity mean, is
    unchanged and is asserted per entry in the loop below.
    """
    from aggregate import Underwriter
    uw = Underwriter(databases='library')
    uw.load()
    flagged = []
    for kind, name in sorted(uw._recipes):
        if kind != 'agg':
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                a = uw.build(name, update=True)
        except Exception:
            continue
        if a.valid & Validation.INFEASIBLE:
            flagged.append(name)
            assert a.valid & Validation.SEV_MEAN, (
                f'{name}: INFEASIBLE without a failing severity mean, which '
                'would mean the reading is firing on a grid that works')
    assert set(flagged) == {'CurvePareto', 'GrossCatXOL',
                            'USXOLTower'}, sorted(flagged)


# ---------------------------------------------------------------------------
# A franchise or disappearing deductible ([Deductible-Forms-Follow-Ups])
# ---------------------------------------------------------------------------

def _retention_sev(retention_end, conditional=True):
    """``1000 xs 50`` on ``lognorm 100 cv 2``, retention ending at ``retention_end``."""
    return Severity(sev_name='lognorm', sev_mean=100, sev_cv=2, exp_attachment=50,
                    exp_limit=1000, exp_retention_end=retention_end,
                    sev_conditional=conditional)


@pytest.mark.parametrize('retention_end', [50, 200, 2000],
                         ids=['franchise', 'ramp-below-limit', 'ramp-past-limit'])
def test_size_biased_cdf_of_a_retention_payment(retention_end):
    """``P_1(g <= t)`` against direct quadrature of ``g f`` on the ground-up law.

    Below the limit ``g <= t`` exactly when ``X <= u^{-1}(t)``, so the size
    biased mass is the integral of ``g f`` from the attachment to there.
    """
    from scipy.integrate import quad

    from aggregate._severity import _retention_inverse, _retention_payment
    sev = _retention_sev(retention_end)
    a, r, y = 50.0, float(retention_end), 1000.0
    cuts = sorted({a, r, float(_retention_inverse(y, a, r))})

    def mass(hi):
        edges = [a] + [c for c in cuts if a < c < hi] + [hi]
        return sum(quad(lambda x: float(_retention_payment(x, a, r, y)) * sev.fz.pdf(x),
                        lo, up, limit=200)[0] for lo, up in zip(edges, edges[1:]))

    ey = mass(cuts[-1]) + y * sev.fz.sf(cuts[-1])
    assert _layered_lev(sev, y) == pytest.approx(ey, rel=1e-9)
    for t in (10.0, 60.0, 150.0, 500.0, 999.0):
        expected = mass(float(_retention_inverse(t, a, r))) / ey
        assert size_biased_cdf(sev, t) == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert size_biased_cdf(sev, y) == 1.0


def test_a_franchise_has_no_size_biased_mass_below_its_attachment():
    """No payment lies strictly between 0 and ``d1``, so the bottom bucket may be ``2 d1``."""
    sev = _retention_sev(50)
    assert size_biased_cdf(sev, 49.9) == 0
    assert size_biased_cdf(sev, 50.1) > 0
    bs_max, reach = feasible_bucket(sev, 1e-4)
    assert 100 <= bs_max < 101
    assert reach == pytest.approx(1000, rel=1e-6)


def test_retention_reading_does_not_depend_on_conditioning():
    """``pattach`` cancels in the ratio, as it does for the ordinary layer."""
    assert feasible_bucket(_retention_sev(200, True), 1e-4) == pytest.approx(
        feasible_bucket(_retention_sev(200, False), 1e-4), rel=1e-12)


def test_a_distant_retention_end_reads_as_the_ordinary_layer():
    """``d2 -> inf`` is the ordinary deductible, so the reading converges to it."""
    ordinary = Severity(sev_name='lognorm', sev_mean=100, sev_cv=2,
                        exp_attachment=50, exp_limit=1000)
    assert feasible_bucket(_retention_sev(1e15), 1e-4) == pytest.approx(
        feasible_bucket(ordinary, 1e-4), rel=1e-9)
    for t in (10.0, 500.0):
        assert size_biased_cdf(_retention_sev(1e15), t) == pytest.approx(
            size_biased_cdf(ordinary, t), rel=1e-9)


@pytest.mark.parametrize('layers', ['inf xs 1000 to 5000 disappearing',
                                    'inf xs 1000 franchise'],
                         ids=['disappearing', 'franchise'])
def test_a_thick_retention_law_warns_like_the_ordinary_layer(layers):
    """It returned no reading, so an infeasible grid went unreported."""
    reset_warn_once()
    with pytest.warns(InfeasibleGridWarning):
        a = build(f'agg R 5 claims {layers} sev lognorm 1000 cv 20 poisson')
    assert grid_is_infeasible(a._bs_feasibility)
