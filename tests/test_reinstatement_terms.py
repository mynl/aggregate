"""Tests for :class:`aggregate.reinstatement.ReinstatementTerms`.

The worked numbers are the **corrected** values from ``dev/reinstatements.md``
(the pre-plan's section 7 paste was numerically corrupted):

* one event, X=175  -> R=75,  A=75,  RP=7.5
* two events        -> R=150, A=150, RP=10
* three events      -> R=210, A=200, RP=10

for a 100 xs 100 layer with one 100% reinstatement, deposit 10, rol 10%.
"""

import numpy as np
import pytest

from aggregate.reinstatement import ReinstatementTerms


# 100 xs 100, one reinstatement at 100%, deposit 10 -> rol 10%
TERMS = ReinstatementTerms(limit=100.0, rates=(1.0,), deposit=10.0)


# ----------------------------------------------------------------------
# derived quantities
# ----------------------------------------------------------------------
def test_derived_quantities():
    assert TERMS.rol == pytest.approx(0.10)
    assert TERMS.n_reinstatements == 1
    assert TERMS.reinstatement_capacity == pytest.approx(100.0)     # m*y
    assert TERMS.total_recovery_capacity == pytest.approx(200.0)    # (m+1)y
    # max RP = r * (R wedge my) at R=my=100 -> 0.1*100 = 10
    assert TERMS.maximum_reinstatement_premium == pytest.approx(10.0)


# ----------------------------------------------------------------------
# the three worked examples (corrected)
# ----------------------------------------------------------------------
def test_example_1_one_event():
    R = 75.0
    assert TERMS.recovery(R) == pytest.approx(75.0)
    assert TERMS.reinstatement_premium(R) == pytest.approx(7.5)
    assert TERMS.ceded_premium(R) == pytest.approx(17.5)


def test_example_2_two_events():
    R = 150.0
    assert TERMS.recovery(R) == pytest.approx(150.0)
    assert TERMS.reinstatement_premium(R) == pytest.approx(10.0)


def test_example_3_three_events_exhaustion():
    R = 210.0
    assert TERMS.recovery(R) == pytest.approx(200.0)      # capped at (m+1)y
    assert TERMS.reinstatement_premium(R) == pytest.approx(10.0)  # capped at my


# ----------------------------------------------------------------------
# vectorization and monotonicity
# ----------------------------------------------------------------------
def test_vectorized_and_monotone():
    R = np.linspace(0, 300, 1001)
    A = TERMS.recovery(R)
    RP = TERMS.reinstatement_premium(R)
    assert A.shape == R.shape and RP.shape == R.shape
    assert np.all(np.diff(A) >= -1e-12)
    assert np.all(np.diff(RP) >= -1e-12)
    assert np.all(A <= TERMS.total_recovery_capacity + 1e-9)
    assert np.all(RP <= TERMS.maximum_reinstatement_premium + 1e-9)


def test_recovery_piecewise_slopes():
    # B(R) = A - RP increasing; slope (1-r) below my, then 1, then 0 (reinstatements.md)
    R = np.array([50.0, 150.0, 250.0])
    B = TERMS.recovery(R) - TERMS.reinstatement_premium(R)
    # at 50: (1-r)*50 = 45 ; at 150: 150 - r*my = 150-10 = 140 ; at 250: (m+1)y - r*my = 190
    assert B == pytest.approx([45.0, 140.0, 190.0])


# ----------------------------------------------------------------------
# free / differently priced schedules at the breakpoints
# ----------------------------------------------------------------------
def test_free_then_half_then_full_schedule():
    # one free, one at 50%, three at 100%: alpha = (0, .5, 1, 1, 1), m=5
    t = ReinstatementTerms(limit=100.0, rates=(0.0, 0.5, 1.0, 1.0, 1.0),
                           deposit=10.0)
    r = 0.10
    assert t.n_reinstatements == 5
    assert t.total_recovery_capacity == pytest.approx(600.0)
    # first tranche free: h(y)=0 at R=100
    assert t.reinstatement_premium(100.0) == pytest.approx(0.0)
    # second tranche at 50%: h(2y) = r*0.5*y = 0.1*0.5*100 = 5
    assert t.reinstatement_premium(200.0) == pytest.approx(5.0)
    # third tranche full: h(3y) = 5 + r*y = 5 + 10 = 15
    assert t.reinstatement_premium(300.0) == pytest.approx(15.0)
    # max: 5 + 3*10 = 35
    assert t.maximum_reinstatement_premium == pytest.approx(35.0)


def test_free_reinstatement_partial_in_tranche():
    t = ReinstatementTerms(limit=100.0, rates=(0.0, 1.0), deposit=10.0)
    # R=150: first 100 free, next 50 at full -> r*50 = 5
    assert t.reinstatement_premium(150.0) == pytest.approx(5.0)


# ----------------------------------------------------------------------
# alternative constructors
# ----------------------------------------------------------------------
def test_from_tranches_unequal_widths():
    # two reinstatements, widths 50 and 150, both full rate
    t = ReinstatementTerms.from_tranches(
        limit=100.0, widths=(50.0, 150.0), rates=(1.0, 1.0), deposit=10.0)
    assert t.reinstatement_capacity == pytest.approx(200.0)
    assert t.total_recovery_capacity == pytest.approx(300.0)
    # R=100: first 50 at full (r*50=5) + next 50 of the second tranche (r*50=5) = 10
    assert t.reinstatement_premium(100.0) == pytest.approx(10.0)


def test_from_callable_validates_and_evaluates():
    t = ReinstatementTerms.from_callable(
        limit=100.0, premium_function=lambda R: 0.1 * np.minimum(R, 100.0),
        deposit=10.0, recovery_cap=200.0)
    assert t.reinstatement_premium(75.0) == pytest.approx(7.5)
    assert t.ceded_premium(75.0) == pytest.approx(17.5)


def test_from_callable_rejects_decreasing():
    with pytest.raises(ValueError, match='nondecreasing'):
        ReinstatementTerms.from_callable(
            limit=100.0, premium_function=lambda R: 200.0 - R,  # nonneg but down
            deposit=10.0, recovery_cap=200.0)


def test_from_callable_rejects_non_vectorized():
    with pytest.raises(ValueError):
        ReinstatementTerms.from_callable(
            limit=100.0, premium_function=lambda R: float(R),  # breaks on array
            deposit=10.0, recovery_cap=200.0)


# ----------------------------------------------------------------------
# input validation
# ----------------------------------------------------------------------
def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ReinstatementTerms(limit=0.0, rates=(1.0,), deposit=10.0)
    with pytest.raises(ValueError):
        ReinstatementTerms(limit=100.0, rates=(1.0,), deposit=-1.0)
    with pytest.raises(ValueError):
        ReinstatementTerms(limit=100.0, rates=(-0.5,), deposit=10.0)


def test_empty_rates_is_zero_reinstatements():
    # An empty ``rates`` tuple is the valid zero-reinstatements case (m=0): a
    # single annual limit y, no reinstatement premium (DecL ``no reinstatements``).
    t = ReinstatementTerms(limit=100.0, rates=(), deposit=10.0)
    assert t.n_reinstatements == 0
    assert t.total_recovery_capacity == pytest.approx(100.0)
    assert t.reinstatement_capacity == pytest.approx(0.0)
    assert t.recovery(250.0) == pytest.approx(100.0)
    assert t.reinstatement_premium(250.0) == pytest.approx(0.0)
    assert t.maximum_reinstatement_premium == pytest.approx(0.0)
