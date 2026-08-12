"""Tests for the ``p = 1`` guard ([Unbounded-Anchor-Guard], 1.0.0a260).

Phase L2 of ``dev/plan-pricing-exhibits.md``. ``p = 1`` on an unbounded risk
used to resolve, silently, to the last grid point carrying mass, so the asset
level and every number priced off it moved with ``log2`` rather than with the
risk. There was no guard anywhere in the chain.

The test is the tail classification, not the realized density: a density frame
cannot tell a bounded law from an unbounded one that ran out of grid, which is
the confusion the guard exists to end.

Three ways past it, and each says something different: ``a=`` names the level,
a ``p`` below 1 asks a question the distribution can answer, and
``bounded = True`` certifies a support the heuristic could not prove.
"""
from __future__ import annotations

import pytest

from aggregate import build

# Poisson frequency, so unbounded even under a bounded per-occurrence severity
# limit: a bounded layer times an unbounded claim count still has no ceiling.
UNBOUNDED = 'agg UA.Book 250 claims sev lognorm 100 cv 1.5 poisson'
LIMITED = ('agg UA.Limited 250 claims sev lognorm 100 cv 1.5 '
           'occurrence net of 900 xs 100 poisson')
BOUNDED = 'agg UA.Dice dfreq [3] dsev [1:6]'
BOOK = ('port UA.Port '
        'agg UA.PU1 250 claims sev lognorm 100 cv 1.5 poisson '
        'agg UA.PU2 dfreq [2] dsev [1 2 3]')


@pytest.fixture(scope='module')
def unbounded():
    return build(UNBOUNDED)


@pytest.fixture(scope='module')
def bounded():
    return build(BOUNDED)


def test_an_unbounded_aggregate_refuses_p_equals_one(unbounded):
    assert not unbounded.bounded
    with pytest.raises(ValueError, match='p=1 does not name an asset level'):
        unbounded.calibrate_distortions(0.15, p=1)


def test_the_message_names_the_three_ways_past_it(unbounded):
    with pytest.raises(ValueError) as excinfo:
        unbounded.calibrate_distortions(0.15, p=1)
    message = str(excinfo.value)
    assert 'UA.Book' in message                       # which object
    assert 'a=' in message and 'bounded = True' in message
    assert 'tail_behavior_df' in message              # where to look


def test_a_bounded_aggregate_accepts_p_equals_one(bounded):
    """On a bounded law p=1 is the maximum loss and means what it says."""
    assert bounded.bounded
    result = bounded.calibrate_distortions(0.15, p=1)
    assert result.a == pytest.approx(bounded.q(1))


def test_certifying_the_support_lets_p_equals_one_through(unbounded):
    """``bounded = True`` is the modeller saying the heuristic missed a cap."""
    a = build(UNBOUNDED)
    with pytest.raises(ValueError, match='p=1'):
        a.calibrate_distortions(0.15, p=1)
    a.bounded = True
    try:
        assert a.calibrate_distortions(0.15, p=1).a == pytest.approx(a.q(1))
    finally:
        a.bounded = False


def test_naming_the_level_reproduces_the_old_number(unbounded):
    """``a=q(1)`` is the escape hatch, and it is exactly what p=1 used to do."""
    a = build(UNBOUNDED)
    result = a.calibrate_distortions(0.15, a=a.q(1))
    assert result.a == pytest.approx(a.q(1))
    assert result.anchor == 'a'


def test_a_p_below_one_is_never_touched(unbounded):
    """The guard is on the exact spelling, not on where the quantile lands."""
    assert unbounded.calibrate_distortions(0.15, p=0.99999).a > 0


def test_the_guard_covers_the_whole_pricing_surface(unbounded):
    with pytest.raises(ValueError, match='price_pentagon:'):
        unbounded.price_pentagon(p=1, ROE=0.1)
    with pytest.raises(ValueError, match='price_pentagon_ex:'):
        unbounded.price_pentagon_ex(p=1, ROE=0.1)
    with pytest.raises(ValueError, match='price_pentagon:'):
        unbounded.price_ccoc(0.1, p=1)


def test_reins_price_df_is_guarded_too():
    ceded = build(LIMITED)
    ceded.calibrate_distortions(0.15, p=0.99)
    with pytest.raises(ValueError, match='reins_price_df:'):
        ceded.reins_price_df(p=1)
    # the structural refusal still wins: no cession is a different complaint
    clean = build(UNBOUNDED)
    with pytest.raises(ValueError, match='carries no reinsurance'):
        clean.reins_price_df(p=1)


def test_a_portfolio_is_bounded_only_when_every_unit_is():
    """``Portfolio.bounded`` is the worst-of over units, and that is the test."""
    book = build(BOOK)
    assert not book.bounded                    # one Poisson unit is enough
    with pytest.raises(ValueError, match='UA.Port is unbounded'):
        book.calibrate_distortions(0.15, p=1)
