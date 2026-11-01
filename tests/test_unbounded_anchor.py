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

from aggregate import Distortion, build

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


# ---------------------------------------------------------------------------
# price: exactly one of p and a ([Using-Notes-Guards])
# ---------------------------------------------------------------------------
PH = Distortion('ph', 0.5)
CCOC = Distortion('ccoc', r=0.1)


@pytest.mark.parametrize('p', [1, 1.0])
def test_price_refuses_p_of_one_on_an_unbounded_risk(unbounded, p):
    with pytest.raises(ValueError, match='p=1'):
        unbounded.price(p, PH)


@pytest.mark.parametrize('p', [1.5, 100, 1000])
def test_price_refuses_a_probability_above_one(unbounded, bounded, p):
    for obj in (unbounded, bounded):
        with pytest.raises(ValueError, match=f'p must be a probability of at most 1, got {p}; '
                                             f'pass an asset level as a={p}'):
            obj.price(p, PH)


def test_price_takes_exactly_one_of_p_and_a(unbounded):
    with pytest.raises(ValueError, match='exactly one of p= .* or a='):
        unbounded.price(0.99, PH, a=100)
    with pytest.raises(ValueError, match='exactly one of p= .* or a='):
        unbounded.price(g=PH)
    with pytest.raises(TypeError, match='pricing distortion'):
        unbounded.price(0.99)


def test_price_at_an_asset_level_matches_the_probability(unbounded):
    a = unbounded.q(0.99)
    by_p = unbounded.price(0.99, PH)
    by_a = unbounded.price(g=PH, a=a)
    assert by_a.P.iloc[0] == pytest.approx(by_p.P.iloc[0], rel=1e-12)


def test_price_refuses_an_infinite_or_off_grid_level(unbounded):
    with pytest.raises(ValueError, match='finite asset level'):
        unbounded.price(g=PH, a=float('inf'))
    with pytest.raises(ValueError, match='above the top of the grid'):
        unbounded.price(g=PH, a=1e12)


def test_price_of_a_bounded_risk_at_its_maximum(bounded):
    """On a bounded law p=1 is the maximum loss, and a= reaches the same row."""
    by_p = bounded.price(1, PH)
    assert by_p.a.iloc[0] == 18
    assert bounded.price(g=PH, a=18).P.iloc[0] == pytest.approx(by_p.P.iloc[0], rel=1e-12)


def test_portfolio_price_takes_a_for_assets():
    book = build(BOOK)
    level = book.q(0.99)
    by_a = book.price(distortion=PH, a=level)
    assert by_a.a_reg == level
    assert by_a.price == pytest.approx(book.price(0.99, PH).price, rel=1e-12)
    with pytest.raises(ValueError, match='pass an asset level as a='):
        book.price(level, PH)
    with pytest.raises(ValueError, match='UA.Port is unbounded'):
        book.price(1, PH)
    with pytest.raises(ValueError, match='exactly one of p= .* or a='):
        book.price(distortion=PH)


# ---------------------------------------------------------------------------
# ccoc at a finite anchor ([Using-Notes-Guards])
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('kw', [{'p': 0.99}, {'p': 0.9}])
def test_ccoc_prices_at_a_finite_probability(unbounded, kw):
    """``P = (L + r a) / (1 + r)``: the mass lands at ``a``, not at the grid top."""
    row = unbounded.price(g=CCOC, **kw).iloc[0]
    a = unbounded.q(kw['p'])
    assert row.P == pytest.approx((row.L + 0.1 * a) / 1.1, rel=1e-10)
    assert row.P == pytest.approx(
        unbounded.price_ccoc(0.1, p=kw['p'])['P'].iloc[0], rel=1e-10)


def test_ccoc_prices_at_a_finite_asset_level(unbounded):
    a = unbounded.q(0.99)
    row = unbounded.price(g=CCOC, a=a).iloc[0]
    assert row.P == pytest.approx((row.L + 0.1 * a) / 1.1, rel=1e-10)


def test_ccoc_refuses_an_unbounded_anchor(unbounded):
    with pytest.raises(ValueError, match='p=1'):
        unbounded.price(1, CCOC)
    with pytest.raises(ValueError, match='finite asset level'):
        unbounded.price(g=CCOC, a=float('inf'))


def test_ccoc_refuses_a_bare_apply_distortion(unbounded):
    """A bare call has no asset level, so the whole frame's top row is the problem."""
    with pytest.raises(ValueError, match='mass distortion .* Use price with p below 1'):
        unbounded.apply_distortion(CCOC)


def test_ccoc_prices_a_law_from_its_characteristic_function():
    law = build('chfagg UA.Chf 25 * gamma 4')
    row = law.price(0.99, CCOC).iloc[0]
    assert row.P == pytest.approx((row.L + 0.1 * law.q(0.99)) / 1.1, rel=1e-10)
    with pytest.raises(ValueError, match='mass distortion'):
        law.apply_distortion(CCOC)


def test_a_portfolio_is_bounded_only_when_every_unit_is():
    """``Portfolio.bounded`` is the worst-of over units, and that is the test."""
    book = build(BOOK)
    assert not book.bounded                    # one Poisson unit is enough
    with pytest.raises(ValueError, match='UA.Port is unbounded'):
        book.calibrate_distortions(0.15, p=1)
