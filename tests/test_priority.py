"""The priority ladder: senior, equal and junior expected recovery.

Three recoveries a unit can be granted out of an estate of ``a``, depending on
where a receivership schedule places its claim. ``lev_{unit}`` is the senior
leg and ``exa_{unit}`` the equal-priority leg, both written by ``add_exa``;
``Portfolio.priority_df`` adds the junior (subordinated) leg and names all
three uniformly. Plan: ``dev/done/plan-a394-priority-junior-leg.md``.

The identities that pin the junior leg down:

* pointwise ``min(X_i, (a - X_{-i})^+) = min(X, a) - min(X_{-i}, a)``, so on a
  two-unit book ``ex_senior_i + ex_junior_j = ex_total`` exactly, both ways
  round;
* the equal legs sum to the total on a book of any size;
* the expected policyholder deficit is monotone in rank, senior <= equal <=
  junior, for every unit.

One case is pinned to arithmetic rather than to another FFT: a ``dfreq`` /
``dsev`` book whose three legs are computable by hand.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build

# Two units, one roughly twice the other, so the asymmetry of the
# redistribution is visible. Declared means 2000 and 1080.
TWO_UNIT = """port Priority.TwoUnit
    agg Direct 40 claims sev lognorm 50 cv 2 poisson
    agg Assumed 12 claims sev lognorm 90 cv 3 poisson"""

THREE_UNIT = """port Priority.ThreeUnit
    agg Direct 40 claims sev lognorm 50 cv 2 poisson
    agg Assumed 12 claims sev lognorm 90 cv 3 poisson
    agg Fac 6 claims sev lognorm 120 cv 1.5 poisson"""

# All three legs are exact by hand here. X_A in {0, 1} and X_B in {0, 2}, each
# outcome 1/2, so T in {0, 1, 2, 3} each 1/4. At a = 1:
#   lev_A = lev_B = 1/2, lev_total = 3/4
#   exa_A = 1/3, exa_B = 5/12 (they sum to 3/4)
#   ex_junior_A = ex_junior_B = 1/4
HAND = """port Priority.Hand
    agg A dfreq [0 1] dsev [1]
    agg B dfreq [0 1] dsev [2]"""

SIGNED = """port Priority.Signed
    agg A 8 claims ssev 20 * lognorm 0.4 - 25 poisson
    agg L 4 claims sev lognorm 30 cv 0.4 poisson"""

# A limit profile whose default grid windows off the origin.
WINDOWED = """port Priority.Windowed
    agg U1 500 claims sev lognorm 50 cv 1.5 poisson
    agg U2 400 claims sev lognorm 60 cv 1.4 poisson"""


@pytest.fixture(scope='module')
def two_unit():
    port = build(TWO_UNIT, update=False)
    port.update(log2=18, bs=1, padding=1)
    return port


@pytest.fixture(scope='module')
def three_unit():
    port = build(THREE_UNIT, update=False)
    port.update(log2=18, bs=1, padding=1)
    return port


@pytest.fixture(scope='module')
def hand():
    port = build(HAND, update=False)
    port.update(log2=6, bs=1, padding=1)
    return port


# ---------------------------------------------------------------------------
# The grid represents the book (guards every test below)
# ---------------------------------------------------------------------------

def test_represented_means(two_unit, three_unit):
    """The senior leg is a limited expected value and is unforgiving about
    severity discretization. If the represented means have drifted from the
    declared means, every identity below can still close while certifying a
    wrong book, so assert the means first."""
    for port, declared in ((two_unit, {'Direct': 2000.0, 'Assumed': 1080.0}),
                           (three_unit, {'Direct': 2000.0, 'Assumed': 1080.0,
                                         'Fac': 720.0})):
        for unit, want in declared.items():
            got = float(port.density_df[f'e_{unit}'].iloc[0])
            assert got == pytest.approx(want, rel=1e-5), (port.name, unit)


# ---------------------------------------------------------------------------
# Identity 1: the two-tier footing, both ways round
# ---------------------------------------------------------------------------

def test_two_unit_footing(two_unit):
    """``ex_senior_i(a) + ex_junior_j(a) == ex_total(a)``, both ways round."""
    pdf = two_unit.priority_df
    total = pdf['ex_total'].to_numpy()
    scale = total[-1]
    for senior, junior in (('Direct', 'Assumed'), ('Assumed', 'Direct')):
        resid = (pdf[f'ex_senior_{senior}'].to_numpy()
                 + pdf[f'ex_junior_{junior}'].to_numpy() - total)
        assert np.abs(resid).max() / scale < 1e-9, (senior, junior)


def test_junior_bracketed_by_the_other_two(three_unit):
    """``ex_senior >= ex_equal >= ex_junior`` pointwise, every unit.

    The tolerance is one part in ``1e6`` of each unit's expected loss, set by
    the **equal** leg: it divides kappa through by the total and carries the
    share noise, two orders of magnitude above the junior convolution's."""
    pdf = three_unit.priority_df
    for unit in three_unit.unit_names:
        tol = 1e-6 * float(three_unit.density_df[f'e_{unit}'].iloc[0])
        senior = pdf[f'ex_senior_{unit}'].to_numpy()
        equal = pdf[f'ex_equal_{unit}'].to_numpy()
        junior = pdf[f'ex_junior_{unit}'].to_numpy()
        assert (senior - equal).min() > -tol, unit
        assert (equal - junior).min() > -tol, unit


# ---------------------------------------------------------------------------
# Identity 2: the equal legs still sum to the total (regression guard)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture', ['two_unit', 'three_unit'])
def test_equal_legs_sum_to_total(fixture, request):
    """``sum_i exa_i(a) == lev_total(a)``, the pre-existing legs.

    Looser than the junior footing above by two orders of magnitude, and that
    is the pre-existing state: the equal leg divides kappa through by the
    total, so it carries the share noise the junior convolution never sees."""
    port = request.getfixturevalue(fixture)
    pdf = port.priority_df
    total = pdf['ex_total'].to_numpy()
    legs = sum(pdf[f'ex_equal_{u}'].to_numpy() for u in port.unit_names)
    assert np.abs(legs - total).max() / total[-1] < 1e-6


# ---------------------------------------------------------------------------
# Identity 3: the deficit is monotone in rank
# ---------------------------------------------------------------------------

def test_epd_monotone_in_rank(three_unit):
    """``epd`` senior <= equal <= junior, every unit, on a three-unit book."""
    epd = three_unit.priority_epd_df(p=0.99)
    for unit in three_unit.unit_names:
        senior = epd.loc[(unit, 'senior'), 'epd']
        equal = epd.loc[(unit, 'equal'), 'epd']
        junior = epd.loc[(unit, 'junior'), 'epd']
        assert senior <= equal + 1e-12 <= junior + 1e-12, unit
        assert senior < junior, unit


def test_epd_total_is_rule_invariant(two_unit):
    """All three rules distribute the same pot, so the total group repeats."""
    epd = two_unit.priority_epd_df(p=0.99)
    recoveries = epd.loc['total', 'recovery'].to_numpy()
    assert np.ptp(recoveries) == 0.0


def test_senior_leg_ignores_the_junior_book(two_unit, three_unit):
    """``min(X_i, a)`` does not involve ``X_{-i}``: the senior leg of a unit is
    the same whatever else the company writes. Compared at a common asset
    level, not a common percentile, since the books differ."""
    a = 7000.0
    two = two_unit.priority_epd_df(a)
    three = three_unit.priority_epd_df(a)
    for unit in ('Direct', 'Assumed'):
        assert (two.loc[(unit, 'senior'), 'recovery']
                == pytest.approx(three.loc[(unit, 'senior'), 'recovery'],
                                 rel=1e-9))
        # the equal and junior legs DO move, because the pool moved
        assert (two.loc[(unit, 'junior'), 'recovery']
                > three.loc[(unit, 'junior'), 'recovery'])


# ---------------------------------------------------------------------------
# Pinned to arithmetic: a discrete book whose legs are exact by hand
# ---------------------------------------------------------------------------

def test_hand_computed_legs(hand):
    """Every leg of the ``dfreq`` / ``dsev`` book at ``a = 1``, by hand."""
    pdf = hand.priority_df
    row = pdf.loc[1.0]
    assert row['ex_total'] == pytest.approx(0.75, abs=1e-12)
    assert row['ex_senior_A'] == pytest.approx(0.5, abs=1e-12)
    assert row['ex_senior_B'] == pytest.approx(0.5, abs=1e-12)
    assert row['ex_equal_A'] == pytest.approx(1 / 3, abs=1e-12)
    assert row['ex_equal_B'] == pytest.approx(5 / 12, abs=1e-12)
    assert row['ex_junior_A'] == pytest.approx(0.25, abs=1e-12)
    assert row['ex_junior_B'] == pytest.approx(0.25, abs=1e-12)


def test_hand_computed_epd(hand):
    """The deficit table off the same book: means 1/2 and 1, ratios by hand."""
    epd = hand.priority_epd_df(1.0)
    assert epd.attrs['assets'] == 1.0
    assert epd.loc[('A', 'senior'), 'mean'] == pytest.approx(0.5, abs=1e-12)
    assert epd.loc[('A', 'senior'), 'epd'] == pytest.approx(0.0, abs=1e-12)
    assert epd.loc[('A', 'equal'), 'epd'] == pytest.approx(1 / 3, abs=1e-12)
    assert epd.loc[('A', 'junior'), 'epd'] == pytest.approx(0.5, abs=1e-12)
    assert epd.loc[('B', 'senior'), 'epd'] == pytest.approx(0.5, abs=1e-12)
    assert epd.loc[('B', 'equal'), 'epd'] == pytest.approx(7 / 12, abs=1e-12)
    assert epd.loc[('B', 'junior'), 'epd'] == pytest.approx(0.75, abs=1e-12)


# ---------------------------------------------------------------------------
# Shape, caching, and the argument contract
# ---------------------------------------------------------------------------

def test_priority_df_shape_and_cache(three_unit):
    pdf = three_unit.priority_df
    assert pdf.shape[1] == 3 * len(three_unit.unit_names) + 1
    assert list(pdf.columns)[-1] == 'ex_total'
    assert three_unit.priority_df is pdf       # cached, not rebuilt


def test_priority_df_aliases_are_aliases(three_unit):
    """The senior and equal legs duplicate ``add_exa`` columns, not logic."""
    pdf = three_unit.priority_df
    df = three_unit.density_df
    for unit in three_unit.unit_names:
        assert np.array_equal(pdf[f'ex_senior_{unit}'], df[f'lev_{unit}'])
        assert np.array_equal(pdf[f'ex_equal_{unit}'], df[f'exa_{unit}'])
    assert np.array_equal(pdf['ex_total'], df['lev_total'])


def test_epd_needs_exactly_one_of_assets_or_p(two_unit):
    with pytest.raises(ValueError, match='exactly one'):
        two_unit.priority_epd_df()
    with pytest.raises(ValueError, match='exactly one'):
        two_unit.priority_epd_df(7000.0, p=0.99)


def test_junior_recovery_rejects_an_unknown_unit(two_unit):
    from aggregate import _portfolio_density
    with pytest.raises(ValueError, match='is not a unit of'):
        _portfolio_density.junior_recovery(two_unit, 'NoSuchUnit')


# ---------------------------------------------------------------------------
# The guards: signed and windowed grids raise
# ---------------------------------------------------------------------------

def test_signed_book_raises():
    """Assets and a receivership waterfall do not mean the same thing on a
    signed P&L grid, where ``exa_{unit}`` is already NaN."""
    port = build(SIGNED)
    assert float(port.density_df.index[0]) < 0
    with pytest.raises(NotImplementedError, match='zero-based'):
        port.priority_df


def test_windowed_book_raises():
    """``lev_{unit}`` is evaluated off the unit's own origin on a windowed
    book, so the shared asset level does not read across."""
    port = build(WINDOWED)
    assert float(port.density_df.index[0]) > 0
    with pytest.raises(NotImplementedError, match='zero-based'):
        port.priority_df


def test_padding_zero_raises():
    port = build(TWO_UNIT, update=False)
    port.update(log2=16, bs=4, padding=0)
    with pytest.raises(NotImplementedError, match='padding'):
        port.priority_df


def test_stale_frame_raises():
    """A frame that is not the independent combine of the units' current
    densities (a swapped or sample-based one, or a unit updated on its own
    since the combine) would read every column here against the wrong senior
    pool, so it raises rather than answering."""
    port = build(TWO_UNIT, update=False)
    port.update(log2=16, bs=4, padding=1)
    port.priority_df                              # fine before
    port._priority_df = None
    port.density_df['p_total'] = port.density_df['p_total'] * 1.001
    with pytest.raises(ValueError, match='do not reproduce'):
        port.priority_df
