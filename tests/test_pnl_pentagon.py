"""``PnL.pentagon_df``: the ledger read as a pentagon ([PnL-Pentagon]).

One row per block that books a result of its own, plus a ``Ceded`` row that is
the whole cession taken together, gross less net on every amount. The suite
pins the three things the frame has to get right:

- **the identities**, ``a == P_tech + Q``, ``M == P_tech - L``, ``CoC == M / Q``
  and ``PQ == P / Q``, on every row including the differenced one;
- **the capital**, a *marginal* quantile of each block's own result at the
  chosen level, which is the author's ruling of 2026-10-06 and is why any level
  is available rather than only the rungs of ``PERCENTILE_LADDER``. It meets
  ``waterfall_df`` where their anchors coincide;
- **the one pair no subtraction can form**, the cession's standard deviation
  and coefficient of variation, which come off the ledger's own impact row and
  decline on a stitched ledger where that row is a per-statistic delta
  ([Delta-Row-Marked]).

The DecL programs are the two-tier peels of ``test_pnl_peel.py``: ``TWO_EACH``
is stitched (two layers in each tier) and ``ONE_EACH`` is per-atom, which is
the pair that separates a real cession law from a delta of statistics.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from aggregate import build
from aggregate._pnl import WATERFALL_RETURN_PERIOD

#: Exact to numerical noise: every identity here is arithmetic over quantities
#: the frame carries, not a modelling tolerance.
EXACT = 1e-9

#: A plain single-group P&L: one block, one result, no cession to difference.
SINGLE = ('pnl PentSingle 1000 premium less agg PentSingle_e 1000 premium '
          'at 70% lr sev lognorm 100 cv 2 poisson')

#: Two layers in each tier, so the walk is stitched and the impact row has no
#: law of its own. The plan's acceptance case.
STITCHED = ('xpnl PentBoth 1000 premium less agg PentBoth_e 1000 premium at '
            '70% lr sev lognorm 100 cv 2 '
            'occurrence net of 100 xs 100 deposit 60 cede 0.2 and '
            '300 xs 200 deposit 40 poisson '
            'aggregate net of 100 xs 300 deposit 25 and 200 xs 400 deposit 15 '
            'peel top-down')

#: One layer in each tier, so the peel stays per-atom and every row, the impact
#: included, carries a distribution.
PER_ATOM = ('xpnl PentOne 1000 premium less agg PentOne_e 1000 premium at 70% '
            'lr sev lognorm 100 cv 2 '
            'occurrence net of 500 xs 500 deposit 100 poisson '
            'aggregate net of 200 xs 400 deposit 40 peel top-down')


@pytest.fixture(scope='module')
def stitched():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return build(STITCHED)


@pytest.fixture(scope='module')
def per_atom():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return build(PER_ATOM)


# ----------------------------------------------------------------------
# Shape
# ----------------------------------------------------------------------
def test_rows_mirror_the_walk_and_close_with_the_cession(stitched):
    """The frame is the walk's rows in the walk's order, then ``Ceded``."""
    pent = stitched.pentagon_df()
    assert list(pent.index) == list(stitched.waterfall_df.index) + ['Ceded']
    assert pent.index.name == 'Step'
    assert pent.index[0] == 'Gross', 'the gross book leads in every builder'
    assert pent.index[-2] == 'All', 'the closing net position precedes Ceded'


def test_a_single_group_pnl_is_one_row(stitched):
    """One block, one result, nothing to cede.

    The walk declines on this shape, having nothing to walk, so the pairing of
    the one block to its one result row is the pentagon's own
    (:meth:`PnL._block_result_rows`).
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pent = build(SINGLE).pentagon_df()
    assert len(pent) == 1
    assert 'Ceded' not in pent.index
    assert pent['Q'].notna().all()


def test_the_level_rides_in_attrs_not_in_a_column_name(stitched):
    assert stitched.pentagon_df().attrs['return_period'] \
        == WATERFALL_RETURN_PERIOD
    assert stitched.pentagon_df(t=250).attrs['return_period'] == 250
    assert not any('100' in c or '250' in c
                   for c in stitched.pentagon_df().columns)


def test_a_return_period_at_or_below_one_names_no_state(stitched):
    for bad in (1, 0.5, 0):
        with pytest.raises(ValueError, match='return period above 1'):
            stitched.pentagon_df(t=bad)


# ----------------------------------------------------------------------
# The identities: the whole point of closing a pentagon
# ----------------------------------------------------------------------
@pytest.mark.parametrize('t', [100, 250, 2000])
def test_the_pentagon_closes_on_every_row(stitched, t):
    """``a = P_tech + Q`` and ``M = P_tech - L``, at any level.

    The second is the identity that makes the frame nearly free: with premium
    and expense fixed on a block, its result is the loss shifted, so the
    ledger's own margin and the pentagon's ``P_tech - L`` are the same number.
    """
    for _step, r in stitched.pentagon_df(t=t).iterrows():
        assert r['a'] == pytest.approx(r['P_tech'] + r['Q'], abs=EXACT)
        assert r['M'] == pytest.approx(r['P_tech'] - r['L'], abs=EXACT)
        assert r['P_tech'] == pytest.approx(r['P'] - r['E'], abs=EXACT)


def test_the_ratios_are_the_quotients_they_claim_to_be(stitched):
    """Each ratio against its own denominator: written for the three the
    ledger already reports, technical for ``TLR``, capital for the last two."""
    for _step, r in stitched.pentagon_df().iterrows():
        assert r['LR'] == pytest.approx(r['L'] / r['P'], abs=EXACT)
        assert r['ER'] == pytest.approx(r['E'] / r['P'], abs=EXACT)
        assert r['CR'] == pytest.approx((r['L'] + r['E']) / r['P'], abs=EXACT)
        assert r['TLR'] == pytest.approx(r['L'] / r['P_tech'], abs=EXACT)
        assert r['PQ'] == pytest.approx(r['P'] / r['Q'], abs=EXACT)
        assert r['CoC'] == pytest.approx(r['M'] / r['Q'], abs=EXACT)


def test_the_amounts_are_the_ledgers_own(stitched):
    """Nothing is newly estimated: the four amounts are the ratio frame's."""
    ratios, pent = stitched.economic_ratios_df, stitched.pentagon_df()
    for step in pent.index.drop('Ceded'):
        for column in ('L', 'E', 'P', 'M', 'SD'):
            assert pent.loc[step, column] \
                == pytest.approx(ratios.loc[step, column], abs=EXACT)


# ----------------------------------------------------------------------
# The capital, and where it meets the walk
# ----------------------------------------------------------------------
def test_the_capital_meets_the_walk_at_its_anchors(stitched):
    """The two frames agree exactly where their anchors coincide.

    ``Capital standalone`` is each row's own quantile, read two-sided by role,
    which is what the pentagon takes on every row. The conditional columns
    coincide with it only at their anchors: the gross cell on the ``Gross`` row
    and the net cell on the closing row ([Waterfall-Gross-Basis]).
    """
    pent, walk = stitched.pentagon_df(), stitched.waterfall_df
    for step in walk.index:
        assert pent.loc[step, 'Q'] \
            == pytest.approx(walk.loc[step, 'Capital standalone'], abs=EXACT)
        assert pent.loc[step, 'CoC'] \
            == pytest.approx(walk.loc[step, 'CoC standalone'], abs=EXACT)
    assert pent.loc['Gross', 'Q'] \
        == pytest.approx(walk.loc['Gross', 'Capital gross'], abs=EXACT)
    assert pent.loc['All', 'Q'] \
        == pytest.approx(walk.loc['All', 'Capital net'], abs=EXACT)


def test_a_cession_reads_its_capital_as_released(stitched):
    """A cover holds no capital, it hands it back, so the column is negative.

    The two-sided reading of [Writer-Standalone]: a ceded step is read in its
    own right tail, the state in which it pays most, and the sign is the marker
    of which reading the row took.
    """
    pent = stitched.pentagon_df()
    assert pent.loc['Gross', 'Q'] > 0
    assert pent.loc['All', 'Q'] > 0
    for step in ('All occurrence', 'All aggregate', 'Ceded'):
        assert pent.loc[step, 'Q'] < 0, step


def test_a_higher_level_calls_for_more_capital_and_nothing_else(stitched):
    """What the solvency level moves, and what it must not.

    Acceptance 5 of the plan: assets, capital, leverage and the cost of capital
    all move with the level; loss, expense, premium, margin and the spread do
    not, being properties of the book rather than of the standard applied to
    it.
    """
    base, raised = stitched.pentagon_df(), stitched.pentagon_df(t=2000)
    for column in ('L', 'E', 'P', 'P_tech', 'M', 'LR', 'TLR', 'ER', 'CR',
                   'SD', 'CV'):
        assert np.allclose(base[column], raised[column],
                           equal_nan=True, atol=EXACT), column
    for step in ('Gross', 'All', 'Ceded'):
        for column in ('Q', 'a', 'PQ', 'CoC'):
            assert raised.loc[step, column] != base.loc[step, column], \
                f'{step} {column}'
    # a risk-bearing block holds more capital against a remoter standard, and
    # earns less on it
    for step in ('Gross', 'All'):
        assert raised.loc[step, 'Q'] > base.loc[step, 'Q'], step
        assert raised.loc[step, 'CoC'] < base.loc[step, 'CoC'], step


def test_an_exhausted_program_releases_less_capital_further_out(stitched):
    """The cession is **not** monotone in the level, and that is the reading.

    Released capital is ``Q`` net less ``Q`` gross, and every cover here is
    bounded. Far enough into the tail the program is exhausted, so the net
    capital grows almost as fast as the gross and the difference between them
    closes: the cession releases 956 at 1-in-100 and only 908 at 1-in-2000.
    The margin given up is fixed, so the price per unit released rises.

    A reader who expects the strip's figures to grow with the level will be
    surprised here, which is the point of letting them see all six.
    """
    base, raised = stitched.pentagon_df(), stitched.pentagon_df(t=2000)
    assert abs(raised.loc['Ceded', 'Q']) < abs(base.loc['Ceded', 'Q'])
    assert abs(raised.loc['Ceded', 'CoC']) > abs(base.loc['Ceded', 'CoC'])


# ----------------------------------------------------------------------
# The Ceded row: a difference, not a position
# ----------------------------------------------------------------------
def test_every_ceded_amount_is_the_difference(stitched):
    """Gross less net on every amount, which is what makes the figure work.

    A node's area is the gross amount and the cession is the part left over, so
    the row has to be a difference by construction rather than a position read
    off the ledger.
    """
    pent = stitched.pentagon_df()
    gross, net, ceded = pent.loc['Gross'], pent.loc['All'], pent.loc['Ceded']
    for column in ('L', 'E', 'P', 'P_tech', 'M', 'a', 'Q'):
        assert ceded[column] == pytest.approx(net[column] - gross[column],
                                              abs=EXACT), column


def test_the_ceded_capital_is_not_its_own_quantile(stitched):
    """Released capital is a difference of two tail readings, and tail measures
    do not add, so the row agrees with neither the ceded steps summed nor any
    quantile of its own."""
    pent = stitched.pentagon_df()
    steps = pent.loc[['All occurrence', 'All aggregate'], 'Q'].sum()
    assert pent.loc['Ceded', 'Q'] != pytest.approx(steps, abs=1.0)


def test_the_cession_has_no_spread_on_a_stitched_ledger(stitched):
    """The one pair no subtraction can form, declining rather than guessing.

    The cession and the gross book ride different marginals here, so their
    difference has no distribution. Differencing the two standard deviations is
    what used to put a negative one on the sheet ([Delta-Row-Marked]).
    """
    assert stitched._stitched
    pent = stitched.pentagon_df()
    assert np.isnan(pent.loc['Ceded', 'SD'])
    assert np.isnan(pent.loc['Ceded', 'CV'])
    # every other row keeps its own, the marginals being real
    assert pent['SD'].drop('Ceded').notna().all()


def test_the_cession_keeps_its_spread_where_the_atoms_carry_the_joint(per_atom):
    """With real atoms the impact row has a law, so the pair is genuine.

    Read off that row rather than recomputed, so the frame and the ledger
    cannot disagree.
    """
    assert not per_atom._stitched
    pent = per_atom.pentagon_df()
    impact = per_atom.economic_df.loc[('All', 'Margin', 'Impact')]
    assert pent.loc['Ceded', 'SD'] == pytest.approx(float(impact['SD']),
                                                    abs=EXACT)
    assert pent.loc['Ceded', 'CV'] == pytest.approx(
        pent.loc['Ceded', 'SD'] / pent.loc['Ceded', 'L'], abs=EXACT)
    assert pent.loc['Ceded', 'SD'] > 0
