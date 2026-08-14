"""Tests for the pricing result objects ([Pricing-Result-Objects], 1.0.0a259).

``calibrate_distortions`` and ``evaluate`` used to return a bare
``DataFrame``. A frame cannot be dispatched on, and the pricing exhibits
dispatch on the result (``dev/plan-pricing-exhibits.md``,
``[Pricing-Keyed-On-Result]``), so both now return a dataclass declared in
``aggregate.results``.

Two properties are worth holding onto and both are asserted here.

* **The stored attributes did not move.** ``obj.distortions`` /
  ``obj.distortion_df`` / ``obj.calibration_df`` are set exactly as before, so
  the break is confined to code that used the *return value* as a frame.
* **The allocation frames are real public frames**, computed on demand and
  cached on the result. That is what keeps the RAW exhibit invariant intact
  over an exhibit built on a calculation rather than on stored state.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from aggregate import build
from aggregate.results import (
    AnalyzeDistortionResult, AnalyzeDistortionsResult, CalibrationResult,
    EvaluationResult, PricingResult,
)

CLEAN = 'agg PR.Clean 250 claims sev lognorm 100 cv 1.5 poisson'
CEDED = ('agg PR.Re 250 claims sev lognorm 100 cv 1.5 '
         'occurrence net of 276 xs 55 poisson')
BOOK = ('port PR.Book '
        'agg PR.UnitA as "Unit Alpha" 1 claim sev gamma 100 cv 0.5 fixed '
        'agg PR.UnitB 1 claim sev gamma 50 cv 0.8 fixed')
PRICED = 'agg PR.Priced 1000 premium at 0.7 lr sev gamma 100 cv 0.5 poisson'


@pytest.fixture(scope='module')
def clean():
    return build(CLEAN)


@pytest.fixture(scope='module')
def ceded():
    return build(CEDED)


@pytest.fixture(scope='module')
def book():
    return build(BOOK)


# --- CalibrationResult ------------------------------------------------------

def test_calibration_returns_a_result_carrying_its_inputs(clean):
    result = clean.calibrate_distortions(0.15, p=0.99)
    assert isinstance(result, CalibrationResult)
    assert result.coc == pytest.approx(0.15)
    assert result.p == pytest.approx(0.99)
    assert result.a == pytest.approx(clean.q(0.99))
    assert result.anchor == 'p'
    assert result.names == ('ccoc', 'ph', 'wang', 'dual', 'tvar')
    assert result.reins_view is None
    assert result._source is clean


def test_calibration_anchor_records_which_keyword_the_caller_used(clean):
    """``p=`` holds the threshold fixed, ``a=`` the capital. Which one the
    caller named is what a derived frame has to re-anchor on."""
    a = float(clean.q(0.99))
    on_p = clean.calibrate_distortions(0.15, p=0.99)
    on_a = clean.calibrate_distortions(0.15, a=a)
    assert on_p.anchor == 'p' and on_p._anchor_kwargs == {'p': 0.99}
    assert on_a.anchor == 'a' and on_a._anchor_kwargs == {'a': a}
    # the pair is resolved either way, and resolves to the same place
    assert on_a.p == pytest.approx(clean.cdf(a))
    assert on_p.a == pytest.approx(on_a.a)


def test_the_stored_attributes_did_not_move(clean):
    """The break is confined to the return value; the object is unchanged."""
    result = clean.calibrate_distortions(0.15, p=0.99)
    assert clean.distortions is result.distortions
    pd.testing.assert_frame_equal(clean.distortion_df, result.distortion_df)
    pd.testing.assert_frame_equal(clean.calibration_df, result.calibration_df)


def test_a_result_borrows_its_sources_identity(book):
    result = book.calibrate_distortions(0.15, p=0.99)
    assert result.name == book.name
    assert result.label == book.label
    assert result._title_name == book._title_name


def test_a_sourceless_result_answers_none_rather_than_raising():
    """Losing the title is not worth an exception; the frames are the point."""
    orphan = EvaluationResult(evaluation_df=pd.DataFrame({'x': [1]}))
    assert orphan.name is None and orphan.label is None
    assert orphan._title_name == 'EvaluationResult'
    frame = pd.DataFrame({'x': [1]})
    assert orphan._relabel(frame) is frame


def test_portfolio_calibration_allocates_across_units(book):
    result = book.calibrate_distortions(0.15, p=0.99)
    allocated = result.pricing_df
    assert list(allocated.index.names) == ['distortion', 'stat']
    # units across, with the book's labels applied
    assert 'Unit Alpha' in allocated.columns
    assert 'total' in allocated.columns
    # cached: the second read is the same object, not a second sweep
    assert result.pricing_df is allocated


def test_a_mass_family_allocates_on_an_unbounded_book(book):
    """[Allocation-Default-Linear]: ccoc reaches the Allocate subtab.

    The symptom this was drafted from: Calibrate at ``p < 1`` on an unbounded
    book warned ``skipping ccoc`` and served an allocation with no ccoc row,
    while Calibrate and Evaluate both carried it. The sweep priced through a
    hardcoded lifted default while :attr:`allocation_method` said linear, and
    lifted genuinely cannot split that tail. Reading the member fixes it.
    """
    assert not book.bounded
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = book.calibrate_distortions(0.10, p=0.99)
        allocated = result.pricing_df
    assert not [w for w in caught if 'skipping' in str(w.message)]
    families = set(allocated.index.get_level_values('distortion')
                   .unique().dropna())
    assert 'ccoc' in families
    ccoc = allocated.xs('ccoc', level='distortion')
    # every cell, per unit and total: the frame is built, not blanked
    assert np.isfinite(ccoc.to_numpy()).all()
    # and the total column closes back on the shared calibration target,
    # which is the round trip the app draws across its two subtabs
    target = result.calibration_df.loc['calibration']
    for stat in ('L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE'):
        assert ccoc.loc[stat, 'total'] == pytest.approx(target[stat], rel=1e-6)


def test_reinsured_aggregate_calibration_allocates_across_views(ceded):
    result = ceded.calibrate_distortions(0.15, p=0.99)
    priced = result.reins_price_df
    assert list(priced.index.names) == ['distortion', 'view']
    assert set(priced.index.get_level_values('view')) == set(ceded.reins_views)
    assert result.reins_price_df is priced


def test_an_allocation_with_nothing_to_spread_says_so(clean):
    """No units and no views: the allocation story is the one calibration row."""
    result = clean.calibrate_distortions(0.15, p=0.99)
    with pytest.raises(AttributeError, match='no units to allocate'):
        result.pricing_df
    with pytest.raises(AttributeError, match='carries no cession'):
        result.reins_price_df


def test_the_allocation_frames_are_public_frames_of_the_result(book, ceded):
    """The RAW exhibit invariant, stated over a result.

    A RAW block names an attribute on the dispatched object and serves the
    frame it returns. Computing that frame on demand is an efficiency
    question; whether it is a real public attribute is the contract question,
    and it is.
    """
    for result, attr in ((book.calibrate_distortions(0.15, p=0.99),
                          'pricing_df'),
                         (ceded.calibrate_distortions(0.15, p=0.99),
                          'reins_price_df')):
        assert hasattr(result, attr)
        assert isinstance(getattr(result, attr), pd.DataFrame)


# --- stand_alone_df ([Portfolio-Standalone-Frame], 1.0.0a282) ---------------

def test_stand_alone_prices_every_unit_at_the_books_asset_level(book):
    """The anchor is the total's, once: nothing re-anchors per unit."""
    result = book.calibrate_distortions(0.15, p=0.99)
    df = result.stand_alone_df
    assert df.index.names == ['distortion', 'unit']
    assert df['a'].nunique() == 1
    assert df['a'].iloc[0] == pytest.approx(result.a)
    # every row is a pentagon at that level, the derived rows included
    assert (df['P'] + df['Q']).to_numpy() == pytest.approx(
        df['a'].to_numpy())


def test_stand_alone_rows_are_the_units_then_the_two_derived(book):
    result = book.calibrate_distortions(0.15, p=0.99)
    df = result.stand_alone_df
    for family in df.index.get_level_values('distortion').unique():
        rows = list(df.xs(family, level='distortion').index)
        assert rows == [*book.unit_names, 'sum of parts', 'total']


def test_a_unit_row_is_that_unit_priced_alone(book):
    """Tied to ``Distortion.price`` called directly, which is what it claims."""
    result = book.calibrate_distortions(0.15, p=0.99)
    df = result.stand_alone_df
    for family, dist in result.distortions.items():
        for unit in book.agg_list:
            quote = dist.price(unit.density_df['p_total'], a=result.a,
                               kind='ask')
            row = df.loc[(family, unit.name)]
            assert row['L'] == pytest.approx(quote.el)
            assert row['P'] == pytest.approx(quote.ask)
            assert row['M'] == pytest.approx(quote.ask - quote.el)


def test_the_total_row_is_the_families_fitted_premium(book):
    """Which is the calibration target plus that family's own miss."""
    result = book.calibrate_distortions(0.15, p=0.99)
    df = result.stand_alone_df
    target = float(result.calibration_df.loc['calibration', 'P'])
    for family in result.distortions:
        fitted = target + float(result.distortion_df.loc[family, 'error'])
        assert df.loc[(family, 'total'), 'P'] == pytest.approx(fitted)


def test_sum_of_parts_is_at_or_above_the_total(book):
    """Sub-additivity, which is what makes the comparison the point of the frame.

    ``min(X, a) <= sum_i min(X_i, a)`` pointwise, so monotonicity and the
    sub-additivity of a concave distortion compose. The gap is what pooling is
    worth under that family.
    """
    result = book.calibrate_distortions(0.15, p=0.99)
    df = result.stand_alone_df
    for family in result.distortions:
        block = df.xs(family, level='distortion')
        assert block.loc['sum of parts', 'P'] >= block.loc['total', 'P'] - 1e-9
        assert block.loc['sum of parts', 'L'] >= block.loc['total', 'L'] - 1e-9


def test_sum_of_parts_adds_the_amounts_and_holds_the_asset_level(book):
    """Author's ruling, 2026-08-14: the sum row is a pentagon at the same ``a``.

    Summing the capital column instead would put the row at ``n`` times the
    asset level, which is a different reading and not the one the comparison
    against ``total`` needs.
    """
    result = book.calibrate_distortions(0.15, p=0.99)
    df = result.stand_alone_df
    units = book.unit_names
    for family in result.distortions:
        block = df.xs(family, level='distortion')
        for stat in ('L', 'M', 'P'):
            assert block.loc['sum of parts', stat] == pytest.approx(
                sum(block.loc[u, stat] for u in units))
        assert block.loc['sum of parts', 'Q'] == pytest.approx(
            result.a - block.loc['sum of parts', 'P'])


def test_a_and_p_calibrations_agree_at_the_same_level(book):
    """No per unit anchoring rule, so there is no corner where they differ."""
    by_p = book.calibrate_distortions(0.15, p=0.99)
    by_a = book.calibrate_distortions(0.15, a=by_p.a)
    pd.testing.assert_frame_equal(by_p.stand_alone_df, by_a.stand_alone_df)


def test_stand_alone_is_cached(book):
    result = book.calibrate_distortions(0.15, p=0.99)
    assert result.stand_alone_df is result.stand_alone_df


def test_stand_alone_needs_units(clean, ceded):
    """An aggregate has one part, which is the whole; there is nothing to sum."""
    for obj in (clean, ceded):
        result = obj.calibrate_distortions(0.15, p=0.99)
        with pytest.raises(AttributeError, match='no units to price'):
            result.stand_alone_df


# --- natural_allocation_df ([Calibration-Natural-Allocation-Frame], a283) ---

@pytest.fixture(scope='module')
def gross_calibration(ceded):
    return ceded.calibrate_distortions(0.15, p=0.99, reins_view='gross')


def test_ceded_and_net_foot_to_gross(gross_calibration):
    """One premium decomposed, so the rows add up. That is the whole point."""
    df = gross_calibration.natural_allocation_df
    assert df.index.names == ['distortion', 'view']
    for family in gross_calibration.distortions:
        block = df.xs(family, level='distortion')
        assert list(block.index) == ['gross', 'ceded', 'net']
        assert block.loc['ceded', 'P'] + block.loc['net', 'P'] == pytest.approx(
            block.loc['gross', 'P'], rel=1e-12)
        assert block.loc['ceded', 'L'] + block.loc['net', 'L'] == pytest.approx(
            block.loc['gross', 'L'], rel=1e-12)


def test_the_gross_row_is_the_one_calibrated_premium(gross_calibration):
    """Constant down the table: one market premium, five sets of fractions.

    Each family's own fitted premium (target plus its ``error``) would be the
    other choice and is the wrong one here: the families differ in how they
    split a premium, not in what it is.
    """
    df = gross_calibration.natural_allocation_df
    target = float(
        gross_calibration.calibration_df.loc['calibration', 'P'])
    gross = df.xs('gross', level='view')['P']
    assert gross.nunique() == 1
    assert float(gross.iloc[0]) == pytest.approx(target, rel=1e-12)


def test_the_reading_is_unlimited(gross_calibration):
    """No asset level, so no capital and no ratio against it."""
    df = gross_calibration.natural_allocation_df
    assert (df['a'] == np.inf).all()
    for stat in ('Q', 'PQ', 'ROE'):
        assert df[stat].isna().all()


def test_the_grid_gap_is_reported_not_absorbed(gross_calibration):
    """``rho_gap`` per family, in attrs, because it is a judgment for the reader.

    The joint's gross marginal is a coarser rebucketed cousin of the fine 1-D
    density the fit was struck on, so the two do not price to the bit. The
    fractions come off the joint and the level does not, and the distance
    between the two readings travels with the frame.
    """
    df = gross_calibration.natural_allocation_df
    gaps = df.attrs['rho_gap']
    assert set(gaps) == set(gross_calibration.distortions)
    assert all(np.isfinite(v) for v in gaps.values())
    target = float(
        gross_calibration.calibration_df.loc['calibration', 'P'])
    # small for the concave families on this program
    for family in ('ph', 'wang', 'dual', 'tvar'):
        assert abs(gaps[family]) < 0.01 * target
    # and large for ccoc, which is the a274 pathology showing up rather than
    # hiding: a mass at zero family read unlimited charges the top grid bucket
    assert abs(gaps['ccoc']) > 10 * abs(gaps['ph'])


def test_the_realized_sizing_travels_with_the_frame(gross_calibration):
    """A priced exhibit must not be readable without the grid it was priced on."""
    sizing = gross_calibration.natural_allocation_df.attrs['joint_sizing']
    assert sizing['bs'] > 0
    assert len(sizing['log2']) == 2
    assert sizing['deficit'] < 1e-5
    assert isinstance(sizing['exact_lattice'], bool)


def test_allocation_needs_a_gross_basis(ceded):
    """A set fitted to net has no gross premium to split. Structural, not taste."""
    on_net = ceded.calibrate_distortions(0.15, p=0.99)
    with pytest.raises(ValueError, match='calibrated on'):
        on_net.natural_allocation_df
    on_ceded = ceded.calibrate_distortions(0.15, p=0.99, reins_view='ceded')
    with pytest.raises(ValueError, match='gross'):
        on_ceded.natural_allocation_df


def test_allocation_needs_an_occurrence_program(clean):
    result = clean.calibrate_distortions(0.15, p=0.99)
    with pytest.raises(ValueError, match='no occurrence program'):
        result.natural_allocation_df


def test_the_allocation_frame_is_cached(gross_calibration):
    assert (gross_calibration.natural_allocation_df
            is gross_calibration.natural_allocation_df)


def test_the_joint_is_built_once_and_held(ceded):
    """Decision 6: the frame and the chart read one joint, not two.

    A 2-D FFT is not something to pay for twice because two surfaces asked the
    same question of one object.
    """
    assert ceded.occ_joint() is ceded.occ_joint()
    assert ceded.occ_joint(views=('gross', 'net')) is not ceded.occ_joint()
    assert ceded.occ_joint(total_log2=18) is not ceded.occ_joint()


def test_the_held_joint_does_not_survive_an_update():
    """A re-update is an honest rebuild; a stale hit would price on the old grid.

    Re-run at the same grid rather than at a new one, because
    ``update(log2=...)`` on a reinsured aggregate raises for an unrelated
    reason (the cession is re-applied against a severity density from the
    previous grid). Identity is what is being checked here either way.
    """
    a = build(CEDED)
    first = a.occ_joint()
    a.update()
    assert a.occ_joint() is not first


# --- EvaluationResult -------------------------------------------------------

def test_aggregate_evaluate_returns_a_result():
    a = build(PRICED)
    result = a.evaluate()
    assert isinstance(result, EvaluationResult)
    assert result.premium == pytest.approx(1000.0)
    assert result.reins_view is None
    assert result._source is a
    assert list(result.evaluation_df.columns) == [
        'role', 'param_name', 'param', 'gini_p', 'error', 'status']


def test_portfolio_evaluate_carries_one_premium_only_for_one_position():
    port = build('port PR.Ev '
                 'agg PR.E1 1000 premium at 0.65 lr sev gamma 100 cv 0.5 poisson '
                 'agg PR.E2 500 premium at 0.75 lr sev gamma 50 cv 1.2 poisson')
    total = port.evaluate()
    assert total.premium == pytest.approx(1500.0)
    profile = port.evaluate(unit=['PR.E1', 'PR.E2'])
    # a profile measures two positions; no single premium stands for them
    assert profile.premium is None
    assert list(profile.evaluation_df.index.get_level_values('Step').unique()) \
        == ['PR.E1', 'PR.E2']


def test_pnl_evaluate_returns_a_result_with_no_single_premium():
    p = build('pnl PR.Ledger 1000 prem less agg PR.LedgerE 850 loss '
              'sev lognorm 100 cv 1 poisson')
    result = p.evaluate()
    assert isinstance(result, EvaluationResult)
    assert result.premium is None      # every row carries its own
    assert result._source is p
    assert result.name == p.name


# --- the three older results ------------------------------------------------

def test_the_older_results_carry_their_source_too(book):
    book.calibrate_distortions(0.15, p=0.99)
    priced = book.price(0.99, book.distortions['ph'])
    assert isinstance(priced, PricingResult) and priced._source is book
    # and PricingResult now relabels its unit level, as its two siblings do
    assert 'Unit Alpha' in priced.df.index.get_level_values('unit')

    one = book.analyze_distortion(book.distortions['ph'], p=0.99)
    assert isinstance(one, AnalyzeDistortionResult) and one._source is book

    many = book.analyze_distortions(p=0.99)
    assert isinstance(many, AnalyzeDistortionsResult) and many._source is book
    assert many.name == book.name
