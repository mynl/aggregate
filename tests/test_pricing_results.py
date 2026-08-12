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
