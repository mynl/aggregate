"""Tests for ``reins_view=`` on the pricing surface (1.0.0a223).

Covers phase A of ``dev/plan-loss-lab-round-3.md`` ([Loss-Lab-Round-3]). A
cession gives an object three to five distributions worth pricing and the
object holds exactly one of them, so every entry point that read ``self``'s
density silently answered about that one. The keyword names the others.

- :attr:`Aggregate.reins_views` / :attr:`Portfolio.reins_views` -- the accepted
  set, which is object dependent, so a caller has to be able to ask.
- ``ceded`` and ``net`` resolve **end to end**: on an occurrence-only program
  they are the occurrence stage, not the empty aggregate-stage columns.
- ``calibrate_distortions`` and ``evaluate`` on any view;
  ``analyze_distortions`` on net alone, refusing the two it cannot allocate.
- The refusals themselves, which are the safety property: an unknown view, a
  view the object does not carry, and an object with no cession at all.

The DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg``
(section RV).
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build


OCC = ('agg RV.Occ 100 claims sev lognorm 50 cv 2 '
       'occurrence net of 100 xs 100 poisson')
BOTH = ('agg RV.Both 100 claims sev lognorm 50 cv 2 '
        'occurrence net of 100 xs 100 poisson aggregate net of 1000 xs 4000')
CEDED = ('agg RV.Ceded 100 claims sev lognorm 50 cv 2 '
         'occurrence ceded to 100 xs 100 poisson')
CLEAN = 'agg RV.Clean 30 claims sev lognorm 40 cv 1.2 poisson'
BOOK = """port RV.Book
    agg RV.Occ 100 claims sev lognorm 50 cv 2 occurrence net of 100 xs 100 poisson
    agg RV.Clean 30 claims sev lognorm 40 cv 1.2 poisson
"""


def mean_of(density):
    """Mean of a pmf ``Series`` indexed by loss."""
    return float(np.asarray(density.index, dtype=float) @ density.to_numpy())


@pytest.fixture(scope='module')
def occ():
    return build(OCC)


@pytest.fixture(scope='module')
def both():
    return build(BOTH)


@pytest.fixture(scope='module')
def book():
    return build(BOOK)


# ----------------------------------------------------------------------------
# reins_views: the accepted set is object dependent
# ----------------------------------------------------------------------------

def test_views_one_stage(occ):
    assert occ.reins_views == ['gross', 'ceded', 'net']


def test_views_two_stages_add_the_occurrence_pair(both):
    assert both.reins_views == ['gross', 'ceded', 'net', 'ceded occ', 'net occ']


def test_views_empty_without_a_cession():
    assert build(CLEAN).reins_views == []


def test_portfolio_views_are_end_to_end_only(book):
    # a book convolves each unit's end-to-end view, so there is no
    # portfolio-wide occurrence stage to name
    assert book.reins_views == ['gross', 'ceded', 'net']


# ----------------------------------------------------------------------------
# The views themselves: end-to-end resolution, and the ceded/net split
# ----------------------------------------------------------------------------

def test_one_stage_ceded_is_the_occurrence_cession(occ):
    """Not ``p_agg_ceded``, which is a point mass at 0 with no aggregate cover.

    This is the silent-wrong-answer the keyword exists to prevent: naming the
    raw column would hand a caller asking for the cession nothing at all.
    """
    assert mean_of(occ._reins_view_density('ceded')) > 0


def test_one_stage_ceded_plus_net_is_gross(occ):
    g = mean_of(occ._reins_view_density('gross'))
    c = mean_of(occ._reins_view_density('ceded'))
    n = mean_of(occ._reins_view_density('net'))
    # the FFT grid deficit of an unlimited lognormal, nothing more
    assert c + n == pytest.approx(g, rel=1e-6)


def test_two_stage_views_chain(both):
    """occ net less the aggregate cession is the end-to-end net."""
    net_occ = mean_of(both._reins_view_density('net occ'))
    ceded = mean_of(both._reins_view_density('ceded'))
    net = mean_of(both._reins_view_density('net'))
    assert net_occ - ceded == pytest.approx(net, rel=1e-10)


def test_net_of_program_holds_its_net(occ):
    assert mean_of(occ._reins_view_density('net')) == pytest.approx(occ.est_m)


def test_ceded_to_program_holds_its_ceded():
    """The wrinkle phase A exists for: a ``ceded to`` object is the cession.

    Reading ``density_df`` and calling it the net view is right for a ``net
    of`` program and wrong for this one.
    """
    a = build(CEDED)
    assert mean_of(a._reins_view_density('ceded')) == pytest.approx(a.est_m)
    assert mean_of(a._reins_view_density('net')) != pytest.approx(a.est_m)


def test_portfolio_own_total_is_its_net_view(book):
    net = book._reins_view_density('net')
    assert np.allclose(net.to_numpy(),
                       book.density_df['p_total'].to_numpy())


# ----------------------------------------------------------------------------
# The refusals
# ----------------------------------------------------------------------------

def test_unknown_view_names_the_alternatives(both):
    with pytest.raises(ValueError, match='ceded occ'):
        both._reins_view_density('nonsense')


def test_view_the_object_does_not_carry_is_refused(occ):
    with pytest.raises(ValueError, match="unknown reins_view 'net occ'"):
        occ._reins_view_density('net occ')


def test_no_cession_refuses_every_view():
    with pytest.raises(ValueError, match='carries no reinsurance'):
        build(CLEAN)._reins_view_density('gross')


def test_signed_support_refuses_a_view():
    a = build('agg RV.Signed 10 claims ssev 100 - lognorm 80 cv 0.2 poisson')
    with pytest.raises((NotImplementedError, ValueError)):
        a.calibrate_distortions(0.1, p=0.99, reins_view='gross')


# ----------------------------------------------------------------------------
# calibrate_distortions
# ----------------------------------------------------------------------------

def test_calibration_default_is_unchanged(occ):
    own = occ.calibrate_distortions(0.10, p=0.999)
    net = occ.calibrate_distortions(0.10, p=0.999, reins_view='net')
    assert np.allclose(own['param'], net['param'])


def test_calibration_on_gross_differs_from_net(occ):
    """The plan's acceptance test: different distributions, different shapes."""
    net = occ.calibrate_distortions(0.10, p=0.999, reins_view='net')['param']
    gross = occ.calibrate_distortions(0.10, p=0.999, reins_view='gross')['param']
    # ccoc is a cost of capital, identical by construction; the shaped
    # families must move
    assert not np.allclose(net.drop('ccoc'), gross.drop('ccoc'))


def test_calibration_records_its_view(occ):
    df = occ.calibrate_distortions(0.10, p=0.999, reins_view='gross')
    assert df.attrs['reins_view'] == 'gross'
    assert occ.calibration_df.attrs['reins_view'] == 'gross'
    occ.calibrate_distortions(0.10, p=0.999)
    assert occ.distortion_df.attrs['reins_view'] is None


def test_calibration_asset_level_follows_the_view(occ):
    """``p=`` holds the threshold fixed, so the gross view needs more assets."""
    occ.calibrate_distortions(0.10, p=0.999, reins_view='gross')
    a_gross = float(occ.calibration_df['a'].iloc[0])
    occ.calibrate_distortions(0.10, p=0.999, reins_view='net')
    a_net = float(occ.calibration_df['a'].iloc[0])
    assert a_gross > a_net


def test_portfolio_calibration_on_gross(book):
    own = book.calibrate_distortions(0.10, p=0.999)['param']
    gross = book.calibrate_distortions(0.10, p=0.999, reins_view='gross')['param']
    assert not np.allclose(own.drop('ccoc'), gross.drop('ccoc'))


# ----------------------------------------------------------------------------
# analyze_distortions: net only, and it says so
# ----------------------------------------------------------------------------

def test_analyze_on_net_matches_the_default(book):
    book.calibrate_distortions(0.10, p=0.999)
    base = book.analyze_distortions(p=0.999).pricing_df
    net = book.analyze_distortions(p=0.999, reins_view='net').pricing_df
    assert np.allclose(base.to_numpy(), net.to_numpy(), equal_nan=True)


@pytest.mark.parametrize('view', ['gross', 'ceded'])
def test_analyze_refuses_what_it_cannot_allocate(book, view):
    book.calibrate_distortions(0.10, p=0.999)
    with pytest.raises(NotImplementedError, match='twin portfolio'):
        book.analyze_distortions(p=0.999, reins_view=view)


def test_analyze_refuses_an_unknown_view(book):
    book.calibrate_distortions(0.10, p=0.999)
    with pytest.raises(ValueError, match='unknown reins_view'):
        book.analyze_distortions(p=0.999, reins_view='net occ')


# ----------------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------------

def test_evaluate_labels_the_step_with_its_view(occ):
    panel = occ.evaluate(6000, reins_view='gross')
    assert panel.index.get_level_values('Step').unique().tolist() == \
        ['RV.Occ gross']


def test_evaluate_views_concatenate(occ):
    import pandas as pd

    both_panels = pd.concat([occ.evaluate(6000),
                             occ.evaluate(6000, reins_view='gross')])
    assert both_panels.index.get_level_values('Step').nunique() == 2


def test_evaluate_gross_is_less_acceptable_than_net(occ):
    """Same premium, more loss: the gross position survives less stress."""
    net = occ.evaluate(6000, reins_view='net')['gini_p']
    gross = occ.evaluate(6000, reins_view='gross')['gini_p']
    assert (gross.to_numpy() < net.to_numpy()).all()


def test_portfolio_evaluate_passes_the_view_to_units(book):
    with pytest.raises(ValueError, match='RV.Clean carries no reinsurance'):
        book.evaluate([3000, 1500], unit=['RV.Occ', 'RV.Clean'],
                      reins_view='gross')


# ----------------------------------------------------------------------------
# reins_price_df: a distortion-priced cession (phase B)
# ----------------------------------------------------------------------------

@pytest.fixture(scope='module')
def priced(occ):
    occ.calibrate_distortions(0.10, p=0.999)
    return occ.reins_price_df()


def test_price_df_shape(priced, occ):
    assert list(priced.columns) == ['a', 'el', 'bid', 'ask', 'margin']
    assert priced.index.names == ['distortion', 'view']
    assert priced.index.get_level_values('view').unique().tolist() == \
        occ.reins_views


def test_price_df_ties_to_reins_stats(priced, occ):
    """The plan's acceptance test: the expected loss leg ties to the moments."""
    stats = occ.reins_stats_df
    ceded_mean = float(stats.loc[('agg', 'mean'), ('occ', 'Ceded')])
    priced_el = priced.loc[('wang', 'ceded'), 'el']
    assert priced_el == pytest.approx(ceded_mean, rel=1e-9)


def test_price_df_margin_is_ask_less_el(priced):
    assert np.allclose(priced['margin'], priced['ask'] - priced['el'])


def test_price_df_bid_below_el_below_ask(priced):
    # ccoc at a=inf on an unbounded support prices the grid ceiling, so it is
    # excluded here; the documented reason it wants a finite asset level
    shaped = priced.drop('ccoc', level='distortion')
    assert (shaped['bid'] <= shaped['el']).all()
    assert (shaped['el'] <= shaped['ask']).all()


def test_price_df_unlimited_el_is_the_mean(priced, occ):
    # ``forwards`` parks the grid deficit at the largest represented outcome,
    # so the priced el sits a touch above the raw first moment of the pmf
    assert priced.loc[('wang', 'gross'), 'el'] == \
        pytest.approx(mean_of(occ._reins_view_density('gross')), rel=1e-6)
    assert np.isinf(priced.loc[('wang', 'gross'), 'a'])


def test_price_df_at_p_uses_each_view_own_assets(occ):
    occ.calibrate_distortions(0.10, p=0.999)
    df = occ.reins_price_df('wang', p=0.999)
    assert df.loc[('wang', 'gross'), 'a'] > df.loc[('wang', 'net'), 'a']
    assert df.loc[('wang', 'net'), 'a'] > df.loc[('wang', 'ceded'), 'a']


def test_price_df_accepts_one_named_distortion(occ):
    occ.calibrate_distortions(0.10, p=0.999)
    df = occ.reins_price_df('ph')
    assert df.index.get_level_values('distortion').unique().tolist() == ['ph']


def test_price_df_accepts_a_distortion_object(occ):
    from aggregate.spectral import Distortion

    d = Distortion('ph', 0.7)
    df = occ.reins_price_df(d, views=['ceded'])
    assert len(df) == 1


def test_price_df_refuses_both_anchors(occ):
    occ.calibrate_distortions(0.10, p=0.999)
    with pytest.raises(ValueError, match='at most one of'):
        occ.reins_price_df(p=0.99, a=1000)


def test_price_df_refuses_without_a_cession():
    with pytest.raises(ValueError, match='no reinsurance'):
        build(CLEAN).reins_price_df()


def test_price_df_refuses_an_unknown_view(occ):
    occ.calibrate_distortions(0.10, p=0.999)
    with pytest.raises(ValueError, match='unknown reins_view'):
        occ.reins_price_df(views=['net occ'])


def test_price_df_refuses_an_unknown_distortion(occ):
    occ.calibrate_distortions(0.10, p=0.999)
    with pytest.raises(ValueError, match='unknown distortion'):
        occ.reins_price_df('nonesuch')


def test_price_df_without_calibration_says_so():
    a = build(OCC.replace('RV.Occ', 'RV.Occ2'))
    with pytest.raises(ValueError, match='no calibrated distortions'):
        a.reins_price_df()


def test_portfolio_price_df(book):
    book.calibrate_distortions(0.10, p=0.999)
    df = book.reins_price_df(p=0.999)
    assert df.index.get_level_values('view').unique().tolist() == \
        ['gross', 'ceded', 'net']
    assert (df['margin'] > 0).all()


# ----------------------------------------------------------------------------
# The de-fuzz invariant (1.0.0a250)
#
# ``density_df`` has called ``remove_fuzz`` since forever and ``reins_density_df``
# never did, so the view columns kept the inverse FFT's sub-epsilon negatives.
# Nothing read them until ``reins_view=`` landed at a223, and then everything
# downstream of a cumulative sum broke at once: negative mass makes ``1 - cumsum``
# tick back *up*, so the survival is not monotone and can exceed 1. That tripped
# the exactness assertion in ``_calibration_survival`` (a bare ``AssertionError``)
# and, past it, sent ``(1 - S) ** shape`` to NaN in the Choquet weights.
# ----------------------------------------------------------------------------

def test_reins_density_df_carries_no_negative_mass(both):
    """The invariant the fix restores: a density frame holds densities.

    Asserted on the frame rather than on a symptom, because the symptoms are
    several and each one is a long way downstream of the cause.
    """
    floats = both.reins_density_df.select_dtypes('float')
    assert (floats.to_numpy() >= 0).all()


def test_every_view_calibrates(both):
    """The symptom, swept over all five views rather than the three that failed."""
    for view in both.reins_views:
        both.calibrate_distortions(0.10, p=0.99, reins_view=view)
        assert np.isclose(both.calibration_df['ROE'].iloc[0], 0.10)


def test_every_view_prices_without_nan(both):
    """Past the calibration, the same dust turned four of five families to NaN."""
    both.calibrate_distortions(0.10, p=0.99, reins_view='gross')
    df = both.reins_price_df(p=0.99)
    assert len(df) == len(both.distortions) * len(both.reins_views)
    assert df.notna().to_numpy().all()
