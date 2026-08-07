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
