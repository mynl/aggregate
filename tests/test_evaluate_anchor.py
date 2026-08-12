"""Tests for the evaluation asset anchor ([Evaluate-Asset-Anchor], 1.0.0a261).

Phase L3 of ``dev/plan-pricing-exhibits.md``, and its acceptance criterion:
**the round trip closes.** Calibrate at ``(coc, p)``, take the implied premium
at the resolved asset level, evaluate that premium at the same anchor, and
recover each family's calibrated parameter.

Why it did not close before. ``calibrate_distortions`` solves
``rho_g(min(X, a)) = P``: its layer integral runs over ``[0, a)`` and stops.
``evaluate`` integrated to the top of the FFT grid instead, so the two solved
different equations and the gap between their answers was whatever the tail
beyond ``a`` was worth. Anchoring caps the loss at ``a``, which makes the two
equations one equation.

The reference programs are the author's, 2026-08-12: ``BasicBook``,
``BasicBookRe`` and the two unit ``port Basic``, each calibrated at
``coc = 0.15, p = 0.99``.
"""
from __future__ import annotations

import pytest

from aggregate import build

COC = 0.15
P_LEVEL = 0.99

BASIC_BOOK = ('agg EA.BasicBook 250 claims 1000 xs 0 '
              'sev lognorm 100 cv 1.5 poisson')
BASIC_BOOK_RE = ('agg EA.BasicBookRe 250 claims 1000 xs 0 '
                 'sev lognorm 100 cv 1.5 occurrence net of 276 xs 55 poisson')
BASIC_PORT = ('port EA.Basic '
              'agg EA.BasicA dfreq [1] sev gamma 100 cv 1 '
              'agg EA.BasicB dfreq [1] sev gamma 50 cv 0.8')


@pytest.fixture(scope='module')
def basic_book():
    return build(BASIC_BOOK)


@pytest.fixture(scope='module')
def basic_book_re():
    return build(BASIC_BOOK_RE)


@pytest.fixture(scope='module')
def basic_port():
    return build(BASIC_PORT)


def _round_trip(obj, **evaluate_kwargs):
    """Calibrate, take the implied premium, evaluate at the same anchor."""
    calibration = obj.calibrate_distortions(COC, p=P_LEVEL)
    premium = float(calibration.calibration_df.loc['calibration', 'P'])
    evaluation = obj.evaluate(premium, p=P_LEVEL, **evaluate_kwargs)
    return calibration, evaluation


@pytest.mark.parametrize('fixture', ['basic_book', 'basic_book_re',
                                     'basic_port'])
def test_the_round_trip_closes(fixture, request):
    """The acceptance criterion, on all three reference programs."""
    obj = request.getfixturevalue(fixture)
    calibration, evaluation = _round_trip(obj)
    calibrated = calibration.distortion_df['param']
    recovered = evaluation.evaluation_df.droplevel('Step')['param']
    assert set(recovered.index) == set(calibrated.index)
    for family in calibrated.index:
        assert recovered[family] == pytest.approx(
            calibrated[family], rel=1e-4), family


def test_the_anchor_is_reported_on_the_result(basic_book):
    calibration, evaluation = _round_trip(basic_book)
    assert evaluation.p == pytest.approx(P_LEVEL)
    assert evaluation.a == pytest.approx(calibration.a)


def test_ccoc_joins_the_families_only_when_anchored(basic_book):
    """Ruled 2026-08-12: the reason for excluding it is the missing anchor."""
    premium = float(basic_book.calibrate_distortions(COC, p=P_LEVEL)
                    .calibration_df.loc['calibration', 'P'])
    anchored = basic_book.evaluate(premium, p=P_LEVEL)
    unanchored = basic_book.evaluate(premium)
    assert anchored.names == ('ccoc', 'ph', 'wang', 'dual', 'tvar')
    assert unanchored.names == ('ph', 'wang', 'dual', 'tvar')
    assert 'ccoc' in anchored.evaluation_df.index.get_level_values('distortion')
    assert 'ccoc' not in \
        unanchored.evaluation_df.index.get_level_values('distortion')


def test_ccoc_recovers_the_cost_of_capital_it_was_calibrated_to(basic_book):
    """Its closed form is shift invariant, which is why the cap is exact."""
    _calibration, evaluation = _round_trip(basic_book)
    panel = evaluation.evaluation_df.droplevel('Step')
    assert panel.loc['ccoc', 'param'] == pytest.approx(COC, rel=1e-6)
    assert panel.loc['ccoc', 'param_name'] == 'r'


def test_the_two_receipts_spell_a_missing_parameter_name_the_same_way(
        basic_book):
    """One name per concept: ``ccoc`` reads ``r`` in both, not ``r`` and
    ``param``. The two receipts only ever met once ``ccoc`` joined the panel."""
    calibration, evaluation = _round_trip(basic_book)
    panel = evaluation.evaluation_df.droplevel('Step')
    for family in calibration.distortion_df.index:
        assert panel.loc[family, 'param_name'] == \
            calibration.distortion_df.loc[family, 'param_name'], family


def test_an_unanchored_evaluation_is_unchanged(basic_book):
    """The default is the whole distribution, the historical reading."""
    premium = float(basic_book.calibrate_distortions(COC, p=P_LEVEL)
                    .calibration_df.loc['calibration', 'P'])
    result = basic_book.evaluate(premium)
    assert result.p is None and result.a is None
    assert (result.evaluation_df['status'] == 'ok').all()
    # and it does not agree with the calibration, which is the whole point of
    # the anchor: the two are integrating over different ranges
    anchored = basic_book.evaluate(premium, p=P_LEVEL).evaluation_df
    for family in ('ph', 'wang', 'dual', 'tvar'):
        assert result.evaluation_df.loc[(basic_book.name, family), 'param'] \
            != pytest.approx(
                anchored.loc[(basic_book.name, family), 'param'], rel=1e-3)


def test_the_anchor_can_be_named_as_an_asset_level(basic_book):
    calibration = basic_book.calibrate_distortions(COC, p=P_LEVEL)
    premium = float(calibration.calibration_df.loc['calibration', 'P'])
    on_p = basic_book.evaluate(premium, p=P_LEVEL)
    on_a = basic_book.evaluate(premium, a=calibration.a)
    assert on_a.a == pytest.approx(on_p.a)
    assert on_a.evaluation_df['param'].to_numpy() == pytest.approx(
        on_p.evaluation_df['param'].to_numpy(), rel=1e-9)


def test_both_anchors_at_once_are_refused(basic_book):
    with pytest.raises(ValueError, match='at most one of p= or a='):
        basic_book.evaluate(5000.0, p=0.99, a=1000.0)


def test_the_p_equals_one_guard_reaches_evaluate(basic_book):
    with pytest.raises(ValueError, match='evaluate:'):
        basic_book.evaluate(5000.0, p=1)


def test_the_anchor_comes_off_the_view_being_evaluated(basic_book_re):
    """A cession has several distributions and the anchor belongs to the one
    measured, so the gross view resolves a higher asset level than the net."""
    # a premium above the gross expected loss, so neither panel degenerates
    # and the comparison is about the anchor rather than about acceptability
    net = basic_book_re.evaluate(30000.0, p=P_LEVEL)
    gross = basic_book_re.evaluate(30000.0, p=P_LEVEL, reins_view='gross')
    assert gross.a > net.a
    assert gross.reins_view == 'gross'


def test_a_portfolio_profile_lets_each_unit_find_its_own_level(basic_port):
    """``p=`` holds the threshold fixed across units, not the capital."""
    profile = basic_port.evaluate([200.0, 100.0], unit=['EA.BasicA',
                                                        'EA.BasicB'],
                                  p=P_LEVEL)
    # no single anchor stands for a two position profile
    assert profile.a is None and profile.p is None
    assert 'ccoc' in profile.evaluation_df.index.get_level_values('distortion')


def test_capping_places_the_tail_rather_than_dropping_it():
    """``min(X, a)`` keeps the mass; it moves it to where the position
    settles. Dropping it would renormalize and change every moment."""
    import numpy as np
    from aggregate._pricing import _cap_loss

    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    p = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    xc, pc = _cap_loss(x, p, 2.0)
    assert xc.tolist() == [0.0, 1.0, 2.0]
    assert pc.sum() == pytest.approx(1.0)
    assert pc[-1] == pytest.approx(0.3 + 0.25 + 0.15)
    # a cap at or above the top is a no-op
    assert _cap_loss(x, p, 4.0)[0].tolist() == x.tolist()
    assert _cap_loss(x, p, 99.0)[1].tolist() == p.tolist()
