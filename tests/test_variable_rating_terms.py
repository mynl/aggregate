"""Tests for the five :class:`aggregate.contract_terms.ContractTerms` features.

Each feature ships a hand-checked, sign-correct worked example (the
``plan-variable-rating.md`` discipline) *before* it becomes a fixture:

* retro  ``basic 1000 lcm 1.10 min 1000 max 2500``:  L=0->1000, 500->1550, 1500->2500
* swing  ``basic 0 lcm 1.0 min 100 max 300``:        A=50->100, 200->200, 400->300
* slide  ``45% at 60% and 25% at 70% and 19% at 80%``: LR .55->.45 .65->.35 .75->.22 .90->.19
* pc     ``pc 25% after 10%``:                        LR .50->.10 .60->.075 .95->0
* corridor ``50% po 30% xs 20%`` (P=1000):           A 100->100 350->275 600->450

The ``target_leg`` / ``loss_basis`` metadata and the shared base are checked too.
"""

import numpy as np
import pytest

from aggregate.contract_terms import (
    ContractTerms,
    CorridorTerms,
    ProfitCommissionTerms,
    RetroTerms,
    SlideTerms,
    SwingTerms,
)


# ----------------------------------------------------------------------
# metadata / taxonomy
# ----------------------------------------------------------------------
def test_target_leg_and_loss_basis_metadata():
    assert RetroTerms.target_leg == 'gross_premium'
    assert RetroTerms.loss_basis == 'net_account_loss'
    assert SwingTerms.target_leg == 'ceded_premium'
    assert SwingTerms.loss_basis == 'ceded_loss'
    assert SlideTerms.target_leg == 'expense'
    assert SlideTerms.loss_basis == 'ceded_lr'
    assert ProfitCommissionTerms.target_leg == 'expense'
    assert ProfitCommissionTerms.loss_basis == 'ceded_lr'
    assert CorridorTerms.target_leg == 'ceded_loss'
    assert CorridorTerms.loss_basis == 'ceded_lr'


def test_all_are_contract_terms():
    for cls, kw in (
        (RetroTerms, dict(basic=1.0, lcm=1.0)),
        (SwingTerms, dict(basic=1.0, lcm=1.0)),
        (SlideTerms, dict(anchors=((0.2, 0.6),))),
        (ProfitCommissionTerms, dict(share=0.2)),
        (CorridorTerms, dict(share=0.5, width=0.3, attachment=0.2)),
    ):
        assert isinstance(cls(**kw), ContractTerms)


# ----------------------------------------------------------------------
# retro -- collared affine gross premium
# ----------------------------------------------------------------------
def test_retro_worked_example():
    t = RetroTerms(basic=1000.0, lcm=1.10, minimum=1000.0, maximum=2500.0)
    L = np.array([0.0, 500.0, 1500.0])
    assert t.phi(L) == pytest.approx([1000.0, 1550.0, 2500.0])


def test_retro_default_collar():
    # minimum defaults to basic; maximum defaults to +inf
    t = RetroTerms(basic=1000.0, lcm=1.10)
    assert t.minimum == pytest.approx(1000.0)
    assert np.isinf(t.maximum)
    assert t.phi(0.0) == pytest.approx(1000.0)          # floored at basic
    assert t.phi(1e6) == pytest.approx(1000.0 + 1.10e6)  # uncapped


def test_retro_monotone_and_vectorized():
    t = RetroTerms(basic=1000.0, lcm=1.10, minimum=1000.0, maximum=2500.0)
    L = np.linspace(0.0, 3000.0, 1001)
    p = t.phi(L)
    assert p.shape == L.shape
    assert np.all(np.diff(p) >= -1e-12)
    assert np.all((p >= 1000.0 - 1e-9) & (p <= 2500.0 + 1e-9))


# ----------------------------------------------------------------------
# swing -- collared affine ceded premium
# ----------------------------------------------------------------------
def test_swing_worked_example():
    t = SwingTerms(basic=0.0, lcm=1.0, minimum=100.0, maximum=300.0)
    A = np.array([50.0, 200.0, 400.0])
    assert t.phi(A) == pytest.approx([100.0, 200.0, 300.0])


# ----------------------------------------------------------------------
# slide -- decreasing PWL ceding commission
# ----------------------------------------------------------------------
SLIDE = SlideTerms.from_anchors((0.45, 0.60), (0.25, 0.70), (0.19, 0.80))


def test_slide_worked_example():
    LR = np.array([0.55, 0.65, 0.75, 0.90])
    assert SLIDE.phi(LR) == pytest.approx([0.45, 0.35, 0.22, 0.19])


def test_slide_anchors_sorted_by_loss_ratio():
    # unsorted input is reordered ascending by loss ratio
    t = SlideTerms.from_anchors((0.19, 0.80), (0.45, 0.60), (0.25, 0.70))
    assert [lr for _, lr in t.anchors] == [0.60, 0.70, 0.80]
    assert [c for c, _ in t.anchors] == [0.45, 0.25, 0.19]


def test_slide_flat_outside_ends_and_at_anchors():
    LR = np.array([0.0, 0.60, 0.70, 0.80, 2.0])
    assert SLIDE.phi(LR) == pytest.approx([0.45, 0.45, 0.25, 0.19, 0.19])


def test_slide_rejects_increasing_commission():
    with pytest.raises(ValueError, match='nonincreasing'):
        SlideTerms.from_anchors((0.20, 0.60), (0.30, 0.70))  # comm rises with LR


def test_slide_rejects_duplicate_loss_ratios():
    with pytest.raises(ValueError, match='distinct'):
        SlideTerms.from_anchors((0.30, 0.60), (0.20, 0.60))


# ----------------------------------------------------------------------
# profit commission
# ----------------------------------------------------------------------
def test_pc_worked_example():
    t = ProfitCommissionTerms(share=0.25, allowance=0.10)
    LR = np.array([0.50, 0.60, 0.95])
    assert t.phi(LR) == pytest.approx([0.10, 0.075, 0.0])


def test_pc_clamped_at_zero_and_monotone():
    t = ProfitCommissionTerms(share=0.25, allowance=0.10)
    LR = np.linspace(0.0, 1.5, 301)
    p = t.phi(LR)
    assert np.all(p >= -1e-12)
    assert np.all(np.diff(p) <= 1e-12)                  # nonincreasing
    assert t.phi(1.0 - 0.10) == pytest.approx(0.0)      # zero at LR = 1-allowance


def test_pc_rejects_bad_share():
    with pytest.raises(ValueError, match='share'):
        ProfitCommissionTerms(share=1.5)
    with pytest.raises(ValueError, match='share'):
        ProfitCommissionTerms(share=-0.1)


# ----------------------------------------------------------------------
# corridor -- loss-ratio band retained by the cedant
# ----------------------------------------------------------------------
def test_corridor_worked_example():
    c = CorridorTerms(share=0.5, width=0.30, attachment=0.20)
    P = 1000.0
    A = np.array([100.0, 350.0, 600.0])
    ceded = c.phi(A / P) * P
    assert ceded == pytest.approx([100.0, 275.0, 450.0])


def test_corridor_flat_reduction_above_band():
    # above the band the cession is reduced by the constant share*width (in LR)
    c = CorridorTerms(share=0.5, width=0.30, attachment=0.20)
    assert c.phi(0.10) == pytest.approx(0.10)            # below band: untouched
    assert c.phi(2.0) == pytest.approx(2.0 - 0.5 * 0.30)  # above: constant credit


def test_corridor_monotone_nondecreasing():
    c = CorridorTerms(share=0.5, width=0.30, attachment=0.20)
    LR = np.linspace(0.0, 2.0, 1001)
    p = c.phi(LR)
    assert np.all(np.diff(p) >= -1e-12)                 # comonotone in LR


def test_corridor_rejects_bad_inputs():
    with pytest.raises(ValueError, match='share'):
        CorridorTerms(share=2.0, width=0.30, attachment=0.20)
    with pytest.raises(ValueError, match='width'):
        CorridorTerms(share=0.5, width=0.0, attachment=0.20)
    with pytest.raises(ValueError, match='attachment'):
        CorridorTerms(share=0.5, width=0.30, attachment=-0.1)


# ----------------------------------------------------------------------
# collar validation (retro / swing share the machinery)
# ----------------------------------------------------------------------
def test_collar_rejects_min_above_max():
    with pytest.raises(ValueError, match='minimum'):
        RetroTerms(basic=0.0, lcm=1.0, minimum=300.0, maximum=100.0)


def test_collar_rejects_negative_lcm():
    with pytest.raises(ValueError, match='lcm'):
        SwingTerms(basic=0.0, lcm=-1.0)


def test_frozen_immutable():
    t = ProfitCommissionTerms(share=0.25, allowance=0.10)
    with pytest.raises(Exception):
        t.share = 0.5
