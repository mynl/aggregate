"""Tests for :class:`aggregate.variable_rating.VariableRatingAnalysis` (1-D path).

A synthetic discrete gross aggregate loss makes the GCN identities hand-checkable.

Gross loss ``L`` atoms ``[0, 100, 200, 300]`` with probabilities
``[0.4, 0.3, 0.2, 0.1]`` (``E[L] = 100``), ceded through a ``100 xs 100`` full-share
layer: ``A(L) = clip(L - 100, 0, 100)`` so ``A = [0, 0, 100, 100]``, ``E[A] = 30``,
``E[net] = 70``. Each feature is checked against numbers computed by hand in the
module docstring of the matching worked example.

The GCN identities now read off two observable surfaces of the analysis:

* ``gcn_df`` -- the new stats x ``['gross','ceded','net','impact']`` table; means
  add on the ``EX`` row (``EX[gross] + EX[ceded] == EX[net]`` on the underwriting
  result).
* ``stats_df`` -- exact per-leg moments; the granular per-leg means (premium,
  loss, underwriting) and the single stochastic leg's CV.
"""

import numpy as np
import pytest

from aggregate.contract_terms import (
    CorridorTerms,
    ProfitCommissionTerms,
    RetroTerms,
    SlideTerms,
    SwingTerms,
)
from aggregate.variable_rating import VariableRatingAnalysis


def _assert_means_add(an):
    """Gross + Ceded == Net on the ``EX`` row of the underwriting GCN waterfall."""
    g = an.gcn_df
    assert g.loc['EX', 'gross'] + g.loc['EX', 'ceded'] == \
        pytest.approx(g.loc['EX', 'net'], abs=1e-6)


def _assert_gained(an):
    """Smoke the surfaces gained in the refactor: ``summary_df`` and ``tail_df``."""
    assert list(an.summary_df.columns) == \
        ['Gross', 'Ceded', 'Net', 'Impact', 'Pct Impact']
    assert list(an.tail_df().columns) == ['gross_uw', 'net_uw', 'benefit']


GRID = np.array([0.0, 100.0, 200.0, 300.0])
PROB = np.array([0.4, 0.3, 0.2, 0.1])
LAYER = (1.0, 100.0, 100.0)            # 100 xs 100, full share


# ----------------------------------------------------------------------
# swing -- stochastic ceded premium = clip(20 + A, 20, 120)
# E[ceded premium] = 20*0.7 + 120*0.3 = 50
# ----------------------------------------------------------------------
def test_swing_gcn():
    an = VariableRatingAnalysis(
        GRID, PROB, SwingTerms(basic=20.0, lcm=1.0, minimum=20.0, maximum=120.0),
        gross_premium=200.0, ceded_premium=50.0, layer=LAYER)
    _assert_means_add(an)
    s = an.stats_df
    # premium row: gross 200, ceded (magnitude) 50, net = gross - ceded = 150
    assert s.loc['gross_premium', 'mean'] == pytest.approx(200.0)
    assert s.loc['ceded_premium', 'mean'] == pytest.approx(50.0)
    assert s.loc['net_premium', 'mean'] == pytest.approx(150.0)
    # loss row: gross 100, ceded 30, net 70
    assert s.loc['gross_loss', 'mean'] == pytest.approx(100.0)
    assert s.loc['ceded_loss', 'mean'] == pytest.approx(30.0)
    assert s.loc['net_loss', 'mean'] == pytest.approx(70.0)
    # underwriting row: gross 100, ceded -20 (30 - 50), net 80
    assert s.loc['gross_uw', 'mean'] == pytest.approx(100.0)
    assert s.loc['ceded_uw', 'mean'] == pytest.approx(-20.0)
    assert s.loc['net_uw', 'mean'] == pytest.approx(80.0)
    # the ceded-premium leg is the stochastic one; gross premium is fixed
    assert s.loc['ceded_premium', 'cv'] > 0
    assert s.loc['gross_premium', 'cv'] == pytest.approx(0.0)
    _assert_gained(an)


# ----------------------------------------------------------------------
# slide -- stochastic commission (expense credit); E_G = 40
# LR = A/50: A=0 -> LR 0 -> 0.45 (flat); A=100 -> LR 2 -> 0.19 (flat)
# E[comm] = (0.45*50)*0.7 + (0.19*50)*0.3 = 22.5*0.7 + 9.5*0.3 = 18.6
# ----------------------------------------------------------------------
def test_slide_gcn():
    an = VariableRatingAnalysis(
        GRID, PROB, SlideTerms.from_anchors((0.45, 0.60), (0.25, 0.70), (0.19, 0.80)),
        gross_premium=200.0, ceded_premium=50.0, layer=LAYER, gross_expense=40.0)
    _assert_means_add(an)
    s = an.stats_df
    assert s.loc['commission', 'mean'] == pytest.approx(18.6)
    assert s.loc['commission', 'cv'] > 0
    # underwriting: gross 200-100-40=60; ceded 30-50+18.6=-1.4; net 58.6
    assert s.loc['gross_uw', 'mean'] == pytest.approx(60.0)
    assert s.loc['ceded_uw', 'mean'] == pytest.approx(-1.4)
    assert s.loc['net_uw', 'mean'] == pytest.approx(58.6)
    _assert_gained(an)


# ----------------------------------------------------------------------
# profit commission -- comm = 0.25*(1 - LR - 0.1)_+ * 50; E_G = 40
# A=0 -> LR 0 -> 0.25*0.9*50 = 11.25; A=100 -> LR 2 -> 0
# E[comm] = 11.25*0.7 = 7.875
# ----------------------------------------------------------------------
def test_pc_gcn():
    an = VariableRatingAnalysis(
        GRID, PROB, ProfitCommissionTerms(share=0.25, allowance=0.10),
        gross_premium=200.0, ceded_premium=50.0, layer=LAYER, gross_expense=40.0)
    _assert_means_add(an)
    s = an.stats_df
    assert s.loc['commission', 'mean'] == pytest.approx(7.875)
    assert s.loc['ceded_uw', 'mean'] == pytest.approx(-12.125)  # 30 - 50 + 7.875
    _assert_gained(an)


# ----------------------------------------------------------------------
# corridor -- A' = phi(A/50)*50, phi(LR) = LR - 0.5*clip(LR-0.2, 0, 0.3)
# A=0 -> 0; A=100 -> LR 2 -> (2 - 0.5*0.3)*50 = 92.5; E[A'] = 92.5*0.3 = 27.75
# ----------------------------------------------------------------------
def test_corridor_gcn():
    an = VariableRatingAnalysis(
        GRID, PROB, CorridorTerms(share=0.5, width=0.30, attachment=0.20),
        gross_premium=200.0, ceded_premium=50.0, layer=LAYER)
    _assert_means_add(an)
    s = an.stats_df
    # loss row: gross 100, ceded reduced to 27.75, net 72.25
    assert s.loc['gross_loss', 'mean'] == pytest.approx(100.0)
    assert s.loc['ceded_loss', 'mean'] == pytest.approx(27.75)
    assert s.loc['net_loss', 'mean'] == pytest.approx(72.25)
    assert s.loc['ceded_loss', 'cv'] > 0
    _assert_gained(an)


# ----------------------------------------------------------------------
# retro -- account-level; gross premium = clip(150 + 1.1 L, 150, 400), no reins
# L=0->150, 100->260, 200->370, 300->400 (clip); E = 0.4*150+0.3*260+0.2*370+0.1*400 = 252
# ----------------------------------------------------------------------
def test_retro_gcn():
    an = VariableRatingAnalysis(
        GRID, PROB, RetroTerms(basic=150.0, lcm=1.1, minimum=150.0, maximum=400.0),
        gross_premium=200.0, ceded_premium=0.0, layer=None)
    _assert_means_add(an)
    s = an.stats_df
    assert s.loc['gross_premium', 'mean'] == pytest.approx(252.0)
    assert s.loc['gross_premium', 'cv'] > 0
    _assert_gained(an)


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------
def test_ratio_feature_requires_ceded_premium():
    # slide / pc / corridor read the ceded LR -> need a positive denominator
    with pytest.raises(ValueError, match='ceded premium'):
        VariableRatingAnalysis(
            GRID, PROB, ProfitCommissionTerms(share=0.25),
            gross_premium=200.0, ceded_premium=0.0, layer=LAYER)


def test_rejects_non_contract_terms():
    with pytest.raises(TypeError, match='ContractTerms'):
        VariableRatingAnalysis(GRID, PROB, object(), gross_premium=100.0)


# ----------------------------------------------------------------------
# Monte-Carlo cross-check (swing ceded premium distribution)
# ----------------------------------------------------------------------
def test_swing_monte_carlo():
    terms = SwingTerms(basic=20.0, lcm=1.0, minimum=20.0, maximum=120.0)
    an = VariableRatingAnalysis(GRID, PROB, terms, gross_premium=200.0,
                                ceded_premium=50.0, layer=LAYER)
    rng = np.random.default_rng(12345)
    draws = rng.choice(GRID, size=400_000, p=PROB)
    a = np.clip(draws - 100.0, 0.0, 100.0)
    sim = terms.phi(a)
    assert an.stats_df.loc['ceded_premium', 'mean'] == pytest.approx(sim.mean(), rel=2e-2)
    assert an.stats_df.loc['ceded_premium', 'sd'] == pytest.approx(sim.std(), rel=2e-2)
