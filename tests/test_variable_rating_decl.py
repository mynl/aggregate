"""End-to-end DecL tests for the four aggregate-basis variable-rating features.

``build('pnl ... aggregate net of <layer> <feature>')`` parses to the locked
``agg_reins_<feature>`` spec key, attaches the matching ``ContractTerms`` to the
engine, and returns an :class:`~aggregate.PnL` value object. ``pnl`` is the
**consolidated** single-group net view ([Decision-PnL-Is-Consolidated]: net
premium / net loss with the feature's map folded in); the two-group step ledger
is the ``xpnl`` **walk**. The feature's terms live on the engine
(``p.engine.variable_terms``); the treaty maps and waterfall are read off the
PnL's own ``economic_df``. Retro (account-level rating clause) varies the gross
premium and is the 1-D case with no reinsurance.
"""

import warnings

import numpy as np
import pytest

from aggregate import PnL, build
from aggregate.contract_terms import (
    CorridorTerms,
    ProfitCommissionTerms,
    RetroTerms,
    SlideTerms,
    SwingTerms,
)

warnings.filterwarnings('ignore', message='.*heavy right tail.*')

_HEAD = ('pnl V 10000 premium less agg V_e 10000 prem at 85% lr sev lognorm 50 cv 3 poisson '
         'aggregate net of 5000 xs 4000 ')


def _means_add(pnl):
    """The signed ledger foots: the leg means sum to the grand result."""
    dd = pnl.density_df
    legs = [k for k in dd if not k.endswith('result') and k != 'margin']
    return abs(sum(dd[k].mean() for k in legs) - pnl.est_m) < 1e-3


def _leg(pnl, label):
    """One declared leg's economic_df row, by Line label."""
    return pnl.economic_df.xs(label, level='Label').iloc[0]


# ----------------------------------------------------------------------
# swing -- replaces the premium clause; stochastic ceded premium
# ----------------------------------------------------------------------
def test_swing_build():
    p = build(_HEAD + 'swing basic 500 lcm 0.5 min 500 max 3000')
    assert isinstance(p, PnL)
    terms = p.engine.variable_terms
    assert isinstance(terms, SwingTerms)
    assert (terms.basic, terms.lcm, terms.minimum, terms.maximum) == \
        (500.0, 0.5, 500.0, 3000.0)
    assert _means_add(p)
    # consolidated: the stochastic ceded premium folds into the net premium
    assert _leg(p, 'net premium')['SD'] > 0
    # the walk (xpnl) shows the ceded premium as its own stochastic leg
    x = build(_HEAD.replace('pnl V', 'xpnl VX', 1)
              + 'swing basic 500 lcm 0.5 min 500 max 3000')
    assert [g.label for g in x.groups] == ['Gross', 'ceded agg']
    assert _leg(x, 'ceded agg premium')['SD'] > 0
    # the two faces agree on the net position
    assert abs(p.est_m - x.est_m) < 1e-9


def test_swing_bare_collar_defaults():
    p = build(_HEAD + 'swing basic 500 lcm 0.5')
    terms = p.engine.variable_terms
    assert terms.minimum == pytest.approx(500.0)         # defaults to basic
    assert np.isinf(terms.maximum)                       # uncapped


def test_swing_terms_scale_by_placement_share():
    # swing basic / min / max are quoted at 100% placement and scaled by the
    # share placed; lcm (a dimensionless loss multiplier on the already-placed
    # ceded loss) is unchanged. At 50% the collar currency terms halve.
    head = ('pnl V 10000 premium less agg V_e 10000 prem at 85% lr sev lognorm '
            '50 cv 3 poisson aggregate net of ')
    sw = ' swing basic 500 lcm 0.5 min 500 max 3000'
    half = build(head + '50% so 5000 xs 4000' + sw)
    t = half.engine.variable_terms
    assert t.basic == pytest.approx(250.0)               # 0.5 x 500
    assert t.minimum == pytest.approx(250.0)             # 0.5 x 500
    assert t.maximum == pytest.approx(1500.0)            # 0.5 x 3000
    assert t.lcm == pytest.approx(0.5)                   # unchanged
    # end to end: the ceded premium (gross - net premium) halves vs 100%
    full = build(head + '5000 xs 4000' + sw)
    ceded_full = 10000 - _leg(full, 'net premium')['EX']
    ceded_half = 10000 - _leg(half, 'net premium')['EX']
    assert ceded_half == pytest.approx(0.5 * ceded_full, rel=1e-6)


# ----------------------------------------------------------------------
# slide -- replaces cede; stochastic commission (expense credit)
# ----------------------------------------------------------------------
def test_slide_build():
    p = build(_HEAD + 'deposit 1500 slide 45% at 60% and 25% at 70% and 19% at 80%')
    terms = p.engine.variable_terms
    assert isinstance(terms, SlideTerms)
    assert terms.anchors == ((0.45, 0.60), (0.25, 0.70), (0.19, 0.80))
    assert _means_add(p)
    assert p.engine.variable_ceded_premium == pytest.approx(1500.0)  # deposit denom
    # consolidated: the sliding commission credit makes the net premium
    # stochastic
    assert _leg(p, 'net premium')['SD'] > 0
    # the walk shows the sliding commission as its own stochastic received leg
    x = build(_HEAD.replace('pnl V', 'xpnl VX', 1)
              + 'deposit 1500 slide 45% at 60% and 25% at 70% and 19% at 80%')
    assert _leg(x, 'sliding commission')['SD'] > 0


# ----------------------------------------------------------------------
# profit commission
# ----------------------------------------------------------------------
def test_pc_build():
    p = build(_HEAD + 'deposit 1500 pc 25% after 10%')
    terms = p.engine.variable_terms
    assert isinstance(terms, ProfitCommissionTerms)
    assert (terms.share, terms.allowance) == (0.25, 0.10)
    assert _means_add(p)
    # consolidated: the profit commission credit rides the net premium
    assert _leg(p, 'net premium')['SD'] > 0
    x = build(_HEAD.replace('pnl V', 'xpnl VX', 1)
              + 'deposit 1500 pc 25% after 10%')
    assert _leg(x, 'profit commission')['SD'] > 0


# ----------------------------------------------------------------------
# corridor
# ----------------------------------------------------------------------
def test_corridor_build():
    p = build(_HEAD + 'deposit 1500 corridor 50% po 30% xs 20%')
    terms = p.engine.variable_terms
    assert isinstance(terms, CorridorTerms)
    assert (terms.share, terms.width, terms.attachment) == (0.50, 0.30, 0.20)
    assert _means_add(p)
    # consolidated: fixed net premium (corridor keeps the deposit split);
    # the corridor-adjusted recovery shapes the net loss
    assert _leg(p, 'net premium')['SD'] == 0
    assert _leg(p, 'Loss (net)')['SD'] > 0
    # the walk shows the corridor-adjusted recovery as the changed stochastic leg
    x = build(_HEAD.replace('pnl V', 'xpnl VX', 1)
              + 'deposit 1500 corridor 50% po 30% xs 20%')
    assert _leg(x, 'ceded agg recovery')['SD'] > 0
    assert abs(p.est_m - x.est_m) < 1e-9


# ----------------------------------------------------------------------
# retro -- account-level rating clause in the pnl premium head
# ----------------------------------------------------------------------
def test_retro_build():
    p = build('pnl R retro basic 3000 lcm 1.1 min 3500 max 8000 premium '
              'less agg R_e 1000 loss sev lognorm 100 cv 2 poisson')
    assert isinstance(p, PnL)
    terms = p.engine.variable_terms
    assert isinstance(terms, RetroTerms)
    assert (terms.basic, terms.lcm, terms.minimum, terms.maximum) == \
        (3000.0, 1.1, 3500.0, 8000.0)
    # gross premium is the stochastic leg; collared between min and max
    prem = _leg(p, 'Premium')
    assert prem['SD'] > 0
    assert 3500.0 <= prem['EX'] <= 8000.0
    assert _means_add(p)


# ----------------------------------------------------------------------
# stochastic premium is exactly where E[L/P] parts company with E[L]/E[P]
# ([PnL-Ratio-Frame]): the two readings coincide for a fixed premium, and a
# retro or swing makes premium random AND correlated with loss, so they must
# not be conflated under one "loss ratio" heading
# ----------------------------------------------------------------------
def test_retro_premium_separates_the_two_loss_ratios():
    """``LR`` and ``E_LR`` differ, and in the direction the retro implies.

    A retro premium rises with loss, so the high-loss atoms carry the high
    premiums and the per-atom ratio is damped: ``E[L/P] < E[L]/E[P]``.
    """
    p = build('pnl R retro basic 3000 lcm 1.1 min 3500 max 8000 premium '
              'less agg R_e 1000 loss sev lognorm 100 cv 2 poisson')
    assert _leg(p, 'Premium')['SD'] > 0            # premium really is random
    row = p.economic_ratios_df.iloc[0]
    assert row['LR'] > 0 and row['E_LR'] > 0
    assert abs(row['E_LR'] - row['LR']) > 1e-3, \
        'a correlated premium must move the mean of the ratio off the ratio ' \
        'of the means'
    assert row['E_LR'] < row['LR']


def test_fixed_premium_makes_the_two_loss_ratios_agree():
    """The control: no correlation to carry, so the readings coincide."""
    p = build('pnl F 5000 premium less agg F_e 1000 loss '
              'sev lognorm 100 cv 2 poisson')
    row = p.economic_ratios_df.iloc[0]
    assert row['E_LR'] == pytest.approx(row['LR'], rel=1e-12)


# ----------------------------------------------------------------------
# the acceptance pair (dev/plan-yapnl.md examples 1-2): a retro program has
# the SAME exhibit shape as the plain gross book -- a feature never changes
# the machinery, it changes one leg's function
# ----------------------------------------------------------------------
def test_acceptance_pair_gross_vs_retro_same_shape():
    gross = build(
        'pnl GrossPNL as "Gross Book PNL" inherit premium as "Gross Premium" '
        'less agg A as "Gross Loss" 10000 premium at 85% lr '
        'sev lognorm 50 cv 3 poisson '
        'less 5% loss expense as LAE '
        '100 fixed expense and 10% premium expense as "Fixed & Acq Exp"')
    retro = build(
        'pnl RetroPNL as "Retro Rated Gross PNL" '
        'retro basic 2000 lcm 1.1 min 8000 max 14000 premium as "Retro Premium" '
        'less agg B as "Gross Loss" 10000 premium at 85% lr '
        'sev lognorm 50 cv 3 poisson '
        'less 5% loss expense as LAE '
        '100 fixed expense as "Fixed Exp" 10% premium expense as "Acq Exp"')
    g, r = gross.economic_df, retro.economic_df
    # identical template: declared labels on every row, same columns
    assert list(g.columns) == list(r.columns)
    assert list(g.index) == [
        ('Consideration', 'Gross Premium'),
        ('Obligation', 'Gross Loss'), ('Obligation', 'LAE'),
        ('Obligation', 'Fixed & Acq Exp'), ('Obligation', 'Total'),
        ('Margin', 'Total')]
    assert list(r.index) == [
        ('Consideration', 'Retro Premium'),
        ('Obligation', 'Gross Loss'), ('Obligation', 'LAE'),
        ('Obligation', 'Fixed Exp'), ('Obligation', 'Acq Exp'),
        ('Obligation', 'Total'), ('Margin', 'Total')]
    # the cards are the SAME fixed shape regardless of the ledgers
    assert list(gross.summary_df.index) == list(retro.summary_df.index) == \
        ['Consideration', 'Obligation', 'Margin']
    # LAE is stochastic (rate * actual loss) in BOTH programs
    assert g.loc[('Obligation', 'LAE'), 'SD'] > 0
    assert r.loc[('Obligation', 'LAE'), 'SD'] > 0
    # the one changed leg: the retro premium is stochastic, the gross fixed
    assert g.loc[('Consideration', 'Gross Premium'), 'SD'] == 0
    assert r.loc[('Consideration', 'Retro Premium'), 'SD'] > 0
    # the EX column foots to the result in both
    for df in (g, r):
        legs = [i for i in df.index if i[1] != 'Total']
        assert df.loc[('Margin', 'Total'), 'EX'] == pytest.approx(
            df.loc[legs, 'EX'].sum(), abs=1e-6)


def test_retro_bare_collar():
    p = build('pnl R retro basic 3000 lcm 1.1 premium '
              'less agg R_e 1000 loss sev lognorm 100 cv 2 poisson')
    terms = p.engine.variable_terms
    assert terms.minimum == pytest.approx(3000.0)        # defaults to basic
    assert np.isinf(terms.maximum)                       # uncapped


def test_retro_with_reinsurance_rejected():
    with pytest.raises(ValueError, match='retro'):
        build('pnl R retro basic 3000 lcm 1.1 min 3500 max 8000 premium '
              'less agg R_e 1000 loss sev lognorm 100 cv 2 poisson '
              'aggregate net of 5000 xs 4000 deposit 1500')


# ----------------------------------------------------------------------
# negative cases (decision 0 + per-feature compatibility)
# ----------------------------------------------------------------------
def test_swing_with_premium_rejected():
    with pytest.raises(ValueError, match='swing'):
        build(_HEAD + 'deposit 1500 swing basic 500 lcm 0.5')


def test_slide_with_cede_rejected():
    with pytest.raises(ValueError, match='slide'):
        build(_HEAD + 'deposit 1500 cede 20% slide 45% at 60% and 25% at 70%')


def test_ratio_feature_without_premium_rejected():
    with pytest.raises(ValueError, match='ceded loss ratio'):
        build(_HEAD + 'pc 25% after 10%')


def test_occurrence_basis_rejected():
    # variable features are aggregate-basis only in this release
    with pytest.raises(ValueError, match='aggregate reinsurance only'):
        build('pnl V 10000 premium less agg V_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 deposit 1500 pc 25% after 10% poisson')
