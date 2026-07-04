"""End-to-end DecL tests for the four aggregate-basis variable-rating features.

``build('pnl ... aggregate net of <layer> <feature>')`` parses to the locked
``agg_reins_<feature>`` spec key, attaches the matching ``ContractTerms``, and
returns an :class:`~aggregate.PnL` value object (the always-PnL face) with the
:class:`~aggregate.variable_rating.VariableRatingAnalysis` attached as
``.analysis``. The returned P&L carries its own fixed exhibits (the
``summary_df`` card and the ``stats_df`` ledger sheet); the Gross/Ceded/Net
waterfall (``gcn_df``) and the treaty maps live on ``.analysis``. Retro (account-level
rating clause) varies the gross premium and is the 1-D case with no
reinsurance.
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
from aggregate.variable_rating import VariableRatingAnalysis

warnings.filterwarnings('ignore', message='.*heavy right tail.*')

_HEAD = ('pnl V 10000 premium less agg V_e 10000 prem at 85% lr sev lognorm 50 cv 3 poisson '
         'aggregate net of 5000 xs 4000 ')


def _means_add(pnl):
    """The signed ledger foots: the leg means sum to the grand result."""
    dd = pnl.density_df
    legs = [k for k in dd if not k.endswith('result') and k != 'margin']
    return abs(sum(dd[k].mean() for k in legs) - pnl.mean) < 1e-3


def _leg(pnl, label):
    """One declared leg's stats_df row, by Line label."""
    return pnl.stats_df.xs(label, level='Line').iloc[0]


def _assert_gained(pnl):
    """Smoke the analysis drill-down surface (the kept domain extra)."""
    assert list(pnl.analysis.tail_df().columns) == \
        ['gross_uw', 'net_uw', 'benefit']


# ----------------------------------------------------------------------
# swing -- replaces the premium clause; stochastic ceded premium
# ----------------------------------------------------------------------
def test_swing_build():
    p = build(_HEAD + 'swing basic 500 lcm 0.5 min 500 max 3000')
    assert isinstance(p, PnL)
    assert isinstance(p.analysis, VariableRatingAnalysis)
    terms = p.analysis.terms
    assert isinstance(terms, SwingTerms)
    assert (terms.basic, terms.lcm, terms.minimum, terms.maximum) == \
        (500.0, 0.5, 500.0, 3000.0)
    assert _means_add(p)
    # ceded premium is the stochastic leg -- in the ledger and the engine
    assert _leg(p, 'ceded agg premium')['SD'] > 0
    assert p.analysis._stats_df.loc['ceded_premium', 'cv'] > 0
    _assert_gained(p)


def test_swing_bare_collar_defaults():
    p = build(_HEAD + 'swing basic 500 lcm 0.5')
    terms = p.analysis.terms
    assert terms.minimum == pytest.approx(500.0)         # defaults to basic
    assert np.isinf(terms.maximum)                       # uncapped


# ----------------------------------------------------------------------
# slide -- replaces cede; stochastic commission (expense credit)
# ----------------------------------------------------------------------
def test_slide_build():
    p = build(_HEAD + 'deposit 1500 slide 45% at 60% and 25% at 70% and 19% at 80%')
    terms = p.analysis.terms
    assert isinstance(terms, SlideTerms)
    assert terms.anchors == ((0.45, 0.60), (0.25, 0.70), (0.19, 0.80))
    assert _means_add(p)
    assert p.analysis.ceded_premium == pytest.approx(1500.0)  # deposit denom
    # the sliding commission is the one changed leg (stochastic, received)
    assert _leg(p, 'sliding commission')['SD'] > 0
    assert p.analysis._stats_df.loc['commission', 'cv'] > 0
    _assert_gained(p)


# ----------------------------------------------------------------------
# profit commission
# ----------------------------------------------------------------------
def test_pc_build():
    p = build(_HEAD + 'deposit 1500 pc 25% after 10%')
    terms = p.analysis.terms
    assert isinstance(terms, ProfitCommissionTerms)
    assert (terms.share, terms.allowance) == (0.25, 0.10)
    assert _means_add(p)
    assert _leg(p, 'profit commission')['SD'] > 0
    assert p.analysis._stats_df.loc['commission', 'cv'] > 0
    _assert_gained(p)


# ----------------------------------------------------------------------
# corridor
# ----------------------------------------------------------------------
def test_corridor_build():
    p = build(_HEAD + 'deposit 1500 corridor 50% po 30% xs 20%')
    terms = p.analysis.terms
    assert isinstance(terms, CorridorTerms)
    assert (terms.share, terms.width, terms.attachment) == (0.50, 0.30, 0.20)
    assert _means_add(p)
    # corridor reduces the cession -> the recovery is the changed leg
    assert _leg(p, 'ceded agg recovery')['SD'] > 0
    assert p.analysis._stats_df.loc['ceded_loss', 'cv'] > 0
    _assert_gained(p)


# ----------------------------------------------------------------------
# retro -- account-level rating clause in the pnl premium head
# ----------------------------------------------------------------------
def test_retro_build():
    p = build('pnl R retro basic 3000 lcm 1.1 min 3500 max 8000 premium '
              'less agg R_e 1000 loss sev lognorm 100 cv 2 poisson')
    assert isinstance(p, PnL)
    assert isinstance(p.analysis, VariableRatingAnalysis)
    terms = p.analysis.terms
    assert isinstance(terms, RetroTerms)
    assert (terms.basic, terms.lcm, terms.minimum, terms.maximum) == \
        (3000.0, 1.1, 3500.0, 8000.0)
    # gross premium is the stochastic leg; collared between min and max
    prem = _leg(p, 'premium')
    assert prem['SD'] > 0
    assert 3500.0 <= prem['EX'] <= 8000.0
    assert _means_add(p)
    _assert_gained(p)


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
    g, r = gross.stats_df, retro.stats_df
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
    terms = p.analysis.terms
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
