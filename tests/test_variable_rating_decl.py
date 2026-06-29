"""End-to-end DecL tests for the four aggregate-basis variable-rating features.

``build('pnl ... aggregate net of <layer> <feature>')`` parses to the locked
``agg_reins_<feature>`` spec key, attaches the matching ``ContractTerms`` to the
inner aggregate, and returns a ``PnL`` whose ``gcn_df`` delegates to a
``VariableRatingAnalysis``. Retro (account-level rating clause) has no DecL surface
yet (pending a syntax decision) and is covered programmatically in
``test_variable_rating_analysis.py``.
"""

import warnings

import pytest

from aggregate import build
from aggregate.contract_terms import (
    CorridorTerms,
    ProfitCommissionTerms,
    SlideTerms,
    SwingTerms,
)

warnings.filterwarnings('ignore', message='.*heavy right tail.*')

_HEAD = ('pnl V 10000 premium less 85% lr sev lognorm 50 cv 3 poisson '
         'aggregate net of 5000 xs 4000 ')


def _means_add(pnl):
    M = pnl.gcn_df.xs('Mean')
    return all(abs(M.loc[r, 'gross'] + M.loc[r, 'ceded'] - M.loc[r, 'net']) < 1e-3
               for r in ('Premium', 'Loss', 'UW'))


# ----------------------------------------------------------------------
# swing -- replaces the premium clause; stochastic ceded premium
# ----------------------------------------------------------------------
def test_swing_build():
    p = build(_HEAD + 'swing basic 500 lcm 0.5 min 500 max 3000')
    terms = p.agg.variable_terms
    assert isinstance(terms, SwingTerms)
    assert (terms.basic, terms.lcm, terms.minimum, terms.maximum) == \
        (500.0, 0.5, 500.0, 3000.0)
    assert p.variable_rating_analysis is not None
    assert _means_add(p)
    # ceded premium is the stochastic leg
    assert p.variable_rating_analysis.stats_df.loc['ceded_premium', 'cv'] > 0


def test_swing_bare_collar_defaults():
    p = build(_HEAD + 'swing basic 500 lcm 0.5')
    terms = p.agg.variable_terms
    assert terms.minimum == pytest.approx(500.0)         # defaults to basic
    import numpy as np
    assert np.isinf(terms.maximum)                       # uncapped


# ----------------------------------------------------------------------
# slide -- replaces cede; stochastic commission (expense credit)
# ----------------------------------------------------------------------
def test_slide_build():
    p = build(_HEAD + 'deposit 1500 slide 45% at 60% and 25% at 70% and 19% at 80%')
    terms = p.agg.variable_terms
    assert isinstance(terms, SlideTerms)
    assert terms.anchors == ((0.45, 0.60), (0.25, 0.70), (0.19, 0.80))
    assert _means_add(p)
    assert p.agg.variable_ceded_premium == pytest.approx(1500.0)  # deposit denom
    assert p.variable_rating_analysis.stats_df.loc['commission', 'cv'] > 0


# ----------------------------------------------------------------------
# profit commission
# ----------------------------------------------------------------------
def test_pc_build():
    p = build(_HEAD + 'deposit 1500 pc 25% after 10%')
    terms = p.agg.variable_terms
    assert isinstance(terms, ProfitCommissionTerms)
    assert (terms.share, terms.allowance) == (0.25, 0.10)
    assert _means_add(p)


# ----------------------------------------------------------------------
# corridor
# ----------------------------------------------------------------------
def test_corridor_build():
    p = build(_HEAD + 'deposit 1500 corridor 50% po 30% xs 20%')
    terms = p.agg.variable_terms
    assert isinstance(terms, CorridorTerms)
    assert (terms.share, terms.width, terms.attachment) == (0.50, 0.30, 0.20)
    assert _means_add(p)
    # corridor reduces the cession -> ceded loss is the stochastic leg
    assert p.variable_rating_analysis.stats_df.loc['ceded_loss', 'cv'] > 0


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
        build('pnl V 10000 premium less 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 deposit 1500 pc 25% after 10% poisson')
