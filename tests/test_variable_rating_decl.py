"""End-to-end DecL tests for the four aggregate-basis variable-rating features.

``build('pnl ... aggregate net of <layer> <feature>')`` parses to the locked
``agg_reins_<feature>`` spec key, attaches the matching ``ContractTerms``, and
returns a :class:`~aggregate.variable_rating.VariableRatingAnalysis` **directly**
(the object *is* the analysis; there is no wrapping ``PnL`` to delegate through).
Retro (account-level rating clause) has no DecL surface yet (pending a syntax
decision) and is covered programmatically in ``test_variable_rating_analysis.py``.
"""

import warnings

import numpy as np
import pytest

from aggregate import build
from aggregate.contract_terms import (
    CorridorTerms,
    ProfitCommissionTerms,
    RetroTerms,
    SlideTerms,
    SwingTerms,
)
from aggregate.variable_rating import VariableRatingAnalysis

warnings.filterwarnings('ignore', message='.*heavy right tail.*')

_HEAD = ('pnl V 10000 premium less 85% lr sev lognorm 50 cv 3 poisson '
         'aggregate net of 5000 xs 4000 ')


def _means_add(pnl):
    """Gross + Ceded == Net on the ``EX`` row of the new stats x GCN table."""
    g = pnl.gcn_df
    return abs(g.loc['EX', 'gross'] + g.loc['EX', 'ceded']
              - g.loc['EX', 'net']) < 1e-3


def _assert_gained(pnl):
    """Smoke the surfaces gained in the refactor: ``summary_df`` and ``tail_df``."""
    assert list(pnl.summary_df.columns) == \
        ['Gross', 'Ceded', 'Net', 'Impact', 'Pct Impact']
    assert list(pnl.tail_df().columns) == ['gross_uw', 'net_uw', 'benefit']


# ----------------------------------------------------------------------
# swing -- replaces the premium clause; stochastic ceded premium
# ----------------------------------------------------------------------
def test_swing_build():
    p = build(_HEAD + 'swing basic 500 lcm 0.5 min 500 max 3000')
    assert isinstance(p, VariableRatingAnalysis)
    terms = p.terms
    assert isinstance(terms, SwingTerms)
    assert (terms.basic, terms.lcm, terms.minimum, terms.maximum) == \
        (500.0, 0.5, 500.0, 3000.0)
    assert _means_add(p)
    # ceded premium is the stochastic leg
    assert p.stats_df.loc['ceded_premium', 'cv'] > 0
    _assert_gained(p)


def test_swing_bare_collar_defaults():
    p = build(_HEAD + 'swing basic 500 lcm 0.5')
    terms = p.terms
    assert terms.minimum == pytest.approx(500.0)         # defaults to basic
    assert np.isinf(terms.maximum)                       # uncapped


# ----------------------------------------------------------------------
# slide -- replaces cede; stochastic commission (expense credit)
# ----------------------------------------------------------------------
def test_slide_build():
    p = build(_HEAD + 'deposit 1500 slide 45% at 60% and 25% at 70% and 19% at 80%')
    terms = p.terms
    assert isinstance(terms, SlideTerms)
    assert terms.anchors == ((0.45, 0.60), (0.25, 0.70), (0.19, 0.80))
    assert _means_add(p)
    assert p.ceded_premium == pytest.approx(1500.0)      # deposit denom
    assert p.stats_df.loc['commission', 'cv'] > 0
    _assert_gained(p)


# ----------------------------------------------------------------------
# profit commission
# ----------------------------------------------------------------------
def test_pc_build():
    p = build(_HEAD + 'deposit 1500 pc 25% after 10%')
    terms = p.terms
    assert isinstance(terms, ProfitCommissionTerms)
    assert (terms.share, terms.allowance) == (0.25, 0.10)
    assert _means_add(p)
    assert p.stats_df.loc['commission', 'cv'] > 0
    _assert_gained(p)


# ----------------------------------------------------------------------
# corridor
# ----------------------------------------------------------------------
def test_corridor_build():
    p = build(_HEAD + 'deposit 1500 corridor 50% po 30% xs 20%')
    terms = p.terms
    assert isinstance(terms, CorridorTerms)
    assert (terms.share, terms.width, terms.attachment) == (0.50, 0.30, 0.20)
    assert _means_add(p)
    # corridor reduces the cession -> ceded loss is the stochastic leg
    assert p.stats_df.loc['ceded_loss', 'cv'] > 0
    _assert_gained(p)


# ----------------------------------------------------------------------
# retro -- account-level rating clause in the pnl premium head
# ----------------------------------------------------------------------
def test_retro_build():
    p = build('pnl R retro basic 3000 lcm 1.1 min 3500 max 8000 premium '
              'less 1000 loss sev lognorm 100 cv 2 poisson')
    assert isinstance(p, VariableRatingAnalysis)
    terms = p.terms
    assert isinstance(terms, RetroTerms)
    assert (terms.basic, terms.lcm, terms.minimum, terms.maximum) == \
        (3000.0, 1.1, 3500.0, 8000.0)
    # gross premium is the stochastic leg; collared between min and max
    sdf = p.stats_df
    assert sdf.loc['gross_premium', 'cv'] > 0
    assert 3500.0 <= sdf.loc['gross_premium', 'mean'] <= 8000.0
    assert _means_add(p)
    _assert_gained(p)


def test_retro_bare_collar():
    p = build('pnl R retro basic 3000 lcm 1.1 premium '
              'less 1000 loss sev lognorm 100 cv 2 poisson')
    terms = p.terms
    assert terms.minimum == pytest.approx(3000.0)        # defaults to basic
    assert np.isinf(terms.maximum)                       # uncapped


def test_retro_with_reinsurance_rejected():
    with pytest.raises(ValueError, match='retro'):
        build('pnl R retro basic 3000 lcm 1.1 min 3500 max 8000 premium '
              'less 1000 loss sev lognorm 100 cv 2 poisson '
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
        build('pnl V 10000 premium less 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 deposit 1500 pc 25% after 10% poisson')
