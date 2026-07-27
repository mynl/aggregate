"""Hygiene-4 batch regressions (1.0.0a54, dev/done/plan-hygiene-4.md).

Four items:

1. ``Portfolio.value_type`` derived from its units; mixed books are a
   construction ``ValueError``.
2. Fixed-layout ``info`` strings across Aggregate / Portfolio / Distortion
   (every row always present, ``n/a`` placeholders, shared label convention).
3. ``stats_df`` ``('meta','prem')`` / ``('meta','lr')`` backfilled from the
   ``pnl`` premium, GROSS basis under reinsurance.
4. ``value_type`` labels configurable via the ``[labels]`` config section;
   the role is the stable boolean ``_is_loss_value``.

DecL programs used here are mirrored in ``src/aggregate/agg/decl-testers.agg``
section H4.
"""

import warnings

import numpy as np
import pytest

import aggregate.config as cfg
from aggregate import build, Distortion, Portfolio


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


@pytest.fixture
def relabeled():
    """Settings with the payoff label renamed to 'asset'; restored after."""
    old = cfg._settings
    cfg._settings = cfg.load_settings(
        path=None, env={'AGGREGATE_VALUE_TYPE_PAYOFF': 'asset'})
    try:
        yield cfg._settings
    finally:
        cfg._settings = old


# ---------------------------------------------------------------------------
# Item 1 -- Portfolio.value_type derived from units
# ---------------------------------------------------------------------------

def test_portfolio_value_type_loss_book():
    p = build('port H4.Book '
              'agg A1 5 claims sev lognorm 50 cv 1 poisson '
              'agg A2 3 claims sev gamma 20 cv .5 poisson', update=False)
    assert p.value_type == 'loss'
    assert p._is_loss_value is True


def test_portfolio_value_type_payoff_book():
    # payoff-orientation aggregates (the supported payoff unit); pnl positions
    # are a separate PnL veneer and not portfolio units.
    a1 = build('agg H4.Pay1 2 claims sev lognorm 30 cv 1 poisson payoff',
               update=False)
    a2 = build('agg H4.Pay2 3 claims sev lognorm 40 cv 1 poisson payoff',
               update=False)
    p = Portfolio('PayBook', [a1, a2])
    assert p.value_type == 'payoff'
    assert p._is_loss_value is False


def test_portfolio_value_type_mixed_raises():
    loss = build('agg H4.Loss 5 claims sev lognorm 50 cv 1 poisson',
                 update=False)
    pay = build('agg H4.Pay1 2 claims sev lognorm 30 cv 1 poisson payoff',
                update=False)
    with pytest.raises(ValueError, match=r"mixed value_type.*H4\.Loss.*H4\.Pay1"):
        Portfolio('Mixed', [loss, pay])


def test_portfolio_value_type_read_only():
    p = build('port H4.Book '
              'agg A1 5 claims sev lognorm 50 cv 1 poisson '
              'agg A2 3 claims sev gamma 20 cv .5 poisson', update=False)
    with pytest.raises(AttributeError):
        p.value_type = 'payoff'


# ---------------------------------------------------------------------------
# Item 2 -- fixed-layout info strings
# ---------------------------------------------------------------------------

AGG_INFO_LABELS = [
    'aggregate object name', 'value_type', 'claim count',
    'frequency distribution', 'severity distribution', 'approximate',
    'bs', 'log2', 'padding', 'sev_calc', 'dsev_bucket', 'normalize',
    'x_min', 'x_max', 'premium', 'expected loss', 'loss ratio', 'P(X=0)',
    'validation_eps', 'reinsurance', 'occurrence reinsurance',
    'aggregate reinsurance', 'validation', 'frequency tail',
    'severity tail', 'aggregate tail', 'bounded', 'id',
]

PORT_INFO_LABELS = [
    'portfolio object name', 'value_type', 'aggregate objects',
    'allocation_method', 'bs', 'log2', 'padding', 'sev_calc', 'normalize',
    'x_min', 'x_max', 'premium', 'expected loss', 'loss ratio', 'P(X=0)',
    'reinsurance', 'aggregate tail', 'bounded', 'last update', 'id',
]

DIST_INFO_LABELS = [
    'distortion object name', 'kind', 'kind name', 'shape', 'shape name',
    'other params', 'weights mean', 'weights max', 'interior atoms',
    'gini_p', 'area', 'id',
]


def _labels(info):
    """Leading label of each info line (text before the 25-col value)."""
    return [line[:25].rstrip() for line in info.split('\n')]


def test_aggregate_info_rows_fixed():
    a = build('agg H4.Loss 5 claims sev lognorm 50 cv 1 poisson',
              update=False)
    assert _labels(a.info) == AGG_INFO_LABELS
    # not updated: grid block renders the placeholder, rows still present
    assert 'bs                       n/a' in a.info
    # updated object: identical row set in the same order
    b = build('agg H4.Loss 5 claims sev lognorm 50 cv 1 poisson')
    assert _labels(b.info) == AGG_INFO_LABELS
    assert 'n/a' not in b.info.split('x_min')[0].split('approximate')[1]


def test_portfolio_info_rows_fixed():
    p = build('port H4.Book '
              'agg A1 5 claims sev lognorm 50 cv 1 poisson '
              'agg A2 3 claims sev gamma 20 cv .5 poisson', update=False)
    assert _labels(p.info) == PORT_INFO_LABELS
    assert 'last update              n/a' in p.info
    q = build('port H4.Book '
              'agg A1 5 claims sev lognorm 50 cv 1 poisson '
              'agg A2 3 claims sev gamma 20 cv .5 poisson')
    assert _labels(q.info) == PORT_INFO_LABELS
    assert 'last update              n/a' not in q.info


@pytest.mark.parametrize('d', [
    Distortion('ph', 0.6),
    Distortion('ccoc', r=0.10),
    Distortion('bitvar', p0=0.6, p1=0.99, w1=0.3),
    Distortion('wtdtvar', ps=[.5, .9, .99], wts=[.3, .4, .3]),
], ids=['ph', 'ccoc', 'bitvar', 'wtdtvar'])
def test_distortion_info_rows_fixed(d):
    assert _labels(d.info) == DIST_INFO_LABELS
    # shared convention: no colon-style rows, no leading indent
    assert ':' not in [line[:25] for line in d.info.split('\n')][1]
    assert not d.info.startswith(' ')


def test_distortion_info_values():
    d = Distortion('ph', 0.6)
    info = d.info
    assert 'kind                     ph' in info
    assert 'kind name                proportional hazard' in info
    assert 'shape                    0.6' in info
    assert 'shape name               a' in info
    assert 'other params             none' in info
    # multi-knot kinds: gini_p / area are n/a, knots in other params
    w = Distortion('wtdtvar', ps=[.5, .9, .99], wts=[.3, .4, .3])
    assert 'gini_p                   n/a' in w.info
    assert 'ps=[0.5, 0.9, 0.99]' in w.info


# ---------------------------------------------------------------------------
# Item 3 -- stats_df prem/lr meta backfill (GROSS)
# ---------------------------------------------------------------------------

def test_exposure_clause_prem_lr_unchanged():
    a = build('agg H4.ExpPrem 100 prem at 0.65 lr sev lognorm 80 cv 1 poisson',
              update=False)
    assert a.stats_df.loc[('meta', 'prem'), 'mixed'] == 100
    assert a.stats_df.loc[('meta', 'lr'), 'mixed'] == pytest.approx(0.65)


# NOTE: the old pnl premium/lr/P(loss) meta-backfill tests were removed when the
# in-place affine was ripped out (1.0.0a103). A pnl is now the PnL veneer: the
# consideration lives on the PnL, not the loss aggregate, and premium / loss
# ratio / P(loss) reporting moves to the signed PnL summary (a later stage).


# ---------------------------------------------------------------------------
# Item 4 -- configurable value_type labels; stable role boolean
# ---------------------------------------------------------------------------

def test_default_labels_unchanged():
    a = build('agg H4.Loss 5 claims sev lognorm 50 cv 1 poisson',
              update=False)
    assert a.value_type == 'loss'
    p = build('agg H4.Pay1 2 claims sev lognorm 30 cv 1 poisson payoff',
              update=False)
    assert p.value_type == 'payoff'
    a.value_type = 'payoff'
    assert a.value_type == 'payoff'
    with pytest.raises(ValueError, match='value_type'):
        a.value_type = 'nonsense'


def test_relabel_changes_display_not_role(relabeled):
    p = build('agg H4.Pay1 2 claims sev lognorm 30 cv 1 poisson payoff',
              update=False)
    # the configured label is reported...
    assert p.value_type == 'asset'
    assert 'value_type               asset' in p.info
    # ...but the canonical role is untouched by the relabel
    assert p._is_loss_value is False
    # setter accepts both the configured label and the canonical token
    p.value_type = 'loss'
    p.value_type = 'asset'
    assert p._is_loss_value is False
    p.value_type = 'payoff'
    assert p._is_loss_value is False
    with pytest.raises(ValueError, match="'loss' or 'asset'"):
        p.value_type = 'profit'


def test_role_stable_across_relabel():
    p = build('agg H4.Pay1 2 claims sev lognorm 30 cv 1 poisson payoff',
              update=False)
    assert p.value_type == 'payoff'
    old = cfg._settings
    cfg._settings = cfg.load_settings(
        path=None, env={'AGGREGATE_VALUE_TYPE_PAYOFF': 'asset'})
    try:
        # an object built before the relabel reports the new label
        assert p.value_type == 'asset'
        assert p._is_loss_value is False
    finally:
        cfg._settings = old
    assert p.value_type == 'payoff'


def test_labels_in_settings_machinery():
    s = cfg.load_settings(path=None, env={})
    assert s.labels.loss == 'loss'
    assert s.labels.payoff == 'payoff'
    keys = [k for k, _, _ in cfg.describe_settings(s)]
    assert 'labels.loss' in keys and 'labels.payoff' in keys
