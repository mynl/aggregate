"""Tests for ``Aggregate.price_pentagon`` / ``Portfolio.price_pentagon``.

The method completes the eight-stat pentagon octet at a fixed capital level
(``p`` or ``a``) given one pricing target, via ``Pentagon.solve``. These tests
pin the three headline targets to hand arithmetic, check that ``p=`` and the
equivalent ``a=self.q(p)`` agree, confirm ``price_ccoc`` is now a thin alias,
and exercise the input guards.
"""

import numpy as np
import pytest

from aggregate import build
from aggregate.pentagon import PENTAGON_STATS


@pytest.fixture(scope="module")
def port():
    return build('port PP.Test '
                 'agg A 1 claim sev lognorm 10 cv .3 fixed '
                 'agg B 1 claim sev lognorm 8 cv .2 fixed')


@pytest.fixture(scope="module")
def agg():
    return build('agg PP.Agg 100 claims sev lognorm 10 cv 0.5 poisson')


def _L_at(obj, a, col):
    return obj.density_df.loc[a, col]


# ---------------------------------------------------------------- Portfolio

def test_roe_target_matches_ccoc_arithmetic(port):
    """roe target reproduces P = (L + roe*a)/(1+roe) and round-trips ROE."""
    p0, r = 0.99, 0.1
    a0 = port.q(p0)
    L = _L_at(port, a0, 'exa_total')
    df = port.price_pentagon(a=a0, roe=r)
    assert np.isclose(df.loc['total', 'L'], L)
    assert np.isclose(df.loc['total', 'a'], a0)
    assert np.isclose(df.loc['total', 'P'], (L + r * a0) / (1 + r))
    assert np.isclose(df.loc['total', 'ROE'], r)


def test_premium_target_recovers_roe(port):
    """Feeding the premium back in recovers the cost of capital."""
    p0, r = 0.99, 0.1
    a0 = port.q(p0)
    L = _L_at(port, a0, 'exa_total')
    prem = (L + r * a0) / (1 + r)
    df = port.price_pentagon(a=a0, P=prem)
    assert np.isclose(df.loc['total', 'ROE'], r)


def test_lr_target_gives_premium_from_loss_ratio(port):
    """lr target gives P = L / lr."""
    p0, lr0 = 0.99, 0.7
    a0 = port.q(p0)
    L = _L_at(port, a0, 'exa_total')
    df = port.price_pentagon(a=a0, lr=lr0)
    assert np.isclose(df.loc['total', 'P'], L / lr0)
    assert np.isclose(df.loc['total', 'LR'], lr0)


def test_p_and_a_give_identical_octet(port):
    """p= and the equivalent a=q(p) produce the same eight stats."""
    p0, r = 0.99, 0.1
    a0 = port.q(p0)
    via_p = port.price_pentagon(p=p0, roe=r)
    via_a = port.price_pentagon(a=a0, roe=r)
    assert np.allclose(via_p.values, via_a.values)


def test_price_ccoc_is_thin_alias(port):
    """price_ccoc(ccoc, p=) == price_pentagon(p=, roe=ccoc), frame-identical."""
    p0, r = 0.99, 0.1
    old = port.price_ccoc(r, p=p0)
    new = port.price_pentagon(p=p0, roe=r)
    assert list(old.columns) == list(new.columns) == list(PENTAGON_STATS)
    assert old.index.equals(new.index)
    assert np.allclose(old.values, new.values)


def test_returns_canonical_one_row_total(port):
    df = port.price_pentagon(p=0.99, roe=0.1)
    assert list(df.columns) == list(PENTAGON_STATS)
    assert df.index.tolist() == ['total']
    assert df.index.name == 'line'


# ---------------------------------------------------------------- Aggregate

def test_aggregate_roe_target(agg):
    pa, r = 0.95, 0.12
    aa = agg.q(pa)
    L = _L_at(agg, aa, 'exa')
    df = agg.price_pentagon(a=aa, roe=r)
    assert np.isclose(df.loc['total', 'L'], L)
    assert np.isclose(df.loc['total', 'ROE'], r)


def test_aggregate_lr_target(agg):
    pa, lr0 = 0.95, 0.8
    aa = agg.q(pa)
    L = _L_at(agg, aa, 'exa')
    df = agg.price_pentagon(p=pa, lr=lr0)
    assert np.isclose(df.loc['total', 'P'], L / lr0)


# ---------------------------------------------------------------- guards

@pytest.mark.parametrize('kwargs', [
    {'roe': 0.1},                       # no capital level
    {'p': 0.99, 'a': 100.0, 'roe': 0.1},  # both p and a
    {'p': 0.99},                        # no target
    {'p': 0.99, 'roe': 0.1, 'lr': 0.7},   # two targets
])
def test_input_guards_raise(port, kwargs):
    with pytest.raises(ValueError):
        port.price_pentagon(**kwargs)
