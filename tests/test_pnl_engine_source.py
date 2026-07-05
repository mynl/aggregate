"""Tests for [PnL-Engine-Source]: a P&L wraps a complete stochastic engine.

The breaking ``pnl NAME <premium> less <engine> [less <expenses>]`` syntax, where
the engine is an inline ``agg``, a stored ``agg.NAME``, or a stored ``port.NAME``.
Covers the ``inherit premium`` head, the two-independent-premiums feature, the
``xpnl`` multi-group walk ([Decision-XPnL-Is-A-Recipe]), and the
deferred/error edges. See dev/plan-pnl-engine-source.md and
dev/plan-pnl-consolidated-xpnl-walk.md.
"""

import numpy as np
import pandas as pd
import pytest

from aggregate import Underwriter
from aggregate._pnl import PnL


@pytest.fixture
def uw():
    """A clean underwriter (no databases) with eager update."""
    return Underwriter(databases=None, update=True)


# ----------------------------------------------------------------------
# Inline engine
# ----------------------------------------------------------------------
def test_inline_fixed_premium(uw):
    """``pnl`` wrapping an inline ``agg`` with a fixed consideration."""
    p = uw('pnl P 12000 premium less agg B 10000 premium at 0.65 lr '
           'sev lognorm 100 cv 2 poisson')
    assert isinstance(p, PnL)


def test_inline_claims_and_expense(uw):
    """The second ``less`` anchors the (optional) expense clause."""
    p = uw('pnl P 12000 premium less agg B 100 claims sev lognorm 100 cv 2 '
           'poisson less 25% premium expenses')
    assert isinstance(p, PnL)


def test_inline_dfreq_engine(uw):
    """A discrete-frequency engine body is a valid inline engine."""
    p = uw('pnl P 50 premium less agg D dfreq [1 2 3] dsev [10 20 30]')
    assert isinstance(p, PnL)


def test_two_independent_premiums(uw):
    """The engine premium (sizing) and the P&L premium (consideration) are
    independent -- booking 12000 over a book sized at 10000 is rate adequacy."""
    rate_adequate = uw('pnl P 12000 premium less agg B 10000 premium at 0.65 lr '
                       'sev lognorm 100 cv 2 poisson')
    technical = uw('pnl Q 10000 premium less agg B 10000 premium at 0.65 lr '
                   'sev lognorm 100 cv 2 poisson')
    # Same engine (E[loss] = 6500) but different consideration -> different margin.
    assert isinstance(rate_adequate, PnL) and isinstance(technical, PnL)


# ----------------------------------------------------------------------
# inherit premium
# ----------------------------------------------------------------------
def test_inherit_premium_from_engine(uw):
    """``inherit premium`` copies the engine's technical premium."""
    inherited = uw('pnl P inherit premium less agg B 8000 premium at 0.65 lr '
                   'sev lognorm 100 cv 2 poisson')
    explicit = uw('pnl Q 8000 premium less agg B 8000 premium at 0.65 lr '
                  'sev lognorm 100 cv 2 poisson')
    assert isinstance(inherited, PnL) and isinstance(explicit, PnL)


def test_inherit_premium_no_premium_errors(uw):
    """``inherit premium`` on an engine with no premium (claims exposure) errors."""
    with pytest.raises(ValueError, match='inherit'):
        uw('pnl P inherit premium less agg B 100 claims sev lognorm 100 cv 2 '
           'poisson')


# ----------------------------------------------------------------------
# agg.NAME reference engine
# ----------------------------------------------------------------------
def test_ref_agg_engine(uw):
    """A ``pnl`` can wrap a stored ``agg.NAME``."""
    uw('agg Stored 100 claims sev lognorm 100 cv 2 poisson')
    p = uw('pnl P 5000 premium less agg.Stored')
    assert isinstance(p, PnL)


def test_ref_agg_inherit(uw):
    """``inherit`` off a stored agg carrying premium; and the no-premium error."""
    uw('agg WithPrem 8000 premium at 0.6 lr sev lognorm 100 cv 2 poisson')
    assert isinstance(uw('pnl P inherit premium less agg.WithPrem'), PnL)
    uw('agg NoPrem 100 claims sev lognorm 100 cv 2 poisson')
    with pytest.raises(ValueError, match='inherit'):
        uw('pnl Q inherit premium less agg.NoPrem')


# ----------------------------------------------------------------------
# port.NAME reference engine + Portfolio premium accumulation
# ----------------------------------------------------------------------
def test_portfolio_accumulates_premium(uw):
    """Portfolio exposes ``exp_premium`` = sum of its units' premium."""
    uw('agg U1 500 premium at 0.6 lr sev lognorm 50 cv 2 poisson')
    uw('agg U2 700 premium at 0.55 lr sev gamma 40 cv 1.5 poisson')
    port = uw('port Bk agg.U1 agg.U2')
    assert port.exp_premium == pytest.approx(1200.0)


def test_ref_port_engine_fixed_and_inherit(uw):
    """A ``pnl`` sourced from a ``port.NAME`` reads the net-net total; ``inherit``
    reads the accumulated portfolio premium."""
    uw('agg U1 500 premium at 0.6 lr sev lognorm 50 cv 2 poisson')
    uw('agg U2 700 premium at 0.55 lr sev gamma 40 cv 1.5 poisson')
    uw('port Bk agg.U1 agg.U2')
    assert isinstance(uw('pnl PB 3000 premium less port.Bk'), PnL)
    assert isinstance(uw('pnl PC inherit premium less port.Bk'), PnL)


def test_port_engine_expenses_supported(uw):
    """Expenses on a portfolio-sourced P&L book like any other (un-NYI)."""
    uw('agg U1 500 premium at 0.6 lr sev lognorm 50 cv 2 poisson')
    uw('port Bk1 agg.U1')
    p = uw('pnl PE 1000 premium less port.Bk1 less 10% premium expense '
           'and 50 fixed expense')
    assert isinstance(p, PnL)
    # one obligation leg for the and-joined group, booked signed
    assert p.stats_df.loc[('Obligation', 'expense'), 'EX'] == pytest.approx(
        -(0.10 * 1000 + 50.0))


# ----------------------------------------------------------------------
# xpnl -> the per-atom multi-group walk ([Decision-XPnL-Is-A-Recipe])
# ----------------------------------------------------------------------
def test_xpnl_returns_walk_pnl_over_gcn_engine(uw):
    """``xpnl`` over an engine with reinsurance economics returns a plain
    **multi-group PnL** -- the step walk (gross -> cover -> All) with the
    exploded (Step, View) card and (Step, View, Line) stats sheet, per-atom
    over the occurrence (gross, ceded) joint."""
    t = uw('xpnl X 1000 premium less agg e 1000 premium at 0.7 lr '
           'sev lognorm 100 cv 2 occurrence ceded to 500 xs 500 deposit 100 '
           'poisson')
    assert isinstance(t, PnL)
    assert [g.label for g in t.groups] == ['Gross', 'ceded occ']
    s = t.stats_df
    assert list(s.index.names) == ['Step', 'View', 'Line']
    # means add down the walk exactly (per-atom partial sums)
    assert s.loc[('All', 'Margin', 'Total'), 'EX'] == pytest.approx(
        s.loc[('Gross', 'Margin', 'Total'), 'EX']
        + s.loc[('ceded occ', 'Margin', 'Total'), 'EX'], abs=1e-6)
    assert s.loc[('All', 'Margin', 'Impact'), 'EX'] == pytest.approx(
        s.loc[('ceded occ', 'Margin', 'Total'), 'EX'], abs=1e-6)
    # one shared joint -> the scenario (κ) ladder; every column foots
    assert 'κ01' in s.columns and 'P01' not in s.columns
    # the card is the exploded (Step, View) blocks + closing All block
    assert list(t.summary_df.index.names) == ['Step', 'View']


def test_xpnl_over_plain_engine_one_step_walk(uw):
    """``xpnl`` over a plain engine is the trivial **one-step walk**
    ([Decision-XPnL-Plain-Is-One-Step-Walk]): the (Step, View, Line) sheet
    with the single Gross block -- no grand rows, no impact."""
    t = uw('xpnl X 1000 premium less agg e 100 claims sev lognorm 100 cv 2 '
           'poisson')
    assert isinstance(t, PnL)
    s = t.stats_df
    assert list(s.index.names) == ['Step', 'View', 'Line']
    steps = list(dict.fromkeys(s.index.get_level_values('Step')))
    assert steps == ['Gross']
    assert ('All', 'Margin', 'Total') not in s.index
    assert list(t.summary_df.index) == [
        ('Gross', 'Consideration'), ('Gross', 'Obligation'),
        ('Gross', 'Margin')]
    # shared atoms -> the ladder stays the scenario (κ) pass
    assert 'κ01' in s.columns


def test_xpnl_over_port_not_implemented(uw):
    """``xpnl`` + ``port`` is rejected -- the total hides its units (decision 6)."""
    uw('agg U1 500 premium at 0.6 lr sev lognorm 50 cv 2 poisson')
    uw('port Bk agg.U1')
    with pytest.raises(NotImplementedError, match='xpnl'):
        uw('xpnl X inherit premium less port.Bk')


# ----------------------------------------------------------------------
# Standalone-agg economics: ignored with a warning
# ([Reins-Economics-On-Agg-Ignore-Warn]; supersedes the decision-7 hard error)
# ----------------------------------------------------------------------
def test_standalone_agg_cede_warns_and_builds(uw):
    """A bare ``agg`` with a ceded-premium clause builds its loss structure,
    warning that the economics are ignored (fold into a pnl to activate)."""
    from aggregate.constants import IgnoredDecLClauseWarning
    with pytest.warns(IgnoredDecLClauseWarning, match='ceded premium'):
        a = uw('agg A 1000 premium at 0.7 lr sev lognorm 100 cv 2 '
               'occurrence ceded to 500 xs 500 deposit 100 poisson')
    assert type(a).__name__ == 'Aggregate'
    assert a.occ_reins is not None
