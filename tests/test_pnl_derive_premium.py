"""Tests for [Derived-Premium]: the ``derive premium`` head on ``pnl``/``xpnl``.

``derive premium`` reads the engine's technical premium T as risk loaded and
grosses it up for the ``less`` clause expenses: with fixed expense total F and
premium expense ratio total r, the derived premium is ``(T + F) / (1 - r)``,
so premium net of expenses returns exactly T and the expected underwriting
result carries the engine risk load and nothing else. Fixed and premium bases
only; a loss basis expense, an engine without premium, and ratios totalling
one or more are build errors. See dev/plan-derived-premium.md.
"""

import pytest

from aggregate import build, Underwriter
from aggregate._pnl import PnL
from aggregate._pnl_builders import derive_consideration, resolve_expense
from aggregate.decl_writer import format_program
from aggregate.parser import DERIVE_PREMIUM

TOL = 1e-9

# The worked example from the plan: T = 100, F = 10 + 2, r = 5% + 15% + 7%.
ENGINE = 'agg DRVe 100 premium at 0.65 lr sev lognorm 50 cv 0.8 poisson'
WORKED = (f'pnl DRV.Worked derive premium less {ENGINE} '
          'less 10 fixed expense as "App Fee" 2 fixed expense as "Admin fee" '
          '5% premium expense as "TLF" 15% premium expense as "Commission" '
          '7% premium expense as "G&A"')
WORKED_PREMIUM = (100 + 12) / (1 - 0.27)


@pytest.fixture
def uw():
    """A clean underwriter (no databases) with eager update."""
    return Underwriter(databases=None, update=True)


def _booked_premium(face):
    """The consideration a built :class:`PnL` actually books."""
    legs = face.legs_df
    return float(legs.loc[legs['kind'] == 'premium', 'EX'].iloc[0])


# ----------------------------------------------------------------------
# parse
# ----------------------------------------------------------------------
def test_parse_records_the_sentinel():
    """The head parses to the DERIVE_PREMIUM sentinel in the spec."""
    uw = Underwriter(databases=None, update=False)
    kind, name, spec = uw.parser.parse(uw.lexer.tokenize(WORKED))
    assert kind == 'pnl'
    assert spec['consideration'] is DERIVE_PREMIUM
    assert repr(DERIVE_PREMIUM) == 'DERIVE_PREMIUM'


# ----------------------------------------------------------------------
# the gross up
# ----------------------------------------------------------------------
def test_worked_example(uw):
    """T = 100, F = 12, r = 0.27 books (100 + 12) / 0.73."""
    p = uw(WORKED)
    assert isinstance(p, PnL)
    assert _booked_premium(p) == pytest.approx(WORKED_PREMIUM, abs=TOL)


def test_premium_net_of_expenses_returns_the_technical_premium(uw):
    """The defining invariant: P less its own expenses equals T exactly."""
    p = uw(WORKED)
    booked = _booked_premium(p)
    expenses = resolve_expense(
        p.engine, [(None, [('fixed', 12.0), ('premium', 0.27)])], booked)
    assert booked - expenses == pytest.approx(100.0, abs=TOL)


def test_multiple_fixed_terms_add(uw):
    """Two fixed terms behave as their sum."""
    two = uw(f'pnl DRV.F2 derive premium less {ENGINE} '
             'less 10 fixed expense and 2 fixed expense')
    one = uw(f'pnl DRV.F1 derive premium less {ENGINE} '
             'less 12 fixed expense')
    assert _booked_premium(two) == pytest.approx(_booked_premium(one), abs=TOL)
    assert _booked_premium(two) == pytest.approx(112.0, abs=TOL)


def test_multiple_premium_terms_add_combined_ratio_style(uw):
    """Premium ratios add whether and-joined, juxtaposed, or written as one."""
    joined = uw(f'pnl DRV.R2 derive premium less {ENGINE} '
                'less 5% premium expense and 20% premium expense')
    grouped = uw(f'pnl DRV.R3 derive premium less {ENGINE} '
                 'less 5% premium expense 20% premium expense')
    single = uw(f'pnl DRV.R1 derive premium less {ENGINE} '
                'less 25% premium expense')
    assert _booked_premium(joined) == pytest.approx(100 / 0.75, abs=TOL)
    assert _booked_premium(grouped) == pytest.approx(100 / 0.75, abs=TOL)
    assert _booked_premium(single) == pytest.approx(100 / 0.75, abs=TOL)


def test_no_expense_clause_equals_inherit(uw):
    """An expense-free derivation is the identity, exactly ``inherit premium``."""
    derived = uw(f'pnl DRV.NoExp derive premium less {ENGINE}')
    inherited = uw(f'pnl DRV.Inh inherit premium less {ENGINE}')
    assert _booked_premium(derived) == pytest.approx(100.0, abs=TOL)
    assert _booked_premium(derived) == pytest.approx(
        _booked_premium(inherited), abs=TOL)


def test_reads_the_engine_fyi_premium(uw):
    """The FYI premium on a claims head is a technical premium to derive from."""
    p = uw('pnl DRV.Fyi derive premium less agg DRVFyiE 5 claims '
           '20000 premium sev lognorm 100 cv 2 poisson '
           'less 20% premium expenses')
    assert _booked_premium(p) == pytest.approx(20000 / 0.8, abs=1e-6)


# ----------------------------------------------------------------------
# errors
# ----------------------------------------------------------------------
def test_loss_basis_expense_errors(uw):
    """Losses are not reliably known by inspection, so the gross up refuses."""
    with pytest.raises(ValueError, match='loss basis'):
        uw(f'pnl DRV.BadLoss derive premium less {ENGINE} '
           'less 10% loss expenses')


def test_no_engine_premium_errors(uw):
    """Nothing to derive from mirrors the inherit error."""
    with pytest.raises(ValueError, match='derive'):
        uw('pnl DRV.BadNoPrem derive premium less agg DRVNoPrem 10 claims '
           'sev lognorm 50 cv 0.8 poisson less 25% premium expenses')


def test_premium_ratios_at_or_above_one_error(uw):
    """No finite premium grosses up a 100 percent expense ratio."""
    with pytest.raises(ValueError, match='no finite premium'):
        uw(f'pnl DRV.BadRatio derive premium less {ENGINE} '
           'less 60% premium expenses and 40% premium expenses')


def test_derive_consideration_direct():
    """The helper stands alone: formula, empty spec, and both refusals."""
    assert derive_consideration(
        [(None, [('fixed', 12.0), ('premium', 0.27)])], 100.0, 'X') \
        == pytest.approx(WORKED_PREMIUM, abs=TOL)
    assert derive_consideration(None, 100.0, 'X') == pytest.approx(100.0)
    with pytest.raises(ValueError, match='loss basis'):
        derive_consideration([('loss', 0.1)], 100.0, 'X')
    with pytest.raises(ValueError, match='no finite premium'):
        derive_consideration([('premium', 1.0)], 100.0, 'X')


# ----------------------------------------------------------------------
# engines: port and xpnl
# ----------------------------------------------------------------------
def test_port_engine_derives(uw):
    """The accumulated portfolio premium grosses up, parallel to inherit."""
    p = uw('pnl DRV.Port derive premium less '
           'port DRVPB agg DRVU1 50 premium at 0.5 lr '
           'sev lognorm 20 cv 0.6 poisson '
           'agg DRVU2 50 premium at 0.5 lr sev lognorm 30 cv 0.9 poisson '
           'less 20% premium expenses')
    assert _booked_premium(p) == pytest.approx(100 / 0.8, abs=1e-6)


def test_port_engine_without_premium_errors(uw):
    """A claims-only portfolio has nothing to derive from."""
    with pytest.raises(ValueError, match='derive'):
        uw('pnl DRV.PortBad derive premium less '
           'port DRVPBad agg DRVUB 10 claims sev lognorm 20 cv 0.6 poisson '
           'less 20% premium expenses')


def test_xpnl_derives(uw):
    """``xpnl`` shares the premium head, so the walk derives too."""
    p = uw('xpnl DRV.X derive premium less '
           'agg DRVXE 100 premium at 0.65 lr sev lognorm 50 cv 0.8 '
           'occurrence net of 30 xs 30 deposit 5 poisson '
           'less 25% premium expenses')
    assert _booked_premium(p) == pytest.approx(100 / 0.75, abs=1e-6)


# ----------------------------------------------------------------------
# round trip
# ----------------------------------------------------------------------
def test_round_trips_through_the_writer(uw):
    """The head renders back as ``derive premium`` and rebuilds identically."""
    out = format_program(WORKED)
    assert 'derive premium' in out
    assert format_program(out) == out
    p = uw(out)
    assert _booked_premium(p) == pytest.approx(WORKED_PREMIUM, abs=TOL)


def test_head_label_round_trips(uw):
    """The consideration ``as`` label survives the writer."""
    prog = (f'pnl DRV.Lab derive premium as "Booked" less {ENGINE} '
            'less 25% premium expenses')
    out = format_program(prog)
    assert 'derive premium as Booked' in out
    p = uw(prog)
    legs = p.legs_df
    assert 'Booked' in legs['Label'].tolist()
