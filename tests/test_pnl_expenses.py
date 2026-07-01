"""Gross expenses on a ``pnl`` (Phase 1, decision 1).

Three explicit forms -- a fraction of premium, a fraction of expected loss, or a
fixed currency amount -- plus the absent default. The expense books a gross
obligation leg (named ``'expense'``); it reduces the result and feeds the
combined ratio. See ``dev/done/plan-pnl-expenses-ceded-premium.md``.

The old ``_gross_expense()`` / ``_expense_spec`` internals are gone: an expense
now surfaces as an obligation leg in :attr:`PnL.summary_df` (its ``EX`` a
magnitude), and the resolver is :func:`aggregate._pnl.resolve_expense`.
"""
import pytest

from aggregate import build
from aggregate._pnl import resolve_expense

TOL = 1e-3
_BASE = 'pnl A 1000 prem less 8 claims sev lognorm 50 cv 1 poisson'  # E[loss] = 8 * 50 = 400


def test_expense_absent_defaults_to_zero():
    # no expense clause -> no expense obligation leg
    assert 'expense' not in build(_BASE).summary_df.index


def test_expense_three_explicit_forms():
    # premium basis: 25% of the 1000 gross premium
    assert build(_BASE + ' 25% premium expenses').summary_df.loc[
        'expense', 'EX'] == pytest.approx(250.0)
    # loss basis: a fraction of the ACTUAL loss (now stochastic -- rate * loss per
    # atom); its EX is still rate * E[loss] = 0.30 * 400 = 120
    assert build(_BASE + ' 30% loss expenses').summary_df.loc[
        'expense', 'EX'] == pytest.approx(120.0, rel=TOL)
    # fixed basis: a currency amount
    assert build(_BASE + ' 200 fixed expenses').summary_df.loc[
        'expense', 'EX'] == pytest.approx(200.0)


def test_loss_basis_expense_is_stochastic():
    """A plain loss-basis (LAE) expense scales with the actual loss, not a point mass.

    The expense leg is ``rate * loss`` per atom, so it carries the loss's spread:
    its CV equals the loss CV (an old point mass at ``rate * E[loss]`` had CV 0).
    """
    df = build(_BASE + ' 30% loss expenses').summary_df
    assert df.loc['expense', 'SD'] > 0
    assert df.loc['expense', 'CV'] == pytest.approx(df.loc['loss', 'CV'], rel=TOL)


def test_expense_basis_is_resolved_by_basis():
    """The three bases resolve distinctly: premium scales, fixed is constant."""
    agg = build('agg A 8 claims sev lognorm 50 cv 1 poisson')      # E[loss] = 400
    # a premium-basis term scales with the gross premium
    assert resolve_expense(agg, [('premium', 0.25)], 1000) == pytest.approx(250.0)
    assert resolve_expense(agg, [('premium', 0.25)], 2000) == pytest.approx(500.0)
    # a fixed-basis term is constant in the premium
    assert resolve_expense(agg, [('fixed', 200.0)], 1000) == pytest.approx(200.0)
    assert resolve_expense(agg, [('fixed', 200.0)], 2000) == pytest.approx(200.0)
    # a loss-basis term is a fraction of the expected gross loss
    assert resolve_expense(agg, [('loss', 0.30)], 1000) == pytest.approx(120.0, rel=TOL)


def test_expense_singular_alias():
    # ``expense`` and ``expenses`` are both accepted
    assert build(_BASE + ' 200 fixed expense').summary_df.loc[
        'expense', 'EX'] == pytest.approx(200.0)


def test_multiple_expense_terms_sum():
    # ``and``-joined terms sum: 25% of 1000 premium + 1000 fixed = 1250
    p = build(_BASE + ' 25% premium expense and 1000 fixed expense')
    assert p.summary_df.loc['expense', 'EX'] == pytest.approx(0.25 * 1000 + 1000.0)
    # three terms, mixing all bases (E[loss] = 400)
    p3 = build(_BASE + ' 10% premium expense and 5% loss expense and 50 fixed expense')
    assert p3.summary_df.loc['expense', 'EX'] == pytest.approx(
        0.10 * 1000 + 0.05 * 400 + 50.0, rel=TOL)


def test_single_tuple_expense_spec_still_accepted_via_api():
    # the Python API single-term tuple form normalizes correctly
    a = build('agg R 100 claims sev lognorm 50 cv 1.5 poisson '
              'aggregate net of 2000 xs 3000')
    p = a.make_pnl(gross=5500, ceded=1800, expense_spec=('fixed', 300.0))
    # net perspective expense == gross expense (no commission credit) == 300
    assert p.summary_df.loc['expense', 'EX'] == pytest.approx(300.0)


def test_expense_reduces_margin_and_drives_combined_ratio():
    p = build(_BASE + ' 200 fixed expenses')
    df = p.summary_df
    assert df.loc['expense', 'EX'] == pytest.approx(200.0)   # a magnitude
    assert df.loc['expense', 'SD'] == pytest.approx(0.0, abs=1e-2)  # deterministic
    # obligation legs add: loss + expense = Total obligation
    assert (df.loc['loss', 'EX'] + df.loc['expense', 'EX']
            == pytest.approx(df.loc['Total obligation', 'EX'], abs=1e-6))
    # the result is reduced by the expense: margin = consideration - Total obligation
    assert (df.loc['margin', 'EX']
            == pytest.approx(df.loc['consideration', 'EX']
                             - df.loc['Total obligation', 'EX'], abs=1e-6))
    # combined ratio = (loss + expense) / premium = 600 / 1000
    combined = ((df.loc['loss', 'EX'] + df.loc['expense', 'EX'])
                / df.loc['consideration', 'EX'])
    assert combined == pytest.approx((400 + 200) / 1000, rel=TOL)


def test_expense_in_gcn_exhibit():
    a = build('agg R 100 claims sev lognorm 50 cv 1.5 poisson '
              'aggregate net of 2000 xs 3000')
    p = a.make_pnl(gross=5500, ceded=1800, expense_spec=('premium', 0.2))
    # net perspective expense == gross expense (no commission yet) == 0.2 * 5500
    assert p.summary_df.loc['expense', 'EX'] == pytest.approx(1100.0)
    # the expense ratio reads off the gross premium (economics via the tower)
    assert 1100.0 / p.tower.economics['gross'] == pytest.approx(0.2, rel=TOL)


def test_expense_only_on_pnl_not_agg():
    # the grammar only attaches an expense clause to a pnl; an agg has no premium
    with pytest.raises(Exception):
        build('agg Z 8 claims sev lognorm 50 cv 1 poisson 25% premium expenses')
