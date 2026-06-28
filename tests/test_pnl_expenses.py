"""Gross expenses on a ``pnl`` (Phase 1, decision 1).

Three explicit forms -- a fraction of premium, a fraction of expected loss, or a
fixed currency amount -- plus the absent default. The expense books a gross
obligation; it reduces the margin and feeds the combined ratio. See
``dev/done/plan-pnl-expenses-ceded-premium.md``.
"""
import pytest

from aggregate import build

TOL = 1e-3
_BASE = 'pnl A 1000 prem less 8 claims sev lognorm 50 cv 1 poisson'  # E[loss] = 8 * 50 = 400


def test_expense_absent_defaults_to_zero():
    assert build(_BASE)._gross_expense() == 0.0
    assert build(_BASE)._expense_spec is None


def test_expense_three_explicit_forms():
    assert build(_BASE + ' 25% premium expenses')._gross_expense() == pytest.approx(250.0)
    # loss basis: a fraction of EXPECTED gross loss (deterministic in Phase 1)
    assert build(_BASE + ' 30% loss expenses')._gross_expense() == pytest.approx(120.0, rel=TOL)
    assert build(_BASE + ' 200 fixed expenses')._gross_expense() == pytest.approx(200.0)


def test_expense_basis_is_stored_explicitly():
    assert build(_BASE + ' 25% premium expenses')._expense_spec == ('premium', 0.25)
    assert build(_BASE + ' 200 fixed expenses')._expense_spec == ('fixed', 200.0)


def test_expense_singular_alias():
    # ``expense`` and ``expenses`` are both accepted
    assert build(_BASE + ' 200 fixed expense')._expense_spec == ('fixed', 200.0)


def test_expense_reduces_margin_and_drives_combined_ratio():
    p = build(_BASE + ' 200 fixed expenses')
    df = p.summary_df
    assert df.loc['Expense', 'EX'] == pytest.approx(-200.0)
    assert df.loc['Expense', 'SD'] == 0.0                 # deterministic in Phase 1
    # Consideration + Obligation + Expense = Margin
    assert (df.loc['Consideration', 'EX'] + df.loc['Obligation', 'EX']
            + df.loc['Expense', 'EX']
            == pytest.approx(df.loc['Margin', 'EX'], abs=1e-6))
    # combined ratio = (loss + expense) / premium
    assert df.loc['Combined ratio', 'EX'] == pytest.approx((400 + 200) / 1000, rel=TOL)


def test_expense_in_gcn_exhibit():
    a = build('agg R 100 claims sev lognorm 50 cv 1.5 poisson '
              'aggregate net of 2000 xs 3000')
    p = a.make_pnl(gross=5500, ceded=1800, expense_spec=('premium', 0.2))
    g = p.gcn_df
    m = g.xs('Mean')
    assert m.loc['Expense', 'gross'] == pytest.approx(-1100.0)   # 0.2 * 5500
    assert m.loc['Expense', 'net'] == pytest.approx(-1100.0)     # no commission yet
    # the Expense row is additive across the split, like every Mean row
    assert m.loc['Expense', 'net'] == pytest.approx(
        m.loc['Expense', 'gross'] + m.loc['Expense', 'ceded'])
    # the expense ratio reads off the means
    assert g.loc[('Ratio', 'ER'), 'gross'] == pytest.approx(0.2, rel=TOL)


def test_expense_only_on_pnl_not_agg():
    # the grammar only attaches an expense clause to a pnl; an agg has no premium
    with pytest.raises(Exception):
        build('agg Z 8 claims sev lognorm 50 cv 1 poisson 25% premium expenses')
