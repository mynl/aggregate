"""Gross expenses on a ``pnl`` (signed ledger form).

Three explicit forms -- a fraction of premium, a fraction of expected loss, or a
fixed currency amount -- plus the absent default. The expense books a gross
obligation leg (named ``'expense'``, **signed**: a sold cost shows negative);
it reduces the result and feeds the combined ratio. The leg resolver is
:func:`aggregate._pnl_builders._resolve_expense_split`; the scalar
:func:`aggregate._pnl_builders.resolve_expense` serves deterministic economics
only and never feeds a leg. See ``dev/plan-yapnl.md``.
"""
import pytest

from aggregate import build
from aggregate._pnl_builders import resolve_expense

TOL = 1e-3
_BASE = 'pnl A 1000 prem less agg A_e 8 claims sev lognorm 50 cv 1 poisson'  # E[loss] = 8 * 50 = 400


def _leg(pnl, label):
    """The economic_df row of one declared leg, looked up by its Line label
    (leg-level detail lives on the stats sheet; summary_df is the fixed card)."""
    return pnl.economic_df.xs(label, level='Label').iloc[0]


def _lines(pnl):
    return list(pnl.economic_df.index.get_level_values('Label'))


def test_expense_absent_defaults_to_zero():
    # no expense clause -> no expense obligation leg
    assert 'expense' not in _lines(build(_BASE))


def test_expense_three_explicit_forms():
    # premium basis: 25% of the 1000 gross premium, booked signed (-)
    assert _leg(build(_BASE + ' less 25% premium expenses'), 'expense')['EX'] \
        == pytest.approx(-250.0)
    # loss basis: a fraction of the ACTUAL loss (stochastic -- rate * loss per
    # atom); its EX is rate * E[loss] = 0.30 * 400 = 120, booked -120
    assert _leg(build(_BASE + ' less 30% loss expenses'), 'expense')['EX'] \
        == pytest.approx(-120.0, rel=TOL)
    # fixed basis: a currency amount
    assert _leg(build(_BASE + ' less 200 fixed expenses'), 'expense')['EX'] \
        == pytest.approx(-200.0)


def test_loss_basis_expense_is_stochastic():
    """A plain loss-basis (LAE) expense scales with the actual loss, not a point mass.

    The expense leg is ``rate * loss`` per atom, so it carries the loss's spread:
    its CV equals the loss CV (an old point mass at ``rate * E[loss]`` had CV 0).
    """
    p = build(_BASE + ' less 30% loss expenses')
    assert _leg(p, 'expense')['SD'] > 0
    assert _leg(p, 'expense')['CV'] == \
        pytest.approx(_leg(p, 'Loss')['CV'], rel=TOL)


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
    assert _leg(build(_BASE + ' less 200 fixed expense'), 'expense')['EX'] \
        == pytest.approx(-200.0)


def test_multiple_expense_terms_sum():
    # ``and``-joined terms sum: 25% of 1000 premium + 1000 fixed = 1250 (signed)
    p = build(_BASE + ' less 25% premium expense and 1000 fixed expense')
    assert _leg(p, 'expense')['EX'] == pytest.approx(-(0.25 * 1000 + 1000.0))
    # three terms, mixing all bases (E[loss] = 400)
    p3 = build(_BASE + ' less 10% premium expense and 5% loss expense and 50 fixed expense')
    assert _leg(p3, 'expense')['EX'] == pytest.approx(
        -(0.10 * 1000 + 0.05 * 400 + 50.0), rel=TOL)


def test_single_tuple_expense_spec_still_accepted_via_api():
    # the Python API single-term tuple form normalizes correctly; in the gcn
    # ledger the expense books on the gross group
    a = build('agg R 100 claims sev lognorm 50 cv 1.5 poisson '
              'aggregate net of 2000 xs 3000')
    p = a.make_pnl(gross=5500, ceded=1800, expense_spec=('fixed', 300.0))
    assert _leg(p, 'expense')['EX'] == pytest.approx(-300.0)


def test_expense_reduces_margin_and_drives_combined_ratio():
    p = build(_BASE + ' less 200 fixed expenses')
    s = p.economic_df
    assert s.loc[('Obligation', 'expense'), 'EX'] == pytest.approx(-200.0)
    assert s.loc[('Obligation', 'expense'), 'SD'] == \
        pytest.approx(0.0, abs=1e-2)                     # deterministic
    # obligation legs add: loss + expense = total obligation (signed)
    assert (s.loc[('Obligation', 'Loss'), 'EX']
            + s.loc[('Obligation', 'expense'), 'EX']
            == pytest.approx(s.loc[('Obligation', 'Total'), 'EX'], abs=1e-6))
    # the EX column foots: margin = consideration + total obligation
    assert (s.loc[('Margin', 'Total'), 'EX']
            == pytest.approx(s.loc[('Consideration', 'Premium'), 'EX']
                             + s.loc[('Obligation', 'Total'), 'EX'], abs=1e-6))
    # economic_ratios_df tells the same story, and now splits loss from expense: the
    # combined ratio is (400 + 200) / 1000, its two parts 0.40 and 0.20
    r = p.economic_ratios_df.iloc[0]
    assert r['LR'] == pytest.approx(400 / 1000, rel=TOL)
    assert r['ER'] == pytest.approx(200 / 1000, rel=TOL)
    assert r['CR'] == pytest.approx((400 + 200) / 1000, rel=TOL)


def test_expense_in_gcn_ledger():
    a = build('agg R 100 claims sev lognorm 50 cv 1.5 poisson '
              'aggregate net of 2000 xs 3000')
    p = a.make_pnl(gross=5500, ceded=1800, expense_spec=('premium', 0.2))
    # the expense books on the gross group: -0.2 * 5500
    assert _leg(p, 'expense')['EX'] == pytest.approx(-1100.0)
    # the expense ratio reads off the gross premium (economics on the PnL)
    assert 1100.0 / p.economics['gross'] == pytest.approx(0.2, rel=TOL)


def test_expense_only_on_pnl_not_agg():
    # the grammar only attaches an expense clause to a pnl; an agg has no premium
    with pytest.raises(Exception):
        build('agg Z 8 claims sev lognorm 50 cv 1 poisson 25% premium expenses')
