"""DecL surface for property-cat reinstatement premiums (``[decl]`` workstream).

Covers the ``pnl ... occurrence net of <layer> <premium> reinstatements ...``
grammar, the transformer spec keys, the three locked validation rules, and the
``build() -> PnL`` plumbing. The reinstatement terms live on the engine
(``p.engine.reinstatement_terms``); every stochastic-cession fact is read off
the PnL's own ``economic_df`` (the group ledger) and the ``(L, R)`` joint source
(``p._source``) -- there is no separate analysis object
([Decommission-Analysis-Classes]).

These DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg`` under
the ``RI.*`` section. Because that corpus predates the SLY snapshot, the spec
assertions here are **hand-written** (parse -> assert ``occ_reins_reinst``), not
captured via ``tests/data/expected_specs.json`` -- see dev/plan-reinstatements.md
"Tests / DecL snapshot wrinkle".
"""

import numpy as np
import pytest

import aggregate
from aggregate import PnL, build
from aggregate.contract_terms import ReinstatementTerms

_UW = aggregate.Underwriter()


def _spec(program):
    """Parse one DecL program to its ``(kind, name, spec)`` triple."""
    return _UW.parser.parse(program)


def _joint_mean(pnl, fn):
    """Exact ``E[fn(L, R)]`` off the PnL's own ``(L, R)`` joint (``p._source``).

    Replaces the deleted analysis's ``_stats_df`` exact-moment store: the
    walk / consolidated ledgers ride this same joint, so a leg's exact mean is
    ``transformed_moments`` of its map straight off it.
    """
    return pnl._source.transformed_moments(fn)['mean']


def _leg(pnl, label):
    """One ledger leg's economic_df row, by ``Line`` label."""
    return pnl.economic_df.xs(label, level='Label').iloc[0]


_HUMAN = ('pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
          'occurrence net of 95% po 100 xs 100 rol 18% '
          'reinstatements 1 free and 1 at 50% and 2 at 100% poisson')
# the walk (xpnl) face: the 2-D step tower the ledger tests assert
# ([Decision-PnL-Is-Consolidated]: the pnl face is the consolidated net view)
_HUMANX = _HUMAN.replace('pnl Cat', 'xpnl Cat', 1)


# ----------------------------------------------------------------------
# grammar / transformer: the three surface forms reduce to the same tuple
# ----------------------------------------------------------------------
def test_group_chain_form_rates():
    _, _, spec = _spec(_HUMAN)
    assert spec['occ_reins_reinst'] == [(0.0, 0.5, 1.0, 1.0)]
    # the base premium clause rides alongside (the reinstatement base rate)
    assert spec['occ_reins_premium'] == [('rol', 0.18)]


def test_explicit_list_form_rates():
    _, _, spec = _spec(
        'pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% reinstatements [0 .5 1 1] poisson')
    assert spec['occ_reins_reinst'] == [(0.0, 0.5, 1.0, 1.0)]


def test_number_word_counts():
    _, _, spec = _spec(
        'pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% '
        'reinstatements one free and three at 100% poisson')
    assert spec['occ_reins_reinst'] == [(0.0, 1.0, 1.0, 1.0)]


def test_single_group_no_free():
    _, _, spec = _spec(
        'pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% reinstatements 3 at 100% poisson')
    assert spec['occ_reins_reinst'] == [(1.0, 1.0, 1.0)]


def test_deposit_base_premium_form():
    _, _, spec = _spec(
        'pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
        'occurrence ceded to 100 xs 100 deposit 1800 reinstatements [0 1 1] poisson')
    assert spec['occ_reins_reinst'] == [(0.0, 1.0, 1.0)]
    assert spec['occ_reins_premium'] == [('deposit', 1800.0)]


def test_no_reinstatements_marker():
    # ``no reinstatements`` -> the empty-tuple marker (m=0), distinct from the
    # omitted clause (None = free + unlimited).
    _, _, spec = _spec(
        'pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% no reinstatements poisson')
    assert spec['occ_reins_reinst'] == [()]


def test_no_reinstatements_is_a_single_annual_limit():
    # zero reinstatements: recovery capped at the single occurrence limit y, no
    # reinstatement premium, so the ceded premium is the deterministic deposit.
    # build returns a PnL; the terms live on the engine.
    p = build(
        'pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% no reinstatements poisson')
    t = p.engine.reinstatement_terms
    assert t.n_reinstatements == 0
    assert t.total_recovery_capacity == pytest.approx(100.0)   # single limit y
    assert t.recovery(250.0) == pytest.approx(100.0)
    assert t.reinstatement_premium(250.0) == pytest.approx(0.0)
    # deterministic ceded premium (= the deposit; h(R) == 0): with no
    # reinstatement premium and no aggregate cover, the consolidated net premium
    # P_G - D is fully deterministic, so its ledger row has zero variance (vs a
    # materially nonzero SD when reinstatements are paid).
    assert _joint_mean(p, lambda l, r: t.reinstatement_premium(r)) == \
        pytest.approx(0.0)
    assert _leg(p, 'net premium')['SD'] == pytest.approx(0.0, abs=1e-6)


def test_number_words_are_not_reserved_as_identifiers():
    # ``one`` is read as a count only in the reinstatement-count position; it
    # remains a valid identifier (severity name) everywhere else.
    k, n, _ = _spec('agg one 3 claims sev lognorm 50 cv 2 poisson')
    assert (k, n) == ('agg', 'one')


# ----------------------------------------------------------------------
# omitting the clause = ordinary occurrence layer (free + unlimited)
# ----------------------------------------------------------------------
def test_no_clause_means_no_terms():
    _, _, spec = _spec(
        'pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% poisson')
    assert 'occ_reins_reinst' not in spec
    # no reinstatements clause + a base premium clause -> a plain guaranteed-cost
    # reinsurance PnL: the engine carries no reinstatement terms.
    p = build(
        'pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% poisson')
    assert isinstance(p, PnL)
    assert not hasattr(p, 'terms')
    assert not hasattr(p.engine, 'reinstatement_terms')


# ----------------------------------------------------------------------
# validation rules (locked, dev/plan-reinstatements.md)
# ----------------------------------------------------------------------
def test_reins_premium_rule():
    # [reins-premium]: a reinstatements clause needs a base premium clause.
    with pytest.raises(ValueError, match=r'reins-premium'):
        _spec('pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 reinstatements [0 1] poisson')


def test_reins_single_layer_rule():
    # [reins-single-layer]: a reinstated layer + a plain second occ layer.
    with pytest.raises(ValueError, match=r'reins-single-layer'):
        _spec('pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 10% reinstatements [0 1] '
              'and 200 xs 200 poisson')


def test_reins_one_clause_rule():
    # [reins-one-clause]: two layers each carrying a reinstatements clause.
    with pytest.raises(ValueError, match=r'reins-one-clause'):
        _spec('pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 10% reinstatements [0 1] '
              'and 200 xs 200 rol 5% reinstatements [0 1] poisson')


def test_reinstatements_rejected_on_aggregate():
    with pytest.raises(ValueError, match=r'occurrence layer'):
        _spec('pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 poisson '
              'aggregate net of 2000 xs 3000 rol 8% reinstatements [0 1]')


def test_negative_rate_rejected():
    with pytest.raises(ValueError, match=r'nonnegative'):
        _spec('pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 18% reinstatements [0 -1] poisson')


# ----------------------------------------------------------------------
# build() return: a PnL wrapping the reinstated engine (terms on the engine)
# ----------------------------------------------------------------------
def test_build_returns_pnl_wrapping_engine():
    p = build(_HUMAN)
    # build returns a PnL value object wrapping the reinstated engine; the
    # terms and gross premium live on p.engine.
    assert isinstance(p, PnL)
    assert type(p.engine).__name__ == 'Aggregate'
    assert isinstance(p.engine.reinstatement_terms, ReinstatementTerms)
    assert p.engine.reinstatement_gross_premium == 10000.0


def test_terms_carry_share_scaled_limit_and_base_premium():
    # 95% po 100 xs 100 rol 18%: effective limit y = share*limit = 95; base
    # premium D = share*rol*limit = 17.1; base rate r = D/y = 0.18 = the rol.
    p = build(_HUMAN)
    t = p.engine.reinstatement_terms
    assert isinstance(t, ReinstatementTerms)
    assert t.limit == pytest.approx(95.0)
    assert t.deposit == pytest.approx(17.1)
    assert t.rol == pytest.approx(0.18)
    assert t.rates == (0.0, 0.5, 1.0, 1.0)


def test_ledger_rows_and_means_add():
    """The reinstatement walk (xpnl) is a two-group ledger over the (L, R)
    joint: gross sell group + occ cession buy group; the EX column adds down
    the sheet and the total impact is the cession's step delta."""
    p = build(_HUMANX)
    s = p.economic_df
    for row in (('Gross', 'Consideration', 'Premium'),
                ('Gross', 'Obligation', 'Loss'),
                ('Gross', 'Margin', 'Gross'),
                ('occ 95% so 100 xs 100', 'Consideration', 'occ 95% so 100 xs 100 premium'),
                ('occ 95% so 100 xs 100', 'Obligation', 'occ 95% so 100 xs 100 recovery'),
                ('occ 95% so 100 xs 100', 'Margin', 'Total'),
                ('occ 95% so 100 xs 100', 'Margin', 'Net'),
                ('All', 'Margin', 'Net'),
                ('All', 'Margin', 'Impact')):
        assert row in s.index, row
    assert s.loc[('All', 'Margin', 'Net'), 'EX'] == pytest.approx(
        s.loc[('Gross', 'Margin', 'Gross'), 'EX']
        + s.loc[('occ 95% so 100 xs 100', 'Margin', 'Total'), 'EX'],
        rel=1e-6, abs=1e-6)
    assert s.loc[('All', 'Margin', 'Impact'), 'EX'] == pytest.approx(
        s.loc[('occ 95% so 100 xs 100', 'Margin', 'Total'), 'EX'], abs=1e-9)


def test_ceded_premium_is_stochastic_in_exhibit():
    p = build(_HUMANX)
    s = p.economic_df
    # the headline effect: the stochastic ceded premium D + h(R) shows a
    # nonzero SD on its ledger row, while the gross premium is fixed...
    assert s.loc[('occ 95% so 100 xs 100', 'Consideration', 'occ 95% so 100 xs 100 premium'), 'SD'] \
        > 0.0
    assert s.loc[('Gross', 'Consideration', 'Premium'), 'SD'] == 0.0


def test_ledger_mean_check_ceded_premium():
    """E[the ceded premium row] = -(D + E[h(R)]) -- the plan's mean check."""
    p = build(_HUMANX)
    t = p.engine.reinstatement_terms
    e_h = _joint_mean(p, lambda l, r: t.reinstatement_premium(r))
    assert p.economic_df.loc[
        ('occ 95% so 100 xs 100', 'Consideration', 'occ 95% so 100 xs 100 premium'), 'EX'] == \
        pytest.approx(-(t.deposit + e_h), rel=1e-9)


def test_terms_and_premium_attached_to_engine():
    # build attaches the terms + gross premium to the engine and calls the
    # source builder arg-free internally, so the engine carries both.
    p = build(_HUMAN)
    assert isinstance(p, PnL)
    assert p.engine.reinstatement_gross_premium == 10000.0
    assert p.engine.reinstatement_terms.rates == (0.0, 0.5, 1.0, 1.0)


# ----------------------------------------------------------------------
# decision 3: a subsequent aggregate cover stays 2-D (builds; terms attached)
# ----------------------------------------------------------------------
def test_subsequent_aggregate_cover_builds():
    p = build(
        'xpnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% reinstatements [0 1] '
        'poisson aggregate net of 85% po 1500 xs 7000 deposit 600')
    # the reinstated occurrence layer + a genuine subsequent agg cover both build;
    # the agg tier is threaded from the pnl economics and rides the same joint.
    assert p.engine.reinstatement_terms is not None
    # threaded from the pnl; the 100%-quoted deposit 600 scaled by the 85%
    # placement -> 510
    assert p.economics['pc_agg'] == pytest.approx(510.0)
    # the agg cover actually recovers on the net-of-occurrence loss L - A(R):
    # its ledger recovery row carries a nonzero mean.
    s = p.economic_df
    assert abs(s.loc[('agg 85% so 1500 xs 7000', 'Obligation', 'agg 85% so 1500 xs 7000 recovery'),
                     'EX']) > 0
    # decision 3: the ledger extends to the inuring both-tiers form -- a third
    # buy group over the SAME joint (no new dimension).
    for row in (('Gross', 'Margin', 'Gross'),
                ('occ 100 xs 100', 'Margin', 'Total'),
                ('occ 100 xs 100', 'Margin', 'Net'),
                ('agg 85% so 1500 xs 7000', 'Consideration', 'agg 85% so 1500 xs 7000 premium'),
                ('agg 85% so 1500 xs 7000', 'Obligation', 'agg 85% so 1500 xs 7000 recovery'),
                ('agg 85% so 1500 xs 7000', 'Margin', 'Total'),
                ('agg 85% so 1500 xs 7000', 'Margin', 'Net'),
                ('All', 'Margin', 'Net')):
        assert row in s.index, row
    # means add tier by tier down the sheet
    assert s.loc[('occ 100 xs 100', 'Margin', 'Net'), 'EX'] == pytest.approx(
        s.loc[('Gross', 'Margin', 'Gross'), 'EX']
        + s.loc[('occ 100 xs 100', 'Margin', 'Total'), 'EX'],
        rel=1e-6, abs=1e-6)
    assert s.loc[('All', 'Margin', 'Net'), 'EX'] == pytest.approx(
        s.loc[('occ 100 xs 100', 'Margin', 'Net'), 'EX']
        + s.loc[('agg 85% so 1500 xs 7000', 'Margin', 'Total'), 'EX'],
        rel=1e-6, abs=1e-6)


def test_expense_and_cede_book_as_ledger_legs():
    # a reinstatement pnl with gross expenses + a flat ceding commission books
    # them as cash-flow legs: the and-joined expense group on the gross group,
    # the commission as a received leg on the cession.
    p = build('xpnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 18% cede 20% reinstatements [0 1] '
              'poisson less 500 fixed expense and 10% premium expense')
    assert p.economics['c_occ'] == pytest.approx(3.6)      # 20% * (18% * 100)
    s = p.economic_df
    assert s.loc[('Gross', 'Obligation', 'expense'), 'EX'] == \
        pytest.approx(-1500.0)                             # 500 + 10% * 10000
    assert s.loc[('occ 100 xs 100', 'Obligation', 'occ 100 xs 100 commission'),
                 'EX'] == pytest.approx(3.6)
    # the gross group result books the whole expense vs the pure gross UW
    # (premium + loss legs)
    pure_gross = (s.loc[('Gross', 'Consideration', 'Premium'), 'EX']
                  + s.loc[('Gross', 'Obligation', 'Loss'), 'EX'])
    assert s.loc[('Gross', 'Margin', 'Gross'), 'EX'] == pytest.approx(
        pure_gross - 1500.0, rel=1e-6, abs=1e-6)
    # and the cession result credits its commission vs the pure ceded UW
    # (ceded premium + recovery legs)
    pure_ceded = (
        s.loc[('occ 100 xs 100', 'Consideration', 'occ 100 xs 100 premium'), 'EX']
        + s.loc[('occ 100 xs 100', 'Obligation', 'occ 100 xs 100 recovery'), 'EX'])
    assert s.loc[('occ 100 xs 100', 'Margin', 'Total'), 'EX'] == pytest.approx(
        pure_ceded + 3.6, rel=1e-6, abs=1e-6)


def test_no_expense_leaves_ledger_pure():
    # without expenses the gross group result is the pure gross underwriting
    # mean (P_G - L).
    p = build('xpnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 18% reinstatements [0 1] poisson')
    s = p.economic_df
    # no expense / commission legs on the ledger
    assert ('Gross', 'Obligation', 'expense') not in s.index
    assert p.economics.get('c_occ', 0.0) == pytest.approx(0.0)
    # the gross group result is the pure gross underwriting mean (premium + loss)
    pure_gross = (s.loc[('Gross', 'Consideration', 'Premium'), 'EX']
                  + s.loc[('Gross', 'Obligation', 'Loss'), 'EX'])
    assert s.loc[('Gross', 'Margin', 'Gross'), 'EX'] == \
        pytest.approx(pure_gross, rel=1e-6, abs=1e-6)


# ----------------------------------------------------------------------
# the consolidated (pnl) face ([Decision-PnL-Is-Consolidated]; closes
# [2D-Deferred]): one sell group of 2-D legs over the SAME joint
# ----------------------------------------------------------------------
def test_consolidated_face_shape_and_exact_means():
    p = build(_HUMAN)
    s = p.economic_df
    assert list(s.index.names) == ['Side', 'Label']
    assert list(p.summary_df.index) == ['Consideration', 'Obligation',
                                        'Margin']
    t = p.engine.reinstatement_terms
    P_G = p.engine.reinstatement_gross_premium
    # net premium = P_G - D - E[h(R)] (no cede here); stochastic (h(R))
    e_h = _joint_mean(p, lambda l, r: t.reinstatement_premium(r))
    row = s.xs('net premium', level='Label').iloc[0]
    assert row['EX'] == pytest.approx(P_G - t.deposit - e_h, rel=1e-9)
    assert row['SD'] > 0
    # loss (net) = -E[L - A(R)]
    e_net_loss = _joint_mean(p, lambda l, r: l - t.recovery(r))
    assert s.xs('Loss (net)', level='Label').iloc[0]['EX'] == pytest.approx(
        -e_net_loss, rel=1e-9)
    # one shared joint -> scenario ladder
    assert 'κ01' in s.columns
    assert p.economics['pc_occ'] == pytest.approx(17.1)


def test_consolidated_agrees_with_walk_exactly():
    """Both faces are pushforwards of the ONE joint, so the consolidated
    net position equals the walk's grand result exactly -- no engine-drift
    tolerance (unlike the guaranteed-cost occ case)."""
    p = build(_HUMAN)
    x = build(_HUMANX)
    assert p.est_m == pytest.approx(
        x.economic_df.loc[('All', 'Margin', 'Net'), 'EX'], abs=1e-12)
