"""DecL surface for property-cat reinstatement premiums (``[decl]`` workstream).

Covers the ``pnl ... occurrence net of <layer> <premium> reinstatements ...``
grammar, the transformer spec keys, the three locked validation rules, and the
``build() -> PnL`` plumbing that backs the GCN exhibit with a lazily built
:class:`~aggregate.reinstatement.ReinstatementAnalysis`.

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
from aggregate.reinstatement import ReinstatementAnalysis, ReinstatementTerms

_UW = aggregate.Underwriter()


def _spec(program):
    """Parse one DecL program to its ``(kind, name, spec)`` triple."""
    return _UW.parser.parse(program)


_HUMAN = ('pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
          'occurrence net of 95% po 100 xs 100 rol 18% '
          'reinstatements 1 free and 1 at 50% and 2 at 100% poisson')


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
    # build returns a PnL; the analysis is attached as p.analysis.
    p = build(
        'pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% no reinstatements poisson')
    a = p.analysis
    t = a.terms
    assert t.n_reinstatements == 0
    assert t.total_recovery_capacity == pytest.approx(100.0)   # single limit y
    assert t.recovery(250.0) == pytest.approx(100.0)
    assert t.reinstatement_premium(250.0) == pytest.approx(0.0)
    # deterministic ceded premium (= the deposit; h(R) == 0) => the exact ceded
    # premium has zero variance, vs a materially nonzero CV when reinstatements
    # are paid.
    assert a._stats_df.loc['reinstatement_premium', 'mean'] == pytest.approx(0.0)
    assert a._stats_df.loc['ceded_premium', 'sd'] == pytest.approx(0.0, abs=1e-6)
    assert a._stats_df.loc['ceded_premium', 'cv'] == pytest.approx(0.0, abs=1e-6)


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
    # no reinstatements clause + a base premium clause -> a plain PnLTower, NOT a
    # ReinstatementAnalysis: it carries no reinstatement terms / analysis surface.
    p = build(
        'pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% poisson')
    assert not isinstance(p, ReinstatementAnalysis)
    assert not hasattr(p, 'terms')


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
# build() return: an analysis-backed PnL (p.analysis IS the analysis)
# ----------------------------------------------------------------------
def test_build_returns_analysis_backed_pnl():
    p = build(_HUMAN)
    # build now returns a PnL value object, with the ReinstatementAnalysis
    # attached as p.analysis.
    assert isinstance(p, PnL)
    assert isinstance(p.analysis, ReinstatementAnalysis)
    assert p.analysis.gross_premium == 10000.0


def test_terms_carry_share_scaled_limit_and_base_premium():
    # 95% po 100 xs 100 rol 18%: effective limit y = share*limit = 95; base
    # premium D = share*rol*limit = 17.1; base rate r = D/y = 0.18 = the rol.
    p = build(_HUMAN)
    t = p.analysis.terms
    assert isinstance(t, ReinstatementTerms)
    assert t.limit == pytest.approx(95.0)
    assert t.deposit == pytest.approx(17.1)
    assert t.rol == pytest.approx(0.18)
    assert t.rates == (0.0, 0.5, 1.0, 1.0)


def test_ledger_rows_and_means_add():
    """The reinstatement pnl is a two-group ledger over the (L, R) joint:
    gross sell group + occ cession buy group; the EX column adds down the
    sheet and the total impact is the cession's step delta."""
    p = build(_HUMAN)
    s = p.summary_df
    for row in ('premium', 'loss', 'gross result', 'ceded occ premium',
                'ceded occ recovery', 'ceded occ result',
                'net through ceded occ', 'margin', 'total impact'):
        assert row in s.index, row
    assert s.loc['margin', 'EX'] == pytest.approx(
        s.loc['gross result', 'EX'] + s.loc['ceded occ result', 'EX'],
        rel=1e-6, abs=1e-6)
    assert s.loc['total impact', 'EX'] == pytest.approx(
        s.loc['ceded occ result', 'EX'], abs=1e-9)


def test_ceded_premium_is_stochastic_in_exhibit():
    p = build(_HUMAN)
    s = p.summary_df
    # the headline effect: the stochastic ceded premium D + h(R) shows a
    # nonzero SD on its ledger row, while the gross premium is fixed...
    assert s.loc['ceded occ premium', 'SD'] > 0.0
    assert s.loc['premium', 'SD'] == 0.0
    # ...and the analysis's exact engine agrees.
    assert p.analysis._stats_df.loc['ceded_premium', 'cv'] > 0.0
    assert p.analysis._stats_df.loc['gross_premium', 'cv'] == 0.0


def test_ledger_mean_check_ceded_premium():
    """E[ceded occ premium row] = -(D + E[h(R)]) -- the plan's mean check."""
    p = build(_HUMAN)
    a = p.analysis
    e_h = a._stats_df.loc['reinstatement_premium', 'mean']
    assert p.summary_df.loc['ceded occ premium', 'EX'] == pytest.approx(
        -(a.terms.deposit + e_h), rel=1e-9)


def test_arg_free_programmatic_entry_point():
    # build attaches terms + gross premium and calls the programmatic entry point
    # arg-free internally, so the attached analysis already carries both.
    p = build(_HUMAN)
    assert isinstance(p, PnL)
    assert isinstance(p.analysis, ReinstatementAnalysis)
    assert p.analysis.gross_premium == 10000.0
    assert p.analysis.terms.rates == (0.0, 0.5, 1.0, 1.0)


# ----------------------------------------------------------------------
# decision 3: a subsequent aggregate cover stays 2-D (builds; terms attached)
# ----------------------------------------------------------------------
def test_subsequent_aggregate_cover_builds():
    p = build(
        'pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% reinstatements [0 1] '
        'poisson aggregate net of 85% po 1500 xs 7000 deposit 600')
    # the reinstated occurrence layer + a genuine subsequent agg cover both build;
    # the attached analysis carries the agg tier threaded from the pnl economics.
    a = p.analysis
    assert a.terms is not None
    assert a.agg_recovery is not None
    assert a.agg_ceded_premium == pytest.approx(600.0)   # threaded from the pnl
    # the agg cover actually recovers on the net-of-occurrence loss L - A(R)
    assert a._stats_df.loc['ceded_agg_loss', 'mean'] > 0
    # decision 3: the ledger extends to the inuring both-tiers form -- a third
    # buy group over the SAME joint (no new dimension).
    s = p.summary_df
    for row in ('gross result', 'ceded occ result', 'net through ceded occ',
                'ceded agg premium', 'ceded agg recovery', 'ceded agg result',
                'net through ceded agg', 'margin'):
        assert row in s.index, row
    # means add tier by tier down the sheet
    assert s.loc['net through ceded occ', 'EX'] == pytest.approx(
        s.loc['gross result', 'EX'] + s.loc['ceded occ result', 'EX'],
        rel=1e-6, abs=1e-6)
    assert s.loc['margin', 'EX'] == pytest.approx(
        s.loc['net through ceded occ', 'EX'] + s.loc['ceded agg result', 'EX'],
        rel=1e-6, abs=1e-6)
    # the additive audit (incl. the agg-tier identities) still passes
    assert a.validation_df['abs_err'].max() < 1e-6
    # the final net (tail) is net of everything
    assert a._final_net_uw == 'net_agg_uw'


def test_expense_and_cede_book_as_ledger_legs():
    # a reinstatement pnl with gross expenses + a flat ceding commission books
    # them as cash-flow legs: the and-joined expense group on the gross group,
    # the commission as a received leg on the cession.
    p = build('pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 18% cede 20% reinstatements [0 1] '
              'poisson less 500 fixed expense and 10% premium expense')
    a = p.analysis
    assert a.gross_expense == pytest.approx(1500.0)        # 500 + 10% * 10000
    assert a.occ_commission == pytest.approx(3.6)          # 20% * (18% * 100)
    s = p.summary_df
    assert s.loc['expense', 'EX'] == pytest.approx(-1500.0)
    assert s.loc['ceded occ commission', 'EX'] == pytest.approx(3.6)
    # the gross group result books the whole expense vs the pure gross UW
    assert s.loc['gross result', 'EX'] == pytest.approx(
        a._stats_df.loc['gross_uw', 'mean'] - 1500.0, rel=1e-6, abs=1e-6)
    # and the cession result credits its commission vs the pure ceded UW
    assert s.loc['ceded occ result', 'EX'] == pytest.approx(
        a._stats_df.loc['ceded_uw', 'mean'] + 3.6, rel=1e-6, abs=1e-6)


def test_no_expense_leaves_ledger_pure():
    # without expenses the gross group result is the pure gross underwriting
    # mean (P_G - L).
    p = build('pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 18% reinstatements [0 1] poisson')
    a = p.analysis
    assert a.gross_expense == pytest.approx(0.0)
    assert a.occ_commission == pytest.approx(0.0)
    assert p.summary_df.loc['gross result', 'EX'] == pytest.approx(
        a._stats_df.loc['gross_uw', 'mean'], rel=1e-6, abs=1e-6)
