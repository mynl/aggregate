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
    assert a.stats_df.loc['reinstatement_premium', 'mean'] == pytest.approx(0.0)
    assert a.stats_df.loc['ceded_premium', 'sd'] == pytest.approx(0.0, abs=1e-6)
    assert a.stats_df.loc['ceded_premium', 'cv'] == pytest.approx(0.0, abs=1e-6)


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


def test_gcn_df_delegates_and_means_add():
    p = build(_HUMAN)
    g = p.margin_df
    # the new stats x waterfall shape: EX/SD/CV/Skew/percentile rows, gross /
    # ceded / net underwriting columns plus a benefit (impact) column.
    cols = list(g.columns)
    assert cols[:3] == ['gross', 'ceded', 'net']
    assert 'EX' in g.index
    # means add on the EX row: gross + ceded = net
    assert g.loc['EX', 'net'] == pytest.approx(
        g.loc['EX', 'gross'] + g.loc['EX', 'ceded'], rel=1e-6, abs=1e-6)
    # the trailing benefit column is the cession impact = net - gross
    assert g.loc['EX', cols[-1]] == pytest.approx(
        g.loc['EX', 'net'] - g.loc['EX', 'gross'], rel=1e-6, abs=1e-6)


def test_ceded_premium_is_stochastic_in_exhibit():
    p = build(_HUMAN)
    g = p.margin_df
    # the headline effect: the stochastic ceded premium D + h(R) => a nonzero CV
    # on the cession column of the waterfall...
    assert g.loc['CV', 'ceded'] > 0.0
    # ...and per leg, the ceded premium is stochastic while the gross premium is
    # the fixed P_G (zero CV).
    assert p.analysis.stats_df.loc['ceded_premium', 'cv'] > 0.0
    assert p.analysis.stats_df.loc['gross_premium', 'cv'] == 0.0


def test_summary_df_is_the_gcn_table():
    p = build(_HUMAN)
    # the attached analysis carries the canonical Gross / Ceded / Net headline
    # table; the PnL's own summary_df is the fixed template (tested elsewhere).
    assert list(p.analysis.summary_df.columns) == [
        'Gross', 'Ceded', 'Net', 'Impact', 'Pct Impact']
    assert list(p.margin_df.columns)[:3] == ['gross', 'ceded', 'net']


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
    assert a.stats_df.loc['ceded_agg_loss', 'mean'] > 0
    # decision 3: the waterfall extends to the inuring both-tiers form. Read the
    # actual columns (a running-net tower): gross, occ cession, net-of-occ,
    # agg cession, final net, and the benefit columns.
    g = p.margin_df
    cols = list(g.columns)
    assert cols == ['gross', 'ceded', 'net/ceded', 'occ benefit', 'ceded agg',
                    'net', 'agg benefit', 'total benefit']
    # means add tier by tier on the EX row: gross + occ cession = net-of-occ;
    # net-of-occ + agg cession = final net.
    assert g.loc['EX', 'net/ceded'] == pytest.approx(
        g.loc['EX', 'gross'] + g.loc['EX', 'ceded'], rel=1e-6, abs=1e-6)
    assert g.loc['EX', 'net'] == pytest.approx(
        g.loc['EX', 'net/ceded'] + g.loc['EX', 'ceded agg'], rel=1e-6, abs=1e-6)
    # the additive audit (incl. the agg-tier identities) still passes
    assert a.validation_df['abs_err'].max() < 1e-6
    # the final net (tail / summary) is net of everything
    assert a._final_net_uw == 'net_agg_uw'


def test_deterministic_expense_and_cede_in_waterfall():
    # a reinstatement pnl with gross expenses + a flat ceding commission shows
    # them in the GCN Expense section, net of the (deterministic) commission,
    # exactly as a plain pnl does. Phase-3 slide/pc make this leg stochastic.
    p = build('pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 18% cede 20% reinstatements [0 1] '
              'poisson less 500 fixed expense and 10% premium expense')
    a = p.analysis
    assert a.gross_expense == pytest.approx(1500.0)        # 500 + 10% * 10000
    assert a.occ_commission == pytest.approx(3.6)          # 20% * (18% * 100)
    g = p.margin_df
    # the deterministic expense / commission is baked into the waterfall UW: the
    # gross column books the whole expense (a -1500 cost vs the pure gross UW),
    # while the cession credits its commission (+3.6 vs the pure ceded UW).
    assert g.loc['EX', 'gross'] == pytest.approx(
        a.stats_df.loc['gross_uw', 'mean'] - 1500.0, rel=1e-6, abs=1e-6)
    assert g.loc['EX', 'ceded'] == pytest.approx(
        a.stats_df.loc['ceded_uw', 'mean'] + 3.6, rel=1e-6, abs=1e-6)
    # means still add across the split on the EX row
    assert g.loc['EX', 'net'] == pytest.approx(
        g.loc['EX', 'gross'] + g.loc['EX', 'ceded'], rel=1e-6, abs=1e-6)
    # the summary UW is net of expense and ties to the gcn_df EX net UW
    s = a.summary_df
    assert s.loc[('', 'Underwriting'), 'Net'] == pytest.approx(
        g.loc['EX', 'net'], rel=1e-6, abs=1e-3)


def test_no_expense_leaves_waterfall_unshifted():
    # without expenses there is no expense / commission shift: the gross column
    # of the waterfall equals the pure gross underwriting mean (P_G - L).
    p = build('pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 18% reinstatements [0 1] poisson')
    a = p.analysis
    assert a.gross_expense == pytest.approx(0.0)
    assert a.occ_commission == pytest.approx(0.0)
    assert p.margin_df.loc['EX', 'gross'] == pytest.approx(
        a.stats_df.loc['gross_uw', 'mean'], rel=1e-6, abs=1e-6)


def test_aggregate_cover_summary_total_cession():
    # the headline summary collapses to Gross / total-Ceded / final-Net
    p = build(
        'pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% reinstatements [0 1] '
        'poisson aggregate net of 85% po 1500 xs 7000 deposit 600')
    s = p.analysis.summary_df
    assert list(s.columns) == ['Gross', 'Ceded', 'Net', 'Impact', 'Pct Impact']
    # premium / loss are magnitudes: the cession flows out, so Gross - Ceded = Net
    for item in ('Premium', 'Loss'):
        assert s.loc[('', item), 'Gross'] - s.loc[('', item), 'Ceded'] == \
            pytest.approx(s.loc[('', item), 'Net'], rel=1e-6, abs=1e-6)
    # the underwriting result is signed: the cession gain adds, Gross + Ceded = Net
    assert s.loc[('', 'Underwriting'), 'Gross'] + \
        s.loc[('', 'Underwriting'), 'Ceded'] == pytest.approx(
            s.loc[('', 'Underwriting'), 'Net'], rel=1e-6, abs=1e-6)
