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
from aggregate import build
from aggregate.reinstatement import ReinstatementAnalysis, ReinstatementTerms

_UW = aggregate.Underwriter()


def _spec(program):
    """Parse one DecL program to its ``(kind, name, spec)`` triple."""
    return _UW.parser.parse(program)


_HUMAN = ('pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
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
        'pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% reinstatements [0 .5 1 1] poisson')
    assert spec['occ_reins_reinst'] == [(0.0, 0.5, 1.0, 1.0)]


def test_number_word_counts():
    _, _, spec = _spec(
        'pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% '
        'reinstatements one free and three at 100% poisson')
    assert spec['occ_reins_reinst'] == [(0.0, 1.0, 1.0, 1.0)]


def test_single_group_no_free():
    _, _, spec = _spec(
        'pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% reinstatements 3 at 100% poisson')
    assert spec['occ_reins_reinst'] == [(1.0, 1.0, 1.0)]


def test_deposit_base_premium_form():
    _, _, spec = _spec(
        'pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
        'occurrence ceded to 100 xs 100 deposit 1800 reinstatements [0 1 1] poisson')
    assert spec['occ_reins_reinst'] == [(0.0, 1.0, 1.0)]
    assert spec['occ_reins_premium'] == [('deposit', 1800.0)]


def test_no_reinstatements_marker():
    # ``no reinstatements`` -> the empty-tuple marker (m=0), distinct from the
    # omitted clause (None = free + unlimited).
    _, _, spec = _spec(
        'pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% no reinstatements poisson')
    assert spec['occ_reins_reinst'] == [()]


def test_no_reinstatements_is_a_single_annual_limit():
    # zero reinstatements: recovery capped at the single occurrence limit y, no
    # reinstatement premium, so the ceded premium is the deterministic deposit.
    p = build(
        'pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% no reinstatements poisson')
    t = p.agg.reinstatement_terms
    assert t.n_reinstatements == 0
    assert t.total_recovery_capacity == pytest.approx(100.0)   # single limit y
    assert t.recovery(250.0) == pytest.approx(100.0)
    assert t.reinstatement_premium(250.0) == pytest.approx(0.0)
    # deterministic ceded premium (= the deposit; h(R) == 0) => the exact ceded
    # premium has zero variance, vs a materially nonzero CV when reinstatements
    # are paid. The exhibit CV carries only the heavy-tail grid-deficit floor.
    an = p.reinstatement_analysis
    assert an.stats_df.loc['reinstatement_premium', 'mean'] == pytest.approx(0.0)
    assert an.stats_df.loc['ceded_premium', 'sd'] == pytest.approx(0.0, abs=1e-6)
    assert p.gcn_df.loc[('Volatility', 'CV Premium'), 'ceded'] == pytest.approx(
        0.0, abs=1e-6)


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
        'pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% poisson')
    assert 'occ_reins_reinst' not in spec
    p = build(
        'pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% poisson')
    assert p.reinstatement_analysis is None
    assert getattr(p.agg, 'reinstatement_terms', None) is None


# ----------------------------------------------------------------------
# validation rules (locked, dev/plan-reinstatements.md)
# ----------------------------------------------------------------------
def test_reins_premium_rule():
    # [reins-premium]: a reinstatements clause needs a base premium clause.
    with pytest.raises(ValueError, match=r'reins-premium'):
        _spec('pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 reinstatements [0 1] poisson')


def test_reins_single_layer_rule():
    # [reins-single-layer]: a reinstated layer + a plain second occ layer.
    with pytest.raises(ValueError, match=r'reins-single-layer'):
        _spec('pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 10% reinstatements [0 1] '
              'and 200 xs 200 poisson')


def test_reins_one_clause_rule():
    # [reins-one-clause]: two layers each carrying a reinstatements clause.
    with pytest.raises(ValueError, match=r'reins-one-clause'):
        _spec('pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 10% reinstatements [0 1] '
              'and 200 xs 200 rol 5% reinstatements [0 1] poisson')


def test_reinstatements_rejected_on_aggregate():
    with pytest.raises(ValueError, match=r'occurrence layer'):
        _spec('pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 poisson '
              'aggregate net of 2000 xs 3000 rol 8% reinstatements [0 1]')


def test_negative_rate_rejected():
    with pytest.raises(ValueError, match=r'nonnegative'):
        _spec('pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 18% reinstatements [0 -1] poisson')


# ----------------------------------------------------------------------
# build() return: a PnL backed by a ReinstatementAnalysis
# ----------------------------------------------------------------------
def test_build_returns_analysis_backed_pnl():
    p = build(_HUMAN)
    an = p.reinstatement_analysis
    assert isinstance(an, ReinstatementAnalysis)
    # the cached engine is reused
    assert p.reinstatement_analysis is an
    assert an.gross_premium == 10000.0


def test_terms_carry_share_scaled_limit_and_base_premium():
    # 95% po 100 xs 100 rol 18%: effective limit y = share*limit = 95; base
    # premium D = share*rol*limit = 17.1; base rate r = D/y = 0.18 = the rol.
    p = build(_HUMAN)
    t = p.agg.reinstatement_terms
    assert isinstance(t, ReinstatementTerms)
    assert t.limit == pytest.approx(95.0)
    assert t.deposit == pytest.approx(17.1)
    assert t.rol == pytest.approx(0.18)
    assert t.rates == (0.0, 0.5, 1.0, 1.0)


def test_gcn_df_delegates_and_means_add():
    p = build(_HUMAN)
    g = p.gcn_df
    assert list(g.columns) == ['gross', 'ceded', 'net', 'impact']
    for row in ('Premium', 'Loss', 'UW'):
        assert g.loc[('Mean', row), 'net'] == pytest.approx(
            g.loc[('Mean', row), 'gross'] + g.loc[('Mean', row), 'ceded'],
            rel=1e-6, abs=1e-6)


def test_ceded_premium_is_stochastic_in_exhibit():
    p = build(_HUMAN)
    g = p.gcn_df
    # the headline effect: stochastic ceded premium => nonzero CV(premium) on
    # the cession / net columns, zero on gross.
    assert g.loc[('Volatility', 'CV Premium'), 'gross'] == 0.0
    assert g.loc[('Volatility', 'CV Premium'), 'ceded'] > 0.0


def test_summary_df_routes_through_gcn():
    p = build(_HUMAN)
    # a GCN PnL's summary_df is the waterfall (consistent with the deterministic
    # ceded-premium path); the richer headline table lives on the analysis.
    assert list(p.summary_df.columns) == ['gross', 'ceded', 'net', 'impact']
    assert list(p.reinstatement_analysis.summary_df.columns) == [
        'Gross', 'Ceded', 'Net', 'Impact', 'Pct Impact']


def test_arg_free_programmatic_entry_point():
    # build attaches terms + gross premium, so the programmatic API works arg-free.
    p = build(_HUMAN)
    an = p.agg.reinstatement_analysis()
    assert an.gross_premium == 10000.0
    assert an.terms.rates == (0.0, 0.5, 1.0, 1.0)


# ----------------------------------------------------------------------
# decision 3: a subsequent aggregate cover stays 2-D (builds; terms attached)
# ----------------------------------------------------------------------
def test_subsequent_aggregate_cover_builds():
    p = build(
        'pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% reinstatements [0 1] '
        'poisson aggregate net of 85% po 1500 xs 7000 deposit 600')
    # the reinstated occurrence layer + a genuine subsequent agg cover both build
    assert p.agg.reinstatement_terms is not None
    assert p.agg.agg_reins is not None
    an = p.reinstatement_analysis
    assert an.agg_recovery is not None
    assert an.agg_ceded_premium == pytest.approx(600.0)   # threaded from the pnl
    # decision 3: the waterfall extends to the five-column inuring form, and the
    # ceded_agg column populates with the agg recovery g(L - A(R)).
    g = p.gcn_df
    assert list(g.columns) == ['gross', 'ceded occ', 'net occ', 'ceded agg',
                               'net agg', 'occ impact', 'agg impact', 'impact']
    assert g.loc[('Mean', 'Loss'), 'ceded agg'] > 0       # real agg recovery
    m = g.xs('Mean')
    # means add tier by tier
    for row in ('Premium', 'Loss', 'UW'):
        assert m.loc[row, 'net occ'] == pytest.approx(
            m.loc[row, 'gross'] + m.loc[row, 'ceded occ'], rel=1e-6, abs=1e-6)
        assert m.loc[row, 'net agg'] == pytest.approx(
            m.loc[row, 'net occ'] + m.loc[row, 'ceded agg'], rel=1e-6, abs=1e-6)
    # the additive audit (incl. the agg-tier identities) still passes
    assert an.validation_df['abs_err'].max() < 1e-6
    # the final net (tail / summary) is net of everything
    assert an._final_net_uw == 'net_agg_uw'


def test_deterministic_expense_and_cede_in_waterfall():
    # a reinstatement pnl with gross expenses + a flat ceding commission shows
    # them in the GCN Expense section, net of the (deterministic) commission,
    # exactly as a plain pnl does. Phase-3 slide/pc make this leg stochastic.
    p = build('pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 18% cede 20% reinstatements [0 1] '
              'poisson 500 fixed expense and 10% premium expense')
    an = p.reinstatement_analysis
    assert an.gross_expense == pytest.approx(1500.0)        # 500 + 10% * 10000
    assert an.occ_commission == pytest.approx(3.6)          # 20% * (18% * 100)
    g = p.gcn_df
    e = g.xs('Mean').loc['Expense']
    # gross books the whole expense (cost, -); the cession credits its commission
    assert e['gross'] == pytest.approx(-1500.0)
    assert e['ceded'] == pytest.approx(3.6)
    assert e['net'] == pytest.approx(-1496.4)               # adds across
    m = g.xs('Mean')
    # Premium + Loss + Expense = UW down each column
    for c in ('gross', 'ceded', 'net'):
        assert m.loc['Premium', c] + m.loc['Loss', c] + m.loc['Expense', c] == \
            pytest.approx(m.loc['UW', c], rel=1e-6, abs=1e-6)
    # the summary UW is net of expense and ties to the gcn_df Mean UW
    s = an.summary_df
    assert s.loc[('', 'Underwriting'), 'Net'] == pytest.approx(
        m.loc['UW', 'net'], rel=1e-6, abs=1e-3)


def test_no_expense_leaves_waterfall_unshifted():
    # without expenses the Expense row is zero (shift is a no-op)
    p = build('pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 18% reinstatements [0 1] poisson')
    e = p.gcn_df.xs('Mean').loc['Expense']
    assert e['gross'] == pytest.approx(0.0)
    assert e['net'] == pytest.approx(0.0)


def test_aggregate_cover_summary_total_cession():
    # the headline summary collapses to Gross / total-Ceded / final-Net
    p = build(
        'pnl Cat 10000 premium less 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 100 xs 100 rol 18% reinstatements [0 1] '
        'poisson aggregate net of 85% po 1500 xs 7000 deposit 600')
    s = p.reinstatement_analysis.summary_df
    assert list(s.columns) == ['Gross', 'Ceded', 'Net', 'Impact', 'Pct Impact']
    # premium / loss are magnitudes: the cession flows out, so Gross - Ceded = Net
    for item in ('Premium', 'Loss'):
        assert s.loc[('', item), 'Gross'] - s.loc[('', item), 'Ceded'] == \
            pytest.approx(s.loc[('', item), 'Net'], rel=1e-6, abs=1e-6)
    # the underwriting result is signed: the cession gain adds, Gross + Ceded = Net
    assert s.loc[('', 'Underwriting'), 'Gross'] + \
        s.loc[('', 'Underwriting'), 'Ceded'] == pytest.approx(
            s.loc[('', 'Underwriting'), 'Net'], rel=1e-6, abs=1e-6)
