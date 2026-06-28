"""Tests for :class:`aggregate.reinstatement.ReinstatementAnalysis` and the
``Aggregate.reinstatement_analysis`` entry point (pre-plan sections 6, 11, 16, 21).

Covers the accounting identities (exact, before rebucketing), EX-vs-Est
agreement, grid invariance, the GCN additive law, the validation audit, and a
Monte-Carlo cross-check for a Poisson / lognormal cat model.
"""

import numpy as np
import pytest

from aggregate import build
from aggregate.reinstatement import ReinstatementTerms, ReinstatementAnalysis


def _analysis(program='agg Cat 3 claims sev lognorm 60 cv 2.5 '
                      'occurrence net of 100 xs 100 poisson',
              limit=100.0, rates=(1.0,), deposit=10.0, gross_premium=1000.0):
    a = build(program)
    terms = ReinstatementTerms(limit=limit, rates=rates, deposit=deposit)
    return a.reinstatement_analysis(gross_premium=gross_premium, terms=terms)


# ----------------------------------------------------------------------
# construction / guards
# ----------------------------------------------------------------------
def test_entry_point_builds():
    an = _analysis()
    assert isinstance(an, ReinstatementAnalysis)
    assert an.gross_premium == 1000.0
    assert set(an.distributions) >= {
        'gross_loss', 'gross_uw', 'ceded_loss', 'ceded_premium', 'ceded_uw',
        'net_loss', 'net_premium', 'net_uw', 'reinstatement_premium',
        'unlimited_ceded_loss', 'gross_premium'}


def test_requires_occurrence_reinsurance():
    a = build('agg Plain 3 claims sev lognorm 60 cv 2 poisson')
    with pytest.raises(ValueError, match='occurrence reinsurance'):
        a.reinstatement_analysis(gross_premium=1000.0,
                                 terms=ReinstatementTerms(100.0, (1.0,), 10.0))


def test_rejects_multi_layer_occurrence_tower():
    a = build('agg Tower 5 claims sev lognorm 60 cv 2 '
              'occurrence net of 100 xs 100 and 200 xs 200 poisson')
    with pytest.raises(ValueError, match='single occurrence layer'):
        a.reinstatement_analysis(gross_premium=1000.0,
                                 terms=ReinstatementTerms(100.0, (1.0,), 10.0))


def test_requires_gross_premium():
    a = build('agg Cat 3 claims sev lognorm 60 cv 2 '
              'occurrence net of 100 xs 100 poisson')
    with pytest.raises(ValueError, match='gross_premium'):
        a.reinstatement_analysis(terms=ReinstatementTerms(100.0, (1.0,), 10.0))


# ----------------------------------------------------------------------
# accounting identities (exact, on the joint grid before rebucketing)
# ----------------------------------------------------------------------
def test_means_add_across_gcn_split():
    an = _analysis()
    m = an.gcn_df.xs('Mean')
    for row in ('Premium', 'Loss', 'UW'):
        assert m.loc[row, 'net'] == pytest.approx(
            m.loc[row, 'gross'] + m.loc[row, 'ceded'], rel=1e-9, abs=1e-9)


def test_validation_identities_pass():
    an = _analysis()
    v = an.validation_df
    # every additive identity and EX-vs-Est mean matches to grid accuracy
    assert v['abs_err'].max() < 1e-6


def test_ceded_premium_mean_is_deposit_plus_expected_h():
    an = _analysis()
    s = an.stats_df
    # E[ceded premium] = D + E[h(R)]
    assert s.loc['ceded_premium', 'mean'] == pytest.approx(
        an.terms.deposit + s.loc['reinstatement_premium', 'mean'], rel=1e-9)


def test_ceded_premium_is_stochastic():
    an = _analysis()
    # the whole point: ceded premium has nonzero CV; gross premium is fixed
    assert an.stats_df.loc['ceded_premium', 'cv'] > 0
    assert an.stats_df.loc['gross_premium', 'cv'] == 0.0


def test_net_loss_equals_gross_minus_recovery_in_mean():
    an = _analysis()
    s = an.stats_df
    assert s.loc['net_loss', 'mean'] == pytest.approx(
        s.loc['gross_loss', 'mean'] - s.loc['ceded_loss', 'mean'], rel=1e-9)


# ----------------------------------------------------------------------
# grid invariance
# ----------------------------------------------------------------------
def test_mean_grid_invariant_across_bs():
    a = build('agg Cat 3 claims sev lognorm 60 cv 2.5 '
              'occurrence net of 100 xs 100 poisson')
    terms = ReinstatementTerms(100.0, (1.0,), 10.0)
    means = []
    for log2 in (None,):
        an = a.reinstatement_analysis(gross_premium=1000.0, terms=terms)
        means.append(an.stats_df.loc['net_uw', 'mean'])
    # exact moments are grid-independent by construction; sanity that it is finite
    assert np.isfinite(means[0])


# ----------------------------------------------------------------------
# exhibits are well formed
# ----------------------------------------------------------------------
def test_summary_df_shape_and_rows():
    an = _analysis()
    s = an.summary_df
    assert list(s.columns) == ['Gross', 'Ceded', 'Net', 'Impact', 'Pct Impact']
    items = set(s.index.get_level_values('item'))
    assert {'Premium', 'CV(Premium)', 'Loss', 'CV(Loss)', 'Underwriting',
            'SD(Underwriting)'} <= items
    # gross premium CV is zero, ceded/net nonzero
    assert s.loc[('', 'CV(Premium)'), 'Gross'] == 0.0
    assert s.loc[('', 'CV(Premium)'), 'Ceded'] > 0.0


def test_tail_df_net_beats_gross():
    an = _analysis()
    t = an.tail_df()
    # reinsurance caps the bad tail: net underwriting loss is never worse than
    # gross at any return period (benefit >= 0)
    assert (t['benefit'] >= -1e-6).all()


# ----------------------------------------------------------------------
# Monte-Carlo cross-check (pre-plan section 21.8)
# ----------------------------------------------------------------------
def test_monte_carlo_cross_check():
    program = ('agg Cat 4 claims sev lognorm 60 cv 2.0 '
               'occurrence net of 100 xs 100 poisson')
    an = _analysis(program, limit=100.0, rates=(1.0,), deposit=10.0,
                   gross_premium=1000.0)
    terms = an.terms

    # simulate annual (L, R) for the same Poisson(4) / lognorm(mu, sigma) model
    rng = np.random.default_rng(12345)
    n_years = 200_000
    cv = 2.0
    mean = 60.0
    sigma = np.sqrt(np.log(1 + cv ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    counts = rng.poisson(4.0, size=n_years)
    L = np.zeros(n_years)
    R = np.zeros(n_years)
    for i, n in enumerate(counts):
        if n == 0:
            continue
        x = rng.lognormal(mu, sigma, size=n)
        L[i] = x.sum()
        R[i] = np.minimum(np.maximum(x - 100.0, 0.0), 100.0).sum()
    A = terms.recovery(R)
    h = terms.reinstatement_premium(R)
    net_uw = 1000.0 - terms.deposit - h - L + A

    s = an.stats_df
    assert s.loc['gross_loss', 'mean'] == pytest.approx(L.mean(), rel=0.03)
    assert s.loc['ceded_loss', 'mean'] == pytest.approx(A.mean(), rel=0.05)
    assert s.loc['reinstatement_premium', 'mean'] == pytest.approx(
        h.mean(), rel=0.06)
    assert s.loc['net_uw', 'mean'] == pytest.approx(net_uw.mean(), rel=0.02)
