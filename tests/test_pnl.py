"""Tests for the ``pnl`` keyword -- premium-minus-loss aggregates.

``pnl NAME <premium> prem - <loss body>`` is a sibling of ``agg`` that builds an
ordinary loss aggregate and applies an aggregate-level affine wrapper
``PnL = premium - A`` (reflect + a single shift by the total premium). The
premium is subtracted **once for the book**, in contrast to a constant inside
``sev``/``dsev``/``ssev`` which is per-claim. See dev/done/plan-pnl-premium.md.

Covers: the three exposure forms (lr / claims / loss), vectorised premium,
moment closed forms (mean shift, sd invariant, skew sign flip), P(loss), the
per-claim-vs-once distinction, the signed-aware SD describe, the payoff
value_type, and the Portfolio combine of a book of pnl units.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from aggregate import build
from aggregate.constants import DefectiveDistributionWarning

# severities chosen light enough that the 12-nines window is well-resolved, so
# the empirical moments match the analytic ones tightly.
TOL = 5e-3


# ----------------------------------------------------------------------
# Parse / spec
# ----------------------------------------------------------------------
def test_value_type_payoff():
    """A pnl aggregate is tagged payoff and is signed via the affine."""
    a = build('pnl B 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    assert a.value_type == 'payoff'
    assert a._agg_affine_active()
    assert a._agg_reflect is True
    assert a._agg_shift == 1000.0
    assert a._signed()
    # the loss severity itself is NOT signed (non-negative loss)
    assert not a._signed_severity()


# ----------------------------------------------------------------------
# Moment closed forms across the three exposure forms
# ----------------------------------------------------------------------
@pytest.mark.parametrize('program,premium,e_loss', [
    ('pnl X 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson', 1000.0, 700.0),
    ('pnl X 100 prem - 7 claims sev gamma 100 cv 0.5 poisson', 100.0, 700.0),
    ('pnl X 100 prem - 700 loss sev gamma 100 cv 0.5 poisson', 100.0, 700.0),
])
def test_mean_across_exposure_forms(program, premium, e_loss):
    """mean = premium - E[loss] for lr / claims / loss exposure heads."""
    a = build(program)
    assert a.est_m == pytest.approx(premium - e_loss, rel=TOL, abs=TOL)


def test_sd_invariant_skew_flips():
    """sd unchanged, skew sign-flipped relative to the bare loss aggregate."""
    pnl = build('pnl X 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    loss = build('agg L 1000 prem at 0.7 lr sev gamma 100 cv 0.5 poisson')
    # mean = premium - E[loss]
    assert pnl.est_m == pytest.approx(1000 - loss.est_m, rel=TOL, abs=TOL)
    # spread invariant under reflect + shift
    assert pnl.est_sd == pytest.approx(loss.est_sd, rel=TOL)
    # reflection flips the skew sign
    assert pnl.est_skew == pytest.approx(-loss.est_skew, rel=2e-2, abs=1e-3)


# ----------------------------------------------------------------------
# Vectorised premium (mirrors agg)
# ----------------------------------------------------------------------
def test_vector_premium():
    """Shift = sum(premium); mean = sum(premium_i * (1 - lr))."""
    a = build('pnl X [100 200 100] prem - .8 lr [1000 2000 5000] xs 0 '
              'sev lognorm 500 cv 2 poisson')
    assert a._agg_shift == pytest.approx(400.0)
    # mean = 400 - 0.8 * 400 = 80
    assert a.est_m == pytest.approx(80.0, rel=TOL, abs=0.5)


# ----------------------------------------------------------------------
# P(loss) = P(PnL < 0) = loss survival at the premium
# ----------------------------------------------------------------------
def test_p_loss_matches_loss_survival():
    pnl = build('pnl X 100 prem - 7 claims sev gamma 100 cv 0.5 poisson')
    loss = build('agg L 7 claims sev gamma 100 cv 0.5 poisson')
    p_loss = float(pnl.agg_density[pnl.xs < 0].sum())
    assert p_loss == pytest.approx(float(loss.sf(100)), rel=2e-2, abs=2e-3)


# ----------------------------------------------------------------------
# Per-claim (ssev) vs once-for-the-book (pnl) distinction
# ----------------------------------------------------------------------
def test_per_claim_vs_once_distinction():
    """``pnl 100 prem - 5 claims`` (premium once) differs from
    ``5 claims ssev 100 - sev`` (constant per claim)."""
    once = build('pnl P 100 prem - 5 claims sev gamma 8 cv 0.5 poisson')
    # per-claim: mean ~ 5 * (100 - 8) = 460, far from the pnl mean 100 - 40 = 60
    per_claim = build('agg S 5 claims ssev 100 - gamma 8 cv 0.5 poisson')
    assert once.est_m == pytest.approx(100 - 5 * 8, rel=TOL, abs=0.5)
    assert per_claim.est_m == pytest.approx(5 * (100 - 8), rel=2e-2, abs=1.0)
    assert abs(once.est_m - per_claim.est_m) > 100


# ----------------------------------------------------------------------
# Mass conservation + signed window brackets the mean
# ----------------------------------------------------------------------
def test_mass_conserved_and_window_brackets_mean():
    a = build('pnl X 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    assert a.agg_density.sum() == pytest.approx(1.0, abs=1e-6)
    assert a.xs[0] < a.est_m < a.xs[-1]
    # the grid straddles 0 (a P&L can be a loss)
    assert a.xs[0] < 0 < a.xs[-1]


# ----------------------------------------------------------------------
# Signed-aware describe: SD trio (not CV), finite even for a mean-zero P&L
# ----------------------------------------------------------------------
def test_describe_signed_sd_columns():
    a = build('pnl X 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    cols = list(a.describe.columns)
    assert 'SD' in cols and 'Est SD' in cols and 'Err SD' in cols
    assert 'CV' not in cols and 'Est CV' not in cols


def test_describe_finite_for_mean_zero_pnl():
    """When premium == E[loss] the mean is ~0; SD describe stays finite."""
    # premium == E[loss] (lr 100%) -> margin ~ 0
    a = build('pnl X 700 prem - 100% lr sev gamma 100 cv 0.5 poisson')
    assert abs(a.est_m) < 5.0          # mean near zero
    df = a.describe
    assert np.isfinite(df.to_numpy().astype(float)).all() or \
        np.isfinite(df['SD'].to_numpy().astype(float)).all()
    # Agg SD is finite and positive regardless of the near-zero mean
    assert df.loc['Agg', 'SD'] > 0


def test_non_pnl_describe_unchanged():
    """An ordinary aggregate still shows the CV trio (byte-for-byte path)."""
    a = build('agg N 100 claims sev gamma 100 cv 0.5 poisson')
    cols = list(a.describe.columns)
    assert 'CV' in cols and 'Est CV' in cols and 'SD' not in cols


# ----------------------------------------------------------------------
# info readout
# ----------------------------------------------------------------------
def test_info_pnl_readout():
    a = build('pnl X 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    info = a.info
    assert 'premium' in info and 'expected loss' in info and 'P(loss)' in info
    assert 'value_type               payoff' in info


# ----------------------------------------------------------------------
# Portfolio: a book of pnl units combines (means add, variances add)
# ----------------------------------------------------------------------
def test_portfolio_of_pnl_combines():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        port = build('''port Book
            pnl A 1000 prem - 80% lr sev gamma 100 cv 0.3 poisson
            pnl B 1000 prem - 80% lr sev gamma 100 cv 0.3 poisson
        ''')
        unit = build('pnl A 1000 prem - 80% lr sev gamma 100 cv 0.3 poisson')
    d = port.density_df
    assert d.p_total.sum() == pytest.approx(1.0, abs=1e-5)
    mean = float((d.loss * d.p_total).sum())
    assert mean == pytest.approx(2 * 200.0, rel=TOL, abs=2.0)
    # per-line means from the native unit pmfs (numerics-2)
    for u in ('A', 'B'):
        ser = port.unit_density(u)
        mu = float((ser.index * ser).sum())
        assert mu == pytest.approx(200.0, rel=TOL, abs=1.0)
    # variances add under independence: sd_total = sqrt(2) * unit sd
    var = float((d.loss ** 2 * d.p_total).sum()) - mean ** 2
    assert var ** 0.5 == pytest.approx(np.sqrt(2) * unit.est_sd, rel=1e-2)


def test_portfolio_pnl_mixed_with_loss_line():
    """A pnl margin line composes with a pnl *cost* line (0 premium).

    A bare ``agg`` loss line would enter the convolution with a positive mean
    (it adds); to carry a pure cost in a P&L book, declare it as a 0-premium
    ``pnl`` so it enters as ``-loss``. Margin 200 + cost (-200) -> ~0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        port = build('''port Mixed
            pnl P 1000 prem - 80% lr sev gamma 100 cv 0.3 poisson
            pnl C 0 prem - 200 loss sev gamma 100 cv 0.3 poisson
        ''')
    d = port.density_df
    assert d.p_total.sum() == pytest.approx(1.0, abs=1e-5)
    mean = float((d.loss * d.p_total).sum())
    # margin 200 plus cost line mean (0 - 200) = -200 -> ~0
    assert mean == pytest.approx(0.0, abs=2.0)


# ----------------------------------------------------------------------
# Signed loss severity (dsev with a negative atom / ssev) under pnl
# ----------------------------------------------------------------------
# A ``pnl`` whose *loss severity* is itself signed convolves the loss on its
# genuine signed grid before the affine relabel onto the P&L window. Previously
# the affine path hard-coded a 0-based loss grid, wrapping the negative atoms to
# the top of the FFT buffer: half the mass dropped and the empirical moments
# read +/-2**15 grid-index garbage. See dev/done/plan-pnl-signed-severity.md.
def test_signed_dsev_pnl_exact():
    """``pnl 5 prem - dfreq[3] dsev[-1 1]`` -> P&L in {2,4,6,8}, mean 5, sd sqrt3."""
    with warnings.catch_warnings():
        # a correctly-sized signed-loss pnl must not warn (no dropped mass)
        warnings.simplefilter('error', category=DefectiveDistributionWarning)
        a = build('pnl GP 5 premium - dfreq[3] dsev[-1 1]', bs=1)
    m = a.agg_density > 1e-12
    support = a.xs[m]
    probs = a.agg_density[m]
    assert a.agg_density.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(support, [2.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(probs, [0.125, 0.375, 0.375, 0.125], atol=1e-12)
    assert a.est_m == pytest.approx(5.0, abs=1e-9)
    assert a.est_sd == pytest.approx(np.sqrt(3.0), abs=1e-9)
    assert a.est_skew == pytest.approx(0.0, abs=1e-9)


def test_signed_dsev_pnl_asymmetric_mean():
    """Mean closed form holds for a non-symmetric signed dsev: E[PnL]=shift-2 E[X]."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('pnl Y 10 premium - dfreq[2] dsev[-2 1 3] [.5 .3 .2]', bs=1)
    # per-claim E[X] = -2(.5) + 1(.3) + 3(.2) = -0.1; two claims -> E[L] = -0.2
    assert a.agg_density.sum() == pytest.approx(1.0, abs=1e-12)
    assert a.est_m == pytest.approx(10.0 - 2 * (-0.1), abs=1e-9)


def test_signed_ssev_pnl_mass_and_mean():
    """Continuous signed severity (``ssev``) under pnl conserves mass; mean shifts."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('pnl Z 100 premium - 5 claims ssev 20 - lognorm 10 cv 0.5 poisson')
    assert a._signed_severity()
    assert a.agg_density.sum() == pytest.approx(1.0, abs=1e-5)
    # loss sev = 20 - lognorm(mean 10) -> per-claim mean 10; 5 claims -> E[L]=50
    assert a.est_m == pytest.approx(100 - 50, abs=0.5)


def test_signed_pnl_unit_in_portfolio_conserves_mass():
    """A signed-loss pnl placed in a book keeps its own unit density mass-conserving."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        port = build('''port SignedBook
            pnl S 5 premium - dfreq[3] dsev[-1 1]
            pnl T 1000 prem - 80% lr sev gamma 100 cv 0.3 poisson
        ''', bs=1)
    # the signed-loss unit's own marginal is intact (no +/-2**15 garbage,
    # full mass); read off the unit's native window (numerics-2)
    assert port.unit_density('S').sum() == pytest.approx(1.0, abs=1e-6)
