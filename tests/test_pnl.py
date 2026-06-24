"""Tests for the ``pnl`` keyword and the first-class :class:`PnL` veneer.

``pnl NAME <consideration> prem - <loss body>`` builds a pure-loss
:class:`Aggregate` X (the obligation) and wraps it in a :class:`PnL` whose net
is ``consideration - X`` (a profit is a negative loss). The consideration is a
single amount for the book, in contrast to a constant inside ``sev``/``dsev``/
``ssev`` which is per-claim. ``build('pnl ...')`` and
``build('agg ...').make_pnl(consideration=...)`` coincide. See dev/plan-pnl.md.

The risky leg is left untouched: ``pnl.agg`` is the honest obligation (density,
moments, plot in loss terms); the net is the derived ``pnl.pnl_df``. A ``pnl``
is **always payoff** (more net money is better). Portfolios / bivariates of
``pnl`` units are deferred (book-level P&L), and rejected with a clear error.

Covers: the PnL return type, value_type, the three exposure forms (lr / claims /
loss), the moment closed forms (mean shift, sd invariant, skew sign flip),
vector consideration, P(loss), the per-claim-vs-once distinction, mass
conservation of the net, the untouched obligation, make_pnl equivalence, signed
loss severity under pnl, function-valued consideration, and the collection
rejections.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from aggregate import build, PnL
from aggregate.constants import DefectiveDistributionWarning

# severities chosen light enough that the 12-nines window is well-resolved, so
# the empirical moments match the analytic ones tightly.
TOL = 5e-3


# ----------------------------------------------------------------------
# Type / orientation
# ----------------------------------------------------------------------
def test_build_pnl_returns_pnl():
    """``build('pnl ...')`` returns a PnL whose net is always payoff."""
    a = build('pnl B 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    assert isinstance(a, PnL)
    assert a.value_type == 'payoff'
    # the risky leg is an ordinary loss aggregate, untouched
    assert a.agg.value_type == 'loss'
    assert not a.agg._signed()                 # loss severity is non-negative


def test_obligation_is_untouched():
    """``pnl.agg`` is the honest loss obligation (E[X]=700), not the net."""
    a = build('pnl B 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    assert a.agg.agg_m == pytest.approx(700.0, rel=TOL)
    # net = 1000 - 700 = 300
    assert a.mean == pytest.approx(300.0, rel=TOL, abs=2.0)


# ----------------------------------------------------------------------
# Moment closed forms across the three exposure forms
# ----------------------------------------------------------------------
@pytest.mark.parametrize('program,consid,e_loss', [
    ('pnl X 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson', 1000.0, 700.0),
    ('pnl X 100 prem - 7 claims sev gamma 100 cv 0.5 poisson', 100.0, 700.0),
    ('pnl X 100 prem - 700 loss sev gamma 100 cv 0.5 poisson', 100.0, 700.0),
])
def test_mean_across_exposure_forms(program, consid, e_loss):
    """net mean = consideration - E[loss] for lr / claims / loss heads."""
    a = build(program)
    assert a.mean == pytest.approx(consid - e_loss, rel=TOL, abs=TOL)


def test_sd_invariant_skew_flips():
    """sd unchanged, skew sign-flipped relative to the bare loss aggregate."""
    pnl = build('pnl X 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    loss = build('agg L 1000 prem at 0.7 lr sev gamma 100 cv 0.5 poisson')
    assert pnl.mean == pytest.approx(1000 - loss.est_m, rel=TOL, abs=TOL)
    # spread invariant under the consideration shift + reflection
    assert pnl.sd == pytest.approx(loss.est_sd, rel=TOL)
    # reflection flips the skew sign
    assert pnl.skew == pytest.approx(-loss.est_skew, rel=2e-2, abs=1e-3)


def test_make_pnl_equivalence():
    """``build('pnl ...')`` == ``build('agg ...').make_pnl(consideration=...)``."""
    direct = build('pnl X 100 prem - 7 claims sev gamma 100 cv 0.5 poisson')
    via = build('agg L 7 claims sev gamma 100 cv 0.5 poisson').make_pnl(100)
    assert via.value_type == 'payoff'
    assert via.mean == pytest.approx(direct.mean, rel=TOL, abs=TOL)
    assert via.sd == pytest.approx(direct.sd, rel=TOL)


# ----------------------------------------------------------------------
# Vector consideration sums to one book amount
# ----------------------------------------------------------------------
def test_vector_consideration():
    """A vector premium sums to one book consideration; mean follows."""
    a = build('pnl X [100 200 100] prem - .8 lr [1000 2000 5000] xs 0 '
              'sev lognorm 500 cv 2 poisson')
    # consideration = sum = 400; mean = 400 - 0.8 * 400 = 80
    assert a.mean == pytest.approx(80.0, rel=TOL, abs=0.5)


# ----------------------------------------------------------------------
# P(loss) = P(net < 0) = loss survival at the consideration
# ----------------------------------------------------------------------
def test_prob_loss_matches_loss_survival():
    pnl = build('pnl X 100 prem - 7 claims sev gamma 100 cv 0.5 poisson')
    loss = build('agg L 7 claims sev gamma 100 cv 0.5 poisson')
    assert pnl.prob_loss == pytest.approx(float(loss.sf(100)), rel=2e-2, abs=2e-3)


# ----------------------------------------------------------------------
# Per-claim (ssev) vs once-for-the-book (pnl) distinction
# ----------------------------------------------------------------------
def test_per_claim_vs_once_distinction():
    """``pnl 100 prem - 5 claims`` (once) differs from a per-claim constant."""
    once = build('pnl P 100 prem - 5 claims sev gamma 8 cv 0.5 poisson')
    per_claim = build('agg S 5 claims ssev 100 - gamma 8 cv 0.5 poisson')
    assert once.mean == pytest.approx(100 - 5 * 8, rel=TOL, abs=0.5)
    assert per_claim.est_m == pytest.approx(5 * (100 - 8), rel=2e-2, abs=1.0)
    assert abs(once.mean - per_claim.est_m) > 100


# ----------------------------------------------------------------------
# Mass conservation + the net grid straddles 0
# ----------------------------------------------------------------------
def test_mass_conserved_and_grid_straddles_zero():
    a = build('pnl X 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    df = a.pnl_df
    assert df.p_total.sum() == pytest.approx(1.0, abs=1e-6)
    net = df.index.to_numpy(float)
    assert net.min() < a.mean < net.max()
    assert net.min() < 0 < net.max()           # a P&L can be a loss


# ----------------------------------------------------------------------
# Distribution functions on the net
# ----------------------------------------------------------------------
def test_cdf_q_sf_consistent():
    a = build('pnl X 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    # sf is exactly 1 - cdf
    assert a.sf(50.0) == pytest.approx(1 - a.cdf(50.0), abs=1e-12)
    # prob_loss = P(net<0); cdf(0) = P(net<=0): equal up to the atom at 0 (one bucket)
    assert a.prob_loss == pytest.approx(a.cdf(0.0), abs=5e-3)
    # median bracketed by the grid; cdf at the median ~ 0.5
    med = a.q(0.5)
    assert a.cdf(med) >= 0.5 - 1e-9


# ----------------------------------------------------------------------
# Function-valued (loss-sensitive) consideration -- passed by hand
# ----------------------------------------------------------------------
def test_function_valued_consideration():
    """A callable consideration f(x) nets bucket-wise: net = f(x) - x.

    A profit-commission style slide: keep 80% of the premium plus a 20%
    rebate on low losses. Here use a simple swing ``f(x) = 100 + 0.5*x`` so
    the net is ``100 - 0.5*x`` -- still comonotone, SD scaled by 0.5.
    """
    base = build('agg L 5 claims sev gamma 100 cv 0.5 poisson')
    flat = base.make_pnl(100.0)
    swing = base.make_pnl(lambda x: 100.0 + 0.5 * x)
    # net mean: flat = 100 - E[X]; swing = 100 - 0.5 E[X]
    assert swing.mean == pytest.approx(100.0 - 0.5 * base.agg_m, rel=TOL, abs=1.0)
    # the swing absorbs half the loss volatility -> SD halved
    assert swing.sd == pytest.approx(0.5 * flat.sd, rel=1e-2)
    assert swing.pnl_df.p_total.sum() == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------------
# Signed loss severity (dsev with a negative atom / ssev) under pnl
# ----------------------------------------------------------------------
def test_signed_dsev_pnl_exact():
    """``pnl 5 prem - dfreq[3] dsev[-1 1]`` -> net in {2,4,6,8}, mean 5, sd sqrt3."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=DefectiveDistributionWarning)
        a = build('pnl GP 5 premium - dfreq[3] dsev[-1 1]', bs=1)
    df = a.pnl_df
    m = df.p_total.to_numpy() > 1e-12
    support = df.index.to_numpy(float)[m]
    probs = df.p_total.to_numpy()[m]
    assert df.p_total.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(support, [2.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(probs, [0.125, 0.375, 0.375, 0.125], atol=1e-12)
    assert a.mean == pytest.approx(5.0, abs=1e-9)
    assert a.sd == pytest.approx(np.sqrt(3.0), abs=1e-9)
    assert a.skew == pytest.approx(0.0, abs=1e-9)


def test_signed_dsev_pnl_asymmetric_mean():
    """Mean closed form for a non-symmetric signed dsev: E[net]=C - 2 E[X]."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('pnl Y 10 premium - dfreq[2] dsev[-2 1 3] [.5 .3 .2]', bs=1)
    # per-claim E[X] = -2(.5) + 1(.3) + 3(.2) = -0.1; two claims -> E[L] = -0.2
    assert a.pnl_df.p_total.sum() == pytest.approx(1.0, abs=1e-12)
    assert a.mean == pytest.approx(10.0 - 2 * (-0.1), abs=1e-9)


def test_signed_ssev_pnl_mass_and_mean():
    """Continuous signed severity (``ssev``) under pnl conserves mass; mean shifts."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('pnl Z 100 premium - 5 claims ssev 20 - lognorm 10 cv 0.5 poisson')
    assert a.agg._signed_severity()
    assert a.pnl_df.p_total.sum() == pytest.approx(1.0, abs=1e-5)
    # loss sev = 20 - lognorm(mean 10) -> per-claim mean 10; 5 claims -> E[L]=50
    assert a.mean == pytest.approx(100 - 50, abs=0.5)


# ----------------------------------------------------------------------
# Portfolios / bivariates of pnl units are deferred -> clear rejection
# ----------------------------------------------------------------------
def test_portfolio_of_pnl_rejected():
    """A book-level P&L (pnl units in a port) is deferred -> NotImplementedError."""
    with pytest.raises(NotImplementedError, match='pnl units in a portfolio'):
        build('''port Book
            pnl A 1000 prem - 80% lr sev gamma 100 cv 0.3 poisson
            pnl B 1000 prem - 80% lr sev gamma 100 cv 0.3 poisson
        ''', update=False)


def test_bivariate_with_pnl_component_rejected():
    """A pnl component in a bivariate (joint P&L) is deferred -> NotImplementedError."""
    with pytest.raises(NotImplementedError, match='pnl components in a bivariate'):
        build('bivariate BV 5 claims '
              'agg A 1 claim sev lognorm 10 cv 1 poisson '
              'pnl B 100 prem - 1 claim sev lognorm 10 cv 1 poisson '
              'copula gumbel 0.4', update=False)
