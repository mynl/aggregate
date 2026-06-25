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


# ----------------------------------------------------------------------
# Signed, additive summary_df (Consideration / Obligation / Margin)
# ----------------------------------------------------------------------
def test_summary_df_signed_additive():
    """A sold cover: Consideration +, Obligation -, Margin = their sum; SD not CV."""
    a = build('pnl B 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    df = a.summary_df
    assert list(df.index) == ['Consideration', 'Obligation', 'Margin']
    assert list(df.columns) == ['EX', 'SD', 'Sk']        # SD trio, no CV
    assert df.loc['Consideration', 'EX'] == pytest.approx(1000.0)
    assert df.loc['Consideration', 'SD'] == 0.0          # a constant is certain
    assert df.loc['Obligation', 'EX'] == pytest.approx(-700.0, rel=TOL)  # loss subtracts
    # rows add (the defining property): Consideration + Obligation = Margin
    assert (df.loc['Consideration', 'EX'] + df.loc['Obligation', 'EX']
            == pytest.approx(df.loc['Margin', 'EX'], abs=1e-6))
    # the margin SD is the obligation SD (constant consideration adds no spread)
    assert df.loc['Margin', 'SD'] == pytest.approx(df.loc['Obligation', 'SD'])


def test_summary_df_bought_flips_both_signs():
    """Buying a payoff: Consideration < 0 (paid) AND Obligation > 0 (held)."""
    b = build('agg P 5 claims sev gamma 8 cv .5 poisson payoff').make_pnl(-100)
    df = b.summary_df
    assert df.loc['Consideration', 'EX'] == pytest.approx(-100.0)
    assert df.loc['Obligation', 'EX'] > 0                 # payoff held adds
    assert (df.loc['Consideration', 'EX'] + df.loc['Obligation', 'EX']
            == pytest.approx(df.loc['Margin', 'EX'], abs=1e-6))


def test_summary_df_function_consideration_has_spread():
    """A loss-sensitive (callable) consideration carries a real SD/Sk."""
    base = build('agg L 5 claims sev gamma 100 cv 0.5 poisson')
    swing = base.make_pnl(lambda x: 100.0 + 0.5 * x)
    df = swing.summary_df
    assert df.loc['Consideration', 'SD'] > 0              # f(X) varies
    assert (df.loc['Consideration', 'EX'] + df.loc['Obligation', 'EX']
            == pytest.approx(df.loc['Margin', 'EX'], abs=1e-6))


# ----------------------------------------------------------------------
# PnL.plot(): Margin density + distribution, no severity panel
# ----------------------------------------------------------------------
def test_plot_has_two_panels_no_sev():
    import matplotlib
    matplotlib.use('Agg')
    a = build('pnl B 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    fig = a.plot()
    # exactly two panels (density + distribution); no severity / Lee panel
    assert len(fig.axes) == 2


def test_plot_discrete_runs():
    import matplotlib
    matplotlib.use('Agg')
    a = build('pnl GP 5 premium - dfreq[3] dsev[-1 1]', bs=1)
    fig = a.plot()
    assert len(fig.axes) == 2


# ----------------------------------------------------------------------
# evaluate(): the Cherny-Madan breakeven acceptability panel
# ----------------------------------------------------------------------
def test_evaluate_panel_shape_and_breakeven():
    """The panel has the default families minus ccoc; breakeven is solved."""
    a = build('pnl B 1000 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    ev = a.evaluate()
    assert list(ev.index) == ['ph', 'wang', 'dual', 'tvar']    # ccoc excluded
    assert list(ev.columns) == ['param_name', 'param', 'error', 'gini_p', 'area']
    # gini_p is the family-agnostic acceptability index in [0, 1]
    assert (ev.gini_p >= 0).all() and (ev.gini_p <= 1).all()
    # area = (gini_p + 1) / 2 = integral g
    assert np.allclose(ev.area, (ev.gini_p + 1) / 2)
    # every family hit its breakeven target (calibration residual ~ 0)
    assert (ev.error.abs() < 1e-2).all()


def test_evaluate_gini_p_monotone_in_profit():
    """A more profitable position survives a larger stress -> larger gini_p."""
    lo = build('pnl L 800 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    hi = build('pnl H 1200 prem - 70% lr sev gamma 100 cv 0.5 poisson')
    for fam in ('ph', 'wang', 'dual', 'tvar'):
        assert lo.evaluate().loc[fam, 'gini_p'] < hi.evaluate().loc[fam, 'gini_p']


def test_evaluate_function_consideration_deferred():
    """Function-valued (loss-sensitive) evaluation is deferred -> clear error."""
    pnl = build('agg L 5 claims sev gamma 100 cv 0.5 poisson').make_pnl(
        lambda x: 100.0 + 0.5 * x)
    with pytest.raises(NotImplementedError, match='constant consideration'):
        pnl.evaluate()


# ----------------------------------------------------------------------
# Reinsurance-aware Gross / Ceded / Net view
# ----------------------------------------------------------------------
_REINS = 'agg R 100 claims sev lognorm 50 cv 1.5 poisson aggregate net of 2000 xs 3000'


def test_gcn_doubly_additive():
    """gcn_df: rows add (Net = Gross + Ceded) AND columns add (Margin = C + O)."""
    g = build(_REINS).make_pnl(gross=5500, ceded=1800).gcn_df
    assert list(g.index) == ['Gross', 'Ceded', 'Net']
    assert list(g.columns) == ['Consideration', 'Obligation', 'Margin']
    # rows: the comonotone legs add
    assert np.allclose(g.loc['Net'].to_numpy(),
                       g.loc['Gross'].to_numpy() + g.loc['Ceded'].to_numpy())
    # columns: Margin = Consideration + Obligation
    assert np.allclose(g['Margin'].to_numpy(),
                       g['Consideration'].to_numpy() + g['Obligation'].to_numpy())
    # the ceded leg is literally negative: pay premium, receive recovery
    assert g.loc['Ceded', 'Consideration'] == pytest.approx(-1800.0)
    assert g.loc['Ceded', 'Obligation'] > 0          # recovery is a gain


def test_gcn_summary_df_is_gcn():
    """summary_df routes to the GCN exhibit for a Gross/Ceded/Net position."""
    p = build(_REINS).make_pnl(gross=5500, ceded=1800)
    assert list(p.summary_df.index) == ['Gross', 'Ceded', 'Net']
    # net consideration defaults to gross - ceded
    assert p.consideration == pytest.approx(3700.0)


def test_gcn_net_override():
    """net= overrides the derived gross - ceded retained premium."""
    p = build(_REINS).make_pnl(gross=5500, ceded=1800, net=4000)
    assert p.consideration == pytest.approx(4000.0)
    assert p.gcn_df.loc['Net', 'Consideration'] == pytest.approx(4000.0)


def test_gcn_net_leg_drives_moments_and_evaluate():
    """Net is the headline: it drives pnl_df / moments / evaluate."""
    agg = build(_REINS)
    p = agg.make_pnl(gross=5500, ceded=1800)
    e_net = float((agg.xs * agg.agg_density_net).sum())
    assert p.mean == pytest.approx(3700.0 - e_net, rel=1e-3)
    # evaluate runs on the net leg
    assert list(p.evaluate().index) == ['ph', 'wang', 'dual', 'tvar']


def test_net_only_on_reins_agg():
    """make_pnl(consideration=) on a reins agg is a net-only P&L (no GCN)."""
    agg = build(_REINS)
    p = agg.make_pnl(consideration=3700)
    assert p._gcn is None
    e_net = float((agg.xs * agg.agg_density_net).sum())
    assert p.mean == pytest.approx(3700.0 - e_net, rel=1e-3)
    assert list(p.summary_df.index) == ['Consideration', 'Obligation', 'Margin']


def test_gcn_requires_both_premiums_and_agg_reins():
    agg = build(_REINS)
    with pytest.raises(ValueError, match='both gross'):
        agg.make_pnl(gross=5500)
    with pytest.raises(ValueError, match='not both'):
        agg.make_pnl(consideration=100, gross=5500, ceded=1800)
    # no aggregate reinsurance -> no gross/ceded/net views
    plain = build('agg P 100 claims sev lognorm 50 cv 1.5 poisson')
    with pytest.raises(ValueError, match='aggregate reinsurance'):
        plain.make_pnl(gross=5500, ceded=1800)


def test_gcn_plot_overlays_three_legs():
    import matplotlib
    matplotlib.use('Agg')
    p = build(_REINS).make_pnl(gross=5500, ceded=1800)
    fig = p.plot()
    assert len(fig.axes) == 2
    # three lines (Gross/Ceded/Net) on the density panel
    assert len(fig.axes[0].get_lines()) >= 3
