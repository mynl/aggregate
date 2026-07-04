"""Tests for the ``pnl`` keyword and the first-class :class:`PnL` value object.

``pnl NAME <consideration> prem less agg NAME_e <loss body>`` builds a pure-loss
aggregate X (the obligation) and snapshots it into a :class:`PnL` whose result
(net) is ``consideration - X``. The consideration is a single amount for the
book, in contrast to a constant inside ``sev``/``dsev``/``ssev`` which is
per-claim. ``build('pnl ...')`` and
``build('agg ...').make_pnl(consideration=...)`` coincide.

A :class:`PnL` **consumes and discards** its stochastic engine: there is no
``pnl.agg`` and no ``value_type``. The ledger rows are **signed cash flows**
(``dev/plan-yapnl.md``): the sold obligation books at ``-E[X]`` and the ``EX``
column adds down the sheet to the ``margin`` result. ``build('pnl ...')``
always returns a :class:`PnL`; a ceded-premium clause books the cession as a
real ``buy`` group in the ledger (with the resolved economics on
``pnl.economics``). Portfolios / bivariates of ``pnl`` units are deferred
(book-level P&L), and rejected with a clear error.

Covers: the PnL return type, the three exposure forms (lr / claims / loss),
the moment closed forms (mean shift, sd invariant, skew sign flip), vector
consideration, P(loss), the per-claim-vs-once distinction, mass conservation
of the net, the signed obligation leg, make_pnl equivalence, signed loss
severity under pnl, function-valued consideration, the summary_df shape, the
Gross/Ceded/Net group ledger, and the collection rejections.
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
    """``build('pnl ...')`` returns a PnL that sells the obligation (net payoff)."""
    a = build('pnl B 1000 prem less agg B_e 1000 prem at 70% lr sev gamma 100 cv 0.5 poisson')
    assert isinstance(a, PnL)
    assert a.role == 'sell'                     # receive consideration, owe loss
    assert a.result_name == 'margin'
    # the obligation row is the signed sold loss: booked non-positive
    assert a.density_df['loss'].x.max() <= 0.0


def test_obligation_books_signed():
    """The obligation row books the signed loss (-E[X] = -700), and EX foots."""
    a = build('pnl B 1000 prem less agg B_e 1000 prem at 70% lr sev gamma 100 cv 0.5 poisson')
    assert a.stats_df.loc[('Obligation', 'loss'), 'EX'] == \
        pytest.approx(-700.0, rel=TOL)
    # net = 1000 - 700 = 300
    assert a.mean == pytest.approx(300.0, rel=TOL, abs=2.0)


# ----------------------------------------------------------------------
# Moment closed forms across the three exposure forms
# ----------------------------------------------------------------------
@pytest.mark.parametrize('program,consid,e_loss', [
    ('pnl X 1000 prem less agg X_e 1000 prem at 70% lr sev gamma 100 cv 0.5 poisson', 1000.0, 700.0),
    ('pnl X 100 prem less agg X_e 7 claims sev gamma 100 cv 0.5 poisson', 100.0, 700.0),
    ('pnl X 100 prem less agg X_e 700 loss sev gamma 100 cv 0.5 poisson', 100.0, 700.0),
])
def test_mean_across_exposure_forms(program, consid, e_loss):
    """net mean = consideration - E[loss] for lr / claims / loss heads."""
    a = build(program)
    assert a.mean == pytest.approx(consid - e_loss, rel=TOL, abs=TOL)


def test_sd_invariant_skew_flips():
    """sd unchanged, skew sign-flipped relative to the bare loss aggregate."""
    pnl = build('pnl X 1000 prem less agg X_e 1000 prem at 70% lr sev gamma 100 cv 0.5 poisson')
    loss = build('agg L 1000 prem at 0.7 lr sev gamma 100 cv 0.5 poisson')
    assert pnl.mean == pytest.approx(1000 - loss.est_m, rel=TOL, abs=TOL)
    # spread invariant under the consideration shift + reflection
    assert pnl.sd == pytest.approx(loss.est_sd, rel=TOL)
    # reflection flips the skew sign
    assert pnl.skew == pytest.approx(-loss.est_skew, rel=2e-2, abs=1e-3)


def test_make_pnl_equivalence():
    """``build('pnl ...')`` == ``build('agg ...').make_pnl(consideration=...)``."""
    direct = build('pnl X 100 prem less agg X_e 7 claims sev gamma 100 cv 0.5 poisson')
    via = build('agg L 7 claims sev gamma 100 cv 0.5 poisson').make_pnl(100)
    assert isinstance(via, PnL)
    assert via.mean == pytest.approx(direct.mean, rel=TOL, abs=TOL)
    assert via.sd == pytest.approx(direct.sd, rel=TOL)


# ----------------------------------------------------------------------
# Vector consideration sums to one book amount
# ----------------------------------------------------------------------
def test_vector_consideration():
    """A vector premium sums to one book consideration; mean follows."""
    a = build('pnl X [100 200 100] prem less agg X_e [100 200 100] prem at .8 lr [1000 2000 5000] xs 0 '
              'sev lognorm 500 cv 2 poisson')
    # consideration = sum = 400; mean = 400 - 0.8 * 400 = 80
    assert a.mean == pytest.approx(80.0, rel=TOL, abs=0.5)


# ----------------------------------------------------------------------
# P(loss) = P(net < 0) = loss survival at the consideration
# ----------------------------------------------------------------------
def test_prob_loss_matches_loss_survival():
    pnl = build('pnl X 100 prem less agg X_e 7 claims sev gamma 100 cv 0.5 poisson')
    loss = build('agg L 7 claims sev gamma 100 cv 0.5 poisson')
    assert pnl.prob_loss == pytest.approx(float(loss.sf(100)), rel=2e-2, abs=2e-3)


# ----------------------------------------------------------------------
# Per-claim (ssev) vs once-for-the-book (pnl) distinction
# ----------------------------------------------------------------------
def test_per_claim_vs_once_distinction():
    """``pnl 100 prem less 5 claims`` (once) differs from a per-claim constant."""
    once = build('pnl P 100 prem less agg P_e 5 claims sev gamma 8 cv 0.5 poisson')
    per_claim = build('agg S 5 claims ssev 100 - gamma 8 cv 0.5 poisson')
    assert once.mean == pytest.approx(100 - 5 * 8, rel=TOL, abs=0.5)
    assert per_claim.est_m == pytest.approx(5 * (100 - 8), rel=2e-2, abs=1.0)
    assert abs(once.mean - per_claim.est_m) > 100


# ----------------------------------------------------------------------
# Mass conservation + the net grid straddles 0
# ----------------------------------------------------------------------
def test_mass_conserved_and_grid_straddles_zero():
    a = build('pnl X 1000 prem less agg X_e 1000 prem at 70% lr sev gamma 100 cv 0.5 poisson')
    gd = a.result
    assert gd.p.sum() == pytest.approx(1.0, abs=1e-6)
    net = gd.x
    assert net.min() < a.mean < net.max()
    assert net.min() < 0 < net.max()           # a P&L can be a loss


# ----------------------------------------------------------------------
# Distribution functions on the net
# ----------------------------------------------------------------------
def test_cdf_q_sf_consistent():
    a = build('pnl X 1000 prem less agg X_e 1000 prem at 70% lr sev gamma 100 cv 0.5 poisson')
    # sf is exactly 1 - cdf
    assert a.sf(50.0) == pytest.approx(1 - a.cdf(50.0), abs=1e-12)
    # prob_loss = P(net<0); cdf(0) = P(net<=0): equal up to the atom at 0 (one bucket)
    assert a.prob_loss == pytest.approx(a.cdf(0.0), abs=5e-3)
    # median bracketed by the grid; cdf at the median ~ 0.5
    med = a.q(0.5)
    assert a.cdf(med) >= 0.5 - 1e-9


def test_q_cdf_tvar_delegate_to_grid_distribution():
    """PnL distribution accessors go through the net GridDistribution.

    The hand-rolled scalar searchsorted (pre-fix) crashed on array input and
    duplicated the canonical kernel; ``q`` / ``cdf`` / ``sf`` / ``var`` must all
    delegate to ``pnl.gd`` (a GridDistribution) so they vectorize and agree with
    the kernel every other class uses. Guards against re-rolling.
    """
    a = build('pnl X 1000 prem less agg X_e 1000 prem at 70% lr sev gamma 100 cv 0.5 poisson')
    from aggregate._grid_distribution import GridDistribution
    assert isinstance(a.gd, GridDistribution)
    # vectorized quantiles -- the array path that used to raise TypeError
    qs = a.q([0.01, 0.5, 0.99])
    assert np.shape(qs) == (3,)
    assert qs[0] <= qs[1] <= qs[2]
    # scalar still works and matches the vector element
    assert a.q(0.5) == pytest.approx(qs[1])
    # vectorized cdf round-trips q
    assert a.cdf(a.q(0.99)) >= 0.99 - 1e-9
    cs = a.cdf([a.q(0.01), a.q(0.99)])
    assert np.shape(cs) == (2,)
    # var is the lower-quantile alias
    assert a.var(0.9) == pytest.approx(a.q(0.9))
    # the GridDistribution is the canonical object (tvar reached there, with the
    # payoff orientation in mind -- deliberately not a bare pnl.tvar)
    assert a.gd.tvar(0.9) >= a.gd.q(0.9)


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
    assert swing.result.to_series().sum() == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------------
# Signed loss severity (dsev with a negative atom / ssev) under pnl
# ----------------------------------------------------------------------
def test_signed_dsev_pnl_exact():
    """``pnl 5 prem less dfreq[3] dsev[-1 1]`` -> net in {2,4,6,8}, mean 5, sd sqrt3."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=DefectiveDistributionWarning)
        a = build('pnl GP 5 premium less agg GP_e dfreq[3] dsev[-1 1]', bs=1)
    gd = a.result
    m = gd.p > 1e-12
    support = gd.x[m]
    probs = gd.p[m]
    assert gd.p.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(support, [2.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(probs, [0.125, 0.375, 0.375, 0.125], atol=1e-12)
    assert a.mean == pytest.approx(5.0, abs=1e-9)
    assert a.sd == pytest.approx(np.sqrt(3.0), abs=1e-9)
    assert a.skew == pytest.approx(0.0, abs=1e-9)


def test_signed_dsev_pnl_asymmetric_mean():
    """Mean closed form for a non-symmetric signed dsev: E[net]=C - 2 E[X]."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('pnl Y 10 premium less agg Y_e dfreq[2] dsev[-2 1 3] [.5 .3 .2]', bs=1)
    # per-claim E[X] = -2(.5) + 1(.3) + 3(.2) = -0.1; two claims -> E[L] = -0.2
    assert a.result.p.sum() == pytest.approx(1.0, abs=1e-12)
    assert a.mean == pytest.approx(10.0 - 2 * (-0.1), abs=1e-9)


def test_signed_ssev_pnl_mass_and_mean():
    """Continuous signed severity (``ssev``) under pnl conserves mass; mean shifts."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('pnl Z 100 premium less agg Z_e 5 claims ssev 20 - lognorm 10 cv 0.5 poisson')
    assert a.result.p.sum() == pytest.approx(1.0, abs=1e-5)
    # loss sev = 20 - lognorm(mean 10) -> per-claim mean 10; 5 claims -> E[L]=50
    assert a.mean == pytest.approx(100 - 50, abs=0.5)


# ----------------------------------------------------------------------
# Portfolios / bivariates of pnl units are deferred -> clear rejection
# ----------------------------------------------------------------------
def test_portfolio_of_pnl_rejected():
    """A book-level P&L (pnl units in a port) is deferred -> NotImplementedError."""
    with pytest.raises(NotImplementedError, match='pnl units in a portfolio'):
        build('''port Book
            pnl A 1000 prem less agg A_e 1000 prem at 80% lr sev gamma 100 cv 0.3 poisson
            pnl B 1000 prem less agg B_e 1000 prem at 80% lr sev gamma 100 cv 0.3 poisson
        ''', update=False)


def test_bivariate_with_pnl_component_rejected():
    """A pnl component in a bivariate (joint P&L) is deferred -> NotImplementedError."""
    with pytest.raises(NotImplementedError, match='pnl components in a bivariate'):
        build('bivariate BV 5 claims '
              'agg A 1 claim sev lognorm 10 cv 1 poisson '
              'pnl B 100 prem less agg B_e 1 claim sev lognorm 10 cv 1 poisson '
              'copula gumbel 0.4', update=False)


# ----------------------------------------------------------------------
# summary_df: the fixed card (Consideration / Obligation / Margin)
# ----------------------------------------------------------------------
_CARD_COLS = ['EX', 'Scaled', 'SD', 'CV', 'Skew', 'P01', 'Median', 'P99']


def test_summary_df_fixed_card_and_additive_result():
    """A sold cover: the fixed three-row card; the EX column adds down it."""
    a = build('pnl B 1000 prem less agg B_e 1000 prem at 70% lr sev gamma 100 cv 0.5 poisson')
    df = a.summary_df
    assert list(df.index) == ['Consideration', 'Obligation', 'Margin']
    assert list(df.columns) == _CARD_COLS
    assert df.loc['Consideration', 'EX'] == pytest.approx(1000.0)
    # a constant consideration renormalizes to SD exactly 0 (no spurious spread)
    assert df.loc['Consideration', 'SD'] == 0.0
    assert df.loc['Obligation', 'EX'] == pytest.approx(-700.0, rel=TOL)
    # the defining identity: the EX column adds down the card
    assert df.loc['Margin', 'EX'] == pytest.approx(
        df.loc['Consideration', 'EX'] + df.loc['Obligation', 'EX'], abs=1e-6)
    # Scaled divides by the committed E[total consideration]: -(loss ratio)
    assert df.loc['Obligation', 'Scaled'] == pytest.approx(-0.70, rel=TOL)
    # the margin SD is the loss SD (constant consideration adds no spread)
    assert df.loc['Margin', 'SD'] == pytest.approx(df.loc['Obligation', 'SD'])


def test_summary_df_negative_consideration():
    """A negative consideration passes through signed; the EX column foots."""
    b = build('agg P 5 claims sev gamma 8 cv .5 poisson').make_pnl(-100)
    df = b.summary_df
    assert df.loc['Consideration', 'EX'] == pytest.approx(-100.0)
    assert df.loc['Obligation', 'EX'] < 0                 # signed sold loss
    assert df.loc['Margin', 'EX'] == pytest.approx(
        df.loc['Consideration', 'EX'] + df.loc['Obligation', 'EX'], abs=1e-6)


def test_summary_df_function_consideration_has_spread():
    """A loss-sensitive (callable) consideration carries a real SD/Skew."""
    base = build('agg L 5 claims sev gamma 100 cv 0.5 poisson')
    swing = base.make_pnl(lambda x: 100.0 + 0.5 * x)
    df = swing.summary_df
    assert df.loc['Consideration', 'SD'] > 0              # f(X) varies
    assert df.loc['Margin', 'EX'] == pytest.approx(
        df.loc['Consideration', 'EX'] + df.loc['Obligation', 'EX'], abs=1e-6)


# ----------------------------------------------------------------------
# PnL.plot(): net density + distribution, no severity panel
# ----------------------------------------------------------------------
def test_plot_has_two_panels_no_sev():
    import matplotlib
    matplotlib.use('Agg')
    a = build('pnl B 1000 prem less agg B_e 1000 prem at 70% lr sev gamma 100 cv 0.5 poisson')
    fig = a.plot()
    # exactly two panels (density + distribution); no severity / Lee panel
    assert len(fig.axes) == 2


def test_plot_discrete_runs():
    import matplotlib
    matplotlib.use('Agg')
    a = build('pnl GP 5 premium less agg GP_e dfreq[3] dsev[-1 1]', bs=1)
    fig = a.plot()
    assert len(fig.axes) == 2


# ----------------------------------------------------------------------
# evaluate(): the Cherny-Madan breakeven acceptability panel
# ----------------------------------------------------------------------
def test_evaluate_panel_shape_and_breakeven():
    """The panel has the default families minus ccoc; breakeven is solved."""
    a = build('pnl B 1000 prem less agg B_e 1000 prem at 70% lr sev gamma 100 cv 0.5 poisson')
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
    lo = build('pnl L 800 prem less agg L_e 800 prem at 70% lr sev gamma 100 cv 0.5 poisson')
    hi = build('pnl H 1200 prem less agg H_e 1200 prem at 70% lr sev gamma 100 cv 0.5 poisson')
    for fam in ('ph', 'wang', 'dual', 'tvar'):
        assert lo.evaluate().loc[fam, 'gini_p'] < hi.evaluate().loc[fam, 'gini_p']


def test_evaluate_function_consideration_runs():
    """Function-valued (loss-sensitive) evaluation now runs over the obligation.

    The single-obligation, regular-grid requirement is met, so ``evaluate``
    produces the standard family panel (the old ``constant consideration``
    deferral was removed).
    """
    pnl = build('agg L 5 claims sev gamma 100 cv 0.5 poisson').make_pnl(
        lambda x: 100.0 + 0.5 * x)
    ev = pnl.evaluate()
    assert list(ev.index) == ['ph', 'wang', 'dual', 'tvar']


# ----------------------------------------------------------------------
# Reinsurance-aware consolidated pnl + the stitched xpnl walk
# ----------------------------------------------------------------------
_REINS = 'agg R 100 claims sev lognorm 50 cv 1.5 poisson aggregate net of 2000 xs 3000'

#: the agg-only walk stats template (gross sell + agg cession buy)
_WALK_ROWS = [
    ('gross', 'Consideration', 'premium'),
    ('gross', 'Obligation', 'loss'),
    ('gross', 'Margin', 'Total'),
    ('ceded agg', 'Consideration', 'ceded agg premium'),
    ('ceded agg', 'Obligation', 'ceded agg recovery'),
    ('ceded agg', 'Margin', 'Total'),
    ('ceded agg', 'Margin', 'Net'),
    ('Total', 'Consideration', 'Total'),
    ('Total', 'Obligation', 'Total'),
    ('Total', 'Margin', 'Total'),
    ('Total', 'Margin', 'Impact')]


def test_consolidated_pnl_single_group_net_view():
    """make_pnl(gross=, ceded=) on a reinsured aggregate returns the
    consolidated single-group net view ([Decision-PnL-Is-Consolidated]):
    net premium consideration, net loss obligation, flat card. The walk is
    the xpnl face."""
    agg = build(_REINS)
    pnl = agg.make_pnl(gross=5500, ceded=1800)
    assert isinstance(pnl, PnL)
    assert len(pnl.groups) == 1
    s = pnl.stats_df
    assert list(s.index) == [('Consideration', 'net premium'),
                             ('Obligation', 'loss (net)'),
                             ('Margin', 'Total')]
    assert s.loc[('Consideration', 'net premium'), 'EX'] == \
        pytest.approx(3700.0)
    # the net loss reads the engine's deepest net marginal
    e_net = float((agg.xs * agg.agg_density_net).sum())
    assert s.loc[('Obligation', 'loss (net)'), 'EX'] == \
        pytest.approx(-e_net, rel=1e-6)
    # the resolved economics carry the scalar-API premiums
    assert pnl.economics['gross'] == pytest.approx(5500.0)
    assert pnl.economics['ceded'] == pytest.approx(1800.0)


def test_consolidated_net_position_drives_moments():
    """The margin is the net position, driving moments / q; the card is the
    flat three-row shape with Consideration = net premium."""
    agg = build(_REINS)
    p = agg.make_pnl(gross=5500, ceded=1800)
    e_net = float((agg.xs * agg.agg_density_net).sum())
    assert p.mean == pytest.approx(3700.0 - e_net, rel=1e-3)
    df = p.summary_df
    assert list(df.index) == ['Consideration', 'Obligation', 'Margin']
    assert df.loc['Consideration', 'EX'] == pytest.approx(3700.0)
    # q delegates to the grand result GridDistribution
    assert p.q(0.5) == pytest.approx(p.gd.q(0.5))


def test_xpnl_walk_rows_and_means_add():
    """The agg-only walk is a two-group stitched tower: gross ``sell`` +
    agg cession ``buy``; the EX column adds down the sheet by linearity;
    the impact row is the cession's step delta; the ladder stays marginal
    (plain P headers -- no shared joint)."""
    from aggregate._pnl_builders import build_xpnl_walk
    agg = build(_REINS)
    x = build_xpnl_walk(agg, gross=5500, ceded=1800)
    assert isinstance(x, PnL)
    s = x.stats_df
    assert list(s.index) == _WALK_ROWS
    assert s.loc[('Total', 'Margin', 'Total'), 'EX'] == pytest.approx(
        s.loc[('gross', 'Margin', 'Total'), 'EX']
        + s.loc[('ceded agg', 'Margin', 'Total'), 'EX'], abs=1e-6)
    # total impact = the cession's step delta (result vs the gross result)
    assert s.loc[('Total', 'Margin', 'Impact'), 'EX'] == pytest.approx(
        s.loc[('ceded agg', 'Margin', 'Total'), 'EX'], abs=1e-6)
    # the cession books contra: -premium, +recovery
    assert s.loc[('ceded agg', 'Consideration', 'ceded agg premium'), 'EX'] \
        == pytest.approx(-1800.0)
    assert s.loc[('ceded agg', 'Obligation', 'ceded agg recovery'), 'EX'] > 0
    assert 'P01' in s.columns and 'κ01' not in s.columns
    # SDs do not add: the cession trims the tail
    assert s.loc[('Total', 'Margin', 'Total'), 'SD'] < \
        s.loc[('gross', 'Margin', 'Total'), 'SD']
    # the walk's grand result mean = the consolidated pnl's margin
    p = agg.make_pnl(gross=5500, ceded=1800)
    assert s.loc[('Total', 'Margin', 'Total'), 'EX'] == \
        pytest.approx(p.mean, abs=1e-9)


def test_net_only_on_reins_agg():
    """make_pnl(consideration=) on a reins agg is a plain net-only P&L (no tower)."""
    agg = build(_REINS)
    p = agg.make_pnl(consideration=3700)
    assert isinstance(p, PnL)
    e_net = float((agg.xs * agg.agg_density_net).sum())
    assert p.mean == pytest.approx(3700.0 - e_net, rel=1e-3)
    # single-group: a flat card and a two-level stats sheet
    assert list(p.summary_df.index) == ['Consideration', 'Obligation', 'Margin']
    assert list(p.density_df) == ['consideration', 'loss', 'margin']


def test_consolidated_requires_both_premiums_and_reins():
    agg = build(_REINS)
    with pytest.raises(ValueError, match='both gross'):
        agg.make_pnl(gross=5500)
    with pytest.raises(ValueError, match='not both'):
        agg.make_pnl(consideration=100, gross=5500, ceded=1800)
    # no reinsurance at all -> nothing to consolidate over
    plain = build('agg P 100 claims sev lognorm 50 cv 1.5 poisson')
    with pytest.raises(ValueError, match='requires reinsurance'):
        plain.make_pnl(gross=5500, ceded=1800)


def test_consolidated_and_walk_plots_run():
    import matplotlib
    matplotlib.use('Agg')
    from aggregate._pnl_builders import build_xpnl_walk
    agg = build(_REINS)
    p = agg.make_pnl(gross=5500, ceded=1800)
    assert len(p.plot().axes) == 2
    x = build_xpnl_walk(agg, gross=5500, ceded=1800)
    assert len(x.plot().axes) == 2
