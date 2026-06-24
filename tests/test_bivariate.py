"""Tests for copula-coupled bivariate aggregates.

Covers the bivariate firm-up (``dev/plan-mv.md``, stages MV-1..7):

- :class:`aggregate.copula.Copula` -- the registry/factory copula hierarchy
  (independent / normal / gumbel / clayton / fgm / shuffle): CDF boundary
  conditions, monotonicity, natural-parameter -> Kendall tau identities, the
  discrete-Sklar ``rectangle_pmf`` (marginals exact, independence factorises).
- :class:`aggregate.bivariate.BivariateAggregate` -- the ``bivariate`` DecL
  statement: marginals reproduce the standalone outer compound, dependence
  ordering (corr increases with the copula parameter, mixed adds common shock),
  the ``pnl`` axis (signed marginal + sign-flipped correlation), the measured
  axis sizing + reporting surface, the shuffle-of-Min copula, and the ``clash``
  statement.

The DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg`` under the
``MV`` section.
"""

import numpy as np
import pytest

from aggregate import build
from aggregate.copula import (
    Copula, CopulaNormal, CopulaGumbel, CopulaClayton, CopulaFGM,
    CopulaIndependent)


# ----------------------------------------------------------------------
# Copula CDF unit tests
# ----------------------------------------------------------------------

COPULA_CASES = [
    ('independent', None),
    ('normal', 0.5),
    ('gumbel', 0.4),
    ('clayton', 0.4),
    ('fgm', 0.3),
]


@pytest.mark.parametrize('kind,param', COPULA_CASES)
def test_copula_boundary_conditions(kind, param):
    c = Copula(kind) if param is None else Copula(kind, param)
    assert np.isclose(float(c.C(0.0, 0.7)), 0.0)
    assert np.isclose(float(c.C(0.7, 0.0)), 0.0)
    assert np.isclose(float(c.C(1.0, 0.7)), 0.7)
    assert np.isclose(float(c.C(0.7, 1.0)), 0.7)
    assert np.isclose(float(c.C(1.0, 1.0)), 1.0)


@pytest.mark.parametrize('kind,param', COPULA_CASES)
def test_copula_monotone_in_each_argument(kind, param):
    c = Copula(kind) if param is None else Copula(kind, param)
    u = np.linspace(0, 1, 60)
    cu = c.C(u, 0.6 * np.ones_like(u))
    cv = c.C(0.6 * np.ones_like(u), u)
    assert np.all(np.diff(cu) >= -1e-12)
    assert np.all(np.diff(cv) >= -1e-12)


def test_copula_normal_C_half_half():
    # C(.5,.5) = 1/4 + arcsin(rho)/(2 pi); for rho=.5 this is 1/3
    assert np.isclose(float(Copula('normal', 0.5).C(0.5, 0.5)), 1.0 / 3.0)


def test_copula_tau_identities():
    assert np.isclose(Copula('gumbel', tau=0.4).tau(), 0.4)
    assert np.isclose(Copula('clayton', tau=0.4).tau(), 0.4)
    # fgm: tau = 2 alpha / 9, alpha = 3 rho_s
    assert np.isclose(Copula('fgm', rho_s=0.3).tau(), 2 * (3 * 0.3) / 9)
    # normal: tau = (2/pi) arcsin(rho)
    assert np.isclose(Copula('normal', 0.5).tau(), 2 / np.pi * np.arcsin(0.5))
    assert np.isclose(Copula('independent').tau(), 0.0)


def test_copula_normal_from_tau_roundtrip():
    c = CopulaNormal.from_tau(0.4)
    assert np.isclose(c.tau(), 0.4)
    assert np.isclose(c.rho, np.sin(np.pi * 0.4 / 2))


def test_copula_factory_dispatch_and_direct():
    assert isinstance(Copula('gumbel', 0.4), CopulaGumbel)
    assert isinstance(Copula('clayton', 0.4), CopulaClayton)
    assert isinstance(Copula('normal', 0.3), CopulaNormal)
    assert isinstance(Copula('fgm', 0.2), CopulaFGM)
    assert isinstance(Copula('independent'), CopulaIndependent)
    # direct subclass construction with the natural kwarg
    assert isinstance(CopulaGumbel(tau=0.4), CopulaGumbel)


def test_copula_unknown_kind_raises():
    with pytest.raises(ValueError):
        Copula('weibull', 0.5)


@pytest.mark.parametrize('kind', ['gumbel', 'clayton', 'normal', 'fgm'])
def test_copula_param_out_of_range_raises(kind):
    with pytest.raises(ValueError):
        Copula(kind, 5.0)


@pytest.mark.parametrize('kind,param', COPULA_CASES)
def test_rectangle_pmf_marginals_exact(kind, param):
    c = Copula(kind) if param is None else Copula(kind, param)
    g0 = np.array([0.2, 0.5, 1.0])
    g1 = np.array([0.3, 0.7, 1.0])
    S = c.rectangle_pmf(g0, g1)
    assert np.allclose(S.sum(axis=1), np.diff(np.r_[0.0, g0]))
    assert np.allclose(S.sum(axis=0), np.diff(np.r_[0.0, g1]))
    assert np.all(S >= -1e-12)
    assert np.isclose(S.sum(), 1.0)


def test_rectangle_pmf_independence_factorises():
    g0 = np.array([0.2, 0.5, 1.0])
    g1 = np.array([0.3, 0.7, 1.0])
    S = Copula('fgm', 0.0).rectangle_pmf(g0, g1)
    outer = np.outer(np.diff(np.r_[0.0, g0]), np.diff(np.r_[0.0, g1]))
    assert np.allclose(S, outer)
    # gumbel tau=0 is also independence
    Sg = Copula('gumbel', 0.0).rectangle_pmf(g0, g1)
    assert np.allclose(Sg, outer)


# ----------------------------------------------------------------------
# BivariateAggregate
# ----------------------------------------------------------------------

def _mv(copula='gumbel 0.4', freq='poisson'):
    prog = f'''bivariate MV 25 claims
        agg A dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
        agg B dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
        copula {copula}
        {freq}'''
    return build(prog)


def test_mv_builds_and_mass_conserved():
    from aggregate.bivariate import BivariateAggregate
    mv = _mv()
    assert isinstance(mv, BivariateAggregate)
    assert np.isclose(mv.density.sum(), 1.0, atol=1e-6)
    assert mv.density.shape[0] >= 256 and mv.density.shape[1] >= 256


def test_mv_marginals_reproduce_standalone_outer_compound():
    # Poisson outer: peril i marginal is the thinned standalone aggregate; at
    # the (coarse) matched grid the means agree to a few percent.
    mv = _mv()
    m0, m1 = mv.marginals
    assert np.isclose(m0.sum(), 1.0, atol=1e-6)
    assert np.isclose(m1.sum(), 1.0, atol=1e-6)
    mean0 = float((m0 * mv.axis_xs[0]).sum())
    mean1 = float((m1 * mv.axis_xs[1]).sum())
    # theoretical loss means: 25 * .7 * 40 = 700, 25 * .5 * 60 = 750
    assert abs(mean0 - 700) / 700 < 0.05
    assert abs(mean1 - 750) / 750 < 0.05


def test_mv_marginal_exact_at_matched_grid():
    # The joint marginal equals the 1D outer compound of the SAME discretised
    # per-event severity g_i (Fourier identity); compare to the inner agg.
    mv = _mv()
    m1 = mv.marginals[1]
    joint_mean = float((m1 * mv.axis_xs[1]).sum())
    inner = build('agg Bev dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5',
                  bs=mv.bs[1], log2=int(np.log2(len(mv.axis_xs[1]))))
    assert np.isclose(joint_mean, inner.est_m * mv.en, rtol=2e-3)


def test_mv_dependence_ordering():
    c0 = _mv('gumbel 0.2').corr
    c1 = _mv('gumbel 0.4').corr
    c2 = _mv('gumbel 0.7').corr
    assert c0 < c1 < c2


def test_mv_independence_baseline_from_shared_count():
    # Independence copula does NOT make the aggregates independent: the shared
    # frequency drives both perils, so the output corr is a positive baseline.
    rho = _mv('fgm 0').corr
    assert 0.0 < rho < 0.4


def test_mv_mixed_adds_common_shock():
    poi = _mv('gumbel 0.4', 'poisson').corr
    mix = _mv('gumbel 0.4', 'mixed gamma .5').corr
    assert mix > poi


def test_mv_clayton_lower_tail_less_agg_corr_than_gumbel():
    # Same Kendall tau, but Clayton's lower-tail dependence yields lower
    # aggregate correlation than Gumbel's upper-tail dependence.
    assert _mv('clayton 0.4').corr < _mv('gumbel 0.4').corr


def test_mv_pnl_component_rejected():
    """A pnl component in a bivariate (joint P&L) is deferred -> clear error."""
    prog = '''bivariate PL 25 claims
        agg A dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
        pnl B 900 prem - dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
        copula gumbel 0.4
        poisson'''
    with pytest.raises(NotImplementedError, match='pnl components in a bivariate'):
        build(prog)


def test_mv_reporting_smoke():
    mv = _mv()
    assert 'bivariate object name' in mv.info
    df = mv.summary_df                      # property: Portfolio-shape validation
    assert ('shared', 'Freq') in df.index
    assert {('A', 'Agg'), ('B', 'Agg'),
            ('total', 'Agg')}.issubset(set(df.index))
    assert set(df.columns) == {'EX', 'Est EX', 'Err EX', 'CV', 'Est CV',
                               'Err CV', 'Sk', 'Est Sk'}
    # total agg theory mean is the additive E[X] + E[Y]
    assert np.isclose(float(df.loc[('total', 'Agg'), 'EX']),
                      float(df.loc[('A', 'Agg'), 'EX'])
                      + float(df.loc[('B', 'Agg'), 'EX']))
    dep = mv.dependency_df                # property: dependence structure
    assert list(dep.index) == ['Sev', 'Agg']
    assert np.isclose(float(dep.loc['Agg', 'corr']), mv.corr)
    sd = mv.stats_df                      # property: marginal moments only
    assert set(sd.columns) == {'A', 'B'}   # joint block moved to dependency_df
    dd = mv.density_df                    # property: wrapper around density
    assert dd.shape == mv.density.shape
    assert np.isclose(dd.to_numpy().sum(), 1.0, atol=1e-6)
    np.testing.assert_array_equal(dd.to_numpy(), mv.density)


# ----------------------------------------------------------------------
# MV-4: the reporting surface (info catalogue, summary_df, bs_window_df, tail_df)
# ----------------------------------------------------------------------

def test_mv_info_row_catalogue_ordered():
    """``info`` uses the fixed info_row catalogue (Agg/Port convention)."""
    from aggregate.constants import INFO_LABEL_WIDTH
    mv = _mv()
    lines = mv.info.splitlines()
    labels = [ln[:INFO_LABEL_WIDTH].strip() for ln in lines]
    # the catalogue is present and in order (a representative spine)
    spine = ['bivariate object name', 'mode', 'components', 'copula',
             'shared frequency', 'claim count', 'padding',
             'axis 0 name', 'axis 0 bs', 'axis 1 name', 'axis 1 bs',
             'correlation', 'copula tau', 'tail deficit', 'validation', 'id']
    idx = [labels.index(s) for s in spine]      # KeyError if any missing
    assert idx == sorted(idx), 'info rows out of order'
    # every line obeys the shared label width
    assert all(len(ln) >= INFO_LABEL_WIDTH for ln in lines)


def test_mv_info_netceded_catalogue():
    """netceded emits the same catalogue, copula -> comonotone, tau n/a."""
    from aggregate.constants import INFO_LABEL_WIDTH
    mv = build(f'netceded {NC_PROG}')
    d = {ln[:INFO_LABEL_WIDTH].strip(): ln[INFO_LABEL_WIDTH:].strip()
         for ln in mv.info.splitlines()}
    assert d['mode'] == 'netceded'
    assert d['copula'] == 'comonotone (netceded)'
    assert d['axis 0 kind'] == 'net'        # x=net, y=ceded convention
    assert d['axis 1 kind'] == 'ceded'
    assert d['copula tau'] == 'n/a'


def test_mv_marginal_reproduces_standalone():
    """The Agg rows of ``summary_df`` validate each marginal vs its standalone.

    The mean error here is vs the *analytic* standalone, so it carries the
    coarse-grid discretization (a few %); the invariant (marginal reproduces the
    standalone at the matched grid) holds and the deficit is tiny, so the
    one-line validation passes.
    """
    mv = _mv('gumbel 0.4', 'mixed gamma .5')
    df = mv.summary_df
    for name in ('A', 'B'):
        assert abs(float(df.loc[(name, 'Agg'), 'Err EX'])) < 0.10
    assert mv.deficit < 1e-6
    assert mv._explain_oneline() == 'not unreasonable'


def test_mv_explain_flags_clipped_book():
    """A pinned (bs, log2) too small to cover clips the joint -> validation flags it."""
    mv = _mv('gumbel 0.4', 'mixed gamma .5')
    mv.update(bs=(1.0, 1.0), log2=(6, 6))    # 64x64 grid pinned far below support
    assert mv.deficit > 1e-5
    assert 'tail deficit' in mv._explain_oneline()


def test_mv_bs_window_df_and_tail_df_per_axis():
    """Both companion frames carry one row per axis with the right columns."""
    mv = _mv()
    bw = mv.bs_window_df
    assert list(bw.index) == ['A', 'B']
    assert {'kind', 'bs', 'log2', 'x_min', 'x_max', 'clipped'}.issubset(bw.columns)
    td = mv.tail_df
    assert list(td.index) == ['A', 'B']
    assert {'support_min', 'support_max', 'mean', 'sd', 'skew',
            'right_heavy'}.issubset(td.columns)


def test_mv_plot_two_panels():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mv = _mv()
    fig, axs = plt.subplots(1, 2)
    out = mv.plot(axs=axs)
    assert out is None                      # returns None (no double-render)
    assert axs.flat[0].get_title() == 'severity'
    assert axs.flat[1].get_title() == 'aggregate'
    assert mv.figure is fig
    plt.close(fig)
    # also works with no axs supplied: figure stored on self.figure
    mv.plot()
    assert len(mv.figure.axes) == 2
    plt.close('all')


def test_mv_help_runs(capsys):
    mv = _mv()
    mv.help('corr')   # should not raise (defaults lod='short', values='short')
    # every lod x values combination runs (fmt='text' to avoid IPython display)
    for lod in ('terse', 'short', 'all'):
        for values in ('none', 'short', 'all'):
            mv.help('corr', lod=lod, values=values, fmt='text')
    # bad options raise ValueError
    with pytest.raises(ValueError):
        mv.help('corr', lod='medium')
    with pytest.raises(ValueError):
        mv.help('corr', values='lots')


def test_mv_no_copula_defaults_independent():
    # copula clause omitted -> independence copula
    prog = '''bivariate MV 25 claims
        agg A dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
        agg B dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
        poisson'''
    mv = build(prog)
    assert mv.copula.kind == 'independent'
    # baseline positive corr from the shared count only
    assert 0.0 < mv.corr < 0.4


def test_mv_copula_independent_no_param():
    # 'copula independent' with no parameter must parse
    prog = '''bivariate MV 25 claims
        agg A dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
        agg B dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
        copula independent'''
    mv = build(prog)
    assert mv.copula.kind == 'independent'


# ----------------------------------------------------------------------
# netceded mode: joint (ceded, net) of one reinsured aggregate
# ----------------------------------------------------------------------

NC_PROG = ('agg NC 8 claims sev 300 * beta 2 3 '
           'occurrence net of 0.7 so 60 xs 40 poisson')


def test_netceded_via_decl():
    from aggregate.bivariate import BivariateAggregate
    mv = build(f'netceded {NC_PROG}')
    assert isinstance(mv, BivariateAggregate)
    assert mv.mode == 'netceded'
    assert mv.unit_names == ['Net', 'Ceded']    # x=net, y=ceded convention
    nd, cd = mv.marginals
    assert np.isclose(nd.sum(), 1.0, atol=1e-6)
    assert np.isclose(cd.sum(), 1.0, atol=1e-6)
    assert mv.corr > 0


@pytest.mark.parametrize('kw,views', [
    ('netceded', ('Net', 'Ceded')),
    ('grossceded', ('Gross', 'Ceded')),
    ('grossnet', ('Gross', 'Net')),
])
def test_view_pair_decl_builds_and_labels(kw, views):
    """Each view-pair prefix parses, builds, and labels its axes x-then-y."""
    from aggregate.bivariate import BivariateAggregate
    mv = build(f'{kw} {NC_PROG}')
    assert isinstance(mv, BivariateAggregate)
    assert mv.mode == 'netceded'
    assert mv.unit_names == list(views)
    m0, m1 = mv.marginals
    assert np.isclose(m0.sum(), 1.0, atol=1e-6)
    assert np.isclose(m1.sum(), 1.0, atol=1e-6)
    assert mv.deficit < 1e-6


@pytest.mark.parametrize('kw,views', [
    ('netceded', ('net', 'ceded')),
    ('grossceded', ('gross', 'ceded')),
    ('grossnet', ('gross', 'net')),
])
def test_view_pair_decl_matches_occ_bivariate(kw, views):
    """The DecL prefix and occ_bivariate(views=...) agree on the joint corr."""
    decl = build(f'{kw} {NC_PROG}')
    method = build(NC_PROG, bs=1, log2=16).occ_bivariate(views=views)
    assert method.corr == pytest.approx(decl.corr, abs=0.02)


@pytest.mark.parametrize('kw', ['netceded', 'grossceded', 'grossnet'])
def test_view_pair_decl_roundtrips_through_unparser(kw):
    """Each view-pair prefix round-trips through the DecL unparser."""
    from aggregate import Underwriter
    from aggregate.decl_writer import spec_to_decl

    uw = Underwriter()
    kind, name, spec = uw.parser.parse(f'{kw} {NC_PROG}')
    assert kind == 'bvagg'
    assert tuple(spec['nc_views']) == {
        'netceded': ('net', 'ceded'),
        'grossceded': ('gross', 'ceded'),
        'grossnet': ('gross', 'net'),
    }[kw]
    text = spec_to_decl(spec, kind, name)
    assert text.startswith(kw + ' ')
    # idempotent: re-parse + re-render is a fixed point
    kind2, name2, spec2 = uw.parser.parse(text)
    assert spec_to_decl(spec2, kind2, name2) == text


def test_netceded_via_occ_bivariate_matches_decl():
    a = build(NC_PROG, bs=1, log2=16)
    mv_method = a.occ_bivariate()
    mv_decl = build(f'netceded {NC_PROG}')
    # both are netceded BivariateAggregates; corr in the same ballpark
    assert mv_method.mode == mv_decl.mode == 'netceded'
    assert mv_method.corr == pytest.approx(mv_decl.corr, abs=0.02)


def test_netceded_additivity_mean():
    # E[Ceded] + E[Net] == E[gross aggregate]
    a = build(NC_PROG, bs=1, log2=16)
    mv = a.occ_bivariate()
    cd, nd = mv.marginals
    e_c = float((cd * mv.axis_xs[0]).sum())
    e_n = float((nd * mv.axis_xs[1]).sum())
    gross = a.reins_stats_df.loc[('agg', 'mean'), ('occ', 'Gross')]
    assert e_c + e_n == pytest.approx(gross, rel=2e-3)


def test_netceded_requires_occ_reins():
    a = build('agg G 10 claims sev lognorm 50 cv 1.5 poisson')
    with pytest.raises(ValueError, match='occurrence reinsurance'):
        a.occ_bivariate()


def test_netceded_reporting_and_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mv = build(f'netceded {NC_PROG}')
    df = mv.summary_df
    assert {('Ceded', 'Agg'), ('Net', 'Agg'),
            ('total', 'Agg')}.issubset(set(df.index))
    dep = mv.dependency_df
    assert np.isnan(dep.loc['Sev', 'tau'])           # no copula in netceded
    assert 'netceded' in mv.info
    mv.plot()
    assert mv.figure.axes[0].get_title() == 'severity'
    plt.close('all')


# ----------------------------------------------------------------------
# MV-2: measure-don't-guess axis sizing + the signed-severity 2-D compound
# ----------------------------------------------------------------------

# The motivating bug (author, 2026-06-18): a signed ``ssev`` book whose grid was
# sized for the single-event severity, not the 200-event marginal, so axis B's
# negative tail wrapped -- a 54% deficit. See dev/plan-mv.md §0, §5.
BUG_PROG = '''bivariate DISCRETE.2
    200 claims
    agg A dfreq[1] ssev uniform - .3
    agg B dfreq[1] ssev uniform - .5
    mixed gamma .5'''


def test_mv_signed_bug_book_deficit_clean():
    """The 54%-deficit signed book is now clean (DoD): deficit < 1e-6."""
    mv = build(BUG_PROG)
    assert mv.deficit < 1e-6, f'deficit {mv.deficit} not clean'
    m0, m1 = mv.marginals
    assert m0.sum() == pytest.approx(1.0, abs=1e-6)
    assert m1.sum() == pytest.approx(1.0, abs=1e-6)


def test_mv_signed_axis_window_straddles_zero():
    """The signed (mean-0) axis B gets a centred two-sided window, not a 0-based one.

    The root-cause fix: a signed marginal is sized two-sided and compounded with
    the 1-D ``i0``/``j0`` wrap-and-roll, so its negative half no longer wraps.
    """
    mv = build(BUG_PROG)
    xb = mv.axis_xs[1]
    assert xb[0] < 0 < xb[-1], f'axis B window [{xb[0]}, {xb[-1]}] does not straddle 0'
    m1 = mv.marginals[1]
    below = float(m1[xb < 0].sum())
    assert below > 0.4, f'expected ~half the mass below 0, got {below}'


def test_mv_signed_marginal_means_match_standalone():
    """Each marginal mean reproduces its standalone aggregate (means exact)."""
    mv = build(BUG_PROG)
    m0, m1 = mv.marginals
    for i, m in enumerate((m0, m1)):
        emp = float((m * mv.axis_xs[i]).sum())
        theory = mv._marg_theory[i][0]
        assert emp == pytest.approx(theory, abs=0.05), f'axis {i} mean {emp} vs {theory}'


def test_mv_signed_marginal_sd_converges_with_budget():
    """The marginal sd is resolution-limited: it converges to the standalone as
    the budget grows (the measured window is honest; only ``bs`` coarsens)."""
    mv20 = build(BUG_PROG)
    mv24 = build(BUG_PROG)
    mv24.update(log2=24)
    sd_theory = mv20._marg_theory[1][1]

    def sd(mv):
        m, x = mv.marginals[1], mv.axis_xs[1]
        mean = (m * x).sum()
        return float(np.sqrt((m * x * x).sum() - mean ** 2))

    err20, err24 = abs(sd(mv20) - sd_theory), abs(sd(mv24) - sd_theory)
    assert err24 < err20, f'sd error did not shrink with budget: {err20} -> {err24}'
    assert err24 / sd_theory < 0.01    # within 1% at log2=24


def test_mv_budget_honoured_and_overridable():
    """``update(log2=B)`` is the TOTAL cell budget; the split falls out of support."""
    mv = build(BUG_PROG)
    for B in (18, 20, 22):
        mv.update(log2=B)
        n0, n1 = mv.density.shape
        assert np.log2(n0) + np.log2(n1) <= B + 1e-9, f'budget {B} exceeded: {n0}x{n1}'
        assert mv.deficit < 1e-5


def test_mv_bs_override_applies_to_both_axes():
    """An explicit ``bs`` is used verbatim on both axes (measured log2 split)."""
    mv = build(BUG_PROG)
    mv.update(bs=0.5)
    assert mv.bs[0] == 0.5 and mv.bs[1] == 0.5


# A non-negative book whose mass lives far from 0 (low CV): 500-claim compound
# of 10*uniform (mean 2500, sd ~129) and 20*uniform (mean 5000, sd ~258). The
# window must NOT be pinned to a 0-based grid -- it must focus on the mass.
FAR_FROM_ZERO_PROG = '''bivariate MV 500 claims
    agg AL 1 claim  sev 10 * uniform fixed
    agg GL 1 claim  sev 20 * uniform fixed
    poisson'''


def test_mv_far_from_zero_window_not_pinned_to_origin():
    """A non-negative low-CV axis windows on its mass, not from 0 (no artificial pin)."""
    mv = build(FAR_FROM_ZERO_PROG)
    assert mv.deficit < 1e-6
    for i, mean in enumerate((2500.0, 5000.0)):
        x = mv.axis_xs[i]
        # origin sits well above 0 (mass is ~20 sd from 0), not pinned to 0
        assert x[0] > 0.5 * mean, f'axis {i} origin {x[0]} should be near the mass, not 0'
        # the window brackets the mean
        assert x[0] < mean < x[-1]
        # marginal mean reproduces the standalone
        m = mv.marginals[i]
        assert float((m * x).sum()) == pytest.approx(mean, rel=1e-3)


def test_mv_symmetric_axis_window_centered():
    """A symmetric (mean-0) axis gets a grid centred on its mass.

    Axis B (``ssev uniform`` symmetric) used to be measured against the 1-D
    loss-convention grid (deep upper, trimmed/clipped lower) and placed with all
    the power-of-two slack above 0 -- so a symmetric axis came out badly
    off-centre (e.g. ``[-29, +99]``). The measurement now recentres a signed
    marginal, the depth backs off the FFT noise floor, and the window is centred
    in the grid (slack split). The grid centre should sit near 0.
    """
    mv = build(BUG_PROG)
    x = mv.axis_xs[1]
    center = 0.5 * (x[0] + x[-1])
    span = x[-1] - x[0]
    assert abs(center) < 0.1 * span, f'axis B grid not centred: [{x[0]}, {x[-1]}]'
    # mass is symmetric about 0 and well inside the grid
    m = mv.marginals[1]
    mean = float((m * x).sum())
    assert abs(mean) < 1e-3
    assert mv.deficit < 1e-6


def test_mv_per_axis_log2_tuple():
    """``log2=(x, y)`` pins the per-axis split; the budget is their sum."""
    mv = build(BUG_PROG, log2=(9, 11))
    assert mv.density.shape == (1 << 9, 1 << 11)
    # the auto split for this book is (11, 9); the tuple overrides it
    auto = build(BUG_PROG)
    assert auto.density.shape == (1 << 11, 1 << 9)
    assert mv.deficit < 1e-6


def test_mv_per_axis_bs_tuple():
    """``bs=(x, y)`` pins the per-axis bucket size verbatim."""
    mv = build(BUG_PROG, bs=(0.5, 0.0625))
    assert mv.bs[0] == 0.5 and mv.bs[1] == 0.0625
    assert mv.deficit < 1e-5


def test_mv_split_changes_marginal_resolution():
    """Giving the hard signed axis more log2 improves its sd (equal budget)."""
    sd_theory = build(BUG_PROG)._marg_theory[1][1]

    def sd_err(split):
        mv = build(BUG_PROG, log2=split)
        m, x = mv.marginals[1], mv.axis_xs[1]
        mean = (m * x).sum()
        return abs(float(np.sqrt((m * x * x).sum() - mean ** 2)) - sd_theory)

    # axis B (mean-0 signed) is resolution-starved; (9, 11) gives it the finer
    # grid than (11, 9) -- both 2**20 cells total.
    assert sd_err((9, 11)) < sd_err((11, 9))


def test_mv_far_from_zero_resolution_beats_zero_based():
    """Focusing on the mass buys finer resolution than a 0-based grid at equal budget.

    Pinning to 0 would force ``bs ~ hi / 2**log2``; focusing on the measured
    window gives ``bs ~ width / 2**log2`` -- here roughly half as coarse.
    """
    mv = build(FAR_FROM_ZERO_PROG)
    # 10*uniform: mass in ~[1700, 3450]. A 0-based grid at log2=10 would need
    # bs ~ 3450/1024 ~ 3.4; the focused grid fits the ~1733-wide window instead.
    assert mv.bs[0] <= 3.0


def test_mv_wrong_component_count_raises():
    prog = '''bivariate Bad 25 claims
        agg A dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
        copula gumbel 0.4
        poisson'''
    # one component -> bv_body has a single child, BivariateAggregate rejects
    with pytest.raises(Exception):
        build(prog)


def test_mv_default_freq_is_poisson():
    # the trailing freq line is optional (defaults to poisson)
    prog = '''bivariate MV 25 claims
        agg A dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
        agg B dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
        copula gumbel 0.4'''
    mv = build(prog)
    assert mv.freq_name == 'poisson'
    assert np.isclose(mv.density.sum(), 1.0, atol=1e-6)


# ----------------------------------------------------------------------
# MV-6a: shuffle-of-Min copula (programmatic, App. A)
# ----------------------------------------------------------------------

from aggregate.copula import ShuffleOfMin, CopulaShuffle  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402


def test_shuffle_tau_limits():
    """n=1 recovers M (tau=1) / W (tau=-1); identity perm is comonotone."""
    assert CopulaShuffle(perm=[0]).tau() == pytest.approx(1.0)
    assert CopulaShuffle(perm=[0], flip=[True]).tau() == pytest.approx(-1.0)
    assert CopulaShuffle(perm=[0, 1, 2, 3]).tau() == pytest.approx(1.0)
    assert CopulaShuffle(perm=[3, 2, 1, 0]).tau() == pytest.approx(-0.5)


@pytest.mark.parametrize('perm,flip', [
    ([2, 0, 3, 1], [False, True, False, True]),
    ([0, 1, 2, 3], None),
    ([3, 2, 1, 0], None),
    ([1, 0], None),
])
def test_shuffle_analytic_tau_matches_empirical(perm, flip):
    """The exact perm/flip tau formula matches the sampled Kendall tau."""
    cop = CopulaShuffle(perm=perm, flip=flip)
    xy = cop.sample(200_000, rng=0)
    assert cop.tau() == pytest.approx(kendalltau(xy[:, 0], xy[:, 1]).statistic, abs=0.01)


def test_shuffle_boundary_and_marginals():
    """The base-class boundary conditions hold and rectangle_pmf reproduces marginals."""
    cop = CopulaShuffle(perm=[2, 0, 3, 1], flip=[False, True, False, True])
    assert float(cop.C(0.0, 0.7)) == pytest.approx(0.0)
    assert float(cop.C(1.0, 0.7)) == pytest.approx(0.7)
    assert float(cop.C(0.7, 1.0)) == pytest.approx(0.7)
    g1 = np.cumsum([0.1, 0.2, 0.3, 0.4])
    g2 = np.cumsum([0.25, 0.25, 0.25, 0.25])
    S = cop.rectangle_pmf(g1, g2)
    assert np.allclose(S.sum(axis=1), [0.1, 0.2, 0.3, 0.4])
    assert np.allclose(S.sum(axis=0), 0.25)
    assert S.sum() == pytest.approx(1.0)


def test_shuffle_perm_validation():
    with pytest.raises(ValueError, match='permutation'):
        ShuffleOfMin(perm=[0, 0, 1])
    with pytest.raises(ValueError, match='programmatic-only'):
        CopulaShuffle()


def test_shuffle_plugs_into_bivariate_and_reproduces():
    """A shuffle copula swapped onto a built bivariate reproduces the marginals."""
    mv = build('''bivariate Shuf 25 claims
        agg A dfreq [0 1] [.4 .6] sev lognorm 40 cv 1.2
        agg B dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
        poisson''')
    mv.copula = CopulaShuffle(perm=[3, 2, 1, 0])
    mv.update()
    assert isinstance(mv.copula, CopulaShuffle)
    df = mv.summary_df
    assert all(abs(float(df.loc[(n, 'Agg'), 'Err EX'])) < 0.10
               for n in mv.unit_names)
    assert mv.deficit < 1e-6
    # the reverse-strip shuffle (tau=-0.5) offsets the shared-count coupling
    assert str(mv.copula) == 'shuffle(n=4)'


# ----------------------------------------------------------------------
# MV-6b: clash statement (independent-trigger shared-event model, App. B)
# ----------------------------------------------------------------------

from aggregate.bivariate import solve_clash_model  # noqa: E402

CLASH_PROG = ('clash Cat 8 5 2 claims sev lognorm 50 cv 1.2 '
              'sev lognorm 60 cv 1.5 poisson')


def test_solve_clash_model_identities():
    """The solver satisfies the independent-trigger 2x2 table identities."""
    sol = solve_clash_model(30, 20, 5)
    assert sol.n0 == pytest.approx(30 * 20 / 5)            # nc*n0 == na*nb
    assert sol.n == pytest.approx(30 + 20 + 5 + sol.n0)
    assert sol.pa == pytest.approx((30 + 5) / sol.n)
    assert sol.pb == pytest.approx((20 + 5) / sol.n)


def test_solve_clash_model_guards():
    with pytest.raises(ValueError, match='nc'):
        solve_clash_model(10, 10, 0)
    with pytest.raises(ValueError):
        solve_clash_model(-1, 10, 2)


def test_clash_builds_and_derives_shared_count():
    """The clash statement builds a bivariate with the derived shared count n."""
    from aggregate.bivariate import BivariateAggregate
    mv = build(CLASH_PROG)
    assert isinstance(mv, BivariateAggregate)
    assert mv.mode == 'copula'
    assert mv.unit_names == ['Cat.A', 'Cat.B']
    sol = solve_clash_model(8, 5, 2)
    assert mv.en == pytest.approx(sol.n)
    assert mv.clash['pa'] == pytest.approx(sol.pa)
    assert mv.copula.kind == 'independent'


def test_clash_marginals_reproduce_standalone():
    """Each clash marginal reproduces its standalone aggregate (the invariant)."""
    mv = build(CLASH_PROG)
    df = mv.summary_df
    assert all(abs(float(df.loc[(n, 'Agg'), 'Err EX'])) < 0.10
               for n in mv.unit_names)
    assert mv.deficit < 1e-6
    assert mv._explain_oneline() == 'not unreasonable'


def test_clash_mixed_adds_common_shock():
    """A gamma-mixed shared frequency gives higher corr than plain poisson."""
    pois = build(CLASH_PROG)
    mix = build('clash Cat 8 5 2 claims sev lognorm 50 cv 1.2 '
                'sev lognorm 60 cv 1.5 mixed gamma 0.3')
    assert mix.corr > pois.corr > 0


def test_clash_roundtrips_through_unparser():
    """The clash statement round-trips through the DecL unparser."""
    from aggregate import Underwriter
    from aggregate.decl_writer import spec_to_decl
    uw = Underwriter()
    prog = ('clash Cat 8 5 2 claims 500 xs 0 sev lognorm 50 cv 1.2 '
            'sev lognorm 60 cv 1.5 mixed gamma 0.3')
    kind, name, spec = uw.parser.parse(prog)
    assert kind == 'bvagg' and 'clash' in spec
    text = spec_to_decl(spec, kind, name)
    assert text.startswith('clash Cat 8 5 2 claims ')
    kind2, name2, spec2 = uw.parser.parse(text)
    assert spec_to_decl(spec2, kind2, name2) == text
