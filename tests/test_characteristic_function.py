"""Tests for ``aggregate.characteristic_function`` ([Characteristic-Function-Engine]).

The oracles of ``dev/done/plan-characteristic-function-engine.md``: the kernel
against the compound engine's round discretization on a shared grid, the
light-tailed families against their scipy references, and the untilted stable
law against the 2026-10-28 measurements.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import scipy.stats as ss

from aggregate import build, qd
from aggregate.characteristic_function import (
    AggregateCHF, FamilyGamma, FamilyStable, chernoff_edge,
    invert_characteristic_function, make_family, window_edges)
from aggregate.stable import Stable


def _reference_masses(ref, xs, bs):
    """Bucket masses of the round discretization of a scipy law."""
    return ref.cdf(xs + bs / 2) - ref.cdf(xs - bs / 2)


# ----------------------------------------------------------------------
# the kernel
# ----------------------------------------------------------------------

def test_kernel_against_the_round_discretization_gamma():
    """``chfagg G gamma 4 25`` and ``agg G dfreq [1] sev 25 * gamma 4`` agree bucket for bucket."""
    a = build('agg G dfreq [1] sev 25 * gamma 4', log2=12, bs=1 / 4)
    c = AggregateCHF('G', 'gamma', shape=4, scale=25)
    c.update(log2=12, bs=1 / 4)
    assert c.x_min == 0 and c.bs == 0.25 and c.log2 == 12
    both = pd.concat([a.density_df['p_total'].rename('agg'),
                      c.density_df['p_total'].rename('chf')], axis=1, join='inner')
    assert len(both) > 4000
    assert np.max(np.abs(both['agg'] - both['chf'])) < 1e-10


def test_kernel_against_the_round_discretization_signed_normal():
    """A signed severity's grid and the ``norm`` family agree where the grids overlap."""
    a = build('agg N dfreq [1] ssev 30 * norm + 100', log2=12, bs=1 / 4)
    c = AggregateCHF('N', 'norm', mean=100, sd=30)
    c.update(log2=12, bs=1 / 4)
    both = pd.concat([a.density_df['p_total'].rename('agg'),
                      c.density_df['p_total'].rename('chf')], axis=1, join='inner')
    assert len(both) > 1000
    assert np.max(np.abs(both['agg'] - both['chf'])) < 1e-10


def test_kernel_against_the_round_discretization_inverse_gaussian():
    a = build('agg I dfreq [1] sev 400 * invgauss 0.25', log2=12, bs=1 / 4)
    c = AggregateCHF('I', 'invgauss', mean=100, shape=400)
    c.update(log2=12, bs=1 / 4)
    both = pd.concat([a.density_df['p_total'].rename('agg'),
                      c.density_df['p_total'].rename('chf')], axis=1, join='inner')
    assert np.max(np.abs(both['agg'] - both['chf'])) < 1e-10


def test_kernel_atom_is_split_out_and_added_back():
    """A point mass is removed from the characteristic function and put in its bucket."""
    ref = ss.gamma(4, scale=25)
    w = 0.3
    chf = lambda s: w + (1 - w) * (1 - 25j * s) ** -4.0
    bs, n = 0.25, 2 ** 12
    p = invert_characteristic_function(chf, 0.0, bs, n, atom=(0.0, w))
    xs = bs * np.arange(n)
    exact = (1 - w) * _reference_masses(ref, xs, bs)
    exact[0] += w
    assert np.max(np.abs(p - exact)) < 1e-10


def test_kernel_without_the_bucket_factor_returns_density_samples():
    ref = ss.norm(100, 30)
    bs, n = 1 / 8, 2 ** 12
    p = invert_characteristic_function(
        lambda s: np.exp(100j * s - 0.5 * 900 * s ** 2), -156.0, bs, n, bucket=False)
    xs = -156.0 + bs * np.arange(n)
    assert np.max(np.abs(p / bs - ref.pdf(xs))) < 1e-10


# ----------------------------------------------------------------------
# exact references
# ----------------------------------------------------------------------

@pytest.mark.parametrize('family, kw', [
    ('norm', dict(mean=100, sd=30)),
    ('norm', dict(mean=0, sd=1)),
    ('gamma', dict(shape=4, scale=25)),
    ('gamma', dict(shape=3, scale=25)),
    ('invgauss', dict(mean=100, shape=400)),
])
def test_light_tailed_families_against_scipy_at_the_default_grid(family, kw):
    c = AggregateCHF('X', family, **kw)
    c.update()
    exact = _reference_masses(c.family.reference, c.xs, c.bs)
    assert np.max(np.abs(c.agg_density - exact)) < 1e-9
    assert abs(c.total_mass - 1) < 1e-9
    assert c.validation_description == 'not unreasonable'
    assert abs(c.est_m - c.agg_m) < 1e-6 * max(1.0, abs(c.agg_m))


def test_cv_forms_are_the_same_laws():
    assert AggregateCHF('X', 'gamma', form='cv', mean=100, cv=0.5).family.to_decl() \
        == 'gamma 100 cv 0.5'
    g = make_family('gamma', 'cv', mean=100, cv=0.5)
    assert (g.shape, g.theta) == (4.0, 25.0)
    n = make_family('norm', 'cv', mean=100, cv=0.3)
    assert n.sd == 30
    i = make_family('invgauss', 'cv', mean=100, cv=0.5)
    assert i.lam == 400


def test_gamma_shape_one_is_truncation_limited_and_flagged():
    """A jump in the density makes |phi| decay as 1/t; the truncation row says so."""
    c = AggregateCHF('E', 'gamma', shape=1, scale=25)
    c.update()
    assert c.truncation_error > 1e-6
    assert c.validation_df.loc['truncation error', 'status'] == 'flag'
    assert 'truncation error' in c.validation_description
    exact = _reference_masses(c.family.reference, c.xs, c.bs)
    assert np.max(np.abs(c.agg_density - exact)) < 1e-4


# ----------------------------------------------------------------------
# window edges
# ----------------------------------------------------------------------

def test_chernoff_edge_of_the_normal_law_is_the_closed_form():
    mu, sd, eps = 100.0, 30.0, 5e-6
    fam = make_family('norm', mean=mu, sd=sd)
    x = chernoff_edge(fam.cgf, np.inf, np.log(eps), sd)
    assert x == pytest.approx(mu + sd * np.sqrt(-2 * np.log(eps)), rel=1e-9)


@pytest.mark.parametrize('family, kw', [
    ('norm', dict(mean=100, sd=30)),
    ('gamma', dict(shape=4, scale=25)),
    ('invgauss', dict(mean=100, shape=400)),
])
def test_window_edges_are_conservative(family, kw):
    fam = make_family(family, **kw)
    eps = 5e-6
    x_l, x_r = window_edges(fam, eps)
    ref = fam.reference
    assert ref.sf(x_r) <= eps
    assert ref.cdf(x_l) <= eps
    assert x_l >= fam.support[0]


def test_stable_power_edge_and_support_bound():
    fam = FamilyStable(dict(alpha=1.5, beta=1, scale=2, loc=3))
    x_l, x_r = window_edges(fam, 5e-6)
    assert x_r == pytest.approx(3 + (fam.stable.tail_constant / 5e-6) ** (1 / 1.5))
    assert x_l < 3
    one_sided = FamilyStable(dict(alpha=0.7, beta=1, scale=1, loc=3))
    assert one_sided.support == (3, np.inf)
    assert window_edges(one_sided, 5e-6)[0] == 3


def test_stable_laplace_transform_matches_the_cgf_on_the_light_side():
    """``K(-s)`` for ``beta = 1`` is the Laplace transform; ``beta = -1`` is its reflection."""
    s = np.array([0.1, 1.0, 3.0])
    for alpha in (0.7, 1.0, 1.5):
        right = FamilyStable(dict(alpha=alpha, beta=1, scale=2, loc=1))
        left = FamilyStable(dict(alpha=alpha, beta=-1, scale=2, loc=-1))
        assert np.allclose(right.cgf(-s), left.cgf(s))
        assert np.all(np.isnan(right.cgf(s)))
        assert np.all(np.isnan(left.cgf(-s)))
    # alpha = 2 is the normal law with variance 2 gamma^2
    nrm = FamilyStable(dict(alpha=2, beta=0, scale=2, loc=1))
    assert np.allclose(nrm.cgf(s), s + 4 * s ** 2)
    assert nrm.power_tails is None and nrm.tilt_side is None


# ----------------------------------------------------------------------
# the stable law, untilted
# ----------------------------------------------------------------------

@pytest.mark.parametrize('log2, bs, errors', [
    (16, 0.05, (8e-6, 4e-4, 0.08)),
    (18, 0.02, (9e-7, 1e-4, 0.03)),
    (20, 0.01, (3e-7, 2e-5, 0.005)),
])
def test_untilted_stable_reproduces_the_measured_table(log2, bs, errors):
    """The 2026-10-28 table: alpha 1.5, beta 1, unit scale, x0 = -20, within a factor of 2."""
    st = Stable(1.5, 1)
    c = AggregateCHF.from_chf('S', st.chf, x_min=-20, bs=bs, log2=log2)
    c.update()
    assert c.x_min == -20 and c.bs == bs and c.log2 == log2
    assert abs(c.total_mass - 1) < 1e-12
    for x, bound in zip((10, 100), errors[:2]):
        ref = st.sf(x + bs / 2)
        assert abs(c.sf(x) - ref) / ref < 2 * bound
    ref = st.tail_constant * 1000.0 ** -1.5
    assert abs(c.sf(1000) - ref) / ref < 2 * errors[2]


def test_rule_a_sizes_an_untilted_stable_window_from_the_power_tail():
    c = AggregateCHF('S', 'stable', alpha=1.5, beta=1)
    c.update(tilt=False)
    assert c.tilt == (0.0, 0.0) and c.outside_mass == 0.0
    fam = c.family
    assert c.x_max >= (fam.stable.tail_constant / 5e-6) ** (1 / 1.5)
    assert c.x_min < 0 and c.x_min % c.bs == 0
    assert abs(c.total_mass - 1) < 1e-9
    assert np.all(c.agg_density > -1e-12)
    assert c.agg_m == 0 and not np.isfinite(c.agg_cv)
    assert c.stats_df.loc['cv', 'theoretical'] != c.stats_df.loc['cv', 'theoretical']  # nan


def test_from_family_accepts_a_stable_object():
    c = AggregateCHF.from_family('S', Stable(1.5, 0.5, 2, 1), note='x')
    assert isinstance(c.family, FamilyStable)
    assert c.family.tilt_side == 'split'
    assert c.family.to_decl() == 'stable 1.5 0.5 2 1'
    assert c.note == 'x'
    with pytest.raises(TypeError):
        AggregateCHF.from_family('S', object())


# ----------------------------------------------------------------------
# the class surface
# ----------------------------------------------------------------------

@pytest.fixture(scope='module')
def gamma_chf():
    c = AggregateCHF('G', 'gamma', shape=4, scale=25, note='a gamma', tags=('t',))
    c.update(log2=14)
    return c


def test_accessors_delegate_to_the_grid_distribution(gamma_chf):
    c = gamma_chf
    x = c.xs[1000]
    assert c.sf(x) == pytest.approx(1 - c.cdf(x))
    assert c.cdf(x) == pytest.approx(c.density_df.loc[x, 'F'])
    assert c.q(c.cdf(x)) == x
    assert c.pmf(x) == c.agg_density[1000]
    assert c.pdf(x) == pytest.approx(c.agg_density[1000] / c.bs)
    assert c.snap(x + 0.3 * c.bs) == x
    assert c.var_dict(0.99) == {'G': c.q(0.99)}
    assert c.tvar(0.99) >= c.q(0.99)
    p, L, a = c.prob_loss_assets(p=0.99)
    assert a == c.q(0.99)
    assert c.bounded is False and c.support == (0.0, np.inf)
    assert c._is_loss_value is True


def test_frames_carry_the_aggregate_columns(gamma_chf):
    c = gamma_chf
    assert list(c.density_df.columns) == ['loss', 'p_total', 'p', 'F', 'S', 'lev', 'exa']
    assert c.density_df.index.name == 'loss'
    a = build('agg G dfreq [1] sev 25 * gamma 4', log2=14)
    assert list(c.summary_df.columns) == list(a.summary_df.columns)
    assert list(c.summary_df.index) == ['Agg']
    assert list(c.tail_df.columns) == list(a.tail_df.columns)
    assert list(c.tail_df.index) == list(a.tail_df.index)
    assert list(c.stats_df.index) == ['mean', 'sd', 'cv', 'skew']
    assert set(c.validation_df['status']) <= {'ok', 'flag'}
    assert c.validation_df.loc['mean rel error', 'status'] == 'ok'
    assert c.validation_description == 'not unreasonable'
    assert 'ok' in c.validation_explanation
    assert c.summary_df.loc['Agg', 'Mean'] == pytest.approx(100)


def test_info_repr_and_html(gamma_chf):
    c = gamma_chf
    info = c.info
    assert info.startswith('chfagg object name')
    assert 'gamma 4 25' in info
    assert "AggregateCHF('G': 'gamma 4 25', log2 14" in repr(c)
    html = c._repr_html_()
    assert '<h3>AggregateCHF object: G</h3>' in html and '<table' in html
    assert 'Validation: not unreasonable.' in c._text_info_blob()
    assert c.note == 'a gamma' and c.tags == ('t',) and c.hints == ''
    assert c.label == 'G' and c.program == ''


def test_qd_prints_the_headline(gamma_chf, capsys):
    qd(gamma_chf)
    out = capsys.readouterr().out
    assert 'AggregateCHF object: G' in out
    assert 'Mean' in out


def test_plot_returns_a_figure(gamma_chf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = gamma_chf.plot()
    assert fig is gamma_chf.figure
    plt.close(fig)
    s = AggregateCHF('S', 'stable', alpha=1.5, beta=1)
    s.update(log2=12)
    plt.close(s.plot())


def test_before_update_the_frames_say_so():
    c = AggregateCHF('G', 'gamma', shape=4, scale=25)
    assert c.tail_df is None
    assert c.validation_description == 'not updated'
    assert np.isnan(c.summary_df.loc['Agg', 'P99'])
    assert c.summary_df.loc['Agg', 'Mean'] == 100
    assert 'not updated' in repr(c)
    with pytest.raises(ValueError, match='update'):
        c.q(0.5)
    with pytest.raises(ValueError, match='update'):
        c.density_df


def test_rule_c_uses_the_given_grid_and_keeps_zero_on_it():
    c = AggregateCHF('N', 'norm', mean=100, sd=30)
    c.update(log2=12, bs=1 / 8)
    assert c.bs == 1 / 8 and c.log2 == 12 and len(c.xs) == 4096
    assert c.x_min % c.bs == 0
    bs, log2, x0, tilt = c.grid_window(12, bs=1 / 8)
    assert (bs, log2, x0, tilt) == (1 / 8, 12, c.x_min, (0.0, 0.0))


def test_recommended_grid_resolves_the_body():
    c = AggregateCHF('N', 'norm', mean=100, sd=30)
    log2, bs = c.recommended_grid()
    assert 30 / bs >= 10
    assert log2 < 16
    c.update(log2=log2, bs=bs)
    assert c.validation_df.loc['buckets per scale', 'status'] == 'ok'
    coarse = AggregateCHF('N', 'norm', mean=100, sd=30)
    coarse.update(log2=6)
    assert coarse.validation_df.loc['buckets per scale', 'status'] == 'flag'


def test_update_ignores_other_hint_keys_with_a_warning(caplog):
    c = AggregateCHF('G', 'gamma', shape=4, scale=25)
    with caplog.at_level(logging.WARNING, logger='aggregate.characteristic_function'):
        c.update(log2=10, padding=1)
    assert 'padding' in caplog.text


def test_refusals():
    with pytest.raises(ValueError, match='unknown family'):
        AggregateCHF('X', 'lognorm', mean=1, cv=1)
    with pytest.raises(ValueError, match='no mean cv form'):
        AggregateCHF('X', 'stable', form='cv', mean=1, cv=1)
    with pytest.raises(ValueError, match='positive'):
        FamilyGamma(dict(shape=-1, scale=1))
    with pytest.raises(ValueError, match='bucket_sizing_p'):
        AggregateCHF('G', 'gamma', shape=4, scale=25).update(bucket_sizing_p=1.5)
    from aggregate.characteristic_function import FamilyUser
    with pytest.raises(ValueError, match='give bs and log2'):
        AggregateCHF('U', FamilyUser(lambda t: np.exp(-t ** 2 / 2))).update()


def test_fcc_contract_lists_the_class():
    from aggregate.constants import FIRST_CLASS_CLASSES, FCC_CONTRACT_EXCEPTIONS
    assert 'AggregateCHF' in FIRST_CLASS_CLASSES
    assert FCC_CONTRACT_EXCEPTIONS == {}


# ----------------------------------------------------------------------
# [Characteristic-Function-Tilting]: the shifted contour, one-sided
# ----------------------------------------------------------------------

def test_round_bucket_down_is_the_ladder_rung_at_or_below():
    from aggregate.utilities import _round_bucket_down
    for bs, rung in [(0.1, 1 / 16), (0.288, 0.25), (0.0024, 1 / 512), (1, 1),
                     (3.4, 2), (5, 5), (57, 50), (99, 80), (0.5, 0.5)]:
        assert _round_bucket_down(bs) == rung, bs


@pytest.mark.parametrize('alpha', [0.7, 1.0, 1.5])
def test_analytic_stable_chf_on_the_real_line_and_at_i_theta(alpha):
    """The continuation equals ``Stable.chf`` on the real line and ``exp(Lambda(theta))`` at ``s = i theta``."""
    fam = FamilyStable(dict(alpha=alpha, beta=1, scale=2, loc=1))
    t = np.linspace(-6, 6, 25)
    assert np.max(np.abs(fam.chf(t + 0j) - fam.stable.chf(t))) < 1e-12
    # a strictly complex array takes the analytic branch
    s = t + 1e-300j
    assert np.max(np.abs(fam.chf(s) - fam.stable.chf(t))) < 1e-12
    for theta in (0.05, 0.3, 2.0):
        got = fam.chf(np.array([1j * theta]))[0]
        want = np.exp(fam.cgf(np.array([-theta]))[0])
        assert abs(got - want) < 1e-12 * abs(want)
    with pytest.raises(ValueError, match='beta = 1'):
        FamilyStable(dict(alpha=alpha, beta=0.5)).chf(np.array([1j]))


def _one_sided_check(alpha, log2, reference_log2_extra=6):
    """Tilted masses against the plan's reference: the untilted kernel on a window 64 times longer for alpha >= 1 (its own closed-form wrap allowed for), the exact series below."""
    c = AggregateCHF('S', 'stable', alpha=alpha, beta=1)
    c.update(log2=log2)
    n, h, x0 = 2 ** log2, c.bs, c.x_min
    fam = c.family
    assert c.tilt[0] == 0 and c.tilt[1] > 0
    assert c.tilt_theta[1] == pytest.approx(c.tilt[1] / (n * h))
    if alpha >= 1:
        ref = invert_characteristic_function(fam.chf, x0, h, 2 ** (log2 + reference_log2_extra))[:n]
        mid, edge = n // 2, n - 1
        p = c.agg_density
        # the reference wraps too, by its own closed-form rho at each point:
        # the power tail at x + m * (64 W) relative to the mass at x
        from scipy.special import zeta
        W_ref = 2 ** reference_log2_extra * n * h
        ref_wrap = [q ** (alpha + 1) * zeta(alpha + 1, 1 + q)
                    for q in ((c.xs[mid] - fam.location) / W_ref,
                              (c.xs[edge] - fam.location) / W_ref)]
    else:
        xs = c.xs
        c_scale = fam.stable.scale * np.cos(np.pi * alpha / 2) ** (-1 / alpha)
        sel = np.flatnonzero(xs - h / 2 > 2 * c_scale)
        hi = np.array([fam.tail_mass_above(x + h / 2) for x in xs[sel]])
        lo = np.array([fam.tail_mass_above(x - h / 2) for x in xs[sel]])
        ref = lo - hi
        p = c.agg_density[sel]
        mid, edge = int(np.searchsorted(xs[sel], x0 + n * h / 2)), len(sel) - 1
        ref_wrap = [0.0, 0.0]                     # the series is exact
    rel_mid = abs(p[mid] - ref[mid]) / ref[mid]
    rel_edge = abs(p[edge] - ref[edge]) / ref[edge]
    assert rel_mid < 1e-4 + ref_wrap[0] and rel_mid < c.edge_error + ref_wrap[0]
    assert rel_edge < 1e-4 + ref_wrap[1] and rel_edge < c.edge_error + ref_wrap[1]
    assert abs(c.total_mass - (1 - c.outside_above)) < 2e-7
    assert np.all(c.agg_density > 0)
    assert c.outside_below == 0.0
    return c


@pytest.mark.parametrize('alpha', [1.5, 1.1, 1.0, 0.7])
def test_one_sided_tilting_at_log2_14(alpha):
    _one_sided_check(alpha, 14)


@pytest.mark.xdist_group('chf_reference')
@pytest.mark.parametrize('alpha, log2', [(1.5, 16), (1.1, 16), (1.0, 16),
                                         (0.7, 16), (0.7, 20), (0.5, 20)])
def test_one_sided_tilting_reproduces_the_measured_table(alpha, log2):
    """The 2026-10-29 one-sided table; the alpha >= 1 references are 2 ** 22 buckets, one worker."""
    c = _one_sided_check(alpha, log2)
    cap = c._bucket_cap()
    assert c.bs <= cap
    if alpha < 1.2:
        assert c.bs == cap                                    # the cap binds
        assert c.validation_df.loc['outside mass', 'status'] == 'flag'


def test_levy_law_is_exact_at_alpha_one_half():
    """``stable 0.5 1`` is ``scipy.stats.levy``, so its masses are known exactly."""
    c = AggregateCHF('L', 'stable', alpha=0.5, beta=1, scale=3, loc=1)
    c.update(log2=16)
    lv = ss.levy(loc=1, scale=3)
    exact = lv.cdf(c.xs + c.bs / 2) - lv.cdf(c.xs - c.bs / 2)
    body = exact > 1e-6
    assert np.max(np.abs(c.agg_density[body] - exact[body]) / exact[body]) < 1e-4
    assert c.outside_above == pytest.approx(lv.sf(c.x_max + c.bs / 2), rel=1e-6)
    assert c.x_min == 1


def test_beta_minus_one_is_the_reflection():
    a = AggregateCHF('A', 'stable', alpha=1.5, beta=1, scale=2, loc=1)
    b = AggregateCHF('B', 'stable', alpha=1.5, beta=-1, scale=2, loc=-1)
    a.update(log2=14)
    b.update(log2=14)
    n, h = 2 ** 14, a.bs
    assert b.bs == h and b.tilt == (a.tilt[1], 0.0)
    assert b.x_max == pytest.approx(-a.x_min) and b.x_min == pytest.approx(-a.x_max)
    assert b.x_min % h == pytest.approx(0, abs=1e-9) or (b.x_min / h) == pytest.approx(round(b.x_min / h))
    # every bucket but the first mirrors; the first also holds outside_below
    assert np.max(np.abs(b.agg_density[1:] - a.agg_density[::-1][1:])) < 1e-12
    assert b.outside_below == a.outside_above and b.outside_above == 0
    assert b.agg_density[0] == pytest.approx(a.agg_density[-1] + b.outside_below)
    assert abs(b.total_mass - 1) < 2e-7
    # the cdf is right across the window: cdf_b(x_k) = sf_a(-x_k - h), up to
    # the residual between a's grid total and 1 - outside_above
    k = np.arange(10, n - 2, 997)
    x = b.xs[k]
    assert np.max(np.abs(b.cdf(x) - a.sf(-x - h))) < 1e-8
    assert b.agg_m == -1 and b.support == (-np.inf, np.inf)


def test_closure_under_addition_untilted():
    """``Stable(1.5, 1, 1) + Stable(1.5, 1, 2)`` equals the product of the two characteristic functions."""
    one, two = Stable(1.5, 1, 1), Stable(1.5, 1, 2)
    s = AggregateCHF.from_family('S', one + two)
    s.update(log2=12, bs=1 / 8, tilt=False)
    prod = AggregateCHF.from_chf('P', lambda t: one.chf(t) * two.chf(t),
                                 x_min=s.x_min, bs=1 / 8, log2=12)
    prod.update()
    assert np.max(np.abs(s.agg_density - prod.agg_density)) < 1e-12


def test_tilt_keyword_forms():
    c = AggregateCHF('S', 'stable', alpha=1.5, beta=1)
    c.update(log2=12, tilt=8)
    assert c.tilt == (0.0, 8.0) and c.tilt_setting == 8
    c.update(log2=12, tilt=1)
    assert c.tilt == (0.0, 1.0)
    c.update(log2=12, tilt=True)
    assert c.tilt[1] not in (0.0, 1.0) and c.tilt[1] > 5
    c.update(log2=12, tilt=False)
    assert c.tilt == (0.0, 0.0) and c.outside_mass == 0 and abs(c.total_mass - 1) < 1e-9
    assert c.edge_error > 0.1                       # the untilted wrap at the edge
    for bad in (-1, 'x', np.inf):
        with pytest.raises(ValueError, match='tilt must be'):
            c.update(log2=12, tilt=bad)
    g = AggregateCHF('G', 'gamma', shape=4, scale=25)
    g.update(log2=12, tilt=True)
    assert g.tilt == (0.0, 0.0) and np.isnan(g.edge_error)
    with pytest.raises(ValueError, match='no power tail'):
        g.update(log2=12, tilt=8)


def test_tilt_is_a_hint_key():
    from aggregate.underwriter import _HINT_KEYS, _parse_hints
    assert 'tilt' in _HINT_KEYS
    assert _parse_hints('tilt=False')['tilt'] is False
    assert _parse_hints('tilt=8')['tilt'] == 8
    assert _parse_hints('tilt=True')['tilt'] is True


def test_split_law_grid_window_places_the_window_by_the_crop():
    c = AggregateCHF('S', 'stable', alpha=1.5, beta=0.5)
    bs, log2, x_min, tilt = c.grid_window(12)
    assert x_min is None and tilt[0] > 0 and tilt[1] > 0
    c.update(log2=12)
    assert c.bs == bs and c.tilt == tilt and c.x_min % bs == 0


def test_tilted_validation_rows_and_info():
    c = AggregateCHF('S', 'stable', alpha=1.0, beta=1)
    c.update(log2=14)
    v = c.validation_df
    assert v.loc['total mass', 'target'] == pytest.approx(1 - c.outside_above)
    assert v.loc['total mass', 'status'] == 'ok'
    assert v.loc['outside mass', 'status'] == 'flag'
    assert 'hints{log2=' in v.loc['outside mass', 'note']
    assert v.loc['tilt', 'value'] == c.tilt[1]
    assert v.loc['edge error', 'status'] == 'ok'
    assert 'outside mass' in c.validation_description
    info = c.info
    assert 'tilt' in info and 'theta' in info and 'outside mass' in info
    log2, bs = c.recommended_grid()
    assert log2 > 14 and bs == c.bs
    assert sum(c._outside_at(log2, bs, c.grid_window(log2)[2])) <= 1e-5
    # the window's cdf is right up to the outside mass, and sf includes it
    assert c.sf(c.x_max) == pytest.approx(c.outside_above, rel=1e-4)


# ----------------------------------------------------------------------
# [Characteristic-Function-Two-Sided]: the split for |beta| < 1
# ----------------------------------------------------------------------

@pytest.mark.parametrize('alpha', [0.7, 1.0, 1.5])
@pytest.mark.parametrize('beta', [0, 0.5, 0.99])
def test_the_split_is_exact(alpha, beta):
    """``phi = phi_plus phi_minus`` on the real line to roundoff, for every alpha including 1."""
    fam = FamilyStable(dict(alpha=alpha, beta=beta, scale=2, loc=1))
    plus, minus = fam.halves()
    t = np.linspace(-6, 6, 41)
    assert np.max(np.abs(fam.chf(t) / (plus.chf(t) * minus.chf(t)) - 1)) < 1e-12
    assert plus.tilt_side == 'right' and minus.tilt_side == 'left'
    assert plus.location == 1 and minus.location == 0
    assert (plus.scale ** alpha + minus.scale ** alpha) == pytest.approx(2 ** alpha)
    with pytest.raises(ValueError, match='one-sided'):
        plus.halves()


TWO_SIDED_ROWS = [(1.5, 0), (1.5, 0.5), (1.5, 0.9), (1.5, 0.99), (1.5, -0.99),
                  (1.1, 0), (1.1, 0.5), (1.1, 0.99),
                  (1.0, 0), (1.0, 0.5), (1.0, 0.99),
                  (0.7, 0), (0.7, 0.5), (0.7, 0.99)]


def _two_sided_check(alpha, beta, log2, reference_log2_extra):
    """The plan's reference: the untilted kernel on a window ``2 ** extra`` times longer, centered on the kept one, allowed its own closed-form wrap at the edges."""
    from scipy.special import zeta
    c = AggregateCHF('S', 'stable', alpha=alpha, beta=beta)
    c.update(log2=log2)
    n, h, x0 = 2 ** log2, c.bs, c.x_min
    fam = c.family
    big = 2 ** (log2 + reference_log2_extra)
    off = (big - n) // 2
    ref = invert_characteristic_function(fam.chf, x0 - off * h, h, big)[off:off + n]
    p = c.agg_density.copy()
    p[0] -= c.outside_below
    rel = np.abs(p - ref) / ref
    worst = float(np.max(rel))
    # the reference's own wrap: each tail at a distance of m times its window,
    # relative to the mass at the kept window's edges
    A, B, _ = fam.power_tails
    W_ref = big * h
    wraps = []
    for const, dist in ((A, c.x_max - fam.location), (B, fam.location - x0)):
        q = dist / W_ref
        wraps.append(q ** (alpha + 1) * zeta(alpha + 1, 1 + q) if const > 0 else 0.0)
    ref_wrap = max(wraps)
    assert worst < c.edge_error + ref_wrap, (worst, c.edge_error, ref_wrap)
    if alpha >= 1:
        assert worst < 1e-4 + ref_wrap + 4 * c.edge_error
    assert np.all(c.agg_density > 0)
    assert c.outside_below > 0 and c.outside_above > 0
    assert 0.5 < c.outside_below / c.outside_above < 2
    assert abs(c.total_mass + c.outside_above - 1) < c.edge_error
    assert c.tilt[0] > 0 and c.tilt[1] > 0 and c.tilt_theta[1] == pytest.approx(c.tilt[1] / (n * h))
    return c


@pytest.mark.parametrize('alpha, beta', TWO_SIDED_ROWS)
def test_two_sided_tilting_at_log2_14(alpha, beta):
    c = _two_sided_check(alpha, beta, 14, 8)
    if (alpha, beta) == (1.5, 0.5):
        assert c.transfer_sigma == 0
    if (alpha, beta) == (1.0, 0.99):
        assert c.transfer_sigma / c.bs > 2


@pytest.mark.xdist_group('chf_reference')
@pytest.mark.parametrize('alpha, beta', TWO_SIDED_ROWS)
def test_two_sided_tilting_reproduces_the_measured_table(alpha, beta):
    """The 2026-10-29 two-sided table at log2 16, against a 64x reference."""
    c = _two_sided_check(alpha, beta, 16, 6)
    assert c.bs <= c._bucket_cap(two_sided=True)
    if alpha < 1.2:
        assert c.validation_df.loc['outside mass', 'status'] == 'flag'


def test_two_sided_untilted_wraps():
    c = AggregateCHF('S', 'stable', alpha=1.5, beta=0.5)
    c.update(log2=14, tilt=False)
    t = AggregateCHF('T', 'stable', alpha=1.5, beta=0.5)
    t.update(log2=14, bs=c.bs)
    both = pd.concat([c.density_df['p_total'].rename('wrap'),
                      t.density_df['p_total'].rename('tilt')], axis=1, join='inner')
    body = both['tilt'] > 1e-12
    rel = (both['wrap'] - both['tilt']).abs() / both['tilt']
    assert rel[body].max() > 0.1
    assert c.tilt == (0.0, 0.0) and c.outside_mass == 0


def test_two_sided_number_tilts_both_halves_and_info_shows_the_transfer():
    c = AggregateCHF('S', 'stable', alpha=1.0, beta=0.99)
    c.update(log2=12, tilt=6)
    assert c.tilt == (6.0, 6.0)
    assert c.transfer_sigma > 0
    assert 'transfer sigma' in c.info
    assert c.validation_df.loc['transfer sigma', 'value'] == c.transfer_sigma
    assert c.validation_df.loc['tilt', 'note'].startswith('e-folds')
