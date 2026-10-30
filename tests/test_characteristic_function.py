"""Tests for ``aggregate.characteristic_function`` ([Characteristic-Function-Engine]).

The oracles of ``dev/done/plan-characteristic-function-engine.md``: the kernel
against the compound engine's round discretization on a shared grid, the
light-tailed families against their scipy references, and the untilted stable
law against the 2026-10-28 measurements.
"""
from __future__ import annotations

import logging
from pathlib import Path

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
    """``chfagg G 25 * gamma 4`` and ``agg G dfreq [1] sev 25 * gamma 4`` agree bucket for bucket."""
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
    assert abs(c.est_m - c.actual_m) < 1e-6 * max(1.0, abs(c.actual_m))


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
    assert c._checks_frame().loc['truncation error', 'status'] == 'flag'
    assert 'characteristic function truncated at pi / bs' in c.validation_description
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
    assert c.tilt == (0.0, 0.0) and 0 < c.outside_mass < 1e-5   # wrapped in, reported
    fam = c.family
    assert c.x_max >= (fam.stable.tail_constant / 5e-6) ** (1 / 1.5)
    assert c.x_min < 0 and c.x_min % c.bs == 0
    assert abs(c.total_mass - 1) < 1e-9
    assert np.all(c.agg_density > -1e-12)
    assert c.actual_m == 0 and not np.isfinite(c.actual_cv)
    assert c.stats_df.loc[('agg', 'cv'), 'mixed'] != c.stats_df.loc[('agg', 'cv'), 'mixed']  # nan


def test_from_family_accepts_a_stable_object():
    c = AggregateCHF.from_family('S', Stable(1.5, 0.5, 2, 1), note='x')
    assert isinstance(c.family, FamilyStable)
    assert c.family.tilt_side == 'split'
    assert c.family.to_decl() == '2 * stable 1.5 0.5 + 1'
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
    assert list(c.summary_df.index) == ['Freq', 'Sev', 'Agg']
    assert list(c.tail_df.columns) == list(a.tail_df.columns)
    assert list(c.tail_df.index) == list(a.tail_df.index)
    assert c.stats_df.index.equals(a.stats_df.index)
    assert set(c._checks_frame()['status']) <= {'ok', 'flag'}
    assert c._checks_frame().loc['mean rel error', 'status'] == 'ok'
    assert c.validation_description == 'not unreasonable'
    assert 'not unreasonable' in c.validation_explanation
    assert c.summary_df.loc['Agg', 'Mean'] == pytest.approx(100)


def test_info_repr_and_html(gamma_chf):
    c = gamma_chf
    info = c.info
    assert info.startswith('aggregate object name')
    assert '25 * gamma 4' in info
    assert "AggregateCHF('G': '25 * gamma 4', log2 14" in repr(c)
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
    assert c.validation_description == 'n/a, not updated'
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
    assert c._checks_frame().loc['buckets per scale', 'status'] == 'ok'
    coarse = AggregateCHF('N', 'norm', mean=100, sd=30)
    coarse.update(log2=6)
    assert coarse._checks_frame().loc['buckets per scale', 'status'] == 'flag'


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
        assert c._checks_frame().loc['outside mass', 'status'] == 'flag'


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
    assert b.actual_m == -1 and b.support == (-np.inf, np.inf)


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
    assert c.tilt == (0.0, 0.0) and 0 < c.outside_mass < 1e-5 and abs(c.total_mass - 1) < 1e-9
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
    v = c._checks_frame()
    assert v.loc['total mass', 'target'] == pytest.approx(1 - c.outside_above)
    assert v.loc['total mass', 'status'] == 'ok'
    assert v.loc['outside mass', 'status'] == 'flag'
    assert 'hints{log2=' in v.loc['outside mass', 'note']
    assert v.loc['tilt', 'value'] == c.tilt[1]
    assert v.loc['edge error', 'status'] == 'ok'
    assert 'pmf deficit' in c.validation_description
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
        assert c._checks_frame().loc['outside mass', 'status'] == 'flag'


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
    assert c.tilt == (0.0, 0.0) and 0 < c.outside_mass < 1e-5


def test_two_sided_number_tilts_both_halves_and_info_shows_the_transfer():
    c = AggregateCHF('S', 'stable', alpha=1.0, beta=0.99)
    c.update(log2=12, tilt=6)
    assert c.tilt == (6.0, 6.0)
    assert c.transfer_sigma > 0
    assert 'transfer sigma' in c.info
    assert c._checks_frame().loc['transfer sigma', 'value'] == c.transfer_sigma
    assert c._checks_frame().loc['tilt', 'note'].startswith('e-folds')


# ----------------------------------------------------------------------
# [Characteristic-Function-Statement]: the chfagg declaration
# ----------------------------------------------------------------------

CHF_PROGRAMS = [
    'chfagg S 2 * stable 1.5 1',
    'chfagg S0 2 * stable 1.5 1 s0 - 2',
    'chfagg W stable 0.7 1 hints{tilt=False}',
    'chfagg T stable 1.5 0.5 hints{tilt=8}',
    'chfagg G 25 * gamma 4',
    'chfagg GC gamma 100 cv 0.5 hints{log2=14}',
    'chfagg N as "Signed" 30 * norm + 100 note{a note}',
    'chfagg I 400 * invgauss 0.25 tags{topic:numerics}',
    'chfagg IC invgauss 100 cv 0.5',
    'chfagg TW tweedie 1.5 100 2',
    'chfagg TWC tweedie 2.5 100 cv 0.5',
]


@pytest.fixture(scope='module')
def underwriter():
    from aggregate import Underwriter
    return Underwriter(update=False)


def _parse(uw, program):
    return uw.parser.parse(uw.lexer.tokenize(program))


@pytest.mark.parametrize('program', CHF_PROGRAMS)
def test_chfagg_parses_and_the_writer_round_trips(program, underwriter):
    """parse -> write -> parse reproduces the spec, and the writer is idempotent."""
    from aggregate.decl_writer import spec_to_decl
    kind, name, spec = _parse(underwriter, program)
    assert kind == 'chfagg' and spec['name'] == name
    assert 'family' in spec and 'form' in spec
    rendered = spec_to_decl(spec, kind, name)
    kind2, name2, spec2 = _parse(underwriter, rendered)
    assert (kind2, name2, spec2) == (kind, name, spec)
    assert spec_to_decl(spec2, kind2, name2) == rendered


def test_chfagg_spec_carries_the_parameters_as_written(underwriter):
    _, _, spec = _parse(underwriter, 'chfagg S0 2 * stable 1.5 1 s0 - 2')
    assert spec['parameterization'] == 's0' and spec['alpha'] == 1.5
    assert spec['chf_scale'] == 2 and spec['chf_loc'] == -2
    _, _, spec = _parse(underwriter, 'chfagg GC gamma 100 cv 0.5')
    assert spec['form'] == 'cv' and spec['mean'] == 100 and spec['cv'] == 0.5
    assert 'chf_scale' not in spec and 'chf_loc' not in spec
    _, _, spec = _parse(underwriter, 'chfagg N as "Signed" 30 * norm + 100')
    assert spec['label'] == 'Signed' and spec['chf_scale'] == 30 and spec['chf_loc'] == 100
    _, _, spec = _parse(underwriter, 'chfagg G 2 * gamma 100 cv 0.5 + 10')
    assert spec['form'] == 'cv' and spec['chf_scale'] == 2 and spec['chf_loc'] == 10


@pytest.mark.parametrize('program, match', [
    ('chfagg X lognorm 1 2', 'unknown family'),
    ('chfagg X stable 1.5', 'stable takes 2 shapes, alpha beta, got 1'),
    ('chfagg X stable 1.5 1 2 0 5', 'stable takes 2 shapes, alpha beta, got 5'),
    ('chfagg X stable 1.5 1 s2', 'must be s0 or s1'),
    ('chfagg X gamma 1 2 3', 'gamma takes 1 shape, shape, got 3'),
    ('chfagg X gamma', 'gamma takes 1 shape, shape, got 0'),
    ('chfagg X norm 100', 'norm takes no shape'),
    ('chfagg X -2 * gamma 4', 'must be positive'),
    ('chfagg X 0 * norm', 'must be positive'),
    ('chfagg X stable 1.5 cv 1', ''),
])
def test_chfagg_refusals_name_the_form(program, match, underwriter):
    with pytest.raises(ValueError, match=match):
        _parse(underwriter, program)


def test_build_chfagg_returns_an_updated_tilted_object():
    from aggregate.characteristic_function import AggregateCHF
    s = build('chfagg S 2 * stable 1.5 1')
    assert isinstance(s, AggregateCHF)
    assert s.agg_density is not None and s.tilt[1] > 5 and s.tilt[0] == 0
    assert s.program == 'chfagg S 2 * stable 1.5 1'
    assert s.pprogram.split() == s.program.split()
    assert not hasattr(s, 'pnl_program') and not hasattr(s, 'reins_program')
    assert hasattr(s, 'sharpen')                     # [CHF-Sharpen], 1.38.0
    t = build('chfagg T 2 * stable 1.5 0.5')
    assert t.tilt[0] > 0 and t.tilt[1] > 0


def test_build_chfagg_hints_reach_update():
    w = build('chfagg W stable 1.5 1 hints{tilt=False}')
    assert w.tilt == (0.0, 0.0) and 0 < w.outside_mass < 1e-5
    e = build('chfagg E stable 1.5 1 hints{tilt=8; log2=12; bs=1/32}')
    assert e.tilt == (0.0, 8.0) and e.log2 == 12 and e.bs == 1 / 32
    g = build('chfagg G 25 * gamma 4', log2=12, bs=1 / 4)
    assert (g.log2, g.bs) == (12, 0.25)
    n = build('chfagg N as "Signed" norm 100 cv 0.3 note{a note}')
    assert n.label == 'Signed' and n.note == 'a note' and n.x_min < 0
    assert build('chfagg G 25 * gamma 4', update=False).agg_density is None


def test_tilt_is_inert_on_an_agg():
    a = build('agg Inert 10 claims sev lognorm 10 cv 1 poisson hints{tilt=8}')
    assert a.est_m > 0


def test_chfagg_is_a_declaration_word():
    from aggregate.decl_words import decl_words
    assert 'chfagg' in decl_words()['declarations']


# ----------------------------------------------------------------------
# [Characteristic-Function-Families]: tweedie for every p, and three more laws
# ----------------------------------------------------------------------

def _against_reference(c, tol):
    ref = c.family.reference
    exact = _reference_masses(ref, c.xs, c.bs)
    assert np.max(np.abs(c.agg_density - exact)) < tol


@pytest.mark.parametrize('p, mean, dispersion', [(0, 100, 900), (2, 100, 0.25), (3, 100, 0.0025)])
def test_tweedie_normal_gamma_inverse_gaussian_members(p, mean, dispersion):
    c = build(f'chfagg T tweedie {p} {mean} {dispersion}')
    _against_reference(c, 1e-9)
    assert c.validation_description == 'not unreasonable'
    assert c.actual_m == mean and c.family.moments[1] == pytest.approx(dispersion * mean ** p)


def test_tweedie_compound_member_against_the_compound_engine_and_the_series():
    from aggregate.tweedie import Tweedie, tweedie_density
    c = build('chfagg T tweedie 1.5 100 2', log2=14, bs=1 / 8)
    tw = Tweedie(1.5, 100, 2)
    f, shape, rate = (float(v) for v in tw.po_gamma_shape_rate)
    a = build(f'agg TW {f} claims sev {1 / rate} * gamma {shape} poisson', log2=14, bs=1 / 8)
    both = pd.concat([a.density_df['p_total'].rename('agg'),
                      c.density_df['p_total'].rename('chf')], axis=1, join='inner')
    # the severity is exponential (shape 1), so the continuous part rings
    # at about 3e-7 on this grid where the compound engine is exact
    assert np.max(np.abs(both['agg'] - both['chf'])) < 1e-6
    assert c.family.atom == (0.0, pytest.approx(np.exp(-f)))
    # bucket 0 holds the atom plus the continuous mass on [0, h / 2)
    assert 0 < c.agg_density[0] - np.exp(-f) < 1e-5
    body = c.xs[1:400]
    dens = np.array([tweedie_density(x, p=1.5, μ=100, σ2=2) for x in body])
    # a bucket mass over bs is the density's bucket average, not its value
    assert np.max(np.abs(c.agg_density[1:400] / c.bs - dens)) < 1e-5
    assert c.family.moments[2] == pytest.approx(1.5 * c.family.moments[1] ** 0.5 / 100)


@pytest.mark.parametrize('p, mean, dispersion', [(-1, 1, 1), (2.5, 100, 0.02), (4, 1, 1)])
def test_tweedie_stable_members_match_their_moments(p, mean, dispersion):
    c = build(f'chfagg T tweedie {p} {mean} {dispersion}')
    assert abs(c.est_m - mean) < 1e-6 * mean
    assert abs(c.est_cv - c.actual_cv) < 1e-4 * c.actual_cv
    assert c.family.reference is None
    assert c.family.tilt_side is None and c.tilt == (0.0, 0.0)


def test_tweedie_cv_form_and_refusals():
    c = build('chfagg T tweedie 2.5 100 cv 0.5')
    assert c.family.dispersion == pytest.approx(0.25 * 100 ** -0.5)
    assert c.pprogram.split() == 'chfagg T tweedie 2.5 100 cv 0.5'.split()
    for program, match in [('chfagg T tweedie 1 100 2', 'lattice law'),
                           ('chfagg T tweedie 0.5 100 2', 'no Tweedie law'),
                           ('chfagg T tweedie inf 100 2', 'stable law with index 1'),
                           ('chfagg T tweedie 1.5 -100 2', 'mean must be positive')]:
        with pytest.raises(ValueError, match=match):
            build(program)


def test_tweedie_fz_is_repaired_for_the_stable_members():
    """``Tweedie.fz`` returns an updated ``AggregateCHF`` where it returned ``None`` or an untilted law."""
    from aggregate.characteristic_function import AggregateCHF
    from aggregate.tweedie import Tweedie
    for p in (2.5, 4, -1):
        fz = Tweedie(p, 1, 1).fz
        assert isinstance(fz, AggregateCHF)
        assert abs(fz.est_m - 1) < 1e-6
    # the p = 0 reproductive law carries its dispersion
    assert Tweedie(0, 100, 900).fz.std() == pytest.approx(30)


def test_normal_inverse_gaussian_against_scipy():
    c = build('chfagg N 3 * norminvgauss 6 1.5 + 1')
    _against_reference(c, 1e-7)          # scipy's cdf is quadrature, about 1e-8
    mean, var, skew = c.family.moments
    g = np.sqrt(4 - 0.25)
    assert mean == pytest.approx(1 + 3 * 0.5 / g) and var == pytest.approx(3 * 4 / g ** 3)
    assert abs(c.est_m - mean) < 1e-6 and abs(c.est_skew - skew) < 1e-4
    with pytest.raises(ValueError, match='beta'):
        build('chfagg N norminvgauss 1 2')


def test_variance_gamma_laplace_special_case_and_cumulants():
    c = build('chfagg V variancegamma 0.2 1 0')
    _against_reference(c, 1e-7)
    assert c.family.reference.dist.name == 'laplace'
    d = build('chfagg V variancegamma 0.2 0.5 -0.1')
    mean, var, skew = d.family.moments
    assert (mean, var) == (-0.1, pytest.approx(0.04 + 0.5 * 0.01))
    assert abs(d.est_m - mean) < 1e-6 and abs(d.est_cv - d.actual_cv) < 1e-5 and abs(d.est_skew - skew) < 1e-3
    assert d.family.reference is None


def test_tempered_stable_cumulants():
    c = build('chfagg T temperedstable 1 5 10 0.5')
    fam = c.family
    mean, var, skew = fam.moments
    assert fam.mu == 0 and fam.cumulant(1) == mean
    assert abs(c.est_m - mean) < 1e-6 and abs(c.est_sd ** 2 - var) < 1e-6 * var
    assert abs(c.est_skew - skew) < 1e-5
    d = build('chfagg T temperedstable 1 5 10 1.5 + 2')
    assert d.family.moments[0] == pytest.approx(d.family.base.moments[0] + 2)
    assert abs(d.est_m - d.family.moments[0]) < 1e-6
    assert ' '.join(d.pprogram.split()).endswith('temperedstable 1 5 10 1.5 + 2')
    for program, match in [('chfagg T temperedstable 1 5 10 2', 'below 2'),
                           ('chfagg T temperedstable 1 5 10 1', 'not 0 or 1'),
                           ('chfagg T temperedstable 1 5 10', 'temperedstable takes 4 shapes, C G M Y, got 3')]:
        with pytest.raises(ValueError, match=match):
            build(program)


def test_untilted_window_balances_truncation_against_coverage():
    """A Tweedie with p > 2 and a small tilt cannot have both at log2 16; the rule picks the balance and both rows flag."""
    c = build('chfagg T tweedie 2.5 100 2')
    v = c._checks_frame()
    assert v.loc['truncation error', 'status'] == 'flag'
    assert v.loc['outside mass', 'status'] == 'flag'
    assert 'wraps in' in v.loc['outside mass', 'note'] or 'hints{' in v.loc['outside mass', 'note']
    log2, bs = c.recommended_grid()
    assert log2 > 16


# ----------------------------------------------------------------------
# [Characteristic-Function-Pricing]: pricing, exhibits and the chart
# ----------------------------------------------------------------------

def _numeric(df):
    return df.select_dtypes('number').to_numpy(dtype=float)


def test_pricing_agrees_with_the_compound_engine_on_one_grid():
    c = build('chfagg G 25 * gamma 4', log2=12, bs=1 / 4)
    a = build('agg G dfreq [1] sev 25 * gamma 4', log2=12, bs=1 / 4)
    rc = c.calibrate_distortions(0.10, p=0.99)
    ra = a.calibrate_distortions(0.10, p=0.99)
    assert list(rc.distortion_df.index) == list(ra.distortion_df.index)
    assert np.max(np.abs(_numeric(rc.distortion_df) - _numeric(ra.distortion_df))) < 1e-8
    pc = c.price_pentagon(p=0.99, ROE=0.1)
    pa = a.price_pentagon(p=0.99, ROE=0.1)
    assert list(pc.columns) == list(pa.columns)
    assert np.max(np.abs(_numeric(pc) - _numeric(pa))) < 1e-8
    with pytest.raises(ValueError):
        c.price_pentagon(p=1, ROE=0.1)             # unbounded: p = 1 is refused


def test_exhibits_and_charts_are_the_generic_four_and_the_one_chart():
    from aggregate.exhibits import available_exhibits, exhibit_frames
    from aggregate.charts import available_charts, build_chart_doc, primary_chart
    c = build('chfagg G 25 * gamma 4', log2=12)
    assert [n for n, _ in available_exhibits(c)] == ['summary', 'tail', 'stats', 'validation']
    for name in ('summary', 'tail', 'stats', 'validation'):
        (block, served, kw), = exhibit_frames(c, name)
        assert block == f'{name}_df' and 'caption' in kw
        pd.testing.assert_frame_equal(served, getattr(c, block))
    assert available_charts(c) == ['chfagg'] and primary_chart(c) == 'chfagg'
    doc = build_chart_doc(c, 'chfagg')
    names = {s.name for s in doc.series}
    assert names == {'gamma', 'Reference'}
    s = build('chfagg S stable 1.5 1', log2=12)
    assert {x.name for x in build_chart_doc(s, 'chfagg').series} == {'stable'}
    assert not hasattr(c, 'pnl_program') and not hasattr(c, 'reins_program')
    before = build('chfagg G 25 * gamma 4', update=False)
    assert available_charts(before) == []


# ----------------------------------------------------------------------
# [Characteristic-Function-Severity]: sev chfagg.NAME and ssev chfagg.NAME
# ----------------------------------------------------------------------

def test_chfagg_as_a_severity_matches_the_direct_severity():
    """``agg X dfreq [2] sev chfagg.G`` and ``agg X dfreq [2] sev 25 * gamma 4`` agree on one grid."""
    g = build('chfagg SR.G 25 * gamma 4 hints{log2=12; bs=1/4}')
    a = build('agg SR.X dfreq [2] sev chfagg.SR.G', log2=12, bs=1 / 4)
    b = build('agg SR.Y dfreq [2] sev 25 * gamma 4', log2=12, bs=1 / 4)
    assert np.max(np.abs(a.density_df['p_total'] - b.density_df['p_total'])) < 1e-10
    assert a.sevs[0].sev_name == 'dhistogram'
    assert a.program.split() == 'agg SR.X dfreq [2] sev chfagg.SR.G'.split()
    assert a.pprogram.split()[-1] == 'chfagg.SR.G'


def test_ssev_chfagg_keeps_the_negative_atoms_and_sev_clamps_them():
    build('chfagg SR.N 30 * norm + 100 hints{log2=12; bs=1/4}')
    signed = build('agg SR.S dfreq [1] ssev chfagg.SR.N', log2=12, bs=1 / 4)
    assert signed.density_df.index.min() < 0
    assert abs(signed.est_m - 100) < 1e-9
    with pytest.warns(UserWarning, match='clamped'):
        clamped = build('agg SR.C dfreq [1] sev chfagg.SR.N', log2=12, bs=1 / 4)
    assert clamped.density_df.index.min() >= 0


def test_hygiene_error_names_with_hints_and_gives_the_chfagg_grid():
    build('chfagg SR.H 25 * gamma 4')
    with pytest.raises(ValueError, match=r'with_hints\(\)') as err:
        build('agg SR.UsesH dfreq [2] sev chfagg.SR.H')
    assert 'hints{log2=16; bs=1/64}' in str(err.value)
    assert 'chfagg.SR.H' in str(err.value)


def test_with_hints_pins_the_grid_and_carries_tilt():
    from aggregate.characteristic_function import AggregateCHF
    s = build('chfagg SR.W stable 1.5 1')
    program = s.with_hints(tilt=8)
    assert program == 'chfagg SR.W stable 1.5 1 hints{log2=16; bs=1/32; tilt=8}'
    pinned = build(program)
    assert isinstance(pinned, AggregateCHF) and pinned.tilt == (0.0, 8.0) and pinned.bs == 1 / 32
    with pytest.raises(ValueError, match='not been updated'):
        AggregateCHF('X', 'gamma', shape=4, scale=25).with_hints()


def test_output_hook_and_support_for_the_reference():
    from aggregate.tail import output_support_max
    g = build('chfagg SR.O 25 * gamma 4', log2=12)
    xs, ps = g._output_xs_ps()
    assert xs is g.xs and ps is g.agg_density
    assert output_support_max(g) == np.inf
    for program in ('agg SR.A dfreq [1] dsev [1]', 'port SR.P agg SR.PA dfreq [1] dsev [1]'):
        xs, ps = build(program, log2=8)._output_xs_ps()
        assert len(xs) == len(ps) > 0
    assert build('chfagg SR.U 25 * gamma 4', update=False)._output_xs_ps() == (None, None)


def test_tilted_reference_reports_its_deficit():
    from aggregate.constants import DefectiveDistributionWarning
    s = build('chfagg SR.T 2 * stable 1.5 1 hints{log2=12; bs=1/8}')
    with pytest.warns(DefectiveDistributionWarning, match='missing'):
        t = build('agg SR.UsesT dfreq [1] ssev chfagg.SR.T', log2=14, bs=1 / 8)
    # ssev keeps the signed atoms; the reference renormalizes over the deficit
    assert abs(t.est_m - s.est_m / (1 - s.outside_above)) < 1e-6 * abs(s.est_m) + 1e-9


def test_sev_chfagg_round_trips_through_the_writer(underwriter):
    from aggregate.decl_writer import spec_to_decl
    underwriter.build('chfagg SR.R 25 * gamma 4 hints{log2=12; bs=1/4}')
    for program in ('agg SR.RT dfreq [2] sev chfagg.SR.R',
                    'agg SR.RS dfreq [1] ssev chfagg.SR.R!'):
        kind, name, spec = _parse(underwriter, program)
        assert spec['sev_ref'] == 'chfagg.SR.R'
        rendered = spec_to_decl(spec, kind, name)
        assert _parse(underwriter, rendered)[2] == spec


# ----------------------------------------------------------------------
# [Characteristic-Function-Pedagogy]: the shim, the moves and the figures
# ----------------------------------------------------------------------

def test_nothing_in_the_library_imports_the_ft_shim():
    import aggregate
    root = Path(aggregate.__file__).parent
    offenders = []
    for path in root.rglob('*.py'):
        if path.name == 'ft.py':
            continue
        text = path.read_text(encoding='utf-8')
        for line in text.splitlines():
            # module-level imports only: Tweedie.ft imports the shim locally, on
            # purpose, to hand back the legacy object
            if line.startswith(('from .ft import', 'from aggregate.ft import',
                                'import aggregate.ft', 'from . import ft')):
                offenders.append(f'{path.relative_to(root)}: {line.strip()}')
    assert offenders == []


def test_the_ft_shim_forwards_and_warns():
    import importlib
    import aggregate.ft as ft
    import aggregate.pedagogy as pedagogy
    ft._warned.clear()
    with pytest.warns(DeprecationWarning, match='moved to aggregate.pedagogy'):
        fn = ft.poisson_example
    assert fn is pedagogy.poisson_example
    with pytest.warns(DeprecationWarning, match='Stable'):
        chf = ft.make_levy_chf(1.5, 1)
    assert np.allclose(chf(np.array([0.5, 2.0])), Stable(1.5, 1).chf(np.array([0.5, 2.0])))
    with pytest.warns(DeprecationWarning, match='from_chf'):
        ft.FourierTools(lambda t: np.exp(-t ** 2 / 2), ss.norm())
    with pytest.raises(AttributeError):
        ft.no_such_name
    assert importlib.import_module('aggregate.ft') is ft


def test_simpson_inversion_and_exact_comparison_agree_on_a_normal():
    from aggregate.pedagogy import chf_exact_comparison, chf_simpson_inversion
    n = build('chfagg P.N 30 * norm + 100', log2=12)
    df = chf_simpson_inversion(n)
    body = df['grid'] > 1e-8
    assert np.max(np.abs(df['simpson'] - df['grid'])[body]) < 1e-6
    ex = chf_exact_comparison(n)
    assert len(ex) <= 257 and np.max(np.abs(ex['grid'] - ex['exact'])) < 1e-6
    tw = build('chfagg P.T tweedie 1.5 100 2', log2=12)
    ex2 = chf_exact_comparison(tw)              # an Aggregate reference
    assert np.max(np.abs(ex2['grid'] - ex2['exact'])[ex2.index > 0]) < 1e-3
    with pytest.raises(ValueError, match='no reference'):
        chf_exact_comparison(build('chfagg P.C temperedstable 1 5 10 0.5', log2=10))


def test_each_pedagogy_figure_runs_on_its_example():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from aggregate.pedagogy import (chf_inversion_figure, chf_simpson_figure,
                                    chf_wrapping_figure)
    n = build('chfagg P.F 30 * norm + 100', log2=10)
    for verbose in (True, False):
        plt.close(chf_inversion_figure(n, suptitle='normal', verbose=verbose))
    fig, table = chf_wrapping_figure(n, wraps=(-1, 1))
    plt.close(fig)
    assert list(table.index) == [-1, 1] and (table['probability'] >= 0).all()
    plt.close(chf_simpson_figure(n))
    s = build('chfagg P.S stable 1.5 1', log2=10)
    plt.close(chf_inversion_figure(s))           # levy_stable: no companion, no error
    with pytest.raises(ValueError, match='reference'):
        chf_wrapping_figure(build('chfagg P.V variancegamma 0.2 0.5 -0.1', log2=10))
    plotly = pytest.importorskip('plotly')
    from aggregate.pedagogy import chf_fourier3d_figure
    assert chf_fourier3d_figure(n) is not None


def test_moved_helpers_run_from_pedagogy():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from aggregate.pedagogy import fft_wrapping_illustration, poisson_example, stats
    poisson_example(2, 3)
    fft_wrapping_illustration(ez=10, en=20)
    plt.close('all')
    assert callable(stats)


def test_tweedie_test_and_audit_over_the_characteristic_function():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from aggregate.characteristic_function import AggregateCHF
    from aggregate.tweedie import Tweedie
    tw = Tweedie(2.5, 100, 0.02)
    audit = tw.test(log2=12, plot=True, simpson=True)
    plt.close('all')
    assert isinstance(tw.chf_law, AggregateCHF) and tw.chf_law.log2 == 12
    assert list(audit.index) == ['EX', 'var', 'CV', 'Skew']
    assert abs(audit.iloc[0, 0] - 100) < 1e-4 * 100
    m, v, k = Tweedie(-1, 1, 1).stats()
    assert abs(m - 1) < 1e-6 and abs(v - 1) < 1e-4
    with pytest.warns(DeprecationWarning, match='chf_law'):
        tw.ft


# ----------------------------------------------------------------------
# [CHF-Arithmetic-Form]: scale and shift as arithmetic around the law
# ----------------------------------------------------------------------

#: Bodies written with scale and location numbers before 1.35.0, the full
#: Python parameters that law has, and the arithmetic spelling now required.
REFUSED_SPELLINGS = [
    ('gamma 4 25', 'gamma', dict(shape=4, scale=25), '25 * gamma 4'),
    ('norm 100 30', 'norm', dict(mean=100, sd=30), '30 * norm + 100'),
    ('invgauss 100 400', 'invgauss', dict(mean=100, shape=400), '400 * invgauss 0.25'),
    ('stable 1.5 1 2 10', 'stable', dict(alpha=1.5, beta=1, scale=2, loc=10), '2 * stable 1.5 1 + 10'),
    ('stable 1.5 0.5 2 0', 'stable', dict(alpha=1.5, beta=0.5, scale=2, loc=0), '2 * stable 1.5 0.5'),
    ('stable 1 0.5 2 3', 'stable', dict(alpha=1, beta=0.5, scale=2, loc=3), '2 * stable 1 0.5 + 3.44127120031'),
    ('stable 1 0.5 2 3 s0', 'stable', dict(alpha=1, beta=0.5, scale=2, loc=3, parameterization='S0'),
     '2 * stable 1 0.5 s0 + 3'),
    ('stable 0.7 1 3 5', 'stable', dict(alpha=0.7, beta=1, scale=3, loc=5), '3 * stable 0.7 1 + 5'),
    ('norminvgauss 2 0.5 3 1', 'norminvgauss', dict(alpha=2, beta=0.5, delta=3, mu=1),
     '3 * norminvgauss 6 1.5 + 1'),
    ('variancegamma 0.2 0.5 -0.1 0.05', 'variancegamma', dict(sigma=0.2, nu=0.5, theta=-0.1, mu=0.05),
     'variancegamma 0.2 0.5 -0.1 + 0.05'),
    ('temperedstable 1 5 10 1.5 2', 'temperedstable', dict(C=1, G=5, M=10, Y=1.5, mu=2),
     'temperedstable 1 5 10 1.5 + 2'),
]


@pytest.mark.parametrize('old, family, params, new', REFUSED_SPELLINGS)
def test_a_refused_spelling_points_at_the_same_law(old, family, params, new):
    """The refusal names the arithmetic spelling, and that spelling builds the old law's masses on the old law's grid."""
    with pytest.raises(ValueError, match='as ' + repr(new).replace('*', r'\*').replace('+', r'\+')):
        build(f'chfagg X {old}')
    c = build(f'chfagg X {new}')
    old_law = AggregateCHF('Y', family, **params)
    old_law.update()
    assert (c.bs, c.log2, c.x_min) == (old_law.bs, old_law.log2, pytest.approx(old_law.x_min, abs=1e-12))
    assert np.max(np.abs(c.agg_density - old_law.agg_density)) < 1e-12


def test_shift_and_scale_are_arithmetic_on_the_law():
    g = build('chfagg G 25 * gamma 4 hints{log2=12; bs=1/4}')
    a = build('agg A dfreq [1] sev 25 * gamma 4', log2=12, bs=1 / 4)
    assert np.max(np.abs(g.agg_density - a.agg_density)) < 1e-10
    s = build('chfagg S 25 * gamma 4 + 10 hints{log2=12; bs=1/4}')
    assert s.est_m == pytest.approx(110, rel=1e-9)
    assert s.family.moments == pytest.approx((110, 2500, 1.0))
    m = build('chfagg M 25 * gamma 4 - 10 hints{log2=12; bs=1/4}')
    assert m.est_m == pytest.approx(90, rel=1e-9) and m.support == (-10, np.inf)
    cv = build('chfagg C 2 * gamma 100 cv 0.5')
    assert cv.family.moments[0] == pytest.approx(200)
    n = build('chfagg N norm')
    assert abs(n.est_m) < 1e-9 and n.est_sd == pytest.approx(1, rel=1e-6)


def test_stable_at_alpha_one_is_the_law_of_c_z_plus_b():
    """``2 * stable 1 0.5 + 3`` is :math:`2Z + 3`: scipy's S1 law at ``loc`` 3 less (2/pi) beta gamma log gamma."""
    c = build('chfagg S 2 * stable 1 0.5 + 3')
    loc1 = 3 - 2 / np.pi * 0.5 * 2 * np.log(2)
    assert c.family.location == pytest.approx(loc1)
    ref = ss.levy_stable(1, 0.5, loc=loc1, scale=2)
    if ss.levy_stable.parameterization == 'S0':
        ref = ss.levy_stable(1, 0.5, loc=3, scale=2)
    sel = (c.xs > -30) & (c.xs < 60)
    edges = c.xs[sel] + c.bs / 2
    F = np.cumsum(c.agg_density)[sel]
    assert np.max(np.abs(F - ref.cdf(edges))) < 1e-7
    assert np.max(np.abs(F - c.family.reference.cdf(edges))) < 1e-7


def test_a_shifted_atom_stays_a_grid_point():
    c = build('chfagg T 2 * tweedie 1.5 100 2 + 10.3')
    xa, w = c.family.atom
    assert xa == 10.3 and c.x_min == pytest.approx(10.3)
    assert c.est_m == pytest.approx(210.3, rel=1e-6)


@pytest.mark.parametrize('program', [
    'chfagg A 25 * gamma 4 + 10', 'chfagg B 25 * gamma 4 - 10', 'chfagg C 2 * gamma 100 cv 0.5 + 10',
    'chfagg D norm', 'chfagg E 3 * norminvgauss 6 1.5 + 1', 'chfagg F variancegamma 0.2 0.5 -0.1 + 0.05',
    'chfagg H 2 * stable 1 0.5 s0 + 3', 'chfagg K 2 * tweedie 1.5 100 2 + 10.3'])
def test_arithmetic_round_trips_and_renders_as_written(program, underwriter):
    from aggregate.decl_writer import spec_to_decl
    kind, name, spec = _parse(underwriter, program)
    rendered = spec_to_decl(spec, kind, name)
    assert ' '.join(rendered.split()) == program
    assert build(program).family.to_decl() == program.split(' ', 2)[2]


def test_every_corpus_chfagg_line_round_trips():
    from aggregate.decl_writer import spec_to_decl
    from aggregate import Underwriter
    uw = Underwriter(update=False)
    root = Path(__file__).resolve().parents[1] / 'src' / 'aggregate' / 'agg'
    lines = [ln.rstrip().rstrip(';') for f in ('_test_suite.agg', 'decl-testers.agg')
             for ln in (root / f).read_text(encoding='utf-8').splitlines() if ln.startswith('chfagg ')]
    assert len(lines) >= 30
    for program in lines:
        kind, name, spec = _parse(uw, program)
        kind2, name2, spec2 = _parse(uw, spec_to_decl(spec, kind, name))
        assert (kind2, name2, spec2) == (kind, name, spec), program


def test_a_glued_minus_is_a_sign_and_a_spaced_minus_is_a_shift(underwriter):
    """``NUMBER`` absorbs a glued minus; the shipped resolver keeps that reading (``KNOWN_AMBIGUOUS`` in ``test_grammar_ambiguity.py``)."""
    _, _, spec = _parse(underwriter, 'chfagg V variancegamma 0.2 0.5 -0.1')
    assert spec['theta'] == -0.1 and 'chf_loc' not in spec
    _, _, spec = _parse(underwriter, 'chfagg G 25 * gamma 4 - 10')
    assert spec['shape'] == 4 and spec['chf_loc'] == -10
    with pytest.raises(ValueError, match="glued to a number is its sign; a shift is written with a space, '- 10'"):
        _parse(underwriter, 'chfagg G 25 * gamma 4 -10')


# ----------------------------------------------------------------------
# [CHF-Aggregate-Frames]: the moment store, summary and validation frames
# ----------------------------------------------------------------------

#: chfagg laws and the Aggregate whose frames they must match in shape: an
#: unsigned compound for the unsigned law, a signed one for the signed laws.
FRAME_PAIRS = [
    ('chfagg F.G 25 * gamma 4', 'agg F.A dfreq [1] sev 25 * gamma 4'),
    ('chfagg F.N 30 * norm + 100', 'agg F.S dfreq [1] ssev 30 * norm + 100'),
    ('chfagg F.S 2 * stable 1.5 1 + 10', 'agg F.S dfreq [1] ssev 30 * norm + 100'),
    ('chfagg F.T stable 1.5 0.5', 'agg F.S dfreq [1] ssev 30 * norm + 100'),
]


@pytest.mark.parametrize('chf_program, agg_program', FRAME_PAIRS)
@pytest.mark.parametrize('update', [False, True])
def test_frames_have_the_aggregate_shapes(chf_program, agg_program, update):
    c = build(chf_program, update=update)
    a = build(agg_program, update=update, log2=10)
    for frame in ('summary_df', 'validation_df', 'stats_df'):
        cf, af = getattr(c, frame), getattr(a, frame)
        assert cf.index.equals(af.index), frame
        assert list(cf.columns) == list(af.columns), frame
        assert cf.index.names == af.index.names, frame
    assert c.summary_df.loc[['Freq', 'Sev']].isna().all().all()
    assert c.stats_df.loc[['freq', 'sev']].isna().all().all()
    assert c.stats_df.dtypes.unique().tolist() == [np.dtype(float)]


def test_the_store_feeds_the_accessors():
    for program in ('chfagg S.G 25 * gamma 4', 'chfagg S.S 2 * stable 1.5 1 + 10', 'chfagg S.T stable 1.5 0.5'):
        c = build(program)
        st = c.stats_df
        assert st.loc[('agg', 'mean'), 'mixed'] == c.actual_m
        assert st.loc[('agg', 'mean'), 'empirical'] == c.est_m
        assert st.loc[('agg', 'mean'), 'gross_empirical'] == c.est_m
        assert st.loc[('meta', 'limit'), 'mixed'] == np.inf and st.loc[('meta', 'wt'), 'mixed'] == 1
        assert np.isnan(st.loc[('meta', 'lr'), 'mixed']) and np.isnan(st.loc[('meta', 'sevcv_param'), 'mixed'])
        assert c.validation_df.loc['Agg', 'EX'] == pytest.approx(c.actual_m)
    s = build('chfagg S.S 2 * stable 1.5 1 + 10')
    assert s.stats_df.loc[('agg', 'ex2'), 'mixed'] == np.inf and s.stats_df.loc[('agg', 'ex3'), 'mixed'] == np.inf
    assert s.stats_df.loc[('agg', 'cv'), 'mixed'] == np.inf and np.isnan(s.stats_df.loc[('agg', 'cv'), 'error'])
    assert np.isnan(s.validation_df.loc['Agg', 'Err SD'])
    assert s.validation_df.loc['Agg', 'Err EX'] == pytest.approx(-5.3e-3, rel=0.01)


def test_info_has_aggregates_rows_then_the_inversion_block():
    from aggregate.constants import INFO_LABEL_WIDTH
    a = build('agg I.A dfreq [1] sev 25 * gamma 4', log2=10)

    def labels(info):
        return [ln[:INFO_LABEL_WIDTH].strip() for ln in info.splitlines()]

    agg_labels = labels(a.info)
    head = agg_labels[:agg_labels.index('validation') + 1]
    root = Path(__file__).resolve().parents[1] / 'src' / 'aggregate' / 'agg'
    lines = [ln.rstrip().rstrip(';') for ln in (root / '_test_suite.agg').read_text(encoding='utf-8').splitlines()
             if ln.startswith('chfagg ')]
    for program in lines:
        c = build(program)
        info = c.info
        lab = labels(info)
        assert lab[:len(head)] == head, program
        assert lab[-5:] == agg_labels[-5:], program
        rows = dict(zip(lab, (ln[INFO_LABEL_WIDTH:].strip() for ln in info.splitlines())))
        checks = c._checks_frame()
        for name in ('total mass', 'most negative bucket', 'truncation error', 'buckets per scale'):
            assert rows[name] != 'n/a', (program, name)
        assert rows['outside mass'].startswith(f'{checks.loc["outside mass", "value"]:.3g}') \
            if 'outside mass' in checks.index else True
        n_ok = int((checks.drop([i for i in checks.index if i.startswith(('mean', 'cv', 'skew'))])['status']
                    == 'ok').sum())
        assert rows['checks'].startswith(f'{n_ok} ok'), program


# ----------------------------------------------------------------------
# [CHF-Validation-Verdict]: Aggregate's verdict, vocabulary and opening
# ----------------------------------------------------------------------

@pytest.mark.parametrize('program, verdict', [
    ('gamma 100 cv 0.5', 'not unreasonable'),
    ('25 * gamma 4', 'not unreasonable'),
    ('30 * norm + 100', 'not unreasonable'),
    ('tweedie 1.5 100 2', 'not unreasonable'),
    ('2 * stable 1.5 1 + 10', 'fails agg mean'),
    ('stable 1.1 1', 'fails agg mean'),
])
def test_the_verdict_is_aggregates(program, verdict):
    c = build(f'chfagg V {program}')
    assert c.validation_description == verdict
    assert c.validation_explanation.startswith(
        f'Validation of the realized grid against the analytic moments: {verdict}. Each of '
        'mean, CV and skewness is compared for the aggregate, and a relative error above '
        '0.0001 fails;')
    assert bool(c.valid.passes) == (verdict == 'not unreasonable')


def test_the_grid_flags():
    from aggregate.constants import Validation
    g = build('chfagg V gamma 0.3 hints{log2=10}')
    assert g.valid & Validation.TRUNCATION
    assert 'characteristic function truncated at pi / bs, lower bs' in g.validation_description
    s = build('chfagg V stable 0.7 1')
    assert s.valid & Validation.DEFECTIVE and s.valid & Validation.INFEASIBLE
    assert s.validation_description.startswith('grid infeasible for this law; fails pmf deficit')
    w = build('chfagg V stable 1.5 1 hints{tilt=False}')
    assert w.valid & Validation.ALIASING
    assert 'mass outside the window wrapped in' in w.validation_description
    u = build('chfagg V 25 * gamma 4', update=False)
    assert u.valid == Validation.NOT_UPDATED and u.validation_description == 'n/a, not updated'
    assert u.validation_explanation.startswith('Not validated')
    a = build('agg V.A dfreq [1] sev 25 * gamma 4', log2=10)
    assert not (a.valid & Validation.TRUNCATION) and Validation.TRUNCATION.value > Validation.INFEASIBLE.value
    assert not Validation.TRUNCATION.passes


def test_the_score_and_eps_are_aggregates():
    c = build('chfagg V 25 * gamma 4')
    assert c.validation_eps == build('agg V.A dfreq [1] sev 25 * gamma 4', log2=10).validation_eps
    assert 0 <= c.validation_score < 1
    assert build('chfagg V 2 * stable 1.5 1 + 10').validation_score > 1


def test_tweedie_truncation_is_the_continuous_part():
    c = build('chfagg T tweedie 1.5 100 2')
    assert c.truncation_error == pytest.approx(1.129e-7, rel=1e-3)
    assert c.recommended_grid()[0] < 24


@pytest.mark.parametrize('program', ['2 * stable 1.5 1 + 10', 'stable 1.1 1', '2 * stable 1.5 0.5', 'stable 1.5 -1'])
def test_the_completed_stable_mean_is_the_theoretical(program):
    c = build(f'chfagg V {program}')
    tail, completed = c._completed_mean()
    assert abs(completed - c.actual_m) < 1e-5 * max(abs(c.actual_m), c.family.scale)
    assert f'{completed:,.6g}' in c.validation_explanation


def test_aggregate_and_portfolio_keep_their_opening():
    from aggregate import build as b
    a = b('agg O.A 10 claims sev lognorm 10 cv 1 poisson')
    p = b('port O.P agg O.X 10 claims sev lognorm 10 cv 1 poisson agg O.Y 5 claims sev gamma 20 cv 0.5 poisson')
    for obj in (a, p):
        assert obj.validation_explanation.startswith(
            f'Validation of the realized grid against the analytic moments: {obj.validation_description}. '
            'Each of mean, CV and skewness is compared for severity and for the aggregate, and a '
            'relative error above 0.0001 fails; only the lowest-order failure is reported, since a '
            'mean that is wrong makes the higher moments uninformative.')


def test_every_corpus_explanation_has_the_shared_opening():
    root = Path(__file__).resolve().parents[1] / 'src' / 'aggregate' / 'agg'
    lines = [ln.rstrip().rstrip(';') for ln in (root / '_test_suite.agg').read_text(encoding='utf-8').splitlines()
             if ln.startswith('chfagg ')]
    for program in lines:
        c = build(program)
        assert c.validation_explanation.startswith(
            f'Validation of the realized grid against the analytic moments: {c.validation_description}. '
            'Each of mean, CV and skewness is compared for the aggregate'), program


# ----------------------------------------------------------------------
# [CHF-Sharpen]: a closed-form sharpen
# ----------------------------------------------------------------------

@pytest.mark.parametrize('program', ['gamma 100 cv 0.5', '2 * stable 1.5 1 + 10', 'stable 1.5 0.5'])
def test_sharpen_confirms_a_sound_grid_without_running(program):
    c = build(f'chfagg H {program}')
    hints = c.hints
    assert c.sharpen() is c
    assert c._sharpen_state['ran'] is False and len(c.sharpen_df) == 1
    assert c.note.startswith('sharpen: grid confirmed') and c.hints == hints
    assert c.log2 == 16 and c.sharpen_description.startswith('Sharpen: not run')


def test_sharpen_grows_one_log2_per_call_to_the_recommended_grid():
    s = build('chfagg H stable 1.1 1')
    assert s.recommended_grid()[0] == 19
    seen = []
    for _ in range(3):
        s.sharpen()
        seen.append(s.log2)
    assert seen == [17, 18, 19]
    assert s.sharpen_df.loc[s.sharpen_df['selected'], 'u_coverage'].iloc[0] <= 1
    assert 'hints{log2=19; bs=1/16}' in s.program
    r = build(s.program)
    assert (r.log2, r.bs) == (s.log2, s.bs)
    assert np.max(np.abs(r.agg_density - s.agg_density)) == 0
    assert s.sharpen_program.startswith('chfagg H')


def test_two_sided_recommended_grid_covers():
    c = build('chfagg H stable 1.5 0.5')
    log2, bs = c.recommended_grid()
    assert log2 == 16
    d = build(f'chfagg H stable 1.5 0.5 hints{{log2={log2}; bs={bs}}}')
    assert d.outside_mass <= 1 - d.bucket_sizing_p


def test_forced_sharpen_is_thrifty_and_restores_without_execute():
    g = build('chfagg H 25 * gamma 4')
    g.sharpen(good_enough=0)
    assert int(g.sharpen_df.loc[g.sharpen_df['selected'], 'log2'].iloc[0]) <= 16
    s = build('chfagg H stable 1.1 1')
    p0 = s.agg_density.copy()
    s.sharpen(execute=False)
    assert s.log2 == 16 and np.array_equal(s.agg_density, p0)
    assert 'probe not executed, recommends log2 17' in s.note


def test_sharpen_df_has_aggregates_columns_then_the_inversion_terms():
    a = build('agg H.A 10 claims sev lognorm 10 cv 1 poisson')
    a.sharpen(good_enough=0)
    c = build('chfagg H stable 1.1 1')
    c.sharpen()
    assert list(c.sharpen_df.columns) == list(a.sharpen_df.columns) + [
        'u_coverage', 'u_edge', 'u_truncation', 'u_resolution']
    assert c.sharpen_df.index.names == a.sharpen_df.index.names
    assert c.sharpen_df['u_sev_mean'].isna().all() and c.sharpen_df['aliasing'].isna().all()
    assert 'power tail' in c.sharpen_explanation
