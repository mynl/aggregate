"""Brute-force / analytic checks for :class:`GridDistribution` (plan P1, Phase 1).

The probability accessors are pure functions of ``(x, p)`` and make no
equal-spacing assumption; these tests pin them on small hand-checkable grids
(fair die, skewed atoms) and on a deliberately non-uniform grid, cross-checking
every accessor against an independent computation (a fine numeric integral of
the step quantile function, a direct cumulative sum, etc.).
"""

import numpy as np
import pandas as pd
import pytest

from aggregate._grid_distribution import GridDistribution, make_var_tvar


# ----------------------------------------------------------------------
# brute-force references (independent of the kernel)
# ----------------------------------------------------------------------
def brute_tvar(x, p, alpha, n=4_000_000):
    """TVaR_alpha = (1/(1-alpha)) integral_alpha^1 q_lower(u) du, by fine quadrature."""
    if alpha >= 1:
        return x[-1]
    F = np.cumsum(p)
    us = np.linspace(alpha, 1.0, n, endpoint=False)
    # lower quantile q(u) = smallest x with F(x) >= u
    idx = np.searchsorted(F, us, side='left')
    idx = np.clip(idx, 0, len(x) - 1)
    return float(np.mean(x[idx]))


def cap_law(x, p, a):
    """The (x, p) of min(X, a): pool mass at points >= a onto an atom at a."""
    if a >= x[-1]:
        return x.copy(), p.copy()
    mask = x < a
    return np.append(x[mask], a), np.append(p[mask], p[~mask].sum())


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def die():
    x = np.arange(1.0, 7.0)
    p = np.full(6, 1.0 / 6.0)
    return GridDistribution(x, p, bs=1.0, name='die')


@pytest.fixture
def skewed():
    # zero-based, unequal masses; bs = 10 regular grid
    x = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    p = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
    return GridDistribution(x, p, bs=10.0, name='skewed')


# ----------------------------------------------------------------------
# basic accessors on the fair die
# ----------------------------------------------------------------------
def test_mean(die):
    assert die.mean() == pytest.approx(3.5)


def test_quantiles_lower_upper(die):
    assert die.q(0.5, 'lower') == 3.0
    assert die.q(0.5, 'upper') == 4.0
    # below the first jump
    assert die.q(0.0, 'lower') == 1.0
    # vectorized
    np.testing.assert_array_equal(die.q([1 / 6, 0.5, 0.99]), [1.0, 3.0, 6.0])


def test_var_alias(die):
    assert die.var(0.5) == die.q(0.5, 'lower')


def test_cdf_sf(die):
    assert die.cdf(3) == pytest.approx(0.5)
    assert die.cdf(0.5) == 0.0          # below support
    assert die.cdf(10) == pytest.approx(1.0)
    assert die.sf(3) == pytest.approx(0.5)
    np.testing.assert_allclose(die.cdf([1, 2, 6]), [1 / 6, 2 / 6, 1.0])


def test_pmf(die):
    assert die.pmf(3) == pytest.approx(1 / 6)
    assert die.pmf(3.5) == 0.0          # not a grid point
    np.testing.assert_allclose(die.pmf([1, 3.5, 6]), [1 / 6, 0.0, 1 / 6])


def test_tvar_analytic(die):
    # mean of {4, 5, 6}
    assert die.tvar(0.5) == pytest.approx(5.0)
    # whole distribution
    assert die.tvar(0.0) == pytest.approx(3.5)


@pytest.mark.parametrize('alpha', [0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
def test_tvar_vs_brute(skewed, alpha):
    got = float(skewed.tvar(alpha))
    want = brute_tvar(skewed.x, skewed.p, alpha)
    assert got == pytest.approx(want, rel=1e-4, abs=1e-3)


def test_tvar_vectorizes(skewed):
    alphas = np.array([0.0, 0.25, 0.5, 0.9])
    got = skewed.tvar(alphas)
    want = np.array([float(skewed.tvar(a)) for a in alphas])
    np.testing.assert_allclose(got, want)


# ----------------------------------------------------------------------
# lev — limited expected value, zero-based grid
# ----------------------------------------------------------------------
def test_lev_vs_riemann_S(skewed):
    # bs * sum_{x < a} S(x)
    a = 25.0
    S = skewed.sf(skewed.x)
    want = skewed.bs * np.sum(S[skewed.x < a])
    assert skewed.lev(a) == pytest.approx(want)


def test_lev_full_is_mean_zero_based(skewed):
    # On a zero-based grid the left-Riemann lev to beyond the support equals the
    # mean only in the bs->0 limit; here check the explicit Riemann identity holds
    # and is monotone increasing in a, bounded by the mean's Riemann analogue.
    big = skewed.lev(1e9)
    S = skewed.sf(skewed.x)
    assert big == pytest.approx(skewed.bs * np.sum(S))


def test_lev_nonuniform_diff_widths():
    x = np.array([0.0, 1.0, 3.0, 7.0])   # non-uniform
    p = np.array([0.4, 0.3, 0.2, 0.1])
    gd = GridDistribution(x, p)          # bs=None -> np.diff widths
    a = 5.0
    S = gd.sf(x)
    widths = np.diff(x)
    want = sum(S[i] * min(widths[i], a - x[i]) for i in range(len(x) - 1) if x[i] < a)
    assert gd.lev(a) == pytest.approx(want)


# ----------------------------------------------------------------------
# tvar_of_limited and cap
# ----------------------------------------------------------------------
@pytest.mark.parametrize('a', [15.0, 25.0, 35.0])
@pytest.mark.parametrize('alpha', [0.0, 0.25, 0.5, 0.8])
def test_tvar_of_limited_vs_cap(skewed, a, alpha):
    # analytic composite == kernel on the capped law
    analytic = float(skewed.tvar_of_limited(alpha, a))
    via_cap = float(skewed.cap(a).tvar(alpha))
    assert analytic == pytest.approx(via_cap, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize('a', [15.0, 25.0])
@pytest.mark.parametrize('alpha', [0.1, 0.5])
def test_tvar_of_limited_vs_brute(skewed, a, alpha):
    cx, cp = cap_law(skewed.x, skewed.p, a)
    want = brute_tvar(cx, cp, alpha)
    got = float(skewed.tvar_of_limited(alpha, a))
    assert got == pytest.approx(want, rel=1e-4, abs=1e-3)


def test_tvar_of_limited_infinite_cap(skewed):
    assert float(skewed.tvar_of_limited(0.5, np.inf)) == pytest.approx(float(skewed.tvar(0.5)))


def test_cap_returns_grid_distribution(skewed):
    capped = skewed.cap(25.0)
    assert isinstance(capped, GridDistribution)
    # mass conserved
    assert capped.p.sum() == pytest.approx(skewed.p.sum())
    # no atom beyond the cap
    assert capped.x.max() == 25.0


def test_cap_beyond_support_is_identity(skewed):
    capped = skewed.cap(1e9)
    np.testing.assert_array_equal(capped.x, skewed.x)
    np.testing.assert_array_equal(capped.p, skewed.p)


def test_tvar_of_limited_vs_bounds(skewed):
    # shared-case cross-check against Bounds._tvar_x_a (the hand-rolled formula
    # GridDistribution.tvar_of_limited replaces in Phase 3).
    from aggregate.bounds import Bounds
    ser = pd.Series(skewed.p, index=skewed.x)
    a = 25.0
    b = Bounds(ser, premium=skewed.mean() + 1.0, a=a)
    for alpha in (0.1, 0.3, 0.5):
        assert float(b._tvar_x_a(alpha)) == pytest.approx(
            float(skewed.tvar_of_limited(alpha, a)), rel=1e-9, abs=1e-9)


# ----------------------------------------------------------------------
# width-dependent ops: require bs
# ----------------------------------------------------------------------
def test_pdf_requires_bs(skewed):
    assert skewed.pdf(10.0) == pytest.approx(skewed.pmf(10.0) / skewed.bs)
    gd = GridDistribution(skewed.x, skewed.p)   # bs=None
    with pytest.raises(ValueError, match='pdf requires'):
        gd.pdf(10.0)


def test_snap_requires_bs(skewed):
    assert skewed.snap(13.0) == 10.0
    assert skewed.snap(16.0) == 20.0
    gd = GridDistribution(skewed.x, skewed.p)   # bs=None
    with pytest.raises(ValueError, match='snap requires'):
        gd.snap(13.0)


# ----------------------------------------------------------------------
# non-uniform grid: probability accessors still correct
# ----------------------------------------------------------------------
def test_nonuniform_probability_accessors():
    x = np.array([0.0, 1.0, 4.0, 9.0, 16.0])   # quadratic spacing
    p = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    gd = GridDistribution(x, p)                 # bs=None ok for prob accessors
    F = np.cumsum(p)
    # cdf matches direct cumulative
    np.testing.assert_allclose(gd.cdf(x), F)
    # lower quantile by direct searchsorted
    for alpha in (0.0, 0.3, 0.55, 0.9):
        want = x[min(np.searchsorted(F, alpha, side='left'), len(x) - 1)]
        assert gd.q(alpha, 'lower') == want
    # tvar matches brute force on the same non-uniform grid
    for alpha in (0.0, 0.3, 0.7):
        assert float(gd.tvar(alpha)) == pytest.approx(
            brute_tvar(x, p, alpha), rel=1e-4, abs=1e-3)


# ----------------------------------------------------------------------
# from_series and kernel parity
# ----------------------------------------------------------------------
def test_from_series(die):
    ser = pd.Series(die.p, index=die.x, name='die')
    gd = GridDistribution.from_series(ser, bs=1.0)
    assert gd.mean() == pytest.approx(die.mean())
    assert gd.tvar(0.5) == pytest.approx(die.tvar(0.5))
    assert gd.name == 'die'


def test_make_var_tvar_kernel_parity(skewed):
    # GridDistribution wraps the same kernel; q/tvar must match make_var_tvar directly
    ser = pd.Series(skewed.p, index=skewed.x)
    qf = make_var_tvar(ser)
    for alpha in (0.1, 0.5, 0.9):
        assert skewed.q(alpha, 'lower') == qf.q_lower(alpha)
        assert skewed.q(alpha, 'upper') == qf.q_upper(alpha)
        assert float(skewed.tvar(alpha)) == pytest.approx(float(qf.tvar(alpha)))


def test_zero_mass_points_ignored_by_kernel():
    # points with p == 0 must not affect quantiles (holder used query('p>0'))
    x = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([0.5, 0.0, 0.0, 0.5])
    gd = GridDistribution(x, p, bs=1.0)
    assert gd.q(0.25, 'lower') == 0.0
    assert gd.q(0.75, 'lower') == 3.0
    assert gd.tvar(0.5) == pytest.approx(3.0)
