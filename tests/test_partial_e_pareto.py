"""Tests for the analytic Type-I Pareto partial expected values.

``_partial_e`` had a closed form only for the *shifted* (Lomax) Pareto, so the
single-parameter form, which is what ``sev {xm} * pareto {alpha}`` builds and
what the rare-event literature uses, fell through to quadrature on every build:
a logged warning, an ``IntegrationWarning`` from integrating a heavy tail to
infinity, and a numerical answer where an exact one exists.

The oracle here is ``_partial_e_numeric``, the auditing path the analytic
branch replaces, plus scipy's own moments at ``a = inf``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.stats as ss
from scipy.integrate import IntegrationWarning

from aggregate import build
from aggregate._severity import (_moms_analytic, _partial_e, _partial_e_numeric,
                                 _partial_e_pareto_type_1)


# Shapes with every moment up to the third finite, so the quadrature oracle
# has something finite to agree with.
FINITE = [(3.5, 1.0), (5.0, 1.0), (3.5, 250.0), (4.25, 1000.0)]


@pytest.mark.parametrize('alpha, lam', FINITE)
@pytest.mark.parametrize('mult', [1.5, 10.0, 1e4])
def test_matches_quadrature(alpha, lam, mult):
    """The closed form reproduces the numeric path it replaces."""
    a = lam * mult
    fz = ss.pareto(alpha, scale=lam, loc=0)
    analytic = _partial_e_pareto_type_1(alpha, lam, a, 3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntegrationWarning)
        numeric = _partial_e_numeric(fz, a, 3)
    assert analytic == pytest.approx(numeric, rel=1e-10)


@pytest.mark.parametrize('alpha, lam', FINITE)
def test_unlimited_matches_scipy(alpha, lam):
    """At ``a = inf`` the answers are scipy's raw moments."""
    fz = ss.pareto(alpha, scale=lam, loc=0)
    analytic = _partial_e_pareto_type_1(alpha, lam, np.inf, 3)
    assert analytic[0] == pytest.approx(1.0)
    for k in (1, 2, 3):
        assert analytic[k] == pytest.approx(float(fz.moment(k)), rel=1e-9)


def test_unlimited_mean_is_the_textbook_form():
    """E[X] = alpha * lam / (alpha - 1) for the single-parameter Pareto."""
    alpha, lam = 2.5, 40.0
    assert _partial_e_pareto_type_1(alpha, lam, np.inf, 1)[1] == \
        pytest.approx(alpha * lam / (alpha - 1))


def test_k_zero_is_the_cdf():
    """The k = 0 term integrates the density, so it is F(a)."""
    alpha, lam, a = 3.0, 2.0, 8.0
    assert _partial_e_pareto_type_1(alpha, lam, a, 0)[0] == \
        pytest.approx(1.0 - (lam / a) ** alpha)


def test_missing_moments_are_infinite():
    """k >= alpha does not exist, and inf says so; nothing raises."""
    ans = _partial_e_pareto_type_1(1.5, 1.0, np.inf, 3)
    assert ans[0] == pytest.approx(1.0)
    assert ans[1] == pytest.approx(1.5 / 0.5)
    assert np.isinf(ans[2]) and np.isinf(ans[3])


def test_removable_singularity_at_k_equals_alpha():
    """k == alpha is the log branch, the limit of the general expression."""
    alpha, lam, a = 2.0, 1.0, np.e
    got = _partial_e_pareto_type_1(alpha, lam, a, 2)[2]
    assert got == pytest.approx(alpha * lam ** alpha * np.log(a / lam))
    # and it is the limit approached from both sides
    for eps in (1e-6, -1e-6):
        near = _partial_e_pareto_type_1(alpha + eps, lam, a, 2)[2]
        assert near == pytest.approx(got, rel=1e-5)


def test_limit_below_the_support_captures_nothing():
    """Support starts at lam, so a <= lam integrates an empty interval."""
    assert _partial_e_pareto_type_1(3.0, 10.0, 5.0, 3) == [0.0] * 4
    assert _partial_e_pareto_type_1(3.0, 10.0, 10.0, 3) == [0.0] * 4


def test_dispatch_takes_the_analytic_branch():
    """A loc-0 Pareto reaches the closed form, not the fallback."""
    alpha, lam = 3.5, 12.0
    fz = ss.pareto(alpha, scale=lam, loc=0)
    assert _partial_e('pareto', fz, 100.0, 3) == pytest.approx(
        _partial_e_pareto_type_1(alpha, lam, 100.0, 3))


def test_shifted_pareto_still_uses_its_own_closed_form():
    """loc = -scale is the Lomax branch and is untouched."""
    alpha, lam = 3.5, 12.0
    fz = ss.pareto(alpha, scale=lam, loc=-lam)
    analytic = _partial_e('pareto', fz, 100.0, 3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntegrationWarning)
        numeric = _partial_e_numeric(fz, 100.0, 3)
    assert analytic == pytest.approx(numeric, rel=1e-8)


def test_numeric_fallback_starts_at_the_support():
    """Quadrature over a dead region is what made quad cry divergence."""
    alpha, lam = 3.5, 1000.0
    fz = ss.pareto(alpha, scale=lam, loc=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        numeric = _partial_e_numeric(fz, np.inf, 1)
    assert numeric[0] == pytest.approx(1.0, rel=1e-6)
    assert numeric[1] == pytest.approx(alpha * lam / (alpha - 1), rel=1e-6)
    assert [x for x in w if x.category is IntegrationWarning] == []


def test_build_is_quiet_and_right():
    """The reproductions-book case: sev {xm} * pareto {alpha}, no chatter."""
    alpha, xm = 2.5, 1.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        a = build(f'agg PT1 1 claim sev {xm} * pareto {alpha} fixed',
                  log2=16, bs=1 / 64)
    assert [x for x in w if x.category is IntegrationWarning] == []
    assert a.sev_m == pytest.approx(alpha * xm / (alpha - 1))


# --------------------------------------------------------------------------
# Layer moments over a severity whose moments do not all exist.
#
# `_moms_analytic` expands E[((X-a)^+ ^ l)^m] binomially in the partial
# expectations. That expansion cannot survive an infinite term: at attachment
# 0 the low-order coefficients are 0, so a term reads 0 * inf; above 0 the
# alternating signs give inf - inf. Both produced nan for a layer moment that
# is simply infinite. Surfaced by the analytic Type-I branch, which reports a
# missing moment as inf where quadrature had returned a large finite number.
# --------------------------------------------------------------------------

@pytest.mark.parametrize('attachment', [0.0, 2.0])
def test_unlimited_layer_reports_infinite_moments(attachment):
    """alpha = 1.5: mean exists, second and third do not, and say so."""
    fz = ss.pareto(1.5, scale=1.0, loc=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        out = _moms_analytic(fz, np.inf, attachment, 3)
    assert np.isfinite(out[0]) and np.isfinite(out[1])
    assert np.isinf(out[2]) and np.isinf(out[3])
    assert not np.isnan(out).any()


def test_unlimited_layer_infinite_mean():
    """alpha < 1: even the mean is gone, from m = 1 up."""
    fz = ss.pareto(0.8, scale=1.0, loc=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        out = _moms_analytic(fz, np.inf, 0.0, 3)
    assert out[0] == pytest.approx(1.0)
    assert np.isinf(out[1:]).all()


@pytest.mark.parametrize('alpha', [0.8, 1.5, 2.5])
@pytest.mark.parametrize('attachment', [0.0, 2.0])
def test_a_finite_limit_is_always_finite(alpha, attachment):
    """A capped layer is bounded by limit**m however heavy the tail."""
    fz = ss.pareto(alpha, scale=1.0, loc=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        out = _moms_analytic(fz, 10.0, attachment, 3)
    assert np.isfinite(out).all()
    assert (out[1:] > 0).all()


def test_finite_layer_matches_quadrature():
    """The short circuit does not disturb the case that always worked."""
    fz = ss.pareto(1.5, scale=1.0, loc=0)
    analytic = _moms_analytic(fz, 10.0, 2.0, 3)
    numeric = _moms_analytic(fz, 10.0, 2.0, 3, False)
    assert analytic == pytest.approx(numeric, rel=1e-8)
