"""Tests for the closed-form beta and Lomax partial expected values.

``[Beta-Lomax-Exact-Moments]``: ``Severity.moms`` computed layered moments in
closed form for ``lognorm``, ``gamma``, ``pareto`` and ``expon`` only. The
badly skewed betas an ELT row produces (tiny ``a``, huge ``b``) returned nan
or raised by quadrature, and a Lomax, the shifted Pareto, was numerical while
the Pareto was exact. The shifted Pareto branch of ``_partial_e`` also divided
by ``i - alpha`` and returned nan at integer shapes.

The oracles are ``_partial_e_numeric``, scipy's own raw moments at ``x = inf``,
and, for layer moments, quadrature of ``m (x - a)^(m-1) S(x)`` over the layer,
which is smooth where the beta density near 0 is not.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.stats as ss
from scipy.integrate import IntegrationWarning, quad

import aggregate._severity as severity_module
from aggregate import build
from aggregate._severity import (Severity, _partial_e, _partial_e_lomax,
                                 _partial_e_numeric, feasible_bucket)


def _quad_quiet(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntegrationWarning)
        return quad(*args, **kwargs)


def _numeric(fz, a, n=3):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntegrationWarning)
        return _partial_e_numeric(fz, a, n)


def _layer_reference(fz, attachment, limit, conditional):
    """``E[min(limit, (X - a)+)^m]``, ``m = 1, 2, 3``, by the survival function.

    ``E[Y^m] = int_a^{a+l} m (x - a)^(m-1) S(x) dx`` for the layer ``Y``,
    divided by ``S(a)`` when conditional.
    """
    out = []
    for m in (1, 2, 3):
        val, _ = _quad_quiet(lambda x: m * (x - attachment) ** (m - 1) * fz.sf(x),
                             attachment, attachment + limit,
                             epsabs=0, epsrel=1e-13, limit=200)
        out.append(val / fz.sf(attachment) if conditional else val)
    return out


# --------------------------------------------------------------------------
# [beta-closed-form]
# --------------------------------------------------------------------------

BENIGN_BETA = [(2.625, 2.625), (2.0, 5.0), (5.0, 2.0), (3.0, 1.5)]

# Every shape from the plan's measurement table, including the ones that
# returned nan or raised by quadrature.
TABLE_BETA = [(2.625, 2.625, 100.0), (0.005, 50.0, 100.0), (1e-4, 1e4, 1e8),
              (1e-5, 1.0, 1e8), (1e-6, 10.0, 1e8), (0.02, 20.0, 100.0)]


@pytest.mark.parametrize('p, q', BENIGN_BETA)
@pytest.mark.parametrize('x', [30.0, 60.0, 90.0])
def test_beta_matches_quadrature(p, q, x):
    """The closed form reproduces the numeric path on benign shapes."""
    fz = ss.beta(p, q, loc=0, scale=100.0)
    analytic = _partial_e('beta', fz, x, 3)
    assert analytic == pytest.approx(_numeric(fz, x), rel=1e-8)


@pytest.mark.parametrize('p, q, scale', TABLE_BETA)
def test_beta_unlimited_matches_scipy(p, q, scale):
    """At ``x = inf``, and past the support, the answers are the raw moments."""
    fz = ss.beta(p, q, loc=0, scale=scale)
    for x in (np.inf, 2 * scale):
        analytic = _partial_e('beta', fz, x, 3)
        assert analytic[0] == 1.0
        for k in (1, 2, 3):
            assert analytic[k] == pytest.approx(float(fz.moment(k)), rel=1e-12)


@pytest.mark.parametrize('p, q, scale', TABLE_BETA)
@pytest.mark.parametrize('frac', [1e-3, 0.1, 0.5])
def test_beta_k_zero_is_the_cdf(p, q, scale, frac):
    """The ``k = 0`` term integrates the density, so it is ``F(x)``."""
    fz = ss.beta(p, q, loc=0, scale=scale)
    x = frac * scale
    assert _partial_e('beta', fz, x, 0)[0] == pytest.approx(fz.cdf(x), rel=1e-12)


@pytest.mark.parametrize('p, q', [(1e-6, 10.0), (1e-5, 1.0)])
def test_beta_extreme_shapes_are_finite_and_exact(p, q):
    """The shapes that gave ``ZeroDivisionError`` and nan by quadrature."""
    sev = Severity('beta', sev_a=p, sev_b=q, sev_scale=1e8)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        got = sev.moms()
    assert np.isfinite(got).all()
    for k, v in zip((1, 2, 3), got):
        assert v == pytest.approx(float(sev.fz.moment(k)), rel=1e-12)


@pytest.mark.parametrize('conditional', [True, False])
def test_beta_layer_matches_survival_reference(conditional):
    """``50 xs 1`` on beta ``0.02 20``, scale 100."""
    sev = Severity('beta', exp_attachment=1.0, exp_limit=50.0, sev_a=0.02,
                   sev_b=20.0, sev_scale=100.0, sev_conditional=conditional)
    ref = _layer_reference(sev.fz, 1.0, 50.0, conditional)
    assert list(sev.moms()) == pytest.approx(ref, rel=1e-10)


# --------------------------------------------------------------------------
# [lomax-closed-form]
# --------------------------------------------------------------------------

LAM = 120.0
LOMAX_SHAPES = [1.0, 2.0, 2.5, 3.0, 4.25]


@pytest.mark.parametrize('alpha', LOMAX_SHAPES)
@pytest.mark.parametrize('mult', [1.0, 10.0, 100.0])
def test_lomax_matches_quadrature(alpha, mult):
    """Partial moments at and above ``lam``; below it the alternating sum
    loses about ``(lam / x)^k`` ulps, so that region is not checked here."""
    fz = ss.lomax(alpha, scale=LAM)
    x = mult * LAM
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        analytic = _partial_e('lomax', fz, x, 3)
    assert analytic == pytest.approx(_numeric(fz, x), rel=1e-8)


@pytest.mark.parametrize('alpha', LOMAX_SHAPES)
def test_lomax_unlimited(alpha):
    """Existing moments are scipy's; ``k >= alpha`` is ``inf``, quietly."""
    fz = ss.lomax(alpha, scale=LAM)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        analytic = _partial_e('lomax', fz, np.inf, 3)
    assert analytic[0] == 1.0
    for k in (1, 2, 3):
        if k < alpha:
            assert analytic[k] == pytest.approx(float(fz.moment(k)), rel=1e-12)
        else:
            assert analytic[k] == np.inf


@pytest.mark.parametrize('alpha', [1.0, 2.0, 3.0])
@pytest.mark.parametrize('attachment, limit', [(0.0, 1000.0), (250.0, 750.0)])
def test_lomax_integer_shape_layers(alpha, attachment, limit):
    """Integer shapes, where the old shifted Pareto branch gave nan."""
    sev = Severity('lomax', exp_attachment=attachment, exp_limit=limit,
                   sev_a=alpha, sev_scale=LAM)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        got = sev.moms()
    assert np.isfinite(got).all()
    ref = _layer_reference(sev.fz, attachment, limit, True)
    assert list(got) == pytest.approx(ref, rel=1e-10)


@pytest.mark.parametrize('alpha', [1.0, 2.0, 2.5, 3.0])
def test_lomax_unlimited_severity_reports_missing_moments(alpha):
    """An unlimited Lomax: ``inf`` exactly where ``k >= alpha``."""
    sev = Severity('lomax', sev_a=alpha, sev_scale=LAM)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        got = sev.moms()
    for k, v in zip((1, 2, 3), got):
        if k < alpha:
            assert v == pytest.approx(float(sev.fz.moment(k)), rel=1e-12)
        else:
            assert v == np.inf


def test_lomax_large_scale_and_shape_do_not_overflow():
    """``lam^alpha`` overflowed at scale 1e8, shape 60; ``expm1`` avoids it."""
    alpha, lam = 60.0, 1e8
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        got = _partial_e_lomax(alpha, lam, np.inf, 3)
    fz = ss.lomax(alpha, scale=lam)
    for k in (1, 2, 3):
        assert got[k] == pytest.approx(float(fz.moment(k)), rel=1e-9)


@pytest.mark.parametrize('alpha', [2.0, 2.5, 3.0])
@pytest.mark.parametrize('x', [LAM, 10 * LAM, np.inf])
def test_lomax_and_shifted_pareto_partial_e_agree(alpha, x):
    """``lomax(a, scale=l)`` and ``pareto(a, loc=-l, scale=l)`` share a branch."""
    lomax = _partial_e('lomax', ss.lomax(alpha, scale=LAM), x, 3)
    pareto = _partial_e('pareto', ss.pareto(alpha, loc=-LAM, scale=LAM), x, 3)
    assert lomax == pareto


@pytest.mark.parametrize('layer', ['', '750 xs 250 '])
def test_lomax_and_shifted_pareto_decl_agree(layer):
    """The two DecL spellings of one law give the same layered moments.

    Agreement is to rounding rather than bit for bit in a layer, where the
    ``limit^m S(d)`` term reads each law's own scipy ``sf``.
    """
    a = build(f'agg T 1 claim {layer}sev 120 * lomax 2.5 fixed', update=False)
    b = build(f'agg T 1 claim {layer}sev 120 * pareto 2.5 - 120 fixed', update=False)
    ma, mb = a.sevs[0].moms(), b.sevs[0].moms()
    for u, v in zip(ma, mb):
        assert u == v or u == pytest.approx(v, rel=1e-12)


# --------------------------------------------------------------------------
# Dispatch and the feasibility decision
# --------------------------------------------------------------------------

def _raise(*args, **kwargs):
    raise AssertionError('quadrature path reached')


@pytest.mark.parametrize('decl', [
    'sev 100 * beta 2 5',
    'sev 1e8 * beta 1e-6 10',
    'sev 120 * lomax 2.5',
    'sev 120 * pareto 2.5 - 120',
])
@pytest.mark.parametrize('layer', ['', '50 xs 10 '])
def test_moms_never_reaches_quadrature(monkeypatch, decl, layer):
    """Beta, Lomax and the shifted Pareto take the closed form, layered or not."""
    a = build(f'agg T 1 claim {layer}{decl} fixed', update=False)
    sev = a.sevs[0]
    monkeypatch.setattr(severity_module, '_numerical_moms', _raise)
    monkeypatch.setattr(severity_module, '_partial_e_numeric', _raise)
    assert sev._uses_closed_form_moments()
    assert not np.isnan(sev.moms()).any()


@pytest.mark.parametrize('decl', ['sev 120 * lomax 2.5', 'sev 100 * beta 2 5'])
def test_feasibility_reading_still_declines(decl):
    """``[Feasibility-Lomax-Beta]`` is deferred: no reading for either kind."""
    a = build(f'agg T 1 claim {decl} fixed', update=False)
    assert feasible_bucket(a.sevs[0], 1e-4) is None
