"""Layer moments on the six closed-form kinds, wherever the layer attaches.

``[Analytic-Layer-Tail-Cancellation]``: ``_moms_analytic`` differenced lower
partial expectations and expanded about 0, so a layer past the median subtracted
nearly equal numbers. ``10 xs 1000`` on ``lognorm 1`` had a second moment above
100, the largest possible, and a negative third. The cure has two parts, because
upper partial expectations remove the factor ``F(a) / S(a)`` and not ``(a / y)^m``:
a thin layer integrates ``t^m f(a + t)`` directly, and any other layer past the
median differences the upper side.

The oracle is the survival integral ``E[Y^m] = int_0^y m t^(m-1) S(a + t) dt`` for
the layer ``Y = min(y, (X - a)+)``. Its integrand is positive, so it cannot cancel,
and it uses no partial expectation.

The DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg`` under the
``LM`` section.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.stats as ss
from scipy.integrate import IntegrationWarning, quad
from scipy.special import binom

from aggregate import build
from aggregate._severity import (_THIN_LAYER_RATIO, Severity, _excess_moments,
                                 _moms_analytic, _partial_e, _partial_e_upper)

LAWS = {
    'lognorm': ss.lognorm(1.0, scale=100.0),
    'gamma': ss.gamma(2.0, scale=100.0),
    'expon': ss.expon(scale=100.0),
    'pareto': ss.pareto(3.5, scale=100.0),
    'lomax': ss.lomax(3.5, scale=100.0),
    'beta': ss.beta(2.0, 3.0, scale=100.0),
}
# F(a) = 0.5 to S(a) = 1e-12, ten attachments
SURVIVALS = [0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]


def _survival_integral(fz, limit, attachment, epsrel=1e-13):
    """``E[min(limit, (X - a)+)^m]``, ``m = 1, 2, 3``, unconditional."""
    top = min(limit, float(fz.support()[1]) - attachment)
    out = []
    for m in (1, 2, 3):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', IntegrationWarning)
            val, _ = quad(lambda t: m * t ** (m - 1) * fz.sf(attachment + t),
                          0.0, top, epsabs=0, epsrel=epsrel, limit=500)
        out.append(val)
    return np.array(out)


def _retention_reference(fz, d1, d2, y):
    """``E[g(X)^m; X > d1]``, ``m = 1, 2, 3``, by the density piece by piece."""
    k = d2 / (d2 - d1) if d2 > d1 else np.inf
    det = d1 + y / k if y <= d2 else y
    out = []
    for m in (1, 2, 3):
        val = 0.0
        if d2 > d1:
            end = min(d2, det)
            val += quad(lambda t: (k * t) ** m * fz.pdf(d1 + t), 0.0, end - d1,
                        epsabs=0, epsrel=1e-13, limit=500)[0]
        if det > d2:
            val += quad(lambda x: x ** m * fz.pdf(x), d2, det,
                        epsabs=0, epsrel=1e-13, limit=500)[0]
        out.append(val + y ** m * fz.sf(det))
    return np.array(out)


# --------------------------------------------------------------------------
# The measured failures
# --------------------------------------------------------------------------

@pytest.mark.parametrize('sigma, scale, limit, attachment', [
    (1.0, 1.0, 10.0, 1000.0),
    (1.0, 100.0, 10.0, 1e5),
    (1.0, 100.0, 100.0, 9e4),
    (1.0, 100.0, 1.0, 1e6),
])
def test_far_tail_lognormal_layer_matches_the_survival_integral(sigma, scale, limit, attachment):
    """The rows that came back with a negative third moment."""
    fz = ss.lognorm(sigma, scale=scale)
    got = _moms_analytic(fz, limit, attachment, 3)[1:]
    assert got == pytest.approx(_survival_integral(fz, limit, attachment), rel=1e-8)
    conditional = got / fz.sf(attachment)
    assert (conditional > 0).all()
    assert conditional[1] <= limit ** 2
    assert conditional[2] <= limit ** 3


def test_far_tail_layer_quoted_values():
    """The conditional moments quoted in the plan."""
    sev = Severity('lognorm', sev_a=1.0, sev_scale=1.0, exp_attachment=1000.0, exp_limit=10.0)
    assert list(sev.moms()) == pytest.approx([9.65674, 95.4381, 948.779], rel=1e-5)
    sev = Severity('lognorm', sev_a=1.0, sev_scale=100.0, exp_attachment=1e5, exp_limit=10.0)
    assert list(sev.moms()) == pytest.approx([9.99648, 99.9530, 999.472], rel=1e-5)


@pytest.mark.parametrize('program', [
    'agg LM.Far 1 claim 10 xs 1e5 sev 100 * lognorm 1 fixed',
    'agg LM.Near 1 claim 10 xs 1000 sev lognorm 1 fixed',
])
def test_far_tail_layer_builds_clean(program):
    """The discretization was right all along; the verdict compared it to a wrong moment.

    The grid is an independent check: its severity skewness comes from no
    partial expectation, and the two agree to far better than the 1e-4 of the
    verdict, on a layer whose variance is 2e-4 of its second moment.
    """
    a = build(program)
    assert a.validation_description == 'not unreasonable'
    assert float(a.est_sev_skew) == pytest.approx(float(a.sev_skew), rel=1e-7)


# --------------------------------------------------------------------------
# Six kinds, ten attachments, thin to wide
# --------------------------------------------------------------------------

@pytest.mark.parametrize('kind', list(LAWS))
@pytest.mark.parametrize('survival', SURVIVALS)
@pytest.mark.parametrize('ratio', [1000.0, 100.0, 17.0, 16.0, 4.0, 1.0, 0.1])
def test_layer_matches_the_survival_integral(kind, survival, ratio):
    """Ratio is attachment over width, which sets how much the expansion loses."""
    fz = LAWS[kind]
    a = float(fz.isf(survival))
    y = a / ratio
    if kind == 'beta':
        # the width that matters is what is left of the support
        y = min(y, float(fz.support()[1]) - a)
    assert _moms_analytic(fz, y, a, 3)[1:] == pytest.approx(
        _survival_integral(fz, y, a), rel=1e-8)


@pytest.mark.parametrize('survival', SURVIVALS)
@pytest.mark.parametrize('fraction', [1.0, 0.5, 0.1, 0.01, 0.001])
def test_beta_near_the_top_of_its_support(survival, fraction):
    """Here ``a / y`` is large at every width, so no upper form can reach it."""
    fz = LAWS['beta']
    a = float(fz.isf(survival))
    y = fraction * (float(fz.support()[1]) - a)
    assert _moms_analytic(fz, y, a, 3)[1:] == pytest.approx(
        _survival_integral(fz, y, a, epsrel=1e-11), rel=1e-8)


@pytest.mark.parametrize('kind', ['lognorm', 'gamma', 'expon'])
@pytest.mark.parametrize('survival', [1e-2, 1e-6, 1e-12])
def test_unlimited_far_tail_layer(kind, survival):
    """An unlimited layer has no width to integrate over, so it takes the upper form."""
    fz = LAWS[kind]
    a = float(fz.isf(survival))
    got = _moms_analytic(fz, np.inf, a, 3)[1:]
    ref = [quad(lambda u: m * (np.exp(u) - a) ** (m - 1) * fz.sf(np.exp(u)) * np.exp(u),
                np.log(a), np.log(a) + 8, epsabs=0, epsrel=1e-12, limit=500)[0]
           for m in (1, 2, 3)]
    assert got == pytest.approx(ref, rel=1e-8)


def test_upper_and_lower_partial_expectations_sum_to_the_moment():
    """``P[X^k; X <= a] + P[X^k; X > a] = P[X^k]`` where the moment exists.

    To 1e-10, the accuracy of scipy's own third moment of the Pareto.
    """
    for kind, fz in LAWS.items():
        for x in fz.ppf([0.1, 0.5, 0.9]):
            lower = _partial_e(kind, fz, float(x), 3)
            upper = _partial_e_upper(kind, fz, float(x), 3)
            total = [1.0] + [float(fz.moment(k)) for k in (1, 2, 3)]
            assert np.add(lower, upper) == pytest.approx(total, rel=1e-10)


# --------------------------------------------------------------------------
# Regimes, and what stayed put
# --------------------------------------------------------------------------

def _lower_form(kind, fz, limit, attachment):
    """The moments as they were: lower partial expectations, expanded about 0."""
    pe_a = _partial_e(kind, fz, attachment, 3)
    pe_d = _partial_e(kind, fz, attachment + limit, 3)
    return np.array([
        sum((-1) ** (m - k) * binom(m, k) * attachment ** (m - k) * (pe_d[k] - pe_a[k])
            for k in range(m + 1)) + limit ** m * fz.sf(attachment + limit)
        for m in range(4)])


@pytest.mark.parametrize('kind', list(LAWS))
@pytest.mark.parametrize('survival', [0.9, 0.7, 0.5001])
@pytest.mark.parametrize('ratio', [0.1, 1.0, 10.0])
def test_ordinary_layer_at_or_below_the_median_is_unchanged(kind, survival, ratio):
    """Exactly the old arithmetic: no switch, so no number moves."""
    fz = LAWS[kind]
    a = float(fz.isf(survival))
    y = a * ratio
    assert np.array_equal(_moms_analytic(fz, y, a, 3), _lower_form(kind, fz, y, a))


@pytest.mark.parametrize('kind, fz, limit, attachment', [
    ('pareto', ss.pareto(1.5, scale=1.0), 10.0, 2.0),
    ('pareto', ss.pareto(2.5, scale=1.0), 10.0, 2.0),
    ('beta', ss.beta(0.02, 20.0, scale=100.0), 50.0, 1.0),
    ('lomax', ss.lomax(1.0, scale=120.0), 750.0, 250.0),
    ('lomax', ss.lomax(2.0, scale=120.0), 750.0, 250.0),
    ('lomax', ss.lomax(3.0, scale=120.0), 750.0, 250.0),
])
def test_existing_layers_past_the_median_keep_their_values(kind, fz, limit, attachment):
    """The layers of the two ``test_partial_e`` files that take the upper form."""
    assert float(fz.sf(attachment)) < 0.5
    assert _moms_analytic(fz, limit, attachment, 3) == pytest.approx(
        _lower_form(kind, fz, limit, attachment), rel=1e-12)


@pytest.mark.parametrize('alpha', [1.5, 2.5])
@pytest.mark.parametrize('limit, attachment', [(1e4, 1e4), (10.0, 1e3)])
def test_finite_layer_on_a_moment_that_does_not_exist(alpha, limit, attachment):
    """The upper partial expectation is infinite, the layer's moment is not."""
    fz = ss.pareto(alpha, scale=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        got = _moms_analytic(fz, limit, attachment, 3)[1:]
    assert got == pytest.approx(_survival_integral(fz, limit, attachment), rel=1e-8)


def test_unlimited_far_tail_layer_keeps_infinite_moments():
    fz = ss.pareto(1.5, scale=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        got = _moms_analytic(fz, np.inf, 100.0, 3)
    assert np.isfinite(got[:2]).all() and np.isinf(got[2:]).all()


def test_layer_straddling_the_bottom_of_a_pareto():
    """The thin rule starts at the support, where the density jumps."""
    fz = ss.pareto(2.5, scale=100.5)
    assert _moms_analytic(fz, 1.0, 100.0, 3)[1:] == pytest.approx(
        _survival_integral(fz, 1.0, 100.0), rel=1e-8)


@pytest.mark.parametrize('q', [0.5, 1.5, 1.9, 2.0, 2.5, 3.0])
@pytest.mark.parametrize('attachment', [99.0, 99.9, 99.99])
@pytest.mark.parametrize('fraction', [1.0, 0.5, 0.02])
def test_thin_layer_near_the_top_of_a_beta_at_any_shape(q, attachment, fraction):
    """``(top - x)^(q - 1)`` stalls a fixed rule and the expansion about ``a`` both.

    At ``q = 1/2`` the rule alone is off by 1.5e-2, and at ``q = 1.5`` the upper
    form is off by 6e-3 on a layer from 99.99 to the top. The fraction is of the
    distance to the top: 1 reaches it, 1/2 stops short and reflects, and 0.02 is
    narrow against that distance and takes the rule.

    The width is computed from the top rather than written as a round number.
    At ``q = 1/2`` the survival function has infinite slope at the top, so
    ``0.01 xs 99.99``, whose detachment rounds to exactly 100 while its width
    does not reach it, is itself uncertain at about 1e-6.
    """
    fz = ss.beta(2.0, q, scale=100.0)
    limit = fraction * (100.0 - attachment)
    assert _moms_analytic(fz, limit, attachment, 3)[1:] == pytest.approx(
        _survival_integral(fz, limit, attachment, epsrel=1e-11), rel=1e-8)


def test_layer_past_the_top_of_a_beta_is_empty():
    fz = LAWS['beta']
    assert np.array_equal(_moms_analytic(fz, 10.0, 200.0, 3), np.zeros(4))


def test_thin_rule_threshold_is_continuous():
    """The two sides of the switch agree, so the switch moves nothing visible."""
    fz = LAWS['lognorm']
    a = float(fz.isf(1e-6))
    below = a / (_THIN_LAYER_RATIO * (1 + 1e-9))
    above = a / (_THIN_LAYER_RATIO * (1 - 1e-9))
    assert _excess_moments('lognorm', fz, a, below, 3) == pytest.approx(
        _excess_moments('lognorm', fz, a, above, 3), rel=1e-8)


def test_zero_limit_returns_zeros():
    """``moms()`` used to raise, unpacking three values from a length three array."""
    assert np.array_equal(_moms_analytic(ss.expon(), 0.0, 10.0, 3), np.zeros(4))
    sev = Severity('expon', sev_scale=1.0, exp_attachment=10.0, exp_limit=0.0)
    assert tuple(sev.moms()) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# Franchise and disappearing deductibles take the same care
# --------------------------------------------------------------------------

@pytest.mark.parametrize('d1, d2, y', [
    (1000.0, 1100.0, 100.0),      # the measured row
    (1000.0, 1100.0, 1500.0),     # ramp and identity piece
    (1000.0, 1000.0, 1500.0),     # franchise
    (1e5, 1e5 + 10.0, 500.0),     # thin ramp
], ids=['disappearing', 'identity', 'franchise', 'thin-ramp'])
def test_far_tail_retention_matches_integration(d1, d2, y):
    fz = ss.lognorm(1.0)
    sev = Severity('lognorm', sev_a=1.0, sev_scale=1.0, exp_attachment=d1, exp_limit=y,
                   exp_retention_end=d2, sev_conditional=False)
    assert list(sev.moms()) == pytest.approx(
        _retention_reference(fz, d1, d2, y), rel=1e-8)
