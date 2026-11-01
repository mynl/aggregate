"""Moments of a shifted severity ``c * law + b``, computed without the shift.

``[Shift-Removed-Moments]``: a shifted law went to quadrature of ``isf ** k`` with
the shift inside the integrand, at a relative tolerance of 1e-4, and the central
moments were differences of numbers of size ``b ** k``. ``100 * gamma 2 + 10000``
had skewness 1.387 and ``+ 1000000`` had ``nan``, so a correct discretization
failed its validation. A layer ``y xs a`` on ``X + b`` is the layer ``y xs (a - b)``
on ``X``, so the moments of ``X`` do the work and the shift never enters.

The oracle for a layer is the survival integral on the unshifted law,
``E[Y^m] = int_0^y m t^(m-1) S(a - b + t) dt``. A skewness is compared with scipy's
own for the unshifted law, which a shift cannot change.

The DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg`` under the
``SM`` section.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.stats as ss
from scipy.integrate import IntegrationWarning, quad
from scipy.special import binom

import aggregate._severity as severity_module
from aggregate import build
from aggregate._severity import Severity, _moms_analytic

UNSHIFTED = {
    '100 * gamma 2': ss.gamma(2, scale=100),
    '100 * lognorm 1': ss.lognorm(1, scale=100),
    '100 * weibull_min 2': ss.weibull_min(2, scale=100),
    '100 * expon': ss.expon(scale=100),
    '100 * beta 2 3': ss.beta(2, 3, scale=100),
    '100 * invgamma 5': ss.invgamma(5, scale=100),
    '100 * gumbel_r': ss.gumbel_r(scale=100),
    '100 * norm': ss.norm(scale=100),
}


def _layer_reference(fz0, shift, attachment, limit):
    """``E[min(y, (X + b - a)+)^m]``, ``m = 1, 2, 3``, by the survival integral."""
    a = attachment - shift
    top = min(limit, float(fz0.support()[1]) - a)
    # S is 1 until the layer reaches the bottom of the support, and has a kink there.
    # Decades, so quadrature over a long range does not step over the body of the law.
    knots = [float(fz0.support()[0]) - a] + [10.0 ** j for j in range(2, 9)]
    knots = [0.0] + sorted(k for k in knots if 0 < k < top) + [top]
    pieces = list(zip(knots[:-1], knots[1:]))
    out = []
    for m in (1, 2, 3):
        val = 0.0
        for lo, hi in pieces:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', IntegrationWarning)
                val += quad(lambda t: m * t ** (m - 1) * fz0.sf(a + t), lo, hi,
                            epsabs=0, epsrel=1e-13, limit=500)[0]
        out.append(val)
    return np.array(out)


# --------------------------------------------------------------------------
# The skewness of the review's table
# --------------------------------------------------------------------------

@pytest.mark.parametrize('law, shift, tolerance', [
    ('100 * gamma 2', 10000, 1e-6),
    ('100 * gamma 2', 100000, 1e-6),
    ('100 * gamma 2', 1000000, 1e-4),
    ('100 * lognorm 1', 10000, 1e-6),
    ('100 * weibull_min 2', 10000, 1e-6),
    # sd 46, so epsilon (b / sd)^3 is 1.1e-6: the rounding limit, not the method
    ('100 * weibull_min 2', 100000, 1e-5),
    ('100 * beta 2 3', 10000, 1e-6),
    ('100 * invgamma 5', 10000, 1e-6),
    ('100 * gumbel_r', 10000, 1e-6),
    ('100 * norm', 10000, 1e-6),
])
def test_skewness_of_a_shifted_law_is_the_skewness_of_the_law(law, shift, tolerance):
    a = build(f'agg SM.T 1 claim sev {law} + {shift} fixed', update=False)
    exact = float(UNSHIFTED[law].stats(moments='s'))
    assert float(a.sev_skew) == pytest.approx(exact, abs=tolerance)


@pytest.mark.parametrize('law', ['100 * gamma 2', '100 * weibull_min 2', '100 * gumbel_r', '100 * norm'])
def test_mean_and_variance_of_a_shifted_law(law):
    a = build(f'agg SM.T 1 claim sev {law} + 10000 fixed', update=False)
    fz = UNSHIFTED[law]
    s = a.sevs[0]
    assert s.actual_m == pytest.approx(float(fz.mean()) + 10000, rel=1e-12)
    assert s.actual_var == pytest.approx(float(fz.var()), rel=1e-6)


def test_a_clamped_normal_stays_clamped():
    """``30 * norm + 100`` reaches below 0, and the clamp is the severity's meaning."""
    a = build('agg SM.Clamp 1 claim sev 30 * norm + 100 fixed', update=False)
    z = 100 / 30
    exact = 100 * ss.norm.cdf(z) + 30 * ss.norm.pdf(z)
    assert float(a.sev_m) == pytest.approx(exact, rel=1e-9)
    assert float(a.sev_m) == pytest.approx(100.0034, abs=5e-5)


@pytest.mark.parametrize('law', ['lognorm', 'gamma', 'weibull_min'])
def test_negative_shift_clamps_the_loss(law):
    """``X - 50`` with no layer clause keeps its mass at 0 and is not conditioned."""
    fz = {'lognorm': ss.lognorm(1, scale=100), 'gamma': ss.gamma(2, scale=100),
          'weibull_min': ss.weibull_min(2, scale=100)}[law]
    shape = {'lognorm': 1, 'gamma': 2, 'weibull_min': 2}[law]
    sev = Severity(law, sev_a=shape, sev_scale=100, sev_loc=-50)
    assert list(sev.moms()) == pytest.approx(_layer_reference(fz, -50, 0.0, 1e7), rel=1e-7)


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------

@pytest.mark.parametrize('law, shift, attachment, limit', [
    ('100 * gamma 2', 1000, 1100, 200),
    ('100 * gamma 2', 1000, 500, 800),
    ('100 * gamma 2', 1000, 500, 300),
    ('100 * lognorm 1', 10000, 10100, 500),
    ('100 * lognorm 1', 10000, 2000, 8500),
    ('100 * weibull_min 2', 10000, 10050, 100),
    ('100 * weibull_min 2', 10000, 9000, 1200),
])
def test_layer_on_a_shifted_law_matches_the_survival_integral(law, shift, attachment, limit):
    """The review's seven layers, three of which failed validation."""
    fz = UNSHIFTED[law]
    a = build(f'agg SM.T 1 claim {limit} xs {attachment} sev {law} + {shift} fixed')
    s = a.sevs[0]
    ref = _layer_reference(fz, shift, attachment, limit) / fz.sf(attachment - shift)
    assert list(s.moms()) == pytest.approx(ref, rel=1e-8)
    assert a.validation_description == 'not unreasonable'


@pytest.mark.parametrize('law', ['100 * gamma 2', '100 * lognorm 1', '100 * beta 2 3', '100 * expon'])
@pytest.mark.parametrize('attachment, limit', [(9000, 1200), (9000, 1020), (9990, 100), (10030, 40)])
def test_binomial_expansion_of_the_method(law, attachment, limit):
    """The author's formula, ``P[(d + W)^n]`` with ``W = min(y - d, X)``, for ``a < b``."""
    fz = UNSHIFTED[law]
    shift = 10000
    got = _moms_analytic(fz, limit, attachment - shift, 3)[1:]
    if attachment >= shift:
        assert got == pytest.approx(_layer_reference(fz, shift, attachment, limit), rel=1e-8)
        return
    d = shift - attachment
    w = _moms_analytic(fz, limit - d, 0.0, 3)
    expanded = [sum(binom(n, k) * d ** (n - k) * w[k] for k in range(n + 1)) for n in (1, 2, 3)]
    assert got == pytest.approx(expanded, rel=1e-12)


@pytest.mark.parametrize('law', ['100 * gamma 2', '100 * lognorm 1', '100 * weibull_min 2', '100 * norm'])
def test_layer_entirely_below_the_law_is_the_constant(law):
    """Every loss pays the limit, as a point mass with no integration behind it."""
    a = build(f'agg SM.T 1 claim 50 xs 0 sev {law} + 10000 fixed', update=False)
    assert list(a.sevs[0].moms()) == pytest.approx([50, 50 ** 2, 50 ** 3], rel=1e-12)


def test_far_tail_layer_on_a_shifted_law():
    """Moving the shift out moves the attachment down the law into the tail."""
    a = build('agg SM.Far 1 claim 100 xs 100000 sev 100 * lognorm 1 + 10000 fixed')
    moms = a.sevs[0].moms()
    assert moms[0] == pytest.approx(99.6154, abs=1e-3)
    assert moms[1] == pytest.approx(9948.73, abs=1e-1)
    assert moms[2] == pytest.approx(994233.6, abs=1)
    assert a.validation_description == 'not unreasonable'


def test_far_tail_layer_on_a_law_with_no_closed_form():
    """Quantile quadrature integrates the excess, so a thin layer does not cancel."""
    fz = ss.weibull_min(0.5, scale=100)
    sev = Severity('weibull_min', sev_a=0.5, sev_scale=100, sev_loc=10000,
                   exp_attachment=100000.0, exp_limit=100.0)
    ref = _layer_reference(fz, 10000, 100000.0, 100.0) / fz.sf(90000.0)
    assert list(sev.moms()) == pytest.approx(ref, rel=1e-8)


# --------------------------------------------------------------------------
# Verdicts and grids
# --------------------------------------------------------------------------

# the four at 1,000,000 are the rounding limit, about epsilon (b / sd)^3
ROUNDING_LIMIT = {('100 * weibull_min 2', 1000000), ('100 * expon', 1000000),
                  ('100 * beta 2 3', 1000000), ('100 * invgamma 5', 1000000)}


@pytest.mark.parametrize('law', ['100 * gamma 2', '100 * lognorm 1', '100 * weibull_min 2',
                                 '100 * expon', '100 * beta 2 3', '100 * invgamma 5'])
@pytest.mark.parametrize('shift', [0, 1000, 10000, 100000, 1000000])
def test_verdict_of_the_thirty_programs(law, shift):
    """13 failed and 6 more had a ``nan`` skewness; the rest of the grid was right."""
    program = f'agg SM.T 1 claim sev {law}' + (f' + {shift}' if shift else '') + ' fixed'
    a = build(program)
    assert np.isfinite(float(a.sev_skew))
    if (law, shift) not in ROUNDING_LIMIT:
        assert a.validation_description == 'not unreasonable'


@pytest.mark.parametrize('program, bs, log2', [
    ('10 claims sev 100 * gamma 2 + 10000 poisson', 8.0, 16),
    ('1 claim sev 100 * gamma 2 + 10000 fixed', 0.25, 16),
    ('10 claims sev 100 * gamma 2 + 1000000 poisson', 800.0, 16),
    ('10 claims sev 100 * lognorm 1 + 10000 poisson', 8.0, 16),
    ('10 claims sev 100 * weibull_min 2 + 100000 poisson', 80.0, 16),
    ('100 claims sev 100 * gamma 2 + 100000 mixed gamma 0.2', 800.0, 16),
])
def test_grid_does_not_move(program, bs, log2):
    """The bad skewness changed the verdict and never the grid. Values from 1.49.0."""
    a = build(f'agg SM.T {program}')
    assert (float(a.bs), a.log2) == (bs, log2)


# --------------------------------------------------------------------------
# Franchise and disappearing deductibles
# --------------------------------------------------------------------------

def _retention_reference(fz0, shift, d1, d2, y):
    """``E[g(X + b)^m; X + b > d1]`` by the density piece by piece, on the unshifted law."""
    k = d2 / (d2 - d1) if d2 > d1 else np.inf
    det = d1 + y / k if y <= d2 else y
    lo = float(fz0.support()[0])
    out = []
    for m in (1, 2, 3):
        val = 0.0
        if d2 > d1:
            start = max(d1 - shift, lo)
            end = min(d2, det) - shift
            if end > start:
                val += quad(lambda x: (k * (x + shift - d1)) ** m * fz0.pdf(x), start, end,
                            epsabs=0, epsrel=1e-11, limit=500)[0]
        if det > d2:
            start = max(d2 - shift, lo)
            end = det - shift
            if end > start:
                val += quad(lambda x: (x + shift) ** m * fz0.pdf(x), start, end,
                            epsabs=0, epsrel=1e-11, limit=500)[0]
        out.append(val + y ** m * fz0.sf(det - shift))
    return np.array(out)


@pytest.mark.parametrize('name, shape, fz', [
    ('lognorm', 1.0, ss.lognorm(1.0, scale=100)),
    ('gamma', 2.0, ss.gamma(2.0, scale=100)),
    ('weibull_min', 2.0, ss.weibull_min(2.0, scale=100)),
], ids=['lognorm', 'gamma', 'weibull_min'])
@pytest.mark.parametrize('d1, d2, y', [
    (10050.0, 10150.0, 12000.0),   # attaches above the shift, ramp and identity piece
    (10050.0, 10150.0, 50.0),      # the limit is reached on the ramp
    (10050.0, 10050.0, 12000.0),   # franchise
    (9000.0, 10100.0, 12000.0),    # attaches below the shift
    (9000.0, 9000.0, 12000.0),     # franchise below the shift
], ids=['above', 'on-ramp', 'franchise', 'below', 'franchise-below'])
def test_retention_on_a_shifted_law(name, shape, fz, d1, d2, y):
    sev = Severity(name, sev_a=shape, sev_scale=100, sev_loc=10000, exp_attachment=d1,
                   exp_limit=y, exp_retention_end=d2, sev_conditional=False)
    assert list(sev.moms()) == pytest.approx(_retention_reference(fz, 10000, d1, d2, y), rel=1e-7)


# --------------------------------------------------------------------------
# Dispatch: who has a law without its shift, and what stayed put
# --------------------------------------------------------------------------

def _raise(*args, **kwargs):
    raise AssertionError('quadrature path reached')


@pytest.mark.parametrize('decl', [
    'sev 100 * gamma 2 + 10000',
    'sev 100 * lognorm 1 + 10000',
    'sev 100 * expon + 10000',
    'sev 100 * beta 2 3 + 10000',
    'sev 100 * pareto 2.5 + 10000',
    'sev 100 * lomax 2.5 + 10000',
])
@pytest.mark.parametrize('layer', ['', '50 xs 10050 ', '500 xs 9000 '])
def test_shifted_closed_form_never_reaches_quadrature(monkeypatch, decl, layer):
    a = build(f'agg SM.T 1 claim {layer}{decl} fixed', update=False)
    sev = a.sevs[0]
    monkeypatch.setattr(severity_module, '_numerical_moms', _raise)
    monkeypatch.setattr(severity_module, '_numerical_moms_shifted', _raise)
    monkeypatch.setattr(severity_module, '_partial_e_numeric', _raise)
    assert sev.fz0 is not None and sev._shift == 10000
    assert sev._uses_closed_form_moments()
    assert not np.isnan(sev.moms()).any()


@pytest.mark.parametrize('decl', [
    'sev 100 * gamma 2',
    'sev 120 * pareto 2.5 - 120',
    'sev 100 * gamma 2 + 50 splice [100 500]',
    'sev 100 - lognorm 1',
    'ssev 100 * gamma 2 + 50',
    'sev 3 * cantor + 5',
    'dsev [1 2 3]',
])
def test_a_law_out_of_scope_has_no_unshifted_law(decl):
    a = build(f'agg SM.T 1 claim {decl} fixed', update=False)
    assert a.sevs[0].fz0 is None


def test_the_shift_comes_from_the_frozen_law():
    """A ``beta`` from a mean and cv is built at location 0 whatever ``sev_loc`` is."""
    sev = Severity('beta', sev_mean=5, sev_cv=0.3, sev_scale=100, sev_loc=10)
    assert sev.moms()[0] == pytest.approx(float(sev.fz.mean()), rel=1e-12)


def test_shifted_mean_cv_form():
    """``lognorm 5 cv .3 + 10`` has the moments of the law at its declared mean and cv."""
    a = build('agg SM.T 1 claim sev lognorm 5 cv .3 + 10 fixed', update=False)
    assert float(a.sev_m) == pytest.approx(15.0, rel=1e-12)
    assert float(a.sevs[0].actual_sd) == pytest.approx(1.5, rel=1e-10)
