"""The Tsallis distortion, the Beta(alpha, 1) mixture of TVaRs (2.1.0a3).

:math:`g_\\alpha(u) = (\\alpha u - u^\\alpha)/(\\alpha - 1)`, with
:math:`g_1(u) = u(1 - \\log u)`. The cases follow the numbered tests of
``dev/done/plan-tsallis-distortion-and-gini.md`` for ``[Tsallis-Distortion]``.
The DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg`` under
the W. distortion section.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy.integrate import quad

from aggregate import Distortion, Underwriter, build
from aggregate._pricing import DEFAULT_CALIBRATION_DISTORTIONS
from aggregate.decl_words import decl_words
from aggregate.decl_writer import spec_to_decl
from aggregate.spectral import TsallisDistortion

ALPHAS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 7.0]
US = [0.05, 0.3, 0.7]


def _kusuoka_g(u, alpha):
    """``g`` as the Beta(alpha, 1) mixture of ``min(u / s, 1)`` over ``s``."""
    below = quad(lambda s: alpha * s ** (alpha - 1), 0, u,
                 epsabs=1e-14, epsrel=1e-13, limit=200)[0]
    above = quad(lambda s: u / s * alpha * s ** (alpha - 1), u, 1,
                 epsabs=1e-14, epsrel=1e-13, limit=200)[0]
    return below + above


# --- 1. closed form against the Kusuoka integral ----------------------------

@pytest.mark.parametrize('alpha', ALPHAS)
def test_g_is_the_beta_mixture_of_tvars(alpha):
    d = Distortion('tsallis', alpha=alpha)
    for u in US:
        assert float(d.g(u)) == pytest.approx(_kusuoka_g(u, alpha), abs=1e-9)


# --- 2. continuity at alpha = 1, and the two ends ---------------------------

def test_continuous_through_alpha_one():
    u = np.array(US)
    one = Distortion('tsallis', alpha=1.0)
    for alpha in (1 - 1e-9, 1 + 1e-9):
        d = Distortion('tsallis', alpha=alpha)
        np.testing.assert_allclose(d.g(u), one.g(u), rtol=0, atol=1e-8)
        np.testing.assert_allclose(d.g_prime(u), one.g_prime(u), rtol=0, atol=1e-8)


def test_extreme_orders_approach_the_mean_and_the_maximum():
    u = np.array([1e-300, 1e-12, *US, 1.0])
    near_mean = Distortion('tsallis', alpha=1e6).g(u)
    near_max = Distortion('tsallis', alpha=1e-6).g(u[2:])
    assert np.all(np.isfinite(near_mean)) and np.all(np.isfinite(near_max))
    np.testing.assert_allclose(near_mean, u, rtol=2e-6)
    np.testing.assert_allclose(near_max, 1.0, atol=1e-4)


def test_tiny_u_at_a_small_order_does_not_overflow():
    """``expm1`` of ``(alpha - 1) log u`` overflows here; the direct form does not."""
    alpha = 1e-6
    u = np.array([5e-324, 1e-310])
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        g = Distortion('tsallis', alpha=alpha).g(u)
    np.testing.assert_allclose(g, (u ** alpha - alpha * u) / (1 - alpha), rtol=1e-12)


# --- 3. special cases --------------------------------------------------------

def test_alpha_two_is_dual_two_and_alpha_half_is_root():
    u = np.linspace(0, 1, 101)
    np.testing.assert_allclose(Distortion('tsallis', 2).g(u),
                               Distortion('dual', 2).g(u), rtol=0, atol=1e-15)
    np.testing.assert_allclose(Distortion('tsallis', 0.5).g(u),
                               2 * np.sqrt(u) - u, rtol=0, atol=1e-15)


# --- 4. Gini index -----------------------------------------------------------

@pytest.mark.parametrize('alpha', ALPHAS)
def test_gini_is_one_over_alpha_plus_one(alpha):
    d = Distortion('tsallis', alpha=alpha)
    assert d.gini == 1 / (alpha + 1)
    area = quad(lambda s: float(d.g(s)), 0, 1, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
    assert 2 * area - 1 == pytest.approx(d.gini, abs=1e-10)


def test_gini_of_order_three_is_exactly_a_quarter():
    assert Distortion('tsallis', alpha=3).gini == 0.25


# --- 5. derivative and inverse -----------------------------------------------

@pytest.mark.parametrize('alpha', ALPHAS)
def test_g_prime_matches_a_central_difference(alpha):
    d = Distortion('tsallis', alpha=alpha)
    h = 1e-6
    for u in US:
        numeric = (float(d.g(u + h)) - float(d.g(u - h))) / (2 * h)
        assert float(d.g_prime(u)) == pytest.approx(numeric, rel=1e-7)


def test_g_prime_at_the_ends():
    assert Distortion('tsallis', 3).g_prime(0.0) == pytest.approx(1.5)
    assert Distortion('tsallis', 0.5).g_prime(0.0) == np.inf
    assert Distortion('tsallis', 1).g_prime(0.0) == np.inf
    for alpha in ALPHAS:
        assert Distortion('tsallis', alpha).g_prime(1.0) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize('alpha', ALPHAS + [0.05, 50.0])
def test_g_of_g_inv_is_the_identity(alpha):
    d = Distortion('tsallis', alpha=alpha)
    y = np.array([0.0, 1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999, 1.0])
    np.testing.assert_allclose(d.g(d.g_inv(y)), y, rtol=0, atol=1e-12)


def test_g_inv_is_zero_below_the_smallest_float():
    """At ``alpha = 0.001`` the root of ``g(u) = 0.1`` is near ``1e-1000``."""
    d = Distortion('tsallis', alpha=1e-3)
    assert d.g_inv(0.1) == 0.0
    assert d.g(d.g_inv(0.7)) == pytest.approx(0.7, abs=1e-12)


# --- 6. describe -------------------------------------------------------------

@pytest.mark.parametrize('alpha', ALPHAS)
def test_closed_form_moments_of_d_g(alpha):
    d = Distortion('tsallis', alpha=alpha)
    cf = d._describe_closed_form()
    raw = [quad(lambda x, k=k: k * x ** (k - 1) * (1 - float(d.g(x))), 0, 1,
                epsabs=1e-13, epsrel=1e-13, limit=200)[0] for k in (1, 2, 3)]
    for k, m in zip((1, 2, 3), raw):
        assert m == pytest.approx(alpha / ((k + 1) * (k + alpha)), rel=1e-10)
    assert cf['mean'] == pytest.approx(raw[0], rel=1e-10)
    assert cf['var'] == pytest.approx(raw[1] - raw[0] ** 2, rel=1e-9)
    assert cf['loading'] == pytest.approx(0.5 - raw[0], rel=1e-10)
    assert d.validation_df['Pass'].all()


def test_kusuoka_measure_is_beta_one_alpha_with_no_atoms():
    d = Distortion('tsallis', alpha=3)
    assert d._kusuoka_atoms() == []
    p = np.array([0.0, 0.5, 0.9])
    np.testing.assert_allclose(d._kusuoka_density(p), 3 * (1 - p) ** 2)
    assert d.has_mass is False and d.mass == 0.0


# --- construction ------------------------------------------------------------

def test_construction_forms_agree():
    forms = [Distortion('tsallis', 3), Distortion('tsallis', alpha=3),
             TsallisDistortion(alpha=3), Distortion.tsallis(3)]
    for d in forms:
        assert type(d) is TsallisDistortion
        assert d.alpha == 3 and d.label == 'Tsallis(3)'
    d = forms[0]
    d.alpha = 1.0
    assert d.gini == 0.5


@pytest.mark.parametrize('alpha', [0, -1, np.inf, np.nan])
def test_alpha_must_be_finite_and_positive(alpha):
    with pytest.raises(ValueError, match='alpha must be finite and positive'):
        Distortion('tsallis', alpha=alpha)


# --- 8. calibration ----------------------------------------------------------

def test_calibrate_on_a_survival_vector():
    S = np.linspace(0.999, 0.001, 100)
    bs = 0.1
    premium = float(np.sum(S) * bs) * 1.2
    d = Distortion('tsallis', alpha=TsallisDistortion._calibration_init_shape)
    d.calibrate(S=S, dx=bs, premium_target=premium)
    assert abs(d.error) < 1e-9
    assert float(np.sum(d.g(S)) * bs) == pytest.approx(premium, abs=1e-9)


def test_calibrate_below_expected_loss_is_a_failed_calibration(caplog):
    S = np.linspace(0.999, 0.001, 100)
    el = float(np.sum(S) * 0.1)
    with caplog.at_level(logging.WARNING, logger='aggregate.spectral'):
        d = Distortion('tsallis', alpha=3).calibrate(S=S, dx=0.1, premium_target=0.9 * el)
    assert d.alpha == pytest.approx(1e9, rel=1e-6)
    assert d.error > 0
    assert 'questionable convergence' in caplog.text


def test_calibrate_distortions_by_name_on_a_book():
    book = build('agg TS.Book 50 claims sev lognorm 40 cv 1.8 poisson')
    result = book.calibrate_distortions(coc=0.1, p=0.99, names=['tsallis'])
    ddf = result.distortion_df
    assert list(ddf.index) == ['tsallis']
    assert abs(ddf.loc['tsallis', 'error']) < 1e-5
    assert ddf.loc['tsallis', 'param_name'] == 'alpha'
    assert ddf.loc['tsallis', 'gini'] == pytest.approx(1 / (ddf.loc['tsallis', 'param'] + 1))


# --- 9. DecL -----------------------------------------------------------------

@pytest.mark.parametrize('program, alpha, label', [
    ('distortion D tsallis 3', 3.0, 'Tsallis(3)'),
    ('distortion D as "T" tsallis 0.5', 0.5, 'T'),
])
def test_decl_builds_and_round_trips(program, alpha, label):
    d = build(program)
    assert type(d) is TsallisDistortion
    assert d.alpha == alpha and d.label == label
    uw = Underwriter()
    kind, name, spec = uw.parser.parse(uw.lexer.tokenize(program))
    assert spec['name'] == 'tsallis' and spec['alpha'] == alpha
    text = spec_to_decl(spec, kind, name)
    assert uw.parser.parse(uw.lexer.tokenize(text))[2] == spec


# --- 13. the default lists are unchanged -------------------------------------

def test_tsallis_is_outside_the_default_lists():
    assert Distortion.available_distortions() == DEFAULT_CALIBRATION_DISTORTIONS
    assert 'tsallis' not in DEFAULT_CALIBRATION_DISTORTIONS
    broad = Distortion.available_distortions(pricing=True, strict=False)
    assert broad[:6] == ('ccoc', 'ph', 'wang', 'tsallis', 'dual', 'tvar')
    assert 'tsallis' in Distortion.available_distortions(pricing=False)
    assert 'tsallis' in decl_words()['distortions']
