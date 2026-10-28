"""Tests for ``aggregate.stable.Stable`` and the bare ``stable`` statement.

``[Stable-Statement]``, ``dev/done/plan-stable-statement.md``. Every oracle is
exact or closed form except where scipy is named: the closure algebra is
checked through the characteristic function, the generalized central limit
theorem through its worked example, and scipy only in the body of the law,
where it is reliable.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.stats as ss

from aggregate import build
from aggregate.stable import Stable

#: A grid of t including 0, where the alpha = 1 logarithm needs care.
T = np.linspace(-3, 3, 13)


def _params(s):
    """The S1 parameters of a ``Stable`` as an array."""
    return np.array([s.alpha, s.beta, s.scale, s.loc])


def _same_law(s, t, rtol=1e-12):
    """Equal S1 parameters, the location to ``rtol`` of the scale."""
    assert np.allclose(_params(s)[:3], _params(t)[:3], rtol=rtol)
    assert abs(s.loc - t.loc) <= rtol * max(abs(s.loc), s.scale)


# ----------------------------------------------------------------------
# The class: closure, parameterizations, closed forms
# ----------------------------------------------------------------------

@pytest.mark.parametrize('alpha', [0.7, 1, 1.5, 1.9])
def test_sum_is_the_product_of_characteristic_functions(alpha):
    """Nolan Prop. 1.3: scales add as gamma ** alpha, locations add."""
    x1 = Stable(alpha, 0.6, 1.3, -0.4)
    x2 = Stable(alpha, -0.9, 0.7, 2.5)
    assert np.allclose((x1 + x2).chf(T), x1.chf(T) * x2.chf(T), rtol=1e-12, atol=0)


@pytest.mark.parametrize('alpha', [0.7, 1, 1.5, 1.9])
@pytest.mark.parametrize('a', [-2.5, 0.3, 4])
def test_scaling_matches_the_characteristic_function(alpha, a):
    """Nolan Prop. 1.4, including the extra location term at alpha = 1."""
    x = Stable(alpha, 0.4, 1.7, 0.8)
    assert np.allclose((a * x).chf(T), x.chf(a * T), rtol=1e-12, atol=0)
    assert np.allclose((x * a).chf(T), x.chf(a * T), rtol=1e-12, atol=0)


@pytest.mark.parametrize('alpha', [0.7, 1, 1.5, 1.9])
def test_negation_reflects_the_characteristic_function(alpha):
    x = Stable(alpha, 0.4, 1.7, 0.8)
    assert np.allclose((-x).chf(T), x.chf(-T), rtol=1e-12, atol=0)


def test_shift_subtraction_and_sum():
    """A number shifts; ``sum()`` works because ``0 + X`` is ``X``."""
    x, y = Stable(1.5, 1, 2, 1), Stable(1.5, -0.5, 1, 3)
    _same_law(x + 2, Stable(1.5, 1, 2, 3))
    _same_law(2 + x, Stable(1.5, 1, 2, 3))
    _same_law(5 - x, -x + 5)
    _same_law(x - y, x + (-y))
    _same_law(sum([x, y]), x + y)


def test_algebra_refusals():
    with pytest.raises(ValueError, match='common alpha'):
        Stable(1.5, 0) + Stable(1.2, 0)
    with pytest.raises(ValueError, match='zero'):
        0 * Stable(1.5, 0)


def test_chf_at_alpha_one_and_t_zero_is_silent():
    """The ``|t| log |t|`` term is 0 at t = 0, with no RuntimeWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert Stable(1, 0.5, 2, 1).chf(0) == 1


@pytest.mark.parametrize('alpha', [0.7, 1, 1.5, 2])
def test_s0_s1_round_trip(alpha):
    """S0 to S1 to S0 is the identity, and the constructor reads either."""
    beta, scale, loc0 = 0.6, 1.7, -0.9
    loc1 = Stable.s0_to_s1(alpha, beta, scale, loc0)
    assert np.isclose(Stable.s1_to_s0(alpha, beta, scale, loc1), loc0, rtol=1e-12)
    s0 = Stable(alpha, beta, scale, loc0, parameterization='S0')
    _same_law(s0, Stable(alpha, beta, scale, loc1))
    assert np.isclose(s0.loc0, loc0, rtol=1e-12)


def test_fz_never_assigns_the_scipy_class_attribute():
    """``fz`` reads ``ss.levy_stable.parameterization`` and matches its loc."""
    before = ss.levy_stable.parameterization
    x = Stable(1.5, 1, 2, 0)
    x.fz
    assert ss.levy_stable.parameterization == before
    try:
        ss.levy_stable.parameterization = 'S0'
        y = Stable(1.5, 1, 2, 0)
        # the same law under either scipy setting: the S1 mean is the 2/3 quantile
        assert np.isclose(y.cdf(0), 2 / 3, rtol=1e-6)
    finally:
        ss.levy_stable.parameterization = before
    assert np.isclose(x.cdf(0), 2 / 3, rtol=1e-6)


@pytest.mark.parametrize('alpha', [0.8, 1.5])
def test_scipy_in_the_body(alpha):
    """``cdf`` is scipy's, with the S1 parameters handed over unchanged."""
    x = Stable(alpha, 0.5, 2, 1)
    xs = x.ppf(np.linspace(0.01, 0.99, 5))
    ref = ss.levy_stable.cdf(xs, alpha, 0.5, loc=1, scale=2)
    assert np.allclose(x.cdf(xs), ref, rtol=1e-8)


def test_tail_constant_against_scipy():
    """At x = 100 scipy is still accurate, and agrees with the asymptotic."""
    x = Stable(1.5, 1)
    assert np.isclose(x.tail_constant * 100 ** -1.5, x.sf(100), rtol=1e-4)
    assert x.left_tail_constant == 0


def test_closed_form_facts():
    x = Stable(1.5, 1, 2, 0)
    assert np.isclose(x.prob_above_loc, 1 / 3, rtol=1e-12)
    assert x.mean == 0
    assert Stable(0.5, 1).mean == np.inf
    assert Stable(0.5, -1).mean == -np.inf
    assert np.isnan(Stable(0.5, 0.3).mean)
    assert np.isclose(Stable(0.5, 1).prob_above_loc, 1, rtol=1e-12)
    assert np.isnan(Stable(1, 0.5).prob_above_loc)
    assert Stable(1, 0).prob_above_loc == 0.5


def test_alpha_two_is_the_normal_with_variance_two_scale_squared():
    x = Stable(2, 0.7, 3, 1)
    assert x.beta == 0
    assert np.isclose(x.fz.var(), 18, rtol=1e-12)
    # the normal with mean 1 and variance 18 has chf exp(i t - 9 t^2)
    assert np.allclose(x.chf(T), np.exp(1j * T - 9 * T ** 2), rtol=1e-12)
    assert 'normal, variance 18' in str(x)


@pytest.mark.parametrize('args', [(0, 0), (2.5, 0), (1.5, 1.1), (1.5, -2),
                                  (1.5, 0, 0), (1.5, 0, -1), (1.5, 0, 1, np.inf)])
def test_constructor_refusals_name_the_parameter(args):
    with pytest.raises(ValueError, match='Stable: (alpha|beta|scale|loc)'):
        Stable(*args)


def test_repr_and_to_decl_are_exact():
    x = Stable(1.5, 1, 2, 0)
    assert repr(x) == 'Stable(alpha=1.5, beta=1, scale=2, loc=0)'
    y = Stable.from_compound_pareto(1000, 1.5, 100, 200)
    _same_law(build(y.to_decl()), y, rtol=0)


# ----------------------------------------------------------------------
# The generalized central limit theorem for a Pareto book
# ----------------------------------------------------------------------

def test_limit_theorem_worked_example():
    """``stable 1000 claims sev 100 * lomax 1.5``: mean and right tail match."""
    x = build('stable 1000 claims sev 100 * lomax 1.5')
    assert isinstance(x, Stable)
    assert np.isclose(x.scale, 18452.701486, rtol=1e-9)
    assert np.isclose(x.loc, 200000, rtol=1e-12)
    assert x.beta == 1
    assert np.isclose(x.loc0, 181547.30, rtol=1e-7)
    # the right tail constant is lambda * theta ** alpha
    assert np.isclose(x.tail_constant, 1000 * 100 ** 1.5, rtol=1e-12)
    _same_law(build('stable 1000 claims sev 100 * pareto 1.5 - 100'), x)


@pytest.mark.parametrize('program', [
    'stable 1000 claims sev 100 * lomax 0.8',
    'stable 1000 claims sev 100 * lomax 2.5',
    'stable 1000 claims sev lognorm 10 cv 1',
    'stable 1000 claims sev 100 * lomax [1.5 1.6] wts [.5 .5]',
    'stable 1000 claims sev -100 * lomax 1.5',
    'stable 1000 claims sev 100 * lomax 1.5 !',
    'stable 0 claims sev 100 * lomax 1.5',
])
def test_limit_theorem_refusals(program):
    with pytest.raises(ValueError, match='stable: '):
        build(program)


# ----------------------------------------------------------------------
# The statement
# ----------------------------------------------------------------------

def test_statement_forms():
    """S1 by default, ``s0`` reads the location in S0, defaults scale 1 loc 0."""
    x = build('stable 1.5 1 2 0')
    assert isinstance(x, Stable)
    assert np.isclose(x.loc0, -2, rtol=1e-12)
    _same_law(build('stable 1.5 1 2 -2 s0'), x)
    _same_law(build('stable 1.5 1 2 -2 S0'), x)
    _same_law(build('stable 1.5 1 2 0 s1'), x)
    _same_law(build('stable 1.5 1'), Stable(1.5, 1, 1, 0))
    _same_law(build('stable 1.5 1 2'), Stable(1.5, 1, 2, 0))


@pytest.mark.parametrize('program', ['stable 1.5 1 2 0', 'stable 1 0.5 2 1',
                                     'stable 0.7 -1', 'stable 2 0 1 3',
                                     'stable 1000 claims sev 100 * lomax 1.5'])
def test_statement_string_is_ascii_and_pastes_back(program):
    x = build(program)
    text = str(x)
    text.encode('ascii')
    s1, s0 = text.split(', ')[0].split(' = ')
    for spelling in (s1, s0):
        _same_law(build(spelling), x, rtol=1e-5)


def test_statement_string_worked_example():
    assert str(build('stable 1.5 1 2 0')) == (
        'stable 1.5 1 2 0 = stable 1.5 1 2 -2 s0, mean 0, Pr(X>mean) 0.333333, '
        'Pr(X>x) ~ 1.12838 x^-1.5')


@pytest.mark.parametrize('program', ['stable 2.5 0', 'stable 0 0', 'stable 1.5 1.5',
                                     'stable 1.5 1 0', 'stable 1.5',
                                     'stable 1.5 1 2 0 5', 'stable 1.5 1 2 0 s2'])
def test_statement_refusals(program):
    with pytest.raises(ValueError, match='(?i)stable'):
        build(program)


def test_stable_is_a_reserved_word():
    with pytest.raises(ValueError, match="Unexpected 'stable'"):
        build('agg stable 1 claims dsev [1] fixed')


def test_statement_stores_nothing():
    before = set(build._recipes)
    rv = build.build_many('stable 1.5 1 2 0')
    assert set(build._recipes) == before
    assert len(rv) == 1 and rv[0].kind == 'conversion'
    assert isinstance(rv[0].object, Stable)
