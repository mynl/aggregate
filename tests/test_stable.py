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
from aggregate.decl_writer import spec_to_decl
from aggregate.stable import Stable
from aggregate.underwriter import RecipeNotFound, ResolvedReference

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


# ----------------------------------------------------------------------
# Named laws ([Named-Conversion-Statements], dev/done/plan-stable-expressions.md)
# ----------------------------------------------------------------------

@pytest.fixture
def uw():
    """A fork, so nothing declared here reaches the shared ``build``."""
    return build.fork(name='stable-expressions')


NAMED = [
    ('stable x0 1.5 0', None, Stable(1.5, 0)),
    ('stable x0 1.5 0.85 2', None, Stable(1.5, 0.85, 2)),
    ('stable x0 1.5 1 2 -3', None, Stable(1.5, 1, 2, -3)),
    ('stable x0 1.5 1 2 -2 s0', None, Stable(1.5, 1, 2, -2, 'S0')),
    ('stable x0 as "My law" 1.5 0', 'My law', Stable(1.5, 0)),
    ('stable x0 as lab 0.7 -1', 'lab', Stable(0.7, -1)),
    ('stable x0 1000 claims sev 100 * lomax 1.5', None,
     Stable.from_compound_pareto(1000, 1.5, 100, 200)),
]


@pytest.mark.parametrize('program, label, law', NAMED)
def test_named_statement_stores_four_numbers_under_its_kind_and_name(uw, program, label, law):
    x = uw(program)
    assert isinstance(x, Stable)
    assert (x.name, x._label, x.program) == ('x0', label, program)
    assert x.label == (label or x._label_default())
    _same_law(x, law)
    spec = uw._recipes[('stable', 'x0')].spec
    numbers = [spec[k] for k in ('alpha', 'beta', 'scale', 'loc')]
    assert all(type(v) is float for v in numbers)
    assert numbers == [x.alpha, x.beta, x.scale, x.loc]


def test_named_statement_keeps_its_trailer(uw):
    uw('stable x0 1.5 0 note{a note} tags{a:b} hints{log2=8}')
    rec = uw.recipe('x0')
    assert (rec.kind, rec.note, rec.tags) == ('stable', 'a note', ('a:b',))
    assert rec.spec['hints'] == 'log2=8'


def test_named_s0_agrees_with_the_anonymous_form(uw):
    _same_law(uw('stable x0 1.5 1 2 -2 s0'), build('stable 1.5 1 2 -2 s0'), rtol=0)


@pytest.mark.parametrize('program', ['x0', 'stable.x0'])
def test_bare_name_and_reference_return_the_named_object(uw, program):
    uw('stable x0 as "My law" 1.5 1 2 3')
    x = uw(program)
    assert isinstance(x, Stable)
    assert (x.name, x._label) == ('x0', 'My law')
    assert np.allclose(_params(x), (1.5, 1, 2, 3), rtol=1e-15, atol=0)


def test_a_reference_is_an_answer_and_stores_nothing(uw):
    uw('stable x0 1.5 0')
    before = set(uw._recipes)
    rv = uw.build_many('stable.x0')
    assert len(rv) == 1 and rv[0].kind == 'conversion'
    assert set(uw._recipes) == before


@pytest.mark.parametrize('program', [p for p, _, _ in NAMED]
                         + ['stable x0 1 0', 'stable x0 as "My law" 1.5 1 2 -3 note{n} tags{a:b}'])
def test_stored_spec_is_an_unparser_fixed_point(uw, program):
    uw.build_many(program)
    spec = uw._recipes[('stable', 'x0')].spec
    text = spec_to_decl(spec, 'stable')
    assert uw.parser.parse(uw.lexer.tokenize(text)) == ('stable', 'x0', spec)


def test_writer_omits_a_default_location_and_scale(uw):
    uw.build_many('stable a 1.5 0.5 1 0\n\nstable b 1.5 0.5 2 0\n\nstable c 1.5 0.5 1 3')
    render = lambda name: spec_to_decl(uw._recipes[('stable', name)].spec, 'stable')  # noqa: E731
    assert render('a') == 'stable a 1.5 0.5'
    assert render('b') == 'stable b 1.5 0.5 2'
    assert render('c') == 'stable c 1.5 0.5 1 3'


def test_to_agg_then_load_round_trips_into_a_fresh_fork(uw, tmp_path):
    uw.build_many('stable x0 as "My law" 1.5 1 2 -3 note{n}\n\ntweedie t0 as T 1.5 10 2\n\n'
                  'agg A dfreq [2] dsev [1]')
    out = uw.to_agg(tmp_path / 'laws')
    fresh = build.fork(name='fresh')
    fresh.load(out)
    for key in (('stable', 'x0'), ('tweedie', 't0')):
        assert fresh._recipes[key].spec == uw._recipes[key].spec


def test_to_decl_declares_a_named_law_again_and_is_unchanged_otherwise(uw):
    x = uw('stable x0 as "My law" 1.5 1 2 -3')
    assert x.to_decl() == 'stable x0 as "My law" 1.5 1 2 -3'
    assert uw.build_many(x.to_decl())[0].spec == uw._recipes[('stable', 'x0')].spec
    assert Stable(1.5, 1, 2, -3).to_decl() == 'stable 1.5 1 2 -3'


def test_repr_of_a_named_law_is_constructor_valid(uw):
    x = uw('stable x0 as "My law" 1.5 1 2 -3')
    assert repr(x) == "Stable(alpha=1.5, beta=1, scale=2, loc=-3, name='x0', label='My law')"
    y = eval(repr(x))
    assert (y.name, y._label) == (x.name, x._label)
    _same_law(y, x, rtol=0)


def test_named_string_leads_with_the_name_and_each_spelling_builds_back(uw):
    text = str(uw('stable x0 1.5 0'))
    assert text == ('x0: stable 1.5 0 1 0 = stable 1.5 0 1 0 s0, mean 0, Pr(X>mean) 0.5, '
                    'Pr(X>x) ~ 0.199471 x^-1.5, Pr(X<-x) ~ 0.199471 x^-1.5')
    s1, s0 = text.split(': ', 1)[1].split(', ')[0].split(' = ')
    for spelling in (s1, s0):
        _same_law(build(spelling), Stable(1.5, 0), rtol=1e-5)
    assert str(uw('stable y as "My law" 1.5 0')).startswith('My law (y): stable 1.5 0 1 0 =')


def test_anonymous_statement_has_no_name_and_no_lead():
    x = build('stable 1.5 1 2 0')
    assert x.name is None and x._label is None and x.program == ''
    assert x.label == 'stable 1.5 1 2 0'
    assert str(x).startswith('stable 1.5 1 2 0 =')
    assert repr(x) == 'Stable(alpha=1.5, beta=1, scale=2, loc=0)'


def test_a_name_in_a_fork_is_not_visible_in_its_parent(uw):
    child = uw.fork(name='child')
    child('stable x0 1.5 0')
    assert ('stable', 'x0') not in uw._recipes
    with pytest.raises(RecipeNotFound):
        uw('stable.x0')


def test_preview_reports_a_reference_and_registers_nothing(uw):
    uw('stable x0 1.5 0')
    before = set(uw._recipes)
    pv = uw.preview('stable.x0')
    assert pv.resolved == (ResolvedReference('stable', 'x0', 'session'),)
    assert set(uw._recipes) == before


@pytest.mark.parametrize('program', ['3 * stable 1.5 0', 'x0 = stable 1.5 0',
                                     'chfagg C stable.x0', 'agg stable.x0 dfreq [1] dsev [1]'])
def test_named_law_refusals_are_parse_errors(uw, program):
    uw('stable x0 1.5 0')
    with pytest.raises(ValueError) as e:
        uw(program)
    assert hasattr(e.value, 'report')


def test_reference_to_nothing_is_recipe_not_found(uw):
    with pytest.raises(RecipeNotFound) as e:
        uw('stable.nope')
    assert (e.value.kind, e.value.name) == ('stable', 'nope')


@pytest.mark.parametrize('name, psi', [
    ('Cauchy', lambda t: -np.abs(t)),
    ('Levy', lambda t: -np.abs(t) ** 0.5 * (1 - 1j * np.sign(t))),
    ('Landau', lambda t: -np.abs(t) * (1 + 2j / np.pi * np.sign(t)
                                       * np.log(np.abs(t), where=t != 0,
                                                out=np.zeros_like(t)))),
    ('Holtsmark', lambda t: -np.abs(t) ** 1.5),
])
def test_library_laws_have_the_classical_characteristic_functions(name, psi):
    x = build(f'stable.{name}')
    assert x.name == name
    assert np.allclose(x.chf(T), np.exp(psi(T)), rtol=1e-12, atol=0)
    assert isinstance(build(name), Stable)


# ----------------------------------------------------------------------
# Arithmetic on references ([Stable-Arithmetic], dev/done/plan-stable-expressions.md)
# ----------------------------------------------------------------------

@pytest.fixture
def laws(uw):
    """The three laws of the plan's Goal, and the two of ``OPERATOR_FORMS``."""
    uw.build_many('stable x0 1.5 0\n\nstable x1 1.5 0.85\n\nstable x2 1.5 -0.5\n\n'
                  'stable x 1.5 0.6 1.3 -0.4\n\nstable y 1.5 -0.9 0.7 2.5')
    return uw


@pytest.mark.parametrize('program, expected', [
    ('3 * stable.x0', (1.5, 0, 3, 0)),
    ('stable.x1 + stable.x2', (1.5, 0.175, 1.5874010519681994, 0)),
    ('stable.x0 + 2 * stable.x1 + 3', (1.5, 0.6279767062808198, 2.4472608147714756, 3)),
])
def test_the_goal_expressions_return_their_measured_values(laws, program, expected):
    assert np.allclose(_params(laws(program)), expected, rtol=1e-14, atol=0)


X, Y = Stable(1.5, 0.6, 1.3, -0.4), Stable(1.5, -0.9, 0.7, 2.5)

OPERATOR_FORMS = [
    ('stable.x * 3', X * 3),
    ('stable.x / 3', X / 3),
    ('3 + stable.x', 3 + X),
    ('3 - stable.x', 3 - X),
    ('stable.x - 3', X - 3),
    ('stable.x -3', X - 3),
    ('stable.x - -3', X + 3),
    ('-stable.x', -X),
    ('- stable.x', -X),
    ('stable.x - stable.y', X - Y),
    ('stable.x -stable.y', X - Y),
    ('stable.x - -stable.y', X - (-Y)),
    ('2 * 3 * stable.x', 2 * (3 * X)),
    ('2 * stable.x * 3', 2 * (X * 3)),
    ('2 * (stable.x + stable.y) - 1', 2 * (X + Y) - 1),
    ('(1/2) * stable.x', 0.5 * X),
    ('1/2 * stable.x', 0.5 * X),
    ('(1+2) * stable.x', 3 * X),
    ('-2 * stable.x', -2 * X),
    ('exp(0) * stable.x', X),
    ('50% * stable.x', 0.5 * X),
    ('stable.x / 4 * 2', X / 4 * 2),
    ('-(stable.x + 1)', -(X + 1)),
    ('(stable.x)', X),
    ('stable.x + 1 + 2', X + 3),
    ('1 + stable.x + 2', X + 3),
    ('1 - stable.x - 2', 1 - X - 2),
    ('stable.x - 3 * stable.y', X - 3 * Y),
    ('stable.x -3 * stable.y', X - 3 * Y),
    ('2 @ stable.x', 2 @ X),
    ('1/2 @ stable.x', 0.5 @ X),
    ('3 * 2 @ stable.x', 3 * (2 @ X)),
    ('2 @ 3 * stable.x', 2 @ (3 * X)),
    ('stable.x + 2 @ stable.y + 1', X + (2 @ Y) + 1),
]


@pytest.mark.parametrize('program, law', OPERATOR_FORMS, ids=[p for p, _ in OPERATOR_FORMS])
def test_every_operator_form_agrees_with_python(laws, program, law):
    _same_law(laws(program), law, rtol=1e-14)


@pytest.mark.parametrize('alpha', [0.7, 1, 1.5])
def test_combination_has_the_product_of_characteristic_functions(uw, alpha):
    """The oracle through the DecL path, which pins the alpha = 1 logarithm."""
    a, b = Stable(alpha, 0.6, 1.3, -0.4), Stable(alpha, -0.9, 0.7, 2.5)
    uw.build_many(f'stable a {alpha} 0.6 1.3 -0.4\n\nstable b {alpha} -0.9 0.7 2.5')
    z = uw('2 * stable.a + stable.b - 3 * stable.a + 1')
    expected = a.chf(2 * T) * b.chf(T) * a.chf(-3 * T) * np.exp(1j * T)
    assert np.allclose(z.chf(T), expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize('alpha', [0.7, 1, 1.5])
def test_each_occurrence_of_a_reference_is_an_independent_copy(uw, alpha):
    uw(f'stable x {alpha} 0.6')
    assert np.isclose(uw('stable.x + stable.x').scale, 2 ** (1 / alpha), rtol=1e-14)
    assert uw('2 * stable.x').scale == 2
    difference = uw('stable.x - stable.x')
    assert difference.loc == 0 and difference.beta == 0
    assert np.isclose(difference.scale, 2 ** (1 / alpha), rtol=1e-14)


@pytest.mark.parametrize('alpha', [0.7, 1, 1.5, 1.9])
def test_copies_operator_is_the_sum_and_the_power_of_the_characteristic_function(alpha):
    """``t @ X`` has characteristic function phi ** t, at alpha = 1 too."""
    x = Stable(alpha, 0.6, 1.3, -0.4)
    _same_law(2 @ x, x + x)
    _same_law(3 @ x, x + x + x)
    # the semigroup at a time that is not an integer
    _same_law(0.5 @ x + 0.5 @ x, x)
    _same_law(0.25 @ (4 @ x), x)
    assert np.allclose((3 @ x).chf(T), x.chf(T) ** 3, rtol=1e-12, atol=0)
    assert (2 @ x).beta == x.beta


@pytest.mark.parametrize('t', [0, -1])
def test_copies_operator_refuses_a_count_that_is_not_positive(laws, t):
    with pytest.raises(ValueError, match='independent copies'):
        laws(f'{t} @ stable.x')


def test_limit_law_of_two_units():
    """The sum of two Pareto limits has the mean and right tail of the two books."""
    uw = build.fork(name='limits')
    uw.build_many('stable A 1000 claims sev 100 * lomax 1.5\n\n'
                  'stable B 400 claims sev 250 * lomax 1.5')
    total = uw('stable.A + stable.B')
    assert total.beta == 1
    assert np.isclose(total.mean, 400000, rtol=1e-12)
    assert np.isclose(total.scale, 34721.690825521684, rtol=1e-12)
    assert np.isclose(total.tail_constant, 1000 * 100 ** 1.5 + 400 * 250 ** 1.5, rtol=1e-12)
    net = uw('stable.A - stable.B')
    assert np.isclose(net.beta, -0.22514822655441366, rtol=1e-12)
    assert net.loc == 0


def test_a_result_is_anonymous_stored_nowhere_and_of_kind_conversion(laws):
    before = set(laws._recipes)
    rv = laws.build_many('stable.x0 + 1')
    assert len(rv) == 1 and rv[0].kind == 'conversion'
    assert rv[0].object.name is None and rv[0].object._label is None
    assert set(laws._recipes) == before


def test_a_lone_reference_keeps_its_name_and_parentheses_do_too(uw):
    uw('stable x0 as "My law" 1.5 0')
    assert uw('stable.x0').name == 'x0'
    assert uw('(stable.x0)').name == 'x0'


def test_stable_name_expression_stores_four_numbers_and_the_program(laws):
    program = 'stable z stable.x0 + 2 * stable.x1 + 3'
    z = laws(program)
    assert (z.name, z.program) == ('z', program)
    spec = laws._recipes[('stable', 'z')].spec
    assert np.allclose([spec[k] for k in ('alpha', 'beta', 'scale', 'loc')],
                       (1.5, 0.6279767062808198, 2.4472608147714756, 3), rtol=1e-14, atol=0)
    assert all(type(spec[k]) is float for k in ('alpha', 'beta', 'scale', 'loc'))
    # bound at parse time, like every dotted reference other than sev agg.NAME
    laws('stable x0 1.5 0 9')
    assert laws('stable.z').scale == z.scale


def test_stable_name_expression_keeps_its_own_label_and_note(laws):
    laws('stable z as Combo stable.x - stable.y note{net}')
    rec = laws.recipe('z')
    assert rec.spec['label'] == 'Combo' and rec.note == 'net'
    assert str(laws('z')).startswith('Combo (z): ')


def test_stable_name_reference_is_a_copy_under_the_new_name(uw):
    uw('stable x0 as "My law" 1.5 0 note{first}')
    z = uw('stable z stable.x0')
    assert (z.name, z._label) == ('z', None)
    assert uw.recipe('z').note == ''
    _same_law(z, uw('stable.x0'), rtol=0)


def test_a_stored_expression_is_an_unparser_fixed_point(laws):
    laws.build_many('stable z as Combo stable.x0 + 2 * stable.x1 - 3 note{n}')
    spec = laws._recipes[('stable', 'z')].spec
    assert laws.parser.parse(laws.lexer.tokenize(spec_to_decl(spec, 'stable'))) == ('stable', 'z', spec)


@pytest.mark.parametrize('program, match', [
    ('stable.x + stable.other', 'common alpha'),
    ('0 * stable.x', 'zero'),
    ('stable.x * 0', 'zero'),
    ('stable.x / 0', 'zero'),
])
def test_arithmetic_refusals(laws, program, match):
    laws('stable other 1.2 0')
    with pytest.raises(ValueError, match=match):
        laws(program)


def test_reference_to_nothing_inside_an_expression(laws):
    with pytest.raises(RecipeNotFound) as e:
        laws('stable.x + stable.nope')
    assert (e.value.kind, e.value.name) == ('stable', 'nope')


@pytest.mark.parametrize('program', ['stable.x0 + tweedie.t0', 'stable.x + 2 * 3', '3 @ tweedie.t0',
                                     'stable.x + stable 1.5 0'])
def test_arithmetic_parse_errors(laws, program):
    laws('tweedie t0 1.5 10 2')
    with pytest.raises(ValueError) as e:
        laws(program)
    assert hasattr(e.value, 'report')


def test_preview_of_an_expression_reports_each_reference_and_registers_nothing(laws):
    before = set(laws._recipes)
    pv = laws.preview('stable.x0 + 2 * stable.x1')
    assert pv.resolved == (ResolvedReference('stable', 'x0', 'session'),
                           ResolvedReference('stable', 'x1', 'session'))
    assert set(laws._recipes) == before


@pytest.mark.parametrize('program, kind, name, spaced', [
    ('stable.x1-stable.x2', 'stable', 'x1-stable.x2', 'stable.x1 - stable.x2'),
    ('agg.A-1', 'agg', 'A-1', 'agg.A - 1'),
])
def test_minus_in_a_reference_has_a_hint(laws, program, kind, name, spaced):
    with pytest.raises(RecipeNotFound) as e:
        laws(program)
    assert (e.value.kind, e.value.name) == (kind, name)
    assert f'no recipe named {name!r} of kind {kind!r}' in e.value.message
    assert f'To subtract, write spaces: {spaced}' in e.value.message


def test_a_dashed_name_that_exists_still_resolves(uw):
    uw('stable a-b 1.5 0')
    assert uw('stable.a-b').name == 'a-b'
    assert uw('stable.a-b + 1').loc == 1


def test_division_agrees_with_multiplication():
    x = Stable(1, 0.6, 1.3, -0.4)
    _same_law(x / 4, x * 0.25, rtol=0)
    assert Stable(1.5, 0).__truediv__('a') is NotImplemented
    assert Stable(1.5, 0).__rmatmul__('a') is NotImplemented
