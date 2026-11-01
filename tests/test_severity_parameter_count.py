"""A ``sev`` clause gives a family exactly the numbers it takes ([Severity-Parameter-Count]).

Positional numbers are a family's shapes, in scipy's order. Scale and location are
the arithmetic ``c *`` and ``+ b``, and ``mean cv v`` is the moment form. Anything
else is refused rather than dropped. A vector of families shares one list, and ``_``
holds the place of a family with no shape to give.

The oracle for the zero-shape ``mean cv`` form is the frozen scipy law, because the
severity itself clamps whatever falls below zero. The oracle for ``_`` is the same
mixture written with ``gamma 1``, the exponential by another name, which reaches the
engine by a different path.

The DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg`` under the
``SP`` section.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.stats as ss

from aggregate import Aggregate, build
from aggregate._severity import Severity, severity_shape_count
from aggregate.decl_pygments import AggLexer
from aggregate.decl_writer import spec_to_decl
from aggregate.underwriter import Underwriter
from pygments.token import Token


def _parse(clause, uw=Underwriter()):
    """Parse ``agg T 1 claim sev <clause> fixed``, returning the spec."""
    return uw.parser.parse(uw.lexer.tokenize(f'agg T 1 claim sev {clause} fixed'))[2]


def _sev(clause):
    """The first severity of an unbuilt aggregate declaring ``clause``."""
    return build(f'agg T 1 claim sev {clause} fixed', update=False).sevs[0]


def _frozen_moments(clause):
    """The mean and standard deviation of the frozen law, before any clamp."""
    mean, var = _sev(clause).fz.stats(moments='mv')
    return float(mean), float(var) ** 0.5


# ----- refusals at parse time, each with the spelling to use instead -------

@pytest.mark.parametrize('clause, fragment', [
    ('norm 100 30', "'30 * norm + 100'"),
    ('norm 100 30', "'norm 100 cv 0.3'"),
    ('norm -100 30', "'30 * norm - 100'"),
    ('expon 5', "'5 * expon'"),
    ('expon 5', "'expon + 5'"),
    ('uniform 0 10', "'10 * uniform'"),
    ('gamma 4 25', "'25 * gamma 4'"),
    ('gamma 4 25', "'gamma 4 cv 25'"),
    ('pareto 1.3 0', "'pareto 1.3'"),
    ('cantor 0.5 0.1', "'0.1 * cantor 0.5'"),
    ('beta 2', 'takes two shape parameters, got one'),
    ('beta', 'takes two shape parameters, got none'),
    ('burr 2', 'takes two shape parameters, got one'),
    ('gamma', 'takes one shape parameter, got none'),
    ('genhyperbolic 1 2', 'takes three shape parameters, got two'),
    ('burr 100 cv 0.5', 'burr has no mean cv form'),
    ('cauchy 10 cv 1', 'cauchy has no finite mean'),
    ('levy 10 cv 1', 'levy has no finite mean'),
])
def test_refused_at_parse_time(clause, fragment):
    with pytest.raises(ValueError) as e:
        _parse(clause)
    assert fragment in str(e.value)


@pytest.mark.parametrize('clause, spelling', [
    ('[lognorm expon] [0.5 1]', '[lognorm expon] [0.5 _]'),
    ('[lognorm norm] [0.5 17]', '[lognorm norm] [0.5 _]'),
    ('[expon pareto] [1 2.5]', '[expon pareto] [_ 2.5]'),
    ('[lognorm expon] 0.5', '[lognorm expon] [0.5 _]'),
    ('[lognorm beta] [0.5 2] [1 3]', '[lognorm beta] [0.5 2] [_ 3]'),
    ('[norm beta] [1 2] [3 4]', '[norm beta] [_ 2] [_ 4]'),
    ('[lognorm gamma] [0.5 1] [2 3]', '[lognorm gamma] [0.5 1]'),
    ('[norm expon] [1 2]', '[norm expon]'),
])
def test_vector_with_a_number_a_family_cannot_take_is_refused(clause, spelling):
    with pytest.raises(ValueError) as e:
        _parse(clause)
    assert f"'{spelling}'" in str(e.value)
    _parse(spelling)


@pytest.mark.parametrize('clause, fragment', [
    ('[lognorm lognorm] [0.5 _]', 'lognorm takes one shape parameter, got none'),
    ('[lognorm beta] [0.5 2]', 'beta takes two shape parameters, got one'),
    ('[lognorm beta] [0.5 2] [3 _]', 'beta takes two shape parameters, got one'),
    ('lognorm [0.5 _]', 'stands alone'),
    ('[lognorm] [_]', 'stands alone'),
    ('[lognorm norm] [0.5 _ 3]', 'one entry per family'),
    ('[lognorm gamma] [0.5 1 2]', 'one entry per family'),
    ('norm _', "Unexpected character '_'"),
])
def test_misplaced_placeholder_is_refused(clause, fragment):
    with pytest.raises(ValueError) as e:
        _parse(clause)
    assert fragment in str(e.value)


@pytest.mark.parametrize('program', [
    'agg T _ claims sev expon fixed',
    'agg T 1 claim _ xs 0 sev expon fixed',
    'agg T 1 claim sev [1 _] * expon fixed',
    'agg T 1 claim sev [lognorm gamma] [1 2] wts [_ 1] fixed',
    'agg T 1 claim sev gamma _ cv 0.5 fixed',
    'agg T 1 claim sev [expon gamma] [_ 100] cv 0.5 fixed',
])
def test_placeholder_reaches_no_other_slot(program):
    uw = Underwriter()
    with pytest.raises(ValueError, match="DecL parse error"):
        uw.parser.parse(uw.lexer.tokenize(program))


# ----- each suggested spelling builds the law the message names -------------

@pytest.mark.parametrize('clause, mean, sd', [
    ('30 * norm + 100', 100, 30),
    ('25 * gamma 4', 100, 50),
    ('5 * expon', 5, 5),
    ('expon + 5', 6, 1),
    ('10 * uniform', 5, 10 / 12 ** 0.5),
])
def test_suggested_spelling_builds_the_named_law(clause, mean, sd):
    assert _frozen_moments(clause) == pytest.approx((mean, sd), rel=1e-12)


# ----- the zero-shape mean cv form -------------------------------------------

@pytest.mark.parametrize('clause, mean, cv, shift', [
    ('norm 100 cv 0.3', 100, 0.3, 0),
    ('2 * norm 100 cv 0.3', 200, 0.3, 0),
    ('expon 10 cv 1', 10, 1, 0),
    ('expon 10 cv 0.5', 10, 0.5, 0),
    ('expon 10 cv 0.5 + 3', 10, 0.5, 3),
    ('uniform 10 cv 0.2', 10, 0.2, 0),
    ('logistic 50 cv 0.2 - 5', 50, 0.2, -5),
    ('gumbel_r 50 cv 0.2', 50, 0.2, 0),
])
def test_zero_shape_mean_cv_has_the_declared_moments(clause, mean, cv, shift, capsys):
    frozen_mean, frozen_sd = _frozen_moments(clause)
    assert frozen_mean == pytest.approx(mean + shift, rel=1e-12)
    assert frozen_sd == pytest.approx(mean * cv, rel=1e-12)
    # _validate_moments prints when the target is missed
    assert 'WARNING' not in capsys.readouterr().out


def test_expon_mean_cv_is_the_exponential_with_that_mean():
    fz = _sev('expon 10 cv 1').fz
    assert fz.kwds['loc'] == pytest.approx(0, abs=1e-12)
    assert fz.kwds['scale'] == pytest.approx(10, rel=1e-12)
    fz = _sev('expon 10 cv 0.5').fz
    assert (fz.kwds['loc'], fz.kwds['scale']) == pytest.approx((5, 5), rel=1e-12)


def test_uniform_mean_cv_has_the_stated_support():
    lo, hi = _sev('uniform 10 cv 0.2').fz.support()
    half_width = 2 * 3 ** 0.5
    assert (lo, hi) == pytest.approx((10 - half_width, 10 + half_width), rel=1e-12)


def test_reflected_zero_shape_mean_cv():
    """``shift - X`` builds ``X`` from the mean and cv, then reflects it."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sev = _sev('500 - norm 100 cv 0.3')
    # the clamp of a reflected law routes these through numerical integration
    assert sev.actual_m == pytest.approx(400, rel=1e-7)
    assert sev.actual_sd == pytest.approx(30, rel=1e-7)


def test_mean_cv_and_arithmetic_spellings_agree():
    a = build('agg A 1 claim sev norm 1000 cv 0.1 fixed', log2=14, bs=1)
    b = build('agg B 1 claim sev 100 * norm + 1000 fixed', log2=14, bs=1)
    assert np.abs(a.density_df.p_total.values - b.density_df.p_total.values).max() < 1e-15


def test_plain_sev_clamps_what_falls_below_zero():
    """The frozen law has the declared moments; ``sev`` clamps and ``ssev`` does not."""
    mu, sigma = 100, 100
    clamped = mu * ss.norm.cdf(mu / sigma) + sigma * ss.norm.pdf(mu / sigma)
    assert _sev('norm 100 cv 1').actual_m == pytest.approx(clamped, rel=1e-6)
    signed = build('agg T 1 claim ssev norm 100 cv 1 fixed', update=False).sevs[0]
    assert signed.actual_m == pytest.approx(mu, rel=1e-9)
    assert signed.actual_sd == pytest.approx(sigma, rel=1e-9)


# ----- a mean cv form the values cannot honor is refused at build ------------

@pytest.mark.parametrize('clause, fragment', [
    ('norm -100 cv 0.3', 'needs both positive'),
    ('norm 100 cv 0', 'needs both positive'),
    ('beta 2 cv 0.2', 'beta has no law'),
    ('beta 0.5 cv 5', 'beta has no law'),
    ('t 100 cv 0.5', 'no shape found'),
    ('lomax 100 cv 0.5', 'no shape found'),
])
def test_mean_cv_the_values_cannot_honor_is_refused(clause, fragment):
    with pytest.raises(ValueError, match=fragment):
        _sev(clause)


# ----- the placeholder --------------------------------------------------------

@pytest.mark.parametrize('held, twin', [
    ('sev 100 * [lognorm expon] [.5 _] wts [0.6 .4]', 'sev 100 * [lognorm gamma] [.5 1] wts [0.6 .4]'),
    ('sev [50 100] * [lognorm expon] [2 _] + 10 wts=2', 'sev [50 100] * [lognorm gamma] [2 1] + 10 wts=2'),
    ('sev [200 200] * [expon pareto] [_ 3] + [0 -200] wts [.25 .75]',
     'sev [200 200] * [gamma pareto] [1 3] + [0 -200] wts [.25 .75]'),
    ('sev 100 * expon + 10', 'sev 100 * gamma 1 + 10'),
    ('sev 1000 * expon !', 'sev 1000 * gamma 1 !'),
])
def test_placeholder_spelling_matches_the_gamma_twin(held, twin):
    """The exponential is the gamma with shape 1, so these are one law by two paths."""
    a = build(f'agg A 8 claims {held} fixed', log2=16, bs=1)
    b = build(f'agg B 8 claims {twin} fixed', log2=16, bs=1)
    assert np.abs(a.density_df.p_total.values - b.density_df.p_total.values).max() < 1e-15


def test_placeholder_gives_a_normal_a_place_in_a_mixture():
    """The case with no twin: a normal has no one-shape family to stand in for it."""
    a = build('agg M 1 claim sev [100 30] * [lognorm norm] [0.5 _] + [0 100] wts [0.6 0.4] fixed',
              update=False)
    lognormal, normal = a.sevs
    assert normal.sev_name == 'norm'
    assert normal.fz.stats('m') == pytest.approx(100, rel=1e-12)
    assert normal.fz.stats('v') == pytest.approx(900, rel=1e-12)
    assert lognormal.fz.stats('m') == pytest.approx(100 * np.exp(0.125), rel=1e-12)


def test_placeholder_in_the_second_list():
    a = build('agg M 1 claim sev [lognorm beta] [0.5 2] [_ 3] wts=2 fixed', update=False)
    assert a.sevs[0].fz.args == pytest.approx((0.5,))
    assert a.sevs[1].fz.args == pytest.approx((2, 3))


@pytest.mark.parametrize('sev', [
    '[lognorm norm] [0.5 _]',
    '[norm lognorm] [_ 0.5]',
    '[norm expon uniform]',
    '[100 30] * [lognorm norm] [0.5 _] + [0 100] wts [0.6 0.4]',
    '[lognorm beta] [0.5 2] [_ 3]',
    '[1_000 30] * [lognorm norm] [1 _]',
])
def test_placeholder_round_trips_through_the_writer(sev):
    uw = Underwriter()
    kind, name, spec = uw.parser.parse(uw.lexer.tokenize(f'agg T 1 claim sev {sev} fixed'))
    text = spec_to_decl(spec, kind, name)
    again = uw.parser.parse(uw.lexer.tokenize(text))
    assert spec_to_decl(again[2], again[0], again[1]) == text
    assert ('_' in text) == ('_' in sev.replace('1_000', ''))


def test_placeholder_is_colored_and_a_number_keeps_its_underscore():
    tokens = [(t, v) for t, v in AggLexer().get_tokens('[0.5 _ 1_000]') if v.strip()]
    assert (Token.Punctuation, '_') in tokens
    assert (Token.Literal.Number, '1_000') in tokens
    assert not any(t is Token.Error for t, _v in tokens)


# ----- the Python API ---------------------------------------------------------

@pytest.mark.parametrize('name, kwargs, fragment', [
    ('gamma', dict(sev_a=4, sev_b=25), 'takes one shape parameter'),
    ('norm', dict(sev_a=100, sev_scale=1), 'no shape parameter'),
    ('norm', dict(sev_b=30, sev_scale=1), 'no shape parameter'),
    ('burr', dict(sev_scale=1), 'takes 2 shape parameters'),
    ('beta', dict(sev_a=2, sev_b=np.nan, sev_scale=1), 'takes 2 shape parameters'),
    ('norm', dict(sev_mean=100, sev_scale=30), 'needs both positive'),
    ('norm', dict(sev_cv=0.3, sev_loc=100), 'needs both positive'),
    ('cauchy', dict(sev_mean=10, sev_cv=1), 'no finite mean'),
])
def test_python_api_refuses(name, kwargs, fragment):
    with pytest.raises(ValueError, match=fragment):
        Severity(name, **kwargs)


@pytest.mark.parametrize('name, kwargs, mean', [
    ('gamma', dict(sev_a=4, sev_scale=25), 100),
    ('gamma', dict(sev_a=4, sev_b=0, sev_scale=25), 100),
    ('norm', dict(sev_loc=100, sev_scale=30), 100),
    ('norm', dict(sev_a=np.nan, sev_b=np.nan, sev_loc=100, sev_scale=30), 100),
    ('norm', dict(sev_mean=100, sev_cv=0.3), 100),
    ('expon', dict(sev_mean=10, sev_cv=1), 10),
    ('beta', dict(sev_mean=5, sev_cv=0.3, sev_scale=100), 5),
    ('beta', dict(sev_a=2, sev_b=3, sev_scale=100), 40),
])
def test_python_api_builds(name, kwargs, mean):
    assert Severity(name, **kwargs).fz.stats('m') == pytest.approx(mean, rel=1e-12)


def test_python_api_mixture_with_an_unset_shape():
    a = Aggregate('A', exp_en=1, sev_name=['lognorm', 'expon'], sev_a=[0.5, np.nan], sev_scale=100.,
                  sev_wt=[0.5, 0.5], freq_name='fixed')
    assert a.sevs[1].fz.stats('m') == pytest.approx(100, rel=1e-12)


# ----- the shape count ---------------------------------------------------------

@pytest.mark.parametrize('name, counts', [
    ('norm', (0, 0)),
    ('expon', (0, 0)),
    ('gamma', (1, 1)),
    ('pareto', (1, 1)),
    ('beta', (2, 2)),
    ('burr', (2, 2)),
    ('genhyperbolic', (3, 3)),
    ('cantor', (0, 1)),
    ('dhistogram', None),
    ('not_a_family', None),
    ('sem', None),
    ('rv_continuous', None),
])
def test_severity_shape_count(name, counts):
    assert severity_shape_count(name) == counts
