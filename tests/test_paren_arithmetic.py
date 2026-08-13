"""Full arithmetic inside parentheses in DecL expressions
([DecL-Paren-Arithmetic], 1.0.0a268).

DecL has always had an expression sub-language (``/``, ``**``, ``^``, ``exp``,
parentheses) evaluated eagerly in the transformer. This adds ``+``, ``-`` and
``*``, **only inside parentheses**, so a user can write a computed exposure, a
grossed-up premium, or a lognormal mean normalizer inline:

.. code-block:: text

    agg TEST (4 + 3*2) claims (100_000/(1-.25)) premium 1000 xs 0
        sev (exp(-1 * .4**2/2)) * lognorm .4 poisson

The paren gate is load bearing, not a convenience. ``PLUS`` / ``MINUS`` /
``TIMES`` already carry meaning at other levels (``sev2`` shift arithmetic,
``sev1`` scaling, the whole ``builtin_agg`` algebra), and ``NUMBER`` absorbs a
glued leading minus, so admitting them into bare ``expr`` would create genuine
Earley ambiguities and destabilize vector lexing. Inside parentheses none of
that exists: a paren holds exactly one expression, never a list, so the only
viable reading of ``(1-.25)`` is the subtraction.

Three behaviors are documented rather than fixed, and pinned here:

1. A minus glued to a literal is part of the literal and binds tighter than
   ``**``, so ``(-.4**2)`` is +0.16 where Python's ``-.4**2`` is -0.16. Negate
   a computed value with ``(-1 * x)`` or ``(0 - x)``.
2. Parentheses hold one expression, never a list: ``(1 -2)`` is -1, while the
   bracket list ``[1 -2]`` keeps its two-element reading.
3. A percentage literal keeps its ``%`` reading only when used literally; any
   arithmetic degrades it to a plain float, which matters in the ``po``
   placement position.

Companion tests: ``tests/test_grammar_ambiguity.py`` holds that the paren
island adds no parse forest branches. The DecL programs here are mirrored in
``src/aggregate/agg/decl-testers.agg`` section PA (this module is the canonical
source).
"""
import math

import numpy as np
import pytest

from aggregate import build
from aggregate.decl_writer import format_program


TOL = 1e-12


def _value(program):
    """Evaluate a bare DecL expression (the ``answer_expr`` route)."""
    return float(build(program))


# ----------------------------------------------------------------------
# [Evaluation-Pins]: the operators, the ladder, the associativity
# ----------------------------------------------------------------------
@pytest.mark.parametrize('program, expected', [
    # the new operators
    ('(4 + 3*2)', 10.0),
    ('(4 -3)', 1.0),
    ('(2*3)', 6.0),
    # precedence: times binds tighter than plus, parens override
    ('(2+3*4)', 14.0),
    ('((2+3)*4)', 20.0),
    # exponent still binds tighter than times, and stays right-associative
    ('(2**3)', 8.0),
    ('(2**3**2)', 512.0),
    ('(2*3**2)', 18.0),
    # division is left-associative and shares a level with times
    ('(1-.25)', 0.75),
    ('(100_000/(1-.25))', 400000 / 3),
    ('(8/4/2)', 1.0),
    ('(2*8/4)', 4.0),
    # the glued minus makes these one token, so both are subtraction
    ('(1 -2)', -1.0),
    ('(2 - -3)', 5.0),
    # exp composes with the new levels for free
    ('exp(-1 * .4**2/2)', math.exp(-0.08)),
    ('(0 - .4**2/2)', -0.08),
    # the documented quirk: a glued minus binds tighter than **
    ('(-.4**2)', 0.16),
])
def test_expression_value(program, expected):
    assert abs(_value(program) - expected) < TOL * max(1.0, abs(expected))


def test_percent_degrades_under_arithmetic():
    """``_PercentNumber`` survives literal use only, by existing design.

    The percent reading is a property of the literal token, not of the value,
    so any arithmetic returns a plain float. ``(50% + 1)`` is 1.5, and a
    computed value in the ``po`` placement position therefore reads as an
    absolute amount rather than a share.
    """
    assert abs(_value('(50% + 1)') - 1.5) < TOL
    # the bare literal keeps its percent identity
    assert abs(_value('50%') - 0.5) < TOL


# ----------------------------------------------------------------------
# [Gate-Negatives]: what stays a parse error
# ----------------------------------------------------------------------
@pytest.mark.parametrize('program', [
    # the gate itself: bare expressions never see + - *
    'agg PaGate 4 + 3*2 claims dsev [1] fixed',
    # brackets are lists, so a spaced minus has no derivation there
    'agg PaVector dfreq [1] dsev [1 - 2]',
    # parentheses hold one expression, never a list
    '(1 2)',
    # a dangling operator
    '(1 +)',
])
def test_parse_error(program):
    with pytest.raises(ValueError):
        build(program)


def test_bracket_list_lexing_is_unchanged():
    """``[1 -2]`` is still the two-element list, not a subtraction.

    This is the reason for the paren gate stated as a test: the vector reading
    depends on ``NUMBER`` absorbing the glued minus, and on bare ``expr`` never
    admitting ``MINUS``.
    """
    a = build('agg PaVec dfreq [1] dsev [1 -2]', update=False)
    assert np.allclose(a.spec['sev_xs'], [1.0, -2.0])
    assert len(a.spec['sev_ps']) == 2


# ----------------------------------------------------------------------
# The target example: paren math in every slot of one program
# ----------------------------------------------------------------------
def test_target_example_builds():
    """Computed exposure, computed FYI premium, computed severity scale."""
    a = build('agg PaTarget (4 + 3*2) claims (100_000/(1-.25)) premium '
              '1000 xs 0 sev (exp(-1 * .4**2/2)) * lognorm .4 poisson',
              update=False)
    assert a.n == 10
    assert abs(a.exp_premium - 400000 / 3) < 1e-9
    assert abs(a.spec['sev_scale'] - math.exp(-0.08)) < TOL


@pytest.mark.parametrize('program, key, expected', [
    ('agg PaFreq 10 claims dsev [1] binomial (0.2 + 0.3)', 'freq_a', 0.5),
    ('agg PaLayer 5 claims (500*2) xs (0+0) sev lognorm 100 cv 2 poisson',
     'exp_limit', 1000.0),
])
def test_computed_value_reaches_the_spec(program, key, expected):
    """Arithmetic lands as a plain float in the ordinary spec slot."""
    spec = build(program, update=False).spec
    assert abs(float(np.ravel(spec[key])[0]) - expected) < TOL


def test_computed_range_expands():
    """``[(1+0):(3*2)]`` is the range 1 to 6, the dice outcomes."""
    a = build('agg PaRange dfreq [(1+0):(3*2)] dsev [1]', update=False)
    assert np.allclose(a.spec['freq_a'], np.arange(1, 7))


# ----------------------------------------------------------------------
# [Unparser-Collapse]: canonical, not verbatim
# ----------------------------------------------------------------------
def test_canonical_text_shows_the_evaluated_literal():
    """The writer sees floats, so the formula collapses on the first pass.

    This is the already-documented "canonical, not verbatim" contract
    (``decl_writer`` module docstring, which names ``exp(.5)`` collapsing to
    its float). Expressions evaluate at parse time and the spec carries plain
    numbers, so there is nothing left for the writer to reconstruct.
    """
    program = ('agg PaCollapse (4 + 3*2) claims '
               'sev (exp(-1 * .4**2/2)) * lognorm .4 poisson')
    text = format_program(program)
    assert '+' not in text
    assert 'exp(' not in text
    assert '10 claims' in text

    # and the canonical text is a fixed point
    assert format_program(text) == text
