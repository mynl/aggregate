"""Guard the DecL grammar against new parse ambiguities.

The shipped parser is built with Lark's default ``ambiguity='resolve'``
(``aggregate.parser``), which silently picks one parse from an ambiguous forest
and never warns. That is fine at runtime -- it is deterministic -- but it means
a grammar edit can introduce an ambiguity, or flip how an existing one
resolves, with no visible signal anywhere in the test suite.

This module builds a *second* Lark with ``ambiguity='explicit'``, which keeps
the whole forest and marks every ambiguous span with an ``_ambig`` node, and
sweeps it over every shipped ``.agg`` library. The set of ambiguous statements
must equal :data:`KNOWN_AMBIGUOUS` exactly -- so a new ambiguity fails here
rather than becoming a silent behaviour change.

Three entries are accepted today; see :data:`KNOWN_AMBIGUOUS` for why each is
tolerated. Neither is a defect in the sense of producing a wrong answer.

Companion: ``tests/test_trailer_attachment.py`` pins *which* parse
``ambiguity='resolve'`` currently picks. This module pins *how many* parses
exist. Both are needed -- a flipped binding is still a round-trip fixed point,
and a new ambiguity does not change any existing binding.
"""

from pathlib import Path
import re

import pytest
from lark import Lark

import aggregate
from aggregate.parser import GRAMMAR_FILE, UnderwritingLexer

AGG_DIR = Path(aggregate.__file__).parent / 'agg'

#: Statements permitted to parse ambiguously, keyed by object name.
#:
#: ``UM.Scaled`` -- ``ssev -3 * lognorm 2 cv 0.5`` parses as either
#: ``sev1_scaled(-3, ...)`` or ``sev1_negate(sev1_scaled(3, ...))``. The two are
#: algebraically identical (-3X = -(3X)), so either parse yields the same
#: distribution. Pre-existing and benign.
#:
#: ``AE.Trailer.BivariateBare`` -- a bivariate with neither a copula clause nor
#: an outer frequency puts its own (nullable) trailer directly after a
#: ``bv_body`` whose last ``agg_out`` also ends in a (nullable) trailer, so a
#: trailing ``note{...}`` can bind either way. ``resolve`` binds it to the
#: bivariate, which is the documented behaviour; this fixture exists precisely
#: to hold that line. See ``tests/test_trailer_attachment.py``.
#:
#: ``CHF.VG``, ``chfagg ... variancegamma 0.2 0.5 -0.1``: ``NUMBER`` absorbs a
#: glued leading minus, so ``-0.1`` is the third shape or ``MINUS 0.1``, a
#: shift of the two-shape law (from 1.35.0, ``[CHF-Arithmetic-Form]``).
#: ``resolve`` reads the glued minus as the sign, the only reading with the
#: family's shape count; a spaced ``- 0.1`` is the shift.
#: ``test_characteristic_function.py`` pins both readings.
KNOWN_AMBIGUOUS = {
    'UM.Scaled',
    'AE.Trailer.BivariateBare',
    'CHF.VG',
}

_NAME_RE = re.compile(
    r'^\s*(?:netceded|grossceded|grossnet)?\s*'
    r'(?:agg|sev|ssev|port|pnl|xpnl|bivariate|bv|clash|dist|distortion|chfagg|stable|tweedie)\s+(\S+)',
    re.I,
)


def _name_of(statement):
    """Best-effort object name for a preprocessed DecL statement."""
    m = _NAME_RE.match(statement)
    return m.group(1) if m else statement[:60]


@pytest.fixture(scope='module')
def explicit_parser():
    """A Lark that keeps the whole parse forest instead of resolving it."""
    return Lark.open(
        str(GRAMMAR_FILE),
        start='answer',
        parser='earley',
        lexer='dynamic',
        maybe_placeholders=True,
        ambiguity='explicit',
    )


def _shipped_statements():
    """Every preprocessed statement in every shipped ``.agg`` library."""
    for path in sorted(AGG_DIR.glob('*.agg')):
        for statement in UnderwritingLexer.preprocess(
                path.read_text(encoding='utf-8')):
            yield path.name, statement


def test_no_new_grammar_ambiguities(explicit_parser):
    """The set of ambiguously-parsing shipped statements is exactly the known set.

    A failure here means a grammar edit either introduced an ambiguity (extra
    names) or removed one (missing names). Both are worth a deliberate decision:
    update :data:`KNOWN_AMBIGUOUS` only after confirming which parse
    ``ambiguity='resolve'`` now picks, via ``test_trailer_attachment.py``.
    """
    found = {}
    for filename, statement in _shipped_statements():
        try:
            tree = explicit_parser.parse(statement)
        except Exception:
            # decl-testers.agg carries intentional parser-error fixtures
            # (section X); an unparseable statement cannot be ambiguous.
            continue
        n = sum(1 for st in tree.iter_subtrees() if st.data == '_ambig')
        if n:
            found[_name_of(statement)] = (filename, n)

    assert set(found) == KNOWN_AMBIGUOUS, (
        f'new={sorted(set(found) - KNOWN_AMBIGUOUS)}, '
        f'gone={sorted(KNOWN_AMBIGUOUS - set(found))}; '
        f'details={found}'
    )


def test_sweep_actually_covers_the_corpus():
    """Guard the guard: the sweep must see the whole shipped corpus.

    Without this, a glob or preprocessing regression could silently reduce
    :func:`_shipped_statements` to nothing and leave the ambiguity test
    passing vacuously.
    """
    statements = list(_shipped_statements())
    files = {name for name, _ in statements}
    assert len(statements) > 500, f'only {len(statements)} statements swept'
    assert 'decl-testers.agg' in files
    assert '_test_suite.agg' in files


# ----------------------------------------------------------------------
# [DecL-Paren-Arithmetic] (1.0.0a268): the paren island must stay unambiguous
# ----------------------------------------------------------------------
#: Programs exercising ``+``, ``-`` and ``*`` inside parentheses, each of which
#: must have exactly one parse.
#:
#: The paren gate exists precisely because these operators are ambiguous at
#: bare expression level: ``PLUS`` / ``MINUS`` / ``TIMES`` already carry
#: meaning in ``sev1`` / ``sev2`` and the ``builtin_agg`` algebra, and
#: ``NUMBER`` absorbs a glued leading minus. Inside parentheses a single
#: expression is the only derivation, so ``(1 -2)`` reads as subtraction while
#: the vector ``[1 -2]`` keeps its two-element reading. The cases below cover
#: the operator ladder, the glued-minus quirk, and embedded uses in a layer, a
#: frequency parameter, a range, a reinsurance clause and a severity scale.
PAREN_MATH_PROGRAMS = [
    '(4 + 3*2)',
    '(1-.25)',
    '(1 -2)',
    '(4 -3)',
    '(2 - -3)',
    '(-.4**2)',
    '(0 - .4**2/2)',
    '(2**3**2)',
    '(2+3*4)',
    '((2+3)*4)',
    '(2**3)',
    '(2*3)',
    '(50% + 1)',
    'exp(-1 * .4**2/2)',
    'agg PA.Freq 10 claims dsev [1] binomial (0.2 + 0.3)',
    'agg PA.Layer 5 claims (500*2) xs (0+0) sev lognorm 100 cv 2 poisson',
    'agg PA.Range dfreq [(1+0):(3*2)] dsev [1]',
    'agg PA.Scale 10 claims sev (exp(-1 * .4**2/2)) * lognorm .4 poisson',
    'agg PA.Reins 5 claims 1000 xs 0 sev lognorm 100 cv 2 '
    'occurrence net of (100*2) xs (50 + 50) poisson',
    'agg PA.Head (4 + 3*2) claims (100_000/(1-.25)) premium 1000 xs 0 '
    'sev (exp(-1 * .4**2/2)) * lognorm .4 poisson',
]


@pytest.mark.parametrize('program', PAREN_MATH_PROGRAMS)
def test_paren_math_is_unambiguous(explicit_parser, program):
    """Paren arithmetic adds no parse forest branches.

    A failure means the sum / product ladder leaked out of the paren island and
    started competing with severity or ``builtin_agg`` arithmetic, which is the
    exact failure mode ``[Paren-Gate]`` was designed to prevent.
    """
    tree = explicit_parser.parse(program)
    n = sum(1 for st in tree.iter_subtrees() if st.data == '_ambig')
    assert n == 0, f'{program!r} has {n} ambiguous span(s)'


# [Named-Conversion-Statements] (1.42.0): a name in the head of a conversion,
# and a whole statement that is a dotted reference, add no forest branches.
NAMED_LAW_PROGRAMS = [
    'stable x0 1.5 0',
    'stable x0 1.5 0 2 -3',
    'stable x0 1.5 1 2 -2 s0',
    'stable s0 1.5 0',
    'stable s0 1.5 0 s0',
    'stable x0 as "My law" 1.5 0',
    'stable x0 as lab 1.5 0 s1 note{n} tags{a:b} hints{log2=8}',
    'stable x0 1000 claims sev 100 * lomax 1.5 note{n}',
    'tweedie t0 1.5 10 2',
    'tweedie t0 as T 1.5 10 cv 2 note{n}',
    'tweedie t0 10 claims sev gamma 10 cv 1 tags{a:b}',
    'stable.x0',
    'tweedie.t0',
    'stable 1.5 -0.5',
    'stable 1.5 0.5 -3',
    'stable 1000 claims sev 100 * lomax 1.5',
    'tweedie 1.5 10 2',
]


@pytest.mark.parametrize('program', NAMED_LAW_PROGRAMS)
def test_named_laws_are_unambiguous(explicit_parser, program):
    """The name slot and the two new reference terminals stay single parse."""
    tree = explicit_parser.parse(program)
    n = len(list(tree.find_data('_ambig')))
    assert n == 0, f'{program!r} has {n} ambiguous span(s)'


# [Stable-Arithmetic] (1.43.0): every operator form has one parse, and an
# ordinary conversion beside them keeps its one parse.
STABLE_ARITHMETIC_PROGRAMS = [
    '3 * stable.x0',
    'stable.x1 + stable.x2',
    'stable.x0 + 2 * stable.x1 + 3',
    'stable z stable.x0 + 2 * stable.x1 + 3',
    'stable z stable.x0',
    'stable z as Combo stable.x - stable.y note{net}',
    'stable z 3 @ stable.x',
    'stable.x * 3',
    'stable.x / 3',
    '3 + stable.x',
    '3 - stable.x',
    'stable.x - 3',
    'stable.x -3',
    '-stable.x',
    '- stable.x',
    'stable.x - -stable.y',
    'stable.x -stable.y',
    'stable.x - -3',
    '2 * 3 * stable.x',
    '2 * stable.x * 3',
    '2 * (stable.x + stable.y) - 1',
    '(1/2) * stable.x',
    '1/2 * stable.x',
    '(1+2) * stable.x',
    '-2 * stable.x',
    'exp(1) * stable.x',
    '2 ** 3 * stable.x',
    '50% * stable.x',
    'stable.x / 3 * 2',
    '-(stable.x + 1)',
    '(stable.x)',
    '((stable.x) + 1) * 2',
    'stable.x + 1 + 2',
    '1 + stable.x + 2',
    '1 - stable.x - 2',
    'stable.x - 3 * stable.y',
    'stable.x -3 * stable.y',
    '2 @ stable.x',
    '3 * 2 @ stable.x',
    '2 @ 3 * stable.x',
    '1/2 @ stable.x',
    'stable.x + 2 @ stable.y + 1',
    'stable.x1-stable.x2',
    'stable 1.5 -0.5',
    'stable 1.5 0.5 -3',
]


@pytest.mark.parametrize('program', STABLE_ARITHMETIC_PROGRAMS)
def test_stable_arithmetic_is_unambiguous(explicit_parser, program):
    """The ladder of ``stable_sum`` has one parse for every operator form."""
    tree = explicit_parser.parse(program)
    n = len(list(tree.find_data('_ambig')))
    assert n == 0, f'{program!r} has {n} ambiguous span(s)'
