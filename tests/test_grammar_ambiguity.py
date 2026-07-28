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

Two entries are accepted today; see :data:`KNOWN_AMBIGUOUS` for why each is
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
KNOWN_AMBIGUOUS = {
    'UM.Scaled',
    'AE.Trailer.BivariateBare',
}

_NAME_RE = re.compile(
    r'^\s*(?:netceded|grossceded|grossnet)?\s*'
    r'(?:agg|sev|ssev|port|pnl|xpnl|bivariate|bv|clash|dist|distortion)\s+(\S+)',
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
