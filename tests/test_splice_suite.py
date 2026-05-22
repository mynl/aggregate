"""Parametric tests for splice / splice-and-layer cases from test_suite2.agg.

Twelve cases hand-curated by the user that exercise the `sev_lb` / `sev_ub`
splice path with and without policy layers. Four of them describe
measure-zero splices (the splice window does not intersect the underlying
distribution's support) and must raise ``ValueError`` at construction time.

Lives alongside (not inside) ``test_suite.agg`` so the established 134-line
parametric test count doesn't move; the two files will be consolidated in
a later pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aggregate.distributions import Aggregate, Severity
from aggregate.parser import UnderwritingLexer
from aggregate.underwriter import Underwriter

SUITE_PATH = Path(__file__).parent.parent / "aggregate" / "agg" / "test_suite2.agg"

# Names of the four cases that describe measure-zero splices — the splice
# window lies entirely outside the underlying distribution's support, so the
# conditional distribution is mathematically undefined.
INVALID_NAMES = {"Splice02", "Splice05", "Splice07", "Splice10"}


def _load_lines() -> list[str]:
    text = SUITE_PATH.read_text(encoding="utf-8")
    return UnderwritingLexer.preprocess(text)


def _line_id(line: str) -> str:
    return (line[:80] + "...") if len(line) > 80 else line


LINES = _load_lines()


@pytest.fixture(scope="module")
def underwriter() -> Underwriter:
    """A fresh Underwriter for the splice suite — no databases preloaded."""
    return Underwriter()


def _short_name(name: str) -> str:
    """Strip the ``G.`` prefix to match INVALID_NAMES."""
    return name.split(".", 1)[1] if "." in name else name


@pytest.mark.parametrize("line", LINES, ids=[_line_id(ln) for ln in LINES])
def test_splice_line_parses(line: str, underwriter):
    """All 12 lines must parse — measure-zero is a construction-time error.

    Parsing the spec must succeed even for the four invalid cases; the
    ``ValueError`` only fires when the Severity is actually constructed.
    """
    kind, name, spec = underwriter.parser.parse(underwriter.lexer.tokenize(line))
    assert kind in {"sev", "agg"}, f"Unexpected kind {kind!r} for line: {line}"
    assert isinstance(name, str) and name, f"Empty name for line: {line}"
    assert isinstance(spec, dict), f"Spec should be a dict for kind {kind}"


@pytest.mark.parametrize("line", LINES, ids=[_line_id(ln) for ln in LINES])
def test_splice_line_builds_or_raises(line: str, underwriter):
    """Valid splice cases build; measure-zero ones raise ``ValueError``.

    Builds the spec by routing through ``Underwriter._factory`` (the same
    code path as ``build('agg ...')`` / ``build('sev ...')``); this exercises
    the Severity constructor under the same conditions a user-typed program
    would.
    """
    kind, name, spec = underwriter.parser.parse(underwriter.lexer.tokenize(line))
    short = _short_name(name)

    if short in INVALID_NAMES:
        with pytest.raises(ValueError, match="zero probability mass"):
            _build_from_spec(kind, spec)
        return

    obj = _build_from_spec(kind, spec)
    if kind == "sev":
        assert isinstance(obj, Severity), f"{name}: expected Severity, got {type(obj).__name__}"
    elif kind == "agg":
        assert isinstance(obj, Aggregate), f"{name}: expected Aggregate, got {type(obj).__name__}"


def _build_from_spec(kind: str, spec: dict):
    """Instantiate the appropriate top-level object from a parsed spec.

    Mirrors the dispatch in ``Underwriter._factory`` for the two kinds that
    appear in ``test_suite2.agg``. We don't go through ``Underwriter.write``
    because the splice failures need to surface as a ``ValueError`` at the
    Severity layer; the Underwriter wraps errors and rebuilds.
    """
    if kind == "sev":
        return Severity(**spec)
    if kind == "agg":
        return Aggregate(**spec)
    raise ValueError(f"Unhandled kind {kind!r}")
