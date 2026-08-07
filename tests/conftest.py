"""Shared pytest fixtures for the aggregate test suite."""

from pathlib import Path

import pytest

from aggregate.constants import reset_warn_once
from aggregate.parser import UnderwritingLexer
from aggregate.underwriter import Underwriter

TEST_SUITE_PATH = Path(__file__).parent.parent / "src" / "aggregate" / "agg" / "_test_suite.agg"


@pytest.fixture(autouse=True)
def _reset_warn_once():
    """Re-arm the once-per-session warnings before every test.

    ``warn_once`` state is process-wide by design, which is right for a
    session and wrong for a test suite: without this, the first test to
    provoke a condition silences ``pytest.warns`` for every later test in the
    same worker, and which test that is depends on ordering and on the xdist
    split. Reset before, not after, so a test can inspect the registry it
    leaves behind.
    """
    reset_warn_once()


@pytest.fixture(scope="session")
def test_suite_lines() -> list[str]:
    """All preprocessed DecL lines from aggregate/agg/_test_suite.agg."""
    text = TEST_SUITE_PATH.read_text(encoding="utf-8")
    return UnderwritingLexer.preprocess(text)


@pytest.fixture(scope="session")
def underwriter(test_suite_lines):
    """An Underwriter with the _test_suite.agg recipes tolerantly preloaded.

    Each line is parsed and added to the recipe base; parse failures are
    swallowed here so they surface as individual test failures in the
    parametrized parse tests rather than as a fixture error.
    """
    uw = Underwriter()
    for line in test_suite_lines:
        try:
            kind, name, spec = uw.parser.parse(uw.lexer.tokenize(line))
        except Exception:
            continue
        uw.add_recipe(kind, name, spec, line)
    return uw
