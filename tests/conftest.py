"""Shared pytest fixtures for the aggregate test suite."""

from pathlib import Path

import pytest

from aggregate.parser import UnderwritingLexer
from aggregate.underwriter import Underwriter

TEST_SUITE_PATH = Path(__file__).parent.parent / "aggregate" / "agg" / "test_suite.agg"


@pytest.fixture(scope="session")
def test_suite_lines() -> list[str]:
    """All preprocessed DecL lines from aggregate/agg/test_suite.agg."""
    text = TEST_SUITE_PATH.read_text(encoding="utf-8")
    return UnderwritingLexer.preprocess(text)


@pytest.fixture(scope="session")
def underwriter(test_suite_lines):
    """An Underwriter with the test_suite.agg knowledge tolerantly preloaded.

    Each line is parsed and added to the knowledge base; parse failures are
    swallowed here so they surface as individual test failures in the
    parametrized parse tests rather than as a fixture error.
    """
    uw = Underwriter()
    for line in test_suite_lines:
        try:
            kind, name, spec = uw.parser.parse(uw.lexer.tokenize(line))
        except Exception:
            continue
        uw._knowledge.loc[(kind, name), :] = [spec, line]
    return uw
