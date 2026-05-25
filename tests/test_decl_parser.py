"""Parse tests for every line of aggregate/agg/test_suite.agg.

These tests are the regression net for the SLY -> Lark parser migration.
Each DecL line in test_suite.agg becomes its own parametrized test case so
a parser failure points at the specific offending line.

Two test functions:

- ``test_line_parses`` only checks that the line parses and returns a valid
  ``(kind, name, spec)`` shape.
- ``test_spec_matches_snapshot`` compares the spec against a JSON snapshot
  captured from SLY before the Lark migration, catching any semantic drift.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from aggregate.parser import UnderwritingLexer

TEST_SUITE_PATH = Path(__file__).parent.parent / "src" / "aggregate" / "agg" / "test_suite.agg"
SNAPSHOT_PATH = Path(__file__).parent / "data" / "expected_specs.json"

VALID_KINDS = {"agg", "sev", "port", "distortion", "expr"}

# Sentinels used by tests/capture_sly_snapshot.py for non-JSON floats.
POS_INF_SENTINEL = "__inf__"
NEG_INF_SENTINEL = "__-inf__"
NAN_SENTINEL = "__nan__"


def _load_lines() -> list[str]:
    text = TEST_SUITE_PATH.read_text(encoding="utf-8")
    return UnderwritingLexer.preprocess(text)


def _line_id(line: str) -> str:
    return (line[:80] + "...") if len(line) > 80 else line


def _from_sentinel(obj):
    """Recursively rehydrate sentinels (inf, -inf, nan) back into floats."""
    if isinstance(obj, str):
        if obj == POS_INF_SENTINEL:
            return math.inf
        if obj == NEG_INF_SENTINEL:
            return -math.inf
        if obj == NAN_SENTINEL:
            return math.nan
        return obj
    if isinstance(obj, dict):
        return {k: _from_sentinel(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_sentinel(x) for x in obj]
    return obj


def _normalize(obj):
    """Normalize a parsed spec value for comparison with a JSON snapshot.

    Converts numpy arrays/scalars to plain Python lists/numbers so that
    ``assert actual == expected`` works against JSON-loaded data.
    """
    if isinstance(obj, np.ndarray):
        return [_normalize(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer, np.floating)):
        return _normalize(obj.item())
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    return obj


def _assert_equal(actual, expected, path: str = ""):
    """Recursively compare two structures with float tolerance via pytest.approx."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"At {path}: expected dict, got {type(actual).__name__}"
        assert set(actual) == set(expected), (
            f"At {path}: key mismatch. Extra={set(actual) - set(expected)}, "
            f"missing={set(expected) - set(actual)}"
        )
        for k in expected:
            _assert_equal(actual[k], expected[k], f"{path}.{k}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"At {path}: expected list, got {type(actual).__name__}"
        assert len(actual) == len(expected), (
            f"At {path}: length {len(actual)} != expected {len(expected)}"
        )
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_equal(a, e, f"{path}[{i}]")
        return
    if isinstance(expected, float):
        if math.isnan(expected):
            assert isinstance(actual, float) and math.isnan(actual), f"At {path}: expected NaN"
            return
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), (
            f"At {path}: {actual} != {expected}"
        )
        return
    assert actual == expected, f"At {path}: {actual!r} != {expected!r}"


LINES = _load_lines()
SNAPSHOT = _from_sentinel(json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8")))


@pytest.mark.parametrize("line", LINES, ids=[_line_id(ln) for ln in LINES])
def test_line_parses(line: str, underwriter):
    """Every DecL line in test_suite.agg must parse to a valid (kind, name, spec)."""
    kind, name, spec = underwriter.parser.parse(underwriter.lexer.tokenize(line))
    assert kind in VALID_KINDS, f"Unexpected kind {kind!r} for line: {line}"
    assert isinstance(name, str) and name, f"Empty name for line: {line}"
    if kind != "expr":
        assert isinstance(spec, dict), f"Spec should be a dict for kind {kind}, got {type(spec)}"


@pytest.mark.parametrize("line", LINES, ids=[_line_id(ln) for ln in LINES])
def test_spec_matches_snapshot(line: str, underwriter):
    """Parsed (kind, name, spec) must match the SLY-captured snapshot exactly."""
    expected = SNAPSHOT.get(line)
    assert expected is not None, f"No snapshot entry for line: {line}"
    if "error" in expected:
        pytest.skip(f"Line errored in SLY snapshot: {expected['error']}")
    kind, name, spec = underwriter.parser.parse(underwriter.lexer.tokenize(line))
    assert kind == expected["kind"]
    assert name == expected["name"]
    _assert_equal(_normalize(spec), expected["spec"])
