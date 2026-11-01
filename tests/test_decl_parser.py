"""Parse tests for every statement of aggregate/agg/decl-testers.agg.

Each statement becomes its own parametrized test case, so a parser failure
points at the specific program.

Two test functions:

- ``test_line_parses`` only checks that the statement parses and returns a valid
  ``(kind, name, spec)`` shape.
- ``test_spec_matches_snapshot`` compares the spec against the JSON snapshot
  written by ``tests/capture_spec_snapshot.py``, a change detector for semantic
  drift in the grammar and the transformer.
"""

import json
import math
import sys
from fractions import Fraction

import numpy as np
import pytest

from aggregate import Severity, build
from aggregate.parser import UnderwritingLexer
from tests.capture_spec_snapshot import (CORPUS, NAN_SENTINEL, NEG_INF_SENTINEL,
                                         OUT as SNAPSHOT_PATH, POS_INF_SENTINEL,
                                         plain_value, statement_keys)

VALID_KINDS = {"agg", "sev", "port", "distortion", "expr", "nttagg", "chfagg",
               "stable", "tweedie", "pnl", "xpnl", "bvagg"}


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
    obj = plain_value(obj)
    if isinstance(obj, Fraction):
        # exact nttagg probabilities; capture_spec_snapshot.jsonify writes
        # the same "a/b" string form
        return f"{obj.numerator}/{obj.denominator}"
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


LINES = UnderwritingLexer.preprocess(CORPUS.read_text(encoding="utf-8"))
KEYS = statement_keys(LINES)
SNAPSHOT = _from_sentinel(json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8")))


@pytest.mark.parametrize("line", LINES, ids=KEYS)
def test_line_parses(line: str, underwriter):
    """Every statement of decl-testers.agg must parse to a valid (kind, name, spec)."""
    kind, name, spec = underwriter.parser.parse(underwriter.lexer.tokenize(line))
    assert kind in VALID_KINDS, f"Unexpected kind {kind!r} for line: {line}"
    assert isinstance(name, str) and name, f"Empty name for line: {line}"
    if kind != "expr":
        assert isinstance(spec, dict), f"Spec should be a dict for kind {kind}, got {type(spec)}"


@pytest.mark.parametrize("key, line", list(zip(KEYS, LINES)), ids=KEYS)
def test_spec_matches_snapshot(key: str, line: str, underwriter):
    """Parsed (kind, name, spec) must match the captured snapshot exactly."""
    expected = SNAPSHOT.get(key)
    assert expected is not None, (
        f"No snapshot entry for {key}: run tests/capture_spec_snapshot.py and read the diff")
    assert "error" not in expected, f"{key} errored at capture: {expected['error']}"
    kind, name, spec = underwriter.parser.parse(underwriter.lexer.tokenize(line))
    assert kind == expected["kind"]
    assert name == expected["name"]
    _assert_equal(_normalize(spec), expected["spec"])


def test_snapshot_has_no_orphan_entries():
    """An entry names a statement of the corpus, so a removed or renamed program is noticed."""
    orphans = sorted(set(SNAPSHOT) - set(KEYS))
    assert not orphans, f"{orphans} left the corpus: run tests/capture_spec_snapshot.py"


def test_list_longer_than_recursion_limit(underwriter):
    """A vector far longer than the recursion limit parses, in the order written.

    Each item of a left recursive list is one level of the parse tree, and the
    recursive ``lark.Transformer`` raised ``RecursionError`` at about 500.
    """
    xs = list(range(1, 2 * sys.getrecursionlimit() + 1))
    line = f"agg LongVector 1 claim dsev [{' '.join(map(str, xs))}] fixed"
    kind, name, spec = underwriter.parser.parse(underwriter.lexer.tokenize(line))
    assert (kind, name) == ("agg", "LongVector")
    assert list(spec["sev_xs"]) == xs


def test_loss_table_longer_than_recursion_limit(underwriter):
    """An inline event loss table far longer than the recursion limit parses."""
    n = 2 * sys.getrecursionlimit()
    rows = " ".join(f"[{i} .001 {i}]" for i in range(1, n + 1))
    line = f"agg LongTable elt [{rows}]"
    kind, name, spec = underwriter.parser.parse(underwriter.lexer.tokenize(line))
    assert (kind, name) == ("agg", "LongTable")
    assert len(spec["elt_sources"][0]["rows"]) == n


def test_long_vector_builds_the_right_distribution():
    """A long severity vector reaches the distribution, not just the spec."""
    from aggregate import build

    n = 600
    xs = " ".join(str(i) for i in range(1, n + 1))
    agg = build(f"agg LongVectorBuilt 1 claim dsev [{xs}] fixed")
    assert agg.est_m == pytest.approx((n + 1) / 2)


# ---------------------------------------------------------------------------
# An unmatched bracket ([Using-Notes-Guards])
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('program, char, line, column', [
    ('agg B1 1 claim dsev [1 2 fixed', '[', 1, 21),
    ('agg B2 1 claim dsev 1 2] fixed', ']', 1, 24),
    ('agg B3 1 claim dsev [[1 2] fixed', '[', 1, 21),
    ('agg B4 1 claim dsev [1 2]] fixed', ']', 1, 26),
    # a position counts in the text as written, past a comment holding a bracket
    ('# a comment [\nagg B5 1 claim\n  dsev [1 2\n  fixed', '[', 3, 8),
])
def test_unmatched_bracket_is_a_located_parse_error(program, char, line, column):
    with pytest.raises(ValueError, match=rf"Unmatched '\{char}'") as err:
        UnderwritingLexer.preprocess(program)
    report = err.value.report
    assert (report.line, report.column, report.got) == (line, column, char)


@pytest.mark.parametrize('program', [
    'agg B6 1 claim dsev [1 2] fixed note{E[x}',
    'agg B7 1 claim dsev [1 2] fixed # stray [ in a comment',
    'agg B8 as "a[1" 1 claim dsev [1 2] fixed',
])
def test_a_bracket_in_a_literal_or_comment_is_not_counted(program):
    assert len(UnderwritingLexer.preprocess(program)) == 1


# ---------------------------------------------------------------------------
# A severity family that does not exist ([Using-Notes-Guards])
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('program, word, close', [
    ('agg S1 3 claims sev lognorml 1 poisson', 'lognorml', 'lognorm'),
    ('agg S2 3 claims sev gammma 100 cv 1 poisson', 'gammma', 'gamma'),
    ('agg S3 3 claims sev [lognorm gammma] [1 2] poisson', 'gammma', 'gamma'),
])
def test_unknown_severity_family_suggests_the_closest(program, word, close):
    with pytest.raises(ValueError, match=f"Unknown severity family '{word}'") as err:
        build(program)
    report = err.value.report
    assert report.suggestions[0] == close
    assert report.got == word
    assert report.source_line[report.column - 1:].startswith(word)


def test_unknown_severity_family_with_nothing_close():
    with pytest.raises(ValueError, match="Unknown severity family 'qqqqzz'") as err:
        build('agg S4 3 claims sev qqqqzz 1 poisson')
    assert err.value.report.suggestions == []


def test_unknown_severity_family_outside_decl():
    with pytest.raises(ValueError, match="unknown severity family 'lognorml'; did you mean lognorm"):
        Severity('lognorml', sev_a=1)

