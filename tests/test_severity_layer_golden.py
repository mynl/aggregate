"""Stage 1d regression test for Severity layer/attachment behaviour.

Compares ``.cdf`` / ``.sf`` / ``.pdf`` / ``.ppf`` / ``.isf`` of freshly-built
Severity instances against the golden values captured by
``tests/capture_severity_golden.py``.

Per-case probe points are encoded in the JSON keys (``repr(float)``), so the
test reads them directly out of the golden file rather than from a module
constant; new cases just need their probes captured into the JSON.

If this test fails after a Severity refactor, the layer/attachment overrides
have drifted from the captured behaviour. The capture script must NOT be
re-run to "fix" a failure — the whole point is to catch behaviour change.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.capture_severity_golden import build_cases, capture_case

GOLDEN_PATH = Path(__file__).parent / 'data' / 'severity_layer_golden.json'
ATOL = 1e-12

METHOD_NAMES = ('cdf', 'sf', 'pdf', 'ppf', 'isf')


@pytest.fixture(scope='module')
def golden() -> dict:
    """Load the captured golden values once per test module."""
    payload = json.loads(GOLDEN_PATH.read_text(encoding='utf-8'))
    return payload['cases']


@pytest.fixture(scope='module')
def fresh_captures() -> dict:
    """Rebuild every case and re-evaluate using the per-case probes."""
    return {
        name: capture_case(sev, x_probes, q_probes)
        for name, (sev, x_probes, q_probes) in build_cases().items()
    }


@pytest.mark.parametrize('case_name', list(build_cases().keys()))
@pytest.mark.parametrize('method_name', METHOD_NAMES)
def test_layer_method_matches_golden(case_name, method_name, golden, fresh_captures):
    """Every (case, method, probe) value matches the captured golden value."""
    expected = golden[case_name][method_name]
    actual = fresh_captures[case_name][method_name]
    assert set(expected.keys()) == set(actual.keys()), (
        f'{case_name}.{method_name}: probe set mismatch '
        f'(expected {sorted(expected)}, got {sorted(actual)})'
    )
    for key, e in expected.items():
        a = actual[key]
        # ``math.isclose`` handles inf == inf correctly (``abs(a - e)`` turns
        # that into NaN); ``rel_tol=0`` keeps the comparison absolute.
        assert math.isclose(a, e, rel_tol=0.0, abs_tol=ATOL), (
            f'{case_name}.{method_name}({key}): expected {e!r}, got {a!r}'
        )
