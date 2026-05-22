"""Stage 1d regression test for Severity layer/attachment behaviour.

Compares ``.cdf`` / ``.sf`` / ``.pdf`` / ``.ppf`` / ``.isf`` of freshly-built
Severity instances against the golden values captured by
``tests/capture_severity_golden.py`` against the pre-refactor REFACTOR head.

If this test fails after the refactor, the layer/attachment overrides have
drifted from the pre-refactor behaviour. The capture script must NOT be re-run
to "fix" a failure — the whole point is to catch behaviour change.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.capture_severity_golden import (
    Q_POINTS,
    X_POINTS,
    build_cases,
    capture_case,
)

GOLDEN_PATH = Path(__file__).parent / 'data' / 'severity_layer_golden.json'
ATOL = 1e-12


@pytest.fixture(scope='module')
def golden() -> dict:
    """Load the captured golden values once per test module."""
    payload = json.loads(GOLDEN_PATH.read_text(encoding='utf-8'))
    return payload['cases']


@pytest.fixture(scope='module')
def fresh_captures() -> dict:
    """Rebuild every case and re-evaluate; cached for the module."""
    return {name: capture_case(sev) for name, sev in build_cases().items()}


@pytest.mark.parametrize('case_name', list(build_cases().keys()))
@pytest.mark.parametrize('method_name', ['cdf', 'sf', 'pdf', 'ppf', 'isf'])
def test_layer_method_matches_golden(case_name, method_name, golden, fresh_captures):
    """Every (case, method, probe) value matches the captured golden value."""
    expected = golden[case_name][method_name]
    actual = fresh_captures[case_name][method_name]
    probes = Q_POINTS if method_name in ('ppf', 'isf') else X_POINTS
    for probe in probes:
        key = repr(probe)
        e = expected[key]
        a = actual[key]
        # ``math.isclose`` handles inf==inf correctly (which ``abs(a-e)``
        # turns into NaN); ``rel_tol=0`` keeps the comparison absolute.
        assert math.isclose(a, e, rel_tol=0.0, abs_tol=ATOL), (
            f'{case_name}.{method_name}({probe}): expected {e!r}, got {a!r}'
        )
