"""Capture pre-refactor Severity layer/attachment values for Stage 1d regression.

Run once before the Stage 1d refactor begins, against the current REFACTOR HEAD
(post-Stage-1b). Subsequent runs after the refactor must produce values equal
to the captured ones within ``ATOL`` for every (case, method, x_or_q) triple.

Usage::

    uv run python tests/capture_severity_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aggregate.distributions import Severity


X_POINTS = [25.0, 50.0, 150.0, 250.0, 400.0]
Q_POINTS = [0.01, 0.25, 0.5, 0.75, 0.99]


def build_cases() -> dict[str, Severity]:
    """Construct the Severity instances under test.

    Returns
    -------
    dict mapping case-name to Severity instance. Names encode the parameter
    triple so the JSON file is self-describing.
    """
    cases: dict[str, Severity] = {}
    for attachment in (None, 50):
        for conditional in (True, False):
            label = (
                f'lognorm_m100_cv2_attach{attachment if attachment is not None else "None"}'
                f'_limit200_cond{conditional}'
            )
            cases[label] = Severity(
                sev_name='lognorm',
                exp_attachment=attachment,
                exp_limit=200,
                sev_mean=100,
                sev_cv=2,
                sev_conditional=conditional,
            )

    cases['dhistogram_attach50_limit200_condTrue'] = Severity(
        sev_name='dhistogram',
        exp_attachment=50,
        exp_limit=200,
        sev_xs=[10.0, 50.0, 100.0, 200.0, 500.0],
        sev_ps=[0.10, 0.20, 0.30, 0.30, 0.10],
        sev_conditional=True,
    )
    cases['dhistogram_attach50_limit200_condFalse'] = Severity(
        sev_name='dhistogram',
        exp_attachment=50,
        exp_limit=200,
        sev_xs=[10.0, 50.0, 100.0, 200.0, 500.0],
        sev_ps=[0.10, 0.20, 0.30, 0.30, 0.10],
        sev_conditional=False,
    )
    return cases


def capture_case(sev: Severity) -> dict[str, dict[str, list[float]]]:
    """Evaluate every relevant method at every probe point for one Severity.

    Returns
    -------
    Nested dict::

        {'cdf': {'25.0': ..., '50.0': ..., ...},
         'sf':  {...},
         'pdf': {...},
         'ppf': {'0.01': ..., ...},
         'isf': {...}}

    Values are plain floats so the result is JSON-serializable.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for method_name in ('cdf', 'sf', 'pdf'):
        method = getattr(sev, method_name)
        out[method_name] = {repr(x): _to_jsonable(method(x)) for x in X_POINTS}
    for method_name in ('ppf', 'isf'):
        method = getattr(sev, method_name)
        out[method_name] = {repr(q): _to_jsonable(method(q)) for q in Q_POINTS}
    return out


def _to_jsonable(value):
    """Coerce scipy/numpy scalar or 0-d array output to a plain Python float."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return [float(v) for v in arr]


def main() -> None:
    cases = build_cases()
    captured = {name: capture_case(sev) for name, sev in cases.items()}

    metadata = {
        'description': 'Stage 1d Severity layer/attachment regression golden values.',
        'captured_against': 'REFACTOR @ 73d9a93 (post Stage 1b Frequency refactor)',
        'x_points': X_POINTS,
        'q_points': Q_POINTS,
    }

    payload = {'_metadata': metadata, 'cases': captured}

    out_path = Path(__file__).parent / 'data' / 'severity_layer_golden.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'Wrote {len(captured)} cases to {out_path}')


if __name__ == '__main__':
    main()
