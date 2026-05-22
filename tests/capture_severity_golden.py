"""Capture pre-refactor Severity layer/attachment values for Stage 1d regression.

Originally captured against post-Stage-1b HEAD (73d9a93). Re-captured against
post-Stage-1d HEAD (d34650f) before the layer/attachment decorator conversion;
this re-capture preserves the original 6-case values bit-for-bit and adds new
``x < 0`` probes for the lognorm cases plus 8 splice / splice+layer cases that
exercise the layer/attachment transform across the spliced support boundary.

Subsequent runs after any further Severity refactor must produce values equal
to the captured ones within ``ATOL`` for every (case, method, probe) triple.

Usage::

    uv run python tests/capture_severity_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aggregate.distributions import Severity


# Default probe sets for the existing 6 lognorm/dhistogram cases. ``x < 0``
# entries lock in the layered-loss x<0 → 0 (cdf), 1 (sf), 0 (pdf) clamps,
# which the decorator conversion must preserve.
DEFAULT_X_POINTS = [-25.0, -5.0, 25.0, 50.0, 150.0, 250.0, 400.0]
DEFAULT_Q_POINTS = [0.01, 0.25, 0.5, 0.75, 0.99]

# Probe set for the splice-only uniform cases. Underlying support is [5, 15];
# probes span "below support", "= splice lb", "inside splice", "= splice ub",
# "above splice", "above support" depending on where the splice sits.
SPLICE_ONLY_X_POINTS = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]

# Probe set for splice+layer cases. After layer/attachment the output lives in
# claim-space [0, limit]; these probes span "x<0 clamp", "at zero", "inside
# layer", "= limit", "above limit".
SPLICE_LAYER_X_POINTS = [-1.0, 0.0, 1.0, 2.5, 4.0, 5.0, 7.0]


def build_cases() -> dict[str, tuple[Severity, list[float], list[float]]]:
    """Construct Severity instances and their per-case probe points.

    Returns
    -------
    dict
        Maps case-name to ``(severity, x_probes, q_probes)``. Names encode
        the configuration so the JSON file is self-describing.
    """
    cases: dict[str, tuple[Severity, list[float], list[float]]] = {}

    # ---- Existing 6 cases: lognorm × {attach=None, 50} × {cond=T, F} ----
    for attachment in (None, 50):
        for conditional in (True, False):
            label = (
                f'lognorm_m100_cv2_attach{attachment if attachment is not None else "None"}'
                f'_limit200_cond{conditional}'
            )
            cases[label] = (
                Severity(
                    sev_name='lognorm',
                    exp_attachment=attachment,
                    exp_limit=200,
                    sev_mean=100,
                    sev_cv=2,
                    sev_conditional=conditional,
                ),
                DEFAULT_X_POINTS,
                DEFAULT_Q_POINTS,
            )

    cases['dhistogram_attach50_limit200_condTrue'] = (
        Severity(
            sev_name='dhistogram',
            exp_attachment=50,
            exp_limit=200,
            sev_xs=[10.0, 50.0, 100.0, 200.0, 500.0],
            sev_ps=[0.10, 0.20, 0.30, 0.30, 0.10],
            sev_conditional=True,
        ),
        DEFAULT_X_POINTS,
        DEFAULT_Q_POINTS,
    )
    cases['dhistogram_attach50_limit200_condFalse'] = (
        Severity(
            sev_name='dhistogram',
            exp_attachment=50,
            exp_limit=200,
            sev_xs=[10.0, 50.0, 100.0, 200.0, 500.0],
            sev_ps=[0.10, 0.20, 0.30, 0.30, 0.10],
            sev_conditional=False,
        ),
        DEFAULT_X_POINTS,
        DEFAULT_Q_POINTS,
    )

    # ---- Splice-only (uniform support [5, 15]) ----
    # No layer/attachment: ``Severity(...)`` with sev_lb/sev_ub only. The
    # ``_apply_lb_ub`` transform reshapes the distribution; no policy layer
    # applies.
    cases['splice_uniform_inside'] = (
        Severity('uniform', sev_loc=5, sev_scale=10, sev_lb=8, sev_ub=12),
        SPLICE_ONLY_X_POINTS,
        DEFAULT_Q_POINTS,
    )
    cases['splice_uniform_left_overlap'] = (
        Severity('uniform', sev_loc=5, sev_scale=10, sev_lb=1, sev_ub=10),
        SPLICE_ONLY_X_POINTS,
        DEFAULT_Q_POINTS,
    )
    cases['splice_uniform_right_overlap'] = (
        Severity('uniform', sev_loc=5, sev_scale=10, sev_lb=12, sev_ub=20),
        SPLICE_ONLY_X_POINTS,
        DEFAULT_Q_POINTS,
    )
    cases['splice_uniform_full'] = (
        Severity('uniform', sev_loc=5, sev_scale=10, sev_lb=5, sev_ub=15),
        SPLICE_ONLY_X_POINTS,
        DEFAULT_Q_POINTS,
    )

    # ---- Splice + layer (both transforms in sequence) ----
    # Mirror the G.Splice11 / G.Splice12 cases conceptually: a spliced
    # severity then wrapped in a policy layer.
    cases['splice_inside_layer_full'] = (
        # splice [8,12] above attach=0/limit=5 → all losses are full-limit
        Severity('uniform', sev_loc=5, sev_scale=10, sev_lb=8, sev_ub=12,
                 exp_attachment=0, exp_limit=5),
        SPLICE_LAYER_X_POINTS,
        DEFAULT_Q_POINTS,
    )
    cases['splice_inside_layer_excess'] = (
        # splice [8,12] with attach=8/limit=2 → proper excess layer
        Severity('uniform', sev_loc=5, sev_scale=10, sev_lb=8, sev_ub=12,
                 exp_attachment=8, exp_limit=2),
        SPLICE_LAYER_X_POINTS,
        DEFAULT_Q_POINTS,
    )
    cases['splice_left_overlap_layer'] = (
        # splice [1,10] effectively [5,10] with attach=2/limit=5 → claim X-2
        # for X in [5,7], full limit 5 for X>7
        Severity('uniform', sev_loc=5, sev_scale=10, sev_lb=1, sev_ub=10,
                 exp_attachment=2, exp_limit=5),
        SPLICE_LAYER_X_POINTS,
        DEFAULT_Q_POINTS,
    )
    cases['splice_full_layer_excess'] = (
        # identity splice [5,15] with attach=3/limit=4 → layer over [3,7]
        Severity('uniform', sev_loc=5, sev_scale=10, sev_lb=5, sev_ub=15,
                 exp_attachment=3, exp_limit=4),
        SPLICE_LAYER_X_POINTS,
        DEFAULT_Q_POINTS,
    )

    return cases


def capture_case(sev: Severity, x_points: list[float],
                 q_points: list[float]) -> dict[str, dict[str, float]]:
    """Evaluate every relevant method at every probe point for one Severity.

    Parameters
    ----------
    sev : Severity
        The Severity instance under test.
    x_points : list of float
        Probe points for ``cdf`` / ``sf`` / ``pdf`` (loss-axis).
    q_points : list of float
        Probe points for ``ppf`` / ``isf`` (probability-axis).

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
    out: dict[str, dict[str, float]] = {}
    for method_name in ('cdf', 'sf', 'pdf'):
        method = getattr(sev, method_name)
        out[method_name] = {repr(x): _to_jsonable(method(x)) for x in x_points}
    for method_name in ('ppf', 'isf'):
        method = getattr(sev, method_name)
        out[method_name] = {repr(q): _to_jsonable(method(q)) for q in q_points}
    return out


def _to_jsonable(value):
    """Coerce scipy/numpy scalar or 0-d array output to a plain Python float."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return [float(v) for v in arr]


def main() -> None:
    cases = build_cases()
    captured = {
        name: capture_case(sev, x_probes, q_probes)
        for name, (sev, x_probes, q_probes) in cases.items()
    }

    metadata = {
        'description': 'Stage 1d follow-up: Severity layer/attachment + splice regression.',
        'captured_against': 'REFACTOR @ d34650f (post Stage 1d Severity refactor)',
        'note': 'Probe points encoded per case in the cases dict keys (repr of float).',
    }

    payload = {'_metadata': metadata, 'cases': captured}

    out_path = Path(__file__).parent / 'data' / 'severity_layer_golden.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'Wrote {len(captured)} cases to {out_path}')


if __name__ == '__main__':
    main()
