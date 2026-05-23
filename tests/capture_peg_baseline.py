"""Capture the PEG Portfolio regression baseline to JSON.

Runs the canonical PEG Portfolio through the current
``calibrate_distortions`` → ``analyze_distortions2`` pipeline and writes
the numerical results to ``tests/data/peg_baseline.json``. Every subsequent
Portfolio-refactor sub-project must reproduce these numbers.

Usage::

    uv run python tests/capture_peg_baseline.py

The output JSON is the contract. Re-capture only when an *intentional*
numerical change is being introduced (and document the reason in the
commit). The regression test that loads this file lives in
``tests/test_portfolio_peg_regression.py``.

Layout of the captured baseline:

- ``meta``: version, captured-date, program text, build parameters
- ``portfolio_moments``: theoretical and empirical moments of the
  aggregate (``agg_m/cv/skew``, ``est_m/cv/skew``)
- ``calibration``: per-distortion ``shape`` and ``error`` from
  ``port.dists``
- ``audit``: minimal calibration inputs (``a_cal = port.q(p)``, ``p``)
  reproducible without parsing ``pricing``
- ``pricing``: full ``analyze_distortions2`` exhibit, nested
  ``{distortion: {column: {stat: value}}}``
"""
from __future__ import annotations

import json
from datetime import date
from importlib.metadata import version as pkg_version
from pathlib import Path

from tests.peg import PEG_PROGRAM, build_peg


P_CAL = 0.995
COC_CAL = 0.15
LOG2 = 16


def extract_calibration(port):
    """Per-distortion ``shape`` and ``error`` from a calibrated Portfolio."""
    return {
        name: {'shape': float(d.shape), 'error': float(d.error)}
        for name, d in port.dists.items()
    }


def extract_pricing(ad):
    """Nested dict view of the ``analyze_distortions2`` exhibit DataFrame.

    ``ad`` is a DataFrame with MultiIndex ``(distortion, stat)`` and columns
    one per unit + ``total``. Return shape is
    ``{distortion: {column: {stat: float}}}`` — the natural cell-by-cell
    layout for the regression test.
    """
    out = {}
    for dname in ad.index.get_level_values(0).unique():
        sub = ad.xs(dname, level=0)  # 8-row, 3-col DataFrame
        out[str(dname)] = {
            col: {stat: float(sub.loc[stat, col]) for stat in sub.index}
            for col in sub.columns
        }
    return out


def main() -> None:
    port = build_peg(update=True, calibrate=True, p=P_CAL, coc=COC_CAL,
                     log2=LOG2)
    ad = port.analyze_distortions2(P_CAL)

    baseline = {
        'meta': {
            'captured_at': date.today().isoformat(),
            'aggregate_version': pkg_version('aggregate'),
            'program': PEG_PROGRAM,
            'log2': LOG2,
            'bs': float(port.bs),
            'p_calibration': P_CAL,
            'coc_calibration': COC_CAL,
        },
        'portfolio_moments': {
            'agg_m':    float(port.agg_m),
            'agg_cv':   float(port.agg_cv),
            'agg_skew': float(port.agg_skew),
            'est_m':    float(port.est_m),
            'est_cv':   float(port.est_cv),
            'est_skew': float(port.est_skew),
        },
        'calibration': extract_calibration(port),
        'audit': {
            'a_cal': float(port.q(P_CAL)),
            'p':     P_CAL,
        },
        'pricing': extract_pricing(ad),
    }

    out_path = Path(__file__).parent / 'data' / 'peg_baseline.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline, indent=2), encoding='utf-8')
    print(f'Wrote PEG baseline to {out_path}')
    print(f'  agg_m  = {baseline["portfolio_moments"]["agg_m"]:.6f}')
    print(f'  est_m  = {baseline["portfolio_moments"]["est_m"]:.6f}')
    print(f'  a_cal  = {baseline["audit"]["a_cal"]:.2f}')
    print(f'  dists  = {list(baseline["calibration"].keys())}')


if __name__ == '__main__':
    main()
