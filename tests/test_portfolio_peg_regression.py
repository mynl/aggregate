"""PEG Portfolio regression test.

Locks the numerical output of ``calibrate_distortions`` and
``analyze_distortions2`` on the two-unit PEG Portfolio against the captured
baseline in ``tests/data/peg_baseline.json``. Every subsequent Portfolio
refactor sub-project must keep these assertions green.

Three tests today; grows to six by the end of Sub-project D when
``pricing_at`` exists:

- ``test_portfolio_moments`` — agg/est moments match baseline (``rtol=1e-10``).
- ``test_calibration_shapes`` — each distortion's ``shape`` matches baseline
  (``rtol=1e-8``) and its calibration residual is within ``1e-5``.
- ``test_pricing`` — every cell of ``analyze_distortions2(.995)`` matches the
  baseline (``rtol=1e-8``).

Notes
-----
The baseline JSON is the contract. If a refactor *intentionally* changes a
number, the right move is to re-capture (``uv run python -m
tests.capture_peg_baseline``) and document the change in the commit. Do not
silently widen tolerances.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.peg import build_peg


BASELINE_PATH = Path(__file__).parent / 'data' / 'peg_baseline.json'
BASELINE = json.loads(BASELINE_PATH.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def peg():
    """Build PEG once, calibrate, and run analyze_distortions2 at p=0.995.

    Returns ``(port, ad)`` where ``ad`` is the exhibit DataFrame.
    """
    p = BASELINE['meta']['p_calibration']
    coc = BASELINE['meta']['coc_calibration']
    log2 = BASELINE['meta']['log2']
    bs = BASELINE['meta']['bs']
    port = build_peg(update=True, calibrate=True, p=p, coc=coc,
                     log2=log2, bs=bs)
    ad = port.analyze_distortions2(p)
    return port, ad


def test_portfolio_moments(peg):
    """Theoretical and empirical aggregate moments match baseline."""
    port, _ = peg
    for key, expected in BASELINE['portfolio_moments'].items():
        actual = getattr(port, key)
        assert np.isclose(actual, expected, rtol=1e-10), \
            f'{key}: {actual!r} vs baseline {expected!r}'


def test_calibration_shapes(peg):
    """Each calibrated distortion reproduces baseline shape and residual."""
    port, _ = peg
    for name, expected in BASELINE['calibration'].items():
        assert name in port.dists, f'distortion {name!r} missing from port.dists'
        d = port.dists[name]
        assert np.isclose(d.shape, expected['shape'], rtol=1e-8), \
            f'{name}.shape: {d.shape!r} vs baseline {expected["shape"]!r}'
        assert abs(d.error) < 1e-5, \
            f'{name}.error: |{d.error!r}| >= 1e-5'


def test_pricing(peg):
    """Every cell of analyze_distortions2(.995) matches baseline.

    Indexed as ``ad.loc[(distortion, stat), column]``. Five distortions × 8
    stats × 3 columns = 120 cells; all must match within ``rtol=1e-8``.
    """
    _, ad = peg
    for dname, by_column in BASELINE['pricing'].items():
        assert dname in ad.index.get_level_values(0).unique(), \
            f'distortion {dname!r} missing from analyze_distortions2 output'
        for column, by_stat in by_column.items():
            for stat, expected in by_stat.items():
                actual = ad.loc[(dname, stat), column]
                assert np.isclose(actual, expected, rtol=1e-8), \
                    f'{dname}.{column}.{stat}: {actual!r} vs baseline {expected!r}'
