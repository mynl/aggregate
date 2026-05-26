"""
One-time pre-refactor snapshot capture for Distortion g(s) / g_inv(s).

Run on the pre-refactor tip to produce ``tests/data/distortion_g_snapshot.csv``
which freezes the numerical behaviour of every documented kind at a
canonical parameter set. ``tests/test_distortion_snapshot.py`` then
reconstructs each distortion through the new natural-kwarg constructor
and asserts that the values match within ``rtol=1e-10``.

Usage
-----
    uv run python tests/capture_distortion_snapshot.py

This is a one-shot artefact; do NOT re-run after the refactor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aggregate.spectral import Distortion


# 101-point uniform grid; corners + interior. Some inverse transforms blow
# up exactly at 0 / 1 so we evaluate g_inv on an inset grid.
S_GRID = np.linspace(0.0, 1.0, 101)
S_GRID_INSET = np.linspace(1e-6, 1 - 1e-6, 101)


# (kind, label, construction-args-pre-refactor). Construction here uses the
# OLD constructor signature; this script is intentionally pre-refactor.
SPECS = [
    ('ph',      'ph',      dict(name='ph',     shape=0.7)),
    ('wang',    'wang',    dict(name='wang',   shape=0.3)),
    ('dual',    'dual',    dict(name='dual',   shape=2.0)),
    ('tvar',    'tvar',    dict(name='tvar',   shape=0.95)),
    ('ccoc',    'ccoc',    dict(name='ccoc',   shape=0.1)),       # r = 0.1
    ('bitvar',  'bitvar',  dict(name='bitvar', shape=0.5, df=[0.95, 0.99])),
    ('wtdtvar', 'wtdtvar', dict(name='wtdtvar', shape=[0.5, 0.9, 1.0],
                                df=[0.3, 0.3, 0.4])),
    ('cll',     'cll',     dict(name='cll',    shape=0.9, r0=0.05)),
    ('clin',    'clin',    dict(name='clin',   shape=2.0, r0=0.05)),
    ('lep',     'lep',     dict(name='lep',    shape=0.15, r0=0.03)),
    ('ly',      'ly',      dict(name='ly',     shape=1.25, r0=0.05)),
    ('beta',    'beta',    dict(name='beta',   shape=[0.7, 1.5])),
    ('power',   'power',   dict(name='power',  shape=2.0, df=[0.01, 1.0])),
]


def _safe_g(d, x):
    try:
        return np.asarray(d.g(x), dtype=float)
    except Exception as exc:                       # noqa: BLE001
        print(f'  g failed for {d}: {exc}')
        return np.full_like(x, np.nan, dtype=float)


def _safe_g_inv(d, x):
    try:
        return np.asarray(d.g_inv(x), dtype=float)
    except Exception as exc:                       # noqa: BLE001
        print(f'  g_inv failed for {d}: {exc}')
        return np.full_like(x, np.nan, dtype=float)


def main():
    out_path = Path(__file__).parent / 'data' / 'distortion_g_snapshot.csv'
    out_path.parent.mkdir(exist_ok=True)

    cols = {'s': S_GRID}
    inset_cols = {'s_inset': S_GRID_INSET}

    for kind, label, kwargs in SPECS:
        print(f'Capturing {label} ...')
        d = Distortion(**kwargs)
        cols[f'{label}_g'] = _safe_g(d, S_GRID)
        inset_cols[f'{label}_g_inv'] = _safe_g_inv(d, S_GRID_INSET)

    df = pd.DataFrame(cols)
    df_inv = pd.DataFrame(inset_cols)
    out = pd.concat([df, df_inv], axis=1)
    out.to_csv(out_path, index=False)
    print(f'\nWrote {out_path} ({out.shape[0]} rows x {out.shape[1]} cols)')


if __name__ == '__main__':
    main()
