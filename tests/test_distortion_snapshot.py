"""
Snapshot regression: rebuild each Distortion kind through the natural-kwarg
constructor and assert the post-refactor ``g`` / ``g_inv`` match the
pre-refactor values captured in ``tests/data/distortion_g_snapshot.csv``.

The CSV is generated once by ``tests/capture_distortion_snapshot.py`` on
the pre-refactor tip. This test then pins numerical behaviour of every
documented distortion to ``rtol=1e-10`` regardless of any further internal
plumbing changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aggregate import Distortion


SNAPSHOT_PATH = Path(__file__).parent / 'data' / 'distortion_g_snapshot.csv'


# (label, kind, natural-kwarg construction). Mirrors the parameter grid
# in capture_distortion_snapshot.py exactly.
SPECS = [
    ('ph',      'ph',      dict(a=0.7)),
    ('wang',    'wang',    dict(lam=0.3)),
    ('dual',    'dual',    dict(b=2.0)),
    ('tvar',    'tvar',    dict(p=0.95)),
    ('ccoc',    'ccoc',    dict(r=0.1)),
    ('bitvar',  'bitvar',  dict(p0=0.95, p1=0.99, w1=0.5)),
    ('wtdtvar', 'wtdtvar', dict(ps=[0.5, 0.9, 1.0], wts=[0.3, 0.3, 0.4])),
    ('cll',     'cll',     dict(r0=0.05, b=0.9)),
    ('clin',    'clin',    dict(r0=0.05, slope=2.0)),
    ('lep',     'lep',     dict(r0=0.03, r=0.15)),
    ('ly',      'ly',      dict(r0=0.05, r=1.25)),
    ('beta',    'beta',    dict(a=0.7, b=1.5)),
    ('power',   'power',   dict(x0=0.01, x1=1.0, alpha=2.0)),
]


@pytest.fixture(scope='module')
def snapshot():
    if not SNAPSHOT_PATH.exists():
        pytest.skip(
            f'{SNAPSHOT_PATH.name} missing; run '
            'tests/capture_distortion_snapshot.py once on the pre-refactor '
            'tip to seed it.')
    return pd.read_csv(SNAPSHOT_PATH)


@pytest.mark.parametrize('label,kind,kwargs', SPECS, ids=[s[0] for s in SPECS])
def test_g_matches_snapshot(snapshot, label, kind, kwargs):
    """``g`` evaluated on a 101-point grid matches the pre-refactor CSV."""
    s = snapshot['s'].values
    expected = snapshot[f'{label}_g'].values
    d = Distortion(kind, **kwargs)
    actual = np.asarray(d.g(s), dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('label,kind,kwargs', SPECS, ids=[s[0] for s in SPECS])
def test_g_inv_matches_snapshot(snapshot, label, kind, kwargs):
    """``g_inv`` evaluated on a 101-point inset grid matches the CSV."""
    s = snapshot['s_inset'].values
    expected = snapshot[f'{label}_g_inv'].values
    d = Distortion(kind, **kwargs)
    actual = np.asarray(d.g_inv(s), dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
