"""Tests for the Grübel–Hermesmeier exponential-tilting pedagogy helpers.

``aggregate.pedagogy.tilted_aggregate_density`` reproduces the tilted FFT
convolution illustrated in the numerical-methods docs
(``docs/2_user_guides/problems/010_gh_example.rst``). Tilting is a teaching
device about aliasing control; it is NOT part of the production convolution
path, which is tilt-free (padding is the operational control). These tests pin:

- ``tilt=None`` reproduces the ordinary (untilted) convolution byte-for-byte;
- a heavy tilt removes the coarse-grid aliasing, converging to the accurate
  high-resolution / exact value (the published-table behaviour).

The DecL program is mirrored in ``src/aggregate/agg/test_decl.agg`` (section GH).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from aggregate import build
from aggregate.pedagogy import (gh_tilting_exhibit, tilt_vector,
                                tilted_aggregate_density)

GH_PROG = 'agg L 20 claim sev levy poisson'


def test_tilt_vector_shape_and_values():
    tv = tilt_vector(5 / 1024, 1024)
    assert tv.shape == (1024,)
    assert tv[0] == pytest.approx(1.0)
    np.testing.assert_allclose(tv, np.exp(-(5 / 1024) * np.arange(1024)))


def test_tilt_none_matches_plain_convolution():
    """``tilt=None`` is the ordinary untilted convolution on the same grid."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build(GH_PROG, update=False)
        a.update(log2=10, bs=1, padding=0, normalize=False)
        plain = pd.Series(a.agg_density, index=a.xs)
        tilted = tilted_aggregate_density(a, log2=10, bs=1, padding=0, tilt=None)
    np.testing.assert_allclose(tilted.to_numpy(), plain.to_numpy(), atol=0, rtol=0)


def test_tilt_reduces_aliasing_at_small_x():
    """At x=1 the coarse untilted value is badly aliased; a heavy tilt fixes it."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build(GH_PROG, update=False)
        # accurate, high-resolution reference
        a.update(log2=16, bs=1, padding=2, normalize=False)
        accurate = float(a.density_df.loc[1, 'p_total'] / a.bs)
        untilted = tilted_aggregate_density(a, log2=10, bs=1, padding=0, tilt=None)
        heavy = tilted_aggregate_density(a, log2=10, bs=1, padding=0, tilt=25 / 1024)
    u = float(untilted.loc[1])
    h = float(heavy.loc[1])
    # untilted coarse grid is aliased high by orders of magnitude
    assert u > 100 * accurate
    # the heavy tilt recovers the accurate value
    assert h == pytest.approx(accurate, rel=1e-3)


def test_gh_tilting_exhibit_reproduces_table():
    """The one-call exhibit assembles accurate + exact + tilt sweep coherently."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = gh_tilting_exhibit()
    assert list(df.index) == [1, 10, 100, 1000]
    assert 'accurate' in df.columns and 'exact' in df.columns
    assert 'tilt 0.0000' in df.columns and 'tilt 0.0244' in df.columns
    # the heaviest tilt column tracks the accurate column across all rows
    np.testing.assert_allclose(df['tilt 0.0244'].to_numpy(),
                               df['accurate'].to_numpy(), rtol=1e-3)
    # accurate and exact agree closely in the body of the distribution
    assert df.loc[100, 'accurate'] == pytest.approx(df.loc[100, 'exact'], rel=1e-3)
