"""Tests for the numerics-1 unit-density accessors (``dev/plan-numerics-1-unit-density.md``).

Covers:

- ``Portfolio.unit_density`` / ``unit_density_df`` -- native-grid unit pmfs
  sourced from the owning ``Aggregate`` objects, independent of the total
  grid (including genuinely disjoint unit windows).
- ``Portfolio.aligned_unit_density_df`` -- the explicitly-named display
  adapter: exact legacy parity on a zero-origin book, a window-mismatch
  warning on a windowed (signed) book.
- The migrated display readers (``percentiles``, ``_limits``, ``plot``,
  pedagogy density/bivariate panels) work without the legacy
  ``density_df['p_{unit}']`` columns.

The DecL programs are mirrored in ``src/aggregate/agg/test_decl.agg``
(section UD).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from aggregate import build
from aggregate.constants import DefectiveDistributionWarning
from aggregate.spectral import Distortion


PLAIN_PROGRAM = '''port UD.Plain
    agg P 50 claims sev lognorm 100 cv 1 poisson
    agg Q 30 claims sev lognorm 80 cv 1 poisson'''

# signed pair: each unit gets its own signed window; the unit grids extend
# above the total grid top (x_min_tot = x_min_A + x_min_B < each x_min_k).
PNL_PROGRAM = '''port UD.PnLpair
    agg A 50 claims ssev 10 * norm + 5 poisson
    agg B 50 claims ssev 10 * norm - 5 poisson'''

# disjoint unit windows: Hi lives near +10,000, Lo near -10,000 (sd ~1,000
# each), so the native windows do not overlap at all.
DISJOINT_PROGRAM = '''port UD.Disjoint
    agg Hi 100 claims ssev 1 * norm + 100 poisson
    agg Lo 100 claims ssev 1 * norm - 100 poisson'''


@pytest.fixture(scope='module')
def plain():
    return build(PLAIN_PROGRAM)


@pytest.fixture(scope='module')
def pnl():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DefectiveDistributionWarning)
        return build(PNL_PROGRAM)


@pytest.fixture(scope='module')
def disjoint():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DefectiveDistributionWarning)
        return build(DISJOINT_PROGRAM)


# ---------------------------------------------------------------------------
# unit_density: native grid, represented mass.
# ---------------------------------------------------------------------------

def test_unit_density_native(plain):
    """pmf is the unit's own ``agg_density`` on the unit's own grid."""
    for u in plain.line_names:
        ser = plain.unit_density(u)
        agg = plain[u]
        assert ser.name == f'p_{u}'
        assert np.array_equal(ser.index.to_numpy(), np.asarray(agg.xs))
        assert np.array_equal(ser.to_numpy(),
                              agg.density_df.p_total.to_numpy())
        # sums to the unit's represented mass
        assert float(ser.sum()) == pytest.approx(
            float(agg.density_df.p_total.sum()), abs=1e-15)


def test_unit_density_views_and_errors(plain):
    """``view='sev'`` reads the severity grid; bad args raise."""
    sev = plain.unit_density('P', view='sev')
    agg = plain['P']
    assert np.array_equal(sev.index.to_numpy(), np.asarray(agg.xs_sev))
    assert np.array_equal(sev.to_numpy(),
                          agg.sev_density_df.p_sev.to_numpy())
    with pytest.raises(KeyError, match='unknown unit'):
        plain.unit_density('NoSuchUnit')
    with pytest.raises(ValueError, match='view'):
        plain.unit_density('P', view='bogus')


def test_unit_density_df_long(plain):
    """Long form: (unit, loss) index, per-unit blocks, audit metadata."""
    df = plain.unit_density_df()
    assert list(df.index.names) == ['unit', 'loss']
    assert list(df.columns) == ['unit', 'loss', 'p', 'F', 'S', 'bs',
                                'x_min', 'x_max', 'mass']
    for u in plain.line_names:
        block = df.loc[u]
        agg = plain[u]
        assert np.array_equal(block.p.to_numpy(),
                              agg.density_df.p_total.to_numpy())
        assert np.array_equal(block.F.to_numpy(),
                              agg.density_df.F.to_numpy())
        assert (block.bs == agg.bs).all()
        assert float(block.x_min.iloc[0]) == float(agg.xs[0])
        assert float(block.x_max.iloc[0]) == float(agg.xs[-1])
        assert float(block.mass.iloc[0]) == pytest.approx(
            float(agg.density_df.p_total.sum()), abs=1e-15)


# ---------------------------------------------------------------------------
# Disjoint windows: the accessor does not depend on total-grid overlap.
# ---------------------------------------------------------------------------

def test_disjoint_windows(disjoint):
    """Unit rows equal each unit's own frame even with disjoint supports.

    The units share the portfolio bs/log2, so their *grids* have the same
    width; what is disjoint is the mass-bearing support (Hi near +10,000,
    Lo near -10,000).
    """
    hi, lo = disjoint['Hi'], disjoint['Lo']
    hi_support = hi.density_df.query('p_total > 1e-12').loss
    lo_support = lo.density_df.query('p_total > 1e-12').loss
    # the premise: the mass-bearing supports genuinely do not overlap
    assert float(hi_support.min()) > float(lo_support.max())
    df = disjoint.unit_density_df()
    for u in disjoint.line_names:
        block = df.loc[u]
        agg = disjoint[u]
        assert np.array_equal(block.index.to_numpy(), np.asarray(agg.xs))
        assert np.array_equal(block.p.to_numpy(),
                              agg.density_df.p_total.to_numpy())
        # each unit carries (essentially) full mass on its own window
        assert float(block.p.sum()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# aligned_unit_density_df: legacy parity and windowed warning.
# ---------------------------------------------------------------------------

def test_aligned_total_legacy_parity(plain):
    """Zero-origin book: grid='total' reproduces each native unit pmf
    exactly (the legacy ``p_{unit}`` columns themselves left
    ``density_df`` at numerics-2; on a zero-origin book the unit grid is
    the total grid, so the aligned view IS the old column)."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        aligned = plain.aligned_unit_density_df(grid='total')
    assert np.array_equal(aligned.index.to_numpy(),
                          plain.density_df.index.to_numpy())
    for u in plain.line_names:
        assert f'p_{u}' not in plain.density_df.columns
        assert np.array_equal(aligned[f'p_{u}'].to_numpy(),
                              plain.unit_density(u).to_numpy())


def test_aligned_union_and_zero(plain):
    """On a zero-origin book union/zero lattices agree with the total grid."""
    total = plain.aligned_unit_density_df(grid='total')
    union = plain.aligned_unit_density_df(grid='union')
    zero = plain.aligned_unit_density_df(grid='zero')
    for u in plain.line_names:
        assert np.array_equal(union[f'p_{u}'].to_numpy(),
                              total[f'p_{u}'].to_numpy())
        assert np.array_equal(zero[f'p_{u}'].to_numpy(),
                              total[f'p_{u}'].to_numpy())
    assert float(zero.index[0]) == 0.0


def test_aligned_total_windowed_warns(pnl):
    """Windowed (signed) book: grid='total' clips the unit windows -> warn."""
    with pytest.warns(UserWarning, match='window'):
        pnl.aligned_unit_density_df(grid='total')
    # acknowledged: no warning
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        clipped = pnl.aligned_unit_density_df(
            grid='total', allow_window_mismatch=True)
    # physical placement: each unit's mode lands at the same loss
    for u in pnl.line_names:
        native = pnl.unit_density(u)
        assert float(clipped[f'p_{u}'].idxmax()) == pytest.approx(
            float(native.idxmax()), abs=pnl.bs / 2)
    # union never clips
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        union = pnl.aligned_unit_density_df(grid='union')
    for u in pnl.line_names:
        assert float(union[f'p_{u}'].sum()) == pytest.approx(
            float(pnl.unit_density(u).sum()), abs=1e-12)


def test_aligned_bs_mismatch_raises(plain):
    """A unit re-updated off the portfolio grid fails loudly."""
    port = build(PLAIN_PROGRAM)
    port['P'].update(log2=port.log2, bs=port.bs / 2)
    with pytest.raises(ValueError, match='bs'):
        port.aligned_unit_density_df()


# ---------------------------------------------------------------------------
# Migrated display readers no longer need the legacy p_{unit} columns.
# ---------------------------------------------------------------------------

@pytest.fixture()
def stripped():
    """A fresh portfolio. The legacy p_{unit} columns no longer exist
    (numerics-2), so this is just a plain build; the fixture name is kept
    to document what these tests guard: display readers must not need
    unit pmfs on the total frame."""
    port = build(PLAIN_PROGRAM)
    assert not [c for c in port.density_df.columns
                if c in (f'p_{u}' for u in port.line_names)]
    return port


def test_percentiles_off_p_unit(stripped):
    df = stripped.percentiles()
    assert df.notna().all().all()
    # interpolated unit percentile sits near the exact step quantile
    p50 = float(df.loc['P'].iloc[0])
    assert abs(p50 - stripped['P'].q(0.5)) <= 2 * stripped.bs


def test_limits_and_plot_off_p_unit(stripped):
    lo, hi = stripped._limits(stat='density')
    assert hi > 0
    lo, hi = stripped._limits(stat='logy')
    assert hi > 0
    stripped.plot()
    plt.close('all')


def test_pedagogy_off_p_unit(stripped):
    from aggregate.pedagogy import ClassicalPremium, plot_bivariate
    cp = ClassicalPremium({'plain': stripped}, 1000)
    for line in ['P', 'total']:
        df, ob, stats, mn, var, sd = cp.distribution('plain', line)
        assert float(df.p.sum()) == pytest.approx(1.0, abs=1e-6)
        assert float(df.F.iloc[-1]) == pytest.approx(1.0, abs=1e-6)
    fig, ax = plt.subplots()
    plot_bivariate(stripped, fig, ax, 0, stripped.q(0.99),
                   64 * stripped.bs)
    plt.close('all')


def test_plot_twelve_off_p_unit():
    """plot_twelve renders from native unit pmfs (density/bivariate panels)."""
    from aggregate.pedagogy import plot_twelve
    port = build(PLAIN_PROGRAM)
    d = Distortion('ph', 0.6)
    port.apply_distortion(d, efficient=False)
    assert f'p_{port.line_names[0]}' not in port.density_df.columns
    fig, axs = plt.subplots(4, 3, figsize=(12, 16))
    plot_twelve(port, fig, axs, d)
    plt.close('all')
