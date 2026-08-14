"""Tests for ``BivariateAggregate.exeqa_df``, the kappa curve.

[Bivariate-Exeqa], 1.0.0a272, phase 1 of
``dev/done/plan-natural-allocation-to-occurrence-net-ceded.md``: the
conditional mean ``E[Y | X = x]`` over the conditioning axis grid, taken out
of the joint rather than out of the 1-D FFT trick. The two axes of a netceded
joint are dependent by construction (a shared claim count, a comonotone
per-claim cession), so Portfolio's independence route is unavailable.

Phase 2, which consumes this, is in ``tests/test_natural_allocation.py``.

The programs are small on purpose and the joint grids are pinned, so the suite
stays in the fast tier, which is where a correctness check belongs.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build


# A share cession on a discrete severity whose halves land on the joint
# lattice: every per-claim point (x, x/2) is a grid point at bs=1, so the
# bilinear scatter never splits and the kappa curve is exact.
QS_ALIGNED = 'agg NA.QS dfreq [3] dsev [2:20:2] occurrence net of 50% po inf xs 0'
# A continuous share cession: same mathematics, but the scatter splits, so the
# curve is accurate to the smear rather than to the bit.
QS_CONTINUOUS = (
    'agg NA.QSC 10 claims sev lognorm 50 cv 1.5 '
    'occurrence net of 50% po inf xs 0 poisson'
)
# An ordinary excess layer: the case the plan was written for.
LAYER = (
    'agg NA.L 10 claims sev lognorm 50 cv 1.5 '
    'occurrence net of 50 xs 50 poisson'
)
# Pinned joint grid, shared by the layer cases so the two view pairs land on
# one common gross axis (see test_two_joint_additivity).
PIN = dict(bs=4, log2_x=10, log2_y=10)


@pytest.fixture(scope='module')
def qs_aligned():
    return build(QS_ALIGNED).occ_bivariate(views=('gross', 'ceded'), bs=1)


@pytest.fixture(scope='module')
def layer_agg():
    return build(LAYER)


@pytest.fixture(scope='module')
def layer_gc(layer_agg):
    return layer_agg.occ_bivariate(views=('gross', 'ceded'), **PIN)


# ---------------------------------------------------------------------------
# exeqa_df
# ---------------------------------------------------------------------------

def test_quota_share_exact(qs_aligned):
    """A lattice-aligned 50% share has kappa_C(g) = g / 2 exactly.

    One assert pins the sign convention, the axis order and the grid
    alignment together: a transposed product or an off-by-one index fails it.
    """
    df = qs_aligned.exeqa_df(axis=0)
    live = df['p'] > 0
    g = df.index.to_numpy(float)[live]
    kappa = df['exeqa_Ceded'].to_numpy()[live]
    assert live.sum() > 20
    assert np.abs(kappa - 0.5 * g).max() < 1e-12


def test_quota_share_continuous_within_a_bucket():
    """The same cession on a continuous severity: right to within one bucket.

    The bilinear split preserves both marginal means exactly but not a
    conditional one, so the curve carries the rebucketing smear. The smear is
    an **absolute** quantity of order ``bs``, which is why the tolerance is
    stated in buckets: a relative one would look terrible near the origin, in
    the one place where a bucket of error does not matter.
    """
    biv = build(QS_CONTINUOUS).occ_bivariate(views=('gross', 'ceded'), **PIN)
    df = biv.exeqa_df(axis=0)
    live = df['p'].to_numpy() > 0
    g = df.index.to_numpy(float)[live]
    err = np.abs(df['exeqa_Ceded'].to_numpy()[live] - 0.5 * g)
    assert err.max() < biv.bs[0]
    assert np.median(err) < 0.05 * biv.bs[0]


def test_ties_to_the_moment_store(layer_gc):
    """Mass weighted kappa reproduces the ceded marginal mean.

    Both are the same sum of ``y_j d[i, j]`` taken in a different order, so
    the tie is to floating point whatever the joint's deficit.
    """
    df = layer_gc.exeqa_df(axis=0)
    live = df['p'] > 0
    via_kappa = float((df['exeqa_Ceded'][live] * df['p'][live]).sum())
    _, ceded_marginal = layer_gc.marginals
    via_marginal = float(ceded_marginal @ layer_gc.axis_xs[1])
    assert via_kappa == pytest.approx(via_marginal, rel=1e-12)


def test_pointwise_agrees_with_the_conditional_law(layer_gc):
    """``exeqa`` at a grid value is the mean of that row's normalized law.

    The independent route ``MassiveBivariateDistribution.slice`` takes: read
    the row, normalize, hand it to a ``GridDistribution``. Checked here
    against the in-core joint, which carries no ``slice`` of its own.
    """
    from aggregate._grid_distribution import GridDistribution

    df = layer_gc.exeqa_df(axis=0)
    y = layer_gc.axis_xs[1]
    rows = np.flatnonzero(df['p'].to_numpy() > 0)[::97]
    assert len(rows) > 5
    for i in rows:
        row = layer_gc.density[i, :]
        gd = GridDistribution(y, row / row.sum(), bs=layer_gc.bs[1])
        assert df['exeqa_Ceded'].iloc[i] == pytest.approx(gd.mean(), rel=1e-12)


def test_two_joint_additivity(layer_agg, layer_gc):
    """``E[C | G] + E[N | G] = g``, to the rebucketing scatter.

    The decomposition check the plan asks for, taken across the two view
    pairs that share a gross axis: the ``(gross, ceded)`` and ``(gross, net)``
    joints are built at one pinned ``bs`` so their gross grids coincide, and
    the two kappa curves are asked to close on the index. Reported at a loose
    tolerance rather than asserted at machine precision, because the scatter
    can put mass an epsilon off the line ``c + n = g``; the split is mean
    preserving, which is why the tie is nonetheless tight.
    """
    gn = layer_agg.occ_bivariate(views=('gross', 'net'), **PIN)
    dc = layer_gc.exeqa_df(axis=0)
    dn = gn.exeqa_df(axis=0)
    assert np.array_equal(dc.index.to_numpy(), dn.index.to_numpy())

    live = (dc['p'].to_numpy() > 0) & (dn['p'].to_numpy() > 0)
    g = dc.index.to_numpy(float)[live]
    total = (dc['exeqa_Ceded'].to_numpy()[live]
             + dn['exeqa_Net'].to_numpy()[live])
    err = np.abs(total - g)
    assert err.max() < layer_gc.bs[0]
    assert np.median(err) < 0.25 * layer_gc.bs[0]


def test_axis_one_is_the_transpose(layer_gc):
    """``axis=1`` conditions on the other variable, with the names to match."""
    df = layer_gc.exeqa_df(axis=1)
    assert df.index.name == 'Ceded'
    assert list(df.columns) == ['p', 'F', 'S', 'exeqa_Ceded', 'exeqa_Gross']
    live = df['p'] > 0
    via_kappa = float((df['exeqa_Gross'][live] * df['p'][live]).sum())
    gross_marginal, _ = layer_gc.marginals
    assert via_kappa == pytest.approx(
        float(gross_marginal @ layer_gc.axis_xs[0]), rel=1e-12)


def test_identity_column_and_masking(layer_gc):
    """The self column is the index where there is mass, and NaN where not."""
    df = layer_gc.exeqa_df(axis=0)
    live = df['p'].to_numpy() > 0
    idx = df.index.to_numpy(float)
    assert np.array_equal(df['exeqa_Gross'].to_numpy()[live], idx[live])
    assert np.isnan(df['exeqa_Gross'].to_numpy()[~live]).all()
    assert np.isnan(df['exeqa_Ceded'].to_numpy()[~live]).all()


def test_probability_columns(layer_gc):
    """``F`` is the running mass and ``S`` its complement, deficit included."""
    df = layer_gc.exeqa_df(axis=0)
    assert df['F'].to_numpy() == pytest.approx(np.cumsum(df['p'].to_numpy()))
    assert (df['F'] + df['S']).to_numpy() == pytest.approx(1.0)
    assert df['S'].iloc[-1] == pytest.approx(layer_gc.deficit, abs=1e-12)


def test_bad_axis_refused(layer_gc):
    with pytest.raises(ValueError, match='axis must be 0 or 1'):
        layer_gc.exeqa_df(axis=2)


def test_requires_update():
    from aggregate.bivariate import BivariateAggregate

    a = build(LAYER)
    biv = BivariateAggregate('probe', mode='netceded', nc_agg=a,
                             nc_views=('gross', 'ceded'))
    with pytest.raises(ValueError, match='not updated'):
        biv.exeqa_df()


# ---------------------------------------------------------------------------
# the conditional band ([Kappa-Band-Columns], 1.0.0a279)
# ---------------------------------------------------------------------------

def test_levels_are_additive(layer_gc):
    """No levels is today's frame, byte for byte."""
    import pandas as pd

    pd.testing.assert_frame_equal(layer_gc.exeqa_df(axis=0),
                                  layer_gc.exeqa_df(axis=0, levels=None))
    assert list(layer_gc.exeqa_df(axis=0, levels=()).columns) == [
        'p', 'F', 'S', 'exeqa_Gross', 'exeqa_Ceded']


def test_band_columns_are_named_for_the_other_axis(layer_gc):
    """The naming follows ``exeqa_<axis>``, which already carries the name."""
    df = layer_gc.exeqa_df(axis=0, levels=(0.01, 0.5, 0.99))
    assert list(df.columns)[-3:] == ['q01_Ceded', 'q50_Ceded', 'q99_Ceded']
    # a fractional level keeps its decimals, so two near levels cannot collide
    fine = layer_gc.exeqa_df(axis=0, levels=(0.001, 0.005))
    assert list(fine.columns)[-2:] == ['q0.1_Ceded', 'q0.5_Ceded']


def test_band_edges_are_monotone_in_the_level(layer_gc):
    """A higher level is a higher quantile, on every row. Always."""
    df = layer_gc.exeqa_df(axis=0, levels=(0.01, 0.5, 0.99))
    live = df['p'].to_numpy() > 0
    lo = df['q01_Ceded'].to_numpy()[live]
    mid = df['q50_Ceded'].to_numpy()[live]
    hi = df['q99_Ceded'].to_numpy()[live]
    assert (lo <= mid).all()
    assert (mid <= hi).all()
    # and it is a band, not a repeat of the mean: an occurrence layer's
    # cession is genuinely uncertain given the gross outcome
    assert (hi - lo).max() > layer_gc.bs[1]


def test_kappa_lies_inside_the_conditional_support(layer_gc):
    """The mean of each row's law lies between that law's extreme quantiles.

    The bracket that is a theorem, and it is a real check because the two
    sides come from different code: the mean is a matrix vector product over
    the band, the quantiles are per row ``GridDistribution`` calls.

    A **percentile** band is not guaranteed to contain the mean, and on this
    joint it does not everywhere: at a small gross outcome the conditional
    cession is a spike at zero carrying a vanishing chance of a full limit
    recovery, so ``q99`` is 0 while the mean is 5e-06, and the mean sits above
    the band. That is a true statement about a very skewed conditional law and
    exactly the kind of thing a mean alone never says, which is the whole
    argument for drawing the band.
    """
    df = layer_gc.exeqa_df(axis=0, levels=(0.0, 1.0))
    live = df['p'].to_numpy() > 0
    lo = df['q00_Ceded'].to_numpy()[live]
    hi = df['q100_Ceded'].to_numpy()[live]
    mean = df['exeqa_Ceded'].to_numpy()[live]
    assert (lo <= mean + 1e-9).all()
    assert (hi >= mean - 1e-9).all()
    assert (hi > lo).any()


def test_band_is_the_row_quantile(layer_gc):
    """Each edge is a quantile of that row's own normalized conditional law."""
    from aggregate._grid_distribution import GridDistribution

    df = layer_gc.exeqa_df(axis=0, levels=(0.25, 0.75))
    y = layer_gc.axis_xs[1]
    rows = np.flatnonzero(df['p'].to_numpy() > 0)[::97]
    assert len(rows) > 5
    for i in rows:
        row = layer_gc.density[i, :]
        gd = GridDistribution(y, row / row.sum(), bs=layer_gc.bs[1])
        assert df['q25_Ceded'].iloc[i] == pytest.approx(gd.q(0.25))
        assert df['q75_Ceded'].iloc[i] == pytest.approx(gd.q(0.75))


def test_a_deterministic_cession_has_no_band(qs_aligned):
    """A lattice-aligned share is a function of the gross, so the band closes.

    The band measures what the kappa curve averages away, and a share cession
    averages nothing away: given the gross outcome the cession is known.
    """
    df = qs_aligned.exeqa_df(axis=0, levels=(0.01, 0.99))
    live = df['p'].to_numpy() > 0
    width = (df['q99_Ceded'].to_numpy()[live]
             - df['q01_Ceded'].to_numpy()[live])
    assert np.abs(width).max() < 1e-12


def test_band_is_nan_off_the_support(qs_aligned):
    """A quantile given a null event has no value, exactly as the mean does.

    On the discrete program, where the gross lattice has genuine gaps; a
    continuous joint carries floating point dust on every row and has none.
    """
    df = qs_aligned.exeqa_df(axis=0, levels=(0.5,))
    dead = df['p'].to_numpy() <= 0
    assert dead.any()
    assert np.isnan(df['q50_Ceded'].to_numpy()[dead]).all()
    assert np.isnan(df['exeqa_Ceded'].to_numpy()[dead]).all()


def test_bad_level_refused(layer_gc):
    with pytest.raises(ValueError, match=r'lies in \[0, 1\]'):
        layer_gc.exeqa_df(axis=0, levels=(0.5, 1.5))


# --- the plotted window -----------------------------------------------------

def test_cdf_range_crops_rather_than_blanks(layer_gc):
    """Cropping keeps ``NaN`` meaning "no mass", never "not measured"."""
    full = layer_gc.exeqa_df(axis=0, levels=(0.01, 0.99))
    win = layer_gc.exeqa_df(axis=0, levels=(0.01, 0.99),
                            cdf_range=(1e-3, 0.999))
    assert len(win) < len(full)
    assert win.index[0] >= full.index[0]
    assert not np.isnan(win['q01_Ceded'].to_numpy()[
        win['p'].to_numpy() > 0]).any()


def test_cdf_range_keeps_the_whole_distribution_probabilities(layer_gc):
    """``F`` and ``S`` describe the law, not the crop, so they still tie out."""
    win = layer_gc.exeqa_df(axis=0, cdf_range=(1e-3, 0.999))
    full = layer_gc.exeqa_df(axis=0)
    assert win['F'].to_numpy() == pytest.approx(
        full.loc[win.index, 'F'].to_numpy())
    assert win['F'].iloc[0] >= 1e-3
    assert (win['F'] + win['S']).to_numpy() == pytest.approx(1.0)


def test_cdf_range_agrees_with_the_uncropped_sweep(layer_gc):
    """A window is a cheaper route to the same numbers, not different ones."""
    full = layer_gc.exeqa_df(axis=0, levels=(0.05, 0.95))
    win = layer_gc.exeqa_df(axis=0, levels=(0.05, 0.95),
                            cdf_range=(0.01, 0.99))
    common = win.index
    for col in ('exeqa_Ceded', 'q05_Ceded', 'q95_Ceded'):
        assert win[col].to_numpy() == pytest.approx(
            full.loc[common, col].to_numpy(), nan_ok=True)


def test_bad_cdf_range_refused(layer_gc):
    with pytest.raises(ValueError, match='increasing probability pair'):
        layer_gc.exeqa_df(axis=0, cdf_range=(0.9, 0.1))
    with pytest.raises(ValueError, match='increasing probability pair'):
        layer_gc.exeqa_df(axis=0, cdf_range=(-0.1, 0.9))


# --- the disk route ---------------------------------------------------------

@pytest.mark.slow
def test_massive_kappa_matches_in_core(tmp_path):
    """The refusal is gone, and what replaced it is the same numbers.

    ``exeqa_df`` refused a disk-backed joint through 1.0.0a278 and took
    ``natural_allocation`` down with it, which was the one hole in an
    otherwise first class massive surface.
    """
    pytest.importorskip('zarr')
    from aggregate import build as _build

    agg = _build(QS_ALIGNED)
    core = agg.occ_bivariate(views=('gross', 'ceded'), bs=1)
    disk = agg.occ_bivariate(views=('gross', 'ceded'), bs=1,
                             store_dir=str(tmp_path / 'kb'))
    a = core.exeqa_df(axis=0, levels=(0.01, 0.99))
    b = disk.exeqa_df(axis=0, levels=(0.01, 0.99))
    assert list(a.columns) == list(b.columns)
    for col in a.columns:
        assert b[col].to_numpy() == pytest.approx(a[col].to_numpy(),
                                                  abs=1e-12, nan_ok=True)


@pytest.mark.slow
def test_massive_natural_allocation(tmp_path):
    """The allocation rides the same fold, so it survives the disk route."""
    pytest.importorskip('zarr')
    from aggregate import build as _build

    agg = _build(QS_ALIGNED)
    dist = _build('dist KB.PH ph 0.5')
    core = agg.occ_bivariate(views=('gross', 'ceded'), bs=1)
    disk = agg.occ_bivariate(views=('gross', 'ceded'), bs=1,
                             store_dir=str(tmp_path / 'kb'))
    want = core.natural_allocation(dist, P=100.0)
    got = disk.natural_allocation(dist, P=100.0)
    # the two routes carry different default padding (1 in core, 0 on disk,
    # its documented default), so they agree to the aliasing rather than to
    # the bit; the additivity below is exact on either.
    assert got['P'].to_numpy() == pytest.approx(want['P'].to_numpy(), rel=1e-6)
    assert (got.loc['ceded', 'P'] + got.loc['net', 'P']
            == pytest.approx(got.loc['gross', 'P'], abs=1e-12))
