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

