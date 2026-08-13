"""Tests for ``BivariateAggregate.natural_allocation``.

[NetCeded-Natural-Allocation], 1.0.0a273, phase 2 of
``dev/done/plan-natural-allocation-to-occurrence-net-ceded.md``: a gross
distorted premium allocated to the occurrence ceded and net components by the
natural (Choquet gradient) rule, evaluated through the kappa curve that phase 1
put on the joint (``tests/test_bivariate_exeqa.py``).

The mathematics: with ``G = C + N`` split by an **occurrence** program, ``N`` is
not comonotone with ``G``, so the split is not a matter of pricing two margins.
Conditioning on ``G`` reduces the natural allocation to
``A_C = sum_i kappa_C(g_i) Delta_gS_i``, and ``A_C + A_N = rho_g(G)`` holds by
construction because ``kappa_C + kappa_N`` is the identity.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build
from aggregate.spectral import Distortion


# A share cession on a discrete severity whose halves land on the joint
# lattice: the bilinear scatter never splits, so the kappa curve is exact and
# the allocated fractions are exactly the share.
QS_ALIGNED = 'agg NA.QS dfreq [3] dsev [2:20:2] occurrence net of 50% po inf xs 0'
# A discrete excess layer on the lattice: no grid deficit, so the identity
# distortion has no parked tail mass to carry and reads the plain means.
DISCRETE = 'agg NA.D dfreq [3] dsev [1:20] occurrence net of 6 xs 6'
# An ordinary excess layer: the case the plan was written for.
LAYER = (
    'agg NA.L 10 claims sev lognorm 50 cv 1.5 '
    'occurrence net of 50 xs 50 poisson'
)
# Pinned joint grid, so the view pairs land on one common gross axis.
PIN = dict(bs=4, log2_x=10, log2_y=10)


@pytest.fixture(scope='module')
def layer_agg():
    return build(LAYER)


@pytest.fixture(scope='module')
def layer_gc(layer_agg):
    return layer_agg.occ_bivariate(views=('gross', 'ceded'), **PIN)


DISTORTIONS = [Distortion('ph', 0.7), Distortion('wang', 0.3),
               Distortion('tvar', 0.9), Distortion('dual', 2.0)]
IDENTITY = Distortion('ph', 1.0)


def test_identity_recovers_the_means():
    """Under g(s) = s the allocation is the component means.

    On a lattice discrete program, where the joint carries no grid deficit and
    so no parked tail mass for the Choquet weights to charge at the top atom.
    """
    biv = build(DISCRETE).occ_bivariate(views=('gross', 'ceded'), bs=1)
    df = biv.natural_allocation(IDENTITY)
    assert df['P'].to_numpy() == pytest.approx(df['L'].to_numpy(), abs=1e-10)
    assert df.loc['ceded', 'L'] + df.loc['net', 'L'] == pytest.approx(
        df.loc['gross', 'L'], rel=1e-12)
    assert df['LR'].to_numpy() == pytest.approx(1.0)


def test_identity_carries_the_parked_deficit(layer_gc):
    """Under a grid deficit the identity price is the mean plus the parking.

    ``choquet_weights`` runs forwards, which parks unrepresented mass at the
    largest represented outcome: the conservative reading, and the reason this
    exceeds the plain mean rather than equalling it. Recorded so the gap is
    not mistaken for an allocation error.
    """
    df = layer_gc.natural_allocation(IDENTITY)
    parked = layer_gc.deficit * layer_gc.axis_xs[0][-1]
    assert df.loc['gross', 'P'] - df.loc['gross', 'L'] == pytest.approx(
        parked, rel=1e-6)


@pytest.mark.parametrize('dist', DISTORTIONS, ids=lambda d: d.name)
def test_components_foot_to_the_total(layer_gc, dist):
    """Ceded plus net is gross, exactly, in both premium and loss."""
    df = layer_gc.natural_allocation(dist)
    for col in ('L', 'M', 'P'):
        assert df.loc['ceded', col] + df.loc['net', col] == pytest.approx(
            df.loc['gross', col], rel=1e-12)


@pytest.mark.parametrize('dist', DISTORTIONS, ids=lambda d: d.name)
def test_quota_share_fractions(dist):
    """A 50% share allocates half the premium under every distortion.

    The kappa curve is ``g / 2`` at every point, so the weights cannot matter:
    whatever ``g`` does to the increments, it does to both halves.
    """
    biv = build(QS_ALIGNED).occ_bivariate(views=('gross', 'ceded'), bs=1)
    df = biv.natural_allocation(dist)
    assert df.loc['ceded', 'P'] / df.loc['gross', 'P'] == pytest.approx(
        0.5, abs=1e-10)


def test_gross_row_ties_to_price(layer_gc):
    """The gross row is ``Distortion.price`` on the joint's own marginal."""
    import pandas as pd

    dist = Distortion('ph', 0.7)
    df = layer_gc.natural_allocation(dist)
    marginal = pd.Series(layer_gc.marginals[0], index=layer_gc.axis_xs[0])
    assert df.loc['gross', 'P'] == pytest.approx(
        dist.price(marginal, kind='ask').ask, rel=1e-12)


def test_premium_argument_scales_the_fractions(layer_gc):
    """A caller supplied ``P`` is split by the fractions, and ties exactly."""
    dist = Distortion('ph', 0.7)
    free = layer_gc.natural_allocation(dist)
    pinned = layer_gc.natural_allocation(dist, P=1000.0)
    assert pinned.loc['gross', 'P'] == pytest.approx(1000.0, rel=1e-14)
    assert pinned.loc['ceded', 'P'] + pinned.loc['net', 'P'] == pytest.approx(
        1000.0, rel=1e-14)
    share = free.loc['ceded', 'P'] / free.loc['gross', 'P']
    assert pinned.loc['ceded', 'P'] / 1000.0 == pytest.approx(share, rel=1e-12)


def test_rho_gap_is_small_and_reported(layer_agg, layer_gc):
    """``rho_gap`` measures the joint's grid against the fine 1-D gross law."""
    dist = Distortion('ph', 0.7)
    df = layer_gc.natural_allocation(dist)
    fine = dist.price(layer_agg._reins_view_density('gross'), kind='ask').ask
    assert df.attrs['rho_fine'] == pytest.approx(fine, rel=1e-12)
    assert df.attrs['rho_joint'] == pytest.approx(df.loc['gross', 'P'])
    assert df.attrs['rho_gap'] == pytest.approx(
        df.attrs['rho_joint'] - fine, rel=1e-12)
    assert abs(df.attrs['rho_gap']) / fine < 1e-2


def test_rho_gap_is_a_grid_reading_not_a_premium_reading(layer_gc):
    """The gap describes the two grids, so a caller supplied ``P`` leaves it."""
    dist = Distortion('ph', 0.7)
    free = layer_gc.natural_allocation(dist)
    pinned = layer_gc.natural_allocation(dist, P=1000.0)
    assert pinned.attrs['rho_gap'] == pytest.approx(free.attrs['rho_gap'])


def test_grossnet_pair_agrees_with_grossceded(layer_agg, layer_gc):
    """Either view pair answers the same question, to the scatter."""
    gn = layer_agg.occ_bivariate(views=('gross', 'net'), **PIN)
    dist = Distortion('ph', 0.7)
    a = layer_gc.natural_allocation(dist)
    b = gn.natural_allocation(dist)
    assert list(a.index) == list(b.index) == ['gross', 'ceded', 'net']
    assert b.loc['ceded', 'P'] / b.loc['gross', 'P'] == pytest.approx(
        a.loc['ceded', 'P'] / a.loc['gross', 'P'], rel=5e-3)


def test_axis_order_does_not_matter(layer_agg):
    """Gross on axis 1 reads the same as gross on axis 0."""
    gc = layer_agg.occ_bivariate(views=('gross', 'ceded'), **PIN)
    cg = layer_agg.occ_bivariate(views=('ceded', 'gross'), **PIN)
    dist = Distortion('ph', 0.7)
    a, b = gc.natural_allocation(dist), cg.natural_allocation(dist)
    assert b.loc['ceded', 'P'] == pytest.approx(a.loc['ceded', 'P'], rel=1e-10)


def test_pentagon_octet_shape(layer_gc):
    """The frame speaks the pentagon vocabulary, capital columns blank."""
    from aggregate.pentagon import PENTAGON_STATS

    df = layer_gc.natural_allocation(Distortion('ph', 0.7))
    assert list(df.columns) == PENTAGON_STATS
    assert df[['L', 'M', 'P', 'LR']].notna().all().all()
    assert df[['Q', 'PQ', 'ROE']].isna().all().all()
    assert np.isinf(df['a']).all()
    assert df['LR'].to_numpy() == pytest.approx(
        (df['L'] / df['P']).to_numpy())


def test_netceded_pair_refused(layer_agg):
    """A ``(net, ceded)`` joint carries no gross axis to condition on."""
    nc = layer_agg.occ_bivariate(views=('net', 'ceded'), **PIN)
    with pytest.raises(ValueError, match='gross'):
        nc.natural_allocation(Distortion('ph', 0.7))


def test_copula_mode_refused():
    """Only a netceded joint has a gross total to allocate.

    Built without updating: the structural refusal comes before the
    call-update one, so the caller is told the object is the wrong shape
    rather than told to spend a 2-D FFT finding out.
    """
    biv = build(
        'bivariate NA.Cop 5 claims '
        'agg W dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2 '
        'agg F dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5 '
        'copula normal 0.4 poisson', update=False)
    with pytest.raises(ValueError, match='netceded'):
        biv.natural_allocation(Distortion('ph', 0.7))
