"""Tests for the joint band iterator and the shared conditional probe.

[Joint-Row-Bands], 1.0.0a278, phase two of
``dev/notes-net-natural-allocation.md``. Everything the kappa band needs is a
row-wise fold, and a fold should not have to know whether the density is a
numpy array or a zarr store. :class:`~aggregate.bivariate.JointBandsMixin`
gives both containers one iterator and one ``slice``.

The in-core cases are fast. The one massive case rides the ``slow`` marker
with the rest of the disk-backed suite.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build

LAYER = ('agg JB.L dfreq [1 2] dsev [10 20 30] '
         'occurrence net of 10 xs 10')
COPULA = ('bivariate JB.MV 5 claims '
          'agg A dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2 '
          'agg B dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5 '
          'copula gumbel 0.4 poisson')


@pytest.fixture(scope='module')
def joint():
    return build(LAYER).occ_bivariate(views=('gross', 'ceded')).bivariate


# --- the iterator -----------------------------------------------------------

def test_in_core_yields_one_band(joint):
    """In core the iterator costs nothing: one band, no copy arithmetic."""
    bands = list(joint._row_bands())
    assert len(bands) == 1
    r0, r1, block = bands[0]
    assert (r0, r1) == (0, joint.density.shape[0])
    assert np.array_equal(block, joint.density)


def test_bands_reassemble_the_joint(joint):
    """Any band size reassembles the density exactly, whatever the remainder."""
    for band_rows in (1, 7, 16, 10_000):
        parts = [block for _r0, _r1, block in joint._row_bands(
            band_rows=band_rows)]
        assert np.array_equal(np.vstack(parts), joint.density)


def test_axis_one_yields_the_transpose(joint):
    """``axis=1`` bands the transpose, so a consumer always folds along rows."""
    parts = [block for _r0, _r1, block in joint._row_bands(axis=1,
                                                           band_rows=3)]
    assert np.array_equal(np.vstack(parts), joint.density.T)


def test_band_ranges_are_half_open_and_cover(joint):
    """The reported range is the block's, which is what a consumer indexes by."""
    n0 = joint.density.shape[0]
    seen = 0
    for r0, r1, block in joint._row_bands(band_rows=5):
        assert block.shape[0] == r1 - r0
        assert r0 == seen
        seen = r1
    assert seen == n0


def test_bad_axis_refused(joint):
    with pytest.raises(ValueError, match='axis must be 0 or 1'):
        list(joint._row_bands(axis=2))


def test_copula_joint_bands_too():
    """The iterator is a property of the container, not of netceded mode."""
    bv = build(COPULA).bivariate
    parts = [block for _r0, _r1, block in bv._row_bands(band_rows=64)]
    assert np.array_equal(np.vstack(parts), bv.density)


# --- the shared probe -------------------------------------------------------

def test_slice_conditions_on_either_axis(joint):
    """One conditional law, one method name, both directions."""
    row = joint.slice(x=20.0)
    assert row.p.sum() == pytest.approx(1.0)
    assert len(row.p) == len(joint.axis1)
    col = joint.slice(y=10.0)
    assert col.p.sum() == pytest.approx(1.0)
    assert len(col.p) == len(joint.axis0)


def test_slice_is_the_normalized_row(joint):
    """The probe is exactly the joint's row, renormalized. No interpolation."""
    idx = int(round(20.0 / joint.bs0))
    raw = joint.density[idx, :]
    assert joint.slice(x=20.0).p == pytest.approx(raw / raw.sum())


def test_slice_names_the_conditioning_event(joint):
    """A conditional law that cannot say what it is conditioned on is a trap."""
    name = joint.slice(x=20.0).name
    assert 'Gross' in name and '20' in name and '|' in name


def test_slice_needs_exactly_one_value(joint):
    with pytest.raises(ValueError, match='exactly one'):
        joint.slice()
    with pytest.raises(ValueError, match='exactly one'):
        joint.slice(x=10.0, y=10.0)


def test_slice_refuses_a_null_event(joint):
    """A conditional expectation given a null event has no value."""
    top = float(joint.axis0[-1])
    with pytest.raises(ValueError, match='no mass on the conditioning slice'):
        joint.slice(x=top)


def test_slice_reached_the_in_core_container():
    """The probe existed only on the massive container before a278."""
    bv = build(COPULA).bivariate
    assert hasattr(bv, 'slice')
    assert bv.slice(x=float(bv.axis0[1])).p.sum() == pytest.approx(1.0)


# --- the disk boundary ------------------------------------------------------

@pytest.mark.slow
def test_massive_bands_match_in_core(tmp_path):
    """The whole point: a fold written against the iterator is route blind."""
    pytest.importorskip('zarr')
    agg = build(LAYER)
    core = agg.occ_bivariate(views=('gross', 'ceded')).bivariate
    disk = agg.occ_bivariate(views=('gross', 'ceded'),
                             store_dir=str(tmp_path / 'jb')).bivariate
    assert disk.density.shape == core.density.shape
    banded = np.vstack([block for _r0, _r1, block in disk._row_bands()])
    assert np.allclose(banded, core.density, atol=1e-12)
    # the store's own chunking is the default read size, so a disk band is
    # bounded whatever the grid is
    bands = list(disk._row_bands())
    assert max(block.shape[0] for _r0, _r1, block in bands) <= max(
        disk.density.chunks[0], 1)
    assert disk.slice(x=20.0).p == pytest.approx(
        core.slice(x=20.0).p, abs=1e-12)
