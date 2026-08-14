"""Tests for the netceded joint's grid sizing.

[Sizing-And-Passthrough], 1.0.0a277, phase one of
``dev/notes-net-natural-allocation.md``. Three defects, one story: the joint
could be built with a grid the caller did not ask for and could not reach.

* the exact common lattice is taken when the budget affords it, so the
  comonotone scatter never fires and the kappa curve is exact;
* a pinned grid that cannot be honored inside the budget **raises**, where it
  used to clip the wider axis and answer off a joint carrying half its mass;
* ``occ_bivariate`` passes ``total_log2`` and ``store_dir`` through, and
  ``update(log2=...)`` on that route resizes the joint rather than doing
  nothing.

The programs are small and the grids pinned, so this stays in the fast tier.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build
from aggregate.bivariate import _float_gcd, _netceded_exact_bs

# A discrete excess layer: images 0, 10, 10 and gross 10, 20, 30, so the exact
# common lattice is the severity step and the budget can pay for it.
LAYER_DISCRETE = 'agg NS.D dfreq [1 2] dsev [10 20 30] occurrence net of 10 xs 10'
# An unbounded lognormal: the exact lattice is far finer than any affordable
# grid, which is the ordinary case and the one the fallback exists for.
LAYER_WIDE = ('agg NS.W 10 claims sev lognorm 50 cv 1.5 '
              'occurrence net of 50 xs 50 poisson')


@pytest.fixture(scope='module')
def wide():
    return build(LAYER_WIDE)


# --- the exact lattice ------------------------------------------------------

def test_exact_lattice_is_taken_when_affordable():
    """A discrete layer lands on one lattice, so the scatter never splits."""
    bv = build(LAYER_DISCRETE).occ_bivariate(views=('gross', 'ceded'))
    sizing = bv._nc_sizing
    assert sizing.exact
    assert sizing.bs == pytest.approx(sizing.bs_exact)
    # exactness is a claim about the scatter, so check the scatter: every
    # per-claim image is a grid point, no bilinear split anywhere.
    for image in (sizing.x0, sizing.x1):
        offset = np.abs(image / sizing.bs - np.round(image / sizing.bs))
        assert offset.max() < 1e-9
    assert abs(bv.deficit) < 1e-12


def test_exact_lattice_reported_in_prose():
    """The one thing the numbers cannot say is said in ``bs_explanation``."""
    exact = build(LAYER_DISCRETE).occ_bivariate(views=('gross', 'ceded'))
    assert 'never splits' in exact.bs_explanation
    assert 'exact' in exact.bs_explanation


def test_unaffordable_lattice_falls_back_and_says_so(wide):
    """An unbounded severity cannot afford exactness; the report is not silent."""
    bv = wide.occ_bivariate(views=('gross', 'ceded'))
    assert not bv._nc_sizing.exact
    assert bv._nc_sizing.bs_exact is None
    assert 'accurate to that smear rather than exact' in bv.bs_explanation
    # the fallback still measures each axis independently: a cession's two
    # windows are structurally asymmetric and a square grid would say otherwise
    n0, n1 = (len(x) for x in bv.axis_xs)
    assert n0 > n1


def test_exact_bs_early_exit_matches_the_full_gcd():
    """The affordability short circuit does not change the answer it returns."""
    x0 = np.array([0.0, 10.0, 20.0, 30.0])
    x1 = np.array([0.0, 0.0, 10.0, 10.0])
    # a budget that can pay: the folded gcd comes back
    assert _netceded_exact_bs(x0, x1, 10.0, 30.0, 10.0, 20) == pytest.approx(10.0)
    # a budget that cannot: None, rather than a value nobody can use
    assert _netceded_exact_bs(x0, x1, 1e-6, 30.0, 10.0, 20) is None


def test_float_gcd_is_the_one_implementation():
    """``_lattice_bs`` and the netceded lattice fold the same helper."""
    assert _float_gcd(12.0, 8.0) == pytest.approx(4.0)
    assert _float_gcd(0.25, 0.1) == pytest.approx(0.05)


# --- the refusal ------------------------------------------------------------

def test_pinned_bs_over_budget_raises(wide):
    """The measured defect: a pinned bs used to clip the gross axis to 512.

    It then answered questions off a joint with a deficit of 0.535 and a
    correlation of -0.2225 for a comonotone pair, after one warning.
    """
    with pytest.raises(ValueError, match='over the budget'):
        wide.occ_bivariate(views=('gross', 'ceded'), bs=0.5)


def test_refusal_names_both_escapes(wide):
    """A refusal a caller cannot act on is only half a refusal."""
    with pytest.raises(ValueError) as exc:
        wide.occ_bivariate(views=('gross', 'ceded'), bs=0.5)
    message = str(exc.value)
    assert 'total_log2=27' in message
    assert 'bs=0.5' in message
    assert 'store_dir' in message


def test_pinned_axes_over_budget_raise(wide):
    """Both axes pinned and over budget: neither number can be honored."""
    with pytest.raises(ValueError, match='over the budget'):
        wide.occ_bivariate(views=('gross', 'ceded'), bs=4,
                           log2_x=16, log2_y=12)


def test_a_raised_budget_honors_the_pin(wide):
    """The escape the refusal names actually works."""
    bv = wide.occ_bivariate(views=('gross', 'ceded'), bs=4, total_log2=21)
    assert bv.bs[0] == bv.bs[1] == 4
    assert bv.deficit < 1e-8


def test_default_never_raises(wide):
    """Nothing pinned means something to coarsen, so the default always builds."""
    bv = wide.occ_bivariate()
    assert bv.deficit < 1e-5


# --- passthrough ------------------------------------------------------------

def test_total_log2_buys_resolution(wide):
    """The budget is the knob, and it reaches the sizing."""
    coarse = wide.occ_bivariate(views=('gross', 'ceded'), total_log2=16)
    fine = wide.occ_bivariate(views=('gross', 'ceded'), total_log2=22)
    assert fine.bs[0] < coarse.bs[0]
    assert len(fine.axis_xs[0]) * len(fine.axis_xs[1]) <= 2 ** 22
    assert len(coarse.axis_xs[0]) * len(coarse.axis_xs[1]) <= 2 ** 16


def test_update_log2_resizes_the_joint(wide):
    """``update(log2=...)`` was a no-op on this route through 1.0.0a276.

    The over-budget warning recommended it by name, which is the pairing the
    notes' [Netceded-Update-Log2-Noop] finding objected to.
    """
    bv = wide.occ_bivariate(views=('gross', 'ceded'))
    before = [len(x) for x in bv.axis_xs]
    bv.update(log2=22)
    after = [len(x) for x in bv.axis_xs]
    assert after != before
    assert after[0] * after[1] <= 2 ** 22


def test_update_log2_pair_pins_the_axes(wide):
    """A pair is per-axis, which is how the notes drove the disk-backed joint."""
    bv = wide.occ_bivariate(views=('gross', 'ceded'), bs=4, total_log2=21)
    bv.update(log2=(12, 8), bs=4)
    assert [len(x) for x in bv.axis_xs] == [1 << 12, 1 << 8]


def test_update_bs_tuple_refused(wide):
    """One aggregate split two ways carries one lattice, not two."""
    bv = wide.occ_bivariate(views=('gross', 'ceded'))
    with pytest.raises(ValueError, match='one common bs'):
        bv.update(bs=(1, 2))


def test_decl_route_keeps_its_reading():
    """On the DecL prefix the two keywords still size the inner aggregate.

    That is what they mean for every other DecL form, and the joint's own
    sizing rides in the constructor's ``nc_kwargs`` there.
    """
    mv = build(f'netceded {LAYER_DISCRETE}', bs=1, log2=12)
    assert mv._nc_agg.bs == 1
    assert mv._nc_agg.log2 == 12
