"""Tests for selectable reinsurance rebucketing and layer-order validation.

Covers the ``reins-buckets`` change (1.0.0a18):

- ``Aggregate.reins_bucket`` switch (``'linear'`` vs ``'nearest'``) and its
  validating setter;
- the vectorized scatter in ``Aggregate._apply_reins_work`` /
  ``_rebucket_to_grid`` -- ``'linear'`` preserves the first moment exactly,
  ``'nearest'`` to within ``bs/2``, both preserve total mass;
- the hard-error layer-order validator ``_validate_reins_layers`` invoked at
  the top of ``make_ceder_netter``.

The DecL programs used here are mirrored in
``src/aggregate/agg/decl-testers.agg`` under the reinsurance section.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build
from aggregate.config import get_settings
from aggregate.distributions import make_ceder_netter, _validate_reins_layers

VALIDATION_NOISE = get_settings().validation.noise
REINS_BUCKET_DEFAULT = get_settings().discretization.reins_bucket


# An excess layer with an off-grid attachment relative to the recommended
# bucket: 0.8 share of 250 xs 175 on a lognormal book.
OFF_GRID_PROG = (
    'agg ReBucket 10 claims sev lognorm 100 cv 2 '
    'occurrence net of 0.8 po 250 xs 175 poisson'
)


def _sev_mean(a, col):
    """First moment of a severity density column of ``reins_density_df``."""
    rd = a.reins_density_df
    return float(np.sum(a.xs * rd[col]))


def test_default_reins_bucket():
    a = build(OFF_GRID_PROG, update=False)
    assert a.reins_bucket == REINS_BUCKET_DEFAULT == 'linear'


def test_reins_bucket_setter_validates():
    a = build(OFF_GRID_PROG, update=False)
    with pytest.raises(ValueError):
        a.reins_bucket = 'banana'
    a.reins_bucket = 'nearest'
    assert a.reins_bucket == 'nearest'


def test_linear_preserves_first_moment():
    """Linear mass-split preserves the gross mean across net + ceded exactly."""
    a = build(OFF_GRID_PROG, update=False)
    a.update(log2=16, reins_bucket='linear')
    gross = _sev_mean(a, 'p_sev_gross')
    recon = _sev_mean(a, 'p_sev_net') + _sev_mean(a, 'p_sev_ceded')
    assert abs(recon - gross) <= 10 * VALIDATION_NOISE * max(1.0, gross)


def test_nearest_within_half_bucket():
    """Nearest scheme keeps the net + ceded mean within bs/2 of gross."""
    a = build(OFF_GRID_PROG, update=False)
    a.update(log2=16, reins_bucket='nearest')
    gross = _sev_mean(a, 'p_sev_gross')
    recon = _sev_mean(a, 'p_sev_net') + _sev_mean(a, 'p_sev_ceded')
    assert abs(recon - gross) <= a.bs / 2


@pytest.mark.parametrize('rb', ['linear', 'nearest'])
def test_mass_conserved(rb):
    """Both schemes preserve total probability mass on each marginal."""
    a = build(OFF_GRID_PROG, update=False)
    a.update(log2=16, reins_bucket=rb)
    rd = a.reins_density_df
    assert abs(rd['p_sev_net'].sum() - 1.0) <= 1e-9
    assert abs(rd['p_sev_ceded'].sum() - 1.0) <= 1e-9


# ----------------------------------------------------------------------------
# layer-order validation
# ----------------------------------------------------------------------------

def test_single_layer_valid():
    # trivially valid; should not raise
    make_ceder_netter([(1, 10, 5)])


def test_ascending_layers_valid():
    make_ceder_netter([(1, 10, 0), (0.5, 30, 20), (0.25, np.inf, 50)])


def test_gap_via_zero_share_accepted():
    # a gap between 10 and 15 expressed as a zero-share layer
    make_ceder_netter([(1, 10, 0), (0, 5, 10), (1, 10, 15)])


def test_out_of_order_raises():
    with pytest.raises(ValueError, match='bottom-up'):
        _validate_reins_layers([(1, 10, 20), (1, 10, 5)])


def test_overlap_raises():
    with pytest.raises(ValueError, match='overlap'):
        _validate_reins_layers([(1, 10, 0), (1, 10, 5)])


def test_inf_top_layer_no_false_overlap():
    # an unlimited top layer must not be treated as overlapping a gap above it
    _validate_reins_layers([(1, 10, 0), (1, np.inf, 10)])


# ----------------------------------------------------------------------------
# ceder knot placement across gaps
#
# ``make_ceder_netter`` emits a layer's left-hand knot only when the layer
# attaches strictly above the running top of the program. Tracking that top by
# accumulation rather than assignment let it run ahead of the truth, so from the
# second gap onward the test silently failed and the ceder interpolated straight
# across the gap instead of holding flat.
# ----------------------------------------------------------------------------

def _ceded_closed_form(reins_list, x):
    """Reference cession: the sum of each layer's own signed-share payout."""
    return sum(share * np.clip(x - attach, 0.0, limit)
               for share, limit, attach in reins_list)


def test_two_gaps_hold_flat_across_each_gap():
    # 100 xs 0, gap, 100 xs 200, gap, 100 xs 400: the ceder must be flat at 100
    # over (100, 200) and flat at 200 over (300, 400).
    layers = [(1, 100, 0), (1, 100, 200), (1, 100, 400)]
    ceder, _netter, xs, ys = make_ceder_netter(layers, debug=True)
    # a knot at every attachment and every layer top
    assert xs == [0, 100, 200, 300, 400, 500, np.inf]
    assert ys == [0, 100, 100, 200, 200, 300, 300]
    assert ceder(350.0) == 200.0        # inside the upper gap
    assert ceder(400.0) == 200.0        # the third layer's attachment
    assert ceder(150.0) == 100.0        # inside the lower gap


@pytest.mark.parametrize('layers', [
    [(1, 100, 0), (1, 100, 200), (1, 100, 400)],            # two gaps
    [(1, 100, 0), (1, 100, 200)],                           # one gap
    [(1, 100, 0), (1, 100, 100), (1, 100, 200)],            # contiguous
    [(0.5, 100, 50), (0.25, 200, 300), (1, 100, 700)],      # shares and gaps
    [(1, 10, 0), (0, 5, 10), (1, 10, 15)],                  # zero-share filler
])
def test_ceder_matches_closed_form(layers):
    """The interpolated ceder equals the sum of the layers' own payouts."""
    ceder, netter = make_ceder_netter(layers)
    x = np.linspace(0, 1000, 2001)
    assert np.allclose(ceder(x), _ceded_closed_form(layers, x),
                       atol=VALIDATION_NOISE)
    # netter is the complement by construction
    assert np.allclose(netter(x), x - ceder(x), atol=VALIDATION_NOISE)


def test_docstring_example_knots():
    """The knot table in the ``make_ceder_netter`` docstring stays true."""
    _c, _n, xs, ys = make_ceder_netter([(1, 10, 0), (0.5, 30, 20)], debug=True)
    assert xs == [0, 10, 20, 50, np.inf]
    assert ys == [0, 10, 10, 25, 25]
