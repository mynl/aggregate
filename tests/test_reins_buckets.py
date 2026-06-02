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
``src/aggregate/agg/test_decl.agg`` under the reinsurance section.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build
from aggregate.constants import REINS_BUCKET_DEFAULT, VALIDATION_NOISE
from aggregate.utilities import make_ceder_netter, _validate_reins_layers


# An excess layer with an off-grid attachment relative to the recommended
# bucket: 0.8 share of 250 xs 175 on a lognormal book.
OFF_GRID_PROG = (
    'agg ReBucket 10 claims sev lognorm 100 cv 2 '
    'occurrence net of 0.8 so 250 xs 175 poisson'
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
