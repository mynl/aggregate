"""Mean-preserving (``linear``) bucketing for discrete severities (1.0.0a28).

``dsev_bucket`` controls how discrete-severity atoms (``dsev`` / ``dhistogram``
/ ``fixed``) are placed onto the model grid during discretization:

- ``'linear'`` (default) splits each off-grid atom's mass across its two
  bracketing buckets so the discretized first moment equals ``Σ xₖ pₖ``
  exactly (mean-preserving), mirroring the reinsurance ``reins_bucket`` linear
  scheme;
- ``'nearest'`` snaps each atom to its closest bucket (the historical
  behaviour), biasing the discretized mean by up to ``bs/2`` per atom when the
  atoms are off-grid.

On-grid atoms (integer atoms with ``bs == 1``, e.g. dice) give ``f == 0`` so
the two schemes coincide -- the common case is invariant.

The discretized-mean checks below read ``est_m`` (the *empirical* FFT mean,
which reflects bucketing) rather than ``agg_m`` (the *theoretical* mean, which
is computed from the exact severity moments and so is bias-free regardless of
the placement scheme).

The DecL programs are mirrored in ``src/aggregate/agg/test_decl.agg`` under the
DISC section.
"""

import logging

import numpy as np
import pytest

from aggregate import build

logging.disable(logging.CRITICAL)

# Off-grid atoms with a non-integer bs: every atom misses the grid, so the two
# schemes diverge and the mean bias is visible.
OFF_GRID = 'agg Off dfreq [1] dsev [0.3 1.7 2.4] [.5 .3 .2]'
OFF_GRID_MEAN = 0.3 * .5 + 1.7 * .3 + 2.4 * .2   # 1.14


def _build(program, **kw):
    a = build(program, update=False)
    a.update(**kw)
    return a


# ----------------------------------------------------------------------
# Default and validation
# ----------------------------------------------------------------------
def test_default_is_linear():
    a = build('agg D dfreq [1] dsev [1:6]')
    assert a.dsev_bucket == 'linear'


@pytest.mark.parametrize('bad', ['round', 'Linear', '', None])
def test_invalid_value_rejected(bad):
    if bad is None:
        # None means "use default" on the constructor/update, never invalid.
        return
    a = build('agg D dfreq [1] dsev [1:6]', update=False)
    with pytest.raises(ValueError, match="dsev_bucket must be"):
        a.dsev_bucket = bad


def test_constructor_rejects_invalid():
    from aggregate.distributions import Aggregate
    with pytest.raises(ValueError, match="dsev_bucket must be"):
        Aggregate('x', exp_en=1, freq_name='fixed', sev_name='dhistogram',
                  sev_xs=[1, 2], sev_ps=[.5, .5], dsev_bucket='round')


def test_update_rejects_invalid():
    a = build('agg D dfreq [1] dsev [1:6]', update=False)
    with pytest.raises(ValueError, match="dsev_bucket must be"):
        a.update(bs=1, log2=8, dsev_bucket='round')


# ----------------------------------------------------------------------
# The headline: linear preserves the mean, nearest biases it
# ----------------------------------------------------------------------
def test_linear_preserves_mean_off_grid():
    a = _build(OFF_GRID, bs=0.5, log2=8, dsev_bucket='linear')
    assert a.est_m == pytest.approx(OFF_GRID_MEAN, abs=1e-12)
    assert a.sev_density.sum() == pytest.approx(1.0, abs=1e-12)


def test_nearest_biases_mean_off_grid():
    a = _build(OFF_GRID, bs=0.5, log2=8, dsev_bucket='nearest')
    # atoms 0.3, 1.7, 2.4 snap to 0.5, 1.5, 2.5 -> mean 1.2, biased high
    assert a.est_m == pytest.approx(1.2, abs=1e-12)
    assert a.est_m > OFF_GRID_MEAN          # positive bias here
    assert a.sev_density.sum() == pytest.approx(1.0, abs=1e-12)


# ----------------------------------------------------------------------
# On-grid invariance: the common integer-atom / bs==1 case is unchanged
# ----------------------------------------------------------------------
def test_on_grid_schemes_agree():
    lin = _build('agg D dfreq [1] dsev [1:6]', bs=1, log2=8, dsev_bucket='linear')
    near = _build('agg D dfreq [1] dsev [1:6]', bs=1, log2=8, dsev_bucket='nearest')
    # f == 0 for every atom -> identical up to FFT/normalisation FP noise.
    assert np.allclose(lin.sev_density, near.sev_density, atol=1e-13, rtol=0)
    assert lin.est_m == pytest.approx(near.est_m, abs=1e-12)
    assert lin.est_m == pytest.approx(3.5, abs=1e-12)


# ----------------------------------------------------------------------
# Signed discrete severity: atoms placed against the severity grid origin,
# not the (possibly wider) output window
# ----------------------------------------------------------------------
def test_signed_dsev_linear_correct():
    # Sum of 3 iid X in {-2, 5} (p .5) -> support {-6, 1, 8, 15}, p {1,3,3,1}/8.
    a = build('agg S dfreq [3] dsev [-2 5] [.5 .5]', update=False)
    a.update(log2=6, bs=1, x_min=-8)        # window forced wider than support
    df = a.density_df
    got = {round(float(x)): round(float(p), 10)
           for x, p in zip(df.index, df.p_total) if abs(p) > 1e-9}
    assert got == {-6: 0.125, 1: 0.375, 8: 0.375, 15: 0.125}


# ----------------------------------------------------------------------
# Scope: continuous severities ignore dsev_bucket; layered discrete falls
# through to the cdf-difference path (Phase 1)
# ----------------------------------------------------------------------
def test_continuous_unaffected_by_dsev_bucket():
    prog = 'agg C 10 claims sev lognorm 100 cv 2 poisson'
    lin = _build(prog, bs=1, log2=12, dsev_bucket='linear')
    near = _build(prog, bs=1, log2=12, dsev_bucket='nearest')
    assert np.array_equal(lin.sev_density, near.sev_density)


def test_layered_discrete_falls_through():
    # A layered discrete severity (5 xs 3) discretises via the cdf-difference
    # regardless of dsev_bucket -- Phase 1 covers unlayered atoms only.
    prog = 'agg L 3 claims 5 xs 3 dsev [1 4 8] [.5 .3 .2] fixed'
    lin = _build(prog, bs=1, log2=10, dsev_bucket='linear')
    near = _build(prog, bs=1, log2=10, dsev_bucket='nearest')
    assert np.array_equal(lin.sev_density, near.sev_density)


# ----------------------------------------------------------------------
# info() surfaces dsev_bucket only when a discrete component is present
# ----------------------------------------------------------------------
def test_info_shows_dsev_bucket_for_discrete():
    a = build('agg D dfreq [1] dsev [1:6]')
    assert 'dsev_bucket' in a.info


def test_info_hides_dsev_bucket_for_continuous():
    a = build('agg C 10 claims sev lognorm 100 cv 2 poisson')
    assert 'dsev_bucket' not in a.info
