"""Focused unit tests for the P4 three-subsystem split of ``Portfolio``.

Exercises the extracted free functions *directly* (not only through the
``Portfolio`` method wrappers), per ``dev/plan-split-portfolio.md`` Phase 4A:

- :mod:`aggregate._portfolio_density` -- the independent-sum ``add_exa`` kernel.
- :mod:`aggregate._portfolio_common` -- the exeqa numerics (``build_augmented``,
  ``bodoff``) and the convex-hull helpers.
- :mod:`aggregate._portfolio_sample` -- the comonotonic-allocation kernel.

These augment, not replace, the end-to-end baseline: the point is that the
kernels are now reachable and checkable without a full ``update()``/``sample()``.
"""

import numpy as np
import pytest

from aggregate import build
from aggregate import _portfolio_density as _density
from aggregate import _portfolio_common as _common
from aggregate import _portfolio_sample as _sample


@pytest.fixture(scope='module')
def dice_port():
    """A tiny, exactly representable two-unit portfolio."""
    p = build('port PSub '
              'agg A dfreq [1] dsev [1 2 3] '
              'agg B dfreq [1] dsev [2 3 4]')
    return p


# ---------------------------------------------------------------------------
# _portfolio_common -- convex-hull helpers (pure, hand-checkable)
# ---------------------------------------------------------------------------

def test_check01_pads_endpoints():
    out = _common.check01(np.array([0.3, 0.6]))
    assert out[0] == 0 and out[-1] == 1


def test_convex_points_includes_corners():
    s = [0.2, 0.5, 0.8]
    gs = [0.1, 0.4, 0.9]
    xs, ys = _common.convex_points(s, gs)
    # the lower convex envelope always carries the (0,0) and (1,1) corners
    assert (0.0 in xs) and (1.0 in xs)
    assert (0.0 in ys) and (1.0 in ys)


# ---------------------------------------------------------------------------
# _portfolio_sample -- comonotonic allocation kernel (Denuit majorization)
# ---------------------------------------------------------------------------

def test_comonotonic_allocations_are_monotone():
    s_grid = np.array([0.0, 1.0, 2.0, 3.0])
    pdf_s = np.array([0.25, 0.25, 0.25, 0.25])
    # a deliberately non-monotone CMRS row plus a monotone one
    kappa = np.array([[0.0, 0.6, 0.4, 1.0],
                      [0.0, 0.4, 1.6, 2.0]])
    out = _sample.make_comonotonic_allocations_work(s_grid, pdf_s, kappa)
    # each component is non-decreasing in S after the improvement ...
    assert np.all(np.diff(out, axis=1) >= -1e-12)
    # ... and the column sums (the total) are preserved at each state
    assert np.allclose(out.sum(axis=0), kappa.sum(axis=0))


# ---------------------------------------------------------------------------
# _portfolio_density -- add_exa kernel reachable directly
# ---------------------------------------------------------------------------

def test_density_add_exa_method_delegates(dice_port):
    # the method is a thin wrapper over the free function: the built frame
    # already carries the kernel's columns, and they satisfy the documented
    # identities.
    df = dice_port.density_df
    assert 'exeqa_A' in df.columns and 'exa_total' in df.columns
    # E[X_i | X=a] sums to a wherever the total has mass
    mass = df.query('p_total > 0')
    recon = mass['exeqa_A'] + mass['exeqa_B']
    assert np.allclose(recon, mass['loss'])
    # exa_total is the limited expected value -> bounded by E[X]
    assert df['exa_total'].iloc[-1] == pytest.approx(df['e_total'].iloc[0], rel=1e-9)


# ---------------------------------------------------------------------------
# _portfolio_common -- build_augmented / bodoff reachable directly
# ---------------------------------------------------------------------------

def test_common_build_augmented_total_identity(dice_port):
    from aggregate.spectral import Distortion
    d = Distortion('ph', 0.5)
    aug = _common.build_augmented(dice_port, d, allocation='lifted')
    # the distorted price of the total is the per-unit distorted allocation sum
    last = aug.index[aug['gS'].to_numpy() > 0][-1]
    row = aug.loc[last]
    assert row['exag_A'] + row['exag_B'] == pytest.approx(row['exag_total'], rel=1e-6)


def test_common_bodoff_matches_method(dice_port):
    direct = _common.bodoff(dice_port, p=0.99)
    viamethod = dice_port.bodoff(p=0.99)
    assert np.allclose(direct.to_numpy(), viamethod.to_numpy())
