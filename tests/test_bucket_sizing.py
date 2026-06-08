"""Targeted regressions for the Portfolio combine grid (``best_window``).

The portfolio auto-sizer combines its units' per-unit window choices into one
shared ``(bs, log2)`` grid. Through 1.0.0a48 this used a root-sum-square
combine (:meth:`Portfolio.best_bucket`) that scaled the wrong way -- adding
units *coarsened* the grid -- and ignored the integer lattice entirely, so an
all-integer discrete book got a fine continuous ``bs`` over the full ``log2``
cap. 1.0.0a49 replaced it with :meth:`Portfolio.best_window`, the
*resolution + span* rule::

    bs = round_bucket(max(min_k bs_k, W_tot / N))

with ``min_k bs_k`` the finest unit bucket (resolution) and ``W_tot / N`` the
no-wrap span floor. These tests pin the three behaviours that rule must deliver
and that the old RMS combine got wrong; they are deliberately small and fast.

See ``dev/done/plan-bucket-combine.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from aggregate import build


def test_motivating_signed_port_uses_unit_resolution():
    """The headline bug: a point-mass + signed-integer book must keep ``bs=1``.

    ``A1`` is a point mass at 12 (``dfreq[4] dsev[3]``) and ``A2`` is
    integer-valued on ``[-3, 3]`` (``dfreq[3] dsev[-1 1]``); both units want
    ``bs=1``. The RMS combine returned ``round_bucket(sqrt(1**2 + 1**2)) =
    round_bucket(1.4142) = 2`` -- adding a second unit coarsened the grid. The
    resolution rule takes ``min_k bs_k = 1`` instead.
    """
    p = build('port BucketMotiv agg M1 dfreq[4] dsev[3] agg M2 dfreq[3] dsev[-1 1]')
    assert p.bs == 1.0, f'expected bs=1, got {p.bs}'
    # signed combine, summed support [9, 15]; no mass lost to wrap.
    assert p._signed()
    assert float(np.sum(p.density_df['p_total'])) == \
        pytest.approx(1.0, abs=1e-9)


def test_tiny_discrete_port_shrinks_log2_and_keeps_bs_one():
    """An all-integer discrete book sizes at ``bs=1`` with ``log2 < cap``.

    ``D1`` (``dfreq[2] dsev[1 2]``) has support ``[2, 4]`` and ``D2``
    (``dfreq[3] dsev[1 2 3]``) has support ``[3, 9]``; the summed support is
    ``[5, 13]``, which fits in ~16 buckets. The RMS combine ignored the lattice
    and returned a fine continuous ``bs`` (~``1/4096``) over the full
    ``log2=16`` cap. ``best_window`` reads each unit's exact-discrete ``bs=1``
    and shrinks ``log2`` to just hold the sum.
    """
    p = build('port BucketTinyDisc agg D1 dfreq[2] dsev[1 2] agg D2 dfreq[3] dsev[1 2 3]')
    assert p.bs == 1.0, f'expected bs=1 on an integer lattice, got {p.bs}'
    assert p.log2 < 16, f'expected log2 shrunk below the cap, got {p.log2}'
    # the grid still holds the full summed support [5, 13]
    assert p.bs * (1 << p.log2) >= 13.0
    assert float(np.sum(p.density_df['p_total'])) == \
        pytest.approx(1.0, abs=1e-9)


def test_fat_tailed_multiline_port_coarsens_via_span():
    """Two fat-tailed lines: the span floor dominates and coarsens ``bs``.

    The summed support is far wider than any single line's, so ``W_tot / N``
    exceeds the per-unit resolution and the grid is *correctly* coarsened to
    hold the sum without aliasing. Guards against the rule under-sizing a wide
    book: ``bs`` must be sensibly large and essentially no mass may wrap.
    """
    p = build('port BucketFat '
              'agg F1 100 claims sev lognorm 100 cv 2 poisson '
              'agg F2 80 claims sev lognorm 120 cv 1.5 poisson')
    assert not p._signed()
    # the finest unit bucket (resolution); span must drive bs strictly above it
    unit_bs = min(a.bs for a in p.agg_list)
    assert p.bs >= unit_bs, f'span floor should not size finer than a unit: {p.bs} < {unit_bs}'
    assert p.bs >= 1.0, f'a wide fat-tailed book should not size absurdly fine: {p.bs}'
    # no aliasing/wrap: total probability is conserved
    assert float(np.sum(p.density_df['p_total'])) > 1.0 - 1e-6


def test_best_bucket_retained_for_comparison():
    """``best_bucket`` (the deprecated RMS combine) is kept but off the live path.

    It is retained only as a side-by-side comparison aid (``DELETE BEFORE
    BETA``); ``build`` now routes through ``best_window``. On the integer
    discrete book the two visibly disagree -- RMS returns a fine continuous
    bucket, ``best_window`` returns ``1`` -- which is the whole point.
    """
    p = build('port BucketCmp agg C1 dfreq[2] dsev[1 2] agg C2 dfreq[3] dsev[1 2 3]')
    rms = p.best_bucket(16)
    assert rms < p.bs, f'RMS ({rms}) should be finer than the live best_window bs ({p.bs})'
    bs, log2, x_min = p.best_window(16, 0)
    assert (bs, x_min) == (1.0, 0.0)
    assert log2 < 16
