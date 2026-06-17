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

import warnings

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


# ---------------------------------------------------------------------------
# Plan B -- the non-zero aggregate output window for high-mean / thin-tail
# aggregates. A concentrated aggregate (agg_cv < 1/z, so its mass band clears 0)
# is computed on a two-sided window far from 0 via the benign FFT wrap: the
# severity is laid at period M*bs and the finished aggregate relabelled by
# round(x_min/bs) (modular np.roll). See dev/done/plan-bucket-window.md.
# ---------------------------------------------------------------------------

def test_window_discrete_high_mean_bs_one():
    """The motivating case: 10M integer claims resolve at ``bs=1`` on a window.

    ``10000000 claims dsev [1 2]`` has mean 15M and sd ~5000 -- a band ~70k
    wide riding ~15M above 0. The 0-based grid wastes its resolution on the
    empty ``[0, 15M]`` (``bs`` ~ hundreds); the windowed grid keeps the exact
    integer lattice ``bs=1`` by growing ``log2`` one notch past the cap, with a
    non-zero origin and matching moments.
    """
    a = build('agg Window 10000000 claims dsev [1 2] poisson')
    df = a._bs_window_df
    assert bool(df.loc['windowed', 'selected'])
    assert a.bs == 1.0, f'expected bs=1 on the integer lattice, got {a.bs}'
    assert a.x_min > 14_000_000
    assert a.est_m == pytest.approx(a.agg_m, rel=1e-6)
    assert a.est_cv == pytest.approx(a.agg_cv, rel=1e-3)
    assert a.agg_density.sum() == pytest.approx(1.0, abs=1e-6)


def test_window_continuous_high_mean_matches_moments():
    """A high-mean continuous book windows and reproduces its moments.

    ``100000 claims sev lognorm 100 cv 0.5`` has mean 10M, agg_cv ~ 0.0035 --
    well below ``1/z`` -- so the band clears 0 and the windowed grid is strictly
    finer than the 0-based moment grid. The continuous severity (no lattice)
    sizes ``bs`` from the band width, not from ``x_max``.
    """
    a = build('agg HM 100000 claims sev lognorm 100 cv 0.5 poisson')
    assert bool(a._bs_window_df.loc['windowed', 'selected'])
    assert a.x_min > 9_000_000
    assert a.est_m == pytest.approx(a.agg_m, rel=1e-4)
    assert a.est_cv == pytest.approx(a.agg_cv, rel=1e-3)
    assert a.agg_density.sum() == pytest.approx(1.0, abs=1e-6)


def test_window_large_roll_conserves_mass():
    """The large modular output roll (``j0 ~ 15M``) loses no mass.

    Relabelling the finished aggregate by ``round(x_min/bs)`` is a modular
    ``np.roll``, so an enormous roll -- and any wrap of the band across the
    buffer end (the straddle case) -- is handled automatically. Pinning mass to
    1 guards the wrap is benign (a single clean copy, no overlap).
    """
    a = build('agg Window 10000000 claims dsev [1 2] poisson')
    j0 = round(a.x_min / a.bs)
    assert j0 > 10000000                       # genuinely large modular roll
    assert a.agg_density.sum() == pytest.approx(1.0, abs=1e-6)


def test_window_severity_overflow_falls_back():
    """When a single severity overflows the window, do not window (fall back).

    An ``approximate`` aggregate carries the whole high-mean distribution on a
    *single* fixed-frequency severity, so that one severity already sits at the
    aggregate mean and cannot fit the windowed extent ``[0, N*bs]``. The
    severity-fit guard marks the windowed row inapplicable; selection quietly
    keeps the 0-based grid, and the moments still match the exact aggregate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        exact = build('agg E 5000 claims sev lognorm 100 cv 2 poisson approximate exact')
        approx = build('agg A 5000 claims sev lognorm 100 cv 2 poisson approximate sgamma')
    assert not bool(approx._bs_window_df.loc['windowed', 'applies'])
    assert not bool(approx._bs_window_df.loc['windowed', 'selected'])
    assert approx.x_min == 0.0
    assert approx.est_m == pytest.approx(exact.est_m, rel=1e-3)


def test_window_occ_reins_not_windowed():
    """Occurrence reinsurance suppresses windowing (severity rides ``xs``).

    The occ-reins severity rebucketing and ``reins_density_df`` carry the
    severity on the output grid, which a non-zero window origin would break, so
    a concentrated book with occ reins keeps the 0-based grid and conserves
    mass. Aggregate reinsurance is unaffected.
    """
    a = build('agg OR 100 claims sev lognorm 50 cv 1.5 '
              'occurrence net of 3500 po 4000 xs 1000 poisson '
              'aggregate net of 2000 xs 3000')
    assert 'windowed' not in a._bs_window_df.index
    assert a.x_min == 0.0
    assert a.density_df.p_total.sum() == pytest.approx(1.0, abs=1e-6)


def test_window_ordinary_aggregate_unchanged():
    """An ordinary aggregate (agg_cv > 1/z) is untouched: 0-based, no window.

    Its two-sided window would include 0, so the windowed candidate never
    clears the ``w_lo > 0`` gate; the grid stays byte-for-byte the legacy
    0-based moment grid.
    """
    a = build('agg Ord 5 claims sev lognorm 100 cv 2 poisson')
    assert a.x_min == 0.0
    sel = a._bs_window_df.index[a._bs_window_df.selected][0]
    assert sel != 'windowed'


# ---------------------------------------------------------------------------
# 1A -- symmetric, convention-aware output windowing (plan-bucket-window-2 §1A).
# Per-edge coverage (the convention skew lever), balanced padding (the band sits
# sensibly in the grid rather than jammed at the bottom), the relaxed "no
# coarser" selection gate, and the explicit Regime-B (heavy severity) branch.
# ---------------------------------------------------------------------------

def test_estimate_agg_window_per_edge_coverage():
    """Per-edge coverage: a shallow edge is trimmed in; a deep edge is unmoved.

    The convention-skew lever. A loss covers its upper (priced) edge deep and
    trims the cheap lower edge shallow, so ``p_lo=6`` raises the lower edge
    while ``p_hi=12`` leaves the upper edge at the symmetric value.
    """
    from aggregate.distributions import estimate_agg_window
    both = estimate_agg_window(1000, 100, 0.3, p=12)
    skew = estimate_agg_window(1000, 100, 0.3, p_lo=6, p_hi=12)
    assert skew[0] > both[0]                      # shallow low edge trimmed up
    assert skew[1] == pytest.approx(both[1])      # deep high edge unchanged


def test_window_balanced_padding_redistributes_slack():
    """A windowed band's origin sits below its band-bottom window origin.

    ``_size`` places the window band-bottom (all power-of-2 slack above);
    balanced padding (``window_pad_skew``) pushes a fraction of the slack below
    the band so the log-density plot is not jammed against the grid floor. The
    realized origin is therefore strictly below the ``windowed`` row's
    band-bottom origin, and mass is conserved.
    """
    a = build('agg HM 100000 claims sev lognorm 100 cv 0.5 poisson')
    df = a._bs_window_df
    assert bool(df.loc['windowed', 'selected'])
    band_bottom = float(df.loc['windowed', 'x_min'])
    used = float(df.loc['used', 'x_min'])
    assert used < band_bottom                     # slack pushed below the band
    assert a.x_min == pytest.approx(used)
    assert a.agg_density.sum() == pytest.approx(1.0, abs=1e-6)


def test_window_convention_override_mirrors_placement():
    """``window_convention`` flips the padding skew; loss and payoff mirror.

    A loss leaves more empty grid above the band (room for the priced right
    tail); a payoff is the mirror image. The two placements are symmetric.
    """
    def fracs(conv):
        a = build('agg HM 100000 claims sev lognorm 100 cv 0.5 poisson')
        a.update(window_convention=conv)
        d = a.agg_density
        nz = np.flatnonzero(d > 1e-12)
        top = a.x_min + a.bs * (1 << a.log2)
        below = (a.xs[nz[0]] - a.x_min) / (top - a.x_min)
        above = (top - a.xs[nz[-1]]) / (top - a.x_min)
        return below, above

    lo_below, lo_above = fracs('loss')
    pay_below, pay_above = fracs('payoff')
    assert lo_above > lo_below                     # loss: room on the right
    assert pay_below > pay_above                   # payoff: room on the left
    assert lo_below == pytest.approx(pay_above, abs=0.02)   # mirror
    assert lo_above == pytest.approx(pay_below, abs=0.02)


def test_window_regime_b_heavy_severity_stays_zero_based(caplog):
    """Heavy severity (Regime B): band clears 0 but the origin cannot move.

    ``5000 claims lognorm 100 cv 2`` has its mass band well above 0, but a
    single cv-2 severity overflows the windowed extent, so the benign FFT wrap
    is invalid. The windowed row is recorded but inapplicable; the grid stays
    0-based and a discoverability ``logger.info`` explains why (no warning --
    this is expected, not defective).
    """
    import logging
    with caplog.at_level(logging.INFO, logger='aggregate.distributions'):
        a = build('agg C3 5000 claims sev lognorm 100 cv 2 poisson')
    assert 'windowed' in a._bs_window_df.index
    assert not bool(a._bs_window_df.loc['windowed', 'applies'])
    assert a.x_min == 0.0
    assert any('non-windowable' in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------------
# 1A-fix -- single-big-jump extent floor (heavy / signed severities)
# ----------------------------------------------------------------------


def test_sbj_signed_severity_recovers_mass_no_alias():
    """Signed heavy-left severity: the grid must cover the reach (no aliasing).

    ``10 claims 100 - lognorm 10 cv 2.5`` has *positive* aggregate skew (the
    ``+100^3`` term dominates) but a reflected tail reaching ~-110k. The MoM
    window misses it entirely, so without the single-big-jump floor the severity
    wraps the FFT buffer and ~47% of the mass is lost. The floor forces the grid
    to span the reach (coarsening ``bs`` within the log2 budget if needed), so
    the law is mass-conserving and its mean matches the analytic.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('agg LNS 10 claims sev 100 - lognorm 10 cv 2.5 poisson')
    assert a.x_min < 0.0                                  # grid reaches below 0
    assert a.agg_density.sum() == pytest.approx(1.0, abs=1e-6)
    assert a.est_m == pytest.approx(a.agg_m, rel=1e-3)
    # the sbj row is recorded and its lower edge drove the (negative) origin
    assert 'sbj' in a._bs_window_df.index
    assert float(a._bs_window_df.loc['sbj', 'x_min']) < 0.0


def test_sbj_positive_heavy_extends_when_log2_allows():
    """Positive heavy unlimited sev: a generous log2 captures the full tail.

    ``5000 claims lognorm 100 cv 2`` reaches ~8e5 in the far tail; the default
    grid clips a sliver. With headroom (``log2=19``) the single-big-jump floor
    extends the window to the reach while keeping a fine bulk ``bs``, so the law
    conserves mass and the mean is accurate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('agg T5 5000 claims sev lognorm 100 cv 2 poisson', log2=19)
    assert a.agg_density.sum() == pytest.approx(1.0, abs=1e-9)
    assert a.est_m == pytest.approx(a.agg_m, rel=1e-4)
    assert float(a._bs_window_df.loc['sbj', 'x_max']) > 7e5


def test_sbj_honors_explicit_log2_no_growth():
    """The single-big-jump floor never grows ``log2`` past an explicit request.

    A small heavy book (``5 claims lognorm 10 cv 2``, agg mean 50) has a far
    tail near 47k. At a tight ``log2=12`` the floor must NOT grow the grid (that
    would break the caller/hint contract) nor coarsen the bulk to uselessness:
    it keeps the MoM window and the requested ``log2``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('agg Small 5 claims sev lognorm 10 cv 2 poisson', log2=12)
    assert a.log2 == 12


def test_sbj_big_en_guard_finite_extent():
    """The ``p**`` depth guard keeps the extent finite for a large ``E[N]``.

    With ``E[N]=5000`` the naive ``p** = 1 - (1-p*)/E[N]`` underflows double
    precision (``q_X(p**) -> inf`` for an unbounded sev). ``sbj_tail_floor``
    floors the probe depth, so the recorded single-big-jump extent is finite.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('agg BigEN 5000 claims sev lognorm 100 cv 2 poisson')
    sbj_hi = float(a._bs_window_df.loc['sbj', 'x_max'])
    assert np.isfinite(sbj_hi) and sbj_hi > a.agg_m


def test_sbj_light_book_byte_stable():
    """A light / thin severity is untouched: the floor's ``max``/``min`` no-op.

    ``50 claims gamma 100 cv 1`` is light-tailed; the single-big-jump extent
    sits inside the MoM window, so the selected grid is identical with and
    without the floor (the ``sbj`` row is recorded but does not bind).
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('agg Light 50 claims sev gamma 100 cv 1 poisson')
    assert a.agg_density.sum() == pytest.approx(1.0, abs=1e-9)
    assert a.est_m == pytest.approx(a.agg_m, rel=1e-5)
    assert 'sbj floor' not in str(a._bs_window_df.loc['moment', 'note'])
