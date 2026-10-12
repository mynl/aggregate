"""Empirical aggregate moments: the noise-floor cut and the one-sided deficit.

``[Moment-Deficit-Genuine-Only]``, ``dev/done/plan-1.2.0a14-moment-deficit.md``.
Before 1.2.0a14 the moment sites cleaned the density with an absolute cutoff at
machine epsilon and ``xsden_to_mwrangler`` relocated whatever the cleaned
density fell short of one to the top of the grid, unconditionally. Two kinds of
shortfall are not truncation and were relocated anyway: genuine far tail the
cutoff had deleted inside the grid, and rounding (of either sign). On a tall
grid the cube of the grid top turned either into a large third moment, so
``est_skew`` read 4.9 for ten dice and 11.9 for a lognormal aggregate whose
skewness is 7.28.

Now ``remove_fuzz`` at the moment sites cuts at ``noise_floor``, the density's
own largest negative magnitude capped at machine epsilon, and the wrangler
relocates only a deficit measured on the raw array and above
``VALIDATION_NOISE``. These cases pin the fixes, the three shapes that caught
the plan's rejected first draft, and the accepted residual.

The DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg``
section ``MD``.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import Underwriter, build
from aggregate.moments import VALIDATION_NOISE, xsden_to_mwrangler
from aggregate.utilities import noise_floor, remove_fuzz

EPS = np.finfo(float).eps

# The programs, with the grids the plan measured on.
TEN_DICE = 'agg MD.TenDice dfreq [10] dsev [1:6]'
POISSON_DIE = 'agg MD.PoiDie 10 claims dsev [1:6] poisson'
LOGNORM_ONE = 'agg MD.Lognorm 1 claim sev lognorm 200 cv 2 fixed'
GEMACT = 'agg MD.GEMAct 3 claims sev lognorm 84541.68 cv 2.1022565437545127 poisson'
PROBE = 'agg MD.Probe 10 claims sev lognorm 100 cv 3 poisson'

# ``J.Re14`` has no theoretical skewness (a net-of-reinsurance mixture); this
# is the value every no-relocation rule gives at every grid tried.
J_RE14_SKEW = -7.691


@pytest.fixture(scope='module')
def uw():
    return Underwriter(databases=['library', '_test_suite'])


def theory_skew(a):
    return float(a.validation_df.loc['Agg', 'Sk'])


def legacy_rule(a):
    """``(mean, cv, skew)`` under the rule shipped before 1.2.0a14.

    Clean at the absolute machine-epsilon cutoff, then relocate the cleaned
    density's whole shortfall to the grid top. Routed through the wrangler
    itself rather than written out, because on a window with a mean of 1.5e7
    and a skewness of 4e-4 the third central moment is a 1e-14 relative
    residue of ``E[X^3]``, and any other summation order moves it by percent.
    Exact only when the cleaned shortfall is above the new gate, which every
    caller's shape satisfies and the assertion checks.
    """
    den = remove_fuzz(a.agg_density)
    pg = 1 - den.sum()
    assert pg > VALIDATION_NOISE
    return xsden_to_mwrangler(a.xs, den, deficit=pg).mcvsk


# ---------------------------------------------------------------- the fixes

def test_ten_dice_tall_grid_is_symmetric():
    """Ten dice at 2**20: rounding relocated to the edge read 4.9; now 0."""
    a = build(TEN_DICE, bs=1, log2=20)
    assert theory_skew(a) == 0.0
    assert abs(a.est_skew) < 1e-6


def test_poisson_die_tall_grid_matches_theory():
    a = build(POISSON_DIE, bs=1, log2=20)
    assert abs(a.est_skew - theory_skew(a)) < 1e-6


def test_lognormal_tall_grid_keeps_its_tail():
    """One lognormal claim at 2**21: the absolute cutoff deleted real tail
    and the relocation made it 142; the noise-floor cut keeps it."""
    a = build(LOGNORM_ONE, bs=6.427, log2=21)
    assert abs(a.est_skew - theory_skew(a)) / theory_skew(a) < 2e-3


def test_gemact_table_two_grid():
    """GEMAct's own grid, 2**22 buckets of 500: 11.9 before, theory 7.284."""
    a = build(GEMACT, bs=500, log2=22)
    assert abs(a.est_skew - theory_skew(a)) / theory_skew(a) < 1e-3


def test_reinsured_mixture_default_grid_no_longer_relocates_rounding(uw):
    """``J.Re14`` at its default grid: raw deficit 4.6e-13 is below the gate."""
    a = uw.build('agg.J.Re14', log2=16)
    assert 0 < 1 - a.agg_density.sum() < VALIDATION_NOISE
    assert abs(a.est_skew - J_RE14_SKEW) / abs(J_RE14_SKEW) < 1e-4


def test_reinsured_mixture_taller_grids_are_the_accepted_residual(uw):
    """At 2**18 and 2**20 the same rounding grows past the 1e-12 gate and is
    still relocated, exactly as before. The plan accepts this: wrong stays
    wrong, nothing right became wrong. The principled fix is to measure the
    mass the padding slice discards; see ``dev/TODO.md``."""
    for log2 in (18, 20):
        a = uw.build('agg.J.Re14', log2=log2)
        deficit = 1 - a.agg_density.sum()
        assert deficit > VALIDATION_NOISE
        assert noise_floor(a.agg_density) == 0.0
        assert np.isclose(a.est_skew, legacy_rule(a)[2], rtol=1e-9)
        assert abs(a.est_skew - J_RE14_SKEW) > 1


# ---------------------------------------------------------------- the guards

def test_signed_window_deficit_is_genuine_and_still_relocates(uw):
    """A signed window has a real two-sided deficit (6e-8). The rejected draft
    switched relocation off under ``normalize`` and read -871 here."""
    a = uw.build('agg.WindowedSimple')
    assert 1 - a.agg_density.sum() > VALIDATION_NOISE
    assert np.isclose(a.est_skew, legacy_rule(a)[2], rtol=1e-6)


def test_mixed_frequency_tight_grid_still_relocates(uw):
    """Inverse Gaussian mixing at ``log2=6``: deficit 2e-4 is genuine
    truncation and must keep relocating with ``normalize=True``."""
    a = uw.build('agg.Cb.Freq12.IG')
    assert 1 - a.agg_density.sum() > 1e-4
    assert np.isclose(a.est_skew, legacy_rule(a)[2], rtol=1e-6)


def test_artifact_bucket_cap_binds(uw):
    """A zero-modified binomial carries one negative bucket at 4.5e-4; the
    floor must cap at machine epsilon or that one bucket zeroes real mass."""
    a = uw.build('agg.Cc.Freq24.Binomial')
    raw = a.agg_density
    assert -raw[raw < 0].min() > 1e-4
    assert noise_floor(raw) == EPS
    capped = xsden_to_mwrangler(a.xs, remove_fuzz(raw, eps=EPS),
                                deficit=1 - raw.sum()).mcvsk[2]
    uncapped = xsden_to_mwrangler(a.xs, remove_fuzz(raw, eps=-raw[raw < 0].min()),
                                  deficit=1 - raw.sum()).mcvsk[2]
    assert np.isclose(a.est_skew, capped, rtol=1e-9)
    assert abs(uncapped - capped) / capped > 1e-3


def test_genuine_truncation_still_relocates_under_normalize():
    """A tight grid under ``normalize=True`` drops 1e-4 of mass past the top;
    ``est_m`` exceeds the plain sum by exactly ``deficit * edge``."""
    a = build(PROBE, bs=50, log2=8)
    assert a.normalize
    raw = a.agg_density
    deficit = 1 - raw.sum()
    assert deficit > 1e-5
    edge = a.xs[-1] + a.bs
    assert np.isclose(a.est_m - np.sum(a.xs * raw), deficit * edge, rtol=1e-9)


# ------------------------------------------------------- unit: the wrangler

DIE_XS = np.arange(1.0, 7.0)
DIE_PS = np.full(6, 1.0 / 6.0)


def _plain_mean(den):
    return float(np.sum(DIE_XS * den))


@pytest.mark.parametrize('deficit', [1e-15, -1e-15, -1e-6])
def test_wrangler_ignores_noise_and_negative_deficits(deficit):
    den = DIE_PS * (1 - deficit)
    ex1 = xsden_to_mwrangler(DIE_XS, den).noncentral[0]
    assert np.isclose(ex1, _plain_mean(den), rtol=0, atol=1e-15)


def test_wrangler_relocates_a_genuine_deficit():
    den = DIE_PS * (1 - 1e-6)
    ex1 = xsden_to_mwrangler(DIE_XS, den).noncentral[0]
    assert np.isclose(ex1, _plain_mean(den) + 1e-6 * 7.0, rtol=1e-12)


def test_wrangler_explicit_deficit_overrides_the_sum():
    """The sum says 0.10 is missing; the caller says 0.02 is. The caller wins:
    the rest was deleted inside the grid, not lost beyond it."""
    den = np.full(6, 0.15)
    ex1 = xsden_to_mwrangler(DIE_XS, den, deficit=0.02).noncentral[0]
    assert np.isclose(ex1, _plain_mean(den) + 0.02 * 7.0, rtol=1e-12)
    ex1_zero = xsden_to_mwrangler(DIE_XS, den, deficit=0.0).noncentral[0]
    assert np.isclose(ex1_zero, _plain_mean(den), rtol=1e-12)


# ------------------------------------------------------ unit: noise_floor

def test_noise_floor_zero_without_negatives():
    assert noise_floor(np.array([0.0, 0.5, 1e-20, 0.5])) == 0.0


def test_noise_floor_is_the_largest_negative_magnitude_below_eps():
    den = np.array([0.5, -3e-18, 0.5, -7e-18, 2e-18])
    floor = noise_floor(den)
    assert floor == np.nextafter(7e-18, np.inf)
    # the strict cut must take the most negative bucket itself
    assert remove_fuzz(den, eps=floor)[3] == 0.0
    assert remove_fuzz(den, eps=floor)[4] == 0.0


def test_noise_floor_caps_at_machine_epsilon():
    assert noise_floor(np.array([0.5, -1e-10, 0.5])) == EPS
    assert noise_floor(np.array([0.5, -EPS, 0.5])) == EPS
