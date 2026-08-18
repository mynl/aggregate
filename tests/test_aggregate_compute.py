"""Direct unit tests for the extracted FFT convolution kernel (Phase 2A).

``freq_sev_convolution`` is the FFT-PGF-iFFT core pulled out of ``Aggregate`` so
it can be exercised on hand-built severity vectors and a frequency PGF, without
a full ``Aggregate.update()``. These augment (do not replace) the end-to-end
golden tests; they pin the kernel against brute-force ``np.convolve`` and the
analytic compound moments, and confirm the extraction reproduces the real
``update()`` byte-for-byte.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build
from aggregate._aggregate_compute import freq_sev_convolution


@pytest.fixture
def small_sev():
    """A 3-atom severity on a length-256 bs=1 grid (mean 1.7)."""
    N = 256
    sev = np.zeros(N)
    sev[1], sev[2], sev[3] = 0.5, 0.3, 0.2
    return {'sev': sev, 'N': N, 'bs': 1.0, 'mean': 1 * 0.5 + 2 * 0.3 + 3 * 0.2}


def test_fixed_count_matches_np_convolve(small_sev):
    """Fixed K = k: the aggregate is the k-fold self-convolution of the
    severity, so the FFT kernel must equal ``np.convolve`` repeated k times."""
    sev, N, bs = small_sev['sev'], small_sev['N'], small_sev['bs']
    for k in (1, 2, 3, 5):
        agg, _ = freq_sev_convolution(sev, lambda n, z, k=k: z ** k, k,
                                      N=N, bs=bs, padding=1)
        support = sev[:4]
        direct = support
        for _ in range(k - 1):
            direct = np.convolve(direct, support)
        np.testing.assert_allclose(agg[:len(direct)], direct, atol=1e-13)
        # everything past the (finite) support is numerically zero
        assert np.abs(agg[len(direct):]).max() < 1e-13
        assert agg.sum() == pytest.approx(1.0)


def test_zero_risk_is_point_mass(small_sev):
    """``n == 0`` (no claims) -> a point mass at 0, independent of severity."""
    sev, N, bs = small_sev['sev'], small_sev['N'], small_sev['bs']
    agg, _ = freq_sev_convolution(sev, lambda n, z: z ** 0, 0,
                                  N=N, bs=bs, padding=1)
    assert agg[0] == pytest.approx(1.0)
    assert np.abs(agg[1:]).max() == 0.0


def test_compound_poisson_moments(small_sev):
    """Compound-Poisson aggregate: mean = λ·E[X], variance = λ·E[X²]
    (the Poisson compound identities)."""
    sev, N, bs = small_sev['sev'], small_sev['N'], small_sev['bs']
    lam = 4.0
    x = np.arange(N) * bs
    agg, _ = freq_sev_convolution(sev, lambda n, z: np.exp(lam * (z - 1)), lam,
                                  N=N, bs=bs, padding=2)
    assert agg.sum() == pytest.approx(1.0, abs=1e-9)
    sev_m1 = float((x * sev).sum())
    sev_m2 = float((x ** 2 * sev).sum())
    mean = float((x * agg).sum())
    var = float((x ** 2 * agg).sum()) - mean ** 2
    assert mean == pytest.approx(lam * sev_m1, rel=1e-6)
    assert var == pytest.approx(lam * sev_m2, rel=1e-6)


def test_kernel_reproduces_update_byte_for_byte():
    """Driving the kernel with a built Aggregate's own state reproduces
    ``agg.agg_density`` exactly -- the extraction is behavior-preserving."""
    a = build('agg Dice dfreq [3] dsev [1:6]')
    a.update()
    agg, _ = freq_sev_convolution(
        a.sev_density, a.frequency.freq_pgf, a.n, N=len(a.xs), bs=a.bs,
        i0=a.i0, x_min=a.x_min, one_claim=a.one_claim, padding=1)
    np.testing.assert_array_equal(agg, a.agg_density)


def test_kernel_reproduces_poisson_update():
    """Same byte-for-byte check for a random (Poisson) frequency."""
    a = build('agg Pois 10 claims sev lognorm 50 cv 2 poisson')
    a.update(log2=16, bs=1)
    agg, _ = freq_sev_convolution(
        a.sev_density, a.frequency.freq_pgf, a.n, N=len(a.xs), bs=a.bs,
        i0=a.i0, x_min=a.x_min, one_claim=a.one_claim, padding=1)
    np.testing.assert_array_equal(agg, a.agg_density)


# ----------------------------------------------------------------------
# [Dfreq-One-Claim-Shortcut]: the two spellings of "exactly one claim"
# ----------------------------------------------------------------------

ONE_CLAIM_PROGRAMS = [
    'agg OneFixed 1 claim dsev [1:6] fixed',
    'agg OneDfreq dfreq [1] dsev [1:6]',
    'agg OneFixedCts 1 claim sev lognorm 50 cv 2 fixed',
    'agg OneDfreqCts dfreq [1] sev lognorm 50 cv 2',
]


@pytest.mark.parametrize('program', ONE_CLAIM_PROGRAMS)
def test_one_claim_aggregate_is_the_severity(program):
    """One claim, however spelled: the aggregate IS the severity, exactly.

    ``dfreq [1]`` used to run the identity FFT round trip and land within
    machine epsilon rather than on the nose; both spellings now take the
    severity-copy shortcut, so equality is byte for byte.
    """
    a = build(program)
    assert a.one_claim is True
    np.testing.assert_array_equal(a.agg_density, a.sev_density)


@pytest.mark.parametrize('program', [
    'agg TwoFixed 2 claims dsev [1:6] fixed',       # mean 2
    'agg MeanOne dfreq [0 2] [.5 .5] dsev [1:6]',   # mean 1, support {0, 2}
    'agg PoisOne 1 claim dsev [1:6] poisson',       # mean 1, random count
])
def test_not_one_claim_keeps_the_convolution(program):
    """Negative controls: a mean of one is not a count of one.

    ``dfreq [0 2] [.5 .5]`` is the case the predicate must test on the
    support rather than the mean.
    """
    a = build(program)
    assert a.one_claim is False
    assert not np.array_equal(a.agg_density, a.sev_density)


def test_one_claim_zero_probability_atom_ignored():
    """A declared but impossible outcome does not defeat the support test."""
    a = build('agg OneWithHole dfreq [0 1] [0 1] dsev [1:6]')
    assert a.one_claim is True
    np.testing.assert_array_equal(a.agg_density, a.sev_density)
