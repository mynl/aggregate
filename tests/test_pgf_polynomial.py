"""Stage-1 tests for [Empirical-PGF-Horner-Dispatch] (`evaluate_pgf_polynomial`).

Parity of the Horner / sorted-gap square-and-multiply dispatcher against the
legacy ``weights @ z**atoms`` matrix expression, plus end-to-end ``build()``
density regressions. See dev/plan-sparre-a.md.
"""

import numpy as np
import pytest

from aggregate import build
from aggregate._aggregate_compute import evaluate_pgf_polynomial


def _legacy(atoms, weights, z):
    a = np.asarray(atoms, dtype=float)
    return np.asarray(weights, dtype=float) @ np.power(z, a.reshape((-1, 1)))


def _rfft_z(log2=8):
    """rfft-shaped complex evaluation points: transform of a random pmf."""
    rng = np.random.default_rng(2026)
    pm = rng.random(1 << log2)
    pm /= pm.sum()
    return np.fft.rfft(pm)


def test_dense_matches_legacy():
    atoms = np.arange(6.0)
    weights = np.full(6, 1 / 6)
    z = _rfft_z()
    got = evaluate_pgf_polynomial(atoms, weights, z)
    assert np.abs(got - _legacy(atoms, weights, z)).max() < 1e-13


def test_sparse_matches_legacy():
    atoms = np.array([1.0, 1000.0, 100000.0])
    weights = np.array([0.3, 0.3, 0.4])
    z = _rfft_z()
    got = evaluate_pgf_polynomial(atoms, weights, z)
    # |z| <= 1 so the huge powers stay bounded; legacy exp/log pow agrees
    # to close to machine precision
    assert np.abs(got - _legacy(atoms, weights, z)).max() < 1e-12


def test_unsorted_duplicates_match_legacy():
    atoms = np.array([5.0, 2.0, 5.0, 0.0])
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    z = _rfft_z()
    got = evaluate_pgf_polynomial(atoms, weights, z)
    assert np.abs(got - _legacy(atoms, weights, z)).max() < 1e-13


def test_real_z_dtype_and_values():
    atoms = np.array([0.0, 3.0])
    weights = np.array([0.5, 0.5])
    z = np.linspace(0.0, 1.0, 11)
    got = evaluate_pgf_polynomial(atoms, weights, z)
    assert got.dtype == np.float64
    assert np.abs(got - (0.5 + 0.5 * z ** 3)).max() < 1e-15


def test_scalar_z_shape_parity():
    atoms = np.array([1.0, 4.0])
    weights = np.array([0.5, 0.5])
    got = evaluate_pgf_polynomial(atoms, weights, 0.5 + 0.1j)
    legacy = _legacy(atoms, weights, 0.5 + 0.1j)
    assert got.shape == legacy.shape == (1,)
    assert np.abs(got - legacy).max() < 1e-15


def test_fractional_atoms_take_legacy_path():
    atoms = np.array([0.5, 1.5])
    weights = np.array([0.5, 0.5])
    z = _rfft_z()
    got = evaluate_pgf_polynomial(atoms, weights, z)
    # identical expression, so exact equality
    assert np.array_equal(got, _legacy(atoms, weights, z))


def test_build_dense_dfreq_density():
    # uniform count 0..5, unit severity: aggregate pmf == count pmf
    a = build('agg PGFDense dfreq [0:5] dsev [1]')
    assert np.abs(a.density_df.loc[0:5, 'p_total'].values - 1 / 6).max() < 1e-12


def test_build_sparse_dfreq_density():
    # sparse counts: aggregate mass sits exactly at 1 and 1000
    a = build('agg PGFSparse dfreq [1 1000] [.6 .4] dsev [1]', log2=11)
    assert a.density_df.loc[1, 'p_total'] == pytest.approx(0.6, abs=1e-12)
    assert a.density_df.loc[1000, 'p_total'] == pytest.approx(0.4, abs=1e-12)


def test_zm_empirical_unsupported():
    # empirical frequency has no ZM form; the wrapper machinery must refuse
    # (composition with the new pgf is exercised by the ZM-capable kinds)
    from aggregate import Frequency
    with pytest.raises(NotImplementedError):
        Frequency('empirical', np.array([0.0, 1.0]), np.array([0.5, 0.5]),
                  True, 0.25)
