"""Tests for the Distortion ``info`` / ``describe`` / ``stats_df`` /
``density_df`` quartet.

The quartet is lazy (cached on first access; invalidated when ``_build``
runs after a parameter change or calibration). Tests cover:

* shape of each property
* the mean-partition identity ``E[D_g] + E[D_g_inv] = 1``
* ``p_equiv`` self-consistency against the closed-form mapping
* numeric-vs-closed-form ``error`` column tolerance
* knot splicing into ``density_df``
* cache invalidation on parameter change
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aggregate import Distortion


# Representative parameter set for each kind. Multi-param kinds and
# combo kinds have separate fixtures below.
SCALAR_SPECS = [
    ('ph',   {'a': 0.7}),
    ('wang', {'lam': 0.3}),
    ('dual', {'b': 2.0}),
    ('tvar', {'p': 0.7}),
    ('ccoc', {'r': 0.1}),
    ('beta', {'a': 0.7, 'b': 1.5}),
    ('power', {'x0': 0.01, 'x1': 1.0, 'alpha': 2.0}),
    ('cll', {'r0': 0.05, 'b': 0.9}),
    ('clin', {'r0': 0.05, 'slope': 2.0}),
    ('lep', {'r0': 0.03, 'r': 0.15}),
    ('ly', {'r0': 0.05, 'r': 1.25}),
    ('bitvar', {'p0': 0.5, 'p1': 0.95, 'w1': 0.4}),
    ('wtdtvar', {'ps': [0.5, 0.9, 1.0], 'wts': [0.3, 0.3, 0.4]}),
]


# ---------------------------------------------------------------------------
# Shape / type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('kind,kw', SCALAR_SPECS, ids=[s[0] for s in SCALAR_SPECS])
def test_info_is_str(kind, kw):
    d = Distortion(kind, **kw)
    s = d.info
    assert isinstance(s, str)
    assert 'Distortion:' in s
    assert kind in s
    assert 'id' in s.lower()


@pytest.mark.parametrize('kind,kw', SCALAR_SPECS, ids=[s[0] for s in SCALAR_SPECS])
def test_describe_is_dataframe(kind, kw):
    d = Distortion(kind, **kw)
    df = d.describe
    assert isinstance(df, pd.DataFrame)
    for col in ('D_g', 'D_g_inv', 'closed_form', 'error'):
        assert col in df.columns
    for row in ('mean', 'std', 'cv', 'skew', 'E[D_g]+E[D_g_inv]'):
        assert row in df.index


@pytest.mark.parametrize('kind,kw', SCALAR_SPECS, ids=[s[0] for s in SCALAR_SPECS])
def test_stats_df_is_dataframe(kind, kw):
    d = Distortion(kind, **kw)
    df = d.stats_df
    assert isinstance(df, pd.DataFrame)
    for col in ('D_g', 'closed_form', 'error'):
        assert col in df.columns
    for row in ('mean', 'var', 'std', 'cv', 'skew',
                'gini', 'p_equiv', 'loading'):
        assert row in df.index


@pytest.mark.parametrize('kind,kw', SCALAR_SPECS, ids=[s[0] for s in SCALAR_SPECS])
def test_describe_has_kusuoka_summary(kind, kw):
    d = Distortion(kind, **kw)
    df = d.describe
    for row in ('mean_mass', 'max_mass', 'interior_atoms'):
        assert row in df.index


@pytest.mark.parametrize('kind,kw', SCALAR_SPECS, ids=[s[0] for s in SCALAR_SPECS])
def test_density_df_has_expected_columns(kind, kw):
    d = Distortion(kind, **kw)
    df = d.density_df
    assert isinstance(df, pd.DataFrame)
    for col in ('g', 'g_inv', 'g_dual', 'g_dual_inv',
                'g_prime', 'g_dual_prime', 'kusuoka'):
        assert col in df.columns
    assert df.index.name == 'x'
    assert df.index[0] == pytest.approx(0.0)
    assert df.index[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Mean partition identity: E[D_g] + E[D_g_inv] = 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('kind,kw', SCALAR_SPECS, ids=[s[0] for s in SCALAR_SPECS])
def test_mean_partition_identity(kind, kw):
    d = Distortion(kind, **kw)
    df = d.describe
    sum_means = df.loc['E[D_g]+E[D_g_inv]', 'D_g']
    # Smooth kinds reach 1e-5 at n=101; kinked kinds (tvar, bitvar, wtdtvar)
    # match the identity to machine precision because knots align with the
    # grid and the piecewise integral is exact.
    assert sum_means == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# p_equiv self-consistency: matches the closed-form mapping
# ---------------------------------------------------------------------------

def test_p_equiv_tvar_identity():
    d = Distortion('tvar', p=0.7)
    assert d.stats_df.loc['p_equiv', 'D_g'] == pytest.approx(0.7, abs=1e-10)


def test_p_equiv_ph_closed_form():
    """``p_equiv`` for PH(a) equals ``(1-a)/(1+a)``."""
    a = 0.7
    expected = (1 - a) / (1 + a)
    d = Distortion('ph', a=a)
    assert d.stats_df.loc['p_equiv', 'D_g'] == pytest.approx(
        expected, abs=2e-4)


def test_p_equiv_dual_closed_form():
    """``p_equiv`` for Dual(b) equals ``(b-1)/(b+1)``."""
    b = 2.0
    expected = (b - 1) / (b + 1)
    d = Distortion('dual', b=b)
    assert d.stats_df.loc['p_equiv', 'D_g'] == pytest.approx(
        expected, abs=2e-4)


def test_p_equiv_ccoc_closed_form():
    """``p_equiv`` for CCoC equals ``d``."""
    d_obj = Distortion('ccoc', r=0.1)
    expected_d = 0.1 / 1.1
    assert d_obj.stats_df.loc['p_equiv', 'D_g'] == pytest.approx(
        expected_d, abs=1e-3)


# ---------------------------------------------------------------------------
# Closed-form vs numeric error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('kind,kw', [
    ('ph',     {'a': 0.7}),
    ('dual',   {'b': 2.0}),
    ('tvar',   {'p': 0.7}),
    ('ccoc',   {'r': 0.1}),
    ('beta',   {'a': 0.7, 'b': 1.5}),
    ('bitvar', {'p0': 0.5, 'p1': 0.95, 'w1': 0.4}),
    ('wtdtvar', {'ps': [0.5, 0.9, 1.0], 'wts': [0.3, 0.3, 0.4]}),
], ids=['ph', 'dual', 'tvar', 'ccoc', 'beta', 'bitvar', 'wtdtvar'])
def test_closed_form_matches_numeric_mean(kind, kw):
    """For kinds with closed-form mean, numeric and closed-form columns
    agree within grid-resolution tolerance."""
    d = Distortion(kind, **kw)
    row = d.stats_df.loc['mean']
    assert not np.isnan(row['closed_form'])
    # Smooth kinds (PH/Dual/Beta) have ~1e-4 trapz error at n=101;
    # piecewise-linear kinds and kinds with masses align exactly with knots
    # so error is much smaller.
    assert abs(row['error']) < 1e-3


# ---------------------------------------------------------------------------
# Knot splicing
# ---------------------------------------------------------------------------

def test_tvar_kink_in_density_df():
    p = 0.7
    d = Distortion('tvar', p=p)
    x = d.density_df.index.to_numpy()
    assert np.any(np.isclose(x, 1 - p))


def test_bitvar_knots_in_density_df():
    p0, p1 = 0.5, 0.95
    d = Distortion('bitvar', p0=p0, p1=p1, w1=0.4)
    x = d.density_df.index.to_numpy()
    assert np.any(np.isclose(x, 1 - p0))
    assert np.any(np.isclose(x, 1 - p1))


def test_wtdtvar_knots_in_density_df():
    ps = [0.5, 0.9, 1.0]
    d = Distortion('wtdtvar', ps=ps, wts=[0.3, 0.3, 0.4])
    x = d.density_df.index.to_numpy()
    for p in ps:
        if 0 < p < 1:
            assert np.any(np.isclose(x, 1 - p))


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

def test_setter_invalidates_cache():
    """Changing the natural parameter re-runs ``_build`` which invalidates
    the cached quartet; the next access recomputes."""
    d = Distortion('ph', a=0.7)
    m1 = d.stats_df.loc['mean', 'D_g']
    d.a = 0.5
    m2 = d.stats_df.loc['mean', 'D_g']
    assert m1 != m2
    # closed-form for PH(0.5): a/(a+1) = 1/3 (trapz at n=101 ~2e-4)
    assert m2 == pytest.approx(1.0 / 3.0, abs=5e-4)


def test_cache_is_used():
    """Repeated access returns the same DataFrame object (cached)."""
    d = Distortion('ph', a=0.7)
    df1 = d.density_df
    df2 = d.density_df
    assert df1 is df2


# ---------------------------------------------------------------------------
# Kusuoka summary
# ---------------------------------------------------------------------------

def test_kusuoka_continuous_kinds_have_no_interior_atoms():
    """Wang, Dual, Beta have continuous mu; PH has only a boundary atom at p=0.
    None of them should report interior atoms in the describe summary."""
    for kind, kw in [('ph', {'a': 0.7}), ('wang', {'lam': 0.3}),
                     ('dual', {'b': 2.0}), ('beta', {'a': 0.7, 'b': 1.5})]:
        d = Distortion(kind, **kw)
        assert not d.describe.loc['interior_atoms', 'D_g']


def test_kusuoka_tvar_interior_atom():
    d = Distortion('tvar', p=0.7)
    assert d.describe.loc['interior_atoms', 'D_g']
    # And the atom shows up in stats_df with label mu_0.700, mass 1.0
    assert d.stats_df.loc['mu_0.700', 'D_g'] == pytest.approx(1.0)
    assert d.stats_df.loc['mu_0.700', 'closed_form'] == pytest.approx(0.7)


def test_kusuoka_ccoc_atoms():
    d = Distortion('ccoc', r=0.1)
    s = d.stats_df
    desc = d.describe
    # mu({0}) = 1-d (mean component); mu({1}) = d (max component)
    expected_d = 0.1 / 1.1
    assert desc.loc['mean_mass', 'D_g'] == pytest.approx(1 - expected_d)
    assert desc.loc['max_mass', 'D_g'] == pytest.approx(expected_d)
    assert not desc.loc['interior_atoms', 'D_g']
    # CCoC has atoms at p=0 and p=1 in stats_df
    assert s.loc['mu_0.000', 'D_g'] == pytest.approx(1 - expected_d)
    assert s.loc['mu_1.000', 'D_g'] == pytest.approx(expected_d)


def test_kusuoka_ph_atom_at_zero():
    """PH(a) has a single Dirac atom at p=0 of mass a (the mean-component
    weight). No atom at p=1."""
    a = 0.7
    d = Distortion('ph', a=a)
    s = d.stats_df
    desc = d.describe
    assert s.loc['mu_0.000', 'D_g'] == pytest.approx(a)
    assert 'mu_1.000' not in s.index
    assert desc.loc['mean_mass', 'D_g'] == pytest.approx(a)
    assert desc.loc['max_mass', 'D_g'] == pytest.approx(0.0)


def test_kusuoka_minimum_picks_up_transition_atom():
    """Min(CCoC, Dual) picks up:
      - inherited atom at p=0 (CCoC is active at s near 1)
      - a NEW interior atom at the active-transition point (Dual <-> CCoC)
    """
    from aggregate.spectral import MinimumDistortion
    d1 = Distortion('ccoc', d=0.2)
    d2 = Distortion('dual', b=2.5)
    mn = MinimumDistortion(distortions=[d1, d2])
    s = mn.stats_df
    # Boundary atom at p=0 inherited from CCoC (active near s=1)
    assert s.loc['mu_0.000', 'D_g'] == pytest.approx(0.8, abs=1e-6)
    # At least one interior atom from the active transition
    interior_atoms = [ix for ix in s.index
                      if ix.startswith('mu_') and ix not in ('mu_0.000', 'mu_1.000')]
    assert len(interior_atoms) >= 1
    assert mn.describe.loc['interior_atoms', 'D_g']


# ---------------------------------------------------------------------------
# Combo kinds (Minimum / Mixture)
# ---------------------------------------------------------------------------

def test_mixture_quartet():
    d1 = Distortion('ph', a=0.7)
    d2 = Distortion('wang', lam=0.3)
    mx = Distortion('mixture', distortions=[d1, d2], wts=[0.4, 0.6])
    assert isinstance(mx.info, str)
    # Mixture g_inv falls back to brentq numerically; density_df should still
    # have a valid g_inv column.
    assert np.isfinite(mx.density_df['g_inv'].iloc[50])
    # Mean partition identity
    sum_means = mx.describe.loc['E[D_g]+E[D_g_inv]', 'D_g']
    assert sum_means == pytest.approx(1.0, abs=1e-3)


def test_minimum_quartet():
    d1 = Distortion('ph', a=0.7)
    d2 = Distortion('wang', lam=0.3)
    mn = Distortion('minimum', distortions=[d1, d2])
    assert isinstance(mn.info, str)
    # mean is well-defined and partition identity holds
    sum_means = mn.describe.loc['E[D_g]+E[D_g_inv]', 'D_g']
    assert sum_means == pytest.approx(1.0, abs=1e-3)
