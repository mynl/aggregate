"""Invariant tests for the numerics-2 objective spine.

Covers ``dev/plan-numerics-2-objective.md``: the shifted-support kappa
(``exeqa_*``) and the direct-sum objective columns on signed / windowed /
zero-spectrum books, the native-pmf stand-alone quantities, and the removal
of the ``p_{unit}`` columns and the EPD family.

The exact finite laws (small dfreq×dsev books) are hand-checkable; the
brute-force reference is a plain ``np.convolve`` carrying unit origins
explicitly. The legacy zero-origin regression lives in
``tests/test_baseline.py`` (key columns) and
``tests/test_baseline_spotchecks.py`` (derived columns).

DecL programs are mirrored in ``src/aggregate/agg/test_decl.agg`` (N2.*).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aggregate import build
from aggregate.utilities import ft

REL = 1e-12


# ---------------------------------------------------------------------------
# Brute-force reference: origin-carrying convolution of native unit pmfs
# ---------------------------------------------------------------------------

def _native(agg, tol=1e-14):
    p = agg.density_df.p_total.to_numpy()
    x = agg.density_df.loss.to_numpy()
    nz = np.flatnonzero(np.abs(p) > tol)
    sl = slice(nz[0], nz[-1] + 1)
    return x[sl], p[sl]


def _brute_kappa(port):
    """(x_tot, p_tot, {line: kappa}) by exact convolution with origins."""
    bs = port.bs
    units = {a.name: _native(a) for a in port.agg_list}
    names = list(units)
    p_tot, o_tot = None, 0.0
    for nm in names:
        x_u, p_u = units[nm]
        if p_tot is None:
            p_tot, o_tot = p_u, x_u[0]
        else:
            p_tot, o_tot = np.convolve(p_tot, p_u), o_tot + x_u[0]
    x_tot = o_tot + bs * np.arange(len(p_tot))
    kappas = {}
    for nm in names:
        x_u, p_u = units[nm]
        num, o_num = x_u * p_u, x_u[0]
        for nm2 in names:
            if nm2 == nm:
                continue
            x_2, p_2 = units[nm2]
            num, o_num = np.convolve(num, p_2), o_num + x_2[0]
        assert np.isclose(o_num, o_tot)
        with np.errstate(invalid='ignore', divide='ignore'):
            kappas[nm] = np.where(np.abs(p_tot) > 1e-12, num / p_tot, np.nan)
    return x_tot, p_tot, kappas


def _compare_kappa(port, rel=REL):
    """Assert each exeqa_{line} matches the brute reference on support."""
    x, p, kappas = _brute_kappa(port)
    df = port.density_df
    jb = np.round(x / port.bs).astype(np.int64)
    jp = np.round(df.loss.to_numpy() / port.bs).astype(np.int64)
    _, bi, pi = np.intersect1d(jb, jp, return_indices=True)
    mat = np.abs(p[bi]) > 1e-9
    assert mat.sum() > 4
    for nm in kappas:
        got = df[f'exeqa_{nm}'].to_numpy()[pi][mat]
        ref = kappas[nm][bi][mat]
        scale = np.max(np.abs(ref))
        assert np.nanmax(np.abs(got - ref)) <= rel * scale, f'exeqa_{nm}'


# ---------------------------------------------------------------------------
# Books
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def signed_port():
    """Signed discrete book: negative origins, total straddles 0."""
    p = build('''port N2.SignedD
        agg A dfreq [1 2] dsev [-3 -1 2] [.5 .3 .2]
        agg B dfreq [1] dsev [-2 1 4]
        agg C dfreq [2] dsev [1 3]''', update=False)
    p.update(log2=8, bs=1, padding=1)
    return p


@pytest.fixture(scope='module')
def zero_spectrum_port():
    """One line with exact zeros in its spectrum (symmetric severity) —
    exercises the prefix/suffix ``ft_nots`` branch."""
    p = build('''port N2.ZeroSpec
        agg S dfreq [1] dsev [-1 1]
        agg T dfreq [1] dsev [1 2]
        agg U dfreq [1 2] dsev [2 3]''', update=False)
    p.update(log2=6, bs=1, padding=1)
    return p


@pytest.fixture(scope='module')
def legacy_port():
    """Plain zero-origin book (positive grid, mass at 0)."""
    p = build('''port N2.Legacy
        agg A dfreq [0 1 2] dsev [1 2 3]
        agg B dfreq [1] dsev [2 4] [.6 .4]''', update=False)
    p.update(log2=6, bs=1, padding=1)
    return p


# ---------------------------------------------------------------------------
# Kappa: the shifted-method key invariant and brute-force agreement
# ---------------------------------------------------------------------------

def test_sum_kappa_equals_x_signed(signed_port):
    df = signed_port.density_df
    mat = df.p_total.to_numpy() > 1e-12
    tot = df.filter(regex='exeqa_[ABC]').sum(axis=1).to_numpy()[mat]
    loss = df.loss.to_numpy()[mat]
    assert loss.min() < 0  # genuinely straddles zero
    assert np.max(np.abs(tot - loss)) <= REL * np.max(np.abs(loss))


def test_kappa_brute_force_signed(signed_port):
    _compare_kappa(signed_port)


def test_kappa_brute_force_zero_spectrum(zero_spectrum_port):
    # confirm the symmetric line really has exact zero bins (the test's
    # whole point is to drive the prefix/suffix branch)
    agg = zero_spectrum_port['S']
    assert np.any(agg.ftagg_density == 0)
    _compare_kappa(zero_spectrum_port)
    df = zero_spectrum_port.density_df
    mat = df.p_total.to_numpy() > 1e-12
    tot = df.filter(regex='exeqa_[STU]').sum(axis=1).to_numpy()[mat]
    loss = df.loss.to_numpy()[mat]
    assert np.max(np.abs(tot - loss)) <= REL * max(np.max(np.abs(loss)), 1.0)


def test_kappa_brute_force_legacy(legacy_port):
    _compare_kappa(legacy_port)


def test_kappa_positive_origin_direct():
    """Shifted kappa with nonzero positive origins, via add_exa directly.

    ``update`` does not yet produce positive-origin total grids (that is
    the numerics-4 windowed combine); this drives ``add_exa`` on a
    hand-built windowed frame to pin the origin machinery now.
    """
    port = build('''port N2.PosOrigin
        agg A dfreq [1] dsev [10 11]
        agg B dfreq [1] dsev [20 22] [.6 .4]''', update=False)
    port.update(log2=6, bs=1, padding=1)

    bs, padding = port.bs, port.padding
    n_out = 32
    origin = 25.0  # window [25, 57) holds the support {30..33}
    loss = origin + np.arange(n_out) * bs

    xa, pa = np.array([10.0, 11.0]), np.array([0.5, 0.5])
    xb, pb = np.array([20.0, 22.0]), np.array([0.6, 0.4])
    m_buf = n_out << padding

    def state(xs, p):
        buf = np.zeros(m_buf)
        buf[np.round(xs / bs).astype(np.int64) % m_buf] = p
        return dict(xs=xs, p=p, ft_p=ft(buf, 0))

    # exact total on the window
    p_tot = np.zeros(n_out)
    kappa_a = np.zeros(n_out)
    for x1, q1 in zip(xa, pa):
        for x2, q2 in zip(xb, pb):
            k = int(x1 + x2 - origin)
            p_tot[k] += q1 * q2
            kappa_a[k] += x1 * q1 * q2
    with np.errstate(invalid='ignore'):
        kappa_a = np.where(p_tot > 0, kappa_a / p_tot, 0.0)

    df = pd.DataFrame(dict(loss=loss, p_total=p_tot), index=loss)
    port.add_exa(df, {'A': state(xa, pa), 'B': state(xb, pb)})

    mat = p_tot > 1e-12
    assert np.allclose(df['exeqa_A'].to_numpy()[mat], kappa_a[mat],
                       rtol=1e-12, atol=1e-12)
    tot = (df['exeqa_A'] + df['exeqa_B']).to_numpy()[mat]
    assert np.allclose(tot, loss[mat], rtol=1e-12)
    # exa_total carries the origin: E[X ∧ a] at/above max support == mean
    e = 10.5 + (20 * 0.6 + 22 * 0.4)
    assert np.isclose(df['exa_total'].iloc[-1], e, rtol=1e-14)
    # ... and equals a for a at the window base (all mass above)
    assert np.isclose(df['exa_total'].iloc[0], origin, rtol=1e-14)


# ---------------------------------------------------------------------------
# Direct-sum objective columns
# ---------------------------------------------------------------------------

def test_exa_total_signed_carries_origin(signed_port):
    """E[X ∧ a] by direct sum against an exact reference on a signed grid."""
    df = signed_port.density_df
    x = df.loss.to_numpy()
    p = df.p_total.to_numpy()
    ref = np.cumsum(x * p) + x * (1 - np.cumsum(p))
    assert np.allclose(df.exa_total.to_numpy(), ref, rtol=0, atol=1e-12)
    # left edge: all mass above a ⇒ E[X ∧ a] = a (negative!)
    assert np.isclose(df.exa_total.iloc[0], x[0], rtol=1e-14)
    assert x[0] < 0


def test_sum_exa_lines_equals_exa_total(legacy_port):
    df = legacy_port.density_df
    tot = (df.exa_A + df.exa_B).to_numpy()
    assert np.allclose(tot, df.exa_total.to_numpy(), rtol=1e-12, atol=1e-12)


def test_share_columns_blanked_on_signed(signed_port):
    """kappa/x is not a recovery share on a signed grid (steering 6)."""
    df = signed_port.density_df
    for col in ['exi_xeqa_A', 'exi_xgta_A', 'exi_xlea_A', 'exi_x_A', 'exa_A']:
        assert df[col].isna().all(), col
    # the conditional means are NOT blanked
    mat = df.p_total > 1e-9
    assert df.loc[mat, 'exeqa_A'].notna().all()
    assert df.loc[mat & (df.F > 1e-9), 'exlea_A'].notna().all()


def test_e_native_reconciles_kappa(signed_port):
    """e_i (native pmf) == Σ kappa_i · p_total within represented tol."""
    df = signed_port.density_df
    for nm in signed_port.line_names:
        e_native = df[f'e_{nm}'].iloc[0]
        e_kappa = float((df[f'exeqa_{nm}'] * df.p_total).sum())
        assert np.isclose(e_native, e_kappa, rtol=1e-12), nm


def test_lev_native_capped_sum(signed_port):
    """Stand-alone lev_i(a) == exact capped native sum below / inside /
    above the unit support, independent of total-window overlap."""
    df = signed_port.density_df
    for nm in signed_port.line_names:
        xs_n, p_n = _native(signed_port[nm], tol=0.0)
        for a in [df.loss.iloc[0], -2.0, 0.0, 3.0, df.loss.iloc[-1]]:
            ref = float(np.sum(np.minimum(xs_n, a) * p_n)
                        + a * (1 - p_n.sum()))
            got = df.at[a, f'lev_{nm}']
            assert np.isclose(got, ref, rtol=1e-12, atol=1e-12), (nm, a)


def test_unit_windows_differ_no_p_unit(signed_port):
    """Units live on their own windows; no p_{unit} on the total frame."""
    origins = {nm: float(signed_port[nm].x_min)
               for nm in signed_port.line_names}
    assert len(set(origins.values())) > 1  # genuinely different windows
    for nm in signed_port.line_names:
        ser = signed_port.unit_density(nm)
        assert np.isclose(ser.sum(), 1.0, rtol=0, atol=1e-10)
        assert float(ser.index[0]) == origins[nm]
        assert f'p_{nm}' not in signed_port.density_df.columns


def test_guards_replace_blanking(legacy_port):
    """Explicit F/S guards: NaN exactly where the denominator is dust."""
    df = legacy_port.density_df
    tolerable = 1e-12
    f_small = df.F.to_numpy() <= tolerable
    assert np.array_equal(df.exlea_total.isna().to_numpy(), f_small)
    s_small = df.S.to_numpy() <= tolerable
    assert np.array_equal(df.exgta_total.isna().to_numpy(), s_small)


# ---------------------------------------------------------------------------
# Signed continuous book vs Monte Carlo
# ---------------------------------------------------------------------------

def test_signed_exeqa_matches_monte_carlo():
    """exeqa on a signed lognorm-shift book matches MC E[X_i | X ≈ a],
    at several asset levels including a < 0."""
    port = build('''port N2.SignedLN
        agg A 8 claims ssev 20 * lognorm 0.4 - 25 poisson
        agg L 4 claims sev lognorm 30 cv 0.4 poisson''', update=False)
    port.update(log2=14, bs=1/8, padding=1)
    df = port.density_df

    rng = np.random.default_rng(20260610)
    n_sim = 2_000_000
    sigma_l = np.sqrt(np.log(1 + 0.4 ** 2))
    scale_l = 30 / np.exp(sigma_l ** 2 / 2)

    n_a = rng.poisson(8, n_sim)
    n_l = rng.poisson(4, n_sim)
    # aggregate by drawing per-sim sums via normal approximation is NOT ok;
    # draw exact: use repeat/segment sums.
    def agg_sum(counts, draw):
        tot = np.zeros(len(counts))
        idx = np.repeat(np.arange(len(counts)), counts)
        np.add.at(tot, idx, draw(int(counts.sum())))
        return tot

    xa = agg_sum(n_a, lambda k: 20 * rng.lognormal(0.0, 0.4, k) - 25)
    xl = agg_sum(n_l, lambda k: rng.lognormal(np.log(scale_l), sigma_l, k))
    x = xa + xl

    for a, half in [(-40.0, 5.0), (0.0, 5.0), (120.0, 10.0), (280.0, 20.0)]:
        sel = (x > a - half) & (x <= a + half)
        assert sel.sum() > 2_000, f'window at {a} too thin'
        mc = xa[sel].mean()
        rows = (df.loss > a - half) & (df.loss <= a + half)
        fft = float((df.loc[rows, 'exeqa_A'] * df.loc[rows, 'p_total']).sum()
                    / df.loc[rows, 'p_total'].sum())
        se = xa[sel].std() / np.sqrt(sel.sum())
        assert abs(mc - fft) < max(5 * se, 0.02 * (abs(fft) + 1)), \
            f'a={a}: mc {mc:.3f} vs fft {fft:.3f} (se {se:.3f})'


# ---------------------------------------------------------------------------
# Removals: EPD family, p_{unit}, Aggregate epd
# ---------------------------------------------------------------------------

def test_removals(legacy_port):
    df = legacy_port.density_df
    assert not df.filter(regex='^(p|epd_[01]|e1xi)_[AB]').columns.tolist()
    assert not hasattr(legacy_port, 'add_exa_details')
    agg = build('agg N2.Dice dfreq [1] dsev [1:6]')
    assert 'epd' not in agg.density_df.columns


def test_aggregate_direct_sums_signed():
    """Aggregate density_df lev/exlea/exgta carry the origin on a signed
    grid; hand-checkable two-atom law."""
    a = build('pnl N2.PnL 10 prem - dfreq [1] dsev [2 5] [.5 .5]',
              update=False)
    a.update(log2=6, bs=1/4)
    df = a.density_df
    # atoms at 10-5=5 and 10-2=8 ... payoff: prem - loss: 10-2=8, 10-5=5
    # E[X] = 6.5; E[X ∧ 6] = .5·5 + .5·6 = 5.5
    assert np.isclose(df.at[6.0, 'lev'], 5.5, rtol=1e-14)
    assert np.isclose(df.at[6.0, 'exlea'], 5.0, rtol=1e-14)
    assert np.isclose(df.at[6.0, 'exgta'], 8.0, rtol=1e-12)
    assert 'epd' not in df.columns


def test_aggregate_direct_sums_dice():
    """Hand checks on the dice law: E[X∧3]=2.5, E[X|X≤3]=2, E[X|X>3]=5."""
    a = build('agg N2.Dice dfreq [1] dsev [1:6]')
    df = a.density_df
    assert np.isclose(df.at[3.0, 'lev'], 2.5, rtol=1e-14)
    assert np.isclose(df.at[3.0, 'exlea'], 2.0, rtol=1e-14)
    assert np.isclose(df.at[3.0, 'exgta'], 5.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# Signed books refuse the (numerics-3) distortion surface
# ---------------------------------------------------------------------------

def test_signed_apply_distortion_prices_totals(signed_port):
    # numerics-3 removed the numerics-2 stopgap refusal: signed books now
    # price the total exactly (dot(x, gp) carrying the origin); the
    # per-line equal-priority distorted columns stay NaN (steering 6).
    # Full coverage in tests/test_numerics3_distortion.py.
    from aggregate.spectral import Distortion
    aug = signed_port.apply_distortion(Distortion('tvar', 0.8))
    assert np.isfinite(aug.exag_total.to_numpy()).all()
    assert aug[f'exag_{signed_port.line_names[0]}'].isna().all()
    signed_port._augmented_dfs.clear()


# ---------------------------------------------------------------------------
# Sampling cluster stays mechanically alive without p_{unit}
# ---------------------------------------------------------------------------

def test_sample_works(legacy_port):
    s = legacy_port.sample(64)
    assert list(s.columns) == ['A', 'B', 'total']
    assert len(s) == 64
    assert np.allclose(s.total, s.A + s.B)


def test_add_exa_sample_and_swap(legacy_port):
    s = legacy_port.sample(256)
    df = legacy_port.add_exa_sample(s)
    assert 'exeqa_A' in df.columns
    mat = df.p_total > 0
    assert np.allclose(
        df.loc[mat].filter(regex='exeqa_[AB]').sum(axis=1),
        df.loc[mat, 'loss'])
