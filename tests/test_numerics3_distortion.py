"""Invariant tests for the numerics-3 distortion spine.

Covers ``dev/plan-numerics-3-distortion.md``: the exact-discrete Choquet
helper (``spectral.choquet_weights``), the ``view × value_type`` 2×2, the
unified linear/lifted builder at ``apply_distortion``, the on-demand
per-line capital, the six-surface consistency target, the builder-level
mass guard, signed total pricing, the widened cache key, and the
pre-change regression lock (``tests/data/numerics3_precapture.json``,
captured at 1.0.0a56 / e20b131).

The exact finite laws (small dfreq×dsev books) are hand-checkable. The
legacy zero-origin regression also lives in ``tests/test_baseline.py``
(recaptured post-change; the linear price engine convention deliberately
changed -- see CHANGELOG).

DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg`` (N3.*).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aggregate.config as cfg
from aggregate import build
from aggregate.constants import DefectiveDistributionError
from aggregate.spectral import Distortion, choquet_weights

REL = 1e-12
PRE = json.loads((Path(__file__).parent / 'data'
                  / 'numerics3_precapture.json').read_text())


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


# ---------------------------------------------------------------------------
# Books (module-scoped: read-only frames; cleared caches where needed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def dice_agg():
    return build('agg N3.Dice dfreq [1:6] dsev [1:6]',
                 log2=8, bs=1, padding=1)


@pytest.fixture(scope='module')
def dice_port():
    p = build('port N3.OneDice agg D dfreq [1:6] dsev [1:6]', update=False)
    p.update(log2=8, bs=1, padding=1)
    return p


@pytest.fixture(scope='module')
def bod():
    p = build('port N3.Bodoff '
              'agg wind  1 claim sev dhistogram xps [0,  99] [0.80, 0.20] fixed '
              'agg quake 1 claim sev dhistogram xps [0, 100] [0.95, 0.05] fixed',
              update=False)
    p.update(log2=8, bs=1, padding=1)
    return p


@pytest.fixture(scope='module')
def signed_port():
    p = build('port N3.SignedD '
              'agg A dfreq [1 2] dsev [-3 -1 2] [.5 .3 .2] '
              'agg B dfreq [1] dsev [-2 1 4] '
              'agg C dfreq [2] dsev [1 3]', update=False)
    p.update(log2=8, bs=1, padding=1)
    return p


DUAL = Distortion('dual', 1.85)
TVAR = Distortion('tvar', 0.65)
CCOC = Distortion('ccoc', r=0.10)
IDENT = Distortion('tvar', 0.0)          # g(s) = s


def _gp_reference(x, p, g):
    p = np.asarray(p, float) / np.sum(p)
    S = np.cumsum(p[::-1])[::-1] - p
    S[-1] = 0.0
    T = np.concatenate(([1.0], S[:-1]))
    return np.asarray(g(T), float) - np.asarray(g(S), float)


# ---------------------------------------------------------------------------
# choquet_weights -- the one exact-discrete engine
# ---------------------------------------------------------------------------

def test_choquet_weights_exact_dice():
    x = np.arange(1.0, 7)
    p = np.full(6, 1 / 6)
    w = choquet_weights(x, p, DUAL.g)
    assert np.isclose(w.gp.sum(), 1.0, rtol=0, atol=1e-15)
    assert w.T[0] == 1.0 and w.S[-1] == 0.0
    assert np.allclose(w.T, w.S + p, rtol=0, atol=1e-15)
    assert np.all(w.gp >= 0)
    rho = float(np.dot(x, w.gp))
    ser = pd.Series(p, index=x)
    assert np.isclose(rho, DUAL.price(ser, kind='ask', method='dx').ask,
                      rtol=REL)
    assert np.isclose(rho, DUAL.price(ser, kind='ask', method='ds').ask,
                      rtol=REL)
    # capped value at a = 4
    capped = float(np.dot(np.minimum(x, 4.0), w.gp))
    assert np.isclose(capped, DUAL.price(ser, a=4.0).ask, rtol=REL)
    # make_q is the same engine (backwards parking)
    q = DUAL.make_q(ser, a=4.0)
    assert np.isclose((q.q * q.index).sum() + 4.0 * q.gS.iloc[-1],
                      capped, rtol=REL)


def test_choquet_weights_signed_orientation_agnostic():
    x = np.array([-3.0, -1.0, 2.0])
    p = np.array([0.5, 0.3, 0.2])
    w = choquet_weights(x, p, DUAL.g)
    rho = float(np.dot(x, w.gp))
    assert np.isclose(rho, DUAL.price(pd.Series(p, index=x)).ask, rtol=REL)
    # layer form reconciles (asserted internally; check externally too)
    layer = x[0] + np.sum(np.diff(x) * w.gS[:-1])
    assert np.isclose(rho, layer, rtol=REL)


def test_choquet_deficit_policy_tiers():
    x = np.arange(1.0, 7)
    p = np.full(6, 1 / 6)
    # dust (<= validation noise): renormalized away
    w = choquet_weights(x, p * (1 - 1e-13), DUAL.g)
    assert np.isclose(w.gp.sum(), 1.0, rtol=0, atol=1e-12)
    # small truncation loss (noise < d <= materiality): parked silently
    w_f = choquet_weights(x, p * (1 - 1e-8), DUAL.g)
    w_b = choquet_weights(x, p * (1 - 1e-8), DUAL.g,
                          S_calculation='backwards')
    assert np.isclose(w_f.deficit, 1e-8, rtol=1e-3)
    # forwards parks at the top atom, backwards at the bottom atom
    assert w_f.gp[-1] > w_b.gp[-1]
    assert w_b.gp[0] > w_f.gp[0]
    # material deficit: raises unless the caller opts in
    with pytest.raises(DefectiveDistributionError):
        choquet_weights(x, p * 0.9, DUAL.g)
    w = choquet_weights(x, p * 0.9, DUAL.g, allow_deficit=True)
    assert np.isclose(w.gp.sum(), 1.0, rtol=0, atol=1e-12)
    # material surplus always raises
    with pytest.raises(DefectiveDistributionError):
        choquet_weights(x, p * 1.1, DUAL.g, allow_deficit=True)


def test_choquet_weights_validation():
    x = np.arange(1.0, 4)
    with pytest.raises(ValueError, match='sorted'):
        choquet_weights(x[::-1], np.full(3, 1 / 3), DUAL.g)
    with pytest.raises(ValueError, match='negative'):
        choquet_weights(x, np.array([0.6, -0.2, 0.6]), DUAL.g)


def test_forwards_backwards_agree_on_clean_law(dice_port):
    f = DUAL.price(dice_port.density_df.p_total, S_calculation='forwards').ask
    b = DUAL.price(dice_port.density_df.p_total, S_calculation='backwards').ask
    assert np.isclose(f, b, rtol=REL)


# ---------------------------------------------------------------------------
# view × value_type -- the 2×2 (D3)
# ---------------------------------------------------------------------------

def test_effective_g_two_by_two():
    for view, role, dual in [('ask', True, False), ('ask', False, True),
                             ('bid', True, True), ('bid', False, False)]:
        g, g_prime, is_dual = DUAL.effective_g(view, is_loss_value=role)
        assert is_dual == dual
        s = 0.3
        expected = DUAL.g_dual(s) if dual else DUAL.g(s)
        assert np.isclose(g(s), expected, rtol=0, atol=1e-15)
    with pytest.raises(ValueError):
        DUAL.effective_g('mid')


def test_payoff_role_prices_as_dual_round_trip():
    """Ask-of-payoff == bid-of-loss for the same law (XOR on the role flag)."""
    a_loss = build('agg N3.RoleL dfreq [1:6] dsev [1:6]',
                   log2=8, bs=1, padding=1)
    a_pay = build('agg N3.RoleP dfreq [1:6] dsev [1:6]',
                  log2=8, bs=1, padding=1)
    a_pay.value_type = 'payoff'
    a_loss.apply_distortion(DUAL, view='bid')
    bid_of_loss = a_loss.density_df.exag.copy()
    a_pay.apply_distortion(DUAL, view='ask')
    ask_of_payoff = a_pay.density_df.exag
    assert np.allclose(ask_of_payoff, bid_of_loss, rtol=REL, atol=1e-14)
    # and bid-of-payoff == ask-of-loss
    a_loss.apply_distortion(DUAL, view='ask')
    a_pay.apply_distortion(DUAL, view='bid')
    assert np.allclose(a_pay.density_df.exag, a_loss.density_df.exag,
                       rtol=REL, atol=1e-14)


def test_payoff_round_trip_survives_relabel():
    """The routing keys on the role flag, never the label string."""
    old = cfg._settings
    cfg._settings = cfg.load_settings(
        path=None, env={'AGGREGATE_VALUE_TYPE_PAYOFF': 'asset'})
    try:
        a_pay = build('agg N3.RoleR dfreq [1:6] dsev [1:6]',
                      log2=8, bs=1, padding=1)
        a_pay.value_type = 'asset'        # the relabeled payoff token
        assert a_pay._is_loss_value is False
        a_loss = build('agg N3.RoleL2 dfreq [1:6] dsev [1:6]',
                       log2=8, bs=1, padding=1)
        a_loss.apply_distortion(DUAL, view='bid')
        a_pay.apply_distortion(DUAL, view='ask')
        assert np.allclose(a_pay.density_df.exag, a_loss.density_df.exag,
                           rtol=REL, atol=1e-14)
    finally:
        cfg._settings = old


def test_homogeneous_payoff_book_prices_through_dual():
    p = build('port N3.PayPair '
              'pnl P1 10 prem - dfreq [1] dsev [2 5] [.5 .5] '
              'pnl P2 8 prem - dfreq [1] dsev [1 3]', update=False)
    p.update(log2=8, bs=1, padding=1)
    assert p._is_loss_value is False
    aug = p.apply_distortion(DUAL)
    # the frame's gS is the dual applied to the (recomputed) S
    assert np.allclose(aug.gS, DUAL.g_dual(aug.S), rtol=REL, atol=1e-14)


def test_mixed_value_type_book_still_raises():
    """Delivered by hygiene-4 Item 1; assert it still holds (D8)."""
    with pytest.raises(ValueError, match='mixed value_type'):
        build('port N3.Mixed '
              'agg L1 dfreq [1] dsev [1 2] '
              'pnl P1 10 prem - dfreq [1] dsev [2 5] [.5 .5]', update=False)


# ---------------------------------------------------------------------------
# Unified linear/lifted builder
# ---------------------------------------------------------------------------

def test_identity_distortion_invariants(bod):
    for allocation in ('lifted', 'linear'):
        aug = bod.apply_distortion(IDENT, allocation=allocation)
        sel = aug.gS > 0
        for line in bod.line_names:
            # beta == alpha under the identity distortion
            assert np.allclose(aug.loc[sel, f'exi_xgtag_{line}'],
                               aug.loc[sel, f'exi_xgta_{line}'],
                               rtol=1e-10, atol=1e-12)
            # premium == loss
            assert np.allclose(aug[f'exag_{line}'], aug[f'exa_{line}'],
                               rtol=1e-10, atol=1e-10)
        pr = bod.price(0.99, IDENT, allocation=allocation)
        assert pr.df['M'].abs().max() < 1e-10
        bod._augmented_dfs.clear()


def test_one_line_port_line_equals_total(dice_port):
    for dist in (IDENT, DUAL):
        aug = dice_port.apply_distortion(dist)
        assert np.allclose(aug['exag_D'], aug['exag_total'],
                           rtol=1e-10, atol=1e-12)
        pa = dice_port.pricing_at(dist, p=0.9)
        # under the identity distortion there is no margin, so the
        # margin-proportional layer construction allocates zero line
        # capital (documented guard); compare Q only when margins exist
        stats = 'LMP' if dist is IDENT else 'LMPQ'
        for c in stats:
            assert np.isclose(pa.loc['D', c], pa.loc['total', c],
                              rtol=1e-9, atol=1e-9)
        dice_port._augmented_dfs.clear()


def test_linear_vs_lifted(bod):
    lifted = bod.apply_distortion(DUAL, allocation='lifted')
    linear = bod.apply_distortion(DUAL, allocation='linear')
    # same schema
    assert list(lifted.columns) == list(linear.columns)
    # sum of lines == total, for both (rho(X ∧ a) at every row)
    for aug in (lifted, linear):
        parts = aug['exag_wind'] + aug['exag_quake']
        assert np.allclose(parts, aug['exag_total'], rtol=1e-10, atol=1e-10)
    # agree at/above the max support (no tail left to split)
    top = aug.index[-1]
    assert np.isclose(lifted.at[top, 'exag_wind'],
                      linear.at[top, 'exag_wind'], rtol=1e-10)
    # differ predictably when a cuts a multi-state tail (a=99: states
    # 100 and 199 remain; the distortion reweights them for lifted only)
    assert not np.isclose(lifted.at[99.0, 'exag_wind'],
                          linear.at[99.0, 'exag_wind'], rtol=1e-6)
    bod._augmented_dfs.clear()


def test_six_surface_consistency(dice_agg, dice_port):
    for dist in (DUAL, TVAR):
        a = dice_port.q(0.9)
        ser = dice_port.density_df.p_total
        vals = [dist.price(ser, a=a, kind='ask', method='dx').ask,
                dist.price(ser, a=a, kind='ask', method='ds').ask,
                float(dice_port.apply_distortion(dist).at[a, 'exag_total']),
                dice_port.price(0.9, dist, allocation='lifted').price,
                dice_port.price(0.9, dist, allocation='linear').price]
        dice_agg.apply_distortion(dist)
        vals.append(float(dice_agg.density_df.at[dice_agg.snap(a), 'exag']))
        vals.append(float(dice_agg.price(0.9, dist).iloc[0]['P']))
        assert max(abs(v - vals[0]) for v in vals) <= REL * abs(vals[0])
        dice_port._augmented_dfs.clear()


def test_mass_guard_in_builder():
    """G6: the lifted builder refuses, not just price()."""
    p = build('port N3.Unb agg U 5 claims sev lognorm 10 cv 1 poisson',
              update=False)
    p.update(log2=13, bs=1/8, padding=1)
    assert not p.bounded
    with pytest.raises(ValueError, match='mass'):
        p.apply_distortion(CCOC)               # lifted default
    # linear stays available (the collapsed law is bounded by
    # construction); beta columns are blanked
    aug = p.apply_distortion(CCOC, allocation='linear')
    assert aug['exi_xgtag_U'].isna().all()
    assert np.isfinite(aug['exag_U'].to_numpy()).all()
    pr = p.price(0.99, CCOC, allocation='linear')
    assert np.isfinite(pr.price)
    # analyze_distortions sweeps skip the unpriceable member with a warning
    with pytest.warns(UserWarning, match='mass'):
        res = p.analyze_distortions(p=0.99,
                                    distortions={'ccoc': CCOC, 'dual': DUAL})
    got = set(res.pricing_df.index.get_level_values(0).unique().dropna())
    assert 'dual' in got and 'ccoc' not in got
    # Aggregate-side guard
    a = build('agg N3.UnbA 5 claims sev lognorm 10 cv 1 poisson',
              log2=12, bs=1/16, padding=1)
    with pytest.raises(ValueError, match='mass'):
        a.apply_distortion(CCOC)
    # bounded support: mass distortion is exact (jump at the top atom)
    d = build('agg N3.DiceB dfreq [1:6] dsev [1:6]', log2=8, bs=1, padding=1)
    assert d.bounded
    d.apply_distortion(CCOC)
    gp = d.density_df.gp_total.to_numpy()
    assert np.isclose(gp.sum(), 1.0, rtol=0, atol=1e-12)


def test_signed_total_pricing(signed_port):
    """Steering 6: signed totals price via dot(kappa, gp); shares rejected."""
    aug = signed_port.apply_distortion(DUAL)
    x = aug.loss.to_numpy()
    gp = aug.gp_total.to_numpy()
    rho = float(aug.exag_total.iloc[-1])
    assert np.isclose(rho, np.dot(x, gp), rtol=REL)
    assert np.isclose(
        rho, DUAL.price(signed_port.density_df.p_total).ask, rtol=REL)
    # sum_i dot(kappa_i, gp) prices the signed total additively
    tot = sum(float(np.dot(aug[f'exeqa_{ln}'].to_numpy(), gp))
              for ln in signed_port.line_names)
    assert np.isclose(tot, rho, rtol=1e-10)
    # equal-priority share columns are refused (NaN), not divided
    for ln in signed_port.line_names:
        assert aug[f'exi_xgtag_{ln}'].isna().all()
        assert aug[f'exag_{ln}'].isna().all()
    signed_port._augmented_dfs.clear()


def test_cache_keys_coexist(dice_port):
    dice_port._augmented_dfs.clear()
    a1 = dice_port.apply_distortion(DUAL)
    a2 = dice_port.apply_distortion(DUAL, view='bid')
    a3 = dice_port.apply_distortion(DUAL, allocation='linear')
    a4 = dice_port.apply_distortion(DUAL, S_calculation='backwards')
    assert len(dice_port.augmented_dfs) == 4
    assert a1 is dice_port.apply_distortion(DUAL)      # cache hit
    assert a1 is not a2 and a1 is not a3 and a1 is not a4
    # bid frame really is the dual
    assert np.allclose(a2.gS, DUAL.g_dual(a2.S), rtol=REL, atol=1e-14)
    dice_port._augmented_dfs.clear()


def test_efficient_removed(dice_port):
    with pytest.raises(TypeError):
        dice_port.apply_distortion(DUAL, efficient=True)
    with pytest.raises(TypeError):
        dice_port.price(0.9, DUAL, efficient=True)


# ---------------------------------------------------------------------------
# On-demand per-line capital (deliverable 4)
# ---------------------------------------------------------------------------

def test_line_capital_reconciles(bod):
    for dist in (DUAL, TVAR, CCOC):
        pa = bod.pricing_at(dist, p=0.99)
        lines = list(bod.line_names)
        assert np.isclose(pa.loc[lines, 'Q'].sum(), pa.loc['total', 'Q'],
                          rtol=1e-9, atol=1e-9)
        bod._augmented_dfs.clear()


def test_pricing_at_matches_price_and_pentagon(bod):
    pa = bod.pricing_at(DUAL, p=0.99)
    pr = bod.price(0.99, DUAL, allocation='lifted')
    dfm = pr.df.droplevel(0)
    for ln in list(bod.line_names) + ['total']:
        for c in 'LMPQ':
            assert np.isclose(pa.loc[ln, c], dfm.loc[ln, c],
                              rtol=1e-12, atol=1e-12)
        peg = bod.pentagon_at(DUAL, p=0.99, line=ln)
        for c in 'LMPQ':
            assert np.isclose(getattr(peg, c), pa.loc[ln, c],
                              rtol=1e-12, atol=1e-12)
    bod._augmented_dfs.clear()


# ---------------------------------------------------------------------------
# allocation_diagnostics (deliverable 6)
# ---------------------------------------------------------------------------

def test_allocation_diagnostics_layer_identities(bod):
    diag = bod.allocation_diagnostics(DUAL, surface='lifted')
    S = diag.S.to_numpy()
    gS = diag.gS.to_numpy()
    assert np.allclose(diag.layer_loss_total, S, rtol=0, atol=1e-15)
    assert np.allclose(diag.layer_premium_total, gS, rtol=0, atol=1e-15)
    assert np.allclose(diag.layer_margin_total, gS - S, rtol=0, atol=1e-15)
    for ln in bod.line_names:
        assert np.allclose(diag[f'layer_loss_{ln}'],
                           S * diag[f'exi_xgta_{ln}'], rtol=0, atol=1e-15)
        assert np.allclose(diag[f'layer_premium_{ln}'],
                           gS * diag[f'exi_xgtag_{ln}'], rtol=0, atol=1e-15)
        assert np.allclose(
            diag[f'cum_margin_{ln}'],
            diag[f'exag_{ln}'] - diag[f'exa_{ln}'], rtol=0, atol=1e-15)
    # exact total capital identity
    assert np.allclose(diag.cum_capital_total,
                       diag.loss - diag['exag_total'], rtol=0, atol=1e-12)
    bod._augmented_dfs.clear()


# ---------------------------------------------------------------------------
# Pre-change regression lock (Step-0 capture, 1.0.0a56)
# ---------------------------------------------------------------------------

def _build_corpus_port(name):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import baseline.corpus as C
    program, grid = C.PORT_CASES[name]
    port = build(program, update=False)
    port.update(**grid)
    return port


@pytest.mark.parametrize('case', list(PRE))
def test_precapture_lifted_surfaces_survive(case):
    """The lifted pricing surfaces reproduce pre-change values.

    Gates: exact discrete books to 1e-13; 64k-row FFT books to 1e-11
    (direct sums replaced ``cumsum·bs`` -- fp order-of-ops drift). The
    *linear* per-line cells deliberately changed (the collapsed-atom
    engine was replaced by the unified strict-tail formula; totals are
    preserved) and are locked by the recaptured ``test_baseline``
    corpus instead.
    """
    port = _build_corpus_port(case)
    rel = 1e-13 if case == 'Port.Bodoff' else 1e-11
    for label, entry in PRE[case].items():
        dist = {'ccoc': CCOC, 'dual': DUAL, 'tvar': TVAR}[label]
        if 'pricing_at' in entry:
            pa = port.pricing_at(dist, p=0.99)
            for ln, vals in entry['pricing_at'].items():
                for c, v in vals.items():
                    assert np.isclose(pa.loc[ln, c], v, rtol=rel,
                                      atol=rel * port.q(0.99)), \
                        f'{case} {label} pricing_at {ln}.{c}'
            for ln, vals in entry['pentagon_at'].items():
                peg = port.pentagon_at(dist, p=0.99, line=ln)
                for c, v in vals.items():
                    assert np.isclose(getattr(peg, c), v, rtol=rel,
                                      atol=rel * port.q(0.99)), \
                        f'{case} {label} pentagon_at {ln}.{c}'
        for method, cap in entry['price'].items():
            if method == 'lifted' and dist.has_mass and not port.bounded:
                continue
            pr = port.price(0.99, dist, allocation=method)
            assert np.isclose(pr.price, cap['price'], rtol=1e-11), \
                f'{case} {label} {method} total price'
            if method == 'lifted':
                dfm = pr.df.droplevel(0)
                for ln, vals in cap['df'].items():
                    for c, v in vals.items():
                        assert np.isclose(dfm.loc[ln, c], v, rtol=rel,
                                          atol=rel * port.q(0.99)), \
                            f'{case} {label} {method} {ln}.{c}'
        port._augmented_dfs.clear()
