"""Step-0 audit for dev/plan-numerics-3-distortion.md.

Measures the CURRENT distorted surface (``_build_augmented`` /
``price`` linear+lifted / ``Aggregate.apply_distortion``) against the
exact-discrete Choquet reference of ``choquet-calc-method.md``
(``gp = g(T) - g(S)``, ``rho(X∧a) = dot(min(x,a), gp)``), on legacy
zero-origin, windowed positive-origin, and signed books. Also probes the
consistency target across the six price surfaces, the cache-key
collision, and the mass-guard placement, and pre-captures the
``pricing_at`` / ``pentagon_at`` / ``price`` scalar readouts for the
baseline corpus (regression targets for deliverable 4).

Run::

    uv run python dev/audit-numerics-3.py

Findings are summarized in dev/audit-numerics-3-findings.md (written by
hand from this output, not auto-generated).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))

from aggregate import build, Distortion

TOL = 1e-9


# ---------------------------------------------------------------------------
# Exact-discrete Choquet reference (choquet-calc-method.md)
# ---------------------------------------------------------------------------

def gp_reference(x, p, g):
    """Exact distorted atom weights on a normalized sorted law.

    Returns (S, T, gS, gp) with S the strict tail, T = S + p inclusive,
    gp = g(T) - g(S) the distorted pmf.
    """
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    S = np.cumsum(p[::-1])[::-1] - p          # strict right tail
    S[-1] = 0.0
    T = S + p
    T[0] = 1.0
    # clean float fuzz before applying g (g may NaN outside [0,1])
    S = np.clip(S, 0.0, 1.0)
    T = np.clip(T, 0.0, 1.0)
    gS = np.asarray(g(S), dtype=float)
    gT = np.asarray(g(T), dtype=float)
    gp = gT - gS
    return S, T, gS, gp


def exag_total_reference(x, gp):
    """rho(X ∧ x_k) at every grid point: cumsum(x·gp) + x·tail(gp)."""
    cum_xgp = np.cumsum(x * gp)
    tail_gp = np.cumsum(gp[::-1])[::-1] - gp   # strict
    return cum_xgp + x * tail_gp


def exag_line_reference(x, gp, gS, kappa, alpha=None):
    """Per-line E_g[X_i(a)] at every grid point (positive-loss book).

    lifted: cumsum(kappa·gp) + x · tail(share·gp)  (= x·gS·beta)
    linear: cumsum(kappa·gp) + x · gS · alpha      (alpha = exi_xgta)
    Returns (lifted, linear, beta).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.where(np.abs(x) > 1e-12, kappa / x, 0.0)
    sgp = share * gp
    tail_g_share = np.cumsum(sgp[::-1])[::-1] - sgp
    cum_kgp = np.cumsum(kappa * gp)
    lifted = cum_kgp + x * tail_g_share
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = np.where(gS > 0, tail_g_share / gS, 0.0)
    linear = None
    if alpha is not None:
        linear = cum_kgp + x * gS * alpha
    return lifted, linear, beta


def cmp(label, got, ref, sel=None, scale=None):
    got = np.asarray(got, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if sel is not None:
        got, ref = got[sel], ref[sel]
    ok = np.isfinite(ref) & np.isfinite(got)
    if not ok.any():
        print(f'  {label:<22} (no comparable rows)')
        return
    if scale is None:
        scale = max(np.nanmax(np.abs(ref[ok])), 1e-30)
    err = np.nanmax(np.abs(got[ok] - ref[ok])) / scale
    print(f'  {label:<22} max rel err {err:9.2e}'
          + ('   ** WRONG' if err > TOL else '   ok'))


# ---------------------------------------------------------------------------
# Audit one portfolio's distorted surface against the reference
# ---------------------------------------------------------------------------

def audit_port(port, dist, label, view='ask'):
    print(f'\n=== {label} | {dist.name} '.ljust(70, '='))
    g = dist.g if view == 'ask' else dist.g_dual

    try:
        aug = port.apply_distortion(dist, view=view)
    except Exception as e:
        print(f'  apply_distortion RAISES: {type(e).__name__}: {e}')
        return None
    # phantom-row check (df.loc[0, ...] = 0 on a windowed index)
    dd = port.density_df
    if len(aug) and not aug.index.is_monotonic_increasing:
        print(f'  ** augmented index NOT monotonic (phantom loc[0] row?); '
              f'len density_df={len(dd)}, len aug={len(aug)}')
    extra = set(aug.index) - set(dd.index)
    if extra:
        print(f'  ** PHANTOM rows added to augmented index: {sorted(extra)[:5]}')

    x = aug.loss.to_numpy()
    p = dd.p_total.reindex(aug.index).to_numpy()
    S_ref, T_ref, gS_ref, gp_ref_v = gp_reference(x, p, g)
    sel = np.abs(p) > 1e-9

    cmp('gS', aug.gS, gS_ref)
    cmp('gp_total', aug.gp_total, gp_ref_v, scale=1.0)
    exag_tot_ref = exag_total_reference(x, gp_ref_v)
    cmp('exag_total', aug.exag_total, exag_tot_ref)

    if np.all(x >= 0):
        for nm in port.line_names:
            kappa = aug[f'exeqa_{nm}'].to_numpy()
            alpha = aug[f'exi_xgta_{nm}'].to_numpy()
            lifted, linear, beta = exag_line_reference(
                x, gp_ref_v, gS_ref, kappa, alpha)
            gS_pos = gS_ref > 1e-7
            cmp(f'exi_xgtag_{nm}', aug[f'exi_xgtag_{nm}'], beta,
                sel=gS_pos, scale=1.0)
            cmp(f'exag_{nm} (lifted)', aug[f'exag_{nm}'], lifted)
    return aug


# ---------------------------------------------------------------------------
# Six-surface consistency target
# ---------------------------------------------------------------------------

def consistency(port, agg, dist, p_level, label):
    """Six-surface total-premium agreement at a = q(p_level)."""
    print(f'\n--- consistency target: {label} | {dist.name} '.ljust(70, '-'))
    a = port.q(p_level)
    ser = port.density_df.p_total
    r_dx = dist.price(ser, a=a, kind='ask', method='dx').ask
    r_ds = dist.price(ser, a=a, kind='ask', method='ds').ask
    vals = {'Distortion.price dx': r_dx, 'Distortion.price ds': r_ds}
    try:
        aug = port.apply_distortion(dist)
        vals['Port exag_total(a)'] = float(aug.at[a, 'exag_total'])
    except Exception as e:
        vals['Port exag_total(a)'] = f'RAISES {type(e).__name__}'
    for alloc in ('lifted', 'linear'):
        try:
            pr = port.price(p_level, dist, allocation=alloc)
            vals[f'Port.price {alloc}'] = pr.price
        except Exception as e:
            vals[f'Port.price {alloc}'] = f'RAISES {type(e).__name__}'
    if agg is not None:
        try:
            agg.apply_distortion(dist)
            a_agg = agg.snap(a)
            vals['Agg exag(a)'] = float(agg.density_df.at[a_agg, 'exag'])
            vals['Agg.price P'] = float(
                agg.price(p_level, dist).iloc[0]['P'])
        except Exception as e:
            vals['Agg side'] = f'RAISES {type(e).__name__}'
    base = r_dx
    for k, v in vals.items():
        if isinstance(v, str):
            print(f'  {k:<22} {v}')
        else:
            rel = abs(v - base) / max(abs(base), 1e-30)
            print(f'  {k:<22} {v:18.10f}   rel-to-dx {rel:9.2e}'
                  + ('   ** OFF' if rel > TOL else ''))


# ---------------------------------------------------------------------------
# Cache key + mass guard probes
# ---------------------------------------------------------------------------

def probe_cache_and_mass(port_unbounded):
    print('\n--- cache-key & mass-guard probes '.ljust(70, '-'))
    d = Distortion('dual', 1.85)
    ask = port_unbounded.apply_distortion(d, view='ask')
    bid = port_unbounded.apply_distortion(d, view='bid')
    same = ask is bid
    print(f'  ask-then-bid returns the SAME cached frame: {same}'
          + ('   ** STALE (cache key is name only)' if same else ''))
    port_unbounded._augmented_dfs.clear()

    ccoc = Distortion('ccoc', r=0.10)
    try:
        aug = port_unbounded.apply_distortion(ccoc)
        top = aug.gp_total.iloc[-1]
        print(f'  apply_distortion(ccoc) on UNBOUNDED builds frame '
              f'(no guard); top-bucket gp = {top:.6f}'
              f'   ** G6: guard only in price(lifted)')
    except Exception as e:
        print(f'  apply_distortion(ccoc) raises: {e}')
    port_unbounded._augmented_dfs.clear()


# ---------------------------------------------------------------------------
# Signed book: legacy formulas measured inline (builder refuses)
# ---------------------------------------------------------------------------

def audit_signed(port, dist, label):
    print(f'\n=== {label} | {dist.name} (legacy formulas inline) '.ljust(70, '='))
    try:
        port.apply_distortion(dist)
        print('  apply_distortion unexpectedly SUCCEEDED on signed support')
    except NotImplementedError as e:
        print(f'  apply_distortion refuses (numerics-2 guard): ok')

    df = port.density_df
    x = df.loss.to_numpy()
    p = df.p_total.to_numpy()
    bs = port.bs
    S_fwd = 1 - np.cumsum(p)
    gS = np.asarray(dist.g(np.maximum(S_fwd, 0)), dtype=float)
    # legacy zero-origin idiom
    legacy = pd.Series(gS).shift(1, fill_value=0).cumsum().to_numpy() * bs
    S_ref, T_ref, gS_ref, gp = gp_reference(x, p, dist.g)
    ref = exag_total_reference(x, gp)
    cmp('exag_total legacy', legacy, ref)
    x0 = x[0]
    cmp('exag_total legacy+x0', legacy + x0, ref)
    rho_ref = float(np.dot(x, gp))
    rho_dx = dist.price(df.p_total, kind='ask', method='dx').ask
    print(f'  rho ref dot(x,gp) = {rho_ref:.8f}; Distortion.price dx = '
          f'{rho_dx:.8f}; rel diff {abs(rho_ref-rho_dx)/abs(rho_ref):.2e}')
    # signed total pricing via dot(kappa, gp) reconciliation
    tot = np.zeros_like(x)
    for nm in port.line_names:
        tot += df[f'exeqa_{nm}'].to_numpy() * gp
    print(f'  sum_i dot(kappa_i, gp) = {tot.sum():.8f} '
          f'(vs rho {rho_ref:.8f}; rel diff '
          f'{abs(tot.sum()-rho_ref)/abs(rho_ref):.2e})')


# ---------------------------------------------------------------------------
# Old linear-price convention audit
# ---------------------------------------------------------------------------

def audit_linear_convention(port, dist, p_level, label):
    """Old collapsed linear engine vs exact-discrete linear formula."""
    print(f'\n--- linear convention: {label} | {dist.name} '.ljust(70, '-'))
    a = port.q(p_level)
    pr = port.price(p_level, dist, allocation='linear')
    old = pr.df.droplevel(0)

    dd = port.density_df
    msk = dd.loss <= a
    x = dd.loss.to_numpy()[msk]
    p = dd.p_total.to_numpy()[msk]
    # collapse: atom at a carries all tail mass (incl. deficit)
    p = p.copy()
    p[-1] = 1.0 - p[:-1].sum()
    S_ref, T_ref, gS_ref, gp = gp_reference(x, p, dist.g)
    tot_ref = float(np.dot(x, gp))
    rel = abs(old.loc['total', 'P'] - tot_ref) / abs(tot_ref)
    print(f'  total P old {old.loc["total", "P"]:.10f} vs collapsed-law '
          f'rho {tot_ref:.10f}: rel {rel:9.2e}'
          + ('   ** convention gap' if rel > TOL else '   ok'))
    # per line: exact formula cum(kappa gp) + a gS alpha at the row below a
    for nm in port.line_names:
        kappa = dd[f'exeqa_{nm}'].to_numpy()[msk]
        alpha_full = dd[f'exi_xgta_{nm}'].to_numpy()
        kgp = kappa[:-1] * gp[:-1]
        # atom at a: a * E[X_i/X | X >= a] * gp_atom; use exi_xgta at a-bs
        ia = np.searchsorted(dd.loss.to_numpy(), a) - 1
        share_atom = alpha_full[ia]
        line_ref = kgp.sum() + a * share_atom * gp[-1]
        got = old.loc[nm, 'P']
        rel = abs(got - line_ref) / max(abs(line_ref), 1e-30)
        print(f'  {nm:<10} P old {got:.10f} vs exact {line_ref:.10f}: '
              f'rel {rel:9.2e}' + ('   ** gap' if rel > TOL else '   ok'))


# ---------------------------------------------------------------------------
# Pre-capture: pricing_at / pentagon_at / price scalars for the corpus
# ---------------------------------------------------------------------------

def precapture(out_path):
    import baseline.corpus as C

    def make_dist(spec):
        return Distortion(name=spec['kind'], **spec['kwargs'])

    out = {}
    for name, (program, grid) in C.PORT_CASES.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            port = build(program, update=False)
            port.update(**grid)
        case = {}
        for spec in C.DISTORTIONS:
            label = spec['label']
            dist = make_dist(spec)
            entry = {}
            methods = C.PORT_METHODS[(name, label)]
            # pricing_at / pentagon_at read the lifted-style augmented
            # frame; skip where the post-change builder will refuse
            # (mass distortion on unbounded support).
            if not (spec['mass'] and not port.bounded):
                pa = port.pricing_at(dist, p=C.PRICING_P)
                entry['pricing_at'] = {
                    ln: {c: float(pa.loc[ln, c]) for c in 'LMPQ'}
                    for ln in pa.index}
                entry['pentagon_at'] = {}
                for ln in list(port.line_names) + ['total']:
                    peg = port.pentagon_at(dist, p=C.PRICING_P, line=ln)
                    entry['pentagon_at'][ln] = {
                        'L': float(peg.L), 'M': float(peg.M),
                        'P': float(peg.P), 'Q': float(peg.Q)}
            entry['price'] = {}
            for method in methods:
                pr = port.price(C.PRICING_P, dist, allocation=method)
                dfm = pr.df.droplevel(0)
                entry['price'][method] = {
                    'price': float(pr.price),
                    'df': {ln: {c: float(dfm.loc[ln, c]) for c in 'LMPQ'}
                           for ln in dfm.index}}
            case[label] = entry
            port._augmented_dfs.clear()
        out[name] = case
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(f'\npre-capture written to {out_path}')


# ---------------------------------------------------------------------------

def main():
    dual = Distortion('dual', 1.85)
    tvar = Distortion('tvar', 0.65)
    ccoc = Distortion('ccoc', r=0.10)
    ident = Distortion('tvar', 0.0)   # g(s) = s

    # ---- legacy zero-origin bounded discrete (exact hand-checkable) ------
    bod = build('''port AuditBodoff
        agg wind  1 claim sev dhistogram xps [0,  99] [0.80, 0.20] fixed
        agg quake 1 claim sev dhistogram xps [0, 100] [0.95, 0.05] fixed''',
        update=False)
    bod.update(log2=8, bs=1, padding=1)
    for d in (dual, tvar, ccoc, ident):
        audit_port(bod, d, 'Bodoff: legacy zero-origin bounded')
        bod._augmented_dfs.clear()
    agg_dice = build('agg AuditDice dfreq [1:6] dsev [1:6]',
                     log2=8, bs=1, padding=1)
    one_dice = build('port AuditOneDice agg D dfreq [1:6] dsev [1:6]',
                     update=False)
    one_dice.update(log2=8, bs=1, padding=1)
    for d in (dual, tvar):
        consistency(one_dice, agg_dice, d, 0.9, 'one-line dice port')
        one_dice._augmented_dfs.clear()
    audit_linear_convention(bod, dual, 0.99, 'Bodoff')
    audit_linear_convention(bod, ccoc, 0.99, 'Bodoff')

    # ---- unbounded book (CNC): cache + mass guard + linear convention ----
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cnc = build(
            'port AuditCNC '
            'agg NonCat 25 claim sev gamma   80 cv 0.15 mixed gamma .2 '
            'agg Cat    5  claim 200 xs 0 sev lognorm 40 cv 1.50 mixed ig .2',
            update=False)
        cnc.update(log2=16, bs=1/4, padding=1)
    audit_port(cnc, dual, 'CNC: legacy zero-origin unbounded')
    cnc._augmented_dfs.clear()
    probe_cache_and_mass(cnc)
    audit_linear_convention(cnc, dual, 0.99, 'CNC')

    # ---- positive-origin window (synthetic) --------------------------------
    # Plan A clamps non-signed total origins to 0 (positive-origin output
    # windows arrive with numerics-4), so demonstrate the x0 exposure on a
    # synthetic shifted atom table: the legacy zero-origin idiom vs the
    # exact capped-dot reference.
    print('\n--- positive-origin window (synthetic shifted dice) '.ljust(70, '-'))
    bs = 1.0
    x = np.arange(101.0, 107.0)              # frame starts at x0 = 101
    p = np.full(6, 1 / 6)
    S_ref, T_ref, gS_ref, gp = gp_reference(x, p, dual.g)
    ref = exag_total_reference(x, gp)
    legacy = pd.Series(gS_ref).shift(1, fill_value=0).cumsum().to_numpy() * bs
    cmp('exag_total legacy', legacy, ref)
    cmp('exag_total legacy+x0', legacy + x[0], ref)

    # ---- signed book -------------------------------------------------------
    sgn = build('''port AuditSignedD
        agg A dfreq [1 2] dsev [-3 -1 2] [.5 .3 .2]
        agg B dfreq [1] dsev [-2 1 4]
        agg C dfreq [2] dsev [1 3]''', update=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sgn.update(log2=8, bs=1, padding=1)
    audit_signed(sgn, dual, 'Signed discrete book')

    # ---- Aggregate surface -------------------------------------------------
    print('\n--- Aggregate.apply_distortion surface '.ljust(70, '-'))
    agg_dice.apply_distortion(dual)
    x = agg_dice.density_df.loss.to_numpy()
    p = agg_dice.density_df.p_total.to_numpy()
    _, _, _, gp = gp_reference(x, p, dual.g)
    ref = exag_total_reference(x, gp)
    cmp('Agg exag (dice)', agg_dice.density_df.exag.to_numpy(), ref)

    # ---- pre-capture corpus scalars ---------------------------------------
    precapture(Path(__file__).resolve().parents[1]
               / 'tests' / 'data' / 'numerics3_precapture.json')


if __name__ == '__main__':
    main()
