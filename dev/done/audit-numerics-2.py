"""Step-0 audit for dev/plan-numerics-2-objective.md.

Force-runs the CURRENT ``Portfolio.add_exa`` on signed / windowed books (the
paths ``update`` today refuses with a warning) and measures every objective
column against an exact brute-force convolution reference that carries unit
origins explicitly. Produces the column -> assumption -> verdict table the
plan requires *before* any edit.

Run::

    uv run python dev/audit-numerics-2.py

Findings are summarized in dev/audit-numerics-2-findings.md (written by hand
from this output, not auto-generated).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from aggregate import build

BS_TOL = 1e-12


# ---------------------------------------------------------------------------
# Exact brute-force reference: discrete convolution carrying origins
# ---------------------------------------------------------------------------

def native_pmf(agg, tol=0.0):
    """(origin, p) for a unit straight off its own density_df.

    No trim by default: trimming sub-1e-13 mass moved the *reference* by
    up to 4e-7 on tight units (measured), polluting the verdicts.
    """
    p = agg.density_df.p_total.to_numpy().copy()
    x = agg.density_df.loss.to_numpy()
    nz = np.flatnonzero(np.abs(p) > tol)
    lo, hi = nz[0], nz[-1] + 1
    return float(x[lo]), p[lo:hi]


def conv(p1, o1, p2, o2):
    """Convolution of two origin-carrying pmfs (same bs).

    ``fftconvolve`` on independently sized (never wrapping) supports —
    ~1e-13 relative accuracy, fast enough for untrimmed 64k vectors.
    """
    from scipy.signal import fftconvolve
    return fftconvolve(p1, p2), o1 + o2


def brute_force(port):
    """Exact total pmf and kappa_i on the brute grid, origins carried.

    Returns (x_tot, p_tot, {line: kappa_i}) where kappa_i is E[X_i | X=x]
    on x_tot (NaN where p_tot ~ 0).
    """
    bs = port.bs
    units = {a.name: native_pmf(a) for a in port.agg_list}
    names = list(units)
    # total
    p_tot, o_tot = units[names[0]][1], units[names[0]][0]
    for nm in names[1:]:
        p_tot, o_tot = conv(p_tot, o_tot, units[nm][1], units[nm][0])
    x_tot = o_tot + bs * np.arange(len(p_tot))
    kappas = {}
    for nm in names:
        o_i, p_i = units[nm]
        m_i = (o_i + bs * np.arange(len(p_i))) * p_i
        # product of the others
        p_not, o_not = None, 0.0
        for nm2 in names:
            if nm2 == nm:
                continue
            if p_not is None:
                o_not, p_not = units[nm2]
            else:
                p_not, o_not = conv(p_not, o_not, units[nm2][1], units[nm2][0])
        if p_not is None:
            num, o_num = m_i, o_i
        else:
            num, o_num = conv(m_i, o_i, p_not, o_not)
        assert abs(o_num - o_tot) < BS_TOL
        with np.errstate(invalid='ignore', divide='ignore'):
            kappas[nm] = np.where(np.abs(p_tot) > 1e-12, num / p_tot, np.nan)
    return x_tot, p_tot, kappas


def brute_objective(x, p, kappas, e_native):
    """Direct-sum objective reference columns on the brute grid."""
    F = np.cumsum(p)
    S = 1 - F
    cum_x = np.cumsum(x * p)
    out = {}
    out['exa_total'] = cum_x + x * S
    with np.errstate(invalid='ignore', divide='ignore'):
        out['exlea_total'] = cum_x / F
        out['exgta_total'] = (np.sum(x * p) - cum_x) / S
    pos = np.all(x >= -BS_TOL)
    for nm, k in kappas.items():
        kp = np.where(np.isnan(k), 0.0, k) * p
        cum_xi = np.cumsum(kp)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[f'exlea_{nm}'] = cum_xi / F
            out[f'exgta_{nm}'] = (e_native[nm] - cum_xi) / S
        if pos:
            with np.errstate(invalid='ignore', divide='ignore'):
                share = np.where(np.abs(x) > BS_TOL,
                                 np.where(np.isnan(k), 0.0, k) / x, 0.0)
            sp = share * p
            tail_share = np.cumsum(sp[::-1])[::-1] - sp
            with np.errstate(invalid='ignore', divide='ignore'):
                out[f'exi_xgta_{nm}'] = tail_share / S
            out[f'exa_{nm}'] = cum_xi + x * tail_share
    return out


def audit(port, label):
    print(f'\n=== {label} ({port.name}) '.ljust(70, '='))
    print(f'window: [{port.density_df.loss.iloc[0]:,.2f}, '
          f'{port.density_df.loss.iloc[-1]:,.2f}], bs={port.bs}')
    for a in port.agg_list:
        print(f'  unit {a.name}: x_min={a.x_min:,.2f}, '
              f'x_max={a.density_df.loss.iloc[-1]:,.2f}, '
              f'mass={a.density_df.p_total.sum():.6f}')

    x, p, kappas = brute_force(port)
    e_native = {a.name: float(np.dot(*[(a.density_df.loss.to_numpy()),
                                       (a.density_df.p_total.to_numpy())]))
                for a in port.agg_list}
    ref = brute_objective(x, p, kappas, e_native)

    df = port.density_df
    # align brute grid onto the port grid by bucket number
    jb = np.round(x / port.bs).astype(np.int64)
    jp = np.round(df.loss.to_numpy() / port.bs).astype(np.int64)
    common, bi, pi = np.intersect1d(jb, jp, return_indices=True)
    material = np.abs(p[bi]) > 1e-9
    F_b = np.cumsum(p)
    S_b = 1 - F_b
    # conditional means are intrinsically ill-conditioned where the
    # conditioning event has tiny probability; compare only where it is
    # resolvable on both sides
    F_ok = F_b[bi] > 1e-7
    S_ok = S_b[bi] > 1e-7

    def cmp(col, refv, extra=None):
        if col not in df.columns:
            print(f'  {col:<18} MISSING from frame')
            return
        sel = material if extra is None else material & extra
        a = df[col].to_numpy()[pi][sel]
        r = refv[bi][sel]
        ok = np.isfinite(r)
        scale = max(np.nanmax(np.abs(r[ok])), 1e-30)
        err = np.nanmax(np.abs(a[ok] - r[ok])) / scale
        print(f'  {col:<18} max rel err {err:9.2e}'
              + ('   ** WRONG' if err > 1e-9 else '   ok'))

    cmp('p_total', p)
    for nm in kappas:
        cmp(f'exeqa_{nm}', kappas[nm])
    cmp('exa_total', ref['exa_total'])
    cmp('lev_total', ref['exa_total'])
    cmp('exlea_total', ref['exlea_total'], F_ok)
    cmp('exgta_total', ref['exgta_total'], S_ok)
    for nm in kappas:
        cmp(f'exlea_{nm}', ref[f'exlea_{nm}'], F_ok)
        cmp(f'exgta_{nm}', ref[f'exgta_{nm}'], S_ok)
        if f'exa_{nm}' in ref:
            cmp(f'exa_{nm}', ref[f'exa_{nm}'])
            cmp(f'exi_xgta_{nm}', ref[f'exi_xgta_{nm}'], S_ok)
        # stand-alone lev against native capped sum at a few asset levels
        agg = port[nm]
        xs_n = agg.density_df.loss.to_numpy()
        ps_n = agg.density_df.p_total.to_numpy()
        for q in (0.5, 0.95):
            a_lvl = float(port.q(q))
            lev_ref = float(np.sum(np.minimum(xs_n, a_lvl) * ps_n)
                            + a_lvl * (1 - ps_n.sum()))
            if f'lev_{nm}' in df.columns and a_lvl in df.index:
                got = df.at[a_lvl, f'lev_{nm}']
                rel = abs(got - lev_ref) / max(abs(lev_ref), 1e-30)
                print(f'  lev_{nm}@q({q})'.ljust(20)
                      + f' rel err {rel:9.2e}'
                      + ('   ** WRONG' if rel > 1e-9 else '   ok'))
    # key invariant: sum_i kappa_i == x where p material (relative to the
    # loss scale of the grid)
    tot = np.zeros_like(df.loss.to_numpy(), dtype=float)
    for nm in kappas:
        if f'exeqa_{nm}' in df.columns:
            tot += df[f'exeqa_{nm}'].to_numpy()
    mat = np.abs(df.p_total.to_numpy()) > 1e-9
    scale = np.max(np.abs(df.loss.to_numpy()[mat]))
    inv = np.nanmax(np.abs(tot[mat] - df.loss.to_numpy()[mat])) / scale
    print(f'  sum kappa_i == x   max rel err {inv:9.2e}'
          + ('   ** VIOLATED' if inv > 1e-9 else '   ok'))


def main():
    # ---- Book A: signed discrete (exact brute force) ----------------------
    pa = build('''port AuditSignedD
        agg A dfreq [1 2] dsev [-3 -1 2] [.5 .3 .2]
        agg B dfreq [1] dsev [-2 1 4]
        agg C dfreq [2] dsev [1 3]''', update=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pa.update(log2=8, bs=1, padding=1)
    audit(pa, 'Book A: signed discrete, total straddles 0')

    # ---- Book B: signed lognorm-shift + ordinary line ---------------------
    pb = build('''port AuditSignedLN
        agg A 8 claims ssev 20 * lognorm 0.4 - 25 poisson
        agg B 6 claims ssev 15 * lognorm 0.3 - 18 poisson
        agg L 4 claims sev lognorm 30 cv 0.4 poisson''', update=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # log2=16 so the window holds the combined support (a 14-bucket
        # window wraps ~1e-8 of tail mass and contaminates the far tail —
        # combine sizing is numerics-4's problem, not allocation's).
        pb.update(log2=16, bs=1/4, padding=1)
    audit(pb, 'Book B: signed lognorm-shift mix')

    # ---- Book C: two tight positive units, windowed combine ---------------
    # Force the signed/windowed combine path so each unit picks its own
    # window (the future numerics-4 default for thin-CV books). High means
    # at a small log2 cap make the windowed (nonzero x_min) grid strictly
    # finer than the zero-origin grid, so it is actually selected.
    pc = build('''port AuditWindowed
        agg U1 1 claim sev lognorm 1000000 cv 0.05 fixed
        agg U2 1 claim sev lognorm 1500000 cv 0.04 fixed''', update=False)
    pc._signed = lambda: True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pc.update(log2=10, bs=0, padding=1)
    audit(pc, 'Book C: windowed positive units (nonzero origin)')

    # ---- Book D: disjoint unit windows ------------------------------------
    # U2's support sits far below U1's; the total window cannot contain
    # both unit supports, so any unit-pmf-on-total-grid view clips.
    pd_ = build('''port AuditDisjoint
        agg U1 1 claim sev lognorm 1000000 cv 0.03 fixed
        agg U2 1 claim sev lognorm 100 cv 0.1 fixed''', update=False)
    pd_._signed = lambda: True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pd_.update(log2=10, bs=0, padding=1)
    audit(pd_, 'Book D: disjoint unit windows')


if __name__ == '__main__':
    main()
