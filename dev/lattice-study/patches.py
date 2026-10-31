"""Prototype fixes as monkeypatches, for measurement only. Nothing here touches the repository."""
import numpy as np

import aggregate._aggregate as A
import aggregate._bucket_window as B
from aggregate.utilities import _dyadic_lattice_step

P = 1.0 - 1e-12


def _layered_atoms(s):
    """Atoms and probs of severity s after its layer, conditional on exceeding the attachment."""
    a = np.asarray(s.support_atoms, float)
    pk = np.asarray(getattr(s.fz, 'pk', np.ones_like(a) / len(a)), float)
    att = float(getattr(s, 'attachment', 0.0) or 0.0)
    lim = float(getattr(s, 'limit', np.inf))
    if att == 0 and not lim < a.max():
        return a, pk
    keep = a > att
    a, pk = a[keep], pk[keep]
    return np.minimum(a - att, lim), pk / pk.sum()


def view_atoms(agg):
    """Atoms of every view of every component: layered gross, ceded and net of each occurrence
    layer, and for an aggregate cover its attachment and limit reduced by the largest gross
    atom (same lattice, no wider window). ``None`` when a component is not discrete."""
    if agg.sevs is None:
        return None
    gross, pool = [], []
    for s in agg.sevs:
        if getattr(s, 'support_atoms', None) is None:
            return None
        g, _ = _layered_atoms(s)
        gross.append(g)
    gross = np.concatenate(gross) if gross else np.array([])
    pool.append(gross)
    occ = getattr(agg, 'occ_reins', None)
    if occ:
        ceded = np.zeros_like(gross)
        for share, y, a in occ:
            c = share * np.minimum(np.maximum(gross - a, 0.0), y)
            ceded = ceded + c
            pool.append(c)
        pool.append(gross - ceded)
    top = float(np.max(np.abs(gross))) if len(gross) else 0.0
    ag = getattr(agg, 'agg_reins', None)
    if ag and top > 0:
        for share, y, a in ag:
            # the ceded aggregate is share * (points of the aggregate lattice, a, a + y)
            pool.append(share * gross)
            for t in (share * a, share * y if np.isfinite(y) else 0.0):
                r = float(np.remainder(t, top))
                if r > 0:
                    pool.append(np.array([r]))
    return np.concatenate(pool)


def final_view(agg):
    """Per-claim atoms and probs of the view ``density_df.p_total`` sums (gross, or occurrence
    net or ceded), plus the width of the hull of the gross support and 0, which any
    severity grid must span (physical 0 sits at index i0, not necessarily 0)."""
    atoms, probs = [], []
    for s in agg.sevs:
        g, pk = _layered_atoms(s)
        atoms.append(g)
        probs.append(pk / len(agg.sevs))
    g = np.concatenate(atoms)
    pk = np.concatenate(probs)
    reach = float(max(g.max(), 0.0) - min(g.min(), 0.0))
    occ = getattr(agg, 'occ_reins', None)
    if occ:
        ceded = np.zeros_like(g)
        for share, y, a in occ:
            ceded = ceded + share * np.minimum(np.maximum(g - a, 0.0), y)
        kind = str(getattr(agg, 'occ_kind', ''))
        g = ceded if kind.startswith('ceded') else g - ceded
    return g, pk, reach


_real_atoms = A.Aggregate._severity_atoms
_real_exact = A.Aggregate._exact_discrete_window


def patch_views():
    """[views] the lattice sees every view, and the exact support is the summed view's."""
    A.Aggregate._severity_atoms = view_atoms

    def exact(self):
        ed = _real_exact(self)
        if ed is None:
            return None
        lo, hi, h, lp_lo, lp_hi = ed
        pool = view_atoms(self)
        fit = _dyadic_lattice_step(pool)
        if fit is None:
            return None
        g, pk, reach = final_view(self)
        freq = self.frequency
        if freq.freq_name in ('empirical', 'renewal'):
            n_atoms = np.asarray(freq.freq_a, float)
        else:
            n_atoms = np.array([float(self.n)])
        s_min, s_max = float(g.min()), float(g.max())
        n_min, n_max = float(n_atoms.min()), float(n_atoms.max())
        hi2 = max(n_max * s_max, n_min * s_max)
        lo2 = min(n_max * s_min, n_min * s_min)
        if np.any(n_atoms == 0):
            hi2, lo2 = max(hi2, 0.0), min(lo2, 0.0)
        # the severity grid must span the hull of the gross support and 0, whatever the origin
        hi2 = max(hi2, reach)
        # keep the original corner-reachability reading; the support is the same shape
        return float(lo2), float(hi2), float(fit[0]), lp_lo, lp_hi
    A.Aggregate._exact_discrete_window = exact


def _unit_reach(u):
    """Width of the hull of a unit's gross severity support and 0, which its grid must span."""
    if u.sevs and all(getattr(s, 'support_atoms', None) is not None for s in u.sevs):
        g = np.concatenate([_layered_atoms(s)[0] for s in u.sevs])
        return float(max(g.max(), 0.0) - min(g.min(), 0.0))
    try:
        return float(u._severity_high_estimate(P))
    except Exception:
        return np.inf


_real_ab = A.Aggregate._bs_window
_real_port = B.port_best_window


def patch_window_holds_severity():
    """[window-holds-severity] a windowed grid grows log2, up to the cap, until N*bs reaches the
    largest severity outcome; the Aggregate stops rejecting the window for that reason, and
    the Portfolio gains the same guard it never had."""
    def ab(self, log2, bs_in, x_min_in, bucket_sizing_p, window_convention=None):
        bs, l2, x0 = _real_ab(self, log2, bs_in, x_min_in, bucket_sizing_p, window_convention)
        if bs_in or x_min_in is not None:
            return bs, l2, x0
        df = getattr(self, '_bs_window_df', None)
        if df is None or 'windowed' not in df.index or bool(df.loc['windowed', 'applies']):
            return bs, l2, x0
        w = df.loc['windowed']
        wbs = float(w['bs'])
        if not wbs < float(bs):
            return bs, l2, x0
        reach = _unit_reach(self)
        if not np.isfinite(reach):
            return bs, l2, x0
        need = max(B._need_log2(reach, wbs), int(w['log2']))
        if need > log2:
            return bs, l2, x0
        lo, hi = float(w['x_min']), float(w['x_max'])
        slack = max((1 << need) * wbs - (hi - lo), 0.0)
        x = float(np.floor(max(lo - slack / 2, 0.0) / wbs) * wbs)
        df.loc['windowed', ['applies', 'log2', 'x_min']] = [True, need, x]
        df.loc['used', ['bs', 'log2', 'x_min']] = [wbs, need, x]
        return wbs, need, x
    A.Aggregate._bs_window = ab

    def port(port_, log2=16, bs_in=0, bucket_sizing_p=B.BUCKET_SIZING_P):
        bs, l2, x_min = _real_port(port_, log2, bs_in, bucket_sizing_p)
        if bs_in > 0 or x_min <= 0:
            return bs, l2, x_min
        reach = max(_unit_reach(u) for u in port_.agg_list)
        if not np.isfinite(reach) or (1 << l2) * bs > reach:
            return bs, l2, x_min
        need = B._need_log2(reach, bs)
        if need <= log2:
            return bs, max(l2, need), x_min
        # the window cannot hold a claim at the cap: a plain grid from 0 if that fits
        hi = float(port_._bs_window_df.loc['used', 'x_max']) if 'used' in port_._bs_window_df.index else np.inf
        need0 = B._need_log2(hi, bs)
        if need0 <= log2:
            return bs, max(need0, 1), 0.0
        return bs, l2, x_min
    B.port_best_window = port


def patch_agg_cover_origin():
    """[agg-cover-origin] no windowed origin under an aggregate cover: the ceded amount can sit
    below the window's start."""
    real = A.Aggregate._bs_window

    def ab(self, log2, bs_in, x_min_in, bucket_sizing_p, window_convention=None):
        if getattr(self, 'agg_reins', None) and x_min_in is None:
            x_min_in = 0.0
        return real(self, log2, bs_in, x_min_in, bucket_sizing_p, window_convention)
    A.Aggregate._bs_window = ab


def patch_port_no_window_with_reins():
    """[port-reins-origin] a Portfolio with a reinsured unit keeps the 0-based grid when that
    fits the cap, the gate the Aggregate already applies to itself."""
    real = B.port_best_window

    def port(port_, log2=16, bs_in=0, bucket_sizing_p=B.BUCKET_SIZING_P):
        bs, l2, x_min = real(port_, log2, bs_in, bucket_sizing_p)
        if bs_in > 0 or x_min <= 0:
            return bs, l2, x_min
        reins = any(getattr(u, 'occ_reins', None) or getattr(u, 'agg_reins', None)
                    for u in port_.agg_list)
        if not reins:
            return bs, l2, x_min
        hi = x_min + (1 << l2) * bs
        need0 = B._need_log2(hi, bs)
        if need0 <= log2:
            return bs, max(need0, 1), 0.0
        return bs, l2, x_min
    B.port_best_window = port


def patch_exact_windowed():
    """[exact-windowed] a fixed or dfreq count on a lattice severity, with no reinsurance and no
    signed atom, takes the exact support as its window wherever that sits, at the lattice
    bucket, with log2 the larger of what the support width and the largest claim need. Taken
    when it fits the cap and is no coarser than the grid otherwise chosen."""
    real = A.Aggregate._bs_window

    def ab(self, log2, bs_in, x_min_in, bucket_sizing_p, window_convention=None):
        bs, l2, x0 = real(self, log2, bs_in, x_min_in, bucket_sizing_p, window_convention)
        if bs_in or x_min_in is not None or self._signed_severity():
            return bs, l2, x0
        if getattr(self, 'occ_reins', None) or getattr(self, 'agg_reins', None):
            return bs, l2, x0
        ed = _real_exact(self)
        if ed is None:
            return bs, l2, x0
        pool = view_atoms(self)
        fit = _dyadic_lattice_step(pool)
        if fit is None:
            return bs, l2, x0
        g, pk, reach = final_view(self)
        freq = self.frequency
        n_atoms = (np.asarray(freq.freq_a, float) if freq.freq_name in ('empirical', 'renewal')
                   else np.array([float(self.n)]))
        lo = float(n_atoms.min() * g.min())
        hi = float(n_atoms.max() * g.max())
        if np.any(n_atoms == 0):
            lo = min(lo, 0.0)
        from aggregate.utilities import _lattice_candidates
        for b in _lattice_candidates(fit[0]):
            x_lo = float(np.floor(lo / b) * b)
            need = max(B._need_log2(hi - x_lo, b), B._need_log2(reach, b), 1)
            if need <= log2 and (b < bs or (b == bs and need < l2)):
                slack = (1 << need) * b - (hi - x_lo)
                x = float(np.floor(max(x_lo - slack / 2, 0.0) / b) * b)
                df = self._bs_window_df
                df.loc['used', ['bs', 'log2', 'x_min']] = [b, need, x]
                return float(b), int(need), x
        return bs, l2, x0
    A.Aggregate._bs_window = ab


def patch_four():
    patch_views()
    patch_window_holds_severity()
    patch_agg_cover_origin()
    patch_port_no_window_with_reins()


def patch_all():
    patch_four()
    patch_exact_windowed()
