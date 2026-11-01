"""Independent oracle for discrete DecL programs.

A program is a dict spec; ``decl(spec)`` renders it and ``exact(spec)`` enumerates the
law of the quantity the library's ``density_df.p_total`` holds. ``feasible(spec)`` says,
from the exact law alone, the smallest ``log2`` at which a dyadic grid holds it exactly,
from 0 and windowed (the severity grid runs from 0, so a windowed grid must still reach
the largest gross severity outcome).
"""
import math
import warnings
from collections import defaultdict
from fractions import Fraction

import numpy as np


TRUNC = 1e-17


# ---------------------------------------------------------------- the law
def _layer(x, a, y):
    return min(max(x - a, 0.0), y)


def sev_views(spec):
    """(gross atoms, probs), ceded atoms, net atoms, all per claim, after any layer."""
    xs = [float(v) for v in spec['atoms']]
    ps = spec.get('probs') or [1.0 / len(xs)] * len(xs)
    lay = spec.get('layer')              # (limit, attachment) on the severity
    if lay is not None:
        y, a = lay
        keep = [(x, p) for x, p in zip(xs, ps) if x > a]
        tot = sum(p for _, p in keep)
        xs = [_layer(x, a, y) for x, _ in keep]
        ps = [p / tot for _, p in keep]
    occ = spec.get('occ')                # list of (share, limit, attachment)
    ced = [0.0] * len(xs)
    if occ:
        for share, y, a in occ:
            ced = [c + share * _layer(x, a, y) for c, x in zip(ced, xs)]
    net = [x - c for x, c in zip(xs, ced)]
    return xs, ps, ced, net


def _counts(spec):
    freq = spec['freq']
    if freq[0] == 'fixed':
        return {int(freq[1]): 1.0}
    if freq[0] == 'dfreq':
        return dict(zip((int(v) for v in freq[1]), freq[2] or [1 / len(freq[1])] * len(freq[1])))
    if freq[0] == 'poisson':
        lam = float(freq[1])
        ns, n, p, cum = {}, 0, math.exp(-lam), 0.0
        while True:
            if p > TRUNC:
                ns[n] = p
            cum += p
            if n > lam and p < TRUNC:
                break
            n += 1
            p *= lam / n
        return ns
    raise ValueError(freq)


def _agg_law(spec, per_claim, ps):
    """Law of the sum over N claims of ``per_claim`` with probs ``ps``: numpy convolution on
    the integer lattice of the atoms (exact in float64 to rounding)."""
    ns = _counts(spec)
    h = dyadic_step(per_claim) or 1.0
    k = np.rint(np.asarray(per_claim, float) / h).astype(int)
    kmin = int(k.min())
    pmf = np.zeros(int(k.max()) - kmin + 1)
    for kk, p in zip(k, ps):
        pmf[kk - kmin] += p
    nmax = max(ns)
    law = defaultdict(float)
    conv = np.array([1.0])
    for n in range(nmax + 1):
        w = ns.get(n, 0.0)
        if w:
            base = n * kmin
            for j in np.flatnonzero(conv > TRUNC):
                law[round((base + j) * h, 9)] += w * conv[j]
        if n == nmax:
            break
        conv = np.convolve(conv, pmf)
    return dict(law)


def exact(spec):
    """Exact law of what ``density_df.p_total`` holds for the program."""
    xs, ps, ced, net = sev_views(spec)
    occ_kind = spec.get('occ_kind')      # 'net' | 'ceded' | None
    per = {'net': net, 'ceded': ced}.get(occ_kind, xs)
    law = _agg_law(spec, per, ps)
    ag = spec.get('agg')                 # (kind, share, limit, attachment)
    if ag:
        kind, share, y, a = ag
        out = defaultdict(float)
        for v, p in law.items():
            c = share * _layer(v, a, y)
            out[round(c if kind == 'ceded' else v - c, 9)] += p
        law = dict(out)
    return law


def exact_port(specs):
    laws = [exact(s) for s in specs]
    tot = {0.0: 1.0}
    for law in laws:
        nxt = defaultdict(float)
        for v, p in tot.items():
            for x, q in law.items():
                nxt[round(v + x, 9)] += p * q
        tot = {v: p for v, p in nxt.items() if p > TRUNC}
    return tot


# ---------------------------------------------------------------- feasibility
def dyadic_step(pts):
    """Exact rational gcd of the points as a float, or None when it is not dyadic or is 0."""
    g = Fraction(0)
    for v in pts:
        f = Fraction(v).limit_denominator(1 << 30)
        g = (Fraction(math.gcd(g.numerator * f.denominator, f.numerator * g.denominator),
                      g.denominator * f.denominator) if g else f)
    g = abs(g)
    if g == 0 or (g.denominator & (g.denominator - 1)):
        return None
    return float(g)


def _need(span, h):
    return 0 if span <= 0 else int(math.ceil(math.log2(span / h + 1)))


def feasible(specs, law):
    """(h, log2 from 0, log2 windowed, reach): the exact grid the law allows, or h None.

    ``reach`` is the width of the hull of the gross severity support and 0: the severity is
    discretized on its own grid ``xs_sev = (arange(N) - i0) * bs`` with physical 0 at index
    ``i0`` (0 for a nonnegative severity, ``ceil(-lo / bs)`` for a signed one), so ``N * bs``
    must span from ``min(lo, 0)`` to ``max(hi, 0)`` whatever the output origin."""
    pts = list(law)
    reach = 0.0
    for s in specs:
        xs, ps, ced, net = sev_views(s)
        pts += xs + ced + net
        reach = max(reach, max(max(xs), 0.0) - min(min(xs), 0.0))
        ag = s.get('agg')
        if ag:
            pts += [ag[1] * ag[2], ag[1] * ag[3]]
    pts = [p for p in pts if np.isfinite(p)]
    h = dyadic_step(pts)
    if h is None:
        return None, None, None, reach
    lo, hi = min(law), max(law)
    from0 = max(_need(max(hi, 0.0) - min(lo, 0.0), h), _need(reach, h))
    win = max(_need(hi - lo, h), _need(reach, h))
    return h, max(from0, 1), max(win, 1), reach


# ---------------------------------------------------------------- DecL
def _fmt(v):
    return f'{v:g}'


def decl(spec, name='X'):
    freq = spec['freq']
    atoms = ' '.join(_fmt(v) for v in spec['atoms'])
    # probabilities at full precision: printed to 6 digits they sum to 1 - 1e-6, and the
    # library computes the severity moments from the probabilities as given (the deficit is an
    # atom at 0), which inflates sd by half and skew a thousandfold on a near-degenerate book
    probs = (' [' + ' '.join(repr(float(p)) for p in spec['probs']) + ']') if spec.get('probs') else ''
    lay = spec.get('layer')
    layer = f' {_fmt(lay[0])} xs {_fmt(lay[1])}' if lay else ''
    if freq[0] == 'fixed':
        head, tail = f'{freq[1]} claims', ' fixed'
    elif freq[0] == 'poisson':
        head, tail = f'{_fmt(freq[1])} claims', ' poisson'
    else:
        fa = (f'{int(freq[1][0])}:{int(freq[1][-1])}' if len(freq[1]) > 2 and
              list(freq[1]) == list(range(int(freq[1][0]), int(freq[1][-1]) + 1))
              else ' '.join(str(int(v)) for v in freq[1]))
        fp = (' [' + ' '.join(repr(float(p)) for p in freq[2]) + ']') if freq[2] else ''
        head, tail = f'dfreq [{fa}]{fp}', ''
    occ = ''
    if spec.get('occ'):
        parts = ' and '.join(f'{_fmt(sh * 100)}% po {_fmt(y)} xs {_fmt(a)}' if sh != 1
                             else f'{_fmt(y)} xs {_fmt(a)}' for sh, y, a in spec['occ'])
        occ = f' occurrence {"net of" if spec["occ_kind"] == "net" else "ceded to"} {parts}'
    ag = ''
    if spec.get('agg'):
        kind, sh, y, a = spec['agg']
        lay_txt = f'{_fmt(sh * 100)}% po {_fmt(y)} xs {_fmt(a)}' if sh != 1 else f'{_fmt(y)} xs {_fmt(a)}'
        ag = f' aggregate {"net of" if kind == "net" else "ceded to"} {lay_txt}'
    return f'agg {name} {head}{layer} dsev [{atoms}]{probs}{occ}{tail}{ag}'


# ---------------------------------------------------------------- the library
LAWS = {}


def run(specs, log2=16, port=False, **kw):
    """Build, compare with the oracle. Returns a dict of findings."""
    from aggregate import build
    from aggregate.constants import (InfeasibleGridWarning, LatticeGridWarning,
                                     reset_warn_once)
    grid_warnings = (InfeasibleGridWarning, LatticeGridWarning)
    if port or len(specs) > 1:
        prog = 'port P ' + ' '.join(decl(s, f'U{i}') for i, s in enumerate(specs))
        if prog not in LAWS:
            LAWS[prog] = exact_port(specs)
    else:
        prog = decl(specs[0])
        if prog not in LAWS:
            LAWS[prog] = exact(specs[0])
    law = LAWS[prog]
    h, from0, win, reach = feasible(specs, law)
    out = dict(prog=prog, h=h, need0=from0, needw=win, reach=reach,
               feasible=(h is not None and min(from0, win) <= log2))
    # warn_once keys on the object's name, and every program here is X or P
    reset_warn_once()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            ob = build(prog, log2=log2, **kw)
        except Exception as e:
            out.update(error=f'{type(e).__name__}: {str(e)[:80]}', wrong=True, flagged=True,
                       warned=False, tv=np.nan, bs=np.nan, log2=np.nan, x_min=np.nan)
            return out
        warned = any(issubclass(x.category, grid_warnings) for x in w)
        other = [x for x in w if not issubclass(x.category, grid_warnings)]
    d = ob.density_df
    lib = defaultdict(float)
    for v, p in zip(d.index.to_numpy(float), d['p_total'].to_numpy(float)):
        if p > 0:
            lib[round(v, 9)] += p
    keys = set(law) | set(lib)
    tv = 0.5 * sum(abs(law.get(k, 0.0) - lib.get(k, 0.0)) for k in keys)
    m_ex = sum(v * p for v, p in law.items())
    m_lib = sum(v * p for v, p in lib.items())
    verdict = ob.validation_description
    out.update(tv=tv, wrong=tv > 1e-4, mean_err=abs(m_lib - m_ex) / max(abs(m_ex), 1e-12),
               flagged=('fails' in verdict or 'infeasible' in verdict), warned=warned,
               view_flagged=any(f'; {word} fails' in verdict for word in ('net', 'ceded', 'output')),
               other_warnings=len(other), verdict=verdict,
               bs=ob.bs, log2=ob.log2, x_min=float(d.index[0]), mass=float(d['p_total'].sum()))
    return out


def classify(r):
    if r.get('error'):
        return 'error'
    if r['h'] is None:
        return 'no-lattice'
    if r['feasible']:
        if not r['wrong']:
            return 'exact'
        if r.get('view_flagged'):
            return 'WRONG-view-flagged'
        return 'WRONG-flagged' if r['flagged'] else 'WRONG-silent'
    if not r['wrong']:
        return 'exact (beyond spec)'
    if r['warned']:
        return 'infeasible-warned'
    return 'infeasible-flagged' if r['flagged'] else 'INFEASIBLE-silent'
