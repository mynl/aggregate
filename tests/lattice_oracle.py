"""An independent oracle for discrete DecL programs ([Correctness-Oracle]).

A program is a dict spec. :func:`decl` renders it, :func:`exact` enumerates the
law of the quantity ``density_df.p_total`` holds, and :func:`feasible` gives,
from that law alone, the smallest ``log2`` at which a dyadic grid holds it. None
of this reads the sizer, so a program is checked against an answer computed
independently of the code under test, where a snapshot can only show that
something moved.

The enumeration is a numpy convolution on the integer lattice of the atoms, exact
to rounding, with the step an exact rational gcd and no depth limit. Where a
program is in ``nttagg``'s scope, :func:`reference_law` returns ``nttagg``'s exact
answer, and the enumeration otherwise.

:func:`make_sets` gives the study's 858 seeded programs in eight sets. The
exploratory copy, with its pickles and prototype patches, is
``dev/lattice-study/``; the evidence is ``dev/note-lattice-grid-sizing.md``.
"""
from __future__ import annotations

import math
import random
import warnings
from collections import defaultdict
from fractions import Fraction

import numpy as np

from aggregate import build
from aggregate.constants import LatticeGridWarning

#: Below this an outcome of a count or an enumerated law is dropped: past the
#: float noise of the probabilities and far under the comparison tolerance.
TRUNC = 1e-17

#: Total variation above this is a wrong law. It admits FFT rounding on a
#: thousand-claim book; the real failures are at 0.5 and above.
TV_TOLERANCE = 1e-4


# ---------------------------------------------------------------- the law

def _layer(x, a, y):
    return min(max(x - a, 0.0), y)


def sev_views(spec):
    """``(gross atoms, probs, ceded atoms, net atoms)`` per claim, after any layer."""
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
        return dict(zip((int(v) for v in freq[1]),
                        freq[2] or [1 / len(freq[1])] * len(freq[1])))
    if freq[0] == 'poisson':
        lam = float(freq[1])
        ns, n, p = {}, 0, math.exp(-lam)
        while True:
            if p > TRUNC:
                ns[n] = p
            if n > lam and p < TRUNC:
                break
            n += 1
            p *= lam / n
        return ns
    raise ValueError(freq)


def _agg_law(spec, per_claim, ps):
    """Law of the sum over N claims of ``per_claim`` with probabilities ``ps``."""
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
    per = {'net': net, 'ceded': ced}.get(spec.get('occ_kind'), xs)
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
    """Exact law of the independent sum of the units."""
    tot = {0.0: 1.0}
    for law in (exact(s) for s in specs):
        nxt = defaultdict(float)
        for v, p in tot.items():
            for x, q in law.items():
                nxt[round(v + x, 9)] += p * q
        tot = {v: p for v, p in nxt.items() if p > TRUNC}
    return tot


# ---------------------------------------------------------- the nttagg reference

def in_ntt_scope(spec):
    """Whether ``nttagg`` can state this program: a ``dfreq`` or fixed count on
    nonnegative integer atoms, with no layer or reinsurance, and a support small
    enough to run as a test."""
    if spec['freq'][0] not in ('fixed', 'dfreq'):
        return False
    if spec.get('layer') or spec.get('occ') or spec.get('agg'):
        return False
    if any(a != int(a) or a < 0 for a in spec['atoms']):
        return False
    counts = [spec['freq'][1]] if spec['freq'][0] == 'fixed' else spec['freq'][1]
    return max(counts) * max(spec['atoms']) <= 1 << 17


def _ntt_law(spec):
    """``nttagg``'s exact law of a program in its scope.

    The generator's probabilities are small-integer weights over their sum, so the
    fraction with the smallest denominator under 100 is the exact value.
    """
    fr = ([Fraction(p).limit_denominator(100) for p in spec['probs']]
          if spec.get('probs') else None)
    probs = (' [' + ' '.join(f'{f.numerator}/{f.denominator}' for f in fr) + ']') if fr else ''
    freq = spec['freq']
    counts = [freq[1]] if freq[0] == 'fixed' else list(freq[1])
    atoms = ' '.join(str(int(a)) for a in spec['atoms'])
    prog = (f"nttagg N dfreq [{' '.join(str(int(v)) for v in counts)}] "
            f"dsev [{atoms}]{probs}")
    support = max(counts) * max(spec['atoms'])
    a = build(prog, log2=max(16, int(support).bit_length()))
    d = a.density_df
    return {round(float(x), 9): float(p) for x, p in zip(d.index, d['p_exact_float']) if p > 0}


def reference_law(specs):
    """The reference answer: ``nttagg`` for one unit in its scope, else the enumeration."""
    if len(specs) == 1:
        return _ntt_law(specs[0]) if in_ntt_scope(specs[0]) else exact(specs[0])
    return exact_port(specs)


# ------------------------------------------------------------------ feasibility

def dyadic_step(pts):
    """Exact rational gcd of the points as a float, or ``None`` when it is not dyadic or is 0."""
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
    """``(h, log2 from 0, log2 windowed, reach)``: the exact grid the law allows, or ``h`` ``None``.

    ``reach`` is the width of the hull of the gross severity support and 0. The
    severity is discretized on ``xs_sev = (arange(N) - i0) * bs`` with physical 0
    at index ``i0``, so ``N * bs`` must span it whatever the output origin.
    """
    pts = list(law)
    reach = 0.0
    for s in specs:
        xs, _, ced, net = sev_views(s)
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


# ------------------------------------------------------------------------- DecL

def _fmt(v):
    return f'{v:g}'


def decl(spec, name='X'):
    """The DecL of a program. Probabilities are printed at full precision."""
    freq = spec['freq']
    atoms = ' '.join(_fmt(v) for v in spec['atoms'])
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
        lay_txt = (f'{_fmt(sh * 100)}% po {_fmt(y)} xs {_fmt(a)}' if sh != 1
                   else f'{_fmt(y)} xs {_fmt(a)}')
        ag = f' aggregate {"net of" if kind == "net" else "ceded to"} {lay_txt}'
    return f'agg {name} {head}{layer} dsev [{atoms}]{probs}{occ}{tail}{ag}'


def program(specs):
    """The DecL of one unit, or of a Portfolio of two."""
    if len(specs) == 1:
        return decl(specs[0])
    return port_program(specs)


def port_program(specs):
    """The DecL of a Portfolio of these units, one unit included."""
    return 'port P ' + ' '.join(decl(s, f'U{i}') for i, s in enumerate(specs))


def law_of(ob):
    """The nonzero outcomes of an object's ``p_total``, keyed as the oracle's are."""
    d = ob.density_df
    out = defaultdict(float)
    for v, p in zip(d.index.to_numpy(float), d['p_total'].to_numpy(float)):
        if p > 0:
            out[round(v, 9)] += p
    return out


def total_variation(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def compare(specs, port=False, log2=16):
    """Build a program and compare it with the reference.

    Returns
    -------
    dict
        ``prog``, the oracle's ``h``, ``need0``, ``needw`` and ``feasible``, and
        the library's ``tv``, ``wrong``, ``warned``, ``verdict``, ``bs``, ``log2``.
    """
    prog = port_program(specs) if port else program(specs)
    law = reference_law(specs)
    h, need0, needw, reach = feasible(specs, law)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        ob = build(prog, log2=log2)
    tv = total_variation(law, law_of(ob))
    return dict(prog=prog, h=h, need0=need0, needw=needw, reach=reach,
                feasible=h is not None and min(need0, needw) <= log2,
                tv=tv, wrong=tv > TV_TOLERANCE,
                warned=any(issubclass(w.category, LatticeGridWarning) for w in rec),
                verdict=ob.validation_description, bs=ob.bs, log2=ob.log2)


# ----------------------------------------------------------------- program sets

ATOM_SETS = {
    'small': [[1], [3], [1, 2], [2, 4, 6], [3, 6, 9], [1, 2, 3, 4, 5], [5, 10, 15], [7, 11, 13]],
    'hundreds': [[100], [100, 200], [100, 400, 900], [250, 500, 1000], [115, 120, 125], [300, 700]],
    'near': [[1000, 1001], [1000, 1003], [1000, 1001, 1002], [10002, 10007], [999, 1000, 1001],
             [1000, 1005]],
    'thousands': [[1000, 2000, 3000], [1000, 2000, 5000], [500, 1000, 2000], [2500, 7500]],
    'fraction': [[0.25, 0.5, 1.75], [0.75, 1.5, 2.25], [0.5, 1], [1.25, 2.5]],
    'signed': [[-5, -4], [-1.75, 0.25, 0.5], [-2, 1, 3], [-10, 10]],
}
COUNTS = [1, 2, 3, 10, 30, 100, 1000]
POISSON = [1, 3, 10, 100]
SEED = 20261030


def _probs(rng, k):
    if rng.random() < 0.5 or k == 1:
        return None
    w = [rng.choice([1, 2, 3, 4]) for _ in range(k)]
    t = sum(w)
    return [x / t for x in w]


def _layer_for(rng, atoms):
    """A layer whose attachment and limit sit at round numbers near the atoms."""
    hi = max(atoms)
    scale = 10 ** max(0, int(round(math.log10(max(abs(hi), 1)))) - 1)
    a = rng.choice([0, scale, 2 * scale, 2.5 * scale, 5 * scale])
    y = rng.choice([scale, 5 * scale, 10 * scale, 15 * scale, 2 * hi])
    return float(y), float(a)


def make_sets(seed=SEED):
    """The eight program sets. Each program is a list of unit specs."""
    rng = random.Random(seed)
    sets = {}
    fams = ['small', 'hundreds', 'near', 'thousands']

    def pick_atoms():
        return rng.choice(ATOM_SETS[rng.choice(fams)])

    sets['A fixed plain'] = [
        [dict(freq=('fixed', n), atoms=atoms, probs=_probs(rng, len(atoms)))]
        for group in ATOM_SETS.values() for atoms in group for n in COUNTS]
    sets['B dfreq plain'] = [
        [dict(freq=('dfreq', fr, None), atoms=atoms, probs=_probs(rng, len(atoms)))]
        for group in ATOM_SETS.values() for atoms in group
        for fr in ([1], [0, 1], [1, 2, 3], list(range(1, 11)), list(range(1, 101)), [0, 5, 10])]
    # signed excluded: the enumeration is slow and signed books are their own plan
    sets['C poisson plain'] = [
        [dict(freq=('poisson', lam), atoms=atoms, probs=_probs(rng, len(atoms)))]
        for fam, group in ATOM_SETS.items() if fam != 'signed'
        for atoms in group for lam in POISSON]
    rows = []
    for _ in range(80):
        atoms = pick_atoms()
        y, a = _layer_for(rng, atoms)
        share = rng.choice([1.0, 1.0, 0.5, 0.25])
        n = rng.choice([1, 2, 3, 10, 30, 100])
        freq = ('fixed', n) if rng.random() < 0.7 else ('poisson', rng.choice([1, 3, 10]))
        rows.append([dict(freq=freq, atoms=atoms, probs=_probs(rng, len(atoms)),
                          occ=[(share, y, a)], occ_kind=rng.choice(['net', 'ceded']))])
    sets['D occurrence reins'] = rows
    rows = []
    for _ in range(60):
        atoms = pick_atoms()
        n = rng.choice([1, 2, 3, 10, 30])
        freq = ('fixed', n) if rng.random() < 0.7 else ('poisson', rng.choice([1, 3, 10]))
        y, a = _layer_for(rng, [n * sum(atoms) / len(atoms)])
        share = rng.choice([1.0, 1.0, 0.5])
        rows.append([dict(freq=freq, atoms=atoms, probs=_probs(rng, len(atoms)),
                          agg=(rng.choice(['net', 'ceded']), share, y, a))])
    sets['E aggregate reins'] = rows
    rows = []
    for _ in range(50):
        atoms = pick_atoms()
        y, a = _layer_for(rng, atoms)
        if a >= max(atoms):
            a = 0.0
        n = rng.choice([1, 2, 3, 10, 30, 100])
        freq = ('fixed', n) if rng.random() < 0.7 else ('poisson', rng.choice([1, 3, 10]))
        rows.append([dict(freq=freq, atoms=atoms, probs=_probs(rng, len(atoms)), layer=(y, a))])
    sets['F layered dsev'] = rows
    pool = [r[0] for key in ('A fixed plain', 'B dfreq plain', 'C poisson plain',
                             'D occurrence reins', 'E aggregate reins', 'F layered dsev')
            for r in sets[key]]
    sets['G port one unit'] = [[rng.choice(pool)] for _ in range(80)]
    sets['H port two units'] = [[rng.choice(pool), rng.choice(pool)] for _ in range(60)]
    return sets
