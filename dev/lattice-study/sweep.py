"""Lattice grid sizing study, 2026-10-30. Study scripts, not shipped code; see dev/note-lattice-grid-sizing.md.

Run from the repository: .venv/Scripts/python.exe dev/lattice-study/sweep.py base|views|window|four|all
Program sets for the lattice study, and the sweep that runs them through the oracle.
"""
import os
import pickle
import random
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

import oracle
from oracle import run, classify

LAWS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'laws.pkl')


def load_laws():
    try:
        oracle.LAWS.update(pickle.load(open(LAWS_PATH, 'rb')))
    except FileNotFoundError:
        pass


def save_laws():
    pickle.dump(oracle.LAWS, open(LAWS_PATH, 'wb'))

ATOM_SETS = {
    'small': [[1], [3], [1, 2], [2, 4, 6], [3, 6, 9], [1, 2, 3, 4, 5], [5, 10, 15], [7, 11, 13]],
    'hundreds': [[100], [100, 200], [100, 400, 900], [250, 500, 1000], [115, 120, 125], [300, 700]],
    'near': [[1000, 1001], [1000, 1003], [1000, 1001, 1002], [10002, 10007], [999, 1000, 1001], [1000, 1005]],
    'thousands': [[1000, 2000, 3000], [1000, 2000, 5000], [500, 1000, 2000], [2500, 7500]],
    'fraction': [[0.25, 0.5, 1.75], [0.75, 1.5, 2.25], [0.5, 1], [1.25, 2.5]],
    'signed': [[-5, -4], [-1.75, 0.25, 0.5], [-2, 1, 3], [-10, 10]],
}
COUNTS = [1, 2, 3, 10, 30, 100, 1000]
POISSON = [1, 3, 10, 100]


def _probs(rng, k):
    if rng.random() < 0.5 or k == 1:
        return None
    w = [rng.choice([1, 2, 3, 4]) for _ in range(k)]
    t = sum(w)
    return [x / t for x in w]


def _layer_for(rng, atoms):
    """A layer whose attachment and limit sit at round numbers near the atoms."""
    lo, hi = min(atoms), max(atoms)
    scale = 10 ** max(0, int(round(__import__('math').log10(max(abs(hi), 1)))) - 1)
    a = rng.choice([0, scale, 2 * scale, 2.5 * scale, 5 * scale])
    y = rng.choice([scale, 5 * scale, 10 * scale, 15 * scale, 2 * hi])
    return float(y), float(a)


def make_sets(seed=20261030):
    rng = random.Random(seed)
    sets = {}

    def pick_atoms(fams):
        fam = rng.choice(fams)
        atoms = rng.choice(ATOM_SETS[fam])
        return fam, atoms

    # A: fixed count, plain
    rows = []
    for fam, group in ATOM_SETS.items():
        for atoms in group:
            for n in COUNTS:
                rows.append([dict(freq=('fixed', n), atoms=atoms, probs=_probs(rng, len(atoms)))])
    sets['A fixed plain'] = rows
    # B: dfreq, plain
    rows = []
    for fam, group in ATOM_SETS.items():
        for atoms in group:
            for fr in ([1], [0, 1], [1, 2, 3], list(range(1, 11)), list(range(1, 101)), [0, 5, 10]):
                rows.append([dict(freq=('dfreq', fr, None), atoms=atoms, probs=_probs(rng, len(atoms)))])
    sets['B dfreq plain'] = rows
    # C: poisson, plain (signed excluded: the enumeration is slow and signed is its own plan)
    rows = []
    for fam, group in ATOM_SETS.items():
        if fam == 'signed':
            continue
        for atoms in group:
            for lam in POISSON:
                rows.append([dict(freq=('poisson', lam), atoms=atoms, probs=_probs(rng, len(atoms)))])
    sets['C poisson plain'] = rows
    # D: occurrence reinsurance
    rows = []
    for _ in range(80):
        fam, atoms = pick_atoms(['small', 'hundreds', 'near', 'thousands'])
        y, a = _layer_for(rng, atoms)
        share = rng.choice([1.0, 1.0, 0.5, 0.25])
        n = rng.choice([1, 2, 3, 10, 30, 100])
        freq = ('fixed', n) if rng.random() < 0.7 else ('poisson', rng.choice([1, 3, 10]))
        rows.append([dict(freq=freq, atoms=atoms, probs=_probs(rng, len(atoms)),
                          occ=[(share, y, a)], occ_kind=rng.choice(['net', 'ceded']))])
    sets['D occurrence reins'] = rows
    # E: aggregate reinsurance
    rows = []
    for _ in range(60):
        fam, atoms = pick_atoms(['small', 'hundreds', 'near', 'thousands'])
        n = rng.choice([1, 2, 3, 10, 30])
        freq = ('fixed', n) if rng.random() < 0.7 else ('poisson', rng.choice([1, 3, 10]))
        m = n * sum(atoms) / len(atoms)
        y, a = _layer_for(rng, [m])
        share = rng.choice([1.0, 1.0, 0.5])
        rows.append([dict(freq=freq, atoms=atoms, probs=_probs(rng, len(atoms)),
                          agg=(rng.choice(['net', 'ceded']), share, y, a))])
    sets['E aggregate reins'] = rows
    # F: layered dsev
    rows = []
    for _ in range(50):
        fam, atoms = pick_atoms(['small', 'hundreds', 'near', 'thousands'])
        y, a = _layer_for(rng, atoms)
        if a >= max(atoms):
            a = 0.0
        n = rng.choice([1, 2, 3, 10, 30, 100])
        freq = ('fixed', n) if rng.random() < 0.7 else ('poisson', rng.choice([1, 3, 10]))
        rows.append([dict(freq=freq, atoms=atoms, probs=_probs(rng, len(atoms)), layer=(y, a))])
    sets['F layered dsev'] = rows
    # G: one-unit Portfolios, drawn from A to F
    pool = [r[0] for key in ('A fixed plain', 'B dfreq plain', 'C poisson plain',
                             'D occurrence reins', 'E aggregate reins', 'F layered dsev')
            for r in sets[key]]
    sets['G port one unit'] = [[rng.choice(pool)] for _ in range(80)]
    # H: two-unit Portfolios
    sets['H port two units'] = [[rng.choice(pool), rng.choice(pool)] for _ in range(60)]
    return sets


_PATCH = None


def _init(patch_name):
    import warnings
    warnings.simplefilter('ignore')
    import logging
    logging.disable(logging.CRITICAL)
    if patch_name:
        import patches
        getattr(patches, patch_name)()


def _one(args):
    key, specs, log2 = args
    port = key.startswith('G') or key.startswith('H')
    r = run(specs, log2=log2, port=port)
    r['set'] = key
    r['class'] = classify(r)
    return r


def sweep(sets, log2=16, tag='base', patch_name=None, workers=8):
    from multiprocessing import Pool
    jobs = [(key, specs, log2) for key, rows in sets.items() for specs in rows]
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(patch_name,)) as pool:
        recs = pool.map(_one, jobs, chunksize=4)
    df = pd.DataFrame(recs)
    df.to_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'res_{tag}.pkl'))
    print(f'{tag}: {len(df)} programs in {time.time() - t0:.0f} s')
    return df


def summary(df):
    order = ['exact', 'exact (beyond spec)', 'WRONG-flagged', 'WRONG-silent',
             'infeasible-warned', 'infeasible-flagged', 'INFEASIBLE-silent', 'no-lattice', 'error']
    t = pd.crosstab(df['set'], df['class']).reindex(columns=[c for c in order if c in df['class'].unique()], fill_value=0)
    t.loc['all'] = t.sum()
    return t


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'base'
    sets = make_sets()
    patch_name = {'base': None, 'views': 'patch_views', 'window': 'patch_window_holds_severity',
                  'origin': 'patch_agg_cover_origin', 'four': 'patch_four', 'all': 'patch_all'}[tag]
    df = sweep(sets, tag=tag, patch_name=patch_name)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
    print(summary(df))
