"""Every single-unit program of sets A to F built twice, as an Aggregate and as a one-unit
Portfolio, at the same log2: the grids each picks and the total variation between their laws.
A one-unit Portfolio adds nothing to the mathematics, so any difference is the Portfolio sizer
or the combine, never the program.

    .venv/Scripts/python.exe dev/lattice-study/port_vs_agg.py base|four
"""
import os
import sys
import time
import warnings
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import sweep
from oracle import decl


def _law(ob):
    d = ob.density_df
    lib = defaultdict(float)
    for v, p in zip(d.index.to_numpy(float), d['p_total'].to_numpy(float)):
        if p > 0:
            lib[round(v, 9)] += p
    return lib


def _tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def _one(args):
    key, spec, log2 = args
    from aggregate import build
    prog = decl(spec, 'U0')
    out = dict(set=key, prog=prog)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = build(prog, log2=log2)
            p = build('port P ' + prog, log2=log2)
    except Exception as e:
        out.update(error=f'{type(e).__name__}: {str(e)[:80]}')
        return out
    la, lp = _law(a), _law(p)
    out.update(a_bs=a.bs, a_log2=a.log2, a_x0=float(a.density_df.index[0]),
               p_bs=p.bs, p_log2=p.log2, p_x0=float(p.density_df.index[0]),
               tv=_tv(la, lp), a_verdict=a.validation_description,
               p_verdict=p.validation_description)
    out['same_grid'] = (out['a_bs'] == out['p_bs'] and out['a_log2'] == out['p_log2']
                        and out['a_x0'] == out['p_x0'])
    return out


def main(tag):
    from multiprocessing import Pool
    patch_name = {'base': None, 'four': 'patch_four'}[tag]
    sets = sweep.make_sets()
    jobs = [(key, specs[0], 16) for key, rows in sets.items()
            if key[0] in 'ABCDEF' for specs in rows]
    t0 = time.time()
    with Pool(8, initializer=sweep._init, initargs=(patch_name,)) as pool:
        recs = pool.map(_one, jobs, chunksize=4)
    df = pd.DataFrame(recs)
    df.to_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'pva_{tag}.pkl'))
    print(f'{tag}: {len(df)} programs in {time.time() - t0:.0f} s')
    df['agree'] = df['tv'] < 1e-9
    t = df.groupby('set').agg(n=('prog', 'size'), same_grid=('same_grid', 'sum'),
                              agree=('agree', 'sum'), errors=('error', 'count')
                              if 'error' in df else ('prog', lambda s: 0))
    t.loc['all'] = t.sum()
    print(t)
    bad = df[~df['agree']]
    print(f'\n{len(bad)} one-unit Portfolios differ from their Aggregate (tv >= 1e-9):')
    for _, r in bad.iterrows():
        print(f"  tv {r['tv']:.3g} | agg bs {r['a_bs']:g} log2 {r['a_log2']} x0 {r['a_x0']:g}"
              f" | port bs {r['p_bs']:g} log2 {r['p_log2']} x0 {r['p_x0']:g} | {r['prog']}")


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    main(sys.argv[1] if len(sys.argv) > 1 else 'base')
