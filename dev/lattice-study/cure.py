import sys, os, warnings, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter('ignore')
import pandas as pd, numpy as np
import oracle, sweep
from oracle import run, classify
which = sys.argv[1]
if which == 'four':
    import patches; patches.patch_four()
sweep.load_laws()
sets = sweep.make_sets()
f = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res_four.pkl'))
inf = f[f['class'].str.startswith('infeasible')]
jobs = [(key, specs) for key, rows in sets.items() for specs in rows]
byprog = {}
for key, specs in jobs:
    port = key.startswith('G') or key.startswith('H')
    prog = ('port P ' + ' '.join(oracle.decl(s, f'U{i}') for i, s in enumerate(specs))) if port else oracle.decl(specs[0])
    byprog[prog] = (key, specs, port)
out = []
for _, r in inf.iterrows():
    key, specs, port = byprog[r.prog]
    need = int(min(r.need0, r.needw))
    rr = run(specs, log2=need, port=port)
    out.append(dict(need=need, cls=classify(rr), bs=rr['bs'], log2=rr['log2'], tv=rr.get('tv'), prog=r.prog[:90]))
df = pd.DataFrame(out)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 95); pd.set_option('display.max_rows', 50)
print(which, ': built at the needed log2 ->', df.cls.value_counts().to_dict())
print(df[df.cls != 'exact'].to_string(index=False))
