"""Cross-check of the study oracle against nttagg, the exact in-library engine, on the integer programs of sets A and B."""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter('ignore')
import numpy as np, pandas as pd
import oracle, sweep
from aggregate import build
from aggregate.ntt import engine_status
print(engine_status())
sweep.load_laws()
sets = sweep.make_sets()
rows = []
t0 = time.time()
for key in ('A fixed plain', 'B dfreq plain'):
    for specs in sets[key]:
        s = specs[0]
        atoms = s['atoms']
        if any(a != int(a) or a < 0 for a in atoms):
            continue
        freq = s['freq']
        fa = [freq[1]] if freq[0] == 'fixed' else list(freq[1])
        nmax, smax = max(fa), max(atoms)
        if nmax * smax > 1 << 17:
            continue
        # the generator's probabilities are small-integer weights over their sum, so the
        # fraction with the smallest denominator under 100 is the exact value
        from fractions import Fraction
        fr = [Fraction(p).limit_denominator(100) for p in s['probs']] if s.get('probs') else None
        probs = (' [' + ' '.join(f'{f.numerator}/{f.denominator}' for f in fr) + ']') if fr else ''
        prog = f"nttagg N dfreq [{' '.join(str(int(v)) for v in fa)}] dsev [{' '.join(str(int(a)) for a in atoms)}]{probs}"
        # the oracle sees the same rounded probabilities as nttagg's text
        spec = dict(s)
        if fr:
            spec['probs'] = [float(f) for f in fr]
        law = oracle.exact(spec)
        try:
            a = build(prog)
        except Exception as e:
            rows.append(dict(prog=prog, err=str(e)[:70])); continue
        d = a.density_df
        lib = {round(float(x), 9): float(p) for x, p in zip(d.index, d['p_exact_float']) if p > 0}
        keys = set(law) | set(lib)
        tv = 0.5 * sum(abs(law.get(k, 0.0) - lib.get(k, 0.0)) for k in keys)
        rows.append(dict(prog=prog, tv=tv, log2=a.log2, n=len(lib)))
df = pd.DataFrame(rows)
print(f'{len(df)} integer programs through nttagg in {time.time()-t0:.0f} s')
print('errors:', df.err.notna().sum() if 'err' in df else 0)
ok = df[df.tv.notna()]
print('max TV oracle vs nttagg exact:', ok.tv.max(), ' programs with TV > 1e-9:', (ok.tv > 1e-9).sum())
print(ok.sort_values('tv', ascending=False).head(5).to_string(index=False))
if 'err' in df: print(df[df.err.notna()].head(5).to_string(index=False))
