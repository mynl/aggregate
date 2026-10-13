"""Sweep the knowledge base comparing empirical-moment rules on every aggregate.

A standalone maintenance aid, sibling to ``bucket_baseline.py``. It builds every
``agg`` program in the chosen knowledge bases and recomputes the aggregate
skewness and CV from the raw ``agg_density`` under several rules, so a change to
the cleaning or deficit handling behind ``est_skew`` / ``est_cv`` can be read as
a per-object table before it lands. It was written for the review of
``dev/done/plan-1.2.0a14-moment-deficit.md`` and is that plan's acceptance check.

Each rule is a pair: how the density is cleaned, and what deficit is relocated
to the top of the grid (``pg * edge**k`` added to the k-th raw moment, the
``xsden_to_mwrangler`` convention).

* ``today``  : ``remove_fuzz`` at machine epsilon, then relocate ``1 - clean.sum()``
  unconditionally. This is the shipped behavior as of 1.2.0a13 and the column
  is checked against the object's own ``est_skew`` as a sanity test.
* ``plan``   : ``remove_fuzz`` at ``noise_floor(raw)``, the largest negative
  magnitude stepped up one float and capped at ``eps``, then relocate
  ``1 - raw.sum()`` only when it exceeds ``VALIDATION_NOISE``. The rule the
  plan adopts, as shipped at 1.2.0a14; the column must equal ``est_skew``
  from that version on.
* ``draft``  : ``remove_fuzz`` at the uncapped largest negative magnitude, then
  relocate ``1 - raw.sum()`` only when ``abs`` exceeds ``VALIDATION_NOISE`` and
  ``normalize`` is false. The first draft's rule, kept so the rejection in the
  plan can be reproduced.

The summary counts movers (relative change above ``1e-6`` from ``today``) and,
on objects with a finite nonzero theoretical skewness, how many move nearer to
or further from theory. Every mover is meant to be read, not gated: the plan
names the expected ones, and any other is a finding.

Usage
-----
From the shell (writes the CSV, prints the summary and the mover table)::

    uv run python scripts/moment_rule_sweep.py [-d library -d reproductions ...]
                                               [-o sweep.csv] [--time-cap 540]

The CSV defaults to the OS temp directory so a stray data file never lands in
the repository.

From Jupyter::

    from moment_rule_sweep import sweep, summarize
    df = sweep(('library', 'reproductions', '_test_suite'))
    summarize(df)
"""
import argparse
import os
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

from aggregate import Underwriter
from aggregate.moments import VALIDATION_NOISE
from aggregate.utilities import remove_fuzz, noise_floor

EPS = np.finfo(float).eps
DEFAULT_DATABASES = ('library', 'reproductions', '_test_suite')
RULES = ('today', 'plan', 'draft')


def moments(xs, den, deficit):
    """Mean, CV and skewness of ``den`` on ``xs`` with ``deficit`` placed at the grid edge."""
    bs = xs[1] - xs[0]
    edge = xs[-1] + bs
    e1 = (xs * den).sum() + deficit * edge
    e2 = (xs ** 2 * den).sum() + deficit * edge ** 2
    e3 = (xs ** 3 * den).sum() + deficit * edge ** 3
    v = e2 - e1 ** 2
    if v <= 0 or e1 == 0:
        return e1, np.nan, np.nan
    return e1, np.sqrt(v) / e1, (e3 - 3 * e1 * e2 + 2 * e1 ** 3) / v ** 1.5


def apply_rules(xs, raw, normalize):
    """Return ``{rule: (mean, cv, skew)}`` plus the raw deficit and largest negative."""
    maxneg = float(-raw[raw < 0].min()) if (raw < 0).any() else 0.0
    d_raw = float(1 - raw.sum())
    clean = remove_fuzz(raw)
    out = {'today': moments(xs, clean, 1 - clean.sum())}
    # the plan: capped floor (the shipped ``noise_floor``, which steps one
    # float above the largest negative magnitude so the strict ``remove_fuzz``
    # cut includes that bucket), one-sided gate on the raw deficit
    plan_clean = remove_fuzz(raw, eps=noise_floor(raw))
    out['plan'] = moments(xs, plan_clean, d_raw if d_raw > VALIDATION_NOISE else 0.0)
    # the rejected first draft: uncapped floor, abs gate, no relocation when normalized
    draft_clean = np.where(raw < maxneg, 0.0, raw)
    d_draft = 0.0 if normalize else (d_raw if abs(d_raw) > VALIDATION_NOISE else 0.0)
    out['draft'] = moments(xs, draft_clean, d_draft)
    return out, d_raw, maxneg


def sweep(databases=DEFAULT_DATABASES, time_cap=540.0, log=print):
    """Build every ``agg`` program in ``databases`` and tabulate the rules.

    Returns a DataFrame with one row per program. Programs that fail to build
    keep their row with the error text in ``err`` and NaN elsewhere.
    """
    warnings.simplefilter('ignore')
    uw = Underwriter(databases=list(databases), update=True)
    names = [n for (k, n) in uw.recipes.index if k == 'agg']
    rows = []
    t0 = time.time()
    for n in names:
        try:
            a = uw.build(n)
        except Exception as e:  # noqa: BLE001, a sweep records and moves on
            rows.append(dict(name=n, err=f'{type(e).__name__}: {str(e)[:80]}'))
            continue
        if a is None or getattr(a, 'agg_density', None) is None:
            continue
        try:
            theory = float(a.validation_df.loc['Agg', 'Sk'])
        except Exception:  # noqa: BLE001, no theoretical skewness for this shape
            theory = np.nan
        res, d_raw, maxneg = apply_rules(a.xs, a.agg_density, a.normalize)
        row = dict(name=n, log2=a.log2, normalize=a.normalize, deficit_raw=d_raw,
                   maxneg=maxneg, theory=theory, est_skew=a.est_skew, est_cv=a.est_cv)
        for r in RULES:
            row[f'{r}_cv'] = res[r][1]
            row[r] = res[r][2]
        rows.append(row)
        if time.time() - t0 > time_cap:
            log(f'time cap {time_cap:.0f}s reached after {len(rows)} programs')
            break
    log(f'{len(rows)} rows in {time.time() - t0:.0f}s')
    return pd.DataFrame(rows)


def summarize(df, log=print, rel=1e-6):
    """Print mover counts per rule and the full mover table."""
    fmt = lambda v: f'{v:.6g}'  # noqa: E731
    if 'err' in df:
        bad = df[df['err'].notna()]
        if len(bad):
            log(f'{len(bad)} programs did not build:')
            log(bad[['name', 'err']].to_string(index=False))
    df = df[df['today'].notna()].copy()
    log(f'built {len(df)}; today vs est_skew max abs diff '
        f'{(df.today - df.est_skew).abs().max():.3g}')
    log(f'largest negative above eps (cap binds): {(df.maxneg > EPS).sum()}; '
        f'no negatives: {(df.maxneg == 0).sum()}; '
        f'raw deficit > 1e-12: {(df.deficit_raw > VALIDATION_NOISE).sum()}; '
        f'raw deficit < -1e-12: {(df.deficit_raw < -VALIDATION_NOISE).sum()}')
    t = df[np.isfinite(df.theory) & (df.theory != 0)]
    e0 = ((t.today - t.theory) / t.theory).abs()
    for r in RULES[1:]:
        e1 = ((t[r] - t.theory) / t.theory).abs()
        moved = (t[r] - t.today).abs() > rel * t.today.abs()
        worse = moved & (e1 > e0 * 1.001 + 1e-9)
        better = moved & (e1 < e0 * 0.999)
        log(f'{r:6s} movers {moved.sum():3d}  better {better.sum():3d}  '
            f'worse {worse.sum():3d}  (of {len(t)} with theory)')
    moved_any = np.zeros(len(df), dtype=bool)
    for r in RULES[1:]:
        moved_any |= ((df[r] - df.today).abs() > rel * df.today.abs()).to_numpy()
    cols = ['name', 'log2', 'normalize', 'deficit_raw', 'maxneg', 'theory', *RULES]
    log(f'\nmovers under any rule: {moved_any.sum()}')
    log(df[moved_any][cols].to_string(index=False, float_format=fmt))
    cv_moved = (df.plan_cv - df.today_cv).abs() > rel * df.today_cv.abs()
    log(f'\nest_cv movers under plan: {cv_moved.sum()}')
    log(df[cv_moved][['name', 'log2', 'today_cv', 'plan_cv']]
        .to_string(index=False, float_format=fmt))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('-d', '--database', action='append',
                   help='knowledge base to sweep; repeatable; default library, '
                        'reproductions, _test_suite')
    p.add_argument('-o', '--output',
                   default=os.path.join(tempfile.gettempdir(), 'moment_rule_sweep.csv'),
                   help='CSV to write (default moment_rule_sweep.csv in the OS temp dir)')
    p.add_argument('--time-cap', type=float, default=540.0,
                   help='stop building after this many seconds (default 540)')
    args = p.parse_args(argv)
    df = sweep(args.database or DEFAULT_DATABASES, time_cap=args.time_cap)
    df.to_csv(args.output, index=False)
    print(f'wrote {args.output}')
    summarize(df)
    return 0


if __name__ == '__main__':
    sys.exit(main())
