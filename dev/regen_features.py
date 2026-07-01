"""Introspect the public surface of the core classes and audit ``dev/FEATURES.csv``.

``dev/FEATURES.csv`` is a *hand-curated* capability / consistency matrix: the
descriptions, thematic grouping, row order and ``notes`` (the ``WRINKLE`` flags)
are written by a human and must not be clobbered. This script therefore does not
overwrite it -- it **audits** it against a live introspection of the classes and
reports what has drifted, so the curated file can be hand-edited in the same
iteration that changes a public surface.

Run from anywhere (paths resolve relative to this file):

    uv run python dev/regen_features.py            # audit FEATURES.csv (exit 1 on drift)
    uv run python dev/regen_features.py --inventory # dump the live inventory, grouped
    uv run python dev/regen_features.py --all       # include scipy-inherited Severity members

The CSV is a *curated subset* -- it documents the capability surface, not every
internal attribute -- so only two kinds of drift are treated as **failures**
(non-zero exit); a third is purely informational:

* **MISMATCH** (fail) -- a ``Y`` / blank cell disagrees with the live class. This
  is the real guard: it catches a rename, a moved method, or a member that
  gained / lost a class.
* **STALE** (fail) -- a CSV row whose ``name`` is on no class (a deleted member
  or a typo).
* **UNDOCUMENTED** (info only) -- a public member absent from the CSV. Split into
  *capabilities* (property / method -- candidates for a new row) and a count of
  internal *attributes* (the long tail the curation deliberately omits).
  Severity members inherited from ``scipy.stats.rv_continuous`` are summarised in
  the CSV, so they are folded into one count unless ``--all`` is passed.

The build programs below exercise reinsurance (so the ``reins_*`` frames exist)
and a copula bivariate; keep them parseable if the DecL grammar moves.
"""
from __future__ import annotations

import csv
import inspect
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as ss

warnings.simplefilter('ignore')

CSV_PATH = Path(__file__).with_name('FEATURES.csv')

# CSV class columns, in order. Keys are the CSV header labels; values the short
# tags used in the grouped ``--inventory`` dump.
CLASS_COLS = ['Aggregate', 'Portfolio', 'BivariateAggregate', 'PnL',
              'ReinstatementAnalysis', 'VariableRatingAnalysis',
              'Severity', 'Frequency', 'Bounds']
SHORT = {'Aggregate': 'Agg', 'Portfolio': 'Port', 'BivariateAggregate': 'Biv',
         'PnL': 'PnL', 'ReinstatementAnalysis': 'Reinst',
         'VariableRatingAnalysis': 'VarRt', 'Severity': 'Sev',
         'Frequency': 'Freq', 'Bounds': 'Bnd'}


def build_objects() -> dict:
    """One live object per class column (built + updated where relevant)."""
    from aggregate import build
    from aggregate.bounds import Bounds

    objs = {}
    a = build('agg E 100 claims sev lognorm 100 cv 2 '
              'occurrence net of 50 xs 50 poisson')
    objs['Aggregate'] = a
    objs['Portfolio'] = build(
        'port P agg A 100 claims sev lognorm 100 cv 2 poisson '
        'agg B 50 claims sev gamma 50 cv 1 poisson')
    objs['BivariateAggregate'] = build(
        'bivariate MV 25 claims '
        'agg Wind dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2 '
        'agg Flood dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5 '
        'copula gumbel 0.4 poisson')
    objs['PnL'] = build('pnl Deal 1000 premium less agg Deal_e 1000 prem at 70% lr sev lognorm 100 cv 2 poisson')
    # the two backing analysis engines (a116 / a119). ``build('pnl ...
    # reinstatements/<feature> ...')`` now ALWAYS returns a ``PnL`` value object
    # (the always-PnL routing); the analysis is attached as ``pnl.analysis``.
    objs['ReinstatementAnalysis'] = build(
        'pnl RI.Human 10000 premium less agg RI.Human_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
        'occurrence net of 95% po 100 xs 100 rol 18% reinstatements '
        '1 free and 1 at 50% and 2 at 100% poisson').analysis
    objs['VariableRatingAnalysis'] = build(
        'pnl VR.Swing 10000 premium less agg VR.Swing_e 10000 prem at 85% lr sev lognorm 50 cv 3 poisson '
        'aggregate net of 5000 xs 4000 swing basic 500 lcm 0.5 min 500 max 3000').analysis
    objs['Severity'] = a.sevs[0]
    objs['Frequency'] = a.frequency
    try:
        objs['Bounds'] = Bounds(a, premium=a.agg_m * 1.1)
    except Exception as e:                      # pragma: no cover - defensive
        print(f'# WARNING: Bounds construction failed: {e}', file=sys.stderr)
    return objs


def member_kind(cls, name):
    """``property`` / ``method`` / ``classattr`` from the class MRO (no instance)."""
    for klass in cls.__mro__:
        if name in klass.__dict__:
            v = klass.__dict__[name]
            if isinstance(v, property):
                return 'property'
            if isinstance(v, (staticmethod, classmethod)) or callable(v):
                return 'method'
            return 'classattr'
    return None


def eval_returns(obj, name, kind):
    """For a zero-arg property, evaluate once and name the return-value kind."""
    if kind != 'property':
        return ''
    try:
        v = getattr(obj, name)
    except Exception:
        return '?'
    if isinstance(v, pd.DataFrame):
        return 'DataFrame'
    if isinstance(v, pd.Series):
        return 'Series'
    if isinstance(v, str):
        return 'str'
    if isinstance(v, bool):
        return 'bool'
    if isinstance(v, (int, float, np.integer, np.floating)):
        return 'scalar'
    if v is None:
        return 'none'
    return type(v).__name__


def first_doc_line(obj, name):
    try:
        member = getattr(type(obj), name, None)
        doc = inspect.getdoc(member) or ''
    except Exception:
        doc = ''
    return doc.strip().split('\n')[0].strip() if doc else ''


def introspect(objs) -> dict:
    """``{name: {classes:set, kind:str, returns:str, doc:str}}`` over all objects."""
    inv = {}
    for cname, obj in objs.items():
        cls = type(obj)
        names = set()
        for klass in cls.__mro__:
            if klass is object:
                continue
            names |= set(klass.__dict__)
        names |= set(vars(obj))
        for name in names:
            if name.startswith('_'):
                continue
            kind = member_kind(cls, name)
            if kind is None:
                kind = 'method' if callable(vars(obj).get(name)) else 'attribute'
            rec = inv.setdefault(name, {'classes': set(), 'kinds': [], 'returns': [], 'doc': ''})
            rec['classes'].add(cname)
            rec['kinds'].append(kind)
            r = eval_returns(obj, name, kind)
            if r:
                rec['returns'].append(r)
            if not rec['doc']:
                rec['doc'] = first_doc_line(obj, name)
    # collapse kinds/returns to the most common label
    for rec in inv.values():
        rec['kind'] = max(set(rec['kinds']), key=rec['kinds'].count)
        rec['ret'] = (max(set(rec['returns']), key=rec['returns'].count)
                      if rec['returns'] else '')
    return inv


def scipy_inherited() -> set:
    """Public member names Severity inherits from scipy (summarised in the CSV)."""
    return {n for n in dir(ss.rv_continuous) if not n.startswith('_')}


def load_csv() -> list:
    with open(CSV_PATH, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def audit(inv, rows, include_all) -> int:
    """Print drift report; return the number of **failures** (mismatch + stale).

    Undocumented members are reported but do not count as failures: the CSV is a
    curated subset, so a new internal attribute is not a defect.
    """
    csv_names = {r['name'] for r in rows}
    skip = set() if include_all else scipy_inherited()
    failures = 0

    print(f'FEATURES.csv: {len(rows)} rows; live inventory: {len(inv)} members\n')

    # MISMATCH (fail): presence cell disagrees with the live class
    print('## MISMATCH (cell vs live class) -- FAILS')
    n = 0
    for r in rows:
        name = r['name']
        if name not in inv:
            continue
        live = inv[name]['classes']
        for c in CLASS_COLS:
            marked = r[c].strip().startswith('Y')
            if marked != (c in live):
                print(f'  {name:<26} {c:<20} csv={r[c]!r:<6} live={c in live}')
                n += 1
    print(f'  ({n} mismatches)\n')
    failures += n

    # STALE (fail): CSV row whose name is on no class
    print('## STALE (in CSV, on no class) -- FAILS')
    n = 0
    for r in rows:
        if r['name'] not in inv:
            print(f"  {r['name']:<26} (group={r['group']})")
            n += 1
    print(f'  ({n} stale rows)\n')
    failures += n

    # UNDOCUMENTED (info): present on a class but not in the CSV
    print('## UNDOCUMENTED (in a class, missing from CSV) -- info only')
    caps, attrs, scipy_only = [], 0, 0
    for name, rec in sorted(inv.items()):
        if name in csv_names:
            continue
        if rec['classes'] == {'Severity'} and name in skip:
            scipy_only += 1
            continue
        if rec['kind'] in ('property', 'method'):
            caps.append((name, rec))
        else:
            attrs += 1
    print('  capabilities (property/method -- candidates for a row):')
    for name, rec in caps:
        marks = ' '.join(SHORT[c] if c in rec['classes'] else '-' for c in CLASS_COLS)
        print(f"    {name:<26} {rec['kind']:<9} {rec['ret']:<10} [{marks}]  {rec['doc'][:55]}")
    print(f'  ({len(caps)} undocumented capabilities, {attrs} undocumented '
          f'internal attributes', end='')
    if scipy_only and not include_all:
        print(f', + {scipy_only} scipy-inherited Severity members', end='')
    print(')\n')
    return failures


def dump_inventory(inv, include_all):
    """Grouped dump of the live inventory (curation aid for ``--inventory``)."""
    skip = set() if include_all else scipy_inherited()

    def bucket(name, rec):
        if name.endswith('_df') or rec['ret'] in ('DataFrame', 'Series'):
            return 'reins_df' if name.startswith('reins') else 'df'
        if rec['ret'] == 'str':
            return 'str'
        if rec['ret'] in ('scalar', 'bool'):
            return 'scalar'
        if rec['kind'] == 'method':
            return 'method'
        if rec['kind'] in ('attribute', 'classattr'):
            return 'attr'
        return 'other'

    groups = {}
    for name, rec in inv.items():
        if rec['classes'] == {'Severity'} and name in skip:
            continue
        groups.setdefault(bucket(name, rec), []).append((name, rec))
    print('marks:', ' '.join(SHORT[c] for c in CLASS_COLS), '\n')
    for g in ['df', 'reins_df', 'str', 'scalar', 'method', 'attr', 'other']:
        items = sorted(groups.get(g, []))
        print(f'\n#### {g} ({len(items)})')
        for name, rec in items:
            marks = ' '.join(SHORT[c] if c in rec['classes'] else '-' for c in CLASS_COLS)
            print(f"  {name:<26} {rec['kind']:<9} {rec['ret']:<10} [{marks}]  {rec['doc'][:60]}")


def main(argv):
    include_all = '--all' in argv
    objs = build_objects()
    inv = introspect(objs)
    if '--inventory' in argv:
        dump_inventory(inv, include_all)
        return 0
    failures = audit(inv, load_csv(), include_all)
    if failures:
        print(f'DRIFT: {failures} mismatch/stale item(s) contradict the code -- '
              f'fix dev/FEATURES.csv.')
        return 1
    print('OK: every documented row in dev/FEATURES.csv matches the live class '
          'surfaces (see UNDOCUMENTED above for optional additions).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
