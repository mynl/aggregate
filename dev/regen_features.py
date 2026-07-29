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
internal attribute -- so only three kinds of drift are treated as **failures**
(non-zero exit); two more are purely informational:

* **MISMATCH** (fail) -- a presence cell disagrees with the live class. This is
  the real guard: it catches a rename, a moved method, or a member that gained /
  lost a class.
* **STALE** (fail) -- a CSV row whose ``name`` is on no class (a deleted member
  or a typo).
* **KIND** (fail) -- the ``kind`` column disagrees with the live descriptor
  (``property`` / ``method`` / ``attribute``).
* **FCC CONTRACT** (fail) -- a class in ``constants.FIRST_CLASS_CLASSES`` is
  missing a member of ``constants.FCC_REQUIRED``, or is still excused for one it
  now has. Unlike the three above this does not read the CSV at all: the CSV
  documents what *is*, the contract states what must *be*. Paired with the
  narrative check, which requires ``*_description`` and ``*_explanation`` to
  appear together. See :func:`fcc_contract`.
* **COLLISION** (info only) -- cells marked ``~``: the name exists on that class
  but is *not* the documented capability (see the cell vocabulary below). Listed
  so the deliberate collisions stay visible rather than hiding behind a ``Y``.
* **UNDOCUMENTED** (info only) -- a public member absent from the CSV, split into
  *capabilities* (property / method) and *attributes*, both listed by name.
  Severity members inherited from ``scipy.stats.rv_continuous`` -- class members
  *and* the instance attributes ``rv_continuous.__init__`` sets -- are summarised
  in the CSV, so they are folded into one count unless ``--all`` is passed.

Cell vocabulary (see the ``# legend`` row in the CSV; all three count as
"present" for the MISMATCH check):

===== =========================================================================
``Y``  the capability, as described
``Y*`` same concept, different shape / semantics -- see the row's ``notes``
``~``  name collision: the name exists but is *not* this capability. The
       canonical case is ``Distortion.tvar`` / ``.mean`` / ``.max``, which are
       static *constructors* of a distortion, not risk measures.
===== =========================================================================

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
              'Severity', 'Frequency', 'GridDistribution', 'Distortion',
              'Bounds', 'AllocationBounds', 'PricingBounds']
SHORT = {'Aggregate': 'Agg', 'Portfolio': 'Port', 'BivariateAggregate': 'Biv',
         'PnL': 'PnL', 'Severity': 'Sev',
         'Frequency': 'Freq', 'GridDistribution': 'GD', 'Distortion': 'Dist',
         'Bounds': 'Bnd',
         'AllocationBounds': 'AllB', 'PricingBounds': 'PrcB'}

#: Cell tokens that mean "the name is present on this class". ``~`` additionally
#: means "present but *not* this capability" -- reported in its own section.
PRESENT_TOKENS = ('Y', '~')
COLLISION_TOKEN = '~'


def build_objects() -> dict:
    """One live object per class column (built + updated where relevant)."""
    from aggregate import build
    from aggregate.bounds import Bounds
    from aggregate.spectral import Distortion

    objs = {}
    a = build('agg E 100 claims sev lognorm 100 cv 2 '
              'occurrence net of 50 xs 50 poisson')
    objs['Aggregate'] = a
    port = build(
        'port P agg A 100 claims sev lognorm 100 cv 2 poisson '
        'agg B 50 claims sev gamma 50 cv 1 poisson')
    objs['Portfolio'] = port
    objs['BivariateAggregate'] = build(
        'bivariate MV 25 claims '
        'agg Wind dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2 '
        'agg Flood dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5 '
        'copula gumbel 0.4 poisson')
    objs['PnL'] = build('pnl Deal 1000 premium less agg Deal_e 1000 prem at 70% lr sev lognorm 100 cv 2 poisson')
    objs['Severity'] = a.sevs[0]
    objs['Frequency'] = a.frequency
    # The canonical quantile engine every q / tvar / cdf / sf routes through, and
    # the return type of PnL.result -- built off the Aggregate rather than
    # hand-rolled so it carries a real grid.
    objs['GridDistribution'] = a._grid_distribution()
    # A plain proportional-hazard distortion: enough to expose the whole
    # Distortion surface (LabeledMixin, info / help / plot, the lazy frames).
    objs['Distortion'] = Distortion('ph', 0.5)
    try:
        objs['Bounds'] = Bounds(a, premium=a.actual_m * 1.1)
    except Exception as e:                      # pragma: no cover - defensive
        print(f'# WARNING: Bounds construction failed: {e}', file=sys.stderr)
    # AllocationBounds / PricingBounds share the _HullEngine slice geometry but
    # are distinct classes (not Bounds subclasses); build both off the Portfolio
    # total, capped at p=0.99 so the deep-tail vertices stay well-conditioned.
    try:
        objs['AllocationBounds'] = port.allocation_bounds(p=0.99)
    except Exception as e:                      # pragma: no cover - defensive
        print(f'# WARNING: AllocationBounds construction failed: {e}', file=sys.stderr)
    try:
        objs['PricingBounds'] = port.pricing_bounds(port.agg_list[0], p=0.99)
    except Exception as e:                      # pragma: no cover - defensive
        print(f'# WARNING: PricingBounds construction failed: {e}', file=sys.stderr)
    return objs


def member_kind(cls, name):
    """``property`` / ``method`` / ``classattr`` from the class MRO (no instance).

    Notes
    -----
    Any **non-data descriptor that is not callable** counts as a ``property``.
    That is what catches :class:`functools.cached_property`, which is neither a
    ``property`` instance nor ``callable`` -- before this test it fell through to
    ``classattr``, mislabelling ``freq_df``, ``Distortion.info`` / ``summary_df``
    / ``stats_df`` / ``density_df``, and the whole lazy ``Bounds`` surface
    (``cloud_df``, ``tvar_df``, ``weight_df``, the envelopes, ``p_knots``,
    ``s_grid``, ``tvar_hinges``, ``tvar_x_p``). From the caller's side a
    ``cached_property`` *is* a property -- ``obj.x``, no parentheses -- so that is
    what the CSV documents and what this reports.
    """
    for klass in cls.__mro__:
        if name in klass.__dict__:
            v = klass.__dict__[name]
            if isinstance(v, property):
                return 'property'
            if isinstance(v, (staticmethod, classmethod)) or callable(v):
                return 'method'
            if hasattr(type(v), '__get__'):      # cached_property & friends
                return 'property'
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
    """Public member names Severity inherits from scipy (summarised in the CSV).

    Notes
    -----
    Two sources, and both are needed. ``dir(rv_continuous)`` gives the *class*
    surface (``rvs``, ``moment``, ``entropy``, ...). But ``rv_continuous.__init__``
    also sets public **instance** attributes -- ``generic_moment``,
    ``moment_type``, ``vecentropy``, ``xtol``, ``badvalue``, ``numargs``,
    ``shapes`` -- which the introspector picks up off ``vars(obj)`` and which
    used to surface as "undocumented capabilities" of :class:`Severity`. They are
    scipy's, not this library's, so they belong in the same summary.
    """
    names = {n for n in dir(ss.rv_continuous) if not n.startswith('_')}
    try:
        bare = ss.rv_continuous()
        names |= {n for n in vars(bare) if not n.startswith('_')}
    except Exception:                           # pragma: no cover - defensive
        pass
    return names


def load_csv() -> list:
    with open(CSV_PATH, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    # Skip metadata/comment rows (group starting with '#'), e.g. the
    # ``# table-version`` stamp -- they are not class members and must not be
    # audited as one.
    return [r for r in rows if not (r.get('group') or '').startswith('#')]


def read_version_stamp() -> str | None:
    """The ``__version__`` the CSV was last stamped against (the ``# table-version`` row)."""
    with open(CSV_PATH, newline='', encoding='utf-8') as fh:
        for row in csv.reader(fh):
            if row and row[0].startswith('#') and len(row) > 1:
                return row[1].strip()
    return None


def is_present(cell: str) -> bool:
    """Whether a presence cell claims the member exists on that class.

    ``Y``, ``Y*`` and ``~`` all mean *present*; only the reason differs (see the
    cell vocabulary in the module docstring). Anything else -- blank, a stray
    note -- means absent.
    """
    cell = cell.strip()
    return cell.startswith('Y') or cell.startswith(COLLISION_TOKEN)


def fcc_contract(inv) -> int:
    """Check the declared FCC contract against the live classes; return failures.

    Notes
    -----
    Two checks, both driven by the contract declared in
    :mod:`aggregate.constants` (``FIRST_CLASS_CLASSES`` / ``FCC_REQUIRED``), so
    this script and ``tests/test_fcc_surface.py`` cannot drift apart:

    **REQUIRED** every member of ``FCC_REQUIRED`` resolves on every class in
    ``FIRST_CLASS_CLASSES``, less the declared ``FCC_CONTRACT_EXCEPTIONS``. A
    *stale* exception (the member is now there, but the class is still excused)
    fails too: that is what stops the excuse list outliving the hole.

    **PAIRS** the optional narrative surface comes in halves. Where a class
    carries ``<stem>_description`` it must carry ``<stem>_explanation``, and the
    other way round, less the stems in ``FCC_UNPAIRED_NARRATIVES``.
    """
    from aggregate.constants import (FIRST_CLASS_CLASSES, FCC_REQUIRED,
                                     FCC_CONTRACT_EXCEPTIONS,
                                     FCC_UNPAIRED_NARRATIVES)
    failures = 0

    print('## FCC CONTRACT (constants.FCC_REQUIRED vs live class) -- FAILS')
    n = 0
    for cname in FIRST_CLASS_CLASSES:
        excused = set(FCC_CONTRACT_EXCEPTIONS.get(cname, ()))
        for member in FCC_REQUIRED:
            present = cname in inv.get(member, {}).get('classes', set())
            if not present and member not in excused:
                print(f'  {cname:<20} MISSING {member}')
                n += 1
            elif present and member in excused:
                print(f'  {cname:<20} {member} is present -- drop it from '
                      f'FCC_CONTRACT_EXCEPTIONS')
                n += 1
    excused_total = sum(len(v) for v in FCC_CONTRACT_EXCEPTIONS.values())
    print(f'  ({n} contract failures; {excused_total} declared exception(s) '
          f'outstanding, must be 0 by 1.0.0b1)\n')
    failures += n

    # PAIRS: the optional narrative surface, checked for symmetry rather than
    # presence. ``sorted`` so the report is stable run to run.
    print('## NARRATIVE PAIRS (*_description <-> *_explanation) -- FAILS')
    n = 0
    for name in sorted(inv):
        for suffix, partner_suffix in (('_description', '_explanation'),
                                       ('_explanation', '_description')):
            if not name.endswith(suffix):
                continue
            stem = name[:-len(suffix)]
            if stem in FCC_UNPAIRED_NARRATIVES:
                continue
            partner = f'{stem}{partner_suffix}'
            missing = inv[name]['classes'] - inv.get(partner, {}).get('classes', set())
            for cname in sorted(missing):
                print(f'  {cname:<20} has {name} but not {partner}')
                n += 1
    print(f'  ({n} unpaired; {len(FCC_UNPAIRED_NARRATIVES)} stem(s) excused: '
          f'{", ".join(FCC_UNPAIRED_NARRATIVES)})\n')
    failures += n
    return failures


def audit(inv, rows, include_all) -> int:
    """Print drift report; return the number of **failures**.

    Failures are MISMATCH + STALE + KIND + the FCC contract checks. Collisions
    and undocumented members are reported but do not count: the CSV is a curated
    subset, so a new internal attribute is not a defect, and a ``~`` cell is a
    deliberate annotation.
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
            cell = r.get(c) or ''      # .get: a not-yet-widened CSV reports, not crashes
            if is_present(cell) != (c in live):
                print(f'  {name:<26} {c:<20} csv={cell!r:<6} live={c in live}')
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

    # KIND (fail): the documented descriptor kind disagrees with the live one.
    # ``attribute`` and ``classattr`` are the same thing to a reader (plain value,
    # no call, no descriptor), so they are accepted for each other.
    print('## KIND (csv kind vs live descriptor) -- FAILS')
    n = 0
    plain = {'attribute', 'classattr'}
    for r in rows:
        name, csv_kind = r['name'], (r['kind'] or '').strip()
        if name not in inv or not csv_kind:
            continue
        live_kind = inv[name]['kind']
        if csv_kind == live_kind or (csv_kind in plain and live_kind in plain):
            continue
        # A member that is a property on one class and an attribute on another
        # (``stats_df``) is documented by its majority kind; the row's notes carry
        # the exception. Only flag when no class agrees with the CSV.
        if csv_kind in inv[name]['kinds']:
            continue
        print(f"  {name:<26} csv={csv_kind:<10} live={live_kind:<10} "
              f"(seen: {sorted(set(inv[name]['kinds']))})")
        n += 1
    print(f'  ({n} kind mismatches)\n')
    failures += n

    # FCC CONTRACT (fail): the declared surface, checked against the live
    # classes. Independent of the CSV -- the CSV documents what IS, the contract
    # states what MUST BE.
    failures += fcc_contract(inv)

    # COLLISION (info): ``~`` cells -- the name is there but is not the capability
    print('## COLLISION (~ cells: name present, NOT this capability) -- info only')
    n = 0
    for r in rows:
        hits = [c for c in CLASS_COLS
                if (r.get(c) or '').strip().startswith(COLLISION_TOKEN)]
        if hits:
            print(f"  {r['name']:<26} {', '.join(hits)}")
            n += 1
    print(f'  ({n} rows carry a collision marker)\n')

    # UNDOCUMENTED (info): present on a class but not in the CSV
    print('## UNDOCUMENTED (in a class, missing from CSV) -- info only')
    caps, attrs, scipy_only = [], [], 0
    for name, rec in sorted(inv.items()):
        if name in csv_names:
            continue
        if rec['classes'] == {'Severity'} and name in skip:
            scipy_only += 1
            continue
        (caps if rec['kind'] in ('property', 'method') else attrs).append((name, rec))

    def show(items):
        for name, rec in items:
            marks = ' '.join(SHORT[c] if c in rec['classes'] else '-'
                             for c in CLASS_COLS)
            print(f"    {name:<26} {rec['kind']:<9} {rec['ret']:<10} "
                  f"[{marks}]  {rec['doc'][:55]}")

    print('  capabilities (property/method -- candidates for a row):')
    show(caps)
    print('  attributes (the long tail the curation may deliberately omit):')
    show(attrs)
    print(f'  ({len(caps)} undocumented capabilities, {len(attrs)} undocumented '
          f'attributes', end='')
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
    # Docstrings in this library carry maths (the Distortion surface has an
    # integral sign), and a Windows console is cp1252 -- print what we can rather
    # than dying halfway through the report.
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:                               # pragma: no cover - defensive
        pass
    include_all = '--all' in argv
    objs = build_objects()
    inv = introspect(objs)
    if '--inventory' in argv:
        dump_inventory(inv, include_all)
        return 0
    try:
        from aggregate import __version__ as live_version
    except Exception:                               # pragma: no cover - defensive
        live_version = '?'
    stamp = read_version_stamp()
    print(f'# table-version stamp: {stamp}  |  live aggregate: {live_version}')
    if stamp and stamp != live_version:
        print(f"# NOTE: stamp {stamp} != live {live_version} -- bump the "
              f"'# table-version' row if a public surface changed this release.")
    print()
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
