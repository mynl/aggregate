"""Capture greater_tables canonical_dict snapshots for the exhibit tests.

Writes ``tests/data/exhibit_snapshots.json``: for each
(exhibit, perspective, kind) case, the list of per block
``canonical_dict(doc, include_hash=False)`` forms.
``tests/test_exhibits.py::test_canonical_snapshot`` compares live builds
against this file, pinning both the business translation (captions, flags,
drops) and the greater_tables IR emission.

Usage
-----
    uv run python tests/capture_exhibit_snapshots.py

Re-run (and commit the diff) whenever an exhibit's presentation
deliberately changes; an unexplained diff is a regression.
"""
from __future__ import annotations

import json
from pathlib import Path

import greater_tables as gt

from aggregate import build
from aggregate.exhibits import available_exhibits, build_exhibit

SNAPSHOT_PATH = Path(__file__).parent / 'data' / 'exhibit_snapshots.json'

# DecL programs. Mirrors tests/test_exhibits.py EXACTLY.
PROGRAMS = {
    'Aggregate': 'agg EX.Dice dfreq [3] dsev [1:6]',
    'Portfolio': ('port EX.Port '
                  'agg EX.UnitA as "Unit Alpha" 1 claim dsev [1 2 3] fixed '
                  'agg EX.UnitB 2 claims dsev [2 4] fixed'),
    'BivariateAggregate': ('bv EX.BV dfreq [1 2] [.7 .3] dbvsev '
                           '[[10 1 .4] [10 2 .1] [20 1 .2] [20 2 .1] '
                           '[20 3 .2]]'),
    'PnL': ('pnl EX.B 1000 premium less agg EX.Be 850 loss '
            'sev lognorm 100 cv 1 poisson'),
    'Distortion': 'dist EX.PH ph 0.5',
    # ceding fixtures for the reins exhibit ([Exhibits-Reins-Insurer])
    'ReinsAggregate': ('agg EX.Re dfreq [1 2] dsev [10 20 30] '
                       'occurrence net of 10 xs 10'),
    'ReinsPortfolio': ('port EX.RePort '
                       'agg EX.ReA dfreq [1 2] dsev [10 20 30] '
                       'occurrence net of 10 xs 10 '
                       'agg EX.ReB 1 claim dsev [5 10] fixed'),
}


def main():
    snapshots = {}
    for kind, program in PROGRAMS.items():
        obj = build(program)
        for name, perspectives in available_exhibits(obj):
            for perspective in perspectives:
                e = build_exhibit(obj, name, perspective)
                key = f'{name}/{perspective.value}/{kind}'
                snapshots[key] = [gt.canonical_dict(doc, include_hash=False)
                                  for doc in e.ir_blocks]
    SNAPSHOT_PATH.write_text(
        json.dumps(snapshots, sort_keys=True, indent=1) + '\n',
        encoding='utf-8', newline='\n')
    print(f'wrote {len(snapshots)} snapshots to {SNAPSHOT_PATH}')


if __name__ == '__main__':
    main()
