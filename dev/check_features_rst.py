#!/usr/bin/env python
"""Execute the ``.. ipython:: python`` blocks of an RST page, in order, in one namespace.

This is the fast done-gate for ``dev/task-features.md``. It replaces the
``jupytext --to ipynb --execute`` gate the page had while it was a Quarto
notebook: same guarantee, that every code block runs clean against the current
API, without standing up a Sphinx build (the doc tree is 500+ pages and the
build is slow, so it stays outside the edit loop; see ``CLAUDE.md``).

What it does *not* check is how the page renders. Directive syntax, cross
references, the csv-table and the ``@savefig`` placement are only exercised by a
real build, which the author runs separately.

Usage::

    .venv/Scripts/python.exe dev/check_features_rst.py
    .venv/Scripts/python.exe dev/check_features_rst.py docs/some/other.rst

Exit status is 0 when every block ran, 1 on the first failure, whose block
number, source line and traceback are printed.
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / 'docs' / '2_aggregate_overview' / 'features.rst'

RE_DIRECTIVE = re.compile(r'^\s*\.\. ipython:: python\s*$')


def blocks(text: str) -> list[tuple[int, str]]:
    """Return ``(first_source_line, source)`` for each ipython block.

    A block runs from the directive line to the first non-blank line indented
    less than its own body. ``@savefig`` lines are directives to the Sphinx
    extension, not Python, so they are dropped.
    """
    out: list[tuple[int, str]] = []
    lines = text.split('\n')
    i, n = 0, len(lines)
    while i < n:
        if not RE_DIRECTIVE.match(lines[i]):
            i += 1
            continue
        start, i = i + 1, i + 1
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        indent = len(lines[i]) - len(lines[i].lstrip())
        body = []
        while i < n:
            if not lines[i].strip():
                body.append('')
                i += 1
                continue
            if len(lines[i]) - len(lines[i].lstrip()) < indent:
                break
            body.append(lines[i][indent:])
            i += 1
        body = [b for b in body if not b.lstrip().startswith('@savefig')]
        out.append((start + 1, '\n'.join(body).rstrip() + '\n'))
    return out


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    import matplotlib
    matplotlib.use('Agg')                      # no display, no figure windows

    bs = blocks(path.read_text(encoding='utf-8'))
    print(f'{path}: {len(bs)} ipython blocks')
    ns: dict = {'__name__': '__main__'}
    for k, (lineno, src) in enumerate(bs, start=1):
        try:
            exec(compile(src, f'{path.name}:{lineno}', 'exec'), ns)
        except Exception:
            print(f'\nFAILED block {k} of {len(bs)}, {path.name} line {lineno}\n')
            print(src)
            traceback.print_exc()
            return 1
    print(f'all {len(bs)} blocks ran clean')
    return 0


if __name__ == '__main__':
    sys.exit(main())
