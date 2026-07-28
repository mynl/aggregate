"""Regenerate this repo's cookbook pages from ``library.agg``.

A thin caller. The generator itself is :mod:`aggregate.cookbook`, part of the
package since 1.0.0a167, because rendering a library is a capability any DecL
library should have and not a chore private to this repo. All this script adds
is *which* library and *where* the pages go::

    .venv/Scripts/python.exe dev/generate_cookbook.py           # write
    .venv/Scripts/python.exe dev/generate_cookbook.py --check   # CI: stale?

Equivalent, without the repo defaults::

    python -m aggregate.cookbook docs/cookbook [--check]

``tests/test_cookbook_generate.py`` guards the committed pages against drift,
so a doc edited without re-running this is caught by the suite.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from aggregate.cookbook import main                       # noqa: E402

#: Where this repo's cookbook lives.
COOKBOOK = REPO_ROOT / 'docs' / 'cookbook'

if __name__ == '__main__':
    raise SystemExit(main([str(COOKBOOK), *sys.argv[1:]]))
