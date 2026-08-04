"""Guardrails for the plotting subsystem's single matplotlib boundary.

The plotting refactor (``dev/plan-plots-subsystem.md``) centralises every
matplotlib use under :mod:`aggregate.plots`. Two invariants protect that:

1. **Lazy load.** ``import aggregate`` must not import matplotlib. Plotting
   pulls it in on first use via the function-local imports in the class stubs.

2. **One boundary.** No module outside ``aggregate.plots`` imports matplotlib
   at top level -- with a single documented exemption: ``pedagogy.py``, the
   non-core paper/blog figure-generator module, which is never imported on the
   ``import aggregate`` path and is a sanctioned figures module.

The exhibits module ([Exhibits-Module]) adds a second lazy boundary:
``import aggregate`` must not import greater_tables, and
``import aggregate.exhibits`` must import neither matplotlib nor
greater_tables (the IR conversion step imports greater_tables lazily).
"""

import ast
import subprocess
import sys
from pathlib import Path

import aggregate

# Source root of the installed/editable package.
_PKG_ROOT = Path(aggregate.__file__).resolve().parent

# Modules allowed to import matplotlib at top level: everything under
# ``plots/`` (the boundary itself) plus the documented ``pedagogy`` exemption.
_ALLOWED = {'pedagogy.py'}


def _imports_matplotlib_toplevel(path: Path) -> bool:
    """True if the module imports matplotlib at module scope."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in tree.body:  # module body only -> top level
        if isinstance(node, ast.Import):
            if any(a.name == 'matplotlib' or a.name.startswith('matplotlib.')
                   for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == 'matplotlib'
                                or node.module.startswith('matplotlib.')):
                return True
    return False


def test_import_aggregate_does_not_load_matplotlib():
    """A fresh ``import aggregate`` must leave matplotlib unloaded."""
    code = (
        "import sys, aggregate; "
        "assert 'matplotlib' not in sys.modules, "
        "'import aggregate pulled in matplotlib'"
    )
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_import_aggregate_does_not_load_greater_tables():
    """A fresh ``import aggregate`` must leave greater_tables unloaded."""
    code = (
        "import sys, aggregate; "
        "assert 'greater_tables' not in sys.modules, "
        "'import aggregate pulled in greater_tables'"
    )
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_import_exhibits_stays_lazy():
    """``import aggregate.exhibits`` loads neither matplotlib nor greater_tables."""
    code = (
        "import sys, aggregate.exhibits; "
        "assert 'matplotlib' not in sys.modules, "
        "'aggregate.exhibits pulled in matplotlib'; "
        "assert 'greater_tables' not in sys.modules, "
        "'aggregate.exhibits pulled in greater_tables'"
    )
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_toplevel_matplotlib_outside_plots():
    """Only ``plots/`` and the exempt modules import matplotlib at top level."""
    offenders = []
    for path in _PKG_ROOT.rglob('*.py'):
        rel = path.relative_to(_PKG_ROOT)
        if rel.parts and rel.parts[0] == 'plots':
            continue  # the boundary itself
        if path.name in _ALLOWED:
            continue  # documented exemption
        if _imports_matplotlib_toplevel(path):
            offenders.append(str(rel))
    assert not offenders, (
        'modules importing matplotlib at top level outside plots/: '
        + ', '.join(sorted(offenders))
    )
