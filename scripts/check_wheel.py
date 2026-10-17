"""Build the wheel, install it into a throwaway venv, and smoke test it.

``[Wheel-Smoke-Check]``. A packaging defect is invisible to pytest by
construction: the suite runs against the editable install, where every module
is on disk, so a wheel that leaves a subpackage behind passes every test. That
is how the 1.0.0 wheel shipped without ``aggregate.charts``, ``.plots`` and
``.exhibits``, and ``a.plot()`` raised ``ModuleNotFoundError`` for anyone who
installed it. This script checks the artifact itself, in three stages.

1. **Build.** ``uv build --wheel`` into a temporary directory, or take an
   existing wheel with ``--wheel``.
2. **Listing.** Every ``src/aggregate/**/__init__.py`` has a counterpart in the
   wheel, and so does every other module and every file the
   ``[tool.setuptools.package-data]`` globs match. A missing data file (the
   grammar, a ``.agg`` library) breaks ``build`` just as a missing subpackage
   breaks ``plot``, so both are listed.
3. **Smoke.** A fresh venv, the wheel installed into it with its declared
   dependencies, and a script run there in isolated mode (``python -I``) from a
   directory outside the repository. Its first check is that ``aggregate`` was
   imported from the venv and not from ``src``: a smoke test that silently
   imports the source tree passes for exactly the reason the suite does. Then
   the version, ``build`` on a small program with a known mean, ``qd``,
   ``.plot()`` on the Agg backend, an import of every subpackage the source
   tree has, and a ``library.agg`` recipe.

Every check is reported, pass or fail, so one failure does not hide the next,
and the exit code is nonzero if any failed. A release-time command like the
numerics gate, deliberately **not** part of the test suite, since it needs a
built artifact and a network for the dependency install.

Usage
-----
From the repository root::

    uv run python scripts/check_wheel.py                 # build, then check
    uv run python scripts/check_wheel.py --wheel dist/aggregate-1.5.2-py3-none-any.whl
    uv run python scripts/check_wheel.py --keep          # keep the temp dir

Only the standard library is used, plus ``uv`` on the path for the build, the
venv and the install.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
PACKAGE = 'aggregate'

#: A small program whose mean is known exactly: three dice, mean 10.5.
SMOKE_PROGRAM = 'agg Dice dfreq [3] dsev [1:6]'
SMOKE_MEAN = 10.5
#: A shipped ``library.agg`` entry, built by name, which needs the packaged
#: library file and the grammar.
SMOKE_RECIPE = 'BasicBook'

# Runs inside the throwaway venv. Prints one JSON object per check so the
# parent can report each one, and never raises past a single check.
SMOKE_SCRIPT = r'''
import json, sys, traceback
from pathlib import Path

venv_prefix, version, program, mean, recipe, subpackages = sys.argv[1:7]
results = []

def check(name, fn):
    try:
        detail = fn()
        results.append({'check': name, 'ok': True, 'detail': detail or ''})
    except Exception:
        results.append({'check': name, 'ok': False,
                        'detail': traceback.format_exc(limit=3).strip()})

def imported_from_venv():
    import aggregate
    where = Path(aggregate.__file__).resolve()
    prefix = Path(venv_prefix).resolve()
    if prefix not in where.parents:
        raise AssertionError(f'aggregate imported from {where}, not from the venv {prefix}')
    return str(where.parent)

def version_matches():
    import aggregate
    if aggregate.__version__ != version:
        raise AssertionError(f'installed {aggregate.__version__}, expected {version}')
    return aggregate.__version__

state = {}

def builds():
    from aggregate import build
    a = build(program)
    if abs(a.est_m - float(mean)) > 1e-9:
        raise AssertionError(f'mean {a.est_m}, expected {mean}')
    state['a'] = a
    return f'{program}: mean {a.est_m}'

def quick_display():
    from aggregate import qd
    qd(state['a'])

def plots():
    import matplotlib
    matplotlib.use('Agg')
    state['a'].plot()
    import matplotlib.pyplot as plt
    plt.close('all')

def subpackages_import():
    import importlib
    names = [s for s in subpackages.split(',') if s]
    for name in names:
        importlib.import_module(name)
    return ', '.join(names)

def library_recipe():
    from aggregate import build
    r = build(recipe)
    return f'{recipe}: mean {r.est_m:.6g}'

check('imported from the venv, not src', imported_from_venv)
check('version', version_matches)
check('build a program', builds)
if 'a' in state:
    check('qd', quick_display)
    check('plot', plots)
check('import every subpackage', subpackages_import)
check('build a library.agg recipe', library_recipe)
print(json.dumps(results))
'''


def source_version() -> str:
    """The version literal in ``src/aggregate/__init__.py``, read without importing."""
    text = (SRC / PACKAGE / '__init__.py').read_text(encoding='utf-8')
    match = re.search(r'''^__version__\s*=\s*['"]([^'"]+)['"]''', text, re.M)
    if not match:
        raise SystemExit('could not find __version__ in src/aggregate/__init__.py')
    return match.group(1)


def source_subpackages() -> list[str]:
    """Dotted names of every package under ``src/aggregate`` (each ``__init__.py``)."""
    return sorted('.'.join(p.parent.relative_to(SRC).parts)
                  for p in (SRC / PACKAGE).rglob('__init__.py'))


def expected_files() -> dict[str, str]:
    """Wheel paths that must be present, mapped to the reason each is expected.

    Notes
    -----
    Three sources, all read from the source tree so nothing here goes stale
    when a module or data file is added: every ``__init__.py`` (the check the
    TODO entry asks for), every other ``.py`` module, and every file matched
    by the ``[tool.setuptools.package-data]`` globs in ``pyproject.toml``.
    ``__pycache__`` is skipped.
    """
    expected = {}
    pkg_root = SRC / PACKAGE
    for path in pkg_root.rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        rel = path.relative_to(SRC).as_posix()
        expected[rel] = 'package' if path.name == '__init__.py' else 'module'
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    globs = (config.get('tool', {}).get('setuptools', {})
             .get('package-data', {}).get(PACKAGE, []))
    for path in pkg_root.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        inner = path.relative_to(pkg_root).as_posix()
        if any(fnmatch(inner, pattern) for pattern in globs):
            expected[path.relative_to(SRC).as_posix()] = 'package-data'
    return expected


def check_listing(wheel: Path) -> list[dict]:
    """Compare the wheel's contents with :func:`expected_files`."""
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    results = []
    expected = expected_files()
    for kind in ('package', 'module', 'package-data'):
        wanted = sorted(p for p, k in expected.items() if k == kind)
        missing = [p for p in wanted if p not in names]
        results.append({
            'check': f'listing: every {kind} file is in the wheel',
            'ok': not missing,
            'detail': (f'{len(wanted)} expected, all present' if not missing
                       else 'missing: ' + ', '.join(missing)),
        })
    return results


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, echoing it, and raise with its output if it fails."""
    print('  $ ' + ' '.join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise RuntimeError(f'command failed ({proc.returncode}):\n'
                           f'{proc.stdout}\n{proc.stderr}')
    return proc


def build_wheel(out_dir: Path) -> Path:
    """``uv build --wheel`` from the repository root into ``out_dir``."""
    run(['uv', 'build', '--wheel', '--out-dir', str(out_dir)], cwd=ROOT)
    wheels = sorted(out_dir.glob(f'{PACKAGE}-*.whl'))
    if len(wheels) != 1:
        raise RuntimeError(f'expected one wheel in {out_dir}, found {wheels}')
    return wheels[0]


def smoke(wheel: Path, work: Path, python: str) -> list[dict]:
    """Install ``wheel`` into a fresh venv under ``work`` and run the smoke script there."""
    venv = work / 'venv'
    run(['uv', 'venv', '--python', python, str(venv)])
    vpython = venv / ('Scripts' if os.name == 'nt' else 'bin') / 'python'
    run(['uv', 'pip', 'install', '--python', str(vpython), str(wheel)])
    script = work / 'smoke.py'
    script.write_text(SMOKE_SCRIPT, encoding='utf-8')
    outside = work / 'run'
    outside.mkdir()
    env = {k: v for k, v in os.environ.items()
           if k not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}
    env['MPLBACKEND'] = 'Agg'
    proc = subprocess.run(
        [str(vpython), '-I', str(script), str(venv), source_version(),
         SMOKE_PROGRAM, str(SMOKE_MEAN), SMOKE_RECIPE,
         ','.join(source_subpackages())],
        cwd=outside, env=env, capture_output=True, text=True)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith('[')]
    if not lines:
        return [{'check': 'smoke script ran', 'ok': False,
                 'detail': (proc.stdout + proc.stderr).strip()[-2000:]}]
    return json.loads(lines[-1])


def main(argv=None) -> int:
    """Parse arguments, run the three stages, print the report; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--wheel', type=Path,
                        help='check this wheel instead of building one')
    parser.add_argument('--python', default=sys.executable,
                        help='interpreter for the throwaway venv '
                             '(default: the one running this script)')
    parser.add_argument('--keep', action='store_true',
                        help='keep the temporary directory and print its path')
    args = parser.parse_args(argv)
    if shutil.which('uv') is None:
        print('uv is not on the path', file=sys.stderr)
        return 2

    work = Path(tempfile.mkdtemp(prefix='aggregate-wheel-'))
    results = []
    try:
        if args.wheel:
            wheel = args.wheel.resolve()
            print(f'checking {wheel.name}')
        else:
            print('building the wheel')
            wheel = build_wheel(work / 'dist')
            print(f'built {wheel.name}')
        version = source_version()
        results.append({'check': 'wheel version matches src',
                        'ok': f'-{version}-' in wheel.name,
                        'detail': f'{wheel.name} against {version}'})
        results += check_listing(wheel)
        print('installing into a throwaway venv')
        results += smoke(wheel, work, args.python)
    except RuntimeError as e:
        results.append({'check': 'build or install', 'ok': False, 'detail': str(e)})
    finally:
        if args.keep:
            print(f'kept {work}')
        else:
            shutil.rmtree(work, ignore_errors=True)

    print()
    width = max(len(r['check']) for r in results)
    for r in results:
        mark = 'PASS' if r['ok'] else 'FAIL'
        print(f'{mark}  {r["check"]:<{width}}  {r["detail"]}'.rstrip())
    failed = [r for r in results if not r['ok']]
    print(f'\n{len(results) - len(failed)} passed, {len(failed)} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
