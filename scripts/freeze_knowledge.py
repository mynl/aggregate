"""Freeze and verify ``aggregate`` knowledge-base outputs (regression harness).

A standalone, dependency-free maintenance tool. It walks every ``agg`` / ``port``
program in a chosen knowledge base, builds and updates each object with default
parameters, and snapshots two frames per object to parquet:

* ``describe``  -- the moment / validation table.
* ``density_df`` filtered to ``filter(regex='p_|exeqa_')`` -- the per-bucket
  probability and conditional-expectation columns.

A companion step rebuilds the same objects later and checks the freshly computed
frames still match the frozen parquet to a tight absolute tolerance. This catches
numerical drift in the FFT / allocation machinery across refactors: freeze
before, check after.

Usage
-----
As importable functions (the real API -- ideal from Jupyter, where ``aggregate``
is already loaded so there is no import overhead)::

    from freeze_knowledge import freeze_knowledge, check_knowledge
    manifest = freeze_knowledge(root=r"T:\\tmp\\aggregate_freeze")
    failures = check_knowledge(manifest_dir)          # [] == all match

As a CLI (stdlib argparse, no third-party dependencies)::

    python scripts/freeze_knowledge.py freeze [-d NAME ...] [--root DIR] [--atol F]
    python scripts/freeze_knowledge.py check DIRECTORY [--atol F]

The ``check`` subcommand exits non-zero if any object drifts beyond tolerance.

Layout written by ``freeze_knowledge``::

    <root>/<YYYY-MM-DD>/
        _manifest.json
        <kind>_<name>_describe.parquet
        <kind>_<name>_density.parquet
        ...

The manifest records, for every frozen object, the exact DecL program text plus
the database list and library version, so ``check_knowledge`` can reconstruct an
identical :class:`~aggregate.Underwriter` and rebuild each object without
guessing. The parquet output is deliberately not tracked in git -- this is a
local aid, not a committed-baseline test.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# Knowledge kinds this harness snapshots. Severities/distortions are skipped:
# they have no aggregate density_df / describe of the relevant shape.
FROZEN_KINDS = ("agg", "port")

# Density columns to keep: per-bucket probabilities and conditional expectations.
DENSITY_REGEX = r"p_|exeqa_"

# Tight default tolerance -- aggregate math is essentially exact, so a recompute
# should reproduce a snapshot to within floating-point noise, not "close enough".
DEFAULT_ATOL = 1e-12

# Default database load request: reproduces the imported ``build``.
DEFAULT_DATABASES = ("test_suite",)

MANIFEST_NAME = "_manifest.json"


def _default_root() -> Path:
    """Machine-agnostic default snapshot root under the OS temp directory.

    Returns
    -------
    pathlib.Path
        ``<tempdir>/aggregate_freeze``. A ``YYYY-MM-DD`` subdirectory is created
        beneath this per freeze. The OS may sweep temp, so pass an explicit
        ``root`` for snapshots that must survive across days.
    """
    return Path(tempfile.gettempdir()) / "aggregate_freeze"


def _sanitize(name: str) -> str:
    """Make an object name safe to use as a filename stem.

    Replaces any character outside ``[A-Za-z0-9._-]`` with an underscore so the
    stem round-trips on Windows and POSIX alike.

    Parameters
    ----------
    name : str
        The knowledge-base object name (may contain spaces or punctuation).

    Returns
    -------
    str
        A filesystem-safe stem.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _frozen_frames(obj):
    """Return the two frames to freeze for a built object.

    Parameters
    ----------
    obj : Aggregate or Portfolio
        A built, updated ``aggregate`` object.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        ``(describe, filtered_density)`` where the second frame keeps only the
        columns matching :data:`DENSITY_REGEX`.
    """
    describe = obj.describe
    density = obj.density_df.filter(regex=DENSITY_REGEX)
    return describe, density


def _build_underwriter(databases):
    """Construct a fresh :class:`~aggregate.Underwriter` for the given databases.

    Imported lazily so the module loads instantly (and ``--help`` stays fast)
    without paying the cost of importing the heavy ``aggregate`` package -- and
    so importing this module in a Jupyter session where ``aggregate`` is already
    loaded is free.

    Parameters
    ----------
    databases : sequence of str
        Database load request(s) passed straight to ``Underwriter``.

    Returns
    -------
    aggregate.Underwriter
        An underwriter whose ``knowledge`` reflects exactly these databases.
    """
    from aggregate import Underwriter

    return Underwriter(databases=list(databases))


def _agg_port_programs(uw):
    """Yield ``(kind, name, program)`` for each agg/port entry in the knowledge.

    Parameters
    ----------
    uw : aggregate.Underwriter
        A loaded underwriter.

    Yields
    ------
    (str, str, str)
        The kind (``'agg'`` or ``'port'``), the object name, and its DecL
        program text.
    """
    k = uw.knowledge
    mask = k.index.get_level_values(0).isin(FROZEN_KINDS)
    for (kind, name), row in k[mask].iterrows():
        yield kind, name, row["program"]


def freeze_knowledge(databases=DEFAULT_DATABASES, root=None, atol=DEFAULT_ATOL) -> dict:
    """Build every agg/port program and snapshot its frames to parquet.

    Creates ``<root>/<today>/`` and writes a ``describe`` and a filtered
    ``density_df`` parquet per object, plus ``_manifest.json``. Objects that
    cannot be built standalone (e.g. measure-zero splices) are skipped with a
    warning and recorded in the manifest.

    Parameters
    ----------
    databases : sequence of str, default ``("test_suite",)``
        Database load request(s). The default reproduces the imported ``build``.
    root : str or pathlib.Path, optional
        Snapshot root; a ``YYYY-MM-DD`` subdirectory is created underneath.
        ``None`` uses :func:`_default_root` (OS temp). Pass an explicit durable
        path for snapshots that must outlive an OS temp sweep.
    atol : float, default :data:`DEFAULT_ATOL`
        Absolute tolerance recorded in the manifest for a later
        :func:`check_knowledge`.

    Returns
    -------
    dict
        The manifest (also written to ``<root>/<today>/_manifest.json``). The
        ``'directory'`` key holds the date directory as a string.
    """
    root = Path(root) if root is not None else _default_root()
    out_dir = root / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    uw = _build_underwriter(databases)
    from aggregate import __version__ as agg_version

    objects = []
    skipped = []
    for kind, name, program in _agg_port_programs(uw):
        stem = f"{kind}_{_sanitize(name)}"
        try:
            obj = uw.build(program)
            describe, density = _frozen_frames(obj)
            describe_file = f"{stem}_describe.parquet"
            density_file = f"{stem}_density.parquet"
            describe.to_parquet(out_dir / describe_file)
            density.to_parquet(out_dir / density_file)
        except Exception as e:  # noqa: BLE001 - one bad program must not abort the run
            print(f"  SKIP {kind} {name!r}: {type(e).__name__}: {e}", file=sys.stderr)
            skipped.append({"kind": kind, "name": name, "error": f"{type(e).__name__}: {e}"})
            continue
        objects.append(
            {
                "kind": kind,
                "name": name,
                "program": program,
                "describe_file": describe_file,
                "density_file": density_file,
            }
        )
        print(f"  froze {kind} {name!r}")

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "aggregate_version": agg_version,
        "databases": list(databases),
        "atol": atol,
        "directory": str(out_dir),
        "objects": objects,
        "skipped": skipped,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nFroze {len(objects)} object(s), skipped {len(skipped)}, into {out_dir}")
    return manifest


def _max_abs_diff(a: pd.DataFrame, b: pd.DataFrame):
    """Largest absolute element-wise difference over shared numeric columns.

    Returns ``None`` when the frames cannot be aligned (different shape or
    columns), signalling a structural rather than numeric mismatch.

    Parameters
    ----------
    a, b : pandas.DataFrame
        Frames to compare.

    Returns
    -------
    float or None
        Maximum absolute difference, or ``None`` if shapes/columns differ.
    """
    if a.shape != b.shape or not a.columns.equals(b.columns):
        return None
    an = a.select_dtypes("number")
    bn = b.select_dtypes("number")
    diff = an.to_numpy() - bn.to_numpy()
    if diff.size == 0:
        return 0.0
    import numpy as np

    return float(np.nanmax(np.abs(diff)))


def check_knowledge(directory, atol=None) -> list:
    """Recompute the objects in ``directory`` and verify they match the snapshot.

    Reads ``_manifest.json``, rebuilds an identical underwriter, recomputes each
    object's frames, and compares them to the frozen parquet.

    Parameters
    ----------
    directory : str or pathlib.Path
        A date folder produced by :func:`freeze_knowledge`.
    atol : float, optional
        Absolute tolerance; ``None`` uses the value recorded in the manifest.

    Returns
    -------
    list of str
        Names of objects that failed (an empty list means everything matched).
    """
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    tol = atol if atol is not None else manifest.get("atol", DEFAULT_ATOL)

    print(
        f"Checking {directory} (frozen {manifest.get('created')}, "
        f"aggregate {manifest.get('aggregate_version')}, atol={tol})"
    )
    uw = _build_underwriter(manifest["databases"])

    failures = []
    for entry in manifest["objects"]:
        kind, name = entry["kind"], entry["name"]
        try:
            obj = uw.build(entry["program"])
            describe_new, density_new = _frozen_frames(obj)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {kind} {name!r}: rebuild error {type(e).__name__}: {e}", file=sys.stderr)
            failures.append(name)
            continue

        ok = True
        for new_df, file_key in (
            (describe_new, "describe_file"),
            (density_new, "density_file"),
        ):
            saved = pd.read_parquet(directory / entry[file_key])
            try:
                pd.testing.assert_frame_equal(
                    saved, new_df, rtol=0, atol=tol, check_dtype=False, check_names=False
                )
            except AssertionError:
                ok = False
                mad = _max_abs_diff(saved, new_df)
                detail = "structure differs" if mad is None else f"max|delta|={mad:.3e}"
                print(f"  FAIL {kind} {name!r} [{entry[file_key]}]: {detail}", file=sys.stderr)

        if ok:
            print(f"  ok   {kind} {name!r}")
        else:
            failures.append(name)

    n = len(manifest["objects"])
    if failures:
        print(f"\n{len(failures)}/{n} object(s) FAILED: {', '.join(sorted(set(failures)))}")
    else:
        print(f"\nAll {n} object(s) match within atol={tol}.")
    return failures


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI: ``freeze`` and ``check`` subcommands."""
    p = argparse.ArgumentParser(
        prog="freeze_knowledge",
        description="Freeze and verify aggregate knowledge-base outputs.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("freeze", help="Snapshot every agg/port object to parquet.")
    pf.add_argument(
        "-d", "--database", dest="databases", action="append", metavar="NAME",
        help="Database name to load (repeatable). Default: test_suite "
             "(reproduces the imported `build`).",
    )
    pf.add_argument(
        "--root", type=Path, default=None,
        help="Snapshot root; a YYYY-MM-DD subdir is created underneath. "
             "Default: <OS temp>/aggregate_freeze.",
    )
    pf.add_argument(
        "--atol", type=float, default=DEFAULT_ATOL,
        help=f"Absolute tolerance recorded for later `check` (default {DEFAULT_ATOL}).",
    )

    pc = sub.add_parser("check", help="Recompute a freeze directory and verify it matches.")
    pc.add_argument("directory", type=Path, help="A date folder produced by `freeze`.")
    pc.add_argument(
        "--atol", type=float, default=None,
        help="Absolute tolerance; overrides the value stored in the manifest.",
    )
    return p


def main(argv=None) -> None:
    """CLI entry point. Exits non-zero if ``check`` finds any mismatch."""
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        databases = tuple(args.databases) if args.databases else DEFAULT_DATABASES
        freeze_knowledge(databases=databases, root=args.root, atol=args.atol)
    elif args.command == "check":
        failures = check_knowledge(args.directory, atol=args.atol)
        if failures:
            sys.exit(1)


if __name__ == "__main__":
    main()
