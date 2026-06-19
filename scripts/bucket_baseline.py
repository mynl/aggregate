"""Snapshot the bucket-sizing decisions for every knowledge-base program.

A standalone, dependency-free maintenance aid (sibling to
``freeze_knowledge.py``). It walks every ``agg`` / ``port`` program in a chosen
knowledge base, builds each object with default parameters, and records the
**grid-sizing outcome** for review:

* a one-row-per-object **summary** -- ``log2``, ``bs``, the signed flag and the
  unit count; and
* the full per-object **``_bs_window_df``** -- the expert-inspectable window
  estimator table (rows are the candidate sizing *methods* for an ``Aggregate``
  -- ``moment`` / ``exact_discrete`` / ``bounded_small`` / ``used`` -- or the
  component *units* plus ``used`` for a ``Portfolio``).

The point is a committed, human-diffable reference so that any future change to
the bucket heuristics (``recommend_bucket`` / ``best_bucket`` / ``_bs_window``)
surfaces as a reviewable CSV diff: *which* programs changed ``bs`` or ``log2``,
and how their windows moved. This is a **baseline-for-review**, not a hard
regression assertion -- some entries are *expected* to change when the heuristic
is improved, and the diff is read, not just gated.

CSV (not parquet) is deliberate: it diffs cleanly in git. A fixed float format
keeps noise out of the diff while preserving enough precision to spot real
moves.

Usage
-----
As importable functions (ideal from Jupyter, where ``aggregate`` is loaded)::

    from bucket_baseline import write_bucket_baseline
    write_bucket_baseline()                      # -> tests/data/bucket_baseline_*.csv

As a CLI (stdlib argparse, no third-party dependencies)::

    python scripts/bucket_baseline.py [-d NAME ...] [--out-dir DIR]

Output (two files in ``--out-dir``, default ``tests/data``)::

    bucket_baseline_summary.csv     one row per object: kind,name,log2,bs,...
    bucket_baseline_windows.csv     every _bs_window_df, concatenated

Both are written in a deterministic ``(kind, name)`` order so re-runs produce
byte-stable output and git diffs are meaningful.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Knowledge kinds with a grid: severities/distortions have no bucket decision.
BAR_KINDS = ("agg", "port")

# Default database load request: reproduces the imported ``build``.
DEFAULT_DATABASES = ("_test_suite",)

# Canonical column order of ``_bs_window_df`` (shared by Aggregate/Portfolio).
WINDOW_COLUMNS = ["x_min", "x_max", "W", "bs", "log2", "coverage", "note"]

# Where the committed baseline lives, relative to the repo root.
DEFAULT_OUT_DIR = Path("tests") / "data"

SUMMARY_NAME = "bucket_baseline_summary.csv"
WINDOWS_NAME = "bucket_baseline_windows.csv"

# Enough precision to catch a real move, tight enough to avoid fp-noise churn.
FLOAT_FORMAT = "%.12g"


def _build_underwriter(databases):
    """Construct a fresh :class:`~aggregate.Underwriter` for ``databases``.

    Imported lazily so the module (and ``--help``) loads without paying the
    ``aggregate`` import cost, and so importing this module inside a Jupyter
    session where ``aggregate`` is already loaded is free.

    Parameters
    ----------
    databases : sequence of str
        Database load request(s) passed straight to ``Underwriter``.

    Returns
    -------
    aggregate.Underwriter
    """
    from aggregate import Underwriter

    return Underwriter(databases=list(databases))


def _agg_port_programs(uw):
    """Yield ``(kind, name, program)`` for each agg/port entry, in sorted order.

    Parameters
    ----------
    uw : aggregate.Underwriter
        A loaded underwriter.

    Yields
    ------
    (str, str, str)
        Kind (``'agg'``/``'port'``), object name, and DecL program text. Sorted
        by ``(kind, name)`` so the baseline is order-stable across runs.
    """
    k = uw.knowledge
    mask = k.index.get_level_values(0).isin(BAR_KINDS)
    for (kind, name), row in k[mask].sort_index().iterrows():
        yield kind, name, row["program"]


def _signed(obj) -> bool:
    """Best-effort read of an object's signed (negative-support) flag.

    Parameters
    ----------
    obj : Aggregate or Portfolio

    Returns
    -------
    bool
        ``obj._signed()`` if available, else ``False``.
    """
    try:
        return bool(obj._signed())
    except Exception:  # pragma: no cover - defensive
        return False


def _unit_count(obj) -> int:
    """Number of component aggregates (1 for a bare ``Aggregate``).

    Parameters
    ----------
    obj : Aggregate or Portfolio

    Returns
    -------
    int
    """
    agg_list = getattr(obj, "agg_list", None)
    return len(agg_list) if agg_list is not None else 1


def collect_bucket_baseline(databases=DEFAULT_DATABASES):
    """Build every agg/port program and collect its bucket-sizing outcome.

    Parameters
    ----------
    databases : sequence of str, default ``("_test_suite",)``
        Database load request(s). The default reproduces the imported ``build``.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        ``(summary, windows)``.

        ``summary`` has one row per object, columns
        ``[kind, name, log2, bs, signed, n_units, program]``.

        ``windows`` is every object's ``_bs_window_df`` concatenated, with
        leading identifier columns ``[kind, name, row]`` (``row`` is the
        original ``_bs_window_df`` index label -- a method name for an
        ``Aggregate``, a unit name or ``used`` for a ``Portfolio``) followed by
        :data:`WINDOW_COLUMNS`.

    Notes
    -----
    Objects that fail to build are skipped with a stderr note (mirrors
    ``freeze_knowledge``); one bad program never aborts the sweep.
    """
    uw = _build_underwriter(databases)

    summary_rows = []
    window_frames = []
    for kind, name, program in _agg_port_programs(uw):
        try:
            obj = uw.build(program)
        except Exception as e:  # noqa: BLE001 - one bad program must not abort
            print(f"  SKIP {kind} {name!r}: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        summary_rows.append(
            {
                "kind": kind,
                "name": name,
                "log2": getattr(obj, "log2", None),
                "bs": getattr(obj, "bs", None),
                "signed": _signed(obj),
                "n_units": _unit_count(obj),
                "program": program,
            }
        )

        wdf = getattr(obj, "_bs_window_df", None)
        if wdf is not None and len(wdf):
            w = wdf.copy()
            # The index label (method/unit/'used') becomes an explicit column so
            # the concatenated table stays flat and diff-friendly.
            w = w.reset_index()
            w = w.rename(columns={w.columns[0]: "row"})
            w.insert(0, "name", name)
            w.insert(0, "kind", kind)
            # Keep only the known columns we can rely on across both classes.
            keep = ["kind", "name", "row"] + [c for c in WINDOW_COLUMNS if c in w.columns]
            window_frames.append(w[keep])
        print(f"  sized {kind} {name!r}: bs={getattr(obj, 'bs', '?')} log2={getattr(obj, 'log2', '?')}")

    summary = pd.DataFrame(summary_rows)
    windows = (
        pd.concat(window_frames, ignore_index=True)
        if window_frames
        else pd.DataFrame(columns=["kind", "name", "row"] + WINDOW_COLUMNS)
    )
    return summary, windows


def write_bucket_baseline(databases=DEFAULT_DATABASES, out_dir=None):
    """Collect the baseline and write the two CSVs.

    Parameters
    ----------
    databases : sequence of str, default ``("_test_suite",)``
        Database load request(s).
    out_dir : str or pathlib.Path, optional
        Destination directory. ``None`` uses :data:`DEFAULT_OUT_DIR`
        (``tests/data`` relative to the current working directory -- run from the
        repo root).

    Returns
    -------
    (pathlib.Path, pathlib.Path)
        The written ``(summary_path, windows_path)``.
    """
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, windows = collect_bucket_baseline(databases)

    summary_path = out_dir / SUMMARY_NAME
    windows_path = out_dir / WINDOWS_NAME
    summary.to_csv(summary_path, index=False, float_format=FLOAT_FORMAT)
    windows.to_csv(windows_path, index=False, float_format=FLOAT_FORMAT)

    print(
        f"\nWrote {len(summary)} object(s) -> {summary_path}\n"
        f"      {len(windows)} window row(s) -> {windows_path}"
    )
    return summary_path, windows_path


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI (single command: write the baseline CSVs)."""
    p = argparse.ArgumentParser(
        prog="bucket_baseline",
        description="Snapshot bucket-sizing (log2, bs, _bs_window_df) for every "
                    "agg/port program to committed CSVs for review.",
    )
    p.add_argument(
        "-d", "--database", dest="databases", action="append", metavar="NAME",
        help="Database name to load (repeatable). Default: _test_suite "
             "(reproduces the imported `build`).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help=f"Destination directory. Default: {DEFAULT_OUT_DIR} "
             "(run from the repo root).",
    )
    return p


def main(argv=None) -> None:
    """CLI entry point: build all programs and write the baseline CSVs."""
    args = _build_parser().parse_args(argv)
    databases = tuple(args.databases) if args.databases else DEFAULT_DATABASES
    write_bucket_baseline(databases=databases, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
