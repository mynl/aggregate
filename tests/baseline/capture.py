"""Capture the pre-refactor baseline.

Reads ``corpus.py``, builds each case at its pinned grid, snapshots the
load-bearing frames to parquet and the scalar readouts to ``manifest.json``,
along with environment versions and the capture commit SHA.

Usage
-----
    uv run python tests/baseline/capture.py

The output lands in ``tests/baseline/data/``. Re-running overwrites the
previous capture; this is the regenerate-baseline path used when a refactor
step *intends* to move numbers.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate import build
from aggregate.spectral import Distortion

# Allow running as a script from the repo root: ensure the project root is on
# sys.path so ``tests.baseline.corpus`` imports cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.baseline import corpus as C  # noqa: E402


DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_columns(df: pd.DataFrame, spec) -> pd.DataFrame:
    """Apply a column filter spec to a DataFrame.

    ``spec`` is one of: ``None`` (keep all), a list of column names (keep the
    intersection with the frame), or ``{'regex': ...}`` (keep matching).
    """
    if spec is None:
        return df
    if isinstance(spec, dict) and "regex" in spec:
        return df.filter(regex=spec["regex"])
    if isinstance(spec, list):
        keep = [c for c in spec if c in df.columns]
        return df[keep]
    raise TypeError(f"Unsupported column-filter spec: {spec!r}")


def _coerce_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Make ``df`` parquet-friendly.

    pyarrow refuses object columns that mix floats and strings.
    ``stats_df`` has a single ``('meta','name')`` string row sitting in
    otherwise-numeric columns — drop it (D3 will do this for real in
    meta.3 — the harness anticipates) and coerce to float.
    """
    out = df.copy()
    # Drop the legacy ('meta','name') row if present (it's the only non-
    # numeric cell in stats_df).
    if isinstance(out.index, pd.MultiIndex) and ("meta", "name") in out.index:
        out = out.drop(index=("meta", "name"))
    # Categorical column index (e.g. pricing_at's pentagon labels) does not
    # round-trip cleanly through parquet — flatten to plain object labels.
    if isinstance(out.columns, pd.CategoricalIndex):
        out.columns = out.columns.astype(str)
    # Stringify any MultiIndex columns / rows.
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["|".join(str(s) for s in tup) for tup in out.columns]
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
        # The reset_index'd label columns are strings; everything else float.
    # Coerce remaining object columns of numbers to float; flatten any
    # CategoricalDtype values to strings.
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(str)
        elif out[c].dtype == object:
            try:
                out[c] = pd.to_numeric(out[c])
            except (TypeError, ValueError):
                out[c] = out[c].astype(str)
    return out


def _write_parquet(df: pd.DataFrame, case: str, frame: str) -> str:
    """Write ``df`` to ``data/<case>__<frame>.parquet``; return the filename."""
    fname = f"{case}__{frame}.parquet"
    path = DATA_DIR / fname
    out = _coerce_for_parquet(df)
    # Persist a RangeIndex as data only if it was a MultiIndex we reset; the
    # default RangeIndex stays out of the file.
    keep_index = not isinstance(out.index, pd.RangeIndex)
    out.to_parquet(path, engine="pyarrow", index=keep_index)
    return fname


def _scalar_readouts(obj) -> dict:
    """``q(p)`` and ``tvar(p)`` at the corpus probabilities."""
    out = {}
    for p in C.SCALAR_PS:
        out[f"q_{p}"] = float(obj.q(p))
    for p in C.TVAR_PS:
        out[f"tvar_{p}"] = float(obj.tvar(p))
    return out


def _build_distortion(spec: dict) -> Distortion:
    """Instantiate a fixed-shape distortion from a corpus DISTORTIONS entry."""
    return Distortion(name=spec["kind"], **spec["kwargs"])


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _env_versions() -> dict:
    import aggregate
    import numpy
    import scipy
    return {
        "aggregate": aggregate.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "python": sys.version.split()[0],
    }


# ---------------------------------------------------------------------------
# Per-case capture
# ---------------------------------------------------------------------------

def capture_agg_case(name: str, program: str, grid: dict) -> dict:
    """Build one Aggregate case, snapshot frames + scalars, return manifest entry."""
    print(f"  building {name} ...", flush=True)
    # update=False so build() does not auto-call recommend_bucket (Def.Pareto
    # has no finite variance, which would raise). We pin the grid ourselves.
    obj = build(program, update=False)
    obj.update(**grid)

    frames: dict[str, str] = {}
    for frame_name, col_spec in C.AGG_FRAMES.items():
        if frame_name == "density_df":
            df = obj.density_df
        elif frame_name == "describe":
            df = obj.describe
        elif frame_name == "stats_df":
            df = obj.stats_df
        else:
            raise KeyError(frame_name)
        df = _select_columns(df, col_spec)
        frames[frame_name] = _write_parquet(df, name, frame_name)

    return dict(
        kind="aggregate",
        program=program,
        grid=grid,
        frames=frames,
        scalars=_scalar_readouts(obj),
    )


def capture_port_case(name: str, program: str, grid: dict) -> dict:
    """Build one Portfolio case, snapshot frames + per-distortion-method augmented + pricing."""
    print(f"  building {name} ...", flush=True)
    # update=False so build() does not auto-call recommend_bucket (Def.Pareto
    # has no finite variance, which would raise). We pin the grid ourselves.
    obj = build(program, update=False)
    obj.update(**grid)

    frames: dict[str, str] = {}
    for frame_name, col_spec in C.PORT_FRAMES.items():
        if frame_name == "density_df":
            df = obj.density_df
        elif frame_name == "describe":
            df = obj.describe
        elif frame_name == "stats_df":
            df = obj.stats_df
        else:
            raise KeyError(frame_name)
        df = _select_columns(df, col_spec)
        frames[frame_name] = _write_parquet(df, name, frame_name)

    # Per-distortion: augmented_df subset (one per method) + pricing_at row at p=PRICING_P.
    distortion_entries = []
    for dspec in C.DISTORTIONS:
        label = dspec["label"]
        kind = dspec["kind"]
        methods = C.PORT_METHODS[(name, label)]
        dist = _build_distortion(dspec)

        # augmented_df is method-independent for now (lifted form) — the
        # ``allocation_method`` member lands in meta.6. Today we get the
        # lifted-shaped augmented_df and snapshot it once. Linear pricing
        # is captured via the price() readout below.
        aug = obj.apply_distortion(dist, efficient=True)
        aug_sub = _select_columns(aug, C.AUGMENTED_COLUMNS)
        frame_key = f"augmented__{label}"
        frames[frame_key] = _write_parquet(aug_sub, name, frame_key)

        # pricing_at row at p=PRICING_P (one DataFrame per distortion).
        pa = obj.pricing_at(dist, p=C.PRICING_P)
        frames[f"pricing_at__{label}"] = _write_parquet(pa, name, f"pricing_at__{label}")

        # price() under each requested method — capture price + per-line df.
        method_entries = []
        for method in methods:
            pr = obj.price(C.PRICING_P, dist, allocation=method)
            # PricingResult: df (per-line), price (scalar), price_dict, a_reg, reg_p.
            price_df = pr.df.copy()
            frame_key = f"price__{label}__{method}"
            frames[frame_key] = _write_parquet(price_df, name, frame_key)
            method_entries.append(dict(
                method=method,
                price=float(pr.price),
                a_reg=float(pr.a_reg),
                reg_p=float(pr.reg_p),
            ))

        distortion_entries.append(dict(
            label=label,
            kind=kind,
            kwargs=dspec["kwargs"],
            mass=dspec["mass"],
            methods=methods,
            pricing=method_entries,
        ))

    return dict(
        kind="portfolio",
        program=program,
        grid=grid,
        frames=frames,
        scalars=_scalar_readouts(obj),
        distortions=distortion_entries,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict = dict(
        env=_env_versions(),
        commit=_git_sha(),
        tolerance=dict(rtol=1e-12, atol=1e-14),
        cases={},
    )

    print("Aggregates:")
    for name, (program, grid) in C.AGG_CASES.items():
        manifest["cases"][name] = capture_agg_case(name, program, grid)

    print("Portfolios:")
    for name, (program, grid) in C.PORT_CASES.items():
        manifest["cases"][name] = capture_port_case(name, program, grid)

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Wrote {manifest_path}")
    print(f"Captured {len(manifest['cases'])} cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
