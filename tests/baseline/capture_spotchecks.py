"""Capture spot-check values for the *derived* objective columns (numerics-2).

The main baseline (``capture.py``) snapshots the **key** columns — portfolio
``p_*`` / ``exeqa_*`` and Aggregate ``p_total/F/S/lev`` — which anchor the
numerics-2 regression (meta D5: everything else is derived from them). This
companion captures **spot-check values** for the derived columns

* Portfolio: ``exa_* / exlea_* / exgta_* / exi_xgta_*`` (+ ``lev_*``)
* Aggregate: ``exa / exlea / exgta``

at a handful of sampled rows per corpus case, so the direct-sum rewrite in
``dev/plan-numerics-2-objective.md`` has a measured target rather than a
guessed one. Run **on pre-change code**::

    uv run python tests/baseline/capture_spotchecks.py

Output: ``tests/baseline/data/spotchecks.json``. Checked by
``tests/test_baseline_spotchecks.py``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate import build

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.baseline import corpus as C  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
OUT_PATH = DATA_DIR / "spotchecks.json"

# Sample rows by total-distribution quantile so the picks live in the live
# region of each grid (away from the blanked / noise-dominated edges).
SAMPLE_PS = (0.05, 0.25, 0.50, 0.90, 0.99, 0.999)

AGG_COLUMNS = ("exa", "exlea", "exgta", "lev")
PORT_COLUMN_PREFIXES = ("exa_", "exlea_", "exgta_", "exi_xgta_", "lev_")


def _sample_rows(obj) -> list[float]:
    """Loss-grid rows at the SAMPLE_PS quantiles of the total (deduped)."""
    return sorted({float(obj.q(p)) for p in SAMPLE_PS})


def _capture(obj, columns) -> dict:
    df = obj.density_df
    rows = _sample_rows(obj)
    out = {}
    for loss in rows:
        rec = {}
        for c in columns:
            v = df.at[loss, c]
            rec[c] = None if pd.isna(v) else float(v)
        out[repr(loss)] = rec
    return out


def main() -> int:
    spot: dict = dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        sample_ps=list(SAMPLE_PS),
        cases={},
    )

    for name, (program, grid) in C.AGG_CASES.items():
        print(f"  agg {name} ...", flush=True)
        obj = build(program, update=False)
        obj.update(**grid)
        spot["cases"][name] = dict(
            kind="aggregate",
            values=_capture(obj, AGG_COLUMNS),
        )

    for name, (program, grid) in C.PORT_CASES.items():
        print(f"  port {name} ...", flush=True)
        obj = build(program, update=False)
        obj.update(**grid)
        cols = [c for c in obj.density_df.columns
                if any(c.startswith(p) for p in PORT_COLUMN_PREFIXES)
                and not c.endswith("_sum")]
        spot["cases"][name] = dict(
            kind="portfolio",
            values=_capture(obj, cols),
        )

    OUT_PATH.write_text(json.dumps(spot, indent=1))
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
