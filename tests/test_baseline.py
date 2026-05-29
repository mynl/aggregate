"""Before/after baseline characterisation test.

Loads ``tests/baseline/data/manifest.json``, rebuilds each case at its pinned
grid, recomputes each snapshotted frame, and compares element-wise against
the stored parquet. **Runs every case before reporting**; per the protocol in
``dev/plan-baseline-harness.md`` §6, divergences are collected into a single
summary and reported all at once.

Run
---
    uv run pytest tests/test_baseline.py
    uv run pytest tests/test_baseline.py -k Port.CNC      # one case
    uv run pytest tests/test_baseline.py -v --tb=short

Regenerate baseline
-------------------
After a refactor step that *intends* to move numbers (regenerating is a
deliberate act, not an automated one)::

    uv run python tests/baseline/capture.py

then commit the new parquet + manifest in the same commit that moves the
numbers, with a message explaining the move.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest

from aggregate import build
from aggregate.spectral import Distortion

from tests.baseline import corpus as C
from tests.baseline.capture import _coerce_for_parquet


DATA_DIR = Path(__file__).parent / "baseline" / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"


# ---------------------------------------------------------------------------
# Per-failure record
# ---------------------------------------------------------------------------

@dataclass
class Divergence:
    case: str
    frame: str
    column: str
    max_abs: float
    max_rel: float
    row_label: str
    observation: str

    def render(self) -> str:
        return (
            f"  {self.case} / {self.frame} / {self.column} : "
            f"max abs {self.max_abs:.3e} at row {self.row_label!r}, "
            f"max rel {self.max_rel:.3e} -- {self.observation}"
        )


def _worst_offender(actual: pd.Series, expected: pd.Series,
                    rtol: float, atol: float) -> Divergence | None:
    """Return a Divergence for the worst-offending cell, or None if all-close.

    Mirrors the np.allclose check on a per-element basis so we can pinpoint
    the failing index.
    """
    a = pd.to_numeric(actual, errors="coerce").to_numpy()
    e = pd.to_numeric(expected, errors="coerce").to_numpy()

    # NaN handling: NaN == NaN is treated as equal here (matches the captured
    # baseline: if the captured value is NaN and the rebuilt value is NaN,
    # they agree).
    both_nan = np.isnan(a) & np.isnan(e)
    diff_abs = np.where(both_nan, 0.0, np.abs(a - e))
    denom = np.where(np.abs(e) > 0, np.abs(e), 1.0)
    diff_rel = np.where(both_nan, 0.0, diff_abs / denom)

    tol = atol + rtol * np.abs(e)
    bad = diff_abs > tol
    # Anywhere a is NaN but e is not (or vice versa) is also bad.
    nan_disagree = np.isnan(a) ^ np.isnan(e)
    bad = bad | nan_disagree
    if not bad.any():
        return None
    idx = int(np.nanargmax(np.where(bad, diff_abs, np.nan)))
    return Divergence(
        case="",  # filled by caller
        frame="",
        column=str(actual.name) if actual.name is not None else "",
        max_abs=float(np.nanmax(diff_abs)),
        max_rel=float(np.nanmax(diff_rel)),
        row_label=str(actual.index[idx]),
        observation=_observe(a, e, bad),
    )


def _observe(a: np.ndarray, e: np.ndarray, bad: np.ndarray) -> str:
    """One-line hint at the shape of the divergence."""
    n = bad.size
    nbad = int(bad.sum())
    if nbad == n:
        return "all rows"
    if nbad / n > 0.5:
        return f"{nbad}/{n} rows (majority)"
    # Where do the bad rows cluster?
    bad_idx = np.flatnonzero(bad)
    if bad_idx.max() == n - 1 and bad_idx.min() > 0.8 * n:
        return f"tail-only ({nbad}/{n} rows, last {n - bad_idx.min()})"
    if bad_idx.min() == 0 and bad_idx.max() < 0.2 * n:
        return f"head-only ({nbad}/{n} rows)"
    return f"{nbad}/{n} rows scattered"


# ---------------------------------------------------------------------------
# Rebuild + compare
# ---------------------------------------------------------------------------

def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        pytest.skip(
            f"No baseline manifest at {MANIFEST_PATH}. "
            "Run `uv run python tests/baseline/capture.py` to capture one."
        )
    return json.loads(MANIFEST_PATH.read_text())


def _load_expected(fname: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / fname)


def _rebuild_agg(name: str, program: str, grid: dict) -> object:
    obj = build(program, update=False)
    obj.update(**grid)
    return obj


def _rebuild_port(name: str, program: str, grid: dict) -> object:
    obj = build(program, update=False)
    obj.update(**grid)
    return obj


def _get_frame(obj, frame_name: str, kind: str) -> pd.DataFrame:
    if frame_name == "stats_df":
        return obj.stats_df
    if frame_name == "describe":
        return obj.describe
    if frame_name == "density_df":
        return obj.density_df
    raise KeyError(frame_name)


def _diff_frame(actual: pd.DataFrame, expected: pd.DataFrame,
                case: str, frame: str,
                rtol: float, atol: float) -> list[Divergence]:
    """Compare two frames column-by-column, collect all divergences."""
    a_coerced = _coerce_for_parquet(actual)
    # Align column sets — only check columns present in the expected snapshot
    # (the snapshot is the authoritative subset).
    divs: list[Divergence] = []
    e = expected
    a = a_coerced
    # Common columns only; missing columns flagged below.
    common = [c for c in e.columns if c in a.columns]
    missing = [c for c in e.columns if c not in a.columns]
    for c in missing:
        divs.append(Divergence(
            case=case, frame=frame, column=str(c),
            max_abs=float("inf"), max_rel=float("inf"),
            row_label="-", observation="column missing from rebuild",
        ))
    # Row-length mismatch is its own (catastrophic) failure.
    if len(a) != len(e):
        divs.append(Divergence(
            case=case, frame=frame, column="<rowcount>",
            max_abs=float("inf"), max_rel=float("inf"),
            row_label="-",
            observation=f"row count {len(a)} vs expected {len(e)}",
        ))
        # Don't try to compare cells when row counts disagree.
        return divs
    # Reset index for positional comparison — both sides went through
    # ``_coerce_for_parquet`` which resets MultiIndex into columns.
    a = a.reset_index(drop=True)
    e = e.reset_index(drop=True)
    for col in common:
        e_col = e[col]
        a_col = a[col]
        if e_col.dtype == object or a_col.dtype == object:
            # Stringified label column from reset_index — equality check.
            if not e_col.equals(a_col):
                divs.append(Divergence(
                    case=case, frame=frame, column=str(col),
                    max_abs=float("inf"), max_rel=float("inf"),
                    row_label="-", observation="non-numeric mismatch",
                ))
            continue
        div = _worst_offender(a_col, e_col, rtol, atol)
        if div is not None:
            div.case = case
            div.frame = frame
            divs.append(div)
    return divs


# ---------------------------------------------------------------------------
# The one collecting test
# ---------------------------------------------------------------------------

def test_baseline():
    """Run every case, collect every divergence, report once."""
    manifest = _load_manifest()
    rtol = manifest["tolerance"]["rtol"]
    atol = manifest["tolerance"]["atol"]

    divs: list[Divergence] = []
    n_cases = 0
    n_failed_cases = 0

    # Squash the recurring divide-by-zero / log warnings that fire on the
    # construction paths — they are not what this test is for.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for case_name, entry in manifest["cases"].items():
            n_cases += 1
            case_divs: list[Divergence] = []
            kind = entry["kind"]
            program = entry["program"]
            grid = entry["grid"]

            if kind == "aggregate":
                obj = _rebuild_agg(case_name, program, grid)
            else:
                obj = _rebuild_port(case_name, program, grid)

            # 1) frames that are obj attributes
            for frame_name in ("stats_df", "describe", "density_df"):
                if frame_name not in entry["frames"]:
                    continue
                expected = _load_expected(entry["frames"][frame_name])
                actual = _get_frame(obj, frame_name, kind)
                # Apply the same column filter that capture used.
                if kind == "aggregate":
                    col_spec = C.AGG_FRAMES.get(frame_name)
                else:
                    col_spec = C.PORT_FRAMES.get(frame_name)
                if col_spec is not None:
                    actual = _apply_filter(actual, col_spec)
                case_divs.extend(_diff_frame(actual, expected,
                                             case_name, frame_name, rtol, atol))

            # 2) scalars
            scalars = entry.get("scalars", {})
            for key, expected_val in scalars.items():
                if key.startswith("q_"):
                    p = float(key[2:])
                    actual_val = float(obj.q(p))
                elif key.startswith("tvar_"):
                    p = float(key[5:])
                    actual_val = float(obj.tvar(p))
                else:
                    continue
                if not np.isclose(actual_val, expected_val, rtol=rtol, atol=atol):
                    case_divs.append(Divergence(
                        case=case_name, frame="scalars", column=key,
                        max_abs=abs(actual_val - expected_val),
                        max_rel=abs(actual_val - expected_val) / max(abs(expected_val), 1.0),
                        row_label="-", observation=f"{actual_val!r} vs {expected_val!r}",
                    ))

            # 3) portfolio per-distortion frames
            if kind == "portfolio":
                case_divs.extend(_diff_portfolio_distortions(
                    obj, case_name, entry, rtol, atol))

            if case_divs:
                n_failed_cases += 1
                divs.extend(case_divs)

    if not divs:
        return  # all good

    lines = [
        "",
        f"Harness: {n_failed_cases} of {n_cases} cases failed.",
        f"(rtol={rtol:.0e}, atol={atol:.0e}; baseline commit "
        f"{manifest.get('commit', '?')[:7]})",
        "",
    ]
    lines.extend(d.render() for d in divs)
    pytest.fail("\n".join(lines), pytrace=False)


def _apply_filter(df: pd.DataFrame, spec) -> pd.DataFrame:
    if spec is None:
        return df
    if isinstance(spec, dict) and "regex" in spec:
        return df.filter(regex=spec["regex"])
    if isinstance(spec, list):
        keep = [c for c in spec if c in df.columns]
        return df[keep]
    return df


def _diff_portfolio_distortions(obj, case_name: str, entry: dict,
                                rtol: float, atol: float) -> list[Divergence]:
    """Rebuild augmented_df / pricing_at / price() per distortion and diff."""
    divs: list[Divergence] = []
    for dspec in entry["distortions"]:
        label = dspec["label"]
        kind = dspec["kind"]
        kwargs = dspec["kwargs"]
        dist = Distortion(name=kind, **kwargs)

        # augmented_df
        aug = obj.apply_distortion(dist, efficient=True)
        aug_sub = _apply_filter(aug, C.AUGMENTED_COLUMNS)
        expected = _load_expected(entry["frames"][f"augmented__{label}"])
        divs.extend(_diff_frame(aug_sub, expected,
                                case_name, f"augmented__{label}", rtol, atol))

        # pricing_at
        pa = obj.pricing_at(dist, p=C.PRICING_P)
        expected = _load_expected(entry["frames"][f"pricing_at__{label}"])
        divs.extend(_diff_frame(pa, expected,
                                case_name, f"pricing_at__{label}", rtol, atol))

        # price() per method
        for method_entry in dspec["pricing"]:
            method = method_entry["method"]
            pr = obj.price(C.PRICING_P, dist, allocation=method)
            expected = _load_expected(entry["frames"][f"price__{label}__{method}"])
            divs.extend(_diff_frame(pr.df, expected,
                                    case_name, f"price__{label}__{method}",
                                    rtol, atol))
            for scalar_key in ("price", "a_reg", "reg_p"):
                actual_val = float(getattr(pr, scalar_key))
                expected_val = float(method_entry[scalar_key])
                if not np.isclose(actual_val, expected_val, rtol=rtol, atol=atol):
                    divs.append(Divergence(
                        case=case_name, frame=f"price__{label}__{method}",
                        column=scalar_key,
                        max_abs=abs(actual_val - expected_val),
                        max_rel=abs(actual_val - expected_val) / max(abs(expected_val), 1.0),
                        row_label="-",
                        observation=f"{actual_val!r} vs {expected_val!r}",
                    ))
    return divs
