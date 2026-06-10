"""Derived-column spot-check regression (numerics-2, meta D5).

The main baseline gates the **key** columns (``p_*`` / ``exeqa_*`` and the
Aggregate ``p_total/F/S/lev``); everything else is derived from them. This
test gates the *derived* objective columns — Portfolio
``exa_* / exlea_* / exgta_* / exi_xgta_* / lev_*`` and Aggregate
``exa / exlea / exgta / lev`` — against values captured **on pre-change
code** (``tests/baseline/capture_spotchecks.py``) at rows sampled by total
quantile, so the direct-sum rewrite has a measured target.

Tolerances (measured on the numerics-2 rewrite, ``cumsum(S)·bs`` → direct
sums; see ``dev/audit-numerics-2-findings.md``):

* most columns drift ≤ 2.5e-14 relative → gate rtol = 1e-13;
* ``exgta`` is ``(e − cum)/S`` — a genuine cancellation near the right
  tail amplifies order-of-operations drift to ≤ 7.5e-12 → rtol = 1e-10;
* ``atol = 1e-14`` absorbs sub-noise dust where the legacy value was an
  exact 0 (e.g. ``exa_{line}`` at ``loss=0``, now 1e-17 kappa fuzz × p).

``KNOWN_FLIPS`` lists the four guard-boundary semantic changes (legacy
unguarded division / ``loss_max`` blanking → explicit ``F/S ≤ tol``
guards), each verified by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aggregate import build

from tests.baseline import corpus as C

SPOT_PATH = Path(__file__).parent / "baseline" / "data" / "spotchecks.json"
ATOL = 1e-14
RTOL_DEFAULT = 1e-13
RTOL_BY_COLUMN_BASE = {
    "exgta": 1e-10,
    # lev_{line} on a bounded unit saturates at the mean; the legacy
    # 65k-term cumsum and the native capped sum accumulate eps differently
    # (measured 1.9e-13)
    "lev": 1e-12,
}

# (case, column, loss): intended NaN-semantics changes at guard boundaries.
KNOWN_FLIPS = {
    # last support atom, S == 0 exactly: legacy 6 + 0/0 noise gave 6.0;
    # the S <= tol guard now blanks (no conditioning event).
    ("Sym.Dice", "exgta", 6.0),
    # legacy loss_max blanking NaN'd the row; E[X|X<=0] = 0 is correct
    # (F(0) = 0.76 is material on this discrete book).
    ("Port.Bodoff", "exlea_total", 0.0),
    # max-loss row, S == 0: legacy divided to -inf; now NaN.
    ("Port.Bodoff", "exgta_total", 199.0),
    ("Port.Bodoff", "exgta_wind", 199.0),
}


def _cases():
    if not SPOT_PATH.exists():
        return []
    spot = json.loads(SPOT_PATH.read_text())
    return list(spot["cases"].items())


def _rtol(col):
    return RTOL_BY_COLUMN_BASE.get(col.split("_")[0], RTOL_DEFAULT)


@pytest.mark.parametrize("name,entry", _cases(), ids=[n for n, _ in _cases()])
def test_derived_spotchecks(name, entry):
    all_cases = {**C.AGG_CASES, **C.PORT_CASES}
    program, grid = all_cases[name]
    obj = build(program, update=False)
    obj.update(**grid)
    df = obj.density_df

    bad = []
    for loss_repr, rec in entry["values"].items():
        loss = float(loss_repr)
        for col, expected in rec.items():
            if (name, col, loss) in KNOWN_FLIPS:
                continue
            actual = df.at[loss, col]
            if expected is None or not np.isfinite(expected):
                if not pd.isna(actual):
                    bad.append(f"{col}@{loss}: expected NaN, got {actual!r}")
                continue
            if pd.isna(actual):
                bad.append(f"{col}@{loss}: expected {expected!r}, got NaN")
                continue
            if not np.isclose(actual, expected, rtol=_rtol(col), atol=ATOL):
                rel = abs(actual - expected) / max(abs(expected), 1e-300)
                bad.append(f"{col}@{loss}: rel err {rel:.3e} "
                           f"({actual!r} vs {expected!r})")
    if bad:
        pytest.fail(f"{name}: {len(bad)} spot-check divergences:\n  "
                    + "\n  ".join(bad))


def test_spotchecks_exist():
    assert SPOT_PATH.exists(), (
        "No derived-column spot-check capture. Run "
        "`uv run python tests/baseline/capture_spotchecks.py` on pre-change "
        "code (regenerating on changed code is a deliberate act)."
    )
