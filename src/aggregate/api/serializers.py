"""DataFrame and object → JSON-friendly helpers.

Centralized so route handlers stay thin: one ``frame_to_payload``
call converts a pandas DataFrame to the ``(columns, rows)`` shape
expected by :class:`aggregate.api.models.FrameResponse`.

Numeric/string coercion notes
-----------------------------

* NaN / Inf serialize to None -- strict JSON forbids them and
  ``json.dumps(float("nan"))`` produces non-parseable output that
  some clients reject.
* MultiIndex columns are flattened with ``.`` joins so the wire
  format stays a plain list-of-strings (``"freq.mean"``,
  ``"sev.cv"``). The client can split back on ``.`` if needed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _safe(value: Any) -> Any:
    """Coerce one cell to a JSON-friendly value.

    * Non-finite floats → None (strict JSON).
    * numpy scalars → native Python (avoids ``numpy.int64`` objects
      breaking ``json.dumps``).
    * numpy arrays / list-likes → nested lists (recursive coerce).
    * ``pd.NA`` / ``pd.NaT`` → None.
    * Everything else falls back to ``str()`` so a single weird
      cell can't 500 the whole response.
    """
    if value is None:
        return None
    # pandas NA / NaT sentinels don't compare cleanly via ``is None``.
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float):
        # math.isfinite covers NaN, +Inf, -Inf.
        if not math.isfinite(value):
            return None
    if isinstance(value, np.ndarray):
        # 0-d arrays ``np.array(3.)``: .tolist() returns a scalar
        # (not iterable). Higher-d: returns a (possibly nested) list.
        as_list = value.tolist()
        if isinstance(as_list, list):
            return [_safe(v) for v in as_list]
        return _safe(as_list)
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    # ``str``, ``int``, ``bool``, ``float`` are JSON-native; let
    # anything else through as its str() form so unexpected dtypes
    # don't blow up serialization.
    if isinstance(value, (str, int, bool, float)):
        return value
    return str(value)


def reset_index_safe(df: pd.DataFrame) -> pd.DataFrame:
    """``df.reset_index()`` that tolerates duplicate names.

    ``Aggregate.density_df`` has its index named ``'loss'`` and a
    column named ``'loss'`` -- a plain ``reset_index()`` raises
    ``ValueError: cannot insert loss, already exists``. Detect that
    case and return the frame unchanged (the column already carries
    the index values).

    For a MultiIndex with one or more names colliding with existing
    columns, we let pandas raise -- that case is rare and signals
    a real upstream issue.
    """
    idx = df.index
    if idx.name is None and not isinstance(idx, pd.MultiIndex):
        # Anonymous single-level index -- nothing to surface.
        return df
    if isinstance(idx, pd.MultiIndex):
        return df.reset_index()
    if idx.name in df.columns:
        # Column already carries the index values; reset would
        # collide. Leave the frame as-is.
        return df
    return df.reset_index()


def _flatten_columns(columns: pd.Index) -> list[str]:
    """MultiIndex → list of ``"a.b.c"`` strings; plain Index → list of str."""
    if isinstance(columns, pd.MultiIndex):
        return [".".join(str(p) for p in tup) for tup in columns.values]
    return [str(c) for c in columns]


def frame_to_payload(
    df: pd.DataFrame,
    *,
    cols: list[str] | None = None,
    start: int | None = None,
    stop: int | None = None,
    downsample: int | None = None,
) -> dict:
    """Convert a DataFrame to the ``FrameResponse`` payload.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame. The index is *not* included in the output
        unless explicitly named -- callers can ``df.reset_index()``
        first if they want it.
    cols : list[str] | None
        Subset of columns to return. Names absent from the frame are
        silently dropped (callers can request a generic set and let
        the api filter).
    start, stop : int | None
        Positional row slice (NOT label slice). ``None`` means
        unbounded on that side.
    downsample : int | None
        If set, return at most this many rows -- evenly spaced
        across whatever slice survived the start/stop. Used by
        the SPA to render a 2**16-row density_df at sensible
        display resolution.
    """
    if cols:
        # Tolerate caller passing column names that aren't on this
        # frame -- the SPA might ask for "exeqa_total" on an
        # Aggregate (no such column) without failing the round-trip.
        existing = [c for c in cols if c in df.columns]
        df = df[existing]

    # Positional slicing first, then downsample. Downsampling
    # *after* slicing means a request like ``start=0, stop=1000,
    # downsample=100`` returns 100 rows from the first 1000, not
    # 100 rows spread across the whole 2**N grid.
    sliced = df.iloc[slice(start, stop)]

    if downsample is not None and len(sliced) > downsample > 0:
        # Even-spaced index sampling. linspace + round + unique
        # avoids duplicate row picks when downsample is close to
        # len(sliced).
        idx = np.unique(
            np.round(np.linspace(0, len(sliced) - 1, downsample)).astype(int)
        )
        sliced = sliced.iloc[idx]

    columns = _flatten_columns(sliced.columns)
    # ``.to_numpy()`` is faster than .values and preserves dtype.
    rows = [
        [_safe(v) for v in row]
        for row in sliced.to_numpy().tolist()
    ]
    return {"columns": columns, "rows": rows}


def info_to_payload(obj: Any) -> dict:
    """Return ``{"info": "..."}``.

    :attr:`Aggregate.info` and :attr:`Portfolio.info` are
    multi-line strings (formatted summaries). We expose them
    verbatim; clients display in a monospaced block.
    """
    info = getattr(obj, "info", "")
    if not isinstance(info, str):
        # Fallback for objects that override .info as something
        # else -- str() coerces to a usable rendering.
        info = str(info)
    return {"info": info}
