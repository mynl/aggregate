"""Pricing dispatch -- distortion-based or constant cost of capital.

The ``POST /v1/objects/{id}/pricing_at`` endpoint accepts a flexible
body and the server picks the underlying method based on which
fields are present:

* ``ccoc`` set                       → :meth:`Portfolio.price_ccoc`
* ``distortion`` set (+ ``p`` or ``a``) → :meth:`Portfolio.pricing_at`

Pricing is Portfolio-only in v1; calling on an Aggregate raises
``ValueError`` and the route turns that into HTTP 400.

Why dispatch in one endpoint
----------------------------

The SPA's pricing pane has one form -- "distortion or CoC, asset
level or probability". A single endpoint matches that UX and keeps
the client honest (server validates the combination, client just
ships the form values).
"""

from __future__ import annotations

from typing import Any


def run_pricing(
    obj: Any,
    *,
    p: float | None,
    a: float | None,
    ccoc: float | None,
    distortion: str | None,
) -> dict:
    """Dispatch to the right Portfolio method.

    Parameters
    ----------
    obj : Portfolio
        The live object.
    p, a : float | None
        Probability or asset level. Exactly one must be set.
    ccoc : float | None
        If set, use ``price_ccoc(p, ccoc)``. Requires ``p``.
    distortion : str | None
        Distortion name (looked up on ``obj.distortions``) for
        ``pricing_at(distortion, p=..., a=...)``.

    Returns
    -------
    dict
        Matches :class:`PricingResponse`: headline fields plus a
        per-line ``rows`` list-of-dicts.
    """
    # Pricing surface is Portfolio-only; Aggregate has no
    # ``pricing_at`` / ``price_ccoc``. Check up front so we can
    # raise a clean error rather than AttributeError mid-call.
    if not hasattr(obj, "pricing_at"):
        raise ValueError("pricing endpoint requires a Portfolio")

    if (p is None) == (a is None):
        # Same invariant the Portfolio methods enforce; flag early
        # so the client gets a sensible 400 instead of bubbling the
        # library ValueError through HTTPException.
        raise ValueError("exactly one of p or a must be provided")

    if ccoc is not None:
        # price_ccoc only takes ``p`` (it computes ``a`` internally
        # via q(p)); reject a stray ``a`` for clarity.
        if p is None:
            raise ValueError("price_ccoc requires p (probability)")
        df = obj.price_ccoc(p, ccoc)
        # The price_ccoc DataFrame has a single row indexed
        # 'total'; expose it as a list-of-dicts for symmetry with
        # the distortion path.
        rows = _df_to_records(df)
        return {
            "a": _scalar(df.loc["total", "a"]) if "a" in df.columns else None,
            "p": p,
            "ccoc": ccoc,
            "distortion": None,
            "rows": rows,
        }

    if distortion is None:
        raise ValueError("must supply either ccoc or distortion")

    df = obj.pricing_at(distortion, p=p, a=a)
    rows = _df_to_records(df, index_name="line")
    # ``pricing_at`` doesn't return the asset level itself in the
    # frame; recover it the same way the method did to surface on
    # the response.
    a_used = a if a is not None else obj.q(p)
    return {
        "a": _scalar(a_used),
        "p": p,
        "ccoc": None,
        "distortion": distortion,
        "rows": rows,
    }


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _df_to_records(df, *, index_name: str | None = None) -> list[dict]:
    """DataFrame → list-of-dicts with the index included.

    Pricing frames are small (≤ 10 rows, ≤ 10 cols) so the verbosity
    of list-of-dicts (vs the wider density_df list-of-lists form) is
    actually a win for readability on the client.
    """
    out: list[dict] = []
    idx_name = index_name or (df.index.name or "index")
    for idx, row in df.iterrows():
        rec: dict = {idx_name: idx}
        for col, val in row.items():
            rec[str(col)] = _scalar(val)
        out.append(rec)
    return out


def _scalar(value):
    """Coerce numpy scalars and non-finite floats for JSON."""
    import math
    import numpy as np

    if value is None:
        return None
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
