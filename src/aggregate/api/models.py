"""Pydantic v2 schemas for the api request and response bodies.

Each endpoint takes / returns one of these dataclass-like models.
FastAPI validates incoming JSON against the request model and
serializes outgoing responses through the declared return-type
model -- the trip through OpenAPI is automatic.

Flask users: think Marshmallow / pydantic-flask, but tighter --
the model *is* the function parameter type, not a separate
schema you call ``schema.load(request.json)`` on.

Conventions
-----------

* Field names are ``snake_case``.
* All response models opt into ``ConfigDict(extra="forbid")`` so
  the client can rely on the documented field set -- a typo in
  the server code triggers a serialization error instead of
  silently shipping a malformed payload.
* The plan calls for ``InfoResponse.info`` to be a ``dict``,
  but :attr:`aggregate.distributions.Aggregate.info` is a
  multi-line string. We expose the string verbatim; clients can
  display it monospaced. A future structured form is a v1.1
  enhancement.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ----------------------------------------------------------------------
# Shared response config
# ----------------------------------------------------------------------
# Pulled out into a constant so every response model uses identical
# settings. ``extra="forbid"`` enforces that response models only
# carry declared fields (catches accidental leakage of internal data).
_RESPONSE_CFG = ConfigDict(extra="forbid")


# ======================================================================
# Objects -- POST /v1/objects and friends
# ======================================================================

class BuildRequest(BaseModel):
    """Body for ``POST /v1/objects``.

    ``log2`` and ``bs`` are optional -- the underlying ``build()``
    will choose sensible defaults when they're omitted.
    """

    decl: str = Field(..., min_length=1, description="DecL source text.")
    log2: int | None = Field(
        None,
        ge=4,
        description="log2 of the FFT grid size. None means 'let the library pick'.",
    )
    bs: float | None = Field(
        None,
        gt=0,
        description="Bucket size. None means 'let the library pick'.",
    )


class BuildResponse(BaseModel):
    """Slim response so the post-build page doesn't pay for unused data.

    The SPA's per-button buttons (info, describe, plot, etc.) each
    hit their own endpoint. With ``cached=True`` the data calls are
    O(1) -- effectively the same as if the build returned everything
    eagerly, minus the wasted serialization.
    """

    model_config = _RESPONSE_CFG

    id: str
    kind: Literal["agg", "port"]
    name: str
    warnings: list[str] = []
    cached: bool
    elapsed_ms: int


class ObjectSummary(BaseModel):
    """One row in ``GET /v1/objects`` (cache listing)."""

    model_config = _RESPONSE_CFG

    id: str
    kind: str
    name: str
    ts: str  # ISO 8601


class ObjectListResponse(BaseModel):
    """Wrapper for ``GET /v1/objects``."""

    model_config = _RESPONSE_CFG

    objects: list[ObjectSummary]


class ObjectManifest(BaseModel):
    """``GET /v1/objects/{id}`` -- metadata about a single cached object."""

    model_config = _RESPONSE_CFG

    id: str
    kind: str
    name: str
    decl: str
    log2: int
    bs: float
    created_at: str  # ISO 8601


class DeleteResponse(BaseModel):
    model_config = _RESPONSE_CFG

    ok: bool


# ======================================================================
# Tabular endpoints -- describe / stats_df / density_df / kappa
# ======================================================================

class FrameResponse(BaseModel):
    """Pandas DataFrame as ``(columns, rows)``.

    Rows are list-of-lists rather than list-of-dicts so a wide
    density_df (50+ columns, 2**16+ rows) doesn't redundantly carry
    the column name string with every cell. Trim payload by an
    order of magnitude vs the dict-per-row form.
    """

    model_config = _RESPONSE_CFG

    columns: list[str]
    rows: list[list[Any]]


# ======================================================================
# Info -- raw multi-line string from Aggregate.info / Portfolio.info
# ======================================================================

class InfoResponse(BaseModel):
    model_config = _RESPONSE_CFG

    info: str


# ======================================================================
# Pricing
# ======================================================================

class PricingRequest(BaseModel):
    """Body for ``POST /v1/objects/{id}/pricing_at``.

    Either ``(distortion, p|a)`` for full distortion pricing or
    ``(ccoc, p)`` for the constant-CoC shortcut. The server picks
    the dispatch based on which fields are present.
    """

    p: float | None = Field(None, gt=0, lt=1, description="VaR probability.")
    a: float | None = Field(None, gt=0, description="Asset level.")
    ccoc: float | None = Field(None, gt=0, description="Constant cost of capital.")
    distortion: str | None = Field(
        None, description="Calibrated distortion name (looked up on the Portfolio)."
    )


class PricingResponse(BaseModel):
    """Per-line breakdown plus headline totals."""

    model_config = _RESPONSE_CFG

    a: float | None
    p: float | None
    ccoc: float | None
    distortion: str | None
    rows: list[dict[str, Any]]


# ======================================================================
# DecL helpers
# ======================================================================

class DeclCompleteRequest(BaseModel):
    decl: str
    cursor: int = Field(..., ge=0)


class Completion(BaseModel):
    model_config = _RESPONSE_CFG

    label: str
    terminal: str
    kind: Literal["keyword", "identifier", "literal"]


class CompletionsResponse(BaseModel):
    model_config = _RESPONSE_CFG

    completions: list[Completion]


class DeclLexRequest(BaseModel):
    decl: str


class LexToken(BaseModel):
    model_config = _RESPONSE_CFG

    type: str
    value: str
    start: int
    end: int
    line: int
    column: int


class LexResponse(BaseModel):
    model_config = _RESPONSE_CFG

    tokens: list[LexToken]


# ======================================================================
# Examples (test_suite.agg)
# ======================================================================

class ExampleItem(BaseModel):
    model_config = _RESPONSE_CFG

    name: str
    decl: str
    note: str | None


class ExampleCategory(BaseModel):
    model_config = _RESPONSE_CFG

    letter: str
    title: str
    items: list[ExampleItem]


class ExamplesResponse(BaseModel):
    model_config = _RESPONSE_CFG

    categories: list[ExampleCategory]


# ======================================================================
# Meta / health
# ======================================================================

class HealthResponse(BaseModel):
    model_config = _RESPONSE_CFG

    ok: bool
    version: str


class MetaResponse(BaseModel):
    model_config = _RESPONSE_CFG

    version: str
    log2_cap: int
    log2_default: int
    build_timeout_s: float
    cache_max: int
    plot_default_format: str
