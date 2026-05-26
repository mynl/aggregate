"""Example-library route.

Single endpoint: ``GET /v1/examples`` -- returns the parsed
``test_suite.agg`` payload grouped by category letter.

The examples loader is cached at module level (``lru_cache``) so
the test suite is parsed once per process. The route is a thin
wrapper.
"""

from __future__ import annotations

from fastapi import APIRouter

from .. import models
from ..examples import load_examples


router = APIRouter()


@router.get("/examples", response_model=models.ExamplesResponse)
def list_examples() -> dict:
    """Return the categorized example library.

    See :class:`ExamplesResponse` for the payload shape.
    """
    return load_examples()
