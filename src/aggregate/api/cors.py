"""CORS middleware setup.

The web SPA (Plan D) can be served from the same origin as the
api (via the conditional ``StaticFiles`` mount in ``app.py``) or
from a different origin -- e.g. ``mynl.com/aggregate/`` calling
``api.mynl.com``. The second case needs the browser-side CORS
preflight handshake to succeed; this helper attaches the
:class:`fastapi.middleware.cors.CORSMiddleware` when the operator
has declared which origins are trusted.

Same-origin deploys don't pay the per-request CORS handling
overhead: empty ``allowed_origins`` → middleware skipped entirely.

Flask users: FastAPI middleware is closer to Starlette/ASGI
middleware than to ``@app.before_request``. The order of
``add_middleware`` calls matters -- the *last* one added is the
*outermost* in the request lifecycle.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def install_cors(app: FastAPI, allowed_origins: list[str]) -> None:
    """Attach ``CORSMiddleware`` if ``allowed_origins`` is non-empty.

    Parameters
    ----------
    app : FastAPI
        The application being configured.
    allowed_origins : list[str]
        Exact-match origins (``["http://localhost:5173",
        "https://mynl.com"]``). No regex / wildcard support in v1
        -- the team-deploy use case is two or three known origins.

    Notes
    -----
    * ``allow_credentials=False`` because v1 has no cookies / auth.
      With credentials enabled the browser also bars ``*`` in
      ``allow_origins``; explicit exact-match avoids that footgun.
    * ``allow_methods`` lists only what the api uses; OPTIONS is
      handled automatically by CORSMiddleware for preflights.
    * ``allow_headers`` is conservative -- if a future endpoint
      needs ``Authorization`` etc., extend this list.
    """
    if not allowed_origins:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
