"""FastAPI service for the :mod:`aggregate` library.

This subpackage stands up an HTTP/JSON wrapper around ``build()``,
the live ``Aggregate`` / ``Portfolio`` objects, plotting, pricing,
and DecL helpers (completions, lexing, grammar).

Quickstart
----------

::

    pip install 'aggregate[api]'
    aggregate-api --port 8001

Then ``POST http://127.0.0.1:8001/v1/objects`` with body
``{"decl": "agg Dice dfreq [3] dsev [1:6]"}`` to build and cache an
object, and follow up with ``GET /v1/objects/{id}/info`` etc.

Public surface
--------------

``create_app()`` returns a configured :class:`fastapi.FastAPI`
instance. The ``aggregate-api`` console script launches it under
uvicorn. Tests build their own ``TestClient`` against
``create_app()``.

Note for Flask users
--------------------

FastAPI uses an *application factory* (``create_app``) plus an ASGI
server (uvicorn) instead of Flask's ``app = Flask(__name__)`` + WSGI.
Routes are grouped on ``APIRouter`` objects (analogous to Flask
``Blueprint``) and mounted onto the app in :func:`app.create_app`.
Request bodies are *validated* by Pydantic models (one model per
endpoint) rather than read out of ``request.json``; the model is a
function parameter and FastAPI deserializes for you.
"""

from .app import create_app

__all__ = ["create_app"]
