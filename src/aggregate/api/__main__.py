"""Entry point for ``python -m aggregate.api`` and ``aggregate-api``.

This is the bootstrap that turns the :func:`create_app` factory
into a running uvicorn process. CLI flags (``--host``, ``--port``,
``--reload``) override the env-var-driven config; with no flags
the server picks up everything from ``AGGAPI_*``.

Flask users
-----------

Equivalent to ``flask run`` -- but with the production server
(uvicorn) baked in. There's no equivalent of Flask's dev/prod
switch; uvicorn is fast enough to be both, and ``--reload``
gives you the dev-mode file-watching behavior.
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Parse CLI flags and launch uvicorn."""
    parser = argparse.ArgumentParser(
        prog="aggregate-api",
        description="Launch the aggregate api (FastAPI + uvicorn).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Bind address (defaults to AGGAPI_HOST, then 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port (defaults to AGGAPI_PORT, then 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on source changes (dev only -- prohibits "
             "multi-worker mode and adds a watchgod thread).",
    )
    args = parser.parse_args()

    # Import inside ``main`` so ``aggregate-api --help`` doesn't pay
    # the cost of loading FastAPI / uvicorn / aggregate.
    import uvicorn

    from .config import get_settings

    settings = get_settings()
    # The ``factory=True`` flag tells uvicorn that the target is a
    # *callable* returning an app rather than an app instance --
    # so we hand it ``aggregate.api.app:create_app`` and it calls
    # the factory itself. This plays nicely with --reload: the
    # factory re-runs on each worker restart, picking up code edits.
    uvicorn.run(
        "aggregate.api.app:create_app",
        factory=True,
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
