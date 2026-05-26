"""SQLite-backed audit log for build attempts.

One row per ``POST /v1/objects`` request, regardless of outcome.
The log captures the DecL source, build knobs, status, error
message (if any), elapsed time, client IP, and timestamp.

Why SQLite
----------

* Stdlib (zero extra deps in the ``[api]`` extra).
* WAL mode lets readers and the audit writer not block each other.
* Trivially inspectable from the shell with ``sqlite3 audit.db
  "select * from builds order by ts desc limit 20"``.

The team-deploy assumption is that audit reads are infrequent and
manual; there's no admin route in v1.

Schema design notes
-------------------

* ``object_id`` is NULLABLE because failed builds (parse errors,
  timeouts) never produce one.
* ``kind`` is NULLABLE for the same reason.
* ``elapsed_ms`` is captured even on failure -- a 9000 ms parse
  error is qualitatively different from a 5 ms one.
* ``status`` is a free-form text label (not an enum) so we can
  add new states ('rate_limited' etc.) without a migration.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------
# DDL
# ----------------------------------------------------------------------
# Wrapped in IF NOT EXISTS so :meth:`AuditLog._ensure_schema` is
# idempotent -- every connection re-runs it cheaply and the schema
# survives server restarts without explicit migrations.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS builds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    ip TEXT NOT NULL,
    object_id TEXT,
    kind TEXT,
    decl TEXT NOT NULL,
    log2 INTEGER,
    bs REAL,
    status TEXT NOT NULL,
    error_msg TEXT,
    elapsed_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS builds_ts ON builds(ts);
CREATE INDEX IF NOT EXISTS builds_ip ON builds(ip);
"""


class AuditLog:
    """Append-only SQLite log.

    Connections aren't shared across threads (sqlite3 forbids it
    by default and the cost of a fresh connection per write is
    irrelevant for our volume). A single lock serializes writes
    so concurrent build endpoints don't race on the AUTOINCREMENT
    sequence.

    Set ``journal_mode=WAL`` on each connection so the audit file
    can be inspected with the ``sqlite3`` CLI while the server is
    running.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        # Ensure the parent directory exists. The default audit-db
        # path lives under ``~/.aggregate/api/`` which probably
        # doesn't exist on a fresh install.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Trigger initial schema creation, set WAL mode.
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        """Open a new connection with sensible defaults.

        ``isolation_level=None`` puts the connection in autocommit
        mode -- we use explicit transactions when we want them.
        ``check_same_thread=False`` is *not* set: each call
        produces a fresh connection that lives only for the
        duration of the caller's ``with`` block.
        """
        conn = sqlite3.connect(self.db_path)
        # WAL gives readers a stable snapshot while writes proceed,
        # so ``sqlite3 audit.db`` from the shell never deadlocks.
        conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL trades durability of the last ~few writes for ~10x
        # write throughput -- fine for an audit log.
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def record_build(
        self,
        *,
        ip: str,
        decl: str,
        log2: int | None,
        bs: float | None,
        status: str,
        object_id: str | None = None,
        kind: str | None = None,
        error_msg: str | None = None,
        elapsed_ms: int = 0,
    ) -> None:
        """Append one row.

        Parameters
        ----------
        ip : str
            Client IP from ``request.client.host`` (or ``"-"`` in tests).
        decl : str
            The raw DecL submitted (not the canonicalized form).
        log2, bs : int|None, float|None
            Build knobs as requested -- may be None when omitted.
        status : str
            One of ``'ok' | 'parse_error' | 'build_error' | 'timeout'
            | 'limit_exceeded'``.
        object_id : str|None
            Set on success.
        kind : str|None
            ``'agg'`` or ``'port'`` on success; None on failure.
        error_msg : str|None
            One-line error summary (e.g. ``ErrorReport.message``).
        elapsed_ms : int
            Wall-clock time, including parse + cache check + build.
        """
        # ISO 8601 with UTC; chosen for sortability and unambiguous TZ.
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO builds
                   (ts, ip, object_id, kind, decl, log2, bs, status, error_msg, elapsed_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ts, ip, object_id, kind, decl, log2, bs, status, error_msg, elapsed_ms),
            )
            conn.commit()

    def recent(self, n: int = 100) -> list[dict]:
        """Most recent ``n`` rows, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM builds ORDER BY ts DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def by_ip(self, ip: str, n: int = 100) -> list[dict]:
        """Recent rows from a specific client."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM builds WHERE ip = ? ORDER BY ts DESC LIMIT ?",
                (ip, n),
            ).fetchall()
        return [dict(r) for r in rows]
