"""In-memory LRU cache for built ``Aggregate`` / ``Portfolio`` objects.

The api's "Option X" cache design: a single, bounded, LRU dict
keyed by *content hash* of the DecL program. Same DecL +
``log2`` + ``bs`` → same id → same cached object. Building is
idempotent for the cache lifetime of one server process.

Why bother
----------

Building a moderately sized portfolio takes seconds; FFTs over
2**18 points across many lines aren't free. The cache lets the
SPA's "per-button-fetch UX" (info, describe, stats_df, plot,
kappa, pricing) all run as O(1) lookups against the prebuilt
object, with the heavy lift paid only once per (decl, log2, bs).

Why not lru_cache
-----------------

``functools.lru_cache`` is per-function and doesn't expose the
inspection / list / delete operations the api needs
(``GET /v1/objects``, ``DELETE /v1/objects/{id}``). An
``OrderedDict`` does, and the move-to-end / popitem(last=False)
pair gives plain LRU semantics in ~10 lines.

Thread-safety
-------------

A single ``threading.Lock`` wraps every mutation. The cache is
small (≤50 entries by default), so coarse-grained locking is
cheaper than any per-entry alternative.

Why not weakref
---------------

We want explicit bounded retention, not "alive until nobody
holds a reference" -- the SPA holds the id, not the object,
so weakref would have no live referents and evict immediately.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any


# DecL comments run from ``#`` to end-of-line. We strip them before
# hashing so trivially commented-out / annotated variants of the same
# program produce the same object id.
_COMMENT = re.compile(r"#[^\n]*")


def canonicalize_decl(decl: str) -> str:
    """Strip comments and trailing whitespace for content hashing.

    Note that interior whitespace is *preserved* -- two programs
    that differ only in indentation hash differently. A stronger
    normalization (whitespace-insensitive via Lark tree round-trip)
    is flagged in the plan as a future enhancement.

    Parameters
    ----------
    decl : str
        Raw DecL source.

    Returns
    -------
    str
        Canonicalized form suitable for hashing.
    """
    no_comments = _COMMENT.sub("", decl)
    return no_comments.rstrip()


def object_id(decl: str, log2: int, bs: float) -> str:
    """Compute the cache id for a (decl, log2, bs) triple.

    16-hex-char prefix of SHA-256. Collision probability is
    negligible at our scale (target ≤ 10k builds per session).

    Parameters
    ----------
    decl : str
        Already-canonicalized DecL.
    log2, bs : int, float
        Build knobs that affect the resulting object.
    """
    # ``bs!r`` because bs is a float; repr() pins exact bit pattern
    # so e.g. 0.1 and 0.1000000000001 hash differently (they would
    # build differently too).
    payload = f"{decl}|{log2}|{bs!r}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class CacheEntry:
    """One slot in the LRU.

    ``obj`` is the live ``Aggregate`` or ``Portfolio`` -- the SPA
    never reaches it directly, but the api's per-button endpoints
    pull data off it on demand.

    The other fields are metadata returned by
    ``GET /v1/objects/{id}`` and used by the build endpoint to
    fill out the response without consulting the underlying object.
    """

    obj: Any
    decl: str
    log2: int
    bs: float
    kind: str
    name: str
    created_at: datetime


class ObjectCache:
    """Bounded LRU keyed by content hash.

    Methods are coarsely thread-safe via a single lock. Reads
    move the entry to MRU; writes evict the LRU when full. The
    ``__contains__`` check does *not* move-to-MRU (peek, not
    promote) -- the standard idiom for "is this in cache" before
    a separate get/put decision.
    """

    def __init__(self, max_entries: int = 50) -> None:
        self._max = max_entries
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, oid: str) -> CacheEntry | None:
        """Return the cached entry for ``oid`` (moving it to MRU) or None."""
        with self._lock:
            entry = self._store.get(oid)
            if entry is None:
                return None
            # ``move_to_end`` is the OrderedDict primitive that makes
            # LRU work; entries closest to the front are evicted first.
            self._store.move_to_end(oid)
            return entry

    def put(self, oid: str, entry: CacheEntry) -> None:
        """Insert or refresh ``entry`` under ``oid``, evicting LRU if full."""
        with self._lock:
            if oid in self._store:
                self._store.move_to_end(oid)
                self._store[oid] = entry
                return
            self._store[oid] = entry
            while len(self._store) > self._max:
                # ``last=False`` pops the *least* recently used (front).
                self._store.popitem(last=False)

    def delete(self, oid: str) -> bool:
        """Remove ``oid`` from the cache; return True if it was present."""
        with self._lock:
            return self._store.pop(oid, None) is not None

    def list(self) -> list[CacheEntry]:
        """Return a snapshot of cache contents, MRU last.

        Returned list is a copy of the internal values -- safe to
        iterate without holding the lock.
        """
        with self._lock:
            return list(self._store.values())

    def clear(self) -> None:
        """Drop every entry. Used by tests."""
        with self._lock:
            self._store.clear()

    def __contains__(self, oid: str) -> bool:
        with self._lock:
            return oid in self._store

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
