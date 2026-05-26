"""Parser for ``aggregate/agg/test_suite.agg`` → grouped DecL examples.

The bundled test suite doubles as an example library for the SPA's
dropdown menus. Each non-comment, non-blank line is a runnable DecL
program. Names follow the convention ``<Letter>.<Name>`` (e.g.
``A.Dice00``, ``G.Mixed03``); the letter prefix identifies the
category.

Categories are sourced from the "Contents" block at the top of the
file, which lists ``# A. Title``, ``# B. Title`` etc. -- one for each
letter the body uses.

Cached as a module-level dict; reloaded only on server restart.

Returned shape mirrors :class:`ExamplesResponse` in
``models.py``::

    {
        "categories": [
            {
                "letter": "A",
                "title": "Creating Aggregates...",
                "items": [
                    {"name": "A.Dice00", "decl": "...", "note": "..."}
                ]
            },
            ...
        ]
    }
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib.resources import files

# Lines like ``# A. Creating Aggregates, Portfolios, and Distortion objects``
# from the Contents block. Capture letter + title.
_CONTENTS_LINE = re.compile(r"^#\s+([A-O])\.\s+(.+)$")

# An item line. Must start with a DecL top-level keyword so we don't
# mistake a comment-stripped section header for a program. The known
# top-level kinds are agg, sev, port, dist, and bare ``expr`` (rare
# in test_suite, ignored).
_ITEM_LINE = re.compile(
    r"^(agg|sev|port|dist)\s+([A-O])\.([A-Za-z0-9_.\-]+)\s+(.*)$"
)

# ``note{...}`` trailing annotation. Allowed to span the rest of the
# line. Captured greedily up to the closing brace.
_NOTE = re.compile(r"\s*note\{([^}]*)\}\s*$")


def _load_contents(text: str) -> dict[str, str]:
    """Parse the Contents block at the top of the file.

    Walks every line and picks out ``# X. Title`` rows -- the
    Contents block lives at the top but the parser is content with
    any matching line position, so reordering the file won't break
    the lookup.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = _CONTENTS_LINE.match(line)
        if m:
            letter, title = m.group(1), m.group(2).strip()
            # Don't let a later duplicate clobber the first hit.
            out.setdefault(letter, title)
    return out


def _load_items(text: str) -> dict[str, list[dict]]:
    """Walk the body for item lines and group by letter prefix.

    Each item carries the original DecL minus the trailing ``note{...}``
    so the client can re-evaluate it directly via ``POST /v1/objects``.
    """
    grouped: dict[str, list[dict]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ITEM_LINE.match(line)
        if not m:
            # Lines that don't match the convention (no Letter.Name)
            # are kept out of the example library; they still parse
            # fine as DecL, but aren't surface-able via the categorized
            # dropdown.
            continue
        kind, letter, _suffix, _body = m.groups()
        # Split off trailing note{...} for the dedicated `note` field.
        note_match = _NOTE.search(line)
        if note_match:
            note = note_match.group(1).strip()
            decl = _NOTE.sub("", line).rstrip()
        else:
            note = None
            decl = line
        # The full Letter.Name name lives at groups (letter, suffix);
        # rebuild as ``letter + '.' + suffix`` for round-trip clarity.
        name = f"{letter}.{m.group(3)}"
        grouped.setdefault(letter, []).append(
            {"name": name, "decl": decl, "note": note}
        )
    return grouped


def _read_suite_text() -> str:
    """Locate test_suite.agg via package resources and return its text."""
    resource = files("aggregate").joinpath("agg/test_suite.agg")
    # ``importlib.resources`` traversables expose .read_text() for files.
    return resource.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def load_examples() -> dict:
    """Return the cached examples payload.

    Cached at the module level via ``lru_cache``; the test suite is
    parsed once per server process. To pick up edits to test_suite.agg
    without restarting, call ``load_examples.cache_clear()``.

    Returns
    -------
    dict
        Matches :class:`aggregate.api.models.ExamplesResponse`:
        ``{"categories": [...]}``.
    """
    text = _read_suite_text()
    titles = _load_contents(text)
    items_by_letter = _load_items(text)
    # Build the output in letter order so the SPA dropdown is
    # alphabetically consistent. Letters appearing only in titles
    # but with no items get an empty list; letters with items but
    # no title fall back to "Section <Letter>".
    letters = sorted(set(titles) | set(items_by_letter))
    categories = [
        {
            "letter": letter,
            "title": titles.get(letter, f"Section {letter}"),
            "items": items_by_letter.get(letter, []),
        }
        for letter in letters
    ]
    return {"categories": categories}
