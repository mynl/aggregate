"""Regenerate ``docs/extract.bib`` from the author's master BibTeX library.

The Sphinx docs cite works via ``:cite:`` roles. Read the Docs builds on
Ubuntu runners with no access to the author's local master library at
``C:/S/TELOS/Biblio/uber-library.bib``, so ``bibtex_bibfiles`` must list only
repo-local files. This script bridges that gap: it scans every cite key used in
``docs/**/*.rst``, pulls the matching entries out of the master library, and
writes them into the committed ``docs/extract.bib``. Because the result is
committed, RTD builds offline with every reference present.

Keys that intentionally do not live in the master library (a handful of
non-uber academic works plus the software citations) are owned by the
hand-maintained ``docs/manual.bib`` and are skipped here.

Usage
-----
Run on the author's machine, where the master library is available::

    uv run python docs/update_extract_bib.py

Options
-------
``--uber <path>``
    Override the master-library path (default
    ``C:/S/TELOS/Biblio/uber-library.bib``; also overridable via the
    ``UBER_LIBRARY`` environment variable). The library is only ever read,
    never written (CLAUDE.md standing order).

Exit status
-----------
Exits non-zero if any cited key is neither found in the master library nor
owned by ``manual.bib`` — a genuinely broken citation, caught before commit.
Also warns (without failing) on keys that resolve to more than one entry.

Notes
-----
BibTeX entries are captured by locating ``@type{key,`` and brace-balancing to
the matching closing ``}``. This handles nested braces in titles and author
lists robustly, which a naive line- or regex-based extractor does not.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

DEFAULT_UBER = Path("C:/S/TELOS/Biblio/uber-library.bib")

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE
EXTRACT_BIB = DOCS_DIR / "extract.bib"
MANUAL_BIB = DOCS_DIR / "manual.bib"

# Matches :cite:`...`, :cite:t:`...`, :cite:p:`...`; the backtick content may be
# a comma-separated list of keys.
CITE_ROLE = re.compile(r":cite:(?:[tp]:)?`([^`]+)`")
# Matches the start of a BibTeX entry: @type{key,
ENTRY_START = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,")


def find_cite_keys(docs_dir: Path) -> dict[str, list[str]]:
    """Collect every cite key used in ``docs/**/*.rst``.

    Returns
    -------
    dict
        Maps each key to the sorted list of files that cite it, so unresolved
        keys can be reported with their source location.
    """
    keys: dict[str, set[str]] = {}
    for rst in sorted(docs_dir.rglob("*.rst")):
        text = rst.read_text(encoding="utf-8")
        for match in CITE_ROLE.finditer(text):
            for raw in match.group(1).split(","):
                key = raw.strip()
                if key:
                    keys.setdefault(key, set()).add(str(rst.relative_to(docs_dir)))
    return {k: sorted(v) for k, v in sorted(keys.items())}


def read_bib_keys(bib_path: Path) -> set[str]:
    """Return the set of entry keys defined in a ``.bib`` file."""
    if not bib_path.exists():
        return set()
    text = bib_path.read_text(encoding="utf-8")
    return {m.group(2) for m in ENTRY_START.finditer(text)}


def extract_entries(uber_text: str) -> dict[str, str]:
    """Index the master library: map each key to its full entry text.

    Each entry runs from ``@type{`` to the brace-balanced closing ``}``.
    """
    entries: dict[str, str] = {}
    for match in ENTRY_START.finditer(uber_text):
        key = match.group(2)
        # Brace-balance from the opening brace after @type to find the entry end.
        open_brace = uber_text.index("{", match.start())
        depth = 0
        i = open_brace
        n = len(uber_text)
        while i < n:
            char = uber_text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        entry = uber_text[match.start():i + 1]
        if key in entries:
            print(f"WARNING: key '{key}' appears more than once in the master "
                  f"library; using the last occurrence.", file=sys.stderr)
        entries[key] = entry
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--uber",
        default=os.environ.get("UBER_LIBRARY", str(DEFAULT_UBER)),
        help="Path to the master BibTeX library (read-only).",
    )
    args = parser.parse_args(argv)

    uber_path = Path(args.uber)
    if not uber_path.exists():
        print(f"ERROR: master library not found: {uber_path}", file=sys.stderr)
        return 2

    cited = find_cite_keys(DOCS_DIR)
    manual_keys = read_bib_keys(MANUAL_BIB)
    uber_entries = extract_entries(uber_path.read_text(encoding="utf-8"))

    print(f"Scanned docs: {len(cited)} distinct cite keys.")
    print(f"manual.bib owns: {len(manual_keys)} keys.")
    print(f"Master library: {len(uber_entries)} entries indexed.")

    wanted = {k: v for k, v in cited.items() if k not in manual_keys}

    pulled: dict[str, str] = {}
    missing: dict[str, list[str]] = {}
    for key, files in wanted.items():
        if key in uber_entries:
            pulled[key] = uber_entries[key]
        else:
            missing[key] = files

    if missing:
        print("\nERROR: the following cited keys are in neither the master "
              "library nor manual.bib:", file=sys.stderr)
        for key in sorted(missing):
            print(f"  {key}  (cited in: {', '.join(missing[key])})", file=sys.stderr)
        print("\nFix the cite, add the key to manual.bib, or add it to the "
              "master library via archivum.", file=sys.stderr)
        return 1

    header = (
        "% docs/extract.bib — GENERATED FILE, DO NOT EDIT BY HAND.\n"
        "%\n"
        f"% Source: {uber_path}\n"
        "% Regenerate with: uv run python docs/update_extract_bib.py\n"
        "%\n"
        "% Contains exactly the master-library entries cited in docs/**/*.rst.\n"
        "% Non-uber academic works and software citations live in docs/manual.bib.\n"
        "%\n"
    )
    body = "\n".join(pulled[key] for key in sorted(pulled))
    EXTRACT_BIB.write_text(header + "\n" + body + "\n", encoding="utf-8")

    print(f"\nWrote {EXTRACT_BIB.relative_to(HERE.parent)}: "
          f"{len(pulled)} entries from the master library.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
