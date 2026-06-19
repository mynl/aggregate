"""One-shot migration for the blank-line / `;` DecL statement rule.

Under the new rule a comment is *transparent* (never separates statements), so
a statement must be separated from the next by a blank line or a `;`. For each
corpus `.agg` file this script:

1. Removes trailing `\\` line-continuations (the rule is gone).
2. Terminates a statement with `;` unless a blank line already separates it from
   the next statement. Concretely: scan forward from the statement's last line;
   if a blank line is reached first the statement is already separated; if the
   next statement-start is reached through only comment lines (or immediately),
   append `;`. The `;` is inserted before any trailing comment on the line.

Statement boundaries use the OLD column-0 layout rule (a non-indented,
non-comment line starts a statement; indented lines continue it). The migration
is spec-neutral -- `preprocess` strips the `;` again -- so the parsed specs (and
the SLY snapshot) are unchanged.

Run from the repo root:  uv run python dev/migrate_decl_newline.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGG_DIR = ROOT / "src" / "aggregate" / "agg"
FILES = [
    "test_suite.agg", "test_suite2.agg", "test_decl.agg",
    "spa_examples.agg", "spa_examples-old.agg",
    "other-distributions.agg", "testers.agg", "examples.agg",
]

_TRAILING_BACKSLASH = re.compile(r"[ \t]*\\[ \t]*$")


def _role(line: str) -> str:
    s = line.strip()
    if s == "":
        return "blank"
    if s.startswith("#") or s.startswith("//"):
        return "comment"
    if line[:1] in (" ", "\t"):
        return "cont"
    return "start"


def _split_trailing_comment(line: str) -> tuple[str, str]:
    """Split a line into (code, trailing-comment), ignoring # / // inside {...}.

    A ``//`` inside a ``note{http://x}`` is brace-nested and is not a comment.
    """
    depth = 0
    for i, c in enumerate(line):
        if c == "{":
            depth += 1
        elif c == "}":
            depth = max(0, depth - 1)
        elif depth == 0 and (c == "#" or line[i:i + 2] == "//"):
            return line[:i], line[i:]
    return line, ""


def _terminate(line: str) -> str:
    code, cmt = _split_trailing_comment(line)
    if code.rstrip().endswith(";"):
        return line
    out = code.rstrip() + ";"
    if cmt:
        out += "  " + cmt
    return out


def migrate(text: str) -> str:
    lines = [_TRAILING_BACKSLASH.sub("", ln).rstrip() for ln in text.split("\n")]
    roles = [_role(ln) for ln in lines]
    n = len(lines)

    j = 0
    while j < n:
        if roles[j] == "start":
            k = j
            while k + 1 < n and roles[k + 1] == "cont":
                k += 1
            # Scan past the statement: a blank line means it is already
            # separated; reaching the next statement-start through only comment
            # lines means it needs a `;`.
            m = k + 1
            need_semi = False
            while m < n:
                if roles[m] == "blank":
                    break
                if roles[m] == "start":
                    need_semi = True
                    break
                m += 1  # comment line -> keep scanning
            if need_semi:
                lines[k] = _terminate(lines[k])
            j = k + 1
        else:
            j += 1

    return "\n".join(lines)


def main() -> None:
    for fn in FILES:
        path = AGG_DIR / fn
        if not path.exists():
            print(f"skip (missing): {fn}")
            continue
        before = path.read_text(encoding="utf-8")
        after = migrate(before)
        if after != before:
            path.write_text(after, encoding="utf-8")
            print(f"migrated {fn}: +{after.count(';') - before.count(';')} terminators")
        else:
            print(f"unchanged: {fn}")


if __name__ == "__main__":
    main()
