"""Strip DecL ``\\`` line-continuations from the four reference pages whose
trailing backslashes are all DecL (not LaTeX ``\\\\`` or Python continuations).

Run from repo root:  uv run python dev/strip_decl_backslash_docs.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    "docs/2_user_guides/DecL/010_Aggregate.rst",
    "docs/2_user_guides/DecL/065_limit_profiles.rst",
    "docs/2_user_guides/DecL/070_vectorization.rst",
    "docs/2_user_guides/DecL/080_reinsurance.rst",
]
TRAILING = re.compile(r"[ \t]*\\[ \t]*$")

for f in FILES:
    p = ROOT / f
    lines = p.read_text(encoding="utf-8").split("\n")
    # Guard: none of these files should carry a LaTeX row-break ``\\``.
    assert not any(ln.rstrip().endswith("\\\\") for ln in lines), f
    n = sum(1 for ln in lines if TRAILING.search(ln))
    out = [TRAILING.sub("", ln) for ln in lines]
    p.write_text("\n".join(out), encoding="utf-8")
    print(f"{f}: stripped {n}")
