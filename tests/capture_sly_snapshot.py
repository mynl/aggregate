"""Capture (kind, name, spec) output from the current SLY parser for every line
of test_suite.agg, writing it to tests/data/expected_specs.json.

This script is a migration artifact: it produces the regression snapshot that
the Lark-based parser is verified against. Once the migration is complete and
the snapshot is in place, the script's only future use is regenerating the
snapshot if test_suite.agg changes meaningfully.

Run with:
    uv run python tests/capture_sly_snapshot.py
"""

import json
import math
from pathlib import Path

import numpy as np

from aggregate.parser import UnderwritingLexer
from aggregate.underwriter import Underwriter

REPO_ROOT = Path(__file__).parent.parent
TEST_SUITE = REPO_ROOT / "aggregate" / "agg" / "test_suite.agg"
OUT = REPO_ROOT / "tests" / "data" / "expected_specs.json"

POS_INF_SENTINEL = "__inf__"
NEG_INF_SENTINEL = "__-inf__"
NAN_SENTINEL = "__nan__"


def jsonify(obj):
    if isinstance(obj, np.ndarray):
        return [jsonify(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(x) for x in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return jsonify(obj.item())
    if isinstance(obj, float):
        if math.isinf(obj):
            return POS_INF_SENTINEL if obj > 0 else NEG_INF_SENTINEL
        if math.isnan(obj):
            return NAN_SENTINEL
        return obj
    return obj


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    uw = Underwriter()
    lex = UnderwritingLexer()
    lines = lex.preprocess(TEST_SUITE.read_text(encoding="utf-8"))

    out: dict[str, dict] = {}
    errors = 0
    for line in lines:
        try:
            kind, name, spec = uw.parser.parse(uw.lexer.tokenize(line))
        except Exception as e:
            out[line] = {"error": repr(e)}
            errors += 1
            continue
        out[line] = {"kind": kind, "name": name, "spec": jsonify(spec)}
        # Populate knowledge so subsequent builtin lookups (sev.X, agg.X) resolve.
        uw._knowledge.loc[(kind, name), :] = [spec, line]

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} entries to {OUT.relative_to(REPO_ROOT)}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
