"""Capture (kind, name, spec) output from the parser for every statement of
decl-testers.agg, writing it to tests/data/expected_specs.json.

The snapshot is a change detector, not a correctness oracle. It is regenerated
from the same parser that tests/test_decl_parser.py checks it against, so it
cannot prove the parser right; what it does is make any grammar or transformer
edit show its full blast radius as a diff you have to look at and accept
deliberately. dev/done/refactor-plan.md calls it the spec-shape canary.

Re-capture after any change that moves the parse of an existing statement, then
read the diff: a statement you did not expect to move is a finding. Adding a
program to the corpus also needs a re-capture, whose diff is the new entries
alone, since an entry is keyed by the words that name its statement and not by
the statement's text (see statement_keys).

Named capture_sly_snapshot.py until 1.0.0a231, from the migration that first
produced it. Nothing here has touched SLY since the Lark parser landed; the
imports below are the current parser.

Run with:
    uv run python tests/capture_spec_snapshot.py
"""

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from aggregate.copula import Copula
from aggregate.parser import DERIVE_PREMIUM, INHERIT_PREMIUM, UnderwritingLexer
from aggregate.underwriter import Underwriter

REPO_ROOT = Path(__file__).parent.parent
CORPUS = REPO_ROOT / "src" / "aggregate" / "agg" / "decl-testers.agg"
OUT = REPO_ROOT / "tests" / "data" / "expected_specs.json"

POS_INF_SENTINEL = "__inf__"
NEG_INF_SENTINEL = "__-inf__"
NAN_SENTINEL = "__nan__"

# The words that can stand before a kind word, as in ``grossnet agg Name``.
VIEW_WORDS = {"grossnet", "grossceded", "netceded"}


def statement_keys(lines: list[str]) -> list[str]:
    """Return the snapshot key of each preprocessed statement.

    Parameters
    ----------
    lines : list of str
        Preprocessed statements, in file order.

    Returns
    -------
    list of str
        One key per statement: the kind word and the name as written, then
        ``#2``, ``#3`` for a later statement with the same head.

    Notes
    -----
    The key comes from the text so that ``tests/test_decl_parser.py`` can
    parametrize without parsing the corpus at collection, which every xdist
    worker would pay for. It groups the corpus exactly as the parsed
    ``(kind, name)`` does, and ``main`` asserts that the name agrees. A spacing or
    ``note{}`` edit leaves the key alone. The repeats are section DP, which holds
    an input and the programs derived from it under one name, so only there does
    a key also depend on order.
    """
    seen: dict[str, int] = {}
    keys = []
    for line in lines:
        words = line.split()
        head = " ".join(words[:3] if words[0] in VIEW_WORDS else words[:2])
        seen[head] = seen.get(head, 0) + 1
        keys.append(head if seen[head] == 1 else f"{head}#{seen[head]}")
    return keys


def plain_value(obj):
    """Return a premium sentinel or a copula as plain data, and anything else unchanged.

    A parsed spec holds these objects themselves, and neither has a JSON form. A
    copula becomes its family and parameter, the two fields
    ``tests/test_decl_unparser.py`` compares.
    """
    if obj is INHERIT_PREMIUM or obj is DERIVE_PREMIUM:
        return repr(obj)
    if isinstance(obj, Copula):
        return {"copula": obj._name, "param": obj.param}
    return obj


def jsonify(obj):
    obj = plain_value(obj)
    if isinstance(obj, Fraction):
        # exact nttagg probabilities; tests/test_decl_parser.py mirrors this
        # "a/b" string form in its _normalize
        return f"{obj.numerator}/{obj.denominator}"
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
    lines = UnderwritingLexer.preprocess(CORPUS.read_text(encoding="utf-8"))

    out: dict[str, dict] = {}
    errors = 0
    for key, line in zip(statement_keys(lines), lines):
        try:
            kind, name, spec = uw.parser.parse(uw.lexer.tokenize(line))
        except Exception as e:
            out[key] = {"error": repr(e)}
            errors += 1
            continue
        assert key.split("#")[0].split()[-1] == name, (key, name)
        out[key] = {"kind": kind, "name": name, "spec": jsonify(spec)}
        # Populate the recipe base so builtin lookups (sev.X, agg.X) resolve.
        uw.add_recipe(kind, name, spec, line)

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} entries to {OUT.relative_to(REPO_ROOT)}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
