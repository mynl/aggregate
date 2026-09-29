# Execution notes: plan-bv-hints-and-dfreq-window

Executed 2026-09-29 against 1.0.0a353. Divergences recorded at the moment they
were made; anything not listed here landed exactly as the plan specifies.

## Workstream [Hints-Tuple-Values] (1.0.0a354)

* **Divergence, corpus line moved out of the snapshot corpus.** The plan asked
  for one `bv ... hints{log2=(9,12); bs=(3,1);}` line in
  `src/aggregate/agg/_test_suite.agg` with a snapshot regen. That corpus cannot
  hold a `bv` at all: `tests/test_decl_parser.py` restricts kinds to
  `{agg, sev, port, distortion, expr}`, and a bv spec embeds a `Copula` object
  that neither `capture_spec_snapshot.jsonify` nor the test's `_normalize` can
  serialize (the capture script crashes with `TypeError: Object of type
  CopulaIndependent is not JSON serializable`). Extending the snapshot harness
  would buy nothing for this feature: `spec['hints']` is raw trailer text at
  parse time, so the pair spelling is invisible to the parser snapshot. The
  corpus line landed instead as `bv HINT.PairBv ...` in the HINTS section of
  `src/aggregate/agg/decl-testers.agg`, which is round-tripped by
  `tests/test_decl_unparser.py` and lexed by `tests/test_grammar_sync.py`;
  `tests/test_hints.py` carries the unit and end-to-end coverage.
  `tests/data/expected_specs.json` is therefore unchanged.
* **Detail, corpus file name.** The plan (and CLAUDE.md) say
  `aggregate/agg/test_suite.agg`; the file on disk is
  `src/aggregate/agg/_test_suite.agg`.
* **Detail, helper added.** The scalar coercion ladder (int, float, `a/b`
  fraction) was extracted into a private `_coerce_hint_scalar` so the pair and
  scalar paths share it; the plan named only `_HINT_PAIR_RE`. No public names.
* **Divergence, one file beyond the plan's list.** The syntax-highlighting
  lexer `src/aggregate/decl_pygments.py` (kept in sync with the grammar by
  `tests/test_grammar_sync.py`) had no comma inside its `hints` state, so the
  new pair spelling lexed with error tokens. The comma joined `=` and `;` as
  punctuation there. Found by the grammar-sync suite, which is the mechanism
  working as designed.
* **Detail, docs.** The pair form was documented in
  `docs/2_aggregate_overview/underwriter.rst` (the trailer reference table).
  `docs/2_aggregate_overview/features.rst` also mentions hints but is dirty
  with the author's in-flight [task-features] work, so it was left untouched.

## Workstream [Bv-Dfreq-Outer-Window] (1.0.0a355)

(To be filled as the workstream executes.)
