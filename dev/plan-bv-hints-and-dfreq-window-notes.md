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

* **Divergence, the dfreq [2] variant's expected mean.** The plan's test
  bullet says the standalone measurement marginal has mean 10.0; the correct
  number is 5.0 (2 outer events, each 5 claims of cantor mean 1/2). Verified
  by experiment and against `_marginal_moments`, which returns (5.0, 1.118,
  0.0) for that axis; the test asserts 5.0 and the agreement with the
  analytic mean.
* **Divergence, the library entry carries an explicit `copula independent`.**
  The plan's entry has no copula clause. Without one, the grammar binds the
  trailing `note{}` to unit B rather than to the bivariate (the bv body has
  no closing clause in the dfreq form), and the canonical-layout lint
  (`test_library_is_written_in_the_canonical_layout`) renders the clause
  explicitly. The entry also spells `bivariate`, matching every neighbor in
  the section, where the plan wrote the `bv` alias.
* **Confirmed at implementation, as the plan directed.** `_marginal_moments`
  returns mean 2.5 for the CantorArt reproducing axis, so it is a valid guard
  target; the shipped `CantorArt` entry builds by name lookup with full mass,
  marginal mean 2.0, and no warning.

## Finding left for the author (pre-existing, not part of this plan)

Under the numerics gate (`-W error::RuntimeWarning`) one case fails:
`test_library_entries.py::test_every_library_entry_builds[agg:RenewalDeterministicWait]`
hits `RuntimeWarning: invalid value encountered in sqrt` at
`src/aggregate/moments.py:500` (`sd = np.sqrt(v)` on a negative central
variance). The entry landed at 1.0.0a327 and `moments.py` has not moved since
the a220 census, and nothing in this plan touches that path, so the warning
predates this work and was simply exposed by running the gate here. The case
passes without the flag. Per the census policy this is a real finding: either
the negative variance wants an `np.errstate` guard with a Notes paragraph, or
a NaN is reaching an answer and wants a fix.
