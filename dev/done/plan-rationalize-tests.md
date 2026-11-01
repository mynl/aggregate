# Plan: the suite consumes `decl-testers.agg`, and the SLY-parity scaffold retires

`[Rationalize-Tests]`, then `[Scaffold-Retirement]`, the two TODO entries at `dev/TODO.md` 81 to
85. Drafted 2026-10-31 against aggregate 1.43.1 (`03aaec6`). Three steps: two change tests and
corpora only (no bump; the author commits), the third removes a public name and bumps once.
Agreed by the author on 2026-10-30 (`D:\Projects\Using\bugs\aggregate-review-answers-1.43.qmd`,
item 4: "new programs go in `decl-testers.agg` from today; retire the scaffold alongside the
oracle work"). The oracle itself is `[Correctness-Oracle]`, the last step of
`dev/done/plan-lattice-grid-sizing.md`, and is in place before this plan runs. It runs last of the
pre-2.0 plans, after `[Severity-Parameter-Count]`, which rewrites five lines of the scaffold and
regenerates its snapshot.

## Goal

One language-stress corpus, `src/aggregate/agg/decl-testers.agg`, is what the suite parses,
round-trips and snapshots; `_test_suite.agg` and `_test_suite2.agg`, written to prove the Lark
parser matched the retired SLY parser, are deleted with every dependent. Nothing the scaffold
covered is lost: the six grammar rules only it exercises, the recipes tests take from it, and
the change-detecting spec snapshot all move to `decl-testers.agg`.

## Current state (measured 2026-10-31)

| Corpus | Statements | Distinct grammar rules |
|:--|--:|--:|
| `_test_suite.agg` | 209 | 165 |
| `_test_suite2.agg` | 12 | none of its own |
| `decl-testers.agg` | 547 | 261 |
| `library.agg` | 206 | 214 |

`decl-testers.agg` loads clean: `Underwriter(databases='decl-testers').load()` gives 537
recipes, every statement parses, and it needs no other corpus first (section X, the parse-error
fixtures, is commented out at 167 to 170). Three test files say otherwise in stale comments
(`test_decl_unparser.py` 51 to 62, `test_agg_libraries.py` 18 to 21, `test_grammar_ambiguity.py`
108 to 109). The configured default database is already `library` (`config.py` 113), so
retiring the scaffold changes no default.

**The six rules only the scaffold exercises**, with the line that does:

| Rule (`decl.lark`) | `_test_suite.agg` line |
|:--|:--|
| `chf_stable_param` (147) | 378 `chfagg CHF.StableS0 2 * stable 1.5 1 s0 - 2 note{the same law written in S0}` |
| `chf_tweedie_cv` (149) | 394 `chfagg CHF.TweedieCV tweedie 2.5 100 cv 0.5 note{a positive extreme stable member, tilted by its own theta}` |
| `ntt_doutcomes_range_step` (766) | 358 `nttagg NTT.Stepped as "Even Counts" dfreq [2:10:2] dsev [5] [1/1] note{...}` |
| `ntt_multiply` (798) | 362 `nttagg NTT.Arithmetic dfreq [1:3] dsev [1:3] [1/2 (1/2-1/(13*11*(19**3))) (1/(13*11*(19**3)))] note{...}` |
| `numbers_range_step` (967) | 292 `agg O.Logo 1 claim [10:250:10] xs 0 sev lognorm 100 cv 1 fixed` |
| `stable_shift_down` (102) | 429 `stable SN.Copies as "Three copies" 3 @ stable.SN.Plain - 1` (needs `SN.Plain`, 411) |

Seven more rules are exercised by the scaffold and `library.agg` but not by `decl-testers.agg`
(`doutcomes_range_step`, `freq_mixed_two`, `layers_xs_disappearing`, `layers_xs_franchise`,
`reins_list_tower`, `sev_builtin`, `sev_clause_builtin`); `library.agg` is not in the unparser's
corpus list, so they would lose round-trip coverage.

**Every consumer of the scaffold** (a grep for `_test_suite` misses two of them):

- `tests/conftest.py` 11, 28 to 50: `TEST_SUITE_PATH`, the `test_suite_lines` and `underwriter`
  fixtures; the fixture is used by `test_decl_parser`, `test_characteristic_function` (685 to
  1210), `test_ntt` (about 40 functions), `test_elt.py` 178 and 185, `test_yelt.py` 133, all of
  which use it only as a parser or builder for inline programs.
- `tests/test_decl_parser.py` 26 to 27, 119 to 139: 209 parse cases and 209 snapshot cases.
- `tests/capture_spec_snapshot.py` 33 and `tests/data/expected_specs.json`: 209 entries keyed
  by the full preprocessed statement text, so any spacing edit orphans an entry.
- `tests/test_decl_unparser.py` 36, 45 to 46 (`sev.One` from `_test_suite.agg` 32), 191.
- `tests/test_grammar_sync.py` 54 to 55.
- `tests/test_grammar_ambiguity.py` 55 to 59 (`KNOWN_AMBIGUOUS` has `CHF.VG`, only at
  `_test_suite.agg` 398), 89, 131, 133.
- `tests/test_splice_suite.py` 23, 28: the 12 lines of `_test_suite2.agg`, four of them the
  measure-zero splices `Splice02`, `05`, `07`, `10`; section T of `decl-testers.agg` (94 to 98)
  mirrors only the valid shapes.
- `tests/test_characteristic_function.py` 1201 (asserts at least 30 `chfagg` lines across the
  scaffold and `decl-testers.agg`: 14 + 20), 1276, 1371 (build each of the scaffold's 14).
- `tests/test_underwriter.py` 209 to 268, 413, 603 to 641 (`databases='_test_suite'`, the
  filename at 246, `>= 140` recipes, `interpret_file()` with its default).
- `tests/test_session_isolation.py` 144 to 150: `global_build.interpret_file()` with no argument.
- `tests/test_moment_deficit.py` 47: `Underwriter(databases=['library', '_test_suite'])` for
  `agg.J.Re14` (210), `agg.Cb.Freq12.IG` (75, uses `sev.One`), `agg.Cc.Freq24.Binomial` (86).
- `src/aggregate/config.py` 51, 61 to 64, 102 (`TEST_SUITE_FILENAME`, public through
  `__all__`); `underwriter.py` 21, 2577 to 2581 (`test_suite_file`), 3492 to 3517
  (`interpret_file` default; with the file gone it raises `AttributeError` on `None`);
  `data/config.default.toml` 36; `agg/decl-testers.agg` 19 to 20.
- `scripts/bucket_baseline.py` 58 to 62, `scripts/freeze_knowledge.py` 71,
  `scripts/moment_rule_sweep.py` 45, 63, 175.
- `docs/4_dec_Language_Reference.rst` 290 to 317 (the `literalinclude` and the `interpret_file`
  example), `docs/2_aggregate_overview/tests.rst` 169 to 179, 227, 232, `features.rst` 2359 to
  2362.
- `CLAUDE.md` 219 to 224 and the skills `test-tiers` 118 to 128, `version-bump` 77 to 78,
  `execute-plan` 115, which describe the snapshot (skill edits are drafted for the author).
- Docstring-only mentions, no read: `test_agg_as_severity.py` 18, `test_cantor_severity.py` 233,
  `test_feasibility.py` 308, `test_renewal_decl.py` 6, `test_agg_libraries.py` 14 to 15,
  `test_validation.py` 218, `capture_severity_golden.py` 126.

## Steps

### `[Decl-Testers-Coverage]` (tests and corpora only)

- A new section at the end of `decl-testers.agg`, `# TS. Programs moved from the SLY-parity
  scaffold ([Scaffold-Retirement])`, in the file's header convention, holding: the six lines
  above (with `SN.Plain` for the shift-down line); `sev.One` (32) and the three moment-deficit
  recipes; the scaffold's 14 `chfagg` lines (so `CHF.VG` and the 30-line floor keep their
  meaning); and the four measure-zero splices, commented out like section X with the test's
  `INVALID_NAMES` kept inline in `test_splice_suite.py`.
- `library.agg` joins `_CORPUS_FILES` in `test_decl_unparser.py`, which keeps round-trip
  coverage of the seven rules; the stale comments in the three files are corrected.
- The Contents list at 28 to 38 and the duplicate section letters (Y and Z each used twice, at
  173 and 734, 198 and 760) are tidied: the later pair renamed.

*Acceptance.* The grammar-rule sweep (the parse of every statement of every corpus, as in the
report of 2026-10-31) shows every rule of `_test_suite.agg` exercised by `decl-testers.agg` or
`library.agg`; `test_decl_unparser` round-trips 206 more statements; the fast suite passes.

### `[Snapshot-On-Decl-Testers]` (tests only)

- `tests/capture_spec_snapshot.py` reads `decl-testers.agg` and keys each entry by `(kind,
  name)` instead of the statement text, so a spacing or `note{}` edit no longer orphans an
  entry; the value keeps `kind`, `name`, `spec`. Section X stays commented and so is skipped.
- `tests/test_decl_parser.py` parametrizes over `decl-testers.agg`: 547 parse cases and 547
  snapshot cases in place of 209 and 209.
- `tests/test_splice_suite.py` takes its 12 programs inline (8 from section T plus the four
  measure-zero ones), no file read.
- `conftest.py`: `test_suite_lines` goes; `underwriter` becomes a session `Underwriter()`
  with `decl-testers` loaded, which every current user of the fixture is content with (they
  pass inline programs). `test_characteristic_function.py` 1201, 1276 and 1371,
  `test_underwriter.py`, `test_session_isolation.py` 147, `test_moment_deficit.py` 47,
  `test_grammar_sync.py` 54 to 55 and `test_grammar_ambiguity.py` 133 read `decl-testers.agg`
  or name it in `databases=` and `interpret_file(...)`.

*Acceptance.* `expected_specs.json` has 547 entries and no `error`; the fast suite passes with
the scaffold still present but unread by any test (a grep of `tests/` for `_test_suite` finds
docstrings only, and `interpret_file()` is never called without an argument).

### `[Scaffold-Retirement]` (one bump)

- Delete `_test_suite.agg` and `_test_suite2.agg`.
- `config.py`: `TEST_SUITE_FILENAME` removed from the module and from `__all__` (a public name,
  so the CHANGELOG entry carries the breaking-change paragraph); the docstring at 102 and
  `data/config.default.toml` 36 lose the line. `underwriter.py`: `test_suite_file` removed;
  `interpret_file(filename)` requires its argument and its docstring records that; the stale docstrings at 791 to
  803 and 4000 (`examples` by default) corrected to `library`.
- Scripts: the three `DEFAULT_DATABASES` point at `library`; `tests/data/bucket_baseline_*.csv`
  are regenerated or deleted (no test reads them; record which).
- Docs: the `literalinclude` and example in `4_dec_Language_Reference.rst` 290 to 317 switch to
  `decl-testers.agg`; `tests.rst` 169 to 232 describes the new fixtures; `features.rst` 2359 to
  2362 is history and stays. Docs are pending a rebuild by the author.
- `CLAUDE.md` Testing section rewritten for the new corpus and keys; the three skill edits
  drafted as a script for the author.
- `dev/TODO.md`: both entries closed with the version.

*Acceptance.* Tier 3 passes; `aggregate.TEST_SUITE_FILENAME` raises `AttributeError`;
`Underwriter(databases='default')` loads without the scaffold; `available_databases()` lists
`library`, `decl-testers`, `reproductions`, `recitations` and nothing else from the package;
the wheel is 27 KB smaller.

## Not in this plan

- A loader test that `reproductions.agg` and `recitations.agg` build (they are reached only
  through the ambiguity glob today); worth its own small item, `[Reproductions-Build-Check]`.
- Rules no corpus exercises (`answer_expr`, the `conversion_*` family, `builtin_agg_*`,
  `chf_minus`, `ntt_add`, `numbers_range`, the `stable_*` arithmetic forms, `lt_path`,
  `agg_source_ref_port`, `clash_out_nofreq`): listed here for a later coverage pass.

## Execution log

Executed 2026-11-01 against 1.60.1 (`7b03e388`), 17 versions after the draft. **Ruling, 2026-11-01:** Claude commits and pushes all three steps. The draft left the first two for the author, but the third step's bump deletes what they migrate away from, so it could not land until they were committed.

**The review, against 1.60.1.** The premises hold: `[Correctness-Oracle]` landed at 1.48.1 and `[Severity-Parameter-Count]` at 1.51.0, the six rules and their scaffold line numbers are as the table states, and the API, PMIR and the Using site use none of the removed names. A sequential parse and a fully loaded underwriter give the same spec for every statement, so no statement depends on load order. Line numbers elsewhere have moved (sections Y and Z of `decl-testers.agg` are now at 195 and 774, 223 and 800; `test_characteristic_function.py` 1371 is 1400), and every named symbol still exists.

### `[Decl-Testers-Coverage]`

Acceptance measured: every rule of `_test_suite.agg` is exercised by `decl-testers.agg` or `library.agg`, and the list of rules only the scaffold exercises is empty. `decl-testers.agg` has 647 statements and 276 rules. Fast suite: 9026 passed, 3 skipped.

Divergences:

- **The later Y and Z are named for their programs.** The plan asked only that the later pair be renamed. They become `LB` and `IG`, the prefixes their programs already carry, and `test_decl_labels.py` follows.
- **Section headings lose their dashes**, since the Contents list is generated from them and the list now names all 71 sections.
- **The library round trip is 198 faithful, 2 skipped, 6 exempt, not 206.** `Capstone.XOL.PnL`, `Capstone.PnL`, `Capstone.Capped`, `SwingRatedAggCover`, `RetroRatedAccount` and `SlidingScaleCommission` are a pnl or xpnl on a named engine. The parser keeps the referenced recipe's note as `engine_note`, the underwriter strips it before build, and the canonical form inlines the engine with nowhere to write it. All six stay idempotent. They go in `_FIDELITY_EXEMPT`, which had been empty since 1.0.0a231, since this step changes tests only, and `[Unparser-Reference-Gaps]` records the loss.
- **The rule census has moved a little.** Five, not seven, rules are exercised by the scaffold and `library.agg` but not `decl-testers.agg`, since later programs picked up `layers_xs_disappearing` and `layers_xs_franchise`.

### `[Snapshot-On-Decl-Testers]`

Acceptance measured: `expected_specs.json` has 647 entries and no `error`, 443 KB against 156 KB. The 23 programs in both the old and the new snapshot have identical specs. The fast suite passes with the scaffold present and unread: 9678 passed, 3 skipped. A grep of `tests/` for `_test_suite` finds five docstrings and no read, and nothing calls `interpret_file()` without an argument.

Divergences:

- **647 entries, not 547,** since the corpus had grown to 624 statements before section TS added 23.
- **The key is the kind word and name as written, with `#2`, `#3` for a repeat, not `(kind, name)`.** The plan's key cannot hold: seven pairs repeat in section DP, which holds an input and the programs derived from it under one name, so 17 statements share them and `(kind, name)` would silently keep 7. The text key needs no parse at collection, which every xdist worker would pay for. It groups the corpus exactly as the parsed pair does (measured, 637 groups each), and `capture_spec_snapshot.main` asserts that the name agrees. A view word stays in the key (`grossnet agg Name`). Spacing and `note{}` edits leave a key alone, which is what the plan wanted; only within DP does a key also depend on order.
- **27 specs hold objects with no JSON form,** `Copula`, `INHERIT_PREMIUM` and `DERIVE_PREMIUM`, which the plan did not anticipate. `plain_value` gives each a plain form for both the capture and the comparison. A copula is its family and parameter, the two fields `test_decl_unparser` compares.
- **`VALID_KINDS` gains `pnl`, `xpnl` and `bvagg`,** which the scaffold never had.
- **`test_decl_parser` imports from the capture script** rather than repeating its sentinels, so `tests/test_decl_parser.py` and `tests/capture_spec_snapshot.py` cannot disagree. An `error` entry now fails where it used to skip, and `test_snapshot_has_no_orphan_entries` is new, so a removed or renamed program is noticed.
- **The `underwriter` fixture keeps its tolerant loop** rather than `load`, which would turn one typo into an error in every test using the fixture.
- **The twelve splices go inline verbatim from `_test_suite2.agg`.** The plan's "8 from section T" is wrong: section T holds five other programs.
- **The two `chfagg` test bodies now cover 35 statements, not 14,** since they select on the `chfagg ` prefix of `decl-testers.agg`. All 35 pass, so no narrowing to `CHF.` was needed.
- **`test_interpret_file_is_written_on_fork` loads the base first.** It compared against `set(_recipes)` of a lazy base, which is empty on a worker where this test runs first, and a fork loads it. Measured in a scratch run; found while repointing the line, and it is a one-line guard.

Consequence for later plans: a program appended to `decl-testers.agg` fails `test_spec_matches_snapshot` until the snapshot is re-captured, so a re-capture is now routine at a bump. The file's header says so, and `CLAUDE.md` and the skills follow at `[Scaffold-Retirement]`.
