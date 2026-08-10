# Plan: [Single-Placement-Keyword] one way to write a partial placement

> **Status: EXECUTED at 1.0.0a249.** Both author decisions came back as recommended: `po` is the survivor, and `so` / `of` were removed hard with no migration stop. Retire the `so` and `of` spellings of a partial placement and leave `po` as the only one. No semantic change: the three rules are already arithmetically identical. The visible change is that the unparser's canonical form flips from `50% so 100 xs 0` to `50% po 100 xs 0`, which moves machine generated P&L step labels and therefore a lot of test string assertions.
>
> **Outcome.** 3891 fast tests pass, 160 of 161 `slow` pass (`test_massive_pnl_one_sweep_ledger` fails identically at `HEAD`, confirmed by stashing, so it is pre-existing and unrelated). The load-bearing check held: the spec snapshot re-keys and **no spec value moved**. Two deviations from the plan as written, both recorded in the phases below: `tests/data/bucket_baseline_summary.csv` could not be regenerated because `scripts/bucket_baseline.py` is broken independent of this change, and the `features.rst` ledger gained no new row because it already stops at a195 and is maintained by `dev/task-features.md`. The new `po` warning found a live bug in the monograph while the work was in flight.

---

## Why

DecL currently offers three keywords for the same thing. `src/aggregate/parser.py:1434-1466` holds `reins_clause_share` (`so`), `reins_clause_of` (`of`) and `reins_clause_part` (`po`), and all three compute the same tuple: a `%` literal is taken as the share directly, a bare number is taken as an absolute amount and the share is `n / limit`. The only behavioral difference in the whole set is a `logger.warning` on `po` when `n / limit < 0.05`, whose text is `"Did you mean share of?"`.

Three reasons to collapse them onto `po`.

- `so` is an ordinary English word, so it is a poor reserved word and it is the one spelling the author does not use. `po` is not a word, which is exactly what you want from a two letter operator.
- `po` is **already** the placement word inside `corridor <share> po <width> xs <attach>` (`decl.lark:430`, rendered by `decl_writer.py:506`). Keeping `po` makes the language say "part of" in one voice; keeping `so` would leave `corridor` speaking a different dialect from the layer it decorates.
- The `%` versus bare distinction that once justified separate spellings is gone. Every rule handles both.

Retiring `of` also removes a real grammar ambiguity. `occurrence net of 3000 of 6000 xs 4000` makes the Earley parser consider two readings of the word `of` in one clause. Deleting the `expr OF expr XS expr` alternative removes that fork outright.

## Finding: nothing records how a placement was written

The author asked how `decl_writer` knows whether a layer was written as a percentage or as an amount. It does not, and there is nothing to know: the distinction is resolved and thrown away at parse time.

- `_PercentNumber` (`parser.py:347-357`) is a `float` subclass whose only job is to carry the `%` marker from the `NUMBER` terminal into the `reins_clause_*` rule. Arithmetic on it degrades to plain `float`, so the marker survives literal use only.
- The rule consumes the marker and returns a plain tuple. `reins_clause_share` returns `(float(n), limit, attach)` when the number was a percentage and `(n / limit, limit, attach)` when it was bare. The marker dies there.
- `_split_reins` (`parser.py:1258-1281`) stores that list under the spec key `occ_reins` / `agg_reins` as `(share, limit, attach)` triples, `share` a plain fraction. Nothing else in the spec records the surface form.
- `_render_reins_clause` (`decl_writer.py:531-536`) therefore applies one rule with no history to consult: `share == 1.0` renders `limit xs attach` and drops the placement clause entirely, anything else renders `{share * 100}% so {limit} xs {attach}`.

Two consequences that shape this plan.

1. The `%` output form is the **only** form the library can produce. The bare amount reading (`5 po 15 xs 5`, "five part of fifteen") is input sugar that no round-trip will ever regenerate. Retiring `so` and `of` therefore costs the round-trip nothing; it narrows the input surface only.
2. Because the writer is the single source of the canonical spelling, one string literal in `decl_writer.py:535` decides what `pprogram` shows **and** what a P&L walk step is called, since `_pnl_builders._layer_descriptor` (`_pnl_builders.py:306-314`) names undeclared cover steps by calling that same renderer. That is why a two character keyword change moves roughly 46 test assertions.

## What is retired and what stays

Retired.

- The `SHARE_OF` terminal (`decl.lark:736`) and the `expr SHARE_OF expr XS expr` alternative (`:371`).
- The `expr OF expr XS expr` alternative (`decl.lark:373`) and the `reins_clause_of` rule.
- `so` leaves the `ID` exclusion list (`decl.lark:795`), so `so` becomes a legal identifier again.

Kept.

- `po` / `PART_OF`, the sole placement keyword, unchanged in meaning: `%` is a share, bare is an amount over the limit.
- The `OF` terminal itself. It is still needed for `occurrence net of` and `aggregate net of` (`decl.lark:347, 351`), so `of` stays a reserved word. Only `so` is freed.
- The full line form `limit xs attach`, which is what a 100% placement renders as.
- `corridor <share> po <width> xs <attach>`, untouched.

## Decisions for the author

**Decision [Placement-Survivor].** `po` is the survivor and the new canonical output. Recommended and assumed by every phase below. The alternative, `of`, produces `occurrence net of 50% of 100 xs 0` on round-trip, which reads badly and keeps the ambiguity.

**Decision [Placement-Migration-Error].** Two options for what `50% so 100 xs 0` does after the change.

- *Hard removal.* `so` is an ordinary identifier, so the clause fails with the generic "unexpected token" report. Cheapest, and it is what frees the word.
- *Migration stop.* Keep a `SHARE_OF` terminal for one release with no grammar rule behind it, and special case it in `parser_errors` to say `'so' was retired at 1.0.0a249; write '50% po 100 xs 0'`. Costs about four lines, but does **not** free `so` as an identifier until it is removed later.

Recommendation: hard removal. Freeing the word is the point, the corpora in this repo are reformatted as part of the change, and a stale `.agg` file gets a parse error pointing at the right column either way. `of` gets no migration stop under either option, since it has only two live uses.

## Phases

### Phase 1 [Placement-Grammar]

- `decl.lark`: delete the `SHARE_OF` terminal, delete the `SHARE_OF` and `OF` alternatives of `reins_layer`, drop `so` from the `ID` negative lookahead alternation. Update the `reins_layer` comment block to describe one placement form.
- `parser.py`: delete `reins_clause_share` and `reins_clause_of`. Rewrite the `_PercentNumber` docstring (`:347-357`), which names `so` / `po` as the pair.
- `parser.py:1461-1465`: rewrite the small share warning. `"Did you mean share of?"` becomes meaningless once "share of" is not a spelling; it should point at the percentage form, along the lines of `Part of clause with proportion {p}. A bare number is an amount; write '{pct}% po {limit} xs {attach}' for a share.`
- `parser_errors.py:154`: drop the `SHARE_OF` label entry.

Check: `test_grammar_sync.py::test_terminal_labels_cover_every_keyword_terminal` is parametrized off the grammar, so it stops asking for a `SHARE_OF` label automatically.

### Phase 2 [Placement-Canonical]

- `decl_writer.py:535`: emit `% po`. Update the `_render_reins_clause` docstring (`:513-519`), which currently names `so` as canonical and lists `so` / `po` as the input pair.
- `_pnl_builders.py:311`: the `_layer_descriptor` docstring quotes `share% so limit xs attach`.
- Confirm nothing else renders a placement. `_reinsurance.py:1224-1244` (`reins_description`) emits English prose, `50% share of 100 xs 50`, not DecL. **Leave it as prose.** It is a sentence, not a program, and "share of" is the correct English for what it describes.

### Phase 3 [Placement-Mirrors]

- `agg.sublime-syntax` lines 116 and 158: drop `so` from both alternations. This is a **hard gate**: `test_grammar_sync.py:335` compares the sublime word set against the grammar in both directions, so a leftover `so` fails the suite as `stale`.
- `decl_pygments.py:275`: drop `so` from the keyword list. Only the grammar to colorer direction is tested, so a leftover is silent, but it would color a now legal identifier as a keyword.
- Out of repo, for the author: the web app's `decl-keywords.json` lives in a separate repository and is deliberately not checked here (`tests/test_grammar_sync.py:27`). It needs the same edit.

### Phase 4 [Placement-Corpora]

Reformat every shipped program. All mechanical, `so` to `po`, since the reading of the number does not change.

- `src/aggregate/agg/library.agg`: 24 clause lines, plus two prose passages that teach the retired words. The header comment at `:61` explains canonicalization using `50% so L xs A`, and the Discussion paragraph at `:878` says outright that placements "can be written `so` (*share of*) or `po` (*part of*)". Both need rewriting to describe one keyword.
- `src/aggregate/agg/_test_suite.agg`: 12 clause lines, plus the section comment at `:196` ("so/po disambiguation").
- `src/aggregate/agg/decl-testers.agg`: 12 `so` clause lines, plus the two `of` cases at `:504-505` (`HY3.OfPct`, `HY3.OfAmt`) and their `note{}` bodies, which exist only to pin the `of` spelling. Convert both to `po` and keep them, since what they really pin is percentage versus bare amount, which is still worth a test; rename them accordingly.
- `src/aggregate/agg/_test_suite2.agg` is clean.
- Add two new `decl-testers.agg` cases for the freed word: one aggregate **named** `so` that builds, and one commented reference line recording that `50% so 100 xs 0` is no longer a placement. Both go in the matching section so they round-trip, per the standing rule about keeping `decl-testers.agg` in sync.

### Phase 5 [Placement-Tests]

Roughly 46 clause lines across 17 modules, nearly all of them assertions on rendered labels rather than on behavior. Heaviest first: `test_reinstatement_decl.py` (14, index labels like `'occ 95% so 100 xs 100'` and `'agg 85% so 1500 xs 7000'`), `test_pnl_ceded_premium.py` (5), `test_decl_unparser.py` (4, including the spread layout assertion at `:275-281`), `test_reins_reporting.py` (4), `test_reins_bivariate.py` (4). Then `test_pnl.py`, `test_pnl_peel.py`, `test_massive_bivariate.py` (2 each), and one line each in `test_bivariate.py`, `test_reins_buckets.py`, `test_variable_rating_decl.py`, `test_trailer_attachment.py`, `test_underwriter.py`, `test_discrete_severity.py`, `test_pgf_polynomial.py`, `tests/capture_distortion_snapshot.py`. `test_hygiene3.py:86` is the `of` case and moves to `po`.

Add one negative test: `50% so 100 xs 0` raises a parse error, and `agg so 10 claims dsev [1] fixed` builds.

### Phase 6 [Placement-Regen]

Four derived artifacts, script runs rather than hand edits. Read the diffs deliberately; a line that moves for a reason other than the keyword is a finding.

- `tests/data/expected_specs.json` is keyed by **program text**, so 12 keys re-key. Run `tests/capture_spec_snapshot.py`. The spec values themselves must be unchanged, which is the real check on this phase.
- `tests/data/bucket_baseline_summary.csv` also stores program text in its last column, 12 rows. **NOT DONE, and not doable here.** `scripts/bucket_baseline.py` raises `AttributeError: 'Underwriter' object has no attribute 'knowledge'` on a clean tree at `HEAD`, so it is broken independent of this change. The CSV was last written at a49 and the script last touched at a71, so the artifact is already ~200 versions stale and no test reads it. Hand-editing a captured baseline would make it claim to be a capture it is not, so it is left alone and tracked as `[Bucket-Baseline-Script-Rot]` in `dev/TODO.md`.
- `docs/4_agg_language_reference/ref_include.rst` is generated from `decl.lark`. Run `grammar(add_to_doc=True)`.
- `docs/cookbook/_recipes_05_reinsurance.qmd` (4 lines) is generated from `library.agg` by `cookbook.py`, so Phase 4 fixes it.

### Phase 7 [Placement-Docs]

- `docs/2_aggregate_overview/features.rst:52`: the a17 ledger row reads "``so``/``po`` by number". It is a historical record of what shipped at a17, so it stays factually true and was left alone. **No a249 row was added:** the ledger already stops at a195, so it lags by 53 versions and is maintained by the separate `dev/task-features.md` task, which is where this belongs. Gate is `dev/check_features_rst.py`.
- `docs/2_aggregate_overview/features.rst:1800`: quotes `agg 85% so 1500 xs 7000` as a **live** example of a walk step label. This one must change.
- `dev/FEATURES.csv:259`: the `program` cell uses `50% so -> 5 so` as its canonicalization example. Regenerate via `dev/regen_features.py` after editing the source text.
- `dev/TODO.md:638`: the `[Walk-Step-Default-Labels]` done entry quotes `'agg 85% so 1500 xs 7000'`.
- `dev/TODO.md:797`: a **pending** item proposes `net of 50% of 500 xs 500 at .3 rol` as future syntax. It uses the retired `of`; rewrite the example so the proposal does not arrive stillborn.
- `CHANGELOG.md` (6 lines) and `dev/done/plan-*.md` are history. Leave them.

**Monograph** (`C:/s/AI/aggregate-monograph`, the source of truth for the user guides; never edit the RST). Seven code sites and three teaching passages.

- `posts/3_user_guides/3_x_10mins.qmd:601`, a bullet defining `so` as "share of".
- `posts/3_user_guides/3_x_pnls.qmd:276, 278`.
- `posts/3_user_guides/3_x_re_pricing.qmd:51, 63, 179-181, 185, 193, 1220`. Line 181 teaches `0.5 so 2 xs 2` explicitly.
- `posts/3_user_guides/problems/_0x0_loss_data_analytics.qmd:248, 257`.
- `posts/3_user_guides/DecL/_080_reinsurance.qmd:13-14, 65, 75, 80`. Note bullets 13 and 14 are **already wrong** today: the bullet introducing `po` illustrates with `0.5 so 3 xs 2` and the bullet introducing `so` illustrates with `1 po 3 xs 2`. Worth fixing regardless of this plan.
- `posts/3_user_guides/problems/_0x0_enterprise_risk_analysis.qmd:31` is English prose, "95% share of 24M xs 1M". Leave it.

**Two monograph bugs surfaced by the rewrite, both pre-existing, both fixed here, both changing executed output when the author next rebuilds.** Flag them for review, since altering a page's arithmetic is the author's call and either can be reverted to the value-preserving spelling.

1. `3_x_re_pricing.qmd:181, 185, 193` taught `0.5 so 2 xs 2` as "50% share of 2 xs 2" and claimed it cedes 0.5 of a 3. It does not: a bare `0.5` against a limit of `2` is a **25%** placement, and has been since the a17 percent-marker change. The page contradicted itself. Rewritten to `50% po 2 xs 2`, which is the placement the prose describes and makes the stated numbers correct. The value-preserving alternative was `0.5 po 2 xs 2`, which would have kept the wrong prose.
2. `problems/_0x0_loss_models.qmd:419` wrote the KPW 9.14 coinsurance as `occurrence net of 0.25 so inf xs 0`. A bare amount over an `inf` limit is `0.25 / inf`, so the share is **zero** and the 25% coinsurance the surrounding text describes was never applied. Now `25% po inf xs 0`. This is exactly the case the reworded warning in Phase 1 catches, which is how it was found.

### Phase 8 [Placement-Close]

- Bump `pyproject.toml` to `1.0.0a249`.
- `CHANGELOG.md`: a `## 1.0.0a249` section. Breaking change, and it should say plainly that `so` and `of` are gone, that `po` is the sole placement keyword and the canonical output, that `so` is a legal identifier again, and that P&L step labels for undeclared partial placement layers now read `po`.
- `dev/TODO.md`: mark the item done.
- Move this plan to `dev/done/`.
- One commit, one line subject: `[Single-Placement-Keyword] a249: one way to write a partial placement, and it is po`.

## Verification

- Edit loop: `pytest -n0 --dist no --testmon-forceselect`.
- Gate before declaring done: `uv run pytest`.
- Version bump: `uv run pytest -m 'slow or not slow'`. Non negotiable here, because three of the touched test modules (`test_bivariate.py`, `test_massive_bivariate.py`, `test_reins_bivariate.py`) are `slow` tagged and invisible to the everyday run.
- The load bearing check is `tests/data/expected_specs.json`: the keys move, the **values** must not. If a spec value changes, the transformer edit was not neutral.

## Risks, accepted by the author

- Dropping `so` from the `ID` exclusion list widens what counts as an identifier. The grammar's `NUMBER ID` adjacencies are after `sev` (distribution names) and in `reinst_group` count words, neither of which sits in `reins_layer` position, so no new ambiguity is expected. Phase 5's positive test (`agg so ...` builds) is the check.
- Any `.agg` file outside this repo that uses `so` or `of` stops parsing. Accepted under Decision [Placement-Migration-Error] as written.
- P&L row labels are output, and downstream notebooks that index an `xpnl` frame by a literal `'agg 85% so 1500 xs 7000'` will `KeyError`. Called out in the CHANGELOG.

## Out of scope

- The `po` bare amount reading stays. `5 po 15 xs 5` is genuine market language and the warning at `parser.py:1461` is the right guard for it.
- `reins_description`'s English "share of" phrasing stays.
- No change to how a 100% placement renders. It still drops the clause and reads `limit xs attach`.
