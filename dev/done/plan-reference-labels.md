# Plan: a DecL reference stops overriding the referring statement's label and tags

`[Reference-Label-Leak]`. Drafted 2026-11-06 against aggregate 2.0.0 (commit `1fe40eb8`). **Status: ruled 2026-11-06; executed 2026-11-06 as 2.1.0a1 (see the execution log).** One pre-release bump (`2.1.0aN`, house rule 2026-11-06). Transformer change only, no grammar change.

## Goal

A label or tag set belongs to the statement it is written on. `agg B as "b label" agg.A` has label `b label`, and a statement that refers to another object does not take that object's `label` or `tags`.

## Current behavior, 2.0.0

Reproduced 2026-11-06 with `build.build_many` on:

```
sev LblSevS as "s label" lognorm 10 cv 1 note{sev note} tags{sevtag};
agg LblX 1 claim sev sev.LblSevS fixed;
agg LblY as "y label" 1 claim sev sev.LblSevS fixed;
agg LblA as "a label" 1 claim dsev [1:4] fixed note{a note} tags{atag};
agg LblB as "b label" agg.LblA;
agg LblC agg.LblA;
agg LblD as "d label" 2 @ agg.LblA;
agg LblE agg.LblA tags{etag}
```

| statement | label | tags | expected label | expected tags |
|---|---|---|---|---|
| `LblX` | `s label` | `sevtag` | `LblX` | none |
| `LblY` | `s label` | `sevtag` | `y label` | none |
| `LblB` | `a label` | `atag` | `b label` | none |
| `LblC` | `a label` | `atag` | `LblC` (ruled) | none |
| `LblD` | `a label` | `atag` | `d label` | none |
| `LblE` | `a label` | `etag` | `LblE` | `etag` |

`note` and `hints` do not leak: the trailer seeds both with `''` and is merged after the body.

**Cause.** `UnderwritingTransformer.agg_out_named` (`parser.py:1572`) returns `{"name": name, **as_label, **body, **trailer}`. For the rename form the body comes from `agg_body_rename` (`parser.py:1430`), which is the referenced entry's stored spec with only `"name"` deleted, so its `"label"` and `"tags"` are merged after `as_label` and win. `sev_clause_builtin` (`parser.py:2738`) and `sev_builtin` (`parser.py:2812`) also delete only `"name"`, so a stored severity's `"label"` and `"tags"` enter the aggregate's flat spec the same way.

**Not affected.** `agg.A` with its own reinsurance (`agg_out_builtin`, `parser.py:1576`) is the object `A` itself and keeps its label, correctly. The engine reference of a `pnl` (`agg_source_ref_agg`, `parser.py:1643`) is `A` and keeps its label, correctly. Portfolio units written as `agg.A` keep theirs, correctly.

## The change

- **[Reference-Strips-Presentation]** One helper on the transformer, `_strip_presentation(spec)`, pops `name`, `label`, `note`, `tags` and `hints` from a looked-up spec. `agg_body_rename`, `sev_clause_builtin` and `sev_builtin` call it in place of their `del b["name"]`. Popping `note` and `hints` changes nothing today, since the seeded trailer overwrites them, and keeps the rule in one place: a reference contributes its body, and the statement contributes its presentation.
- **[Sev-Reference-Label]** In `sev_clause_builtin`, the inline severity's interior label (`_severity_label`, gathered into `label_map['severity']` by `_pop_interior_labels`) is the clause's `as` label when written, otherwise the stored severity's label (ruling).
- Docstrings on the three methods and the helper state the rule.

## Rulings

- **[Rename-Label-Default]** Ruled by the author, 2026-11-06: `agg C agg.A` has label `C`, through the ordinary fallback to the name, and does not inherit `A`'s label.
- **Tags** follow the same rule as the label and are not inherited. This extends the author's ruling on the label to the other presentation key that leaks; the author may overrule it.
- **[Sev-Reference-Label]** Ruled by the author, 2026-11-06: `sev sev.S` with no `as` labels the severity inside the aggregate with `S`'s label, since that severity is `S`, unlike a renamed aggregate.

## Blast radius

- `library.agg` has eight renames and one severity reference, none with an `as` clause, at lines 197, 250, 260, 1124, 1133, 1140, 1378, 1383 and 1388 (`Capstone.CappedQS`, `Capstone.GrossNet`, `SwingRated`, `RetroRated`, `SlidingScale`, `BivariateGrossNet`, `BivariateGrossCeded`, `BivariateNetCeded`, and the aggregate using `sev.Capstone.Sev`). Their labels move from the referenced entry's to their own name, and their inherited tags go. The execution log lists each, and any whose tags were relied on for a recipe filter (`Underwriter` tag queries) is a finding to raise, not to absorb.
- `tests/data/expected_specs.json` moves for any `decl-testers.agg` program of these forms; the diff is read entry by entry.

## Tests

In `tests/test_decl_labels.py`:

1. The six statements of the table above give the expected label and tags.
2. `agg LblZ 1 claim sev sev.LblSevS as "zsev" fixed` gives aggregate label `LblZ` and `label_map['severity'] == 'zsev'`; without `as "zsev"`, `label_map['severity'] == 's label'` (per ruling).
3. `agg.LblA occurrence net of 1 xs 1` keeps label `a label` and tags `atag`.
4. `port P agg.LblA agg.LblB` units keep `a label` and `b label`.
5. The rename and severity-reference forms, with and without `as`, appended to `decl-testers.agg`; the snapshot recaptured and its diff read.

## Files

- `src/aggregate/parser.py`: the helper, the three methods.
- `tests/test_decl_labels.py`; `src/aggregate/agg/decl-testers.agg`; `tests/data/expected_specs.json` (recaptured).
- No Using page: this fixes documented behavior and adds nothing a user calls.

## Acceptance check

- The tests pass; tier 2 passes; tier 3 passes at the bump.
- `build('agg B as "b label" agg.A')` (with `A` as above) has `label == 'b label'`.
- The snapshot diff contains only the label, tags and severity label moves described here.

## Execution log

Executed 2026-11-06 as one bump, 2.1.0a1, from `1fe40eb8` (2.0.0). The table above was reproduced exactly before any edit, and `sev S2 sev.S` (a severity declared as a reference to another) leaked the same way through `sev_builtin`; the helper fixes it, and `S2` is labeled `S2` by the rename rule.

### Divergences

- **[Sev-Reference-Label] lands on two routes, not one.** The plan places it in `sev_clause_builtin`, but that method handles only the bare form `agg A 1 claim sev.S fixed`. The plan's own test form, `sev sev.S`, parses as `SEV sev as_label`, so the leaf is `sev_builtin` and the clause is `sev_clause_sev` (or `sev_clause_ssev`). `sev_builtin` therefore passes the stored label on as the existing `_severity_label` temp key, and `sev_clause_sev` and `sev_clause_ssev` let a written `as` override it. The other consumers of the `sev` rule drop the key: `sev_out_sev`, `_sev_to_wait` (its drop set), and the tweedie and stable compound forms (`_SEV_METADATA_KEYS`). A `sev.S` leaf can carry only `!` and picks, never scaling or a shift (`sev0` does not derive `BUILTIN_SEV`), so the severity it labels is always `S`.
- **"The stored severity's label" read as the explicit label only.** When `S` has no `as` clause, the severity site stays unlabeled (`labels.severity is None`) rather than taking `S`'s name, so references to unlabeled severities, which is every one in `library.agg`, are unchanged. Taking the name as well is a one-line change if wanted.
- **Plan test 3 rewritten.** `agg.LblA occurrence net of 1 xs 1` does not parse: `agg_out_builtin` is `builtin_agg agg_reins trailer`, aggregate reinsurance only. The test uses `agg.LblA aggregate net of 1 xs 1`.
- **Popping `note` changes the parsed spec of a pnl engine rename.** The plan's claim that popping `note` and `hints` changes nothing holds for a top level statement, whose trailer is merged last, but `pnl ... less agg E agg.A` builds its engine through `agg_source_inline`, which has no trailer. There `A`'s note used to become `engine_note`, and `A`'s label `engine_label`. `engine_note` is discarded before build (`underwriter.py`, the pnl branch), so behavior is unchanged; `engine_label` names the P&L loss leg, so a renamed engine's leg is now its own `as` label or the default, following [Rename-Label-Default]. Hints were already skipped by `_pnl_spec`.
- **No `dev/TODO.md` tick and no control list row.** The plan has no TODO entry, and `V:/worktrees/dev-files.md`, still listed in the project `execute-plan` skill, no longer exists.

### Blast radius as measured

- **`library.agg`: no label or tag moves.** Every one of the 212 library entries was parsed before and after, and `label`, `tags`, `label_map`, `engine_label` and `engine_note` compared. The renames the plan lists, and four more it missed because the reference sits on the line after the name (`Capstone.ExposureRating`, `Capstone.LossPicksTest`, `Capstone.XOL`, `Capstone.FullProgram`), all carry their own `tags{}` trailer and reference entries with no `as` label, so nothing they show moves. `Capstone.Gross` (line 182, the bare `sev.Capstone.Sev` clause) is a second severity reference the plan's count missed; it is unchanged for the same reason. No `Underwriter` tag query is affected.
- **Four library pnl entries lose `engine_note`:** `Capstone.Capped`, `SwingRatedAggCover`, `RetroRatedAccount` and `SlidingScaleCommission`, whose engines rename `Capstone.XOL` or `VariableFeature.Base`. The key is discarded before build, so nothing built changes.
- **Snapshot.** Eight new entries (section LB of `decl-testers.agg`), each reading as intended. One existing entry moved in key order only, `agg Cb.Freq12.IG` (`10 claims sev.One ...`), whose `note` and `hints` now follow the frequency keys instead of holding the slot the stored severity gave them; the values are unchanged and the snapshot test compares key sets.
- **Docs.** No page has a rename inheriting a label or tags, so no `.rst` changes.

### Tests

- `tests/test_decl_labels.py`: the plan's six statements plus the bare `sev.S` clause and `sev S2 sev.S` (label and tags); the referenced entries keep their own; [Sev-Reference-Label] with and without `as`, through `sev`, `ssev` and the bare clause; an unlabeled stored severity leaves the site unlabeled; `agg.A aggregate net of` keeps `A`'s label and tags; portfolio units by reference keep theirs; a pnl engine rename takes no presentation, with and without `as`, against `less agg.A`, which keeps `A`'s; a wait clause on `sev.S` drops the stored label; the tweedie compound form accepts a labeled `sev.G`.
- Tier 1 (`test_decl_parser.py`, `test_decl_labels.py`, `test_library_entries.py`, `test_underwriter.py`): 1,593 passed. Tier 2: 9,760 passed, 3 skipped (pre-existing: plotly absent, two combinator distortions that cannot round-trip).
- Tier 3 under the default `-n auto` failed three times, each on a different test unrelated to the change: first `test_decl_words.py::test_module_entry_point_prints_the_list_as_json`, because `__version__` was bumped while the run was in flight (the worker held 2.0.0, its subprocess read 2.1.0a1); then two runs of allocation failures (`_ArrayMemoryError`, 16 MiB and then a 4 GiB bivariate grid) in `test_ruin.py`, `test_renewal_agg.py` and `test_bivariate.py`, each passing alone. The gate of record is `uv run pytest -m 'slow or not slow' -n 4`: 11,553 passed, 3 skipped. The memory flake under `-n auto` is a finding for the author, not part of this plan.
